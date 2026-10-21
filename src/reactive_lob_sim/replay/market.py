"""Replay of an I1 tape (contract 6.2) with an injected experimental agent and queue-position tracking.

The recorded book and trades are replayed unchanged, so the market never reacts to the agent (this is the
defining limitation of replay). The agent's orders interact with the recorded liquidity as follows.

Marketable orders walk the visible L2 book. The quantity they take is hidden from the agent until the next
book update at that level, so the agent cannot consume the same liquidity twice.
A resting limit order joins the back of the queue: ``queue_ahead`` starts at the visible size at its price.
A recorded trade at the order's price first consumes ``queue_ahead`` and then fills the order; a trade
through a worse price (or the opposite touch moving through the order's price) fills it directly.
Size decreases not explained by recorded trades are cancellations. Under ``risk_averse`` they never advance
the queue (they are assumed to come from behind); under ``proportional`` they advance it by the share of the
level that is ahead of the order (as in the queue models of hftbacktest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reactive_lob_sim.engine.accounting import FeeSchedule
from reactive_lob_sim.engine.types import (
    Liquidity,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
    TimeInForce,
)
from reactive_lob_sim.tape.l2 import (
    KIND_DELTA,
    KIND_FUNDING,
    KIND_OI,
    KIND_SNAPSHOT,
    KIND_TRADE,
    L2Book,
    merge_events,
)
from reactive_lob_sim.tape.tape import Tape
from reactive_lob_sim.timeutil import NS_PER_S
from reactive_lob_sim.venue import AgentFill, MarketView, OrderAck

QueueModel = Literal["risk_averse", "proportional"]
_BPS = 10_000.0


@dataclass(slots=True)
class _Resting:
    order_id: int
    side: Side
    tick: int
    remaining: float
    queue_ahead: float
    ts_submit: float
    arrival_mid: float
    distance_bps: float
    queue_at_entry: float


class ReplayMarket:
    def __init__(
        self,
        tape: Tape,
        tick_size: float,
        lot_size: float,
        qty_tolerance: float,
        fees: FeeSchedule,
        queue_model: QueueModel,
        start_ns: int | None = None,
    ) -> None:
        self._tick = tick_size
        self._lot = lot_size
        self._eps = qty_tolerance
        self._fees = fees
        self._queue_model = queue_model
        ev = merge_events(tape, tick_size)
        if len(ev) == 0:
            raise ValueError("cannot replay an empty tape")
        self._start_ns = int(ev.ts[0]) if start_ns is None else start_ns
        self._ts = ev.ts.tolist()
        self._kind = ev.kind.tolist()
        self._bid = ev.is_bid.tolist()
        self._tk = ev.tick.tolist()
        self._size = ev.size.tolist()
        self._value = ev.value.tolist()
        self._group = ev.group.tolist()
        self._i = 0
        self._t = 0.0
        self.book = L2Book()
        self._snap_group = -1
        self._consumed: dict[tuple[bool, int], float] = {}
        self._pending_trade: dict[tuple[bool, int], float] = {}
        self._orders: dict[int, _Resting] = {}
        self._next_id = 1
        self._fills: list[AgentFill] = []
        self._last_trade: int | None = None
        self.funding_rate = 0.0
        self.open_interest = 0.0

    # ------------------------------------------------------------------ MarketBackend
    @property
    def tick_size(self) -> float:
        return self._tick

    @property
    def lot_size(self) -> float:
        return self._lot

    @property
    def t(self) -> float:
        return self._t

    def advance(self, t: float) -> None:
        limit_ns = self._start_ns + round(t * NS_PER_S)
        n = len(self._ts)
        while self._i < n and self._ts[self._i] <= limit_ns:
            i = self._i
            self._i += 1
            self._t = max(self._t, (self._ts[i] - self._start_ns) / NS_PER_S)
            kind = self._kind[i]
            if kind == KIND_SNAPSHOT:
                if self._group[i] != self._snap_group:
                    self.book.clear()
                    self._consumed.clear()
                    self._snap_group = self._group[i]
                self.book.set(self._bid[i], self._tk[i], self._size[i])
            elif kind == KIND_DELTA:
                self._on_delta(self._bid[i], self._tk[i], self._size[i])
            elif kind == KIND_TRADE:
                self._on_trade(self._bid[i], self._tk[i], self._size[i])
            elif kind == KIND_FUNDING:
                self.funding_rate = self._value[i]
            elif kind == KIND_OI:
                self.open_interest = self._value[i]
        self._t = max(self._t, t)

    def _eff_levels(self, is_bid: bool) -> list[tuple[int, float]]:
        out = []
        for tick, qty in self.book.levels(is_bid):
            eff = qty - self._consumed.get((is_bid, tick), 0.0)
            if eff > self._eps:
                out.append((tick, eff))
        return out

    def _eff_best(self, is_bid: bool) -> tuple[int, float] | None:
        for tick, qty in self.book.levels(is_bid):
            eff = qty - self._consumed.get((is_bid, tick), 0.0)
            if eff > self._eps:
                return tick, eff
        return None

    def mid_price(self) -> float | None:
        bid, ask = self._eff_best(True), self._eff_best(False)
        if bid is None or ask is None:
            return None
        return (bid[0] + ask[0]) / 2.0 * self._tick

    def view(self) -> MarketView:
        bid, ask = self._eff_best(True), self._eff_best(False)
        return MarketView(
            t=self._t,
            best_bid=None if bid is None else bid[0] * self._tick,
            best_ask=None if ask is None else ask[0] * self._tick,
            bid_size=0.0 if bid is None else bid[1],
            ask_size=0.0 if ask is None else ask[1],
            last_trade=None if self._last_trade is None else self._last_trade * self._tick,
            funding_rate=self.funding_rate,
            open_interest=self.open_interest,
            tick_size=self._tick,
        )

    def submit(
        self,
        side: Side,
        order_type: OrderType,
        qty: float,
        price: float | None = None,
        tif: TimeInForce = TimeInForce.GTC,
    ) -> OrderAck:
        order_id = self._next_id
        self._next_id += 1
        qty = max(round(qty / self._lot), 1) * self._lot
        tick = None if price is None else round(price / self._tick)
        if order_type is OrderType.LIMIT and tick is None:
            return OrderAck(order_id, OrderStatus.REJECTED, RejectReason.INVALID_PRICE, 0.0)
        opp_bid = side is Side.SELL
        opp_levels = self._eff_levels(opp_bid)
        limit = tick if order_type is OrderType.LIMIT else None

        def reachable(level_tick: int) -> bool:
            if limit is None:
                return True
            return level_tick <= limit if side is Side.BUY else level_tick >= limit

        crossing = [(t, q) for t, q in opp_levels if reachable(t)]
        if order_type is OrderType.MARKET and not opp_levels:
            return OrderAck(order_id, OrderStatus.REJECTED, RejectReason.NO_LIQUIDITY, 0.0)
        if tif is TimeInForce.POST_ONLY and crossing:
            return OrderAck(order_id, OrderStatus.REJECTED, RejectReason.POST_ONLY_WOULD_CROSS, 0.0)
        if tif is TimeInForce.FOK and sum(q for _, q in crossing) < qty - self._eps:
            return OrderAck(order_id, OrderStatus.REJECTED, RejectReason.FOK_INSUFFICIENT_LIQUIDITY, 0.0)
        arrival = self.mid_price() or (price or 0.0)
        touch = self._eff_best(side is Side.BUY)
        distance = 0.0
        if limit is not None and touch is not None and arrival > 0:
            distance = side.sign * (touch[0] - limit) * self._tick / arrival * _BPS
        remaining = qty
        for level_tick, avail in crossing:
            if remaining <= self._eps:
                break
            take = min(avail, remaining)
            key = (opp_bid, level_tick)
            self._consumed[key] = self._consumed.get(key, 0.0) + take
            remaining -= take
            self._fill(
                order_id, side, order_type, self._t, arrival, distance, 0.0, level_tick, take, Liquidity.TAKER
            )
        filled = qty - remaining
        rests = (
            limit is not None and tif in (TimeInForce.GTC, TimeInForce.POST_ONLY) and remaining > self._eps
        )
        if rests:
            assert limit is not None
            visible = self.book.size(side is Side.BUY, limit) - self._consumed.get(
                (side is Side.BUY, limit), 0.0
            )
            ahead = max(visible, 0.0)
            self._orders[order_id] = _Resting(
                order_id, side, limit, remaining, ahead, self._t, arrival, distance, ahead
            )
        if remaining <= self._eps:
            status = OrderStatus.FILLED
        elif filled > self._eps:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = OrderStatus.ACCEPTED if rests else OrderStatus.CANCELLED
        return OrderAck(order_id, status, None, filled)

    def cancel(self, order_id: int) -> bool:
        return self._orders.pop(order_id, None) is not None

    def open_orders(self) -> list[int]:
        return list(self._orders)

    def drain_fills(self) -> list[AgentFill]:
        out, self._fills = self._fills, []
        return out

    def tape(self) -> Tape | None:
        return None

    # ------------------------------------------------------------------ queue tracking
    def _fill(
        self,
        order_id: int,
        side: Side,
        order_type: OrderType,
        ts_submit: float,
        arrival: float,
        distance: float,
        queue_at_entry: float,
        tick: int,
        qty: float,
        liquidity: Liquidity,
    ) -> None:
        price = tick * self._tick
        self._fills.append(
            AgentFill(
                order_id=order_id,
                side=side,
                order_type=order_type,
                ts_submit_s=ts_submit,
                ts_fill_s=self._t,
                price=price,
                qty=qty,
                fee=self._fees.fee(price, qty, liquidity),
                liquidity=liquidity,
                arrival_mid=arrival,
                distance_to_touch_bps=distance,
                queue_ahead_at_entry=queue_at_entry,
            )
        )

    def _fill_resting(self, order: _Resting, qty: float) -> None:
        qty = min(qty, order.remaining)
        if qty <= self._eps:
            return
        order.remaining -= qty
        self._fill(
            order.order_id,
            order.side,
            OrderType.LIMIT,
            order.ts_submit,
            order.arrival_mid,
            order.distance_bps,
            order.queue_at_entry,
            order.tick,
            qty,
            Liquidity.MAKER,
        )
        if order.remaining <= self._eps:
            del self._orders[order.order_id]

    def _on_trade(self, aggressor_buy: bool, tick: int, size: float) -> None:
        self._last_trade = tick
        passive_bid = not aggressor_buy
        key = (passive_bid, tick)
        self._pending_trade[key] = self._pending_trade.get(key, 0.0) + size
        passive_side = Side.BUY if passive_bid else Side.SELL
        left = size
        for order in sorted(
            (o for o in self._orders.values() if o.side is passive_side),
            key=lambda o: -o.tick if passive_bid else o.tick,
        ):
            if left <= self._eps:
                break
            better = order.tick > tick if passive_bid else order.tick < tick
            if better:
                take = min(left, order.remaining)
                self._fill_resting(order, take)
                left -= take
            elif order.tick == tick:
                order.queue_ahead -= left
                if order.queue_ahead < 0:
                    take = min(-order.queue_ahead, order.remaining)
                    order.queue_ahead = 0.0
                    self._fill_resting(order, take)
                    left = max(left - take, 0.0)

    def _on_delta(self, is_bid: bool, tick: int, size: float) -> None:
        old = self.book.set(is_bid, tick, size)
        self._consumed.pop((is_bid, tick), None)
        if size < old:
            decrease = old - size
            key = (is_bid, tick)
            explained = min(decrease, self._pending_trade.get(key, 0.0))
            if explained:
                self._pending_trade[key] -= explained
            cancelled = decrease - explained
            side = Side.BUY if is_bid else Side.SELL
            for order in list(self._orders.values()):
                if order.side is not side or order.tick != tick:
                    continue
                if self._queue_model == "proportional" and cancelled > 0 and old > 0:
                    order.queue_ahead -= cancelled * order.queue_ahead / old
                order.queue_ahead = max(min(order.queue_ahead, size), 0.0)
        self._cross_check()

    def _cross_check(self) -> None:
        """Fill resting agent orders that the opposite touch has moved through."""
        best_ask = self.book.best(False)
        best_bid = self.book.best(True)
        for order in list(self._orders.values()):
            if (order.side is Side.BUY and best_ask is not None and best_ask <= order.tick) or (
                order.side is Side.SELL and best_bid is not None and best_bid >= order.tick
            ):
                self._fill_resting(order, order.remaining)
