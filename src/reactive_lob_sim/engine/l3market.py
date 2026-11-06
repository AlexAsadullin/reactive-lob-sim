"""Shared machinery of the simulators that run on the L3 matching engine (ABM, generative, ground truth).

It owns the order book, records the I1 tape, keeps the impact state, enforces the exchange rules (whole lots,
price tick, daily price band, trading phase) and books the experimental agent's orders and fills. Subclasses
supply the background order flow by implementing :meth:`_next_event_time` and :meth:`_process_next_event`.

Trading phases (:class:`~reactive_lob_sim.moex.sessions.Phase`): in ``continuous`` orders match; in an
auction they are collected (a market order becomes a limit at its band edge, the MOEX treatment of market
orders in an auction) and :meth:`L3Market.set_phase` uncrosses the book when the auction ends; in ``halted``,
``break`` and ``closed`` new orders are rejected and cancels are allowed.
"""

from __future__ import annotations

import abc
import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from reactive_lob_sim.engine.accounting import FeeSchedule
from reactive_lob_sim.engine.auction import equilibrium
from reactive_lob_sim.engine.book import OrderBook
from reactive_lob_sim.engine.rules import TradingRules
from reactive_lob_sim.engine.types import (
    Fill,
    Liquidity,
    OrderRequest,
    OrderStatus,
    OrderType,
    Owner,
    RejectReason,
    Side,
    SubmitResult,
    TimeInForce,
)
from reactive_lob_sim.impact.model import ImpactState
from reactive_lob_sim.moex.sessions import SESSION_CODES, STATUS_CODES, Phase
from reactive_lob_sim.params import ImpactParams, InitialBook, LiquidityParams
from reactive_lob_sim.tape.tape import Tape, TapeRecorder
from reactive_lob_sim.venue import AgentFill, MarketView, OrderAck

_BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class _OrderMeta:
    side: Side
    order_type: OrderType
    ts_submit: float
    arrival_mid: float
    distance_bps: float
    queue_ahead: float


class IdPool:
    """Set of order ids with O(1) add, remove and uniform sampling."""

    def __init__(self) -> None:
        self._items: list[int] = []
        self._pos: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: int) -> bool:
        return item in self._pos

    def add(self, item: int) -> None:
        if item not in self._pos:
            self._pos[item] = len(self._items)
            self._items.append(item)

    def discard(self, item: int) -> None:
        pos = self._pos.pop(item, None)
        if pos is None:
            return
        last = self._items.pop()
        if pos < len(self._items):
            self._items[pos] = last
            self._pos[last] = pos

    def sample(self, rng: np.random.Generator) -> int:
        return self._items[int(rng.integers(0, len(self._items)))]

    def items(self) -> list[int]:
        return list(self._items)


class L3Market(abc.ABC):
    """Base class of simulated venues backed by :class:`OrderBook`."""

    def __init__(
        self,
        rules: TradingRules,
        qty_tolerance: float,
        fees: FeeSchedule,
        impact: ImpactParams | None,
        recorder: TapeRecorder | None,
    ) -> None:
        self.rules = rules
        self._tick = rules.tick
        self._lot = rules.lot
        self.book = OrderBook(qty_tolerance)
        self.phase = Phase.CONTINUOUS
        self.session: str | None = "main"
        self.n_uncrosses = 0
        self.auction_volume = 0.0
        self._fees = fees
        self.impact_params = impact
        self.impact = ImpactState(impact) if impact is not None else None
        self._recorder = recorder
        self._t = 0.0
        self._agent_meta: dict[int, _OrderMeta] = {}
        self._agent_fills: list[AgentFill] = []
        self._pools: dict[tuple[int, Side], IdPool] = {}
        self.last_trade_tick: int | None = None
        self.funding_rate = 0.0
        self.open_interest = 0.0
        self._mid_t: list[float] = []
        self._mid_v: list[float] = []

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
        while True:
            nxt = self._next_event_time()
            if nxt is None or nxt > t:
                break
            self._t = max(self._t, nxt)
            self._process_next_event()
        self._t = max(self._t, t)
        self._on_clock(self._t)

    def view(self) -> MarketView:
        bid, ask = self.book.best_bid(), self.book.best_ask()
        return MarketView(
            t=self._t,
            best_bid=None if bid is None else bid * self._tick,
            best_ask=None if ask is None else ask * self._tick,
            bid_size=0.0 if bid is None else self.book.level_qty(Side.BUY, bid),
            ask_size=0.0 if ask is None else self.book.level_qty(Side.SELL, ask),
            last_trade=None if self.last_trade_tick is None else self.last_trade_tick * self._tick,
            funding_rate=self.funding_rate,
            open_interest=self.open_interest,
            tick_size=self._tick,
            lot_size=self._lot,
            phase=self.phase.value,
        )

    def submit(
        self,
        side: Side,
        order_type: OrderType,
        qty: float,
        price: float | None = None,
        tif: TimeInForce = TimeInForce.GTC,
    ) -> OrderAck:
        reason = self.rules.check(qty, price)
        if reason is None and not self.phase.accepts_orders:
            reason = RejectReason.TRADING_HALTED
        if reason is not None:
            return OrderAck(0, OrderStatus.REJECTED, reason, 0.0)
        tick = None if price is None else self.rules.to_tick(price)
        mid = self.mid_price()
        arrival = mid if mid is not None else (price or 0.0)
        touch = self.book.best(side)
        if order_type is OrderType.LIMIT and tick is not None and touch is not None and arrival > 0:
            distance = side.sign * (touch - tick) * self._tick / arrival * _BPS
        else:
            distance = 0.0
        req = OrderRequest(Owner.EXPERIMENT, side, order_type, qty, tick, tif)
        result = self.execute(req)
        if result.status is not OrderStatus.REJECTED:
            meta = _OrderMeta(side, order_type, self._t, arrival, distance, result.queue_ahead)
            self._agent_meta[result.order_id] = meta
            for fill in result.fills:
                self._book_agent_fill(result.order_id, fill.price_tick, fill.qty, Liquidity.TAKER)
        return OrderAck(result.order_id, result.status, result.reject_reason, result.filled_qty)

    def cancel(self, order_id: int) -> bool:
        order = self.book.order(order_id)
        if order is None or order.owner != Owner.EXPERIMENT:
            return False
        self.pool(order.owner, order.side).discard(order_id)
        res = self.book.cancel(order_id)
        self._record_levels(res.touched)
        return res.cancelled

    def open_orders(self) -> list[int]:
        return [o.order_id for o in self.book.orders() if o.owner == Owner.EXPERIMENT]

    def drain_fills(self) -> list[AgentFill]:
        out, self._agent_fills = self._agent_fills, []
        return out

    def tape(self) -> Tape | None:
        return None if self._recorder is None else self._recorder.to_tape()

    # ------------------------------------------------------------------ helpers for subclasses
    def mid_price(self) -> float | None:
        mid = self.book.mid_tick()
        return None if mid is None else mid * self._tick

    def round_qty(self, qty: float) -> float:
        return self.rules.round_qty(qty)

    def pool(self, owner: int, side: Side) -> IdPool:
        key = (owner, side)
        if key not in self._pools:
            self._pools[key] = IdPool()
        return self._pools[key]

    def mid_at(self, t: float) -> float | None:
        """Mid price recorded at or before ``t`` (from the per-event history)."""
        i = bisect_right(self._mid_t, t) - 1
        return self._mid_v[i] if i >= 0 else None

    def note_mid(self) -> None:
        mid = self.mid_price()
        if mid is not None:
            self._mid_t.append(self._t)
            self._mid_v.append(mid)

    def execute(self, req: OrderRequest) -> SubmitResult:
        """Submit an order of any owner under the trading rules, record its tape footprint and feed the
        impact state."""
        if not self.phase.accepts_orders:
            return SubmitResult(0, OrderStatus.REJECTED, (), 0.0, RejectReason.TRADING_HALTED, 0.0, ())
        if req.order_type is OrderType.LIMIT and req.price_tick is not None:
            if not self.rules.in_band(req.price_tick):
                return SubmitResult(0, OrderStatus.REJECTED, (), 0.0, RejectReason.PRICE_OUT_OF_BAND, 0.0, ())
        elif req.order_type is OrderType.MARKET:
            cap = self.rules.protection_tick(req.side is Side.BUY)
            if self.phase.is_auction:
                if cap is None:
                    return SubmitResult(
                        0, OrderStatus.REJECTED, (), 0.0, RejectReason.NOT_ALLOWED_IN_PHASE, 0.0, ()
                    )
                req = OrderRequest(req.owner, req.side, OrderType.LIMIT, req.qty, cap, TimeInForce.GTC)
            elif cap is not None:
                req = OrderRequest(req.owner, req.side, req.order_type, req.qty, cap, req.tif)
        result = self.book.submit(req)
        self._after_submit(req, result)
        return result

    # ------------------------------------------------------------------ trading phases
    def set_phase(self, phase: Phase, session: str | None = None) -> list[Fill]:
        """Move to ``phase``. Leaving an auction uncrosses the book first; the uncross fills are returned."""
        fills: list[Fill] = []
        if phase is self.phase and session == self.session:
            return fills
        if self.phase.is_auction and not phase.is_auction:
            fills = self.uncross()
        self.phase = phase
        self.session = session
        self.book.matching = not phase.is_auction
        if self._recorder is not None:
            self._recorder.trading_status(
                self._t, STATUS_CODES[phase], phase.accepts_orders, phase is Phase.CONTINUOUS
            )
            self._recorder.trading_session = SESSION_CODES.get(session) if session else None
        return fills

    def auction_reference_tick(self) -> float | None:
        """Reference price of the uncross tie-break: the last trade, else the mid of the collected book."""
        if self.last_trade_tick is not None:
            return float(self.last_trade_tick)
        bid, ask = self.book.best_bid(), self.book.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def uncross(self) -> list[Fill]:
        """Execute the crossed part of the collected book at the equilibrium price."""
        bids, asks = self.book.levels(Side.BUY), self.book.levels(Side.SELL)
        eq = equilibrium(bids, asks, self.auction_reference_tick())
        self.book.matching = True
        if eq is None:
            return []
        before = [(Side.BUY, t) for t, _ in bids if t >= eq.price_tick]
        before += [(Side.SELL, t) for t, _ in asks if t <= eq.price_tick]
        fills = self.book.uncross(eq.price_tick)
        for fill in fills:
            if self._recorder is not None:
                self._recorder.trade(self._t, fill.price_tick, fill.qty, fill.taker_side)
            for oid, owner, side in (
                (fill.maker_order_id, fill.maker_owner, fill.taker_side.opposite),
                (fill.taker_order_id, fill.taker_owner, fill.taker_side),
            ):
                if self.book.order(oid) is None:
                    self.pool(owner, side).discard(oid)
                if owner == Owner.EXPERIMENT:
                    liquidity = Liquidity.MAKER if oid == fill.maker_order_id else Liquidity.TAKER
                    self._book_agent_fill(oid, fill.price_tick, fill.qty, liquidity)
        if fills:
            self.last_trade_tick = eq.price_tick
            self.n_uncrosses += 1
            self.auction_volume += sum(f.qty for f in fills)
        self._record_levels(tuple(before))
        self.note_mid()
        return fills

    def cancel_background(self, order_id: int) -> bool:
        order = self.book.order(order_id)
        if order is not None:
            self.pool(order.owner, order.side).discard(order_id)
        res = self.book.cancel(order_id)
        if res.cancelled:
            self._record_levels(res.touched)
        return res.cancelled

    def record_snapshot(self, depth: int) -> None:
        if self._recorder is not None:
            self._recorder.snapshot(
                self._t, self.book.levels(Side.BUY, depth), self.book.levels(Side.SELL, depth)
            )

    @property
    def recorder(self) -> TapeRecorder | None:
        return self._recorder

    # ------------------------------------------------------------------ internals
    def _after_submit(self, req: OrderRequest, result: SubmitResult) -> None:
        if (
            result.status is not OrderStatus.REJECTED
            and req.order_type is OrderType.LIMIT
            and result.resting_qty > 0
        ):
            self.pool(req.owner, req.side).add(result.order_id)
        if not result.fills:
            self._record_levels(result.touched)
            return
        total = 0.0
        for fill in result.fills:
            total += fill.qty
            if self._recorder is not None:
                self._recorder.trade(self._t, fill.price_tick, fill.qty, fill.taker_side)
            if self.book.order(fill.maker_order_id) is None:
                self.pool(fill.maker_owner, fill.taker_side.opposite).discard(fill.maker_order_id)
            if fill.maker_owner == Owner.EXPERIMENT:
                self._book_agent_fill(fill.maker_order_id, fill.price_tick, fill.qty, Liquidity.MAKER)
        self.last_trade_tick = result.fills[-1].price_tick
        self._record_levels(result.touched)
        self._on_trade(req.owner, req.side, total, result.fills[-1].price_tick * self._tick)

    def _on_trade(self, taker_owner: int, side: Side, qty: float, price: float) -> None:
        if self.impact is None or self.impact_params is None:
            return
        if self.impact_params.source == "experimental" and taker_owner != Owner.EXPERIMENT:
            return
        self.impact.add_trade(self._t, side.sign, qty, price)

    def _record_levels(self, touched: tuple[tuple[Side, int], ...]) -> None:
        if self._recorder is None:
            return
        seen: set[tuple[Side, int]] = set()
        for side, tick in touched:
            if (side, tick) in seen:
                continue
            seen.add((side, tick))
            self._recorder.book_delta(self._t, side, tick, self.book.level_qty(side, tick))

    def _book_agent_fill(self, order_id: int, price_tick: int, qty: float, liquidity: Liquidity) -> None:
        meta = self._agent_meta[order_id]
        price = price_tick * self._tick
        self._agent_fills.append(
            AgentFill(
                order_id=order_id,
                side=meta.side,
                order_type=meta.order_type,
                ts_submit_s=meta.ts_submit,
                ts_fill_s=self._t,
                price=price,
                qty=qty,
                fee=self._fees.fee(price, qty, liquidity),
                liquidity=liquidity,
                arrival_mid=meta.arrival_mid,
                distance_to_touch_bps=meta.distance_bps,
                queue_ahead_at_entry=meta.queue_ahead,
            )
        )

    def perceived_value(self, fundamental: float) -> float:
        return fundamental + (self.impact.value(self._t) if self.impact is not None else 0.0)

    @staticmethod
    def lognormal_size(rng: np.random.Generator, log_mean: float, log_sd: float) -> float:
        return float(math.exp(log_mean + log_sd * rng.standard_normal()))

    # ------------------------------------------------------------------ liquidity helpers
    @staticmethod
    def raw_quote_tick(side: Side, ref_tick: float, offset: int) -> int:
        """Quote ``offset`` ticks away from the reference without looking at the touch (auctions: the
        collected book may cross, and the uncross resolves it)."""
        if side is Side.BUY:
            return math.floor(ref_tick - 0.5) - offset
        return math.ceil(ref_tick + 0.5) + offset

    def quote_tick(self, side: Side, ref_tick: float, offset: int) -> int:
        """Quote ``offset`` ticks away from the reference, never crossing the opposite touch."""
        if side is Side.BUY:
            tick = math.floor(ref_tick - 0.5) - offset
            ask = self.book.best_ask()
            return min(tick, ask - 1) if ask is not None else tick
        tick = math.ceil(ref_tick + 0.5) + offset
        bid = self.book.best_bid()
        return max(tick, bid + 1) if bid is not None else tick

    def post_quote(
        self, owner: int, side: Side, tick: int, lp: LiquidityParams, rng: np.random.Generator
    ) -> None:
        size = self.round_qty(self.lognormal_size(rng, lp.size.log_mean, lp.size.log_sd))
        self.execute(OrderRequest(owner, side, OrderType.LIMIT, size, tick, TimeInForce.POST_ONLY))

    def seed_book(
        self, ib: InitialBook, lp: LiquidityParams, price: float, owner: int, rng: np.random.Generator
    ) -> None:
        centre = round(price / self._tick)
        half = ib.spread_ticks / 2.0
        for level in range(ib.levels_per_side):
            bid = math.floor(centre - half) - level
            ask = math.ceil(centre + half) + level
            for _ in range(ib.orders_per_level):
                self.post_quote(owner, Side.BUY, bid, lp, rng)
                self.post_quote(owner, Side.SELL, ask, lp, rng)

    def ensure_liquidity(
        self, owner: int, lp: LiquidityParams, ref_tick: float, extra_offset: int, rng: np.random.Generator
    ) -> int:
        """Refill a side with ``refill_orders`` quotes when it holds fewer than ``min_orders_per_side`` orders
        of ``owner``; returns the number of refilled sides. This keeps the book two-sided after shocks."""
        refilled = 0
        for side in (Side.BUY, Side.SELL):
            if len(self.pool(owner, side)) >= lp.min_orders_per_side:
                continue
            refilled += 1
            for k in range(lp.refill_orders):
                self.post_quote(
                    owner,
                    side,
                    self.quote_tick(side, ref_tick, lp.min_offset_ticks + extra_offset + k),
                    lp,
                    rng,
                )
        return refilled

    # ------------------------------------------------------------------ subclass hooks
    @abc.abstractmethod
    def _next_event_time(self) -> float | None:
        """Time of the next background event, or ``None`` when the flow is exhausted."""

    @abc.abstractmethod
    def _process_next_event(self) -> None:
        """Apply the next background event at ``self.t``."""

    def _on_clock(self, t: float) -> None:  # noqa: B027  # optional hook
        """Called after every :meth:`advance`; subclasses run time-driven processes here."""
