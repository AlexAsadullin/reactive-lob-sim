"""L3 limit order book with price-time priority.

Each price level is an insertion-ordered ``dict`` of resting orders, which gives FIFO iteration and O(1)
cancellation by id. Sorted price lists per side give the touch in O(1) and insert/remove in O(levels).

Supported: limit and market orders, cancel, modify (a size reduction keeps queue priority; a price change or
a size increase loses it, as on most venues), and the time-in-force flags GTC, IOC, FOK and POST_ONLY.
Market orders behave as IOC. There is no self-trade prevention.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections.abc import Iterator

from reactive_lob_sim.engine.types import (
    CancelResult,
    Fill,
    OrderRequest,
    OrderStatus,
    OrderType,
    RejectReason,
    RestingOrder,
    Side,
    SubmitResult,
    TimeInForce,
)


class OrderBook:
    """A single-instrument L3 book.

    Args:
        qty_tolerance: quantities at or below this are treated as zero (float64 residue guard).
    """

    def __init__(self, qty_tolerance: float) -> None:
        if qty_tolerance <= 0:
            raise ValueError("qty_tolerance must be positive")
        self._eps = qty_tolerance
        self._levels: dict[Side, dict[int, dict[int, RestingOrder]]] = {Side.BUY: {}, Side.SELL: {}}
        self._level_qty: dict[Side, dict[int, float]] = {Side.BUY: {}, Side.SELL: {}}
        self._prices: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: dict[int, RestingOrder] = {}
        self._next_id = 1
        self._arrival = 0

    # ------------------------------------------------------------------ queries
    @property
    def qty_tolerance(self) -> float:
        return self._eps

    def best(self, side: Side) -> int | None:
        prices = self._prices[side]
        if not prices:
            return None
        return prices[-1] if side is Side.BUY else prices[0]

    def best_bid(self) -> int | None:
        return self.best(Side.BUY)

    def best_ask(self) -> int | None:
        return self.best(Side.SELL)

    def mid_tick(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def level_qty(self, side: Side, price_tick: int) -> float:
        return self._level_qty[side].get(price_tick, 0.0)

    def levels(self, side: Side, n: int | None = None) -> list[tuple[int, float]]:
        """Price levels from the touch outwards as ``(tick, qty)``."""
        prices = self._prices[side]
        ordered = reversed(prices) if side is Side.BUY else iter(prices)
        out: list[tuple[int, float]] = []
        for tick in ordered:
            if n is not None and len(out) >= n:
                break
            out.append((tick, self._level_qty[side][tick]))
        return out

    def level_orders(self, side: Side, price_tick: int) -> list[RestingOrder]:
        return list(self._levels[side].get(price_tick, {}).values())

    def order(self, order_id: int) -> RestingOrder | None:
        return self._orders.get(order_id)

    def orders(self) -> Iterator[RestingOrder]:
        return iter(self._orders.values())

    def n_orders(self, side: Side | None = None) -> int:
        if side is None:
            return len(self._orders)
        return sum(len(level) for level in self._levels[side].values())

    def total_qty(self, side: Side | None = None) -> float:
        sides = (Side.BUY, Side.SELL) if side is None else (side,)
        return sum(sum(self._level_qty[s].values()) for s in sides)

    def queue_ahead(self, order_id: int) -> float:
        """Quantity resting ahead of ``order_id`` at its price level."""
        order = self._orders[order_id]
        ahead = 0.0
        for other in self._levels[order.side][order.price_tick].values():
            if other.order_id == order_id:
                return ahead
            ahead += other.qty
        raise KeyError(order_id)

    def is_crossed(self) -> bool:
        bid, ask = self.best_bid(), self.best_ask()
        return bid is not None and ask is not None and bid >= ask

    # ------------------------------------------------------------------ commands
    def submit(self, req: OrderRequest) -> SubmitResult:
        order_id = self._next_id
        self._next_id += 1
        return self._execute(order_id, req)

    def cancel(self, order_id: int) -> CancelResult:
        order = self._orders.get(order_id)
        if order is None:
            return CancelResult(order_id, False, 0.0, ())
        self._remove(order)
        return CancelResult(order_id, True, order.qty, ((order.side, order.price_tick),))

    def modify(
        self, order_id: int, new_qty: float | None = None, new_price_tick: int | None = None
    ) -> SubmitResult:
        """Amend a resting order.

        A pure size reduction keeps the order's queue position. A price change or a size increase is a
        cancel-replace under the same ``order_id``: the order goes to the back of the queue and may trade
        immediately if the new price crosses.
        """
        order = self._orders.get(order_id)
        if order is None:
            return self._reject(order_id, RejectReason.UNKNOWN_ORDER)
        qty = order.qty if new_qty is None else new_qty
        price = order.price_tick if new_price_tick is None else new_price_tick
        if qty <= self._eps:
            return self._reject(order_id, RejectReason.INVALID_QTY)
        if price == order.price_tick and qty <= order.qty:
            delta = order.qty - qty
            order.qty = qty
            self._level_qty[order.side][price] -= delta
            touched = ((order.side, price),)
            ahead = self.queue_ahead(order_id)
            return SubmitResult(order_id, OrderStatus.ACCEPTED, (), qty, None, ahead, touched)
        self._remove(order)
        replace = OrderRequest(order.owner, order.side, OrderType.LIMIT, qty, price, TimeInForce.GTC)
        result = self._execute(order_id, replace)
        touched = ((order.side, order.price_tick), *result.touched)
        return SubmitResult(
            result.order_id,
            result.status,
            result.fills,
            result.resting_qty,
            result.reject_reason,
            result.queue_ahead,
            touched,
        )

    # ------------------------------------------------------------------ internals
    def _reject(self, order_id: int, reason: RejectReason) -> SubmitResult:
        return SubmitResult(order_id, OrderStatus.REJECTED, (), 0.0, reason, 0.0, ())

    def _crosses(self, side: Side, price_tick: int | None) -> bool:
        opp = self.best(side.opposite)
        if opp is None:
            return False
        if price_tick is None:
            return True
        return price_tick >= opp if side is Side.BUY else price_tick <= opp

    def _available(self, side: Side, price_tick: int | None, needed: float) -> float:
        """Opposite-side quantity reachable by a taker of ``side`` within ``price_tick``."""
        total = 0.0
        for tick, qty in self.levels(side.opposite):
            if price_tick is not None and (tick > price_tick if side is Side.BUY else tick < price_tick):
                break
            total += qty
            if total >= needed - self._eps:
                break
        return total

    def _execute(self, order_id: int, req: OrderRequest) -> SubmitResult:
        if req.qty <= self._eps:
            return self._reject(order_id, RejectReason.INVALID_QTY)
        if req.order_type is OrderType.LIMIT and req.price_tick is None:
            return self._reject(order_id, RejectReason.INVALID_PRICE)
        limit = req.price_tick if req.order_type is OrderType.LIMIT else None
        if req.tif is TimeInForce.POST_ONLY and self._crosses(req.side, limit):
            return self._reject(order_id, RejectReason.POST_ONLY_WOULD_CROSS)
        if req.tif is TimeInForce.FOK and self._available(req.side, limit, req.qty) < req.qty - self._eps:
            return self._reject(order_id, RejectReason.FOK_INSUFFICIENT_LIQUIDITY)
        if req.order_type is OrderType.MARKET and self.best(req.side.opposite) is None:
            return self._reject(order_id, RejectReason.NO_LIQUIDITY)

        fills, remaining, touched = self._match(order_id, req, limit)
        rests = (
            req.order_type is OrderType.LIMIT
            and req.tif in (TimeInForce.GTC, TimeInForce.POST_ONLY)
            and remaining > self._eps
        )
        queue_ahead = 0.0
        if rests:
            assert limit is not None
            queue_ahead = self._level_qty[req.side].get(limit, 0.0)
            self._insert(RestingOrder(order_id, req.owner, req.side, limit, remaining, self._arrival))
            self._arrival += 1
            touched.append((req.side, limit))
        filled = req.qty - remaining
        if remaining <= self._eps:
            status = OrderStatus.FILLED
        elif filled > self._eps:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = OrderStatus.ACCEPTED if rests else OrderStatus.CANCELLED
        return SubmitResult(
            order_id,
            status,
            tuple(fills),
            remaining if rests else 0.0,
            None,
            queue_ahead,
            tuple(touched),
        )

    def _match(
        self, order_id: int, req: OrderRequest, limit: int | None
    ) -> tuple[list[Fill], float, list[tuple[Side, int]]]:
        maker_side = req.side.opposite
        remaining = req.qty
        fills: list[Fill] = []
        touched: list[tuple[Side, int]] = []
        while remaining > self._eps:
            tick = self.best(maker_side)
            if tick is None or (limit is not None and not self._crosses(req.side, limit)):
                break
            level = self._levels[maker_side][tick]
            touched.append((maker_side, tick))
            for maker in list(level.values()):
                take = min(remaining, maker.qty)
                fills.append(Fill(maker.order_id, maker.owner, order_id, req.owner, req.side, tick, take))
                remaining -= take
                maker.qty -= take
                self._level_qty[maker_side][tick] -= take
                if maker.qty <= self._eps:
                    self._remove(maker)
                if remaining <= self._eps:
                    break
        return fills, max(remaining, 0.0), touched

    def _insert(self, order: RestingOrder) -> None:
        side_levels = self._levels[order.side]
        level = side_levels.get(order.price_tick)
        if level is None:
            level = {}
            side_levels[order.price_tick] = level
            self._level_qty[order.side][order.price_tick] = 0.0
            insort(self._prices[order.side], order.price_tick)
        level[order.order_id] = order
        self._level_qty[order.side][order.price_tick] += order.qty
        self._orders[order.order_id] = order

    def _remove(self, order: RestingOrder) -> None:
        side_levels = self._levels[order.side]
        level = side_levels[order.price_tick]
        del level[order.order_id]
        del self._orders[order.order_id]
        if not level:
            del side_levels[order.price_tick]
            del self._level_qty[order.side][order.price_tick]
            prices = self._prices[order.side]
            del prices[bisect_left(prices, order.price_tick)]
        else:
            residual = self._level_qty[order.side][order.price_tick] - max(order.qty, 0.0)
            self._level_qty[order.side][order.price_tick] = max(residual, 0.0)
