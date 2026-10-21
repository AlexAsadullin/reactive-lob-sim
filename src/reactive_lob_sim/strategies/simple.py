"""The three reference strategies of the gap-score study: passive market making, momentum taking, TWAP.

They are deliberately simple; what matters for the gap score is that they stress different simulator
properties: queue position and adverse selection (market maker), price dynamics and impact (momentum), and
liquidity cost of a fixed-size execution (TWAP).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from reactive_lob_sim.engine.types import OrderStatus, OrderType, Side, TimeInForce
from reactive_lob_sim.settings import StrategySpec
from reactive_lob_sim.strategies.base import Gateway, Strategy
from reactive_lob_sim.venue import AgentFill, MarketView

_BPS = 10_000.0


class _PositionTracker:
    def __init__(self) -> None:
        self.position = 0.0

    def on_fill(self, fill: AgentFill) -> None:
        self.position += fill.side.sign * fill.qty


@dataclass(frozen=True)
class MarketMakerParams:
    quote_size: float
    offset_ticks: int
    max_inventory: float
    requote_ticks: int


class MarketMaker(_PositionTracker):
    """Posts post-only quotes ``offset_ticks`` behind the touch on both sides and re-quotes when the touch
    moves more than ``requote_ticks`` away; it stops quoting the side that would breach ``max_inventory``."""

    def __init__(self, strategy_id: str, p: MarketMakerParams) -> None:
        super().__init__()
        self.strategy_id = strategy_id
        self.p = p
        self._quotes: dict[Side, tuple[int, float]] = {}

    def on_start(self, view: MarketView, horizon: float) -> None:
        self._quotes.clear()

    def on_step(self, view: MarketView, gw: Gateway) -> None:
        if view.best_bid is None or view.best_ask is None:
            return
        live = set(gw.open_orders())
        tick = view.tick_size
        targets = {
            Side.BUY: view.best_bid - self.p.offset_ticks * tick,
            Side.SELL: view.best_ask + self.p.offset_ticks * tick,
        }
        for side, target in targets.items():
            quote = self._quotes.get(side)
            if quote is not None and quote[0] not in live:
                quote = None
                del self._quotes[side]
            allowed = side.sign * self.position + self.p.quote_size <= self.p.max_inventory
            if quote is not None:
                stale = abs(quote[1] - target) > self.p.requote_ticks * tick + tick / 2
                if stale or not allowed:
                    gw.cancel(quote[0])
                    del self._quotes[side]
                    quote = None
            if quote is None and allowed:
                ack = gw.submit(side, OrderType.LIMIT, self.p.quote_size, target, TimeInForce.POST_ONLY)
                if ack.status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                    self._quotes[side] = (ack.order_id, target)


@dataclass(frozen=True)
class MomentumParams:
    lookback_s: float
    threshold_bps: float
    trade_size: float
    max_position: float


class Momentum(_PositionTracker):
    """Targets +max (-max) position when the mid return over ``lookback_s`` exceeds +threshold (-threshold)
    and flat otherwise; it moves toward the target by one market order of ``trade_size`` per step."""

    def __init__(self, strategy_id: str, p: MomentumParams) -> None:
        super().__init__()
        self.strategy_id = strategy_id
        self.p = p
        self._hist: deque[tuple[float, float]] = deque()

    def on_start(self, view: MarketView, horizon: float) -> None:
        self._hist.clear()

    def on_step(self, view: MarketView, gw: Gateway) -> None:
        mid = view.mid
        if mid is None:
            return
        self._hist.append((view.t, mid))
        while len(self._hist) > 1 and self._hist[1][0] <= view.t - self.p.lookback_s:
            self._hist.popleft()
        t0, m0 = self._hist[0]
        if view.t - t0 < self.p.lookback_s - 1e-9:
            return
        ret_bps = (mid / m0 - 1.0) * _BPS
        if ret_bps > self.p.threshold_bps:
            target = self.p.max_position
        elif ret_bps < -self.p.threshold_bps:
            target = -self.p.max_position
        else:
            target = 0.0
        gap = target - self.position
        if abs(gap) < gw.lot_size / 2:
            return
        side = Side.BUY if gap > 0 else Side.SELL
        gw.submit(side, OrderType.MARKET, min(abs(gap), self.p.trade_size))


@dataclass(frozen=True)
class TwapParams:
    side: Side
    total_qty: float
    style: str
    catchup_fraction: float


class Twap(_PositionTracker):
    """Executes ``total_qty`` evenly over the horizon.

    ``market`` style sends a market order for the shortfall against the linear schedule at each step.
    ``passive`` style keeps one post-only order at the touch for the shortfall, and switches to market
    orders once the shortfall exceeds ``catchup_fraction`` of the total, or at the final step.
    """

    def __init__(self, strategy_id: str, p: TwapParams) -> None:
        super().__init__()
        self.strategy_id = strategy_id
        self.p = p
        self._t0 = 0.0
        self._horizon = 1.0
        self._order: int | None = None

    def on_start(self, view: MarketView, horizon: float) -> None:
        self._t0 = view.t
        self._horizon = horizon
        self._order = None

    @property
    def executed(self) -> float:
        return self.p.side.sign * self.position

    def on_step(self, view: MarketView, gw: Gateway) -> None:
        if view.best_bid is None or view.best_ask is None:
            return
        frac = min((view.t - self._t0) / self._horizon, 1.0)
        shortfall = self.p.total_qty * frac - self.executed
        if self._order is not None and self._order in gw.open_orders():
            gw.cancel(self._order)
        self._order = None
        if shortfall < gw.lot_size / 2:
            return
        urgent = shortfall > self.p.total_qty * self.p.catchup_fraction or frac >= 1.0
        if self.p.style == "market" or urgent:
            gw.submit(self.p.side, OrderType.MARKET, shortfall)
            return
        touch = view.best_bid if self.p.side is Side.BUY else view.best_ask
        ack = gw.submit(self.p.side, OrderType.LIMIT, shortfall, touch, TimeInForce.POST_ONLY)
        if ack.status is OrderStatus.ACCEPTED:
            self._order = ack.order_id


def build_strategy(spec: StrategySpec) -> Strategy:
    p = spec.params
    if spec.kind == "market_maker":
        return MarketMaker(
            spec.id,
            MarketMakerParams(
                float(p["quote_size"]),
                int(p["offset_ticks"]),
                float(p["max_inventory"]),
                int(p["requote_ticks"]),
            ),
        )
    if spec.kind == "momentum":
        return Momentum(
            spec.id,
            MomentumParams(
                float(p["lookback_s"]),
                float(p["threshold_bps"]),
                float(p["trade_size"]),
                float(p["max_position"]),
            ),
        )
    style = str(p["style"])
    if style not in ("market", "passive"):
        raise ValueError(f"unknown TWAP style {style!r}")
    return Twap(
        spec.id, TwapParams(Side(str(p["side"])), float(p["total_qty"]), style, float(p["catchup_fraction"]))
    )
