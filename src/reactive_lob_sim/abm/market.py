"""Agent-based market on the L3 engine.

Event timing comes from a six-dimensional Hawkes process (market buy/sell, limit bid/ask, cancel bid/ask),
optionally modulated by a Markov volatility regime. What each event does depends on the current state:

market events are sent by noise, fundamental or momentum traders (mixture weights ``traders``); noise
traders tilt toward selling when funding is positive (longs paying funding de-lever); fundamental traders
trade toward the perceived value V = F + I(t) when the mispricing exceeds a threshold; momentum traders
follow the sign of the mid change over a lookback.
limit events are zero-intelligence quotes around ``mid + pull * (V - mid)``.
cancel events remove ``Poisson(n_orders / n_ref)`` resting quotes on their side.

F is a geometric random walk (the index price) and I(t) is the propagator impact state, whose strength is the
coupling ``impact.kappa``. A liquidation book turns adverse moves into forced market orders, conditioned on
funding and open interest. Shocks from a :class:`~reactive_lob_sim.scenarios.models.Scenario` modify the
flow at given times.

Randomness uses separate generators per component. The event times and the index path depend only on the
seed, so two runs with the same seed see the same background clock and the same index path whatever the
experimental agent does.
"""

from __future__ import annotations

import math

import numpy as np

from reactive_lob_sim.abm.liquidation import LiquidationBook
from reactive_lob_sim.abm.macro import FundingOI
from reactive_lob_sim.engine.accounting import FeeSchedule
from reactive_lob_sim.engine.l3market import L3Market
from reactive_lob_sim.engine.types import OrderRequest, OrderType, Owner, Side, TimeInForce
from reactive_lob_sim.hawkes.mle import simulate_hawkes
from reactive_lob_sim.params import (
    CXL_ASK,
    CXL_BID,
    LIM_ASK,
    LIM_BID,
    MKT_BUY,
    MKT_SELL,
    ABMParams,
)
from reactive_lob_sim.scenarios.models import Scenario, Shock, ShockSchedule
from reactive_lob_sim.tape.tape import TapeRecorder

_STREAMS = ("events", "regimes", "index", "flow", "macro", "liquidation")


def regime_path(params: ABMParams, horizon: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant regime indices: ``(break times starting at 0, regime index per segment)``."""
    breaks = [0.0]
    states = [0]
    n = len(params.regimes)
    if n == 1:
        return np.zeros(1), np.zeros(1, dtype=np.int64)
    t, state = 0.0, 0
    while True:
        t += float(rng.exponential(params.regimes[state].mean_duration_s))
        if t >= horizon:
            break
        state = int((state + 1 + rng.integers(0, n - 1)) % n)
        breaks.append(t)
        states.append(state)
    return np.asarray(breaks), np.asarray(states, dtype=np.int64)


class ABMMarket(L3Market):
    def __init__(
        self,
        params: ABMParams,
        horizon: float,
        seed: int,
        tick_size: float,
        lot_size: float,
        qty_tolerance: float,
        fees: FeeSchedule,
        recorder: TapeRecorder | None = None,
        scenario: Scenario | None = None,
        snapshot_depth: int = 0,
    ) -> None:
        super().__init__(tick_size, lot_size, qty_tolerance, fees, params.impact, recorder)
        self.p = params
        self.horizon = horizon
        seeds = np.random.SeedSequence(seed).spawn(len(_STREAMS))
        self._rng = {name: np.random.default_rng(s) for name, s in zip(_STREAMS, seeds, strict=True)}
        breaks, states = regime_path(params, horizon, self._rng["regimes"])
        self._regime_breaks = breaks
        self._regime_states = states
        rate_mult = np.asarray([params.regimes[s].rate_multiplier for s in states])
        event_seed = int(self._rng["events"].integers(0, 2**31 - 1))
        self._ev_t, self._ev_k = simulate_hawkes(params.hawkes, horizon, event_seed, breaks, rate_mult)
        self._ev_i = 0
        self._ev_t_list: list[float] = self._ev_t.tolist()
        self._ev_k_list: list[int] = self._ev_k.tolist()
        self.fundamental = params.initial_price
        self._fund_t = 0.0
        self.schedule = ShockSchedule(scenario.shocks if scenario else [])
        self.macro = FundingOI(params.macro, self._rng["macro"])
        self.funding_rate = self.macro.rate
        self.open_interest = self.macro.open_interest
        self.liquidations = LiquidationBook(params.liquidation, self._rng["liquidation"])
        self.n_liquidations = 0
        self.n_refills = 0
        self._snapshot_depth = snapshot_depth
        self._seed_book()

    # ------------------------------------------------------------------ setup
    def _seed_book(self) -> None:
        ib = self.p.initial_book
        centre = round(self.p.initial_price / self.tick_size)
        half = ib.spread_ticks / 2.0
        lp = self.p.liquidity
        flow = self._rng["flow"]
        for level in range(ib.levels_per_side):
            bid = math.floor(centre - half) - level
            ask = math.ceil(centre + half) + level
            for _ in range(ib.orders_per_level):
                for side, tick in ((Side.BUY, bid), (Side.SELL, ask)):
                    size = self.round_qty(self.lognormal_size(flow, lp.size.log_mean, lp.size.log_sd))
                    self.execute(OrderRequest(Owner.LIQUIDITY, side, OrderType.LIMIT, size, tick))
        if self._snapshot_depth:
            self.record_snapshot(self._snapshot_depth)
        self.note_mid()

    # ------------------------------------------------------------------ state
    def regime(self, t: float) -> int:
        i = int(np.searchsorted(self._regime_breaks, t, side="right") - 1)
        return int(self._regime_states[max(i, 0)])

    def _step_fundamental(self, t: float) -> None:
        dt = t - self._fund_t
        if dt <= 0:
            return
        sigma = self.p.fundamental_vol_per_sqrt_s * self.p.regimes[self.regime(t)].vol_multiplier
        z = self._rng["index"].standard_normal()
        self.fundamental *= math.exp(sigma * math.sqrt(dt) * z - 0.5 * sigma * sigma * dt)
        self._fund_t = t

    def value(self) -> float:
        return self.perceived_value(self.fundamental)

    # ------------------------------------------------------------------ L3Market hooks
    def _next_event_time(self) -> float | None:
        nxt = self._ev_t_list[self._ev_i] if self._ev_i < len(self._ev_t_list) else None
        fix = self.macro.next_fix if self.macro.next_fix < self.horizon else None
        refresh = self.liquidations.next_refresh if self.p.liquidation.enabled else None
        candidates = [x for x in (nxt, fix, refresh) if x is not None and x < self.horizon]
        return min(candidates) if candidates else None

    def _process_next_event(self) -> None:
        t = self.t
        for shock in self.schedule.due(t):
            self._start_shock(shock)
        if self.macro.due(t):
            self._fix_funding()
            return
        if self.p.liquidation.enabled and t >= self.liquidations.next_refresh:
            mid = self.mid_price()
            if mid is not None:
                self.liquidations.refresh(
                    mid, self.funding_rate, self.open_interest, self.schedule.long_share_override(t)
                )
            self.liquidations.next_refresh += self.p.liquidation.refresh_interval_s
            return
        kind = self._ev_k_list[self._ev_i]
        self._ev_i += 1
        self._step_fundamental(t)
        if kind in (MKT_BUY, MKT_SELL):
            self._market_event(Side.BUY if kind == MKT_BUY else Side.SELL)
        elif kind in (LIM_BID, LIM_ASK):
            if self._rng["flow"].random() < self.schedule.limit_rate_factor(t):
                self._limit_event(Side.BUY if kind == LIM_BID else Side.SELL)
        elif kind in (CXL_BID, CXL_ASK):
            self._cancel_event(Side.BUY if kind == CXL_BID else Side.SELL)
        self._ensure_liquidity()
        self._check_liquidations()
        self.note_mid()

    def _on_clock(self, t: float) -> None:
        self._ensure_liquidity()

    # ------------------------------------------------------------------ flow
    def _fix_funding(self) -> None:
        mid = self.mid_price() or self.fundamental
        self._step_fundamental(self.t)
        rate = self.macro.fix(mid, self.fundamental, self.schedule.funding_override(self.t))
        self.funding_rate = rate
        self.open_interest = self.macro.open_interest
        rec = self.recorder
        if rec is not None:
            rec.funding(self.t, rate, mid, self.fundamental, self.macro.next_fix)
            rec.open_interest(self.t, self.open_interest, mid)

    def _market_event(self, hawkes_side: Side) -> None:
        tr = self.p.traders
        flow = self._rng["flow"]
        weights = np.asarray([tr.noise, tr.fundamental, tr.momentum])
        who = int(flow.choice(3, p=weights / weights.sum()))
        side = hawkes_side
        mid = self.mid_price()
        owner = Owner.NOISE
        if who == 0:
            tilt = tr.funding_flow_sensitivity * math.tanh(self.funding_rate / tr.funding_scale)
            if flow.random() < abs(tilt):
                side = Side.SELL if tilt > 0 else Side.BUY
        elif who == 1 and mid is not None:
            owner = Owner.FUNDAMENTAL
            gap = self.value() - mid
            if abs(gap) > tr.fundamental_threshold_ticks * self.tick_size:
                side = Side.BUY if gap > 0 else Side.SELL
        elif who == 2 and mid is not None:
            owner = Owner.MOMENTUM
            past = self.mid_at(self.t - tr.momentum_lookback_s)
            if past is not None and past != mid:
                side = Side.BUY if mid > past else Side.SELL
        size = self.round_qty(self.lognormal_size(flow, tr.size.log_mean, tr.size.log_sd))
        if self.book.best(side.opposite) is not None:
            self.execute(OrderRequest(owner, side, OrderType.MARKET, size))

    def _reference_tick(self) -> float:
        mid = self.book.mid_tick()
        value_tick = self.value() / self.tick_size
        if mid is None:
            return value_tick
        return mid + self.p.liquidity.pull_to_value * (value_tick - mid)

    def _quote_tick(self, side: Side, ref: float, offset: int) -> int:
        if side is Side.BUY:
            tick = math.floor(ref - 0.5) - offset
            ask = self.book.best_ask()
            return min(tick, ask - 1) if ask is not None else tick
        tick = math.ceil(ref + 0.5) + offset
        bid = self.book.best_bid()
        return max(tick, bid + 1) if bid is not None else tick

    def _limit_event(self, side: Side) -> None:
        lp = self.p.liquidity
        flow = self._rng["flow"]
        offset = (
            lp.min_offset_ticks
            + self.schedule.extra_offset_ticks(self.t)
            + int(flow.geometric(lp.offset_geom_p))
            - 1
        )
        tick = self._quote_tick(side, self._reference_tick(), offset)
        size = self.round_qty(self.lognormal_size(flow, lp.size.log_mean, lp.size.log_sd))
        self.execute(OrderRequest(Owner.LIQUIDITY, side, OrderType.LIMIT, size, tick, TimeInForce.POST_ONLY))

    def _cancel_event(self, side: Side) -> None:
        pool = self.pool(Owner.LIQUIDITY, side)
        if not len(pool):
            return
        flow = self._rng["flow"]
        n = int(flow.poisson(len(pool) / self.p.liquidity.n_ref_orders))
        for _ in range(min(n, len(pool))):
            self.cancel_background(pool.sample(flow))

    def _ensure_liquidity(self) -> None:
        lp = self.p.liquidity
        flow = self._rng["flow"]
        for side in (Side.BUY, Side.SELL):
            if len(self.pool(Owner.LIQUIDITY, side)) >= lp.min_orders_per_side:
                continue
            self.n_refills += 1
            ref = self._reference_tick()
            for k in range(lp.refill_orders):
                offset = lp.min_offset_ticks + self.schedule.extra_offset_ticks(self.t) + k
                tick = self._quote_tick(side, ref, offset)
                size = self.round_qty(self.lognormal_size(flow, lp.size.log_mean, lp.size.log_sd))
                self.execute(
                    OrderRequest(Owner.LIQUIDITY, side, OrderType.LIMIT, size, tick, TimeInForce.POST_ONLY)
                )

    def _check_liquidations(self) -> None:
        if not self.p.liquidation.enabled:
            return
        while True:
            mid = self.mid_price()
            if mid is None:
                return
            forced = self.liquidations.triggered(mid)
            if not forced:
                return
            for side, size in forced:
                qty = self.round_qty(size)
                if self.book.best(side.opposite) is None:
                    continue
                result = self.execute(OrderRequest(Owner.LIQUIDATION, side, OrderType.MARKET, qty))
                self.n_liquidations += 1
                self.macro.reduce_oi(result.filled_qty)
                self.open_interest = self.macro.open_interest
                rec = self.recorder
                if rec is not None and result.fills:
                    rec.liquidation(
                        self.t, side, result.fills[-1].price_tick * self.tick_size, result.filled_qty
                    )
            self._ensure_liquidity()

    # ------------------------------------------------------------------ scenarios
    def _start_shock(self, shock: Shock) -> None:
        flow = self._rng["flow"]
        if shock.kind == "spread_widening":
            mid = self.book.mid_tick()
            band = shock.params["extra_ticks"] + self.p.liquidity.min_offset_ticks
            if mid is not None:
                for side in (Side.BUY, Side.SELL):
                    for oid in self.pool(Owner.LIQUIDITY, side).items():
                        order = self.book.order(oid)
                        if order is not None and abs(order.price_tick - mid) <= band:
                            self.cancel_background(oid)
        elif shock.kind == "liquidity_withdrawal":
            for side in (Side.BUY, Side.SELL):
                for oid in self.pool(Owner.LIQUIDITY, side).items():
                    if flow.random() < shock.params["cancel_fraction"]:
                        self.cancel_background(oid)
        elif shock.kind == "liquidation_cascade":
            self.macro.scale_oi(shock.params["oi_multiplier"])
            self.open_interest = self.macro.open_interest
            mid = self.mid_price()
            if mid is not None:
                crowd = (
                    self.p.liquidation.oi_fraction_at_risk * self.open_interest * shock.params["long_share"]
                )
                self.liquidations.add_crowding(mid, crowd, is_long=True)
                depth = sum(q for _, q in self.book.levels(Side.BUY, self.p.initial_book.levels_per_side))
                qty = self.round_qty(depth * shock.params["trigger_depth_fraction"])
                self.execute(OrderRequest(Owner.LIQUIDATION, Side.SELL, OrderType.MARKET, qty))
            self._check_liquidations()
        self._ensure_liquidity()
