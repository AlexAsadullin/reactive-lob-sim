"""Agent-based market on the L3 engine.

Event timing comes from a six-dimensional Hawkes process (market buy/sell, limit bid/ask, cancel bid/ask). Its
baseline is modulated by a Markov volatility regime and by the MOEX trading phase (``session_flow``: the
continuous phase of each session has its own intensity, auctions have one, closed periods and breaks none).
What each event does depends on the current state:

market events are sent by noise, fundamental or momentum traders (mixture weights ``traders``); fundamental
traders trade toward the perceived value V = F + I(t) when the mispricing exceeds a threshold; momentum
traders follow the sign of the mid change over a lookback. In an auction a market order becomes a limit at
the price-band edge.
limit events are zero-intelligence quotes around ``mid + pull * (V - mid)``.
cancel events remove ``Poisson(n_orders / n_ref)`` resting quotes on their side.

F is a geometric random walk (the index price) and I(t) is the propagator impact state, whose strength is the
coupling ``impact.kappa``. The trading phase follows the session schedule (segments passed in by the caller)
unless a scenario shock forces an auction or a halt. Shocks from a
:class:`~reactive_lob_sim.scenarios.models.Scenario` modify the flow at given times.

The optional crypto module (``params.crypto``) adds funding and open interest, a funding tilt of noise
traders, and a liquidation book that turns adverse moves into forced market orders.

Randomness uses separate generators per component. The event times and the index path depend only on the
seed, so two runs with the same seed see the same background clock and the same index path whatever the
experimental agent does.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from reactive_lob_sim.crypto.liquidation import LiquidationBook
from reactive_lob_sim.crypto.macro import FundingOI
from reactive_lob_sim.engine.accounting import FeeSchedule
from reactive_lob_sim.engine.l3market import L3Market
from reactive_lob_sim.engine.rules import TradingRules
from reactive_lob_sim.engine.types import OrderRequest, OrderType, Owner, Side, TimeInForce
from reactive_lob_sim.hawkes.mle import simulate_hawkes
from reactive_lob_sim.moex.sessions import Phase, Segment, always_continuous
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

_BPS = 10_000.0
_STREAMS = ("events", "regimes", "index", "flow", "macro", "liquidation")
# Orders of the opening-auction-imbalance shock are spread over this many price points.
_IMBALANCE_ORDERS = 10


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


def phase_timeline(segments: list[Segment], shocks: list[Shock]) -> list[Segment]:
    """Scheduled segments with the phases forced by scenario shocks laid over them."""
    cuts = {s.t0 for s in segments} | {segments[-1].t1}
    for sh in shocks:
        cuts |= {sh.start_s, sh.start_s + sh.duration_s}
    end = segments[-1].t1
    ordered = sorted(c for c in cuts if 0.0 <= c <= end)
    out: list[Segment] = []
    for a, b in itertools.pairwise(ordered):
        base = next(s for s in segments if s.t0 <= a < s.t1)
        phase, session = base.phase, base.session
        for sh in shocks:
            if sh.active(a) and base.phase.accepts_orders:
                if sh.kind == "trading_halt" and sh.params["reopen_auction"] < 0.5:
                    phase = Phase.HALTED
                elif sh.kind == "trading_halt":
                    phase = Phase.AUCTION
                else:
                    phase = Phase.OPENING_AUCTION
        if out and out[-1].phase is phase and out[-1].session == session:
            out[-1] = Segment(out[-1].t0, b, session, phase)
        else:
            out.append(Segment(a, b, session, phase))
    return out


class ABMMarket(L3Market):
    def __init__(
        self,
        params: ABMParams,
        horizon: float,
        seed: int,
        rules: TradingRules,
        qty_tolerance: float,
        fees: FeeSchedule,
        recorder: TapeRecorder | None = None,
        scenario: Scenario | None = None,
        snapshot_depth: int = 0,
        segments: list[Segment] | None = None,
    ) -> None:
        super().__init__(rules, qty_tolerance, fees, params.impact, recorder)
        if scenario is not None and scenario.needs_crypto and params.crypto is None:
            raise ValueError(f"scenario {scenario.id!r} needs the crypto module, which is off")
        self.p = params
        self.horizon = horizon
        seeds = np.random.SeedSequence(seed).spawn(len(_STREAMS))
        self._rng = {name: np.random.default_rng(s) for name, s in zip(_STREAMS, seeds, strict=True)}
        breaks, states = regime_path(params, horizon, self._rng["regimes"])
        self._regime_breaks = breaks
        self._regime_states = states
        self.schedule = ShockSchedule(scenario.scaled_shocks() if scenario else [])
        self.timeline = phase_timeline(segments or always_continuous(horizon), self.schedule.phase_shocks())
        hawkes_breaks, rate_mult = self._rate_path()
        event_seed = int(self._rng["events"].integers(0, 2**31 - 1))
        self._ev_t, self._ev_k = simulate_hawkes(params.hawkes, horizon, event_seed, hawkes_breaks, rate_mult)
        self._ev_i = 0
        self._ev_t_list: list[float] = self._ev_t.tolist()
        self._ev_k_list: list[int] = self._ev_k.tolist()
        self.fundamental = params.initial_price
        self._fund_t = 0.0
        self.crypto = params.crypto
        self.macro = FundingOI(params.crypto.macro, self._rng["macro"]) if params.crypto else None
        self.liquidations = (
            LiquidationBook(params.crypto.liquidation, self._rng["liquidation"]) if params.crypto else None
        )
        if self.macro is not None:
            self.funding_rate = self.macro.rate
            self.open_interest = self.macro.open_interest
        self.n_liquidations = 0
        self.n_refills = 0
        self._snapshot_depth = snapshot_depth
        self._seg_i = 0
        self._seed_book()

    # ------------------------------------------------------------------ setup
    def _rate_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Breaks and baseline multipliers: regime multiplier times the phase multiplier."""
        cuts = sorted({*self._regime_breaks.tolist(), *(s.t0 for s in self.timeline)})
        mult = np.asarray([self.regime_rate(t) * self.phase_rate(self.segment_at(t)) for t in cuts])
        return np.asarray(cuts, dtype=np.float64), mult

    def regime_rate(self, t: float) -> float:
        return self.p.regimes[self.regime(t)].rate_multiplier

    def phase_rate(self, seg: Segment) -> float:
        if seg.phase is Phase.CONTINUOUS:
            return self.p.session_flow.multiplier(seg.session or "main")
        if seg.phase.is_auction:
            return self.p.session_flow.auction_rate_multiplier
        return 0.0

    def segment_at(self, t: float) -> Segment:
        for seg in self.timeline:
            if seg.t0 <= t < seg.t1:
                return seg
        return self.timeline[-1]

    def _seed_book(self) -> None:
        self.seed_book(
            self.p.initial_book, self.p.liquidity, self.p.initial_price, Owner.LIQUIDITY, self._rng["flow"]
        )
        if self._snapshot_depth:
            self.record_snapshot(self._snapshot_depth)
        self.note_mid()
        first = self.timeline[0]
        self.set_phase(first.phase, first.session)

    # ------------------------------------------------------------------ state
    @property
    def n_scheduled_events(self) -> int:
        """Number of background flow events drawn for the whole horizon."""
        return len(self._ev_t_list)

    def regime(self, t: float) -> int:
        i = int(np.searchsorted(self._regime_breaks, t, side="right") - 1)
        return int(self._regime_states[max(i, 0)])

    def _step_fundamental(self, t: float) -> None:
        dt = t - self._fund_t
        if dt <= 0:
            return
        sigma = self.p.fundamental_vol_per_sqrt_s * self.p.regimes[self.regime(t)].vol_multiplier
        sigma *= self.schedule.vol_multiplier(t)
        z = self._rng["index"].standard_normal()
        self.fundamental *= math.exp(sigma * math.sqrt(dt) * z - 0.5 * sigma * sigma * dt)
        self._fund_t = t

    def value(self) -> float:
        return self.perceived_value(self.fundamental)

    # ------------------------------------------------------------------ L3Market hooks
    def _next_boundary(self) -> float | None:
        nxt = self._seg_i + 1
        return self.timeline[nxt].t0 if nxt < len(self.timeline) else None

    def _next_event_time(self) -> float | None:
        nxt = self._ev_t_list[self._ev_i] if self._ev_i < len(self._ev_t_list) else None
        fix = self.macro.next_fix if self.macro is not None else None
        refresh = (
            self.liquidations.next_refresh
            if self.liquidations is not None and self._liquidations_on()
            else None
        )
        events = (nxt, fix, refresh, self._next_boundary())
        candidates = [x for x in events if x is not None and x < self.horizon]
        return min(candidates) if candidates else None

    def _liquidations_on(self) -> bool:
        return self.crypto is not None and self.crypto.liquidation.enabled

    def _process_next_event(self) -> None:
        t = self.t
        boundary = self._next_boundary()
        crossed = boundary is not None and t >= boundary
        if crossed:
            self._seg_i += 1
            seg = self.timeline[self._seg_i]
            self.set_phase(seg.phase, seg.session)
        for shock in self.schedule.due(t):
            self._start_shock(shock)
        if crossed:
            self._ensure_liquidity()
            self.note_mid()
            return
        if self.macro is not None and self.macro.due(t):
            self._fix_funding()
            return
        if self.liquidations is not None and self._liquidations_on() and t >= self.liquidations.next_refresh:
            mid = self.mid_price()
            if mid is not None:
                self.liquidations.refresh(
                    mid, self.funding_rate, self.open_interest, self.schedule.long_share_override(t)
                )
            assert self.crypto is not None
            self.liquidations.next_refresh += self.crypto.liquidation.refresh_interval_s
            return
        kind = self._ev_k_list[self._ev_i]
        self._ev_i += 1
        self._step_fundamental(t)
        if not self.phase.accepts_orders:
            return
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
        assert self.macro is not None
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
        if who == 0 and self.crypto is not None:
            c = self.crypto
            tilt = c.funding_flow_sensitivity * math.tanh(self.funding_rate / c.funding_scale)
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
        if self.phase.is_auction or self.book.best(side.opposite) is not None:
            self.execute(OrderRequest(owner, side, OrderType.MARKET, size))

    def _reference_tick(self) -> float:
        mid = self.book.mid_tick()
        value_tick = self.value() / self.tick_size
        if mid is None or self.book.is_crossed():
            return value_tick
        return mid + self.p.liquidity.pull_to_value * (value_tick - mid)

    def _limit_event(self, side: Side) -> None:
        lp = self.p.liquidity
        flow = self._rng["flow"]
        offset = (
            lp.min_offset_ticks
            + self.schedule.extra_offset_ticks(self.t)
            + int(flow.geometric(lp.offset_geom_p))
            - 1
        )
        ref = self._reference_tick()
        if self.phase.is_auction:
            tick = self.raw_quote_tick(side, ref, offset)
        else:
            tick = self.quote_tick(side, ref, offset)
        self.post_quote(Owner.LIQUIDITY, side, self.rules.clamp(tick), lp, flow)

    def _cancel_event(self, side: Side) -> None:
        pool = self.pool(Owner.LIQUIDITY, side)
        if not len(pool):
            return
        flow = self._rng["flow"]
        n = int(flow.poisson(len(pool) / self.p.liquidity.n_ref_orders))
        for _ in range(min(n, len(pool))):
            self.cancel_background(pool.sample(flow))

    def _ensure_liquidity(self) -> None:
        if self.phase is not Phase.CONTINUOUS:
            return
        extra = self.schedule.extra_offset_ticks(self.t)
        self.n_refills += self.ensure_liquidity(
            Owner.LIQUIDITY, self.p.liquidity, self._reference_tick(), extra, self._rng["flow"]
        )

    def _check_liquidations(self) -> None:
        if self.liquidations is None or not self._liquidations_on() or self.macro is None:
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
                # A forced order may meet an emptied side; liquidity providers refill it first, so the
                # liquidation still executes (at a worse price) instead of silently disappearing.
                self._ensure_liquidity()
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
    def _cancel_liquidity(self, fraction: float, band_ticks: float | None = None) -> None:
        flow = self._rng["flow"]
        mid = self.book.mid_tick()
        for side in (Side.BUY, Side.SELL):
            for oid in self.pool(Owner.LIQUIDITY, side).items():
                order = self.book.order(oid)
                if order is None:
                    continue
                if band_ticks is not None and (mid is None or abs(order.price_tick - mid) > band_ticks):
                    continue
                if fraction >= 1.0 or flow.random() < fraction:
                    self.cancel_background(oid)

    def _jump_fundamental(self, bps: float) -> None:
        self._step_fundamental(self.t)
        self.fundamental *= 1.0 + bps / _BPS

    def _start_shock(self, shock: Shock) -> None:
        p = shock.params
        if shock.kind == "spread_widening":
            self._cancel_liquidity(1.0, p["extra_ticks"] + self.p.liquidity.min_offset_ticks)
        elif shock.kind == "liquidity_withdrawal":
            self._cancel_liquidity(p["cancel_fraction"])
        elif shock.kind == "opening_auction_imbalance":
            self._add_imbalance(p["imbalance"], p["depth_multiple"], p["price_range_bps"])
        elif shock.kind == "dividend_gap":
            self._jump_fundamental(-p["dividend_bps"])
            self._cancel_liquidity(1.0)
        elif shock.kind == "key_rate_surprise":
            self._jump_fundamental(p["surprise_bp"] * p["bps_per_bp"])
            self._cancel_liquidity(p["cancel_fraction"])
        elif shock.kind == "trading_halt":
            self._jump_fundamental(p["jump_bps"])
        elif shock.kind == "liquidation_cascade":
            self._liquidation_cascade(shock)
        self._ensure_liquidity()

    def _add_imbalance(self, imbalance: float, depth_multiple: float, range_bps: float) -> None:
        if imbalance == 0:
            return
        side = Side.BUY if imbalance > 0 else Side.SELL
        levels = self.book.levels(side.opposite, self.p.initial_book.levels_per_side)
        touch = levels[0][0] if levels else self.book.best(side)
        if touch is None:
            return
        depth = sum(q for _, q in levels) or self.p.initial_book.levels_per_side * self.lot_size
        total = abs(imbalance) * depth_multiple * depth
        span = range_bps / _BPS * touch
        child = self.round_qty(total / _IMBALANCE_ORDERS)
        for k in range(_IMBALANCE_ORDERS):
            tick = self.rules.clamp(round(touch + side.sign * span * k / max(_IMBALANCE_ORDERS - 1, 1)))
            self.execute(OrderRequest(Owner.AUCTION, side, OrderType.LIMIT, child, tick, TimeInForce.GTC))

    def _liquidation_cascade(self, shock: Shock) -> None:
        assert self.macro is not None
        assert self.liquidations is not None
        assert self.crypto is not None
        p = shock.params
        self.macro.scale_oi(p["oi_multiplier"])
        self.open_interest = self.macro.open_interest
        mid = self.mid_price()
        if mid is not None:
            crowd = self.crypto.liquidation.oi_fraction_at_risk * self.open_interest * p["long_share"]
            self.liquidations.add_crowding(mid, crowd, int(p["layers"]), p["layer_gap_bps"] / _BPS)
            depth = sum(q for _, q in self.book.levels(Side.BUY, self.p.initial_book.levels_per_side))
            qty = self.round_qty(depth * p["trigger_depth_fraction"])
            self.execute(OrderRequest(Owner.LIQUIDATION, Side.SELL, OrderType.MARKET, qty))
        self._check_liquidations()
