"""Generative simulator: the trained order-flow model drives the L3 engine.

The model samples the next event's type, time gap, offset and size. The event is then applied to the
current book: offsets are taken from a reference price ``mid + kappa * I(t)`` (the same calibrated impact
coupling as the agent-based market), so the generated flow reacts to the experimental agent's trades;
cancellations remove the generated quote whose offset is closest to the sampled one. The conditioning vector
(funding, open-interest change, volatility regime) is computed from the simulated market itself.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch

from reactive_lob_sim.crypto.macro import MIN_OI_FRACTION, FundingOI
from reactive_lob_sim.engine.accounting import FeeSchedule
from reactive_lob_sim.engine.l3market import L3Market
from reactive_lob_sim.engine.rules import TradingRules
from reactive_lob_sim.engine.types import OrderRequest, OrderType, Owner, Side
from reactive_lob_sim.generative.model import OrderFlowGRU
from reactive_lob_sim.generative.tokenizer import Tokenizer, TokenStream
from reactive_lob_sim.params import CXL_ASK, CXL_BID, LIM_ASK, LIM_BID, MKT_BUY, MKT_SELL, ABMParams
from reactive_lob_sim.tape.tape import TapeRecorder


def _sample(logits: torch.Tensor, rng: np.random.Generator, temperature: float) -> int:
    z = (logits.double().numpy() / temperature).ravel()
    p = np.exp(z - z.max())
    p /= p.sum()
    return int(rng.choice(p.size, p=p))


class GenerativeMarket(L3Market):
    def __init__(
        self,
        model: OrderFlowGRU,
        tok: Tokenizer,
        prime: TokenStream,
        params: ABMParams,
        horizon: float,
        seed: int,
        rules: TradingRules,
        qty_tolerance: float,
        fees: FeeSchedule,
        temperature: float,
        recorder: TapeRecorder | None = None,
        snapshot_depth: int = 0,
    ) -> None:
        super().__init__(rules, qty_tolerance, fees, params.impact, recorder)
        self.model = model
        self.tok = tok
        self.p = params
        self.horizon = horizon
        self._temp = temperature
        seeds = np.random.SeedSequence(seed).spawn(3)
        self._rng = np.random.default_rng(seeds[0])
        # Funding and open interest exist only with the crypto module; for MOEX the conditioning is 0.
        crypto = params.crypto
        self.macro = FundingOI(crypto.macro, np.random.default_rng(seeds[1])) if crypto is not None else None
        self._lp_rng = np.random.default_rng(seeds[2])
        self.funding_rate = self.macro.rate if self.macro is not None else 0.0
        self.open_interest = self.macro.open_interest if self.macro is not None else 0.0
        self._last_log_oi = math.log(self.open_interest) if self.open_interest > 0 else 0.0
        self._oi_change = 0.0
        self._r2: deque[tuple[float, float]] = deque()
        self._r2_sum = 0.0
        self._last_mid: float | None = None
        self.seed_book(
            params.initial_book, params.liquidity, params.initial_price, Owner.GENERATIVE, self._lp_rng
        )
        if snapshot_depth:
            self.record_snapshot(snapshot_depth)
        self.note_mid()
        with torch.no_grad():
            out, self._hidden = model(
                torch.as_tensor(prime.kind[None, :]),
                torch.as_tensor(prime.offset[None, :]),
                torch.as_tensor(prime.size[None, :]),
                torch.as_tensor(prime.dt[None, :]),
                torch.as_tensor(prime.cond[None, :]),
            )
        self._out = out[0, -1]
        self.n_events = 0
        self._pending = self._draw(0.0)

    # ------------------------------------------------------------------ model stepping
    def _draw(self, now: float) -> tuple[float, int, int, int, int] | None:
        with torch.no_grad():
            kind = _sample(self.model.head_type(self._out), self._rng, self._temp)
            dt_l, off_l, size_l = self.model.conditional_heads(self._out, torch.as_tensor(kind))
            dt_bin = _sample(dt_l, self._rng, self._temp)
            off = _sample(off_l, self._rng, self._temp)
            size = _sample(size_l, self._rng, self._temp)
        t = now + self.tok.decode_dt(dt_bin, float(self._rng.random()))
        if t >= self.horizon:
            return None
        return t, kind, off, size, dt_bin

    def _feed(self, kind: int, off: int, size: int, dt_bin: int) -> None:
        cond = self.tok.cond_vector(self.funding_rate, self._oi_change, self._volatility())
        with torch.no_grad():
            out, self._hidden = self.model(
                torch.as_tensor([[kind]]),
                torch.as_tensor([[off]]),
                torch.as_tensor([[size]]),
                torch.as_tensor([[dt_bin]]),
                torch.as_tensor(cond[None, None, :]),
                self._hidden,
            )
        self._out = out[0, -1]

    def _volatility(self) -> float:
        cutoff = self.t - self.tok.vol_window_s
        while self._r2 and self._r2[0][0] < cutoff:
            self._r2_sum -= self._r2.popleft()[1]
        return math.sqrt(max(self._r2_sum, 0.0))

    def _track_mid(self) -> None:
        mid = self.mid_price()
        if mid is None:
            return
        if self._last_mid is not None and mid != self._last_mid:
            r2 = math.log(mid / self._last_mid) ** 2
            self._r2.append((self.t, r2))
            self._r2_sum += r2
        self._last_mid = mid

    # ------------------------------------------------------------------ L3Market hooks
    def _next_event_time(self) -> float | None:
        candidates = []
        if self._pending is not None:
            candidates.append(self._pending[0])
        if self.macro is not None and self.macro.next_fix < self.horizon:
            candidates.append(self.macro.next_fix)
        return min(candidates) if candidates else None

    def _process_next_event(self) -> None:
        if self.macro is not None and self.p.crypto is not None and self.macro.due(self.t):
            mid = self.mid_price() or self.p.initial_price
            self.funding_rate = self.macro.fix(mid, mid, None)
            self.open_interest = self.macro.open_interest
            log_oi = math.log(max(self.open_interest, self.p.crypto.macro.oi_initial * MIN_OI_FRACTION))
            self._oi_change, self._last_log_oi = log_oi - self._last_log_oi, log_oi
            rec = self.recorder
            if rec is not None:
                rec.funding(self.t, self.funding_rate, mid, mid, self.macro.next_fix)
                rec.open_interest(self.t, self.open_interest, mid)
            return
        assert self._pending is not None
        _, kind, off, size_bin, dt_bin = self._pending
        self._apply(kind, off, size_bin)
        self.n_events += 1
        self.ensure_liquidity(Owner.GENERATIVE, self.p.liquidity, self._reference_tick(), 0, self._lp_rng)
        self._track_mid()
        self.note_mid()
        self._feed(kind, off, size_bin, dt_bin)
        self._pending = self._draw(self.t)

    def _reference_tick(self) -> float:
        mid = self.book.mid_tick()
        shift = self.impact.value(self.t) / self.tick_size if self.impact is not None else 0.0
        base = mid if mid is not None else self.p.initial_price / self.tick_size
        return base + shift

    def _apply(self, kind: int, off: int, size_bin: int) -> None:
        size = self.round_qty(float(self.tok.size_values[size_bin]))
        if kind in (MKT_BUY, MKT_SELL):
            side = Side.BUY if kind == MKT_BUY else Side.SELL
            if self.book.best(side.opposite) is not None:
                self.execute(OrderRequest(Owner.GENERATIVE, side, OrderType.MARKET, size))
        elif kind in (LIM_BID, LIM_ASK):
            side = Side.BUY if kind == LIM_BID else Side.SELL
            tick = self.quote_tick(side, self._reference_tick(), off)
            self.execute(OrderRequest(Owner.GENERATIVE, side, OrderType.LIMIT, size, tick))
        elif kind in (CXL_BID, CXL_ASK):
            self._cancel_near(Side.BUY if kind == CXL_BID else Side.SELL, off)

    def _cancel_near(self, side: Side, off: int) -> None:
        pool = self.pool(Owner.GENERATIVE, side)
        if not len(pool):
            return
        target = self.quote_tick(side, self._reference_tick(), off)
        best_id, best_gap = None, None
        for oid in pool.items():
            order = self.book.order(oid)
            if order is None:
                continue
            gap = abs(order.price_tick - target)
            if best_gap is None or gap < best_gap:
                best_id, best_gap = oid, gap
        if best_id is not None:
            self.cancel_background(best_id)
