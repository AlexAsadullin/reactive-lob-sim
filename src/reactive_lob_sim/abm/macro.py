"""Funding rate and open interest of the simulated perpetual swap."""

from __future__ import annotations

import math

import numpy as np

from reactive_lob_sim.params import MacroParams

# Liquidations can wipe out open interest; the log-OU step needs a positive floor relative to the start value.
_MIN_OI_FRACTION = 1e-6


class FundingOI:
    """Funding fixes at every ``funding_interval_s``; open interest is a log-OU process between fixes.

    funding_k = clip(base + premium_coeff * (mark - index) / index + noise_sd * z, -clip, clip)
    log OI_{k+1} = log OI_k - theta (log OI_k - log OI_0) dt + sigma sqrt(dt) z'
    """

    def __init__(self, params: MacroParams, rng: np.random.Generator) -> None:
        self.p = params
        self._rng = rng
        self.rate = params.funding_base
        self.open_interest = params.oi_initial
        self._log_oi0 = math.log(params.oi_initial)
        self.next_fix = params.funding_interval_s

    def due(self, t: float) -> bool:
        return t >= self.next_fix

    def fix(self, mark: float, index: float, override: float | None) -> float:
        """Compute the funding at the current fix time and schedule the next one."""
        p = self.p
        if override is not None:
            self.rate = override
        else:
            premium = (mark - index) / index if index > 0 else 0.0
            raw = (
                p.funding_base
                + p.funding_premium_coeff * premium
                + p.funding_noise_sd * self._rng.standard_normal()
            )
            self.rate = float(np.clip(raw, -p.funding_clip, p.funding_clip))
        dt = p.funding_interval_s
        log_oi = math.log(max(self.open_interest, p.oi_initial * _MIN_OI_FRACTION))
        log_oi += -p.oi_reversion_per_s * (log_oi - self._log_oi0) * dt
        log_oi += p.oi_vol_per_sqrt_s * math.sqrt(dt) * self._rng.standard_normal()
        self.open_interest = math.exp(log_oi)
        self.next_fix += dt
        return self.rate

    def reduce_oi(self, qty: float) -> None:
        self.open_interest = max(self.open_interest - qty, 0.0)

    def scale_oi(self, factor: float) -> None:
        self.open_interest *= factor
