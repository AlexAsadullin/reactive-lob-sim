"""Liquidation-cascade agent conditioned on funding and open interest.

The agent holds a rolling book of leveraged positions. At every refresh it opens one long and one short at
the current mid. Their sizes split ``oi_fraction_at_risk * OI / max_positions`` between longs and shorts by
the crowding share ``0.5 + crowding_sensitivity * tanh(funding / funding_scale)``, and their leverages are
drawn from ``leverages``. A long at entry ``p`` with leverage ``L`` is liquidated when the mid falls to
``p (1 - 1/L + maintenance_margin)``; a short when it rises to ``p (1 + 1/L - maintenance_margin)``.
Liquidations are market orders, so they move the mid and can trigger the next layer of liquidations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from reactive_lob_sim.engine.types import Side
from reactive_lob_sim.params import LiquidationParams


@dataclass(slots=True)
class Position:
    is_long: bool
    entry: float
    leverage: float
    size: float
    trigger: float


class LiquidationBook:
    def __init__(self, params: LiquidationParams, rng: np.random.Generator) -> None:
        self.p = params
        self._rng = rng
        self.positions: list[Position] = []
        self.next_refresh = 0.0

    def long_share(self, funding: float, override: float | None = None) -> float:
        if override is not None:
            return override
        return 0.5 + self.p.crowding_sensitivity * math.tanh(funding / self.p.funding_scale)

    def _trigger(self, is_long: bool, entry: float, leverage: float) -> float:
        cushion = 1.0 / leverage - self.p.maintenance_margin
        return entry * (1.0 - cushion) if is_long else entry * (1.0 + cushion)

    def refresh(
        self, mid: float, funding: float, open_interest: float, long_share: float | None = None
    ) -> None:
        """Open a new long/short pair at ``mid`` and drop the oldest positions beyond ``max_positions``."""
        share = self.long_share(funding, long_share)
        unit = self.p.oi_fraction_at_risk * open_interest / self.p.max_positions
        for is_long, weight in ((True, share), (False, 1.0 - share)):
            leverage = float(self._rng.choice(self.p.leverages))
            size = 2.0 * unit * weight
            if size > 0:
                self.positions.append(
                    Position(is_long, mid, leverage, size, self._trigger(is_long, mid, leverage))
                )
        excess = len(self.positions) - self.p.max_positions
        if excess > 0:
            del self.positions[:excess]

    def add_crowding(self, mid: float, size: float, is_long: bool) -> None:
        """Add a crowded position near the current price (used by the liquidation-cascade scenario)."""
        leverage = float(max(self.p.leverages))
        self.positions.append(Position(is_long, mid, leverage, size, self._trigger(is_long, mid, leverage)))

    def triggered(self, mid: float) -> list[tuple[Side, float]]:
        """Remove and return the forced orders ``(side, size)`` whose trigger the mid has crossed."""
        forced: list[tuple[Side, float]] = []
        keep: list[Position] = []
        for pos in self.positions:
            hit = mid <= pos.trigger if pos.is_long else mid >= pos.trigger
            if hit:
                forced.append((Side.SELL if pos.is_long else Side.BUY, pos.size))
            else:
                keep.append(pos)
        self.positions = keep
        return forced
