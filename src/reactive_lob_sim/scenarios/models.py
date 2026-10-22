"""Counterfactual stress scenarios (contract 6.5 scenario library).

A scenario is a list of timed shocks applied to the agent-based market. Each shock kind has a fixed set of
parameters:

spread_widening: ``extra_ticks`` added to the minimum quote offset of liquidity providers, and resting
liquidity inside the widened band is pulled at the shock start.
liquidity_withdrawal: ``rate_factor`` (0..1) thins limit-order arrivals, and ``cancel_fraction`` of resting
liquidity is cancelled at the shock start.
liquidation_cascade: ``oi_multiplier`` scales open interest and ``long_share`` forces long crowding; the
crowded longs at risk are placed in ``layers`` layers whose liquidation prices sit ``layer_gap_bps`` apart
below the mid, and a forced sell of ``trigger_depth_fraction`` of the visible bid depth starts the cascade.
funding_shock: the funding rate is held at ``rate`` for the shock duration.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ShockKind = Literal["spread_widening", "liquidity_withdrawal", "liquidation_cascade", "funding_shock"]

REQUIRED_PARAMS: dict[str, frozenset[str]] = {
    "spread_widening": frozenset({"extra_ticks"}),
    "liquidity_withdrawal": frozenset({"rate_factor", "cancel_fraction"}),
    "liquidation_cascade": frozenset(
        {"oi_multiplier", "long_share", "trigger_depth_fraction", "layers", "layer_gap_bps"}
    ),
    "funding_shock": frozenset({"rate"}),
}


class Shock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ShockKind
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    params: dict[str, float]

    @model_validator(mode="after")
    def _check_params(self) -> Self:
        missing = REQUIRED_PARAMS[self.kind] - set(self.params)
        if missing:
            raise ValueError(f"shock {self.kind} is missing params: {sorted(missing)}")
        return self

    def active(self, t: float) -> bool:
        return self.start_s <= t < self.start_s + self.duration_s


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str
    default_duration_s: float = Field(gt=0)
    shocks: list[Shock] = Field(default_factory=list)
    abm_overrides: dict[str, Any] = Field(default_factory=dict)


class ShockSchedule:
    """Time queries over a scenario's shocks, used by the simulators."""

    def __init__(self, shocks: list[Shock]) -> None:
        self._shocks = sorted(shocks, key=lambda s: s.start_s)
        self._next = 0

    def due(self, t: float) -> list[Shock]:
        """Shocks whose start time has been reached since the last call (each is returned once)."""
        out: list[Shock] = []
        while self._next < len(self._shocks) and self._shocks[self._next].start_s <= t:
            out.append(self._shocks[self._next])
            self._next += 1
        return out

    def _active(self, kind: str, t: float) -> list[Shock]:
        return [s for s in self._shocks if s.kind == kind and s.active(t)]

    def extra_offset_ticks(self, t: float) -> int:
        return int(sum(s.params["extra_ticks"] for s in self._active("spread_widening", t)))

    def limit_rate_factor(self, t: float) -> float:
        factor = 1.0
        for s in self._active("liquidity_withdrawal", t):
            factor *= s.params["rate_factor"]
        return factor

    def funding_override(self, t: float) -> float | None:
        active = self._active("funding_shock", t)
        return active[-1].params["rate"] if active else None

    def long_share_override(self, t: float) -> float | None:
        active = self._active("liquidation_cascade", t)
        return active[-1].params["long_share"] if active else None
