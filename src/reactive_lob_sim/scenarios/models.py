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

Timing. ``time_scale`` (default 1) and ``time_offset_s`` (default 0) are optional document fields that stretch
and shift every shock: a shock runs from ``time_offset_s + time_scale * start_s`` for ``time_scale *
duration_s`` seconds, the same mapping the Shadow Exchange (I6) applies from its own configuration. They let
a consumer request longer shocks (for example ``time_scale: 144`` turns a 600 s scenario into a 24 h one)
without rewriting the shock list. :meth:`Scenario.to_document` omits them when they hold their defaults, so
files with default timing keep exactly the key set that existing consumers parse.
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


# Keys that every seeded document carries; optional timing keys are written only when not at default.
DOCUMENT_KEYS: frozenset[str] = frozenset(
    {"schema_version", "id", "description", "default_duration_s", "shocks", "abm_overrides"}
)
TIMING_DEFAULTS: dict[str, float] = {"time_scale": 1.0, "time_offset_s": 0.0}


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str
    default_duration_s: float = Field(gt=0)
    shocks: list[Shock] = Field(default_factory=list)
    abm_overrides: dict[str, Any] = Field(default_factory=dict)
    time_scale: float = Field(default=TIMING_DEFAULTS["time_scale"], gt=0)
    time_offset_s: float = Field(default=TIMING_DEFAULTS["time_offset_s"], ge=0)

    def scaled_shocks(self) -> list[Shock]:
        """Shocks with the timing mapping applied (identical to ``shocks`` at default timing)."""
        return [
            s.model_copy(
                update={
                    "start_s": self.time_offset_s + self.time_scale * s.start_s,
                    "duration_s": self.time_scale * s.duration_s,
                }
            )
            for s in self.shocks
        ]

    @property
    def effective_duration_s(self) -> float:
        """``default_duration_s`` under the timing mapping."""
        return self.time_offset_s + self.time_scale * self.default_duration_s

    def to_document(self) -> dict[str, Any]:
        """The YAML document: every field, except timing fields that hold their defaults."""
        doc: dict[str, Any] = self.model_dump(mode="json")
        for key, default in TIMING_DEFAULTS.items():
            if doc[key] == default:
                del doc[key]
        return doc


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
