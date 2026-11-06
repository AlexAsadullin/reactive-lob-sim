"""Typed project settings.

Defaults live in ``config/settings.yaml``; environment variables with the ``SIM_`` prefix override them
(nested delimiter ``__``, e.g. ``SIM_FEES__TAKER_BPS=4.5``), and ``TRADEQ_DATA_DIR`` sets the data root.
``SIM_CONFIG_FILE`` points to an alternative YAML file.
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from reactive_lob_sim.params import ABMParams

PROJECT_NAME = "reactive-lob-sim"
CONFIG_ENV_VAR = "SIM_CONFIG_FILE"


def default_config_file() -> Path:
    """``config/settings.yaml`` of the repository (or ``$SIM_CONFIG_FILE``)."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServiceSettings(_Section):
    host: str
    port: int = Field(ge=1, le=65535)
    max_duration_s: float = Field(gt=0)


class ContractSettings(_Section):
    """Locations of the contracts this project consumes. ``None`` means the platform default under the data
    root."""

    tape_root: Path | None = None
    fills_root: Path | None = None
    pit_base_url: str
    pit_timeout_s: float = Field(gt=0)
    pit_enabled: bool
    pit_funding_dataset: str
    pit_open_interest_dataset: str


class MarketSettings(_Section):
    venue: str
    symbol: str
    tick_size: float = Field(gt=0)
    lot_size: float = Field(gt=0)
    qty_tolerance: float = Field(gt=0)
    epoch: datetime
    capture_region: str
    snapshot_depth: int = Field(ge=1)

    @field_validator("epoch")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("market.epoch must be timezone-aware (UTC)")
        return v


class FeeSettings(_Section):
    maker_bps: float
    taker_bps: float


class InstrumentsSettings(_Section):
    """MOEX instruments; see :mod:`reactive_lob_sim.moex.instruments`. File paths are under the data root."""

    commission_bps: float = Field(ge=0)
    zero_fee_commission_bps: float = Field(ge=0)
    universe_file: str | None = None
    zero_fee_file: str | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    symbols: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SessionWindowSettings(_Section):
    session: str
    phase: str
    start: str
    end: str


class SessionsSettings(_Section):
    """MOEX trading day; ``apply`` makes simulated runs follow it from ``market.epoch``."""

    apply: bool = False
    utc_offset_h: float = 3.0
    weekend_enabled: bool = True
    weekday: list[SessionWindowSettings] = Field(default_factory=list)
    weekend: list[SessionWindowSettings] = Field(default_factory=list)


class ReplaySettings(_Section):
    queue_model: Literal["risk_averse", "proportional"]


class HawkesFitSettings(_Section):
    init_beta: float = Field(gt=0)
    beta_min: float = Field(gt=0)
    beta_max: float = Field(gt=0)
    max_iter: int = Field(ge=1)
    max_spectral_radius: float = Field(gt=0, lt=1)
    tolerance: float = Field(gt=0)
    init_branching: float = Field(gt=0, lt=1)
    param_floor: float = Field(gt=0)


class ImpactFitSettings(_Section):
    window_s: float = Field(gt=0)
    n_bins: int = Field(ge=3)
    lag_bin_s: float = Field(gt=0)
    max_lag_bins: int = Field(ge=2)
    delta_min: float = Field(gt=0)
    delta_max: float = Field(gt=0)
    beta_max: float = Field(gt=0)
    n_exponentials: int = Field(ge=1)
    kappa_grid: list[float] = Field(min_length=1)
    probe_duration_s: float = Field(gt=0)
    probe_volume_fraction: float = Field(gt=0)
    probe_child_interval_s: float = Field(gt=0)
    probe_seeds: int = Field(ge=1)


class FlowFitSettings(_Section):
    depth_levels: int = Field(ge=1)
    vol_interval_s: float = Field(gt=0)
    min_events: int = Field(ge=1)


class CalibrationSettings(_Section):
    hawkes: HawkesFitSettings
    impact: ImpactFitSettings
    flow: FlowFitSettings


class GenerativeSettings(_Section):
    offset_clip_ticks: int = Field(ge=1)
    n_size_bins: int = Field(ge=2)
    n_dt_bins: int = Field(ge=2)
    dt_floor_s: float = Field(gt=0)
    embed_dim: int = Field(ge=1)
    hidden_dim: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    seq_len: int = Field(ge=2)
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    torch_threads: int = Field(ge=1)
    vol_window_s: float = Field(gt=0)
    n_vol_regimes: int = Field(ge=1)
    temperature: float = Field(gt=0)
    warmup_events: int = Field(ge=1)


class RealismSettings(_Section):
    sample_intervals_s: list[float] = Field(min_length=1)
    acf_lags: int = Field(ge=1)
    depth_levels: int = Field(ge=1)
    response_lags_s: list[float] = Field(min_length=1)
    hill_tail_fraction: float = Field(gt=0, lt=0.5)
    ofi_interval_s: float = Field(gt=0)


class GapSettings(_Section):
    metrics: list[Literal["pnl", "fill_rate", "slippage_bps"]] = Field(min_length=1)
    bootstrap_samples: int = Field(ge=10)
    ci_level: float = Field(gt=0, lt=1)
    weight_rank: float = Field(ge=0)
    weight_error: float = Field(ge=0)
    seed: int


class RunnerSettings(_Section):
    decision_interval_s: float = Field(gt=0)


class StrategySpec(_Section):
    id: str
    kind: Literal["market_maker", "momentum", "twap"]
    params: dict[str, float | int | str]


class ReproduceSettings(_Section):
    seed: int
    calibration_duration_s: float = Field(gt=0)
    episode_duration_s: float = Field(gt=0)
    n_episodes: int = Field(ge=2)
    abm_seeds: int = Field(ge=1)
    generative_seeds: int = Field(ge=1)
    ground_truth_overrides: dict[str, Any]


class ReportSettings(_Section):
    kappa_grid: list[float] = Field(min_length=2)
    kappa_seeds: int = Field(ge=2)
    kappa_duration_s: float = Field(gt=0)
    speed_events: list[int] = Field(min_length=1)
    speed_repeats: int = Field(ge=2)
    speed_n_ticks: int = Field(ge=10)
    speed_p_limit: float = Field(gt=0, lt=1)
    speed_p_market: float = Field(gt=0, lt=1)
    speed_offset_geom_p: float = Field(gt=0, le=1)
    speed_size_log_mean: float
    speed_size_log_sd: float = Field(ge=0)
    speed_seed: int


class LoggingSettings(_Section):
    level: str
    json_logs: bool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIM_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Field(validation_alias=AliasChoices("TRADEQ_DATA_DIR", "data_dir"))
    service: ServiceSettings
    contracts: ContractSettings
    market: MarketSettings
    fees: FeeSettings
    instruments: InstrumentsSettings
    sessions: SessionsSettings = Field(default_factory=SessionsSettings)
    replay: ReplaySettings
    abm: ABMParams
    calibration: CalibrationSettings
    generative: GenerativeSettings
    realism: RealismSettings
    gap: GapSettings
    runner: RunnerSettings
    strategies: list[StrategySpec] = Field(min_length=1)
    reproduce: ReproduceSettings
    report: ReportSettings
    logging: LoggingSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=default_config_file())
        return init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings

    # ------------------------------------------------------------------ data layout (section 3, 6)
    @cached_property
    def project_dir(self) -> Path:
        return self.data_dir / PROJECT_NAME

    @property
    def tape_root(self) -> Path:
        """I1 tape root (contract 6.2)."""
        return self.contracts.tape_root or self.data_dir / "market-tape-recorder" / "tape"

    @property
    def fills_root(self) -> Path:
        """I6 fills export (contract 6.4)."""
        return self.contracts.fills_root or self.data_dir / "shadow-exchange" / "fills"

    @property
    def calibration_dir(self) -> Path:
        return self.project_dir / "calibration"

    @property
    def scenarios_dir(self) -> Path:
        return self.project_dir / "scenarios"

    @property
    def simulations_dir(self) -> Path:
        return self.project_dir / "simulations"

    @property
    def reproduce_dir(self) -> Path:
        return self.project_dir / "reproduce"

    @property
    def reports_dir(self) -> Path:
        return self.project_dir / "reports"

    @property
    def models_dir(self) -> Path:
        return self.project_dir / "models"


def load_settings(**overrides: Any) -> Settings:
    """Build settings from YAML, ``.env`` and environment; keyword overrides win."""
    return Settings(**overrides)
