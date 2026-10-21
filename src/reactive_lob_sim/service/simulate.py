"""Scenario simulation used by the sim-service and the CLI (contract 6.5)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import structlog

from reactive_lob_sim.contracts.calibration_file import read_abm
from reactive_lob_sim.params import ABMParams
from reactive_lob_sim.scenarios.library import apply_overrides, load_scenario
from reactive_lob_sim.settings import Settings
from reactive_lob_sim.simctx import SimContext
from reactive_lob_sim.tape.io import write_tape
from reactive_lob_sim.timeutil import iso

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SimulationResult:
    simulation_id: str
    tape_path: Path
    calibrated: bool
    n_trades: int
    n_liquidations: int


def market_params(settings: Settings, venue: str, symbol: str) -> tuple[ABMParams, bool]:
    """Calibrated parameters for ``venue/symbol`` if a calibration file exists, else the settings defaults."""
    calibrated = read_abm(settings.calibration_dir, venue, symbol)
    return (calibrated, True) if calibrated is not None else (settings.abm, False)


def run_simulation(
    settings: Settings, scenario_id: str, seed: int, duration_s: float, venue: str, symbol: str
) -> SimulationResult:
    if duration_s > settings.service.max_duration_s:
        raise ValueError(f"duration_s exceeds the limit of {settings.service.max_duration_s} s")
    scenario = load_scenario(settings.scenarios_dir, scenario_id)
    params, calibrated = market_params(settings, venue, symbol)
    params = apply_overrides(params, scenario.abm_overrides)
    simulation_id = uuid.uuid4().hex
    ctx = SimContext.from_settings(settings).with_symbol(venue, symbol)
    rec = ctx.recorder(simulation_id)
    market = ctx.abm(params, duration_s, seed, rec, scenario)
    market.advance(duration_s)
    tape = rec.to_tape()
    out = settings.simulations_dir / simulation_id
    tape_root = out / "tape"
    write_tape(tape, tape_root)
    meta = {
        "schema_version": "1",
        "simulation_id": simulation_id,
        "scenario": scenario_id,
        "seed": seed,
        "duration_s": duration_s,
        "venue": venue,
        "symbol": symbol,
        "calibrated": calibrated,
        "start": iso(ctx.epoch_ns),
        "n_trades": tape["trades"].num_rows,
        "n_liquidations": market.n_liquidations,
        "tape_path": str(tape_root),
    }
    (out / "simulation.json").write_text(json.dumps(meta, indent=2))
    log.info("simulation_done", **meta)
    return SimulationResult(
        simulation_id, tape_root, calibrated, tape["trades"].num_rows, market.n_liquidations
    )
