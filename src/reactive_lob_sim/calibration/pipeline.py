"""End-to-end calibration: tape -> events -> flow, impact and coupling -> calibration files (contract 6.5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from reactive_lob_sim.calibration.coupling import CouplingFit, calibrate_coupling
from reactive_lob_sim.calibration.events import extract_events
from reactive_lob_sim.calibration.flow import FlowFit, fit_flow, zero_intelligence
from reactive_lob_sim.contracts.calibration_file import impact_document, write_calibration
from reactive_lob_sim.contracts.pit_client import ConditioningSource
from reactive_lob_sim.engine.l3market import L3Market
from reactive_lob_sim.impact.fit import ImpactFit, fit_impact
from reactive_lob_sim.params import ABMParams
from reactive_lob_sim.settings import CalibrationSettings
from reactive_lob_sim.simctx import SimContext
from reactive_lob_sim.tape.l2 import book_series, trade_series
from reactive_lob_sim.tape.tape import Tape
from reactive_lob_sim.timeutil import iso, utc_now

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CalibrationResult:
    abm: ABMParams
    zero_intelligence: ABMParams
    flow: FlowFit
    impact: ImpactFit
    coupling: CouplingFit
    sample_start_ns: int
    sample_end_ns: int


def calibrate(
    tape: Tape,
    prior: ABMParams,
    cfg: CalibrationSettings,
    ctx: SimContext,
    conditioning: ConditioningSource,
    seed: int,
) -> CalibrationResult:
    start_ns, end_ns = tape.time_range()
    book = book_series(tape, ctx.tick_size, cfg.flow.depth_levels)
    trades = trade_series(tape)
    events = extract_events(tape, ctx.tick_size, ctx.qty_tolerance)
    cond = conditioning.load(start_ns, end_ns + 1, end_ns)
    flow = fit_flow(prior, events, book, cond, ctx.tick_size, cfg.hawkes, cfg.flow)
    impact = fit_impact(book, trades, cfg.impact)
    base = flow.params.model_copy(update={"impact": impact.params})

    def factory(kappa: float, s: int, horizon: float) -> L3Market:
        params = base.model_copy(update={"impact": base.impact.model_copy(update={"kappa": kappa})})
        return ctx.abm(params, horizon, s)

    coupling = calibrate_coupling(factory, impact.params, cfg.impact, seed)
    abm = base.model_copy(update={"impact": base.impact.model_copy(update={"kappa": coupling.kappa})})
    abm = ABMParams.model_validate(abm.model_dump())
    log.info("calibrated", venue=ctx.venue, symbol=ctx.symbol, kappa=coupling.kappa, n_events=flow.n_events)
    return CalibrationResult(abm, zero_intelligence(abm), flow, impact, coupling, start_ns, end_ns)


def save_calibration(result: CalibrationResult, calibration_root: Path, venue: str, symbol: str) -> Path:
    doc = impact_document(
        result.abm.impact,
        iso(int(utc_now().timestamp() * 1e9)),
        iso(result.sample_start_ns),
        iso(result.sample_end_ns),
    )
    return write_calibration(calibration_root, venue, symbol, doc, result.abm)
