"""``make reproduce``: the offline end-to-end study.

1. A hidden ground-truth market (the ABM with the ``reproduce.ground_truth_overrides``) writes a calibration
   tape and, for every held-out episode, a recorded "real" tape without the experimental agent.
2. In each episode every strategy also runs inside the ground-truth market itself (same seed, so the same
   background clock and index path). Those fills are the "live" results; they are exported in the
   contract 6.4 fills format and read back through the same reader that consumes I6 fills.
3. Simulators are calibrated on the calibration tape only: zero-intelligence (Poisson flow), the Hawkes ABM
   with calibrated impact coupling, and the generative GRU model. Replay uses the held-out episode tape.
4. Each simulator runs the same strategies; outcomes are scored against the live outcomes (gap score) and
   the simulators' own tapes against the real episode tapes (stylised facts and distances).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from reactive_lob_sim.calibration.events import extract_events
from reactive_lob_sim.calibration.pipeline import CalibrationResult, calibrate, save_calibration
from reactive_lob_sim.contracts.fills import FillContext, fills_table, read_fills, write_fills
from reactive_lob_sim.contracts.pit_client import TapeConditioningSource
from reactive_lob_sim.gap.outcomes import Outcome, outcome_from_run, outcomes_from_fills
from reactive_lob_sim.gap.score import SimulatorScore, gap_scores
from reactive_lob_sim.generative.market import GenerativeMarket
from reactive_lob_sim.generative.model import OrderFlowGRU, TrainReport, save_model, train_model
from reactive_lob_sim.generative.tokenizer import Tokenizer, TokenStream, fit_tokenizer, tokenize
from reactive_lob_sim.params import ABMParams
from reactive_lob_sim.realism.distances import RealismScore, score_realism
from reactive_lob_sim.realism.facts import StylisedFacts, stylised_facts
from reactive_lob_sim.scenarios.library import apply_overrides, seed_library
from reactive_lob_sim.settings import Settings
from reactive_lob_sim.simctx import SimContext
from reactive_lob_sim.strategies.base import run_strategy
from reactive_lob_sim.strategies.simple import build_strategy
from reactive_lob_sim.tape.io import write_tape
from reactive_lob_sim.tape.tape import Tape
from reactive_lob_sim.timeutil import NS_PER_S
from reactive_lob_sim.venue import MarketBackend

log = structlog.get_logger(__name__)

SIMULATORS = ("replay", "zero_intelligence", "hawkes_abm", "generative")
# Seed offsets keep the simulators' random streams disjoint from the ground truth's.
_SIM_SEED_OFFSET = 1_000
_EPISODE_SEED_STRIDE = 10


@dataclass
class Generative:
    model: OrderFlowGRU
    tok: Tokenizer
    prime: TokenStream
    report: TrainReport


@dataclass
class StudyResult:
    calibration: CalibrationResult
    generative: TrainReport
    scores: list[SimulatorScore]
    realism: dict[str, list[RealismScore]]
    real_facts: list[StylisedFacts]
    sim_facts: dict[str, list[StylisedFacts]]
    live: dict[tuple[str, str], Outcome]
    predictions: dict[str, dict[tuple[str, str], list[Outcome]]]
    out_dir: Path


def ground_truth_params(settings: Settings) -> ABMParams:
    return apply_overrides(settings.abm, settings.reproduce.ground_truth_overrides)


def train_generative(tape: Tape, ctx: SimContext, settings: Settings, seed: int) -> Generative:
    events = extract_events(tape, ctx.tick_size, ctx.qty_tolerance)
    start, end = tape.time_range()
    cond = TapeConditioningSource(tape).load(start, end + 1, end)
    tok = fit_tokenizer(events, cond, ctx.tick_size, settings.generative)
    stream = tokenize(tok, events, cond, ctx.tick_size)
    model, report = train_model(tok, stream, settings.generative, seed)
    k = settings.generative.warmup_events
    prime = TokenStream(
        stream.kind[-k:], stream.offset[-k:], stream.size[-k:], stream.dt[-k:], stream.cond[-k:]
    )
    return Generative(model, tok, prime, report)


def generative_market(
    gen: Generative,
    params: ABMParams,
    horizon: float,
    seed: int,
    ctx: SimContext,
    settings: Settings,
    record: bool,
) -> GenerativeMarket:
    rec = ctx.recorder(f"gen-{seed}") if record else None
    return GenerativeMarket(
        gen.model,
        gen.tok,
        gen.prime,
        params,
        horizon,
        seed,
        ctx.tick_size,
        ctx.lot_size,
        ctx.qty_tolerance,
        ctx.fees,
        settings.generative.temperature,
        rec,
        ctx.snapshot_depth,
    )


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {
            (
                k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple) else str(k)
            ): _jsonable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return _jsonable(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_study(settings: Settings, out_dir: Path | None = None) -> StudyResult:
    rp = settings.reproduce
    ctx = SimContext.from_settings(settings)
    out = out_dir or settings.reproduce_dir
    out.mkdir(parents=True, exist_ok=True)
    seed_library(settings.scenarios_dir)
    truth = ground_truth_params(settings)
    dataset_root = out / "dataset" / "tape"
    live_root = out / "live" / "fills"
    for stale in [*dataset_root.rglob("*.parquet"), *live_root.rglob("*.parquet")]:
        stale.unlink()

    # 1. ground-truth calibration tape
    rec = ctx.recorder("truth-calibration")
    gt = ctx.abm(truth, rp.calibration_duration_s, rp.seed, rec)
    gt.advance(rp.calibration_duration_s)
    cal_tape = rec.to_tape()
    write_tape(cal_tape, dataset_root)
    log.info("ground_truth_calibration_tape", trades=cal_tape["trades"].num_rows)

    # 2. calibration (simulators only ever see cal_tape)
    calib = calibrate(
        cal_tape, settings.abm, settings.calibration, ctx, TapeConditioningSource(cal_tape), rp.seed
    )
    save_calibration(calib, settings.calibration_dir, ctx.venue, ctx.symbol)
    gen = train_generative(cal_tape, ctx, settings, rp.seed)
    save_model(gen.model, gen.tok, settings.models_dir / f"venue={ctx.venue}" / f"symbol={ctx.symbol}")

    horizon = rp.episode_duration_s
    run_len = horizon - settings.runner.decision_interval_s
    specs = settings.strategies
    live_final: dict[tuple[str, str], float] = {}
    live_submitted: dict[tuple[str, str], float] = {}
    predictions: dict[str, dict[tuple[str, str], list[Outcome]]] = {s: {} for s in SIMULATORS}
    real_facts: list[StylisedFacts] = []
    sim_facts: dict[str, list[StylisedFacts]] = {s: [] for s in SIMULATORS if s != "replay"}
    realism: dict[str, list[RealismScore]] = {s: [] for s in SIMULATORS if s != "replay"}

    for e in range(rp.n_episodes):
        episode = f"episode-{e}"
        truth_seed = rp.seed + 1 + e
        epoch = ctx.epoch_ns + round((rp.calibration_duration_s + e * horizon) * NS_PER_S)
        rec = ctx.recorder(f"truth-{episode}", epoch)
        real = ctx.abm(truth, horizon, truth_seed, rec)
        real.advance(horizon)
        real_tape = rec.to_tape()
        write_tape(real_tape, dataset_root)
        facts_real = stylised_facts(real_tape, ctx.tick_size, settings.realism)
        real_facts.append(facts_real)

        for spec in specs:
            unit = (spec.id, episode)
            live_run = run_strategy(
                ctx.abm(truth, horizon, truth_seed),
                build_strategy(spec),
                run_len,
                settings.runner.decision_interval_s,
            )
            ctx_f = FillContext(episode, spec.id, ctx.venue, ctx.symbol, epoch)
            write_fills(fills_table(live_run.fills, ctx_f), live_root, f"{episode}-{spec.id}")
            live_final[unit] = live_run.final_mid
            live_submitted[unit] = live_run.submitted_qty

        def sim_seed(k: int, e: int = e) -> int:
            return rp.seed + _SIM_SEED_OFFSET + _EPISODE_SEED_STRIDE * e + k

        def replay_factory(k: int, rec_: bool, tape: Tape = real_tape) -> MarketBackend:
            return ctx.replay(tape)

        factories: dict[str, tuple[int, Callable[[int, bool], MarketBackend]]] = {
            "replay": (1, replay_factory),
            "zero_intelligence": (
                rp.abm_seeds,
                lambda k, rec_: ctx.abm(
                    calib.zero_intelligence, horizon, sim_seed(k), ctx.recorder("zi") if rec_ else None
                ),
            ),
            "hawkes_abm": (
                rp.abm_seeds,
                lambda k, rec_: ctx.abm(
                    calib.abm, horizon, sim_seed(k), ctx.recorder("abm") if rec_ else None
                ),
            ),
            "generative": (
                rp.generative_seeds,
                lambda k, rec_: generative_market(gen, calib.abm, horizon, sim_seed(k), ctx, settings, rec_),
            ),
        }
        for sim, (n_seeds, make) in factories.items():
            if sim != "replay":
                background = make(0, True)
                background.advance(horizon)
                tape = background.tape()
                assert tape is not None
                facts = stylised_facts(tape, ctx.tick_size, settings.realism)
                sim_facts[sim].append(facts)
                realism[sim].append(score_realism(facts_real, facts))
            for spec in specs:
                runs = [
                    outcome_from_run(
                        run_strategy(
                            make(k, False), build_strategy(spec), run_len, settings.runner.decision_interval_s
                        )
                    )
                    for k in range(n_seeds)
                ]
                predictions[sim][(spec.id, episode)] = runs
        log.info("episode_done", episode=episode)

    # Live units are keyed (strategy_id, account_id), and the account id is the episode id.
    live_units = outcomes_from_fills(read_fills(live_root), live_final, live_submitted)
    g = settings.gap
    scores = gap_scores(
        {
            s: {u: [asdict(o) for o in runs] for u, runs in by_unit.items()}
            for s, by_unit in predictions.items()
        },
        {u: asdict(o) for u, o in live_units.items()},
        g.metrics,
        g.bootstrap_samples,
        g.ci_level,
        g.weight_rank,
        g.weight_error,
        g.seed,
    )
    result = StudyResult(
        calib, gen.report, scores, realism, real_facts, sim_facts, live_units, predictions, out
    )
    write_study(result, settings)
    return result


def realism_table(result: StudyResult) -> dict[str, dict[str, float]]:
    """Mean and standard error over episodes of the realism scores per simulator."""
    out: dict[str, dict[str, float]] = {}
    for sim, scores in result.realism.items():
        w = np.asarray([s.mean_wasserstein_norm for s in scores])
        ks = np.asarray([s.mean_ks for s in scores])
        resp = np.asarray([s.response_error for s in scores])
        n = len(scores)
        out[sim] = {
            "wasserstein_norm_mean": float(np.nanmean(w)),
            "wasserstein_norm_se": float(np.nanstd(w, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "ks_mean": float(np.nanmean(ks)),
            "ks_se": float(np.nanstd(ks, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "response_error_mean": float(np.nanmean(resp)),
            "response_error_se": float(np.nanstd(resp, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "n_episodes": float(n),
        }
    return out


def facts_table(result: StudyResult) -> dict[str, dict[str, tuple[float, float]]]:
    """Per source (real and each simulator), mean and standard error of each stylised fact over episodes."""
    sources: dict[str, list[StylisedFacts]] = {"real": result.real_facts, **result.sim_facts}
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for name, facts in sources.items():
        rows = [f.summary() for f in facts]
        keys = rows[0].keys()
        out[name] = {}
        for k in keys:
            v = np.asarray([r[k] for r in rows], dtype=np.float64)
            se = float(np.nanstd(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
            out[name][k] = (float(np.nanmean(v)), se)
    return out


def write_study(result: StudyResult, settings: Settings) -> None:
    out = result.out_dir
    cal = result.calibration
    doc = {
        "schema_version": "1",
        "calibration": {
            "hawkes": cal.flow.hawkes.params.model_dump(),
            "hawkes_loglik": cal.flow.hawkes.loglik,
            "hawkes_spectral_radius": cal.flow.hawkes.params.spectral_radius(),
            "impact": cal.abm.impact.model_dump(),
            "sqrt_law_r2": cal.impact.sqrt_r2,
            "coupling": asdict(cal.coupling),
            "n_events": cal.flow.n_events,
        },
        "ground_truth_impact": ground_truth_params(settings).impact.model_dump(),
        "generative": asdict(result.generative),
        "gap_scores": [asdict(s) for s in result.scores],
        "realism": realism_table(result),
        "facts": facts_table(result),
        "realism_detail": {s: [asdict(r) for r in rs] for s, rs in result.realism.items()},
        "live": {f"{k[0]}|{k[1]}": asdict(v) for k, v in result.live.items()},
        "predictions": {
            s: {f"{k[0]}|{k[1]}": [asdict(o) for o in v] for k, v in p.items()}
            for s, p in result.predictions.items()
        },
    }
    (out / "report.json").write_text(json.dumps(_jsonable(doc), indent=2))
    (out / "summary.md").write_text(summary_markdown(result))


def _fmt(x: float, digits: int = 3) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.{digits}f}"


def summary_markdown(result: StudyResult) -> str:
    lines = [
        "# Sim-to-real gap score (synthetic ground truth)",
        "",
        "| rank | simulator | gap score G (95% CI) | P(best) | rho PnL | rho fill rate | rho slippage |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in result.scores:
        rho = {m: _fmt(ms.rho.value, 2) for m, ms in s.metrics.items()}
        gap = f"{_fmt(s.gap.value)} ({_fmt(s.gap.low)} to {_fmt(s.gap.high)})"
        rhos = " | ".join(rho.get(m, "-") for m in ("pnl", "fill_rate", "slippage_bps"))
        lines.append(f"| {s.rank} | {s.simulator} | {gap} | {_fmt(s.p_best, 2)} | {rhos} |")
    lines += ["", "| simulator | mean normalised W1 | mean KS | impact response error |", "|---|---|---|---|"]
    for sim, row in realism_table(result).items():
        cells = [
            f"{_fmt(row[f'{k}_mean'])} (se {_fmt(row[f'{k}_se'])})"
            for k in ("wasserstein_norm", "ks", "response_error")
        ]
        lines.append(f"| {sim} | " + " | ".join(cells) + " |")
    cal = result.calibration
    radius = cal.flow.hawkes.params.spectral_radius()
    lines += [
        "",
        f"Calibrated coupling kappa = {_fmt(cal.coupling.kappa)}; "
        f"square-root law Y = {_fmt(cal.abm.impact.Y)}, delta = {_fmt(cal.abm.impact.delta)}; "
        f"Hawkes spectral radius = {_fmt(radius)}.",
        "",
    ]
    return "\n".join(lines)
