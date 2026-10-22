"""Experiments behind docs/report.md (``make report``).

E1 simulator comparison: the ``reproduce`` study (gap score and realism of replay, zero-intelligence,
Hawkes ABM and generative simulators against a hidden ground truth).
E2 impact coupling and fat tails: the calibrated ABM run with the coupling kappa on a grid, several seeds
per value; excess kurtosis and Hill index of mid returns with t-based 95% intervals across seeds.
E3 matching-engine speed: the same zero-intelligence order stream through the Numba batch kernel and the
pure-Python object book, repeated; events per second with 95% intervals, plus an equivalence check.
All tables go to ``$TRADEQ_DATA_DIR/reactive-lob-sim/reports/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from reactive_lob_sim.engine.fastbook import run_events, run_events_reference, zero_intelligence_stream
from reactive_lob_sim.realism.facts import stylised_facts
from reactive_lob_sim.reproduce import StudyResult, _jsonable, facts_table, realism_table, run_study
from reactive_lob_sim.settings import Settings
from reactive_lob_sim.simctx import SimContext


def mean_ci(values: np.ndarray, level: float) -> tuple[float, float, int]:
    """Mean and the half-width of the Student-t confidence interval."""
    v = values[np.isfinite(values)]
    if v.size < 2:
        return float(np.mean(v)) if v.size else float("nan"), float("nan"), int(v.size)
    half = float(stats.t.ppf(0.5 + level / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return float(v.mean()), half, int(v.size)


def kappa_sweep(settings: Settings, study: StudyResult) -> dict[str, object]:
    rp = settings.report
    ctx = SimContext.from_settings(settings)
    base = study.calibration.abm
    level = settings.gap.ci_level
    rows = []
    raw: dict[float, dict[str, list[float]]] = {}
    for kappa in rp.kappa_grid:
        params = base.model_copy(update={"impact": base.impact.model_copy(update={"kappa": kappa})})
        acc: dict[str, list[float]] = {}
        for s in range(rp.kappa_seeds):
            rec = ctx.recorder(f"kappa-{kappa}-{s}")
            market = ctx.abm(params, rp.kappa_duration_s, settings.reproduce.seed + 5_000 + s, rec)
            market.advance(rp.kappa_duration_s)
            tape = rec.to_tape()
            facts = stylised_facts(tape, ctx.tick_size, settings.realism)
            for k, v in facts.summary().items():
                acc.setdefault(k, []).append(v)
        raw[kappa] = acc
        row: dict[str, float] = {"kappa": kappa}
        for k in ("kurtosis_1s", "kurtosis_10s", "hill_1s", "abs_acf_mean_1s", "acf1_1s"):
            m, h, n = mean_ci(np.asarray(acc[k]), level)
            row[k], row[f"{k}_ci"], row["n_seeds"] = m, h, n
        rows.append(row)
    # Monotone effect: Spearman correlation between kappa and per-seed kurtosis; and a Welch t-test of the
    # largest kappa against kappa = 0.
    tests: dict[str, dict[str, float]] = {}
    for key in ("kurtosis_1s", "kurtosis_10s"):
        xs = [k for k in rp.kappa_grid for _ in raw[k][key]]
        ys = [v for k in rp.kappa_grid for v in raw[k][key]]
        sp = stats.spearmanr(xs, ys)
        welch = stats.ttest_ind(raw[rp.kappa_grid[-1]][key], raw[rp.kappa_grid[0]][key], equal_var=False)
        tests[key] = {
            "spearman_rho": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "n": float(len(xs)),
            "welch_t": float(welch.statistic),
            "welch_p": float(welch.pvalue),
            "welch_df": float(welch.df),
        }
    return {
        "rows": rows,
        "tests": tests,
        "calibrated_kappa": study.calibration.coupling.kappa,
    }


def engine_speed(settings: Settings) -> dict[str, object]:
    rp = settings.report
    m = settings.market
    rows = []
    level = settings.gap.ci_level
    warm = zero_intelligence_stream(
        100,
        rp.speed_n_ticks,
        rp.speed_p_limit,
        rp.speed_p_market,
        rp.speed_offset_geom_p,
        rp.speed_size_log_mean,
        rp.speed_size_log_sd,
        m.lot_size,
        rp.speed_seed,
    )
    run_events(warm, rp.speed_n_ticks, m.qty_tolerance)  # JIT compilation is excluded from the timings
    for n in rp.speed_events:
        batch = zero_intelligence_stream(
            n,
            rp.speed_n_ticks,
            rp.speed_p_limit,
            rp.speed_p_market,
            rp.speed_offset_geom_p,
            rp.speed_size_log_mean,
            rp.speed_size_log_sd,
            m.lot_size,
            rp.speed_seed,
        )
        fast, slow = [], []
        for _ in range(rp.speed_repeats):
            t0 = time.perf_counter()
            res_fast = run_events(batch, rp.speed_n_ticks, m.qty_tolerance)
            fast.append(n / (time.perf_counter() - t0))
            t0 = time.perf_counter()
            res_slow = run_events_reference(batch, rp.speed_n_ticks, m.qty_tolerance)
            slow.append(n / (time.perf_counter() - t0))
        f_m, f_h, _ = mean_ci(np.asarray(fast), level)
        s_m, s_h, _ = mean_ci(np.asarray(slow), level)
        diff = max(
            float(np.max(np.abs(res_fast.filled - res_slow.filled))),
            float(np.max(np.abs(res_fast.bid_qty - res_slow.bid_qty))),
            float(np.max(np.abs(res_fast.ask_qty - res_slow.ask_qty))),
        )
        rows.append(
            {
                "n_events": n,
                "numba_events_per_s": f_m,
                "numba_ci": f_h,
                "python_events_per_s": s_m,
                "python_ci": s_h,
                "speedup": f_m / s_m,
                "max_abs_difference": diff,
                "repeats": rp.speed_repeats,
            }
        )
    ctx = SimContext.from_settings(settings)
    abm_rates = []
    for s in range(rp.speed_repeats):
        market = ctx.abm(settings.abm, rp.kappa_duration_s, rp.speed_seed + s)
        t0 = time.perf_counter()
        market.advance(rp.kappa_duration_s)
        abm_rates.append(market.n_scheduled_events / (time.perf_counter() - t0))
    a_m, a_h, _ = mean_ci(np.asarray(abm_rates), level)
    return {"rows": rows, "abm_interactive_events_per_s": a_m, "abm_interactive_ci": a_h}


def _f(x: float, d: int = 3) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _interval(value: float, low: float, high: float, d: int = 2) -> str:
    return f"{_f(value, d)} ({_f(low, d)} to {_f(high, d)})"


def tables_markdown(study: StudyResult, sweep: dict[str, object], speed: dict[str, object]) -> str:
    lines = ["# Report tables (generated by make report)", "", "## E1 gap score", ""]
    lines.append(
        "| rank | simulator | G | 95% CI | P(best) | rho pnl (CI) | rho fill rate (CI) "
        "| rho slippage (CI) | NMAE pnl (CI) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in study.scores:
        rhos = [s.metrics[m].rho for m in ("pnl", "fill_rate", "slippage_bps")]
        e = s.metrics["pnl"].nmae
        cells = [
            str(s.rank),
            s.simulator,
            _f(s.gap.value),
            f"{_f(s.gap.low)} to {_f(s.gap.high)}",
            _f(s.p_best, 2),
            *(_interval(x.value, x.low, x.high) for x in rhos),
            _interval(e.value, e.low, e.high),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    n_units = study.scores[0].metrics["pnl"].n_units
    lines += [
        "",
        f"Units: {n_units} (strategy, episode) pairs.",
        "",
        "## E1 realism (mean over episodes, standard error)",
        "",
    ]
    lines.append("| simulator | mean normalised W1 | se | mean KS | se | response error | se |")
    lines.append("|---|---|---|---|---|---|---|")
    realism_keys = ("wasserstein_norm", "ks", "response_error")
    for sim, row in realism_table(study).items():
        cells = [f"{_f(row[f'{k}_mean'])} | {_f(row[f'{k}_se'])}" for k in realism_keys]
        lines.append(f"| {sim} | " + " | ".join(cells) + " |")
    lines += ["", "## E1 stylised facts (mean over episodes, standard error)", ""]
    ft = facts_table(study)
    keys = [
        "kurtosis_1s",
        "kurtosis_10s",
        "hill_1s",
        "acf1_1s",
        "abs_acf_mean_1s",
        "spread_mean_ticks",
        "depth_mean",
        "ofi_r2",
    ]
    lines.append("| source | " + " | ".join(keys) + " |")
    lines.append("|---" * (len(keys) + 1) + "|")
    for src, vals in ft.items():
        lines.append(
            f"| {src} | " + " | ".join(f"{_f(vals[k][0], 2)} ({_f(vals[k][1], 2)})" for k in keys) + " |"
        )
    lines += ["", "## E2 coupling kappa and fat tails (mean, 95% CI half-width)", ""]
    lines.append(
        "| kappa | kurtosis 1s | kurtosis 10s | Hill 1s | abs-return ACF 1s | return ACF lag 1 | seeds |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    sweep_keys = (
        ("kurtosis_1s", 2),
        ("kurtosis_10s", 2),
        ("hill_1s", 2),
        ("abs_acf_mean_1s", 3),
        ("acf1_1s", 3),
    )
    for row in sweep["rows"]:  # type: ignore[attr-defined]
        cells = [f"{_f(row[k], d)} ({_f(row[f'{k}_ci'], d)})" for k, d in sweep_keys]
        lines.append(f"| {row['kappa']:g} | " + " | ".join(cells) + f" | {row['n_seeds']} |")
    lines.append("")
    for key, t in sweep["tests"].items():  # type: ignore[attr-defined]
        lines.append(
            f"{key}: Spearman rho = {_f(t['spearman_rho'])} "
            f"(p = {_f(t['spearman_p'], 4)}, n = {int(t['n'])}); "
            f"Welch t = {_f(t['welch_t'])} (df = {_f(t['welch_df'], 1)}, "
            f"p = {_f(t['welch_p'], 4)}), largest kappa vs 0."
        )
    lines += [
        "",
        f"Calibrated kappa: {_f(sweep['calibrated_kappa'])}.",  # type: ignore[arg-type]
        "",
        "## E3 matching-engine speed (events per second, 95% CI half-width)",
        "",
        "| events | Numba kernel | CI | pure-Python book | CI | speed-up | max abs difference | repeats |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in speed["rows"]:  # type: ignore[attr-defined]
        lines.append(
            f"| {row['n_events']} | {row['numba_events_per_s']:.0f} | {row['numba_ci']:.0f} "
            f"| {row['python_events_per_s']:.0f} | {row['python_ci']:.0f} | {row['speedup']:.1f} "
            f"| {row['max_abs_difference']:.2e} | {row['repeats']} |"
        )
    abm_rate = float(speed["abm_interactive_events_per_s"])  # type: ignore[arg-type]
    abm_ci = float(speed["abm_interactive_ci"])  # type: ignore[arg-type]
    lines += [
        "",
        f"Interactive ABM (Python agents on the object book): {abm_rate:.0f} events/s (CI {abm_ci:.0f}).",
        "",
    ]
    return "\n".join(lines)


def run_report(settings: Settings) -> Path:
    out = settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    study = run_study(settings)
    sweep = kappa_sweep(settings, study)
    speed = engine_speed(settings)
    (out / "kappa_sweep.json").write_text(json.dumps(_jsonable(sweep), indent=2))
    (out / "engine_speed.json").write_text(json.dumps(_jsonable(speed), indent=2))
    (out / "study.json").write_text((study.out_dir / "report.json").read_text())
    (out / "tables.md").write_text(tables_markdown(study, sweep, speed))
    return out
