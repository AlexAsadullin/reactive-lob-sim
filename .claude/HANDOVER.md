# HANDOVER: reactive-lob-sim (Tradeq S3)

Last updated 2026-10-22 by the orchestrating session, from the transcript of the agent that stopped at the
session rate limit.

## Goal

The project is a reactive crypto limit-order-book simulator with an explicitly calibrated impact coupling (kappa),
plus a sim-to-real gap score that measures how far strategy outcomes in a simulator diverge from outcomes on real tape.
The spec is in `../prompts/09-reactive-lob-sim.md`. The default port is 8609. The project produces contract 6.5
(simulation service, impact calibration and scenario library) and consumes 6.2 (tape), 6.3 (PIT) and 6.4 (fills).
For the report, the model choice needs evidence: an agent-based model (ABM) with propagator impact, compared with
Hawkes order flow, a generative GRU model and plain replay.

## Stack

Python 3.13, uv `.venv`, and pins from `../constraints.txt`. The project uses numba (the matching kernel and Hawkes
kernels), numpy, scipy, pyarrow, torch (the generative GRU model, imported by `generative/`), FastAPI and uvicorn,
httpx, typer (the `sim` CLI), pydantic-settings (`config/settings.yaml`) and structlog.

## Architecture and data flow

`calibration/pipeline.py` takes the tape, extracts events (`calibration/events.py`), fits the flow parameters
(`flow.py`), impact (`impact/fit.py`, using a square-root law and propagator decay) and the coupling kappa
(`coupling.py`, using a probe metaorder), and writes calibration files (contract 6.5). The simulators run on the L3
matching engine (`engine/book.py` and the shared machinery in `l3market.py`). They are the ABM (`abm/market.py` with
the liquidation-cascade agent and the funding and OI model), Hawkes flow (`hawkes/`), the generative model
(`generative/`) and tape replay (`replay/market.py`). The reference strategies in `strategies/simple.py` (passive
market making, momentum taking and TWAP) run on each simulator. `gap/score.py` compares their outcomes with outcomes on
real tape (the gap score), and `realism/` adds stylised facts and LOB-Bench-style distances. `simctx.py` is the single
place that constructs the simulators. `service/app.py` is the sim-service.

## Files

| Path | Purpose |
|---|---|
| src/reactive_lob_sim/engine/{book,fastbook,l3market,types,accounting}.py | L3 price-time book, Numba batch kernel, fees |
| src/reactive_lob_sim/abm/{market,liquidation,macro}.py | Agent-based market, liquidation cascades, funding and open interest |
| src/reactive_lob_sim/hawkes/{kernels,mle}.py | Multivariate exponential Hawkes: maximum-likelihood fit and simulation |
| src/reactive_lob_sim/generative/{tokenizer,model,market}.py | GRU order-flow model that drives the engine |
| src/reactive_lob_sim/impact/{fit,model}.py | Propagator impact with square-root size dependence |
| src/reactive_lob_sim/calibration/*.py | Tape to calibration files |
| src/reactive_lob_sim/gap/{score,outcomes}.py | Sim-to-real gap score |
| src/reactive_lob_sim/realism/{facts,distances}.py | Stylised facts and distributional distances |
| src/reactive_lob_sim/scenarios/*.py | Counterfactual stress library (YAML under the data root) |
| src/reactive_lob_sim/tape/*.py, contracts/*.py | I1 tape I/O, fills, PIT client, calibration-file contracts |
| src/reactive_lob_sim/reproduce.py, experiments.py | `make reproduce` study; `make report` tables |
| tests/unit/*, tests/integration/test_study.py | 40 tests, including a shrunken end-to-end study and the service |

## State at handover

1. Commit 6ffabbe fixed a real bug in `abm/market.py`. `triggered()` removes positions before execution, and the old `continue` then silently dropped a forced liquidation (and its open-interest reduction) after a sweep had emptied the opposite side. A regression test in `tests/unit/test_abm_strategies.py` fails without the fix (1 liquidation where 2 are expected).
2. Commit 624f984 made ruff and mypy --strict clean, and commit 467eb82 raised coverage from 62% to 93%. On 2026-10-22 `make test` gave 40 passed and 93% coverage.
3. `make report` outputs are in `$TRADEQ_DATA_DIR/reactive-lob-sim/reports/` (study.json, tables.md, engine_speed.json, kappa_sweep.json), written at 15:40 after the liquidation fix.
4. Commit fba11c8 wrote the lean README with a key-results table.
5. `docs/report.md` was checked number by number against tables.md, study.json, kappa_sweep.json and engine_speed.json. The corrected text went back to the orchestrator, which writes and commits it (subagents cannot write report files). Corrections: the Hawkes ABM OFI R squared is 0.78, not 0.81; P(best) of replay is exactly 1.0 (all 1000 resamples); the claim "1 hour of simulated market per 10 seconds" was wrong by a factor of about 14 (12 events/s times 3600 s at 61823 events/s is about 0.7 s); the Hawkes ABM does not have the closest spread among reactive simulators (the generative one does), so the Decision now cites KS, OFI R squared and the sign of the autocorrelation; the R squared of the square-root fit is over the 8 bin means, not over 59 windows; the response-error result (Hawkes ABM 1.027 against zero-intelligence 0.860) and the 10-second critical values were added; Smith, Farmer et al. (2003) and the arXiv links of DiffLOB, the LLM-market replication and Lopez-Lira were added to the references.

## Mistakes and lessons

1. The first build left coverage at 62% against the 85% platform target, and lint and mypy were not clean, even though the repo was described as done. Run `make lint typecheck test` before claiming done.
2. The B023 closure bug in reproduce.py (a lambda that captured a loop variable) passed tests and was found only by lint.
3. The first report draft had numbers that were not in the outputs (an OFI R squared copied from the wrong row, a derived speed figure computed wrongly, a "closest spread" claim that the table contradicts). Check every number in a report against the generated files, and recompute every derived number.
4. These lessons apply platform-wide. Compound shell commands were denied for subagents; `make -C`, `git -C` and `.venv/bin/python` work. Subagents may not write report files (a harness rule), so return the report text to the orchestrator. Commit early, because of the rate limit.
