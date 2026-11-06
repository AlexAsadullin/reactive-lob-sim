# HANDOVER: reactive-lob-sim (Tradeq S3)

Last updated 2026-11-06 (MOEX adaptation). The sections from "Goal" to "Update 2026-10-23" describe the v0.1
crypto build; the section "MOEX adaptation (2026-11-06)" at the end is the current state and wins where they
differ.

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
| src/reactive_lob_sim/abm/market.py | Agent-based market with MOEX phases and scenarios |
| src/reactive_lob_sim/crypto/{liquidation,macro}.py | Optional crypto module: liquidation cascades, funding and open interest |
| src/reactive_lob_sim/engine/{rules,auction}.py, moex/{instruments,sessions}.py | Lots, tick, band; uncross; instrument registry and fees; session schedule |
| src/reactive_lob_sim/hawkes/{kernels,mle}.py | Multivariate exponential Hawkes: maximum-likelihood fit and simulation |
| src/reactive_lob_sim/generative/{tokenizer,model,market}.py | GRU order-flow model that drives the engine |
| src/reactive_lob_sim/impact/{fit,model}.py | Propagator impact with square-root size dependence |
| src/reactive_lob_sim/calibration/*.py | Tape to calibration files |
| src/reactive_lob_sim/gap/{score,outcomes}.py | Sim-to-real gap score |
| src/reactive_lob_sim/realism/{facts,distances}.py | Stylised facts and distributional distances |
| src/reactive_lob_sim/scenarios/*.py | Counterfactual stress library (YAML under the data root) |
| src/reactive_lob_sim/tape/*.py, contracts/*.py | I1 tape I/O, fills, PIT client, calibration-file contracts |
| src/reactive_lob_sim/reproduce.py, experiments.py | `make reproduce` study; `make report` tables |
| tests/unit/*, tests/integration/test_study.py | 55 tests, including MOEX rules (test_moex.py) and a shrunken end-to-end study |

## State at handover

1. Commit 6ffabbe fixed a real bug in `abm/market.py`. `triggered()` removes positions before execution, and the old `continue` then silently dropped a forced liquidation (and its open-interest reduction) after a sweep had emptied the opposite side. A regression test in `tests/unit/test_abm_strategies.py` fails without the fix (1 liquidation where 2 are expected).
2. Commit 624f984 made ruff and mypy --strict clean, and commit 467eb82 raised coverage from 62% to 93%. On 2026-10-22 `make test` gave 40 passed and 93% coverage.
3. `make report` outputs are in `$TRADEQ_DATA_DIR/reactive-lob-sim/reports/` (study.json, tables.md, engine_speed.json, kappa_sweep.json), written at 15:40 after the liquidation fix.
4. Commit fba11c8 wrote the lean README with a key-results table.
5. `docs/report.md` was checked number by number against tables.md, study.json, kappa_sweep.json and engine_speed.json. The corrected text went back to the orchestrator, which writes and commits it (subagents cannot write report files). Corrections: the Hawkes ABM OFI R squared is 0.78, not 0.81; P(best) of replay is exactly 1.0 (all 1000 resamples); the claim "1 hour of simulated market per 10 seconds" was wrong by a factor of about 14 (12 events/s times 3600 s at 61823 events/s is about 0.7 s); the Hawkes ABM does not have the closest spread among reactive simulators (the generative one does), so the Decision now cites KS, OFI R squared and the sign of the autocorrelation; the R squared of the square-root fit is over the 8 bin means, not over 59 windows; the response-error result (Hawkes ABM 1.027 against zero-intelligence 0.860) and the 10-second critical values were added; Smith, Farmer et al. (2003) and the arXiv links of DiffLOB, the LLM-market replication and Lopez-Lira were added to the references.

## Update 2026-10-23: I6 adapter findings

I6 reads impact.json and scenarios/*.yaml through shadow-exchange/src/shadow_exchange/s3_adapter.py, which
rejects unknown document keys, unknown shock keys and unknown params. Keep those formats stable.

1. Commit 390884b added the optional scenario keys `time_scale` (default 1) and `time_offset_s` (default 0). A
   shock runs from `time_offset_s + time_scale * start_s` for `time_scale * duration_s`, the same mapping the
   adapter applies from its own config. `Scenario.to_document()` omits them at their defaults, so the shipped
   files keep the six keys the adapter accepts. The adapter rejects a file that sets them, until I6 decides to
   read them from the file. The ABM applies them through `Scenario.scaled_shocks()`.
2. `sim seed-scenarios --refresh` rewrites shipped default files that differ from the current defaults (never
   files with other ids), then validates every file and exits 1 on an invalid one. It was run on the data
   root on 2026-10-23. liquidation-cascade.yaml now has `layers` and `layer_gap_bps`, and every file validates.
   `tests/unit/test_scenario_contract.py` fails if a seeded file misses a required field or carries keys
   beyond the contract set.
3. The funding-shock description and the README state that the 300 s shock covers no 8-hour settlement.
4. Commit c19eef6 documents every impact.json and scenario field in the README.
5. The report numbers did not change, because the report experiments use no scenario file and default
   timing leaves the ABM unchanged. `make test` gave 43 passed and 92% coverage; lint and mypy are clean.

## Mistakes and lessons

1. The first build left coverage at 62% against the 85% platform target, and lint and mypy were not clean, even though the repo was described as done. Run `make lint typecheck test` before claiming done.
2. The B023 closure bug in reproduce.py (a lambda that captured a loop variable) passed tests and was found only by lint.
3. The first report draft had numbers that were not in the outputs (an OFI R squared copied from the wrong row, a derived speed figure computed wrongly, a "closest spread" claim that the table contradicts). Check every number in a report against the generated files, and recompute every derived number.
4. These lessons apply platform-wide. Compound shell commands were denied for subagents; `make -C`, `git -C` and `.venv/bin/python` work. Subagents may not write report files (a harness rule), so return the report text to the orchestrator. Commit early, because of the rate limit.

## MOEX adaptation (2026-11-06)

1. Default market: SBER-RUB-TQBR (lot 10, tick 0.01, band 20% around the start price), commission 25 bps both
   sides, 0 for T-Capital funds (`instruments.*`, I1's zero-fee list). `instruments.symbols` lists SBER, GAZP,
   LKOH and TMOS; I1's `universe_moex.json` overrides lot and tick when present.
2. Engine: `TradingRules` rejects odd lots, off-grid prices and prices outside the band; market orders carry
   the band edge as protection. Phases (`moex.sessions.Phase`): in an auction orders are collected (market
   orders become limits at the band edge) and `set_phase` uncrosses at the equilibrium price; halted, break
   and closed reject new orders. `sessions.apply: true` follows the MSK schedule (morning, main, evening,
   weekend sessions with opening and closing auctions).
3. Crypto: `abm.crypto: null`; its parameters are in `config/crypto.yaml`, the modules in `crypto/`. Scenarios
   `liquidation_cascade` and `funding_shock` need it and raise an error without it. The generative market's
   funding/OI conditioning is 0 without it.
4. Scenarios: opening-auction-imbalance, dividend-gap, key-rate-surprise, trading-halt added (9 in the library;
   formats unchanged, so I6's s3_adapter only needs the new kinds and their params).
5. Strategies are sized in lots (`quote_lots`, `trade_lots`, `total_lots`, `*_lots` limits).
6. Fixes found on the way: empty size bins of the GRU tokenizer gave NaN sizes with discrete lot sizes (now the
   nearest filled bin).
7. `make lint typecheck test`: 55 passed, 91% coverage. `make report` rerun (46 s): numbers in README.
8. Open: real calibration needs a longer T-Bank capture (48 events in 2 minutes, 200 needed); kappa probe gave 0
   on the MOEX world; halts with a reopening auction collect orders for the whole halt.

Lesson: an interrupted refactor left the engine signature changed and four callers unmigrated; run
`make typecheck` first when resuming someone else's work.
