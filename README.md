# Reactive LOB Simulator (S3)

A reactive simulator of a crypto limit order book (LOB) with an explicit, calibrated impact coupling kappa,
which sets how strongly simulated order flow reacts to an agent's trades. The project also computes a
sim-to-real gap score G: it runs one strategy set in each simulator and on the real tape, and it measures how
well each simulator predicts the real profit and loss (PnL), fill rate and slippage (0 is perfect). On
synthetic data, tape replay scored G = 0.115 (95% CI 0.075 to 0.191, 24 strategy-episode units), and the
reactive simulators scored 0.285 to 0.383. Raising kappa from 0 to 4 raised the excess kurtosis of 1-second
returns from 5.83 to 20.52 (Spearman rho 0.62, p < 0.0001, 40 runs). Details are in
[docs/report.md](docs/report.md).

Specification: `quant_platform_plan_full.md`, section "S3. Generative and agent-based market simulation".

## Quickstart

```bash
cp .env.example .env            # set TRADEQ_DATA_DIR=/Users/.../Tradeq/data
make setup                      # uv venv --python 3.13 .venv, uv sync --all-extras
make lint typecheck test        # ruff, mypy --strict, pytest with coverage (offline)
make reproduce                  # synthetic study: ground truth, calibration, 4 simulators, gap score
make report                     # reproduce + kappa sweep + engine speed, tables behind docs/report.md
make serve                      # sim-service on 127.0.0.1:8609
```

CLI: `sim seed-scenarios | calibrate --venue --symbol --start --end | simulate --scenario | serve | reproduce | report`.

## Configuration

Defaults live in `config/settings.yaml`. Environment variables override them with the prefix `SIM_` and the
nested delimiter `__`, for example `SIM_FEES__TAKER_BPS=4.5`. `TRADEQ_DATA_DIR` is the data root, and
`SIM_CONFIG_FILE` selects another settings file. The settings that matter most:

| Setting | Default | Meaning |
|---|---|---|
| `abm.impact.kappa` | 1.0 | impact coupling used when no calibration file exists for a symbol |
| `calibration.impact.kappa_grid` | 0 to 3 | grid of the probe-metaorder search for kappa |
| `contracts.pit_enabled` | false | read funding and open interest from I2 instead of the tape |
| `replay.queue_model` | proportional | whether cancels (not only trades) advance a resting order's queue |
| `gap.bootstrap_samples` | 1000 | bootstrap resamples behind the gap-score intervals |
| `reproduce.*`, `report.*` | see file | ground-truth market, episodes, seeds and report experiments |

## Architecture

```
I1 tape --> calibration (events, Hawkes MLE, square-root law + propagator, kappa probe) --> impact.json, abm.json
calibration --> simulators: replay | zero-intelligence | Hawkes ABM (+ liquidations, funding, OI) | GRU generator
                all simulators share the L3 price-time matching engine (engine/)
strategies (market making, momentum, TWAP) x simulators  +  I6 fills (live)  --> gap score G, realism scores
```

`engine/` is the L3 book with limit, market, cancel, modify, IOC, FOK and post-only orders and maker and taker
fees, plus a Numba batch kernel for exogenous order streams. `abm/` is the agent-based market (ABM) with noise,
fundamental and momentum traders, a liquidation-cascade agent and a funding and open-interest (OI) model.
`hawkes/` fits and simulates the multivariate Hawkes order flow. `generative/` is a small conditional GRU over
tokenised order events. `gap/` and `realism/` hold the gap score, the stylised facts and the LOB-Bench-style
Wasserstein and Kolmogorov-Smirnov distances. `simctx.py` is the only place that builds simulators, and
`service/` is the sim-service.

## Contracts

Produced: contract 6.5 (conventions section 6.5): `GET /v1/scenarios`, `POST /v1/simulations` (output in the
I1 tape schema), `GET /v1/simulations/{id}`, calibration files at
`$TRADEQ_DATA_DIR/reactive-lob-sim/calibration/venue=<v>/symbol=<s>/impact.json`, and the stress library
(baseline, spread widening, liquidity withdrawal, liquidation cascade, funding shock) at
`$TRADEQ_DATA_DIR/reactive-lob-sim/scenarios/*.yaml`. Consumed: 6.2 (I1 tape), 6.3 (I2 point-in-time funding
and OI, optional) and 6.4 (I6 shadow fills, the live side of the gap score).

## Key results

Synthetic data only (a hidden ground-truth ABM with two volatility regimes); see
[docs/report.md](docs/report.md) for the methods and all intervals.

| Result | Value (95% CI) | Sample |
|---|---|---|
| Gap score G, tape replay | 0.115 (0.075 to 0.191) | 24 units, 1000 bootstrap resamples |
| Gap score G, Hawkes ABM / zero-intelligence / GRU | 0.383 (0.240 to 0.511) / 0.335 (0.264 to 0.449) / 0.285 (0.233 to 0.391) | 24 units |
| Excess kurtosis of 1 s returns, kappa 0 vs 4 | 5.83 (3.99 to 7.67) vs 20.52 (6.45 to 34.59) | 8 seeds of 600 s each |
| Calibrated kappa vs true kappa | 1.32 vs 1.0 | 1 calibration tape of 1800 s |
| Numba kernel vs pure-Python book | 70.3 million vs 0.42 million events/s (168 times) | 20000 events, 5 repeats |

## Limitations

The evidence is synthetic, and the ground truth belongs to the same model family as the Hawkes ABM. The
24-unit gap score cannot order the three reactive simulators, whose intervals overlap. The square-root-law
exponent hit its upper bound of 1.0 on the calibration tape, so kappa came out at 1.32 against a true 1.0.
The GRU generator produces a book about three times deeper than the real one and is experimental.

## Roadmap

1. Run the gap score on real I1 tape and I6 shadow fills, with more episodes, so that the reactive simulators
   can be ordered.
2. Bound the error of the kappa calibration (bootstrap over calibration windows) and fit delta with a prior.
3. A LOB-Bench-format export, so that the published LOB-Bench code can score the synthetic tapes directly.
