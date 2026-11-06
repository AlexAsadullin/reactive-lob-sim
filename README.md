# Reactive LOB Simulator (S3)

A reactive simulator of a Moscow Exchange (MOEX) share's limit order book (LOB). It has an explicit,
calibrated impact coupling kappa, which sets how strongly simulated order flow reacts to an agent's trades. The
simulator models MOEX microstructure: orders in whole lots, a price tick per security, daily price bands, call
auctions (opening, closing, and discrete auctions after halts), trading halts, the morning, main, evening and
weekend sessions, and the T-Invest commission (0.25% per fill, 0% for T-Capital funds). The project also
computes a sim-to-real gap score G: it runs one strategy set in each simulator and on the real tape, and
measures how well each simulator predicts the real profit and loss (PnL), fill rate and slippage (0 is
perfect). On the synthetic MOEX world (SBER-like, 10-share lots), tape replay scored G = 0.133 (95% CI 0.090 to
0.223, 24 strategy-episode units) and the reactive simulators 0.328 to 0.523. Raising kappa from 0 to 4 raised
the excess kurtosis of 1-second returns from 4.27 to 10.21 (Spearman rho 0.64, p < 0.0001, 40 runs). Details
are in [docs/report.md](docs/report.md).

The v0.1 crypto features (perpetual funding, open interest, liquidation cascades) are kept as an optional
module that is off by default (`abm.crypto: null`, see `config/crypto.yaml`).

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

CLI: `sim seed-scenarios [--refresh] | calibrate --venue --symbol --start --end | simulate --scenario | serve | reproduce | report`.

## Configuration

Defaults live in `config/settings.yaml`. Environment variables override them with the prefix `SIM_` and the
nested delimiter `__`, for example `SIM_FEES__TAKER_BPS=4.5`. `TRADEQ_DATA_DIR` is the data root, and
`SIM_CONFIG_FILE` selects another settings file. The settings that matter most:

| Setting | Default | Meaning |
|---|---|---|
| `market.symbol` | SBER-RUB-TQBR | default instrument; lot, tick and band come from `instruments.symbols` |
| `instruments.commission_bps` | 25 | T-Invest basic tariff, maker and taker alike; T-Capital funds pay `zero_fee_commission_bps` (0) |
| `instruments.universe_file` | I1 `reference/universe_moex.json` | lot and tick per security (wins over settings; optional) |
| `sessions.apply` | false | follow the MOEX trading day from `market.epoch` (auctions uncross at their end) |
| `abm.crypto` | null | the optional crypto module; `config/crypto.yaml` holds its parameters |
| `abm.impact.kappa` | 1.0 | impact coupling used when no calibration file exists for a symbol |
| `calibration.impact.kappa_grid` | 0 to 3 | grid of the probe-metaorder search for kappa |
| `replay.queue_model` | proportional | whether cancels (not only trades) advance a resting order's queue |
| `gap.bootstrap_samples` | 1000 | bootstrap resamples behind the gap-score intervals |
| `reproduce.*`, `report.*` | see file | ground-truth market, episodes, seeds and report experiments |

## Architecture

```
I1 tape --> calibration (events, Hawkes MLE, square-root law + propagator, kappa probe) --> impact.json, abm.json
calibration --> simulators: replay | zero-intelligence | Hawkes ABM | GRU generator (+ optional crypto module)
                all simulators share the L3 price-time matching engine (engine/) with MOEX rules and auctions
strategies (market making, momentum, TWAP) x simulators  +  I6 fills (live)  --> gap score G, realism scores
```

`engine/` is the L3 book with limit, market, cancel, modify, IOC, FOK and post-only orders and maker and taker
fees, plus a Numba batch kernel for exogenous order streams. `engine/rules.py` enforces whole lots, the tick
grid and the price band; `engine/auction.py` is the uncross (maximum volume, then minimum surplus, then
closest to the reference price). `moex/` holds the instrument registry (lot, tick, band, zero-fee funds) and
the session schedule. `abm/` is the agent-based market (ABM) with noise, fundamental and momentum traders and
phase-dependent flow; `crypto/` is the optional liquidation-cascade agent and funding and open-interest model.
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

### impact.json (read by I6)

Keys: `schema_version` ("1"), `model` (always `propagator_sqrt_v1`), `params`, `fitted_at` (ISO-8601 UTC),
`sample_start` and `sample_end` (the calibration window, ISO-8601 UTC). A trade of signed size `s*q` at time
`t_i` moves the price by `I(t) = kappa * s * Y * sigma_window * P * (q / volume_window)^delta * (1 + (t - t_i) / tau0_s)^(-beta)`,
in price units, where `P` is the current price.

| `params` key | Type | Meaning |
|---|---|---|
| `Y` | float | prefactor of the square-root law |
| `delta` | float | size exponent of the square-root law (fit bounds 0.2 to 1.0) |
| `sigma_window` | float | relative mid volatility over one window of `window_s` |
| `volume_window` | float | mean traded volume per window, in base units |
| `window_s` | float | estimation window of the square-root law, seconds |
| `tau0_s` | float | propagator time scale, seconds |
| `beta` | float | propagator decay exponent (0 means permanent impact) |
| `kappa` | float | calibrated coupling; 1 means the raw tape-fitted impact |
| `source` | string | `all` (every trade feeds the impact state) or `experimental` (only the tested agent) |
| `n_exponentials` | float | terms of the internal sum-of-exponentials approximation; no meaning outside S3 |

`abm.json` in the same directory is the full ABM calibration; it is S3-internal and may change.

### scenarios/*.yaml (read by I6)

Every file has the keys `schema_version` ("1"), `id` (lower-case, digits and hyphens; equals the file name),
`description`, `default_duration_s` (the run length the shocks were written for), `shocks` (list, empty for
the baseline) and `abm_overrides` (partial ABM parameters, empty in all shipped files). Two optional keys stretch
and shift every shock: a shock runs from `time_offset_s + time_scale * start_s` for `time_scale * duration_s`
seconds. Their defaults are `time_scale: 1` and `time_offset_s: 0`, and files at the defaults do not contain
them, so the shipped files keep exactly the six keys above. Each shock has `kind`, `start_s`, `duration_s` and
`params`:

| `kind` | `params` | Effect in the S3 market |
|---|---|---|
| `spread_widening` | `extra_ticks` | quotes move `extra_ticks` further out; quotes inside that band are pulled |
| `liquidity_withdrawal` | `rate_factor`, `cancel_fraction` | `cancel_fraction` of resting quotes cancelled at the start; new quotes arrive at `rate_factor` of the normal rate |
| `liquidation_cascade` | `oi_multiplier`, `long_share`, `trigger_depth_fraction`, `layers`, `layer_gap_bps` | open interest times `oi_multiplier`, long share forced to `long_share`, crowded longs in `layers` layers `layer_gap_bps` apart, and a forced sell of `trigger_depth_fraction` of visible bid depth |
| `funding_shock` | `rate` | funding held at `rate` per interval |

All `params` are required. `sim seed-scenarios --refresh` rewrites shipped files that differ from the
current defaults (other files are left alone) and exits non-zero if any file in the directory fails
validation. The shipped shocks last 120 to 300 s inside 600 s runs. The funding shock lasts 300 s, so it
covers no 8-hour settlement and pays nothing in a replay with 8-hour funding unless `time_scale` stretches it
over one; the S3 market fixes funding every 60 s, where the shock is paid five times.

## Key results

Synthetic data (a hidden ground-truth ABM with two volatility regimes, in the MOEX world: SBER-like prices,
10-share lots, 0.25% commission), rerun on 2026-11-06; see [docs/report.md](docs/report.md) for the methods.

| Result | Value (95% CI) | Sample |
|---|---|---|
| Gap score G, tape replay | 0.133 (0.090 to 0.223) | 24 units, 1000 bootstrap resamples |
| Gap score G, Hawkes ABM / zero-intelligence / GRU | 0.328 (0.243 to 0.503) / 0.483 (0.391 to 0.705) / 0.523 (0.438 to 0.746) | 24 units |
| Excess kurtosis of 1 s returns, kappa 0 vs 4 | 4.27 (3.36 to 5.18) vs 10.21 (7.68 to 12.74) | 8 seeds of 600 s each |
| Calibrated kappa vs true kappa | 0.0 vs 1.0 | 1 calibration tape of 1800 s |
| Numba kernel vs pure-Python book | 81.8 million vs 0.46 million events/s (177 times) | 20000 events, 5 repeats |

Real data: the 2-minute T-Bank capture of SBER (2026-11-05) holds 48 order-flow events, below the 200 that
calibration needs (`sim calibrate --venue moex-tbank --symbol SBER-RUB-TQBR ...` refuses it), so no simulator
is calibrated on real MOEX tape yet.

## Limitations

The evidence is synthetic, and the ground truth belongs to the same model family as the Hawkes ABM. The
24-unit gap score orders replay first but cannot separate the reactive simulators reliably. In the MOEX world
the kappa probe returned 0 against a true 1.0: with 10-share lots and a 0.25% commission the probe metaorder's
footprint is coarse, so the calibration needs a longer tape or finer probes. Auction price determination
follows the generic equilibrium rule (MOEX trading rules); MOEX's exact tie-breaks and the closing-price
algorithm are not reproduced. Price bands are a fixed fraction (20%) around the episode's start price, not the
exchange's published limits. During a halt that reopens with an auction the simulator collects orders, while
MOEX rejects them until the reopening auction starts.

## Roadmap

1. Calibrate on a full trading day of T-Bank tape (order books and trades) and run the gap score on real I1
   tape and I6 shadow fills, with more episodes, so that the reactive simulators can be ordered.
2. Bound the error of the kappa calibration (bootstrap over calibration windows) and fit delta with a prior.
3. A LOB-Bench-format export, so that the published LOB-Bench code can score the synthetic tapes directly.
