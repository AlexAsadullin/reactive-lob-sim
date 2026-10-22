# Measuring the sim-to-real gap: evidence for the Reactive LOB Simulator

The present report gives the evidence for the model choices of the Reactive LOB Simulator, which is project S3
of the Tradeq platform. All numbers come from the command make report, which runs the command sim report with
fixed seeds and writes the files tables.md, study.json, kappa_sweep.json and engine_speed.json under the
directory $TRADEQ_DATA_DIR/reactive-lob-sim/reports/. When the text cites Table E1, E2 or E3, it refers to the
section of tables.md that carries that name.

## 1. Question

The present project asks which simulator of a limit order book (LOB), which is the list of resting buy and
sell orders of a venue, best predicts the outcomes that a set of strategies later obtains in live trading. The
specification of the platform, section S3, states that nobody closes the loop from generator to realism to
live degradation and that the missing quantity is a sim-to-real gap score, so we can conclude that the
comparison must score simulators on the outcomes of strategies and not only on the statistics of their prices.
The second question is whether the impact coupling kappa, which sets how strongly the simulated order flow
reacts to past trades, controls the fat tails of simulated returns. The specification, section S3, cites a
2026 replication which found that fat tails appear only when the price-impact coupling is strong enough, so
the present study tests that mechanism directly. The third question is how fast the matching engine runs,
because the stress library needs many simulated hours.

## 2. Data

The present study uses synthetic data only, because the platform has no recorded tape from the tape recorder
(project I1) and no shadow fills from the Shadow Exchange (project I6) yet, so the results show how the method
behaves and are not market evidence. A hidden ground-truth market generates the data, and it is the
agent-based model of the present project with the parameters that the section reproduce.ground_truth_overrides
of config/settings.yaml sets. That market has two volatility regimes, whose volatility multipliers are 0.8 and
2.0. Its impact coupling is kappa = 1.0, and its square-root impact law, which section 3 defines, has Y = 0.8
and delta = 0.5, as the entry ground_truth_impact of study.json records. The ground-truth market first writes
a calibration tape of 1800 seconds, and the simulators see only that tape. It then writes 4 held-out episodes
of 300 seconds each, which serve as the real tape.

The strategy set has six members of three kinds, because the file config/settings.yaml defines two market
makers, two momentum traders and two time-weighted average price (TWAP) executions. In every episode each
strategy runs inside the ground-truth market itself with the seed of that episode, and these runs are the live
results. The present study writes their fills in the format that section 6.4 of the platform conventions
defines, and it reads them back through the reader that will consume the I6 export, so the live path of the
study is the path that real shadow fills will take. The 6 strategies and the 4 episodes give 24 units, where a
unit is one pair of a strategy and an episode.

## 3. Methods

The present study compares four simulators, which all see the same calibration tape. Replay plays back the
recorded tape of the held-out episode and injects the orders of the strategy, and a resting order of the
strategy joins the back of its queue and advances when recorded trades or cancellations consume the size ahead
of it. Replay therefore knows the realised path, but it never reacts to the strategy. The zero-intelligence
simulator, which follows Smith, Farmer et al. (2003), draws every order type from a Poisson process whose
rates equal the stationary rates of the fitted flow, and it has no impact coupling and no liquidations. The
Hawkes ABM is an agent-based model (ABM), in which a multivariate Hawkes process draws the timing of six event
types. A Hawkes process is a point process in which every event raises the future rate of other events by an
amount that decays exponentially. Noise traders, fundamental traders that trade toward a hidden value and
momentum traders that follow the recent trend decide what each event does, and a liquidation agent turns
adverse moves of leveraged positions into forced market orders. The generative simulator draws the next event
from a gated recurrent unit (GRU) network, which is a small recurrent neural network, and the network
conditions on funding, open interest and the volatility regime.

The calibration of the Hawkes ABM first fits the Hawkes parameters by maximum likelihood, which chooses the
parameters that make the event sequence of the calibration tape most probable. It then fits the square-root
law of Toth et al. (2011), which states that the mean relative price change of a metaorder is Y times the
window volatility times the participation to the power delta. The fit cuts the tape into 30-second windows,
groups the windows into 8 bins of participation, and regresses the logarithm of the bin means on the logarithm
of the participation. The decay of impact over time follows the propagator (1 + tau/tau0) to the power -beta,
which Gatheral (2010) describes and which the module impact/fit.py fits on lagged order flow. The coupling
kappa then comes from a probe experiment, which the module calibration/coupling.py implements. For each kappa
on the grid 0, 0.25, 0.5, 1, 1.5, 2 and 3, the simulator runs a probe that buys half of the mean window volume
and, on the same seed, a probe that sells the same amount, and the measured impact is half the difference of
the two price changes. The calibrated kappa is the point where the measured curve crosses the impact that the
fitted square-root law predicts, which the code finds by linear interpolation on the grid.

Each simulator runs every strategy in every episode, and the stochastic simulators run 2 seeds per unit, whose
mean serves as the prediction. The gap score G of a simulator, which the module gap/score.py defines, compares
the predicted and the live values of three outcomes, which are the profit and loss (PnL), the fill rate and
the slippage in basis points. For each outcome the score takes the Spearman rank correlation rho, which is the
correlation of the ranks of the predicted and the live values across the 24 units. It also takes the
normalised mean absolute error (NMAE), which is the mean absolute prediction error divided by the mean
absolute deviation of the live values from their own mean. An NMAE below 1 therefore means that the simulator
beats a forecast that always states the average live outcome. G is the mean over the three outcomes of 0.5 (1
minus rho) / 2 plus 0.5 NMAE, so a perfect simulator has G = 0. The confidence intervals come from 1000
bootstrap resamples, which draw 24 units with replacement, and every simulator sees the same resampled units.
The share of resamples in which a simulator has the lowest G therefore estimates its probability of ranking
first, which the tables call P(best).

The realism scores follow LOB-Bench, because Nagy et al. (2025), section 4, compare real and generated samples
by the Wasserstein-1 distance and by the two-sample Kolmogorov-Smirnov (KS) statistic. The Wasserstein-1
distance is the area between the two cumulative distribution functions, and the present study divides it by
the standard deviation of the real sample, so that the scores of seven samples with different units can be
averaged. The KS statistic is the largest vertical gap between the same two functions, so it reacts to the
worst point of the fit. The seven samples, which the module realism/distances.py compares, are 1-second and
10-second returns, absolute 1-second returns, spread, depth, trade inter-arrival times and trade sizes. The
response error compares the market-impact response functions of Bouchaud et al. (2004), which are the mean
signed mid-price change at lags from 1 to 30 seconds after a trade. The error is the root-mean-square
difference of the two functions, which the study divides by the root-mean-square of the real response. The
stylised facts of Cont (2001), which are statistical regularities of asset returns, add further statistics.
The excess kurtosis of mid returns measures how much heavier the tails are than those of a normal
distribution, whose excess kurtosis is 0. The Hill (1975) tail index is computed from the largest 5% of
absolute returns, and a lower index means a fatter tail. The study also reports the autocorrelation of returns
and of absolute returns (ACF), and the R squared of the regression of mid changes on the order-flow imbalance
(OFI) of Cont, Kukanov and Stoikov (2014), which is the net change of the bid and ask queue sizes.

## 4. Experiments and results

The first result is that replay predicts live outcomes best, while the three reactive simulators cannot be
separated from each other on the gap score. As Table E1 shows, replay has G = 0.115 with a 95% interval from
0.075 to 0.191, while the generative simulator has G = 0.285 with an interval from 0.233 to 0.391. The two
intervals do not overlap, and replay had the lowest G in all 1000 paired resamples, because study.json records
P(best) = 1.0. The generative simulator, the zero-intelligence simulator and the Hawkes ABM have G values of
0.285, 0.335 and 0.383, whose intervals overlap, so their order is not reliable at 24 units. The advantage of
replay has a known cause, which is that replay plays back the very episode in which the live run took place,
while the other simulators run on independent seeds and cannot know the realised path.

| Simulator (24 units) | G (95% CI) | rho PnL | rho fill rate | rho slippage | NMAE PnL |
|---|---|---|---|---|---|
| replay | 0.115 (0.075 to 0.191) | 0.83 (0.55 to 0.94) | 0.99 (0.95 to 1.00) | 0.93 (0.76 to 0.99) | 0.47 (0.27 to 0.81) |
| generative | 0.285 (0.233 to 0.391) | 0.43 (-0.05 to 0.74) | 0.91 (0.72 to 0.95) | 0.87 (0.68 to 0.95) | 1.06 (0.87 to 1.45) |
| zero-intelligence | 0.335 (0.264 to 0.449) | 0.33 (-0.11 to 0.67) | 0.94 (0.84 to 0.97) | 0.91 (0.73 to 0.97) | 1.00 (0.79 to 1.33) |
| Hawkes ABM | 0.383 (0.240 to 0.511) | 0.15 (-0.36 to 0.64) | 0.97 (0.92 to 0.99) | 0.93 (0.78 to 0.98) | 1.61 (0.95 to 2.19) |

The second result is that every simulator ranks the strategies correctly on fill rate and slippage, while only
replay ranks them on PnL. As Table E1 shows, the rank correlation of fill rate lies between 0.91 and 0.99 for
all four simulators, and the rank correlation of slippage lies between 0.87 and 0.93. The rank correlation of
PnL is 0.83 for replay, while the intervals of the three reactive simulators include zero. The NMAE of PnL is
0.47 for replay, but it is 1.00 or more for the reactive simulators, so these simulators predict the PnL of a
strategy no better than the average live PnL does. We conclude that the strategy kind determines fill rate and
slippage, which every simulator reproduces, while PnL depends on the realised path, which only replay sees.

The third result is that the Hawkes ABM reproduces the microstructure of the real tape more closely than the
zero-intelligence simulator on most facts, but the distributional distances do not separate them
significantly. As the stylised-fact table of tables.md shows, the real tape has a mean spread of 1.37 ticks,
an OFI R squared of 0.81 and a lag-1 return autocorrelation of 0.17 over 4 episodes. The Hawkes ABM gives 1.56
ticks, 0.78 and 0.24, while the zero-intelligence simulator gives 2.20 ticks, 0.50 and -0.12, so only the
Hawkes ABM reproduces the sign of the autocorrelation and the strength of the OFI relation. The mean KS
statistic is 0.185 with a standard error of 0.021 for the Hawkes ABM, and it is 0.233 with a standard error of
0.020 for the zero-intelligence simulator. Welch's t-test for two independent samples with unequal variances,
which we computed from these standard errors under the null hypothesis of equal mean KS statistics, equals
1.66 with about 6.0 degrees of freedom, while the critical value at 95% confidence is 2.45, so the difference
in KS distance is not significant over 4 episodes. The response error points the other way, because Table E1
gives 1.027 with a standard error of 0.540 for the Hawkes ABM and 0.860 with a standard error of 0.064 for the
zero-intelligence simulator. Both simulators understate the excess kurtosis of 1-second returns, which is
11.41 with a standard error of 1.50 in the real tape, because they give 5.04 and 5.27.

| Source (4 episodes) | Kurtosis 1 s | Spread, ticks | Depth | OFI R squared | Return ACF lag 1 | Mean KS |
|---|---|---|---|---|---|---|
| real tape | 11.41 (1.50) | 1.37 (0.05) | 26.46 (1.01) | 0.81 (0.04) | 0.17 (0.01) | not applicable |
| zero-intelligence | 5.27 (0.86) | 2.20 (0.01) | 26.36 (1.04) | 0.50 (0.07) | -0.12 (0.06) | 0.233 (0.020) |
| Hawkes ABM | 5.04 (0.78) | 1.56 (0.08) | 21.38 (0.67) | 0.78 (0.03) | 0.24 (0.07) | 0.185 (0.021) |
| generative | 8.99 (1.77) | 1.26 (0.07) | 73.34 (6.04) | 0.21 (0.02) | -0.07 (0.02) | 0.273 (0.021) |

In this table the values in parentheses are standard errors, which the study computed over the 4 episodes. The
generative simulator comes closest to the real kurtosis, but its tape has a mean depth of 73.34 against 26.46
in the real tape, and its OFI R squared is 0.21 against 0.81, so its book does not respond to order flow as
the real book does. Its mean normalised Wasserstein distance is 1.430 with a standard error of 0.209, while
the Hawkes ABM scores 0.404 and the zero-intelligence simulator scores 0.425. Its Hill index of 1926.54 with a
standard error of 1074.96 is not a usable estimate, because the Hill formula takes the inverse of the mean log
ratio of the largest absolute returns, and that mean approaches zero when those returns are nearly equal,
which is what a coarse tick grid produces.

The fourth result is that the coupling kappa raises the kurtosis of 1-second returns in the Hawkes ABM. As
Table E2 shows, the excess kurtosis of 1-second returns rises from 5.83 at kappa = 0, with a 95% margin of
error of 1.84, to 20.52 at kappa = 4, with a margin of 14.07, over 8 seeds of 600 seconds per value. The
Spearman rank correlation between kappa and the per-seed kurtosis over all 40 runs equals 0.623 with a p-value
below 0.0001, while the critical value at 95% confidence for 40 pairs is 0.31, so we reject the null
hypothesis of no monotone relation. Welch's t-test for two independent samples with unequal variances, which
compares kappa = 4 with kappa = 0, equals 2.45 with 7.2 degrees of freedom and a p-value of 0.043, while the
critical value at 95% confidence is 2.35, so the mean kurtosis at the strongest coupling exceeds the mean at
zero coupling. The same tests on 10-second returns give a Spearman correlation of 0.286 with a p-value of
0.073 and a Welch t of 1.74 with 11.8 degrees of freedom and a p-value of 0.107, while the critical values are
0.31 and 2.18, so the effect is not significant at the 10-second scale. The lag-1 autocorrelation of 1-second
returns also rises from 0.101 to 0.311, which shows that strong coupling adds short-term momentum.

| kappa (8 seeds each) | Kurtosis 1 s | Kurtosis 10 s | Hill index 1 s | Abs-return ACF 1 s | Return ACF lag 1 |
|---|---|---|---|---|---|
| 0 | 5.83 (1.84) | 0.90 (0.99) | 3.62 (1.22) | 0.082 (0.033) | 0.101 (0.076) |
| 0.5 | 8.38 (6.11) | 1.57 (1.75) | 6.04 (2.05) | 0.073 (0.031) | 0.124 (0.059) |
| 1 | 8.90 (4.61) | 1.93 (1.28) | 4.48 (1.58) | 0.104 (0.031) | 0.212 (0.071) |
| 2 | 15.69 (7.27) | 2.91 (2.55) | 4.56 (1.69) | 0.131 (0.040) | 0.314 (0.079) |
| 4 | 20.52 (14.07) | 2.26 (1.56) | 3.52 (0.82) | 0.124 (0.029) | 0.311 (0.042) |

In Table E2 the values in parentheses are the half-widths of 95% Student t intervals, which the study computed
across the 8 seeds. The Hill index does not fall monotonically with kappa, because it is 3.62 at kappa = 0 and
3.52 at kappa = 4, while its intervals are wider than the change.

The fifth result is that the calibration recovers kappa only roughly, because the fitted impact law differs
from the true one. As study.json shows, the calibrated kappa is 1.32 against a true value of 1.0. The tape fit
gave Y = 1.22 and delta = 1.0 against true values of 0.8 and 0.5, and its R squared on the bin means was 0.63.
The exponent delta stopped at the upper bound of 1.0 that the setting calibration.impact.delta_max allows. The
probe curve is not monotone, because the measured impact is 0.000200 at kappa = 0 and 0.000169 at kappa =
0.25, and it is 0.000657 at kappa = 1.5 and 0.000656 at kappa = 2, while the standard errors over 4 probe
seeds lie between 0.000028 and 0.000198. The crossing of a noisy curve with a biased target therefore gives an
estimate whose error the present study cannot yet bound.

The sixth result is that the Numba batch kernel, which compiles the matching loop to machine code, processes
an exogenous order stream about 150 times faster than the pure-Python object book. As Table E3 shows, the
kernel processed 70.3 million events per second on 20000 events, with a 95% margin of 2.1 million over 5
repeats, while the object book processed 0.419 million events per second, so the speed-up is 167.9. On 200000
events the two rates are 56.2 million and 0.369 million, so the speed-up is 152.1. The largest absolute
difference between the fills and depths of the two implementations is 8.9e-16, which is the rounding error of
floating-point numbers, so the two implementations agree. The interactive Hawkes ABM, whose Python agents act
on the object book, runs at 61823 scheduled events per second with a margin of 3554, as Table E3 shows. The
calibration tape holds 21741 events in 1800 seconds, which is about 12 events per second, so one simulated
hour takes about 0.7 seconds of computation if each tape event corresponds to one scheduled event.

## 5. Comparison with analogs

The specification names ABIDES, JAX-LOB, TRADES with DeepMarket, LOB-Bench, DiffLOB and the LLM market of
Lopez-Lira as the analogs, and the present study ran none of them, because each one needs equity data, a GPU
or paid model calls that the project lacks. The comparison therefore rests on two baselines that the present
study ran, which are replay and the zero-intelligence simulator, and on the documented behaviour of the
analogs. The specification, section S3, states that ABIDES uses hand-crafted agents with manual calibration,
so we can conclude that the Hawkes ABM of the present project is the same kind of simulator with a fitted flow
and a fitted coupling. Table E1 shows that this calibration did not yet make the Hawkes ABM better than the
zero-intelligence baseline at predicting live outcomes, because their G values of 0.383 and 0.335 have
overlapping intervals.

| Analog | Documented behaviour (specification, section S3) | Evidence in the present study |
|---|---|---|
| ABIDES | hand-crafted agents, manual calibration | Hawkes ABM with fitted flow: G = 0.383, not separable from the baselines |
| JAX-LOB | GPU books, up to 75 times faster per message | Numba kernel on CPU: 152 to 168 times the object book (Table E3) |
| TRADES with DeepMarket | diffusion order flow, two equity stocks | GRU generator: G = 0.285, depth and OFI unrealistic |
| LOB-Bench | realism metrics on LOBSTER equity data | same Wasserstein and KS distances, applied to synthetic crypto tapes |
| DiffLOB | regime-conditioned counterfactuals on level-2 snapshots | stress library of message-level scenarios, no realism score yet |
| Lopez-Lira LLM market | LLM trading agents, expensive and small scale | not run, because it needs paid model calls |

The speed comparison with JAX-LOB is not direct, because Frey et al. (2023) measure their speed-up against a
different CPU implementation, while Table E3 measures the Numba kernel against the object book of the present
project. The present study reimplemented the distances of LOB-Bench, but it did not run the LOB-Bench code, so
the scores cannot be compared with the published LOB-Bench numbers.

## 6. Decision

The present project adopts the calibrated Hawkes ABM with an explicit coupling kappa as the default engine of
the sim-service, and it keeps replay as the reference for strategies whose own trades do not move the price.
Table E1 supports replay as the best predictor of live outcomes when the realised path is known, but replay
cannot react, so it cannot serve the stress library, whose scenarios need the market to respond to a shock.
Table E2 supports an explicit kappa, because kappa is the parameter that moves the kurtosis of 1-second
returns from 5.83 to 20.52, which is the mechanism that the specification names. Among the reactive
simulators, the Hawkes ABM had the lowest mean KS statistic, the OFI R squared closest to the real tape and
the only positive return autocorrelation in the stylised-fact table, while its gap score did not differ
significantly from the others. The generative simulator stays experimental, because its depth and OFI relation
fail the stylised-fact table even though its gap score is the lowest of the reactive simulators. The present
project uses the Numba kernel for exogenous order streams, because Table E3 shows a speed-up of at least 152
with results that agree to 8.9e-16.

During the study we changed the handling of forced liquidations in the module abm/market.py, because a
regression test showed that a forced market order was dropped whenever an earlier forced order of the same
batch had emptied the opposite side of the book. Before the change the liquidation agent skipped that order,
which left open interest too high. After the change liquidity providers refill the side first, so the order
executes and reduces open interest. This change altered the random streams of the Hawkes ABM, so all numbers
in the present report come from the run after the change.

## 7. Limitations

The present study rests on synthetic data, and its ground truth belongs to the same model family as the Hawkes
ABM, so the results do not show how the simulators perform on a real crypto venue. The platform needs I1 tape
and I6 fills before it can test that. The gap score uses 24 units from 4 episodes, so its intervals are wide,
and the order of the three reactive simulators is not reliable. The share of units whose live value lies
inside the range of the 2 seeds lies between 4% and 54% in study.json, which shows that 2 seeds do not
describe the spread of a prediction. The calibration of kappa depends on a square-root fit whose exponent
stopped at its bound, so the coupling estimate is biased in this sample. The generative model trained for 4
epochs on 306 windows of 64 events, and its training loss fell only from 7.92 to 7.61, while its validation
loss of 7.64 shows that it did not overfit but also did not learn much structure.

## 8. References

Berti, L., Prenkaj, B. and Velardi, P. (2025). TRADES: Generating realistic market simulations with diffusion
models. arXiv 2502.07071.

Bouchaud, J.-P., Gefen, Y., Potters, M. and Wyart, M. (2004). Fluctuations and response in financial markets:
the subtle nature of random price changes. Quantitative Finance 4(2).

Bouchaud, J.-P., Bonart, J., Donier, J. and Gould, M. (2018). Trades, Quotes and Prices. Cambridge University
Press.

Byrd, D., Hybinette, M. and Balch, T. H. (2020). ABIDES: Towards high-fidelity multi-agent market simulation.
Proceedings of the ACM SIGSIM Conference on Principles of Advanced Discrete Simulation.

Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. Quantitative
Finance 1(2).

Cont, R., Kukanov, A. and Stoikov, S. (2014). The price impact of order book events. Journal of Financial
Econometrics 12(1).

Frey, S., Li, K., Nagy, P., Sapora, S., Lu, C., Zohren, S., Foerster, J. and Calinescu, A. (2023). JAX-LOB: A
GPU-accelerated limit order book simulator. arXiv 2308.13289.

Gatheral, J. (2010). No-dynamic-arbitrage and market impact. Quantitative Finance 10(7).

Hill, B. M. (1975). A simple general approach to inference about the tail of a distribution. Annals of
Statistics 3(5).

Nagy, P. et al. (2025). LOB-Bench: Benchmarking generative AI for finance, an application to limit order book
data. Proceedings of ICML 2025, arXiv 2502.09172.

Smith, E., Farmer, J. D., Gillemot, L. and Krishnamurthy, S. (2003). Statistical theory of the continuous
double auction. Quantitative Finance 3(6).

Toth, B., Lemperiere, Y., Deremble, C., de Lataillade, J., Kockelkoren, J. and Bouchaud, J.-P. (2011).
Anomalous price impact and the critical nature of liquidity in financial markets. Physical Review X 1.

The DiffLOB paper (arXiv 2602.03776), the LLM-market replication on price-impact coupling (arXiv 2608.11215)
and the Lopez-Lira LLM market simulation (arXiv 2504.10789) are cited as the platform plan,
quant_platform_plan_full.md, section S3, links them, and that section gives the list of analogs and their
documented limits.
