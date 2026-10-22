"""Hawkes, impact and accounting maths against closed forms and known parameters."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from reactive_lob_sim.engine.accounting import Account, FeeSchedule
from reactive_lob_sim.engine.types import Liquidity, Side
from reactive_lob_sim.hawkes import kernels
from reactive_lob_sim.hawkes.mle import fit_hawkes, log_likelihood, simulate_hawkes
from reactive_lob_sim.impact.fit import fit_impact
from reactive_lob_sim.impact.model import ImpactState, exponential_sum, peak_impact, power_law_decay
from reactive_lob_sim.params import HawkesParams, ImpactParams
from reactive_lob_sim.settings import HawkesFitSettings, ImpactFitSettings
from reactive_lob_sim.tape.l2 import BookSeries, TradeSeries
from reactive_lob_sim.timeutil import NS_PER_S

HAWKES = HawkesParams(mu=[0.5, 0.3], alpha=[[0.3, 0.1], [0.2, 0.4]], beta=2.0)
FIT = HawkesFitSettings(
    init_beta=1.0,
    beta_min=0.01,
    beta_max=50.0,
    max_iter=500,
    max_spectral_radius=0.99,
    tolerance=1e-10,
    init_branching=0.5,
    param_floor=1e-6,
)


def test_hawkes_poisson_loglik_closed_form() -> None:
    p = HawkesParams(mu=[2.0], alpha=[[0.0]], beta=1.0)
    times = np.array([0.5, 1.0, 3.0])
    # Poisson: n log(mu) - mu T
    assert log_likelihood(p, times, np.zeros(3, dtype=np.int64), 4.0) == pytest.approx(3 * np.log(2.0) - 8.0)


def test_hawkes_gradient_matches_finite_differences() -> None:
    times, types = simulate_hawkes(HAWKES, 300.0, 1)
    mu, alpha, beta = HAWKES.mu_array(), HAWKES.alpha_array(), HAWKES.beta
    ll, g_mu, g_alpha, g_beta = kernels.loglik_grad(mu, alpha, beta, times, types, 300.0)
    h = 1e-6
    num_mu = (kernels.loglik_grad(mu + np.array([h, 0.0]), alpha, beta, times, types, 300.0)[0] - ll) / h
    a2 = alpha.copy()
    a2[1, 0] += h
    num_alpha = (kernels.loglik_grad(mu, a2, beta, times, types, 300.0)[0] - ll) / h
    num_beta = (kernels.loglik_grad(mu, alpha, beta + h, times, types, 300.0)[0] - ll) / h
    assert g_mu[0] == pytest.approx(num_mu, rel=1e-3)
    assert g_alpha[1, 0] == pytest.approx(num_alpha, rel=1e-3)
    assert g_beta == pytest.approx(num_beta, rel=1e-3)


def test_hawkes_mle_recovers_parameters() -> None:
    times, types = simulate_hawkes(HAWKES, 15_000.0, 3)
    rates = np.bincount(types) / 15_000.0
    np.testing.assert_allclose(rates, HAWKES.stationary_rates(), rtol=0.1)
    fit = fit_hawkes(times, types, 2, 15_000.0, FIT)
    assert fit.converged
    np.testing.assert_allclose(fit.params.mu, HAWKES.mu, rtol=0.15)
    np.testing.assert_allclose(fit.params.alpha, HAWKES.alpha, atol=0.06)
    assert fit.params.beta == pytest.approx(2.0, rel=0.15)
    assert np.isfinite(fit.aic)


def test_hawkes_regime_multiplier_and_errors() -> None:
    p = HawkesParams(mu=[1.0], alpha=[[0.0]], beta=1.0)
    t, _ = simulate_hawkes(p, 2000.0, 5, np.array([0.0, 1000.0]), np.array([1.0, 3.0]))
    assert (t >= 1000).sum() / (t < 1000).sum() == pytest.approx(3.0, rel=0.15)
    with pytest.raises(ValueError, match="non-stationary"):
        simulate_hawkes(HawkesParams(mu=[1.0], alpha=[[1.2]], beta=1.0), 10.0, 0)
    with pytest.raises(ValueError, match="breaks"):
        simulate_hawkes(p, 10.0, 0, np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="sorted"):
        fit_hawkes(np.array([2.0, 1.0, 3.0, 4.0]), np.zeros(4, dtype=np.int64), 1, 5.0, FIT)
    with pytest.raises(ValueError, match="square matrix"):
        HawkesParams(mu=[1.0], alpha=[[0.1, 0.2]], beta=1.0)


def test_hawkes_intensity_kernel() -> None:
    out = kernels.intensity_at(
        np.array([1.0]),
        np.array([[0.5]]),
        2.0,
        np.array([1.0]),
        np.zeros(1, dtype=np.int64),
        np.array([0.5, 2.0]),
    )
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(1.0 + 0.5 * 2.0 * np.exp(-2.0))


def impact(**kw: float) -> ImpactParams:
    base: dict[str, float | str] = {
        "Y": 0.5,
        "delta": 0.5,
        "sigma_window": 1e-3,
        "volume_window": 100.0,
        "window_s": 30.0,
        "tau0_s": 5.0,
        "beta": 0.5,
        "kappa": 1.0,
        "source": "all",
        "n_exponentials": 8,
    }
    base.update(kw)
    return ImpactParams.model_validate(base)


def test_exponential_sum_approximates_power_law() -> None:
    w, r = exponential_sum(5.0, 0.5, 8)
    tau = np.array([0.0, 1.0, 5.0, 50.0, 500.0])
    approx = np.exp(-np.outer(tau, r)) @ w
    np.testing.assert_allclose(approx, power_law_decay(tau, 5.0, 0.5), rtol=0.05)


def test_impact_state_square_root_and_decay() -> None:
    p = impact()
    state = ImpactState(p)
    state.add_trade(0.0, 1, 25.0, 100.0)
    peak = peak_impact(p, 25.0, 100.0)
    assert peak == pytest.approx(0.5 * 1e-3 * 100.0 * 0.5)
    assert state.value(0.0) == pytest.approx(peak, rel=0.05)
    assert state.value(5.0) == pytest.approx(peak * 2**-0.5, rel=0.05)
    off = ImpactState(impact(kappa=0.0))
    off.add_trade(0.0, 1, 25.0, 100.0)
    assert off.value(1.0) == 0.0


def test_fit_impact_recovers_square_root_law() -> None:
    rng = np.random.default_rng(0)
    n_w, w_s, v_w = 2000, 30.0, 100.0
    q = rng.normal(0, 1, n_w) * v_w * 0.5
    sigma = 1e-3
    true_y, true_delta = 0.8, 0.5
    r = np.sign(q) * true_y * sigma * (np.abs(q) / v_w) ** true_delta + rng.normal(0, sigma * 0.3, n_w)
    mid = 100.0 * np.cumprod(np.concatenate([[1.0], 1 + r]))
    ts = (np.arange(n_w + 1) * w_s * NS_PER_S).astype(np.int64)
    book = BookSeries(
        ts,
        mid / 0.01 - 0.5,
        mid / 0.01 + 0.5,
        np.ones(n_w + 1),
        np.ones(n_w + 1),
        np.ones(n_w + 1),
        np.ones(n_w + 1),
    )
    trade_ts = ts[:-1] + NS_PER_S
    trades = TradeSeries(trade_ts, mid[:-1], np.abs(q) + 1e-9, np.sign(q))
    fit_cfg = ImpactFitSettings(
        window_s=w_s,
        n_bins=8,
        lag_bin_s=w_s,
        max_lag_bins=5,
        delta_min=0.2,
        delta_max=1.0,
        beta_max=3.0,
        n_exponentials=8,
        kappa_grid=[0.0, 1.0],
        probe_duration_s=60.0,
        probe_volume_fraction=0.5,
        probe_child_interval_s=2.0,
        probe_seeds=1,
    )
    fit = fit_impact(book, trades, fit_cfg)
    assert fit.params.delta == pytest.approx(true_delta, abs=0.12)
    assert pytest.approx(true_y, rel=0.25) == fit.params.Y
    assert fit.params.kappa == 1.0


def test_fees_and_account_are_exact() -> None:
    fees = FeeSchedule.from_bps(-0.5, 4.0)
    assert fees.fee(100.1, 0.3, Liquidity.TAKER) == Decimal("0.012012")
    assert fees.fee(100.0, 1.0, Liquidity.MAKER) == Decimal("-0.005")
    acct = Account()
    acct.apply_fill(Side.BUY, 100.0, 2.0, Decimal("0.08"))
    acct.apply_fill(Side.SELL, 101.0, 1.0, Decimal("0.04"))
    assert acct.position == Decimal("1.0")
    assert acct.equity(102.0) == Decimal("102.0") - Decimal("200.0") + Decimal("101.0") - Decimal("0.12")
