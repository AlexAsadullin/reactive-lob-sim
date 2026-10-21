"""Numba kernels for the multivariate exponential Hawkes process.

Kernel: phi_ij(t) = alpha_ij * beta * exp(-beta t). With the state S_j(t) = sum_{t_k^j < t} beta exp(-beta (t -
t_k^j)), the intensity is lambda_i(t) = mu_i m(t) + sum_j alpha_ij S_j(t), and S is updated recursively
(Ogata 1981; Ozaki 1979), so both simulation and the likelihood cost O(n D) for n events in D dimensions.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def simulate(mu, alpha, beta, horizon, breaks, mult, seed):  # type: ignore[no-untyped-def]  # pragma: no cover
    """Ogata thinning on ``[0, horizon)``.

    ``breaks`` (ascending, starting at 0) and ``mult`` define a piecewise-constant baseline multiplier
    m(t) = mult[k] for breaks[k] <= t < breaks[k+1]. Returns event times and types.
    """
    np.random.seed(seed)
    d = mu.shape[0]
    s = np.zeros(d)
    times = np.empty(1024)
    types = np.empty(1024, dtype=np.int64)
    n = 0
    t = 0.0
    seg = 0
    nseg = breaks.shape[0]
    mmax = mult.max()
    while True:
        lam_bar = mmax * mu.sum()
        for i in range(d):
            for j in range(d):
                lam_bar += alpha[i, j] * s[j]
        if lam_bar <= 0.0:
            break
        w = np.random.exponential(1.0 / lam_bar)
        decay = np.exp(-beta * w)
        for j in range(d):
            s[j] *= decay
        t += w
        if t >= horizon:
            break
        while seg + 1 < nseg and breaks[seg + 1] <= t:
            seg += 1
        m = mult[seg]
        lam = np.empty(d)
        total = 0.0
        for i in range(d):
            li = mu[i] * m
            for j in range(d):
                li += alpha[i, j] * s[j]
            lam[i] = li
            total += li
        u = np.random.random() * lam_bar
        if u >= total:
            continue
        acc = 0.0
        k = d - 1
        for i in range(d):
            acc += lam[i]
            if u < acc:
                k = i
                break
        if n == times.shape[0]:
            times = np.concatenate((times, np.empty(n)))
            types = np.concatenate((types, np.empty(n, dtype=np.int64)))
        times[n] = t
        types[n] = k
        n += 1
        s[k] += beta
    return times[:n].copy(), types[:n].copy()


@njit(cache=True)
def loglik_grad(mu, alpha, beta, times, types, horizon):  # type: ignore[no-untyped-def]  # pragma: no cover
    """Log-likelihood and its exact gradient with respect to (mu, alpha, beta).

    log L = sum_k log lambda_{c_k}(t_k) - sum_i [mu_i T + sum_j alpha_ij sum_{t^j} (1 - exp(-beta (T - t^j)))].

    With A_j = sum exp(-beta tau) and B_j = sum tau exp(-beta tau) over past type-j events at lags tau,
    S_j = beta A_j and dS_j/dbeta = A_j - beta B_j; both follow the recursion A <- A e^{-beta dt},
    B <- (B + dt A) e^{-beta dt}.
    """
    d = mu.shape[0]
    a = np.zeros(d)
    b = np.zeros(d)
    g_mu = np.zeros(d)
    g_alpha = np.zeros((d, d))
    g_beta = 0.0
    ll = 0.0
    last = 0.0
    n = times.shape[0]
    for k in range(n):
        dt = times[k] - last
        e = np.exp(-beta * dt)
        for j in range(d):
            b[j] = (b[j] + dt * a[j]) * e
            a[j] = a[j] * e
        last = times[k]
        i = types[k]
        lam = mu[i]
        dlam_beta = 0.0
        for j in range(d):
            lam += alpha[i, j] * beta * a[j]
            dlam_beta += alpha[i, j] * (a[j] - beta * b[j])
        if lam <= 0.0:
            return -np.inf, g_mu, g_alpha, g_beta
        ll += np.log(lam)
        inv = 1.0 / lam
        g_mu[i] += inv
        for j in range(d):
            g_alpha[i, j] += beta * a[j] * inv
        g_beta += dlam_beta * inv
        a[i] += 1.0
    comp = np.zeros(d)
    dcomp = np.zeros(d)
    for k in range(n):
        tau = horizon - times[k]
        e = np.exp(-beta * tau)
        comp[types[k]] += 1.0 - e
        dcomp[types[k]] += tau * e
    for i in range(d):
        ll -= mu[i] * horizon
        g_mu[i] -= horizon
        for j in range(d):
            ll -= alpha[i, j] * comp[j]
            g_alpha[i, j] -= comp[j]
            g_beta -= alpha[i, j] * dcomp[j]
    return ll, g_mu, g_alpha, g_beta


@njit(cache=True)
def intensity_at(mu, alpha, beta, times, types, query):  # type: ignore[no-untyped-def]  # pragma: no cover
    """Intensity vector just before each (ascending) query time, for diagnostics and tests."""
    d = mu.shape[0]
    out = np.empty((query.shape[0], d))
    s = np.zeros(d)
    last = 0.0
    k = 0
    n = times.shape[0]
    for q in range(query.shape[0]):
        tq = query[q]
        while k < n and times[k] < tq:
            dt = times[k] - last
            e = np.exp(-beta * dt)
            for j in range(d):
                s[j] *= e
            s[types[k]] += beta
            last = times[k]
            k += 1
        e = np.exp(-beta * (tq - last))
        for i in range(d):
            li = mu[i]
            for j in range(d):
                li += alpha[i, j] * s[j] * e
            out[q, i] = li
    return out
