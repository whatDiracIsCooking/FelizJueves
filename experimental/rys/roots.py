"""Rys roots and weights for N <= 3, by two independent routes.

The measure is dmu(x) = (1/2) x^(-1/2) exp(-T x) dx on [0, 1], so that the
N-point Gauss rule (x_n, W_n) has SUM_n W_n x_n^m == F_m(T) for m <= 2N - 1.

Shipping route (fp64), ``roots_weights(n, T)``:

* T < T_SWITCH: substitute x = t^2, which turns dmu into exp(-T t^2) dt on
  t in [0, 1]. Discretise that with the positive half of an M-point
  Gauss-Legendre rule on [-1, 1] (exact to rounding here: the integrand is a
  Gaussian of width >= 1/sqrt(T_SWITCH) times a polynomial), run the Stieltjes
  procedure on the discrete measure for the three-term recurrence of the monic
  orthogonal polynomials in x, and take nodes and weights from the Jacobi
  matrix (Golub-Welsch). The Legendre table is computed once in ``decimal``
  and rounded: numpy's and scipy's tables carry ~1e-14 relative errors in
  individual weights, erratic in M, and those were the whole of this route's
  error (up to 1.5e-13 in the Rys weights) until they were replaced.
* T >= T_SWITCH: with x = s / T the measure is T^(-1/2) (1/2) s^(-1/2) e^(-s)
  ds on [0, T]. Dropping [T, inf) leaves generalised Gauss-Laguerre with
  alpha = -1/2, so x_n = s_n / T and W_n = w_n / (2 sqrt(T)). The dropped tail
  is O(e^-T T^(2N - 3/2)) relative: about 2e-16 of F_5 at T = 50, less for
  every lower moment and every larger T.

The large-T branch is needed: with M fixed, the Legendre discretisation cannot
resolve a Gaussian of width 1/sqrt(T) once T is large (at T = 1e6 the whole
measure sits inside the first Legendre node). ``check.py`` measures both sides
of the switch.

Exact route, ``roots_weights_exact(n, T, digits)``: the moments F_m(T) in
``decimal`` (the Taylor series below T = 200, the asymptotic F_0 plus upward
recurrence above it), Gautschi's Chebyshev algorithm for the recurrence
coefficients, Newton on the orthogonal polynomial for the nodes, and the
Christoffel numbers for the weights. It shares nothing with the fp64 route but
the definition of the measure.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from functools import cache

import numpy as np
from scipy.special import roots_genlaguerre

N_MAX = 3
T_MAX = 1e6
T_SWITCH = 50.0
M_LEGENDRE = 80  # 48 misses by 1e-10 near T_SWITCH, 56 and up do not

# pi to 100 digits; decimal has no constant for it.
_PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164"
    "062862089986280348253421170679"
)


def _legendre_p(x: Decimal, m: int) -> tuple[Decimal, Decimal]:
    """P_m(x) and P_m'(x)."""
    p0, p1 = Decimal(1), x
    for k in range(2, m + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1, m * (x * p1 - p0) / (x * x - 1)


@cache
def _legendre_t() -> tuple[np.ndarray, np.ndarray]:
    """The positive nodes t of the M-point Gauss-Legendre rule on [-1, 1] and
    their weights, Newton-polished at 40 digits and rounded once. On the
    positive half these weights integrate over [0, 1]."""
    ts, ws = [], []
    with decimal.localcontext() as ctx:
        ctx.prec = 40
        tol = Decimal(10) ** -35
        for i in range(1, M_LEGENDRE // 2 + 1):
            t = Decimal(math.cos(math.pi * (i - 0.25) / (M_LEGENDRE + 0.5)))
            for _ in range(100):
                p, dp = _legendre_p(t, M_LEGENDRE)
                t -= p / dp
                if abs(p / dp) < tol:
                    break
            _, dp = _legendre_p(t, M_LEGENDRE)
            ts.append(float(t))
            ws.append(float(2 / ((1 - t * t) * dp * dp)))
    return np.array(ts), np.array(ws)


def _gauss_from_discrete(n: int, x: np.ndarray, w: np.ndarray):
    """Stieltjes procedure on a discrete measure, then Golub-Welsch."""
    a = np.zeros(n)
    b = np.zeros(n)
    p_prev = np.zeros_like(x)
    p = np.ones_like(x)
    norm_prev = 1.0
    for k in range(n):
        norm = np.dot(w, p * p)
        a[k] = np.dot(w, x * p * p) / norm
        b[k] = norm if k == 0 else norm / norm_prev
        p_next = (x - a[k]) * p - (b[k] * p_prev if k else 0.0)
        p_prev, p, norm_prev = p, p_next, norm
    jac = np.diag(a) + np.diag(np.sqrt(b[1:]), 1) + np.diag(np.sqrt(b[1:]), -1)
    nodes, vecs = np.linalg.eigh(jac)
    return nodes, b[0] * vecs[0] ** 2


def roots_weights(n: int, T: float) -> tuple[np.ndarray, np.ndarray]:
    """The n-point Rys rule at T, fp64: nodes x in (0, 1), weights W."""
    if not 1 <= n <= N_MAX:
        raise ValueError(f"n = {n} outside 1..{N_MAX}")
    if not 0.0 <= T <= T_MAX:
        raise ValueError(f"T = {T} outside [0, {T_MAX:g}]")
    if T >= T_SWITCH:
        s, w = roots_genlaguerre(n, -0.5)
        return s / T, w / (2.0 * np.sqrt(T))
    t, wt = _legendre_t()
    return _gauss_from_discrete(n, t * t, wt * np.exp(-T * t * t))


# --- the exact route --------------------------------------------------------


def boys_exact(mmax: int, T: Decimal) -> list[Decimal]:
    """F_0..F_mmax at T in the current decimal context."""
    if T == 0:
        return [Decimal(1) / (2 * m + 1) for m in range(mmax + 1)]
    eps = Decimal(10) ** -(decimal.getcontext().prec + 5)
    if T <= 200:
        out = []
        emt = (-T).exp()
        for m in range(mmax + 1):
            # F_m = e^-T SUM_k (2T)^k / ((2m+1)(2m+3)...(2m+2k+1)), all positive.
            term = Decimal(1) / (2 * m + 1)
            total = term
            k = 0
            while term > eps * total:
                k += 1
                term = term * 2 * T / (2 * m + 2 * k + 1)
                total += term
            out.append(emt * total)
        return out
    # erfc(sqrt(T)) < e^-200 is below any precision used here: F_0 is
    # sqrt(pi / T) / 2, and the upward recurrence is stable for T > mmax.
    emt = (-T).exp()
    f = [(_PI / T).sqrt() / 2]
    for m in range(mmax):
        f.append(((2 * m + 1) * f[m] - emt) / (2 * T))
    return f


def _poly(x: Decimal, a, b, n):
    """p_0..p_n(x) and p_n'(x) by the three-term recurrence."""
    p = [Decimal(1), x - a[0]]
    dp = [Decimal(0), Decimal(1)]
    for k in range(1, n):
        p.append((x - a[k]) * p[k] - b[k] * p[k - 1])
        dp.append(p[k] + (x - a[k]) * dp[k] - b[k] * dp[k - 1])
    return p, dp[n]


def roots_weights_exact(n: int, T: float, digits: int = 50):
    """The n-point Rys rule at T to ~``digits`` digits, as Decimals."""
    return roots_weights_exact_decimal(n, Decimal(T), digits)  # T's exact value


def roots_weights_exact_decimal(n: int, Td: Decimal, digits: int = 50):
    """``roots_weights_exact`` at a Decimal T."""
    with decimal.localcontext() as ctx:
        ctx.prec = digits + 20
        mu = boys_exact(2 * n - 1, Td)
        # Gautschi's Chebyshev algorithm: sigma[k][l] = <p_k, x^l>.
        a = [mu[1] / mu[0]]
        b = [mu[0]]
        sig_prev = [Decimal(0)] * (2 * n)
        sig = list(mu)
        for k in range(1, n):
            new = [Decimal(0)] * (2 * n)
            for ll in range(k, 2 * n - k):
                new[ll] = sig[ll + 1] - a[k - 1] * sig[ll] - b[k - 1] * sig_prev[ll]
            a.append(new[k + 1] / new[k] - sig[k] / sig[k - 1])
            b.append(new[k] / sig[k - 1])
            sig_prev, sig = sig, new
        # Starting guesses from the Jacobi matrix in floats, then Newton.
        jac = np.diag([float(v) for v in a])
        off = [float(v.sqrt()) for v in b[1:]]
        jac += np.diag(off, 1) + np.diag(off, -1)
        guesses = np.linalg.eigvalsh(jac)
        tol = Decimal(10) ** -(digits + 10)
        xs, ws = [], []
        for g in guesses:
            x = Decimal(float(g))
            for _ in range(200):
                p, dpn = _poly(x, a, b, n)
                step = p[n] / dpn
                x -= step
                if abs(step) <= tol * abs(x):
                    break
            else:
                raise RuntimeError(f"Newton did not converge: n={n} T={Td}")
            p, _ = _poly(x, a, b, n)
            h = Decimal(1)
            s = Decimal(0)
            for k in range(n):
                h = h * b[k]  # h_k = <p_k, p_k> = b_0 b_1 ... b_k
                s += p[k] * p[k] / h
            xs.append(+x)
            ws.append(1 / s)
        if len({str(x) for x in xs}) != n:
            raise RuntimeError(f"Newton merged two roots: n={n} T={Td}")
        return xs, ws
