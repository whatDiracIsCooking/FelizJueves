"""The Boys function F_m(T), by quadrature -- the reference a kernel's Boys is
checked against.

    F_m(T) = integral_0^1 t^(2m) exp(-t^2 T) dt      ([JEngine, v2, eqs. 13, 18])

evaluated by Gauss-Legendre on [0, 1] in float64, and by nothing else.

WHY QUADRATURE AND NOT A RECURRENCE. The downward recurrence

    F_m(T) = (2T F_{m+1}(T) + exp(-T)) / (2m + 1)

is what a kernel will use, so it cannot also be the reference for it: a
reference with the same ancestry as the thing under test agrees with it by
construction. The test file uses that recurrence as a consistency check *on
this quadrature*, never as its implementation.

CONVERGENCE, MEASURED. Against the Kummer series

    F_m(T) = exp(-T) sum_k (2T)^k / ((2m+1)(2m+3)...(2m+2k+1))

summed in 50-digit `decimal` (an independent closed form, exact to well below
float64), the worst relative error over the grid

    m in 0..40,  T in {0, 1e-6, 1e-3, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 75,
                       100, 150, 200, 300, 500}

is, by node count (m <= 16 is the range the plan layer reaches: L <= 16 in
[JEngine] Tables 1-3, and the fundamentals of a class run m = 0..L):

    nodes   m <= 16    m <= 40
      64    1.3e-12    1.9e-09   (m = 38, T = 500: not converged)
      72    1.5e-15    1.0e-11   (m = 40, T = 500: not converged)
      80    9.6e-16    3.2e-14
      88    1.1e-15    3.1e-15
     112    1.1e-15    1.4e-15   <- NODES, the default
     128    3.2e-15    7.3e-15   (rounding of t^80 near t = 1, not truncation)

so truncation is gone by 88 nodes and the default keeps 24 in hand; past that
the error is float64 rounding and more nodes only add to it. The weights
matter as much as the count: with numpy's `leggauss` weights unpolished the
floor was ~1e-13 at every count (see `_rule`). `test/test_reference_boys.py`
re-measures the default on that grid and on off-grid points, and asserts
under 1e-14.

The plan layer has no T at all (it counts; `[JEngine, v2, §7.1]` excludes the
Boys side), so the T range is ours: `T_MAX = 500` is where the measurement
stops, and `boys` refuses T outside [0, T_MAX] and m outside 0..M_MAX rather
than return an unmeasured number. Large T is where Gauss-Legendre on one
panel degrades (the integrand's width is ~1/sqrt(T)); widening the domain means
re-measuring, not just raising the constant.

float64 throughout. The device's precision is the kernel tier's business.
Nothing that emits kernel text may import this module.
"""

from __future__ import annotations

from functools import cache

import numpy as np

NODES = 112
M_MAX = 40
T_MAX = 500.0


def _legendre(n: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P_n(x) and P_n'(x), by Bonnet's three-term recurrence."""
    p0, p1 = np.ones_like(x), x
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1, n * (x * p1 - p0) / (x * x - 1.0)


@cache
def _rule(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights mapped from [-1, 1] onto [0, 1].

    numpy's `leggauss` is only the starting guess: its weights carry ~1e-13
    relative error at these node counts, which was the whole error floor of
    the first measurement. Two Newton steps on P_n polish the nodes, and the
    weights are recomputed from them as 2 / ((1 - x^2) P_n'(x)^2).
    """
    x, _ = np.polynomial.legendre.leggauss(nodes)
    for _ in range(2):
        p, dp = _legendre(nodes, x)
        x = x - p / dp
    _, dp = _legendre(nodes, x)
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    return 0.5 * (x + 1.0), 0.5 * w


def boys(m: int, T: float, *, nodes: int = NODES) -> float:
    """F_m(T) = integral_0^1 t^(2m) exp(-t^2 T) dt, by Gauss-Legendre on [0, 1].

    Quadrature, not a recurrence: the downward recurrence is what a kernel will
    use, so it cannot also be the reference for it.
    """
    if type(m) is not int or not 0 <= m <= M_MAX:
        raise ValueError(f"m = {m!r} outside the measured range 0..{M_MAX}")
    if not 0.0 <= T <= T_MAX:
        raise ValueError(f"T = {T!r} outside the measured range [0, {T_MAX}]")
    t, w = _rule(nodes)
    return float(np.dot(w, t ** (2 * m) * np.exp(-(t * t) * T)))
