"""The Rys rule to ~`digits` digits in `decimal`: a ground truth for
`rys.roots`, sharing nothing with its fp64 route but the measure's
definition (#468's exact route, `experimental/rys/roots.py`).

    roots_weights_exact(n, T, digits=40) -> (x, w)   Decimals, ascending
    boys_exact(mmax, T)                  -> [F_0 .. F_mmax]

The moments `F_m(T)` (the positive Kummer series below `T = 200`; above
it `F_0 = sqrt(pi / T) / 2`, `erfc(sqrt(T)) < e^-200` being below any
precision used, and the upward recurrence), Gautschi's Chebyshev algorithm
for the recurrence coefficients, Newton on the orthogonal polynomial and
the Christoffel numbers (`roots.gauss_decimal`). Used by the tests only.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

from felizjueves.rys.roots import PI, gauss_decimal

__all__ = ["boys_exact", "roots_weights_exact"]


def boys_exact(mmax: int, T: Decimal) -> list[Decimal]:
    """F_0..F_mmax at T in the current decimal context."""
    if T == 0:
        return [Decimal(1) / (2 * m + 1) for m in range(mmax + 1)]
    eps = Decimal(10) ** -(decimal.getcontext().prec + 5)
    emt = (-T).exp()
    if T <= 200:
        out = []
        for m in range(mmax + 1):
            term = Decimal(1) / (2 * m + 1)
            total = term
            k = 0
            while term > eps * total:
                k += 1
                term = term * 2 * T / (2 * m + 2 * k + 1)
                total += term
            out.append(emt * total)
        return out
    f = [(PI / T).sqrt() / 2]
    for m in range(mmax):
        f.append(((2 * m + 1) * f[m] - emt) / (2 * T))
    return f


def roots_weights_exact(n: int, T: float | Decimal, digits: int = 40
                        ) -> tuple[list[Decimal], list[Decimal]]:
    """The n-point Rys rule at `T` (a float is taken at its exact value)."""
    Td = Decimal(T) if not isinstance(T, Decimal) else T
    with decimal.localcontext() as ctx:
        ctx.prec = digits + 20
        mu = boys_exact(2 * n - 1, Td)
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
        return gauss_decimal(a, b, n, digits)
