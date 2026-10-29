"""The Rys rule in fp64, as the device computes it, for `N` in 1..`N_MAX`.

    roots_weights(n, T) -> (x, w)     nodes x in (0, 1), weights w, fp64
    device_text()       -> str        the `fj_rys<N>` routine the writer carries
    T_SWITCH, M_LEGENDRE, N_MAX, BISECTIONS, legendre_half(), laguerre(n)

The measure is `dmu(x) = (1/2) x^(-1/2) exp(-T x) dx` on `[0, 1]`, so the
N-point Gauss rule has `SUM_n w_n x_n^m = F_m(T)` for `m <= 2N - 1`, and a
primitive quartet of total angular momentum `L` needs `N = L // 2 + 1`
nodes. `N_MAX = 5` reaches `(dd|dd)`, the fused F ladder's top.

THE ROUTE (#468's spike, `experimental/rys/roots.py`, settled there for
`N <= 3` against a 50-digit route; widened here to `N <= 5` and made
device-shaped -- no eigensolver, no library, every loop a fixed count):

* `T >= T_SWITCH`: `x = s / T` turns the measure into generalised
  Gauss-Laguerre with `alpha = -1/2` on `[0, T]`; dropping `[T, inf)` leaves
  the rule `x_n = s_n / T`, `w_n = v_n / (2 sqrt(T))` on CONSTANTS `(s, v)`
  (`laguerre(n)`, computed once in `decimal` and rounded). The dropped tail
  is `O(e^-T T^(2N - 3/2))` relative: at `N = 5` it is ~1e-11 at T = 60 and
  ~1e-17 at T = 80, which is where the switch sits (#468's 50 held for
  `N <= 3` only, as its README says).
* `T < T_SWITCH`: with `x = t^2` the measure is `exp(-T t^2) dt` on
  `[0, 1]`; discretise it on the positive half of an `M_LEGENDRE`-point
  Gauss-Legendre rule (the nodes squared, `y_j`, and the weights `u_j`, both
  constants), `mu_j = u_j exp(-T y_j)`. The Stieltjes procedure gives the
  monic three-term coefficients `a_k`, `b_k` (each `p_k(y_j)` re-run from
  the coefficients so far, so nothing but `mu` is an array); the nodes are
  the Jacobi matrix's eigenvalues by Sturm bisection on `[0, 1]`
  (`BISECTIONS` halvings, the LDL^T pivot count), and the weights the
  Christoffel numbers `1 / SUM_k p_k(x)^2 / (b_0 ... b_k)`.

MEASURED (test/test_rys_roots.py, against `rys.exact` at 40 digits, every
`N`, `T` from 0 through `1e-300` to `1e6` and densely across the switch):
nodes and weights within 7e-15 relative. `M_LEGENDRE = 88` keeps margin:
72 misses by 1e-12 at `N = 5` just below the switch, 80 is the edge.
Newton polishing after the bisection was tried and bought nothing.

`roots_weights` is the device routine's operation order, statement for
statement, so the interpreter (`rys=` injected) and the card run one
algorithm; only `exp` and `sqrt` are the platform's. `device_text` writes
that routine with the same constants, each spelled by `repr` (shortest
round trip), so the card reads exactly the doubles the host does.
"""

from __future__ import annotations

import decimal
import math
from decimal import Decimal
from functools import cache

N_MAX = 5  # (dd|dd): L = 8
T_SWITCH = 80.0
M_LEGENDRE = 88  # points on [-1, 1]; the positive half, 44, is used
BISECTIONS = 64
TINY = 1e-300  # a zero Sturm pivot is replaced by this, the usual guard

# pi to 100 digits; decimal has no constant for it.
PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164"
    "062862089986280348253421170679"
)

__all__ = ["BISECTIONS", "M_LEGENDRE", "N_MAX", "PI", "T_SWITCH",
           "device_text", "gauss_decimal", "laguerre", "legendre_half",
           "n_roots", "roots_weights"]


def n_roots(L: int) -> int:
    """The rule's size for a primitive quartet of total angular momentum
    `L`: the integrand is a polynomial of degree `L` in `x`."""
    if type(L) is not int or L < 0:
        raise ValueError(f"L = {L!r} is not an int >= 0")
    return L // 2 + 1


# --- the constants, in decimal, rounded once -----------------------------------


def _legendre_p(x: Decimal, m: int) -> tuple[Decimal, Decimal]:
    """P_m(x) and P_m'(x)."""
    p0, p1 = Decimal(1), x
    for k in range(2, m + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1, m * (x * p1 - p0) / (x * x - 1)


@cache
def legendre_half() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`(y, u)`: the positive nodes `t` of the `M_LEGENDRE`-point
    Gauss-Legendre rule, SQUARED (`y = t^2`, the node in `x`) and their
    weights, Newton-polished at 40 digits and rounded once. numpy's tables
    carry ~1e-14 relative errors in single weights (#468)."""
    ys, us = [], []
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
            ys.append(float(t * t))
            us.append(float(2 / ((1 - t * t) * dp * dp)))
    return tuple(ys), tuple(us)


def _poly(x: Decimal, a, b, n):
    """p_0..p_n(x) (monic) and p_n'(x) by the three-term recurrence."""
    p = [Decimal(1), x - a[0]]
    dp = [Decimal(0), Decimal(1)]
    for k in range(1, n):
        p.append((x - a[k]) * p[k] - b[k] * p[k - 1])
        dp.append(p[k] + (x - a[k]) * dp[k] - b[k] * dp[k - 1])
    return p, dp[n]


def gauss_decimal(a, b, n: int, digits: int) -> tuple[list[Decimal],
                                                      list[Decimal]]:
    """The n-point Gauss rule of the monic recurrence `(a, b)` (`b[0]` the
    measure's mass) in the current-plus-`digits` decimal context: the nodes
    by Newton on `p_n` from the Jacobi matrix's float eigenvalues (bisection
    in floats, so no numpy), the weights as Christoffel numbers. Ascending."""
    guesses = _float_eigenvalues([float(v) for v in a],
                                 [float(v) for v in b], n)
    tol = Decimal(10) ** -(digits + 10)
    xs, ws = [], []
    for g in guesses:
        x = Decimal(g)
        for _ in range(200):
            p, dpn = _poly(x, a, b, n)
            step = p[n] / dpn
            x -= step
            if abs(step) <= tol * abs(x):
                break
        else:
            raise RuntimeError(f"Newton did not converge: n={n}")
        p, _ = _poly(x, a, b, n)
        h, s = Decimal(1), Decimal(0)
        for k in range(n):
            h = h * b[k]  # <p_k, p_k> = b_0 b_1 ... b_k
            s += p[k] * p[k] / h
        xs.append(+x)
        ws.append(1 / s)
    if len({str(x) for x in xs}) != n:
        raise RuntimeError(f"Newton merged two roots: n={n}")
    return xs, ws


def _float_eigenvalues(a: list[float], b: list[float], n: int) -> list[float]:
    """The Jacobi matrix's eigenvalues, ascending, by Sturm bisection on a
    Gershgorin interval: a starting guess, refined in decimal after."""
    off = [math.sqrt(v) for v in b[1:n]] + [0.0]
    lo = min(a[k] - off[k] - (off[k - 1] if k else 0.0) for k in range(n))
    hi = max(a[k] + off[k] + (off[k - 1] if k else 0.0) for k in range(n))
    out = []
    for i in range(n):
        L, H = lo, hi
        for _ in range(200):
            mid = 0.5 * (L + H)
            if _sturm(a, b, n, mid) > i:
                H = mid
            else:
                L = mid
        out.append(0.5 * (L + H))
    return out


@cache
def laguerre(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`(s, v)`: the n-point generalised Gauss-Laguerre rule, `alpha =
    -1/2` (`a_k = 2k + 1/2`, `b_0 = sqrt(pi)`, `b_k = k (k - 1/2)`), at 40
    digits, rounded once. Ascending."""
    _check_n(n)
    with decimal.localcontext() as ctx:
        ctx.prec = 60
        a = [Decimal(2 * k) + Decimal("0.5") for k in range(n)]
        b = [PI.sqrt()] + [Decimal(k) * (Decimal(k) - Decimal("0.5"))
                           for k in range(1, n)]
        xs, ws = gauss_decimal(a, b, n, 40)
        return tuple(float(x) for x in xs), tuple(float(w) for w in ws)


def _check_n(n: int) -> None:
    if type(n) is not int or not 1 <= n <= N_MAX:
        raise ValueError(f"n = {n!r} outside 1..{N_MAX}")


# --- the fp64 rule: the device routine, statement for statement ----------------


def _sturm(a, b, n: int, x: float) -> int:
    """How many eigenvalues of the Jacobi matrix lie below `x`: the
    negative pivots of `J - x I = L D L^T`."""
    c = 0
    d = a[0] - x
    if d < 0.0:
        c += 1
    for k in range(1, n):
        if d == 0.0:
            d = TINY
        d = (a[k] - x) - b[k] / d
        if d < 0.0:
            c += 1
    return c


def roots_weights(n: int, T: float) -> tuple[tuple[float, ...],
                                             tuple[float, ...]]:
    """The n-point Rys rule at `T`, fp64: nodes ascending, weights. The
    device's `fj_rys<n>` in the same operation order (module docstring)."""
    _check_n(n)
    T = float(T)
    if not T >= 0.0 or math.isinf(T):
        raise ValueError(f"T = {T!r} is not a finite T >= 0")
    if T >= T_SWITCH:
        s, v = laguerre(n)
        h = 2.0 * math.sqrt(T)
        return tuple(x / T for x in s), tuple(w / h for w in v)
    y, u = legendre_half()
    mu = [u[j] * math.exp(-T * y[j]) for j in range(len(y))]
    a = [0.0] * n
    b = [0.0] * n
    norm_prev = 1.0
    for k in range(n):
        nrm, xn = 0.0, 0.0
        for j in range(len(y)):
            yj = y[j]
            p0, p1 = 0.0, 1.0
            for m in range(k):
                pn = (yj - a[m]) * p1 - b[m] * p0
                p0, p1 = p1, pn
            pp = mu[j] * p1 * p1
            nrm = nrm + pp
            xn = xn + pp * yj
        a[k] = xn / nrm
        b[k] = nrm if k == 0 else nrm / norm_prev
        norm_prev = nrm
    x = [0.0] * n
    for i in range(n):
        lo, hi = 0.0, 1.0
        for _ in range(BISECTIONS):
            mid = 0.5 * (lo + hi)
            if _sturm(a, b, n, mid) > i:
                hi = mid
            else:
                lo = mid
        x[i] = 0.5 * (lo + hi)
    w = [0.0] * n
    for i in range(n):
        p0, p1, h, s = 0.0, 1.0, 1.0, 0.0
        for k in range(n):
            h = h * b[k]
            s = s + p1 * p1 / h
            pn = (x[i] - a[k]) * p1 - b[k] * p0
            p0, p1 = p1, pn
        w[i] = 1.0 / s
    return tuple(x), tuple(w)


# --- the device routine's text -------------------------------------------------


def _table(name: str, values) -> str:
    body = ",\n".join("    " + ", ".join(repr(float(v)) for v in values[i:i + 3])
                      for i in range(0, len(values), 3))
    return f"__constant__ double {name}[{len(values)}] = {{\n{body}\n}};\n"


def device_text(*, prefix: str = "fj_rys", qualifier: str = "__device__"
                ) -> str:
    """`fj_rys<N>(T, x, w)`, the Rys rule as `roots_weights` computes it,
    with its constant tables: one text, carried by every unit that holds an
    `ir.Rys` (emit/_c.py). `N` above `N_MAX` is a compile-time refusal."""
    y, u = legendre_half()
    mh = len(y)
    ls = [x for n in range(1, N_MAX + 1) for x in laguerre(n)[0]]
    lw = [x for n in range(1, N_MAX + 1) for x in laguerre(n)[1]]
    up = prefix.upper()
    return (
        f"// The Rys rule (felizjueves.rys.roots): the N nodes x in (0, 1) and\n"
        f"// weights w of (1/2) x^(-1/2) exp(-T x) dx on [0, 1], SUM w x^m =\n"
        f"// F_m(T) for m <= 2N - 1. From T = {T_SWITCH!r}: generalised\n"
        f"// Gauss-Laguerre (alpha = -1/2) on constants, x = s / T, w = v /\n"
        f"// (2 sqrt(T)). Below: the measure on the positive half of a\n"
        f"// {M_LEGENDRE}-point Gauss-Legendre rule in t = sqrt(x), Stieltjes for the\n"
        f"// three-term coefficients, Sturm bisection ({BISECTIONS} halvings) for the\n"
        f"// nodes, Christoffel numbers for the weights.\n"
        + _table(f"{up}_Y", y) + _table(f"{up}_U", u)
        + _table(f"{up}_LS", ls) + _table(f"{up}_LW", lw) +
        f"template <int N>\n"
        f"{qualifier} __forceinline__ void {prefix}(double T, double (&x)[N], "
        f"double (&w)[N]) {{\n"
        f"    static_assert(N >= 1 && N <= {N_MAX}, \"{prefix}: N in 1..{N_MAX}\");\n"
        f"    if (T >= {T_SWITCH!r}) {{\n"
        f"        const int o = (N - 1) * N / 2;\n"
        f"        const double h = 2.0 * sqrt(T);\n"
        f"        for (int i = 0; i < N; ++i) {{\n"
        f"            x[i] = {up}_LS[o + i] / T;\n"
        f"            w[i] = {up}_LW[o + i] / h;\n"
        f"        }}\n"
        f"        return;\n"
        f"    }}\n"
        f"    double mu[{mh}];\n"
        f"    for (int j = 0; j < {mh}; ++j) {{\n"
        f"        mu[j] = {up}_U[j] * exp(-T * {up}_Y[j]);\n"
        f"    }}\n"
        f"    double a[N];\n"
        f"    double b[N];\n"
        f"    double norm_prev = 1.0;\n"
        f"    for (int k = 0; k < N; ++k) {{\n"
        f"        double nrm = 0.0;\n"
        f"        double xn = 0.0;\n"
        f"        for (int j = 0; j < {mh}; ++j) {{\n"
        f"            const double yj = {up}_Y[j];\n"
        f"            double p0 = 0.0;\n"
        f"            double p1 = 1.0;\n"
        f"            for (int m = 0; m < k; ++m) {{\n"
        f"                const double pn = (yj - a[m]) * p1 - b[m] * p0;\n"
        f"                p0 = p1;\n"
        f"                p1 = pn;\n"
        f"            }}\n"
        f"            const double pp = mu[j] * p1 * p1;\n"
        f"            nrm = nrm + pp;\n"
        f"            xn = xn + pp * yj;\n"
        f"        }}\n"
        f"        a[k] = xn / nrm;\n"
        f"        b[k] = (k == 0) ? nrm : nrm / norm_prev;\n"
        f"        norm_prev = nrm;\n"
        f"    }}\n"
        f"    for (int i = 0; i < N; ++i) {{\n"
        f"        double lo = 0.0;\n"
        f"        double hi = 1.0;\n"
        f"        for (int it = 0; it < {BISECTIONS}; ++it) {{\n"
        f"            const double mid = 0.5 * (lo + hi);\n"
        f"            int c = 0;\n"
        f"            double d = a[0] - mid;\n"
        f"            if (d < 0.0) {{\n"
        f"                ++c;\n"
        f"            }}\n"
        f"            for (int k = 1; k < N; ++k) {{\n"
        f"                if (d == 0.0) {{\n"
        f"                    d = {TINY!r};\n"
        f"                }}\n"
        f"                d = (a[k] - mid) - b[k] / d;\n"
        f"                if (d < 0.0) {{\n"
        f"                    ++c;\n"
        f"                }}\n"
        f"            }}\n"
        f"            if (c > i) {{\n"
        f"                hi = mid;\n"
        f"            }} else {{\n"
        f"                lo = mid;\n"
        f"            }}\n"
        f"        }}\n"
        f"        x[i] = 0.5 * (lo + hi);\n"
        f"    }}\n"
        f"    for (int i = 0; i < N; ++i) {{\n"
        f"        double p0 = 0.0;\n"
        f"        double p1 = 1.0;\n"
        f"        double h = 1.0;\n"
        f"        double s = 0.0;\n"
        f"        for (int k = 0; k < N; ++k) {{\n"
        f"            h = h * b[k];\n"
        f"            s = s + p1 * p1 / h;\n"
        f"            const double pn = (x[i] - a[k]) * p1 - b[k] * p0;\n"
        f"            p0 = p1;\n"
        f"            p1 = pn;\n"
        f"        }}\n"
        f"        w[i] = 1.0 / s;\n"
        f"    }}\n"
        f"}}\n"
    )
