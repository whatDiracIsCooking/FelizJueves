"""The host helpers the AO layer's matrix builds share (#503).

    checked_density(basis, D)       -> rows   D as n x n floats, checked
    checked_tau(tau)                -> float  a screening threshold, checked
    epair(sa, sb, recs=None)        -> EPair  one shell pair's dressed records
    make_kets(basis, rows, pairs, *, pq, tau) -> [Ket]
                                              ket records, `D_q` pre-contracted
    pair_bound_pq(zeta, L, *, pq)   -> float  the record bound through `pq`

Extracted from `ao/j.py`, where they were born and where their contracts are
still stated (that module's docstring: "THE DENSITY MUST BE SYMMETRIC", "THE
SCREEN, `tau`", "THE `D_q` AND `J_ab` SEAMS" for `EPair`, "THE SORTED
`break`" for `Ket.bound`). They live here so that `ao.k`, `ao.fock`,
`ao.schwarz`, `ao.staged`, `ao.drive` and the launch plans take them from a
shared module instead of from the J reference. Every body is moved verbatim:
the arithmetic, the operation order and the error messages are `ao.j`'s, and
`ao.j` imports them back.

`pair_bound_pq` is J's record bound and `ao.j` re-exports it under the same
name; it is defined here only because `make_kets` evaluates it, and a
definition left in `ao.j` would make the two modules import each other.

MEMOISE NOTHING (`ao.j`'s rule): every record is built per call and dropped.

Pure Python floats; `D` is anything indexable as `D[m][n]`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from felizjueves.ao.pairs import PrimitivePair, dress, primitive_pairs
from felizjueves.ao.shell import Basis, Shell
from felizjueves.mmd.hermite import hermite_order

Vec = tuple[float, float, float]
Ang = tuple[int, int, int]
PQ = Callable[[Ang, Ang, Vec, float], float]
Matrix = tuple[tuple[float, ...], ...]
ERec = tuple[Vec, float, float]  # (P, zeta, s = c_a c_b (pi/zeta)^{3/2})


@dataclass(frozen=True)
class EPair:
    """One shell pair as the `dq=` / `jab=` seams receive it (`ao.j`'s module
    docstring): `(la, lb)`, the centres, and per primitive pair, in
    `primitive_pairs` order, `(P, zeta, s)` with `s` from `pairs.dress`."""

    la: int
    lb: int
    A: Vec
    B: Vec
    recs: tuple[ERec, ...]


_ORIGIN = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Ket:
    """One ket index record: `Q`, `eta`, the weighted dressed `D_q` (and its
    |.| twin), its Hermite indices, and the screen's `I_ket * Dmax`."""

    Q: Vec
    eta: float
    qs: tuple[Ang, ...]
    dq: tuple[float, ...]
    dq_abs: tuple[float, ...]
    bound: float  # I_ket * Dmax; 0.0 (unused) when tau == 0


def checked_density(basis: Basis, D: object) -> list[list[float]]:
    n = basis.n_ao
    try:
        rows = [[float(D[i][j]) for j in range(n)] for i in range(n)]  # type: ignore[index]
        extra = len(D) != n or any(len(D[i]) != n for i in range(n))  # type: ignore[arg-type,index]
    except (TypeError, IndexError, KeyError):
        raise ValueError(f"D is not an {n} x {n} matrix") from None
    if extra:
        raise ValueError(f"D is not an {n} x {n} matrix")
    if not all(math.isfinite(x) for r in rows for x in r):
        raise ValueError("D has a non-finite entry")
    for i in range(n):
        for j in range(i):
            if rows[i][j] != rows[j][i]:
                raise ValueError(
                    f"D is not symmetric: D[{i}][{j}] = {rows[i][j]!r} but "
                    f"D[{j}][{i}] = {rows[j][i]!r}. An asymmetric density is "
                    "rejected, not symmetrised (ao/j.py's module docstring)")
    return rows


def checked_tau(tau: float) -> float:
    if not (isinstance(tau, int | float) and math.isfinite(tau) and tau >= 0.0):
        raise ValueError(f"tau = {tau!r} is not a finite float >= 0")
    return float(tau)


def pair_bound_pq(zeta: float, L: int, *, pq: PQ) -> float:
    """`I = sum_p sqrt([p|p](PQ = 0, theta = sqrt(zeta/2)))` over
    `hermite_order(L)` -- `ao.pairs.pair_bound` through the `[p|q]` evaluator
    (`ao.j`'s module docstring). Raises on a negative radicand, never `abs`es
    it."""
    theta = math.sqrt(zeta / 2.0)
    out = []
    for p in hermite_order(L):
        s = pq(p, p, _ORIGIN, theta)
        if s < 0.0:
            raise ValueError(
                f"[p|p](0) = {s!r} < 0 at p = {p}, zeta = {zeta!r}: a diagonal "
                "of a positive-definite form, so the evaluator's sign "
                "convention is wrong")
        out.append(math.sqrt(s))
    return math.fsum(out)


def make_kets(basis: Basis, D: list[list[float]], pairs, *, pq: PQ,
              tau: float) -> list[Ket]:
    """The ket records of the shell pairs `pairs` = [(c, d, w)], `D_q`
    pre-contracted with weight `w`."""
    shells, off = basis.shells, basis.offsets
    out = []
    for c, d, w in pairs:
        sc, sd = shells[c], shells[d]
        # D_cd over this pair's Cartesian pairs, c outer: the block's row order.
        dcd = [D[off[c] + i][off[d] + j]
               for i in range(sc.n_cart) for j in range(sd.n_cart)]
        qs = hermite_order(sc.l + sd.l)
        for k in primitive_pairs(sc, sd):
            dq = [0.0] * len(qs)
            da = [0.0] * len(qs)
            for x, row in zip(dcd, k.block, strict=True):
                if x:
                    ax = abs(x)
                    for t, e in enumerate(row):
                        dq[t] += x * e
                        da[t] += ax * abs(e)
            dq = tuple(w * v for v in dq)
            da = tuple(w * v for v in da)
            bound = 0.0
            if tau > 0.0:
                bound = (pair_bound_pq(k.zeta, k.L, pq=pq)
                         * max(abs(v) for v in dq))
            out.append(Ket(k.P, k.zeta, qs, dq, da, bound))
    return out


def epair(sa: Shell, sb: Shell,
          recs: Sequence[PrimitivePair] | None = None) -> EPair:
    """The `EPair` of the ordered shell pair `(sa, sb)` (`ao.j`'s module
    docstring); `recs` is `primitive_pairs(sa, sb)`, passed when the caller
    has it."""
    if recs is None:
        recs = primitive_pairs(sa, sb)
    return EPair(sa.l, sb.l, sa.centre, sb.centre, tuple(
        (k.P, k.zeta, dress((1.0,), ca=sa.coefficients[k.i],
                            cb=sb.coefficients[k.j], zeta=k.zeta)[0])
        for k in recs))
