"""The host reference of `Target.JX` and of the whole Coulomb derivative
matrix `J^X` (#688).

    coulomb_ip1(basis, D, one_centre=)            -> J1[3][n][n]
    coulomb_derivative_matrix(basis, D, nuclei, one_centre=, parts=)
                                                  -> JX[n_atom][3][n][n]

`J1[k][m][n] = SUM_ls (dA_k m n|ls) D_ls`, `A` the centre of `m` (PySCF's
`grad.rhf.get_jk` `vj`), and

    JX[X][k][m][n] = SUM_ls D_ls d(mn|ls) / dR_{X,k}
                   = BRA + KET
    BRA = [m on X] J1[k][m][n] + [n on X] J1[k][n][m]
    KET = SUM_ls (mn|d_{X,k}(ls)) D_ls

for a fixed symmetric `D` (PySCF's `hessian.rhf` `h1ao` Coulomb part:
`vj1 + vj1^T` the ket half, `vj2` on X's rows plus its transpose the
bra half). `parts` is "both", "bra" or "ket".

THE SWEEP is `gradient.host`'s: bra shell pairs `i <= j` row-major, then
`primitive_pairs` order (every ORDERED primitive pair, a diagonal shell
pair's too), the kets over every unique pair `c <= d` at weight 2 off the
diagonal (`gradient.host.coulomb_kets`). Per bra record (`alpha` on `A`,
`beta` on `B`, `s = c_a c_b (pi/zeta)^{3/2}`) and Cartesian pair `(a, b)`
(`m = oa + i`, `n = ob + j`):

- J1: `J_t` at Hermite order `L + 1` over the kets
  (`gradient.host.coulomb_jp`), then `s SUM_t (dA_k E^{ab})_t J_t` into
  `J1[k][m][n]` and, off the diagonal, `s SUM_t (dB_k E^{ab})_t J_t` into
  `J1[k][n][m]` (`dA E^{ij} = 2 alpha E^{i+1,j} - i E^{i-1,j}` on axis `k`,
  the derivative formed BEFORE the sum; the kernel folds raised rows). A
  diagonal shell pair writes the `A` term alone: its `B` term is the `A`
  term of the transposed row at the swapped primitives, a record of its
  own (`hessian.kernel_jx`).
- KET: per atom `X` and axis `k`, `J_t` at Hermite order `L` over the
  DERIVATIVE KETS of `(X, k)` (`hessian.host.coulomb_dkets`, `hdq`'s
  slabs gathered), then `v = s SUM_t E^{ab}_t J_t` into `J[m][n]` and, off
  the diagonal, `J[n][m]` -- the ordinary J build (`jab`'s scatter) over
  other kets.

`absolute=True` (J1 only) is the same sum with every factor in `|.|`: the
kernel test's round-off scale. `[t]^(0)` is INJECTED (`one_centre=`), as
in `gradient.host`. Pure Python floats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from felizjueves.ao.density import checked_density
from felizjueves.ao.pairs import OneCentre, primitive_pairs
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.gradient.host import coulomb_jp, coulomb_kets, shell_atoms
from felizjueves.hessian.host import coulomb_dkets
from felizjueves.mmd.hermite import e_axis, hermite_order
from felizjueves.one_electron.potential import Nucleus

Mat = list[list[float]]
Pairs = Sequence[tuple[int, int]] | None


def _records(sa: Shell, sb: Shell):
    """`(zeta, P, s, alpha, beta)` per primitive pair (`gradient.host`'s)."""
    for pp in primitive_pairs(sa, sb):
        ca, cb = sa.coefficients[pp.i], sb.coefficients[pp.j]
        s = ca * cb * (math.pi / pp.zeta) ** 1.5
        yield (pp.zeta, pp.P, s, sa.exponents[pp.i], sb.exponents[pp.j])


def _unique_pairs(basis: Basis) -> list[tuple[int, int]]:
    n = len(basis.shells)
    return [(i, j) for i in range(n) for j in range(i, n)]


def _zeros(n: int) -> Mat:
    return [[0.0] * n for _ in range(n)]


def _tables(zeta: float, P, A, B, imax: int, jmax: int):
    return [[[e_axis(i, j, zeta, P[d] - A[d], P[d] - B[d])
              for j in range(jmax + 1)] for i in range(imax + 1)]
            for d in range(3)]


def _at(f: Sequence[float], t: int) -> float:
    return f[t] if t < len(f) else 0.0


def _deriv(tab, d: int, i: int, j: int, c2: float, side: str,
           fix) -> list[float]:
    """`dA E^{ij}` (`side` "A", `c2 = 2 alpha`) or `dB E^{ij}` on axis `d`,
    every `t <= i + j + 1`; `fix` is `abs` for the absolute twin."""
    if side == "A":
        hi, n = tab[d][i + 1][j], i
        lo = tab[d][i - 1][j] if i else None
    else:
        hi, n = tab[d][i][j + 1], j
        lo = tab[d][i][j - 1] if j else None
    out = [fix(c2 * x) for x in hi]
    if lo is not None and n:
        for t, x in enumerate(lo):
            out[t] = (out[t] + fix(n * x)) if fix is abs else out[t] - n * x
    return out


def _paired(basis: Basis, pairs: Pairs):
    for i, j in (_unique_pairs(basis) if pairs is None else pairs):
        if not 0 <= i <= j < len(basis.shells):
            raise ValueError(f"shell pair ({i}, {j}) is not i <= j")
        yield i, j


def coulomb_ip1(basis: Basis, D, *, one_centre: OneCentre,
                pairs: Pairs = None, kets=None,
                absolute: bool = False) -> list[Mat]:
    """`J1[k][m][n] = SUM_ls (dA_k m n|ls) D_ls` (module docstring), over
    every bra shell pair or `pairs=` (`i <= j`) only. `kets` is
    `gradient.host.coulomb_kets(basis, D)`, passed when the caller has it.
    `absolute=True`: every factor in `|.|`."""
    fix = abs if absolute else (lambda v: v)
    rows = checked_density(basis, D)
    if kets is None:
        kets = coulomb_kets(basis, rows)
    n = basis.n_ao
    J1 = [_zeros(n) for _ in range(3)]
    for i, j in _paired(basis, pairs):
        sa, sb = basis.shells[i], basis.shells[j]
        oa, ob = basis.offsets[i], basis.offsets[j]
        ts = hermite_order(sa.l + sb.l + 1)
        for zeta, P, s, alpha, beta in _records(sa, sb):
            s = fix(s)
            tab = _tables(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 1)
            if absolute:
                tab = [[[[abs(x) for x in e] for e in row] for row in ax]
                       for ax in tab]
            jp = [fix(v) for v in coulomb_jp(P, zeta, sa.l + sb.l + 1, kets,
                                             one_centre=one_centre)]
            for r, a in enumerate(cartesians(sa.l)):
                for c, b in enumerate(cartesians(sb.l)):
                    e = [tab[d][a[d]][b[d]] for d in range(3)]
                    for k in range(3):
                        for side, c2, (x, y) in (
                                ("A", 2.0 * alpha, (oa + r, ob + c)),
                                ("B", 2.0 * beta, (ob + c, oa + r))):
                            if side == "B" and i == j:
                                continue
                            f = list(e)
                            f[k] = _deriv(tab, k, a[k], b[k], c2, side, fix)
                            v = sum(_at(f[0], t[0]) * _at(f[1], t[1])
                                    * _at(f[2], t[2]) * jt
                                    for t, jt in zip(ts, jp, strict=True))
                            J1[k][x][y] += s * v
    return J1


def _coulomb_over(basis: Basis, kets, *, one_centre: OneCentre,
                  pairs: Pairs = None) -> Mat:
    """`J_mn = SUM_q [mn|q] D_q` over `kets` (any `coulomb_jp` ket list):
    the ordinary J build, `jab`'s scatter, every element once."""
    n = basis.n_ao
    J = _zeros(n)
    for i, j in _paired(basis, pairs):
        sa, sb = basis.shells[i], basis.shells[j]
        oa, ob = basis.offsets[i], basis.offsets[j]
        L = sa.l + sb.l
        ts = hermite_order(L)
        for zeta, P, s, _, _ in _records(sa, sb):
            tab = _tables(zeta, P, sa.centre, sb.centre, sa.l, sb.l)
            jp = coulomb_jp(P, zeta, L, kets, one_centre=one_centre)
            for r, a in enumerate(cartesians(sa.l)):
                for c, b in enumerate(cartesians(sb.l)):
                    e = [tab[d][a[d]][b[d]] for d in range(3)]
                    v = s * sum(_at(e[0], t[0]) * _at(e[1], t[1])
                                * _at(e[2], t[2]) * jt
                                for t, jt in zip(ts, jp, strict=True))
                    J[oa + r][ob + c] += v
                    if i != j:
                        J[ob + c][oa + r] += v
    return J


def coulomb_derivative_matrix(basis: Basis, D, nuclei: Sequence[Nucleus], *,
                              one_centre: OneCentre, parts: str = "both",
                              kets=None, dkets=None,
                              J1: Sequence[Mat] | None = None
                              ) -> list[list[Mat]]:
    """`JX[X][k][m][n] = SUM_ls D_ls d(mn|ls)/dR_{X,k}` (module docstring),
    its bra half from `J1` (computed when not given) and its ket half over
    `dkets` (`hessian.host.coulomb_dkets`, computed when not given)."""
    if parts not in ("both", "bra", "ket"):
        raise ValueError(f"parts = {parts!r} is not 'both', 'bra' or 'ket'")
    rows = checked_density(basis, D)
    at = shell_atoms(basis, nuclei)
    n, nat = basis.n_ao, len(nuclei)
    ao_atom = [at[s] for s, sh in enumerate(basis.shells)
               for _ in range(sh.n_cart)]
    out = [[_zeros(n) for _ in range(3)] for _ in range(nat)]
    if parts != "ket":
        if J1 is None:
            J1 = coulomb_ip1(basis, rows, one_centre=one_centre, kets=kets)
        for k in range(3):
            for m in range(n):
                X = ao_atom[m]
                for q in range(n):
                    out[X][k][m][q] += J1[k][m][q]
                    out[X][k][q][m] += J1[k][m][q]
    if parts != "bra":
        if dkets is None:
            dkets = coulomb_dkets(basis, rows, nuclei)
        for (X, k), ks in dkets.items():
            if not ks:
                continue
            J = _coulomb_over(basis, ks, one_centre=one_centre)
            for m in range(n):
                for q in range(n):
                    out[X][k][m][q] += J[m][q]
    return out


__all__ = ["coulomb_derivative_matrix", "coulomb_ip1"]
