"""The host reference of `kernel_d2eri3`: the contracted nuclear Hessian of
the three-center DF integral `(mn|P)`.

    eri3_hessian(basis, aux, Gam, nuclei, *, pq, absolute=)  -> H

`H` is `n_atom x 3 x n_atom x 3`,

    H[A][k][B][l] = SUM_mnP Gam_mnP d2(mn|P) / dR_{A,k} dR_{B,l}

over every cell (an AO shell pair `i <= j` crossed with an aux shell), or
over `cells=` only, every centre of every cell on atom `A` (and `B`)
summed. `Gam` is any `[n_ao, n_ao, n_aux]` tensor (`eri3_tensor`'s
layout), not necessarily symmetric in `(m, n)`: the weights are
`gradient.host_df`'s rule, `Gam[a, b, P] + Gam[b, a, P]` off a diagonal
shell pair and `Gam[a, a', P]` on it, at each aux function. A DF-J
skeleton Hessian passes `Gam = D (x) d` (with the metric's `-1/2 d d^T`
term beside it; that term is not built here). The sweep, the atoms
(`atoms=`, `aux_atoms=`) and the injected `[t|tau]` (`pq=`) are
`gradient.host_df.eri3_gradient`'s.

THE ROUTE IS NOT THE KERNEL'S. Every centre, the aux one included, is
differentiated as a FUNCTION, `hessian.host_1e`'s route: a Cartesian `v`
with exponent `a` on `A` moves to the list `d_centre(a, v, k) = [(2a, v +
1_k), (-v_k, v - 1_k)]`, applied twice for `d2`, and every term is the
ordinary three-center integral of the moved Cartesians,

    X(u, v | x) = SUM_t SUM_tau E^{uv}_t [t|tau] E^x_tau

on the per-axis `E` two indices above the class and the single-center `E`
(`e_axis(i, 0, gamma, 0, 0)`) two above the aux shell. So per primitive
pair (`alpha` on `A`, `beta` on `B`, `s_ab = c_a c_b (pi/zeta)^{3/2}`), aux
record (`gamma` on `C`, `s_aux`), Cartesian pair `(a, b)` and aux Cartesian
`x` of weight `w`, with `s = s_ab s_aux`:

    H[A, k, A, l] += s w SUM c X(u, b | x)       over d2_A(a; k, l)
    H[B, k, B, l], H[C, k, C, l]                 likewise
    H[A, k, B, l] += s w SUM c c' X(u, u' | x)   over d_A(a; k) x d_B(b; l)
    H[A, k, C, l], H[B, k, C, l]                 likewise
    H[Y, l, X, k] += the same                    (one value, both places)

The kernel moves the aux centre by the Hermite shift instead (its
single-center `E` does not depend on `C`); the two are one number, so
agreement checks both.

NO TRANSLATIONAL INVARIANCE IS USED: every centre is differentiated on its
own, so `SUM_A H[A, k, B, l] = 0` and `H = H^T` are checks.

`absolute=True` is the same sum with every factor in `|.|` (coefficients,
`E`, `[t|tau]`, `w`, `s`): a round-off scale for the kernel's interpreted
test, as `hessian.host_1e`'s.

Pure Python floats: `src/felizjueves` has no runtime dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis, cartesians
from felizjueves.eri.density_fit.metric import aux_records
from felizjueves.gradient.host import shell_atoms
from felizjueves.hessian.host_1e import Hess, Terms, d2_centre, d_centre, zeros
from felizjueves.mmd.hermite import e_axis, hermite_order
from felizjueves.one_electron.potential import Nucleus

Ang = tuple[int, int, int]
Vec = tuple[float, float, float]
PQ = Callable[[Ang, Ang, Vec, float], float]
Cells = Sequence[tuple[int, int, int]] | None

_KL = tuple((k, l) for k in range(3) for l in range(3))  # noqa: E741


def _rows(basis: Basis, nuclei: Sequence[Nucleus],
          atoms: Sequence[int] | None) -> tuple[int, ...]:
    if atoms is None:
        return shell_atoms(basis, nuclei)
    atoms = tuple(int(a) for a in atoms)
    if len(atoms) != len(basis.shells) or min(atoms, default=0) < 0:
        raise ValueError("atoms= needs one non-negative row per shell")
    return atoms


def _cell_weights(Gam, basis: Basis, aux: Basis, i: int, j: int, n: int
                  ) -> list[list[float]]:
    """`w[r][q]` of cell `(i, j | n)`, `i <= j`, rows `a` outer
    (`gradient.host_df`'s rule, restated: a private name is not
    imported)."""
    sa, sb, sp = basis.shells[i], basis.shells[j], aux.shells[n]
    oa, ob, op = basis.offsets[i], basis.offsets[j], aux.offsets[n]
    out = []
    for r in range(sa.n_cart):
        for c in range(sb.n_cart):
            if i == j:
                out.append([float(Gam[oa + r][oa + c][op + q])
                            for q in range(sp.n_cart)])
            else:
                out.append([float(Gam[oa + r][ob + c][op + q])
                            + float(Gam[ob + c][oa + r][op + q])
                            for q in range(sp.n_cart)])
    return out


class _Record:
    """One primitive pair crossed with one aux record: `X(u, v | x)` of
    moved Cartesians (module docstring), cached."""

    def __init__(self, zeta: float, P: Vec, A: Vec, B: Vec, la: int,
                 lb: int, gamma: float, C: Vec, lP: int, pq: PQ,
                 f: Callable[[float], float]) -> None:
        self.f = f
        self.ab = [[[e_axis(i, j, zeta, P[d] - A[d], P[d] - B[d])
                     for j in range(lb + 3)] for i in range(la + 3)]
                   for d in range(3)]
        self.c = [e_axis(i, 0, gamma, 0.0, 0.0) for i in range(lP + 3)]
        R = (P[0] - C[0], P[1] - C[1], P[2] - C[2])
        theta = math.sqrt(zeta * gamma / (zeta + gamma))
        self.m = {(t, tau): f(pq(t, tau, R, theta))
                  for t in hermite_order(la + lb + 2)
                  for tau in hermite_order(lP + 2)}
        self.cache: dict[tuple[Ang, Ang, Ang], float] = {}

    def __call__(self, u: Ang, v: Ang, x: Ang) -> float:
        key = (u, v, x)
        if key not in self.cache:
            f = self.f
            e = [self.ab[d][u[d]][v[d]] for d in range(3)]
            g = [self.c[x[d]] for d in range(3)]
            ts = [((tx, ty, tz), f(ex) * f(ey) * f(ez))
                  for tx, ex in enumerate(e[0]) for ty, ey in enumerate(e[1])
                  for tz, ez in enumerate(e[2])]
            taus = [((sx, sy, sz), f(gx) * f(gy) * f(gz))
                    for sx, gx in enumerate(g[0]) for sy, gy in enumerate(g[1])
                    for sz, gz in enumerate(g[2])]
            acc = 0.0
            for t, et in ts:
                if not et:
                    continue
                for tau, es in taus:
                    if es:
                        acc += et * self.m[t, tau] * es
            self.cache[key] = acc
        return self.cache[key]


def eri3_hessian(basis: Basis, aux: Basis, Gam, nuclei: Sequence[Nucleus],
                 *, pq: PQ, cells: Cells = None,
                 atoms: Sequence[int] | None = None,
                 aux_atoms: Sequence[int] | None = None,
                 absolute: bool = False) -> Hess:
    """`H[A][k][B][l] = SUM_mnP Gam_mnP d2(mn|P) / dR_{A,k} dR_{B,l}`
    (module docstring); `Gam` is indexed `[m][n][P]`."""
    sh, ash = basis.shells, aux.shells
    rows = _rows(basis, nuclei, atoms)
    arows = _rows(aux, nuclei, aux_atoms)
    H = zeros(max(len(nuclei), max((*rows, *arows), default=-1) + 1))
    f: Callable[[float], float] = abs if absolute else float
    if cells is None:
        cells = [(i, j, n) for i in range(len(sh)) for j in range(i, len(sh))
                 for n in range(len(ash))]
    for i, j, n in cells:
        if not (0 <= i <= j < len(sh) and 0 <= n < len(ash)):
            raise ValueError(f"cell ({i}, {j} | {n}) is not i <= j over "
                             "the two bases")
        sa, sb, sp = sh[i], sh[j], ash[n]
        A, B, C = rows[i], rows[j], arows[n]
        w = _cell_weights(Gam, basis, aux, i, j, n)
        rows_ab = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
        cq = cartesians(sp.l)
        for pp in primitive_pairs(sa, sb):
            alpha, beta = sa.exponents[pp.i], sb.exponents[pp.j]
            sab = (sa.coefficients[pp.i] * sb.coefficients[pp.j]
                   * (math.pi / pp.zeta) ** 1.5)
            for c in aux_records(sp):
                X = _Record(pp.zeta, pp.P, sa.centre, sb.centre, sa.l, sb.l,
                            c.alpha, c.A, sp.l, pq, f)

                def one(ta: Terms, tb: Terms, tc: Terms, X=X) -> float:
                    return sum(f(c1) * f(c2) * f(c3) * X(u, v, x)
                               for c1, u in ta for c2, v in tb
                               for c3, x in tc)

                s = f(sab) * f(c.s)
                for r, (a, b) in enumerate(rows_ab):
                    for q, x in enumerate(cq):
                        if not w[r][q]:
                            continue
                        sw = s * f(w[r][q])
                        pa, pb, px = [(1.0, a)], [(1.0, b)], [(1.0, x)]
                        for k, l in _KL:  # noqa: E741
                            dak = d_centre(alpha, a, k)
                            dbk, dbl = d_centre(beta, b, k), d_centre(beta, b, l)
                            dcl = d_centre(c.alpha, x, l)
                            H[A][k][A][l] += sw * one(
                                d2_centre(alpha, a, k, l), pb, px)
                            H[B][k][B][l] += sw * one(
                                pa, d2_centre(beta, b, k, l), px)
                            H[C][k][C][l] += sw * one(
                                pa, pb, d2_centre(c.alpha, x, k, l))
                            for X0, Y0, val in ((A, B, one(dak, dbl, px)),
                                                (A, C, one(dak, pb, dcl)),
                                                (B, C, one(pa, dbk, dcl))):
                                H[X0][k][Y0][l] += sw * val
                                H[Y0][l][X0][k] += sw * val
    return H
