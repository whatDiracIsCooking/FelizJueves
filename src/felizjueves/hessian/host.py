"""The host references of the d2 kernels: contracted nuclear hessians.

    overlap_hessian(basis, M, nuclei, absolute=)                 -> H
    kinetic_hessian(basis, M, nuclei, absolute=)                 -> H
    potential_hessian(basis, M, nuclei, one_centre=, absolute=)  -> H
    nuclear_repulsion_hessian(nuclei)                            -> H

`H` is `n_atom x 3 x n_atom x 3`, `H[A][k][B][l] = SUM_mn M_mn d2X_mn /
dR_{A,k} dR_{B,l}` (the package docstring) -- over every shell pair, or
over `pairs=` (`i <= j`) only -- every centre of every pair on atom `A`
(and `B`) summed. Nuclei, atoms and the sweep are `gradient.host`'s: a
nucleus is `(Z, centre)`, each shell sits on the nucleus whose centre it
equals exactly (`gradient.host.shell_atoms`), shell pairs `i <= j`
row-major, then `primitive_pairs` order, and each pair's block is weighted
per Cartesian pair by `gradient.host.pair_weights` (`M[a, b] + M[b, a]`
off the diagonal, `M[a, a']` on it): `SUM_mn M_mn X_mn` exactly for any
`M` and any `X` symmetric in `m <-> n`.

THE ROUTE IS NOT THE KERNEL'S. The kernels factor every term per axis
(`E_x E_y E_z`, an operator on one axis's factor). This reference is
`experimental/mmd/hessian/ints2.py`'s: each FUNCTION is differentiated,
as a list of `(coefficient, Cartesian)` -- `d_centre(a, v, k) =
[(2a, v + 1_k), (-v_k, v - 1_k)]`, applied twice for `d2` -- and every
term is an ordinary 3-D integral of the moved Cartesians:

- `S(u, v) = PROD_d E^{u_d v_d}_0`
- `T(u, v) = -1/2 SUM_k [4 b^2 S(u, v + 2_k) - 2b (2 v_k + 1) S(u, v)
  + v_k (v_k - 1) S(u, v - 2_k)]`, the Laplacian on the ket (a moved ket
  Cartesian keeps its exponent `b`)
- `V(u, v; n) = -Z SUM_t E^{uv}_t [t + n]^(0)(P - C)`, the shift `n`
  the operator's derivative: `+1_l` with sign `-1` per shift, so
  `dC_l V = +Z SUM E [t + 1_l]`, `dC_k dC_l V = -Z SUM E [t + 1_k + 1_l]`

per primitive pair (`alpha` on `A`, `beta` on `B`, the host's `s = c_a c_b
(pi/zeta)^{3/2}`), and per Cartesian pair `(a, b)` of weight `w`:

    H[A, k, A, l] += s w SUM c V(u, b)       over d2_A(a; k, l)
    H[B, k, B, l] += s w SUM c V(a, u)       over d2_B(b; k, l)
    H[A, k, B, l] += s w SUM c c' V(u, u')   over d_A(a; k) x d_B(b; l)
    H[B, l, A, k] += the same                 (one value, both places)

and for `V`, per nucleus `C`, the centre-nucleus blocks
`dA_k dC_l` (`H[A, k, C, l]` and `H[C, l, A, k]`), `dB_k dC_l` likewise,
and the nucleus-nucleus `dC_k dC_l` (`H[C, k, C, l]`).

`absolute=True` is the same sum with every factor in `|.|` (coefficients,
`E`, `[t]^(0)`, `Z`, `w`, `s`): a round-off scale, for the kernels'
interpreted test.

`[t]^(0)` is INJECTED (`one_centre=`), as in `one_electron.potential`;
`theta = sqrt(zeta)`. Pure Python floats.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from felizjueves.ao.pairs import OneCentre, primitive_pairs
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.gradient.host import Pairs, pair_weights, shell_atoms
from felizjueves.mmd.hermite import e_axis, hermite_order
from felizjueves.one_electron.potential import Nucleus

Ang = tuple[int, int, int]
Hess = list[list[list[list[float]]]]
Matrix = Sequence[Sequence[float]]
Terms = list[tuple[float, Ang]]


def _check_nuclei(nuclei: Sequence[Nucleus]) -> tuple[Nucleus, ...]:
    out = []
    for n in nuclei:
        if len(n) != 2 or len(n[1]) != 3:
            raise ValueError(f"nucleus {n!r} is not (Z, (x, y, z))")
        Z, c = n
        out.append((float(Z), (float(c[0]), float(c[1]), float(c[2]))))
    return tuple(out)


def _matrix(basis: Basis, M: Matrix) -> list[list[float]]:
    n = basis.n_ao
    try:
        rows = [[float(M[i][j]) for j in range(n)] for i in range(n)]
        bad = len(M) != n or any(len(M[i]) != n for i in range(n))
    except (TypeError, IndexError, KeyError):
        raise ValueError(f"M is not an {n} x {n} matrix") from None
    if bad:
        raise ValueError(f"M is not an {n} x {n} matrix")
    return rows


def zeros(n_atom: int) -> Hess:
    return [[[[0.0] * 3 for _ in range(n_atom)] for _ in range(3)]
            for _ in range(n_atom)]


def _shift(v: Ang, d: int, by: int) -> Ang:
    return tuple(c + by * (k == d) for k, c in enumerate(v))  # type: ignore[return-value]


def d_centre(expo: float, v: Ang, k: int) -> Terms:
    """`d/dA_k` of the Cartesian `v` with exponent `expo` on `A`."""
    out = [(2.0 * expo, _shift(v, k, 1))]
    if v[k]:
        out.append((-float(v[k]), _shift(v, k, -1)))
    return out


def d2_centre(expo: float, v: Ang, k: int, l: int) -> Terms:  # noqa: E741
    """`d2/dA_k dA_l` of `v`: `d_centre` twice."""
    return [(c1 * c2, w) for c1, u in d_centre(expo, v, k)
            for c2, w in d_centre(expo, u, l)]


class _Axes:
    """One primitive pair's per-axis `E^{ij}_t`, `i <= imax`, `j <= jmax`."""

    def __init__(self, zeta: float, P, A, B, imax: int, jmax: int) -> None:
        self.tab = [[[e_axis(i, j, zeta, P[d] - A[d], P[d] - B[d])
                      for j in range(jmax + 1)] for i in range(imax + 1)]
                    for d in range(3)]

    def e(self, d: int, i: int, j: int) -> tuple[float, ...]:
        return self.tab[d][i][j]


def _records(sa: Shell, sb: Shell):
    """`(zeta, P, s, alpha, beta)` per primitive pair, `primitive_pairs`
    order; `s` as `pairs.dress` forms it."""
    for pp in primitive_pairs(sa, sb):
        ca, cb = sa.coefficients[pp.i], sb.coefficients[pp.j]
        s = ca * cb * (math.pi / pp.zeta) ** 1.5
        yield (pp.zeta, pp.P, s, sa.exponents[pp.i], sb.exponents[pp.j])


# One record's integral of two moved Cartesians. `shift` is the operator's
# derivative (V only; `()` elsewhere); a value per nucleus for V, else a
# one-element list.
Integral = Callable[[Ang, Ang, tuple[int, ...]], list[float]]

_KL = tuple((k, l) for k in range(3) for l in range(3))  # noqa: E741


def _sweep(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
           integral_of, *, pairs: Pairs, absolute: bool, nuclear: bool
           ) -> Hess:
    """The module docstring's sweep. `integral_of(sa, sb, record)` gives
    the record's `Integral`."""
    nuc = _check_nuclei(nuclei)
    atoms = shell_atoms(basis, nuc)
    Mr = _matrix(basis, M)
    f = abs if absolute else float
    H = zeros(len(nuc))
    n = len(basis.shells)
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i, n)]
    for i, j in pairs:
        if not 0 <= i <= j < n:
            raise ValueError(f"shell pair ({i}, {j}) is not i <= j")
        sa, sb = basis.shells[i], basis.shells[j]
        A, B = atoms[i], atoms[j]
        w = pair_weights(Mr, i, j, basis)
        rows = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
        for rec in _records(sa, sb):
            _, _, s, alpha, beta = rec
            X = integral_of(sa, sb, rec)

            def one(terms_a: Terms, terms_b: Terms,
                    shift: tuple[int, ...] = (), X=X) -> list[float]:
                acc = None
                for c1, u in terms_a:
                    for c2, v in terms_b:
                        x = X(u, v, shift)
                        c = f(c1) * f(c2)
                        if acc is None:
                            acc = [0.0] * len(x)
                        for q, y in enumerate(x):
                            acc[q] += c * y
                return acc if acc is not None else [0.0]

            for wr, (a, b) in zip(w, rows, strict=True):
                sw = f(s) * f(wr)
                for k, l in _KL:  # noqa: E741
                    aa = sum(one(d2_centre(alpha, a, k, l), [(1.0, b)]))
                    bb = sum(one([(1.0, a)], d2_centre(beta, b, k, l)))
                    ab = sum(one(d_centre(alpha, a, k), d_centre(beta, b, l)))
                    H[A][k][A][l] += sw * aa
                    H[B][k][B][l] += sw * bb
                    H[A][k][B][l] += sw * ab
                    H[B][l][A][k] += sw * ab
                if not nuclear:
                    continue
                for k, l in _KL:  # noqa: E741
                    aC = one(d_centre(alpha, a, k), [(1.0, b)], (l,))
                    bC = one([(1.0, a)], d_centre(beta, b, k), (l,))
                    cc = one([(1.0, a)], [(1.0, b)], (k, l))
                    for C in range(len(nuc)):
                        H[A][k][C][l] += sw * aC[C]
                        H[C][l][A][k] += sw * aC[C]
                        H[B][k][C][l] += sw * bC[C]
                        H[C][l][B][k] += sw * bC[C]
                        H[C][k][C][l] += sw * cc[C]
    return H


def overlap_hessian(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
                    *, pairs: Pairs = None, absolute: bool = False) -> Hess:
    """`H[A][k][B][l] = SUM_mn M_mn d2S_mn / dR_{A,k} dR_{B,l}`."""
    f = abs if absolute else float

    def integral_of(sa, sb, rec):
        zeta, P, _, _, _ = rec
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 2, sb.l + 2)

        def X(u, v, shift):
            return [f(ax.e(0, u[0], v[0])[0]) * f(ax.e(1, u[1], v[1])[0])
                    * f(ax.e(2, u[2], v[2])[0])]
        return X

    return _sweep(basis, M, nuclei, integral_of, pairs=pairs,
                  absolute=absolute, nuclear=False)


def kinetic_hessian(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
                    *, pairs: Pairs = None, absolute: bool = False) -> Hess:
    """`H[A][k][B][l] = SUM_mn M_mn d2T_mn / dR_{A,k} dR_{B,l}`."""
    f = abs if absolute else float

    def integral_of(sa, sb, rec):
        zeta, P, _, _, beta = rec
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 2, sb.l + 4)

        def S(u, v):
            return (f(ax.e(0, u[0], v[0])[0]) * f(ax.e(1, u[1], v[1])[0])
                    * f(ax.e(2, u[2], v[2])[0]))

        def X(u, v, shift):
            out = 0.0
            for k in range(3):
                j = v[k]
                out += f(4.0 * beta * beta) * S(u, _shift(v, k, 2))
                out += f(-2.0 * beta * (2 * j + 1)) * S(u, v)
                if j >= 2:
                    out += f(j * (j - 1)) * S(u, _shift(v, k, -2))
            return [f(-0.5) * out]
        return X

    return _sweep(basis, M, nuclei, integral_of, pairs=pairs,
                  absolute=absolute, nuclear=False)


def potential_hessian(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
                      *, one_centre: OneCentre, pairs: Pairs = None,
                      absolute: bool = False) -> Hess:
    """`H[A][k][B][l] = SUM_mn M_mn d2V_mn / dR_{A,k} dR_{B,l}`, the
    operator's centre-nucleus and nucleus-nucleus terms included."""
    nuc = _check_nuclei(nuclei)
    f = abs if absolute else float

    def integral_of(sa, sb, rec):
        zeta, P, _, _, _ = rec
        L = sa.l + sb.l + 2
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 2, sb.l + 2)
        theta = math.sqrt(zeta)
        R = []  # per nucleus, every |t| <= L
        for _, C in nuc:
            pc = (P[0] - C[0], P[1] - C[1], P[2] - C[2])
            R.append({t: f(one_centre(t, 0, pc, theta))
                      for t in hermite_order(L)})
        cache: dict = {}

        def X(u, v, shift):
            key = (u, v, shift)
            if key not in cache:
                n = [0, 0, 0]
                for k in shift:
                    n[k] += 1
                e = [ax.e(d, u[d], v[d]) for d in range(3)]
                sign = -1.0 if len(shift) % 2 == 0 else 1.0
                out = []
                for c, (Z, _) in enumerate(nuc):
                    acc = 0.0
                    for tx, ex in enumerate(e[0]):
                        for ty, ey in enumerate(e[1]):
                            for tz, ez in enumerate(e[2]):
                                acc += (f(ex) * f(ey) * f(ez)) * R[c][
                                    (tx + n[0], ty + n[1], tz + n[2])]
                    out.append(f(sign * Z) * acc)
                cache[key] = out
            return cache[key]
        return X

    return _sweep(basis, M, nuc, integral_of, pairs=pairs, absolute=absolute,
                  nuclear=True)


def nuclear_repulsion_hessian(nuclei: Sequence[Nucleus]) -> Hess:
    """`d2/dR_{A,k} dR_{B,l} SUM_{B<C} Z_B Z_C / |R_B - R_C|`."""
    nuc = _check_nuclei(nuclei)
    H = zeros(len(nuc))
    for a, (Za, A) in enumerate(nuc):
        for b in range(a + 1, len(nuc)):
            Zb, B = nuc[b]
            r = [A[k] - B[k] for k in range(3)]
            d = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            for k in range(3):
                for l in range(3):  # noqa: E741
                    x = Za * Zb * (3.0 * r[k] * r[l] / d ** 5
                                   - (k == l) / d ** 3)
                    H[a][k][a][l] += x
                    H[b][k][b][l] += x
                    H[a][k][b][l] -= x
                    H[b][k][a][l] -= x
    return H
