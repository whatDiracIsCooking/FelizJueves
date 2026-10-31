"""The host references of the d1 kernels: contracted nuclear gradients.

    shell_atoms(basis, nuclei)                       -> atom of each shell
    overlap_gradient(basis, M, nuclei)               -> G   SUM M dS/dR
    kinetic_gradient(basis, M, nuclei)               -> G   SUM M dT/dR
    potential_gradient(basis, M, nuclei, one_centre=) -> G  SUM M dV/dR
    multipole_gradient(basis, M, nuclei, C, e=, q=)  -> [G]  SUM M dMU^c/dR
    coulomb_gradient(basis, D, nuclei, one_centre=)  -> G   dE_J/dR
    exchange_gradient(basis, D, nuclei, one_centre=) -> G   dE_K/dR
    hgp_exchange_launches(basis, nuclei, am)         -> kernel_ghk's launches
    nuclear_repulsion_gradient(nuclei)               -> G   dE_nuc/dR
    records / pair_records(basis, nuclei, la, lb)    -> the d1 kernels' arrays

`G` is `n_atom x 3`, `G[A][k] = SUM_mn M_mn dX_mn/dR_{A,k}` (the package
docstring) -- over every shell pair, or over `pairs=` (`i <= j`) only
-- every centre of every pair on atom `A` summed. A nucleus is
`(Z, centre)`, as in `one_electron.potential`; the atoms ARE the nuclei, in
their order, and each shell sits on the nucleus whose centre it equals
EXACTLY (`shell_atoms` refuses a shell on no nucleus, so a ghost centre is
a nucleus of charge 0).

THE SWEEP is the kernels' record order: shell pairs `i <= j` row-major
(`one_electron.one_electron.matrix`'s), then `primitive_pairs` order. A
pair contributes its block once, weighted per Cartesian pair `(r, c)`

    off-diagonal (i < j):  M[a_r, b_c] + M[b_c, a_r]
    diagonal     (i == j): M[a_r, a_c]

which is `SUM_mn M_mn X_mn` exactly for any `M` (no symmetry assumed) and
any `X` symmetric in `m <-> n`, as `S`, `T`, `V` and `J` are.

PER RECORD (exponents `alpha` on `A`, `beta` on `B`, the host's `s =
c_a c_b (pi/zeta)^{3/2}` of `ao.density.epair`): per axis the undressed
`E^{ij}_t` of `mmd.hermite.e_axis`, one index higher on each side, and

    dA E^{ij}_t = 2 alpha E^{i+1,j}_t - i E^{i-1,j}_t
    dB E^{ij}_t = 2 beta  E^{i,j+1}_t - j E^{i,j-1}_t

on the differentiated axis, the plain `E` on the other two. Then

- `S`:  `E^{ab}_0` with one axis differentiated.
- `T`:  the kinetic bracket `K_x S_y S_z + S_x K_y S_z + S_x S_y K_z`
  (`one_electron.kinetic.kinetic_axis`) with the differentiated axis's
  factor replaced by its derivative -- `dA K_ij = 2 alpha K_{i+1,j} -
  i K_{i-1,j}`, `dB K_ij = 2 beta K_{i,j+1} - j K_{i,j-1}` (the operator
  commutes with a translation, so the rule applies to the ket function
  before the Laplacian, as `experimental/mmd/gradient/ints.py` does).
- `MU` (the dipole, `e = 1`) and the quadrupole's raw second moments
  (`e = 2`), `<a| q (r - C)^(ex, ey, ez) |b>` about a FIXED origin `C`
  (`one_electron.multipole`): per axis the factor `F^n_{ij} = SUM_{t <=
  min(n, i+j)} E^{ij}_t M^n_t` (`multipole.multipole_axis`, its moments
  `multipole.hermite_moments(n, P - C, zeta)`), and its derivative
  `dA F^n_{ij} = 2 alpha F^n_{i+1,j} - i F^n_{i-1,j}` (`dB` likewise) --
  the rule acts on the basis function, so the moments, at the record's
  own `P`, are unchanged. One `G` per component (`cartesians(e)` order),
  times `q s`. The origin is not a nucleus: no operator term, and
  `SUM_A G^c[A, k]` is minus the origin derivative, a check.
- `V`:  the Hermite densities `H0_t = SUM_r w_r E^r_t` (`|t| <= L`) and
  `HA_k,t`, `HB_k,t` (`|t| <= L + 1`), then per nucleus
  `-Z SUM_t HA_k,t [t]^(0)` on `A` (and `B`) and `+Z SUM_t H0_t
  [t + 1_k]^(0)` on the nucleus itself, `[t]^(0)` at `P - C`,
  `theta = sqrt(zeta)` (`one_electron.potential`'s identity).
- `J`:  `J_p` of the bra record over every ket, at Hermite order `L + 1`
  (`eri.j_coulomb.j`'s `J_p = SUM_k SUM_q [p|q] D_q`, its kets from
  `ao.density.make_kets` over the unique ket pairs at weight 2 off the
  diagonal), then `SUM_t HA_k,t J_t` on `A` and `HB` on `B`, with
  `M = D`. That is the whole `dE_J` (`E_J = 1/2 SUM D_ab D_cd (ab|cd)`;
  the ket derivative is the bra derivative, the package docstring).
- `K`:  `E_K = 1/2 SUM_abcd D_ab D_cd (ac|bd)` (`K_ab = SUM_cd D_cd
  (ac|bd)`, `eri.k_exchange.k`'s pairing: bra `(a, c)`, ket `(b, d)`).
  Its ket derivative is its bra derivative for a SYMMETRIC `D` (rename
  `a <-> b`, `c <-> d` in the ket term and use `D_ba = D_ab`), so

      dE_K/dR = SUM over every ORDERED bra shell pair (sa, sc) and ket
                shell pair (sb, sd), SUM_abcd D_ab D_cd d_bra (ac|bd)

  per (bra record, ket record): `[r]^(0)` at `P - Q`, `theta^2 = zeta eta
  / (zeta + eta)`, `|r| <= L + 1`; per `(b,d)` row `I[p] = SUM_q
  (-1)^{|q|} [p+q]^(0) s_bd E^{bd}_q` over `hermite_order(lp + 1)`; per
  `(a,c)` row and axis `k`, `SUM_p (dA_k E^{ac})_p I[p]` on `A` (and
  `dC` on `C`) times `D_ab D_cd s_ac`. No pair weight: every ordered
  quartet is visited once. The RHF exchange energy is `-1/2 E_K`; the
  factor is the caller's. An asymmetric `D` is refused
  (`ao.density.checked_density`).

`atoms=` replaces `shell_atoms`: one row of `G` per shell, each row after
the nuclei's (so `V`'s nucleus rows stay where they are). A test passes
every shell a row of its own to see each centre's contribution apart --
the scale of a sum whose two centres cancel on one atom.

`[t]^(0)` is INJECTED (`one_centre=`), as in `one_electron.potential`;
tests pass `reference.pq.one_centre`. `J`'s `[p|q]` is
`(-1)^{|q|} [p + q]^(0)` of it (`reference.pq.pq`'s identity), one
`[r]^(0)` table per (bra record, ket record).

Pure Python floats: `src/felizjueves` has no runtime dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from felizjueves.ao.density import checked_density, make_kets
from felizjueves.ao.pairs import OneCentre, primitive_pairs
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.mmd.hermite import e_axis, hermite_order
from felizjueves.one_electron.kinetic import kinetic_axis
from felizjueves.one_electron.multipole import hermite_moments, multipole_axis
from felizjueves.one_electron.potential import Nucleus

Ang = tuple[int, int, int]
Grad = list[list[float]]
Matrix = Sequence[Sequence[float]]


def _check_nuclei(nuclei: Sequence[Nucleus]) -> tuple[Nucleus, ...]:
    """`one_electron.potential`'s nucleus check, restated (a private name is
    not imported across modules)."""
    out = []
    for n in nuclei:
        if len(n) != 2 or len(n[1]) != 3:
            raise ValueError(f"nucleus {n!r} is not (Z, (x, y, z))")
        Z, c = n
        out.append((float(Z), (float(c[0]), float(c[1]), float(c[2]))))
    return tuple(out)


def shell_atoms(basis: Basis, nuclei: Sequence[Nucleus]) -> tuple[int, ...]:
    """The nucleus each shell sits on: the first whose centre equals the
    shell's exactly. Refuses a shell on no nucleus."""
    nuc = _check_nuclei(nuclei)
    out = []
    for n, s in enumerate(basis.shells):
        hit = [k for k, (_, c) in enumerate(nuc) if tuple(c) == s.centre]
        if not hit:
            raise ValueError(f"shell {n} at {s.centre} sits on no nucleus")
        out.append(hit[0])
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


def pair_weights(M: Sequence[Sequence[float]], i: int, j: int,
                 basis: Basis) -> list[float]:
    """The weight of each Cartesian pair of shell pair `(i, j)`, `i <= j`,
    rows `a` outer (the module docstring's rule)."""
    sa, sb = basis.shells[i], basis.shells[j]
    oa, ob = basis.offsets[i], basis.offsets[j]
    out = []
    for r in range(sa.n_cart):
        for c in range(sb.n_cart):
            if i == j:
                out.append(M[oa + r][oa + c])
            else:
                out.append(M[oa + r][ob + c] + M[ob + c][oa + r])
    return out


def _unique_pairs(basis: Basis) -> list[tuple[int, int]]:
    n = len(basis.shells)
    return [(i, j) for i in range(n) for j in range(i, n)]


def _shift(v: Ang, d: int, by: int) -> Ang:
    return tuple(c + by * (k == d) for k, c in enumerate(v))  # type: ignore[return-value]


class _Axes:
    """One primitive pair's per-axis `E^{ij}_t`, `i <= imax`, `j <= jmax`."""

    def __init__(self, zeta: float, P, A, B, imax: int, jmax: int) -> None:
        self.tab = [[[e_axis(i, j, zeta, P[d] - A[d], P[d] - B[d])
                      for j in range(jmax + 1)] for i in range(imax + 1)]
                    for d in range(3)]

    def e(self, d: int, i: int, j: int) -> tuple[float, ...]:
        return self.tab[d][i][j]


def _deriv(lo: tuple[float, ...] | None, hi: tuple[float, ...], c2: float,
           n: int) -> list[float]:
    """`c2 * hi - n * lo` entry by entry (`hi` one entry longer)."""
    out = [c2 * x for x in hi]
    if lo is not None and n:
        for t, x in enumerate(lo):
            out[t] -= n * x
    return out


def _axis_derivs(ax: _Axes, d: int, i: int, j: int, alpha: float,
                 beta: float) -> tuple[list[float], list[float]]:
    """`dA E^{ij}` and `dB E^{ij}` on axis `d`, every `t <= i + j + 1`."""
    da = _deriv(ax.e(d, i - 1, j) if i else None, ax.e(d, i + 1, j),
                2.0 * alpha, i)
    db = _deriv(ax.e(d, i, j - 1) if j else None, ax.e(d, i, j + 1),
                2.0 * beta, j)
    return da, db


def _at(f: Sequence[float], t: int) -> float:
    return f[t] if t < len(f) else 0.0


def _hermite(factors: list[Sequence[float]], ts: Sequence[Ang]) -> list[float]:
    """`PROD_d f_d[t_d]` over `ts`, zero where an axis has no such entry."""
    return [_at(factors[0], t[0]) * _at(factors[1], t[1])
            * _at(factors[2], t[2]) for t in ts]


def _records(sa: Shell, sb: Shell):
    """`(zeta, P, s, alpha, beta)` per primitive pair, `primitive_pairs`
    order; `s` as `pairs.dress` forms it."""
    for pp in primitive_pairs(sa, sb):
        ca, cb = sa.coefficients[pp.i], sb.coefficients[pp.j]
        s = ca * cb * (math.pi / pp.zeta) ** 1.5
        yield (pp.zeta, pp.P, s, sa.exponents[pp.i], sb.exponents[pp.j])


def _add(G: Grad, atom: int, g: Sequence[float], s: float) -> None:
    for k in range(3):
        G[atom][k] += s * g[k]


Pairs = Sequence[tuple[int, int]] | None


def _one_electron(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
                  per_record, pairs: Pairs = None,
                  atoms: Sequence[int] | None = None) -> Grad:
    nuc = _check_nuclei(nuclei)
    if atoms is None:
        atoms = shell_atoms(basis, nuc)
        n_rows = len(nuc)
    else:  # the caller's own rows (module docstring, `atoms=`)
        atoms = tuple(atoms)
        if len(atoms) != len(basis.shells) or min(atoms) < len(nuc):
            raise ValueError("atoms= needs one row per shell, each after "
                             "the nuclei's rows")
        n_rows = max(atoms) + 1
    Mr = _matrix(basis, M)
    G = [[0.0, 0.0, 0.0] for _ in range(n_rows)]
    for i, j in (_unique_pairs(basis) if pairs is None else pairs):
        if not 0 <= i <= j < len(basis.shells):
            raise ValueError(f"shell pair ({i}, {j}) is not i <= j")
        sa, sb = basis.shells[i], basis.shells[j]
        w = pair_weights(Mr, i, j, basis)
        for rec in _records(sa, sb):
            per_record(G, atoms[i], atoms[j], sa, sb, w, rec)
    return G


def overlap_gradient(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
                     *, pairs: Pairs = None,
                     atoms: Sequence[int] | None = None) -> Grad:
    """`G[A][k] = SUM_mn M_mn dS_mn / dR_{A,k}` (module docstring)."""
    def per_record(G, at_a, at_b, sa, sb, w, rec):
        zeta, P, s, alpha, beta = rec
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 1)
        gA, gB = [0.0] * 3, [0.0] * 3
        rows = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
        for wr, (a, b) in zip(w, rows, strict=True):
            e0 = [ax.e(d, a[d], b[d])[0] for d in range(3)]
            for k in range(3):
                da, db = _axis_derivs(ax, k, a[k], b[k], alpha, beta)
                fa, fb = list(e0), list(e0)
                fa[k], fb[k] = da[0], db[0]
                gA[k] += wr * (fa[0] * fa[1] * fa[2])
                gB[k] += wr * (fb[0] * fb[1] * fb[2])
        _add(G, at_a, gA, s)
        _add(G, at_b, gB, s)

    return _one_electron(basis, M, nuclei, per_record, pairs, atoms)


def kinetic_gradient(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
                     *, pairs: Pairs = None,
                     atoms: Sequence[int] | None = None) -> Grad:
    """`G[A][k] = SUM_mn M_mn dT_mn / dR_{A,k}` (module docstring)."""
    def per_record(G, at_a, at_b, sa, sb, w, rec):
        zeta, P, s, alpha, beta = rec
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 3)

        def S(d, i, j):
            return ax.e(d, i, j)[0]

        def K(d, i, j):
            row = [ax.e(d, i, jj)[0] for jj in range(j + 3)]
            return kinetic_axis(row, j, beta)

        def dS(d, i, j):
            return (2.0 * alpha * S(d, i + 1, j) - (i * S(d, i - 1, j) if i
                                                    else 0.0),
                    2.0 * beta * S(d, i, j + 1) - (j * S(d, i, j - 1) if j
                                                   else 0.0))

        def dK(d, i, j):
            return (2.0 * alpha * K(d, i + 1, j) - (i * K(d, i - 1, j) if i
                                                    else 0.0),
                    2.0 * beta * K(d, i, j + 1) - (j * K(d, i, j - 1) if j
                                                   else 0.0))

        gA, gB = [0.0] * 3, [0.0] * 3
        rows = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
        for wr, (a, b) in zip(w, rows, strict=True):
            s0 = [S(d, a[d], b[d]) for d in range(3)]
            k0 = [K(d, a[d], b[d]) for d in range(3)]
            for k in range(3):
                dsk, dkk = dS(k, a[k], b[k]), dK(k, a[k], b[k])
                for side, g in ((0, gA), (1, gB)):
                    tot = 0.0
                    for m in range(3):  # the bracket's term with K on axis m
                        f = [k0[d] if d == m else s0[d] for d in range(3)]
                        f[k] = (dkk if k == m else dsk)[side]
                        tot += f[0] * f[1] * f[2]
                    g[k] += wr * tot
        _add(G, at_a, gA, s)
        _add(G, at_b, gB, s)

    return _one_electron(basis, M, nuclei, per_record, pairs, atoms)


def _hermite_densities(ax: _Axes, sa: Shell, sb: Shell, w: Sequence[float],
                       alpha: float, beta: float, L: int):
    """`H0` over `hermite_order(L)` and `HA[k]`, `HB[k]` over
    `hermite_order(L + 1)` (module docstring)."""
    t0, t1 = hermite_order(L), hermite_order(L + 1)
    H0 = [0.0] * len(t0)
    HA = [[0.0] * len(t1) for _ in range(3)]
    HB = [[0.0] * len(t1) for _ in range(3)]
    rows = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
    for wr, (a, b) in zip(w, rows, strict=True):
        e = [ax.e(d, a[d], b[d]) for d in range(3)]
        for n, v in enumerate(_hermite(e, t0)):
            H0[n] += wr * v
        for k in range(3):
            da, db = _axis_derivs(ax, k, a[k], b[k], alpha, beta)
            for H, dk in ((HA, da), (HB, db)):
                f = list(e)
                f[k] = dk
                for n, v in enumerate(_hermite(f, t1)):
                    H[k][n] += wr * v
    return H0, HA, HB


def multipole_gradient(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
                       C: Sequence[float], e: int = 1, q: float = 1.0, *,
                       pairs: Pairs = None,
                       atoms: Sequence[int] | None = None) -> list[Grad]:
    """`G[c][A][k] = SUM_mn M_mn dMU^c_mn / dR_{A,k}`, one `G` per component
    `c` of `cartesians(e)` -- the dipole at `e = 1`, the quadrupole's raw
    second moments at `e = 2` (module docstring, `MU`)."""
    if type(e) is not int or e < 0:
        raise ValueError(f"e = {e!r} is not an int >= 0")
    c = tuple(float(x) for x in C)
    if len(c) != 3 or not all(math.isfinite(x) for x in c):
        raise ValueError(f"origin C = {C!r} is not three finite floats")
    q = float(q)
    comps = cartesians(e)
    out: list[Grad] = []

    def per_record(G, at_a, at_b, sa, sb, w, rec):
        if not out:  # one G per component, `_one_electron`'s rows
            out.extend([r[:] for r in G] for _ in comps)
        zeta, P, s, alpha, beta = rec
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 1)
        mom = [[hermite_moments(n, P[d] - c[d], zeta) for n in range(e + 1)]
               for d in range(3)]

        def F(d: int, n: int, i: int, j: int) -> float:
            return multipole_axis(ax.e(d, i, j), mom[d][n])

        def dF(d: int, n: int, i: int, j: int) -> tuple[float, float]:
            da = 2.0 * alpha * F(d, n, i + 1, j)
            if i:
                da -= i * F(d, n, i - 1, j)
            db = 2.0 * beta * F(d, n, i, j + 1)
            if j:
                db -= j * F(d, n, i, j - 1)
            return da, db

        rows = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
        qs = q * s
        for n, ek in enumerate(comps):
            gA, gB = [0.0] * 3, [0.0] * 3
            for wr, (a, b) in zip(w, rows, strict=True):
                f = [F(d, ek[d], a[d], b[d]) for d in range(3)]
                for k in range(3):
                    da, db = dF(k, ek[k], a[k], b[k])
                    fa, fb = list(f), list(f)
                    fa[k], fb[k] = da, db
                    gA[k] += wr * (fa[0] * fa[1] * fa[2])
                    gB[k] += wr * (fb[0] * fb[1] * fb[2])
            _add(out[n], at_a, gA, qs)
            _add(out[n], at_b, gB, qs)

    G0 = _one_electron(basis, M, nuclei, per_record, pairs, atoms)
    return out or [[r[:] for r in G0] for _ in comps]


def potential_gradient(basis: Basis, M: Matrix, nuclei: Sequence[Nucleus],
                       *, one_centre: OneCentre, pairs: Pairs = None,
                       atoms: Sequence[int] | None = None) -> Grad:
    """`G[A][k] = SUM_mn M_mn dV_mn / dR_{A,k}`, the Hellmann-Feynman term
    on every nucleus included (module docstring)."""
    nuc = _check_nuclei(nuclei)

    def per_record(G, at_a, at_b, sa, sb, w, rec):
        zeta, P, s, alpha, beta = rec
        L = sa.l + sb.l
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 1)
        H0, HA, HB = _hermite_densities(ax, sa, sb, w, alpha, beta, L)
        t0, t1 = hermite_order(L), hermite_order(L + 1)
        index = {t: n for n, t in enumerate(t1)}
        theta = math.sqrt(zeta)
        gA, gB = [0.0] * 3, [0.0] * 3
        for c, (Z, C) in enumerate(nuc):
            pc = (P[0] - C[0], P[1] - C[1], P[2] - C[2])
            R = [one_centre(t, 0, pc, theta) for t in t1]
            gC = [0.0] * 3
            for k in range(3):
                gA[k] -= Z * sum(h * r for h, r in zip(HA[k], R, strict=True))
                gB[k] -= Z * sum(h * r for h, r in zip(HB[k], R, strict=True))
                gC[k] = Z * sum(h * R[index[_shift(t, k, 1)]]
                                for h, t in zip(H0, t0, strict=True))
            _add(G, c, gC, s)
        _add(G, at_a, gA, s)
        _add(G, at_b, gB, s)

    return _one_electron(basis, M, nuc, per_record, pairs, atoms)


# ---------------------------------------------------------------------------
# The derivative MATRICES (#687): hessian.kernel_ip1e's references
# ---------------------------------------------------------------------------

Mat3 = list[list[list[float]]]  # X[k][m][n]


def _matrix_sweep(basis: Basis, per_record, pairs: Pairs = None) -> Mat3:
    """THE MATRIX SWEEP: the gradient sweep's shell pairs `i <= j` and
    records; `per_record(rec, sa, sb)` returns the record's `row(a, b) ->
    (vA, vB)`, one Cartesian row's per-axis values before `s`. Then
    `X[k][m][n] += s vA[k]` (`m = oa + r`, `n = ob + c`) and, off the
    diagonal (`i < j`), `X[k][n][m] += s vB[k]`. A diagonal pair (`i ==
    j`) writes `vA` alone: its records are every ORDERED primitive pair, so
    `vA` over them is already the whole first-index derivative of the
    block (`ao.pairs.primitive_pairs`)."""
    n = basis.n_ao
    X = [[[0.0] * n for _ in range(n)] for _ in range(3)]
    for i, j in (_unique_pairs(basis) if pairs is None else pairs):
        if not 0 <= i <= j < len(basis.shells):
            raise ValueError(f"shell pair ({i}, {j}) is not i <= j")
        sa, sb = basis.shells[i], basis.shells[j]
        oa, ob = basis.offsets[i], basis.offsets[j]
        for rec in _records(sa, sb):
            s = rec[2]
            row = per_record(rec, sa, sb)
            for r, a in enumerate(cartesians(sa.l)):
                for c, b in enumerate(cartesians(sb.l)):
                    vA, vB = row(a, b)
                    for k in range(3):
                        X[k][oa + r][ob + c] += s * vA[k]
                        if i != j:
                            X[k][ob + c][oa + r] += s * vB[k]
    return X


def overlap_ip1(basis: Basis, *, pairs: Pairs = None) -> Mat3:
    """`X[k][m][n] = dS_mn / dA_k`, `A` the centre of `m` (`-int1e_ipovlp`):
    `E^{ab}_0` with axis `k` differentiated, on the matrix sweep."""
    def per_record(rec, sa, sb):
        zeta, P, _, alpha, beta = rec
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 1)

        def row(a, b):
            e0 = [ax.e(d, a[d], b[d])[0] for d in range(3)]
            vA, vB = [0.0] * 3, [0.0] * 3
            for k in range(3):
                da, db = _axis_derivs(ax, k, a[k], b[k], alpha, beta)
                fa, fb = list(e0), list(e0)
                fa[k], fb[k] = da[0], db[0]
                vA[k] = fa[0] * fa[1] * fa[2]
                vB[k] = fb[0] * fb[1] * fb[2]
            return vA, vB
        return row

    return _matrix_sweep(basis, per_record, pairs)


def kinetic_ip1(basis: Basis, *, pairs: Pairs = None) -> Mat3:
    """`X[k][m][n] = dT_mn / dA_k` (`-int1e_ipkin`): `kinetic_gradient`'s
    bracket per row, on the matrix sweep."""
    def per_record(rec, sa, sb):
        zeta, P, _, alpha, beta = rec
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 3)

        def S(d, i, j):
            return ax.e(d, i, j)[0]

        def K(d, i, j):
            return kinetic_axis([ax.e(d, i, jj)[0] for jj in range(j + 3)],
                                j, beta)

        def rule(f, d, i, j):
            return (2.0 * alpha * f(d, i + 1, j) - (i * f(d, i - 1, j) if i
                                                    else 0.0),
                    2.0 * beta * f(d, i, j + 1) - (j * f(d, i, j - 1) if j
                                                   else 0.0))

        def row(a, b):
            s0 = [S(d, a[d], b[d]) for d in range(3)]
            k0 = [K(d, a[d], b[d]) for d in range(3)]
            out = ([0.0] * 3, [0.0] * 3)
            for k in range(3):
                dsk, dkk = rule(S, k, a[k], b[k]), rule(K, k, a[k], b[k])
                for side in (0, 1):
                    for m in range(3):  # the bracket's term with K on axis m
                        f = [k0[d] if d == m else s0[d] for d in range(3)]
                        f[k] = (dkk if k == m else dsk)[side]
                        out[side][k] += f[0] * f[1] * f[2]
            return out
        return row

    return _matrix_sweep(basis, per_record, pairs)


def potential_ip1(basis: Basis, nuclei: Sequence[Nucleus], *,
                  one_centre: OneCentre, pairs: Pairs = None) -> Mat3:
    """`X[k][m][n] = dV_mn / dA_k` (`-int1e_ipnuc`), the basis function's
    derivative only, every nucleus's operator held fixed: per record the
    nuclei's potential in Hermite space, `U_t = -SUM_C Z_C [t]^(0)(P - C)`
    over `|t| <= L + 1`, then per row `SUM_t (dA_k E^{ab})_t U_t`."""
    nuc = _check_nuclei(nuclei)

    def per_record(rec, sa, sb):
        zeta, P, _, alpha, beta = rec
        t1 = hermite_order(sa.l + sb.l + 1)
        theta = math.sqrt(zeta)
        U = [0.0] * len(t1)
        for Z, C in nuc:
            pc = (P[0] - C[0], P[1] - C[1], P[2] - C[2])
            for n, t in enumerate(t1):
                U[n] -= Z * one_centre(t, 0, pc, theta)
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 1)

        def row(a, b):
            e = [ax.e(d, a[d], b[d]) for d in range(3)]
            vA, vB = [0.0] * 3, [0.0] * 3
            for k in range(3):
                da, db = _axis_derivs(ax, k, a[k], b[k], alpha, beta)
                for v, dk in ((vA, da), (vB, db)):
                    f = list(e)
                    f[k] = dk
                    v[k] = sum(h * u for h, u in zip(_hermite(f, t1), U,
                                                     strict=True))
            return vA, vB
        return row

    return _matrix_sweep(basis, per_record, pairs)


def potential_rinv(basis: Basis, nuclei: Sequence[Nucleus], c: int, *,
                   one_centre: OneCentre, pairs: Pairs = None) -> Mat3:
    """`X[k][m][n] = dV^C_mn / dC_k`, nucleus `c`'s operator term: `V^C =
    -Z_C <m|1/|r - C||n>` moved with its nucleus, `+Z_C SUM_t E^{ab}_t
    [t + 1_k]^(0)(P - C)` per row (`d/dC_k [t]^(0) = -[t + 1_k]^(0)`) --
    `-Z_C (iprinv + iprinv^T)` with libcint's origin at `C`. Symmetric in
    `m <-> n`, so a row writes the same value both ways."""
    nuc = _check_nuclei(nuclei)
    if type(c) is not int or not 0 <= c < len(nuc):
        raise ValueError(f"nucleus {c!r} is not in 0..{len(nuc) - 1}")
    Z, C = nuc[c]

    def per_record(rec, sa, sb):
        zeta, P, _, _, _ = rec
        L = sa.l + sb.l
        t0, t1 = hermite_order(L), hermite_order(L + 1)
        index = {t: n for n, t in enumerate(t1)}
        pc = (P[0] - C[0], P[1] - C[1], P[2] - C[2])
        R = [one_centre(t, 0, pc, math.sqrt(zeta)) for t in t1]
        up = [[R[index[_shift(t, k, 1)]] for t in t0] for k in range(3)]
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l, sb.l)

        def row(a, b):
            E = _hermite([ax.e(d, a[d], b[d]) for d in range(3)], t0)
            v = [Z * sum(h * r for h, r in zip(E, up[k], strict=True))
                 for k in range(3)]
            return v, v
        return row

    return _matrix_sweep(basis, per_record, pairs)


def coulomb_jp(P, zeta: float, L: int, kets, *,
               one_centre: OneCentre) -> list[float]:
    """`J_t = SUM_k SUM_q [t|q] D_q` of one bra record `(P, zeta)` at
    Hermite order `L` (`hermite_order(L)`), over `kets`
    (`coulomb_kets`), with `[t|q] = (-1)^{|q|} [t + q]^(0)`: one `[r]^(0)`
    table per ket, `|r| <= L + l_q`."""
    ps = hermite_order(L)
    out = [0.0] * len(ps)
    for k in kets:
        theta = math.sqrt(zeta * k.eta / (zeta + k.eta))
        PQv = (P[0] - k.Q[0], P[1] - k.Q[1], P[2] - k.Q[2])
        lq = sum(k.qs[-1])
        R = {r: one_centre(r, 0, PQv, theta) for r in hermite_order(L + lq)}
        signs = [-1.0 if sum(q) % 2 else 1.0 for q in k.qs]
        for n, p in enumerate(ps):
            acc = 0.0
            for q, sg, dq in zip(k.qs, signs, k.dq, strict=True):
                acc += (sg * R[(p[0] + q[0], p[1] + q[1], p[2] + q[2])]) * dq
            out[n] += acc
    return out


def _no_pq(p, q, PQv, theta) -> float:
    raise AssertionError("no [p|q] is evaluated building kets at tau = 0")


def coulomb_kets(basis: Basis, D: Matrix):
    """`ao.density.make_kets` over every unique ket pair `c <= d`, weight 2
    off the diagonal: the kets of `eri.j_coulomb.j.coulomb_matrix`."""
    rows = checked_density(basis, D)
    pairs = [(c, d, 1.0 if c == d else 2.0) for c, d in _unique_pairs(basis)]
    return make_kets(basis, rows, pairs, pq=_no_pq, tau=0.0)


def coulomb_gradient(basis: Basis, D: Matrix, nuclei: Sequence[Nucleus],
                     *, one_centre: OneCentre, pairs: Pairs = None,
                     atoms: Sequence[int] | None = None,
                     kets=None) -> Grad:
    """`dE_J / dR_{A,k}`, `E_J = 1/2 SUM D_ab D_cd (ab|cd)` (module
    docstring): the bra derivative alone, at `M = D`, against `J_p` one
    Hermite order higher. `kets` is `coulomb_kets(basis, D)`, passed when
    the caller has it."""
    if kets is None:
        kets = coulomb_kets(basis, D)

    def per_record(G, at_a, at_b, sa, sb, w, rec):
        zeta, P, s, alpha, beta = rec
        L = sa.l + sb.l
        ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 1, sb.l + 1)
        _, HA, HB = _hermite_densities(ax, sa, sb, w, alpha, beta, L)
        jp = coulomb_jp(P, zeta, L + 1, kets, one_centre=one_centre)
        gA = [sum(h * j for h, j in zip(HA[k], jp, strict=True))
              for k in range(3)]
        gB = [sum(h * j for h, j in zip(HB[k], jp, strict=True))
              for k in range(3)]
        _add(G, at_a, gA, s)
        _add(G, at_b, gB, s)

    return _one_electron(basis, D, nuclei, per_record, pairs, atoms)


def _exchange_quartet(G: Grad, at_a: int, at_c: int, sa: Shell, sc: Shell,
                      sb: Shell, sd: Shell, Dab, Dcd, *,
                      one_centre: OneCentre, absolute: bool) -> None:
    """One ordered shell quartet's `SUM D_ab D_cd d_bra (ac|bd)` into `G`
    (module docstring, `K`). `Dab[i][j]` / `Dcd[k][l]` are the two density
    blocks. `absolute` takes every factor in `|.|`, the derivative rule's
    two terms both added: the round-off scale."""
    f = abs if absolute else float
    lp, lq = sa.l + sc.l, sb.l + sd.l
    ps, qs = hermite_order(lp + 1), hermite_order(lq)
    rows_ac = [(a, c) for a in cartesians(sa.l) for c in cartesians(sc.l)]
    rows_bd = [(b, d) for b in cartesians(sb.l) for d in cartesians(sd.l)]
    nc, nd = sc.n_cart, sd.n_cart
    kets = []
    for eta, Q, s_bd, _, _ in _records(sb, sd):
        ax = _Axes(eta, Q, sb.centre, sd.centre, sb.l, sd.l)
        kets.append((eta, Q, [
            [s_bd * v for v in _hermite([ax.e(d, b[d], e[d])
                                         for d in range(3)], qs)]
            for b, e in rows_bd]))
    for zeta, P, s_ac, alpha, gamma in _records(sa, sc):
        ax = _Axes(zeta, P, sa.centre, sc.centre, sa.l + 1, sc.l + 1)
        dE = []  # per (a,c) row, per side (A, C), per axis: over `ps`
        for a, c in rows_ac:
            e = [ax.e(d, a[d], c[d]) for d in range(3)]
            sides = []
            for side, two in ((0, 2.0 * alpha), (1, 2.0 * gamma)):
                per_k = []
                for k in range(3):
                    if absolute:
                        n = (a, c)[side][k]
                        hi = [a[k], c[k]]
                        hi[side] += 1
                        dk = [two * abs(x) for x in ax.e(k, *hi)]
                        if n:
                            lo = [a[k], c[k]]
                            lo[side] -= 1
                            for t, x in enumerate(ax.e(k, *lo)):
                                dk[t] += n * abs(x)
                        fk = [[abs(x) for x in v] for v in e]
                    else:
                        dk = _axis_derivs(ax, k, a[k], c[k], alpha,
                                          gamma)[side]
                        fk = list(e)
                    fk[k] = dk
                    per_k.append(_hermite(fk, ps))
                sides.append(per_k)
            dE.append(sides)
        gA, gC = [0.0] * 3, [0.0] * 3
        for eta, Q, ebd in kets:
            theta = math.sqrt(zeta * eta / (zeta + eta))
            PQv = (P[0] - Q[0], P[1] - Q[1], P[2] - Q[2])
            R = {r: f(one_centre(r, 0, PQv, theta))
                 for r in hermite_order(lp + lq + 1)}
            for rbd in range(len(rows_bd)):
                ib, id_ = divmod(rbd, nd)
                col = []
                for p in ps:
                    acc = 0.0
                    for q, e in zip(qs, ebd[rbd], strict=True):
                        x = R[(p[0] + q[0], p[1] + q[1], p[2] + q[2])] * f(e)
                        acc += x if absolute or sum(q) % 2 == 0 else -x
                    col.append(acc)
                for rac in range(len(rows_ac)):
                    ia, ic = divmod(rac, nc)
                    w = f(Dab[ia][ib]) * f(Dcd[ic][id_])
                    for side, g in ((0, gA), (1, gC)):
                        for k in range(3):
                            v = 0.0
                            for x, y in zip(dE[rac][side][k], col,
                                            strict=True):
                                v += x * y
                            g[k] += w * v
        _add(G, at_a, gA, f(s_ac))
        _add(G, at_c, gC, f(s_ac))


Quartets = Sequence[tuple[int, int, int, int]] | None


def exchange_gradient(basis: Basis, D: Matrix, nuclei: Sequence[Nucleus],
                      *, one_centre: OneCentre, quartets: Quartets = None,
                      atoms: Sequence[int] | None = None,
                      absolute: bool = False) -> Grad:
    """`dE_K / dR_{A,k}`, `E_K = 1/2 SUM D_ab D_cd (ac|bd)` (module
    docstring): the bra derivative alone over every ordered shell quartet,
    or over `quartets=` `(i, k, j, l)` -- bra `(sa, sc) = (i, k)`, ket
    `(sb, sd) = (j, l)`, K's Hermite pairing. `D` must be symmetric.
    `absolute=True` is the same sum with every factor in `|.|`, a
    round-off scale. `atoms=` is `_one_electron`'s."""
    nuc = _check_nuclei(nuclei)
    rows = checked_density(basis, D)
    if atoms is None:
        atoms = shell_atoms(basis, nuc)
        n_rows = len(nuc)
    else:  # the caller's own rows, as `_one_electron`'s
        atoms = tuple(atoms)
        if len(atoms) != len(basis.shells) or min(atoms) < len(nuc):
            raise ValueError("atoms= needs one row per shell, each after "
                             "the nuclei's rows")
        n_rows = max(atoms) + 1
    sh, off = basis.shells, basis.offsets
    n = len(sh)
    if quartets is None:
        quartets = [(i, k, j, m) for i in range(n) for k in range(n)
                    for j in range(n) for m in range(n)]

    def block(x: int, y: int) -> list[list[float]]:
        return [[rows[off[x] + i][off[y] + j] for j in range(sh[y].n_cart)]
                for i in range(sh[x].n_cart)]

    G = [[0.0, 0.0, 0.0] for _ in range(n_rows)]
    for i, k, j, m in quartets:
        if not all(0 <= x < n for x in (i, k, j, m)):
            raise ValueError(f"shell quartet {(i, k, j, m)} is out of range")
        _exchange_quartet(G, atoms[i], atoms[k], sh[i], sh[k], sh[j], sh[m],
                          block(i, j), block(k, m), one_centre=one_centre,
                          absolute=absolute)
    return G


def exchange_pairs(basis: Basis, la: int, lb: int
                   ) -> list[tuple[int, int]]:
    """Every ORDERED shell pair `(i, j)` of class `(la, lb)`, row-major."""
    sh = basis.shells
    return [(i, j) for i in range(len(sh)) for j in range(len(sh))
            if (sh[i].l, sh[j].l) == (la, lb)]


def class_quartets(basis: Basis, am: tuple[int, int, int, int]
                   ) -> list[tuple[int, int, int, int]]:
    """The ordered shell quartets `(i, k, j, l)` of K class `am = (la, lc,
    lb, ld)`: every bra pair of `exchange_pairs(la, lc)` against every ket
    pair of `exchange_pairs(lb, ld)`."""
    la, lc, lb, ld = am
    return [(i, k, j, m) for i, k in exchange_pairs(basis, la, lc)
            for j, m in exchange_pairs(basis, lb, ld)]


def exchange_records(basis: Basis, nuclei: Sequence[Nucleus],
                     am: tuple[int, int, int, int]) -> tuple[dict, dict]:
    """`kernel_gk`'s per-record arrays of K class `am = (la, lc, lb, ld)`,
    as lists: the bra side (`P`, `zeta_ac`, `XPA`, `XPC`, `s_ac`, `ea_bra`,
    `ec_bra`, `off_a`, `off_c`, `atom_a`, `atom_c`) over every pair of
    `exchange_pairs(la, lc)`, the ket side (`Q`, `eta`, `XQB`, `XQD`,
    `s_bd`, `off_b`, `off_d`) over `exchange_pairs(lb, ld)`, each in
    `primitive_pairs` order. One launch over both sides is every quartet
    of `class_quartets(am)`."""
    la, lc, lb, ld = am
    atoms = shell_atoms(basis, nuclei)
    sh, off = basis.shells, basis.offsets
    bra: dict[str, list] = {k: [] for k in (
        "P", "zeta_ac", "XPA", "XPC", "s_ac", "ea_bra", "ec_bra", "off_a",
        "off_c", "atom_a", "atom_c")}
    for i, k in exchange_pairs(basis, la, lc):
        A, C = sh[i].centre, sh[k].centre
        for zeta, P, s, alpha, gamma in _records(sh[i], sh[k]):
            bra["P"].append(tuple(P))
            bra["zeta_ac"].append(zeta)
            bra["XPA"].append(tuple(P[d] - A[d] for d in range(3)))
            bra["XPC"].append(tuple(P[d] - C[d] for d in range(3)))
            bra["s_ac"].append(s)
            bra["ea_bra"].append(alpha)
            bra["ec_bra"].append(gamma)
            bra["off_a"].append(off[i])
            bra["off_c"].append(off[k])
            bra["atom_a"].append(atoms[i])
            bra["atom_c"].append(atoms[k])
    ket: dict[str, list] = {k: [] for k in (
        "Q", "eta", "XQB", "XQD", "s_bd", "off_b", "off_d")}
    for j, m in exchange_pairs(basis, lb, ld):
        B, Dc = sh[j].centre, sh[m].centre
        for eta, Q, s, _, _ in _records(sh[j], sh[m]):
            ket["Q"].append(tuple(Q))
            ket["eta"].append(eta)
            ket["XQB"].append(tuple(Q[d] - B[d] for d in range(3)))
            ket["XQD"].append(tuple(Q[d] - Dc[d] for d in range(3)))
            ket["s_bd"].append(s)
            ket["off_b"].append(off[j])
            ket["off_d"].append(off[m])
    return bra, ket


def _hgp_side(basis: Basis, pairs, atoms) -> dict[str, list]:
    """One side of `hgp_exchange_launches`: `kernel_f`'s record per
    primitive pair, `i` outer, by `eri.hgp_drive.primitive_scalars`' own
    expressions (`U` carries `K_ab` and both coefficients)."""
    sh, off = basis.shells, basis.offsets
    out: dict[str, list] = {k: [] for k in ("prim", "exp", "sep", "ao",
                                           "atom")}
    for a, b in pairs:
        sa, sb = sh[a], sh[b]
        A, B = sa.centre, sb.centre
        AB = tuple(A[x] - B[x] for x in range(3))
        r2 = AB[0] ** 2 + AB[1] ** 2 + AB[2] ** 2
        for i, x in enumerate(sa.exponents):
            for j, y in enumerate(sb.exponents):
                p = x + y
                P = tuple((x * A[d] + y * B[d]) / p for d in range(3))
                U = (sa.coefficients[i] * sb.coefficients[j]
                     * (math.pi / p) ** 1.5 * math.exp(-x * y / p * r2))
                out["prim"] += [*P, *(P[d] - A[d] for d in range(3)), p, U]
                out["exp"] += [x, y]
        out["sep"] += list(AB)
        out["ao"] += [off[a], off[b]]
        out["atom"] += [atoms[a], atoms[b]]
    return out


def hgp_exchange_launches(basis: Basis, nuclei: Sequence[Nucleus],
                          am: tuple[int, int, int, int]) -> list[dict]:
    """`kernel_ghk`'s launches of K class `am = (la, lc, lb, ld)`: every
    ordered bra pair of `exchange_pairs(la, lc)` against every ordered ket
    pair of `exchange_pairs(lb, ld)`, cut by contraction degree (a side is
    degree-uniform, `kernel_f.pack`'s rule), one dict per `(k_ab, k_cd)`
    holding every parameter but `n_ao`, `n_atom`, `M` and `G`, as flat
    lists. Together the launches visit every quartet of
    `class_quartets(am)` once."""
    la, lc, lb, ld = am
    atoms = shell_atoms(basis, nuclei)
    sh = basis.shells

    def by_degree(pairs) -> dict[int, list]:
        out: dict[int, list] = {}
        for a, b in pairs:
            out.setdefault(len(sh[a].exponents) * len(sh[b].exponents),
                           []).append((a, b))
        return out

    bras = by_degree(exchange_pairs(basis, la, lc))
    kets = by_degree(exchange_pairs(basis, lb, ld))
    launches = []
    for kab, bp in bras.items():
        b = _hgp_side(basis, bp, atoms)
        for kcd, kp in kets.items():
            k = _hgp_side(basis, kp, atoms)
            launches.append({
                "n_bra": len(bp), "n_ket": len(kp), "k_ab": kab, "k_cd": kcd,
                "bra_prim": b["prim"], "bra_exp": b["exp"], "AB": b["sep"],
                "bra_ao": b["ao"], "bra_atom": b["atom"],
                "ket_prim": k["prim"], "CD": k["sep"], "ket_ao": k["ao"]})
    return launches


def nuclear_repulsion_gradient(nuclei: Sequence[Nucleus]) -> Grad:
    """`d/dR_A SUM_{B<C} Z_B Z_C / |R_B - R_C|`."""
    nuc = _check_nuclei(nuclei)
    G = [[0.0, 0.0, 0.0] for _ in nuc]
    for a, (Za, A) in enumerate(nuc):
        for b, (Zb, B) in enumerate(nuc):
            if a == b:
                continue
            r = [A[k] - B[k] for k in range(3)]
            d = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            for k in range(3):
                G[a][k] -= Za * Zb * r[k] / d ** 3
    return G


def class_pairs(basis: Basis, la: int, lb: int) -> list[tuple[int, int]]:
    """The shell pairs `i <= j` of class `(la, lb)`, row-major: the sweep's
    order, restricted to one class."""
    sh = basis.shells
    return [(i, j) for i, j in _unique_pairs(basis)
            if (sh[i].l, sh[j].l) == (la, lb)]


def records(basis: Basis, nuclei: Sequence[Nucleus], la: int, lb: int,
            *, pairs: Sequence[tuple[int, int]] | None = None
            ) -> dict[str, list]:
    """The d1 kernels' per-record arrays of class `(la, lb)` (one list per
    parameter, records in `class_pairs` then `primitive_pairs` order):
    `zeta_ab`, `p_*`, `pa_*`, `pb_*`, `s_ab`, `ea_bra`, `eb_ket`, `off_a`,
    `off_b`, `atom_a`, `atom_b`. The J chain's interleaved `XPA` / `XPB`
    are `pa_*` / `pb_*` stacked by the caller."""
    atoms = shell_atoms(basis, nuclei)
    if pairs is None:
        pairs = class_pairs(basis, la, lb)
    out: dict[str, list] = {k: [] for k in (
        "zeta_ab", "p_x", "p_y", "p_z", "pa_x", "pa_y", "pa_z", "pb_x",
        "pb_y", "pb_z", "s_ab", "ea_bra", "eb_ket", "off_a", "off_b",
        "atom_a", "atom_b")}
    for i, j in pairs:
        sa, sb = basis.shells[i], basis.shells[j]
        if (sa.l, sb.l) != (la, lb):
            raise ValueError(f"shell pair ({i}, {j}) is not class ({la}, {lb})")
        for zeta, P, s, alpha, beta in _records(sa, sb):
            out["zeta_ab"].append(zeta)
            for d, a in enumerate("xyz"):
                out[f"p_{a}"].append(P[d])
                out[f"pa_{a}"].append(P[d] - sa.centre[d])
                out[f"pb_{a}"].append(P[d] - sb.centre[d])
            out["s_ab"].append(s)
            out["ea_bra"].append(alpha)
            out["eb_ket"].append(beta)
            out["off_a"].append(basis.offsets[i])
            out["off_b"].append(basis.offsets[j])
            out["atom_a"].append(atoms[i])
            out["atom_b"].append(atoms[j])
    return out


def pair_records(basis: Basis, nuclei: Sequence[Nucleus], la: int, lb: int,
                 *, pairs: Sequence[tuple[int, int]] | None = None
                 ) -> dict[str, list]:
    """The HGP d1 kernels' arrays of class `(la, lb)`
    (`gradient.kernel_hgt`, one thread per contracted shell pair):
    `records`' per-record arrays less its per-record offsets and atoms,
    then one entry per shell pair -- `rec0` / `rec1` its record range
    (contiguous: `records` sweeps a pair's primitive pairs together),
    `off_a`, `off_b`, `atom_a`, `atom_b`, and `ab_*` = `A - B` from the
    shells' centres."""
    if pairs is None:
        pairs = class_pairs(basis, la, lb)
    rec = records(basis, nuclei, la, lb, pairs=pairs)
    out = {k: v for k, v in rec.items()
           if not k.startswith(("off_", "atom_"))}
    atoms = shell_atoms(basis, nuclei)
    sh = basis.shells
    per_pair: dict[str, list] = {k: [] for k in (
        "rec0", "rec1", "off_a", "off_b", "atom_a", "atom_b",
        "ab_x", "ab_y", "ab_z")}
    start = 0
    for i, j in pairs:
        stop = start + len(primitive_pairs(sh[i], sh[j]))
        per_pair["rec0"].append(start)
        per_pair["rec1"].append(stop)
        per_pair["off_a"].append(basis.offsets[i])
        per_pair["off_b"].append(basis.offsets[j])
        per_pair["atom_a"].append(atoms[i])
        per_pair["atom_b"].append(atoms[j])
        for d, a in enumerate("xyz"):
            per_pair[f"ab_{a}"].append(sh[i].centre[d] - sh[j].centre[d])
        start = stop
    assert start == len(rec["s_ab"]), "the pair ranges tile the records"
    return out | per_pair
