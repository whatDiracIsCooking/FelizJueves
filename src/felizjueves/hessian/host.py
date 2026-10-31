"""The host reference of the d2 kernels: the Coulomb energy's nuclear
Hessian.

    ket_derivative_records(basis, D, lc, ld)  -> per-record dD_q    (hdq's)
    coulomb_dkets(basis, D, nuclei)           -> {(C, l): [DKet]}  the kets
    coulomb_hessian(basis, D, nuclei, one_centre=) -> H   d2 E_J / dR dR
    exchange_hessian(basis, D, nuclei, one_centre=) -> H  d2 E_K / dR dR
    exchange_hessian_records(basis, nuclei, am) -> hkx's per-record arrays
    nuclear_repulsion_hessian(nuclei)         -> H   d2 E_nuc / dR dR

`H` is `3 n_atom x 3 n_atom`, row and column `3 A + k` the coordinate
`R_{A,k}`: `H[3A + k][3B + l] = d2 E_J / dR_{A,k} dR_{B,l}`, `E_J = 1/2
SUM D_ab D_cd (ab|cd)` for a fixed symmetric `D` (the package docstring),
over every bra shell pair or `pairs=` (`i <= j`) only.

THE SWEEP is `gradient.host`'s: bra shell pairs `i <= j` row-major, then
`primitive_pairs` order, each pair's Cartesian pair weighted by
`gradient.host.pair_weights` at `M = D` (`D_ab + D_ba` off the diagonal),
the kets over every unique pair `c <= d` at weight 2 off the diagonal
(`gradient.host.coulomb_kets`). Per bra record (`alpha` on `A`, `beta` on
`B`, `s = c_a c_b (pi/zeta)^{3/2}`), with the per-axis undressed `E^{ij}`
of `mmd.hermite.e_axis` two indices higher on each side:

    dA E^{ij}    = 2 alpha E^{i+1,j}    - i E^{i-1,j}
    dB E^{ij}    = 2 beta  E^{i,j+1}    - j E^{i,j-1}
    dA dA E^{ij} = 2 alpha dA E^{i+1,j} - i dA E^{i-1,j}
    dA dB E^{ij} = 2 beta  dA E^{i,j+1} - j dA E^{i,j-1}
    dB dB E^{ij} = 2 beta  dB E^{i,j+1} - j dB E^{i,j-1}

on one axis, or a first derivative on each of two axes; the plain `E` on
the rest. Then, with every row's factor `w_r` and the Hermite densities
`HXY_{kl,t} = SUM_r w_r (dX_k dY_l E)^r_t` (`|t| <= L + 2`) and
`HX_{k,t} = SUM_r w_r (dX_k E)^r_t` (`|t| <= L + 1`):

- BRA-BRA: `J2_t`, the bra record's `J_p` at Hermite order `L + 2` over the
  kets (`gradient.host.coulomb_jp`), and

      H[A k][A l] += s SUM_t HAA_{kl,t} J2_t      (and B B)
      H[A k][B l] += s SUM_t HAB_{kl,t} J2_t,  H[B l][A k] the same

- BRA-KET: per atom `C` and axis `l`, `J1^{(C,l)}_t = SUM_q [t|q]
  D^{(C,l)}_q` at Hermite order `L + 1` over the DERIVATIVE KETS of `C`
  (`coulomb_dkets`), and

      H[A k][C l] += s SUM_t HA_{k,t} J1^{(C,l)}_t     (and B)

  never mirrored: summed over every bra this term is symmetric itself.

THE DERIVATIVE KETS. Per unique ket shell pair `c <= d` and primitive pair
(`gamma` on `C`, `delta` on `D`, the record scalar `s_cd`), with the
pair's weights `w_r` (`pair_weights` at `M = D`: the factor 2 of an
off-diagonal ket is in them):

    dD^C_{l,q} = s_cd SUM_r w_r (dC_l E^{cd})^r_q
    dD^D_{l,q} = s_cd SUM_r w_r (dD_l E^{cd})^r_q       |q| <= Lket + 1

and the record is a ket of `(atom_c, l)` carrying `dD^C_l` and a ket of
`(atom_d, l)` carrying `dD^D_l` (two kets of one atom when both shells sit
on it; the `[p|q]` sum is linear). `[t|q] = (-1)^{|q|} [t + q]^(0)`
(`coulomb_jp`), as for any ket.

THE EXCHANGE HESSIAN. `E_K = 1/2 SUM_abcd D_ab D_cd (ac|bd)` (K's
pairing: bra `(a, c)` on `A`, `C`, ket `(b, d)` on `B`, `D`). For a
symmetric `D` the same renaming as `E_J`'s (`(ac) <-> (bd)`, `D_ba =
D_ab`) cancels the half against the twos, so over every ORDERED quartet

    d2 E_K / dx dy = SUM D_ab D_cd [d_bra^x d_bra^y + d_bra^x d_ket^y] (ac|bd)

per (bra record, ket record), with `[r]^(0)` at `P - Q`, `|r| <= L + 2`:

- BRA-BRA: per `(b,d)` row `I0[p] = SUM_q (-1)^{|q|} [p+q]^(0) s_bd
  E^{bd}_q` over `hermite_order(lp + 2)`, then per `(a,c)` row `s_ac
  D_ab D_cd SUM_p (dX_k dY_l E^{ac})_p I0[p]`, `XY` = `AA`, `AC`, `CC`
  (the `AC` value also on `[C l][A k]`);
- BRA-KET: per `(b,d)` row, ket centre `Y` (`B` or `D`) and axis `l`,
  `I1[p] = SUM_q (-1)^{|q|} [p+q]^(0) s_bd (dY_l E^{bd})_q` over
  `hermite_order(lp + 1)`, `q` to `lq + 1`, then `s_ac D_ab D_cd SUM_p
  (dX_k E^{ac})_p I1[p]` on `[X k][Y l]`, never mirrored.

The derivatives of `E^{ac}` and `E^{bd}` are formed BEFORE the fold, by
the rule above on `_Axes` (the kernel folds raised rows instead). The RHF
exchange energy is `-1/2 E_K`: the factor is the caller's.

`absolute=True` is the same sum with every factor in `|.|` (the kernel
tests' round-off scale). `atoms=` gives each shell a row of its own after
the nuclei's, for the bra AND the kets; `[t]^(0)` is INJECTED
(`one_centre=`), as in `gradient.host`. Pure Python floats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from felizjueves.ao.density import checked_density
from felizjueves.ao.pairs import OneCentre, primitive_pairs
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.gradient.host import (
    class_pairs,
    coulomb_jp,
    coulomb_kets,
    exchange_pairs,
    exchange_records,
    pair_weights,
    shell_atoms,
)
from felizjueves.mmd.hermite import e_axis, hermite_order
from felizjueves.one_electron.potential import Nucleus

Ang = tuple[int, int, int]
Hess = list[list[float]]
Pairs = Sequence[tuple[int, int]] | None

# The second-derivative kinds: both on the first centre, one on each, both
# on the second.
KINDS = ("AA", "AB", "BB")


@dataclass(frozen=True)
class DKet:
    """One derivative ket: what `gradient.host.coulomb_jp` reads of a ket
    (`Q`, `eta`, its Hermite indices `qs` at `Lket + 1` and the density
    `dq`)."""

    Q: tuple[float, float, float]
    eta: float
    qs: tuple[Ang, ...]
    dq: tuple[float, ...]


def _sub(x: Sequence[float], c: float, y: Sequence[float] | None,
         n: int, plus: bool = False) -> list[float]:
    """`c x - n y` entry by entry, `y` shorter (absent when `n == 0`);
    `c x + n y` when `plus` (the absolute twin)."""
    out = [c * v for v in x]
    if y is not None and n:
        for t, v in enumerate(y):
            if plus:
                out[t] += n * v
            else:
                out[t] -= n * v
    return out


class _Axes:
    """One primitive pair's per-axis `E^{ij}_t` and its derivatives,
    `i <= la + 2`, `j <= lb + 2` (module docstring); `absolute`, every
    entry and every term in `|.|`."""

    def __init__(self, zeta: float, P, A, B, imax: int, jmax: int,
                 alpha: float, beta: float, absolute: bool = False) -> None:
        self.tab = [[[tuple(abs(v) for v in e_axis(i, j, zeta, P[d] - A[d],
                                                      P[d] - B[d]))
                      if absolute else
                      e_axis(i, j, zeta, P[d] - A[d], P[d] - B[d])
                      for j in range(jmax + 1)] for i in range(imax + 1)]
                    for d in range(3)]
        self.a2, self.b2 = 2.0 * alpha, 2.0 * beta
        self.plus = absolute

    def _sub(self, x, c, y, n) -> list[float]:
        return _sub(x, c, y, n, self.plus)

    def e(self, d: int, i: int, j: int) -> list[float]:
        return list(self.tab[d][i][j])

    def dA(self, d: int, i: int, j: int) -> list[float]:
        return self._sub(self.e(d, i + 1, j), self.a2,
                    self.e(d, i - 1, j) if i else None, i)

    def dB(self, d: int, i: int, j: int) -> list[float]:
        return self._sub(self.e(d, i, j + 1), self.b2,
                    self.e(d, i, j - 1) if j else None, j)

    def d2(self, kind: str, d: int, i: int, j: int) -> list[float]:
        if kind == "AA":
            return self._sub(self.dA(d, i + 1, j), self.a2,
                        self.dA(d, i - 1, j) if i else None, i)
        if kind == "AB":
            return self._sub(self.dA(d, i, j + 1), self.b2,
                        self.dA(d, i, j - 1) if j else None, j)
        return self._sub(self.dB(d, i, j + 1), self.b2,
                    self.dB(d, i, j - 1) if j else None, j)

    def first(self, side: str, d: int, i: int, j: int) -> list[float]:
        return self.dA(d, i, j) if side == "A" else self.dB(d, i, j)


def _at(f: Sequence[float], t: int) -> float:
    return f[t] if t < len(f) else 0.0


def _hermite(f: list[Sequence[float]], ts: Sequence[Ang]) -> list[float]:
    return [_at(f[0], t[0]) * _at(f[1], t[1]) * _at(f[2], t[2]) for t in ts]


def _rows(sa: Shell, sb: Shell) -> list[tuple[Ang, Ang]]:
    return [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]


def first_densities(ax: _Axes, sa: Shell, sb: Shell, w: Sequence[float]
                    ) -> dict[tuple[str, int], list[float]]:
    """`HA_k`, `HB_k` over `hermite_order(L + 1)`, keyed `("A", k)`."""
    ts = hermite_order(sa.l + sb.l + 1)
    out = {(s, k): [0.0] * len(ts) for s in "AB" for k in range(3)}
    for wr, (a, b) in zip(w, _rows(sa, sb), strict=True):
        for (side, k), H in out.items():
            f = [ax.e(d, a[d], b[d]) for d in range(3)]
            f[k] = ax.first(side, k, a[k], b[k])
            for n, v in enumerate(_hermite(f, ts)):
                H[n] += wr * v
    return out


def second_densities(ax: _Axes, sa: Shell, sb: Shell, w: Sequence[float]
                     ) -> dict[tuple[str, int, int], list[float]]:
    """`HXY_{kl}` over `hermite_order(L + 2)`, keyed `(kind, k, l)`: `k`
    the axis of the first letter's centre, `l` the second's."""
    ts = hermite_order(sa.l + sb.l + 2)
    out = {(kind, k, l): [0.0] * len(ts)
           for kind in KINDS for k in range(3) for l in range(3)}  # noqa: E741
    for wr, (a, b) in zip(w, _rows(sa, sb), strict=True):
        for (kind, k, l), H in out.items():  # noqa: E741
            f = [ax.e(d, a[d], b[d]) for d in range(3)]
            if k == l:
                f[k] = ax.d2(kind, k, a[k], b[k])
            else:
                f[k] = ax.first(kind[0], k, a[k], b[k])
                f[l] = ax.first(kind[1], l, a[l], b[l])
            for n, v in enumerate(_hermite(f, ts)):
                H[n] += wr * v
    return out


def _records(sa: Shell, sb: Shell):
    """`(zeta, P, s, alpha, beta)` per primitive pair (`gradient.host`'s)."""
    for pp in primitive_pairs(sa, sb):
        ca, cb = sa.coefficients[pp.i], sb.coefficients[pp.j]
        s = ca * cb * (math.pi / pp.zeta) ** 1.5
        yield (pp.zeta, pp.P, s, sa.exponents[pp.i], sb.exponents[pp.j])


def _unique_pairs(basis: Basis) -> list[tuple[int, int]]:
    n = len(basis.shells)
    return [(i, j) for i in range(n) for j in range(i, n)]


def _atoms(basis: Basis, nuclei: Sequence[Nucleus],
           atoms: Sequence[int] | None) -> tuple[tuple[int, ...], int]:
    if atoms is None:
        return shell_atoms(basis, nuclei), len(nuclei)
    atoms = tuple(atoms)
    if len(atoms) != len(basis.shells) or min(atoms) < len(nuclei):
        raise ValueError("atoms= needs one row per shell, each after the "
                         "nuclei's rows")
    return atoms, max(atoms) + 1


def _ket_derivatives(basis: Basis, rows: list[list[float]], c: int, d: int):
    """Per primitive pair of ket shell pair `(c, d)`: `(P, eta, {(side,
    l): dD})` (module docstring, "THE DERIVATIVE KETS")."""
    sc, sd = basis.shells[c], basis.shells[d]
    w = pair_weights(rows, c, d, basis)
    for zeta, P, s, gamma, delta in _records(sc, sd):
        ax = _Axes(zeta, P, sc.centre, sd.centre, sc.l + 1, sd.l + 1,
                   gamma, delta)
        dens = first_densities(ax, sc, sd, w)
        yield P, zeta, {key: [s * v for v in H] for key, H in dens.items()}


def ket_derivative_records(basis: Basis, D, lc: int, ld: int, *,
                           pairs: Pairs = None) -> list[list[list[float]]]:
    """`hdq`'s output for ket class `(lc, ld)`: `[6][n_rec][n_q1]`, slab
    `3 side + l` (`side` 0 the first shell `c`, 1 the second `d`), records
    in `gradient.host.class_pairs` then `primitive_pairs` order."""
    rows = checked_density(basis, D)
    out: list[list[list[float]]] = [[] for _ in range(6)]
    for c, d in (class_pairs(basis, lc, ld) if pairs is None else pairs):
        for _, _, dens in _ket_derivatives(basis, rows, c, d):
            for side, s in enumerate("AB"):
                for l in range(3):  # noqa: E741
                    out[3 * side + l].append(dens[s, l])
    return out


def coulomb_dkets(basis: Basis, D, nuclei: Sequence[Nucleus], *,
                  atoms: Sequence[int] | None = None
                  ) -> dict[tuple[int, int], list[DKet]]:
    """The derivative kets of every atom `C` and axis `l` (module
    docstring), over every unique ket pair `c <= d`."""
    at, n_rows = _atoms(basis, nuclei, atoms)
    rows = checked_density(basis, D)
    out: dict[tuple[int, int], list[DKet]] = {
        (C, l): [] for C in range(n_rows) for l in range(3)}  # noqa: E741
    for c, d in _unique_pairs(basis):
        qs = hermite_order(basis.shells[c].l + basis.shells[d].l + 1)
        for P, eta, dens in _ket_derivatives(basis, rows, c, d):
            for side, atom in (("A", at[c]), ("B", at[d])):
                for l in range(3):  # noqa: E741
                    out[atom, l].append(DKet(tuple(P), eta, qs,
                                             tuple(dens[side, l])))
    return out


def _dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(x, y, strict=True))


def coulomb_hessian(basis: Basis, D, nuclei: Sequence[Nucleus], *,
                    one_centre: OneCentre, pairs: Pairs = None,
                    atoms: Sequence[int] | None = None, kets=None,
                    dkets=None, parts: str = "both",
                    absolute: bool = False) -> Hess:
    """`d2 E_J / dR dR` (module docstring), the bra-bra and bra-ket terms
    (`parts` "both", or "bb" / "bk" for one of them). `kets` is
    `gradient.host.coulomb_kets(basis, D)` and `dkets`
    `coulomb_dkets(basis, D, nuclei, atoms=atoms)`, passed when the caller
    has them. `absolute=True` is the same sum with every factor -- each
    `E` entry, each derivative term, `w_r`, `s` and each `J_t` -- in
    `|.|`: the size of what cancels, the kernel tests' round-off scale."""
    fix = abs if absolute else (lambda v: v)
    if parts not in ("both", "bb", "bk"):
        raise ValueError(f"parts = {parts!r} is not 'both', 'bb' or 'bk'")
    at, n_rows = _atoms(basis, nuclei, atoms)
    rows = checked_density(basis, D)
    if kets is None:
        kets = coulomb_kets(basis, rows)
    if dkets is None:
        dkets = coulomb_dkets(basis, rows, nuclei, atoms=atoms)
    H = [[0.0] * (3 * n_rows) for _ in range(3 * n_rows)]
    for i, j in (_unique_pairs(basis) if pairs is None else pairs):
        if not 0 <= i <= j < len(basis.shells):
            raise ValueError(f"shell pair ({i}, {j}) is not i <= j")
        sa, sb = basis.shells[i], basis.shells[j]
        L = sa.l + sb.l
        w = [fix(v) for v in pair_weights(rows, i, j, basis)]
        cen = {"A": at[i], "B": at[j]}
        for zeta, P, s, alpha, beta in _records(sa, sb):
            s = fix(s)
            ax = _Axes(zeta, P, sa.centre, sb.centre, sa.l + 2, sb.l + 2,
                       alpha, beta, absolute)
            if parts != "bk":
                j2 = [fix(v) for v in coulomb_jp(P, zeta, L + 2, kets,
                                                 one_centre=one_centre)]
                for (kind, k, l), Hd in second_densities(ax, sa, sb,  # noqa: E741
                                                         w).items():
                    v = s * _dot(Hd, j2)
                    x, y = 3 * cen[kind[0]] + k, 3 * cen[kind[1]] + l
                    H[x][y] += v
                    if kind == "AB":
                        H[y][x] += v
            if parts != "bb":
                first = first_densities(ax, sa, sb, w)
                for (C, l), ks in dkets.items():  # noqa: E741
                    if not ks:
                        continue
                    j1 = [fix(v) for v in coulomb_jp(P, zeta, L + 1, ks,
                                                     one_centre=one_centre)]
                    for (side, k), Hd in first.items():
                        H[3 * cen[side] + k][3 * C + l] += s * _dot(Hd, j1)
    return H


Quartets = Sequence[tuple[int, int, int, int]] | None


def _fold_q(R: dict, col: Sequence[float], ps: Sequence[Ang],
            qs: Sequence[Ang], absolute: bool) -> list[float]:
    """`SUM_q (-1)^{|q|} [p+q]^(0) col_q` for every `p` (`+` throughout
    when `absolute`)."""
    out = []
    for p in ps:
        acc = 0.0
        for q, e in zip(qs, col, strict=True):
            x = R[(p[0] + q[0], p[1] + q[1], p[2] + q[2])] * e
            acc += x if absolute or sum(q) % 2 == 0 else -x
        out.append(acc)
    return out


def _exchange_quartet(H: Hess, cb: dict[str, int], ck: dict[str, int],
                      sa: Shell, sc: Shell, sb: Shell, sd: Shell, Dab, Dcd,
                      *, one_centre: OneCentre, parts: str,
                      absolute: bool) -> None:
    """One ordered shell quartet's `SUM D_ab D_cd [bra bra + bra ket]
    (ac|bd)` into `H` (module docstring, THE EXCHANGE HESSIAN). `cb` /
    `ck` map `_Axes`' centre letters, `A` the first shell and `B` the
    second, to the bra's and the ket's atom rows."""
    fix = abs if absolute else float
    lp, lq = sa.l + sc.l, sb.l + sd.l
    p2, p1 = hermite_order(lp + 2), hermite_order(lp + 1)
    q0, q1 = hermite_order(lq), hermite_order(lq + 1)
    rows_ac, rows_bd = _rows(sa, sc), _rows(sb, sd)
    nc, nd = sc.n_cart, sd.n_cart
    kets = []
    for eta, Q, s_bd, beta, delta in _records(sb, sd):
        ax = _Axes(eta, Q, sb.centre, sd.centre, sb.l + 1, sd.l + 1,
                   beta, delta, absolute)
        sq = fix(s_bd)
        plain = [[sq * v for v in _hermite(
            [ax.e(d, b[d], e[d]) for d in range(3)], q0)]
            for b, e in rows_bd]
        first: dict[tuple[str, int], list[list[float]]] = {}
        for side in "AB":
            for l in range(3):  # noqa: E741
                rows = []
                for b, e in rows_bd:
                    f = [ax.e(d, b[d], e[d]) for d in range(3)]
                    f[l] = ax.first(side, l, b[l], e[l])
                    rows.append([sq * v for v in _hermite(f, q1)])
                first[side, l] = rows
        kets.append((eta, Q, plain, first))

    for zeta, P, s_ac, alpha, gamma in _records(sa, sc):
        ax = _Axes(zeta, P, sa.centre, sc.centre, sa.l + 2, sc.l + 2,
                   alpha, gamma, absolute)
        sec: dict[tuple[str, int, int], list[list[float]]] = {}
        fst: dict[tuple[str, int], list[list[float]]] = {}
        for a, c in rows_ac:
            for kind in KINDS:
                for k in range(3):
                    for l in range(3):  # noqa: E741
                        f = [ax.e(d, a[d], c[d]) for d in range(3)]
                        if k == l:
                            f[k] = ax.d2(kind, k, a[k], c[k])
                        else:
                            f[k] = ax.first(kind[0], k, a[k], c[k])
                            f[l] = ax.first(kind[1], l, a[l], c[l])
                        sec.setdefault((kind, k, l), []).append(
                            _hermite(f, p2))
            for side in "AB":
                for k in range(3):
                    f = [ax.e(d, a[d], c[d]) for d in range(3)]
                    f[k] = ax.first(side, k, a[k], c[k])
                    fst.setdefault((side, k), []).append(_hermite(f, p1))
        acc: dict[tuple[int, int], float] = {}
        for eta, Q, plain, first in kets:
            theta = math.sqrt(zeta * eta / (zeta + eta))
            PQv = (P[0] - Q[0], P[1] - Q[1], P[2] - Q[2])
            R = {r: fix(one_centre(r, 0, PQv, theta))
                 for r in hermite_order(lp + lq + 2)}
            for rbd in range(len(rows_bd)):
                ib, id_ = divmod(rbd, nd)
                I0 = (_fold_q(R, plain[rbd], p2, q0, absolute)
                      if parts != "bk" else None)
                I1 = ({key: _fold_q(R, rows[rbd], p1, q1, absolute)
                       for key, rows in first.items()}
                      if parts != "bb" else {})
                for rac in range(len(rows_ac)):
                    ia, ic = divmod(rac, nc)
                    w = fix(Dab[ia][ib]) * fix(Dcd[ic][id_])
                    if I0 is not None:
                        for (kind, k, l), rows in sec.items():  # noqa: E741
                            v = w * _dot(rows[rac], I0)
                            xy = (3 * cb[kind[0]] + k, 3 * cb[kind[1]] + l)
                            acc[xy] = acc.get(xy, 0.0) + v
                            if kind == "AB":
                                yx = (xy[1], xy[0])
                                acc[yx] = acc.get(yx, 0.0) + v
                    for (side, k), rows in fst.items():
                        for (kside, l), col in I1.items():  # noqa: E741
                            xy = (3 * cb[side] + k, 3 * ck[kside] + l)
                            acc[xy] = acc.get(xy, 0.0) + w * _dot(
                                rows[rac], col)
        for (x, y), v in acc.items():
            H[x][y] += fix(s_ac) * v


def exchange_hessian(basis: Basis, D, nuclei: Sequence[Nucleus], *,
                     one_centre: OneCentre, quartets: Quartets = None,
                     atoms: Sequence[int] | None = None, parts: str = "both",
                     absolute: bool = False) -> Hess:
    """`d2 E_K / dR dR`, `E_K = 1/2 SUM D_ab D_cd (ac|bd)` (module
    docstring, THE EXCHANGE HESSIAN): the bra-bra and bra-ket terms over
    every ordered shell quartet, or over `quartets=` `(i, k, j, l)` -- bra
    `(sa, sc) = (i, k)`, ket `(sb, sd) = (j, l)`, K's Hermite pairing
    (`gradient.host.class_quartets`). `D` must be symmetric. `parts`,
    `atoms=` and `absolute=` are `coulomb_hessian`'s."""
    if parts not in ("both", "bb", "bk"):
        raise ValueError(f"parts = {parts!r} is not 'both', 'bb' or 'bk'")
    at, n_rows = _atoms(basis, nuclei, atoms)
    rows = checked_density(basis, D)
    sh, off = basis.shells, basis.offsets
    n = len(sh)
    if quartets is None:
        quartets = [(i, k, j, m) for i in range(n) for k in range(n)
                    for j in range(n) for m in range(n)]

    def block(x: int, y: int) -> list[list[float]]:
        return [[rows[off[x] + i][off[y] + j] for j in range(sh[y].n_cart)]
                for i in range(sh[x].n_cart)]

    H = [[0.0] * (3 * n_rows) for _ in range(3 * n_rows)]
    for i, k, j, m in quartets:
        if not all(0 <= x < n for x in (i, k, j, m)):
            raise ValueError(f"shell quartet {(i, k, j, m)} is out of range")
        _exchange_quartet(H, {"A": at[i], "B": at[k]},
                          {"A": at[j], "B": at[m]}, sh[i], sh[k], sh[j],
                          sh[m], block(i, j), block(k, m),
                          one_centre=one_centre, parts=parts,
                          absolute=absolute)
    return H


def exchange_hessian_records(basis: Basis, nuclei: Sequence[Nucleus],
                             am: tuple[int, int, int, int], *,
                             bra_pairs: Sequence[tuple[int, int]] | None = None,
                             ket_pairs: Sequence[tuple[int, int]] | None = None
                             ) -> tuple[dict, dict, dict]:
    """`kernel_hkx`'s per-record arrays of K class `am = (la, lc, lb, ld)`,
    as lists: `gradient.host.exchange_records`' bra side unchanged; its
    ket side with the ket's exponents `eb_ket` (on `B`) and `ed_ket` (on
    `D`) added, STABLY regrouped by the ket's atom pair `(atom_b,
    atom_d)`; and the segments of that grouping, `seg_start` (`n_seg + 1`
    record offsets), `seg_b`, `seg_d` (each segment's two atoms). One
    launch over both sides is every quartet of `class_quartets(am)`.

    `bra_pairs` / `ket_pairs` (ordered shell pairs of the class; None, the
    default, for every one) keep only those pairs' records, so one launch
    is the quartets of `bra_pairs x ket_pairs`: a witness over part of a
    class (#691)."""
    _, _, lb, ld = am
    bra, ket = exchange_records(basis, nuclei, am)
    atoms = shell_atoms(basis, nuclei)
    sh, off = basis.shells, basis.offsets
    if bra_pairs is not None:
        keep_ac = {(off[i], off[k]) for i, k in bra_pairs}
        rows = [r for r, o in enumerate(zip(bra["off_a"], bra["off_c"],
                                            strict=True)) if o in keep_ac]
        bra = {k: [v[r] for r in rows] for k, v in bra.items()}
    extra: list[tuple[float, float, int, int]] = []
    pair_of: list[tuple[int, int]] = []
    for j, m in exchange_pairs(basis, lb, ld):
        for _, _, _, beta, delta in _records(sh[j], sh[m]):
            extra.append((beta, delta, atoms[j], atoms[m]))
            pair_of.append((j, m))
    if len(extra) != len(ket["s_bd"]):
        raise AssertionError("the ket records and their exponents disagree")
    kept = (range(len(extra)) if ket_pairs is None else
            [r for r in range(len(extra))
             if pair_of[r] in {tuple(p) for p in ket_pairs}])
    order = sorted(kept, key=lambda r: extra[r][2:])
    out = {k: [v[r] for r in order] for k, v in ket.items()}
    out["eb_ket"] = [extra[r][0] for r in order]
    out["ed_ket"] = [extra[r][1] for r in order]
    seg: dict[str, list[int]] = {"seg_start": [], "seg_b": [], "seg_d": []}
    last = None
    for n, r in enumerate(order):
        if extra[r][2:] != last:
            last = extra[r][2:]
            seg["seg_start"].append(n)
            seg["seg_b"].append(last[0])
            seg["seg_d"].append(last[1])
    seg["seg_start"].append(len(order))
    return bra, out, seg


def nuclear_repulsion_hessian(nuclei: Sequence[Nucleus]) -> Hess:
    """`d2/dR dR SUM_{B<C} Z_B Z_C / |R_B - R_C|`: `Z_A Z_B (3 r r^T / r^5
    - I / r^3)` on the four blocks of each pair, signed."""
    nuc = [(float(Z), tuple(float(x) for x in C)) for Z, C in nuclei]
    n = len(nuc)
    H = [[0.0] * (3 * n) for _ in range(3 * n)]
    for a in range(n):
        for b in range(a + 1, n):
            (Za, A), (Zb, B) = nuc[a], nuc[b]
            r = [A[k] - B[k] for k in range(3)]
            d = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            for k in range(3):
                for l in range(3):  # noqa: E741
                    v = Za * Zb * (3.0 * r[k] * r[l] / d ** 5
                                   - (k == l) / d ** 3)
                    H[3 * a + k][3 * a + l] += v
                    H[3 * b + k][3 * b + l] += v
                    H[3 * a + k][3 * b + l] -= v
                    H[3 * b + k][3 * a + l] -= v
    return H
