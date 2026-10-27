"""The host reference of the d2 kernels: the Coulomb energy's nuclear
Hessian.

    ket_derivative_records(basis, D, lc, ld)  -> per-record dD_q    (hdq's)
    coulomb_dkets(basis, D, nuclei)           -> {(C, l): [DKet]}  the kets
    coulomb_hessian(basis, D, nuclei, one_centre=) -> H   d2 E_J / dR dR
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
