"""The three-center Coulomb integral `(mn|P)` of density fitting, host-side.

    eri3_block_and_scale(sA, sB, sP, *, pq) -> ((mn|P) block, its |.| twin)
    eri3_block(sA, sB, sP, *, pq)           -> (mn|P) of one ordered shell triple
    eri3_tensor(basis, aux, *, pq)          -> (mn|P) over every (m, n, P)
    scale_tensor(basis, aux, *, pq)         -> the scale of every element, alike
    eri3_classes(basis, aux)                -> the staging, per class (la, lb, lP)
    class_block(cls, k, *, pq)              -> staged cell k's block, from records
    place_staged(cls, k, block, out)        -> the scatter, transposing a swap

`(mn|P) = INT INT m(r1) n(r1) P(r2) / r12` (#443; `Target.ERI3`, spec.py):
`m`, `n` AOs of an ordinary basis, `P` a function of an auxiliary basis, both
ordinary contracted Cartesian `Shell`s (`ao/shell.py`).

THE INTEGRAL. For a bra primitive pair `ij` of `(sA, sB)` (`p = zeta_i +
zeta_j`, centre `P_ij`) and an aux primitive `k` of `sP` (exponent `alpha_k`,
centre `A_k`),

    (mn|P) = sum_{i in m} sum_{j in n} sum_{k in P} w_ij s_k
             sum_t E^{mn}_t(i, j) sum_tau (-1)^{|tau|}
                   [t + tau]^(0)(P_ij - A_k, theta) E^P_tau(k),
    theta^2 = p alpha_k / (p + alpha_k),

the `[p|q]` contraction of `eri.eri` with the ket's Hermite "pair" a single
Gaussian (`eri.density_fit.metric`'s module docstring: `zeta = alpha`, `Q = A`, `X_PA =
0`,
no `K_CD`). `[t|tau]` is the INJECTED evaluator's (`pq(t, tau, PQ, theta)`,
sign and `(2/sqrt(pi)) theta` inside it); `ao` imports neither `reference/`
nor `plan/`.

THE TWO SIDES, each dressed where its own module dresses it, and nowhere here:

- THE BRA is `ao.pairs.primitive_pairs`' record: `w_ij E^{mn}` is its DRESSED
  block, `c_i c_j (pi/p)^{3/2}` multiplied in by `ao.pairs.dress` (the only
  place a bra coefficient is), the Gaussian product `K_AB` in `E`'s seed
  (`mmd.hermite`).
- THE KET is `eri.density_fit.metric.aux_records` unchanged: `s_k = c_k
(pi/alpha_k)^{3/2}`
  by `eri.density_fit.metric.aux_dress`, with NO extra normalisation (it is in `c`; see
  `eri.density_fit.metric`), and `E^P` is `eri.density_fit.metric.center_e`.

So `(2/sqrt(pi)) theta (pi/p)^{3/2} (pi/alpha)^{3/2} = 2 pi^{5/2} / (p alpha
sqrt(p + alpha))` is McMurchie & Davidson's `lambda`, as for `(P|Q)` and
`(ab|cd)`.

THE SCALE is the same sum with every term replaced by its absolute value
(`eri.eri`'s and `eri.density_fit.metric`'s), so a tolerance relative to it is a
statement
about rounding, not about how much the sum cancelled.

THE LAYOUT, decided here and honoured by the kernel: `[m][n][P]`, the aux
index FASTEST. A block is `block[a][b][q]` over `cartesians(la)`,
`cartesians(lb)`, `cartesians(lP)`; the tensor is `(n_ao, n_ao, n_aux)` as
nested lists. `eri3_tensor` places every AO shell pair `i <= j` in basis order
at `[off_i + a][off_j + b][off_P + q]` and its `(m, n)` mirror; a DIAGONAL
shell pair writes `a >= b` only, mirrored (`eri.density_fit.metric`'s, `eri.schwarz`'s
and
`kernel_jab`'s rule), so `(mn|P) == (nm|P)` EXACTLY in the tensor, whereas
the two blocks `(ab|P)` and `(ba|P)` are equal exactly but not always
bitwise (`E^{ab}` and `E^{ba}` round differently).

THE STAGING (`eri3_classes`), shaped for the kernel (#444: one thread per
cell, a cell one (AO shell pair, aux shell); the thread walks the cell's bra
record range, then its aux record range). Every AO shell pair `i <= j` by
index, crossed with every aux shell, is a cell of class `(la, lb, lP)` with
`la <= lb`: a pair whose first shell has the higher `l` is staged SWAPPED,
`(j, i)`, the flag recorded, and its staged block lands with `m, n`
TRANSPOSED at `[off_i][off_j]` (`place_staged`; with the mirror either
placement fills the same elements). A class holds two record tables, each
entry once, contiguous, in first-touch order: the BRA records of every shell
pair it touches (`zeta_ab`, `P`, `XPA = P - A`, `XPB = P - B`, `s_ab = c_i c_j
(pi/p)^{3/2}`, `eri.schwarz`'s record) and the AUX records of every aux shell
it touches (`A`, `alpha`, `s`, `eri.density_fit.metric`'s). A cell is two ranges into
them, `[ab_rec0, ab_rec1)` and `[p_rec0, p_rec1)`, and three offsets.

`class_block` re-forms each bra `E` from the record alone, as
`s_ab * e_block(a, b, zeta, 0, -XPA, -XPB)`: `e_block` reads only `p - A` and
`p - B`, and `0 - (-x) = x` exactly, so this is `primitive_pairs`' dressed
block bitwise (`s_ab` is `dress`'s factor and `dress` forms `factor * E`), and
`class_block` is `eri3_block` of the staged shells bitwise.

Plain tuples and floats, like the rest of `ao/`: `src/felizjueves` has no
runtime dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from felizjueves.ao.pairs import PrimitivePair, dress, primitive_pairs
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.eri.density_fit.metric import AuxRec, aux_records, center_e
from felizjueves.mmd.hermite import e_block, hermite_order

Vec = tuple[float, float, float]
Ang = tuple[int, int, int]
PQ = Callable[[Ang, Ang, Vec, float], float]  # eri.eri's, restated
Block = tuple[tuple[tuple[float, ...], ...], ...]  # [a][b][q]
Tensor = list[list[list[float]]]  # [m][n][P]

_ORIGIN: Vec = (0.0, 0.0, 0.0)


def _matmul(a: Sequence[Sequence[float]],
            b: Sequence[Sequence[float]]) -> list[list[float]]:
    out = []
    for row in a:
        acc = [0.0] * len(b[0])
        for x, brow in zip(row, b, strict=True):
            if x:
                for k, y in enumerate(brow):
                    acc[k] += x * y
        out.append(acc)
    return out


def _abs(a: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[abs(x) for x in r] for r in a]


def _add(into: list[list[float]], add: list[list[float]]) -> None:
    for r_into, r_add in zip(into, add, strict=True):
        for k, v in enumerate(r_add):
            r_into[k] += v


def _block(la: int, lb: int, bras: Sequence[PrimitivePair], lP: int,
           auxs: Sequence[AuxRec], pq: PQ, want_scale: bool):
    """The sum over the bra and aux record ranges (module docstring), bra
    outer: `E_bra [t|tau] (s E_P)^T` per (bra, aux) primitive pair. Rows are
    the AO pairs `(a, b)` (a outer), columns `cartesians(lP)`."""
    ts, taus = hermite_order(la + lb), hermite_order(lP)
    nab = len(cartesians(la)) * len(cartesians(lb))
    nP = len(cartesians(lP))
    total = [[0.0] * nP for _ in range(nab)]
    scale = [[0.0] * nP for _ in range(nab)] if want_scale else None
    # (s E_P)^T per aux record, rows tau, columns the aux Cartesians
    aux_t = [(r, [list(c) for c in zip(*[[r.s * x for x in row]
                                         for row in center_e(lP, r.alpha)],
                                       strict=True)])
             for r in auxs]
    for bra in bras:
        for r, ep_t in aux_t:
            R = (bra.P[0] - r.A[0], bra.P[1] - r.A[1], bra.P[2] - r.A[2])
            theta = math.sqrt(bra.zeta * r.alpha / (bra.zeta + r.alpha))
            m = [[pq(t, tau, R, theta) for tau in taus] for t in ts]
            _add(total, _matmul(bra.block, _matmul(m, ep_t)))
            if scale is not None:
                _add(scale, _matmul(_abs(bra.block),
                                    _matmul(_abs(m), _abs(ep_t))))
    return total, scale


def _shape(rows: list[list[float]], nb: int) -> Block:
    """AO-pair rows `(a, b)`, a outer, as `[a][b][q]`."""
    return tuple(tuple(tuple(rows[a * nb + b]) for b in range(nb))
                 for a in range(len(rows) // nb))


def eri3_block_and_scale(sA: Shell, sB: Shell, sP: Shell, *, pq: PQ
                         ) -> tuple[Block, Block]:
    """`(mn|P)` of the ordered shell triple `(sA, sB | sP)` and its |.| twin,
    each `[a][b][q]`."""
    total, scale = _block(sA.l, sB.l, primitive_pairs(sA, sB), sP.l,
                          aux_records(sP), pq, True)
    nb = sB.n_cart
    return _shape(total, nb), _shape(scale, nb)


def eri3_block(sA: Shell, sB: Shell, sP: Shell, *, pq: PQ) -> Block:
    """`(mn|P)`: `block[a][b][q]` over `cartesians(sA.l)`, `cartesians(sB.l)`,
    `cartesians(sP.l)`."""
    return _shape(_block(sA.l, sB.l, primitive_pairs(sA, sB), sP.l,
                         aux_records(sP), pq, False)[0], sB.n_cart)


def _place(basis: Basis, aux: Basis,
           value: Callable[[Shell, Shell, Shell], Block]) -> Tensor:
    sh, off = basis.shells, basis.offsets
    ash, aoff = aux.shells, aux.offsets
    n, naux = basis.n_ao, aux.n_ao
    out = [[[0.0] * naux for _ in range(n)] for _ in range(n)]
    for i in range(len(sh)):
        for j in range(i, len(sh)):
            for k in range(len(ash)):
                blk = value(sh[i], sh[j], ash[k])
                for a in range(sh[i].n_cart):
                    for b in range(sh[j].n_cart):
                        if i == j and b > a:
                            continue  # diagonal: the lower triangle, mirrored
                        m, nn = off[i] + a, off[j] + b
                        for q, v in enumerate(blk[a][b]):
                            out[m][nn][aoff[k] + q] = v
                            out[nn][m][aoff[k] + q] = v
    return out


def eri3_tensor(basis: Basis, aux: Basis, *, pq: PQ) -> Tensor:
    """`(mn|P)` for every AO pair and aux function, `[m][n][P]`, exactly
    symmetric in `(m, n)`. It computes EVERY triple's class, so its cost is
    the highest `la + lb + lP` the two bases reach; a caller bounds that by
    the bases it passes."""
    return _place(basis, aux, lambda a, b, p: eri3_block(a, b, p, pq=pq))


def scale_tensor(basis: Basis, aux: Basis, *, pq: PQ) -> Tensor:
    """The scale of every `(mn|P)`, placed as `eri3_tensor`."""
    return _place(basis, aux,
                  lambda a, b, p: eri3_block_and_scale(a, b, p, pq=pq)[1])


@dataclass(frozen=True)
class Eri3Class:
    """Every cell of one class `(la, lb, lP)`, `la <= lb`, staged.

    A cell `k` is `cells[k] = (a, b, P)`: the AO shell pair as staged
    (`l_a <= l_b`) and one aux shell. `swapped[k]` says the pair is `(j, i)`
    of the basis pair `i < j`, so its block lands with `m, n` TRANSPOSED in
    the basis order's upper triangle (`place_staged`). Its records are
    `[ab_rec0[k], ab_rec1[k])` of the bra table (`zeta_ab`, `P`, `XPA`,
    `XPB`, `s_ab`; every staged pair the class touches once, `bra_pairs`
    with `bra_rec0` / `bra_rec1`) and `[p_rec0[k], p_rec1[k])` of the aux
    table (`A`, `alpha`, `s`; every aux shell once, `aux_shells` with
    `aux_rec0` / `aux_rec1`); its offsets `off_a`, `off_b`, `off_p`."""

    la: int
    lb: int
    lP: int
    cells: tuple[tuple[int, int, int], ...]
    swapped: tuple[bool, ...]
    ab_rec0: tuple[int, ...]
    ab_rec1: tuple[int, ...]
    p_rec0: tuple[int, ...]
    p_rec1: tuple[int, ...]
    off_a: tuple[int, ...]
    off_b: tuple[int, ...]
    off_p: tuple[int, ...]
    bra_pairs: tuple[tuple[int, int], ...]
    bra_rec0: tuple[int, ...]
    bra_rec1: tuple[int, ...]
    zeta_ab: tuple[float, ...]
    P: tuple[Vec, ...]
    XPA: tuple[Vec, ...]
    XPB: tuple[Vec, ...]
    s_ab: tuple[float, ...]
    aux_shells: tuple[int, ...]
    aux_rec0: tuple[int, ...]
    aux_rec1: tuple[int, ...]
    A: tuple[Vec, ...]
    alpha: tuple[float, ...]
    s: tuple[float, ...]

    BRA_ARRAYS = ("zeta_ab", "P", "XPA", "XPB", "s_ab")
    AUX_ARRAYS = ("A", "alpha", "s")
    CELL_ARRAYS = ("ab_rec0", "ab_rec1", "p_rec0", "p_rec1", "off_a",
                   "off_b", "off_p")

    @property
    def n_cell(self) -> int:
        return len(self.cells)

    @property
    def n_bra_rec(self) -> int:
        return len(self.zeta_ab)

    @property
    def n_aux_rec(self) -> int:
        return len(self.alpha)

    def records(self, k: int
                ) -> tuple[tuple[PrimitivePair, ...], tuple[AuxRec, ...]]:
        """Cell `k`'s two record ranges: the bra as `PrimitivePair`s whose
        dressed block is re-formed from the record (module docstring; `i`
        and `j` are not recorded and read 0), the aux as `AuxRec`s."""
        carts = [(a, b) for a in cartesians(self.la)
                 for b in cartesians(self.lb)]
        L = self.la + self.lb
        bras = []
        for r in range(self.ab_rec0[k], self.ab_rec1[k]):
            mA = tuple(-x for x in self.XPA[r])
            mB = tuple(-x for x in self.XPB[r])
            z = self.zeta_ab[r]
            bras.append(PrimitivePair(0, 0, self.P[r], z, L, tuple(
                tuple(self.s_ab[r] * x
                      for x in e_block(a, b, z, _ORIGIN, mA, mB))
                for a, b in carts)))
        p0 = self.p_rec0[k]
        auxs = tuple(AuxRec(r - p0, self.A[r], self.alpha[r], self.s[r])
                     for r in range(p0, self.p_rec1[k]))
        return tuple(bras), auxs

    def arrays(self) -> dict[str, list]:
        """The bra, aux and cell arrays, in that order, as lists."""
        return {k: [list(v) if isinstance(v, tuple) else v
                    for v in getattr(self, k)]
                for k in (*self.BRA_ARRAYS, *self.AUX_ARRAYS,
                          *self.CELL_ARRAYS)}


def _sub(p: Sequence[float], c: Sequence[float]) -> Vec:
    return (p[0] - c[0], p[1] - c[1], p[2] - c[2])


def eri3_classes(basis: Basis, aux: Basis) -> tuple[Eri3Class, ...]:
    """Every AO shell pair `i <= j` of `basis` (staged `l_a <= l_b`) crossed
    with every aux shell of `aux`, one cell each, in its class; classes
    sorted by `(la, lb, lP)`, cells in basis order (pair outer, aux shell
    inner) within one, records in first-touch order."""
    sh, off = basis.shells, basis.offsets
    ash, aoff = aux.shells, aux.offsets
    cols: dict[tuple[int, int, int], dict[str, list]] = {}
    bra_at: dict[tuple[int, int, int], dict[tuple[int, int], tuple[int, int]]]
    bra_at = {}
    aux_at: dict[tuple[int, int, int], dict[int, tuple[int, int]]] = {}
    keys = ("cells", "swapped", *Eri3Class.CELL_ARRAYS, "bra_pairs",
            "bra_rec0", "bra_rec1", *Eri3Class.BRA_ARRAYS, "aux_shells",
            "aux_rec0", "aux_rec1", *Eri3Class.AUX_ARRAYS)

    def bra_recs(c: dict[str, list], w: dict, a: int, b: int
                 ) -> tuple[int, int]:
        if (a, b) not in w:
            r0 = len(c["zeta_ab"])
            sa, sb = sh[a], sh[b]
            for pp in primitive_pairs(sa, sb):
                c["zeta_ab"].append(pp.zeta)
                c["P"].append(pp.P)
                c["XPA"].append(_sub(pp.P, sa.centre))
                c["XPB"].append(_sub(pp.P, sb.centre))
                c["s_ab"].append(dress((1.0,), ca=sa.coefficients[pp.i],
                                       cb=sb.coefficients[pp.j],
                                       zeta=pp.zeta)[0])
            w[(a, b)] = (r0, len(c["zeta_ab"]))
            c["bra_pairs"].append((a, b))
            c["bra_rec0"].append(r0)
            c["bra_rec1"].append(w[(a, b)][1])
        return w[(a, b)]

    def aux_recs(c: dict[str, list], w: dict, n: int) -> tuple[int, int]:
        if n not in w:
            r0 = len(c["alpha"])
            for r in aux_records(ash[n]):
                c["A"].append(r.A)
                c["alpha"].append(r.alpha)
                c["s"].append(r.s)
            w[n] = (r0, len(c["alpha"]))
            c["aux_shells"].append(n)
            c["aux_rec0"].append(r0)
            c["aux_rec1"].append(w[n][1])
        return w[n]

    for i in range(len(sh)):
        for j in range(i, len(sh)):
            swap = sh[i].l > sh[j].l
            a, b = (j, i) if swap else (i, j)
            for n in range(len(ash)):
                cls = (sh[a].l, sh[b].l, ash[n].l)
                c = cols.setdefault(cls, {k: [] for k in keys})
                br = bra_recs(c, bra_at.setdefault(cls, {}), a, b)
                pr = aux_recs(c, aux_at.setdefault(cls, {}), n)
                c["cells"].append((a, b, n))
                c["swapped"].append(swap)
                c["ab_rec0"].append(br[0])
                c["ab_rec1"].append(br[1])
                c["p_rec0"].append(pr[0])
                c["p_rec1"].append(pr[1])
                c["off_a"].append(off[a])
                c["off_b"].append(off[b])
                c["off_p"].append(aoff[n])
    return tuple(Eri3Class(la, lb, lP, **{k: tuple(v) for k, v in c.items()})
                 for (la, lb, lP), c in sorted(cols.items()))


def class_block(cls: Eri3Class, k: int, *, pq: PQ) -> Block:
    """Staged cell `k`'s block from the class's own records, `[a][b][q]`
    over `cartesians(la)`, `cartesians(lb)`, `cartesians(lP)`: what the
    kernel computes for that thread, and `eri3_block` of the staged shells
    bitwise (module docstring)."""
    bras, auxs = cls.records(k)
    return _shape(_block(cls.la, cls.lb, bras, cls.lP, auxs, pq, False)[0],
                  len(cartesians(cls.lb)))


def place_staged(cls: Eri3Class, k: int,
                 block: Sequence[Sequence[Sequence[float]]],
                 out: Tensor) -> None:
    """Scatter staged cell `k`'s block (`[a][b][q]`) into `out`
    (`[m][n][P]`) by the swap: unswapped at `[off_a + a][off_b + b]`,
    swapped with `m, n` TRANSPOSED, `[off_b + b][off_a + a]` -- the basis
    order's upper triangle either way -- then mirrored in `(m, n)`; a
    diagonal pair writes `a >= b` only."""
    pa, pb, _ = cls.cells[k]
    oa, ob, op = cls.off_a[k], cls.off_b[k], cls.off_p[k]
    for a, plane in enumerate(block):
        for b, row in enumerate(plane):
            if pa == pb and b > a:
                continue
            m, n = (ob + b, oa + a) if cls.swapped[k] else (oa + a, ob + b)
            for q, v in enumerate(row):
                out[m][n][op + q] = v
                out[n][m][op + q] = v
