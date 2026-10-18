"""The Schwarz matrix `sqrt((mn|mn))`, host-side: the reference and the records.

    diagonal_and_scale(sa, sb, *, pq) -> ((mn|mn), scale) per row `mn`
    diagonal_matrix(basis, *, pq)     -> (mn|mn) over every AO pair
    schwarz_matrix(basis, *, pq)      -> sqrt(max((mn|mn), 0)) over every AO pair
    schwarz_classes(basis)            -> the `kernel_schwarz` records, per class

THE INEQUALITY. The Coulomb kernel `1/r12` is positive definite, so `(rho|sigma)
= <rho, sigma>` is an inner product on charge distributions, and with `rho =
chi_m chi_n`, `sigma = chi_r chi_s` (real Cartesian Gaussians) Cauchy-Schwarz
reads

    |(mn|rs)| <= sqrt((mn|mn)) sqrt((rs|rs)),

`(mn|mn) = ||chi_m chi_n||^2 >= 0`. The matrix `Q[m][n] = sqrt((mn|mn))` is the
Schwarz factor of every AO pair, standalone output: nothing in the tree screens
on it yet. It is EXACT where `ao.pairs`' `I_p` and `G_[ab]` are upper bounds
built from the `PQ = 0` self-integral and the triangle inequality.

THE REFERENCE IS `ao.eri`, NOT A SECOND ERI PATH. `(mn|mn)` is the component
diagonal of the block `eri_block((sa, sb), (sa, sb))`, so every value here is
`ao.eri`'s arithmetic, already witnessed against libcint (test/test_ao_eri.py),
and `scale` is `eri_block_and_scale`'s absolute-value twin, the size of what was
added (a tolerance relative to it is a statement about rounding).

THE CLAMP. `(mn|mn) >= 0` exactly, but a computed one for a diffuse or distant
pair can round to a tiny negative; `sqrt` of it is a NaN. So `schwarz_matrix`
takes `sqrt(max(x, 0.0))`, as the kernel does -- a clamp of a computed value,
not the `abs` that `ao.pairs.hermite_bound` refuses (that one guards a sign
convention, and is exact). A rounded `sqrt` is exact only to rounding, so `Q`
bounds `|(mn|rs)|` to rounding: a screen that must never drop a kept integral
pads `tau`, which is the consumer's business.

THE PLACEMENT is `kernel_jab`'s scatter rule, so host and kernel fill the same
elements from the same rows: shell pairs `i <= j` in basis order, row `(a, b)`
to `Q[off_i + a][off_j + b]` and its mirror; a DIAGONAL shell pair writes its
lower triangle `a >= b` only, mirrored, so `(mn|mn)` and `(nm|nm)` -- equal
exactly, not always bitwise -- are one value.

THE RECORDS (`schwarz_classes`). One `SchwarzClass` per pair class `(la, lb)`
with `la <= lb`: a shell pair whose first shell has the higher `l` is staged
SWAPPED (`(j, i)`), which changes no element of a symmetric matrix and keeps the
ladder at `la <= lb` (the emit CLI's `--schwarz-l`). Each shell pair contributes
its primitive pairs (`ao.j.epair`: `P`, `zeta`, `s = c_a c_b (pi/zeta)^{3/2}`,
and `XPA = P - A`, `XPB = P - B`) as one contiguous record range
`[rec0, rec1)`, which is both the bra and the ket side of its quartet.

`pq` is INJECTED (`ao.eri`'s signature); `ao` imports no `reference/`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from felizjueves.ao.eri import PQ, eri_block_and_scale
from felizjueves.ao.j import epair
from felizjueves.ao.shell import Basis, Shell

Matrix = list[list[float]]


def diagonal_and_scale(sa: Shell, sb: Shell, *, pq: PQ
                       ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`(mn|mn)` and its scale for every row `mn` of the ordered pair
    `(sa, sb)`, rows in `cartesians(la) x cartesians(lb)` order, `a` outer."""
    blk, scale = eri_block_and_scale((sa, sb), (sa, sb), pq=pq)
    return (tuple(blk[k][k] for k in range(len(blk))),
            tuple(scale[k][k] for k in range(len(scale))))


def _place(basis: Basis, value) -> Matrix:
    """`value(sa, sb)`'s rows placed by the scatter rule (module docstring)."""
    sh, off = basis.shells, basis.offsets
    n = basis.n_ao
    out = [[0.0] * n for _ in range(n)]
    for i in range(len(sh)):
        for j in range(i, len(sh)):
            vals = value(sh[i], sh[j])
            nb = sh[j].n_cart
            for a in range(sh[i].n_cart):
                for b in range(nb):
                    if i == j and b > a:
                        continue  # diagonal: the lower triangle, mirrored
                    v = vals[a * nb + b]
                    out[off[i] + a][off[j] + b] = v
                    out[off[j] + b][off[i] + a] = v
    return out


def diagonal_matrix(basis: Basis, *, pq: PQ) -> Matrix:
    """`(mn|mn)` for every AO pair, `n_ao x n_ao`, symmetric."""
    return _place(basis, lambda sa, sb: diagonal_and_scale(sa, sb, pq=pq)[0])


def scale_matrix(basis: Basis, *, pq: PQ) -> Matrix:
    """The scale of every `(mn|mn)` (`ao.eri`'s), placed as `diagonal_matrix`."""
    return _place(basis, lambda sa, sb: diagonal_and_scale(sa, sb, pq=pq)[1])


def schwarz_matrix(basis: Basis, *, pq: PQ) -> Matrix:
    """`Q[m][n] = sqrt(max((mn|mn), 0))`, `n_ao x n_ao`, symmetric."""
    return [[math.sqrt(max(x, 0.0)) for x in row]
            for row in diagonal_matrix(basis, pq=pq)]


@dataclass(frozen=True)
class SchwarzClass:
    """Every shell pair of one pair class `(la, lb)`, `la <= lb`, and its
    primitive-pair records; `arrays()` is them in `kernel_schwarz`'s
    parameter order.

    `pairs[k]` is the shell pair `(A, B)` as staged (`l_A <= l_B`, so possibly
    swapped from basis order); its records are `[rec0[k], rec1[k])`, its AO
    offsets `off_a[k]`, `off_b[k]`."""

    la: int
    lb: int
    pairs: tuple[tuple[int, int], ...]
    zeta_ab: tuple[float, ...]
    P: tuple[tuple[float, float, float], ...]
    XPA: tuple[tuple[float, float, float], ...]
    XPB: tuple[tuple[float, float, float], ...]
    s_ab: tuple[float, ...]
    rec0: tuple[int, ...]
    rec1: tuple[int, ...]
    off_a: tuple[int, ...]
    off_b: tuple[int, ...]

    @property
    def n_pair(self) -> int:
        return len(self.pairs)

    @property
    def n_rec(self) -> int:
        return len(self.zeta_ab)

    # The kernel's record arrays (#422, `prim_pair.PrimPairSoA`): one flat
    # `n_rec` array per scalar and one per axis of each vector, `XP YP ZP`
    # for `P`, `XPA YPA ZPA` for `XPA`, `XPB YPB ZPB` for `XPB`.
    RECORD_ARRAYS = ("zeta_ab", "s_ab",
                     *(f"{a}{v}" for v in ("P", "PA", "PB") for a in "XYZ"))
    PAIR_ARRAYS = ("rec0", "rec1", "off_a", "off_b")

    def arrays(self) -> dict[str, list]:
        """The kernel's array parameters, in its parameter order, as flat
        lists: each record vector split per axis."""
        out: dict[str, list] = {"zeta_ab": list(self.zeta_ab),
                                "s_ab": list(self.s_ab)}
        for vec, tail in (("P", "P"), ("XPA", "PA"), ("XPB", "PB")):
            for d, a in enumerate("XYZ"):
                out[f"{a}{tail}"] = [v[d] for v in getattr(self, vec)]
        return out | {k: list(getattr(self, k)) for k in self.PAIR_ARRAYS}


def _sub(p: Sequence[float], c: Sequence[float]) -> tuple[float, float, float]:
    return (p[0] - c[0], p[1] - c[1], p[2] - c[2])


def schwarz_classes(basis: Basis) -> tuple[SchwarzClass, ...]:
    """Every shell pair `i <= j` of `basis`, staged (`l_A <= l_B`) into its
    class, classes sorted by `(la, lb)`, pairs in basis order within one."""
    sh, off = basis.shells, basis.offsets
    cols: dict[tuple[int, int], dict[str, list]] = {}
    for i in range(len(sh)):
        for j in range(i, len(sh)):
            a, b = (i, j) if sh[i].l <= sh[j].l else (j, i)
            sa, sb = sh[a], sh[b]
            c = cols.setdefault((sa.l, sb.l), {k: [] for k in (
                "pairs", "zeta_ab", "P", "XPA", "XPB", "s_ab",
                *SchwarzClass.PAIR_ARRAYS)})
            c["pairs"].append((a, b))
            c["rec0"].append(len(c["zeta_ab"]))
            for P, zeta, s in epair(sa, sb).recs:
                c["zeta_ab"].append(zeta)
                c["P"].append(tuple(P))
                c["XPA"].append(_sub(P, sa.centre))
                c["XPB"].append(_sub(P, sb.centre))
                c["s_ab"].append(s)
            c["rec1"].append(len(c["zeta_ab"]))
            c["off_a"].append(off[a])
            c["off_b"].append(off[b])
    return tuple(SchwarzClass(la, lb, **{k: tuple(v) for k, v in c.items()})
                 for (la, lb), c in sorted(cols.items()))
