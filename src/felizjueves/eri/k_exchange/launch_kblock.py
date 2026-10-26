"""The launch plan of the one-block-per-`K_AB` exchange kernel, as a value (#481).

    kb_launches(basis, D, *, bound, tau, build=None, decay=None)
                                                         -> KBLaunchPlan
    KBLaunchPlan   iterable of KBClassLaunch, plus tau / dm
    KBClassLaunch(am, bras, kets, blocks, kernel)
    KBlock(a, b, bra0, bra1, ket0, ket1, dmax)
    walk_kb(plan, basis, D, *, pq)                       -> eri.k_exchange.k.KScreened
    e_table(l1, l2, zeta, P, A, B, *, sign=False, dense=False)
                                                         -> one record's E

`launch_k.py`'s counterpart for `kernel_kblock` (opt-UM's mapping): which
K class kernels run, over which blocks, each block a bra RANGE and a ket
RANGE -- and nothing else. THE PLAN IS DATA: no text is emitted, nothing is
compiled or launched, no `[p|q]` evaluated (the bound is injected).

IT IS `k_launches`' PLAN, REGROUPED. The records, the ordered sweep (every
ordered shell pair on both sides, every quadruple at weight 1, the bra-ket
swap unused), the injected bound `G`, the ket sort (descending by `I_bd`
alone, stably) and the class order are `launch_k.k_launches`', called here
with the Schwarz-only key; two things change, both the block kernel's
contract (`kernel_kblock.py`):

- THE BRAS of a class are grouped by FIRST SHELL `a` (ascending), and each
  group sorted descending by `G`, stably from pair order -- the bra loop's
  `break` reads that order. (`k_launches` keeps bras in pair order: one
  warp per bra record, by index, sorts nothing.)
- THE KETS are grouped by first shell `b` (ascending), each group a
  subsequence of `k_launches`' sorted list, so still sorted (the regrouping
  `plan_driver.k_driver` does for `kernel_k`).

A `KBlock(a, b, bra0, bra1, ket0, ket1, dmax)` is one `(a group, b group)`
pair: the kernel's block `blk` contracts `bras[bra0:bra1]` against
`kets[ket0:ket1]` into `K[a][b]`, screened with the density factor
`dmax`. Every `(a, b)` of the class is a block, once; so one launch of the
class's kernel writes each `K[a][b]` from one block. A screen that drops
whole blocks would drop entries of `blocks` here and change nothing else:
a launch filter, not built (`kernel_kblock`'s docstring).

THE DENSITY FACTOR PER BLOCK (#484). With `decay=None` every block's `dmax`
is `max|D|`, the plan's `dm`: #481's screen exactly. With `decay`
(`eri.k_exchange.k.density_decay` at the plan's `tau` and density) each block carries
`decay.factor(a, b, lc, ld)`, [Fock, §4.3.1]'s `F_ab`, never above `dm`.

THE WALK (`walk_kb`): the plan in Python, the kernel's loops. Per class, per
block, per bra in order: when screening, a bra with `(G_bra * G_ket0) *
dmax < tau` (`ket0` the block's first, largest ket; `dmax` the block's)
ends the block's bra loop and every pair of it and of every later bra is
skipped; else down the block's kets with `launch_k.walk_k`'s `break`.
Each surviving pair adds `eri.k_exchange.k._k_quartet` of that one record pair (the
host's staged arithmetic) to `K[a][b]`, every element summed with
`math.fsum`. Both bounds use `walk_k`'s association `(G_bra * G_ket) *
dmax`, and both breaks are sound because the products are monotone in a
sorted factor (the block's `dmax` is one constant), so the walk skips
EXACTLY the ordered record pairs `walk_k` skips at the same `tau` -- the
same terms, fsummed: its `K` is `walk_k`'s to 0, and so `eri.k_exchange.k`'s to
rounding (test/test_launch_kblock.py) -- with every `dmax = dm`; a
per-block factor skips more (#484), within `KScreened.bound`, which the
walk fills (per element, each skipped pair's `(G_bra G_ket) max|D_cd|`).
`n_examined` counts every bound test, the bra loop's included; `scale` is
counted from the plan, as `walk_k`'s.

THE ROLLED KERNEL'S `E` (#571, `kernel_kblock_rolled`). That kernel forms
no `E`: `e_table` is one primitive pair's undressed `E^{l1 l2}` block as it
reads it, `mmd.hermite.e_block` of every row (`cartesians(l1) x
cartesians(l2)`, `l1` outer) with its structurally zero entries (`t_k >
x_k + y_k`) dropped, row-major; `sign=True` (the ket side) negates every
entry of odd `|t|`, the `(-1)^|q|` of `[p|q] = (-1)^|q| [p+q]`.
`dense=True` (#575, `kernel_kblock_rolled.build_kblock_dense_kernel`)
keeps every entry, structural zeros included: each row is the whole
`hermite_order(l1 + l2)`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace

from felizjueves.ao.density import checked_density as _density
from felizjueves.ao.density import epair
from felizjueves.ao.shell import Basis, cartesians
from felizjueves.eri.k_exchange.k import (
    PQ,
    DensityDecay,
    KScreened,
    _check_decay,
    _dcd,
    _dm,
    _k_quartet,
)
from felizjueves.eri.k_exchange.launch_k import KBound, k_class_spec, k_launches
from felizjueves.eri.launch_common import PairRec
from felizjueves.ir import Kernel
from felizjueves.mmd.hermite import e_block, hermite_order
from felizjueves.spec import Spec

__all__ = ["KBClassLaunch", "KBLaunchPlan", "KBlock", "e_table",
           "kb_launches", "walk_kb"]

KBBuild = Callable[[Spec], Kernel]
Am = tuple[int, int, int, int]


@dataclass(frozen=True)
class KBlock:
    """One `K[a][b]` block: bras `[bra0, bra1)` (first shell `a`) against
    kets `[ket0, ket1)` (first shell `b`) of its class's lists, screened
    with the density factor `dmax` (`kb_launches` always sets it)."""

    a: int
    b: int
    bra0: int
    bra1: int
    ket0: int
    ket1: int
    dmax: float | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.bra0 < self.bra1 and 0 <= self.ket0 < self.ket1):
            raise ValueError(f"block {self}: an empty or negative range")


@dataclass(frozen=True)
class KBClassLaunch:
    """One K class `am = (la, lc, lb, ld)`'s block-kernel launch: its bras
    grouped by `a`, its kets by `b`, each group sorted descending by `G`,
    and one `KBlock` per `(a, b)`, `a` outer. `kernel` is None when no
    builder was injected."""

    am: Am
    bras: tuple[PairRec, ...]
    kets: tuple[PairRec, ...]
    blocks: tuple[KBlock, ...]
    kernel: Kernel | None

    def __post_init__(self) -> None:
        for blk in self.blocks:
            bs = self.bras[blk.bra0:blk.bra1]
            ks = self.kets[blk.ket0:blk.ket1]
            if {r.i for r in bs} != {blk.a} or {r.i for r in ks} != {blk.b}:
                raise ValueError(f"K class {self.am}: block {blk} mixes "
                                 "first shells")
            for side in (bs, ks):
                if any(x.G < y.G for x, y in zip(side, side[1:],
                                                 strict=False)):
                    raise ValueError(f"K class {self.am}: block {blk} is "
                                     "not sorted descending by G")
            if blk.dmax is None or not blk.dmax >= 0.0:
                raise ValueError(f"K class {self.am}: block {blk} carries "
                                 "no density factor >= 0")

    @property
    def L(self) -> int:
        return sum(self.am)

    @property
    def spec(self) -> Spec:
        return k_class_spec(self.am)


@dataclass(frozen=True)
class KBLaunchPlan:
    """`kb_launches()`'s value: the class launches in `am` order, `tau` and
    `dm = max|D|`, the screen's density factor."""

    classes: tuple[KBClassLaunch, ...]
    tau: float
    dm: float

    def __iter__(self) -> Iterator[KBClassLaunch]:
        return iter(self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def n_quartets(self) -> int:
        """(bra record, ket record) pairs the plan covers, unscreened."""
        return sum((b.bra1 - b.bra0) * (b.ket1 - b.ket0)
                   for c in self.classes for b in c.blocks)


def _groups(recs: Sequence[PairRec], *, sort: bool
            ) -> tuple[tuple[PairRec, ...], dict[int, tuple[int, int]]]:
    """`recs` grouped by first shell (ascending), each group in its given
    order or sorted descending by `G`, stably; and each group's range."""
    out: list[PairRec] = []
    rng: dict[int, tuple[int, int]] = {}
    for i in sorted({r.i for r in recs}):
        grp = [r for r in recs if r.i == i]
        if sort:
            grp.sort(key=lambda r: -r.G)  # stable: ties keep pair order
        rng[i] = (len(out), len(out) + len(grp))
        out.extend(grp)
    return tuple(out), rng


def _schwarz(b: int, d: int, i_bd: float) -> float:
    return i_bd


def kb_launches(basis: Basis, D: object, *, bound: KBound, tau: float,
                build: KBBuild | None = None,
                decay: DensityDecay | None = None) -> KBLaunchPlan:
    """The block kernel's launch plan of `K` for `basis` and the symmetric
    density `D` (module docstring): `bound(epair) -> I per record` injected,
    `build(spec) -> Kernel` the block-kernel builder, or none; `decay` the
    per-block density factor, or `max|D|` for every block."""
    kp = k_launches(basis, D, bound=bound, key=_schwarz, tau=tau)
    if decay is not None:
        _check_decay(decay, kp.tau, kp.dm, len(basis.shells))
    built: dict[Spec, Kernel] = {}
    out = []
    for c in kp:
        bras, brng = _groups(c.bras, sort=True)
        kets, krng = _groups(c.kets, sort=False)
        lc, ld = c.am[1], c.am[3]
        blocks = tuple(KBlock(a, b, *brng[a], *krng[b],
                              kp.dm if decay is None
                              else decay.factor(a, b, lc, ld))
                       for a in brng for b in krng)
        kernel = None
        if build is not None:
            if c.spec not in built:
                built[c.spec] = build(c.spec)
            kernel = built[c.spec]
        out.append(KBClassLaunch(c.am, bras, kets, blocks, kernel))
    return KBLaunchPlan(tuple(out), kp.tau, kp.dm)


def walk_kb(plan: KBLaunchPlan, basis: Basis, D: object, *,
            pq: PQ) -> KScreened:
    """`K` and its screening record by walking `plan` as the block kernel
    does (module docstring, "THE WALK")."""
    rows = _density(basis, D)
    if _dm(rows) != plan.dm:
        raise ValueError(f"max|D| = {_dm(rows)!r} is not the plan's dm "
                         f"{plan.dm!r}: planned from another density")
    tau = plan.tau
    screen = tau > 0.0
    shells, off = basis.shells, basis.offsets
    sides = sorted({(r.i, r.j) for c in plan for r in (*c.bras, *c.kets)})
    eps = {xy: epair(shells[xy[0]], shells[xy[1]]) for xy in sides}
    one = {(x, y, k): replace(ep, recs=(rec,))
           for (x, y), ep in eps.items() for k, rec in enumerate(ep.recs)}

    n = basis.n_ao
    terms: list[list[list[float]]] = [[[] for _ in range(n)] for _ in range(n)]
    cover = [[0] * len(shells) for _ in shells]
    dcd_max: dict[tuple[int, int], float] = {}

    def dmax_cd(c: int, d: int) -> float:
        if (c, d) not in dcd_max:
            dcd_max[c, d] = max(abs(v) for v in _dcd(basis, rows, c, d))
        return dcd_max[c, d]

    lost = [[[] for _ in shells] for _ in shells]
    n_pairs = n_skipped = n_examined = 0
    dropped: list[float] = []
    for c in plan:
        for blk in c.blocks:
            bras = c.bras[blk.bra0:blk.bra1]
            kets = c.kets[blk.ket0:blk.ket1]
            n_pairs += len(bras) * len(kets)
            cover[blk.a][blk.b] += len(bras) * len(kets)
            g0, fb = kets[0].G, blk.dmax
            lb = lost[blk.a][blk.b]
            for ib, br in enumerate(bras):
                if screen:
                    n_examined += 1
                    if (br.G * g0) * fb < tau:  # every later bra too
                        rest = bras[ib:]
                        n_skipped += len(rest) * len(kets)
                        dropped.extend((r.G * k.G) * fb
                                       for r in rest for k in kets)
                        lb.extend((r.G * k.G) * dmax_cd(r.j, k.j)
                                  for r in rest for k in kets)
                        break
                bra = one[br.i, br.j, br.prim]
                for ik, kr in enumerate(kets):
                    if screen:
                        n_examined += 1
                        if (br.G * kr.G) * fb < tau:  # every later ket too
                            rest = kets[ik:]
                            n_skipped += len(rest)
                            dropped.extend((br.G * r.G) * fb for r in rest)
                            lb.extend((br.G * r.G) * dmax_cd(br.j, r.j)
                                      for r in rest)
                            break
                    X = _k_quartet(bra, one[kr.i, kr.j, kr.prim],
                                   _dcd(basis, rows, br.j, kr.j), pq=pq,
                                   want_scale=False)[0]
                    oa, ob = off[blk.a], off[blk.b]
                    for ii, row in enumerate(X):
                        for jj, v in enumerate(row):
                            terms[oa + ii][ob + jj].append(v)
    K = tuple(tuple(math.fsum(t) for t in r) for r in terms)
    sh_of = [x for x in range(len(shells)) for _ in range(shells[x].n_cart)]
    S = tuple(tuple(float(cover[sh_of[m]][sh_of[v]]) for v in range(n))
              for m in range(n))
    lsum = [[math.fsum(x) for x in r] for r in lost]
    B = tuple(tuple(lsum[sh_of[m]][sh_of[v]] for v in range(n))
              for m in range(n))
    return KScreened(K, S, n_pairs, n_skipped, n_examined,
                     max(dropped, default=0.0), math.fsum(dropped), B)


def e_table(l1: int, l2: int, zeta: float, P: Sequence[float],
            A: Sequence[float], B: Sequence[float], *,
            sign: bool = False, dense: bool = False) -> tuple[float, ...]:
    """One record's `E^{l1 l2}` as `kernel_kblock_rolled` reads it: the
    nonzero entries of `mmd.hermite.e_block`, row by row (module docstring,
    "THE ROLLED KERNEL'S `E`"); `sign` negates the odd-`|t|` entries;
    `dense` keeps the structural zeros, every row whole (#575's kernel)."""
    ts = hermite_order(l1 + l2)
    out: list[float] = []
    for x in cartesians(l1):
        for y in cartesians(l2):
            row = e_block(x, y, zeta, tuple(P), tuple(A), tuple(B))
            out += [-v if sign and sum(t) % 2 else v
                    for t, v in zip(ts, row, strict=True)
                    if dense or all(t[k] <= x[k] + y[k] for k in range(3))]
    return tuple(out)
