"""The whole-molecule launch plan, as a value (#138; milestone 6).

    launches(basis, D, *, bound, n_batches, tau, density_sort=True,
             strict=True, resolve=None)           -> LaunchPlan
    LaunchPlan    iterable of ClassLaunch, plus tau / n_batches / refused
    ClassLaunch(lp, lq, n_batches, bras, kets, kernels, bra_blocks)
    PairRec(i, j, prim, L, G, key)                one index record

The counterpart of the predecessor's `jgen/launch.py` `launches()`: which
`[p|q]` kernels run, over which bra and ket index records, in which order --
and nothing else. THE PLAN IS DATA: no text is emitted, nothing is compiled,
nothing is launched, no `[p|q]` is evaluated. A walker (#142's oracle) or a
staged driver (#143) consumes it.

NOT `Launch`. `interp.Launch` is the thread geometry (grid, block, lanes);
this module's unit is `ClassLaunch` (milestone description (a)).

RECORDS, NOT SHELL PAIRS (milestone (b)). The kernel maps one warp per
primitive-pair index record, and `ao/pairs.py`'s bound is proved per record,
so the lists here are record lists. A `PairRec` names its shell pair
`(i, j)` and which primitive pair of it it is (`prim`, the index into
`ao.pairs.primitive_pairs(shells[i], shells[j])`, i outer) -- that is the
record -> shell pair map, and `ClassLaunch.bra_blocks` is its inverse on the
bra side: `(a, b, start, stop)`, the rows of the bra list (hence of `J_p`)
that eq. 20 transforms into the block `J[a][b]`.

WHICH SHELL PAIRS: the UNIQUE sweep of `ao.j.coulomb_matrix` /
`coulomb_matrix_screened`, `a >= b` by shell index, on both sides, with the
ket off-diagonal weight `w = 2` folded into `D_q` (and so into `Dmax`), and
each bra block mirrored by the scatter. That is the sweep #142's oracle must
equal to `0`, so it is the only one whose screen decisions -- which depend
on `key` -- can match; an all-ordered-pairs plan at `w = 1` halves every
off-diagonal `Dmax` and screens differently. The ket pre-contraction is
`ao.j`'s own (`_kets`), not a restatement, so `Dmax` here is bit-identical to
the host sweep's.

THE KEYS. A record's `G` is `I_pair`, the pair-factor-free Schwarz bound of
`ao/pairs.py`, from the INJECTED `bound(zeta, L) -> float` (tests pass
`ao.pairs.pair_bound` with `reference.pq.one_centre`, or `ao.j.pair_bound_pq`
with `reference.pq.pq`; this module reaches `reference/` nowhere). It is the
real bound on the BRA side as well -- not `1.0`, which is the `jp=` seam's
deliberate unscreened placeholder -- because the kernel screens on
`Ib * Iq * Dmax`. A ket's `key` is `G * Dmax`, `Dmax = max_q |D_q|` over its
weighted, dressed ket vector: exactly `ao.j._Ket.bound`, the factor the
screen multiplies `I_bra` by. A bra's `key` is its `G`; the bra list stays in
pair order, because the kernel maps one warp per bra BY INDEX.

THE SORT. Each class's kets are sorted DESCENDING by `key`, STABLY --
`sorted(..., key=..., reverse=True)`, `ao.j._sorted_kets`' choice, matched and
not re-decided (`key=lambda r: -r.key` is not the same on ties). That is the
precondition of Alg. 1's `break`: a fixed non-negative `G_bra` times a
non-increasing `key` is non-increasing, so the first ket under `tau` ends the
row. The partition of one global ket list into per-class lists cannot change
which kets clear the threshold for a given bra.

`density_sort=False`: every ket's `Dmax` is replaced by ONE global value, the
largest per-record `Dmax` over all kets, so the order is by `G` alone -- the
measurable alternative `jgen` keeps. It is still a sound screen (the global
value bounds every record's), only looser. It is NOT the AO density's largest
element: that does not bound the dressed `D_q` (`ao/pairs.py`). It changes
the ket ORDER and the keys, never the ket SET.

THE CLASSES. One `ClassLaunch` per `(lp, lq)` with records on BOTH sides,
sorted by `(lp, lq)`; a class with records on one side only is absent, never
present and empty (`ClassLaunch` refuses an empty side). Its kernels are
`kernel.build_pq_kernels` of the class at its batch count -- the kernel is a
function of `(lp, lq)` and `n_batches` only, so ONE build serves every shell
class `(la, lb | lc, ld)` that sums to it, and one launch carries every batch
kernel. Kernels are memoised per Spec (building is pure); records are not
(#142's hazard: `ao.j` memoises nothing, and the tau ladder must measure the
screen, not a cache).

THE BATCH COUNT is asked of the plan per class, `ao.drive.KernelJP`'s
`strict` contract: a count `plan.pipeline.plan` refuses (#64) raises with the
class and the count (`strict=True`, the default), or the class is built at
ONE batch and `(lp, lq, n_batches)` recorded in `LaunchPlan.refused`
(`strict=False`). `_batches_for` is the one place a count is chosen: with
`resolve=` (#145) the count of class `(lp, lq)` is `resolve(lp, lq)` instead
of `n_batches` -- a policy injected, never imported (`dispatch.Resolver` is
one), and asked of the plan exactly as a fixed count is. Without it the plan
is today's; `LaunchPlan.n_batches` records the argument either way, and each
`ClassLaunch.n_batches` the count it was built at.

`tau` is carried, validated, for the consumer: it is the kernel's runtime
argument, and the plan itself skips nothing.

THE K PLAN (#204) sits BESIDE this one, in `launch_k.py`: it shares
`PairRec`, the stable descending sort (`_sorted_desc`), the bound's value
check (`_nonneg`) and the batch-count admission (`_batches_for`), and none
of `ClassLaunch`, `LaunchPlan` or `launches`, which are J-shaped -- why is
`launch_k`'s module docstring. Nothing here changed behaviour for it: the J
plans are pinned by a fingerprint taken before #204 (test/test_launch.py).

THE ONE-ELECTRON LAUNCH (#241) sits beside both, at the end of this module:

    one_electron_launch(basis, nuclei, *, tau, bound) -> OneELaunch
    OneEClass(la, lb, recs, blocks)                     one pair class
    OneERec(i, j, prim, zeta, P, XPA, XPB, s, eb, off_a, off_b)
    walk_one_electron(plan, basis, target, *, one_centre=None) -> S, T or V

ONE record set for `S`, `T` and `V`, and NOT a `ClassLaunch`: that type is
J-shaped (a bra side, a density-keyed sorted ket side, a batch count), and
`OneEClass`'s docstring says why a sibling type was the right call. None of
`PairRec`, `ClassLaunch`, `LaunchPlan` or `launches` changed for it.

THE CROSS-BASIS OVERLAP LAUNCH (`Target.SX`) is the last sibling:

    cross_overlap_launch(basis_a, basis_b, *, tau, bound) -> XLaunch
    XClass(la, lb, recs, blocks)                one ORDERED pair class
    XRec(i, j, prim, zeta, XPA, XPB, s, off_a, off_b)
    walk_cross_overlap(plan, basis_a, basis_b)  -> S_{B1,B2}

Not a `OneELaunch`: that record set is ONE basis's `i <= j` pairs, its
diagonal read from `off_a == off_b`, and it carries the nuclei and the ket
exponent for `T` and `V`. A cross build has two bases, every ordered pair
`(i in A, j in B)`, two offsets into different index sets (so no
`diagonal`), one operator and no nuclei.

THE ORACLE (#142) -- the plan WALKED, in Python:

    walk(plan, basis, D, *, pq)                       -> ao.j.Screened
    coulomb_launched(basis, D, *, pq, tau, n_batches=1, bound=None,
                     density_sort=True, strict=True)  -> ao.j.Screened

`jgen`'s `coulomb_launched`. Per `ClassLaunch`, per bra record (pair order),
down that class's sorted ket list, Alg. 1's `break` at the first ket with
`G_bra * key_ket < tau`; each surviving (bra, ket) pair contributes
`sum_q [p|q] D_q` per `p` (eq. 21, `pq` injected), every class's
contributions to one bra record are summed with `math.fsum`, and eq. 20
transforms the record's `J_p` into its block `J[a][b]`, mirrored (eq. 22's
scatter). The result is `ao.j.Screened`, the ONE screening record both paths
fill (`ao/j.py`, "THE SORTED `break`"). It is the independent witness the
staged device driver (#143) must reproduce, and it is pure Python, so it
runs in the push gate.

WHAT EQUALS WHAT. With `bound` defaulted to `ao.j.pair_bound_pq` through the
same `pq`, every key is bit-identical to the host sweep's, so the oracle
skips EXACTLY the pairs `ao.j.coulomb_matrix_screened` skips -- the
partition of the one global ket list into per-class lists cannot change
which kets clear the threshold for a given bra -- and, each per-ket term
being computed by the host's own arithmetic and each `J_p` accumulated with
`fsum` (order-independent), `J` equals the host's TO 0. `n_pairs`,
`n_skipped`, `max_skipped` and `dropped` equal the host's too.
`n_examined` does NOT, by design: the walk breaks once per (bra record,
class) row, the host once per bra record over one global list, so the
oracle examines `n_pairs - n_skipped` plus one per row that broke.

The walk rebuilds the per-pair data the plan does not carry (`P`, `zeta`,
the dressed `E` block, `Q`, `eta`, `D_q`) from `basis` and `D` -- once per
call and dropped, through `ao.j._kets` and `ao.pairs.primitive_pairs`, the
host's own constructors. NOTHING IS MEMOISED: the tau ladder must measure
the screen, not a cache. `plan` must have been planned from the same
`basis` and `D`; a record that does not name a primitive pair of `basis`
raises.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache

from felizjueves.ao.hermite import Ang, e_axis, hermite_order
from felizjueves.ao.j import (
    PQ,
    Screened,
    _density,
    _kets,
    _tau,
    epair,
    pair_bound_pq,
)
from felizjueves.ao.kinetic import kinetic_axis
from felizjueves.ao.one_electron import matrix, rectangular
from felizjueves.ao.pairs import (
    OneCentre,
    cross_pair_list,
    pair_list,
    primitive_pairs,
)
from felizjueves.ao.potential import _check_nuclei
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.ir import Kernel
from felizjueves.kernel import build_pq_kernels
from felizjueves.plan.pipeline import plan
from felizjueves.spec import Operator, Spec, Target

__all__ = ["Bound", "ClassLaunch", "LaunchPlan", "OneEClass", "OneELaunch",
           "OneERec", "PairRec", "XClass", "XLaunch", "XRec", "class_spec",
           "coulomb_launched", "cross_overlap_launch", "launches",
           "one_electron_launch", "per_axis", "walk", "walk_cross_overlap",
           "walk_one_electron"]

Bound = Callable[[float, int], float]  # (zeta, L) -> I_pair >= 0


@dataclass(frozen=True)
class PairRec:
    """One bra or ket index record (module docstring, "THE KEYS").

    `i, j` the shell pair (`i >= j` in a J plan; any ORDERED pair in a K
    plan, `launch_k`, where `G` is K's record bound and `key` the injected
    ket key), `prim` its primitive pair's index in
    `ao.pairs.primitive_pairs(shells[i], shells[j])`, `L` the pair momentum,
    `G` the injected bound `I_pair`, `key` = `G * Dmax` for a ket, `G` for a
    bra.
    """

    i: int
    j: int
    prim: int
    L: int
    G: float
    key: float

    @property
    def shell_pair(self) -> tuple[int, int]:
        return (self.i, self.j)


@dataclass(frozen=True)
class ClassLaunch:
    """Every batch kernel of one `(lp, lq)` class, over its bra records in
    pair order and its ket records sorted descending by `key`. `bra_blocks`
    is `(a, b, start, stop)` per bra shell pair: `bras[start:stop]`."""

    lp: int
    lq: int
    n_batches: int
    bras: tuple[PairRec, ...]
    kets: tuple[PairRec, ...]
    kernels: tuple[Kernel, ...]
    bra_blocks: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not self.bras or not self.kets:
            raise ValueError(
                f"class (lp, lq) = ({self.lp}, {self.lq}) has "
                f"{len(self.bras)} bra and {len(self.kets)} ket records: a "
                "class with records on one side only is absent, not empty")
        if any(r.L != self.lp for r in self.bras) or any(
                r.L != self.lq for r in self.kets):
            raise ValueError(f"a record outside class ({self.lp}, {self.lq})")
        if not self.kernels:
            raise ValueError(f"class ({self.lp}, {self.lq}) has no kernels")

    @property
    def L(self) -> int:
        return self.lp + self.lq


@dataclass(frozen=True)
class LaunchPlan:
    """`launches()`'s value: the class launches, in `(lp, lq)` order, and
    what they were planned with. Iterating it iterates the launches."""

    classes: tuple[ClassLaunch, ...]
    tau: float
    n_batches: int
    density_sort: bool
    refused: tuple[tuple[int, int, int], ...]  # (lp, lq, n_batches), strict=False

    def __iter__(self) -> Iterator[ClassLaunch]:
        return iter(self.classes)

    def __len__(self) -> int:
        return len(self.classes)


def class_spec(lp: int, lq: int, n_batches: int = 1) -> Spec:
    """The generation key of class `(lp, lq)`: `ao.drive.class_spec` of the
    canonical shell class `(lp, 0 | lq, 0)` -- the kernel reads only the sums
    (`kernel.split`), so every shell class with these sums builds the same
    kernels (test/test_launch.py pins it)."""
    return Spec(Operator.COULOMB, 0, (lp, 0, lq, 0), Target.J, (), n_batches)


@cache
def _admits(lp: int, lq: int, n_batches: int) -> bool:
    try:
        plan(class_spec(lp, lq, n_batches))
    except ValueError:
        return False
    return True


@cache
def _kernels(spec: Spec) -> tuple[Kernel, ...]:
    return build_pq_kernels(spec)


def _batches_for(lp: int, lq: int, n_batches: int, *, strict: bool,
                 refused: list[tuple[int, int, int]],
                 resolve: Callable[[int, int], int] | None = None) -> int:
    if resolve is not None:
        n_batches = resolve(lp, lq)
    if _admits(lp, lq, n_batches):
        return n_batches
    if strict:
        raise ValueError(f"the plan refuses n_batches = {n_batches} for class "
                         f"(lp, lq) = ({lp}, {lq})")
    refused.append((lp, lq, n_batches))
    return 1


def _nonneg(g: object, what: str) -> float:
    """A bound's value, refused unless a finite float >= 0 -- shared with the
    K plan (`launch_k`, #204), whose bound has another signature."""
    if not (isinstance(g, int | float) and math.isfinite(g) and g >= 0.0):
        raise ValueError(f"{what} = {g!r} is not a finite float >= 0")
    return float(g)


def _g(bound: Bound, zeta: float, L: int) -> float:
    return _nonneg(bound(zeta, L), f"bound({zeta!r}, {L})")


def _no_pq(p, q, PQv, theta) -> float:
    raise AssertionError("the plan evaluates no [p|q]")


def launches(basis: Basis, D: object, *, bound: Bound, n_batches: int,
             tau: float, density_sort: bool = True,
             strict: bool = True,
             resolve: Callable[[int, int], int] | None = None) -> LaunchPlan:
    """The launch plan of `J` for `basis` and the symmetric density `D`
    (module docstring). `bound(zeta, L)` is `I_pair`, injected."""
    rows = _density(basis, D)
    tau = _tau(tau)
    shells = basis.shells
    pairs = [(a, b) for a in range(len(shells)) for b in range(a + 1)]

    bras: dict[int, list[PairRec]] = {}
    blocks: dict[int, list[tuple[int, int, int, int]]] = {}
    kets: dict[int, list[PairRec]] = {}
    ket_dmax: list[float] = []  # parallel to the kets, in pair order
    ket_all: list[PairRec] = []
    for a, b in pairs:
        L = shells[a].l + shells[b].l
        gs = [_g(bound, r.zeta, L)
              for r in primitive_pairs(shells[a], shells[b])]
        side = bras.setdefault(L, [])
        start = len(side)
        side.extend(PairRec(a, b, k, L, g, g) for k, g in enumerate(gs))
        blocks.setdefault(L, []).append((a, b, start, len(side)))
        # ao.j's pre-contraction, one record per primitive pair in the same
        # (i outer) order, with the unique sweep's weight folded in.
        w = 1.0 if a == b else 2.0
        kts = _kets(basis, rows, [(a, b, w)], pq=_no_pq, tau=0.0)
        for k, (g, kt) in enumerate(zip(gs, kts, strict=True)):
            dmax = max(abs(v) for v in kt.dq)
            ket_dmax.append(dmax)
            ket_all.append(PairRec(a, b, k, L, g, g * dmax))
    if not density_sort:
        dglobal = max(ket_dmax, default=0.0)
        ket_all = [PairRec(r.i, r.j, r.prim, r.L, r.G, r.G * dglobal)
                   for r in ket_all]
    for r in ket_all:
        kets.setdefault(r.L, []).append(r)

    refused: list[tuple[int, int, int]] = []
    out = []
    for lp in sorted(bras):
        for lq in sorted(kets):
            n = _batches_for(lp, lq, n_batches, strict=strict,
                             refused=refused, resolve=resolve)
            out.append(ClassLaunch(
                lp, lq, n, tuple(bras[lp]),
                tuple(_sorted_desc(kets[lq])),
                _kernels(class_spec(lp, lq, n)), tuple(blocks[lp])))
    return LaunchPlan(tuple(out), tau, n_batches, density_sort,
                      tuple(refused))


def _sorted_desc(recs: Sequence[PairRec]) -> list[PairRec]:
    """Descending by `key`, STABLE -- `ao.j._sorted_kets`' form exactly."""
    return sorted(recs, key=lambda r: r.key, reverse=True)


# --- the oracle (#142): the plan walked in Python ---------------------------


def walk(plan: LaunchPlan, basis: Basis, D: object, *, pq: PQ) -> Screened:
    """`J` and its screening record by walking `plan` (module docstring, "THE
    ORACLE"): per class, per bra record, down the sorted kets to the first
    `G_bra * key < plan.tau` -- a `break`, never a `continue`."""
    rows = _density(basis, D)
    tau = plan.tau
    shells, off = basis.shells, basis.offsets
    sides = {r.shell_pair for c in plan for r in (*c.bras, *c.kets)}
    # Per-call record data, dropped on return (never memoised). `_kets` at
    # tau = 0 evaluates no bound; its `qs` is the pair's hermite_order(L),
    # which serves the bra side too (both sides sweep the same pairs).
    kets = {(a, b): _kets(basis, rows, [(a, b, 1.0 if a == b else 2.0)],
                          pq=_no_pq, tau=0.0) for a, b in sorted(sides)}
    bras = {(a, b): primitive_pairs(shells[a], shells[b])
            for a, b in sorted(sides)}

    terms: dict[tuple[int, int], list[list[float]]] = {}  # (lp, bra row)
    n_ket: dict[int, int] = {}  # per lp: the kets a bra record walks
    blocks: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
    n_pairs = n_skipped = n_examined = 0
    dropped: list[float] = []
    for c in plan:
        n_ket[c.lp] = n_ket.get(c.lp, 0) + len(c.kets)
        blocks.setdefault(c.lp, c.bra_blocks)
        n_pairs += len(c.bras) * len(c.kets)
        for row, br in enumerate(c.bras):
            bra = bras[br.shell_pair][br.prim]
            ps = kets[br.shell_pair][0].qs
            zeta, P = bra.zeta, bra.P
            acc = terms.setdefault((c.lp, row), [[] for _ in ps])
            for i, kr in enumerate(c.kets):
                n_examined += 1
                if br.G * kr.key < tau:  # sorted: every later ket is too
                    rest = c.kets[i:]
                    n_skipped += len(rest)
                    dropped.extend(br.G * r.key for r in rest)
                    break
                k = kets[kr.shell_pair][kr.prim]
                theta = math.sqrt(zeta * k.eta / (zeta + k.eta))
                PQv = (P[0] - k.Q[0], P[1] - k.Q[1], P[2] - k.Q[2])
                for ip, p in enumerate(ps):
                    s = 0.0
                    for q, dq in zip(k.qs, k.dq, strict=True):
                        s += pq(p, q, PQv, theta) * dq
                    acc[ip].append(s)

    n = basis.n_ao
    J = [[0.0] * n for _ in range(n)]
    S = [[0.0] * n for _ in range(n)]
    for lp, blks in blocks.items():
        for a, b, start, stop in blks:
            sa, sb = shells[a], shells[b]
            recs = bras[(a, b)]
            if stop - start != len(recs):
                raise ValueError(f"the plan's block ({a}, {b}) has "
                                 f"{stop - start} records, basis has "
                                 f"{len(recs)}")
            jab = [0.0] * len(recs[0].block)
            sab = [0.0] * len(recs[0].block)
            for bra, row in zip(recs, range(start, stop), strict=True):
                jp = [math.fsum(t) for t in terms[(lp, row)]]
                for r, erow in enumerate(bra.block):
                    s = 0.0
                    for e, v in zip(erow, jp, strict=True):
                        s += e * v
                    jab[r] += s
                    sab[r] += n_ket[lp] * math.fsum(abs(e) for e in erow)
            for i in range(sa.n_cart):
                for j in range(sb.n_cart):
                    r = i * sb.n_cart + j
                    J[off[a] + i][off[b] + j] = J[off[b] + j][off[a] + i] = jab[r]
                    S[off[a] + i][off[b] + j] = S[off[b] + j][off[a] + i] = sab[r]
    return Screened(tuple(tuple(r) for r in J), tuple(tuple(r) for r in S),
                    n_pairs, n_skipped, n_examined, max(dropped, default=0.0),
                    math.fsum(dropped))


def coulomb_launched(basis: Basis, D: object, *, pq: PQ, tau: float,
                     n_batches: int = 1, bound: Bound | None = None,
                     density_sort: bool = True,
                     strict: bool = True) -> Screened:
    """`walk(launches(...), basis, D, pq=pq)`. `bound` defaults to
    `ao.j.pair_bound_pq` through `pq` -- the host sweep's own bound, so the
    two screen identically (module docstring, "WHAT EQUALS WHAT")."""
    if bound is None:
        def bound(zeta: float, L: int) -> float:
            return pair_bound_pq(zeta, L, pq=pq)
    return walk(launches(basis, D, bound=bound, n_batches=n_batches, tau=tau,
                         density_sort=density_sort, strict=strict),
                basis, D, pq=pq)


# --- the one-electron launch (#241): S, T and V, BESIDE ClassLaunch ---------

Vec3 = tuple[float, float, float]
Nucleus = tuple[float, Vec3]
PairBound = Callable[[Shell, Shell], float]  # ao.pairs.Bound: per shell pair


@dataclass(frozen=True)
class OneERec:
    """One primitive-pair record of a one-electron build: EVERYTHING the
    `S`, `T` and `V` kernels read per record, and nothing else.

    `i <= j` the shell pair (`ao.one_electron.matrix`'s upper triangle by
    shell), `prim` its index in `ao.pairs.primitive_pairs(shells[i],
    shells[j])`; `zeta`, `P`, `XPA = P - A`, `XPB = P - B`; `s = c_a c_b
    (pi/zeta)^{3/2}` (`ao.j.epair`'s, the kernels' `s` / `s_ab`); `eb` the
    KET primitive's exponent (`kernel_t`'s `eb_ket`); `off_a`, `off_b` the
    AO offsets of shells `i` and `j`. The diagonal weight is `diagonal`:
    the kernels read it from `off_a == off_b` (`kernel_jab`'s scatter rule,
    a diagonal block's lower triangle written once and mirrored), so it is
    a property of the record, not a parameter of its own.
    """

    i: int
    j: int
    prim: int
    zeta: float
    P: Vec3
    XPA: Vec3
    XPB: Vec3
    s: float
    eb: float
    off_a: int
    off_b: int

    @property
    def shell_pair(self) -> tuple[int, int]:
        return (self.i, self.j)

    @property
    def diagonal(self) -> bool:
        """`off_a == off_b`: one shell against itself, written once."""
        return self.off_a == self.off_b


_ONE_E_OPERATOR = {Target.S: Operator.OVERLAP, Target.T: Operator.KINETIC,
                   Target.V: Operator.NUCLEAR}


@dataclass(frozen=True)
class OneEClass:
    """Every record of one pair class `(la, lb)`, in pair order (shell pair
    `i` outer, then `j`, then `primitive_pairs` order). `blocks` is
    `(i, j, start, stop)` per shell pair: `recs[start:stop]` are its records.

    IS `ClassLaunch` J-SHAPED? YES, and that is why this is a sibling type,
    not a widening. `ClassLaunch` is a bra side AND a ket side, its ket
    records keyed by `I_q * max|D_q|` -- a quantity of the DENSITY -- and
    sorted on that key so Alg. 1's `break` can end a bra row, plus a batch
    count the plan admits per `(lp, lq)` and the `[p|q]` kernels built at
    it. A one-electron build has ONE side, no density, so no `D_q`, no ket
    key and no sort (the sort key #147 measured does not exist here), and
    nothing to break on: its screen is a per-shell-pair DROP before the
    launch (`ao.pairs.pair_list`), not a per-row break inside it. Its batch
    count is 1 by the milestone's decision. Widening `ClassLaunch` would
    leave `kets`, `key`, the sort and `bra_blocks`' eq. 20 meaning empty or
    false here, so `ClassLaunch`, `PairRec` and `launches` are untouched,
    byte for byte (test/test_launch.py's J fingerprint pins them).

    ONE RECORD SET, THREE MATRICES. `S`, `T` and `V` of one basis take the
    SAME records; only the kernel differs (`T` reads `eb` as well, `V` reads
    `P` and the launch's nuclei). `spec(target)` is the kernel's key and
    `arrays(target)` its per-record parameter arrays under that kernel's own
    names.
    """

    la: int
    lb: int
    recs: tuple[OneERec, ...]
    blocks: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not self.recs:
            raise ValueError(f"pair class ({self.la}, {self.lb}) has no "
                             "records: an absent class is absent, not empty")
        if (sum(stop - start for _, _, start, stop in self.blocks)
                != len(self.recs)):
            raise ValueError(f"pair class ({self.la}, {self.lb}): the blocks "
                             "do not cover the records")

    @property
    def L(self) -> int:
        return self.la + self.lb

    def spec(self, target: Target) -> Spec:
        """The kernel key of this class for `Target.S`, `T` or `V` --
        `kernel_s.s_spec` / `kernel_t.t_spec` / `kernel_v.v_spec`, restated
        (building kernels is not this plan's business; a test pins the
        restatement equal)."""
        if target not in _ONE_E_OPERATOR:
            raise ValueError(f"{target!r} is not a one-electron target")
        return Spec(_ONE_E_OPERATOR[target], 0, (self.la, self.lb), target)

    def arrays(self, target: Target) -> dict[str, list]:
        """The per-record parameter arrays of `target`'s kernel, in record
        order and in its parameter block's order, under its names
        (`kernel_s` / `kernel_t` / `kernel_v`), as plain lists. `n_rec` is
        `len(recs)`; `V`'s `n_nuc`, `Z` and `Cnuc` are the launch's."""
        self.spec(target)
        rs = self.recs
        out: dict[str, list] = {"zeta_ab": [r.zeta for r in rs]}
        if target is Target.V:
            out |= per_axis("p", [r.P for r in rs])
        out |= per_axis("pa", [r.XPA for r in rs])
        out |= per_axis("pb", [r.XPB for r in rs])
        out["s_ab"] = [r.s for r in rs]
        if target is Target.T:
            out["eb_ket"] = [r.eb for r in rs]
        out["off_a"] = [r.off_a for r in rs]
        out["off_b"] = [r.off_b for r in rs]
        return out


@dataclass(frozen=True)
class OneELaunch:
    """`one_electron_launch()`'s value: one `OneEClass` per pair class with
    a kept shell pair, sorted by `(la, lb)`; the nuclei, ONCE per launch
    (`V`'s `Z` and `Cnuc`); and the screen -- `tau`, and the shell pairs it
    `kept` and `dropped` (`ao.pairs.pair_list`'s, row-major). Iterating it
    iterates the classes."""

    classes: tuple[OneEClass, ...]
    nuclei: tuple[Nucleus, ...]
    tau: float
    kept: tuple[tuple[int, int], ...]
    dropped: tuple[tuple[int, int], ...]

    def __iter__(self) -> Iterator[OneEClass]:
        return iter(self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def n_pairs(self) -> int:
        return len(self.kept) + len(self.dropped)

    @property
    def skip_fraction(self) -> float:
        return len(self.dropped) / self.n_pairs if self.n_pairs else 0.0

    @property
    def Z(self) -> list[float]:
        return [z for z, _ in self.nuclei]

    @property
    def Cnuc(self) -> list[list[float]]:
        return [list(c) for _, c in self.nuclei]


def one_electron_launch(basis: Basis, nuclei: Sequence[Nucleus], *,
                        tau: float, bound: PairBound) -> OneELaunch:
    """The one-electron launch plan of `basis` (`OneEClass`'s docstring).

    The shell pairs `i <= j` are screened by `ao.pairs.pair_list(basis,
    tau=tau, bound=bound)` -- `bound` injected (`overlap_bound`,
    `kinetic_bound`, or a closure over `potential_bound`), and the guarantee
    is `pair_list`'s: every element of a dropped block of the matrix `bound`
    bounds is below `tau`. `tau = 0` drops nothing. THE PLAN IS DATA: no
    kernel is built, launched or interpreted, and no integral is evaluated.
    """
    nuc = _check_nuclei(nuclei)
    pl = pair_list(basis, tau=tau, bound=bound)
    shells, off = basis.shells, basis.offsets
    recs: dict[tuple[int, int], list[OneERec]] = {}
    blocks: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for i, j in pl.kept:  # row-major, i outer: ao.one_electron.matrix's
        sa, sb = shells[i], shells[j]
        pps = primitive_pairs(sa, sb)
        ep = epair(sa, sb, pps)
        side = recs.setdefault((sa.l, sb.l), [])
        start = len(side)
        for k, (pp, (P, zeta, s)) in enumerate(zip(pps, ep.recs,
                                                   strict=True)):
            side.append(OneERec(
                i, j, k, zeta, P,
                (P[0] - sa.centre[0], P[1] - sa.centre[1],
                 P[2] - sa.centre[2]),
                (P[0] - sb.centre[0], P[1] - sb.centre[1],
                 P[2] - sb.centre[2]),
                s, float(sb.exponents[pp.j]), off[i], off[j]))
        blocks.setdefault((sa.l, sb.l), []).append((i, j, start, len(side)))
    return OneELaunch(
        tuple(OneEClass(la, lb, tuple(recs[(la, lb)]),
                        tuple(blocks[(la, lb)]))
              for la, lb in sorted(recs)),
        nuc, pl.tau, pl.kept, pl.dropped)


# --- the one-electron walker: the plan, in Python ---------------------------


def _e_rec(a: Ang, b: Ang, r: OneERec) -> tuple[float, ...]:
    """`ao.hermite.e_block` from the RECORD's `XPA` / `XPB` (the block reads
    only `P - A` and `P - B`), in its order and association."""
    axes = [e_axis(a[d], b[d], r.zeta, r.XPA[d], r.XPB[d]) for d in range(3)]
    out = []
    for t in hermite_order(sum(a) + sum(b)):
        v = 1.0
        for d in range(3):
            v = v * axes[d][t[d]] if t[d] < len(axes[d]) else 0.0
        out.append(v)
    return tuple(out)


def _s_rec(carts, r: OneERec) -> list[float]:
    return [r.s * _e_rec(a, b, r)[0] for a, b in carts]


def _t_rec(la: int, lb: int, carts, r: OneERec) -> list[float]:
    S = [tuple(tuple(e_axis(i, j, r.zeta, r.XPA[d], r.XPB[d])[0]
                     for j in range(lb + 3)) for i in range(la + 1))
         for d in range(3)]
    out = []
    for a, b in carts:
        s = [S[d][a[d]][b[d]] for d in range(3)]
        k = [kinetic_axis(S[d][a[d]], b[d], r.eb) for d in range(3)]
        out.append(r.s * (k[0] * s[1] * s[2] + s[0] * k[1] * s[2]
                          + s[0] * s[1] * k[2]))
    return out


def _v_rec(la: int, lb: int, carts, r: OneERec, nuclei: Sequence[Nucleus],
           one_centre: OneCentre, acc: list[float]) -> None:
    """Adds INTO the shell pair's `acc`, one `-Z v` per (nucleus, row):
    `ao.potential.potential_block` keeps ONE accumulator chain over the
    records AND the nuclei, and a per-record partial sum would reassociate
    it (one ulp apart on H2 when tried)."""
    block = [tuple(r.s * x for x in _e_rec(a, b, r)) for a, b in carts]
    ts = hermite_order(la + lb)
    theta = math.sqrt(r.zeta)
    for Z, C in nuclei:
        pc = (r.P[0] - C[0], r.P[1] - C[1], r.P[2] - C[2])
        R = [one_centre(t, 0, pc, theta) for t in ts]
        for k, row in enumerate(block):
            v = 0.0
            for e, x in zip(row, R, strict=True):
                v += e * x
            acc[k] += -Z * v


def walk_one_electron(plan: OneELaunch, basis: Basis, target: Target, *,
                      one_centre: OneCentre | None = None
                      ) -> list[list[float]]:
    """`S`, `T` or `V` (`target`) from the PLAN alone.

    Each record's contribution is computed from the record's own fields
    (for `V` also the launch's nuclei and the INJECTED `one_centre`, the
    signature `ao.pairs.OneCentre`), the contributions of one shell pair
    are summed in record order from `0.0` -- `ao.overlap`, `ao.kinetic` and
    `ao.potential`'s arithmetic, in their order -- and the blocks are placed
    by `ao.one_electron.matrix` over `plan.kept`. So the walk equals
    `overlap_matrix(basis, pairs=plan.kept)` (and the other two) TO 0: a
    difference means a record carries a wrong value, never a rounding.
    `basis` supplies only the momenta and the AO layout. Nothing is
    memoised, and no kernel is built or interpreted.
    """
    if target not in _ONE_E_OPERATOR:
        raise ValueError(f"{target!r} is not a one-electron target")
    if target is Target.V and one_centre is None:
        raise ValueError("V needs the injected one_centre evaluator")
    shells = basis.shells
    out: dict[tuple[int, int], tuple[tuple[float, ...], ...]] = {}
    for c in plan:
        ca, cb = cartesians(c.la), cartesians(c.lb)
        carts = [(a, b) for a in ca for b in cb]
        nb = len(cb)
        for i, j, start, stop in c.blocks:
            if (shells[i].l, shells[j].l) != (c.la, c.lb):
                raise ValueError(f"the plan's block ({i}, {j}) is not class "
                                 f"({c.la}, {c.lb}) in this basis")
            acc = [0.0] * len(carts)
            for r in c.recs[start:stop]:
                if target is Target.S:
                    vs = _s_rec(carts, r)
                elif target is Target.T:
                    vs = _t_rec(c.la, c.lb, carts, r)
                else:
                    _v_rec(c.la, c.lb, carts, r, plan.nuclei, one_centre, acc)
                    continue
                for n, v in enumerate(vs):
                    acc[n] += v
            out[(i, j)] = tuple(tuple(acc[k * nb:(k + 1) * nb])
                                for k in range(len(ca)))
    if sorted(out) != sorted(plan.kept):
        raise ValueError("the plan's classes do not cover its kept pairs")
    order = iter(plan.kept)

    def block(sa: Shell, sb: Shell) -> tuple[tuple[float, ...], ...]:
        i, j = next(order)
        if shells[i] is not sa or shells[j] is not sb:
            raise ValueError(f"the kept pair ({i}, {j}) is out of order")
        return out[(i, j)]

    return matrix(basis, block, pairs=plan.kept)


# --- the cross-basis overlap launch (Target.SX), BESIDE OneELaunch ----------


@dataclass(frozen=True)
class XRec:
    """One primitive-pair record of a cross-basis overlap build: `i` a
    shell of the ROW basis `A`, `j` of the COLUMN basis `B`, `prim` the
    index in `ao.pairs.primitive_pairs(A.shells[i], B.shells[j])`; `zeta`,
    `XPA = P - A`, `XPB = P - B`, `s = c_a c_b (pi/zeta)^{3/2}`
    (`ao.j.epair`'s); `off_a` the AO offset of `i` in `A`, `off_b` of `j`
    in `B` -- offsets into DIFFERENT index sets, so a record has no
    diagonal (`kernel_sx`)."""

    i: int
    j: int
    prim: int
    zeta: float
    XPA: Vec3
    XPB: Vec3
    s: float
    off_a: int
    off_b: int

    @property
    def shell_pair(self) -> tuple[int, int]:
        return (self.i, self.j)


@dataclass(frozen=True)
class XClass:
    """Every record of one ORDERED pair class `(la in A, lb in B)`, in pair
    order (`i` outer, then `j`, then `primitive_pairs` order); `blocks` is
    `(i, j, start, stop)` per shell pair, `recs[start:stop]` its records."""

    la: int
    lb: int
    recs: tuple[XRec, ...]
    blocks: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not self.recs:
            raise ValueError(f"pair class ({self.la}, {self.lb}) has no "
                             "records: an absent class is absent, not empty")
        if (sum(stop - start for _, _, start, stop in self.blocks)
                != len(self.recs)):
            raise ValueError(f"pair class ({self.la}, {self.lb}): the blocks "
                             "do not cover the records")

    def spec(self) -> Spec:
        """`kernel_sx.sx_spec(la, lb)`, restated (a test pins it equal)."""
        return Spec(Operator.OVERLAP, 0, (self.la, self.lb), Target.SX)

    def arrays(self) -> dict[str, list]:
        """`kernel_sx`'s per-record parameter arrays, in its parameter
        block's order and under its names, as plain lists."""
        rs = self.recs
        return {"zeta_ab": [r.zeta for r in rs],
                **per_axis("pa", [r.XPA for r in rs]),
                **per_axis("pb", [r.XPB for r in rs]),
                "s_ab": [r.s for r in rs],
                "off_a": [r.off_a for r in rs],
                "off_b": [r.off_b for r in rs]}


def per_axis(name: str, vecs: Sequence[Vec3]) -> dict[str, list[float]]:
    """A record vector as the one-electron kernels take it: one flat array
    per axis, `<name>_x`, `<name>_y`, `<name>_z` (`kernel_s.pa_name`)."""
    return {f"{name}_{a}": [v[d] for v in vecs] for d, a in enumerate("xyz")}


@dataclass(frozen=True)
class XLaunch:
    """`cross_overlap_launch()`'s value: one `XClass` per ordered pair
    class with a kept shell pair, sorted by `(la, lb)`; the two AO counts;
    and the screen (`ao.pairs.cross_pair_list`'s `tau`, `kept`,
    `dropped`, row-major over `(i in A, j in B)`)."""

    classes: tuple[XClass, ...]
    n_a: int
    n_b: int
    tau: float
    kept: tuple[tuple[int, int], ...]
    dropped: tuple[tuple[int, int], ...]

    def __iter__(self) -> Iterator[XClass]:
        return iter(self.classes)

    def __len__(self) -> int:
        return len(self.classes)


def cross_overlap_launch(basis_a: Basis, basis_b: Basis, *, tau: float,
                         bound: PairBound) -> XLaunch:
    """The cross-basis overlap launch of rows `basis_a`, columns `basis_b`
    (`XClass`'s docstring). The ordered shell pairs are screened by
    `ao.pairs.cross_pair_list(basis_a, basis_b, tau=tau, bound=bound)`,
    `bound` injected (`overlap_bound`); `tau = 0` drops nothing. THE PLAN
    IS DATA: no kernel is built and no integral is evaluated."""
    pl = cross_pair_list(basis_a, basis_b, tau=tau, bound=bound)
    sha, shb = basis_a.shells, basis_b.shells
    recs: dict[tuple[int, int], list[XRec]] = {}
    blocks: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for i, j in pl.kept:
        sa, sb = sha[i], shb[j]
        ep = epair(sa, sb)
        side = recs.setdefault((sa.l, sb.l), [])
        start = len(side)
        for k, (P, zeta, s) in enumerate(ep.recs):
            side.append(XRec(
                i, j, k, zeta,
                (P[0] - sa.centre[0], P[1] - sa.centre[1],
                 P[2] - sa.centre[2]),
                (P[0] - sb.centre[0], P[1] - sb.centre[1],
                 P[2] - sb.centre[2]),
                s, basis_a.offsets[i], basis_b.offsets[j]))
        blocks.setdefault((sa.l, sb.l), []).append((i, j, start, len(side)))
    return XLaunch(
        tuple(XClass(la, lb, tuple(recs[(la, lb)]), tuple(blocks[(la, lb)]))
              for la, lb in sorted(recs)),
        basis_a.n_ao, basis_b.n_ao, pl.tau, pl.kept, pl.dropped)


def walk_cross_overlap(plan: XLaunch, basis_a: Basis, basis_b: Basis
                       ) -> list[list[float]]:
    """`S_{B1,B2}` from the PLAN alone: each record's `s E^{ab}_0` from its
    own fields, a shell pair's records summed in order from `0.0`, the
    blocks placed by `ao.one_electron.rectangular` over `plan.kept` -- so
    the walk equals `cross_overlap_matrix(basis_a, basis_b,
    pairs=plan.kept)` TO 0. The bases supply only the momenta and layout."""
    sha, shb = basis_a.shells, basis_b.shells
    out: dict[tuple[int, int], tuple[tuple[float, ...], ...]] = {}
    for c in plan:
        ca, cb = cartesians(c.la), cartesians(c.lb)
        carts = [(a, b) for a in ca for b in cb]
        nb = len(cb)
        for i, j, start, stop in c.blocks:
            if (sha[i].l, shb[j].l) != (c.la, c.lb):
                raise ValueError(f"the plan's block ({i}, {j}) is not class "
                                 f"({c.la}, {c.lb}) in these bases")
            acc = [0.0] * len(carts)
            for r in c.recs[start:stop]:
                for n, v in enumerate(_s_rec(carts, r)):
                    acc[n] += v
            out[(i, j)] = tuple(tuple(acc[k * nb:(k + 1) * nb])
                                for k in range(len(ca)))
    if sorted(out) != sorted(plan.kept):
        raise ValueError("the plan's classes do not cover its kept pairs")
    order = iter(plan.kept)

    def block(sa: Shell, sb: Shell) -> tuple[tuple[float, ...], ...]:
        i, j = next(order)
        if sha[i] is not sa or shb[j] is not sb:
            raise ValueError(f"the kept pair ({i}, {j}) is out of order")
        return out[(i, j)]

    return rectangular(basis_a, basis_b, block, pairs=plan.kept)
