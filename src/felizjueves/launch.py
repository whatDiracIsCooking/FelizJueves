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

from felizjueves.ao.j import (
    PQ,
    Screened,
    _density,
    _kets,
    _tau,
    pair_bound_pq,
)
from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis
from felizjueves.ir import Kernel
from felizjueves.kernel import build_pq_kernels
from felizjueves.plan.pipeline import plan
from felizjueves.spec import Operator, Spec, Target

__all__ = ["Bound", "ClassLaunch", "LaunchPlan", "PairRec", "class_spec",
           "coulomb_launched", "launches", "walk"]

Bound = Callable[[float, int], float]  # (zeta, L) -> I_pair >= 0


@dataclass(frozen=True)
class PairRec:
    """One bra or ket index record (module docstring, "THE KEYS").

    `i, j` the shell pair (`i >= j`), `prim` its primitive pair's index in
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


def _g(bound: Bound, zeta: float, L: int) -> float:
    g = bound(zeta, L)
    if not (isinstance(g, int | float) and math.isfinite(g) and g >= 0.0):
        raise ValueError(f"bound({zeta!r}, {L}) = {g!r} is not a finite "
                         "float >= 0")
    return float(g)


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
