"""The batched VRR: the boundary cut into schedule-contiguous groups, staged in a
thread-private array (#350, milestone 13 lane H1).

`kernel_f` holds EVERY boundary accumulator of a contracted quartet live
across the K_CD x K_AB primitive nest -- 961 of them at `(dd|dd)` (255
registers and 97,464 B of spill stores on sm_86, `kernels.json`) -- because
eq 28's HRR needs contracted values (milestone 12, pin 5). This lever trades
recompute for registers:

    boundary[n_bnd] = 0                     thread-private, Space.LOCAL
    for each group g of the boundary (schedule-contiguous, below):
        for kcd < K_CD:                     the nest, once per group
            ket_prim
            for kab < K_AB:
                quartet scalars             the ones g's VRR reads
                the VRR of g's closure only (hgp.vrr.group_dag)
                boundary[slot(t)] += [t]^(0)   for t in g
    the HRR reads boundary[slot(e, f)]      the literal HRR, its leaves loads

so only one group's VRR interior is live in the nest, and the accumulators
live in an array ptxas / the AMDGPU backend place in (per-thread) local
memory: `ir.Decl` with a shape and `Space.LOCAL`, no IR widening. The slot of
`(e, f)` is its index in `hgp.recurrence.boundary_order` (= the HRR's leaf
order), whatever group computes it.

THE GROUPS (`hgp.vrr`, #350's additive part): the targets in the order the
pruned DAG's `order` produces them, cut into `n` contiguous near-equal runs
(`vrr.group_targets`); each run's VRR is its closure under the parent DAG's
own terms, in the parent's order filtered (`vrr.group_dag`). A node two
groups share is recomputed by both, from the same expression on the same
inputs.

THE ARITHMETIC IS THE LITERAL KERNEL'S, BIT FOR BIT. Every VRR node a group
computes is the literal node's expression (the parent's terms) on the same
quartet scalars (the same expressions, re-formed per group); each slot is
summed over the same primitive pairs in the same order (kcd outer, kab
inner) from 0.0, as its register accumulator was; the HRR, the tap and the
digestion read the same values. No fold is reassociated, so `F` and the
`eris=True` tap `g` are the literal kernel's exactly
(test/test_f_lever_batched_vrr.py, interpreted, every class with L <= 4).

THE GROUP COUNT, A DECLARED FUNCTION OF THE CLASS (`n_groups`). `Spec`'s
`n_batches` stays 1 for F (`spec._F_BATCH_REFUSAL`), and a kernel-body
lever's name is one entry of a closed vocabulary (`levers.VOCABULARY`, the
key's `+`-joined segment), so the count is not a field and not part of the
name: it is `rule(cls)`, unless `OVERRIDES` names the class. The rule is the
predecessor's, "the smallest count that drops the VRR phase to the HRR
floor", in this tree's live model (`hgp.vrr.peak_live`, statement order):

    vrr_phase(cls, n) = max over the n groups of vrr_peak(group)
                        (the accumulators are in local memory, not counted)
    literal_phase(cls) = n_bnd + vrr_peak(whole DAG)
                        (the literal kernel carries every accumulator)
    hrr_floor(cls)    = peak_live of the HRR in its emitted order, its ERIs
                        live to the end (the predigest reads them after it)
    rule(cls, floor)  = the smallest n in 1..n_bnd with
                        vrr_phase(cls, n) <= floor (default hrr_floor(cls));
                        if no n reaches it, the n of least vrr_phase

`floor=` is the composition point for #356: a sliced HRR's floor is one
slice's live set, far below the whole HRR's, and the same rule then cuts more
groups. On today's unsliced HRR the floor is high (`(dd|dd)`: 1,550 against
a whole-DAG VRR peak of 1,078), so the rule gives 1 to 3 groups on the 21
classes -- 3 only at `(sd|dd)` -- (the table is pinned by
test/test_f_lever_batched_vrr.py): where it is 1, the staging alone drops
the VRR phase below the HRR's, which is the honest reading of the rule here,
not a defect of the lever. `with_groups(n)` builds the same lever at any
other count (an experiment's arm; never shipped under the name).

THE HOOKS IT MOVES: `boundary_init` (the array and its zeroing),
`nest` (one nest per group), `accumulate` (into the array) and
`boundary_read` (the HRR's leaf loads). Each group's nest runs the COMPOSED
`ket_prim`, `quartet_scalars`, `vrr` and `accumulate` hooks on a context
whose `vdag` is the group's sub-DAG (`group_context`), so

  - #357 (a per-group schedule) moves `vrr` (or rebuilds `ctx.vdag` with
    `vrr.group_dag(..., order=)`) and sees each group's DAG;
  - #356 (the sliced HRR) reads the staged boundary through `boundary_read`
    (`cell`), whatever it does to `hrr` / `contracted`;
  - #355 (precision) reaches the array's type through `real(ctx,
    "boundary")` and its zeroing through `cast`.

Like the seams, nothing here imports `plan/`, `ao/`, the interpreter,
`reference/`, a backend, a device or `kernel_f`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from felizjueves.f_levers.seams import KAB, KCD, FContext, FHooks
from felizjueves.hgp import hrr as _hrr
from felizjueves.hgp import vrr as _vrr
from felizjueves.hgp.recurrence import boundary_order, vrr_name
from felizjueves.hgp.vrr import AmClass, Cart
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Fn,
    For,
    Index,
    Lit,
    Note,
    Space,
    Stmt,
    Var,
)
from felizjueves.spec import Spec

__all__ = [
    "ARRAY",
    "NAME",
    "OVERRIDES",
    "ZERO",
    "apply",
    "cell",
    "group_context",
    "hrr_floor",
    "literal_phase",
    "n_groups",
    "nest_at",
    "rule",
    "slot",
    "vrr_phase",
    "with_groups",
]

NAME = "batched-vrr"
ARRAY = "boundary"  # the staged boundary: thread-private, Space.LOCAL
ZERO = "ib"  # the zeroing loop's index

# Per-class overrides of `rule` (class -> group count). Empty: every class
# ships the rule's count. An entry here is a decision, stated beside it.
OVERRIDES: Mapping[AmClass, int] = MappingProxyType({})


# ---------------------------------------------------------------------------
# The group count (module docstring, THE GROUP COUNT)
# ---------------------------------------------------------------------------


@cache
def hrr_floor(cls: AmClass) -> int:
    """Peak live values of the literal HRR in its emitted order, its ERIs
    live to the end (`hgp.vrr.peak_live` over `hgp.hrr.build(cls)`)."""
    h = _hrr.build(tuple(cls))
    return _vrr.peak_live(h.order, lambda n: (t.child for t in h.terms[n]),
                          h.targets)


@cache
def vrr_phase(cls: AmClass, n: int) -> int:
    """The batched nest's VRR-phase demand at `n` groups: the largest
    group's `vrr_peak` (the staged accumulators are not registers)."""
    return max(_vrr.vrr_peak(g) for g in _vrr.groups(tuple(cls), n))


@cache
def literal_phase(cls: AmClass) -> int:
    """The literal kernel's VRR-phase demand: every boundary accumulator
    carried, plus the whole DAG's `vrr_peak`."""
    dag = _vrr.build(tuple(cls))
    return len(dag.targets) + _vrr.vrr_peak(dag)


def rule(cls: AmClass, floor: int | None = None) -> int:
    """The smallest group count whose VRR phase is at most `floor` (default
    `hrr_floor(cls)`); the count of least VRR phase if none reaches it."""
    cls = tuple(cls)
    floor = hrr_floor(cls) if floor is None else floor
    if type(floor) is not int or floor < 1:
        raise ValueError(f"floor {floor!r} is not a positive int")
    n_bnd = len(_vrr.targets(cls))
    best = (vrr_phase(cls, 1), 1)
    for n in range(1, n_bnd + 1):
        demand = vrr_phase(cls, n)
        if demand <= floor:
            return n
        best = min(best, (demand, n))
    return best[1]


def n_groups(cls: AmClass, floor: int | None = None) -> int:
    """The shipped group count of `cls`: `OVERRIDES`, else `rule` (at
    `floor`, which a composing lever passes: #356's sliced HRR)."""
    cls = tuple(cls)
    return OVERRIDES[cls] if cls in OVERRIDES else rule(cls, floor)


# ---------------------------------------------------------------------------
# The hooks
# ---------------------------------------------------------------------------


def slot(ctx: FContext, e: Cart, f: Cart) -> int:
    """`(e, f)`'s index in the staged array: `boundary_order`'s."""
    return _slots(ctx.cls)[(e, f)]


@cache
def _slots(cls: AmClass) -> Mapping[tuple[Cart, Cart], int]:
    return MappingProxyType({ef: i for i, ef in
                             enumerate(boundary_order(cls))})


def cell(ctx: FContext, e: Cart, f: Cart) -> Index:
    """The staged boundary value `(e0|f0)`: `boundary[slot]`."""
    return Index(ARRAY, (Lit(slot(ctx, e, f)),))


def boundary_init(ctx: FContext) -> tuple[Stmt, ...]:
    n = len(ctx.bnd)
    return (
        Note(f"batched VRR (#350): the {n} contracted boundary values "
             "staged in a thread-private array (local memory), zeroed per "
             "ket; the HRR reads its leaves from it"),
        Decl(ARRAY, ctx.hooks.real(ctx, "boundary"), (Lit(n),), Space.LOCAL),
        For(ZERO, Lit(0), Lit(n), Lit(1), (
            Assign(Index(ARRAY, (Var(ZERO),)),
                   ctx.hooks.cast(ctx, Lit(0.0), "boundary")),)),
    )


def group_context(ctx: FContext, group: _vrr.VrrDag, g: int, n: int
                  ) -> FContext:
    """`ctx` with `vdag` the group's sub-DAG (so `vbody` / `vneed` are the
    group's) and `extra[NAME] = (g, n)`."""
    return dataclasses.replace(ctx, vdag=group,
                               extra={**ctx.extra, NAME: (g, n)})


def accumulate(ctx: FContext) -> tuple[Stmt, ...]:
    """eq 27 into the staged array, for the targets of `ctx.vdag` (a group's
    under `nest`, every boundary value otherwise)."""
    out: list[Stmt] = []
    for e, f, _ in ctx.vdag.targets:
        c: Expr = cell(ctx, e, f)
        out.append(Assign(c, Call(Fn.ADD, (c, Var(vrr_name((e, f, 0)))))))
    return tuple(out)


def _nest(ctx: FContext, n: int) -> tuple[Stmt, ...]:
    h, out = ctx.hooks, []
    for g, gdag in enumerate(_vrr.groups(ctx.cls, n)):
        sub = group_context(ctx, gdag, g, n)
        quartet = (*h.quartet_scalars(sub), *sub.vbody, *h.accumulate(sub))
        out += [
            Note(f"VRR group {g + 1} of {n}: {len(gdag.targets)} boundary "
                 f"values, a closure of {gdag.n_nodes} nodes; the K_CD x "
                 "K_AB nest re-run for it"),
            For(KCD, Lit(0), Var("k_cd"), Lit(1), (
                *h.ket_prim(sub),
                For(KAB, Lit(0), Var("k_ab"), Lit(1), quartet))),
        ]
    return tuple(out)


def with_groups(n: int | None = None):
    """The lever's `apply` at a fixed group count `n` (`None`: the class's
    `n_groups`). `apply` is `with_groups()`."""
    if n is not None and (type(n) is not int or n < 1):
        raise ValueError(f"group count {n!r} is not a positive int")

    def apply_n(hooks: FHooks, spec: Spec) -> FHooks:
        count = n_groups(tuple(spec.am)) if n is None else n
        n_bnd = len(_vrr.targets(tuple(spec.am)))
        if count > n_bnd:
            raise ValueError(f"{tuple(spec.am)}: {count} groups of {n_bnd} "
                             "boundary values")
        return dataclasses.replace(
            hooks, boundary_init=boundary_init, accumulate=accumulate,
            nest=nest_at(count),
            boundary_read=lambda ctx, e, f: cell(ctx, e, f))

    return apply_n


def nest_at(n: int):
    """The `nest` hook at `n` groups: what `with_groups(n)` installs, for a
    composing lever that re-cuts the groups (#356 installs the count
    `n_groups(cls, floor=)` gives at its sliced HRR's floor)."""
    return lambda ctx: _nest(ctx, n)


apply = with_groups()
