"""The dedicated sub-DAG VRR schedule: each VRR group scheduled for itself
(#357, milestone 13 lane H6).

#350's batched VRR (`eri/fock/levers/batched_vrr.py`) cuts the boundary into
schedule-contiguous groups and runs each group's closure of the VRR DAG
inside its own primitive nest, in the PARENT DAG's order filtered to the
closure (`hgp.vrr.group_dag`'s default). That order's priorities are the
whole DAG's: a value the parent still reads later is kept in step with the
parent's schedule even where no node of the group reads it again. This
lever re-schedules every group's closure FOR THAT GROUP: a greedy
live-minimising list schedule over the closure's own edges, its consumer
counts taken inside the closure, kept only where it lowers the group's peak
live in `hgp.vrr.peak_live`'s statement-order model
(`hgp.vrr.dedicated_order`; the module text there is the rule). A cheap
scheduling change, not a second architecture:

    for each group g of #350's cut (its count unchanged):
        for kcd: ket_prim; for kab:
            quartet scalars
            the VRR of g's closure, in hgp.vrr.dedicated_order(g)   <- here
            boundary[slot(t)] += [t]^(0) for t in g

A COMPOSITION, NOT A STANDALONE LEVER: `subdag-vrr` REQUIRES `batched-vrr`
in the same spec (`batched-vrr+subdag-vrr`, the key spelling both; #356's
`sliced-hrr` set the precedent). Without #350 there are no groups: the
literal kernel's VRR is one whole-DAG closure whose targets are register
accumulators live across the nest, and re-ordering it would be a different
lever (a whole-DAG schedule) under a name that says "per group". That
kernel is refused (`REFUSAL`), not built. Implying the staging instead would
let the key `subdag-vrr` name a kernel that also carries #350's staging and
grouping, which no key segment says. `batched-vrr+sliced-hrr+subdag-vrr` is
legal: #356 re-cuts #350's groups at the sliced floor, and this lever
schedules whatever groups the nest it finds hands the `vrr` hook.

THE GROUP COUNT DOES NOT MOVE. The cut is #350's (or #356's re-cut), at the
count its rule chose from the FILTERED order's peaks. A dedicated schedule
lowers those peaks, so #350's rule read at the dedicated peaks could choose
fewer groups; that would change two things at once, and the issue asks for
the schedule alone, so the count is the one the nest was built with.

THE HOOK IT MOVES: `vrr`, wrapped. The previous `vrr` hook (the default, or
a lever's before this one) is called on the context with `vdag` replaced by
`hgp.vrr.dedicated(ctx.vdag)` -- the same closure, the same terms, the
dedicated order -- so a lever that rewrites the VRR's statements (#354's
ETR, #355's precision) still applies, to the re-ordered nodes. `apply`
checks that `batched-vrr` staged the boundary first (`hooks_for` applies key
order: `batched-vrr`, `sliced-hrr`, `subdag-vrr`) rather than assume it.

THE ARITHMETIC IS #350's, AND SO THE LITERAL KERNEL's, BIT FOR BIT. Each VRR
node is one declaration whose expression (its terms, `reduce`'s order) is
unchanged; only the order of the declarations moves, and every order is
topological (checked by `group_dag`). No value is computed differently, no
sum is reassociated, the accumulation, HRR, tap and digestion are #350's,
so `F` and the `eris=True` tap `g` equal the literal kernel's exactly
(test/test_f_lever_subdag_vrr.py, interpreted, every class with L <= 4).

Like the seams, nothing here imports `plan/`, `ao/`, the interpreter,
`reference/`, a backend, a device or `kernel_f`.
"""

from __future__ import annotations

import dataclasses

from felizjueves.eri.fock.levers import batched_vrr as _bv
from felizjueves.eri.fock.levers.seams import FContext, FHooks
from felizjueves.hgp import vrr as _vrr
from felizjueves.spec import Spec

__all__ = ["NAME", "REFUSAL", "REQUIRES", "apply", "group_peaks", "schedule"]

NAME = "subdag-vrr"
REQUIRES = _bv.NAME

REFUSAL = (
    "subdag-vrr requires batched-vrr in the same spec (#357): without #350's "
    "groups the literal VRR is one whole-DAG closure feeding register "
    "accumulators live across the nest, so there is no group sub-DAG to "
    "schedule; re-ordering the whole DAG is another lever, and implying the "
    "staging would let the key name a kernel it does not spell"
)


_CACHE: dict[tuple, _vrr.VrrDag] = {}


def schedule(dag: _vrr.VrrDag) -> _vrr.VrrDag:
    """`hgp.vrr.dedicated(dag)`, memoised on the DAG's content (its class,
    targets, order and terms), so a kernel built twice schedules once."""
    key = (dag.cls, dag.targets, tuple(dag.order),
           tuple(dag.terms[n] for n in dag.order))
    hit = _CACHE.get(key)
    if hit is None:
        hit = _CACHE[key] = _vrr.dedicated(dag)
    return hit


def group_peaks(cls, n: int | None = None) -> tuple[tuple[int, int, int],
                                                     ...]:
    """Per group of #350's cut at `n` groups (default `batched_vrr.n_groups`):
    `(statements, filtered peak, dedicated peak)` -- the closure's node
    count (the same in both orders) and `hgp.vrr.vrr_peak` of each order."""
    cls = tuple(cls)
    n = _bv.n_groups(cls) if n is None else n
    return tuple((g.n_nodes, _vrr.vrr_peak(g), _vrr.vrr_peak(schedule(g)))
                 for g in _vrr.groups(cls, n))


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """Wrap the `vrr` hook to schedule each group's closure for itself.
    Refused without `batched-vrr` (`REFUSAL`)."""
    if REQUIRES not in spec.kernel_body_levers():
        raise ValueError(f"{spec.key()}: {REFUSAL}")
    if hooks.is_default("boundary_read"):
        raise ValueError(f"{spec.key()}: {NAME} applied before {REQUIRES} "
                         "staged the boundary (hooks_for applies key order)")
    prev = hooks.vrr

    def vrr(ctx: FContext):
        return prev(dataclasses.replace(ctx, vdag=schedule(ctx.vdag)))

    return dataclasses.replace(hooks, vrr=vrr)
