"""Slice-interior sub-slicing: each sliced-HRR slice's HRR cut into ordered
sub-slices, so fewer of its ERIs and intermediates are live at once (#361,
milestone 13 lane H5).

Under #356's sliced HRR and #360's reduced buffers, a slice `(a, b)` runs
its whole HRR -- every `(a b | c d)` and its closure -- then taps and
digests its ERIs, so all `n_c n_d` of them are live together at the end of
its HRR (`hgp.hrr.hrr_peak`, its ERIs live out): `sliced_hrr.hrr_phase`, 125
values at `(dd|dd)`. This lever cuts the slice's interior into `n_sub(cls)`
sub-slices, each a contiguous run of the slice's ERIs in target order and
its own HRR sub-DAG (`hgp.hrr.sub_slices`):

    for each bra pair s = (a, b):          #356's scoped one-trip slices
        { #360's reduced rows, zeroed      unchanged
          for each sub-slice u of s:       one-trip scope SUB, in order
              { the HRR of u's ERIs only   closure of the run, the parent's
                                           terms in the parent's order
                the tap of u's ERIs        at their global target indices
                the predigest of u's ERIs } #360's statements, unchanged
          #360's rows flushed }            unchanged

so a sub-slice's ERIs die after its predigest, before the next sub-slice's
HRR: the slice-interior peak is `sub_phase(cls)` (41 at `(dd|dd)`, 3.05x
below the slice's 125). The shared nodes -- a bra-HRR `(a b | f 0)` two
sub-slices read, a staged leaf -- are recomputed by each (`statements`
counts them: 10,098 HRR statements per quartet at `(dd|dd)` -> 17,280).

THE CUT, A DECLARED FUNCTION OF THE CLASS (`n_sub`). The slice's targets
are `(a b | c d)` with `c` outer and `d` inner (`hgp.hrr.targets`), and the
ket HRR reads, for each `c`, a chain of `(a b | c' 0)` the other `c`'s
share only partly. The rule cuts at the ket's outer index:

    n_sub(cls) = n_c        one sub-slice per ket component c, its n_d ERIs
                 n_d        when n_c = 1 (lc = 0): one per ERI

(`OVERRIDES` empty). `(ss|ss)` is 1: one ERI per slice, nothing to cut, and
the kernel is #360's with one more scope. All 21 classes' counts, slice
peaks, sub-slice peaks and statement counts are pinned by
test/test_f_lever_sub_slice.py.

ATOMIC TRAFFIC UNCHANGED. The rows are #360's, zeroed before the slice's
first sub-slice and flushed after its last; `f_cd` is still streamed per
ERI by #360's `predigest_eri`; `f_ab` is untouched. So the `AtomicAdd`s,
their targets and their per-thread ORDER are #360's exactly
(`slim_fock.ket_atomics(cls, lever=True)` per worked quartet).

THE ARITHMETIC IS #360'S, BIT FOR BIT. Every HRR node a sub-slice computes
is the literal node's expression on the same staged leaves; the sub-slices
of a slice are contiguous runs in target order, so the predigest visits the
slice's ERIs in #360's order with #360's statements, every buffer
element and every atomic receives the same products in the same order.
`F` and the tap `g` equal #360's exactly (interpreted, every class with L
<= 4), and so are held to the literal kernel by #360's derived bound.

THE GROUPS DO NOT MOVE. #356 re-cut #350's VRR groups at its sliced floor;
this lever does not re-cut them at the sub-sliced one. The re-cut count
would be large (`batched_vrr.n_groups(cls, floor=sub_phase(cls))`: 233
against 19 at `(pp|dd)`), multiplying the VRR recompute, and it would make
the ledger's A/B against #360 two changes instead of one. So the VRR phase
is #360's (`batched_vrr.vrr_phase(cls, sliced_hrr.n_groups(cls))`) and the
model's win is the HRR phase's alone -- the issue's "small win on the model".

A COMPOSITION ON #360: `sub-slice` REQUIRES `slim-fock` (and so
`sliced-hrr` and `batched-vrr`) in the same spec, the key
`batched-vrr+sliced-hrr+slim-fock+sub-slice`: the sub-slices write "into
the same reduced buffers". Refused alone with `REFUSAL`. `hooks_for` applies
key order and `sub-slice` sorts after `slim-fock`, so `apply` checks that
`contracted` is exactly #360's (`slim_fock.contracted`): still #356's or the
frozen one means it ran first, and any other -- e.g. #359's interleaved
slice body, which this lever cannot verify it preserves -- is refused rather
than replaced silently.

THE HOOK IT MOVES: `contracted` only, to `slim_fock.contracted_over(body)`
with this lever's per-slice body (`sub_body`). The `hrr` hook is called once
per sub-slice on the sub-slice's context (`sub_context`: `hdag` its DAG,
`extra[sliced_hrr.NAME]` a `Slice` whose `indices` are the run's global
target indices, `extra[NAME]` its `(u, n_sub)`), so a lever wrapping `hrr`
wraps every sub-slice; `eri`, `predigest_eri`, `zero_buffers` and
`flush_ket` are called through the composed hooks.

Like the seams, nothing here imports `plan/`, `ao/`, the interpreter,
`reference/`, a backend, a device or `kernel_f`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from felizjueves.eri.fock.levers import sliced_hrr as _sh
from felizjueves.eri.fock.levers import slim_fock as _sf
from felizjueves.eri.fock.levers.seams import FContext, FHooks
from felizjueves.hgp import hrr as _hrr
from felizjueves.hgp.vrr import AmClass, cartesians
from felizjueves.ir import For, Lit, Note, Stmt
from felizjueves.spec import Spec

__all__ = [
    "NAME",
    "OVERRIDES",
    "REFUSAL",
    "REQUIRES",
    "SUB",
    "apply",
    "contracted",
    "n_sub",
    "rule",
    "statements",
    "sub_body",
    "sub_context",
    "sub_peaks",
    "sub_phase",
    "sub_slices_of",
]

NAME = "sub-slice"
REQUIRES = _sf.NAME
SUB = "su"  # the one-trip loop that scopes one sub-slice

REFUSAL = (
    "sub-slice requires slim-fock (and so sliced-hrr and batched-vrr) in the "
    "same spec (#361): the sub-slices cut a sliced-HRR slice's interior and "
    "write into #360's reduced per-slice buffers; without the slices there "
    "is no slice interior to cut")

# Per-class overrides of `rule` (class -> n_sub). Empty: every class ships
# the rule's count. An entry here is a decision, stated beside it.
OVERRIDES: Mapping[AmClass, int] = MappingProxyType({})


# ---------------------------------------------------------------------------
# The declared functions of the class
# ---------------------------------------------------------------------------


def rule(cls: AmClass) -> int:
    """`n_c`, one sub-slice per ket component `c`; `n_d` when `n_c = 1`."""
    _, _, lc, ld = cls
    nc, nd = len(cartesians(lc)), len(cartesians(ld))
    return nc if nc > 1 else nd


def n_sub(cls: AmClass) -> int:
    """The sub-slices per slice: `OVERRIDES`, else `rule`."""
    cls = tuple(cls)
    return OVERRIDES.get(cls, rule(cls))


@cache
def _sub_dags(cls: AmClass) -> tuple[tuple[_hrr.HrrDag, ...], ...]:
    return _hrr.sub_slices(cls, n_sub(cls))


@cache
def sub_peaks(cls: AmClass) -> tuple[int, ...]:
    """Each sub-slice's live set (`hgp.hrr.hrr_peak`, its ERIs live out),
    slice-major."""
    return tuple(_hrr.hrr_peak(d) for s in _sub_dags(tuple(cls)) for d in s)


def sub_phase(cls: AmClass) -> int:
    """The slice-interior peak under the lever: the largest sub-slice's live
    set, against `sliced_hrr.hrr_phase(cls)`."""
    return max(sub_peaks(tuple(cls)))


def statements(cls: AmClass, *, lever: bool) -> int:
    """HRR statements per quartet: summed over #356's slices (`lever=False`)
    or over the sub-slices (shared nodes recomputed)."""
    cls = tuple(cls)
    if not lever:
        return sum(d.n_statements for d in _hrr.slices(cls))
    return sum(d.n_statements for s in _sub_dags(cls) for d in s)


# ---------------------------------------------------------------------------
# The sub-slices, as contexts
# ---------------------------------------------------------------------------


def sub_slices_of(sl: _sh.Slice, cls: AmClass) -> tuple[_sh.Slice, ...]:
    """Slice `sl`'s sub-slices, each a `Slice` of the same bra pair whose
    `dag` is the run's sub-DAG and `first` the run's global index; checked
    to partition `sl`'s ERIs in order."""
    out, first = [], sl.first
    for d in _sub_dags(tuple(cls))[sl.index]:
        out.append(_sh.Slice(sl.index, sl.a, sl.b, first, d))
        first += d.n_eri
    assert sum((s.dag.targets for s in out), ()) == sl.dag.targets
    return tuple(out)


def sub_context(sub: FContext, u: int, part: _sh.Slice, n: int) -> FContext:
    """The slice context `sub` narrowed to sub-slice `u` of `n`: `hdag` its
    DAG, `extra[sliced_hrr.NAME]` its `Slice`, `extra[NAME] = (u, n)`."""
    ctx = _sh.slice_context(sub, part)
    return dataclasses.replace(ctx, extra={**ctx.extra, NAME: (u, n)})


def sub_body(sub: FContext) -> tuple[Stmt, ...]:
    """One slice's statements: per sub-slice, in order, its own scope with
    #356's default slice body (HRR, tap, predigest) on its context."""
    sl: _sh.Slice = sub.extra[_sh.NAME]
    parts = sub_slices_of(sl, sub.cls)
    out: list[Stmt] = []
    for u, part in enumerate(parts):
        ctx = sub_context(sub, u, part, len(parts))
        out += [
            Note(f"sub-slice (#361) {u + 1} of {len(parts)}: ERIs "
                 f"{part.first}..{part.first + part.dag.n_eri - 1}, "
                 f"{part.dag.n_statements} HRR nodes"),
            For(SUB, Lit(u), Lit(u + 1), Lit(1), tuple(_sh.slice_body(ctx))),
        ]
    return tuple(out)


contracted = _sf.contracted_over(sub_body)


# ---------------------------------------------------------------------------
# The lever
# ---------------------------------------------------------------------------


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """Sub-slice #360's slices. Refused without `slim-fock` (`REFUSAL`),
    when applied before it, or after a lever that moved `contracted`."""
    if REQUIRES not in spec.kernel_body_levers():
        raise ValueError(f"{spec.key()}: {REFUSAL}")
    if hooks.contracted is not _sf.contracted:
        early = (hooks.is_default("contracted")
                 or hooks.contracted is _sh.contracted)
        raise ValueError(
            f"{spec.key()}: {NAME} cuts #360's slices, but `contracted` is "
            + ("not yet #360's (applied before slim-fock; hooks_for applies "
               "key order)" if early else
               "another lever's, which this lever would replace silently"))
    return dataclasses.replace(hooks, contracted=contracted)
