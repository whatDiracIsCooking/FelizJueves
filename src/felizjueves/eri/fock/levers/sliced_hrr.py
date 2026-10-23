"""The sliced HRR: one HRR per bra pair, reading the staged boundary (#356,
milestone 13 lane H2).

`kernel_f` runs eq 28's HRR WHOLE after the primitive nest: every node of
`hgp.hrr.build(cls)`, then the tap, then the predigest of all `n_eri`
targets, so the whole HRR's live set -- 1,550 values at `(dd|dd)` in
`hgp.vrr.peak_live`'s model (`batched_vrr.hrr_floor`) -- is one phase. This
lever cuts that phase per bra pair `(a, b)`:

    digest_setup; zero_buffers              once per quartet, as before
    for each bra pair s = (a, b), in (a, b) order:
        { the HRR of slice s only           hgp.hrr.slices(cls)[s]: the
                                            closure of (a b | * *), its
                                            leaves loads from boundary[]
          the tap of slice s's ERIs         (eris=True), at their global
                                            hgp.hrr.targets indices
          the predigest of slice s's ERIs } six multiply-adds each
    flush_ket

so only one slice's interior is ever live (the predecessor's ledger row 4).
Each slice is its own C scope -- a one-trip loop over `SLICE`, `s` to
`s + 1` -- because two slices declare the nodes they share (a bra-HRR
intermediate `(a' b' | f 0)`, a leaf) under the same `hrr_name`.

A COMPOSITION, NOT A STANDALONE LEVER: `sliced-hrr` REQUIRES `batched-vrr`
in the same spec (`batched-vrr+sliced-hrr`, the key spelling both). Without
#350 the boundary is `n_bnd` register accumulators declared before the nest
and read by the slices, so every one stays live until the last slice reads
it: the HRR phase's live set is still at least `n_bnd` (961 at `(dd|dd)`),
and slicing would add the recompute of the shared nodes and remove nothing.
That kernel is refused (`REFUSAL`), not built. Implying the staging instead
would let the key `sliced-hrr` name a kernel that also carries #350's
staging and grouping, which no key segment says; the key names every lever a
kernel carries. The composition is where the issue's "the composition with
row 2 is what makes the whole family pay" lives, and its contribution is
separable because #350 alone is its own key (`batched-vrr`).

THE GROUP COUNT MOVES WITH IT. `batched_vrr.rule(cls, floor=)` is the
smallest group count whose VRR phase drops to the HRR's floor. With slices
the floor is one slice's live set (`hrr_phase`), far below the whole
HRR's, so `apply` re-installs #350's nest at `batched_vrr.n_groups(cls,
floor=hrr_phase(cls))` (`n_groups` here). `hooks_for` applies the levers in
key order, so `batched-vrr` has already moved its four hooks when `apply`
runs; `apply` checks that (`boundary_read` staged) rather than assumes it.

THE ARITHMETIC IS THE LITERAL KERNEL'S, BIT FOR BIT. Every HRR node a slice
computes is the literal node's expression (the parent's terms, the same
`AB` / `CD` scalars) on the same staged leaves, which #350 sums exactly as
the literal accumulators are summed; the predigest visits the ERIs in the
literal order (slices in `(a, b)` order are contiguous runs of
`hgp.hrr.targets`) with the literal statements (`predigest_eri`), and
`digest_setup` reads only the density, so moving it ahead of the HRR moves
no value. `F` and the tap `g` are the literal kernel's exactly
(test/test_f_lever_sliced_hrr.py, interpreted, every class with L <= 4).

THE DECLARED FUNCTIONS OF THE CLASS: `n_slices(cls)` = `n_a n_b`;
`slice_peaks(cls)`, each slice's `hgp.hrr.hrr_peak` (its ERIs live to the
end, as the predigest reads them after the slice's HRR); `hrr_phase(cls)`,
their maximum; `n_groups(cls)`, #350's count at that floor. All 21 are
pinned by the test.

THE HOOKS IT MOVES: `contracted` (the slices, in place of the whole HRR, the
whole tap and `digest`) and `nest` (#350's, re-cut). The default `hrr` hook
is still called, once per slice on that slice's context (`slice_context`:
`hdag` the slice's DAG, `extra[NAME]` its `Slice`), so a lever wrapping
`hrr` (#355's precision) wraps every slice; `predigest_eri`,
`digest_setup`, `zero_buffers`, `flush_ket` and `eri` are called through the
composed hooks too. `tap`, `digest` and `predigest` are NOT called: a lever
that moved one of them before this one is refused (its change would be
dropped silently). For the levers that build on the slices:

  - #359 (interleaved digest per slice) supplies its own per-slice body to
    `contracted_with(body)`; `slice_body` is the default body (HRR, tap,
    predigest) and `slice_tap` / `slice_predigest` its parts;
  - #360 (per-bra-element Fock buffers) moves `zero_buffers` / `flush_ket`
    to the reduced set and flushes per slice in its body;
  - #361 (sub-slicing) cuts `Slice.dag` with `hgp.hrr.sub_dag`.

Like the seams, nothing here imports `plan/`, `ao/`, the interpreter,
`reference/`, a backend, a device or `kernel_f`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from felizjueves.eri.fock.levers import batched_vrr as _bv
from felizjueves.eri.fock.levers.seams import OUT, FContext, FHooks, cart_names
from felizjueves.hgp import hrr as _hrr
from felizjueves.hgp.vrr import AmClass, cartesians
from felizjueves.ir import Assign, For, Index, Lit, Note, Stmt
from felizjueves.spec import Spec

__all__ = [
    "NAME",
    "REFUSAL",
    "REQUIRES",
    "SLICE",
    "Slice",
    "apply",
    "contracted",
    "contracted_with",
    "hrr_phase",
    "n_groups",
    "n_slices",
    "slice_body",
    "slice_context",
    "slice_peaks",
    "slice_predigest",
    "slice_tap",
    "slices_of",
]

NAME = "sliced-hrr"
REQUIRES = _bv.NAME
SLICE = "sl"  # the one-trip loop that scopes one slice

REFUSAL = (
    "sliced-hrr requires batched-vrr in the same spec (#356): without #350's "
    "staged boundary the HRR's leaves are register accumulators declared "
    "before the nest and read by the slices, so all of them stay live until "
    "the last slice and slicing cuts no live set, only adds recompute")

_BYPASSED = ("tap", "digest", "predigest")


# ---------------------------------------------------------------------------
# The declared functions of the class
# ---------------------------------------------------------------------------


def n_slices(cls: AmClass) -> int:
    """One slice per bra pair `(a, b)`: `n_a n_b`."""
    return len(_hrr.slice_targets(tuple(cls)))


@cache
def slice_peaks(cls: AmClass) -> tuple[int, ...]:
    """Each slice's live set: `hgp.hrr.hrr_peak` of its sub-DAG."""
    return tuple(_hrr.hrr_peak(s) for s in _hrr.slices(tuple(cls)))


def hrr_phase(cls: AmClass) -> int:
    """The sliced HRR phase's demand: the largest slice's live set (the
    staged leaves are not registers). #350's floor under this lever."""
    return max(slice_peaks(tuple(cls)))


def n_groups(cls: AmClass) -> int:
    """#350's group count at the sliced floor."""
    return _bv.n_groups(tuple(cls), floor=hrr_phase(cls))


# ---------------------------------------------------------------------------
# The slices, as contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Slice:
    """One bra pair's slice: its index, `(a, b)` components, its sub-DAG,
    and the global `hgp.hrr.targets` index of its first ERI."""

    index: int
    a: int  # component index of a within its shell
    b: int
    first: int
    dag: _hrr.HrrDag

    @property
    def indices(self) -> range:
        """The global target indices of the slice's ERIs, in order."""
        return range(self.first, self.first + self.dag.n_eri)


@cache
def _slices(cls: AmClass) -> tuple[Slice, ...]:
    nb = len(cartesians(cls[1]))
    out, first = [], 0
    for s, dag in enumerate(_hrr.slices(cls)):
        out.append(Slice(s, s // nb, s % nb, first, dag))
        first += dag.n_eri
    return tuple(out)


def slices_of(ctx: FContext) -> tuple[Slice, ...]:
    """The slices of `ctx`'s class, each checked against `ctx.hdag`."""
    out = _slices(ctx.cls)
    assert sum(s.dag.n_eri for s in out) == ctx.hdag.n_eri
    return out


def slice_context(ctx: FContext, sl: Slice) -> FContext:
    """`ctx` with `hdag` the slice's sub-DAG (so the `hrr` hook builds that
    slice only) and `extra[NAME] = sl`."""
    return dataclasses.replace(ctx, hdag=sl.dag,
                               extra={**ctx.extra, NAME: sl})


# ---------------------------------------------------------------------------
# One slice's body
# ---------------------------------------------------------------------------


def slice_tap(sub: FContext) -> tuple[Stmt, ...]:
    """The slice's ERIs into the tap `g`, at their global indices."""
    if not sub.eris:
        return ()
    sl: Slice = sub.extra[NAME]
    return tuple(
        Assign(Index(OUT, (sub.bra, sub.k, Lit(i))), sub.hooks.eri(sub, t))
        for i, t in zip(sl.indices, sl.dag.targets, strict=True))


def slice_predigest(sub: FContext) -> tuple[Stmt, ...]:
    """The slice's ERIs into the buffers, through `predigest_eri`."""
    sl: Slice = sub.extra[NAME]
    comps = cart_names(sub.cls)
    out: list[Stmt] = []
    for i, t in zip(sl.indices, sl.dag.targets, strict=True):
        out += sub.hooks.predigest_eri(sub, t, comps[i])
    return tuple(out)


def slice_body(sub: FContext) -> tuple[Stmt, ...]:
    """The default per-slice body: the slice's HRR (the `hrr` hook on the
    slice's context), its tap, its predigest."""
    return (*sub.hooks.hrr(sub), *slice_tap(sub), *slice_predigest(sub))


def contracted_with(body: Callable[[FContext], tuple[Stmt, ...]]):
    """The `contracted` hook with `body(sub)` as each slice's statements
    (`slice_body` by default; #359 passes its interleaved one)."""

    def hook(ctx: FContext) -> tuple[Stmt, ...]:
        h = ctx.hooks
        sls = slices_of(ctx)
        out: list[Stmt] = [
            *h.digest_setup(ctx), *h.zero_buffers(ctx),
            Note(f"sliced HRR (#356): {len(sls)} bra-pair slices, each the "
                 "HRR of (a b | * *) only, read from the staged boundary, "
                 "then its ERIs digested; one slice's interior live at a "
                 "time"),
        ]
        for sl in sls:
            sub = slice_context(ctx, sl)
            out += [
                Note(f"slice {sl.index + 1} of {len(sls)}: bra pair "
                     f"({sl.a}, {sl.b}), {sl.dag.n_eri} ERIs, "
                     f"{sl.dag.n_statements} HRR nodes"),
                For(SLICE, Lit(sl.index), Lit(sl.index + 1), Lit(1),
                    tuple(body(sub))),
            ]
        out += h.flush_ket(ctx)
        return tuple(out)

    return hook


contracted = contracted_with(slice_body)


# ---------------------------------------------------------------------------
# The lever
# ---------------------------------------------------------------------------


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """Slice the HRR and re-cut #350's groups at the sliced floor. Refused
    without `batched-vrr` (`REFUSAL`), or when a lever before it moved a
    hook the slices bypass."""
    if REQUIRES not in spec.kernel_body_levers():
        raise ValueError(f"{spec.key()}: {REFUSAL}")
    if hooks.is_default("boundary_read"):
        raise ValueError(f"{spec.key()}: {NAME} applied before {REQUIRES} "
                         "staged the boundary (hooks_for applies key order)")
    moved = [h for h in _BYPASSED if not hooks.is_default(h)]
    if moved:
        raise ValueError(f"{spec.key()}: {NAME} replaces {moved} by its "
                         "slices; a lever that moved them first would be "
                         "dropped silently")
    cls = tuple(spec.am)
    return dataclasses.replace(hooks, contracted=contracted,
                               nest=_bv.nest_at(n_groups(cls)))
