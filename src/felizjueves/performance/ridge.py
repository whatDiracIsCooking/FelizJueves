"""Does batching cross OUR ridge? (#114): the claim, declared, then scored.

THE RULE -- DECLARED BEFORE ANY ROOFLINE ROW EXISTED -- transcribed verbatim
from issue #114 (created 2026-10-11T18:07:53Z, unedited); rows from PR #134
merged 2026-10-12T04:27Z.

PROVENANCE, PLAINLY. This module was written AFTER the roofline rows existed
(data/measured/roofline.json, #112 / PR #134) and after that PR's body had
reported their summary. What predates the rows is the rule's TEXT, in the
issue's body, which GitHub records as created 2026-10-11T18:07:53Z and never
edited (`gh issue view 114 --json createdAt,updatedAt`: equal). The rule is
therefore not designed here; it is copied, and the code below adds only the
readings the issue's words need to become a function, each forced by a
definition this repo already had before the rows (listed under THE TERMS).
Nothing was chosen after reading a row.

THE RULE, VERBATIM (#114, "Scope"):

    declared before any roofline row is read: for one arch and one `L`,
    batching *crosses the ridge* iff the cell is memory-bound at
    `n_batches = 1` and compute-bound at some higher count in `SUBGRID`,
    with the crossing count recorded; the per-kernel variant ("every kernel
    of the partition is compute-bound") is declared separately, because the
    paper's `[7|7]` result distinguishes them;

    scored per card, with the two allowed to disagree;

THE TERMS, each as the repo defined it before #112's rows:

- MEMORY-BOUND / COMPUTE-BOUND is `device.verdict` against the card's
  MEASURED ridge (#109, data/measured/ceilings.json): memory-bound iff the
  intensity (fp64 FLOP / DRAM byte) is below the ridge, compute-bound at or
  past it. A card without both ceilings has no ridge and no verdict.
- THE CELL is one `measured_store.SUBGRID` cell, `pq_spec(L, n_batches=b)`,
  and its verdict is `measured_store.cell_roofline` (#112: every counter
  summed over the cell's batch kernels).
- "SOME HIGHER COUNT IN SUBGRID" is any SUBGRID count above 1 at that L
  (today 3, 6, 9, 12).
- THE CROSSING COUNT is the smallest such count at which the condition holds
  (the count batching first crosses at). The verdict at every count is
  recorded beside it, so a non-monotone sequence is seen, not hidden.
- THE PER-KERNEL VARIANT is the same rule with "compute-bound at some higher
  count" read as "every kernel of that count's partition compute-bound",
  each kernel's verdict being `measured_store.kernel_roofline`'s. At
  `n_batches = 1` the partition is one kernel whose counters ARE the cell's,
  so the "memory-bound at 1" half is the same reading in both variants.

THE ANSWERS, per (arch, L) and per variant -- the two variants are scored
apart and reported side by side, and neither rescues the other:

- CROSSED, at the crossing count: memory-bound at 1, and the variant's
  condition holds at some higher count.
- ALREADY: compute-bound already at one batch. The rule's antecedent fails,
  so batching does not cross: there is nothing to cross. That is a statement
  about the card's ridge, not about batching (#114, Scope: our cards' fp64
  ridge is far lower than an A100's).
- NEVER: memory-bound at 1, and at no higher count does the variant's
  condition hold.
- COULD NOT MEASURE: the card is in `measured_store.UNMEASURED` -- it has no
  counter rows by decision, with the reason recorded there. That is an absent
  answer, never NEVER and never a zero. (gfx1200 today: #112.)

A measured card missing a cell's rows raises (KeyError): an incomplete card
is not scored. Fig. 7's A100 ridge is that card's number; nothing here reads
it or compares against it.

Nothing under `plan/` or `literature/` may import this module
(test/test_layering.py): a measurement may never feed a literature floor; the
verdict is recorded by hand in claims.json / limits.json and recomputed by
the `claim`-marked test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from felizjueves.performance.device import ARCHS
from felizjueves.performance.measured_store import (
    SUBGRID,
    UNMEASURED,
    Measured,
    cell_roofline,
    kernel_roofline,
    split_kernel_key,
)

CELL, PER_KERNEL = "cell", "per-kernel"
VARIANTS = (CELL, PER_KERNEL)

CROSSED = "crossed"
ALREADY = "compute-bound already at 1 batch"
NEVER = "never crossed"
COULD_NOT_MEASURE = "could not measure"


@dataclass(frozen=True)
class Crossing:
    """The rule's answer at one (arch, L), for one variant."""

    arch: str
    L: int
    variant: str
    answer: str  # CROSSED / ALREADY / NEVER / COULD_NOT_MEASURE
    count: int | None  # the crossing count, iff CROSSED
    # (n_batches, compute-bound under the variant) at every SUBGRID count
    # of this L, ascending; empty when the card could not be measured
    bound: tuple[tuple[int, bool], ...]
    reason: str = ""  # why not, iff COULD_NOT_MEASURE


def counts(L: int) -> tuple[int, ...]:
    """The SUBGRID batch counts at L, ascending; ValueError if L is not in
    SUBGRID or its counts do not start at 1."""
    bs = tuple(sorted(s.n_batches for s in SUBGRID if s.total_am == L))
    if not bs or bs[0] != 1:
        raise ValueError(f"L={L}: SUBGRID has counts {bs}, not 1 and higher")
    return bs


def cell_key(store: Mapping[str, Measured], arch: str, L: int, b: int) -> str:
    """The counter store's one cell key for pq_spec(L, b) on `arch`.
    KeyError if it has none, or more than one (two toolchains)."""
    (spec,) = [s for s in SUBGRID if s.total_am == L and s.n_batches == b]
    prefix = f"{spec.key()}@{arch}/"
    got = sorted({c for c in (split_kernel_key(k)[0] for k in store)
                  if c.startswith(prefix)})
    if len(got) != 1:
        raise KeyError(f"L={L}, n_batches={b} @ {arch}: {len(got)} counter "
                       "cells, not one")
    return got[0]


def compute_bound(store: Mapping[str, Measured], arch: str, L: int, b: int,
                  variant: str) -> bool:
    """Whether pq_spec(L, b) on `arch` is compute-bound under `variant`:
    the cell's summed counters (CELL), or every kernel (PER_KERNEL)."""
    cell = cell_key(store, arch, L, b)
    if variant == CELL:
        return not cell_roofline(store, cell).memory_bound
    if variant == PER_KERNEL:
        return not any(p.memory_bound for p in kernel_roofline(store, cell))
    raise ValueError(f"variant {variant!r} is not one of {VARIANTS}")


def crossing(store: Mapping[str, Measured], arch: str, L: int,
             variant: str) -> Crossing:
    """The declared rule (module docstring) at one (arch, L), one variant."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} is not one of {VARIANTS}")
    if arch in UNMEASURED:
        return Crossing(arch, L, variant, COULD_NOT_MEASURE, None, (),
                        UNMEASURED[arch])
    bound = tuple((b, compute_bound(store, arch, L, b, variant))
                  for b in counts(L))
    if bound[0][1]:
        return Crossing(arch, L, variant, ALREADY, None, bound)
    higher = [b for b, ok in bound[1:] if ok]
    if higher:
        return Crossing(arch, L, variant, CROSSED, higher[0], bound)
    return Crossing(arch, L, variant, NEVER, None, bound)


def scored(store: Mapping[str, Measured],
           arches: Sequence[str] = tuple(a.name for a in ARCHS),
           ) -> dict[tuple[str, int, str], Crossing]:
    """Every (arch, L, variant) the rule scores: each card apart, each
    SUBGRID L, both variants."""
    Ls = sorted({s.total_am for s in SUBGRID})
    return {(a, L, v): crossing(store, a, L, v)
            for a in arches for L in Ls for v in VARIANTS}
