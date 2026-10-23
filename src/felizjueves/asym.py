"""The asymmetric split, measured and refused (#113): a STUDY, not a lever.

`[JEngine, v2, §5]` says splits that break the three-way symmetry -- a batch
count that is neither 1 nor a multiple of three -- are "unnecessary and
worse". The repo asserts the first half as code: `spec.is_batch_count` and
`plan.batching.partition` REFUSE such a count, and that refusal is the witness
of the `asserted` claim `jengine/batch-count-multiple-of-3`
(test_plan_batching.py::test_an_illegal_batch_count_raises). This module
measures the second half WITHOUT loosening either: it constructs the
asymmetric batch list itself, checks it is a partition itself (coverage and
disjointness, since `partition`'s validation is bypassed), builds each batch's
kernel through the SAME builder the literal arm uses, and keys its rows as
their own arm, `.../b<n>/j/asym@<arch>/<toolchain>` -- #55's lever-arm shape
-- so every literal row, and every reader of one (`kernelrank`,
`batchreport`, `nine_vs_twelve`), reads exactly what it read before. An
`AsymCell` is not a `Spec` and cannot become one: nothing outside this study
can ask the generator for an asymmetric split.

THE STUDY'S RULE -- DECLARED BEFORE ANY ASYMMETRIC ROW EXISTED, and not to be
tuned after reading one.

- THE CELLS are `ASYM_CELLS`: L in {13, 16} x n_batches in {4, 5, 7, 8}, on
  both cards. Why these and no more: two arms per cell doubles the compile
  cost of the slow tier, so the set is small on purpose. L = 13 and 16 are
  where splitting pays at all on our cards (#56: wide-T picks one batch
  everywhere; near-T splits from L = 13, and the cards disagree at L = 16),
  so "which split" is a live question only there -- at L = 7 and 10 any split
  loses to one batch and an asymmetric loss would say nothing. 4 and 5 sit
  between the legal 3 and 6, 7 and 8 between 6 and 9: the two steps where
  the best legal count lives at these L, each bracketed from both sides.
- THE SPLIT of n is the best asymmetric split by the paper's own criterion,
  so the claim is tested against its strongest asymmetric rival, not a
  strawman. `direction_counts(L, n)`: (kx, ky, kz) sub-batches per direction,
  each >= 1, summing to n, MINIMISING the largest batch's target count (§5's
  minimax) and, among those, the next largest, and so on (the batch sizes
  sorted descending, compared lexicographically -- minimax alone is
  indifferent once an unsplit direction holds the largest batch, and would
  let n = 5 pile three sub-batches onto x); remaining ties to the
  lexicographically LARGEST (kx, ky, kz), i.e. extra sub-batches to x first
  (x holds (0, 0, 0) and the all-equal targets, so it is the largest
  direction). Within a direction, `direction_cuts`: the cut
  points minimising that direction's largest batch over its OWN counts,
  lowest boundaries on a tie, no empty batch -- `batching.MINIMAX_TARGETS`'s
  rule on one direction, since a cut cannot be shared when the directions
  have different k. For (k, k, k) this reproduces `partition(L, 3k)` at
  every L of the study (test/test_measured.py), so the asymmetric arm differs
  from the literal one by the count alone.
- THE NEIGHBOURS are the nearest legal counts, lo = 3 floor(n / 3) and hi =
  lo + 3 (3 and 6 for n = 4, 5; 6 and 9 for n = 7, 8): literal SUBGRID cells.
- THE CONTROL IS TIMED IN THE SAME SESSION, as #55's: the wall-clock sweep,
  asked for this study (`FJ_WALLCLOCK_ASYM`), compiles and checks the
  asymmetric arm and BOTH neighbours and times them back to back -- near-T
  lo, asym, hi; wide-T hi, asym, lo, so no arm always runs first or last --
  writing the asymmetric row with each neighbour's same-session statistics
  in its `raw` as `wallclock:control:b<lo>:*` / `wallclock:control:b<hi>:*`
  (and `...:near:*`). The literal rows are NOT re-timed or rewritten.
- PER INPUT (wide-T and near-T, judged apart): BEST is the neighbour with
  the lower same-session median on that input (a tie goes to lo). With b
  that neighbour and a the asymmetric arm, the asymmetric split is FASTER
  iff median(b) - median(a) > max(IQR(b), IQR(a)), SLOWER iff median(a) -
  median(b) > that same max, else NEUTRAL -- #60's rule, symmetric, as
  `measured_store.lever_ab` applies it.
- PER CELL: WIN if FASTER on either input (a win on EITHER input counts
  against the claim, as in #60); WORSE if SLOWER on at least one input and
  FASTER on none; NEUTRAL otherwise.
- PER ARCH: WORSE iff every cell is WORSE; CONTRADICTED iff any cell WINS;
  otherwise NOT SHOWN (no asymmetric split beats the symmetric family --
  "unnecessary" holds -- but "worse" is not shown everywhere). The two
  arches are judged apart and may disagree.
- THE CLAIM `jengine/symmetry-breaking-splits-are-worse` is `asserted` iff
  the verdict is WORSE on BOTH arches. Anything else -- an asymmetric split
  winning somewhere, or merely tying -- is the result, not a bug: the claim
  becomes a LIMIT (data/literature/jengine/limits.json) with the per-cell
  verdicts attached, per arch.
- Registers and spills (the asymmetric row against each neighbour's literal
  row, same compiler) are REPORTED beside the verdict and never enter it:
  the clock is the verdict. A spill is summed over a cell's batch kernels and
  is a mechanism; "worse" is a statement about the time to produce `J_p`.

Nothing under `plan/` or `literature/` may import this module
(test/test_layering.py): a measurement may never feed a literature floor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import combinations

from felizjueves.device import Arch
from felizjueves.ir import Kernel
from felizjueves.ir.passes import atomize
from felizjueves.measured_store import (
    FASTER,
    NEUTRAL,
    SLOWER,
    Measured,
    _sign,
    rows,
)
from felizjueves.mmd.kernel import _pq_kernel, pq_spec, split
from felizjueves.plan.batching import AXES, Batch, in_batch, target_counts
from felizjueves.plan.dag import batch_dag
from felizjueves.plan.sieve import sieve
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Spec, is_batch_count

ASYM = "asym"  # the arm's name, in the lever slot of its key

# THE declared cell set (module docstring, THE CELLS).
ASYM_LS = (13, 16)
ASYM_COUNTS = (4, 5, 7, 8)

WIN, WORSE = "win", "worse"
CONTRADICTED, NOT_SHOWN = "contradicted", "not shown"


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def _ranges(cuts: Sequence[int], total_am: int) -> list[tuple[int, int]]:
    edges = (0, *cuts, total_am + 1)
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def direction_cuts(counts: Sequence[int], k: int) -> tuple[int, ...]:
    """One direction's k - 1 cut points: minimax over its own counts, lowest
    boundaries on a tie, no empty sub-batch. Refuses a k with no such cut."""
    total_am = len(counts) - 1
    best: tuple[int, ...] | None = None
    best_size: int | None = None
    for cuts in combinations(range(1, total_am + 1), k - 1):
        sizes = [sum(counts[a:b + 1]) for a, b in _ranges(cuts, total_am)]
        if not all(sizes):
            continue
        if best_size is None or max(sizes) < best_size:
            best, best_size = cuts, max(sizes)
    if best is None:
        raise ValueError(f"no cut of {k} non-empty sub-batches at L = {total_am}")
    return best


def _sizes(total_am: int, ks: tuple[int, int, int]) -> tuple[int, ...] | None:
    """The split's batch sizes, sorted descending; None if it has no cut."""
    counts = target_counts(total_am)
    sizes: list[int] = []
    for d, k in zip(AXES, ks, strict=True):
        try:
            cuts = direction_cuts(counts[d], k)
        except ValueError:
            return None
        sizes += [sum(counts[d][a:b + 1]) for a, b in _ranges(cuts, total_am)]
    return tuple(sorted(sizes, reverse=True))


@cache
def direction_counts(total_am: int, n_batches: int) -> tuple[int, int, int]:
    """(kx, ky, kz) for an ASYMMETRIC n_batches (module docstring, THE
    SPLIT). Refuses a legal count: that split is the literal arm's."""
    if type(n_batches) is not int or n_batches < 4 or is_batch_count(n_batches):
        raise ValueError(f"n_batches {n_batches!r} is not an asymmetric count "
                         "(an int >= 4, neither 1 nor a multiple of 3)")
    best: tuple[int, int, int] | None = None
    best_size: tuple[int, ...] | None = None
    for kx in range(n_batches - 1, 0, -1):  # lexicographically largest first
        for ky in range(n_batches - kx - 1, 0, -1):
            ks = (kx, ky, n_batches - kx - ky)
            size = _sizes(total_am, ks)
            if size is not None and (best_size is None or size < best_size):
                best, best_size = ks, size
    if best is None:
        raise ValueError(f"L = {total_am} admits no {n_batches}-batch split")
    return best


def check_partition(total_am: int, batches: Sequence[Batch]) -> None:
    """Every target of the class in EXACTLY one batch, and no batch empty --
    `partition`'s guarantees, asserted here because it is bypassed."""
    seen: dict = {}
    for r in hermite_targets(total_am):
        owners = [b for b in batches if in_batch(r, *b)]
        if len(owners) != 1:
            raise ValueError(f"target {r} lies in {len(owners)} batches "
                             f"({owners}), not exactly one")
        seen.setdefault(owners[0], []).append(r)
    empty = [b for b in batches if b not in seen]
    if empty:
        raise ValueError(f"batches {empty} hold no target")


def split_batches(total_am: int, ks: tuple[int, int, int]) -> tuple[Batch, ...]:
    """The batches of (kx, ky, kz) at L = total_am: direction-major, each
    direction's ranges ascending from 0, as `partition` orders them. Any ks,
    symmetric included (the study's check against the literal partition)."""
    counts = target_counts(total_am)
    out = tuple((d, a, b)
                for d, k in zip(AXES, ks, strict=True)
                for a, b in _ranges(direction_cuts(counts[d], k), total_am))
    check_partition(total_am, out)
    return out


def split_kernels(spec: Spec, ks: tuple[int, int, int]) -> tuple[Kernel, ...]:
    """One kernel per batch of `split_batches(L, ks)` for the one-batch class
    `spec`: MILD DAG, sieve and `_pq_kernel` exactly as `build_pq_kernels`
    runs them, each `passes.atomize`d and named as it names a batch."""
    if spec.n_batches != 1 or spec.levers:
        raise ValueError(f"{spec.key()}: the class is a literal, one-batch Spec")
    lp, lq = split(spec)
    L = spec.total_am
    batches = split_batches(L, ks)
    # A symmetric ks is built byte-identically to build_pq_kernels (the
    # study's check that only the count differs); an asymmetric one says so.
    note = ("" if len(set(ks)) == 1
            else f" -- ASYMMETRIC split {ks} (felizjueves.asym, #113)")
    out = []
    for i, batch in enumerate(batches):
        direction, a, b = batch
        out.append(atomize(_pq_kernel(
            spec, sieve(batch_dag(L, batch)),
            f"pq_L{L}_p{lp}_q{lq}_{direction}{a}_{b}",
            f"batch {i + 1} of {len(batches)}, {direction.upper()}_{{{a},{b}}}"
            + note)))
    return tuple(out)


# ---------------------------------------------------------------------------
# The cell and its key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsymCell:
    """One asymmetric cell: the default class of total L, split into an
    asymmetric `n_batches`. Reads like a `Spec` where a row needs one
    (`total_am`, `n_batches`, `levers`, `key()`); is not one."""

    L: int
    n_batches: int

    def __post_init__(self) -> None:
        direction_counts(self.L, self.n_batches)  # refuses a legal count

    @property
    def total_am(self) -> int:
        return self.L

    @property
    def levers(self) -> tuple[str, ...]:
        return (ASYM,)

    @property
    def spec(self) -> Spec:
        """The class, one batch: what every batch kernel is built against."""
        return pq_spec(self.L)

    @property
    def ks(self) -> tuple[int, int, int]:
        return direction_counts(self.L, self.n_batches)

    @property
    def batches(self) -> tuple[Batch, ...]:
        return split_batches(self.L, self.ks)

    @property
    def neighbours(self) -> tuple[Spec, Spec]:
        """The nearest legal counts below and above, as literal Specs."""
        lo = 3 * (self.n_batches // 3)
        return pq_spec(self.L, n_batches=lo), pq_spec(self.L, n_batches=lo + 3)

    def key(self) -> str:
        """E.g. ``coulomb/d0/(7,0|6,0)/b4/j/asym``: a Spec key's shape, with
        the arm in the lever slot."""
        return self.spec.key().replace("/b1/", f"/b{self.n_batches}/") \
            .removesuffix("/literal") + f"/{ASYM}"

    def kernels(self) -> tuple[Kernel, ...]:
        return split_kernels(self.spec, self.ks)


ASYM_CELLS: tuple[AsymCell, ...] = tuple(
    AsymCell(L, n) for L in ASYM_LS for n in ASYM_COUNTS)


@dataclass(frozen=True)
class AsymKey:
    """What keys an asymmetric row: `MeasuredKey`'s shape, an `AsymCell` in
    the Spec's place (`measured_store.merge` reads `arch`, `toolchain` and
    `key()` and nothing else)."""

    spec: AsymCell
    arch: Arch
    toolchain: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AsymCell):
            raise ValueError(f"{self.spec!r} is not an AsymCell")
        if not isinstance(self.arch, Arch):
            raise ValueError(f"arch {self.arch!r} is not an Arch")
        if not isinstance(self.toolchain, str) or not self.toolchain:
            raise ValueError(f"toolchain {self.toolchain!r} must be non-empty")

    def key(self) -> str:
        return f"{self.spec.key()}@{self.arch.name}/{self.toolchain}"


def control_prefix(neighbour: Spec) -> str:
    """The raw prefix of a neighbour's same-session statistics."""
    return f"wallclock:control:b{neighbour.n_batches}:"


# ---------------------------------------------------------------------------
# The verdict (module docstring, THE STUDY'S RULE)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsymVsNeighbours:
    """One asymmetric cell against its two same-session neighbours, one arch."""

    arch: str
    L: int
    n_batches: int
    ks: tuple[int, int, int]
    neighbours: tuple[int, int]  # (lo, hi) batch counts
    # (median, IQR) per arm and input, in the order (lo, asym, hi)
    wide: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    near: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    # reported, never judged: (lo, asym, hi) from the compile rows
    spill: tuple[int, int, int]
    registers: tuple[int, int, int]

    @staticmethod
    def _against_best(t) -> tuple[int, str]:
        lo, asym, hi = t
        best = lo if lo[0] <= hi[0] else hi
        return (0 if best is lo else 1), _sign(best, asym)

    @property
    def wide_sign(self) -> str:
        return self._against_best(self.wide)[1]

    @property
    def near_sign(self) -> str:
        return self._against_best(self.near)[1]

    @property
    def wide_best(self) -> int:
        return self.neighbours[self._against_best(self.wide)[0]]

    @property
    def near_best(self) -> int:
        return self.neighbours[self._against_best(self.near)[0]]

    @property
    def verdict(self) -> str:
        signs = {self.wide_sign, self.near_sign}
        if FASTER in signs:
            return WIN
        if SLOWER in signs:
            return WORSE
        return NEUTRAL


def asym_vs_neighbours(store: Mapping[str, Measured], arch: str,
                       cell: AsymCell) -> AsymVsNeighbours:
    """The comparison for one cell on `arch`, from the store. KeyError when
    the asymmetric row is missing, uncompiled, untimed or has no
    same-session control, or a neighbour's compile row is missing."""

    def one(s: Spec | AsymCell) -> Measured:
        got = rows(spec=s, arch=arch, store=store)
        if len(got) != 1:
            raise KeyError(f"{s.key()} @ {arch}: {len(got)} rows")
        (r,) = got.values()
        if r.spill_store is None or r.spill_load is None or r.registers is None:
            raise KeyError(f"{s.key()} @ {arch}: not compiled")
        return r

    lo, hi = cell.neighbours
    r = one(cell)
    pl, ph = control_prefix(lo), control_prefix(hi)
    if r.runtime_ns is None or r.spread_ns is None or any(
            f"{p}median_ns" not in r.raw for p in (pl, ph)):
        raise KeyError(f"{cell.key()} @ {arch}: not timed against both "
                       "same-session neighbours")
    rlo, rhi = one(lo), one(hi)

    def stats(prefix: str) -> tuple[float, float]:
        return (r.raw[f"{prefix}median_ns"],
                r.raw[f"{prefix}q3_ns"] - r.raw[f"{prefix}q1_ns"])

    def spill(x: Measured) -> int:
        return x.spill_store + x.spill_load

    return AsymVsNeighbours(
        arch=arch, L=cell.L, n_batches=cell.n_batches, ks=cell.ks,
        neighbours=(lo.n_batches, hi.n_batches),
        wide=(stats(pl), (r.runtime_ns, r.spread_ns), stats(ph)),
        near=(stats(pl + "near:"), stats("wallclock:near:"),
              stats(ph + "near:")),
        spill=(spill(rlo), spill(r), spill(rhi)),
        registers=(rlo.registers, r.registers, rhi.registers))


def arch_verdict(store: Mapping[str, Measured], arch: str,
                 cells: Sequence[AsymCell] = ASYM_CELLS) -> str:
    """WORSE, CONTRADICTED or NOT SHOWN on one arch, over every cell."""
    got = [asym_vs_neighbours(store, arch, c).verdict for c in cells]
    if WIN in got:
        return CONTRADICTED
    if all(v == WORSE for v in got):
        return WORSE
    return NOT_SHOWN


def claim_holds(store: Mapping[str, Measured], arches: Sequence[str],
                cells: Sequence[AsymCell] = ASYM_CELLS) -> bool:
    """The claim is `asserted` iff the verdict is WORSE on every arch."""
    return all(arch_verdict(store, a, cells) == WORSE for a in arches)
