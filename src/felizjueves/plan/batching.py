"""[JEngine, v2, Alg. 4]: which batch a target belongs to, and the partition.

THE PREDICATE. A target goes to the direction of its MAXIMUM component, and a
tie goes to the cyclic PREDECESSOR: {x,y} -> x, {y,z} -> y, {z,x} -> z, and
the all-equal targets (including (0, 0, 0)) -> x. Within a direction, a batch
`(direction, a, b)` holds the targets whose maximum component lies in a..b.
Transcribed from docs/implementation/j_engine.md ("The scheme"), which is in
turn Alg. 4 lines 1-10. This tie-break is the INVERSE of MILD's (successor,
all-equal -> z); the two differ by what looks like a typo.

THE PARTITION. `n_batches` is 1, or 3k: k sub-batches per direction. The a..b
ranges are OUTPUT, not input. [JEngine, v2, §5] states the rule: choose a and
b "such that the maximum number of target integrals per batch is minimized"
-- MINIMAX, not "equalise". The VALUE of that minimum is unique (Table 3
depends on nothing else); WHICH optimal cut is taken is a choice the paper
leaves open, and on the literal board it is fixed arbitrarily as:

- among optimal cuts, the LOWEST BOUNDARIES (lexicographically smallest cut
  points);
- ONE set of cut points SHARED by X, Y and Z, as [JEngine, v2, Fig. 4] draws
  them (X_{0,2}, X_{3,6}, Y_{1,2}, Y_{3,6}, Z_{1,2}, Z_{3,6} at L = 6; Y_{0,0}
  and Z_{0,0} are empty and not drawn).

Which optimal cut is taken does not move Table 3, but it will move Tables 1-2
(intermediates and FLOPs) once those are computed.

NO BATCH IS EMPTY. `partition` refuses a cut that leaves a batch with no
target, naming L, n_batches and the empty batches (#64), rather than hand the
sieve and the kernel builder a batch with nothing to contract. Only Y_{0,0}
and Z_{0,0} can be empty (see `empty_batches`), so a split is refused exactly
when its first cut is 1 -- or, at L = 0, whenever it splits at all. Which
(L, n_batches) the DEFAULT cut admits is therefore not monotone in either
argument: (5, 9) is refused while (6, 9) and (5, 12) plan, because each
minimax cut is chosen on its own counts. test_plan_batching.py pins the map.

The first range always starts at 0: (0, 0, 0) is a target, and its maximum
component is 0, so a range starting at 1 would drop it from every batch.

ONE BATCH is the whole class, not a direction. It is spelled with the
direction `ALL` ("xyz"), and `in_batch(r, ALL, 0, L)` holds for every target.

`cut` is a seam: a `CutRule` maps per-direction, per-magnitude target counts
and k to the cut points. Nobody passes one today; a search plugs in there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

from felizjueves.plan.targets import R, hermite_targets
from felizjueves.spec import is_batch_count

AXES = ("x", "y", "z")
ALL = "xyz"  # the unsplit class: one batch, every direction

Batch = tuple[str, int, int]  # direction, a..b on the max component

# counts[direction][m] = targets in that direction whose max component is m,
# for m = 0..L. A CutRule returns the k - 1 cut points c_1 < ... < c_{k-1} in
# 1..L; sub-batch i covers c_{i-1} .. c_i - 1, with c_0 = 0 and c_k = L + 1.
Counts = Mapping[str, Sequence[int]]
CutRule = Callable[[Counts, int], tuple[int, ...]]


def batch_direction(r: R) -> str:
    """Direction of r's maximum component; ties by cyclic predecessor."""
    if r[0] == r[1] == r[2]:
        return "x"  # three-way tie
    hi = max(r)
    for i in (0, 1, 2):
        if r[i] == hi and r[(i + 1) % 3] == hi:
            return AXES[i]
    return AXES[r.index(hi)]


def in_batch(r: R, direction: str, a: int, b: int) -> bool:
    if direction == ALL:
        return a <= max(r) <= b
    if direction not in AXES:
        raise ValueError(f"direction {direction!r} is not one of {AXES} or {ALL!r}")
    d = batch_direction(r)
    return d == direction and a <= r[AXES.index(d)] <= b


def target_counts(total_am: int) -> dict[str, list[int]]:
    """counts[direction][m]: the targets of that direction with max component m."""
    counts = {d: [0] * (total_am + 1) for d in AXES}
    for r in hermite_targets(total_am):
        counts[batch_direction(r)][max(r)] += 1
    return counts


def _ranges(cuts: Sequence[int], total_am: int) -> list[tuple[int, int]]:
    edges = (0, *cuts, total_am + 1)
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def largest_batch(counts: Counts, cuts: Sequence[int]) -> int:
    """The largest batch under these shared cut points."""
    total_am = len(counts["x"]) - 1
    return max(
        sum(counts[d][a : b + 1]) for d in AXES for a, b in _ranges(cuts, total_am)
    )


def empty_batches(counts: Counts, cuts: Sequence[int]) -> tuple[Batch, ...]:
    """The batches these shared cut points leave with NO target, in partition
    order. Only Y_{0,0} and Z_{0,0} can be: (0, 0, 0) is the one target with
    max component 0, and `batch_direction` sends it to x; every m >= 1 has a
    target in each direction ((m, 0, 0), (0, m, 0), (0, 0, m))."""
    total_am = len(counts["x"]) - 1
    return tuple(
        (d, a, b)
        for d in AXES
        for a, b in _ranges(cuts, total_am)
        if not sum(counts[d][a : b + 1])
    )


def MINIMAX_TARGETS(counts: Counts, k: int) -> tuple[int, ...]:
    """Cut points minimising the largest batch; ties to the lowest boundaries.

    Exhaustive over every contiguous cut, in lexicographic order, keeping only
    a strict improvement -- so among optimal cuts the lexicographically
    smallest wins. C(L, k - 1) candidates: 120 at L = 16, k = 3.
    """
    total_am = len(counts["x"]) - 1
    best: tuple[int, ...] = ()
    best_size: int | None = None
    for cuts in combinations(range(1, total_am + 1), k - 1):
        size = largest_batch(counts, cuts)
        if best_size is None or size < best_size:
            best, best_size = cuts, size
    return best


def partition(
    total_am: int, n_batches: int, *, cut: CutRule = MINIMAX_TARGETS
) -> tuple[Batch, ...]:
    """n_batches is 1, or 3k: k sub-batches per direction. The a..b ranges are
    OUTPUT, not input.

    Batches come out direction-major (X, then Y, then Z), each direction's
    ranges ascending from a = 0. The cut rule and its tie-break are in the
    module docstring.
    """
    if type(total_am) is not int or total_am < 0:
        raise ValueError(f"total angular momentum {total_am!r} is not an int >= 0")
    if not is_batch_count(n_batches):
        raise ValueError(
            f"n_batches {n_batches!r} is neither 1 nor a positive multiple of 3"
        )
    if n_batches == 1:
        return ((ALL, 0, total_am),)
    k = n_batches // 3
    if k > total_am + 1:
        raise ValueError(
            f"{k} sub-batches per direction, but the max component only takes "
            f"{total_am + 1} values at L = {total_am}"
        )
    cuts = tuple(cut(target_counts(total_am), k))
    if len(cuts) != k - 1 or cuts != tuple(sorted(set(cuts))) or (
        cuts and not 1 <= cuts[0] <= cuts[-1] <= total_am
    ):
        raise ValueError(
            f"cut rule returned {cuts!r}, not {k - 1} ascending cut points in 1..L"
        )
    empty = empty_batches(target_counts(total_am), cuts)
    if empty:
        raise ValueError(
            f"L = {total_am}, n_batches = {n_batches} (cut {cuts!r}) leaves "
            f"{len(empty)} batches with no targets: "
            + ", ".join(f"{d.upper()}_{{{a},{b}}}" for d, a, b in empty)
            + ". (0, 0, 0) is the only target with max component 0 and it "
            "belongs to x, so a y or z sub-batch covering only m = 0 is empty; "
            "an empty batch has no contraction to emit"
        )
    return tuple((d, a, b) for d in AXES for a, b in _ranges(cuts, total_am))


def batch_size(total_am: int, batch: Batch) -> int:
    return sum(1 for r in hermite_targets(total_am) if in_batch(r, *batch))


def max_targets_per_batch(total_am: int, n_batches: int) -> int:
    """[JEngine, v2, Table 3]'s quantity: the largest batch's target count.
    Targets only, so the Boys function is excluded by construction
    ([JEngine, v2, §7.1])."""
    return max(batch_size(total_am, b) for b in partition(total_am, n_batches))
