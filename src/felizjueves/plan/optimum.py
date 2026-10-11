"""The optimum as a computable object: the exact Pareto frontier of MD schemes.

[JEngine, v2, §4.1] puts MILD 1-2% behind a near-optimal tree search on FLOPs
and 2-6% ahead on intermediates. That search is not here, so this module
replaces it: it COMPUTES the optimum, for every L it can reach, so MILD's gap
can be measured directly (the follow-up issue records that gap; this module
only computes).

THE SEARCH SPACE. Every assignment of ONE reduction direction to each
materialised node [r]^(m), with a node shared between all its consumers --
the space MILD (plan/mild.py) picks one point of. One batch: the whole class,
targets `hermite_targets(L)`. NOT in the space: algebraic rewriting of the
recurrence, and recomputing a node to shorten a live range. Every result here
is "optimal" only over this space.

THE COST is plan/cost.py's, unchanged -- the calibrated instrument:

- FLOPs: `cost.node_flops(r_d)` per non-fundamental node, summed. The Boys
  function is free.
- intermediates: every node with m > 0, fundamentals [0]^(m) included
  (cost.py's convention 2).

So MILD's own one-batch plan, `cost.cost(dag.build(Spec(...)))`, is a point
in the space and is weakly dominated by the frontier this returns.

WHY AN EXACT DP IS POSSIBLE, three properties of plan/dag.py's graph:

- LAYERING. Every edge runs m -> m+1, so the state is the set of r required at
  one layer (a `Layer`), and a choice at layer m constrains only layer m+1.
  The cost of computing a layer down to the fundamentals does not depend on m,
  so the memo is keyed on the set alone.
- MONOTONICITY. If T is a subset of T', every point of T''s frontier is weakly
  dominated by one of T's (restrict T''s assignment to T). So inside one
  transition, a candidate next layer dominated by a SUBSET with no more FLOPs
  is dropped without risking the optimum (`_transitions`). The test checks
  this against an unpruned brute force.
- SYMMETRY. x, y and z are interchangeable in the recurrence and in
  `node_flops`, so a layer is memoised under its `canonical` image among the
  six axis permutations.

THE RESULT IS A FRONTIER, `((intermediates, flops), ...)`, ascending in the
first and strictly descending in the second: the two quantities the paper
trades against each other. One number would hide the trade.

THE CEILING, `EXACT_MAX_L`. `optimum(L)` REFUSES above it rather than running;
see `CEILING_REASON` for the measurement. The exact range is BELOW the
published grid (L = 7..16); it is the range §4.1's own claim is about (it puts
the onset of MILD's suboptimality at L = 5), and a record built on it must say
the tables were not scored by it.

ABOVE THE CEILING, `beam(L, width)`: the same layer-by-layer recurrence keeping
only the best `width` states per layer (and `width` candidates per
transition). Every point it returns is an ACHIEVABLE scheme, so its cost is an
UPPER bound on the optimum and therefore a LOWER bound on any heuristic's gap:
it can show MILD far from optimal, never close to it. Below the ceiling the
test checks it against `optimum`.

THIS IS NOT A DEFAULT. Nothing in `pipeline`, `search` or `recovery` calls it
(test_plan_optimum.py asserts no module imports it), and nothing it returns
reaches a literature floor.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import permutations

from felizjueves.plan.cost import node_flops
from felizjueves.plan.targets import R, hermite_targets

Layer = frozenset[R]  # the r required at one layer m
Point = tuple[int, int]  # (intermediates, flops)
Frontier = tuple[Point, ...]

ZERO: R = (0, 0, 0)
PERMUTATIONS: tuple[tuple[int, int, int], ...] = tuple(permutations(range(3)))

EXACT_MAX_L = 5
CEILING_REASON = (
    "the exact DP enumerates, per transition, the antichain of undominated "
    "candidate next layers (a next layer dominated by a subset with no more "
    "FLOPs is dropped), and out of the L = 6 target layer that antichain does "
    "not stay small: measured on this tree (#106, one core), all of L = 0..5 "
    "completes in about 2 s, while L = 6's FIRST transition held 20,517 "
    "undominated entries after 59 of the layer's 83 non-fundamental nodes, "
    "62 s in, still growing, and was killed at a 120 s bound with no result "
    "(the predecessor tree's jgen/optimum.py: past 18,000 a third of the way "
    "through, no result after twenty hours). Use beam(L, width) above the "
    "ceiling: an upper bound on the optimum, not the optimum"
)


class BeyondExactCeiling(ValueError):
    """`optimum(L)` with L > EXACT_MAX_L: refused, not attempted."""


def canonical(layer: Iterable[R]) -> Layer:
    """The representative of `layer` among its six axis-permutation images:
    the image whose SORTED CONTENTS are least. The `key=` is load-bearing:
    `min` over bare frozensets compares by subset inclusion, a partial order
    under which two distinct images of one size are incomparable, and `min`
    would quietly return whichever came first."""
    layer = tuple(layer)
    images = (frozenset(tuple(r[p] for p in perm) for r in layer)
              for perm in PERMUTATIONS)
    return min(images, key=lambda s: tuple(sorted(s)))


def pareto(points: Iterable[Point]) -> Frontier:
    """The undominated points, ascending in intermediates and strictly
    descending in FLOPs."""
    out: list[Point] = []
    for p in sorted(set(points)):
        if not out or p[1] < out[-1][1]:
            out.append(p)
    return tuple(out)


def choices(r: R) -> tuple[tuple[int, Layer], ...]:
    """Every way to reduce [r]^(m): per axis d with r_d >= 1, its FLOPs and the
    parents [r - 1d]^(m+1) and, when r_d >= 2, [r - 2d]^(m+1)."""
    out = []
    for i, c in enumerate(r):
        if c < 1:
            continue
        one = tuple(v - (k == i) for k, v in enumerate(r))
        ps = {one}
        if c >= 2:
            ps.add(tuple(v - 2 * (k == i) for k, v in enumerate(r)))
        out.append((node_flops(c), frozenset(ps)))
    return tuple(out)


def _rank(entry: tuple[Layer, int]) -> tuple[int, int, tuple[R, ...]]:
    t, f = entry
    return (len(t), f, tuple(sorted(t)))


def _transitions(
    layer: Layer, cap: int | None = None
) -> tuple[tuple[Layer, int], ...]:
    """The undominated (next layer, FLOPs at this layer) pairs reachable from
    `layer`, built one node at a time. An entry is dropped when another's next
    layer is a subset of its own with no more FLOPs (MONOTONICITY above).
    `cap` keeps only the best `cap` by (size, FLOPs) after each node: the
    beam's inner bound; None is exact."""
    entries: list[tuple[Layer, int]] = [(frozenset(), 0)]
    for r in sorted(r for r in layer if r != ZERO):
        best: dict[Layer, int] = {}
        for t, f in entries:
            for c, ps in choices(r):
                u = t | ps
                if best.get(u, f + c + 1) > f + c:
                    best[u] = f + c
        kept: list[tuple[Layer, int]] = []
        # Sorted by size then FLOPs, a dominator always precedes what it
        # dominates (equal size and subset means equal, and those are merged).
        for u, f in sorted(best.items(), key=_rank):
            if not any(g <= f and v <= u for v, g in kept):
                kept.append((u, f))
                if cap is not None and len(kept) == cap:
                    break
        entries = kept
    return tuple(entries)


def frontier_of(layer: Iterable[R]) -> Frontier:
    """The exact frontier for computing every [r] in `layer` at one layer m
    down to the fundamentals: intermediates counted from layer m + 1 on, FLOPs
    from layer m on. No ceiling: the caller owns the size of `layer`."""
    memo: dict[Layer, Frontier] = {}

    def solve(s: Layer) -> Frontier:
        key = canonical(s)
        if key in memo:
            return memo[key]
        if all(r == ZERO for r in key):
            out: Frontier = ((0, 0),)
        else:
            out = pareto(
                (len(t) + i, f + g)
                for t, f in _transitions(key)
                for i, g in solve(t)
            )
        memo[key] = out
        return out

    return solve(frozenset(layer))


def optimum(total_am: int) -> Frontier:
    """The exact (intermediates, FLOPs) frontier of the one-batch class
    L = total_am. Refuses above `EXACT_MAX_L` with the measured reason."""
    if type(total_am) is not int or total_am < 0:
        raise ValueError(f"total angular momentum {total_am!r} is not an int >= 0")
    if total_am > EXACT_MAX_L:
        raise BeyondExactCeiling(
            f"L = {total_am} is above EXACT_MAX_L = {EXACT_MAX_L}: "
            f"{CEILING_REASON}"
        )
    return frontier_of(hermite_targets(total_am))


def beam(total_am: int, width: int) -> Frontier:
    """An ACHIEVABLE frontier for the one-batch class L = total_am: the
    layer-by-layer recurrence keeping the best `width` states per layer,
    ranked by (accumulated FLOPs + intermediates, then FLOPs, then the next
    layer's size and contents), and `width` candidates per transition. Each
    point is a real scheme's cost, so each is weakly dominated by `optimum`'s
    frontier -- an upper bound on the optimum."""
    if type(width) is not int or width < 1:
        raise ValueError(f"beam width {width!r} is not an int >= 1")
    states: list[tuple[Layer, int, int]] = [
        (canonical(hermite_targets(total_am)), 0, 0)
    ]
    # Every transition lowers max |r| by exactly one, so after L steps every
    # state is {[0]}: all paths have the same length.
    for _ in range(total_am):
        cands: dict[tuple[Layer, int, int], None] = {}
        for s, inter, flops in states:
            for t, f in _transitions(s, cap=width):
                cands[(canonical(t), inter + len(t), flops + f)] = None
        states = sorted(
            cands,
            key=lambda c: (c[1] + c[2], c[2], len(c[0]), tuple(sorted(c[0]))),
        )[:width]
    return pareto((inter, flops) for _, inter, flops in states)


def dominated_by(points: Iterable[Point], frontier: Frontier) -> bool:
    """Every point is weakly dominated by some point of `frontier`."""
    return all(
        any(i <= p[0] and f <= p[1] for i, f in frontier) for p in points
    )
