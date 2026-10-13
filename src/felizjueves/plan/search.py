"""The `searched` board's plans: search what the paper leaves open.

[JEngine, v2, §5] fixes WHAT the batch cut optimises -- minimax targets per
batch -- and leaves open WHICH cut is taken; [JEngine, v2, §4.1] leaves the
sieve's seed order open and says the authors did not optimise it. The literal
board fixes both arbitrarily (plan/batching.py: lowest minimax boundaries;
plan/sieve.py: `hermite_targets` order). This module searches both instead,
and scores the full plan, not the cut: a candidate is partition -> MILD DAG
per batch -> sieve -> PlanCost, so Tables 1 and 2 are the post-sieve numbers
the paper prints.

WHAT IS SEARCHED, per (L, n_batches), exhaustively -- bounded and
deterministic:

- THE CUT. Every contiguous cut `batching.partition` accepts: k - 1
  ascending cut points in 1..L, ONE set shared by X, Y and Z (the shape
  [JEngine, v2, Fig. 4] draws, and the only one `partition` builds), and
  leaving no batch empty (#64: a first cut at 1 empties Y_{0,0} and Z_{0,0},
  and `partition` refuses it). C(L, k - 1) - C(L - 1, k - 2) of them, at most
  C(16, 2) - 15 = 105. With 1 batch, or 3 (k = 1), the only cut is ().
- THE SEED ORDER. Each of plan/recovery.py's 58 `seed_orders(widened=True)`
  (#168): #14's 26 structured orders (`hermite_targets` and its reverse,
  and every sort by |r| then the components) first, unchanged and in the
  same order, then #108's 32 widened rules, imported, not redefined. Each is
  a rule over r and L, never a per-cell fit. Searching them here is a choice
  of this board only: #108's rules-only doctrine for the literal board's
  `exact_given_inputs` is untouched.

So at most 105 x 58 = 6090 full plans per cell; the whole board is about
8 minutes serial (test_plan_search.py's `slow` tests). Until #168 the
search ranged over the 26 alone, and its residual was read as unreachable
"independent of the objective"; that was a bound of the space searched, not
of the problem -- at (12, 1) a widened order lands on the printed cell
exactly.

WHAT IS NOT SEARCHED. Per-direction cuts (plan/recovery.py's cut is one; it
is the paper's recovered INPUT, judged on the literal board, not a choice
this board makes), seed orders outside the 58 rules (the factorial space;
#14's record searched random ones and stopped there on purpose), the
sieve's re-expansion choice and sweep, and `no-sieve`. Everything else is the
literal reconstruction, sieve on.

ONE PLAN PER CELL, chosen in two steps (#171): the FRONTIER of the feasible
plans, then ONE point of it by a SELECTION RULE stated in #171 before the
board was re-read, and not tuned to it. The rule, verbatim:

    Among feasible plans (max targets within `MINIMAX_TOLERANCE` of the
    cell's minimax), take the Pareto frontier over (max_intermediates,
    FLOPs) and select the point minimising `intermediates /
    intermediates_min + flops / flops_min`, intermediates breaking ties.
    Equal weight because the paper prints the two in separate tables and
    trades them against each other without naming a price; normalised
    because the two quantities differ by an order of magnitude and an
    unnormalised sum would be a FLOPs-only rule wearing a disguise.

Spelled out as the code runs it:

    0. FEASIBLE: max targets per batch <= the cell's minimax + 1
       (MINIMAX_TOLERANCE). Minimax is the paper's own cut criterion; the
       tolerance admits a cut one target off it.
    1. FRONTIER (`pareto`): the feasible plans whose point
       (max_intermediates, FLOPs) -- Tables 2 and 1 -- no other feasible
       plan's point weakly dominates; one plan per distinct point, the first
       found: cuts in `combinations` order (lowest boundaries first), and
       within a cut the seed orders in `SEED_ORDERS` order (`hermite`, the
       literal one, first). Ascending in intermediates, strictly descending
       in FLOPs -- the convention of #170's `frontier`, which
       test_plan_search.py checks this one against.
    2. SELECT (`select`): the frontier point with the smallest
       i / i_min + f / f_min, in exact rationals, with i_min and f_min the
       frontier's (equally, the feasible plans') fewest intermediates and
       fewest FLOPs; equal scores to the fewer intermediates.

The whole frontier is recorded beside the chosen point, so a reader with a
real exchange rate between the two quantities can re-select without
re-searching.

Why a frontier at all: neither lexicographic order works. Until #171 the
objective was intermediates first, then FLOPs (the per-batch live set is the
register-pressure bound batching exists to cut, [JEngine, v2, §5]); at
#168's cuts it left 9 cells of Table 1 over, and FLOPs first leaves 6 but
puts 2 back on Table 2. The tolerance is bounded, not dropped: without the
minimax constraint the cheapest-intermediates cut is a lopsided one that
puts the targets in one batch.

What a cell motivated, said plainly: the first search of #13 put targets
FIRST (tolerance 0), and its one large residual, (15, 9) Table 2 at +7.50%,
was an artefact of that -- the cut (6, 9) there is one target off minimax
and at or under in all three tables. The tolerance was set to the smallest
value that admits it (tolerance 2 picked identically under the old
objective; 3 broke (9, 9) on Table 3). The selection rule is #171's, quoted
above, and was not changed after the board was read. What it costs, in the
cells it does not help, is pinned in test_plan_search.py and limits.json:
against the lexicographic objective it moves one cell, (13, 1), onto the
printed values exactly; in (14, 1) and (14, 3) the frontier holds a point at
or under all three tables and the rule selects another, over Table 1.

An EMPTY batch (Y_{0,0} and Z_{0,0} whenever the first cut is 1; Fig. 4
does not draw them at L = 6) is not a candidate: `partition` refuses it
(#64), so the cuts that make one are not enumerated. Excluding them moves no
recorded choice: none of searched_cuts.json's cuts starts at 1, and on the
L = 7..16 grid the minimax is reached by a cut that does not either (the
default cut, which takes the lowest optimal boundaries, never starts at 1
there), so neither the feasibility bound nor the winner can change.

THE RECORD. The chosen cut and seed order per (L, n_batches) are data, in
data/scoreboards/jengine/searched_cuts.json, so a replay (#15) builds the plan
from `searched_plan(L, n_batches, cuts, seed)` without searching again. Each
cell also carries its frontier -- every point's cut, seed order and counts --
and the index of the selected point in it (#171).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from felizjueves.plan.batching import (
    Counts,
    CutRule,
    empty_batches,
    partition,
    target_counts,
)
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.dag import BatchDag, batch_dag
from felizjueves.plan.recovery import SeedOrder, seed_orders
from felizjueves.plan.sieve import sieve

Cuts = tuple[int, ...]
CUTS_SCHEMA = 2

# How many targets over the cell's minimax a chosen cut may hold. See above.
MINIMAX_TOLERANCE = 1

LITERAL_SEED = "hermite"
SEED_ORDERS: Mapping[str, SeedOrder] = dict(seed_orders(widened=True))

# #171's selection rule, verbatim from the issue, declared before the board
# was re-read. `select` is its code; the record quotes this text.
SELECTION_RULE = (
    "Among feasible plans (max targets within `MINIMAX_TOLERANCE` of the "
    "cell's minimax), take the Pareto frontier over (max_intermediates, "
    "FLOPs) and select the point minimising `intermediates / "
    "intermediates_min + flops / flops_min`, intermediates breaking ties. "
    "Equal weight because the paper prints the two in separate tables and "
    "trades them against each other without naming a price; normalised "
    "because the two quantities differ by an order of magnitude and an "
    "unnormalised sum would be a FLOPs-only rule wearing a disguise."
)


def fixed_cut(cuts: Cuts) -> CutRule:
    """A CutRule that ignores the counts and returns `cuts`. partition()
    validates them (k - 1 ascending points in 1..L, no batch empty)."""
    cuts = tuple(cuts)

    def rule(counts: Counts, k: int) -> Cuts:
        return cuts

    return rule


def candidate_cuts(total_am: int, n_batches: int) -> Iterator[Cuts]:
    """Every shared contiguous cut `partition` accepts -- none leaves a batch
    empty -- in lexicographic order; () when there is nothing to cut (1
    batch, or one sub-batch per direction)."""
    if n_batches == 1:
        yield ()
        return
    counts = target_counts(total_am)
    for cuts in combinations(range(1, total_am + 1), n_batches // 3 - 1):
        if not empty_batches(counts, cuts):
            yield cuts


def _seed_order(name: str) -> SeedOrder:
    try:
        return SEED_ORDERS[name]
    except KeyError:
        raise ValueError(
            f"seed order {name!r} is not one of recovery.seed_orders(widened=True)"
        ) from None


def _sieved(
    total_am: int, dags: Iterator[BatchDag] | tuple[BatchDag, ...], seed: str
) -> tuple[BatchDag, ...]:
    order = _seed_order(seed)(total_am)
    return tuple(sieve(d, seed_order=order) for d in dags)


def searched_plan(
    total_am: int, n_batches: int, cuts: Cuts, seed: str = LITERAL_SEED
) -> tuple[BatchDag, ...]:
    """The full plan for one cut and seed order: partition, MILD DAG per
    batch, sieve seeded in the named `seed_orders(widened=True)` order."""
    batches = partition(total_am, n_batches, cut=fixed_cut(cuts))
    return _sieved(total_am, (batch_dag(total_am, b) for b in batches), seed)


Point = tuple[int, int]  # (max_intermediates, FLOPs), both minimised
Plan = tuple[Cuts, str, PlanCost]  # (cut points, seed order name, its cost)


def point(c: PlanCost) -> Point:
    """The pair the frontier is taken over: (max_intermediates, FLOPs)."""
    return (c.max_intermediates, c.flops)


def feasible(c: PlanCost, minimax: int) -> bool:
    return c.max_targets <= minimax + MINIMAX_TOLERANCE


def pareto(plans: Iterator[Plan] | tuple[Plan, ...] | list[Plan]) -> tuple[Plan, ...]:
    """The plans whose `point` no other plan's point weakly dominates, one per
    distinct point (the first met, so the caller's order is the tie-break),
    ascending in intermediates and strictly descending in FLOPs."""
    first: dict[Point, Plan] = {}
    for p in plans:
        first.setdefault(point(p[2]), p)
    out: list[Plan] = []
    best_flops: int | None = None
    for pt in sorted(first):
        if best_flops is None or pt[1] < best_flops:
            out.append(first[pt])
            best_flops = pt[1]
    return tuple(out)


def reference(points: Iterator[Point] | tuple[Point, ...] | list[Point]) -> Point:
    """(intermediates_min, flops_min) over a frontier's points: the rule's
    normalisers. Over a frontier they equal the feasible plans' minima."""
    pts = list(points)
    if not pts:
        raise ValueError("an empty frontier has no reference point")
    return (min(p[0] for p in pts), min(p[1] for p in pts))


def score(p: Point, ref: Point) -> Fraction:
    """#171's selection score, in exact rationals:
    intermediates / intermediates_min + flops / flops_min."""
    return Fraction(p[0], ref[0]) + Fraction(p[1], ref[1])


def select(points: tuple[Point, ...] | list[Point]) -> int:
    """The index of the frontier point #171's rule selects: the smallest
    `score`, equal scores to the fewer intermediates."""
    ref = reference(points)
    return min(range(len(points)),
               key=lambda k: (score(points[k], ref), points[k][0]))


@dataclass(frozen=True)
class SearchResult:
    total_am: int
    n_batches: int
    cuts: Cuts
    seed: str
    cost: PlanCost
    n_searched: int
    frontier: tuple[Plan, ...]  # the feasible plans' Pareto set, see `pareto`
    selected: int  # the index of (cuts, seed) in `frontier`


def all_plans(total_am: int, n_batches: int) -> Iterator[Plan]:
    """The whole search space, costed, in tie-break order: `candidate_cuts`
    outer, `seed_orders(widened=True)` inner."""
    for cuts in candidate_cuts(total_am, n_batches):
        batches = partition(total_am, n_batches, cut=fixed_cut(cuts))
        dags = tuple(batch_dag(total_am, b) for b in batches)
        for seed in SEED_ORDERS:
            yield cuts, seed, cost(_sieved(total_am, dags, seed))


def search(total_am: int, n_batches: int) -> SearchResult:
    """Exhaustive over `all_plans`. The minimax is the smallest max target
    count over the cuts (the seed order cannot move it: the sieve never
    deletes a target); the frontier is taken over the feasible plans, and
    `select` picks one point of it."""
    plans = list(all_plans(total_am, n_batches))
    minimax = min(c.max_targets for _, _, c in plans)
    front = pareto(p for p in plans if feasible(p[2], minimax))
    k = select([point(c) for _, _, c in front])
    cuts, seed, best = front[k]
    return SearchResult(total_am, n_batches, cuts, seed, best, len(plans),
                        front, k)


# -- the record ---------------------------------------------------------------

Choice = tuple[Cuts, str]  # (cut points, seed order name)


@dataclass(frozen=True)
class FrontierPoint:
    """One recorded frontier point: its plan and the counts it replays to."""

    cuts: Cuts
    seed: str
    max_intermediates: int
    flops: int
    max_targets: int

    @property
    def point(self) -> Point:
        return (self.max_intermediates, self.flops)


def _point_json(cuts: Cuts, seed: str, c: PlanCost) -> str:
    return json.dumps({"cut": list(cuts), "seed_order": seed,
                       "max_intermediates": c.max_intermediates,
                       "flops": c.flops, "max_targets": c.max_targets})


def dump_cuts(results: Mapping[tuple[int, int], SearchResult]) -> str:
    """searched_cuts.json: per "L/n_batches", the chosen cut and seed order,
    the index of that plan in the cell's frontier, and the frontier, one
    point per line; sorted by (L, nb)."""
    doc = {
        "schema": CUTS_SCHEMA,
        "about": (
            "The cut and sieve seed order plan/search.py chose per "
            "(L, n_batches) for the searched board: the shared cut points "
            "c_1 < ... < c_{k-1} that batching.partition takes, "
            "k = n_batches // 3, and the name of one of "
            "plan.recovery.seed_orders(widened=True). Replay with "
            "plan.search.searched_plan(L, n_batches, cut, seed_order); do not "
            "re-search. Each cell also records its frontier -- the feasible "
            "plans' Pareto set over (max_intermediates, flops), ascending in "
            "max_intermediates, with each point's plan and counts -- and "
            "'selected', the index of the chosen plan in it, so a reader "
            "with another exchange rate between the two can re-select "
            "without re-searching (#171)."
        ),
        "objective": {
            "feasible": "max_targets <= minimax + minimax_tolerance",
            "minimax_tolerance": MINIMAX_TOLERANCE,
            "frontier": (
                "the feasible plans whose (max_intermediates, flops) no other "
                "feasible plan's weakly dominates, one plan per distinct point"
            ),
            "select": (
                "minimise max_intermediates / intermediates_min + "
                "flops / flops_min over the frontier (the minima are the "
                "frontier's), in exact rationals; equal scores to the fewer "
                "max_intermediates"
            ),
            "rule": SELECTION_RULE,
            "ties": (
                "one plan per point: the first in search order, lowest cut "
                "points, then seed_orders(widened=True) order"
            ),
        },
    }
    blocks = []
    for L, nb in sorted(results):
        r = results[L, nb]
        head = json.dumps({"cut": list(r.cuts), "seed_order": r.seed,
                           "selected": r.selected})
        pts = ",\n".join(f"      {_point_json(*p)}" for p in r.frontier)
        blocks.append(f'    "{L}/{nb}": {head[:-1]}, "frontier": [\n{pts}\n    ]}}')
    head = json.dumps(doc, indent=2)
    return head[:-2] + ',\n  "cuts": {\n' + ",\n".join(blocks) + "\n  }\n}\n"


def _read(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != CUTS_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, not {CUTS_SCHEMA}")
    return doc


def _frontier_of(key: str, cell: Mapping) -> tuple[FrontierPoint, ...]:
    pts = tuple(
        FrontierPoint(tuple(p["cut"]), p["seed_order"], p["max_intermediates"],
                      p["flops"], p["max_targets"])
        for p in cell["frontier"]
    )
    for p in pts:
        _seed_order(p.seed)
    k = cell["selected"]
    if not 0 <= k < len(pts) or (pts[k].cuts, pts[k].seed) != (
        tuple(cell["cut"]), cell["seed_order"]
    ):
        raise ValueError(f"cell {key}: 'selected' {k} is not the recorded "
                         f"cut and seed order's index in its frontier")
    return pts


def load_cuts(path: Path) -> dict[tuple[int, int], Choice]:
    """(cut, seed order) per cell. A cell that records a frontier must name
    its own plan as the frontier's selected point."""
    doc = _read(path)
    out: dict[tuple[int, int], Choice] = {}
    for key, cell in doc["cuts"].items():
        L, nb = (int(x) for x in key.split("/"))
        _seed_order(cell["seed_order"])  # refuse a name no replay can build
        if "frontier" in cell:
            _frontier_of(key, cell)
        out[L, nb] = (tuple(cell["cut"]), cell["seed_order"])
    return out


def load_frontiers(
    path: Path,
) -> dict[tuple[int, int], tuple[tuple[FrontierPoint, ...], int]]:
    """Per cell, the recorded frontier and the index of the selected point."""
    doc = _read(path)
    out = {}
    for key, cell in doc["cuts"].items():
        L, nb = (int(x) for x in key.split("/"))
        out[L, nb] = (_frontier_of(key, cell), cell["selected"])
    return out
