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
- THE SEED ORDER. Each of plan/recovery.py's 26 structured `seed_orders()`
  (#14's family: `hermite_targets` and its reverse, and every sort by |r|
  then the components), imported, not redefined.

So at most 105 x 26 = 2730 full plans per cell; the whole board is a few
minutes serial (test_plan_search.py's one `slow` test).

WHAT IS NOT SEARCHED. Per-direction cuts (plan/recovery.py's cut is one; it
is the paper's recovered INPUT, judged on the literal board, not a choice
this board makes), seed orders outside the structured family (the factorial
space; #14's record searched random ones and stopped there on purpose), the
sieve's re-expansion choice and sweep, and `no-sieve`. Everything else is the
literal reconstruction, sieve on.

ONE PLAN PER CELL, chosen by ONE objective, stated before the board is read
and not tuned to it:

    0. FEASIBLE: max targets per batch <= the cell's minimax + 1
       (MINIMAX_TOLERANCE). Minimax is the paper's own cut criterion; the
       tolerance admits a cut one target off it.
    1. max intermediates       (Table 2)
    2. FLOPs                   (Table 1)

and among equal keys, the first found: cuts in `combinations` order (lowest
boundaries first), and within a cut the seed orders in `seed_orders()` order
(`hermite`, the literal one, first).

Why that order, on generator grounds and not table grounds: every target is
written to memory whatever the batch, so a target over minimax is cheap,
while `max_intermediates` is the per-batch live set -- the register-pressure
bound batching exists to cut ([JEngine, v2, §5]) and the quantity that
decides spills. FLOPs are what the sieve trims at the margin. The tolerance is
bounded, not dropped: without the minimax constraint the cheapest-intermediates
cut is a lopsided one that puts the targets in one batch.

What a cell motivated, said plainly (the issue requires it): the first
search of #13 put targets FIRST (tolerance 0), and its one large residual,
(15, 9) Table 2 at +7.50%, was an artefact of that -- the cut (6, 9) there
is one target off minimax and at or under in all three tables. The tolerance
was set to the smallest value that admits it (tolerance 2 picks identically;
3 breaks (9, 9) on Table 3), and the order of 1 and 2 above was NOT changed
to suit the board. What it costs, in the cells it does not help, is pinned in
test_plan_search.py: putting intermediates before FLOPs takes a FLOP-costlier
seed order in some cells, worst at (16, 1), +2.04% over Table 1 where the
FLOPs-first order would be +0.20%.

An EMPTY batch (Y_{0,0} and Z_{0,0} whenever the first cut is 1; Fig. 4
does not draw them at L = 6) is not a candidate: `partition` refuses it
(#64), so the cuts that make one are not enumerated. Excluding them moves no
recorded choice: none of searched_cuts.json's cuts starts at 1, and on the
L = 7..16 grid the minimax is reached by a cut that does not either (the
default cut, which takes the lowest optimal boundaries, never starts at 1
there), so neither the feasibility bound nor the winner can change.

THE RECORD. The chosen cut and seed order per (L, n_batches) are data, in
data/scoreboards/jengine/searched_cuts.json, so a replay (#15) builds the plan
from `searched_plan(L, n_batches, cuts, seed)` without searching again.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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
SEED_ORDERS: Mapping[str, SeedOrder] = dict(seed_orders())


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
            f"seed order {name!r} is not one of recovery.seed_orders()"
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
    batch, sieve seeded in the named `seed_orders()` order."""
    batches = partition(total_am, n_batches, cut=fixed_cut(cuts))
    return _sieved(total_am, (batch_dag(total_am, b) for b in batches), seed)


def objective(c: PlanCost) -> tuple[int, int]:
    """Among feasible plans (`feasible`): intermediates, then FLOPs; smaller
    is better."""
    return (c.max_intermediates, c.flops)


def feasible(c: PlanCost, minimax: int) -> bool:
    return c.max_targets <= minimax + MINIMAX_TOLERANCE


@dataclass(frozen=True)
class SearchResult:
    total_am: int
    n_batches: int
    cuts: Cuts
    seed: str
    cost: PlanCost
    n_searched: int


def all_plans(total_am: int, n_batches: int) -> Iterator[tuple[Cuts, str, PlanCost]]:
    """The whole search space, costed, in tie-break order: `candidate_cuts`
    outer, `seed_orders()` inner."""
    for cuts in candidate_cuts(total_am, n_batches):
        batches = partition(total_am, n_batches, cut=fixed_cut(cuts))
        dags = tuple(batch_dag(total_am, b) for b in batches)
        for seed in SEED_ORDERS:
            yield cuts, seed, cost(_sieved(total_am, dags, seed))


def search(total_am: int, n_batches: int) -> SearchResult:
    """Exhaustive over `all_plans`. The minimax is the smallest max target
    count over the cuts (the seed order cannot move it: the sieve never
    deletes a target); the first feasible plan with the smallest `objective`
    wins."""
    plans = list(all_plans(total_am, n_batches))
    minimax = min(c.max_targets for _, _, c in plans)
    cuts, seed, best = min(
        (p for p in plans if feasible(p[2], minimax)), key=lambda p: objective(p[2])
    )
    return SearchResult(total_am, n_batches, cuts, seed, best, len(plans))


# -- the record ---------------------------------------------------------------

Choice = tuple[Cuts, str]  # (cut points, seed order name)


def dump_cuts(results: Mapping[tuple[int, int], SearchResult]) -> str:
    """searched_cuts.json: {"L/n_batches": {"cut": [...], "seed_order": name}},
    sorted by (L, nb), one cell per line."""
    cells = {
        f"{L}/{nb}": {"cut": list(results[L, nb].cuts),
                      "seed_order": results[L, nb].seed}
        for L, nb in sorted(results)
    }
    doc = {
        "schema": CUTS_SCHEMA,
        "about": (
            "The cut and sieve seed order plan/search.py chose per "
            "(L, n_batches) for the searched board: the shared cut points "
            "c_1 < ... < c_{k-1} that batching.partition takes, "
            "k = n_batches // 3, and the name of one of "
            "plan.recovery.seed_orders(). Replay with "
            "plan.search.searched_plan(L, n_batches, cut, seed_order); do not "
            "re-search."
        ),
        "objective": {
            "feasible": "max_targets <= minimax + minimax_tolerance",
            "minimax_tolerance": MINIMAX_TOLERANCE,
            "then": ["max_intermediates", "flops"],
            "ties": "lowest cut points, then seed_orders() order",
        },
        "cuts": cells,
    }
    lines = [f'    "{key}": {json.dumps(v)}' for key, v in cells.items()]
    head = json.dumps({k: v for k, v in doc.items() if k != "cuts"}, indent=2)
    return head[:-2] + ',\n  "cuts": {\n' + ",\n".join(lines) + "\n  }\n}\n"


def load_cuts(path: Path) -> dict[tuple[int, int], Choice]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != CUTS_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, not {CUTS_SCHEMA}")
    out: dict[tuple[int, int], Choice] = {}
    for key, cell in doc["cuts"].items():
        L, nb = (int(x) for x in key.split("/"))
        _seed_order(cell["seed_order"])  # refuse a name no replay can build
        out[L, nb] = (tuple(cell["cut"]), cell["seed_order"])
    return out
