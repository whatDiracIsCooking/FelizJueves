"""The `searched` board's plans: search the cut the paper leaves open.

[JEngine, v2, §5] fixes WHAT the batch cut optimises -- minimax targets per
batch -- and leaves open WHICH of the optimal cuts is taken. The literal board
fixes that arbitrarily (plan/batching.py: lowest boundaries). This module
searches it instead, and scores the full plan, not the cut: a candidate is
partition -> MILD DAG per batch -> sieve -> PlanCost, so Tables 1 and 2 are
the post-sieve numbers the paper prints.

WHAT IS SEARCHED. Every contiguous cut `batching.partition` accepts: k - 1
ascending cut points in 1..L, ONE set shared by X, Y and Z (the shape
[JEngine, v2, Fig. 4] draws, and the only one `partition` builds). That is
C(L, k - 1) candidates per cell -- at most C(16, 2) = 120 -- so the search is
exhaustive, bounded and deterministic. With 1 batch, or 3 (k = 1), there is
no cut, and the single candidate is the literal plan.

WHAT IS NOT SEARCHED. The sieve's seed order (sieve.py fixes it to
`hermite_targets` order and exposes no parameter; #14 owns that seam), the
sieve's re-expansion choice, per-direction cuts, and `no-sieve`. Everything
but the cut is the literal reconstruction, sieve on.

ONE PLAN PER CELL, chosen by ONE objective, lexicographic:

    1. max targets per batch   (Table 3; the paper's own minimax criterion)
    2. max intermediates       (Table 2; the register pressure batching exists
                                to cut, [JEngine, v2, §5])
    3. FLOPs                   (Table 1)

and among equal keys, the lexicographically smallest cut points (the first
found in `combinations` order). The objective never reads the printed tables,
so a cell at or under the paper is a property of the plan, not of the search
peeking at the answer. Putting targets first keeps every chosen cut minimax:
the search only ever picks among the cuts the paper's rule allows.

An EMPTY batch (Y_{0,0} and Z_{0,0} whenever the first cut is 1; Fig. 4 has
them at L = 6) is kept, as the paper keeps it: it contributes no node and no
FLOP, and is not handed to the sieve, which seeds from a batch's targets.

THE RECORD. The chosen cut per (L, n_batches) is data, in
data/scoreboards/jengine/searched_cuts.json, so a replay (#15) builds the plan
from `searched_plan(L, n_batches, cuts)` without searching again.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from felizjueves.plan.batching import Counts, CutRule, partition
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.dag import BatchDag, batch_dag
from felizjueves.plan.sieve import sieve

Cuts = tuple[int, ...]
CUTS_SCHEMA = 1


def fixed_cut(cuts: Cuts) -> CutRule:
    """A CutRule that ignores the counts and returns `cuts`. partition()
    validates them (k - 1 ascending points in 1..L)."""
    cuts = tuple(cuts)

    def rule(counts: Counts, k: int) -> Cuts:
        return cuts

    return rule


def candidate_cuts(total_am: int, n_batches: int) -> Iterator[Cuts]:
    """Every shared contiguous cut, in lexicographic order; () when there is
    nothing to cut (1 batch, or one sub-batch per direction)."""
    k = 1 if n_batches == 1 else n_batches // 3
    yield from combinations(range(1, total_am + 1), k - 1)


def searched_plan(total_am: int, n_batches: int, cuts: Cuts) -> tuple[BatchDag, ...]:
    """The full plan for one cut: partition, MILD DAG per batch, sieve."""
    batches = partition(total_am, n_batches, cut=fixed_cut(cuts))
    dags = (batch_dag(total_am, b) for b in batches)
    return tuple(sieve(d) if d.targets else d for d in dags)


def objective(c: PlanCost) -> tuple[int, int, int]:
    """Targets, then intermediates, then FLOPs; smaller is better."""
    return (c.max_targets, c.max_intermediates, c.flops)


@dataclass(frozen=True)
class SearchResult:
    total_am: int
    n_batches: int
    cuts: Cuts
    cost: PlanCost
    n_searched: int


def search_cut(total_am: int, n_batches: int) -> SearchResult:
    """Exhaustive over `candidate_cuts`; the first cut with the smallest
    `objective` wins, so ties go to the lowest boundaries."""
    best: tuple[Cuts, PlanCost] | None = None
    n = 0
    for cuts in candidate_cuts(total_am, n_batches):
        n += 1
        c = cost(searched_plan(total_am, n_batches, cuts))
        if best is None or objective(c) < objective(best[1]):
            best = (cuts, c)
    assert best is not None  # candidate_cuts always yields at least ()
    return SearchResult(total_am, n_batches, best[0], best[1], n)


# -- the record ---------------------------------------------------------------


def dump_cuts(results: Mapping[tuple[int, int], SearchResult]) -> str:
    """searched_cuts.json: {"L/n_batches": [cut points]}, sorted by (L, nb)."""
    cells = {
        f"{L}/{nb}": list(results[L, nb].cuts) for L, nb in sorted(results)
    }
    doc = {
        "schema": CUTS_SCHEMA,
        "about": (
            "The cut plan/search.py chose per (L, n_batches) for the searched "
            "board: the shared cut points c_1 < ... < c_{k-1} that "
            "batching.partition takes, k = n_batches // 3. Replay with "
            "plan.search.searched_plan(L, n_batches, cuts); do not re-search."
        ),
        "objective": ["max_targets", "max_intermediates", "flops"],
        "cuts": cells,
    }
    lines = [f'    "{key}": {json.dumps(v)}' for key, v in cells.items()]
    head = json.dumps({k: v for k, v in doc.items() if k != "cuts"}, indent=2)
    return head[:-2] + ',\n  "cuts": {\n' + ",\n".join(lines) + "\n  }\n}\n"


def load_cuts(path: Path) -> dict[tuple[int, int], Cuts]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != CUTS_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, not {CUTS_SCHEMA}")
    out: dict[tuple[int, int], Cuts] = {}
    for key, cuts in doc["cuts"].items():
        L, nb = (int(x) for x in key.split("/"))
        out[L, nb] = tuple(cuts)
    return out
