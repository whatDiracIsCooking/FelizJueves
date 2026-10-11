"""plan/search.py: the cut search behind the `searched` board.

Fast tests replay the recorded cuts (data/scoreboards/jengine/searched_cuts.json)
and check what the record must satisfy; the one `slow` test re-runs the whole
search and asserts it reproduces the record byte for byte. How the replayed
plans score on the searched board is pinned in test_scoreboards.py.
"""

from __future__ import annotations

from functools import cache
from math import comb
from pathlib import Path

import pytest

from felizjueves.literature.tables import load_table
from felizjueves.plan.batching import (
    MINIMAX_TARGETS,
    largest_batch,
    partition,
    target_counts,
)
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.search import (
    candidate_cuts,
    dump_cuts,
    fixed_cut,
    load_cuts,
    objective,
    search_cut,
    searched_plan,
)
from felizjueves.plan.sieve import plan
from felizjueves.spec import Operator, Spec, Target

DATA = Path(__file__).resolve().parents[1] / "data"
CUTS = DATA / "scoreboards" / "jengine" / "searched_cuts.json"
JENGINE = DATA / "literature" / "jengine"
TABLES = [load_table(JENGINE / f"table{n}.json") for n in (1, 2, 3)]
GRID = [(L, nb) for L in range(7, 17) for nb in (1, 3, 6, 9)]


@cache
def _recorded() -> dict[tuple[int, int], tuple[int, ...]]:
    return load_cuts(CUTS)


@cache
def _replayed(L: int, nb: int) -> PlanCost:
    return cost(searched_plan(L, nb, _recorded()[L, nb]))


@cache
def _literal(L: int, nb: int) -> PlanCost:
    return cost(plan(Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J,
                          n_batches=nb)))


def test_the_record_covers_exactly_the_table_grid() -> None:
    assert set(_recorded()) == set(GRID)
    for t in TABLES:
        assert set(t.cells) == set(GRID)


@pytest.mark.parametrize(("L", "nb"), GRID)
def test_each_recorded_cut_is_a_valid_minimax_cut(L: int, nb: int) -> None:
    cuts = _recorded()[L, nb]
    assert cuts in set(candidate_cuts(L, nb))
    partition(L, nb, cut=fixed_cut(cuts))  # raises on a malformed cut
    if nb > 3:
        counts = target_counts(L)
        best = largest_batch(counts, MINIMAX_TARGETS(counts, nb // 3))
        assert largest_batch(counts, cuts) == best
    else:
        assert cuts == ()


@pytest.mark.parametrize(("L", "nb"), GRID)
def test_the_searched_plan_is_never_worse_than_the_literal_one(L, nb) -> None:
    # The literal cut is one of the candidates, so the search can only match
    # or beat it under its own objective; with nothing to cut it IS literal.
    searched, literal = _replayed(L, nb), _literal(L, nb)
    assert objective(searched) <= objective(literal)
    if nb in (1, 3):
        assert searched == literal


def test_the_search_space_is_bounded_and_counted() -> None:
    for L, nb in GRID:
        k = 1 if nb == 1 else nb // 3
        assert sum(1 for _ in candidate_cuts(L, nb)) == comb(L, k - 1)
    assert max(comb(L, 2) for L in range(7, 17)) == 120


def test_an_empty_batch_is_kept_and_costs_nothing() -> None:
    # A first cut at 1 leaves Y_{0,0} and Z_{0,0} empty (as Fig. 4 at L = 6).
    dags = searched_plan(7, 6, (1,))
    empty = [d for d in dags if not d.targets]
    assert [d.batch for d in empty] == [("y", 0, 0), ("z", 0, 0)]
    assert all(not d.parents for d in empty)


def test_the_objective_picks_minimax_over_a_cut_that_would_pass_15_9() -> None:
    # The one cell where a cut exists that is at or under in all three tables
    # but the objective declines it: (6, 9) at L = 15 has 116 targets in its
    # largest batch, the paper's printed value, where minimax gets 115. The
    # paper's own cut there was not minimax (PR #18); ours is, by objective.
    off = cost(searched_plan(15, 9, (6, 9)))
    assert (off.flops, off.max_intermediates, off.max_targets) == (5637, 272, 116)
    printed = tuple(t.cells[15, 9] for t in TABLES)
    assert printed == (5786, 280, 116)
    chosen = _replayed(15, 9)
    assert chosen.max_targets == 115 < off.max_targets
    assert chosen.max_intermediates == 301 > printed[1]


def test_a_cut_file_with_the_wrong_schema_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "cuts.json"
    bad.write_text('{"schema": 2, "cuts": {}}')
    with pytest.raises(ValueError, match="schema"):
        load_cuts(bad)


@pytest.mark.slow
def test_the_search_reproduces_the_record() -> None:
    results = {(L, nb): search_cut(L, nb) for L, nb in GRID}
    assert dump_cuts(results) == CUTS.read_text()
    for (L, nb), r in results.items():
        assert r.cost == _replayed(L, nb)
