"""plan/search.py: the cut and seed-order search behind the `searched` board.

Fast tests replay the recorded choices (data/scoreboards/jengine/
searched_cuts.json) and check what the record must satisfy; the one `slow`
test re-runs the whole search and asserts it reproduces the record byte for
byte. How the replayed plans score on the searched board is pinned in
test_scoreboards.py.
"""

from __future__ import annotations

import json
from functools import cache
from math import comb
from pathlib import Path

import pytest

from felizjueves.literature.limits import load_limits
from felizjueves.literature.tables import load_table
from felizjueves.plan.batching import (
    MINIMAX_TARGETS,
    largest_batch,
    partition,
    target_counts,
)
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.recovery import seed_orders
from felizjueves.plan.search import (
    LITERAL_SEED,
    MINIMAX_TOLERANCE,
    SEED_ORDERS,
    all_plans,
    candidate_cuts,
    dump_cuts,
    feasible,
    fixed_cut,
    load_cuts,
    objective,
    search,
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
def _recorded() -> dict[tuple[int, int], tuple[tuple[int, ...], str]]:
    return load_cuts(CUTS)


@cache
def _replayed(L: int, nb: int) -> PlanCost:
    return cost(searched_plan(L, nb, *_recorded()[L, nb]))


@cache
def _literal(L: int, nb: int) -> PlanCost:
    return cost(plan(Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J,
                          n_batches=nb)))


def _minimax(L: int, nb: int) -> int:
    """The smallest max-targets any cut gives: the literal plan's, whose cut
    is batching's minimax one (and at 1 batch the only one)."""
    return _literal(L, nb).max_targets


def test_the_record_covers_exactly_the_table_grid() -> None:
    assert set(_recorded()) == set(GRID)
    for t in TABLES:
        assert set(t.cells) == set(GRID)


def test_the_record_states_the_objective_the_code_runs() -> None:
    objective_ = json.loads(CUTS.read_text())["objective"]
    assert objective_["minimax_tolerance"] == MINIMAX_TOLERANCE == 1
    assert objective_["then"] == ["max_intermediates", "flops"]


@pytest.mark.parametrize(("L", "nb"), GRID)
def test_each_recorded_cut_is_within_the_minimax_tolerance(L: int, nb: int) -> None:
    cuts, seed = _recorded()[L, nb]
    assert cuts in set(candidate_cuts(L, nb))
    assert seed in SEED_ORDERS
    partition(L, nb, cut=fixed_cut(cuts))  # raises on a malformed cut
    if nb > 3:
        counts = target_counts(L)
        minimax = largest_batch(counts, MINIMAX_TARGETS(counts, nb // 3))
        assert minimax == _minimax(L, nb)
        assert largest_batch(counts, cuts) <= minimax + MINIMAX_TOLERANCE
    else:
        assert cuts == ()
    assert feasible(_replayed(L, nb), _minimax(L, nb))


@pytest.mark.parametrize(("L", "nb"), GRID)
def test_the_searched_plan_is_never_worse_than_the_literal_one(L, nb) -> None:
    # The literal plan (lowest minimax cut, hermite order) is one of the
    # feasible candidates, so the search can only match or beat it under its
    # own objective.
    searched, literal = _replayed(L, nb), _literal(L, nb)
    assert feasible(literal, _minimax(L, nb))
    assert objective(searched) <= objective(literal)


def test_the_literal_plan_is_the_hermite_seed_of_the_literal_cut() -> None:
    assert SEED_ORDERS[LITERAL_SEED] is next(iter(seed_orders()))[1]
    for L, nb in ((9, 1), (12, 6)):
        cut = MINIMAX_TARGETS(target_counts(L), max(1, nb // 3))
        assert cost(searched_plan(L, nb, cut)) == _literal(L, nb)


def test_the_search_space_is_bounded_and_counted() -> None:
    # Every cut but those whose first point is 1 (#64: they empty Y_{0,0} and
    # Z_{0,0}, and partition refuses them): C(L, k - 1) - C(L - 1, k - 2).
    for L, nb in GRID:
        k = 1 if nb == 1 else nb // 3
        expected = 1 if k == 1 else comb(L, k - 1) - comb(L - 1, k - 2)
        got = list(candidate_cuts(L, nb))
        assert len(got) == expected
        assert all(not c or c[0] > 1 for c in got)
    assert len(SEED_ORDERS) == 26
    biggest = max(comb(L, 2) - comb(L - 1, 1) for L in range(7, 17))
    assert biggest * len(SEED_ORDERS) == 2730


def test_a_cut_that_empties_a_batch_is_refused() -> None:
    # A first cut at 1 leaves Y_{0,0} and Z_{0,0} empty ([JEngine, v2, Fig. 4]
    # does not draw them at L = 6); #64 refuses it at the partition, so it is
    # neither a search candidate nor a replayable plan.
    assert (1,) not in set(candidate_cuts(7, 6))
    with pytest.raises(ValueError, match=r"Y_\{0,0\}, Z_\{0,0\}"):
        searched_plan(7, 6, (1,))


def test_an_unknown_seed_order_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed order 'nope'"):
        searched_plan(7, 1, (), "nope")
    bad = tmp_path / "cuts.json"
    bad.write_text('{"schema": 2, "cuts": {"7/1": {"cut": [], '
                   '"seed_order": "nope"}}}')
    with pytest.raises(ValueError, match="seed order 'nope'"):
        load_cuts(bad)


def test_the_tolerance_takes_the_one_target_off_cut_at_15_9() -> None:
    # The cell that motivated MINIMAX_TOLERANCE: (6, 9) at L = 15 holds 116
    # targets in its largest batch, the paper's printed value, where minimax
    # holds 115. With the tolerance the search takes it, and it is at or
    # under in all three tables; at tolerance 0, Table 2 was +7.50% here.
    printed = tuple(t.cells[15, 9] for t in TABLES)
    assert printed == (5786, 280, 116)
    assert _minimax(15, 9) == 115
    assert _recorded()[15, 9][0] == (6, 9)
    chosen = _replayed(15, 9)
    assert (chosen.flops, chosen.max_intermediates, chosen.max_targets) == (
        5636, 272, 116
    )


def test_what_intermediates_before_flops_costs_at_16_1() -> None:
    # The objective's price, stated: at (16, 1) it takes the seed order with
    # the fewest intermediates (1637), which costs FLOPs -- +2.04% over the
    # printed 5633, where the literal hermite order is +0.20% at 1639
    # intermediates. Both are under the printed 1647 intermediates, and both
    # are over Table 1: the objective moves this cell's margin, not its FAIL.
    hermite = cost(searched_plan(16, 1, (), LITERAL_SEED))
    chosen = _replayed(16, 1)
    assert (hermite.flops, hermite.max_intermediates) == (5644, 1639)
    assert (chosen.flops, chosen.max_intermediates) == (5748, 1637)
    assert TABLES[0].cells[16, 1] == 5633 and TABLES[1].cells[16, 1] == 1647


def test_a_cut_file_with_the_wrong_schema_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "cuts.json"
    bad.write_text('{"schema": 1, "cuts": {}}')
    with pytest.raises(ValueError, match="schema"):
        load_cuts(bad)


@pytest.mark.slow
def test_the_search_reproduces_the_record() -> None:
    # ~4 minutes serial: 40 cells x up to 105 cuts x 26 seed orders.
    results = {(L, nb): search(L, nb) for L, nb in GRID}
    assert dump_cuts(results) == CUTS.read_text()
    for (L, nb), r in results.items():
        assert r.cost == _replayed(L, nb)


@pytest.mark.slow
def test_no_plan_in_the_space_is_at_or_under_all_three_in_a_residual_cell() -> None:
    # limits.json's "unreachable, independent of the objective": in every
    # cell the searched board leaves over, no plan in the whole space -- any
    # cut, any seed order, any target count -- is at or under all three
    # tables at once. (test_scoreboards.py checks the cheap 1- and 3-batch
    # cells on the fast tier.)
    lim = next(
        lim for lim in load_limits(JENGINE / "limits.json")
        if lim.asserted_value["board"] == "searched"
    )
    residual = lim.asserted_value["residual_percent_over"]
    cells = {(L, nb) for t in residual.values() for L, nb, _ in t}
    for L, nb in sorted(cells):
        printed = [t.cells[L, nb] for t in TABLES]
        for cuts, seed, c in all_plans(L, nb):
            got = (c.flops, c.max_intermediates, c.max_targets)
            assert any(g > p for g, p in zip(got, printed, strict=True)), (
                L, nb, cuts, seed)
