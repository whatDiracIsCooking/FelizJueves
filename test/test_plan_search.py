"""plan/search.py: the cut and seed-order search behind the `searched` board.

Fast tests replay the recorded choices (data/scoreboards/jengine/
searched_cuts.json) and check what the record must satisfy; the `slow`
tests range over the whole 58-order space (#168): one re-runs the search and
asserts it reproduces the record byte for byte, and two check limits.json's
split of the residual into cells unreachable in that space and cells where
#171's selection rule passes over a frontier point at or under all three
tables. How the replayed plans score on the searched board is pinned in
test_scoreboards.py.
"""

from __future__ import annotations

import json
from fractions import Fraction
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
from felizjueves.plan.order_search import frontier as order_search_frontier
from felizjueves.plan.recovery import seed_orders
from felizjueves.plan.search import (
    LITERAL_SEED,
    MINIMAX_TOLERANCE,
    SEED_ORDERS,
    SELECTION_RULE,
    FrontierPoint,
    all_plans,
    candidate_cuts,
    dump_cuts,
    feasible,
    fixed_cut,
    load_cuts,
    load_frontiers,
    pareto,
    point,
    reference,
    score,
    search,
    searched_plan,
    select,
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
def _frontiers() -> dict[tuple[int, int], tuple[tuple[FrontierPoint, ...], int]]:
    return load_frontiers(CUTS)


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


# #171's rule, as the issue body states it (line breaks aside). It was
# declared before the board was re-read; this pins that the code quotes it
# unchanged, so a change to the rule is a diff here.
ISSUE_171_RULE = (
    "Among feasible plans (max targets within `MINIMAX_TOLERANCE` of the "
    "cell's minimax), take the Pareto frontier over (max_intermediates, "
    "FLOPs) and select the point minimising `intermediates / "
    "intermediates_min + flops / flops_min`, intermediates breaking ties. "
    "Equal weight because the paper prints the two in separate tables and "
    "trades them against each other without naming a price; normalised "
    "because the two quantities differ by an order of magnitude and an "
    "unnormalised sum would be a FLOPs-only rule wearing a disguise."
)


def test_the_record_states_the_selection_rule_the_code_runs() -> None:
    objective_ = json.loads(CUTS.read_text())["objective"]
    assert objective_["minimax_tolerance"] == MINIMAX_TOLERANCE == 1
    assert objective_["rule"] == SELECTION_RULE == ISSUE_171_RULE
    assert "then" not in objective_  # the lexicographic objective is gone
    # The code is the rule: equal weight, each axis normalised by the
    # frontier's minimum, exact rationals, ties to fewer intermediates.
    pts = ((10, 200), (11, 150), (12, 140))
    assert reference(pts) == (10, 140)
    assert score((11, 150), (10, 140)) == Fraction(11, 10) + Fraction(150, 140)
    assert select(pts) == 1  # 1.1 + 15/14 < 1.2 + 1 < 1 + 10/7
    # An exact tie (both score 1 + 11/10) goes to the fewer intermediates.
    assert select(((10, 11), (11, 10))) == 0
    # Normalised, not summed: the plain sum would take (100, 1000) at 1100
    # over 1101; normalised, 1 + 1011/1000 beats 10/9 + 1.
    assert select(((90, 1011), (100, 1000))) == 0


@pytest.mark.parametrize(("L", "nb"), GRID)
def test_each_recorded_frontier_replays_and_selects_the_recorded_plan(
    L: int, nb: int
) -> None:
    # The replay reproduces every recorded count exactly; the points form a
    # Pareto set (ascending intermediates, strictly descending FLOPs), every
    # one feasible; and #171's rule, run on the recorded points, selects
    # the recorded plan.
    front, k = _frontiers()[L, nb]
    for fp in front:
        c = cost(searched_plan(L, nb, fp.cuts, fp.seed))
        assert (c.max_intermediates, c.flops, c.max_targets) == (
            fp.max_intermediates, fp.flops, fp.max_targets)
        assert feasible(c, _minimax(L, nb))
    pts = [fp.point for fp in front]
    assert all(a[0] < b[0] and a[1] > b[1] for a, b in zip(pts, pts[1:], strict=False))
    assert select(pts) == k
    assert (front[k].cuts, front[k].seed) == _recorded()[L, nb]
    assert front[k].point == point(_replayed(L, nb))


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
    # feasible candidates, so some frontier point weakly dominates it, and
    # the selected point scores no worse than that one under the cell's own
    # normalisers.
    searched, literal = _replayed(L, nb), _literal(L, nb)
    assert feasible(literal, _minimax(L, nb))
    front, _ = _frontiers()[L, nb]
    ref = reference([fp.point for fp in front])
    assert any(fp.point[0] <= literal.max_intermediates
               and fp.point[1] <= literal.flops for fp in front)
    assert score(point(searched), ref) <= score(point(literal), ref)


def test_the_frontier_is_order_searchs_frontier() -> None:
    # #170's `frontier` is the reference implementation of the Pareto set;
    # search.py keeps its own (#172 wires the order search into the
    # package), and this pins that the two agree, point and tie-break, on a
    # whole cell's space: (13, 3), 58 plans, a 5-point frontier.
    plans = list(all_plans(13, 3))
    minimax = min(c.max_targets for _, _, c in plans)
    ok = [p for p in plans if feasible(p[2], minimax)]
    ours = pareto(ok)
    assert ours == order_search_frontier(ok, key=lambda p: point(p[2]))
    assert len(ours) == 5
    assert [(fp.cuts, fp.seed) for fp in _frontiers()[13, 3][0]] == [
        (cuts, seed) for cuts, seed, _ in ours]


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
    # #168: the widened family, the 26 structured orders first, unchanged.
    assert len(SEED_ORDERS) == 58
    assert list(SEED_ORDERS)[:26] == [name for name, _ in seed_orders()]
    assert list(SEED_ORDERS) == [name for name, _ in seed_orders(widened=True)]
    biggest = max(comb(L, 2) - comb(L - 1, 1) for L in range(7, 17))
    assert biggest * len(SEED_ORDERS) == 105 * 58 == 6090


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


def test_a_selected_index_that_is_not_the_recorded_plan_is_refused(
    tmp_path: Path,
) -> None:
    doc = json.loads(CUTS.read_text())
    doc["cuts"]["13/1"]["selected"] = 0  # the lexicographic point, not ours
    bad = tmp_path / "cuts.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="cell 13/1: 'selected' 0"):
        load_cuts(bad)
    doc["cuts"]["13/1"]["selected"] = 9
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="'selected' 9"):
        load_frontiers(bad)


def test_the_tolerance_takes_the_one_target_off_cut_at_15_9() -> None:
    # The cell that motivated MINIMAX_TOLERANCE: (6, 9) at L = 15 holds 116
    # targets in its largest batch, the paper's printed value, where minimax
    # holds 115. With the tolerance the search takes it, and it is at or
    # under in all three tables; at tolerance 0, Table 2 was +7.50% here.
    # (#168's 58 orders keep the cut and move the seed order: (5636, 272)
    # under the 26, (5633, 271) now.)
    printed = tuple(t.cells[15, 9] for t in TABLES)
    assert printed == (5786, 280, 116)
    assert _minimax(15, 9) == 115
    assert _recorded()[15, 9][0] == (6, 9)
    chosen = _replayed(15, 9)
    assert (chosen.flops, chosen.max_intermediates, chosen.max_targets) == (
        5633, 271, 116
    )


def test_what_the_selection_rule_costs_at_16_1() -> None:
    # The rule where it does not help, at (16, 1): the frontier has five
    # points, and the rule takes the fewest-intermediates one (5653 FLOPs,
    # 1633 intermediates), +0.36% over the printed 5633 on Table 1. The
    # fewest FLOPs on the frontier, 5639, comes with 1651 intermediates,
    # over Table 2's 1647, and every point is over Table 1: no selection
    # rule passes this cell, so the rule moves its margin, not its FAIL. The
    # literal hermite order (5644, 1639) is the frontier's fourth point.
    # (Under the 26 orders, before #168, the chosen order was +2.04% here.)
    front, k = _frontiers()[16, 1]
    assert [(fp.flops, fp.max_intermediates) for fp in front] == [
        (5653, 1633), (5647, 1635), (5645, 1638), (5644, 1639), (5639, 1651)]
    assert k == 0 and _recorded()[16, 1] == ((), "prunable asc, ties hermite")
    assert front[3].seed == LITERAL_SEED
    ref = reference([fp.point for fp in front])
    scores = [score(fp.point, ref) for fp in front]
    assert scores == sorted(scores)  # here the score rises along the frontier
    assert TABLES[0].cells[16, 1] == 5633 and TABLES[1].cells[16, 1] == 1647
    assert all(fp.flops > 5633 for fp in front)


# The selection rule's price, cell by cell, read off the recorded frontiers
# (whose counts the replay test above reproduces): in each residual cell of
# Table 1, the frontier points that are at or under all three tables. Where
# that list is non-empty, another point would have passed and the selected
# one does not -- the price #171 asked to be named. (14, 1) and (14, 3) are
# those cells; the other six are unreachable in the space whatever is
# selected (limits.json, and the slow test below).
PRICE = {
    (13, 3): [], (15, 1): [], (15, 3): [], (16, 1): [], (16, 3): [],
    (16, 6): [],
    (14, 1): [(3634, 1071, 680)],
    (14, 3): [(3719, 388, 230)],
}


def test_what_the_selection_rule_costs_cell_by_cell() -> None:
    residual = {(L, nb) for (L, nb) in GRID
                if not _at_or_under_all_three(L, nb, _replayed(L, nb))}
    assert residual == set(PRICE)
    for (L, nb), passing in PRICE.items():
        front, k = _frontiers()[L, nb]
        got = [(fp.flops, fp.max_intermediates, fp.max_targets) for fp in front
               if all(g <= t.cells[L, nb] for g, t in zip(
                   (fp.flops, fp.max_intermediates, fp.max_targets), TABLES,
                   strict=True))]
        assert got == passing, (L, nb)
    # (14, 1): (3655, 1060) is selected, 2.0058 against 2.0104 for the
    # printed (3634, 1071): 11 fewer intermediates outweigh 21 more FLOPs.
    # (14, 3): (3720, 386) at 2.0003 against 2.0052 for (3719, 388).
    for (L, nb), sel, other in (((14, 1), (1060, 3655), (1071, 3634)),
                                ((14, 3), (386, 3720), (388, 3719))):
        front, k = _frontiers()[L, nb]
        ref = reference([fp.point for fp in front])
        assert front[k].point == sel
        assert score(sel, ref) < score(other, ref)


def test_where_the_selection_rule_helps_13_1() -> None:
    # The one cell the rule moves against #168's lexicographic objective
    # (fewest intermediates, then FLOPs -- the frontier's first point): at
    # (13, 1) it takes (2872, 843), the printed cell in all three tables,
    # over (2882, 842), +0.35% on Table 1.
    printed = tuple(t.cells[13, 1] for t in TABLES)
    assert printed == (2872, 843, 560)
    front, k = _frontiers()[13, 1]
    assert [(fp.flops, fp.max_intermediates) for fp in front] == [
        (2882, 842), (2872, 843)]
    assert k == 1
    assert _recorded()[13, 1] == ((), "+max/-mid, ties hermite")
    chosen = _replayed(13, 1)
    assert (chosen.flops, chosen.max_intermediates, chosen.max_targets) == printed
    # Everywhere else the selected point is the frontier's first.
    assert {c for c, (_, k) in _frontiers().items() if k != 0} == {(13, 1)}


def test_the_widened_space_reaches_12_1_exactly() -> None:
    # #168's evidence that the 26-order residual was a bound of the space,
    # not of the problem: at (12, 1) a widened order is the printed cell in
    # all three tables, and the search takes it.
    printed = tuple(t.cells[12, 1] for t in TABLES)
    assert printed == (2221, 645, 455)
    assert _recorded()[12, 1] == ((), "+max/-mid, ties hermite")
    chosen = _replayed(12, 1)
    assert (chosen.flops, chosen.max_intermediates, chosen.max_targets) == printed


def test_a_cut_file_with_the_wrong_schema_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "cuts.json"
    bad.write_text('{"schema": 1, "cuts": {}}')
    with pytest.raises(ValueError, match="schema"):
        load_cuts(bad)


def _searched_limit_value() -> dict:
    lim = next(
        lim for lim in load_limits(JENGINE / "limits.json")
        if isinstance(lim.asserted_value, dict)
        and lim.asserted_value.get("board") == "searched"
    )
    return lim.asserted_value


def _at_or_under_all_three(L: int, nb: int, c: PlanCost) -> bool:
    got = (c.flops, c.max_intermediates, c.max_targets)
    return all(g <= t.cells[L, nb] for g, t in zip(got, TABLES, strict=True))


def test_the_residual_is_split_into_unreachable_and_the_selection_price() -> None:
    # limits.json names each residual cell once: unreachable in the 58-order
    # space, or a cell where #171's rule passes over a frontier point at or
    # under all three tables. The witness plans are recorded frontier points
    # that replay to what the record says and are at or under all three;
    # the selected point in those cells is not. The cell the rule moved,
    # (13, 1), is in neither: it is at or under now.
    value = _searched_limit_value()
    assert value["seed_orders"] == len(SEED_ORDERS) == 58
    residual = {(L, nb) for L, nb, _ in value["residual_percent_over"]
                ["jengine/table1"]}
    unreachable = {tuple(c) for c in value["unreachable_in_space"]}
    price = {tuple(w["cell"]): w for w in value["selection_price"]}
    assert "objective_price" not in value
    assert set(price) == {(14, 1), (14, 3)}
    assert unreachable == {(13, 3), (15, 1), (15, 3), (16, 1), (16, 3), (16, 6)}
    assert unreachable | set(price) == residual == set(PRICE)
    assert not unreachable & set(price)
    for (L, nb), w in price.items():
        c = cost(searched_plan(L, nb, tuple(w["cut"]), w["seed_order"]))
        assert [c.flops, c.max_intermediates, c.max_targets] == w[
            "flops_intermediates_targets"]
        assert _at_or_under_all_three(L, nb, c)
        assert feasible(c, _minimax(L, nb))
        front, k = _frontiers()[L, nb]
        assert (tuple(w["cut"]), w["seed_order"]) in [
            (fp.cuts, fp.seed) for fp in front]
        sel = _replayed(L, nb)
        assert [sel.flops, sel.max_intermediates] == w[
            "selected_flops_intermediates"]
        assert not _at_or_under_all_three(L, nb, sel)
    (moved,) = value["moved_by_the_selection_rule"]
    assert tuple(moved["cell"]) == (13, 1) and (13, 1) not in residual
    front, k = _frontiers()[13, 1]
    assert [front[0].flops, front[0].max_intermediates] == moved[
        "lexicographic_flops_intermediates"]
    assert [front[k].flops, front[k].max_intermediates] == moved[
        "selected_flops_intermediates"]


@pytest.mark.slow
def test_the_search_reproduces_the_record() -> None:
    # ~8 minutes serial: 40 cells x up to 105 cuts x 58 seed orders.
    results = {(L, nb): search(L, nb) for L, nb in GRID}
    assert dump_cuts(results) == CUTS.read_text()
    for (L, nb), r in results.items():
        assert r.cost == _replayed(L, nb)
        assert r.frontier[r.selected][2] == r.cost


@pytest.mark.slow
def test_no_plan_in_the_space_is_at_or_under_all_three_in_a_residual_cell() -> None:
    # limits.json's "unreachable in the 58-order space": in each of these
    # residual cells, no plan in the whole space -- any cut, any of the 58
    # seed orders (#168), any target count -- is at or under all three
    # tables at once. (test_scoreboards.py checks the cheap 1- and 3-batch
    # cells on the fast tier.)
    assert len(SEED_ORDERS) == 58
    cells = {tuple(c) for c in _searched_limit_value()["unreachable_in_space"]}
    assert len(cells) == 6
    for L, nb in sorted(cells):
        for cuts, seed, c in all_plans(L, nb):
            assert not _at_or_under_all_three(L, nb, c), (L, nb, cuts, seed)


@pytest.mark.slow
def test_in_the_selection_price_cells_the_space_holds_a_plan_at_or_under() -> None:
    # limits.json's "the selection rule's price": in (14, 1) and (14, 3) the
    # 58-order space DOES hold a plan at or under all three tables, the
    # recorded witness is the first one in the search's own order, it is a
    # point of the cell's recorded frontier, and #171's rule selects another.
    for w in _searched_limit_value()["selection_price"]:
        L, nb = w["cell"]
        hits = [(cuts, seed) for cuts, seed, c in all_plans(L, nb)
                if _at_or_under_all_three(L, nb, c)]
        assert hits and hits[0] == (tuple(w["cut"]), w["seed_order"]), (L, nb)
        assert _recorded()[L, nb] not in hits
        assert hits[0] in [(fp.cuts, fp.seed) for fp in _frontiers()[L, nb][0]]
