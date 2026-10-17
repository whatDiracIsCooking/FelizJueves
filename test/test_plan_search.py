"""plan/search.py: the cut and seed-order search behind the `searched` board.

Tests replay the recorded choices (data/scoreboards/jengine/
searched_cuts.json, schema 3 since #172: one explicit seed order per batch)
and check what the record must satisfy, and that a doctored record is
refused. Every cell is L = 7..16 > FAST_L_MAX (test/tiers.py), and loading
the record builds each cell's batches to check its orders, so all of that is
`slow`; the fast tier keeps what reads the record's JSON and the code's
constants. test_the_build_step_reproduces_the_record re-runs the build step
and asserts it reproduces the record byte for byte. How the replayed plans
score on the searched board is pinned in test_scoreboards.py.
"""

from __future__ import annotations

import copy
import json
import os
from fractions import Fraction
from functools import cache
from math import comb
from pathlib import Path

import pytest

from felizjueves.literature.limits import load_limits
from felizjueves.literature.tables import load_table
from felizjueves.plan import order_search
from felizjueves.plan.batching import (
    MINIMAX_TARGETS,
    largest_batch,
    partition,
    target_counts,
)
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.recovery import seed_orders
from felizjueves.plan.search import (
    CLIMB_BUDGET,
    CLIMB_SEED,
    CLIMB_SWEEP,
    CUTS_SCHEMA,
    LITERAL_SEED,
    MINIMAX_TOLERANCE,
    SEED_ORDERS,
    SELECTION_RULE,
    FrontierPoint,
    batch_starts,
    build,
    candidate_cuts,
    check_orders,
    climb_starts,
    dump_cuts,
    feasible,
    feasible_cuts,
    fixed_cut,
    load_cuts,
    load_frontiers,
    point,
    reference,
    rule_order,
    score,
    search,
    searched_plan,
    select,
    space_record,
    units,
)
from felizjueves.plan.sieve import plan
from felizjueves.spec import Operator, Spec, Target

DATA = Path(__file__).resolve().parents[1] / "data"
CUTS = DATA / "scoreboards" / "jengine" / "searched_cuts.json"
JENGINE = DATA / "literature" / "jengine"
TABLES = [load_table(JENGINE / f"table{n}.json") for n in (1, 2, 3)]
GRID = [(L, nb) for L in range(7, 17) for nb in (1, 3, 6, 9)]


@cache
def _recorded():
    return load_cuts(CUTS)


@cache
def _frontiers() -> dict[tuple[int, int], tuple[tuple[FrontierPoint, ...], int]]:
    return load_frontiers(CUTS)


@cache
def _doc() -> dict:
    return json.loads(CUTS.read_text())


@cache
def _replayed(L: int, nb: int) -> PlanCost:
    return cost(searched_plan(L, nb, *_recorded()[L, nb]))


@cache
def _literal(L: int, nb: int) -> PlanCost:
    return cost(plan(Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J,
                          n_batches=nb)))


@cache
def _rule_point(L: int, nb: int, cuts: tuple[int, ...], name: str):
    """One plan of the 58-rule space (#168): every batch seeded by `name`."""
    return point(cost(searched_plan(L, nb, cuts, name)))


@cache
def _rule_selection(L: int, nb: int):
    """What "the best of the 58" selects, computed here and read from no
    record: #171's rule over the frontier of the 58-rule plans on the cell's
    feasible cuts. A search that degenerated to the rule space would record
    exactly this point."""
    return _rule_selected_plan(L, nb)[1]


@cache
def _rule_selected_plan(L: int, nb: int):
    """((cut, rule name), point) of `_rule_selection`."""
    plans = [(c, n) for c in feasible_cuts(L, nb) for n in SEED_ORDERS]
    front = order_search.frontier(plans, key=lambda cn: _rule_point(L, nb, *cn))
    pts = [_rule_point(L, nb, *cn) for cn in front]
    k = select(pts)
    return front[k], pts[k]


def _strictly_better(choices) -> set[tuple[int, int]]:
    """The cells where the plan replayed from `choices` (cell -> (cut,
    orders)) strictly beats the live 58-rule selection; refused outright if
    it is worse anywhere."""
    out = set()
    for (L, nb), choice in choices.items():
        got, rule = point(cost(searched_plan(L, nb, *choice))), _rule_selection(L, nb)
        assert order_search.dominates(got, rule), (L, nb, got, rule)
        if got != rule:
            out.add((L, nb))
    return out


def _minimax(L: int, nb: int) -> int:
    """The smallest max-targets any cut gives: the literal plan's, whose cut
    is batching's minimax one (and at 1 batch the only one)."""
    return _literal(L, nb).max_targets


def _counts(c) -> tuple[int, int, int]:
    """(flops, max_intermediates, max_targets): Tables 1, 2, 3's order."""
    return (c.flops, c.max_intermediates, c.max_targets)


def _at_or_under_all_three(L: int, nb: int, c) -> bool:
    return all(g <= t.cells[L, nb]
               for g, t in zip(_counts(c), TABLES, strict=True))


def test_the_record_covers_exactly_the_table_grid() -> None:
    # From the JSON; that it loads, and its batches check, is the slow twin.
    assert {tuple(int(x) for x in k.split("/")) for k in _doc()["cuts"]} == set(GRID)
    for t in TABLES:
        assert set(t.cells) == set(GRID)


@pytest.mark.slow  # load_cuts builds every cell's batches: L > FAST_L_MAX
def test_the_loaded_record_covers_exactly_the_table_grid() -> None:
    assert set(_recorded()) == set(GRID)


# #171's rule, as the issue body states it (line breaks aside). It was
# declared before the board was re-read; this pins that the code quotes it
# unchanged, so a change to the rule is a diff here. #172 does not touch it.
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
    objective_ = _doc()["objective"]
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


def test_the_record_states_the_space_the_code_searches() -> None:
    # #172's acceptance: the space actually searched is named in the record
    # -- starts, neighbourhood, sweep, climb budget and seed.
    doc = _doc()
    assert doc["schema"] == CUTS_SCHEMA == 3
    assert doc["space"] == space_record()
    space = doc["space"]
    assert space["neighbourhood"] == ["transposition", "move"]
    assert space["sweep"] == ["1", "3/4", "1/2", "1/4", "0"]
    assert [Fraction(w) for w in space["sweep"]] == list(CLIMB_SWEEP)
    assert space["budget"] == CLIMB_BUDGET
    assert space["seed"] == CLIMB_SEED == 172
    assert "58" in space["starts"] and len(SEED_ORDERS) == 58


@pytest.mark.slow
@pytest.mark.parametrize(("L", "nb"), GRID)
def test_each_recorded_frontier_replays_and_selects_the_recorded_plan(
    L: int, nb: int
) -> None:
    # The replay reproduces every recorded count exactly, from the recorded
    # permutations; the points form a Pareto set (ascending intermediates,
    # strictly descending FLOPs), every one feasible; and #171's rule, run
    # on the recorded points, selects the recorded plan.
    front, k = _frontiers()[L, nb]
    for fp in front:
        c = cost(searched_plan(L, nb, fp.cuts, fp.orders))
        assert (c.max_intermediates, c.flops, c.max_targets) == (
            fp.max_intermediates, fp.flops, fp.max_targets)
        assert feasible(c, _minimax(L, nb))
        assert fp.cuts in feasible_cuts(L, nb)
        assert all(s in SEED_ORDERS for s in fp.starts)
    pts = [fp.point for fp in front]
    assert all(a[0] < b[0] and a[1] > b[1] for a, b in zip(pts, pts[1:], strict=False))
    assert select(pts) == k
    assert front[k].choice == _recorded()[L, nb]
    assert front[k].point == point(_replayed(L, nb))
    cell = _doc()["cuts"][f"{L}/{nb}"]
    assert (tuple(cell["cut"]), tuple(cell["start"])) == (
        front[k].cuts, front[k].starts)


@pytest.mark.slow
@pytest.mark.parametrize(("L", "nb"), GRID)
def test_each_recorded_cut_is_within_the_minimax_tolerance(L: int, nb: int) -> None:
    cuts, orders = _recorded()[L, nb]
    assert cuts in set(candidate_cuts(L, nb))
    partition(L, nb, cut=fixed_cut(cuts))  # raises on a malformed cut
    assert len(orders) == nb
    if nb > 3:
        counts = target_counts(L)
        minimax = largest_batch(counts, MINIMAX_TARGETS(counts, nb // 3))
        assert minimax == _minimax(L, nb)
        assert largest_batch(counts, cuts) <= minimax + MINIMAX_TOLERANCE
    else:
        assert cuts == ()
    assert feasible(_replayed(L, nb), _minimax(L, nb))


@pytest.mark.slow
@pytest.mark.parametrize(("L", "nb"), GRID)
def test_the_climbed_frontier_dominates_every_rule_plan(L: int, nb: int) -> None:
    # Every start is evaluated, so every plan of the 58-rule space (#168) on
    # a feasible cut is weakly dominated by a recorded frontier point: the
    # climb only adds to the space #171 selected from. The literal plan
    # (lowest minimax cut, hermite order) is one of them, and the selected
    # point scores no worse than it under the cell's own normalisers.
    front, _ = _frontiers()[L, nb]
    pts = [fp.point for fp in front]
    for cuts in feasible_cuts(L, nb):
        for name in SEED_ORDERS:
            p = _rule_point(L, nb, cuts, name)
            assert any(order_search.dominates(q, p) for q in pts), (cuts, name)
    literal = _literal(L, nb)
    assert feasible(literal, _minimax(L, nb))
    assert score(point(_replayed(L, nb)), reference(pts)) <= score(
        point(literal), reference(pts))


def test_the_frontier_is_order_searchs_frontier() -> None:
    # #171 kept a private Pareto set while order_search was imported by
    # nothing; #172 wires the module in and the duplicate is gone.
    import felizjueves.plan.search as search_module

    assert not hasattr(search_module, "pareto")
    assert search_module.frontier is order_search.frontier
    assert search_module.product is order_search.product
    assert search_module.point is order_search.point

@pytest.mark.slow

def test_the_literal_plan_is_the_hermite_seed_of_the_literal_cut() -> None:
    assert SEED_ORDERS[LITERAL_SEED] is next(iter(seed_orders()))[1]
    for L, nb in ((9, 1), (12, 6)):
        cut = MINIMAX_TARGETS(target_counts(L), max(1, nb // 3))
        assert cost(searched_plan(L, nb, cut)) == _literal(L, nb)
        # a rule name and its explicit permutations are one plan
        dags = searched_plan(L, nb, cut)
        orders = [rule_order(L, d, LITERAL_SEED) for d in dags]
        again = searched_plan(L, nb, cut, orders)
        assert [(d.batch, d.targets, dict(d.parents)) for d in again] == [
            (d.batch, d.targets, dict(d.parents)) for d in dags]

@pytest.mark.slow

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
    # #172: the climbs. One per feasible cut, batch and DISTINCT rule order,
    # each under every weight of the sweep, each at most CLIMB_BUDGET
    # neighbours.
    assert CLIMB_SWEEP == order_search.SWEEP and len(CLIMB_SWEEP) == 5
    us = units(7, 3, budget=0)
    assert len(us) == sum(len(climb_starts(7, d)[0])
                          for d in searched_plan(7, 3, ()))
    assert all(u.budget == 0 and u.seed == CLIMB_SEED for u in us)
    # Only the feasible cuts are climbed: one per cell but three.
    assert {(L, nb) for L, nb in GRID if len(feasible_cuts(L, nb)) > 1} == {
        (7, 9), (8, 9), (15, 9)}

@pytest.mark.slow

def test_starts_are_the_distinct_rule_orders_of_a_batch() -> None:
    dag = searched_plan(8, 3, ())[1]
    starts = batch_starts(8, dag)
    assert list(starts)[0] == LITERAL_SEED
    assert len(set(starts.values())) == len(starts) <= 58
    mine = {r for r, _ in dag.targets}
    for name, o in starts.items():
        assert set(o) == mine and len(o) == len(mine)
        assert o == rule_order(8, dag, name)
    # every rule's order is one of the starts, under its first namer
    assert {rule_order(8, dag, n) for n in SEED_ORDERS} == set(starts.values())

@pytest.mark.slow

def test_a_search_does_not_depend_on_the_worker_count() -> None:
    # The build step spreads climbs over processes; each is a pure function
    # of its unit, so the record is the same serially and pooled.
    serial = search(7, 3, budget=3)
    assert build([(7, 3)], workers=2, budget=3)[7, 3] == serial
    assert dump_cuts({(7, 3): serial}, budget=3) == dump_cuts(
        build([(7, 3)], workers=1, budget=3), budget=3)

@pytest.mark.slow

def test_a_cut_that_empties_a_batch_is_refused() -> None:
    # A first cut at 1 leaves Y_{0,0} and Z_{0,0} empty ([JEngine, v2, Fig. 4]
    # does not draw them at L = 6); #64 refuses it at the partition, so it is
    # neither a search candidate nor a replayable plan.
    assert (1,) not in set(candidate_cuts(7, 6))
    with pytest.raises(ValueError, match=r"Y_\{0,0\}, Z_\{0,0\}"):
        searched_plan(7, 6, (1,))

@pytest.mark.slow

def test_an_unknown_seed_order_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed order 'nope'"):
        searched_plan(7, 1, (), "nope")
    doc = copy.deepcopy(_doc())
    doc["cuts"]["7/1"]["frontier"][0]["start"] = ["nope"]
    bad = tmp_path / "cuts.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="seed order 'nope'"):
        load_cuts(bad)


# -- a doctored record is refused (#172; #173 pins more) -----------------------


def _doctor(tmp_path: Path, key: str, edit) -> Path:
    doc = copy.deepcopy(_doc())
    cell = doc["cuts"][key]
    edit(cell["frontier"][cell["selected"]]["orders"])
    bad = tmp_path / "cuts.json"
    bad.write_text(json.dumps(doc))
    return bad


@pytest.mark.slow
@pytest.mark.parametrize(("edit", "message"), [
    (lambda o: o[0].pop(), r"cell 9/3, batch 0 \(x0\.\.9\): the seed order "
     r"is not a permutation"),
    (lambda o: o[1].__setitem__(0, o[1][1]), r"cell 9/3, batch 1 \(y0\.\.9\): "
     r"the seed order repeats a target"),
    (lambda o: o.__setitem__(slice(0, 2), [o[1], o[0]]),
     r"cell 9/3, batch 0 \(x0\.\.9\): the seed order is not a permutation"),
    (lambda o: o[2].__setitem__(0, [9, 9, 9]), r"cell 9/3, batch 2 \(z0\.\.9\)"),
    (lambda o: o.pop(), r"cell 9/3: 2 seed orders for 3 batches"),
])
def test_a_doctored_permutation_is_refused(tmp_path: Path, edit, message) -> None:
    # A short permutation, a duplicated target, a permutation of the WRONG
    # batch (x's and y's swapped), a target from outside the class, a
    # missing batch: each refused on load, naming the cell and the batch.
    bad = _doctor(tmp_path, "9/3", edit)
    with pytest.raises(ValueError, match=message):
        load_cuts(bad)
    with pytest.raises(ValueError, match=message):
        load_frontiers(bad)


@pytest.mark.slow
@pytest.mark.parametrize(("key", "edit", "message"), [
    # A permutation of the wrong batch where the two batches are the same
    # size: x5..10's and y5..10's orders swapped (55 targets each), so only
    # the target SET can tell them apart.
    ("10/6", lambda o: o.__setitem__(slice(1, 4, 2), [o[3], o[1]]),
     r"cell 10/6, batch 1 \(x5\.\.10\): the seed order is not a permutation "
     r"of the batch's 55 targets"),
    # One order more than the cut has batches: a batch the cut does not make.
    ("9/3", lambda o: o.append(o[0]), r"cell 9/3: 4 seed orders for 3 batches"),
])
def test_a_permutation_of_another_batch_is_refused(
    tmp_path: Path, key: str, edit, message
) -> None:
    bad = _doctor(tmp_path, key, edit)
    with pytest.raises(ValueError, match=message):
        load_cuts(bad)


@pytest.mark.slow
@pytest.mark.parametrize(("key", "cut", "message"), [
    # The recorded orders are the batches of cut (5,); the doctored point
    # names a cut that does not produce them (#173), one point lower or
    # higher, so the first batch gains or loses targets.
    ("10/6", [4], r"cell 10/6, batch 0 \(x0\.\.3\): the seed order is not a "
     r"permutation of the batch's 24 targets \(43 given, 19 not in the batch"),
    ("10/6", [6], r"cell 10/6, batch 0 \(x0\.\.5\): the seed order is not a "
     r"permutation of the batch's 63 targets \(43 given, 0 not in the batch, "
     r"20 missing\)"),
    ("16/9", [7, 10], r"cell 16/9, batch 1 \(x7\.\.9\)"),
])
def test_a_record_naming_batches_its_cut_does_not_produce_is_refused(
    tmp_path: Path, key: str, cut: list[int], message: str
) -> None:
    doc = copy.deepcopy(_doc())
    cell = doc["cuts"][key]
    assert cell["cut"] != cut
    cell["frontier"][cell["selected"]]["cut"] = cut
    cell["cut"] = cut  # the cell agrees with its point: only the orders lie
    bad = tmp_path / "cuts.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=message):
        load_cuts(bad)
    with pytest.raises(ValueError, match=message):
        load_frontiers(bad)

@pytest.mark.slow

def test_the_recorded_counts_are_not_believed(tmp_path: Path) -> None:
    # #173: nothing the record says about a plan's COST is read. Every count
    # of every frontier point is doctored; the loaded choices are unchanged,
    # and the plan replayed from them -- the recorded permutations, sieved
    # here -- still gives the true counts. (load_frontiers carries the
    # recorded counts as written; that they equal the replay is held cell by
    # cell by test_each_recorded_frontier_replays_and_selects_the_recorded_plan.)
    doc = copy.deepcopy(_doc())
    for cell in doc["cuts"].values():
        for p in cell["frontier"]:
            p["flops"], p["max_intermediates"], p["max_targets"] = 1, 1, 1
    bad = tmp_path / "cuts.json"
    bad.write_text(json.dumps(doc))
    doctored = load_cuts(bad)
    assert doctored == _recorded()
    for L, nb in [(9, 3), (13, 1), (16, 9)]:
        c = cost(searched_plan(L, nb, *doctored[L, nb]))
        assert c == _replayed(L, nb)
        assert (c.flops, c.max_intermediates, c.max_targets) != (1, 1, 1)

@pytest.mark.slow

def test_searched_plan_refuses_what_is_not_a_permutation() -> None:
    cut, orders = _recorded()[9, 3]
    with pytest.raises(ValueError, match="batch 1"):
        searched_plan(9, 3, cut, (orders[0], orders[0], orders[2]))
    with pytest.raises(ValueError, match="1 seed orders for 3 batches"):
        check_orders(9, 3, cut, orders[:1])
    assert check_orders(9, 3, cut, [[list(r) for r in o] for o in orders]) == orders

@pytest.mark.slow

def test_a_selected_index_that_is_not_the_recorded_plan_is_refused(
    tmp_path: Path,
) -> None:
    doc = copy.deepcopy(_doc())
    key = next(k for k, c in doc["cuts"].items() if len(c["frontier"]) > 1)
    cell = doc["cuts"][key]
    cell["selected"] = 1 - min(cell["selected"], 1)
    if cell["frontier"][cell["selected"]]["start"] == cell["start"]:
        cell["start"] = ["hermite-reversed"] * len(cell["start"])
    bad = tmp_path / "cuts.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=f"cell {key}: 'selected'"):
        load_cuts(bad)
    cell["selected"] = 99
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="'selected' 99"):
        load_frontiers(bad)


def test_a_cut_file_with_the_wrong_schema_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "cuts.json"
    bad.write_text('{"schema": 2, "cuts": {}}')
    with pytest.raises(ValueError, match="schema 2, not 3"):
        load_cuts(bad)


def _searched_limit_value() -> dict:
    lim = next(
        lim for lim in load_limits(JENGINE / "limits.json")
        if isinstance(lim.asserted_value, dict)
        and lim.asserted_value.get("board") == "searched"
    )
    return lim.asserted_value

@pytest.mark.slow

def test_the_searched_plans_are_at_or_under_all_three_everywhere() -> None:
    # #172's outcome: no residual. 22 cells land on the printed triple
    # exactly; the other 18 come in under it.
    assert all(_at_or_under_all_three(L, nb, _replayed(L, nb)) for L, nb in GRID)
    exact = {(L, nb) for L, nb in GRID
             if _counts(_replayed(L, nb)) == tuple(t.cells[L, nb] for t in TABLES)}
    assert len(exact) == 22


# #171's residual, cell by cell, and what the climb selects there now:
# (flops, max_intermediates, max_targets), printed then chosen. Every one is
# at or under all three tables; each is a pinned figure, so a re-selection
# is a diff here.
PREVIOUS_RESIDUAL = {
    (13, 3): ((2922, 309, 190), (2920, 303, 190)),
    (14, 1): ((3634, 1071, 680), (3631, 1049, 680)),
    (14, 3): ((3719, 390, 230), (3716, 384, 230)),
    (15, 1): ((4540, 1329, 816), (4539, 1303, 816)),
    (15, 3): ((4613, 482, 276), (4607, 475, 276)),
    (16, 1): ((5633, 1647, 969), (5601, 1601, 969)),
    (16, 3): ((5712, 591, 327), (5676, 574, 327)),
    (16, 6): ((6426, 405, 164), (6414, 402, 164)),
}

@pytest.mark.slow

def test_the_previous_residual_cells_cell_by_cell() -> None:
    value = _searched_limit_value()
    prev = value["previous_residual"]
    assert {(L, nb) for L, nb, _ in prev["jengine/table1"]} == set(PREVIOUS_RESIDUAL)
    assert {tuple(c) for c in prev["unreachable_in_space"]} | {
        tuple(c) for c in prev["selection_price"]} == set(PREVIOUS_RESIDUAL)
    for (L, nb), (printed, chosen) in PREVIOUS_RESIDUAL.items():
        assert tuple(t.cells[L, nb] for t in TABLES) == printed
        assert _counts(_replayed(L, nb)) == chosen, (L, nb)
        assert all(c <= p for c, p in zip(chosen, printed, strict=True))


def test_the_limit_names_the_space_and_records_no_residual() -> None:
    # The space actually searched is named in the record, and is the one
    # searched_cuts.json states and the code runs.
    value = _searched_limit_value()
    assert value["seed_orders"] == len(SEED_ORDERS) == 58
    assert value["space"] == _doc()["space"] == space_record()
    assert value["residual_percent_over"] == {}
    assert value["unreachable_in_space"] == []
    assert value["selection_price"] == []


# The cells where the climbed selection is strictly better than #171's
# selection from the 58 rules -- better on one axis and no worse on the
# other -- with #171's (flops, max_intermediates) there. In the other 26
# cells the climb selects the same point.
STRICTLY_BETTER = {
    (10, 1): (1252, 367), (12, 1): (2221, 645), (13, 1): (2872, 843),
    (13, 3): (2948, 305), (14, 1): (3655, 1060), (14, 3): (3720, 386),
    (14, 6): (4224, 269), (15, 1): (4566, 1320), (15, 3): (4624, 478),
    (15, 6): (5231, 360), (16, 1): (5653, 1633), (16, 3): (5759, 583),
    (16, 6): (6447, 402), (16, 9): (7104, 324),
}

@pytest.mark.slow

def test_where_the_climb_beats_the_rule_selection() -> None:
    value = _searched_limit_value()
    assert {tuple(c) for c in value["strictly_dominates_58_rule_selection"]} == set(
        STRICTLY_BETTER)
    for (L, nb), (f, i) in STRICTLY_BETTER.items():
        c = _replayed(L, nb)
        assert c.flops <= f and c.max_intermediates <= i
        assert (c.flops, c.max_intermediates) != (f, i)


# #173: the search is not a no-op. at_or_under gives no credit for coming in
# under, so a search that silently degenerated to "the best of the 58 rules"
# could pass the board. N is the number of cells where the recorded plan is
# STRICTLY better than the 58-rule selection, computed live
# (`_rule_selection`), not read from STRICTLY_BETTER's pinned figures. It is
# a ratchet: raise N (and add the cells) when a better search finds more,
# never lower it. #172 measured 14.
N_STRICTLY_BETTER = 14

@pytest.mark.slow

def test_the_search_is_not_the_best_of_the_58() -> None:
    better = _strictly_better(_recorded())
    assert len(better) >= N_STRICTLY_BETTER == len(STRICTLY_BETTER)
    assert better == set(STRICTLY_BETTER)  # a new cell is a ratchet: add it
    # The live 58-rule selection is #171's, as STRICTLY_BETTER pins it.
    for (L, nb), (f, i) in STRICTLY_BETTER.items():
        assert _rule_selection(L, nb) == (i, f), (L, nb)

@pytest.mark.slow

def test_the_guard_catches_a_search_degenerated_to_the_best_of_the_58() -> None:
    # The degenerate record, built here: in every cell the 58-rule selection,
    # written as schema 3 writes it (one explicit order per batch). It loads
    # (each order is a permutation of its batch) and no plan is worse than
    # the rule selection -- yet the count above finds 0 cells, so
    # test_the_search_is_not_the_best_of_the_58 fails on it.
    degenerate = {}
    for L, nb in GRID:
        (cuts, name), _ = _rule_selected_plan(L, nb)
        dags = searched_plan(L, nb, cuts, name)
        degenerate[L, nb] = (cuts, check_orders(
            L, nb, cuts, [rule_order(L, d, name) for d in dags]))
    assert _strictly_better(degenerate) == set()


# Of the N cells, those whose recorded plan seeds every batch with SOME rule's
# order -- a different rule per batch, which the 58-rule space (one rule for
# all batches) does not hold. In the other 11 at least one batch's order is
# no rule's: the climb itself moved it.
MIXED_RULES_ONLY = {(14, 6), (15, 6), (16, 9)}


@pytest.mark.slow
@pytest.mark.parametrize(("L", "nb"), sorted(STRICTLY_BETTER))
def test_the_strictly_better_plans_leave_the_rule_space(L: int, nb: int) -> None:
    # No one of the 58 rules seeds every batch of the recorded plan.
    cuts, orders = _recorded()[L, nb]
    # A batch's rule orders depend only on its target set, which the sieve
    # keeps.
    dags = searched_plan(L, nb, cuts, orders)
    for name in SEED_ORDERS:
        assert tuple(rule_order(L, d, name) for d in dags) != orders, name
    starts = [set(batch_starts(L, d).values()) for d in dags]
    climbed = any(o not in s for o, s in zip(orders, starts, strict=True))
    assert climbed == ((L, nb) not in MIXED_RULES_ONLY)

@pytest.mark.slow

def test_the_tolerance_still_takes_the_one_target_off_cut_at_15_9() -> None:
    # The cell that motivated MINIMAX_TOLERANCE: (6, 9) at L = 15 holds 116
    # targets in its largest batch, the paper's printed value, where minimax
    # holds 115. With the tolerance the search takes it, and it is at or
    # under in all three tables; at tolerance 0, Table 2 was +7.50% here.
    printed = tuple(t.cells[15, 9] for t in TABLES)
    assert printed == (5786, 280, 116)
    assert _minimax(15, 9) == 115
    assert _recorded()[15, 9][0] == (6, 9)
    assert _counts(_replayed(15, 9)) == (5633, 271, 116)


def test_the_search_subcommand_refuses_bad_arguments(capsys) -> None:
    from felizjueves.plan.__main__ import EXIT_COULD_NOT_RUN, main

    assert main(["search", "--workers", "0"]) == EXIT_COULD_NOT_RUN
    assert "could not run" in capsys.readouterr().err


@pytest.mark.slow
def test_the_build_step_reproduces_the_record() -> None:
    # `python -m felizjueves.plan search`'s work, over FJ_SEARCH_WORKERS
    # processes (default 6): see searched_cuts.json's space and the PR for
    # its wall-clock.
    workers = int(os.environ.get("FJ_SEARCH_WORKERS", "6"))
    results = build(GRID, workers=workers)
    assert dump_cuts(results) == CUTS.read_text()
    for (L, nb), r in results.items():
        assert r.cost == _replayed(L, nb)
