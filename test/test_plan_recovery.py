"""#14: the inputs behind the printed cells. Table 3's are found (the cut);
Tables 1-2's are not, and limits.json records why. #108 widened the seed-order
family to 58 rules and decided the doctrine: rules only, so Tables 1-2 are
refused, and every number the records quote about that is recomputed here."""

from __future__ import annotations

from pathlib import Path

import pytest

from felizjueves.literature.limits import load_limits
from felizjueves.literature.rules import Candidate, ExactGivenInputs, Status
from felizjueves.literature.scoreboard import Scoreboard, load_scoreboard, score
from felizjueves.literature.tables import load_table
from felizjueves.plan.batching import (
    AXES,
    MINIMAX_TARGETS,
    batch_size,
    in_batch,
    partition,
    target_counts,
)
from felizjueves.plan.cost import cost
from felizjueves.plan.dag import build
from felizjueves.plan.recovery import (
    CUT_INPUTS,
    exact_counts,
    recovered_batches,
    recovered_candidates,
    recovered_costs,
    recovered_plan,
    seed_orders,
    two_ended_cuts,
)
from felizjueves.plan.sieve import plan, sieve
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

DATA = Path(__file__).resolve().parents[1] / "data"
JENGINE = DATA / "literature" / "jengine"
TABLES = {
    t.id: t for t in (load_table(JENGINE / f"table{n}.json") for n in (1, 2, 3))
}
CELLS = sorted(TABLES["jengine/table3"].cells)
T3 = TABLES["jengine/table3"]


@pytest.fixture(scope="module")
def candidates() -> dict[str, Candidate]:
    return recovered_candidates(TABLES)


# -- the cut rule ---------------------------------------------------------------


def test_two_ended_cuts_by_hand() -> None:
    # L = 7, X: [1, 3, 7, 11, 10, 6, 3, 1], 42 targets, T = 14 at k = 3.
    # Bottom: 1+3+7 = 11 (|11-14| = 3) beats 22 (8) -> m 0..2.
    # Top:    1+3+6 = 10 (4) beats 20 (6)          -> m 5..7. Middle 3..4 = 21.
    x = target_counts(7)["x"]
    assert x == [1, 3, 7, 11, 10, 6, 3, 1]
    assert two_ended_cuts(x, 3) == (3, 5)
    assert two_ended_cuts(x, 2) == (4,)  # 22 (|22-21| = 1) beats 11
    assert two_ended_cuts(x, 1) == ()


def test_a_tie_takes_the_shorter_run() -> None:
    # T = 2: 1 and 3 are equally far; the shorter run (m = 0 alone) wins.
    assert two_ended_cuts([1, 2, 1], 2) == (1,)


def test_too_many_sub_batches_is_refused() -> None:
    with pytest.raises(ValueError, match="none for the middle"):
        two_ended_cuts([5, 0, 0], 3)


@pytest.mark.parametrize(("L", "nb"), CELLS)
def test_recovered_batches_cover_every_target_once(L: int, nb: int) -> None:
    batches = recovered_batches(L, nb)
    assert len(batches) == nb
    for r in hermite_targets(L):
        assert sum(in_batch(r, *b) for b in batches) == 1


def test_one_and_three_batches_are_the_literal_partition() -> None:
    for L in range(7, 17):
        for nb in (1, 3):
            assert recovered_batches(L, nb) == partition(L, nb)


def test_the_cut_is_per_direction_and_y_differs_from_x_at_10_9() -> None:
    # Table 3 cannot see this (max targets is 39 either way); Tables 1-2 can.
    by_dir = {d: [(a, b) for dd, a, b in recovered_batches(10, 9) if dd == d]
              for d in AXES}
    assert by_dir["x"] == [(0, 3), (4, 5), (6, 10)]
    assert by_dir["y"] == by_dir["z"] == [(0, 4), (5, 5), (6, 10)]


# -- Table 3: found -------------------------------------------------------------


def test_the_cut_reproduces_all_of_table3() -> None:
    got = {(L, nb): max(batch_size(L, b) for b in recovered_batches(L, nb))
           for L, nb in CELLS}
    assert got == dict(T3.cells)


def test_minimax_misses_exactly_the_two_cells_the_rule_recovers() -> None:
    # The lead in #14: the paper's 9-batch cut at L = 7 and 15 is not minimax.
    def mm(L: int, nb: int) -> int:
        return max(batch_size(L, b) for b in partition(L, nb, cut=MINIMAX_TARGETS))
    assert {c for c in CELLS if mm(*c) != T3.cells[c]} == {(7, 9), (15, 9)}


def test_table3_exact_given_inputs_passes_with_the_recorded_inputs(
    candidates: dict[str, Candidate],
) -> None:
    board = load_scoreboard(DATA / "scoreboards" / "jengine" / "literal.json",
                            "literal")
    (rule,) = [r for r in board.rules[T3.id] if isinstance(r, ExactGivenInputs)]
    assert dict(rule.inputs) == dict(CUT_INPUTS)
    (verdict,) = score(Scoreboard("literal", {T3.id: (rule,)}),
                       {T3.id: T3}, {T3.id: candidates[T3.id]})
    assert verdict.status is Status.PASS


def _exact_limits():
    """limits.json's records for the literal board's exact_given_inputs (#14);
    the searched board's at_or_under record (#13) is test_scoreboards.py's."""
    return [
        lim for lim in load_limits(JENGINE / "limits.json")
        if (lim.asserted_value["board"], lim.asserted_value["rule"])
        == ("literal", "exact_given_inputs")
    ]


def test_tables_1_and_2_record_no_inputs_and_a_limit_each() -> None:
    board = load_scoreboard(DATA / "scoreboards" / "jengine" / "literal.json",
                            "literal")
    for tid in ("jengine/table1", "jengine/table2"):
        (rule,) = [r for r in board.rules[tid] if isinstance(r, ExactGivenInputs)]
        assert rule.inputs is None
        assert rule.refused is not None  # #108: refused, not left null
    limited = [lim.asserted_value["table"] for lim in _exact_limits()]
    assert limited == ["jengine/table1", "jengine/table2"]
    assert {lim.asserted_value["doctrine"] for lim in _exact_limits()} == {
        "rules-only"
    }


# -- Tables 1-2: what the recovered cut moves, and the limit -------------------


def test_the_recovered_cut_reproduces_all_three_tables_at_7_9_and_10_9() -> None:
    for L, nb in ((7, 9), (10, 9)):
        c = cost(recovered_plan(L, nb))
        assert (c.flops, c.max_intermediates, c.max_targets) == tuple(
            TABLES[f"jengine/table{n}"].cells[L, nb] for n in (1, 2, 3)
        )


def test_the_limits_name_exactly_the_cells_still_unreproduced(
    candidates: dict[str, Candidate],
) -> None:
    for lim in _exact_limits():
        tid = lim.asserted_value["table"]
        table, cand = TABLES[tid], candidates[tid]
        wrong = sorted(c for c, p in table.cells.items() if cand.cells[c] != p)
        assert [list(c) for c in wrong] == lim.asserted_value["unreproduced_cells"]


def test_the_bounded_seed_order_search() -> None:
    # The structured family limits.json reports: 26 orders, best 17/40 on
    # Table 1 and 29/40 on Table 2 (different orders), and Table 3 never moves.
    t12 = {k: TABLES[k] for k in ("jengine/table1", "jengine/table2")}
    results = {name: exact_counts(TABLES, recovered_costs(CELLS, order))
               for name, order in seed_orders()}
    assert len(results) == 26
    assert all(e["jengine/table3"] == 40 for e in results.values())
    assert max(e["jengine/table1"] for e in results.values()) == 17
    assert max(e["jengine/table2"] for e in results.values()) == 29
    assert exact_counts(t12, recovered_costs(CELLS)) == {
        "jengine/table1": 9, "jengine/table2": 17,
    }
    assert results["hermite"] == exact_counts(TABLES, recovered_costs(CELLS))


# -- #108: the widened family, and the doctrine ---------------------------------

BEST_WIDENED = "+max/-mid, ties reversed"
QUANTITY = {"jengine/table1": "flops", "jengine/table2": "max_intermediates"}


def test_widening_moves_no_default() -> None:
    # The default family is still #13's 26 (plan/search.py is pinned to it);
    # widened=True appends 32 rules after them, in the same order.
    base = [name for name, _ in seed_orders()]
    wide = [name for name, _ in seed_orders(widened=True)]
    assert len(base) == 26 and len(wide) == 58 == len(set(wide))
    assert wide[:26] == base
    assert BEST_WIDENED in wide[26:]


@pytest.fixture(scope="module")
def best_widened_costs():
    order = dict(seed_orders(widened=True))[BEST_WIDENED]
    return recovered_costs(CELLS, order)


def test_the_best_widened_rule_and_its_residual(best_widened_costs) -> None:
    # limits.json's (5): 33/40 and 38/40, Table 3 untouched, and a residual
    # with one sign per table -- every Table 1 miss over, every Table 2 miss
    # under -- which is what argues against fitting the seed order per cell.
    c = best_widened_costs
    assert exact_counts(TABLES, c) == {
        "jengine/table1": 33, "jengine/table2": 38, "jengine/table3": 40,
    }
    sign = {"jengine/table1": 1, "jengine/table2": -1}
    board = load_scoreboard(DATA / "scoreboards" / "jengine" / "literal.json",
                            "literal")
    for lim in _exact_limits():
        tid, wide = lim.asserted_value["table"], lim.asserted_value["widened"]
        miss = sorted(cl for cl, p in TABLES[tid].cells.items()
                      if getattr(c[cl], QUANTITY[tid]) != p)
        assert wide["best"] == BEST_WIDENED and wide["orders"] == 58
        assert wide["exact"] == 40 - len(miss)
        assert [list(cl) for cl in miss] == wide["unreproduced_cells"]
        for cl in miss:
            diff = getattr(c[cl], QUANTITY[tid]) - TABLES[tid].cells[cl]
            assert diff * sign[tid] > 0, (tid, cl, diff)
        (rule,) = [r for r in board.rules[tid] if isinstance(r, ExactGivenInputs)]
        assert f"{wide['exact']} of 40" in rule.refused
        assert "58 seed-order rules" in rule.refused


def test_the_batching_exceptions_under_the_widened_family(
    best_widened_costs,
) -> None:
    # claims.json's jengine/batching-trades-flops-for-intermediates. Under the
    # best widened rule: L = 7 FLOPs rise 1 -> 3 as printed (424, 425), but
    # L = 10's intermediates stay flat 6 -> 9 (95, 95) and L = 7's RISE 6 -> 9
    # (39, 40) -- and the paper's own Table 2 prints exactly those two.
    c, t1, t2 = best_widened_costs, *(TABLES[f"jengine/table{n}"] for n in (1, 2))
    assert (c[7, 1].flops, c[7, 3].flops) == (t1.cells[7, 1], t1.cells[7, 3])
    assert (t1.cells[7, 1], t1.cells[7, 3]) == (424, 425)
    for cells, printed in ((((10, 6), (10, 9)), (95, 95)),
                           (((7, 6), (7, 9)), (39, 40))):
        assert tuple(c[cl].max_intermediates for cl in cells) == printed
        assert tuple(t2.cells[cl] for cl in cells) == printed


@pytest.mark.slow
def test_the_widened_seed_order_search() -> None:
    # Every number limits.json's (5) and claims.json quote about the 58
    # orders, recomputed (about 30 s, hence slow; the best rule's numbers
    # are checked in the fast tier above).
    costs = {name: recovered_costs(CELLS, order)
             for name, order in seed_orders(widened=True)}
    results = {name: exact_counts(TABLES, c) for name, c in costs.items()}
    assert all(e["jengine/table3"] == 40 for e in results.values())
    for tid, best in (("jengine/table1", 33), ("jengine/table2", 38)):
        assert max(e[tid] for e in results.values()) == best
        assert [n for n, e in results.items() if e[tid] == best] == [BEST_WIDENED]
    assert sum(c[7, 1].flops == 424 for c in costs.values()) == 37
    assert {(c[10, 6].max_intermediates, c[10, 9].max_intermediates)
            for c in costs.values()} == {(95, 95)}
    assert {(c[7, 6].max_intermediates, c[7, 9].max_intermediates)
            for c in costs.values()} == {(39, 40)}


# -- nothing literal moves -------------------------------------------------------


@pytest.mark.parametrize("nb", [1, 3, 6, 9])
@pytest.mark.parametrize("L", [7, 12, 16])
def test_the_seed_order_default_is_the_literal_sieve(L: int, nb: int) -> None:
    spec = Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, n_batches=nb)
    for d, lit in zip(build(spec), plan(spec), strict=True):
        explicit = sieve(d, seed_order=hermite_targets(L))
        assert dict(explicit.parents) == dict(lit.parents)
