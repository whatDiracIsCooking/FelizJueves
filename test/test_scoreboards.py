"""The two [JEngine] scoreboards: same tables, separate thresholds.

Every literal Agreement threshold is set, each to what the plan layer
measured: Table 3 from the batch partition, Tables 1 and 2 from the sieved
plan. Every exact_given_inputs is still unrecorded."""

from __future__ import annotations

from pathlib import Path

import pytest

from felizjueves.literature.limits import StructuralLimit, load_limits
from felizjueves.literature.rules import (
    Agreement,
    Candidate,
    ExactGivenInputs,
    Status,
)
from felizjueves.literature.scoreboard import Scoreboard, load_scoreboard, score
from felizjueves.literature.tables import load_table
from felizjueves.plan.batching import max_targets_per_batch
from felizjueves.plan.cost import cost
from felizjueves.plan.search import load_cuts, searched_plan
from felizjueves.plan.sieve import plan
from felizjueves.spec import Operator, Spec, Target

DATA = Path(__file__).resolve().parents[1] / "data"
BOARDS = DATA / "scoreboards" / "jengine"
JENGINE = DATA / "literature" / "jengine"
TABLES = {
    t.id: t for t in (load_table(JENGINE / f"table{n}.json") for n in (1, 2, 3))
}


def _as_printed(board: str) -> dict[str, Candidate]:
    return {tid: Candidate(dict(t.cells), board=board) for tid, t in TABLES.items()}


def _board(name: str):
    return load_scoreboard(BOARDS / f"{name}.json", name)


@pytest.mark.parametrize("name", ["literal", "searched"])
def test_both_boards_score_the_same_tables(name: str) -> None:
    assert set(_board(name).rules) == set(TABLES)


def test_a_board_cannot_load_the_other_boards_file() -> None:
    with pytest.raises(ValueError, match="not 'searched'"):
        load_scoreboard(BOARDS / "literal.json", "searched")
    with pytest.raises(ValueError, match="not 'literal'"):
        load_scoreboard(BOARDS / "searched.json", "literal")


def test_every_agreement_is_set_and_no_input_is_recorded() -> None:
    # Tables 1 and 2 are computed post-sieve (#12), the paper's algorithm as
    # written, and their floors and ceilings are what that measured; Table 3's
    # are the batch partition's. From here a floor only rises and a ceiling
    # only falls (Agreement.tightened). Every exact_given_inputs is still
    # unrecorded (#14).
    for rules in _board("literal").rules.values():
        for rule in rules:
            if isinstance(rule, Agreement):
                assert rule.floor_exact is not None
                assert rule.ceiling_worst_rel_dev is not None
            else:
                assert rule.inputs is None


def test_the_printed_tables_score_as_unset_and_at_or_under() -> None:
    # Scoring the paper against itself: on the literal board every Agreement
    # passes (40 exact, no deviation) and every exact_given_inputs is unset;
    # the searched board's rule holds with equality.
    literal = score(_board("literal"), TABLES, _as_printed("literal"))
    passed = {(v.table, v.rule) for v in literal if v.status is Status.PASS}
    assert passed == {(tid, "agreement") for tid in TABLES}
    assert {v.status for v in literal} == {Status.UNSET, Status.PASS}
    searched = score(_board("searched"), TABLES, _as_printed("searched"))
    assert {v.status for v in searched} == {Status.PASS}


@pytest.mark.parametrize(("board", "produced_for"), [
    ("literal", "searched"),
    ("searched", "literal"),
    ("literal", "synthetic"),
])
def test_a_board_refuses_another_boards_candidate(board: str, produced_for: str):
    with pytest.raises(ValueError, match="refuses candidates produced for"):
        score(_board(board), TABLES, _as_printed(produced_for))


def test_one_foreign_candidate_is_enough_to_refuse() -> None:
    mixed = {**_as_printed("literal")}
    mixed["jengine/table2"] = Candidate(
        dict(TABLES["jengine/table2"].cells), board="searched"
    )
    with pytest.raises(ValueError, match=r"jengine/table2 \(board 'searched'\)"):
        score(_board("literal"), TABLES, mixed)


def test_exact_given_inputs_covers_all_three_tables() -> None:
    for tid, rules in _board("literal").rules.items():
        assert any(isinstance(r, ExactGivenInputs) for r in rules), tid


def test_a_table_not_supplied_is_named() -> None:
    partial = {k: v for k, v in TABLES.items() if k != "jengine/table2"}
    with pytest.raises(ValueError, match="not supplied: jengine/table2"):
        score(_board("searched"), partial, _as_printed("searched"))


def test_a_candidate_not_supplied_fails() -> None:
    cands = {k: v for k, v in _as_printed("searched").items() if k != "jengine/table3"}
    verdicts = score(_board("searched"), TABLES, cands)
    missing = [v for v in verdicts if v.table == "jengine/table3"]
    assert [v.status for v in missing] == [Status.FAIL]
    assert missing[0].reason == "no candidate supplied"


def _table3_candidate() -> Candidate:
    t3 = TABLES["jengine/table3"]
    return Candidate(
        cells={
            (L, nb): max_targets_per_batch(L, nb)
            for L in t3.row_values
            for nb in t3.column_values
        },
        board="literal",
    )


def test_table3_scores_on_the_literal_board() -> None:
    t3 = TABLES["jengine/table3"]
    board = _board("literal")
    only_t3 = Scoreboard(board.name, {t3.id: board.rules[t3.id]})
    agreement, exact_given_inputs = score(
        only_t3, {t3.id: t3}, {t3.id: _table3_candidate()}
    )
    assert agreement.status is Status.PASS
    assert agreement.measured == {"exact": 38, "worst_rel_dev": 1 / 21}
    assert exact_given_inputs.status is Status.UNSET  # inputs still null


def _plan_candidates() -> dict[str, Candidate]:
    """Tables 1 and 2 from the sieved PlanCost (levers=(), the literal
    reconstruction: the paper's algorithm, sieve on), Table 3 as above."""
    t1 = TABLES["jengine/table1"]
    costs = {
        (L, nb): cost(plan(Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J,
                                 n_batches=nb)))
        for L in t1.row_values
        for nb in t1.column_values
    }
    return {
        "jengine/table1": Candidate(
            {c: k.flops for c, k in costs.items()}, board="literal"
        ),
        "jengine/table2": Candidate(
            {c: k.max_intermediates for c, k in costs.items()}, board="literal"
        ),
        "jengine/table3": _table3_candidate(),
    }


def test_tables_1_and_2_score_post_sieve_on_the_literal_board() -> None:
    # What the sieved plan measures, pinned so a change to it is a diff here.
    # Exact cells -- Table 1: (7,3) (7,6) (8,6) (8,9) (9,6) (9,9) (11,9);
    # Table 2: (7,3) (7,6) (8,6) (8,9) (9,6) (9,9) (10,6) (10,9) (11,6)
    # (11,9) (12,6) (12,9) (13,9) (14,9) (15,3). Pre-sieve these were 5 and 4;
    # Table 1's (11,6) was exact pre-sieve (2015) and the sieve moves it off
    # (2016). Worst -- Table 1: (7,9), 510 against 560, unchanged by the
    # sieve, where the paper's 9-batch partition is not ours (Table 3 differs
    # there too); Table 2: (15,9), 301 against 280.
    by_table = {
        (v.table, v.rule): v
        for v in score(_board("literal"), TABLES, _plan_candidates())
    }
    t1 = by_table["jengine/table1", "agreement"]
    t2 = by_table["jengine/table2", "agreement"]
    assert t1.status is t2.status is Status.PASS  # at exactly the thresholds
    assert t1.measured == {"exact": 7, "worst_rel_dev": 50 / 560}
    assert t2.measured == {"exact": 15, "worst_rel_dev": (301 - 280) / 280}
    assert by_table["jengine/table3", "agreement"].status is Status.PASS


def test_the_full_board_with_the_plan_layers_candidates() -> None:
    # Every table has a candidate, so nothing FAILs for want of one: every
    # Agreement passes, and every exact_given_inputs is UNSET.
    verdicts = score(_board("literal"), TABLES, _plan_candidates())
    assert {(v.table, v.rule) for v in verdicts if v.status is Status.PASS} == {
        (tid, "agreement") for tid in TABLES
    }
    assert {v.status for v in verdicts} == {Status.PASS, Status.UNSET}


def _searched_candidates() -> dict[str, Candidate]:
    """One plan per cell, replayed from the recorded cut (plan/search.py),
    scores all three tables."""
    cuts = load_cuts(BOARDS / "searched_cuts.json")
    costs = {c: cost(searched_plan(*c, cut)) for c, cut in cuts.items()}
    return {
        tid: Candidate({c: getattr(k, attr) for c, k in costs.items()},
                       board="searched")
        for tid, attr in (("jengine/table1", "flops"),
                          ("jengine/table2", "max_intermediates"),
                          ("jengine/table3", "max_targets"))
    }


def test_the_searched_board_with_the_searched_plans() -> None:
    # #13's outcome, pinned so a change to it is a diff here. Table 3 passes:
    # the objective puts minimax targets first, and minimax is at or under
    # the paper everywhere. Tables 1 and 2 FAIL at_or_under, and these are the
    # cells. Twenty of each table's are 1- or 3-batch cells, where there is
    # no cut to search and the plan is the literal one. Of the 6- and 9-batch
    # cells still over, no shared cut, sieve on or off, is at or under in all
    # three tables, except (15, 9), where the one that is is not minimax
    # (test_plan_search.py).
    by_table = {
        v.table: v
        for v in score(_board("searched"), TABLES, _searched_candidates())
    }
    assert by_table["jengine/table3"].status is Status.PASS
    t1, t2 = by_table["jengine/table1"], by_table["jengine/table2"]
    assert t1.status is t2.status is Status.FAIL
    one_or_three = {(L, nb) for L in range(7, 17) for nb in (1, 3)}
    assert set(t1.cells) - one_or_three == {
        (10, 6), (11, 6), (12, 6), (12, 9), (13, 6), (13, 9),
        (14, 6), (14, 9), (15, 6), (16, 6), (16, 9),
    }
    assert set(t1.cells) & one_or_three == one_or_three - {(7, 3)}
    assert t1.measured == {"over": 30}
    assert set(t2.cells) == {
        (7, 1), (8, 1), (8, 3), (9, 1), (9, 3), (10, 1), (10, 3),
        (11, 1), (11, 3), (12, 1), (12, 3), (13, 1), (15, 9),
    }
    assert t2.measured == {"over": 13}


def test_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        TABLES["jengine/table1"].cells[(7, 1)] = 0  # type: ignore[index]


def test_limits_record_tables_1_and_2() -> None:
    # #14: the sieve's seed order is not recovered for Tables 1-2, so each
    # carries a StructuralLimit (test_plan_recovery.py checks its cells).
    limits = load_limits(JENGINE / "limits.json")
    assert [lim.asserted_value["table"] for lim in limits] == [
        "jengine/table1", "jengine/table2",
    ]


def test_a_limit_record_needs_every_field() -> None:
    StructuralLimit("c", "m", "e", 1, "[JEngine, Table 1]")
    with pytest.raises(ValueError, match="mechanism"):
        StructuralLimit("c", "", "e", 1, "[JEngine, Table 1]")
