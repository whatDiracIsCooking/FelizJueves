"""The two [JEngine] scoreboards: same tables, separate thresholds.

Only Table 3's literal Agreement thresholds are set; Tables 1 and 2 are scored
pre-sieve and their thresholds wait for the sieve."""

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
from felizjueves.plan.dag import build
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


def test_no_floor_is_set_before_the_sieve() -> None:
    # Tables 1 and 2 are computed by the plan layer, but PRE-sieve (plan/dag.py
    # builds MILD's DAG and nothing prunes it), and they are post-sieve
    # numbers. The sieve can move a cell either way -- a re-expansion that
    # deletes an intermediate can cost FLOPs -- so a floor set now could be
    # one the sieve has to lower, and a floor never falls. Their thresholds
    # stay unset until the sieve (#12) measures them. Table 3 is targets
    # only, which the sieve does not touch, and has its measured Agreement
    # floor and ceiling; every exact_given_inputs is still unrecorded (#14).
    for tid, rules in _board("literal").rules.items():
        for rule in rules:
            if tid == "jengine/table3" and isinstance(rule, Agreement):
                continue
            if isinstance(rule, Agreement):
                assert rule.floor_exact is None
                assert rule.ceiling_worst_rel_dev is None
            else:
                assert rule.inputs is None


def test_the_printed_tables_score_as_unset_and_at_or_under() -> None:
    # Scoring the paper against itself: the literal board has nothing to judge
    # against except Table 3's Agreement, which the printed values pass (40
    # exact, no deviation); the searched board's rule holds with equality.
    literal = score(_board("literal"), TABLES, _as_printed("literal"))
    passed = {(v.table, v.rule) for v in literal if v.status is Status.PASS}
    assert passed == {("jengine/table3", "agreement")}
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
    """Tables 1 and 2 from the pre-sieve PlanCost, Table 3 as above."""
    t1 = TABLES["jengine/table1"]
    costs = {
        (L, nb): cost(build(Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J,
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


def test_tables_1_and_2_score_pre_sieve_on_the_literal_board() -> None:
    # What the pre-sieve plan measures, pinned so a change to it is a diff
    # here. Exact cells -- Table 1: (7,3) (7,6) (8,6) (8,9) (11,6); Table 2:
    # (7,6) (8,9) (10,6) (10,9). Worst -- Table 1: (7,9), 510 against 560,
    # where the paper's 9-batch partition is not ours (Table 3 differs there
    # too); Table 2: (16,1), 2021 against 1647, the cell with most to prune.
    by_table = {
        (v.table, v.rule): v
        for v in score(_board("literal"), TABLES, _plan_candidates())
    }
    t1 = by_table["jengine/table1", "agreement"]
    t2 = by_table["jengine/table2", "agreement"]
    assert t1.status is t2.status is Status.UNSET  # thresholds null until #12
    assert t1.measured == {"exact": 5, "worst_rel_dev": 50 / 560}
    assert t2.measured == {"exact": 4, "worst_rel_dev": (2021 - 1647) / 1647}
    assert by_table["jengine/table3", "agreement"].status is Status.PASS


def test_the_full_board_with_the_plan_layers_candidates() -> None:
    # Every table now has a candidate, so nothing FAILs for want of one: Table
    # 3's Agreement passes, and everything else is UNSET.
    verdicts = score(_board("literal"), TABLES, _plan_candidates())
    assert {(v.table, v.rule) for v in verdicts if v.status is Status.PASS} == {
        ("jengine/table3", "agreement")
    }
    assert {v.status for v in verdicts} == {Status.PASS, Status.UNSET}


def test_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        TABLES["jengine/table1"].cells[(7, 1)] = 0  # type: ignore[index]


def test_limits_record_exists_and_is_empty() -> None:
    assert load_limits(JENGINE / "limits.json") == ()


def test_a_limit_record_needs_every_field() -> None:
    StructuralLimit("c", "m", "e", 1, "[JEngine, Table 1]")
    with pytest.raises(ValueError, match="mechanism"):
        StructuralLimit("c", "", "e", 1, "[JEngine, Table 1]")
