"""The two [JEngine] scoreboards: same tables, separate thresholds, none set."""

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


def test_no_floor_is_set_yet() -> None:
    # Nothing in the repo computes a cell, so every threshold must be unset.
    # The first change that produces numbers fills these with what it measured.
    for rules in _board("literal").rules.values():
        for rule in rules:
            if isinstance(rule, Agreement):
                assert rule.floor_exact is None
                assert rule.ceiling_worst_rel_dev is None
            else:
                assert rule.inputs is None


def test_the_printed_tables_score_as_unset_and_at_or_under() -> None:
    # Scoring the paper against itself: the literal board has nothing to judge
    # against, and the searched board's rule holds with equality.
    literal = score(_board("literal"), TABLES, _as_printed("literal"))
    assert {v.status for v in literal} == {Status.UNSET}
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
    assert agreement.status is Status.UNSET  # until the floor is set
    assert agreement.measured["exact"] == 38
    assert exact_given_inputs.status is Status.UNSET  # inputs still null


def test_the_full_board_fails_while_two_tables_have_no_candidate() -> None:
    # A partial run cannot be told apart from a regression: the plan layer
    # computes Table 3 only, so Tables 1 and 2 are FAILs by design.
    verdicts = score(
        _board("literal"), TABLES, {"jengine/table3": _table3_candidate()}
    )
    failed = {v.table for v in verdicts if v.status is Status.FAIL}
    assert failed == {"jengine/table1", "jengine/table2"}
    assert all(
        v.reason == "no candidate supplied"
        for v in verdicts
        if v.table != "jengine/table3"
    )


def test_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        TABLES["jengine/table1"].cells[(7, 1)] = 0  # type: ignore[index]


def test_limits_record_exists_and_is_empty() -> None:
    assert load_limits(JENGINE / "limits.json") == ()


def test_a_limit_record_needs_every_field() -> None:
    StructuralLimit("c", "m", "e", 1, "[JEngine, Table 1]")
    with pytest.raises(ValueError, match="mechanism"):
        StructuralLimit("c", "", "e", 1, "[JEngine, Table 1]")
