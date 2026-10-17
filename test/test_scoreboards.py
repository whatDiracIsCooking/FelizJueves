"""The two [JEngine] scoreboards: same tables, separate thresholds.

Every literal Agreement threshold is set, each to what the plan layer
measured: Table 3 from the batch partition, Tables 1 and 2 from the sieved
plan. Table 3's exact_given_inputs records the recovered cut (#14); Tables
1 and 2's are refused under the rules-only doctrine (#108), each refusal
naming the StructuralLimit in limits.json that holds the evidence.

Every check that plans the tables' cells (L = 7..16: the partition, the
sieved plan, the searched replay) is `slow`, above `tiers.FAST_L_MAX`; the
boards scored against the printed values stay fast."""

from __future__ import annotations

import json
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
from felizjueves.plan.search import SEED_ORDERS, load_cuts, searched_plan
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


def test_every_agreement_is_set_and_only_table3_records_inputs() -> None:
    # Tables 1 and 2 are computed post-sieve (#12), the paper's algorithm as
    # written, and their floors and ceilings are what that measured; Table 3's
    # are the batch partition's. From here a floor only rises and a ceiling
    # only falls (Agreement.tightened). #14 recorded Table 3's inputs (the
    # cut, plan/recovery.py); Tables 1 and 2 have none, and #108 refused them
    # under the rules-only doctrine: no slot is left null and silent.
    for tid, rules in _board("literal").rules.items():
        for rule in rules:
            if isinstance(rule, Agreement):
                assert rule.floor_exact is not None
                assert rule.ceiling_worst_rel_dev is not None
            elif tid == "jengine/table3":
                assert rule.inputs is not None and rule.refused is None
            else:
                assert rule.inputs is None
                assert rule.refused is not None
                assert "rules only" in rule.refused
                assert "limits.json" in rule.refused


def test_the_refusals_score_unset_with_their_reason() -> None:
    # #108: a refusal is not a pass. The printed values themselves -- every
    # cell exact -- still score UNSET on Tables 1-2, carrying the refusal.
    refusals = [
        v for v in score(_board("literal"), TABLES, _as_printed("literal"))
        if v.rule == "exact_given_inputs" and v.table != "jengine/table3"
    ]
    assert len(refusals) == 2
    for v in refusals:
        assert v.status is Status.UNSET
        assert v.reason.startswith("refused")
        assert "rules only" in v.measured["refused"]


def test_the_printed_tables_score_as_unset_and_at_or_under() -> None:
    # Scoring the paper against itself: on the literal board every Agreement
    # passes (40 exact, no deviation); exact_given_inputs is unset on Tables
    # 1-2 and inapplicable on Table 3 (the printed values were not produced
    # from its recorded inputs by this candidate);
    # the searched board's rule holds with equality.
    literal = score(_board("literal"), TABLES, _as_printed("literal"))
    passed = {(v.table, v.rule) for v in literal if v.status is Status.PASS}
    assert passed == {(tid, "agreement") for tid in TABLES}
    assert {v.status for v in literal} == {
        Status.UNSET, Status.PASS, Status.INAPPLICABLE
    }
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


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_table3_scores_on_the_literal_board() -> None:
    t3 = TABLES["jengine/table3"]
    board = _board("literal")
    only_t3 = Scoreboard(board.name, {t3.id: board.rules[t3.id]})
    agreement, exact_given_inputs = score(
        only_t3, {t3.id: t3}, {t3.id: _table3_candidate()}
    )
    assert agreement.status is Status.PASS
    assert agreement.measured == {"exact": 38, "worst_rel_dev": 1 / 21}
    # The minimax cut is not the recorded one (#14: two-ended nearest fill).
    assert exact_given_inputs.status is Status.INAPPLICABLE


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


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
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


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_full_board_with_the_plan_layers_candidates() -> None:
    # Every table has a candidate, so nothing FAILs for want of one: every
    # Agreement passes; exact_given_inputs is UNSET on Tables 1-2 and
    # INAPPLICABLE on Table 3, whose recorded inputs these are not.
    verdicts = score(_board("literal"), TABLES, _plan_candidates())
    assert {(v.table, v.rule) for v in verdicts if v.status is Status.PASS} == {
        (tid, "agreement") for tid in TABLES
    }
    assert {v.status for v in verdicts} == {
        Status.PASS, Status.UNSET, Status.INAPPLICABLE
    }


def _searched_candidates(path: Path = BOARDS / "searched_cuts.json"
                         ) -> dict[str, Candidate]:
    """One plan per cell, replayed from the recorded cut and seed order
    (plan/search.py), scores all three tables."""
    cuts = load_cuts(path)
    costs = {c: cost(searched_plan(*c, *choice)) for c, choice in cuts.items()}
    return {
        tid: Candidate({c: getattr(k, attr) for c, k in costs.items()},
                       board="searched")
        for tid, attr in (("jengine/table1", "flops"),
                          ("jengine/table2", "max_intermediates"),
                          ("jengine/table3", "max_targets"))
    }


def _searched_limit() -> StructuralLimit:
    (lim,) = [
        lim for lim in load_limits(JENGINE / "limits.json")
        if (lim.asserted_value["board"], lim.asserted_value["rule"])
        == ("searched", "at_or_under")
    ]
    return lim


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_searched_board_with_the_searched_plans() -> None:
    # #13's outcome, pinned so a change to it is a diff here. Since #172
    # (explicit per-batch seed orders, climbed from the 58 rules) all three
    # tables PASS at_or_under in every cell, and the StructuralLimit records
    # no residual. (Under 26 orders it was 21 and 3 cells over, within
    # +2.04%; under #168's lexicographic objective 9 cells of Table 1; under
    # #171's frontier over the 58 rules 8, within +0.89% -- the record's
    # previous_residual.)
    by_table = {
        v.table: v
        for v in score(_board("searched"), TABLES, _searched_candidates())
    }
    for n in (1, 2, 3):
        t = by_table[f"jengine/table{n}"]
        assert t.status is Status.PASS, t
        assert t.measured["over"] == 0 and t.measured["worst_overshoot"] == 0.0
    value = _searched_limit().asserted_value
    assert value["residual_percent_over"] == {}
    assert value["unreachable_in_space"] == [] and value["selection_price"] == []


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_searched_board_scores_the_replay_not_the_records_counts(
    tmp_path: Path,
) -> None:
    # #173: the counts in searched_cuts.json are not taken on trust. Doctor
    # every one of them -- each frontier point claims the printed triple's
    # half, which would PASS at_or_under by itself -- and the board's
    # candidates are unchanged, because they are recomputed from the
    # recorded permutations, not read.
    doc = json.loads((BOARDS / "searched_cuts.json").read_text())
    for key, cell in doc["cuts"].items():
        L, nb = (int(x) for x in key.split("/"))
        for p in cell["frontier"]:
            p["flops"] = TABLES["jengine/table1"].cells[L, nb] // 2
            p["max_intermediates"] = TABLES["jengine/table2"].cells[L, nb] // 2
            p["max_targets"] = TABLES["jengine/table3"].cells[L, nb] // 2
    bad = tmp_path / "searched_cuts.json"
    bad.write_text(json.dumps(doc))
    genuine = _searched_candidates()
    doctored = _searched_candidates(bad)
    assert {t: dict(c.cells) for t, c in doctored.items()} == {
        t: dict(c.cells) for t, c in genuine.items()}
    assert doctored["jengine/table3"].cells[15, 9] == 116  # not 58


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_previous_residual_cells_are_reached_only_by_the_climb() -> None:
    # The limit's previous_residual, re-checked where it is cheap: in its
    # 1- and 3-batch cells there is no cut, so #171's space was the 58 seed
    # orders alone, and every one of them is still over Table 1 there --
    # while the recorded climbed plan is at or under all three tables. The
    # climb, not a change of rule, is what moved these cells.
    prev = _searched_limit().asserted_value["previous_residual"]
    cells = {(L, nb) for L, nb, _ in prev["jengine/table1"] if nb in (1, 3)}
    assert cells == {(13, 3), (14, 1), (14, 3), (15, 1), (15, 3), (16, 1),
                     (16, 3)}
    assert len(SEED_ORDERS) == 58
    printed = [TABLES[f"jengine/table{n}"] for n in (1, 2, 3)]
    recorded = load_cuts(BOARDS / "searched_cuts.json")
    for L, nb in sorted(cells - {(14, 1), (14, 3)}):
        for seed in SEED_ORDERS:
            c = cost(searched_plan(L, nb, (), seed))
            assert c.flops > printed[0].cells[L, nb], (L, nb, seed)
    for L, nb in sorted(cells):
        c = cost(searched_plan(L, nb, *recorded[L, nb]))
        got = (c.flops, c.max_intermediates, c.max_targets)
        assert all(g <= t.cells[L, nb] for g, t in zip(got, printed, strict=True))


def test_tables_are_read_only() -> None:
    with pytest.raises(TypeError):
        TABLES["jengine/table1"].cells[(7, 1)] = 0  # type: ignore[index]


def test_limits_record_tables_1_and_2() -> None:
    # #14: the sieve's seed order is not recovered for Tables 1-2, so each
    # carries a StructuralLimit (test_plan_recovery.py checks its cells, and
    # #108's widened family and doctrine);
    # #13: the searched board's residual at_or_under cells carry one more.
    # Only the board-scoped records are this test's: a record about a claim
    # rather than a scoreboard rule (board null or absent; #111's, say) is
    # pinned by its own test, so an unrelated append does not break this one.
    limits = [
        lim for lim in load_limits(JENGINE / "limits.json")
        if isinstance(lim.asserted_value, dict)
        and lim.asserted_value.get("board") is not None
    ]
    assert [
        (lim.asserted_value["board"], lim.asserted_value["rule"],
         lim.asserted_value.get("table")) for lim in limits
    ] == [
        ("literal", "exact_given_inputs", "jengine/table1"),
        ("literal", "exact_given_inputs", "jengine/table2"),
        ("searched", "at_or_under", None),
    ]


def test_a_limit_record_needs_every_field() -> None:
    StructuralLimit("c", "m", "e", 1, "[JEngine, Table 1]")
    with pytest.raises(ValueError, match="mechanism"):
        StructuralLimit("c", "", "e", 1, "[JEngine, Table 1]")
