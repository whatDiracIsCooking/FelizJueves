"""The scoring rules, on synthetic candidates.

No generator exists, so nothing real is scored yet. What can be tested is the
rules themselves: each must reject a regression, accept an improvement, and
treat an equal result as passing -- and report UNSET, not a guess, when its
threshold has not been measured.
"""

from __future__ import annotations

import dataclasses
import math
from types import MappingProxyType

import pytest

from felizjueves.literature.rules import (
    Agreement,
    AtOrUnder,
    Candidate,
    ExactGivenInputs,
    Status,
)
from felizjueves.literature.tables import Table

PRINTED = {(7, 1): 100, (7, 3): 200, (8, 1): 400, (8, 3): 1000}


def _table() -> Table:
    return Table(
        id="synthetic/t",
        source=MappingProxyType({}),
        quantity=MappingProxyType({}),
        row_name="L",
        row_values=(7, 8),
        column_name="batches",
        column_values=(1, 3),
        cells=MappingProxyType(PRINTED),
        notes=(),
    )


def _cand(**overrides: float) -> Candidate:
    # overrides keyed "L_b", e.g. _cand(**{"7_1": 101})
    cells = dict(PRINTED)
    for k, v in overrides.items():
        L, b = map(int, k.split("_"))
        cells[(L, b)] = v
    return Candidate(cells, board="synthetic")


T = _table()

# -- Agreement ---------------------------------------------------------------

# A floor measured at 2 exact cells and a worst deviation of 5%:
# (7,1) exact, (7,3) exact, (8,1) off by 5%, (8,3) off by 2%.
AT_FLOOR = _cand(**{"8_1": 420, "8_3": 1020})
RULE = Agreement(floor_exact=2, ceiling_worst_rel_dev=0.05)


def test_agreement_equal_passes() -> None:
    v = RULE.judge(T, AT_FLOOR)
    assert v.status is Status.PASS
    assert v.measured == {"exact": 2, "worst_rel_dev": pytest.approx(0.05)}


def test_agreement_accepts_improvement() -> None:
    assert RULE.judge(T, _cand(**{"8_1": 410})).status is Status.PASS
    assert RULE.judge(T, _cand()).status is Status.PASS


def test_agreement_rejects_fewer_exact() -> None:
    v = RULE.judge(T, _cand(**{"7_1": 101, "8_1": 420, "8_3": 1020}))
    assert v.status is Status.FAIL and "floor" in v.reason
    assert (7, 1) in v.cells


def test_agreement_rejects_worse_deviation() -> None:
    v = RULE.judge(T, _cand(**{"8_1": 440, "8_3": 1020}))
    assert v.status is Status.FAIL and "ceiling" in v.reason


def test_agreement_unset_reports_without_judging() -> None:
    v = Agreement().judge(T, _cand(**{"8_1": 9999}))
    assert v.status is Status.UNSET
    assert v.measured["exact"] == 3


def test_agreement_half_set_still_fails_on_the_set_half() -> None:
    v = Agreement(floor_exact=4).judge(T, AT_FLOOR)
    assert v.status is Status.FAIL
    assert Agreement(floor_exact=2).judge(T, AT_FLOOR).status is Status.UNSET


def test_tightened_only_moves_one_way() -> None:
    r = Agreement().tightened(floor_exact=2, ceiling_worst_rel_dev=0.05)
    assert r == RULE
    assert r.tightened(floor_exact=3) == Agreement(3, 0.05)
    assert r.tightened(floor_exact=2) == r  # equal is not a loosening
    with pytest.raises(ValueError, match="only rise"):
        r.tightened(floor_exact=1)
    with pytest.raises(ValueError, match="only fall"):
        r.tightened(ceiling_worst_rel_dev=0.06)


def test_rules_are_frozen() -> None:
    with pytest.raises(AttributeError):
        RULE.floor_exact = 0  # type: ignore[misc]


# -- AtOrUnder ---------------------------------------------------------------


def test_at_or_under_equal_passes() -> None:
    assert AtOrUnder().judge(T, _cand()).status is Status.PASS


def test_at_or_under_accepts_improvement() -> None:
    assert AtOrUnder().judge(T, _cand(**{"8_3": 900})).status is Status.PASS


def test_at_or_under_rejects_regression() -> None:
    v = AtOrUnder().judge(T, _cand(**{"8_3": 900, "7_3": 201}))
    assert v.status is Status.FAIL and v.cells == ((7, 3),)


def test_at_or_under_reports_the_margin_per_cell_and_worst() -> None:
    v = AtOrUnder().judge(T, _cand(**{"7_1": 101, "7_3": 230, "8_3": 900}))
    assert v.status is Status.FAIL
    assert v.cells == ((7, 1), (7, 3))
    assert v.measured["over"] == 2
    assert v.measured["overshoot"] == (((7, 1), 0.01), ((7, 3), 0.15))
    assert v.measured["worst_overshoot"] == 0.15


def test_at_or_under_a_tiny_margin_still_fails() -> None:
    # No tolerance: the margin is reported, never judged.
    v = AtOrUnder().judge(T, _cand(**{"8_3": 1000.001}))
    assert v.status is Status.FAIL and v.cells == ((8, 3),)
    assert 0 < v.measured["worst_overshoot"] < 1e-5


def test_at_or_under_zero_overshoot_when_every_cell_is_equal() -> None:
    v = AtOrUnder().judge(T, _cand())
    assert v.status is Status.PASS
    assert v.measured == {"over": 0, "worst_overshoot": 0.0, "overshoot": ()}


def test_at_or_under_zero_overshoot_when_every_cell_is_under() -> None:
    v = AtOrUnder().judge(
        T, _cand(**{"7_1": 50, "7_3": 199, "8_1": 1, "8_3": 999})
    )
    assert v.status is Status.PASS
    assert v.measured == {"over": 0, "worst_overshoot": 0.0, "overshoot": ()}


def test_at_or_under_over_a_zero_printed_cell_is_unbounded() -> None:
    zero = dataclasses.replace(T, cells=MappingProxyType({**PRINTED, (7, 1): 0}))
    v = AtOrUnder().judge(zero, _cand(**{"7_1": 1}))
    assert v.status is Status.FAIL
    assert v.measured["worst_overshoot"] == math.inf
    assert AtOrUnder().judge(zero, _cand(**{"7_1": 0})).status is Status.PASS


# -- ExactGivenInputs --------------------------------------------------------

INPUTS = {"split": "paper"}


def _with_inputs(c: Candidate, inputs: dict) -> Candidate:
    return Candidate(c.cells, board=c.board, inputs=inputs)


def test_exact_equal_passes() -> None:
    r = ExactGivenInputs(INPUTS)
    assert r.judge(T, _with_inputs(_cand(), INPUTS)).status is Status.PASS


def test_exact_rejects_any_difference() -> None:
    r = ExactGivenInputs(INPUTS)
    # An "improvement" is still a disagreement: given the paper's inputs, the
    # paper's number is the only right answer.
    for value in (99, 101):
        v = r.judge(T, _with_inputs(_cand(**{"7_1": value}), INPUTS))
        assert v.status is Status.FAIL and v.cells == ((7, 1),)


def test_exact_other_inputs_are_out_of_scope() -> None:
    r = ExactGivenInputs(INPUTS)
    v = r.judge(T, _with_inputs(_cand(**{"7_1": 1}), {"split": "searched"}))
    assert v.status is Status.INAPPLICABLE


def test_exact_unrecorded_inputs_is_unset() -> None:
    assert ExactGivenInputs().judge(T, _cand()).status is Status.UNSET


def test_exact_recorded_inputs_are_read_only() -> None:
    r = ExactGivenInputs(dict(INPUTS))
    with pytest.raises(TypeError):
        r.inputs["split"] = "other"  # type: ignore[index]


REFUSAL = "rules only (#108): no rule reproduces the table; see limits.json"


def test_a_refusal_is_unset_and_says_why() -> None:
    # #108: a refusal is recorded, but nothing was reproduced, so it is
    # never PASS -- not even for a candidate whose every cell is exact.
    r = ExactGivenInputs(refused=REFUSAL)
    for cand in (_cand(), _with_inputs(_cand(), INPUTS)):
        v = r.judge(T, cand)
        assert v.status is Status.UNSET
        assert v.reason.startswith("refused")
        assert v.measured == {"refused": REFUSAL}


def test_a_refusal_and_inputs_cannot_coexist() -> None:
    with pytest.raises(ValueError, match="not both"):
        ExactGivenInputs(INPUTS, refused=REFUSAL)


@pytest.mark.parametrize("why", ["", "   "])
def test_a_refusal_needs_a_reason(why: str) -> None:
    with pytest.raises(ValueError, match="say why"):
        ExactGivenInputs(refused=why)


# -- coverage, for every rule ------------------------------------------------


@pytest.mark.parametrize(
    "rule", [Agreement(0, 1.0), AtOrUnder(), ExactGivenInputs({})]
)
def test_missing_or_extra_cells_fail(rule) -> None:
    missing = Candidate(
        {k: v for k, v in PRINTED.items() if k != (8, 3)}, board="synthetic"
    )
    extra = Candidate({**PRINTED, (9, 1): 1}, board="synthetic")
    for c in (missing, extra):
        v = rule.judge(T, c)
        assert v.status is Status.FAIL, (rule, v)


def test_a_zero_printed_cell_is_named_not_divided_by() -> None:
    zero = dataclasses.replace(T, cells=MappingProxyType({**PRINTED, (7, 1): 0}))
    with pytest.raises(ValueError, match=r"synthetic/t cell \(7, 1\) is printed as 0"):
        Agreement().judge(zero, _cand(**{"7_1": 0}))
