"""[JEngine] Tables 1-3, checked against their own structure.

There is no second copy of these numbers to diff against, so the tables'
internal invariants are the check on the transcription. A failure here is a
misread digit until shown otherwise -- not a finding about the paper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from felizjueves.literature.tables import Table, load_table

JENGINE = Path(__file__).resolve().parents[1] / "data" / "literature" / "jengine"
LS = tuple(range(7, 17))
BATCHES = (1, 3, 6, 9)


@pytest.fixture(scope="module", params=[1, 2, 3])
def table(request: pytest.FixtureRequest) -> Table:
    return load_table(JENGINE / f"table{request.param}.json")


def _t(n: int) -> Table:
    return load_table(JENGINE / f"table{n}.json")


def _row(t: Table, L: int) -> list[int]:
    return [t.cells[(L, b)] for b in BATCHES]


def test_shape_and_provenance(table: Table) -> None:
    assert table.row_name == "L" and table.row_values == LS
    assert table.column_name == "batches" and table.column_values == BATCHES
    assert len(table.cells) == 40
    assert table.source["key"] == "[JEngine]"
    assert table.source["references_md_entry"] == 35
    assert table.id == f"jengine/table{table.source['table']}"
    assert table.notes


def test_quantities() -> None:
    assert [_t(n).quantity["id"] for n in (1, 2, 3)] == [
        "total_flops",
        "max_intermediates_per_batch",
        "max_targets_per_batch",
    ]


@pytest.mark.parametrize("L", LS)
def test_table3_one_batch_is_the_hermite_count(L: int) -> None:
    # One batch holds every [r](0): the (t, u, v) with t + u + v <= L.
    assert _t(3).cells[(L, 1)] == (L + 1) * (L + 2) * (L + 3) // 6


@pytest.mark.parametrize("L", LS)
def test_table1_rises_strictly_with_batches(L: int) -> None:
    row = _row(_t(1), L)
    assert all(a < b for a, b in zip(row, row[1:], strict=False)), row


@pytest.mark.parametrize("L", LS)
def test_table3_falls_strictly_with_batches(L: int) -> None:
    row = _row(_t(3), L)
    assert all(a > b for a, b in zip(row, row[1:], strict=False)), row


def test_table2_exceptions_are_pinned_as_printed() -> None:
    # Table 2 falls left to right EXCEPT these two rows, which are printed this
    # way and recorded this way. Not misprints; do not regularise them.
    t = _t(2)
    assert _row(t, 7) == [125, 51, 39, 40]  # 6 -> 9 batches rises
    assert _row(t, 10) == [367, 139, 95, 95]  # 6 and 9 batches tie
    notes = " ".join(t.notes)
    assert "39 then 40" in notes and "95 and 95" in notes


def test_table2_falls_strictly_everywhere_else() -> None:
    # Exceptions keyed by (L, the batch count on the left of the pair).
    exceptions = {(7, 6), (10, 6)}
    t = _t(2)
    for L in LS:
        row = _row(t, L)
        for i in range(len(BATCHES) - 1):
            if (L, BATCHES[i]) not in exceptions:
                assert row[i] > row[i + 1], (L, row)
