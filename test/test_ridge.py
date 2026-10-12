"""#114: does batching cross OUR ridge? The rule (felizjueves.ridge), declared
in the issue before any roofline row existed, then scored on the rows.

Fast tier: synthetic counter rows for the rule's every branch, and the
tracked counter store (data/measured/roofline.json, #112) for the score. No
compiler, no card, no profiler.
"""

from __future__ import annotations

import pytest

from felizjueves import measured_store, ridge
from felizjueves.device import SM_86, MeasuredKey
from felizjueves.kernel import pq_spec

PROV = {"toolchain": "nvcc 13.0", "arch": "sm_86", "host": "h", "gpu": "g",
        "date": "2026-10-12", "counter_tool": "ncu x", "counter_unit": "u",
        "command": "ncu --metrics x"}
BYTES = 1_000_000


def _store(L: int, by_count: dict[int, tuple[float, ...]]) -> dict:
    """Counter rows for pq_spec(L, b): one row per kernel, each at the given
    intensity in units of sm_86's measured ridge."""
    out = {}
    for b, xs in by_count.items():
        assert len(xs) == b
        k = MeasuredKey(pq_spec(L, n_batches=b), SM_86, "nvcc 13.0")
        for i, x in enumerate(xs):
            out[measured_store.kernel_key(k, i)] = measured_store.Measured(
                fp64_flops=round(x * SM_86.ridge * BYTES),
                dram_bytes_read=BYTES, dram_bytes_written=0,
                counter_ns=1e6, kernel=f"k{i}", **PROV)
    return out


def _all(L: int, x: float, **over: tuple[float, ...]) -> dict:
    by = {b: (x,) * b for b in ridge.counts(L)}
    by.update({int(k[1:]): v for k, v in over.items()})
    return _store(L, by)


# --- the declaration ---------------------------------------------------------


def test_the_rule_was_declared_before_its_rows_and_says_so() -> None:
    doc = " ".join(ridge.__doc__.split())
    assert "DECLARED BEFORE ANY ROOFLINE ROW EXISTED" in doc
    assert "transcribed verbatim from issue #114 (created " \
           "2026-10-11T18:07:53Z, unedited); rows from PR #134 merged " \
           "2026-10-12T04:27Z" in doc
    # the issue's words, as the issue spells them
    for words in (
            "for one arch and one `L`, batching *crosses the ridge* iff the "
            "cell is memory-bound at `n_batches = 1` and compute-bound at "
            "some higher count in `SUBGRID`, with the crossing count "
            "recorded; the per-kernel variant (\"every kernel of the "
            "partition is compute-bound\") is declared separately",
            "scored per card, with the two allowed to disagree"):
        assert words in doc, words
    for part in ("THE TERMS", "THE ANSWERS", "COULD NOT MEASURE",
                 "neither rescues the other"):
        assert part in doc, part


def test_the_counts_are_the_subgrids() -> None:
    for L in measured_store.SUBGRID_LS:
        assert ridge.counts(L) == (1, 3, 6, 9, 12)
    with pytest.raises(ValueError):
        ridge.counts(8)


# --- the rule, on synthetic rows ----------------------------------------------


def test_memory_bound_at_one_and_compute_bound_later_crosses_at_the_first() -> None:
    s = _all(7, 2.0, b1=(0.5,), b3=(0.9, 0.9, 0.9))
    for v in ridge.VARIANTS:
        c = ridge.crossing(s, "sm_86", 7, v)
        assert (c.answer, c.count) == (ridge.CROSSED, 6), v
        assert c.bound == ((1, False), (3, False), (6, True), (9, True),
                           (12, True))


def test_compute_bound_at_one_is_already_not_crossed() -> None:
    s = _all(7, 2.0)
    for v in ridge.VARIANTS:
        c = ridge.crossing(s, "sm_86", 7, v)
        assert (c.answer, c.count) == (ridge.ALREADY, None)
        assert all(ok for _, ok in c.bound)


def test_memory_bound_everywhere_never_crosses() -> None:
    s = _all(7, 0.5)
    for v in ridge.VARIANTS:
        c = ridge.crossing(s, "sm_86", 7, v)
        assert (c.answer, c.count) == (ridge.NEVER, None)
        assert not any(ok for _, ok in c.bound)


def test_the_two_variants_are_scored_apart() -> None:
    # at b3 the cell (counts summed) is past the ridge but one kernel is not;
    # from b6 on every kernel is: the cell crosses at 3, the kernels at 6
    s = _all(7, 3.0, b1=(0.5,), b3=(0.5, 3.0, 3.0))
    cell = ridge.crossing(s, "sm_86", 7, ridge.CELL)
    kern = ridge.crossing(s, "sm_86", 7, ridge.PER_KERNEL)
    assert (cell.answer, cell.count) == (ridge.CROSSED, 3)
    assert (kern.answer, kern.count) == (ridge.CROSSED, 6)
    # and when a memory-bound kernel survives every split, the per-kernel
    # variant NEVER crosses while the cell does: neither rescues the other
    s = _store(7, {b: (0.5,) + (3.0,) * (b - 1) for b in ridge.counts(7)})
    assert ridge.crossing(s, "sm_86", 7, ridge.CELL).count == 3
    assert ridge.crossing(s, "sm_86", 7, ridge.PER_KERNEL).answer == \
        ridge.NEVER


def test_an_unmeasured_card_could_not_be_measured_never_never_crossed() -> None:
    c = ridge.crossing({}, "gfx1200", 16, ridge.CELL)
    assert c.answer == ridge.COULD_NOT_MEASURE != ridge.NEVER
    assert c.count is None and c.bound == ()
    assert c.reason == measured_store.UNMEASURED["gfx1200"]


def test_a_measured_card_missing_a_cell_is_not_scored() -> None:
    s = _all(7, 2.0)
    s = {k: r for k, r in s.items() if "/b9/" not in k}
    with pytest.raises(KeyError, match="n_batches=9"):
        ridge.crossing(s, "sm_86", 7, ridge.CELL)
    with pytest.raises(ValueError):
        ridge.crossing(_all(7, 2.0), "sm_86", 7, "per-cell")
