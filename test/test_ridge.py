"""#114: does batching cross OUR ridge? The rule (felizjueves.ridge), declared
in the issue before any roofline row existed, then scored on the rows.

Fast tier: synthetic counter rows for the rule's every branch, and the
tracked counter store (data/measured/roofline.json, #112) for the score. No
compiler, no card, no profiler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from felizjueves import measured_store, ridge
from felizjueves.device import GFX1200, SM_86, MeasuredKey
from felizjueves.kernel import pq_spec
from felizjueves.literature.claims import ClaimStatus, load_claims
from felizjueves.literature.limits import load_limits

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


# --- the score, on the tracked rows ------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
JENGINE = ROOT / "data" / "literature" / "jengine"
CLAIM = "jengine/batching-crosses-our-ridge"
COUNTERS = measured_store.load_counters()
ROWS = measured_store.load()
LS = (7, 10, 13, 16)


def _label(c: ridge.Crossing) -> str:
    return f"crossed at {c.count}" if c.answer == ridge.CROSSED else c.answer


def _cell(L: int, b: int) -> str:
    return ridge.cell_key(COUNTERS, "sm_86", L, b)


@pytest.mark.claim("jengine/batching-crosses-our-ridge")
def test_batching_crosses_our_ridge_on_sm_86_and_gfx1200_could_not_measure(
) -> None:
    """#114, scored under the rule the issue declared before any roofline row
    existed. Every number claims.json and limits.json quote for this claim is
    recomputed here from data/measured/roofline.json, ceilings.json and
    kernels.json. A re-measurement that moves an answer fails here: news."""
    got = ridge.scored(COUNTERS)
    # sm_86: L = 16 crosses at 3, both variants; L = 7, 10, 13 are
    # compute-bound already at one batch; no L never crosses
    for v in ridge.VARIANTS:
        assert {L: _label(got["sm_86", L, v]) for L in LS} == {
            7: ridge.ALREADY, 10: ridge.ALREADY, 13: ridge.ALREADY,
            16: "crossed at 3"}, v
        for L in LS:  # compute-bound at every count from 1 (or from 3)
            assert [ok for _, ok in got["sm_86", L, v].bound] == \
                [L != 16] + [True] * 4, (L, v)
    # gfx1200: the answer is absent -- could not measure, never "never"
    for L in LS:
        for v in ridge.VARIANTS:
            c = got["gfx1200", L, v]
            assert c.answer == ridge.COULD_NOT_MEASURE and c.bound == ()
            assert "libdw.so.1" in c.reason and "fp64" in c.reason
    # the rows the score read: 124 kernel rows over 20 cells, sm_86 only
    assert len(COUNTERS) == 124
    assert len({measured_store.split_kernel_key(k)[0] for k in COUNTERS}) == 20
    # each card's own measured ridge
    assert (round(SM_86.ridge, 2), round(GFX1200.ridge, 2)) == (0.65, 1.49)
    # the intensities quoted, FLOP/B
    one = {L: measured_store.cell_roofline(COUNTERS, _cell(L, 1)).intensity
           for L in LS}
    assert (round(one[7]), round(one[10], 2), round(one[13], 2),
            round(one[16], 2)) == (300, 3.54, 0.96, 0.40)
    assert round(measured_store.cell_roofline(
        COUNTERS, _cell(16, 3)).intensity, 2) == 1.52
    assert round(min(p.intensity for p in measured_store.kernel_roofline(
        COUNTERS, _cell(16, 3))), 2) == 1.46
    # the cross-read against #60: every kernel of every 9-batch cell is
    # compute-bound on sm_86, while 9 fails #60 there on spills alone (12
    # spills less at L = 13 and 16) and 12 is slower than 9 at every L on
    # both inputs
    for L in LS:
        assert ridge.compute_bound(COUNTERS, "sm_86", L, 9, ridge.PER_KERNEL)
    nine = {L: measured_store.nine_vs_twelve(ROWS, "sm_86", L) for L in LS}
    assert {L for L, t in nine.items() if t.fewer_spills} == {13, 16}
    for t in nine.values():
        assert t.wide[1] > t.wide[0] and t.near[1] > t.near[0], t.L
    # the records quote what was recomputed
    (claim,) = [c for c in load_claims(JENGINE / "claims.json") if c.id == CLAIM]
    assert claim.status is ClaimStatus.ASSERTED
    for s in ("0.65 FLOP/B", "1.49 FLOP/B", "0.40 FLOP/B", "1.52 FLOP/B",
              "1.46 FLOP/B", "300, 3.54 and 0.96 FLOP/B", "could not measure",
              "ALREADY AT 1 BATCH", "Fig. 7's A100 ridge is a different number"):
        assert s in claim.statement, s
    (lim,) = [lim for lim in load_limits(JENGINE / "limits.json")
              if lim.asserted_value.get("claim") == CLAIM]
    v = lim.asserted_value
    assert v["ridge_flop_per_byte"] == {"sm_86": 0.65, "gfx1200": 1.49}
    assert v["answers"] == {
        "sm_86": {str(L): {var: _label(got["sm_86", L, var])
                           for var in ridge.VARIANTS} for L in LS},
        "gfx1200": ridge.COULD_NOT_MEASURE}
    assert "not quoted and not compared" in v["a100_ridge"]
    assert "2026-10-11T18:07:53Z" in v["rule_declared"]
    for s in ("124 kernel rows, 20 cells", "(300, 3.54, 0.96 FLOP/B)",
              "(0.40 FLOP/B)", "cell 1.52 FLOP/B, smallest kernel 1.46"):
        assert s in lim.evidence, s
    for s in ("0.65 FLOP/B on sm_86 and 1.49 on gfx1200",
              "compute-bound at 0.96 FLOP/B against 0.65"):
        assert s in lim.mechanism, s
