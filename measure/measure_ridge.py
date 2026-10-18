"""#114: does batching cross OUR ridge? The rule (felizjueves.ridge), scored
on the tracked rows.

A CHECK OF MEASUREMENT ARTEFACTS, moved from test/test_ridge.py (which keeps
the rule's every branch on synthetic counter rows): the score is recomputed
from data/measured/roofline.json (#112), ceilings.json and kernels.json, and
every number claims.json and limits.json quote for
`jengine/batching-crosses-our-ridge` must match it. The claim mark lives here,
and test/test_claims.py's registry check scans measure/ as well as test/ for
it. No compiler, no card, no profiler: `devtools/measure.sh check`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from felizjueves import measured_store, ridge
from felizjueves.device import GFX1200, SM_86
from felizjueves.literature.claims import ClaimStatus, load_claims
from felizjueves.literature.limits import load_limits

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
