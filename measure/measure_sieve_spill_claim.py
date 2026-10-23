"""#164: `jengine/sieve-non-monotone-in-spills`, read on our two cards.

[JEngine, v2, §4.1]: the sieve's effect on spills is non-monotone across
classes -- about 4% fewer on average, but [5|6] 3% more and [8|8] 11% fewer.
#162 recorded each `no-sieve` cell of `measured_store.SIEVE_CELLS` (the
subgrid and L = 11, one batch) against its literal, sieved control on sm_86
and gfx1200; `measured_store.sieve_spills` reads the published pattern off
those rows, half by half (its comment block is the rule).

A CHECK OF MEASUREMENT ARTEFACTS, moved from test/ (`devtools/measure.sh
check`; the reading on synthetic rows stayed, as
test/test_sieve_spill_claim.py). Here: what the gfx1200 rows actually carry
(a spilled-register count, not bytes); then the reading on the tracked rows,
recomputed from the store and held against every number claims.json and
limits.json quote. The claim stays UNVERIFIABLE: the pattern
fails on our rows on both cards, and our rows are not the paper's kernels,
compiler or card, so they can neither discharge it nor refute it
(limits.json's record says why). Nothing here reads Table 4.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from felizjueves.literature.claims import ClaimStatus, load_claims
from felizjueves.literature.limits import load_limits
from felizjueves.performance import measured_store
from felizjueves.performance.measured_store import (
    FEWER,
    MORE,
    sieve_spill_claim_holds,
    sieve_spills,
)

ROOT = Path(__file__).resolve().parents[1]
JENGINE = ROOT / "data" / "literature" / "jengine"
CLAIM = "jengine/sieve-non-monotone-in-spills"
ARCHES = ("sm_86", "gfx1200")
FLOOR = 0.01  # a 1% floor: is the non-monotonicity more than a few registers?


# -- what the gfx1200 rows carry -----------------------------------------------


def _summed(raw: dict, field: str) -> int:
    return sum(v for k, v in raw.items()
               if k == field or k.endswith(f":{field}"))


def _sieve_rows(arch: str) -> list:
    store = measured_store.load()
    out = []
    for c in measured_store.SIEVE_CELLS:
        for s in (replace(c, levers=()), c):
            (r,) = measured_store.rows(spec=s, arch=arch, store=store).values()
            out.append((s, r))
    return out


def test_the_gfx1200_rows_carry_a_spilled_register_count() -> None:
    """hipcc's kernel-resource-usage reports spilled REGISTERS (VGPRs Spill +
    SGPRs Spill), with no store/load split; the store writes that count into
    both spill fields. So a gfx1200 spill is a register count, twice over in
    store + load, and never bytes; on sm_86 the fields are ptxas's bytes."""
    rows = _sieve_rows("gfx1200")
    assert len(rows) == 2 * len(measured_store.SIEVE_CELLS) == 42
    for s, r in rows:
        assert r.spill_unit.startswith("registers (hipcc kernel-resource-usage"), s
        assert r.spill_store == r.spill_load, s
        assert r.spill_store == (_summed(r.raw, "VGPRs Spill")
                                 + _summed(r.raw, "SGPRs Spill")), s
        assert _summed(r.raw, "SGPRs Spill") == 0, s  # every spill is a VGPR
    for s, r in _sieve_rows("sm_86"):
        assert r.spill_unit.startswith("bytes (ptxas -v"), s


def _scratch_disagreements() -> list[tuple[int, int]]:
    """Cells where gfx1200's ScratchSize (per-lane scratch bytes, which also
    holds non-spill stack) moves the other way from the spilled-register
    count -- why the register count is read and scratch is not."""
    rows = dict(_sieve_rows("gfx1200"))
    out = []
    for c in measured_store.SIEVE_CELLS:
        lit, ns = rows[replace(c, levers=())], rows[c]
        a, b = _summed(lit.raw, "ScratchSize"), _summed(ns.raw, "ScratchSize")
        regs = (lit.spill_store > ns.spill_store) - (lit.spill_store < ns.spill_store)
        if (a > b) - (a < b) != regs:
            out.append((c.total_am, c.n_batches))
    return out


# -- the reading, on the tracked rows ------------------------------------------


def _pct(x: float) -> float:
    return round(100 * x, 2)


def _reading(arch: str) -> dict:
    """What limits.json's record states for `arch`, recomputed from the store."""
    p = sieve_spills(measured_store.load(), arch)
    one = [c for c in p.spilling if c.n_batches == 1]
    big = [c for c in p.cells if c.change is not None and abs(c.change) >= FLOOR]
    return {
        "halves": p.halves(),
        "holds": p.holds,
        "cells": len(p.cells),
        "spilling": len(p.spilling),
        "mean_change_pct": _pct(p.mean_change),
        "one_batch_mean_change_pct": _pct(sum(c.change for c in one) / len(one)),
        "total_change_pct": _pct(p.total_change),
        "more": [f"L{c.L} b{c.n_batches}" for c in p.cells if c.sign == MORE],
        "fewer": [f"L{c.L} b{c.n_batches}" for c in p.cells if c.sign == FEWER],
        "non_monotone_at_1pct_floor": ({MORE, FEWER} <= {c.sign for c in big}),
        "eight_eight_pct": _pct(p.at(16, 1).change),
        "five_six_stand_in_pct": _pct(p.at(11, 1).change),
        "change_pct": {f"L{c.L} b{c.n_batches}": _pct(c.change)
                       for c in p.spilling},
    }


def _record():
    (lim,) = [lim for lim in load_limits(JENGINE / "limits.json")
              if isinstance(lim.asserted_value, dict)
              and lim.asserted_value.get("claim") == CLAIM]
    return lim


def test_the_sieve_spill_pattern_on_both_cards() -> None:
    """The published pattern, read on #162's rows: on both cards the effect
    is non-monotone (more spills at some cells, fewer at others, also above
    a 1% floor) and [8|8] spills less with the sieve; on both the [5|6]
    stand-in, (6,0|5,0), spills LESS, not more; and the mean falls on
    gfx1200 and rises on sm_86 (where the summed spills still fall). So the
    pattern fails on both cards -- and a test asserting it would fail on our
    rows without saying anything about the paper's (limits.json)."""
    store = measured_store.load()
    got = {a: _reading(a) for a in ARCHES}
    assert got["sm_86"]["halves"] == {"mean falls": False, "non-monotone": True,
                                      "[5|6] more": False, "[8|8] fewer": True}
    assert got["gfx1200"]["halves"] == {"mean falls": True,
                                        "non-monotone": True,
                                        "[5|6] more": False,
                                        "[8|8] fewer": True}
    for a in ARCHES:
        assert got[a]["non_monotone_at_1pct_floor"], a
        assert got[a]["total_change_pct"] < 0, a
    assert not sieve_spill_claim_holds(store)
    assert not any(sieve_spill_claim_holds(store, (a,)) for a in ARCHES)

    # limits.json's record states exactly this, recomputed, never re-typed
    av = _record().asserted_value
    assert av["board"] is None and av["rule"] is None
    assert av["cards"] == got
    assert av["gfx1200_scratch_disagrees_at"] == [
        f"L{L} b{b}" for L, b in _scratch_disagreements()]
    assert av["verdict"] == {a: ("holds" if got[a]["holds"] else "fails")
                             for a in ARCHES}


def test_the_claim_stays_unverifiable_and_quotes_the_store() -> None:
    (claim,) = [c for c in load_claims(JENGINE / "claims.json") if c.id == CLAIM]
    assert claim.status is ClaimStatus.UNVERIFIABLE
    reason = " ".join(claim.reason.split())
    assert ("measure/measure_sieve_spill_claim.py::"
            "test_the_sieve_spill_pattern_on_both_cards") in reason
    assert "limits.json" in reason
    sm, gfx = _reading("sm_86"), _reading("gfx1200")
    for quoted in (
        f"sm_86 {sm['mean_change_pct']:+.2f}%",
        f"gfx1200 {gfx['mean_change_pct']:+.2f}%",
        f"summed spills {sm['total_change_pct']:+.2f}% and "
        f"{gfx['total_change_pct']:+.2f}%",
        f"[8|8] {sm['eight_eight_pct']:+.2f}% and {gfx['eight_eight_pct']:+.2f}%",
        f"(6,0|5,0) {sm['five_six_stand_in_pct']:+.2f}% and "
        f"{gfx['five_six_stand_in_pct']:+.2f}%",
    ):
        assert quoted in reason, quoted
    mech = " ".join(_record().mechanism.split())
    assert "Table 4" in mech and "never compared" in mech
