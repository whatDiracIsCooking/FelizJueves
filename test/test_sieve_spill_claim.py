"""#164: `jengine/sieve-non-monotone-in-spills`, read on our two cards.

[JEngine, v2, §4.1]: the sieve's effect on spills is non-monotone across
classes -- about 4% fewer on average, but [5|6] 3% more and [8|8] 11% fewer.
#162 recorded each `no-sieve` cell of `measured_store.SIEVE_CELLS` (the
subgrid and L = 11, one batch) against its literal, sieved control on sm_86
and gfx1200; `measured_store.sieve_spills` reads the published pattern off
those rows, half by half (its comment block is the rule).

Here: the reading on synthetic rows first; then what the gfx1200 rows
actually carry (a spilled-register count, not bytes); then the reading on the
tracked rows, recomputed from the store and held against every number
claims.json and limits.json quote. The claim stays UNVERIFIABLE: the pattern
fails on our rows on both cards, and our rows are not the paper's kernels,
compiler or card, so they can neither discharge it nor refute it
(limits.json's record says why). Nothing here reads Table 4.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from felizjueves import measured_store
from felizjueves.device import MeasuredKey, by_name
from felizjueves.kernel import pq_spec
from felizjueves.literature.claims import ClaimStatus, load_claims
from felizjueves.literature.limits import load_limits
from felizjueves.measured_store import (
    FEWER,
    MORE,
    NO_SIEVE,
    SAME,
    SieveSpill,
    sieve_spill_claim_holds,
    sieve_spills,
)

ROOT = Path(__file__).resolve().parents[1]
JENGINE = ROOT / "data" / "literature" / "jengine"
CLAIM = "jengine/sieve-non-monotone-in-spills"
ARCHES = ("sm_86", "gfx1200")
TOOLCHAIN = {"sm_86": "nvcc 13.0", "gfx1200": "hipcc / ROCm 7.2.4"}
FLOOR = 0.01  # a 1% floor: is the non-monotonicity more than a few registers?


def _cell(L: int, n_batches: int = 1):
    return pq_spec(L, n_batches=n_batches, levers=(NO_SIEVE,))


def _store(spills: dict, arch: str = "sm_86") -> dict:
    """Synthetic compile-only rows: {no-sieve cell: (sieved, unsieved)}."""
    prov = {"toolchain": TOOLCHAIN[arch], "arch": arch, "host": "h",
            "gpu": "g", "date": "2026-10-13"}
    store: dict = {}
    for spec, (s, u) in spills.items():
        for sp, n in ((replace(spec, levers=()), s), (spec, u)):
            key = MeasuredKey(sp, by_name(arch), TOOLCHAIN[arch])
            store = measured_store.merge(store, key, measured_store.Measured(
                registers=255, spill_store=n // 2, spill_load=n - n // 2,
                **prov))
    return store


# -- the reading, on synthetic rows --------------------------------------------

PUBLISHED_SHAPE = {  # the published shape: mean falls, [5|6] more, [8|8] fewer
    _cell(16): (89, 100),  # -11%
    _cell(11): (103, 100),  # +3%
    _cell(13): (96, 100),  # -4%
    _cell(7, 3): (0, 0),  # spills in neither arm: SAME, out of the mean
}


def test_the_reading_on_synthetic_rows() -> None:
    cells = tuple(PUBLISHED_SHAPE)
    p = sieve_spills(_store(PUBLISHED_SHAPE), "sm_86", cells)
    assert [c.sign for c in p.cells] == [FEWER, MORE, FEWER, SAME]
    assert p.at(7, 3).change is None and len(p.spilling) == 3
    assert p.mean_change == pytest.approx((-0.11 + 0.03 - 0.04) / 3)
    assert p.total_change == pytest.approx((288 - 300) / 300)
    assert p.halves() == {"mean falls": True, "non-monotone": True,
                          "[5|6] more": True, "[8|8] fewer": True}
    assert p.holds
    assert sieve_spill_claim_holds(_store(PUBLISHED_SHAPE), ("sm_86",), cells)

    # each half can fail alone, and then the pattern does not hold
    def halves(**moved) -> dict[str, bool]:
        got = dict(PUBLISHED_SHAPE)
        for L, v in moved.items():
            got[_cell(int(L[1:]))] = v
        q = sieve_spills(_store(got), "sm_86", cells)
        assert not q.holds
        return {k: v for k, v in q.halves().items() if not v}

    assert halves(L11=(97, 100)) == {"non-monotone": False,
                                     "[5|6] more": False}
    assert halves(L16=(100, 100)) == {"[8|8] fewer": False}
    assert halves(L11=(150, 100)) == {"mean falls": False}
    # a mean that falls is not the pattern: monotone fewer everywhere
    assert halves(L11=(99, 100), L13=(99, 100)) == {"non-monotone": False,
                                                    "[5|6] more": False}


def test_the_reading_refuses_what_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="relative change from zero"):
        SieveSpill("sm_86", (8, 0, 8, 0), 16, 1, sieved=4, unsieved=0)
    with pytest.raises(ValueError, match="negative"):
        SieveSpill("sm_86", (8, 0, 8, 0), 16, 1, sieved=-1, unsieved=4)
    store = _store(PUBLISHED_SHAPE)
    with pytest.raises(ValueError, match="not a no-sieve arm"):
        sieve_spills(store, "sm_86", (pq_spec(16),))
    with pytest.raises(KeyError, match="0 rows"):
        sieve_spills(store, "sm_86", (_cell(10),))
    with pytest.raises(KeyError, match="0 rows"):
        sieve_spills(store, "gfx1200", (_cell(16),))
    # a timed-only literal row carries no spill fields: not compiled
    lit = MeasuredKey(pq_spec(10), by_name("sm_86"), "nvcc 13.0")
    lev = MeasuredKey(_cell(10), by_name("sm_86"), "nvcc 13.0")
    prov = {"toolchain": "nvcc 13.0", "arch": "sm_86", "host": "h",
            "gpu": "g", "date": "2026-10-13"}
    timed_only = measured_store.merge({}, lit, measured_store.Measured(
        runtime_ns=1.0, repeats=1, spread_ns=0.0, **prov,
        raw=dict.fromkeys(measured_store.GPU_UTIL_FIELDS)))
    timed_only = measured_store.merge(timed_only, lev, measured_store.Measured(
        registers=255, spill_store=0, spill_load=0, **prov))
    with pytest.raises(KeyError, match="not compiled"):
        sieve_spills(timed_only, "sm_86", (_cell(10),))
    flat = sieve_spills(_store({_cell(10): (0, 0)}), "sm_86", (_cell(10),))
    with pytest.raises(ValueError, match="no cell spills"):
        flat.mean_change  # noqa: B018
    with pytest.raises(KeyError, match="0 cells"):
        flat.at(16, 1)


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
    assert ("test/test_sieve_spill_claim.py::"
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
