"""#164: `jengine/sieve-non-monotone-in-spills`, the reading's rule on
SYNTHETIC rows.

[JEngine, v2, §4.1]: the sieve's effect on spills is non-monotone across
classes -- about 4% fewer on average, but [5|6] 3% more and [8|8] 11% fewer.
`measured_store.sieve_spills` reads the published pattern off a store's
`no-sieve` rows against their literal controls, half by half (its comment
block is the rule). Here: that reading on synthetic rows, and its refusals.
What the tracked gfx1200 rows carry and the reading on the tracked rows, held
against every number claims.json and limits.json quote, are checks of
measurement artefacts: measure/measure_sieve_spill_claim.py
(`devtools/measure.sh check`).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from felizjueves import measured_store
from felizjueves.device import MeasuredKey, by_name
from felizjueves.measured_store import (
    FEWER,
    MORE,
    NO_SIEVE,
    SAME,
    SieveSpill,
    sieve_spill_claim_holds,
    sieve_spills,
)
from felizjueves.mmd.kernel import pq_spec

TOOLCHAIN = {"sm_86": "nvcc 13.0", "gfx1200": "hipcc / ROCm 7.2.4"}


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
