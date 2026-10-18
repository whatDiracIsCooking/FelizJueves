"""batchreport.py's rule, on SYNTHETIC rows (#56): a recommendation is
resolved only beyond the larger IQR, it reads wall-clock only (moving a
row's spills or registers moves nothing), a cell crossing timing sessions is
flagged, and an untimed or missing cell is refused, never skipped.

Nothing here reads data/measured/. That batch_recommendation.json is the
report recomputed from the tracked rows, that j_engine.md carries its tables
verbatim, and that its pinned verdict and `about` still read off those rows
are checks of measurement artefacts: measure/measure_batchreport.py,
`devtools/measure.sh check`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from felizjueves import batchreport
from felizjueves import measured_store as measured
from felizjueves.device import ARCHS, MeasuredKey
from felizjueves.kernel import pq_spec

# --- the rule, on synthetic rows ----------------------------------------------


def _store(medians: dict[int, tuple[int, int]], near: dict[int, tuple[int, int]],
           *, L: int = 13, arch=ARCHS[0]) -> dict[str, measured.Measured]:
    out = {}
    for b in batchreport.BATCHES:
        (wm, wi), (nm, ni) = medians[b], near[b]
        row = measured.Measured(
            runtime_ns=wm, repeats=31, spread_ns=wi, spill_store=100 * b,
            spill_load=100 * b, registers=255, toolchain="tc", arch=arch.name,
            host="h", gpu="g", date="2026-10-11",
            raw={"wallclock:near:median_ns": nm, "wallclock:near:q1_ns": 0,
                 "wallclock:near:q3_ns": ni})
        out[MeasuredKey(pq_spec(L, n_batches=b), arch, "tc").key()] = row
    return out


FLAT = {b: (1000 + b, 1) for b in batchreport.BATCHES}


def test_resolved_needs_a_margin_beyond_the_larger_iqr() -> None:
    near = {1: (1000, 10), 3: (1015, 20), 6: (1100, 5), 9: (1200, 5),
            12: (1300, 5)}
    r = batchreport.recommend(_store(FLAT, near), ARCHS[0].name, 13, "near")
    assert r.best.n_batches == 1 and r.runner_up.n_batches == 3
    assert r.margin_ns == 15 and not r.resolved  # 15 <= max(10, 20)
    assert r.within_spread == (1, 3)
    near[3] = (1021, 20)
    r = batchreport.recommend(_store(FLAT, near), ARCHS[0].name, 13, "near")
    assert r.resolved and r.within_spread == (1,)


def test_wall_clock_only_spills_and_registers_move_nothing() -> None:
    near = {b: (5000 - b, 1) for b in batchreport.BATCHES}
    store = _store(FLAT, near)
    before = {w: batchreport.recommend(store, ARCHS[0].name, 13, w)
              for w in batchreport.INPUTS}
    moved = {k: replace(r, spill_store=0, spill_load=0, registers=1)
             for k, r in store.items()}
    after = {w: batchreport.recommend(moved, ARCHS[0].name, 13, w)
             for w in batchreport.INPUTS}
    assert before == after
    assert before["wide"].best.n_batches == 1
    assert before["near"].best.n_batches == 12


def test_a_cell_crossing_sweeps_is_flagged(monkeypatch) -> None:
    # The tracked rows are one session per arch (#93); the flag is for a store
    # that mixes sweeps, as #53's and #60's did.
    monkeypatch.setattr(batchreport, "SESSION",
                        {**batchreport.SESSION, 12: "another sweep"})
    near = {1: (1000, 1), 3: (1100, 1), 6: (1200, 1), 9: (990, 1), 12: (980, 1)}
    r = batchreport.recommend(_store(FLAT, near), ARCHS[0].name, 13, "near")
    assert (r.best.n_batches, r.runner_up.n_batches) == (12, 9)
    assert not r.same_session
    near[12] = (2000, 1)
    r = batchreport.recommend(_store(FLAT, near), ARCHS[0].name, 13, "near")
    assert (r.best.n_batches, r.runner_up.n_batches) == (9, 1)
    assert r.same_session


def test_an_untimed_or_missing_cell_is_refused_not_skipped() -> None:
    store = _store(FLAT, FLAT)
    key = MeasuredKey(pq_spec(13, n_batches=6), ARCHS[0], "tc").key()
    untimed = {**store, key: replace(store[key], runtime_ns=None, repeats=None,
                                     spread_ns=None)}
    with pytest.raises(KeyError):
        batchreport.recommend(untimed, ARCHS[0].name, 13, "wide")
    no_near = {**store, key: replace(store[key], raw={})}
    with pytest.raises(KeyError):
        batchreport.recommend(no_near, ARCHS[0].name, 13, "near")
    gone = {k: v for k, v in store.items() if k != key}
    with pytest.raises(KeyError):
        batchreport.recommend(gone, ARCHS[0].name, 13, "wide")
