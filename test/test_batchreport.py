"""The recommended batch count per (L, arch), from wall-clock only (#56).

Three things are held here, all fast-tier (no compiler, no card):

- THE DATA IS THE DERIVATION. data/measured/batch_recommendation.json equals
  `batchreport.report` recomputed from data/measured/kernels.json, and the
  tables in docs/implementation/j_engine.md are `batchreport.markdown` of it,
  verbatim -- so the store, the data file and the doc cannot drift apart.
- WALL-CLOCK ONLY. Moving a row's spills or registers moves nothing; moving
  its median does. The import side is test/test_layering.py's.
- NOT A POLICY. `Spec.n_batches` still defaults to 1.

The pinned verdict (`PINNED`) is the result on the tracked rows, so a
re-timing that moves a recommendation is seen as a diff here.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace

import pytest

from felizjueves import batchreport, measured
from felizjueves.device import ARCHS, MeasuredKey
from felizjueves.kernel import pq_spec
from felizjueves.spec import Operator, Spec, Target

ROWS = measured.load()
DOC = json.loads(batchreport.RECOMMENDATION.read_text())
J_ENGINE = (measured.REPO / "docs" / "implementation" / "j_engine.md").read_text()


# --- the data, the derivation and the doc agree -------------------------------


def test_the_data_file_is_the_report_recomputed_from_the_store() -> None:
    assert DOC == batchreport.report(ROWS), (
        "rerun: PYTHONPATH=src uv run python -m felizjueves.batchreport --write")


def test_the_j_engine_tables_are_the_data_verbatim() -> None:
    assert batchreport.markdown(DOC) in J_ENGINE, (
        "paste `python -m felizjueves.batchreport --markdown` into j_engine.md")


def test_lever_rows_are_not_read() -> None:
    # #55's recycle arm is in the store; the report is of the literal rows only.
    levered = {k for k in ROWS if "/literal@" not in k}
    assert levered, "expected lever rows in the store (#55)"
    literal = {k: r for k, r in ROWS.items() if k not in levered}
    assert batchreport.report(literal) == batchreport.report(ROWS)


def test_every_subgrid_cell_on_both_arches_is_covered_and_none_dropped() -> None:
    got = {(c["arch"], c["L"]) for c in DOC["cells"]}
    assert got == {(a.name, L) for a in ARCHS for L in measured.SUBGRID_LS}
    assert len(DOC["cells"]) == len(got)
    assert DOC["header"]["dropped_cells"] == []
    assert DOC["header"]["subgrid"]["n_batches"] == list(batchreport.BATCHES)
    for c in DOC["cells"]:
        for w in batchreport.INPUTS:
            assert set(c[w]["all"]) == {str(b) for b in batchreport.BATCHES}
            assert c[w]["repeats"] > 1  # one timing is not a recommendation


def test_the_provenance_names_the_host_both_cards_and_both_toolchains() -> None:
    p = DOC["header"]["provenance"]
    assert p["sm_86"]["gpu"] == "NVIDIA GeForce RTX 3080"
    assert p["sm_86"]["toolchain"] == "nvcc 13.0"
    assert p["gfx1200"]["gpu"].endswith("gfx1200)")
    assert p["gfx1200"]["toolchain"] == "hipcc / ROCm 7.2.4"
    assert p["sm_86"]["host"] == p["gfx1200"]["host"]
    for arch in p.values():
        assert arch["host"] in J_ENGINE and arch["toolchain"] in J_ENGINE


# --- the verdict on the tracked rows ------------------------------------------

# (arch, L): (best on wide-T, best on near-T, near-T resolved)
PINNED = {
    ("sm_86", 7): (1, 1, True),
    ("sm_86", 10): (1, 1, True),
    ("sm_86", 13): (1, 3, True),
    ("sm_86", 16): (1, 9, True),
    ("gfx1200", 7): (1, 1, True),
    ("gfx1200", 10): (1, 3, False),
    ("gfx1200", 13): (1, 3, False),
    ("gfx1200", 16): (1, 12, True),
}


def test_the_recommendation_on_the_tracked_rows() -> None:
    got = {(c["arch"], c["L"]): (c["wide"]["n_batches"], c["near"]["n_batches"],
                                 c["near"]["resolved"]) for c in DOC["cells"]}
    assert got == PINNED
    # wide-T: one batch, beyond its spread, everywhere
    assert all(c["wide"]["resolved"] and c["wide"]["within_spread"] == [1]
               for c in DOC["cells"])


def test_the_answer_depends_on_the_input_and_the_arches_disagree() -> None:
    # The finding, not a defect: the premise "the batch count with the best
    # median" has two answers, and at L = 16 the two cards give different ones.
    assert any(not c["agree"] for c in DOC["cells"])
    near16 = {c["arch"]: c["near"]["n_batches"] for c in DOC["cells"]
              if c["L"] == 16}
    assert near16["sm_86"] != near16["gfx1200"]


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


def test_a_cell_crossing_sweeps_is_flagged() -> None:
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


# --- not a policy -------------------------------------------------------------


def test_spec_n_batches_still_defaults_to_one() -> None:
    field = {f.name: f for f in dataclasses.fields(Spec)}["n_batches"]
    assert field.default == 1
    assert Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J).n_batches == 1
    assert pq_spec(16).n_batches == 1
    assert "does not change" in DOC["header"]["not_policy"]
