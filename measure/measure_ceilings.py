"""The tracked ceiling record, data/measured/ceilings.json, checked (#109,
#141): every recorded ceiling carries its provenance, its probe text's hash
and an idle card state, and gfx1200's fp64 is bounded per counted cycle.

Moved from test/test_ceilings.py, which keeps the gate's and the recorder's
rules on synthetic probe output: this half reads the tracked file, so it is a
check of a measurement artefact, run by `devtools/measure.sh check` (no
compiler, no card). The probes that write the record are
measure/measure_ceilings_device.py's.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from conftest import VRAM_PROBE

from felizjueves.performance import ceilings, gate
from felizjueves.performance.device import ARCHS, GFX1200, QUANTITIES, SM_86

# -- the recorded data ----------------------------------------------------------


def test_the_recorded_ceilings_carry_their_provenance() -> None:
    data = ceilings.load()
    assert set(data["published"]) == {a.name for a in ARCHS}
    sha = hashlib.sha256(ceilings.probe_source(VRAM_PROBE).encode()).hexdigest()
    backend = {SM_86.name: "cuda", GFX1200.name: "hip"}
    for name, rows in data.get("ceilings", {}).items():
        assert name in backend, f"a ceiling for an undeclared arch {name}"
        published = data["published"][name]
        assert published["source"]
        for q, row in rows.items():
            assert q in QUANTITIES
            assert row["arch"] == name and row["kind"] == "achievable"
            assert row["unit"] == ceilings.UNIT_OF[q]
            assert row["command"] == ceilings.COMMAND.format(backend=backend[name])
            assert row["source_sha256"] == sha, (
                f"{name} {q}: recorded from other probe text -- re-record it")
            assert row["repeats"] == ceilings.REPEATS
            for k in ("host", "date", "toolchain", "gpu", "probe", "compile"):
                assert row[k], (name, q, k)
            raw = row["raw"]
            # the card's state at probe time, as #93 records it: both
            # readings present and idle -- a refused phase is never here
            assert gate.is_idle(raw["gpu_util_before"]), (name, q)
            assert gate.is_idle(raw["gpu_util_after"]), (name, q)
            assert 0 < row["value"] <= raw["best"]
            assert raw["published"] == published[q]
            assert raw["ratio_to_published"] == pytest.approx(
                row["value"] / published[q])
            bound = published.get("per_clock", {}).get(q)
            if bound is None:
                assert raw["best"] <= published[q], (name, q)
                assert "per_clock" not in raw
                continue
            # #141: bounded per counted GPU cycle; the flop/s figure is at the
            # rated clock, and exceeding it is explained by the counted clock
            pc = raw["per_clock"]
            assert pc["counter"] == ceilings.CLOCK_COUNTER
            assert pc["command"] == ceilings.CYCLES_COMMAND and pc["tool"]
            assert pc["bound_flop_per_clk"] == bound["flop_per_clk"]
            assert gate.is_idle(pc["gpu_util_before"]), (name, q)
            assert gate.is_idle(pc["gpu_util_after"]), (name, q)
            assert set(pc["configs"]) == {str(c) for c in ceilings.FMA_CHAINS}
            for ev in pc["configs"].values():
                assert 0 < ev["flop_per_clk_max"] <= bound["flop_per_clk"]
                assert ev["clock_hz_min"] <= ev["clock_hz_median"] <= ev[
                    "clock_hz_max"]
            chosen = pc["configs"][str(pc["chosen"])]
            assert raw["probe_header"].startswith(
                f"probe fma chains {pc['chosen']} ")
            if raw["best"] > published[q]:
                assert chosen["clock_hz_max"] > bound["rated_clock_hz"], (
                    f"{name} {q} exceeds its rated-clock figure without a "
                    "counted clock above the rated one: a probe bug")


def test_gfx1200_fp64_bound_is_free_of_the_dual_issue_assumption() -> None:
    # #141: the bound is CUs x fp64 FMA lanes x 2 per clock, and the flop/s
    # figure is that at the rated clock -- AMD's dual-issue fp32 figure
    # (25.6 TFLOPS, and the 0.801 T derived from it) appears nowhere.
    pub = ceilings.load()["published"]["gfx1200"]
    b = pub["per_clock"]["fp64_flops_per_s"]
    assert b["flop_per_clk"] == (b["compute_units"]
                                 * b["fp64_fma_lanes_per_cu_per_clk"]
                                 * b["flop_per_fma"]) == 128
    assert pub["fp64_flops_per_s"] == pytest.approx(
        b["flop_per_clk"] * b["rated_clock_hz"])
    text = json.dumps(pub)
    assert "801" not in text.replace("0.801 TFLOPS figure that assumed", "")
    assert "ASSUMPTION" not in text and "SETTLED (#141)" in pub["note"]
    # no other card or quantity is bounded per clock
    for name, p in ceilings.load()["published"].items():
        assert set(p.get("per_clock", {})) <= (
            {"fp64_flops_per_s"} if name == "gfx1200" else set())
