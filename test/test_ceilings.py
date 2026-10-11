"""The card gate and the ceiling record, without a card (#109).

The probes themselves run in test_ceilings_device.py (`gpu`). Here: the gate
refuses a contended or unreadable card, a refused phase records NOTHING, a
rate above the published peak is a probe bug, and the recorded ceilings carry
the provenance and card state the issue asks for.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from conftest import VRAM_PROBE

from felizjueves import ceilings, gate
from felizjueves.device import ARCHS, GFX1200, QUANTITIES, SM_86

NO_SLEEP = {"sleep": lambda s: None}
QUIET_CPU = {"cpu_busy": lambda: 0.01}


def _reads(*values):
    """An SMI reader returning `values` in turn, then the last forever."""
    it = iter(values)
    last = [values[-1]]

    def read(backend: str):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]
    return read


def _output(mode: str = "fma", per_launch: float = 1e9, ns: float = 1e7,
            configs: int = 1) -> str:
    lines = []
    for c in range(configs):
        unit = "flops" if mode == "fma" else "bytes"
        lines.append(f"probe {mode} chains {8 << c} grid 1 block 1 sms 1 "
                     f"{unit}_per_launch {per_launch * (c + 1):.0f}")
        lines += [f"rep {i} ns {ns + i:.0f}" for i in range(ceilings.REPEATS)]
    return "\n".join([*lines, "probe ok"]) + "\n"


def _store() -> dict:
    return {"schema": ceilings.SCHEMA,
            "published": {"sm_86": {"fp64_flops_per_s": 465e9,
                                    "mem_bandwidth_bytes_per_s": 760e9}},
            "ceilings": {}}


PROV = {"probe": "p", "compile": "nvcc ...", "source_sha256": "0" * 64,
        "toolchain": "nvcc 13.0", "gpu": "card", "date": "2026-10-11"}


# -- the gate -----------------------------------------------------------------


def test_none_is_unknown_never_idle() -> None:
    assert not gate.is_idle(None)
    assert gate.is_idle(0) and gate.is_idle(gate.GPU_BUSY_MAX)
    assert not gate.is_idle(gate.GPU_BUSY_MAX + 1)
    assert gate.card_util("cuda", read=_reads(None, None, None), **NO_SLEEP) is None
    assert gate.card_util("cuda", read=_reads(None, 12, 3), **NO_SLEEP) == 12


@pytest.mark.parametrize("reading", [gate.GPU_BUSY_MAX + 1, 100, None])
def test_the_gate_refuses_a_contended_or_unreadable_card(reading) -> None:
    with pytest.raises(gate.Contended, match="contended|unreadable"):
        gate.wait_card("cuda", read=_reads(reading), wait_s=0, say=lambda s: None,
                       **NO_SLEEP)


def test_the_gate_refuses_a_busy_cpu() -> None:
    with pytest.raises(gate.Contended, match="CPU"):
        gate.wait_quiet("cuda", read=_reads(0), wait_s=0,
                        cpu_busy=lambda: 0.9, **NO_SLEEP)


@pytest.mark.parametrize("after", [gate.GPU_BUSY_MAX + 1, None])
def test_gated_refuses_on_the_after_reading(after) -> None:
    ran = []
    # before: three idle samples; after: one discarded read, then three more
    read = _reads(0, 0, 0, 99, after, after, after)
    with pytest.raises(gate.Contended, match="after the phase"):
        gate.gated(lambda: ran.append(1), "cuda", read=read, wait_s=0,
                   **QUIET_CPU, **NO_SLEEP)
    assert ran == [1]


def test_gated_lets_a_quiet_phase_through_with_both_readings() -> None:
    read = _reads(0, 4, 2, 99, 1, 0, 0)  # the 99 is the discarded read
    out, phase = gate.gated(lambda: "x", "cuda", read=read, wait_s=0,
                            **QUIET_CPU, **NO_SLEEP)
    assert out == "x" and (phase.gpu_before, phase.gpu_after) == (4, 1)
    assert phase.cpu_before == 0.01


def test_the_wallclock_path_reads_the_same_gate() -> None:
    # One helper, not a copy: test_kernel_device.py's names ARE the gate's.
    import test_kernel_device as tkd

    assert tkd.GPU_BUSY_MAX is gate.GPU_BUSY_MAX
    assert tkd.GPU_SETTLE_S is gate.GPU_SETTLE_S
    assert tkd._card_util is gate.card_util and tkd._cpu_busy is gate.cpu_busy
    assert (tkd.GPU_BUSY_MAX, tkd.GPU_QUIET_WAIT_S, tkd.GPU_SETTLE_S,
            tkd.BUSY_MAX, tkd.QUIET_WAIT_S) == (30, 900.0, 1.0, 0.10, 120.0)


# -- the record ---------------------------------------------------------------


@pytest.mark.parametrize("reads", [
    (80,),                          # contended before: never let through
    (None,),                        # unreadable before
    (0, 0, 0, 0, 75, 75, 75),       # quiet before, contended after
    (0, 0, 0, 0, None, None, None),  # quiet before, unreadable after
])
def test_a_refused_probe_records_nothing(reads) -> None:
    store = _store()
    before = json.dumps(store)
    new, ceiling, why = ceilings.record(
        store, SM_86, "fp64_flops_per_s", lambda: _output(), backend="cuda",
        provenance=PROV, read=_reads(*reads), wait_s=0, **QUIET_CPU, **NO_SLEEP)
    assert ceiling is None and why.startswith("REFUSED sm_86 fp64_flops_per_s")
    assert json.dumps(new) == before and new["ceilings"] == {}


def test_a_quiet_probe_records_its_value_provenance_and_card_state() -> None:
    store = _store()
    new, c, why = ceilings.record(
        store, SM_86, "fp64_flops_per_s", lambda: _output(configs=2),
        backend="cuda", provenance=PROV, read=_reads(3, 3, 3, 90, 2, 1, 0),
        wait_s=0, **QUIET_CPU, **NO_SLEEP)
    assert c is not None and why.startswith("recorded")
    assert store["ceilings"] == {}  # the input is not mutated
    row = new["ceilings"]["sm_86"]["fp64_flops_per_s"]
    # the second configuration is twice the work in the same time: chosen
    assert row["value"] == pytest.approx(2e9 / ((1e7 + 15) * 1e-9))
    assert row["raw"]["probe_header"].startswith("probe fma chains 16")
    assert len(row["raw"]["configs_median"]) == 2
    assert (row["raw"]["gpu_util_before"], row["raw"]["gpu_util_after"]) == (3, 2)
    assert row["raw"]["ratio_to_published"] == pytest.approx(row["value"] / 465e9)
    assert row["kind"] == "achievable" and row["unit"] == "flop/s"
    assert row["command"] == ceilings.COMMAND.format(backend="cuda")
    assert row["repeats"] == ceilings.REPEATS and row["arch"] == "sm_86"
    for k in ("host", "date", "toolchain", "gpu", "source_sha256", "probe"):
        assert row[k]


def test_a_rate_above_the_published_peak_is_a_probe_bug() -> None:
    with pytest.raises(ceilings.ProbeBug, match="bug in the probe"):
        ceilings.check_published("fp64_flops_per_s", 466e9,
                                 {"fp64_flops_per_s": 465e9})
    assert ceilings.check_published("fp64_flops_per_s", 232.5e9,
                                    {"fp64_flops_per_s": 465e9}) == 0.5
    store = _store()
    with pytest.raises(ceilings.ProbeBug):  # 1e9 B / 1 ms = 1e12 B/s > 760e9
        ceilings.record(store, SM_86, "mem_bandwidth_bytes_per_s",
                        lambda: _output("triad", ns=1e6), backend="cuda",
                        provenance=PROV, read=_reads(0), wait_s=0, **QUIET_CPU,
                        **NO_SLEEP)
    assert store["ceilings"] == {}


def test_the_wrong_probe_for_a_quantity_is_refused() -> None:
    with pytest.raises(ValueError, match="triad"):
        ceilings.record(_store(), SM_86, "mem_bandwidth_bytes_per_s",
                        lambda: _output("fma"), backend="cuda", provenance=PROV,
                        read=_reads(0), wait_s=0, **QUIET_CPU, **NO_SLEEP)


def test_an_incomplete_probe_output_is_an_error() -> None:
    with pytest.raises(ValueError):
        ceilings.parse_probe(_output().replace("probe ok", ""))
    with pytest.raises(ValueError):
        ceilings.parse_probe(_output().replace("rep 3 ", "xrep 3 "))


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
            assert 0 < row["value"] <= raw["best"] <= published[q]
            assert raw["published"] == published[q]
            assert raw["ratio_to_published"] == pytest.approx(
                row["value"] / published[q])
