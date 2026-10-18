"""The card gate and the ceiling record, without a card (#109).

The probes themselves run in measure/measure_ceilings_device.py
(`devtools/measure.sh probes | ceilings`), and the tracked record is checked
by measure/measure_ceilings.py (`devtools/measure.sh check`). Here, on
synthetic probe output only: the gate refuses a contended or unreadable card,
a refused phase records NOTHING, a rate above the published peak is a probe
bug, and a recorded ceiling carries the provenance and card state the issue
asks for.
"""

from __future__ import annotations

import json

import pytest

from felizjueves import ceilings, gate
from felizjueves.device import SM_86

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


# -- the per-clock bound (#141) ------------------------------------------------


def _cycles_csv(per_chain: dict[int, float], ns: float = 1e8) -> str:
    """rocprofv3's counter_collection CSV: GRBM_GUI_ACTIVE per fj_fma launch."""
    head = ('"Dispatch_Id","Kernel_Name","Counter_Name","Counter_Value",'
            '"Start_Timestamp","End_Timestamp"')
    rows, d = [head], 0
    for chains, cycles in per_chain.items():
        for _ in range(ceilings.WARMUP + ceilings.REPEATS):
            d += 1
            rows.append(f'{d},"void fj_fma<{chains}>(int, double, double, '
                        f'double*)","{ceilings.CLOCK_COUNTER}",{cycles},'
                        f'1000,{1000 + ns:.0f}')
            rows.append(f'{d},"void fj_fma<{chains}>(int, double, double, '
                        f'double*)","SQ_WAVES",4096,1000,{1000 + ns:.0f}')
    return "\n".join(rows) + "\n"


def _pc_store() -> dict:
    store = _store()
    store["published"]["sm_86"]["fp64_flops_per_s"] = 0.5e9  # rated clock
    store["published"]["sm_86"]["per_clock"] = {
        "fp64_flops_per_s": {"flop_per_clk": 1.0, "rated_clock_hz": 0.5e9}}
    return store


# _output(configs=2): 1e9 and 2e9 flops per launch for chains 8 and 16
AT_BOUND = {8: 1e9, 16: 2e9}  # 1 flop per cycle each: on the bound


def test_the_per_clock_bound_counts_cycles_not_a_clock() -> None:
    runs = ceilings.parse_probe(_output(configs=2))
    got = ceilings.check_per_clock(runs, ceilings.parse_cycles(
        _cycles_csv(AT_BOUND)), 1.0)
    assert set(got) == {"8", "16"}
    assert got["16"]["flop_per_clk_max"] == pytest.approx(1.0)
    assert got["16"]["clock_hz_median"] == pytest.approx(2e9 / 0.1)
    with pytest.raises(ceilings.ProbeBug, match="per-clock bound"):
        ceilings.check_per_clock(runs, ceilings.parse_cycles(
            _cycles_csv({8: 1e9, 16: 1.9e9})), 1.0)
    with pytest.raises(ValueError, match="counted launches"):
        ceilings.check_per_clock(runs, ceilings.parse_cycles(
            _cycles_csv({8: 1e9})), 1.0)
    with pytest.raises(ValueError, match="did not run"):
        ceilings.check_per_clock(runs, ceilings.parse_cycles(
            _cycles_csv({**AT_BOUND, 32: 1e9})), 1.0)
    with pytest.raises(ValueError, match="no fj_fma"):
        ceilings.parse_cycles(_cycles_csv({}))


def test_a_per_clock_quantity_may_exceed_its_rated_figure_and_says_so() -> None:
    # The value (~2e11 flop/s) is far above the 0.5e9 rated-clock figure, yet
    # on the per-clock bound: recorded, with its ratio, never clamped.
    new, c, why = ceilings.record(
        _pc_store(), SM_86, "fp64_flops_per_s", lambda: _output(configs=2),
        backend="cuda", provenance={**PROV, "counter_tool": "rocprofv3 x"},
        read=_reads(0), wait_s=0, cycles=lambda: _cycles_csv(AT_BOUND),
        **QUIET_CPU, **NO_SLEEP)
    assert c is not None and why.startswith("recorded")
    raw = new["ceilings"]["sm_86"]["fp64_flops_per_s"]["raw"]
    assert raw["ratio_to_published"] > 1 and raw["best_ratio_to_published"] > 1
    pc = raw["per_clock"]
    assert pc["chosen"] == 16 and pc["tool"] == "rocprofv3 x"
    assert pc["bound_flop_per_clk"] == 1.0
    assert pc["value_at_bound_clock_hz"] == pytest.approx(c.value)
    assert (pc["gpu_util_before"], pc["gpu_util_after"]) == (0, 0)


def test_a_per_clock_quantity_needs_its_counter_and_a_quiet_counter_phase(
        ) -> None:
    store = _pc_store()
    with pytest.raises(ValueError, match="cycle-counter"):
        ceilings.record(store, SM_86, "fp64_flops_per_s",
                        lambda: _output(configs=2), backend="cuda",
                        provenance=PROV, read=_reads(0), wait_s=0, **QUIET_CPU,
                        **NO_SLEEP)
    # the timed phase quiet (3 before, discarded + 3 after), the counter
    # phase's before-reading contended: nothing recorded
    new, c, why = ceilings.record(
        store, SM_86, "fp64_flops_per_s", lambda: _output(configs=2),
        backend="cuda", provenance=PROV,
        read=_reads(0, 0, 0, 0, 0, 0, 0, 90), wait_s=0,
        cycles=lambda: _cycles_csv(AT_BOUND), **QUIET_CPU, **NO_SLEEP)
    assert c is None and "cycle counter" in why
    assert new is store and new["ceilings"] == {}
    with pytest.raises(ceilings.ProbeBug):
        ceilings.record(store, SM_86, "fp64_flops_per_s",
                        lambda: _output(configs=2), backend="cuda",
                        provenance=PROV, read=_reads(0), wait_s=0,
                        cycles=lambda: _cycles_csv({8: 1e9, 16: 1e9}),
                        **QUIET_CPU, **NO_SLEEP)
    assert store["ceilings"] == {}
