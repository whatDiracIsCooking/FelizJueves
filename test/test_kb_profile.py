"""The kb / kbr ncu profile's parsers, invocation and reading rule (#574).

Fast tier, synthetic CSV: `measured.parse_ncu_metrics` (the details page,
ratios allowed, units checked), `measured.parse_ncu_source` (the source
page, per file, per line, per SASS instruction), the CPU cap on
`profile_commands` (opt-in: the roofline sweep's invocation unchanged) and
`kb_profile_verdicts` on every branch of `KB_PROFILE_DECLARED`. The sweep
itself is measure/measure_kblock_rolled.py's, behind devtools/measure.sh.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from felizjueves.performance import measured

HEAD = ('"ID","Process ID","Process Name","Host Name","Kernel Name",'
        '"Context","Stream","Block Size","Grid Size","Device","CC",'
        '"Section Name","Metric Name","Metric Unit","Metric Value"')
LAUNCH = ('"0","56369","rolled.bin","127.0.0.1","kbr_22_22","1","7",'
          '"(1, 32, 1)","(256, 1, 1)","0","8.6","Command line profiler metrics"')
UNITS = {"gpu__time_duration.sum": "ns",
         "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "inst",
         "launch__grid_size": "",
         "sm__warps_active.avg.pct_of_peak_sustained_active": "%"}
VALUES = {"gpu__time_duration.sum": "2495811392",
          "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "2.36",
          "launch__grid_size": "256",
          "sm__warps_active.avg.pct_of_peak_sustained_active": "7.78"}


def _details(values=VALUES, units=UNITS) -> str:
    return "\n".join(["==PROF== Connected to process 56369", HEAD, *(
        f'{LAUNCH},"{m}","{units[m]}","{v}"' for m, v in values.items())]) + "\n"


def test_the_details_page_is_read_by_name_with_ratios() -> None:
    ((kernel, got),) = measured.parse_ncu_metrics(_details(), UNITS)
    assert kernel == "kbr_22_22"
    assert got["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"] \
        == 2.36
    assert got["launch__grid_size"] == 256.0


def test_the_details_page_refuses_a_unit_a_gap_and_an_na() -> None:
    other = {**UNITS, "gpu__time_duration.sum": "usecond"}
    with pytest.raises(ValueError, match="usecond"):
        measured.parse_ncu_metrics(_details(units=other), UNITS)
    gap = {k: v for k, v in VALUES.items() if k != "launch__grid_size"}
    with pytest.raises(ValueError, match="launch__grid_size"):
        measured.parse_ncu_metrics(_details(values=gap), UNITS)
    na = {**VALUES, "launch__grid_size": "(!) n/a"}
    with pytest.raises(ValueError, match="n/a"):
        measured.parse_ncu_metrics(_details(values=na), UNITS)
    with pytest.raises(ValueError, match="header"):
        measured.parse_ncu_metrics("==ERROR== LaunchFailed\n", UNITS)


# --page source --print-source cuda,sass, as ncu 2025.3.1.0 prints it: a
# block per file, a header naming the metrics, a source row then its SASS,
# `...` rows for elided addresses (values rebuilt, layout as printed).
SRC = "\n".join([
    "==PROF== Connected to process 56998 (rolled.bin)",
    '"File Path","/usr/local/cuda/include/sm_30_intrinsics.hpp"',
    '"Function Name","kbr_22_22"',
    '"Line No","Source","Address","Source","smsp__pcsamp_sample_count",'
    '"smsp__pcsamp_warps_issue_stalled_wait"',
    '"409","ret = __nvvm_shfl_bfly_sync(mask, var, laneMask, c);","-","-","7","1"',
    '"","","0x7b2a80","      SHFL.BFLY PT, R7, R5, 0x10, 0x1f","7","1"',
    '"","","...","...","-","-"',
    '"File Path","/w/.kernel-tier/test-cache/85/fj_kb0_kbr_22_22.cu"',
    '"Function Name","kbr_22_22"',
    '"Line No","Source","Address","Source","smsp__pcsamp_sample_count",'
    '"smsp__pcsamp_warps_issue_stalled_wait"',
    '"692","for (int ie = bd_ptr[rbd]; ie < bd_ptr[(rbd + 1)]; ie += 1) {",'
    '"-","-","0","0"',
    '"693","acc = (acc + (R[hpq[(ip * 35 + bd_q[ie])]] * (sq * '
    'Ebd[(k * 336 + ie)])));","-","-","93","40"',
    '"","","0x7b9c7b280aa0","      LDG.E R15, [R20.64]","3","2"',
    '"","","...","...","-","-"',
    '"","","0x7b9c7b280ac0","      LDG.E.64 R16, [R22.64]","90","38"',
]) + "\n"


def test_the_source_page_is_read_per_file_line_and_instruction() -> None:
    files = measured.parse_ncu_source(SRC)
    assert len(files) == 2
    (lib,) = [v for p, v in files.items() if p.endswith(".hpp")]
    assert [ln.line for ln in lib] == [409]
    assert len(lib[0].sass) == 1  # the `...` row skipped
    (mine,) = [v for p, v in files.items() if p.endswith("kbr_22_22.cu")]
    assert [ln.line for ln in mine] == [692, 693]
    ebd = mine[1]
    assert "Ebd[" in ebd.source
    assert ebd.values == {"smsp__pcsamp_sample_count": 93.0,
                          "smsp__pcsamp_warps_issue_stalled_wait": 40.0}
    assert [t for t, _ in ebd.sass] == ["LDG.E R15, [R20.64]",
                                        "LDG.E.64 R16, [R22.64]"]
    assert ebd.sass[1][1]["smsp__pcsamp_sample_count"] == 90.0


def test_the_cpu_cap_is_opt_in_and_recorded() -> None:
    base = measured.profile_commands("cuda", "/usr/local/cuda/bin/ncu",
                                     ["./b"], kernel_regex="k",
                                     outdir=Path("/o"))
    (cmd,) = base
    assert cmd[0].endswith("ncu")  # the roofline sweep's invocation as before
    assert cmd[cmd.index("--metrics") + 1] == ",".join(measured.NCU_METRICS)
    (capped,) = measured.profile_commands(
        "cuda", "/usr/local/cuda/bin/ncu", ["./b"], kernel_regex="k",
        outdir=Path("/o"), cpus="0-15", ncu_args=["--metrics", "x"])
    assert capped[:3] == ["taskset", "-c", "0-15"]
    assert capped[capped.index("--metrics") + 1] == "x"
    shown = measured._shown(capped, {"./b": "<b>"}, "0-15")
    assert shown.startswith("OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                            "OPENBLAS_NUM_THREADS=1 taskset -c 0-15 ncu --csv")
    assert shown.endswith("<b>")
    assert measured._shown(cmd, {}, None).startswith("ncu --csv")
    hip = measured.profile_commands("hip", "rocprofv3", ["./b"],
                                    kernel_regex="k", outdir=Path("/o"),
                                    cpus="0-15")
    assert all(c[:3] == ["taskset", "-c", "0-15"] for c in hip)
    with pytest.raises(ValueError, match="ncu"):
        measured.profile_commands("hip", "rocprofv3", ["./b"],
                                  kernel_regex="k", outdir=Path("/o"),
                                  ncu_args=["--metrics", "x"])


def _counters(*, long=1.0, short=0.0, wait=0.0, hit=0.9, issue=10.0,
              warps=1.0) -> dict[str, float]:
    c = {measured.ncu_stall(r): 0.0 for r in measured.KB_PROFILE_STALLS}
    c[measured.ncu_stall("selected")] = 1.0
    c[measured.ncu_stall("long_scoreboard")] = long
    c[measured.ncu_stall("short_scoreboard")] = short
    c[measured.ncu_stall("wait")] = wait
    c["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"] = 1000.0
    c["l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum"] = 1000 * hit
    c["smsp__issue_active.avg.pct_of_peak_sustained_active"] = issue
    c["smsp__warps_active.avg.per_cycle_active"] = warps
    return c


def test_the_declared_terms() -> None:
    t = measured.kb_profile_terms(_counters(long=6.0, short=2.0, wait=1.0))
    assert t["S"] == 10.0  # 6 + 2 + 1 + selected's 1
    assert (t["mem"], t["long"], t["largest"]) == (0.8, 0.6, "long_scoreboard")
    assert t["hit"] == 0.9


@pytest.mark.parametrize(("kb", "kbr", "hot", "want"), [
    # H1 supported, H2 supported by the hit rate, H3 supported
    (_counters(long=1.0, hit=0.95, issue=20.0),
     _counters(long=10.0, issue=5.0), 0.9,
     {"H1": "supported", "H2": "supported", "H3": "supported"}),
    # H1 contradicted (kbr waits on fixed latency), H2 contradicted (kb's
    # local loads miss and stall it), H3 contradicted (no more issue)
    (_counters(long=8.0, hit=0.3, issue=5.0),
     _counters(long=0.5, wait=8.0, issue=5.0), 0.9,
     {"H1": "contradicted", "H2": "contradicted", "H3": "contradicted"}),
    # the bands between: hot loops cold, hit in [0.5, 0.8) with long > 0.25,
    # issue 1.3x; and H2 supported by long alone, whatever the hit rate
    (_counters(long=3.0, hit=0.6, issue=13.0),
     _counters(long=10.0, issue=10.0), 0.3,
     {"H1": "not resolved", "H2": "not resolved", "H3": "not resolved"}),
    (_counters(long=0.2, hit=0.1, issue=13.0, warps=2.0),
     _counters(long=1.0, short=1.0, wait=2.0, issue=5.0), 0.9,
     {"H1": "not resolved", "H2": "supported", "H3": "not resolved"}),
])
def test_the_verdicts_follow_the_declared_rule(kb, kbr, hot, want) -> None:
    got = measured.kb_profile_verdicts(kb, kbr, hot)
    assert {h: v["verdict"] for h, v in got.items()} == want


def test_the_rule_is_declared_for_all_three() -> None:
    d = measured.KB_PROFILE_DECLARED
    assert d["issue"] == 574 and d["otherwise"] == "not resolved"
    for h in ("H1", "H2", "H3"):
        assert {"hypothesis", "supported", "contradicted"} <= set(d[h])
    # every counter the terms read is one the sweep asks for
    for m in ("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
              "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum",
              "smsp__issue_active.avg.pct_of_peak_sustained_active",
              "smsp__warps_active.avg.per_cycle_active",
              *(measured.ncu_stall(r) for r in measured.KB_PROFILE_STALLS)):
        assert m in measured.KB_PROFILE_UNITS
