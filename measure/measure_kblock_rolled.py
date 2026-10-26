"""The rolled block K kernels against `kb`: compiled at d and f, timed at d and f.

    devtools/measure.sh kblock-rolled-compile [--backend cuda|hip]   a compiler
    devtools/measure.sh kblock-rolled [--backend cuda|hip]           timed
    devtools/measure.sh kblock-rolled-profile                       ncu (#574)

OPT-IN: each test skips unless FJ_KBLOCK_ROLLED_STORE names the store
(measure.sh sets it to config.sh's MEASURE_KBLOCK_ROLLED_STORE, the tracked
data/measured/kblock_rolled.json). The clock runs ALONE.

THE COMPILE (no card). #571's checkpoint is the compile, so each unit is
compiled on its own with the measured store's flags (`measured.BACKENDS`),
through `os.wait4` so the child's rusage carries its peak RSS (nvcc's
`cicc` / `ptxas`, hipcc's `clang -cc1`, all waited for under it): source
bytes, compile wall-clock, peak compiler RSS, registers, spill, stack.
`(dd|dd)` both kernels; `(ff|ff)` the rolled one only -- `kb_33_33` did not
finish in nvcc (killed in `cicc` at 2.8 h, 2.28 GB, #569's measurement,
quoted in the store's `kb_ff_not_compiled`, not re-run). One row per
(arch, toolchain, unit).

THE CLOCK, `(dd|dd)` only (the one class both kernels compile), on #569's
input -- 16 d shells on a 4 x 4 grid, 3 primitives a shell: 2,304 bra and
2,304 ket records, 256 blocks, `tau = 0` -- and #569's protocol: `ROUNDS`
rounds, each running `kb` then `rolled` (WARMUP untimed builds, REPEATS
timed), each phase behind `gate.wait_quiet` with its after-reading taken,
a contended phase retaken up to RETAKES times and then the row refused.
The timed binaries are first held to each other (1e-13 of max|K|). The
issue makes this clock a RECORD, not a gate: `kb` stays the default
through d; the rolled kernel need only win where nothing compiles now.

THE VERDICT RULE is the store's `declared`, written before any row.

THE DENSE-HERMITE ARM (#575), `kernel_kblock_rolled.build_kblock_dense_kernel`
(`kbd_*`): the (a,c) and (b,d) rows rolled, p and q unrolled over dense `E`.
Compiled at `(dd|dd)` and `(ff|ff)` beside the others; timed at `(dd|dd)`
as a third arm of the same rounds (`kb`, `rolled`, `dense`, interleaved),
and at `(ff|ff)` against `rolled` alone (`kb_33_33` does not compile) on
#575's f input -- 16 f shells on the same 4 x 4 grid, one primitive a
shell: 256 bra and 256 ket records, 256 blocks, `tau = 0` (with two
primitives a shell, 1,024 a side, one dense launch took over 30 s and the
run outlasted the driver's 300 s timeout). Its rule is
the store's `dense_declared` (DENSE_DECLARED), written into the store by
whichever of the compile or the clock touches it first, before any dense
row; a store whose declaration differs is refused. The clock rows are keyed
by (arch, toolchain, am). To add an arm: a builder in `ARM_BUILD`, its
`class_binaries` flags in `ARM_FLAGS`, and its name in a `CLOCKS` entry.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from kblock_driver import class_binaries, run_classes, timings
from test_k_ondevice import ORACLE_TOL
from test_kernel_device import BACKEND, _toolchain

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_kblock import build_kblock_kernel
from felizjueves.eri.k_exchange.kernel_kblock_rolled import (
    build_kblock_dense_kernel,
    build_kblock_rolled_kernel,
)
from felizjueves.eri.k_exchange.launch_kblock import kb_launches
from felizjueves.performance import gate, measured
from felizjueves.performance.gate import GPU_BUSY_MAX
from felizjueves.reference.pq import pq

STORE = "FJ_KBLOCK_ROLLED_STORE"
ARMS = ("kb", "rolled")  # #574's profile pair; the clocks' arms: CLOCKS
ROUNDS, WARMUP, REPEATS, RETAKES = 3, 3, 7, 3
SCHEMA = 1
AM = (2, 2, 2, 2)
COMPILE_UNITS = (("kb", (2, 2, 2, 2)), ("rolled", (2, 2, 2, 2)),
                 ("rolled", (3, 3, 3, 3)), ("dense", (2, 2, 2, 2)),
                 ("dense", (3, 3, 3, 3)))
AM_F = (3, 3, 3, 3)
COMPILE_TIMEOUT_S = 3600.0
BAND = 0.05  # the declared rule's band around 1

DECLARED = {
    "issue": 571,
    "hypothesis": "E read from global memory and the contraction as loops "
                  "make every l <= 3 K class compile in minutes; the loops "
                  "cost (dd|dd) wall-clock against the unrolled kb, which "
                  "keeps its compile-time constant indices",
    "rule": "per arch: the median over the timed runs of the (dd|dd) "
            "class's events, rolled / kb: 'faster' below 0.95, 'slower' "
            "above 1.05, else 'neutral'. A RECORD, not a gate (#571): kb "
            "stays the default through d whatever it reads",
    "expected": "slower: the rolled stages read E and the tables from "
                "global memory and R from local memory per term",
    "declared": "2026-10-26, before any row",
}

DENSE_DECLARED = {
    "issue": 575,
    "hypothesis": "with the (a,c) and (b,d) rows rolled and p, q unrolled "
                  "over dense E, every p + q is a compile-time constant: "
                  "the R array, the hpq / bd_q / ac_p indirection and "
                  "#574's dependent bd_q -> hpq -> R load chain are gone, "
                  "at the price of the structural zeros' multiplies; the "
                  "text grows as n_p n_q, not n_ac n_bd n_p n_q, so (ff|ff) "
                  "compiles",
    "rule": "per arch and class: the median over the timed runs of the "
            "class's events, dense / other for each other arm timed beside "
            "it (kb and rolled at (dd|dd), rolled at (ff|ff)): 'faster' "
            "below 0.95, 'slower' above 1.05, else 'neutral'. The form a "
            "class should use is the fastest arm that compiles, read from "
            "these medians; a RECORD, not a gate",
    "expected": "(dd|dd): dense faster than rolled, slower than kb; "
                "(ff|ff): dense faster than rolled",
    "declared": "2026-10-26, before any dense row",
}

KB_FF = {
    "issue": 569,
    "unit": "kb_33_33",
    "source_bytes": 22_200_000,
    "nvcc": "killed in cicc at 2.8 h, 2.28 GB RSS (E composed); 2.7 h, "
            "2.26 GB (E factored)",
    "re_run": False,
}


def _basis() -> tuple[Basis, np.ndarray]:
    """#569's input: 16 d shells on a 4 x 4 grid (2 bohr), 3 primitives
    each, and a symmetric, decaying density."""
    ex, co = (2.1, 0.7, 0.25), (0.3, 0.5, 0.4)
    shells = tuple(Shell(2, (2.0 * (i % 4), 2.0 * (i // 4), 0.3 * (i % 3)),
                         ex, co) for i in range(16))
    basis = Basis(shells)
    n = basis.n_ao
    D = np.array([[np.exp(-0.05 * abs(i - j)) * (1.0 + 0.1 * ((i * j) % 5))
                   for j in range(n)] for i in range(n)])
    return basis, (D + D.T) / 2.0


def _stats(xs) -> dict[str, float]:
    q1, med, q3 = (float(x) for x in np.percentile(xs, [25, 50, 75]))
    return {"median_ns": round(med), "q1_ns": round(q1), "q3_ns": round(q3),
            "n": len(xs)}


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 571,
            "about": __doc__.split("\n\n")[1], "declared": DECLARED,
            "kb_ff_not_compiled": KB_FF, "compile": [], "clock": []}


def _save(path: Path, store: dict, section: str, row: dict,
          key: tuple[str, ...]) -> None:
    rows = [r for r in store[section]
            if tuple(r[k] for k in key) != tuple(row[k] for k in key)]
    rows.append(row)
    rows.sort(key=lambda r: tuple(str(r[k]) for k in key))
    store[section] = rows
    path.write_text(json.dumps(store, indent=1) + "\n")


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the rolled-K records are opt-in: set {STORE} "
                    "(devtools/measure.sh kblock-rolled[-compile])")
    return Path(p)


ARM_BUILD = {"kb": build_kblock_kernel, "rolled": build_kblock_rolled_kernel,
             "dense": build_kblock_dense_kernel}
ARM_FLAGS = {"kb": {}, "rolled": {"rolled": True},
             "dense": {"rolled": True, "dense": True}}


def _build(arm: str, am):
    return ARM_BUILD[arm](am)


def _declare_dense(path: Path, store: dict) -> None:
    """`dense_declared` into the store before any dense row; refuse another."""
    declared = json.loads(json.dumps(DENSE_DECLARED))
    if "dense_declared" not in store:
        store["dense_declared"] = declared
        path.write_text(json.dumps(store, indent=1) + "\n")
    assert store["dense_declared"] == declared, (
        "the store's dense_declared is not DENSE_DECLARED: the dense rows "
        "would be read under another rule")


def _verdict(ratio: float) -> str:
    return ("faster" if ratio < 1 - BAND else
            "slower" if ratio > 1 + BAND else "neutral")


def _compile(backend: str, kern, workdir: Path) -> dict:
    """One unit through `os.wait4`: the measured flags, wall-clock, peak RSS
    of the compiler and everything it waited for, and the resource report."""
    b = measured.BACKENDS[backend]
    text = BACKEND[backend][0](kern, b.arch)
    src = workdir / f"{kern.name}.{b.ext}"
    obj = workdir / f"{kern.name}.o"
    src.write_text(text)
    cmd = b.command(b.compiler(), src, obj)
    log = workdir / f"{kern.name}.log"
    t0 = time.perf_counter()
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        # its own session, so the timer kills cicc / ptxas / clang too
        timer = threading.Timer(COMPILE_TIMEOUT_S, os.killpg,
                                (proc.pid, signal.SIGKILL))
        timer.start()
        # reaped by wait4 itself: the rusage of THIS child, its waited-for
        # descendants included (RUSAGE_BOTH), never another unit's
        _, status, ru = os.wait4(proc.pid, 0)
        timer.cancel()
    proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0
    out = log.read_text()
    if wall >= COMPILE_TIMEOUT_S:
        pytest.fail(f"{kern.name}: not compiled within {COMPILE_TIMEOUT_S} s")
    peak_kb = ru.ru_maxrss
    assert proc.returncode == 0, out[-4000:]
    raw = (measured.parse_ptxas(out, kern.name) if backend == "cuda"
           else measured.parse_amd_remarks(out, kern.name))
    regs, store, load, _ = measured._numbers(b, raw)
    return {"unit": kern.name, "source_bytes": len(text),
            "compile_s": round(wall, 2), "peak_rss_kb": peak_kb,
            "registers": regs, "spill_store": store, "spill_load": load,
            "stack_bytes": raw.get("stack_frame_bytes",
                                   raw.get("ScratchSize"))}


def _compile_rows(backend: str, request, tmp_path: Path) -> None:
    path = _store_path()
    _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    store = _load(path)
    _declare_dense(path, store)
    head = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
            "host": measured.host(),
            "date": datetime.date.today().isoformat()}
    for arm, am in COMPILE_UNITS:
        r = _compile(backend, _build(arm, am), tmp_path)
        row = {**head, "arm": arm, "am": list(am), **r}
        _save(path, store, "compile", row, ("arch", "toolchain", "unit"))
        print(f"\n{backend} {b.arch.name} {r['unit']}: {r['source_bytes']} B "
              f"source, {r['compile_s']} s, {r['peak_rss_kb']} kB peak, "
              f"r{r['registers']} spill {r['spill_store']}/{r['spill_load']} "
              f"stack {r['stack_bytes']}")


def _basis_f() -> tuple[Basis, np.ndarray]:
    """#575's f input: `_basis`'s grid and density shape with 16 f shells,
    one primitive each."""
    ex, co = (0.9,), (1.0,)
    shells = tuple(Shell(3, (2.0 * (i % 4), 2.0 * (i // 4), 0.3 * (i % 3)),
                         ex, co) for i in range(16))
    basis = Basis(shells)
    n = basis.n_ao
    D = np.array([[np.exp(-0.05 * abs(i - j)) * (1.0 + 0.1 * ((i * j) % 5))
                   for j in range(n)] for i in range(n)])
    return basis, (D + D.T) / 2.0


# (am, input, its records a side, its blocks, the arms in round order)
CLOCKS = {
    "dd": (AM, _basis, 2304, 256, ("kb", "rolled", "dense")),
    "ff": (AM_F, _basis_f, 256, 256, ("rolled", "dense")),
}


def _clock(backend: str, request, run_device_binary, tmp_path: Path,
           which: str = "dd") -> None:
    path = _store_path()
    tc = _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    am, make, n_rec, n_blk, arms_run = CLOCKS[which]
    store = _load(path)
    if "dense" in arms_run:
        _declare_dense(path, store)
    basis, D = make()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=0.0,
                       build=build_kblock_kernel if "kb" in arms_run else None)
    assert [c.am for c in plan] == [am]
    (cl,) = plan
    assert len(cl.bras) == len(cl.kets) == n_rec and len(cl.blocks) == n_blk
    parts = {arm: class_binaries(tc, replace(plan, classes=(cl,)), basis, D,
                                 backend, tmp_path / arm, with_k=False,
                                 **ARM_FLAGS[arm])
             for arm in arms_run}
    Ks = {arm: run_classes(parts[arm], basis, "kb", run_device_binary,
                           backend)[0][0] for arm in arms_run}
    ref = Ks[arms_run[0]]
    rels = {arm: float(np.max(np.abs(Ks[arm] - ref)) / np.max(np.abs(ref)))
            for arm in arms_run[1:]}
    assert np.max(np.abs(ref)) > 0.0, "K is zero"
    for arm, rel in rels.items():
        assert rel <= ORACLE_TOL, f"{arm} vs {arms_run[0]} {rel:.2e}"

    per: dict[str, list[float]] = {a: [] for a in arms_run}
    readings, discarded = [], 0
    for _ in range(ROUNDS):
        for arm in arms_run:
            for take in range(RETAKES + 1):
                try:
                    busy, util = gate.wait_quiet(
                        backend, say=lambda s: print(s, flush=True))
                except gate.Contended as e:
                    discarded += 1
                    print(f"  DISCARDED {arm}: {e} (take {take + 1})")
                    continue
                _, outs, _ = run_classes(parts[arm], basis, "kb",
                                         run_device_binary, backend,
                                         warmup=WARMUP, reps=REPEATS)
                after = gate.after_reading(backend)
                if util is not None and after is not None \
                        and after <= GPU_BUSY_MAX:
                    break
                discarded += 1
                print(f"  DISCARDED {arm}: card {util} -> {after}%")
            else:
                pytest.fail(f"{arm}: contended through {RETAKES + 1} takes "
                            "-- not recorded")
            readings.append({"arm": arm, "cpu_busy_permille":
                             round(1000 * busy), "gpu_util_before": util,
                             "gpu_util_after": after})
            (out,) = outs
            per[arm].extend(timings(out)["0"])

    arms = {arm: _stats(per[arm]) for arm in arms_run}

    def over(a: str, o: str) -> float:
        return arms[a]["median_ns"] / arms[o]["median_ns"]

    row = {
        "arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
        "gpu": b.gpu(), "host": measured.host(),
        "date": datetime.date.today().isoformat(),
        "am": list(am), "n_ao": basis.n_ao, "n_bra": n_rec, "n_ket": n_rec,
        "n_blocks": n_blk, "tau": 0.0, "reference_arm": arms_run[0],
        "rel_vs_reference": rels,
        "rounds": ROUNDS, "warmup": WARMUP, "repeats": REPEATS,
        "gate": readings, "gate_discarded_runs": discarded, "arms": arms,
    }
    if {"kb", "rolled"} <= set(arms_run):  # #571's rule
        row["rolled_vs_kb_rel"] = rels["rolled"]
        row["rolled_over_kb"] = round(over("rolled", "kb"), 4)
        row["verdict"] = _verdict(over("rolled", "kb"))
    if "dense" in arms_run:  # #575's rule, DENSE_DECLARED
        row["dense"] = {
            o: {"dense_over": round(over("dense", o), 4),
                "verdict": _verdict(over("dense", o))}
            for o in arms_run if o != "dense"}
        row["fastest"] = min(arms_run, key=lambda a: arms[a]["median_ns"])
    store = _load(path)
    _save(path, store, "clock", row, ("arch", "toolchain", "am"))
    print(f"\n{backend} {b.arch.name} {am}: " + ", ".join(
        f"{a} {arms[a]['median_ns'] / 1e6:.3f} ms" for a in arms_run)
        + f"; {json.dumps(row.get('dense'))}; -> {path}")


@pytest.mark.cuda
def test_cuda_kblock_rolled_compile(request, tmp_path) -> None:
    _compile_rows("cuda", request, tmp_path)


@pytest.mark.hip
def test_hip_kblock_rolled_compile(request, tmp_path) -> None:
    _compile_rows("hip", request, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_kblock_rolled_clock(request, run_device_binary,
                                  tmp_path) -> None:
    _clock("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_kblock_rolled_clock(request, run_device_binary,
                                 tmp_path) -> None:
    _clock("hip", request, run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_kblock_rolled_clock_ff(request, run_device_binary,
                                     tmp_path) -> None:
    """#575: `(ff|ff)`, the dense-Hermite kernel against the rolled one."""
    _clock("cuda", request, run_device_binary, tmp_path, "ff")


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_kblock_rolled_clock_ff(request, run_device_binary,
                                    tmp_path) -> None:
    _clock("hip", request, run_device_binary, tmp_path, "ff")


# --- THE PROFILE (#574): kb_22_22 against kbr_22_22 under ncu ---------------
#
#     devtools/measure.sh kblock-rolled-profile     (CUDA only, timed kind)
#
# The clock's input and binaries (MODE 0, `rolled=` False / True), each
# held to the other first, then per kernel three ncu runs of ONE launch
# (`0 0 1`: no warm-up, one rep, `--launch-count 1`), in this order, never
# two at once and nothing compiling beside them: `details` (the counters of
# `measured.KB_PROFILE_UNITS`), `pcsamp` (PC-sampled warp states per source
# line and SASS instruction, `measured.KB_PROFILE_PCSAMP`) and `sectors`
# (`measured.KB_PROFILE_SECTORS`, SASS-patched per-instruction requests and
# sectors: where the Ebd load's sectors per request would be read; when ncu
# cannot, its reason is recorded). Each run behind `gate.wait_quiet`, its
# after-reading taken, a contended run (or one ncu ends in LaunchFailed,
# seen intermittently here) retaken up to PROFILE_RETAKES times
# and then nothing written. The profiler and the binary are pinned to
# PROFILE_CPUS (`measured.profile`'s CPU CAP).
#
# THE READING RULE is `measured.KB_PROFILE_DECLARED`, copied into the store
# as `profile_declared` before any row; a store whose declaration differs
# is refused. One `profile` row per (arch, toolchain, unit); one
# `profile_reading` row per (arch, toolchain) with the three verdicts.

PROFILE_CPUS = "0-15"
PROFILE_RETAKES = 3
HOT_SHARE = 0.01  # a source line kept in the row: >= 1% of the samples
_EBD = re.compile(r"Ebd\[\(k \* (\d+) \+ ie\)\]")
_E_GLOBAL = re.compile(r"\bE(bd|ac)\[\(")


def _gated(kind: str, unit: str, run, readings: list) -> tuple[object, int]:
    """`run()` behind the gate; (its result, runs discarded)."""
    discarded = 0
    for take in range(PROFILE_RETAKES + 1):
        try:
            busy, util = gate.wait_quiet(
                "cuda", say=lambda s: print(s, flush=True))
        except gate.Contended as e:
            discarded += 1
            print(f"  DISCARDED {unit} {kind}: {e} (take {take + 1})")
            continue
        try:
            got = run()
        except measured.CouldNotMeasure as e:
            # seen intermittently on kbr's details / pcsamp replays here:
            # the same run passes on a retake, so it is retaken like a
            # contended one; anything else is a refusal
            if "LaunchFailed" not in str(e):
                raise
            discarded += 1
            print(f"  DISCARDED {unit} {kind}: ncu LaunchFailed "
                  f"(take {take + 1})")
            continue
        after = gate.after_reading("cuda")
        if gate.is_idle(util) and gate.is_idle(after):
            readings.append({"unit": unit, "run": kind,
                             "cpu_busy_permille": round(1000 * busy),
                             "gpu_util_before": util,
                             "gpu_util_after": after})
            return got, discarded
        discarded += 1
        print(f"  DISCARDED {unit} {kind}: card {util} -> {after}%")
    pytest.fail(f"{unit} {kind}: contended through {PROFILE_RETAKES + 1} "
                "takes -- nothing written")


def _nonzero(vals) -> dict[str, float]:
    pre = "smsp__pcsamp_warps_issue_stalled_"
    return {k.removeprefix(pre): v for k, v in vals.items()
            if v and k != "smsp__pcsamp_sample_count"}


def _pcsamp(files: dict, unit_file: str) -> dict:
    """The kernel's PC samples: total, the generated unit's hot lines, and
    the lines that read Ebd[ / Eac[ with their SASS."""
    n = "smsp__pcsamp_sample_count"
    total = sum(ln.values.get(n, 0.0) for lines in files.values()
                for ln in lines)
    assert total > 0, "no PC samples"
    (mine,) = [lines for p, lines in files.items() if p.endswith(unit_file)]

    def row(ln, sass: bool) -> dict:
        r = {"line": ln.line, "source": ln.source.strip()[:200],
             "samples": int(ln.values.get(n, 0)),
             "share": round(ln.values.get(n, 0) / total, 4),
             "stalls": _nonzero(ln.values)}
        if sass:
            # the instructions holding >= 0.1% of the kernel's samples
            r["sass"] = [{"sass": t, "samples": int(v.get(n, 0)),
                          "stalls": _nonzero(v)} for t, v in ln.sass
                         if v.get(n, 0) >= 1e-3 * total]
        return r

    hot = [row(ln, False) for ln in mine if ln.values.get(n, 0) >= HOT_SHARE * total]
    # the rolled loops read E from global memory, `Ebd[(k * 336 + ie)]`;
    # kb's inlined E is a local array, `Ebd[0][0]`, and is not matched
    loops = [ln for ln in mine if _E_GLOBAL.search(ln.source)]
    return {"samples": int(total),
            "hot_lines": sorted(hot, key=lambda r: -r["samples"]),
            "e_lines": [row(ln, True) for ln in loops],
            "e_share": round(sum(ln.values.get(n, 0) for ln in loops) / total,
                             4)}


def _sectors(files: dict, unit_file: str) -> dict:
    """Per 64-bit global load on the Ebd line: requests, sectors, ratio."""
    (mine,) = [lines for p, lines in files.items() if p.endswith(unit_file)]
    req, sec = KB_SECTOR_REQ, KB_SECTOR_SEC
    out = []
    for ln in mine:
        if not _EBD.search(ln.source):
            continue
        for t, v in ln.sass:
            if t.split()[0].lstrip("@P!0123456789T ").startswith("LDG") \
                    and ".64" in t and v.get(req):
                out.append({"line": ln.line, "sass": t,
                            "requests": int(v[req]), "sectors": int(v[sec]),
                            "sectors_per_request": round(v[sec] / v[req], 3)})
    return {"ebd_loads": out}


KB_SECTOR_REQ, KB_SECTOR_SEC = measured.KB_PROFILE_SECTORS


def _profile_kernel(arm: str, part, unit: str, text: str, work: Path,
                    readings: list) -> tuple[dict, int]:
    argv = [str(part.binary), str(part.staged), str(work / f"{arm}.out"),
            "0", "0", "1"]
    unit_file = f"{unit}.cu"
    regex = f"^{unit}$"
    once = ["--launch-count", "1"]
    source = ["--page", "source", "--print-source", "cuda,sass"]

    def ncu(args):
        return measured.ncu_raw(argv, ncu_args=[*once, *args],
                                kernel_regex=regex, cpus=PROFILE_CPUS)

    discarded = 0
    try:
        (det, tool, c_det), d = _gated("details", unit, lambda: ncu(
            ["--metrics", ",".join(measured.KB_PROFILE_UNITS)]), readings)
        discarded += d
        (pcs, _, c_pcs), d = _gated("pcsamp", unit, lambda: ncu(
            ["--metrics", ",".join(measured.KB_PROFILE_PCSAMP), *source]),
            readings)
        discarded += d
    except measured.CouldNotMeasure as e:
        pytest.skip(f"{e} -- nothing written")
    ((kernel, counters),) = measured.parse_ncu_metrics(
        det, measured.KB_PROFILE_UNITS)
    assert kernel == unit, kernel

    def sectors():
        try:
            return ncu(["--metrics", ",".join(measured.KB_PROFILE_SECTORS),
                        *source])
        except measured.CouldNotMeasure as e:
            return e

    got, d = _gated("sectors", unit, sectors, readings)
    discarded += d
    if isinstance(got, measured.CouldNotMeasure):
        sec = {"could_not_measure": str(got)}
        c_sec = None
    else:
        sec = _sectors(measured.parse_ncu_source(got[0]), unit_file)
        c_sec = got[2]
    m = _EBD.search(text)
    if m:  # the stride the text fixes, not a measurement
        sec["ebd_from_text"] = {
            "index": m.group(0), "lane": "k = k0 + threadIdx.y, blockDim "
            "(1, 32): consecutive lanes, consecutive ket records",
            "lane_stride_bytes": 8 * int(m.group(1)),
            "sectors_per_full_warp_request": 32 if 8 * int(m.group(1)) >= 32
            else None}
    row = {
        "unit": unit, "arm": arm, "counter_tool": tool,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "command": {"details": c_det, "pcsamp": c_pcs, "sectors": c_sec},
        "counters": counters,
        "terms": measured.kb_profile_terms(counters),
        "pcsamp": _pcsamp(measured.parse_ncu_source(pcs), unit_file),
        "sectors": sec,
    }
    return row, discarded


def _summary(kb: dict, kbr: dict) -> dict:
    """The few numbers a reader (and #575, #576) wants first, per kernel:
    derived from the two rows, never measured separately."""
    def per(row: dict) -> dict:
        c = row["counters"]
        g = "l1tex__t_{}_pipe_lsu_mem_global_op_ld{}.sum"
        loc = "l1tex__t_{}_pipe_lsu_mem_local_op_{}.sum"
        return {
            "ms_under_ncu": round(c["gpu__time_duration.sum"] / 1e6, 1),
            "warp_inst": c["smsp__inst_executed.sum"],
            "fp64_flops": (c[measured.NCU_DADD] + c[measured.NCU_DMUL]
                           + 2 * c[measured.NCU_DFMA]),
            "global_ld_requests": c[g.format("requests", "")],
            "global_ld_sectors_per_request": round(
                c[g.format("sectors", "")] / c[g.format("requests", "")], 3),
            "global_ld_l1_hit": round(c[g.format("sectors", "_lookup_hit")]
                                      / c[g.format("sectors", "")], 4),
            "local_ld_sectors": c[loc.format("sectors", "ld")],
            "local_st_sectors": c[loc.format("sectors", "st")],
            "local_ld_l1_hit": round(
                c[loc.format("sectors", "ld_lookup_hit")]
                / c[loc.format("sectors", "ld")], 4),
            "shared_ld_wavefronts":
                c["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"],
            "dram_bytes_read": c[measured.NCU_READ],
            "cycles_per_issue": c[
                "smsp__average_warp_latency_per_inst_issued.ratio"],
            "issue_active_pct": c[
                "smsp__issue_active.avg.pct_of_peak_sustained_active"],
            "achieved_occupancy_pct": c[
                "sm__warps_active.avg.pct_of_peak_sustained_active"],
            "theoretical_occupancy_pct": c[
                "sm__maximum_warps_per_active_cycle_pct"],
        }

    a, b = per(kb), per(kbr)
    e = kbr["pcsamp"]["e_lines"]
    ebd = [x for x in e if _EBD.search(x["source"])]
    return {
        "kb": a, "kbr": b,
        "kbr_over_kb": {k: round(b[k] / a[k], 3) for k in (
            "ms_under_ncu", "warp_inst", "fp64_flops", "cycles_per_issue",
            "issue_active_pct")},
        "ebd": {
            "line": ebd[0]["line"] if ebd else None,
            "sample_share": ebd[0]["share"] if ebd else None,
            "sectors_per_request_measured":
                kbr["sectors"].get("ebd_loads") or None,
            "why_not_measured": kbr["sectors"].get("could_not_measure"),
            "from_text": kbr["sectors"].get("ebd_from_text"),
        },
    }


def _profile(request, run_device_binary, tmp_path: Path) -> None:
    path = _store_path()
    tc = _toolchain("cuda", request)
    b = measured.BACKENDS["cuda"]
    store = _load(path)
    declared = json.loads(json.dumps(measured.KB_PROFILE_DECLARED))
    if "profile_declared" not in store:
        store["profile_declared"] = declared
        store.setdefault("profile", [])
        store.setdefault("profile_reading", [])
        path.write_text(json.dumps(store, indent=1) + "\n")
    assert store["profile_declared"] == declared, (
        "the store's profile_declared is not measured.KB_PROFILE_DECLARED: "
        "the rows would be read under another rule")

    basis, D = _basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=0.0, build=build_kblock_kernel)
    (cl,) = plan
    assert cl.am == AM and len(cl.blocks) == 256
    assert len(cl.bras) == len(cl.kets) == 2304
    # compiled one after the other, all before the first ncu run
    parts = {arm: class_binaries(tc, replace(plan, classes=(cl,)), basis, D,
                                 "cuda", tmp_path / arm, jobs=1,
                                 rolled=arm == "rolled", with_k=False)
             for arm in ARMS}
    K0, K1 = (run_classes(parts[arm], basis, "kb", run_device_binary,
                          "cuda")[0][0] for arm in ARMS)
    rel = float(np.max(np.abs(K1 - K0)) / np.max(np.abs(K0)))
    assert np.max(np.abs(K0)) > 0.0 and rel <= ORACLE_TOL, rel

    head = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
            "gpu": b.gpu(), "host": measured.host(),
            "date": datetime.date.today().isoformat(), "am": list(AM),
            "n_ao": basis.n_ao, "n_bra": 2304, "n_ket": 2304,
            "n_blocks": 256, "tau": 0.0, "cpus": PROFILE_CPUS,
            "launch": "one launch: MODE 0, warm-up 0, reps 1, "
                      "--launch-count 1"}
    rows = {}
    for arm in ARMS:
        (part,) = parts[arm]
        kern = part.drv.classes[0].kb
        text = BACKEND["cuda"][0](kern, b.arch)
        readings: list = []
        work = tmp_path / f"{arm}-ncu"
        work.mkdir()
        row, discarded = _profile_kernel(arm, part, kern.name, text, work,
                                         readings)
        rows[arm] = {**head, **row, "gate": readings,
                     "gate_discarded_runs": discarded}
        t = row["terms"]
        print(f"\n{kern.name}: {row['counters']['gpu__time_duration.sum'] / 1e6:.0f}"
              f" ms under ncu; S {t['S']:.2f} mem {t['mem']:.3f} long "
              f"{t['long']:.3f} ({t['largest']}), local-ld hit {t['hit']:.3f},"
              f" issue {t['issue']}%, warps {t['warps']}", flush=True)

    verdicts = measured.kb_profile_verdicts(
        rows["kb"]["counters"], rows["rolled"]["counters"],
        rows["rolled"]["pcsamp"]["e_share"])
    store = _load(path)
    for r in rows.values():
        _save(path, store, "profile", r, ("arch", "toolchain", "unit"))
    _save(path, store, "profile_reading",
          {"arch": head["arch"], "toolchain": head["toolchain"],
           "date": head["date"], **verdicts,
           "summary": _summary(rows["kb"], rows["rolled"])},
          ("arch", "toolchain"))
    for h, v in verdicts.items():
        print(f"{h}: {v}")
    print(f"-> {path}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_kblock_rolled_profile(request, run_device_binary,
                                    tmp_path) -> None:
    """#574: kb_22_22 and kbr_22_22 under ncu, one launch each (RTX 3080
    only: the RX 9060 XT has no fp64 or byte counters, #140)."""
    _profile(request, run_device_binary, tmp_path)


TRACKED = Path(__file__).resolve().parent.parent / "data/measured/kblock_rolled.json"


def test_the_profile_reading_is_what_its_rows_recompute_to() -> None:
    """`devtools/measure.sh check`: the tracked store's profile section was
    read under today's declared rule, and each `profile_reading` row is
    `kb_profile_verdicts` of its two `profile` rows."""
    store = json.loads(TRACKED.read_text())
    if "profile_declared" not in store:
        pytest.skip("no #574 profile in the tracked store")
    assert store["profile_declared"] == json.loads(
        json.dumps(measured.KB_PROFILE_DECLARED))
    assert store["profile_reading"], "declared, but nothing read"
    for reading in store["profile_reading"]:
        at = (reading["arch"], reading["toolchain"])
        rows = {r["arm"]: r for r in store["profile"]
                if (r["arch"], r["toolchain"]) == at}
        assert set(rows) == set(ARMS), at
        for r in rows.values():
            assert r["terms"] == json.loads(json.dumps(
                measured.kb_profile_terms(r["counters"])))
        got = measured.kb_profile_verdicts(
            rows["kb"]["counters"], rows["rolled"]["counters"],
            rows["rolled"]["pcsamp"]["e_share"])
        assert {h: reading[h] for h in got} == json.loads(json.dumps(got))
