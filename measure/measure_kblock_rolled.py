"""The rolled block K kernel against `kb`: compiled at d and f, timed at d (#571).

    devtools/measure.sh kblock-rolled-compile [--backend cuda|hip]   a compiler
    devtools/measure.sh kblock-rolled [--backend cuda|hip]           timed

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
"""

from __future__ import annotations

import datetime
import json
import os
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
    build_kblock_rolled_kernel,
)
from felizjueves.eri.k_exchange.launch_kblock import kb_launches
from felizjueves.performance import gate, measured
from felizjueves.performance.gate import GPU_BUSY_MAX
from felizjueves.reference.pq import pq

STORE = "FJ_KBLOCK_ROLLED_STORE"
ARMS = ("kb", "rolled")
ROUNDS, WARMUP, REPEATS, RETAKES = 3, 3, 7, 3
SCHEMA = 1
AM = (2, 2, 2, 2)
COMPILE_UNITS = (("kb", (2, 2, 2, 2)), ("rolled", (2, 2, 2, 2)),
                 ("rolled", (3, 3, 3, 3)))
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


def _build(arm: str, am):
    return (build_kblock_rolled_kernel(am) if arm == "rolled"
            else build_kblock_kernel(am))


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


def _clock(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    path = _store_path()
    tc = _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    basis, D = _basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=0.0, build=build_kblock_kernel)
    assert [c.am for c in plan] == [AM]
    (cl,) = plan
    assert len(cl.bras) == len(cl.kets) == 2304 and len(cl.blocks) == 256
    parts = {arm: class_binaries(tc, replace(plan, classes=(cl,)), basis, D,
                                 backend, tmp_path / arm,
                                 rolled=arm == "rolled", with_k=False)
             for arm in ARMS}
    K0, K1 = (run_classes(parts[arm], basis, "kb", run_device_binary,
                          backend)[0][0] for arm in ARMS)
    rel = float(np.max(np.abs(K1 - K0)) / np.max(np.abs(K0)))
    assert np.max(np.abs(K0)) > 0.0 and rel <= ORACLE_TOL, (
        f"rolled vs kb {rel:.2e}")

    per: dict[str, list[float]] = {a: [] for a in ARMS}
    readings, discarded = [], 0
    for _ in range(ROUNDS):
        for arm in ARMS:
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

    arms = {arm: _stats(per[arm]) for arm in ARMS}
    ratio = arms["rolled"]["median_ns"] / arms["kb"]["median_ns"]
    verdict = ("faster" if ratio < 1 - BAND else
               "slower" if ratio > 1 + BAND else "neutral")
    row = {
        "arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
        "gpu": b.gpu(), "host": measured.host(),
        "date": datetime.date.today().isoformat(),
        "am": list(AM), "n_ao": basis.n_ao, "n_bra": 2304, "n_ket": 2304,
        "n_blocks": 256, "tau": 0.0, "rolled_vs_kb_rel": rel,
        "rounds": ROUNDS, "warmup": WARMUP, "repeats": REPEATS,
        "gate": readings, "gate_discarded_runs": discarded,
        "arms": arms, "rolled_over_kb": round(ratio, 4), "verdict": verdict,
    }
    store = _load(path)
    _save(path, store, "clock", row, ("arch", "toolchain"))
    print(f"\n{backend} {b.arch.name} (dd|dd): kb "
          f"{arms['kb']['median_ns'] / 1e6:.3f} ms, rolled "
          f"{arms['rolled']['median_ns'] / 1e6:.3f} ms; rolled/kb "
          f"{ratio:.3f} -> {verdict}; -> {path}")


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
