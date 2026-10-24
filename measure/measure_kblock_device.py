"""The block exchange kernel against `kernel_k`, measured per class (#481).

    devtools/measure.sh kblock [--backend cuda|hip]

OPT-IN, AND RUN ALONE: each test skips unless FJ_KBLOCK_STORE names the
store (measure.sh sets it to config.sh's MEASURE_KBLOCK_STORE, the tracked
data/measured/kblock.json). It writes one row per (arch, toolchain).

WHAT IS MEASURED. H2O/cc-pVDZ, the tracked `h2o_ccpvdz_k` fixture's
molecule and density, at tau = 0 -- every K class with every `l <= 2`, the
81 classes through `(dd|dd)` -- planned by `launch_kblock.kb_launches` and
run by test/kblock_driver.py's per-class binaries (`class_binaries`: one
driver, one upload per class), each in its two modes:
`kb` (the block kernel, one launch per class into `K_mn`) and `k`
(`kernel_k`, one launch per (class, b shell) into its per-record scratch,
#206's launch). Per mode, per class: the median, over the timed runs, of the
GPU events around that class's launches; and the whole build's, per timed
run the classes' events summed. `k`'s host scatter of its scratch into `K_mn` is NOT in
its time: the comparison favours `kernel_k` by that pass. Beside the clock,
each kernel's registers and spills from the compiler (`measured.compile_unit`,
the store's own flags and parsers), one unit per kernel.

THE PROTOCOL: `ROUNDS` rounds, each running `kb` then `k` over every class
(per class one process: WARMUP untimed builds, then REPEATS timed), each
such phase behind `gate.wait_quiet` with its after-reading taken; a
contended phase is retaken up to RETAKES times and then refuses the row.
Before timing, each mode's `K` is held to `walk_kb`'s (1e-13 of max|K|), so
the timed binaries are the checked ones.

NOTHING HERE IS A VERDICT FOR THE PAPER: [Fock]'s tables are A100/V100
wall-clock of a whole Fock build (limits.json), and this is two of this
repo's kernels on one fixture. It is the "measure the mapping alone first"
of #481, before any block screen.
"""

from __future__ import annotations

import datetime
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from kblock_driver import MODES, class_binaries, run_classes, timings
from test_k_ondevice import ORACLE_TOL
from test_kblock_ondevice import Cell, _case
from test_kernel_device import BACKEND, _toolchain

from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_kblock import build_kblock_kernel
from felizjueves.eri.k_exchange.launch_kblock import kb_launches, walk_kb
from felizjueves.performance import gate, measured
from felizjueves.performance.gate import GPU_BUSY_MAX
from felizjueves.reference.pq import pq

STORE = "FJ_KBLOCK_STORE"
CELL = Cell("h2o_ccpvdz_k", 0.0)
ROUNDS, WARMUP, REPEATS, RETAKES = 3, 3, 11, 3
SCHEMA = 1


def _stats(xs) -> dict[str, float]:
    q1, med, q3 = (float(x) for x in np.percentile(xs, [25, 50, 75]))
    return {"median_ns": round(med), "q1_ns": round(q1), "q3_ns": round(q3),
            "n": len(xs)}


def _resources(backend: str, kern, workdir: Path) -> dict[str, int]:
    """Registers and spills of one kernel as its own unit (measured's flags
    and parsers)."""
    b = measured.BACKENDS[backend]
    emit = BACKEND[backend][0]
    unit = measured.Unit(kern.name, emit(kern, b.arch), 0)
    raw, _, _ = measured.compile_unit(b, unit, compiler=b.compiler(),
                                      workdir=workdir)
    regs, store, load, _ = measured._numbers(b, raw)
    return {"registers": regs, "spill_store": store, "spill_load": load}


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 481, "about": __doc__.split("\n\n")[2],
            "rows": []}


def _measure(backend: str, request, run_device_binary, tmp_path: Path):
    store_path = os.environ.get(STORE)
    if not store_path:
        pytest.skip(f"the block-K clock is opt-in and must run alone: set "
                    f"{STORE} (devtools/measure.sh kblock)")
    tc = _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    _, basis, D = _case(CELL.case)
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=CELL.tau, build=build_kblock_kernel)
    assert {c.am for c in plan} == {
        (a, c, x, d) for a in range(3) for c in range(3) for x in range(3)
        for d in range(3)}
    parts = class_binaries(tc, plan, basis, D, backend, tmp_path)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    for mode in MODES:  # the timed binaries are the checked ones
        Ks, _, _ = run_classes(parts, basis, mode, run_device_binary, backend)
        rel = float(np.max(np.abs(Ks[0] - Kw)) / np.max(np.abs(Kw)))
        assert rel <= ORACLE_TOL, f"{mode}: K vs walk_kb {rel:.2e}"

    # per mode, per class: every timed run's events (each class's binary
    # holds that one class: its index there is 0)
    per: dict[str, list[list[float]]] = {m: [[] for _ in parts]
                                         for m in MODES}
    readings, discarded = [], 0
    for _ in range(ROUNDS):
        for mode in MODES:
            for take in range(RETAKES + 1):
                try:
                    busy, util = gate.wait_quiet(
                        backend, say=lambda s: print(s, flush=True))
                except gate.Contended as e:
                    discarded += 1
                    print(f"  DISCARDED {mode}: {e} (take {take + 1})")
                    continue
                _, outs, _ = run_classes(parts, basis, mode,
                                         run_device_binary, backend,
                                         warmup=WARMUP, reps=REPEATS)
                after = gate.after_reading(backend)
                if util is not None and after is not None \
                        and after <= GPU_BUSY_MAX:
                    break
                discarded += 1
                print(f"  DISCARDED {mode}: card {util} -> {after}%")
            else:
                pytest.fail(f"{mode}: contended through {RETAKES + 1} takes "
                            "-- not recorded")
            readings.append({"mode": mode, "cpu_busy_permille":
                             round(1000 * busy), "gpu_util_before": util,
                             "gpu_util_after": after})
            for c, out in enumerate(outs):
                per[mode][c].extend(timings(out)["0"])

    work = tmp_path / "res"
    work.mkdir()
    classes = []
    for c, (cl, part) in enumerate(zip(plan, parts, strict=True)):
        (x,) = part.drv.classes
        classes.append({
            "am": list(cl.am), "L": cl.L, "n_blocks": len(cl.blocks),
            "n_bra": x.n_bra, "n_ket": x.n_ket,
            "k_launches": len(x.segs),
            "kb": {**_stats(per["kb"][c]),
                   **_resources(backend, x.kb, work)},
            "k": {**_stats(per["k"][c]),
                  **_resources(backend, x.k, work)},
        })
    compiler = b.compiler()
    row = {
        "arch": b.arch.name, "toolchain": b.toolchain(compiler),
        "gpu": b.gpu(), "host": measured.host(),
        "date": datetime.date.today().isoformat(),
        "case": CELL.case, "tau": CELL.tau, "n_ao": basis.n_ao,
        "rounds": ROUNDS, "warmup": WARMUP, "repeats": REPEATS,
        "gate": readings, "gate_discarded_runs": discarded,
        # the build: per timed run, the classes' events summed (run i of
        # every class's process; the processes ran back to back)
        "total": {m: _stats(np.sum(per[m], axis=0)) for m in MODES},
        "classes": classes,
    }
    path = Path(store_path)
    store = _load(path)
    store["rows"] = [r for r in store["rows"]
                     if (r["arch"], r["toolchain"]) != (row["arch"],
                                                        row["toolchain"])]
    store["rows"].append(row)
    store["rows"].sort(key=lambda r: (r["arch"], r["toolchain"]))
    path.write_text(json.dumps(store, indent=1) + "\n")
    kb, k = row["total"]["kb"]["median_ns"], row["total"]["k"]["median_ns"]
    print(f"\n{backend} {b.arch.name}: whole build kb {kb / 1e6:.3f} ms, "
          f"k {k / 1e6:.3f} ms (k/kb {k / kb:.2f}); -> {path}")
    for x in classes:
        print("  ({}{}|{}{})".format(*x["am"]) + f" L={x['L']}: kb "
              f"{x['kb']['median_ns'] / 1e3:.1f} us r{x['kb']['registers']} "
              f"s{x['kb']['spill_store']}, k {x['k']['median_ns'] / 1e3:.1f} "
              f"us r{x['k']['registers']} s{x['k']['spill_store']}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_kblock_clock(request, run_device_binary, tmp_path) -> None:
    _measure("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_kblock_clock(request, run_device_binary, tmp_path) -> None:
    _measure("hip", request, run_device_binary, tmp_path)
