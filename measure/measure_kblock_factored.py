"""The block K kernel with E factored against E composed, (dd|dd) (#569).

    devtools/measure.sh kblock-factored [--backend cuda|hip]

OPT-IN, AND RUN ALONE: each test skips unless FJ_KBLOCK_FACTORED_STORE names
the store (measure.sh sets it to config.sh's MEASURE_KBLOCK_FACTORED_STORE,
the tracked data/measured/kblock_factored.json). It writes one row per
(arch, toolchain).

WHAT IS MEASURED. #569 moved the K kernels' `E` from two composed blocks per
record (`Eac[n_ac][n_p]`, `Ebd[n_bd][n_q]`) to three axis tables per side,
each `E_t` formed where it is read (`seams.bind_e_ac` and the rest). The
issue makes the `(dd|dd)` wall-clock against the pre-#569 kernel its accept /
reject gate, because a factored form trades storage for two multiplies per
read. So: `kernel_kblock` built both ways on ONE plan -- `composed` is
`build_kblock_kernel(composed_e=True)`, the pre-#569 text byte for byte
(test/test_k_seams.py pins it), and `factored` the default -- each compiled
into its own per-class driver (test/kblock_driver.py, MODE 0, one launch per
class into `K_mn`), on the issue's input: a synthetic d-only basis of 16
centres, 3 primitives a shell, so 2,304 bra and 2,304 ket records and 256
blocks, at `tau = 0` (`BASIS`). Beside the clock, each kernel's registers
and spills as its own unit (the measured store's flags and parsers).

THE PROTOCOL, `measure_kblock_device.py`'s: `ROUNDS` rounds, each running
`composed` then `factored` (WARMUP untimed builds, then REPEATS timed), each
phase behind `gate.wait_quiet` with its after-reading taken; a contended
phase is retaken up to RETAKES times and then refuses the row. Before
timing, the two arms' `K` are held to each other (1e-13 of max|K|), so the
timed binaries are the checked ones; `walk_kb` over this input's 5.3M
record pairs is hours of Python, and both forms are held to it on small
inputs by test/test_kblock_ondevice.py.

THE VERDICT RULE is the store's `declared`, written before any row.
"""

from __future__ import annotations

import datetime
import json
import os
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
from felizjueves.eri.k_exchange.launch_kblock import kb_launches
from felizjueves.performance import gate, measured
from felizjueves.performance.gate import GPU_BUSY_MAX
from felizjueves.reference.pq import pq

STORE = "FJ_KBLOCK_FACTORED_STORE"
ARMS = ("composed", "factored")
ROUNDS, WARMUP, REPEATS, RETAKES = 3, 3, 7, 3
SCHEMA = 1
AM = (2, 2, 2, 2)
TOL = 1.05  # the declared rule's threshold, factored / composed

DECLARED = {
    "issue": 569,
    "hypothesis": "holding E factored (three axis tables per side, each E_t "
                  "formed at its read) removes the per-thread E blocks that "
                  "keep (f, f)-pair K classes off the RTX 3080, at a cost of "
                  "two multiplies per stage-2 read that may slow (dd|dd)",
    "rule": "per arch: the median over the timed runs of the (dd|dd) class's "
            "events, factored / composed. 'accepted' when <= 1.05, "
            "'regressed' when > 1.05; a regression is reported as the trade "
            "the f classes pay for, never hidden",
    "expected": "neutral within the 5% band: the compilers already hoist "
                "the composed E^ac out of the ket loop (stack frames within "
                "4% of each other at (dd|dd) and (ff|ss) before any row)",
    "declared": "2026-10-26, before any row",
}


def _basis() -> tuple[Basis, np.ndarray]:
    """16 d shells on a 4 x 4 grid (2 bohr), 3 primitives each, and a
    symmetric, decaying density: 2,304 records a side at (dd|dd)."""
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


def _resources(backend: str, kern, workdir: Path) -> dict[str, int]:
    b = measured.BACKENDS[backend]
    emit = BACKEND[backend][0]
    unit = measured.Unit(kern.name, emit(kern, b.arch), 0)
    raw, compile_s, _ = measured.compile_unit(b, unit, compiler=b.compiler(),
                                              workdir=workdir)
    regs, store, load, _ = measured._numbers(b, raw)
    return {"registers": regs, "spill_store": store, "spill_load": load,
            "source_bytes": len(unit.text), "compile_s": round(compile_s, 1)}


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 569,
            "about": __doc__.split("\n\n")[2], "declared": DECLARED,
            "rows": []}


def _measure(backend: str, request, run_device_binary, tmp_path: Path):
    store_path = os.environ.get(STORE)
    if not store_path:
        pytest.skip(f"the factored-E clock is opt-in and must run alone: set "
                    f"{STORE} (devtools/measure.sh kblock-factored)")
    tc = _toolchain(backend, request)
    b = measured.BACKENDS[backend]
    basis, D = _basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=0.0, build=build_kblock_kernel)
    assert [c.am for c in plan] == [AM]
    (cl,) = plan
    assert len(cl.bras) == len(cl.kets) == 2304 and len(cl.blocks) == 256
    parts = {}
    for arm in ARMS:  # one plan, the class's kernel built per arm
        kern = build_kblock_kernel(AM, composed_e=arm == "composed")
        armed = replace(plan, classes=(replace(cl, kernel=kern),))
        parts[arm] = class_binaries(tc, armed, basis, D, backend,
                                    tmp_path / arm, with_k=False)
    # the timed binaries are the checked ones: the two arms against each
    # other (the same values per term; `walk_kb` over 5.3M record pairs in
    # Python is hours, and the arms are held to it on small inputs by
    # test/test_kblock_ondevice.py)
    Kc, Kf = (run_classes(parts[arm], basis, "kb", run_device_binary,
                          backend)[0][0] for arm in ARMS)
    rel = float(np.max(np.abs(Kf - Kc)) / np.max(np.abs(Kc)))
    assert np.max(np.abs(Kc)) > 0.0 and rel <= ORACLE_TOL, (
        f"factored vs composed {rel:.2e}")

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

    work = tmp_path / "res"
    work.mkdir()
    arms = {}
    for arm in ARMS:
        ((x,),) = [p.drv.classes for p in parts[arm]]
        arms[arm] = {**_stats(per[arm]), **_resources(backend, x.kb, work)}
    ratio = arms["factored"]["median_ns"] / arms["composed"]["median_ns"]
    compiler = b.compiler()
    row = {
        "arch": b.arch.name, "toolchain": b.toolchain(compiler),
        "gpu": b.gpu(), "host": measured.host(),
        "date": datetime.date.today().isoformat(),
        "am": list(AM), "n_ao": basis.n_ao, "n_bra": 2304, "n_ket": 2304,
        "n_blocks": 256, "tau": 0.0,
        "rounds": ROUNDS, "warmup": WARMUP, "repeats": REPEATS,
        "gate": readings, "gate_discarded_runs": discarded,
        "arms": arms, "factored_over_composed": round(ratio, 4),
        "verdict": "accepted" if ratio <= TOL else "regressed",
    }
    path = Path(store_path)
    store = _load(path)
    store["rows"] = [r for r in store["rows"]
                     if (r["arch"], r["toolchain"]) != (row["arch"],
                                                        row["toolchain"])]
    store["rows"].append(row)
    store["rows"].sort(key=lambda r: (r["arch"], r["toolchain"]))
    path.write_text(json.dumps(store, indent=1) + "\n")
    c, f = arms["composed"], arms["factored"]
    print(f"\n{backend} {b.arch.name} (dd|dd): composed "
          f"{c['median_ns'] / 1e6:.3f} ms r{c['registers']} "
          f"s{c['spill_store']}, factored {f['median_ns'] / 1e6:.3f} ms "
          f"r{f['registers']} s{f['spill_store']}; factored/composed "
          f"{ratio:.3f} -> {row['verdict']}; -> {path}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_kblock_factored_clock(request, run_device_binary,
                                    tmp_path) -> None:
    _measure("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_kblock_factored_clock(request, run_device_binary,
                                   tmp_path) -> None:
    _measure("hip", request, run_device_binary, tmp_path)
