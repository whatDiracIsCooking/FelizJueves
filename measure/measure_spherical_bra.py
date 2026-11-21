"""The fully spherical three-center kernel against its two-pass twin (#747).

    devtools/measure.sh spherical-bra [--backend cuda|hip]     timed, alone

OPT-IN: skips unless FJ_SPHERICAL_BRA_STORE names the store (measure.sh
sets it to config.sh's MEASURE_SPHERICAL_BRA_STORE, the tracked
data/measured/spherical_bra.json). FJ_SPHERICAL_BRA_ONLY (comma-separated
`ssperi3_` names) narrows the ladder.

THE QUESTION (#745's follow-up): per class, is `ssperi3_` (the bra
transformed in the kernel, per column and primitive triple, #745) faster
than what FreeBreakfast runs today, `speri3_` (#743) then a bra transform
of its packed output? The second pass is `sbra3_` (#747, `kernel_eri3.
build_bra_transform_kernel`): one thread per (pair, aux column), each
element read once and written once, the traffic floor of any transform of
that output, a GEMM's included -- so a row's two-pass arm is a LOWER bound
on the consumer's, and a "slower" verdict for `ssperi3_` is conservative.

PER CLASS of the ladder FreeBreakfast emits, `la <= lb <= 3`, `lP <= 6`
(70), on #606's clock bases (`measure_df_lowstack._clock_bases`: 16 shells
of the bra `l` when `la = lb`, else 12 of each, two primitives; 64 aux
shells of `lP`, one primitive; the class's cells only):

- COMPILE: `ssperi3_`, `speri3_` and `sbra3_`, each alone with the measured
  store's flags at the run's arch (nvcc `-Xptxas -v`; hipcc's resource
  remarks): source bytes, wall, registers, spill, stack.
- WITNESS: one run of each arm; the two outputs (spherical bra, spherical
  aux, NaN where nothing stores) agree within `WITNESS_REL` of the max --
  a sanity check that both arms compute the same block, not the oracle
  (test_kernel_eri3_fully_spherical.py, test_kernel_bra_transform.py).
- CLOCK: `ROUNDS` rounds, each the fused arm then the two-pass arm
  (`WARMUP` untimed runs, `REPEATS` timed), each phase behind
  `gate.wait_quiet` with its after-reading, retaken up to `RETAKES` times
  and then refused. The two-pass arm's per-launch events are kept too.

THE RULE is the store's `declared` (DECLARED), written before any row.
NOT A TEST: nothing here gates; `measure.sh check` reads no row of it.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from eri3_driver import THREADS, packed_steps
from harness_driver import Step, build_runner, run_sequence, units_runner
from measure_df_lowstack import _clock_bases
from plan_driver import _unit
from test_kernel_device import BACKEND, _toolchain

from felizjueves.eri.density_fit import kernel_eri3 as ke
from felizjueves.eri.density_fit.eri3 import (
    bra_pairs,
    eri3_classes,
    packed_cells,
    packed_columns,
    spherical_offsets,
)
from felizjueves.performance import gate, measured
from felizjueves.performance.device import Vendor, from_env
from felizjueves.performance.gate import GPU_BUSY_MAX

STORE = "FJ_SPHERICAL_BRA_STORE"
SCHEMA = 1
ROUNDS, WARMUP, REPEATS, RETAKES = 3, 2, 7, 3
BAND = 0.05
WITNESS_REL = 1e-11
COMPILE_BUDGET_S = 300.0
LADDER = [(a, b, p) for a in range(4) for b in range(a, 4) for p in range(7)]
TWO_PASS = "speri3+sbra3"
FUSED = "ssperi3"

DECLARED = {
    "issue": 747,
    "question": "per class, is ssperi3_ (#745) faster than speri3_ (#743) "
                "followed by a bra transform of its packed output (sbra3_)",
    "rule": "per class and card: ratio = median(ssperi3_ runs) / "
            "median(speri3_ + sbra3_ runs, the whole two-launch sequence); "
            "'faster' below 1 - BAND, 'slower' above 1 + BAND, else "
            "'neutral'; BAND = 0.05. sbra3_ reads and writes each element "
            "once, so the two-pass arm is a lower bound on any transform's",
    "declared": "2026-11-21, before any row",
}


def _store_path() -> Path:
    p = os.environ.get(STORE)
    if not p:
        pytest.skip(f"the spherical-bra clock is opt-in: set {STORE} "
                    "(devtools/measure.sh spherical-bra)")
    return Path(p)


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": SCHEMA, "issue": 747,
            "about": __doc__.split("\n\n")[2], "compile": [], "clock": []}


def _declare(path: Path, store: dict) -> None:
    declared = json.loads(json.dumps(DECLARED))
    if "declared" not in store:
        store["declared"] = declared
        path.write_text(json.dumps(store, indent=1) + "\n")
    assert store["declared"] == declared, (
        "the store's declared rule is not this file's")


def _save(path: Path, section: str, row: dict, key: tuple[str, ...]) -> None:
    store = _load(path)
    rows = [r for r in store[section]
            if tuple(r[k] for k in key) != tuple(row[k] for k in key)]
    rows.append(row)
    rows.sort(key=lambda r: tuple(str(r[k]) for k in key))
    store[section] = rows
    path.write_text(json.dumps(store, indent=1) + "\n")


def _backend(backend: str) -> measured.Backend:
    """The measured store's backend at the run's arch: `CUDA_ARCH` /
    `HIP_ARCH` (`device.from_env`), so a Hopper run compiles `sm_90`."""
    vendor = Vendor.NVIDIA if backend == "cuda" else Vendor.AMD
    return dataclasses.replace(measured.BACKENDS[backend],
                               arch=from_env(vendor))


def _compile(b: measured.Backend, backend: str, kern, workdir: Path) -> dict:
    """One unit alone, measured flags, at `b.arch`."""
    text = BACKEND[backend][0](kern, b.arch)
    src, obj = (workdir / f"{kern.name}.{x}" for x in (b.ext, "o"))
    src.write_text(text)
    t0 = time.perf_counter()
    done = subprocess.run(b.command(b.compiler(), src, obj),
                          capture_output=True, text=True,
                          timeout=COMPILE_BUDGET_S)
    wall = time.perf_counter() - t0
    out = done.stdout + done.stderr
    assert done.returncode == 0, out[-4000:]
    src.unlink()
    obj.unlink(missing_ok=True)
    raw = (measured.parse_ptxas(out, kern.name) if backend == "cuda"
           else measured.parse_amd_remarks(out, kern.name))
    regs, store, load, _ = measured._numbers(b, raw)
    return {"unit": kern.name, "source_bytes": len(text),
            "compile_s": round(wall, 2), "registers": regs,
            "spill_store": store, "spill_load": load,
            "stack_bytes": raw.get("stack_frame_bytes",
                                   raw.get("ScratchSize"))}


def _stats(xs) -> dict[str, float]:
    q1, med, q3 = (float(x) for x in np.percentile(xs, [25, 50, 75]))
    return {"median_ns": round(med), "q1_ns": round(q1), "q3_ns": round(q3),
            "n": len(xs)}


def _verdict(ratio: float) -> str:
    return ("faster" if ratio < 1 - BAND else
            "slower" if ratio > 1 + BAND else "neutral")


def _arms(cls, backend: str, tc, work: Path):
    """The two arms' runners, sequences and output sizes for one class."""
    la, lb, lP = cls.la, cls.lb, cls.lP
    _, aux = _clock_bases("eri3", (la, lb, lP))
    offs, n_aux = spherical_offsets(aux)
    cart, n_cart = packed_columns([cls])
    sph, n_sph = packed_columns([cls], spherical_bra=True)
    k_ss = ke.build_eri3_kernel((la, lb, lP), packed=True,
                                spherical_aux=True, spherical_bra=True)
    k_s = ke.build_eri3_kernel((la, lb, lP), packed=True, spherical_aux=True)
    k_t = ke.build_bra_transform_kernel(la, lb)
    work.mkdir(parents=True, exist_ok=True)

    def runner(name, kerns):
        units = [(_unit(k, backend, work, None), [k.name]) for k in kerns]
        return build_runner(tc, units_runner(backend, units, work, name),
                            work / f"{name}_bin", timeout=COMPILE_BUDGET_S)

    fused_bin = runner("fused", [k_ss])
    fused = packed_steps([cls], [packed_cells(cls, sph, n_aux, offs)],
                         n_sph, n_aux, [k_ss])
    two_bin = runner("two_pass", [k_s, k_t])
    buffers, (step,) = packed_steps(
        [cls], [packed_cells(cls, cart, n_aux, offs)], n_cart, n_aux, [k_s])
    buffers["cart"] = buffers.pop(ke.OUT)
    buffers[ke.OUT] = np.full(n_sph * n_aux, np.nan)
    pairs = bra_pairs(cls, cart, sph)
    n_pair = len(pairs["src_col"])
    for k in ke.BRA_PAIR_ARRAYS:
        buffers[f"t.{k}"] = np.asarray(pairs[k], dtype=np.int32)
    grid, block = ke.launch_shape(n_pair * n_aux, THREADS)
    t_args = {"n_pair": n_pair, "n_aux": n_aux, "n_src_col": n_cart,
              "n_dst_col": n_sph, **{k: f"t.{k}" for k in ke.BRA_PAIR_ARRAYS},
              ke.BRA_SRC: "cart", ke.OUT: ke.OUT}
    two = (buffers, [Step(step.kernel, step.grid, step.block,
                          {**step.args, ke.OUT: "cart"}),
                     Step(k_t, grid, block, t_args)])
    size = {"n_aux": n_aux, "n_pair": n_pair, "n_cell": cls.n_cell,
            "n_bra_rec": cls.n_bra_rec, "cartesian_doubles": n_cart * n_aux,
            "spherical_doubles": n_sph * n_aux}
    return {FUSED: (fused_bin, fused), TWO_PASS: (two_bin, two)}, size, (
        k_ss, k_s, k_t)


def _clock(backend: str, request, tmp_path: Path) -> None:
    path = _store_path()
    tc = _toolchain(backend, request)
    b = _backend(backend)
    store = _load(path)
    _declare(path, store)
    head = {"arch": b.arch.name, "toolchain": b.toolchain(b.compiler()),
            "gpu": b.gpu(), "host": measured.host(),
            "date": datetime.date.today().isoformat()}
    only = os.environ.get("FJ_SPHERICAL_BRA_ONLY")
    for am in LADDER:
        name = ke.eri3_name(am, spherical_bra=True)
        if only and name not in only.split(","):
            continue
        basis, aux = _clock_bases("eri3", am)
        (cls,) = [c for c in eri3_classes(basis, aux)
                  if (c.la, c.lb, c.lP) == am]
        work = tmp_path / name
        arms, size, kerns = _arms(cls, backend, tc, work)
        for k in kerns:
            row = {**head, "am": list(am), **_compile(b, backend, k, work)}
            _save(path, "compile", row, ("arch", "toolchain", "unit"))

        def run(arm, warmup, reps, arms=arms, work=work):
            binary, (buffers, steps) = arms[arm]
            ran = run_sequence(binary, work / arm, buffers, steps,
                               outs=[ke.OUT], runs=reps, warmup=warmup)
            return ran
        outs = {a: np.load(run(a, 0, 1).outs[ke.OUT][0]) for a in arms}
        ref = outs[FUSED]
        assert np.array_equal(np.isnan(ref), np.isnan(outs[TWO_PASS]))
        live = ~np.isnan(ref)
        s = float(np.max(np.abs(ref[live])))
        rel = float(np.max(np.abs(outs[TWO_PASS][live] - ref[live]))) / s
        assert rel <= WITNESS_REL, f"{name}: arms differ by {rel:.2e}"
        per = {a: [] for a in arms}
        launches: list[list[float]] = []
        readings, discarded = [], 0
        for _ in range(ROUNDS):
            for arm in arms:
                for take in range(RETAKES + 1):
                    try:
                        busy, util = gate.wait_quiet(
                            backend, say=lambda x: print(x, flush=True))
                    except gate.Contended as e:
                        discarded += 1
                        print(f"  DISCARDED {name}: {e} (take {take + 1})")
                        continue
                    ran = run(arm, WARMUP, REPEATS)
                    after = gate.after_reading(backend)
                    if util is not None and after is not None \
                            and after <= GPU_BUSY_MAX:
                        break
                    discarded += 1
                    print(f"  DISCARDED {name}: card {util} -> {after}%")
                else:
                    pytest.fail(f"{name}: contended through {RETAKES + 1} "
                                "takes -- not recorded")
                readings.append({"arm": arm,
                                 "cpu_busy_permille": round(1000 * busy),
                                 "gpu_util_before": util,
                                 "gpu_util_after": after})
                per[arm].extend(ran.events_ns)
                if arm == TWO_PASS:
                    launches.extend(ran.launch_ns)
        stats = {a: _stats(v) for a, v in per.items()}
        ratio = stats[FUSED]["median_ns"] / stats[TWO_PASS]["median_ns"]
        row = {**head, "am": list(am), "unit": name, **size,
               "rounds": ROUNDS, "warmup": WARMUP, "repeats": REPEATS,
               "gate": readings, "gate_discarded_runs": discarded,
               "arms": stats,
               "two_pass_launches": {
                   "speri3": _stats([x[0] for x in launches]),
                   "sbra3": _stats([x[1] for x in launches])},
               "witness_rel": rel, "fused_over_two_pass": round(ratio, 4),
               "verdict": _verdict(ratio)}
        _save(path, "clock", row, ("arch", "toolchain", "unit"))
        print(f"{backend} {name}: fused {stats[FUSED]['median_ns'] / 1e6:.3f}"
              f" ms, two-pass {stats[TWO_PASS]['median_ns'] / 1e6:.3f} ms "
              f"-> {row['verdict']} ({row['fused_over_two_pass']})",
              flush=True)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_spherical_bra_clock(request, tmp_path) -> None:
    _clock("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_spherical_bra_clock(request, tmp_path) -> None:
    _clock("hip", request, tmp_path)
