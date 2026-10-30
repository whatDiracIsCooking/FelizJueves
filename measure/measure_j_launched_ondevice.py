"""The whole molecule on the clock: #146's build wall-clock and #147's
sort-key A/B, on the RTX 3080 and the RX 9060 XT, into
data/measured/builds.json.

Moved from test/test_j_launched_ondevice.py, whose correctness checks (the
launch oracle, libcint, the fp64-atomic run-to-run bound) every timed driver
passes first: `_check` and `_check_counts` are imported, not copied, so the
TIMED binary is the CHECKED binary. Nothing here is a correctness claim.

RUN THROUGH devtools/measure.sh, alone, one session per card:

    devtools/measure.sh builds   --backend cuda   # BUILD_CELLS (#146)
    devtools/measure.sh sort-key --backend cuda   # SORT_KEY_BUILD_CELLS (#147)

Both set FJ_LAUNCH_WALLCLOCK_STORE to the tracked data/measured/builds.json
unless it is already set; without it each test skips. The section comment
below is the protocol and `measured_store.BUILD_TIMED_REGION` the clock's
boundary. `time_builds` takes any cells, so the sort-key A/B is two arms of
each cell in one session.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from plan_driver import (
    compile_command,
    read_back,
    run,
)
from plan_driver import reps as rep_lines
from test_j_launched_ondevice import (
    DISPATCH,
    DISPATCH_INPUT,
    _check,
    _check_counts,
)
from test_kernel_device import BACKEND, _toolchain

from felizjueves.performance import gate, measured_store
from felizjueves.performance.measured import BACKENDS as MEASURED_BACKENDS
from felizjueves.performance.measured import host as measured_host

# ---------------------------------------------------------------------------
# #146: the whole-build wall-clock, behind the card gate (opt-in, run alone)
# ---------------------------------------------------------------------------
#
# OPT-IN, AND RUN ALONE. It skips unless FJ_LAUNCH_WALLCLOCK_STORE names the
# build store to write (normally data/measured/builds.json, which must exist:
# a missing store is an error, as for every measured store). Inside the
# kernel tier it would share the box with xdist workers compiling other
# tests, and a contended timing is garbage, not a noisy measurement -- so
# run it by itself, one session per card:
#
#   devtools/measure.sh builds --backend cuda
#
# WHAT ONE ROW IS: `measured_store.BUILD_TIMED_REGION`, verbatim in the
# store's header -- the per-build memsets, every batch kernel of every class
# in plan order, one synchronize and the J_p read-back; NOT the uploads
# (once per process, timed apart as raw 'build:upload_ns'), NOT eq. 20 and
# the scatter (host Python), NOT compilation.
#
# THE PROTOCOL, in order, all inside one test so the phases cannot overlap:
#
# 1. COMPILE every cell's driver, with config.sh's NVCC_FLAGS / HIPCC_FLAGS
#    and #92's warning filter (`compile_checked`). No compile after this.
# 2. CHECK every compiled binary in its correctness mode (`_check`: counts,
#    the fp64-atomic run-to-run bound, the launch oracle, libcint) -- the
#    TIMED binary is the CHECKED binary.
# 3. TIME each cell: the gate
#    (`gate.wait_quiet`: card at most GPU_BUSY_MAX by its own SMI, CPU at
#    most BUSY_MAX), ONE process running BUILD_WARMUP untimed builds and
#    BUILD_REPEATS timed ones, then `gate.after_reading`. A phase whose
#    after-reading is non-idle or unknown is DISCARDED -- nothing from it
#    is kept -- and retaken, up to BUILD_RETAKES times; the row counts its
#    discards. A card the gate never lets through fails the test with
#    nothing recorded. The timed process's first and last J_p must agree
#    with the checked run's within the same fp64-atomic bound.
# 4. WRITE one row per cell (`measured_store.BuildRow`), one save after
#    every cell is timed.
#
# THE ARMS (#147). A cell is `BuildCell(fixture, tau, n, density_sort, arm)`;
# `time_builds` takes any tuple of them, so a same-session A/B is two cells
# that differ in `density_sort` only, timed in one call (one session, one
# store write; every row carries raw 'build:session'). #146's sweep is
# BUILD_CELLS, the plan's default sort, arm `measured_store.BASELINE_ARM`;
# #147's is SORT_KEY_BUILD_CELLS, both arms of every
# `measured_store.SORT_KEY_CELLS` cell under arm `SORT_AB_ARM`, the arm that
# runs first alternating from cell to cell -- run alone, one session per
# card, with the same store variable:
#
#   devtools/measure.sh sort-key --backend cuda
#
# The verdict (`measured_store.sort_key_ab`) was declared in that module's
# docstring before any of its rows existed, and is computed, never stored.

LAUNCH_WALLCLOCK_STORE = "FJ_LAUNCH_WALLCLOCK_STORE"
BUILD_WARMUP = 5
BUILD_REPEATS = 31
BUILD_RETAKES = 5


@dataclass(frozen=True)
class BuildCell:
    """One timed build: a fixture's molecule through `_plan`'s launch plan."""

    fixture: str
    tau: float
    n: int | str  # a fixed n_batches, or DISPATCH (dispatch.Resolver)
    density_sort: bool = True
    arm: str = measured_store.BASELINE_ARM

    @property
    def n_batches(self) -> int | str:
        """The row's n_batches: the count, or 'dispatch:<input>'."""
        return (f"{measured_store.DISPATCH_PREFIX}{DISPATCH_INPUT}"
                if self.n == DISPATCH else self.n)

    def key(self, arch: str, toolchain: str) -> str:
        return measured_store.build_key(self.fixture, self.tau, self.n_batches,
                                        self.density_sort, self.arm, arch,
                                        toolchain)


# H2O/STO-3G one batch (s and p shells, 9 classes); H2O/cc-pVDZ dispatched
# (d shells, 25 classes) unscreened, and at tau = 1e-3, where the screen
# skips 123896 of its 351649 record pairs (the launch oracle's count) --
# the cell #147's sort-key A/B reads against.
BUILD_CELLS = (
    BuildCell("h2o_sto3g_j", 0.0, 1),
    BuildCell("h2o_ccpvdz_j", 0.0, DISPATCH),
    BuildCell("h2o_ccpvdz_j", 1e-3, DISPATCH),
)


def _sort_key_cells() -> tuple[BuildCell, ...]:
    """Both arms of every SORT_KEY_CELLS cell, product first on even cells
    and global first on odd ones, so neither arm always runs first."""
    out = []
    for i, (fixture, tau, n) in enumerate(measured_store.SORT_KEY_CELLS):
        n = DISPATCH if n == f"{measured_store.DISPATCH_PREFIX}" \
            f"{DISPATCH_INPUT}" else n
        arms = (True, False) if i % 2 == 0 else (False, True)
        out.extend(BuildCell(fixture, tau, n, ds, measured_store.SORT_AB_ARM)
                   for ds in arms)
    return tuple(out)


SORT_KEY_BUILD_CELLS = _sort_key_cells()


def _quartiles(xs) -> tuple[float, float, float]:
    q1, med, q3 = (float(x) for x in np.percentile(xs, [25, 50, 75]))
    return q1, med, q3


def time_builds(backend: str, cells, store_path: Path, request,
                run_device_binary, tmp_path: Path) -> dict[str, object]:
    """The protocol above over `cells`, on `backend`'s card; writes the rows
    into `store_path` and returns them by key."""
    tc = _toolchain(backend, request)
    arch = BACKEND[backend][1]
    mb = MEASURED_BACKENDS[backend]
    toolchain, gpu = mb.toolchain(mb.compiler()), mb.gpu()
    store = measured_store.load_builds(store_path)
    keys = [c.key(arch.name, toolchain) for c in cells]
    assert len(set(keys)) == len(keys), f"two cells, one key: {keys}"
    session = int(time.time())  # every row of this call: one session

    # 1-2. compile, then check, every cell -- before any timing
    checked = []
    for i, cell in enumerate(cells):
        work = tmp_path / f"cell{i}"
        work.mkdir()
        ck = _check(backend, cell.fixture, cell.tau, cell.n, cell.density_sort,
                    tc, run_device_binary, work)
        checked.append(ck)
        print(f"checked {keys[i]}", flush=True)

    # 3. time, one cell at a time
    got: dict[str, object] = {}
    table = []
    for cell, key, ck in zip(cells, keys, checked, strict=True):
        discarded = 0
        for take in range(1 + BUILD_RETAKES):
            try:
                cpu, before = gate.wait_quiet(
                    backend, say=lambda line: print(line, flush=True))
            except gate.Contended as e:
                pytest.fail(f"{key}: {e} -- nothing recorded")
            load1 = os.getloadavg()[0]
            ran = run(ck.drv, ck.binary, ck.buffers,
                      tmp_path / f"timed{len(got)}_{take}",
                      runs=BUILD_REPEATS, warmup=BUILD_WARMUP)
            after = gate.after_reading(backend)
            if gate.is_idle(after):
                break
            discarded += 1
            print(f"  DISCARDED {key}: card {before}% -> {after}% (take "
                  f"{take + 1})", flush=True)
        else:
            pytest.fail(f"{key}: the card was taken during {BUILD_RETAKES + 1} "
                        "phases -- nothing recorded")

        _check_counts(ck.drv, ck.plan)
        rows = rep_lines(ran)
        assert len(rows) == BUILD_REPEATS, rows
        wall, dev, back, events = (list(x) for x in zip(*rows, strict=True))
        timed = read_back(ck.drv, ran)
        assert len(timed) == BUILD_REPEATS
        first, last = timed[0], timed[-1]
        bitwise = True
        for lp, bound in ck.bound.items():
            for one in (first, last):
                diff = np.abs(one[lp] - ck.jp[lp])
                assert np.all(diff <= bound), (
                    f"{key}: timed J_p[{lp}] off the checked run by "
                    f"{float(np.max(diff)):.3e}, over the atomic bound")
            bitwise &= bool(np.array_equal(first[lp], last[lp]))

        q1, med, q3 = _quartiles(wall)
        dq1, dmed, dq3 = _quartiles(dev)
        raw = {
            "build:warmup": BUILD_WARMUP,
            "build:min_ns": int(min(wall)),
            "build:q1_ns": round(q1),
            "build:q3_ns": round(q3),
            "build:max_ns": int(max(wall)),
            "build:device_median_ns": round(dmed),
            "build:device_q1_ns": round(dq1),
            "build:device_q3_ns": round(dq3),
            "build:readback_median_ns": round(float(np.median(back))),
            "build:events_median_ns": round(float(np.median(events))),
            "build:upload_ns": ran.upload_ns,
            "build:classes": len(ck.plan),
            "build:kernels": len(ck.drv.launches),
            "build:bra_classes": len(ck.drv.bra),
            "build:ket_classes": len(ck.drv.ket),
            "build:n_pairs": ck.n_pairs,
            "build:n_skipped": ck.n_skipped,
            "build:refused": len(ck.plan.refused),
            "build:J_bitwise_repeatable": int(bitwise),
            "build:loadavg1_milli": round(1000 * load1),
            "build:gpu_util_before": before,
            "build:gpu_util_after": after,
            "build:gpu_busy_max": gate.GPU_BUSY_MAX,
            "build:cpu_busy_permille_before": round(1000 * cpu),
            "build:gate_discarded_phases": discarded,
            measured_store.SESSION_FIELD: session,
        }
        text = ck.drv.source + "".join(u.read_text() for u in ck.drv.units)
        cmd = compile_command(tc, ck.drv, ck.binary)
        cmd = [Path(cmd[0]).name, *cmd[1:-3], "<driver>", "-o", "<binary>"]
        row = measured_store.BuildRow(
            fixture=cell.fixture, tau=cell.tau, n_batches=cell.n_batches,
            density_sort=cell.density_sort, arm=cell.arm,
            runtime_ns=round(med), spread_ns=round(q3 - q1),
            repeats=BUILD_REPEATS, toolchain=toolchain, arch=arch.name,
            host=measured_host(), gpu=gpu,
            date=datetime.date.today().isoformat(), command=" ".join(cmd),
            source_sha256=hashlib.sha256(text.encode()).hexdigest(), raw=raw)
        assert row.key() == key, (row.key(), key)
        got[key] = row
        table.append((key, med, q3 - q1, dmed, raw))
        print(f"{key}: median {med / 1e3:.1f} us, IQR {(q3 - q1) / 1e3:.1f} "
              f"us, range [{min(wall) / 1e3:.1f}, {max(wall) / 1e3:.1f}] us "
              f"(device {dmed / 1e3:.1f} us, read-back "
              f"{raw['build:readback_median_ns'] / 1e3:.1f} us, upload once "
              f"{ran.upload_ns / 1e3:.1f} us); {len(ck.drv.launches)} "
              f"kernels/build; {BUILD_REPEATS} timed (+{BUILD_WARMUP} "
              f"warm-up); card {before}% -> {after}%, CPU {cpu:.1%}, "
              f"discarded {discarded}; J_p bit-repeatable {bitwise}",
              flush=True)

    # 4. one write, after every cell is timed
    for row in got.values():
        store = measured_store.merge_build(store, row)
    measured_store.save_builds(store, store_path)
    print(f"\n{backend} {arch.name} ({gpu}), {toolchain}: {len(got)} build "
          f"rows -> {store_path}")
    return got


def _build_wallclock(backend: str, request, run_device_binary,
                     tmp_path: Path) -> None:
    store = os.environ.get(LAUNCH_WALLCLOCK_STORE)
    if not store:
        pytest.skip(f"the build wall-clock is opt-in and must run alone, "
                    f"uncontended: set {LAUNCH_WALLCLOCK_STORE} (see the "
                    "section comment)")
    time_builds(backend, BUILD_CELLS, Path(store), request, run_device_binary,
                tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_build_wallclock(request, run_device_binary, tmp_path) -> None:
    """The whole-molecule J_p build's wall-clock on the RTX 3080 (#146)."""
    _build_wallclock("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_build_wallclock(request, run_device_binary, tmp_path) -> None:
    """The whole-molecule J_p build's wall-clock on the RX 9060 XT (#146)."""
    _build_wallclock("hip", request, run_device_binary, tmp_path)


def _sort_key_ab(backend: str, request, run_device_binary,
                 tmp_path: Path) -> None:
    store = os.environ.get(LAUNCH_WALLCLOCK_STORE)
    if not store:
        pytest.skip(f"#147's sort-key A/B is opt-in and must run alone, "
                    f"uncontended: set {LAUNCH_WALLCLOCK_STORE} (see the "
                    "section comment)")
    got = time_builds(backend, SORT_KEY_BUILD_CELLS, Path(store), request,
                      run_device_binary, tmp_path)
    arch = BACKEND[backend][1].name
    print(f"\n#147 on {arch} (verdict declared before these rows):")
    for cell in measured_store.SORT_KEY_CELLS:
        ab = measured_store.sort_key_ab(got, arch, cell)
        pm, piqr, gm, giqr = ab.clock
        print(f"  {cell[0]} tau={cell[1]:g} b={cell[2]}: skipped "
              f"{ab.skipped[0]} (product) vs {ab.skipped[1]} (global) of "
              f"{ab.n_pairs}; median {pm / 1e3:.1f} us (IQR {piqr / 1e3:.1f}) "
              f"vs {gm / 1e3:.1f} us (IQR {giqr / 1e3:.1f}): global "
              f"{ab.clock_sign}, skips {ab.skip_sign} -> {ab.verdict}")
    print(f"  {arch}: {measured_store.sort_key_arch_verdict(got, arch)}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_sort_key_ab(request, run_device_binary, tmp_path) -> None:
    """#147's two sort keys, one session, on the RTX 3080."""
    _sort_key_ab("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_sort_key_ab(request, run_device_binary, tmp_path) -> None:
    """#147's two sort keys, one session, on the RX 9060 XT."""
    _sort_key_ab("hip", request, run_device_binary, tmp_path)
