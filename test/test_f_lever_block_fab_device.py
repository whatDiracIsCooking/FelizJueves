"""The block-level `f_ab` reduction (#351) through the compilers and on the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): every F class with L <= 4
   (milestone 13's cap), the literal kernel and the lever's kernel each its
   own translation unit, compiled with the tier's flags plus the measurement
   flag (`measured.Backend.command`: `-Xptxas -v` on nvcc, the AMD
   `kernel-resource-usage` remark on hipcc). A warning in generated text
   fails (#92). The figures -- registers, spill and static shared bytes
   (nvcc's `smem`; hipcc's LDS) -- literal against lever are PRINTED (`-s`)
   as observations, never store rows, beside the declared footprint and the
   bra flush's atomics per block (`block_fab.bra_flush_atomics` at 128
   threads). The static shared bytes ARE asserted: the lever's exceed the
   literal's, by at most `block_fab.reduce_bytes(cls)`, the declared
   footprint (a compiler may not allocate less than it reads after a
   barrier, but nothing here assumes it rounds the same way on both).
2. ON THE CARD (`gpu` + `cuda` / `hip`): every F class with L <= 4, its
   launches in H2O/cc-pVDZ's plan at `tau = 0` (test_fock_ondevice.py's
   part 1: `_class_recs`, the same recorded packed records, `diag`, `Dsh`
   and `D`), each relaunched with the lever's kernel at 32 threads (one
   warp: the shared stage reduces one row) and 128 (four warps), held by
   test_fock_ondevice.py's own `_check_class` -- `TOL = 1e-13` of `scale_G`
   against the restricted sweep, `2 TOL` against the literal launches
   interpreted, and two runs of one binary within `2 TOL` -- unchanged. A
   block of 128 is a multiple of both cards' wavefront (32) and under
   `block_fab.launch_shape`'s contract, asserted before any launch.

Run part 2 as (each backend; nothing above L = 4 and nothing `slow` is
selected; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k f_lever_block_fab
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path

import pytest
import test_fock_ondevice as tfo
from fock_driver import FRec, fock_stage, run_on_card

from felizjueves import kernel_f as KF
from felizjueves import measured
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import block_fab
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.lever_ladder import unit_name

NAME = block_fab.NAME
L4 = tuple(c for c in CLASSES if sum(c) <= 4)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in L4]
DEVICE_THREADS = (32, 128)
WAVEFRONT = 32  # both cards' (device.SM_86, device.GFX1200)


def lever(cls) -> Kernel:
    k = KF.build_f_kernel(KF.f_class_spec(*cls, (NAME,)))
    return dataclasses.replace(k, name=unit_name(k.name, (NAME,)))


# ---------------------------------------------------------------------------
# 1. Compile-only: registers, spill and shared, literal vs lever
# ---------------------------------------------------------------------------

EMIT = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}
WARNING = re.compile(r"^(?P<file>[^\s:]+):\d+:\d+: warning:", re.MULTILINE)


def _compile(backend: str, compiler: str, kern: Kernel, workdir: Path
             ) -> tuple[int, int, int]:
    """(registers or VGPRs, spill bytes or spilled registers, static shared
    bytes) of one unit."""
    be = measured.BACKENDS[backend]
    emit, arch = EMIT[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / f"{kern.name}.{be.ext}"
    obj = workdir / f"{kern.name}.o"
    src.write_text(emit(kern, arch))
    done = subprocess.run(be.command(compiler, src, obj), capture_output=True,
                          text=True, timeout=1800)
    report = done.stdout + done.stderr
    assert done.returncode == 0, f"{kern.name}:\n{report[-4000:]}"
    warned = [m[0] for m in WARNING.finditer(report)
              if Path(m["file"]).name == src.name]
    assert not warned, f"{kern.name}: warning in generated text (#92): {warned}"
    if backend == "cuda":
        raw = measured.parse_ptxas(report, kern.name)
        return (raw["registers"], raw["spill_store_bytes"],
                raw.get("smem_bytes", 0))
    raw = measured.parse_amd_remarks(report, kern.name)
    return (raw["VGPRs"], raw["VGPRs Spill"] + raw.get("SGPRs Spill", 0),
            raw.get("LDS Size", 0))


def _figures(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    lit = _compile(backend, compiler, KF.build_f_kernel(cls),
                   tmp_path / "literal")
    lev = _compile(backend, compiler, lever(cls), tmp_path / "lever")
    declared = block_fab.reduce_bytes(cls)
    assert lit[2] < lev[2] <= lit[2] + declared, (lit, lev, declared)
    unit = ("registers/spill-store bytes/smem" if backend == "cuda"
            else "VGPRs/spilled registers/LDS")
    print(f"\n{backend} {cls} ({unit}): literal {lit[0]}/{lit[1]}/{lit[2]} "
          f"-> {NAME} {lev[0]}/{lev[1]}/{lev[2]}; fab_sh declared "
          f"{declared} B; bra-flush atomics per block at 128 threads "
          f"{block_fab.bra_flush_atomics(cls, 128, lever=False)} -> "
          f"{block_fab.bra_flush_atomics(cls, 128, lever=True)}", flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_block_fab_kernels_compile_with_their_figures(cls, nvcc,
                                                           tmp_path) -> None:
    _figures("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_block_fab_kernels_compile_with_their_figures(cls, hipcc,
                                                          tmp_path) -> None:
    _figures("hip", hipcc.path, cls, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the lever's launches against the host
# ---------------------------------------------------------------------------


def lever_recs(cls, threads: int) -> tuple[FRec, ...]:
    """test_fock_ondevice.py's recorded launches of `cls`, relaunched with
    the lever's kernel at `threads` (the same arguments)."""
    k = lever(cls)
    out = []
    for r in tfo._class_recs(cls):
        grid, (t, _) = block_fab.launch_shape(r.grid, threads, WAVEFRONT)
        out.append(FRec(k, grid, t, r.args))
    return tuple(out)


def _run(backend: str, cls, threads: int, request, run_device_binary,
         tmp_path: Path) -> None:
    tc = tfo._toolchain(backend, request)
    recs = lever_recs(cls, threads)
    kernels = [recs[0].kernel]
    binary = tfo._build(backend, tc, kernels, tmp_path, kernels[0].name)
    basis, _, _ = tfo.molecule(tfo.PART1)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    w = tfo._check_class(backend, cls, Fs)
    print(f"\n{backend} {cls} {NAME} at {threads} threads on {tfo.PART1}: "
          f"{len(recs)} launches; worst |F - sweep| / scale "
          f"{w['sweep']:.2e} (tol {tfo.TOL:g}), |F - literal interp| / "
          f"scale {w['interp']:.2e} (tol {2 * tfo.TOL:g}), runs "
          f"{w['runs']:.2e}, bit-identical {bool(w['bitwise_runs'])}",
          flush=True)


DEVICE_PARAMS = [pytest.param(c, t, id=f"{''.join(map(str, c))}-t{t}")
                 for c in L4 for t in DEVICE_THREADS]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("cls", "threads"), DEVICE_PARAMS)
def test_cuda_f_lever_block_fab_matches_the_host(cls, threads, request,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """The lever's launches on the RTX 3080, against the host."""
    _run("cuda", cls, threads, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("cls", "threads"), DEVICE_PARAMS)
def test_hip_f_lever_block_fab_matches_the_host(cls, threads, request,
                                                run_device_binary,
                                                tmp_path) -> None:
    """The lever's launches on the RX 9060 XT, against the host."""
    _run("hip", cls, threads, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells and the relaunched records
# ---------------------------------------------------------------------------


def test_the_device_cells_are_l_at_most_4_and_cover_it() -> None:
    assert all(sum(c) <= 4 for c in L4)
    assert len(L4) == sum(1 for c in CLASSES if sum(c) <= 4)
    assert (SM_86.wavefront, GFX1200.wavefront) == (WAVEFRONT, WAVEFRONT)
    assert all(t % WAVEFRONT == 0 for t in DEVICE_THREADS)


def test_the_relaunched_records_differ_only_in_kernel_and_block() -> None:
    cls = (0, 0, 0, 1)
    for a, b in zip(tfo._class_recs(cls), lever_recs(cls, 128), strict=True):
        assert (a.grid, a.args) == (b.grid, b.args)
        assert b.threads == 128 and b.kernel.name == "f_0001_lv_block_fab"
        assert b.kernel.params == a.kernel.params
