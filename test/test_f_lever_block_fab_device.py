"""The block-level `f_ab` reduction (#351) through the compilers and on the cards.

THREE PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): every F class with L <= 4
   (milestone 13's cap), the literal kernel and the lever's kernel each its
   own translation unit, compiled with the tier's flags and the resource
   report (test_f_lever_batched_vrr_device.py's `shared_bytes`). A warning
   in generated text fails (#92). One reading is asserted, because it is a
   property of the launch, not a figure of merit: the lever's static shared
   bytes (nvcc's `smem`; hipcc's LDS) exceed the literal's, by at most
   `block_fab.reduce_bytes(cls)`, the declared footprint (a compiler may
   not allocate less than it reads after a barrier, but nothing here
   assumes it rounds the same way on both). Registers and spills are the
   lever ledger's (`measured --remeasure ... --ledger`), not this file's.
2. ON THE CARD, PER CLASS (`gpu` + `cuda` / `hip`, `slow`): every F class
   with L <= 4, its launches in H2O/cc-pVDZ's plan at `tau = 0`
   (test_fock_ondevice.py's part 1: `_class_recs`, the same recorded packed
   records, `diag`, `Dsh` and `D`), each relaunched with the lever's kernel
   at 32 threads (one warp: the shared stage reduces one row) and 128 (four
   warps), held by test_fock_ondevice.py's own `_check_class` -- `TOL =
   1e-13` of `scale_G` against the restricted sweep, `2 TOL` against the
   literal launches interpreted, and two runs of one binary within `2 TOL`
   -- unchanged. A block of 128 is a multiple of both cards' wavefront (32)
   and under `block_fab.launch_shape`'s contract, asserted before any
   launch. `slow` because cc-pVDZ is not a small input: its host
   references cost 15-30 s a class, past the PR tier's budget.
3. ON THE CARD, PER MOLECULE (`gpu` + `cuda` / `hip`): the PR tier's
   witness of the lever. H2O/STO-3G (s and p shells: every class it meets
   has L <= 4) at `tau = 0` at 32 threads and at `tau = 0.1, n = 4` at 128,
   every launch of the plan relaunched with the lever's kernel, against
   walk_f's interpreted literal build at `2 TOL`, at `tau = 0` against the
   sweep at `TOL` and libcint at `LIBCINT_TOL`, two runs within `2 TOL`.

Run parts 2 and 3 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip --full -j 6 -- \\
        -k f_lever_block_fab

(without `--full` only part 3 is selected, nothing `slow`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_fock_ondevice as tfo
from fock_driver import FRec, fock_stage, run_on_card
from test_f_lever_batched_vrr_device import shared_bytes
from test_fock_build import LIBCINT_TOL, TOL, molecule, sweep, walk

from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock.levers import block_fab
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.ir.lever_ladder import unit_name
from felizjueves.performance.device import GFX1200, SM_86

NAME = block_fab.NAME
L4 = tuple(c for c in CLASSES if sum(c) <= 4)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in L4]
DEVICE_THREADS = (32, 128)
WAVEFRONT = 32  # both cards' (device.SM_86, device.GFX1200)
# part 3: (molecule, tau, n, threads) -- the small input, each block width
# once (one warp, four warps)
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, 32, id="h2o_sto3g-tau0-t32"),
    pytest.param("h2o_sto3g", 0.1, 4, 128, id="h2o_sto3g-tau0.1-t128"),
]


def lever(cls) -> Kernel:
    k = KF.build_f_kernel(KF.f_class_spec(*cls, (NAME,)))
    return dataclasses.replace(k, name=unit_name(k.name, (NAME,)))


# ---------------------------------------------------------------------------
# 1. Compile-only: literal and lever, and the shared footprint
# ---------------------------------------------------------------------------


def _compiles(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    lit = shared_bytes(backend, compiler, KF.build_f_kernel(cls),
                       tmp_path / "literal")
    lev = shared_bytes(backend, compiler, lever(cls), tmp_path / "lever")
    declared = block_fab.reduce_bytes(cls)
    assert lit < lev <= lit + declared, (lit, lev, declared)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_block_fab_kernels_compile(cls, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_block_fab_kernels_compile(cls, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc.path, cls, tmp_path)


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
@pytest.mark.slow  # H2O/cc-pVDZ's launches: not a small input
@pytest.mark.parametrize(("cls", "threads"), DEVICE_PARAMS)
def test_cuda_f_lever_block_fab_matches_the_host(cls, threads, request,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """The lever's launches on the RTX 3080, against the host."""
    _run("cuda", cls, threads, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow  # H2O/cc-pVDZ's launches: not a small input
@pytest.mark.parametrize(("cls", "threads"), DEVICE_PARAMS)
def test_hip_f_lever_block_fab_matches_the_host(cls, threads, request,
                                                run_device_binary,
                                                tmp_path) -> None:
    """The lever's launches on the RX 9060 XT, against the host."""
    _run("hip", cls, threads, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# 3. On the card, per molecule: the small input
# ---------------------------------------------------------------------------


def molecule_recs(name: str, tau: float, n: int, threads: int
                  ) -> tuple[FRec, ...]:
    """Every recorded launch of the molecule's plan, relaunched with the
    lever's kernel of its class at `threads` (the same arguments)."""
    kernels: dict[str, Kernel] = {}
    out = []
    for r in tfo._plan(name, tau, n)[0]:
        cls = tuple(int(c) for c in r.name[2:6])
        k = kernels.setdefault(r.name, lever(cls))
        grid, (t, _) = block_fab.launch_shape(r.grid, threads, WAVEFRONT)
        out.append(FRec(k, grid, t, r.args))
    return tuple(out)


def _per_molecule(backend: str, name: str, tau: float, n: int, threads: int,
                  request, run_device_binary, tmp_path: Path) -> None:
    tc = tfo._toolchain(backend, request)
    recs = molecule_recs(name, tau, n, threads)
    kernels = tfo._kernels(recs)
    binary = tfo._build(backend, tc, kernels, tmp_path, f"fock_{name}_lv")
    basis, _, libcint = molecule(name)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    got = [np.array(KF.close(F)) for F in Fs]
    w_int, _ = walk(name, tau, n)  # the literal kernels, interpreted
    scale = sweep(name, block="scale")
    w = {"interp": tfo._worst(got[0], np.array(w_int.F), scale),
         "runs": tfo._worst(got[0], got[1], scale)}
    assert tfo._within(got[0], np.array(w_int.F), scale, 2 * TOL), (
        backend, name, tau, w)
    assert tfo._within(got[1], got[0], scale, 2 * TOL), (backend, name, tau, w)
    if tau == 0.0:
        w["sweep"] = tfo._worst(got[0], sweep(name), scale)
        w["libcint"] = tfo._worst(got[0], libcint, scale)
        assert tfo._within(got[0], sweep(name), scale, TOL), (backend, name, w)
        assert tfo._within(got[0], libcint, scale, LIBCINT_TOL), (backend,
                                                                  name, w)
    print(f"\n{backend} {name} tau={tau} n={n} {NAME} at {threads} threads: "
          f"{len(recs)} launches of {[k.name for k in kernels]}; worst / "
          "scale " + ", ".join(f"{k} {v:.2e}" for k, v in w.items()))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n", "threads"), MOLECULES)
def test_cuda_f_lever_block_fab_whole_molecule_matches(
        name, tau, n, threads, request, run_device_binary, tmp_path) -> None:
    _per_molecule("cuda", name, tau, n, threads, request, run_device_binary,
                  tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n", "threads"), MOLECULES)
def test_hip_f_lever_block_fab_whole_molecule_matches(
        name, tau, n, threads, request, run_device_binary, tmp_path) -> None:
    _per_molecule("hip", name, tau, n, threads, request, run_device_binary,
                  tmp_path)


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
