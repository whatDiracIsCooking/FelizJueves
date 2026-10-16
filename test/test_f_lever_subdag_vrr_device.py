"""The dedicated sub-DAG VRR schedule (#357) through the device compilers and
on the cards.

The lever is a composition on #350 (`batched-vrr+subdag-vrr`; alone it is
refused), so every kernel here is the composition's, named as the CLI names
a unit (`lever_ladder.unit_name`: `f_0101_lv_batched_vrr_lv_subdag_vrr`).
Every F class with L = la + lb + lc + ld <= 4 (milestone 13's cap; 13 of
the 21):

1. COMPILE-ONLY (`cuda` / `hip`, no card): each class's literal kernel,
   #350's (`batched-vrr` alone) and the composition, each its own
   translation unit, compiled with the tier's flags plus the measurement
   flag (`measured.Backend.command`: `-Xptxas -v` on nvcc, the AMD
   `kernel-resource-usage` remark on hipcc). A warning in generated text
   fails (#92's rule). The figures -- registers, spill-store bytes and stack
   frame (nvcc), VGPRs, spilled VGPRs and scratch bytes (hipcc) -- literal ->
   batched-vrr -> batched-vrr+subdag-vrr, and the schedule's direction
   against #350, are PRINTED (run with `-s`) as observations: compiler
   readings, never store rows, no threshold; a null result is a pass. The
   harness programs part 2 runs are compiled too.
2. ON THE CARD (`gpu` + `cuda` / `hip`): test_fock_ondevice.py's launches
   (`fock_driver.record`), every kernel swapped for the composition's of its
   class: (a) per class, H2O/cc-pVDZ's launches of that class at `tau = 0`
   against those launches interpreted with the literal kernel -- which the
   composition equals bit for bit on the host
   (test/test_f_lever_subdag_vrr.py) -- and `ao.fock`'s sweep, with
   test_fock_ondevice's `_check_class` and its tolerances unchanged; (b) per
   molecule, H2O/STO-3G at `tau = 0` and `tau = 0.1, n = 4` and H2O/6-31G at
   `tau = 0` (every class L <= 4), against walk_f's interpreted build and,
   at `tau = 0`, the sweep and libcint. Nothing is recorded.

Run as (each backend; no F unit is on KERNEL_GEN_CMD's ladder, the driver
emits its own, so the generator is only capped):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- \\
        -k f_lever_subdag_vrr

which selects parts 1 and 2 and nothing `slow` or above L = 4.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_fock_ondevice as fo
from fock_driver import fock_stage, run_on_card
from test_fock_build import LIBCINT_TOL, TOL, molecule, sweep, walk
from test_kernel_device import BACKEND, _toolchain

from felizjueves import kernel_f as KF
from felizjueves import measured
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import subdag_vrr as sv
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.lever_ladder import unit_name

NAME = sv.NAME
BOTH = (bv.NAME, NAME)
SUFFIX = unit_name("", BOTH)  # "_lv_batched_vrr_lv_subdag_vrr"
CAP = 4
CAPPED = tuple(c for c in CLASSES if sum(c) <= CAP)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in CAPPED]
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, id="h2o_sto3g-tau0"),
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0"),
]


@cache
def _built(cls, lv: tuple[str, ...]) -> Kernel:
    cls = tuple(cls)
    return KF.build_f_kernel(KF.f_class_spec(*cls, lv), name=unit_name(
        "f_" + "".join(map(str, cls)), lv))


def lever_kernel(cls) -> Kernel:
    """The composition's kernel of `cls`, named as the CLI names its unit."""
    return _built(tuple(cls), BOTH)


def _cls_of(kernel: Kernel) -> tuple[int, ...]:
    m = re.fullmatch(r"f_(\d)(\d)(\d)(\d)", kernel.name.removesuffix(SUFFIX))
    assert m, kernel.name
    return tuple(int(x) for x in m.groups())


def _swap(recs) -> tuple:
    """`recs` with every literal kernel replaced by the composition's."""
    return tuple(dataclasses.replace(r, kernel=lever_kernel(_cls_of(r.kernel)))
                 for r in recs)


# ---------------------------------------------------------------------------
# 1. Compile-only: registers, spill and local memory, three arms
# ---------------------------------------------------------------------------

EMIT = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}
WARNING = re.compile(r"^(?P<file>[^\s:]+):\d+:\d+: warning:", re.MULTILINE)


def _compile(backend: str, compiler: str, kern: Kernel, workdir: Path
             ) -> tuple[int, int, int]:
    """(registers or VGPRs, spill-store bytes or spilled VGPRs, stack frame
    or scratch bytes) of one unit."""
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
                raw["stack_frame_bytes"])
    raw = measured.parse_amd_remarks(report, kern.name)
    return (raw["VGPRs"], raw["VGPRs Spill"],
            raw.get("ScratchSize", raw.get("Private Segment Size", -1)))


def _direction(was: int, now: int) -> str:
    return "=" if was == now else ("down" if now < was else "up")


def _figures(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    arms = (("literal", KF.build_f_kernel(cls)),
            (bv.NAME, _built(tuple(cls), (bv.NAME,))),
            ("+".join(BOTH), lever_kernel(cls)))
    got = [_compile(backend, compiler, k, tmp_path / a) for a, k in arms]
    unit = ("registers/spill-store bytes/stack frame" if backend == "cuda"
            else "VGPRs/spilled VGPRs/scratch bytes")
    peaks = sv.group_peaks(cls)
    print(f"\n{backend} {cls} ({unit}; {len(peaks)} groups, peak "
          f"{max(p[1] for p in peaks)} -> {max(p[2] for p in peaks)}): "
          + " -> ".join("/".join(map(str, g)) for g in got)
          + f" (vs {bv.NAME}: regs {_direction(got[1][0], got[2][0])}, "
          f"spill {_direction(got[1][1], got[2][1])}, local "
          f"{_direction(got[1][2], got[2][2])})", flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_subdag_vrr_compiles_with_its_figures(cls, nvcc,
                                                          tmp_path) -> None:
    _figures("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_subdag_vrr_compiles_with_its_figures(cls, hipcc,
                                                         tmp_path) -> None:
    _figures("hip", hipcc.path, cls, tmp_path)


def _program_kernels(cell) -> tuple[list[Kernel], str]:
    if cell[0] == "class":
        return [lever_kernel(cell[1])], fo._class_name(cell[1]) + "_sv"
    recs = _swap(fo._plan(*cell[1:])[0])
    return fo._kernels(recs), f"fock_{cell[1]}_sv"


PROGRAM_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}")
      for c in CAPPED),
    *(pytest.param(("molecule", *p.values), id=p.id) for p in MOLECULES),
]


@pytest.mark.cuda
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_cuda_f_lever_subdag_vrr_programs_compile(cell, nvcc,
                                                  tmp_path) -> None:
    kernels, name = _program_kernels(cell)
    fo._build("cuda", nvcc, kernels, tmp_path, name)


@pytest.mark.hip
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_hip_f_lever_subdag_vrr_programs_compile(cell, hipcc,
                                                 tmp_path) -> None:
    kernels, name = _program_kernels(cell)
    fo._build("hip", hipcc, kernels, tmp_path, name)


# ---------------------------------------------------------------------------
# 2. On the card
# ---------------------------------------------------------------------------


def _per_class(backend: str, cls, request, run_device_binary,
               tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs = _swap(fo._class_recs(cls))
    kernels = [recs[0].kernel]
    binary = fo._build(backend, tc, kernels, tmp_path, kernels[0].name)
    basis, _, _ = molecule(fo.PART1)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    w = fo._check_class(backend, cls, Fs)  # the literal's references
    print(f"\n{backend} {BACKEND[backend][1].name} {cls} {'+'.join(BOTH)} "
          f"({bv.n_groups(cls)} groups) on {fo.PART1}: {len(recs)} launches; "
          f"worst |F - sweep| / scale {w['sweep']:.2e} (tol {TOL:g}), "
          f"|F - literal interp| / scale {w['interp']:.2e} (tol "
          f"{2 * TOL:g}), runs {w['runs']:.2e}, bit-identical "
          f"{bool(w['bitwise_runs'])}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_subdag_vrr_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_subdag_vrr_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("hip", cls, request, run_device_binary, tmp_path)


def _per_molecule(backend: str, name: str, tau: float, n: int, request,
                  run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs = _swap(fo._plan(name, tau, n)[0])
    kernels = fo._kernels(recs)
    binary = fo._build(backend, tc, kernels, tmp_path, f"fock_{name}_sv")
    basis, _, libcint = molecule(name)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    got = [np.array(KF.close(F)) for F in Fs]
    w_int, _ = walk(name, tau, n)  # the literal kernels, interpreted
    F_int = np.array(w_int.F)
    scale = sweep(name, block="scale")
    w = {"interp": fo._worst(got[0], F_int, scale),
         "runs": fo._worst(got[0], got[1], scale)}
    assert fo._within(got[0], F_int, scale, 2 * TOL), (backend, name, tau, w)
    assert fo._within(got[1], got[0], scale, 2 * TOL), (backend, name, tau, w)
    if tau == 0.0:
        w["sweep"] = fo._worst(got[0], sweep(name), scale)
        w["libcint"] = fo._worst(got[0], libcint, scale)
        assert fo._within(got[0], sweep(name), scale, TOL), (backend, name, w)
        assert fo._within(got[0], libcint, scale, LIBCINT_TOL), (backend,
                                                                 name, w)
    print(f"\n{backend} {BACKEND[backend][1].name} {name} tau={tau} n={n} "
          f"{'+'.join(BOTH)}: {len(recs)} launches of "
          f"{[k.name for k in kernels]}; worst / scale "
          + ", ".join(f"{k} {v:.2e}" for k, v in w.items()))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_cuda_f_lever_subdag_vrr_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("cuda", name, tau, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_f_lever_subdag_vrr_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("hip", name, tau, n, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells stay under the cap; the swap is the composition's
# ---------------------------------------------------------------------------


def test_the_cells_are_l_at_most_4_and_the_swap_is_the_compositions() -> None:
    assert SUFFIX == "_lv_batched_vrr_lv_subdag_vrr"
    assert len(CAPPED) == 13 and all(sum(c) <= CAP for c in CAPPED)
    for p in MOLECULES:
        recs = _swap(fo._plan(*p.values)[0])
        assert {sum(_cls_of(r.kernel)) for r in recs} <= set(range(CAP + 1))
        assert all(r.kernel.name.endswith(SUFFIX) for r in recs)
    k = lever_kernel((0, 1, 0, 1))
    lit = KF.build_f_kernel((0, 1, 0, 1))
    assert k.params == lit.params and k.name == "f_0101" + SUFFIX
    assert dataclasses.replace(k, name=lit.name) == KF.build_f_kernel(
        KF.f_class_spec(0, 1, 0, 1, BOTH))
