"""Mixed-precision fused F (#355) through the device compilers and on the cards.

TWO PARTS, the issue's acceptance on the device side, every F class with
L = la + lb + lc + ld <= 4 (milestone 13's cap; 13 of the 21):

1. COMPILE-ONLY (`cuda` / `hip`, no card): each class's literal kernel and
   the `fmixed` kernel, each its own translation unit, compiled with the
   tier's flags (test_f_lever_batched_vrr_device.py's `compile_unit`). A
   warning in generated text fails (#92's rule). Nothing is read off the
   compiler: the arms' registers, spills and frames are the lever ledger's
   (`measured --remeasure ... --ledger`, #362), not a test's. The harness
   programs part 2 runs are compiled too.
2. ON THE CARD (`gpu` + `cuda` / `hip`): test_fock_ondevice.py's recorded
   launches (`fock_driver.record`: launch_f's plan walked by walk_f), every
   kernel swapped for the `fmixed` kernel of its class
   (`lever_ladder.unit_name`: `f_0101_lv_fmixed`): (a) `slow` (cc-pVDZ is
   not a small input: its host references cost 15-30 s a class): per class,
   H2O/cc-pVDZ's launches of that class at `tau = 0`, against `ao.fock`'s
   restricted sweep and against the same launches INTERPRETED WITH THE
   LEVER'S KERNEL (`fock_driver.interpret`, whose F32 arithmetic rounds as
   the card's does), and run against run; (b) per molecule, H2O/STO-3G at
   `tau = 0` and `tau = 0.1, n = 4` and, `slow`, H2O/6-31G at `tau = 0` (s
   and p shells only: every class L <= 4), every launch of the plan, against
   walk_f's build interpreted with the lever's kernels and, at `tau = 0`,
   the sweep and libcint. EVERY comparison is at `F32_REL` of `scale_G`
   (test/oracle/tolerance.py's one fp32 tier, never widened); none at an
   fp64 tolerance, which the build is not claimed to meet. Nothing is
   recorded.

Run part 2 as (each backend; no F unit is on KERNEL_GEN_CMD's ladder, the
driver emits its own, so the generator is only capped):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip --full -j 6 -- \\
        -k f_lever_mixed

which selects part 2 and nothing above L = 4; without `--full`, only 2(b)'s
H2O/STO-3G cells (nothing `slow`), the PR tier's (the pattern does not match
test_k_lever_mixed*.py or test_kernel_mixed*.py).
"""

from __future__ import annotations

import dataclasses
import re
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_fock_ondevice as fo
from fock_driver import fock_stage, interpret, run_on_card
from oracle.tolerance import F32_REL
from test_f_lever_batched_vrr_device import compile_unit
from test_fock_build import molecule, sweep
from test_fock_build import run as interp_run
from test_kernel_device import BACKEND, _toolchain

from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock.launch_f import f_launches
from felizjueves.eri.fock.levers import mixed
from felizjueves.eri.fock.walk_f import walk_f
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.ir.lever_ladder import unit_name

NAME = mixed.NAME
CAP = 4
CAPPED = tuple(c for c in CLASSES if sum(c) <= CAP)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in CAPPED]
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, id="h2o_sto3g-tau0"),
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0",
                 marks=pytest.mark.slow),  # not a small input: the PR tier's cut
]
SUFFIX = "_lv_" + NAME


@cache
def lever_kernel(cls) -> Kernel:
    """The lever's kernel of `cls`, named as the CLI names its unit."""
    cls = tuple(cls)
    return KF.build_f_kernel(KF.f_class_spec(*cls, (NAME,)), name=unit_name(
        "f_" + "".join(map(str, cls)), (NAME,)))


def _cls_of(name: str) -> tuple[int, ...]:
    m = re.fullmatch(r"f_(\d)(\d)(\d)(\d)(?:" + SUFFIX + ")?", name)
    assert m, name
    return tuple(int(x) for x in m.groups())


def _swap(recs) -> tuple:
    """`recs` with every literal kernel replaced by the lever's."""
    return tuple(dataclasses.replace(r, kernel=lever_kernel(
        _cls_of(r.kernel.name))) for r in recs)


# ---------------------------------------------------------------------------
# 1. Compile-only: registers and spill, literal vs lever
# ---------------------------------------------------------------------------


def _compiles(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    for arm, k in (("literal", KF.build_f_kernel(cls)),
                   ("lever", lever_kernel(cls))):
        compile_unit(backend, compiler, k, tmp_path / arm)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_mixed_compiles(cls, nvcc,
                                                      tmp_path) -> None:
    _compiles("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_mixed_compiles(cls, hipcc,
                                                     tmp_path) -> None:
    _compiles("hip", hipcc.path, cls, tmp_path)


def _program_kernels(cell) -> tuple[list[Kernel], str]:
    if cell[0] == "class":
        return [lever_kernel(cell[1])], fo._class_name(cell[1]) + "_lvm"
    recs = _swap(fo._plan(*cell[1:])[0])
    return fo._kernels(recs), f"fock_{cell[1]}_lvm"


PROGRAM_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}")
      for c in CAPPED),
    *(pytest.param(("molecule", *p.values), id=p.id, marks=p.marks)
      for p in MOLECULES),
]


@pytest.mark.cuda
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_cuda_f_lever_mixed_programs_compile(cell, nvcc, tmp_path) -> None:
    kernels, name = _program_kernels(cell)
    fo._build("cuda", nvcc, kernels, tmp_path, name)


@pytest.mark.hip
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_hip_f_lever_mixed_programs_compile(cell, hipcc, tmp_path) -> None:
    kernels, name = _program_kernels(cell)
    fo._build("hip", hipcc, kernels, tmp_path, name)


# ---------------------------------------------------------------------------
# 2. On the card, at the fp32 tier
# ---------------------------------------------------------------------------


@cache
def _class_reference(cls):
    """(the lever's launches interpreted, closed; the sweep G; scale_G)."""
    recs = _swap(fo._class_recs(cls))
    basis, _, _ = molecule(fo.PART1)
    F = np.array(KF.close(interpret(recs, basis.n_ao)))
    _, G, scale = fo._class_reference(cls)
    return F, G, scale


def _per_class(backend: str, cls, request, run_device_binary,
               tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs = _swap(fo._class_recs(cls))
    kernels = [recs[0].kernel]
    binary = fo._build(backend, tc, kernels, tmp_path, kernels[0].name)
    basis, _, _ = molecule(fo.PART1)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    got = [np.array(KF.close(F)) for F in Fs]
    F_int, G, scale = _class_reference(cls)
    w = {"sweep": fo._worst(got[0], G, scale),
         "interp": fo._worst(got[0], F_int, scale),
         "runs": fo._worst(got[0], got[1], scale)}
    assert fo._within(got[0], G, scale, F32_REL), (backend, cls, w)
    assert fo._within(got[0], F_int, scale, F32_REL), (backend, cls, w)
    assert fo._within(got[1], got[0], scale, F32_REL), (backend, cls, w)
    print(f"\n{backend} {BACKEND[backend][1].name} {cls} {NAME} on "
          f"{fo.PART1}: {len(recs)} launches; worst / scale_G: sweep "
          f"{w['sweep']:.2e}, fmixed interp {w['interp']:.2e}, runs "
          f"{w['runs']:.2e} (tol F32_REL {F32_REL:.2e})")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow  # H2O/cc-pVDZ's launches: not a small input
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_mixed_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow  # H2O/cc-pVDZ's launches: not a small input
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_mixed_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("hip", cls, request, run_device_binary, tmp_path)


def _lever_run(kernel, grid, block, args):
    """walk_f's executor with the literal kernel swapped for the lever's."""
    lev = dataclasses.replace(lever_kernel(_cls_of(kernel.name)),
                              name=kernel.name)
    return interp_run(lev, grid, block, args)


@cache
def _lever_walk(name: str, tau: float, n: int) -> np.ndarray:
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=fo.BOUND, tau=tau, n=n)
    return np.array(walk_f(plan, basis, D, run=_lever_run).F)


def _per_molecule(backend: str, name: str, tau: float, n: int, request,
                  run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs = _swap(fo._plan(name, tau, n)[0])
    kernels = fo._kernels(recs)
    binary = fo._build(backend, tc, kernels, tmp_path, f"fock_{name}_lvm")
    basis, _, libcint = molecule(name)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    got = [np.array(KF.close(F)) for F in Fs]
    F_int = _lever_walk(name, tau, n)
    scale = sweep(name, block="scale")
    w = {"interp": fo._worst(got[0], F_int, scale),
         "runs": fo._worst(got[0], got[1], scale)}
    assert fo._within(got[0], F_int, scale, F32_REL), (backend, name, tau, w)
    assert fo._within(got[1], got[0], scale, F32_REL), (backend, name, tau, w)
    if tau == 0.0:
        w["sweep"] = fo._worst(got[0], sweep(name), scale)
        w["libcint"] = fo._worst(got[0], libcint, scale)
        assert fo._within(got[0], sweep(name), scale, F32_REL), (backend,
                                                                 name, w)
        assert fo._within(got[0], libcint, scale, F32_REL), (backend, name, w)
    print(f"\n{backend} {BACKEND[backend][1].name} {name} tau={tau} n={n} "
          f"{NAME}: {len(recs)} launches of {[k.name for k in kernels]}; "
          "worst / scale_G " + ", ".join(f"{k} {v:.2e}" for k, v in w.items())
          + f" (tol F32_REL {F32_REL:.2e})")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_cuda_f_lever_mixed_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("cuda", name, tau, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_f_lever_mixed_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("hip", name, tau, n, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells stay under the cap; the swap is the lever's kernel
# ---------------------------------------------------------------------------


def test_the_cells_are_l_at_most_4_and_the_swap_is_the_levers() -> None:
    assert len(CAPPED) == 13 and all(sum(c) <= CAP for c in CAPPED)
    for p in MOLECULES:
        recs = _swap(fo._plan(*p.values)[0])
        assert {sum(_cls_of(r.kernel.name)) for r in recs} <= set(
            range(CAP + 1))
        assert all(r.kernel.name.endswith(SUFFIX) for r in recs)
    k = lever_kernel((0, 1, 0, 1))
    lit = KF.build_f_kernel((0, 1, 0, 1))
    assert k.params == lit.params and k.name == "f_0101" + SUFFIX
    assert dataclasses.replace(k, name=lit.name) == KF.build_f_kernel(
        KF.f_class_spec(0, 1, 0, 1, (NAME,)))
