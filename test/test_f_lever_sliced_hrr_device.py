"""The sliced HRR (#356) through the device compilers and on the cards.

The lever is a composition (`batched-vrr+sliced-hrr`, f_levers/sliced_hrr.py),
so every unit here carries both levers. TWO PARTS, every F class with
L = la + lb + lc + ld <= 4 (milestone 13's cap; 13 of the 21):

1. COMPILE-ONLY (`cuda` / `hip`, no card): each class's literal kernel,
   #350's alone and the composition, each its own translation unit, compiled
   with the tier's flags (test_f_lever_batched_vrr_device.py's
   `compile_unit`). A warning in generated text fails (#92's rule). Nothing
   is read off the compiler: the arms' registers, spills and frames are the
   lever ledger's (`measured --remeasure ... --ledger`, #362), not a test's.
   The harness programs part 2 runs are compiled too.
2. ON THE CARD (`gpu` + `cuda` / `hip`): test_fock_ondevice.py's launches
   with every kernel swapped for the composition's kernel of its class
   (`f_0101_lv_batched_vrr_lv_sliced_hrr`): (a) `slow` (cc-pVDZ is not a
   small input: its host references cost 15-30 s a class): per class,
   H2O/cc-pVDZ's launches of that class at `tau = 0`, against those launches
   interpreted with the LITERAL kernel -- which the composition equals bit
   for bit on the host (test_f_lever_sliced_hrr.py) -- and `ao.fock`'s
   restricted sweep, by test_fock_ondevice's `_check_class` and its stated
   tolerances unchanged; (b) per molecule, H2O/STO-3G at `tau = 0` and
   `tau = 0.1, n = 4` and, `slow`, H2O/6-31G at `tau = 0`, every launch of the
   plan, against walk_f's interpreted build and, at `tau = 0`, the sweep and
   libcint. Nothing is recorded.

Run as (each backend; no F unit is on KERNEL_GEN_CMD's ladder, the driver
emits its own, so the generator is only capped):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip --full -j 6 -- \\
        -k f_lever_sliced_hrr

which selects part 2 and nothing above L = 4; without `--full`, only 2(b)'s
H2O/STO-3G cells (nothing `slow`), the PR tier's.
"""

from __future__ import annotations

import dataclasses
import re
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_fock_ondevice as fo
from fock_driver import fock_stage, run_on_card
from test_f_lever_batched_vrr_device import compile_unit
from test_fock_build import LIBCINT_TOL, TOL, molecule, sweep, walk
from test_kernel_device import BACKEND, _toolchain

from felizjueves import kernel_f as KF
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import sliced_hrr as sh
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.ir.lever_ladder import unit_name

BOTH = (bv.NAME, sh.NAME)
CAP = 4
CAPPED = tuple(c for c in CLASSES if sum(c) <= CAP)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in CAPPED]
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, id="h2o_sto3g-tau0"),
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0",
                 marks=pytest.mark.slow),  # not a small input: the PR tier's cut
]
SUFFIX = unit_name("", BOTH)  # "_lv_batched_vrr_lv_sliced_hrr"


def _kernel(cls, levers) -> Kernel:
    cls = tuple(cls)
    return KF.build_f_kernel(KF.f_class_spec(*cls, levers), name=unit_name(
        "f_" + "".join(map(str, cls)), levers))


@cache
def lever_kernel(cls) -> Kernel:
    """The composition's kernel of `cls`, named as the CLI would name it."""
    return _kernel(cls, BOTH)


def _cls_of(name: str) -> tuple[int, ...]:
    m = re.fullmatch(r"f_(\d)(\d)(\d)(\d)(?:_lv_\w+)?", name)
    assert m, name
    return tuple(int(x) for x in m.groups())


def _swap(recs) -> tuple:
    """`recs` with every literal kernel replaced by the composition's."""
    return tuple(dataclasses.replace(
        r, kernel=lever_kernel(_cls_of(r.kernel.name))) for r in recs)


# ---------------------------------------------------------------------------
# 1. Compile-only: literal, #350 alone, the composition
# ---------------------------------------------------------------------------


def _compiles(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    arms = (("literal", KF.build_f_kernel(cls)),
            (bv.NAME, _kernel(cls, (bv.NAME,))),
            ("+".join(BOTH), lever_kernel(cls)))
    for arm, k in arms:
        compile_unit(backend, compiler, k, tmp_path / arm)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_sliced_hrr_compiles(cls, nvcc,
                                                          tmp_path) -> None:
    _compiles("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_sliced_hrr_compiles(cls, hipcc,
                                                         tmp_path) -> None:
    _compiles("hip", hipcc.path, cls, tmp_path)


def _program_kernels(cell) -> tuple[list[Kernel], str]:
    if cell[0] == "class":
        return [lever_kernel(cell[1])], fo._class_name(cell[1]) + "_lvs"
    recs = _swap(fo._plan(*cell[1:])[0])
    return fo._kernels(recs), f"fock_{cell[1]}_lvs"


PROGRAM_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}")
      for c in CAPPED),
    *(pytest.param(("molecule", *p.values), id=p.id, marks=p.marks)
      for p in MOLECULES),
]


@pytest.mark.cuda
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_cuda_f_lever_sliced_hrr_programs_compile(cell, nvcc,
                                                  tmp_path) -> None:
    kernels, name = _program_kernels(cell)
    fo._build("cuda", nvcc, kernels, tmp_path, name)


@pytest.mark.hip
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_hip_f_lever_sliced_hrr_programs_compile(cell, hipcc,
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
          f"({sh.n_slices(cls)} slices, {sh.n_groups(cls)} groups) on "
          f"{fo.PART1}: {len(recs)} launches; worst |F - sweep| / scale "
          f"{w['sweep']:.2e} (tol {TOL:g}), |F - literal interp| / scale "
          f"{w['interp']:.2e} (tol {2 * TOL:g}), runs {w['runs']:.2e}, "
          f"bit-identical {bool(w['bitwise_runs'])}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow  # H2O/cc-pVDZ's launches: not a small input
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_sliced_hrr_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow  # H2O/cc-pVDZ's launches: not a small input
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_sliced_hrr_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("hip", cls, request, run_device_binary, tmp_path)


def _per_molecule(backend: str, name: str, tau: float, n: int, request,
                  run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs = _swap(fo._plan(name, tau, n)[0])
    kernels = fo._kernels(recs)
    binary = fo._build(backend, tc, kernels, tmp_path, f"fock_{name}_lvs")
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
def test_cuda_f_lever_sliced_hrr_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("cuda", name, tau, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_f_lever_sliced_hrr_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("hip", name, tau, n, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells stay under the cap; the swap is the composition's
# ---------------------------------------------------------------------------


def test_the_cells_are_l_at_most_4_and_the_swap_is_the_composition() -> None:
    assert len(CAPPED) == 13 and all(sum(c) <= CAP for c in CAPPED)
    assert SUFFIX == "_lv_batched_vrr_lv_sliced_hrr"
    for p in MOLECULES:
        recs = _swap(fo._plan(*p.values)[0])
        assert {sum(_cls_of(r.kernel.name)) for r in recs} <= set(
            range(CAP + 1))
        assert all(r.kernel.name.endswith(SUFFIX) for r in recs)
    k = lever_kernel((0, 1, 0, 1))
    lit = KF.build_f_kernel((0, 1, 0, 1))
    assert k.params == lit.params and k.name == "f_0101" + SUFFIX
    assert dataclasses.replace(k, name=lit.name) == KF.build_f_kernel(
        KF.f_class_spec(0, 1, 0, 1, BOTH))
