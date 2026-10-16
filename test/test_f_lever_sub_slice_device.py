"""Slice-interior sub-slicing (#361) through the device compilers and on the
cards.

The lever is a composition (`batched-vrr+sliced-hrr+slim-fock+sub-slice`,
f_levers/sub_slice.py), so every unit here carries the four levers. TWO
PARTS, every F class with L = la + lb + lc + ld <= 4 (milestone 13's cap;
13 of the 21):

1. COMPILE-ONLY (`cuda` / `hip`, no card): each class's literal kernel,
   #360's composition (`batched-vrr+sliced-hrr+slim-fock`, H4) and this
   one, each its own translation unit, compiled with the tier's flags plus
   the measurement flag (test_f_lever_batched_vrr_device.py's helper:
   `-Xptxas -v` on nvcc, the AMD `kernel-resource-usage` remark on hipcc).
   A warning in generated text fails (#92's rule). The figures --
   registers, spill-store bytes and stack frame (nvcc), VGPRs, spilled
   VGPRs and scratch bytes (hipcc) -- are PRINTED (run with `-s`) beside
   the declared `n_sub` and the slice-interior peak, #356's slice peak
   against the sub-slice peak, so the lever's contribution against H4 is
   separable (the issue's acceptance): compiler readings, never store rows,
   no threshold (#362 is the ledger). The harness programs part 2 runs are
   compiled too.
2. ON THE CARD (`gpu` + `cuda` / `hip`): test_fock_ondevice.py's launches
   with every kernel swapped for this composition's kernel of its class:
   (a) per class, H2O/cc-pVDZ's launches of that class at `tau = 0`,
   against those launches interpreted with the LITERAL kernel and
   `ao.fock`'s restricted sweep, by test_fock_ondevice's `_check_class` and
   its stated tolerances unchanged (the arithmetic is #360's bit for bit,
   which its `2 TOL` against the interpreted literal already admits); (b)
   per molecule, H2O/STO-3G at `tau = 0` and `tau = 0.1, n = 4` and
   H2O/6-31G at `tau = 0`, every launch of the plan, against walk_f's
   interpreted build and, at `tau = 0`, the sweep and libcint. Nothing is
   recorded.

Run as (each backend; no F unit is on KERNEL_GEN_CMD's ladder, the driver
emits its own, so the generator is only capped):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- \\
        -k f_lever_sub_slice

which selects parts 1 and 2 and nothing `slow` or above L = 4.
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
from test_f_lever_batched_vrr_device import _compile
from test_fock_build import LIBCINT_TOL, TOL, molecule, sweep, walk
from test_kernel_device import BACKEND, _toolchain

from felizjueves import kernel_f as KF
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import sliced_hrr as sh
from felizjueves.f_levers import slim_fock as sf
from felizjueves.f_levers import sub_slice as ss
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.lever_ladder import unit_name

H4 = (bv.NAME, sh.NAME, sf.NAME)
STACK = (*H4, ss.NAME)
CAP = 4
CAPPED = tuple(c for c in CLASSES if sum(c) <= CAP)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in CAPPED]
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, id="h2o_sto3g-tau0"),
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0"),
]
SUFFIX = unit_name("", STACK)  # "..._lv_sub_slice_lv_sub_slice"


def _kernel(cls, levers) -> Kernel:
    cls = tuple(cls)
    return KF.build_f_kernel(KF.f_class_spec(*cls, levers), name=unit_name(
        "f_" + "".join(map(str, cls)), levers))


@cache
def lever_kernel(cls) -> Kernel:
    """The composition's kernel of `cls`, named as the CLI would name it."""
    return _kernel(cls, STACK)


def _cls_of(name: str) -> tuple[int, ...]:
    m = re.fullmatch(r"f_(\d)(\d)(\d)(\d)(?:_lv_\w+)?", name)
    assert m, name
    return tuple(int(x) for x in m.groups())


def _swap(recs) -> tuple:
    """`recs` with every literal kernel replaced by the composition's."""
    return tuple(dataclasses.replace(
        r, kernel=lever_kernel(_cls_of(r.kernel.name))) for r in recs)


# ---------------------------------------------------------------------------
# 1. Compile-only: literal, #360's composition (H4), this one
# ---------------------------------------------------------------------------


def _figures(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    arms = (("literal", KF.build_f_kernel(cls)),
            ("+".join(H4), _kernel(cls, H4)),
            ("+".join(STACK), lever_kernel(cls)))
    got = [_compile(backend, compiler, k, tmp_path / arm) for arm, k in arms]
    unit = ("registers/spill-store bytes/stack frame" if backend == "cuda"
            else "VGPRs/spilled VGPRs/scratch bytes")
    print(f"\n{backend} {cls} ({unit}): " + " -> ".join(
        f"{arm} {a}/{b}/{c}" for (arm, _), (a, b, c) in zip(arms, got,
                                                            strict=True))
        + f" (n_sub {ss.n_sub(cls)}; slice-interior peak "
        f"{sh.hrr_phase(cls)} -> {ss.sub_phase(cls)}; HRR statements per "
        f"quartet {ss.statements(cls, lever=False)} -> "
        f"{ss.statements(cls, lever=True)}; ket atomics per quartet "
        f"{sf.ket_atomics(cls, lever=True)}, unchanged)", flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_sub_slice_compiles_with_its_figures(cls, nvcc,
                                                          tmp_path) -> None:
    _figures("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_sub_slice_compiles_with_its_figures(cls, hipcc,
                                                         tmp_path) -> None:
    _figures("hip", hipcc.path, cls, tmp_path)


def _program_kernels(cell) -> tuple[list[Kernel], str]:
    if cell[0] == "class":
        return [lever_kernel(cell[1])], fo._class_name(cell[1]) + "_lvss"
    recs = _swap(fo._plan(*cell[1:])[0])
    return fo._kernels(recs), f"fock_{cell[1]}_lvss"


PROGRAM_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}")
      for c in CAPPED),
    *(pytest.param(("molecule", *p.values), id=p.id) for p in MOLECULES),
]


@pytest.mark.cuda
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_cuda_f_lever_sub_slice_programs_compile(cell, nvcc,
                                                  tmp_path) -> None:
    kernels, name = _program_kernels(cell)
    fo._build("cuda", nvcc, kernels, tmp_path, name)


@pytest.mark.hip
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_hip_f_lever_sub_slice_programs_compile(cell, hipcc,
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
    print(f"\n{backend} {BACKEND[backend][1].name} {cls} {'+'.join(STACK)} "
          f"(slice peak {sh.hrr_phase(cls)} -> {ss.sub_phase(cls)}) on "
          f"{fo.PART1}: {len(recs)} launches; worst |F - sweep| / scale "
          f"{w['sweep']:.2e} (tol {TOL:g}), |F - literal interp| / scale "
          f"{w['interp']:.2e} (tol {2 * TOL:g}), runs {w['runs']:.2e}, "
          f"bit-identical {bool(w['bitwise_runs'])}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_sub_slice_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_sub_slice_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("hip", cls, request, run_device_binary, tmp_path)


def _per_molecule(backend: str, name: str, tau: float, n: int, request,
                  run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs = _swap(fo._plan(name, tau, n)[0])
    kernels = fo._kernels(recs)
    binary = fo._build(backend, tc, kernels, tmp_path, f"fock_{name}_lvss")
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
          f"{'+'.join(STACK)}: {len(recs)} launches of "
          f"{[k.name for k in kernels]}; worst / scale "
          + ", ".join(f"{k} {v:.2e}" for k, v in w.items()))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_cuda_f_lever_sub_slice_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("cuda", name, tau, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_f_lever_sub_slice_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("hip", name, tau, n, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells stay under the cap; the swap is the composition's
# ---------------------------------------------------------------------------


def test_the_cells_are_l_at_most_4_and_the_swap_is_the_composition() -> None:
    assert len(CAPPED) == 13 and all(sum(c) <= CAP for c in CAPPED)
    assert SUFFIX == ("_lv_batched_vrr_lv_sliced_hrr_lv_slim_fock"
                      "_lv_sub_slice")
    for p in MOLECULES:
        recs = _swap(fo._plan(*p.values)[0])
        assert {sum(_cls_of(r.kernel.name)) for r in recs} <= set(
            range(CAP + 1))
        assert all(r.kernel.name.endswith(SUFFIX) for r in recs)
    k = lever_kernel((0, 1, 0, 1))
    lit = KF.build_f_kernel((0, 1, 0, 1))
    assert k.params == lit.params and k.name == "f_0101" + SUFFIX
    assert dataclasses.replace(k, name=lit.name) == KF.build_f_kernel(
        KF.f_class_spec(0, 1, 0, 1, STACK))
