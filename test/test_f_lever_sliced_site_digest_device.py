"""The interleaved digest per slice (#359) through the device compilers and
on the cards.

The lever is a composition on #356 (`batched-vrr+sliced-hrr+
sliced-site-digest`, f_levers/sliced_site_digest.py), so every unit here
carries all three levers. TWO PARTS, every F class with L = la + lb + lc +
ld <= 4 (milestone 13's cap; 13 of the 21):

1. COMPILE-ONLY (`cuda` / `hip`, no card): each class's literal kernel,
   #356's composition (H2) and this one (H3), each its own translation
   unit, compiled with the tier's flags plus the measurement flag
   (test_f_lever_batched_vrr_device.py's `_compile`). A warning in generated
   text fails (#92's rule). The figures -- registers, spill-store bytes and
   the stack frame (nvcc), VGPRs, spilled VGPRs and scratch bytes (hipcc) --
   are PRINTED (run with `-s`) literal -> H2 -> H3, the issue's acceptance
   ("ptxas/VGPR both cards vs H2; a null is a pass, recorded as one"):
   compiler readings, never store rows, no threshold (#362 is the ledger).
   The harness programs part 2 runs are compiled too.
2. ON THE CARD (`gpu` + `cuda` / `hip`): test_fock_ondevice.py's launches
   with every kernel swapped for this composition's kernel of its class:
   (a) per class, H2O/cc-pVDZ's launches of that class at `tau = 0`,
   against the LITERAL kernel interpreted -- which the composition equals
   bit for bit on the host (test_f_lever_sliced_site_digest.py) -- and
   `ao.fock`'s sweep, by test_fock_ondevice's `_check_class` and its stated
   tolerances unchanged; (b) per molecule, H2O/STO-3G at `tau = 0` and
   `tau = 0.1, n = 4` and H2O/6-31G at `tau = 0`, every launch of the plan,
   against walk_f's interpreted build and, at `tau = 0`, the sweep and
   libcint. Nothing is recorded.

Run as (each backend; no F unit is on KERNEL_GEN_CMD's ladder, the driver
emits its own, so the generator is only capped):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- \\
        -k f_lever_sliced_site_digest

which selects parts 1 and 2 and nothing `slow` or above L = 4.
"""

from __future__ import annotations

import dataclasses
from functools import cache
from pathlib import Path

import pytest
import test_f_lever_sliced_hrr_device as h2d
import test_fock_ondevice as fo
from test_f_lever_batched_vrr_device import _compile

from felizjueves import kernel_f as KF
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import sliced_hrr as sh
from felizjueves.f_levers import sliced_site_digest as si
from felizjueves.ir import Kernel
from felizjueves.lever_ladder import unit_name

H2 = (bv.NAME, sh.NAME)
ALL = (bv.NAME, sh.NAME, si.NAME)
CAP = h2d.CAP
CAPPED = h2d.CAPPED
CLASS_PARAMS = h2d.CLASS_PARAMS
MOLECULES = h2d.MOLECULES
SUFFIX = unit_name("", ALL)


@cache
def lever_kernel(cls) -> Kernel:
    """This composition's kernel of `cls`, named as the CLI would name it."""
    return h2d._kernel(cls, ALL)


def _swap(recs) -> tuple:
    """`recs` with every literal kernel replaced by this composition's."""
    return tuple(dataclasses.replace(
        r, kernel=lever_kernel(h2d._cls_of(r.kernel.name))) for r in recs)


@pytest.fixture
def interleaved(monkeypatch: pytest.MonkeyPatch):
    """#356's device helpers pointed at this composition's kernels."""
    monkeypatch.setattr(h2d, "lever_kernel", lever_kernel)
    monkeypatch.setattr(h2d, "BOTH", ALL)


# ---------------------------------------------------------------------------
# 1. Compile-only: literal, H2, H3
# ---------------------------------------------------------------------------


def _figures(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    arms = (("literal", KF.build_f_kernel(cls)),
            ("H2 " + "+".join(H2), h2d.lever_kernel(cls)),
            ("H3 +" + si.NAME, lever_kernel(cls)))
    got = [_compile(backend, compiler, k, tmp_path / str(i))
           for i, (_, k) in enumerate(arms)]
    unit = ("registers/spill-store bytes/stack frame" if backend == "cuda"
            else "VGPRs/spilled VGPRs/scratch bytes")
    print(f"\n{backend} {cls} ({unit}): " + " -> ".join(
        f"{arm} {a}/{b}/{c}" for (arm, _), (a, b, c) in zip(arms, got,
                                                            strict=True))
        + f" ({sh.n_slices(cls)} slices, {si.n_groups(cls)} groups)"
        + ("  [H3 == H2: null]" if got[2] == got[1] else ""), flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_sliced_site_digest_compiles_with_its_figures(
        cls, nvcc, tmp_path) -> None:
    _figures("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_sliced_site_digest_compiles_with_its_figures(
        cls, hipcc, tmp_path) -> None:
    _figures("hip", hipcc.path, cls, tmp_path)


PROGRAM_CELLS = h2d.PROGRAM_CELLS


@pytest.mark.cuda
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_cuda_f_lever_sliced_site_digest_programs_compile(
        cell, nvcc, tmp_path, interleaved) -> None:
    kernels, name = h2d._program_kernels(cell)
    fo._build("cuda", nvcc, kernels, tmp_path, name)


@pytest.mark.hip
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_hip_f_lever_sliced_site_digest_programs_compile(
        cell, hipcc, tmp_path, interleaved) -> None:
    kernels, name = h2d._program_kernels(cell)
    fo._build("hip", hipcc, kernels, tmp_path, name)


# ---------------------------------------------------------------------------
# 2. On the card (#356's checks, on this composition's kernels)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_sliced_site_digest_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path, interleaved) -> None:
    h2d._per_class("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_sliced_site_digest_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path, interleaved) -> None:
    h2d._per_class("hip", cls, request, run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_cuda_f_lever_sliced_site_digest_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path,
        interleaved) -> None:
    h2d._per_molecule("cuda", name, tau, n, request, run_device_binary,
                      tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_f_lever_sliced_site_digest_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path,
        interleaved) -> None:
    h2d._per_molecule("hip", name, tau, n, request, run_device_binary,
                      tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells stay under the cap; the swap is this composition's
# ---------------------------------------------------------------------------


def test_the_cells_are_l_at_most_4_and_the_swap_is_this_composition(
        interleaved) -> None:
    assert len(CAPPED) == 13 and all(sum(c) <= CAP for c in CAPPED)
    assert SUFFIX == "_lv_batched_vrr_lv_sliced_hrr_lv_sliced_site_digest"
    for p in MOLECULES:
        for recs in (_swap(fo._plan(*p.values)[0]),
                     h2d._swap(fo._plan(*p.values)[0])):
            assert {sum(h2d._cls_of(r.kernel.name)) for r in recs} <= set(
                range(CAP + 1))
            assert all(r.kernel.name.endswith(SUFFIX) for r in recs)
    k = lever_kernel((0, 1, 0, 1))
    lit = KF.build_f_kernel((0, 1, 0, 1))
    assert k.params == lit.params and k.name == "f_0101" + SUFFIX
    assert dataclasses.replace(k, name=lit.name) == KF.build_f_kernel(
        KF.f_class_spec(0, 1, 0, 1, ALL))
    cell = ("class", (0, 1, 0, 1))
    assert h2d._program_kernels(cell)[0] == [k]
