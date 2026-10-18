"""The coalesced K write-back lever (#345): the device compilers and the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): the coalesced kernel of a set of
   K classes with L <= 4 (milestone 13's cap), each its own translation
   unit, beside the class's literal kernel, compiled with the tier's flags
   (test_f_lever_batched_vrr_device.py's `compile_unit`). A warning in
   generated text fails (#92's rule) -- the lever spells no warp sync, so
   there is none for hipcc to refuse. Nothing is read off the compiler:
   registers and spills are the lever ledger's (`measured --remeasure ...
   --ledger`, #362), not a test's.
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4
   at one batch and, where the plan admits it, three (the lane-parallel
   atomic form through #345's `atomize` seam; the kernels launched in order
   onto one zeroed `K`), with the lever, through test_kernel_device.py's K
   harness
   (`_run_k`: its inputs, its host staging, its pin `K_REL = 1e-12` of the
   quartet scale, unchanged), which builds the cell from
   `measured.cell_kernels` and so compiles the lever's kernel as
   `<name>_lv_coalesced`. Nothing is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_coalesced
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from test_f_lever_batched_vrr_device import compile_unit

from felizjueves import measured
from felizjueves.kernel import k_spec
from felizjueves.kernel_k import build_k_kernels
from felizjueves.spec import Operator, Spec, Target

COALESCED = "coalesced"


def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# The default K class of each L = 1..4, and two with a d shell (the largest
# K[a, b] blocks within the cap: 36 and 18 elements), one batch each; and two
# classes at three batches, the atomic form.
COMPILE_CELLS = ([(k_spec(L).am, 1) for L in range(1, 5)]
                 + [((2, 0, 2, 0), 1), ((2, 1, 1, 0), 1),
                    ((1, 0, 1, 0), 3), ((1, 1, 1, 1), 3)])
COMPILE_PARAMS = [pytest.param(am, n, id=f"{_id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _lever(am, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K, (COALESCED,),
                n_batches)


# ---------------------------------------------------------------------------
# 1. Compile-only: literal against the lever
# ---------------------------------------------------------------------------

def _compiles(backend: str, compiler: str, am, n: int, tmp_path: Path
              ) -> None:
    literal = build_k_kernels(dataclasses.replace(_lever(am, n), levers=()))
    lever = build_k_kernels(_lever(am, n))
    assert [k.name for k in lever] == [k.name for k in literal]
    for lk, vk in zip(literal, lever, strict=True):
        compile_unit(backend, compiler, lk, tmp_path / "literal")
        compile_unit(backend, compiler, dataclasses.replace(
            vk, name=f"{vk.name}_lv_{COALESCED}"), tmp_path / "lever")


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_coalesced_kernels_compile(am, n, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_coalesced_kernels_compile(am, n, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the coalesced kernels against the host staging
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    for L in range(5):
        for b in (1, 3):
            spec = dataclasses.replace(k_spec(L, n_batches=b),
                                       levers=(COALESCED,))
            try:
                build_k_kernels(spec)
            except ValueError:
                continue  # L = 0 at 3 batches: an empty batch, refused
            out.append(pytest.param(spec, id=f"L{L}-b{b}-k-coalesced"))
    return out


DEVICE_CELLS = _device_cells()


def test_the_driver_builds_a_coalesced_cell_as_build_k_kernels() -> None:
    for spec in (k_spec(2), k_spec(4, n_batches=3)):
        spec = dataclasses.replace(spec, levers=(COALESCED,))
        want = build_k_kernels(spec)
        got = measured.cell_kernels(spec)
        assert [k.name for k in got] == [f"{k.name}_lv_{COALESCED}"
                                         for k in want]
        assert [dataclasses.replace(k, name=w.name)
                for k, w in zip(got, want, strict=True)] == list(want)


def _run(backend: str, spec: Spec, request, compile_device_source,
         run_device_binary, tmp_path) -> None:
    import test_kernel_device as tkd

    tkd._run_k(backend, spec, request, compile_device_source,
               run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_coalesced_matches_the_host(spec, request,
                                                 compile_device_source,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """The coalesced K kernel on the RTX 3080, against the host."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_coalesced_matches_the_host(spec, request,
                                                compile_device_source,
                                                run_device_binary,
                                                tmp_path) -> None:
    """The coalesced K kernel on the RX 9060 XT, against the host."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [(s.values[0].total_am, s.values[0].n_batches)
            for s in DEVICE_CELLS] == [(0, 1), (1, 1), (1, 3), (2, 1), (2, 3),
                                       (3, 1), (3, 3), (4, 1), (4, 3)]
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)
