"""The per-element split K lever (#341) through the device compilers and on the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): every element kernel of a set of
   K classes with L <= 4 (milestone 13's cap), each its own translation
   unit, beside the class's literal block kernel, compiled with the tier's
   flags (test_f_lever_batched_vrr_device.py's `compile_unit`). A warning in
   generated text fails (#92's rule). Nothing is read off the compiler:
   registers and spills are the lever ledger's (`measured --remeasure ...
   --ledger`, #362), not a test's. (The spilling classes are #341's X4's.)
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4,
   at one batch and, where the plan admits it, three, with the split lever:
   the element kernels launched in order onto one zeroed `K` by
   test_kernel_device.py's K harness (`_run_k`: its inputs, its host
   staging `ao.k._k_quartet` on `reference.pq`, its pin `K_REL = 1e-12` of
   the quartet scale, unchanged). The harness builds a cell from
   `measured.cell_kernels`, which since #339 passes a kernel-body K lever to
   `build_k_kernels` (renaming each unit `<name>_lv_split`), so the lever's
   kernels are what is compiled and run. Nothing is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_split
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_f_lever_batched_vrr_device import compile_unit

from felizjueves import measured
from felizjueves.kernel_k import build_k_kernel, build_k_kernels
from felizjueves.mmd.kernel import k_spec
from felizjueves.spec import Operator, Spec, Target

SPLIT = "split"

def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# The compile set, (am, n_batches): the default K class of each L = 1..4 and
# two classes with a d shell (the element count is largest within the cap),
# at one batch; and two classes at three batches, the atomized form.
COMPILE_CELLS = ([(k_spec(L).am, 1) for L in range(1, 5)]
                 + [((2, 0, 2, 0), 1), ((2, 1, 1, 0), 1),
                    ((1, 0, 1, 0), 3), ((1, 1, 1, 1), 3)])
COMPILE_PARAMS = [pytest.param(am, n, id=f"{_id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _lever(am, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K, (SPLIT,),
                n_batches)


# ---------------------------------------------------------------------------
# 1. Compile-only: the literal block and every element kernel
# ---------------------------------------------------------------------------

def _compiles(backend: str, compiler: str, am, n: int, tmp_path: Path
              ) -> None:
    literal = (build_k_kernel(tuple(am)),) if n == 1 else build_k_kernels(
        Spec(Operator.COULOMB, 0, tuple(am), Target.K, (), n))
    kernels = (*literal, *build_k_kernels(_lever(am, n)))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda k: compile_unit(backend, compiler, k, tmp_path),
                      kernels))


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_split_kernels_compile(am, n, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_split_kernels_compile(am, n, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the element kernels against the host staging
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    for L in range(5):
        for b in (1, 3):
            spec = dataclasses.replace(k_spec(L, n_batches=b), levers=(SPLIT,))
            try:
                build_k_kernels(spec)
            except ValueError:
                continue  # L = 0 at 3 batches: an empty batch, refused
            out.append(pytest.param(spec, id=f"L{L}-b{b}-k-split"))
    return out


DEVICE_CELLS = _device_cells()


def test_the_driver_builds_a_split_cell_as_build_k_kernels() -> None:
    # #339: `measured.cell_kernels` passes a kernel-body K lever to the
    # builder, so the harness compiles the lever's kernels as they are --
    # only renamed `<name>_lv_split`, the generator's lever-unit names.
    for spec in (k_spec(2), k_spec(4, n_batches=3)):
        spec = dataclasses.replace(spec, levers=(SPLIT,))
        want = build_k_kernels(spec)
        got = measured.cell_kernels(spec)
        assert [k.name for k in got] == [f"{k.name}_lv_split" for k in want]
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
def test_cuda_k_lever_split_matches_the_host(spec, request,
                                             compile_device_source,
                                             run_device_binary,
                                             tmp_path) -> None:
    """The split cell's element kernels on the RTX 3080, against the host."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_split_matches_the_host(spec, request,
                                            compile_device_source,
                                            run_device_binary,
                                            tmp_path) -> None:
    """The split cell's element kernels on the RX 9060 XT, against the host."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [s.values[0].total_am for s in DEVICE_CELLS] == [0, 1, 1, 2, 2, 3,
                                                            3, 4, 4]
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)
