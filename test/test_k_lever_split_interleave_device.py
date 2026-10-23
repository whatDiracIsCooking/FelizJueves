"""The split arm (#348), interleave+split: through the device compilers, on the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): the FOUR ARMS of a set of K
   classes with L <= 4 (milestone 13's cap) -- literal, interleave, split,
   split+interleave -- every kernel its own translation unit, compiled with
   the tier's flags (test_f_lever_batched_vrr_device.py's `compile_unit`). A
   warning in generated text fails (#92's rule). The arms' kernel names are
   asserted to pair up (every element kernel `e` fanned out of one literal
   batch kernel `b`, interleave's batches the literal's). Nothing is read
   off the compiler: registers and spills are the lever ledger's (`measured
   --remeasure ... --ledger`, #362), not a test's. The composition's
   interaction, `dB - (dS + dI)` against the literal batch kernel, is a
   ledger reading too.
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4,
   at one batch and, where the plan admits it, three, with interleave+split:
   the element kernels launched in order onto one zeroed `K` by
   test_kernel_device.py's K harness (`_run_k`: its inputs, its host staging
   `ao.k._k_quartet` on `reference.pq`, its pin `K_REL = 1e-12` of the
   quartet scale, unchanged), built by `measured.cell_kernels`, which passes
   both kernel-body levers to `build_k_kernels` (#339; units renamed
   `<name>_lv_interleave_lv_split`). Nothing is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_split_interleave
"""

from __future__ import annotations

import dataclasses
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_f_lever_batched_vrr_device import compile_unit

from felizjueves.k_levers import interleave, split
from felizjueves.kernel_k import build_k_kernels
from felizjueves.mmd.kernel import k_spec
from felizjueves.performance import measured
from felizjueves.spec import Operator, Spec, Target

BOTH = tuple(sorted((interleave.NAME, split.NAME)))
ARMS = {"L": (), "I": (interleave.NAME,), "S": (split.NAME,), "B": BOTH}


def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# The compile set, (am, n_batches): the default K class of each L = 1..4,
# two classes with a d shell, at one batch; two classes at three batches.
COMPILE_CELLS = ([(k_spec(L).am, 1) for L in range(1, 5)]
                 + [((2, 0, 2, 0), 1), ((2, 1, 1, 0), 1),
                    ((1, 0, 1, 0), 3), ((1, 1, 1, 1), 3)])
COMPILE_PARAMS = [pytest.param(am, n, id=f"{_id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _spec(am, n_batches: int, levers: tuple[str, ...]) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K, levers, n_batches)


# ---------------------------------------------------------------------------
# 1. Compile-only: the four arms
# ---------------------------------------------------------------------------

def _parent(name: str, batch_names: list[str]) -> str:
    """The batch kernel an element kernel `<batch>_e{i}_{j}` was fanned out of."""
    (p,) = [b for b in batch_names
            if re.fullmatch(re.escape(b) + r"_e\d+_\d+", name)]
    return p


def _compiles(backend: str, compiler: str, am, n: int, tmp_path: Path
              ) -> None:
    built = {a: build_k_kernels(_spec(am, n, lv)) for a, lv in ARMS.items()}
    assert [k.name for k in built["S"]] == [k.name for k in built["B"]]
    assert [k.name for k in built["L"]] == [k.name for k in built["I"]]
    batches = [k.name for k in built["L"]]
    assert all(_parent(k.name, batches) for k in built["B"])
    jobs = [(a, k) for a, ks in built.items() for k in ks]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda job: compile_unit(
            backend, compiler, job[1], tmp_path / job[0]), jobs))


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_split_interleave_arms_compile(
        am, n, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_split_interleave_arms_compile(
        am, n, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the composed element kernels against the host staging
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    for L in range(5):
        for b in (1, 3):
            spec = dataclasses.replace(k_spec(L, n_batches=b), levers=BOTH)
            try:
                build_k_kernels(spec)
            except ValueError:
                continue  # L = 0 at 3 batches: an empty batch, refused
            out.append(pytest.param(spec, id=f"L{L}-b{b}-k-split-interleave"))
    return out


DEVICE_CELLS = _device_cells()


def test_the_driver_builds_a_composed_cell_as_build_k_kernels() -> None:
    suffix = "_lv_interleave_lv_split"
    for spec in (k_spec(2), k_spec(4, n_batches=3)):
        spec = dataclasses.replace(spec, levers=BOTH)
        want = build_k_kernels(spec)
        got = measured.cell_kernels(spec)
        assert [k.name for k in got] == [k.name + suffix for k in want]
        assert [dataclasses.replace(k, name=w.name)
                for k, w in zip(got, want, strict=True)] == list(want)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [s.values[0].total_am for s in DEVICE_CELLS] == [0, 1, 1, 2, 2, 3,
                                                            3, 4, 4]
    assert all(s.values[0].levers == BOTH for s in DEVICE_CELLS)
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)


def _run(backend: str, spec: Spec, request, compile_device_source,
         run_device_binary, tmp_path) -> None:
    import test_kernel_device as tkd

    tkd._run_k(backend, spec, request, compile_device_source,
               run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_split_interleave_matches_the_host(
        spec, request, compile_device_source, run_device_binary,
        tmp_path) -> None:
    """The composed cell's element kernels on the RTX 3080, against the host."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_split_interleave_matches_the_host(
        spec, request, compile_device_source, run_device_binary,
        tmp_path) -> None:
    """The composed cell's element kernels on the RX 9060 XT, against the host."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path)
