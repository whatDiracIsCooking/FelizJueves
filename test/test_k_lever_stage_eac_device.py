"""`E^ac` staged in shared memory (#342) through the device compilers and on the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): a set of K classes with L <= 4
   (milestone 13's cap), the literal kernel and the lever's kernel each its
   own translation unit, compiled with the tier's flags and the resource
   report (test_f_lever_batched_vrr_device.py's `shared_bytes`). A warning
   in generated text fails (#92's rule). One reading is asserted, because it
   is what a launch must budget for, not a figure of merit: the static
   shared bytes (nvcc's `smem`, hipcc's LDS) -- the literal kernel has none,
   and the lever's are at most `stage_eac.staged_bytes(am)`, the declared
   footprint, on both compilers. AT MOST, not equal: nvcc may drop staged
   entries nothing reads, and -- the IR having no barrier, so nothing orders
   one lane's stores before another's loads -- a compiler may forward a
   lane's own stores and promote the block back to registers (hipcc does at
   the smallest classes, LDS 0). Nothing else is read off the compiler:
   registers and spills are the lever ledger's (`measured --remeasure ... --ledger`,
   #362), not a test's.
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4,
   at one batch and, where the plan admits it, three, plus `(dd|ss)` (the
   largest `E^ac` block within the cap), with the lever, launched by
   test_kernel_device.py's K harness (`_run_k`: its inputs, its host staging
   `ao.k._k_quartet` on `reference.pq`, its pin `K_REL = 1e-12` of the
   quartet scale, unchanged), at the harness's full warp -- so the
   barrier-free, identical-value staging is exercised by every lane of a
   real warp. `measured.cell_kernels` passes a kernel-body K lever to
   `build_k_kernels` (#339), renaming each unit `<name>_lv_stage_eac`.
   Nothing is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2 --klever-stage-eac-k 0 1 2 3 4" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_stage_eac
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from test_f_lever_batched_vrr_device import shared_bytes

from felizjueves import measured
from felizjueves.k_levers import stage_eac
from felizjueves.kernel import k_spec
from felizjueves.kernel_k import build_k_kernels
from felizjueves.spec import Operator, Spec, Target

NAME = stage_eac.NAME


def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# The compile set, (am, n_batches): the default K class of each L = 1..4,
# the classes with a d shell on the bra (the E^ac block is largest there),
# at one batch; and two classes at three batches, the atomized form.
COMPILE_CELLS = ([(k_spec(L).am, 1) for L in range(1, 5)]
                 + [((2, 0, 2, 0), 1), ((2, 1, 1, 0), 1), ((2, 2, 0, 0), 1),
                    ((1, 0, 1, 0), 3), ((1, 1, 1, 1), 3)])
COMPILE_PARAMS = [pytest.param(am, n, id=f"{_id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _spec(am, n_batches: int = 1, lever: bool = True) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K,
                (NAME,) if lever else (), n_batches)


# ---------------------------------------------------------------------------
# 1. Compile-only: literal and lever, and the staged block's shared bytes
# ---------------------------------------------------------------------------

def _compiles(backend: str, compiler: str, am, n: int, tmp_path: Path
              ) -> None:
    literal = build_k_kernels(_spec(am, n, lever=False))
    lever = build_k_kernels(_spec(am, n))
    assert [k.name for k in literal] == [k.name for k in lever]
    declared = stage_eac.staged_bytes(tuple(am))
    for a, b in zip(literal, lever, strict=True):
        lit = shared_bytes(backend, compiler, a, tmp_path / "literal")
        lev = shared_bytes(backend, compiler, b, tmp_path / "lever")
        # the literal kernel has no static shared (Ksh is dynamic, reported
        # as 0), and the declared footprint BOUNDS what the compiler
        # allocates for the staged block: it may drop entries nothing reads
        # (nvcc) or, with no barrier to stop it, promote the whole block
        # back to registers (hipcc, small classes) -- never exceed it
        assert lit == 0, (a.name, lit)
        assert 0 <= lev <= declared, (b.name, lev, declared)


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_stage_eac_kernels_compile(am, n, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_stage_eac_kernels_compile(am, n, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the lever's kernels against the host staging
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    for L in range(5):
        for b in (1, 3):
            spec = dataclasses.replace(k_spec(L, n_batches=b), levers=(NAME,))
            try:
                build_k_kernels(spec)
            except ValueError:
                continue  # L = 0 at 3 batches: an empty batch, refused
            out.append(pytest.param(spec, id=f"L{L}-b{b}-k-stage-eac"))
    # Only k_spec(L)'s classes: the shared K harness (`_k_check`) builds its
    # inputs from `total_am` alone, as k_spec(L)'s class, so another class at
    # the same L -- (dd|ss) at L = 4 -- gets an input of the wrong length and
    # the harness refuses it ("size mismatch") before any launch. (dd|ss)
    # stays a compile cell above.
    return out


DEVICE_CELLS = _device_cells()


def test_the_driver_builds_a_stage_eac_cell_as_build_k_kernels() -> None:
    for spec in (k_spec(2), k_spec(4, n_batches=3)):
        spec = dataclasses.replace(spec, levers=(NAME,))
        want = build_k_kernels(spec)
        got = measured.cell_kernels(spec)
        assert [k.name for k in got] == [f"{k.name}_lv_stage_eac"
                                         for k in want]
        assert [dataclasses.replace(k, name=w.name)
                for k, w in zip(got, want, strict=True)] == list(want)
        # the harness sizes the dynamic Ksh only; the staged block is static
        assert ([measured.jsh_slots(k) for k in got]
                == [measured.jsh_slots(k) for k in build_k_kernels(
                    dataclasses.replace(spec, levers=()))])


def _run(backend: str, spec: Spec, request, compile_device_source,
         run_device_binary, tmp_path) -> None:
    import test_kernel_device as tkd

    tkd._run_k(backend, spec, request, compile_device_source,
               run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_stage_eac_matches_the_host(spec, request,
                                                 compile_device_source,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """The lever's kernels on the RTX 3080, against the host."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_stage_eac_matches_the_host(spec, request,
                                                compile_device_source,
                                                run_device_binary,
                                                tmp_path) -> None:
    """The lever's kernels on the RX 9060 XT, against the host."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [s.values[0].total_am for s in DEVICE_CELLS] == [0, 1, 1, 2, 2, 3,
                                                            3, 4, 4]
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)
