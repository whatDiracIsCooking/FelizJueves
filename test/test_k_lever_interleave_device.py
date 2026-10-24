"""Interleaved `I_p`, warp arm (#343): through the device compilers, on the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): every K class of a set with L <=
   4 (milestone 13's cap), the literal kernel and the interleaved one, each
   its own translation unit, compiled with the tier's flags
   (test_f_lever_batched_vrr_device.py's `compile_unit`). A warning in
   generated text fails (#92's rule). Nothing is read off the compiler:
   registers and spills are the lever ledger's (`measured --remeasure ...
   --ledger`, #362), not a test's. (#343's guardrail stands: the predecessor
   measured this arm as a regression, and a regression is still a pass.)
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4,
   at one batch and, where the plan admits it, three, with the interleave
   lever: the cell's kernels launched in order onto one zeroed `K` by
   test_kernel_device.py's K harness (`_run_k`: its inputs, its host staging
   `ao.k._k_quartet` on `reference.pq`, its pin `K_REL = 1e-12` of the
   quartet scale, unchanged). The harness builds a cell from
   `measured.cell_kernels`, which refuses a K lever (#205's store has no
   lever rows); here it is pointed at `build_k_kernels` for the test's
   duration, as test_k_lever_split_device.py does. Nothing is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_interleave
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_f_lever_batched_vrr_device import compile_unit

from felizjueves.eri.k_exchange.kernel_k import build_k_kernels
from felizjueves.ir import Kernel
from felizjueves.mmd.kernel import k_spec
from felizjueves.performance import measured
from felizjueves.spec import Operator, Spec, Target

INTERLEAVE = "interleave"


def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# The compile set, (am, n_batches): the default K class of each L = 1..4 and
# the classes through L = 4 with a d shell or four p shells, at one batch; and
# two classes at three batches, the atomized form. Not narrowed to the
# classes that improve (#343's guardrail).
COMPILE_CELLS = ([(k_spec(L).am, 1) for L in range(1, 5)]
                 + [((2, 0, 2, 0), 1), ((2, 1, 1, 0), 1),
                    ((2, 2, 0, 0), 1), ((0, 0, 2, 2), 1), ((0, 2, 1, 1), 1),
                    ((1, 0, 1, 0), 3), ((1, 1, 1, 1), 3)])
COMPILE_PARAMS = [pytest.param(am, n, id=f"{_id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _spec(am, n_batches: int = 1, levers: tuple[str, ...] = (INTERLEAVE,)
          ) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K, levers, n_batches)


# ---------------------------------------------------------------------------
# 1. Compile-only: literal and interleave, signed
# ---------------------------------------------------------------------------

def _compiles(backend: str, compiler: str, am, n: int, tmp_path: Path
              ) -> None:
    literal = build_k_kernels(_spec(am, n, ()))
    lever = build_k_kernels(_spec(am, n))
    assert [k.name for k in literal] == [k.name for k in lever]
    jobs = ([(k, tmp_path / "literal") for k in literal]
            + [(k, tmp_path / "interleave") for k in lever])
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda kw: compile_unit(backend, compiler, *kw), jobs))


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_interleave_kernels_compile(am, n, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_interleave_kernels_compile(am, n, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the interleaved kernels against the host staging
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    for L in range(5):
        for b in (1, 3):
            spec = dataclasses.replace(k_spec(L, n_batches=b),
                                       levers=(INTERLEAVE,))
            try:
                build_k_kernels(spec)
            except ValueError:
                continue  # L = 0 at 3 batches: an empty batch, refused
            out.append(pytest.param(spec, id=f"L{L}-b{b}-k-interleave"))
    return out


DEVICE_CELLS = _device_cells()


def _cell_kernels(spec: Spec) -> tuple[Kernel, ...]:
    if spec.target is Target.K and spec.levers == (INTERLEAVE,):
        return build_k_kernels(spec)
    raise AssertionError(f"{spec.key()}: not an interleave cell")


def _run(backend: str, spec: Spec, request, compile_device_source,
         run_device_binary, tmp_path, monkeypatch) -> None:
    import test_kernel_device as tkd

    monkeypatch.setattr(measured, "cell_kernels", _cell_kernels)
    tkd._run_k(backend, spec, request, compile_device_source,
               run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_interleave_matches_the_host(spec, request,
                                                  compile_device_source,
                                                  run_device_binary, tmp_path,
                                                  monkeypatch) -> None:
    """The interleave cell's kernels on the RTX 3080, against the host."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path, monkeypatch)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_interleave_matches_the_host(spec, request,
                                                 compile_device_source,
                                                 run_device_binary, tmp_path,
                                                 monkeypatch) -> None:
    """The interleave cell's kernels on the RX 9060 XT, against the host."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path, monkeypatch)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [s.values[0].total_am for s in DEVICE_CELLS] == [0, 1, 1, 2, 2, 3,
                                                            3, 4, 4]
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)
