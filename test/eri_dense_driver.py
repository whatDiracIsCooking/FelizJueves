"""The dense J- and K-layout ERI matrices, a row block per launch, built ON A CARD.

    dense_steps(classes, n_ao, layout, m0, n_row) -> [Step]
    interpret(classes, n_ao, layout, m0, n_row, *, init) -> the row block
    build_on_card(classes, n_ao, layout, m0, n_row, backend, tc, workdir)
                                                     -> the row block
    vec_product(M, D)                                -> (M vec(D)) as n x n

The records and tasks are `eri.full_eri.dense.dense_classes`', the kernels
`kernel_dense.build_dense_kernel`'s, emitted by `emit/cuda.py` /
`emit/hip.py` unchanged into `workdir` -- harness, in `test/`, not a second
emitter.

ON THE SHARED RUNNER (#649): one runner per class's unit, one launch per
class, one thread per task, the `n_row x n_ao^2` block carried from class to
class from zero. Every element is owned by one task and written by a plain
store, so the chain runs `RUNS` times and every run must be bit-identical to
the first. The GEMV is the host's (`vec_product`): `J` and `K` are one
matrix-vector product each over a dense matrix, which is the BLAS's job,
not a kernel of this family.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, build_steps, run_chain
from plan_driver import RUNS

from felizjueves.eri.full_eri import kernel_dense
from felizjueves.eri.full_eri.packed import EriClass
from felizjueves.ir import Intent, Kernel
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys

THREADS = 32  # lanes per block, one task per thread; not numerical

_INT = ("brec0", "brec1", "off_a", "off_b", "krec0", "krec1", "off_c",
        "off_d", "task_bra", "task_ket")


def _check_signature(kern: Kernel) -> None:
    from felizjueves.eri.full_eri import kernel_eri

    names = [p.name for p in kern.params]
    want = [*kernel_dense.SCALARS, *kernel_eri.RECORD_ARRAYS,
            *kernel_eri.INDEX_ARRAYS[:-1], kernel_dense.OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the staging "
                         f"wants {want}")


def _args(c: EriClass, n_ao: int, m0: int, n_row: int) -> dict:
    arr = {k: np.asarray(v, dtype=np.int32 if k in _INT else np.float64)
           for k, v in c.arrays().items()}
    return {"n_task": c.n_task, "n_brec": c.bra.n_rec, "n_krec": c.ket.n_rec,
            "n_bpair": c.bra.n_pair, "n_kpair": c.ket.n_pair, "n_ao": n_ao,
            "m0": m0, "n_row": n_row, "n_col": n_ao * n_ao, **arr}


def dense_steps(classes: Sequence[EriClass], n_ao: int, layout: str,
                m0: int, n_row: int) -> list[Step]:
    """One launch per class in `classes`, every argument but `eri`."""
    steps = []
    for c in classes:
        kern = kernel_dense.build_dense_kernel(c.am, layout)
        _check_signature(kern)
        grid, block = kernel_dense.launch_shape(c.n_task, THREADS)
        steps.append(Step(kern, grid, block, _args(c, n_ao, m0, n_row)))
    return steps


def interpret(classes: Sequence[EriClass], n_ao: int, layout: str, m0: int,
              n_row: int, *, init: np.ndarray, threads: int = 3
              ) -> np.ndarray:
    """The row block from the interpreted kernels of `classes`, onto
    `init`."""
    out = init
    for c in classes:
        grid, block = kernel_dense.launch_shape(c.n_task, threads)
        out = Interpreter(kernel_dense.build_dense_kernel(c.am, layout),
                          launch=Launch(grid, block, 1), boys=boys).run(
            **_args(c, n_ao, m0, n_row), eri=out)[kernel_dense.OUT]
    return out


def build_on_card(classes: Sequence[EriClass], n_ao: int, layout: str,
                  m0: int, n_row: int, backend: str, tc: Toolchain,
                  workdir: Path) -> np.ndarray:
    """The row block of `classes` built on the card, the chain run `RUNS`
    times bit-identically. The caller holds the card (its device fixture)."""
    steps = dense_steps(classes, n_ao, layout, m0, n_row)
    binaries = build_steps(tc, backend, steps, workdir)
    outs = run_chain(binaries, steps, kernel_dense.OUT,
                     np.zeros((n_row, n_ao * n_ao)), workdir, runs=RUNS)
    assert all(np.array_equal(outs[0], x) for x in outs[1:]), "runs differ"
    return outs[0]


def vec_product(M: np.ndarray, D) -> np.ndarray:
    """`(M vec(D))` reshaped `n x n`: `J` over `M_J`, `K` over `M_K`."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    return (M @ D.reshape(n * n)).reshape(n, n)
