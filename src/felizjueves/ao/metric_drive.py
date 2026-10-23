"""`(P|Q)` from the INTERPRETED GENERATED metric kernel (#414).

    metric_kernel(lP, lQ)                         -> Kernel (memoised)
    launch_class(cls, M, *, boys, threads=32)     -> M with cls's elements
    metric_matrix_kernel(aux, *, boys, classes=None, threads=32) -> Run

The `ao/drive.py` pattern for `Target.METRIC`: `kernel_metric.
build_metric_kernel((lP, lQ))` run in `interp.Interpreter` over
`ao.metric.metric_classes`' staging -- one launch per class, every staged
aux shell pair one thread, the class's record and pair arrays passed by
name (`MetricClass.arrays()`), into ONE zeroed `n_aux x n_aux` matrix. The
kernel's plain mirrored stores write every element of the launched pairs
once, so a class subset leaves the other elements 0.0 and the caller sees
which classes ran (`Run.ran`).

BOYS IS INJECTED (`boys=`), as the interpreter requires; nothing here imports
`reference/`. Kernels are memoised per class (building is pure); results are
not. Its own layering rule (test/test_layering.py, `metric_adapter`): the
builder, `interp`, `ir` and `ao`, never `reference/` nor a backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache

import numpy as np

from felizjueves.ao.metric import MetricClass, metric_classes
from felizjueves.ao.shell import Basis
from felizjueves.ir import Kernel
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.kernel_metric import (
    OUT,
    PARAM_OF,
    build_metric_kernel,
    launch_shape,
)

Boys = Callable[[int, float], float]


@cache
def metric_kernel(lP: int, lQ: int) -> Kernel:
    """`build_metric_kernel((lP, lQ))`, built once."""
    return build_metric_kernel((lP, lQ))


def _arrays(cls: MetricClass) -> dict[str, np.ndarray]:
    return {PARAM_OF.get(k, k): np.array(
        v, dtype=np.int32 if k in cls.PAIR_ARRAYS else float)
        for k, v in cls.arrays().items()}


def launch_class(cls: MetricClass, M: np.ndarray, *, boys: Boys,
                 threads: int = 32) -> np.ndarray:
    """One interpreted launch of `cls`'s kernel over every staged pair,
    writing into (a copy of) `M`, which is returned."""
    n = M.shape[0]
    grid, block = launch_shape(cls.n_pair, threads)
    return Interpreter(metric_kernel(cls.lP, cls.lQ),
                       launch=Launch(grid, block, 1), boys=boys).run(
        n_pair=cls.n_pair, n_rec=cls.n_rec, n_aux=n, **{OUT: M},
        **_arrays(cls))[OUT]


@dataclass(frozen=True)
class Run:
    """The matrix, and the classes `(lP, lQ)` whose kernels wrote it."""

    M: np.ndarray
    ran: tuple[tuple[int, int], ...]


def metric_matrix_kernel(aux: Basis, *, boys: Boys,
                         classes: Iterable[tuple[int, int]] | None = None,
                         threads: int = 32) -> Run:
    """`(P|Q)` over `aux` from the interpreted kernels of `classes` (every
    class of `aux` when None); the elements of any other class are 0.0."""
    want = None if classes is None else set(classes)
    M = np.zeros((aux.n_ao, aux.n_ao))
    ran = []
    for cls in metric_classes(aux):
        if want is None or (cls.lP, cls.lQ) in want:
            M = launch_class(cls, M, boys=boys, threads=threads)
            ran.append((cls.lP, cls.lQ))
    return Run(M, tuple(ran))
