"""`(P|Q)` from the INTERPRETED GENERATED metric kernel (#414).

    metric_kernel(lP, lQ)                         -> Kernel (memoised)
    launch_class(cls, M, *, boys, threads=32)     -> M with cls's elements
    metric_matrix_kernel(aux, *, boys, classes=None, threads=32) -> Run

The `eri/j_coulomb/drive.py` pattern for `Target.METRIC`: `kernel_metric.
build_metric_kernel((lP, lQ))` run in `interp.Interpreter` over
`eri.density_fit.metric.metric_classes`' staging -- one launch per class, every staged
aux shell pair one thread, the class's record and pair arrays passed by
name (`MetricClass.arrays()`), into ONE zeroed `n_aux x n_aux` matrix. The
kernel's plain mirrored stores write every element of the launched pairs
once, so a class subset leaves the other elements 0.0 and the caller sees
which classes ran (`Run.ran`).

`engine=Engine.HGP` runs `kernel_hgp.build_hmetric_kernel` instead (the HGP
twin, `docs/implementation/hgp_df.md`): the same parameters, launch and
output, so nothing else here changes. `engine=Engine.RYS` runs
`kernel_rmetric.build_rmetric_kernel` (the Rys twin, the same again), and
then the Rys rule is injected too (`rys=`, the interpreter's own hook).

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

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.kernel_hgp import build_hmetric_kernel
from felizjueves.eri.density_fit.kernel_metric import (
    OUT,
    PARAM_OF,
    build_metric_kernel,
    launch_shape,
)
from felizjueves.eri.density_fit.kernel_rmetric import build_rmetric_kernel
from felizjueves.eri.density_fit.metric import MetricClass, metric_classes
from felizjueves.ir import Kernel
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.spec import Engine

Boys = Callable[[int, float], float]
RysRule = Callable[[int, float], tuple[tuple[float, ...], tuple[float, ...]]]


@cache
def metric_kernel(lP: int, lQ: int, engine: Engine = Engine.MMD,
                  lowstack: bool | None = None) -> Kernel:
    """`build_metric_kernel((lP, lQ))`, or on `Engine.HGP`
    `kernel_hgp.build_hmetric_kernel((lP, lQ))`, on `Engine.RYS`
    `kernel_rmetric.build_rmetric_kernel((lP, lQ))` (the same parameters,
    launch and output), built once. `lowstack` forces the MMD kernel's form (None:
    the builder's default)."""
    if engine is Engine.HGP:
        return build_hmetric_kernel((lP, lQ))
    if engine is Engine.RYS:
        return build_rmetric_kernel((lP, lQ))
    if engine is not Engine.MMD:
        raise ValueError(f"engine {engine!r} is not an Engine")
    return build_metric_kernel((lP, lQ), lowstack=lowstack)


def _arrays(cls: MetricClass) -> dict[str, np.ndarray]:
    return {PARAM_OF.get(k, k): np.array(
        v, dtype=np.int32 if k in cls.PAIR_ARRAYS else float)
        for k, v in cls.arrays().items()}


def launch_class(cls: MetricClass, M: np.ndarray, *, boys: Boys,
                 threads: int = 32, engine: Engine = Engine.MMD,
                 lowstack: bool | None = None,
                 rys: RysRule | None = None) -> np.ndarray:
    """One interpreted launch of `cls`'s kernel over every staged pair,
    writing into (a copy of) `M`, which is returned. `rys` is the Rys rule
    `(n, T) -> (nodes, weights)`, which `Engine.RYS` needs."""
    n = M.shape[0]
    grid, block = launch_shape(cls.n_pair, threads)
    return Interpreter(metric_kernel(cls.lP, cls.lQ, engine, lowstack),
                       launch=Launch(grid, block, 1), boys=boys,
                       rys=rys).run(
        n_pair=cls.n_pair, n_rec=cls.n_rec, n_aux=n, **{OUT: M},
        **_arrays(cls))[OUT]


@dataclass(frozen=True)
class Run:
    """The matrix, and the classes `(lP, lQ)` whose kernels wrote it."""

    M: np.ndarray
    ran: tuple[tuple[int, int], ...]


def metric_matrix_kernel(aux: Basis, *, boys: Boys,
                         classes: Iterable[tuple[int, int]] | None = None,
                         threads: int = 32, engine: Engine = Engine.MMD,
                         lowstack: bool | None = None,
                         rys: RysRule | None = None) -> Run:
    """`(P|Q)` over `aux` from the interpreted kernels of `classes` (every
    class of `aux` when None); the elements of any other class are 0.0."""
    want = None if classes is None else set(classes)
    M = np.zeros((aux.n_ao, aux.n_ao))
    ran = []
    for cls in metric_classes(aux):
        if want is None or (cls.lP, cls.lQ) in want:
            M = launch_class(cls, M, boys=boys, threads=threads,
                             engine=engine, lowstack=lowstack, rys=rys)
            ran.append((cls.lP, cls.lQ))
    return Run(M, tuple(ran))
