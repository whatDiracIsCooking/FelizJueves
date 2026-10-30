"""`(mn|P)` from the INTERPRETED GENERATED three-center kernel (#444).

    eri3_kernel(la, lb, lP)                        -> Kernel (memoised)
    launch_class(cls, out, *, boys, threads=32)    -> out with cls's elements
    eri3_tensor_kernel(basis, aux, *, boys, classes=None, threads=32) -> Run

The `eri/density_fit/metric_drive.py` pattern for `Target.ERI3`: `kernel_eri3.
build_eri3_kernel((la, lb, lP))` run in `interp.Interpreter` over
`eri.density_fit.eri3.eri3_classes`' staging -- one launch per class, every staged cell
(AO shell pair, aux shell) one thread, the class's record and cell arrays
passed by name (`Eri3Class.arrays()`), into ONE zeroed
`n_ao x n_ao x n_aux` tensor, `[m][n][P]`. The kernel's plain mirrored
stores write every element of the launched cells once, so a class subset
leaves the other elements 0.0 and the caller sees which classes ran
(`Run.ran`).

`engine=Engine.HGP` runs `kernel_hgp.build_heri3_kernel` instead (the HGP
twin, `docs/implementation/hgp_df.md`): the same parameters, launch and
output, so nothing else here changes. `engine=Engine.RYS` runs
`kernel_rys.build_reri3_kernel` (the Rys twin), the same again; its
`ir.Rys` rule is injected as `rys=` (`rys.roots.roots_weights`), as Boys is.

BOYS IS INJECTED (`boys=`), as the interpreter requires; nothing here imports
`reference/`. Kernels are memoised per class (building is pure); results are
not. Its own layering rule (test/test_layering.py, `eri3_adapter`): the
builder, `interp`, `ir` and `ao`, never `reference/` nor a backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache

import numpy as np

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.eri3 import Eri3Class, eri3_classes
from felizjueves.eri.density_fit.kernel_eri3 import (
    OUT,
    PARAM_OF,
    build_eri3_kernel,
    launch_shape,
)
from felizjueves.eri.density_fit.kernel_hgp import build_heri3_kernel
from felizjueves.eri.density_fit.kernel_rys import build_reri3_kernel
from felizjueves.ir import Kernel
from felizjueves.ir.interp import Interpreter, Launch, RysRule
from felizjueves.spec import Engine

Boys = Callable[[int, float], float]


@cache
def eri3_kernel(la: int, lb: int, lP: int, engine: Engine = Engine.MMD,
                lowstack: bool | None = None) -> Kernel:
    """`build_eri3_kernel((la, lb, lP))`, or on `Engine.HGP`
    `kernel_hgp.build_heri3_kernel((la, lb, lP))` (the same parameters, launch and
    output), and on `Engine.RYS` `kernel_rys.build_reri3_kernel`, built
    once. `lowstack` forces the MMD kernel's form (None: the builder's
    default)."""
    if engine is Engine.HGP:
        return build_heri3_kernel((la, lb, lP))
    if engine is Engine.RYS:
        return build_reri3_kernel((la, lb, lP))
    if engine is not Engine.MMD:
        raise ValueError(f"engine {engine!r} is not an Engine")
    return build_eri3_kernel((la, lb, lP), lowstack=lowstack)


def _arrays(cls: Eri3Class) -> dict[str, np.ndarray]:
    return {PARAM_OF.get(k, k): np.array(
        v, dtype=np.int32 if k in cls.CELL_ARRAYS else float)
        for k, v in cls.arrays().items()}


def launch_class(cls: Eri3Class, out: np.ndarray, *, boys: Boys,
                 threads: int = 32, engine: Engine = Engine.MMD,
                 lowstack: bool | None = None,
                 rys: RysRule | None = None) -> np.ndarray:
    """One interpreted launch of `cls`'s kernel over every staged cell,
    writing into (a copy of) `out`, `(n_ao, n_ao, n_aux)`, which is
    returned."""
    n, _, naux = out.shape
    grid, block = launch_shape(cls.n_cell, threads)
    return Interpreter(eri3_kernel(cls.la, cls.lb, cls.lP, engine, lowstack),
                       launch=Launch(grid, block, 1), boys=boys,
                       rys=rys).run(
        n_cell=cls.n_cell, n_bra_rec=cls.n_bra_rec, n_aux_rec=cls.n_aux_rec,
        n_ao=n, n_aux=naux, **{OUT: out}, **_arrays(cls))[OUT]


@dataclass(frozen=True)
class Run:
    """The tensor, and the classes `(la, lb, lP)` whose kernels wrote it."""

    T: np.ndarray
    ran: tuple[tuple[int, int, int], ...]


def eri3_tensor_kernel(basis: Basis, aux: Basis, *, boys: Boys,
                       classes: Iterable[tuple[int, int, int]] | None = None,
                       threads: int = 32, engine: Engine = Engine.MMD,
                       lowstack: bool | None = None,
                       rys: RysRule | None = None) -> Run:
    """`(mn|P)` over `basis` and `aux` from the interpreted kernels of
    `classes` (every class the two bases reach when None); the elements of
    any other class are 0.0."""
    want = None if classes is None else set(classes)
    T = np.zeros((basis.n_ao, basis.n_ao, aux.n_ao))
    ran = []
    for cls in eri3_classes(basis, aux):
        if want is None or (cls.la, cls.lb, cls.lP) in want:
            T = launch_class(cls, T, boys=boys, threads=threads,
                             engine=engine, lowstack=lowstack, rys=rys)
            ran.append((cls.la, cls.lb, cls.lP))
    return Run(T, tuple(ran))
