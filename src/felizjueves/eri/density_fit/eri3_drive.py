"""`(mn|P)` from the INTERPRETED GENERATED three-center kernel (#444).

    eri3_kernel(la, lb, lP)                        -> Kernel (memoised)
    launch_class(cls, out, *, boys, threads=32)    -> out with cls's elements
    eri3_tensor_kernel(basis, aux, *, boys, classes=None, threads=32) -> Run
    launch_packed(cls, cells, out, *, boys, threads=32) -> out, packed blocks
                    (spherical_aux=, spherical_bra=: speri3_ / ssperi3_)
    launch_bra_transform(la, lb, src, pairs, n_dst_col, *, threads=32)
                                                   -> sbra3_'s output (#747)
    cell_bounds(cls, basis, aux, *, pq, spherical_bra=True) -> per-cell bound
    precision_split(bounds, *, tau_screen, tau32)  -> {screened, fp32, fp64}
    subset_cells(cells, idx)                       -> those cells' arrays
    screened_slots(cls, cells, idx, n_aux, *, spherical_bra=True) -> flat slots
    launch_dynamic(cls, cells, out, bins, *, boys, ...) -> out (#754)

DYNAMIC PRECISION WITH SCREENING (#754; Luehr, Ufimtsev and Martinez, JCTC
7, 949 (2011)). The Coulomb kernel is positive definite, so per Cartesian
element `|(ab|q)| <= sqrt((ab|ab)) sqrt((q|q))` (Cauchy-Schwarz, as
`eri.schwarz`), and a spherical element `SUM c^{la} c^{lb} c^{lP} (ab|q)`
is bounded per cell by

    bound = max_ab sqrt((ab|ab)) max_q sqrt((q|q)) K_la K_lb K_lP,
    K_l = max_m SUM_q |aux_harmonics(l)[m][q]|          (1 for a Cartesian bra)

(`cell_bounds`, `(ab|ab)` from `eri.schwarz.diagonal_and_scale`, `(q|q)`
from `metric.metric_block`, each clamped at 0 before its root, as the
Schwarz matrix is). `precision_split` bins the cells: below `tau_screen`
screened (not launched; their slots set to 0.0, the error at most the
bound), below `tau32` the mixed-precision kernel (#753), else the FP64
one. Both kernels take any cell subset and write only its slots, so the
split is host-side and changes no kernel.

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
`kernel_reri3.build_reri3_kernel` (the Rys twin), the same again; its
`ir.Rys` rule is injected as `rys=` (`rys.roots.roots_weights`), as Boys is.

THE PACKED OUTPUT (`build_eri3_kernel(..., packed=True)`, `eri.density_fit.
eri3`'s module docstring): `launch_packed` runs it over the cell arrays
`eri.density_fit.eri3.packed_cells` formed (or any subset a caller forms),
into a flat output of `n_col n_aux` doubles.

BOYS IS INJECTED (`boys=`), as the interpreter requires; nothing here imports
`reference/`. Kernels are memoised per class (building is pure); results are
not. Its own layering rule (test/test_layering.py, `eri3_adapter`): the
builder, `interp`, `ir` and `ao`, never `reference/` nor a backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np

from felizjueves.ao.shell import Basis, cartesians
from felizjueves.eri.density_fit.eri3 import Eri3Class, eri3_classes
from felizjueves.eri.density_fit.kernel_eri3 import (
    BRA_SRC,
    OUT,
    PACKED_CELL_ARRAYS,
    PARAM_OF,
    aux_harmonics,
    build_bra_transform_kernel,
    build_eri3_kernel,
    build_warp_eri3_kernel,
    launch_shape,
    warp_launch_shape,
)
from felizjueves.eri.density_fit.kernel_hgp import build_heri3_kernel
from felizjueves.eri.density_fit.kernel_reri3 import build_reri3_kernel
from felizjueves.eri.density_fit.metric import metric_block
from felizjueves.eri.schwarz import diagonal_and_scale
from felizjueves.ir import Kernel
from felizjueves.ir.interp import Interpreter, Launch, RysRule
from felizjueves.spec import Engine

Boys = Callable[[int, float], float]


@cache
def eri3_kernel(la: int, lb: int, lP: int, engine: Engine = Engine.MMD,
                lowstack: bool | None = None) -> Kernel:
    """`build_eri3_kernel((la, lb, lP))`, or on `Engine.HGP`
    `kernel_hgp.build_heri3_kernel((la, lb, lP))` (the same parameters, launch and
    output), and on `Engine.RYS` `kernel_reri3.build_reri3_kernel`, built
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


@cache
def packed_kernel(la: int, lb: int, lP: int,
                  lowstack: bool | None = None,
                  spherical_aux: bool = False,
                  spherical_bra: bool = False,
                  rolled: bool = False, warp: bool = False,
                  mixed: bool = False) -> Kernel:
    """`build_eri3_kernel((la, lb, lP), packed=True)`, built once;
    `spherical_aux` its solid-harmonic aux mode, `spherical_bra` with it
    the fully spherical one (#745), and `rolled` either one's rolled form
    (#748); `warp` the warp-per-cell form of either (#749); `mixed` the
    rolled form's mixed-precision one."""
    if warp:
        if not spherical_aux or rolled or lowstack is False:
            raise ValueError("the warp form is a spherical-aux one")
        return build_warp_eri3_kernel((la, lb, lP),
                                      spherical_bra=spherical_bra)
    return build_eri3_kernel((la, lb, lP), lowstack=lowstack, packed=True,
                             spherical_aux=spherical_aux,
                             spherical_bra=spherical_bra, rolled=rolled,
                             mixed=mixed)


def launch_packed(cls: Eri3Class, cells: dict[str, Sequence[int]],
                  out: np.ndarray, *, boys: Boys, threads: int = 32,
                  lowstack: bool | None = None,
                  spherical_aux: bool = False,
                  spherical_bra: bool = False,
                  rolled: bool = False, warp: bool = False,
                  mixed: bool = False) -> np.ndarray:
    """One interpreted launch of `cls`'s packed kernel over the cells
    `cells` names (`PACKED_CELL_ARRAYS`, `packed_cells`' arrays), writing
    into (a copy of) `out`, `(n_col, n_aux)`, which is returned;
    `spherical_aux` launches the solid-harmonic aux mode, `n_aux` and `dst`
    in spherical aux columns; `spherical_bra` (with it) the fully spherical
    mode, `n_col` in spherical bra functions (#745); `rolled` the rolled
    form of either (#748); `warp` the warp-per-cell form (#749), a block
    of `WARP_LANES` per cell (`threads` unused)."""
    n_col, naux = out.shape
    n_cell = len(cells["dst"])
    grid, block = (warp_launch_shape(n_cell) if warp
                   else launch_shape(n_cell, threads))
    records = {PARAM_OF.get(k, k): np.array(v, dtype=float)
               for k, v in cls.arrays().items()
               if k in (*cls.BRA_ARRAYS, *cls.AUX_ARRAYS)}
    flat = Interpreter(packed_kernel(cls.la, cls.lb, cls.lP, lowstack,
                                     spherical_aux, spherical_bra, rolled,
                                     warp, mixed),
                       launch=Launch(grid, block, 1), boys=boys).run(
        n_cell=n_cell, n_bra_rec=cls.n_bra_rec, n_aux_rec=cls.n_aux_rec,
        n_col=n_col, n_aux=naux, **{OUT: out.reshape(-1)}, **records,
        **{k: np.array(cells[k], dtype=np.int32)
           for k in PACKED_CELL_ARRAYS})[OUT]
    return np.asarray(flat).reshape(n_col, naux)


@cache
def bra_transform_kernel(la: int, lb: int) -> Kernel:
    """`build_bra_transform_kernel(la, lb)`, built once."""
    return build_bra_transform_kernel(la, lb)


def launch_bra_transform(la: int, lb: int, src: np.ndarray,
                         pairs: dict[str, Sequence[int]], n_dst_col: int, *,
                         threads: int = 32) -> np.ndarray:
    """One interpreted launch of `sbra3_{la}{lb}` (#747) over the pairs
    `pairs` names (`eri3.bra_pairs`), reading `src`, `(n_src_col, n_aux)`,
    a spherical-aux packed output, into a NaN-filled `(n_dst_col, n_aux)`,
    which is returned."""
    n_src_col, naux = src.shape
    n_pair = len(pairs["src_col"])
    grid, block = launch_shape(n_pair * naux, threads)
    flat = Interpreter(bra_transform_kernel(la, lb),
                       launch=Launch(grid, block, 1)).run(
        n_pair=n_pair, n_aux=naux, n_src_col=n_src_col, n_dst_col=n_dst_col,
        **{k: np.array(pairs[k], dtype=np.int32)
           for k in ("src_col", "dst_col")},
        **{BRA_SRC: src.reshape(-1),
           OUT: np.full(n_dst_col * naux, np.nan)})[OUT]
    return np.asarray(flat).reshape(n_dst_col, naux)


# ---------------------------------------------------------------------------
# Dynamic precision with screening (#754)
# ---------------------------------------------------------------------------

BINS = ("screened", "fp32", "fp64")


def harmonic_norm(l: int) -> float:  # noqa: E741
    """`K_l = max_m SUM_q |aux_harmonics(l)[m][q]|`: how much a spherical
    combination can exceed the largest Cartesian element it combines."""
    return max(sum(abs(c) for c in col) for col in aux_harmonics(l))


def cell_bounds(cls: Eri3Class, basis: Basis, aux: Basis, *,
                pq: Callable, spherical_bra: bool = True
                ) -> tuple[float, ...]:
    """Each cell's bound on every element of its spherical-aux block (with
    `spherical_bra`, of its fully spherical block): the module docstring's
    Cauchy-Schwarz product, `pq` the injected `[t|tau]` evaluator."""
    k = harmonic_norm(cls.lP)
    if spherical_bra:
        k *= harmonic_norm(cls.la) * harmonic_norm(cls.lb)
    q_pair: dict[tuple[int, int], float] = {}
    q_aux: dict[int, float] = {}
    out = []
    for a, b, p in cls.cells:
        if (a, b) not in q_pair:
            diag, _ = diagonal_and_scale(basis.shells[a], basis.shells[b],
                                         pq=pq)
            q_pair[(a, b)] = math.sqrt(max(max(diag), 0.0))
        if p not in q_aux:
            blk = metric_block(aux.shells[p], aux.shells[p], pq=pq)
            q_aux[p] = math.sqrt(max(max(blk[i][i] for i in range(len(blk))),
                                     0.0))
        out.append(q_pair[(a, b)] * q_aux[p] * k)
    return tuple(out)


def precision_split(bounds: Sequence[float], *, tau_screen: float,
                    tau32: float) -> dict[str, tuple[int, ...]]:
    """The cells (indices into `bounds`) in each bin: `screened` below
    `tau_screen`, `fp32` below `tau32`, `fp64` the rest."""
    if not 0.0 <= tau_screen <= tau32:
        raise ValueError(f"need 0 <= tau_screen <= tau32, got {tau_screen}, "
                         f"{tau32}")
    bins: dict[str, list[int]] = {b: [] for b in BINS}
    for i, x in enumerate(bounds):
        bins["screened" if x < tau_screen else "fp32" if x < tau32
             else "fp64"].append(i)
    return {b: tuple(v) for b, v in bins.items()}


def subset_cells(cells: dict[str, Sequence[int]], idx: Sequence[int]
                 ) -> dict[str, tuple[int, ...]]:
    """The packed cell arrays of the cells `idx` only."""
    return {k: tuple(v[i] for i in idx) for k, v in cells.items()}


def screened_slots(cls: Eri3Class, cells: dict[str, Sequence[int]],
                   idx: Sequence[int], n_aux: int, *,
                   spherical_bra: bool = True) -> np.ndarray:
    """The flat output indices every slot of the cells `idx` owns."""
    nA, nB = ((2 * cls.la + 1, 2 * cls.lb + 1) if spherical_bra else
              (len(cartesians(cls.la)), len(cartesians(cls.lb))))
    block = (n_aux * np.arange(nA * nB)[:, None]
             + np.arange(2 * cls.lP + 1)[None, :]).reshape(-1)
    return np.array([cells["dst"][i] + x for i in idx for x in block],
                    dtype=np.int64)


def launch_dynamic(cls: Eri3Class, cells: dict[str, Sequence[int]],
                   out: np.ndarray, bins: dict[str, Sequence[int]], *,
                   boys: Boys, threads: int = 32,
                   spherical_bra: bool = True) -> np.ndarray:
    """`cls`'s spherical output by bins (`precision_split` over `cells`,
    every cell of `cls` laid out, in cell order): the FP64 kernel over
    `fp64`, the mixed one over `fp32`, `screened` slots set to 0.0, into a
    copy of `out`, `(n_col, n_aux)`, which is returned."""
    if len(cells["dst"]) != cls.n_cell:
        raise ValueError("launch_dynamic takes every cell of the class")
    n_aux = out.shape[1]
    out = out.copy()
    out.reshape(-1)[screened_slots(cls, cells, bins["screened"], n_aux,
                                   spherical_bra=spherical_bra)] = 0.0
    for b, mixed in (("fp64", False), ("fp32", True)):
        if bins[b]:
            out = launch_packed(cls, subset_cells(cells, bins[b]), out,
                                boys=boys, threads=threads,
                                spherical_aux=True,
                                spherical_bra=spherical_bra, mixed=mixed)
    return out
