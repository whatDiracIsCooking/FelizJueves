"""The three-center DF integral `(mn|P)` built ON A CARD -- harness.

    eri3_kernel(cls, engine, lowstack)                -> Kernel
    eri3_program(classes, n_ao, n_aux, backend, workdir) -> Program
    eri3_steps(classes, n_ao, n_aux, kernels)         -> (buffers, [Step])
    build_on_card(classes, n_ao, n_aux, backend, tc, workdir) -> T
    compile_timed / run_timed                         -> measure_df_lowstack's clock
    packed_layout(classes, n_aux)                     -> (cells per class, n_col)
    gather_packed(T, classes, cells, n_col, n_aux)    -> the square T's blocks, packed
    build_packed_on_card(classes, cells, n_col, n_aux, backend, tc, workdir) -> B

The records are `eri.density_fit.eri3.eri3_classes`' (every AO shell pair `i <= j`
staged `la <= lb`, swapped when the basis order has the higher `l` first,
crossed with every aux shell; per class one bra record table, one aux record
table, and two record ranges and three offsets per cell), the kernels
`kernel_eri3.build_eri3_kernel`'s (or `kernel_hgp`'s `build_heri3_kernel`,
`engine=HGP`, or `kernel_reri3`'s `build_reri3_kernel`, `engine=RYS`),
emitted by `emit/cuda.py` / `emit/hip.py` unchanged into `workdir` (the
kernel tier's `KERNEL_GEN_CMD` writes no eri3 unit).

ON THE SHARED RUNNER (#649, ported by #657): ONE runner over every class's
unit (`harness_driver.units_runner`, each unit in its own namespace),
compiled by `plan_driver.compile_checked` under #92's warning filter, and
ONE sequence: per class one launch, one thread per cell (`launch_shape`,
`THREADS` lanes), into one zeroed `n_ao x n_ao x n_aux` tensor (`[m][n][P]`,
aux fastest). Every array a launch reads is a buffer of its class
(`c<i>.<parameter>`), named from the kernel's own parameter list, so the
call cannot drift from the signature; the record array `s` is the kernel's
`s_aux` (`kernel_eri3.PARAM_OF`).

THE RUN-TO-RUN WITNESS: each element is owned by one cell and written by a
plain store (mirrored in `(m, n)`), so the sequence runs `RUNS` times and
the runner fails unless every run is bit-identical.

THE FORM (#606): `lowstack=` reaches `build_eri3_kernel` unchanged (None,
the default, is the gate's choice); `timeout=` bounds the runner's compile.

THE PACKED OUTPUT (FreeBreakfast#1680, `build_eri3_kernel(..., packed=
True)`): `build_packed_on_card` runs the packed units of `classes` over
`packed_layout`'s cells (`eri.density_fit.eri3.packed_columns` /
`packed_cells`: every pair once, class order, from column 0) into one
`n_col x n_aux` output, flat, that starts NaN, so an element no cell
stores is seen; `gather_packed` reads the same blocks out of the square
kernels' `T` for the bitwise comparison.

TIMED (#606, measure/measure_df_lowstack.py): one class, `warmup + reps`
runs of its one launch on the runner's clock, `{"0": [ns per timed run]}`
(`metric_driver.run_timed`'s shape).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, Zeros, build_runner, run_sequence, units_runner
from plan_driver import RUNS, Program, _unit

from felizjueves.ao.shell import cartesians
from felizjueves.eri.density_fit.eri3 import (
    Eri3Class,
    packed_cells,
    packed_columns,
)
from felizjueves.eri.density_fit.kernel_eri3 import (
    OUT,
    PACKED_CELL_ARRAYS,
    PARAM_OF,
    build_eri3_kernel,
    build_warp_eri3_kernel,
    launch_shape,
    warp_launch_shape,
)
from felizjueves.eri.density_fit.kernel_hgp import build_heri3_kernel
from felizjueves.eri.density_fit.kernel_reri3 import build_reri3_kernel
from felizjueves.ir import Intent, Kernel
from felizjueves.spec import Engine

THREADS = 32  # lanes per block, one cell per thread; not numerical
SIZES = ("n_cell", "n_bra_rec", "n_aux_rec", "n_ao", "n_aux")


def _param(name: str) -> str:
    return PARAM_OF.get(name, name)


def eri3_kernel(cls: Eri3Class, engine: Engine = Engine.MMD,
                lowstack: bool | None = None) -> Kernel:
    """The class's kernel, its signature checked against the records."""
    am = (cls.la, cls.lb, cls.lP)
    if engine is Engine.HGP:
        kern = build_heri3_kernel(am)
    elif engine is Engine.RYS:
        kern = build_reri3_kernel(am)
    else:
        kern = build_eri3_kernel(am, lowstack=lowstack)
    names = [p.name for p in kern.params]
    want = [*SIZES,
            *(_param(n) for n in (*Eri3Class.BRA_ARRAYS,
                                  *Eri3Class.AUX_ARRAYS)),
            *Eri3Class.CELL_ARRAYS, OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the records "
                         f"want {want}")
    return kern


def eri3_program(classes: Sequence[Eri3Class], n_ao: int, n_aux: int,
                 backend: str, workdir: Path, *,
                 engine: Engine = Engine.MMD, lowstack: bool | None = None,
                 name: str = "eri3_runner") -> Program:
    """The runner over every class's unit, emitted into `workdir`
    (`n_ao`, `n_aux` are the sequence's, not the runner's)."""
    del n_ao, n_aux
    workdir.mkdir(parents=True, exist_ok=True)
    units = []
    for cls in classes:
        kern = eri3_kernel(cls, engine, lowstack)
        units.append((_unit(kern, backend, workdir, None), [kern.name]))
    return units_runner(backend, units, workdir, name)


def eri3_steps(classes: Sequence[Eri3Class], n_ao: int, n_aux: int,
               kernels: Sequence[Kernel]) -> tuple[dict, list[Step]]:
    """The sequence's buffers and its launches, one per class, into `T`."""
    buffers: dict = {OUT: Zeros((n_ao, n_ao, n_aux))}
    steps = []
    for c, (cls, kern) in enumerate(zip(classes, kernels, strict=True)):
        arr = cls.arrays()
        args: dict = {"n_cell": cls.n_cell, "n_bra_rec": cls.n_bra_rec,
                      "n_aux_rec": cls.n_aux_rec, "n_ao": n_ao,
                      "n_aux": n_aux, OUT: OUT}
        for name in (*Eri3Class.BRA_ARRAYS, *Eri3Class.AUX_ARRAYS):
            buf = f"c{c}.{_param(name)}"
            buffers[buf] = np.asarray(arr[name], dtype=np.float64)
            args[_param(name)] = buf
        for name in Eri3Class.CELL_ARRAYS:
            buf = f"c{c}.{name}"
            buffers[buf] = np.asarray(arr[name], dtype=np.int32)
            args[name] = buf
        grid, block = launch_shape(cls.n_cell, THREADS)
        steps.append(Step(kern, grid, block, args))
    return buffers, steps


def build_on_card(classes: Sequence[Eri3Class], n_ao: int, n_aux: int,
                  backend: str, tc: Toolchain, workdir: Path, *,
                  engine: Engine = Engine.MMD, lowstack: bool | None = None,
                  timeout: float = 1800.0) -> np.ndarray:
    """`(mn|P)` of `classes` built on the card, `[n_ao, n_ao, n_aux]`, the
    sequence run `RUNS` times bit-identically. The caller holds the card
    (its device fixture)."""
    prog = eri3_program(classes, n_ao, n_aux, backend, workdir,
                        engine=engine, lowstack=lowstack)
    binary = build_runner(tc, prog, workdir / "eri3_bin", timeout=timeout)
    kernels = [eri3_kernel(c, engine, lowstack) for c in classes]
    buffers, steps = eri3_steps(classes, n_ao, n_aux, kernels)
    ran = run_sequence(binary, workdir / "case", buffers, steps, outs=[OUT],
                       runs=RUNS)
    (T,) = ran.load(OUT)
    assert T.shape == (n_ao, n_ao, n_aux), (T.shape, n_ao, n_aux)
    return T


def compile_timed(classes: Sequence[Eri3Class], n_ao: int, n_aux: int,
                  backend: str, tc: Toolchain, workdir: Path, *,
                  lowstack: bool | None, timeout: float) -> Path:
    """The runner of the ONE class in `classes`, in one form, compiled."""
    if len(classes) != 1:
        raise ValueError(f"a timed eri3 is one class, got {len(classes)}")
    prog = eri3_program(classes, n_ao, n_aux, backend, workdir,
                        lowstack=lowstack, name="eri3_timed")
    return build_runner(tc, prog, workdir / "eri3_timed", timeout=timeout)


def run_timed(binary: Path, classes: Sequence[Eri3Class], n_ao: int,
              n_aux: int, workdir: Path, *, lowstack: bool | None,
              warmup: int, reps: int) -> tuple[np.ndarray, dict]:
    """The tensor and the event times of `warmup + reps` launches from a
    zeroed tensor, as `{"0": [ns per timed run]}`."""
    (cls,) = classes
    buffers, steps = eri3_steps(classes, n_ao, n_aux,
                                [eri3_kernel(cls, lowstack=lowstack)])
    ran = run_sequence(binary, workdir / "timed", buffers, steps, outs=[OUT],
                       runs=reps, warmup=warmup)
    (T,) = ran.load(OUT)
    return T, {"0": ran.events_ns}


PACKED_SIZES = ("n_cell", "n_bra_rec", "n_aux_rec", "n_col", "n_aux")


def packed_kernel(cls: Eri3Class, lowstack: bool | None = None,
                  spherical_aux: bool = False,
                  spherical_bra: bool = False,
                  rolled: bool = False, warp: bool = False) -> Kernel:
    """The class's packed kernel (its spherical-aux mode when
    `spherical_aux`), its signature checked against the records and the
    packed cell arrays."""
    kern = (build_warp_eri3_kernel((cls.la, cls.lb, cls.lP),
                                   spherical_bra=spherical_bra) if warp
            else build_eri3_kernel((cls.la, cls.lb, cls.lP),
                                   lowstack=lowstack, packed=True,
                                   spherical_aux=spherical_aux,
                                   spherical_bra=spherical_bra,
                                   rolled=rolled))
    names = [p.name for p in kern.params]
    want = [*PACKED_SIZES,
            *(_param(n) for n in (*Eri3Class.BRA_ARRAYS,
                                  *Eri3Class.AUX_ARRAYS)),
            *PACKED_CELL_ARRAYS, OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the records "
                         f"want {want}")
    return kern


def packed_layout(classes: Sequence[Eri3Class], n_aux: int
                  ) -> tuple[list[dict[str, tuple[int, ...]]], int]:
    """Every pair once, in class order from column 0: each class's packed
    cell arrays (`packed_cells`) and the column count."""
    columns, n_col = packed_columns(classes)
    return [packed_cells(c, columns, n_aux) for c in classes], n_col


def gather_packed(T: np.ndarray, classes: Sequence[Eri3Class],
                  cells: Sequence[dict[str, tuple[int, ...]]], n_col: int,
                  n_aux: int) -> np.ndarray:
    """The packed output the square tensor `T` implies: each cell's staged
    block `T[off_a + a, off_b + b, off_p + q]` at `dst + n_aux (a nB + b)
    + q`, `(n_col, n_aux)`. Elements no cell owns are NaN."""
    out = np.full(n_col * n_aux, np.nan)
    for cls, cl in zip(classes, cells, strict=True):
        nA, nB = len(cartesians(cls.la)), len(cartesians(cls.lb))
        nP = len(cartesians(cls.lP))
        assert len(cl["dst"]) == cls.n_cell  # every cell is laid out
        for k in range(cls.n_cell):
            oa, ob, op = cls.off_a[k], cls.off_b[k], cls.off_p[k]
            blk = T[oa:oa + nA, ob:ob + nB, op:op + nP]  # [a][b][q]
            idx = (cl["dst"][k] + n_aux * np.arange(nA * nB)[:, None]
                   + np.arange(nP)[None, :])
            out[idx] = blk.reshape(nA * nB, nP)
    return out.reshape(n_col, n_aux)


def packed_program(classes: Sequence[Eri3Class], backend: str,
                   workdir: Path, *, lowstack: bool | None = None,
                   name: str = "peri3_runner",
                   spherical_aux: bool = False,
                   spherical_bra: bool = False,
                   rolled: bool = False, warp: bool = False) -> Program:
    """The runner over every class's packed unit, emitted into `workdir`."""
    workdir.mkdir(parents=True, exist_ok=True)
    units = []
    for cls in classes:
        kern = packed_kernel(cls, lowstack, spherical_aux, spherical_bra,
                             rolled, warp)
        units.append((_unit(kern, backend, workdir, None), [kern.name]))
    return units_runner(backend, units, workdir, name)


def packed_steps(classes: Sequence[Eri3Class],
                 cells: Sequence[dict[str, tuple[int, ...]]], n_col: int,
                 n_aux: int, kernels: Sequence[Kernel], *,
                 warp: bool = False) -> tuple[dict, list[Step]]:
    """The sequence's buffers and its launches, one per class, into the
    packed output, which starts NaN (module docstring)."""
    buffers: dict = {OUT: np.full(n_col * n_aux, np.nan)}
    steps = []
    for c, (cls, cl, kern) in enumerate(zip(classes, cells, kernels,
                                            strict=True)):
        arr = cls.arrays()
        n_cell = len(cl["dst"])
        args: dict = {"n_cell": n_cell, "n_bra_rec": cls.n_bra_rec,
                      "n_aux_rec": cls.n_aux_rec, "n_col": n_col,
                      "n_aux": n_aux, OUT: OUT}
        for name in (*Eri3Class.BRA_ARRAYS, *Eri3Class.AUX_ARRAYS):
            buf = f"c{c}.{_param(name)}"
            buffers[buf] = np.asarray(arr[name], dtype=np.float64)
            args[_param(name)] = buf
        for name in PACKED_CELL_ARRAYS:
            buf = f"c{c}.{name}"
            buffers[buf] = np.asarray(cl[name], dtype=np.int32)
            args[name] = buf
        grid, block = (warp_launch_shape(n_cell) if warp
                       else launch_shape(n_cell, THREADS))
        steps.append(Step(kern, grid, block, args))
    return buffers, steps


def build_packed_on_card(classes: Sequence[Eri3Class],
                         cells: Sequence[dict[str, tuple[int, ...]]],
                         n_col: int, n_aux: int, backend: str,
                         tc: Toolchain, workdir: Path, *,
                         lowstack: bool | None = None,
                         timeout: float = 1800.0,
                         spherical_aux: bool = False,
                         spherical_bra: bool = False,
                         rolled: bool = False,
                         warp: bool = False) -> np.ndarray:
    """The packed `(mn|P)` of `classes` built on the card over `cells`
    (`packed_layout`'s, or a subset's of one layout), `(n_col, n_aux)`,
    NaN where no launched cell stores, the sequence run `RUNS` times
    bit-identically. The caller holds the card. `spherical_aux` runs the
    solid-harmonic aux mode (`n_aux` and `cells`' `dst` in its columns),
    `spherical_bra` with it the fully spherical one (`n_col` in spherical
    bra functions, #745)."""
    prog = packed_program(classes, backend, workdir, lowstack=lowstack,
                          spherical_aux=spherical_aux,
                          spherical_bra=spherical_bra, rolled=rolled,
                          warp=warp)
    binary = build_runner(tc, prog, workdir / "peri3_bin", timeout=timeout)
    kernels = [packed_kernel(c, lowstack, spherical_aux, spherical_bra,
                             rolled, warp) for c in classes]
    buffers, steps = packed_steps(classes, cells, n_col, n_aux, kernels,
                                  warp=warp)
    ran = run_sequence(binary, workdir / "case", buffers, steps, outs=[OUT],
                       runs=RUNS)
    (B,) = ran.load(OUT)
    assert B.shape == (n_col * n_aux,), (B.shape, n_col, n_aux)
    return B.reshape(n_col, n_aux)
