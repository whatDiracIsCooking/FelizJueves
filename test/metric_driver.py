"""The DF Coulomb metric `(P|Q)` of an aux basis built ON A CARD -- harness.

    metric_kernel(cls, engine, lowstack)            -> Kernel
    metric_runners(classes, backend, workdir)       -> [(Kernel, Program)]
    build_on_card(classes, n_aux, backend, tc, workdir) -> M
    compile_timed / run_timed                       -> measure_df_lowstack's clock

The records are `eri.density_fit.metric.metric_classes`' (every aux shell pair `P <= Q`
staged with `lP <= lQ`, swapped when the basis order has the higher `l`
first; per class one record table and two record ranges per pair), the
kernels `kernel_metric.build_metric_kernel`'s (or `kernel_hgp`'s
`build_hmetric_kernel`, `engine=HGP`), emitted by `emit/cuda.py` /
`emit/hip.py` unchanged into `workdir` (the kernel tier's `KERNEL_GEN_CMD`
writes no metric unit).

ON THE SHARED RUNNER (#649, ported by #654): one runner per class's unit
(compiled by `plan_driver.compile_checked` under #92's warning filter; it
depends on the class alone, so the compiler cache (#186) serves every basis
after the first), and one launch per class, one thread per aux shell pair
(`launch_shape`, `THREADS` lanes). The spec names every argument from the
kernel's own parameter list, so the call cannot drift from the signature;
the record array `s` is the kernel's `s_aux` (`kernel_metric.PARAM_OF`).

THE MATRIX is carried from class to class -- a zeroed `n_aux x n_aux` into
the first, each output into the next, read and updated. Each element is
owned by one aux shell pair and written by a plain store, so each class runs
`RUNS` times and the runner fails unless every run is bit-identical.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import build_runner, run_case, runner_program
from plan_driver import RUNS, Program, _unit

from felizjueves.eri.density_fit.kernel_hgp import build_hmetric_kernel
from felizjueves.eri.density_fit.kernel_metric import (
    OUT,
    PARAM_OF,
    build_metric_kernel,
    launch_shape,
)
from felizjueves.eri.density_fit.metric import MetricClass
from felizjueves.ir import Intent, Kernel
from felizjueves.spec import Engine

THREADS = 32  # lanes per block, one aux shell pair per thread; not numerical


def _param(name: str) -> str:
    return PARAM_OF.get(name, name)


def metric_kernel(cls: MetricClass, engine: Engine = Engine.MMD,
                  lowstack: bool | None = None) -> Kernel:
    """The class's kernel, its signature checked against the records."""
    lq = (cls.lP, cls.lQ)
    kern = (build_hmetric_kernel(lq) if engine is Engine.HGP
            else build_metric_kernel(lq, lowstack=lowstack))
    names = [p.name for p in kern.params]
    want = ["n_pair", "n_rec", "n_aux",
            *(_param(n) for n in MetricClass.RECORD_ARRAYS),
            *MetricClass.PAIR_ARRAYS, OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the records "
                         f"want {want}")
    return kern


def metric_runners(classes: Sequence[MetricClass], backend: str,
                   workdir: Path, *, engine: Engine = Engine.MMD,
                   lowstack: bool | None = None
                   ) -> list[tuple[Kernel, Program]]:
    """Per class, its kernel and its runner's source, emitted into
    `workdir`; `lowstack=` as `eri3_driver.eri3_program`'s (#606)."""
    workdir.mkdir(parents=True, exist_ok=True)
    out = []
    for cls in classes:
        kern = metric_kernel(cls, engine, lowstack)
        unit = _unit(kern, backend, workdir, None)
        out.append((kern, runner_program(backend, unit, [kern.name], workdir)))
    return out


def _launch(binary: Path, kern: Kernel, cls: MetricClass, n_aux: int,
            M: np.ndarray, workdir: Path, **repeat):
    """One class into `M` (read and updated); `run_case`'s `Ran`."""
    arr = cls.arrays()
    args: dict = {"n_pair": cls.n_pair, "n_rec": cls.n_rec, "n_aux": n_aux,
                  OUT: M}
    for name in MetricClass.RECORD_ARRAYS:
        args[_param(name)] = np.asarray(arr[name], dtype=np.float64)
    for name in MetricClass.PAIR_ARRAYS:
        args[name] = np.asarray(arr[name], dtype=np.int32)
    grid, block = launch_shape(cls.n_pair, THREADS)
    return run_case(binary, kern, workdir, grid, block, args, **repeat)


def build_on_card(classes: Sequence[MetricClass], n_aux: int, backend: str,
                  tc: Toolchain, workdir: Path, *,
                  engine: Engine = Engine.MMD, lowstack: bool | None = None,
                  timeout: float = 1800.0) -> np.ndarray:
    """`(P|Q)` of `classes` built on the card, `[n_aux, n_aux]`, every class
    run `RUNS` times bit-identically. The caller holds the card (its device
    fixture)."""
    M = np.zeros((n_aux, n_aux))
    runners = metric_runners(classes, backend, workdir, engine=engine,
                             lowstack=lowstack)
    for c, (cls, (kern, prog)) in enumerate(zip(classes, runners,
                                                strict=True)):
        binary = build_runner(tc, prog, workdir / f"runner_{c}_{kern.name}",
                              timeout=timeout)
        ran = _launch(binary, kern, cls, n_aux, M, workdir / f"case{c}",
                      runs=RUNS)
        M = np.load(ran.outs[OUT])
        assert M.shape == (n_aux, n_aux), (M.shape, n_aux)
    return M


def compile_timed(classes: Sequence[MetricClass], n_aux: int, backend: str,
                  tc: Toolchain, workdir: Path, *, lowstack: bool | None,
                  timeout: float) -> Path:
    """The runner of the ONE class in `classes`, in one form, compiled."""
    if len(classes) != 1:
        raise ValueError(f"a timed metric is one class, got {len(classes)}")
    ((kern, prog),) = metric_runners(classes, backend, workdir,
                                     lowstack=lowstack)
    return build_runner(tc, prog, workdir / f"runner_{kern.name}",
                        timeout=timeout)


def run_timed(binary: Path, classes: Sequence[MetricClass], n_aux: int,
              workdir: Path, *, lowstack: bool | None, warmup: int,
              reps: int) -> tuple[np.ndarray, dict]:
    """The matrix and the event times of `warmup + reps` launches from a
    zeroed matrix, as `{"0": [ns per timed run]}` (`kblock_driver.timings`'s
    shape, the one class at index 0)."""
    (cls,) = classes
    ran = _launch(binary, metric_kernel(cls, lowstack=lowstack), cls, n_aux,
                  np.zeros((n_aux, n_aux)), workdir / "timed", runs=reps,
                  warmup=warmup)
    return np.load(ran.outs[OUT]), {"0": ran.events_ns}
