"""The packed full ERI, and `J` and `K` by GEMV over it, built ON A CARD.

    eri_steps(classes, n_ao)                          -> [Step]
    gemv_step(n_pp, eri, Dp)                          -> Step
    kperm_step(eri)                                   -> Step
    interpret(classes, n_ao, *, init)                 -> ERI_pp, interpreted
    interpret_gemv(eri, D)                            -> J (or K over X_pp)
    interpret_kperm(eri)                              -> X_pp, interpreted
    build_on_card(classes, n_ao, backend, tc, workdir) -> ERI_pp
    coulomb_on_card(eri, D, backend, tc, workdir)     -> J
    exchange_on_card(eri, D, backend, tc, workdir)    -> (X_pp, K)

The records and tasks are `eri.full_eri.packed.eri_classes`', the kernels
`kernel_eri.build_eri_kernel`'s, `kernel_kperm.build_kperm_kernel`'s and
`kernel_gemv.build_gemv_kernel`'s, emitted
by `emit/cuda.py` / `emit/hip.py` unchanged into `workdir` -- harness, in
`test/`, not a second emitter.

ON THE SHARED RUNNER (#649): one runner per class's unit, one launch per
class, one thread per task, the `n_pp x n_pp` matrix carried from class to
class. Every element is owned by one task and written by a plain store, so
the chain runs `RUNS` times and every run must be bit-identical to the
first; the permute and the GEMV are one thread per column and
bit-reproducible too. `K` is the same GEMV kernel over `X_pp`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, build_steps, run_chain
from plan_driver import RUNS

from felizjueves.eri.full_eri import kernel_eri, kernel_gemv, kernel_kperm
from felizjueves.eri.full_eri.packed import (
    EriClass,
    n_packed,
    pack_density,
    pair_index,
    pair_of,
    unpack,
)
from felizjueves.ir import Intent, Kernel
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys

THREADS = 32  # lanes per block; not numerical
JOBS = min(8, os.cpu_count() or 1)  # unit compiles at once, per build
GEMV_THREADS = 128

_INT = ("brec0", "brec1", "off_a", "off_b", "krec0", "krec1", "off_c",
        "off_d", "task_bra", "task_ket")


def _check_signature(kern: Kernel) -> None:
    names = [p.name for p in kern.params]
    want = [*kernel_eri.SCALARS, *kernel_eri.RECORD_ARRAYS,
            *kernel_eri.INDEX_ARRAYS, kernel_eri.OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the staging "
                         f"wants {want}")


def _args(c: EriClass, n_ao: int) -> dict:
    arr = {k: np.asarray(v, dtype=np.int32 if k in _INT else np.float64)
           for k, v in c.arrays().items()}
    return {"n_task": c.n_task, "n_brec": c.bra.n_rec, "n_krec": c.ket.n_rec,
            "n_bpair": c.bra.n_pair, "n_kpair": c.ket.n_pair, "n_ao": n_ao,
            "n_pp": n_packed(n_ao), **arr,
            "pidx": np.asarray(pair_index(n_ao), dtype=np.int32)}


def eri_steps(classes: Sequence[EriClass], n_ao: int, *,
              lowstack: bool | None = None) -> list[Step]:
    """One launch per class in `classes`, every argument but `eri`;
    `lowstack` is `build_eri_kernel`'s (None: each class's rule)."""
    steps = []
    for c in classes:
        kern = kernel_eri.build_eri_kernel(c.am, lowstack=lowstack)
        _check_signature(kern)
        grid, block = kernel_eri.launch_shape(
            c.n_task, THREADS, rows=kernel_eri.task_threads(c.am, lowstack))
        steps.append(Step(kern, grid, block, _args(c, n_ao)))
    return steps


def gemv_step(n_pp: int, eri: np.ndarray, Dp: np.ndarray) -> Step:
    """The GEMV's launch, every argument but `Jp`."""
    grid, block = kernel_gemv.launch_shape(n_pp, GEMV_THREADS)
    return Step(kernel_gemv.build_gemv_kernel(), grid, block,
                {"n_pp": n_pp, "eri": np.ascontiguousarray(eri),
                 "Dp": np.asarray(Dp, dtype=np.float64)})


def _kperm_args(eri: np.ndarray) -> dict:
    n_pp = eri.shape[0]
    n_ao = (int(np.sqrt(8 * n_pp + 1)) - 1) // 2
    assert n_packed(n_ao) == n_pp
    pm, pn = pair_of(n_ao)
    return {"n_ao": n_ao, "n_pp": n_pp,
            "pm": np.asarray(pm, dtype=np.int32),
            "pn": np.asarray(pn, dtype=np.int32),
            "pidx": np.asarray(pair_index(n_ao), dtype=np.int32),
            "eri": np.ascontiguousarray(eri)}


def kperm_step(eri: np.ndarray) -> Step:
    """The exchange permute's launch, every argument but `X`."""
    grid, block = kernel_kperm.launch_shape(eri.shape[0], GEMV_THREADS)
    return Step(kernel_kperm.build_kperm_kernel(), grid, block,
                _kperm_args(eri))


def interpret_kperm(eri: np.ndarray, *, threads: int = 5) -> np.ndarray:
    """`X_pp` from the interpreted permute over `eri`."""
    n_pp = eri.shape[0]
    grid, block = kernel_kperm.launch_shape(n_pp, threads)
    return Interpreter(kernel_kperm.build_kperm_kernel(),
                       launch=Launch(grid, block, 1)).run(
        **_kperm_args(eri), X=np.full((n_pp, n_pp), np.nan))[kernel_kperm.OUT]


def interpret(classes: Sequence[EriClass], n_ao: int, *,
              init: np.ndarray, threads: int = 3,
              lowstack: bool | None = None) -> np.ndarray:
    """`ERI_pp` from the interpreted kernels of `classes`, onto `init`."""
    eri = init
    for c in classes:
        grid, block = kernel_eri.launch_shape(
            c.n_task, threads, rows=kernel_eri.task_threads(c.am, lowstack))
        eri = Interpreter(kernel_eri.build_eri_kernel(c.am,
                                                      lowstack=lowstack),
                          launch=Launch(grid, block, 1), boys=boys).run(
            **_args(c, n_ao), eri=eri)[kernel_eri.OUT]
    return eri


def interpret_gemv(eri: np.ndarray, D, *, threads: int = 7) -> np.ndarray:
    """`J` from the interpreted GEMV over `eri` and `pack_density(D)`."""
    n_pp = eri.shape[0]
    grid, block = kernel_gemv.launch_shape(n_pp, threads)
    Jp = Interpreter(kernel_gemv.build_gemv_kernel(),
                     launch=Launch(grid, block, 1)).run(
        n_pp=n_pp, eri=eri, Dp=np.array(pack_density(D)),
        Jp=np.zeros(n_pp))[kernel_gemv.OUT]
    return np.array(unpack(list(Jp), len(D)))


def build_on_card(classes: Sequence[EriClass], n_ao: int, backend: str,
                  tc: Toolchain, workdir: Path, *,
                  lowstack: bool | None = None) -> np.ndarray:
    """`ERI_pp` of `classes` built on the card, the chain run `RUNS` times
    bit-identically. The caller holds the card (its device fixture)."""
    steps = eri_steps(classes, n_ao, lowstack=lowstack)
    binaries = build_steps(tc, backend, steps, workdir, jobs=JOBS)
    n_pp = n_packed(n_ao)
    outs = run_chain(binaries, steps, kernel_eri.OUT, np.zeros((n_pp, n_pp)),
                     workdir, runs=RUNS)
    assert all(np.array_equal(outs[0], x) for x in outs[1:]), "runs differ"
    return outs[0]


def coulomb_on_card(eri: np.ndarray, D, backend: str, tc: Toolchain,
                    workdir: Path) -> np.ndarray:
    """`J` by the GEMV on the card over `eri` and `pack_density(D)`."""
    n_pp = eri.shape[0]
    step = gemv_step(n_pp, eri, pack_density(D))
    binaries = build_steps(tc, backend, [step], workdir / "gemv")
    outs = run_chain(binaries, [step], kernel_gemv.OUT, np.zeros(n_pp),
                     workdir / "gemv", runs=RUNS)
    assert all(np.array_equal(outs[0], x) for x in outs[1:]), "runs differ"
    return np.array(unpack(list(outs[0]), len(D)))


def exchange_on_card(eri: np.ndarray, D, backend: str, tc: Toolchain,
                     workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(X_pp, K)`: the permute on the card over `eri`, then `K` by the GEMV
    on the card over `X_pp` and `pack_density(D)`, each `RUNS` times."""
    n_pp = eri.shape[0]
    step = kperm_step(eri)
    binaries = build_steps(tc, backend, [step], workdir / "kperm")
    outs = run_chain(binaries, [step], kernel_kperm.OUT,
                     np.zeros((n_pp, n_pp)), workdir / "kperm", runs=RUNS)
    assert all(np.array_equal(outs[0], x) for x in outs[1:]), "runs differ"
    X = outs[0]
    K = coulomb_on_card(X, D, backend, tc, workdir / "k")
    return X, K
