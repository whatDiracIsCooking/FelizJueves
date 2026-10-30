"""S, T or V of a basis built ON A CARD from its one-electron launch (#242)
-- harness.

    kernel_of(cls, target)                                      -> Kernel
    one_e_steps(launch, target, n_ao)                           -> [Step]
    build_on_card(launch, target, n_ao, backend, tc, workdir,
                  units=)                                       -> [run] -> M
    tier_units(backend, kernels)                                -> {name: path}

THE ONE FUNCTION a later test reuses (#243's `h = T + V` capstone) is
`build_on_card`: it compiles one runner per class for one target of one
launch with devtools/config.sh's flags under #92's warning filter
(`harness_driver.build_steps`, through `compile_checked`), runs
them behind the caller's card lock and returns each run's matrix. Nothing
else is decided here: the launch is `launch.one_electron_launch`'s value,
the records are `OneEClass.arrays(target)` under the kernel's own parameter
names, and the kernels are `kernel_s` / `kernel_t` / `kernel_v`'s, emitted
by `emit/cuda.py` / `emit/hip.py` unchanged.

ON THE SHARED RUNNER (#649, ported by #658): one runner per class's unit,
one launch per pair class, one thread per record (`launch_shape`, `THREADS`
lanes), the `n_ao x n_ao` matrix carried from class to class from zero.
Every class ADDS (each kernel's scatter is an `AtomicAdd`, kernel_jab's
rule), so several classes into one matrix is the kernels' own contract: an
element is owned by exactly one shell pair. The spec names every argument
from the kernel's own parameter list (V's `Z` / `Cnuc` are the launch's),
so the call cannot drift from the signature.

THE RUNS: the fp64 atomics' order may differ from run to run, so the
runner's bitwise witness (`runs=`) does not apply. The whole chain is run
`RUNS` times instead, each from a zeroed matrix, and the caller bounds the
spread (test_one_electron_device.py's run-to-run bound).

THE UNITS. `units` maps a kernel name to a file already written -- under
devtools/kernel-tier.sh, the generate phase's own (`tier_units`, which
asserts each byte-identical to `emit(...)`); any class without one (a class
`la > lb` the ladder does not write, or a run outside the tier) is emitted
into `workdir` here, the same text.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, build_steps, carried, run_chain
from plan_driver import RUNS
from test_kernel_device import BACKEND

from felizjueves.eri.j_coulomb.launch import OneEClass, OneELaunch
from felizjueves.ir import Intent, Kernel
from felizjueves.one_electron.kernel_s import build_s_kernel
from felizjueves.one_electron.kernel_s import launch_shape as s_launch_shape
from felizjueves.one_electron.kernel_t import build_t_kernel
from felizjueves.one_electron.kernel_t import launch_shape as t_launch_shape
from felizjueves.one_electron.kernel_v import build_v_kernel
from felizjueves.one_electron.kernel_v import launch_shape as v_launch_shape
from felizjueves.spec import Target

THREADS = 32  # lanes per block, one record per thread; not numerical

BUILD: dict[Target, Callable[[tuple[int, int]], Kernel]] = {
    Target.S: build_s_kernel, Target.T: build_t_kernel,
    Target.V: build_v_kernel}
SHAPE = {Target.S: s_launch_shape, Target.T: t_launch_shape,
         Target.V: v_launch_shape}
NUCLEI = ("Z", "Cnuc")  # V's per-nucleus arrays, the launch's


def kernel_of(cls: OneEClass, target: Target) -> Kernel:
    """The class's kernel of `target`: its `spec(target)` built."""
    return BUILD[target](cls.spec(target))


def _check_signature(kern: Kernel, cls: OneEClass, target: Target) -> None:
    """The kernel's parameters are the scalars, the record arrays in
    `arrays(target)`'s order, V's nuclei, and the one OUT matrix."""
    names = [p.name for p in kern.params]
    out = [p.name for p in kern.params if p.intent is Intent.OUT]
    scalars = ["n_rec", "n_ao"] + (["n_nuc"] if target is Target.V else [])
    want = [*scalars, *cls.arrays(target),
            *(NUCLEI if target is Target.V else ()), *out]
    if names != want or len(out) != 1:
        raise ValueError(f"{kern.name}: signature {names}, the launch's "
                         f"records want {want}")


def one_e_steps(launch: OneELaunch, target: Target, n_ao: int) -> list[Step]:
    """One launch per class of `launch`, every argument but the matrix."""
    if target not in BUILD:
        raise ValueError(f"{target!r} is not a one-electron target")
    steps = []
    for cls in launch:
        kern = kernel_of(cls, target)
        _check_signature(kern, cls, target)
        n = len(cls.recs)
        args: dict = {"n_rec": n, "n_ao": n_ao,
                      **{k: np.asarray(v)
                         for k, v in cls.arrays(target).items()}}
        if target is Target.V:
            args |= {"n_nuc": len(launch.nuclei), "Z": np.asarray(launch.Z),
                     "Cnuc": np.asarray(launch.Cnuc)}
        grid, block = SHAPE[target](n, THREADS)
        steps.append(Step(kern, grid, block, args))
    return steps


def tier_units(backend: str, kernels: Sequence[Kernel]) -> dict[str, Path]:
    """Under the kernel tier, the generate phase's own file of every kernel
    in `kernels` it WROTE, each asserted byte-identical to `emit(...)`;
    outside the tier (or for another backend), nothing."""
    out = os.environ.get("KERNEL_TIER_OUT")
    if not out or os.environ.get("KERNEL_TIER_BACKEND") != backend:
        return {}
    emit, arch, ext = BACKEND[backend]
    got = {}
    for k in kernels:
        path = Path(out) / f"{k.name}.{ext}"
        if not path.exists():
            continue
        assert path.read_text() == emit(k, arch), (
            f"{path} differs from what emit() writes now -- a stale generate "
            "phase, or a nondeterministic emitter")
        got[k.name] = path
    return got


def build_on_card(launch: OneELaunch, target: Target, n_ao: int, backend: str,
                  tc: Toolchain, workdir: Path, *,
                  units: Mapping[str, Path] | None = None) -> np.ndarray:
    """`target`'s matrix of `launch` built on the card, `[run][n_ao, n_ao]`
    (`RUNS` whole runs, each from zero). The caller holds the card (its
    device fixture)."""
    steps = one_e_steps(launch, target, n_ao)
    binaries = build_steps(tc, backend, steps, workdir, units=units)
    return np.stack(run_chain(binaries, steps, carried(steps),
                              np.zeros((n_ao, n_ao)), workdir, runs=RUNS))
