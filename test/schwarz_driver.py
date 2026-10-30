"""The Schwarz matrix of a basis built ON A CARD -- harness.

    schwarz_steps(classes, n_ao)                     -> [Step]
    build_on_card(classes, n_ao, backend, tc, workdir) -> Q

The records are `eri.schwarz.schwarz_classes`' (every shell pair staged with
`la <= lb`, one contiguous record range per pair), the kernels
`kernel_schwarz.build_schwarz_kernel`'s, emitted by `emit/cuda.py` /
`emit/hip.py` unchanged into `workdir` (the kernel tier's `KERNEL_GEN_CMD`
writes no Schwarz unit) -- harness, in `test/`, not a second emitter.

ON THE SHARED RUNNER (#649, ported by #658): one runner per class's unit
(compiled by `plan_driver.compile_checked` under #92's warning filter,
through `harness_driver.build_chain`), one launch per pair class, one
thread per shell pair (`launch_shape`, `THREADS` lanes), the `n_ao x n_ao`
matrix carried from class to class from zero. The spec names every argument
from the kernel's own parameter list, so the call cannot drift from the
signature. Each element is owned by one shell pair and written by a plain
store, so each class runs `RUNS` times and the runner fails unless every
run is bit-identical.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, build_chain, run_chain, typed
from plan_driver import RUNS

from felizjueves.eri.kernel_schwarz import OUT, build_schwarz_kernel, launch_shape
from felizjueves.eri.schwarz import SchwarzClass
from felizjueves.ir import Intent, Kernel

THREADS = 32  # lanes per block, one shell pair per thread; not numerical


def _check_signature(kern: Kernel) -> None:
    names = [p.name for p in kern.params]
    want = ["n_pair", "n_rec", "n_ao", *SchwarzClass.RECORD_ARRAYS,
            *SchwarzClass.PAIR_ARRAYS, OUT]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the records "
                         f"want {want}")


def schwarz_steps(classes: Sequence[SchwarzClass], n_ao: int) -> list[Step]:
    """One launch per class in `classes`, every argument but `Q`."""
    steps = []
    for cls in classes:
        kern = build_schwarz_kernel((cls.la, cls.lb))
        _check_signature(kern)
        grid, block = launch_shape(cls.n_pair, THREADS)
        steps.append(Step(kern, grid, block, {
            "n_pair": cls.n_pair, "n_rec": cls.n_rec, "n_ao": n_ao,
            **typed(kern, cls.arrays())}))
    return steps


def build_on_card(classes: Sequence[SchwarzClass], n_ao: int, backend: str,
                  tc: Toolchain, workdir: Path) -> np.ndarray:
    """`Q` of `classes` built on the card, `[n_ao, n_ao]`, every class run
    `RUNS` times bit-identically. The caller holds the card (its device
    fixture)."""
    steps = schwarz_steps(classes, n_ao)
    binaries = build_chain(tc, backend, steps, workdir)
    return run_chain(binaries, steps, np.zeros((n_ao, n_ao)), workdir,
                     runs=RUNS)
