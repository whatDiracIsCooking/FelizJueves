"""The cross-basis overlap `S_{B1,B2}` built ON A CARD from its launch --
harness.

    sx_steps(launch)                             -> [Step]
    build_on_card(launch, backend, tc, workdir,
                  units=)                        -> [run] -> S[n_a, n_b]

The launch is `launch.cross_overlap_launch`'s value, the records
`XClass.arrays()` under the kernel's own parameter names, and the kernels
`kernel_sx.build_sx_kernel`'s, emitted by `emit/cuda.py` / `emit/hip.py`
unchanged -- harness, in `test/`, not a second emitter.

ON THE SHARED RUNNER (#649, ported by #658), as `one_electron_driver.py`
does for `S`, `T` and `V`: one runner per ordered class's unit, compiled
with devtools/config.sh's flags under #92's warning filter
(`harness_driver.build_steps`), one launch per class, one thread per record
(`launch_shape`, `THREADS` lanes), the `n_a x n_b` matrix carried from
class to class from zero. Every element is owned by exactly one ordered
shell pair, whose records add into it atomically -- so the whole chain is
run `RUNS` times, each from zero, and the caller bounds the spread rather
than the runner demanding bitwise runs.

THE UNITS. `units` maps a kernel name to a file already written (the kernel
tier's generate phase, when `KERNEL_GEN_CMD` asks for `--sx-l`; it does not
today); any other class is emitted into `workdir` here, the same text.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, build_steps, run_chain
from plan_driver import RUNS

from felizjueves.eri.j_coulomb.launch import XClass, XLaunch
from felizjueves.ir import Intent, Kernel
from felizjueves.one_electron.kernel_sx import build_sx_kernel, launch_shape

THREADS = 32  # lanes per block, one record per thread; not numerical


def _check_signature(kern: Kernel, cls: XClass) -> None:
    names = [p.name for p in kern.params]
    want = ["n_rec", "n_a", "n_b", *cls.arrays(), "S"]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the launch's "
                         f"records want {want}")


def sx_steps(launch: XLaunch) -> list[Step]:
    """One launch per ordered class of `launch`, every argument but `S`."""
    steps = []
    for cls in launch:
        kern = build_sx_kernel(cls.spec())
        _check_signature(kern, cls)
        n = len(cls.recs)
        grid, block = launch_shape(n, THREADS)
        steps.append(Step(kern, grid, block, {
            "n_rec": n, "n_a": launch.n_a, "n_b": launch.n_b,
            **{k: np.asarray(v) for k, v in cls.arrays().items()}}))
    return steps


def build_on_card(launch: XLaunch, backend: str, tc: Toolchain, workdir: Path,
                  *, units: Mapping[str, Path] | None = None) -> np.ndarray:
    """`S_{B1,B2}` of `launch` built on the card, `[run][n_a, n_b]` (`RUNS`
    whole runs, each from zero). The caller holds the card (its device
    fixture)."""
    steps = sx_steps(launch)
    binaries = build_steps(tc, backend, steps, workdir, units=units)
    return np.stack(run_chain(binaries, steps, "S",
                              np.zeros((launch.n_a, launch.n_b)), workdir,
                              runs=RUNS))
