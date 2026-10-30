"""The dipole `MU` of a basis built ON A CARD from its one-electron launch
(#523) -- harness.

    mu_steps(launch, n_ao, origin, q, target=)          -> [Step]
    compile_mu(launch, n_ao, backend, tc, workdir, units=) -> MuBinary
    MuBinary.run(origin, q)                             -> [run][3][n_ao][n_ao]
    tier_units(backend, kernels)                        -> {name: path}

Every entry point takes `target=` (`Target.DIPOLE` by default, so every
dipole call is unchanged): `Target.QUADRUPOLE` builds the same chain over
`kernel_mu.build_quad_kernel`'s `quad_{la}{lb}` units, the same parameters
and records, and a `6 x n_ao x n_ao` `MU` (xx, xy, xz, yy, yz, zz), its
own cap (`QUAD_LMAX`, (dd)) below the dipole's (`MU_LMAX`, (ff), #627).

`one_electron_driver.py`'s pattern for `Target.DIPOLE`, with one
difference: the origin `C` and the charge `q` are the kernel's RUNTIME
scalars (`kernel_mu`'s module docstring), so they are spec VALUES, not
spelled into a compiled program. One set of compiled runners serves every
origin and every charge -- the origin-shift and `q = -1` checks run the
SAME machine code at two inputs, so they test the kernel's runtime
arguments, not two compilations.

Nothing else is decided here: the launch is `launch.one_electron_launch`'s
value (no nuclei: the dipole reads none), the records are
`OneEClass.arrays(Target.DIPOLE)` under the kernel's own parameter names,
and the kernels are `kernel_mu.build_mu_kernel`'s (`mu_{la}{lb}`, every
component in one launch), emitted by `emit/cuda.py` / `emit/hip.py`
unchanged -- harness, in `test/`, not a second emitter.

ON THE SHARED RUNNER (#649, ported by #658): one runner per class's unit
(`harness_driver.build_chain`, under #92's warning filter), one launch per
pair class, one thread per record (`kernel_mu.launch_shape`, `THREADS`
lanes), `MU` carried from class to class from zero (the kernel's
precondition, `kernel_mu.ZEROED_MU`). Every class ADDS (kernel_jab's
scatter rule, per component) through fp64 atomics whose order may differ
from run to run, so the whole chain runs `RUNS` times, each from zero, and
the caller bounds the spread rather than the runner demanding bitwise runs.

THE UNITS. As `one_electron_driver.py`: a kernel the tier's generate phase
wrote (`--mu-l`) is that file, asserted byte-identical to `emit(...)`; any
other class (`la > lb`, which the ladder never writes, or a tier run whose
generator did not ask for `--mu-l`) is emitted into `workdir` here, the same
text. Nothing above (ff) is ever emitted for the dipole (`MU_LMAX`, #627)
nor above (dd) for the quadrupole (`QUAD_LMAX`): refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, build_chain, run_chain, typed
from one_electron_driver import tier_units
from plan_driver import RUNS

from felizjueves.eri.j_coulomb.launch import OneEClass, OneELaunch
from felizjueves.ir import Intent, Kernel
from felizjueves.one_electron.kernel_mu import (
    MU,
    ORDER,
    build_mu_kernel,
    launch_shape,
    n_components,
)
from felizjueves.spec import Target

__all__ = ["MU_LMAX", "QUAD_LMAX", "MuBinary", "compile_mu", "kernel_of",
           "mu_steps", "tier_units", "units_of"]

THREADS = 32  # lanes per block, one record per thread; not numerical
MU_LMAX = 3  # #627: every shell l <= 3, nothing above (ff)
QUAD_LMAX = 2  # the quadrupole: every shell l <= 2, nothing above (dd)
# each target's cap, and the class it names
LMAX = {Target.DIPOLE: (MU_LMAX, "(ff)"), Target.QUADRUPOLE: (QUAD_LMAX, "(dd)")}
# Each target's component count: x, y, z; and xx, xy, xz, yy, yz, zz.
NCOMPS = {t: n_components(e=e) for t, e in ORDER.items()}
SCALARS = ("C_x", "C_y", "C_z", "q")  # the runtime scalars


def kernel_of(cls: OneEClass, target: Target = Target.DIPOLE) -> Kernel:
    """The class's dipole (or quadrupole) kernel, refusing anything above
    the target's cap ((ff) for the dipole, (dd) for the quadrupole)."""
    if target not in ORDER:
        raise ValueError(f"{target!r} is not a multipole target")
    cap, top = LMAX[target]
    if max(cls.la, cls.lb) > cap:
        raise ValueError(f"pair class ({cls.la}, {cls.lb}) is above {top}: "
                         f"nothing is emitted there for {target.name}")
    return build_mu_kernel(cls.spec(target))


def _check_signature(kern: Kernel, cls: OneEClass,
                     target: Target = Target.DIPOLE) -> None:
    """`n_rec n_ao`, the record arrays in `arrays(target)`'s order, the
    origin and charge, and the one OUT `MU`."""
    names = [p.name for p in kern.params]
    want = ["n_rec", "n_ao", *cls.arrays(target), *SCALARS, MU]
    if names != want or kern.params[-1].intent is not Intent.OUT:
        raise ValueError(f"{kern.name}: signature {names}, the launch's "
                         f"records want {want}")


def mu_steps(launch: OneELaunch, n_ao: int, origin: Sequence[float],
             q: float, target: Target = Target.DIPOLE) -> list[Step]:
    """One launch per class of `launch` at origin `C` and charge `q`, every
    argument but `MU`."""
    head = [float(x) for x in origin]
    if len(head) != 3:
        raise ValueError(f"origin {origin!r} is not three floats")
    scalars = dict(zip(SCALARS, (*head, float(q)), strict=True))
    steps = []
    for cls in launch:
        kern = kernel_of(cls, target)
        _check_signature(kern, cls, target)
        n = len(cls.recs)
        grid, block = launch_shape(n, THREADS)
        steps.append(Step(kern, grid, block, {
            "n_rec": n, "n_ao": n_ao, **typed(kern, cls.arrays(target)),
            **scalars}))
    return steps


@dataclass(frozen=True)
class MuBinary:
    """The compiled runners of one launch's `MU`; `run` feeds them an origin
    and a charge."""

    launch: OneELaunch
    n_ao: int
    binaries: tuple[Path, ...]
    workdir: Path
    target: Target = Target.DIPOLE

    def run(self, origin: Sequence[float], q: float) -> np.ndarray:
        """`MU` at `origin` and charge `q`, `[run][ncomp][n_ao][n_ao]`
        (`RUNS` whole runs, each from zero). The caller holds the card (its
        device fixture)."""
        steps = mu_steps(self.launch, self.n_ao, origin, q, self.target)
        cases = self.workdir / "cases"  # one directory per call
        at = cases / str(len(list(cases.glob("*"))) if cases.exists() else 0)
        zero = np.zeros((NCOMPS[self.target], self.n_ao, self.n_ao))
        return np.stack([run_chain(self.binaries, steps, zero, at / f"run{r}")
                         for r in range(RUNS)])


def compile_mu(launch: OneELaunch, n_ao: int, backend: str, tc: Toolchain,
               workdir: Path, *, units: Mapping[str, Path] | None = None,
               target: Target = Target.DIPOLE) -> MuBinary:
    """Compile every class's runner of `launch` with devtools/config.sh's
    flags, FAILING on a warning located in generated text or a harness
    header (#92). The runners do not depend on the origin or the charge."""
    steps = mu_steps(launch, n_ao, (0.0, 0.0, 0.0), 1.0, target)
    binaries = build_chain(tc, backend, steps, workdir, units=units)
    return MuBinary(launch, n_ao, tuple(binaries), workdir, target)


def units_of(backend: str, launch: OneELaunch,
             target: Target = Target.DIPOLE) -> dict[str, Path]:
    """The tier's own files of the launch's kernels it wrote (none unless
    its generator asked for `--mu-l` / `--quad-l`)."""
    return tier_units(backend, [kernel_of(c, target) for c in launch])
