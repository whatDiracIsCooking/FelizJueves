"""S, T, V or the dipole of a basis built ON A CARD by the HGP one-electron
kernels (`one_electron.kernel_hgp`, #598) -- harness.

    hgp_one_e_steps(launch, target, basis, origin=, q=)       -> [Step]
    build_on_card(launch, target, basis, backend, tc, workdir,
                  origin=, q=)                                -> M

`one_electron_driver.py`'s shape, for the other engine's staging: one
thread per CONTRACTED SHELL PAIR (`kernel_hgp.launch_shape(n_pair)`), the
arrays `OneEClass.hgp_arrays(target, basis)` gives, and PLAIN STORES (no
atomics: each element has one owner). Every argument is named from the
kernel's own parameter list, so the call cannot drift from the signature:
an `int` scalar is a launch extent (`n_pair`, `n_rec`, `n_ao`, `n_nuc`), a
`double` scalar the dipole's `C_x C_y C_z q` (spec values: one runner per
class serves every origin), an array its `hgp_arrays` entry (V's `Z` and
`Cnuc` the launch's), and the one OUT array the matrix.

ON THE SHARED RUNNER (#649, ported by #658): one runner per class's unit
(`harness_driver.build_chain`, compiled under #92's warning filter), the
output carried from class to class from zero. Plain stores into disjoint
elements, so each class runs `RUNS` times and the runner fails unless every
run is bit-identical.

Harness, in `test/`, not a second emitter: the units are `emit/cuda.py` /
`emit/hip.py`'s text of `build_hgp_kernel`, unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Step, build_chain, run_chain, typed
from plan_driver import RUNS

from felizjueves.ao.shell import Basis
from felizjueves.eri.j_coulomb.launch import OneEClass, OneELaunch
from felizjueves.ir import Intent, Kernel
from felizjueves.one_electron import kernel_hgp as kh
from felizjueves.spec import Target

THREADS = 32  # lanes per block, one shell pair per thread; not numerical
DIPOLE_SCALARS = ("C_x", "C_y", "C_z", "q")


def kernel_of(cls: OneEClass, target: Target) -> Kernel:
    return kh.build_hgp_kernel(kh.hgp_spec(target, cls.la, cls.lb))


def _arrays(launch: OneELaunch, cls: OneEClass, target: Target,
            basis: Basis) -> dict[str, object]:
    """Every input array of the class's kernel, by parameter name."""
    out: dict[str, object] = dict(cls.hgp_arrays(target, basis))
    if target is Target.V:
        out["Z"] = launch.Z
        out["Cnuc"] = launch.Cnuc
    return out


def hgp_one_e_steps(launch: OneELaunch, target: Target, basis: Basis, *,
                    origin: Sequence[float] = (0.0, 0.0, 0.0),
                    q: float = 1.0) -> list[Step]:
    """One launch per class of `launch`, every argument but the matrix."""
    if target not in kh.TARGETS:
        raise ValueError(f"{target!r} is not an HGP one-electron target")
    scalars = {"n_ao": basis.n_ao, "n_nuc": len(launch.nuclei),
               **dict(zip(DIPOLE_SCALARS, (*map(float, origin), float(q)),
                          strict=True))}
    steps = []
    for cls in launch:
        kern = kernel_of(cls, target)
        arrs = _arrays(launch, cls, target, basis)
        want = {p.name for p in kern.params
                if p.array and p.intent is not Intent.OUT}
        if want != set(arrs):
            raise ValueError(f"{kern.name}: arrays {sorted(want)}, staged "
                             f"{sorted(arrs)}")
        args = {"n_pair": len(cls.blocks), "n_rec": len(cls.recs), **scalars}
        args = {p.name: args[p.name] for p in kern.params if not p.array}
        grid, block = kh.launch_shape(len(cls.blocks), THREADS)
        steps.append(Step(kern, grid, block, {**args, **typed(kern, arrs)}))
    return steps


def build_on_card(launch: OneELaunch, target: Target, basis: Basis,
                  backend: str, tc: Toolchain, workdir: Path, *,
                  origin=(0.0, 0.0, 0.0), q: float = 1.0) -> np.ndarray:
    """The matrix (`[3, n, n]` for the dipole) built on the card, every
    class run `RUNS` times bit-identically."""
    steps = hgp_one_e_steps(launch, target, basis, origin=origin, q=q)
    binaries = build_chain(tc, backend, steps, workdir)
    n = basis.n_ao
    shape = (3, n, n) if target is Target.DIPOLE else (n, n)
    return run_chain(binaries, steps, np.zeros(shape), workdir, runs=RUNS)
