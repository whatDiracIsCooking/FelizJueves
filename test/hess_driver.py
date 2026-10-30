"""The contracted one-electron nuclear hessians built ON A CARD by the d2
kernels -- harness.

    HessLaunch(kernel, n_rec, arrays, scalars)          one kernel launch
    one_electron_launches(basis, nuclei, target, M)     -> [HessLaunch]
    hess_steps(launches)                                -> [harness_driver.Step]
    hess_runners(launches, backend, workdir, units=)    -> {kernel: Program}
    compile_hess(launches, n_atom, backend, tc, workdir, units=) -> HessBinary
    HessBinary.run()                           -> [run][n_atom][3][n_atom][3]

`grad_driver.py`'s pattern with the one OUT `H[n_atom, 3, n_atom, 3]` in
place of `G[n_atom, 3]` (`hdq_`'s is `D1`, sized by `n_out`), on the shared
runner (#649, ported by #659): one runner per distinct kernel
(`harness_driver.chain_runners`, compiled under #92's warning filter), one
launch per `HessLaunch` -- one thread per record, `THREADS` lanes -- every
argument named from the kernel's parameter list, and the OUT carried from
zero (the kernels' precondition, `kernel_h1e.ZEROED_H`) through every
launch. The kernels are `hessian.kernel_h1e`'s (and `kernel_hjab` /
`kernel_hdq`'s), emitted by `emit/cuda.py` / `emit/hip.py` unchanged; a
unit the tier's generate phase wrote (`--hess-l`) is that file.

THE RUNS: the whole chain `RUNS` times, each from a zeroed OUT in its own
processes (`harness_driver.run_chain`), so the fp64 atomics' run-to-run
spread is witnessed by the test, not assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import Program, Step, build_chain, chain_runners, run_chain
from plan_driver import RUNS

from felizjueves.gradient import host
from felizjueves.hessian.kernel_h1e import build_h1e_kernel, launch_shape
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.spec import Target

__all__ = ["HessBinary", "HessLaunch", "compile_hess", "hess_runners",
           "hess_steps", "one_electron_launches"]

THREADS = 32  # lanes per block, one record per thread; not numerical
H = "H"


@dataclass(frozen=True)
class HessLaunch:
    """One kernel over `n_rec` records: its array arguments by parameter
    name (every array parameter but `H`) and its scalar ones."""

    kernel: Kernel
    n_rec: int
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, int] = field(default_factory=dict)
    out: str = H  # the one OUT, shared by every launch (`hdq_`'s is `D1`)

    def __post_init__(self) -> None:
        want = {p.name for p in self.kernel.params if p.array} - {self.out}
        if set(self.arrays) != want:
            raise ValueError(f"{self.kernel.name}: arrays {sorted(self.arrays)}"
                             f", the kernel takes {sorted(want)}")
        out = [p for p in self.kernel.params if p.intent is Intent.OUT]
        if [p.name for p in out] != [self.out]:
            raise ValueError(f"{self.kernel.name}: the one OUT is not "
                             f"{self.out}")


def one_electron_launches(basis, nuclei, target: Target,
                          M) -> list[HessLaunch]:
    """One `d2s_` / `d2t_` / `d2v_` launch per ordered-by-`i <= j` pair
    class of `basis`, every record of the class."""
    out = []
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        kern = build_h1e_kernel((target, la, lb))
        rec = host.records(basis, nuclei, la, lb)
        types = {p.name: p.type for p in kern.params if p.array}
        arrs = {k: np.asarray(v, dtype=np.int32 if types[k] is Scalar.I32
                              else np.float64)
                for k, v in rec.items() if k in types}
        arrs["M"] = np.asarray(M, dtype=np.float64)
        if target is Target.V:
            arrs["Z"] = np.array([z for z, _ in nuclei], dtype=np.float64)
            arrs["Cnuc"] = np.array([c for _, c in nuclei], dtype=np.float64)
        n = len(rec["s_ab"])
        out.append(HessLaunch(kern, n, arrs, {
            "n_rec": n, "n_ao": basis.n_ao, "n_atom": len(nuclei)}))
    return out


def hess_steps(launches: Sequence[HessLaunch]) -> list[Step]:
    """Each launch as a `harness_driver.Step`, one thread per record,
    `THREADS` lanes; its OUT carried."""
    out = []
    for x in launches:
        grid, block = launch_shape(x.n_rec, THREADS)
        out.append(Step(x.kernel, grid, block, {**x.arrays, **x.scalars}))
    return out


def hess_runners(launches: Sequence[HessLaunch], backend: str, workdir: Path,
                 *, units: Mapping[str, Path] | None = None
                 ) -> dict[str, Program]:
    """One runner per distinct kernel of `launches` (module docstring)."""
    return chain_runners((x.kernel for x in launches), backend, workdir,
                         units=units)


@dataclass(frozen=True)
class HessBinary:
    """Every runner of a hessian's launches, compiled; `run` returns
    `[run][n_atom][3][n_atom][3]`, or `[run][n_out]` given `n_out`."""

    launches: tuple[HessLaunch, ...]
    n_atom: int
    backend: str
    binaries: Mapping[str, Path]  # by kernel name
    workdir: Path
    n_out: int | None = None

    def run(self) -> np.ndarray:
        """`RUNS` runs, each from a zeroed OUT through every launch. The
        caller holds the card (its device fixture)."""
        outs = {x.out for x in self.launches}
        if len(outs) != 1:
            raise ValueError(f"the launches carry {sorted(outs)}, not one OUT")
        n = self.n_atom
        init = (np.zeros((n, 3, n, 3)) if self.n_out is None
                else np.zeros(self.n_out))
        return np.array(run_chain(self.binaries, hess_steps(self.launches),
                                  outs.pop(), init, self.workdir / "runs",
                                  runs=RUNS))


def compile_hess(launches: Sequence[HessLaunch], n_atom: int, backend: str,
                 tc: Toolchain, workdir: Path, *,
                 units: Mapping[str, Path] | None = None,
                 n_out: int | None = None,
                 timeout: float = 1800.0) -> HessBinary:
    """Compile every runner with devtools/config.sh's flags, FAILING on a
    warning located in generated text (#92)."""
    runners = hess_runners(launches, backend, workdir, units=units)
    return HessBinary(tuple(launches), n_atom, backend,
                      build_chain(tc, runners, workdir, timeout=timeout),
                      workdir, n_out)
