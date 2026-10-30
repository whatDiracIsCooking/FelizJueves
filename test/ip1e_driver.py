"""The one-electron derivative matrices (#687) built ON A CARD by
`hessian.kernel_ip1e` -- harness.

    ip1e_steps(basis, nuclei, target, form, nuc=None) -> [harness_driver.Step]
    ip1e_runners(steps, backend, workdir, units=)      -> {kernel: Program}
    compile_ip1e(steps, backend, tc, workdir, units=)  -> Ip1eBinary
    Ip1eBinary.run()                                   -> [run][3][n_ao][n_ao]

`hess_driver.py`'s pattern on the shared runner (#649): one runner per
distinct kernel (`harness_driver.chain_runners`, compiled under #92's
warning filter), one launch per pair class of the basis -- one thread per
record, `THREADS` lanes -- every argument named from the kernel's parameter
list, and the one OUT (`X1`, or `XC` for the rinv form at nucleus `nuc`)
carried from zero (`kernel_ip1e.ZEROED_X`) through every launch. The
kernels are emitted by `emit/cuda.py` / `emit/hip.py` unchanged; a unit the
tier's generate phase wrote (`--ip-one-e-l`) is that file.

THE RUNS: the whole chain `RUNS` times, each from a zeroed OUT in its own
processes (`harness_driver.run_chain`), so the fp64 atomics' run-to-run
spread is witnessed by the test, not assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import (
    Program,
    Step,
    build_chain,
    carried,
    chain_runners,
    run_chain,
)
from plan_driver import RUNS

from felizjueves.gradient import host
from felizjueves.hessian.kernel_ip1e import build_ip1e_kernel, launch_shape
from felizjueves.ir import Scalar
from felizjueves.spec import Target

__all__ = ["Ip1eBinary", "compile_ip1e", "ip1e_runners", "ip1e_steps"]

THREADS = 32  # lanes per block, one record per thread; not numerical


def ip1e_steps(basis, nuclei, target: Target, form: str,
               nuc: int | None = None) -> list[Step]:
    """One launch per ordered-by-`i <= j` pair class of `basis`, every
    record of the class; `nuc` the rinv form's nucleus."""
    out = []
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        kern = build_ip1e_kernel((target, la, lb, form))
        rec = host.records(basis, nuclei, la, lb)
        types = {p.name: p.type for p in kern.params if p.array}
        args: dict = {k: np.asarray(v, dtype=np.int32 if types[k] is Scalar.I32
                                    else np.float64)
                      for k, v in rec.items() if k in types}
        n = len(rec["s_ab"])
        args |= {"n_rec": n, "n_ao": basis.n_ao}
        if target is Target.V:
            args["Z"] = np.array([z for z, _ in nuclei], dtype=np.float64)
            args["Cnuc"] = np.array([c for _, c in nuclei], dtype=np.float64)
            args["n_atom"] = len(nuclei)
        if form == "rinv":
            args["nuc"] = nuc
        grid, block = launch_shape(n, THREADS)
        out.append(Step(kern, grid, block, args))
    return out


def ip1e_runners(steps: Sequence[Step], backend: str, workdir: Path, *,
                 units: Mapping[str, Path] | None = None
                 ) -> dict[str, Program]:
    """One runner per distinct kernel of `steps` (module docstring)."""
    return chain_runners((s.kernel for s in steps), backend, workdir,
                         units=units)


@dataclass(frozen=True)
class Ip1eBinary:
    """Every runner of one matrix's launches, compiled; `run` returns
    `[run][3][n_ao][n_ao]`."""

    steps: tuple[Step, ...]
    n_ao: int
    binaries: Mapping[str, Path]  # by kernel name
    workdir: Path

    def run(self) -> np.ndarray:
        """`RUNS` runs, each from a zeroed OUT through every launch. The
        caller holds the card (its device fixture)."""
        n = self.n_ao
        return np.array(run_chain(self.binaries, list(self.steps),
                                  carried(self.steps), np.zeros((3, n, n)),
                                  self.workdir / "runs", runs=RUNS))


def compile_ip1e(steps: Sequence[Step], n_ao: int, backend: str,
                 tc: Toolchain, workdir: Path, *,
                 units: Mapping[str, Path] | None = None,
                 timeout: float = 1800.0) -> Ip1eBinary:
    """Compile every runner with devtools/config.sh's flags, FAILING on a
    warning located in generated text (#92)."""
    runners = ip1e_runners(steps, backend, workdir, units=units)
    return Ip1eBinary(tuple(steps), n_ao,
                      build_chain(tc, runners, workdir, timeout=timeout),
                      workdir)
