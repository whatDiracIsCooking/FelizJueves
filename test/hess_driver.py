"""The contracted one-electron nuclear hessians built ON A CARD by the d2
kernels -- harness.

    HessLaunch(kernel, n_rec, arrays, scalars)          one kernel launch
    one_electron_launches(basis, nuclei, target, M)     -> [HessLaunch]
    d2metric_launches(aux, nuclei, W, *, lQ=)          -> [HessLaunch]  (#694)
    d2eri3_launches(basis, aux, nuclei, Gam, *, lP=)   -> [HessLaunch]  (#694)
    hkx_launches(basis, nuclei, D, classes, *, pairs=) -> [HessLaunch]  (#691)
    hess_steps(launches)                               -> [harness_driver.Step]
    hess_runners(launches, backend, workdir, units=)    -> {kernel: Program}
    compile_hess(launches, n_atom, backend, tc, workdir, units=) -> HessBinary
    HessBinary.run()                           -> [run][n_atom][3][n_atom][3]

`grad_driver.py`'s pattern with the one OUT `H[n_atom, 3, n_atom, 3]` in
place of `G[n_atom, 3]` (`hdq_`'s is `D1`, sized by `n_out`), on the shared
runner (#649, ported by #659): one runner per distinct kernel
(`harness_driver.chain_runners`, compiled under #92's warning filter), one
launch per `HessLaunch` -- one thread per record, `THREADS` lanes, or for
`hkx_` (`kernel_hkx`, the exchange Hessian) one warp of `THREADS` lanes per
bra record over its kets grouped by atom pair -- every
argument named from the kernel's parameter list, and the OUT carried from
zero (the kernels' precondition, `kernel_h1e.ZEROED_H`) through every
launch. The kernels are `hessian.kernel_h1e`'s (and `kernel_hjab` /
`kernel_hdq`'s, and the DF `kernel_d2metric` / `kernel_d2eri3`'s, one
thread per aux shell pair or cell, on `df_grad_launch`'s arrays), emitted
by `emit/cuda.py` / `emit/hip.py` unchanged; a unit the tier's generate
phase wrote (`--hess-l`) is that file.

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
from df_grad_launch import eri3_arrays, metric_arrays
from harness_driver import Program, Step, build_chain, chain_runners, run_chain
from plan_driver import RUNS

from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.gradient import host
from felizjueves.hessian import host as hessian_host
from felizjueves.hessian.kernel_d2eri3 import build_d2eri3_kernel
from felizjueves.hessian.kernel_d2metric import build_d2metric_kernel
from felizjueves.hessian.kernel_h1e import build_h1e_kernel, launch_shape
from felizjueves.hessian.kernel_hkx import build_hkx_kernel
from felizjueves.hessian.kernel_hkx import launch_shape as hkx_launch_shape
from felizjueves.ir import Intent, Kernel, Scalar
from felizjueves.spec import Target

__all__ = ["HessBinary", "HessLaunch", "compile_hess", "d2eri3_launches",
           "d2metric_launches", "hess_runners", "hess_steps",
           "hkx_launches", "one_electron_launches"]

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
    warp_per_record: bool = False  # `hkx_`: one warp per bra record

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


def d2metric_launches(aux, nuclei, W, *, lQ: int | None = None
                      ) -> list[HessLaunch]:
    """One `d2metric_` launch per metric class of `aux` (`metric_classes`'
    staging, the interpreted tests'), each in its default form; `lQ` keeps
    the classes with that ket `l` only."""
    out = []
    for c in metric_classes(aux):
        if lQ is not None and c.lQ != lQ:
            continue
        arrs = metric_arrays(c, aux, nuclei) | {
            "M": np.asarray(W, dtype=np.float64)}
        out.append(HessLaunch(build_d2metric_kernel((c.lP, c.lQ)), c.n_pair,
                              arrs, {"n_pair": c.n_pair, "n_rec": c.n_rec,
                                     "n_aux": aux.n_ao,
                                     "n_atom": len(nuclei)}))
    return out


def d2eri3_launches(basis, aux, nuclei, Gam, *, lP: int | None = None
                    ) -> list[HessLaunch]:
    """One `d2eri3_` launch per three-center class of `basis` x `aux`
    (`eri3_classes`' staging), each in its default form; `lP` keeps the
    classes with that aux `l` only."""
    out = []
    for c in eri3_classes(basis, aux):
        if lP is not None and c.lP != lP:
            continue
        arrs = eri3_arrays(c, basis, aux, nuclei) | {
            "M": np.asarray(Gam, dtype=np.float64)}
        out.append(HessLaunch(build_d2eri3_kernel((c.la, c.lb, c.lP)),
                              c.n_cell, arrs, {
            "n_cell": c.n_cell, "n_bra_rec": c.n_bra_rec,
            "n_aux_rec": c.n_aux_rec, "n_ao": basis.n_ao,
            "n_aux": aux.n_ao, "n_atom": len(nuclei)}))
    return out


def hkx_launches(basis, nuclei, D, classes=None, *, pairs=None,
                 lowstack: bool | None = None) -> list[HessLaunch]:
    """One `hkx_` launch per K class `(la, lc, lb, ld)` (#691), every
    ordered bra record of `(la, lc)` against every ordered ket record of
    `(lb, ld)`, the kets grouped by atom pair
    (`hessian.host.exchange_hessian_records`); one warp of `THREADS` lanes
    per bra record (`kernel_hkx.launch_shape`). `classes` defaults to every
    class the basis reaches; `pairs` maps a class to `(bra_pairs,
    ket_pairs)`, the shell pairs a witness launches (all when absent);
    `lowstack` forces a form (the default form otherwise)."""
    if classes is None:
        ls = sorted({s.l for s in basis.shells})
        classes = [(a, c, b, d) for a in ls for c in ls for b in ls
                   for d in ls]
    out = []
    for am in classes:
        kern = build_hkx_kernel(am, lowstack=lowstack)
        bp, kp = (pairs or {}).get(am, (None, None))
        bra, ket, seg = hessian_host.exchange_hessian_records(
            basis, nuclei, am, bra_pairs=bp, ket_pairs=kp)
        types = {p.name: p.type for p in kern.params if p.array}
        arrs = {k: np.asarray(v, dtype=np.int32 if types[k] is Scalar.I32
                              else np.float64)
                for k, v in {**bra, **ket, **seg}.items() if k in types}
        arrs["M"] = np.asarray(D, dtype=np.float64)
        n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
        if not n_bra or not n_ket:
            continue
        out.append(HessLaunch(kern, n_bra, arrs, {
            "n_bra": n_bra, "n_ket": n_ket, "n_seg": len(seg["seg_b"]),
            "n_ao": basis.n_ao, "n_atom": len(nuclei)},
            warp_per_record=True))
    return out


def hess_steps(launches: Sequence[HessLaunch]) -> list[Step]:
    """Each launch as a `harness_driver.Step`, one thread per record,
    `THREADS` lanes -- or for `hkx_` one warp of `THREADS` lanes per bra
    record; its OUT carried."""
    out = []
    for x in launches:
        shape = hkx_launch_shape if x.warp_per_record else launch_shape
        grid, block = shape(x.n_rec, THREADS)
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
