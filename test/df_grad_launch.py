"""The DF gradient kernels launched in the interpreter, and their arrays.

    metric_arrays(cls, aux, nuclei)             -> gmetric's arrays by name
    eri3_arrays(cls, basis, aux, nuclei)        -> geri3's arrays by name
    run_metric(aux, nuclei, W, *, boys, ...)    -> (G, classes run)
    run_eri3(basis, aux, nuclei, Gam, *, boys, ...) -> (G, classes run)
    run_d2eri3(basis, aux, nuclei, Gam, *, boys, ...) -> (H, classes run)

The staging is the d0 kernels' (`metric_classes`, `eri3_classes`), `s` passed
as `s_aux` (`kernel_metric.PARAM_OF`), plus `gradient.host_df`'s atoms and
exponents. One launch per class into one `G` (the Hessian's `H`), zeroed
here once. Shared by test_kernel_gmetric.py, test_kernel_geri3.py,
test_kernel_d2eri3.py and test/grad_df_driver.py (which stages the same
arrays for the card).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from felizjueves.eri.density_fit.eri3 import Eri3Class, eri3_classes
from felizjueves.eri.density_fit.kernel_metric import PARAM_OF
from felizjueves.eri.density_fit.metric import MetricClass, metric_classes
from felizjueves.gradient import host_df
from felizjueves.gradient.kernel_geri3 import build_geri3_kernel
from felizjueves.gradient.kernel_gmetric import (
    build_gmetric_kernel,
    launch_shape,
)
from felizjueves.hessian.kernel_d2eri3 import build_d2eri3_kernel
from felizjueves.ir.interp import Interpreter, Launch

I32 = ("rec0", "rec1", "off_", "atom_")


def _np(name: str, v) -> np.ndarray:
    return np.array(v, dtype=np.int32 if any(s in name for s in I32)
                    else float)


def metric_arrays(cls: MetricClass, aux, nuclei) -> dict[str, np.ndarray]:
    out = {PARAM_OF.get(k, k): _np(k, v) for k, v in cls.arrays().items()}
    out |= {k: _np(k, v) for k, v in
            host_df.metric_records(cls, aux, nuclei).items()}
    return out


def eri3_arrays(cls: Eri3Class, basis, aux, nuclei
                ) -> dict[str, np.ndarray]:
    out = {PARAM_OF.get(k, k): _np(k, v) for k, v in cls.arrays().items()}
    out |= {k: _np(k, v) for k, v in
            host_df.eri3_records(cls, basis, aux, nuclei).items()}
    return out


def run_metric(aux, nuclei, W, *, boys, classes: Iterable | None = None,
               threads: int = 7, G: np.ndarray | None = None):
    want = None if classes is None else set(classes)
    G = np.zeros((len(nuclei), 3)) if G is None else G
    ran = []
    for cls in metric_classes(aux):
        if want is not None and (cls.lP, cls.lQ) not in want:
            continue
        grid, block = launch_shape(cls.n_pair, threads)
        G = Interpreter(build_gmetric_kernel((cls.lP, cls.lQ)),
                        launch=Launch(grid, block, 1), boys=boys).run(
            n_pair=cls.n_pair, n_rec=cls.n_rec, n_aux=aux.n_ao,
            n_atom=len(G), M=np.asarray(W, dtype=float), G=G,
            **metric_arrays(cls, aux, nuclei))["G"]
        ran.append((cls.lP, cls.lQ))
    return G, tuple(ran)


def run_eri3(basis, aux, nuclei, Gam, *, boys,
             classes: Iterable | None = None, threads: int = 7,
             G: np.ndarray | None = None):
    want = None if classes is None else set(classes)
    G = np.zeros((len(nuclei), 3)) if G is None else G
    ran = []
    for cls in eri3_classes(basis, aux):
        c = (cls.la, cls.lb, cls.lP)
        if want is not None and c not in want:
            continue
        grid, block = launch_shape(cls.n_cell, threads)
        G = Interpreter(build_geri3_kernel(c), launch=Launch(grid, block, 1),
                        boys=boys).run(
            n_cell=cls.n_cell, n_bra_rec=cls.n_bra_rec,
            n_aux_rec=cls.n_aux_rec, n_ao=basis.n_ao, n_aux=aux.n_ao,
            n_atom=len(G), M=np.asarray(Gam, dtype=float), G=G,
            **eri3_arrays(cls, basis, aux, nuclei))["G"]
        ran.append(c)
    return G, tuple(ran)


def run_d2eri3(basis, aux, nuclei, Gam, *, boys,
               classes: Iterable | None = None, threads: int = 7,
               H: np.ndarray | None = None):
    """The d2 `(mn|P)` kernels on `geri3`'s arrays, into `H[n_atom, 3,
    n_atom, 3]`."""
    want = None if classes is None else set(classes)
    n = len(nuclei)
    H = np.zeros((n, 3, n, 3)) if H is None else H
    ran = []
    for cls in eri3_classes(basis, aux):
        c = (cls.la, cls.lb, cls.lP)
        if want is not None and c not in want:
            continue
        grid, block = launch_shape(cls.n_cell, threads)
        H = Interpreter(build_d2eri3_kernel(c),
                        launch=Launch(grid, block, 1), boys=boys).run(
            n_cell=cls.n_cell, n_bra_rec=cls.n_bra_rec,
            n_aux_rec=cls.n_aux_rec, n_ao=basis.n_ao, n_aux=aux.n_ao,
            n_atom=n, M=np.asarray(Gam, dtype=float), H=H,
            **eri3_arrays(cls, basis, aux, nuclei))["H"]
        ran.append(c)
    return H, tuple(ran)
