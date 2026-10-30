"""The DF gradient kernels launched in the interpreter, and their arrays.

    metric_arrays(cls, aux, nuclei)             -> gmetric's arrays by name
    eri3_arrays(cls, basis, aux, nuclei)        -> geri3's arrays by name
    run_metric(aux, nuclei, W, *, boys, ...)    -> (G, classes run)
    run_eri3(basis, aux, nuclei, Gam, *, boys, ...) -> (G, classes run)
    run_d2metric(aux, nuclei, W, *, boys, ...)   -> (H, classes run)
    run_d2eri3(basis, aux, nuclei, Gam, *, boys, ...) -> (H, classes run)
    xeri3_arrays(cls, basis, aux, nuclei, keep)  -> xeri3's arrays by name
    run_xeri3(basis, aux, nuclei, *, boys, aux_shells=, ...) -> (X, classes run)

The staging is the d0 kernels' (`metric_classes`, `eri3_classes`), `s` passed
as `s_aux` (`kernel_metric.PARAM_OF`), plus `gradient.host_df`'s atoms and
exponents. One launch per class into one `G` (or `H`), zeroed here once.
`lowstack=` is the d1 builders' (#613): `None` their gate, `True` /
`False` either form forced. `run_eri3(engine=Engine.HGP)` launches the
HGP sibling `gradient.kernel_hgeri3` on the same arrays (it has one form, so
`lowstack` must stay `None` there).
`False` either form forced. `run_metric`'s `engine=Engine.HGP` launches
the HGP twin (`gradient.kernel_hgmetric`) on the same arrays instead.
Shared by test_kernel_gmetric.py, test_kernel_geri3.py,
test_kernel_d2metric.py, test_kernel_d2eri3.py and test/grad_df_driver.py
(which stages the same arrays for the card). `run_d2metric` is the d2
metric kernel (`hessian.kernel_d2metric`) on `run_metric`'s arrays, and
`run_d2eri3` the d2 three-center kernel (`hessian.kernel_d2eri3`) on
`run_eri3`'s, `H` in place of `G`. `run_xeri3` is the uncontracted d1
kernel (`gradient.kernel_xeri3`, #692) on `run_eri3`'s arrays less the
atoms, over one aux block and, optionally, a subset of its cells.
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
from felizjueves.gradient.kernel_hgeri3 import build_hgeri3_kernel
from felizjueves.gradient.kernel_hgmetric import build_hgmetric_kernel
from felizjueves.gradient.kernel_xeri3 import build_xeri3_kernel
from felizjueves.hessian.kernel_d2eri3 import build_d2eri3_kernel
from felizjueves.hessian.kernel_d2metric import build_d2metric_kernel
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.spec import Engine

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
               threads: int = 7, G: np.ndarray | None = None,
               lowstack: bool | None = None, engine: Engine = Engine.MMD):
    want = None if classes is None else set(classes)
    G = np.zeros((len(nuclei), 3)) if G is None else G
    ran = []
    for cls in metric_classes(aux):
        if want is not None and (cls.lP, cls.lQ) not in want:
            continue
        grid, block = launch_shape(cls.n_pair, threads)
        kernel = (build_hgmetric_kernel((cls.lP, cls.lQ))
                  if engine is Engine.HGP else
                  build_gmetric_kernel((cls.lP, cls.lQ), lowstack=lowstack))
        G = Interpreter(kernel,
                        launch=Launch(grid, block, 1), boys=boys).run(
            n_pair=cls.n_pair, n_rec=cls.n_rec, n_aux=aux.n_ao,
            n_atom=len(G), M=np.asarray(W, dtype=float), G=G,
            **metric_arrays(cls, aux, nuclei))["G"]
        ran.append((cls.lP, cls.lQ))
    return G, tuple(ran)


def run_eri3(basis, aux, nuclei, Gam, *, boys,
             classes: Iterable | None = None, threads: int = 7,
             G: np.ndarray | None = None, lowstack: bool | None = None,
             engine: Engine = Engine.MMD):
    if engine is Engine.HGP and lowstack is not None:
        raise ValueError("the HGP geri3 sibling has one form")
    want = None if classes is None else set(classes)
    G = np.zeros((len(nuclei), 3)) if G is None else G
    ran = []
    for cls in eri3_classes(basis, aux):
        c = (cls.la, cls.lb, cls.lP)
        if want is not None and c not in want:
            continue
        grid, block = launch_shape(cls.n_cell, threads)
        kernel = (build_hgeri3_kernel(c) if engine is Engine.HGP
                  else build_geri3_kernel(c, lowstack=lowstack))
        G = Interpreter(kernel,
                        launch=Launch(grid, block, 1), boys=boys).run(
            n_cell=cls.n_cell, n_bra_rec=cls.n_bra_rec,
            n_aux_rec=cls.n_aux_rec, n_ao=basis.n_ao, n_aux=aux.n_ao,
            n_atom=len(G), M=np.asarray(Gam, dtype=float), G=G,
            **eri3_arrays(cls, basis, aux, nuclei))["G"]
        ran.append(c)
    return G, tuple(ran)


def xeri3_arrays(cls: Eri3Class, basis, aux, nuclei,
                 keep: Iterable[int] | None = None) -> dict[str, np.ndarray]:
    """`xeri3`'s arrays: `geri3`'s less the atoms, the per-cell ones
    restricted to the cells `keep` (indices into `cls.cells`, default all;
    the bra and aux record arrays stay whole)."""
    arrs = eri3_arrays(cls, basis, aux, nuclei)
    for k in ("atom_a", "atom_b", "atom_p"):
        del arrs[k]
    if keep is not None:
        idx = np.asarray(list(keep), dtype=np.int64)
        for k in Eri3Class.CELL_ARRAYS:
            arrs[k] = arrs[k][idx]
    return arrs


def run_xeri3(basis, aux, nuclei, *, boys, classes: Iterable | None = None,
              aux_shells: range | None = None, cells: Iterable | None = None,
              threads: int = 7, X: np.ndarray | None = None):
    """The uncontracted `xeri3` kernels over ONE AUX BLOCK: the aux shells
    `aux_shells` (a contiguous range, default all), so `X` is `[3, n_ao,
    n_ao, n_blk]` from `aux0 = aux.offsets[aux_shells.start]`. `cells`, if
    given, further restricts the launch to those `(i, j, n)` cells (basis
    order, `i <= j`). `X` (default zeros) is passed in and returned."""
    sh = range(len(aux.shells)) if aux_shells is None else aux_shells
    aux0 = aux.offsets[sh.start] if len(sh) else 0
    n_blk = sum(aux.shells[n].n_cart for n in sh)
    want = None if classes is None else set(classes)
    only = None if cells is None else set(cells)
    X = np.zeros((3, basis.n_ao, basis.n_ao, n_blk)) if X is None else X
    ran = []
    for cls in eri3_classes(basis, aux):
        c = (cls.la, cls.lb, cls.lP)
        if want is not None and c not in want:
            continue
        keep = [i for i, (a, b, n) in enumerate(cls.cells) if n in sh and (
            only is None or (min(a, b), max(a, b), n) in only)]
        if not keep:
            continue
        grid, block = launch_shape(len(keep), threads)
        X = Interpreter(build_xeri3_kernel(c),
                        launch=Launch(grid, block, 1), boys=boys).run(
            n_cell=len(keep), n_bra_rec=cls.n_bra_rec,
            n_aux_rec=cls.n_aux_rec, n_ao=basis.n_ao, n_blk=n_blk,
            aux0=aux0, X=X,
            **xeri3_arrays(cls, basis, aux, nuclei, keep))["X"]
        ran.append(c)
    return X, tuple(ran)


def run_d2metric(aux, nuclei, W, *, boys, classes: Iterable | None = None,
                threads: int = 7, H: np.ndarray | None = None):
    want = None if classes is None else set(classes)
    n = len(nuclei)
    H = np.zeros((n, 3, n, 3)) if H is None else H
    ran = []
    for cls in metric_classes(aux):
        if want is not None and (cls.lP, cls.lQ) not in want:
            continue
        grid, block = launch_shape(cls.n_pair, threads)
        H = Interpreter(build_d2metric_kernel((cls.lP, cls.lQ)),
                        launch=Launch(grid, block, 1), boys=boys).run(
            n_pair=cls.n_pair, n_rec=cls.n_rec, n_aux=aux.n_ao,
            n_atom=len(H), M=np.asarray(W, dtype=float), H=H,
            **metric_arrays(cls, aux, nuclei))["H"]
        ran.append((cls.lP, cls.lQ))
    return H, tuple(ran)


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
