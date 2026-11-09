"""The HGP two-electron Hessian kernels run in the interpreter.

    hgp2_run(spec, diag, bra, ket, n_atom, D, H, *, boys, threads=)  one launch
    hgp2_hessian(target, basis, nuclei, D, *, boys, threads=)        whole basis

`hessian.kernel_hgp2`'s `d2f_` (`Target.F`), `d2hj_` (`J` on HGP) and
`d2hk_` (`K` on HGP) over `gradient.interpreted.gf_launches`' stream --
every unique quartet of the basis once, the canonical half and `diag`
as `gf_` reads them -- into one zeroed `H[n_atom, 3, n_atom, 3]`. Each
launch's class is the ERI's AO order; for `K` that tuple is the K class
`(la, lc | lb, ld)` (`kernel_hgp2`'s docstring), so one stream serves all
three targets.

BOYS IS INJECTED (`boys=`), as the interpreter requires: nothing here
imports `reference/`. The interpreted `H` is the sum the card's
`atomicAdd`s make, in the interpreter's fixed order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache

import numpy as np

from felizjueves.ao.shell import Basis
from felizjueves.gradient import kernel_gf as GF
from felizjueves.gradient.interpreted import gf_launches
from felizjueves.hessian import kernel_hgp2 as H2
from felizjueves.ir import Kernel, Scalar
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.one_electron.potential import Nucleus
from felizjueves.spec import Spec, Target

Boys = Callable[[int, float], float]

__all__ = ["hgp2_hessian", "hgp2_run"]


@cache
def _kernel(spec: Spec) -> Kernel:
    return H2.build_d2_kernel(spec)


def hgp2_run(spec: Spec, diag: int, bra: GF.Packed, ket: GF.Packed,
             n_atom: int, D: np.ndarray, H: np.ndarray, *, boys: Boys,
             threads: int = 4, kernel: Kernel | None = None) -> np.ndarray:
    """One launch of `spec`'s kernel over packed sides `bra` and `ket`; `H`
    (`n_atom x 3 x n_atom x 3`) carried in and returned. `kernel` replaces
    the built one (a test's mutation)."""
    kern = _kernel(spec) if kernel is None else kernel
    arrs = {"n_bra": bra.n, "n_ket": ket.n, "k_ab": bra.degree,
            "k_cd": ket.degree, "n_atom": n_atom, "n_ao": D.shape[0],
            "diag": diag, "bra_prim": bra.prim, "AB": bra.sep,
            "bra_sh": bra.sh, "bra_ao": bra.ao, "bra_at": bra.at,
            "ket_prim": ket.prim, "CD": ket.sep, "ket_sh": ket.sh,
            "ket_ao": ket.ao, "ket_at": ket.at, "D": D, "H": H}
    args = {}
    for p in kern.params:
        v = arrs[p.name]
        if p.array and not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.int32 if p.type is Scalar.I32
                           else np.float64)
        args[p.name] = v
    grid, block = H2.launch_shape(bra.n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1),
                       boys=boys).run(**args)["H"]


def hgp2_hessian(target: Target, basis: Basis, nuclei: Sequence[Nucleus],
                 D, *, boys: Boys, threads: int = 4) -> np.ndarray:
    """`d2 E / dR dR` of `target` (F: `E_J - 1/2 E_K`; J; K) from the
    interpreted HGP kernels, every launch of `gf_launches` into one zeroed
    `H[n_atom, 3, n_atom, 3]`."""
    n = len(nuclei)
    H = np.zeros((n, 3, n, 3))
    Dx = np.asarray(D, dtype=np.float64)
    if Dx.shape != (basis.n_ao, basis.n_ao) or not np.array_equal(Dx, Dx.T):
        raise ValueError(f"D is not a symmetric {basis.n_ao} x {basis.n_ao} "
                         "matrix (the derived weights assume D_ab = D_ba)")
    for am, diag, pb, pk in gf_launches(basis, nuclei):
        H = hgp2_run(H2.d2_spec(target, am), diag, pb, pk, n, Dx, H,
                     boys=boys, threads=threads)
    return H
