"""The d1 kernels as `gradient.rhf.Runners`, run in the interpreter (#565).

    interpreted_runners(boys=, lanes=)   -> Runners
    g1e_gradient(target, basis, nuclei, M, *, boys)      gs_ / gt_ / gv_
    gjab_gradient(basis, nuclei, D, *, boys)             [p|q]@L+1 -> gjab_
    gk_gradient(basis, nuclei, D, *, boys, lanes=)       gk_

Every term of the RHF gradient from the GENERATED kernels, with no compiler
and no card: one launch per class into one zeroed `G` (the kernels'
precondition, `kernel_g1e.ZEROED_G`), classes in sorted order, each launch's
arrays packed as `test/grad_driver.py` packs them for a card:

- `gs_` / `gt_` / `gv_{la}{lb}`: every record of the class's shell pairs
  `i <= j` (`gradient.host.records`); `gv_` gets the nuclei (`Z`, `Cnuc`)
  and computes the Hellmann-Feynman term itself. The default form
  (`kernel_g1e.g1e_lowstack`'s rule) is the one built.
- THE J CHAIN: per bra class `(la, lb)`, `J_p` at Hermite order `L + 1`
  from the interpreted d0 `[p|q]` kernel of `kernel_gjab.gjab_pq_spec`'s
  class (the bra raised by one, an ordinary `Target.J` class run by
  `eri.j_coulomb.drive.KernelJP`) over `gradient.host.coulomb_kets`, one
  ket class `lq` at a time; then `gjab_{la}{lb}` reading it. The host's
  `J_p` is not used.
- `gk_{la}{lc}_{lb}{ld}`: every K class the basis's `l`s reach, every
  ordered bra record against every ordered ket record
  (`gradient.host.exchange_records`), one warp of `lanes` per bra record.

BOYS IS INJECTED (`boys=`), as the interpreter requires: nothing here
imports `reference/`. The interpreted `G` is the sum the card's `atomicAdd`s
make, in a fixed order (the interpreter's), so it is reproducible here and
not on a card.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache

import numpy as np

from felizjueves.ao.shell import Basis
from felizjueves.eri.j_coulomb.drive import KernelJP
from felizjueves.gradient import host
from felizjueves.gradient.kernel_g1e import build_g1e_kernel, launch_shape
from felizjueves.gradient.kernel_gjab import build_gjab_kernel, n_p1
from felizjueves.gradient.kernel_gk import build_gk_kernel
from felizjueves.gradient.kernel_gk import launch_shape as gk_launch_shape
from felizjueves.gradient.rhf import Matrix, Runners
from felizjueves.ir import Kernel, Scalar
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd.hermite import hermite_order
from felizjueves.one_electron.potential import Nucleus
from felizjueves.spec import Target

Boys = Callable[[int, float], float]
THREADS = 32  # threads per block for the per-record kernels; not numerical


@cache
def _g1e(target: Target, la: int, lb: int) -> Kernel:
    return build_g1e_kernel((target, la, lb))


@cache
def _gjab(la: int, lb: int) -> Kernel:
    return build_gjab_kernel((la, lb))


@cache
def _gk(am: tuple[int, int, int, int]) -> Kernel:
    return build_gk_kernel(am)


def _pair_classes(basis: Basis) -> list[tuple[int, int]]:
    sh = basis.shells
    return sorted({(a.l, b.l) for k, a in enumerate(sh) for b in sh[k:]})


def _arrays(rec: dict[str, list], kern: Kernel) -> dict[str, np.ndarray]:
    """`rec`'s lists the kernel takes, typed by its parameters."""
    params = {p.name: p for p in kern.params if p.array}
    return {k: np.asarray(v, dtype=np.int32 if params[k].type is Scalar.I32
                          else np.float64)
            for k, v in rec.items() if k in params}


def _zero(nuclei: Sequence[Nucleus]) -> np.ndarray:
    return np.zeros((len(nuclei), 3))


def g1e_gradient(target: Target, basis: Basis, nuclei: Sequence[Nucleus],
                 M: Matrix, *, boys: Boys) -> np.ndarray:
    """`SUM M dX/dR`, `X` = `S`, `T` or `V` (Hellmann-Feynman included)."""
    G = _zero(nuclei)
    Mx = np.asarray(M, dtype=np.float64)
    for la, lb in _pair_classes(basis):
        kern = _g1e(target, la, lb)
        rec = host.records(basis, nuclei, la, lb)
        arrs = _arrays(rec, kern)
        if target is Target.V:
            arrs["Z"] = np.array([float(z) for z, _ in nuclei])
            arrs["Cnuc"] = np.array([[float(x) for x in c] for _, c in nuclei])
        n = len(rec["s_ab"])
        grid, block = launch_shape(n, THREADS)
        G = Interpreter(kern, launch=Launch(grid, block, 1), boys=boys).run(
            n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), M=Mx, G=G,
            **arrs)["G"]
    return G


def _chain_jp(la: int, lb: int, rec: dict[str, list], kets,
              jp: KernelJP) -> np.ndarray:
    """`J_p` at `L + 1` per record, from the interpreted `[p|q]` kernel of
    the class with the bra raised by one, one ket class at a time."""
    bras = [((rec["p_x"][r], rec["p_y"][r], rec["p_z"][r]), rec["zeta_ab"][r])
            for r in range(len(rec["s_ab"]))]
    lq_of = {len(hermite_order(L)): L for L in range(16)}
    by_lq: dict[int, list] = {}
    for k in kets:
        by_lq.setdefault(lq_of[len(k.qs)], []).append((k.Q, k.eta, k.dq))
    out = np.zeros((len(bras), n_p1(la, lb)))
    for lq, ks in sorted(by_lq.items()):
        out += jp((la + 1, lb), (lq, 0), bras, ks)
    return out


def gjab_gradient(basis: Basis, nuclei: Sequence[Nucleus], D: Matrix, *,
                  boys: Boys) -> np.ndarray:
    """`dE_J/dR` through the interpreted chain `[p|q]@L+1 -> gjab_`."""
    G = _zero(nuclei)
    Dx = np.asarray(D, dtype=np.float64)
    kets = host.coulomb_kets(basis, D)
    jp = KernelJP(boys=boys)
    for la, lb in _pair_classes(basis):
        kern = _gjab(la, lb)
        rec = host.records(basis, nuclei, la, lb)
        n = len(rec["s_ab"])
        arrs = _arrays(rec, kern) | {
            "XPA": np.array([rec[f"pa_{a}"] for a in "xyz"]).T.copy(),
            "XPB": np.array([rec[f"pb_{a}"] for a in "xyz"]).T.copy(),
            "s": np.asarray(rec["s_ab"], dtype=np.float64),
            "row": np.arange(n, dtype=np.int32),
            "Jp": _chain_jp(la, lb, rec, kets, jp)}
        grid, block = launch_shape(n, THREADS)
        G = Interpreter(kern, launch=Launch(grid, block, 1)).run(
            n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), n_rows=n, M=Dx,
            G=G, **arrs)["G"]
    return G


def gk_gradient(basis: Basis, nuclei: Sequence[Nucleus], D: Matrix, *,
                boys: Boys, lanes: int = 4) -> np.ndarray:
    """`dE_K/dR`, `E_K = 1/2 SUM D_ab D_cd (ac|bd)`, every K class."""
    G = _zero(nuclei)
    Dx = np.asarray(D, dtype=np.float64)
    ls = sorted({s.l for s in basis.shells})
    for am in [(a, c, b, d) for a in ls for c in ls for b in ls for d in ls]:
        kern = _gk(am)
        bra, ket = host.exchange_records(basis, nuclei, am)
        n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
        if not (n_bra and n_ket):
            continue
        grid, block = gk_launch_shape(n_bra, lanes)
        G = Interpreter(kern, launch=Launch(grid, block, lanes),
                        boys=boys).run(
            n_bra=n_bra, n_ket=n_ket, n_ao=basis.n_ao, n_atom=len(nuclei),
            M=Dx, G=G, **_arrays({**bra, **ket}, kern))["G"]
    return G


def interpreted_runners(*, boys: Boys, lanes: int = 4) -> Runners:
    """Every electronic term of `gradient.rhf` from the interpreted kernels
    (module docstring)."""
    return Runners(
        overlap=lambda b, n, M: g1e_gradient(Target.S, b, n, M, boys=boys),
        kinetic=lambda b, n, M: g1e_gradient(Target.T, b, n, M, boys=boys),
        potential=lambda b, n, M: g1e_gradient(Target.V, b, n, M, boys=boys),
        coulomb=lambda b, n, M: gjab_gradient(b, n, M, boys=boys),
        exchange=lambda b, n, M: gk_gradient(b, n, M, boys=boys,
                                             lanes=lanes))


__all__ = ["g1e_gradient", "gjab_gradient", "gk_gradient",
           "interpreted_runners"]
