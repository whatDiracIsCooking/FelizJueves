"""Check the DF gradients: ``python experimental/mmd/gradient/df_check.py``.

Prints five tables and exits nonzero on a miss:

1. ``(P|Q)`` and ``(mn|P)`` against libcint (``int2c2e``, ``int3c2e``, both
   Cartesian), the DF ``J`` and ``K`` against PySCF's ``with_df.get_jk``,
   and the DF-RHF energy against PySCF's;
2. their nuclear derivatives against libcint's (``int2c2e_ip1``,
   ``int3c2e_ip1`` on ``m`` and ``n``, ``int3c2e_ip2`` on ``P``), assembled
   over the centres on each atom;
3. translational invariance, ``sum_atoms dX/dR = 0`` (no centre inferred);
4. every derivative, and ``dE_J``, ``dE_K`` at fixed ``D``, against central
   differences of this module's own integrals (``h = 1e-4`` bohr);
5. the DF-RHF gradient, auxiliary response included, against PySCF's
   ``df.grad.rhf`` at the converged DF density.

Molecules: ``check.py``'s distorted water in STO-3G and 6-31G and HCN in
STO-3G, each with ``cc-pvdz-jkfit`` cut to its s and p shells (Cartesian;
``AUX_LMAX``), so the whole check stays at p. The finite
differences run on STO-3G only.

Needs the ``oracle`` group (PySCF): ``uv sync --group oracle``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check import CASES, FD_H, _atom_mask, _gate, _rel  # noqa: E402
from check import failures as _failures  # noqa: E402
from df import (  # noqa: E402
    df_coulomb,
    df_exchange,
    df_j_gradient,
    df_k_gradient,
    df_rhf_energy,
    df_rhf_gradient,
    metric,
    three_center,
)
from ints import one_electron  # noqa: E402
from mmd import displaced, shells_of  # noqa: E402

AUXBASIS = "cc-pvdz-jkfit"
AUX_LMAX = 1  # the aux basis cut to s and p, as the orbital basis is
INT_TOL = 1e-12  # of max|X|, against libcint / PySCF
DERIV_TOL = 1e-12  # of max|dX|, against libcint
INVAR_TOL = 1e-12  # of max|dX|
FD_TOL = 1e-7  # of max|dX|: O(h^2) truncation
GRAD_TOL = 1e-10  # Hartree/bohr, absolute
E_TOL = 1e-10  # Hartree


def _aux_basis(mol):
    """``AUXBASIS`` per element with every shell above ``AUX_LMAX`` dropped."""
    from pyscf import gto

    return {
        el: [b for b in gto.load(AUXBASIS, el) if b[0] <= AUX_LMAX]
        for el in {mol.atom_symbol(i) for i in range(mol.natm)}
    }


def _aux_mask(aux):
    mask = np.zeros((aux.natm, aux.nao), dtype=bool)
    for A, (_, _, p0, p1) in enumerate(aux.aoslice_by_atom()):
        mask[A, p0:p1] = True
    return mask


def _libcint_derivs(mol, aux):
    """``d(P|Q)`` and ``d(mn|P)`` from libcint's ip integrals; d/dA = -nabla."""
    from pyscf.df import incore

    nao, naux = mol.nao, aux.nao
    am, xm = _atom_mask(mol), _aux_mask(aux)
    ip2c = aux.intor("int2c2e_ip1")  # (nabla P|Q)
    ip1 = incore.aux_e2(mol, aux, "int3c2e_ip1", aosym="s1", comp=3)
    ip1 = ip1.reshape(3, nao, nao, naux)  # (nabla m n|P)
    ip2 = incore.aux_e2(mol, aux, "int3c2e_ip2", aosym="s1", comp=3)
    ip2 = ip2.reshape(3, nao, nao, naux)  # (m n|nabla P)
    dM = np.zeros((mol.natm, 3, naux, naux))
    d3 = np.zeros((mol.natm, 3, nao, nao, naux))
    for A in range(mol.natm):
        m, x = am[A], xm[A]
        dM[A][:, x, :] -= ip2c[:, x, :]
        dM[A][:, :, x] -= ip2c[:, x, :].transpose(0, 2, 1)
        d3[A][:, m] -= ip1[:, m]
        d3[A][:, :, m] -= ip1.transpose(0, 2, 1, 3)[:, :, m]
        d3[A][:, :, :, x] -= ip2[:, :, :, x]
    return dM, d3


def _fd(shells, aux, natm, D):
    """Central differences of ``(P|Q)``, ``(mn|P)``, ``E_J`` and ``E_K``
    (``D`` fixed) in every coordinate."""
    out = [None, None, np.zeros((natm, 3)), np.zeros((natm, 3))]
    for A in range(natm):
        for k in range(3):
            vals = []
            for h in (FD_H, -FD_H):
                sh, ax = displaced(shells, A, k, h), displaced(aux, A, k, h)
                M, _ = metric(ax, natm, deriv=False)
                e3, _ = three_center(sh, ax, natm, deriv=False)
                ej = 0.5 * np.sum(D * df_coulomb(e3, M, D))
                ek = -0.25 * np.sum(D * df_exchange(e3, M, D))
                vals.append((M, e3, ej, ek))
            d = [(p - m) / (2 * FD_H) for p, m in zip(*vals, strict=True)]
            for n in (0, 1):
                if out[n] is None:
                    out[n] = np.zeros((natm, 3) + d[n].shape)
                out[n][A, k] = d[n]
            out[2][A, k], out[3][A, k] = d[2], d[3]
    return out


def main() -> int:
    from pyscf import df, gto, scf

    rows = {n: [] for n in range(1, 6)}
    t0 = time.time()
    for name, atom, basis, do_fd in CASES:
        label = f"{name}/{basis}"
        mol = gto.M(atom=atom, basis=basis, cart=True, verbose=0)
        auxbasis = _aux_basis(mol)
        auxmol = df.make_auxmol(mol, auxbasis)
        shells, aux = shells_of(mol), shells_of(auxmol)
        Z = mol.atom_charges().astype(float)
        R = mol.atom_coords()
        mf = scf.RHF(mol).density_fit(auxbasis=auxbasis).run(conv_tol=1e-12)
        assert mf.with_df.auxmol.nao == auxmol.nao
        D = mf.make_rdm1()
        grad = mf.nuc_grad_method()
        W = grad.make_rdm1e()

        S, T, V, *_ = one_electron(shells, Z, R)
        M, dM = metric(aux, mol.natm)
        e3, d3 = three_center(shells, aux, mol.natm)

        # 1. integrals
        from pyscf.df import incore

        vj, vk = mf.with_df.get_jk(D)
        for nm, mine, ref in (
            ("(P|Q)", M, auxmol.intor("int2c2e")),
            ("(mn|P)", e3, incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1")
             .reshape(e3.shape)),
            ("J_DF", df_coulomb(e3, M, D), vj),
            ("K_DF", df_exchange(e3, M, D), vk),
        ):
            rows[1].append((label, nm, _gate(f"1 {label} {nm}", _rel(mine, ref),
                                             INT_TOL)))
        e = abs(df_rhf_energy(S, T, V, e3, M, D, Z, R) - mf.e_tot)
        rows[1].append((label, "E_DF-RHF (abs)", _gate(f"1 {label} E", e, E_TOL)))

        # 2. derivatives against libcint
        rM, r3 = _libcint_derivs(mol, auxmol)
        for nm, mine, ref in (("d(P|Q)", dM, rM), ("d(mn|P)", d3, r3)):
            err = _gate(f"2 {label} {nm}", _rel(mine, ref), DERIV_TOL)
            rows[2].append((label, nm, err))

        # 3. translational invariance
        for nm, d in (("d(P|Q)", dM), ("d(mn|P)", d3)):
            err = float(np.abs(d.sum(axis=0)).max() / np.abs(d).max())
            rows[3].append((label, nm, _gate(f"3 {label} {nm}", err, INVAR_TOL)))

        # 4. finite differences of our own integrals, D fixed
        if do_fd:
            fd = _fd(shells, aux, mol.natm, D)
            mine = (dM, d3, df_j_gradient(e3, d3, M, dM, D),
                    df_k_gradient(e3, d3, M, dM, D))
            for nm, a, ref in zip(("d(P|Q)", "d(mn|P)", "dE_J", "dE_K"), mine, fd,
                                  strict=True):
                err = float(np.abs(a - ref).max() / np.abs(a).max())
                rows[4].append((label, nm, _gate(f"4 {label} {nm}", err, FD_TOL)))

        # 5. the DF-RHF gradient
        g, _ = df_rhf_gradient(shells, aux, Z, R, D, W)
        ref = grad.kernel()
        err = float(np.abs(g - ref).max())
        rows[5].append((label, "total", _gate(f"5 {label} grad", err, GRAD_TOL)))
        rows[5].append((label, "max|grad|", f"{np.abs(ref).max():.3e}"))

    titles = {
        1: f"1. DF integrals against libcint / PySCF, aux {AUXBASIS} s+p (of max|X|; "
           "E absolute)",
        2: "2. nuclear derivatives against libcint (of max|dX|)",
        3: "3. translational invariance |sum_A dX/dR_A| (of max|dX|)",
        4: f"4. analytic against central differences, h = {FD_H} (of max|dX|)",
        5: "5. DF-RHF gradient against pyscf df.grad.rhf (Hartree/bohr, absolute)",
    }
    for n, title in titles.items():
        print(f"\n{title}")
        for label, nm, val in rows[n]:
            print(f"  {label:14s} {nm:16s} {val}")
    print(f"\n{time.time() - t0:.0f} s")
    if _failures:
        print("\nFAIL:\n  " + "\n  ".join(_failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
