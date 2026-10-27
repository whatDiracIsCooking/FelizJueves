"""Check the MMD gradients: ``python experimental/mmd/gradient/check.py``.

Prints five tables and exits nonzero on a miss:

1. the integrals ``S, T, V, (ab|cd)`` against libcint (``int1e_ovlp``,
   ``int1e_kin``, ``int1e_nuc``, ``int2e``, all Cartesian);
2. their nuclear derivatives against libcint's (``int1e_ipovlp``,
   ``int1e_ipkin``, ``int2e_ip1`` assembled over the centres on each atom;
   ``dV`` as PySCF's ``hcore_generator`` minus ``dT``, so it carries the
   operator term), and ``dJ``, ``dK`` from them for the RHF density;
3. translational invariance, ``sum_atoms dX/dR = 0``, which holds only if the
   derivative on every centre (and on the nuclei, for ``V``) is right, since
   none of them is inferred from the others;
4. every derivative against a central finite difference of this module's own
   integrals (``h = 1e-4`` bohr), an oracle independent of libcint;
5. the RHF gradient, its J and K parts among them, against
   ``pyscf.grad.RHF`` at the converged density.

Molecules: a distorted water (no symmetry, so no derivative cancels by
symmetry) in STO-3G and 6-31G, and HCN in STO-3G (three p atoms, linear).
The finite differences run on STO-3G only.

Needs the ``oracle`` group (PySCF): ``uv sync --group oracle``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ints import (  # noqa: E402
    coulomb,
    coulomb_deriv,
    exchange,
    exchange_deriv,
    one_electron,
    rhf_energy,
    rhf_gradient,
    two_electron,
)
from mmd import displaced, shells_of  # noqa: E402

INT_TOL = 1e-12  # of max|X|, against libcint
DERIV_TOL = 1e-12  # of max|dX|, against libcint
INVAR_TOL = 1e-12  # of max|dX|
FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|dX|: O(h^2) truncation, ~1e-9 measured
GRAD_TOL = 1e-10  # Hartree/bohr, absolute
E_TOL = 1e-10  # Hartree

WATER = "O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5"
HCN = "H 0 0 -1.064; C 0 0 0; N 0 0 1.156"
CASES = (("H2O", WATER, "sto-3g", True), ("H2O", WATER, "6-31g", False),
         ("HCN", HCN, "sto-3g", True))

failures: list[str] = []


def _gate(name, err, tol):
    ok = err <= tol
    if not ok:
        failures.append(f"{name}: {err:.3e} > {tol:.0e}")
    return f"{err:.2e}{'' if ok else '  FAIL'}"


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _atom_mask(mol):
    """``mask[A, m]``: AO ``m`` sits on atom ``A``."""
    mask = np.zeros((mol.natm, mol.nao), dtype=bool)
    for A, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        mask[A, p0:p1] = True
    return mask


def _libcint_derivs(mol, mf):
    """dS, dT, dV, d(ab|cd), each ``[natm, 3, ...]``, from libcint's ip
    integrals (``<nabla a|b>``, ``(nabla a b|cd)``); d/dA = -nabla_a."""
    mask = _atom_mask(mol)
    ip_s = mol.intor("int1e_ipovlp")
    ip_t = mol.intor("int1e_ipkin")
    ip1 = mol.intor("int2e_ip1")
    hcore = mf.nuc_grad_method().hcore_generator(mol)

    def one(ip):
        out = np.zeros((mol.natm, 3, mol.nao, mol.nao))
        for A in range(mol.natm):
            m = mask[A]
            out[A][:, m, :] -= ip[:, m, :]
            out[A][:, :, m] -= ip[:, m, :].transpose(0, 2, 1)
        return out

    dS, dT = one(ip_s), one(ip_t)
    dV = np.array([hcore(A) for A in range(mol.natm)]) - dT
    terms = (
        ip1,
        ip1.transpose(0, 2, 1, 3, 4),  # nabla on b
        ip1.transpose(0, 3, 4, 1, 2),  # on c
        ip1.transpose(0, 3, 4, 2, 1),  # on d
    )
    deri = np.zeros((mol.natm, 3) + (mol.nao,) * 4)
    for A in range(mol.natm):
        m = mask[A]
        for pos, t in enumerate(terms):
            sl = [slice(None)] * 5
            sl[1 + pos] = m
            deri[A][tuple(sl)] -= t[tuple(sl)]
    return dS, dT, dV, deri


def _fd(shells, charges, coords, natm):
    """Central differences of ``S, T, V, (ab|cd)`` in every coordinate."""
    out = None
    for A in range(natm):
        for k in range(3):
            vals = []
            for h in (FD_H, -FD_H):
                sh = displaced(shells, A, k, h)
                c = coords.copy()
                c[A, k] += h
                S, T, V, *_ = one_electron(sh, charges, c)
                eri, _ = two_electron(sh, natm, deriv=False)
                vals.append((S, T, V, eri))
            d = [(p - m) / (2 * FD_H) for p, m in zip(*vals, strict=True)]
            if out is None:
                out = [np.zeros((natm, 3) + x.shape) for x in d]
            for o, x in zip(out, d, strict=True):
                o[A, k] = x
    return out


def main() -> int:
    from pyscf import gto, scf

    rows = {n: [] for n in range(1, 6)}
    t0 = time.time()
    for name, atom, basis, do_fd in CASES:
        label = f"{name}/{basis}"
        mol = gto.M(atom=atom, basis=basis, cart=True, verbose=0)
        shells = shells_of(mol)
        Z = mol.atom_charges().astype(float)
        R = mol.atom_coords()
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        D = mf.make_rdm1()
        grad = mf.nuc_grad_method()
        W = grad.make_rdm1e()

        S, T, V, dS, dT, dV = one_electron(shells, Z, R)
        eri, deri = two_electron(shells, mol.natm)

        # 1. integrals
        for nm, mine, ref in (
            ("S", S, mol.intor("int1e_ovlp")),
            ("T", T, mol.intor("int1e_kin")),
            ("V", V, mol.intor("int1e_nuc")),
            ("(ab|cd)", eri, mol.intor("int2e")),
            ("J", coulomb(eri, D), mf.get_j(dm=D)),
            ("K", exchange(eri, D), mf.get_k(dm=D)),
        ):
            err = _gate(f"1 {label} {nm}", _rel(mine, ref), INT_TOL)
            rows[1].append((label, nm, err))
        e = abs(rhf_energy(S, T, V, eri, D, Z, R) - mf.e_tot)
        rows[1].append((label, "E_RHF (abs)", _gate(f"1 {label} E", e, E_TOL)))

        # 2. derivatives against libcint
        rS, rT, rV, rderi = _libcint_derivs(mol, mf)
        for nm, mine, ref in (
            ("dS", dS, rS),
            ("dT", dT, rT),
            ("dV", dV, rV),
            ("d(ab|cd)", deri, rderi),
            ("dJ", coulomb_deriv(deri, D), coulomb_deriv(rderi, D)),
            ("dK", exchange_deriv(deri, D), exchange_deriv(rderi, D)),
        ):
            err = _gate(f"2 {label} {nm}", _rel(mine, ref), DERIV_TOL)
            rows[2].append((label, nm, err))

        # 3. translational invariance
        for nm, d in (("dS", dS), ("dT", dT), ("dV", dV), ("d(ab|cd)", deri)):
            err = float(np.abs(d.sum(axis=0)).max() / np.abs(d).max())
            rows[3].append((label, nm, _gate(f"3 {label} {nm}", err, INVAR_TOL)))

        # 4. finite differences of our own integrals
        if do_fd:
            fd = _fd(shells, Z, R, mol.natm)
            for nm, mine, ref in zip(("dS", "dT", "dV", "d(ab|cd)"),
                                     (dS, dT, dV, deri), fd, strict=True):
                err = float(np.abs(mine - ref).max() / np.abs(mine).max())
                rows[4].append((label, nm, _gate(f"4 {label} {nm}", err, FD_TOL)))

        # 5. the RHF gradient
        g, parts = rhf_gradient(shells, Z, R, D, W)
        ref = grad.kernel()
        err = float(np.abs(g - ref).max())
        rows[5].append((label, "total", _gate(f"5 {label} grad", err, GRAD_TOL)))
        vhf = grad.get_veff(mol, D)  # -(nabla a b|cd) contracted, bra side only
        mask = _atom_mask(mol)
        ref_jk = np.array([2.0 * np.einsum("kmn,mn->k", vhf[:, mask[A]], D[mask[A]])
                           for A in range(mol.natm)])
        err = float(np.abs(parts["J"] + parts["K"] - ref_jk).max())
        rows[5].append((label, "J + K part", _gate(f"5 {label} JK", err, GRAD_TOL)))
        rows[5].append((label, "max|grad|", f"{np.abs(ref).max():.3e}"))

    titles = {
        1: "1. integrals against libcint (of max|X|; E absolute)",
        2: "2. nuclear derivatives against libcint (of max|dX|)",
        3: "3. translational invariance |sum_A dX/dR_A| (of max|dX|)",
        4: f"4. analytic against central differences, h = {FD_H} (of max|dX|)",
        5: "5. RHF gradient against pyscf.grad.RHF (Hartree/bohr, absolute)",
    }
    for n, title in titles.items():
        print(f"\n{title}")
        for label, nm, val in rows[n]:
            print(f"  {label:14s} {nm:12s} {val}")
    print(f"\n{time.time() - t0:.0f} s")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
