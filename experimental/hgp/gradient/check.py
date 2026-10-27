"""Check the HGP gradients: ``python experimental/hgp/gradient/check.py``.

Prints seven tables and exits nonzero on a miss:

1. the route: ``lanes.boys_lanes`` against the scalar ``reference.boys``, and
   ``grad.block`` (the VRR and HRR kernels over lanes) against
   ``eri.hgp_drive.eri_block`` (the same kernels in ``ir.interp``, one
   primitive quartet at a time) on raised and lowered classes of every case;
2. the integrals ``(ab|cd)``, ``J``, ``K`` and ``G = J - K/2`` against libcint
   (``int2e``, ``get_j``, ``get_k``), all Cartesian;
3. the nuclear derivatives ``d(ab|cd)/dR`` against libcint's ``int2e_ip1``
   assembled over the four centres, and ``dJ``, ``dK``, ``dG`` at the RHF
   density from both;
4. translational invariance, ``sum_atoms d(ab|cd)/dR = 0``, which holds only
   if every centre's derivative is right, since none is inferred;
5. every derivative against a central difference of this module's own
   ``(ab|cd)`` (``h = 1e-4`` bohr), an oracle independent of libcint
   (s/p cases);
6. the cross-engine pin: ``d(ab|cd)/dR`` against
   ``experimental/mmd/gradient``'s McMurchie-Davidson derivative on the same
   molecule (s/p cases; that spike stops at p);
7. the energy gradients: the fused ``two_electron_gradient`` against the
   dense contraction of table 3's tensor, its J and K parts against PySCF's
   ``get_veff`` gradient, and the whole RHF gradient (libcint's one-electron
   and nuclear terms plus this module's two-electron one) against
   ``pyscf.grad.RHF``;
8. mutations: each factor of ``grad.FORM`` (the raise weight ``2a``, the
   lowering factor ``i``, the digestion's ``1/2`` and ``-1/4``) perturbed alone
   on H2O/STO-3G, and the table that must catch it shown failing.

Molecules: ``experimental/mmd/gradient``'s distorted water (no symmetry) in
STO-3G and 6-31G, and HCN in STO-3G, plus the same water in 6-31G* (a d shell
on O, so the derivative runs an f class). Densities from PySCF's RHF at
``conv_tol = 1e-12``. Gates are those of ``experimental/mmd/gradient``,
stated before the numbers.

Needs the ``oracle`` group (PySCF): ``uv sync --group oracle``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "test"), str(ROOT / "src")]

import grad as G  # noqa: E402
from lanes import boys_lanes  # noqa: E402
from oracle import basis as ob  # noqa: E402

from felizjueves.eri.hgp_drive import eri_block  # noqa: E402
from felizjueves.reference.boys import boys  # noqa: E402

BOYS_TOL = 1e-15  # relative, lanes against the scalar reference
ROUTE_TOL = 1e-14  # of max|X|, lanes against the interpreter
INT_TOL = 1e-12  # of max|X|, against libcint
DERIV_TOL = 1e-12  # of max|dX|, against libcint
INVAR_TOL = 1e-12  # of max|dX|
FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|dX|: O(h^2) truncation
MMD_TOL = 1e-13  # of max|dX|, HGP against MMD on the same shells
FUSED_TOL = 1e-13  # Hartree/bohr, fused against dense
GRAD_TOL = 1e-10  # Hartree/bohr, absolute
WATER = "O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5"
HCN = "H 0 0 -1.064; C 0 0 0; N 0 0 1.156"
# (name, atom, basis, finite differences and MMD pin)
CASES = (("H2O", WATER, "sto-3g", True), ("H2O", WATER, "6-31g", False),
         ("HCN", HCN, "sto-3g", True), ("H2O", WATER, "6-31g*", False))

failures: list[str] = []


def _gate(name, err, tol):
    ok = err <= tol
    if not ok:
        failures.append(f"{name}: {err:.3e} > {tol:.0e}")
    return f"{err:.2e}{'' if ok else '  FAIL'}"


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _atom_mask(mol):
    mask = np.zeros((mol.natm, mol.nao), dtype=bool)
    for A, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        mask[A, p0:p1] = True
    return mask


def _shell_atoms(mol, shells):
    R = mol.atom_coords()
    return [int(np.argmin(np.linalg.norm(R - np.array(s.centre), axis=1)))
            for s in shells]


def _libcint_deri(mol):
    """``d(ab|cd)/dR`` ``[natm, 3] + [nao]*4`` from ``int2e_ip1``
    (``(nabla a b|cd)``; ``d/dA = -nabla_a``) over all four centres."""
    mask = _atom_mask(mol)
    ip1 = mol.intor("int2e_ip1")
    terms = (ip1, ip1.transpose(0, 2, 1, 3, 4), ip1.transpose(0, 3, 4, 1, 2),
             ip1.transpose(0, 3, 4, 2, 1))
    deri = np.zeros((mol.natm, 3) + (mol.nao,) * 4)
    for A in range(mol.natm):
        for pos, t in enumerate(terms):
            sl = [slice(None)] * 5
            sl[1 + pos] = mask[A]
            deri[A][tuple(sl)] -= t[tuple(sl)]
    return deri


def _displaced(shells, atoms, A, k, h):
    from dataclasses import replace
    out = []
    for s, a in zip(shells, atoms, strict=True):
        if a == A:
            c = list(s.centre)
            c[k] += h
            s = replace(s, centre=tuple(c))
        out.append(s)
    return out


def _fd(shells, atoms, natm):
    out = None
    for A in range(natm):
        for k in range(3):
            v = [G.two_electron(_displaced(shells, atoms, A, k, h), atoms,
                                natm, deriv=False)[0] for h in (FD_H, -FD_H)]
            if out is None:
                out = np.zeros((natm, 3) + v[0].shape)
            out[A, k] = (v[0] - v[1]) / (2 * FD_H)
    return out


def _route_rows(shells, label, rows):
    """Table 1's block check: a few quartets, each class raised and lowered
    as ``quartet`` runs it, lanes against the interpreter."""
    n = len(shells)
    picks = {(0, 0, 0, 0), (n - 1, n - 1, n - 1, n - 1), (0, n - 1, 1, n - 2)}
    picks |= {tuple(int(i) for i in np.argsort([-s.l for s in shells])[:4])}
    worst = 0.0
    for idx in sorted(picks):
        q = [shells[i] for i in idx]
        s = G._lane_scalars(tuple(q))
        for pos in range(4):
            for dl in (+1, 0, -1):
                ls = tuple(x.l + (dl if i == pos else 0) for i, x in enumerate(q))
                if min(ls) < 0:
                    continue
                w = 2.0 * s["_exp"][pos] if dl == 1 else None
                mine = G.block(tuple(q), ls, s, w)
                # The interpreter sees the same class through shells carrying
                # the weighted coefficients: lane weight 2a on one position.
                from dataclasses import replace
                qq = list(q)
                qq[pos] = replace(q[pos], l=ls[pos],
                                  coefficients=tuple(
                                      c * (2 * e if dl == 1 else 1.0)
                                      for c, e in zip(q[pos].coefficients,
                                                      q[pos].exponents,
                                                      strict=True)))
                ref = np.array(eri_block(tuple(qq[:2]), tuple(qq[2:]),
                                         boys=boys)).reshape(mine.shape)
                worst = max(worst, _rel(mine, ref))
    rows.append((label, "lanes vs interp",
                 _gate(f"1 {label} route", worst, ROUTE_TOL)))


def main() -> int:
    from pyscf import gto, scf

    rows = {n: [] for n in range(1, 9)}
    keep = None
    t0 = time.time()

    # 1a. Boys over lanes
    T = np.concatenate([[0.0, 1e-8], np.geomspace(1e-6, 1e6, 400),
                        np.linspace(100, 150, 101)])
    worst = 0.0
    for top in (0, 3, 9, 17):
        F = boys_lanes(top, T)
        for m in range(top + 1):
            ref = np.array([boys(m, float(t)) for t in T])
            worst = max(worst, float(np.max(np.abs(F[m] - ref) / ref)))
    rows[1].append(("Boys", "m <= 17, T <= 1e6", _gate("1 boys", worst, BOYS_TOL)))

    for name, atom, basis, small in CASES:
        label = f"{name}/{basis}"
        mol = gto.M(atom=atom, basis=basis, cart=True, verbose=0)
        shells = ob.from_mole(mol).shells
        atoms = _shell_atoms(mol, shells)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        D = mf.make_rdm1()

        _route_rows(shells, label, rows[1])

        t = time.time()
        eri, deri = G.two_electron(shells, atoms, mol.natm)
        t_sweep = time.time() - t

        # 2. integrals
        for nm, mine, ref in (
            ("(ab|cd)", eri, mol.intor("int2e")),
            ("J", G.coulomb(eri, D), mf.get_j(dm=D)),
            ("K", G.exchange(eri, D), mf.get_k(dm=D)),
            ("G = J - K/2", G.fock(eri, D), mf.get_j(dm=D) - 0.5 * mf.get_k(dm=D)),
        ):
            rows[2].append((label, nm, _gate(f"2 {label} {nm}", _rel(mine, ref),
                                             INT_TOL)))

        # 3. derivatives against libcint
        rderi = _libcint_deri(mol)
        for nm, fn in (("d(ab|cd)", None), ("dJ", G.coulomb_deriv),
                       ("dK", G.exchange_deriv), ("dG", G.fock_deriv)):
            mine = deri if fn is None else fn(deri, D)
            ref = rderi if fn is None else fn(rderi, D)
            rows[3].append((label, nm, _gate(f"3 {label} {nm}", _rel(mine, ref),
                                             DERIV_TOL)))
        rows[3].append((label, "sweep time", f"{t_sweep:.1f} s"))

        # 4. translational invariance
        err = float(np.abs(deri.sum(axis=0)).max() / np.abs(deri).max())
        rows[4].append((label, "d(ab|cd)", _gate(f"4 {label}", err, INVAR_TOL)))

        if small:
            # 5. finite differences of our own integrals
            fd = _fd(shells, atoms, mol.natm)
            err = float(np.abs(deri - fd).max() / np.abs(deri).max())
            rows[5].append((label, "d(ab|cd)", _gate(f"5 {label}", err, FD_TOL)))
            # 6. against experimental/mmd/gradient
            sys.path.insert(0, str(HERE.parents[1] / "mmd" / "gradient"))
            from ints import two_electron as mmd_two_electron
            from mmd import shells_of
            _, mderi = mmd_two_electron(shells_of(mol), mol.natm)
            rows[6].append((label, "d(ab|cd)",
                            _gate(f"6 {label}", _rel(deri, mderi), MMD_TOL)))

        # 7. energy gradients
        t = time.time()
        fused = G.two_electron_gradient(shells, atoms, mol.natm, D)
        t_fused = time.time() - t
        dense = {"J": 0.5 * np.einsum("xymn,mn->xy", G.coulomb_deriv(deri, D), D),
                 "K": -0.25 * np.einsum("xymn,mn->xy", G.exchange_deriv(deri, D), D)}
        dense["G"] = 0.5 * np.einsum("xymn,mn->xy", G.fock_deriv(deri, D), D)
        err = max(float(np.abs(fused[k] - dense[k]).max()) for k in "JKG")
        rows[7].append((label, "fused vs dense",
                        _gate(f"7 {label} fused", err, FUSED_TOL)))
        g = mf.nuc_grad_method()
        mask = _atom_mask(mol)
        # PySCF's vj, vk are -(nabla a b|cd) D, the bra centre's d/dA only;
        # the other three centres follow by the orbit, so
        # dE_J/dA = 2 SUM_{a on A} vj_ab D_ab and dE_K/dA = -SUM vk_ab D_ab.
        vj, vk = g.get_jk(mol, D)
        for nm, v, f in (("J part", vj, 2.0), ("K part", vk, -1.0)):
            ref = np.array([f * np.einsum("kmn,mn->k", v[:, mask[A]], D[mask[A]])
                            for A in range(mol.natm)])
            err = float(np.abs(fused[nm[0]] - ref).max())
            rows[7].append((label, nm, _gate(f"7 {label} {nm}", err, GRAD_TOL)))
        hcore = g.hcore_generator(mol)
        s1 = mol.intor("int1e_ipovlp")
        W = g.make_rdm1e()
        total = g.grad_nuc().copy()
        for A in range(mol.natm):
            total[A] += np.einsum("kmn,mn->k", hcore(A), D)
            total[A] -= 2.0 * np.einsum("kmn,mn->k", -s1[:, mask[A]], W[mask[A]])
        ref = g.kernel()
        if keep is None:
            keep = (shells, atoms, mol.natm, D, rderi, total.copy(), ref)
        total += fused["G"]
        rows[7].append((label, "RHF total", _gate(f"7 {label} RHF",
                        float(np.abs(total - ref).max()), GRAD_TOL)))
        rows[7].append((label, "max|grad|", f"{np.abs(ref).max():.3e}"))
        rows[7].append((label, "fused time", f"{t_fused:.1f} s"))

    # 8. mutations, each alone, each must be caught
    shells, atoms, natm, D, rderi, one, ref = keep
    for key, value in (("raise", 1.0), ("lower", 0.0), ("J", 0.25),
                       ("K", -0.5)):
        saved = G.FORM[key]
        G.FORM[key] = value
        try:
            if key in ("raise", "lower"):
                _, deri = G.two_electron(shells, atoms, natm)
                err, tol, what = _rel(deri, rderi), DERIV_TOL, "table 3"
            else:
                gG = G.two_electron_gradient(shells, atoms, natm, D)["G"]
                err, tol, what = float(np.abs(one + gG - ref).max()), \
                    GRAD_TOL, "table 7"
        finally:
            G.FORM[key] = saved
        caught = err > tol
        if not caught:
            failures.append(f"8 FORM[{key!r}] = {value} not caught")
        rows[8].append(("H2O/sto-3g", f"{key} {saved} -> {value}",
                        f"{err:.2e} by {what}, "
                        f"{'caught' if caught else 'MISSED'}"))

    titles = {
        1: "1. the route: Boys over lanes (relative); lanes against "
           "ir.interp (of max|X|)",
        2: "2. integrals against libcint (of max|X|)",
        3: "3. nuclear derivatives against libcint int2e_ip1 (of max|dX|)",
        4: "4. translational invariance |sum_A dX/dR_A| (of max|dX|)",
        5: f"5. analytic against central differences, h = {FD_H} (of max|dX|)",
        6: "6. HGP against experimental/mmd/gradient's MMD (of max|dX|)",
        7: "7. energy gradients (Hartree/bohr, absolute)",
        8: "8. mutations of grad.FORM, each alone (must exceed its gate)",
    }
    for n, title in titles.items():
        print(f"\n{title}")
        for label, nm, val in rows[n]:
            print(f"  {label:14s} {nm:18s} {val}")
    print(f"\n{time.time() - t0:.0f} s")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
