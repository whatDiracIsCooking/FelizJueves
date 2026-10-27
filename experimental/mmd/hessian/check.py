"""Check the MMD hessian: ``python experimental/mmd/hessian/check.py``.

Prints four tables and exits nonzero on a miss:

1. the second-derivative integrals ``d2S, d2T, d2V, d2(ab|cd)`` against
   libcint's (``int1e_ipipovlp`` / ``_ipovlpip``, ``int1e_ipipkin`` /
   ``_ipkinip``, ``int2e_ipip1`` / ``_ipvip1`` / ``_ip1ip2``, assembled over
   every ordered pair of centres on each pair of atoms; ``d2V`` as PySCF's
   hessian ``hcore_generator`` minus ``d2T``, so it carries the operator
   terms);
2. translational invariance, ``sum_A d2X/dR_A dR_B = 0`` for every ``B``,
   which holds only if every centre pair (and nucleus, for ``V``) is right,
   since none is inferred from the others;
3. every second derivative against a central difference of the gradient
   spike's analytic first derivatives (``h = 1e-4`` bohr), an oracle
   independent of libcint;
4. the RHF hessian: its skeleton against PySCF's ``partial_hess_elec`` plus
   ``hess_nuc``, the whole against ``pyscf.hessian.rhf`` with its CPHF
   solved densely (``_pyscf_dense``; the Krylov result is printed, not
   gated, because it misses PySCF's own equations by ~1e-6), its symmetry
   ``H_xy = H_yx`` and invariance (neither imposed: the CPHF response is not
   symmetric by construction), and against central differences of the
   gradient spike's analytic RHF gradient with the SCF re-converged at each
   displaced geometry (STO-3G only).

Molecules: the gradient spike's, a distorted water (no symmetry) in STO-3G and
6-31G and HCN in STO-3G. Every SCF runs at ``conv_tol = 1e-14``,
``conv_tol_grad = 1e-11``: at the gradient spike's ``1e-12`` the residual
``F_ai`` alone moves the hessian by ~5e-9, which is no defect of either code.

Needs the ``oracle`` group (PySCF): ``uv sync --group oracle``.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gradient")]

from hessian import rhf_hessian  # noqa: E402
from ints import _PERMS, one_electron, rhf_gradient, two_electron  # noqa: E402
from ints2 import one_electron2, two_electron2  # noqa: E402
from mmd import displaced, shells_of  # noqa: E402

# The gradient spike's 1e-12, widened tenfold: a same-atom block of a second
# derivative is exactly zero by invariance yet a sum of terms carrying
# (2a)^2 with a ~ 130 (O 1s), and libcint's own d2T and d2(ab|cd) miss
# invariance by up to 2.6e-12 of max on H2O/6-31G.
INT_TOL = 1e-11  # of max|d2X|, against libcint
INVAR_TOL = 1e-11  # of max|d2X|
FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|d2X|: O(h^2) truncation
HESS_TOL = 1e-10  # Hartree/bohr^2, absolute, against PySCF
SYM_TOL = 1e-10  # Hartree/bohr^2, absolute
FD_HESS_TOL = 1e-6  # Hartree/bohr^2, absolute: truncation plus SCF noise / h
SCF = {"conv_tol": 1e-14, "conv_tol_grad": 1e-11}

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


def _assemble(mask, term, npos):
    """``d2X[A, k, B, l, ...]`` from ``term(p, q)``, the ``[3, 3, nao^npos]``
    second derivative with the first nabla on position ``p`` and the second on
    ``q``. d/dA = -nabla, twice: the sign is +."""
    natm, nao = mask.shape
    out = np.zeros((natm, 3, natm, 3) + (nao,) * npos)
    for p, q in itertools.product(range(npos), range(npos)):
        t = term(p, q)
        for A, B in itertools.product(range(natm), range(natm)):
            w = np.ones((nao,) * npos)
            for pos, at in ((p, A), (q, B)):
                shape = [1] * npos
                shape[pos] = nao
                w = w * mask[at].reshape(shape)
            out[A, :, B, :] += t * w
    return out


def _libcint_second(mol, mf):
    """d2S, d2T, d2V, d2(ab|cd), each ``[natm, 3, natm, 3, ...]``."""
    nao, mask = mol.nao, _atom_mask(mol)

    def one(aa, ab):
        aa = mol.intor(aa, comp=9).reshape(3, 3, nao, nao)  # (nabla nabla a|b)
        ab = mol.intor(ab, comp=9).reshape(3, 3, nao, nao)  # (nabla a|nabla b)
        terms = {(0, 0): aa, (1, 1): aa.transpose(0, 1, 3, 2), (0, 1): ab,
                 (1, 0): ab.transpose(1, 0, 2, 3)}
        return _assemble(mask, lambda p, q: terms[(p, q)], 2)

    d2S = one("int1e_ipipovlp", "int1e_ipovlpip")
    d2T = one("int1e_ipipkin", "int1e_ipkinip")
    hcore = mf.Hessian().hcore_generator(mol)
    d2V = np.zeros_like(d2T)
    for A, B in itertools.product(range(mol.natm), range(mol.natm)):
        d2V[A, :, B, :] = hcore(A, B) - d2T[A, :, B, :]

    canon = {s: mol.intor(name, comp=9).reshape((3, 3) + (nao,) * 4)
             for s, name in ((0, "int2e_ipip1"), (1, "int2e_ipvip1"),
                             (2, "int2e_ip1ip2"))}

    def two(p, q):
        """Move position ``p`` to slot 0 and ``q`` to slot ``s`` by one of the
        eight permutational symmetries of ``(ab|cd)``."""
        s = 0 if p == q else (1 if p // 2 == q // 2 else 2)
        perm = next(pi for pi in _PERMS if pi[0] == p and pi[s] == q)
        sub = "kl" + "".join("abcd"[perm[m]] for m in range(4))
        return np.einsum(f"{sub}->klabcd", canon[s])

    return d2S, d2T, d2V, _assemble(mask, two, 4)


def _fd_integrals(shells, charges, coords, natm):
    """Central differences of the gradient spike's ``dS, dT, dV, d(ab|cd)``:
    ``fd[A, k, B, l] = d/dR_{B,l} dX/dR_{A,k}``."""
    out = None
    for B, l in itertools.product(range(natm), range(3)):  # noqa: E741
        vals = []
        for h in (FD_H, -FD_H):
            sh = displaced(shells, B, l, h)
            c = coords.copy()
            c[B, l] += h
            _, _, _, dS, dT, dV = one_electron(sh, charges, c)
            _, deri = two_electron(sh, natm)
            vals.append((dS, dT, dV, deri))
        d = [(p - m) / (2 * FD_H) for p, m in zip(*vals, strict=True)]
        if out is None:
            out = [np.zeros(x.shape[:2] + (natm, 3) + x.shape[2:]) for x in d]
        for o, x in zip(out, d, strict=True):
            o[:, :, B, l] = x
    return out


def _fd_hessian(mol, dm0):
    """Central differences of the gradient spike's RHF gradient, the SCF
    re-converged (from ``dm0``) at every displaced geometry."""
    from pyscf import scf

    natm = mol.natm
    Z = mol.atom_charges().astype(float)
    R0 = mol.atom_coords()
    out = np.zeros((natm, 3, natm, 3))
    for B, l in itertools.product(range(natm), range(3)):  # noqa: E741
        g = []
        for h in (FD_H, -FD_H):
            R = R0.copy()
            R[B, l] += h
            m = mol.set_geom_(R, unit="Bohr", inplace=False)
            mf = scf.RHF(m).run(dm0, **SCF)
            W = mf.nuc_grad_method().make_rdm1e()
            g.append(rhf_gradient(shells_of(m), Z, R, mf.make_rdm1(), W)[0])
        out[:, :, B, l] = (g[0] - g[1]) / (2 * FD_H)
    return out


def _pyscf_dense(mf, hs):
    """PySCF's hessian with its CPHF solved densely: its own ``make_h1``,
    ``gen_vind`` (the response operator, as a matrix) and ``hess_elec``,
    only the Krylov solve replaced. Its Krylov solution leaves a residual of
    ~1e-6 in its own equations here even at ``conv_tol_cpscf = 1e-14``."""
    from pyscf.hessian.rhf import gen_vind

    mol, C, e, occ = mf.mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ
    nao, nmo = C.shape
    no = int((occ > 0).sum())
    nv = nmo - no
    ei, ea = e[:no], e[no:]
    fx = gen_vind(mf, C, occ)
    unit = np.zeros((nv * no, nmo, no))
    unit[np.arange(nv * no), no + np.arange(nv * no) // no, np.arange(nv * no) % no] = 1
    A = fx(unit).reshape(nv * no, nmo, no)[:, no:].reshape(nv * no, -1).T
    A += np.diag(np.subtract.outer(ea, ei).ravel())
    h1ao = hs.make_h1(C, occ)
    s1a = -mol.intor("int1e_ipovlp", comp=3)
    mo1s, e1s = [], []
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        s1ao = np.zeros((3, nao, nao))
        s1ao[:, p0:p1] += s1a[:, p0:p1]
        s1ao[:, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
        s1 = np.einsum("mp,xmn,ni->xpi", C, s1ao, C[:, :no])
        h1 = np.einsum("mp,xmn,ni->xpi", C, h1ao[ia], C[:, :no])
        hs1 = h1 - s1 * ei
        mo1 = np.zeros((3, nmo, no))
        mo1[:, :no] = -0.5 * s1[:, :no]
        rhs = -(hs1[:, no:] + fx(mo1)[:, no:]).reshape(3, -1)
        mo1[:, no:] = np.linalg.solve(A, rhs.T).T.reshape(3, nv, no)
        hs1 = hs1 + fx(mo1)  # as pyscf.scf.cphf.solve_withs1 finishes
        mo1s.append(np.einsum("mp,xpi->xmi", C, mo1))
        e1s.append(hs1[:, :no] + mo1[:, :no] * (ei[:, None] - ei))
    de2 = hs.hess_elec(mo1=mo1s, mo_e1=e1s, h1ao=h1ao)
    return (de2 + hs.hess_nuc()).transpose(0, 2, 1, 3)


def main() -> int:
    from pyscf import gto, scf

    rows = {n: [] for n in range(1, 5)}
    t0 = time.time()
    for name, atom, basis, do_fd in CASES:
        label = f"{name}/{basis}"
        mol = gto.M(atom=atom, basis=basis, cart=True, verbose=0)
        shells = shells_of(mol)
        Z = mol.atom_charges().astype(float)
        R = mol.atom_coords()
        mf = scf.RHF(mol).run(**SCF)
        nocc = mol.nelectron // 2

        d2S, d2T, d2V = one_electron2(shells, Z, R)
        d2eri = two_electron2(shells, mol.natm)
        mine = {"d2S": d2S, "d2T": d2T, "d2V": d2V, "d2(ab|cd)": d2eri}

        # 1. second-derivative integrals against libcint
        ref = dict(zip(mine, _libcint_second(mol, mf), strict=True))
        for nm, x in mine.items():
            err = _gate(f"1 {label} {nm}", _rel(x, ref[nm]), INT_TOL)
            rows[1].append((label, nm, err))

        # 2. translational invariance
        for nm, x in mine.items():
            err = float(np.abs(x.sum(axis=0)).max() / np.abs(x).max())
            rows[2].append((label, nm, _gate(f"2 {label} {nm}", err, INVAR_TOL)))

        # 3. finite differences of the gradient spike's first derivatives
        if do_fd:
            fd = _fd_integrals(shells, Z, R, mol.natm)
            for (nm, x), ref_x in zip(mine.items(), fd, strict=True):
                err = float(np.abs(x - ref_x).max() / np.abs(x).max())
                rows[3].append((label, nm, _gate(f"3 {label} {nm}", err, FD_TOL)))

        # 4. the RHF hessian
        H, parts = rhf_hessian(shells, Z, R, mf.mo_coeff, mf.mo_energy, nocc)
        mf.conv_tol_cpscf = 1e-14  # PySCF's Krylov CPHF; its default is 1e-9
        hs = mf.Hessian()
        hs.max_cycle = 200
        skel = sum(v for k, v in parts.items() if k != "response")
        skel_ref = (hs.partial_hess_elec() + hs.hess_nuc()).transpose(0, 2, 1, 3)
        err = float(np.abs(skel - skel_ref).max())
        rows[4].append((label, "skeleton", _gate(f"4 {label} skeleton", err, HESS_TOL)))
        ref_h = _pyscf_dense(mf, hs)
        err = float(np.abs(H - ref_h).max())
        rows[4].append((label, "total", _gate(f"4 {label} total", err, HESS_TOL)))
        err = float(np.abs(H - hs.kernel().transpose(0, 2, 1, 3)).max())
        rows[4].append((label, "(Krylov CPHF)", f"{err:.2e}  not gated"))
        n3 = 3 * mol.natm
        Hm = H.reshape(n3, n3)
        err = float(np.abs(Hm - Hm.T).max())
        rows[4].append((label, "|H - H^T|", _gate(f"4 {label} sym", err, SYM_TOL)))
        err = float(np.abs(H.sum(axis=0)).max())
        rows[4].append((label, "|sum_A H|", _gate(f"4 {label} invar", err, SYM_TOL)))
        if do_fd:
            err = float(np.abs(H - _fd_hessian(mol, mf.make_rdm1())).max())
            rows[4].append((label, "FD of gradient",
                            _gate(f"4 {label} fd", err, FD_HESS_TOL)))
        rows[4].append((label, "max|H|", f"{np.abs(ref_h).max():.3e}"))

    titles = {
        1: "1. second-derivative integrals against libcint (of max|d2X|)",
        2: "2. translational invariance |sum_A d2X/dR_A dR_B| (of max|d2X|)",
        3: f"3. analytic against central differences of the gradient spike's "
           f"first derivatives, h = {FD_H} (of max|d2X|)",
        4: "4. RHF hessian (Hartree/bohr^2, absolute): against pyscf.hessian.rhf, "
           "symmetry, invariance, FD of the analytic gradient",
    }
    for n, title in titles.items():
        print(f"\n{title}")
        for label, nm, val in rows[n]:
            print(f"  {label:14s} {nm:14s} {val}")
    print(f"\n{time.time() - t0:.0f} s")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
