"""Check the HGP hessians: ``python experimental/hgp/hessian/check.py``.

Prints eight tables and exits nonzero on a miss:

1. the route: ``grad.block`` (the VRR and HRR kernels over lanes) with a
   second-derivative weight, ``(2a)^2`` on one position or ``2a 2b`` on two,
   against ``eri.hgp_drive.eri_block`` (the same kernels in ``ir.interp``, one
   primitive quartet at a time, the weight carried by the coefficients), on
   every shifted class one quartet of each case runs, elementwise in eps of
   ``eri_block_and_scale``'s absolute-value twin;
2. the second derivatives ``d2(ab|cd)/dR dR`` against libcint's
   ``int2e_ipip1`` / ``_ipvip1`` / ``_ip1ip2`` assembled over all 16 ordered
   centre pairs (``experimental/mmd/hessian/check.py``'s assembly), and
   ``d2J``, ``d2K``, ``d2G`` at the RHF density from both;
3. translational invariance, ``sum_A d2(ab|cd)/dR_A dR_B = 0`` for every
   ``B``, which holds only if every centre pair is right, since none is
   inferred;
4. finite differences, ``h = 1e-4`` bohr (STO-3G): ``d2(ab|cd)`` against
   central differences of ``experimental/hgp/gradient``'s analytic
   ``d(ab|cd)``, and the fused ``G`` hessian against central differences of
   its fused gradient at fixed ``D``, an oracle independent of libcint;
5. the cross-engine pin: ``d2(ab|cd)`` against ``experimental/mmd/hessian``'s
   McMurchie-Davidson ``two_electron2`` on the same molecule (s/p cases;
   that spike stops at p);
6. the energy hessians at fixed ``D``: the fused ``two_electron_hessian``'s
   ``J``, ``K`` and ``G`` against the dense contraction of this module's
   ``d2(ab|cd)`` and of libcint's;
7. the RHF hessian (libcint's one-electron and nuclear terms, everything
   two-electron by HGP, ``experimental/mmd/hessian``'s dense CPHF): the
   skeleton against PySCF's ``partial_hess_elec + hess_nuc``, the whole
   against ``pyscf.hessian.rhf`` with its CPHF solved densely
   (``experimental/mmd/hessian/check.py``'s ``_pyscf_dense``; the shipped
   Krylov result printed, not gated), ``|H - H^T|`` and ``|sum_A H|``;
8. mutations: each factor of ``grad2.FORM`` (the raise weight ``2a``, the
   lowering factor ``i``, the digestion's ``1/2`` and ``-1/4``) perturbed
   alone on H2O/STO-3G, and the table that must catch it shown failing.

Molecules: the gradient spikes' distorted water (no symmetry) in STO-3G,
6-31G and 6-31G* (a d shell on O, so the second derivative runs g classes),
and HCN in STO-3G. Every SCF at ``conv_tol = 1e-14``, ``conv_tol_grad =
1e-11`` (``experimental/mmd/hessian``: at ``1e-12`` the residual ``F_ai``
alone moves the hessian by ~5e-9). Gates are ``experimental/mmd/hessian``'s,
stated before the numbers.

Needs the ``oracle`` group (PySCF): ``uv sync --group oracle``.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MMD = HERE.parents[1] / "mmd"
sys.path[:0] = [str(HERE), str(HERE.parent / "gradient"), str(ROOT / "test"),
                str(ROOT / "src")]

import grad as G  # noqa: E402
import grad2 as G2  # noqa: E402
from oracle import basis as ob  # noqa: E402
from rhf import rhf_hessian  # noqa: E402

from felizjueves.eri.hgp_drive import eri_block_and_scale  # noqa: E402
from felizjueves.reference.boys import boys  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# experimental/mmd/hessian's check: its libcint assembly and dense-CPHF PySCF
# hessian, loaded under its own name (both files are check.py).
MC = _load("mmd_hessian_check", MMD / "hessian" / "check.py")

# Lanes against the interpreter, elementwise in eps of eri.hgp_drive's
# absolute-value twin (its "THE SCALE"), not of max|X|: a (2a)^2 weight on a
# core exponent (C 1s, a ~ 71) makes the boundary ~1e4 times the block, and
# the HRR cancels it back down, which a max|X| gate reads as a 5e-13 miss.
EPS = float(np.finfo(float).eps)
ROUTE_EPS = 16.0
INT_TOL = 1e-11  # of max|d2X|, against libcint (mmd/hessian's)
INVAR_TOL = 1e-11  # of max|d2X|
FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|d2X|: O(h^2) truncation
MMD_TOL = 1e-11  # of max|d2X|, HGP against MMD on the same shells
FUSED_TOL = 1e-11  # Hartree/bohr^2, fused against dense
HESS_TOL = 1e-10  # Hartree/bohr^2, absolute, against PySCF
SYM_TOL = 1e-10  # Hartree/bohr^2, absolute
SCF = {"conv_tol": 1e-14, "conv_tol_grad": 1e-11}
WATER = "O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5"
HCN = "H 0 0 -1.064; C 0 0 0; N 0 0 1.156"
# (name, atom, basis, s/p only: finite differences and the MMD pin)
CASES = (("H2O", WATER, "sto-3g", True), ("H2O", WATER, "6-31g", False),
         ("HCN", HCN, "sto-3g", True), ("H2O", WATER, "6-31g*", False))
SP_PIN = {"6-31g"}  # s/p without finite differences: MMD pin only

failures: list[str] = []


def _gate(name, err, tol):
    ok = err <= tol
    if not ok:
        failures.append(f"{name}: {err:.3e} > {tol:.0e}")
    return f"{err:.2e}{'' if ok else '  FAIL'}"


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _shell_atoms(mol, shells):
    R = mol.atom_coords()
    return [int(np.argmin(np.linalg.norm(R - np.array(s.centre), axis=1)))
            for s in shells]


def _displaced(shells, atoms, A, k, h):
    out = []
    for s, a in zip(shells, atoms, strict=True):
        if a == A:
            c = list(s.centre)
            c[k] += h
            s = replace(s, centre=tuple(c))
        out.append(s)
    return out


def _fd(shells, atoms, natm, D):
    """Central differences of the gradient spike's ``d(ab|cd)`` and of its
    fused ``G`` gradient at fixed ``D``: ``[natm, 3, natm, 3, ...]``, the
    displaced nucleus last."""
    d2, hG = None, np.zeros((natm, 3, natm, 3))
    for B, l in itertools.product(range(natm), range(3)):  # noqa: E741
        v, g = [], []
        for h in (FD_H, -FD_H):
            sh = _displaced(shells, atoms, B, l, h)
            v.append(G.two_electron(sh, atoms, natm)[1])
            g.append(G.two_electron_gradient(sh, atoms, natm, D)["G"])
        if d2 is None:
            d2 = np.zeros((natm, 3, natm, 3) + v[0].shape[2:])
        d2[:, :, B, l] = (v[0] - v[1]) / (2 * FD_H)
        hG[:, :, B, l] = (g[0] - g[1]) / (2 * FD_H)
    return d2, hG


def _route(shells, label, rows):
    """Table 1: every weighted shifted block ``quartet2`` asks for, on the
    quartet of the four highest shells, lanes against the interpreter (the
    weight carried by the coefficients of the shells it falls on)."""
    idx = tuple(int(i) for i in np.argsort([-s.l for s in shells], kind="stable")[:4])
    q = tuple(shells[i] for i in idx)
    ls = tuple(x.l for x in q)
    s = G._lane_scalars(q)
    keys = {(sh, w) for p in range(4) for r in range(p, 4)
            for sh, w, _ in G2.terms(ls, p, r, 0, 0)}
    worst = of_max = 0.0
    for sh, w in sorted(keys):
        cls = tuple(x + d for x, d in zip(ls, sh, strict=True))
        weight = np.ones_like(s["_exp"][0])
        for pos, n in enumerate(w):
            weight = weight * (2.0 * s["_exp"][pos]) ** n
        mine = G.block(q, cls, s, weight)
        qq = [replace(x, l=c, coefficients=tuple(
                  cf * (2.0 * e) ** n for cf, e in
                  zip(x.coefficients, x.exponents, strict=True)))
              for x, c, n in zip(q, cls, w, strict=True)]
        ref, scale = (np.array(x).reshape(mine.shape) for x in
                      eri_block_and_scale(tuple(qq[:2]), tuple(qq[2:]),
                                          boys=boys))
        err = np.abs(mine - ref)
        live = scale > 0  # where the twin is 0 every term is 0: must match
        worst = max(worst, float(np.max(err[live] / scale[live])) / EPS,
                    np.inf if err[~live].any() else 0.0)
        of_max = max(of_max, _rel(mine, ref))
    rows.append((label, f"{len(keys)} blocks of {ls}",
                 f"{_gate(f'1 {label} route', worst, ROUTE_EPS)} eps of "
                 f"the twin ({of_max:.2e} of max, not gated)"))


def _one_electron(mol, mf, D, W):
    """libcint's one-electron and nuclear terms for ``rhf.rhf_hessian``."""
    nao = mol.nao
    hcore = mf.nuc_grad_method().hcore_generator(mol)
    dh = np.array([hcore(A) for A in range(mol.natm)])
    s1 = -mol.intor("int1e_ipovlp", comp=3)
    dS = np.zeros((mol.natm, 3, nao, nao))
    for A, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        dS[A][:, p0:p1] += s1[:, p0:p1]
        dS[A][:, :, p0:p1] += s1[:, p0:p1].transpose(0, 2, 1)
    d2S, d2T, d2V = _libcint_one(mol, mf)
    H = (mf.Hessian().hess_nuc().transpose(0, 2, 1, 3)
         + np.einsum("akblmn,mn->akbl", d2T + d2V, D)
         - np.einsum("akblmn,mn->akbl", d2S, W))
    return {"h": mf.get_hcore(), "dh": dh, "dS": dS, "H": H}


def _libcint_one(mol, mf):
    """``experimental/mmd/hessian``'s ``_libcint_second`` without the
    four-index tensor: ``d2S, d2T, d2V``."""
    nao, mask = mol.nao, MC._atom_mask(mol)

    def one(aa, ab):
        aa = mol.intor(aa, comp=9).reshape(3, 3, nao, nao)
        ab = mol.intor(ab, comp=9).reshape(3, 3, nao, nao)
        terms = {(0, 0): aa, (1, 1): aa.transpose(0, 1, 3, 2), (0, 1): ab,
                 (1, 0): ab.transpose(1, 0, 2, 3)}
        return MC._assemble(mask, lambda p, q: terms[(p, q)], 2)

    d2S = one("int1e_ipipovlp", "int1e_ipovlpip")
    d2T = one("int1e_ipipkin", "int1e_ipkinip")
    hcore = mf.Hessian().hcore_generator(mol)
    d2V = np.zeros_like(d2T)
    for A, B in itertools.product(range(mol.natm), range(mol.natm)):
        d2V[A, :, B, :] = hcore(A, B) - d2T[A, :, B, :]
    return d2S, d2T, d2V


def _energy(d2eri, D):
    J = 0.5 * np.einsum("akblmn,mn->akbl", G2.coulomb_deriv2(d2eri, D), D)
    K = -0.25 * np.einsum("akblmn,mn->akbl", G2.exchange_deriv2(d2eri, D), D)
    return {"J": J, "K": K, "G": J + K}


def main() -> int:
    from pyscf import gto, scf

    rows = {n: [] for n in range(1, 9)}
    keep = None
    t0 = time.time()
    for name, atom, basis, small in CASES:
        label = f"{name}/{basis}"
        mol = gto.M(atom=atom, basis=basis, cart=True, verbose=0)
        shells = ob.from_mole(mol).shells
        atoms = _shell_atoms(mol, shells)
        natm = mol.natm
        mf = scf.RHF(mol).run(**SCF)
        nocc = mol.nelectron // 2
        D = mf.make_rdm1()
        Co = mf.mo_coeff[:, :nocc]
        W = 2.0 * (Co * mf.mo_energy[:nocc]) @ Co.T

        # 1. the route
        if small or basis == "6-31g*":
            _route(shells, label, rows[1])

        t = time.time()
        d2 = G2.two_electron2(shells, atoms, natm)
        t_dense = time.time() - t

        # 2. against libcint
        ref = MC._libcint_second(mol, mf)[3]
        rows[2].append((label, "d2(ab|cd)",
                        _gate(f"2 {label} d2(ab|cd)", _rel(d2, ref), INT_TOL)))
        for nm, fn in (("d2J", G2.coulomb_deriv2), ("d2K", G2.exchange_deriv2),
                       ("d2G", G2.fock_deriv2)):
            rows[2].append((label, nm, _gate(f"2 {label} {nm}",
                                             _rel(fn(d2, D), fn(ref, D)),
                                             INT_TOL)))
        rows[2].append((label, "dense time", f"{t_dense:.1f} s"))

        # 3. translational invariance
        err = float(np.abs(d2.sum(axis=0)).max() / np.abs(d2).max())
        rows[3].append((label, "d2(ab|cd)", _gate(f"3 {label}", err, INVAR_TOL)))

        # 6. fused against dense
        t = time.time()
        fused = G2.two_electron_hessian(shells, atoms, natm, D)
        t_fused = time.time() - t
        dense, dense_ref = _energy(d2, D), _energy(ref, D)
        for k in "JKG":
            err = float(np.abs(fused[k] - dense[k]).max())
            rows[6].append((label, f"{k} vs dense",
                            _gate(f"6 {label} {k} dense", err, FUSED_TOL)))
            err = float(np.abs(fused[k] - dense_ref[k]).max())
            rows[6].append((label, f"{k} vs libcint",
                            _gate(f"6 {label} {k} libcint", err, FUSED_TOL)))
        rows[6].append((label, "fused time", f"{t_fused:.1f} s"))

        if small:
            # 4. finite differences of the gradient spike
            fd, fdG = _fd(shells, atoms, natm, D)
            err = float(np.abs(d2 - fd).max() / np.abs(d2).max())
            rows[4].append((label, "d2(ab|cd)", _gate(f"4 {label}", err, FD_TOL)))
            err = float(np.abs(fused["G"] - fdG).max() / np.abs(fused["G"]).max())
            rows[4].append((label, "fused G", _gate(f"4 {label} G", err, FD_TOL)))
        if small or basis in SP_PIN:
            # 5. against experimental/mmd/hessian
            from ints2 import two_electron2 as mmd_two_electron2
            from mmd import shells_of
            mderi = mmd_two_electron2(shells_of(mol), natm)
            rows[5].append((label, "d2(ab|cd)",
                            _gate(f"5 {label}", _rel(d2, mderi), MMD_TOL)))

        # 7. the RHF hessian
        one = _one_electron(mol, mf, D, W)
        t = time.time()
        H, parts = rhf_hessian(shells, atoms, natm, mf.mo_coeff, mf.mo_energy,
                               nocc, one)
        t_rhf = time.time() - t
        mf.conv_tol_cpscf = 1e-14
        hs = mf.Hessian()
        hs.max_cycle = 200
        skel = parts["one"] + parts["J"] + parts["K"]
        skel_ref = (hs.partial_hess_elec() + hs.hess_nuc()).transpose(0, 2, 1, 3)
        rows[7].append((label, "skeleton", _gate(
            f"7 {label} skeleton", float(np.abs(skel - skel_ref).max()), HESS_TOL)))
        ref_h = MC._pyscf_dense(mf, hs)
        rows[7].append((label, "total", _gate(
            f"7 {label} total", float(np.abs(H - ref_h).max()), HESS_TOL)))
        err = float(np.abs(H - hs.kernel().transpose(0, 2, 1, 3)).max())
        rows[7].append((label, "(Krylov CPHF)", f"{err:.2e}  not gated"))
        Hm = H.reshape(3 * natm, 3 * natm)
        rows[7].append((label, "|H - H^T|", _gate(
            f"7 {label} sym", float(np.abs(Hm - Hm.T).max()), SYM_TOL)))
        rows[7].append((label, "|sum_A H|", _gate(
            f"7 {label} invar", float(np.abs(H.sum(axis=0)).max()), SYM_TOL)))
        rows[7].append((label, "max|H|", f"{np.abs(ref_h).max():.3e}"))
        rows[7].append((label, "time", f"{t_rhf:.1f} s"))
        if keep is None:
            keep = (shells, atoms, natm, D, ref, H, parts, ref_h)

    # 8. mutations, each alone, each must be caught
    shells, atoms, natm, D, ref, H, parts, ref_h = keep
    for key, value in (("raise", 1.0), ("lower", 0.0), ("J", 0.25),
                       ("K", -0.5)):
        saved = G2.FORM[key]
        G2.FORM[key] = value
        try:
            if key in ("raise", "lower"):
                err = _rel(G2.two_electron2(shells, atoms, natm), ref)
                tol, what = INT_TOL, "table 2"
            else:
                two = G2.two_electron_hessian(shells, atoms, natm, D)
                Hm = H - parts["J"] - parts["K"] + two["J"] + two["K"]
                err, tol, what = float(np.abs(Hm - ref_h).max()), HESS_TOL, \
                    "table 7"
        finally:
            G2.FORM[key] = saved
        caught = err > tol
        if not caught:
            failures.append(f"8 FORM[{key!r}] = {value} not caught")
        rows[8].append(("H2O/sto-3g", f"{key} {saved} -> {value}",
                        f"{err:.2e} by {what}, "
                        f"{'caught' if caught else 'MISSED'}"))

    titles = {
        1: "1. the route: weighted shifted blocks, lanes against ir.interp "
           "(elementwise, eps of hgp_drive's absolute-value twin)",
        2: "2. second derivatives against libcint (of max|d2X|)",
        3: "3. translational invariance |sum_A d2X/dR_A dR_B| (of max|d2X|)",
        4: f"4. analytic against central differences of the gradient spike, "
           f"h = {FD_H} (of max)",
        5: "5. HGP against experimental/mmd/hessian's MMD (of max|d2X|)",
        6: "6. energy hessians at fixed D, fused (Hartree/bohr^2, absolute)",
        7: "7. RHF hessian (Hartree/bohr^2, absolute): against "
           "pyscf.hessian.rhf, symmetry, invariance",
        8: "8. mutations of grad2.FORM, each alone (must exceed its gate)",
    }
    for n, title in titles.items():
        print(f"\n{title}")
        for label, nm, val in rows[n]:
            print(f"  {label:14s} {nm:26s} {val}")
    print(f"\n{time.time() - t0:.0f} s")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
