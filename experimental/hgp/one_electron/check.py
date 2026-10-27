"""Settle the HGP one-electron spike: ``python experimental/hgp/one_electron/check.py``.

Prints three tables and exits nonzero on a miss:

1. ``boys`` against ``scipy.special.gammainc`` over ``T`` in [0, 1e3], m <= 4;
2. S, T, V and the dipole of four molecules (s and p shells, uncontracted
   and contracted, p shells on distinct off-axis centres, a non-zero dipole
   origin) against libcint's ``int1e_ovlp_cart``, ``int1e_kin_cart``,
   ``int1e_nuc_cart`` and ``int1e_r_cart``, plus the symmetry of each
   matrix (every ordered shell pair is evaluated, so it is a check);
3. every sign in ``hgp1e.FORM`` flipped alone: the oracle must catch each.

Needs the ``oracle`` group (PySCF): ``uv sync --group oracle``.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gamma, gammainc

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hgp1e  # noqa: E402
from hgp1e import dipole, kinetic, nuclear, overlap, shells_of  # noqa: E402

BOYS_TOL = 4e-15
INT_TOL = 1e-13  # of max|M| per matrix
ORIGIN = (0.3, -0.7, 1.1)

MOLECULES = {
    "H2O/STO-3G": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "sto-3g"),
    "H2O/6-31G": ("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "6-31g"),
    # off-axis, no symmetry: every PA, PC, AB component nonzero
    "NH3/6-31G (distorted)": (
        "N 0.05 -0.02 0.11; H 0.94 0.13 -0.31; H -0.41 0.88 -0.27; H -0.37 -0.79 -0.42",
        "6-31g",
    ),
    # uncontracted p on two heavy centres
    "CO/uncontracted": (
        "C 0.1 0.2 -0.3; O 0.4 -0.1 0.85",
        {"C": "unc-6-31g", "O": "unc-6-31g"},
    ),
}


def boys_ref(m: int, T: float) -> float:
    if T < 1e-3:
        return sum(
            (-T) ** k / (math.factorial(k) * (2 * m + 2 * k + 1)) for k in range(8)
        )
    s = m + 0.5
    return gamma(s) * gammainc(s, T) / (2.0 * T**s)


def check_boys() -> bool:
    print(f"1. boys against gammainc, m <= 4, tolerance {BOYS_TOL:.0e} relative")
    worst = 0.0
    for T in [0.0, 1e-12, 1e-6, 1e-3, 0.1, 0.5, 1, 2, 5, 10, 20, 29.999, 30, 30.001,
              40, 60, 100, 300, 1e3]:
        F = hgp1e.boys(4, T)
        for m in range(5):
            r = boys_ref(m, T)
            worst = max(worst, abs(F[m] - r) / r)
    ok = worst <= BOYS_TOL
    print(f"   worst {worst:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def build(mol):
    shells = shells_of(mol)
    return {
        "S": overlap(shells),
        "T": kinetic(shells),
        "V": nuclear(shells, mol.atom_charges(), mol.atom_coords()),
        "MU": dipole(shells, ORIGIN),
    }


def oracle(mol):
    with mol.with_common_orig(ORIGIN):
        r = mol.intor("int1e_r_cart")
    return {
        "S": mol.intor("int1e_ovlp_cart"),
        "T": mol.intor("int1e_kin_cart"),
        "V": mol.intor("int1e_nuc_cart"),
        "MU": r,
    }


def errors(mine, ref):
    """Per matrix: |mine - ref| / max|ref| and the asymmetry / max|ref|."""
    out = {}
    for k in ref:
        scale = np.abs(ref[k]).max()
        err = np.abs(mine[k] - ref[k]).max() / scale
        asym = np.abs(mine[k] - np.swapaxes(mine[k], -1, -2)).max() / scale
        out[k] = (err, asym)
    return out


def molecules():
    from pyscf import gto

    out = {}
    for name, (atom, basis) in MOLECULES.items():
        mol = gto.M(atom=atom, basis=basis, cart=True, verbose=0)
        out[name] = (mol, oracle(mol))
    return out


def check_oracle(mols) -> bool:
    print(f"2. against libcint (int1e_*_cart), |err| / max|ref|, "
          f"tolerance {INT_TOL:.0e}; dipole origin {ORIGIN}")
    head = " ".join(f"{k:>9}" for k in ("S", "T", "V", "MU"))
    print(f"   {'molecule':<24} {'nao':>4} {head} {'asym':>9}")
    ok = True
    for name, (mol, ref) in mols.items():
        e = errors(build(mol), ref)
        worst_asym = max(a for _, a in e.values())
        good = all(v <= INT_TOL for v, _ in e.values()) and worst_asym <= INT_TOL
        ok &= good
        print(f"   {name:<24} {mol.nao:>4} "
              + " ".join(f"{e[k][0]:>9.2e}" for k in ("S", "T", "V", "MU"))
              + f" {worst_asym:>9.2e}  {'PASS' if good else 'FAIL'}")
    return ok


def check_flips(mols) -> bool:
    print("3. each sign of hgp1e.FORM flipped alone: caught where it matters")
    ok = True
    for key in hgp1e.FORM:
        hgp1e.FORM[key] = -1.0
        try:
            caught = {}
            for name, (mol, ref) in mols.items():
                for k, (err, _) in errors(build(mol), ref).items():
                    if err > INT_TOL:
                        caught.setdefault(k, []).append(name.split("/")[0])
        finally:
            hgp1e.FORM[key] = 1.0
        good = bool(caught)
        ok &= good
        where = ", ".join(f"{k} ({len(v)}/{len(mols)})" for k, v in caught.items())
        print(f"   {key:<10} {'caught in ' + where if good else 'NOT CAUGHT'}")
    return ok


def main() -> int:
    t0 = time.time()
    ok = check_boys()
    mols = molecules()
    ok &= check_oracle(mols)
    ok &= check_flips(mols)
    print(f"{'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
