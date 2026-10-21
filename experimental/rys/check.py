"""Settle the Rys spike: ``python experimental/rys/check.py``.

Prints three tables and exits nonzero on a miss:

1. the moment identity SUM_n W_n x_n^m == F_m(T), m <= 2N - 1, against F_m
   from ``scipy.special.gammainc``;
2. the fp64 roots and weights against the 50-digit ``decimal`` route;
3. every ordered (ab|cd) class with each l in {0, 1} against PySCF's
   ``int2e_cart``, uncontracted and contracted, p shells on distinct centres;
   the uncontracted blocks also against the same quadrature at 50 digits,
   which referees a disagreement with libcint.

Needs the ``oracle`` group (PySCF): ``uv sync --group oracle``.
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gamma, gammainc

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rys2d  # noqa: E402
from roots import N_MAX, T_SWITCH, roots_weights, roots_weights_exact  # noqa: E402
from rys2d import Shell, contracted_block, prim_quartet_exact  # noqa: E402

MOMENT_TOL = 4e-15
ROOT_TOL = 1e-14
ERI_TOL = 1e-13

T_GRID = sorted(
    {0.0, 1e-300, 1e-200, 1e-30, 1e-15, 1e-10, 1e-6, 1e-4, 1e-2, 0.1, 0.5}
    | {1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0}
    | {np.nextafter(T_SWITCH, 0.0), T_SWITCH, np.nextafter(T_SWITCH, np.inf)}
    | {60.0, 100.0, 300.0, 1e3, 1e4, 1e5, 1e6}
)


def boys_ref(m: int, T: float) -> float:
    """F_m(T) = Gamma(m+1/2) P(m+1/2, T) / (2 T^(m+1/2)); small T by series."""
    if T < 1e-3:
        return sum(
            (-T) ** k / (math.factorial(k) * (2 * m + 2 * k + 1)) for k in range(8)
        )
    s = m + 0.5
    return gamma(s) * gammainc(s, T) / (2.0 * T**s)


def check_moments() -> bool:
    print(f"1. moment identity, worst |SUM W x^m - F_m| / F_m over m <= 2N-1, "
          f"tolerance {MOMENT_TOL:.0e}")
    print(f"   {'T':>12} " + " ".join(f"{'N=' + str(n):>9}" for n in range(1, 4)))
    worst = 0.0
    for T in T_GRID:
        row = []
        for n in range(1, N_MAX + 1):
            x, w = roots_weights(n, T)
            err = max(
                abs(np.dot(w, x**m) - boys_ref(m, T)) / boys_ref(m, T)
                for m in range(2 * n)
            )
            row.append(err)
            worst = max(worst, err)
        print(f"   {T:12.4g} " + " ".join(f"{e:9.2e}" for e in row))
    ok = worst <= MOMENT_TOL
    print(f"   worst {worst:.2e}: {'ok' if ok else 'MISS'}\n")
    return ok


def check_roots() -> bool:
    print(f"2. fp64 roots/weights against the 50-digit route, worst relative "
          f"error, tolerance {ROOT_TOL:.0e}")
    print(f"   {'T':>12} " + " ".join(f"{'x N=' + str(n):>9} {'W N=' + str(n):>9}"
                                       for n in range(1, 4)))
    worst = 0.0
    self_worst = 0.0
    for T in T_GRID:
        row = []
        for n in range(1, N_MAX + 1):
            x, w = roots_weights(n, T)
            xe, we = roots_weights_exact(n, T, 50)
            # The exact route against itself at 70 digits: how exact it is.
            xe2, we2 = roots_weights_exact(n, T, 70)
            self_worst = max(
                self_worst,
                max(float(abs(a - b) / b) for a, b in zip(xe, xe2, strict=True)),
                max(float(abs(a - b) / b) for a, b in zip(we, we2, strict=True)),
            )
            xe = np.array([float(v) for v in xe])
            we = np.array([float(v) for v in we])
            ex = float(np.max(np.abs(x - xe) / xe))
            ew = float(np.max(np.abs(w - we) / we))
            row += [ex, ew]
            worst = max(worst, ex, ew)
        print(f"   {T:12.4g} " + " ".join(f"{e:9.2e}" for e in row))
    ok = worst <= ROOT_TOL
    print(f"   worst {worst:.2e}: {'ok' if ok else 'MISS'}  "
          f"(50- vs 70-digit route: worst {self_worst:.1e})\n")
    return ok


# --- ERIs against libcint ---------------------------------------------------

ATOMS = "C 0 0 0; N 0.15 -0.2 1.3; O 1.2 0.45 -0.55"
COMMON_FAC = {0: 0.5 / math.sqrt(math.pi), 1: math.sqrt(3.0 / (4.0 * math.pi))}


def shells_of(mol) -> list[Shell]:
    """Every shell of ``mol`` with libcint's normalisation, read from the Mole:
    ``_libcint_ctr_coeff`` (the radial normalisation times the contraction)
    and libcint's s/p common factor (CINTcommon_fac_sp), which int2e applies."""
    out = []
    for ib in range(mol.nbas):
        am = int(mol.bas_angular(ib))
        coeff = mol._libcint_ctr_coeff(ib)
        assert coeff.shape[1] == 1, "generally contracted shell"
        out.append(
            Shell(
                am,
                tuple(float(v) for v in mol.bas_coord(ib)),
                tuple(float(v) for v in mol.bas_exp(ib)),
                tuple(float(v) * COMMON_FAC[am] for v in coeff[:, 0]),
            )
        )
    return out


def build_mols():
    from pyscf import gto

    ref = gto.M(atom=ATOMS, basis="6-31g", cart=True, spin=1)
    # Contracted: each atom's 3-primitive valence s and p shells.
    contracted = {}
    # Uncontracted: per atom, two s and two p primitives with distinct
    # exponents (the valence middle and the outer one), each its own shell.
    uncontracted = {}
    for ia in range(ref.natm):
        sym = ref.atom_pure_symbol(ia)
        shells = [ib for ib in range(ref.nbas) if ref.bas_atom(ib) == ia]
        val = [ib for ib in shells if ref.bas_nprim(ib) == 3]
        outer = [ib for ib in shells if ref.bas_nprim(ib) == 1]
        contracted[sym] = [
            [int(ref.bas_angular(ib))]
            + [[e, c] for e, c in zip(ref.bas_exp(ib), ref.bas_ctr_coeff(ib)[:, 0],
                                      strict=True)]
            for ib in val
        ]
        uncontracted[sym] = []
        for am in (0, 1):
            ib_v = next(ib for ib in val if ref.bas_angular(ib) == am)
            ib_o = next(ib for ib in outer if ref.bas_angular(ib) == am)
            for e in (ref.bas_exp(ib_v)[1], ref.bas_exp(ib_o)[0]):
                uncontracted[sym].append([am, [float(e), 1.0]])
    kw = dict(atom=ATOMS, cart=True, spin=1)
    return gto.M(basis=uncontracted, **kw), gto.M(basis=contracted, **kw)


def _blocks(mol, cls, t_log, truth: bool):
    """(ours, libcint, truth-or-None, p on more than one centre) per block."""
    eri = mol.intor("int2e_cart")
    loc = mol.ao_loc_nr(cart=True)
    shells = shells_of(mol)
    by_l = {am: [i for i, s in enumerate(shells) if s.am == am] for am in (0, 1)}
    for quad in itertools.product(*(by_l[am] for am in cls)):
        sh = [shells[i] for i in quad]
        ours = contracted_block(*sh, t_log=t_log)
        theirs = eri[tuple(slice(loc[i], loc[i + 1]) for i in quad)]
        exact = None
        if truth:  # single-primitive shells only
            k = math.prod(s.coeffs[0] for s in sh)
            exact = k * prim_quartet_exact(
                cls, [s.centre for s in sh], [s.exps[0] for s in sh]
            )
        atoms = [mol.bas_atom(i) for i in quad]
        p_atoms = {a for a, am in zip(atoms, cls, strict=True) if am == 1}
        yield ours, theirs, exact, len(p_atoms) > 1


def check_eris() -> bool:
    print(f"3. (ab|cd) against PySCF int2e_cart, worst max|difference| / "
          f"max|libcint| per block, tolerance {ERI_TOL:.0e}")
    print("   A block libcint gives as exactly zero (parity) is held to the "
          "class's largest entry.\n"
          "   Uncontracted blocks are also evaluated at 50 digits ('truth': the "
          "same quadrature\n   in decimal, exact roots). A block that misses "
          "libcint passes only if it is within\n   tolerance of the truth AND "
          "libcint is the farther of the two from it ('referred').")
    ok = True
    t_seen: list[float] = []
    rows = {}
    for label, mol in zip(("uncontracted", "contracted"), build_mols(), strict=True):
        truth = label == "uncontracted"
        for cls in itertools.product((0, 1), repeat=4):
            blocks = list(_blocks(mol, cls, t_seen, truth))
            scale = max(np.max(np.abs(b[1])) for b in blocks)
            r = dict(lib=0.0, truth=0.0, lib_truth=0.0, referred=0, n=len(blocks),
                     distinct=sum(b[3] for b in blocks),
                     zero=sum(np.max(np.abs(b[1])) == 0.0 for b in blocks))
            for ours, theirs, exact, _ in blocks:
                ref = np.max(np.abs(theirs)) or scale
                d_lib = np.max(np.abs(ours - theirs)) / ref
                r["lib"] = max(r["lib"], d_lib)
                if exact is None:
                    ok &= d_lib <= ERI_TOL
                    continue
                d_truth = np.max(np.abs(ours - exact)) / ref
                lib_truth = np.max(np.abs(theirs - exact)) / ref
                r["truth"] = max(r["truth"], d_truth)
                r["lib_truth"] = max(r["lib_truth"], lib_truth)
                ok &= d_truth <= ERI_TOL
                if d_lib > ERI_TOL:
                    referred = lib_truth > d_truth
                    r["referred"] += referred
                    ok &= referred
            rows[label, cls] = r
    names = "sp"
    print(f"\n   {'':>9} {'':>2} {'---------- uncontracted ----------':>45}"
          f" {'contracted':>11}")
    print(f"   {'class':>9} {'N':>2} {'vs libcint':>11} {'vs truth':>9}"
          f" {'libcint vs truth':>17} {'referred':>9} {'vs libcint':>11}"
          f" {'blocks':>7} {'p >1 ctr':>9} {'zero':>5}")
    for cls in itertools.product((0, 1), repeat=4):
        name = f"({names[cls[0]]}{names[cls[1]]}|{names[cls[2]]}{names[cls[3]]})"
        u = rows["uncontracted", cls]
        c = rows["contracted", cls]
        print(f"   {name:>9} {sum(cls) // 2 + 1:>2} {u['lib']:11.2e} {u['truth']:9.2e}"
              f" {u['lib_truth']:17.2e} {u['referred']:>9} {c['lib']:11.2e}"
              f" {u['n'] + c['n']:>7} {u['distinct'] + c['distinct']:>9}"
              f" {u['zero'] + c['zero']:>5}")
    t = np.array(t_seen)
    print(f"   T over the primitive quartets: min {t.min():.2e}, max {t.max():.2e}, "
          f"{np.mean(t >= T_SWITCH) * 100:.2f}% on the large-T branch")
    worst_c = max(r["lib"] for (lab, _), r in rows.items() if lab == "contracted")
    worst_t = max(r["truth"] for (lab, _), r in rows.items() if lab == "uncontracted")
    worst_u = max(r["lib"] for (lab, _), r in rows.items() if lab == "uncontracted")
    referred = sum(r["referred"] for r in rows.values())
    print(f"   contracted vs libcint worst {worst_c:.2e}; uncontracted vs truth "
          f"worst {worst_t:.2e}, vs libcint worst {worst_u:.2e} "
          f"({referred} blocks referred): {'ok' if ok else 'MISS'}\n")
    return ok


def check_mutations() -> bool:
    """Flip each sign of the form in turn; the settled form must be the only
    one that agrees, and every flip must be caught by some ordered class."""
    print("4. each sign of the form flipped alone: worst error vs libcint per "
          "class (uncontracted,\n   the first quartet on three centres with A != B "
          "and C != D); '.' is under 1e-10")
    mol, _ = build_mols()
    eri = mol.intor("int2e_cart")
    loc = mol.ao_loc_nr(cart=True)
    shells = shells_of(mol)
    by_l = {am: [i for i, s in enumerate(shells) if s.am == am] for am in (0, 1)}
    classes = list(itertools.product((0, 1), repeat=4))
    picks = {}
    for cls in classes:
        for quad in itertools.product(*(by_l[am] for am in cls)):
            at = [mol.bas_atom(i) for i in quad]
            if at[0] != at[1] and at[2] != at[3] and len(set(at)) == 3:
                picks[cls] = quad
                break
    names = "sp"
    head = " ".join(
        f"{names[c[0]]}{names[c[1]]}{names[c[2]]}{names[c[3]]}" for c in classes
    )
    print(f"   {'flipped':>8} {head}")
    ok = True
    for key in rys2d.FORM:
        rys2d.FORM[key] = -1.0
        try:
            errs = []
            for cls in classes:
                quad = picks[cls]
                ours = contracted_block(*(shells[i] for i in quad))
                theirs = eri[tuple(slice(loc[i], loc[i + 1]) for i in quad)]
                errs.append(np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs)))
        finally:
            rys2d.FORM[key] = 1.0
        caught = sum(e > 1e-10 for e in errs)
        ok &= caught > 0
        cells = " ".join(f"{'X' if e > 1e-10 else '.':>4}" for e in errs)
        print(f"   {key:>8} {cells}  caught by {caught}/16")
    print(f"   {'ok' if ok else 'MISS'}: every flip is caught\n")
    return ok


def main() -> int:
    t0 = time.perf_counter()
    ok = check_moments()
    ok &= check_roots()
    ok &= check_eris()
    ok &= check_mutations()
    print(f"{'PASS' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
