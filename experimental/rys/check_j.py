"""Check the Rys J build: ``python experimental/rys/check_j.py``.

Prints three tables and exits nonzero on a miss:

1. bookkeeping: the unique-quartet sweep against the dense contraction
   SUM_cd (ab|cd) D_cd over every ordered quartet of the same Rys blocks, for
   a non-symmetric D; this isolates the sweep's weights from the integrals;
2. J and E_J = (1/2) Tr(D J) against PySCF's ``get_jk`` (libcint), on
   H2O/STO-3G and H2O/6-31G with their RHF densities and on the C, N, O
   molecule of ``check.py`` (contracted 6-31G valence) with a random
   non-symmetric D;
3. the Schwarz screen on a chain of four H2 (6-31G, its RHF density): for
   each tau, the quartets skipped and |J(tau) - J(0)| against the build's own
   elementwise bound, which must hold with no tolerance but rounding (a
   slack of 1e-14 of max|J|: at tau = 1e-12 one element's 4.4e-16 difference,
   two ulps of a 1.92 entry, sits above its 3.1e-16 bound).

Needs the ``oracle`` group (PySCF): ``uv sync --group oracle``.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check import ATOMS, build_mols  # noqa: E402
from jbuild import ao_offsets, coulomb_energy, coulomb_matrix  # noqa: E402
from rys2d import contracted_block, shells_of  # noqa: E402

BOOK_TOL = 1e-14  # of max|J|: the same blocks, summed in another order
J_TOL = 1e-12  # of max|J|, against libcint (the repo's J-against-libcint gate)
E_TOL = 1e-12  # relative
ROUND_SLACK = 1e-14  # of max|J|, on the screen's bound
TAU_LADDER = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)

H2O = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
# Four H2 along z, 3 A apart: extended enough for the screen to skip.
H2_CHAIN = "; ".join(f"H 0 0 {3.0 * i}; H 0 0 {3.0 * i + 0.74}" for i in range(4))


def _mol(atom, basis, **kw):
    from pyscf import gto

    return gto.M(atom=atom, basis=basis, cart=True, verbose=0, **kw)


def _rhf_density(mol) -> np.ndarray:
    from pyscf import scf

    return scf.RHF(mol).run(conv_tol=1e-10).make_rdm1()


def _libcint_j(mol, D) -> np.ndarray:
    from pyscf import scf

    vj, _ = scf.hf.get_jk(mol, D, hermi=0, with_k=False)
    return vj


def _random_density(n: int, seed: int) -> np.ndarray:
    """Non-symmetric on purpose: J sees only (D + D^T) / 2, and the check
    should show the sweep agrees with libcint's hermi=0 path anyway."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (n, n))


def _check_offsets(mol, shells) -> None:
    assert np.array_equal(ao_offsets(shells), mol.ao_loc_nr(cart=True)), (
        "AO offsets disagree with PySCF's"
    )


def check_bookkeeping() -> bool:
    print(f"1. the sweep against the dense contraction over every ordered "
          f"quartet, same Rys blocks,\n   non-symmetric random D; tolerance "
          f"{BOOK_TOL:.0e} of max|J|")
    mol = _mol(H2O, "sto-3g")
    shells = shells_of(mol)
    _check_offsets(mol, shells)
    loc = ao_offsets(shells)
    n = int(loc[-1])
    eri = np.zeros((n, n, n, n))
    for quad in itertools.product(range(len(shells)), repeat=4):
        eri[tuple(slice(loc[i], loc[i + 1]) for i in quad)] = contracted_block(
            *(shells[i] for i in quad)
        )
    D = _random_density(n, 1)
    dense = np.einsum("abcd,cd->ab", eri, D)
    swept = coulomb_matrix(shells, D).J
    err = np.max(np.abs(swept - dense)) / np.max(np.abs(dense))
    ok = err <= BOOK_TOL
    print(f"   H2O/STO-3G, {len(shells)} shells, {n} AOs: {err:.2e}: "
          f"{'ok' if ok else 'MISS'}\n")
    return ok


def check_libcint() -> bool:
    print(f"2. J against PySCF get_jk (libcint); tolerance {J_TOL:.0e} of "
          f"max|J|, E_J {E_TOL:.0e} relative")
    cases = []
    for basis in ("sto-3g", "6-31g"):
        mol = _mol(H2O, basis)
        cases.append((f"H2O/{basis.upper()}, RHF D", mol, _rhf_density(mol)))
    _, cno = build_mols()
    cases.append(("C N O 6-31G valence, random D", cno, _random_density(cno.nao, 2)))
    print(f"   {'case':>30} {'shells':>6} {'AOs':>4} {'max|J|':>9} "
          f"{'J err':>9} {'E_J':>13} {'E_J err':>9} {'s':>5}")
    ok = True
    for name, mol, D in cases:
        shells = shells_of(mol)
        _check_offsets(mol, shells)
        t0 = time.perf_counter()
        J = coulomb_matrix(shells, D).J
        dt = time.perf_counter() - t0
        ref = _libcint_j(mol, D)
        scale = np.max(np.abs(ref))
        err = np.max(np.abs(J - ref)) / scale
        e, e_ref = coulomb_energy(D, J), coulomb_energy(D, ref)
        e_err = abs(e - e_ref) / abs(e_ref)
        ok &= err <= J_TOL and e_err <= E_TOL
        print(f"   {name:>30} {len(shells):>6} {mol.nao:>4} {scale:9.3f} "
              f"{err:9.2e} {e:13.8f} {e_err:9.2e} {dt:5.1f}")
    print(f"   {'ok' if ok else 'MISS'}\n")
    return ok


def check_screen() -> bool:
    print("3. the Schwarz screen, four H2 along z, 3 A apart, 6-31G, RHF D:\n"
          "   |J(tau) - J(0)| against the build's bound, which must hold "
          "elementwise (slack "
          f"{ROUND_SLACK:.0e} of max|J| for rounding)")
    mol = _mol(H2_CHAIN, "6-31g")
    D = _rhf_density(mol)
    shells = shells_of(mol)
    cache: dict = {}

    def eri(*sh):
        key = tuple(map(id, sh))
        if key not in cache:
            cache[key] = contracted_block(*sh)
        return cache[key]

    J0 = coulomb_matrix(shells, D, eri=eri).J
    scale = np.max(np.abs(J0))
    print(f"   {'tau':>8} {'skipped':>13} {'max|dJ|':>9} {'max bound':>9} "
          f"{'worst dJ/bound':>14}")
    ok = True
    for tau in TAU_LADDER:
        b = coulomb_matrix(shells, D, tau=tau, eri=eri)
        dj = np.abs(b.J - J0)
        holds = bool(np.all(dj <= b.bound + ROUND_SLACK * scale))
        ok &= holds
        nz = b.bound > 0
        ratio = float(np.max(dj[nz] / b.bound[nz])) if nz.any() else 0.0
        print(f"   {tau:8.0e} {b.n_skipped:>6}/{b.n_quartets:<6} "
              f"{dj.max():9.2e} {b.bound.max():9.2e} {ratio:14.3f}"
              f"{'' if holds else '  MISS'}")
    print(f"   {'ok' if ok else 'MISS'}: the bound holds at every tau\n")
    return ok


def main() -> int:
    t0 = time.perf_counter()
    print(f"C N O molecule: {ATOMS}\n")
    ok = check_bookkeeping()
    ok &= check_libcint()
    ok &= check_screen()
    print(f"{'PASS' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
