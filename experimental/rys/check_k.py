"""Check the Rys K build: ``python experimental/rys/check_k.py``.

Prints three tables and exits nonzero on a miss:

1. bookkeeping: the unique-quartet sweep against the dense contraction
   SUM_bc (ab|cd) D_bc over every ordered quartet of the same Rys blocks, for
   a non-symmetric D; this isolates the orbit digestion from the integrals;
2. K, E_K = (1/2) Tr(D K) and G = J - K/2 against PySCF's ``get_jk``
   (libcint, ``hermi=0``), on the cases of ``check_j.py``: H2O/STO-3G and
   H2O/6-31G with their RHF densities and the C, N, O molecule of
   ``check.py`` with a random non-symmetric D, which tells K's index
   orientation from its transpose;
3. the Schwarz screen on ``check_j.py``'s chain of four H2: for each tau, the
   quartets skipped and |K(tau) - K(0)| against the build's own elementwise
   bound, which must hold with no tolerance but rounding (the same slack of
   1e-14 of max|K|).

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
from check_j import (  # noqa: E402
    BOOK_TOL,
    E_TOL,
    H2_CHAIN,
    H2O,
    J_TOL,
    ROUND_SLACK,
    TAU_LADDER,
    _check_offsets,
    _mol,
    _random_density,
    _rhf_density,
)
from jbuild import ao_offsets, coulomb_matrix  # noqa: E402
from kbuild import exchange_energy, exchange_matrix  # noqa: E402
from rys2d import contracted_block, shells_of  # noqa: E402

K_TOL = J_TOL  # of max|K|, against libcint


def _libcint_jk(mol, D) -> tuple[np.ndarray, np.ndarray]:
    from pyscf import scf

    return scf.hf.get_jk(mol, D, hermi=0)


def check_bookkeeping() -> bool:
    print(f"1. the sweep against the dense contraction over every ordered "
          f"quartet, same Rys blocks,\n   non-symmetric random D; tolerance "
          f"{BOOK_TOL:.0e} of max|K|")
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
    dense = np.einsum("abcd,bc->ad", eri, D)
    swept = exchange_matrix(shells, D).K
    err = np.max(np.abs(swept - dense)) / np.max(np.abs(dense))
    # The transpose is a different matrix for this D: the check can see it.
    sep = np.max(np.abs(swept - dense.T)) / np.max(np.abs(dense))
    ok = err <= BOOK_TOL and sep > 1e3 * BOOK_TOL
    print(f"   H2O/STO-3G, {len(shells)} shells, {n} AOs: {err:.2e} "
          f"(against the transpose {sep:.2e}): {'ok' if ok else 'MISS'}\n")
    return ok


def check_libcint() -> bool:
    print(f"2. K and G = J - K/2 against PySCF get_jk (libcint, hermi=0); "
          f"tolerance {K_TOL:.0e} of max,\n   E_K {E_TOL:.0e} relative")
    cases = []
    for basis in ("sto-3g", "6-31g"):
        mol = _mol(H2O, basis)
        cases.append((f"H2O/{basis.upper()}, RHF D", mol, _rhf_density(mol)))
    _, cno = build_mols()
    cases.append(("C N O 6-31G valence, random D", cno, _random_density(cno.nao, 2)))
    print(f"   {'case':>30} {'AOs':>4} {'max|K|':>9} {'K err':>9} "
          f"{'G err':>9} {'E_K':>13} {'E_K err':>9} {'s':>5}")
    ok = True
    for name, mol, D in cases:
        shells = shells_of(mol)
        _check_offsets(mol, shells)
        t0 = time.perf_counter()
        K = exchange_matrix(shells, D).K
        dt = time.perf_counter() - t0
        J = coulomb_matrix(shells, D).J
        vj, vk = _libcint_jk(mol, D)
        scale = np.max(np.abs(vk))
        err = np.max(np.abs(K - vk)) / scale
        g_ref = vj - 0.5 * vk
        g_err = np.max(np.abs((J - 0.5 * K) - g_ref)) / np.max(np.abs(g_ref))
        e, e_ref = exchange_energy(D, K), exchange_energy(D, vk)
        e_err = abs(e - e_ref) / abs(e_ref)
        ok &= err <= K_TOL and g_err <= K_TOL and e_err <= E_TOL
        print(f"   {name:>30} {mol.nao:>4} {scale:9.3f} {err:9.2e} "
              f"{g_err:9.2e} {e:13.8f} {e_err:9.2e} {dt:5.1f}")
    print(f"   {'ok' if ok else 'MISS'}\n")
    return ok


def check_screen() -> bool:
    print("3. the Schwarz screen, four H2 along z, 3 A apart, 6-31G, RHF D:\n"
          "   |K(tau) - K(0)| against the build's bound, which must hold "
          "elementwise (slack "
          f"{ROUND_SLACK:.0e} of max|K| for rounding)")
    mol = _mol(H2_CHAIN, "6-31g")
    D = _rhf_density(mol)
    shells = shells_of(mol)
    cache: dict = {}

    def eri(*sh):
        key = tuple(map(id, sh))
        if key not in cache:
            cache[key] = contracted_block(*sh)
        return cache[key]

    K0 = exchange_matrix(shells, D, eri=eri).K
    scale = np.max(np.abs(K0))
    print(f"   {'tau':>8} {'skipped':>13} {'max|dK|':>9} {'max bound':>9} "
          f"{'worst dK/bound':>14}")
    ok = True
    for tau in TAU_LADDER:
        b = exchange_matrix(shells, D, tau=tau, eri=eri)
        dk = np.abs(b.K - K0)
        holds = bool(np.all(dk <= b.bound + ROUND_SLACK * scale))
        ok &= holds
        nz = b.bound > 0
        ratio = float(np.max(dk[nz] / b.bound[nz])) if nz.any() else 0.0
        print(f"   {tau:8.0e} {b.n_skipped:>6}/{b.n_quartets:<6} "
              f"{dk.max():9.2e} {b.bound.max():9.2e} {ratio:14.3f}"
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
