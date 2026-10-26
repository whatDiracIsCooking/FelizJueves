"""Check the fused Rys Fock build: ``python experimental/rys/check_f.py``.

Prints three tables and exits nonzero on a miss:

1. bookkeeping: G, J and K of the fused sweep against the dense contractions
   over every ordered quartet of the same Rys blocks, for a non-symmetric D
   (which tells K from its transpose), and against ``jbuild`` - ``kbuild``/2;
2. G and E_2 = (1/2) Tr(D G) against PySCF's ``get_jk`` (libcint,
   ``hermi=0``) ``vj - vk/2``, on ``check_j.py``'s three cases, with the block
   evaluations the fused sweep makes against the separate J and K builds'
   (every evaluation counted, half of them by construction);
3. the screen on G, on ``check_j.py``'s chain of four H2: for each tau, the
   quartets skipped (beside the J and K builds' own screens) and
   |G(tau) - G(0)|, |J(tau) - J(0)|, |K(tau) - K(0)| against the build's
   elementwise bounds, which must hold with no tolerance but rounding (the
   same slack of 1e-14 of max).

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
from fbuild import EXCHANGE_FRACTION, fock_energy, fock_matrix  # noqa: E402
from jbuild import ao_offsets, coulomb_matrix  # noqa: E402
from kbuild import exchange_matrix  # noqa: E402
from rys2d import contracted_block, shells_of  # noqa: E402

G_TOL = J_TOL  # of max|G|, against libcint


class Counted:
    """``contracted_block``, counting its calls."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, *sh):
        self.n += 1
        return contracted_block(*sh)


class Cached:
    """``contracted_block``, each shell quartet evaluated once per instance,
    so a tau ladder walks one set of blocks."""

    def __init__(self) -> None:
        self.cache: dict = {}

    def __call__(self, *sh):
        key = tuple(map(id, sh))
        if key not in self.cache:
            self.cache[key] = contracted_block(*sh)
        return self.cache[key]


def _rel(x, ref) -> float:
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


def check_bookkeeping() -> bool:
    print(f"1. the fused sweep against the dense contractions over every "
          f"ordered quartet, same Rys\n   blocks, non-symmetric random D; "
          f"tolerance {BOOK_TOL:.0e} of max")
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
    dJ = np.einsum("abcd,cd->ab", eri, D)
    dK = np.einsum("abcd,bc->ad", eri, D)
    dG = dJ - EXCHANGE_FRACTION * dK
    f = fock_matrix(shells, D)
    sep = _rel(f.K, dK.T)
    split = coulomb_matrix(shells, D).J - EXCHANGE_FRACTION * exchange_matrix(
        shells, D
    ).K
    rows = [
        ("G against dense", _rel(f.G, dG)),
        ("J against dense", _rel(f.J, dJ)),
        ("K against dense", _rel(f.K, dK)),
        ("G against jbuild - kbuild/2", _rel(f.G, split)),
    ]
    ok = sep > 1e3 * BOOK_TOL
    print(f"   H2O/STO-3G, {len(shells)} shells, {n} AOs "
          f"(K against the transpose {sep:.2e}):")
    for name, err in rows:
        ok &= err <= BOOK_TOL
        print(f"   {name:>30} {err:9.2e}")
    print(f"   {'ok' if ok else 'MISS'}\n")
    return ok


def check_libcint() -> bool:
    from pyscf import scf

    print(f"2. G = J - K/2 against PySCF get_jk vj - vk/2 (libcint, hermi=0); "
          f"tolerance {G_TOL:.0e}\n   of max, E_2 {E_TOL:.0e} relative; "
          f"block evaluations fused against J + K built apart")
    cases = []
    for basis in ("sto-3g", "6-31g"):
        mol = _mol(H2O, basis)
        cases.append((f"H2O/{basis.upper()}, RHF D", mol, _rhf_density(mol)))
    _, cno = build_mols()
    cases.append(("C N O 6-31G valence, random D", cno, _random_density(cno.nao, 2)))
    print(f"   {'case':>30} {'AOs':>4} {'G err':>9} {'J err':>9} {'K err':>9} "
          f"{'E_2':>13} {'E_2 err':>9} {'evals':>11} {'s fused':>7} "
          f"{'s J+K':>6}")
    ok = True
    for name, mol, D in cases:
        shells = shells_of(mol)
        _check_offsets(mol, shells)
        fused = Counted()
        t0 = time.perf_counter()
        f = fock_matrix(shells, D, eri=fused)
        t_fused = time.perf_counter() - t0
        apart = Counted()
        t0 = time.perf_counter()
        coulomb_matrix(shells, D, eri=apart)
        exchange_matrix(shells, D, eri=apart)
        t_apart = time.perf_counter() - t0
        vj, vk = scf.hf.get_jk(mol, D, hermi=0)
        g_ref = vj - EXCHANGE_FRACTION * vk
        g_err, j_err, k_err = _rel(f.G, g_ref), _rel(f.J, vj), _rel(f.K, vk)
        e, e_ref = fock_energy(D, f.G), fock_energy(D, g_ref)
        e_err = abs(e - e_ref) / abs(e_ref)
        once = fused.n == f.n_quartets and apart.n == 2 * f.n_quartets
        ok &= max(g_err, j_err, k_err) <= G_TOL and e_err <= E_TOL and once
        print(f"   {name:>30} {mol.nao:>4} {g_err:9.2e} {j_err:9.2e} "
              f"{k_err:9.2e} {e:13.8f} {e_err:9.2e} "
              f"{fused.n:>5}/{apart.n:<5} {t_fused:7.1f} {t_apart:6.1f}"
              f"{'' if once else '  MISS'}")
    print(f"   {'ok' if ok else 'MISS'}\n")
    return ok


def check_screen() -> bool:
    print("3. the screen on G, four H2 along z, 3 A apart, 6-31G, RHF D:\n"
          "   |X(tau) - X(0)| against the build's bound for X = G, J, K, "
          "which must hold\n   elementwise (slack "
          f"{ROUND_SLACK:.0e} of max|X| for rounding); skips beside J's and "
          "K's own screens")
    mol = _mol(H2_CHAIN, "6-31g")
    D = _rhf_density(mol)
    shells = shells_of(mol)
    eri = Cached()
    f0 = fock_matrix(shells, D, eri=eri)
    print(f"   {'tau':>8} {'skipped':>13} {'J skips':>7} {'K skips':>7} "
          f"{'max|dG|':>9} {'max bound':>9} {'dG/bound':>8} "
          f"{'dJ/bJ':>6} {'dK/bK':>6}")
    ok = True
    for tau in TAU_LADDER:
        f = fock_matrix(shells, D, tau=tau, eri=eri)
        nj = coulomb_matrix(shells, D, tau=tau, eri=eri).n_skipped
        nk = exchange_matrix(shells, D, tau=tau, eri=eri).n_skipped
        ratios = []
        holds = True
        for x, x0, b in (
            (f.G, f0.G, f.bound_G),
            (f.J, f0.J, f.bound_J),
            (f.K, f0.K, f.bound_K),
        ):
            dx = np.abs(x - x0)
            holds &= bool(np.all(dx <= b + ROUND_SLACK * np.max(np.abs(x0))))
            nz = b > 0
            ratios.append(float(np.max(dx[nz] / b[nz])) if nz.any() else 0.0)
        ok &= holds
        dg = np.abs(f.G - f0.G)
        print(f"   {tau:8.0e} {f.n_skipped:>6}/{f.n_quartets:<6} {nj:>7} "
              f"{nk:>7} {dg.max():9.2e} {f.bound_G.max():9.2e} "
              f"{ratios[0]:8.3f} {ratios[1]:6.3f} {ratios[2]:6.3f}"
              f"{'' if holds else '  MISS'}")
    print(f"   {'ok' if ok else 'MISS'}: the bounds hold at every tau\n")
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
