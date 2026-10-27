"""Check the HGP DF tensors: ``python experimental/hgp/df/check.py``.

Prints four tables and exits nonzero on a miss:

1. ``(P|Q)`` and ``(mn|P)`` against libcint, live through PySCF
   (``int2c2e_cart``, ``int3c2e_cart``), on ``experimental/mmd/gradient``'s
   three molecules: a distorted water (no symmetry) in STO-3G and 6-31G and
   HCN in STO-3G, each with ``cc-pvdz-jkfit`` cut to its s and p shells;
2. the same two tensors against the TRACKED libcint fixtures, no PySCF
   (``h2o_ccpvtz_jkfit_metric``, ``h2o_sto3g_jkfit_3c``), each aux basis cut
   to s and p;
3. the cross-engine pin: the same tensors on the same shells against the MMD
   host references ``eri.density_fit.metric.metric_matrix`` and
   ``eri.density_fit.eri3.eri3_tensor`` (``reference.pq``);
4. structure: every element under its ``|.|`` twin, both tensors exactly
   symmetric, every ``(ba|P)`` block the transpose of ``(ab|P)`` to rounding
   (the sweep only evaluates ``a <= b``), and a d shell on either side
   refused.

Every gate is relative to ``max|X|`` of the tensor it checks, stated before
the numbers and not tuned to them. Table 1 needs the ``oracle`` group (PySCF,
``uv sync --group oracle``) and is skipped, said so, without it; tables 2-4
need only numpy.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "test"), str(ROOT / "src")]

from df import (  # noqa: E402
    metric,
    three_center,
    three_center_block,
)
from oracle import basis as ob  # noqa: E402
from oracle import manifest  # noqa: E402

from felizjueves.ao.shell import Basis, Shell  # noqa: E402
from felizjueves.eri.density_fit.eri3 import eri3_tensor  # noqa: E402
from felizjueves.eri.density_fit.metric import metric_matrix  # noqa: E402
from felizjueves.reference.boys import boys  # noqa: E402
from felizjueves.reference.pq import pq  # noqa: E402

AUXBASIS = "cc-pvdz-jkfit"
LIBCINT_TOL = 1e-12  # of max|X|, against libcint (live or tracked)
MMD_TOL = 1e-13  # of max|X|, HGP against MMD on the same shells
SWAP_TOL = 1e-14  # of max|X|, (ba|P) against (ab|P) transposed
CASES = (
    ("H2O/STO-3G", "O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5",
     "sto-3g"),
    ("H2O/6-31G", "O 0 0 0.1; H 0.1 -0.757 0.587; H -0.05 0.8 0.5", "6-31g"),
    ("HCN/STO-3G", "H 0 0 -1.064; C 0 0 0; N 0 0 1.156", "sto-3g"),
)

failures: list[str] = []


def _gate(name: str, worst: float, tol: float) -> str:
    ok = worst <= tol
    if not ok:
        failures.append(f"{name}: {worst:.2e} > {tol:.0e}")
    return f"{worst:.2e} {'ok' if ok else 'FAIL'}"


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _sp(basis: Basis) -> tuple[list[Shell], np.ndarray]:
    """The s and p shells of ``basis`` and the AO indices they occupy."""
    keep, idx = [], []
    for s, o in zip(basis.shells, basis.offsets, strict=False):
        if s.l <= 1:
            keep.append(s)
            idx.extend(range(o, o + s.n_cart))
    return keep, np.array(idx)


# --- 1. against libcint, live ----------------------------------------------


def table_libcint() -> list[tuple]:
    try:
        from pyscf import df as pdf
        from pyscf import gto
    except ImportError:
        print("\n1. against libcint (live): SKIPPED, no PySCF "
              "(uv sync --group oracle)")
        return []
    print(f"\n1. against libcint (live), aux {AUXBASIS} cut to s and p "
          f"(gate {LIBCINT_TOL:.0e} of max|X|)")
    print(f"{'case':<12} {'nao':>4} {'naux':>5} {'(P|Q)':>14} {'(mn|P)':>14}"
          f" {'time':>7}")
    built = []
    for name, atom, bas in CASES:
        mol = gto.M(atom=atom, basis=bas, cart=True, verbose=0)
        auxm = pdf.make_auxmol(mol, AUXBASIS)
        AO = ob.from_mole(mol).shells
        X, idx = _sp(ob.from_mole(auxm))
        t = time.time()
        M = metric(X, boys=boys)
        T = three_center(AO, X, boys=boys)
        dt = time.time() - t
        Mref = auxm.intor("int2c2e_cart")[np.ix_(idx, idx)]
        Tref = pdf.incore.aux_e2(mol, auxm, "int3c2e_cart",
                                 aosym="s1")[:, :, idx]
        Tref = Tref.reshape(mol.nao, mol.nao, len(idx))
        print(f"{name:<12} {mol.nao:>4} {len(idx):>5} "
              f"{_gate(f'{name} (P|Q)', _rel(M.value, Mref), LIBCINT_TOL):>14}"
              f" {_gate(f'{name} (mn|P)', _rel(T.value, Tref), LIBCINT_TOL):>14}"
              f" {dt:>6.1f}s")
        built.append((name, AO, X, M, T))
    return built


# --- 2. against the tracked fixtures ---------------------------------------


def table_fixtures() -> list[tuple]:
    print(f"\n2. against the tracked libcint fixtures, aux cut to s and p "
          f"(gate {LIBCINT_TOL:.0e} of max|X|)")
    built = []
    ms = manifest.metric_set("h2o_ccpvtz_jkfit_metric")
    X, idx = _sp(ob.from_fixture(ms))
    M = metric(X, boys=boys)
    w = _rel(M.value, ms.metric[np.ix_(idx, idx)])
    print(f"(P|Q)  h2o_ccpvtz_jkfit_metric  naux {len(idx):>3} of {ms.nao}: "
          f"{_gate('fixture (P|Q)', w, LIBCINT_TOL)}")
    built.append(("fixture (P|Q)", None, X, M, None))
    ts = manifest.three_center_set("h2o_sto3g_jkfit_3c")
    AO = list(ob.from_fixture(ts.parent).shells)
    X, idx = _sp(ob.from_fixture(ts))
    T = three_center(AO, X, boys=boys)
    w = _rel(T.value, ts.tensor[:, :, idx])
    print(f"(mn|P) h2o_sto3g_jkfit_3c       naux {len(idx):>3} of {ts.nao}: "
          f"{_gate('fixture (mn|P)', w, LIBCINT_TOL)}")
    built.append(("fixture (mn|P)", AO, X, None, T))
    return built


# --- 3. against MMD ---------------------------------------------------------


def table_mmd(built: list[tuple]) -> None:
    print(f"\n3. HGP against the MMD host references, same shells "
          f"(gate {MMD_TOL:.0e} of max|X|)")
    for name, AO, X, M, T in built:
        cells = []
        if M is not None:
            ref = np.array(metric_matrix(Basis(tuple(X)), pq=pq))
            w = _rel(M.value, ref)
            cells.append(f"(P|Q) {_gate(f'{name} MMD (P|Q)', w, MMD_TOL)}")
        if T is not None:
            ref = np.array(eri3_tensor(Basis(tuple(AO)), Basis(tuple(X)),
                                       pq=pq))
            w = _rel(T.value, ref)
            cells.append(f"(mn|P) {_gate(f'{name} MMD (mn|P)', w, MMD_TOL)}")
        print(f"{name:<16} " + "   ".join(cells))


# --- 4. structure -----------------------------------------------------------


def table_structure(built: list[tuple]) -> None:
    print(f"\n4. structure: |X| <= twin, symmetry exact, (ba|P) = (ab|P)^T "
          f"(gate {SWAP_TOL:.0e} of max|X|), d refused")
    for name, AO, X, M, T in built:
        cells = []
        for tag, D in (("(P|Q)", M), ("(mn|P)", T)):
            if D is None:
                continue
            bounded = bool(np.all(D.scale >= np.abs(D.value)))
            sym = (np.array_equal(D.value, D.value.T) if D.value.ndim == 2
                   else np.array_equal(D.value, D.value.transpose(1, 0, 2)))
            if not bounded:
                failures.append(f"{name} {tag}: an element exceeds its twin")
            if not sym:
                failures.append(f"{name} {tag}: not exactly symmetric")
            cells.append(f"{tag} twin {'ok' if bounded else 'FAIL'}, "
                         f"sym {'ok' if sym else 'FAIL'}")
        if T is not None:
            worst = 0.0
            for i, sa in enumerate(AO):
                for sb in AO[i + 1:]:
                    for sp in X:
                        ab, _ = three_center_block(sa, sb, sp, boys=boys)
                        ba, _ = three_center_block(sb, sa, sp, boys=boys)
                        worst = max(worst, float(np.max(np.abs(
                            ba.transpose(1, 0, 2) - ab))))
            worst /= float(np.max(np.abs(T.value)))
            cells.append(f"swap {_gate(f'{name} swap', worst, SWAP_TOL)}")
        print(f"{name:<16} " + "; ".join(cells))
    d = Shell(2, (0.0, 0.0, 0.0), (1.0,), (1.0,))
    s = Shell(0, (0.0, 0.0, 1.0), (1.0,), (1.0,))
    refused = 0
    for call in (lambda: metric((d, s), boys=boys),
                 lambda: three_center((d,), (s,), boys=boys),
                 lambda: three_center((s,), (d,), boys=boys)):
        try:
            call()
        except ValueError:
            refused += 1
    if refused != 3:
        failures.append(f"d refusal: {refused} of 3 refused")
    print(f"d shell refused: {refused} of 3 {'ok' if refused == 3 else 'FAIL'}")


def main() -> int:
    t = time.time()
    built = table_libcint()
    fixtures = table_fixtures()
    table_mmd(built + fixtures)
    table_structure(built + fixtures)
    print(f"\n{time.time() - t:.0f} s")
    if failures:
        print("FAIL\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
