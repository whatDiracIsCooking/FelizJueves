"""`hessian.host_df`: the contracted nuclear Hessian of `(mn|P)`, against
an oracle independent of the second-derivative rule.

THE ORACLE is a central finite difference of `gradient.host_df.
eri3_gradient` -- itself held to finite differences of the
libcint-witnessed `(mn|P)` (test/test_gradient_host_df.py) -- in every
coordinate of every atom, the atom's AO shells, aux shells and nucleus
moved together:

    H[:, :, B, l]   ~   (G(R + h e_{B,l}) - G(R - h e_{B,l})) / 2h

with `h = 1e-4` bohr, so the truncation is `O(h^2)`: measured ~5e-9 of
`max|H|` on these inputs, held to `FD_TOL = 1e-7`. An order-unity error (a
wrong `4 alpha^2`, a dropped `i (i - 1) f_{i-2}`, a missing bra-aux block)
is ~1 of `max|H|`. `Gam` is a random tensor, NOT symmetric in `(m, n)`
(the weight rule claims any `Gam`).

TRANSLATIONAL INVARIANCE, `SUM_A H[A, k, B, l] = 0`, holds only if every
centre's derivative is right, since none is inferred from another:
`INVAR_TOL = 1e-12` of `max|H|`. SYMMETRY, `H = H^T`, holds by
construction for the blocks written twice and must hold for the
same-centre blocks: `SYM_TOL = 1e-13`.

The fast tier: a three-atom basis with s, p AO shells and s, p, d aux
shells (every class through `(pp|d)`, L <= 4 at d2 + 2, see
test/test_kernel_d2eri3.py for the tier cut). `slow`: H2O/STO-3G against
the cc-pVTZ-JKFIT s and p shells.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.shell import Basis, Shell
from felizjueves.gradient import host_df as grad
from felizjueves.gradient.host import shell_atoms
from felizjueves.hessian import host_df as host
from felizjueves.reference.pq import pq

FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|H|: the O(h^2) truncation, ~5e-9 measured
INVAR_TOL = 1e-12  # of max|H|
SYM_TOL = 1e-13  # of max|H|

_A, _B, _C = (0.1, -0.2, 0.3), (1.2, 0.4, -0.5), (-0.6, 0.9, 0.2)
NUCLEI = ((3.0, _A), (1.0, _B), (2.0, _C))
BASIS = Basis((Shell(0, _A, (1.3,), (0.9,)), Shell(1, _A, (0.8,), (1.1,)),
               Shell(1, _B, (1.7, 0.4), (0.5, 0.6)),
               Shell(0, _C, (0.6,), (1.0,))))
AUX = Basis((Shell(0, _C, (1.1,), (0.8,)), Shell(1, _B, (0.7,), (1.0,)),
             Shell(2, _A, (1.5,), (1.0,))))


def _gam(basis, aux, seed=3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(
        (basis.n_ao, basis.n_ao, aux.n_ao))


@cache
def _h2o():
    ts = manifest.three_center_set("h2o_sto3g_jkfit_3c")
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return basis, Basis(tuple(s for s in aux.shells if s.l <= 1)), nuclei


def _case(name: str):
    return (BASIS, AUX, NUCLEI) if name == "spd" else _h2o()


def _moved(basis: Basis, atoms, A: int, k: int, h: float) -> Basis:
    return Basis(tuple(
        replace(s, centre=tuple(c + h * (d == k)
                                for d, c in enumerate(s.centre)))
        if atoms[i] == A else s for i, s in enumerate(basis.shells)))


def _fd(basis, aux, nuclei, Gam) -> np.ndarray:
    at, aat = shell_atoms(basis, nuclei), shell_atoms(aux, nuclei)
    n = len(nuclei)
    H = np.zeros((n, 3, n, 3))
    for B in range(n):
        for l in range(3):  # noqa: E741
            g = []
            for h in (FD_H, -FD_H):
                nuc = tuple((Z, tuple(c + h * (d == l)
                                      for d, c in enumerate(C)))
                            if a == B else (Z, C)
                            for a, (Z, C) in enumerate(nuclei))
                g.append(np.array(grad.eri3_gradient(
                    _moved(basis, at, B, l, h), _moved(aux, aat, B, l, h),
                    Gam, nuc, pq=pq)))
            H[:, :, B, l] = (g[0] - g[1]) / (2 * FD_H)
    return H


CASES = [pytest.param("spd", id="spd"),
         pytest.param("h2o", id="h2o_sto3g-jkfit_sp", marks=pytest.mark.slow)]


@pytest.mark.parametrize("name", CASES)
def test_the_hessian_is_the_finite_difference_of_the_gradient(name) -> None:
    basis, aux, nuclei = _case(name)
    Gam = _gam(basis, aux)
    H = np.array(host.eri3_hessian(basis, aux, Gam, nuclei, pq=pq))
    scale = np.abs(H).max()
    assert scale > 0
    err = np.abs(H - _fd(basis, aux, nuclei, Gam)).max()
    assert err <= FD_TOL * scale, f"{name}: {err / scale:.2e}"


@pytest.mark.parametrize("name", CASES)
def test_translational_invariance_and_symmetry(name) -> None:
    basis, aux, nuclei = _case(name)
    H = np.array(host.eri3_hessian(basis, aux, _gam(basis, aux), nuclei,
                                   pq=pq))
    scale = np.abs(H).max()
    assert np.abs(H.sum(axis=0)).max() <= INVAR_TOL * scale
    assert np.abs(H.sum(axis=2)).max() <= INVAR_TOL * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= SYM_TOL * scale


def test_the_absolute_twin_bounds_the_hessian() -> None:
    Gam = _gam(BASIS, AUX)
    H = np.array(host.eri3_hessian(BASIS, AUX, Gam, NUCLEI, pq=pq))
    T = np.array(host.eri3_hessian(BASIS, AUX, Gam, NUCLEI, pq=pq,
                                   absolute=True))
    assert (T >= 0).all() and (np.abs(H) <= T * (1 + 1e-12)).all()


def test_the_cells_partition_the_hessian() -> None:
    Gam = _gam(BASIS, AUX)
    whole = np.array(host.eri3_hessian(BASIS, AUX, Gam, NUCLEI, pq=pq))
    ns, na = len(BASIS.shells), len(AUX.shells)
    parts = sum(np.array(host.eri3_hessian(BASIS, AUX, Gam, NUCLEI, pq=pq,
                                           cells=[(i, j, n)]))
                for i in range(ns) for j in range(i, ns) for n in range(na))
    assert np.abs(parts - whole).max() <= 1e-14 * np.abs(whole).max()


def test_a_cell_that_is_not_i_le_j_is_refused() -> None:
    with pytest.raises(ValueError, match="is not i <= j"):
        host.eri3_hessian(BASIS, AUX, _gam(BASIS, AUX), NUCLEI, pq=pq,
                          cells=[(1, 0, 0)])
