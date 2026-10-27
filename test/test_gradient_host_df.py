"""`gradient.host_df`: the contracted nuclear gradients of `(P|Q)` and
`(mn|P)`, against an oracle independent of the derivative rule.

THE ORACLE is a central finite difference of this repo's own integrals --
`eri.density_fit.metric.metric_matrix` and `eri.density_fit.eri3.eri3_tensor`,
each witnessed against libcint's `int2c2e_cart` / `int3c2e_cart` through the
tracked fixtures (test_ao_metric.py, test_ao_eri3.py) -- in every coordinate
of every atom, the atom's AO AND aux shells moved together:

    d/dR_{A,k} SUM W X(R)   ~   (f(R + h e) - f(R - h e)) / 2h,   h = 1e-4

`O(h^2)`: measured ~2e-9 to 5e-9 of `max|G|` here, held to `FD_TOL = 1e-7`
of `max|G|`. An order-unity error (a wrong `2 alpha`, a dropped `- a E^{a-1}`,
a missing centre) is ~1 of `max|G|`. `W` and `Gam` are random, `Gam` not
symmetric in `(m, n)` (the weight rule claims any).

THE DF-J ENERGY. `E_J = 1/2 b.d`, `b_P = SUM_mn (P|mn) D_mn`, `(P|Q) d = b`,
whose gradient at fixed `d` (it is stationary in `d`) is
`eri3_gradient(D (x) d) + metric_gradient(-1/2 d d^T)`: held to a finite
difference of `E_J` itself, the solve redone at every displaced geometry.

TRANSLATIONAL INVARIANCE, `SUM_A G[A] = 0`, is a check (no centre is
inferred from another): `INVAR_TOL = 1e-13` of `max|G|`.

The fast tier: a three-atom basis with s, p and d AO shells and s, p, d aux
shells (every class through L = 4 on both integrals). `slow`: the DF-J
energy of H2O/STO-3G against the cc-pVTZ-JKFIT s and p shells.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit.eri3 import eri3_classes, eri3_tensor
from felizjueves.eri.density_fit.metric import metric_classes, metric_matrix
from felizjueves.gradient import host_df
from felizjueves.gradient.host import shell_atoms
from felizjueves.reference.pq import pq

FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|G|
INVAR_TOL = 1e-13  # of max|G|

_A, _B, _C = (0.1, -0.2, 0.3), (1.2, 0.4, -0.5), (-0.6, 0.9, 0.2)
NUCLEI = ((8.0, _A), (1.0, _B), (1.0, _C))
BASIS = Basis((Shell(0, _A, (1.3, 0.4), (0.9, 0.3)),
               Shell(1, _A, (0.8,), (1.1,)),
               Shell(2, _B, (0.6,), (0.7,)),
               Shell(0, _B, (1.7,), (0.5,))))
AUX = Basis((Shell(0, _C, (2.0, 0.4), (0.7, 0.5)),
             Shell(1, _A, (0.85,), (1.0,)),
             Shell(2, _B, (1.4,), (0.6,)),
             Shell(0, _A, (3.0,), (1.0,))))


def _moved(basis, nuclei, A, k, h):
    at = shell_atoms(basis, nuclei)
    return Basis(tuple(
        replace(s, centre=tuple(c + h * (d == k)
                                for d, c in enumerate(s.centre)))
        if at[i] == A else s for i, s in enumerate(basis.shells)))


def _fd(basis, aux, nuclei, f) -> np.ndarray:
    G = np.zeros((len(nuclei), 3))
    for A in range(len(nuclei)):
        for k in range(3):
            p = f(_moved(basis, nuclei, A, k, FD_H),
                  _moved(aux, nuclei, A, k, FD_H))
            m = f(_moved(basis, nuclei, A, k, -FD_H),
                  _moved(aux, nuclei, A, k, -FD_H))
            G[A, k] = (p - m) / (2 * FD_H)
    return G


def _check(G, want, what) -> None:
    scale = np.abs(G).max()
    err = np.abs(G - want).max() / scale
    print(f"\n{what}: {err:.2e} of max|G| against the finite difference")
    assert err <= FD_TOL, what
    assert np.abs(G.sum(axis=0)).max() <= INVAR_TOL * scale, what


def test_the_bases_reach_every_class_through_d() -> None:
    assert {(c.lP, c.lQ) for c in metric_classes(AUX)} == {
        (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
    assert max(c.la + c.lb + c.lP for c in eri3_classes(BASIS, AUX)) == 6


def test_the_metric_gradient_is_the_finite_difference() -> None:
    W = np.random.default_rng(3).standard_normal((AUX.n_ao, AUX.n_ao))
    G = np.array(host_df.metric_gradient(AUX, W, NUCLEI, pq=pq))
    _check(G, _fd(BASIS, AUX, NUCLEI, lambda b, x: float(
        np.sum(W * np.array(metric_matrix(x, pq=pq))))), "(P|Q)")


@pytest.mark.slow
def test_the_eri3_gradient_is_the_finite_difference() -> None:
    Gam = np.random.default_rng(4).standard_normal(
        (BASIS.n_ao, BASIS.n_ao, AUX.n_ao))
    G = np.array(host_df.eri3_gradient(BASIS, AUX, Gam, NUCLEI, pq=pq))
    _check(G, _fd(BASIS, AUX, NUCLEI, lambda b, x: float(
        np.sum(Gam * np.array(eri3_tensor(b, x, pq=pq))))), "(mn|P)")


def _sp(basis: Basis) -> Basis:
    return Basis(tuple(s for s in basis.shells if s.l <= 1))


def test_the_eri3_gradient_of_s_and_p_shells_is_the_finite_difference(
        ) -> None:
    basis, aux = _sp(BASIS), _sp(AUX)
    Gam = np.random.default_rng(4).standard_normal(
        (basis.n_ao, basis.n_ao, aux.n_ao))
    G = np.array(host_df.eri3_gradient(basis, aux, Gam, NUCLEI, pq=pq))
    _check(G, _fd(basis, aux, NUCLEI, lambda b, x: float(
        np.sum(Gam * np.array(eri3_tensor(b, x, pq=pq))))), "(mn|P) s, p")


def _density(n: int) -> np.ndarray:
    X = np.random.default_rng(7).standard_normal((n, n))
    return X @ X.T / n


def _dfj(basis, aux, D) -> tuple[float, np.ndarray]:
    T = np.array(eri3_tensor(basis, aux, pq=pq))
    b = np.einsum("mnp,mn->p", T, D)
    d = np.linalg.solve(np.array(metric_matrix(aux, pq=pq)), b)
    return 0.5 * float(b @ d), d


def _dfj_gradient(basis, aux, nuclei, D) -> np.ndarray:
    _, d = _dfj(basis, aux, D)
    Gam = D[:, :, None] * d[None, None, :]
    return (np.array(host_df.eri3_gradient(basis, aux, Gam, nuclei, pq=pq))
            + np.array(host_df.metric_gradient(
                aux, -0.5 * np.outer(d, d), nuclei, pq=pq)))


def test_the_df_coulomb_energy_gradient_is_the_finite_difference() -> None:
    basis, aux = _sp(BASIS), _sp(AUX)
    D = _density(basis.n_ao)
    G = _dfj_gradient(basis, aux, NUCLEI, D)
    _check(G, _fd(basis, aux, NUCLEI, lambda b, x: _dfj(b, x, D)[0]),
           "DF-J s, p")


@cache
def _h2o():
    ts = manifest.three_center_set("h2o_sto3g_jkfit_3c")
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return ob.from_fixture(ts.parent), _sp(ob.from_fixture(ts)), nuclei


@pytest.mark.slow
def test_the_df_coulomb_energy_gradient_of_h2o_is_the_finite_difference(
        ) -> None:
    basis, aux, nuclei = _h2o()
    D = _density(basis.n_ao)
    G = _dfj_gradient(basis, aux, nuclei, D)
    _check(G, _fd(basis, aux, nuclei, lambda b, x: _dfj(b, x, D)[0]),
           "DF-J H2O/STO-3G")


def test_the_staging_records_follow_the_staged_order() -> None:
    at, aat = shell_atoms(BASIS, NUCLEI), shell_atoms(AUX, NUCLEI)
    for c in eri3_classes(BASIS, AUX):
        r = host_df.eri3_records(c, BASIS, AUX, NUCLEI)
        assert len(r["ea_bra"]) == len(r["eb_ket"]) == c.n_bra_rec
        assert r["atom_a"] == [at[a] for a, _, _ in c.cells]
        assert r["atom_p"] == [aat[p] for _, _, p in c.cells]
    for c in metric_classes(AUX):
        r = host_df.metric_records(c, AUX, NUCLEI)
        assert r["atom_q"] == [aat[q] for _, q in c.pairs]


def test_a_bad_pair_or_cell_is_refused() -> None:
    W = np.zeros((AUX.n_ao, AUX.n_ao))
    with pytest.raises(ValueError, match="is not i <= j"):
        host_df.metric_gradient(AUX, W, NUCLEI, pq=pq, pairs=[(1, 0)])
    Gam = np.zeros((BASIS.n_ao, BASIS.n_ao, AUX.n_ao))
    with pytest.raises(ValueError, match="is not i <= j"):
        host_df.eri3_gradient(BASIS, AUX, Gam, NUCLEI, pq=pq,
                              cells=[(0, 0, 9)])
