"""`gradient.host`: the contracted nuclear gradients of `S`, `T`, `V` and
`E_J`, against an oracle independent of the derivative rule.

THE ORACLE is a central finite difference of this repo's own integrals --
`one_electron.overlap_matrix`, `kinetic_matrix`, `potential_matrix` and
`eri.j_coulomb.j.coulomb_matrix`, each witnessed against libcint through
the tracked fixtures -- in every coordinate of every atom, the atom's
shells AND its nucleus moved together:

    d/dR_{A,k} SUM_mn M_mn X_mn(R)   ~   (f(R + h e) - f(R - h e)) / 2h

with `h = 1e-4` bohr, so the truncation is `O(h^2)`: measured ~2e-9 to
6e-9 of `max|G|` on these inputs, held to `FD_TOL = 1e-7` of `max|G|`. An
order-unity error (a wrong `2 alpha`, a dropped `- i E^{i-1}`, a missing
Hellmann-Feynman term, a factor 2 on `E_J`) is ~1 of `max|G|`. `M` is a
NON-symmetric matrix for `S`, `T` and `V` (the weight rule claims any `M`)
and a symmetric positive definite one for `E_J` (a density).

TRANSLATIONAL INVARIANCE, `SUM_A G[A] = 0`, holds only if the derivative
on every centre (and every nucleus, for `V`) is right, since none is
inferred from another: held to `INVAR_TOL = 1e-13` of `max|G|`.

The fast tier: H2O/STO-3G for `S`, `T`, `V`; a two-atom s, p basis for
`E_J` (one primitive per shell, so `coulomb_matrix` is cheap). `slow`:
H2O/cc-pVDZ (d shells) and H2O/STO-3G's `E_J`.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.j_coulomb.j import coulomb_matrix
from felizjueves.gradient import host
from felizjueves.one_electron.kinetic import kinetic_matrix
from felizjueves.one_electron.overlap import overlap_matrix
from felizjueves.one_electron.potential import potential_matrix
from felizjueves.reference.pq import one_centre, pq

FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|G|: the O(h^2) truncation, ~5e-9 measured
INVAR_TOL = 1e-13  # of max|G|

# Two atoms, an s and a p shell each, one primitive per shell: every
# one-electron and J class with l <= 1, off-diagonal and diagonal pairs.
_A, _B = (0.1, -0.2, 0.3), (1.2, 0.4, -0.5)
PAIR_SP = (Basis((Shell(0, _A, (1.3,), (0.9,)), Shell(1, _A, (0.8,), (1.1,)),
                  Shell(0, _B, (0.6,), (0.7,)), Shell(1, _B, (1.7,), (0.5,)))),
           ((3.0, _A), (1.0, _B)))


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


def _case(name: str):
    return PAIR_SP if name == "pair_sp" else _fixture(name)


def _matrices(n: int):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, n))
    return X, X @ X.T / n  # non-symmetric M; a density D


def _moved(basis, nuclei, atoms, A: int, k: int, h: float):
    sh = tuple(replace(s, centre=tuple(c + h * (d == k)
                                       for d, c in enumerate(s.centre)))
               if atoms[i] == A else s for i, s in enumerate(basis.shells))
    nuc = tuple((Z, tuple(c + h * (d == k) for d, c in enumerate(C)))
                if a == A else (Z, C) for a, (Z, C) in enumerate(nuclei))
    return Basis(sh), nuc


def _fd(basis, nuclei, f) -> np.ndarray:
    atoms = host.shell_atoms(basis, nuclei)
    G = np.zeros((len(nuclei), 3))
    for A in range(len(nuclei)):
        for k in range(3):
            p = f(*_moved(basis, nuclei, atoms, A, k, FD_H))
            m = f(*_moved(basis, nuclei, atoms, A, k, -FD_H))
            G[A, k] = (p - m) / (2 * FD_H)
    return G


def _energy(which: str, M, D):
    if which == "S":
        return lambda b, nu: float(np.sum(M * np.array(overlap_matrix(b))))
    if which == "T":
        return lambda b, nu: float(np.sum(M * np.array(kinetic_matrix(b))))
    if which == "V":
        return lambda b, nu: float(np.sum(M * np.array(
            potential_matrix(b, nu, one_centre=one_centre))))
    return lambda b, nu: 0.5 * float(np.sum(D * np.array(
        coulomb_matrix(b, D, pq=pq))))


def _analytic(which: str, basis, nuclei, M, D) -> np.ndarray:
    if which == "S":
        return np.array(host.overlap_gradient(basis, M, nuclei))
    if which == "T":
        return np.array(host.kinetic_gradient(basis, M, nuclei))
    if which == "V":
        return np.array(host.potential_gradient(basis, M, nuclei,
                                                one_centre=one_centre))
    return np.array(host.coulomb_gradient(basis, D, nuclei,
                                          one_centre=one_centre))


_SLOW = (pytest.mark.slow,)
CASES = [
    *(pytest.param("h2o_sto3g_j", w, id=f"h2o_sto3g-{w}") for w in "STV"),
    pytest.param("pair_sp", "J", id="pair_sp-J"),
    pytest.param("pair_sp", "V", id="pair_sp-V"),
    pytest.param("h2o_sto3g_j", "J", id="h2o_sto3g-J", marks=_SLOW),
    *(pytest.param("h2o_ccpvdz_j", w, id=f"h2o_ccpvdz-{w}", marks=_SLOW)
      for w in "STV"),
]


@pytest.mark.parametrize(("name", "which"), CASES)
def test_the_gradient_is_the_finite_difference_of_the_integrals(
        name: str, which: str) -> None:
    basis, nuclei = _case(name)
    M, D = _matrices(basis.n_ao)
    g = _analytic(which, basis, nuclei, M, D)
    fd = _fd(basis, nuclei, _energy(which, M, D))
    scale = np.abs(g).max()
    assert scale > 0.0
    assert np.abs(g - fd).max() <= FD_TOL * scale, (
        f"{name} {which}: |G - FD| / max|G| = "
        f"{np.abs(g - fd).max() / scale:.2e}")


@pytest.mark.parametrize(("name", "which"), [
    *(pytest.param("h2o_sto3g_j", w, id=f"h2o_sto3g-{w}") for w in "STVJ"),
    *(pytest.param("pair_sp", w, id=f"pair_sp-{w}") for w in "STVJ"),
])
def test_the_gradient_is_translationally_invariant(name: str,
                                                   which: str) -> None:
    basis, nuclei = _case(name)
    M, D = _matrices(basis.n_ao)
    g = _analytic(which, basis, nuclei, M, D)
    assert np.abs(g.sum(axis=0)).max() <= INVAR_TOL * np.abs(g).max()


def test_a_bare_nucleus_is_moved_by_the_hellmann_feynman_term_alone() -> None:
    # A third nucleus carrying no basis function: only the operator term
    # d/dC [t](P - C) reaches its row, so the row is non-zero, matches the
    # finite difference, and the gradient still sums to zero.
    basis, nuclei = PAIR_SP
    nuclei = (*nuclei, (2.0, (-0.7, 0.9, 0.4)))
    M, _ = _matrices(basis.n_ao)
    g = np.array(host.potential_gradient(basis, M, nuclei,
                                         one_centre=one_centre))
    fd = _fd(basis, nuclei, _energy("V", M, None))
    assert np.abs(g[2]).max() > 1e-3 * np.abs(g).max()
    assert np.abs(g - fd).max() <= FD_TOL * np.abs(g).max()
    assert np.abs(g.sum(axis=0)).max() <= INVAR_TOL * np.abs(g).max()
    ghost = tuple((0.0, c) for _, c in nuclei)
    assert np.all(np.array(host.potential_gradient(
        basis, M, ghost, one_centre=one_centre)) == 0.0)


def test_the_weights_are_exact_for_any_m() -> None:
    # SUM_mn M_mn X_mn over the unique pairs, by the weight rule, equals the
    # full double sum for a symmetric X and a non-symmetric M.
    basis, _ = PAIR_SP
    M, _ = _matrices(basis.n_ao)
    S = np.array(overlap_matrix(basis))
    off = basis.offsets
    tot = 0.0
    for i, j in host._unique_pairs(basis):
        w = host.pair_weights(M, i, j, basis)
        blk = S[off[i]:off[i] + basis.shells[i].n_cart,
                off[j]:off[j] + basis.shells[j].n_cart].ravel()
        tot += float(np.dot(w, blk))
    assert tot == pytest.approx(float(np.sum(M * S)), rel=1e-14)


def test_a_shell_on_no_nucleus_is_refused() -> None:
    basis, nuclei = PAIR_SP
    with pytest.raises(ValueError, match="sits on no nucleus"):
        host.shell_atoms(basis, nuclei[:1])
    assert host.shell_atoms(basis, nuclei) == (0, 0, 1, 1)


def test_the_nuclear_repulsion_gradient_is_its_finite_difference() -> None:
    _, nuclei = _fixture("h2o_sto3g_j")

    def e(nu):
        out = 0.0
        for a, (Za, A) in enumerate(nu):
            for Zb, B in nu[a + 1:]:
                out += Za * Zb / float(np.linalg.norm(np.subtract(A, B)))
        return out

    g = np.array(host.nuclear_repulsion_gradient(nuclei))
    fd = np.zeros_like(g)
    for A in range(len(nuclei)):
        for k in range(3):
            def at(h, A=A, k=k):
                return e([(Z, tuple(c + h * (d == k) for d, c in enumerate(C)))
                          if a == A else (Z, C)
                          for a, (Z, C) in enumerate(nuclei)])
            fd[A, k] = (at(FD_H) - at(-FD_H)) / (2 * FD_H)
    assert np.abs(g - fd).max() <= FD_TOL * np.abs(g).max()
    assert np.abs(g.sum(axis=0)).max() <= INVAR_TOL * np.abs(g).max()
