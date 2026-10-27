"""`hessian.host`: the Coulomb energy's nuclear Hessian, against an oracle
independent of the twice-applied derivative rule.

THE ORACLE is a central finite difference of `gradient.host.coulomb_gradient`
(itself held to finite differences of the libcint-witnessed
`coulomb_matrix`, test/test_gradient_host.py), in every coordinate of every
atom, the atom's shells and nucleus moved together, `D` fixed:

    H[:, 3A + k]   ~   (G(R + h e_{A,k}) - G(R - h e_{A,k})) / 2h

with `h = 1e-4` bohr, so the truncation is `O(h^2)`: measured ~6e-9 of
`max|H|` on the s, p basis, held to `FD_TOL = 1e-7` of `max|H|`. An
order-unity error (a dropped bra-ket term, a factor 2, a wrong `- i` in the
second derivative) is ~1 of `max|H|`.

THE CHECKS NOTHING INFERS: `H = H^T` (each part is symmetric only summed
over every bra; one bra pair's bra-ket term is not) and `SUM_A H[A k, B l]
= 0` (every centre is differentiated), both to `1e-13` of `max|H|`; and
the two parts add to the whole. The nuclear repulsion's Hessian against
finite differences of its gradient.

The exchange energy's Hessian (`host.exchange_hessian`) is held to the same
checks against `gradient.host.exchange_gradient`, plus its class quartets
summing to the whole and `exchange_hessian_records`' ket segments.

The fast tier: the two-atom s, p basis of test_gradient_host.py. `slow`:
H2O/STO-3G (three atoms, a contracted basis).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_gradient_host import PAIR_SP, _moved

from felizjueves.gradient import host as gh
from felizjueves.hessian import host
from felizjueves.reference.pq import one_centre

FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|H|: the O(h^2) truncation, ~6e-9 measured
EXACT_TOL = 1e-13  # of max|H|: symmetry, invariance, the parts' sum


@cache
def _case(name: str):
    if name == "pair_sp":
        return PAIR_SP
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


@cache
def _density(n: int) -> np.ndarray:
    X = np.random.default_rng(7).standard_normal((n, n))
    return X @ X.T / n


@cache
def _hessian(name: str, parts: str = "both") -> np.ndarray:
    basis, nuclei = _case(name)
    return np.array(host.coulomb_hessian(
        basis, _density(basis.n_ao), nuclei, one_centre=one_centre,
        parts=parts))


CASES = [pytest.param("pair_sp", id="pair_sp"),
         pytest.param("h2o_sto3g_j", id="h2o_sto3g",
                      marks=(pytest.mark.slow,))]


@pytest.mark.parametrize("name", CASES)
def test_the_hessian_is_the_finite_difference_of_the_gradient(name) -> None:
    basis, nuclei = _case(name)
    D = _density(basis.n_ao)
    atoms = gh.shell_atoms(basis, nuclei)
    H = _hessian(name)
    F = np.zeros_like(H)
    for A in range(len(nuclei)):
        for k in range(3):
            g = [np.array(gh.coulomb_gradient(
                *_moved(basis, nuclei, atoms, A, k, h)[:1], D,
                _moved(basis, nuclei, atoms, A, k, h)[1],
                one_centre=one_centre)) for h in (FD_H, -FD_H)]
            F[:, 3 * A + k] = ((g[0] - g[1]) / (2 * FD_H)).ravel()
    scale = np.abs(H).max()
    assert scale > 0.1
    assert np.abs(H - F).max() <= FD_TOL * scale


@pytest.mark.parametrize("name", CASES)
def test_the_hessian_is_symmetric_and_translationally_invariant(name) -> None:
    H = _hessian(name)
    n = H.shape[0] // 3
    scale = np.abs(H).max()
    assert np.abs(H - H.T).max() <= EXACT_TOL * scale
    assert np.abs(H.reshape(n, 3, 3 * n).sum(axis=0)).max() <= (
        EXACT_TOL * scale)


def test_the_two_parts_are_the_whole_and_symmetric_only_when_summed() -> None:
    H, bb, bk = (_hessian("pair_sp", p) for p in ("both", "bb", "bk"))
    scale = np.abs(H).max()
    assert np.abs(bb + bk - H).max() <= EXACT_TOL * scale
    # Summed over every quartet, each part is symmetric by itself (rename
    # (ab) <-> (cd)); over one bra pair the bra-ket part is not.
    assert np.abs(bk).max() > 0.1 * scale
    for part in (bb, bk):
        assert np.abs(part - part.T).max() <= EXACT_TOL * scale
    basis, nuclei = PAIR_SP
    one = np.array(host.coulomb_hessian(
        basis, _density(basis.n_ao), nuclei, one_centre=one_centre,
        pairs=[(0, 3)], parts="bk"))
    assert np.abs(one - one.T).max() > 1e-3 * np.abs(one).max()


def test_the_derivative_kets_carry_the_ket_gradient() -> None:
    # SUM over every derivative ket of atom C, axis l, of D^(C,l)_q at q = 0
    # is SUM_cd D_cd d/dR_{C,l} S_cd with S the (s-weighted) overlap's t = 0
    # entry: the overlap gradient at M = D, D symmetric, with the ket's own
    # weight 2 off the diagonal (gradient.host.overlap_gradient's rule).
    basis, nuclei = PAIR_SP
    D = _density(basis.n_ao)
    dk = host.coulomb_dkets(basis, D, nuclei)
    G = np.array(gh.overlap_gradient(basis, D, nuclei))
    got = np.array([[sum(k.dq[0] for k in dk[C, ax]) for ax in range(3)]
                    for C in range(len(nuclei))])
    assert np.abs(got - G).max() <= 1e-14 * np.abs(G).max()


def test_atoms_give_every_shell_its_own_rows_and_fold_back() -> None:
    basis, nuclei = PAIR_SP
    nat, nsh = len(nuclei), len(basis.shells)
    real = gh.shell_atoms(basis, nuclei)
    rows = tuple(nat + i for i in range(nsh))
    Hs = np.array(host.coulomb_hessian(basis, _density(basis.n_ao), nuclei,
                                       one_centre=one_centre, atoms=rows))
    fold = np.zeros((3 * nat, 3 * (nat + nsh)))
    for i, r in enumerate(rows):
        fold[3 * real[i]:3 * real[i] + 3, 3 * r:3 * r + 3] = np.eye(3)
    H = _hessian("pair_sp")
    assert np.abs(fold @ Hs @ fold.T - H).max() <= (
        EXACT_TOL * np.abs(H).max())


def test_the_nuclear_repulsion_hessian_is_its_finite_difference() -> None:
    nuclei = ((8.0, (0.0, 0.0, 0.1)), (1.0, (0.1, -1.4, 1.1)),
              (1.0, (-0.1, 1.5, 0.9)))
    H = np.array(host.nuclear_repulsion_hessian(nuclei))
    F = np.zeros_like(H)
    for A in range(3):
        for k in range(3):
            g = []
            for h in (FD_H, -FD_H):
                nu = tuple((Z, tuple(c + h * (a == A and d == k)
                                     for d, c in enumerate(C)))
                           for a, (Z, C) in enumerate(nuclei))
                g.append(np.array(gh.nuclear_repulsion_gradient(nu)))
            F[:, 3 * A + k] = ((g[0] - g[1]) / (2 * FD_H)).ravel()
    assert np.abs(H - F).max() <= FD_TOL * np.abs(H).max()
    assert np.abs(H - H.T).max() <= EXACT_TOL * np.abs(H).max()


def test_a_bad_parts_and_bad_atoms_are_refused() -> None:
    basis, nuclei = PAIR_SP
    D = _density(basis.n_ao)
    with pytest.raises(ValueError, match="parts"):
        host.coulomb_hessian(basis, D, nuclei, one_centre=one_centre,
                             parts="kk")
    with pytest.raises(ValueError, match="atoms="):
        host.coulomb_hessian(basis, D, nuclei, one_centre=one_centre,
                             atoms=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="not symmetric"):
        host.coulomb_hessian(basis, D + np.triu(D, 1), nuclei,
                             one_centre=one_centre)


# --- the exchange energy's Hessian --------------------------------------------
#
# The same oracle on `E_K = 1/2 SUM D_ab D_cd (ac|bd)`: central differences
# of `gradient.host.exchange_gradient` (held to finite differences of the
# libcint-witnessed `exchange_matrix`), measured ~6e-9 of `max|H|` on the s,
# p basis. Here the two parts are each ~10x `max|H|` and cancel, so a
# dropped or mis-signed term is far above `FD_TOL`.


@cache
def _k_hessian(name: str, parts: str = "both") -> np.ndarray:
    basis, nuclei = _case(name)
    return np.array(host.exchange_hessian(
        basis, _density(basis.n_ao), nuclei, one_centre=one_centre,
        parts=parts))


@pytest.mark.parametrize("name", CASES)
def test_the_exchange_hessian_is_the_finite_difference_of_its_gradient(
        name) -> None:
    basis, nuclei = _case(name)
    D = _density(basis.n_ao)
    atoms = gh.shell_atoms(basis, nuclei)
    H = _k_hessian(name)
    F = np.zeros_like(H)
    for A in range(len(nuclei)):
        for k in range(3):
            g = []
            for h in (FD_H, -FD_H):
                b, nu = _moved(basis, nuclei, atoms, A, k, h)
                g.append(np.array(gh.exchange_gradient(
                    b, D, nu, one_centre=one_centre)))
            F[:, 3 * A + k] = ((g[0] - g[1]) / (2 * FD_H)).ravel()
    scale = np.abs(H).max()
    assert scale > 0.1
    assert np.abs(H - F).max() <= FD_TOL * scale


@pytest.mark.parametrize("name", CASES)
def test_the_exchange_hessian_is_symmetric_and_invariant(name) -> None:
    H = _k_hessian(name)
    n = H.shape[0] // 3
    scale = np.abs(H).max()
    assert np.abs(H - H.T).max() <= EXACT_TOL * scale
    assert np.abs(H.reshape(n, 3, 3 * n).sum(axis=0)).max() <= (
        EXACT_TOL * scale)


def test_the_exchange_parts_are_the_whole_and_each_symmetric() -> None:
    H, bb, bk = (_k_hessian("pair_sp", p) for p in ("both", "bb", "bk"))
    scale = np.abs(H).max()
    assert np.abs(bb + bk - H).max() <= EXACT_TOL * scale
    assert np.abs(bk).max() > scale  # the parts cancel: neither is H
    for part in (bb, bk):
        assert np.abs(part - part.T).max() <= EXACT_TOL * scale


def test_the_exchange_quartets_sum_to_the_whole_and_atoms_fold_back() -> None:
    basis, nuclei = PAIR_SP
    D = _density(basis.n_ao)
    H = _k_hessian("pair_sp")
    r = range(2)
    S = sum(np.array(host.exchange_hessian(
        basis, D, nuclei, one_centre=one_centre,
        quartets=gh.class_quartets(basis, am)))
        for am in ((a, b, c, d) for a in r for b in r for c in r for d in r))
    assert np.abs(S - H).max() <= EXACT_TOL * np.abs(H).max()
    nat, nsh = len(nuclei), len(basis.shells)
    real = gh.shell_atoms(basis, nuclei)
    rows = tuple(nat + i for i in range(nsh))
    Hs = np.array(host.exchange_hessian(basis, D, nuclei,
                                        one_centre=one_centre, atoms=rows))
    fold = np.zeros((3 * nat, 3 * (nat + nsh)))
    for i, row in enumerate(rows):
        fold[3 * real[i]:3 * real[i] + 3, 3 * row:3 * row + 3] = np.eye(3)
    assert np.abs(fold @ Hs @ fold.T - H).max() <= (
        EXACT_TOL * np.abs(H).max())


def test_the_exchange_records_group_the_kets_by_atom_pair() -> None:
    basis, nuclei = PAIR_SP
    am = (1, 0, 1, 1)
    bra, ket, seg = host.exchange_hessian_records(basis, nuclei, am)
    gbra, gket = gh.exchange_records(basis, nuclei, am)
    assert bra == gbra
    n = len(gket["s_bd"])
    start = seg["seg_start"]
    assert start[0] == 0 and start[-1] == n
    assert len(start) == len(seg["seg_b"]) + 1 == len(seg["seg_d"]) + 1
    assert all(a < b for a, b in zip(start, start[1:], strict=False))
    # the same records, each once, stably regrouped
    key = list(zip(gket["Q"], gket["off_b"], gket["off_d"], strict=True))
    got = list(zip(ket["Q"], ket["off_b"], ket["off_d"], strict=True))
    assert sorted(got) == sorted(key)
    atoms = gh.shell_atoms(basis, nuclei)
    off_atom = {basis.offsets[i]: atoms[i] for i in range(len(basis.shells))}
    for s, (b, d) in enumerate(zip(seg["seg_b"], seg["seg_d"], strict=True)):
        for r in range(start[s], start[s + 1]):
            assert (off_atom[ket["off_b"][r]], off_atom[ket["off_d"][r]]) == (
                b, d)
    assert len(set(zip(seg["seg_b"], seg["seg_d"], strict=True))) == len(
        seg["seg_b"])


def test_the_exchange_hessian_refuses_what_the_coulomb_one_does() -> None:
    basis, nuclei = PAIR_SP
    D = _density(basis.n_ao)
    with pytest.raises(ValueError, match="parts"):
        host.exchange_hessian(basis, D, nuclei, one_centre=one_centre,
                              parts="kk")
    with pytest.raises(ValueError, match="not symmetric"):
        host.exchange_hessian(basis, D + np.triu(D, 1), nuclei,
                              one_centre=one_centre)
    with pytest.raises(ValueError, match="out of range"):
        host.exchange_hessian(basis, D, nuclei, one_centre=one_centre,
                              quartets=[(0, 0, 0, 9)])
