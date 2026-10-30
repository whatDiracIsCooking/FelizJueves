"""`hessian.host_kx`: the exchange derivative matrix `K^X` (#689) on the
host, held to PySCF.

- THE ip1 HALF, `K1[k, a, b] = SUM_cd D_cd (dA_k a c|b d)`, against the
  gradient fixture's own `vk1` (`grad.rhf.get_jk`'s `vk`, libcint's
  `int2e_ip1` contracted `ijkl,jk->il`, the sign taken: manifest's
  docstring), H2O/STO-3G, H2O/STO-3G plus a d on O, H2O/cc-pVDZ.
- THE WHOLE `K^X` per atom, `P_X K1 + (P_X K1)^T + KC_X + KC_X^T`,
  against the tracked witness of PySCF's `hessian.rhf.make_h1` exchange
  part (`manifest.kx_witness`: it records `K^X`; `make_h1`'s is `-1/2
  K^X`), the same three inputs.
- `SUM_X K^X = 0`: on the host (a small s, p case) and on the witness.

The host is pure Python at seconds to minutes a molecule, so every
comparison against a fixture is `slow` by `tiers.by_l` on the highest
raised class it integrates (`4 lmax + 1`, test_rhf_gradient.py's rule
for a K term); the witness's own checks need no host and are fast.
`ABS_TOL = 1e-10` absolute, test_rhf_gradient.py's: measured at most
8.3e-14 (`K1`, cc-pVDZ) and 3.6e-14 (`K^X`, the d case).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_gradient_host import FD_H, PAIR_SP, _moved
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.k_exchange.k import exchange_matrix
from felizjueves.gradient import host as ghost
from felizjueves.hessian import host_kx
from felizjueves.reference.pq import one_centre, pq

ABS_TOL = 1e-10
FD_REL = 1e-7  # the central difference's O(h^2) floor, of max|K^X|


@cache
def _inputs(parent: str):
    fx = manifest.gradient_fixture(parent)
    nuclei = tuple((float(x.charge), x.coord) for x in fx.atoms)
    return fx, ob.from_fixture(fx), nuclei


@cache
def _k1(parent: str) -> np.ndarray:
    fx, basis, nuclei = _inputs(parent)
    return np.array(host_kx.exchange_derivative_ip1(
        basis, fx.density, nuclei, one_centre=one_centre))


def _l(parent: str) -> int:
    return 4 * max(s.l for s in _inputs(parent)[1].shells) + 1


PARENTS = [by_l(p, _l(p), id=p) for p in manifest.KX_PARENTS]


@pytest.mark.parametrize("parent", PARENTS)
def test_the_ip1_half_is_pyscfs_vk(parent: str) -> None:
    fx = _inputs(parent)[0]
    err = float(np.abs(_k1(parent) - fx.arrays["vk1"]).max())
    assert err <= ABS_TOL, f"{parent}: {err:.2e} from PySCF's vk1"


@pytest.mark.parametrize("parent", PARENTS)
def test_the_whole_k_x_is_pyscfs_make_h1_exchange(parent: str) -> None:
    fx, basis, nuclei = _inputs(parent)
    want = manifest.kx_witness(parent).kx
    k1 = _k1(parent).tolist()
    for atom in range(len(nuclei)):
        got = np.array(host_kx.exchange_derivative(
            basis, fx.density, nuclei, atom, one_centre=one_centre, k1=k1))
        err = float(np.abs(got - want[atom]).max())
        assert err <= ABS_TOL, f"{parent} atom {atom}: {err:.2e}"


@pytest.mark.parametrize("parent", manifest.KX_PARENTS)
def test_the_witness_loads_is_symmetric_and_sums_to_zero(parent) -> None:
    w = manifest.kx_witness(parent)
    assert w.kx.shape == (len(w.parent.atoms), 3, w.parent.nao,
                          w.parent.nao)
    assert np.array_equal(w.kx, w.kx.transpose(0, 1, 3, 2))
    assert np.abs(w.kx.sum(axis=0)).max() <= 1e-11 * np.abs(w.kx).max()
    assert np.abs(w.kx).max() > 0.1


# --- on the host alone -----------------------------------------------------------

_A, _B = (0.0, 0.1, -0.2), (1.4, -0.3, 0.5)
S_ONLY = (Basis((Shell(0, _A, (1.1, 0.4), (0.7, 0.5)),
                 Shell(0, _B, (0.8,), (1.2,)), Shell(0, _B, (0.3,), (0.9,)))),
          ((2.0, _A), (1.0, _B)))


def _density(basis: Basis) -> np.ndarray:
    X = np.random.default_rng(7).standard_normal((basis.n_ao, basis.n_ao))
    return X @ X.T / basis.n_ao


def _sum_over_atoms(case) -> tuple[np.ndarray, float]:
    basis, nuclei = case
    D = _density(basis)
    k1 = host_kx.exchange_derivative_ip1(basis, D, nuclei,
                                         one_centre=one_centre)
    total = sum(np.array(host_kx.exchange_derivative(
        basis, D, nuclei, X, one_centre=one_centre, k1=k1))
        for X in range(len(nuclei)))
    return total, float(np.abs(k1).max())


def test_k_x_sums_to_zero_over_the_atoms_s_only() -> None:
    total, scale = _sum_over_atoms(S_ONLY)
    assert np.abs(total).max() <= 1e-13 * scale


@pytest.mark.slow
def test_k_x_sums_to_zero_over_the_atoms_s_p() -> None:
    total, scale = _sum_over_atoms(PAIR_SP)
    assert np.abs(total).max() <= 1e-13 * scale


@pytest.mark.parametrize("case", [
    S_ONLY, pytest.param(PAIR_SP, marks=pytest.mark.slow)],
    ids=["s_only", "pair_sp"])
def test_k_x_is_the_finite_difference_of_k(case) -> None:
    # independent of PySCF: d K / d R_{X,k} by central differences of the
    # libcint-witnessed exchange_matrix, moving atom X's shells
    basis, nuclei = case
    D = _density(basis)
    atoms = ghost.shell_atoms(basis, nuclei)
    k1 = host_kx.exchange_derivative_ip1(basis, D, nuclei,
                                         one_centre=one_centre)
    for X in range(len(nuclei)):
        got = np.array(host_kx.exchange_derivative(
            basis, D, nuclei, X, one_centre=one_centre, k1=k1))
        for k in range(3):
            p, m = (np.array(exchange_matrix(
                _moved(basis, nuclei, atoms, X, k, h)[0], D, pq=pq))
                for h in (FD_H, -FD_H))
            fd = (p - m) / (2 * FD_H)
            err = float(np.abs(got[k] - fd).max())
            assert err <= FD_REL * float(np.abs(got).max()), (X, k, err)


def test_an_atom_out_of_range_is_refused() -> None:
    basis, nuclei = S_ONLY
    with pytest.raises(ValueError, match="out of range"):
        host_kx.exchange_derivative_contracted(
            basis, _density(basis), nuclei, 2, one_centre=one_centre)


def test_the_records_of_one_atom_are_the_bra_pairs_whose_c_sits_on_it() -> None:
    basis, nuclei = PAIR_SP
    am = (1, 0, 0, 1)
    every, ket = host_kx.kx_records(basis, nuclei, am)
    assert "atom_a" not in every and "atom_c" not in every
    full, _ = ghost.exchange_records(basis, nuclei, am)
    n = 0
    for X in range(len(nuclei)):
        bra, ket_x = host_kx.kx_records(basis, nuclei, am, atom=X)
        assert ket_x == ket
        keep = [i for i, a in enumerate(full["atom_c"]) if a == X]
        assert bra["off_c"] == [full["off_c"][i] for i in keep]
        n += len(bra["s_ac"])
    assert n == len(every["s_ac"])
