"""`hessian.host_1e`: the contracted nuclear hessians of `S`, `T` and `V`,
against an oracle independent of the second-derivative rule. (`E_nuc`'s
is `hessian.host`'s, test/test_hessian_host.py.)

THE ORACLE is a central finite difference of `gradient.host` -- itself held
to finite differences of the libcint-witnessed `S`, `T`, `V`
(test/test_gradient_host.py) -- in every coordinate of every atom, the
atom's shells AND its nucleus moved together:

    H[:, :, B, l]   ~   (G(R + h e_{B,l}) - G(R - h e_{B,l})) / 2h

with `h = 1e-4` bohr, so the truncation is `O(h^2)`: measured ~3e-9 to
8e-9 of `max|H|` on these inputs, held to `FD_TOL = 1e-7` of `max|H|`.
An order-unity error (a wrong `4 alpha^2`, a dropped `i (i - 1) f_{i-2}`,
a missing centre-nucleus or nucleus-nucleus term) is ~1 of `max|H|`.
`M` is NON-symmetric (the weight rule claims any `M`).

TRANSLATIONAL INVARIANCE, `SUM_A H[A, k, B, l] = 0`, holds only if every
centre's and every nucleus's derivative is right, since none is inferred
from another: held to `INVAR_TOL = 1e-10` of `max|H|`. Not tighter: the
same-atom blocks of a tight core function (O 1s on O, `alpha ~ 130`) are
sums of terms carrying `(2 alpha)^2` that cancel to zero, and their
round-off reaches 4.5e-12 of `max|H|` for `T` on H2O/STO-3G (libcint's own
`d2T` misses invariance by up to 2.6e-12 on H2O/6-31G,
experimental/mmd/hessian/README.md). SYMMETRY, `H[A, k, B, l] =
H[B, l, A, k]`, holds by construction for the blocks written twice and
must hold for the rest (the same-axis `dA dA` blocks): `SYM_TOL = 1e-13`.

The fast tier: H2O/STO-3G and a two-atom s, p basis. `slow`: H2O/cc-pVDZ
(d shells).
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.shell import Basis, Shell
from felizjueves.gradient import host as grad
from felizjueves.hessian import host_1e as host
from felizjueves.reference.pq import one_centre

FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|H|: the O(h^2) truncation, ~8e-9 measured
INVAR_TOL = 1e-10  # of max|H| (module docstring)
SYM_TOL = 1e-13  # of max|H|

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


def _m(n: int) -> np.ndarray:
    return np.random.default_rng(7).standard_normal((n, n))


def _moved(basis, nuclei, atoms, A: int, k: int, h: float):
    sh = tuple(replace(s, centre=tuple(c + h * (d == k)
                                       for d, c in enumerate(s.centre)))
               if atoms[i] == A else s for i, s in enumerate(basis.shells))
    nuc = tuple((Z, tuple(c + h * (d == k) for d, c in enumerate(C)))
                if a == A else (Z, C) for a, (Z, C) in enumerate(nuclei))
    return Basis(sh), nuc


def _gradient(which: str, M):
    if which == "S":
        return lambda b, nu: grad.overlap_gradient(b, M, nu)
    if which == "T":
        return lambda b, nu: grad.kinetic_gradient(b, M, nu)
    return lambda b, nu: grad.potential_gradient(b, M, nu,
                                                 one_centre=one_centre)


def _hessian(which: str, basis, nuclei, M, **kw) -> np.ndarray:
    if which == "S":
        return np.array(host.overlap_hessian(basis, M, nuclei, **kw))
    if which == "T":
        return np.array(host.kinetic_hessian(basis, M, nuclei, **kw))
    return np.array(host.potential_hessian(basis, M, nuclei,
                                           one_centre=one_centre, **kw))


def _fd(basis, nuclei, g) -> np.ndarray:
    atoms = grad.shell_atoms(basis, nuclei)
    n = len(nuclei)
    H = np.zeros((n, 3, n, 3))
    for B in range(n):
        for l in range(3):  # noqa: E741
            p = np.array(g(*_moved(basis, nuclei, atoms, B, l, FD_H)))
            m = np.array(g(*_moved(basis, nuclei, atoms, B, l, -FD_H)))
            H[:, :, B, l] = (p - m) / (2 * FD_H)
    return H


_SLOW = (pytest.mark.slow,)
CASES = [
    *(pytest.param("h2o_sto3g_j", w, id=f"h2o_sto3g-{w}") for w in "STV"),
    *(pytest.param("pair_sp", w, id=f"pair_sp-{w}") for w in "STV"),
    *(pytest.param("h2o_ccpvdz_j", w, id=f"h2o_ccpvdz-{w}", marks=_SLOW)
      for w in "STV"),
]


@pytest.mark.parametrize(("name", "which"), CASES)
def test_the_hessian_is_the_finite_difference_of_the_gradient(
        name, which) -> None:
    basis, nuclei = _case(name)
    M = _m(basis.n_ao)
    H = _hessian(which, basis, nuclei, M)
    scale = np.abs(H).max()
    assert scale > 0
    err = np.abs(H - _fd(basis, nuclei, _gradient(which, M))).max()
    assert err <= FD_TOL * scale, f"{name} {which}: {err / scale:.2e}"


@pytest.mark.parametrize(("name", "which"), CASES)
def test_translational_invariance_and_symmetry(name, which) -> None:
    basis, nuclei = _case(name)
    H = _hessian(which, basis, nuclei, _m(basis.n_ao))
    scale = np.abs(H).max()
    assert np.abs(H.sum(axis=0)).max() <= INVAR_TOL * scale
    assert np.abs(H.sum(axis=2)).max() <= INVAR_TOL * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= SYM_TOL * scale


def test_a_subset_of_pairs_sums_to_the_whole() -> None:
    basis, nuclei = _case("pair_sp")
    M = _m(basis.n_ao)
    n = len(basis.shells)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    whole = _hessian("V", basis, nuclei, M)
    parts = sum(_hessian("V", basis, nuclei, M, pairs=[p]) for p in pairs)
    assert np.abs(parts - whole).max() <= 1e-14 * np.abs(whole).max()


def test_the_absolute_twin_bounds_the_hessian() -> None:
    basis, nuclei = _case("h2o_sto3g_j")
    M = _m(basis.n_ao)
    for which in "STV":
        H = _hessian(which, basis, nuclei, M)
        A = _hessian(which, basis, nuclei, M, absolute=True)
        assert np.all(A >= 0) and np.all(np.abs(H) <= A * (1 + 1e-12))


def test_m_must_be_square_and_pairs_ordered() -> None:
    basis, nuclei = _case("pair_sp")
    with pytest.raises(ValueError, match="matrix"):
        host.overlap_hessian(basis, [[1.0]], nuclei)
    with pytest.raises(ValueError, match="is not i <= j"):
        host.overlap_hessian(basis, _m(basis.n_ao), nuclei, pairs=[(1, 0)])
