"""`hessian.host_df`: the contracted nuclear Hessian of the DF metric
`(P|Q)`, against an oracle independent of the second-derivative rule.

THE ORACLE is a central finite difference of `gradient.host_df.
metric_gradient` -- itself held to finite differences of the
libcint-witnessed `metric_matrix` (test/test_gradient_host_df.py) -- in
every coordinate of every atom, the atom's aux shells AND its nucleus moved
together:

    H[:, :, B, l]   ~   (G(R + h e_{B,l}) - G(R - h e_{B,l})) / 2h

with `h = 1e-4` bohr, so the truncation is `O(h^2)`: measured ~2.5e-9 of
`max|H|` on these inputs, held to `FD_TOL = 1e-7` of `max|H|`. An
order-unity error (a wrong `4 alpha^2`, a dropped `i (i - 1) f_{i-2}`, a
mixed block on the wrong atom) is ~1 of `max|H|`. `W` is NON-symmetric
(the weight rule claims any `W`).

TRANSLATIONAL INVARIANCE, `SUM_A H[A, k, B, l] = 0`, holds only if both
centres' derivatives are right, since neither is inferred from the other:
`INVAR_TOL = 1e-12` of `max|H|`. SYMMETRY, `H[A, k, B, l] = H[B, l, A,
k]`, holds by construction for the mixed blocks and must hold for the
same-centre ones: `SYM_TOL = 1e-13`.

L for the tier cut is `lP + lQ + 2`, the Hermite order the Hessian reaches
(test/tiers.py): the fast tier is an s, p aux basis on three centres (at
most `(p|p)`, L = 4); `slow` adds a d basis (L = 6) and one with f and g
shells (L = 10).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from felizjueves.ao.shell import Basis, Shell
from felizjueves.gradient import host_df as grad
from felizjueves.gradient.host import shell_atoms
from felizjueves.hessian import host_df as host
from felizjueves.reference.pq import pq

FD_H = 1e-4  # bohr
FD_TOL = 1e-7  # of max|H|: the O(h^2) truncation, ~2.5e-9 measured
INVAR_TOL = 1e-12  # of max|H|
SYM_TOL = 1e-13  # of max|H|

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
NUCLEI = ((8.0, A), (1.0, B), (1.0, C))
# s, p (and d) on three centres, contracted and not, two p shells on one
# atom (a same-atom off-diagonal pair).
SPD = Basis((
    Shell(0, C, (2.0, 0.4), (0.7, 0.5)),
    Shell(1, A, (0.85,), (1.0,)),
    Shell(2, B, (1.4, 0.35), (0.6, 0.45)),
    Shell(1, B, (1.1, 0.5), (0.6, 0.5)),
    Shell(0, A, (5.0,), (1.0,)),
    Shell(1, B, (0.7,), (1.0,)),
))
SP = Basis(tuple(sh for sh in SPD.shells if sh.l <= 1))
FG = Basis((
    Shell(0, C, (0.9,), (1.0,)),
    Shell(3, B, (0.7,), (1.0,)),
    Shell(4, A, (1.2,), (1.0,)),
    Shell(1, C, (1.1, 0.5), (0.6, 0.5)),
))
CASES = [pytest.param(SP, id="sp"),
         pytest.param(SPD, id="spd", marks=pytest.mark.slow),
         pytest.param(FG, id="fg", marks=pytest.mark.slow)]


def _w(n: int) -> np.ndarray:
    return np.random.default_rng(7).standard_normal((n, n))


def _moved(aux, nuclei, atoms, at: int, k: int, h: float):
    sh = tuple(replace(s, centre=tuple(c + h * (d == k)
                                       for d, c in enumerate(s.centre)))
               if atoms[i] == at else s for i, s in enumerate(aux.shells))
    nuc = tuple((Z, tuple(c + h * (d == k) for d, c in enumerate(R)))
                if a == at else (Z, R) for a, (Z, R) in enumerate(nuclei))
    return Basis(sh), nuc


def _fd(aux, nuclei, W) -> np.ndarray:
    atoms = shell_atoms(aux, nuclei)
    n = len(nuclei)
    H = np.zeros((n, 3, n, 3))
    for at in range(n):
        for l in range(3):  # noqa: E741
            g = []
            for s in (1, -1):
                b, nu = _moved(aux, nuclei, atoms, at, l, s * FD_H)
                g.append(np.array(grad.metric_gradient(b, W, nu, pq=pq)))
            H[:, :, at, l] = (g[0] - g[1]) / (2 * FD_H)
    return H


def _hessian(aux, W, **kw) -> np.ndarray:
    return np.array(host.metric_hessian(aux, W, NUCLEI, pq=pq, **kw))


@pytest.mark.parametrize("aux", CASES)
def test_the_hessian_is_the_finite_difference_of_the_gradient(aux) -> None:
    W = _w(aux.n_ao)
    H = _hessian(aux, W)
    scale = np.abs(H).max()
    assert scale > 0
    err = np.abs(H - _fd(aux, NUCLEI, W)).max()
    assert err <= FD_TOL * scale, f"{err / scale:.2e}"


@pytest.mark.parametrize("aux", CASES)
def test_translational_invariance_and_symmetry(aux) -> None:
    H = _hessian(aux, _w(aux.n_ao))
    scale = np.abs(H).max()
    assert np.abs(H.sum(axis=0)).max() <= INVAR_TOL * scale
    assert np.abs(H.sum(axis=2)).max() <= INVAR_TOL * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= SYM_TOL * scale


def test_a_subset_of_pairs_sums_to_the_whole() -> None:
    W = _w(SP.n_ao)
    n = len(SP.shells)
    whole = _hessian(SP, W)
    parts = sum(_hessian(SP, W, pairs=[(i, j)])
                for i in range(n) for j in range(i, n))
    assert np.abs(parts - whole).max() <= 1e-14 * np.abs(whole).max()


def test_only_the_symmetric_part_of_w_counts() -> None:
    W = _w(SP.n_ao)
    H = _hessian(SP, W)
    Hs = _hessian(SP, 0.5 * (W + W.T))
    assert np.abs(H - Hs).max() <= 1e-13 * np.abs(H).max()


def test_the_absolute_twin_bounds_the_hessian() -> None:
    W = _w(SP.n_ao)
    H = _hessian(SP, W)
    Ab = _hessian(SP, W, absolute=True)
    assert np.all(Ab >= 0) and np.all(np.abs(H) <= Ab * (1 + 1e-12))


def test_split_atoms_give_every_shell_its_own_rows() -> None:
    W = _w(SP.n_ao)
    split = tuple(len(NUCLEI) + i for i in range(len(SP.shells)))
    H = _hessian(SP, W, atoms=split)
    real = shell_atoms(SP, NUCLEI)
    folded = np.zeros((len(NUCLEI), 3, len(NUCLEI), 3))
    for i, a in enumerate(split):
        for j, b in enumerate(split):
            folded[real[i], :, real[j], :] += H[a, :, b, :]
    whole = _hessian(SP, W)
    assert np.abs(folded - whole).max() <= 1e-13 * np.abs(whole).max()


def test_pairs_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="is not i <= j"):
        host.metric_hessian(SP, _w(SP.n_ao), NUCLEI, pq=pq, pairs=[(1, 0)])
    with pytest.raises(ValueError, match="one non-negative row per shell"):
        host.metric_hessian(SP, _w(SP.n_ao), NUCLEI, pq=pq, atoms=(0,))
