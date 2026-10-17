"""`ao.hermite`: the `E` coefficients, checked by an identity, not a table.

`[JEngine, v2]` does not state the recurrence (McMurchie & Davidson do, entry
46 of docs/references.md, eqs. 2.20-2.22), and nobody publishes a table of `E`
values to score against. So the check is the expansion `E` exists to satisfy,

    x_A^i x_B^j exp(-alpha x_A^2) exp(-beta x_B^2) = sum_t E^{ij}_t Lambda_t(x)

with `Lambda_t(x) = zeta^{t/2} H_t(sqrt(zeta)(x-P)) exp(-zeta(x-P)^2)` evaluated
by `numpy.polynomial.hermite.hermval` -- a closed form with no ancestry in the
two-term recurrence under test.

TOLERANCES ARE STATED AT A SCALE, never tuned. A sum of signed terms is
accurate relative to the sum of their MAGNITUDES, not to its own value (which
may be small by cancellation), so every comparison below is

    |lhs - rhs| <= K * eps * scale,   scale = sum of |each term|

with `K = 64`: the recurrence takes at most i+j = 8 raising steps of 3 terms
each, `hermval` a degree <= 8 Clenshaw sum, and the product side 3 roundings,
so each term carries O(30) roundings at worst; 64 covers that with margin and
is still ~1e-14 relative -- far below any wrong coefficient, which is O(1).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from numpy.polynomial import hermite as H
from tiers import split_by_l

from felizjueves.ao.hermite import e_axis, e_block, hermite_order
from felizjueves.plan.targets import hermite_targets, n_targets

EPS = np.finfo(np.float64).eps
K = 64
IJ = [(i, j) for i in range(5) for j in range(5)]
IJ_PARAMS = split_by_l(IJ, sum, ids=lambda c: f"{c[0]}-{c[1]}")  # i+j > 4 slow


def _pair(rng: random.Random) -> tuple[float, float, float, float]:
    return (rng.uniform(0.1, 5.0), rng.uniform(0.1, 5.0),
            rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))


def _derived(alpha: float, beta: float, A: float, B: float):
    zeta = alpha + beta
    P = (alpha * A + beta * B) / zeta
    mu = alpha * beta / zeta  # NOT alpha beta / zeta^2, NOT zeta / 2
    return zeta, P, mu


def _lambda(t: int, zeta: float, P: float, x: np.ndarray) -> np.ndarray:
    c = np.zeros(t + 1)
    c[t] = 1.0
    y = math.sqrt(zeta) * (x - P)
    return zeta ** (t / 2) * H.hermval(y, c) * np.exp(-y * y)


@pytest.mark.parametrize("ij", IJ_PARAMS)
def test_expansion_identity_pointwise(ij) -> None:
    i, j = ij
    rng = random.Random(1000 * i + j)
    for _ in range(20):
        alpha, beta, A, B = _pair(rng)
        zeta, P, _ = _derived(alpha, beta, A, B)
        E = e_axis(i, j, zeta, P - A, P - B)
        x = P + np.array([rng.uniform(-2.5, 2.5) for _ in range(9)]) / math.sqrt(zeta)
        lhs = (x - A) ** i * (x - B) ** j * np.exp(-alpha * (x - A) ** 2) \
            * np.exp(-beta * (x - B) ** 2)
        terms = np.array([E[t] * _lambda(t, zeta, P, x) for t in range(len(E))])
        rhs = terms.sum(axis=0)
        scale = np.abs(terms).sum(axis=0) + np.abs(lhs)
        assert np.all(np.abs(lhs - rhs) <= K * EPS * scale), (alpha, beta, A, B)


@pytest.mark.parametrize("ij", IJ_PARAMS)
def test_moments_by_gauss_hermite(ij) -> None:
    i, j = ij
    # int x^n Omega dx == sum_t E_t int x^n Lambda_t dx for n = 0..i+j. With
    # x = P + y/sqrt(zeta) every integrand is (polynomial of degree <= n+i+j)
    # times exp(-y^2), which an N-point rule integrates exactly for
    # 2N - 1 >= 2(i+j).
    rng = random.Random(7 + 1000 * i + j)
    y, w = H.hermgauss(i + j + 2)
    for _ in range(10):
        alpha, beta, A, B = _pair(rng)
        zeta, P, mu = _derived(alpha, beta, A, B)
        E = e_axis(i, j, zeta, P - A, P - B)
        x = P + y / math.sqrt(zeta)
        omega = (x - A) ** i * (x - B) ** j * math.exp(-mu * (A - B) ** 2)
        lam = [zeta ** (t / 2) * H.hermval(y, np.eye(t + 1)[t]) for t in range(len(E))]
        for n in range(i + j + 1):
            lhs_terms = w * x ** n * omega
            rhs_terms = np.array([E[t] * w * x ** n * lam[t] for t in range(len(E))])
            scale = np.abs(lhs_terms).sum() + np.abs(rhs_terms).sum()
            assert abs(lhs_terms.sum() - rhs_terms.sum()) <= K * EPS * scale


def test_e00_is_the_gaussian_product_prefactor() -> None:
    # Computed here as exp(-mu X_AB^2) with mu = alpha beta / zeta, from the
    # exponents; the module forms it as exp(zeta X_PA X_PB). Equal in exact
    # arithmetic, so the difference is a few roundings of the exponent:
    # |d exp(u)| <= exp(u) * |du|, |du| <= K eps (1 + |u|).
    rng = random.Random(3)
    for _ in range(200):
        alpha, beta, A, B = _pair(rng)
        zeta, P, mu = _derived(alpha, beta, A, B)
        (e00,) = e_axis(0, 0, zeta, P - A, P - B)
        u = -mu * (A - B) ** 2
        want = math.exp(u)
        assert abs(e00 - want) <= K * EPS * (1 + abs(u)) * want
    assert e_axis(0, 0, 1.3, 0.0, 0.0) == (1.0,)


@pytest.mark.parametrize("ij", IJ_PARAMS)
def test_nothing_above_i_plus_j(ij) -> None:
    i, j = ij
    # E^{ij}_t == 0 for t > i+j is enforced by the table's LENGTH; the top entry
    # is exactly (2 zeta)^{-(i+j)} E^{00}_0 -- only the 1/2zeta term reaches it
    # -- so an off-by-one at the top of the table shows here at every i+j >= 1.
    zeta, pa, pb = 1.7, 0.31, -0.42
    E = e_axis(i, j, zeta, pa, pb)
    assert len(E) == i + j + 1
    (e00,) = e_axis(0, 0, zeta, pa, pb)
    top = e00 / (2 * zeta) ** (i + j)
    assert abs(E[-1] - top) <= K * EPS * top


@pytest.mark.parametrize("ij", IJ_PARAMS)
def test_same_centre_odd_parity_vanishes(ij) -> None:
    i, j = ij
    # A == B puts P on both centres: X_PA = X_PB = 0, and x^{i+j} has parity
    # i+j, so only t with i+j-t even survive -- exactly, not to a tolerance.
    E = e_axis(i, j, 2.3, 0.0, 0.0)
    for t, v in enumerate(E):
        if (i + j - t) % 2:
            assert v == 0.0
        else:
            assert v > 0.0


@pytest.mark.parametrize("ij", IJ_PARAMS)
def test_not_symmetric_unless_the_centres_swap(ij) -> None:
    i, j = ij
    # Swapped with the centres, equal up to rounding (the two raise in a
    # different order); swapped alone, different by O(1).
    zeta, pa, pb = 1.1, 0.37, -0.53
    E = np.array(e_axis(i, j, zeta, pa, pb))
    scale = np.abs(E).sum()
    assert np.all(np.abs(E - e_axis(j, i, zeta, pb, pa)) <= K * EPS * scale)
    if i != j:
        assert np.abs(E - e_axis(j, i, zeta, pa, pb)).max() > 1e-3 * scale


def test_hermite_order_is_the_plan_layers() -> None:
    for L in range(13):
        assert hermite_order(L) == hermite_targets(L)


def test_e_block_is_the_product_of_its_axes() -> None:
    # i+j differs per axis (3, 1, 0), so a transposed axis changes which
    # entries are zero and fails here.
    a, b = (2, 1, 0), (1, 0, 0)
    zeta, p, A, B = 1.9, (0.1, -0.2, 0.3), (0.5, 0.4, -0.7), (-0.3, 0.2, 0.9)
    L = sum(a) + sum(b)
    blk = e_block(a, b, zeta, p, A, B)
    assert len(blk) == n_targets(L)
    ax = [e_axis(a[d], b[d], zeta, p[d] - A[d], p[d] - B[d]) for d in range(3)]
    assert [len(e) for e in ax] == [4, 2, 1]
    for t, v in zip(hermite_targets(L), blk, strict=True):
        if all(t[d] < len(ax[d]) for d in range(3)):
            assert v == ax[0][t[0]] * ax[1][t[1]] * ax[2][t[2]]
        else:
            assert v == 0.0


def test_bad_input_refused() -> None:
    for bad in [(-1, 0, 1.0), (0, 1.5, 1.0), (0, 0, 0.0), (0, 0, -1.0)]:
        with pytest.raises(ValueError):
            e_axis(*bad, 0.1, 0.2)
