"""reference/pq.py: the direct-recurrence [r]^(m) and [p|q].

The one check here that reaches outside the recurrence is the direct
quadrature: [r]^(m) evaluated from its defining integral,

    [r]^(m) = (2/sqrt(pi)) theta (-2 theta^2)^m
              d^r/dPQ^r  integral_0^1 t^(2m) exp(-t^2 theta^2 |PQ|^2) dt,

with the PQ-derivatives taken under the integral sign in closed form -- the
derivatives of a Gaussian are Hermite polynomials,
d^n/dx^n exp(-a x^2) = (-sqrt a)^n H_n(sqrt a x) exp(-a x^2) -- and the t
integral done by composite Gauss-Legendre, a different rule from boys.py's.
No F_m value and no step of the MD recurrence is used on that side.

The theta convention under test is j_engine.md's pin ("Two defects in the
source", issue #8), not [JEngine, v2, eq. 13] as printed:
`test_fundamental_differentiates_down_in_m` is the check that tells the two
apart.
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from felizjueves.reference.pq import fundamental, one_centre, pq, theta_of

# (pq, theta) cases; T = theta^2 |PQ|^2 stays well inside boys' domain.
CASES = [
    ((0.3, -0.7, 1.1), 0.9),
    ((1.4, 0.2, -0.5), 1.7),
    ((-0.05, 0.6, 0.25), 0.4),
    ((0.0, 0.0, 0.0), 1.3),
    ((2.0, -1.5, 0.8), 1.1),
]


def _indices(max_total: int) -> list[tuple[int, int, int]]:
    return [
        (x, y, z)
        for x, y, z in itertools.product(range(max_total + 1), repeat=3)
        if x + y + z <= max_total
    ]


def _composite_gauss_legendre(panels: int, order: int) -> tuple[np.ndarray, ...]:
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, 1.0, panels + 1)
    a, b = edges[:-1, None], edges[1:, None]
    t = (0.5 * (b - a) * (x + 1.0) + a).ravel()
    wt = (0.5 * (b - a) * w).ravel()
    return t, wt


RULE = _composite_gauss_legendre(panels=40, order=20)


def direct(r, m, pqv, theta, rule=RULE) -> float:
    """[r]^(m) from the defining integral; no F_m, no MD recurrence."""
    t_nodes, t_weights = rule
    s = t_nodes * theta  # sqrt(a) with a = t^2 theta^2
    integrand = t_nodes ** (2 * m)
    for n, x in zip(r, pqv, strict=True):
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0
        hermite = np.polynomial.hermite.hermval(s * x, coeffs)  # physicists' H_n
        integrand = integrand * (-s) ** n * hermite * np.exp(-(s * x) ** 2)
    prefactor = 2.0 / math.sqrt(math.pi) * theta * (-2.0 * theta * theta) ** m
    return prefactor * float(np.dot(t_weights, integrand))


def test_direct_quadrature_is_converged() -> None:
    """The oracle itself: halving the panel count changes nothing material."""
    coarse_rule = _composite_gauss_legendre(panels=20, order=20)
    grid = [(r, m, p, th) for r in _indices(3) for m in (0, 2) for p, th in CASES]
    fine = [direct(*args) for args in grid]
    coarse = [direct(*args, rule=coarse_rule) for args in grid]
    np.testing.assert_allclose(fine, coarse, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize(("pqv", "theta"), CASES)
def test_one_centre_matches_the_defining_integral(pqv, theta) -> None:
    for r in _indices(4):
        for m in range(0, 3):
            expected = direct(r, m, pqv, theta)
            got = one_centre(r, m, pqv, theta)
            # Cancellation between Hermite terms can make a value small next
            # to its siblings, so the absolute floor scales with [0]^(m).
            scale = abs(fundamental(m, (0.0, 0.0, 0.0), theta))
            assert got == pytest.approx(expected, rel=1e-11, abs=1e-12 * scale), (
                r, m, pqv, theta)


def test_fundamental_differentiates_down_in_m() -> None:
    """d/dPQ_x [0]^(m) = PQ_x [0]^(m+1): the pinned (-2 theta^2)^m.

    This is the r = 0 step of the recurrence, and it holds only with the
    m-dependence j_engine.md pins; the printed (-2 theta)^m of [JEngine, v2,
    eq. 13] breaks it whenever theta != 1, which the second half shows.
    """
    h = 1e-5
    for (x, y, z), theta in CASES:
        for m in range(0, 4):
            fd = (fundamental(m, (x + h, y, z), theta)
                  - fundamental(m, (x - h, y, z), theta)) / (2 * h)
            assert fd == pytest.approx(
                x * fundamental(m + 1, (x, y, z), theta), rel=1e-7, abs=1e-9)

    def printed(m: int, pqv: tuple[float, float, float], theta: float) -> float:
        # (-2 theta)^m in place of (-2 theta^2)^m, everything else the same.
        return fundamental(m, pqv, theta) * (-2 * theta) ** m / (
            -2 * theta * theta) ** m

    (x, y, z), theta = CASES[1]  # theta = 1.7
    fd = (printed(0, (x + h, y, z), theta) - printed(0, (x - h, y, z), theta)) / (
        2 * h)
    assert fd != pytest.approx(x * printed(1, (x, y, z), theta), rel=1e-3)


def test_theta_is_the_reduced_exponent() -> None:
    for zeta, eta in ((1.0, 1.0), (0.3, 7.0), (12.5, 0.04)):
        theta = theta_of(zeta, eta)
        assert theta * theta == pytest.approx(zeta * eta / (zeta + eta), rel=1e-15)


@pytest.mark.parametrize(("pqv", "theta"), CASES[:3])
def test_recurrence_holds_along_every_direction(pqv, theta) -> None:
    """one_centre reduces x-first; the value must not depend on that choice."""
    for r in _indices(6):
        for m in range(0, 2):
            for d in range(3):
                if not r[d]:
                    continue
                down = list(r)
                down[d] -= 1
                value = pqv[d] * one_centre(tuple(down), m + 1, pqv, theta)
                if r[d] >= 2:
                    down[d] -= 1
                    value += (r[d] - 1) * one_centre(tuple(down), m + 1, pqv,
                                                     theta)
                assert one_centre(r, m, pqv, theta) == pytest.approx(
                    value, rel=1e-12, abs=1e-14), (r, m, d)


def test_one_centre_has_the_parity_of_its_derivative_order() -> None:
    """[r](-PQ) = (-1)^|r| [r](PQ): derivatives of an even function of PQ."""
    for pqv, theta in CASES:
        neg = tuple(-c for c in pqv)
        for r in _indices(5):
            assert one_centre(r, 0, neg, theta) == pytest.approx(
                (-1) ** sum(r) * one_centre(r, 0, pqv, theta), rel=1e-13,
                abs=1e-300)


def test_pq_is_the_signed_sum_index_over_random_p_q() -> None:
    rng = random.Random(20)
    for _ in range(200):
        p = tuple(rng.randint(0, 4) for _ in range(3))
        q = tuple(rng.randint(0, 4) for _ in range(3))
        pqv = tuple(rng.uniform(-1.5, 1.5) for _ in range(3))
        theta = rng.uniform(0.2, 2.0)
        r = tuple(a + b for a, b in zip(p, q, strict=True))
        assert pq(p, q, pqv, theta) == (-1) ** sum(q) * one_centre(r, 0, pqv, theta)
        # Swapping bra and ket and inverting PQ is the identity:
        # [q|p](-PQ) = (-1)^|p| (-1)^(|p|+|q|) [p+q](PQ) = [p|q](PQ).
        neg = tuple(-c for c in pqv)
        assert pq(q, p, neg, theta) == pytest.approx(
            pq(p, q, pqv, theta), rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("bad", [(1, 0), (1, -1, 0), (1.0, 0, 0), (0, 0, 0, 0)])
def test_refuses_a_malformed_index(bad) -> None:
    with pytest.raises(ValueError):
        one_centre(bad, 0, (0.1, 0.2, 0.3), 1.0)
