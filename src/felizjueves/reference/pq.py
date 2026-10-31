"""The McMurchie-Davidson Hermite integrals [r]^(m) and [p|q], run directly.

    [r + 1d]^(m) = PQ_d [r]^(m+1) + r_d [r - 1d]^(m+1)   ([JEngine, v2, eqs. 10-12])
    [p|q]        = (-1)^|q| [p + q]^(0)                  ([JEngine, v2, eqs. 7-9])

`one_centre` is a memoised recursion over (r, m) and nothing else: no MILD, no
DAG, no sieve, no schedule. Each node is reduced along its first nonzero
component, x before y before z -- a fixed rule, deliberately not the one the
plan layer chooses -- and the fundamentals come from `reference.boys`, which is
quadrature rather than the downward recurrence a kernel uses. The IR path
chooses directions by MILD, prunes with the sieve and runs a schedule; this
path does none of it, so the two agreeing means something.

THE THETA CONVENTION LIVES IN `fundamental`, AND NOWHERE ELSE IN THIS PACKAGE.
`theta` is the argument every function here takes; `theta_of` is how it is
formed from the pair exponents. Both are pinned there, with the citation.

float64 throughout. Read by tests and by the interpreter's pins; nothing that
emits kernel text may import this module.
"""

from __future__ import annotations

import math
from functools import cache, lru_cache

from felizjueves.reference.boys import boys

R = tuple[int, int, int]
Vec = tuple[float, float, float]

# THE CONVENTION, stated once. Pinned in docs/implementation/j_engine.md,
# "Two defects in the source" (issue #8), by our derivation against McMurchie &
# Davidson (doi:10.1016/0021-9991(78)90092-X, eqs. 3.30-3.32, 4.2-4.8);
# [JEngine, v2, eq. 13] never defines theta and prints the m-dependence wrong:
#
#     theta^2  = zeta eta / (zeta + eta),   zeta = alpha + beta,
#                                           eta  = gamma + delta
#     [0]^(m)  = (2 / sqrt(pi)) theta (-2 theta^2)^m F_m(theta^2 |PQ|^2)
#
# (-2 theta^2)^m, NOT the printed (-2 theta)^m: only the former makes
# d/dPQ_x [0]^(m) = PQ_x [0]^(m+1), which is what the recurrence assumes.
# The pair factors (pi/zeta)^(3/2) and (pi/eta)^(3/2) are NOT here; they belong
# to the bra and ket pair data with the E coefficients.
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


def theta_of(zeta: float, eta: float) -> float:
    """theta = sqrt(zeta eta / (zeta + eta)); the convention is pinned above."""
    return math.sqrt(zeta * eta / (zeta + eta))


def fundamental(m: int, pq: Vec, theta: float) -> float:
    """[0]^(m) = (2/sqrt(pi)) theta (-2 theta^2)^m F_m(theta^2 |PQ|^2)."""
    t2 = theta * theta
    T = t2 * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2])
    return _TWO_OVER_SQRT_PI * theta * (-2.0 * t2) ** m * boys(m, T)


def _check_r(r: R) -> None:
    if len(r) != 3 or any(type(c) is not int or c < 0 for c in r):
        raise ValueError(f"r = {r!r} is not a Cartesian index of three ints >= 0")


def one_centre(r: R, m: int, pq: Vec, theta: float) -> float:
    """[r](m) by the raising recurrence of [JEngine, v2, eqs. 10-12], directly.

    A memoised recursion over (r, m) and nothing else: no MILD, no DAG, no
    sieve, no schedule. That is the whole point -- the IR path chooses
    directions by MILD, prunes with the sieve and runs a schedule, and this
    path does none of it, so the two agreeing means something.
    """
    _check_r(r)
    if type(m) is not int or m < 0:
        raise ValueError(f"m = {m!r} is not an int >= 0")
    pq = (float(pq[0]), float(pq[1]), float(pq[2]))
    return _nodes(pq, theta)(tuple(r), m)


def _nodes(pq: Vec, theta: float):
    """The memoised `node(r, m)` of one `(PQ, theta)`, kept across calls.

    A caller walks every `(p, q)` of a block at one `(PQ, theta)`, and each
    walk shares its whole tree with the others; a memo made afresh per call
    recomputed it, Boys quadratures included, for every element (most of a
    fast-tier reference's time). The recursion and its arithmetic are the
    ones a fresh memo runs, so a value is the same float either way. Keyed
    by `float.hex`, which tells -0.0 from 0.0 where `==` would not.
    """
    return _nodes_at(tuple(x.hex() for x in pq), float(theta).hex(), pq, theta)


@lru_cache(maxsize=64)
def _nodes_at(key_pq: tuple[str, ...], key_theta: str, pq: Vec, theta: float):
    del key_pq, key_theta  # the key; pq and theta ride along unhashed-equal

    @cache
    def node(r: R, m: int) -> float:
        for d in range(3):
            if r[d]:
                # [r]^(m) = PQ_d [r - 1d]^(m+1) + (r_d - 1) [r - 2d]^(m+1)
                down = r[:d] + (r[d] - 1,) + r[d + 1 :]
                value = pq[d] * node(down, m + 1)
                if r[d] >= 2:
                    down2 = r[:d] + (r[d] - 2,) + r[d + 1 :]
                    value += (r[d] - 1) * node(down2, m + 1)
                return value
        return fundamental(m, pq, theta)

    return node


def pq(p: R, q: R, pq: Vec, theta: float) -> float:
    """[p|q] = (-1)^|q| [p+q](0), with component-wise vector addition."""
    _check_r(p)
    _check_r(q)
    r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
    sign = -1.0 if sum(q) % 2 else 1.0
    return sign * one_centre(r, 0, pq, theta)
