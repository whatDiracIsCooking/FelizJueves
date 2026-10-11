"""`ao.pairs`: the dressed pair data, and a Schwarz bound PROVEN to bound.

The inequality the kernel's screen rests on (`Ip * (Iq * Dmax) < tau`,
ao/pairs.py's docstring has the derivation) is checked here numerically over
randomised geometry, not taken on authority. The `[p|q]` values come from
`reference.pq` -- passed in to `ao.pairs` as the injected evaluator, never
imported by it.

THE SWEEP. Exponents log-uniform over four orders of magnitude, [1e-2, 1e2]
per primitive, because the hazard (`theta = zeta` in place of
`sqrt(zeta/2)`) holds on compact pairs and fails on diffuse ones; centres
uniform in a 4-bohr box; Hermite classes `lp, lq <= 3`. A draw whose Boys
argument `theta^2 |PQ|^2` exceeds `reference.boys.T_MAX` is redrawn: the
reference refuses it rather than return an unmeasured number.

NO SLACK BEYOND 1e-14 RELATIVE. Every inequality is asserted as
`lhs <= rhs * (1 + 1e-14)`. The bound is attained exactly at `PQ = 0`,
`zeta = eta`, `p = q`, so that case is in the test too: it is the one where
rounding, not the inequality, decides, and the margin covers a few ulps of
`theta_of` against `sqrt(zeta/2)` and of the square root.
"""

from __future__ import annotations

import ast
import inspect
import math
import random

import numpy as np
import pytest

from felizjueves.ao import hermite, pairs
from felizjueves.ao.hermite import e_block, hermite_order
from felizjueves.ao.pairs import (
    dmax,
    dress,
    hermite_bound,
    hermite_bounds,
    pair_bound,
    primitive_pairs,
)
from felizjueves.ao.shell import Shell, cartesians, primitive_norm
from felizjueves.reference.boys import T_MAX
from felizjueves.reference.pq import fundamental, one_centre, pq, theta_of

REL = 1e-14
LMAX = 3
N_CASES = 60


def _exp(rng: random.Random) -> float:
    return 10.0 ** rng.uniform(-2.0, 2.0)


def _vec(rng: random.Random, half: float = 2.0) -> tuple[float, float, float]:
    return (rng.uniform(-half, half), rng.uniform(-half, half),
            rng.uniform(-half, half))


def _case(rng: random.Random):
    """(zeta, P, eta, Q): two primitive pairs whose Boys argument is in range."""
    while True:
        a, b, c, d = (_exp(rng) for _ in range(4))
        A, B, C, D = (_vec(rng) for _ in range(4))
        zeta, eta = a + b, c + d
        P = tuple((a * A[k] + b * B[k]) / zeta for k in range(3))
        Q = tuple((c * C[k] + d * D[k]) / eta for k in range(3))
        PQ = tuple(P[k] - Q[k] for k in range(3))
        if theta_of(zeta, eta) ** 2 * sum(x * x for x in PQ) <= T_MAX:
            return zeta, P, eta, Q, PQ


def _sweep():
    rng = random.Random(83)
    return [_case(rng) for _ in range(N_CASES)]


SWEEP = _sweep()


def _pq_table(lp: int, lq: int, PQ, theta: float) -> np.ndarray:
    return np.array([[pq(p, q, PQ, theta) for q in hermite_order(lq)]
                     for p in hermite_order(lp)])


def test_the_sweep_spans_four_orders_of_magnitude() -> None:
    zs = [z for z, _, e, _, _ in SWEEP for z in (z, e)]
    assert min(zs) < 0.1 and max(zs) > 20.0, (min(zs), max(zs))


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------


def test_bounds_are_non_negative() -> None:
    # Over every zeta in the sweep; hermite_bound raises (never abs-es) on a
    # negative radicand, so reaching the assert means the sign came out right.
    for zeta, _, eta, _, _ in SWEEP:
        for z in (zeta, eta):
            for p in hermite_order(2 * LMAX):
                assert hermite_bound(z, p, one_centre=one_centre) >= 0.0


def test_the_bound_bounds_elementwise() -> None:
    for zeta, _, eta, _, PQ in SWEEP:
        theta = theta_of(zeta, eta)
        Ip = np.array(hermite_bounds(zeta, LMAX, one_centre=one_centre))
        Iq = np.array(hermite_bounds(eta, LMAX, one_centre=one_centre))
        G = _pq_table(LMAX, LMAX, PQ, theta)
        bound = np.outer(Ip, Iq)
        worst = np.max(np.abs(G) / bound)
        assert worst <= 1.0 + REL, (zeta, eta, PQ, worst)


@pytest.mark.parametrize("lp,lq", [(lp, lq) for lp in range(LMAX + 1)
                                   for lq in range(LMAX + 1)])
def test_the_bound_bounds_a_contraction(lp: int, lq: int) -> None:
    rng = np.random.default_rng(1000 + 10 * lp + lq)
    nq = len(hermite_order(lq))
    for zeta, _, eta, _, PQ in SWEEP:
        theta = theta_of(zeta, eta)
        G = _pq_table(lp, lq, PQ, theta)
        Ip = np.array(hermite_bounds(zeta, lp, one_centre=one_centre))
        Ipair = pair_bound(eta, lq, one_centre=one_centre)
        D = rng.uniform(-1.0, 1.0, size=nq) * 10.0 ** rng.uniform(-3, 3)
        J = G @ D
        rhs = Ip * Ipair * dmax(D)
        assert np.all(np.abs(J) <= rhs * (1.0 + REL)), (zeta, eta, PQ)


def test_the_bound_is_attained_at_the_self_integral() -> None:
    # PQ = 0, zeta = eta, p = q: Cauchy-Schwarz with equality. If the bound
    # had slack built in, this would show it; it must be tight to rounding.
    for zeta, *_ in SWEEP:
        theta = theta_of(zeta, zeta)
        for p in hermite_order(LMAX):
            Ip = hermite_bound(zeta, p, one_centre=one_centre)
            v = pq(p, p, (0.0, 0.0, 0.0), theta)
            assert v > 0.0
            assert abs(v / (Ip * Ip) - 1.0) <= REL, (zeta, p)


def test_the_sweep_catches_theta_equal_zeta() -> None:
    # The hazard: theta = zeta instead of sqrt(zeta/2). It is a valid bound
    # for compact pairs (zeta > 1/2) and a wrong one for diffuse pairs, so a
    # sweep that did not span the exponents would pass it. This one must not.
    def wrong(z: float, p) -> float:
        r = tuple(2 * c for c in p)
        v = one_centre(r, 0, (0.0, 0.0, 0.0), z)
        return math.sqrt(-v if sum(p) % 2 else v)

    violated = 0
    for zeta, _, eta, _, PQ in SWEEP:
        G = _pq_table(LMAX, LMAX, PQ, theta_of(zeta, eta))
        Ip = np.array([wrong(zeta, p) for p in hermite_order(LMAX)])
        Iq = np.array([wrong(eta, q) for q in hermite_order(LMAX)])
        violated += int(np.any(np.abs(G) > np.outer(Ip, Iq) * (1.0 + REL)))
    assert violated > 0


def test_a_wrong_sign_is_raised_not_hidden() -> None:
    def flipped(r, m, pq_, theta):
        return -one_centre(r, m, pq_, theta)

    with pytest.raises(ValueError, match="sign convention"):
        hermite_bound(1.0, (1, 0, 0), one_centre=flipped)


def test_the_evaluator_is_what_is_called() -> None:
    seen = []

    def spy(r, m, pq_, theta):
        seen.append((r, m, pq_, theta))
        return one_centre(r, m, pq_, theta)

    hermite_bound(0.8, (1, 2, 0), one_centre=spy)
    assert seen == [((2, 4, 0), 0, (0.0, 0.0, 0.0), math.sqrt(0.4))]


def test_pair_bound_and_dmax() -> None:
    ib = hermite_bounds(1.3, 2, one_centre=one_centre)
    assert len(ib) == len(hermite_order(2))
    assert pair_bound(1.3, 2, one_centre=one_centre) == pytest.approx(
        sum(ib), rel=1e-15)
    assert pair_bound(1.3, 2, one_centre=one_centre) >= max(ib)
    assert dmax((0.5, -3.0, 2.0)) == 3.0
    with pytest.raises(ValueError):
        dmax(())


# ---------------------------------------------------------------------------
# The dressed ket: Dmax over D_q, not over the AO density
# ---------------------------------------------------------------------------


def _shell(rng: random.Random, l: int, n: int) -> Shell:  # noqa: E741
    return Shell(l, _vec(rng), tuple(_exp(rng) for _ in range(n)),
                 tuple(rng.uniform(-1.0, 1.0) for _ in range(n)))


@pytest.mark.parametrize("la,lb,lc,ld", [(0, 0, 0, 0), (1, 0, 1, 1),
                                         (1, 1, 2, 0), (2, 1, 1, 2),
                                         (1, 2, 3, 0)])
def test_the_bound_bounds_with_dressed_kets(la, lb, lc, ld) -> None:
    rng = random.Random(8300 + la * 1000 + lb * 100 + lc * 10 + ld)
    nrng = np.random.default_rng(8300 + la * 1000 + lb * 100 + lc * 10 + ld)
    for _ in range(4):
        sa, sb, sc, sd = (_shell(rng, am, 2) for am in (la, lb, lc, ld))
        dens = nrng.uniform(-1.0, 1.0, size=(sc.n_cart, sd.n_cart)).ravel()
        for bra in primitive_pairs(sa, sb):
            Ip = hermite_bounds(bra.zeta, bra.L, one_centre=one_centre)
            Ibra = pair_bound(bra.zeta, bra.L, one_centre=one_centre)
            for ket in primitive_pairs(sc, sd):
                PQ = tuple(bra.P[k] - ket.P[k] for k in range(3))
                theta = theta_of(bra.zeta, ket.zeta)
                if theta * theta * sum(x * x for x in PQ) > T_MAX:
                    continue
                Dq = np.array(ket.block).T @ dens   # dressed D_q
                G = _pq_table(bra.L, ket.L, PQ, theta)
                J = G @ Dq
                Iket = pair_bound(ket.zeta, ket.L, one_centre=one_centre)
                rhs = np.array(Ip) * Iket * dmax(Dq)
                assert np.all(np.abs(J) <= rhs * (1.0 + REL))
                # the bra record's scalar bounds every J_p it accumulates
                assert np.all(np.abs(J) <= Ibra * Iket * dmax(Dq) * (1 + REL))


# ---------------------------------------------------------------------------
# dress, and the placement of the pair factor
# ---------------------------------------------------------------------------


def test_dress_applies_exactly_the_two_factors() -> None:
    # c_a c_b (pi/zeta)^{3/2}, and no normalisation: that is in the
    # coefficients (ao/shell.py's convention; #85 removed a second one here).
    e = (1.0, -2.0, 0.25)
    got = dress(e, ca=0.3, cb=-1.7, zeta=1.9)
    want = [0.3 * -1.7 * (math.pi / 1.9) ** 1.5 * x for x in e]
    assert got == pytest.approx(want, rel=1e-15)
    with pytest.raises(ValueError):
        dress(e, ca=1.0, cb=1.0, zeta=0.0)
    with pytest.raises(TypeError):  # no normalisation parameter, dead or live
        dress(e, ca=1.0, cb=1.0, na=2.0, nb=0.5, zeta=1.9)


def test_the_placement_is_complete() -> None:
    # (2/sqrt(pi)) theta (pi/zeta)^{3/2} (pi/eta)^{3/2} = MD's lambda,
    # 2 pi^{5/2} / (zeta eta sqrt(zeta + eta)). The kernel's half comes from
    # the reference's [0]^(0) at T = 0 (F_0(0) = 1); the host's half is what
    # `dress` multiplies in with every other factor at 1.
    rng = random.Random(5)
    for _ in range(200):
        zeta, eta = _exp(rng), _exp(rng)
        kernel = fundamental(0, (0.0, 0.0, 0.0), theta_of(zeta, eta))
        host_bra = dress((1.0,), ca=1.0, cb=1.0, zeta=zeta)[0]
        host_ket = dress((1.0,), ca=1.0, cb=1.0, zeta=eta)[0]
        lam = 2.0 * math.pi ** 2.5 / (zeta * eta * math.sqrt(zeta + eta))
        assert kernel * host_bra * host_ket == pytest.approx(lam, rel=1e-14)


def _pi_users(module) -> set[str]:
    """The top-level functions of `module` whose body mentions `pi`."""
    tree = ast.parse(inspect.getsource(module))
    users = set()
    for node in tree.body:
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute) and sub.attr == "pi") or (
                    isinstance(sub, ast.Name) and sub.id == "pi"):
                users.add(getattr(node, "name", "<module>"))
    return users


def test_dress_is_the_only_place_a_pair_factor_appears() -> None:
    assert _pi_users(pairs) == {"dress"}
    assert _pi_users(hermite) == set()


# ---------------------------------------------------------------------------
# The pair data, against an independent overlap
# ---------------------------------------------------------------------------


def _overlap_1d(i: int, j: int, alpha: float, beta: float, A: float,
                B: float) -> float:
    """int x_A^i x_B^j exp(-alpha x_A^2 - beta x_B^2) dx by Gauss-Hermite,
    exact for this polynomial degree -- no ancestry in the E recurrence."""
    zeta = alpha + beta
    P = (alpha * A + beta * B) / zeta
    x, w = np.polynomial.hermite.hermgauss(i + j + 2)
    s = x / math.sqrt(zeta) + P
    pref = math.exp(-alpha * beta / zeta * (A - B) ** 2) / math.sqrt(zeta)
    return pref * float(np.sum(w * (s - A) ** i * (s - B) ** j))


def test_the_dressed_t0_column_is_the_dressed_overlap() -> None:
    # int Lambda_t = 0 for t != 0 and (pi/zeta)^{3/2} for t = 0, so the
    # dressed t = 0 coefficient is c_a c_b S_ab exactly, S_ab the overlap of
    # the UNNORMALISED primitives (by quadrature below) -- the coefficients
    # carry any normalisation, so nothing else may appear (#85).
    rng = random.Random(831)
    for la in range(3):
        for lb in range(3):
            sa, sb = _shell(rng, la, 2), _shell(rng, lb, 3)
            pp = primitive_pairs(sa, sb)
            assert [(r.i, r.j) for r in pp] == [(i, j) for i in range(2)
                                                for j in range(3)]
            for rec in pp:
                alpha, beta = sa.exponents[rec.i], sb.exponents[rec.j]
                assert rec.zeta == alpha + beta and rec.L == la + lb
                coef = sa.coefficients[rec.i] * sb.coefficients[rec.j]
                rows = [(a, b) for a in cartesians(la) for b in cartesians(lb)]
                assert len(rec.block) == len(rows)
                for (a, b), row in zip(rows, rec.block, strict=True):
                    assert len(row) == len(hermite_order(la + lb))
                    S = math.prod(_overlap_1d(a[d], b[d], alpha, beta,
                                              sa.centre[d], sb.centre[d])
                                  for d in range(3))
                    assert row[0] == pytest.approx(coef * S, rel=1e-12,
                                                   abs=1e-14 * abs(coef))


def test_an_axis_primitive_has_unit_self_overlap() -> None:
    for l in range(5):  # noqa: E741
        for alpha in (0.01, 1.0, 100.0):
            # The normalisation is the coefficient's to carry (#85).
            s = Shell(l, (0.3, -0.2, 1.1), (alpha,),
                      (primitive_norm(l, alpha),))
            (rec,) = primitive_pairs(s, s)
            assert rec.block[0][0] == pytest.approx(1.0, rel=1e-14)


def test_the_block_is_the_dressed_pure_block() -> None:
    # Wiring, not physics: the record is e_block dressed once, in the order
    # the docstring states. The physics is the overlap test above.
    rng = random.Random(832)
    sa, sb = _shell(rng, 2, 1), _shell(rng, 1, 1)
    (rec,) = primitive_pairs(sa, sb)
    k = 0
    for a in cartesians(2):
        for b in cartesians(1):
            pure = e_block(a, b, rec.zeta, rec.P, sa.centre, sb.centre)
            assert rec.block[k] == dress(
                pure, ca=sa.coefficients[0], cb=sb.coefficients[0],
                zeta=rec.zeta)
            k += 1
