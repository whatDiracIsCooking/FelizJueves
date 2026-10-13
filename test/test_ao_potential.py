"""`ao.potential`: `V` as the point-charge limit of the J engine's `[p|q]`.

The `[t]^(0)` evaluator (`reference.pq.one_centre`) and the Boys function
(`reference.boys`) arrive by injection or are used only by the test's own
closed forms, so this file shares no ancestry with the recurrence it checks.
The libcint comparison is a later issue's (milestone 11, `O1`). Fast tier,
no marker, no PySCF.
"""

from __future__ import annotations

import math
import random

import pytest

from felizjueves.ao.hermite import hermite_order
from felizjueves.ao.pairs import dress, primitive_pairs
from felizjueves.ao.potential import potential_block, potential_matrix
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import fundamental, one_centre, theta_of

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
SHELLS = (
    Shell(0, A, (3.4, 0.62, 0.17), (0.15, 0.53, 0.44)),
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(2, B, (0.8,), (1.0,)),
    Shell(0, B, (1.3, 0.21), (0.6, 0.47)),
    Shell(3, C, (0.55, 0.12), (0.8, 0.3)),
    Shell(1, C, (0.9,), (1.0,)),
)
NUCLEI = ((8.0, A), (1.0, B), (1.0, (0.5, 0.5, -0.4)))


def _ss(alpha, Aa, beta, Bb, Cc) -> float:
    """`<s_A|1/r_C|s_B> = (2 pi / zeta) K_AB F_0(zeta |P - C|^2)`, unnormalised
    primitives, `F_0` from `reference.boys`."""
    zeta, mu = alpha + beta, alpha * beta / (alpha + beta)
    ab2 = sum((x - y) ** 2 for x, y in zip(Aa, Bb, strict=True))
    P = [(alpha * x + beta * y) / zeta for x, y in zip(Aa, Bb, strict=True)]
    pc2 = sum((p - c) ** 2 for p, c in zip(P, Cc, strict=True))
    return 2 * math.pi / zeta * math.exp(-mu * ab2) * boys(0, zeta * pc2)


def test_the_point_charge_constant_is_the_one_electron_prefactor() -> None:
    # (2/sqrt(pi)) sqrt(zeta) (pi/zeta)^{3/2} = 2 pi / zeta: the kernel's half
    # is the reference's [0]^(0) at T = 0 and theta = sqrt(zeta); the host's
    # half is what `dress` multiplies in with every other factor at 1. Nothing
    # else forms a constant, so this is the placement being complete.
    rng = random.Random(235)
    for _ in range(200):
        zeta = math.exp(rng.uniform(math.log(1e-2), math.log(1e3)))
        kernel = fundamental(0, (0.0, 0.0, 0.0), math.sqrt(zeta))
        host = dress((1.0,), ca=1.0, cb=1.0, zeta=zeta)[0]
        assert kernel * host == pytest.approx(2 * math.pi / zeta, rel=1e-14)
        # ...and sqrt(zeta) IS the eta -> inf limit of the J convention
        assert theta_of(zeta, 1e12 * zeta) == pytest.approx(math.sqrt(zeta),
                                                            rel=1e-11)


@pytest.mark.parametrize(("alpha", "beta", "Bb", "Cc"), [
    (0.5, 1.3, (0.0, 0.0, 1.4), (0.2, -0.3, 0.5)),
    (2.0, 0.07, (0.9, -1.1, 0.3), (-1.0, 2.0, 0.0)),
    (11.0, 3.0, (-0.2, 0.5, 0.6), (0.0, 0.0, 0.0)),
    (0.3, 0.3, (1.0, 1.0, 1.0), (5.0, -4.0, 3.0)),
])
def test_two_s_primitives_against_the_closed_form(alpha, beta, Bb, Cc) -> None:
    sa = Shell(0, A, (alpha,), (1.0,))
    sb = Shell(0, Bb, (beta,), (1.0,))
    want = _ss(alpha, A, beta, Bb, Cc)
    for x, y in ((sa, sb), (sb, sa)):
        (row,) = potential_block(x, y, ((1.0, Cc),), one_centre=one_centre)
        assert row[0] == pytest.approx(-want, rel=1e-13)
    # Z scales linearly, and two nuclei add
    (row,) = potential_block(sa, sb, ((2.5, Cc), (0.5, B)),
                             one_centre=one_centre)
    both = 2.5 * want + 0.5 * _ss(alpha, A, beta, Bb, B)
    assert row[0] == pytest.approx(-both, rel=1e-13)


@pytest.mark.parametrize("alpha", [1e-2, 0.37, 1.0, 42.0])
def test_one_centre_s_s_is_two_pi_over_zeta(alpha: float) -> None:
    # A = B = C: K_AB = 1, F_0(0) = 1.
    s = Shell(0, C, (alpha,), (1.0,))
    (row,) = potential_block(s, s, ((1.0, C),), one_centre=one_centre)
    assert row[0] == pytest.approx(-2 * math.pi / (2 * alpha), rel=1e-15)


def test_p_s_against_the_derivative_of_the_closed_form() -> None:
    # d/dA_x exp(-alpha |r - A|^2) = 2 alpha (x - A_x) exp(...), so
    # <p_x|1/r_C|s> = (1/2 alpha) d/dA_x <s_A|1/r_C|s_B>: a witness of the
    # t != 0 columns and the sign of PQ = P - C, by central differences.
    alpha, beta, Cc = 0.9, 0.4, (0.3, -0.5, 0.8)
    Aa = (0.1, 0.2, -0.3)
    sb = Shell(0, B, (beta,), (1.0,))
    p = Shell(1, Aa, (alpha,), (1.0,))
    got = potential_block(p, sb, ((1.0, Cc),), one_centre=one_centre)
    h = 1e-5
    for d, (comp, row) in enumerate(zip(cartesians(1), got, strict=True)):
        assert comp[d] == 1
        up = list(Aa)
        dn = list(Aa)
        up[d] += h
        dn[d] -= h
        fd = (_ss(alpha, up, beta, B, Cc) - _ss(alpha, dn, beta, B, Cc)) / (2 * h)
        assert row[0] == pytest.approx(-fd / (2 * alpha), rel=1e-7)


@pytest.mark.parametrize(("ia", "ib"), [(2, 0), (2, 1), (2, 2), (0, 2),
                                        (4, 2), (2, 4)])
def test_a_d_pair_reads_its_structural_zeros_as_zeros(ia: int, ib: int) -> None:
    # The first consumer of the FULL block. A column t with t_d > a_d + b_d on
    # some axis is exactly 0.0 in every dressed row, and summing only the
    # reachable columns gives the same block BITWISE: they contribute nothing.
    sa, sb = SHELLS[ia], SHELLS[ib]
    ts = hermite_order(sa.l + sb.l)
    rows = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
    nb = sb.n_cart
    want = [0.0] * len(rows)
    n_zero = 0
    for pp in primitive_pairs(sa, sb):
        R = [[one_centre(t, 0, tuple(p - c for p, c in zip(pp.P, Cc,
                                                             strict=True)),
                         math.sqrt(pp.zeta)) for t in ts]
             for _, Cc in NUCLEI]
        for k, ((a, b), row) in enumerate(zip(rows, pp.block, strict=True)):
            reach = [all(t[d] <= a[d] + b[d] for d in range(3)) for t in ts]
            for e, ok in zip(row, reach, strict=True):
                if not ok:
                    assert e == 0.0
                    n_zero += 1
            for (Z, _), Rc in zip(NUCLEI, R, strict=True):
                v = 0.0
                for e, r, ok in zip(row, Rc, reach, strict=True):
                    if ok:
                        v += e * r
                want[k] += -Z * v
    assert n_zero > 0  # the test set really has structural zeros
    got = potential_block(sa, sb, NUCLEI, one_centre=one_centre)
    assert [x for r in got for x in r] == want
    assert len(got) == sa.n_cart and all(len(r) == nb for r in got)


def test_potential_matrix_is_exactly_symmetric() -> None:
    basis = Basis(SHELLS)
    V = potential_matrix(basis, NUCLEI, one_centre=one_centre)
    n = basis.n_ao
    assert len(V) == n and all(len(r) == n for r in V)
    for i in range(n):
        for j in range(n):
            assert V[i][j] == V[j][i]  # bitwise: copied, not recomputed
    # the lower triangle by shell agrees with the recomputed transpose to
    # rounding: a witness of <a|V|b> = <b|V|a>, not of the copy
    off = basis.offsets
    for i, sa in enumerate(SHELLS):
        for j, sb in enumerate(SHELLS[:i]):
            blk = potential_block(sa, sb, NUCLEI, one_centre=one_centre)
            for r in range(sa.n_cart):
                for c in range(sb.n_cart):
                    assert V[off[i] + r][off[j] + c] == pytest.approx(
                        blk[r][c], rel=1e-12, abs=1e-14)


def test_a_positive_charge_attracts() -> None:
    # -Z is folded in: one positive charge gives a negative diagonal.
    basis = Basis(SHELLS)
    V = potential_matrix(basis, ((1.0, (0.2, 0.1, -0.3)),),
                         one_centre=one_centre)
    assert all(V[i][i] < 0.0 for i in range(basis.n_ao))


def test_no_nuclei_is_the_zero_matrix_and_bad_nuclei_are_refused() -> None:
    basis = Basis(SHELLS[:2])
    V = potential_matrix(basis, (), one_centre=one_centre)
    assert all(x == 0.0 for r in V for x in r)
    with pytest.raises(ValueError):
        potential_matrix(basis, ((1.0, (0.0, 0.0)),), one_centre=one_centre)
    with pytest.raises(ValueError):
        potential_matrix(basis, ((1.0,),), one_centre=one_centre)



# --- #236: the screened build ----------------------------------------------
# `potential_matrix(basis, nuclei, pairs=ao.pairs.pair_list(...).kept)` with
# `potential_bound`: at tau = 0 it IS the unscreened V, bitwise; at every tau
# of the ladder a kept block is unchanged bitwise and a dropped block is
# exactly 0.0 where the unscreened block is below tau. The extended case is
# test_ao_pairs.py's H2 chain at THREE molecules: a fourth puts a nucleus far
# enough from a compact pair that `reference.boys` refuses `T > T_MAX`.

from oracle import basis as _ob236  # noqa: E402
from oracle import manifest as _mf236  # noqa: E402

from felizjueves.ao.pairs import pair_list, potential_bound  # noqa: E402

_TAUS_236 = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)


def _chain_236(n: int):
    h = _ob236.from_fixture(_mf236.load(_mf236.FIXTURES / "h2_sto3g_j.json"))
    s = h.shells[0]
    centres = [(0.0, 0.0, 6.0 * m + dz) for m in range(n) for dz in (0.0, 1.4)]
    return (Basis(tuple(Shell(0, c, s.exponents, s.coefficients)
                        for c in centres)),
            tuple((1.0, c) for c in centres))


def _cases_236():
    fx = _mf236.load(_mf236.FIXTURES / "h2o_sto3g_j.json")
    return {
        "SHELLS": (Basis(SHELLS), NUCLEI),
        "h2_chain_3": _chain_236(3),
        "h2o_sto3g_j": (_ob236.from_fixture(fx),
                        tuple((float(a.charge), tuple(a.coord))
                              for a in fx.atoms)),
    }


def _bound_236(nuclei):
    return lambda sa, sb: potential_bound(sa, sb, nuclei,
                                          one_centre=one_centre)


def test_236_screened_potential_at_tau_zero_is_the_unscreened() -> None:
    for name, (basis, nuc) in _cases_236().items():
        pl = pair_list(basis, tau=0.0, bound=_bound_236(nuc))
        assert pl.dropped == ()
        assert potential_matrix(basis, nuc, one_centre=one_centre,
                                pairs=pl.kept) == potential_matrix(
            basis, nuc, one_centre=one_centre), name


def test_236_screened_potential_drops_only_blocks_below_tau() -> None:
    for name, (basis, nuc) in _cases_236().items():
        V0 = potential_matrix(basis, nuc, one_centre=one_centre)
        off, sh = basis.offsets, basis.shells
        for tau in _TAUS_236:
            pl = pair_list(basis, tau=tau, bound=_bound_236(nuc))
            V = potential_matrix(basis, nuc, one_centre=one_centre,
                                 pairs=pl.kept)
            dropped = set(pl.dropped)
            for i, sa in enumerate(sh):
                for j, sb in enumerate(sh):
                    key = (min(i, j), max(i, j))
                    for r in range(sa.n_cart):
                        for c in range(sb.n_cart):
                            got = V[off[i] + r][off[j] + c]
                            want = V0[off[i] + r][off[j] + c]
                            if key in dropped:
                                assert got == 0.0 and abs(want) < tau, (
                                    name, tau, key)
                            else:
                                assert got == want, (name, tau, key)
    basis, nuc = _chain_236(3)
    assert pair_list(basis, tau=1e-6, bound=_bound_236(nuc)).dropped
