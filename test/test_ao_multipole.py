"""`one_electron.multipole`: the host dipole reference, `<a| q (r - C) |b>`.

Witnesses: libcint's `int1e_r_cart` at a non-zero origin and at 0 (#517's
three H2O fixtures, whole matrices, and #520's `hf_ccpvtz_spd_dipole`, every
shell-pair block, which reaches every class (ss) through (dd) between two
centres), PySCF's RHF `dip_moment`, the (s|s) and (s|p) closed forms, the
origin-shift identity against `one_electron.overlap` and against libcint's
`int1e_ovlp`, and the charge as an exact factor. Every class is L <= 4 and
every shell l <= 2 (milestone 19's cap), so all of it is fast tier; no
PySCF is needed (the fixtures are tracked).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from oracle import manifest
from oracle.basis import from_fixture

from felizjueves.ao.shell import Shell
from felizjueves.one_electron.multipole import (
    hermite_moments,
    multipole_block,
    multipole_matrix,
)
from felizjueves.one_electron.overlap import overlap_block, overlap_matrix

# The host and libcint associate differently; measured worst 8.8e-16 of
# max |r| over every fixture below. The acceptance's bound:
LIBCINT_TOL = 1e-12
ZERO = (0.0, 0.0, 0.0)
DIPOLE_MOLECULES = ("h2o_631g", "h2o_ccpvdz", "h2o_sto3g")


def _rel(got, want) -> float:
    want = np.asarray(want)
    return float(np.max(np.abs(np.asarray(got) - want))
                 / np.max(np.abs(want)))


def test_the_dipole_fixtures_are_the_ones_named_here() -> None:
    assert manifest.dipole_pairs() == list(DIPOLE_MOLECULES)


# --- libcint, whole matrices (#517) ----------------------------------------


@pytest.mark.parametrize("name", DIPOLE_MOLECULES)
def test_the_whole_matrix_is_libcints_int1e_r_cart(name) -> None:
    p = manifest.dipole_pair(name)
    basis = from_fixture(p.dipole)
    assert max(s.l for s in basis.shells) <= 2
    for C, want in ((p.origin, p.r), (ZERO, p.r0)):
        mu = np.array(multipole_matrix(basis, C))
        assert mu.shape == (3, basis.n_ao, basis.n_ao)
        assert _rel(mu, want) <= LIBCINT_TOL, (name, tuple(C))


@pytest.mark.parametrize("name", DIPOLE_MOLECULES)
def test_every_component_is_exactly_symmetric(name) -> None:
    p = manifest.dipole_pair(name)
    mu = np.array(multipole_matrix(from_fixture(p.dipole), p.origin))
    assert np.array_equal(mu, mu.transpose(0, 2, 1))


# --- libcint, every block of the off-centre (dd) witness (#520) ------------


def test_every_block_of_hf_spd_is_libcints_and_every_class_is_off_centre(
        ) -> None:
    d = manifest.dipole_integrals()
    basis = from_fixture(d.fixture)
    shells, off = basis.shells, basis.offsets
    off_centre: set[tuple[int, int]] = set()
    for C, want in ((d.origin, d.r), (ZERO, d.r0)):
        scale = float(np.max(np.abs(want)))
        for i, sa in enumerate(shells):
            for j, sb in enumerate(shells):
                got = np.array(multipole_block(sa, sb, C))
                ref = want[:, off[i]:off[i] + sa.n_cart,
                           off[j]:off[j] + sb.n_cart]
                assert got.shape == ref.shape
                assert np.max(np.abs(got - ref)) <= LIBCINT_TOL * scale, (
                    i, j, tuple(C))
                if sa.centre != sb.centre:
                    off_centre.add((sa.l, sb.l))
    # every class (ss) through (dd), in both orders, between two centres
    assert off_centre == {(la, lb) for la in range(3) for lb in range(3)}


# --- PySCF's dip_moment (#517) ---------------------------------------------


@pytest.mark.parametrize("name", DIPOLE_MOLECULES)
def test_dip_moment_is_nuclear_minus_the_density_traced_with_mu(name) -> None:
    """`-Tr(D mu) + SUM_A Z_A (R_A - C)` at both origins."""
    p = manifest.dipole_pair(name)
    basis = from_fixture(p.dipole)
    D = p.density
    Z = p.dipole.arrays[manifest.DIP_CHARGES_ARRAY]
    R = p.dipole.arrays[manifest.DIP_COORDS_ARRAY]
    for C, want in ((p.origin, p.dip), (np.zeros(3), p.dip0)):
        mu = np.array(multipole_matrix(basis, C))
        el = np.einsum("xij,ji->x", mu, D)
        nuc = Z @ (R - np.asarray(C)[None, :])
        scale = np.max(np.abs(nuc)) + np.max(np.abs(el))
        assert np.max(np.abs((nuc - el) - want)) <= LIBCINT_TOL * scale


# --- the closed forms ------------------------------------------------------

SA = Shell(0, (0.1, -0.3, 0.2), (2.7, 0.41), (0.33, 0.72))
SB_S = Shell(0, (-0.6, 0.9, 1.3), (1.1, 0.19), (0.58, 0.49))
SB_P = Shell(1, (-0.6, 0.9, 1.3), (0.85, 0.23), (0.61, 0.44))
ORIGIN = (0.37, -0.81, 1.23)


def _pair(alpha, beta, A, B):
    zeta = alpha + beta
    P = tuple((alpha * A[d] + beta * B[d]) / zeta for d in range(3))
    mu = alpha * beta / zeta
    kab = math.exp(-mu * sum((A[d] - B[d]) ** 2 for d in range(3)))
    return zeta, P, kab * (math.pi / zeta) ** 1.5


def test_s_s_closed_form() -> None:
    """`<s_A| x_k - C_k |s_B> = SUM c c (pi/zeta)^{3/2} K_AB (P_k - C_k)`."""
    want = [0.0, 0.0, 0.0]
    for a, ca in zip(SA.exponents, SA.coefficients, strict=True):
        for b, cb in zip(SB_S.exponents, SB_S.coefficients, strict=True):
            _, P, w = _pair(a, b, SA.centre, SB_S.centre)
            for k in range(3):
                want[k] += ca * cb * w * (P[k] - ORIGIN[k])
    got = multipole_block(SA, SB_S, ORIGIN)
    for k in range(3):
        assert got[k][0][0] == pytest.approx(want[k], rel=1e-14, abs=0.0)


def test_s_p_closed_form() -> None:
    """`<s_A| x_k - C_k |(x_j - B_j) s_B>`: per primitive pair,
    `(1/(2 zeta) + X_PC X_PB)` on the shared axis when `k == j`, else
    `X_PC,k X_PB,j`, times `c c (pi/zeta)^{3/2} K_AB`."""
    want = [[0.0] * 3 for _ in range(3)]
    for a, ca in zip(SA.exponents, SA.coefficients, strict=True):
        for b, cb in zip(SB_P.exponents, SB_P.coefficients, strict=True):
            zeta, P, w = _pair(a, b, SA.centre, SB_P.centre)
            pc = [P[d] - ORIGIN[d] for d in range(3)]
            pb = [P[d] - SB_P.centre[d] for d in range(3)]
            for k in range(3):
                for j in range(3):  # cartesians(1) is x, y, z
                    f = (0.5 / zeta + pc[k] * pb[k] if k == j
                         else pc[k] * pb[j])
                    want[k][j] += ca * cb * w * f
    got = multipole_block(SA, SB_P, ORIGIN)
    for k in range(3):
        for j in range(3):
            assert got[k][0][j] == pytest.approx(want[k][j], rel=1e-13,
                                                 abs=1e-16), (k, j)
    # and the transpose: (p|s) is (s|p) read the other way round
    back = multipole_block(SB_P, SA, ORIGIN)
    for k in range(3):
        for j in range(3):
            assert back[k][j][0] == pytest.approx(got[k][0][j], rel=1e-14,
                                                  abs=1e-16)


def test_the_hermite_moments_at_e_1_and_2() -> None:
    zeta, x = 1.7, -0.45
    assert hermite_moments(0, x, zeta) == (1.0,)
    assert hermite_moments(1, x, zeta) == (x, 1.0)
    m2 = hermite_moments(2, x, zeta)
    assert m2 == pytest.approx((x * x + 0.5 / zeta, 2 * x, 2.0), rel=1e-15)


# --- the origin shift and the charge ---------------------------------------


@pytest.mark.parametrize("q", [1.0, 0.75])
def test_the_origin_shift_against_the_host_overlap(q) -> None:
    """`mu(C') = mu(C) + q (C - C') S`, `S` from `one_electron.overlap`."""
    p = manifest.dipole_pair("h2o_ccpvdz")
    basis = from_fixture(p.dipole)
    C, C2 = np.array(p.origin), np.array((-1.1, 0.25, 0.6))
    S = np.array(overlap_matrix(basis))
    mu = np.array(multipole_matrix(basis, C, q=q))
    mu2 = np.array(multipole_matrix(basis, C2, q=q))
    want = mu + q * (C - C2)[:, None, None] * S[None]
    assert _rel(mu2, want) <= 1e-14


def test_the_origin_shift_against_libcints_overlap() -> None:
    d = manifest.dipole_integrals()
    basis = from_fixture(d.fixture)
    mu = np.array(multipole_matrix(basis, d.origin))
    mu0 = np.array(multipole_matrix(basis, ZERO))
    want = mu0 - np.asarray(d.origin)[:, None, None] * d.ovlp[None]
    assert _rel(mu, want) <= LIBCINT_TOL


def test_q_minus_one_is_exactly_the_negation() -> None:
    d = manifest.dipole_integrals()
    basis = from_fixture(d.fixture)
    plus = multipole_matrix(basis, d.origin, q=1.0)
    minus = multipole_matrix(basis, d.origin, q=-1.0)
    assert np.array_equal(np.array(minus), -np.array(plus))


def test_e_0_is_the_overlap_block_bitwise() -> None:
    d = manifest.dipole_integrals()
    basis = from_fixture(d.fixture)
    for sa in basis.shells:
        for sb in basis.shells:
            (got,) = multipole_block(sa, sb, ORIGIN, e=0)
            assert [list(r) for r in overlap_block(sa, sb)] == got


def test_the_components_are_cartesians_of_e() -> None:
    assert len(multipole_block(SA, SB_P, ORIGIN, e=0)) == 1
    assert len(multipole_block(SA, SB_P, ORIGIN, e=1)) == 3
    assert len(multipole_block(SA, SB_P, ORIGIN, e=2)) == 6


@pytest.mark.parametrize(("kw", "says"), [
    ({"e": -1}, "is not an int >= 0"),
    ({"e": 1.0}, "is not an int >= 0"),
    ({"C": (0.0, 0.0)}, "three finite floats"),
    ({"C": (0.0, math.nan, 0.0)}, "three finite floats"),
])
def test_refusals(kw, says) -> None:
    args = {"C": ORIGIN, "e": 1} | kw
    with pytest.raises(ValueError, match=says):
        multipole_block(SA, SB_S, args["C"], args["e"])
