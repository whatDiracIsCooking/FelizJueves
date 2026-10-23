"""`ao.kinetic`: `T` by the fixed +/-2 ket shift on the overlap axis tables.

The witnesses are closed forms derivable by hand (two `s` primitives; parity
on one centre), the self-adjointness of the operator against a recurrence that
is NOT symmetric in `i <-> j`, the `j >= 2` guard of the `S_{i,j-2}` term, and
an independent one-dimensional quadrature of `1/2 <a'|b'>` (integration by
parts: no ket shift, no `E` table), and, since #240, libcint's `int1e_kin`
through the tracked fixtures (the block at the end of this file; s/p fast,
`slow` from l = 2). Fast tier otherwise, no marker, no PySCF -- except a
case that reaches a class above `la + lb = 4` (test/tiers.py's cut), which
is `slow`.
"""

from __future__ import annotations

import math

import pytest
from tiers import by_l, is_fast

from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.one_electron.kinetic import kinetic_axis, kinetic_block, kinetic_matrix
from felizjueves.one_electron.overlap import overlap_block

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
# SHELLS without its f shell: every pair at L <= FAST_L_MAX (tiers.py), for
# the fast tier's whole-matrix checks; SHELLS itself reaches (ff), `slow`.
SHELLS_THROUGH_D = tuple(s for s in SHELLS if s.l <= 2)


def _pair_param(ia: int, ib: int, id: str):
    return by_l((ia, ib), SHELLS[ia].l + SHELLS[ib].l, id=id)


# --- closed forms -----------------------------------------------------------


@pytest.mark.parametrize(("alpha", "beta", "R"), [
    (0.5, 1.3, (0.0, 0.0, 1.4)),
    (2.0, 0.07, (0.9, -1.1, 0.3)),
    (11.0, 3.0, (-0.2, 0.5, 0.6)),
    (0.8, 0.8, (0.7, 0.7, -0.7)),
])
def test_two_s_primitives_closed_form(alpha, beta, R) -> None:
    # T = mu (3 - 2 mu |R|^2) S. Fails if the +/-2 shift is misplaced (wrong
    # |R|^2 dependence) or K_AB is reinstated (off by exp(-mu R^2)).
    sa = Shell(0, A, (alpha,), (1.0,))
    sb = Shell(0, R, (beta,), (1.0,))
    zeta, mu = alpha + beta, alpha * beta / (alpha + beta)
    r2 = sum(x * x for x in R)
    S = (math.pi / zeta) ** 1.5 * math.exp(-mu * r2)
    want = mu * (3.0 - 2.0 * mu * r2) * S
    assert kinetic_block(sa, sb)[0][0] == pytest.approx(want, rel=1e-13)
    assert kinetic_block(sb, sa)[0][0] == pytest.approx(want, rel=1e-13)
    assert want * math.exp(-mu * r2) != pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize(("alpha", "beta"), [(0.3, 0.3), (1.7, 0.4),
                                             (25.0, 0.9)])
def test_two_s_primitives_on_one_centre(alpha, beta) -> None:
    sa = Shell(0, C, (alpha,), (1.0,))
    sb = Shell(0, C, (beta,), (1.0,))
    mu = alpha * beta / (alpha + beta)
    S = overlap_block(sa, sb)[0][0]
    assert kinetic_block(sa, sb)[0][0] == pytest.approx(3.0 * mu * S,
                                                        rel=1e-14)


def test_s_against_p_on_one_centre_is_zero_by_parity() -> None:
    # At the origin P = A = B exactly, so every odd table entry is exactly 0.
    p = Shell(1, A, (1.7, 0.3), (0.4, 0.9))
    s = Shell(0, A, (0.8, 0.11), (0.5, 0.6))
    assert kinetic_block(p, s) == ((0.0,), (0.0,), (0.0,))
    assert kinetic_block(s, p) == ((0.0, 0.0, 0.0),)
    # Off the origin P rounds away from B by an ulp: zero to rounding.
    p = Shell(1, B, (1.7, 0.3), (0.4, 0.9))
    s = Shell(0, B, (0.8, 0.11), (0.5, 0.6))
    scale = math.sqrt(kinetic_block(p, p)[0][0] * kinetic_block(s, s)[0][0])
    for v in (*(r[0] for r in kinetic_block(p, s)), *kinetic_block(s, p)[0]):
        assert abs(v) <= 1e-14 * scale


# --- the j >= 2 guard --------------------------------------------------------


@pytest.mark.parametrize("j", [0, 1])
def test_the_j_minus_2_term_is_absent_below_j_2(j: int) -> None:
    # Every column kinetic_axis must not read is NaN: a zero-WEIGHTED term
    # would read column j - 2 (which wraps to the far end of the row) and
    # turn 0 * NaN into NaN. Absent, the result is finite and exact.
    beta = 0.7
    s = [math.nan] * 6
    s[j], s[j + 2] = 1.25, -0.5
    want = -2.0 * beta * beta * s[j + 2] + beta * (2 * j + 1) * s[j]
    assert kinetic_axis(s, j, beta) == want


@pytest.mark.parametrize("j", [2, 3])
def test_the_j_minus_2_term_is_read_at_j_2_and_above(j: int) -> None:
    beta = 0.7
    s = [0.0] * (j + 3)
    s[j - 2] = 1.0
    assert kinetic_axis(s, j, beta) == -0.5 * j * (j - 1)


def test_kinetic_axis_refuses_a_row_short_of_j_plus_2() -> None:
    with pytest.raises(ValueError, match="lb \\+ 2"):
        kinetic_axis([1.0, 2.0], 0, 1.0)


def test_a_d_ket_reaches_the_j_minus_2_term() -> None:
    # s bra against d_xx ket on one centre: the x axis has j = 2, and dropping
    # its S_(0,0) term changes the value by 1/2 * 2 * 1 * S_00 * S_y S_z.
    alpha, beta = 0.9, 0.6
    sa = Shell(0, B, (alpha,), (1.0,))
    sb = Shell(2, B, (beta,), (1.0,))
    xx = cartesians(2).index((2, 0, 0))
    got = kinetic_block(sa, sb)[0][xx]
    zeta = alpha + beta
    s00 = math.sqrt(math.pi / zeta)
    # one centre: S_02 = s00 / (2 zeta), S_04 = 3 s00 / (4 zeta^2)
    s02, s04 = s00 / (2 * zeta), 3 * s00 / (4 * zeta * zeta)
    kx = -2 * beta**2 * s04 + 5 * beta * s02 - s00
    ky = -2 * beta**2 * s02 + beta * s00
    want = kx * s00 * s00 + 2 * s02 * ky * s00
    assert got == pytest.approx(want, rel=1e-13)
    assert got != pytest.approx(want + s00**3, rel=1e-6)


# --- an independent quadrature ---------------------------------------------


def _quad_1d(i, j, alpha, beta, a, b):
    """(S, T) on one axis by the trapezoid rule, T as 1/2 <f'|g'>."""
    lo, hi = min(a, b) - 14.0, max(a, b) + 14.0
    n = 12000
    h = (hi - lo) / n
    s = t = 0.0
    for k in range(n + 1):
        x = lo + k * h
        xa, xb = x - a, x - b
        ga, gb = math.exp(-alpha * xa * xa), math.exp(-beta * xb * xb)
        f, g = xa**i * ga, xb**j * gb
        df = (i * xa ** (i - 1) if i else 0.0) * ga - 2 * alpha * xa * f
        dg = (j * xb ** (j - 1) if j else 0.0) * gb - 2 * beta * xb * g
        w = 0.5 if k in (0, n) else 1.0
        s += w * f * g
        t += w * df * dg
    return s * h, 0.5 * t * h


@pytest.mark.parametrize("pair", [_pair_param(ia, ib, f"{ia}-{ib}") for ia, ib
                                  in [(1, 2), (2, 4), (4, 2), (2, 2), (5, 4),
                                      (0, 4)]])
def test_block_against_quadrature(pair) -> None:
    ia, ib = pair
    sa, sb = SHELLS[ia], SHELLS[ib]
    want = [[0.0] * sb.n_cart for _ in range(sa.n_cart)]
    for alpha, ca in zip(sa.exponents, sa.coefficients, strict=True):
        for beta, cb in zip(sb.exponents, sb.coefficients, strict=True):
            q = {}
            for r, a in enumerate(cartesians(sa.l)):
                for c, b in enumerate(cartesians(sb.l)):
                    st = []
                    for d in range(3):
                        key = (d, a[d], b[d])
                        if key not in q:
                            q[key] = _quad_1d(a[d], b[d], alpha, beta,
                                              sa.centre[d], sb.centre[d])
                        st.append(q[key])
                    (sx, tx), (sy, ty), (sz, tz) = st
                    want[r][c] += ca * cb * (tx * sy * sz + sx * ty * sz
                                             + sx * sy * tz)
    got = kinetic_block(sa, sb)
    scale = max(abs(v) for row in want for v in row)
    for r in range(sa.n_cart):
        for c in range(sb.n_cart):
            assert abs(got[r][c] - want[r][c]) <= 1e-11 * scale, (r, c)



# --- self-adjointness and the matrix ---------------------------------------


@pytest.mark.parametrize("pair", [_pair_param(ia, ib, f"{ib}-{ia}")
                                  for ia in range(len(SHELLS))
                                  for ib in range(len(SHELLS))])
def test_swapped_block_is_the_transpose(pair) -> None:
    ia, ib = pair
    # The recurrence shifts only the ket, so (sb, sa) runs a different
    # computation from (sa, sb); agreement is the operator's self-adjointness.
    sa, sb = SHELLS[ia], SHELLS[ib]
    ab, ba = kinetic_block(sa, sb), kinetic_block(sb, sa)
    scale = math.sqrt(max(abs(kinetic_block(sa, sa)[r][r])
                          for r in range(sa.n_cart))
                      * max(abs(kinetic_block(sb, sb)[c][c])
                            for c in range(sb.n_cart)))
    for r in range(sa.n_cart):
        for c in range(sb.n_cart):
            assert abs(ab[r][c] - ba[c][r]) <= 1e-13 * scale, (r, c)


def _symmetric_and_n_ao_square(basis: Basis) -> None:
    T = kinetic_matrix(basis)
    n = basis.n_ao
    assert len(T) == n and all(len(r) == n for r in T)
    for i in range(n):
        for j in range(n):
            assert T[i][j] == T[j][i]
    assert all(T[i][i] > 0.0 for i in range(n))  # kinetic energy is positive


def test_kinetic_matrix_is_exactly_symmetric_and_n_ao_square() -> None:
    _symmetric_and_n_ao_square(Basis(SHELLS_THROUGH_D))


@pytest.mark.slow  # SHELLS reaches (ff), L > FAST_L_MAX (tiers.py)
def test_kinetic_matrix_is_exactly_symmetric_and_n_ao_square_with_f() -> None:
    _symmetric_and_n_ao_square(Basis(SHELLS))


def _places_each_block(shells) -> None:
    basis = Basis(shells)
    T = kinetic_matrix(basis)
    off = basis.offsets
    for i, sa in enumerate(shells):
        for j in range(i, len(shells)):
            sb = shells[j]
            blk = kinetic_block(sa, sb)
            for r in range(sa.n_cart):
                for c in range(sb.n_cart):
                    if i == j and c < r:
                        continue
                    assert T[off[i] + r][off[j] + c] == blk[r][c]


def test_kinetic_matrix_places_each_block_at_its_offsets() -> None:
    _places_each_block(SHELLS_THROUGH_D)


@pytest.mark.slow  # SHELLS reaches (ff), L > FAST_L_MAX (tiers.py)
def test_kinetic_matrix_places_each_block_at_its_offsets_with_f() -> None:
    _places_each_block(SHELLS)



# --- #236: the screened build ----------------------------------------------
# `kinetic_matrix(basis, pairs=ao.pairs.pair_list(...).kept)` with
# `kinetic_bound`: at tau = 0 it IS the unscreened T, bitwise; at every tau
# of the ladder a kept block is unchanged bitwise and a dropped block is
# exactly 0.0 where the unscreened block is below tau. The extended case is
# test_ao_pairs.py's H2 chain.

from oracle import basis as _ob236  # noqa: E402
from oracle import manifest as _mf236  # noqa: E402

from felizjueves.ao.pairs import kinetic_bound, pair_list  # noqa: E402

_TAUS_236 = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)


def _chain_236(n: int) -> Basis:
    h = _ob236.from_fixture(_mf236.load(_mf236.FIXTURES / "h2_sto3g_j.json"))
    s = h.shells[0]
    return Basis(tuple(Shell(0, (0.0, 0.0, 6.0 * m + dz), s.exponents,
                             s.coefficients)
                       for m in range(n) for dz in (0.0, 1.4)))


def _bases_236(fast: bool | None = None) -> dict[str, Basis]:
    """The cases; with `fast`, those on that side of test/tiers.py's cut (a
    basis reaches the pair class `2 l_max`: SHELLS and HF/cc-pVTZ, (ff))."""
    out = {"SHELLS": Basis(SHELLS), "h2_chain_8": _chain_236(8)}
    for name in ("h2o_sto3g_j", "hf_ccpvtz"):
        out[name] = _ob236.from_fixture(_mf236.load(_mf236.FIXTURES
                                                    / f"{name}.json"))
    if fast is None:
        return out
    return {n: b for n, b in out.items()
            if is_fast(2 * max(s.l for s in b.shells)) == fast}


def _236_at_tau_zero(fast: bool) -> None:
    for name, basis in _bases_236(fast).items():
        pl = pair_list(basis, tau=0.0, bound=kinetic_bound)
        assert pl.dropped == ()
        assert kinetic_matrix(basis, pairs=pl.kept) == kinetic_matrix(basis), name


def test_236_screened_kinetic_at_tau_zero_is_the_unscreened() -> None:
    _236_at_tau_zero(fast=True)


@pytest.mark.slow  # SHELLS and HF/cc-pVTZ reach (ff) (tiers.py)
def test_236_screened_kinetic_at_tau_zero_is_the_unscreened_above_l4() -> None:
    _236_at_tau_zero(fast=False)


def _236_drops_only_blocks_below_tau(fast: bool) -> None:
    for name, basis in _bases_236(fast).items():
        T0 = kinetic_matrix(basis)
        off, sh = basis.offsets, basis.shells
        for tau in _TAUS_236:
            pl = pair_list(basis, tau=tau, bound=kinetic_bound)
            T = kinetic_matrix(basis, pairs=pl.kept)
            dropped = set(pl.dropped)
            for i, sa in enumerate(sh):
                for j, sb in enumerate(sh):
                    key = (min(i, j), max(i, j))
                    for r in range(sa.n_cart):
                        for c in range(sb.n_cart):
                            got = T[off[i] + r][off[j] + c]
                            want = T0[off[i] + r][off[j] + c]
                            if key in dropped:
                                assert got == 0.0 and abs(want) < tau, (
                                    name, tau, key)
                            else:
                                assert got == want, (name, tau, key)


def test_236_screened_kinetic_drops_only_blocks_below_tau() -> None:
    _236_drops_only_blocks_below_tau(fast=True)
    assert pair_list(_chain_236(8), tau=1e-6, bound=kinetic_bound).dropped


@pytest.mark.slow  # SHELLS and HF/cc-pVTZ reach (ff) (tiers.py)
def test_236_screened_kinetic_drops_only_blocks_below_tau_above_l4() -> None:
    _236_drops_only_blocks_below_tau(fast=False)

# --- #240: T against libcint's int1e_kin, every fixture that carries it -----
#
# THE TOLERANCE, stated as test_ao_overlap.py's #240 block states S's:
#
#     |T - int1e_kin| <= 1e-12 one_e_scale(E_T),  element by element,
#     E_T[a][b] = SUM_pp |c_a c_b| (pi/zeta)^{3/2}
#                 (|K_x| |S_y| |S_z| + |S_x| |K_y| |S_z| + |S_x| |S_y| |K_z|)
#
# with |K| the three-term ket shift taken term by term in absolute value
# (2 beta^2 |S_{i,j+2}| + beta (2j+1) |S_ij| + 1/2 j(j-1) |S_{i,j-2}|): every
# product `ao.kinetic` adds, before any of it cancels. Measured worst on this
# box (2026-10-13, pyscf 2.14.0 / libcint 6.1.3), printed with -s: 7.6e-16
# of scale (hf_ccpvtz). Fast tier for s/p, `slow` from l = 2; no PySCF.

import numpy as np  # noqa: E402
from oracle import basis as _ob  # noqa: E402
from oracle import manifest as _manifest  # noqa: E402
from test_ao_overlap import ONE_E_PARAMS, ONE_E_TOL, one_e_scale  # noqa: E402

from felizjueves.mmd.hermite import e_axis  # noqa: E402
from felizjueves.one_electron.one_electron import matrix as _matrix  # noqa: E402


def _abs_k(s, j: int, beta: float) -> float:
    k = 2.0 * beta * beta * abs(s[j + 2]) + beta * (2 * j + 1) * abs(s[j])
    if j >= 2:
        k += 0.5 * j * (j - 1) * abs(s[j - 2])
    return k


def kinetic_scale_block(sa: Shell, sb: Shell):
    """`E_T` for one shell pair (the block comment), `kinetic_block`'s shape."""
    A, B = sa.centre, sb.centre
    carts = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
    nb = sb.n_cart
    acc = [0.0] * len(carts)
    for alpha, ca in zip(sa.exponents, sa.coefficients, strict=True):
        for beta, cb in zip(sb.exponents, sb.coefficients, strict=True):
            zeta = alpha + beta
            P = [(alpha * A[d] + beta * B[d]) / zeta for d in range(3)]
            S = [[[e_axis(i, j, zeta, P[d] - A[d], P[d] - B[d])[0]
                   for j in range(sb.l + 3)] for i in range(sa.l + 1)]
                 for d in range(3)]
            w = abs(ca * cb) * (math.pi / zeta) ** 1.5
            for n, (a, b) in enumerate(carts):
                s = [abs(S[d][a[d]][b[d]]) for d in range(3)]
                k = [_abs_k(S[d][a[d]], b[d], beta) for d in range(3)]
                acc[n] += w * (k[0] * s[1] * s[2] + s[0] * k[1] * s[2]
                               + s[0] * s[1] * k[2])
    return tuple(tuple(acc[r * nb:(r + 1) * nb]) for r in range(sa.n_cart))


@pytest.mark.parametrize("name", ONE_E_PARAMS)
def test_kinetic_matrix_against_libcint(name: str) -> None:
    fx = _manifest.load(_manifest.FIXTURES / f"{name}.json")
    basis = _ob.from_fixture(fx)
    T = np.array(kinetic_matrix(basis))
    scale = one_e_scale(np.array(_matrix(basis, kinetic_scale_block)))
    ref = fx.arrays[_manifest.KIN_ARRAY]
    assert T.shape == ref.shape == (fx.nao, fx.nao)
    assert np.all(scale > 0)
    err = np.abs(T - ref)
    worst = float(np.max(err / scale))
    print(f"\n{name}: max|T - int1e_kin| = {err.max():.2e}, "
          f"max err/scale = {worst:.2e}")
    assert np.all(err <= ONE_E_TOL * scale), (
        f"{name}: T disagrees with libcint, worst err/scale {worst:.2e} "
        f"against {ONE_E_TOL:.0e}")
