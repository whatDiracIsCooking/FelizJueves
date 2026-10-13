"""`ao.overlap` and `ao.one_electron`: `S` as the `t = 0` column, and its assembly.

The witnesses here are the Hermite identity (`S` IS element 0 of the block
`ao.pairs.primitive_pairs` dresses, summed) and closed forms derivable by
hand. The libcint comparison is a later issue's (milestone 11, `O1`), not
this one's. Fast tier, no marker, no PySCF.
"""

from __future__ import annotations

import math

import pytest

from felizjueves.ao.hermite import hermite_order
from felizjueves.ao.one_electron import matrix
from felizjueves.ao.overlap import overlap_block, overlap_matrix
from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis, Shell, primitive_norm

# A small, deliberately asymmetric basis: contracted shells of l = 0..3 on
# three centres, unequal contraction lengths.
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


def test_t0_is_index_zero_of_every_hermite_order() -> None:
    for L in range(9):
        assert hermite_order(L)[0] == (0, 0, 0)


@pytest.mark.parametrize("ia", range(len(SHELLS)))
@pytest.mark.parametrize("ib", range(len(SHELLS)))
def test_overlap_block_is_the_summed_t0_column(ia: int, ib: int) -> None:
    # Asserted directly, bitwise: a drift in `dress` or the block layout
    # surfaces here rather than in a device test.
    sa, sb = SHELLS[ia], SHELLS[ib]
    pairs = primitive_pairs(sa, sb)
    want = [0.0] * (sa.n_cart * sb.n_cart)
    for pp in pairs:
        for k, row in enumerate(pp.block):
            want[k] += row[0]
    got = overlap_block(sa, sb)
    assert len(got) == sa.n_cart
    assert all(len(r) == sb.n_cart for r in got)
    assert [x for r in got for x in r] == want


@pytest.mark.parametrize("alpha", [1e-2, 0.37, 1.0, 42.0])
def test_one_s_primitive_with_itself(alpha: float) -> None:
    s = Shell(0, (0.1, -0.2, 0.3), (alpha,), (1.0,))
    (row,) = overlap_block(s, s)
    assert row[0] == pytest.approx((math.pi / (2 * alpha)) ** 1.5,
                                   rel=1e-15)


@pytest.mark.parametrize(("alpha", "beta", "R"), [
    (0.5, 1.3, (0.0, 0.0, 1.4)),
    (2.0, 0.07, (0.9, -1.1, 0.3)),
    (11.0, 3.0, (-0.2, 0.5, 0.6)),
])
def test_two_s_primitives_on_different_centres(alpha, beta, R) -> None:
    # The K_AB-placement test: E already carries exp(-mu X_AB^2) per axis, so
    # a reinstated K_AB would be off by exactly that smooth factor.
    sa = Shell(0, (0.0, 0.0, 0.0), (alpha,), (1.0,))
    sb = Shell(0, R, (beta,), (1.0,))
    zeta, mu = alpha + beta, alpha * beta / (alpha + beta)
    r2 = sum(x * x for x in R)
    want = (math.pi / zeta) ** 1.5 * math.exp(-mu * r2)
    assert overlap_block(sa, sb)[0][0] == pytest.approx(want, rel=1e-14)
    assert overlap_block(sb, sa)[0][0] == pytest.approx(want, rel=1e-14)
    # the doubled-K_AB error is not within that tolerance on these cases
    assert want * math.exp(-mu * r2) != pytest.approx(want, rel=1e-6)


def test_p_against_s_on_one_centre_is_zero_by_parity() -> None:
    # At the origin P = A = B exactly, so X_PA = 0 and the zero is exact.
    p = Shell(1, A, (1.7, 0.3), (0.4, 0.9))
    s = Shell(0, A, (0.8, 0.11), (0.5, 0.6))
    assert overlap_block(p, s) == ((0.0,), (0.0,), (0.0,))
    assert overlap_block(s, p) == ((0.0, 0.0, 0.0),)
    # Off the origin P = (alpha B + beta B)/zeta rounds away from B by an
    # ulp, so the zero holds to rounding against the shells' own scale.
    p = Shell(1, B, (1.7, 0.3), (0.4, 0.9))
    s = Shell(0, B, (0.8, 0.11), (0.5, 0.6))
    scale = math.sqrt(overlap_block(p, p)[0][0] * overlap_block(s, s)[0][0])
    for v in (*(r[0] for r in overlap_block(p, s)), *overlap_block(s, p)[0]):
        assert abs(v) <= 1e-14 * scale


@pytest.mark.parametrize("l", range(4))
def test_normalised_axis_components_have_unit_self_overlap(l: int) -> None:  # noqa: E741
    # ao/shell.py: primitive_norm normalises the AXIS component x^l, so with
    # it folded into the coefficient by the caller, S_xx = 1. Nothing in
    # ao.overlap normalises; this checks it multiplies nothing in either.
    alpha = 0.83
    s = Shell(l, C, (alpha,), (primitive_norm(l, alpha),))
    blk = overlap_block(s, s)
    assert blk[0][0] == pytest.approx(1.0, rel=1e-14)


def test_overlap_matrix_is_exactly_symmetric_and_n_ao_square() -> None:
    basis = Basis(SHELLS)
    S = overlap_matrix(basis)
    n = basis.n_ao
    assert len(S) == n and all(len(r) == n for r in S)
    for i in range(n):
        for j in range(n):
            assert S[i][j] == S[j][i]  # bitwise: copied, not recomputed
    assert all(S[i][i] > 0.0 for i in range(n))


def test_overlap_matrix_places_each_block_at_its_offsets() -> None:
    basis = Basis(SHELLS)
    S = overlap_matrix(basis)
    off = basis.offsets
    for i, sa in enumerate(SHELLS):
        for j, sb in enumerate(SHELLS):
            if j < i:
                continue
            blk = overlap_block(sa, sb)
            for r in range(sa.n_cart):
                for c in range(sb.n_cart):
                    if i == j and c < r:
                        continue
                    assert S[off[i] + r][off[j] + c] == blk[r][c]
    # the lower triangle by shell agrees with the recomputed transpose to
    # rounding (it is copied, so this witnesses <a|b> = <b|a>, not the copy)
    for i, sa in enumerate(SHELLS):
        for j, sb in enumerate(SHELLS[:i]):
            blk = overlap_block(sa, sb)
            for r in range(sa.n_cart):
                for c in range(sb.n_cart):
                    assert S[off[i] + r][off[j] + c] == pytest.approx(
                        blk[r][c], rel=1e-13, abs=1e-16)


def test_matrix_calls_block_once_per_unordered_shell_pair() -> None:
    basis = Basis(SHELLS)
    calls: list[tuple[int, int]] = []
    index = {id(s): k for k, s in enumerate(basis.shells)}

    def block(sa, sb):
        calls.append((index[id(sa)], index[id(sb)]))
        return overlap_block(sa, sb)

    matrix(basis, block)
    n = len(SHELLS)
    assert calls == [(i, j) for i in range(n) for j in range(i, n)]


def test_matrix_reads_only_the_upper_triangle_of_a_diagonal_block() -> None:
    s = Shell(1, A, (1.0,), (1.0,))
    blk = ((1.0, 2.0, 3.0), (-9.0, 4.0, 5.0), (-9.0, -9.0, 6.0))
    M = matrix(Basis((s,)), lambda sa, sb: blk)
    assert M == [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]


@pytest.mark.parametrize("bad", [
    ((0.0, 0.0),),                       # too few rows
    ((0.0,), (0.0,), (0.0,), (0.0,)),    # too many rows
    ((0.0, 0.0), (0.0,), (0.0,)),        # a ragged row
])
def test_matrix_refuses_a_block_of_the_wrong_shape(bad) -> None:
    p = Shell(1, A, (1.0,), (1.0,))
    s = Shell(0, B, (1.0,), (1.0,))
    basis = Basis((p, s))

    def block(sa, sb):
        if sa.l == 1 and sb.l == 0:
            return bad
        return overlap_block(sa, sb)

    with pytest.raises(ValueError, match="shape"):
        matrix(basis, block)


def test_matrix_refuses_a_non_basis() -> None:
    with pytest.raises(ValueError):
        matrix(SHELLS, overlap_block)  # type: ignore[arg-type]



# --- #236: the screened build ----------------------------------------------
# `overlap_matrix(basis, pairs=ao.pairs.pair_list(...).kept)`: at tau = 0 it
# IS the unscreened S, bitwise; at every tau of the ladder a kept block is
# unchanged bitwise and a dropped block is exactly 0.0 where the unscreened
# block is below tau. The extended case is test_ao_pairs.py's H2 chain, the
# one tracked case where a two-centre pair is dropped.

from oracle import basis as _ob236  # noqa: E402
from oracle import manifest as _mf236  # noqa: E402

from felizjueves.ao.pairs import overlap_bound, pair_list  # noqa: E402

_TAUS_236 = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)


def _chain_236(n: int) -> Basis:
    h = _ob236.from_fixture(_mf236.load(_mf236.FIXTURES / "h2_sto3g_j.json"))
    s = h.shells[0]
    return Basis(tuple(Shell(0, (0.0, 0.0, 6.0 * m + dz), s.exponents,
                             s.coefficients)
                       for m in range(n) for dz in (0.0, 1.4)))


def _bases_236() -> dict[str, Basis]:
    out = {"SHELLS": Basis(SHELLS), "h2_chain_8": _chain_236(8)}
    for name in ("h2o_sto3g_j", "hf_ccpvtz"):
        out[name] = _ob236.from_fixture(_mf236.load(_mf236.FIXTURES
                                                    / f"{name}.json"))
    return out


def test_236_screened_overlap_at_tau_zero_is_the_unscreened() -> None:
    for name, basis in _bases_236().items():
        pl = pair_list(basis, tau=0.0, bound=overlap_bound)
        assert pl.dropped == ()
        assert overlap_matrix(basis, pairs=pl.kept) == overlap_matrix(basis), name


def test_236_screened_overlap_drops_only_blocks_below_tau() -> None:
    for name, basis in _bases_236().items():
        S0 = overlap_matrix(basis)
        off, sh = basis.offsets, basis.shells
        for tau in _TAUS_236:
            pl = pair_list(basis, tau=tau, bound=overlap_bound)
            S = overlap_matrix(basis, pairs=pl.kept)
            dropped = set(pl.dropped)
            for i, sa in enumerate(sh):
                for j, sb in enumerate(sh):
                    key = (min(i, j), max(i, j))
                    for r in range(sa.n_cart):
                        for c in range(sb.n_cart):
                            got = S[off[i] + r][off[j] + c]
                            want = S0[off[i] + r][off[j] + c]
                            if key in dropped:
                                assert got == 0.0 and abs(want) < tau, (
                                    name, tau, key)
                            else:
                                assert got == want, (name, tau, key)
    # the chain really drops something at the loosest rung
    assert pair_list(_chain_236(8), tau=1e-6, bound=overlap_bound).dropped


def test_236_matrix_refuses_a_pair_outside_the_upper_triangle() -> None:
    basis = Basis(SHELLS)
    n = len(SHELLS)
    assert matrix(basis, overlap_block, pairs=[
        (i, j) for i in range(n) for j in range(i, n)]) == overlap_matrix(basis)
    for bad in ((1, 0), (0, n), (-1, 0), (0.0, 1)):
        with pytest.raises(ValueError, match="pair"):
            matrix(basis, overlap_block, pairs=[bad])
