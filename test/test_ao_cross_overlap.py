"""The cross-basis overlap `S_{B1,B2}` on the host: `ao.overlap.
cross_overlap_matrix`, its placement `ao.one_electron.rectangular`, its
screen `ao.pairs.cross_pair_list`, and its launch plan
`launch.cross_overlap_launch` / `walk_cross_overlap` (`Target.SX`).

THE OUTSIDE WITNESS is libcint's `gto.intor_cross('int1e_ovlp', rows,
cols)`, through the three tracked `*_x_*` fixtures (`oracle.manifest.
cross_overlap`): H2O with rows STO-3G and columns 6-31G or cc-pVDZ, and
6-31G at another geometry. Each is read in BOTH orientations -- the
fixture, and its transpose against the bases swapped -- so every ordered
class the bases reach is compared, `(2, 0)` and `(2, 1)` among them. The
tolerance is test_ao_overlap.py's, unchanged: `ONE_E_TOL` of the element's
absolute-value scale, Cauchy-Schwarz completed from each basis's own
diagonal (`one_e_scale`'s reason, rectangular).

THE INSIDE WITNESSES. A basis against itself is `overlap_matrix` exactly
off the diagonal shell pairs (the same block entry read) and within
`8 eps |S|` on them (`matrix` mirrors a diagonal block's upper triangle;
`rectangular` reads the whole block). The walk is `==` the host. All fast
tier: `S` has no Boys and nothing here interprets a kernel.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_ao_overlap import ONE_E_TOL, _overlap_scale_block

from felizjueves.ao.pairs import cross_pair_list, overlap_bound
from felizjueves.eri.j_coulomb.launch import cross_overlap_launch, walk_cross_overlap
from felizjueves.one_electron.kernel_sx import sx_spec
from felizjueves.one_electron.one_electron import matrix, rectangular
from felizjueves.one_electron.overlap import (
    cross_overlap_matrix,
    overlap_block,
    overlap_matrix,
)

EPS = 2.0 ** -52
CROSS = sorted(p.stem for p in manifest.cross_fixtures())


@cache
def _fx(name: str):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


@cache
def _cross(name: str):
    """`(rows basis, cols basis, libcint S)` of a cross fixture."""
    fx = _fx(name)
    rows, S = manifest.cross_overlap(fx)
    return ob.from_fixture(rows), ob.from_fixture(fx), S


def _orient(name: str, swapped: bool):
    A, B, S = _cross(name)
    return (B, A, S.T) if swapped else (A, B, S)


ORIENTED = [pytest.param(n, sw, id=f"{n}-{'BA' if sw else 'AB'}")
            for n in CROSS for sw in (False, True)]


def _scale(A, B) -> np.ndarray:
    entry = np.array(rectangular(A, B, _overlap_scale_block))
    da = np.sqrt(np.diag(np.array(matrix(A, _overlap_scale_block))))
    db = np.sqrt(np.diag(np.array(matrix(B, _overlap_scale_block))))
    return np.maximum(entry, np.outer(da, db))


def test_the_cross_fixtures_are_the_three_declared():
    assert CROSS == ["h2o_sto3g_x_631g", "h2o_sto3g_x_631g_moved",
                     "h2o_sto3g_x_ccpvdz"]


def test_the_oriented_cases_reach_every_ordered_class_through_d():
    reached = set()
    for name in CROSS:
        for sw in (False, True):
            A, B, _ = _orient(name, sw)
            reached |= {(a.l, b.l) for a in A.shells for b in B.shells}
    assert reached == {(la, lb) for la in range(3) for lb in range(3)
                       if min(la, lb) <= 1}  # STO-3G has no d


@pytest.mark.parametrize(("name", "swapped"), ORIENTED)
def test_cross_overlap_matrix_against_libcint(name, swapped):
    A, B, ref = _orient(name, swapped)
    S = np.array(cross_overlap_matrix(A, B))
    assert S.shape == ref.shape == (A.n_ao, B.n_ao)
    scale = _scale(A, B)
    assert np.all(scale > 0)
    err = np.abs(S - ref)
    worst = float(np.max(err / scale))
    print(f"\n{name}{' (swapped)' if swapped else ''}: "
          f"max|S - intor_cross| = {err.max():.2e}, err/scale {worst:.2e}")
    assert np.all(err <= ONE_E_TOL * scale)


@pytest.mark.parametrize(("name", "swapped"), ORIENTED)
def test_the_swapped_bases_give_the_transpose_exactly(name, swapped):
    A, B, _ = _orient(name, swapped)
    S = np.array(cross_overlap_matrix(A, B))
    T = np.array(cross_overlap_matrix(B, A))
    # <a|b> and <b|a> walk different E paths: equal to rounding, not bitwise
    assert np.all(np.abs(S - T.T) <= 8 * EPS * _scale(A, B))


@pytest.mark.parametrize("name", ["h2o_sto3g_j", "h2o_ccpvdz_j", "hf_ccpvtz"])
def test_a_basis_against_itself_is_the_overlap_matrix(name):
    basis = ob.from_fixture(_fx(name))
    X = np.array(cross_overlap_matrix(basis, basis))
    S = np.array(overlap_matrix(basis))
    sh = basis.shells
    shell_of = np.repeat(np.arange(len(sh)), [s.n_cart for s in sh])
    si, sj = shell_of[:, None], shell_of[None, :]
    upper = np.triu(np.ones_like(S, dtype=bool))
    # `matrix` reads block(i, j) for i <= j only, and a diagonal pair's upper
    # triangle, and mirrors the rest; `rectangular` computes every element
    # from its own block. Where both read the same block entry: equal.
    same = (si < sj) | ((si == sj) & upper)
    assert np.array_equal(X[same], S[same])
    # elsewhere the entry is <b|a> raised along the other E path: rounding
    assert np.all(np.abs(X - S) <= 8 * EPS * _scale(basis, basis))


# ---------------------------------------------------------------------------
# The rectangular contract
# ---------------------------------------------------------------------------


def test_rectangular_calls_block_once_per_ordered_pair_in_row_major_order():
    A, B, _ = _cross("h2o_sto3g_x_ccpvdz")
    calls = []

    def block(sa, sb):
        calls.append((A.shells.index(sa), B.shells.index(sb)))
        return overlap_block(sa, sb)

    rectangular(A, B, block)
    assert calls == [(i, j) for i in range(len(A.shells))
                     for j in range(len(B.shells))]


def test_rectangular_places_each_block_at_its_two_offsets():
    A, B, _ = _cross("h2o_sto3g_x_ccpvdz")
    S = np.array(cross_overlap_matrix(A, B))
    for i, sa in enumerate(A.shells):
        for j, sb in enumerate(B.shells):
            oa, o_b = A.offsets[i], B.offsets[j]
            assert np.array_equal(
                S[oa:oa + sa.n_cart, o_b:o_b + sb.n_cart],
                np.array(overlap_block(sa, sb)))


@pytest.mark.parametrize("bad", [(-1, 0), (0, -1), (99, 0), (0, 99), (0.0, 0)])
def test_rectangular_refuses_a_pair_outside_the_two_bases(bad):
    A, B, _ = _cross("h2o_sto3g_x_631g")
    with pytest.raises(ValueError, match="is not"):
        cross_overlap_matrix(A, B, pairs=[bad])


def test_rectangular_refuses_a_block_of_the_wrong_shape_and_a_non_basis():
    A, B, _ = _cross("h2o_sto3g_x_631g")
    with pytest.raises(ValueError, match="shape"):
        rectangular(A, B, lambda sa, sb: ((0.0,),) * (sa.n_cart + 1))
    with pytest.raises(ValueError, match="not a Basis"):
        rectangular(A, "B", overlap_block)


# ---------------------------------------------------------------------------
# The screen and the launch plan
# ---------------------------------------------------------------------------


def test_the_cross_screen_at_tau_zero_keeps_every_ordered_pair():
    A, B, _ = _cross("h2o_sto3g_x_ccpvdz")
    pl = cross_pair_list(A, B, tau=0.0, bound=overlap_bound)
    assert pl.kept == tuple((i, j) for i in range(len(A.shells))
                            for j in range(len(B.shells)))
    assert not pl.dropped
    assert np.array_equal(
        np.array(cross_overlap_matrix(A, B, pairs=pl.kept)),
        np.array(cross_overlap_matrix(A, B)))


def _tau_dropping(A, B, fraction: float) -> float:
    """A `tau` at which the screen drops about `fraction` of the pairs (the
    compact molecule's bounds are all large, so a fixed small `tau` drops
    nothing)."""
    bounds = sorted(overlap_bound(a, b) for a in A.shells for b in B.shells)
    return bounds[int(fraction * len(bounds))]


@pytest.mark.parametrize("fraction", [0.2, 0.5])
def test_the_cross_screen_drops_only_blocks_below_tau(fraction):
    A, B, _ = _cross("h2o_sto3g_x_631g_moved")
    tau = _tau_dropping(A, B, fraction)
    pl = cross_pair_list(A, B, tau=tau, bound=overlap_bound)
    assert pl.dropped  # the screen does something at this tau
    full = np.array(cross_overlap_matrix(A, B))
    got = np.array(cross_overlap_matrix(A, B, pairs=pl.kept))
    assert np.max(np.abs(full - got)) < tau
    for i, j in pl.dropped:
        assert max(abs(x) for row in overlap_block(A.shells[i], B.shells[j])
                   for x in row) < tau


@pytest.mark.parametrize("fraction", [0.0, 0.3])
@pytest.mark.parametrize(("name", "swapped"), ORIENTED)
def test_the_walk_of_the_launch_is_the_host_to_zero(name, swapped, fraction):
    A, B, _ = _orient(name, swapped)
    tau = _tau_dropping(A, B, fraction) if fraction else 0.0
    plan = cross_overlap_launch(A, B, tau=tau, bound=overlap_bound)
    assert bool(plan.dropped) == bool(fraction)
    assert (plan.n_a, plan.n_b) == (A.n_ao, B.n_ao)
    assert np.array_equal(
        np.array(walk_cross_overlap(plan, A, B)),
        np.array(cross_overlap_matrix(A, B, pairs=plan.kept)))


def test_the_launch_restates_the_kernel_key_and_its_parameter_order():
    from felizjueves.one_electron.kernel_sx import build_sx_kernel

    A, B, _ = _cross("h2o_sto3g_x_ccpvdz")
    plan = cross_overlap_launch(A, B, tau=0.0, bound=overlap_bound)
    assert [(c.la, c.lb) for c in plan] == sorted(
        {(a.l, b.l) for a in A.shells for b in B.shells})
    for c in plan:
        assert c.spec() == sx_spec(c.la, c.lb)
        params = [p.name for p in build_sx_kernel(c.spec()).params]
        assert params == ["n_rec", "n_a", "n_b", *c.arrays(), "S"]
        for _, _, start, stop in c.blocks:
            assert stop > start
