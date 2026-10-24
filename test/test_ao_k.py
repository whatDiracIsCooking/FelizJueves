"""`ao.k`: the exchange matrix `K_mn` by the staged contraction, two sweeps
that must agree, and a third index path through `ao.eri`.

THE WITNESSES (ao/k.py's module docstring).

  internal  `exchange_matrix_reference` (every ORDERED quadruple at weight 1,
            no symmetry) against `exchange_matrix` (one quadruple per
            bra-ket orbit, K's weights: the partner as `X^T`, and `X + X^T`
            -- never `2 X` -- on a diagonal block). It catches a weight error
            and nothing else.
  eri path  `k_from_eris`: every ordered quadruple's full `ao.eri.eri_block`,
            contracted element by element -- no staging, no symmetry.
  libcint   not here: #199 adds the tracked K fixtures.

TAU = 0 THROUGHOUT: K's screen is #200's, tested in test/test_ao_k_screening.py;
here only `tau`'s refusals (negative, and `tau > 0` through the `k=` seam).

THE L <= 2 RULE (milestone 10's standing instruction: no class with
`L = la + lc + lb + ld > 2` is run before #231, host-side Python included).
A whole-basis sweep over a basis holding a `p` shell visits `(pp|pp)`
(L = 4), so:

  fast   H2/STO-3G (s only, L = 0), and the s sub-basis of H2O/STO-3G (its
         four s shells and the matching block of the fixture's density --
         four shells, so every orbit kind `a > b`, `a = b, c > d` and
         `a = b, c = d` is present). `p` enters through single quartets with
         L <= 2 on H2O/STO-3G's O 2p shell (`k_quartet` against `eri_block`,
         and the transpose identity the weights rest on), and through the
         unique sweep over the WHOLE H2O/STO-3G basis with a `k=` hook that
         is `k_quartet` on every quartet with L <= 2 and returns zeros,
         computing nothing, above -- the set of quartets with L <= 2 is
         closed under the bra-ket swap, so K's weights must reproduce the
         ordered sum of that truncated K exactly as they do the full one.
  slow   H2O/STO-3G whole, the issue's acceptance case: deferred to #231.

THE TOLERANCES, stated and not tuned (ao/j.py's, carried over).

  sweeps     |K - K_ref| <= 1e-14 max|K_ref| element by element, and the
             same for each against `k_from_eris`.
  symmetry   |K_ref - K_ref^T| <= 1e-15 max|K_ref| (the reference computes
             the two triangles independently; the unique sweep mirrors).
  quartet    |X - X_eri| <= 1e-14 scale, element by element, with the scale
             `exchange_matrix_and_scale` / `eri_block_and_scale` define.
  scale      scale_K = sum_{cd} |D_cd| scale_eri(ac|bd) to 1e-14 relative:
             the equality that makes `|err K| <= t scale_K` follow from
             `|err eri| <= t scale_eri` by the triangle inequality.

A failure is a defect to find, not a tolerance to widen.
"""

from __future__ import annotations

from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.density import epair
from felizjueves.ao.shell import Basis, n_cartesians
from felizjueves.eri.eri import eri_block, eri_block_and_scale
from felizjueves.eri.k_exchange.k import (
    exchange_matrix,
    exchange_matrix_and_scale,
    exchange_matrix_reference,
    k_from_eris,
    k_quartet,
)
from felizjueves.reference.pq import pq

SWEEP_TOL = 1e-14
SYM_TOL = 1e-15
QUARTET_TOL = 1e-14
L_MAX_RUN = 2  # the milestone's rule: nothing above this is evaluated


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return ob.from_fixture(fx), np.array(fx.density)


@cache
def _case(name: str) -> tuple[Basis, np.ndarray]:
    """The sweep cases: two fixtures, and the s sub-basis of H2O/STO-3G."""
    if name != "h2o_sto3g_s":
        return _fixture(name)
    basis, D = _fixture("h2o_sto3g_j")
    keep = [i for i, s in enumerate(basis.shells) if s.l == 0]
    aos = [basis.offsets[i] for i in keep]  # one AO per s shell
    sub = Basis(tuple(basis.shells[i] for i in keep))
    return sub, D[np.ix_(aos, aos)]


def _lmax_quartet(basis: Basis) -> int:
    """The largest `L` a whole sweep over `basis` evaluates: 4 l_max."""
    return 4 * max(s.l for s in basis.shells)


FAST = ("h2_sto3g_j", "h2o_sto3g_s")
SLOW = ("h2o_sto3g_j",)  # (pp|pp), L = 4: deferred to #231
PARAMS = ([pytest.param(n, id=n) for n in FAST]
          + [pytest.param(n, id=n, marks=pytest.mark.slow) for n in SLOW])


def test_the_fast_cases_keep_to_l_le_2():
    """The rule, checked rather than trusted: every fast sweep case reaches
    only L <= 2, and the slow one is slow because it does not."""
    for name in FAST:
        assert _lmax_quartet(_case(name)[0]) <= L_MAX_RUN, name
    for name in SLOW:
        assert _lmax_quartet(_fixture(name)[0]) > L_MAX_RUN, name
    sub, D = _case("h2o_sto3g_s")
    assert len(sub.shells) == 4 and np.array_equal(D, D.T)


@pytest.mark.parametrize("name", PARAMS)
def test_exchange_matrix(name):
    """Both sweeps agree, each agrees with `k_from_eris`, the reference is
    symmetric, and `exchange_matrix_and_scale` is `exchange_matrix`."""
    basis, D = _case(name)
    K, scale = (np.array(x) for x in exchange_matrix_and_scale(basis, D, pq=pq))
    K0 = np.array(exchange_matrix(basis, D, pq=pq, tau=0.0))
    R = np.array(exchange_matrix_reference(basis, D, pq=pq, tau=0.0))
    E = np.array(k_from_eris(basis, D, pq=pq))
    assert np.array_equal(K, K0)
    big = np.max(np.abs(R))
    sweeps = float(np.max(np.abs(K - R)) / big)
    eris = max(float(np.max(np.abs(K - E))), float(np.max(np.abs(R - E)))) / big
    sym = float(np.max(np.abs(R - R.T)) / big)
    print(f"\n{name}: sweeps {sweeps:.2e}, vs k_from_eris {eris:.2e}, "
          f"symmetry ref {sym:.2e}, max|K| {big:.3g}")
    assert sweeps <= SWEEP_TOL, f"{name}: the sweeps disagree, {sweeps:.2e}"
    assert eris <= SWEEP_TOL, f"{name}: k_from_eris disagrees, {eris:.2e}"
    assert sym <= SYM_TOL, f"{name}: reference K asymmetric, {sym:.2e}"
    assert np.array_equal(K, K.T)  # the unique sweep mirrors exactly
    assert np.all(scale >= np.abs(K))


@pytest.mark.parametrize("name", PARAMS)
def test_the_scale_is_inherited_from_the_eri_scale(name):
    """THE INEQUALITY. `K_ab = sum_cd D_cd (ac|bd)`, so if every ERI element
    carries `|err| <= t scale_eri`, then

        |err K_ab| <= sum_cd |D_cd| |err (ac|bd)|
                   <= t sum_cd |D_cd| scale_eri(ac|bd) = t scale_K[a][b]

    -- the last step being the equality this test pins (to rounding), so the
    K tolerance is #85's per-ERI one, inherited and not chosen."""
    basis, D = _case(name)
    _, scale = exchange_matrix_and_scale(basis, D, pq=pq)
    scale = np.array(scale)
    want = np.zeros_like(scale)
    sh, off = basis.shells, basis.offsets
    for a, sa in enumerate(sh):
        for b, sb in enumerate(sh):
            for c, sc in enumerate(sh):
                for d, sd in enumerate(sh):
                    _, s = eri_block_and_scale((sa, sc), (sb, sd), pq=pq)
                    s = np.array(s).reshape(sa.n_cart, sc.n_cart,
                                            sb.n_cart, sd.n_cart)
                    dcd = np.abs(D[off[c]:off[c] + sc.n_cart,
                                   off[d]:off[d] + sd.n_cart])
                    want[off[a]:off[a] + sa.n_cart,
                         off[b]:off[b] + sb.n_cart] += np.einsum(
                             "ikjl,kl->ij", s, dcd)
    assert np.all(want > 0)
    assert np.all(np.abs(scale - want) <= SWEEP_TOL * want)


def test_an_asymmetric_density_is_rejected():
    basis, D = _case("h2o_sto3g_s")
    bad = D.copy()
    bad[0, 1] += 1e-3
    for f in (exchange_matrix, exchange_matrix_reference,
              exchange_matrix_and_scale, k_from_eris):
        with pytest.raises(ValueError, match="not symmetric"):
            f(basis, bad, pq=pq)


def test_tau_is_checked_and_the_seam_is_unscreened():
    """The screen itself is #200's (test/test_ao_k_screening.py). Here: a
    negative `tau` is refused by both sweeps, and the `k=` seam refuses any
    `tau > 0` -- screening is not a hook's business; tau = 0 runs."""
    basis, D = _case("h2_sto3g_j")
    for f in (exchange_matrix, exchange_matrix_reference):
        with pytest.raises(ValueError, match="tau"):
            f(basis, D, pq=pq, tau=-1.0)
        f(basis, D, pq=pq, tau=0.0)
    hook = partial(k_quartet, pq=pq)
    with pytest.raises(ValueError, match="unscreened"):
        exchange_matrix(basis, D, pq=pq, tau=1e-10, k=hook)
    exchange_matrix(basis, D, pq=pq, tau=0.0, k=hook)


# --- the k seam -------------------------------------------------------------


@pytest.mark.parametrize("name", FAST)
def test_the_k_seam_is_the_host_path(name):
    """`k=partial(k_quartet, pq=pq)` is `==` the path without it."""
    basis, D = _case(name)
    K = exchange_matrix(basis, D, pq=pq)
    calls = []

    def hook(bra, ket, dcd):
        calls.append((bra.la, bra.lb, ket.la, ket.lb))
        return k_quartet(bra, ket, dcd, pq=pq)

    assert exchange_matrix(basis, D, pq=pq, k=hook) == K
    assert exchange_matrix(basis, D, pq=pq, k=partial(k_quartet, pq=pq)) == K
    n = len(basis.shells)
    # one call per bra-ket orbit: n^2 (n^2 + 1) / 2 of the n^4 quadruples
    assert len(calls) == n * n * (n * n + 1) // 2


def test_the_k_seam_refuses_a_wrong_shape():
    basis, D = _case("h2_sto3g_j")
    with pytest.raises(ValueError, match="wrong shape"):
        exchange_matrix(basis, D, pq=pq, k=lambda bra, ket, dcd: [[0.0, 0.0]])


# --- p shells, at L <= 2 ----------------------------------------------------

_H2O = "h2o_sto3g_j"


def _shells():
    """H2O/STO-3G's shells by role: the O 2p, and three s on three centres."""
    basis, _ = _fixture(_H2O)
    sh = basis.shells
    p = next(s for s in sh if s.l == 1)
    s_shells = [s for s in sh if s.l == 0]
    o1s, h1, h2 = s_shells[0], s_shells[2], s_shells[3]
    return p, o1s, h1, h2


def _quartets():
    """Every K class `(la, lc, lb, ld)` with a `p` and L <= 2, as shell
    quartets `(a, c, b, d)` on H2O/STO-3G's shells, the s on distinct
    centres so nothing is symmetric by accident."""
    p, o1s, h1, h2 = _shells()
    return {
        "(ps|ss)": (p, h1, o1s, h2),
        "(sp|ss)": (h1, p, o1s, h2),
        "(ss|ps)": (o1s, h2, p, h1),
        "(ss|sp)": (o1s, h2, h1, p),
        "(pp|ss)": (p, p, h1, h2),
        "(ss|pp)": (h1, h2, p, p),
        "(ps|ps)": (p, h1, p, h2),
        "(ps|sp)": (p, h1, h2, p),
        "(sp|ps)": (h1, p, p, h2),
        "(sp|sp)": (h1, p, h2, p),
    }


def _dcd(sc, sd, seed):
    """A density block over `cartesians(lc) x cartesians(ld)`, c outer."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (sc.n_cart, sd.n_cart))


@pytest.mark.parametrize("cls", sorted(_quartets()))
def test_k_quartet_is_the_eri_contraction(cls):
    """`k_quartet` (the staged form, `V` never formed) against `eri_block`
    contracted with `D_cd`, per element to 1e-14 of the scale."""
    a, c, b, d = _quartets()[cls]
    L = a.l + b.l + c.l + d.l
    assert L <= L_MAX_RUN
    dcd = _dcd(c, d, seed=L + 7 * a.l + 3 * b.l)
    X = np.array(k_quartet(epair(a, c), epair(b, d), dcd.ravel(), pq=pq))
    blk, s = (np.array(x).reshape(a.n_cart, c.n_cart, b.n_cart, d.n_cart)
              for x in eri_block_and_scale((a, c), (b, d), pq=pq))
    want = np.einsum("ikjl,kl->ij", blk, dcd)
    scale = np.einsum("ikjl,kl->ij", s, np.abs(dcd))
    assert X.shape == (a.n_cart, b.n_cart)
    assert np.all(np.abs(X - want) <= QUARTET_TOL * scale), cls


def test_the_transpose_identity_and_why_not_two_x():
    """What K's weights rest on, with a `p` shell at L = 2: for a symmetric
    `D`, `X(b,d|a,c) = X(a,c|b,d)^T` -- and on a diagonal block (`a = b`,
    `c != d`) `X` is NOT symmetric, so the partner adds `X^T`, and J's scalar
    `w = 2` (`2 X`) would be wrong by `X - X^T`, far above rounding.

    The `c`, `d` are the two H 1s: with either on the p shell's own centre
    `X` IS symmetric (write `r - A = (r - P) + (P - A)` on each side; with
    `P_1 = A` the only non-isotropic term is `(P_2 - A)_i (P_2 - A)_j`,
    which is symmetric -- measured 3e-17 relative), so those quartets
    cannot show the difference."""
    p, o1s, h1, h2 = _shells()
    rng = np.random.default_rng(197)
    for c, d in ((h1, h2), (h2, h1)):
        dcd = rng.uniform(-1.0, 1.0, (c.n_cart, d.n_cart))
        X = np.array(k_quartet(epair(p, c), epair(p, d), dcd.ravel(), pq=pq))
        Y = np.array(k_quartet(epair(p, d), epair(p, c), dcd.T.ravel(), pq=pq))
        big = np.max(np.abs(X))
        assert np.max(np.abs(Y - X.T)) <= SWEEP_TOL * big
        assert np.max(np.abs(X - X.T)) > 1e-3 * big
    # and off the diagonal, a p bra against an s ket: the partner is X^T
    dcd = rng.uniform(-1.0, 1.0, (1, 1))
    X = np.array(k_quartet(epair(p, h1), epair(o1s, h2), dcd.ravel(), pq=pq))
    Y = np.array(k_quartet(epair(o1s, h2), epair(p, h1), dcd.ravel(), pq=pq))
    assert np.max(np.abs(Y - X.T)) <= SWEEP_TOL * np.max(np.abs(X))
    # the staged form contracts exactly what eri_block holds
    blk = np.array(eri_block((p, h1), (o1s, h2), pq=pq)).reshape(3, 1, 1, 1)
    assert np.allclose(X, np.einsum("ikjl,kl->ij", blk, dcd), rtol=1e-14,
                       atol=0.0)


def _truncated(calls):
    """`k_quartet` on L <= 2; zeros above, with nothing evaluated."""

    def hook(bra, ket, dcd):
        L = bra.la + bra.lb + ket.la + ket.lb
        calls.append(L)
        if L > L_MAX_RUN:
            return [[0.0] * n_cartesians(ket.la)
                    for _ in range(n_cartesians(bra.la))]
        return k_quartet(bra, ket, dcd, pq=pq)

    return hook


def test_the_weights_hold_with_p_shells_at_l_le_2():
    """The unique sweep over the WHOLE H2O/STO-3G basis, `p` included, with
    every quartet above L = 2 zeroed by the hook (never computed): K's
    weights must still give the ordered sum, because L is invariant under
    the bra-ket swap. The ordered sum is formed here, test-side, through the
    same hook. And the `a = b, c > d` blocks carry `X - X^T` far above
    rounding, so a `2 X` weight could not pass."""
    basis, D = _fixture(_H2O)
    calls: list[int] = []
    hook = _truncated(calls)
    K = np.array(exchange_matrix(basis, D, pq=pq, k=hook))
    assert max(calls) > L_MAX_RUN and min(calls) == 0  # the cut is real
    sh, off = basis.shells, basis.offsets
    n = len(sh)
    R = np.zeros_like(K)
    skew = 0.0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    dcd = D[off[c]:off[c] + sh[c].n_cart,
                            off[d]:off[d] + sh[d].n_cart]
                    X = np.array(hook(epair(sh[a], sh[c]), epair(sh[b], sh[d]),
                                      list(dcd.ravel())))
                    R[off[a]:off[a] + sh[a].n_cart,
                      off[b]:off[b] + sh[b].n_cart] += X
                    if a == b and c > d:
                        skew = max(skew, float(np.max(np.abs(X - X.T))))
    big = np.max(np.abs(R))
    assert np.max(np.abs(K - R)) <= SWEEP_TOL * big
    assert np.max(np.abs(R - R.T)) <= SYM_TOL * big
    assert skew > 1e-3 * big


# --- libcint: the tracked K fixtures (#199) ---------------------------------
#
# THE OUTSIDE WITNESS. `fixtures/*_k.json` record PySCF/libcint's `get_jk` K
# (`manifest.K_ARRAY`, `vk_il = sum_jk (ij|kl) D_jk` -- our `K_ab = sum_cd
# D_cd (ac|bd)`) for the density stored in the file, which is the `_j`
# sibling's, bit for bit (checked below), so a K disagreement is the
# contraction's and not the input's. `generate.k_array` cross-checks `get_jk`
# against `einsum(int2e, dm)` before writing. Comparing against a tracked
# fixture needs no PySCF and carries no `oracle` marker; the drift check of
# the fixtures themselves is test_oracle_manifest.py's `oracle` test.
#
# THE TOLERANCE is J's, inherited the same way: |K - vk| <= 1e-12 scale_K
# element by element, `scale_K = sum_cd |D_cd| scale_eri(ac|bd)` (the
# equality `test_the_scale_is_inherited_from_the_eri_scale` pins), so #85's
# per-ERI 1e-12 carries over by the triangle inequality.
#
# THE L <= 2 RULE. A whole K over a basis with a `p` shell evaluates
# `(pp|pp)`. So the fast tier runs H2/STO-3G whole (s only), and on H2O/
# STO-3G and H2O/6-31G (s and p, no d) the s-s BLOCK of K: an element
# `K_ab` with `a`, `b` both s sums `(a c|b d)` over every `c`, `d`, and that
# quartet's L is `lc + ld <= 2` -- so the block is exact while every quartet
# above L = 2 is zeroed by the hook, never computed (such a quartet has a
# `p` at `a` or `b`, and writes only rows or columns of a `p`). The whole
# matrices, and every cc-pVDZ element (an s-s element there sums `(sd|sd)`,
# L = 4), are `slow`: deferred to #231.

KJ_TOL = 1e-12

K_FAST_WHOLE = ("h2_sto3g_k",)
K_FAST_SS = ("h2o_sto3g_k", "h2o_631g_k")
K_SLOW_WHOLE = ("h2o_sto3g_k", "h2o_631g_k", "h2o_ccpvdz_k")
K_CASES = ("h2_sto3g_k", "h2o_sto3g_k", "h2o_631g_k", "h2o_ccpvdz_k")


@cache
def _kfx(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx)


def _report(name, K, ref, scale, where):
    err = np.abs(K - ref)
    assert np.all(scale > 0)
    worst = float(np.max(err / scale))
    r = ref.ravel()
    ratio = float(K.ravel() @ r / (r @ r))  # a uniform constant shows here
    print(f"\n{name} ({where}): max|K - vk| = {err.max():.2e}, "
          f"max err/scale = {worst:.2e}, K/vk ratio = {ratio:.15g}")
    bad = err > KJ_TOL * scale
    assert not bad.any(), (
        f"{name} ({where}): {int(bad.sum())} of {err.size} elements exceed "
        f"1e-12 * scale; worst err/scale {worst:.2e}; K/vk ratio {ratio:.6g}")


def test_the_k_fixtures_are_the_j_fixtures_inputs():
    """Every `_k` fixture carries `vk` and not `vj`, and its molecule, basis
    (raw and converted), AO layout and density are its `_j` sibling's,
    exactly -- the same input, so only the contraction differs."""
    for name in K_CASES:
        fk, _ = _kfx(name)
        fj = manifest.load(manifest.FIXTURES / f"{name[:-1]}j.json")
        assert manifest.K_ARRAY in fk.arrays, name
        assert manifest.J_ARRAY not in fk.arrays, name
        assert fk.provenance.mol == fj.provenance.mol, name
        assert fk.provenance.basis == fj.provenance.basis, name
        assert fk.provenance.cart is fj.provenance.cart is True, name
        assert fk.density_recipe == fj.density_recipe, name
        assert (fk.atoms, fk.shells, fk.nao, fk.ao_labels, fk.ao_loc,
                fk.converted) == (fj.atoms, fj.shells, fj.nao, fj.ao_labels,
                                  fj.ao_loc, fj.converted), name
        assert np.array_equal(fk.density, fj.density), name
        vk = fk.arrays[manifest.K_ARRAY]
        # K of a symmetric D is symmetric; get_jk(hermi=1) records it so
        assert np.max(np.abs(vk - vk.T)) <= SYM_TOL * np.max(np.abs(vk)), name


def test_the_k_cases_keep_to_the_rule():
    """Fast whole cases are s only; the s-s block cases have no `d`, so an
    s-s element's quartets reach at most L = 2; cc-pVDZ is slow throughout."""
    for name in K_FAST_WHOLE:
        assert _lmax_quartet(_kfx(name)[1]) <= L_MAX_RUN, name
    for name in K_FAST_SS:
        basis = _kfx(name)[1]
        assert max(s.l for s in basis.shells) == 1, name
        assert any(s.l == 0 for s in basis.shells), name
    for name in K_SLOW_WHOLE:
        assert _lmax_quartet(_kfx(name)[1]) > L_MAX_RUN, name
    assert set(K_FAST_WHOLE) | set(K_SLOW_WHOLE) == set(K_CASES)


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, id=n) for n in K_FAST_WHOLE]
    + [pytest.param(n, id=n, marks=pytest.mark.slow) for n in K_SLOW_WHOLE])
def test_exchange_matrix_matches_libcint(name):
    """The whole K against `vk`, to 1e-12 of `scale_K`, element by element.
    The H2O cases are `slow` (they evaluate L > 2): deferred to #231."""
    fx, basis = _kfx(name)
    K, scale = (np.array(x) for x in
                exchange_matrix_and_scale(basis, fx.density, pq=pq))
    assert np.array_equal(K, np.array(exchange_matrix(basis, fx.density,
                                                      pq=pq, tau=0.0)))
    _report(name, K, fx.arrays[manifest.K_ARRAY], scale, "whole")


# H2O/6-31G is `slow` by cost (~12 s interpreted), not by L: off the gate.
@pytest.mark.parametrize("name", [pytest.param(n, id=n, marks=(pytest.mark.slow,)
                                               if n == "h2o_631g_k" else ())
                                  for n in K_FAST_SS])
def test_the_s_block_matches_libcint_at_l_le_2(name):
    """The s-s block of K over the WHOLE basis (`c`, `d` over every shell,
    `p` included), from `exchange_matrix` with every quartet above L = 2
    zeroed by the hook and never computed. The scale is the same block of
    `sum_cd |D_cd| scale_eri(ac|bd)`, from `eri_block_and_scale` on the
    quartets an s-s element sums -- every one at L <= 2."""
    fx, basis = _kfx(name)
    D = np.array(fx.density)
    calls: list[int] = []
    K = np.array(exchange_matrix(basis, D, pq=pq, k=_truncated(calls)))
    assert max(calls) > L_MAX_RUN  # the cut is real: p quartets were zeroed
    sh, off = basis.shells, basis.offsets
    s_idx = [i for i, s in enumerate(sh) if s.l == 0]
    aos = [off[i] for i in s_idx]  # one AO per s shell
    scale = np.zeros((len(aos), len(aos)))
    for ia, a in enumerate(s_idx):
        for ib, b in enumerate(s_idx):
            for c, sc in enumerate(sh):
                for d, sd in enumerate(sh):
                    assert sh[a].l + sc.l + sh[b].l + sd.l <= L_MAX_RUN
                    _, s = eri_block_and_scale((sh[a], sc), (sh[b], sd),
                                               pq=pq)
                    s = np.array(s).reshape(sc.n_cart, sd.n_cart)
                    dcd = np.abs(D[off[c]:off[c] + sc.n_cart,
                                   off[d]:off[d] + sd.n_cart])
                    scale[ia, ib] += float(np.sum(s * dcd))
    ref = fx.arrays[manifest.K_ARRAY][np.ix_(aos, aos)]
    _report(name, K[np.ix_(aos, aos)], ref, scale,
            f"s-s block, {len(aos)}^2 of {basis.n_ao}^2")
