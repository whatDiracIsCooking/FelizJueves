"""plan/optimum.py (#106): the exact frontier, its pruning, its ceiling, the beam.

No compiler, no card, and no test runs the exact search above `EXACT_MAX_L`
(the L = 6 search does not finish; the module refuses it, and the refusal is
what is tested). The L = 5 cases and the beam above the ceiling are `slow`
(L > FAST_L_MAX, test/tiers.py); the rest is fast tier.

The pruning is checked against `_brute`, written here and sharing nothing with
the module but `cost.node_flops` (the instrument both must use): every
assignment of a direction to every node, by `itertools.product`, memoised on
the RAW layer -- no canonical form, no dominance.
"""

from __future__ import annotations

import time
from functools import cache
from itertools import permutations, product

import pytest
from test_layering import EDGES, _graph
from tiers import split_by_l

from felizjueves.plan import optimum as opt
from felizjueves.plan.cost import cost, node_flops
from felizjueves.plan.dag import build
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

EXACT_LS = range(opt.EXACT_MAX_L + 1)
TIERED_LS = split_by_l(EXACT_LS, lambda L: L)  # L = 5 is `slow`

# The exact frontier per L, as this module computed it (#106). Below L = 5 it
# equals MILD's own one-batch plan; at L = 5 MILD is (51, 161).
OPTIMUM = {
    0: ((0, 0),),
    1: ((1, 3),),
    2: ((5, 15),),
    3: ((13, 41),),
    4: ((26, 86),),
    5: ((48, 160),),
}

# A layer whose frontier is a real trade (found by a seeded random sweep over
# small subsets of the L = 6 targets): one more intermediate buys one FLOP.
# The whole-class frontiers at L <= 5 are single points, so without this the
# frontier tests would be vacuous.
TRADE_LAYER = ((0, 1, 0), (0, 3, 1), (1, 0, 3), (1, 3, 2), (2, 0, 4))
TRADE_FRONTIER = ((25, 45), (26, 44))


def _brute(layer: frozenset) -> tuple[tuple[int, int], ...]:
    @cache
    def solve(s: frozenset) -> frozenset[tuple[int, int]]:
        live = sorted(r for r in s if any(r))
        if not live:
            return frozenset({(0, 0)})
        per_node = []
        for r in live:
            opts = []
            for i, c in enumerate(r):
                if c:
                    ps = {tuple(v - (k == i) for k, v in enumerate(r))}
                    if c >= 2:
                        ps.add(tuple(v - 2 * (k == i) for k, v in enumerate(r)))
                    opts.append((node_flops(c), ps))
            per_node.append(opts)
        out = set()
        for pick in product(*per_node):
            nxt = frozenset().union(*(ps for _, ps in pick))
            here = sum(c for c, _ in pick)
            out.update((len(nxt) + i, here + f) for i, f in solve(nxt))
        return frozenset(out)

    pts = sorted(solve(frozenset(layer)))
    return tuple(p for k, p in enumerate(pts)
                 if all(q[1] > p[1] for q in pts[:k]))


def _mild_point(L: int) -> tuple[int, int]:
    c = cost(build(Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J)))
    return (c.max_intermediates, c.flops)


def _images(layer) -> list[frozenset]:
    return [frozenset(tuple(r[p] for p in perm) for r in layer)
            for perm in permutations(range(3))]


# --- canonical --------------------------------------------------------------


@pytest.mark.parametrize("layer", [
    TRADE_LAYER,
    ((1, 0, 0), (0, 2, 0)),
    ((3, 1, 0), (0, 0, 2), (1, 1, 1)),
    tuple(r for r in hermite_targets(4) if r[0] >= r[2]),
])
def test_canonical_is_equal_for_all_six_images(layer) -> None:
    images = _images(layer)
    canon = {opt.canonical(im) for im in images}
    assert len(canon) == 1
    (c,) = canon
    assert c in images
    assert c == min(images, key=lambda s: tuple(sorted(s)))


def test_canonical_orders_by_sorted_contents_not_inclusion() -> None:
    # Two distinct images of one size are incomparable under `<`, so a bare
    # `min` returns whichever comes first -- the identity, silently. The `key=`
    # is what makes `canonical` a canonical form.
    layer = ((1, 0, 0), (0, 2, 0))
    images = _images(layer)
    assert len(set(images)) == 6
    assert min(images) != min(images[::-1])  # bare min depends on the order
    assert opt.canonical(images[0]) == opt.canonical(images[-1])
    assert opt.canonical(layer) == frozenset({(0, 0, 1), (0, 2, 0)})


# --- pruning ----------------------------------------------------------------


@pytest.mark.parametrize("L", range(4))
def test_pruned_dp_equals_brute_force_on_whole_classes(L) -> None:
    assert opt.optimum(L) == _brute(frozenset(hermite_targets(L)))


@pytest.mark.parametrize("layer", [
    TRADE_LAYER,
    ((2, 2, 0), (1, 1, 1), (0, 0, 3)),
    ((3, 1, 0), (2, 0, 2), (0, 1, 3), (1, 2, 1)),
])
def test_pruned_dp_equals_brute_force_on_small_layers(layer) -> None:
    assert opt.frontier_of(layer) == _brute(frozenset(layer))


# --- the frontier -----------------------------------------------------------


def _is_frontier(fr) -> bool:
    asc = all(a[0] < b[0] and a[1] > b[1] for a, b in zip(fr, fr[1:], strict=False))
    undominated = not any(
        p != q and p[0] <= q[0] and p[1] <= q[1] for p in fr for q in fr)
    return bool(fr) and asc and undominated


def test_the_trade_layer_has_a_real_frontier() -> None:
    fr = opt.frontier_of(TRADE_LAYER)
    assert fr == TRADE_FRONTIER
    assert _is_frontier(fr)


@pytest.mark.parametrize("L", TIERED_LS)
def test_optimum_is_a_frontier_and_pinned(L) -> None:
    fr = opt.optimum(L)
    assert _is_frontier(fr)
    assert fr == OPTIMUM[L]


def test_pareto_drops_dominated_and_duplicate_points() -> None:
    pts = [(5, 10), (3, 12), (5, 10), (4, 12), (6, 9), (7, 9), (2, 20)]
    assert opt.pareto(pts) == ((2, 20), (3, 12), (5, 10), (6, 9))


@pytest.mark.parametrize("L", TIERED_LS)
def test_mild_is_a_point_in_the_space(L) -> None:
    # Same cost convention as plan/cost.py, reached by a separate path: MILD's
    # one-batch plan is weakly dominated, and equal to the optimum below L = 5.
    mild = _mild_point(L)
    fr = opt.optimum(L)
    assert opt.dominated_by([mild], fr)
    if L < 5:
        assert fr == (mild,)


# --- the beam ---------------------------------------------------------------


@pytest.mark.parametrize("L", TIERED_LS)
@pytest.mark.parametrize("width", [1, 4, 16, 64])
def test_beam_is_never_cheaper_than_the_optimum(L, width) -> None:
    b = opt.beam(L, width)
    assert _is_frontier(b)
    assert opt.dominated_by(b, opt.optimum(L))


@pytest.mark.parametrize("L", TIERED_LS)
def test_a_wide_beam_reaches_the_optimum(L) -> None:
    assert opt.beam(L, 64) == opt.optimum(L)


@pytest.mark.slow  # L = 6 > FAST_L_MAX (tiers.py)
def test_beam_runs_above_the_ceiling() -> None:
    b = opt.beam(opt.EXACT_MAX_L + 1, 4)
    assert _is_frontier(b)


def test_beam_refuses_a_bad_width() -> None:
    with pytest.raises(ValueError, match="beam width"):
        opt.beam(3, 0)


# --- the ceiling ------------------------------------------------------------


@pytest.mark.parametrize("L", [opt.EXACT_MAX_L + 1, 7, 16])
def test_optimum_refuses_above_the_ceiling_without_running(L) -> None:
    t0 = time.monotonic()
    with pytest.raises(opt.BeyondExactCeiling) as err:
        opt.optimum(L)
    assert time.monotonic() - t0 < 1.0
    assert isinstance(err.value, ValueError)
    msg = str(err.value)
    assert f"L = {L}" in msg and "EXACT_MAX_L" in msg
    assert "antichain" in msg and "beam" in msg


# --- not a default ----------------------------------------------------------


def test_nothing_in_the_package_imports_the_optimum() -> None:
    # It computes an optimum; it must not become a default. EDGES is checked
    # against the tree both ways by test_layering.py, and the graph here is
    # the tree itself.
    assert not [m for m, e in EDGES.items() if "plan.optimum" in e]
    assert not [m for m, e in _graph().items() if "plan.optimum" in e]
