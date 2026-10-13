"""plan/order_search.py (#170): the frontier, the pruned product, the climb.

Fast tier. Every claim is checked where the answer is checkable by
enumeration: the frontier and the product against brute force over random
point sets and over the real plan's rule-seeded batches, the neighbourhood
against every permutation of a short order, and the climb on a real batch of
(L = 8, 3 batches), where the worst of the 58 rules is one intermediate and
one FLOP behind the best.
"""

from __future__ import annotations

import random
from fractions import Fraction
from functools import cache
from itertools import permutations
from itertools import product as cross
from pathlib import Path

import pytest

from felizjueves.plan import order_search
from felizjueves.plan.batching import partition
from felizjueves.plan.cost import cost
from felizjueves.plan.dag import BatchDag, batch_dag
from felizjueves.plan.optimum import pareto
from felizjueves.plan.order_search import (
    NEIGHBOURHOOD,
    SWEEP,
    Evaluated,
    climb,
    combine,
    dominates,
    frontier,
    moves,
    neighbours,
    point,
    product,
)
from felizjueves.plan.recovery import seed_orders
from felizjueves.plan.sieve import sieve

SRC = Path(__file__).resolve().parents[1] / "src" / "felizjueves"


def _ident(p):
    return p


def _brute_frontier(points):
    """The non-dominated distinct points, by definition."""
    pts = set(points)
    return {p for p in pts
            if not any(q != p and dominates(q, p) for q in pts)}


def _random_points(rng: random.Random, n: int, hi: int = 30):
    return [(rng.randint(0, hi), rng.randint(0, hi)) for _ in range(n)]


# -- the frontier --------------------------------------------------------------


@pytest.mark.parametrize("trial", range(40))
def test_frontier_is_exactly_the_non_dominated_points(trial):
    rng = random.Random(trial)
    pts = _random_points(rng, rng.randint(1, 60))
    got = frontier(pts, key=_ident)
    assert set(got) == _brute_frontier(pts)
    assert len(got) == len(set(got))
    # ascending intermediates, strictly descending FLOPs: optimum's convention
    assert got == pareto(pts)


def test_frontier_keeps_the_first_item_at_a_point():
    items = [("a", (3, 5)), ("b", (2, 9)), ("c", (3, 5)), ("d", (4, 4)),
             ("e", (3, 6))]
    got = frontier(items, key=lambda x: x[1])
    assert [name for name, _ in got] == ["b", "a", "d"]
    assert frontier([], key=_ident) == ()


# -- the product ---------------------------------------------------------------


def _explicit(batches):
    """The cell frontier by brute force: every cross-product combination."""
    return _brute_frontier(combine(parts) for parts in cross(*batches))


@pytest.mark.parametrize("trial", range(30))
def test_product_equals_the_frontier_of_the_explicit_cross_product(trial):
    rng = random.Random(1000 + trial)
    batches = [_random_points(rng, rng.randint(1, 12))
               for _ in range(rng.choice((1, 2, 3)))]
    got = product(batches, key=_ident)
    assert {c.point for c in got} == _explicit(batches)
    for c in got:
        assert len(c.parts) == len(batches)
        assert all(p in b for p, b in zip(c.parts, batches, strict=True))
        assert c.point == combine(c.parts)


def test_product_is_not_the_cell_of_each_batchs_best():
    """Why a product at all: max and sum are not separable. The FLOPs-best
    point of each batch makes a cell the product dominates."""
    a = [(10, 5), (4, 9)]
    b = [(4, 6), (3, 9)]
    got = {c.point for c in product([a, b], key=_ident)}
    assert got == {(4, 15), (10, 11)}
    # the intermediates-best of each batch: (4, 9) + (3, 9) -> (4, 18),
    # which (4, 15) strictly dominates
    assert combine([(4, 9), (3, 9)]) not in got


@cache
def _batch(L: int, nb: int, i: int) -> BatchDag:
    return batch_dag(L, partition(L, nb)[i])


@cache
def _rule_items(L: int, nb: int, i: int) -> tuple[Evaluated, ...]:
    d = _batch(L, nb, i)
    out = []
    for name, rule in seed_orders(widened=True):
        order = tuple(r for r in rule(L) if (r, 0) in d.targets)
        out.append(Evaluated(order, point(cost((sieve(d, seed_order=order),))),
                             name))
    return tuple(out)


def test_combine_is_plan_costs_max_and_sum():
    """The cell point the product assumes is the one cost.py counts."""
    for name, rule in list(seed_orders(widened=True))[:6]:
        sieved = tuple(sieve(_batch(8, 3, i), seed_order=rule(8))
                       for i in range(3))
        per_batch = [point(cost((d,))) for d in sieved]
        assert combine(per_batch) == point(cost(sieved)), name


def test_product_on_the_real_plan_matches_the_cross_product():
    """(L = 8, 3 batches), every batch seeded by each of the 58 rules on its
    own: the pruned product against all 58^3 combinations."""
    batches = [_rule_items(8, 3, i) for i in range(3)]
    got = product(batches, key=lambda e: e.point)
    pts = [[e.point for e in b] for b in batches]
    assert {c.point for c in got} == _explicit(pts)
    for c in got:
        assert all(isinstance(p, Evaluated) for p in c.parts)


def test_product_refuses_an_empty_cell_or_batch():
    with pytest.raises(ValueError):
        product([], key=_ident)
    with pytest.raises(ValueError):
        product([[(1, 1)], []], key=_ident)
    with pytest.raises(ValueError):
        combine([])


# -- the neighbourhood ---------------------------------------------------------


@pytest.mark.parametrize("n", range(0, 7))
def test_neighbourhood_is_every_transposition_and_move_once(n):
    order = tuple(range(n))
    got = list(neighbours(order))
    assert len(moves(n)) == n * (n - 1) // 2 + max(n - 1, 0) * max(n - 2, 0)
    assert len(got) == len(set(got)), "a neighbour counted twice"
    assert order not in got
    expected = set()
    for p in permutations(order):
        diff = [k for k in range(n) if p[k] != order[k]]
        if len(diff) == 2:  # a transposition
            expected.add(p)
        for i in range(n):  # a single-element move
            rest = list(order[:i] + order[i + 1:])
            for j in range(n):
                if tuple(rest[:j] + [order[i]] + rest[j:]) == p and p != order:
                    expected.add(p)
    assert set(got) == expected


# -- the climb -----------------------------------------------------------------


def _synthetic(order):
    """A cheap cost with a real trade-off over permutations of 0..n-1:
    intermediates reward ascending pairs, FLOPs reward displacement."""
    inv = sum(1 for a in range(len(order)) for b in range(a + 1, len(order))
              if order[a] > order[b])
    disp = sum(abs(v - k) for k, v in enumerate(order))
    return (inv, 3 * len(order) - disp + (order[0] if order else 0))


def test_climb_frontier_is_the_frontier_of_everything_it_evaluated():
    seen = []

    def evaluate(order):
        seen.append(order)
        return _synthetic(order)

    starts = {"up": tuple(range(5)), "down": tuple(range(4, -1, -1)),
              "mixed": (2, 0, 4, 1, 3)}
    res = climb(starts, evaluate, budget=10_000, seed=7)
    assert len(seen) == len(set(seen)) == res.evaluated, "memo"
    assert res.evaluated <= 120
    assert {e.point for e in res.frontier} == _brute_frontier(
        _synthetic(o) for o in seen)
    for e in res.frontier:
        assert e.point == _synthetic(e.order)
        assert sorted(e.order) == list(range(5))
    assert len(res.runs) == len(starts) * len(SWEEP)
    assert res.space.starts == ("up", "down", "mixed")
    assert res.space.neighbourhood == NEIGHBOURHOOD
    assert res.space.sweep == SWEEP


def test_an_unexhausted_run_ends_at_a_local_optimum():
    starts = {"down": tuple(range(5, -1, -1))}
    res = climb(starts, _synthetic, budget=10_000, seed=3)
    ref = (min(_synthetic(o)[0] for o in starts.values()),
           min(_synthetic(o)[1] for o in starts.values()))
    for run in res.runs:
        assert not run.exhausted
        rank = order_search._rank(run.weight, ref)
        here = rank(run.final.point)
        assert all(rank(_synthetic(nb)) >= here
                   for nb in neighbours(run.final.order))


def test_the_sweep_runs_from_pure_intermediates_to_pure_flops():
    assert SWEEP[0] == 1 and SWEEP[-1] == 0
    assert list(SWEEP) == sorted(SWEEP, reverse=True)
    starts = {"down": tuple(range(5, -1, -1))}
    res = climb(starts, _synthetic, budget=10_000)
    by_w = {run.weight: run.final.point for run in res.runs}
    # each extreme's run ends no worse on its own axis than the other's
    assert by_w[Fraction(1)][0] <= by_w[Fraction(0)][0]
    assert by_w[Fraction(0)][1] <= by_w[Fraction(1)][1]


def test_a_climb_from_the_worst_rule_reaches_the_best_rules_point():
    """(L = 8, 3 batches), batch 1: the 58 rules reach two points, (70, 211)
    and (71, 212). Started from the worst rule alone, the climb's frontier
    weakly dominates the best rule's point."""
    items = _rule_items(8, 3, 1)
    worst = max(items, key=lambda e: e.point)
    best = min(items, key=lambda e: e.point)
    assert worst.point > best.point, "the rules must disagree here"
    d = _batch(8, 3, 1)

    def evaluate(order):
        return point(cost((sieve(d, seed_order=order),)))

    res = climb({worst.start: worst.order}, evaluate,
                sweep=(Fraction(1), Fraction(0)), budget=150)
    assert any(dominates(e.point, best.point) for e in res.frontier)
    for e in res.frontier:
        assert sorted(e.order) == sorted(worst.order)


def test_two_climbs_with_one_seed_are_identical():
    starts = {"down": tuple(range(6, -1, -1)), "mixed": (3, 1, 6, 0, 5, 2, 4)}
    a = climb(starts, _synthetic, budget=60, seed=11)
    b = climb(starts, _synthetic, budget=60, seed=11)
    assert a == b
    assert a.space.seed == 11 and a.space.budget == 60
    # the seed reaches the neighbour order: some seed walks differently
    others = [climb(starts, _synthetic, budget=60, seed=s).runs
              for s in range(12, 20)]
    assert any(r != a.runs for r in others)


def test_a_run_does_not_depend_on_the_runs_before_it():
    """Each run's RNG is its own, seeded from (seed, start, weight)."""
    s1 = {"down": tuple(range(6, -1, -1))}
    s2 = {"mixed": (3, 1, 6, 0, 5, 2, 4), **s1}
    r1 = {(r.start, r.weight): r.final.order
          for r in climb(s1, _synthetic, budget=10_000, seed=5).runs}
    r2 = {(r.start, r.weight): r.final.order
          for r in climb(s2, _synthetic, budget=10_000, seed=5).runs}
    assert all(r2[k] == v for k, v in r1.items())


def test_the_budget_bounds_every_run():
    res = climb({"down": tuple(range(7, -1, -1))}, _synthetic, budget=5)
    assert all(r.examined <= 5 for r in res.runs)
    assert any(r.exhausted for r in res.runs)
    res0 = climb({"down": (2, 1, 0)}, _synthetic, budget=0)
    assert res0.evaluated == 1
    assert all(r.final.order == (2, 1, 0) and r.steps == 0 for r in res0.runs)


@pytest.mark.parametrize("starts, kw", [
    ({}, {}),
    ({"a": (0, 1, 2), "b": (0, 1, 3)}, {}),
    ({"a": (0, 1, 1)}, {}),
    ({"a": (0, 1, 2)}, {"budget": -1}),
    ({"a": (0, 1, 2)}, {"sweep": ()}),
    ({"a": (0, 1, 2)}, {"sweep": (Fraction(3, 2),)}),
])
def test_climb_refuses(starts, kw):
    with pytest.raises(ValueError):
        climb(starts, _synthetic, **kw)


# -- wired to nothing ----------------------------------------------------------


def test_no_module_imports_order_search():
    """#170's acceptance: imported by nothing outside its own test (and
    test_layering.py's allow-list refuses any new edge to it)."""
    users = [p.relative_to(SRC) for p in SRC.rglob("*.py")
             if p.name != "order_search.py" and "order_search" in p.read_text()]
    assert users == []
