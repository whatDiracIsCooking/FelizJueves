"""Pipeline step 4, the schedule: a deterministic topological sort of a DAG.

Every property runs on both the pre-sieve MILD DAG (dag.build) and the
sieved one (sieve.plan), over the table grid L = 7..16 x n_batches in
{1, 3, 6, 9}. "Order" here is the EMISSION order of nodes, not the order
targets are visited in while a DAG is built (that is test_plan_order.py).
"""

from __future__ import annotations

import random
from functools import cache
from types import MappingProxyType

import pytest

from felizjueves.plan.dag import BatchDag, build
from felizjueves.plan.schedule import (
    DESCENDING_M,
    Order,
    check,
    schedule,
    tie_break_key,
)
from felizjueves.plan.sieve import NO_SIEVE, plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

GRID_L = range(7, 17)
GRID_B = (1, 3, 6, 9)
SIEVES = ("mild", "sieved")


def _spec(L: int, n_batches: int, levers: tuple[str, ...] = ()) -> Spec:
    return Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, levers, n_batches)


@cache
def _dags(L: int, n_batches: int, which: str) -> tuple[BatchDag, ...]:
    if which == "mild":
        return build(_spec(L, n_batches))
    return plan(_spec(L, n_batches))


grid = pytest.mark.parametrize(
    ("L", "n_batches", "which"),
    [(L, b, w) for L in GRID_L for b in GRID_B for w in SIEVES],
)


@grid
def test_every_node_follows_its_parents(L: int, n_batches: int, which: str) -> None:
    for dag in _dags(L, n_batches, which):
        pos = {n: i for i, n in enumerate(schedule(dag))}
        for node, ps in dag.parents.items():
            for parent in ps:
                assert pos[parent] < pos[node], (node, parent)


@grid
def test_the_schedule_is_a_permutation_of_the_dag(
    L: int, n_batches: int, which: str
) -> None:
    for dag in _dags(L, n_batches, which):
        order = schedule(dag)
        assert len(order) == len(set(order))
        assert sorted(order) == sorted(dag.nodes())


@grid
def test_every_edge_raises_m_by_exactly_one(
    L: int, n_batches: int, which: str
) -> None:
    # Why DESCENDING_M is a valid topological order at all, asserted rather
    # than assumed -- on the sieve's re-parented edges too.
    for dag in _dags(L, n_batches, which):
        for (_, m), ps in dag.parents.items():
            assert all(pm == m + 1 for _, pm in ps)


def _shuffled(dag: BatchDag, rng: random.Random) -> BatchDag:
    """The same DAG, its parents mapping and target set rebuilt in a shuffled
    insertion order."""
    items = list(dag.parents.items())
    rng.shuffle(items)
    targets = list(dag.targets)
    rng.shuffle(targets)
    return BatchDag(dag.batch, frozenset(targets), MappingProxyType(dict(items)))


@grid
def test_the_order_is_deterministic(L: int, n_batches: int, which: str) -> None:
    rng = random.Random(f"{L}-{n_batches}-{which}")
    for dag in _dags(L, n_batches, which):
        want = schedule(dag)
        for _ in range(3):
            assert schedule(_shuffled(dag, rng)) == want


def test_the_order_does_not_depend_on_the_target_visiting_order() -> None:
    # The pre-sieve DAG is independent of the visiting order (test_plan_order),
    # so its schedule is too. Not to be confused with the property above.
    L, n_batches = 9, 3
    targets = list(hermite_targets(L))
    random.Random(0).shuffle(targets)
    a = build(_spec(L, n_batches))
    b = build(_spec(L, n_batches), targets=targets)
    assert [schedule(d) for d in a] == [schedule(d) for d in b]


def test_the_stated_tie_break_key() -> None:
    # The docstring's key, spelled out: descending m, ascending |r|, then
    # lexicographic r.
    assert tie_break_key(((1, 2, 3), 4)) == (-4, 6, 1, 2, 3)
    (dag,) = build(_spec(7, 1))
    order = schedule(dag)
    assert order[0] == ((0, 0, 0), 7)
    assert order[-1][1] == 0
    keys = [tie_break_key(n) for n in order]
    assert keys == sorted(keys)
    assert schedule(dag, strategy=DESCENDING_M) == order


def test_a_strategy_is_a_seam_and_is_checked() -> None:
    (dag,) = build(_spec(7, 1))

    def ascending_m(d: BatchDag) -> Order:
        return tuple(reversed(DESCENDING_M(d)))

    with pytest.raises(ValueError, match="before its parent"):
        schedule(dag, strategy=ascending_m)
    with pytest.raises(ValueError, match="permutation"):
        schedule(dag, strategy=lambda d: DESCENDING_M(d)[1:])
    with pytest.raises(ValueError, match="permutation"):
        check(dag, DESCENDING_M(dag) + DESCENDING_M(dag)[:1])


def test_no_sieve_is_the_mild_dag() -> None:
    # The "mild" leg of the grid is what a Spec naming no-sieve schedules.
    a = plan(_spec(8, 3, (NO_SIEVE,)))
    b = build(_spec(8, 3))
    assert [schedule(d) for d in a] == [schedule(d) for d in b]
