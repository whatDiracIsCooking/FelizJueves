"""Before the sieve, the plan does not depend on the order targets are visited.

WHY THIS HOLDS: MILD is stateless. `mild_direction(r)` is a pure function of
`r`, so a node's parents are fixed by the node alone, and a batch's DAG is the
closure of its target set under `mild_parents` -- whatever order the walk in
`batch_dag` meets the targets in. The node set, the parent map and therefore
the `PlanCost` are functions of the target set alone. This is what keeps #10's
golden counts stable.

WHAT THIS DOES NOT SAY: the pipeline is not order-independent. The sieve
([JEngine, v2, Alg. 3, line 4], #12) re-expands nodes against what it has
already kept, so its output depends on the order it visits them in; §4.1
reports differences up to ~2%. Once the sieve runs, this property stops
holding, and this test is the baseline that dependence is measured against.

Nor is this the emitter's topological sort (pipeline step 4), which is an
order a compiler does see. The order shuffled here is the target-visiting
order of plan construction, nothing else.
"""

from __future__ import annotations

import random
from functools import cache

import pytest

from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.dag import BatchDag, build
from felizjueves.plan.targets import R, hermite_targets
from felizjueves.spec import Operator, Spec, Target

GRID_L = range(7, 17)
GRID_B = (1, 3, 6, 9)
SEEDS = (0, 1, 2, 3, 4)  # fixed, never wall-clock: a failure must reproduce


def _spec(L: int, n_batches: int) -> Spec:
    return Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, (), n_batches)


def _shuffled(L: int, seed: int) -> tuple[R, ...]:
    order = list(hermite_targets(L))
    random.Random(seed).shuffle(order)
    return tuple(order)


@cache
def _reference(L: int, n_batches: int) -> tuple[tuple[BatchDag, ...], PlanCost]:
    dags = build(_spec(L, n_batches))
    return dags, cost(dags)


def _orders(L: int) -> list[tuple[str, tuple[R, ...]]]:
    out = [("reversed", tuple(reversed(hermite_targets(L))))]
    out += [(f"seed={s}", _shuffled(L, s)) for s in SEEDS]
    return out


@pytest.mark.parametrize(
    ("L", "n_batches"), [(L, b) for L in GRID_L for b in GRID_B]
)
def test_the_pre_sieve_dag_is_independent_of_target_order(
    L: int, n_batches: int
) -> None:
    ref_dags, ref_cost = _reference(L, n_batches)
    for label, order in _orders(L):
        assert order != hermite_targets(L), label  # the shuffle shuffled
        dags = build(_spec(L, n_batches), targets=order)
        assert len(dags) == len(ref_dags), label
        for i, (got, ref) in enumerate(zip(dags, ref_dags, strict=True)):
            where = f"{label}, batch {i}"
            assert got.batch == ref.batch, where
            assert got.targets == ref.targets, where
            assert got.nodes() == ref.nodes(), where
            assert dict(got.parents) == dict(ref.parents), where
        assert cost(dags) == ref_cost, label


def test_targets_must_be_a_permutation_of_the_class() -> None:
    order = hermite_targets(7)
    with pytest.raises(ValueError, match="permutation"):
        build(_spec(7, 1), targets=order[1:])
    with pytest.raises(ValueError, match="permutation"):
        build(_spec(7, 1), targets=order + order[:1])
