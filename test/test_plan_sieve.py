"""Pipeline step 3, the sieve ([JEngine, v2, Alg. 3]), and its `no-sieve` lever.

Structural tests run over the table grid, L = 7..16 x n_batches in {1, 3, 6,
9}, against plan/dag.py's MILD DAG as the "before". One registered claim is
discharged here: `jengine/rule-4-needs-the-sieve`.
"""

from __future__ import annotations

from functools import cache

import pytest

from felizjueves.plan.batching import AXES
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.dag import BatchDag, Node, build, is_fundamental, mild_parents
from felizjueves.plan.mild import mild_direction
from felizjueves.plan.sieve import NO_SIEVE, parents_along, plan, sieve
from felizjueves.spec import Operator, Spec, Target

GRID_L = range(7, 17)
GRID_B = (1, 3, 6, 9)


def _spec(L: int, n_batches: int, levers: tuple[str, ...] = ()) -> Spec:
    return Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, levers, n_batches)


@cache
def _mild(L: int, n_batches: int) -> tuple[BatchDag, ...]:
    return build(_spec(L, n_batches))


@cache
def _sieved(L: int, n_batches: int) -> tuple[BatchDag, ...]:
    return plan(_spec(L, n_batches))


@cache
def _cost(L: int, n_batches: int) -> PlanCost:
    return cost(_sieved(L, n_batches))


grid = pytest.mark.parametrize(
    ("L", "n_batches"), [(L, b) for L in GRID_L for b in GRID_B]
)


# -- structure -----------------------------------------------------------------


@grid
def test_the_sieved_dag_is_closed(L: int, n_batches: int) -> None:
    # Every non-fundamental node has its parents recorded, and every recorded
    # parent is a node; nothing is left keyed that is not a node.
    for d in _sieved(L, n_batches):
        nodes = d.nodes()
        assert {n for n in nodes if not is_fundamental(n)} == set(d.parents)
        assert all(p in nodes for ps in d.parents.values() for p in ps)


@grid
def test_every_target_is_still_produced(L: int, n_batches: int) -> None:
    for before, after in zip(_mild(L, n_batches), _sieved(L, n_batches), strict=True):
        assert after.batch == before.batch
        assert after.targets == before.targets
        assert all(t in after.parents for t in after.targets if not is_fundamental(t))


@grid
def test_every_edge_is_a_recurrence_and_m_rises(L: int, n_batches: int) -> None:
    # A node may now deviate from MILD, but only to another legal reduction:
    # its parents are exactly [r - 1d]^(m+1) (and [r - 2d]^(m+1)) for some d.
    for d in _sieved(L, n_batches):
        for n, ps in d.parents.items():
            assert ps in {parents_along(n, k) for k in range(3) if n[0][k]}, n
            assert all(p[1] == n[1] + 1 for p in ps)


@grid
def test_every_node_is_still_needed(L: int, n_batches: int) -> None:
    # The upward sweep leaves nothing that no target reaches.
    for d in _sieved(L, n_batches):
        reached, stack = set(), list(d.targets)
        while stack:
            n = stack.pop()
            if n not in reached:
                reached.add(n)
                stack.extend(d.parents.get(n, ()))
        assert reached == d.nodes()


@grid
def test_the_sieve_never_grows_a_dag(L: int, n_batches: int) -> None:
    for before, after in zip(_mild(L, n_batches), _sieved(L, n_batches), strict=True):
        assert after.nodes() <= before.nodes()
        assert len(after.intermediates()) <= len(before.intermediates())


@pytest.mark.parametrize("L", GRID_L)
def test_the_sieve_prunes_something_at_one_batch(L: int) -> None:
    # Not a vacuous pass: at one batch, every L of the grid loses nodes.
    (before,), (after,) = _mild(L, 1), _sieved(L, 1)
    assert len(after.nodes()) < len(before.nodes())


@pytest.mark.parametrize(("L", "n_batches"), [(7, 1), (12, 3), (16, 9)])
def test_the_sieve_is_deterministic_and_pure(L: int, n_batches: int) -> None:
    fresh = build(_spec(L, n_batches))
    snapshot = [dict(d.parents) for d in fresh]
    once = [sieve(d) for d in fresh]
    twice = [sieve(d) for d in fresh]
    for a, b in zip(once, twice, strict=True):
        assert dict(a.parents) == dict(b.parents)
        assert list(a.parents) == list(b.parents)  # same insertion order too
    assert [dict(d.parents) for d in fresh] == snapshot  # the input is untouched


# -- the lever -----------------------------------------------------------------


@pytest.mark.parametrize(("L", "n_batches"), [(7, 1), (10, 3), (16, 9)])
def test_no_sieve_is_milds_dag(L: int, n_batches: int) -> None:
    off = plan(_spec(L, n_batches, (NO_SIEVE,)))
    assert [dict(d.parents) for d in off] == [
        dict(d.parents) for d in _mild(L, n_batches)
    ]
    for d in off:
        assert all(ps == mild_parents(n) for n, ps in d.parents.items())


def test_literal_is_sieve_on_and_the_lever_changes_the_key() -> None:
    on, off = _spec(7, 1), _spec(7, 1, (NO_SIEVE,))
    assert on.levers == ()
    assert on.key() != off.key()
    assert cost(plan(on)) != cost(plan(off))


@pytest.mark.parametrize("levers", [("sieve",), ("no-sieve", "other")])
def test_any_other_lever_is_still_refused(levers: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="lever"):
        plan(_spec(7, 1, levers))


def test_parents_along_a_zero_component_is_refused() -> None:
    with pytest.raises(ValueError, match="not reducible"):
        parents_along(((0, 2, 3), 0), AXES.index("x"))


# -- rule 4 --------------------------------------------------------------------


@pytest.mark.claim("jengine/rule-4-needs-the-sieve")
def test_the_sieve_recovers_common_parents_where_mild_cannot() -> None:
    """Johnson's rule 4: reduce to a parent and grandparent already in the tree.

    In the L = 5 tree (one batch), MILD reduces the target [0 2 3]^(0) along y
    -- a pure function of r, so it cannot see that z's parents [0 2 2]^(1) and
    [0 2 1]^(1) are on the graph already -- and keeps [0 1 3]^(1) alive for
    that one child (test_plan_mild.py exhibits this half). After the sieve the
    target reduces to those common parents, and [0 1 3]^(1) is gone.
    """
    (mild,) = build(_spec(5, 1))
    (sieved,) = plan(_spec(5, 1))
    target: Node = ((0, 2, 3), 0)
    common = (((0, 2, 2), 1), ((0, 2, 1), 1))
    sole = ((0, 1, 3), 1)

    assert mild_direction(target[0]) == "y"
    assert mild.parents[target] == mild_parents(target)
    assert sole in mild.parents[target]
    assert all(p in mild.nodes() for p in common)  # on the graph, unused by it

    assert sieved.parents[target] == common
    assert sole not in sieved.nodes()
    assert len(sieved.nodes()) < len(mild.nodes())


# -- batching, post-sieve -------------------------------------------------------


@pytest.mark.parametrize("L", GRID_L)
def test_batching_post_sieve(L: int) -> None:
    """Across 1 -> 3 -> 6 -> 9 batches, post-sieve: targets and intermediates
    per batch fall, and total FLOPs rise strictly -- except FLOPs at L = 7 from
    1 to 3 batches, which stay flat (425, 425; the paper prints 424, 425). That
    exception is why `jengine/batching-trades-flops-for-intermediates` is not
    discharged here, and this test carries no claim mark.
    """
    costs = [_cost(L, nb) for nb in GRID_B]
    for lo, hi in zip(costs, costs[1:], strict=False):
        assert hi.max_targets < lo.max_targets
        assert hi.max_intermediates <= lo.max_intermediates
        assert hi.flops >= lo.flops
    assert costs[1].max_intermediates < costs[0].max_intermediates
    rises = [lo.flops < hi.flops for lo, hi in zip(costs, costs[1:], strict=False)]
    assert rises == ([False, True, True] if L == 7 else [True, True, True])


# -- the j_engine.md anchors ----------------------------------------------------

# (L, batches): printed (targets, intermediates, FLOPs), and ours post-sieve.
ANCHORS = {
    (7, 1): ((120, 125, 424), (120, 126, 425)),
    (7, 3): ((42, 51, 425), (42, 51, 425)),
    (16, 1): ((969, 1647, 5633), (969, 1639, 5644)),
    (16, 3): ((327, 591, 5712), (327, 584, 5763)),
}


@pytest.mark.parametrize("cell", sorted(ANCHORS))
def test_the_anchors_post_sieve(cell: tuple[int, int]) -> None:
    """(7, 3) now reproduces in all three columns. (7, 1) is one intermediate
    and one FLOP high. At L = 16 the sieve prunes past the paper's
    intermediates (1639 < 1647, 584 < 591) while staying above its FLOPs: our
    re-expansion choice trades the other way from theirs."""
    printed, ours = ANCHORS[cell]
    c = _cost(*cell)
    assert (c.max_targets, c.max_intermediates, c.flops) == ours
    assert ours[0] == printed[0]
