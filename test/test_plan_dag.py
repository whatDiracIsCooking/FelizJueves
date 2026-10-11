"""Pipeline step 2, the MILD DAG per batch, and the PlanCost counted from it.

Structural tests run over the table grid, L = 7..16 x n_batches in {1, 3, 6,
9}. Everything here is pre-sieve: nothing deviates from MILD, and the counts
are ours, not the paper's -- the anchors at the bottom say where they differ.
"""

from __future__ import annotations

from functools import cache
from itertools import combinations

import pytest

from felizjueves.plan.batching import AXES, max_targets_per_batch
from felizjueves.plan.cost import PlanCost, cost, node_flops
from felizjueves.plan.dag import BatchDag, Node, batch_dag, build, is_fundamental
from felizjueves.plan.mild import mild_direction
from felizjueves.spec import Operator, Spec, Target

GRID_L = range(7, 17)
GRID_B = (1, 3, 6, 9)


def _spec(L: int, n_batches: int, levers: tuple[str, ...] = ()) -> Spec:
    # A Spec meets the tables through total_am alone (spec.py), so one shell
    # carries all of L.
    return Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, levers, n_batches)


@cache
def _dags(L: int, n_batches: int) -> tuple[BatchDag, ...]:
    return build(_spec(L, n_batches))


@cache
def _cost(L: int, n_batches: int) -> PlanCost:
    return cost(_dags(L, n_batches))


def _expected_parents(node: Node) -> tuple[Node, ...]:
    """Independent of dag.py: [r - 1d]^(m+1), and [r - 2d]^(m+1) if r_d >= 2."""
    (rx, ry, rz), m = node
    d = mild_direction((rx, ry, rz))
    step = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[d]
    r_d = (rx, ry, rz)[AXES.index(d)]
    out = [(tuple(c - s for c, s in zip((rx, ry, rz), step, strict=True)), m + 1)]
    if r_d >= 2:
        out.append(
            (tuple(c - 2 * s for c, s in zip((rx, ry, rz), step, strict=True)), m + 1)
        )
    return tuple(out)


grid = pytest.mark.parametrize(
    ("L", "n_batches"), [(L, b) for L in GRID_L for b in GRID_B]
)


# -- structure -----------------------------------------------------------------


@grid
def test_the_dag_is_closed(L: int, n_batches: int) -> None:
    for dag in _dags(L, n_batches):
        nodes = dag.nodes()
        assert dag.targets <= nodes
        assert all(m == 0 for _, m in dag.targets)
        for node in nodes:
            # every non-fundamental node is reduced, and no fundamental is
            assert (node in dag.parents) is not is_fundamental(node), node
        for ps in dag.parents.values():
            assert set(ps) <= nodes


@grid
def test_m_rises_parent_ward_so_there_is_no_cycle(L: int, n_batches: int) -> None:
    for dag in _dags(L, n_batches):
        for (r, m), ps in dag.parents.items():
            assert all(pm == m + 1 for _, pm in ps)
            assert all(sum(pr) < sum(r) for pr, _ in ps)


@grid
def test_every_edge_is_milds_before_the_sieve(L: int, n_batches: int) -> None:
    for dag in _dags(L, n_batches):
        for node, ps in dag.parents.items():
            assert ps == _expected_parents(node), node


@grid
def test_every_node_is_needed(L: int, n_batches: int) -> None:
    # Nothing in the DAG is unreachable from the batch's own targets.
    for dag in _dags(L, n_batches):
        reached = set(dag.targets)
        stack = list(dag.targets)
        while stack:
            for p in dag.parents.get(stack.pop(), ()):
                if p not in reached:
                    reached.add(p)
                    stack.append(p)
        assert reached == dag.nodes()


@grid
def test_max_targets_is_table_3s_quantity(L: int, n_batches: int) -> None:
    assert _cost(L, n_batches).max_targets == max_targets_per_batch(L, n_batches)


# -- what the batching scheme rests on: [JEngine, v2, §5] -----------------------


def _axial(node: Node) -> bool:
    return sum(1 for c in node[0] if c) <= 1


@pytest.mark.parametrize("L", GRID_L)
@pytest.mark.parametrize("n_batches", (3, 6, 9))
def test_directions_share_only_fundamentals(L: int, n_batches: int) -> None:
    for a, b in combinations(_dags(L, n_batches), 2):
        if a.batch[0] == b.batch[0]:
            continue
        shared = a.intermediates() & b.intermediates()
        assert all(is_fundamental(n) for n in shared), (a.batch, b.batch)


@pytest.mark.parametrize("L", GRID_L)
@pytest.mark.parametrize("n_batches", (6, 9))
def test_sub_batches_of_one_direction_share_only_axials(L: int, n_batches) -> None:
    pairs = 0
    for a, b in combinations(_dags(L, n_batches), 2):
        if a.batch[0] != b.batch[0]:
            continue
        pairs += 1
        shared = a.intermediates() & b.intermediates()
        assert all(_axial(n) for n in shared), (a.batch, b.batch)
    assert pairs


def test_sharing_between_batches_is_counted_once_per_batch() -> None:
    # Each batch is its own kernel: a fundamental every batch needs is in
    # every batch's intermediates, so the max over batches is not the size of
    # the union.
    dags = _dags(10, 3)
    union = set().union(*(d.intermediates() for d in dags))
    assert sum(len(d.intermediates()) for d in dags) > len(union)
    assert all(((0, 0, 0), 1) in d.intermediates() for d in dags)


@pytest.mark.claim("jengine/batching-trades-flops-for-intermediates")
@pytest.mark.parametrize("L", GRID_L)
def test_batching_trades_flops_for_intermediates(L: int) -> None:
    """Across 1 -> 3 -> 6 -> 9 batches: targets and intermediates per batch do
    not rise, and total FLOPs do not fall -- on our own counts.

    1 -> 3 is flat in FLOPs, pre-sieve: the X, Y and Z groups share only
    fundamentals (asserted above), which are Boys-function output and cost no
    FLOP here, so nothing is recomputed. The paper's +1 to +79 at 1 -> 3
    (Table 1) is post-sieve. From 3 on, sub-batches of one direction recompute
    axial intermediates and the rise is strict.
    """
    costs = [_cost(L, nb) for nb in GRID_B]
    for lo, hi in zip(costs, costs[1:], strict=False):
        assert hi.max_targets <= lo.max_targets
        assert hi.max_intermediates <= lo.max_intermediates
        assert hi.flops >= lo.flops
    assert costs[1].flops == costs[0].flops
    assert costs[1].flops < costs[2].flops < costs[3].flops
    assert costs[3].max_intermediates < costs[1].max_intermediates


# -- the cost model ------------------------------------------------------------


@pytest.mark.parametrize(("r_d", "flops"), [(1, 1), (2, 2), (3, 3), (16, 3)])
def test_flops_per_node(r_d: int, flops: int) -> None:
    # PQ_d . always; + once there is a second term; (r_d - 1) . unless it is 1.
    assert node_flops(r_d) == flops


def test_a_zero_component_is_not_a_reduction() -> None:
    with pytest.raises(ValueError):
        node_flops(0)


def test_l_2_by_hand() -> None:
    # Targets: [0], the three units, xx yy zz (r_d = 2), and xy xz yz (MILD
    # reduces a unit). Intermediates: [1x] [1y] [1z] at m = 1, and [0] at
    # m = 1, 2. FLOPs: 3 units x 1 + 3 squares x 2 + 3 mixed x 1 at m = 0,
    # then 3 units x 1 at m = 1.
    assert _cost(2, 1) == PlanCost(flops=15, max_intermediates=5, max_targets=10)


def test_a_plan_has_a_batch() -> None:
    with pytest.raises(ValueError):
        cost(())


def test_a_lever_is_refused() -> None:
    with pytest.raises(ValueError, match="lever"):
        build(_spec(7, 1, ("sieve",)))


def test_build_follows_the_spec_batch_count() -> None:
    assert [d.batch for d in build(_spec(8, 6))] == [
        ("x", 0, 3), ("x", 4, 8),
        ("y", 0, 3), ("y", 4, 8),
        ("z", 0, 3), ("z", 4, 8),
    ]
    assert build(_spec(8, 6))[0].batch == batch_dag(8, ("x", 0, 3)).batch


# -- the j_engine.md anchors ----------------------------------------------------

# (L, batches): printed (targets, intermediates, FLOPs), and ours pre-sieve.
ANCHORS = {
    (7, 1): ((120, 125, 424), (120, 137, 425)),
    (7, 3): ((42, 51, 425), (42, 53, 425)),
    (16, 1): ((969, 1647, 5633), (969, 2021, 6095)),
    (16, 3): ((327, 591, 5712), (327, 701, 6095)),
}


@pytest.mark.parametrize("cell", sorted(ANCHORS))
def test_the_anchors_pre_sieve(cell: tuple[int, int]) -> None:
    """Targets reproduce at all four; FLOPs at (7, 3) only; intermediates at
    none. Every miss is at or above the printed post-sieve value, so it is in
    the direction a pass that deletes intermediates moves."""
    printed, ours = ANCHORS[cell]
    c = _cost(*cell)
    assert (c.max_targets, c.max_intermediates, c.flops) == ours
    assert ours[0] == printed[0]
    assert all(o >= p for o, p in zip(ours, printed, strict=True))
