"""Pipeline step 3, the sieve ([JEngine, v2, Alg. 3]), and its `no-sieve` lever.

Structural tests run over the table grid, L = 7..16 x n_batches in {1, 3, 6,
9}, against plan/dag.py's MILD DAG as the "before". One registered claim is
discharged here: `jengine/rule-4-needs-the-sieve`.
"""

from __future__ import annotations

from functools import cache
from unittest import mock

import pytest

import felizjueves.plan.sieve as sv
from felizjueves.plan.batching import ALL, AXES
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.dag import BatchDag, Node, build, is_fundamental, mild_parents
from felizjueves.plan.mild import mild_direction
from felizjueves.plan.sieve import NO_SIEVE, parents_along, plan, sieve
from felizjueves.spec import Operator, Spec, Target

# Every grid cell is L > FAST_L_MAX (tiers.py), so a test that plans one is
# `slow`.
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


@pytest.mark.slow
@grid
def test_the_sieved_dag_is_closed(L: int, n_batches: int) -> None:
    # Every non-fundamental node has its parents recorded, and every recorded
    # parent is a node; nothing is left keyed that is not a node.
    for d in _sieved(L, n_batches):
        nodes = d.nodes()
        assert {n for n in nodes if not is_fundamental(n)} == set(d.parents)
        assert all(p in nodes for ps in d.parents.values() for p in ps)


@pytest.mark.slow
@grid
def test_every_target_is_still_produced(L: int, n_batches: int) -> None:
    for before, after in zip(_mild(L, n_batches), _sieved(L, n_batches), strict=True):
        assert after.batch == before.batch
        assert after.targets == before.targets
        assert all(t in after.parents for t in after.targets if not is_fundamental(t))


@pytest.mark.slow
@grid
def test_every_edge_is_a_recurrence_and_m_rises(L: int, n_batches: int) -> None:
    # A node may now deviate from MILD, but only to another legal reduction:
    # its parents are exactly [r - 1d]^(m+1) (and [r - 2d]^(m+1)) for some d.
    for d in _sieved(L, n_batches):
        for n, ps in d.parents.items():
            assert ps in {parents_along(n, k) for k in range(3) if n[0][k]}, n
            assert all(p[1] == n[1] + 1 for p in ps)


@pytest.mark.slow
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


@pytest.mark.slow
@grid
def test_the_sieve_never_grows_a_dag(L: int, n_batches: int) -> None:
    for before, after in zip(_mild(L, n_batches), _sieved(L, n_batches), strict=True):
        assert after.nodes() <= before.nodes()
        assert len(after.intermediates()) <= len(before.intermediates())


@pytest.mark.slow
@pytest.mark.parametrize("L", GRID_L)
def test_the_sieve_prunes_something_at_one_batch(L: int) -> None:
    # Not a vacuous pass: at one batch, every L of the grid loses nodes.
    (before,), (after,) = _mild(L, 1), _sieved(L, 1)
    assert len(after.nodes()) < len(before.nodes())


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
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


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize(("L", "n_batches"), [(7, 1), (10, 3), (16, 9)])
def test_no_sieve_is_milds_dag(L: int, n_batches: int) -> None:
    off = plan(_spec(L, n_batches, (NO_SIEVE,)))
    assert [dict(d.parents) for d in off] == [
        dict(d.parents) for d in _mild(L, n_batches)
    ]
    for d in off:
        assert all(ps == mild_parents(n) for n, ps in d.parents.items())


@pytest.mark.slow  # L = 7 > FAST_L_MAX (tiers.py)
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


# -- #160: Alg. 3's open choices (ii) and (iii), pinned ----------------------------
#
# Alg. 3 leaves open which other direction a child is re-expanded in (line
# 12) and how far deleting i reaches (line 16). plan/sieve.py fixes both
# (its docstring, TWO CHOICES); data/literature/jengine/seed_order_spread.json
# names them and limits.json records each as a reconstruction choice. These
# tests pin them, so a refactor cannot change either silently. Neither is a
# lever, and nothing here measures the alternative.


def _closed(targets, chosen: dict[Node, tuple[Node, ...]]) -> BatchDag:
    """A hand-made DAG: `chosen` parents where given, MILD's elsewhere,
    closed upward. Every edge is a legal recurrence (parents_along)."""
    parents: dict[Node, tuple[Node, ...]] = {}
    stack = [t for t in targets if any(t[0])]
    while stack:
        n = stack.pop()
        if n in parents:
            continue
        ps = chosen.get(n) or mild_parents(n)
        assert ps in {parents_along(n, k) for k in range(3) if n[0][k]}, n
        parents[n] = ps
        stack.extend(p for p in ps if not is_fundamental(p))
    return BatchDag((ALL, 0, 0), frozenset(targets), parents)


@pytest.mark.parametrize(
    ("r", "here", "present", "want"),
    [
        # x (r_x = 2) and z (r_z = 3) both qualify: the smaller component.
        ((2, 1, 3), "y", "xz", "x"),
        # x and z tie at 2: x, by axis order.
        ((2, 1, 2), "y", "xz", "x"),
        # y (3) and z (2) qualify: z, though y comes first by axis.
        ((1, 3, 2), "x", "yz", "z"),
        # only the costlier direction qualifies: it is taken.
        ((2, 1, 3), "y", "z", "z"),
        # nothing else qualifies: no re-expansion (the node survives).
        ((2, 1, 3), "y", "", None),
    ],
)
def test_reexpansion_takes_the_smallest_component_ties_by_axis(
    r, here, present, want
) -> None:
    # Choice (ii), sieve._reexpansion: among the OTHER directions whose
    # parents are all present (and none of them the node being deleted),
    # the smallest reduced component r_d', ties by x, y, z.
    node: Node = (r, 1)
    current = parents_along(node, AXES.index(here))
    avail = {p for a in present for p in parents_along(node, AXES.index(a))}
    gone: Node = ((9, 9, 9), 9)  # deletes nothing the node could use
    got = sv._reexpansion(node, current, avail | set(current), gone)
    assert got == (None if want is None else parents_along(node, AXES.index(want)))
    # The current direction is never "another" one, even when present.
    assert got != current


def test_reexpansion_never_uses_the_node_being_deleted() -> None:
    node: Node = ((2, 1, 3), 1)
    x = parents_along(node, AXES.index("x"))
    z = parents_along(node, AXES.index("z"))
    present = set(x) | set(z)
    assert sv._reexpansion(node, parents_along(node, 1), present, x[0]) == z


def _contested(L: int, n_batches: int) -> list[tuple[Node, bool]]:
    """Every re-expansion test the literal sieve makes on (L, n_batches)
    where more than one other direction qualified -- the case line 12 leaves
    open -- with whether the one taken is the smallest r_d', ties by axis."""
    real, contested = sv._reexpansion, []

    def watch(node, current, present, gone):
        got = real(node, current, present, gone)
        here = sv._axis(node, current)
        ok = [k for k in range(3) if k != here and node[0][k]
              and all(p in present and p != gone
                      for p in parents_along(node, k))]
        if len(ok) > 1:
            best = min(ok, key=lambda k: (node[0][k], k))
            contested.append((node, got == parents_along(node, best)))
        return got

    with mock.patch.object(sv, "_reexpansion", watch):
        for d in _mild(L, n_batches):
            sv.sieve(d)
    return contested


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize(("L", "n_batches"), [(10, 1), (12, 3), (16, 9)])
def test_reexpansion_end_to_end_takes_the_cheaper_direction(
    L: int, n_batches: int
) -> None:
    # Choice (ii) in the shipped plans: where several other directions
    # qualify, the one taken is the smallest r_d', ties by axis. The choice
    # is exercised, not vacuous: at these cells such a case occurs.
    contested = _contested(L, n_batches)
    assert contested, (L, n_batches)
    assert all(ok for _, ok in contested), [n for n, ok in contested if not ok]


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_reexpansion_is_first_contested_at_l10() -> None:
    # In the literal one-batch plans, line 12's choice never arises below
    # L = 10 (one qualifying direction at most); at L = 10 it first does.
    # seed_order_spread.json's open_choices quotes this.
    assert not any(_contested(L, 1) for L in range(5, 10))
    assert len(_contested(10, 1)) == 3


def test_deletion_sweeps_the_childs_other_old_parent() -> None:
    # Choice (iii), line 16's "any unnecessary parents of i". The target
    # [0 2 1] is built along y from a = [0 1 1]^(1) and b = [0 0 1]^(1);
    # [0 3 0] puts [0 2 0]^(1) on the graph, so [0 2 1] re-expands along z
    # out of it. Whichever of a and b the walk tests first is deleted, and
    # the OTHER is not its parent -- Alg. 3's literal reach would leave it in
    # the DAG with no child. The sweep deletes both.
    t1, t2 = ((0, 2, 1), 0), ((0, 3, 0), 0)
    a, b = ((0, 1, 1), 1), ((0, 0, 1), 1)
    dag = _closed({t1, t2}, {t1: parents_along(t1, AXES.index("y"))})
    assert dag.parents[t1] == (a, b)
    assert {c for c, ps in dag.parents.items() if a in ps} == {t1}
    assert {c for c, ps in dag.parents.items() if b in ps} == {t1}
    assert b not in dag.parents[a] and a not in dag.parents.get(b, ())

    out = sieve(dag)
    assert out.parents[t1] == parents_along(t1, AXES.index("z"))
    assert a not in out.nodes() and b not in out.nodes()


# -- rule 4 --------------------------------------------------------------------


@pytest.mark.claim("jengine/rule-4-needs-the-sieve")
@pytest.mark.slow  # L = 5 > FAST_L_MAX (tiers.py)
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


@pytest.mark.slow
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


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
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
