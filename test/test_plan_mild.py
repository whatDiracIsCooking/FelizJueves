"""MILD, [JEngine, v2, Alg. 2]: the tie-break, Johnson's rules, and rule 4's absence.

Three registered claims are discharged here (data/literature/jengine/
claims.json): MILD's tie-break, rules 1-2-3-5, and the batching pass's
tie-break -- the last because the claim is relational, "deliberately the
opposite convention to MILD's", and is asserted beside MILD's own.

Johnson, Gill & Pople (doi:10.1002/qua.560400610) give five rules an optimal
tree obeys. `[JEngine, v2, §4.1]` claims MILD keeps 1, 2, 3 and 5; those are
asserted here as properties over every node up to L = 16. Rule 4 is NOT
asserted: it is exhibited failing, because it is a graph property and MILD is a
pure function of `r` (docs/implementation/mild.md, "MILD against the exact
rules").

Components in the rule tests are named by magnitude, not by axis: `{0 1 j}` is
one component 0, one 1, one j, in any axis order.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from felizjueves.plan.batching import batch_direction
from felizjueves.plan.mild import AXES, R, mild_direction

L_MAX = 16


def _every_r(L: int) -> Iterator[R]:
    """Every Hermite index with 0 < |r| <= L -- every node MILD is asked about."""
    for x in range(L + 1):
        for y in range(L + 1 - x):
            for z in range(L + 1 - x - y):
                if x + y + z:
                    yield (x, y, z)


def _reduced(r: R) -> int:
    """The component MILD reduces, by value."""
    return r[AXES.index(mild_direction(r))]


# -- the tie-break -------------------------------------------------------------


@pytest.mark.parametrize(
    "r,expected",
    [
        ((1, 1, 2), "y"),  # rx == ry < rz
        ((2, 1, 1), "z"),  # ry == rz < rx
        ((1, 2, 1), "x"),  # rx == rz < ry
        ((2, 2, 2), "z"),  # all equal
    ],
)
def test_the_round_robin_takes_the_cyclic_successor(r: R, expected: str) -> None:
    assert mild_direction(r) == expected


@pytest.mark.parametrize(
    "r,expected",
    [
        ((0, 3, 5), "y"),  # a zero is excluded from the minimum, not the winner
        ((4, 0, 0), "x"),
        ((0, 2, 2), "z"),  # a tie among the non-zero only: y, z -> z
        ((3, 0, 3), "x"),  # z, x -> x
    ],
)
def test_a_zero_is_excluded_from_the_minimum(r: R, expected: str) -> None:
    assert mild_direction(r) == expected


@pytest.mark.parametrize("r", [(0, 0, 0), (-1, 1, 1), (1, 1)])
def test_a_node_with_no_direction_raises(r) -> None:
    # [0](m) is a fundamental and is never reduced: asking is a caller bug.
    with pytest.raises(ValueError):
        mild_direction(r)


def test_the_direction_is_always_reducible() -> None:
    for r in _every_r(L_MAX):
        assert _reduced(r) > 0


def test_the_direction_is_the_minimum_non_zero_component() -> None:
    for r in _every_r(L_MAX):
        assert _reduced(r) == min(c for c in r if c)


def _tied(r: R, value: int) -> list[int]:
    """The axes whose component equals `value`, ascending."""
    return [i for i in range(3) if r[i] == value]


@pytest.mark.claim("jengine/mild-tie-break")
def test_the_whole_tie_break_on_every_node() -> None:
    # Restated as the claim states it, not as mild.py computes it: the axes
    # holding the smallest non-zero value; one wins outright, two go to the
    # cyclic successor of the pair, three go to z.
    successor = {(0, 1): 1, (1, 2): 2, (0, 2): 0}
    seen = {1: 0, 2: 0, 3: 0}
    for r in _every_r(L_MAX):
        tied = _tied(r, min(c for c in r if c))
        seen[len(tied)] += 1
        want = tied[0] if len(tied) == 1 else successor.get(tuple(tied), 2)
        assert mild_direction(r) == AXES[want], r
    assert all(seen.values())


@pytest.mark.claim("jengine/batch-tie-break")
def test_mild_and_the_batching_pass_break_ties_oppositely() -> None:
    """[JEngine, v2, Alg. 2] against Alg. 4: MILD minimises, the batching pass
    maximises, and their tie-breaks are deliberately inverse -- successor and
    z for MILD, predecessor and x for batching (j_engine.md, "The scheme").

    The batching rule is restated here exhaustively, as the claim states it.
    Then the comparison: where both rules face the SAME tied set -- every
    non-zero component equal, so the smallest non-zero is also the maximum --
    they disagree on every node with a real tie and agree on every node
    without one. That is "disagree exactly where the specs say they do".
    """
    predecessor = {(0, 1): 0, (1, 2): 1, (0, 2): 2}
    successor = {(0, 1): 1, (1, 2): 2, (0, 2): 0}
    disagreed = 0
    for r in _every_r(L_MAX):
        hi = _tied(r, max(r))
        want = hi[0] if len(hi) == 1 else predecessor.get(tuple(hi), 0)
        assert batch_direction(r) == AXES[want], r

        lo = _tied(r, min(c for c in r if c))
        if lo != hi:
            continue  # different candidate sets: the comparison says nothing
        if len(lo) == 1:
            assert mild_direction(r) == batch_direction(r), r  # forced, axial
            continue
        disagreed += 1
        assert batch_direction(r) == AXES[predecessor.get(tuple(lo), 0)], r
        assert mild_direction(r) == AXES[successor.get(tuple(lo), 2)], r
    # (a, a, 0) in three rotations and (a, a, a), for every a with |r| <= 16
    assert disagreed == 3 * (L_MAX // 2) + L_MAX // 3


# -- Johnson, Gill & Pople's rules ---------------------------------------------


# (lo, mid) of the sorted components, the least j, and the reduced component's
# value -- or None for "the only non-zero one", which is rule 1's forced choice.
JOHNSON_RULES = {
    # {0 0 j}: one non-zero component, so there is no choice to make -- MILD
    # reduces it because it is the only reducible component.
    1: ((0, 0), 1, None),
    # {0 1 j}, j >= 2: reduce the unit component.
    2: ((0, 1), 2, 1),
    # {1 1 j}, j >= 2: never reduce the j. (Which unit component is free.)
    3: ((1, 1), 2, 1),
    # {0 2 j}, j >= 3: reduce the 2.
    5: ((0, 2), 3, 2),
}


@pytest.mark.claim("jengine/mild-keeps-rules-1-2-3-5")
@pytest.mark.parametrize("rule", sorted(JOHNSON_RULES))
def test_johnsons_rule_holds_on_every_node(rule: int) -> None:
    (lo_mid, j_min, reduced) = JOHNSON_RULES[rule]
    seen = 0
    for r in _every_r(L_MAX):
        lo, mid, hi = sorted(r)
        if (lo, mid) != lo_mid or hi < j_min:
            continue
        seen += 1
        if reduced is None:
            assert [AXES[i] for i, c in enumerate(r) if c] == [mild_direction(r)]
        else:
            assert _reduced(r) == reduced, r
    assert seen  # no rule passes vacuously


# -- rule 4, exhibited rather than asserted ------------------------------------

Node = tuple[R, int]  # ([r], m)


def _parents(node: Node, direction: str) -> list[Node]:
    """[r]^(m) along i needs [r - 1_i]^(m+1), and [r - 2_i]^(m+1) if r_i >= 2."""
    r, m = node
    i = AXES.index(direction)
    out = [(tuple(c - (k == i) for k, c in enumerate(r)), m + 1)]
    if r[i] >= 2:
        out.append((tuple(c - 2 * (k == i) for k, c in enumerate(r)), m + 1))
    return out


def _mild_tree(L: int) -> tuple[set[Node], dict[Node, set[Node]]]:
    """Every node MILD puts in the graph for the targets [r]^(0), |r| <= L,
    and each node's children."""
    nodes: set[Node] = set()
    children: dict[Node, set[Node]] = {}
    stack: list[Node] = [((0, 0, 0), 0), *((r, 0) for r in _every_r(L))]
    while stack:
        node = stack.pop()
        if node in nodes:
            continue
        nodes.add(node)
        if sum(node[0]) == 0:
            continue
        for p in _parents(node, mild_direction(node[0])):
            children.setdefault(p, set()).add(node)
            stack.append(p)
    return nodes, children


def test_mild_ignores_common_parents_and_the_sieve_recovers_them() -> None:
    """Johnson's rule 4 does not hold under MILD, and this is not an oversight.

    In the L = 5 tree, the target [0 2 3]^(0) has its common parents on the
    graph already: reduced along z it needs [0 2 2]^(1) and [0 2 1]^(1), both of
    which MILD put there for other nodes. MILD reduces along y instead (the
    minimum non-zero, which is also rule 5's choice), and so keeps [0 1 3]^(1)
    alive for this one child alone.

    No per-node rule can see that: it is a property of the graph built so far.
    The SIEVE, [JEngine, v2, Alg. 3], is the pass that recovers it -- it walks
    the built graph and re-expands a child in another direction when that lets
    an intermediate be deleted. Not a `StructuralLimit`: no published claim goes
    unpredicted here; a stateless rule simply cannot see the graph.
    """
    nodes, children = _mild_tree(5)
    target: Node = ((0, 2, 3), 0)
    assert target in nodes

    assert mild_direction(target[0]) == "y"
    common = _parents(target, "z")
    assert common == [((0, 2, 2), 1), ((0, 2, 1), 1)]
    assert all(p in nodes for p in common)

    # MILD's own choice holds a node in the graph that only this target uses.
    only_for_target = ((0, 1, 3), 1)
    assert only_for_target in _parents(target, "y")
    assert children[only_for_target] == {target}


def test_the_first_ignored_common_parents_are_at_l_5() -> None:
    # Below L = 5 no node keeps an intermediate alive for itself alone while
    # another direction's parents are all on the graph. Consistent with
    # mild.md's "MILD is non-optimal in node count for L >= 5", though this
    # pattern is narrower than node-count optimality and is all it checks.
    for L in range(1, 5):
        nodes, children = _mild_tree(L)
        for node in nodes:
            r, _ = node
            if not sum(r):
                continue
            mine = mild_direction(r)
            sole = [p for p in _parents(node, mine) if children[p] == {node}]
            if not sole:
                continue
            for other in AXES:
                if other == mine or not r[AXES.index(other)]:
                    continue
                assert not all(p in nodes for p in _parents(node, other)), (L, node)
