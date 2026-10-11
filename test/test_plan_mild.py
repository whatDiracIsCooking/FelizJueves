"""MILD, [JEngine, v2, Alg. 2]: the tie-break, Johnson's rules, and rule 4's absence.

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


# -- Johnson, Gill & Pople's rules ---------------------------------------------


def test_rule_1_the_axial_reduction_is_forced() -> None:
    # {0 0 j}: one non-zero component, so there is no choice to make -- MILD
    # reduces it because it is the only reducible component.
    for r in _every_r(L_MAX):
        if sorted(r)[:2] == [0, 0]:
            assert [AXES[i] for i, c in enumerate(r) if c] == [mild_direction(r)]


def test_rule_2_reduces_the_unit_component_of_an_axial_child() -> None:
    # {0 1 j}, j >= 2: reduce the unit component.
    seen = 0
    for r in _every_r(L_MAX):
        lo, mid, hi = sorted(r)
        if (lo, mid) == (0, 1) and hi >= 2:
            assert _reduced(r) == 1
            seen += 1
    assert seen


def test_rule_3_never_reduces_the_large_component() -> None:
    # {1 1 j}, j >= 2: never reduce the j. (Which unit component is free.)
    seen = 0
    for r in _every_r(L_MAX):
        lo, mid, hi = sorted(r)
        if (lo, mid) == (1, 1) and hi >= 2:
            assert _reduced(r) == 1
            seen += 1
    assert seen


def test_rule_5_reduces_the_two() -> None:
    # {0 2 j}, j >= 3: reduce the 2.
    seen = 0
    for r in _every_r(L_MAX):
        lo, mid, hi = sorted(r)
        if (lo, mid) == (0, 2) and hi >= 3:
            assert _reduced(r) == 2
            seen += 1
    assert seen


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
