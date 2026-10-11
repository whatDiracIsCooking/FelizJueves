"""Pipeline step 3: the sieve, [JEngine, v2, Alg. 3], and its off switch.

After Gill, Head-Gordon & Pople (doi:10.1002/qua.560360831). It walks each
batch's MILD DAG (plan/dag.py) from the targets upward and deletes an
intermediate wherever every node formed from it can be re-expanded in another
direction out of nodes already present. That re-expansion is where a child
deviates from MILD, and it is what recovers Johnson's rule 4 (reduce to common
parents), which no per-node rule can keep. docs/implementation/mild.md is the
specification this follows; the line numbers below are Alg. 3's.

    line 1      I = every node MILD's DAG holds
    lines 4-5   SEED: for each target [r](0), in `hermite_targets` order (a
                FIXED order, so the output is deterministic and golden files
                are stable), enqueue its parents in the order dag.py stores
                them, [r - 1d] before [r - 2d]
    lines 6-10  VISIT breadth-first, skipping a node already processed and,
                here, a node a previous prune has already deleted
    lines 11-13 TEST: can every child j of i be re-expanded in some direction
                other than its current one, with every parent of j in
                I \\ {i}?
    lines 14-16 PRUNE: if so, re-parent every child, delete i, and sweep
                upward deleting every node left with no child
    line 18     ELSE: enqueue i's parents and continue

TWO CHOICES THE PAPER LEAVES OPEN, fixed here.

1. WHICH OTHER DIRECTION, when several qualify: the one with the smallest
   reduced component r_d' (the cheapest, by plan/cost.py's 1/2/3 FLOPs, and
   with r_d' = 1 the one with a single parent), ties by x, y, z order. This is
   MILD's own criterion restricted to the directions the graph allows.
2. THE UPWARD SWEEP deletes every node left with no child and not a target --
   i's parents, as Alg. 3 says, and also a re-parented child's OTHER old
   parent ([r - 2d] when i was [r - 1d], or the reverse), which the
   re-parenting can orphan just as surely. Anything less leaves a node in the
   DAG that nothing needs.

A deleted node is never re-admitted: the test only ever re-parents onto nodes
already in I, so the sieve can only shrink a DAG. A target is never deleted:
it is m = 0 and the queue holds only parents, which are m >= 1.

THE SWITCH. `no-sieve` is a lever ([JEngine, v2, §4.1] measures the sieve as
non-monotone in spills, so for a given class and card either may win; see
mild.md, "What the sieve actually buys"). `plan(spec)` is the plan layer's
entry point: it builds MILD's DAGs and sieves them, unless the Spec names
`no-sieve`. `levers=()` is the literal reconstruction, the paper's algorithm
as written, sieve on. dag.build() refuses every lever, so this module strips
the one it implements before calling it, and any other lever still reaches
build() and is still refused there.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from types import MappingProxyType

from felizjueves.plan.batching import AXES
from felizjueves.plan.dag import BatchDag, Node, build
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Spec

NO_SIEVE = "no-sieve"


def parents_along(node: Node, axis: int) -> tuple[Node, ...]:
    """[r - 1_axis]^(m+1), then [r - 2_axis]^(m+1) if r_axis >= 2."""
    r, m = node
    if r[axis] < 1:
        raise ValueError(f"{node}: component {AXES[axis]} is 0 and not reducible")
    one = tuple(c - (k == axis) for k, c in enumerate(r))
    if r[axis] < 2:
        return ((one, m + 1),)
    two = tuple(c - 2 * (k == axis) for k, c in enumerate(r))
    return ((one, m + 1), (two, m + 1))


def _axis(node: Node, parents: tuple[Node, ...]) -> int:
    """The axis a node is currently reduced along, read off its first parent."""
    (r, _), (r1, _) = node, parents[0]
    return next(k for k, (c, c1) in enumerate(zip(r, r1, strict=True)) if c != c1)


def _reexpansion(
    node: Node, current: tuple[Node, ...], present: set[Node], gone: Node
) -> tuple[Node, ...] | None:
    """`node`'s parents along another direction, all in present \\ {gone};
    None if no direction qualifies. Choice 1 of the module docstring."""
    here = _axis(node, current)
    r = node[0]
    for axis in sorted((k for k in range(3) if k != here and r[k]),
                       key=lambda k: (r[k], k)):
        ps = parents_along(node, axis)
        if all(p in present and p != gone for p in ps):
            return ps
    return None


def sieve(dag: BatchDag) -> BatchDag:
    """[JEngine, v2, Alg. 3] on one batch's DAG. Pure: `dag` is not changed."""
    parents: dict[Node, tuple[Node, ...]] = dict(dag.parents)
    present: set[Node] = set(dag.nodes())
    children: dict[Node, set[Node]] = {n: set() for n in present}
    for n, ps in parents.items():
        for p in ps:
            children[p].add(n)

    def unlink(child: Node) -> None:
        for p in parents[child]:
            children[p].discard(child)

    def sweep(start: list[Node]) -> None:
        """Delete every non-target node left with no child, upward."""
        stack = list(start)
        while stack:
            n = stack.pop()
            if n not in present or children[n] or n in dag.targets:
                continue
            present.discard(n)
            if n in parents:
                stack.extend(parents[n])
                unlink(n)
                del parents[n]
            del children[n]

    queue: deque[Node] = deque()
    for t in _target_order(dag):  # lines 4-5
        queue.extend(parents.get(t, ()))
    seen: set[Node] = set()
    while queue:  # lines 6-10
        i = queue.popleft()
        if i in seen or i not in present:
            continue
        seen.add(i)
        new = {}
        for j in sorted(children[i]):  # lines 11-13
            ps = _reexpansion(j, parents[j], present, i)
            if ps is None:
                break
            new[j] = ps
        else:
            if children[i]:  # lines 14-16
                orphans: list[Node] = [i]
                for j, ps in new.items():
                    orphans.extend(parents[j])
                    unlink(j)
                    parents[j] = ps
                    for p in ps:
                        children[p].add(j)
                sweep(orphans)
                continue
        queue.extend(parents.get(i, ()))  # line 18

    return replace(dag, parents=MappingProxyType(parents))


def _target_order(dag: BatchDag) -> list[Node]:
    """The batch's targets in `hermite_targets` order: the fixed seed order."""
    top = max(sum(r) for r, _ in dag.targets)
    return [(r, 0) for r in hermite_targets(top) if (r, 0) in dag.targets]


def plan(spec: Spec) -> tuple[BatchDag, ...]:
    """One DAG per batch: MILD's, sieved unless the Spec names `no-sieve`."""
    rest = tuple(lever for lever in spec.levers if lever != NO_SIEVE)
    dags = build(replace(spec, levers=rest))
    if NO_SIEVE in spec.levers:
        return dags
    return tuple(sieve(d) for d in dags)
