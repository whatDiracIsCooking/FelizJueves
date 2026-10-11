"""Pipeline step 2: the intermediate-dependency DAG, per batch, under MILD.

[JEngine, v2, §6] step 1 splits the targets (`batching.partition`); step 2,
this module, builds each batch's DAG "based on the MILD heuristic"; step 3,
the sieve ([JEngine, v2, Alg. 3]), prunes it and is NOT applied here. So this
is the paper's algorithm minus its pruning pass, and every node's parents
follow from `mild_direction` with no exception.

THE EDGES are the recurrence in reduction form ([JEngine, v2, eqs. 10-12], as
rewritten in docs/implementation/j_engine.md):

    [r]^(m) = PQ_d . [r - 1d]^(m+1) + (r_d - 1) . [r - 2d]^(m+1)

with d = mild_direction(r). A node's parents are [r - 1d]^(m+1) and, when
r_d >= 2, [r - 2d]^(m+1). Recursion stops at [0]^(m), the fundamentals: they
are Boys-function output, leaves, and never reduced.

ONE DAG PER BATCH, and nothing is shared between them: each batch is its own
kernel, so an intermediate two batches need is a node in both DAGs and is
counted, and paid for, in both ([JEngine, v2, §5], item 3). That duplication
is what Table 1's FLOP column grows by.

A NODE IS AN INTERMEDIATE WHEN m > 0, FUNDAMENTALS INCLUDED. [0]^(m), m > 0,
has m > 0, and it is held live like any other intermediate even though the
Boys function computes it. That is the counting convention the literal board
fixes, and the evidence for it is in plan/cost.py's docstring.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from felizjueves.plan.batching import AXES, Batch, in_batch, partition
from felizjueves.plan.mild import mild_direction
from felizjueves.plan.targets import R, hermite_targets
from felizjueves.spec import Spec

Node = tuple[R, int]  # ([r], m)


def is_fundamental(node: Node) -> bool:
    """[0]^(m): Boys-function output, a leaf, never reduced."""
    return not any(node[0])


def mild_parents(node: Node) -> tuple[Node, ...]:
    """[r - 1d]^(m+1), then [r - 2d]^(m+1) if r_d >= 2, with d from MILD."""
    r, m = node
    i = AXES.index(mild_direction(r))
    one = tuple(c - (k == i) for k, c in enumerate(r))
    if r[i] < 2:
        return ((one, m + 1),)
    two = tuple(c - 2 * (k == i) for k, c in enumerate(r))
    return ((one, m + 1), (two, m + 1))


@dataclass(frozen=True, eq=False)
class BatchDag:
    """One batch's DAG. `parents` has a key for every non-fundamental node and
    for nothing else; a fundamental is a node only as somebody's parent, or as
    the target [0]^(0)."""

    batch: Batch
    targets: frozenset[Node]  # m == 0
    parents: Mapping[Node, tuple[Node, ...]]  # every non-fundamental node

    def nodes(self) -> frozenset[Node]:
        out = set(self.targets) | set(self.parents)
        for ps in self.parents.values():
            out.update(ps)
        return frozenset(out)

    def intermediates(self) -> frozenset[Node]:
        """Every node with m > 0, fundamentals [0]^(m) included."""
        return frozenset(n for n in self.nodes() if n[1] > 0)

    def reduced_component(self, node: Node) -> int:
        """r_d of a non-fundamental node: the component its edges reduce."""
        (r, _), (r1, _) = node, self.parents[node][0]
        return next(c for c, c1 in zip(r, r1, strict=True) if c != c1)


def batch_dag(total_am: int, batch: Batch) -> BatchDag:
    """The MILD DAG for the targets of one batch of the class L = total_am."""
    targets = frozenset(
        (r, 0) for r in hermite_targets(total_am) if in_batch(r, *batch)
    )
    parents: dict[Node, tuple[Node, ...]] = {}
    stack = [n for n in targets if not is_fundamental(n)]
    while stack:
        node = stack.pop()
        if node in parents:
            continue
        parents[node] = ps = mild_parents(node)
        stack.extend(p for p in ps if not is_fundamental(p) and p not in parents)
    return BatchDag(batch, targets, MappingProxyType(parents))


def build(spec: Spec) -> tuple[BatchDag, ...]:
    """One DAG per batch of `partition(spec.total_am, spec.n_batches)`, in the
    partition's order. Only the literal reconstruction exists: a Spec naming
    any lever is refused rather than silently built as if it named none."""
    if spec.levers:
        raise ValueError(f"no lever is implemented in the plan layer: {spec.levers}")
    L = spec.total_am
    return tuple(batch_dag(L, b) for b in partition(L, spec.n_batches))
