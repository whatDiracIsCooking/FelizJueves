"""Pipeline step 4: emit the [p|q] evaluation as a topological sort of the
(pruned) DAG ([JEngine, v2, §6]).

This is the bridge from the plan layer's counts to a kernel. It produces an
ORDER, not text: a linear order of one batch's nodes in which every node
follows its parents, so an emitter walking it never reads an intermediate
before it is formed.

THIS IS NOT THE TARGET-VISITING ORDER. `dag.build(..., targets=...)` and the
sieve's `seed_order` are orders in which targets are *visited* while a DAG is
built or pruned; this module produces the order in which nodes are *emitted*
from a DAG already built. The two are independent.

THE DAG IS AN ARGUMENT. `schedule` takes a `BatchDag` and works on whatever it
is given -- MILD's DAG from dag.py, or the sieve's pruned one -- and does not
reach for the sieve itself.

WHY DESCENDING m IS FREE. Every edge of this recurrence raises m by exactly
one ([r](m) is formed from [r-1d](m+1) and [r-2d](m+1)), MILD's edges and the
sieve's re-parented ones alike (sieve.parents_along). A parent therefore
always has a strictly larger m than its child, and sorting by descending m is
already a topological order; no graph walk is needed.

THE TIE-BREAK KEY of `DESCENDING_M`, stated so golden files are stable: nodes
are sorted ascending by

    (-m, r_x + r_y + r_z, r_x, r_y, r_z)

i.e. descending m; within one m, ascending total Hermite order |r| (so the
fundamental [0](m) comes first); within one |r|, lexicographically ascending
(r_x, r_y, r_z). Nodes of one m share no edge, so any tie-break is valid; this
one is fixed and depends only on the node set, never on the DAG's insertion
order.

`strategy` IS A SEAM, the same shape as batching's `cut`: a `Strategy` maps a
DAG to an order, and `schedule` checks that what it returns is a topological
permutation of the DAG's nodes. Nobody passes one today. The order decides
live ranges, and therefore what nvcc and hipcc each do with their registers;
an order that spills less is a lever, and a lever needs a measurement to
justify it, so no optimisation lives here.
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.plan.dag import BatchDag, Node

Order = tuple[Node, ...]
Strategy = Callable[[BatchDag], Order]


def tie_break_key(node: Node) -> tuple[int, int, int, int, int]:
    """(-m, |r|, r_x, r_y, r_z): DESCENDING_M's sort key, ascending."""
    (rx, ry, rz), m = node
    return (-m, rx + ry + rz, rx, ry, rz)


def descending_m(dag: BatchDag) -> Order:
    """Every node of `dag`, sorted by `tie_break_key`."""
    return tuple(sorted(dag.nodes(), key=tie_break_key))


DESCENDING_M: Strategy = descending_m


def check(dag: BatchDag, order: Order) -> None:
    """Raise ValueError unless `order` is a permutation of the DAG's nodes in
    which every node follows its parents."""
    nodes = dag.nodes()
    pos = {n: i for i, n in enumerate(order)}
    if len(pos) != len(order) or pos.keys() != nodes:
        raise ValueError("schedule is not a permutation of the DAG's nodes")
    for node, ps in dag.parents.items():
        for p in ps:
            if pos[p] >= pos[node]:
                raise ValueError(f"schedule puts {node} before its parent {p}")


def schedule(dag: BatchDag, *, strategy: Strategy = DESCENDING_M) -> Order:
    """A linear order of `dag`'s nodes in which every node follows its parents.

    The default, `DESCENDING_M`, sorts by (-m, |r|, r_x, r_y, r_z) ascending:
    descending m, then ascending |r|, then lexicographic r. `strategy` is a
    seam, not a parameter anyone passes today; whatever it returns is checked
    to be a topological permutation of the DAG's nodes.
    """
    order = tuple(strategy(dag))
    check(dag, order)
    return order
