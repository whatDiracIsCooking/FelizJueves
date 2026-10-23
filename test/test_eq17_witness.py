"""`[JEngine, v2, eq. 17]` misprints `1y` as `(1,1,0)`; the generator uses `(0,1,0)`.

`recurrence.reduction` spells eq. 17's reduction form,

    [r]^(m) = PQ_d [r - 1d]^(m+1) + (r_d - 1) [r - 2d]^(m+1),

with the shift `1d` applied as the direction's unit vector. This file holds
that shift to the corrected unit vectors, transcribed here as data, on every
node a real plan reduces along y with `r_y >= 1` AND `r_x >= 1` -- the nodes
where `(1,1,0)` and `(0,1,0)` name different (and both in-range) parents, so the
misprint is reachable and the test must break under it. A node with `r_x = 0`
would not do: `r - (1,1,0)` there has a negative component and could only fail
by accident, which is not a witness.

The comparison is per term, against the IR the kernel is built from: the
`PQ_y` term must read `[r - 1y]^(m+1)` and the `(r_y - 1)` term
`[r - 2y]^(m+1)`. A check on a total (a node count, a sum of components) would
not see an x component smuggled into the shift.
"""

from __future__ import annotations

import pytest
from tiers import FAST_L_MAX

from felizjueves.ir import Factor, Term, Var
from felizjueves.mmd.kernel import pq_spec
from felizjueves.mmd.recurrence import node_name, reduction
from felizjueves.plan.dag import BatchDag, Node
from felizjueves.plan.pipeline import plan
from felizjueves.plan.sieve import NO_SIEVE

# eq. 17's unit vectors, corrected (docs/implementation/j_engine.md, "Two
# defects in the source"). The printed 1y is (1,1,0).
ONE_Y = (0, 1, 0)
Y = 1


def _minus(r: tuple[int, ...], k: int, unit: tuple[int, int, int]) -> tuple[int, ...]:
    return tuple(c - k * u for c, u in zip(r, unit, strict=True))


def _y_nodes_where_the_misprint_bites(dag: BatchDag) -> list[Node]:
    """Non-fundamental nodes the DAG reduces along y, with r_y >= 1, r_x >= 1."""
    out = []
    for node in dag.parents:
        (r, _), (r1, _) = node, dag.parents[node][0]
        reduced_along_y = r[Y] != r1[Y]
        if reduced_along_y and r[0] >= 1 and r[Y] >= 1:
            out.append(node)
    return sorted(out)


def _witness(dag: BatchDag, node: Node) -> None:
    (r, m) = node
    terms = reduction(dag, node).terms
    want = [Term(1, Factor.PQ_Y, Var(node_name((_minus(r, 1, ONE_Y), m + 1))))]
    if r[Y] >= 2:
        want.append(
            Term(r[Y] - 1, Factor.ONE, Var(node_name((_minus(r, 2, ONE_Y), m + 1))))
        )
    assert list(terms) == want, (dag.batch, node, terms)


def _witness_plans(levers: tuple[str, ...], Ls: range, two_term_L: int) -> None:
    seen = 0
    for L in Ls:
        for n_batches in (1, 3):
            for dag in plan(pq_spec(L, levers, n_batches)).dags:
                for node in _y_nodes_where_the_misprint_bites(dag):
                    _witness(dag, node)
                    seen += 1
    # Not vacuous: the misprint is reachable on these plans, for both terms.
    assert seen > 0
    two_term = [
        n
        for L in (two_term_L,)
        for dag in plan(pq_spec(L, levers)).dags
        for n in _y_nodes_where_the_misprint_bites(dag)
        if n[0][Y] >= 2
    ]
    assert two_term, "no y-reduced node with r_y >= 2 and r_x >= 1 was reached"


# The plans at L = 2..8 are split at test/tiers.py's cut: the fast half
# witnesses L = 2..4 (both terms reached at L = 4), the `slow` half L = 5..8
# (both terms reached at L = 8). The claim marks the fast half, the test
# claims.json names; test_claims.py refuses one id on two tests.
@pytest.mark.claim("jengine/eq17-1y-misprint")
@pytest.mark.parametrize("levers", [(), (NO_SIEVE,)], ids=["sieved", "no-sieve"])
def test_y_reduction_shifts_by_the_unit_vector_0_1_0(
    levers: tuple[str, ...],
) -> None:
    _witness_plans(levers, range(2, FAST_L_MAX + 1), 4)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("levers", [(), (NO_SIEVE,)], ids=["sieved", "no-sieve"])
def test_y_reduction_shifts_by_the_unit_vector_0_1_0_above_l4(
    levers: tuple[str, ...],
) -> None:
    _witness_plans(levers, range(FAST_L_MAX + 1, 9), 8)
