"""`hgp/vrr.py` (#273): the HGP vertical recurrence as an `(e, f, m)` DAG.

Fast tier: pure Python, no toolchain, no card. The counts below are what this
tree computes under `hgp.vrr`'s stated target set, reduction direction and
pruning; the predecessor's (a3) table is not on this box, so they are pinned
from the code, with the one published figure reconciled in
`test_dd_dd_has_961_targets_and_the_issues_1225_is_the_unbounded_count`.
"""

from __future__ import annotations

from math import comb

import pytest

from felizjueves.ao.shell import cartesians as ao_cartesians
from felizjueves.hgp import vrr
from felizjueves.hgp.vrr import CLASSES, ZERO, build, reachable, target_set
from felizjueves.plan.dag import mild_parents

# (la, lb, lc, ld): (targets, nodes, edges, nodes at m = 0) of the PRUNED DAG
# (the default), then (nodes, edges, nodes at m = 0) of the forward build.
PINNED: dict[tuple[int, int, int, int], tuple[int, ...]] = {
    (0, 0, 0, 0): (1, 1, 0, 1, 1, 0, 1),
    (0, 0, 0, 1): (4, 5, 6, 4, 5, 6, 4),
    (0, 0, 0, 2): (10, 15, 30, 10, 15, 30, 10),
    (0, 0, 1, 1): (9, 15, 30, 10, 15, 30, 10),
    (0, 0, 1, 2): (19, 35, 86, 20, 35, 86, 20),
    (0, 0, 2, 2): (31, 70, 192, 35, 70, 192, 35),
    (0, 1, 0, 1): (16, 21, 39, 16, 24, 45, 16),
    (0, 1, 0, 2): (40, 61, 150, 40, 70, 174, 40),
    (0, 1, 1, 1): (36, 61, 150, 40, 70, 174, 40),
    (0, 1, 1, 2): (76, 141, 399, 80, 160, 455, 80),
    (0, 1, 2, 2): (124, 281, 863, 140, 315, 969, 140),
    (0, 2, 0, 2): (100, 155, 426, 100, 200, 555, 100),
    (0, 2, 1, 1): (90, 155, 426, 100, 200, 555, 100),
    (0, 2, 1, 2): (190, 355, 1086, 200, 450, 1385, 200),
    (0, 2, 2, 2): (310, 705, 2306, 350, 875, 2870, 350),
    (1, 1, 1, 1): (81, 146, 402, 91, 200, 555, 100),
    (1, 1, 1, 2): (171, 336, 1030, 181, 450, 1385, 200),
    (1, 1, 2, 2): (279, 671, 2200, 316, 875, 2870, 350),
    (1, 2, 1, 2): (361, 696, 2228, 381, 1000, 3219, 400),
    (1, 2, 2, 2): (589, 1381, 4694, 666, 1925, 6560, 700),
    (2, 2, 2, 2): (961, 2330, 8122, 1089, 3675, 12856, 1225),
}


def _ncart(lo: int, hi: int) -> int:
    return sum(comb(n + 2, 2) for n in range(lo, hi + 1))


def test_the_ladder_is_the_21_canonical_classes_with_every_l_at_most_2() -> None:
    every = {(a, b, c, d) for a in range(3) for b in range(3)
             for c in range(3) for d in range(3)}
    canon = {k for k in every
             if k[0] <= k[1] and k[2] <= k[3] and k[:2] <= k[2:]}
    assert len(CLASSES) == len(set(CLASSES)) == 21
    assert set(CLASSES) == canon
    assert set(PINNED) == canon
    assert CLASSES[-1] == (2, 2, 2, 2)


def test_cartesians_restate_the_ao_layer() -> None:
    for l in range(9):  # noqa: E741
        assert vrr.cartesians(l) == ao_cartesians(l)


@pytest.mark.parametrize("cls", CLASSES)
def test_counts_are_pinned(cls) -> None:
    dag, fwd = build(cls), build(cls, prune=False)
    got = (len(dag.targets), dag.n_nodes, dag.n_edges, len(dag.at_m(0)),
           fwd.n_nodes, fwd.n_edges, len(fwd.at_m(0)))
    assert got == PINNED[cls]
    assert dag.pruned and not fwd.pruned


@pytest.mark.parametrize("cls", CLASSES)
def test_target_set_is_the_bounded_boundary_set(cls) -> None:
    la, lb, lc, ld = cls
    ts = target_set(cls)
    assert len(ts) == _ncart(la, la + lb) * _ncart(lc, lc + ld)
    assert all(la <= sum(e) <= la + lb and lc <= sum(f) <= lc + ld
               for e, f in ts)
    dag = build(cls)
    assert {(e, f) for e, f, _ in dag.targets} == ts
    assert all(m == 0 for *_, m in dag.targets)
    assert len(dag.targets) == len(ts)


def test_dd_dd_has_961_targets_and_the_issues_1225_is_the_unbounded_count() -> None:
    # #273 and #274 print "1,225" targets / boundary values for (dd|dd), and
    # both define the set as la <= |e| <= la+lb, lc <= |f| <= lc+ld. That
    # definition gives (6 + 10 + 15)^2 = 31^2 = 961: eq 28's HRR never reads
    # an (e0|f0) with |e| < |a|. 1,225 = 35^2 is the same count with the
    # lower bound dropped (|e|, |f| in 0..4) -- which is also the forward
    # build's m = 0 layer -- so the printed figure is read as a misprint and
    # the bounded set is pinned (orchestrator decision on #273).
    dd = (2, 2, 2, 2)
    assert len(target_set(dd)) == 961 == 31 ** 2
    unbounded = {(e, f) for e in vrr._range(0, 4) for f in vrr._range(0, 4)}
    assert len(unbounded) == 1225 == 35 ** 2
    assert len(build(dd, prune=False).at_m(0)) == 1225
    # After pruning, m = 0 keeps 31 x 35 ket-phase nodes plus the 4 bra
    # nodes with |e| < 2: 1,089.
    assert len(build(dd).at_m(0)) == 1089 == 31 * 35 + 4


@pytest.mark.parametrize("cls", CLASSES)
def test_the_dag_is_closed_and_ends_in_boys_leaves(cls) -> None:
    for prune in (True, False):
        dag = build(cls, prune=prune)
        for n, ts in dag.terms.items():
            assert all(t.child in dag.terms for t in ts)
            assert (ts == ()) == vrr.is_leaf(n)
            assert len({t.child for t in ts}) == len(ts)  # one edge per term
        leaves = {n for n, ts in dag.terms.items() if not ts}
        assert {m for *_, m in leaves} == set(range(sum(cls) + 1))


def test_the_dag_is_not_graded_by_m() -> None:
    # MMD (plan/dag.py): every edge raises m by exactly one ...
    r, m = (2, 1, 1), 0
    todo = [(r, m)]
    while todo:
        node = todo.pop()
        for child in mild_parents(node) if any(node[0]) else ():
            assert child[1] == node[1] + 1
            todo.append(child)
    # ... HGP: same-m edges exist (the PA/QC terms), and so do +1 edges.
    for cls in CLASSES[1:]:
        dag = build(cls)
        steps = {c[2] - p[2] for p, c in dag.edges()}
        assert steps == {0, 1}, cls
    dd = build((2, 2, 2, 2))
    same = [(p, t) for p, ts in dd.terms.items() for t in ts
            if t.child[2] == p[2]]
    assert {t.coef for _, t in same} == {"PA", "QC", "inv2p", "inv2q"}
    assert ((((1, 0, 0), ZERO, 0)), vrr.Term("PA", 1, (ZERO, ZERO, 0))) in same


@pytest.mark.parametrize("cls", CLASSES)
def test_the_build_order_is_bra_first_and_topological(cls) -> None:
    dag = build(cls)
    assert set(dag.order) == dag.nodes and len(dag.order) == dag.n_nodes
    pos = {n: i for i, n in enumerate(dag.order)}
    for p, c in dag.edges():
        assert pos[c] < pos[p]
    phase = [n[1] != ZERO for n in dag.order]
    assert phase == sorted(phase)  # every bra node precedes every ket node
    # Bra-phase nodes use eq 25 with f = 0 (no ket coupling term); ket-phase
    # nodes use eq 26.
    for n, ts in dag.terms.items():
        names = {t.coef for t in ts}
        assert names <= set(vrr.KET if n[1] != ZERO else vrr.BRA)


def test_the_ket_recurrence_reads_lower_bra_indices() -> None:
    # The paper's reason for the order: eq 26's e_d/(2(p+q)) term reads
    # [e-1_d, 0 | f, 0]^(m+1), a lower bra, so the bra must be built first.
    dag = build((2, 2, 2, 2))
    coupling = [(p, t.child) for p, ts in dag.terms.items() for t in ts
                if t.coef == "inv2pq"]
    assert coupling
    for p, c in coupling:
        assert sum(c[0]) == sum(p[0]) - 1 and c[2] == p[2] + 1
    assert any(c[1] == ZERO for _, c in coupling)  # reaches the bra phase


def test_terms_follow_eqs_25_and_26() -> None:
    # [(2,0,0),0|(1,1,0),0]^(0): eq 26 along x on f.
    node = ((2, 0, 0), (1, 1, 0), 0)
    assert vrr.reduce(node) == (
        vrr.Term("QC", 1, ((2, 0, 0), (0, 1, 0), 0)),
        vrr.Term("WQ", 1, ((2, 0, 0), (0, 1, 0), 1)),
        vrr.Term("inv2pq", 2, ((1, 0, 0), (0, 1, 0), 1)),
    )
    # [(0,3,0),0|0,0]^(1): eq 25 along y, with the e_d/(2p) pair.
    node = ((0, 3, 0), ZERO, 1)
    assert vrr.reduce(node) == (
        vrr.Term("PA", 1, ((0, 2, 0), ZERO, 1)),
        vrr.Term("WP", 1, ((0, 2, 0), ZERO, 2)),
        vrr.Term("inv2p", 2, ((0, 1, 0), ZERO, 1)),
        vrr.Term("inv2p*a_over_p", -2, ((0, 1, 0), ZERO, 2)),
    )
    assert vrr.reduce((ZERO, ZERO, 3)) == ()


@pytest.mark.parametrize("cls", CLASSES)
def test_pruning_removes_only_nodes_no_target_reaches(cls) -> None:
    fwd, dag = build(cls, prune=False), build(cls)
    assert dag.nodes <= fwd.nodes
    for t in dag.targets:
        # Each target's reachable set, rebuilt from the pruned DAG, equals
        # the one in the forward build: pruning cut nothing a target needs.
        assert reachable(dag.terms, [t]) == reachable(fwd.terms, [t])
    # And every removed node is one no target reaches.
    union = reachable(fwd.terms, dag.targets)
    assert dag.nodes == union
    assert not (fwd.nodes - dag.nodes) & union
    assert all(dag.terms[n] == fwd.terms[n] for n in dag.nodes)


def test_pruning_is_on_by_default_and_removes_something_at_dd_dd() -> None:
    dd = (2, 2, 2, 2)
    assert build(dd).pruned
    assert build(dd).n_nodes < build(dd, prune=False).n_nodes


def test_bad_classes_are_refused() -> None:
    for bad in ((0, 0, 0), (0, 0, 0, -1), (0, 0, 0, 1.0)):
        with pytest.raises(ValueError):
            build(bad)
