"""`hgp/hrr.py` (#274): the HGP horizontal recurrence as an `(a, b, c, d)` DAG.

Fast tier for `L <= FAST_L_MAX` (tiers.py), `slow` above: pure Python, no
toolchain, no card. The counts are what this tree computes under `hgp.hrr`'s
stated boundary set, side order and statement convention; the predecessor's
(a3) table is not on this box, so they are pinned from the code, with #274's
three printed `(dd|dd)` figures reconciled in
`test_dd_dd_the_issues_figures`.
"""

from __future__ import annotations

import math

import pytest
from tiers import split_by_l

from felizjueves.ao.eri import eri_block
from felizjueves.ao.shell import Shell
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.hrr import COEFS, ZERO, build, is_boundary
from felizjueves.hgp.vrr import CLASSES, cartesians, target_set
from felizjueves.reference.pq import pq

# (la, lb, lc, ld): (ERIs, boundary values, interior statements with the bra
# HRR first in build order -- the default -- and with the ket HRR first).
PINNED: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {
    (0, 0, 0, 0): (1, 1, 0, 0),
    (0, 0, 0, 1): (3, 4, 3, 3),
    (0, 0, 0, 2): (6, 10, 15, 15),
    (0, 0, 1, 1): (9, 9, 9, 9),
    (0, 0, 1, 2): (18, 19, 41, 41),
    (0, 0, 2, 2): (36, 31, 79, 79),
    (0, 1, 0, 1): (9, 16, 21, 21),
    (0, 1, 0, 2): (18, 40, 75, 78),
    (0, 1, 1, 1): (27, 36, 54, 63),
    (0, 1, 1, 2): (54, 76, 180, 218),
    (0, 1, 2, 2): (108, 124, 330, 424),
    (0, 2, 0, 2): (36, 100, 240, 240),
    (0, 2, 1, 1): (54, 90, 189, 225),
    (0, 2, 1, 2): (108, 190, 531, 680),
    (0, 2, 2, 2): (216, 310, 939, 1330),
    (1, 1, 1, 1): (81, 81, 162, 162),
    (1, 1, 1, 2): (162, 171, 540, 531),
    (1, 1, 2, 2): (324, 279, 990, 1035),
    (1, 2, 1, 2): (324, 361, 1517, 1517),
    (1, 2, 2, 2): (648, 589, 2693, 2977),
    (2, 2, 2, 2): (1296, 961, 5293, 5293),
}

LADDER = split_by_l(CLASSES, sum)


def test_the_pins_cover_the_ladder() -> None:
    assert set(PINNED) == set(CLASSES)


@pytest.mark.parametrize("cls", LADDER)
def test_counts(cls) -> None:
    n_eri, n_bd, n_bra, n_ket = PINNED[cls]
    for first, n_int in (("bra", n_bra), ("ket", n_ket)):
        dag = build(cls, first)
        assert dag.n_eri == n_eri == math.prod(len(cartesians(x)) for x in cls)
        assert dag.n_boundary == len(dag.boundary) == n_bd
        assert dag.n_interior == n_int
        assert dag.n_statements == n_bd + n_int


@pytest.mark.parametrize("cls", LADDER)
@pytest.mark.parametrize("first", hrr.SIDES)
def test_boundary_is_the_vrr_target_set(cls, first) -> None:
    # #274's acceptance: the set the HRR asks for IS the set the VRR builds.
    dag = build(cls, first)
    assert dag.boundary == target_set(cls)
    assert dag.boundary == {(e, f) for e, f, _ in vrr.targets(cls)}


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_dd_dd_the_issues_figures() -> None:
    # #274 prints 6,254 statements, 1,296 ERIs and 1,225 boundary values.
    # The first two hold; the third is the boundary with its lower bound
    # dropped (35^2; #273's comment), and the bounded 961 is what makes the
    # statement count come out at 6,254 under "every node is a statement".
    dag = build((2, 2, 2, 2))
    assert dag.n_eri == 1296 == 6 ** 4
    assert dag.n_statements == 6254 == 961 + 5293
    assert dag.n_boundary == 961 == 31 ** 2
    unbounded = sum(len(cartesians(n)) for n in range(5)) ** 2
    assert unbounded == 1225
    assert dag.n_interior + unbounded == 6518 != 6254
    assert build((2, 2, 2, 2), "ket").n_statements == 6254


def test_side_order_is_cheaper_or_equal_except_pp_pd() -> None:
    dearer = [c for c in CLASSES if PINNED[c][2] > PINNED[c][3]]
    cheaper = [c for c in CLASSES if PINNED[c][2] < PINNED[c][3]]
    assert dearer == [(1, 1, 1, 2)]
    assert len(cheaper) == 9


@pytest.mark.parametrize("cls", LADDER)
def test_no_node_carries_an_m(cls) -> None:
    # Structural: a node is four Cartesian 3-vectors and nothing else -- no
    # Boys index anywhere, on any node or child.
    dag = build(cls)
    for n, ts in dag.terms.items():
        for x in (n, *(t.child for t in ts)):
            assert isinstance(x, tuple) and len(x) == 4
            assert all(isinstance(v, tuple) and len(v) == 3
                       and all(type(i) is int and i >= 0 for i in v)
                       for v in x)
    assert set(hrr.Term.__dataclass_fields__) == {"coef", "child"}


@pytest.mark.parametrize("cls", LADDER)
@pytest.mark.parametrize("first", hrr.SIDES)
def test_dag_shape_and_order(cls, first) -> None:
    dag = build(cls, first)
    la, lb, lc, ld = cls
    pos = {n: i for i, n in enumerate(dag.order)}
    assert set(pos) == dag.nodes and len(pos) == len(dag.order)
    assert set(dag.targets) <= dag.nodes
    for n, ts in dag.terms.items():
        a, b, c, d = n
        # No term lowers a (or c) or raises |a| + |b| (|c| + |d|).
        assert la <= sum(a) <= sum(a) + sum(b) <= la + lb
        assert lc <= sum(c) <= sum(c) + sum(d) <= lc + ld
        assert (ts == ()) == is_boundary(n)
        if ts:
            assert len(ts) == 2 and ts[0].coef == "1"
            assert ts[1].coef in COEFS[1:]
        for t in ts:
            assert pos[t.child] < pos[n]  # topological
    # Every node is reached from a target: the closure keeps nothing dead.
    parents = {t.child for ts in dag.terms.values() for t in ts}
    assert dag.nodes == parents | set(dag.targets)
    n_bd = dag.n_boundary
    assert all(is_boundary(n) for n in dag.order[:n_bd])


def test_reduce_refuses_an_unknown_side_and_a_bad_class() -> None:
    with pytest.raises(ValueError):
        hrr.reduce(((0, 0, 0),) * 4, "middle")
    with pytest.raises(ValueError):
        build((2, 2, 2))
    assert hrr.reduce((ZERO,) * 4) == ()


# ---------------------------------------------------------------------------
# The numeric witness: eq 28 on CONTRACTED integrals, against ao.eri.
#
# Each shell has two primitives and the weights `_W`, so ao.eri returns the
# contracted integral of unnormalised primitives. The boundary `(e0|f0)` is
# the same quartet with A at l = |e| and B at l = 0 (B's exponents and
# weights), and likewise C, D. That B's s function keeps B's
# contraction is what makes eq 28 valid: AB is common to every primitive pair.
# ---------------------------------------------------------------------------

_CENTRES = ((0.0, 0.0, 0.0), (0.9, -0.4, 0.3), (-0.5, 0.7, 1.1),
            (0.4, 1.2, -0.6))
_EXPS = ((1.3, 0.4), (0.9, 0.25), (1.1, 0.35), (0.7, 0.3))
_W = (0.6, 0.5)


def _raw(i: int, l: int) -> Shell:  # noqa: E741
    # ao.shell's coefficients multiply UNNORMALISED primitives, so `_W` is
    # the contraction as given, the same at every l.
    return Shell(l, _CENTRES[i], _EXPS[i], _W)


def _block(la: int, lb: int, lc: int, ld: int):
    blk = eri_block((_raw(0, la), _raw(1, lb)), (_raw(2, lc), _raw(3, ld)),
                    pq=pq)
    out = {}
    nc, nd = len(cartesians(lc)), len(cartesians(ld))
    for i, (a, b) in enumerate((a, b) for a in cartesians(la)
                               for b in cartesians(lb)):
        for j, (c, d) in enumerate((c, d) for c in cartesians(lc)
                                   for d in cartesians(ld)):
            assert len(blk[i]) == nc * nd
            out[a, b, c, d] = blk[i][j]
    return out


def _evaluate(dag, boundary):
    ab = [p - q for p, q in zip(_CENTRES[0], _CENTRES[1], strict=True)]
    cd = [p - q for p, q in zip(_CENTRES[2], _CENTRES[3], strict=True)]
    coef = {"1": 1.0} | {f"AB_{k}": ab[i] for i, k in enumerate(hrr.AXES)} \
        | {f"CD_{k}": cd[i] for i, k in enumerate(hrr.AXES)}
    val = {}
    for n in dag.order:
        ts = dag.terms[n]
        val[n] = (boundary[n[0], n[2]] if not ts
                  else sum(coef[t.coef] * val[t.child] for t in ts))
    return val


@pytest.mark.parametrize("cls", LADDER)
@pytest.mark.parametrize("first", hrr.SIDES)
def test_hrr_on_contracted_values_matches_ao_eri(cls, first) -> None:
    la, lb, lc, ld = cls
    dag = build(cls, first)
    boundary = {}
    for le in range(la, la + lb + 1):
        for lf in range(lc, lc + ld + 1):
            for (e, _, f, _), v in _block(le, 0, lf, 0).items():
                boundary[e, f] = v
    assert set(boundary) == dag.boundary
    got = _evaluate(dag, boundary)
    want = _block(*cls)
    assert set(want) == set(dag.targets)
    # Rounding scale: the largest boundary magnitude times the HRR's growth
    # bound (1 + |AB|_inf)^lb (1 + |CD|_inf)^ld; 64 ulp of that.
    big = max(abs(v) for v in boundary.values())
    grow = (1 + 0.9) ** lb * (1 + 1.7) ** ld
    tol = 64 * 2.0 ** -52 * big * grow
    for n in dag.targets:
        assert abs(got[n] - want[n]) <= tol, (n, got[n], want[n])
