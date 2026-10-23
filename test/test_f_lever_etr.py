"""The ETR as a second VRR, `hgp/etr.py` + `f_levers/etr.py` (#354), on the host.

What is asserted:

1. THE DECLARATION: `etr` is legal on `Target.F` only, refused on J and K
   with its own reasons, keyed as any lever is, registered in
   `f_levers.REGISTRY`; on a class the rule gives the ETR it moves `vrr`,
   `ket_prim` and `quartet_scalars` only, on a cascade class nothing; the
   literal F kernels are untouched (their digests are test_f_seams.py's and
   test_emit_fock.py's pins).
2. THE ETR DAG (`hgp.etr`, additive): the cascade's targets exactly; every
   bra node eq 25 verbatim (`hgp.vrr.reduce`), every ket node the transfer
   at `m = 0` with its children the shifts the relation names; the order
   topological; the forward build closed under `reduce` and the pruned DAG
   the targets' closure; node/edge counts pinned for the 21 classes (pure
   Python, no kernel built above L = 4); `hgp.vrr.build` unchanged.
3. THE COST (`hgp.cost.etr_cost`, additive): the triple pinned for the 21
   classes, `(ss|ss)` equal to the cascade's, the new steps only in
   `ETR_GEOMETRY`, and `data/hgp/cost.json` untouched (test_hgp_cost.py).
4. THE ARBITRATION: the rule recomputed from its definition and pinned for
   the 21 classes (no build), `OVERRIDES` winning, `forced` arms, and
   `data/hgp/etr_arbitration.json` equal to `render()`. THE ACCURACY GATE:
   its tolerance the F `TOL`; its record recomputed per class with L <= 4
   (`slow` but for `(ss|dd)`: cc-pVDZ is not a small input) by
   interpreting the per-class device cells' own launches (H2O/cc-pVDZ,
   the ETR forced) against the restricted sweep -- the inputs the first
   interpreted tests missed and #354's device run failed on; the two
   refused classes, `(ss|dd)` and `(sp|pd)`, pinned as the finding.
5. THE SAME INTEGRALS, ANOTHER PATH (the issue's "the ETR DAG reaches the
   same ERIs as the ket cascade on the host"): the standalone ETR VRR
   kernel interpreted per primitive quartet with `hgp.recurrence`'s HRR on
   the contracted sums, against `ao/eri.py`'s MMD integrals and against
   libcint (the tracked fixtures), test_hgp_eri.py's quartets, scale and
   `1e-12` tolerance unchanged, every class with L <= 4.
6. THE KERNEL: its VRR is exactly the ETR closure (no eq 26 node, no
   cascade-only scalar declared), each transfer scalar declared once; the
   tap `g` against `ao/eri.py` to test_kernel_f's `TOL` of scale on every
   class with L <= 4 under the shipped choice AND with the ETR forced
   everywhere (both recurrences on every class); `F` against `ao.fock`'s
   sweep (one orbit per class, the degeneracy cases within the cap) to
   `TOL`; whole molecules (H2/STO-3G, and, `slow`, H2O/STO-3G and
   H2O/6-31G, every class L <= 4) through `walk_f` against the sweep and
   libcint's `vj - vk/2`.
   Not bitwise against the literal kernel: a different recurrence.
7. COMPOSITION: with `batched-vrr` (#350) the ETR kernel's `F` and tap bit
   for bit; with `sliced-hrr` (#356) on it, the sweep's to TOL; with
   `subdag-vrr` (#357) and `fmixed` (#355) refused, with the reasons.
8. THE GOLDENS: `(sp|sp)` with the lever, one unit per backend.

Fast tier: no compiler, no card, no PySCF. Nothing above L = 4 is built or
run there (milestone 13's cap); the DAG, cost and arbitration checks over the
whole ladder run their L > FAST_L_MAX classes (tiers.py) in `slow` twins.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_kernel_f as tk
from oracle import basis as ob
from oracle import manifest
from test_fock_build import BOUND, LIBCINT_TOL, TOL, molecule, run, sweep
from tiers import is_fast, split_by_l

from felizjueves import f_levers, levers, walk_f
from felizjueves.ao import hgp_drive
from felizjueves.ao.eri import eri_block_and_scale as mmd_eri
from felizjueves.ao.shell import Basis
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import etr as E
from felizjueves.f_levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.hgp import cost, etr, hrr, vrr
from felizjueves.hgp import recurrence as R
from felizjueves.ir import Decl, Note, walk
from felizjueves.ir.interp import Interpreter
from felizjueves.kernel_f import build_f_kernel, close, f_class_spec
from felizjueves.launch_f import f_launches
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

NAME = E.NAME
CAP = 4  # milestone 13: every test at L <= 4
CAPPED = [c for c in vrr.CLASSES if sum(c) <= CAP]
ETR_CAPPED = [c for c in CAPPED if E.choice(c) == E.ETR]
FAST = [c for c in vrr.CLASSES if is_fast(sum(c))]
ABOVE = [c for c in vrr.CLASSES if not is_fast(sum(c))]
ROOT = Path(__file__).resolve().parents[1]


def lever_spec(cls, more=()) -> Spec:
    return f_class_spec(*cls, tuple(sorted((NAME, *more))))


def _forced(cls, eris=True):
    spec = lever_spec(cls)
    return build_f_kernel(spec, eris=eris,
                          hooks=E.forced(E.ETR)(DEFAULT_HOOKS, spec))


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_etr_is_declared_legal_on_f_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[NAME]
    assert (lever.name, lever.issue, lever.legal) == (NAME, 354,
                                                      frozenset({"f"}))
    assert "Hermite" in lever.refusals["j"]
    assert "Hermite" in lever.refusals["k"]
    for slot in ("dq", "jab", "s", "t", "v", "schwarz"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="no bra/ket"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (NAME,))
    with pytest.raises(ValueError, match="no bra/ket"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))


def _moved(spec) -> set[str]:
    return {f.name for f in dataclasses.fields(FHooks)
            if not f_levers.hooks_for(spec).is_default(f.name)}


def test_etr_is_registered_keyed_and_moves_three_hooks_where_chosen() -> None:
    assert f_levers.REGISTRY[NAME] is E.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == "coulomb/d0/(0,1|0,1)/b1/f/etr"
    assert spec.kernel_body_levers() == (NAME,)
    assert E.choice((0, 1, 0, 1)) == E.ETR
    assert _moved(spec) == {"vrr", "ket_prim", "quartet_scalars"}
    # a cascade class: the hooks are the literal ones, the kernel the
    # literal kernel plus the provenance Note
    cas = lever_spec((0, 0, 1, 1))
    assert E.choice((0, 0, 1, 1)) == E.CASCADE
    assert f_levers.hooks_for(cas) is DEFAULT_HOOKS
    lit = build_f_kernel((0, 0, 1, 1))
    kern = build_f_kernel(cas)
    prov = Note(levers.provenance(NAME))
    assert kern.body == (lit.body[0], prov, *lit.body[1:])
    assert levers.provenance(NAME) in [s.prov for s in walk(
        build_f_kernel(spec).body) if isinstance(s, Note)]


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_literal_kernel_is_untouched(cls) -> None:
    kern = build_f_kernel(cls, eris=True)
    assert not any(isinstance(s, Decl) and s.name.startswith("etr_")
                   for s in walk(kern.body))


# ---------------------------------------------------------------------------
# 2. The ETR DAG
# ---------------------------------------------------------------------------

# class -> (pruned nodes, pruned edges, forward nodes): hgp.etr.build's
# counts, pinned so a change to the direction, the reduction or the pruning
# is a visible diff (pure Python; no kernel is built for these).
COUNTS = {
    (0, 0, 0, 0): (1, 0, 1),
    (0, 0, 0, 1): (8, 12, 8),
    (0, 0, 0, 2): (30, 66, 33),
    (0, 0, 1, 1): (30, 66, 33),
    (0, 0, 1, 2): (80, 206, 99),
    (0, 0, 2, 2): (175, 492, 245),
    (0, 1, 0, 1): (27, 57, 27),
    (0, 1, 0, 2): (85, 216, 89),
    (0, 1, 1, 1): (85, 216, 89),
    (0, 1, 1, 2): (205, 572, 230),
    (0, 1, 2, 2): (420, 1243, 511),
    (0, 2, 0, 2): (185, 506, 190),
    (0, 2, 1, 1): (185, 506, 190),
    (0, 2, 1, 2): (420, 1228, 451),
    (0, 2, 2, 2): (826, 2526, 938),
    (1, 1, 1, 1): (179, 491, 190),
    (1, 1, 1, 2): (410, 1202, 451),
    (1, 1, 2, 2): (811, 2486, 938),
    (1, 2, 1, 2): (741, 2240, 788),
    (1, 2, 2, 2): (1420, 4452, 1568),
    (2, 2, 2, 2): (2220, 7095, 2444),
}


def _counts_are_the_dags(classes) -> None:
    for cls in classes:
        n, e, fwd = COUNTS[cls]
        d = etr.build(cls)
        assert (d.n_nodes, d.n_edges, etr.build(cls, prune=False).n_nodes
                ) == (n, e, fwd), cls


def test_the_counts_cover_the_ladder_and_are_the_dags() -> None:
    assert tuple(COUNTS) == vrr.CLASSES
    _counts_are_the_dags(FAST)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_counts_are_the_dags_above_l4() -> None:
    _counts_are_the_dags(ABOVE)


@pytest.mark.parametrize("cls", split_by_l(vrr.CLASSES, sum, ids=str))
def test_the_etr_dag_has_the_cascades_targets_and_the_relation(cls) -> None:
    d, c = etr.build(cls), vrr.build(cls)
    assert d.targets == c.targets == etr.targets(cls)
    la, lb, lc, ld = cls
    L = sum(cls)
    seen: set = set()
    for n in d.order:
        e, f, m = n
        ts = d.terms[n]
        assert all(t.child in seen for t in ts), n  # topological
        seen.add(n)
        if vrr.is_leaf(n):
            assert ts == () and m <= L
        elif not etr.is_transfer(n):
            assert ts == vrr.reduce(n)  # eq 25, the cascade's bra phase
            assert sum(e) <= L - m
        else:
            assert m == 0 and 1 <= sum(f) <= lc + ld
            k = next(i for i, x in enumerate(f) if x)
            g = vrr._minus(f, k)
            got = [(t.coef, t.mult, t.child) for t in ts]
            want = [("etr_c", 1, (e, g, 0))]
            if e[k]:
                want.append(("inv2q", e[k], (vrr._minus(e, k), g, 0)))
            if g[k]:
                want.append(("inv2q", g[k], (e, vrr._minus(g, k), 0)))
            want.append(("p_over_q", -1, (etr._plus(e, k), g, 0)))
            assert got == want, n
    assert seen == d.nodes
    # pruning: the targets' closure of the forward build, which is closed
    fwd = etr.build(cls, prune=False)
    assert all(t.child in fwd.terms for ts in fwd.terms.values() for t in ts)
    assert d.nodes == vrr.reachable(fwd.terms, d.targets)
    assert all(d.terms[n] == fwd.terms[n] for n in d.nodes)
    # the cascade's bra phase stops at la + lb; the ETR's reaches L
    assert max(sum(n[0]) for n in d.nodes) == L
    assert max(sum(n[0]) for n in c.nodes) == la + lb
    with pytest.raises(ValueError, match="m = 0 only"):
        etr.reduce(((0, 0, 0), (1, 0, 0), 1))


def test_building_the_etr_leaves_the_cascade_unchanged() -> None:
    for cls in CAPPED:
        before = vrr.build(cls)
        etr.build(cls)
        after = vrr.build(cls)
        assert (after.order, dict(after.terms), after.targets) == (
            before.order, dict(before.terms), before.targets)


# ---------------------------------------------------------------------------
# 3. The cost
# ---------------------------------------------------------------------------

# class -> etr_cost's (x, y, z): pinned (the cascade's are data/hgp/cost.json)
ETR_TRIPLES = {
    (0, 0, 0, 0): (18, 40, 16), (0, 0, 0, 1): (50, 46, 22),
    (0, 0, 0, 2): (145, 48, 46), (0, 0, 1, 1): (144, 48, 34),
    (0, 0, 1, 2): (398, 48, 98), (0, 0, 2, 2): (931, 48, 174),
    (0, 1, 0, 1): (136, 48, 58), (0, 1, 0, 2): (431, 48, 166),
    (0, 1, 1, 1): (427, 48, 124), (0, 1, 1, 2): (1106, 48, 376),
    (0, 1, 2, 2): (2401, 48, 676), (0, 2, 0, 2): (1006, 48, 496),
    (0, 2, 1, 1): (996, 48, 394), (0, 2, 1, 2): (2421, 48, 1078),
    (0, 2, 2, 2): (4992, 48, 1894), (1, 1, 1, 1): (963, 48, 340),
    (1, 1, 1, 2): (2357, 48, 1096), (1, 1, 2, 2): (4890, 48, 1996),
    (1, 2, 1, 2): (4488, 48, 3050), (1, 2, 2, 2): (8948, 48, 5402),
    (2, 2, 2, 2): (14447, 48, 10602),
}


def _etr_cost_is_pinned(classes) -> None:
    for cls in classes:
        c, k = cost.etr_cost(cls), cost.cost(cls)
        assert c.triple == ETR_TRIPLES[cls], cls
        assert c.k0 == k.k0, cls  # the same HRR and separations
        assert c.k4["vrr"] == sum(cost.vrr_node_flops(t) for t in
                                  etr.build(cls).terms.values())
        assert c.k4["accumulate"] == k.k4["accumulate"]


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_etr_cost_is_pinned_above_l4() -> None:
    _etr_cost_is_pinned(ABOVE)


def test_the_etr_cost_is_pinned_and_counts_its_own_dag() -> None:
    assert tuple(ETR_TRIPLES) == vrr.CLASSES
    _etr_cost_is_pinned(FAST)
    assert cost.etr_cost((0, 0, 0, 0)).triple == cost.cost((0, 0, 0, 0)).triple
    # (ss|sp): [0|1_d] = etr_c [0|0] - p/q [1_d|0], 3 FLOPs each; the bra
    # [1_d|0]^(0) by eq 25, 3 each; one more K^4 step: p/q and etr_c (7)
    c = cost.etr_cost((0, 0, 0, 1))
    assert c.k4["vrr"] == 3 * 3 + 3 * 3
    assert {"p_over_q", "etr_c"} <= c.steps
    assert not {"p_over_q", "etr_c"} & set(cost.GEOMETRY)
    assert set(cost.GEOMETRY) < set(cost.ETR_GEOMETRY)


# ---------------------------------------------------------------------------
# 4. The arbitration
# ---------------------------------------------------------------------------

# class -> (cascade peak, ETR peak, cascade x, ETR x, rule): the declared
# rule's inputs and output, pinned for the 21 classes (no kernel built).
ARBITRATION = {
    (0, 0, 0, 0): (1, 1, 18, 18, "cascade"),
    (0, 0, 0, 1): (5, 5, 35, 50, "cascade"),
    (0, 0, 0, 2): (12, 14, 82, 145, "cascade"),
    (0, 0, 1, 1): (12, 14, 81, 144, "cascade"),
    (0, 0, 1, 2): (21, 30, 191, 398, "cascade"),
    (0, 0, 2, 2): (34, 57, 400, 931, "cascade"),
    (0, 1, 0, 1): (18, 17, 103, 136, "etr"),
    (0, 1, 0, 2): (43, 41, 312, 431, "etr"),
    (0, 1, 1, 1): (40, 38, 308, 427, "etr"),
    (0, 1, 1, 2): (80, 77, 792, 1106, "etr"),
    (0, 1, 2, 2): (138, 137, 1702, 2401, "etr"),
    (0, 2, 0, 2): (103, 101, 845, 1006, "etr"),
    (0, 2, 1, 1): (96, 92, 835, 996, "etr"),
    (0, 2, 1, 2): (204, 191, 2136, 2421, "etr"),
    (0, 2, 2, 2): (348, 311, 4562, 4992, "etr"),
    (1, 1, 1, 1): (87, 83, 787, 963, "etr"),
    (1, 1, 1, 2): (185, 172, 2018, 2357, "etr"),
    (1, 1, 2, 2): (317, 283, 4335, 4890, "etr"),
    (1, 2, 1, 2): (389, 362, 4423, 4488, "etr"),
    (1, 2, 2, 2): (661, 590, 9390, 8948, "etr"),
    (2, 2, 2, 2): (1078, 962, 16427, 14447, "etr"),
}


def _arbitration_is_the_rule(classes) -> None:
    for cls in classes:
        cp, ep, cx, ex, rule = ARBITRATION[cls]
        s = E.score(cls)
        assert (s.cascade_peak, s.etr_peak, s.cascade_x, s.etr_x) == (
            cp, ep, cx, ex), cls
        # the rule, from its definition
        assert cp == vrr.vrr_peak(vrr.build(cls))
        assert ep == vrr.vrr_peak(etr.build(cls))
        assert (cx, ex) == (cost.cost(cls).x, cost.etr_cost(cls).x)
        assert rule == ("etr" if (ep, ex) < (cp, cx) else "cascade")
        assert s.rule == rule
        # the shipped choice: the model's rule AND the accuracy gate
        assert E.choice(cls) == ("etr" if rule == "etr" and E.gate(cls)
                                 == E.ADMITTED else "cascade")
        assert s.peak_margin == cp - ep and s.x_margin == cx - ex
    # the two opinions disagree on most ETR classes (the model's choice is
    # an opinion): on all but (pd|dd) and (dd|dd)
    disagree = [c for c in classes if not E.score(c).opinions_agree]
    assert disagree == [c for c in classes
                        if E.score(c).rule == "etr" and c not in
                        ((1, 2, 2, 2), (2, 2, 2, 2))]


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_arbitration_table_is_the_rule_above_l4() -> None:
    _arbitration_is_the_rule(ABOVE)


def test_the_arbitration_table_is_the_rule() -> None:
    assert tuple(ARBITRATION) == vrr.CLASSES
    _arbitration_is_the_rule(FAST)
    # the arbitration does real work: both outcomes
    rules = {r[-1] for r in ARBITRATION.values()}
    assert rules == {"etr", "cascade"}
    # the issue's reference points: wins on (sp|sp), (pp|pp), (sd|sd);
    # losses on the pure-ket (ss|pp), (ss|dd)
    for c in ((0, 1, 0, 1), (1, 1, 1, 1), (0, 2, 0, 2)):
        assert E.choice(c) == "etr"
    for c in ((0, 0, 1, 1), (0, 0, 2, 2)):
        assert E.choice(c) == "cascade"


def test_an_override_wins_and_forced_fixes_the_recurrence(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cls = (0, 1, 0, 1)
    monkeypatch.setattr(E, "OVERRIDES", {cls: E.CASCADE})
    assert E.choice(cls) == E.CASCADE and E.score(cls).rule == E.ETR
    assert E.apply(DEFAULT_HOOKS, lever_spec(cls)) is DEFAULT_HOOKS
    hooks = E.forced(E.ETR)(DEFAULT_HOOKS, lever_spec(cls))
    assert not hooks.is_default("vrr")
    assert E.forced(E.CASCADE)(DEFAULT_HOOKS, lever_spec(
        (1, 1, 1, 1))) is DEFAULT_HOOKS
    with pytest.raises(ValueError, match="is not"):
        E.forced("hrr")


ARBITRATION_DATA = ROOT / "data" / "hgp" / "etr_arbitration.json"


def test_the_arbitration_data_rows_at_fast_l_are_report(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # The L <= FAST_L_MAX rows only, recomputed through `report()`; the
    # byte-for-byte file is the `slow` test below.
    monkeypatch.setattr(E._vrr, "CLASSES", tuple(FAST))
    want = E.report()
    doc = json.loads(ARBITRATION_DATA.read_text())
    assert doc["about"] == want["about"]
    assert [r for r in doc["classes"] if tuple(r["class"]) in FAST] == \
        want["classes"]


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_arbitration_data_is_render() -> None:
    # Regenerate (never hand-edit):
    #   PYTHONPATH=src uv run python -c "from felizjueves.f_levers import \
    #   etr; print(etr.render(), end='')" > data/hgp/etr_arbitration.json
    assert ARBITRATION_DATA.read_text() == E.render()
    rows = E.report()["classes"]
    assert [tuple(r["class"]) for r in rows] == list(vrr.CLASSES)
    # no override ships: the choice is the rule through the gate
    assert all(r["choice"] == ("etr" if r["rule"] == "etr"
                               and r["gate"] == E.ADMITTED else "cascade")
               for r in rows)
    assert all(r["accuracy"] == E.ACCURACY.get(tuple(r["class"]))
               for r in rows)


# ---------------------------------------------------------------------------
# 4b. The accuracy gate, recomputed on the device cells' inputs
# ---------------------------------------------------------------------------

# THE GAP #354's first device run found: the interpreted tests above held
# the ETR's F on synthetic shells and on s/p molecules only, while the
# per-class device cells run H2O/cc-pVDZ's contracted shells, where the
# transfer's -p/q term cancels. These tests interpret exactly the device
# cells' launches (test_fock_ondevice's `_class_recs`, the ETR forced, the
# device test's own `forced_kernel`) against the same restricted sweep, at
# the F tolerance, so the interpreted tier sees what the cards saw.
REFUSED = ((0, 0, 2, 2), (0, 1, 1, 2))  # (ss|dd), (sp|pd): the finding


def test_the_gate_is_held_to_the_f_tolerance_and_covers_the_cap() -> None:
    assert E.ACCURACY_TOL == TOL  # test_fock_build's, never loosened
    assert set(E.ACCURACY) == set(CAPPED)
    assert all(E.gate(c) == E.UNVERIFIED
               for c in vrr.CLASSES if sum(c) > CAP)
    assert tuple(c for c in CAPPED if E.gate(c) == E.REFUSED) == REFUSED
    # the finding, pinned: both refused by a wide margin, one of them a
    # class the model alone would ship with the ETR
    assert all(E.ACCURACY[c] > 5 * TOL for c in REFUSED)
    assert E.score((0, 1, 1, 2)).rule == E.ETR
    assert E.choice((0, 1, 1, 2)) == E.CASCADE
    assert ETR_CAPPED == [(0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 1, 1),
                          (0, 2, 0, 2), (0, 2, 1, 1), (1, 1, 1, 1)]


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py): `choice` scores the class
def test_above_the_cap_the_choice_is_the_cascade() -> None:
    assert all(E.choice(c) == E.CASCADE
               for c in vrr.CLASSES if sum(c) > CAP)


def _gate_error(cls) -> float:
    import test_f_lever_etr_device as ed
    import test_fock_ondevice as fo
    from fock_driver import interpret

    recs = ed._swap(fo._class_recs(cls), ed.forced_kernel)
    assert {r.kernel.name for r in recs} == {ed.forced_kernel(cls).name}
    basis, _, _ = molecule(fo.PART1)
    F = np.array(close(interpret(recs, basis.n_ao)))
    G, scale = fo.restricted_sweep(fo.PART1, fo._met(recs))
    assert np.max(np.abs(G)) > 0
    return fo._worst(F, G, scale)


# H2O/cc-pVDZ's launches of a class interpreted: 4-30 s each, not a small
# input, so `slow` -- but for `(ss|dd)`, a REFUSED class and the cheapest
# (4 s), the fast tier's one witness that the record is the interpretation
GATE_PARAMS = [pytest.param(c, id=str(c), marks=() if c == REFUSED[0]
                            else pytest.mark.slow) for c in CAPPED]


@pytest.mark.parametrize("cls", GATE_PARAMS)
def test_the_gates_record_is_the_interpreted_etr_on_the_device_inputs(
        cls) -> None:
    """The ETR forced on every class with L <= 4, interpreted over
    H2O/cc-pVDZ's launches of the class, against `ao.fock`'s restricted
    sweep: the recorded error to 1%, and so the gate's verdict exactly.
    On the two refused classes this is the failure #354's device run saw."""
    w = _gate_error(cls)
    print(f"\n{cls} ETR forced on the device inputs: worst / scale {w:.3e}"
          f" (recorded {E.ACCURACY[cls]:.3e}, tol {TOL:g}) -> {E.gate(cls)}")
    assert w == pytest.approx(E.ACCURACY[cls], rel=1e-2)
    assert (w <= TOL) == (E.gate(cls) == E.ADMITTED)


# ---------------------------------------------------------------------------
# 5. The same integrals by another path: ETR VRR + HRR, contracted
# ---------------------------------------------------------------------------

TOL_ERI = 1e-12  # test_hgp_eri.py's, against MMD and libcint
MOLECULES = ("hf_ccpvtz", "h2o_ccpvdz_j", "prim_spdf")
LIBCINT = ("h2o_631g_eri", "hf_ccpvtz_eri")


@cache
def _fixture(name: str):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


@cache
def _mol_shells(name: str):
    return ob.from_fixture(_fixture(name)).shells


@cache
def _etr_kernels(cls, absolute: bool):
    spec = hgp_drive.f_spec(cls)
    vk = E.build_etr_vrr_kernel(spec, absolute=absolute)
    hk = R.build_hrr_kernel(spec)
    if absolute:
        return hgp_drive.absolute_kernel(vk), hgp_drive.absolute_kernel(hk)
    return vk, hk


def etr_block(bra, ket, absolute: bool = False) -> np.ndarray:
    """`hgp_drive._run`'s composition with the ETR VRR: per primitive
    quartet, summed in primitive order, then `hgp.recurrence`'s HRR. With
    `absolute`, the ETR's |.| twin on |scalars| (`build_etr_vrr_kernel`):
    `SUM |term|` through the ETR's own sums."""
    sa, sb = bra
    sc, sd = ket
    cls = (sa.l, sb.l, sc.l, sd.l)
    vk, hk = _etr_kernels(cls, absolute)
    fix = abs if absolute else (lambda x: x)
    run_v = Interpreter(vk, boys=boys)
    n = len(R.boundary_order(cls))
    bnd = np.zeros(n)
    for i, j, k, l in itertools.product(  # noqa: E741
            *(range(len(s.exponents)) for s in (sa, sb, sc, sd))):
        s = hgp_drive.primitive_scalars(sa, sb, sc, sd, i, j, k, l)
        bnd += run_v.run(**{x: fix(s[x]) for x in R.VRR_SCALARS},
                         out=np.zeros(n))[R.OUT]
    AB = tuple(a - b for a, b in zip(sa.centre, sb.centre, strict=True))
    CD = tuple(a - b for a, b in zip(sc.centre, sd.centre, strict=True))
    geo = {**{f"AB_{a}": AB[x] for x, a in enumerate("xyz")},
           **{f"CD_{a}": CD[x] for x, a in enumerate("xyz")}}
    return Interpreter(hk).run(
        **{x: fix(geo[x]) for x in R.HRR_SCALARS}, boundary=bnd,
        out=np.zeros(len(hrr.targets(cls))))[R.OUT]


def _scale(bra, ket) -> np.ndarray:
    """test_hgp_eri.py's yardstick, for THIS engine: its own |.| twin (the
    ETR's, which sees the `- p/q` transfer's cancellation, as hgp_drive's
    sees the HRR's) plus MMD's -- the triangle-inequality bound on two
    fp64 evaluations of one sum (test_hgp_eri.py, "THE TOLERANCE"). The
    tolerance is that file's `1e-12`, unchanged."""
    _, ms = mmd_eri(bra, ket, pq=pq)
    return etr_block(bra, ket, absolute=True) + np.array(ms).ravel()


def _worst(got, want, scale) -> float:
    err = np.abs(got - want)
    assert np.all(err[scale == 0.0] == 0.0)
    return float(np.max(np.divide(err, scale, out=np.zeros_like(err),
                                  where=scale > 0)))


def _pick(name, cls):
    """test_hgp_eri.py's quartet rule: most distinct centres, then most
    primitive quartets, then first in shell-index order."""
    sh = _mol_shells(name)
    best = None
    for q in itertools.product(range(len(sh)), repeat=4):
        if tuple(sh[s].l for s in q) != cls:
            continue
        key = (len({sh[s].centre for s in q}),
               np.prod([len(sh[s].exponents) for s in q]))
        if best is None or key > best[0]:
            best = (key, q)
    return best[1]


@pytest.mark.parametrize("cls", CAPPED, ids=str)
@pytest.mark.parametrize("mol", MOLECULES)
def test_the_etr_reaches_the_cascades_integrals_contracted(mol, cls) -> None:
    sh = _mol_shells(mol)
    q = _pick(mol, cls)
    bra, ket = (sh[q[0]], sh[q[1]]), (sh[q[2]], sh[q[3]])
    got = etr_block(bra, ket)
    want = np.array(mmd_eri(bra, ket, pq=pq)[0]).ravel()
    w = _worst(got, want, _scale(bra, ket))
    print(f"\n{mol} {cls} {q}: ETR vs MMD max err/scale = {w:.2e}")
    assert w <= TOL_ERI, (mol, cls, q, w)
    # and against the cascade's own block: the two HGP engines' twins
    casc, cs = (np.array(x).ravel() for x in hgp_drive.eri_block_and_scale(
        bra, ket, boys=boys))
    assert _worst(got, casc, etr_block(bra, ket, True) + cs) <= TOL_ERI


def test_the_etr_twin_bounds_the_block_and_sees_its_cancellation() -> None:
    """The ETR's |.| twin is >= |value| everywhere; on the pure-ket
    `(ss|dd)`, where the transfer carries |e| = 4 back to the ket, it is
    far above the cascade's twin -- the cancellation a cascade scale cannot
    see, which is why the yardstick is the engine's own."""
    sh = _mol_shells("h2o_ccpvdz_j")
    q = _pick("h2o_ccpvdz_j", (0, 0, 2, 2))
    bra, ket = (sh[q[0]], sh[q[1]]), (sh[q[2]], sh[q[3]])
    v, s = etr_block(bra, ket), etr_block(bra, ket, absolute=True)
    _, cs = (np.array(x).ravel() for x in hgp_drive.eri_block_and_scale(
        bra, ket, boys=boys))
    assert np.all(s >= np.abs(v))
    assert np.max(s / cs) > 10.0


def test_the_rounding_budget_covers_every_etr_quartet() -> None:
    """`1e-12` is `gamma_n` for `n` about 9000 (test_hgp_eri.py): each
    pinned quartet's K^4 contraction sum plus the ETR's depth (eq 25 to
    |e| = L, the transfer to lc + ld, the HRR's lb + ld), 3 x 5 roundings a
    level (a level at most five terms), stays under it."""
    n_max = TOL_ERI / 2.0 ** -53
    for mol in MOLECULES:
        sh = _mol_shells(mol)
        for cls in CAPPED:
            q = _pick(mol, cls)
            n_prim = int(np.prod([len(sh[s].exponents) for s in q]))
            depth = sum(cls) + cls[2] + cls[3] + cls[1] + cls[3]
            assert n_prim + 15 * depth < n_max, (mol, cls)
    assert max(len(t) for c in CAPPED for t in etr.build(c).terms.values()
               ) <= 5


def _libcint_params():
    out = []
    for name in LIBCINT:
        sh = _mol_shells(name)
        for q in sorted(manifest.eri_blocks(_fixture(name))):
            ls = tuple(sh[s].l for s in q)
            if max(ls) <= 2 and sum(ls) <= CAP:
                out.append(pytest.param(name, q, id=f"{name}-{q}"))
    return out


LIBCINT_PARAMS = _libcint_params()


def test_the_libcint_cases_reach_every_class_within_the_cap() -> None:
    seen = set()
    for p in LIBCINT_PARAMS:
        name, q = p.values
        ls = [_mol_shells(name)[s].l for s in q]
        seen.add(frozenset((tuple(sorted(ls[:2])), tuple(sorted(ls[2:])))))
    assert seen == {frozenset(((c[0], c[1]), (c[2], c[3]))) for c in CAPPED}


@pytest.mark.parametrize("name,q", LIBCINT_PARAMS)
def test_the_etr_matches_libcint(name, q) -> None:
    sh = _mol_shells(name)
    bra, ket = (sh[q[0]], sh[q[1]]), (sh[q[2]], sh[q[3]])
    ref = manifest.eri_blocks(_fixture(name))[q].ravel()
    w = _worst(etr_block(bra, ket), ref, _scale(bra, ket))
    assert w <= TOL_ERI, (name, q, w)


def test_a_perturbed_transfer_coefficient_breaks_the_pin(monkeypatch) -> None:
    """Can it fail: `inv2q` (read only by the ETR's lowering terms on a
    class with no ket cascade) off by 1e-6 misses by far more than TOL."""
    sh = _mol_shells("hf_ccpvtz")
    cls = (1, 1, 1, 1)
    q = _pick("hf_ccpvtz", cls)
    bra, ket = (sh[q[0]], sh[q[1]]), (sh[q[2]], sh[q[3]])
    want = np.array(mmd_eri(bra, ket, pq=pq)[0]).ravel()
    real = hgp_drive.primitive_scalars

    def bent(*a):
        s = real(*a)
        s["inv2q"] *= 1.0 + 1e-6
        return s

    monkeypatch.setattr(hgp_drive, "primitive_scalars", bent)
    assert _worst(etr_block(bra, ket), want, _scale(bra, ket)) > 1e3 * TOL_ERI


# ---------------------------------------------------------------------------
# 6. The kernel
# ---------------------------------------------------------------------------


def _decls(kern) -> list[str]:
    return [s.name for s in walk(kern.body) if isinstance(s, Decl)]


@pytest.mark.parametrize("cls", ETR_CAPPED, ids=str)
def test_the_kernels_vrr_is_the_etr_closure(cls) -> None:
    kern = build_f_kernel(lever_spec(cls), eris=True)
    names = _decls(kern)
    vs = {n for n in names if n.startswith("V")}
    assert vs == {R.vrr_name(x) for x in etr.build(cls).nodes}
    assert len(vs) == len([n for n in names if n.startswith("V")])
    # nothing only the ket cascade reads is formed
    for gone in ("WQ_x", "WQ_y", "WQ_z", "p_ppq", "inv2pq",
                 "inv2q_alpha_over_q"):
        assert gone not in names, gone
    for once in ("etr_pq", "QC_x", "QC_y", "QC_z", "PA_x", "PA_y", "PA_z",
                 "etr_c_x", "etr_c_y", "etr_c_z"):
        assert names.count(once) == 1, once
    assert names.count("inv2q") == (1 if E.needs(etr.build(cls)) >= {
        "inv2q"} else 0)


def _tap(kern, cls, monkeypatch, shells=None):
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    sh = shells or tk._shells(cls)
    basis = Basis(tuple(sh))
    bra, ket = tk.records(basis, 0, 1), tk.records(basis, 2, 3)
    _, g, _ = tk.fused(cls, basis, tk._density(basis.n_ao), [bra], [ket],
                       threads=1, eris=True)
    return g[0, 0], sh


@pytest.mark.parametrize("arm", ["shipped", "forced-etr"])
@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_tap_is_the_mmd_integrals(cls, arm, monkeypatch) -> None:
    """Every class with L <= 4, under the shipped choice and with the ETR
    forced (both recurrences on every class): `g` against `ao/eri.py` to
    test_kernel_f's TOL of scale, its own bound for the same ERIs in two
    operation orders."""
    kern = (build_f_kernel(lever_spec(cls), eris=True) if arm == "shipped"
            else _forced(cls))
    g, sh = _tap(kern, cls, monkeypatch)
    ref, scale = tk.mmd(cls, sh)
    err = np.abs(g - ref)
    assert np.any(ref != 0)
    assert np.all(err <= tk.TOL * scale), float(np.max(err / scale))
    if arm == "forced-etr" and sum(cls[2:]) > 0:
        # a different recurrence (its bits may still coincide on a class
        # as small as (ss|sp), so the program is compared, not the values)
        assert kern.body != build_f_kernel(cls, eris=True).body


def _against_sweep(kern, cls, basis, D, bra_ix, ket_ix, diag, monkeypatch):
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    worst, *_ = tk._check_launch(cls, basis, D, bra_ix, ket_ix, diag=diag)
    return worst


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_one_orbit_equals_its_ordered_members_in_the_sweep(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    basis = Basis(tuple(tk._shells(cls)))
    D = tk._density(basis.n_ao)
    for kern in (build_f_kernel(lever_spec(cls)), _forced(cls, False)):
        worst = _against_sweep(kern, cls, basis, D, [(0, 1)], [(2, 3)], 0,
                               monkeypatch)
        assert worst <= tk.TOL, worst


@pytest.mark.parametrize("case", [c for c in tk.COINCIDENCE
                                  if sum(tk.COINCIDENCE[c][0]) <= CAP])
def test_every_degeneracy_case_within_the_cap_equals_the_sweep(
        case, monkeypatch: pytest.MonkeyPatch) -> None:
    cls, n, bra_ix, ket_ix, diag = tk.COINCIDENCE[case]
    basis = Basis(tuple(tk._same_l_shells(cls[0], n)))
    D = tk._density(basis.n_ao)
    for kern in (build_f_kernel(lever_spec(cls)), _forced(cls, False)):
        worst = _against_sweep(kern, cls, basis, D, bra_ix, ket_ix, diag,
                               monkeypatch)
        assert worst <= tk.TOL, (case, worst)


# s and p only: L <= 4. H2O's walks are 6 s (STO-3G) and 23 s (6-31G)
# interpreted, past the fast tier's ~3 s, so `slow`; H2/STO-3G (no ETR
# class: it reaches (ss|ss) only) stays
SMALL = ("h2_sto3g", pytest.param("h2o_sto3g", marks=pytest.mark.slow),
         pytest.param("h2o_631g", marks=pytest.mark.slow))


def _with_lever(plan):
    return replace(plan, launches=tuple(
        replace(L, spec=replace(L.spec, levers=(NAME,)))
        for L in plan.launches))


@pytest.mark.parametrize("name", SMALL)
def test_the_whole_molecule_is_the_sweep_and_libcint(name) -> None:
    """`walk_f` over launch_f's plan at tau = 0, every launch's kernel the
    lever's: F against the full sweep to TOL of scale and libcint's
    `vj - vk/2` to LIBCINT_TOL (test_fock_build.py's checks)."""
    basis, D, libcint = molecule(name)
    plan = _with_lever(f_launches(basis, D, bound=BOUND, tau=0.0, n=1))
    assert {sum(L.spec.am) for L in plan.launches} <= set(range(CAP + 1))
    assert any(E.choice(tuple(L.spec.am)) == E.ETR for L in plan.launches) \
        or name == "h2_sto3g"
    w = walk_f.walk_f(plan, basis, D, run=run)
    F = np.array(w.F)
    want, scale = sweep(name), sweep(name, block="scale")
    worst = tk._worst(F, want, scale)
    cint = tk._worst(F, libcint, scale)
    print(f"\n{name}: worst / scale vs sweep {worst:.2e}, libcint {cint:.2e}")
    assert worst <= tk.TOL
    assert cint <= LIBCINT_TOL


# ---------------------------------------------------------------------------
# 7. Composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ETR_CAPPED, ids=str)
def test_with_batched_vrr_it_is_the_etr_kernel_bitwise(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = lever_spec(cls, ("batched-vrr",))
    assert spec.key().endswith("/f/batched-vrr+etr")
    both = build_f_kernel(spec, eris=True)
    alone = build_f_kernel(lever_spec(cls), eris=True)
    assert both != alone
    geoms = [tk.CENTRES, tk._moved((0.3, -0.2, 0.1))]
    basis = Basis(tuple(s for g in geoms for s in tk._shells(cls, g)))
    bras = [tk.records(basis, 0, 1), tk.records(basis, 4, 5)]
    kets = [tk.records(basis, 2, 3), tk.records(basis, 6, 7)]
    out = []
    for kern in (both, alone):
        monkeypatch.setattr(tk, "_kernel", lambda *_, k=kern: k)
        F, g, _ = tk.fused(cls, basis, tk._density(basis.n_ao), bras, kets,
                           threads=2, eris=True)
        out.append((F, g))
    assert np.any(out[1][1] != 0)
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])


def test_with_subdag_vrr_it_is_refused() -> None:
    # the refusal is the module's own (`etr` sorts before `subdag-vrr`)
    spec = Spec(Operator.COULOMB, 0, (0, 1, 0, 1), Target.F,
                ("etr", "subdag-vrr"))
    with pytest.raises(ValueError, match="silently discarded"):
        E.apply(DEFAULT_HOOKS, spec)
    with pytest.raises(ValueError, match="subdag-vrr"):
        E.forced(E.CASCADE)(DEFAULT_HOOKS, spec)
    with pytest.raises(ValueError, match="silently discarded"):
        build_f_kernel(lever_spec((0, 1, 0, 1), ("batched-vrr",
                                                 "subdag-vrr")))


def test_with_fmixed_it_is_refused() -> None:
    """The accuracy gate is an fp64 record: the ETR is not shipped in fp32
    (`FMIXED_REFUSAL`), on a cascade class too (the refusal is the
    composition's, not the class's)."""
    for cls in ((0, 1, 0, 1), (0, 0, 1, 1)):
        with pytest.raises(ValueError, match="fp64 record"):
            build_f_kernel(lever_spec(cls, ("fmixed",)))


@pytest.mark.parametrize("cls", ETR_CAPPED, ids=str)
def test_with_batched_vrr_and_sliced_hrr_it_is_the_sweep(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    """`sliced-hrr` (#356, on `batched-vrr`) moves the HRR, the ETR the
    VRR: the composed kernel keeps the ETR's VRR and its F is the sweep's
    to TOL (one orbit, test_kernel_f's shells)."""
    kern = build_f_kernel(lever_spec(cls, ("batched-vrr", "sliced-hrr")))
    assert any(isinstance(s, Decl) and s.name == E.PQ
               for s in walk(kern.body))
    basis = Basis(tuple(tk._shells(cls)))
    worst = _against_sweep(kern, cls, basis, tk._density(basis.n_ao),
                           [(0, 1)], [(2, 3)], 0, monkeypatch)
    assert worst <= tk.TOL, worst


# ---------------------------------------------------------------------------
# 8. The goldens: (sp|sp) with the lever, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel_f import build_f_kernel, close, f_class_spec; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, ('etr',))); \
#   Path(f'test/golden/cuda/f_etr/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); Path(f'test/golden/hip/f_etr/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_etr_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_f_kernel(lever_spec(GOLDEN_CLS))
    where = GOLDEN / backend / "f_etr"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert "double etr_pq = (p / q);" in text
    assert "WQ_x" not in text
