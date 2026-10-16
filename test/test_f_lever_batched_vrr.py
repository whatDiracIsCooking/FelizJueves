"""The batched VRR, `f_levers/batched_vrr.py` (#350), interpreted.

The lever stages the fused F kernel's boundary in a thread-private
`Space.LOCAL` array and runs the primitive nest once per schedule-contiguous
group of it, each group computing only its closure of the VRR DAG. What is
asserted:

1. THE DECLARATION: `batched-vrr` is legal on `Target.F` only, refused on
   the other targets with stated reasons, keyed as any lever is, registered
   in `f_levers.REGISTRY`, moving `boundary_init`, `nest`, `accumulate` and
   `boundary_read` only, and absent from every literal build (the literal
   F kernels' digests are test_f_seams.py's pins and test_emit_fock.py's
   goldens, unchanged).
2. THE GROUPING (`hgp.vrr`'s new functions, additive): the cut is
   contiguous in the DAG's schedule and partitions the targets; a group's
   sub-DAG is its closure, with the parent's terms, in the parent's order
   filtered (or a given order, checked); `peak_live` on hand-checked cases;
   and the DAGs `build` returns are unchanged (their counts stay pinned by
   test_hgp_vrr.py and hgp/cost.json by test_hgp_cost.py).
3. THE GROUP COUNT, a declared function of the class: the rule's inputs and
   output pinned for all 21 classes (no build), the rule read off its
   definition, `OVERRIDES` winning, `floor=` moving it, `with_groups`
   building any count and refusing a bad one.
4. THE KERNEL'S SHAPE: one `boundary[n_bnd]` in local memory, no register
   accumulator left, one primitive nest per group whose VRR is exactly the
   group's closure, every boundary slot accumulated exactly once, and the
   HRR's leaves read from the array.
5. EXACTNESS (the issue's "every class matches the literal fused kernel on
   the host shim; the `eris=True` tap bitwise-pinned"): interpreted, the
   lever's `F` and tap `g` are the literal kernel's BIT FOR BIT on every
   class with L <= 4 (milestone 13's cap) at the rule's count, and at other
   counts (2, 3 and one group per boundary value) on every class with
   L <= 2 and a sample to L = 4; one launch against `ao.fock`'s sweep to
   `test_kernel_f.TOL`, and the degeneracy cases within the cap; classes
   above L = 4 are `slow` and not run under the cap.
6. THE GOLDENS: `(sp|sp)` with the lever (two groups), one unit per backend.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_kernel_f as tk

from felizjueves import f_levers, levers
from felizjueves.ao.shell import Basis
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import seams
from felizjueves.f_levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import boundary_order, vrr_name
from felizjueves.ir import Assign, Decl, For, Index, Lit, Note, Space, walk
from felizjueves.kernel_f import build_f_kernel, f_class_spec
from felizjueves.spec import Operator, Spec, Target

NAME = bv.NAME
CAP = 4  # milestone 13: every test at L <= 4


def _speed(cls):
    return pytest.param(cls, id="".join(map(str, cls)),
                        marks=(pytest.mark.slow,) if sum(cls) > CAP else ())


LADDER = [_speed(c) for c in vrr.CLASSES]
L2 = [c for c in vrr.CLASSES if sum(c) <= 2]
CAPPED = [c for c in vrr.CLASSES if sum(c) <= CAP]


def lever_spec(cls) -> Spec:
    return f_class_spec(*cls, (NAME,))


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_batched_vrr_is_declared_legal_on_f_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[NAME]
    assert (lever.name, lever.issue, lever.legal) == (NAME, 350,
                                                      frozenset({"f"}))
    assert "boundary" in lever.refusals["j"]
    assert "boundary" in lever.refusals["k"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="HGP boundary"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (NAME,))
    with pytest.raises(ValueError, match="HGP VRR boundary"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))


def test_batched_vrr_is_registered_keyed_and_moves_four_hooks() -> None:
    assert f_levers.REGISTRY[NAME] is bv.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == "coulomb/d0/(0,1|0,1)/b1/f/batched-vrr"
    assert spec.kernel_body_levers() == (NAME,)
    with pytest.raises(ValueError):  # the group count is not n_batches
        Spec(Operator.COULOMB, 0, (0, 1, 0, 1), Target.F, (NAME,), 3)
    moved = {f.name for f in dataclasses.fields(FHooks)
             if not f_levers.hooks_for(spec).is_default(f.name)}
    assert moved == {"boundary_init", "nest", "accumulate", "boundary_read"}
    assert f_levers.hooks_for(f_class_spec(0, 1, 0, 1)) is DEFAULT_HOOKS
    kern = build_f_kernel(spec)
    assert kern.name == "f_0101"
    assert levers.provenance(NAME) in [s.prov for s in walk(kern.body)
                                       if isinstance(s, Note)]


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_literal_kernel_is_untouched(cls) -> None:
    kern = build_f_kernel(cls, eris=True)
    assert not any(isinstance(s, Decl) and s.name == bv.ARRAY
                   for s in walk(kern.body))


# ---------------------------------------------------------------------------
# 2. The grouping, in hgp.vrr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_cut_is_schedule_contiguous_and_partitions_the_targets(cls
                                                                   ) -> None:
    dag = vrr.build(cls)
    seq = vrr.scheduled_targets(dag)
    assert sorted(seq) == sorted(dag.targets) and len(set(seq)) == len(seq)
    pos = {n: i for i, n in enumerate(dag.order)}
    assert [pos[n] for n in seq] == sorted(pos[n] for n in seq)
    for n in sorted({1, 2, 3, len(seq)} & set(range(1, len(seq) + 1))):
        gs = vrr.group_targets(dag, n)
        assert len(gs) == n and all(gs)
        assert tuple(t for g in gs for t in g) == seq  # contiguous, in order
        sizes = [len(g) for g in gs]
        assert max(sizes) - min(sizes) <= 1  # near-equal


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_a_group_dag_is_its_closure_in_the_parents_order(cls) -> None:
    dag = vrr.build(cls)
    n = min(3, len(dag.targets))
    subs = vrr.groups(cls, n)
    covered: set = set()
    for g, sub in zip(vrr.group_targets(dag, n), subs, strict=True):
        assert sub.targets == g and sub.cls == dag.cls
        assert sub.nodes == vrr.reachable(dag.terms, g)
        assert all(sub.terms[x] == dag.terms[x] for x in sub.nodes)
        assert sub.order == tuple(x for x in dag.order if x in sub.nodes)
        covered |= sub.nodes
        # the order is topological: every child before its parent
        seen: set = set()
        for x in sub.order:
            assert all(t.child in seen for t in sub.terms[x])
            seen.add(x)
    assert covered == dag.nodes  # the pruned DAG is every target's closure
    assert vrr.groups(cls, 1)[0].nodes == dag.nodes


def test_a_given_group_order_is_checked() -> None:
    dag = vrr.build((0, 1, 0, 1))
    (g,) = vrr.group_targets(dag, 1)
    rev = tuple(reversed(dag.order))
    with pytest.raises(ValueError, match="precedes a node it reads"):
        vrr.group_dag(dag, g, order=rev)
    with pytest.raises(ValueError, match="not the group's closure"):
        vrr.group_dag(dag, g, order=dag.order[1:])
    # a different topological order is admitted (#357's hook): leaves last
    # among the m = 0 ... here just the parent's, re-sorted stably by |e|+|f|
    alt = tuple(sorted(dag.order, key=lambda x: sum(x[0]) + sum(x[1])))
    assert vrr.group_dag(dag, g, order=alt).order == alt
    with pytest.raises(ValueError, match="targets"):
        vrr.group_dag(dag, (((9, 9, 9), (0, 0, 0), 0),))
    with pytest.raises(ValueError, match="targets"):
        vrr.group_dag(dag, ())
    with pytest.raises(ValueError, match="group count"):
        vrr.group_targets(dag, 0)
    with pytest.raises(ValueError, match="group count"):
        vrr.group_targets(dag, len(dag.targets) + 1)


def test_peak_live_on_hand_checked_orders() -> None:
    chain = {"a": (), "b": ("a",), "c": ("b",)}
    assert vrr.peak_live("abc", chain.__getitem__) == 2
    assert vrr.peak_live("abc", chain.__getitem__, "abc") == 3
    fan = {"a": (), "b": (), "c": (), "d": ("a", "b", "c")}
    assert vrr.peak_live("abcd", fan.__getitem__) == 4
    assert vrr.peak_live("abcd", fan.__getitem__, "d") == 4
    assert vrr.peak_live("ab", {"a": (), "b": ()}.__getitem__) == 1
    with pytest.raises(ValueError, match="not produced before"):
        vrr.peak_live("ba", chain.__getitem__)
    with pytest.raises(ValueError, match="twice"):
        vrr.peak_live("aa", chain.__getitem__)
    with pytest.raises(ValueError, match="live-out"):
        vrr.peak_live("a", chain.__getitem__, "z")
    # (ss|ss): one leaf, one target
    assert vrr.vrr_peak(vrr.build((0, 0, 0, 0))) == 1


def test_build_is_unchanged_by_the_grouping() -> None:
    # The new functions read `build`; they never feed it. The pinned counts
    # (test_hgp_vrr.py) and hgp/cost.json (test_hgp_cost.py) are the check;
    # here, that building groups leaves a fresh build equal to an old one.
    for cls in CAPPED:
        before = vrr.build(cls)
        vrr.groups(cls, min(2, len(before.targets)))
        after = vrr.build(cls)
        assert (after.order, dict(after.terms), after.targets) == (
            before.order, dict(before.terms), before.targets)


# ---------------------------------------------------------------------------
# 3. The group count
# ---------------------------------------------------------------------------

# class -> (hrr_floor, literal_phase, rule's count, vrr_phase at that count):
# hgp.vrr.peak_live's statement-order model (f_levers/batched_vrr.py,
# THE GROUP COUNT). Computed by the rule, pinned so a change to the DAGs'
# orders, the cut or the rule is a visible diff.
RULE_TABLE = {
    (0, 0, 0, 0): (1, 2, 1, 1),
    (0, 0, 0, 1): (5, 9, 1, 5),
    (0, 0, 0, 2): (11, 22, 2, 9),
    (0, 0, 1, 1): (11, 21, 2, 9),
    (0, 0, 1, 2): (25, 40, 1, 21),
    (0, 0, 2, 2): (45, 65, 1, 34),
    (0, 1, 0, 1): (17, 34, 2, 11),
    (0, 1, 0, 2): (41, 83, 2, 32),
    (0, 1, 1, 1): (37, 76, 2, 28),
    (0, 1, 1, 2): (77, 156, 2, 72),
    (0, 1, 2, 2): (131, 262, 2, 130),
    (0, 2, 0, 2): (101, 203, 2, 75),
    (0, 2, 1, 1): (91, 186, 2, 70),
    (0, 2, 1, 2): (191, 394, 2, 177),
    (0, 2, 2, 2): (311, 658, 3, 244),
    (1, 1, 1, 1): (91, 168, 1, 87),
    (1, 1, 1, 2): (209, 356, 1, 185),
    (1, 1, 2, 2): (389, 596, 1, 317),
    (1, 2, 1, 2): (457, 750, 1, 389),
    (1, 2, 2, 2): (776, 1250, 1, 661),
    (2, 2, 2, 2): (1550, 2039, 1, 1078),
}


def test_the_rule_table_covers_the_ladder_and_is_the_rule() -> None:
    assert tuple(RULE_TABLE) == vrr.CLASSES
    for cls, (floor, lit, n, phase) in RULE_TABLE.items():
        assert bv.hrr_floor(cls) == floor, cls
        assert bv.literal_phase(cls) == lit, cls
        assert bv.rule(cls) == bv.n_groups(cls) == n, cls
        assert bv.vrr_phase(cls, n) == phase <= floor, cls
        # smallest: one group fewer does not reach the floor
        assert n == 1 or bv.vrr_phase(cls, n - 1) > floor, cls
        # the staging alone (one group) never raises the VRR phase
        assert bv.vrr_phase(cls, 1) == lit - len(vrr.targets(cls)), cls


def test_the_floor_is_the_composition_point() -> None:
    cls = (0, 1, 1, 1)
    assert bv.rule(cls, floor=bv.hrr_floor(cls)) == bv.rule(cls) == 2
    # a lower floor (a sliced HRR's, #356) cuts more groups
    assert bv.rule(cls, floor=20) == min(
        n for n in range(1, 37) if bv.vrr_phase(cls, n) <= 20)
    assert bv.rule(cls, floor=20) > 2
    # unreachable: the count of least demand
    n_bnd = len(vrr.targets(cls))
    least = min((bv.vrr_phase(cls, n), n) for n in range(1, n_bnd + 1))
    assert bv.rule(cls, floor=1) == least[1]
    with pytest.raises(ValueError, match="positive"):
        bv.rule(cls, floor=0)


def test_an_override_wins_and_with_groups_fixes_any_count(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cls = (0, 1, 0, 1)
    monkeypatch.setattr(bv, "OVERRIDES", {cls: 5})
    assert bv.n_groups(cls) == 5 and bv.rule(cls) == 2
    assert _nests(build_f_kernel(lever_spec(cls))) == 5
    hooks = bv.with_groups(3)(DEFAULT_HOOKS, lever_spec(cls))
    assert _nests(build_f_kernel(lever_spec(cls), hooks=hooks)) == 3
    with pytest.raises(ValueError, match="positive"):
        bv.with_groups(0)
    with pytest.raises(ValueError, match="17 groups of 16"):
        bv.with_groups(17)(DEFAULT_HOOKS, lever_spec(cls))


# ---------------------------------------------------------------------------
# 4. The kernel's shape
# ---------------------------------------------------------------------------


def _nests(kern) -> int:
    return sum(1 for s in walk(kern.body)
               if isinstance(s, For) and s.var == seams.KCD)


def _hooks(cls, n):
    return bv.with_groups(n)(DEFAULT_HOOKS, lever_spec(cls))


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_kernel_stages_the_boundary_and_runs_one_nest_per_group(cls
                                                                    ) -> None:
    n = bv.n_groups(cls)
    kern = build_f_kernel(lever_spec(cls), eris=True)
    stmts = list(walk(kern.body))
    staged = [s for s in stmts if isinstance(s, Decl) and s.name == bv.ARRAY]
    n_bnd = len(boundary_order(cls))
    assert len(staged) == 1
    assert staged[0].space is Space.LOCAL
    assert staged[0].shape == (Lit(n_bnd),)
    assert not any(isinstance(s, Decl) and s.name.startswith("C")
                   and s.name not in ("CD_x", "CD_y", "CD_z")
                   for s in stmts)  # no register accumulator left
    nests = [s for s in stmts if isinstance(s, For) and s.var == seams.KCD]
    assert len(nests) == n
    slots = {ef: i for i, ef in enumerate(boundary_order(cls))}
    hit: list[int] = []
    for nest, sub in zip(nests, vrr.groups(cls, n), strict=True):
        body = list(walk(nest.body))
        vdecl = {s.name for s in body if isinstance(s, Decl)
                 and s.name.startswith("V")}
        assert vdecl == {vrr_name(x) for x in sub.nodes}  # the closure only
        acc = [s for s in body if isinstance(s, Assign)
               and isinstance(s.dst, Index) and s.dst.array == bv.ARRAY]
        assert [a.dst.at[0].value for a in acc] == [
            slots[(e, f)] for e, f, _ in sub.targets]
        hit += [a.dst.at[0].value for a in acc]
    assert sorted(hit) == list(range(n_bnd))  # every slot exactly once
    # the HRR's leaves are loads from the array
    leaves = [s for s in stmts if isinstance(s, Decl) and s.name.startswith(
        "H") and isinstance(s.init, Index) and s.init.array == bv.ARRAY]
    assert len(leaves) == n_bnd == hrr.build(cls).n_boundary


# ---------------------------------------------------------------------------
# 5. Exactness: the literal kernel bit for bit, and the sweep
# ---------------------------------------------------------------------------


def _launch(kern, cls, monkeypatch: pytest.MonkeyPatch):
    """`F` (unclosed) and the tap `g` of `kern` over two bras by two kets,
    `test_kernel_f`'s shells and density, two threads."""
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    geoms = [tk.CENTRES, tk._moved((0.3, -0.2, 0.1))]
    basis = Basis(tuple(s for g in geoms for s in tk._shells(cls, g)))
    bras = [tk.records(basis, 0, 1), tk.records(basis, 4, 5)]
    kets = [tk.records(basis, 2, 3), tk.records(basis, 6, 7)]
    F, g, _ = tk.fused(cls, basis, tk._density(basis.n_ao), bras, kets,
                       threads=2, eris=True)
    return F, g


def _bitwise(cls, kern, monkeypatch) -> None:
    F, g = _launch(kern, cls, monkeypatch)
    F0, g0 = _launch(build_f_kernel(cls, eris=True), cls, monkeypatch)
    assert np.any(g0 != 0) and np.any(F0 != 0)  # not a comparison of zeros
    assert np.array_equal(g, g0)
    assert np.array_equal(F, F0)


@pytest.mark.parametrize("cls", LADDER)
def test_the_lever_is_the_literal_kernel_bitwise(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every class, at the rule's count, the tap and F."""
    kern = build_f_kernel(lever_spec(cls), eris=True)
    assert kern != build_f_kernel(cls, eris=True)
    _bitwise(cls, kern, monkeypatch)


OTHER_COUNTS = [
    pytest.param(c, n, id=f"{''.join(map(str, c))}-g{n}")
    for c in vrr.CLASSES if sum(c) <= 2
    for n in sorted({2, 3, len(vrr.targets(c))})
    if n <= len(vrr.targets(c))
] + [
    pytest.param(c, n, id=f"{''.join(map(str, c))}-g{n}")
    for c, n in (((0, 1, 1, 1), 5), ((0, 0, 2, 2), 4), ((1, 1, 1, 1), 3),
                 ((0, 2, 0, 2), 7))
]


@pytest.mark.parametrize(("cls", "n"), OTHER_COUNTS)
def test_any_group_count_is_the_literal_kernel_bitwise(
        cls, n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    kern = build_f_kernel(lever_spec(cls), eris=True, hooks=_hooks(cls, n))
    assert _nests(kern) == n
    _bitwise(cls, kern, monkeypatch)


def _against_sweep(cls, basis, D, bra_ix, ket_ix, diag, monkeypatch):
    kern = build_f_kernel(lever_spec(cls))
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    worst, *_ = tk._check_launch(cls, basis, D, bra_ix, ket_ix, diag=diag)
    return worst


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_one_orbit_equals_its_ordered_members_in_the_sweep(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    basis = Basis(tuple(tk._shells(cls)))
    worst = _against_sweep(cls, basis, tk._density(basis.n_ao), [(0, 1)],
                           [(2, 3)], 0, monkeypatch)
    assert worst <= tk.TOL, worst


@pytest.mark.parametrize("case", [c for c in tk.COINCIDENCE
                                  if sum(tk.COINCIDENCE[c][0]) <= CAP])
def test_every_degeneracy_case_within_the_cap_equals_the_sweep(
        case, monkeypatch: pytest.MonkeyPatch) -> None:
    cls, n, bra_ix, ket_ix, diag = tk.COINCIDENCE[case]
    basis = Basis(tuple(tk._same_l_shells(cls[0], n)))
    worst = _against_sweep(cls, basis, tk._density(basis.n_ao), bra_ix,
                           ket_ix, diag, monkeypatch)
    assert worst <= tk.TOL, (case, worst)


# ---------------------------------------------------------------------------
# 6. The goldens: (sp|sp) with the lever, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, ('batched-vrr',))); \
#   Path(f'test/golden/cuda/f_batched_vrr/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); Path(f'test/golden/hip/f_batched_vrr/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_batched_vrr_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_f_kernel(lever_spec(GOLDEN_CLS))
    where = GOLDEN / backend / "f_batched_vrr"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert f"double {bv.ARRAY}[16];" in text
    assert text.count("for (int kcd = 0;") == bv.n_groups(GOLDEN_CLS) == 2
