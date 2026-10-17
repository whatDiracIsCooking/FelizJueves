"""The dedicated sub-DAG VRR schedule, `f_levers/subdag_vrr.py` (#357),
interpreted.

The lever re-orders each of #350's VRR groups by a live-minimising list
schedule of its own closure (`hgp.vrr.dedicated_order`), on top of
`batched-vrr`, which it requires. What is asserted:

1. THE DECLARATION: `subdag-vrr` is legal on `Target.F` only, refused on
   the other targets with stated reasons, registered, keyed
   `batched-vrr+subdag-vrr`, refused alone (`REFUSAL`) and before #350's
   staging, moving only the `vrr` hook beyond #350's four, absent from every
   literal and `batched-vrr`-only build.
2. THE SCHEDULE (`hgp.vrr`'s new functions, additive): in-closure consumer
   counts, every list schedule a topological permutation of the closure, the
   dedicated order never above the filtered one's peak and equal to it
   where nothing is better, the closure and terms unchanged, `build`'s DAGs
   unchanged.
3. THE NUMBERS, per group of #350's cut for all 21 classes (no build): the
   statements (one per node, the same in both orders) and the peak live of
   the filtered and the dedicated order (`GROUP_TABLE`, pinned).
4. THE KERNEL'S SHAPE: per nest, the same VRR declarations as #350's kernel
   (the same statements, by value) in the dedicated order; everything else
   in the kernel unchanged.
5. EXACTNESS: interpreted, `F` and the tap `g` equal the literal kernel's
   BIT FOR BIT on every class with L <= 4 at #350's count and at other
   counts (L <= 2, and a sample to L = 4); one launch against `ao.fock`'s
   sweep to `test_kernel_f.TOL`, and the degeneracy cases within the cap;
   classes above L = 4 are `slow` and not run under milestone 13's cap.
6. THE GOLDENS: `(sp|sp)` with `batched-vrr+subdag-vrr`, one per backend.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_kernel_f as tk
from tiers import is_fast, split_by_l

from felizjueves import f_levers, levers
from felizjueves.ao.shell import Basis
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import seams
from felizjueves.f_levers import subdag_vrr as sv
from felizjueves.f_levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.hgp import vrr
from felizjueves.hgp.recurrence import vrr_name
from felizjueves.ir import Decl, For, If, Note, walk
from felizjueves.kernel_f import build_f_kernel, f_class_spec
from felizjueves.spec import Operator, Spec, Target

NAME = sv.NAME
BOTH = (bv.NAME, NAME)
CAP = 4  # milestone 13: every test at L <= 4

LADDER = split_by_l(vrr.CLASSES, sum, ids=lambda c: "".join(map(str, c)))
CAPPED = [c for c in vrr.CLASSES if sum(c) <= CAP]


def lever_spec(cls) -> Spec:
    return f_class_spec(*cls, BOTH)


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_subdag_vrr_is_declared_legal_on_f_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[NAME]
    assert (lever.name, lever.issue, lever.legal) == (NAME, 357,
                                                      frozenset({"f"}))
    assert "VRR groups" in lever.refusals["j"]
    assert "VRR groups" in lever.refusals["k"]
    for slot in ("dq", "jab", "s", "t", "v", "schwarz"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="VRR groups"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (NAME,))
    with pytest.raises(ValueError, match="VRR groups"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))


def test_subdag_vrr_is_registered_keyed_and_wraps_the_vrr_hook() -> None:
    assert f_levers.REGISTRY[NAME] is sv.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == "coulomb/d0/(0,1|0,1)/b1/f/batched-vrr+subdag-vrr"
    assert spec.kernel_body_levers() == BOTH
    hooks = f_levers.hooks_for(spec)
    moved = {f.name for f in dataclasses.fields(FHooks)
             if not hooks.is_default(f.name)}
    assert moved == {"boundary_init", "nest", "accumulate", "boundary_read",
                     "vrr"}
    alone = f_levers.hooks_for(f_class_spec(0, 1, 0, 1, (bv.NAME,)))
    assert alone.is_default("vrr")  # #350 alone keeps the filtered order
    kern = build_f_kernel(spec)
    notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
    assert levers.provenance(NAME) in notes
    assert levers.provenance(bv.NAME) in notes


def test_subdag_vrr_alone_is_refused_with_its_reason() -> None:
    spec = f_class_spec(0, 1, 0, 1, (NAME,))  # legal as a key ...
    assert spec.key() == "coulomb/d0/(0,1|0,1)/b1/f/subdag-vrr"
    with pytest.raises(ValueError, match="requires batched-vrr"):
        build_f_kernel(spec)  # ... but no kernel is built for it
    with pytest.raises(ValueError, match="requires batched-vrr"):
        sv.apply(DEFAULT_HOOKS, spec)
    # requested together but applied before #350's staging: refused
    with pytest.raises(ValueError, match="staged the boundary"):
        sv.apply(DEFAULT_HOOKS, lever_spec((0, 1, 0, 1)))


# ---------------------------------------------------------------------------
# 2. The schedule, in hgp.vrr
# ---------------------------------------------------------------------------


def _topological(dag, order) -> bool:
    seen: set = set()
    for x in order:
        if any(t.child not in seen for t in dag.terms[x]):
            return False
        seen.add(x)
    return True


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_every_list_schedule_is_a_topological_order_of_the_closure(cls
                                                                   ) -> None:
    for g in vrr.groups(cls, bv.n_groups(cls)):
        for tie in vrr.TIES:
            order = vrr.list_schedule(g, tie)
            assert len(order) == g.n_nodes and set(order) == g.nodes
            assert _topological(g, order)
        d = vrr.dedicated(g)
        assert d.nodes == g.nodes and d.targets == g.targets
        assert all(d.terms[x] == g.terms[x] for x in g.nodes)
        assert _topological(g, d.order)
        assert vrr.vrr_peak(d) <= vrr.vrr_peak(g)
        assert vrr.order_peak(g, d.order) == vrr.vrr_peak(d)
    with pytest.raises(ValueError, match="tie"):
        vrr.list_schedule(vrr.build(cls), "nonesuch")


def test_consumer_counts_are_the_closures_own() -> None:
    dag = vrr.build((0, 1, 1, 1))
    full = vrr.consumers(dag)
    g0 = vrr.groups((0, 1, 1, 1), 2)[0]
    mine = vrr.consumers(g0)
    assert set(mine) == g0.nodes
    for x in g0.nodes:
        assert mine[x] == full[x] & g0.nodes
    # somewhere the parent reads a node the group no longer does
    assert any(mine[x] < full[x] for x in g0.nodes)
    # every non-target of a closure has a reader in it
    assert all(mine[x] for x in g0.nodes - set(g0.targets))


def test_the_filtered_order_is_kept_where_nothing_is_better() -> None:
    for cls in ((0, 0, 0, 0), (0, 0, 0, 1)):
        (g,) = vrr.groups(cls, 1)
        assert vrr.dedicated_order(g) == g.order  # the first candidate wins
    # and is replaced where a list schedule is strictly better
    (g,) = vrr.groups((1, 1, 1, 1), 1)
    assert vrr.dedicated_order(g) != g.order


def test_build_is_unchanged_by_the_schedule() -> None:
    for cls in CAPPED:
        before = vrr.build(cls)
        for g in vrr.groups(cls, bv.n_groups(cls)):
            vrr.dedicated(g)
        after = vrr.build(cls)
        assert (after.order, dict(after.terms), after.targets) == (
            before.order, dict(before.terms), before.targets)
        assert vrr.groups(cls, 1)[0].order == before.order


# ---------------------------------------------------------------------------
# 3. The numbers: per group, statements and peak live (filtered, dedicated)
# ---------------------------------------------------------------------------

# class -> per group of #350's cut at `batched_vrr.n_groups`:
# (statements, filtered peak, dedicated peak), `hgp.vrr.vrr_peak`'s
# statement-order model. Computed, pinned so a change to the DAGs, the cut
# or the schedule is a visible diff. The statements are the same in both
# orders by construction; the dedicated peak is never higher.
GROUP_TABLE = {
    (0, 0, 0, 0): ((1, 1, 1),),
    (0, 0, 0, 1): ((5, 5, 5),),
    (0, 0, 0, 2): ((8, 7, 6), (12, 9, 7)),
    (0, 0, 1, 1): ((8, 7, 6), (12, 9, 7)),
    (0, 0, 1, 2): ((35, 21, 20),),
    (0, 0, 2, 2): ((70, 34, 32),),
    (0, 1, 0, 1): ((11, 10, 9), (17, 11, 10)),
    (0, 1, 0, 2): ((28, 22, 21), (55, 32, 24)),
    (0, 1, 1, 1): ((30, 21, 19), (52, 28, 22)),
    (0, 1, 1, 2): ((67, 44, 39), (136, 72, 46)),
    (0, 1, 2, 2): ((139, 78, 64), (281, 130, 85)),
    (0, 2, 0, 2): ((71, 54, 51), (139, 75, 55)),
    (0, 2, 1, 1): ((79, 51, 46), (132, 70, 50)),
    (0, 2, 1, 2): ((168, 104, 96), (346, 177, 113)),
    (0, 2, 2, 2): ((243, 129, 107), (440, 212, 137), (580, 244, 177)),
    (1, 1, 1, 1): ((146, 87, 82),),
    (1, 1, 1, 2): ((336, 185, 172),),
    (1, 1, 2, 2): ((671, 317, 280),),
    (1, 2, 1, 2): ((696, 389, 362),),
    (1, 2, 2, 2): ((1381, 661, 590),),
    (2, 2, 2, 2): ((2330, 1078, 962),),
}


def _group_table_is_the_schedule(classes) -> None:
    for cls in classes:
        rows = GROUP_TABLE[cls]
        assert sv.group_peaks(cls) == rows, cls
        assert len(rows) == bv.n_groups(cls), cls
        assert max(r[1] for r in rows) == bv.vrr_phase(cls, len(rows)), cls
        assert all(d <= f for _, f, d in rows), cls  # one-directional


def test_the_group_table_covers_the_ladder_and_is_the_schedule() -> None:
    assert tuple(GROUP_TABLE) == vrr.CLASSES
    _group_table_is_the_schedule(c for c in vrr.CLASSES if is_fast(sum(c)))


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_group_table_is_the_schedule_above_l4() -> None:
    _group_table_is_the_schedule(c for c in vrr.CLASSES if not is_fast(sum(c)))


# ---------------------------------------------------------------------------
# 4. The kernel's shape
# ---------------------------------------------------------------------------


def _nests(kern) -> list[For]:
    return [s for s in walk(kern.body)
            if isinstance(s, For) and s.var == seams.KCD]


def _vdecls(nest: For) -> list[Decl]:
    return [s for s in walk(nest.body) if isinstance(s, Decl)
            and s.name.startswith("V")]


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_each_nest_is_batched_vrrs_statements_in_the_dedicated_order(cls
                                                                     ) -> None:
    n = bv.n_groups(cls)
    kern = build_f_kernel(lever_spec(cls), eris=True)
    base = build_f_kernel(f_class_spec(*cls, (bv.NAME,)), eris=True)
    nests, bnests = _nests(kern), _nests(base)
    assert len(nests) == len(bnests) == n
    changed = False
    for nest, bnest, g in zip(nests, bnests, vrr.groups(cls, n), strict=True):
        mine, theirs = _vdecls(nest), _vdecls(bnest)
        assert sorted(mine, key=lambda d: d.name) == sorted(
            theirs, key=lambda d: d.name)  # the same statements, by value
        assert [d.name for d in mine] == [
            vrr_name(x) for x in vrr.dedicated_order(g)]
        changed |= mine != theirs
        # the rest of the nest (ket prim, scalars, accumulation) unchanged
        assert [s for s in nest.body if not isinstance(s, For)] == [
            s for s in bnest.body if not isinstance(s, For)]
    assert changed == any(d < f for _, f, d in GROUP_TABLE[cls])
    # outside the VRR declarations (and the lever's Note) the kernel is
    # #350's, statement for statement, in order
    assert _rest(kern) == _rest(base)


def _rest(kern) -> list:
    mine = levers.provenance(NAME)
    return [s for s in walk(kern.body) if not isinstance(s, (For, If))
            and not (isinstance(s, Decl) and s.name.startswith("V"))
            and not (isinstance(s, Note) and s.prov == mine)]


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
    """Every class, at #350's count: the tap and F."""
    kern = build_f_kernel(lever_spec(cls), eris=True)
    _bitwise(cls, kern, monkeypatch)


def _hooks(cls, n):
    spec = lever_spec(cls)
    return sv.apply(bv.with_groups(n)(DEFAULT_HOOKS, spec), spec)


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
    assert len(_nests(kern)) == n
    for nest, g in zip(_nests(kern), vrr.groups(cls, n), strict=True):
        assert [d.name for d in _vdecls(nest)] == [
            vrr_name(x) for x in vrr.dedicated_order(g)]
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
# 6. The goldens: (sp|sp) with batched-vrr+subdag-vrr, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, \
#   ('batched-vrr', 'subdag-vrr'))); \
#   Path(f'test/golden/cuda/f_subdag_vrr/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); Path(f'test/golden/hip/f_subdag_vrr/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_subdag_vrr_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_f_kernel(lever_spec(GOLDEN_CLS))
    where = GOLDEN / backend / "f_subdag_vrr"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    base = emit(build_f_kernel(f_class_spec(*GOLDEN_CLS, (bv.NAME,))), arch)
    assert text != base
    assert sorted(text.splitlines()) != sorted(base.splitlines())  # + Note
    assert text.count("for (int kcd = 0;") == bv.n_groups(GOLDEN_CLS) == 2
