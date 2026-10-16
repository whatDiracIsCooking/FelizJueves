"""The sliced HRR, `f_levers/sliced_hrr.py` (#356), interpreted.

The lever runs the fused F kernel's HRR once per bra pair `(a, b)` on
#350's staged boundary, each slice's ERIs tapped and predigested before the
next slice, and re-cuts #350's VRR groups at the sliced floor. What is
asserted:

1. THE DECLARATION: `sliced-hrr` is legal on `Target.F` only, refused on
   the other targets with stated reasons, registered in `f_levers.REGISTRY`;
   it is a COMPOSITION: refused without `batched-vrr` (with its reason, and
   by the CLI with exit 2), refused when applied before the staging or
   after a lever that moved a hook its slices bypass, and with
   `batched-vrr` it moves `contracted` and `nest` beyond #350's four hooks.
   The literal kernel and #350 alone are untouched (their digests and
   goldens are test_f_seams.py's, test_emit_fock.py's and
   test_f_lever_batched_vrr.py's pins).
2. THE SLICES (`hgp.hrr`'s new functions, additive): the per-bra-pair cut
   is contiguous and partitions the targets in their order; a slice is its
   closure, with the parent's terms, in the parent's order filtered (or a
   given order, checked); `hrr_peak` reads the slice; `build` is unchanged
   (its counts stay pinned by test_hgp_hrr.py and hgp/cost.json).
3. THE DECLARED FUNCTIONS OF THE CLASS, pinned for all 21 classes (no
   build): the slice count `n_a n_b`, the per-slice live set's maximum
   (`hrr_phase`, never above #350's whole-HRR floor), the group count #350's
   rule gives at that floor, and the HRR statements the slices emit
   (recompute of the shared nodes included).
4. THE KERNEL'S SHAPE: one scoped one-trip loop per slice, holding exactly
   that slice's HRR nodes, its tap writes and its predigest; no HRR node
   outside a slice; the tap indices partition `0..n_eri`; one primitive
   nest per group at the re-cut count.
5. EXACTNESS: interpreted, the composed kernel's `F` and tap `g` are the
   literal kernel's BIT FOR BIT on every class with L <= 4 (milestone 13's
   cap; above it `slow`, not run under the cap), and so are the slices on
   #350 at its own (unsliced) group count -- the slicing's contribution
   separated from the re-cut; one launch against `ao.fock`'s sweep to
   `test_kernel_f.TOL`, and the degeneracy cases within the cap.
6. THE GOLDENS: `(sp|sp)` with both levers, one unit per backend.

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
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import seams
from felizjueves.f_levers import sliced_hrr as sh
from felizjueves.f_levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import hrr_name
from felizjueves.ir import Assign, Decl, For, Index, Lit, Note, Var, walk
from felizjueves.kernel_f import build_f_kernel, f_class_spec
from felizjueves.spec import Operator, Spec, Target

NAME = sh.NAME
BOTH = (bv.NAME, NAME)
CAP = 4  # milestone 13: every test at L <= 4


def _speed(cls):
    return pytest.param(cls, id="".join(map(str, cls)),
                        marks=(pytest.mark.slow,) if sum(cls) > CAP else ())


LADDER = [_speed(c) for c in vrr.CLASSES]
CAPPED = [c for c in vrr.CLASSES if sum(c) <= CAP]


def lever_spec(cls) -> Spec:
    return f_class_spec(*cls, BOTH)


# ---------------------------------------------------------------------------
# 1. The declaration: a composition
# ---------------------------------------------------------------------------


def test_sliced_hrr_is_declared_legal_on_f_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[NAME]
    assert (lever.name, lever.issue, lever.legal) == (NAME, 356,
                                                      frozenset({"f"}))
    assert "HRR" in lever.refusals["j"] and "HRR" in lever.refusals["k"]
    assert "batched-vrr" in lever.what
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="bra-pair HRR"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (NAME,))
    with pytest.raises(ValueError, match="HGP HRR"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))


def test_sliced_hrr_is_registered_keyed_and_composes_on_batched_vrr() -> None:
    assert f_levers.REGISTRY[NAME] is sh.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == "coulomb/d0/(0,1|0,1)/b1/f/batched-vrr+sliced-hrr"
    assert spec.kernel_body_levers() == BOTH  # key order: #350 applied first
    hooks = f_levers.hooks_for(spec)
    moved = {f.name for f in dataclasses.fields(FHooks)
             if not hooks.is_default(f.name)}
    assert moved == {"boundary_init", "nest", "accumulate", "boundary_read",
                     "contracted"}
    kern = build_f_kernel(spec)
    assert kern.name == "f_0101"
    notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
    assert levers.provenance(NAME) in notes
    assert levers.provenance(bv.NAME) in notes


def test_sliced_hrr_alone_is_refused_with_its_reason(tmp_path) -> None:
    spec = f_class_spec(0, 1, 0, 1, (NAME,))  # legal on F: the key admits it
    with pytest.raises(ValueError, match="requires batched-vrr") as err:
        build_f_kernel(spec)
    assert sh.REFUSAL in str(err.value)
    # the CLI's lever flag builds the lever alone: refused, exit 2
    assert emit_cli.main(["--backend", "cuda", "--out", str(tmp_path),
                          "--klever-sliced-hrr-fock-l", "1"]) == 2
    assert not list(tmp_path.iterdir())


def test_an_incoherent_application_order_or_a_bypassed_hook_is_refused(
        ) -> None:
    spec = lever_spec((0, 1, 0, 1))
    with pytest.raises(ValueError, match="before batched-vrr staged"):
        sh.apply(DEFAULT_HOOKS, spec)
    staged = bv.apply(DEFAULT_HOOKS, spec)
    for hook in ("tap", "digest", "predigest"):
        moved = dataclasses.replace(staged, **{hook: lambda ctx: ()})
        with pytest.raises(ValueError, match="dropped silently"):
            sh.apply(moved, spec)


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_literal_and_the_batched_kernels_are_untouched(cls) -> None:
    for kern in (build_f_kernel(cls, eris=True),
                 build_f_kernel(f_class_spec(*cls, (bv.NAME,)), eris=True)):
        assert not any(isinstance(s, For) and s.var == sh.SLICE
                       for s in walk(kern.body))
    assert f_levers.hooks_for(f_class_spec(*cls, (bv.NAME,))).is_default(
        "contracted")


# ---------------------------------------------------------------------------
# 2. The slices, in hgp.hrr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_slices_partition_the_targets_contiguously_per_bra_pair(cls
                                                                    ) -> None:
    tg = hrr.targets(cls)
    cut = hrr.slice_targets(cls)
    na, nb, nc, nd = (len(vrr.cartesians(x)) for x in cls)
    assert len(cut) == na * nb == sh.n_slices(cls)
    assert tuple(t for s in cut for t in s) == tg  # contiguous, in order
    for s, run in enumerate(cut):
        a, b = vrr.cartesians(cls[0])[s // nb], vrr.cartesians(cls[1])[s % nb]
        assert len(run) == nc * nd
        assert all(t[0] == a and t[1] == b for t in run)


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_a_slice_is_its_closure_in_the_parents_order(cls) -> None:
    dag = hrr.build(cls)
    covered: set = set()
    for run, sub in zip(hrr.slice_targets(cls), hrr.slices(cls), strict=True):
        assert sub.targets == run and (sub.cls, sub.first) == (dag.cls,
                                                               dag.first)
        assert sub.nodes == hrr.closure(dag.terms, run)
        assert all(sub.terms[x] == dag.terms[x] for x in sub.nodes)
        assert sub.order == tuple(x for x in dag.order if x in sub.nodes)
        seen: set = set()
        for x in sub.order:  # topological
            assert all(t.child in seen for t in sub.terms[x])
            seen.add(x)
        assert sub.boundary <= dag.boundary
        covered |= sub.nodes
    assert covered == dag.nodes  # every node is some ERI's ancestor
    assert sum(s.n_statements for s in hrr.slices(cls)) >= dag.n_statements


def test_a_given_slice_order_is_checked() -> None:
    dag = hrr.build((0, 1, 0, 1))
    run = hrr.slice_targets((0, 1, 0, 1))[1]
    sub = hrr.sub_dag(dag, run)
    with pytest.raises(ValueError, match="precedes a node it reads"):
        hrr.sub_dag(dag, run, order=tuple(reversed(sub.order)))
    with pytest.raises(ValueError, match="not the closure"):
        hrr.sub_dag(dag, run, order=sub.order[1:])
    # another topological order is admitted (#361's shape): the leaves
    # reversed, then the interior in the parent's order
    leaves = [x for x in sub.order if hrr.is_boundary(x)]
    alt = (*reversed(leaves), *(x for x in sub.order
                                if not hrr.is_boundary(x)))
    assert alt != sub.order
    assert hrr.sub_dag(dag, run, order=alt).order == alt
    with pytest.raises(ValueError, match="targets"):
        hrr.sub_dag(dag, ())
    with pytest.raises(ValueError, match="targets"):
        hrr.sub_dag(dag, (((9, 9, 9),) * 4,))
    with pytest.raises(ValueError, match="targets"):
        hrr.sub_dag(dag, (run[0], run[0]))


def test_hrr_peak_on_hand_checked_classes() -> None:
    # (ss|ss): one leaf, which is the one ERI
    assert hrr.hrr_peak(hrr.build((0, 0, 0, 0))) == 1
    # (ss|sp): three leaves (s|p), each read by one ERI and live out
    whole = hrr.build((0, 0, 0, 1))
    assert hrr.hrr_peak(whole) == bv.hrr_floor((0, 0, 0, 1)) == 5
    for cls in CAPPED:  # the whole DAG's peak is #350's floor
        assert hrr.hrr_peak(hrr.build(cls)) == bv.hrr_floor(cls)


def test_build_is_unchanged_by_the_slicing() -> None:
    for cls in CAPPED:
        before = hrr.build(cls)
        hrr.slices(cls)
        after = hrr.build(cls)
        assert (after.order, dict(after.terms), after.targets) == (
            before.order, dict(before.terms), before.targets)


# ---------------------------------------------------------------------------
# 3. The declared functions of the class
# ---------------------------------------------------------------------------

# class -> (n_slices, hrr_phase, the re-cut group count, the slices' HRR
# statements): `hgp.vrr.peak_live`'s model (f_levers/sliced_hrr.py). Pinned
# so a change to the DAGs' orders, the cut or #350's rule is a visible diff.
SLICE_TABLE = {
    (0, 0, 0, 0): (1, 1, 1, 1),
    (0, 0, 0, 1): (1, 5, 1, 7),
    (0, 0, 0, 2): (1, 11, 2, 25),
    (0, 0, 1, 1): (1, 11, 2, 18),
    (0, 0, 1, 2): (1, 25, 1, 60),
    (0, 0, 2, 2): (1, 45, 1, 110),
    (0, 1, 0, 1): (3, 9, 3, 45),
    (0, 1, 0, 2): (3, 21, 5, 135),
    (0, 1, 1, 1): (3, 19, 5, 108),
    (0, 1, 1, 2): (3, 39, 6, 294),
    (0, 1, 2, 2): (3, 63, 6, 516),
    (0, 2, 0, 2): (6, 41, 5, 480),
    (0, 2, 1, 1): (6, 37, 5, 405),
    (0, 2, 1, 2): (6, 77, 8, 987),
    (0, 2, 2, 2): (6, 125, 9, 1683),
    (1, 1, 1, 1): (9, 19, 13, 324),
    (1, 1, 1, 2): (9, 39, 16, 882),
    (1, 1, 2, 2): (9, 63, 19, 1548),
    (1, 2, 1, 2): (18, 77, 16, 2961),
    (1, 2, 2, 2): (18, 125, 20, 5049),
    (2, 2, 2, 2): (36, 125, 39, 10098),
}


def test_the_slice_table_covers_the_ladder_and_is_the_definition() -> None:
    assert tuple(SLICE_TABLE) == vrr.CLASSES
    for cls, (n, phase, groups, stmts) in SLICE_TABLE.items():
        na, nb = (len(vrr.cartesians(x)) for x in cls[:2])
        assert sh.n_slices(cls) == n == na * nb, cls
        assert sh.slice_peaks(cls) == tuple(
            hrr.hrr_peak(s) for s in hrr.slices(cls)), cls
        assert sh.hrr_phase(cls) == phase == max(sh.slice_peaks(cls)), cls
        assert phase <= bv.hrr_floor(cls), cls  # never above the whole HRR
        assert (n == 1) == (phase == bv.hrr_floor(cls)), cls
        assert sh.n_groups(cls) == groups == bv.rule(cls, floor=phase), cls
        assert groups >= bv.n_groups(cls), cls  # a lower floor, more groups
        assert sum(s.n_statements for s in hrr.slices(cls)) == stmts, cls


def test_an_override_of_350_s_count_wins_under_the_composition(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cls = (0, 1, 0, 1)
    monkeypatch.setattr(bv, "OVERRIDES", {cls: 2})
    assert sh.n_groups(cls) == 2
    assert _nests(build_f_kernel(lever_spec(cls))) == 2


# ---------------------------------------------------------------------------
# 4. The kernel's shape
# ---------------------------------------------------------------------------


def _nests(kern) -> int:
    return sum(1 for s in walk(kern.body)
               if isinstance(s, For) and s.var == seams.KCD)


def _hrr_decls(stmts) -> list[str]:
    return [s.name for s in stmts if isinstance(s, Decl)
            and s.name.startswith("H")]


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_one_scoped_slice_per_bra_pair_holding_its_hrr_tap_and_predigest(
        cls) -> None:
    kern = build_f_kernel(lever_spec(cls), eris=True)
    stmts = list(walk(kern.body))
    loops = [s for s in stmts if isinstance(s, For) and s.var == sh.SLICE]
    subs = hrr.slices(cls)
    assert len(loops) == len(subs) == sh.n_slices(cls)
    assert _nests(kern) == sh.n_groups(cls)
    inside = 0
    taps: list[int] = []
    for i, (loop, sub) in enumerate(zip(loops, subs, strict=True)):
        assert all(isinstance(x, Lit) for x in (loop.start, loop.stop,
                                                 loop.step))
        assert (loop.start.value, loop.stop.value, loop.step.value) == (
            i, i + 1, 1)
        body = list(walk(loop.body))
        names = _hrr_decls(body)
        assert names == [hrr_name(x) for x in sub.order]  # its nodes only
        inside += len(names)
        leaves = [s for s in body if isinstance(s, Decl) and isinstance(
            s.init, Index) and s.init.array == bv.ARRAY]
        assert len(leaves) == sub.n_boundary  # read from the staged array
        tap = [s.dst.at[2].value for s in body if isinstance(s, Assign)
               and isinstance(s.dst, Index) and s.dst.array == seams.OUT]
        assert tap == list(range(len(taps), len(taps) + sub.n_eri))
        taps += tap
        fock = [s for s in body if isinstance(s, Assign)
                and isinstance(s.dst, Var) and s.dst.name.startswith("f_")]
        assert len(fock) == len(seams.BLOCKS) * sub.n_eri
    assert len(_hrr_decls(stmts)) == inside  # no HRR node outside a slice
    assert taps == list(range(hrr.build(cls).n_eri))


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
def test_the_composition_is_the_literal_kernel_bitwise(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    kern = build_f_kernel(lever_spec(cls), eris=True)
    assert kern != build_f_kernel(cls, eris=True)
    _bitwise(cls, kern, monkeypatch)


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_slices_alone_on_350_s_own_count_are_bitwise(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    # the slicing separated from the re-cut: #350 at its unsliced count,
    # only `contracted` moved
    spec = lever_spec(cls)
    hooks = dataclasses.replace(bv.apply(DEFAULT_HOOKS, spec),
                                contracted=sh.contracted)
    kern = build_f_kernel(spec, eris=True, hooks=hooks)
    assert _nests(kern) == bv.n_groups(cls)
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
# 6. The goldens: (sp|sp) with both levers, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, \
#   ('batched-vrr', 'sliced-hrr'))); \
#   Path(f'test/golden/cuda/f_sliced_hrr/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); Path(f'test/golden/hip/f_sliced_hrr/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_sliced_hrr_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_f_kernel(lever_spec(GOLDEN_CLS))
    where = GOLDEN / backend / "f_sliced_hrr"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert f"double {bv.ARRAY}[16];" in text
    assert text.count("for (int kcd = 0;") == sh.n_groups(GOLDEN_CLS) == 3
    assert text.count(f"for (int {sh.SLICE} = ") == sh.n_slices(GOLDEN_CLS)
