"""Slice-interior sub-slicing, `f_levers/sub_slice.py` (#361), interpreted.

The lever cuts each of #356's slices `(a, b)` into `n_sub(cls)` sub-slices
-- contiguous runs of its ERIs in target order, each its own HRR sub-DAG
(`hgp.hrr.sub_slices`), tapped and digested into #360's reduced buffers
before the next. What is asserted, the issue's acceptance in order:

1. THE DECLARATION: `sub-slice` is legal on `Target.F` only, refused on the
   other targets with its own reasons, registered in `f_levers.REGISTRY`;
   it is a COMPOSITION on `batched-vrr+sliced-hrr+slim-fock`: refused
   without `slim-fock` (with its reason, and by the CLI with exit 2),
   refused when applied before #360 or after a lever that moved
   `contracted`; it composes with #351's `block-fab`, #352's `ket-break`
   and #353's `struct-reads`. It moves `contracted` only.
2. THE DECLARED FUNCTIONS OF THE CLASS, pinned for all 21 classes: `n_sub`,
   the slice peak (#356's `hrr_phase`) against the sub-slice peak, and the
   HRR statements per quartet before and after (the recompute). The
   additive `hgp.hrr` functions partition each slice's ERIs in order.
3. THE KERNEL'S SHAPE: `n_sub` sub-slice scopes per slice, each with its
   sub-DAG's HRR nodes, no Fock buffer declared in a sub-slice, and the
   `AtomicAdd` statements exactly #360's in number and order.
4. PARITY, interpreted, on EVERY F class with L <= 4 (milestone 13's cap;
   above it `slow`, not run under the cap): `F` and the tap `g` equal
   #360's kernel BIT FOR BIT (same products, same order); `F` against the
   literal kernel by #360's DERIVED re-association bound over the leaves
   (a witness kernel atomically adding every product as it is formed);
   `F` against `ao.fock`'s sweep at the literal kernel's own `TOL`, and
   every degeneracy case within the cap.
5. THE ATOMIC TRAFFIC AS A COUNT: unchanged from #360,
   `slim_fock.ket_atomics(cls, lever=True)` per worked quartet.
6. THE GOLDENS: `(sp|sp)` with the four levers, one unit per backend.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_f_lever_slim_fock as tsf
import test_kernel_f as tk
from tiers import is_fast, split_by_l

from felizjueves import levers
from felizjueves.ao.shell import Basis
from felizjueves.emit import __main__ as emit_cli
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock import levers as f_levers
from felizjueves.eri.fock.kernel_f import build_f_kernel, f_class_spec
from felizjueves.eri.fock.levers import batched_vrr as bv
from felizjueves.eri.fock.levers import block_fab, ket_break, seams, struct_reads
from felizjueves.eri.fock.levers import sliced_hrr as sh
from felizjueves.eri.fock.levers import slim_fock as sf
from felizjueves.eri.fock.levers import sub_slice as ss
from felizjueves.eri.fock.levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.hgp import hrr, vrr
from felizjueves.ir import AtomicAdd, Decl, For, walk
from felizjueves.spec import Operator, Spec, Target

NAME = ss.NAME
H4 = (bv.NAME, sh.NAME, sf.NAME)
STACK = (*H4, NAME)
CAP = 4
LADDER = tsf.LADDER
CAPPED = tsf.CAPPED


def lever_spec(cls, extra=()) -> Spec:
    return f_class_spec(*cls, tuple(sorted((*STACK, *extra))))


def lever(cls, eris: bool = False, extra=()) -> KF.Kernel:
    return build_f_kernel(lever_spec(cls, extra), eris=eris)


def h4(cls, eris: bool = False, extra=()) -> KF.Kernel:
    return build_f_kernel(f_class_spec(*cls, tuple(sorted((*H4, *extra)))),
                          eris=eris)


def _moved(hooks: FHooks) -> set[str]:
    return {f.name for f in dataclasses.fields(FHooks)
            if not hooks.is_default(f.name)}


# ---------------------------------------------------------------------------
# 1. The declaration: a composition on #360
# ---------------------------------------------------------------------------


def test_sub_slice_is_declared_legal_on_f_only_with_its_own_reasons() -> None:
    lv = levers.VOCABULARY[NAME]
    assert (lv.name, lv.issue, lv.legal) == (NAME, 361, frozenset({"f"}))
    assert "split" not in NAME
    assert "sliced-HRR slice" in lv.refusals["j"]
    assert "sliced HGP HRR" in lv.refusals["k"]
    assert "slim-fock" in lv.what
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lv.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    for target in (Target.J, Target.K):
        with pytest.raises(ValueError, match="sliced"):
            Spec(Operator.COULOMB, 0, (1, 0, 1, 0), target, (NAME,))


def test_sub_slice_is_registered_keyed_and_sorts_after_slim_fock() -> None:
    assert f_levers.REGISTRY[NAME] is ss.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == ("coulomb/d0/(0,1|0,1)/b1/f/"
                          "batched-vrr+sliced-hrr+slim-fock+sub-slice")
    assert spec.kernel_body_levers() == STACK  # key order: #360 before it
    hooks = f_levers.hooks_for(spec)
    base = f_levers.hooks_for(f_class_spec(0, 1, 0, 1, H4))
    assert _moved(hooks) == _moved(base)
    assert hooks.contracted is ss.contracted
    mine = ss.apply(base, spec)  # on #360's very hooks: contracted only
    for f in dataclasses.fields(FHooks):
        if f.name != "contracted":
            assert getattr(mine, f.name) is getattr(base, f.name), f.name
    assert sh.n_groups((0, 1, 0, 1)) == 3  # the groups do not move
    kern = build_f_kernel(spec)
    assert kern.name == "f_0101"
    notes = [s.prov for s in walk(kern.body) if hasattr(s, "prov")]
    assert all(levers.provenance(n) in notes for n in STACK)


@pytest.mark.parametrize("alone", [(NAME,), (bv.NAME, NAME),
                                   (bv.NAME, sh.NAME, NAME)],
                         ids=["alone", "on-batched-vrr", "on-sliced-hrr"])
def test_sub_slice_without_slim_fock_is_refused_with_its_reason(
        alone, tmp_path) -> None:
    spec = f_class_spec(0, 1, 0, 1, alone)  # legal on F: the key admits it
    with pytest.raises(ValueError, match="requires") as err:
        build_f_kernel(spec)
    if sh.NAME in alone:  # the slices are there: this lever's own refusal
        assert ss.REFUSAL in str(err.value)
    assert emit_cli.main(["--backend", "cuda", "--out", str(tmp_path),
                          "--klever-sub-slice-fock-l", "1"]) == 2
    assert not list(tmp_path.iterdir())


def test_an_incoherent_order_or_a_moved_contracted_is_refused() -> None:
    spec = lever_spec((0, 1, 0, 1))
    staged = bv.apply(DEFAULT_HOOKS, spec)
    with pytest.raises(ValueError, match="applied before slim-fock"):
        ss.apply(staged, spec)
    sliced = sh.apply(staged, spec)
    with pytest.raises(ValueError, match="applied before slim-fock"):
        ss.apply(sliced, spec)
    slim = sf.apply(sliced, spec)
    other = dataclasses.replace(slim, contracted=sf.contracted_over(
        sh.slice_body))  # another lever's contracted, even an equal one
    with pytest.raises(ValueError, match="replace silently"):
        ss.apply(other, spec)
    assert _moved(ss.apply(slim, spec)) == _moved(f_levers.hooks_for(spec))


@pytest.mark.parametrize("other", [block_fab.NAME, ket_break.NAME,
                                   struct_reads.NAME])
def test_it_composes_with_the_levers_that_move_other_hooks(other) -> None:
    spec = lever_spec((1, 1, 1, 1), (other,))
    alone = f_levers.hooks_for(f_class_spec(1, 1, 1, 1, (other,)))
    hooks = f_levers.hooks_for(spec)
    assert _moved(hooks) == _moved(f_levers.hooks_for(lever_spec(
        (1, 1, 1, 1)))) | _moved(alone)
    assert hooks.contracted is ss.contracted
    build_f_kernel(spec)  # builds


# ---------------------------------------------------------------------------
# 2. The declared functions of the class
# ---------------------------------------------------------------------------

# class -> (n_sub, slice peak (#356's hrr_phase), sub-slice peak, HRR
# statements per quartet sliced, sub-sliced). Pinned so a change to the cut
# is a visible diff.
SUB_TABLE = {
    (0, 0, 0, 0): (1, 1, 1, 1, 1),
    (0, 0, 0, 1): (3, 5, 3, 7, 9),
    (0, 0, 0, 2): (6, 11, 5, 25, 39),
    (0, 0, 1, 1): (3, 11, 5, 18, 21),
    (0, 0, 1, 2): (3, 25, 11, 60, 75),
    (0, 0, 2, 2): (6, 45, 11, 110, 150),
    (0, 1, 0, 1): (3, 9, 5, 45, 63),
    (0, 1, 0, 2): (6, 21, 9, 135, 243),
    (0, 1, 1, 1): (3, 19, 9, 108, 135),
    (0, 1, 1, 2): (3, 39, 21, 294, 405),
    (0, 1, 2, 2): (6, 63, 21, 516, 810),
    (0, 2, 0, 2): (6, 41, 17, 480, 927),
    (0, 2, 1, 1): (3, 37, 17, 405, 522),
    (0, 2, 1, 2): (3, 77, 41, 987, 1440),
    (0, 2, 2, 2): (6, 125, 41, 1683, 2880),
    (1, 1, 1, 1): (3, 19, 9, 324, 405),
    (1, 1, 1, 2): (3, 39, 21, 882, 1215),
    (1, 1, 2, 2): (6, 63, 21, 1548, 2430),
    (1, 2, 1, 2): (3, 77, 41, 2961, 4320),
    (1, 2, 2, 2): (6, 125, 41, 5049, 8640),
    (2, 2, 2, 2): (6, 125, 41, 10098, 17280),
}


def test_the_sub_slice_table_covers_the_ladder_and_is_the_definition(
        ) -> None:
    assert tuple(SUB_TABLE) == vrr.CLASSES
    assert not ss.OVERRIDES
    _sub_slice_table_is_the_definition(c for c in vrr.CLASSES if is_fast(sum(c)))
    # (dd|dd): the slice peak 125 -> 41, 3.05x; HRR recompute 1.71x
    assert round(125 / 41, 2) == 3.05


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_sub_slice_table_is_the_definition_above_l4() -> None:
    _sub_slice_table_is_the_definition(c for c in vrr.CLASSES if not is_fast(sum(c)))


def _sub_slice_table_is_the_definition(classes) -> None:
    for cls in classes:
        n, slice_pk, sub_pk, st0, st1 = SUB_TABLE[cls]
        nc, nd = (len(vrr.cartesians(x)) for x in cls[2:])
        assert ss.n_sub(cls) == ss.rule(cls) == n == (nc if nc > 1 else nd)
        assert sh.hrr_phase(cls) == slice_pk, cls
        assert ss.sub_phase(cls) == sub_pk <= slice_pk, cls
        assert sub_pk < slice_pk or n == 1, cls  # every real cut lowers it
        assert len(ss.sub_peaks(cls)) == sh.n_slices(cls) * n, cls
        assert ss.statements(cls, lever=False) == st0, cls
        assert ss.statements(cls, lever=True) == st1 >= st0, cls


@pytest.mark.parametrize("cls", split_by_l(vrr.CLASSES, sum, ids=str))
def test_the_sub_slices_partition_each_slice_in_order(cls) -> None:
    n = ss.n_sub(cls)
    whole = hrr.build(cls)
    for sl, parts in zip(hrr.slices(cls), hrr.sub_slices(cls, n),
                         strict=True):
        assert len(parts) == n
        assert sum((p.targets for p in parts), ()) == sl.targets
        for p in parts:  # each a closed sub-DAG of the parent, its order
            assert all(p.terms[x] == whole.terms[x] for x in p.order)
            assert p.order == tuple(x for x in whole.order if x in p.terms)
        assert set().union(*(p.nodes for p in parts)) == set(sl.nodes)


def test_runs_refuses_a_bad_count() -> None:
    seq = hrr.targets((0, 0, 1, 1))
    assert [len(r) for r in hrr.runs(seq, 4)] == [2, 2, 3, 2]
    for bad in (0, len(seq) + 1, 2.0):
        with pytest.raises(ValueError, match="run count"):
            hrr.runs(seq, bad)


# ---------------------------------------------------------------------------
# 3. The kernel's shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_each_slice_holds_n_sub_scopes_and_the_atomics_are_h4s(cls) -> None:
    kern, base = lever(cls), h4(cls)
    n = ss.n_sub(cls)
    slices = [s for s in walk(kern.body) if isinstance(s, For)
              and s.var == sh.SLICE]
    assert len(slices) == sh.n_slices(cls)
    parts = hrr.sub_slices(cls, n)
    for loop, dags in zip(slices, parts, strict=True):
        subs = [s for s in loop.body if isinstance(s, For)
                and s.var == ss.SUB]
        assert len(subs) == n
        for sub, dag in zip(subs, dags, strict=True):
            decls = [s.name for s in walk(sub.body) if isinstance(s, Decl)]
            assert not any(d.startswith("f_") for d in decls)  # no buffer
            assert len(decls) == dag.n_statements, (cls, len(decls))

    def atomics(k):
        return [(s.dst, s.value) for s in walk(k.body)
                if isinstance(s, AtomicAdd)]

    assert atomics(kern) == atomics(base)
    # the groups do not move: #360's nest, statement for statement
    def nests(k):
        return [s for s in walk(k.body) if isinstance(s, For)
                and s.var == seams.KCD]

    assert nests(kern) == nests(base) and len(nests(kern)) == sh.n_groups(cls)


# ---------------------------------------------------------------------------
# 4. Parity; 5. the atomic count
# ---------------------------------------------------------------------------


def witness(cls) -> KF.Kernel:
    """This lever's kernel with every product streamed and no buffer
    flushed (#360's witness, on the sub-slices): the leaves of `F`."""
    spec = lever_spec(cls)
    hooks = dataclasses.replace(
        f_levers.hooks_for(spec), predigest_eri=tsf._stream_all,
        zero_buffers=lambda ctx: (), flush_ket=lambda ctx: (),
        flush_bra=lambda ctx: ())
    return build_f_kernel(spec, hooks=hooks)


@pytest.mark.parametrize("cls", LADDER)
def test_parity_with_h4_the_literal_kernel_and_the_sweep(cls,
                                                         monkeypatch) -> None:
    F0, g0, _ = tsf._run(build_f_kernel(cls, eris=True), cls, monkeypatch,
                         eris=True)
    F4, g4, rec4 = tsf._run(h4(cls, eris=True), cls, monkeypatch, eris=True,
                            record=True)
    F1, g1, rec1 = tsf._run(lever(cls, eris=True), cls, monkeypatch,
                            eris=True, record=True)
    _, _, leaves = tsf._run(witness(cls), cls, monkeypatch, record=True)
    assert np.any(g0 != 0) and np.max(np.abs(F0)) > 0
    # 4a. #360's kernel, bit for bit: the ERIs and F
    assert np.array_equal(g1, g4) and np.array_equal(g1, g0)
    assert np.array_equal(F1, F4)
    # 4b. F against the literal kernel by #360's derived bound
    bound = tsf._bound(leaves)
    assert np.all(np.abs(F1 - F0) <= bound), float(np.max(
        np.abs(F1 - F0) - bound))
    # 4c. against the sweep, at the literal kernel's own TOL
    basis, bras, kets = tsf._two_by_two(cls)
    want, scale = tk.sweep(basis, tk._density(basis.n_ao), tk._members(
        [r.sh for r in bras], [r.sh for r in kets]))
    assert tk._worst(np.array(KF.close(F1)), want, scale) <= tk.TOL
    # 5. the atomic traffic is #360's, element by element
    assert np.array_equal(rec1.count, rec4.count)
    na, nb = (len(vrr.cartesians(x)) for x in cls[:2])
    bra_flush = len(bras) * 2 * na * nb
    assert int(rec1.count.sum()) - bra_flush == len(bras) * len(
        kets) * sf.ket_atomics(cls, lever=True)


@pytest.mark.parametrize("cls", [(0, 1, 0, 1), (1, 1, 1, 1)], ids=str)
def test_with_block_fab_it_is_h4_with_block_fab_bitwise(cls,
                                                        monkeypatch) -> None:
    F4, _, _ = tsf._run(h4(cls, extra=(block_fab.NAME,)), cls, monkeypatch,
                        threads=32)
    F1, _, _ = tsf._run(lever(cls, extra=(block_fab.NAME,)), cls,
                        monkeypatch, threads=32)
    assert np.max(np.abs(F4)) > 0 and np.array_equal(F1, F4)


def _against_sweep(cls, basis, bra_ix, ket_ix, diag, monkeypatch):
    kern = lever(cls)
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    worst, *_ = tk._check_launch(cls, basis, tk._density(basis.n_ao), bra_ix,
                                 ket_ix, diag=diag)
    return worst


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_one_orbit_equals_its_ordered_members_in_the_sweep(
        cls, monkeypatch) -> None:
    basis = Basis(tuple(tk._shells(cls)))
    worst = _against_sweep(cls, basis, [(0, 1)], [(2, 3)], 0, monkeypatch)
    assert worst <= tk.TOL, worst


@pytest.mark.parametrize("case", [c for c in tk.COINCIDENCE
                                  if sum(tk.COINCIDENCE[c][0]) <= CAP])
def test_every_degeneracy_case_within_the_cap_equals_the_sweep(
        case, monkeypatch) -> None:
    cls, n, bra_ix, ket_ix, diag = tk.COINCIDENCE[case]
    basis = Basis(tuple(tk._same_l_shells(cls[0], n)))
    worst = _against_sweep(cls, basis, bra_ix, ket_ix, diag, monkeypatch)
    assert worst <= tk.TOL, (case, worst)


# ---------------------------------------------------------------------------
# 6. The goldens: (sp|sp) with the four levers, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.eri.fock.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, \
#   ('batched-vrr', 'sliced-hrr', 'slim-fock', 'sub-slice'))); \
#   Path(f'test/golden/cuda/f_sub_slice/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); Path(f'test/golden/hip/f_sub_slice/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)


@pytest.mark.parametrize("backend", sorted(tsf.EMITTERS))
def test_the_sub_slice_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = tsf.EMITTERS[backend]
    kern = lever(GOLDEN_CLS)
    where = GOLDEN / backend / "f_sub_slice"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert text.count(f"for (int {sh.SLICE} = ") == sh.n_slices(GOLDEN_CLS)
    assert text.count(f"for (int {ss.SUB} = ") == sh.n_slices(
        GOLDEN_CLS) * ss.n_sub(GOLDEN_CLS)
    assert "f_cd_" not in text
