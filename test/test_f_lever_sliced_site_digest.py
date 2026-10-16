"""The interleaved digest per slice, `f_levers/sliced_site_digest.py` (#359),
interpreted.

The lever digests each ERI of a #356 slice into the Fock buffers (and the
tap) right after the HRR statement that produces it, instead of after the
slice's whole HRR, in the predigest's order. What is asserted:

1. THE DECLARATION: `sliced-site-digest` is legal on `Target.F` only,
   refused on the other targets with stated reasons, registered in
   `f_levers.REGISTRY`; it is a COMPOSITION on #356: refused without
   `sliced-hrr` (alone, or on `batched-vrr` only, and by the CLI with exit
   2), refused when applied before #356 or on a `contracted` another lever
   moved; with both it moves the same hooks as #356 (`contracted` is its
   own body, `nest` the same count). #356's kernel and the literal kernel
   are untouched.
2. THE DECLARED FUNCTIONS OF THE CLASS, pinned for all 21 classes: within a
   slice the ERIs are the last nodes, in targets order (why the digest
   order is unchanged); the interleaved live set never above #356's, below
   it only at `(ss|pp)`; the group count at the interleaved floor is #356's.
3. THE KERNEL'S SHAPE: one slice per bra pair as in #356, the same HRR
   nodes, tap writes and multiply-adds, each ERI's six multiply-adds (and
   its tap write) directly after its declaration; `digest_points` delays,
   never reorders, and refuses an ERI not declared exactly once.
4. EXACTNESS: interpreted, `F` and the tap `g` are the literal kernel's
   and #356's BIT FOR BIT on every class with L <= 4 (milestone 13's cap;
   above it `slow`, not run under the cap): the per-buffer summation order
   is unchanged; one launch against `ao.fock`'s sweep to
   `test_kernel_f.TOL`, and the degeneracy cases within the cap.
5. THE GOLDENS: `(sp|sp)` with the three levers, one unit per backend.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import test_f_lever_sliced_hrr as tsh
import test_kernel_f as tk

from felizjueves import f_levers, levers
from felizjueves.ao.shell import Basis
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import seams
from felizjueves.f_levers import sliced_hrr as sh
from felizjueves.f_levers import sliced_site_digest as si
from felizjueves.f_levers.seams import FHooks
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import hrr_name
from felizjueves.ir import (
    Assign,
    Decl,
    For,
    Index,
    Lit,
    Note,
    Scalar,
    Var,
    walk,
)
from felizjueves.kernel_f import build_f_kernel, f_class_spec
from felizjueves.spec import Operator, Spec, Target

NAME = si.NAME
H2 = (bv.NAME, sh.NAME)
ALL = (bv.NAME, sh.NAME, NAME)
CAP = 4  # milestone 13: every test at L <= 4
LADDER = tsh.LADDER
CAPPED = tsh.CAPPED


def lever_spec(cls) -> Spec:
    return f_class_spec(*cls, ALL)


# ---------------------------------------------------------------------------
# 1. The declaration: a composition on #356
# ---------------------------------------------------------------------------


def test_sliced_site_digest_is_declared_legal_on_f_only_with_its_reasons(
        ) -> None:
    lever = levers.VOCABULARY[NAME]
    assert (lever.name, lever.issue, lever.legal) == (NAME, 359,
                                                      frozenset({"f"}))
    assert "predigest" in lever.refusals["j"]
    assert "predigest" in lever.refusals["k"]
    assert "sliced-hrr" in lever.what
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="slice digest"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (NAME,))
    with pytest.raises(ValueError, match="HRR slices"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))


def test_sliced_site_digest_is_registered_keyed_and_composes_on_356() -> None:
    assert f_levers.REGISTRY[NAME] is si.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == ("coulomb/d0/(0,1|0,1)/b1/f/"
                          "batched-vrr+sliced-hrr+sliced-site-digest")
    assert spec.kernel_body_levers() == ALL  # key order: #356 applied first
    hooks = f_levers.hooks_for(spec)
    moved = {f.name for f in dataclasses.fields(FHooks)
             if not hooks.is_default(f.name)}
    assert moved == {"boundary_init", "nest", "accumulate", "boundary_read",
                     "contracted"}
    assert hooks.contracted is si.contracted
    kern = build_f_kernel(spec)
    assert kern.name == "f_0101"
    notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
    for name in ALL:
        assert levers.provenance(name) in notes


@pytest.mark.parametrize("lv", [(NAME,), (bv.NAME, NAME)], ids=str)
def test_sliced_site_digest_without_sliced_hrr_is_refused(lv,
                                                         tmp_path) -> None:
    spec = f_class_spec(0, 1, 0, 1, lv)  # legal on F: the key admits it
    with pytest.raises(ValueError, match="requires sliced-hrr") as err:
        build_f_kernel(spec)
    assert si.REFUSAL in str(err.value)


def test_the_cli_flag_builds_the_lever_alone_and_is_refused(tmp_path) -> None:
    assert emit_cli.main(["--backend", "cuda", "--out", str(tmp_path),
                          "--klever-sliced-site-digest-fock-l", "1"]) == 2
    assert not list(tmp_path.iterdir())


def test_an_earlier_or_foreign_contracted_is_refused() -> None:
    spec = lever_spec((0, 1, 0, 1))
    staged = bv.apply(seams.DEFAULT_HOOKS, spec)
    with pytest.raises(ValueError, match="not sliced-hrr's own"):
        si.apply(staged, spec)  # before #356
    sliced = sh.apply(staged, spec)
    foreign = dataclasses.replace(
        sliced, contracted=sh.contracted_with(sh.slice_body))
    with pytest.raises(ValueError, match="dropped silently"):
        si.apply(foreign, spec)  # another lever's slice body
    with pytest.raises(ValueError, match="not sliced-hrr's own"):
        si.apply(si.apply(sliced, spec), spec)  # applied twice


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_356_and_the_literal_kernel_are_untouched(cls) -> None:
    assert f_levers.hooks_for(f_class_spec(*cls, H2)).contracted is (
        sh.contracted)
    assert f_levers.hooks_for(f_class_spec(*cls, ())).is_default("contracted")


# ---------------------------------------------------------------------------
# 2. The declared functions of the class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", vrr.CLASSES, ids=str)
def test_within_a_slice_the_eris_are_its_last_nodes_in_targets_order(cls
                                                                     ) -> None:
    for sub in hrr.slices(cls):
        assert sub.order[-sub.n_eri:] == sub.targets


# class -> the interleaved phase (max over slices); #356's is
# test_f_lever_sliced_hrr.SLICE_TABLE's second column.
LOWER = {(0, 0, 1, 1): 10}


def test_the_interleaved_phase_and_group_count_against_356() -> None:
    for cls in vrr.CLASSES:
        ip, sp = si.interleaved_peaks(cls), sh.slice_peaks(cls)
        assert len(ip) == len(sp) == sh.n_slices(cls), cls
        assert all(a <= b for a, b in zip(ip, sp, strict=True)), cls
        phase = tsh.SLICE_TABLE[cls][1]
        assert si.interleaved_phase(cls) == LOWER.get(cls, phase), cls
        assert si.n_groups(cls) == sh.n_groups(cls), cls  # a null on #350


# ---------------------------------------------------------------------------
# 3. The kernel's shape
# ---------------------------------------------------------------------------


def _is_fock(s) -> bool:
    return (isinstance(s, Assign) and isinstance(s.dst, Var)
            and s.dst.name.startswith("f_"))


def _is_tap(s) -> bool:
    return (isinstance(s, Assign) and isinstance(s.dst, Index)
            and s.dst.array == seams.OUT)


def _loops(kern):
    return [s for s in walk(kern.body) if isinstance(s, For)
            and s.var == sh.SLICE]


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_each_eri_is_digested_right_after_its_production(cls) -> None:
    kern = build_f_kernel(lever_spec(cls), eris=True)
    base = build_f_kernel(f_class_spec(*cls, H2), eris=True)
    loops, loops0 = _loops(kern), _loops(base)
    assert len(loops) == len(loops0) == sh.n_slices(cls)
    assert tsh._nests(kern) == tsh._nests(base) == si.n_groups(cls)
    nb = len(seams.BLOCKS)
    for loop, loop0, sub in zip(loops, loops0, hrr.slices(cls), strict=True):
        body, body0 = list(loop.body), list(loop0.body)
        # the same statements, only moved: HRR nodes, taps, multiply-adds
        assert sorted(map(repr, body)) == sorted(map(repr, body0))
        assert [s for s in body if not (_is_fock(s) or _is_tap(s))] == [
            s for s in body0 if not (_is_fock(s) or _is_tap(s))]
        assert [s for s in body if _is_fock(s)] == [
            s for s in body0 if _is_fock(s)]  # the same order per buffer
        for t in sub.targets:
            i = next(j for j, s in enumerate(body) if isinstance(s, Decl)
                     and s.name == hrr_name(t))
            assert _is_tap(body[i + 1])
            assert all(_is_fock(s) for s in body[i + 2:i + 2 + nb])
            assert body[i + 1].value == Var(hrr_name(t))
            assert all(s.value.args[1].args[1] == Var(hrr_name(t))
                       for s in body[i + 2:i + 2 + nb])
        # and #356's places every multiply-add after the slice's last node
        last = max(j for j, s in enumerate(body0) if isinstance(s, Decl))
        assert all(j > last for j, s in enumerate(body0) if _is_fock(s))


def test_digest_points_delay_never_reorder_and_refuse_a_missing_eri(
        ) -> None:
    t = [((0, 0, 0),) * 4, ((1, 0, 0),) * 4]
    d = [Decl(hrr_name(x), Scalar.F64, init=Lit(0.0)) for x in t]
    pad = Note("x")
    assert si.digest_points((d[0], pad, d[1]), t) == (0, 2)
    assert si.digest_points((d[1], pad, d[0]), t) == (2, 2)  # delayed
    with pytest.raises(ValueError, match="0 times"):
        si.digest_points((d[0],), t)
    with pytest.raises(ValueError, match="2 times"):
        si.digest_points((d[0], d[1], d[1]), t)


# ---------------------------------------------------------------------------
# 4. Exactness: the literal kernel and #356 bit for bit, and the sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", LADDER)
def test_the_composition_is_the_literal_kernel_bitwise(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    kern = build_f_kernel(lever_spec(cls), eris=True)
    assert kern != build_f_kernel(f_class_spec(*cls, H2), eris=True)
    tsh._bitwise(cls, kern, monkeypatch)


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_composition_is_356_bitwise(cls,
                                        monkeypatch: pytest.MonkeyPatch
                                        ) -> None:
    F, g = tsh._launch(build_f_kernel(lever_spec(cls), eris=True), cls,
                       monkeypatch)
    F0, g0 = tsh._launch(build_f_kernel(f_class_spec(*cls, H2), eris=True),
                         cls, monkeypatch)
    assert np.any(F0 != 0) and np.any(g0 != 0)
    assert np.array_equal(F, F0) and np.array_equal(g, g0)


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
# 5. The goldens: (sp|sp) with the three levers, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, \
#   ('batched-vrr', 'sliced-hrr', 'sliced-site-digest'))); \
#   Path(f'test/golden/cuda/f_sliced_site_digest/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); \
#   Path(f'test/golden/hip/f_sliced_site_digest/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_sliced_site_digest_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_f_kernel(lever_spec(GOLDEN_CLS))
    where = GOLDEN / backend / "f_sliced_site_digest"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    h2 = emit(build_f_kernel(f_class_spec(*GOLDEN_CLS, H2)), arch)
    # #356's text with the HRR lines moved and the provenance note added
    mine, theirs = Counter(text.splitlines()), Counter(h2.splitlines())
    assert text != h2 and not theirs - mine
    assert all("//" in line for line in mine - theirs)
    assert text.count("for (int kcd = 0;") == si.n_groups(GOLDEN_CLS) == 3
    assert text.count(f"for (int {sh.SLICE} = ") == sh.n_slices(GOLDEN_CLS)
