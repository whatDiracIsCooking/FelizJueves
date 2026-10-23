"""Mixed-precision fused F, `f_levers/mixed.py` (#355), interpreted.

The lever `fmixed` makes the fused F kernel's VRR, contracted boundary and
HRR FP32 (Boys in fp64 on `TO_F64(T)`, narrowed), and keeps the screen, the
density reads, the six Fock buffers, the parameters and the atomics FP64.
What is asserted, nothing compiled or launched:

1. THE DECLARATION: `fmixed` is a kernel-body lever legal on `Target.F`
   only, refused on J and K with reasons naming their own precision levers
   (#346, #347), keyed, registered in `f_levers.REGISTRY`, and moving
   exactly `real`, `cast`, `quartet_scalars`, `vrr`, `hrr` and `eri`.
2. THE FP64 ARM IS UNTOUCHED: no lever, no F32 anywhere and
   `DEFAULT_HOOKS` (the literal kernels' digests are test_f_seams.py's pin
   and their goldens test_emit_fock.py's, unchanged by this lever).
3. THE FP64 BOUNDARY, AS IR, every class with L <= 4: the parameters are
   the literal ones (no `const float*`); the F64 locals are exactly the
   "fock" region's (`Gb`, `d_ab`, the screen, `deg`, `hb` / `hk`, `s_j` /
   `s_k`, the weighted densities, the six buffers) and every other floating
   local is F32 -- the VRR, the boundary, the HRR and their scalars; every
   `BOYS` reads `TO_F64(T)`; every `TO_F64` widens an HRR node (the ERI
   into the digestion) or `T`; every AtomicAdd stores an F64 buffer.
   Dropping the ERI's widening makes the kernel refuse to build (#338's
   checker holds the boundary, not this lever).
4. THE NUMBERS, at THE fp32 tier, `F32_REL` of `scale_G`
   (test/oracle/tolerance.py's one declared fp32 tier, never widened):
   one orbit against `ao.fock`'s sweep on every class with L <= 4, the
   degeneracy cases within the cap, and (`slow`: 7 s and more, interpreted)
   H2O/STO-3G's and 6-31G's (s and p shells only) whole `F` from
   `launch_f`'s plan (every kernel the lever's) against the sweep and
   libcint's `vj - vk/2`; each comparison is also shown to MISS the fp64
   tier (`test_kernel_f.TOL`), so the build really is fp32. The interpreter
   rounds every F32 operation where the device does (`interp._round32`).
5. COMPOSITION with #350's `batched-vrr`: both levers on one spec build,
   the staged `boundary[]` is F32, and `F` is the `fmixed` kernel's BIT FOR
   BIT (batched-vrr reorders no sum; each group runs the same narrowed VRR).
6. THE GOLDENS: `(sp|sp)` with the lever, one unit per backend.

WHY `F32_REL` AND NOT `assert_within`: `F` is an F64 array by the boundary
(the six buffers and the atomics stay fp64) while the ERIs are F32; item 3
is what shows the build is not an fp64 one (test_k_lever_mixed.py says the
same for K).

Fast tier unless `slow`: no compiler, no card, no PySCF. The whole-molecule
check is the device tier's in the PR gate (test_f_lever_mixed_device.py's
H2O/STO-3G cells).
"""

from __future__ import annotations

import dataclasses
import re
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_fock_build as fb
import test_kernel_f as tk
from oracle.tolerance import F32_REL

from felizjueves import f_levers, levers
from felizjueves.ao.shell import Basis
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import batched_vrr as bv
from felizjueves.f_levers import mixed
from felizjueves.f_levers.seams import DEFAULT_HOOKS, FHooks
from felizjueves.hgp import vrr
from felizjueves.ir import (
    AtomicAdd,
    Boys,
    Call,
    Decl,
    Fn,
    Index,
    Kernel,
    Lit,
    Note,
    PrecisionError,
    Scalar,
    Var,
    _subexprs,
    walk,
)
from felizjueves.kernel_f import build_f_kernel, f_class_spec
from felizjueves.launch_f import f_launches
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Operator, Spec, Target
from felizjueves.walk_f import walk_f

NAME = mixed.NAME
CAP = 4  # milestone 13: every test at L <= 4
CAPPED = [c for c in vrr.CLASSES if sum(c) <= CAP]


def lever_spec(cls, more: tuple[str, ...] = ()) -> Spec:
    return f_class_spec(*cls, tuple(sorted((NAME, *more))))


@cache
def lever_kernel(cls, eris: bool = False) -> Kernel:
    return build_f_kernel(lever_spec(tuple(cls)), eris=eris)


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_fmixed_is_declared_legal_on_f_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[NAME]
    assert (lever.name, lever.issue, lever.legal) == (NAME, 355,
                                                      frozenset({"f"}))
    assert "#346" in lever.refusals["j"]
    assert "#347" in lever.refusals["k"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="#346"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (NAME,))
    with pytest.raises(ValueError, match="#347"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, (NAME,))
    # J's and K's precision levers stay theirs: F refuses them
    for other in ("mixed", "kmixed", "kmixed-de"):
        with pytest.raises(ValueError):
            f_class_spec(0, 1, 0, 1, (other,))


def test_fmixed_is_registered_keyed_and_moves_six_hooks() -> None:
    assert f_levers.REGISTRY[NAME] is mixed.apply
    spec = lever_spec((0, 1, 0, 1))
    assert spec.key() == "coulomb/d0/(0,1|0,1)/b1/f/fmixed"
    assert spec.kernel_body_levers() == (NAME,)
    moved = {f.name for f in dataclasses.fields(FHooks)
             if not f_levers.hooks_for(spec).is_default(f.name)}
    assert moved == {"real", "cast", "quartet_scalars", "vrr", "hrr", "eri"}
    kern = build_f_kernel(spec)
    assert kern.name == "f_0101"
    assert levers.provenance(NAME) in [s.prov for s in walk(kern.body)
                                       if isinstance(s, Note)]
    assert mixed.NOTE in [s.prov for s in walk(kern.body)
                          if isinstance(s, Note)]


# ---------------------------------------------------------------------------
# 2. The literal kernel is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_literal_kernel_declares_no_f32(cls) -> None:
    assert f_levers.hooks_for(f_class_spec(*cls)) is DEFAULT_HOOKS
    kern = build_f_kernel(cls, eris=True)
    assert all(s.type is not Scalar.F32 for s in walk(kern.body)
               if isinstance(s, Decl))
    assert all(p.type is not Scalar.F32 for p in kern.params)


# ---------------------------------------------------------------------------
# 3. The FP64 boundary, as IR
# ---------------------------------------------------------------------------

# The "fock" region's locals, by name: what stays F64.
FOCK_LOCAL = re.compile(
    r"Gb|bound|dmax(_[a-d]{2})?|deg|hb|hk|s_j|s_k|d_ab_\d+_\d+"
    r"|[fw]_[a-d]{2}_\d+_\d+")


def _calls(kern: Kernel, fn: Fn) -> list[Call]:
    out = []
    for s in walk(kern.body):
        for root in (getattr(s, "init", None), getattr(s, "value", None)):
            if root is not None:
                out += [x for x in _subexprs(root)
                        if isinstance(x, Call) and x.fn is fn]
    return out


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_the_fp64_boundary_is_the_fock_region(cls) -> None:
    kern = lever_kernel(cls, eris=True)
    lit = build_f_kernel(cls, eris=True)
    assert kern.params == lit.params  # no const float*: D, Dsh, F stay fp64
    floats = [s for s in walk(kern.body) if isinstance(s, Decl)
              and s.type in (Scalar.F64, Scalar.F32) and not s.shape]
    f64 = {s.name for s in floats if s.type is Scalar.F64}
    f32 = {s.name for s in floats if s.type is Scalar.F32}
    assert f64 and f32
    assert all(FOCK_LOCAL.fullmatch(n) for n in f64), sorted(
        n for n in f64 if not FOCK_LOCAL.fullmatch(n))
    assert not any(FOCK_LOCAL.fullmatch(n) for n in f32)
    # the literal kernel's floating locals, partitioned by name and type
    lit_names = {s.name for s in walk(lit.body) if isinstance(s, Decl)
                 and s.type is Scalar.F64 and not s.shape}
    assert f64 | f32 == lit_names
    assert {"T", "S", "theta", "pair"} <= f32
    assert any(n.startswith("V") for n in f32)  # the VRR
    assert any(n.startswith("C") for n in f32)  # the boundary
    assert any(n.startswith("H") for n in f32)  # the HRR
    # Boys: an fp64 array filled from TO_F64(T), each order narrowed at
    # its read
    fills = [s for s in walk(kern.body) if isinstance(s, Boys)]
    assert fills and all(b.t == Call(Fn.TO_F64, (Var("T"),)) for b in fills)
    (arr,) = [s for s in walk(kern.body)
              if isinstance(s, Decl) and s.name == fills[0].dst]
    assert arr.type is Scalar.F64
    narrowed = {x.args[0] for x in _calls(kern, Fn.TO_F32)
                if isinstance(x.args[0], Index)}
    assert narrowed and {e.array for e in narrowed} >= {fills[0].dst}
    # every widening is an ERI into the digestion
    widened = {x.args[0].name for x in _calls(kern, Fn.TO_F64)}
    assert widened and all(n.startswith("H") for n in widened)
    # the atomics store fp64 buffers
    adds = [s for s in walk(kern.body) if isinstance(s, AtomicAdd)]
    assert adds and all(isinstance(s.value, Var) and s.value.name in f64
                        for s in adds)


def test_dropping_the_eri_widening_refuses_the_build() -> None:
    hooks = f_levers.hooks_for(lever_spec((0, 1, 0, 1)))
    bare = dataclasses.replace(hooks, eri=DEFAULT_HOOKS.eri)
    with pytest.raises(PrecisionError, match="F32 with F64"):
        build_f_kernel((0, 1, 0, 1), hooks=bare)


def test_narrow_sources_casts_reads_and_literals_only() -> None:
    e =Call(Fn.SUB, (Lit(0.0), Call(Fn.MUL, (Var("x"), Index("A", (Lit(1),))))))
    got = mixed.narrow_sources(e)
    assert got == Call(Fn.SUB, (Call(Fn.TO_F32, (Lit(0.0),)), Call(
        Fn.MUL, (Var("x"), Call(Fn.TO_F32, (Index("A", (Lit(1),)),))))))
    with pytest.raises(ValueError, match="already cast"):
        mixed.narrow_sources(Call(Fn.TO_F64, (Var("x"),)))


# ---------------------------------------------------------------------------
# 4. The numbers, at the fp32 tier
# ---------------------------------------------------------------------------


def _against_sweep(cls, basis, D, bra_ix, ket_ix, diag, monkeypatch,
                   kern=None):
    kern = kern or lever_kernel(tuple(cls))
    monkeypatch.setattr(tk, "_kernel", lambda *_: kern)
    worst, G, *_ = tk._check_launch(cls, basis, D, bra_ix, ket_ix, diag=diag)
    return worst, G


@pytest.mark.parametrize("cls", CAPPED, ids=str)
def test_one_orbit_is_the_sweep_at_the_fp32_tier(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    basis = Basis(tuple(tk._shells(cls)))
    worst, _ = _against_sweep(cls, basis, tk._density(basis.n_ao), [(0, 1)],
                              [(2, 3)], 0, monkeypatch)
    assert worst <= F32_REL, worst
    assert worst > tk.TOL, worst  # and misses fp64's: the build is fp32


@pytest.mark.parametrize("case", [c for c in tk.COINCIDENCE
                                  if sum(tk.COINCIDENCE[c][0]) <= CAP])
def test_every_degeneracy_case_within_the_cap_is_the_sweep_at_fp32(
        case, monkeypatch: pytest.MonkeyPatch) -> None:
    cls, n, bra_ix, ket_ix, diag = tk.COINCIDENCE[case]
    basis = Basis(tuple(tk._same_l_shells(cls[0], n)))
    worst, _ = _against_sweep(cls, basis, tk._density(basis.n_ao), bra_ix,
                              ket_ix, diag, monkeypatch)
    assert worst <= F32_REL, (case, worst)


def _lever_run(kernel, grid, block, args):
    """test_fock_build's executor with every literal kernel swapped for the
    lever's kernel of its class."""
    m = re.fullmatch(r"f_(\d)(\d)(\d)(\d)", kernel.name)
    assert m, kernel.name
    cls = tuple(int(x) for x in m.groups())
    lev = dataclasses.replace(lever_kernel(cls), name=kernel.name)
    assert lev.params == kernel.params
    return fb.run(lev, grid, block, args)


@pytest.mark.slow  # 7 s at STO-3G, interpreted: past the fast tier's ~3 s
@pytest.mark.parametrize("name", ["h2o_sto3g", "h2o_631g"])
def test_a_whole_molecule_is_the_sweep_and_libcint_at_fp32(name) -> None:
    basis, D, libcint = fb.molecule(name)
    plan = f_launches(basis, D, bound=fb.BOUND, tau=0.0, n=1)
    got = np.array(walk_f(plan, basis, D, run=_lever_run).F)
    scale = fb.sweep(name, block="scale")
    worst = fb._worst(got, fb.sweep(name), scale)
    assert worst <= F32_REL, worst
    assert worst > fb.TOL, worst
    assert fb._worst(got, np.array(libcint), scale) <= F32_REL


# ---------------------------------------------------------------------------
# 5. Composition with batched-vrr (#350)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [c for c in CAPPED if sum(c) <= 2]
                         + [(0, 1, 1, 1), (1, 1, 1, 1)], ids=str)
def test_with_batched_vrr_it_is_the_fmixed_kernel_bitwise(
        cls, monkeypatch: pytest.MonkeyPatch) -> None:
    both = build_f_kernel(lever_spec(cls, (bv.NAME,)))
    staged = [s for s in walk(both.body)
              if isinstance(s, Decl) and s.name == bv.ARRAY]
    assert len(staged) == 1 and staged[0].type is Scalar.F32
    basis = Basis(tuple(tk._shells(cls)))
    D = tk._density(basis.n_ao)
    _, G1 = _against_sweep(cls, basis, D, [(0, 1)], [(2, 3)], 0, monkeypatch)
    _, G2 = _against_sweep(cls, basis, D, [(0, 1)], [(2, 3)], 0, monkeypatch,
                           kern=both)
    assert np.any(G1 != 0) and np.array_equal(G1, G2)


# ---------------------------------------------------------------------------
# 6. The goldens: (sp|sp) with the lever, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel_f import build_f_kernel, f_class_spec; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_f_kernel(f_class_spec(0, 1, 0, 1, ('fmixed',))); \
#   Path(f'test/golden/cuda/f_fmixed/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); Path(f'test/golden/hip/f_fmixed/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_CLS = (0, 1, 0, 1)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_fmixed_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_f_kernel(lever_spec(GOLDEN_CLS))
    where = GOLDEN / backend / "f_fmixed"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert "float C0_0_0__0_0_0 = 0.0f;" in text
    assert "fj_boys<2>(((double)T), boys);" in text  # fp64 Boys, #301
    assert "((float)boys[0])" in text  # each order narrowed at its read
    assert "const float*" not in text
