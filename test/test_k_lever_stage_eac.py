"""`E^ac` staged in shared memory, `k_levers/stage_eac.py` (#342), interpreted.

The lever moves the bra record's `E^ac` block from a per-lane local array to
one static `__shared__` copy per block (one warp per bra record, so the bra is
block-uniform). What is asserted:

1. THE DECLARATION: `stage-eac` is legal on `Target.K` only, refused on the
   other targets with a stated reason, keyed as any lever is, registered in
   `k_levers.REGISTRY`, moving `bind_e_ac` and `e_ac` only, and absent from
   every literal build (the literal K kernels' digest is test_k_seams.py's
   pin, unchanged).
2. THE FOOTPRINT, a declared function of the class: the built kernel's
   `EacS` is `staged_doubles(am)` doubles, `(n_ac, n_p)`, static and SHARED,
   declared once per bra record, with no local `Eac` left; the dynamic `Ksh`
   is the literal one, untouched; `footprint_bytes` is `kernel_k.shared_bytes`
   plus `staged_bytes` for every class through f; and `apply` refuses
   exactly the classes over the 48 KiB budget at 32 lanes.
3. THE BARRIER-FREE PRECONDITION: every `E^ac` fragment the lever stages
   writes each entry once and never reads its array, and a fragment that
   did would be refused.
4. EXACTNESS (the issue's "host-shim parity with kernel_k at L <= 2"): the
   arithmetic is the literal kernel's, so per bra record the lever's kernel
   is `ao.k.k_quartet` BIT FOR BIT and `==` the literal kernel's `K` --
   every class with L <= 2 (fast), a sample through L = 4 (`slow`, the cap);
   with several lanes (the redundant identical-value stores, in the
   interpreter's sequentially consistent model); and with a 3-batch split,
   onto a zeroed `K`, the literal batched kernels' `K` bit for bit.
5. COMPOSITION with #341's `split`: both levers pulled give the split
   element kernels with `E^ac` staged, the split block bit for bit.
6. THE GOLDENS: `(p s|s s)` with the lever, one unit per backend.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pytest
import test_kernel_k as tk

from felizjueves import e_fragment as ef
from felizjueves import k_levers, levers
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import (
    Assign,
    Decl,
    For,
    If,
    Index,
    Inline,
    Kernel,
    Lit,
    Note,
    Space,
    Var,
    walk,
)
from felizjueves.k_levers import split, stage_eac
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KSH, KHooks, ncart
from felizjueves.kernel_k import (
    KET,
    build_k_kernel,
    build_k_kernels,
    launch_shape,
    shared_bytes,
)
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]
NAME = stage_eac.NAME

FAST: list[Am] = tk.FAST  # every K class with L <= 2
SLOW_SAMPLE: list[Am] = [(1, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1),
                         (2, 0, 2, 0), (2, 2, 0, 0), (0, 0, 2, 2),
                         (2, 1, 1, 0), (0, 2, 1, 1)]
CLASSES = ([pytest.param(am, id=tk._id(am)) for am in FAST]
           + [pytest.param(am, id=tk._id(am), marks=pytest.mark.slow)
              for am in SLOW_SAMPLE])
# Every class with no shell above f, for the footprint arithmetic (no build).
THROUGH_F: list[Am] = list(itertools.product(range(4), repeat=4))


def lever_spec(am: Am, n_batches: int = 1, more: tuple[str, ...] = ()
               ) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, tuple(sorted(
        (NAME, *more))), n_batches)


def literal_spec(am: Am, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, (), n_batches)


def _n_p(lp: int) -> int:
    return (lp + 1) * (lp + 2) * (lp + 3) // 6


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_stage_eac_is_declared_legal_on_k_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[NAME]
    assert (lever.name, lever.issue, lever.legal) == (NAME, 342,
                                                      frozenset({"k"}))
    assert "E^ac" in lever.refusals["j"]
    assert "#340" in lever.refusals["f"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="no bra E\\^ac"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (NAME,))


def test_stage_eac_is_registered_keyed_and_moves_two_hooks() -> None:
    assert k_levers.REGISTRY[NAME] is stage_eac.apply
    spec = lever_spec((1, 0, 1, 0))
    assert spec.key() == "coulomb/d0/(1,0|1,0)/b1/k/stage-eac"
    assert spec.kernel_body_levers() == (NAME,)
    moved = {f.name for f in dataclasses.fields(KHooks)
             if not k_levers.hooks_for(spec).is_default(f.name)}
    assert moved == {"bind_e_ac", "e_ac"}
    assert k_levers.hooks_for(literal_spec((1, 0, 1, 0))) is DEFAULT_HOOKS
    # one kernel, the literal kernel's name
    assert [k.name for k in build_k_kernels(spec)] == ["k_10_10"]
    assert build_k_kernel(spec).name == "k_10_10"


def test_the_literal_kernel_is_untouched() -> None:
    for am in FAST:
        kern = build_k_kernel(am)
        assert not any(isinstance(s, Decl) and s.name == stage_eac.ARRAY
                       for s in walk(kern.body))
        assert stage_eac.PATTERN not in [s.prov for s in walk(kern.body)
                                         if isinstance(s, Note)]


# ---------------------------------------------------------------------------
# 2. The footprint
# ---------------------------------------------------------------------------


def _decl(kern: Kernel, name: str) -> Decl:
    (d,) = [s for s in walk(kern.body) if isinstance(s, Decl)
            and s.name == name]
    return d


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST])
def test_the_staged_block_is_the_declared_footprint(am: Am) -> None:
    la, lc, _, _ = am
    kern = build_k_kernel(lever_spec(am))
    lit = build_k_kernel(am)
    eacs = _decl(kern, stage_eac.ARRAY)
    assert eacs.space is Space.SHARED and eacs.type is _decl(lit, "Eac").type
    shape = tuple(d.value for d in eacs.shape)  # static: literal extents
    assert shape == (ncart(la) * ncart(lc), _n_p(la + lc))
    assert shape[0] * shape[1] == stage_eac.staged_doubles(am)
    assert stage_eac.staged_bytes(am) == 8 * stage_eac.staged_doubles(am)
    # no local E^ac left; the dynamic Ksh is the literal one
    assert not any(isinstance(s, Decl) and s.name == "Eac"
                   for s in walk(kern.body))
    assert _decl(kern, KSH) == _decl(lit, KSH)
    # declared once per bra record: inside the `bra < n_bra` guard, not in
    # the ket loop, and not at the top level (where the device harness
    # counts per-lane dynamic rows, measured.jsh_slots)
    assert not any(isinstance(s, Decl) and s.name == stage_eac.ARRAY
                   for s in kern.body)
    (guard,) = [s for s in kern.body if isinstance(s, If)]
    assert eacs in guard.then
    (loop,) = [s for s in walk(kern.body) if isinstance(s, For)
               and s.var == KET]
    assert eacs not in tuple(walk(loop.body))
    notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
    assert levers.provenance(NAME) in notes and stage_eac.PATTERN in notes


@pytest.mark.parametrize("lanes", [1, 32])
def test_the_footprint_is_ksh_plus_the_static_block(lanes: int) -> None:
    for am in THROUGH_F:
        la, lc, _, _ = am
        assert (stage_eac.staged_doubles(am)
                == ncart(la) * ncart(lc) * _n_p(la + lc))
        assert (stage_eac.footprint_bytes(am, lanes)
                == shared_bytes(am, lanes) + stage_eac.staged_bytes(am))
    # one copy per block: the staged part does not scale with lanes
    am = (2, 2, 2, 2)
    assert (stage_eac.footprint_bytes(am, 64) - stage_eac.footprint_bytes(am, 32)
            == shared_bytes(am, 64) - shared_bytes(am, 32))
    # (dd|dd), the predecessor's ledger cell: 10,080 B staged + 18,176 B Ksh
    assert stage_eac.staged_bytes(am) == 36 * 35 * 8 == 10_080
    assert stage_eac.footprint_bytes(am, 32) == 28_256


def test_apply_refuses_exactly_the_classes_over_the_budget() -> None:
    refused = []
    for am in THROUGH_F:
        spec = Spec(Operator.COULOMB, 0, am, Target.K)
        over = (stage_eac.footprint_bytes(am, stage_eac.BUDGET_LANES)
                > stage_eac.BUDGET_BYTES)
        if over:
            with pytest.raises(ValueError, match="over the 49152 B budget"):
                stage_eac.apply(DEFAULT_HOOKS, spec)
            refused.append(am)
        else:
            stage_eac.apply(DEFAULT_HOOKS, spec)
    assert (2, 2, 2, 2) not in refused  # (dd|dd) fits
    assert (3, 3, 3, 3) in refused  # (ff|ff): 67,200 B staged alone
    assert stage_eac.staged_bytes((3, 3, 0, 0)) == 67_200
    # nothing through L = 4 (milestone 13's cap) is refused
    assert not [am for am in refused if sum(am) <= 4]
    with pytest.raises(ValueError, match="stage-eac"):
        build_k_kernels(lever_spec((3, 3, 3, 3)))


# ---------------------------------------------------------------------------
# 3. The barrier-free precondition
# ---------------------------------------------------------------------------


def test_every_staged_fragment_is_write_once_and_never_read() -> None:
    for la, lc in itertools.product(range(4), repeat=2):
        stage_eac._write_once(ef.block_fragment(la, lc))


def test_a_fragment_that_reads_or_rewrites_its_array_is_refused() -> None:
    frag = ef.block_fragment(1, 0)
    (first, *_) = [s for s in frag.body if isinstance(s, Assign)
                   and isinstance(s.dst, Index) and s.dst.array == ef.E_ARRAY]
    reread = Assign(Index(ef.E_ARRAY, (Lit(0), Lit(0))), first.dst)
    with pytest.raises(ValueError, match="reads its own array"):
        stage_eac._write_once(dataclasses.replace(
            frag, body=(*frag.body, reread)))
    with pytest.raises(ValueError, match="twice"):
        stage_eac._write_once(dataclasses.replace(
            frag, body=(*frag.body, first)))


def test_the_staged_fragment_is_bound_to_the_shared_array() -> None:
    kern = build_k_kernel(lever_spec((2, 1, 1, 0)))
    (inl,) = [s for s in walk(kern.body) if isinstance(s, Inline)
              and s.fragment == ef.block_fragment(2, 1)]
    assert dict(inl.bindings)[ef.E_ARRAY] == Var(stage_eac.ARRAY)


# ---------------------------------------------------------------------------
# 4. Exactness: against ao.k.k_quartet per bra record, and the literal kernel
# ---------------------------------------------------------------------------


def run(kernels: tuple[Kernel, ...], am: Am, *, zero: bool, lanes: int = 1,
        contracted: bool = True, shift: float = 0.0) -> np.ndarray:
    """The kernels in order onto one `K`, test_kernel_k's shells (NaN-filled
    unless `zero`)."""
    bra, ket = tk._pairs(am, contracted=contracted)
    args = tk.k_args(am, bra, [(ket, 0)], [tk._dcd(am[1], am[3], shift)])
    if zero:
        args["K"] = np.zeros_like(args["K"])
    grid, block = launch_shape(args["n_bra"], lanes)
    for kern in kernels:
        args["K"] = Interpreter(kern, launch=Launch(grid, block, lanes),
                                boys=boys).run(**args)["K"]
    return args["K"]


@pytest.mark.parametrize("am", CLASSES)
def test_each_bra_record_is_the_host_staged_quartet_bitwise(am: Am) -> None:
    """The issue's acceptance: host-shim parity with `kernel_k`, per bra
    record against `ao.k.k_quartet`, bit for bit."""
    bra, ket = tk._pairs(am)
    dcd = tk._dcd(am[1], am[3])
    K = run(build_k_kernels(lever_spec(am)), am, zero=False)
    assert not np.isnan(K).any()
    for i in range(len(bra.recs)):
        assert K[i].tolist() == tk._host(tk._one(bra, i), ket, dcd, am), i
    assert np.any(K != 0.0)
    assert K.tolist() == run((build_k_kernel(am),), am, zero=False).tolist()


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST])
def test_an_uncontracted_bra_is_the_whole_quartet_bitwise(am: Am) -> None:
    bra, ket = tk._pairs(am, contracted=False)
    dcd = tk._dcd(am[1], am[3], shift=-0.2)
    K = run(build_k_kernels(lever_spec(am)), am, zero=False,
            contracted=False, shift=-0.2)
    assert K[0].tolist() == tk._host(bra, ket, dcd, am)


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST
                                if sum(a) > 0])
def test_several_lanes_store_the_same_block_and_agree_with_the_literal(
        am: Am) -> None:
    """Every lane stores the whole block (the redundant, identical-value
    staging); with 4 lanes the lever's `K` is the literal kernel's at 4 lanes,
    bit for bit (both reassociate the same way, by the same butterfly)."""
    lever = run(build_k_kernels(lever_spec(am)), am, zero=False, lanes=4)
    literal = run((build_k_kernel(am),), am, zero=False, lanes=4)
    assert not np.isnan(lever).any()
    assert lever.tolist() == literal.tolist()


SPLIT3 = ([pytest.param(a, id=tk._id(a)) for a in FAST if sum(a) > 0]
          + [pytest.param(a, id=tk._id(a), marks=pytest.mark.slow)
             for a in SLOW_SAMPLE])


@pytest.mark.parametrize("am", SPLIT3)
def test_with_a_batch_split_it_is_the_literal_batched_kernels_bitwise(
        am: Am) -> None:
    kernels = build_k_kernels(lever_spec(am, 3))
    literal = build_k_kernels(literal_spec(am, 3))
    assert [k.name for k in kernels] == [k.name for k in literal]
    for kern in kernels:  # each batch stages the whole block, as it forms it
        la, lc, _, _ = am
        assert (tuple(d.value for d in _decl(kern, stage_eac.ARRAY).shape)
                == (ncart(la) * ncart(lc), _n_p(la + lc)))
    assert (run(kernels, am, zero=True).tolist()
            == run(literal, am, zero=True).tolist())


# ---------------------------------------------------------------------------
# 5. Composition with #341's split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("am", [(1, 0, 1, 0), (1, 1, 0, 0), (0, 1, 1, 0)])
def test_it_composes_with_the_per_element_split(am: Am) -> None:
    both = build_k_kernels(lever_spec(am, more=(split.NAME,)))
    alone = build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K,
                                 (split.NAME,)))
    assert [k.name for k in both] == [k.name for k in alone]
    for kern in both:
        _decl(kern, stage_eac.ARRAY)  # staged in every element kernel
    assert (run(both, am, zero=False).tolist()
            == run(alone, am, zero=False).tolist())


# ---------------------------------------------------------------------------
# 6. The goldens: (p s|s s) with the lever, one unit per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.spec import Spec, Operator, Target; \
#   from felizjueves.kernel_k import build_k_kernels; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   [(Path(f'test/golden/cuda/k_stage_eac/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)), Path(f'test/golden/hip/k_stage_eac/{k.name}.hip' \
#   ).write_text(hip.emit(k, GFX1200))) for k in build_k_kernels(Spec( \
#   Operator.COULOMB, 0, (1, 0, 0, 0), Target.K, ('stage-eac',)))]"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_AM: Am = (1, 0, 0, 0)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_stage_eac_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kernels = build_k_kernels(lever_spec(GOLDEN_AM))
    where = GOLDEN / backend / "k_stage_eac"
    assert sorted(p.name for p in where.iterdir()) == sorted(
        f"{k.name}.{ext}" for k in kernels)
    for kern in kernels:
        text = emit(kern, arch)
        assert text == (where / f"{kern.name}.{ext}").read_text()
        assert f"__shared__ double {stage_eac.ARRAY}[3][4];" in text
