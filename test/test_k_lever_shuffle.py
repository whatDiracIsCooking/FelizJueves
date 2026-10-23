"""The register K write-back lever, `k_levers/shuffle.py` (#349), interpreted.

The lever keeps each reduced `K[a, b]` in a register and has the lane that
owns element `o` select `Kab[o]` and store it, one lane-parallel store per
round, instead of #345's in-place `Ksh` stores; a block wider than the warp
falls back to #345's write-back statement for statement. What is asserted:

1. THE DECLARATION: `shuffle` is legal on `Target.K` only, refused on the
   other targets with a stated reason, keyed as any lever is, registered,
   moving `write_back` and `atomize` only.
2. ELIGIBILITY IS THE WARP'S: `MIN_WAVEFRONT` is the narrowest wavefront
   of `device.ARCHS`; a 36-element block, `(d s|d s)`, falls back at 32 and
   takes the register form at 64; the fallback IS `coalesced.write_back`.
3. THE SHAPE: no lane-0 guard, no store into `Ksh` after the contraction
   (the in-place reductions are gone), one round loop, one `K` write under
   the owner guard, the literal `Ksh` (zero extra bytes), no `Raw`, and no
   `LaneBroadcast` (the butterfly already leaves the sum on every lane; the
   module docstring says why).
4. PARITY: per bra record against `ao.k.k_quartet` bit for bit at one lane;
   against the literal kernel and against `coalesced` bit for bit at 1, 4
   and 32 lanes (the same butterfly on the same values); the 3-batch split
   as lane-parallel atomics, bit for bit the literal batched kernels; the
   composition with #341's split. L <= 2 fast, a sample to L = 4 `slow`.
5. THE GOLDENS: `(p s|p s)` with the lever, one kernel per backend, and the
   `LaneBroadcast` node's spelling on each backend.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_kernel_k as tk

from felizjueves import device, k_levers, levers
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Fn,
    For,
    If,
    Index,
    Intent,
    Kernel,
    LaneBroadcast,
    Lit,
    Note,
    Param,
    Raw,
    Scalar,
    Var,
    WarpReduce,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.ir.passes import LANE0
from felizjueves.k_levers import coalesced, shuffle, split
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KSH, KContext, KHooks, ncart
from felizjueves.kernel_k import build_k_kernel, build_k_kernels, launch_shape
from felizjueves.mmd.kernel import builder_plan
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

FAST: list[Am] = tk.FAST  # every K class with L <= 2
SLOW_SAMPLE: list[Am] = [(1, 1, 1, 0), (1, 1, 1, 1), (2, 0, 2, 0),
                         (2, 1, 1, 0), (0, 2, 1, 1)]
CLASSES = ([pytest.param(am, id=tk._id(am)) for am in FAST]
           + [pytest.param(am, id=tk._id(am), marks=pytest.mark.slow)
              for am in SLOW_SAMPLE])
FAST_PARAMS = [pytest.param(am, id=tk._id(am)) for am in FAST]
WIDE: Am = (2, 0, 2, 0)  # 36 elements: wider than a wave32 warp


def lever_spec(am: Am, n_batches: int = 1, *more: str) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K,
                tuple(sorted((shuffle.NAME, *more))), n_batches)


def _n(am: Am) -> int:
    return ncart(am[0]) * ncart(am[2])


def _ctx(spec: Spec) -> KContext:
    (dag,) = builder_plan(spec).dags
    return KContext.of(spec, dag, "x", None, k_levers.hooks_for(spec))


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_shuffle_is_declared_legal_on_k_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[shuffle.NAME]
    assert (lever.name, lever.issue, lever.legal) == ("shuffle", 349,
                                                      frozenset({"k"}))
    assert "J_p" in lever.refusals["j"]
    assert "#340" in lever.refusals["f"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="lane-0 J_p write"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (shuffle.NAME,))


def test_shuffle_is_registered_keyed_and_moves_its_two_hooks() -> None:
    assert k_levers.REGISTRY[shuffle.NAME] is shuffle.apply
    spec = lever_spec((1, 0, 1, 0))
    assert spec.key() == "coulomb/d0/(1,0|1,0)/b1/k/shuffle"
    assert spec.kernel_body_levers() == (shuffle.NAME,)
    hooks = k_levers.hooks_for(spec)
    moved = {f.name for f in dataclasses.fields(KHooks)
             if not hooks.is_default(f.name)}
    assert moved == {"write_back", "atomize"}
    assert hooks.write_back is shuffle.write_back
    assert build_k_kernel(spec).name == build_k_kernel((1, 0, 1, 0)).name


# ---------------------------------------------------------------------------
# 2. Eligibility, derived from the warp
# ---------------------------------------------------------------------------


def test_the_build_width_is_the_narrowest_wavefront_of_the_cards() -> None:
    assert shuffle.MIN_WAVEFRONT == min(a.wavefront for a in device.ARCHS)
    assert shuffle.EXTRA_SHARED_BYTES == 0


def test_eligibility_follows_the_wavefront() -> None:
    # the issue's example: 36 accumulators fall back at wave 32, fit at 64
    assert not shuffle.eligible(_n(WIDE), 32)
    assert shuffle.eligible(_n(WIDE), 64)
    assert shuffle.eligible(32, 32) and not shuffle.eligible(33, 32)
    for arch in device.ARCHS:
        assert shuffle.eligible(9, arch.wavefront)
    with pytest.raises(ValueError):
        shuffle.eligible(0, 32)


def test_a_block_wider_than_the_warp_falls_back_to_coalesced() -> None:
    ctx = _ctx(lever_spec(WIDE))
    assert shuffle.write_back(ctx) == coalesced.write_back(ctx)
    assert shuffle.write_back_at(32)(ctx) == coalesced.write_back(ctx)
    wave64 = shuffle.write_back_at(64)(ctx)
    assert wave64 != coalesced.write_back(ctx)
    assert any(isinstance(s, For) and s.var == shuffle.ROUND_VAR
               for s in wave64)
    # and the whole kernel is the coalesced lever's but for the provenance
    def strip(k: Kernel) -> Kernel:
        def body(stmts):
            out = []
            for s in stmts:
                if isinstance(s, Note) and s.prov in (
                        levers.provenance(shuffle.NAME),
                        levers.provenance(coalesced.NAME)):
                    continue
                if isinstance(s, If):
                    s = If(s.cond, body(s.then), body(s.orelse))
                out.append(s)
            return tuple(out)
        return dataclasses.replace(k, body=body(k.body))
    assert strip(build_k_kernel(lever_spec(WIDE))) == strip(build_k_kernel(
        Spec(Operator.COULOMB, 0, WIDE, Target.K, (coalesced.NAME,))))


# ---------------------------------------------------------------------------
# 3. The shape
# ---------------------------------------------------------------------------


def _k_writes(kern: Kernel) -> list[Assign | AtomicAdd]:
    return [s for s in walk(kern.body) if isinstance(s, Assign | AtomicAdd)
            and isinstance(s.dst, Index) and s.dst.array == "K"]


def _ksh_stores_after_contraction(kern: Kernel) -> list:
    """Stores into `Ksh` from the first `WarpReduce` on: the write-back's."""
    flat = list(walk(kern.body))
    first = next(i for i, s in enumerate(flat) if isinstance(s, WarpReduce))
    return [s for s in flat[first:] if isinstance(s, Assign | WarpReduce)
            and isinstance(s.dst, Index) and s.dst.array == KSH]


@pytest.mark.parametrize("am", FAST_PARAMS)
def test_the_write_is_one_owner_store_in_one_round_loop(am: Am) -> None:
    kern, literal = build_k_kernel(lever_spec(am)), build_k_kernel(am)
    assert not any(isinstance(s, If) and s.cond == LANE0
                   for s in walk(kern.body))
    (loop,) = [s for s in walk(kern.body) if isinstance(s, For)
               and s.var == shuffle.ROUND_VAR]
    assert (loop.start, loop.stop, loop.step) == (
        Lit(0), Lit(_n(am)), Call(Fn.BLOCK_DIM_Y))
    (write,) = _k_writes(kern)
    assert isinstance(write, Assign) and write.value == Var(shuffle.MINE_VAR)
    guard = next(s for s in loop.body if isinstance(s, If) and write in s.then)
    assert guard.cond == Call(Fn.LT, (Var(shuffle.OWN_VAR), Lit(_n(am))))
    # one reduction per element, into a register, and no Ksh store after
    reds = [s for s in walk(kern.body) if isinstance(s, WarpReduce)]
    assert len(reds) == _n(am)
    assert all(isinstance(r.dst, Var) for r in reds)
    assert not _ksh_stores_after_contraction(kern)
    assert len(_ksh_stores_after_contraction(build_k_kernel(Spec(
        Operator.COULOMB, 0, am, Target.K, (coalesced.NAME,))))) == _n(am)
    # the footprint: the literal Ksh; nothing vendor-specific, no shuffle node
    decl = [s for s in walk(kern.body) if isinstance(s, Decl) and s.name == KSH]
    assert decl == [s for s in walk(literal.body)
                    if isinstance(s, Decl) and s.name == KSH]
    assert not any(isinstance(s, Raw | LaneBroadcast) for s in walk(kern.body))
    notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
    assert levers.provenance(shuffle.NAME) in notes
    assert shuffle.SELECT_NOTE in notes and shuffle.PLAIN_WRITE in notes


# ---------------------------------------------------------------------------
# 4. Parity
# ---------------------------------------------------------------------------


def run(kern: Kernel, am: Am, lanes: int) -> np.ndarray:
    bra, ket = tk._pairs(am)
    args = tk.k_args(am, bra, [(ket, 0)], [tk._dcd(am[1], am[3])])
    grid, block = launch_shape(args["n_bra"], lanes)
    return Interpreter(kern, launch=Launch(grid, block, lanes),
                       boys=boys).run(**args)["K"]


@pytest.mark.parametrize("am", CLASSES)
def test_each_bra_record_is_the_host_staged_quartet_bitwise(am: Am) -> None:
    bra, ket = tk._pairs(am)
    dcd = tk._dcd(am[1], am[3])
    K = run(build_k_kernel(lever_spec(am)), am, 1)
    assert not np.isnan(K).any()
    for i in range(len(bra.recs)):
        assert K[i].tolist() == tk._host(tk._one(bra, i), ket, dcd, am), i
    assert np.any(K != 0.0)


@pytest.mark.parametrize("lanes", [1, 4, 32])
@pytest.mark.parametrize("am", CLASSES)
def test_the_literal_and_coalesced_paths_bitwise_at_every_lane_count(
        am: Am, lanes: int) -> None:
    """4 lanes under 9 elements takes three rounds; 32 lanes over fewer
    elements leaves lanes owning nothing; (d s|d s) is the fallback."""
    got = run(build_k_kernel(lever_spec(am)), am, lanes)
    assert not np.isnan(got).any()
    assert got.tolist() == run(build_k_kernel(am), am, lanes).tolist()
    assert got.tolist() == run(build_k_kernel(Spec(
        Operator.COULOMB, 0, am, Target.K, (coalesced.NAME,))), am,
        lanes).tolist()


@pytest.mark.slow
@pytest.mark.parametrize("lanes", [4, 32])
def test_the_wave64_form_of_a_wide_block_is_bitwise_too(lanes: int) -> None:
    """(d s|d s) built for wave64 (36 <= 64), run on narrower warps: the
    round loop covers it, bit for bit the literal kernel."""
    hooks = dataclasses.replace(DEFAULT_HOOKS,
                                write_back=shuffle.write_back_at(64))
    kern = build_k_kernel(WIDE, hooks=hooks)
    assert any(isinstance(s, For) and s.var == shuffle.ROUND_VAR
               for s in walk(kern.body))
    assert run(kern, WIDE, lanes).tolist() == run(
        build_k_kernel(WIDE), WIDE, lanes).tolist()


def run_all(kernels, am: Am, lanes: int) -> np.ndarray:
    bra, ket = tk._pairs(am)
    args = tk.k_args(am, bra, [(ket, 0)], [tk._dcd(am[1], am[3])])
    args["K"] = np.zeros_like(args["K"])
    grid, block = launch_shape(args["n_bra"], lanes)
    for kern in kernels:
        args["K"] = Interpreter(kern, launch=Launch(grid, block, lanes),
                                boys=boys).run(**args)["K"]
    return args["K"]


SPLIT3 = ([pytest.param(a, id=tk._id(a)) for a in FAST if sum(a) > 0]
          + [pytest.param(a, id=tk._id(a), marks=pytest.mark.slow)
             for a in SLOW_SAMPLE])


@pytest.mark.parametrize("am", SPLIT3)
def test_a_split_class_writes_by_owner_lane_atomics(am: Am) -> None:
    kernels = build_k_kernels(lever_spec(am, 3))
    literal = build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K, (), 3))
    assert [k.name for k in kernels] == [k.name for k in literal]
    for kern, lit in zip(kernels, literal, strict=True):
        assert not any(isinstance(s, If) and s.cond == LANE0
                       for s in walk(kern.body))
        writes = _k_writes(kern)
        assert writes and all(isinstance(s, AtomicAdd) for s in writes)
        n = len(_k_writes(lit))
        if shuffle.eligible(n, shuffle.MIN_WAVEFRONT):
            assert len(writes) == 1
            notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
            assert shuffle.ATOMIC_WRITE in notes
            assert not any("NOT atomic" in x for x in notes)


@pytest.mark.parametrize("lanes", [1, 4, 32])
@pytest.mark.parametrize("am", SPLIT3)
def test_a_split_class_is_the_literal_batched_kernels_bitwise(
        am: Am, lanes: int) -> None:
    got = run_all(build_k_kernels(lever_spec(am, 3)), am, lanes)
    want = run_all(build_k_kernels(
        Spec(Operator.COULOMB, 0, am, Target.K, (), 3)), am, lanes)
    assert got.tolist() == want.tolist()
    assert np.any(got != 0.0)


def test_a_non_contiguous_block_selects_its_column() -> None:
    """A batch that writes a gapped set of `K[a, b]` (e.g. `(p s|p s)`'s
    y and z batches) takes the `kcol` form; the bitwise test above runs
    those kernels."""
    gapped = [kern.name for kern in build_k_kernels(lever_spec((1, 0, 1, 0), 3))
              if any(isinstance(s, Decl) and s.name == shuffle.COL_VAR
                     for s in walk(kern.body))]
    assert gapped == ["k_10_10_y0_2", "k_10_10_z0_2"]
    for kern in build_k_kernels(lever_spec((1, 0, 1, 0), 3)):
        (w,) = _k_writes(kern)
        if kern.name in gapped:
            assert w.dst.at[1] == Var(shuffle.COL_VAR)
        else:
            assert w.dst.at[1] == Var(shuffle.OWN_VAR)


def test_composed_with_split_each_element_is_one_owner_store() -> None:
    am = (1, 0, 1, 0)
    for n_batches in (1, 3):
        kernels = build_k_kernels(lever_spec(am, n_batches, split.NAME))
        plain = build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K,
                                     (split.NAME,), n_batches))
        assert [k.name for k in kernels] == [k.name for k in plain]
        for kern in kernels:
            assert len(_k_writes(kern)) == 1
        want = run_all(build_k_kernels(Spec(Operator.COULOMB, 0, am,
                                            Target.K, (), n_batches)), am, 4)
        assert run_all(kernels, am, 4).tolist() == want.tolist()


# ---------------------------------------------------------------------------
# 5. The goldens, and the node's two spellings
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.spec import Spec, Operator, Target; \
#   from felizjueves.kernel_k import build_k_kernel; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_k_kernel(Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, \
#   ('shuffle',))); \
#   Path(f'test/golden/cuda/k_shuffle/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); \
#   Path(f'test/golden/hip/k_shuffle/{k.name}.hip').write_text( \
#   hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_shuffle_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_k_kernel(lever_spec((1, 0, 1, 0)))
    where = GOLDEN / backend / "k_shuffle"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert "syncwarp" not in text and "__syncthreads" not in text
    assert "__shfl_sync(" not in text  # the butterfly only: no broadcast


def _bcast() -> Kernel:
    ty = Call(Fn.THREAD_Y)
    return Kernel("bcast", (Param("v", Scalar.F64, array=True),
                            Param("out", Scalar.F64, array=True,
                                  intent=Intent.OUT)), (
        Decl("got", Scalar.F64),
        LaneBroadcast(Var("got"), Index("v", (ty,)), Lit(0)),
        Assign(Index("out", (ty,)), Var("got"))))


def test_the_lane_broadcast_is_spelled_on_each_backend() -> None:
    assert ("    got = __shfl_sync(0xffffffffu, v[(int)threadIdx.y], 0, 32);"
            in cuda.emit(_bcast(), SM_86).splitlines())
    assert ("    got = __shfl_sync(0xffffffffull, v[(int)threadIdx.y], 0, 32);"
            in hip.emit(_bcast(), GFX1200).splitlines())
