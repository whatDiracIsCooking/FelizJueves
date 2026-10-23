"""The coalesced K write-back lever, `k_levers/coalesced.py` (#345), interpreted.

The lever replaces lane 0's serial `K[a, b]` stores with one lane-strided
store loop, the reduced block broadcast through each lane's own `Ksh`
accumulator column. What is asserted:

1. THE DECLARATION: `coalesced` is legal on `Target.K` only, refused on the
   other targets with a stated reason, keyed as any lever is, registered in
   `k_levers.REGISTRY`, moving `write_back` only, and absent from every
   literal build (the literal K kernels' digest is test_k_seams.py's pin).
2. THE SHAPE: no write into `K` under the lane-0 guard; one `WarpReduce`
   per `K[a, b]`, in place in its `Ksh` slot; one strided loop
   (`THREAD_Y`, stride `BLOCK_DIM_Y`) storing every element once; the `Ksh`
   declaration is the literal kernel's (the footprint declared: zero extra
   bytes); nothing vendor-specific (no `Raw`), so no warp sync to spell.
3. PARITY WITH THE LANE-0 PATH (the issue's acceptance, L <= 2): per bra
   record against `ao.k.k_quartet` bit for bit at one lane, and against the
   literal kernel bit for bit at 1, 4 and 32 lanes -- the same butterfly on
   the same values, so the lever moves no bit. Above L = 2 within the cap of
   L = 4, a sample, `slow`.
4. THE SPLIT CLASS (3 batches, the atomic write-back mode): every batch
   kernel stores through lane-parallel `AtomicAdd`s with no lane-0 guard,
   through the `atomize` seam (#345); onto a zeroed `K` the batches in order
   are the literal batched kernels' `K` bit for bit at 1, 4 and 32 lanes
   (L <= 2 fast, a sample to L = 4 `slow`). Composed with #341's split (one
   element per kernel) the loop is a run of one and the union is still the
   literal block, at one batch and at three.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import test_kernel_k as tk

from felizjueves import k_levers, levers
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
    Kernel,
    Note,
    Raw,
    WarpReduce,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.ir.passes import LANE0
from felizjueves.k_levers import coalesced, split
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KSH, KHooks, ncart
from felizjueves.kernel_k import (
    build_k_kernel,
    build_k_kernels,
    launch_shape,
    shared_bytes,
)
from felizjueves.mmd.kernel import k_spec
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

FAST: list[Am] = tk.FAST  # every K class with L <= 2
SLOW_SAMPLE: list[Am] = [(1, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1),
                         (2, 0, 2, 0), (2, 1, 1, 0), (0, 2, 1, 1),
                         (2, 1, 0, 1)]
CLASSES = ([pytest.param(am, id=tk._id(am)) for am in FAST]
           + [pytest.param(am, id=tk._id(am), marks=pytest.mark.slow)
              for am in SLOW_SAMPLE])
FAST_PARAMS = [pytest.param(am, id=tk._id(am)) for am in FAST]


def lever_spec(am: Am, n_batches: int = 1, *more: str) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K,
                tuple(sorted((coalesced.NAME, *more))), n_batches)


def _n(am: Am) -> int:
    return ncart(am[0]) * ncart(am[2])


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_coalesced_is_declared_legal_on_k_only_with_its_own_reasons() -> None:
    lever = levers.VOCABULARY[coalesced.NAME]
    assert (lever.name, lever.issue, lever.legal) == ("coalesced", 345,
                                                      frozenset({"k"}))
    assert "J_p" in lever.refusals["j"]
    assert "#340" in lever.refusals["f"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="lane-0 J_p write"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (coalesced.NAME,))


def test_coalesced_is_registered_keyed_and_moves_write_back_only() -> None:
    assert k_levers.REGISTRY[coalesced.NAME] is coalesced.apply
    spec = lever_spec((1, 0, 1, 0))
    assert spec.key() == "coulomb/d0/(1,0|1,0)/b1/k/coalesced"
    assert spec.kernel_body_levers() == (coalesced.NAME,)
    hooks = k_levers.hooks_for(spec)
    moved = {f.name for f in dataclasses.fields(KHooks)
             if not hooks.is_default(f.name)}
    assert moved == {"write_back", "atomize"}
    assert hooks.write_back is coalesced.write_back
    assert k_levers.hooks_for(dataclasses.replace(spec, levers=())) is (
        DEFAULT_HOOKS)
    assert build_k_kernel(spec).name == build_k_kernel((1, 0, 1, 0)).name


# ---------------------------------------------------------------------------
# 2. The shape: the write, the reductions, the footprint
# ---------------------------------------------------------------------------


def _decl(kern: Kernel, name: str) -> Decl:
    (d,) = [s for s in walk(kern.body) if isinstance(s, Decl) and s.name == name]
    return d


def _k_writes(kern: Kernel) -> list[Assign | AtomicAdd]:
    return [s for s in walk(kern.body) if isinstance(s, Assign | AtomicAdd)
            and isinstance(s.dst, Index) and s.dst.array == "K"]


@pytest.mark.parametrize("am", FAST_PARAMS)
def test_the_write_is_one_lane_strided_loop_over_the_block(am: Am) -> None:
    kern = build_k_kernel(lever_spec(am))
    assert not any(isinstance(s, If) and s.cond == LANE0
                   for s in walk(kern.body))
    loops = [s for s in walk(kern.body) if isinstance(s, For)
             and s.var == coalesced.LANE_VAR]
    assert len(loops) == 1  # every element kept: one run
    (loop,) = loops
    assert loop.start == Call(Fn.THREAD_Y)
    assert loop.step == Call(Fn.BLOCK_DIM_Y)
    assert loop.stop.value == _n(am)
    (write,) = _k_writes(kern)
    assert isinstance(write, Assign) and loop.body == (write,)
    assert write.value.array == KSH  # read back from the lane's own column
    assert write.value.at[1] == Call(Fn.THREAD_Y)


@pytest.mark.parametrize("am", FAST_PARAMS)
def test_one_in_place_reduction_per_element_and_no_extra_shared(am: Am
                                                                ) -> None:
    kern, literal = build_k_kernel(lever_spec(am)), build_k_kernel(am)
    reds = [s for s in walk(kern.body) if isinstance(s, WarpReduce)]
    assert len(reds) == _n(am)
    assert all(r.dst == r.value and r.dst.array == KSH for r in reds)
    assert len({r.dst for r in reds}) == _n(am)
    assert len([s for s in walk(literal.body)
                if isinstance(s, WarpReduce)]) == _n(am)
    # the footprint declared: the literal Ksh, nothing added
    assert coalesced.EXTRA_SHARED_BYTES == 0
    assert _decl(kern, KSH) == _decl(literal, KSH)
    assert shared_bytes(am, 32) == (_decl(kern, KSH).shape[0].value * 32 * 8)
    # nothing vendor-specific: no Raw, so no warp sync for a backend to spell
    assert not any(isinstance(s, Raw) for s in walk(kern.body))
    notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
    assert levers.provenance(coalesced.NAME) in notes
    assert coalesced.REDUCE_NOTE in notes and coalesced.PLAIN_WRITE in notes


@pytest.mark.parametrize("am", FAST_PARAMS)
def test_everything_before_the_write_back_is_the_literal_kernels(am: Am
                                                                 ) -> None:
    """Only the write-back differs: the literal kernel's per-bra body up to
    its first `WarpReduce` is the lever's, statement for statement."""
    def per_bra(kern: Kernel) -> tuple:
        (guard,) = [s for s in kern.body if isinstance(s, If)]
        body = guard.then
        cut = next(i for i, s in enumerate(body) if isinstance(s, Note)
                   and ("reduce K[a,b]" in s.prov or "coalesced" in s.prov))
        return body[:cut]
    assert per_bra(build_k_kernel(lever_spec(am))) == per_bra(
        build_k_kernel(am))


# ---------------------------------------------------------------------------
# 3. Parity with the lane-0 path
# ---------------------------------------------------------------------------


def run(kern: Kernel, am: Am, lanes: int, *, contracted: bool = True,
        shift: float = 0.0) -> np.ndarray:
    bra, ket = tk._pairs(am, contracted=contracted)
    args = tk.k_args(am, bra, [(ket, 0)], [tk._dcd(am[1], am[3], shift)])
    grid, block = launch_shape(args["n_bra"], lanes)
    return Interpreter(kern, launch=Launch(grid, block, lanes),
                       boys=boys).run(**args)["K"]


@pytest.mark.parametrize("am", CLASSES)
def test_each_bra_record_is_the_host_staged_quartet_bitwise(am: Am) -> None:
    bra, ket = tk._pairs(am)
    dcd = tk._dcd(am[1], am[3])
    K = run(build_k_kernel(lever_spec(am)), am, 1)
    assert not np.isnan(K).any()  # every element written
    for i in range(len(bra.recs)):
        assert K[i].tolist() == tk._host(tk._one(bra, i), ket, dcd, am), i
    assert np.any(K != 0.0)


@pytest.mark.parametrize("lanes", [1, 4, 32])
@pytest.mark.parametrize("am", CLASSES)
def test_the_lane_0_path_bitwise_at_every_lane_count(am: Am, lanes: int
                                                     ) -> None:
    """The literal kernel and the lever at the same lane count: the same
    butterfly on the same per-lane partials, so the same bits. 4 lanes with
    up to 36 elements exercises the strided loop's later rounds; 32 lanes
    with fewer elements than lanes leaves some lanes storing nothing."""
    got = run(build_k_kernel(lever_spec(am)), am, lanes)
    want = run(build_k_kernel(am), am, lanes)
    assert not np.isnan(got).any()
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("am", FAST_PARAMS)
def test_an_uncontracted_bra_is_the_whole_quartet_bitwise(am: Am) -> None:
    bra, ket = tk._pairs(am, contracted=False)
    dcd = tk._dcd(am[1], am[3], shift=-0.2)
    K = run(build_k_kernel(lever_spec(am)), am, 1, contracted=False,
            shift=-0.2)
    assert K[0].tolist() == tk._host(bra, ket, dcd, am)


# ---------------------------------------------------------------------------
# 4. The split class refused; the composition with #341
# ---------------------------------------------------------------------------


def test_the_one_batch_class_through_either_builder_is_one_kernel() -> None:
    spec = lever_spec((1, 0, 1, 0))
    assert build_k_kernels(spec) == (build_k_kernel(spec),)


def run_all(kernels, am: Am, lanes: int) -> np.ndarray:
    """The kernels in order onto one ZEROED `K` (the batch contract)."""
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
def test_a_split_class_writes_by_lane_parallel_atomics(am: Am) -> None:
    kernels = build_k_kernels(lever_spec(am, 3))
    literal = build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K, (), 3))
    assert [k.name for k in kernels] == [k.name for k in literal]
    for kern in kernels:
        assert not any(isinstance(s, If) and s.cond == LANE0
                       for s in walk(kern.body))
        writes = _k_writes(kern)
        assert writes and all(isinstance(s, AtomicAdd) for s in writes)
        loops = [s for s in walk(kern.body) if isinstance(s, For)
                 and s.var == coalesced.LANE_VAR]
        assert [s for lp in loops for s in lp.body] == writes
        notes = [s.prov for s in walk(kern.body) if isinstance(s, Note)]
        assert coalesced.ATOMIC_WRITE in notes
        assert coalesced.PLAIN_WRITE not in notes
        assert not any("NOT atomic" in n for n in notes)
        # the batch writes what the literal batch kernel writes, each once
        lit = next(k for k in literal if k.name == kern.name)
        lit_slots = sorted(s.dst.at[1].value for s in _k_writes(lit))
        mine = sum(lp.stop.value for lp in loops)
        assert mine == len(lit_slots)


@pytest.mark.parametrize("lanes", [1, 4, 32])
@pytest.mark.parametrize("am", SPLIT3)
def test_a_split_class_is_the_literal_batched_kernels_bitwise(
        am: Am, lanes: int) -> None:
    got = run_all(build_k_kernels(lever_spec(am, 3)), am, lanes)
    want = run_all(build_k_kernels(
        Spec(Operator.COULOMB, 0, am, Target.K, (), 3)), am, lanes)
    assert got.tolist() == want.tolist()
    assert np.any(got != 0.0)


def test_the_atomize_seam_hands_a_lane_0_kernel_to_the_prior() -> None:
    """Composition: a kernel whose write is still the lane-0 form (another
    lever's write-back) goes to the `atomize` the lever was handed."""
    spec = lever_spec((1, 0, 1, 0), 3)
    hooks = k_levers.hooks_for(spec)
    plain = build_k_kernels(Spec(Operator.COULOMB, 0, (1, 0, 1, 0),
                                 Target.K, (), 3))
    # a kernel whose K writes are all atomic already is accepted as it is
    assert hooks.atomize(None, plain[0]) is plain[0]
    lane0 = build_k_kernels(dataclasses.replace(spec, levers=()),
                            hooks=dataclasses.replace(
                                DEFAULT_HOOKS, atomize=lambda c, k: k))[0]
    assert hooks.atomize(None, lane0) == plain[0]


def test_composed_with_split_each_element_is_a_run_of_one() -> None:
    am = (1, 0, 1, 0)
    kernels = build_k_kernels(lever_spec(am, 1, split.NAME))
    plain = build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K,
                                 (split.NAME,)))
    assert [k.name for k in kernels] == [k.name for k in plain]
    for kern in kernels:
        (loop,) = [s for s in walk(kern.body) if isinstance(s, For)
                   and s.var == coalesced.LANE_VAR]
        assert loop.stop.value == 1
        assert len(_k_writes(kern)) == 1
    bra, ket = tk._pairs(am)
    args = tk.k_args(am, bra, [(ket, 0)], [tk._dcd(am[1], am[3])])
    grid, block = launch_shape(args["n_bra"], 4)
    for kern in kernels:
        args["K"] = Interpreter(kern, launch=Launch(grid, block, 4),
                                boys=boys).run(**args)["K"]
    assert args["K"].tolist() == run(build_k_kernel(am), am, 4).tolist()


def test_composed_with_split_at_three_batches() -> None:
    am = (1, 0, 1, 0)
    kernels = build_k_kernels(lever_spec(am, 3, split.NAME))
    plain = build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K,
                                 (split.NAME,), 3))
    assert [k.name for k in kernels] == [k.name for k in plain]
    for kern in kernels:
        (write,) = _k_writes(kern)
        assert isinstance(write, AtomicAdd)
    assert (run_all(kernels, am, 4).tolist()
            == run_all(build_k_kernels(Spec(Operator.COULOMB, 0, am, Target.K,
                                            (), 3)), am, 4).tolist())


def test_runs_are_one_run_of_the_whole_block_at_every_ladder_l() -> None:
    from felizjueves.k_levers.seams import KContext
    from felizjueves.mmd.kernel import builder_plan
    for L in range(5):
        spec = dataclasses.replace(k_spec(L), levers=(coalesced.NAME,))
        (dag,) = builder_plan(spec).dags
        ctx = KContext.of(spec, dag, "x", None, k_levers.hooks_for(spec))
        assert coalesced.runs(ctx) == ((0, len(ctx.i_rows), _n(spec.am)),)


# ---------------------------------------------------------------------------
# 5. The goldens: (p s|p s) coalesced, one kernel per backend
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.spec import Spec, Operator, Target; \
#   from felizjueves.kernel_k import build_k_kernel; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   k = build_k_kernel(Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, \
#   ('coalesced',))); \
#   Path(f'test/golden/cuda/k_coalesced/{k.name}.cu').write_text( \
#   cuda.emit(k, SM_86)); \
#   Path(f'test/golden/hip/k_coalesced/{k.name}.hip').write_text( \
#   hip.emit(k, GFX1200))"

GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_AM: Am = (1, 0, 1, 0)
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_coalesced_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    kern = build_k_kernel(lever_spec(GOLDEN_AM))
    where = GOLDEN / backend / "k_coalesced"
    assert sorted(p.name for p in where.iterdir()) == [f"{kern.name}.{ext}"]
    text = emit(kern, arch)
    assert text == (where / f"{kern.name}.{ext}").read_text()
    assert "syncwarp" not in text and "__syncthreads" not in text
