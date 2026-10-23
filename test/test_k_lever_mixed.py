"""Mixed-precision K and the `D`/`E` operand narrowing (#347), interpreted.

Two arms of one axis, `k_levers/mixed.py`: `kmixed` (the compute-only arm,
the `[p|q]` pair region F32) and `kmixed-de` (that, plus `E^ac` / `E^bd`
formed in F32 and `XPA XPC XQB XQD Dcd` as host-packed `const float*`).
What is asserted, nothing compiled or launched:

1. THE DECLARATION. Both are kernel-body levers legal on `Target.K` only,
   with their own reasons elsewhere, registered in `k_levers.REGISTRY`, in
   the key; one spec carrying both is refused (all or nothing: two points,
   not two composable changes). `mixed` stays J's.
2. THE FP64 ARM IS UNTOUCHED. No lever: `DEFAULT_HOOKS` and no F32 anywhere
   (the literal K kernels' digest is test/test_k_seams.py's pin, and their
   bitwise agreement with `ao.k.k_quartet` test/test_kernel_k.py's).
3. THE FP64 BOUNDARY, AS IR. `kmixed`: the parameters are the literal ones;
   the F32 declarations are exactly the narrowed operands (`PQ_d`,
   `theta`), the prelude (`t2`, `T`, `s`, `W{m}`) and the recurrence's
   nodes; `Ksh` (the `I` column and the `K` accumulator) is F64; every
   stage-1 read of a target is a `TO_F64`. `kmixed-de`: also exactly the
   five F32 parameters and the two F32 `E` arrays, every `E` and `D_cd`
   read a `TO_F64`. Dropping one widening makes the kernel refuse to build
   (#338's checker holds the boundary, not this lever).
4. THE NUMBERS, at THE fp32 tier: per bra record, `|K - X| <= F32_REL * S`
   with `X, S = ao.k._k_quartet(...)` on `reference.pq` in fp64 -- the
   host staging and its |.| twin, the scale test_kernel_device.py's K
   harness uses -- on the UNNARROWED inputs (so `kmixed-de`'s packing
   error is inside the budget, not excused). `F32_REL` is
   test/oracle/tolerance.py's one declared fp32 tier, not widened. Every K
   class with L <= 2 at one batch and, where the plan admits it, three
   (fast); a sample of L = 3, 4 classes with no shell above d (`slow`,
   milestone 13's cap). The same comparison MISSES the fp64 tier: the
   build really is fp32. The interpreter rounds every F32 operation where
   the device does (`interp._round32`).
5. COMPOSITION with #341's `split`: the element kernels of `kmixed(+de)` +
   `split` are, together, the arm's block kernel BIT FOR BIT (split prunes
   before this lever's `finish`, which narrows by name).
6. THE HOST PACKING, `ao.pairs.narrowed`: one round-to-nearest per value,
   numpy's float32 exactly, an overflow refused.
7. THE GOLDENS: `(p s|s s)`, one batch, each arm, both backends.

WHY `F32_REL` AND NOT `assert_within`: `K` is an F64 array by the boundary
invariant while the computation is F32 (test/test_kernel_mixed.py, the J
twin, says the same); item 3 is what shows this build is not an fp64 one.

Fast tier unless `slow`: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import itertools
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_kernel_k as tk
from oracle.tolerance import F32_REL, F64_REL

from felizjueves import k_levers, levers
from felizjueves.ao.k import _k_quartet
from felizjueves.ao.pairs import narrowed
from felizjueves.emit import cuda, hip
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Fn,
    For,
    If,
    Index,
    Inline,
    Kernel,
    PrecisionError,
    Scalar,
    Space,
    Var,
    WarpReduce,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.k_levers import mixed, split
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KET, KSH
from felizjueves.kernel_k import build_k_kernel, build_k_kernels, launch_shape
from felizjueves.mmd.kernel import builder_plan
from felizjueves.mmd.recurrence import PQ_PARAMS, THETA, node_name
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.plan.schedule import schedule
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq as reference_pq
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]
ARMS = mixed.ARMS

FAST: list[Am] = tk.FAST  # every K class with L <= 2
SLOW_SAMPLE: list[Am] = [(1, 1, 1, 0), (1, 0, 1, 1), (2, 0, 1, 0),
                         (1, 1, 1, 1), (2, 0, 2, 0), (2, 1, 1, 0),
                         (0, 2, 1, 1), (2, 2, 0, 0)]
assert all(2 < sum(am) <= 4 and max(am) <= 2 for am in SLOW_SAMPLE)


def lever_spec(am: Am, arm: str, n_batches: int = 1,
               more: tuple[str, ...] = ()) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, (arm, *more), n_batches)


def literal_spec(am: Am, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, am, Target.K, (), n_batches)


def _admitted(am: Am, n: int) -> bool:
    try:
        builder_plan(literal_spec(am, n))
    except ValueError:
        return False
    return True


@cache
def _kernels(spec: Spec) -> tuple[Kernel, ...]:
    return build_k_kernels(spec)


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
def test_both_arms_are_k_only_kernel_body_levers(arm: str) -> None:
    lever = levers.VOCABULARY[arm]
    assert (lever.name, lever.issue, lever.legal) == (arm, 347,
                                                      frozenset({"k"}))
    assert "#346" in lever.refusals["j"]
    assert "#355" in lever.refusals["f"]
    for slot in ("dq", "jab", "s", "t", "v"):
        assert lever.refusals[slot] == levers.DEFAULT_REFUSALS[slot]
    with pytest.raises(ValueError, match="#346"):
        Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J, (arm,))
    assert k_levers.REGISTRY[arm] is mixed.apply
    spec = lever_spec((1, 0, 1, 0), arm)
    assert spec.key() == f"coulomb/d0/(1,0|1,0)/b1/k/{arm}"
    assert spec.kernel_body_levers() == (arm,)
    assert k_levers.hooks_for(spec) != DEFAULT_HOOKS
    # `mixed` (#346) is still J's, with #347 named as the K instance
    assert levers.VOCABULARY["mixed"].legal == frozenset({"j"})
    assert "#347" in levers.VOCABULARY["mixed"].refusals["k"]


def test_the_two_arms_do_not_compose() -> None:
    spec = Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.K, ARMS)
    with pytest.raises(ValueError, match="all or nothing"):
        build_k_kernel(spec)


def test_the_hooks_each_arm_moves() -> None:
    import dataclasses as dc

    from felizjueves.k_levers.seams import KHooks

    def moved(arm: str) -> set[str]:
        h = mixed.apply(DEFAULT_HOOKS, lever_spec((1, 0, 1, 0), arm))
        return {f.name for f in dc.fields(KHooks) if not h.is_default(f.name)}

    assert moved(mixed.COMPUTE) == {"finish"}
    assert moved(mixed.FULL) == {"finish", "bind_e_ac", "e_ac", "bind_e_bd",
                                 "e_bd"}


# ---------------------------------------------------------------------------
# 2-3. The literal arm, and the boundary as IR
# ---------------------------------------------------------------------------


def _loop(k: Kernel) -> For:
    (loop,) = [s for s in walk(k.body) if isinstance(s, For) and s.var == KET]
    return loop


def _f32_decls(body) -> set[str]:
    return {s.name for s in walk(body, into_inline=False)
            if isinstance(s, Decl) and s.type is Scalar.F32}


@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST])
def test_no_lever_is_the_literal_kernel(am: Am) -> None:
    lit = build_k_kernel(am)
    assert k_levers.hooks_for(literal_spec(am)) is DEFAULT_HOOKS
    assert build_k_kernels(literal_spec(am)) == (lit,)
    assert not _f32_decls(lit.body)
    assert all(p.type is not Scalar.F32 for p in lit.params)


def _reads(e, name: str) -> list[str]:
    """How each read of `name` in `e` is taken: "wide" under a TO_F64,
    else "bare"."""
    if isinstance(e, Call):
        if e.fn is Fn.TO_F64:
            inner = e.args[0]
            if (isinstance(inner, Var) and inner.name == name) or (
                    isinstance(inner, Index) and inner.array == name):
                return ["wide"]
        return [x for a in e.args for x in _reads(a, name)]
    if isinstance(e, Var):
        return ["bare"] if e.name == name else []
    if isinstance(e, Index):
        own = ["bare"] if e.array == name else []
        return own + [x for i in e.at for x in _reads(i, name)]
    return []


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("am", [pytest.param(a, id=tk._id(a)) for a in FAST])
def test_the_fp64_boundary(am: Am, arm: str) -> None:
    spec = lever_spec(am, arm)
    lit, mix = build_k_kernel(am), build_k_kernel(spec)
    # the parameters
    f32_params = {p.name for p in mix.params if p.type is Scalar.F32}
    assert [p.name for p in mix.params] == [p.name for p in lit.params]
    assert f32_params == (set(mixed.NARROWED_PARAMS) if arm == mixed.FULL
                          else set())
    # the shared I column and K accumulator, the reductions: F64
    (ksh,) = [s for s in walk(mix.body) if isinstance(s, Decl)
              and s.space is Space.SHARED]
    assert ksh.name == KSH and ksh.type is Scalar.F64
    assert all(isinstance(s.dst, Var) for s in walk(mix.body)
               if isinstance(s, WarpReduce))
    assert {s.type for s in walk(mix.body) if isinstance(s, Decl)
            and s.name.startswith("Kab")} == {Scalar.F64}
    # F32 is exactly the operands, the prelude and the nodes (+ E arrays)
    (dag,) = builder_plan(spec).dags
    nodes = {node_name(n) for n in schedule(dag)}
    top = max(sum(r) for r, _ in dag.targets)
    prelude = {"t2", "T", *(f"W{m}" for m in range(top + 1))} | (
        {"s"} if top else set())
    want = {*PQ_PARAMS, THETA} | prelude | nodes
    assert _f32_decls(_loop(mix).body) == want | (
        {"Ebd"} if arm == mixed.FULL else set())
    assert _f32_decls(mix.body) == _f32_decls(_loop(mix).body) | (
        {"Eac"} if arm == mixed.FULL else set())
    # the E fragments: F32 inside under kmixed-de, untouched under kmixed
    for s in walk(mix.body, into_inline=False):
        if isinstance(s, Inline):
            types = {d.type for d in walk(s.fragment.body)
                     if isinstance(d, Decl)}
            assert types == ({Scalar.F32} if arm == mixed.FULL
                             else {Scalar.F64})
    # every Ksh update reads each target, E entry and D_cd through TO_F64
    updates = [s for s in walk(_loop(mix).body) if isinstance(s, Assign)
               and isinstance(s.dst, Index) and s.dst.array == KSH]
    assert updates
    targets = {node_name(t) for t in dag.targets}
    narrowed_arrays = ("Ebd", "Eac", "Dcd") if arm == mixed.FULL else ()
    seen = set()
    for u in updates:
        for name in (*targets, *narrowed_arrays):
            how = _reads(u.value, name)
            assert "bare" not in how, (name, u)
            if how:
                seen.add(name)
    assert seen == targets | set(narrowed_arrays)


@pytest.mark.parametrize("arm", ARMS)
def test_the_hygiene_checker_holds_the_boundary(arm: str) -> None:
    """Strip every `TO_F64` from one stage-1 update and the kernel refuses
    to build: F32 cannot leak into the F64 `I` column unannounced."""
    k = build_k_kernel(lever_spec((1, 0, 1, 0), arm))
    loop = _loop(k)

    def strip(e):
        if isinstance(e, Call):
            if e.fn is Fn.TO_F64:
                return strip(e.args[0])
            return Call(e.fn, tuple(strip(a) for a in e.args))
        return e

    i = next(n for n, s in enumerate(loop.body) if isinstance(s, Assign)
             and isinstance(s.dst, Index) and s.dst.array == KSH)
    bad = list(loop.body)
    bad[i] = dataclasses.replace(bad[i], value=strip(bad[i].value))

    def swap(body):
        return tuple(dataclasses.replace(s, body=tuple(bad)) if s is loop else
                     dataclasses.replace(s, then=swap(s.then))
                     if isinstance(s, If) else s for s in body)

    with pytest.raises(PrecisionError):
        Kernel(k.name, k.params, swap(k.body))


def test_f32_fragment_refuses_what_it_cannot_narrow() -> None:
    from felizjueves.ir import Fragment, Lit

    frag = Fragment("probe", ("x",), (Decl("y", Scalar.F64, init=Call(
        Fn.MUL, (Var("x"), Lit(0.5)))),))
    with pytest.raises(ValueError, match="float literal"):
        mixed.f32_fragment(frag)


def test_a_changed_recurrence_is_refused_not_narrowed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`finish` narrows the literal pair region only: a node that is not the
    literal recurrence's statement is refused."""
    from felizjueves.mmd import recurrence

    real = recurrence.recurrence_body

    def other(dag, order, fundamental):
        body = list(real(dag, order, fundamental))
        i = next(n for n, s in enumerate(body) if isinstance(s, Decl))
        body[i] = dataclasses.replace(body[i], init=Call(
            Fn.ADD, (body[i].init, body[i].init)))
        return tuple(body)

    monkeypatch.setattr("felizjueves.kernel_k.recurrence_body", other)
    with pytest.raises(ValueError, match="not the literal recurrence"):
        build_k_kernel(lever_spec((1, 0, 1, 0), mixed.COMPUTE))


def test_the_split_form_is_atomized_and_keeps_the_lever() -> None:
    from felizjueves.ir import AtomicAdd, Note

    for arm in ARMS:
        ks = _kernels(lever_spec((1, 0, 1, 0), arm, 3))
        lit = _kernels(literal_spec((1, 0, 1, 0), 3))
        assert [k.name for k in ks] == [k.name for k in lit]
        note = mixed.COMPUTE_NOTE if arm == mixed.COMPUTE else mixed.FULL_NOTE
        for k in ks:
            assert any(isinstance(s, AtomicAdd) for s in walk(k.body))
            provs = [s.prov for s in walk(k.body) if isinstance(s, Note)]
            assert note in provs and levers.provenance(arm) in provs


# ---------------------------------------------------------------------------
# 4. The numbers, at the fp32 tier
# ---------------------------------------------------------------------------


def _args(am: Am, arm: str) -> tuple[dict, object, object, list[float]]:
    bra, ket = tk._pairs(am)
    dcd = tk._dcd(am[1], am[3])
    args = tk.k_args(am, bra, [(ket, 0)], [dcd])
    args["K"] = np.zeros_like(args["K"])
    if arm == mixed.FULL:  # the host's packing: narrowed copies, fp32
        for n in mixed.NARROWED_PARAMS:
            a = args[n]
            args[n] = np.array(narrowed(a.ravel()),
                               dtype=np.float32).reshape(a.shape)
    return args, bra, ket, dcd


def _err(am: Am, arm: str, n_batches: int = 1, more: tuple[str, ...] = ()
         ) -> float:
    """max over elements of |K - X| / S, after asserting the fp32 tier."""
    args, bra, ket, dcd = _args(am, arm)
    grid, block = launch_shape(args["n_bra"], 1)
    K = args["K"]
    for k in _kernels(lever_spec(am, arm, n_batches, more)):
        K = Interpreter(k, launch=Launch(grid, block, 1), boys=boys).run(
            **{**args, "K": K})["K"]
    assert K.dtype == np.float64  # the boundary: K stays fp64
    worst = 0.0
    for i in range(len(bra.recs)):
        X, S = _k_quartet(tk._one(bra, i), ket, dcd, pq=reference_pq,
                          want_scale=True)
        X, S = np.array(X).ravel(), np.array(S).ravel()
        assert np.all(np.abs(K[i] - X) <= F32_REL * S), (
            f"{arm} {am} b{n_batches}: {np.max(np.abs(K[i] - X) / S):.3e} "
            f"> F32_REL {F32_REL:.3e}")
        worst = max(worst, float(np.max(np.abs(K[i] - X) / S)))
    return worst


CELLS = ([pytest.param(am, n, id=f"{tk._id(am)}-b{n}")
          for am in FAST for n in (1, 3) if _admitted(am, n)]
         + [pytest.param(am, n, id=f"{tk._id(am)}-b{n}",
                         marks=pytest.mark.slow)
            for am in SLOW_SAMPLE for n in (1, 3) if _admitted(am, n)])


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize(("am", "n_batches"), CELLS)
def test_each_bra_record_meets_the_fp32_tier(am: Am, n_batches: int,
                                             arm: str) -> None:
    err = _err(am, arm, n_batches)
    assert err > F64_REL, (
        f"{arm} {am}: {err:.3e} meets the fp64 tier -- not running in fp32")


# ---------------------------------------------------------------------------
# 5. Composition with split (#341)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("am", [(1, 0, 1, 0), (1, 1, 0, 0), (1, 0, 0, 1)],
                         ids=tk._id)
def test_split_composes_bitwise(am: Am, arm: str) -> None:
    """The element kernels of the arm + `split` write, together, the arm's
    block kernel's `K` bit for bit: `split` keeps the literal operations per
    element and prunes only dead nodes, and the narrowing is by name."""
    args, _, _, _ = _args(am, arm)
    grid, block = launch_shape(args["n_bra"], 1)

    def run(kernels) -> np.ndarray:
        K = np.full_like(args["K"], np.nan)
        for k in kernels:
            K = Interpreter(k, launch=Launch(grid, block, 1), boys=boys).run(
                **{**args, "K": K})["K"]
        return K

    both = build_k_kernels(lever_spec(am, arm, 1, (split.NAME,)))
    assert len(both) > 1
    for k in both:
        assert _f32_decls(_loop(k).body)
    got = run(both)
    assert not np.isnan(got).any()
    assert got.tolist() == run(build_k_kernels(lever_spec(am, arm))).tolist()


# ---------------------------------------------------------------------------
# 6. The host packing
# ---------------------------------------------------------------------------


def test_narrowed_is_one_round_to_nearest_float32() -> None:
    rng = np.random.default_rng(347)
    x = np.concatenate([rng.normal(size=200), rng.uniform(-1e30, 1e30, 50),
                        [0.0, -0.0, 1.0 + 2.0**-24, 1.0 + 3 * 2.0**-24]])
    got = narrowed(x)
    assert isinstance(got, tuple) and len(got) == len(x)
    assert np.array(got, dtype=np.float64).tolist() == \
        x.astype(np.float32).astype(np.float64).tolist()
    assert got[-2] == 1.0 and got[-1] == 1.0 + 2.0**-22  # ties to even
    assert x[0] == np.float64(x[0])  # the input is untouched (a copy)
    with pytest.raises(ValueError, match="overflows"):
        narrowed([1.0, 1e39])


# ---------------------------------------------------------------------------
# 7. The goldens: (p s|s s), one batch, each arm, both backends
# ---------------------------------------------------------------------------
#
# Regenerate (never hand-edit):
#
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.spec import Spec, Operator, Target; \
#   from felizjueves.kernel_k import build_k_kernel; \
#   from felizjueves.performance.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   [(Path(f'test/golden/cuda/k_mixed/k_10_00_{a}.cu').write_text(cuda.emit( \
#   k, SM_86)), Path(f'test/golden/hip/k_mixed/k_10_00_{a}.hip').write_text( \
#   hip.emit(k, GFX1200))) for a in ('kmixed', 'kmixed-de') for k in \
#   [build_k_kernel(Spec(Operator.COULOMB, 0, (1, 0, 0, 0), Target.K, \
#   (a,)))]]"

GOLDEN = Path(__file__).resolve().parent / "golden"
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"),
            "hip": (hip.emit, GFX1200, "hip")}


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_mixed_goldens_are_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    where = GOLDEN / backend / "k_mixed"
    assert sorted(p.name for p in where.iterdir()) == sorted(
        f"k_10_00_{a}.{ext}" for a in ARMS)
    for arm in ARMS:
        kern = build_k_kernel(lever_spec((1, 0, 0, 0), arm))
        assert emit(kern, arch) == (where / f"k_10_00_{arm}.{ext}").read_text()


def test_the_slow_sample_is_within_the_cap() -> None:
    assert all(sum(am) <= 4 for am in SLOW_SAMPLE)
    assert {sum(am) for am in SLOW_SAMPLE} == {3, 4}
    assert list(itertools.chain(FAST)) == tk.FAST
