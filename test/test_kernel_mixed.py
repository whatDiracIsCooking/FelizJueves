"""The `mixed` lever (#346): an FP32 recurrence inside the FP64 boundary.

WHAT IS PINNED, all interpreted, nothing compiled or launched:

1. THE DECLARATION. `mixed` is a kernel-body lever (`levers.VOCABULARY`),
   legal on `Target.J` only, with its own reason on `K` (#347's); it is in
   the key, and it composes with `fused` (one Spec carries both).
2. THE LITERAL ARM IS UNTOUCHED. No lever: the very same `Kernel`; the
   goldens and `kernels.json`'s `source_sha256` are the byte check.
3. THE FP64 BOUNDARY, AS IR. Every parameter, the shared `J_p`
   accumulator, the screen, the reduction and the write are F64 exactly as
   in the literal kernel; the F32 declarations are exactly the narrowed
   operands (`PQ_d`, `theta`), the prelude (`t2`, `T`, `s`, `W{m}`) and the
   recurrence's nodes; every F32 -> F64 crossing is a `TO_F64` (the IR's
   hygiene checker, #338, refuses any other), and the contraction reads
   each target through one. Outside the ket loop the kernel is the literal
   one plus the provenance `Note`.
4. THE NUMBERS, one batch (L = 0..4, both (lp, lq) extremes too) and split
   (`passes.atomize`d batch kernels, every admitted (L <= 4, n_batches)),
   with and without `fused`: against `reference.pq.one_centre` contracted
   with the same `D_q` -- eqs. 20-22 -- at THE fp32 tier,
   `|J_p - want_p| <= F32_REL * SUM |[p|q] D_q|` (test/oracle/tolerance.py,
   the one declared fp32 tolerance; not widened, and not a tier chosen for
   this lever). The interpreter rounds every F32 operation where the
   device does (`interp._round32`, witnessed against the cards by
   test_f32_device.py), so this is the fp32 arithmetic itself, not an fp64
   run rounded at the end.
5. THE BUILD IS THE CLAIMED ONE. The same comparison MISSES the fp64 tier
   (`F64_REL`) from L = 1: the kernel really computes in fp32 -- a builder
   that silently stayed fp64 would pass item 4 and fail this.

WHY `F32_REL` AND NOT `assert_within`: the result is `J`, an F64 array by
the boundary invariant, while the computation's precision is F32; the
tolerance module's guard refuses a float64 result at the fp32 tier so that
an fp64 BUILD is never let off with it. Item 3 is what shows this build is
not an fp64 one, so the tier is applied to its F64 output directly.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from oracle.tolerance import F32_REL, F64_REL

from felizjueves import levers
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Fn,
    For,
    If,
    Index,
    Kernel,
    Note,
    PrecisionError,
    Scalar,
    Var,
    WarpReduce,
    walk,
)
from felizjueves.j_levers.fused import FUSED
from felizjueves.j_levers.mixed import MIXED, MIXED_NOTE, WIDENED, narrow, operands
from felizjueves.kernel import (
    JSH,
    bra_functions,
    build_pq_kernel,
    build_pq_kernels,
    builder_plan,
    ket_functions,
    launch_shape,
    lever_notes,
    pq_spec,
    split,
)
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.recurrence import PQ_PARAMS, THETA, node_name
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, theta_of
from felizjueves.spec import Spec, Target


def _admitted(L: int, n: int) -> bool:
    try:
        plan(pq_spec(L, n_batches=n))
    except ValueError:
        return False
    return True


SPLIT = [(L, n) for L in range(5) for n in (3, 6, 9) if _admitted(L, n)]
ARMS = {"mixed": (MIXED,), "fused+mixed": (FUSED, MIXED)}


def _mixed(L: int, n_batches: int = 1, arm: str = "mixed") -> Spec:
    return pq_spec(L, ARMS[arm], n_batches)


@cache
def _kernels(spec: Spec) -> tuple[Kernel, ...]:
    return build_pq_kernels(spec)


def _inputs(spec: Spec, n_bra: int, n_ket: int, seed: int) -> dict:
    """As test_kernel_fused.py's: kets sorted descending by Iq * Dmax."""
    rng = np.random.default_rng(seed)
    n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
    Iq = rng.uniform(0.2, 1.0, n_ket)
    Dmax = rng.uniform(0.5, 1.0, n_ket)
    srt = np.argsort(-(Iq * Dmax), kind="stable")
    return dict(
        n_bra=n_bra, n_ket=n_ket, tau=0.0,
        P=rng.uniform(-0.5, 0.5, (n_bra, 3)), zeta=rng.uniform(0.6, 1.4, n_bra),
        Ip=rng.uniform(0.5, 1.0, n_bra),
        Q=rng.uniform(-0.5, 0.5, (n_ket, 3)), eta=rng.uniform(0.6, 1.4, n_ket),
        Iq=Iq[srt].copy(), Dmax=Dmax[srt].copy(),
        D=rng.uniform(-1.0, 1.0, (n_ket, n_q)),
        J=np.zeros((n_bra, n_p)),
    )


def _run(spec: Spec, args: dict, lanes: int) -> np.ndarray:
    grid, block = launch_shape(args["n_bra"], lanes)
    J = np.zeros_like(args["J"])
    for k in _kernels(spec):
        it = Interpreter(k, launch=Launch(grid, block, lanes), boys=boys)
        J = it.run(**{**args, "J": J})["J"]
    return J


def _reference(spec: Spec, args: dict) -> tuple[np.ndarray, np.ndarray]:
    """J_p and its scale SUM |[p|q] D_q| from `reference/` in fp64."""
    lp, lq = split(spec)
    ps, qs = bra_functions(spec), ket_functions(spec)
    want = np.zeros((args["n_bra"], len(ps)))
    scale = np.zeros_like(want)
    for b in range(args["n_bra"]):
        for k in range(args["n_ket"]):
            if args["Ip"][b] * (args["Iq"][k] * args["Dmax"][k]) < args["tau"]:
                continue
            PQ = tuple(float(c) for c in args["P"][b] - args["Q"][k])
            th = theta_of(float(args["zeta"][b]), float(args["eta"][k]))
            R = {r: one_centre(r, 0, PQ, th) for r in hermite_targets(lp + lq)}
            for i, p in enumerate(ps):
                for j, q in enumerate(qs):
                    r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
                    t = (-1.0) ** sum(q) * R[r] * args["D"][k, j]
                    want[b, i] += t
                    scale[b, i] += abs(t)
    return want, scale


def _err(spec: Spec, seed: int, *, n_bra: int = 2, n_ket: int = 3,
         lanes: int = 2) -> float:
    """max |J - want| / scale, after asserting the fp32 tier."""
    args = _inputs(spec, n_bra=n_bra, n_ket=n_ket, seed=seed)
    got = _run(spec, args, lanes)
    assert got.dtype == np.float64  # the boundary: J stays fp64
    want, scale = _reference(spec, args)
    err = float(np.max(np.abs(got - want) / scale))
    assert err <= F32_REL, f"{spec.key()}: {err:.3e} > F32_REL {F32_REL:.3e}"
    return err


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_mixed_is_a_j_only_kernel_body_lever() -> None:
    lever = levers.VOCABULARY[MIXED]
    assert (lever.name, lever.issue, lever.legal) == (MIXED, 346,
                                                     frozenset({"j"}))
    assert pq_spec(4, (MIXED,)).kernel_body_levers() == (MIXED,)
    assert pq_spec(4, (FUSED, MIXED)).kernel_body_levers() == (FUSED, MIXED)

    def dags(p) -> tuple:
        return tuple((d.batch, d.targets, dict(d.parents)) for d in p.dags)

    assert dags(builder_plan(pq_spec(4, (MIXED,)))) == dags(plan(pq_spec(4)))
    with pytest.raises(ValueError, match="#347"):
        Spec(pq_spec(4).operator, 0, (1, 1, 1, 1), Target.K, (MIXED,))


def test_the_lever_is_in_the_key() -> None:
    assert pq_spec(4, (MIXED,)).key() != pq_spec(4).key()
    assert pq_spec(4, (MIXED,)).key().endswith("/j/mixed")
    assert pq_spec(4, (FUSED, MIXED)).key().endswith("/j/fused+mixed")


# ---------------------------------------------------------------------------
# 2-3. The literal arm, and the boundary as IR
# ---------------------------------------------------------------------------


def _loop(k: Kernel) -> For:
    (loop,) = [s for s in walk(k.body) if isinstance(s, For)]
    return loop


def _outside(k: Kernel) -> list:
    inner = {id(s) for s in walk(_loop(k).body)}
    return [s for s in walk(k.body) if id(s) not in inner
            and not isinstance(s, (For, If))]


@pytest.mark.parametrize("L", range(5))
def test_no_lever_is_the_literal_kernel(L: int) -> None:
    lit = build_pq_kernel(L)
    assert build_pq_kernel(pq_spec(L)) == lit
    assert not any(s.type is Scalar.F32 for s in walk(lit.body)
                   if isinstance(s, Decl))
    assert not any(isinstance(s, Note) and s.prov == MIXED_NOTE
                   for s in walk(lit.body))


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("L", range(5))
def test_the_fp64_boundary(L: int, arm: str) -> None:
    spec = _mixed(L, arm=arm)
    lit, mix = build_pq_kernel(replace(spec, levers=tuple(
        lv for lv in spec.levers if lv != MIXED))), build_pq_kernel(spec)
    # the parameters and everything outside the ket loop: the arm without
    # `mixed`, plus one provenance Note
    assert mix.params == lit.params
    assert all(p.type is not Scalar.F32 for p in mix.params)
    out = _outside(mix)
    prov = lever_notes(pq_spec(L, (MIXED,)))[0]
    assert out.count(prov) == 1
    out.remove(prov)
    assert out == _outside(lit)
    assert any(isinstance(s, WarpReduce) for s in out)
    (jsh,) = [s for s in walk(mix.body) if isinstance(s, Decl) and s.name == JSH]
    assert jsh.type is Scalar.F64
    # inside it: F32 is exactly the operands, the prelude and the nodes
    (dag,) = builder_plan(spec).dags
    nodes = {node_name(n) for n in schedule(dag)}
    top = max(sum(r) for r, _ in dag.targets)
    prelude = {"t2", "T", *(f"W{m}" for m in range(top + 1))} | (
        {"s"} if top else set())
    f32 = {s.name for s in walk(_loop(mix).body)
           if isinstance(s, Decl) and s.type is Scalar.F32}
    widened = {n + WIDENED for n in (node_name(t) for t in dag.targets)}
    if FUSED in spec.levers:
        assert f32 == {*PQ_PARAMS, THETA} | prelude | (
            nodes - {node_name(t) for t in dag.targets}) | widened
    else:
        assert f32 == {*PQ_PARAMS, THETA} | prelude | nodes
    f64 = {s.name for s in walk(_loop(mix).body)
           if isinstance(s, Decl) and s.type is Scalar.F64}
    assert f64 == {"bound", "et"} | (
        {node_name(t) for t in dag.targets} if FUSED in spec.levers else set())
    # every J_p update is F64 and reads its targets through TO_F64
    updates = [s for s in walk(_loop(mix).body) if isinstance(s, Assign)
               and isinstance(s.dst, Index) and s.dst.array == JSH]
    assert updates
    if FUSED not in spec.levers:
        def reads(e) -> list:
            if isinstance(e, Call):
                if e.fn is Fn.TO_F64:
                    return [("wide", e.args[0])]
                return [x for a in e.args for x in reads(a)]
            if isinstance(e, Var):
                return [("bare", e)]
            return []

        got = [r for u in updates for r in reads(u.value)]
        targets = {node_name(t) for t in dag.targets}
        assert {v.name for how, v in got if how == "wide"} == targets
        assert not [v for how, v in got if how == "bare" and v.name in targets]


def test_the_hygiene_checker_holds_the_boundary() -> None:
    """Drop one widening and the kernel refuses to build: F32 cannot leak
    into the F64 accumulator unannounced (#338's checker, not this lever's
    promise)."""
    spec = _mixed(2)
    (dag,) = builder_plan(spec).dags
    k = build_pq_kernel(spec)
    loop = _loop(k)
    bad, done = [], False
    for s in loop.body:
        if isinstance(s, Assign) and isinstance(s.dst, Index) \
                and s.dst.array == JSH and not done:
            bad.append(Assign(s.dst, Call(Fn.ADD, (s.dst, Var(node_name(
                next(iter(dag.targets))))))))
            done = True
        else:
            bad.append(s)
    assert done and len(bad) == len(loop.body)

    def swap(body):
        return tuple(replace(s, body=tuple(bad)) if s is loop else
                     replace(s, then=swap(s.then)) if isinstance(s, If)
                     else s for s in body)

    with pytest.raises(PrecisionError):
        Kernel(k.name, k.params, swap(k.body))


def test_narrow_and_operands_refuse_what_they_cannot_narrow() -> None:
    with pytest.raises(ValueError, match="not declared"):
        operands(())
    with pytest.raises(ValueError, match="declares F32"):
        narrow((Decl("R0_0_0_0", Scalar.F32),), (), widen_targets=False)


def test_the_split_form_is_atomized_and_keeps_the_lever() -> None:
    from felizjueves.passes import ATOMIC_NOTE
    ks = _kernels(_mixed(4, 3))
    lit = _kernels(pq_spec(4, n_batches=3))
    assert [k.name for k in ks] == [k.name for k in lit]
    for k in ks:
        provs = [s.prov for s in walk(k.body) if isinstance(s, Note)]
        assert ATOMIC_NOTE in provs and MIXED_NOTE in provs


# ---------------------------------------------------------------------------
# 4-5. The numbers, at the fp32 tier, and the build is fp32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("L", range(5))
def test_one_batch_meets_the_fp32_tier(L: int, arm: str) -> None:
    for lp, lq in sorted({split(pq_spec(L)), (L, 0), (0, L)}):
        spec = Spec(pq_spec(L).operator, 0, (lp, 0, lq, 0), Target.J,
                    ARMS[arm])
        err = _err(spec, seed=346 + L)
        if L >= 1:
            assert err > F64_REL, (
                f"{spec.key()}: {err:.3e} meets the fp64 tier -- the "
                "recurrence is not running in fp32")


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize(("L", "n_batches"), SPLIT)
def test_split_class_meets_the_fp32_tier(L: int, n_batches: int,
                                         arm: str) -> None:
    _err(_mixed(L, n_batches, arm), seed=3460 + 10 * L + n_batches)


def test_a_wide_warp_screened_kets_and_a_wide_t_spread() -> None:
    spec = _mixed(4)
    args = _inputs(spec, n_bra=3, n_ket=40, seed=46)
    args["Q"] = args["Q"] * 8.0  # T up to ~ 1e2: the Boys argument's range
    args["tau"] = float(np.median(args["Ip"][1] * args["Iq"] * args["Dmax"]))
    got = _run(spec, args, lanes=32)
    want, scale = _reference(spec, args)
    assert np.all(np.abs(got - want) <= F32_REL * scale), (
        np.max(np.abs(got - want) / scale))
