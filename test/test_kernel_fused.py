"""The `fused` lever (#344): each [r]^(0) contracted as the recurrence forms it.

WHAT IS PINNED, all interpreted, nothing compiled or launched:

1. THE DECLARATION. `fused` is a kernel-body lever (`levers.VOCABULARY`),
   legal on `Target.J` only, with its own reason on `K`.
2. THE LITERAL ARM IS UNTOUCHED. No lever: the very same `Kernel` as before
   (the goldens and `kernels.json`'s `source_sha256` are the byte check).
3. WHAT CHANGES, AS IR. The fused kernel differs from the literal one only
   inside the ket loop: the same statements, the literal contraction block
   gone, and one shared update per contracted (p, q) pair placed directly
   after the `Decl` of its target `[p+q]^(0)`. Nothing reads a target after
   its updates: one target live at a time, and `lower.KernelCost`'s
   `peak_live` is no larger than the literal kernel's at any L, and strictly
   smaller from L = 2 (L = 0..4 fast, 5..8 `slow`).
4. THE NUMBERS, one batch (L = 0..4 fast, 5..8 `slow`) and split
   (`passes.atomize`d batch kernels: every admitted (L <= 4, n_batches)
   fast, L = 7 x {3, 6} `slow`), against `reference.pq.one_centre`
   contracted with the same `D_q` -- eqs. 20-22 -- at test_kernel.py's
   tolerance, UNCHANGED: `|J_p - want_p| <= 1e-12 * SUM |[p|q] D_q|`.

THE LEVER IS NOT BITWISE against the literal kernel, by construction: the
literal kernel adds a slot's terms for one ket as a balanced tree and then
once into the slot; the fused one adds them into the slot one at a time in
the schedule's target order. Same products, same signs, different
association. So the two are held to each other by the summation bound, not
by equality: each is a sum of the same `n` terms per (bra, p) in some order,
so each is within `(n - 1) u SUM|t|` of the exact sum, and the two are
within `2 (n - 1) u * scale` of each other (u = 2^-53, `scale = SUM|t|`,
`n` = n_ket * n_q). That bound is derived, and it is far inside the
reference pin above, which is the one both arms answer to.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from tiers import FAST_L_MAX, split_by_l

from felizjueves import levers
from felizjueves.eri.j_coulomb.levers.fused import FUSED, FUSED_NOTE, fuse
from felizjueves.ir import Assign, Decl, For, If, Index, Kernel, Note, Var, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.ir.lower import kernel_cost
from felizjueves.ir.passes import ATOMIC_NOTE
from felizjueves.mmd.kernel import (
    JSH,
    batch_contraction,
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
from felizjueves.mmd.recurrence import node_name
from felizjueves.plan.pipeline import plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, theta_of
from felizjueves.spec import Spec, Target

U = 2.0**-53
J_REL = 1e-12  # test_kernel.py's pin, unchanged


def _admitted(L: int, n: int) -> bool:
    try:
        plan(pq_spec(L, n_batches=n))
    except ValueError:
        return False
    return True


# L = 0..8, `slow` above test/tiers.py's cut: the numbers (4.) and the IR
# checks (3.) alike build the class.
ONE_BATCH = split_by_l(range(9), lambda L: L)
SPLIT = [(L, n) for L in range(FAST_L_MAX + 1) for n in (3, 6, 9)
         if _admitted(L, n)]
SPLIT_SLOW = [pytest.param(7, n, marks=pytest.mark.slow) for n in (3, 6)]


def _fused(L: int, n_batches: int = 1) -> Spec:
    return pq_spec(L, (FUSED,), n_batches)


@cache
def _kernels(spec: Spec) -> tuple[Kernel, ...]:
    return build_pq_kernels(spec)


def _inputs(spec: Spec, n_bra: int, n_ket: int, seed: int) -> dict:
    """As test_kernel.py's: kets sorted descending by Iq * Dmax, J zeroed."""
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
    """Every batch kernel in Plan.dags order, J threaded through from zero."""
    grid, block = launch_shape(args["n_bra"], lanes)
    J = np.zeros_like(args["J"])
    for k in _kernels(spec):
        it = Interpreter(k, launch=Launch(grid, block, lanes), boys=boys)
        J = it.run(**{**args, "J": J})["J"]
    return J


def _reference(spec: Spec, args: dict) -> tuple[np.ndarray, np.ndarray]:
    """J_p = SUM_k [bound_k >= tau] SUM_q (-1)^|q| [p+q]^(0) D_q and its
    scale SUM_k SUM_q |[p|q] D_q| ([JEngine, v2, eqs. 20-22]); no plan, no
    batching, no kernel."""
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


def _check(spec: Spec, seed: int, *, n_bra: int = 2, n_ket: int = 3,
           lanes: int = 2) -> None:
    args = _inputs(spec, n_bra=n_bra, n_ket=n_ket, seed=seed)
    got = _run(spec, args, lanes)
    want, scale = _reference(spec, args)
    assert got.shape == want.shape
    assert np.all(np.abs(got - want) <= J_REL * scale), (
        np.max(np.abs(got - want) / scale))
    # ... and against the literal arm, by the summation bound (docstring)
    literal = replace(spec, levers=())
    lit = _run(literal, args, lanes)
    n = n_ket * len(ket_functions(spec))
    assert np.all(np.abs(got - lit) <= 2 * max(n - 1, 1) * U * scale), (
        np.max(np.abs(got - lit) / scale))


# ---------------------------------------------------------------------------
# 1. The declaration
# ---------------------------------------------------------------------------


def test_fused_is_a_j_only_kernel_body_lever() -> None:
    lever = levers.VOCABULARY[FUSED]
    assert (lever.name, lever.issue, lever.legal) == (FUSED, 344,
                                                     frozenset({"j"}))
    assert levers.is_kernel_lever(FUSED)
    assert pq_spec(4, (FUSED,)).kernel_body_levers() == (FUSED,)
    def dags(p) -> tuple:
        return tuple((d.batch, d.targets, dict(d.parents)) for d in p.dags)

    assert dags(builder_plan(pq_spec(4, (FUSED,)))) == dags(plan(pq_spec(4)))
    with pytest.raises(ValueError, match="staged through both E fragments"):
        Spec(pq_spec(4).operator, 0, (1, 1, 1, 1), Target.K, (FUSED,))


def test_the_lever_is_in_the_key() -> None:
    assert pq_spec(4, (FUSED,)).key() != pq_spec(4).key()
    assert FUSED in pq_spec(4, (FUSED,)).key()


# ---------------------------------------------------------------------------
# 2-3. The literal arm, and what the lever changes as IR
# ---------------------------------------------------------------------------


def _loop(k: Kernel) -> For:
    (loop,) = [s for s in walk(k.body) if isinstance(s, For)]
    return loop


def _jsh_updates(body) -> list[Assign]:
    return [s for s in body if isinstance(s, Assign)
            and isinstance(s.dst, Index) and s.dst.array == JSH]


def _names(e) -> set[str]:
    out: set[str] = set()
    if isinstance(e, Var):
        out.add(e.name)
    for f in getattr(e, "args", ()) or ():
        out |= _names(f)
    for t in getattr(e, "terms", ()) or ():
        out |= _names(t.src)
    if isinstance(e, Index):
        for a in e.at:
            out |= _names(a)
    return out


@pytest.mark.parametrize("L", range(5))
def test_no_lever_is_the_literal_kernel(L: int) -> None:
    assert build_pq_kernel(pq_spec(L)) == build_pq_kernel(L)
    assert not any(isinstance(s, Note) and s.prov == FUSED_NOTE
                   for s in walk(build_pq_kernel(L).body))


@pytest.mark.parametrize("L", ONE_BATCH)
def test_only_the_ket_loop_tail_changes(L: int) -> None:
    lit, fus = build_pq_kernel(L), build_pq_kernel(_fused(L))
    assert lit.name == fus.name and lit.params == fus.params
    # outside the ket loop: the literal kernel's leaf statements, plus the
    # lever's provenance Note
    def outside(k: Kernel) -> list:
        inner = {id(s) for s in walk(_loop(k).body)}
        return [s for s in walk(k.body) if id(s) not in inner
                and not isinstance(s, (For, If))]

    (prov,) = lever_notes(_fused(L))
    fus_out = outside(fus)
    assert fus_out.count(prov) == 1
    fus_out.remove(prov)
    assert fus_out == outside(lit)
    lb, fb = _loop(lit).body, _loop(fus).body
    # every literal non-contraction statement survives, in order
    lit_rest = [s for s in lb if s not in _jsh_updates(lb)
                and not (isinstance(s, Note) and s.prov.startswith("J_p +="))]
    fus_rest = [s for s in fb if s not in _jsh_updates(fb)
                and not (isinstance(s, Note) and s.prov == FUSED_NOTE)]
    assert fus_rest == lit_rest
    # one update per contracted (p, q) pair
    (dag,) = plan(pq_spec(L)).dags
    pairs = sum(len(qs) for _, _, qs in batch_contraction(pq_spec(L), dag))
    assert len(_jsh_updates(fb)) == pairs


@pytest.mark.parametrize("L", ONE_BATCH)
def test_each_target_is_consumed_right_after_it_is_formed(L: int) -> None:
    body = list(_loop(build_pq_kernel(_fused(L))).body)
    targets = {node_name((r, 0)) for r in hermite_targets(L)}
    formed = {s.name: i for i, s in enumerate(body)
              if isinstance(s, Decl) and s.name in targets}
    assert set(formed) == targets
    for name, i in formed.items():
        # its updates follow it directly, and nothing after them reads it
        j = i + 1
        while j < len(body) and isinstance(body[j], Assign) \
                and name in _names(body[j].value):
            j += 1
        assert j > i + 1, f"{name} is formed and never contracted"
        later = [s for s in body[j:] if name in _names(
            getattr(s, "init", None) or getattr(s, "value", None))]
        assert not later, f"{name} is still read after its contraction"


@pytest.mark.parametrize("L", ONE_BATCH)
def test_peak_live_never_grows_and_falls_from_l2(L: int) -> None:
    lit = kernel_cost(build_pq_kernel(L), registers=64).peak_live
    fus = kernel_cost(build_pq_kernel(_fused(L)), registers=64).peak_live
    assert fus <= lit
    if L >= 2:
        assert fus < lit, (L, lit, fus)


def test_fuse_refuses_a_target_the_recurrence_never_forms() -> None:
    spec = pq_spec(2)
    (dag,) = plan(spec).dags
    touched = batch_contraction(spec, dag)
    with pytest.raises(ValueError, match="never formed"):
        fuse((), touched, jsh=lambda s: Index(JSH, ()),
             density=lambda j: Var("d"))
    with pytest.raises(ValueError, match="contracts nothing"):
        fuse((), (), jsh=lambda s: Index(JSH, ()), density=lambda j: Var("d"))


def test_the_split_form_is_atomized_and_keeps_the_lever() -> None:
    ks = _kernels(_fused(4, 3))
    lit = _kernels(pq_spec(4, n_batches=3))
    assert [k.name for k in ks] == [k.name for k in lit]
    for k in ks:
        provs = [s.prov for s in walk(k.body) if isinstance(s, Note)]
        assert ATOMIC_NOTE in provs and FUSED_NOTE in provs


# ---------------------------------------------------------------------------
# 4. The numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", ONE_BATCH)
def test_one_batch_matches_the_reference(L: int) -> None:
    for lp, lq in sorted({split(pq_spec(L)), (L, 0), (0, L)}):
        spec = Spec(pq_spec(L).operator, 0, (lp, 0, lq, 0), Target.J,
                    (FUSED,))
        _check(spec, seed=344 + L)


@pytest.mark.parametrize(("L", "n_batches"), [*SPLIT, *SPLIT_SLOW])
def test_split_class_matches_the_reference(L: int, n_batches: int) -> None:
    _check(_fused(L, n_batches), seed=3440 + 10 * L + n_batches)


def test_a_wide_warp_and_screened_kets() -> None:
    spec = _fused(4)
    args = _inputs(spec, n_bra=3, n_ket=40, seed=44)
    args["tau"] = float(np.median(args["Ip"][1] * args["Iq"] * args["Dmax"]))
    got = _run(spec, args, lanes=32)
    want, scale = _reference(spec, args)
    assert np.all(np.abs(got - want) <= J_REL * scale)
