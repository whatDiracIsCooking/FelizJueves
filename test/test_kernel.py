"""kernel.py: the whole [p|q] kernel of [JEngine, v2, Alg. 1], interpreted.

THE PIN. `build_pq_kernel` for every class the plan layer can schedule at
n_batches = 1 (L = 0..16, default split plus bra-only and ket-only splits) is
run in the interpreter over a few bra and ket indices, and `J_p` is compared
against `reference.pq.one_centre` contracted with the same `D_q` -- a path with
no MILD, no DAG, no sieve, no schedule and no kernel.

THE TOLERANCE IS THE RECURRENCE PIN'S, PROPAGATED. test/test_interp.py pins
each [r]^(0) at rel = 1e-12. `J_p` is a signed sum of [p|q] D_q, so an
element-wise relative bound on it would measure the cancellation in the
random density, not the kernel; the bound that follows from the pin is

    |J_p - want_p| <= 1e-12 * SUM_k SUM_q |[p|q] D_q|

which is asserted, not loosened. (Measured on this host, 2026-10-11: over
the whole pin the worst |J_p - want_p| is 9.5e-15 of that scale.)

Nothing compiles and nothing launches on a card: interpreting is not emitting.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Break,
    Call,
    Decl,
    Expr,
    Fn,
    For,
    If,
    Index,
    Kernel,
    Lit,
    Note,
    Raw,
    Space,
    Stmt,
    Var,
    WarpReduce,
    Weighted,
    walk,
)
from felizjueves.kernel import (
    JSH,
    KET,
    SORTED_KETS,
    bra_functions,
    build_pq_kernel,
    build_pq_kernels,
    ket_functions,
    launch_shape,
    pq_spec,
    split,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import SEARCHED_CUTS, TUNED, plan, tuned_plan
from felizjueves.plan.search import load_frontiers, searched_plan
from felizjueves.plan.sieve import NO_SIEVE
from felizjueves.plan.sieve import plan as sieved_dags
from felizjueves.plan.targets import hermite_targets
from felizjueves.recurrence import fundamental_prelude, recurrence_kernels
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, pq, theta_of
from felizjueves.spec import Operator, Spec, Target

SRC = Path(__file__).resolve().parents[1] / "src" / "felizjueves"


def _spec(lp: int, lq: int, levers: tuple[str, ...] = ()) -> Spec:
    return Spec(Operator.COULOMB, 0, (lp, 0, lq, 0), Target.J, levers)


@cache
def _kernel(spec: Spec) -> Kernel:
    return build_pq_kernel(spec)


def _inputs(spec: Spec, n_bra: int, n_ket: int, seed: int) -> dict:
    """Bra and ket indices with T = theta^2 |PQ|^2 well inside boys' domain;
    kets sorted descending by Iq * Dmax, as the kernel requires."""
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


def _run(spec: Spec, args: dict, lanes: int) -> tuple[np.ndarray, Interpreter]:
    grid, block = launch_shape(args["n_bra"], lanes)
    it = Interpreter(_kernel(spec), launch=Launch(grid, block, lanes), boys=boys)
    return it.run(**args)["J"], it


def _reference(spec: Spec, args: dict, *, skip_screened: bool = True):
    """J_p = SUM_k [bound_k >= tau] SUM_q (-1)^|q| [p+q]^(0) D_q, and the scale
    SUM_k SUM_q |[p|q] D_q|. Every ket is tested on its own (a continue), so
    this does not depend on the sort."""
    lp, lq = split(spec)
    ps, qs = bra_functions(spec), ket_functions(spec)
    want = np.zeros((args["n_bra"], len(ps)))
    scale = np.zeros_like(want)
    for b in range(args["n_bra"]):
        for k in range(args["n_ket"]):
            bound = args["Ip"][b] * (args["Iq"][k] * args["Dmax"][k])
            if skip_screened and bound < args["tau"]:
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


def _assert_matches(got: np.ndarray, want: np.ndarray, scale: np.ndarray) -> None:
    assert got.shape == want.shape
    assert np.all(np.abs(got - want) <= 1e-12 * scale), (
        np.max(np.abs(got - want) / scale))


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------


def test_the_reference_contraction_is_reference_pq() -> None:
    """The helper's (-1)^|q| [p+q]^(0) is reference.pq.pq, exactly."""
    PQ, th = (0.3, -0.2, 0.4), 0.8
    for p in hermite_targets(3):
        for q in hermite_targets(2):
            r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
            assert (-1.0) ** sum(q) * one_centre(r, 0, PQ, th) == pq(p, q, PQ, th)


@pytest.mark.parametrize("L", range(0, 17))
def test_j_p_matches_the_reference_for_every_schedulable_class(L: int) -> None:
    splits = {split(pq_spec(L)), (L, 0), (0, L)}
    for lp, lq in sorted(splits):
        spec = _spec(lp, lq)
        args = _inputs(spec, n_bra=2, n_ket=3, seed=L)
        got, _ = _run(spec, args, lanes=2)
        _assert_matches(got, *_reference(spec, args))


@pytest.mark.parametrize("L", [0, 5, 12])
def test_the_unsieved_dag_gives_the_same_j_p(L: int) -> None:
    lp, lq = split(pq_spec(L))
    spec = _spec(lp, lq, (NO_SIEVE,))
    args = _inputs(spec, n_bra=2, n_ket=3, seed=100 + L)
    got, _ = _run(spec, args, lanes=2)
    _assert_matches(got, *_reference(spec, args))


def test_the_split_is_the_sum_of_each_side() -> None:
    spec = Spec(Operator.COULOMB, 0, (2, 1, 1, 1), Target.J)
    assert split(spec) == (3, 2)
    args = _inputs(spec, n_bra=2, n_ket=3, seed=7)
    got, _ = _run(spec, args, lanes=2)
    _assert_matches(got, *_reference(spec, args))


@pytest.mark.parametrize("lanes", [1, 2, 4, 32])
def test_j_p_does_not_depend_on_the_warp_width(lanes: int) -> None:
    spec = pq_spec(3)
    args = _inputs(spec, n_bra=3, n_ket=40, seed=3)
    got, _ = _run(spec, args, lanes)
    _assert_matches(got, *_reference(spec, args))


# ---------------------------------------------------------------------------
# The screening break, against the real kernel
# ---------------------------------------------------------------------------


def _screened_args(spec: Spec, lanes: int) -> tuple[dict, list[int]]:
    args = _inputs(spec, n_bra=3, n_ket=24, seed=25)
    bounds = [args["Ip"][b] * args["Iq"] * args["Dmax"] for b in range(3)]
    args["tau"] = float(np.median(bounds[1]))
    above = [int((bb >= args["tau"]).sum()) for bb in bounds]
    assert all(lanes < a < args["n_ket"] - lanes for a in above)
    return args, above


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_screen_is_a_break_counted_in_iterations(lanes: int) -> None:
    spec = pq_spec(4)
    args, above = _screened_args(spec, lanes)
    got, it = _run(spec, args, lanes)
    _assert_matches(got, *_reference(spec, args))
    n = args["n_ket"]
    # A continue would enter the body once per ket per bra: n_bra * n_ket.
    # The break stops each lane at its first ket under tau.
    want = sum(len(range(y, a, lanes)) + any(j >= a for j in range(y, n, lanes))
               for a in above for y in range(lanes))
    assert it.iterations[KET] == want < args["n_bra"] * n


def test_unsorted_kets_make_the_break_a_wrong_answer() -> None:
    """The precondition is load-bearing: break on unsorted kets drops work."""
    spec = pq_spec(4)
    args, _ = _screened_args(spec, lanes=1)
    rev = {k: (v[::-1].copy() if k in ("Q", "eta", "Iq", "Dmax", "D") else v)
           for k, v in args.items()}
    got, it = _run(spec, rev, lanes=1)
    want, scale = _reference(spec, rev)
    assert not np.all(np.abs(got - want) <= 1e-12 * scale)
    assert it.iterations[KET] == args["n_bra"]  # each bra stops at ket 0


# ---------------------------------------------------------------------------
# The contract, as IR nodes
# ---------------------------------------------------------------------------


def _loops(k: Kernel) -> list[For]:
    return [s for s in walk(k.body) if isinstance(s, For)]


def test_the_contract_is_ir_nodes() -> None:
    k = build_pq_kernel(6)
    n_p = len(bra_functions(pq_spec(6)))
    stmts = list(walk(k.body))

    # One warp per bra index: bra = THREAD_X + BLOCK_X * BLOCK_DIM_X.
    (bra,) = [s for s in stmts if isinstance(s, Decl) and s.name == "bra"]
    assert bra.init == Call(Fn.ADD, (Call(Fn.THREAD_X), Call(
        Fn.MUL, (Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))))
    assert launch_shape(5, 32) == (5, (1, 32))

    # Kets strip-mined over THREAD_Y, stride BLOCK_DIM_Y: the only loop.
    (loop,) = _loops(k)
    assert (loop.var, loop.start, loop.step) == (
        KET, Call(Fn.THREAD_Y), Call(Fn.BLOCK_DIM_Y))

    # The screen is an If holding a Break, directly in the ket loop.
    screens = [s for s in loop.body if isinstance(s, If) and s.then == (Break(),)]
    assert len(screens) == 1 and screens[0].cond.fn is Fn.LT

    # Shared accumulate, one slot per lane, then one WarpReduce per p.
    (jsh,) = [s for s in stmts if isinstance(s, Decl) and s.name == JSH]
    assert jsh.space is Space.SHARED
    assert jsh.shape == (Lit(n_p), Call(Fn.BLOCK_DIM_Y))
    reduces = [s for s in stmts if isinstance(s, WarpReduce)]
    assert len(reduces) == n_p
    assert all(isinstance(r.value, Index) and r.value.array == JSH for r in reduces)

    # The global write is a plain Assign by lane 0, with its Note; no atomics.
    assert not any(isinstance(s, (AtomicAdd, Raw)) for s in stmts)
    writes = [s for s in stmts if isinstance(s, Assign)
              and isinstance(s.dst, Index) and s.dst.array == "J"]
    assert len(writes) == n_p
    (lane0,) = [s for s in stmts if isinstance(s, If)
                and s.cond == Call(Fn.LT, (Call(Fn.THREAD_Y), Lit(1)))]
    assert set(lane0.then) == set(writes)
    notes = [s.prov for s in stmts if isinstance(s, Note)]
    assert any("NOT atomic" in n and "Alg. 1" in n and "§5" in n for n in notes)


def test_the_sort_precondition_is_in_the_kernel() -> None:
    k = build_pq_kernel(4)
    pos = {id(s): i for i, s in enumerate(walk(k.body))}
    (note,) = [s for s in walk(k.body) if isinstance(s, Note) and s.prov == SORTED_KETS]
    (loop,) = _loops(k)
    assert pos[id(note)] < pos[id(loop)]
    assert "break" in SORTED_KETS and "wrong answer" in SORTED_KETS


@pytest.mark.parametrize("L", [4, 9, 16])
def test_ab_cd_is_never_materialised(L: int) -> None:
    """The only array is the shared J_p accumulator; per-quartet values are
    the recurrence's scalars, one per DAG node, and nothing of rank 4."""
    spec = pq_spec(L)
    k = build_pq_kernel(spec)
    decls = [s for s in walk(k.body) if isinstance(s, Decl)]
    arrays = [d for d in decls if d.shape]
    assert [d.name for d in arrays] == [JSH]
    assert arrays[0].shape[0] == Lit(len(bra_functions(spec)))
    (dag,) = plan(spec).dags
    (loop,) = _loops(k)
    in_loop = [s for s in walk(loop.body) if isinstance(s, Decl)]
    assert all(not d.shape for d in in_loop)
    # The DAG's nodes, plus a fixed handful: bound, PQ_x/y/z, eta, theta,
    # t2, T, s (present for L >= 1), and one Boys weight W{m} per fundamental [0]^(m).
    n_fund = sum(1 for (r, _) in dag.nodes() if not any(r))
    assert len(in_loop) == len(dag.nodes()) + 9 + n_fund


def test_the_kernel_is_deterministic_and_single_batch() -> None:
    assert build_pq_kernel(5) == build_pq_kernel(5)
    assert build_pq_kernel(5) == build_pq_kernel(pq_spec(5))
    assert hash(build_pq_kernel(2)) == hash(build_pq_kernel(2))
    with pytest.raises(ValueError, match="atomics"):
        build_pq_kernel(Spec(Operator.COULOMB, 0, (4, 0, 3, 0), Target.J, (), 3))
    with pytest.raises(ValueError):
        pq_spec(-1)


def test_the_kernel_module_is_vendor_neutral() -> None:
    """No device facts, no reference/, no interpreter: it only builds IR."""
    tree = ast.parse((SRC / "kernel.py").read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    ours = {m for m in mods if m and m.startswith("felizjueves")}
    # #344: `j_levers` holds the J kernel-body levers' behaviour (`fused`),
    # reached by one hook in `_pq_kernel`; it is IR in, IR out, as here.
    assert ours == {"felizjueves.ir", "felizjueves.j_levers",
                    "felizjueves.passes",
                    "felizjueves.plan.dag", "felizjueves.plan.pipeline",
                    "felizjueves.plan.schedule", "felizjueves.plan.targets",
                    "felizjueves.recurrence", "felizjueves.spec"}


# ---------------------------------------------------------------------------
# No dead declarations (#62)
# ---------------------------------------------------------------------------


def _names_in(e: Expr | None) -> Iterator[str]:
    """Every name an expression mentions: Vars, indexed arrays, the scalar a
    non-unit Factor binds."""
    if e is None:
        return
    if isinstance(e, Var):
        yield e.name
    elif isinstance(e, Index):
        yield e.array
        for i in e.at:
            yield from _names_in(i)
    elif isinstance(e, Call):
        for a in e.args:
            yield from _names_in(a)
    elif isinstance(e, Weighted):
        for t in e.terms:
            if t.g.name_in_kernel is not None:
                yield t.g.name_in_kernel
            yield from _names_in(t.src)


def _referenced(body: tuple[Stmt, ...]) -> set[str]:
    """Every name some statement of `body` mentions, declarations aside."""
    names: set[str] = set()
    for s in walk(body):
        if isinstance(s, Decl):
            exprs: tuple[Expr | None, ...] = (*s.shape, s.init)
        elif isinstance(s, (Assign, WarpReduce, AtomicAdd)):
            exprs = (s.dst, s.value)
        elif isinstance(s, For):
            exprs = (s.start, s.stop, s.step)
        elif isinstance(s, If):
            exprs = (s.cond,)
        else:
            exprs = ()
        for e in exprs:
            names.update(_names_in(e))
    return names


def _dead_decls(k: Kernel) -> list[str]:
    declared = [s.name for s in walk(k.body) if isinstance(s, Decl)]
    declared += [s.var for s in walk(k.body) if isinstance(s, For)]
    used = _referenced(k.body)
    return [n for n in declared if n not in used]


def test_s_is_declared_only_when_a_weight_reads_it() -> None:
    """`s = -2 theta^2` is the ratio W{m+1} = W{m} s: at L = 0 only W0 exists,
    and declaring `s` anyway is nvcc's warning #177-D (#62)."""
    def names(ms: list[int]) -> list[str]:
        stmts, _ = fundamental_prelude(ms)
        return [s.name for s in stmts if isinstance(s, Decl)]

    assert names([0]) == ["t2", "T", "W0"]
    assert names([0, 1]) == ["t2", "T", "s", "W0", "W1"]


@pytest.mark.parametrize("L", range(17))
def test_no_statement_declares_a_name_nothing_references(L: int) -> None:
    """The property #177-D is one instance of: every Decl (and every For
    variable) of the generated [p|q] kernel is mentioned by some statement."""
    assert _dead_decls(build_pq_kernel(L)) == []


@pytest.mark.parametrize("L", [0, 1, 4])
def test_the_recurrence_only_kernel_has_no_dead_declaration(L: int) -> None:
    """`build_recurrence_kernel` shares the prelude, and so the guard."""
    for _, k in recurrence_kernels(pq_spec(L)):
        assert _dead_decls(k) == []


# ---------------------------------------------------------------------------
# The `tuned` lever (#161): the searched board's recorded plan, opt-in
# ---------------------------------------------------------------------------
#
# Every assertion RECOMPUTES from searched_cuts.json rather than pinning a
# count, so a later re-selection of the file (#161's hazard) moves the tuned
# kernels and these tests follow it.


def _recorded_cells() -> list[tuple[int, int]]:
    return sorted(load_frontiers(SEARCHED_CUTS))


def _tuned(L: int, n: int) -> Spec:
    return pq_spec(L, (TUNED,), n)


def _same(a: tuple[BatchDag, ...], b: tuple[BatchDag, ...]) -> bool:
    """BatchDag has no __eq__ (eq=False): batch, targets and parents, in
    order -- the order `schedule` reads them in included."""
    def key(d: BatchDag) -> tuple:
        return d.batch, d.targets, tuple(d.parents.items())
    return [key(d) for d in a] == [key(d) for d in b]


def test_tuned_plan_cost_is_the_searched_boards_recorded_counts() -> None:
    """The plan the board replayed, not a third thing: every recorded cell's
    `PlanCost` under `("tuned",)` is its selected frontier point's counts."""
    frontiers = load_frontiers(SEARCHED_CUTS)
    assert frontiers
    for (L, n), (front, k) in sorted(frontiers.items()):
        got = plan(_tuned(L, n))
        want = front[k]
        assert (got.cost.max_intermediates, got.cost.flops,
                got.cost.max_targets) == (
            want.max_intermediates, want.flops, want.max_targets), (L, n)
        # and the plan is the recorded cut and orders, batch for batch
        assert _same(got.dags, searched_plan(L, n, want.cuts, want.orders)), (
            L, n)


def test_the_tuned_lever_moves_the_kernel_somewhere() -> None:
    """Otherwise the lever is a no-op, and that is the finding."""
    for L, n in _recorded_cells():
        if build_pq_kernels(_tuned(L, n)) != build_pq_kernels(pq_spec(L, (), n)):
            return
    pytest.fail("no recorded cell's tuned kernel differs from literal")


def test_no_lever_is_still_the_literal_plan() -> None:
    """The default does not move: `()` is `sieve.plan` (the goldens and
    kernels.json's source_sha256 pin the emitted text itself)."""
    for L, n in _recorded_cells()[:4]:
        spec = pq_spec(L, (), n)
        assert _same(plan(spec).dags, sieved_dags(spec))


def _run_batches(spec: Spec, args: dict, lanes: int) -> np.ndarray:
    """Every batch kernel in Plan.dags order, J threaded through from zero."""
    grid, block = launch_shape(args["n_bra"], lanes)
    J = np.zeros_like(args["J"])
    for k in build_pq_kernels(spec):
        it = Interpreter(k, launch=Launch(grid, block, lanes), boys=boys)
        J = it.run(**{**args, "J": J})["J"]
    return J


@pytest.mark.parametrize("n", [1, 3, 6, 9])
def test_the_tuned_kernel_interprets_against_the_reference(n: int) -> None:
    L = _recorded_cells()[0][0]
    spec = _tuned(L, n)
    args = _inputs(spec, n_bra=2, n_ket=3, seed=161 + n)
    _assert_matches(_run_batches(spec, args, lanes=2), *_reference(spec, args))


def test_a_cell_with_no_recorded_choice_is_refused() -> None:
    """Never a silent fall back to literal: a tuned build is always traceable
    to a recorded search."""
    recorded = set(_recorded_cells())
    for L, n in ((6, 1), (16, 12)):
        assert (L, n) not in recorded
        with pytest.raises(ValueError, match="no recorded choice"):
            build_pq_kernels(_tuned(L, n))
    with pytest.raises(ValueError, match="no recorded choice"):
        build_pq_kernel(_tuned(6, 1))


def test_tuned_is_its_own_arm() -> None:
    """Neither implies nor admits `no-sieve`; any other lever is refused."""
    L, n = _recorded_cells()[0]
    with pytest.raises(ValueError, match="two arms at once"):
        plan(pq_spec(L, (NO_SIEVE, TUNED), n))
    with pytest.raises(ValueError, match="no lever but"):
        plan(pq_spec(L, ("recycle", TUNED), n))
    with pytest.raises(ValueError, match="needs the 'tuned' lever"):
        tuned_plan(pq_spec(L, (), n))
    spec = pq_spec(L, (NO_SIEVE,), n)
    assert _same(plan(spec).dags, sieved_dags(spec))
