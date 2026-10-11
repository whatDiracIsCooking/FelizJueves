"""Lowering: liveness, opt-in slot recycling, and KernelCost (lower.py).

THE KERNEL UNDER TEST is built here, not imported: `_recurrence_kernel` turns
one scheduled DAG (plan/schedule.py, MILD and sieved, n_batches = 1) into the
straight-line recurrence -- a `Decl` per node in schedule order, fundamentals
read from `F[m]`, each target stored to `out[k]` right after it is formed. It
is a test fixture, not the [p|q] kernel (Refs #25).

SEMANTICS ARE CHECKED TWICE. On the fixture, by `_evaluate`, a test-local
evaluator over the straight-line subset (Decl, Assign, Weighted, Index, Var,
Lit) sharing no code with lower.py; and, at the bottom (#55), on the [p|q]
kernel the generator ships, by interp.py: `interpret(recycle(k)) ==
interpret(k)` for every batch kernel of a cell. Both must agree BITWISE,
since the arithmetic and its order are unchanged.

TIMING: the fixture tests (L = 7..16 on both DAGs) run in ~10 s on the dev
box. The #55 interpreted cells take ~1-4 s each; those at L >= 13 are marked
`slow`, so the pre-push gate runs L <= 10 and the rest run by hand.
"""

from __future__ import annotations

import ast
import random
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from felizjueves import measured
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Factor,
    Fn,
    For,
    If,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Space,
    Stmt,
    Term,
    Var,
    Weighted,
    walk,
)
from felizjueves.kernel import (
    bra_functions,
    build_pq_kernel,
    build_pq_kernels,
    ket_functions,
    launch_shape,
    pq_spec,
)
from felizjueves.lower import (
    LEVERS,
    RECYCLE,
    KernelCost,
    kernel_cost,
    liveness,
    lower,
    recycle,
    spilled,
)
from felizjueves.plan.dag import BatchDag, Node, build, is_fundamental
from felizjueves.plan.schedule import schedule
from felizjueves.plan.sieve import plan
from felizjueves.reference.boys import boys
from felizjueves.spec import Operator, Spec, Target

GRID_L = range(7, 17)
WHICH = ("mild", "sieved")
SRC = Path(__file__).resolve().parents[1] / "src" / "felizjueves"


# ---------------------------------------------------------------------------
# The fixture kernel: a scheduled DAG as straight-line IR
# ---------------------------------------------------------------------------


def _spec(L: int) -> Spec:
    return Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, (), 1)


@cache
def _dag(L: int, which: str) -> BatchDag:
    (dag,) = build(_spec(L)) if which == "mild" else plan(_spec(L))
    return dag


def _name(node: Node) -> str:
    (rx, ry, rz), m = node
    return f"r{rx}_{ry}_{rz}_m{m}"


def _weighted(dag: BatchDag, node: Node) -> Weighted:
    """[r]^(m) = PQ_d [r-1d]^(m+1) + (r_d - 1) [r-2d]^(m+1), reduction form."""
    ps = dag.parents[node]
    (r, _), (r1, _) = node, ps[0]
    d = next(k for k in range(3) if r[k] != r1[k])
    terms = [Term(1, Factor.pq(d), Var(_name(ps[0])))]
    if len(ps) == 2:
        terms.append(Term(r[d] - 1, Factor.ONE, Var(_name(ps[1]))))
    return Weighted(tuple(terms))


def _recurrence_kernel(dag: BatchDag) -> tuple[Kernel, tuple[Node, ...]]:
    order = schedule(dag)
    targets = sorted(dag.targets)
    slot = {t: k for k, t in enumerate(targets)}
    body: list[Stmt] = [Note("[JEngine, v2, eqs. 10-12], reduction form")]
    for node in order:
        if is_fundamental(node):
            init = Index("F", (Lit(node[1]),))
        else:
            init = _weighted(dag, node)
        body.append(Decl(_name(node), Scalar.F64, init=init))
        if node in slot:
            body.append(Assign(Index("out", (Lit(slot[node]),)), Var(_name(node))))
    params = (
        Param("PQ_x", Scalar.F64),
        Param("PQ_y", Scalar.F64),
        Param("PQ_z", Scalar.F64),
        Param("F", Scalar.F64, array=True),
        Param("out", Scalar.F64, array=True, intent=Intent.OUT),
    )
    return Kernel("recurrence", params, tuple(body)), order


@cache
def _kernel(L: int, which: str) -> tuple[Kernel, tuple[Node, ...]]:
    return _recurrence_kernel(_dag(L, which))


grid = pytest.mark.parametrize(
    ("L", "which"), [(L, w) for L in GRID_L for w in WHICH]
)


# ---------------------------------------------------------------------------
# A test-local evaluator: straight-line subset only, independent of lower.py
# ---------------------------------------------------------------------------


def _evaluate(kernel: Kernel, scalars: dict, arrays: dict) -> dict[int, float]:
    env: dict[str, float] = dict(scalars)
    mem: dict[str, dict[int, float]] = {k: dict(v) for k, v in arrays.items()}

    def e(x):
        if isinstance(x, Lit):
            return x.value
        if isinstance(x, Var):
            return env[x.name]
        if isinstance(x, Index):
            (i,) = x.at
            return mem[x.array][e(i)]
        if isinstance(x, Weighted):
            acc = 0.0
            for t in x.terms:
                g = 1.0 if t.g is Factor.ONE else env[t.g.name_in_kernel]
                acc += t.c * g * e(t.src)
            return acc
        raise AssertionError(f"not in the straight-line subset: {x!r}")

    def store(dst, value):
        if isinstance(dst, Var):
            env[dst.name] = value
        else:
            (i,) = dst.at
            mem.setdefault(dst.array, {})[e(i)] = value

    for s in kernel.body:
        if isinstance(s, Note):
            continue
        if isinstance(s, Decl):
            if s.shape:
                mem[s.name] = {}
            elif s.init is not None:
                env[s.name] = e(s.init)
        elif isinstance(s, Assign):
            store(s.dst, e(s.value))
        else:
            raise AssertionError(f"not in the straight-line subset: {s!r}")
    return mem["out"]


def _inputs(L: int, seed: int) -> tuple[dict, dict]:
    rng = random.Random(seed)
    scalars = {f"PQ_{a}": rng.uniform(-2, 2) for a in "xyz"}
    arrays = {"F": {m: rng.uniform(-1, 1) for m in range(L + 1)}, "out": {}}
    return scalars, arrays


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


def _independent_peak(dag: BatchDag, order: tuple[Node, ...]) -> int:
    """A straight sweep over the SCHEDULE, never over the IR: a node is live
    from its position to its last consumer's (closed). Targets are stored
    the statement after they are formed and have no consumer, so their one
    extra IR statement can only lower the count, never raise it."""
    pos = {n: i for i, n in enumerate(order)}
    last = dict(pos)
    for child, ps in dag.parents.items():
        for p in ps:
            last[p] = max(last[p], pos[child])
    best = 0
    for i in range(len(order)):
        best = max(best, sum(1 for n in order if pos[n] <= i <= last[n]))
    return best


@grid
def test_peak_live_matches_an_independent_liveness_count(L: int, which: str) -> None:
    kernel, order = _kernel(L, which)
    want = _independent_peak(_dag(L, which), order)
    assert liveness(kernel).peak == want
    assert kernel_cost(kernel, registers=10**6).peak_live == want


def test_liveness_holds_a_value_across_a_loop_that_reads_it() -> None:
    # A value read inside a loop it was declared outside of is live to the
    # loop's end: the next iteration reads it again.
    k = Kernel(
        "k",
        (Param("out", Scalar.F64, array=True, intent=Intent.OUT),),
        (
            Decl("a", Scalar.F64, init=Lit(1.0)),  # 0
            Decl("n", Scalar.I32, init=Lit(4)),  # 1
            For(  # 2
                "i",
                Lit(0),
                Var("n"),
                Lit(1),
                (
                    Assign(Index("out", (Var("i"),)), Var("a")),  # 3
                    Decl("t", Scalar.F64, init=Var("a")),  # 4
                    Note("tail"),  # 5
                ),
            ),
            Decl("b", Scalar.F64, init=Lit(2.0)),  # 6
            If(Call(Fn.LT, (Var("b"), Lit(3.0))), (Note("x"),)),  # 7, 8
        ),
    )
    live = liveness(k)
    assert len(live.stmts) == sum(1 for _ in walk(k.body))
    assert (live.start["a"], live.end["a"]) == (0, 5)  # to the loop's last stmt
    assert (live.start["n"], live.end["n"]) == (1, 5)  # `stop`, re-evaluated
    assert (live.start["t"], live.end["t"]) == (4, 4)  # declared inside: no extension
    assert (live.start["b"], live.end["b"]) == (6, 7)
    assert live.reads["a"] == 2 and live.reads["t"] == 0
    assert live.peak == 3 and live.live_at(4) == {"a", "n", "t"}


def test_liveness_refuses_what_it_cannot_key() -> None:
    twice = (Decl("a", Scalar.F64, init=Lit(1.0)), Decl("a", Scalar.F64))
    with pytest.raises(ValueError, match="declared twice"):
        liveness(Kernel("k", (), twice))
    early = (Assign(Var("a"), Lit(1.0)), Decl("a", Scalar.F64))
    with pytest.raises(ValueError, match="before it is declared"):
        liveness(Kernel("k", (), early))


# ---------------------------------------------------------------------------
# Recycling
# ---------------------------------------------------------------------------


@grid
def test_recycling_preserves_semantics(L: int, which: str) -> None:
    kernel, _ = _kernel(L, which)
    recycled = recycle(kernel)
    assert recycled != kernel
    for seed in range(3):
        scalars, arrays = _inputs(L, seed)
        named = _evaluate(kernel, scalars, arrays)
        assert len(named) == len(_dag(L, which).targets)
        assert _evaluate(recycled, scalars, arrays) == named  # bitwise


def _slot_refs(x, array: str) -> set[int]:
    if isinstance(x, Index):
        here = {x.at[0].value} if x.array == array else set()
        return here.union(*(_slot_refs(i, array) for i in x.at))
    if isinstance(x, Weighted):
        return set().union(*(_slot_refs(t.src, array) for t in x.terms))
    if isinstance(x, Call):
        return set().union(*(_slot_refs(a, array) for a in x.args))
    return set()


@grid
def test_a_recycled_slot_never_aliases_an_input_of_its_own_writer(
    L: int, which: str
) -> None:
    kernel, _ = _kernel(L, which)
    recycled = recycle(kernel)
    head = recycled.body[0]
    assert isinstance(head, Decl) and head.name == "slot" and head.shape
    reused = 0
    seen: set[int] = set()
    for s in recycled.body:
        if isinstance(s, Assign) and isinstance(s.dst, Index) and s.dst.array == "slot":
            k = s.dst.at[0].value
            assert k not in _slot_refs(s.value, "slot"), s
            reused += k in seen
            seen.add(k)
    assert reused > 0  # the check above is not vacuous: slots really are reused


@grid
def test_recycling_uses_exactly_peak_live_slots(L: int, which: str) -> None:
    # Values are an interval graph, so a greedy colouring is optimal.
    kernel, _ = _kernel(L, which)
    (n_slots,) = recycle(kernel).body[0].shape
    assert n_slots == Lit(liveness(kernel).peak)


def test_recycling_is_off_by_default() -> None:
    kernel, _ = _kernel(7, "mild")
    assert lower(kernel) is kernel
    assert lower(kernel, recycle_slots=False) is kernel
    assert lower(kernel, recycle_slots=True) == recycle(kernel)


def test_recycling_leaves_what_it_must_alone() -> None:
    # A Factor's binding stays named, an I32 stays named, a SHARED or array
    # Decl is not a value; and the slot array dodges a name already taken.
    k = Kernel(
        "k",
        (
            Param("slot", Scalar.F64),
            Param("out", Scalar.F64, array=True, intent=Intent.OUT),
        ),
        (
            Decl("PQ_x", Scalar.F64, init=Var("slot")),
            Decl("i", Scalar.I32, init=Lit(0)),
            Decl("sh", Scalar.F64, (Lit(4),), Space.SHARED),
            Decl("a", Scalar.F64, init=Weighted((Term(1, Factor.PQ_X, Var("slot")),))),
            Assign(Index("out", (Var("i"),)), Var("a")),
        ),
    )
    r = recycle(k)
    assert r.body[0] == Decl("_slot", Scalar.F64, (Lit(1),))
    assert r.body[1:4] == k.body[:3]
    assert r.body[4] == Assign(Index("_slot", (Lit(0),)), k.body[3].init)
    assert r.body[5] == Assign(Index("out", (Var("i"),)), Index("_slot", (Lit(0),)))
    nothing = Kernel("k", (), (Decl("i", Scalar.I32, init=Lit(0)),))
    assert recycle(nothing) is nothing


# ---------------------------------------------------------------------------
# KernelCost
# ---------------------------------------------------------------------------


def test_kernel_cost_states_the_plan_cost_separation() -> None:
    doc = KernelCost.__doc__
    assert "PlanCost" in doc and "never merge" in doc
    assert "Nothing in `KernelCost` may feed a literature floor" in doc


def test_nothing_on_the_literature_side_imports_lower() -> None:
    # The direction this split exists to prevent, as a structural check.
    for pkg in ("plan", "literature"):
        for path in sorted((SRC / pkg).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                assert not any(n.split(".")[-1] == "lower" for n in names), path


def test_the_register_budget_has_no_default() -> None:
    kernel, _ = _kernel(7, "mild")
    with pytest.raises(TypeError):
        kernel_cost(kernel)  # type: ignore[call-arg]
    for bad in (-1, 1.5, None):
        with pytest.raises(ValueError):
            kernel_cost(kernel, registers=bad)  # type: ignore[arg-type]
    assert kernel_cost(kernel, registers=3).registers == 3


@grid
def test_traffic_brackets(L: int, which: str) -> None:
    kernel, _ = _kernel(L, which)
    live = liveness(kernel)
    peak = live.peak
    everything = sum(1 + n for n in live.reads.values())
    assert kernel_cost(kernel, registers=peak).traffic == 0
    assert kernel_cost(kernel, registers=0).traffic == everything
    t = [kernel_cost(kernel, registers=b).traffic for b in (0, peak // 4, peak // 2)]
    assert t == sorted(t, reverse=True) and t[-1] > 0
    # Whatever is left unspilled fits the budget at every position.
    budget = peak // 2
    out = spilled(live, budget)
    for p in range(len(live.stmts)):
        assert len(live.live_at(p) - out) <= budget


@grid
def test_n_instructions_counts_the_computing_statements(L: int, which: str) -> None:
    kernel, order = _kernel(L, which)
    n = kernel_cost(kernel, registers=0).n_instructions
    assert n == len(order) + len(_dag(L, which).targets)  # a Decl each, a store each


# ---------------------------------------------------------------------------
# #55: `recycle` as a Spec lever on the real [p|q] kernel
# ---------------------------------------------------------------------------
#
# The interpreter exists now (#23), so the issue's property is asserted on
# the kernel the generator ships, not only on the straight-line fixture above:
# `interpret(recycle(k)) == interpret(k)`, BITWISE, for every batch kernel of
# a cell -- the arithmetic and its order are unchanged, only where each value
# is stored. The interpreter refuses a read of a local element nothing wrote,
# so a slot read before its writer is an error here, not a wrong number.

LEVER_CELLS = [
    pytest.param(s, id=s.key(),
                 marks=[pytest.mark.slow] if s.total_am >= 13 else [])
    for s in (*(pq_spec(L) for L in (0, 1, 4)), *measured.SUBGRID)]
GOLDEN = Path(__file__).resolve().parent / "golden"
EMITTERS = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}


def _pq_inputs(spec: Spec, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_bra, n_ket = 2, 3
    n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
    Iq, Dmax = rng.uniform(0.2, 1.0, n_ket), rng.uniform(0.5, 1.0, n_ket)
    srt = np.argsort(-(Iq * Dmax), kind="stable")
    return dict(
        n_bra=n_bra, n_ket=n_ket, tau=0.0,
        P=rng.uniform(-0.5, 0.5, (n_bra, 3)), zeta=rng.uniform(0.6, 1.4, n_bra),
        Ip=rng.uniform(0.5, 1.0, n_bra),
        Q=rng.uniform(-0.5, 0.5, (n_ket, 3)), eta=rng.uniform(0.6, 1.4, n_ket),
        Iq=Iq[srt].copy(), Dmax=Dmax[srt].copy(),
        D=rng.uniform(-1.0, 1.0, (n_ket, n_q)), J=np.zeros((n_bra, n_p)))


def _interpret(k: Kernel, args: dict) -> np.ndarray:
    grid_, block = launch_shape(args["n_bra"], 2)
    it = Interpreter(k, launch=Launch(grid_, block, 2), boys=boys)
    return it.run(**args)["J"]


def _with(spec: Spec, *levers: str) -> Spec:
    return replace(spec, levers=tuple(sorted(levers)))


@pytest.mark.parametrize("spec", LEVER_CELLS)
def test_interpret_recycle_k_is_interpret_k_on_the_pq_kernel(spec: Spec) -> None:
    named = build_pq_kernels(spec)
    assert measured.cell_kernels(spec) is named or \
        measured.cell_kernels(spec) == named  # no lever: the builder's own
    recycled = measured.cell_kernels(_with(spec, RECYCLE))
    assert len(recycled) == len(named) == spec.n_batches
    args = _pq_inputs(spec, seed=55 + spec.total_am)
    for k, r in zip(named, recycled, strict=True):
        assert r == recycle(k)  # the lever IS the pass, on the finished kernel
        assert r.name == k.name and r.params == k.params
        assert r != k
        got, want = _interpret(r, args), _interpret(k, args)
        assert np.array_equal(got, want), (spec.key(), k.name)  # bitwise


@pytest.mark.parametrize("spec", LEVER_CELLS)
def test_the_recycled_slot_count_is_the_kernels_peak_live(spec: Spec) -> None:
    # peak over the values recycle takes: fp64, not a Factor's binding (the
    # I32 `bra` and PQ_x/y/z stay named, so `liveness.peak` over-counts)
    keep = {f.name_in_kernel for f in Factor} - {None}
    for k, r in zip(build_pq_kernels(spec),
                    measured.cell_kernels(_with(spec, RECYCLE)), strict=True):
        live = liveness(k)
        f64 = [s.name for s in live.stmts if isinstance(s, Decl) and not s.shape
               and s.space is Space.LOCAL and s.type is Scalar.F64
               and s.name not in keep]
        delta = [0] * (len(live.stmts) + 1)
        for v in f64:
            delta[live.start[v]] += 1
            delta[live.end[v] + 1] -= 1
        peak = run = 0
        for d in delta:
            run += d
            peak = max(peak, run)
        head = r.body[0]
        assert head.name == "slot" and head.space is Space.LOCAL
        assert head.shape == (Lit(peak),)
        assert peak < len(f64) or len(f64) <= 1  # slots really are shared


def test_the_lever_is_named_in_the_key_and_not_shown_to_the_plan_layer() -> None:
    spec = pq_spec(10, (RECYCLE,), n_batches=3)
    assert RECYCLE == "recycle" and LEVERS == {RECYCLE}
    assert spec.key() == "coulomb/d0/(5,0|5,0)/b3/j/recycle"
    assert pq_spec(10, n_batches=3).key().endswith("/b3/j/literal")
    # lowering is after building: the plan layer and the builder refuse the
    # lever (they do not own it), and the driver strips it, builds, lowers
    with pytest.raises(ValueError, match="no lever"):
        plan(spec)
    with pytest.raises(ValueError, match="no lever"):
        build_pq_kernels(spec)
    assert len(measured.cell_kernels(spec)) == 3


def test_the_default_arm_is_untouched() -> None:
    """No lever: the kernel is NOT lowered, and the existing goldens hold."""
    for L in (0, 4, 10):
        assert build_pq_kernel(L) == build_pq_kernel(pq_spec(L, ()))
        assert not any(isinstance(s, Decl) and s.name == "slot"
                       for s in walk(build_pq_kernel(L).body))
    for backend, (emit, arch, ext) in EMITTERS.items():
        assert emit(build_pq_kernel(4), arch) == (
            GOLDEN / backend / f"pq_L4_p2_q2.{ext}").read_text()


# THE LEVER GOLDENS are the recycle arm of the class the literal goldens pin,
# (2,0|2,0) one batch, in their own directory so no literal golden's glob
# sees them. Regenerate (and review the diff) with:
#   PYTHONPATH=src uv run python -c "from pathlib import Path; \
#   from felizjueves.kernel import pq_spec; \
#   from felizjueves.measured import cell_kernels; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   (k,) = cell_kernels(pq_spec(4, ('recycle',))); \
#   Path('test/golden/cuda/recycle/pq_L4_p2_q2.cu').write_text(cuda.emit(k, SM_86)); \
#   Path('test/golden/hip/recycle/pq_L4_p2_q2.hip').write_text(hip.emit(k, GFX1200))"


@pytest.mark.parametrize("backend", sorted(EMITTERS))
def test_the_recycle_golden_is_unchanged(backend: str) -> None:
    emit, arch, ext = EMITTERS[backend]
    (k,) = measured.cell_kernels(pq_spec(4, (RECYCLE,)))
    text = emit(k, arch)
    assert text == (GOLDEN / backend / RECYCLE / f"pq_L4_p2_q2.{ext}").read_text()
    # a statically indexed local array, and no named fp64 temporary left but
    # the Factor bindings (PQ_x, PQ_y, PQ_z), which must stay named
    assert "double slot[" in text
    named = {line.split()[1] for line in text.splitlines()
             if line.strip().startswith("double ") and "=" in line
             and "fj_boys" not in line}
    assert named <= {"PQ_x", "PQ_y", "PQ_z", "term", "sum", "next", "fj_v"}, named
