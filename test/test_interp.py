"""interp.py: a Kernel run in numpy, pinned against reference/.

THE PIN. Every class the plan layer can schedule -- L = 0..16 at n_batches = 1,
plus the table grid's batched counts {3, 6, 9} at L = 7..16, sieved and with
the `no-sieve` lever -- is built as a recurrence-only Kernel by
`felizjueves.recurrence` (MILD/sieve DAG -> schedule -> IR), interpreted, and
compared element by element against `reference.pq.one_centre` at rel = 1e-12.
`reference.pq.pq` is `(-1)^|q|` times that value, so this is the [p|q] pin
for every (p, q) splitting of each target. The reference recurses directly
(x-first, no DAG, no sieve, no schedule); the IR path does all of those.

MEASURED WALL-CLOCK (this dev host's CPU, serial `pytest -n0 --durations=0`,
2026-10-11): the n_batches = 1, L = 0..16 sieved pin -- every schedulable
class, four geometries each -- takes 1.6 s; the no-sieve pin 1.5 s; the
batched grid (L = 7..16 x {3, 6, 9}) 5.1 s; the whole module about 8.5 s.
Nothing compiles and nothing launches on a card.

THE GEOMETRIES ARE THE WELL-CONDITIONED ONES, AND THAT IS A MEASUREMENT, NOT A
CHOICE OF CONVENIENCE. At |PQ| ~ 2.6 (`ILL`), a few targets of L >= 15 are
~1e-5 of their class's largest value, formed by cancellation, and there the
interpreted IR and the reference disagree element-wise by up to 2.2e-10. Both
float paths are equally far from the same recurrence evaluated in exact
rational arithmetic (the interpreter is in fact the closer, 8e-11 against
1.4e-10 at L = 15), so the gap is conditioning of the recurrence, not a defect
of either. `test_the_ill_conditioned_geometry_is_conditioning` pins that claim
against the exact evaluation instead of loosening the pin's tolerance.

Interpreting is not emitting: nothing here says a kernel compiles.
"""

from __future__ import annotations

import ast
import dataclasses
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from felizjueves.interp import InterpError, Interpreter, Launch, interpret
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Break,
    Call,
    Decl,
    Fn,
    For,
    If,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Raw,
    Scalar,
    Space,
    Stmt,
    Var,
    WarpReduce,
    walk,
)
from felizjueves.plan.sieve import NO_SIEVE
from felizjueves.recurrence import recurrence_kernels, target_order
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import fundamental, one_centre
from felizjueves.spec import Operator, Spec, Target

SRC = Path(__file__).resolve().parents[1] / "src" / "felizjueves"

# (PQ, theta): T = theta^2 |PQ|^2 stays well inside boys' measured domain.
GEOMETRIES = [
    ((0.3, -0.7, 1.1), 0.9),
    ((-0.05, 0.6, 0.25), 0.4),
    ((0.5, 0.4, -0.3), 1.0),
    ((0.0, 0.0, 0.0), 1.3),
]
ILL = ((2.0, -1.5, 0.8), 1.1)


def _spec(L: int, n_batches: int = 1, levers: tuple[str, ...] = ()) -> Spec:
    return Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, levers, n_batches)


@cache
def _kernels(L: int, n_batches: int, levers: tuple[str, ...] = ()):
    return recurrence_kernels(_spec(L, n_batches, levers))


def _run(kernel: Kernel, n: int, pq, theta) -> np.ndarray:
    return interpret(kernel, boys=boys, PQ_x=pq[0], PQ_y=pq[1], PQ_z=pq[2],
                     theta=theta, out=np.zeros(n))["out"]


def _pin(L: int, n_batches: int, levers: tuple[str, ...] = ()) -> None:
    seen = []
    for dag, kernel in _kernels(L, n_batches, levers):
        targets = target_order(dag)
        seen.extend(targets)
        for pq, theta in GEOMETRIES:
            got = _run(kernel, len(targets), pq, theta)
            want = [one_centre(r, 0, pq, theta) for r in targets]
            assert got.tolist() == pytest.approx(want, rel=1e-12), (
                L, n_batches, dag.batch, pq, theta)
    # Together the batches cover every target of the class, once each.
    assert len(seen) == len(set(seen)) == (L + 1) * (L + 2) * (L + 3) // 6


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", range(0, 17))
def test_the_interpreted_ir_reproduces_the_reference(L: int) -> None:
    _pin(L, 1)


@pytest.mark.parametrize("L", range(0, 17))
def test_the_unsieved_dag_reproduces_the_reference(L: int) -> None:
    _pin(L, 1, (NO_SIEVE,))


@pytest.mark.parametrize(("L", "n_batches"),
                         [(L, b) for L in range(7, 17) for b in (3, 6, 9)])
def test_every_batch_reproduces_the_reference(L: int, n_batches: int) -> None:
    _pin(L, n_batches)


def test_the_ill_conditioned_geometry_is_conditioning() -> None:
    """At ILL the element-wise gap to reference/ is ~1e-10; this shows why.

    Against the recurrence in exact rationals (from the same float [0]^(m)),
    the interpreter is no worse than the reference: the gap is the problem's,
    not the path's. Normwise (by the class's largest value) both are ~1e-14.
    """
    L = 16
    pq, theta = ILL
    P = [Fraction(c) for c in pq]
    F = [Fraction(fundamental(m, pq, theta)) for m in range(L + 1)]

    @cache
    def exact(r, m):  # the recurrence, x-first, in exact arithmetic
        for d in range(3):
            if r[d]:
                v = P[d] * exact(r[:d] + (r[d] - 1,) + r[d + 1:], m + 1)
                if r[d] >= 2:
                    v += (r[d] - 1) * exact(r[:d] + (r[d] - 2,) + r[d + 1:], m + 1)
                return v
        return F[m]

    ((dag, kernel),) = _kernels(L, 1)
    targets = target_order(dag)
    got = _run(kernel, len(targets), pq, theta)
    ref = np.array([one_centre(r, 0, pq, theta) for r in targets])
    ex = np.array([float(exact(r, 0)) for r in targets])
    scale = np.abs(ex).max()
    assert np.abs(got - ex).max() / scale < 1e-13
    assert np.abs(ref - ex).max() / scale < 1e-13
    worst = lambda a: (np.abs(a - ex) / np.abs(ex)).max()  # noqa: E731
    assert worst(ref) > 1e-11  # the reference is off element-wise too
    assert worst(got) <= worst(ref)


# ---------------------------------------------------------------------------
# The screening break
# ---------------------------------------------------------------------------

def _f64(name: str, intent: Intent = Intent.IN, array: bool = True) -> Param:
    return Param(name, Scalar.F64, array=array, intent=intent)


def _screened(*, use_break: bool) -> Kernel:
    """SUM_q bound[q] over the kets with bound[q] >= tau, strip-mined over y.

    With `use_break` the screen is the IR's `Break`; without, the same answer
    by skipping (what a `continue` would do), which keeps iterating.
    """
    q, bound = Var("q"), Index("bound", (Var("q"),))
    add = Assign(Var("acc"), Call(Fn.ADD, (Var("acc"), bound)))
    under = Call(Fn.LT, (bound, Var("tau")))
    screen = If(under, (Break(),)) if use_break else If(under, (), (add,))
    body = (screen, add) if use_break else (screen,)
    return Kernel(
        "screened",
        (_f64("bound"), _f64("tau", array=False), Param("n", Scalar.I32),
         _f64("out", Intent.OUT)),
        (
            Note("kets arrive sorted descending by bound"),
            Decl("acc", Scalar.F64, init=Lit(0.0)),
            For(q.name, Call(Fn.THREAD_Y), Var("n"), Call(Fn.BLOCK_DIM_Y), body),
            WarpReduce(Var("acc"), Var("acc")),
            Assign(Index("out", (Call(Fn.THREAD_Y),)), Var("acc")),
        ),
    )


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_screening_break_actually_breaks(lanes: int) -> None:
    rng = np.random.default_rng(23)
    bound = np.sort(rng.uniform(0.0, 1.0, 64))[::-1].copy()
    tau = 0.5
    above = int((bound >= tau).sum())
    assert 0 < above < len(bound) - lanes
    args = dict(bound=bound, tau=tau, n=len(bound), out=np.zeros(lanes))
    launch = Launch(block=(1, lanes), lanes=lanes)

    broke, skipped = (Interpreter(_screened(use_break=b), launch=launch)
                      for b in (True, False))
    got, alt = broke.run(**args)["out"], skipped.run(**args)["out"]
    # The same answer -- which is exactly why the answer cannot tell them apart.
    assert got.tolist() == pytest.approx([bound[:above].sum()] * lanes, rel=1e-14)
    assert alt.tolist() == pytest.approx(got.tolist(), rel=1e-14)
    # The iteration count can: each lane stops at its first ket under tau.
    assert skipped.iterations["q"] == len(bound)
    assert broke.iterations["q"] < len(bound)
    assert broke.iterations["q"] == sum(
        len(range(y, above, lanes)) + 1 for y in range(lanes))


# ---------------------------------------------------------------------------
# The warp reduction
# ---------------------------------------------------------------------------

def _per_lane_sum() -> Kernel:
    """Each thread sums its y-strip of `v`, the warp reduces, every lane writes."""
    body = (
        Decl("acc", Scalar.F64, init=Lit(0.0)),
        For("q", Call(Fn.THREAD_Y), Var("n"), Call(Fn.BLOCK_DIM_Y), (
            Assign(Var("acc"), Call(Fn.ADD, (Var("acc"), Index("v", (Var("q"),))))),
        )),
        Decl("s", Scalar.F64),
        WarpReduce(Var("s"), Var("acc")),
        Assign(Index("out", (Call(Fn.THREAD_Y),)), Var("s")),
    )
    return Kernel("lane_sum", (_f64("v"), Param("n", Scalar.I32),
                               _f64("out", Intent.OUT)), body)


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16, 32, 64])
def test_the_warp_reduction_does_not_depend_on_the_lane_count(lanes: int) -> None:
    v = np.arange(1.0, 301.0)  # integers: every summation order is exact
    out = interpret(_per_lane_sum(), launch=Launch(block=(1, lanes), lanes=lanes),
                    v=v, n=len(v), out=np.zeros(lanes))["out"]
    assert out.tolist() == [v.sum()] * lanes

    w = np.random.default_rng(lanes).uniform(-1.0, 1.0, 300)
    out = interpret(_per_lane_sum(), launch=Launch(block=(1, lanes), lanes=lanes),
                    v=w, n=len(w), out=np.zeros(lanes))["out"]
    assert len(set(out.tolist())) == 1  # every lane sees the same sum
    assert out[0] == pytest.approx(w.sum(), rel=1e-13)


def test_a_warp_is_lanes_consecutive_threads() -> None:
    """Two warps in one block: each reduces only its own lanes."""
    v = np.arange(1.0, 129.0)
    out = interpret(_per_lane_sum(), launch=Launch(block=(1, 64), lanes=32),
                    v=v, n=len(v), out=np.zeros(64))["out"]
    strips = [v[y::64].sum() for y in range(64)]
    assert out[:32].tolist() == [sum(strips[:32])] * 32
    assert out[32:].tolist() == [sum(strips[32:])] * 32


def test_a_partial_or_divergent_warp_reduction_raises() -> None:
    v = np.arange(1.0, 11.0)
    with pytest.raises(InterpError, match="partial warp"):
        interpret(_per_lane_sum(), launch=Launch(block=(1, 6), lanes=4),
                  v=v, n=len(v), out=np.zeros(6))
    diverge = Kernel("diverge", (_f64("out", Intent.OUT),), (
        Decl("s", Scalar.F64),
        If(Call(Fn.LT, (Call(Fn.THREAD_Y), Lit(1))),
           (WarpReduce(Var("s"), Lit(1.0)),)),
    ))
    with pytest.raises(InterpError, match="diverged"):
        interpret(diverge, launch=Launch(block=(1, 2), lanes=2), out=np.zeros(1))


# ---------------------------------------------------------------------------
# Notes are inert
# ---------------------------------------------------------------------------

def _strip(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    out = []
    for s in body:
        if isinstance(s, Note):
            continue
        if isinstance(s, For):
            s = dataclasses.replace(s, body=_strip(s.body))
        elif isinstance(s, If):
            s = dataclasses.replace(s, then=_strip(s.then), orelse=_strip(s.orelse))
        out.append(s)
    return tuple(out)


def _without_notes(k: Kernel) -> Kernel:
    stripped = dataclasses.replace(k, body=_strip(k.body))
    assert any(isinstance(s, Note) for s in walk(k.body))
    assert not any(isinstance(s, Note) for s in walk(stripped.body))
    return stripped


def test_notes_are_inert() -> None:
    for L in (0, 5, 12):
        ((dag, kernel),) = _kernels(L, 1)
        n = len(target_order(dag))
        for pq, theta in GEOMETRIES:
            assert np.array_equal(_run(kernel, n, pq, theta),
                                  _run(_without_notes(kernel), n, pq, theta))
    bound = np.linspace(1.0, 0.0, 20)
    args = dict(bound=bound, tau=0.3, n=20, out=np.zeros(4))
    launch = Launch(block=(1, 4), lanes=4)
    k = _screened(use_break=True)
    a, b = Interpreter(k, launch=launch), Interpreter(_without_notes(k), launch=launch)
    assert np.array_equal(a.run(**args)["out"], b.run(**args)["out"])
    assert a.iterations == b.iterations


# ---------------------------------------------------------------------------
# The machine: what it executes, and what it refuses
# ---------------------------------------------------------------------------


def test_the_interpreter_does_not_know_mcmurchie_davidson() -> None:
    """Its only felizjueves import is the IR; Boys is injected (`boys=`)."""
    tree = ast.parse((SRC / "interp.py").read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
             for a in n.names}
    ours = {m for m in mods if m and m.startswith("felizjueves")}
    assert ours == {"felizjueves.ir"}


def test_shared_memory_atomics_and_geometry() -> None:
    """Per-thread slots in SHARED, AtomicAdd to global, the bra index formula."""
    bra = Call(Fn.ADD, (Call(Fn.THREAD_X),
                        Call(Fn.MUL, (Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))))
    k = Kernel("atomics", (Param("hits", Scalar.I32, array=True, intent=Intent.OUT),
                           _f64("out", Intent.OUT)), (
        Decl("sh", Scalar.F64, (Call(Fn.BLOCK_DIM_X),), Space.SHARED),
        Assign(Index("sh", (Call(Fn.THREAD_X),)), Call(Fn.DIV, (Lit(1.0), Lit(4.0)))),
        Assign(Index("out", (bra,)), Index("sh", (Call(Fn.THREAD_X),))),
        AtomicAdd(Index("hits", (Call(Fn.DIV, (bra, Lit(3))),)), Lit(1)),
    ))
    got = interpret(k, launch=Launch(grid=3, block=(4, 1), lanes=4),
                    hits=np.zeros(4, np.int32), out=np.zeros(12))
    assert got["out"].tolist() == [0.25] * 12
    assert got["hits"].tolist() == [3, 3, 3, 3]


def test_the_machine_refuses_what_has_no_meaning() -> None:
    out = _f64("out", Intent.OUT)
    def run(*body, **inputs):
        return interpret(Kernel("k", (out, _f64("a")), tuple(body)),
                         out=np.zeros(2), **inputs)
    a = np.ones(2)
    with pytest.raises(InterpError, match="opaque"):
        run(Raw("__nanosleep(1);"), a=a)
    with pytest.raises(InterpError, match="IN array"):
        run(Assign(Index("a", (Lit(0),)), Lit(1.0)), a=a)
    with pytest.raises(InterpError, match="outside shape"):
        run(Assign(Index("out", (Lit(-1),)), Lit(1.0)), a=a)
    with pytest.raises(InterpError, match="before it was written"):
        run(Decl("x", Scalar.F64), Assign(Index("out", (Lit(0),)), Var("x")), a=a)
    with pytest.raises(InterpError, match="into an I32"):
        run(Decl("i", Scalar.I32, init=Lit(0.5)), a=a)
    with pytest.raises(InterpError, match="ndarray of float64"):
        run(a=np.ones(2, np.float32))
    with pytest.raises(InterpError, match="missing"):
        run()
    # Boys is injected, never defaulted from reference/ (test/test_layering.py).
    with pytest.raises(InterpError, match="no boys= evaluator"):
        run(Assign(Index("out", (Lit(0),)), Call(Fn.BOYS, (Lit(0), Lit(1.0)))), a=a)
    # Integer division truncates toward zero, as C does.
    got = run(Assign(Index("out", (Lit(0),)), Call(Fn.DIV, (Lit(-7), Lit(2)))), a=a)
    assert got["out"][0] == -3.0


# ---------------------------------------------------------------------------
# Inputs against the declared shapes (#59)
# ---------------------------------------------------------------------------

N_P, N_Q = 3, 2


def _shaped() -> Kernel:
    """J[bra, i] = D[bra, 0] * i + D[bra, 1], one thread per bra: the
    `[p|q]` kernel's parameter shapes, (n_bra, n_p) and (n_bra, n_q)."""
    bra, nb = Call(Fn.BLOCK_X), Var("n_bra")
    d = lambda j: Index("D", (bra, Lit(j)))  # noqa: E731
    return Kernel("shaped", (
        Param("n_bra", Scalar.I32),
        Param("D", Scalar.F64, array=True, shape=(nb, Lit(N_Q))),
        Param("J", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nb, Lit(N_P))),
    ), tuple(
        Assign(Index("J", (bra, Lit(i))),
               Call(Fn.ADD, (Call(Fn.MUL, (d(0), Lit(float(i)))), d(1))))
        for i in range(N_P)))


def _run_shaped(n_bra: int, D: np.ndarray, J: np.ndarray) -> np.ndarray:
    return interpret(_shaped(), launch=Launch(grid=n_bra), n_bra=n_bra,
                     D=D, J=J)["J"]


def test_an_input_off_its_declared_extents_raises_naming_it() -> None:
    n, D = 4, np.ones((4, N_Q))
    with pytest.raises(InterpError, match=r"^J: .*\(4, 3\) = 12 .*11 elements"):
        _run_shaped(n, D, np.zeros(n * N_P - 1))           # a short J
    with pytest.raises(InterpError, match=r"^J: .*= 12 .*\(5, 3\) = 15"):
        _run_shaped(n, D, np.zeros((n + 1, N_P)))          # a long J
    with pytest.raises(InterpError, match=r"^D: .*\(4, 2\) = 8 .*\(4, 1\)"):
        _run_shaped(n, np.ones((n, N_Q - 1)), np.zeros((n, N_P)))  # short D


def test_flat_is_the_contract() -> None:
    """A flat J and the same-sized 2-D J both run, agree, and come back in the
    shape they went in; the contract is the size, not the ndarray shape."""
    n = 5
    D = np.random.default_rng(59).uniform(-1, 1, (n, N_Q))
    want = D[:, :1] * np.arange(N_P) + D[:, 1:]
    two_d = _run_shaped(n, D, np.zeros((n, N_P)))
    flat = _run_shaped(n, D.ravel(), np.zeros(n * N_P))
    assert two_d.shape == (n, N_P) and flat.shape == (n * N_P,)
    assert np.array_equal(two_d, want) and np.array_equal(flat, want.ravel())


def test_an_unevaluable_extent_is_skipped_and_recorded() -> None:
    """An extent the I32 scalars cannot evaluate is not a failure."""
    out = _f64("out", Intent.OUT)
    k = Kernel("k", (
        _f64("tau", array=False), Param("n", Scalar.I32),
        Param("a", Scalar.F64, array=True, shape=(Var("tau"),)),
        Param("b", Scalar.F64, array=True,
              shape=(Call(Fn.DIV, (Var("n"), Lit(0))),)),
        out,
    ), (Assign(Index("out", (Lit(0),)), Index("a", (Lit(1),))),))
    it = Interpreter(k)
    got = it.run(tau=0.5, n=3, a=np.arange(4.0), b=np.ones(7), out=np.zeros(1))
    assert got["out"][0] == 1.0
    assert set(it.unchecked) == {"a", "b"}
    assert "tau" in it.unchecked["a"] and "division" in it.unchecked["b"]
