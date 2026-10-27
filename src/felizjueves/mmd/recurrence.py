"""The recurrence half of a [p|q] kernel, built as IR from a scheduled DAG.

This is the bridge from plan/ to ir/__init__.py: it takes one batch's DAG (MILD's, or
the sieve's pruned one) and its schedule (plan/schedule.py, pipeline step 4)
and turns every node into one IR statement, in the schedule's order:

    [0]^(m)   a fundamental     Decl R0_0_0_m = <fundamental(m)>
    [r]^(m)   an intermediate   Decl Rx_y_z_m = Weighted(
                                    1       * PQ_d * [r - 1d]^(m+1)
                                  + (r_d-1) * 1    * [r - 2d]^(m+1) )

which is the reduction form of `[JEngine, v2, eqs. 10-12]`
(docs/implementation/j_engine.md); the second term is absent when r_d = 1.
The direction d is read off the DAG, never re-derived: MILD chose it, or the
sieve re-parented the node, and this module only spells what the plan decided.

THIS MODULE KNOWS McMURCHIE-DAVIDSON; `interp.py` MUST NOT. The interpreter
executes the statements produced here and is pinned against `reference/`,
which recurses directly (no MILD, no DAG, no sieve, no schedule). This module
is the thing under test, so it is the one place the recurrence may live on the
IR side.

TWO LAYERS, so the full [p|q] kernel (Alg. 1, the issue that follows #23) can
reuse the inner one:

- `recurrence_body(dag, order, fundamental)` -- the statements alone. The
  caller decides what a fundamental is (an `Expr` per m) and where the
  targets go; nodes are bound as LOCAL F64 scalars named `node_name(node)`.
- `build_recurrence_kernel(dag)` / `recurrence_kernels(spec)` -- a
  recurrence-only `Kernel`: scalar params `PQ_x PQ_y PQ_z theta`, the
  fundamentals computed in the kernel from them, and every target [r](0)
  written to `out[i]` with i its position in `target_order(dag)`.

THE FUNDAMENTAL, in the convention reference/pq.py pins (j_engine.md, "Two
defects in the source"):

    [0]^(m) = (2/sqrt(pi)) theta (-2 theta^2)^m F_m(theta^2 |PQ|^2)

with `(-2 theta^2)^m` formed as a running product (`Fn` has no power) and
every F_m, m = 0..top, by ONE `Boys` statement into the local array `boys`
(`boys_fill`, #301): one evaluation of the top order and a recurrence give
them all, where a call per order summed a series per order. The pair factors
(pi/zeta)^(3/2) etc. are not here, as they are not in the reference.

fp64 throughout, as an IR fact (ir.Scalar).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from felizjueves.ir import (
    Assign,
    Boys,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Term,
    Var,
    Weighted,
)
from felizjueves.plan.dag import BatchDag, Node, is_fundamental
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import R, hermite_targets
from felizjueves.spec import Spec

OUT = "out"
PQ_PARAMS = tuple(Factor.pq(d).name_in_kernel for d in range(3))
THETA = "theta"
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)

Fundamental = Callable[[int], Expr]


def node_name(node: Node) -> str:
    """The LOCAL scalar that holds [r]^(m): `R{x}_{y}_{z}_{m}`."""
    (x, y, z), m = node
    return f"R{x}_{y}_{z}_{m}"


def target_order(dag: BatchDag) -> tuple[R, ...]:
    """The batch's targets [r](0) in `hermite_targets` order: `out`'s layout."""
    top = max(sum(r) for r, _ in dag.targets)
    return tuple(r for r in hermite_targets(top) if (r, 0) in dag.targets)


def _axis(node: Node, parent: Node) -> int:
    """The direction `node` is reduced along, read off its first parent."""
    (r, _), (r1, _) = node, parent
    return next(k for k, (c, c1) in enumerate(zip(r, r1, strict=True)) if c != c1)


def reduction(dag: BatchDag, node: Node) -> Weighted:
    """`[r]^(m) = PQ_d [r-1d]^(m+1) + (r_d - 1) [r-2d]^(m+1)`, d from the DAG.

    Checks the DAG's parents are exactly that pair (or the single first one
    when r_d = 1), so a malformed DAG raises here instead of building a
    kernel that computes something else.
    """
    (r, m), ps = node, dag.parents[node]
    d = _axis(node, ps[0])
    want = [tuple(c - (k == d) for k, c in enumerate(r))]
    if r[d] >= 2:
        want.append(tuple(c - 2 * (k == d) for k, c in enumerate(r)))
    if list(ps) != [(w, m + 1) for w in want]:
        raise ValueError(f"{node}: parents {ps} are not a reduction along {d}")
    terms = [Term(1, Factor.pq(d), Var(node_name(ps[0])))]
    if r[d] >= 2:
        terms.append(Term(r[d] - 1, Factor.ONE, Var(node_name(ps[1]))))
    return Weighted(tuple(terms))


def recurrence_body(
    dag: BatchDag, order: Order, fundamental: Fundamental
) -> tuple[Stmt, ...]:
    """One `Decl` per node of `dag`, in `order` (a checked schedule of it).

    `fundamental(m)` is the `Expr` that forms [0]^(m); everything else is a
    `reduction`. Each node is a LOCAL F64 scalar named `node_name(node)`.
    """
    body: list[Stmt] = [Note("[JEngine, v2, eqs. 10-12], reduction form")]
    for node in order:
        value = fundamental(node[1]) if is_fundamental(node) else reduction(dag, node)
        body.append(Decl(node_name(node), Scalar.F64, init=value))
    return tuple(body)


def store_targets(body: tuple[Stmt, ...], targets: frozenset[R], L: int,
                  array: str) -> tuple[Stmt, ...]:  # noqa: N803
    """`body` with `array[i] = [r_i]^(0)` right after each target's `Decl`,
    `i` the place of `r` in `hermite_targets(L)`, so no target stays live
    past its own store (#606: the low-stack DF kernels' `R`). Raises when
    `body` declares some target nowhere."""
    at = {node_name((r, 0)): i for i, r in enumerate(hermite_targets(L))
          if r in targets}
    out: list[Stmt] = []
    for st in body:
        out.append(st)
        if isinstance(st, Decl) and st.name in at:
            out.append(Assign(Index(array, (Lit(at.pop(st.name)),)),
                              Var(st.name)))
    if at:
        raise ValueError(f"the body declares no {sorted(at)}")
    return tuple(out)


def _mul(a: Expr, b: Expr) -> Call:
    return Call(Fn.MUL, (a, b))


def _add(a: Expr, b: Expr) -> Call:
    return Call(Fn.ADD, (a, b))


# The local array every `Boys` of a prelude fills: `boys[m] = F_m(T)`.
BOYS_ARRAY = "boys"


def boys_fill(top: int, t: Expr) -> tuple[Stmt, ...]:
    """`boys[m] = F_m(t)` for m = 0..top: the array's `Decl` and the one
    `Boys` that fills it (#301)."""
    return (Decl(BOYS_ARRAY, Scalar.F64, shape=(Lit(top + 1),)),
            Boys(BOYS_ARRAY, top, t))


def boys_read(m: int) -> Index:
    """`F_m(T)`, as a prelude's `boys_fill` left it."""
    return Index(BOYS_ARRAY, (Lit(m),))


def fundamental_prelude(ms: list[int]) -> tuple[tuple[Stmt, ...], Fundamental]:
    """Statements forming `T` and the weights `W{m}`, and the `Fundamental`.

        t2 = theta^2,  T = t2 |PQ|^2,  W0 = (2/sqrt(pi)) theta,
        W{m+1} = W{m} * (-2 t2),       boys[m] = F_m(T), m = 0..max(ms),
        [0]^(m) = W{m} * boys[m]
    """
    pqx, pqy, pqz = (Var(n) for n in PQ_PARAMS)
    theta = Var(THETA)
    top = max(ms, default=0)
    stmts: list[Stmt] = [
        Note("[0]^(m) = (2/sqrt(pi)) theta (-2 theta^2)^m F_m(theta^2 |PQ|^2), "
             "j_engine.md 'Two defects in the source'"),
        Decl("t2", Scalar.F64, init=_mul(theta, theta)),
        Decl("T", Scalar.F64, init=_mul(Var("t2"), _add(
            _add(_mul(pqx, pqx), _mul(pqy, pqy)), _mul(pqz, pqz)))),
    ]
    # `s` is read only by the `W{m}`, m >= 1, below: at top == 0 (L = 0) it
    # would be a dead declaration (nvcc warning #177-D, issue #62).
    if top >= 1:
        stmts.append(Decl("s", Scalar.F64, init=_mul(Lit(-2.0), Var("t2"))))
    stmts.append(Decl("W0", Scalar.F64, init=_mul(Lit(_TWO_OVER_SQRT_PI), theta)))
    for m in range(1, top + 1):
        stmts.append(Decl(f"W{m}", Scalar.F64, init=_mul(Var(f"W{m - 1}"), Var("s"))))
    stmts.extend(boys_fill(top, Var("T")))

    def fundamental(m: int) -> Expr:
        return _mul(Var(f"W{m}"), boys_read(m))

    return tuple(stmts), fundamental


def build_recurrence_kernel(
    dag: BatchDag, order: Order | None = None, *, name: str | None = None
) -> Kernel:
    """A recurrence-only Kernel for one batch: `out[i] = [target_order[i]](0)`.

    Params: `PQ_x PQ_y PQ_z theta` (F64 scalars) and `out` (F64 array, OUT,
    length `len(target_order(dag))`). `order` defaults to `schedule(dag)`.
    """
    order = schedule(dag) if order is None else order
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    body = [*prelude, *recurrence_body(dag, order, fundamental)]
    for i, r in enumerate(target_order(dag)):
        body.append(Assign(Index(OUT, (Lit(i),)), Var(node_name((r, 0)))))
    if name is None:
        direction, a, b = dag.batch
        top = max(sum(r) for r, _ in dag.targets)
        name = f"pq_recurrence_L{top}_{direction}{a}_{b}"
    params = (
        *(Param(n, Scalar.F64) for n in (*PQ_PARAMS, THETA)),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT),
    )
    return Kernel(name, params, tuple(body))


def recurrence_kernels(spec: Spec) -> tuple[tuple[BatchDag, Kernel], ...]:
    """One (dag, recurrence-only Kernel) per batch of `pipeline.plan(spec)`."""
    return tuple((dag, build_recurrence_kernel(dag)) for dag in plan(spec).dags)
