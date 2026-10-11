"""The `[p|q]` kernel of `[JEngine, v2, Alg. 1]`, built as a `Kernel`, not text.

    build_pq_kernel(L | spec) -> Kernel

One kernel per class, ONE BATCH per class (n_batches = 1), no atomics. It
consumes the plan (pipeline steps 1-3: the sieved MILD DAG), the schedule
(step 4) and the IR, and is pipeline step 5's input: a backend (`emit/`) only
spells it. It is not under `plan/`, whose package is host-side counting.

THE DATA, per launch. A *bra index* is one Hermite Gaussian product of the bra
side (centre `P`, exponent `zeta`, Schwarz bound `Ip`) and carries the vector
`J_p` over the bra Hermite functions `p` (|p| <= lp); a *ket index* is one of
the ket side (`Q`, `eta`, `Iq`, its density bound `Dmax`) and carries the
density vector `D_q` over |q| <= lq. With L = lp + lq, every `[p|q]` is
`(-1)^|q| [p+q]^(0)` ([JEngine, v2, eqs. 7-9]), so the recurrence of one
(bra, ket) pair -- `recurrence.recurrence_body`, the class's DAG in its
schedule -- forms exactly the targets `[r]^(0)`, |r| <= L, and nothing else.

    params  n_bra n_ket (I32)  tau (F64)
            P[n_bra, 3]  zeta[n_bra]  Ip[n_bra]                  bra side
            Q[n_ket, 3]  eta[n_ket]   Iq[n_ket]  Dmax[n_ket]     ket side
            D[n_ket, n_q]                                         density
            J[n_bra, n_p]  (OUT)                                  J_p

with `p` and `q` in `plan.targets.hermite_targets(lp)` / `(lq)` order.

THE CONTRACT, from docs/implementation/j_engine.md, each as an IR node:

- one warp per bra index, `bra = THREAD_X + BLOCK_X * BLOCK_DIM_X`; the
  launch is `launch_shape(n_bra, lanes)` -- blockDim = (1, lanes), so the
  `lanes` consecutive threads of a warp share one bra index. The warp width
  is the device target's fact, never the IR's: it appears here only as the
  argument a caller reads from `device.Arch`.
- kets strip-mined over `THREAD_Y`, stride `BLOCK_DIM_Y`: a `For`;
- the screen `Dmax * Ip * Iq < tau` is a `Break`, not a skip, because the
  kets ARRIVE SORTED descending by `Iq * Dmax` -- a precondition carried in
  the kernel as a `Note`. Unsorted kets make the break a WRONG ANSWER, not a
  slow one (test/test_kernel.py shows it);
- `J_p` accumulated in a `Space.SHARED` array, one slot per lane, then one
  `WarpReduce` per `p`;
- the global write is a plain `Assign` by lane 0, one per bra index per
  block, NOT an `AtomicAdd`: this is the single-batch case. `[JEngine, v2,
  Alg. 1]` presents line 10 as atomic-free and §5 corrects it once a class is
  split -- milestone 3, which is why `n_batches != 1` is refused here.

ONLY THE KET CONTRACTION IS AT SHELL-QUARTET LEVEL: `J_p = SUM_q [p|q] D_q`
([JEngine, v2, eqs. 20-22]). The `E` transform is bra-pair-only and is not in
this kernel, so it never materialises `[ab|cd]`: the only per-quartet values
are the recurrence's LOCAL scalars, and the only array it declares is the
shared `J_p` accumulator.

SHARED FOOTPRINT: `n_p * blockDim.y * 8` bytes. The default class for an int
L (`pq_spec`) splits L as lp = ceil(L/2), lq = floor(L/2), so for L <= 16,
lp <= 8, n_p <= 165, and the accumulator is at most 42 240 bytes at 32
lanes -- under the 48 KiB both of this box's cards give a block. A bra-heavy
split such as (16,0|0,0) needs 969 slots per lane and does not fit; that is a
launch-time limit, not an IR one, and the kernel tier is where it is measured.

Nothing here is CUDA- or HIP-specific, and nothing here imports `reference/`.
"""

from __future__ import annotations

from felizjueves.ir import (
    Assign,
    Break,
    Call,
    Decl,
    Expr,
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
    Var,
    WarpReduce,
)
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import R, hermite_targets
from felizjueves.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
)
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "JSH",
    "KET",
    "SORTED_KETS",
    "build_pq_kernel",
    "bra_functions",
    "ket_functions",
    "launch_shape",
    "pq_spec",
    "split",
]

KET = "k"  # the strip-mined ket loop's variable: what iteration counts key on
JSH = "Jsh"  # the shared J_p accumulator

SORTED_KETS = (
    "PRECONDITION: kets arrive sorted by Iq * Dmax, descending. The screen "
    "below is a break, not a continue: once Dmax * Ip * Iq < tau the whole "
    "tail is under it too. Unsorted kets make the break a wrong answer, not a "
    "slow one ([JEngine, v2, Alg. 1]; j_engine.md, 'The [p|q] kernel contract')"
)


# ---------------------------------------------------------------------------
# The class
# ---------------------------------------------------------------------------


def pq_spec(L: int, levers: tuple[str, ...] = ()) -> Spec:
    """The default class of total L: (ceil(L/2), 0 | floor(L/2), 0), one batch."""
    if type(L) is not int or L < 0:
        raise ValueError(f"L = {L!r} is not an int >= 0")
    return Spec(Operator.COULOMB, 0, ((L + 1) // 2, 0, L // 2, 0), Target.J, levers)


def split(spec: Spec) -> tuple[int, int]:
    """(lp, lq) = (l1 + l2, l3 + l4): the bra and ket Hermite orders."""
    l1, l2, l3, l4 = spec.am
    return l1 + l2, l3 + l4


def bra_functions(spec: Spec) -> tuple[R, ...]:
    """The bra Hermite functions p, |p| <= lp: the layout of `J[bra, :]`."""
    return hermite_targets(split(spec)[0])


def ket_functions(spec: Spec) -> tuple[R, ...]:
    """The ket Hermite functions q, |q| <= lq: the layout of `D[ket, :]`."""
    return hermite_targets(split(spec)[1])


def launch_shape(n_bra: int, lanes: int) -> tuple[int, tuple[int, int]]:
    """(grid, (blockDim.x, blockDim.y)) = (n_bra, (1, lanes)): one warp per bra.

    `lanes` is the device's warp width (`device.Arch`), passed in -- the IR
    does not know it. A wider block puts two warps on one bra index, each
    writing its own partial sum: the contract forbids it.
    """
    return n_bra, (1, lanes)


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _tree(terms: list[Expr]) -> Expr:
    """A balanced ADD tree: shallow, so a long sum does not nest 165 deep."""
    if len(terms) == 1:
        return terms[0]
    mid = len(terms) // 2
    return _c(Fn.ADD, _tree(terms[:mid]), _tree(terms[mid:]))


def _f64(name: str, *, array: bool = True, intent: Intent = Intent.IN) -> Param:
    return Param(name, Scalar.F64, array=array, intent=intent)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


def build_pq_kernel(cls: int | Spec, *, name: str | None = None) -> Kernel:
    """The [p|q] kernel of `[JEngine, v2, Alg. 1]` for one class, one batch.

    `cls` is a total L (the class `pq_spec(L)`) or a `Spec`. The recurrence is
    `pipeline.plan(spec)`'s DAG in `schedule`'s order; `n_batches != 1` is
    refused, because a split class needs the atomic write-back of milestone 3.
    """
    spec = pq_spec(cls) if isinstance(cls, int) else cls
    if not isinstance(spec, Spec):
        raise TypeError(f"build_pq_kernel takes an int L or a Spec, not {cls!r}")
    if spec.n_batches != 1:
        raise ValueError(
            f"n_batches = {spec.n_batches}: a split class writes J_p from "
            "several kernels and needs atomics ([JEngine, v2, §5]); milestone 3")
    lp, lq = split(spec)
    ps, qs = bra_functions(spec), ket_functions(spec)
    (dag,) = plan(spec).dags
    order = schedule(dag)

    k, ty = Var(KET), Call(Fn.THREAD_Y)
    bra = Var("bra")

    def jsh(i: int, lane: Expr = ty) -> Index:
        return Index(JSH, (Lit(i), lane))

    def ket(arr: str, *more: Expr) -> Index:
        return Index(arr, (k, *more))

    # --- one (bra, ket) pair: the screen, PQ and theta, the recurrence ------
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    pq_decls = [
        Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"), ket("Q", Lit(d))))
        for d, n in enumerate(PQ_PARAMS)
    ]
    contract: list[Stmt] = [
        Note("J_p += SUM_q (-1)^|q| [p+q]^(0) D_q, [JEngine, v2, eqs. 7-9, "
             "20-22]: the only shell-quartet-level contraction; [ab|cd] is "
             "never formed"),
    ]
    for i, p in enumerate(ps):
        plus: list[Expr] = []
        minus: list[Expr] = []
        for j, q in enumerate(qs):
            r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
            term = _c(Fn.MUL, Var(node_name((r, 0))), ket("D", Lit(j)))
            (minus if sum(q) % 2 else plus).append(term)
        value = _c(Fn.ADD, jsh(i), _tree(plus))
        if minus:
            value = _c(Fn.SUB, value, _tree(minus))
        contract.append(Assign(jsh(i), value))

    ket_body: tuple[Stmt, ...] = (
        Decl("bound", Scalar.F64,
             init=_c(Fn.MUL, Var("Ib"), _c(Fn.MUL, ket("Iq"), ket("Dmax")))),
        Note("screen: Dmax * Ip * Iq < tau, and every later ket is under it "
             "too (sorted) -- so break"),
        If(_c(Fn.LT, Var("bound"), Var("tau")), (Break(),)),
        *pq_decls,
        Decl("et", Scalar.F64, init=ket("eta")),
        Note("theta^2 = zeta eta / (zeta + eta), j_engine.md 'Two defects in "
             "the source'"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ze"), Var("et")),
            _c(Fn.ADD, Var("ze"), Var("et"))))),
        *prelude,
        *recurrence_body(dag, order, fundamental),
        *contract,
    )

    # --- one bra index: zero, strip-mine the kets, reduce, write ------------
    per_bra: list[Stmt] = [
        Note("accumulate J_p in shared memory, one slot per lane"),
        *(Assign(jsh(i), Lit(0.0)) for i in range(len(ps))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (bra, Lit(d))))
          for d in range(3)),
        Decl("ze", Scalar.F64, init=Index("zeta", (bra,))),
        Decl("Ib", Scalar.F64, init=Index("Ip", (bra,))),
        Note("kets strip-mined over threadIdx.y, stride blockDim.y"),
        For(KET, ty, Var("n_ket"), Call(Fn.BLOCK_DIM_Y), ket_body),
        Note("reduce J_p across the warp: every lane sees the sum"),
    ]
    for i in range(len(ps)):
        per_bra += [Decl(f"Jp{i}", Scalar.F64), WarpReduce(Var(f"Jp{i}"), jsh(i))]
    per_bra += [
        Note("plain write, NOT atomic: one batch per class, so this kernel is "
             "the only writer of J_p. [JEngine, v2, Alg. 1] line 10 is "
             "atomic-free; §5 corrects it once a class is split (milestone 3)"),
        If(_c(Fn.LT, ty, Lit(1)), tuple(
            Assign(Index("J", (bra, Lit(i))), Var(f"Jp{i}"))
            for i in range(len(ps)))),
    ]

    body: tuple[Stmt, ...] = (
        Note(f"[JEngine, v2, Alg. 1]: J_p = SUM_q [p|q] D_q, class lp={lp} "
             f"lq={lq} (L={lp + lq}), one batch, no atomics"),
        Note(SORTED_KETS),
        Note("launch: blockDim = (1, warp width), one warp per bra index"),
        Decl("bra", Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        Decl(JSH, Scalar.F64, (Lit(len(ps)), Call(Fn.BLOCK_DIM_Y)), Space.SHARED),
        If(_c(Fn.LT, bra, Var("n_bra")), tuple(per_bra)),
    )
    params = (
        Param("n_bra", Scalar.I32),
        Param("n_ket", Scalar.I32),
        _f64("tau", array=False),
        _f64("P"), _f64("zeta"), _f64("Ip"),
        _f64("Q"), _f64("eta"), _f64("Iq"), _f64("Dmax"),
        _f64("D"),
        _f64("J", intent=Intent.OUT),
    )
    if name is None:
        name = f"pq_L{lp + lq}_p{lp}_q{lq}"
    return Kernel(name, params, body)
