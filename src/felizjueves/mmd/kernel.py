"""The `[p|q]` kernel of `[JEngine, v2, Alg. 1]`, built as a `Kernel`, not text.

    build_pq_kernel(L | spec) -> Kernel              n_batches = 1
    build_pq_kernels(spec)    -> tuple[Kernel, ...]  one per batch, atomized
    pq_name(L | spec)         -> str                 build_pq_kernel's name, unbuilt

`build_pq_kernel` is one kernel per class, ONE BATCH (n_batches = 1), no
atomics; `build_pq_kernels` is a split class as a set of kernels, one per
batch of the plan, each contracting only its batch's targets and each
writing `J_p` through `passes.atomize` (see its docstring for the launch
convention and why the asymmetry is deliberate). Each
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
  `WarpReduce` per `p` -- per `p` the batch TOUCHES (`batch_contraction`);
  in the one-batch case that is every `p`;
- the global write is a plain `Assign` by lane 0, one per bra index per
  block, NOT an `AtomicAdd`, in the single-batch case. `[JEngine, v2,
  Alg. 1]` presents line 10 as atomic-free and §5 corrects it once a class is
  split, which is why a batch kernel's write is atomized.

ONLY THE KET CONTRACTION IS AT SHELL-QUARTET LEVEL: `J_p = SUM_q [p|q] D_q`
([JEngine, v2, eqs. 20-22]). The `E` transform is bra-pair-only and is not in
this kernel, so it never materialises `[ab|cd]`: the only per-quartet values
are the recurrence's LOCAL scalars, and the only array it declares is the
shared `J_p` accumulator.

SHARED FOOTPRINT: `n_p * blockDim.y * 8` bytes for one batch, and for a
batch the number of `p` it touches in place of `n_p`. The default class for an int
L (`pq_spec`) splits L as lp = ceil(L/2), lq = floor(L/2), so for L <= 16,
lp <= 8, n_p <= 165, and the accumulator is at most 42 240 bytes at 32
lanes -- under the 48 KiB both of this box's cards give a block. A bra-heavy
split such as (16,0|0,0) needs 969 slots per lane and does not fit; that is a
launch-time limit, not an IR one, and the kernel tier is where it is measured.

Nothing here is CUDA- or HIP-specific, and nothing here imports `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.j_coulomb.levers import fused, mixed
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
from felizjueves.ir.passes import atomize
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import R, hermite_targets
from felizjueves.spec import Operator, Spec, Target, lever_provenance

__all__ = [
    "JSH",
    "KET",
    "SORTED_KETS",
    "batch_contraction",
    "builder_plan",
    "lever_notes",
    "build_pq_kernel",
    "build_pq_kernels",
    "pq_name",
    "bra_functions",
    "ket_functions",
    "launch_shape",
    "k_spec",
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


def pq_spec(L: int, levers: tuple[str, ...] = (), n_batches: int = 1) -> Spec:
    """The default class of total L: (ceil(L/2), 0 | floor(L/2), 0), one
    batch unless `n_batches` says otherwise."""
    if type(L) is not int or L < 0:
        raise ValueError(f"L = {L!r} is not an int >= 0")
    return Spec(Operator.COULOMB, 0, ((L + 1) // 2, 0, L // 2, 0), Target.J,
                levers, n_batches)


def k_spec(L: int, *, n_batches: int = 1) -> Spec:
    """The default K class of total L: the one declared way to name a K cell.

    L splits as `pq_spec` splits it -- lp = ceil(L/2), lq = floor(L/2) -- so
    the K cell and the J cell of one L share (lp, lq), hence one plan and one
    Tables 1-3 row. Each pair's order is then distributed over its two
    shells, larger first: (la, lc) = (ceil(lp/2), floor(lp/2)), and
    (lb, ld) likewise from lq. `am` is `(la, lc, lb, ld)`, the Hermite
    pairing, bra first (`spec.py`, #196), so `split` reads (la + lc, lb + ld)
    unchanged.
    """
    if type(L) is not int or L < 0:
        raise ValueError(f"L = {L!r} is not an int >= 0")
    lp, lq = (L + 1) // 2, L // 2
    am = ((lp + 1) // 2, lp // 2, (lq + 1) // 2, lq // 2)
    return Spec(Operator.COULOMB, 0, am, Target.K, (), n_batches)


def split(spec: Spec) -> tuple[int, int]:
    """(lp, lq) = (l1 + l2, l3 + l4): the bra and ket Hermite orders.

    For a K class, `am = (la, lc, lb, ld)`, that is (la + lc, lb + ld)."""
    l1, l2, l3, l4 = spec.am
    return l1 + l2, l3 + l4


def pq_name(cls: int | Spec) -> str:
    """The name `build_pq_kernel(cls)` gives its kernel, without building it:
    `pq_L{L}_p{lp}_q{lq}`, a fixed function of the class (#419)."""
    spec = pq_spec(cls) if isinstance(cls, int) else cls
    lp, lq = split(spec)
    return f"pq_L{lp + lq}_p{lp}_q{lq}"


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


def _f64(name: str, *, array: bool = True, intent: Intent = Intent.IN,
         shape: tuple[Expr, ...] = ()) -> Param:
    return Param(name, Scalar.F64, array=array, intent=intent, shape=shape)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


def build_pq_kernel(cls: int | Spec, *, name: str | None = None) -> Kernel:
    """The [p|q] kernel of `[JEngine, v2, Alg. 1]` for one class, one batch.

    `cls` is a total L (the class `pq_spec(L)`) or a `Spec`. The recurrence is
    `pipeline.plan(spec)`'s DAG in `schedule`'s order; `n_batches != 1` is
    refused, because a split class writes `J_p` from several kernels and
    needs atomics: that is `build_pq_kernels`.

    The plan's levers are the plan layer's: `no-sieve`, and `tuned` (#161),
    the searched board's recorded cut and seed orders
    (`pipeline.tuned_plan`; a cell with no recorded choice is refused). A
    kernel-body lever (`levers.VOCABULARY`, #337) stays with the builder,
    which records it in the kernel as a `Note` (`lever_notes`); the plan
    never sees it (`builder_plan`). With no lever every emitted byte is the
    literal reconstruction's.
    """
    spec = pq_spec(cls) if isinstance(cls, int) else cls
    if not isinstance(spec, Spec):
        raise TypeError(f"build_pq_kernel takes an int L or a Spec, not {cls!r}")
    if spec.n_batches != 1:
        raise ValueError(
            f"n_batches = {spec.n_batches}: a split class writes J_p from "
            "several kernels and needs atomics ([JEngine, v2, §5]); "
            "build_pq_kernels builds it")
    (dag,) = builder_plan(spec).dags
    if name is None:
        name = pq_name(spec)
    return _pq_kernel(spec, dag, name, "one batch")


def build_pq_kernels(spec: Spec) -> tuple[Kernel, ...]:
    """One kernel per batch of `plan(spec)`, in partition order, each atomized.

    `n_batches == 1` delegates to `build_pq_kernel`: one kernel, a plain
    write, byte-identical to it -- so its golden files stay the control.

    A batch's kernel is named `pq_L{L}_p{lp}_q{lq}_{direction}{a}_{b}`, after
    `recurrence.build_recurrence_kernel`'s convention, and contracts only the
    `(p, q)` pairs whose `p + q` is one of that batch's targets (see
    `batch_contraction`); its write-back is `passes.atomize`d.

    THE ASYMMETRY IS DELIBERATE. The one-batch kernel keeps the plain write
    and a split one is atomic throughout: `[JEngine, v2, Alg. 1]` line 10 is
    atomic-free, and §5 corrects it only for the split case, where several
    kernels write the same `J_p`. Do not "unify" the two without a
    measurement -- an atomic the one-batch class does not need is a cost.

    THE LAUNCH, a convention rather than a type (its consumer, the kernel
    tier, has not fixed a signature): every kernel takes the same arguments
    and the same `launch_shape(n_bra, lanes)`, and they run in the returned
    order -- `Plan.dags` order -- into ONE `J` that the caller ZEROES first.
    Each batch adds its partial `J_p`; the batches' targets partition the
    class, so the partial sums add up to the whole. The sorted-ket
    precondition (`SORTED_KETS`) is per launch, so every batch carries it.
    The fundamentals are recomputed per batch, on purpose ([JEngine, v2,
    §5]): hoisting them across batches is a lever needing a measurement, and
    would reintroduce the cross-batch dependency batching exists to remove.
    """
    if not isinstance(spec, Spec):
        raise TypeError(f"build_pq_kernels takes a Spec, not {spec!r}")
    if spec.n_batches == 1:
        return (build_pq_kernel(spec),)
    lp, lq = split(spec)
    out = []
    for i, dag in enumerate(builder_plan(spec).dags):
        direction, a, b = dag.batch
        name = f"pq_L{lp + lq}_p{lp}_q{lq}_{direction}{a}_{b}"
        head = (f"batch {i + 1} of {spec.n_batches}, "
                f"{direction.upper()}_{{{a},{b}}}")
        out.append(atomize(_pq_kernel(spec, dag, name, head)))
    return tuple(out)


def builder_plan(spec: Spec):
    """`pipeline.plan` of `spec` without its kernel-body levers (#337).

    Those levers are the builder's (`levers.VOCABULARY`), not the plan
    layer's, which refuses any lever it does not implement; the rest (`no-
    sieve`, `tuned`) reach the plan as they always have."""
    return plan(spec.without_kernel_body_levers())


def lever_notes(spec: Spec) -> tuple[Note, ...]:
    """One provenance `Note` per kernel-body lever `spec` carries, in key
    order: what makes a lever-carrying build observable in its `Kernel`
    (#337). Empty for every spec without one, so no literal byte moves."""
    return tuple(Note(lever_provenance(lv))
                 for lv in spec.kernel_body_levers())


Contraction = tuple[tuple[int, R, tuple[tuple[int, R], ...]], ...]


def batch_contraction(spec: Spec, dag: BatchDag) -> Contraction:
    """What one batch's kernel contracts: `((i, p, ((j, q), ...)), ...)`.

    Every bra function `p` (index `i` in `bra_functions(spec)`) with at least
    one ket function `q` (index `j` in `ket_functions(spec)`) such that
    `(p + q, 0)` is one of `dag.targets`, with those `q` in order. A `p` with
    no such `q` is UNTOUCHED by the batch and is absent: no shared slot, no
    `WarpReduce`, no write -- not an added zero. This is the ONE list the
    kernel's shared extent, its reductions and its write-back all key on.
    """
    out = []
    for i, p in enumerate(bra_functions(spec)):
        qs = tuple((j, q) for j, q in enumerate(ket_functions(spec))
                   if ((p[0] + q[0], p[1] + q[1], p[2] + q[2]), 0) in dag.targets)
        if qs:
            out.append((i, p, qs))
    return tuple(out)


def _contraction(touched, jsh, ket, target=Var) -> list[Stmt]:
    """The contraction block: one shared update per touched `p`, reading
    each target `[p+q]^(0)` as `target(name)` (a plain `Var`, or `mixed`'s
    widening of an F32 one)."""
    contract: list[Stmt] = [
        Note("J_p += SUM_q (-1)^|q| [p+q]^(0) D_q, [JEngine, v2, eqs. 7-9, "
             "20-22]: the only shell-quartet-level contraction; [ab|cd] is "
             "never formed"),
    ]
    for slot, (_, p, pairs) in enumerate(touched):
        plus: list[Expr] = []
        minus: list[Expr] = []
        for j, q in pairs:
            r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
            term = _c(Fn.MUL, target(node_name((r, 0))), ket("D", Lit(j)))
            (minus if sum(q) % 2 else plus).append(term)
        value: Expr = jsh(slot)
        if plus:
            value = _c(Fn.ADD, value, _tree(plus))
        if minus:
            value = _c(Fn.SUB, value, _tree(minus))
        contract.append(Assign(jsh(slot), value))
    return contract


def _pair_region(spec, dag, order, operands, touched, jsh,
                 ket) -> tuple[Stmt, ...]:
    """The ket loop's tail: the pair operands (`PQ_d`, `eta`, `theta`), the
    fundamentals' prelude, the recurrence region, then the contraction --
    or, under the `fused` lever (#344, `j_levers.fused`), the two
    interleaved so each target is contracted as it is formed; under the
    `mixed` lever (#346, `j_levers.mixed`), the operands narrowed and the
    prelude and recurrence F32 inside the FP64 boundary. The two compose."""
    levers = spec.kernel_body_levers()
    is_fused = fused.FUSED in levers
    ms = [m for (r, m) in order if not any(r)]
    target = Var
    if mixed.MIXED in levers:
        prelude, fundamental = mixed.prelude(ms)
        head = (*mixed.operands(operands), *prelude)
        body = mixed.narrow(recurrence_body(dag, order, fundamental),
                            dag.targets, widen_targets=is_fused)
        target = mixed.widen
    else:
        prelude, fundamental = fundamental_prelude(ms)
        head = (*operands, *prelude)
        body = recurrence_body(dag, order, fundamental)
    if is_fused:
        return (*head, *fused.fuse(body, touched, jsh=jsh,
                                   density=lambda j: ket("D", Lit(j))))
    return (*head, *body, *_contraction(touched, jsh, ket, target))


def _pq_kernel(spec: Spec, dag: BatchDag, name: str, head: str) -> Kernel:
    """The [p|q] kernel over one batch's DAG, with the plain lane-0 write."""
    lp, lq = split(spec)
    ps, qs = bra_functions(spec), ket_functions(spec)
    order = schedule(dag)
    touched = batch_contraction(spec, dag)
    if not touched:
        raise ValueError(f"batch {dag.batch} of {spec.key()} contracts nothing")

    k, ty = Var(KET), Call(Fn.THREAD_Y)
    bra = Var("bra")

    def jsh(slot: int, lane: Expr = ty) -> Index:
        return Index(JSH, (Lit(slot), lane))

    def ket(arr: str, *more: Expr) -> Index:
        return Index(arr, (k, *more))

    # --- one (bra, ket) pair: the screen, PQ and theta, the recurrence ------
    pq_decls = [
        Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"), ket("Q", Lit(d))))
        for d, n in enumerate(PQ_PARAMS)
    ]
    operands: tuple[Stmt, ...] = (
        *pq_decls,
        Decl("et", Scalar.F64, init=ket("eta")),
        Note("theta^2 = zeta eta / (zeta + eta), j_engine.md 'Two defects in "
             "the source'"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ze"), Var("et")),
            _c(Fn.ADD, Var("ze"), Var("et"))))),
    )

    ket_body: tuple[Stmt, ...] = (
        Decl("bound", Scalar.F64,
             init=_c(Fn.MUL, Var("Ib"), _c(Fn.MUL, ket("Iq"), ket("Dmax")))),
        Note("screen: Dmax * Ip * Iq < tau, and every later ket is under it "
             "too (sorted) -- so break"),
        If(_c(Fn.LT, Var("bound"), Var("tau")), (Break(),)),
        *_pair_region(spec, dag, order, operands, touched, jsh, ket),
    )

    # --- one bra index: zero, strip-mine the kets, reduce, write ------------
    per_bra: list[Stmt] = [
        Note("accumulate J_p in shared memory, one slot per lane"),
        *(Assign(jsh(slot), Lit(0.0)) for slot in range(len(touched))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (bra, Lit(d))))
          for d in range(3)),
        Decl("ze", Scalar.F64, init=Index("zeta", (bra,))),
        Decl("Ib", Scalar.F64, init=Index("Ip", (bra,))),
        Note("kets strip-mined over threadIdx.y, stride blockDim.y"),
        For(KET, ty, Var("n_ket"), Call(Fn.BLOCK_DIM_Y), ket_body),
        Note("reduce J_p across the warp: every lane sees the sum"),
    ]
    for slot, (i, _, _) in enumerate(touched):
        per_bra += [Decl(f"Jp{i}", Scalar.F64),
                    WarpReduce(Var(f"Jp{i}"), jsh(slot))]
    per_bra += [
        Note("plain write, NOT atomic: one batch per class, so this kernel is "
             "the only writer of J_p. [JEngine, v2, Alg. 1] line 10 is "
             "atomic-free; §5 corrects it once a class is split (milestone 3)"),
        If(_c(Fn.LT, ty, Lit(1)), tuple(
            Assign(Index("J", (bra, Lit(i))), Var(f"Jp{i}"))
            for i, _, _ in touched)),
    ]

    body: tuple[Stmt, ...] = (
        Note(f"[JEngine, v2, Alg. 1]: J_p = SUM_q [p|q] D_q, class lp={lp} "
             f"lq={lq} (L={lp + lq}), {head}, no atomics"),
        *lever_notes(spec),
        Note(SORTED_KETS),
        Note("launch: blockDim = (1, warp width), one warp per bra index"),
        Decl("bra", Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        Decl(JSH, Scalar.F64, (Lit(len(touched)), Call(Fn.BLOCK_DIM_Y)),
             Space.SHARED),
        If(_c(Fn.LT, bra, Var("n_bra")), tuple(per_bra)),
    )
    nb, nk = Var("n_bra"), Var("n_ket")
    params = (
        Param("n_bra", Scalar.I32),
        Param("n_ket", Scalar.I32),
        _f64("tau", array=False),
        _f64("P", shape=(nb, Lit(3))), _f64("zeta", shape=(nb,)),
        _f64("Ip", shape=(nb,)),
        _f64("Q", shape=(nk, Lit(3))), _f64("eta", shape=(nk,)),
        _f64("Iq", shape=(nk,)), _f64("Dmax", shape=(nk,)),
        _f64("D", shape=(nk, Lit(len(qs)))),
        _f64("J", intent=Intent.OUT, shape=(nb, Lit(len(ps)))),
    )
    return Kernel(name, params, body)
