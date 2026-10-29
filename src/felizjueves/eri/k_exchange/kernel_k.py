"""The exchange kernel, `Target.K`: the staged contraction as IR (#198).

    build_k_kernel((la, lc, lb, ld) | spec) -> Kernel     n_batches = 1
    build_k_kernels(spec) -> tuple[Kernel, ...]  one per batch, atomized (#203)
    k_class_spec(la, lc, lb, ld)            -> Spec       the class's key
    launch_shape(n_bra, lanes)              -> (grid, (1, lanes))
    shared_doubles(am) / shared_bytes(am, lanes)          the footprint

`K_ab = SUM_cd D_cd (ac|bd)` on the J engine's recurrence, never by forming
`(ac|bd)`. The BRA pair record is `(a, c)` and the KET pair record is
`(b, d)` -- the two Hermite pairs, not the AO quartet order -- so the class
is `am = (la, lc, lb, ld)`, bra first, and `(lp, lq) = (la + lc, lb + ld)`
(`spec.py`, #196). Per primitive quartet (milestone 10's contraction, the
host's `eri.k_exchange.k.k_quartet`, #197):

    stage 1 (ket)     I[p]     = SUM_q [p|q] E^bd_q      per (b,d) row
    stage 2 (bra)     V(a,c)   = SUM_p E^ac_p I[p]       per (a,c) row
    stage 3 (digest)  K[a,b]  += D[c,d] V(a,c)

THE PLAN IS J's. `[p|q] = (-1)^|q| [p+q]^(0)` and the plan depends only on
the two Hermite orders, so the recurrence region is `recurrence.recurrence_body`
over `pipeline.plan(spec)`'s one DAG in `schedule`'s order -- the statements
`kernel.build_pq_kernel` emits for the `Target.J` class with the same `am`
(test/test_kernel_k.py asserts them `==`). Only the digestion differs.

THE THREAD MAPPING IS J's, unchanged (milestone 10's pin): one warp per bra
pair record, `bra = THREAD_X + BLOCK_X * BLOCK_DIM_X` with blockDim =
(1, lanes) (`launch_shape`, which is `kernel.launch_shape`), and the kets
strip-mined over `THREAD_Y`, stride `BLOCK_DIM_Y`. The output is one
`K[a, b]` block PER BRA RECORD, as `J_p` is one vector per bra record: the
sum over a bra shell pair's records, and the scatter into `K_mn`, are the
host's (the launch plan's, D1), not this kernel's.

    params  n_bra n_ket n_dc (I32)  tau maxD (F64)
            P[n_bra, 3]  zeta_ac[n_bra]  XPA[n_bra, 3]  XPC[n_bra, 3]
            s_ac[n_bra]  Ip[n_bra]  dc[n_bra] (I32)                bra side
            Q[n_ket, 3]  eta[n_ket]  XQB[n_ket, 3]  XQD[n_ket, 3]
            s_bd[n_ket]  Iq[n_ket]  dd[n_ket] (I32)                ket side
            Dcd[n_dc, n_c n_d]                                     density
            K[n_bra, n_a n_b]  (OUT)

- `zeta_ac`, not `zeta`: both inlined `E` fragments bind a SCALAR named
  `zeta`, and `ir`'s hygiene (#149) refuses a host array of that name
  (`kernel_jab.py`'s `zeta_ab`, the same reason). `XPA = P - A`,
  `XPC = P - C`, `XQB = Q - B`, `XQD = Q - D` arrive from the host, as
  `kernel_dq.py`'s `XPA` / `XPB` do.
- THE DENSITY STRADDLES THE TWO RECORDS: `D_cd` has its `c` on the bra and
  its `d` on the ket. Its row is `dc[bra] + dd[k]` -- the host lays `Dcd`
  out so that sum names the `(c shell, d shell)` block (one row per shell
  pair, `cartesians(lc) x cartesians(ld)`, c outer); the kernel only adds.
  Kets of several `d` shells may share a launch; they must share the one
  `b` shell whose column of `K` the block is, which is the launch plan's
  business (D1), not checked here.
- `n_dc` is `Dcd`'s first extent: an extent may use only scalar `Param`s.

THE SCREEN is J's `break`, on K's bound: `Ip * Iq * maxD < tau`, `maxD` the
host's global `max |D|`, with the kets SORTED DESCENDING BY `Iq` ALONE
(`SORTED_BY_IQ`). The J engine's key `I_ket * Dmax_ket` is not available to
K (`NO_DENSITY_KEY`): `D_cd` straddles the two records, so no ket order can
track it -- `[Fock, §4.3.1]`'s sentence, cited as prior art (milestone 10).
The bound itself is proved host-side (#200); at `tau = 0` the bound is
`>= 0`, never `< tau`, and nothing is skipped.

THE DIGESTION, in the host's order (`eri.k_exchange.k.k_quartet`):

- `E^ac` is formed ONCE per bra record, into a LOCAL `Eac[n_ac, n_p]`, from
  `e_fragment.block_fragment(la, lc)` inlined; `E^bd` once per ket record,
  into `Ebd[n_bd, n_q]`, from `block_fragment(lb, ld)` inlined -- the first
  kernel to inline two. Both are the UNDRESSED `E`; the dressings
  `s_ac = c_a c_c (pi/zeta)^{3/2}` and `s_bd` arrive as one host scalar per
  record per side (`ao.pairs.dress` via `ao.density.epair`), and every use is
  `s * E[r, t]`, as the host forms `s * e`. Neither fragment forms a
  contraction coefficient or a `(pi/zeta)^{3/2}`: m7's dressing pin, on two
  sides at once.
- per `(b,d)` row (`b` outer): stage 1 writes `I[p]` for every `p`, a left
  fold over `q` in `hermite_targets(lq)` order from `+0.0`, each term
  `[p+q]^(0) * (s_bd * Ebd[bd, q])` ADDED for even `|q|` and SUBTRACTED for
  odd (`acc - x` is `acc + (-x)` exactly, and `(-r) * e` is `-(r * e)`);
  then per `(a,c)` row (`a` outer) stage 2 is a left fold over `p` from
  `+0.0` of `(s_ac * Eac[ac, p]) * I[p]`, and stage 3 adds `Dcd[row, cd] *`
  that fold into `K[a, b]`. STAGES 2 AND 3 ARE FUSED: `V` is an expression,
  never a local, never an array. Forced, not chosen: held as arrays, `I` is
  `n_p x n_bd` and `V` is `n_ac x n_bd`, 2,556 doubles per lane at
  `(dd|dd)`; one column at a time leaves `n_p + n_a n_b` (35 + 36).
- A term whose `E` entry is STRUCTURALLY ZERO (`t_d > a_d + b_d` on some
  axis) is ABSENT -- the fragment never writes it, and the interpreter
  refuses a read before a write. The host adds `x * (s * 0.0) = +-0.0`
  there, and adding `+-0.0` to an accumulator that started at `+0.0` is the
  identity. The host also multiplies a zero `D_cd` in, as the kernel does.
  `kernel_dq.py`'s two departures, the same argument.

So, interpreted with one lane, the kernel is `==` to `eri.k_exchange.k.k_quartet` on a
bra pair of that one record, bit for bit, given a `pq=` whose values are the
recurrence's own (test/test_kernel_k.py). With more lanes the per-lane
partial sums are combined by the warp's butterfly: a reassociation.

SHARED MEMORY. `I` and the `K[a, b]` accumulator live in ONE `Space.SHARED`
array `Ksh[n_p + n_a n_b, BLOCK_DIM_Y]`, where `J_p` sits in the J kernel --
rows `0 .. n_p-1` the lane's `I` column, rows `n_p ..` its `K` accumulator,
slot `n_p + a n_b + b`. One array, because the backends admit one
dynamically sized shared array (`emit/_c.py`). Then one `WarpReduce` per
`(a, b)`, and lane 0 writes `K[bra, a n_b + b]` -- a plain write, one batch.
The footprint is

    shared_bytes = (n_p + n_a n_b) * lanes * 8,
    n_p = (lp+1)(lp+2)(lp+3)/6,  n_x = (l_x+1)(l_x+2)/2,

at most `(35 + 36) * 32 * 8 = 18,176` bytes through `(dd|dd)`, against the
48 KiB both of this box's cards give a block. It is arithmetic of the class,
not of a built kernel (`shared_doubles`), so a table needs no build.

THE SPLIT CLASS (#203): `build_k_kernels(spec)`, one kernel per batch of the
plan, and where the partial sums go. A batch holds only some of the Hermite
targets `r = p + q`, so a batch kernel can complete stage 1 for only some
`p`, while stage 2 sums over ALL `p`. Two forms are possible:

- THE PARTIAL-SUM FORM -- BUILT, the default and the only form built. Each
  batch kernel runs the batch's own DAG (the plan's, exactly as
  `kernel.build_pq_kernels` runs it for `Target.J`) and contracts only that
  batch's terms: stage 1 the `(p, q)` with `p + q` in the batch (and
  `E^bd_q` structurally nonzero), stage 2 the `p` whose `I[p]` the batch
  holds for that `(b,d)` row (and `E^ac_p` nonzero), stage 3 once per
  `(a,c)` row whose `V` is not structurally zero IN THIS BATCH. It adds its
  partial `K[a, b]` through `passes.atomize` (`array="K"`), as a split J
  kernel adds its partial `J_p` -- so the fused stage-2/3 digest runs once
  per batch per `(b,d)` row, and the `E^ac` contraction of a `p` that
  several batches hold is DUPLICATED across them. That duplication is what
  `kdup.py` (E1, #202) predicts, and test/test_kernel_k_batched.py counts it
  off the built kernels and asserts the two equal. Chosen because it is the
  direct analogue of the J engine's split, and so the arm the
  `jengine/k-recompute-has-more-merit` claim compares against. What a slot
  holds is per batch: `Ksh`'s rows are the `p` the batch holds, then the
  `K[a, b]` it writes, compacted; a `p` or a `K[a, b]` the batch never
  touches gets no slot, no `WarpReduce` and no write, not an added zero.
  Each batch kernel also re-forms both `E` blocks (whole, as the one-batch
  kernel does): `n_batches` evaluations of each, which E1 does not count
  (its count is LOW on purpose).
- THE RECOMPUTE FORM -- NOT BUILT, the lever E1/E2 score against: a batch
  kernel recomputes whatever `[r]^(0)` it needs so that every `p` of a
  given `K[a, b]` is in one kernel (`[JEngine, v2, §6]`'s "extreme case"),
  paying re-evaluated two-centre Hermite integrals instead of duplicated
  stage-2 work. It does not fall out of this builder: its batches would be
  partitioned by `p` (or by output), not by `r`, which is a different plan
  partition and a different DAG per kernel -- a second code path, so per
  #203 it is not emitted.

WHICH THE PAPER'S SENTENCE PREDICTS SHOULD WIN, written down before any
count (E1, #202, counts the arithmetic; this builder only cross-checks it):
`[JEngine, v2, §6]` says recomputing `[r]^(0)` in several kernels "has more
merit when applied to the calculation of the exchange matrix K due to the
additional contractions that must be completed at the shell quartet level".
So the sentence predicts the RECOMPUTE form beats this partial-sum form for
K (and by more than for J): the partial-sum form pays the duplicated
quartet-level stage-2 contraction, which J's split does not have (J's
products are partitioned by `r`, never duplicated). Whether that holds on
a card is E2's measurement and E3's score, not this module's.

EXACTNESS OF THE SPLIT. Interpreted with one lane, each batch kernel is
`==` `eri.k_exchange.k.k_quartet` with `pq` restricted to the batch's targets (a zero
elsewhere: every term the kernel omits is the host's `+-0.0`, the identity
on an accumulator), bit for bit; and the batches, run in `Plan.dags` order
onto a zeroed `K`, are `==` the host's left fold of those partials in that
order (an atomic add onto a zero is exact). Against the ONE-BATCH kernel
the sum agrees to ROUNDING ONLY, never bitwise in general: the split
reassociates every sum it cuts -- `I[p]`'s fold over `q` is cut by `r`,
`V`'s fold over `p` is cut by batch, and `D (V_1 + V_2)` becomes
`D V_1 + D V_2`. On a card the atomics also add in hardware order, so a
batched `K` is not bit-reproducible run to run (`passes.atomize`'s note).

THE LEVER SEAMS (#367). `_k_kernel` below is a SKELETON: the parts no
milestone-13 lever moves are laid out here, and every part a lever does move
-- the accumulator slicing, a fan-out into several kernels, the shared
array, the `E^ac` / `E^bd` bindings and reads, the contraction region (stage
1, stages 2 and 3), the accumulator's zeroing, the write-back, the parameters
and a last rewrite, and (#345) a batch kernel's atomic form -- is asked of
a `k_levers.seams.KHooks`, whose defaults are this module's code before
#367, moved statement for statement (the literal kernel is byte-identical:
test/test_k_seams.py). A K lever lands as
its own module, `eri/k_exchange/levers/<name>.py`, registered in `k_levers.REGISTRY`;
the
builders resolve a spec's levers through `k_levers.hooks_for` and this file
does not change for a new lever.

Nothing here is CUDA- or HIP-specific, and nothing imports `ao/` (the host
staging it is checked against), the interpreter, `reference/`, a backend or
a device.
"""

from __future__ import annotations

from felizjueves.eri.k_exchange.levers import hooks_for
from felizjueves.eri.k_exchange.levers.seams import (
    KET,
    KSH,
    PARAMS,
    SPLIT_WRITE,
    EmptyKernel,
    KContext,
    KHooks,
    ncart,
)
from felizjueves.ir import (
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
    Scalar,
    Stmt,
    Var,
)
from felizjueves.mmd.kernel import builder_plan, lever_notes, split
from felizjueves.mmd.kernel import launch_shape as _j_launch_shape
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    recurrence_body,
)
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "KET",
    "KSH",
    "NO_DENSITY_KEY",
    "PARAMS",
    "SORTED_BY_IQ",
    "SPLIT_WRITE",
    "build_k_kernel",
    "build_k_kernels",
    "k_class_spec",
    "ket_quartet",
    "launch_shape",
    "shared_bytes",
    "shared_doubles",
]

SORTED_BY_IQ = (
    "PRECONDITION: kets arrive sorted by Iq ALONE, descending. The screen "
    "below is a break, not a continue: once Ip * Iq * maxD < tau the whole "
    "tail is under it too. Unsorted kets make the break a wrong answer, not "
    "a slow one ([JEngine, v2, Alg. 1]'s break; K's bound, #200)")

NO_DENSITY_KEY = (
    "The J engine's ket key I_ket * Dmax_ket is UNAVAILABLE to K: D_cd "
    "straddles the two records (c on the bra, d on the ket), so no ket "
    "order can track it, and the bound carries the global maxD instead. "
    "Prior art, [Fock, §4.3.1]: 'Unlike the J scheme, we cannot order our "
    "shell pairs based on the magnitude of the density matrix.'")


def k_class_spec(la: int, lc: int, lb: int, ld: int) -> Spec:
    """The K class `(la, lc | lb, ld)` -- Hermite pairing, bra first."""
    return Spec(Operator.COULOMB, 0, (la, lc, lb, ld), Target.K)


def _spec_of(cls: tuple[int, int, int, int] | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = k_class_spec(*cls)
    else:
        raise TypeError("build_k_kernel takes (la, lc, lb, ld) or a Spec, "
                        f"not {cls!r}")
    if spec.target is not Target.K:
        raise ValueError(f"{spec.key()} is not a Target.K class")
    if spec.n_batches != 1:
        raise ValueError(
            f"n_batches = {spec.n_batches}: a split K class writes K[a,b] "
            "from several kernels and needs atomics (the partial-sum form, "
            "#203); build_k_kernels builds it")
    return spec


def shared_doubles(am: tuple[int, int, int, int]) -> int:
    """Doubles of `Ksh` per lane: `n_p + n_a n_b`, from the class alone."""
    la, lc, lb, _ = am
    lp = la + lc
    n_p = (lp + 1) * (lp + 2) * (lp + 3) // 6
    return n_p + ncart(la) * ncart(lb)


def shared_bytes(am: tuple[int, int, int, int], lanes: int) -> int:
    """The block's shared footprint: `shared_doubles(am) * lanes * 8`."""
    return shared_doubles(am) * lanes * 8


def launch_shape(n_bra: int, lanes: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (1, lanes))`: one warp per bra record -- `kernel.launch_shape`,
    the J mapping, unchanged."""
    return _j_launch_shape(n_bra, lanes)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _hooks(spec: Spec, hooks: KHooks | None) -> KHooks:
    """`hooks` when given (a test's sketch), else the spec's levers composed
    through `k_levers.REGISTRY` -- `DEFAULT_HOOKS` for a literal spec."""
    if hooks is None:
        return hooks_for(spec)
    if not isinstance(hooks, KHooks):
        raise TypeError(f"hooks= takes a KHooks, not {hooks!r}")
    return hooks


def build_k_kernel(cls: tuple[int, int, int, int] | Spec, *,
                   name: str | None = None,
                   hooks: KHooks | None = None) -> Kernel:
    """The exchange kernel of one K class, one batch (module docstring).

    `cls` is `(la, lc, lb, ld)` -- the Hermite pairing, bra first, NOT the AO
    quartet order -- or a `Target.K` `Spec`. A kernel-body lever
    (`levers.VOCABULARY`, #337) is kept from the plan (`kernel.builder_plan`)
    and recorded in the kernel as a `Note` (`kernel.lever_notes`); a
    registered one also moves its hooks (`k_levers`, #367). `hooks=`
    replaces that resolution outright. A lever that fans the class out into
    several kernels is refused here: `build_k_kernels` builds it.
    """
    spec = _spec_of(cls)
    hooks = _hooks(spec, hooks)
    variants = hooks.fan_out(spec, hooks)
    if len(variants) != 1:
        raise ValueError(
            f"{spec.key()} fans out into {len(variants)} kernels; "
            "build_k_kernels builds it")
    ((suffix, hooks),) = variants
    la, lc, lb, ld = spec.am
    (dag,) = builder_plan(spec).dags
    if name is None:
        name = f"k_{la}{lc}_{lb}{ld}{suffix}"
    return _k_kernel(spec, dag, name, None, hooks)


def build_k_kernels(spec: Spec, *,
                    hooks: KHooks | None = None) -> tuple[Kernel, ...]:
    """One kernel per batch of `plan(spec)`, in `Plan.dags` order: THE
    PARTIAL-SUM FORM, each kernel's `K` write `passes.atomize`d (module
    docstring, "THE SPLIT CLASS").

    `n_batches == 1` delegates to `build_k_kernel`: one kernel, a plain
    write, `==` to it. A batch kernel is named
    `k_{la}{lc}_{lb}{ld}_{direction}{a}_{b}` (`kernel.build_pq_kernels`'
    convention). The batch count is `Spec`'s rule (1 or a multiple of 3,
    refused when the `Spec` is made) and the plan's (an empty batch is
    refused by `plan`, #64); this builder adds no rule of its own.

    A lever's `fan_out` hook (#367) expands each batch's kernel into its
    variants, in the order it returns them, each name suffixed by the
    variant's suffix; with no fan-out the tuple is exactly the above.

    THE LAUNCH: every kernel takes the same arguments and the same
    `launch_shape(n_bra, lanes)`, in the returned order, into ONE `K` the
    caller ZEROES first; each adds its partial `K[a, b]`.
    """
    if not isinstance(spec, Spec):
        raise TypeError(f"build_k_kernels takes a Spec, not {spec!r}")
    if spec.target is not Target.K:
        raise ValueError(f"{spec.key()} is not a Target.K class")
    hooks = _hooks(spec, hooks)
    variants = hooks.fan_out(spec, hooks)
    la, lc, lb, ld = spec.am
    if spec.n_batches == 1 and len(variants) == 1:
        return (build_k_kernel(spec, hooks=hooks),)

    def variant(dag, name: str, head: str | None, h: KHooks):
        try:
            return _k_kernel(spec, dag, name, head, h)
        except EmptyKernel:
            if len(variants) == 1:
                raise
            return None  # a fan-out variant this batch never reaches

    out = []
    for i, dag in enumerate(builder_plan(spec).dags):
        if spec.n_batches == 1:
            name, head = f"k_{la}{lc}_{lb}{ld}", None
        else:
            direction, a, b = dag.batch
            name = f"k_{la}{lc}_{lb}{ld}_{direction}{a}_{b}"
            head = (f"batch {i + 1} of {spec.n_batches}, "
                    f"{direction.upper()}_{{{a},{b}}}")
        for suffix, h in variants:
            kern = variant(dag, name + suffix, head, h)
            if kern is not None:
                out.append(kern)  # a batch kernel arrives atomized
    return tuple(out)


def _k_kernel(spec: Spec, dag, name: str, head: str | None,
              hooks: KHooks) -> Kernel:
    """The K kernel over one batch's DAG, with the plain lane-0 write: the
    skeleton, every lever-movable part asked of `hooks` (#367).

    `head is None` is the one-batch kernel (`build_k_kernel`); otherwise the
    batch's label, and the kernel holds only that batch's terms (module
    docstring, "THE SPLIT CLASS")."""
    la, lc, lb, ld = spec.am
    lp, lq = split(spec)
    ctx = KContext.of(spec, dag, name, head, hooks)
    if head is None and hooks.is_default("keep"):
        assert ctx.n_shared == shared_doubles(spec.am)

    ty, bra = ctx.ty, ctx.bra
    per_bra: list[Stmt] = [
        *bra_scalars(ctx),
        *hooks.bind_e_ac(ctx),
        *hooks.init_acc(ctx),
        Note("kets strip-mined over threadIdx.y, stride blockDim.y"),
        For(KET, ty, Var("n_ket"), Call(Fn.BLOCK_DIM_Y), ket_body(ctx)),
        *hooks.write_back(ctx),
    ]

    what = ("one batch, no atomics" if head is None
            else f"{head}, the partial-sum form (#203), atomic write-back")
    body: tuple[Stmt, ...] = (
        Note(f"K[a,b] += D[c,d] SUM_p E^ac_p SUM_q [p|q] E^bd_q, K class "
             f"(la,lc|lb,ld) = ({la},{lc}|{lb},{ld}), lp={lp} lq={lq} "
             f"(L={lp + lq}), {what}: milestone 10's staged "
             "contraction (this repo's design; [Fock, §4.3.2] prior art)"),
        *lever_notes(spec),
        Note(SORTED_BY_IQ),
        Note(NO_DENSITY_KEY),
        Note("launch: blockDim = (1, warp width), one warp per bra record"),
        Decl("bra", Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        hooks.shared_decl(ctx),
        If(_c(Fn.LT, bra, Var("n_bra")), tuple(per_bra)),
    )
    kernel = hooks.finish(ctx, Kernel(name, hooks.params(ctx), body))
    # a batch kernel's write made atomic, after `finish` (the `atomize` hook,
    # #345: by default `passes.atomize(kernel, array="K")`, the lane-0 form)
    return kernel if head is None else hooks.atomize(ctx, kernel)


def bra_scalars(ctx: KContext) -> tuple[Stmt, ...]:
    """The bra record's scalars at `ctx.bra`: `P{d}`, `ze`, `Ib`, `sb` and
    `dcb`, the names `ket_body` and the hooks read (#481 shares them)."""
    bra = ctx.bra
    return (
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (bra, Lit(d))))
          for d in range(3)),
        Decl("ze", Scalar.F64, init=Index("zeta_ac", (bra,))),
        Decl("Ib", Scalar.F64, init=Index("Ip", (bra,))),
        Decl("sb", Scalar.F64, init=Index("s_ac", (bra,))),
        Decl("dcb", Scalar.I32, init=Index("dc", (bra,))),
    )


def ket_body(ctx: KContext) -> tuple[Stmt, ...]:
    """One (bra, ket) pair at ket `k`: the screen's `break`, the recurrence,
    `E^bd` and the contraction -- the strip-mined loop's body (#481 shares
    it)."""
    return (
        Decl("bound", Scalar.F64,
             init=_c(Fn.MUL, Var("Ib"), _c(Fn.MUL, ctx.ket("Iq"),
                                           Var("maxD")))),
        Note("screen: Ip * Iq * maxD < tau, and every later ket is under it "
             "too (sorted by Iq) -- so break"),
        If(_c(Fn.LT, Var("bound"), Var("tau")), (Break(),)),
        *ket_quartet(ctx),
    )


def ket_quartet(ctx: KContext) -> tuple[Stmt, ...]:
    """`ket_body` after its screen: the recurrence at ket `k`, `E^bd`, `sq`,
    `dk` and the contraction hook. #583's column form runs it behind a
    warp-uniform screen of its own instead of the per-lane `break`."""
    hooks = ctx.hooks
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in ctx.order if not any(r)])
    return (
        *(Decl(n, Scalar.F64,
               init=_c(Fn.SUB, Var(f"P{d}"), ctx.ket("Q", Lit(d))))
          for d, n in enumerate(PQ_PARAMS)),
        Decl("et", Scalar.F64, init=ctx.ket("eta")),
        Note("theta^2 = zeta eta / (zeta + eta), j_engine.md 'Two defects in "
             "the source'"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ze"), Var("et")),
            _c(Fn.ADD, Var("ze"), Var("et"))))),
        *prelude,
        *recurrence_body(ctx.dag, list(ctx.order), fundamental),
        *hooks.bind_e_bd(ctx),
        Decl("sq", Scalar.F64, init=ctx.ket("s_bd")),
        Note("the row of Dcd: D_cd straddles the records, c on the bra and d "
             "on the ket"),
        Decl("dk", Scalar.I32, init=_c(Fn.ADD, Var("dcb"), ctx.ket("dd"))),
        *hooks.contract(ctx),
    )
