"""The K kernel's lever seams: `KHooks`, `KContext` and the frozen defaults (#367).

`kernel_k._k_kernel` is a SKELETON: it lays out the parts of the exchange
kernel that no milestone-13 lever moves (the head notes, the bra index, the
bra-record scalars, the screen, the recurrence region, the ket loop) and asks
a `KHooks` for every part a lever does move. Each hook's default below is the
code `kernel_k.py` held before #367, moved here statement for statement, so
the literal kernel is byte-identical (test/test_k_seams.py digests every K
class through L = 4, both backends, against the pre-#367 tree's digest).

THE HOOKS, and the lever each was cut for (milestone 13, lane SK):

    keep(spec, ia, ib)        accumulator slicing: which K[a_i, b_j] a kernel
                              carries (#341, the per-element split)
    fan_out(spec, hooks)      one kernel -> several, `(suffix, hooks)` each
                              (#341: na*nb element kernels per batch)
    shared_decl(ctx)          the one `Space.SHARED` array and its extent
                              (#342's staged E^ac, #345's broadcast buffer)
    bind_e_ac(ctx), e_ac(ctx, rac, i)
                              the E^ac binding and every read of it (#342:
                              staged in shared; #347: narrowed)
    bind_e_bd(ctx), e_bd(ctx, rbd, j)
                              the same, E^bd (#347)
    contract(ctx)             the whole per-ket contraction region: the note,
                              then per (b,d) row `ket_stage` + `bra_digest`
                              (#343, #348: interleaved I_p replaces it)
    ket_stage(ctx, rbd)       stage 1, I[p] into the shared column
    bra_digest(ctx, rbd)      stages 2 and 3 fused, K[a,b] += D V
    init_acc(ctx)             the K accumulator's zeroing
    write_back(ctx)           the warp reduction and the K write (#345's
                              coalesced shared broadcast, #349's shuffle)
    params(ctx)               the parameter list (#347's narrowed D and E)
    finish(ctx, kernel)       a last rewrite of the built `Kernel` (#347's
                              precision, or anything the above cannot reach)
    atomize(ctx, kernel)      a BATCH kernel's write made atomic, after
                              `finish` (the partial-sum form, #203: default
                              `passes.atomize(kernel, array="K")`, the lane-0
                              form only; #345's lane-parallel atomics). Never
                              called for the one-batch kernel

A hook receives the `KContext`, which carries everything the skeleton derived
from `(spec, dag)` -- the term lists `s1` / `s2`, the shared slots, the
fragments -- and `ctx.hooks`, so a hook calls its siblings through the
composed set, never through the defaults directly (an `e_ac` override is seen
by the default `bra_digest`).

A LEVER IS A MODULE: `felizjueves/eri/k_exchange/levers/<name>.py` exposing
`apply(hooks, spec) -> KHooks` (`dataclasses.replace` of the hooks it moves),
registered by name in `k_levers.REGISTRY`. `kernel_k` resolves a spec's
kernel-body levers through that registry (`k_levers.hooks_for`) and never
changes for a new lever.

Like `kernel_k`, nothing here imports `ao/`, the interpreter, `reference/`, a
backend or a device; the `Kernel` it helps build is arch-neutral.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
    If,
    Index,
    Inline,
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
from felizjueves.ir.passes import atomize as _atomize
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.recurrence import node_name
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Spec

__all__ = [
    "DEFAULT_HOOKS",
    "EmptyKernel",
    "KET",
    "KSH",
    "PARAMS",
    "SPLIT_WRITE",
    "KContext",
    "KHooks",
    "absent",
    "ncart",
    "plus",
]

KET = "k"  # the strip-mined ket loop's variable, as in kernel.py
KSH = "Ksh"  # the shared I column and K accumulator, one array

PARAMS = ("n_bra", "n_ket", "n_dc", "tau", "maxD",
          "P", "zeta_ac", "XPA", "XPC", "s_ac", "Ip", "dc",
          "Q", "eta", "XQB", "XQD", "s_bd", "Iq", "dd",
          "Dcd", "K")

SPLIT_WRITE = (
    "atomic write, by lane 0 only: the partial-sum form (#203). A batch "
    "holds only some targets r = p + q, so it completes stage 1 for only "
    "some p, while stage 2 sums over ALL p: every batch kernel of the class "
    "adds its partial K[a,b] to the same K[a,b], so the write must "
    "accumulate (passes.atomize, as a split J kernel writes J_p, [JEngine, "
    "v2, §5]). The lane-0 guard stays: every lane holds the warp sum")


def ncart(l: int) -> int:  # noqa: E741
    return (l + 1) * (l + 2) // 2


def absent(t: tuple[int, int, int], x: tuple[int, int, int],
           y: tuple[int, int, int]) -> bool:
    """`E^{xy}_t` is structurally zero: `t_d > x_d + y_d` on some axis."""
    return any(t[d] > x[d] + y[d] for d in range(3))


def plus(p: tuple[int, int, int], q: tuple[int, int, int]
         ) -> tuple[int, int, int]:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def bindings(frag, zeta: Expr, pa: str, pb: str, at: Expr
             ) -> dict[str, Expr]:
    """One fragment's binds: `zeta`, `1/2zeta`, `PA_d`, `PB_d`, the array."""
    out: dict[str, Expr] = {ef.ZETA: zeta}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), zeta)
    for d in range(3):
        out[Factor.pa(d).name_in_kernel] = Index(pa, (at, Lit(d)))
        out[Factor.pb(d).name_in_kernel] = Index(pb, (at, Lit(d)))
    return out


# ---------------------------------------------------------------------------
# The context: what the skeleton derives from (spec, dag), once
# ---------------------------------------------------------------------------


class EmptyKernel(ValueError):
    """A kernel that would contract nothing. The literal builder refuses it
    (a `ValueError`, as before #367); a fan-out's variant that a batch never
    touches (e.g. one `K[a, b]` no term of the batch reaches) is DROPPED by
    `kernel_k.build_k_kernels` instead -- no kernel, not an empty one."""


@dataclass(frozen=True)
class KContext:
    """One kernel's derived quantities, shared by every hook.

    `s1[(rbd, i)]` is stage 1's `(j, q)` list for `(b,d)` row `rbd` and
    `p = ps[i]`; `s2[(rbd, rac)]` is stage 2's `i` list for that pair of
    rows; `islot` / `kslot` map a `p` index / a flat `a n_b + b` to its
    `Ksh` row; `k_outs` are the flat `K[a, b]` the kernel writes. `head` is
    None for the one-batch kernel, else the batch's label (`kernel_k`).
    """

    spec: Spec
    dag: object
    name: str
    head: str | None
    hooks: KHooks
    order: tuple
    ps: tuple
    qs: tuple
    rows_ac: tuple
    rows_bd: tuple
    frag_ac: object
    frag_bd: object
    s1: dict
    s2: dict
    i_rows: tuple[int, ...]
    k_outs: tuple[int, ...]
    islot: dict
    kslot: dict
    extra: dict = field(default_factory=dict)  # a lever's own derived data

    @classmethod
    def of(cls, spec: Spec, dag, name: str, head: str | None,
           hooks: KHooks) -> KContext:
        """The term lists and the slots: what this batch contracts, term by
        term, restricted to the `K[a, b]` `hooks.keep` keeps.

        Stage 1: per (b,d) row and p, the q with E^bd_q structurally nonzero
        AND p + q one of the batch's targets. Stage 2: per (b,d) row and
        (a,c) row, the p whose I[p] this batch holds for that row AND whose
        E^ac_p is structurally nonzero. Absent terms are absent, not added
        zeros; with one batch every list is the whole class's (q = 0 and
        p = 0 are never absent). A stage-1 entry no kept stage-2 entry
        reads is dropped: with every `K[a, b]` kept there is none (every
        `p` of `hermite_targets(la + lc)` is nonzero for some `(a, c)` row).
        """
        la, lc, lb, ld = spec.am
        lp, lq = la + lc, lb + ld
        ps, qs = hermite_targets(lp), hermite_targets(lq)
        rows_ac, rows_bd = ef.block_rows(la, lc), ef.block_rows(lb, ld)
        nc, nb, nd = ncart(lc), ncart(lb), ncart(ld)
        held = {r for (r, _) in dag.targets}

        s1 = {(rbd, i): [(j, q) for j, q in enumerate(qs)
                         if not absent(q, b, d) and plus(p, q) in held]
              for rbd, (b, d) in enumerate(rows_bd) for i, p in enumerate(ps)}
        s1 = {key: v for key, v in s1.items() if v}
        s2 = {(rbd, rac): [i for i, p in enumerate(ps)
                           if (rbd, i) in s1 and not absent(p, a, c)]
              for rbd in range(len(rows_bd))
              for rac, (a, c) in enumerate(rows_ac)
              if hooks.keep(spec, rac // nc, rbd // nd)}
        s2 = {key: v for key, v in s2.items() if v}
        if not s2:
            raise EmptyKernel(
                f"batch {dag.batch} of {spec.key()} contracts nothing")
        read = {(rbd, i) for (rbd, _), v in s2.items() for i in v}
        s1 = {key: v for key, v in s1.items() if key in read}
        i_rows = tuple(sorted({i for _, i in s1}))  # the p held: I's rows
        k_outs = tuple(sorted({divmod(rac, nc)[0] * nb + divmod(rbd, nd)[0]
                               for rbd, rac in s2}))  # the K[a,b] it writes
        n_i = len(i_rows)
        return cls(
            spec=spec, dag=dag, name=name, head=head, hooks=hooks,
            order=tuple(schedule(dag)), ps=tuple(ps), qs=tuple(qs),
            rows_ac=tuple(rows_ac), rows_bd=tuple(rows_bd),
            frag_ac=ef.block_fragment(la, lc), frag_bd=ef.block_fragment(lb, ld),
            s1=s1, s2=s2, i_rows=i_rows, k_outs=k_outs,
            islot={i: n for n, i in enumerate(i_rows)},
            kslot={s: n_i + n for n, s in enumerate(k_outs)})

    # --- the class's shapes -------------------------------------------------
    @property
    def nc(self) -> int:
        return ncart(self.spec.am[1])

    @property
    def nb(self) -> int:
        return ncart(self.spec.am[2])

    @property
    def nd(self) -> int:
        return ncart(self.spec.am[3])

    @property
    def n_ab(self) -> int:
        return ncart(self.spec.am[0]) * self.nb

    @property
    def n_shared(self) -> int:
        """`Ksh` rows the default layout uses: the I column, then K."""
        return len(self.i_rows) + len(self.k_outs)

    # --- the names every part of the kernel shares --------------------------
    @property
    def ty(self) -> Call:
        return Call(Fn.THREAD_Y)

    @property
    def bra(self) -> Var:
        return Var("bra")

    def sh(self, row: int) -> Index:
        """The lane's `Ksh` slot `row`."""
        return Index(KSH, (Lit(row), self.ty))

    def ket(self, arr: str, *more: Expr) -> Index:
        """`arr[k, ...]`, a ket-side array at the loop's ket record."""
        return Index(arr, (Var(KET), *more))


# ---------------------------------------------------------------------------
# The frozen defaults: kernel_k.py's code before #367, statement for statement
# ---------------------------------------------------------------------------


def keep_all(spec: Spec, ia: int, ib: int) -> bool:
    """Every `K[a, b]` of the block: the warp-per-bra kernel's one mapping."""
    return True


def one_kernel(spec: Spec, hooks: KHooks) -> tuple[tuple[str, KHooks], ...]:
    """No fan-out: one kernel per batch, its name unsuffixed."""
    return (("", hooks),)


def shared_decl(ctx: KContext) -> Stmt:
    """`Ksh[n_i + n_K, BLOCK_DIM_Y]`: the I column, then the K slots."""
    return Decl(KSH, Scalar.F64, (Lit(ctx.n_shared), Call(Fn.BLOCK_DIM_Y)),
                Space.SHARED)


def bind_e_ac(ctx: KContext) -> tuple[Stmt, ...]:
    """The local `Eac[n_ac, n_p]`, from the fragment inlined once per bra."""
    return (
        Note("the undressed E^ac block of this bra record, once: "
             "e_fragment.block_fragment(la, lc), inlined; s_ac is the host's"),
        Decl("Eac", Scalar.F64, (Lit(len(ctx.rows_ac)), Lit(len(ctx.ps)))),
        Inline(ctx.frag_ac, {**bindings(ctx.frag_ac, Var("ze"), "XPA", "XPC",
                                        ctx.bra),
                             ef.E_ARRAY: Var("Eac")}),
    )


def e_ac(ctx: KContext, rac: int, i: int) -> Expr:
    return Index("Eac", (Lit(rac), Lit(i)))


def bind_e_bd(ctx: KContext) -> tuple[Stmt, ...]:
    """The local `Ebd[n_bd, n_q]`, from the fragment inlined once per ket."""
    return (
        Note("the undressed E^bd block of this ket record: "
             "e_fragment.block_fragment(lb, ld), inlined; s_bd is the host's"),
        Decl("Ebd", Scalar.F64, (Lit(len(ctx.rows_bd)), Lit(len(ctx.qs)))),
        Inline(ctx.frag_bd, {**bindings(ctx.frag_bd, Var("et"), "XQB", "XQD",
                                        Var(KET)),
                             ef.E_ARRAY: Var("Ebd")}),
    )


def e_bd(ctx: KContext, rbd: int, j: int) -> Expr:
    return Index("Ebd", (Lit(rbd), Lit(j)))


def contract(ctx: KContext) -> tuple[Stmt, ...]:
    """The note, then per `(b,d)` row (`b` outer): stage 1, then 2 and 3."""
    h = ctx.hooks
    out: list[Stmt] = [
        Note("per (b,d) row: stage 1, I[p] = SUM_q [p|q] E^bd_q into the "
             "shared column; then stages 2 and 3 FUSED, K[a,b] += D[c,d] "
             "SUM_p E^ac_p I[p] -- V is never an array (ao.k.k_quartet's "
             "order; absent E entries are structural zeros)"),
    ]
    for rbd in range(len(ctx.rows_bd)):
        out += h.ket_stage(ctx, rbd)
        out += h.bra_digest(ctx, rbd)
    return tuple(out)


def ket_stage(ctx: KContext, rbd: int) -> tuple[Stmt, ...]:
    """Stage 1: `I[p]`, a left fold over `q` from `+0.0`, into `Ksh`."""
    out: list[Stmt] = []
    for i, p in enumerate(ctx.ps):
        if (rbd, i) not in ctx.s1:
            continue  # I[p] is structurally zero in this batch: no slot
        acc: Expr = Lit(0.0)
        for j, q in ctx.s1[rbd, i]:
            term = _c(Fn.MUL, Var(node_name((plus(p, q), 0))), _c(
                Fn.MUL, Var("sq"), ctx.hooks.e_bd(ctx, rbd, j)))
            acc = _c(Fn.SUB if sum(q) % 2 else Fn.ADD, acc, term)
        out.append(Assign(ctx.sh(ctx.islot[i]), acc))
    return tuple(out)


def bra_digest(ctx: KContext, rbd: int) -> tuple[Stmt, ...]:
    """Stages 2 and 3 fused: `K[a,b] += D[c,d] * SUM_p (s_ac E^ac_p) I[p]`."""
    ib, id_ = divmod(rbd, ctx.nd)
    out: list[Stmt] = []
    for rac in range(len(ctx.rows_ac)):
        if (rbd, rac) not in ctx.s2:
            continue  # V(a,c) is structurally zero in this batch
        ia, ic = divmod(rac, ctx.nc)
        v: Expr = Lit(0.0)
        for i in ctx.s2[rbd, rac]:
            v = _c(Fn.ADD, v, _c(Fn.MUL, _c(
                Fn.MUL, Var("sb"), ctx.hooks.e_ac(ctx, rac, i)),
                ctx.sh(ctx.islot[i])))
        slot = ctx.kslot[ia * ctx.nb + ib]
        dv = Index("Dcd", (Var("dk"), Lit(ic * ctx.nd + id_)))
        out.append(Assign(ctx.sh(slot), _c(
            Fn.ADD, ctx.sh(slot), _c(Fn.MUL, dv, v))))
    return tuple(out)


def init_acc(ctx: KContext) -> tuple[Stmt, ...]:
    """Zero the lane's `K[a, b]` slots, after its I column."""
    return (
        Note("accumulate K[a,b] in shared memory, one slot per lane, after "
             "the lane's I column"),
        *(Assign(ctx.sh(ctx.kslot[s]), Lit(0.0)) for s in ctx.k_outs),
    )


def write_back(ctx: KContext) -> tuple[Stmt, ...]:
    """One `WarpReduce` per `K[a, b]`, then lane 0 writes each, serially."""
    out: list[Stmt] = [
        Note("reduce K[a,b] across the warp: every lane sees the sum")]
    for s in ctx.k_outs:
        out += [Decl(f"Kab{s}", Scalar.F64),
                WarpReduce(Var(f"Kab{s}"), ctx.sh(ctx.kslot[s]))]
    out += [
        Note("plain write, NOT atomic: one batch, one warp per bra record, so "
             "this kernel is the only writer of its row of K"
             if ctx.head is None else SPLIT_WRITE),
        If(_c(Fn.LT, ctx.ty, Lit(1)), tuple(
            Assign(Index("K", (ctx.bra, Lit(s))), Var(f"Kab{s}"))
            for s in ctx.k_outs)),
    ]
    return tuple(out)


def params(ctx: KContext) -> tuple[Param, ...]:
    """`PARAMS`, typed and shaped (`kernel_k`'s module docstring)."""
    nbv, nkv, ndc = Var("n_bra"), Var("n_ket"), Var("n_dc")

    def f64(n: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(n, Scalar.F64, array=True, intent=intent, shape=shape)

    def i32(n: str, *shape: Expr) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    out = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("n_dc", Scalar.I32),
        Param("tau", Scalar.F64), Param("maxD", Scalar.F64),
        f64("P", nbv, Lit(3)), f64("zeta_ac", nbv),
        f64("XPA", nbv, Lit(3)), f64("XPC", nbv, Lit(3)),
        f64("s_ac", nbv), f64("Ip", nbv), i32("dc", nbv),
        f64("Q", nkv, Lit(3)), f64("eta", nkv),
        f64("XQB", nkv, Lit(3)), f64("XQD", nkv, Lit(3)),
        f64("s_bd", nkv), f64("Iq", nkv), i32("dd", nkv),
        f64("Dcd", ndc, Lit(ctx.nc * ctx.nd)),
        f64("K", nbv, Lit(ctx.n_ab), intent=Intent.OUT),
    )
    assert tuple(p.name for p in out) == PARAMS
    return out


def finish(ctx: KContext, kernel: Kernel) -> Kernel:
    """The built kernel, unchanged."""
    return kernel


def atomize(ctx: KContext, kernel: Kernel) -> Kernel:
    """A batch kernel's lane-0 `K` write made atomic: `passes.atomize`, the
    call `kernel_k.build_k_kernels` made before #345 (SPLIT_WRITE)."""
    return _atomize(kernel, array="K")


@dataclass(frozen=True)
class KHooks:
    """Every point a K lever may intervene, each defaulting to the frozen
    path (module docstring). Compose with `dataclasses.replace`."""

    keep: Callable[[Spec, int, int], bool] = keep_all
    fan_out: Callable[[Spec, KHooks], tuple[tuple[str, KHooks], ...]] = (
        one_kernel)
    shared_decl: Callable[[KContext], Stmt] = shared_decl
    bind_e_ac: Callable[[KContext], tuple[Stmt, ...]] = bind_e_ac
    e_ac: Callable[[KContext, int, int], Expr] = e_ac
    bind_e_bd: Callable[[KContext], tuple[Stmt, ...]] = bind_e_bd
    e_bd: Callable[[KContext, int, int], Expr] = e_bd
    contract: Callable[[KContext], tuple[Stmt, ...]] = contract
    ket_stage: Callable[[KContext, int], tuple[Stmt, ...]] = ket_stage
    bra_digest: Callable[[KContext, int], tuple[Stmt, ...]] = bra_digest
    init_acc: Callable[[KContext], tuple[Stmt, ...]] = init_acc
    write_back: Callable[[KContext], tuple[Stmt, ...]] = write_back
    params: Callable[[KContext], tuple[Param, ...]] = params
    finish: Callable[[KContext, Kernel], Kernel] = finish
    atomize: Callable[[KContext, Kernel], Kernel] = atomize

    def is_default(self, hook: str) -> bool:
        """True when `hook` is still the frozen path's."""
        return getattr(self, hook) is getattr(DEFAULT_HOOKS, hook)


DEFAULT_HOOKS = KHooks()
