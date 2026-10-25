"""The block exchange kernel split per element, with a density stack (#548).

    build_kblock_split_kernels(cls, granularity, R=1, lanes=32)
                                          -> tuple[Kernel, ...]   the set
    accumulators_per_density(am, granularity)        n_a n_b | n_a | 1
    shared_doubles(am, accumulators, R) / shared_bytes(..., lanes)
    max_stack(am, granularity, lanes)                the largest R that fits
    launch_shape(n_blk, lanes)                       `kernel_kblock`'s

A SIBLING of `kernel_kblock`, the way `kernel_kblock` is a sibling of
`kernel_k`: [Fock, §4.4]'s per-element split on the block mapping. The
arithmetic, the mapping, the block table and both `break`s are
`kernel_kblock`'s, statement for statement (`kernel_k.bra_scalars`,
`kernel_k.ket_body`, the default hooks); what moves is the ACCUMULATOR
EXTENT, through the one hook cut for it, `KHooks.keep` (#341's): each kernel
of the set carries only some `K[a_i, b_j]` of the block, and the set tiles
it. The granularity is a parameter, not a constant (#548, "Granularity"):

    "block"    one kernel, every `K[a, b]`: `n_a n_b` accumulators per
               density -- `kernel_kblock`'s extent, kept as the control
    "column"   one kernel per ket output index `b_j`, `K[:, b_j]`: `n_a`
               accumulators per density. Stage 1 (`I[p]` per `(b, d)` row)
               is NOT recomputed across the set -- each `b_j`'s kernel forms
               only its own rows -- so the recompute is the recurrence and
               the two `E` blocks (#548's ~1.15x at `(dd|dd)`)
    "element"  one kernel per `K[a_i, b_j]`: ONE accumulator per density
               ([Fock, §4.4]; the predecessor's `kgen/emit/cuda_split.py`),
               stage 1 shared by the `n_a` kernels of a `b_j` recomputed
               `n_a` times (~3.9x)

At `(dd|dd)` `n_a = n_b = 6`, so "column" is #548's "split on `b`, 6
accumulators" row. A column is chosen over a row because a row (`K[a_i, :]`)
would recompute every `(b, d)` row's stage 1 in each of its `n_a` kernels.

WHY NOT A LEVER: `kernel_kblock`'s own reason ("WHY NOT A LEVER" there) --
the K levers move `kernel_k`'s skeleton, and this builds on the block
kernel's, which takes none. A spec carrying a lever, or `n_batches != 1`, is
refused, as there; the split is `granularity`, an argument.

THE DENSITY STACK, `R` right-hand sides. Stages 1 and 2 are
density-free and only stage 3 reads `Dcd`, so the `R` densities share one
primitive-quartet sweep: per `(b,d)` row and `(a,c)` row, `V = SUM_p s_ac
E^ac_p I[p]` is formed ONCE and digested into each density's accumulator,
`K[r, a, b] += Dcd[r, dk, (c,d)] V`. `R` is a BUILD constant (it sizes the
shared slots), and both array parameters gain it as a leading extent:

    Dcd[R, n_dc, n_c n_d]       K[R, n_ao, n_ao]   (OUT, ADDED INTO)

so at `R = 1` the memory layout is `kernel_kblock`'s, every other parameter
is `kernel_kblock.PARAMS` unchanged (`PARAMS` here `==` it), and the set
launches on `kernel_kblock`'s arguments. The screen is still one scalar per
block: `dmax[blk]` is the STACK's bound (the elementwise `max |D_r|`), which
the record order -- density-independent ([Fock, §4.3.1]) -- needs no
change for. Forming it is the host's (#548's follow-up, not built here).

SHARED MEMORY, declared from `(am, accumulators, R)` alone:

    shared_doubles = n_p + accumulators * R     per lane, `n_p` the class's
                                                bra Hermite count

`Ksh[n_i + n_K R, BLOCK_DIM_Y]`, the I column then the `K` slots density-
major. A kernel of a split set may hold fewer `I` rows than `n_p` (an
element's `a_i` reaches only some `p`), so the declared figure is the
set's bound, which every kernel's own extent is under (asserted at build).
`SHARED_LIMIT` (48 KiB, the static allocation both cards give one block
without opting in) is refused at build time (`SHARED_REFUSAL`).

THE LAUNCH: every kernel of the set takes the same arguments and
`launch_shape(n_blk, lanes)`, in the returned order, into ONE `K` the caller
zeroes; each adds its own elements, lane 0, one `AtomicAdd` per element per
density. Two kernels of a set never write the same element, and within one
launch no two blocks share an `(A, B)` (`kernel_kblock`'s "THE WRITE"), so
no two atomics meet and the result does not depend on the set's order.

WHAT IS EXACT. Interpreted, a split kernel's elements are `kernel_kblock`'s
BIT FOR BIT: each kept slot sees the same terms in the same order (stage 1's
`(rbd, i)` lists and stage 2's `(rbd, rac)` lists are the block kernel's,
filtered by `keep`, never refolded), and the warp reduction is per slot.
#548's caution that the association differs is for the predecessor's shape
(one scalar per element summed over `(c, d)` then over threads); this one
keeps the block kernel's per-slot fold. A density `r` of the stack is the
same fold with `Dcd[r]`, so the stack's `r`-th `K` is a one-density build
with `Dcd[r]`, bit for bit. On a card the compiler may contract a different
multiply-add, so a device comparison is to rounding.

A column's or an element's kernel reads only some `I[p]`, so only some
recurrence nodes: #341's `levers.split.prune` drops every unread scalar of
the ket loop, to a fixed point, and removes no arithmetic anything reads.

Like `kernel_kblock`, nothing here imports `ao/`, the interpreter,
`reference/`, a backend or a device.
"""

from __future__ import annotations

import dataclasses

from felizjueves.eri.k_exchange.kernel_k import (
    NO_DENSITY_KEY,
    bra_scalars,
    ket_body,
)
from felizjueves.eri.k_exchange.kernel_kblock import (
    PARAMS,
    SORTED_GROUPS,
    kblock_name,
    launch_shape,
    spec_of,
)
from felizjueves.eri.k_exchange.kernel_kblock import params as kb_params
from felizjueves.eri.k_exchange.levers.seams import (
    DEFAULT_HOOKS,
    KET,
    KSH,
    KContext,
    KHooks,
    ncart,
)
from felizjueves.eri.k_exchange.levers.split import prune
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
    Param,
    Scalar,
    Space,
    Stmt,
    Var,
    WarpReduce,
)
from felizjueves.mmd.kernel import builder_plan, split
from felizjueves.spec import Spec

__all__ = [
    "GRANULARITIES",
    "PARAMS",
    "SHARED_LIMIT",
    "SHARED_REFUSAL",
    "accumulators_per_density",
    "build_kblock_split_kernels",
    "launch_shape",
    "max_stack",
    "shared_bytes",
    "shared_doubles",
    "split_name",
]

GRANULARITIES = ("block", "column", "element")

SHARED_LIMIT = 48 * 1024  # bytes: one block's static shared, both cards

SHARED_REFUSAL = (
    "the split kernel's Ksh is (n_p + accumulators * R) * lanes * 8 bytes "
    "per block, declared from (am, granularity, R) alone; over 48 KiB it is "
    "refused, not opted into: 48 KiB is what both cards give one block "
    "without a per-kernel attribute, and the block kernel this replaces "
    "already spends most of it ((dd|dd): 36,352 B at wave64). A finer "
    "granularity or a smaller R is the way under it (#548)")

STACK_NOTE = (
    "the density stack: V = SUM_p s_ac E^ac_p I[p] is density-free, formed "
    "once per (a,c) row and digested into each of the R densities' slots, "
    "K[r,a,b] += Dcd[r,dk,(c,d)] V (#548)")

WRITE_NOTE = (
    "lane 0 ADDS this kernel's elements of the block into K_mn, per density: "
    "the set's kernels hold disjoint elements and one (A, B) per block and "
    "per launch, so no two atomics meet; the caller zeroes K once (#548)")


def accumulators_per_density(am: tuple[int, int, int, int],
                             granularity: str) -> int:
    """`n_a n_b`, `n_a` or 1: the `K` slots one kernel carries per density."""
    la, _, lb, _ = am
    if granularity == "block":
        return ncart(la) * ncart(lb)
    if granularity == "column":
        return ncart(la)
    if granularity == "element":
        return 1
    raise ValueError(f"granularity {granularity!r}: one of {GRANULARITIES}")


def shared_doubles(am: tuple[int, int, int, int], accumulators: int,
                   R: int) -> int:
    """Doubles of `Ksh` per lane: `n_p + accumulators * R` (module
    docstring, "SHARED MEMORY") -- the set's bound, from the class alone."""
    if accumulators < 1 or R < 1:
        raise ValueError(f"accumulators = {accumulators}, R = {R}: both >= 1")
    la, lc, _, _ = am
    lp = la + lc
    return (lp + 1) * (lp + 2) * (lp + 3) // 6 + accumulators * R


def shared_bytes(am: tuple[int, int, int, int], accumulators: int, R: int,
                 lanes: int) -> int:
    return shared_doubles(am, accumulators, R) * lanes * 8


def max_stack(am: tuple[int, int, int, int], granularity: str,
              lanes: int) -> int:
    """The largest `R` whose `shared_bytes` is within `SHARED_LIMIT` (0 when
    not even one density fits)."""
    acc = accumulators_per_density(am, granularity)
    n_p = shared_doubles(am, acc, 1) - acc
    return max(0, (SHARED_LIMIT // (8 * lanes) - n_p) // acc)


def split_name(am: tuple[int, int, int, int], granularity: str, R: int,
               part: tuple[int, ...] = ()) -> str:
    """`kbs_{la}{lc}_{lb}{ld}_{g}{R}[_{i}{j}]`: `g` the granularity's
    initial, `R` the stack, then the element (`element`) or the column
    (`column`) this kernel of the set holds."""
    base = kblock_name(am).replace("kb_", "kbs_", 1)
    tail = "_" + "_".join(str(x) for x in part) if part else ""
    return f"{base}_{granularity[0]}{R}{tail}"


# ---------------------------------------------------------------------------
# The hooks this builder moves: the slots (R per kept K[a, b]) and stage 3
# ---------------------------------------------------------------------------


def _stack(ctx: KContext) -> int:
    return ctx.extra["R"]


def _slot(ctx: KContext, s: int, r: int) -> int:
    """`Ksh` row of `K[a, b]` (flat `s`) for density `r`: density-major."""
    n_i, n_k = len(ctx.i_rows), len(ctx.k_outs)
    return n_i + r * n_k + ctx.k_outs.index(s)


def _shared_decl(ctx: KContext) -> Stmt:
    rows = len(ctx.i_rows) + len(ctx.k_outs) * _stack(ctx)
    return Decl(KSH, Scalar.F64, (Lit(rows), Call(Fn.BLOCK_DIM_Y)),
                Space.SHARED)


def _init_acc(ctx: KContext) -> tuple[Stmt, ...]:
    return (
        Note("accumulate this kernel's K[a,b] in shared memory, per density "
             "one slot per lane, after the lane's I column"),
        *(Assign(ctx.sh(_slot(ctx, s, r)), Lit(0.0))
          for r in range(_stack(ctx)) for s in ctx.k_outs),
    )


def _bra_digest(ctx: KContext, rbd: int) -> tuple[Stmt, ...]:
    """Stages 2 and 3: `seams.bra_digest`'s fold of `V`, then, per density,
    `K[r,a,b] += Dcd[r,dk,(c,d)] V`. At `R = 1` the update is the default's
    expression; above it `V` is a local, read `R` times."""
    R = _stack(ctx)
    ib, id_ = divmod(rbd, ctx.nd)
    out: list[Stmt] = []
    for rac in range(len(ctx.rows_ac)):
        if (rbd, rac) not in ctx.s2:
            continue
        ia, ic = divmod(rac, ctx.nc)
        v: Expr = Lit(0.0)
        for i in ctx.s2[rbd, rac]:
            v = Call(Fn.ADD, (v, Call(Fn.MUL, (Call(Fn.MUL, (
                Var("sb"), ctx.hooks.e_ac(ctx, rac, i))),
                ctx.sh(ctx.islot[i])))))
        if R > 1:
            out.append(Decl(f"V{rac}_{rbd}", Scalar.F64, init=v))
            v = Var(f"V{rac}_{rbd}")
        s = ia * ctx.nb + ib
        for r in range(R):
            slot = ctx.sh(_slot(ctx, s, r))
            dv = Index("Dcd", (Lit(r), Var("dk"), Lit(ic * ctx.nd + id_)))
            out.append(Assign(slot, Call(Fn.ADD, (
                slot, Call(Fn.MUL, (dv, v))))))
    return tuple(out)


def _hooks(granularity: str, part: tuple[int, ...]) -> KHooks:
    if granularity == "block":
        def keep(spec, ia, ib):
            return True
    elif granularity == "column":
        (jb,) = part

        def keep(spec, ia, ib):
            return ib == jb
    else:
        ja, jb = part

        def keep(spec, ia, ib):
            return ia == ja and ib == jb
    return dataclasses.replace(
        DEFAULT_HOOKS, keep=keep, shared_decl=_shared_decl,
        init_acc=_init_acc, bra_digest=_bra_digest)


def _params(ctx: KContext) -> tuple[Param, ...]:
    """`kernel_kblock.PARAMS`, `Dcd` and `K` with the stack's leading `R`."""
    R = Lit(_stack(ctx))
    out = []
    for p in kb_params(ctx):
        if p.name in ("Dcd", "K"):
            p = dataclasses.replace(p, shape=(R, *p.shape))
        out.append(p)
    out = tuple(out)
    assert tuple(p.name for p in out) == PARAMS
    return out


def _write_back(ctx: KContext) -> tuple[Stmt, ...]:
    R = _stack(ctx)
    out: list[Stmt] = [
        Note("reduce this kernel's K[a,b] across the warp, once per block "
             "and density: every lane sees the sum")]
    adds: list[Stmt] = []
    for r in range(R):
        for s in ctx.k_outs:
            n = f"Kab{s}" if R == 1 else f"Kab{r}_{s}"
            out += [Decl(n, Scalar.F64),
                    WarpReduce(Var(n), ctx.sh(_slot(ctx, s, r)))]
            adds.append(AtomicAdd(Index("K", (
                Lit(r), Call(Fn.ADD, (Var("ka"), Lit(s // ctx.nb))),
                Call(Fn.ADD, (Var("kb"), Lit(s % ctx.nb))))), Var(n)))
    out += [Note(WRITE_NOTE),
            If(Call(Fn.LT, (ctx.ty, Lit(1))), tuple(adds))]
    return tuple(out)


def _kernel(spec: Spec, dag, name: str, granularity: str,
            part: tuple[int, ...], R: int) -> Kernel:
    """`kernel_kblock.build_kblock_kernel`'s body, its slots and stage 3
    moved by the hooks above, its write per density."""
    la, lc, lb, ld = spec.am
    lp, lq = split(spec)
    hooks = _hooks(granularity, part)
    ctx = KContext.of(spec, dag, name, None, hooks)
    ctx = dataclasses.replace(ctx, extra={"R": R})
    acc = accumulators_per_density(spec.am, granularity)
    assert len(ctx.k_outs) == acc
    assert (len(ctx.i_rows) + acc * R
            <= shared_doubles(spec.am, acc, R))
    blk = Var("blk")

    per_bra: list[Stmt] = [
        *bra_scalars(ctx),
        Note("bra screen: Ib * Iq[ket0] * maxD < tau bounds every pair of "
             "this bra and of every later one (both groups sorted) -- so "
             "break; uniform across the warp"),
        If(Call(Fn.LT, (Call(Fn.MUL, (Var("Ib"), Call(Fn.MUL, (
            Var("Iq0"), Var("maxD"))))), Var("tau"))), (Break(),)),
        *hooks.bind_e_ac(ctx),
        Note("the block's kets strip-mined over threadIdx.y, stride "
             "blockDim.y"),
        For(KET, Call(Fn.ADD, (Var("k0"), ctx.ty)), Var("k1"),
            Call(Fn.BLOCK_DIM_Y), ket_body(ctx)),
    ]
    per_blk: list[Stmt] = [
        *(Decl(v, Scalar.I32, init=Index(p, (blk,)))
          for v, p in (("b0", "bra0"), ("b1", "bra1"), ("k0", "ket0"),
                       ("k1", "ket1"), ("ka", "ko_a"), ("kb", "ko_b"))),
        Note("the block's density factor, the stack's bound, read once"),
        Decl("maxD", Scalar.F64, init=Index("dmax", (blk,))),
        Decl("Iq0", Scalar.F64, init=Index("Iq", (Var("k0"),))),
        *hooks.init_acc(ctx),
        Note("every lane walks the block's whole bra list"),
        For("bra", Var("b0"), Var("b1"), Lit(1), tuple(per_bra)),
        *_write_back(ctx),
    ]
    if granularity == "block":
        held = "every K[a,b] of the block"
    elif granularity == "column":
        held = f"the column K[:, b_{part[0]}]"
    else:
        held = f"the element K[a_{part[0]}, b_{part[1]}]"
    body: tuple[Stmt, ...] = (
        Note(f"K[r,a,b] += D[r,c,d] SUM_p E^ac_p SUM_q [p|q] E^bd_q, K class "
             f"(la,lc|lb,ld) = ({la},{lc}|{lb},{ld}), lp={lp} lq={lq} "
             f"(L={lp + lq}), one batch: kernel_kblock's mapping split by "
             f"{granularity} ([Fock, §4.4]; #548), this kernel {held}, "
             f"{acc} accumulator(s) per density, a stack of R={R}"),
        Note(SORTED_GROUPS),
        Note(NO_DENSITY_KEY),
        Note(STACK_NOTE),
        Note("launch: gridDim = n_blk, blockDim = (1, warp width)"),
        Decl("blk", Scalar.I32, init=Call(Fn.BLOCK_X)),
        hooks.shared_decl(ctx),
        If(Call(Fn.LT, (blk, Var("n_blk"))), tuple(per_blk)),
    )
    # a column or an element reads only some I[p], so only some recurrence
    # nodes: #341's `prune` drops the unread scalars of the ket loop (an
    # unread local is a warning in generated text, #92), nothing else
    return prune(Kernel(name, _params(ctx), body))


def build_kblock_split_kernels(cls: tuple[int, int, int, int] | Spec,
                               granularity: str, *, R: int = 1,
                               lanes: int = 32) -> tuple[Kernel, ...]:
    """The split kernel set of one K class (module docstring): one kernel for
    `"block"`, `n_b` for `"column"` (in `b_j` order), `n_a n_b` for
    `"element"` (`a_i` outer). `lanes` is the warp width the set will run
    at, read only by the shared-memory refusal."""
    spec = spec_of(cls)
    if not isinstance(R, int) or R < 1:
        raise ValueError(f"R = {R!r}: a stack of at least one density")
    acc = accumulators_per_density(spec.am, granularity)
    need = shared_bytes(spec.am, acc, R, lanes)
    if need > SHARED_LIMIT:
        raise ValueError(
            f"{kblock_name(spec.am)} by {granularity}, R = {R}, {lanes} "
            f"lanes: {need} B of Ksh > {SHARED_LIMIT} B; at most R = "
            f"{max_stack(spec.am, granularity, lanes)}. {SHARED_REFUSAL}")
    (dag,) = builder_plan(spec).dags
    la, _, lb, _ = spec.am
    if granularity == "block":
        parts: list[tuple[int, ...]] = [()]
    elif granularity == "column":
        parts = [(j,) for j in range(ncart(lb))]
    else:
        parts = [(i, j) for i in range(ncart(la)) for j in range(ncart(lb))]
    return tuple(
        _kernel(spec, dag, split_name(spec.am, granularity, R, part),
                granularity, part, R)
        for part in parts)
