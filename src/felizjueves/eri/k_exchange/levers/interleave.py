"""Interleaved `I_p`, the warp arm (#343, milestone 13 lane M3).

The literal K kernel (`seams.contract`) stages the ket contraction per
`(b,d)` row in two passes: stage 1 MATERIALISES the whole `I` family,
`I[p] = SUM_q [p+q]^(0) (s_bd E^bd_q)` for every `p`, into the lane's `Ksh`
column; then stages 2 and 3, per `(a,c)` row, fold `V = SUM_p (s_ac E^ac_p)
I[p]` and digest `K[a,b] += D[c,d] V`. This lever INTERLEAVES the two: per
`(b,d)` row it computes ONE Hermite index's `I_p` into a local scalar,
contracts it into every `(a,c)` row's running `V` it feeds, and discards it
before forming the next `p`; after the last `p` each `V` is digested into
`K[a,b]` exactly as before. The `I` column leaves shared memory (`Ksh` holds
the `K` slots only); what is live instead is ONE `I_p` and one running `V`
per `(a,c)` row the `(b,d)` row reaches -- the trade, recorded in the kernel
as a `Note`.

THE WARP ARM. This is the lever on `kernel_k`'s warp-per-bra mapping; the
same lever on #341's per-element split kernels is #348 (the split arm). The
split arm is NOT a lever of its own: it is the COMPOSITION `("interleave",
"split")` in one spec (key `.../k/interleave+split`), which `hooks_for`
builds by applying this lever and then `split`. The one thing the
composition adds is a `Note`, `SPLIT_ARM`, after `TRADE` in each element
kernel's region (`arm(spec)` says which arm a spec is); the arithmetic is
this lever's region on `split`'s narrowed context, unchanged. The
predecessor measured this arm as a REGRESSION (spill +79% at `(dp|dp)`,
+58% at `(21|21)`) and kept it pinned as a falsified hypothesis; reproducing
that is this issue's expected outcome, and the lever is not tuned toward a
win.

THE ARITHMETIC IS THE LITERAL KERNEL'S. Each `I_p` is the literal stage-1
left fold over `q` from `+0.0`, term for term; each `V` is the literal
stage-2 fold over `p` from `+0.0` in ascending `p` (the `s2` list's order),
now accumulated across the `p` loop instead of in one expression; and the
digest `K[a,b] += D V` is the literal one, in the literal `(b,d)`-row order.
No operation is reassociated or distributed, so the kernel's `K` is the
literal kernel's BIT FOR BIT (test/test_k_lever_interleave.py, against
`ao.k.k_quartet` and the literal kernel). The interleaved order is per
`(b,d)` row: the `D V` digest is NOT moved inside the `p` loop, which would
distribute `D` over the `p` sum and change the rounding.

THE HOOKS IT MOVES: `contract` (replaced: the interleaved region, built from
`ctx.s1` / `ctx.s2` and reading `E` through `ctx.hooks.e_ac` / `e_bd`, so a
lever narrowing or staging `E` is seen), and `shared_decl`, `init_acc`,
`write_back` (WRAPPED: each calls the hook it was handed with the context
re-slotted by `k_only`, so `Ksh` is the `K` slots only). It moves neither
`keep` nor `fan_out` nor `finish`, so #341's `split` composes with it by
construction in either application order (#348): the split keeps one
element, `KContext.of` narrows `s1` / `s2` and `k_outs`, and this lever's
region and slots follow them.

THE INTERFACE (#348 builds on it): `NAME`, `TRADE`, `SPLIT_ARM`,
`arm(spec)`, `k_only(ctx)`, `contract(ctx)` and `apply(hooks, spec)`. Like
the seams, nothing here imports `ao/`, the interpreter, `reference/`, a
backend or a device (`split`, a sibling lever, for its `NAME` only).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from felizjueves.eri.k_exchange.levers import split
from felizjueves.eri.k_exchange.levers.seams import KContext, KHooks, plus
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Fn,
    Index,
    Lit,
    Note,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.mmd.recurrence import node_name
from felizjueves.spec import Spec

__all__ = ["NAME", "SPLIT", "SPLIT_ARM", "TRADE", "WARP", "apply", "arm",
           "contract", "i_name", "k_only", "v_name"]

NAME = "interleave"

TRADE = (
    "interleaved I_p, warp arm (#343): per (b,d) row ONE Hermite index's "
    "I_p = SUM_q [p+q]^(0) s_bd E^bd_q is formed in a register, contracted "
    "into every (a,c) row's running V = SUM_p s_ac E^ac_p I_p, and "
    "discarded before the next p; then K[a,b] += D[c,d] V per (a,c) row, as "
    "the literal kernel digests. The I column leaves shared memory; one I_p "
    "and one running V per (a,c) row are live instead (the literal kernel's "
    "folds, in its order: bitwise the same K)")


WARP, SPLIT = "warp", "split"  # the two arms `arm(spec)` names

SPLIT_ARM = (
    "split arm (#348): the interleave above on ONE element kernel of #341's "
    "split -- not a lever of its own, the composition interleave+split. "
    "The split cuts the accumulator set to K[a_i,b_j]'s one slot; the "
    "interleave cuts the live I set to one I_p; only the (b,d) rows and "
    "I_p this element reads are formed (the literal folds: bitwise the "
    "same K)")


def arm(spec: Spec) -> str:
    """`SPLIT` when `spec` also carries #341's `split` (the element kernels,
    #348), else `WARP` (#343's warp-per-bra kernel)."""
    return SPLIT if split.NAME in spec.kernel_body_levers() else WARP


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def i_name(rbd: int, i: int) -> str:
    """The local holding `I_p`, `p = ps[i]`, of `(b,d)` row `rbd`."""
    return f"I{rbd}_{i}"


def v_name(rbd: int, rac: int) -> str:
    """The local running `V` of `(b,d)` row `rbd` and `(a,c)` row `rac`."""
    return f"V{rbd}_{rac}"


def k_only(ctx: KContext) -> KContext:
    """`ctx` with no `I` rows in `Ksh`: the `K` slots renumbered from 0, so
    the default `shared_decl` / `init_acc` / `write_back` lay out, zero and
    reduce the `K` accumulators alone."""
    return dataclasses.replace(
        ctx, i_rows=(), islot={},
        kslot={s: n for n, s in enumerate(ctx.k_outs)})


def contract(ctx: KContext) -> tuple[Stmt, ...]:
    """The interleaved region: per `(b,d)` row (`b` outer, as the literal),
    the running `V`s from `+0.0`, then per `p` its `I_p` and every `V` it
    feeds, then the `D V` digest per `(a,c)` row."""
    e_ac, e_bd = ctx.hooks.e_ac, ctx.hooks.e_bd
    slots = k_only(ctx)
    out: list[Stmt] = [Note(TRADE)]
    if arm(ctx.spec) == SPLIT:
        out.append(Note(SPLIT_ARM))
    for rbd in range(len(ctx.rows_bd)):
        racs = [rac for rac in range(len(ctx.rows_ac))
                if (rbd, rac) in ctx.s2]
        if not racs:
            continue  # this batch contracts nothing through the row
        ib, id_ = divmod(rbd, ctx.nd)
        out += [Decl(v_name(rbd, rac), Scalar.F64, init=Lit(0.0))
                for rac in racs]
        for i, p in enumerate(ctx.ps):
            if (rbd, i) not in ctx.s1:
                continue  # I_p is structurally zero (or unread): not formed
            acc: Expr = Lit(0.0)
            for j, q in ctx.s1[rbd, i]:
                term = _c(Fn.MUL, Var(node_name((plus(p, q), 0))), _c(
                    Fn.MUL, Var("sq"), e_bd(ctx, rbd, j)))
                acc = _c(Fn.SUB if sum(q) % 2 else Fn.ADD, acc, term)
            out.append(Decl(i_name(rbd, i), Scalar.F64, init=acc))
            for rac in racs:
                if i not in ctx.s2[rbd, rac]:
                    continue  # E^ac_p is structurally zero for this row
                v = Var(v_name(rbd, rac))
                out.append(Assign(v, _c(Fn.ADD, v, _c(Fn.MUL, _c(
                    Fn.MUL, Var("sb"), e_ac(ctx, rac, i)),
                    Var(i_name(rbd, i))))))
        for rac in racs:
            ia, ic = divmod(rac, ctx.nc)
            slot = slots.sh(slots.kslot[ia * ctx.nb + ib])
            dv = Index("Dcd", (Var("dk"), Lit(ic * ctx.nd + id_)))
            out.append(Assign(slot, _c(
                Fn.ADD, slot, _c(Fn.MUL, dv, Var(v_name(rbd, rac))))))
    return tuple(out)


def _reslotted(prior: Callable[[KContext], object]
               ) -> Callable[[KContext], object]:
    def hook(ctx: KContext):
        return prior(k_only(ctx))
    hook.__name__ = getattr(prior, "__name__", "hook")
    return hook


def apply(hooks: KHooks, spec: Spec) -> KHooks:
    """`hooks` with the interleaved contraction and the `K`-only `Ksh`."""
    return dataclasses.replace(
        hooks, contract=contract,
        shared_decl=_reslotted(hooks.shared_decl),
        init_acc=_reslotted(hooks.init_acc),
        write_back=_reslotted(hooks.write_back))
