"""The register write-back for small K blocks (#349, milestone 13 lane M7).

`kernel_k`'s literal write-back (`seams.write_back`) reduces each `K[a, b]`
across the warp into a register `Kab{s}` and stores the block from LANE 0,
one store after another. #345's `coalesced` lever stores it lane-parallel,
but to hold `K[a, b]` at an INDEXABLE place it first writes every reduced
value back into the lane's own `Ksh` column (`n_a n_b` in-place shared
stores) and reads one back per store. This lever drops those shared stores
for a block no wider than the warp: the reduced values stay in registers,
and the lane that OWNS element `o` (lane `o`) picks `Kab{o}` out of them by
a flat ladder of predicated selects, then the warp issues its `n_a n_b`
stores (or atomics) as ONE lane-parallel instruction.

WHY NO `LaneBroadcast` IS EMITTED, though this issue adds the node. The
predecessor's form was "reduce, then broadcast each `K[a, b]` from lane 0 to
its owner lane over a shuffle": its reduction left the sum in lane 0 only.
FJ's `WarpReduce` is the xor butterfly, and every lane ends with the sum
(`ir.WarpReduce`, `interp._butterfly`) -- the broadcast is already done by
the reduction. A `LaneBroadcast(Kab{s}, lane 0)` would move a value the
owner already holds, one extra shuffle per element, and change nothing.
What the owner still needs is to SELECT its element among registers whose
names are fixed at build time; a shuffle selects among lanes, not among a
lane's registers, so the select is the ladder below. The node itself
(`ir.LaneBroadcast`, its interpreter semantics and both spellings) lands
here as the issue's acceptance asks, for a consumer that reduces to one lane.

THE FOOTPRINT, HONESTLY. The issue predicted a shared-memory saving (the
predecessor's 32-288 B per-warp broadcast buffer). Against `coalesced` that
saving is ZERO: `coalesced` already reuses the accumulator slots and adds no
shared memory (`coalesced.EXTRA_SHARED_BYTES == 0`), and so does this lever
(`EXTRA_SHARED_BYTES == 0`, the `Ksh` declaration is the literal one). What
it drops is `n_a n_b` shared STORES and as many shared loads; whether that
moves registers or spills is a compiler observation (the PR body), not a
claim made here.

ELIGIBILITY IS THE WARP'S. The ladder is a single round when `n_a n_b <=`
the warp width, and `ceil(n_a n_b / W)` rounds otherwise (each round re-runs
the reductions), so a block wider than the warp falls back to `coalesced`'s
write-back, statement for statement (`write_back_at`). The IR does not know
the width (`ir/__init__.py`), and a K lever may not import `device` (test_layering's
`k_lever` rule), so the width it builds for is `MIN_WAVEFRONT`, the
narrowest wavefront the kernel is emitted for -- 32 on both cards
(`device.ARCHS`, pinned by a test) -- and `write_back_at(64)` is the wave64
form: a 36-element block such as `(d s|d s)` falls back at 32 and is
eligible at 64. Either form is CORRECT at any runtime width: the round loop
strides by `blockDim.y`, so the interpreter's 1- and 4-lane warps take more
rounds and the same bits.

THE ARITHMETIC IS THE LITERAL KERNEL'S: the same `WarpReduce` of the same
`Ksh` slot into the same `Kab{s}`, copied unchanged into `K` -- bit for bit
the literal kernel's `K` at any lane count, and with one lane
`eri.k_exchange.k.k_quartet`'s (test/test_k_lever_shuffle.py).

THE SPLIT CLASS (`n_batches > 1`): the owner lane's store is an `AtomicAdd`,
each element added by exactly one lane, so no lane-0 guard is needed (as in
`coalesced`), and `atomize` is `coalesced`'s: it accepts a kernel whose `K`
writes are already all atomic and hands any other to the prior `atomize`.

THE HOOKS IT MOVES: `write_back` and `atomize`. Nothing here imports `ao/`,
the interpreter, `reference/`, a backend or a device.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from felizjueves.eri.k_exchange.levers import coalesced
from felizjueves.eri.k_exchange.levers.seams import KContext, KHooks
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Expr,
    Fn,
    For,
    If,
    Index,
    Lit,
    Note,
    Scalar,
    Stmt,
    Var,
    WarpReduce,
)
from felizjueves.spec import Spec

__all__ = ["ATOMIC_WRITE", "COL_VAR", "EXTRA_SHARED_BYTES", "MINE_VAR",
           "MIN_WAVEFRONT", "NAME", "OWN_VAR", "PLAIN_WRITE", "ROUND_VAR",
           "SELECT_NOTE", "apply", "eligible", "write_back", "write_back_at"]

NAME = "shuffle"

MIN_WAVEFRONT = 32
"""The narrowest wavefront the K kernel is emitted for: `min(a.wavefront for
a in device.ARCHS)` (a test pins it; the lever cannot import `device`)."""

EXTRA_SHARED_BYTES = 0
"""Shared memory added to the literal `Ksh`: none, as for `coalesced`."""

ROUND_VAR = "kr"   # the round loop's variable
OWN_VAR = "kown"   # the element this lane owns in this round
MINE_VAR = "kmine"  # its reduced value
COL_VAR = "kcol"   # its K column, when the written elements are not one run

SELECT_NOTE = (
    "register write-back (#349): each K[a,b] is reduced across the warp into "
    "a register (every lane sees the sum, so no broadcast is needed) and the "
    "lane that owns element o = threadIdx.y + round selects Kab[o] by a "
    "ladder of predicated selects -- no in-place Ksh stores, no extra shared "
    "memory")

PLAIN_WRITE = (
    "plain write, NOT atomic: one batch, one warp per bra record, so this "
    "kernel is the only writer of its row of K. Lane o stores element o: the "
    "warp's stores in one instruction, not lane 0's one after another")

ATOMIC_WRITE = (
    "atomic write, lane-parallel: the partial-sum form (#203), each element "
    "added by exactly ONE lane (its owner), so no lane-0 guard is needed")


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def eligible(n_writes: int, wavefront: int) -> bool:
    """One round of selects covers the block: `n_writes <= wavefront`."""
    if n_writes < 1 or wavefront < 1:
        raise ValueError(f"eligible({n_writes}, {wavefront}): both must be "
                         "positive")
    return n_writes <= wavefront


def _contiguous(outs: tuple[int, ...]) -> bool:
    return outs == tuple(range(outs[0], outs[0] + len(outs)))


def _register_write_back(ctx: KContext) -> tuple[Stmt, ...]:
    outs = ctx.k_outs
    n, batch = len(outs), ctx.head is not None
    own, mine, col = Var(OWN_VAR), Var(MINE_VAR), Var(COL_VAR)
    run = _contiguous(outs)
    body: list[Stmt] = [Decl(OWN_VAR, Scalar.I32,
                             init=_c(Fn.ADD, ctx.ty, Var(ROUND_VAR))),
                        Decl(MINE_VAR, Scalar.F64)]
    if not run:
        body.append(Decl(COL_VAR, Scalar.I32))
    for i, s in enumerate(outs):
        kab = Var(f"Kab{s}")
        pick: tuple[Stmt, ...] = (Assign(mine, kab),) + (
            () if run else (Assign(col, Lit(s)),))
        body += [Decl(kab.name, Scalar.F64),
                 WarpReduce(kab, ctx.sh(ctx.kslot[s]))]
        # i <= own, so the last pick standing is element own (i ascending)
        body += pick if i == 0 else (If(_c(Fn.LT, Lit(i - 1), own), pick),)
    at = (col if not run else own if outs[0] == 0
          else _c(Fn.ADD, Lit(outs[0]), own))
    store = AtomicAdd if batch else Assign
    body.append(If(_c(Fn.LT, own, Lit(n)),
                   (store(Index("K", (ctx.bra, at)), mine),)))
    return (Note(SELECT_NOTE), Note(ATOMIC_WRITE if batch else PLAIN_WRITE),
            For(ROUND_VAR, Lit(0), Lit(n), Call(Fn.BLOCK_DIM_Y), tuple(body)))


def write_back_at(wavefront: int) -> Callable[[KContext], tuple[Stmt, ...]]:
    """The write-back built for a warp of `wavefront` lanes: the register
    form when `eligible`, else `coalesced.write_back` unchanged."""
    def write_back(ctx: KContext) -> tuple[Stmt, ...]:
        if eligible(len(ctx.k_outs), wavefront):
            return _register_write_back(ctx)
        return coalesced.write_back(ctx)
    return write_back


write_back = write_back_at(MIN_WAVEFRONT)


def apply(hooks: KHooks, spec: Spec) -> KHooks:
    """`hooks` with the register `write_back` and `coalesced`'s `atomize`;
    every other hook as given."""
    return dataclasses.replace(hooks, write_back=write_back,
                               atomize=coalesced._atomize(hooks.atomize))
