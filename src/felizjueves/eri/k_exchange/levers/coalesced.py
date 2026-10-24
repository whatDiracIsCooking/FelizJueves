"""The coalesced K write-back (#345, milestone 13 lane M6).

`kernel_k`'s literal write-back (`seams.write_back`) reduces each `K[a, b]`
accumulator across the warp into a register `Kab{s}` and then writes them
from LANE 0, one store after another: `n_a * n_b` store instructions, each
with one active lane, so one transaction each -- serial in the accumulator
count. This lever issues those stores IN PARALLEL ACROSS THE LANES: lane `t`
writes `K[bra, t]`, `K[bra, t + W]`, ... (`W = blockDim.y`, the warp), so a
warp's stores to one bra row's contiguous `K[bra, :]` are one coalesced
transaction per `W` entries.

THE BROADCAST BUFFER IS THE ACCUMULATOR ITSELF. To store `K[a, b]` lane `t`
must hold it at an INDEXABLE place (a dynamic index into registers is a
local-memory array). `WarpReduce` gives EVERY lane the sum, so each lane
reduces its `Ksh` K slots IN PLACE -- `Ksh[n_i + s, t] <- SUM_lanes` -- and
after that every lane's own column of `Ksh` holds the whole reduced block.
Lane `t` then reads rows `n_i + t + k W` OF ITS OWN COLUMN, which it wrote
itself: no lane reads another lane's shared memory, so

- THERE IS NO WARP SYNC TO SPELL. The predecessor's form (lane 0 stores the
  block into a per-warp buffer, `__syncwarp()`, every lane reads) needs a
  barrier between a write and another lane's read, and a bare
  `__syncwarp()` hard-fails on hipcc. Here there is no cross-lane shared
  read, so no barrier is needed on either backend, the IR gains no node and
  neither emitter gains a spelling: the issue's "`FOCK_SYNCWARP`-style seam"
  is met by construction, not by a per-backend spelling.
- THE FOOTPRINT IS UNCHANGED. The issue budgets a broadcast buffer
  (`n_a n_b` doubles per warp, 288 B at `(dp|dp)`); reusing the accumulator
  slots costs none: `EXTRA_SHARED_BYTES = 0`, the `Ksh` declaration is the
  literal kernel's (`seams.shared_decl`, not moved), and
  `kernel_k.shared_bytes(am, lanes)` stays the class's footprint.
- THE ARITHMETIC IS THE LITERAL KERNEL'S. The same `WarpReduce` (the same
  xor butterfly on both backends and in the interpreter) on the same value,
  its sum stored to a shared double instead of a local one, then copied to
  `K` unchanged: interpreted with any lane count the lever's `K` is the
  literal kernel's at that lane count BIT FOR BIT
  (test/test_k_lever_coalesced.py), and with one lane `ao.k.k_quartet`'s.

The `K[a, b]` a kernel writes (`KContext.k_outs`, compacted into slots
`kslot`) are cut into maximal runs of consecutive flat indices `s` whose slots
are consecutive too; each run is one strided loop. With every element kept
(the literal mapping) that is ONE run, `s = 0 .. n_a n_b - 1`; composed with
#341's split (one element kept) it is a run of one, written by lane 0.

THE SPLIT CLASS (`n_batches > 1`): LANE-PARALLEL ATOMICS. A batch kernel
adds its partial `K[a, b]` to a `K` other batch kernels also add to (the
partial-sum form, #203), so its store is an `AtomicAdd`, issued by the same
strided loop: lane `t` adds `K[bra, s0 + t + k W]`, once per element per
warp (each element is stored by exactly one lane, so nothing is added once
per lane -- the reason the literal form keeps the lane-0 guard). A batch
writes only the `K[a, b]` it reaches (`KContext.k_outs`), which may be
several runs; each run is one loop. The default batch form,
`passes.atomize`, admits only the lane-0 write, so this lever also moves the
`atomize` seam (#345): `atomize` below accepts a kernel whose `K` writes are
already all atomic and hands any other kernel to the `atomize` it was given.
Onto a zeroed `K`, interpreted, the batches in order are the literal batched
kernels' `K` bit for bit at any lane count (the same partials, added in the
same batch order); on a card, as for the literal form, atomics add in
hardware order.

THE HOOKS IT MOVES: `write_back` and `atomize`, nothing else. Like the
seams, nothing here imports `ao/`, the interpreter, `reference/`, a backend
or a device.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from felizjueves.eri.k_exchange.levers.seams import KSH, KContext, KHooks
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Expr,
    Fn,
    For,
    Index,
    Kernel,
    Lit,
    Note,
    Stmt,
    Var,
    WarpReduce,
    walk,
)
from felizjueves.spec import Spec

__all__ = ["EXTRA_SHARED_BYTES", "LANE_VAR", "NAME", "PLAIN_WRITE",
           "REDUCE_NOTE", "ATOMIC_WRITE", "apply", "runs", "write_back"]

NAME = "coalesced"

EXTRA_SHARED_BYTES = 0
"""Shared memory the lever adds to the literal kernel's `Ksh`: none (the
reduced block is broadcast through the accumulator slots themselves)."""

LANE_VAR = "w"  # the strided write loop's variable

REDUCE_NOTE = (
    "coalesced write-back (#345): reduce each K[a,b] across the warp IN "
    "PLACE in the lane's Ksh slot -- every lane sees the sum, so after this "
    "every lane's own Ksh column holds the whole reduced block (the "
    "broadcast buffer is the accumulator: no extra shared memory, and no "
    "lane reads another lane's column, so no warp sync is needed)")

PLAIN_WRITE = (
    "plain write, NOT atomic: one batch, one warp per bra record, so this "
    "kernel is the only writer of its row of K. Coalesced (#345): lane t "
    "stores K[bra, s0 + t + k*blockDim.y] from its own Ksh column, the "
    "warp's stores in parallel, not lane 0's one after another")

ATOMIC_WRITE = (
    "atomic write, lane-parallel (#345): the partial-sum form (#203) -- a "
    "batch holds only some targets r = p + q, so every batch kernel of the "
    "class adds its partial K[a,b] to the same K[a,b] and the write must "
    "accumulate. Lane t adds K[bra, s0 + t + k*blockDim.y] from its own Ksh "
    "column: each element is added by exactly ONE lane, so no lane-0 guard "
    "is needed (the warp sum is not added once per lane)")


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _plus(base: int, e: Expr) -> Expr:
    return e if base == 0 else _c(Fn.ADD, Lit(base), e)


def runs(ctx: KContext) -> tuple[tuple[int, int, int], ...]:
    """`(s0, slot0, length)` per maximal run of the written `K[a, b]`: flat
    indices `s0 .. s0 + length - 1` in `Ksh` rows `slot0 ..`, both
    consecutive. One run, `(0, n_i, n_a n_b)`, when every element is kept."""
    out: list[list[int]] = []
    for s in ctx.k_outs:
        slot = ctx.kslot[s]
        if out and s == out[-1][0] + out[-1][2] and slot == out[-1][1] + out[-1][2]:
            out[-1][2] += 1
        else:
            out.append([s, slot, 1])
    return tuple(tuple(r) for r in out)


def write_back(ctx: KContext) -> tuple[Stmt, ...]:
    """In-place `WarpReduce` per `K[a, b]`, then one lane-strided store loop
    per run of `runs(ctx)`: a plain store for the one-batch kernel, an
    `AtomicAdd` for a batch kernel (module docstring)."""
    batch = ctx.head is not None
    ty, step = ctx.ty, Call(Fn.BLOCK_DIM_Y)
    out: list[Stmt] = [Note(REDUCE_NOTE)]
    out += [WarpReduce(ctx.sh(ctx.kslot[s]), ctx.sh(ctx.kslot[s]))
            for s in ctx.k_outs]
    out.append(Note(ATOMIC_WRITE if batch else PLAIN_WRITE))
    store = AtomicAdd if batch else Assign
    w = Var(LANE_VAR)
    for s0, slot0, n in runs(ctx):
        out.append(For(LANE_VAR, ty, Lit(n), step, (
            store(Index("K", (ctx.bra, _plus(s0, w))),
                  Index(KSH, (_plus(slot0, w), ty))),)))
    return tuple(out)


def _k_writes(kernel: Kernel) -> list[Stmt]:
    return [s for s in walk(kernel.body) if isinstance(s, Assign | AtomicAdd)
            and isinstance(s.dst, Index) and s.dst.array == "K"]


def _atomize(prior: Callable[[KContext, Kernel], Kernel]
             ) -> Callable[[KContext, Kernel], Kernel]:
    def atomize(ctx: KContext, kernel: Kernel) -> Kernel:
        writes = _k_writes(kernel)
        if not any(isinstance(s, AtomicAdd) for s in writes):
            return prior(ctx, kernel)  # not this lever's write: as given
        plain = [s for s in writes if not isinstance(s, AtomicAdd)]
        if plain:
            raise ValueError(
                f"kernel {kernel.name!r} mixes atomic and plain writes into "
                "K: the coalesced batch form is all atomic")
        return kernel
    return atomize


def apply(hooks: KHooks, spec: Spec) -> KHooks:
    """`hooks` with the coalesced `write_back` and the `atomize` that admits
    its atomic batch form; every other hook as given."""
    return dataclasses.replace(hooks, write_back=write_back,
                               atomize=_atomize(hooks.atomize))
