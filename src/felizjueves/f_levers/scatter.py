"""The atomic-free write-back: launch slot buffers, then a scatter kernel
(#358, milestone 13 lane H8; the predecessor's ledger row 5).

The literal fused kernel writes `F` with `AtomicAdd` only: every thread's
five ket-dependent buffers after each quartet (`flush_ket`) and its `f_ab`
once after its ket loop (`flush_bra`). This lever removes EVERY atomic from
the F kernel. Each launch writes its contributions with plain stores into a
per-launch buffer `W`, and a second, class-agnostic kernel
(`kernel_f_scatter`, its own unit `f_scatter`) sums `W` into `F` in a fixed
order, one thread per element of `F` the launch touches:

    F kernel (this lever)                   W[slot] = value  (no atomics)
    scatter  (kernel_f_scatter)             F[r, c] += SUM W[src[t]]

So `F` is DETERMINISTIC: bit-identical from run to run on one card, which
no atomic path is (the order of the fp64 atomics is the hardware's).

THE BUFFER, AND WHY IT IS SLOTS, NOT AN AO-INDEXED TILE. A plain store is
race-free only with one writer per element. Per block (one bra pair) the
bra block `f_ab` has `blockDim` writers, one partial per thread, and the
cross blocks `f_ac`, `f_ad`, `f_bc`, `f_bd` and `f_cd` land on AO elements
two kets of one block can SHARE (two ket pairs with the same `c` shell
write the same `(a_i, c_k)`; test_f_lever_scatter.py finds such kets in
H2O's plan), so an AO-indexed tile of the launch's footprint would need an
atomic, or a barrier-synchronised reduction across the strided ket loop,
to be written. What has ONE writer is:

  * `f_ab` AFTER #351's block reduction: one column sum per element per
    block (`block_fab`). This lever therefore CONTAINS block-fab's
    reduction (`apply` composes `block_fab.apply` itself) and replaces
    only its final `AtomicAdd` by a store. Naming `block-fab` beside
    `scatter` is REFUSED: it would be the same kernel under a second key.
  * each quartet's five ket-dependent blocks: one thread per contracted
    quartet, so the slot `(bra, k)` is written by exactly one thread.

So the launch's buffer is, per bra, its reduced `f_ab` and then the five
blocks of each of its `n_ket` quartets, element-major so that consecutive
threads (consecutive `k`) store to consecutive addresses:

    row      = n_ab + S5 n_ket                S5 = n_cd + n_ac + n_ad
                                                   + n_bc + n_bd
    W[bra row + e]                            f_ab element e = i n_b + j
    W[bra row + n_ab + (o_x + i n_y + j) n_ket + k]
                                              block x = (u, v) of
                                              BLOCKS[1:], o_x its offset
    n_slot   = n_bra (n_ab + S5 n_ket)        (`n_slot`, the declared size)

sized by the LAUNCH (`n_bra x n_ket`), never by `N^2` or by the block
width. It is `W`'s declared extent (`Param.shape`), so the interpreter
refuses a buffer one element short at launch, and every subscript is
bounds-checked: a buffer too small is an error, never a dropped term.
A screened quartet (and the canonical half's skipped `k < bra`) stores
nothing: its slots stay the zeros the host wrote before the launch, and
the scatter adds them exactly. `launch_f.launch_footprint` declares the
same `n_slot` from the plan, independently, and the AO footprint the
scatter writes (`kernel_f_scatter.footprint_bound`).

THE ARITHMETIC IS RE-ASSOCIATED, NOT CHANGED: every slot holds the value
the literal kernel (or block-fab, for `f_ab`) would have added atomically,
bit for bit; only the order of the sums into `F` differs. So the lever is
held to block-fab's kernel by the derived re-association bound, `|F_scatter
- F_blockfab| <= 2 gamma_{n-1} SUM |a_i|` per element over its `n` addends,
and to `ao.fock`'s sweep at the literal kernel's own `TOL`.

THE LAUNCH CONTRACT is block-fab's (`block_fab.launch_shape`: a block a
multiple of the wavefront, at most 1024), and after every F launch the host
launches the scatter (`kernel_f_scatter.launch_shape`) over that launch's
map (`kernel_f_scatter.scatter_map`), before the next F launch reuses `W`.

COMPOSITION. A lever that moves `flush_ket` or `flush_bra` itself is
refused (both writes are this lever's); `params` is WRAPPED, so a lever
that adds a parameter (#352's `M_ab`) composes, its `F` replaced by `W`.
"""

from __future__ import annotations

import dataclasses

from felizjueves.f_levers import block_fab
from felizjueves.f_levers.seams import BLOCKS, FOCK, FContext, FHooks
from felizjueves.hgp import vrr as _vrr
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
    Intent,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.spec import Spec

__all__ = [
    "I32_MAX",
    "NAME",
    "REDUNDANT",
    "SLOTS",
    "apply",
    "block_size",
    "ket_offsets",
    "ket_width",
    "n_ab",
    "n_slot",
    "row_width",
    "slot_ab",
    "slot_ket",
]

NAME = "scatter"
SLOTS = "W"  # the launch's slot buffer, Intent.OUT of the F kernel
I32_MAX = 2 ** 31 - 1  # a slot index is an I32

REDUNDANT = (
    "scatter already contains block-fab's f_ab reduction (its one writer "
    "per element is what makes the f_ab store atomic-free): naming both "
    "would build the same kernel under a second key")

_Q = "sc_q"  # this thread's first ket slot


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


# --- the layout, as Python (the host's map and the declared size) ------------


def _ncart(cls) -> tuple[int, ...]:
    return tuple(len(_vrr.cartesians(x)) for x in tuple(cls))


def block_size(cls, b: tuple[int, int]) -> int:
    n = _ncart(cls)
    return n[b[0]] * n[b[1]]


def n_ab(cls) -> int:
    """The bra block's elements: `n_a n_b`."""
    return block_size(cls, BLOCKS[0])


def ket_offsets(cls) -> dict[tuple[int, int], int]:
    """Each ket-dependent block's offset inside a quartet's `S5` slots,
    `BLOCKS[1:]` order."""
    out, at = {}, 0
    for b in BLOCKS[1:]:
        out[b] = at
        at += block_size(cls, b)
    return out


def ket_width(cls) -> int:
    """`S5`: the five ket-dependent blocks' elements, one quartet's slots."""
    return sum(block_size(cls, b) for b in BLOCKS[1:])


def row_width(cls, n_ket: int) -> int:
    """One bra's slots: `n_ab + S5 n_ket`."""
    return n_ab(cls) + ket_width(cls) * n_ket


def n_slot(cls, n_bra: int, n_ket: int) -> int:
    """The launch's slot count, `W`'s declared extent; refuses one an I32
    slot index cannot address."""
    for name, v in (("n_bra", n_bra), ("n_ket", n_ket)):
        if type(v) is not int or v < 1:
            raise ValueError(f"{name} = {v!r} is not a positive int")
    n = n_bra * row_width(cls, n_ket)
    if n > I32_MAX:
        raise ValueError(f"{n} slots exceed an I32 index ({I32_MAX}): split "
                         "the launch")
    return n


def slot_ab(cls, n_ket: int, bra: int, e: int) -> int:
    """`f_ab` element `e = i n_b + j` of bra `bra`."""
    return bra * row_width(cls, n_ket) + e


def slot_ket(cls, n_ket: int, bra: int, k: int, b: tuple[int, int], i: int,
             j: int) -> int:
    """Element `(i, j)` of ket-dependent block `b` of quartet `(bra, k)`."""
    o = ket_offsets(cls)[b] + i * _ncart(cls)[b[1]] + j
    return bra * row_width(cls, n_ket) + n_ab(cls) + o * n_ket + k


# --- the layout, as IR (the kernel's stores) ---------------------------------


def _row(ctx: FContext) -> Expr:
    return _c(Fn.ADD, Lit(n_ab(ctx.cls)),
              _c(Fn.MUL, Lit(ket_width(ctx.cls)), Var("n_ket")))


def _params(prev):
    def params(ctx: FContext) -> tuple[Param, ...]:
        out = tuple(prev(ctx))
        if [p.name for p in out].count(FOCK) != 1:
            raise ValueError(f"{ctx.name}: scatter replaces the one {FOCK} "
                             "parameter, and there is not one")
        w = Param(SLOTS, Scalar.F64, array=True, intent=Intent.OUT,
                  shape=(_c(Fn.MUL, Var("n_bra"), _row(ctx)),))
        return tuple(w if p.name == FOCK else p for p in out)
    return params


def flush_ket(ctx: FContext) -> tuple[Stmt, ...]:
    """The five ket-dependent buffers, stored to this quartet's slots."""
    q = Var(_Q)
    out: list[Stmt] = [
        Note("#358: store the five ket-dependent buffers to this quartet's "
             "slots of W, element-major (consecutive k, consecutive "
             "addresses); one writer per slot, no atomics"),
        Decl(_Q, Scalar.I32, init=_c(Fn.ADD, _c(
            Fn.ADD, _c(Fn.MUL, ctx.bra, _row(ctx)), Lit(n_ab(ctx.cls))),
            ctx.k)),
    ]
    off = ket_offsets(ctx.cls)
    for b in BLOCKS[1:]:
        nj = ctx.ncart[b[1]]
        for i, j in ctx.elems(b):
            o = off[b] + i * nj + j
            at = q if o == 0 else _c(Fn.ADD, q, _c(Fn.MUL, Lit(o),
                                                   Var("n_ket")))
            out.append(Assign(Index(SLOTS, (at,)), Var(ctx.buf(b, i, j))))
    return tuple(out)


def _store_fab(ctx: FContext, body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """block-fab's column loop with its `AtomicAdd` into `F` a store to the
    bra's `f_ab` slot, and the two AO indices only that atomic read gone."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, AtomicAdd) and s.dst.array == FOCK:
            out.append(Assign(Index(SLOTS, (_c(Fn.ADD, _c(
                Fn.MUL, ctx.bra, _row(ctx)), Var(block_fab._E)),)), s.value))
        elif isinstance(s, Decl) and s.name in (block_fab._I, block_fab._J):
            continue
        elif isinstance(s, For):
            out.append(For(s.var, s.start, s.stop, s.step,
                           _store_fab(ctx, s.body)))
        elif isinstance(s, If):
            out.append(If(s.cond, _store_fab(ctx, s.then),
                          _store_fab(ctx, s.orelse)))
        else:
            out.append(s)
    return tuple(out)


def _flush_bra(prev):
    def flush_bra(ctx: FContext) -> tuple[Stmt, ...]:
        body = tuple(prev(ctx))
        n = sum(isinstance(s, AtomicAdd) for s in _walk(body))
        if n != 1:
            raise ValueError(f"{ctx.name}: block-fab's flush has {n} atomics, "
                             "not its one column-sum AtomicAdd")
        return (Note("#358: block-fab's reduction, its one column sum per "
                     "element stored to the bra's f_ab slot of W"),
                *_store_fab(ctx, body))
    return flush_bra


def _walk(body):
    for s in body:
        yield s
        if isinstance(s, For):
            yield from _walk(s.body)
        elif isinstance(s, If):
            yield from _walk(s.then)
            yield from _walk(s.orelse)


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """#358: block-fab's reduction, then every write to `F` a store to `W`
    and `F` replaced by `W` in the signature (module docstring)."""
    if block_fab.NAME in tuple(spec.levers):
        raise ValueError(f"{spec.key()}: {REDUNDANT}")
    moved = [h for h in ("flush_ket", "flush_bra") if not hooks.is_default(h)]
    if moved:
        raise ValueError(f"{spec.key()}: another lever moved {moved}, the "
                         "writes scatter replaces; refused, not composed")
    hooks = block_fab.apply(hooks, spec)
    return dataclasses.replace(hooks, flush_ket=flush_ket,
                               flush_bra=_flush_bra(hooks.flush_bra),
                               params=_params(hooks.params))
