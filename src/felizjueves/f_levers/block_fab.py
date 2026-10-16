"""The block-level `f_ab` reduction: warp shuffle, then shared memory, then
ONE atomic per element per block (#351, milestone 13 lane H7).

`kernel_f`'s mapping is one block per bra shell pair, one thread per
contracted quartet: every thread of a block shares the bra, so every
thread's `f_ab` buffer -- the bra block's `n_a n_b` partial sums, carried
across the thread's whole ket loop -- targets the SAME `n_a n_b` elements of
`F`. The literal `flush_bra` issues them as `blockDim * n_a n_b` serial
`AtomicAdd`s on those elements (the predecessor's ledger row 3). This lever
reduces across the block first and flushes once per element:

    ws    = WarpReduce(1)                   the warp's width, MEASURED
    w, l  = tid / ws, tid - w ws            the warp index and the lane
    r_e   = WarpReduce(f_ab_e)              per element: the xor butterfly
    if l < 1:  fab_sh[w, e] = r_e           lane 0 of each warp, per element
    Barrier                                 (ir.Barrier, #351: __syncthreads)
    for e = tid; e < n_a n_b; e += blockDim:
        s = fab_sh[0, e] + fab_sh[1, e] + ... + fab_sh[nw - 1, e]
        AtomicAdd(F[oa + e / n_b, ob + e % n_b], s)

THE WIDTH IS MEASURED, NOT KNOWN. The IR deliberately does not know the
wavefront (ir.py, "WHAT IS DELIBERATELY NOT HERE"; test_ir pins that no
vocabulary names it), and the fused kernel's block is `(threads, 1)` with
the kets strided over `threadIdx.x` (`kernel_f.launch_shape`), so the warp
index is not a launch coordinate either. `WarpReduce(1)` -- the sum of 1
over the warp's lanes -- IS the width, on every vendor and in the
interpreter (`Launch.lanes`), at the cost of one integer butterfly per
block. Nothing in the kernel's text names 32 or 64.

THE BARRIER is `ir.Barrier` (#351 adds it; #333 named it missing): the one
cross-warp step needs every warp's lane-0 write visible before any thread
sums a column. It sits under `bra < n_bra`, which is block-uniform
(`bra = blockIdx.x`), after the ket loop every thread leaves, so every
thread of the block reaches it -- the contract `ir.Barrier` states and the
interpreter enforces.

THE LAUNCH CONTRACT: `threads` a positive multiple of the wavefront (a
`WarpReduce` in a partial warp is undefined; the interpreter refuses it)
and at most 1024 (both vendors' block limit). Then the warp count is at
most `1024 / 32 = MAX_WARPS`, the smallest wavefront either vendor has, so
`fab_sh`'s extent is a literal and the footprint a function of the class
alone (`launch_shape` below refuses a block outside it).

THE SHARED FOOTPRINT, declared per class (no build needed):

    reduce_doubles(cls)  = MAX_WARPS * n_a * n_b
    reduce_bytes(cls)    = 8 * reduce_doubles(cls)              (FP64)
    footprint_bytes(cls) = 8 * MAX_PRIM_PAIRS * len(REC) + reduce_bytes(cls)

`(ss|..)` 256 B, `(pp|..)` 2,304 B, `(dd|..)` 9,216 B beside the literal
8,192 B of `ab_sh` (17,408 B in all for a `dd` bra).

THE ATOMIC TRAFFIC, as a count (`bra_flush_atomics`): per block, the
literal flush is `threads * n_a n_b` atomics on `F`, this one `n_a n_b`;
the five ket-dependent blocks' flushes (`flush_ket`, a different element per
quartet) are unchanged, so per launch the reduction removes exactly
`n_bra (threads - 1) n_a n_b` atomics (test/test_f_lever_block_fab.py
counts them in the interpreter).

THE ARITHMETIC IS RE-ASSOCIATED, NOT CHANGED: every thread's `f_ab` partial
is the literal kernel's value, bit for bit; only the order in which they
(and the ket flushes landing on the same element) are summed into `F`
differs. So the lever is held to the literal kernel by a DERIVED bound,
`|F_lever - F_literal| <= 2 gamma_{n-1} SUM |a_i|` per element over its `n`
addends (both are sum trees over the same leaves), never bitwise, and to
`ao.fock`'s sweep at the literal kernel's own `TOL`.

What is NOT here: the ket-dependent buffers' reduction (each thread's
`c`, `d` differ, so there is nothing block-uniform to reduce), the
atomic-free write-back (#358, which builds on this), and the cooperative
`|ab)` staging the barrier now admits (kernel_f's "THE STAGING IS PER
THREAD" names it; another lever's).
"""

from __future__ import annotations

import dataclasses

from felizjueves.f_levers.seams import (
    AB_SH,
    AO_OF,
    BLOCKS,
    FOCK,
    MAX_PRIM_PAIRS,
    REC,
    FContext,
    FHooks,
)
from felizjueves.hgp import vrr as _vrr
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Barrier,
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
    Space,
    Stmt,
    Var,
    WarpReduce,
)
from felizjueves.spec import Spec

__all__ = [
    "MAX_THREADS",
    "MAX_WARPS",
    "NAME",
    "RED_SH",
    "apply",
    "bra_flush_atomics",
    "footprint_bytes",
    "launch_shape",
    "n_elems",
    "reduce_bytes",
    "reduce_doubles",
]

NAME = "block-fab"
RED_SH = "fab_sh"  # the per-warp partial sums, Space.SHARED
MAX_THREADS = 1024  # both vendors' block limit
MIN_WAVEFRONT = 32  # the narrowest wavefront either vendor has
MAX_WARPS = MAX_THREADS // MIN_WAVEFRONT

_WS, _W, _LANE, _NW = "fab_ws", "fab_w", "fab_lane", "fab_nw"
_E, _V, _S, _I, _J = "fab_e", "fab_v", "fab_s", "fab_i", "fab_j"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def n_elems(cls) -> int:
    """`n_a n_b`: the bra block's elements, the reduction's columns."""
    la, lb = tuple(cls)[:2]
    return len(_vrr.cartesians(la)) * len(_vrr.cartesians(lb))


def reduce_doubles(cls) -> int:
    return MAX_WARPS * n_elems(cls)


def reduce_bytes(cls) -> int:
    """`fab_sh`'s static shared bytes (FP64; module docstring)."""
    return 8 * reduce_doubles(cls)


def footprint_bytes(cls) -> int:
    """The kernel's whole static shared footprint: `ab_sh` plus `fab_sh`."""
    return 8 * MAX_PRIM_PAIRS * len(REC) + reduce_bytes(cls)


def bra_flush_atomics(cls, threads: int, *, lever: bool) -> int:
    """The `f_ab` flush's atomics on `F` per block: `threads * n_a n_b`
    literal, `n_a n_b` with the lever."""
    return n_elems(cls) * (1 if lever else threads)


def launch_shape(n_bra: int, threads: int, wavefront: int
                 ) -> tuple[int, tuple[int, int]]:
    """`kernel_f.launch_shape`'s `(n_bra, (threads, 1))`, refusing a block
    the reduction's contract does not cover (module docstring)."""
    for name, v in (("n_bra", n_bra), ("threads", threads),
                    ("wavefront", wavefront)):
        if type(v) is not int or v < 1:
            raise ValueError(f"{name} = {v!r} is not a positive int")
    if wavefront < MIN_WAVEFRONT:
        raise ValueError(f"wavefront {wavefront} < {MIN_WAVEFRONT}: fab_sh "
                         f"holds {MAX_WARPS} warps")
    if threads % wavefront or threads > MAX_THREADS:
        raise ValueError(f"threads = {threads}: the block reduction needs a "
                         f"multiple of the wavefront ({wavefront}) of at "
                         f"most {MAX_THREADS}")
    return n_bra, (threads, 1)


def _shared_decl(prev):
    def shared_decl(ctx: FContext) -> tuple[Stmt, ...]:
        n = n_elems(ctx.cls)
        return (*prev(ctx),
                Note(f"#351: the per-warp f_ab partial sums, {MAX_WARPS} "
                     f"warps (blockDim <= {MAX_THREADS}, wavefront >= "
                     f"{MIN_WAVEFRONT}) x {n} elements, "
                     f"{reduce_bytes(ctx.cls)} B beside {AB_SH}"),
                Decl(RED_SH, ctx.hooks.real(ctx, "fock"),
                     (Lit(MAX_WARPS), Lit(n)), Space.SHARED))
    return shared_decl


def flush_bra(ctx: FContext) -> tuple[Stmt, ...]:
    """The block reduction of `f_ab` (module docstring), one AtomicAdd per
    element per block."""
    ab = BLOCKS[0]
    elems = list(ctx.elems(ab))
    n, nb = len(elems), ctx.ncart[1]
    real = ctx.hooks.real(ctx, "fock")
    tid, dim = Call(Fn.THREAD_X), Call(Fn.BLOCK_DIM_X)
    ws, w, lane, nw = Var(_WS), Var(_W), Var(_LANE), Var(_NW)
    e, v, s = Var(_E), Var(_V), Var(_S)

    def red(i: int, j: int) -> str:
        return f"fab_r_{i}_{j}"

    out: list[Stmt] = [
        Note("#351: reduce f_ab across the block, then flush once per "
             "element: warp shuffle, lane 0 into shared, a barrier, one "
             "column sum and one AtomicAdd per element"),
        Note("the warp's width, measured: the sum of 1 over its lanes"),
        Decl(_WS, Scalar.I32),
        WarpReduce(ws, Lit(1)),
        Decl(_W, Scalar.I32, init=_c(Fn.DIV, tid, ws)),
        Decl(_LANE, Scalar.I32, init=_c(Fn.SUB, tid, _c(Fn.MUL, w, ws))),
    ]
    for i, j in elems:
        out += [Decl(red(i, j), real),
                WarpReduce(Var(red(i, j)), Var(ctx.buf(ab, i, j)))]
    out += [
        If(_c(Fn.LT, lane, Lit(1)), tuple(
            Assign(Index(RED_SH, (w, Lit(k))), Var(red(i, j)))
            for k, (i, j) in enumerate(elems))),
        Note("every warp's partial sums visible to every thread"),
        Barrier(),
        Decl(_NW, Scalar.I32, init=_c(Fn.DIV, _c(
            Fn.ADD, dim, _c(Fn.SUB, ws, Lit(1))), ws)),
        For(_E, tid, Lit(n), dim, (
            Decl(_S, real, init=Index(RED_SH, (Lit(0), e))),
            For(_V, Lit(1), nw, Lit(1), (
                Assign(s, _c(Fn.ADD, s, Index(RED_SH, (v, e)))),)),
            Decl(_I, Scalar.I32, init=_c(Fn.DIV, e, Lit(nb))),
            Decl(_J, Scalar.I32, init=_c(Fn.SUB, e, _c(
                Fn.MUL, Var(_I), Lit(nb)))),
            AtomicAdd(Index(FOCK, (_c(Fn.ADD, Var(AO_OF[0]), Var(_I)),
                                   _c(Fn.ADD, Var(AO_OF[1]), Var(_J)))), s),
        )),
    ]
    return tuple(out)


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """#351: `fab_sh` beside the staged `|ab)`, and the block-reduced
    `f_ab` flush; nothing else moves."""
    return dataclasses.replace(hooks, shared_decl=_shared_decl(
        hooks.shared_decl), flush_bra=flush_bra)
