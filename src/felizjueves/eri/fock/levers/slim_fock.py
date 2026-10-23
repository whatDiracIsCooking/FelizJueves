"""The per-bra-element Fock-buffer split: each sliced-HRR slice keeps only its
own bra elements' buffers live (#360, milestone 13 lane H4).

Under #356's sliced HRR the digestion still holds the literal kernel's six
FULL buffers for the whole quartet: the five ket-dependent blocks
(`f_cd`, `f_ac`, `f_ad`, `f_bc`, `f_bd`) are zeroed before the first slice
and flushed after the last, so all `six_block_size - n_a n_b` of them are
live in every slice beside the bra's `f_ab`. Once the HRR is sliced, those
buffers are what rule the phase (the predecessor's ledger row 6b). This
lever gives each slice `(a, b)` a REDUCED buffer set:

    digest_setup                         once per quartet, as before
    for each bra pair s = (a, b):        #356's scoped one-trip slices
        { f_ac[a, :], f_ad[a, :],        the slice's four rows, zeroed:
          f_bc[b, :], f_bd[b, :] = 0     2 n_c + 2 n_d registers
          the slice's HRR and tap        #356's, unchanged
          per ERI (a b | c d), the literal block order:
            f_ab[a, b] += w_cd x         the per-bra buffer, as before
            AtomicAdd(F[c, d], w_ab x)   f_cd STREAMED: no buffer at all
            f_ac[a, c] += w_bd x  ...    the four rows
          AtomicAdd the four rows } into F

so a slice writes `1 + 2 n_c + 2 n_d` buffer elements (`slice_set`, the
issue's reduced set) and holds `n_a n_b + 2 n_c + 2 n_d` Fock buffers live
(`live_buffers`: the bra's `f_ab`, carried across the ket loop exactly as
the literal kernel carries it, plus the rows), against the literal
`six_block_size` -- 216 against 60 at `(dd|dd)`, 3.6x.

WHY `f_ab` IS NOT CUT. It is the one buffer the literal kernel does not
zero per quartet: it spans the thread's whole ket loop and is flushed once
(`flush_bra`), so it is live in every phase whatever the digestion does,
and cutting it per slice would turn its one flush per thread into `n_a n_b`
atomics per quartet. Leaving it is also what composes with #351
(`block-fab`), which reduces exactly that buffer across the block.

THE ATOMIC TRAFFIC MOVES, AND IS DECLARED (`ket_atomics`). Per worked
quartet the literal kernel flushes the five blocks once,
`six_block_size - n_a n_b` atomics; this lever issues `n_a n_b n_c n_d`
(the streamed `f_cd`, one per ERI) plus `n_a n_b (2 n_c + 2 n_d)` (the
rows, per slice): 180 against 2,160 at `(dd|dd)`. `f_ab`'s flush is
unchanged. test/test_f_lever_slim_fock.py counts both in the interpreter.

THE ARITHMETIC IS RE-ASSOCIATED, NOT CHANGED. Every ERI value is #356's,
hence the literal kernel's, bit for bit (the tap `g` is asserted bitwise);
every product `w x` is the literal one; `f_ab` is the literal per-thread
partial. What moves is how the products that land on one element of `F`
are grouped into atomics (per ERI for `f_cd`, per slice for the rows,
instead of per quartet). Both `F`s are sum trees over the same products, so
the lever is held to the literal kernel by the DERIVED bound
`|F_lever - F_literal| <= 2 gamma_{n-1} SUM |p_i|` per element over its `n`
products `p_i` -- recorded by a witness kernel that atomically adds every
product as it is formed -- never bitwise, and to `ao.fock`'s sweep at the
literal kernel's own `TOL`.

A COMPOSITION ON #356: `slim-fock` REQUIRES `sliced-hrr` (and so
`batched-vrr`) in the same spec, the key `batched-vrr+sliced-hrr+slim-fock`.
Without slices there is no per-bra-element phase to reduce the set to: one
HRR yields every ERI before the digestion, all of whose rows are then
written. Refused alone with `REFUSAL`. `hooks_for` applies key order, so
`sliced-hrr`'s `contracted` is installed when `apply` runs; `apply` checks
that it is exactly #356's (another lever that moved `contracted` would be
replaced silently) and that `zero_buffers`, `predigest_eri` and `flush_ket`
are still the frozen ones (a lever that moved them first -- #358's
scatter's ket flush, say -- would be dropped or contradicted: this lever's
flushes ARE atomics). `flush_bra` (#351's block reduction), `screen` /
`bra_setup` / `params` (#352's break) and `field` (#353's struct reads) are
untouched and compose.

THE HOOKS IT MOVES: `contracted` (#356's slices, each body wrapped by
`slice_wrap`: the slice's rows zeroed before it, flushed after it),
`zero_buffers` and `flush_ket` (slice-aware: on a slice's context they are
that slice's rows; at the quartet level, where #356's `contracted` still
calls them, they are a `Note` only) and `predigest_eri` (the literal six
multiply-adds with `f_cd`'s replaced by its streamed `AtomicAdd`). For the
levers that build on it: #359 (interleaved digest per slice) passes its own
per-slice body to `contracted_over(body)`; #361 (sub-slicing) cuts the
slice's HRR beneath the same rows.

Like the seams, nothing here imports `plan/`, `ao/`, the interpreter,
`reference/`, a backend, a device or `kernel_f`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from felizjueves.eri.fock.levers import sliced_hrr as _sh
from felizjueves.eri.fock.levers.seams import BLOCKS, FOCK, FContext, FHooks
from felizjueves.hgp.vrr import AmClass, cartesians
from felizjueves.ir import Assign, AtomicAdd, Call, Fn, Index, Lit, Note, Stmt, Var
from felizjueves.spec import Spec

__all__ = [
    "NAME",
    "REFUSAL",
    "REQUIRES",
    "ROWS",
    "apply",
    "contracted",
    "contracted_over",
    "flush_ket",
    "ket_atomics",
    "live_buffers",
    "n_cart",
    "predigest_eri",
    "reduced_rows",
    "six_block",
    "slice_set",
    "slice_wrap",
    "zero_buffers",
]

NAME = "slim-fock"
REQUIRES = _sh.NAME
CD = BLOCKS[1]  # (2, 3): the block streamed per ERI
ROWS = BLOCKS[2:]  # ac, ad, bc, bd: one row per slice each

REFUSAL = (
    "slim-fock requires sliced-hrr (and so batched-vrr) in the same spec "
    "(#360): the per-bra-element buffer set is a set PER SLICE, and without "
    "#356's slices one HRR yields every ERI of the quartet before the "
    "digestion, so every row of every block is written and none can be "
    "dropped")

_FROZEN = ("zero_buffers", "predigest_eri", "flush_ket")


# ---------------------------------------------------------------------------
# The declared functions of the class
# ---------------------------------------------------------------------------


def n_cart(cls: AmClass) -> tuple[int, ...]:
    """`(n_a, n_b, n_c, n_d)`: the four shells' Cartesian counts."""
    return tuple(len(cartesians(x)) for x in cls)


def six_block(cls: AmClass) -> int:
    """The literal kernel's six buffers, `kernel_f.six_block_size`."""
    n = n_cart(cls)
    return sum(n[i] * n[j] for i, j in BLOCKS)


def reduced_rows(cls: AmClass) -> int:
    """The buffers a slice declares: `2 n_c + 2 n_d` (its four rows)."""
    _, _, nc, nd = n_cart(cls)
    return 2 * nc + 2 * nd


def slice_set(cls: AmClass) -> int:
    """The buffer elements one slice writes, the issue's reduced set:
    `1 + 2 n_c + 2 n_d` (its `f_ab` element and its four rows)."""
    return 1 + reduced_rows(cls)


def live_buffers(cls: AmClass) -> int:
    """The Fock buffers live in a slice: the carried `f_ab` (`n_a n_b`) plus
    the rows; against `six_block` in the literal kernel and under #356."""
    na, nb, _, _ = n_cart(cls)
    return na * nb + reduced_rows(cls)


def ket_atomics(cls: AmClass, *, lever: bool) -> int:
    """The ket-dependent `AtomicAdd`s per worked quartet: the five blocks'
    one flush literal (`six_block - n_a n_b`), the streamed `f_cd` plus the
    rows per slice with the lever (`n_a n_b (n_c n_d + 2 n_c + 2 n_d)`)."""
    na, nb, nc, nd = n_cart(cls)
    if not lever:
        return six_block(cls) - na * nb
    return na * nb * (nc * nd + reduced_rows(cls))


# ---------------------------------------------------------------------------
# The hooks
# ---------------------------------------------------------------------------


def _row_elems(ctx: FContext, sl) -> list[tuple[tuple[int, int], int, int]]:
    """`(block, i, j)` of the slice's four rows, block order: `a`'s rows of
    `ac` / `ad`, then `b`'s of `bc` / `bd`."""
    own = {0: sl.a, 1: sl.b}
    return [(b, own[b[0]], j) for b in ROWS for j in range(ctx.ncart[b[1]])]


def zero_buffers(ctx: FContext) -> tuple[Stmt, ...]:
    """On a slice's context its four rows, zeroed; at the quartet level
    (#356's `contracted` calls it there too) nothing to zero."""
    sl = ctx.extra.get(_sh.NAME)
    if sl is None:
        return (Note("slim-fock (#360): no quartet-level ket buffers; each "
                     "slice zeroes its own rows and f_cd is streamed"),)
    return (Note(f"slim-fock (#360): slice ({sl.a}, {sl.b})'s reduced set, "
                 f"rows a = {sl.a} of ac, ad and b = {sl.b} of bc, bd"),
            *(ctx.scalar(ctx.buf(b, i, j), Lit(0.0), "fock")
              for b, i, j in _row_elems(ctx, sl)))


def flush_ket(ctx: FContext) -> tuple[Stmt, ...]:
    """On a slice's context its four rows' `AtomicAdd`; at the quartet
    level nothing left to flush."""
    sl = ctx.extra.get(_sh.NAME)
    if sl is None:
        return (Note("slim-fock (#360): every ket buffer was flushed by its "
                     "slice (rows) or as it was formed (f_cd)"),)
    return (Note(f"slim-fock (#360): flush slice ({sl.a}, {sl.b})'s rows"),
            *(AtomicAdd(Index(FOCK, ctx.ao(b, i, j)), Var(ctx.buf(b, i, j)))
              for b, i, j in _row_elems(ctx, sl)))


def predigest_eri(ctx: FContext, t, idx: tuple[int, ...]) -> tuple[Stmt, ...]:
    """The literal six multiply-adds of ERI `t`, block order, with `f_cd`'s
    replaced by `AtomicAdd(F[c, d], w_ab x)`: the same product, streamed."""
    x = ctx.hooks.eri(ctx, t)
    out: list[Stmt] = []
    for b in BLOCKS:
        c = ctx.comp(b)
        prod = Call(Fn.MUL, (Var(ctx.dens(c, idx[c[0]], idx[c[1]])), x))
        if b == CD:
            out.append(AtomicAdd(Index(FOCK, ctx.ao(b, idx[b[0]], idx[b[1]])),
                                 prod))
            continue
        f = Var(ctx.buf(b, idx[b[0]], idx[b[1]]))
        out.append(Assign(f, Call(Fn.ADD, (f, prod))))
    return tuple(out)


def slice_wrap(body: Callable[[FContext], tuple[Stmt, ...]]
               ) -> Callable[[FContext], tuple[Stmt, ...]]:
    """`body(sub)` between the slice's `zero_buffers` and `flush_ket`, both
    through the composed hooks."""

    def wrapped(sub: FContext) -> tuple[Stmt, ...]:
        h = sub.hooks
        return (*h.zero_buffers(sub), *body(sub), *h.flush_ket(sub))

    return wrapped


def contracted_over(body: Callable[[FContext], tuple[Stmt, ...]]):
    """#356's `contracted` with each slice's `body` (default
    `sliced_hrr.slice_body`) wrapped in its reduced set (#359's seam)."""
    return _sh.contracted_with(slice_wrap(body))


contracted = contracted_over(_sh.slice_body)


# ---------------------------------------------------------------------------
# The lever
# ---------------------------------------------------------------------------


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """The reduced per-slice set on #356's slices. Refused without
    `sliced-hrr` (`REFUSAL`), when applied before it, or after a lever that
    moved `contracted` or a hook this one replaces."""
    if REQUIRES not in spec.kernel_body_levers():
        raise ValueError(f"{spec.key()}: {REFUSAL}")
    if hooks.contracted is not _sh.contracted:
        raise ValueError(
            f"{spec.key()}: {NAME} wraps #356's slices, but `contracted` is "
            + ("still the frozen one (applied before sliced-hrr; hooks_for "
               "applies key order)" if hooks.is_default("contracted") else
               "another lever's, which this lever would replace silently"))
    moved = [h for h in _FROZEN if not hooks.is_default(h)]
    if moved:
        raise ValueError(f"{spec.key()}: {NAME} replaces {moved}; a lever "
                         "that moved them first would be dropped silently "
                         "(its flushes are atomics, per slice and per ERI)")
    return dataclasses.replace(hooks, contracted=contracted,
                               zero_buffers=zero_buffers, flush_ket=flush_ket,
                               predigest_eri=predigest_eri)
