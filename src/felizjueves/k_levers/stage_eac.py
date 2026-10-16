"""`E^ac` staged in shared memory, once per bra record (#342, milestone 13 lane M2).

`kernel_k`'s mapping is one warp per bra record -- block-per-`K[ab]`:
blockDim = (1, lanes), `bra = blockIdx.x`, so the bra record and everything
derived from it is BLOCK-UNIFORM. The literal kernel nonetheless binds the
bra's `E^ac` block per THREAD: each lane inlines `e_fragment.block_fragment(la,
lc)` into a local `Eac[n_ac, n_p]` and reads it from there in every ket
iteration. At `(dd|dd)` that is 1,260 doubles a lane carries through the ket
loop, in registers or, past the register file, in (global-backed) local
memory. This lever moves that block into one `__shared__` copy per block, as
the predecessor's `cuda_block.py` emitter did (its ledger row 4: `(dd|dd)`
spill loads -12% at 31 KB of shared on nvcc, and +156% on hipcc -- a
direction flip, which is why the lever is measured on both of FJ's cards and
not taken on the predecessor's word).

THE STAGING PATTERN, AND WHY THERE IS NO BARRIER. The IR has no barrier
(#333), and adding one is outside this lever's reach. So the staging is
BLOCK-UNIFORM AND REDUNDANT: every lane runs the same inlined fragment, from
the same block-uniform scalars (`zeta_ac[bra]`, `XPA[bra, :]`,
`XPC[bra, :]`), and stores every entry of the block into the one shared
array `EacS[n_ac][n_p]` -- the literal kernel's arithmetic, its target moved.
Two facts make that correct without synchronisation, and both are checked
when the kernel is built (`_write_once`):

- the fragment WRITES EACH ENTRY ONCE and NEVER READS THE ARRAY (every entry
  one `Weighted` from scalars, #152), so no lane ever reads a half-formed
  entry another lane is still accumulating; and
- every lane writes the BIT-IDENTICAL value to a given entry (the same
  instructions on the same inputs; no fast-math flag, ever), and a lane
  reads an entry only after its own store of it, in program order.

So whatever store a read observes -- its own lane's or another's -- it reads
the value the literal kernel's local held. Formally the identical-value
stores are a race in the vendors' memory models (plain conflicting writes
with no synchronisation); it is relied on as benign, the device test is the
witness, and a cooperative store (each lane a slice, then `__syncwarp` --
the block is one warp) is the form a barrier node would admit, named, NOT
BUILT (#333). The trade, stated: the `E^ac` FLOPs are still paid by every
lane (as in the literal kernel); what moves is the STORAGE -- registers or
local memory per lane become one shared copy per block. Whether a compiler
honours that (it may forward a lane's own stores to its loads and keep the
values in registers anyway) is exactly what the compile figures and the
cards answer.

THE ARITHMETIC IS THE LITERAL KERNEL'S, bit for bit: the same fragment, the
same values, the same reads in the same order, only `Eac[r][t]` spelled
`EacS[r][t]` (test/test_k_lever_stage_eac.py, interpreted, against
`ao.k.k_quartet` per bra record and against the literal kernel).

THE FOOTPRINT, a declared function of the class (no build needed), beside
`kernel_k.shared_bytes`' dynamic `Ksh`:

    staged_doubles(am) = n_ac * n_p,   n_ac = n_a n_c,
                         n_p = (lp+1)(lp+2)(lp+3)/6,  lp = la + lc
    staged_bytes(am)   = 8 * staged_doubles(am)        STATIC, per block
    footprint_bytes(am, lanes) = shared_bytes(am, lanes) + staged_bytes(am)

`staged_bytes` does not scale with `lanes` (one copy per block, not per
lane). `(dd|dd)`: 36 * 35 * 8 = 10,080 B staged + 18,176 B of `Ksh` at 32
lanes = 28,256 B. A split class's batch kernel stages the whole block too,
as the literal one forms it whole. THE BUDGET is `BUDGET_BYTES` = 48 KiB per
block (both cards' default, and the static `__shared__` ceiling) at
`BUDGET_LANES` = 32, the warp / wave32 width of both cards; `apply` REFUSES
a class whose footprint is over it (`(ff|ff)`: 67,200 B staged alone),
rather than emit a kernel that cannot be compiled or launched.

THE HOOKS IT MOVES: `bind_e_ac` (the shared declaration, the notes, the
fragment inlined into it) and `e_ac` (every read). Nothing else, so it
composes with a lever that moves the contraction, the fan-out or the
write-back (the reads go through `ctx.hooks.e_ac`).

Like the seams, nothing here imports `ao/`, the interpreter, `reference/`, a
backend or a device.
"""

from __future__ import annotations

import dataclasses

from felizjueves import e_fragment as ef
from felizjueves.ir import (
    Assign,
    Decl,
    Expr,
    Index,
    Inline,
    Lit,
    Note,
    Scalar,
    Space,
    Stmt,
    Var,
    _exprs_of,
    _subexprs,
    walk,
)
from felizjueves.k_levers.seams import KContext, KHooks, bindings, ncart
from felizjueves.spec import Spec

__all__ = ["ARRAY", "BUDGET_BYTES", "BUDGET_LANES", "NAME", "PATTERN",
           "apply", "footprint_bytes", "staged_bytes", "staged_doubles"]

NAME = "stage-eac"
ARRAY = "EacS"  # the staged block: one static __shared__ copy per block

BUDGET_BYTES = 48 * 1024  # per block: both cards' default, the static ceiling
BUDGET_LANES = 32  # warp (sm_86) / wave32 (gfx1200) width

PATTERN = (
    "stage-eac (#342): E^ac is block-uniform (one warp per bra record), so "
    "it is staged ONCE PER BLOCK in __shared__ EacS, not per lane. No "
    "barrier (the IR has none, #333): every lane runs the same fragment on "
    "the same bra scalars and stores the bit-identical value to each entry, "
    "each entry written once and never read by the fragment, and a lane "
    "reads an entry only after its own store -- so any store it observes "
    "holds the literal value (identical-value stores: a benign race, "
    "witnessed on the card). The E^ac FLOPs stay per lane; the storage "
    "moves from registers/local to one shared copy")


def staged_doubles(am: tuple[int, int, int, int]) -> int:
    """`n_ac * n_p`: the staged `E^ac` block's doubles, from the class alone."""
    la, lc, _, _ = am
    lp = la + lc
    n_p = (lp + 1) * (lp + 2) * (lp + 3) // 6
    return ncart(la) * ncart(lc) * n_p


def staged_bytes(am: tuple[int, int, int, int]) -> int:
    """The staged block's STATIC shared bytes per block (not per lane)."""
    return 8 * staged_doubles(am)


def footprint_bytes(am: tuple[int, int, int, int], lanes: int) -> int:
    """The block's whole shared footprint with the lever: the dynamic `Ksh`
    (`kernel_k.shared_bytes`, the same arithmetic, restated here because a
    lever may not import `kernel_k`) plus the static staged block."""
    la, lc, lb, _ = am
    lp = la + lc
    n_p = (lp + 1) * (lp + 2) * (lp + 3) // 6
    ksh = (n_p + ncart(la) * ncart(lb)) * lanes * 8
    return ksh + staged_bytes(am)


def _write_once(frag) -> None:
    """The precondition of the barrier-free staging: the fragment writes each
    `E` entry at most once and never reads the `E` array."""
    seen: set = set()
    for s in walk(frag.body):
        for root in _exprs_of(s):
            if root is None:
                continue
            for e in _subexprs(root):
                if (isinstance(e, Index) and e.array == ef.E_ARRAY
                        and not (isinstance(s, Assign) and e is s.dst)):
                    raise ValueError(f"{NAME}: the E^ac fragment reads its own "
                                     "array; staging it without a barrier "
                                     "would race")
        if (isinstance(s, Assign) and isinstance(s.dst, Index)
                and s.dst.array == ef.E_ARRAY):
            at = tuple(i.value for i in s.dst.at)
            if at in seen:
                raise ValueError(f"{NAME}: the E^ac fragment writes {at} "
                                 "twice; staging it without a barrier would "
                                 "race")
            seen.add(at)


def bind_e_ac(ctx: KContext) -> tuple[Stmt, ...]:
    """The block's `EacS[n_ac][n_p]` in shared, from the fragment inlined by
    every lane (module docstring, "THE STAGING PATTERN")."""
    _write_once(ctx.frag_ac)
    return (
        Note(PATTERN),
        Note("the undressed E^ac block of this bra record, once per block: "
             "e_fragment.block_fragment(la, lc), inlined; s_ac is the host's"),
        Decl(ARRAY, Scalar.F64, (Lit(len(ctx.rows_ac)), Lit(len(ctx.ps))),
             Space.SHARED),
        Inline(ctx.frag_ac, {**bindings(ctx.frag_ac, Var("ze"), "XPA", "XPC",
                                        ctx.bra),
                             ef.E_ARRAY: Var(ARRAY)}),
    )


def e_ac(ctx: KContext, rac: int, i: int) -> Expr:
    return Index(ARRAY, (Lit(rac), Lit(i)))


def apply(hooks: KHooks, spec: Spec) -> KHooks:
    """`hooks` with `E^ac` staged in shared; refused over `BUDGET_BYTES`."""
    need = footprint_bytes(spec.am, BUDGET_LANES)
    if need > BUDGET_BYTES:
        raise ValueError(
            f"{NAME}: {spec.key()} needs {need} B of shared per block at "
            f"{BUDGET_LANES} lanes ({staged_bytes(spec.am)} B staged E^ac), "
            f"over the {BUDGET_BYTES} B budget")
    return dataclasses.replace(hooks, bind_e_ac=bind_e_ac, e_ac=e_ac)
