"""Struct-typed reads over the packed pair stream: the `struct-reads` F lever (#353).

The predecessor's ledger row T: read the packed shell-pair stream through
typed views instead of magic indices into a flat array. There it shipped as
an ADDITIVE `typed=` mode, never the default, because the two spellings are
separately frozen; here it is the same shape, a lever: the literal kernel's
text is unchanged (its goldens and digests are the check), and with the
lever every packed pair-record read the `field` hook routes is an
`ir.Member` -- the same element, read through a struct view of its row.

THE LAYOUT, DECLARED ONCE (`VIEWS`). `kernel_f.pack` stages one side of a
launch as struct-of-arrays, and each array's ROW is one record:

    array               row                      Record (struct fj_<name>)
    bra_prim, ket_prim  one primitive pair       prim_pair: P_x P_y P_z X_x
    ab_sh (shared)        (`seams.REC`, 8 f64)     X_y X_z zeta U    (64 B)
    AB, CD              one pair's A-B / C-D     sep: x y z, f64    (24 B)
    bra_sh, ket_sh      the two shell indices    index_pair: a b, i32 (8 B)
    bra_ao, ket_ao      the two AO offsets       index_pair

`G_ab` and `G_cd` are one double per pair, so their positional read already
IS the typed one and they are left alone. The views add no byte: each
struct is exactly its row (the writer `static_assert`s the size), with
natural alignment only, so the views ask nothing of a base pointer the flat
arrays do not -- which matters, because test/fock_driver.py carves every
double array of a launch out of one buffer and `ket_prim`'s base is only
8-byte aligned when `lb = 0` (no `AB`): a 16-byte-aligned `prim_pair`,
which might let a compiler issue 128-bit loads, would need a padded packing
and is NOT this lever (a follow-up). `test/test_f_lever_struct_reads.py`
asserts the declaration against `kernel_f.pack` / `PrimRec.row` member by
member, so a drift in either fails a test instead of corrupting a kernel.

WHAT CHANGES. Only the spelling of the reads: the IR differs from the
literal kernel's in exactly the record reads (`Index` -> `Member`, the same
`array` and `at`), so the interpreter, which reads a `Member` as the
`Index` it is, gives the literal kernel's `F` and tap bit for bit, and the
device arithmetic is the literal kernel's operation for operation. Whether
nvcc or hipcc emit different machine code for the typed spelling is a
compiler reading (the issue's "ptxas/VGPR both cards for both modes"),
printed by test/test_f_lever_struct_reads_device.py, never a store row.

It moves ONE hook, `field`, and refuses to compose over another lever's
`field` (two readers of one record would be a decision).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from felizjueves.eri.fock.levers import seams
from felizjueves.eri.fock.levers.seams import FContext, FHooks
from felizjueves.ir import Expr, Member, Record, Scalar
from felizjueves.spec import Spec

__all__ = ["INDEX_PAIR", "NAME", "PRIM_PAIR", "SEP", "VIEWS", "apply",
           "typed_field"]

NAME = "struct-reads"

PRIM_PAIR = Record("prim_pair", Scalar.F64, seams.REC)
SEP = Record("sep", Scalar.F64, ("x", "y", "z"))
INDEX_PAIR = Record("index_pair", Scalar.I32, ("a", "b"))

# packed array -> the Record one of its rows is (module docstring's table).
VIEWS: Mapping[str, Record] = MappingProxyType({
    "bra_prim": PRIM_PAIR, "ket_prim": PRIM_PAIR, seams.AB_SH: PRIM_PAIR,
    "AB": SEP, "CD": SEP,
    "bra_sh": INDEX_PAIR, "ket_sh": INDEX_PAIR,
    "bra_ao": INDEX_PAIR, "ket_ao": INDEX_PAIR,
})

# The records `seams.field_read` routes that keep their positional read.
POSITIONAL = frozenset(("G_ab", "G_cd"))


def typed_field(ctx: FContext, arr: str, idx: tuple[Expr, ...]) -> Expr:
    """A record read through its row's struct view; `G_ab` / `G_cd`
    positional; anything else refused (an unviewed record would be read
    silently untyped)."""
    rec = VIEWS.get(arr)
    if rec is not None:
        return Member(arr, idx, rec)
    if arr in POSITIONAL:
        return seams.field_read(ctx, arr, idx)
    raise ValueError(f"{NAME}: {arr!r} is a packed read with no declared "
                     "view (struct_reads.VIEWS)")


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """`field` -> `typed_field`; refused over another lever's `field`."""
    if not hooks.is_default("field"):
        raise ValueError(f"{NAME} moves the field hook, which another lever "
                         "already moved: two readers of one record")
    return dataclasses.replace(hooks, field=typed_field)
