"""`PrimPairSoA`: the one-electron kernels' primitive-pair records, as a struct.

    PRIM_PAIR_SOA                      the `ir.Bundle`
    AXES                               ("X", "Y", "Z")
    bundled(arg="pp", **binds) -> ir.Bundled

A primitive-pair class reaches a one-electron kernel (`S`, `T`, `V`, `SX`)
as STRUCTURE OF ARRAYS: one flat array per scalar of a record, `n_rec` long,
and every Cartesian quantity split per axis -- `XPA`, `YPA`, `ZPA` rather
than one `XPA[n_rec, 3]`. With one thread per record, the 32 lanes of a warp
then read 32 consecutive doubles of one array on every load, where the
interleaved `[n_rec, 3]` row put them 24 bytes apart.

Those arrays cross the call boundary as ONE argument, a struct of pointers
passed by value, so a kernel's signature is

    s_00(int n_rec, int n_ao, PrimPairSoA pp, double* S)

and not fourteen positional pointers a host can transpose without a
diagnostic (every one is a `const double*`). The struct is the SUPERSET the
four families read: a kernel binds the fields it uses and leaves the rest
unread, so one value, filled once per class, serves every family.

`include/felizjueves/prim_pair.h` is its C spelling for a `gpu` unit, and a
test holds that header to `emit._c.bundle_struct(PRIM_PAIR_SOA)` byte for
byte; the vendor backends spell the same text in the unit.

THE FIELDS, in declaration order (the ABI; a host fills them by name):

    zeta            alpha + beta
    s               c_a c_b (pi / zeta)^{3/2}, the host's one scalar
    eb              beta, the KET primitive's exponent (`T`'s shift)
    XP  YP  ZP      P, the product centre (`V`'s P - C)
    XPA YPA ZPA     P - A
    XPB YPB ZPB     P - B
    off_a off_b     the two shells' first AO (int)
"""

from __future__ import annotations

from felizjueves.ir import Bundle, Bundled, Scalar

__all__ = ["AXES", "PRIM_PAIR_SOA", "bundled"]

AXES = ("X", "Y", "Z")

PRIM_PAIR_SOA = Bundle(
    "PrimPairSoA", "felizjueves/prim_pair.h",
    (("zeta", Scalar.F64), ("s", Scalar.F64), ("eb", Scalar.F64),
     *((f"{a}P", Scalar.F64) for a in AXES),
     *((f"{a}PA", Scalar.F64) for a in AXES),
     *((f"{a}PB", Scalar.F64) for a in AXES),
     ("off_a", Scalar.I32), ("off_b", Scalar.I32)))


def bundled(*fields: str, arg: str = "pp", **renamed: str) -> Bundled:
    """`PRIM_PAIR_SOA` carrying each of `fields` as the param of the same
    name, and each `renamed` field as the param it names (`zeta="zeta_ab"`),
    in the struct's field order."""
    want = dict.fromkeys(fields, None) | renamed
    binds = tuple((f, want[f] or f) for f, _ in PRIM_PAIR_SOA.fields
                  if f in want)
    missing = set(want) - {f for f, _ in binds}
    if missing:
        raise ValueError(f"{sorted(missing)} are not fields of PrimPairSoA")
    return Bundled(PRIM_PAIR_SOA, arg, binds)
