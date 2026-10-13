"""The McMurchie-Davidson `E` coefficients as device code: two `Fragment`s.

`E` is the first quantity built as an `ir.Fragment` (#149, #152). It is the
UNDRESSED coefficient `ao/hermite.py` defines -- no contraction coefficient,
no primitive normalisation, no `(pi/zeta)^{3/2}`; those are `ao/pairs.py`'s
`dress`, and a fragment carrying one would break the pointwise Hermite
identity it is checked by (test/test_e_fragment.py).

- `axis_fragment(la, lb, d)`: `E^{ij}_t` on axis `d` for every `i <= la`,
  `j <= lb`, `t <= i+j`, one scalar local per entry (`E{x,y,z}_i_j_t`). The
  seed is `E^{00}_0 = exp(zeta X_PA X_PB)` -- formed from the `(X_PA, X_PB)`
  pair, as `ao/hermite.py`'s module docstring argues, so `mu X_AB^2` is never
  a separate input. Every other entry is ONE `Weighted`:

      E^{i+1,j}_t = (1/2zeta) E^{ij}_{t-1} + X_PA E^{ij}_t + (t+1) E^{ij}_{t+1}
      E^{i,j+1}_t = (1/2zeta) E^{ij}_{t-1} + X_PB E^{ij}_t + (t+1) E^{ij}_{t+1}

  with a term whose source has `t < 0` or `t > i+j` ABSENT, not a zero: an
  entry with `t > i+j` is never declared. `i` is raised first through
  `X_PA`, then `j` through `X_PB`, the order `ao.hermite.e_axis` uses, so the
  fragment performs the reference's operations in the reference's order.

- `block_fragment(la, lb)`: the three axes and the products
  `E^{ab}_t = E_x E_y E_z`, written into the array bind `E[row, col]` --
  one row per Cartesian pair in `cartesians(la) x cartesians(lb)` order (`a`
  outer), one column per `t` in `hermite_order(la + lb)` order:
  `ao.pairs.PrimitivePair.block`'s layout. An entry with `t_d > a_d + b_d`
  on some axis is exactly zero and is NOT written -- the consumer owns those
  slots -- so no computed zero is ever emitted.

- `overlap_fragment(la, lb)` (#237): the three axes and the `t = 0`
  products ONLY, `E0[row] = E_x E_y E_z` at `t = (0, 0, 0)`, written into
  the 1D array bind `E0` in `block_rows` order -- column 0 of the block,
  since `hermite_order(L)[0]` is `t = (0, 0, 0)` for every `L`. It is what
  `S` reads (`kernel_s.py`), and it is NOT `block_fragment`: the block's
  other `(L+1)(L+2)(L+3)/6 - 1` columns would be dead code in every `S`
  kernel. No `t = 0` entry is ever structurally zero (`0 <= a_d + b_d`), so
  every row is written. The axis entries are `_axis_body`'s, with the ones
  nothing reads DROPPED (`_live`): `E^{i+1,j}_0` still reads `E^{ij}_1`,
  but e.g. `E^{la,lb}_{t>0}` feeds nothing a `t = 0` product needs, and a
  declared-but-unread local is an `-Wunused-variable` on hipcc and a
  `#177-D` on nvcc (#92's ratchet). Dropping a dead `Decl` changes no value
  that is kept. `half_inv_zeta` is bound only when a live entry reads it
  (it is not, in (ps) and (sp)).

ONE LEVEL, COMPOSED AT BUILD TIME. `ir.Fragment` admits no `Inline` inside a
fragment (#149: one level, names only), so the block does not `Inline` the
axis fragment three times. Instead both are built from the one function that
writes the axis recurrence, `_axis_body(la, lb, d)`: the axis fragment IS that
body, and the block's body is the three instantiations of it (distinct local
names per axis, `Ex_`, `Ey_`, `Ez_`, and that axis's `Factor`s) followed by
the products. One source of truth for the recurrence, no nested `Inline`.
`overlap_fragment` is the same three instantiations (`_axes_body`) followed
by the `t = 0` products only, its dead entries dropped by `_live`; a later
product family (the kinetic ket shift, #238) is one more such entry point
over `_axes_body`, `_product` and `_live`, never a second recurrence.

The provenance `Note` is `_PROVENANCE`, written once here and placed once at
the head of each fragment -- never per axis.

Binds: `zeta` (the seed's exponent), `half_inv_zeta` (`1/2zeta`, a
`Factor`, bound once rather than divided per term -- only when some entry
is raised), and each axis's `PA_d` / `PB_d` (`Factor`s, read by the seed as
plain scalars too); the block adds the array `E`, the overlap fragment
the array `E0`.

This is a builder: it produces vendor-neutral IR. It must not import
`ao.hermite` (the reference it is checked against) nor the interpreter.
"""

from __future__ import annotations

from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Factor,
    Fn,
    Fragment,
    Index,
    Lit,
    Note,
    Scalar,
    Stmt,
    Term,
    Var,
    Weighted,
)
from felizjueves.plan.targets import hermite_targets

_PROVENANCE = ("McMurchie & Davidson report UCRL-13791 (docs/references.md "
               "entry 46), eqs. 2.20-2.21: E^{i+1,j}_t and E^{i,j+1}_t from "
               "E^{00}_0 = exp(zeta X_PA X_PB); undressed")

_AXES = "xyz"
HALF_INV_ZETA = Factor.HALF_INV_ZETA.name_in_kernel
ZETA = "zeta"
E_ARRAY = "E"
E0_ARRAY = "E0"


def _check_am(name: str, n: int) -> None:
    if type(n) is not int or n < 0:
        raise ValueError(f"{name} = {n!r} is not an int >= 0")


def cartesians(l: int) -> tuple[tuple[int, int, int], ...]:  # noqa: E741
    """Every `(i, j, k)` with `i + j + k = l`: `i` descending, then `j`.

    Restated from `ao.shell.cartesians` (a builder does not import the AO
    layer it is checked against); the test asserts the two are equal.
    """
    _check_am("l", l)
    return tuple(
        (i, j, l - i - j) for i in range(l, -1, -1) for j in range(l - i, -1, -1)
    )


def e_name(d: int, i: int, j: int, t: int) -> str:
    """The local holding `E^{ij}_t` on axis `d`: `Ex_1_0_1`, ..."""
    return f"E{_AXES[d]}_{i}_{j}_{t}"


def _raise_entry(d: int, src: tuple[int, int], dst: tuple[int, int],
                 g: Factor) -> list[Stmt]:
    """`E^{dst}_t` for `t = 0..|dst|` from `E^{src}` (one index lower)."""
    n = src[0] + src[1] + 1  # entries of the source: t = 0..n-1
    out: list[Stmt] = []
    for t in range(n + 1):
        terms = []
        if t >= 1:
            terms.append(Term(1, Factor.HALF_INV_ZETA, Var(e_name(d, *src, t - 1))))
        if t < n:
            terms.append(Term(1, g, Var(e_name(d, *src, t))))
        if t + 1 < n:
            terms.append(Term(t + 1, Factor.ONE, Var(e_name(d, *src, t + 1))))
        out.append(Decl(e_name(d, *dst, t), Scalar.F64, init=Weighted(terms)))
    return out


def _axis_body(la: int, lb: int, d: int) -> tuple[Stmt, ...]:
    """The axis recurrence on axis `d`, without provenance: the one source."""
    _check_am("la", la)
    _check_am("lb", lb)
    if d not in (0, 1, 2):
        raise ValueError(f"axis {d!r} is not 0, 1 or 2")
    pa, pb = Factor.pa(d), Factor.pb(d)
    seed = Call(Fn.EXP, (Call(Fn.MUL, (
        Call(Fn.MUL, (Var(ZETA), Var(pa.name_in_kernel))),
        Var(pb.name_in_kernel))),))
    body: list[Stmt] = [Decl(e_name(d, 0, 0, 0), Scalar.F64, init=seed)]
    for i in range(la + 1):
        if i:
            body += _raise_entry(d, (i - 1, 0), (i, 0), pa)
        for j in range(1, lb + 1):
            body += _raise_entry(d, (i, j - 1), (i, j), pb)
    return tuple(body)


def _scalar_binds(la: int, lb: int, axes: tuple[int, ...]) -> tuple[str, ...]:
    binds = [ZETA]
    if la + lb:
        binds.append(HALF_INV_ZETA)
    for d in axes:
        binds += [Factor.pa(d).name_in_kernel, Factor.pb(d).name_in_kernel]
    return tuple(binds)


def axis_fragment(la: int, lb: int, d: int) -> Fragment:
    """`E^{ij}_t` on axis `d` for `i <= la`, `j <= lb`, `t <= i+j`, as locals."""
    body = (Note(_PROVENANCE), *_axis_body(la, lb, d))
    return Fragment(f"e_axis_{_AXES[d]}_{la}{lb}", _scalar_binds(la, lb, (d,)),
                    body)


def block_rows(la: int, lb: int) -> tuple[tuple[tuple[int, int, int],
                                                tuple[int, int, int]], ...]:
    """The block's rows: `cartesians(la) x cartesians(lb)`, `a` outer."""
    return tuple((a, b) for a in cartesians(la) for b in cartesians(lb))


def block_shape(la: int, lb: int) -> tuple[int, int]:
    """`(rows, columns)` of the block the fragment writes `E` into."""
    return len(block_rows(la, lb)), len(hermite_targets(la + lb))


def _axes_body(la: int, lb: int) -> list[Stmt]:
    """The three axis recurrences, x then y then z: every 3D fragment's head."""
    body: list[Stmt] = []
    for d in range(3):
        body += _axis_body(la, lb, d)
    return body


def _product(a: tuple[int, int, int], b: tuple[int, int, int],
             t: tuple[int, int, int]) -> Call:
    """`(E_x E_y) E_z` for Cartesians `a`, `b` at Hermite index `t`: the
    order `ao.hermite.e_block` multiplies in."""
    x, y, z = (Var(e_name(d, a[d], b[d], t[d])) for d in range(3))
    return Call(Fn.MUL, (Call(Fn.MUL, (x, y)), z))


def block_fragment(la: int, lb: int) -> Fragment:
    """The undressed 3D block `E^{ab}_t`, written into the array bind `E`.

    Only the structurally nonzero entries are written; see the module doc.
    """
    body: list[Stmt] = [Note(_PROVENANCE), *_axes_body(la, lb)]
    for row, (a, b) in enumerate(block_rows(la, lb)):
        for col, t in enumerate(hermite_targets(la + lb)):
            if any(t[d] > a[d] + b[d] for d in range(3)):
                continue  # exactly zero: absent, not a computed zero
            body.append(Assign(Index(E_ARRAY, (Lit(row), Lit(col))),
                               _product(a, b, t)))
    return Fragment(f"e_block_{la}{lb}",
                    (*_scalar_binds(la, lb, (0, 1, 2)), E_ARRAY), tuple(body))


def _reads(e: object) -> set[str]:
    """Every name an expression reads: `Var`s, arrays, `Factor` scalars."""
    if isinstance(e, Var):
        return {e.name}
    if isinstance(e, Index):
        return {e.array}.union(*(_reads(i) for i in e.at))
    if isinstance(e, Call):
        return set().union(*(_reads(a) for a in e.args))
    if isinstance(e, Weighted):
        out: set[str] = set()
        for t in e.terms:
            out |= _reads(t.src)
            if t.g.name_in_kernel is not None:
                out.add(t.g.name_in_kernel)
        return out
    return set()  # Lit


def _live(body: list[Stmt]) -> tuple[tuple[Stmt, ...], set[str]]:
    """`body` without the `Decl`s nothing after them reads, and the names
    the kept statements read. One backward pass over a flat body of `Decl`,
    `Assign` and `Note` -- what the fragments here are made of."""
    kept: list[Stmt] = []
    live: set[str] = set()
    for st in reversed(body):
        if isinstance(st, Decl):
            if st.name not in live:
                continue  # dead: nothing kept reads it
            live |= _reads(st.init)
        elif isinstance(st, Assign):
            live |= _reads(st.dst) | _reads(st.value)
        elif not isinstance(st, Note):
            raise TypeError(f"_live takes a flat body, got {type(st).__name__}")
        kept.append(st)
    return tuple(reversed(kept)), live


def overlap_fragment(la: int, lb: int) -> Fragment:
    """The undressed `t = 0` column `E^{ab}_0`, written into the 1D array
    bind `E0`, one entry per row of `block_rows(la, lb)` (module doc)."""
    body: list[Stmt] = [Note(_PROVENANCE), *_axes_body(la, lb)]
    for row, (a, b) in enumerate(block_rows(la, lb)):
        body.append(Assign(Index(E0_ARRAY, (Lit(row),)),
                           _product(a, b, (0, 0, 0))))
    kept, read = _live(body)
    binds = [b for b in _scalar_binds(la, lb, (0, 1, 2)) if b in read]
    return Fragment(f"e_overlap_{la}{lb}", (*binds, E0_ARRAY), kept)
