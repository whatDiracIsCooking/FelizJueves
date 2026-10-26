"""The McMurchie-Davidson `E` coefficients as device code: two `Fragment`s.

`E` is the first quantity built as an `ir.Fragment` (#149, #152). It is the
UNDRESSED coefficient `mmd/hermite.py` defines -- no contraction coefficient,
no primitive normalisation, no `(pi/zeta)^{3/2}`; those are `ao/pairs.py`'s
`dress`, and a fragment carrying one would break the pointwise Hermite
identity it is checked by (test/test_e_fragment.py).

- `axis_fragment(la, lb, d)`: `E^{ij}_t` on axis `d` for every `i <= la`,
  `j <= lb`, `t <= i+j`, one scalar local per entry (`E{x,y,z}_i_j_t`). The
  seed is `E^{00}_0 = exp(zeta X_PA X_PB)` -- formed from the `(X_PA, X_PB)`
  pair, as `mmd/hermite.py`'s module docstring argues, so `mu X_AB^2` is never
  a separate input. Every other entry is ONE `Weighted`:

      E^{i+1,j}_t = (1/2zeta) E^{ij}_{t-1} + X_PA E^{ij}_t + (t+1) E^{ij}_{t+1}
      E^{i,j+1}_t = (1/2zeta) E^{ij}_{t-1} + X_PB E^{ij}_t + (t+1) E^{ij}_{t+1}

  with a term whose source has `t < 0` or `t > i+j` ABSENT, not a zero: an
  entry with `t > i+j` is never declared. `i` is raised first through
  `X_PA`, then `j` through `X_PB`, the order `mmd.hermite.e_axis` uses, so the
  fragment performs the reference's operations in the reference's order.

- `block_fragment(la, lb)`: the three axes and the products
  `E^{ab}_t = E_x E_y E_z`, written into the array bind `E[row, col]` --
  one row per Cartesian pair in `cartesians(la) x cartesians(lb)` order (`a`
  outer), one column per `t` in `hermite_order(la + lb)` order:
  `ao.pairs.PrimitivePair.block`'s layout. An entry with `t_d > a_d + b_d`
  on some axis is exactly zero and is NOT written -- the consumer owns those
  slots -- so no computed zero is ever emitted.

- `axes_fragment(la, lb)` (#569): the three axes and NO products -- every
  `E^{ij}_t` written into the 1D array bind `EA` at `axes_slot(la, lb, d,
  i, j, t)` (axis outer, then `i`, `j`, `t`; `axes_size(la, lb)` slots, 336
  at `(f, f)` against the block's 8,400). The FACTORED `E`: a consumer forms
  `E^{ab}_t` where it reads it, `axes_product` -- `(E_x E_y) E_z`, the
  block's association, so every value is `block_fragment`'s bit for bit.
  It is what the K kernels hold (`k_exchange.levers.seams`), where the
  composed block per record side was 134,504 B of locals a thread at
  `(ff|ff)`.

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

- `kinetic_fragment(la, lb)` (#238): the three axes at `lb + 2` -- the same
  `_axis_body(la, lb + 2, d)`, unchanged -- then the fixed ket shift per
  axis, one local `K{x,y,z}_i_j` per `i <= la`, `j <= lb`,

      K_ij = -2 eb^2 E^{i,j+2}_0 + eb (2j+1) E^{ij}_0 - 1/2 j(j-1) E^{i,j-2}_0

  and the three-term product `K_x S_y S_z + S_x K_y S_z + S_x S_y K_z` (with
  `S_d = E^{a_d b_d}_0`), written into the 1D array bind `KE` in
  `block_rows` order. The `j(j-1)` term is ABSENT for `j < 2`, not a zero
  weight -- the fragment's convention for an out-of-range source. Every
  operation is `one_electron.kinetic`'s, in its association: `K_ij` is
  `((-2.0 eb) eb) E^{i,j+2}_0 + (eb (2j+1)) E^{ij}_0`, then minus
  `(1/2 j(j-1)) E^{i,j-2}_0` (`kinetic_axis`), and the product is
  `((K_x S_y) S_z + (S_x K_y) S_z) + (S_x S_y) K_z`. `eb` -- the KET
  primitive's exponent -- is a scalar bind of its own: the shift is the one
  place a single primitive's exponent enters a fragment, so it is bound,
  not reconstructed from `zeta`. Only the ket widens; the entries at
  `j = lb + 1, lb + 2` are read ONLY through `K_ij` (the products read
  `S_d` at `j <= lb`), and dead entries are dropped by `_live` as in the
  overlap fragment.

- `multipole_fragment(la, lb, e)` (#519): the Cartesian multipole moments
  of order `e` about a point `C`, by the Hermite route. The three axes
  (`_axes_body(la, lb)`, unchanged), then per axis `X_PC = P_d - C_d` (one
  local `XPC_{x,y,z}` from the scalar binds `P_d`, `C_d`), the Hermite
  moments `M^{e'}_t` for `e' <= e`, `t <= e'` (one local `Mt{x,y,z}_e_t`),

      M^0_0 = 1,   M^{e+1}_t = t M^e_{t-1} + X_PC M^e_t + M^e_{t+1} / (2 zeta),

  (`M^e_t = int x_C^e Lambda_t dx` without its `sqrt(pi/zeta)`, which is
  `dress`'s), a term with `t - 1 < 0` or `t + 1 > e` ABSENT, and the
  per-axis moment

      M^{(e')}_{d,ij} = SUM_{t <= min(e', i+j)} E^{ij}_t M^{e'}_t,

  one local `M{x,y,z}_e_i_j`, the entries `t > i + j` DROPPED from the
  sum, never read as zero. Then one value per component `(ex, ey, ez)` in
  `cartesians(e)` and row of `block_rows(la, lb)`, `(Mx^{ex} My^{ey})
  Mz^{ez}` (the overlap product's association), written as it is formed
  into the 2D array bind `M[comp, row]` -- or handed to a consumer's
  `store(comp, row, value)` statement instead, so a kernel can `AtomicAdd`
  each value where it is formed and hold no `[n_comp, rows]` local array.
  `M^e_t` entries that are integers (`M^0_0 = 1`, `M^e_e = e!`) are folded
  at build time, never multiplied in, and a moment that is a bare local is
  an alias, not a copy: so `e = 0` is the overlap fragment's values
  bitwise, and `e = 1` is `E^{ij}_0 X_PC + E^{ij}_1`. Built general in
  `e`; `e = 1` (the dipole) is what ships. Dead entries are dropped by
  `_live`.

- `center_fragment(l)` (#410): the SINGLE-CENTER `E^{i}_t` of one
  auxiliary primitive -- a Hermite "pair" with `zeta = alpha`, `P = A`, so
  `X_PA = 0` and no `K_AB` -- per axis, one local `C{x,y,z}_i_t`, then the
  products `E^{a}_t = (C_x C_y) C_z` written into the array bind
  `E[row, col]`, rows `cartesians(l)`, columns `hermite_targets(l)`. The
  recurrence is eq. 2.20 at `X_PA = 0`,

      E^{i+1}_t = (1/2zeta) E^{i}_{t-1} + (t+1) E^{i}_{t+1},   E^0_0 = 1,

  the `X_PA` term ABSENT, not a zero weight. Then `E^i_t` is exactly zero
  whenever `i - t` is odd (each step moves `t` by one), and such an entry is
  never declared, nor read, nor written: the consumer owns those slots, as
  it owns the block's `t_d > a_d + b_d` ones. The seed is the literal `1`
  (no `exp`, so no `zeta` bind). It is a SECOND axis recurrence beside
  `_axis_body`, deliberately: `_axis_body` at `X_PA = 0` would compute
  every parity zero, and the general body has no way to drop them short
  of a rewrite. Its locals are `C`, not `E`, so it can be inlined beside
  the block fragment (the `(mn|P)` follow-up) without a name collision.

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
plain scalars too); the block adds the array `E`, the axes fragment the
array `EA`, the overlap fragment
the array `E0`, the kinetic fragment the scalar `eb` and the array `KE`,
the multipole fragment the scalars `P_d`, `C_d` (when `e >= 1`) and the
array `M` (or, with a `store`, the consumer's `store_binds`).
The center fragment binds only `half_inv_zeta` (when `l >= 1`; there
`zeta` is the aux primitive's own exponent) and the array `E`.

This is a builder: it produces vendor-neutral IR. It must not import
`mmd.hermite` (the reference it is checked against) nor the interpreter.
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
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
KE_ARRAY = "KE"
EB = "eb"  # the ket primitive's exponent: the kinetic shift's scalar


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
    order `mmd.hermite.e_block` multiplies in."""
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


AXES_ARRAY = "EA"


def axes_entries(la: int, lb: int) -> tuple[tuple[int, int, int, int], ...]:
    """Every `(d, i, j, t)` of the three axis tables, in `axes_fragment`'s
    slot order: axis outer, then `i`, `j`, `t` -- `i <= la`, `j <= lb`,
    `t <= i + j`, so no slot is a structural zero."""
    _check_am("la", la)
    _check_am("lb", lb)
    return tuple((d, i, j, t) for d in range(3) for i in range(la + 1)
                 for j in range(lb + 1) for t in range(i + j + 1))


def axes_size(la: int, lb: int) -> int:
    """Slots of the array `axes_fragment(la, lb)` writes:
    `3 SUM_{i,j} (i + j + 1)` -- 336 at `(f, f)`, against the composed
    block's `100 x 84`."""
    return len(axes_entries(la, lb))


def axes_slot(la: int, lb: int, d: int, i: int, j: int, t: int) -> int:
    """The slot of `E^{ij}_t` on axis `d` in `axes_fragment(la, lb)`'s array."""
    if not (d in (0, 1, 2) and 0 <= i <= la and 0 <= j <= lb
            and 0 <= t <= i + j):
        raise ValueError(f"E^{{{i}{j}}}_{t} on axis {d} is not an entry of "
                         f"the ({la}, {lb}) axis tables")
    per_ij = [(ii, jj) for ii in range(la + 1) for jj in range(lb + 1)]
    per_axis = sum(ii + jj + 1 for ii, jj in per_ij)
    off = sum(ii + jj + 1 for ii, jj in per_ij[:i * (lb + 1) + j])
    return d * per_axis + off + t


def axes_fragment(la: int, lb: int) -> Fragment:
    """The three undressed axis tables `E^{ij}_t`, x then y then z, written
    into the 1D array bind `EA` at `axes_slot` (#569): the FACTORED `E`.

    The same `_axes_body(la, lb)` as `block_fragment`, but NO products: a
    consumer forms `E^{ab}_t = (E_x E_y) E_z` where it reads it
    (`axes_product`), the block fragment's association, so every value it
    forms is the block's bit for bit. `axes_size(la, lb)` doubles, against
    the block's `rows x columns`: 336 against 8,400 at `(f, f)`.
    """
    body: list[Stmt] = [Note(_PROVENANCE), *_axes_body(la, lb)]
    for d, i, j, t in axes_entries(la, lb):
        body.append(Assign(Index(AXES_ARRAY, (Lit(axes_slot(la, lb, d, i, j,
                                                            t)),)),
                           Var(e_name(d, i, j, t))))
    return Fragment(f"e_axes_{la}{lb}",
                    (*_scalar_binds(la, lb, (0, 1, 2)), AXES_ARRAY),
                    tuple(body))


def axes_product(la: int, lb: int, array: str, a: tuple[int, int, int],
                 b: tuple[int, int, int], t: tuple[int, int, int]) -> Call:
    """`(E_x E_y) E_z` of Cartesians `a`, `b` at Hermite index `t`, read from
    `axes_fragment(la, lb)`'s table bound to `array`: `block_fragment`'s
    product, its operands from the table. `t_d > a_d + b_d` is the caller's
    to skip (the entry is a structural zero, never formed)."""
    x, y, z = (Index(array, (Lit(axes_slot(la, lb, d, a[d], b[d], t[d])),))
               for d in range(3))
    return Call(Fn.MUL, (Call(Fn.MUL, (x, y)), z))


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


def k_name(d: int, i: int, j: int) -> str:
    """The local holding the kinetic factor `K_ij` on axis `d`: `Kx_1_0`."""
    return f"K{_AXES[d]}_{i}_{j}"


def _mul(x: Expr, y: Expr) -> Call:
    return Call(Fn.MUL, (x, y))


def _shift(d: int, i: int, j: int) -> Decl:
    """`K_ij` on axis `d` from the widened table: `one_electron.kinetic.kinetic_axis`'s
    operations in its association, the `j(j-1)` term absent for `j < 2`."""
    def e(jj: int) -> Var:
        return Var(e_name(d, i, jj, 0))

    eb = Var(EB)
    k = Call(Fn.ADD, (_mul(_mul(_mul(Lit(-2.0), eb), eb), e(j + 2)),
                      _mul(_mul(eb, Lit(float(2 * j + 1))), e(j))))
    if j >= 2:
        k = Call(Fn.SUB, (k, _mul(Lit(0.5 * j * (j - 1)), e(j - 2))))
    return Decl(k_name(d, i, j), Scalar.F64, init=k)


def kinetic_fragment(la: int, lb: int) -> Fragment:
    """The undressed kinetic bracket `K_x S_y S_z + S_x K_y S_z + S_x S_y K_z`,
    written into the 1D array bind `KE`, one entry per row of
    `block_rows(la, lb)`, from the axis tables at `lb + 2` (module doc)."""
    _check_am("la", la)
    _check_am("lb", lb)
    body: list[Stmt] = [Note(_PROVENANCE), *_axes_body(la, lb + 2)]
    for d in range(3):
        for i in range(la + 1):
            for j in range(lb + 1):
                body.append(_shift(d, i, j))
    for row, (a, b) in enumerate(block_rows(la, lb)):
        s = [Var(e_name(d, a[d], b[d], 0)) for d in range(3)]
        k = [Var(k_name(d, a[d], b[d])) for d in range(3)]
        value = Call(Fn.ADD, (
            Call(Fn.ADD, (_mul(_mul(k[0], s[1]), s[2]),
                          _mul(_mul(s[0], k[1]), s[2]))),
            _mul(_mul(s[0], s[1]), k[2])))
        body.append(Assign(Index(KE_ARRAY, (Lit(row),)), value))
    kept, read = _live(body)
    binds = [b for b in _scalar_binds(la, lb + 2, (0, 1, 2)) if b in read]
    return Fragment(f"e_kinetic_{la}{lb}", (*binds, EB, KE_ARRAY), kept)


# ---------------------------------------------------------------------------
# The multipole moments, the Hermite route (#519)
# ---------------------------------------------------------------------------

M_ARRAY = "M"
_MULTIPOLE_PROVENANCE = (
    "Hermite moments M^e_t = int x_C^e Lambda_t (undressed): M^0_0 = 1, "
    "M^{e+1}_t = t M^e_{t-1} + X_PC M^e_t + M^e_{t+1} / (2 zeta); "
    "S^{(e)}_ij = SUM_{t <= min(e, i+j)} E^{ij}_t M^e_t")


def p_name(d: int) -> str:
    """The scalar bind holding the pair centre `P_d`: `P_x`. Upper case
    (#521) so it is not a host name: `kernel_mu` takes the per-record
    centres as the arrays `p_x p_y p_z` (`kernel_v`'s names), and `ir`
    refuses a bind that shares a host name unless bound to that very name
    -- `kernel_s`'s `zeta_ab` hygiene, from the other side."""
    return f"P_{_AXES[d]}"


def origin_name(d: int) -> str:
    """The scalar bind holding the moments' origin `C_d`: `C_x`."""
    return f"C_{_AXES[d]}"


def xpc_name(d: int) -> str:
    """The local holding `X_PC = P_d - C_d`: `XPC_x`."""
    return f"XPC_{_AXES[d]}"


def mt_name(d: int, e: int, t: int) -> str:
    """The local holding the Hermite moment `M^e_t` on axis `d`: `Mtx_2_0`."""
    return f"Mt{_AXES[d]}_{e}_{t}"


def m_name(d: int, e: int, i: int, j: int) -> str:
    """The local holding the axis moment `M^{(e)}_{ij}` on axis `d`:
    `Mx_1_0_1`."""
    return f"M{_AXES[d]}_{e}_{i}_{j}"


# A Hermite moment at build time: an int (folded, never multiplied in) or
# an `Expr` (a `Var`: a local, or `XPC_d` itself).
_Moment = int | Expr


def _scaled(c: int, src: _Moment) -> _Moment:
    """`c * src`, `c` a positive int."""
    if isinstance(src, int):
        return c * src
    return src if c == 1 else _mul(Lit(float(c)), src)


def _times(v: Expr, src: _Moment) -> _Moment:
    """`v * src`, an integer `src` of 1 folded away."""
    if isinstance(src, int):
        return v if src == 1 else _mul(v, Lit(float(src)))
    return _mul(v, src)


def _sum(parts: list[_Moment]) -> _Moment:
    """The parts added left to right, the integers folded into one."""
    const = sum(p for p in parts if isinstance(p, int))
    exprs = [p for p in parts if not isinstance(p, int)]
    if not exprs:
        return const
    out = exprs[0]
    for x in exprs[1:]:
        out = Call(Fn.ADD, (out, x))
    return Call(Fn.ADD, (out, Lit(float(const)))) if const else out


def _bind_local(body: list[Stmt], name: str, value: Expr) -> Expr:
    """`value` as a local `name`, unless it already is a bare `Var`."""
    if isinstance(value, Var):
        return value
    body.append(Decl(name, Scalar.F64, init=value))
    return Var(name)


def _hermite_moments(d: int, e: int,
                     body: list[Stmt]) -> dict[tuple[int, int], _Moment]:
    """`M^{n}_t` on axis `d` for `n <= e`, `t <= n`, appending its locals
    to `body`; an integer entry is folded, not declared."""
    xpc, hiz = Var(xpc_name(d)), Var(HALF_INV_ZETA)
    tab: dict[tuple[int, int], _Moment] = {(0, 0): 1}
    for n in range(e):  # M^{n+1} from M^n
        for t in range(n + 2):
            parts: list[_Moment] = []
            if t >= 1:
                parts.append(_scaled(t, tab[n, t - 1]))
            if t <= n:
                parts.append(_times(xpc, tab[n, t]))
            if t + 1 <= n:
                src = tab[n, t + 1]
                parts.append(_times(hiz, src) if isinstance(src, int)
                             else _mul(src, hiz))
            m = _sum(parts)
            tab[n + 1, t] = (m if isinstance(m, int)
                             else _bind_local(body, mt_name(d, n + 1, t), m))
    return tab


def multipole_components(e: int) -> tuple[tuple[int, int, int], ...]:
    """The components of order `e`, in the order `multipole_fragment` writes
    them: `cartesians(e)` (for `e = 1`: x, then y, then z)."""
    return cartesians(e)


def multipole_shape(la: int, lb: int, e: int) -> tuple[int, int]:
    """`(components, rows)` of the array `M` the fragment writes."""
    return len(multipole_components(e)), len(block_rows(la, lb))


def multipole_fragment(
    la: int, lb: int, e: int, *,
    store: Callable[[int, int, Expr], Stmt] | None = None,
    store_binds: tuple[str, ...] = (),
    from_order: int | None = None,
) -> Fragment:
    """The undressed Cartesian multipole moments of order `e` about `C`, one
    value per component of `multipole_components(e)` and row of
    `block_rows(la, lb)` (module doc).

    By default each value is written into the 2D array bind `M[comp, row]`
    where it is formed. With `store`, the statement `store(comp, row,
    value)` is placed there instead (an `AtomicAdd`, say), there is no `M`
    bind, and `store_binds` -- every name those statements read beyond
    `value` -- are the fragment's last binds.

    `from_order` (#521; default `e`) writes every order `from_order..e`,
    ascending, each order's components in `multipole_components` order and
    `comp` counting across them: `from_order = 0` at `e = 1` is the `e = 0`
    product first (component 0, `overlap_fragment`'s `(E_x E_y) E_z`), then
    x, y, z, all from the one set of axis tables. The default's fragment is
    unchanged.
    """
    _check_am("la", la)
    _check_am("lb", lb)
    _check_am("e", e)
    lo = e if from_order is None else from_order
    if type(lo) is not int or not 0 <= lo <= e:
        raise ValueError(f"from_order = {from_order!r} is not an int in "
                         f"[0, e = {e}]")
    if store is None and store_binds:
        raise ValueError("store_binds given without a store")
    body: list[Stmt] = [Note(f"{_PROVENANCE}; {_MULTIPOLE_PROVENANCE}"),
                        *_axes_body(la, lb)]
    moments: list[dict[tuple[int, int, int], Expr]] = []
    for d in range(3):
        if e:
            body.append(Decl(xpc_name(d), Scalar.F64, init=Call(
                Fn.SUB, (Var(p_name(d)), Var(origin_name(d))))))
        tab = _hermite_moments(d, e, body)
        axis: dict[tuple[int, int, int], Expr] = {}
        for n in range(e + 1):
            for i in range(la + 1):
                for j in range(lb + 1):
                    parts: list[_Moment] = []
                    for t in range(min(n, i + j) + 1):  # t > i+j: dropped
                        ev, mt = Var(e_name(d, i, j, t)), tab[n, t]
                        parts.append(_times(ev, mt) if isinstance(mt, int)
                                     else _mul(ev, mt))
                    m = _sum(parts)
                    assert not isinstance(m, int)  # every part reads an E
                    axis[n, i, j] = _bind_local(body, m_name(d, n, i, j), m)
        moments.append(axis)
    comps = [ex for n in range(lo, e + 1) for ex in multipole_components(n)]
    for comp, ex in enumerate(comps):
        for row, (a, b) in enumerate(block_rows(la, lb)):
            x, y, z = (moments[d][ex[d], a[d], b[d]] for d in range(3))
            body.append(Assign(Index(M_ARRAY, (Lit(comp), Lit(row))),
                               _mul(_mul(x, y), z)))
    kept, read = _live(body)
    sink: tuple[str, ...] = (M_ARRAY,)
    if store is not None:
        kept = tuple(
            store(st.dst.at[0].value, st.dst.at[1].value, st.value)
            if isinstance(st, Assign) else st for st in kept)
        sink = tuple(store_binds)
    candidates = [ZETA, HALF_INV_ZETA,
                  *_scalar_binds(la, lb, (0, 1, 2))[1 + bool(la + lb):],
                  *(n for d in range(3) for n in (p_name(d), origin_name(d)))]
    binds = [b for b in candidates if b in read]
    name = (f"e_multipole_{la}{lb}_{e}" if lo == e
            else f"e_multipole_{la}{lb}_{lo}_{e}")
    return Fragment(name, (*binds, *sink), kept)


_CENTER_PROVENANCE = (
    "McMurchie & Davidson report UCRL-13791 (docs/references.md entry 46), "
    "eq. 2.20 at X_PA = 0 (a single-center primitive, zeta = alpha, P = A): "
    "E^{i+1}_t from E^0_0 = 1; undressed; i - t odd entries absent")


def c_name(d: int, i: int, t: int) -> str:
    """The local holding the single-center `E^{i}_t` on axis `d`: `Cx_2_0`."""
    return f"C{_AXES[d]}_{i}_{t}"


def center_live(i: int, t: int) -> bool:
    """True when the single-center `E^{i}_t` is not a structural zero:
    `0 <= t <= i` and `i - t` even."""
    return 0 <= t <= i and (i - t) % 2 == 0


def _center_axis_body(l: int, d: int) -> list[Stmt]:  # noqa: E741
    """The single-center recurrence on axis `d`, every live `E^{i}_t` for
    `i <= l` (module doc): the seed `1`, then one `Weighted` per entry."""
    body: list[Stmt] = [Decl(c_name(d, 0, 0), Scalar.F64, init=Lit(1.0))]
    for i in range(1, l + 1):
        for t in range(i + 1):
            if not center_live(i, t):
                continue  # a parity zero: never declared
            terms = []
            if t >= 1:  # E^{i-1}_{t-1}: live, the parity is the same
                terms.append(Term(1, Factor.HALF_INV_ZETA,
                                  Var(c_name(d, i - 1, t - 1))))
            if t + 1 <= i - 1:  # E^{i-1}_{t+1}
                terms.append(Term(t + 1, Factor.ONE,
                                  Var(c_name(d, i - 1, t + 1))))
            body.append(Decl(c_name(d, i, t), Scalar.F64,
                             init=Weighted(terms)))
    return body


def center_shape(l: int) -> tuple[int, int]:  # noqa: E741
    """`(rows, columns)` of the block `center_fragment(l)` writes `E` into."""
    return len(cartesians(l)), len(hermite_targets(l))


def center_absent(a: tuple[int, int, int], t: tuple[int, int, int]) -> bool:
    """True when `E^{a}_t` is a structural zero: on some axis `t_d > a_d` or
    `a_d - t_d` odd. Such an entry is never written."""
    return not all(center_live(a[d], t[d]) for d in range(3))


def center_fragment(l: int) -> Fragment:  # noqa: E741
    """The undressed single-center block `E^{a}_t`, written into the array
    bind `E` (rows `cartesians(l)`, columns `hermite_targets(l)`); only the
    structurally nonzero entries are declared or written (module doc)."""
    _check_am("l", l)
    body: list[Stmt] = [Note(_CENTER_PROVENANCE)]
    for d in range(3):
        body += _center_axis_body(l, d)
    for row, a in enumerate(cartesians(l)):
        for col, t in enumerate(hermite_targets(l)):
            if center_absent(a, t):
                continue  # exactly zero: absent, not a computed zero
            x, y, z = (Var(c_name(d, a[d], t[d])) for d in range(3))
            body.append(Assign(Index(E_ARRAY, (Lit(row), Lit(col))),
                               _mul(_mul(x, y), z)))
    kept, read = _live(body)
    binds = [b for b in (HALF_INV_ZETA,) if b in read]
    return Fragment(f"e_center_{l}", (*binds, E_ARRAY), kept)
