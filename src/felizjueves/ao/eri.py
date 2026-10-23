"""The Cartesian ERI block `(ab|cd)` of a shell quartet, by the Hermite path.

    eri_block(bra_pair, ket_pair, *, pq)            -> (ab|cd)
    eri_block_and_scale(bra_pair, ket_pair, *, pq)  -> ((ab|cd), scale)

For an ordered bra shell pair `(sa, sb)` and ket shell pair `(sc, sd)`,

    (ab|cd) = sum_{bra prims} sum_{ket prims}
              sum_p sum_q E^{ab}_p E^{cd}_q [p|q],
    [p|q]   = (-1)^{|q|} [p+q]^{(0)}(PQ = P - Q, theta)

with `E` the DRESSED coefficients of `ao.pairs.primitive_pairs` (contraction
coefficient, primitive normalisation and `(pi/zeta)^{3/2}` already in, by
`pairs.dress` and nowhere else), `p` over `hermite_order(la + lb)` and `q`
over `hermite_order(lc + ld)`. Nothing in this module multiplies a constant
in: `(2/sqrt(pi)) theta` lives inside the evaluator's `[0]^{(m)}`, so with the
two pair factors already on `E` the product is McMurchie & Davidson's
`lambda` (`ao/pairs.py`'s module docstring), and the sum is the integral.

THE SIGN `(-1)^{|q|}` belongs to the KET HERMITE INDEX `q` -- not to the ket
AO pair and not to `|c| + |d|` -- and it is the evaluator's: `pq(p, q, ...)`
is `[p|q]` with the sign already applied, so this module never forms `p + q`
and never computes a sign. The three index orderings (`p`, `q`, `p + q`) are
kept apart by passing the index TUPLES to the evaluator, never by arithmetic
on positions: `p` and `q` are enumerated from `hermite_order` in the same
order the `E` blocks' columns are, and the evaluator adds them itself.

`theta^2 = zeta eta / (zeta + eta)` is the convention `reference/pq.py`
(`fundamental`) pins; it is restated here because `ao` may not import
`reference` (test/test_layering.py, rule "ao"), and the oracle comparison in
test/test_ao_eri.py is what holds the two to each other.

THE EVALUATOR IS INJECTED, as in `ao/pairs.py`: `pq(p, q, PQ, theta) ->
[p|q]` arrives as an argument (tests pass `reference.pq.pq`).

THE SCALE. `scale` is the same sum with every term replaced by its absolute
value, `sum |E^{ab}_p| |E^{cd}_q| |[p|q]|`, element by element: the size of
what was added, so that a tolerance relative to it is a statement about
rounding, not about how much the sum happened to cancel.

THE LAYOUT. Both results are tuples of rows: row `ab` runs over
`cartesians(la) x cartesians(lb)` (a outer), column `cd` over
`cartesians(lc) x cartesians(ld)` (c outer) -- the row order of
`PrimitivePair.block` on each side, so `block[ab][cd]` reshapes to
`[a][b][c][d]` in AO order.

Pure Python floats, like the rest of `ao/`: `src/felizjueves` has no runtime
dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from felizjueves.ao.pairs import PrimitivePair, primitive_pairs
from felizjueves.ao.shell import Shell
from felizjueves.mmd.hermite import hermite_order

Vec = tuple[float, float, float]
Ang = tuple[int, int, int]
PQ = Callable[[Ang, Ang, Vec, float], float]
Block = tuple[tuple[float, ...], ...]


def _matmul(a: Sequence[Sequence[float]],
            b: Sequence[Sequence[float]]) -> list[list[float]]:
    """`a @ b`, row by row; `b` is read by rows so the inner loop is a zip."""
    if not a:
        return []
    out = []
    for row in a:
        acc = [0.0] * len(b[0])
        for x, brow in zip(row, b, strict=True):
            if x:
                for k, y in enumerate(brow):
                    acc[k] += x * y
        out.append(acc)
    return out


def _transpose(a: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a, strict=True)]


def _quartet(bra: PrimitivePair, ket: PrimitivePair, ps: Sequence[Ang],
             qs: Sequence[Ang], pq: PQ, want_scale: bool):
    """One primitive quartet: `E_bra [p|q] E_ket^T`, and its |.| twin."""
    zeta, eta = bra.zeta, ket.zeta
    theta = math.sqrt(zeta * eta / (zeta + eta))
    PQv = (bra.P[0] - ket.P[0], bra.P[1] - ket.P[1], bra.P[2] - ket.P[2])
    m = [[pq(p, q, PQv, theta) for q in qs] for p in ps]
    # (E_bra M) E_ket^T, with E_ket^T read by rows: M E_ket^T = M @ ket^T.
    ket_t = _transpose(ket.block)
    blk = _matmul(bra.block, _matmul(m, ket_t))
    if not want_scale:
        return blk, None
    am = [[abs(x) for x in row] for row in m]
    abra = [[abs(x) for x in row] for row in bra.block]
    aket_t = [[abs(x) for x in row] for row in ket_t]
    return blk, _matmul(abra, _matmul(am, aket_t))


def _accumulate(into: list[list[float]] | None,
                add: list[list[float]]) -> list[list[float]]:
    if into is None:
        return add
    for r_into, r_add in zip(into, add, strict=True):
        for k, v in enumerate(r_add):
            r_into[k] += v
    return into


def eri_block_and_scale(bra_pair: tuple[Shell, Shell],
                        ket_pair: tuple[Shell, Shell], *,
                        pq: PQ) -> tuple[Block, Block]:
    """`(ab|cd)` and its absolute-value twin `scale` (module docstring)."""
    sa, sb = bra_pair
    sc, sd = ket_pair
    bras = primitive_pairs(sa, sb)
    kets = primitive_pairs(sc, sd)
    ps = hermite_order(sa.l + sb.l)
    qs = hermite_order(sc.l + sd.l)
    total = scale = None
    for bra in bras:
        for ket in kets:
            blk, sc_ = _quartet(bra, ket, ps, qs, pq, True)
            total = _accumulate(total, blk)
            scale = _accumulate(scale, sc_)
    return (tuple(tuple(r) for r in total), tuple(tuple(r) for r in scale))


def eri_block(bra_pair: tuple[Shell, Shell], ket_pair: tuple[Shell, Shell], *,
              pq: PQ) -> Block:
    """`(ab|cd)` of the ordered quartet `(sa sb|sc sd)`: rows `ab`, columns
    `cd`, each in `cartesians` order with the first index outer."""
    sa, sb = bra_pair
    sc, sd = ket_pair
    ps = hermite_order(sa.l + sb.l)
    qs = hermite_order(sc.l + sd.l)
    kets = primitive_pairs(sc, sd)
    total = None
    for bra in primitive_pairs(sa, sb):
        for ket in kets:
            blk, _ = _quartet(bra, ket, ps, qs, pq, False)
            total = _accumulate(total, blk)
    return tuple(tuple(r) for r in total)
