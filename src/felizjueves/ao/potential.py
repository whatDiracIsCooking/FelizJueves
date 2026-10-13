"""The nuclear attraction `V_ab = -SUM_C Z_C <a|1/r_C|b>`, as the point-charge
limit of this engine's own `[p|q]`.

    potential_block(sa, sb, nuclei, *, one_centre) -> V_ab   n_cart(sa) x n_cart(sb)
    potential_matrix(basis, nuclei, *, one_centre) -> V      ao.one_electron.matrix

A nucleus is `(Z, centre)`: a charge `Z` and a point `centre = (x, y, z)`.
The attractive sign `-Z_C` is folded in HERE and nowhere else.

THE IDENTITY (milestone 11, #235). Expand the primitive product in Hermite
Gaussians, `G_a G_b = SUM_t E^{ab}_t Lambda_t(zeta, P)` (McMurchie & Davidson,
docs/references.md entry 46), and let the ket be a unit point charge at `C`.
That is the `eta -> inf` limit of a `J` ket, so the convention
`theta^2 = zeta eta / (zeta + eta)` gives `theta -> sqrt(zeta)`, and with
`PQ = P - C` the J engine's own base (`reference.pq.fundamental`) is

    [0]^(m)(P - C, sqrt(zeta)) = (2/sqrt(pi)) sqrt(zeta) (-2 zeta)^m F_m(zeta |P-C|^2)

-- the physical Hermite Coulomb base `(-2p)^m F_m` times one constant. That
constant is completed EXACTLY by the `(pi/zeta)^{3/2}` that `ao.pairs.dress`
already multiplies into every bra block:

    (2/sqrt(pi)) sqrt(zeta) (pi/zeta)^{3/2} = 2 pi / zeta

the one-electron prefactor (asserted by test/test_ao_potential.py, the
analogue of test_ao_pairs.py's `lambda`). So

    V_ab = -SUM_C Z_C SUM_pp SUM_t  block_pp[ab][t] * [t]^(0)(P - C, sqrt(zeta))

over the FULL Hermite axis `hermite_order(la + lb)` of the dressed block
`ao.pairs.primitive_pairs` produces (the overlap read only `t = 0`). This
module forms NO constant: no `2 pi / p`, no second `(pi/p)`, no `K_AB`
(the `E` seed carries it per axis, ao/overlap.py's note (1)).

THE EVALUATOR IS INJECTED. `one_centre(r, m, pq, theta) -> [r]^{(m)}` has the
signature `ao.pairs.OneCentre`; tests pass `reference.pq.one_centre`. Importing
`reference` here would make `V` share ancestry with the recurrence it is
checked against (test/test_layering.py, rule "ao"). No new reference code.

THE SIGN. The bra's Hermite derivatives are in `P`, and `PQ = P - C` enters
`[t]` with the sign of `P`, so `[t]^{(0)}(P - C)` is `+(d/dP)^t` of the base
-- the `q = 0` case of `[p|q] = (-1)^{|q|} [p + q]^{(0)}`, no flip.

THE SUM ORDER. Primitive pairs in `primitive_pairs` order (`i` outer), then
nuclei in the order given; per (pair, nucleus) one `[t]^{(0)}` per Hermite
index, then per Cartesian row `v = SUM_t block[t] * R[t]` left to right from
`0.0`, and the row accumulates `(-Z) * v`, from `0.0`. Nuclei are serial
inside the record -- the order a one-thread-per-record kernel accumulates in
when interpreted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from felizjueves.ao.hermite import hermite_order
from felizjueves.ao.one_electron import matrix
from felizjueves.ao.pairs import OneCentre, primitive_pairs
from felizjueves.ao.shell import Basis, Shell

Vec = tuple[float, float, float]
Nucleus = tuple[float, Vec]


def _check_nuclei(nuclei: Sequence[Nucleus]) -> tuple[Nucleus, ...]:
    out = []
    for n in nuclei:
        if len(n) != 2:
            raise ValueError(f"nucleus {n!r} is not (Z, centre)")
        Z, c = n
        if len(c) != 3:
            raise ValueError(f"nucleus centre {c!r} is not three coordinates")
        out.append((float(Z), (float(c[0]), float(c[1]), float(c[2]))))
    return tuple(out)


def potential_block(sa: Shell, sb: Shell, nuclei: Sequence[Nucleus], *,
                    one_centre: OneCentre) -> tuple[tuple[float, ...], ...]:
    """`-SUM_C Z_C <a|1/r_C|b>` for every `a` in `cartesians(sa.l)` (rows)
    and `b` in `cartesians(sb.l)` (columns), over the full Hermite axis of the
    dressed blocks of `primitive_pairs(sa, sb)` (module docstring)."""
    nuc = _check_nuclei(nuclei)
    ts = hermite_order(sa.l + sb.l)
    nb = sb.n_cart
    acc = [0.0] * (sa.n_cart * nb)
    for pp in primitive_pairs(sa, sb):
        theta = math.sqrt(pp.zeta)
        for Z, C in nuc:
            pc = (pp.P[0] - C[0], pp.P[1] - C[1], pp.P[2] - C[2])
            R = [one_centre(t, 0, pc, theta) for t in ts]
            for k, row in enumerate(pp.block):
                v = 0.0
                for e, r in zip(row, R, strict=True):
                    v += e * r
                acc[k] += -Z * v
    return tuple(tuple(acc[r * nb:(r + 1) * nb]) for r in range(sa.n_cart))


def potential_matrix(basis: Basis, nuclei: Sequence[Nucleus], *,
                     one_centre: OneCentre) -> list[list[float]]:
    """`V`, `basis.n_ao x basis.n_ao`, exactly symmetric
    (`ao.one_electron.matrix`)."""
    nuc = _check_nuclei(nuclei)
    return matrix(basis, lambda sa, sb: potential_block(
        sa, sb, nuc, one_centre=one_centre))
