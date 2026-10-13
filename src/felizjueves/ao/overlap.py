"""The overlap matrix `S_ab = <a|b>`, as the `t = 0` column of the dressed block.

    overlap_block(sa, sb) -> S_ab    n_cart(sa) rows of n_cart(sb) floats
    overlap_matrix(basis) -> S       ao.one_electron.matrix over overlap_block

THE IDENTITY (milestone 11, #233). A Cartesian primitive product expands in
Hermite Gaussians, `G_a G_b = SUM_t E^{ab}_t Lambda_t(zeta, P)`
(McMurchie & Davidson, docs/references.md entry 46), and every
`Lambda_t` with `t != 0` is a derivative of a Gaussian and integrates to zero
over all space. Only `t = 0` survives, and `INT Lambda_0 = (pi/zeta)^{3/2}`:

    S_ab = SUM_pp c_a c_b (pi/zeta)^{3/2} E^{ab}_0
         = SUM_pp dress(e_block(a, b, zeta, P, A, B), ca, cb, zeta)[0]

i.e. ELEMENT 0 OF THE BLOCK `ao.pairs.primitive_pairs` ALREADY DRESSES
(`hermite_order(L)[0]` is `t = (0, 0, 0)` for every `L`), summed over the
primitive pairs. This module forms no constant and introduces no dressing.

TWO THINGS THE PREDECESSOR DOES THAT MUST NOT BE DONE HERE.

  (1) No `K_AB` reinstatement. Our `E` seed is
      `E^{00}_0 = exp(zeta X_PA X_PB) = exp(-mu X_AB^2)` per axis
      (`ao/hermite.py`), so the three-axis product already IS
      `K_AB = exp(-mu |A - B|^2)`. Multiplying it in again -- as
      `md/um26/sgen/overlap.py` does on a bare `E` -- would square it: a
      smooth per-primitive-pair factor, which test/test_ao_overlap.py's
      two-centre closed form exists to catch.
  (2) No `primitive_norm`. A `Shell`'s coefficients multiply UNNORMALISED
      primitives and any normalisation is already in them (#85,
      ao/shell.py's module docstring).

THE SUM ORDER. Primitive pairs in `primitive_pairs` order (`i` outer), added
left to right from `0.0` -- one record at a time, the order a one-thread-per-
record kernel accumulates in when it is interpreted, so that kernel can be
held to this bitwise.
"""

from __future__ import annotations

from felizjueves.ao.one_electron import matrix
from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis, Shell


def overlap_block(sa: Shell, sb: Shell) -> tuple[tuple[float, ...], ...]:
    """`<a|b>` for every `a` in `cartesians(sa.l)` (rows) and `b` in
    `cartesians(sb.l)` (columns): the summed `t = 0` column of the dressed
    blocks of `primitive_pairs(sa, sb)` (module docstring)."""
    nb = sb.n_cart
    acc = [0.0] * (sa.n_cart * nb)
    for pp in primitive_pairs(sa, sb):
        for k, row in enumerate(pp.block):
            acc[k] += row[0]
    return tuple(tuple(acc[r * nb:(r + 1) * nb]) for r in range(sa.n_cart))


def overlap_matrix(basis: Basis) -> list[list[float]]:
    """`S`, `basis.n_ao x basis.n_ao`, exactly symmetric
    (`ao.one_electron.matrix`)."""
    return matrix(basis, overlap_block)
