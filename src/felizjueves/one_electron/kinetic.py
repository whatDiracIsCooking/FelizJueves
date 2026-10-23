"""The kinetic-energy matrix `T_ab = <a| -1/2 grad^2 |b>`, by a fixed ket shift.

    kinetic_axis(s, j, beta)   -> K_ij  one axis, from the overlap row S_{i,.}
    kinetic_block(sa, sb)      -> T_ab  n_cart(sa) rows of n_cart(sb) floats
    kinetic_matrix(basis)      -> T     ao.one_electron.matrix over kinetic_block

THE IDENTITY (milestone 11, #234). The Laplacian is separable, and on one
axis of a Cartesian ket `x_B^j exp(-beta x_B^2)`

    -1/2 d^2/dx^2  ->  -2 beta^2 x_B^{j+2} + beta (2j+1) x_B^j
                       - 1/2 j(j-1) x_B^{j-2}

(times the same Gaussian), so the one-dimensional kinetic factor is a fixed
combination of one-dimensional OVERLAP factors of the same bra:

    K_ij = -2 beta^2 S_{i,j+2} + beta (2j+1) S_{i,j} - 1/2 j(j-1) S_{i,j-2}
    T_ab = K_x S_y S_z + S_x K_y S_z + S_x S_y K_z

The `S_{i,j-2}` term EXISTS only for `j >= 2`; for `j = 0, 1` it is absent,
not multiplied by a zero weight (`kinetic_axis` never reads column `j - 2`
there, so a negative index cannot wrap round to the far end of the row).

THE AXIS TABLES. Per axis, `S_{ij}` is `ao.hermite.e_axis(i, j, zeta, pa,
pb)[0]` -- the `t = 0` entry, every other `Lambda_t` integrating to zero --
for `i = 0..la` and `j = 0..lb + 2`: the table is built at `lb + 2` so that
`S_{i,j+2}` is in range, and the overlap is its first `lb + 1` columns. Only
the ket widens; `i` is never shifted. The `sqrt(pi/zeta)` each axis's true
overlap carries is NOT in the table: the three of them are the pair factor.

THE DRESSING. `c_a c_b (pi/zeta)^{3/2}`, once per primitive pair, multiplies
the whole three-term bracket through `ao.pairs.dress` (the only place either
dressing multiplies in) -- every term of the bracket carries exactly one
factor per axis, so one `(pi/zeta)^{3/2}` is right for all three.

NO `K_AB` REINSTATEMENT. The `e_axis` seed is `exp(zeta X_PA X_PB) =
exp(-mu X_AB^2)`, each axis's share of the Gaussian-product constant, so the
product across axes already carries `K_AB` -- in every term, which is why the
product rule across axes is clean here. The predecessor
(`md/um26/tgen/kinetic.py`) keeps bare tables and multiplies `K_AB` back in;
doing that here would square it (test/test_ao_kinetic.py's two-centre closed
form catches it). No `primitive_norm` either: normalisation is already in a
`Shell`'s coefficients (ao/shell.py).

THE SUM ORDER. Primitive pairs in `ao.pairs.primitive_pairs` order (`i`
outer), each pair's dressed block added left to right from `0.0` -- the order
`ao.overlap` uses, and the one a one-thread-per-record kernel accumulates in
when interpreted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from felizjueves.ao.pairs import dress
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.mmd.hermite import e_axis
from felizjueves.one_electron.one_electron import matrix


def kinetic_axis(s: Sequence[float], j: int, beta: float) -> float:
    """`K_ij` on one axis from `s[k] = S_{i,k}` (`k = 0..j+2` at least).

    The `S_{i,j-2}` term is taken only for `j >= 2` (module docstring): for
    `j = 0, 1` column `j - 2` is never read.
    """
    if type(j) is not int or j < 0:
        raise ValueError(f"j = {j!r} is not an int >= 0")
    if len(s) < j + 3:
        raise ValueError(
            f"an overlap row of {len(s)} entries does not reach S_(i,j+2) at "
            f"j = {j}: the axis table is built at lb + 2")
    k = -2.0 * beta * beta * s[j + 2] + beta * (2 * j + 1) * s[j]
    if j >= 2:
        k -= 0.5 * j * (j - 1) * s[j - 2]
    return k


def _axis_table(la: int, lb: int, zeta: float, pa: float,
                pb: float) -> tuple[tuple[float, ...], ...]:
    """`S_ij = E^{ij}_0` for `i = 0..la`, `j = 0..lb + 2` (undressed)."""
    return tuple(
        tuple(e_axis(i, j, zeta, pa, pb)[0] for j in range(lb + 3))
        for i in range(la + 1)
    )


def kinetic_block(sa: Shell, sb: Shell) -> tuple[tuple[float, ...], ...]:
    """`<a| -1/2 grad^2 |b>` for every `a` in `cartesians(sa.l)` (rows) and
    `b` in `cartesians(sb.l)` (columns), summed over the primitive pairs of
    `(sa, sb)` (module docstring)."""
    A, B = sa.centre, sb.centre
    la, lb = sa.l, sb.l
    carts = [(a, b) for a in cartesians(la) for b in cartesians(lb)]
    nb = sb.n_cart
    acc = [0.0] * len(carts)
    for alpha, ca in zip(sa.exponents, sa.coefficients, strict=True):
        for beta, cb in zip(sb.exponents, sb.coefficients, strict=True):
            zeta = alpha + beta
            P = tuple((alpha * A[d] + beta * B[d]) / zeta for d in range(3))
            S = [_axis_table(la, lb, zeta, P[d] - A[d], P[d] - B[d])
                 for d in range(3)]
            bracket = []
            for a, b in carts:
                s = [S[d][a[d]][b[d]] for d in range(3)]
                k = [kinetic_axis(S[d][a[d]], b[d], beta) for d in range(3)]
                bracket.append(k[0] * s[1] * s[2] + s[0] * k[1] * s[2]
                               + s[0] * s[1] * k[2])
            for n, v in enumerate(dress(bracket, ca=ca, cb=cb, zeta=zeta)):
                acc[n] += v
    return tuple(tuple(acc[r * nb:(r + 1) * nb]) for r in range(sa.n_cart))


def kinetic_matrix(basis: Basis, *,
                   pairs: Iterable[tuple[int, int]] | None = None
                   ) -> list[list[float]]:
    """`T`, `basis.n_ao x basis.n_ao`, exactly symmetric
    (`ao.one_electron.matrix`); `pairs` screens it (#236)."""
    return matrix(basis, kinetic_block, pairs=pairs)
