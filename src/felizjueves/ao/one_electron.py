"""The one-electron assembly: a symmetric `n_ao x n_ao` matrix from shell blocks.

    matrix(basis, block) -> M     M[m][n], a list of n_ao lists of n_ao floats

ONE FUNCTION, SHARED BY `S`, `T` AND `V` (milestone 11, #233). Each of the
three one-electron matrices differs only in the shell-pair block it computes;
the loop that places those blocks is this one, and `ao/overlap.py` (`S`),
the kinetic (`T`, #234) and the nuclear attraction (`V`, #235) take it as
given rather than writing their own.

THE BLOCK CONTRACT. `block(sa, sb)` is called with two shells of `basis` and
returns `n_cartesians(sa.l)` rows of `n_cartesians(sb.l)` floats each:
`block(sa, sb)[i][j] = <a_i | O | b_j>` for `a_i` the i-th component of
`cartesians(sa.l)` and `b_j` the j-th of `cartesians(sb.l)` -- the row-major
`cartesians(la) x cartesians(lb)` layout `ao.pairs.PrimitivePair.block` uses,
`a` outer. A block of any other shape is refused (`ValueError`), never padded
or truncated.

THE LAYOUT. Shell `s` of `basis` occupies the `n_cartesians(s.l)` contiguous
AOs starting at `basis.offsets[s]`, in `Basis` order -- the order the libcint
fixtures use (`test/oracle/basis.py`), so the result compares to `int1e_*`
with no permutation.

SYMMETRY IS USED, NOT CHECKED. For real Gaussians and a Hermitian operator
`<a|O|b> = <b|O|a>`, so `block` is called once per UNORDERED shell pair --
`(basis.shells[i], basis.shells[j])` for `i <= j` only, i.e. the upper
triangle by shell -- and the lower triangle is the transpose, COPIED, not
recomputed. On a diagonal pair `i == i` only the block's upper triangle
(`row <= col`) is read and its lower triangle is the mirror, so the whole
matrix is symmetric EXACTLY (bitwise), whatever rounding the block's own
evaluation carries. That is `kernel_jab`'s diagonal rule, read from the other
side (`ao.j.jab_from_host` keeps the lower triangle; which triangle a
symmetric block is read from is immaterial to the result, only that one is).

Plain lists and floats, like `ao/j.py`: `src/felizjueves` has no runtime
dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from felizjueves.ao.shell import Basis, Shell

Block = Callable[[Shell, Shell], Sequence[Sequence[float]]]


def matrix(basis: Basis, block: Block) -> list[list[float]]:
    """The symmetric `basis.n_ao x basis.n_ao` matrix assembled from `block`.

    `block(sa, sb)` is called for every shell pair `i <= j` of `basis`, once,
    in row-major shell order (`i` outer); the module docstring has its
    contract and the symmetry rule.
    """
    if not isinstance(basis, Basis):
        raise ValueError(f"{basis!r} is not a Basis")
    n = basis.n_ao
    M = [[0.0] * n for _ in range(n)]
    shells, off = basis.shells, basis.offsets
    for i, sa in enumerate(shells):
        for j in range(i, len(shells)):
            sb = shells[j]
            b = block(sa, sb)
            na, nb = sa.n_cart, sb.n_cart
            if len(b) != na or any(len(row) != nb for row in b):
                shape = (len(b), *sorted({len(row) for row in b}))
                raise ValueError(
                    f"block(shell {i}, shell {j}) has shape {shape}, not "
                    f"({na}, {nb}): rows are cartesians(la), columns "
                    "cartesians(lb)")
            oa, ob = off[i], off[j]
            for r in range(na):
                for c in range(nb):
                    if i == j and c < r:
                        continue  # a diagonal pair: its upper triangle only
                    v = float(b[r][c])
                    M[oa + r][ob + c] = v
                    M[ob + c][oa + r] = v
    return M
