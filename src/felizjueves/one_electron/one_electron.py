"""The one-electron assembly: a symmetric `n_ao x n_ao` matrix from shell blocks.

    matrix(basis, block) -> M     M[m][n], a list of n_ao lists of n_ao floats
    rectangular(basis_a, basis_b, block) -> M    n_a lists of n_b floats

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

ACROSS TWO BASES, NO SYMMETRY. `rectangular(basis_a, basis_b, block)` is
the `n_a x n_b` matrix `M[m][n] = <m|O|n>`, `m` an AO of `basis_a` (rows)
and `n` of `basis_b` (columns) -- the cross-basis overlap `S_{B1,B2}`
(`ao.overlap.cross_overlap_matrix`). Nothing relates `M[m][n]` to any other
element, so `block` is called once per ORDERED shell pair `(i in A, j in
B)`, every element written once, nothing mirrored and nothing read from a
triangle. The block contract is `matrix`'s.

Plain lists and floats, like `ao/j.py`: `src/felizjueves` has no runtime
dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from felizjueves.ao.shell import Basis, Shell

Block = Callable[[Shell, Shell], Sequence[Sequence[float]]]


def matrix(basis: Basis, block: Block, *,
           pairs: Iterable[tuple[int, int]] | None = None
           ) -> list[list[float]]:
    """The symmetric `basis.n_ao x basis.n_ao` matrix assembled from `block`.

    `block(sa, sb)` is called for every shell pair `i <= j` of `basis`, once,
    in row-major shell order (`i` outer); the module docstring has its
    contract and the symmetry rule.

    THE SCREENED BUILD (#236). With `pairs` -- shell index pairs `(i, j)`,
    `i <= j`, typically `ao.pairs.pair_list(...).kept` -- `block` is called
    for those pairs only (still in row-major order) and every other block is
    left exactly `0.0`. `pairs=None` is every pair, and so is a `pairs` that
    lists them all: the same calls in the same order, the same matrix bitwise.
    A pair outside `0 <= i <= j < n_shells` is refused.
    """
    if not isinstance(basis, Basis):
        raise ValueError(f"{basis!r} is not a Basis")
    n = basis.n_ao
    M = [[0.0] * n for _ in range(n)]
    shells, off = basis.shells, basis.offsets
    keep = None
    if pairs is not None:
        keep = set()
        for p in pairs:
            i, j = p
            if not (type(i) is int and type(j) is int
                    and 0 <= i <= j < len(shells)):
                raise ValueError(
                    f"pair {p!r} is not (i, j) with 0 <= i <= j < "
                    f"{len(shells)} shells")
            keep.add((i, j))
    for i, sa in enumerate(shells):
        for j in range(i, len(shells)):
            if keep is not None and (i, j) not in keep:
                continue  # screened out: the block stays exactly 0.0
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


def _check_block(b, sa: Shell, sb: Shell, where: str) -> None:
    na, nb = sa.n_cart, sb.n_cart
    if len(b) != na or any(len(row) != nb for row in b):
        shape = (len(b), *sorted({len(row) for row in b}))
        raise ValueError(
            f"block({where}) has shape {shape}, not ({na}, {nb}): rows are "
            "cartesians(la), columns cartesians(lb)")


def rectangular(basis_a: Basis, basis_b: Basis, block: Block, *,
                pairs: Iterable[tuple[int, int]] | None = None
                ) -> list[list[float]]:
    """The `basis_a.n_ao x basis_b.n_ao` matrix assembled from `block`
    (module docstring, "ACROSS TWO BASES").

    `block(sa, sb)` is called for every ordered shell pair, `sa` shell `i`
    of `basis_a` and `sb` shell `j` of `basis_b`, once, in row-major order
    (`i` outer). With `pairs` -- `(i, j)` index pairs, typically
    `ao.pairs.cross_pair_list(...).kept` -- only those are called and every
    other block is left exactly `0.0`; a pair outside the two bases' shell
    ranges is refused.
    """
    for name, b in (("basis_a", basis_a), ("basis_b", basis_b)):
        if not isinstance(b, Basis):
            raise ValueError(f"{name} {b!r} is not a Basis")
    sha, shb = basis_a.shells, basis_b.shells
    M = [[0.0] * basis_b.n_ao for _ in range(basis_a.n_ao)]
    keep = None
    if pairs is not None:
        keep = set()
        for p in pairs:
            i, j = p
            if not (type(i) is int and type(j) is int
                    and 0 <= i < len(sha) and 0 <= j < len(shb)):
                raise ValueError(
                    f"pair {p!r} is not (i, j) with 0 <= i < {len(sha)} and "
                    f"0 <= j < {len(shb)} shells")
            keep.add((i, j))
    for i, sa in enumerate(sha):
        oa = basis_a.offsets[i]
        for j, sb in enumerate(shb):
            if keep is not None and (i, j) not in keep:
                continue  # screened out: the block stays exactly 0.0
            b = block(sa, sb)
            _check_block(b, sa, sb, f"shell {i} of A, shell {j} of B")
            ob = basis_b.offsets[j]
            for r in range(sa.n_cart):
                for c in range(sb.n_cart):
                    M[oa + r][ob + c] = float(b[r][c])
    return M
