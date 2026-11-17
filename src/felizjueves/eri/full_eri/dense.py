"""The dense AO ERI matrices, host-side: the staging, the row range, the reference.

    LAYOUTS                                 ("j", "k")
    row_range(basis, shells)                -> (m0, n_row), checked
    pair_class(basis, la, lb, first=None)   -> the ordered shell pairs of (la, lb)
    dense_classes(basis, shells=None)       -> the `kernel_dense` launches
    dense_eri_and_scale(basis, layout, *, pq, shells=None, keep=None)
                                            -> (M rows, their |.| scale)
    owned(basis, layout, shells=None, keep=None) -> which elements are written
    row_index(n_ao, layout, m, n, l, s) / col_index(...)

THE TWO LAYOUTS. With every AO pair unpacked, `(m, n) -> m n_ao + n`, the
Cartesian tensor `(mn|ls)` is laid out as one of two dense `n_ao^2 x n_ao^2`
matrices,

    J layout   M_J[(m,n), (l,s)] = (mn|ls),   J = M_J vec(D),
    K layout   M_K[(m,n), (l,s)] = (ml|ns),   K = M_K vec(D),

`vec(D)[(l,s)] = D_ls` row-major, so `J_mn = SUM_ls (mn|ls) D_ls` and
`K_mn = SUM_ls (ml|ns) D_ls` are each ONE matrix-vector product, for any `D`:
nothing is packed and nothing assumes `D` symmetric.

NO SYMMETRY, BY DECISION. A task is one ORDERED shell quartet `(A B|C D)`,
every one of them: its block `(ab|cd)` is computed once and stored, element
for element, at `M_J[(a,b), (c,d)]` or at `M_K[(a,c), (b,d)]` (`m = a`,
`l = b`, `n = c`, `s = d`). Each element of either matrix is in exactly one
ordered quartet's block, so every element has one owner and one plain store,
and nothing is mirrored. The price is recomputation: every unique integral is
computed by each of the up to eight orderings of its quartet, 8x
`kernel_eri`'s unique-quartet build at the limit of distinct shells
(`n_shell^4` tasks against about `n_shell^4 / 8`).

THE ROW RANGE. Both layouts' row `(m, n)` has `m` in the bra's FIRST shell
`A`, so a launch over the shells `[sh0, sh1)` of `m` owns the contiguous rows
`[m0 n_ao, (m0 + n_m) n_ao)`, `m0 = off[sh0]`, `n_m` the AOs of those shells,
all `n_ao^2` columns: the tasks whose `A` is one of those shells, every `B`,
`C`, `D`. Ranges over disjoint shell sets own disjoint rows, so `G` launches
over a partition of the shells write `G` disjoint row blocks that together
are the whole matrix -- collected by a gather, never a reduce -- and do the
same total work as one launch over every shell. The output is the LOCAL row
block, `n_row = n_m n_ao` rows from row `m0 n_ao`: its flat index is
`(row - m0 n_ao) n_ao^2 + col`, so the `int32` bound is `n_row n_ao^2 < 2^31`
PER LAUNCH (`row_range` refuses past it), not `n_ao^4`. There is no
`N_AO_MAX`: sizing the row blocks to the card is the caller's job.

THE STAGING (`dense_classes`). One `kernel_dense` launch per ordered class
`(la lb|lc ld)` the basis has, bra and ket each a `pair_class` -- every
ORDERED shell pair `(i, j)` with `l_i = la`, `l_j = lb`, in basis order,
`i` outer, its primitive pairs (`ao.density.epair(sh[i], sh[j])`, `XPA =
P - A_i`) one contiguous record range -- the bra's restricted to `i` in the
range; the tasks are every (bra pair, ket pair) of the two. The arrays are
`packed.EriClass`'s, so a launch takes `kernel_eri`'s record and task
parameters unchanged.

THE REFERENCE is `eri.eri.eri_block_and_scale` of each task's ordered quartet,
placed by the kernel's rule: `eri.eri` is witnessed against libcint
(test/test_ao_eri.py). `pq` is INJECTED; nothing here imports `reference/`.
"""

from __future__ import annotations

from felizjueves.ao.density import epair
from felizjueves.ao.shell import Basis
from felizjueves.eri.eri import PQ, eri_block_and_scale
from felizjueves.eri.full_eri.packed import EriClass
from felizjueves.eri.schwarz import SchwarzClass

LAYOUTS = ("j", "k")
INT32_MAX = 2 ** 31 - 1
Matrix = list[list[float]]


def _layout(layout: str) -> str:
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r} is not one of {LAYOUTS}")
    return layout


def _shells(basis: Basis, shells: tuple[int, int] | None) -> tuple[int, int]:
    n = len(basis.shells)
    sh0, sh1 = (0, n) if shells is None else shells
    if not (type(sh0) is int and type(sh1) is int and 0 <= sh0 < sh1 <= n):
        raise ValueError(f"shells {shells!r} is not a nonempty range of "
                         f"0..{n}")
    return sh0, sh1


def row_range(basis: Basis, shells: tuple[int, int] | None = None
              ) -> tuple[int, int]:
    """`(m0, n_row)`: the first AO `m` of the shells `[sh0, sh1)` (every
    shell when None) and the rows `n_m n_ao` they own; refuses a block whose
    `n_row n_ao^2` passes `int32`."""
    sh0, sh1 = _shells(basis, shells)
    off = (*basis.offsets, basis.n_ao)
    m0, n_m = off[sh0], off[sh1] - off[sh0]
    n_row = n_m * basis.n_ao
    if n_row * basis.n_ao ** 2 > INT32_MAX:
        raise ValueError(
            f"shells {sh0}..{sh1}: {n_row} rows x {basis.n_ao ** 2} columns "
            "passes int32 (the kernel's flat index); split the range")
    return m0, n_row


def row_index(n_ao: int, layout: str, m: int, n: int, ll: int, s: int
              ) -> int:
    """The global row of the element `(mn|ls)` in `layout`."""
    return m * n_ao + (n if _layout(layout) == "j" else ll)


def col_index(n_ao: int, layout: str, m: int, n: int, ll: int, s: int
              ) -> int:
    """The column of the element `(mn|ls)` in `layout`."""
    return (ll if _layout(layout) == "j" else n) * n_ao + s


def pair_class(basis: Basis, la: int, lb: int,
               first: tuple[int, int] | None = None) -> SchwarzClass:
    """Every ordered shell pair `(i, j)`, `l_i = la`, `l_j = lb`, `i` in
    `first` (a shell range, every shell when None), in basis order, with its
    records (module docstring, THE STAGING)."""
    sh, off = basis.shells, basis.offsets
    lo, hi = (0, len(sh)) if first is None else first
    c: dict[str, list] = {k: [] for k in (
        "pairs", *SchwarzClass.RECORD_ARRAYS, *SchwarzClass.PAIR_ARRAYS)}
    for i in range(lo, hi):
        if sh[i].l != la:
            continue
        for j in range(len(sh)):
            if sh[j].l != lb:
                continue
            c["pairs"].append((i, j))
            c["rec0"].append(len(c["zeta_ab"]))
            for P, zeta, s in epair(sh[i], sh[j]).recs:
                c["zeta_ab"].append(zeta)
                c["P"].append(tuple(P))
                c["XPA"].append(tuple(p - x for p, x in zip(
                    P, sh[i].centre, strict=True)))
                c["XPB"].append(tuple(p - x for p, x in zip(
                    P, sh[j].centre, strict=True)))
                c["s_ab"].append(s)
            c["rec1"].append(len(c["zeta_ab"]))
            c["off_a"].append(off[i])
            c["off_b"].append(off[j])
    return SchwarzClass(la, lb, **{k: tuple(v) for k, v in c.items()})


def dense_classes(basis: Basis, shells: tuple[int, int] | None = None
                  ) -> tuple[EriClass, ...]:
    """Every ordered class `(la lb|lc ld)` with a task whose first shell is
    in `shells` (every shell when None), sorted by `am`, with its tasks
    (module docstring, THE STAGING)."""
    first = _shells(basis, shells)
    row_range(basis, first)  # the int32 check
    ls = sorted({s.l for s in basis.shells})
    out = []
    for la in ls:
        for lb in ls:
            bra = pair_class(basis, la, lb, first)
            if not bra.n_pair:
                continue
            for lc in ls:
                for ld in ls:
                    ket = pair_class(basis, lc, ld)
                    tb = [b for b in range(bra.n_pair)
                          for _ in range(ket.n_pair)]
                    tk = [k for _ in range(bra.n_pair)
                          for k in range(ket.n_pair)]
                    out.append(EriClass(bra, ket, tuple(tb), tuple(tk)))
    return tuple(sorted(out, key=lambda c: c.am))


def _place(basis: Basis, layout: str, shells, keep, value) -> Matrix:
    """`value(sa, sb, sc, sd)`'s block of every task placed by the kernel's
    rule into the local row block; only the classes in `keep` (a set of
    `am`) when given, 0.0 elsewhere."""
    _layout(layout)
    sh = basis.shells
    n = basis.n_ao
    m0, n_row = row_range(basis, shells)
    r0 = m0 * n
    out = [[0.0] * (n * n) for _ in range(n_row)]
    for c in dense_classes(basis, shells):
        if keep is not None and c.am not in keep:
            continue
        for tb, tk in zip(c.task_bra, c.task_ket, strict=True):
            i, j = c.bra.pairs[tb]
            k, q = c.ket.pairs[tk]
            blk = value(sh[i], sh[j], sh[k], sh[q])
            oa, ob, oc, od = c.bra.off_a[tb], c.bra.off_b[tb], \
                c.ket.off_a[tk], c.ket.off_b[tk]
            nb, nd = sh[j].n_cart, sh[q].n_cart
            for a in range(sh[i].n_cart):
                for b in range(nb):
                    for cc in range(sh[k].n_cart):
                        for d in range(nd):
                            e = (oa + a, ob + b, oc + cc, od + d)
                            out[row_index(n, layout, *e) - r0][
                                col_index(n, layout, *e)] = \
                                blk[a * nb + b][cc * nd + d]
    return out


def dense_eri_and_scale(basis: Basis, layout: str, *, pq: PQ,
                        shells: tuple[int, int] | None = None, keep=None
                        ) -> tuple[Matrix, Matrix]:
    """The local row block of `M_J` or `M_K` over `shells` and `eri.eri`'s
    absolute-value scale of each element; only the classes in `keep` when
    given."""
    cache: dict = {}

    def both(sa, sb, sc, sd):
        key = (id(sa), id(sb), id(sc), id(sd))
        if key not in cache:
            cache[key] = eri_block_and_scale((sa, sb), (sc, sd), pq=pq)
        return cache[key]

    return (_place(basis, layout, shells, keep, lambda *s: both(*s)[0]),
            _place(basis, layout, shells, keep, lambda *s: both(*s)[1]))


def owned(basis: Basis, layout: str, shells: tuple[int, int] | None = None,
          keep=None) -> list[list[bool]]:
    """Which elements of the local row block the classes in `keep` (every
    class when None) write."""
    def ones(sa, sb, sc, sd):
        return [[1.0] * (sc.n_cart * sd.n_cart)] * (sa.n_cart * sb.n_cart)
    return [[x == 1.0 for x in row]
            for row in _place(basis, layout, shells, keep, ones)]
