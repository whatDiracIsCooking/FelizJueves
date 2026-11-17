"""The packed three-center kernel (FreeBreakfast#1680), interpreted:
`build_eri3_kernel(cls, packed=True)`, `peri3_{la}{lb}{lP}`.

THE CONTRACT (`eri.density_fit.eri3`'s THE PACKED OUTPUT): the square
kernel's cells and records, each cell's whole staged block stored at the
cell table's destination offset `dst`, element `(a, b, q)` at `dst + n_aux
(a nB + b) + q` (the aux index fastest); a diagonal shell pair (`diag = 1`)
stores its full square block.

THE WITNESS is the square kernel of the same class and form, through the
same interpreter: every packed element equals the square tensor's
`[off_a + a][off_b + b][off_p + q]` BITWISE. The arithmetic is the same
statements in the same order and every store is plain, so there is no
rounding to bound; the square kernel's own witnesses (the host, libcint)
carry over through this equality.

SCOPE: every class `la + lb + lP <= 4` on `test_kernel_eri3`'s synthetic
bases (swapped, diagonal and off-diagonal pairs), in BOTH forms (unrolled
and low-stack, forced), s and p fast and the rest `slow`, as
test_kernel_eri3.py; a batch of pairs with batch-relative offsets; the
packed layout helpers. The card's run is test_eri3_packed_device.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_kernel_eri3 as te
from eri3_driver import gather_packed, packed_layout

from felizjueves.ao.shell import cartesians
from felizjueves.eri.density_fit.eri3 import (
    eri3_classes,
    packed_cells,
    packed_columns,
)
from felizjueves.eri.density_fit.eri3_drive import launch_class, launch_packed
from felizjueves.eri.density_fit.kernel_eri3 import (
    OUT,
    PACKED_CELL_ARRAYS,
    build_eri3_kernel,
    eri3_name,
)
from felizjueves.ir import AtomicAdd, walk
from felizjueves.reference.boys import boys

FORMS = {"unrolled": False, "lowstack": True}
N_AO, N_AUX = te.SYNTHETIC.n_ao, te.SYNTHETIC_AUX.n_ao


def _classes():
    return eri3_classes(te.SYNTHETIC, te.SYNTHETIC_AUX)


def _cls(am):
    return next(c for c in _classes() if (c.la, c.lb, c.lP) == am)


def _square(cls, lowstack: bool) -> np.ndarray:
    return launch_class(cls, np.zeros((N_AO, N_AO, N_AUX)), boys=boys,
                        threads=5, lowstack=lowstack)


def _packed(cls, cells, n_col: int, lowstack: bool) -> np.ndarray:
    return launch_packed(cls, cells, np.full((n_col, N_AUX), np.nan),
                         boys=boys, threads=5, lowstack=lowstack)


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("am", [te._param(c) for c in te.CLASSES])
def test_the_packed_block_is_the_square_blocks_bitwise(am, form):
    """Every cell of the class, in a layout over every pair of the bases:
    the packed output is the square tensor's blocks gathered, bitwise, and
    every column of the class's pairs is stored."""
    cls = _cls(am)
    cells, n_col = packed_layout(_classes(), N_AUX)
    k = _classes().index(cls)
    got = _packed(cls, cells[k], n_col, FORMS[form])
    want = gather_packed(_square(cls, FORMS[form]), [cls], [cells[k]], n_col,
                         N_AUX)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert (~np.isnan(got)).sum() == cls.n_cell * len(
        cartesians(cls.la)) * len(cartesians(cls.lb)) * len(
        cartesians(cls.lP))
    assert np.array_equal(got, want, equal_nan=True), am


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("am", [(1, 1, 1), (0, 1, 1)])
def test_a_batch_of_pairs_with_relative_offsets_is_the_same_columns(am, form):
    """Batching over shell pairs is table contents: the class's pairs split
    in two, each batch laid out from its own column 0, gives the one-layout
    columns of its pairs bitwise ((pp|p) has diagonal pairs, (sp|p) swapped
    ones)."""
    cls = _cls(am)
    whole, n_col = packed_columns([cls])
    full = _packed(cls, packed_cells(cls, whole, N_AUX), n_col, FORMS[form])
    pairs = sorted(whole, key=whole.get)
    first = set(pairs[:len(pairs) // 2])
    for keep in (first.__contains__, lambda p: p not in first):
        cols, n = packed_columns([cls], keep)
        assert 0 < n < n_col and min(cols.values()) == 0
        got = _packed(cls, packed_cells(cls, cols, N_AUX), n, FORMS[form])
        nab = len(cartesians(cls.la)) * len(cartesians(cls.lb))
        for pair, c in cols.items():
            at = whole[pair]
            assert np.array_equal(got[c:c + nab], full[at:at + nab],
                                  equal_nan=True), pair  # other aux: NaN


@pytest.mark.parametrize("form", sorted(FORMS))
def test_a_diagonal_pair_stores_its_full_square_block(form):
    """(pp|s): a diagonal pair's block is complete and exactly symmetric in
    (a, b); an off-diagonal pair's is not symmetric."""
    cls = _cls((1, 1, 0))
    cols, n_col = packed_columns([cls])
    got = _packed(cls, packed_cells(cls, cols, N_AUX), n_col, FORMS[form])
    seen = set()
    for k, (a, b, n) in enumerate(cls.cells):
        op = cls.off_p[k]
        blk = got[cols[(a, b)]:cols[(a, b)] + 9, op].reshape(3, 3)
        assert not np.isnan(blk).any(), (a, b, n)
        if a == b:
            assert np.array_equal(blk, blk.T), (a, b, n)
        else:
            seen.add(np.array_equal(blk, blk.T))
    assert seen == {False}


# --- the layout ----------------------------------------------------------------


def test_the_columns_are_in_class_order_and_tile_the_output():
    classes = _classes()
    cols, n_col = packed_columns(classes)
    nab = {}
    for c in classes:
        for a, b, _ in c.cells:
            nab[(a, b)] = (len(cartesians(c.la)) * len(cartesians(c.lb)),
                           (c.la, c.lb))
    assert set(cols) == set(nab)  # every pair, once
    order = sorted(cols, key=cols.get)
    assert [nab[p][1] for p in order] == sorted(nab[p][1] for p in order)
    at = 0
    for p in order:
        assert cols[p] == at
        at += nab[p][0]
    assert at == n_col


def test_the_cell_arrays_are_the_records_and_the_offsets():
    classes = _classes()
    cols, _ = packed_columns(classes)
    for c in classes:
        got = packed_cells(c, cols, N_AUX)
        assert tuple(got) == PACKED_CELL_ARRAYS
        for name in ("ab_rec0", "ab_rec1", "p_rec0", "p_rec1"):
            assert got[name] == getattr(c, name)
        assert got["dst"] == tuple(cols[(a, b)] * N_AUX + c.off_p[k]
                                   for k, (a, b, _) in enumerate(c.cells))
        assert got["diag"] == tuple(int(a == b) for a, b, _ in c.cells)
        assert not any(got["diag"]) or c.la == c.lb


def test_a_kept_subset_never_adds_a_pair():
    classes = _classes()
    every, _ = packed_columns(classes)
    some, _ = packed_columns(classes, lambda p: p[0] % 2 == 0)
    assert set(some) < set(every)
    assert all(a % 2 == 0 for a, _ in some)


# --- the kernel ----------------------------------------------------------------


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_packed_kernel_is_named_and_shaped_apart(form):
    for am in ((0, 0, 0), (1, 1, 1), (0, 2, 1), (2, 2, 0)):
        k = build_eri3_kernel(am, packed=True, lowstack=FORMS[form])
        sq = build_eri3_kernel(am, lowstack=FORMS[form])
        assert k.name == eri3_name(am, packed=True) == "peri3_" + "".join(
            map(str, am))
        assert sq.name == eri3_name(am)
        names = [p.name for p in k.params]
        assert names[:3] == [p.name for p in sq.params][:3]
        assert names[3:5] == ["n_col", "n_aux"]
        assert names[5:13] == [p.name for p in sq.params][5:13]
        assert tuple(names[13:19]) == PACKED_CELL_ARRAYS
        assert names[19:] == [OUT] and len(k.params[-1].shape) == 1
        assert not any(isinstance(s, AtomicAdd) for s in walk(k.body))


def test_the_default_is_the_square_kernel():
    for am in ((0, 0, 0), (1, 1, 1), (2, 2, 0)):
        assert build_eri3_kernel(am) == build_eri3_kernel(am, packed=False)
