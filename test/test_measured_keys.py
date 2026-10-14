"""#150: every key in `data/measured/kernels.json`, re-derived from its row.

`Spec.key()` is the join key of the measured store. Widening `Spec` to admit a
pair class (`Target.DQ`, `Target.JAB`) must not move a single byte of a
quartet key, or 196 rows silently stop joining. This test rebuilds each row's
key from scratch -- the class, batch count and lever read back by a parser
that does not call `Spec.key()`, the card and compiler from the row's own
`arch` and `toolchain` fields -- through the same constructors the driver
uses (`MeasuredKey`, `asym.AsymKey`), and asserts the string is unchanged.
Fast: reads a tracked file, compiles nothing.
"""

from __future__ import annotations

import itertools
import re

from felizjueves.asym import ASYM, ASYM_CELLS, AsymCell, AsymKey
from felizjueves.device import MeasuredKey, by_name
from felizjueves.kernel import k_spec, pq_spec, split
from felizjueves.lower import RECYCLE
from felizjueves.measured_store import (
    SUBGRID,
    TUNED_CELLS,
    load,
    split_key,
    target_of,
)
from felizjueves.spec import Operator, Spec, Target

# Independent of Spec.key(): the quartet shape as the store spells it.
_SPEC_KEY = re.compile(
    r"coulomb/d0/\((\d+),(\d+)\|(\d+),(\d+)\)/b(\d+)/j/([a-z+-]+)\Z"
)
# #205's K rows: the same shape, target slot `k`, literal only.
_K_KEY = re.compile(
    r"coulomb/d0/\((\d+),(\d+)\|(\d+),(\d+)\)/b(\d+)/k/literal\Z"
)

# #288's F rows: the class in AO order, one batch, target slot `f`, literal.
_F_KEY = re.compile(
    r"coulomb/d0/\((\d+),(\d+)\|(\d+),(\d+)\)/b1/f/literal\Z"
)

N_ROWS = 196  # the J rows; #205 ADDS K rows beside them, #288 F rows
N_K_ROWS = 20  # #205's
# #288's: the five classes of total L <= 2 (`(00|00) (00|01) (00|02) (00|11)
# (01|01)`), on both cards
F_CLASSES_RECORDED = ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 1),
                      (0, 1, 0, 1))
N_F_ROWS = 2 * len(F_CLASSES_RECORDED)


def _j_rows(rows):
    return {k: r for k, r in rows.items()
            if not _K_KEY.fullmatch(split_key(k)[0])
            and not _F_KEY.fullmatch(split_key(k)[0])}


def _derive(key: str, arch: str, toolchain: str) -> str:
    """Rebuild one row's key from its parts, the driver's way."""
    spec_key, _, _ = split_key(key)
    m = _SPEC_KEY.fullmatch(spec_key)
    assert m, f"{key!r} is not a quartet J key"
    l1, l2, l3, l4, n, lever = m.groups()
    am = tuple(map(int, (l1, l2, l3, l4)))
    L, n = sum(am), int(n)
    # Every row's class is the default class of its L.
    assert pq_spec(L).am == am, key
    if lever == ASYM:
        cell = AsymCell(L, n)
        assert cell in ASYM_CELLS, key
        return AsymKey(cell, by_name(arch), toolchain).key()
    levers = () if lever == "literal" else (lever,)
    spec = pq_spec(L, levers=levers, n_batches=n)
    assert spec.target is Target.J and len(spec.am) == 4
    return MeasuredKey(spec, by_name(arch), toolchain).key()


def test_every_measured_key_is_rederived_byte_for_byte() -> None:
    rows = load()
    assert len(_j_rows(rows)) == N_ROWS
    for key, row in _j_rows(rows).items():
        assert _derive(key, row.arch, row.toolchain) == key
    for key, row in rows.items():
        if key in _j_rows(rows):
            continue
        f = _F_KEY.fullmatch(split_key(key)[0])
        if f:  # #288: the class read back, the spec rebuilt
            am = tuple(map(int, f.groups()))
            assert am in F_CLASSES_RECORDED, key
            assert MeasuredKey(Spec(Operator.COULOMB, 0, am, Target.F),
                               by_name(row.arch), row.toolchain).key() == key
            continue
        m = _K_KEY.fullmatch(split_key(key)[0])
        am = tuple(map(int, m.groups()[:4]))
        L, n = sum(am), int(m.group(5))
        assert k_spec(L).am == am, key  # the default K class of its L
        assert MeasuredKey(k_spec(L, n_batches=n), by_name(row.arch),
                           row.toolchain).key() == key


def test_the_rederived_rows_cover_the_declared_cells() -> None:
    # 66 literal (17 one-batch classes + 16 split subgrid cells, per arch),
    # 40 recycle (every subgrid cell, per arch), 16 asym, and 42 no-sieve
    # (every subgrid cell plus L = 11 one-batch, per arch; #162), and 32
    # tuned (every subgrid cell but the four 12-batch ones, which have no
    # recorded search choice, per arch; #163).
    rows = _j_rows(load())
    arches = {r.arch for r in rows.values()}
    assert arches == {"sm_86", "gfx1200"}
    literal = {pq_spec(L) for L in range(17)} | set(SUBGRID)
    recycle = {pq_spec(s.total_am, (RECYCLE,), s.n_batches) for s in SUBGRID}
    sieve = {pq_spec(s.total_am, ("no-sieve",), s.n_batches) for s in SUBGRID} | {
        pq_spec(11, ("no-sieve",))}
    tuned = {pq_spec(s.total_am, ("tuned",), s.n_batches) for s in SUBGRID
             if s.n_batches != 12}
    assert tuned == set(TUNED_CELLS)
    want = set(rows)
    got = set()
    for arch in arches:
        tc = next(r.toolchain for r in rows.values() if r.arch == arch)
        got |= {MeasuredKey(s, by_name(arch), tc).key()
                for s in literal | recycle | sieve | tuned}
        got |= {AsymKey(c, by_name(arch), tc).key() for c in ASYM_CELLS}
    assert got == want
    assert len(literal) * 2 == 66 and len(recycle) * 2 == 40
    assert len(ASYM_CELLS) * 2 == 16 and len(sieve) * 2 == 42
    assert len(tuned) * 2 == 32


def test_a_k_cell_joins_no_measured_j_row() -> None:
    # #196: `Target.K` widens the enum without moving a J key (the two tests
    # above), and its own cells -- named by `k_spec` -- share no key with any
    # J row, so a K cell never reads a J row by accident; #205's K rows are
    # every one of them a K key.
    rows = _j_rows(load())
    assert not any(split_key(key)[0].split("/")[4] == "k" for key in rows)
    assert all(split_key(key)[0].split("/")[4] in ("k", "f")
               for key in set(load()) - set(rows))
    arches = {r.arch: r.toolchain for r in rows.values()}
    k_keys = {
        MeasuredKey(k_spec(L, n_batches=n), by_name(arch), tc).key()
        for L in range(17) for n in (1, 3, 6, 9, 12) for arch, tc in arches.items()
    }
    assert k_keys.isdisjoint(rows)
    # Same L, same (lp, lq), different key.
    for L in range(17):
        assert split(k_spec(L)) == split(pq_spec(L))
        assert k_spec(L).key() != pq_spec(L).key()


def test_target_f_moves_no_measured_key_and_joins_no_row() -> None:
    # #272: `Target.F` widens the enum (milestone 12, pin 4); the two
    # re-derivation tests above hold every stored key byte for byte, and
    # this pins the store's size and that the only F keys holding a row are
    # #288's. The store holds 226 rows -- 196 J, #205's 20 K and #288's 10 F.
    rows = load()
    assert len(rows) == N_ROWS + N_K_ROWS + N_F_ROWS
    assert {split_key(key)[0].split("/")[4] for key in rows} == {"j", "k",
                                                                 "f"}
    arches = {r.arch: r.toolchain for r in rows.values()}
    f_keys = {
        am: {MeasuredKey(Spec(Operator.COULOMB, 0, am, Target.F),
                         by_name(arch), tc).key()
             for arch, tc in arches.items()}
        for am in itertools.product(range(3), repeat=4)
    }
    for am, keys in f_keys.items():
        if am in F_CLASSES_RECORDED:
            assert keys <= set(rows), am
        else:
            assert keys.isdisjoint(rows), am


def test_an_f_key_reads_back_as_an_f_target_and_its_parts() -> None:
    # #278: the store admits an F cell. An F key splits like every other key
    # (spec key, arch, toolchain), its target slot reads back as `Target.F`,
    # and the independent J and K parsers above reject it -- so an F row can
    # never be re-derived as a J or K row, and the 226 tracked rows (196 J,
    # 20 K, 10 F) all still re-derive (the tests above) with the F check in
    # place.
    spec = Spec(Operator.COULOMB, 0, (2, 1, 1, 0), Target.F)
    key = MeasuredKey(spec, by_name("sm_86"), "nvcc 13.0").key()
    assert key == "coulomb/d0/(2,1|1,0)/b1/f/literal@sm_86/nvcc 13.0"
    assert split_key(key) == (spec.key(), "sm_86", "nvcc 13.0")
    assert target_of(key) is Target.F
    assert not _SPEC_KEY.fullmatch(spec.key())
    assert not _K_KEY.fullmatch(spec.key())
    assert len(load()) == N_ROWS + N_K_ROWS + N_F_ROWS
