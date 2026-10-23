"""#150: every key in `data/measured/kernels.json`, re-derived from its row.

`Spec.key()` is the join key of the measured store. Widening `Spec` to admit a
pair class (`Target.DQ`, `Target.JAB`) must not move a single byte of a
quartet key, or 196 rows silently stop joining. This test rebuilds each row's
key from scratch -- the class, batch count and lever read back by a parser
that does not call `Spec.key()`, the card and compiler from the row's own
`arch` and `toolchain` fields -- through the same constructors the driver
uses (`MeasuredKey`, `asym.AsymKey`), and asserts the string is unchanged.
A CHECK OF A MEASUREMENT ARTEFACT, moved from test/ (`devtools/measure.sh
check`): reads a tracked file, compiles nothing. The parser (`_derive`) and
its round trip on a synthetic lever-keyed row stayed, as
test/test_measured_keys.py.
"""

from __future__ import annotations

import itertools
import re

from test_measured_keys import (
    _SPEC_KEY,
    _derive,
)

from felizjueves.ir.lower import RECYCLE
from felizjueves.mmd.kernel import k_spec, pq_spec, split
from felizjueves.performance.asym import ASYM_CELLS, AsymKey
from felizjueves.performance.device import MeasuredKey, by_name
from felizjueves.performance.measured_store import (
    KERNEL,
    SUBGRID,
    TUNED_CELLS,
    ledger_cells,
    ledger_classes,
    split_key,
    target_of,
)
from felizjueves.performance.measured_store import load as _load_all
from felizjueves.spec import Operator, Spec, Target


def _is_ledger(key: str, row) -> bool:
    """#362's lever-ledger rows: a kernel lever arm (KERNEL), or a K literal
    class at L <= 4 it is read against."""
    return row.arm_kind == KERNEL or split_key(key)[0] in {
        c.key() for c in ledger_classes(Target.K)}


def load():
    """The store as the tests below pin it: every row but #362's ledger
    rows, which `test_the_ledger_rows_are_its_declared_cells` holds."""
    return {k: r for k, r in _load_all().items() if not _is_ledger(k, r)}


def test_the_ledger_rows_are_its_declared_cells() -> None:
    # #362 ADDS its rows; each is keyed on a ledger cell, by the driver's
    # MeasuredKey, on one of the two cards
    rows = {k: r for k, r in _load_all().items() if _is_ledger(k, r)}
    assert rows
    cells = {c.key(): c for c in ledger_cells()}
    for key, row in rows.items():
        spec = cells[split_key(key)[0]]
        assert MeasuredKey(spec, by_name(row.arch), row.toolchain).key() == key
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
# #288's five classes of total L <= 2 (`(00|00) (00|01) (00|02) (00|11)
# (01|01)`) and #290's sixteen above them: the whole 21-class ladder (every
# shell l <= 2, each pair ascending, bra pair <= ket pair), on both cards
_F_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
F_CLASSES_RECORDED = tuple(bra + ket for i, bra in enumerate(_F_PAIRS)
                           for ket in _F_PAIRS[i:])
assert len(F_CLASSES_RECORDED) == 21
N_F_ROWS = 2 * len(F_CLASSES_RECORDED)  # 42


def _j_rows(rows):
    return {k: r for k, r in rows.items()
            if not _K_KEY.fullmatch(split_key(k)[0])
            and not _F_KEY.fullmatch(split_key(k)[0])}


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
    # #288's and #290's. The store holds 258 rows -- 196 J, #205's 20 K, and
    # 42 F (#288's 10, #290's 32).
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
    # never be re-derived as a J or K row, and the 258 tracked rows (196 J,
    # 20 K, 42 F) all still re-derive (the tests above) with the F check in
    # place.
    spec = Spec(Operator.COULOMB, 0, (2, 1, 1, 0), Target.F)
    key = MeasuredKey(spec, by_name("sm_86"), "nvcc 13.0").key()
    assert key == "coulomb/d0/(2,1|1,0)/b1/f/literal@sm_86/nvcc 13.0"
    assert split_key(key) == (spec.key(), "sm_86", "nvcc 13.0")
    assert target_of(key) is Target.F
    assert not _SPEC_KEY.fullmatch(spec.key())
    assert not _K_KEY.fullmatch(spec.key())
    assert len(load()) == N_ROWS + N_K_ROWS + N_F_ROWS


def test_target_metric_moves_no_measured_key_and_joins_no_row() -> None:
    # #410: `Target.METRIC` widens the enum; the re-derivation tests above
    # hold all 196 J keys (and the K and F rows) byte for byte. A metric key
    # is a pair class with target slot `metric`: the J, K and F parsers
    # reject it, it reads back as `Target.METRIC`, and no tracked row --
    # nothing about the metric is measured -- carries one.
    rows = _load_all()
    assert len(_j_rows(load())) == N_ROWS
    assert not any(target_of(k) is Target.METRIC for k in rows)
    arches = {r.arch: r.toolchain for r in rows.values()}
    for lp in range(5):
        for lq in range(5):
            spec = Spec(Operator.COULOMB, 0, (lp, lq), Target.METRIC)
            for regex in (_SPEC_KEY, _K_KEY, _F_KEY):
                assert not regex.fullmatch(spec.key())
            for arch, tc in arches.items():
                key = MeasuredKey(spec, by_name(arch), tc).key()
                assert key not in rows
                assert split_key(key) == (spec.key(), arch, tc)
                assert target_of(key) is Target.METRIC
    assert MeasuredKey(Spec(Operator.COULOMB, 0, (2, 1), Target.METRIC),
                       by_name("sm_86"), "nvcc 13.0").key() == (
        "coulomb/d0/(2,1)/b1/metric/literal@sm_86/nvcc 13.0")
