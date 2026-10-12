"""#150: every key in `data/measured/kernels.json`, re-derived from its row.

`Spec.key()` is the join key of the measured store. Widening `Spec` to admit a
pair class (`Target.DQ`, `Target.JAB`) must not move a single byte of a
quartet key, or 122 rows silently stop joining. This test rebuilds each row's
key from scratch -- the class, batch count and lever read back by a parser
that does not call `Spec.key()`, the card and compiler from the row's own
`arch` and `toolchain` fields -- through the same constructors the driver
uses (`MeasuredKey`, `asym.AsymKey`), and asserts the string is unchanged.
Fast: reads a tracked file, compiles nothing.
"""

from __future__ import annotations

import re

from felizjueves.asym import ASYM, ASYM_CELLS, AsymCell, AsymKey
from felizjueves.device import MeasuredKey, by_name
from felizjueves.kernel import pq_spec
from felizjueves.lower import RECYCLE
from felizjueves.measured_store import SUBGRID, load, split_key
from felizjueves.spec import Target

# Independent of Spec.key(): the quartet shape as the store spells it.
_SPEC_KEY = re.compile(
    r"coulomb/d0/\((\d+),(\d+)\|(\d+),(\d+)\)/b(\d+)/j/([a-z+-]+)\Z"
)

N_ROWS = 122


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
    assert len(rows) == N_ROWS
    for key, row in rows.items():
        assert _derive(key, row.arch, row.toolchain) == key


def test_the_rederived_rows_cover_the_declared_cells() -> None:
    # 66 literal (17 one-batch classes + 16 split subgrid cells, per arch),
    # 40 recycle (every subgrid cell, per arch), 16 asym, as CLAUDE.md says.
    rows = load()
    arches = {r.arch for r in rows.values()}
    assert arches == {"sm_86", "gfx1200"}
    literal = {pq_spec(L) for L in range(17)} | set(SUBGRID)
    recycle = {pq_spec(s.total_am, (RECYCLE,), s.n_batches) for s in SUBGRID}
    want = set(rows)
    got = set()
    for arch in arches:
        tc = next(r.toolchain for r in rows.values() if r.arch == arch)
        got |= {MeasuredKey(s, by_name(arch), tc).key() for s in literal | recycle}
        got |= {AsymKey(c, by_name(arch), tc).key() for c in ASYM_CELLS}
    assert got == want
    assert len(literal) * 2 == 66 and len(recycle) * 2 == 40
    assert len(ASYM_CELLS) * 2 == 16
