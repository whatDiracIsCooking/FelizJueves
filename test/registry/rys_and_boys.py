"""Rys `F`, `J` and `K`, and the device Boys: tree-state's
`references/rys-and-boys.md`."""

from __future__ import annotations

from .rules import Allow, Classes, ladder

CASES = [
    # Interpreted and emitted; never compiled by nvcc / hipcc on a run of
    # record, never on a card.
    *(case for q in "fjk" for case in ladder("rys", q, 0, compile=range(9))),
]

BOYS = ("the device Boys template fj_boys<M>: an order M, not an integral "
        "class")

TESTS = {
    "test/test_emit_frys.py::test_X_compiles_the_frys_class": Classes("rys", "f"),
    "test/test_emit_rj.py::test_X_compiles_the_rj_unit": Classes("rys", "j"),
    "test/test_emit_rk.py::test_X_compiles_the_rk_unit": Classes("rys", "k"),
    "test/test_emit_X.py::test_the_template_compiles": Allow(BOYS),
}
