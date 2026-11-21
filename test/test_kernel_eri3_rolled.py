"""The rolled spherical three-center kernels (#748), interpreted, and the
IR's constant table they read.

`build_eri3_kernel(cls, packed=True, spherical_aux=True, rolled=True)`
(`rsperi3_`) and with `spherical_bra=True` (`rssperi3_`): the aux columns
and the bra harmonics are loops reading `aux_harmonics` from `Const`
tables (CSR), instead of unrolled literals.

THE ORACLE is the unrolled kernel itself (`speri3_` / `ssperi3_`, held to
the host by test_kernel_eri3_spherical.py and
test_kernel_eri3_fully_spherical.py): the rolled form performs the same
products and sums in the same order (each sum a left fold over the same
nonzero terms, from 0, and `0 + x` is `x`), so the interpreted outputs are
compared BITWISE, NaN fill included. A tolerance here would hide a
reordered term.

SCOPE (minimal, by decision): (pp|d), (pd|p), (dd|f), (sf|d) in both
modes -- an identity bra, an `la != lb` transform, a diagonal transform,
an s bra against f -- on three AO shells and one aux shell. The card run
is test_eri3_rolled_device.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from felizjueves.ao.shell import Basis, Shell
from felizjueves.emit import cuda, hip
from felizjueves.eri.density_fit.eri3 import (
    eri3_classes,
    packed_cells,
    packed_columns,
    spherical_offsets,
)
from felizjueves.eri.density_fit.eri3_drive import launch_packed
from felizjueves.eri.density_fit.kernel_eri3 import build_eri3_kernel
from felizjueves.ir import (
    Assign,
    Call,
    Const,
    Decl,
    Fn,
    Index,
    Intent,
    Kernel,
    Lit,
    Param,
    Scalar,
    Var,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.reference.boys import boys

A, B, C = (0.0, 0.0, 0.0), (0.3, -0.7, 1.1), (-1.2, 0.4, 0.25)
CLASSES = [(1, 1, 2), (1, 2, 1), (2, 2, 3), (0, 3, 2)]


def layout(am):
    """Three AO shells (two of `la`, one of `lb`; diagonal and off-diagonal
    pairs) and one contracted aux shell; the class and its packed layout
    in both bra modes."""
    la, lb, lP = am
    ao = Basis((Shell(la, A, (0.8,), (1.0,)),
                Shell(lb, B, (1.1, 0.35), (0.6, 0.5)),
                Shell(la, C, (0.9,), (1.0,))))
    aux = Basis((Shell(lP, C, (0.7, 0.3), (0.8, 0.4)),))
    (cls,) = [c for c in eri3_classes(ao, aux) if (c.la, c.lb, c.lP) == am]
    offs, n_aux = spherical_offsets(aux)
    return cls, offs, n_aux


def _run(am, spherical_bra, rolled):
    cls, offs, n_aux = layout(am)
    cols, n_col = packed_columns([cls], spherical_bra=spherical_bra)
    cells = packed_cells(cls, cols, n_aux, offs)
    return launch_packed(cls, cells, np.full((n_col, n_aux), np.nan),
                         boys=boys, threads=5, spherical_aux=True,
                         spherical_bra=spherical_bra, rolled=rolled)


@pytest.mark.parametrize("spherical_bra", [False, True],
                         ids=["rsperi3", "rssperi3"])
@pytest.mark.parametrize("am", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_the_rolled_kernel_is_the_unrolled_one_bitwise(am, spherical_bra):
    assert np.array_equal(_run(am, spherical_bra, True),
                          _run(am, spherical_bra, False), equal_nan=True)


def test_the_rolled_kernels_are_named_apart_with_the_same_signature():
    for sb, head in ((False, "rsperi3_"), (True, "rssperi3_")):
        k = build_eri3_kernel((3, 3, 6), packed=True, spherical_aux=True,
                              spherical_bra=sb, rolled=True)
        u = build_eri3_kernel((3, 3, 6), packed=True, spherical_aux=True,
                              spherical_bra=sb)
        assert k.name == head + "336" and k.params == u.params
        assert k.consts and not u.consts
    with pytest.raises(ValueError, match="spherical_aux"):
        build_eri3_kernel((1, 1, 1), packed=True, rolled=True)


# --- the IR's constant table ---------------------------------------------------


def _table_kernel() -> Kernel:
    """out[i] = tab[i] * x[i]: one read-only table, one input, one output."""
    i = Var("i")
    return Kernel("tab_k", (
        Param("x", Scalar.F64, array=True),
        Param("out", Scalar.F64, array=True, intent=Intent.OUT)), (
        Decl("i", Scalar.I32, init=Call(Fn.THREAD_X)),
        Assign(Index("out", (i,)), Call(Fn.MUL, (Index("tab", (i,)),
                                                  Index("x", (i,)))))),
        (Const("tab", Scalar.F64, (0.5, -2.0, 3.25)),))


def test_a_const_table_is_read_interpreted_and_emitted_once_at_file_scope():
    k = _table_kernel()
    got = Interpreter(k, launch=Launch(1, (3, 1), 32)).run(
        x=np.array([2.0, 1.0, 4.0]), out=np.zeros(3))["out"]
    assert got.tolist() == [1.0, -2.0, 13.0]
    for text in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)):
        assert text.count("__constant__ double tab[3] = {") == 1
        assert "0.5, (-2.0), 3.25," in text
        assert text.index("__constant__") < text.index("__global__")


def test_a_const_table_refuses_a_store_and_a_clash():
    k = _table_kernel()
    with pytest.raises(ValueError, match="stores to the Const"):
        Kernel(k.name, k.params, (*k.body, Assign(
            Index("tab", (Lit(0),)), Lit(1.0))), k.consts)
    with pytest.raises(ValueError, match="unique"):
        Kernel(k.name, k.params, k.body,
               (Const("x", Scalar.F64, (1.0,)),))
    with pytest.raises(ValueError, match="I32"):
        Const("t", Scalar.I32, (1.5,))
