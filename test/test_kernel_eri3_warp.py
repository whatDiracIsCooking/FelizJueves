"""The warp-per-cell spherical three-center kernels (#749), interpreted:
`build_warp_eri3_kernel(cls)` (`wsperi3_`) and with `spherical_bra=True`
(`wssperi3_`), a block of `WARP_LANES` per cell, the tower, bra tables,
`U` and `Vab` in SHARED memory between barriers, each lane's output slots
in registers and stored once.

THE ORACLE is the thread-per-cell kernel (`speri3_` / `ssperi3_`, held to
the host by test_kernel_eri3_spherical.py and
test_kernel_eri3_fully_spherical.py): every `U`, `Vab`, bra-transform and
accumulator update of the warp form is that kernel's statement, in the same
order per output slot, only split across lanes, so the interpreted outputs
are compared BITWISE, NaN fill included. The interpreter also refuses a
barrier any thread misses and a read of a shared element no lane has
written yet, so a missing barrier fails here too.

SCOPE (minimal, by decision): (pp|d), (pd|p), (dd|f), (sf|d) in both modes
(an identity bra, an `la != lb` transform, a diagonal transform, an s bra
against f), on three AO shells and two aux shells; and (pp|d) with a
four-primitive aux shell at phase widths 1 and 3 (several phases, a
partial last one). The card run is test_eri3_warp_device.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit.eri3 import (
    eri3_classes,
    packed_cells,
    packed_columns,
    spherical_offsets,
)
from felizjueves.eri.density_fit.eri3_drive import launch_packed
from felizjueves.eri.density_fit.kernel_eri3 import (
    ACC_ARRAY,
    OUT,
    PACKED_CELL_ARRAYS,
    PARAM_OF,
    WARP_LANES,
    build_eri3_kernel,
    build_warp_eri3_kernel,
    warp_launch_shape,
)
from felizjueves.ir import AtomicAdd, Barrier, Decl, Lit, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys

A, B, C = (0.0, 0.0, 0.0), (0.3, -0.7, 1.1), (-1.2, 0.4, 0.25)
CLASSES = [(1, 1, 2), (1, 2, 1), (2, 2, 3), (0, 3, 2)]


def layout(am, aux_exps=((0.7, 0.3), (1.2,))):
    """Three AO shells (diagonal and off-diagonal pairs, `la != lb` when
    the class has it) and one aux shell per entry of `aux_exps`."""
    la, lb, lP = am
    ao = Basis((Shell(la, A, (0.8,), (1.0,)),
                Shell(lb, B, (1.1, 0.35), (0.6, 0.5)),
                Shell(la, C, (0.9,), (1.0,))))
    aux = Basis(tuple(Shell(lP, (C, A)[i % 2], e, tuple(
        0.5 + 0.1 * k for k in range(len(e))))
        for i, e in enumerate(aux_exps)))
    (cls,) = [c for c in eri3_classes(ao, aux) if (c.la, c.lb, c.lP) == am]
    offs, n_aux = spherical_offsets(aux)
    return cls, offs, n_aux


@pytest.mark.parametrize("spherical_bra", [False, True],
                         ids=["wsperi3", "wssperi3"])
@pytest.mark.parametrize("am", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_the_warp_kernel_is_the_thread_per_cell_one_bitwise(am,
                                                            spherical_bra):
    cls, offs, n_aux = layout(am)
    cols, n_col = packed_columns([cls], spherical_bra=spherical_bra)
    cells = packed_cells(cls, cols, n_aux, offs)

    def run(warp):
        return launch_packed(cls, cells, np.full((n_col, n_aux), np.nan),
                             boys=boys, threads=5, spherical_aux=True,
                             spherical_bra=spherical_bra, warp=warp)
    assert np.array_equal(run(True), run(False), equal_nan=True)


@pytest.mark.parametrize("slots", [1, 3])
def test_several_phases_and_a_partial_one_change_nothing(slots):
    """Four aux records in phases of `slots`: the same output, bitwise."""
    am = (1, 1, 2)
    cls, offs, n_aux = layout(am, ((0.7, 0.3, 1.9, 0.12),))
    assert cls.n_aux_rec == 4
    cols, n_col = packed_columns([cls], spherical_bra=True)
    cells = packed_cells(cls, cols, n_aux, offs)
    kern = build_warp_eri3_kernel(am, spherical_bra=True, slots=slots)
    grid, block = warp_launch_shape(len(cells["dst"]))
    records = {PARAM_OF.get(k, k): np.array(v, dtype=float)
               for k, v in cls.arrays().items()
               if k in (*cls.BRA_ARRAYS, *cls.AUX_ARRAYS)}
    got = Interpreter(kern, launch=Launch(grid, block, WARP_LANES),
                      boys=boys).run(
        n_cell=len(cells["dst"]), n_bra_rec=cls.n_bra_rec,
        n_aux_rec=cls.n_aux_rec, n_col=n_col, n_aux=n_aux,
        **{OUT: np.full(n_col * n_aux, np.nan)}, **records,
        **{k: np.array(cells[k], dtype=np.int32)
           for k in PACKED_CELL_ARRAYS})[OUT]
    want = launch_packed(cls, cells, np.full((n_col, n_aux), np.nan),
                         boys=boys, spherical_aux=True, spherical_bra=True)
    assert np.array_equal(got, want.reshape(-1), equal_nan=True)


def test_the_warp_kernels_are_named_apart_with_the_packed_signature():
    for sb, head in ((False, "wsperi3_"), (True, "wssperi3_")):
        k = build_warp_eri3_kernel((3, 3, 6), spherical_bra=sb)
        u = build_eri3_kernel((3, 3, 6), packed=True, spherical_aux=True,
                              spherical_bra=sb)
        assert k.name == head + "336" and k.params == u.params
        assert any(isinstance(s, Barrier) for s in walk(k.body))
        assert not any(isinstance(s, AtomicAdd) for s in walk(k.body))
    # (ff|i): 49 slots over 32 lanes, two a lane, times 13 columns
    acc = {s.name: s for s in walk(build_warp_eri3_kernel(
        (3, 3, 6), spherical_bra=True).body) if isinstance(s, Decl)}
    assert acc[ACC_ARRAY].shape == (Lit(2 * 13),)
    with pytest.raises(ValueError, match="slots"):
        build_warp_eri3_kernel((1, 1, 1), slots=WARP_LANES + 1)
