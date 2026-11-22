"""Dynamic precision with screening for the fully spherical three-center
kernel (#754), interpreted: `eri3_drive.cell_bounds`, `precision_split`
and `launch_dynamic` -- per cell, screened below `tau_screen`, the
mixed-precision kernel (#753) below `tau32`, the FP64 kernel above.

THE ORACLES, one per claim:

- THE BOUND is rigorous (Cauchy-Schwarz, eri3_drive's module docstring):
  every element of every cell's block, the host's FP64 Cartesian block
  contracted on all three indices, is at most the cell's bound.
- FP64 CELLS are the full FP64 run's, BITWISE: the same kernel over the
  same cells, each writing only its own slots.
- FP32 CELLS are the host's contraction at the declared FP32 tier
  (`F32_REL` times the contracted scale), as test_kernel_eri3_mixed.py.
- SCREENED CELLS are exactly 0.0, and the host's |B| there is at most
  `tau_screen` -- the error the screen admits, by the bound.

SCOPE (minimal): (pp|d) and (dd|d) on a chain of AO shells from touching to
far apart, with a near and a far aux shell, the thresholds placed between
the class's bounds so that every bin holds cells. The card run is
test_eri3_dynamic_device.py.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
import test_kernel_eri3 as te
from oracle.tolerance import F32_REL

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit import eri3_drive as dr
from felizjueves.eri.density_fit.eri3 import (
    class_block,
    eri3_classes,
    packed_cells,
    packed_columns,
    spherical_offsets,
)
from felizjueves.eri.density_fit.kernel_eri3 import aux_harmonics
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

CLASSES = [(1, 1, 2), (2, 2, 2)]


@cache
def chain(am):
    """Four AO shells of `la` along z, 0 to 16 bohr, and two aux shells of
    `lP`; the class, its layout, its bounds and thresholds that fill every
    bin: `tau_screen` above the smallest bound, `tau32` at the median."""
    la, _, lP = am
    ao = Basis(tuple(Shell(la, (0.0, 0.0, z), (1.2, 0.4), (0.6, 0.5))
                     for z in (0.0, 1.5, 4.0, 16.0)))
    aux = Basis(tuple(Shell(lP, (0.0, 0.5, z), (0.9,), (1.0,))
                      for z in (0.0, 6.0)))
    (cls,) = [c for c in eri3_classes(ao, aux) if (c.la, c.lb, c.lP) == am]
    offs, n_aux = spherical_offsets(aux)
    cols, n_col = packed_columns([cls], spherical_bra=True)
    cells = packed_cells(cls, cols, n_aux, offs)
    bounds = dr.cell_bounds(cls, ao, aux, pq=pq)
    b = sorted(set(bounds))
    tau_screen, tau32 = b[0] * 1.01, float(np.median(bounds)) * 1.0001
    bins = dr.precision_split(bounds, tau_screen=tau_screen, tau32=tau32)
    assert all(bins[k] for k in dr.BINS), bins
    return ao, aux, cls, cells, cols, n_col, n_aux, offs, bounds, (
        tau_screen, tau32, bins)


def host(am, k):
    """Cell `k`'s host block, `[mu][nu][m]`, and its contracted scale."""
    ao, aux, cls, *_ = chain(am)
    a, b, p = cls.cells[k]
    c = [np.array(aux_harmonics(x)) for x in am]
    blk = np.array(class_block(cls, k, pq=pq))
    scale = np.array(te._scale(ao.shells[a], ao.shells[b], aux.shells[p]))
    return (np.einsum("ia,jb,mq,abq->ijm", c[0], c[1], c[2], blk),
            np.einsum("ia,jb,mq,abq->ijm", *(np.abs(x) for x in c), scale))


def slots(am, k):
    _, _, cls, cells, _, _, n_aux, *_ = chain(am)
    sA, sB, sP = (2 * x + 1 for x in am)
    return (cells["dst"][k] + n_aux * np.arange(sA * sB).reshape(sA, sB)
            [:, :, None] + np.arange(sP))


@pytest.mark.parametrize("am", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_every_element_is_within_its_cells_bound(am):
    *_, bounds, _ = chain(am)
    for k, bound in enumerate(bounds):
        assert np.max(np.abs(host(am, k)[0])) <= bound


def check_dynamic(am, got: np.ndarray) -> dict[str, int]:
    """The four claims of the module docstring on `got`, flat; the bin
    sizes."""
    _, _, cls, cells, _, n_col, n_aux, *_ , (tau_screen, _, bins) = chain(am)
    full = dr.launch_packed(cls, cells, np.full((n_col, n_aux), np.nan),
                            boys=boys, threads=5, spherical_aux=True,
                            spherical_bra=True).reshape(-1)
    got = got.reshape(-1)
    for k in bins["fp64"]:
        assert np.array_equal(got[slots(am, k)], full[slots(am, k)])
    for k in bins["fp32"]:
        want, scale = host(am, k)
        assert np.all(np.abs(got[slots(am, k)] - want) <= F32_REL * scale)
    for k in bins["screened"]:
        assert np.all(got[slots(am, k)] == 0.0)
        assert np.max(np.abs(host(am, k)[0])) <= tau_screen
    assert not np.isnan(got).any()  # every slot of every cell, one bin each
    return {b: len(v) for b, v in bins.items()}


@pytest.mark.parametrize("am", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_the_dynamic_launch_keeps_each_bins_claim(am):
    _, _, cls, cells, _, n_col, n_aux, *_, (_, _, bins) = chain(am)
    got = dr.launch_dynamic(cls, cells, np.full((n_col, n_aux), np.nan),
                            bins, boys=boys, threads=5)
    print(f"\n{am}: {check_dynamic(am, got)}")


def test_the_split_refuses_crossed_thresholds():
    with pytest.raises(ValueError, match="tau_screen"):
        dr.precision_split([1.0], tau_screen=2.0, tau32=1.0)
    assert dr.precision_split([0.5, 1.5, 3.0], tau_screen=1.0, tau32=2.0) == {
        "screened": (0,), "fp32": (1,), "fp64": (2,)}
