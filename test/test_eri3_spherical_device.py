"""The spherical-aux packed three-center kernel (FelizJueves#743) ON A
CARD: `speri3_{l}{l}{l}`, l = 0..3.

THE WITNESS AND THE BOUND are test_kernel_eri3_spherical.py's: per cell,
the host's Cartesian block contracted by `aux_harmonics(lP)`, within
`(2 TOL_ULPS + nq + lP) eps` of the contracted scale. The card's
arithmetic is the interpreter's (no fast math, no reassociation), so the
host bound carries over unchanged. The output starts NaN, so an element no
cell stores fails too, and the runner repeats the sequence `RUNS` times
bit-identically.

SCOPE, by decision (#743): (ss|s) and (pp|p) at the fast cut; (dd|d) and
(ff|f) `slow`, and (ff|f), L = 9, also `high_l` (above `DEVICE_L_MAX`),
run by hand. One program per class. The consumer's full ladder
(FreeBreakfast#1736) compiles and runs the rest.

NOT A MEASUREMENT: nothing is written to `data/measured/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import test_kernel_eri3 as te
from eri3_driver import build_packed_on_card
from test_kernel_device import BACKEND, _toolchain
from test_kernel_eri3_spherical import (
    CLASSES,
    EPS,
    _cls,
    _host,
    _layout,
    _param,
)

from felizjueves.ao.shell import cartesians


def _case(backend: str, am, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    cls = _cls(am)
    cells, cols, n_col, n_aux, offs = _layout(cls)
    got = build_packed_on_card([cls], [cells], n_col, n_aux, backend, tc,
                               tmp_path, spherical_aux=True).reshape(-1)
    tol = (2 * te.TOL_ULPS + len(cartesians(cls.lP)) + cls.lP) * EPS
    stored = np.zeros(got.shape, dtype=bool)
    worst = 0.0
    for k in range(cls.n_cell):
        want, bound, at = _host(cls, k, cols, n_aux, offs)
        err = np.abs(got[at] - want)
        pos = bound > 0
        worst = max(worst, float(np.max(err[pos] / (EPS * bound[pos]),
                                        initial=0.0)))
        assert np.all(err <= tol * bound), (am, cls.cells[k])
        stored[at] = True
    print(f"\n{backend} {BACKEND[backend][1].name} {am}: worst {worst:.2f} "
          f"eps of the contracted scale, bound {tol / EPS:.0f}")
    assert not np.isnan(got[stored]).any()
    assert np.isnan(got[~stored]).all()


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_cuda_spherical_eri3_is_the_contracted_host(am, request, tmp_path):
    _case("cuda", am, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_hip_spherical_eri3_is_the_contracted_host(am, request, tmp_path):
    _case("hip", am, request, tmp_path)
