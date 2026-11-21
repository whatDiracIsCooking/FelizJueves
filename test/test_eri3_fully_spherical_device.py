"""The fully spherical packed three-center kernel (FelizJueves#745) ON A
CARD: `ssperi3_{a}{a}{p}`, a = 0..3, p = 0..6.

THE WITNESS AND THE BOUND are test_kernel_eri3_fully_spherical.py's: per
cell, the host's Cartesian block contracted by `aux_harmonics` on all
three indices, within its bound of the contracted scale; a diagonal pair's
block symmetric in `(mu, nu)` exactly; exactly the class's elements stored
(the output starts NaN). The card's arithmetic is the interpreter's (no
fast math, no reassociation), so the host bound carries over unchanged.
The runner repeats the sequence `RUNS` times bit-identically.

SCOPE, by decision (#745): the 28 classes (aa|p); L <= 4 at the fast cut,
the rest `slow`, and L > 8 also `high_l`, run by hand. One program per
class.

NOT A MEASUREMENT: nothing is written to `data/measured/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eri3_driver import build_packed_on_card
from test_kernel_device import BACKEND, _toolchain
from test_kernel_eri3_fully_spherical import (
    CLASSES,
    EPS,
    _cls,
    _layout,
    _param,
    check,
    tolerance,
)


def _case(backend: str, am, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    cls = _cls(am)
    cells, _, n_col, n_aux, _ = _layout(am)
    got = build_packed_on_card([cls], [cells], n_col, n_aux, backend, tc,
                               tmp_path, spherical_aux=True,
                               spherical_bra=True)
    worst = check(am, got)
    print(f"\n{backend} {BACKEND[backend][1].name} {am}: worst {worst:.2f} "
          f"eps of the contracted scale, bound {tolerance(am) / EPS:.0f}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_cuda_fully_spherical_eri3_is_the_contracted_host(am, request,
                                                          tmp_path):
    _case("cuda", am, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_hip_fully_spherical_eri3_is_the_contracted_host(am, request,
                                                         tmp_path):
    _case("hip", am, request, tmp_path)
