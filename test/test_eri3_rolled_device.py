"""The rolled fully spherical kernel (#748) ON A CARD: `rssperi3_` and
`ssperi3_` against the host, at (dd|f) and (ff|i).

THE ORACLE is the host (test_kernel_eri3_fully_spherical.py's `check`: the
Cartesian block contracted on all three indices within its bound, the
diagonal symmetry and the NaN fill), for the rolled AND the unrolled
output of the same card over the same cells. Interpreted, the two forms
are bitwise (test_kernel_eri3_rolled.py); on a card they need not be: the
device compiler contracts products into fused multiply-adds by the text it
sees, and the two texts differ. On the GH200 (nvcc 13.0, `sm_90`) they
are NOT bitwise, both within the bound (#748). Bitwise equality is printed,
never asserted.

SCOPE (minimal, by decision): two classes, the d transform and the top of
the ladder. (ff|i), L = 12, is `slow` and `high_l`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from eri3_driver import build_packed_on_card
from test_kernel_device import _toolchain
from test_kernel_eri3_fully_spherical import _cls, _layout, _param, check

CLASSES = [(2, 2, 3), (3, 3, 6)]


def _case(backend: str, am, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    cls = _cls(am)
    cells, _, n_col, n_aux, _ = _layout(am)
    got = {r: build_packed_on_card(
        [cls], [cells], n_col, n_aux, backend, tc, tmp_path / str(r),
        spherical_aux=True, spherical_bra=True, rolled=r)
        for r in (True, False)}
    worst = {r: check(am, got[r]) for r in got}
    same = np.array_equal(got[True], got[False], equal_nan=True)
    print(f"\n{backend} {am}: worst {worst[True]:.2f} eps rolled, "
          f"{worst[False]:.2f} unrolled; bitwise: {same}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_cuda_rolled_eri3_is_the_contracted_host(am, request, tmp_path):
    _case("cuda", am, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_hip_rolled_eri3_is_the_contracted_host(am, request, tmp_path):
    _case("hip", am, request, tmp_path)
