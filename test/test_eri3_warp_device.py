"""The warp-per-cell fully spherical kernel (#749) ON A CARD: `wssperi3_` and
`ssperi3_` against the host, at (dd|f) and (ff|i).

THE ORACLE is the host (test_kernel_eri3_fully_spherical.py's `check`: the
Cartesian block contracted on all three indices within its bound, the
diagonal symmetry and the NaN fill), for the warp AND the thread-per-cell
output of the same card over the same cells. Interpreted the two forms are
bitwise (test_kernel_eri3_warp.py); on a card they need not be (each
compiles its own text, so its own fused multiply-adds), and bitwise
equality is printed, never asserted. The warp form's barriers and shared
memory run here for real: a missing barrier would show as a wrong block.

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
        spherical_aux=True, spherical_bra=True, warp=r)
        for r in (True, False)}
    worst = {r: check(am, got[r]) for r in got}
    same = np.array_equal(got[True], got[False], equal_nan=True)
    print(f"\n{backend} {am}: worst {worst[True]:.2f} eps warp, "
          f"{worst[False]:.2f} thread-per-cell; bitwise: {same}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_cuda_warp_eri3_is_the_contracted_host(am, request, tmp_path):
    _case("cuda", am, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_hip_warp_eri3_is_the_contracted_host(am, request, tmp_path):
    _case("hip", am, request, tmp_path)
