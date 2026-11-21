"""The mixed-precision fully spherical kernel ON A CARD: `mssperi3_` at
(dd|f) and (ff|i).

THE ORACLE is test_kernel_eri3_mixed.py's: the host's FP64 contraction on
all three indices, at the declared FP32 tier (`F32_REL` times the
contracted scale), with the diagonal symmetry and the NaN fill exact. The
device forms its own fused multiply-adds in the FP32 region, so the card is
not the interpreter bitwise; the tier holds either.

SCOPE (minimal): two classes, the d transform and the top of the ladder;
(ff|i), L = 12, is `slow` and `high_l`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eri3_driver import build_packed_on_card
from test_kernel_device import _toolchain
from test_kernel_eri3_fully_spherical import _cls, _layout, _param
from test_kernel_eri3_mixed import check_f32

CLASSES = [(2, 2, 3), (3, 3, 6)]


def _case(backend: str, am, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    cells, _, n_col, n_aux, _ = _layout(am)
    got = build_packed_on_card([_cls(am)], [cells], n_col, n_aux, backend,
                               tc, tmp_path, spherical_aux=True,
                               spherical_bra=True, mixed=True)
    print(f"\n{backend} {am}: worst {check_f32(am, got):.1e} of F32_REL")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_cuda_mixed_eri3_is_the_contracted_host(am, request, tmp_path):
    _case("cuda", am, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("am", [_param(c) for c in CLASSES])
def test_hip_mixed_eri3_is_the_contracted_host(am, request, tmp_path):
    _case("hip", am, request, tmp_path)
