"""Dynamic precision with screening (#754) ON A CARD: (dd|d) on
test_kernel_eri3_dynamic.py's chain, ONE runner over `ssperi3_` and
`mssperi3_` (`eri3_driver.build_dynamic_on_card`).

THE ORACLES are test_kernel_eri3_dynamic.py's, with one change: the
card's FP64 cells are not claimed bitwise the interpreter (the card forms
its own fused multiply-adds), so they are held to the host at the FP64
form's bound (test_kernel_eri3_fully_spherical.py's `tolerance`). FP32
cells are at the declared FP32 tier, screened slots exactly 0.0 with the
host's |B| at most `tau_screen`, and no slot is left NaN.

SCOPE (minimal): one class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import test_kernel_eri3_fully_spherical as fs
from eri3_driver import build_dynamic_on_card
from oracle.tolerance import F32_REL
from test_kernel_device import _toolchain
from test_kernel_eri3_dynamic import chain, host, slots

AM = (2, 2, 2)


def _case(backend: str, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    _, _, cls, cells, _, n_col, n_aux, *_, (tau_screen, _, bins) = chain(AM)
    got = build_dynamic_on_card(cls, cells, bins, n_col, n_aux, backend, tc,
                                tmp_path).reshape(-1)
    tol64 = fs.tolerance(AM)
    for k in bins["fp64"]:
        want, scale = host(AM, k)
        assert np.all(np.abs(got[slots(AM, k)] - want) <= tol64 * scale)
    for k in bins["fp32"]:
        want, scale = host(AM, k)
        assert np.all(np.abs(got[slots(AM, k)] - want) <= F32_REL * scale)
    for k in bins["screened"]:
        assert np.all(got[slots(AM, k)] == 0.0)
        assert np.max(np.abs(host(AM, k)[0])) <= tau_screen
    assert not np.isnan(got).any()
    print(f"\n{backend} {AM}: " + str({b: len(v) for b, v in bins.items()}))


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_dynamic_eri3_keeps_each_bins_claim(request, tmp_path):
    _case("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_dynamic_eri3_keeps_each_bins_claim(request, tmp_path):
    _case("hip", request, tmp_path)
