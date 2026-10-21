"""`gpu.emit(..., any_wavefront=True)`'s [p|q] on each card, against the oracle.

The width-generic text (emit/gpu.py, "Unless the consumer erases it") is the
fixed-width text with the width spelled `FJ_WAVEFRONT`; this runs it through
test_kernel_device.py's harness, inputs and reference, built with the
header's default `FJ_WAVEFRONT` = 32, on the RTX 3080 and the RX 9060 XT.
Both cards are 32-lane: no wave64 card is on this box, so the 64 reading of
the same text is compiled (test_emit_gpu.py), never run.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_device import (
    CASES,
    J_REL,
    PORTABLE,
    PQ_HARNESS,
    _inputs,
    _pack,
    _reference,
    _toolchain,
)

from felizjueves.device import SM_86
from felizjueves.emit import gpu
from felizjueves.kernel import build_pq_kernel, launch_shape

LANES = 32  # FJ_WAVEFRONT's default, and both cards' width


def _run(backend: str, L: int, request, compile_device_source,
         run_device_binary, tmp_path) -> None:
    tc = _toolchain(backend, request)
    k = build_pq_kernel(L)
    kpath = tmp_path / f"{k.name}.{tc.ext}"
    kpath.write_text(gpu.emit(k, SM_86, any_wavefront=True))
    for name, text in gpu.emit_boys().items():
        (tmp_path / name).write_text(text)
    binary = compile_device_source(tc, PQ_HARNESS.format(
        kernel=kpath, portable=PORTABLE, kernel_name=k.name))
    for case, (wide, tq, n_bra, n_ket) in CASES.items():
        a = _inputs(L, wide=wide, tau_quantile=tq, seed=1000 * L + len(case),
                    n_bra=n_bra, n_ket=n_ket)
        assert launch_shape(n_bra, LANES) == (n_bra, (1, LANES))
        src, dst = tmp_path / f"{case}.in", tmp_path / f"{case}.out"
        _pack(L, a, LANES).tofile(src)
        run_device_binary(binary, str(src), str(dst), vram=backend)
        got = np.fromfile(dst, dtype=np.float64).reshape(n_bra, -1)
        want, scale, _ = _reference(L, a)
        assert np.all(np.isfinite(got)), f"{case}: non-finite J_p"
        err = float(np.max(np.abs(got - want) / scale))
        assert err <= J_REL, f"L={L} {case}: {err:.3e} > {J_REL:.0e}"


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("L", [0, 2, 4])
def test_cuda_any_wavefront_j_p_matches_the_reference(
        L, request, compile_device_source, run_device_binary, tmp_path) -> None:
    _run("cuda", L, request, compile_device_source, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("L", [0, 2, 4])
def test_hip_any_wavefront_j_p_matches_the_reference(
        L, request, compile_device_source, run_device_binary, tmp_path) -> None:
    _run("hip", L, request, compile_device_source, run_device_binary, tmp_path)
