"""`gpu.emit_boys()`'s `eval_boys<M>` on each card, against `reference.boys`.

The gpu backend's Boys is the vendor `fj_boys<M>` with the series loop
replaced by a fixed polynomial per order (emit/gpu.py, "The Boys series"),
so it runs through test_kernel_device.py's Boys harness, its T grid and its
upward-branch bound unchanged, for every order the headers are written for,
`M` in 0..BOYS_MAX, each switch `T = M + 8` sampled on both sides down to the
adjacent doubles. Only the series branch's bound is restated, u = 2^-53:

- `eval_boys_series<M>(x)` is Horner over the n + 1 rounded `c_k`, one fma a
  step; every `c_k` and `x` is positive, so the term `c_k x^k` carries its
  coefficient's rounding and at most n fma roundings, (n + 1) u, and the
  truncated tail is under 1 u of the sum (by construction, `gpu.boys_series`).
  `exp` <= 2u and the product 1u: (n + 6) u at M, one u spare for the
  second-order terms;
- each downward step rounds four times and passes its error on with weight
  <= 1, as in `fj_boys`: F_m within (n + 6 + 4 (M - m)) u.

Every point is also held to BOYS_REL = 1e-13, which every bound sits under.
Measured worst: printed by the test (`-s`).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from test_kernel_device import (
    BOYS_HARNESS,
    BOYS_LAUNCH,
    BOYS_REF_ERR,
    BOYS_REL,
    BOYS_T,
    PORTABLE,
    U,
    _toolchain,
    boys_bound,
)

from felizjueves.emit import gpu
from felizjueves.emit._c import BOYS_SWITCH_PAD
from felizjueves.reference.boys import boys

TOPS = range(0, gpu.BOYS_MAX + 1)
# test_kernel_device's grid already holds the switches of M <= 16
T_GRID = sorted({*BOYS_T, *(t for M in TOPS for sw in [float(M + BOYS_SWITCH_PAD)]
                            for t in (sw - 0.5, math.nextafter(sw, 0.0), sw,
                                      math.nextafter(sw, math.inf), sw + 0.5))})


def eval_boys_bound(M: int, T: float) -> list[float]:
    """The per-order forward error bound of `eval_boys<M>(T)`, F_0..F_M
    (module docstring), reference.boys's own error included."""
    if T >= M + BOYS_SWITCH_PAD:
        return boys_bound(M, T)  # the same upward branch, the same text
    n = len(gpu.boys_series(M)) - 1
    return [(n + 6 + 4 * (M - m)) * U + BOYS_REF_ERR for m in range(M + 1)]


def _run(backend: str, request, compile_device_source, run_device_binary,
         tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    for name, text in gpu.emit_boys().items():
        (tmp_path / name).write_text(text)
    prologue = "#include <hip/hip_runtime.h>" if backend == "hip" else ""
    launches = "\n".join(BOYS_LAUNCH.format(M=M, w=M + 1) for M in TOPS)
    source = BOYS_HARNESS.format(
        prologue=prologue, portable=PORTABLE,
        boys=f'#include "{gpu.BOYS_HEADER}"', launches=launches,
        width=sum(M + 1 for M in TOPS))
    assert source.count("fj_boys<M>(") == 1
    binary = compile_device_source(
        tc, source.replace("fj_boys<M>(", "eval_boys<M>("))
    src, dst = tmp_path / "boys.in", tmp_path / "boys.out"
    np.concatenate([[float(len(T_GRID))], T_GRID]).tofile(src)
    run_device_binary(binary, str(src), str(dst), vram=backend)
    got = np.fromfile(dst, dtype=np.float64)
    assert got.shape == (len(T_GRID) * sum(M + 1 for M in TOPS),)
    worst = {"series": (0.0, None), "upward": (0.0, None)}
    bad = []
    at = 0
    for M in TOPS:
        for T in T_GRID:
            bounds = eval_boys_bound(M, T)
            branch = "series" if T < M + BOYS_SWITCH_PAD else "upward"
            for m in range(M + 1):
                g, ref = got[at + m], boys(m, T)
                rel = abs(g - ref) / ref
                assert bounds[m] < BOYS_REL
                if not (math.isfinite(g) and rel <= bounds[m]):
                    bad.append((M, m, T, g, rel, bounds[m]))
                if rel > worst[branch][0]:
                    worst[branch] = (rel, (M, m, T))
            at += M + 1
    print(f"\n{backend} eval_boys<M>: M in 0..{gpu.BOYS_MAX}, {len(T_GRID)} T "
          f"in [0, 1e6]; worst rel err {worst['series'][0]:.2e} at (M, m, T) = "
          f"{worst['series'][1]} below the switch, {worst['upward'][0]:.2e} "
          f"at {worst['upward'][1]} from it")
    assert not bad, f"eval_boys outside its forward error bound at {bad[:5]}"


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_eval_boys_matches_reference_boys(request, compile_device_source,
                                               run_device_binary, tmp_path) -> None:
    """The gpu backend's Boys headers on the RTX 3080, T across [0, 1e6]."""
    _run("cuda", request, compile_device_source, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_eval_boys_matches_reference_boys(request, compile_device_source,
                                              run_device_binary, tmp_path) -> None:
    """The gpu backend's Boys headers on the RX 9060 XT, T across [0, 1e6]."""
    _run("hip", request, compile_device_source, run_device_binary, tmp_path)
