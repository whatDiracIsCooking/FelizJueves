"""F32 in the IR (#338), host side: the parity programs, their spelling, the
tolerance tier, and compile-only checks on both device compilers.

Fast tier, except the `cuda` / `hip` compile cases (a compiler, no card).
The card's half of the guardrail -- the interpreter's rounding against a
DEVICE run -- is test/test_f32_device.py.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from f32_programs import (
    BIG,
    DISCRIMINATING,
    contracting_kernel,
    exact_kernel,
    inputs,
    numpy_exact,
    run,
)
from oracle.tolerance import F32_REL, F64_REL, REL, assert_within

from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.ir import Scalar, interp

EMIT = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}


# ---------------------------------------------------------------------------
# The tolerance tier: one declared constant, derived, and the tiers do not cross
# ---------------------------------------------------------------------------


def test_the_fp32_tier_is_the_fp64_gate_in_float32_ulps() -> None:
    assert F64_REL == 1e-12
    assert F32_REL == F64_REL * (2.0**-23 / 2.0**-52)
    assert REL == {Scalar.F64: F64_REL, Scalar.F32: F32_REL}


def test_the_tiers_do_not_cross() -> None:
    want = np.ones(3)
    assert_within(np.ones(3, np.float32), want, 1.0, precision=Scalar.F32)
    assert_within(np.ones(3), want, 1.0, precision=Scalar.F64)
    with pytest.raises(TypeError, match="do not cross"):
        assert_within(np.ones(3, np.float32), want, 1.0, precision=Scalar.F64)
    with pytest.raises(TypeError, match="do not cross"):
        assert_within(np.ones(3), want, 1.0, precision=Scalar.F32)
    # an fp32 result one float32 ulp off passes its tier and fails fp64's
    off = np.nextafter(np.float32(1.0), np.float32(2.0))
    assert_within(np.array([off]), [1.0], 1.0, precision=Scalar.F32)
    with pytest.raises(AssertionError):
        assert_within(np.array([float(off)]), [1.0], 1.0, precision=Scalar.F64)


# ---------------------------------------------------------------------------
# The parity programs, interpreted: exact against numpy float32, op by op
# ---------------------------------------------------------------------------


def test_the_exact_program_is_numpy_float32_bitwise() -> None:
    data = inputs()
    got, want = run(exact_kernel(), data), numpy_exact(data)
    for k in want:
        assert got[k].dtype == want[k].dtype
        assert np.array_equal(got[k], want[k]), k
    # the discriminating columns are nonzero rounding effects, not zeros
    a = data["x"]
    assert np.all(got["out"][:, 6] != a)  # (a + 2^24) - 2^24 lost a
    assert np.count_nonzero(got["out"][:, 0]) > len(a) // 4


def test_without_rounding_the_exact_program_is_not_float32(
        monkeypatch: pytest.MonkeyPatch) -> None:
    data = inputs()
    want = numpy_exact(data)
    monkeypatch.setattr(interp, "_round32", lambda x: x)
    got = run(exact_kernel(), data)
    for col in DISCRIMINATING["f32_exact"]:
        scale = float(np.max(np.abs(want["out"][:, col])))
        err = float(np.max(np.abs(got["out"][:, col].astype(np.float64)
                                  - want["out"][:, col]))) / scale
        assert err > F32_REL, (col, err)
    # the unrounded (a + 2^24) - 2^24 is a itself
    assert np.array_equal(got["out"][:, 6], data["x"])


def test_the_contracting_program_interprets() -> None:
    data = inputs()
    got = run(contracting_kernel(), data)["out"]
    assert got.dtype == np.float32 and np.all(np.isfinite(got))
    assert np.all(got[:, 12] % 2 == 0)  # integers above 2^24 are even
    assert BIG == 2.0**24


# ---------------------------------------------------------------------------
# The spelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(EMIT))
def test_the_f32_spelling(backend: str) -> None:
    emit, arch = EMIT[backend]
    exact = emit(exact_kernel(), arch)
    contracting = emit(contracting_kernel(), arch)
    assert "fp32 and fp64 as declared" in exact
    assert "const float* x" in exact and "const double* d" in exact
    assert "float s = (a + b);" in exact
    assert "sqrtf(s)" in exact and "(((float)d[i]) - a)" in exact
    assert "(((double)s) - d[i])" in exact and "16777216.0f" in exact
    assert "float acc = 0.0f;" in exact and "(s < 2.5f)" in exact
    assert "((s + (-1.0f * a)) + (-1.0f * b))" in exact
    assert "expf((0.0f - a))" in contracting
    assert "__shared__ float sh[64];" in contracting
    assert "((PQ_x * r2) + (2.0f * r1))" in contracting
    for text in (exact, contracting):
        assert "fmaf" not in text and "fma(" not in text  # the IR has no FMA
        # neither program holds an F64 literal, so no unsuffixed float
        # literal may appear: one would silently widen its fp32 expression
        code = "\n".join(ln for ln in text.splitlines()
                         if not ln.lstrip().startswith("//"))
        assert not re.findall(r"\d\.\d+(?:e[+-]?\d+)?(?![\df])", code)


# ---------------------------------------------------------------------------
# Compile-only: both device compilers accept the text, with no warning
# ---------------------------------------------------------------------------


def _compile(backend: str, request, compile_device_source) -> None:
    emit, arch = EMIT[backend]
    tc = request.getfixturevalue("nvcc" if backend == "cuda" else "hipcc")
    for k in (exact_kernel(), contracting_kernel()):
        compile_device_source(tc, emit(k, arch), link=False)


@pytest.mark.cuda
def test_cuda_compiles_the_f32_programs(request, compile_device_source) -> None:
    _compile("cuda", request, compile_device_source)


@pytest.mark.hip
def test_hip_compiles_the_f32_programs(request, compile_device_source) -> None:
    _compile("hip", request, compile_device_source)
