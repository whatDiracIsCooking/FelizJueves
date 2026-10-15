"""The oracle's tolerance tiers: one declared constant per precision (#338).

    |got - want| <= REL[precision] * scale

`scale` is the comparison's own magnitude (the largest |want|, or the sum of
absolute values a contraction carries -- each test states which). The fp64
tier is the `1e-12 * scale` gate every fp64 comparison against libcint and
the host paths has used since milestone 4; tests that predate this module
spell it locally, and `F64_REL` is that same number.

THE FP32 TIER IS DERIVED, NOT TUNED. `F32_REL` is the fp64 gate scaled by the
ratio of the two formats' unit roundoffs, 2^-23 / 2^-52 = 2^29: the same
number of units in the last place of `scale` (about 4.5e3), counted in
float32's ulp instead of float64's. It is ~5.4e-4. Nothing was measured to
choose it, and it is not to be widened to make a comparison pass: if an fp32
build misses it, the semantics are wrong or the build is not the claimed
one, and that is the finding.

WHAT IT MUST CATCH (#338's guardrail). An interpreter that computes an F32
program in fp64 and rounds only at the output agrees with itself and looks
plausible; test/test_f32_device.py shows that such an interpreter misses the
card by more than `F32_REL * scale` on the parity programs, while the
rounding interpreter meets it (and meets the card bitwise where no FMA can
form).

THE TIERS DO NOT CROSS. `assert_within` takes the build's precision and the
result's array, and refuses a float32 result at the fp64 tier and a float64
result at the fp32 tier: an fp32 build is never judged -- and so never
silently accepted or blamed -- by the fp64 gate, and an fp64 build is never
let off with the fp32 one.
"""

from __future__ import annotations

import numpy as np

from felizjueves.ir import Scalar

F64_REL = 1e-12
F32_REL = F64_REL * 2.0**29

REL = {Scalar.F64: F64_REL, Scalar.F32: F32_REL}
_DTYPE = {Scalar.F64: np.dtype(np.float64), Scalar.F32: np.dtype(np.float32)}


def assert_within(got: np.ndarray, want: np.ndarray, scale: float, *,
                  precision: Scalar, what: str = "result") -> float:
    """Assert `max |got - want| <= REL[precision] * scale`; return the error
    as a fraction of `scale`. `got` must be in `precision`'s dtype."""
    if precision not in REL:
        raise ValueError(f"no tolerance tier for {precision}")
    got = np.asarray(got)
    if got.dtype != _DTYPE[precision]:
        raise TypeError(
            f"{what}: a {got.dtype} result judged at the {precision.name} tier "
            "-- the tiers do not cross")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"{what}: scale {scale!r} is not a positive number")
    err = float(np.max(np.abs(got.astype(np.float64)
                              - np.asarray(want, dtype=np.float64)))) / scale
    assert err <= REL[precision], (
        f"{what}: max |got - want| / scale = {err:.3e} > "
        f"{REL[precision]:.3e} ({precision.name} tier)")
    return err
