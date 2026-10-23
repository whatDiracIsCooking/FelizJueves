"""The `mixed` lever's [p|q] kernels (#346) on the card, at the fp32 tier.

The same harness as test_kernel_device.py's lever arms (`_batched_source`:
the cell's whole launch sequence; `_case`: every `CASES` input and its
oracle, `reference.pq.one_centre` contracted with `D_q` in fp64), held to
THE fp32 tier, `F32_REL * scale` (test/oracle/tolerance.py, the one declared
fp32 tolerance) -- the lever computes the recurrence in fp32 by design, so
the fp64 `J_REL` is not its pin and a tier looser than `F32_REL` is not
admitted either. `J` comes back fp64 (the boundary), so the tier is applied
to that array directly (test/test_kernel_mixed.py's docstring says why
not `assert_within`).

The card may differ from the interpreter by FMA contraction of the
recurrence's `PQ_d * a + c` terms (one rounding where the interpreter has
two) and by `fj_boys`'s own fp64 rounding; both are far inside the tier, as
test_f32_device.py's contracting program witnesses on each card. The error
is also asserted ABOVE the fp64 tier from L = 1: the card really ran fp32.

Cells: one batch at L = 0..4 and the split cell L = 4 x 3 batches (atomic
write-back), each alone and composed with `fused`.

Run on each card (the GPU agent), compiling the L <= 4 ladder:
    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda -- -k mixed_j_p
    (the same with --backend hip)
"""

from __future__ import annotations

import numpy as np
import pytest
from oracle.tolerance import F32_REL, F64_REL
from test_kernel_device import (
    BACKEND,
    CASES,
    _batched_source,
    _case,
    _pack,
    _toolchain,
)

from felizjueves.j_levers.fused import FUSED
from felizjueves.j_levers.mixed import MIXED
from felizjueves.mmd.kernel import launch_shape, pq_spec


def _cells():
    out = []
    for lv, tag in (((MIXED,), "mixed"), ((FUSED, MIXED), "fused+mixed")):
        out += [pytest.param(pq_spec(L, lv), id=f"L{L}-b1-{tag}")
                for L in range(5)]
        out.append(pytest.param(pq_spec(4, lv, 3), id=f"L4-b3-{tag}"))
    return out


CELLS = _cells()


def _run_mixed(backend: str, spec, request, compile_device_source,
               run_device_binary, tmp_path) -> None:
    tc = _toolchain(backend, request)
    src, _, _ = _batched_source(backend, spec, tmp_path)
    binary = compile_device_source(tc, src)
    arch = BACKEND[backend][1]
    L = spec.total_am
    results = []
    for case in CASES:
        a, (want, scale, t_max) = _case(L, case)
        n_bra = a["P"].shape[0]
        _, (_, lanes) = launch_shape(n_bra, arch.wavefront)
        src_in, dst = tmp_path / f"{case}.in", tmp_path / f"{case}.out"
        _pack(L, a, lanes).tofile(src_in)
        run_device_binary(binary, str(src_in), str(dst), "0", "1",
                          vram=backend)
        got = np.fromfile(dst, dtype=np.float64).reshape(2, n_bra, -1)[0]
        assert got.shape == want.shape
        assert np.all(np.isfinite(got)), f"{case}: non-finite J_p"
        err = float(np.max(np.abs(got - want) / scale))
        assert err <= F32_REL, (
            f"{spec.key()} {case} on {arch.name}: max |J - want| / scale = "
            f"{err:.3e} > F32_REL {F32_REL:.3e} (T up to {t_max:.1f})")
        if L >= 1:
            assert err > F64_REL, (
                f"{spec.key()} {case} on {arch.name}: {err:.3e} meets the "
                "fp64 tier -- the recurrence did not run in fp32")
        results.append((case, err, t_max))
    print(f"\n{backend} {arch.name} {spec.key()}: " + ", ".join(
        f"{c} err/scale {e:.2e} (T<={t:.0f})" for c, e, t in results))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", CELLS)
def test_cuda_mixed_j_p_meets_the_fp32_tier(spec, request,
                                            compile_device_source,
                                            run_device_binary,
                                            tmp_path) -> None:
    """A mixed-lever cell's launch sequence on the RTX 3080, against the oracle."""
    _run_mixed("cuda", spec, request, compile_device_source,
               run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", CELLS)
def test_hip_mixed_j_p_meets_the_fp32_tier(spec, request,
                                           compile_device_source,
                                           run_device_binary,
                                           tmp_path) -> None:
    """A mixed-lever cell's launch sequence on the RX 9060 XT, against the oracle."""
    _run_mixed("hip", spec, request, compile_device_source,
               run_device_binary, tmp_path)
