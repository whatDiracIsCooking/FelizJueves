"""emit/hip.py: a `Kernel` spelled as HIP C++, its golden file, and a compile.

THE SEAM (#27). The HIP backend consumes the identical `Kernel` the CUDA
backend does; every difference between the two texts is in the writers' three
hooks (prologue, warp reduction, atomic add), never in the IR.
`test_the_two_texts_differ_only_in_the_vendor_hooks` pins exactly which lines
differ, so a vendor difference that leaks out of the hooks shows up here.

THE GOLDEN FILE is `test/golden/hip/pq_L4_p2_q2.hip`: the same class as the
CUDA golden (`build_pq_kernel(4)`, (2,0|2,0), one batch), for `GFX1200`, so the
two can be read side by side. After an intended change, regenerate it (never
hand-edit) and review the diff:

    PYTHONPATH=src uv run python -c "from felizjueves.kernel import \
build_pq_kernel; from felizjueves.device import GFX1200; from \
felizjueves.emit.hip import emit; print(emit(build_pq_kernel(4), GFX1200), \
end='')" > test/golden/hip/pq_L4_p2_q2.hip

THE COMPILE carries the `hip` marker and needs `hipcc` only: a device compiler
needs no card, so it is green on a device-less box and proves nothing about a
number. Running the kernel on the card is the kernel tier's (#28).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from felizjueves.device import GFX1200, SM_86, Arch, Vendor
from felizjueves.emit import cuda, hip
from felizjueves.kernel import build_pq_kernel

L = 4
GOLDEN = Path(__file__).resolve().parent / "golden" / "hip" / "pq_L4_p2_q2.hip"


def test_the_golden_kernel_is_unchanged() -> None:
    assert hip.emit(build_pq_kernel(L), GFX1200) == GOLDEN.read_text()


def test_both_backends_consume_the_identical_kernel() -> None:
    # The seam, asserted: divergence is confined to the writers.
    k = build_pq_kernel(L)
    assert cuda.emit(k, SM_86) != hip.emit(k, GFX1200)  # text differs
    assert k == build_pq_kernel(L)  # and nothing mutated it


def test_the_two_texts_differ_only_in_the_vendor_hooks() -> None:
    k = build_pq_kernel(L)
    c = cuda.emit(k, SM_86).splitlines()
    h = hip.emit(k, GFX1200).splitlines()
    # The prologue: HIP's is CUDA's plus the runtime include.
    assert h[0] == c[0].replace("emit.cuda", "emit.hip")
    assert h[1] == c[1].replace(SM_86.name, GFX1200.name)
    assert h[2] == "#include <hip/hip_runtime.h>"
    body_c, body_h = c[2:], h[3:]
    assert len(body_c) == len(body_h)
    changed = {(a.strip(), b.strip())
               for a, b in zip(body_c, body_h, strict=True) if a != b}
    # The warp reduction: only the mask's width differs (HIP's is 64-bit).
    assert changed == {
        ("fj_v += __shfl_xor_sync(0xffffffffu, fj_v, fj_o, 32);",
         "fj_v += __shfl_xor_sync(0xffffffffull, fj_v, fj_o, 32);"),
    }


def test_the_wavefront_is_read_off_the_arch() -> None:
    # A hypothetical wave64 AMD arch changes the shuffle and nothing else.
    wide = Arch(Vendor.AMD, "gfx_test", wavefront=64)
    k = build_pq_kernel(L)
    a, b = hip.emit(k, GFX1200).splitlines(), hip.emit(k, wide).splitlines()
    assert len(a) == len(b)
    changed = {(x.strip(), y.strip()) for x, y in zip(a, b, strict=True)
               if x != y}
    assert changed == {
        ("// arch gfx1200, wavefront 32; fp64 throughout.",
         "// arch gfx_test, wavefront 64; fp64 throughout."),
        ("for (int fj_o = 16; fj_o > 0; fj_o /= 2) {",
         "for (int fj_o = 32; fj_o > 0; fj_o /= 2) {"),
        ("fj_v += __shfl_xor_sync(0xffffffffull, fj_v, fj_o, 32);",
         "fj_v += __shfl_xor_sync(0xffffffffffffffffull, fj_v, fj_o, 64);"),
    }


def test_emit_takes_an_amd_arch_value() -> None:
    k = build_pq_kernel(0)
    with pytest.raises(TypeError, match="device.Arch"):
        hip.emit(k, "gfx1200")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="AMD"):
        hip.emit(k, SM_86)


@pytest.mark.hip
@pytest.mark.parametrize("L_", [0, 1, L, 8])
def test_it_compiles(hipcc, compile_device_source, L_: int) -> None:
    # A device compiler needs no card: this is green on a device-less box and
    # proves nothing about a number.
    compile_device_source(hipcc, hip.emit(build_pq_kernel(L_), GFX1200),
                          link=False)
