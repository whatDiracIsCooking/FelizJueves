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
from test_ir import synthetic_inline_kernel

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


INLINE_GOLDEN = GOLDEN.parent / "inline_probe.hip"


def test_the_inline_golden_is_unchanged() -> None:
    # #149: a kernel splicing a synthetic Fragment. Regenerate (never hand-edit):
    #   PYTHONPATH=src:test uv run python -c "from test_ir import \
    #   synthetic_inline_kernel as k; from felizjueves.device import GFX1200; \
    #   from felizjueves.emit.hip import emit; print(emit(k(), GFX1200), \
    #   end='')" > test/golden/hip/inline_probe.hip
    assert hip.emit(synthetic_inline_kernel(), GFX1200) == \
        INLINE_GOLDEN.read_text()


def test_the_inline_texts_differ_only_in_the_prologue() -> None:
    # The splice is spelled once, in emit/_c.py: no vendor hook touches it.
    k = synthetic_inline_kernel()
    c = cuda.emit(k, SM_86).splitlines()
    h = hip.emit(k, GFX1200).splitlines()
    assert h[2] == "#include <hip/hip_runtime.h>"
    assert c[2:] == h[3:]


@pytest.mark.hip
def test_the_inline_golden_compiles(hipcc, compile_device_source) -> None:
    # Compile only: a device compiler needs no card.
    compile_device_source(hipcc, hip.emit(synthetic_inline_kernel(), GFX1200),
                          link=False)


@pytest.mark.hip
@pytest.mark.parametrize("L_", [0, 1, L, 8])
def test_it_compiles(hipcc, compile_device_source, L_: int) -> None:
    # A device compiler needs no card: this is green on a device-less box and
    # proves nothing about a number.
    compile_device_source(hipcc, hip.emit(build_pq_kernel(L_), GFX1200),
                          link=False)


# --- the pair ladder's kernels (#156) ----------------------------------------
# The same two classes as test_emit_cuda.py's pair goldens. Regenerate (never
# hand-edit), as there, with `emit.hip` and `GFX1200` > test/golden/hip/
# dq_1_1.hip and jab_11.hip.

PAIR_GOLDENS = {"dq_1_1": "dq", "jab_11": "jab"}


def _pair_kernel(family: str):
    from felizjueves.kernel_dq import build_dq_kernel
    from felizjueves.kernel_jab import build_jab_kernel
    return {"dq": build_dq_kernel, "jab": build_jab_kernel}[family]((1, 1))


@pytest.mark.parametrize("name", sorted(PAIR_GOLDENS))
def test_the_pair_goldens_are_unchanged(name: str) -> None:
    k = _pair_kernel(PAIR_GOLDENS[name])
    assert k.name == name
    assert hip.emit(k, GFX1200) == (GOLDEN.parent / f"{name}.hip").read_text()


@pytest.mark.parametrize("name", sorted(PAIR_GOLDENS))
def test_the_pair_texts_differ_only_in_the_prologue(name: str) -> None:
    # No warp reduction in either family, and JAB's AtomicAdd is spelled the
    # same by both vendors: only the prologue differs.
    k = _pair_kernel(PAIR_GOLDENS[name])
    c = cuda.emit(k, SM_86).splitlines()
    h = hip.emit(k, GFX1200).splitlines()
    assert h[0] == c[0].replace("emit.cuda", "emit.hip")
    assert h[1] == c[1].replace(SM_86.name, GFX1200.name)
    assert h[2] == "#include <hip/hip_runtime.h>"
    assert c[2:] == h[3:]


@pytest.mark.hip
@pytest.mark.parametrize("name", sorted(PAIR_GOLDENS))
def test_the_pair_goldens_compile(hipcc, compile_device_source, name: str) -> None:
    # Compile only: a device compiler needs no card.
    compile_device_source(hipcc, hip.emit(_pair_kernel(PAIR_GOLDENS[name]),
                                          GFX1200), link=False)
