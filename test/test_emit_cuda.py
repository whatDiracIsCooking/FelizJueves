"""emit/cuda.py: a `Kernel` spelled as CUDA C++, its golden file, and a compile.

THE GOLDEN FILE is `test/golden/cuda/pq_L4_p2_q2.cu`: `build_pq_kernel(4)`,
the class (2,0|2,0) in one batch, for `SM_86`. L = 4 is the smallest default
class with everything the writer spells -- a two-term reduction (r_d >= 2), a
ket side (so the (-1)^|q| contraction has both signs), the screening break,
the shared accumulator and the warp reduction -- while staying short enough
to review by eye. A diff in it is the whole signal when the generator
changes. After an intended change, regenerate it and review the diff:

    PYTHONPATH=src uv run python -c "from felizjueves.kernel import \
build_pq_kernel; from felizjueves.device import SM_86; from \
felizjueves.emit.cuda import emit; print(emit(build_pq_kernel(4), SM_86), \
end='')" > test/golden/cuda/pq_L4_p2_q2.cu

THE COMPILE carries the `cuda` marker and needs `nvcc` only: a device
compiler needs no card, so it is green on a device-less box and proves
nothing about a number. Running the kernel on the card is the kernel tier's
(#28).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from felizjueves.device import GFX1200, SM_86, Arch, Vendor
from felizjueves.emit._c import EmitError
from felizjueves.emit.cuda import emit
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Fn,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Space,
    Var,
)
from felizjueves.kernel import build_pq_kernel

L = 4
GOLDEN = Path(__file__).resolve().parent / "golden" / "cuda" / "pq_L4_p2_q2.cu"


def test_the_golden_kernel_is_unchanged() -> None:
    assert emit(build_pq_kernel(L), SM_86) == GOLDEN.read_text()


def test_emit_is_pure_and_deterministic() -> None:
    k = build_pq_kernel(L)
    a = emit(k, SM_86)
    assert emit(k, SM_86) == a
    assert k == build_pq_kernel(L)  # nothing mutated the kernel


def test_notes_render_as_comments_beside_their_lines() -> None:
    text = emit(build_pq_kernel(L), SM_86)
    assert "// [JEngine, v2, eqs. 10-12], reduction form\n" in text
    note = text.index("// [JEngine, v2, eqs. 10-12]")
    assert text.index("double R0_0_0_", note) > note  # the recurrence follows it
    assert "// PRECONDITION: kets arrive sorted" in text


def test_the_wavefront_is_read_off_the_arch() -> None:
    # A hypothetical narrower NVIDIA arch changes the shuffle and nothing else.
    narrow = Arch(Vendor.NVIDIA, "sm_test", wavefront=16)
    k = build_pq_kernel(L)
    wide, thin = emit(k, SM_86).splitlines(), emit(k, narrow).splitlines()
    assert len(wide) == len(thin)
    changed = {(a.strip(), b.strip()) for a, b in zip(wide, thin, strict=True)
               if a != b}
    assert changed == {
        ("// arch sm_86, wavefront 32; fp64 throughout.",
         "// arch sm_test, wavefront 16; fp64 throughout."),
        ("for (int fj_o = 16; fj_o > 0; fj_o /= 2) {",
         "for (int fj_o = 8; fj_o > 0; fj_o /= 2) {"),
        ("fj_v += __shfl_xor_sync(0xffffffffu, fj_v, fj_o, 32);",
         "fj_v += __shfl_xor_sync(0xffffu, fj_v, fj_o, 16);"),
    }


def test_emit_takes_an_nvidia_arch_value() -> None:
    k = build_pq_kernel(0)
    with pytest.raises(TypeError, match="device.Arch"):
        emit(k, "sm_86")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NVIDIA"):
        emit(k, GFX1200)


def _k(*body, params=()) -> Kernel:
    return Kernel("probe", params, body)


def test_a_global_array_is_linearised_by_its_declared_shape() -> None:
    out = Param("out", Scalar.F64, array=True, intent=Intent.OUT,
                shape=(Var("n"), Lit(3)))
    n = Param("n", Scalar.I32)
    text = emit(_k(Assign(Index("out", (Lit(1), Lit(2))), Lit(0.5)),
                   params=(n, out)), SM_86)
    assert "out[(1 * 3 + 2)] = 0.5;" in text
    # Without a shape the writer would have to guess the stride: it refuses.
    bare = Param("out", Scalar.F64, array=True, intent=Intent.OUT)
    with pytest.raises(EmitError, match="linearised by its shape"):
        emit(_k(Assign(Index("out", (Lit(1), Lit(2))), Lit(0.5)),
                params=(bare,)), SM_86)


def test_the_writer_refuses_to_decide_a_shared_layout() -> None:
    dyn = (Lit(2), Call(Fn.BLOCK_DIM_Y))
    one = Decl("a", Scalar.F64, dyn, Space.SHARED)
    two = Decl("b", Scalar.F64, dyn, Space.SHARED)
    assert "extern __shared__ double a[];" in emit(_k(one), SM_86)
    with pytest.raises(EmitError, match="decision"):
        emit(_k(one, two), SM_86)
    with pytest.raises(EmitError, match="launch-time extent"):
        emit(_k(Decl("c", Scalar.F64, dyn)), SM_86)
    static = Decl("d", Scalar.F64, (Lit(2), Lit(3)), Space.SHARED)
    assert "__shared__ double d[2][3];" in emit(_k(static), SM_86)


def test_the_writers_prefix_is_reserved() -> None:
    with pytest.raises(EmitError, match="fj_"):
        emit(_k(Decl("fj_v", Scalar.F64, init=Lit(1.0))), SM_86)


def test_literals_round_trip_and_keep_their_type() -> None:
    text = emit(_k(Decl("x", Scalar.F64, init=Lit(-2.0)),
                   Decl("i", Scalar.I32, init=Lit(7)),
                   Note("a note")), SM_86)
    assert "double x = (-2.0);" in text and "int i = 7;" in text
    assert "// a note" in text


def test_a_param_shape_uses_only_launch_time_values() -> None:
    with pytest.raises(ValueError, match="extent"):
        Kernel("probe", (Param("a", Scalar.F64, array=True,
                               shape=(Call(Fn.THREAD_X), Lit(2))),), ())
    with pytest.raises(ValueError, match="no shape"):
        Param("a", Scalar.F64, shape=(Lit(2),))


@pytest.mark.cuda
@pytest.mark.parametrize("L_", [0, 1, L, 8])
def test_it_compiles(nvcc, compile_device_source, L_: int) -> None:
    # A device compiler needs no card: this is green on a device-less box and
    # proves nothing about a number.
    compile_device_source(nvcc, emit(build_pq_kernel(L_), SM_86), link=False)
