"""emit/gpu.py: one text for both device compilers, its goldens, and compiles.

THE SEAM, a third time. The `gpu` backend consumes the identical `Kernel` the
vendor backends do, and its text differs from CUDA's only in the hooks: the
prologue (FelizJueves's own header instead of nothing, and the width assert
when the text has a width) and the shuffle mask, cast to the header's
`FJ_LANE_MASK_T` instead of spelled per vendor -- and in two spellings: every
integer is `int32_t`, never a bare `int` (`CWriter.INT`), and every cast is a
`static_cast`, a sum of launch geometry taking one over its unsigned
arithmetic (`CWriter.cast`; the vendor backends keep C casts, as they keep
`int`, so `kernels.json`'s pins do not move).
`test_it_differs_from_cuda_only_in_the_hooks` pins exactly which lines differ,
once its casts and `int32_t` are read back in CUDA's spelling (`as_cuda`).

THE WIDTH. A unit with a shuffle carries `static_assert(FJ_WAVEFRONT == w)`;
a unit with none has no width in its text at all, so it is byte-identical
for any wavefront and carries no assert.

THE GOLDEN FILES are `test/golden/gpu/pq_L4_p2_q2.cu` (a warp reduction: the
width assert) and `s_11.cu` (no shuffle: none), and `s_11.specialized.cu`
and `s.h`, the same kernel as `gpu.Specialization("s", (1, 1))` and the
header that unit includes, over the classes (0, 0), (0, 1) and (1, 1)
(`emit(k, SM_86, spec)` and `emit_template(k, SM_86, spec, classes)`), and
`pq_22.specialized.cu` and `pq.h`, `build_pq_kernel(4)` as
`gpu.Specialization("pq", (2, 2))` under a fixed block, `block_dim = (1, 32)`,
over every `(lp, lq)` with both at most 2; and `pq_22.any_wavefront.cu` and
`pq.any_wavefront.h`, the same pair emitted `any_wavefront=True` with
`block_dim = (1, gpu.WAVEFRONT)`: one text for both widths; and
`pair_bound_11.cu`, the record bound (no shuffle, no width), and
`record_bound_10.cu`, K's dressed record bound of the ordered class (p, s)
(#499; no shuffle, no width); and
`eval_boys.h`, the device Boys routine every unit with a fundamental includes,
and the `eval_boys_{M}.h` it includes, one series per order (`gpu.emit_boys()`).
Regenerate after an intended
change (never hand-edit) and review the diff:

    PYTHONPATH=src uv run python -c "from pathlib import Path; \
from felizjueves.kernel import build_pq_kernel; \
from felizjueves.kernel_s import build_s_kernel; \
from felizjueves.device import SM_86; from felizjueves.emit.gpu import emit; \
[Path(f'test/golden/gpu/{k.name}.cu').write_text(emit(k, SM_86)) \
for k in (build_pq_kernel(4), build_s_kernel((1, 1)))]"

THE COMPILES are the point: the SAME string goes to nvcc (`cuda` marker) and
to hipcc (`hip` marker), with `include/` as the only search path, and a
wave64 build of a wave32 unit is refused at compile time. A device compiler
needs no card, so these prove nothing about a number.
"""

from __future__ import annotations

import dataclasses
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from conftest import INCLUDE
from test_ir import synthetic_inline_kernel
from test_k_lever_shuffle import _bcast

from felizjueves.device import GFX1200, SM_86, Arch, Vendor
from felizjueves.emit import cuda, gpu
from felizjueves.emit._c import BOYS_DEVICE, BOYS_SWITCH_PAD, EmitError
from felizjueves.ir.passes import split_axes, transpose
from felizjueves.kernel import build_pq_kernel
from felizjueves.kernel_dq import build_dq_kernel
from felizjueves.kernel_f import build_f_kernel, f_class_spec
from felizjueves.kernel_pair_bound import build_pair_bound_kernel
from felizjueves.kernel_record_bound import build_record_bound_kernel
from felizjueves.kernel_s import build_s_kernel
from felizjueves.kernel_t import build_t_kernel
from felizjueves.kernel_v import build_v_kernel
from felizjueves.spec import Operator, Spec, Target

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "test" / "golden" / "gpu"
WAVE64 = Arch(Vendor.AMD, "gfx_test", wavefront=64)
CUDA_MASK = "0xffffffffu"
GPU_MASK = "static_cast<FJ_LANE_MASK_T>(0xffffffffull)"
BARE_INT = re.compile(r"\bint\b")
C_CAST = re.compile(r"\((?:int|int32_t|double|float|FJ_LANE_MASK_T)\)")
GEOMETRY = r"((?:threadIdx|blockIdx|blockDim)\.[xy])"


BOYS_INCLUDE = f'#include "{gpu.BOYS_HEADER}"'


def _boys(where: Path) -> None:
    """`eval_boys.h` and its per-order headers beside the units, as a
    consumer writes them."""
    for name, text in gpu.emit_boys().items():
        (where / name).write_text(text)


def cuda_lines(k) -> list[str]:
    """CUDA's text of `k` without the Boys routine it carries inline, and its
    calls by the name the gpu backend includes it under."""
    text = cuda.emit(k, SM_86).replace("\n" + BOYS_DEVICE, "", 1)
    return text.replace("fj_boys<", "eval_boys<").splitlines()


def gpu_lines(text: str) -> list[str]:
    """A gpu text without the include that stands for that routine."""
    return [x for x in text.splitlines() if x != BOYS_INCLUDE]


def as_cuda(line: str) -> str:
    """A gpu line read back in CUDA's spelling: its `static_cast`s as C casts,
    a sum of launch geometry under one cast as one cast per term (the IR's
    tree, as CUDA spells it), and its fixed-width integers as `int`."""
    line = re.sub(rf"static_cast<int32_t>\({GEOMETRY} \+ {GEOMETRY} \* {GEOMETRY}\)",
                  r"((int)\1 + ((int)\2 * (int)\3))", line)
    line = re.sub(rf"static_cast<int32_t>\({GEOMETRY}\)", r"(int)\1", line)
    line = re.sub(r"static_cast<double>\(([^()]*)\)", r"(double)(\1)", line)
    return re.sub(r"\bint32_t\b", "int", line)


@pytest.mark.parametrize("kernel", [build_pq_kernel(4), build_s_kernel((1, 1)),
                                    build_pair_bound_kernel((1, 1)),
                                    build_record_bound_kernel((1, 0))],
                         ids=lambda k: k.name)
def test_the_goldens_are_unchanged(kernel) -> None:
    assert gpu.emit(kernel, SM_86) == (GOLDEN / f"{kernel.name}.cu").read_text()


def test_the_boys_goldens_are_unchanged() -> None:
    headers = gpu.emit_boys()
    assert sorted(headers) == sorted(
        [gpu.BOYS_HEADER, *(f"eval_boys_{m}.h" for m in range(gpu.BOYS_MAX + 1))])
    for name, text in headers.items():
        assert text == (GOLDEN / name).read_text(), name


def test_the_header_it_includes_is_ours() -> None:
    # FelizJueves's include/, not a vendor's and not another library's.
    assert (REPO / "include" / gpu.HEADER).is_file()
    assert (REPO / "include" / "felizjueves" / "backend.h").is_file()
    for k in (build_pq_kernel(4), build_s_kernel((1, 1))):
        assert gpu.emit(k, SM_86).splitlines()[2] == f'#include "{gpu.HEADER}"'


def test_it_differs_from_cuda_only_in_the_hooks() -> None:
    k = build_pq_kernel(4)
    c = cuda_lines(k)
    g = gpu_lines(gpu.emit(k, SM_86))
    assert g[:4] == [
        "// pq_L4_p2_q2: generated by felizjueves.emit.gpu -- do not edit.",
        "// wavefront 32; fp64 throughout; nvcc and hipcc alike.",
        '#include "felizjueves/runtime.h"',
        'static_assert(FJ_WAVEFRONT == 32, '
        '"pq_L4_p2_q2 was emitted for a 32-lane wavefront");',
    ]
    assert c[1] == "// arch sm_86, wavefront 32; fp64 throughout."
    body_c, body_g = c[2:], [as_cuda(x) for x in g[4:]]
    assert len(body_c) == len(body_g)
    changed = {(a.strip(), b.strip())
               for a, b in zip(body_c, body_g, strict=True) if a != b}
    assert changed == {(
        f"fj_v += __shfl_xor_sync({CUDA_MASK}, fj_v, fj_o, 32);",
        f"fj_v += __shfl_xor_sync({GPU_MASK}, fj_v, fj_o, 32);",
    )}


def test_a_unit_with_no_shuffle_has_no_width() -> None:
    for k in (build_s_kernel((1, 1)), build_v_kernel((1, 1)),
              synthetic_inline_kernel()):
        text = gpu.emit(k, SM_86)
        assert "static_assert(FJ_WAVEFRONT" not in text
        assert text.splitlines()[1].startswith("// any wavefront;")
        assert gpu.emit(k, WAVE64) == text
        # and only the prologue separates it from CUDA's text
        assert cuda_lines(k)[2:] == [as_cuda(x) for x in gpu_lines(text)[3:]]


def test_the_width_is_read_off_the_arch_and_asserted() -> None:
    k = build_pq_kernel(4)
    a, b = gpu.emit(k, SM_86).splitlines(), gpu.emit(k, WAVE64).splitlines()
    assert len(a) == len(b)
    changed = {(x.strip(), y.strip()) for x, y in zip(a, b, strict=True)
               if x != y}
    assert changed == {
        ("// wavefront 32; fp64 throughout; nvcc and hipcc alike.",
         "// wavefront 64; fp64 throughout; nvcc and hipcc alike."),
        ('static_assert(FJ_WAVEFRONT == 32, '
         '"pq_L4_p2_q2 was emitted for a 32-lane wavefront");',
         'static_assert(FJ_WAVEFRONT == 64, '
         '"pq_L4_p2_q2 was emitted for a 64-lane wavefront");'),
        ("for (int32_t fj_o = 16; fj_o > 0; fj_o /= 2) {",
         "for (int32_t fj_o = 32; fj_o > 0; fj_o /= 2) {"),
        (f"fj_v += __shfl_xor_sync({GPU_MASK}, fj_v, fj_o, 32);",
         "fj_v += __shfl_xor_sync(static_cast<FJ_LANE_MASK_T>(0xffffffffffffffffull), "
         "fj_v, fj_o, 64);"),
    }


def test_the_lane_broadcast_takes_the_mask_and_the_assert() -> None:
    lines = gpu.emit(_bcast(), SM_86).splitlines()
    assert (f"    got = __shfl_sync({GPU_MASK}, "
            "v[static_cast<int32_t>(threadIdx.y)], 0, 32);") in lines
    assert lines[3].startswith("static_assert(FJ_WAVEFRONT == 32,")


@pytest.mark.parametrize("unit", ["pq_L4", "s_11", "v_11", "bcast", "inline",
                                  "eval_boys"])
def test_no_integer_is_a_bare_int(unit: str) -> None:
    # Declarations, loop counters, the geometry casts and eval_boys alike;
    # comments included, since none of them says `int`.
    text = _text(unit)
    assert "int32_t" in text
    assert [x for x in text.splitlines() if BARE_INT.search(x)] == []


@pytest.mark.parametrize("unit", ["pq_L4", "s_11", "v_11", "bcast", "inline",
                                  "eval_boys"])
def test_every_cast_is_a_static_cast(unit: str) -> None:
    # The geometry, Fn.TO_F32 / TO_F64, the shuffle mask and eval_boys's own
    # text alike. `(void)n;` discards a value; it converts nothing.
    text = _text(unit)
    assert [x for x in text.splitlines() if C_CAST.search(x)] == []


def test_a_sum_of_launch_geometry_takes_one_cast() -> None:
    # Unsigned arithmetic throughout, then one conversion: the hardware's own
    # values, and no mixing of signed and unsigned under the cast.
    lines = gpu.emit(build_s_kernel((1, 1)), SM_86).splitlines()
    assert ("    int32_t rec = static_cast<int32_t>("
            "threadIdx.x + blockIdx.x * blockDim.x);") in lines
    # a sum that also reads a signed value casts each geometry term instead
    assert "Jsh[(0 * static_cast<int32_t>(blockDim.y) + " \
        "static_cast<int32_t>(threadIdx.y))] = 0.0;" in \
        [x.strip() for x in gpu.emit(build_pq_kernel(4), SM_86).splitlines()]


def test_the_vendor_backends_still_spell_int() -> None:
    # kernels.json's pins hash the cuda / hip text, which must not move.
    assert "int32_t" not in cuda.emit(build_pq_kernel(4), SM_86)


def test_any_vendors_arch_of_one_width_writes_one_text() -> None:
    # The text names no arch: sm_86 and gfx1200 are both wave32.
    for k in (build_pq_kernel(4), build_s_kernel((1, 1)), _bcast()):
        assert gpu.emit(k, SM_86) == gpu.emit(k, GFX1200)


def test_emit_takes_an_arch_value() -> None:
    with pytest.raises(TypeError, match="device.Arch"):
        gpu.emit(build_pq_kernel(0), "sm_86")  # type: ignore[arg-type]


def test_a_writer_reused_does_not_carry_the_width_over() -> None:
    w = gpu.GpuWriter(SM_86)
    assert "static_assert" in w.kernel(build_pq_kernel(4))
    assert "static_assert" not in w.kernel(build_s_kernel((1, 1)))


# --- one text, both compilers -------------------------------------------------

TOOLCHAINS = [pytest.param("nvcc", marks=pytest.mark.cuda),
              pytest.param("hipcc", marks=pytest.mark.hip)]
UNITS = {
    "pq_L0": lambda: build_pq_kernel(0),
    "pq_L4": lambda: build_pq_kernel(4),
    "pq_L4_split": lambda: split_axes(build_pq_kernel(4), {
        "P": ("p_x", "p_y", "p_z"), "Q": ("q_x", "q_y", "q_z")}),
    "pair_bound_22": lambda: build_pair_bound_kernel((2, 2)),
    # K's dressed record bound, as FreeBreakfast stages an ordered record
    # (pa_* / pb_* per axis, #499)
    "record_bound_21_split": lambda: split_axes(
        build_record_bound_kernel((2, 1)), {
            "XPA": ("pa_x", "pa_y", "pa_z"), "XPB": ("pb_x", "pb_y", "pb_z")}),
    "dq_12_ao": lambda: split_axes(build_dq_kernel((1, 2), density="ao"), {
        "XPA": ("pa_x", "pa_y", "pa_z"), "XPB": ("pb_x", "pb_y", "pb_z")}),
    "dq_11_split": lambda: split_axes(build_dq_kernel((1, 1)), {
        "XPA": ("pa_x", "pa_y", "pa_z"), "XPB": ("pb_x", "pb_y", "pb_z")}),
    # D q-major, as FreeBreakfast writes it (dq) and reads it ([p|q])
    "pq_L4_transposed": lambda: transpose(split_axes(build_pq_kernel(4), {
        "P": ("p_x", "p_y", "p_z"), "Q": ("q_x", "q_y", "q_z")}), ("D",)),
    "dq_12_ao_transposed": lambda: transpose(split_axes(
        build_dq_kernel((1, 2), density="ao"), {
            "XPA": ("pa_x", "pa_y", "pa_z"),
            "XPB": ("pb_x", "pb_y", "pb_z")}), ("D",)),
    "s_11": lambda: build_s_kernel((1, 1)),
    "v_11": lambda: build_v_kernel((1, 1)),
    "bcast": _bcast,
    "inline": synthetic_inline_kernel,
}


def test_a_transposed_d_is_read_with_the_ket_fastest() -> None:
    # the point of passes.transpose: consecutive lanes (kets k) read
    # consecutive doubles, D[q * n_ket + k], where the default is n_q apart
    row = gpu.emit(build_pq_kernel(2), SM_86)
    col = gpu.emit(UNITS["pq_L4_transposed"](), SM_86)
    assert "D[(k * 4 + 1)]" in row
    assert "D[(1 * n_ket + k)]" in col and "D[(k * " not in col
    dq = gpu.emit(UNITS["dq_12_ao_transposed"](), SM_86)
    assert "D[(1 * n_rec + k)] = dq1;" in dq


def _text(unit: str) -> str:
    """A unit's gpu text, or every Boys header's."""
    if unit == "eval_boys":
        return "".join(gpu.emit_boys().values())
    return gpu.emit(UNITS[unit](), SM_86)


@pytest.mark.parametrize("tool", TOOLCHAINS)
@pytest.mark.parametrize("unit", sorted(UNITS))
def test_one_text_compiles_under_both(request, tmp_path, compile_device_source,
                                      tool: str, unit: str) -> None:
    tc = request.getfixturevalue(tool)
    _boys(tmp_path)
    compile_device_source(tc, gpu.emit(UNITS[unit](), SM_86), link=False)


@pytest.mark.hip
def test_a_wave32_unit_is_refused_on_a_wave64_build(hipcc, tmp_path,
                                                    compile_device_source) -> None:
    # The define stands in for the build's -DFJ_WAVEFRONT=64.
    _boys(tmp_path)
    text = "#define FJ_WAVEFRONT 64\n" + gpu.emit(build_pq_kernel(0), SM_86)
    with pytest.raises(AssertionError, match="emitted for a 32-lane wavefront"):
        compile_device_source(hipcc, text, link=False)


@pytest.mark.hip
def test_a_unit_with_no_width_builds_for_wave64(hipcc,
                                                compile_device_source) -> None:
    text = "#define FJ_WAVEFRONT 64\n" + gpu.emit(build_s_kernel((1, 1)), SM_86)
    compile_device_source(hipcc, text, link=False)


@pytest.mark.cuda
def test_nvcc_refuses_a_wave64_build(nvcc, compile_device_source) -> None:
    text = "#define FJ_WAVEFRONT 64\n" + gpu.emit(build_s_kernel((1, 1)), SM_86)
    with pytest.raises(AssertionError, match="32 on every NVIDIA arch"):
        compile_device_source(nvcc, text, link=False)


# --- the template spelling (gpu.Specialization) ------------------------------

S11 = gpu.Specialization("s", (1, 1))
S_CLASSES = [(0, 0), (0, 1), (1, 1)]


def _params(k) -> str:
    """The C spelling's parameter lines, through the closing `)`."""
    c = gpu.emit(k, SM_86).splitlines()
    start = c.index(f'extern "C" __global__ void {k.name}(') + 1
    return "\n".join(c[start:c.index("{")])


def test_the_specialized_goldens_are_unchanged() -> None:
    k = build_s_kernel((1, 1))
    assert gpu.emit(k, SM_86, S11) == (GOLDEN / "s_11.specialized.cu").read_text()
    assert gpu.emit_template(k, SM_86, S11, S_CLASSES) == (GOLDEN / "s.h").read_text()


def test_a_specialization_moves_only_the_declarator() -> None:
    k = build_s_kernel((1, 1))
    c = gpu.emit(k, SM_86).splitlines()
    t = gpu.emit(k, SM_86, S11).splitlines()
    assert t[1] == c[1].removesuffix(".") + "; specialization s<1, 1>."
    assert t[3] == '#include "s.h"  // the primary'
    i_c = c.index('extern "C" __global__ void s_11(')
    i_t = t.index("template <>")
    assert c[2:i_c] == [t[2], *t[4:i_t]]
    assert t[i_t + 1] == "__global__ void s<1, 1>("
    # the parameters and the body, character for character
    assert c[i_c + 1:] == t[i_t + 2:]


def test_the_header_declares_the_primary_and_every_class() -> None:
    k = build_s_kernel((1, 1))
    params = _params(k)
    text = gpu.emit_template(k, SM_86, S11, S_CLASSES)
    blocks = text.split("\n\n")
    assert S11.header == "s.h"
    assert blocks[0].endswith('#pragma once\n#include "felizjueves/runtime.h"')
    # uint8_t, lettered by position; fixed arity, never a pack: nvcc 13.0
    # miscompiles the host stub of a variadic __global__ template's
    # explicit specialization
    assert blocks[1] == ("template <uint8_t La, uint8_t Lb>\n"
                         f"__global__ void s(\n{params};")
    # each class declared before the table names it, in emission order
    assert blocks[2:5] == [f"template <>\n__global__ void s<{a}, {b}>(\n{params};"
                           for a, b in S_CLASSES]
    assert blocks[5] == (
        "// indexed by the class's args; nullptr where no class was emitted.\n"
        "inline constexpr decltype(&s<0, 0>) s_kernels[2][2] = {\n"
        "    {&s<0, 0>, &s<0, 1>},\n"
        "    {nullptr, &s<1, 1>},\n"
        "};\n"
    )
    # any unit of the template writes the same header
    assert gpu.emit_template(build_s_kernel((0, 1)), SM_86,
                             gpu.Specialization("s", (0, 1)), S_CLASSES) == text
    with pytest.raises(TypeError, match="gpu.Specialization"):
        gpu.emit_template(k, SM_86, None, S_CLASSES)  # type: ignore[arg-type]


def test_each_extent_is_one_past_the_largest_arg_at_its_position() -> None:
    text = gpu.emit_template(build_s_kernel((1, 1)), SM_86, S11, [(0, 2), (1, 1)])
    assert text.endswith(
        "inline constexpr decltype(&s<0, 2>) s_kernels[2][3] = {\n"
        "    {nullptr, nullptr, &s<0, 2>},\n"
        "    {nullptr, &s<1, 1>, nullptr},\n"
        "};\n"
    )


def test_the_arity_follows_the_args() -> None:
    k = build_f_kernel(f_class_spec(0, 0, 0, 0))
    spec = gpu.Specialization("f", (0, 0, 0, 0))
    h = gpu.emit_template(k, SM_86, spec, [(0, 0, 0, 0)])
    assert "template <uint8_t La, uint8_t Lb, uint8_t Lc, uint8_t Ld>" in h
    assert h.endswith(
        "f_kernels[1][1][1][1] = {\n"
        "    {\n"
        "        {\n"
        "            {&f<0, 0, 0, 0>},\n"
        "        },\n"
        "    },\n"
        "};\n"
    )
    assert "__global__ void f<0, 0, 0, 0>(" in gpu.emit(k, SM_86, spec)


@pytest.mark.parametrize(("classes", "match"), [
    ([], "no classes"),
    ([(0, 0), (0,)], "other than 2 args"),
    ([(0, 1), (0, 1)], "listed twice"),
    ([(0, 1.0)], "nonempty tuple of ints"),
    ([(0, 256)], "uint8_t"),
])
def test_the_header_is_refused_when_its_classes_are_not_a_list(
        classes, match) -> None:
    with pytest.raises(ValueError, match=match):
        gpu.emit_template(build_s_kernel((1, 1)), SM_86, S11, classes)


def test_a_unit_with_a_fundamental_includes_eval_boys() -> None:
    v = build_v_kernel((1, 1))
    f = build_f_kernel(f_class_spec(0, 0, 0, 0))
    for text in (gpu.emit(v, SM_86),
                 gpu.emit(v, SM_86, gpu.Specialization("v", (1, 1))),
                 gpu.emit(f, SM_86, gpu.Specialization("f", (0, 0, 0, 0)))):
        assert text.count(BOYS_INCLUDE) == 1
        assert "eval_boys<" in text
        # the unit carries no routine of its own, guarded or not
        assert "fj_boys" not in text and "FJ_BOYS" not in text
        assert "__device__ __forceinline__" not in text
    # the include follows the runtime's, and the primary's when there is one
    assert gpu.emit(v, SM_86).splitlines()[3] == BOYS_INCLUDE
    assert gpu.emit(v, SM_86, gpu.Specialization("v", (1, 1))).splitlines()[
        3:5] == ['#include "v.h"  // the primary', BOYS_INCLUDE]
    assert BOYS_INCLUDE not in gpu.emit(build_s_kernel((1, 1)), SM_86)


def test_eval_boys_is_the_vendor_routine_with_the_series_replaced() -> None:
    # The erf branch and both recurrences are fj_boys's (#301), in this
    # backend's spelling, with internal linkage, so every unit that includes
    # it carries its own copy and none is a duplicate device symbol; only the
    # series loop is a call, to the polynomial of its order.
    h = gpu.emit_boys()[gpu.BOYS_HEADER]
    assert h.splitlines()[3:5] == ["#pragma once", f'#include "{gpu.HEADER}"']
    assert h.count("static __device__ __forceinline__ void eval_boys(") == 1
    assert "fj_boys" not in h
    orders = [x for x in h.splitlines() if x.startswith("#include \"eval_boys_")]
    assert orders == [f'#include "eval_boys_{m}.h"'
                      for m in range(gpu.BOYS_MAX + 1)]
    ours = [as_cuda(x) for x in h.split(
        "static __device__ __forceinline__ void eval_boys(", 1)[1].splitlines()]
    vendor = BOYS_DEVICE.split(
        "__device__ __forceinline__ void fj_boys(", 1)[1].splitlines()
    loop = vendor.index("    double term = 1.0 / (double)(2 * M + 1);")
    end = vendor.index("    double f = e * sum;")
    assert ours[0] == vendor[0]
    assert ours[1:3] == ["    static_assert(M >= 0 && M <= 20,",
                         '                  "eval_boys_M.h is written for M in '
                         '0..20");']
    assert ours[3:] == [*vendor[1:loop],
                        "    double f = e * eval_boys_series<M>(2.0 * T);",
                        *vendor[end + 1:]]


def _tail_degree(m: int) -> int:
    """The least n whose tail is under 2^-53 of the Kummer sum at the switch,
    by 400 exact terms (far past where any term matters)."""
    x = Fraction(2 * (m + BOYS_SWITCH_PAD))
    terms = [Fraction(1, 2 * m + 1)]
    for k in range(1, 400):
        terms.append(terms[-1] * x / (2 * m + 2 * k + 1))
    whole = sum(terms)
    tail = whole
    for n, t in enumerate(terms):
        tail -= t
        if tail * 2**53 < whole:
            return n
    raise AssertionError(m)


@pytest.mark.parametrize("m", range(gpu.BOYS_MAX + 1))
def test_each_series_is_the_exact_coefficients_rounded_once(m: int) -> None:
    c = gpu.boys_series(m)
    assert len(c) - 1 == _tail_degree(m)
    want = [1.0 / (2 * m + 1)]
    exact = Fraction(1, 2 * m + 1)
    for k in range(1, len(c)):
        exact /= 2 * m + 2 * k + 1
        want.append(float(exact))  # the correctly rounded quotient
    assert list(c) == want
    # the header spells exactly these, highest first, one fma a step
    lines = gpu.emit_boys()[f"eval_boys_{m}.h"].splitlines()
    first = lines.index("    double s = " + repr(c[-1]) + ";")
    steps = lines[first + 1:lines.index("    return s;")]
    assert steps == [f"    s = fma(s, x, {ck!r});" for ck in reversed(c[:-1])]
    assert lines[first - 1] == ("__device__ __forceinline__ double "
                                f"eval_boys_series<{m}>(double x) {{")


def test_the_degrees_grow_with_the_switch() -> None:
    # 40 at M = 0 to 61 at M = 20: the switch T = M + 8 moves with M
    degrees = [len(gpu.boys_series(m)) - 1 for m in range(gpu.BOYS_MAX + 1)]
    assert degrees[0] == 40 and degrees[-1] == 61
    assert degrees == sorted(degrees)


@pytest.mark.parametrize("m", [-1, gpu.BOYS_MAX + 1, 2.0])
def test_a_series_past_the_written_orders_is_refused(m) -> None:
    with pytest.raises(ValueError, match="0 <= M <= 20"):
        gpu.boys_series(m)


def test_the_series_reproduces_the_kummer_sum_below_the_switch() -> None:
    # Horner over the rounded coefficients, in float64 without fma, against
    # the exact series: within (2n + 2) u at every sampled x below the switch.
    for m in range(gpu.BOYS_MAX + 1):
        c = gpu.boys_series(m)
        for t in (0.0, 0.25, 1.0, 3.0, m + 2.0, m + BOYS_SWITCH_PAD - 1e-9):
            x = 2.0 * t
            s = c[-1]
            for ck in reversed(c[:-1]):
                s = s * x + ck
            xe, exact, term, k = Fraction(x), Fraction(0), Fraction(1, 2 * m + 1), 0
            while term * 2**80 > exact or k < 2:
                exact += term
                k += 1
                term = term * xe / (2 * m + 2 * k + 1)
            rel = abs(Fraction(s) - exact) / exact
            assert rel <= (2 * len(c) + 2) * Fraction(1, 2**53), (m, t, float(rel))
            assert math.isfinite(s)


@pytest.mark.parametrize("name", ["eval_boys", "eval_boys_series"])
def test_a_kernel_may_not_take_eval_boyss_names(name: str) -> None:
    k = dataclasses.replace(build_s_kernel((1, 1)), name=name)
    with pytest.raises(EmitError, match="eval_boys.h"):
        gpu.emit(k, SM_86)
    assert name in cuda.emit(k, SM_86)  # the vendors carry fj_boys


def test_a_specialization_refuses_what_its_siblings_would_repeat() -> None:
    # pq's dynamic shared array, and struct-reads' record struct (#353)
    with pytest.raises(EmitError, match="dynamic shared array"):
        gpu.emit(build_pq_kernel(0), SM_86, gpu.Specialization("pq", (0,)))
    rec = build_f_kernel(f_class_spec(0, 0, 0, 0, ("struct-reads",)))
    with pytest.raises(EmitError, match="record structs"):
        gpu.emit(rec, SM_86, gpu.Specialization("f", (0, 0, 0, 0)))
    gpu.emit(rec, SM_86)  # the C spelling still takes it


# --- a fixed block (block_dim): pq's shared array sized at emit time ---------

PQ22 = gpu.Specialization("pq", (2, 2))
PQ_CLASSES = [(lp, lq) for lp in range(3) for lq in range(3)]
PQ_BLOCK = (1, SM_86.wavefront)  # kernel.launch_shape's block


def _pq(lp: int, lq: int):
    """The [p|q] kernel of Hermite class (lp, lq), one batch."""
    am = ((lp + 1) // 2, lp // 2, (lq + 1) // 2, lq // 2)
    return build_pq_kernel(Spec(Operator.COULOMB, 0, am, Target.J))


def test_the_fixed_block_goldens_are_unchanged() -> None:
    k = build_pq_kernel(4)
    assert gpu.emit(k, SM_86, PQ22, block_dim=PQ_BLOCK) == (
        GOLDEN / "pq_22.specialized.cu").read_text()
    assert gpu.emit_template(k, SM_86, PQ22, PQ_CLASSES, block_dim=PQ_BLOCK) == (
        GOLDEN / "pq.h").read_text()


def test_a_fixed_block_moves_only_the_shared_declaration() -> None:
    k = build_pq_kernel(4)  # Jsh: 10 x blockDim.y
    dynamic = gpu.emit(k, SM_86).splitlines()
    fixed = gpu.emit(k, SM_86, block_dim=PQ_BLOCK).splitlines()
    i = dynamic.index("    extern __shared__ double Jsh[];")
    assert fixed[i] == "    __shared__ double Jsh[320];"
    assert "sized for blockDim = (1, 32)" in fixed[i - 1]
    # the body still indexes by blockDim.y, character for character
    assert dynamic[:i - 1] == fixed[:i - 1]
    assert dynamic[i + 1:] == fixed[i + 1:]


def test_a_fixed_block_makes_pq_a_specialization() -> None:
    k = build_pq_kernel(4)
    with pytest.raises(EmitError, match="dynamic shared array"):
        gpu.emit(k, SM_86, PQ22)
    text = gpu.emit(k, SM_86, PQ22, block_dim=PQ_BLOCK)
    assert "template <>\n__global__ void pq<2, 2>(" in text
    assert "extern __shared__" not in text
    # eval_boys included, as in every unit with a fundamental
    assert BOYS_INCLUDE in text and "static __device__" not in text


def test_the_header_names_the_block_the_units_were_sized_for() -> None:
    k = build_pq_kernel(4)
    fixed = gpu.emit_template(k, SM_86, PQ22, PQ_CLASSES, block_dim=PQ_BLOCK)
    assert "inline constexpr uint32_t pq_block_dim[2] = {1, 32};" in fixed
    assert "block_dim" not in gpu.emit_template(k, SM_86, PQ22, PQ_CLASSES)


def test_a_kernel_with_no_launch_time_extent_ignores_the_block() -> None:
    for k in (build_s_kernel((1, 1)), build_v_kernel((1, 1))):
        assert gpu.emit(k, SM_86, block_dim=PQ_BLOCK) == gpu.emit(k, SM_86)
    k = build_s_kernel((1, 1))
    assert gpu.emit(k, SM_86, S11, block_dim=PQ_BLOCK) == gpu.emit(k, SM_86, S11)


@pytest.mark.parametrize("block_dim", [(32,), (1, 32, 1), (0, 32), (1, 32.0),
                                       [1, 32], (1, True), (1, "FJ_WAVEFRONT")])
def test_a_block_is_refused_when_it_is_not_two_positive_ints(block_dim) -> None:
    with pytest.raises(ValueError, match="two positive ints"):
        gpu.emit(build_pq_kernel(4), SM_86, block_dim=block_dim)


@pytest.mark.parametrize(("template", "args", "match"), [
    ("s<1>", (1,), "not an identifier"),
    ("fj_s", (1,), "prefix"),
    ("s", (), "nonempty tuple of ints"),
    ("s", [1, 1], "nonempty tuple of ints"),
    ("s", (1, True), "nonempty tuple of ints"),
    ("s", (0, 256), "uint8_t"),
    ("s", (-1, 0), "uint8_t"),
])
def test_a_specialization_is_refused_when_it_cannot_be_spelled(
        template, args, match) -> None:
    with pytest.raises(ValueError, match=match):
        gpu.Specialization(template, args)


def test_emit_takes_a_specialization_value() -> None:
    with pytest.raises(TypeError, match="gpu.Specialization"):
        gpu.emit(build_s_kernel((1, 1)), SM_86, ("s", (1, 1)))  # type: ignore[arg-type]


# --- what a consumer builds: each unit its own TU, launched through the table --

LMAX = 2
PAIRS = [(la, lb) for la in range(LMAX + 1) for lb in range(la, LMAX + 1)]
FAMILIES = {"s": build_s_kernel, "v": build_v_kernel}  # v: six eval_boys

# Read at run time, as a launcher reads it to launch: that read is what makes
# each cell a symbol to link. Through a volatile, since a function's address
# is never null and -O2 would otherwise fold the comparison (hipcc does) and
# link nothing.
LAUNCHER = f"""\
#include "s.h"
#include "v.h"

int main(int argc, char**) {{
    const int i = argc % {LMAX + 1}, j = (argc / {LMAX + 1}) % {LMAX + 1};
    volatile auto s = s_kernels[i][j];
    volatile auto v = v_kernels[i][j];
    return s == nullptr || v == nullptr;
}}
"""


def _headers(where: Path, pairs=PAIRS) -> None:
    # `{template}.h` and `eval_boys.h` beside the units, as a consumer writes them
    _boys(where)
    for name, build in FAMILIES.items():
        spec = gpu.Specialization(name, pairs[0])
        (where / spec.header).write_text(
            gpu.emit_template(build(pairs[0]), SM_86, spec, pairs))


def _units(ext: str, pairs=PAIRS) -> dict[str, str]:
    return {f"{name}_{la}{lb}.{ext}":
            gpu.emit(build((la, lb)), SM_86, gpu.Specialization(name, (la, lb)))
            for name, build in FAMILIES.items() for la, lb in pairs}


def _link(tc, where: Path, units: dict[str, str], *, rdc: bool = True):
    """Every unit and the launcher as separate sources of one executable --
    the device-library build a consumer runs. nvcc needs -rdc for it."""
    for name, text in {**units, f"main.{tc.ext}": LAUNCHER}.items():
        (where / name).write_text(text)
    flags = [*tc.compile_flags(), f"-I{INCLUDE}", f"-I{where}"]
    if tc.name == "nvcc" and rdc:
        flags.append("-rdc=true")
    return subprocess.run(
        [tc.path, *flags, *units, f"main.{tc.ext}", "-o", "probe"],
        cwd=where, capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("tool", TOOLCHAINS)
def test_every_class_of_two_families_links_as_its_own_unit(
        request, tmp_path, tool: str) -> None:
    tc = request.getfixturevalue(tool)
    _headers(tmp_path)
    done = _link(tc, tmp_path, _units(tc.ext))
    assert done.returncode == 0, done.stderr
    assert "20280" not in done.stderr


@pytest.mark.cuda
def test_nvcc_warns_on_a_unit_of_its_own_without_relocatable_device_code(
        nvcc, tmp_path) -> None:
    # whole-program mode's static template stubs want every specialization
    # the launcher names defined in the launcher's own TU: nvcc 13.0 links,
    # but says so (#20280), and -Werror makes it an error
    _headers(tmp_path)
    done = _link(nvcc, tmp_path, _units(nvcc.ext), rdc=False)
    assert "must have a definition in the current translation unit" in done.stderr


@pytest.mark.parametrize("tool", TOOLCHAINS)
def test_a_class_not_compiled_in_fails_to_link(
        request, tmp_path, tool: str) -> None:
    # The header still declares (1, 2) and its table names it, but its units
    # are left out: the primary has no definition to fall back on.
    tc = request.getfixturevalue(tool)
    _headers(tmp_path)
    units = {n: t for n, t in _units(tc.ext).items() if "_12." not in n}
    done = _link(tc, tmp_path, units)
    assert done.returncode != 0
    assert re.search(r"[Uu]ndefined (reference|symbol)", done.stderr), done.stderr


@pytest.mark.parametrize("tool", TOOLCHAINS)
def test_every_class_of_two_families_still_links_in_one_tu(
        request, tmp_path, compile_device_source, tool: str) -> None:
    tc = request.getfixturevalue(tool)
    _headers(tmp_path)
    compile_device_source(tc, "".join(_units(tc.ext).values()) + LAUNCHER)


@pytest.mark.parametrize("tool", TOOLCHAINS)
def test_a_unit_whose_parameters_drifted_fails_to_compile(
        request, tmp_path, compile_device_source, tool: str) -> None:
    # The header declares S's parameters; a T unit spelled as s<1, 1> has
    # T's (eb_ket besides), so it specializes nothing.
    tc = request.getfixturevalue(tool)
    _headers(tmp_path)
    text = gpu.emit(build_t_kernel((1, 1)), SM_86, S11)
    with pytest.raises(AssertionError,
                       match="no function template matches|no instance of function "
                             "template|does not match|"
                             "not an entity that can be explicitly specialized"):
        compile_device_source(tc, text, link=False)


# --- any_wavefront: one text, the width chosen by -DFJ_WAVEFRONT -------------

ANY = (1, gpu.WAVEFRONT)


def test_the_any_wavefront_goldens_are_unchanged() -> None:
    k = build_pq_kernel(4)
    assert gpu.emit(k, SM_86, PQ22, block_dim=ANY, any_wavefront=True) == (
        GOLDEN / "pq_22.any_wavefront.cu").read_text()
    assert gpu.emit_template(k, SM_86, PQ22, PQ_CLASSES, block_dim=ANY,
                             any_wavefront=True) == (
        GOLDEN / "pq.any_wavefront.h").read_text()


def test_wavefront_in_a_block_is_the_archs_width_by_default() -> None:
    k = build_pq_kernel(4)
    assert gpu.emit(k, SM_86, PQ22, block_dim=ANY) == gpu.emit(
        k, SM_86, PQ22, block_dim=PQ_BLOCK)
    assert "Jsh[640]" in gpu.emit(k, WAVE64, block_dim=ANY)


def test_any_wavefront_is_one_text_for_every_arch() -> None:
    k = build_pq_kernel(4)
    texts = {gpu.emit(k, a, PQ22, block_dim=ANY, any_wavefront=True)
             for a in (SM_86, GFX1200, WAVE64)}
    assert len(texts) == 1
    (text,) = texts
    assert "static_assert" not in text and "0xffffffff" not in text
    assert ", 32)" not in text and "= 16;" not in text


def test_any_wavefront_moves_only_the_width() -> None:
    """Read back at width 32, the text is the fixed unit's, less its assert
    and with the banner naming the width as chosen at compile time."""
    k = build_pq_kernel(4)
    fixed = gpu.emit(k, SM_86, PQ22, block_dim=PQ_BLOCK).splitlines()
    anyw = gpu.emit(k, SM_86, PQ22, block_dim=ANY,
                    any_wavefront=True).splitlines()
    assert anyw[1] == fixed[1].replace(
        "wavefront 32", "wavefront FJ_WAVEFRONT, chosen at compile time")
    assert [ln for ln in fixed if ln.startswith("static_assert")] == [
        'static_assert(FJ_WAVEFRONT == 32, "pq_L4_p2_q2 was emitted for a '
        '32-lane wavefront");']
    fixed = [ln for ln in fixed[2:] if not ln.startswith("static_assert")]
    read = [ln.replace("FJ_WAVEFRONT / 2", "16")
              .replace("FJ_FULL_LANE_MASK", GPU_MASK)
              .replace("10 * FJ_WAVEFRONT", "320")
              .replace("FJ_WAVEFRONT", "32") for ln in anyw[2:]]
    assert read == fixed


@pytest.mark.parametrize("build", [lambda: build_s_kernel((1, 1)),
                                   lambda: build_v_kernel((1, 1)),
                                   lambda: build_dq_kernel((1, 1))],
                         ids=["s", "v", "dq"])
def test_a_unit_with_no_width_is_one_text_either_way(build) -> None:
    k = build()
    assert gpu.emit(k, SM_86, any_wavefront=True) == gpu.emit(k, SM_86)


def test_any_wavefront_is_a_bool() -> None:
    with pytest.raises(TypeError, match="any_wavefront is a bool"):
        gpu.emit(build_pq_kernel(4), SM_86, any_wavefront=1)  # type: ignore[arg-type]


WIDTHS = [pytest.param("nvcc", None, 32, marks=pytest.mark.cuda, id="nvcc-32"),
          pytest.param("hipcc", None, 32, marks=pytest.mark.hip, id="gfx1200-32"),
          pytest.param("hipcc", "gfx942", 64, marks=pytest.mark.hip,
                       id="gfx942-64")]


@pytest.mark.parametrize(("tool", "arch", "width"), WIDTHS)
def test_one_any_wavefront_text_compiles_at_either_width(
        request, tmp_path, compile_device_source, tool, arch, width) -> None:
    # The define stands in for the build's -DFJ_WAVEFRONT; gfx942 is CDNA 3,
    # wave64. Compile only: no wave64 card here.
    tc = request.getfixturevalue(tool)
    _boys(tmp_path)
    if arch is not None:
        tc = dataclasses.replace(tc, arch=arch)
    text = gpu.emit(build_pq_kernel(4), SM_86, any_wavefront=True)
    compile_device_source(tc, f"#define FJ_WAVEFRONT {width}\n" + text,
                          link=False)


PQ_LAUNCHER = """\
#include "pq.h"

int main(int argc, char**) {
    const int i = argc % 3, j = (argc / 3) % 3;
    volatile auto pq = pq_kernels[i][j];
    return pq == nullptr || pq_block_dim[1] != FJ_WAVEFRONT;
}
"""


PQ_AXES = {"P": ("p_x", "p_y", "p_z"), "Q": ("q_x", "q_y", "q_z")}


@pytest.mark.parametrize("any_wavefront", [False, True],
                         ids=["fixed-width", "any-width"])
@pytest.mark.parametrize("split", [False, True], ids=["interleaved", "split"])
@pytest.mark.parametrize("tool", TOOLCHAINS)
def test_every_pq_class_links_as_its_own_unit_under_a_fixed_block(
        request, tmp_path, tool: str, split: bool, any_wavefront: bool) -> None:
    # split: passes.split_axes before both emitters, as a per-axis consumer
    # composes them; the header and every unit then agree on p_x ... q_z
    tc = request.getfixturevalue(tool)

    def pq(lp: int, lq: int):
        k = _pq(lp, lq)
        return split_axes(k, PQ_AXES) if split else k

    _boys(tmp_path)
    (tmp_path / PQ22.header).write_text(gpu.emit_template(
        pq(2, 2), SM_86, PQ22, PQ_CLASSES, block_dim=ANY,
        any_wavefront=any_wavefront))
    units = {f"pq_{lp}{lq}.{tc.ext}": gpu.emit(
                 pq(lp, lq), SM_86, gpu.Specialization("pq", (lp, lq)),
                 block_dim=ANY, any_wavefront=any_wavefront)
             for lp, lq in PQ_CLASSES}
    for name, text in {**units, f"main.{tc.ext}": PQ_LAUNCHER}.items():
        (tmp_path / name).write_text(text)
    flags = [*tc.compile_flags(), f"-I{INCLUDE}", f"-I{tmp_path}"]
    if tc.name == "nvcc":
        flags.append("-rdc=true")
    done = subprocess.run(
        [tc.path, *flags, *units, f"main.{tc.ext}", "-o", "probe"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("tool", TOOLCHAINS)
def test_two_units_with_eval_boys_share_one_tu(
        request, tmp_path, compile_device_source, tool: str) -> None:
    tc = request.getfixturevalue(tool)
    _headers(tmp_path)
    f = gpu.Specialization("f", (0, 0, 0, 0))
    fk = build_f_kernel(f_class_spec(0, 0, 0, 0))
    (tmp_path / f.header).write_text(gpu.emit_template(fk, SM_86, f, [f.args]))
    text = (gpu.emit(build_v_kernel((0, 0)), SM_86,
                     gpu.Specialization("v", (0, 0)))
            + gpu.emit(fk, SM_86, f))
    # one routine, #pragma once (#301): the second unit's include is skipped
    compile_device_source(tc, text, link=False)


@pytest.mark.parametrize("tool", TOOLCHAINS)
def test_an_order_past_the_headers_is_a_compile_error(
        request, tmp_path, compile_device_source, tool: str) -> None:
    tc = request.getfixturevalue(tool)
    _boys(tmp_path)
    probe = (f'#include "{gpu.BOYS_HEADER}"\n'
             "__global__ void probe(double T, double* out) {\n"
             "    double F[{m} + 1];\n"
             "    eval_boys<{m}>(T, F);\n"
             "    *out = F[0];\n"
             "}\n")
    compile_device_source(tc, probe.replace("{m}", str(gpu.BOYS_MAX)), link=False)
    with pytest.raises(AssertionError, match="written for M in 0..20"):
        compile_device_source(tc, probe.replace("{m}", str(gpu.BOYS_MAX + 1)),
                              link=False)
