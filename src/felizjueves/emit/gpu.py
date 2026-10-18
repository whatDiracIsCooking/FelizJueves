"""The vendor-neutral backend: a `Kernel` as one text both device compilers take.

    emit(kernel, arch, specialization=None) -> str
    emit_classes(name, classes) -> str

A pure function: no decisions, no I/O. Where `emit/cuda.py` and `emit/hip.py`
each spell a kernel for one vendor, this spells it once for both: the text
compiles under nvcc and under hipcc / clang -x hip alike. Everything the two
vendors spell alike is still `emit/_c.py`'s `CWriter`; the rest is pushed
into FelizJueves's own headers (`include/felizjueves/`, no GPU library
behind them), which the unit includes and the compiler resolves:

- the prologue: a header comment naming the kernel and the wavefront it was
  emitted for, and `#include "felizjueves/runtime.h"`, which pulls in the
  vendor runtime the compiling front end needs (none under nvcc, the HIP
  runtime under HIP). A consumer puts `include/` on the include path;
- `WarpReduce` and `LaneBroadcast` (#349) as the vendors' shared
  `__shfl_xor_sync` / `__shfl_sync`, the full mask cast to
  `FJ_LANE_MASK_T` -- CUDA's 32-bit mask type or HIP's 64-bit one, chosen by
  the header, so the literal is the one thing written here;
- `AtomicAdd` as `atomicAdd` (the shared default);
- every integer as `int32_t`, never a bare `int` -- declarations, loop
  counters, the thread-geometry casts and `fj_boys` alike (`CWriter.INT`).
  The header brings in `<stdint.h>`. The vendor backends still spell `int`,
  so their text and its pins are unchanged.

THE WIDTH IS NOT ERASED. The shuffle hooks read the wavefront off the `Arch`,
as the vendor backends do, and fix it in the text. A unit that calls either
hook therefore carries `static_assert(FJ_WAVEFRONT == w, ...)` after the
include, and the header's `FJ_WAVEFRONT` (32 unless the build passes
`-DFJ_WAVEFRONT=64`) refuses a mismatched compile. A unit that calls neither
-- `S`, `T`, `V`, `DQ`, `JAB` -- has no width in its text, carries no
assert, and its banner says so. Whether a hook was called is read after the
body is written, as `CWriter.kernel` already does for `fj_boys` and records:
a spelling of what the kernel contains, not a decision about it.

The `Arch` is taken for its wavefront only: any vendor's arch is accepted,
and the text names no arch, so one emission serves every arch of that width.

TWO SPELLINGS OF THE KERNEL'S NAME, chosen by the consumer and never by the
kernel: the IR has no templates, and which spelling a tree links is not a fact
about the kernel. By default a unit is `extern "C" __global__ void s_01(...)`,
one C name per class. With a `Specialization` it is one explicit
specialization of a kernel template, `template <> __global__ void s<0, 1>(...)`,
so a consumer can take `&s<la, lb>` from a compile-time `(la, lb)` and build its
dispatch table with no name written per class. Only the declarator moves; the
body is the same text, character for character. The template's primary is
declared in every unit and defined in none, so a class whose unit was not
compiled in is a link error, never a launch through a null pointer. Its
parameters are fixed in number, `int32_t L0, int32_t L1, ...`, one per
`args` entry: nvcc 13.0 miscompiles the host stub of an explicit
specialization of a variadic `__global__` template (`'__T0' was not
declared`), where clang takes both.

A specialization's units are meant to share one translation unit, so its text
is what can be included beside its siblings: `fj_boys` sits behind an include
guard (`FJ_BOYS_NARROW` / `FJ_BOYS_WIDE`, so a unit of each spelling in one
TU is a redefinition, not a silent pick), and a kernel with a record struct
or a dynamically sized shared array is refused, both being per-unit
declarations a second unit would repeat. The C spelling is unchanged and
still means one unit per TU, or one namespace each.

`emit_classes` writes the list of classes a consumer emitted, as a header of
one `constexpr` array, in either spelling: the units a consumer compiled are
then a fact its code can `static_assert` against, not a second list kept in
step by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from felizjueves.device import Arch
from felizjueves.emit._c import (
    BOYS_DEVICE_WIDE,
    RESERVED,
    CWriter,
    EmitError,
    _records,
    precision,
)
from felizjueves.ir import Kernel, _ident

__all__ = ["HEADER", "Specialization", "emit", "emit_classes"]

# The one include every unit carries; resolved against FelizJueves's include/.
HEADER = "felizjueves/runtime.h"


def _name(name: str, what: str) -> None:
    _ident(name, what)
    if name.startswith(RESERVED):
        raise ValueError(f"{what} {name!r} uses the writer's {RESERVED!r} prefix")


def _class(args: object, what: str) -> tuple[int, ...]:
    if not isinstance(args, tuple) or not args or not all(
            isinstance(a, int) and not isinstance(a, bool) for a in args):
        raise ValueError(f"{what} {args!r} is not a nonempty tuple of ints")
    return args


@dataclass(frozen=True)
class Specialization:
    """One explicit specialization, `template<args...>`, of a kernel template.

    Every unit of one template gives the same number of `args`: the primary
    each unit declares is `template <int32_t L0, ..., int32_t Ln>`.
    """

    template: str
    args: tuple[int, ...]

    def __post_init__(self) -> None:
        _name(self.template, "template name")
        _class(self.args, "Specialization args")

    def __str__(self) -> str:
        return f"{self.template}<{', '.join(map(str, self.args))}>"


class GpuWriter(CWriter):
    INT = "int32_t"  # fixed width; <stdint.h> comes in through HEADER

    def __init__(self, arch: Arch,
                 specialization: Specialization | None = None) -> None:
        super().__init__(arch)
        if specialization is not None and not isinstance(specialization,
                                                         Specialization):
            raise TypeError(f"specialization is a gpu.Specialization, "
                            f"not {specialization!r}")
        self.specialization = specialization
        self._width_used = False

    def kernel(self, k: Kernel) -> str:
        self._width_used = False
        if self.specialization is not None and _records(k):
            raise EmitError(f"{k.name}: a specialization shares its TU, and "
                            "its record structs would be declared per unit")
        text = super().kernel(k)
        if self.specialization is not None and self._dynamic_shared:
            raise EmitError(f"{k.name}: a specialization shares its TU, and "
                            "its dynamic shared array would be declared per "
                            "unit")
        return text

    def prologue(self, k: Kernel) -> str:
        w = self.arch.wavefront
        width = f"wavefront {w}" if self._width_used else "any wavefront"
        spec = self.specialization
        named = "" if spec is None else f"; specialization {spec}"
        head = (
            f"// {k.name}: generated by felizjueves.emit.gpu -- do not edit.\n"
            f"// {width}; {precision(k)}; nvcc and hipcc alike{named}.\n"
            f'#include "{HEADER}"\n'
        )
        if self._width_used:
            head += (
                f"static_assert(FJ_WAVEFRONT == {w}, "
                f'"{k.name} was emitted for a {w}-lane wavefront");\n'
            )
        return head

    def signature(self, k: Kernel, sig: str) -> str:
        spec = self.specialization
        if spec is None:
            return super().signature(k, sig)
        params = ", ".join(f"{self.INT} L{i}" for i in range(len(spec.args)))
        return (
            "// the primary is declared in every unit and defined in none: a\n"
            "// class whose unit is not compiled in fails to link\n"
            f"template <{params}>\n"
            f"__global__ void {spec.template}(\n{sig});\n"
            "\n"
            "template <>\n"
            f"__global__ void {spec}(\n{sig})"
        )

    def boys(self, text: str) -> str:
        spelled = super().boys(text)
        if self.specialization is None:
            return spelled
        guard = "FJ_BOYS_WIDE" if text == BOYS_DEVICE_WIDE else "FJ_BOYS_NARROW"
        return f"#ifndef {guard}\n#define {guard}\n{spelled}#endif\n"

    def _mask(self) -> str:
        self._width_used = True
        return f"((FJ_LANE_MASK_T)0x{(1 << self.arch.wavefront) - 1:x}ull)"

    def warp_reduce(self, dst: str, value: str) -> list[str]:
        w, mask = self.arch.wavefront, self._mask()
        return [
            "{",
            f"    auto fj_v = {value};",
            f"    for ({self.INT} fj_o = {w // 2}; fj_o > 0; fj_o /= 2) {{",
            f"        fj_v += __shfl_xor_sync({mask}, fj_v, fj_o, {w});",
            "    }",
            f"    {dst} = fj_v;",
            "}",
        ]

    def lane_broadcast(self, dst: str, value: str, lane: str) -> str:
        w, mask = self.arch.wavefront, self._mask()
        return f"{dst} = __shfl_sync({mask}, {value}, {lane}, {w});"


def emit(kernel: Kernel, arch: Arch,
         specialization: Specialization | None = None) -> str:
    """Render a Kernel as vendor-neutral GPU C++. No decisions."""
    return GpuWriter(arch, specialization).kernel(kernel)


def emit_classes(name: str, classes: Sequence[tuple[int, ...]]) -> str:
    """The header `{name}_classes`: one row per class a consumer emitted.

    `classes` in emission order, every row the same length and none twice.
    The text names no kernel, so it is the same under either spelling.
    """
    _name(name, "class list name")
    rows = [_class(c, "class") for c in classes]
    if not rows:
        raise ValueError(f"{name}: no classes")
    if len({len(r) for r in rows}) != 1:
        raise ValueError(f"{name}: classes of different lengths {rows!r}")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{name}: a class is listed twice in {rows!r}")
    body = "".join(f"    {{{', '.join(map(str, r))}}},\n" for r in rows)
    return (
        f"// {name}_classes: generated by felizjueves.emit.gpu -- do not edit.\n"
        "// one row per emitted unit, in emission order.\n"
        "#pragma once\n"
        f'#include "{HEADER}"\n'
        "\n"
        f"inline constexpr int32_t {name}_classes[{len(rows)}][{len(rows[0])}] "
        "= {\n"
        f"{body}}};\n"
    )
