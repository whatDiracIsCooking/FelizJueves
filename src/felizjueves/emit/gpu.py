"""The vendor-neutral backend: a `Kernel` as one text both device compilers take.

    emit(kernel, arch, specialization=None) -> str
    emit_template(kernel, arch, specialization, classes) -> str

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
body is the same text, character for character.

The template is declared in a header of its own, `{template}.h`, which
`emit_template` writes from the classes the consumer emitted and every unit
of the template includes by that name, so the consumer writes it beside the
units. It holds, in order:

- the primary, defined nowhere, so a class whose unit was not compiled in is
  a link error, never a launch through a null pointer; and a unit whose
  parameters have drifted from the header's no longer specializes it, a
  compile error. Its parameters are `uint8_t La, uint8_t Lb, ...`, one per
  `args` entry, lettered by position (an angular momentum fits a byte;
  `Specialization` refuses one that does not), and fixed in number: nvcc 13.0
  miscompiles the host stub of an explicit specialization of a variadic
  `__global__` template (`'__T0' was not declared`), where clang takes both;
- one explicit specialization declared per class, which C++ requires before
  the table below names it: a specialization first used undeclared is an
  implicit instantiation, and the unit that then defines it is ill-formed;
- `{template}_kernels`, the consumer's dispatch table: an array indexed by
  the class's `args`, each extent one past the largest value at that
  position, holding `&{template}<args...>` for every class emitted and
  `nullptr` elsewhere. The classes are written once, here, so the table, the
  declarations and the units the consumer compiles are one list.

Each unit is its own translation unit, and the consumer launches it from
another through the table. Build it as relocatable device code
(`-rdc=true`) under nvcc: in whole-program mode nvcc 13.0's static template
stubs want a `__global__` specialization defined in every TU that names it,
and it says so (warning #20280, an error under `-Werror`). hipcc takes it
either way. `fj_boys` is `static` in a specialization, so the
same routine in every unit is never a duplicate symbol at the device link.

A specialization's units may still share one translation unit, so its text
is what can be included beside its siblings: `fj_boys` sits behind an include
guard (`FJ_BOYS_NARROW` / `FJ_BOYS_WIDE`, so a unit of each spelling in one
TU is a redefinition, not a silent pick), and a kernel with a record struct
or a dynamically sized shared array is refused, both being per-unit
declarations a second unit would repeat. The C spelling is unchanged and
still means one unit per TU, or one namespace each.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from felizjueves.device import Arch
from felizjueves.emit._c import (
    _GEOMETRY,
    BOYS_DEVICE_WIDE,
    INDENT,
    RESERVED,
    CWriter,
    EmitError,
    _records,
    precision,
)
from felizjueves.ir import Call, Expr, Fn, Kernel, _ident

__all__ = ["HEADER", "Specialization", "emit", "emit_template"]

# The one include every unit carries; resolved against FelizJueves's include/.
HEADER = "felizjueves/runtime.h"


# The primary's parameters, by position: `La, Lb, ...`.
_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# A C cast in `fj_boys`'s fixed text, `(double)(2 * m + 1)`.
_C_CAST = re.compile(r"\((double|float)\)\(([^()]*)\)")


def _geometry_only(e: Expr) -> bool:
    """Launch geometry and `+`/`*` of it, nothing else: unsigned throughout."""
    if not isinstance(e, Call):
        return False
    if e.fn in _GEOMETRY:
        return True
    return e.fn in (Fn.ADD, Fn.MUL) and all(_geometry_only(a) for a in e.args)


def _geometry(e: Call) -> str:
    """A `_geometry_only` expression in the hardware's own unsigned values."""
    if e.fn in _GEOMETRY:
        return _GEOMETRY[e.fn]
    if e.fn is Fn.ADD:
        return " + ".join(_geometry(x) for x in e.args)
    # `*` binds tighter than `+`, so only a sum under a product is bracketed
    return " * ".join(f"({_geometry(x)})" if x.fn is Fn.ADD else _geometry(x)
                      for x in e.args)


def _name(name: str, what: str) -> None:
    _ident(name, what)
    if name.startswith(RESERVED):
        raise ValueError(f"{what} {name!r} uses the writer's {RESERVED!r} prefix")


def _class(args: object, what: str) -> tuple[int, ...]:
    if not isinstance(args, tuple) or not args or not all(
            isinstance(a, int) and not isinstance(a, bool) for a in args):
        raise ValueError(f"{what} {args!r} is not a nonempty tuple of ints")
    return args


def _table(extents: list[int], at: tuple[int, ...], cells: dict[tuple[int, ...], str],
           indent: str) -> str:
    """The braced initializer of the dispatch table from position `at` down:
    one row per line at the innermost level, `nullptr` where no class is."""
    if len(at) + 1 == len(extents):
        return "{" + ", ".join(cells.get((*at, i), "nullptr")
                               for i in range(extents[-1])) + "}"
    inner = indent + INDENT
    rows = "".join(f"{inner}{_table(extents, (*at, i), cells, inner)},\n"
                   for i in range(extents[len(at)]))
    return "{\n" + rows + indent + "}"


@dataclass(frozen=True)
class Specialization:
    """One explicit specialization, `template<args...>`, of a kernel template.

    Every unit of one template gives the same number of `args`: the primary
    `{template}.h` declares (`emit_template`) and each unit includes is
    `template <uint8_t La, ..., uint8_t Ln>`.
    """

    template: str
    args: tuple[int, ...]

    def __post_init__(self) -> None:
        _name(self.template, "template name")
        _class(self.args, "Specialization args")
        if not all(0 <= a <= 255 for a in self.args):
            raise ValueError(f"Specialization args {self.args!r} do not fit "
                             "the primary's uint8_t parameters")
        if len(self.args) > len(_LETTERS):
            raise ValueError(f"Specialization args {self.args!r}: more than "
                             f"{len(_LETTERS)} parameters")

    def __str__(self) -> str:
        return f"{self.template}<{', '.join(map(str, self.args))}>"

    @property
    def header(self) -> str:
        """The file the primary is declared in, beside the units."""
        return f"{self.template}.h"

    def primary(self) -> str:
        """The primary's template head: one `uint8_t` per `args` entry,
        lettered by position, `La, Lb, ...`."""
        params = ", ".join(f"uint8_t L{c}" for c in _LETTERS[:len(self.args)])
        return f"template <{params}>"


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
        if spec is not None:
            head += f'#include "{spec.header}"  // the primary\n'
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
        return f"template <>\n__global__ void {spec}(\n{sig})"

    def primary(self, k: Kernel, classes: Sequence[tuple[int, ...]]) -> str:
        """`{template}.h`: the primary, each class's declaration, and the
        dispatch table over `classes`."""
        spec = self.specialization
        if spec is None:
            raise TypeError("a primary needs a gpu.Specialization")
        name = spec.template
        rows = [Specialization(name, _class(c, "class")) for c in classes]
        if not rows:
            raise ValueError(f"{name}: no classes")
        if any(len(r.args) != len(spec.args) for r in rows):
            raise ValueError(f"{name}: classes of other than {len(spec.args)} "
                             f"args in {[r.args for r in rows]!r}")
        if len(set(rows)) != len(rows):
            raise ValueError(f"{name}: a class is listed twice in "
                             f"{[r.args for r in rows]!r}")
        sig = ",\n".join(INDENT + self.param(p) for p in k.params)
        declared = "".join(f"template <>\n__global__ void {r}(\n{sig});\n\n"
                           for r in rows)
        extents = [max(r.args[i] for r in rows) + 1 for i in range(len(spec.args))]
        cells = {r.args: f"&{r}" for r in rows}
        dims = "".join(f"[{n}]" for n in extents)
        return (
            f"// {name}: generated by felizjueves.emit.gpu -- do not edit.\n"
            f"// the kernel template every {name}<...> unit specializes, one\n"
            "// declaration per class emitted, and its dispatch table: the\n"
            "// primary is defined in no unit, so a class whose unit is not\n"
            "// compiled in fails to link.\n"
            "#pragma once\n"
            f'#include "{HEADER}"\n'
            "\n"
            f"{spec.primary()}\n"
            f"__global__ void {name}(\n{sig});\n"
            "\n"
            f"{declared}"
            f"// indexed by the class's args; nullptr where no class was emitted.\n"
            f"inline constexpr decltype({cells[rows[0].args]}) {name}_kernels{dims} = "
            f"{_table(extents, (), cells, '')};\n"
        )

    def call(self, e: Call) -> str:
        if _geometry_only(e):
            return f"static_cast<{self.INT}>({_geometry(e)})"
        return super().call(e)

    def cast(self, t: str, text: str) -> str:
        return f"static_cast<{t}>({text})"

    def boys(self, text: str) -> str:
        spelled = _C_CAST.sub(r"static_cast<\1>(\2)", super().boys(text))
        if self.specialization is None:
            return spelled
        # internal linkage: every unit of a template carries its own copy
        spelled = spelled.replace("__device__ double fj_boys(",
                                  "static __device__ double fj_boys(", 1)
        guard = "FJ_BOYS_WIDE" if text == BOYS_DEVICE_WIDE else "FJ_BOYS_NARROW"
        return f"#ifndef {guard}\n#define {guard}\n{spelled}#endif\n"

    def _mask(self) -> str:
        self._width_used = True
        return f"static_cast<FJ_LANE_MASK_T>(0x{(1 << self.arch.wavefront) - 1:x}ull)"

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


def emit_template(kernel: Kernel, arch: Arch, specialization: Specialization,
                  classes: Sequence[tuple[int, ...]]) -> str:
    """`specialization.header`'s text: the template's primary, one
    specialization declared per class, and the dispatch table over them.

    The parameters are `kernel`'s, as any of the template's units spells them;
    a unit whose parameters differ no longer specializes it. `classes` are
    the `args` of every unit the consumer emitted, in emission order, each as
    many as `specialization`'s and none twice.
    """
    if not isinstance(specialization, Specialization):
        raise TypeError(f"specialization is a gpu.Specialization, "
                        f"not {specialization!r}")
    return GpuWriter(arch, specialization).primary(kernel, classes)
