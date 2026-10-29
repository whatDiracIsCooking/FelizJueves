"""The vendor-neutral backend: a `Kernel` as one text both device compilers take.

    emit(kernel, arch, specialization=None) -> str
    emit_template(kernel, arch, specialization, classes) -> str
    emit_boys() -> dict[str, str]

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
  counters, the thread-geometry casts and `eval_boys` alike (`CWriter.INT`).
  The header brings in `<stdint.h>`. The vendor backends still spell `int`,
  so their text and its pins are unchanged;
- the device Boys routine, `eval_boys<M>`, as a header of its own,
  `eval_boys.h` (`BOYS_HEADER`, written by `emit_boys` with the per-order
  headers it includes), which a unit with a fundamental includes after the
  runtime and calls by that name. `static`, so every unit that includes it
  carries its own copy and none is a duplicate symbol at the device link.
  It is `_c.py`'s `BOYS_DEVICE` with the series loop replaced ("THE BOYS
  SERIES", below); the vendor backends still carry that text inline as
  `fj_boys`, so their text and its pins are unchanged.

THE BOYS SERIES, below the switch `T = M + BOYS_SWITCH_PAD`, is a polynomial
of fixed degree, not a loop. `F_M(T) = exp(-T) SUM_k c_k x^k` with `x = 2T`
and `c_k = 1 / ((2M+1)(2M+3)...(2M+2k+1))`, and M is fixed when the kernel
is built, so each `c_k` is a constant: `eval_boys_series<M>(x)` in
`eval_boys_{M}.h`, one header per order `M <= BOYS_MAX`, is Horner in `x`,
one `fma` a step from `c_n` down to `c_0`, each `c_k` the double nearest the
exact rational. `n` is the least degree whose tail is under 2^-53 of the
whole sum at `T = M + BOYS_SWITCH_PAD`, the most the branch is given (every
`c_k` and `x` positive, the tail's share of the sum grows with `x`): 40 at
M = 0, 61 at M = 20. Where the loop took up to that many dependent
divisions and a data-dependent exit, every lane now runs the same fma
chain. `eval_boys<M>` for `M > BOYS_MAX` is refused at compile time.

THE WIDTH IS NOT ERASED. The shuffle hooks read the wavefront off the `Arch`,
as the vendor backends do, and fix it in the text. A unit that calls either
hook therefore carries `static_assert(FJ_WAVEFRONT == w, ...)` after the
include, and the header's `FJ_WAVEFRONT` (32 unless the build passes
`-DFJ_WAVEFRONT=64`) refuses a mismatched compile. A unit that calls neither
-- `S`, `T`, `V`, `DQ`, `JAB` -- has no width in its text, carries no
assert, and its banner says so. Whether a hook was called is read after the
body is written, as `CWriter.kernel` already does for Boys and records:
a spelling of what the kernel contains, not a decision about it.

The `Arch` is taken for its wavefront only: any vendor's arch is accepted,
and the text names no arch, so one emission serves every arch of that width.

UNLESS THE CONSUMER ERASES IT: `any_wavefront=True` writes one text for
both widths. The shuffle hooks then spell the width as the header's
`FJ_WAVEFRONT` -- the reduction's first offset `FJ_WAVEFRONT / 2`, every
shuffle's width `FJ_WAVEFRONT`, its mask `FJ_FULL_LANE_MASK` -- and the unit
carries no `static_assert`: the build's `-DFJ_WAVEFRONT` (32, or 64 for
CDNA) chooses the width at compile time, as WarpWraps's `WWR_WARP_SIZE`
does for its own device code. A configure-time `-D` is the one spelling
that is the same in both of HIP's compile passes: the vendors'
(`__AMDGCN_WAVEFRONT_SIZE__`, which reads 64 in the host pass and 32 in an
RDNA device pass and which clang 20 deprecates; `warpSize`, which is not
a constant expression) are not. The price is the guarantee above: such a
unit's text no longer fixes the width its machine code has, so a build that
passes the wrong `-DFJ_WAVEFRONT` for its arch computes a wrong sum rather
than failing to compile. That is why it is opt-in and the default text,
every golden and every `source_sha256` pin are unchanged. `gpu.WAVEFRONT`
stands for the width in a `block_dim` (`(1, gpu.WAVEFRONT)`, `[p|q]`'s
`launch_shape`): under `any_wavefront` a block-sized shared array is
`Jsh[10 * FJ_WAVEFRONT]` and the header's `{template}_block_dim` is
`{1, FJ_WAVEFRONT}`; without it, `gpu.WAVEFRONT` is the arch's own width.
A unit with no shuffle and no block-sized array is the same text either way.

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
either way. `eval_boys` is `static`, so the same routine in every unit is
never a duplicate symbol at the device link.

A specialization's units may still share one translation unit, so its text
is what can be included beside its siblings: `eval_boys.h` is `#pragma once`,
and a kernel with a record struct or a dynamically sized shared array is
refused, both being per-unit declarations a second unit would repeat. The C
spelling is unchanged and still means one unit per TU, or one namespace each.

A FIXED BLOCK, chosen by the consumer and never by the kernel: with
`block_dim=(x, y)`, a SHARED array whose extents are literals and the block's
own dimensions (`BLOCK_DIM_X`, `BLOCK_DIM_Y`, and `+`/`*` of them) is sized
at emit time and declared `__shared__ T name[n];`, flat, in place of the
block's dynamic shared memory. The body is the same text, character for
character -- it still indexes by `blockDim`, row-major -- so the unit is
correct only when launched with exactly that block, which its declaration's
comment names and `emit_template(..., block_dim=)` writes into the header
as `{template}_block_dim`, for the launcher to read rather than restate. A
kernel left with no dynamic shared array is no longer refused as a
specialization: `[p|q]` (`Jsh`, `n_p x blockDim.y`) becomes one with
`block_dim=(1, wavefront)`, its `launch_shape`. An extent of anything else
(a parameter, say) stays dynamic. A kernel with no launch-time extent
ignores the option, so its text does not move.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from felizjueves.emit._c import (
    _GEOMETRY,
    BOYS_SWITCH_PAD,
    INDENT,
    RESERVED,
    CWriter,
    EmitError,
    _Array,
    _records,
    has_boys,
    kernel_names,
    precision,
)
from felizjueves.ir import Call, Decl, Expr, Fn, Kernel, Lit, Space, _ident
from felizjueves.performance.device import Arch

__all__ = ["BOYS_HEADER", "BOYS_MAX", "HEADER", "WAVEFRONT", "Specialization",
           "boys_series", "emit", "emit_boys", "emit_template"]

# The one include every unit carries; resolved against FelizJueves's include/.
HEADER = "felizjueves/runtime.h"

# The device Boys routine's header (`emit_boys`), which the consumer writes
# beside the units or on their include path, and its one name; the series
# it calls, one header per order 0..BOYS_MAX ("The Boys series").
BOYS_HEADER = "eval_boys.h"
BOYS_MAX = 20
_BOYS = "eval_boys"
_SERIES = "eval_boys_series"

# The header's spellings of the width and of a full mask (any_wavefront).
_WIDTH = "FJ_WAVEFRONT"
_FULL_MASK = "FJ_FULL_LANE_MASK"


class _Wavefront:
    """The lanes of a warp, as a `block_dim` entry: the arch's width, or
    `FJ_WAVEFRONT` under `any_wavefront`."""

    def __repr__(self) -> str:
        return "gpu.WAVEFRONT"


WAVEFRONT = _Wavefront()


# The primary's parameters, by position: `La, Lb, ...`.
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _geometry_only(e: Expr) -> bool:
    """Launch geometry and `+`/`*` of it, nothing else: unsigned throughout."""
    if not isinstance(e, Call):
        return False
    if e.fn in _GEOMETRY:
        return True
    return e.fn in (Fn.ADD, Fn.MUL) and all(_geometry_only(a) for a in e.args)


def _arith(fn: Fn, a: int | str, b: int | str) -> int | str:
    """`a + b` or `a * b`: folded when both are ints, else spelled, a sum
    parenthesised so a product of it reads right."""
    if isinstance(a, int) and isinstance(b, int):
        return a + b if fn is Fn.ADD else a * b
    if fn is Fn.ADD:
        return f"({a} + {b})"
    if a == 1 or b == 1:
        return b if a == 1 else a
    return f"{a} * {b}"


def _block_extent(e: Expr, block_dim: tuple[int | str, int | str]
                  ) -> int | str | None:
    """`e`'s value under a fixed block -- an int, or C text when the block
    names `FJ_WAVEFRONT` -- or None when it is not literals and the block's
    dimensions under `+`/`*`."""
    if isinstance(e, Lit):
        return e.value if type(e.value) is int else None
    if not isinstance(e, Call):
        return None
    if e.fn is Fn.BLOCK_DIM_X:
        return block_dim[0]
    if e.fn is Fn.BLOCK_DIM_Y:
        return block_dim[1]
    if e.fn not in (Fn.ADD, Fn.MUL):
        return None
    a, b = (_block_extent(x, block_dim) for x in e.args)
    if a is None or b is None:
        return None
    return _arith(e.fn, a, b)


def _block_dim(block_dim: object, width: int | str
               ) -> tuple[int | str, int | str] | None:
    """`block_dim` with `WAVEFRONT` resolved to `width`."""
    if block_dim is None:
        return None
    if (not isinstance(block_dim, tuple) or len(block_dim) != 2
            or any(d is not WAVEFRONT and (type(d) is not int or d < 1)
                   for d in block_dim)):
        raise ValueError(f"block_dim is (x, y), two positive ints or "
                         f"gpu.WAVEFRONT, not {block_dim!r}")
    x, y = (width if d is WAVEFRONT else d for d in block_dim)
    return x, y


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
    BOYS = _BOYS  # from BOYS_HEADER, never carried in the unit
    RYS = None  # the Rys rule has no header of its own yet: refused

    def __init__(self, arch: Arch,
                 specialization: Specialization | None = None, *,
                 block_dim: tuple[int, int] | None = None,
                 any_wavefront: bool = False) -> None:
        super().__init__(arch)
        if specialization is not None and not isinstance(specialization,
                                                         Specialization):
            raise TypeError(f"specialization is a gpu.Specialization, "
                            f"not {specialization!r}")
        if type(any_wavefront) is not bool:
            raise TypeError(f"any_wavefront is a bool, not {any_wavefront!r}")
        self.specialization = specialization
        self.any_wavefront = any_wavefront
        # the width as the text spells it: the arch's, or the header's macro
        self.width: int | str = _WIDTH if any_wavefront else arch.wavefront
        self.block_dim = _block_dim(block_dim, self.width)
        self._width_used = False

    def kernel(self, k: Kernel) -> str:
        self._width_used = False
        taken = {_BOYS, _SERIES} & set(kernel_names(k))
        if taken:
            raise EmitError(f"{k.name}: the names {sorted(taken)} are "
                            f"{BOYS_HEADER}'s")
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
        w = self.width
        if not self._width_used:
            width = "any wavefront"
        elif self.any_wavefront:
            width = f"wavefront {_WIDTH}, chosen at compile time"
        else:
            width = f"wavefront {w}"
        spec = self.specialization
        named = "" if spec is None else f"; specialization {spec}"
        head = (
            f"// {k.name}: generated by felizjueves.emit.gpu -- do not edit.\n"
            f"// {width}; {precision(k)}; nvcc and hipcc alike{named}.\n"
            f'#include "{HEADER}"\n'
        )
        if spec is not None:
            head += f'#include "{spec.header}"  // the primary\n'
        if has_boys(k):
            head += f'#include "{BOYS_HEADER}"\n'
        if self._width_used and not self.any_wavefront:
            head += (
                f"static_assert(FJ_WAVEFRONT == {w}, "
                f'"{k.name} was emitted for a {w}-lane wavefront");\n'
            )
        return head

    def decl(self, s: Decl, depth: int) -> None:
        if (self.block_dim is None or s.space is not Space.SHARED
                or not s.shape or all(isinstance(d, Lit) for d in s.shape)):
            return super().decl(s, depth)
        extents = [_block_extent(d, self.block_dim) for d in s.shape]
        if None in extents:
            return super().decl(s, depth)
        n: int | str = 1
        for e in extents:
            n = _arith(Fn.MUL, n, e)
        t = self.ctype(s.type)
        x, y = self.block_dim
        dims = " x ".join(self.expr(d) for d in s.shape)
        self._bind(s.name, _Array(s.shape, True), s.type)
        self._emit(depth, f"// {s.name}: fixed shared memory, row-major, "
                          f"{dims} {t}s, sized for blockDim = ({x}, {y}): "
                          "launch with exactly that block")
        self._emit(depth, f"__shared__ {t} {s.name}[{n}];")

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
        block = "" if self.block_dim is None else (
            "// the block every unit's fixed shared memory was sized for: "
            "launch with it.\n"
            f"inline constexpr uint32_t {name}_block_dim[2] = "
            f"{{{self.block_dim[0]}, {self.block_dim[1]}}};\n\n")
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
            f"{block}"
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
        return ""  # the prologue includes BOYS_HEADER instead

    def _mask(self) -> str:
        self._width_used = True
        if self.any_wavefront:
            return _FULL_MASK
        return f"static_cast<FJ_LANE_MASK_T>(0x{(1 << self.arch.wavefront) - 1:x}ull)"

    def warp_reduce(self, dst: str, value: str) -> list[str]:
        w, mask = self.width, self._mask()
        half = f"{w} / 2" if self.any_wavefront else w // 2
        return [
            "{",
            f"    auto fj_v = {value};",
            f"    for ({self.INT} fj_o = {half}; fj_o > 0; fj_o /= 2) {{",
            f"        fj_v += __shfl_xor_sync({mask}, fj_v, fj_o, {w});",
            "    }",
            f"    {dst} = fj_v;",
            "}",
        ]

    def lane_broadcast(self, dst: str, value: str, lane: str) -> str:
        w, mask = self.width, self._mask()
        return f"{dst} = __shfl_sync({mask}, {value}, {lane}, {w});"


def emit(kernel: Kernel, arch: Arch,
         specialization: Specialization | None = None, *,
         block_dim: tuple[int, int] | None = None,
         any_wavefront: bool = False) -> str:
    """Render a Kernel as vendor-neutral GPU C++. No decisions: `block_dim`
    (x, y) is the consumer's, and fixes the shared arrays the block sizes
    (the module docstring, "A fixed block"); `any_wavefront` spells the
    width as `FJ_WAVEFRONT` instead of the arch's ("Unless the consumer
    erases it")."""
    return GpuWriter(arch, specialization, block_dim=block_dim,
                     any_wavefront=any_wavefront).kernel(kernel)


@functools.cache
def boys_series(m: int) -> tuple[float, ...]:
    """`c_0..c_n` of `eval_boys_series<m>` ("The Boys series"): each the
    double nearest `1 / ((2m+1)(2m+3)...(2m+2k+1))`, `n` the least degree
    whose tail is under 2^-53 of the sum at `x = 2 (m + BOYS_SWITCH_PAD)`.
    Exact rational arithmetic throughout; only the coefficients are rounded."""
    if not (isinstance(m, int) and 0 <= m <= BOYS_MAX):
        raise ValueError(f"the Boys series is written for 0 <= M <= {BOYS_MAX}, "
                         f"not {m!r}")
    x = 2 * (m + BOYS_SWITCH_PAD)
    terms = [Fraction(1, 2 * m + 1)]
    # Past k with x / (2m + 2k + 3) <= 1/2 the tail after k is at most the
    # k-th term, so stopping there with that term under 2^-80 of the sum
    # leaves an unsummed remainder that cannot move the degree.
    while not (2 * x <= 2 * m + 2 * len(terms) + 1
               and terms[-1] * 2**80 < sum(terms)):
        terms.append(terms[-1] * x / (2 * m + 2 * len(terms) + 1))
    whole = sum(terms) + terms[-1]  # the remainder's bound included
    tail = whole
    for n, t in enumerate(terms):
        tail -= t
        if tail * 2**53 < whole:
            return tuple(float(Fraction(1, _odd_product(m, k)))
                         for k in range(n + 1))
    raise AssertionError("unreachable: the remainder is under 2^-80")


def _odd_product(m: int, k: int) -> int:
    """`(2m+1)(2m+3)...(2m+2k+1)`."""
    p = 1
    for j in range(k + 1):
        p *= 2 * m + 2 * j + 1
    return p


def emit_boys() -> dict[str, str]:
    """Every header of the device Boys routine, by file name: `BOYS_HEADER`
    (`eval_boys<M>`, which every unit with a fundamental includes) and the
    `eval_boys_{M}.h` it includes, one series per order 0..BOYS_MAX. One text
    for every kernel, arch and option; the consumer writes them all once, in
    one directory, beside the units or on their include path."""
    banner = "generated by felizjueves.emit.gpu -- do not edit.\n"
    fp64 = "// fp64 throughout; nvcc and hipcc alike.\n"
    orders = "".join(f'#include "{_BOYS}_{m}.h"\n' for m in range(BOYS_MAX + 1))
    pad = BOYS_SWITCH_PAD
    headers = {BOYS_HEADER: f"""\
// {_BOYS}: {banner}\
// the device Boys routine every unit with a fundamental includes;
{fp64}\
#pragma once
#include "{HEADER}"

// F_m(T) = integral_0^1 u^(2m) exp(-T u^2) du ([JEngine, v2, eqs. 13, 18]),
// F[m] for every m in 0..M, M <= {BOYS_MAX}. From T = M + {pad}: F_0 = (1/2)
// sqrt(pi/T) erf(sqrt(T)) and upward, F_(m+1) = ((2m+1) F_m - exp(-T)) / (2T).
// Below: F_M = exp(-T) {_SERIES}<M>(2T), the Kummer series
// SUM_k x^k / ((2M+1)(2M+3)...(2M+2k+1)) as a polynomial of fixed degree
// ({_BOYS}_M.h), and downward, F_m = (2T F_(m+1) + exp(-T)) / (2m+1).
template <int32_t M>
static __device__ __forceinline__ double {_SERIES}(double x);

{orders}
template <int32_t M>
static __device__ __forceinline__ void {_BOYS}(double T, double (&F)[M + 1]) {{
    static_assert(M >= 0 && M <= {BOYS_MAX},
                  "{_BOYS}_M.h is written for M in 0..{BOYS_MAX}");
    const double e = exp(-T);
    if (T >= static_cast<double>(M + {pad})) {{
        const double h = 0.5 / T;
        double f = 0.5 * sqrt({math.pi!r} / T) * erf(sqrt(T));
        F[0] = f;
#pragma unroll
        for (int32_t m = 1; m <= M; ++m) {{
            f = (static_cast<double>(2 * m - 1) * f - e) * h;
            F[m] = f;
        }}
        return;
    }}
    double f = e * {_SERIES}<M>(2.0 * T);
    F[M] = f;
#pragma unroll
    for (int32_t m = M - 1; m >= 0; --m) {{
        f = (2.0 * T * f + e) * (1.0 / static_cast<double>(2 * m + 1));
        F[m] = f;
    }}
}}
"""}
    for m in range(BOYS_MAX + 1):
        c = boys_series(m)
        steps = "".join(f"{INDENT}s = fma(s, x, {ck!r});\n"
                        for ck in reversed(c[:-1]))
        headers[f"{_BOYS}_{m}.h"] = f"""\
// {_BOYS}_{m}: {banner}\
// {_SERIES}<{m}>(x), x = 2T < {2 * (m + pad)}: SUM_(k=0..{len(c) - 1}) c_k x^k,
// c_k the double nearest 1 / ({2 * m + 1} * {2 * m + 3} * ... * (2k + {2 * m + 1})),
// by Horner; past degree {len(c) - 1} the tail is under 2^-53 of the sum.
// Included by {BOYS_HEADER}, which declares the template; not on its own.
{fp64}\
#pragma once

template <>
__device__ __forceinline__ double {_SERIES}<{m}>(double x) {{
{INDENT}double s = {c[-1]!r};
{steps}{INDENT}return s;
}}
"""
    return headers


def emit_template(kernel: Kernel, arch: Arch, specialization: Specialization,
                  classes: Sequence[tuple[int, ...]], *,
                  block_dim: tuple[int, int] | None = None,
                  any_wavefront: bool = False) -> str:
    """`specialization.header`'s text: the template's primary, one
    specialization declared per class, and the dispatch table over them.

    The parameters are `kernel`'s, as any of the template's units spells them;
    a unit whose parameters differ no longer specializes it. `classes` are
    the `args` of every unit the consumer emitted, in emission order, each as
    many as `specialization`'s and none twice. `block_dim` and
    `any_wavefront`, the ones the units were emitted with, write
    `{template}_block_dim`.
    """
    if not isinstance(specialization, Specialization):
        raise TypeError(f"specialization is a gpu.Specialization, "
                        f"not {specialization!r}")
    return GpuWriter(arch, specialization, block_dim=block_dim,
                     any_wavefront=any_wavefront).primary(kernel, classes)
