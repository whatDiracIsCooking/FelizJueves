"""The vendor-neutral half of a backend: a `Kernel` spelled as device C++.

    CWriter(arch).kernel(kernel) -> str

CUDA and HIP are both C++ with the same thread-geometry builtins, the same
`__global__` / `__device__` / `__shared__` qualifiers and the same libm, so
almost every node is spelled identically for both and is spelled HERE, once.
A backend (`emit/cuda.py`, `emit/hip.py`) subclasses `CWriter` and supplies
only what its vendor spells differently -- the hooks at the bottom of the
class: the file prologue, the warp reduction, the atomic add. The IR does not
know which subclass is reading it; that is the seam #27 validates.

NO DECISIONS. Every rule below is a spelling, chosen once and applied
uniformly; nothing depends on what the kernel computes:

- expressions are fully parenthesised, so no precedence is relied on;
- a `Weighted` term `c * g * src` drops a factor only when it is exactly 1,
  and the sum is formed left to right from its first term -- the
  interpreter's order (`0.0 + x` is exact, so starting from the first term
  rounds the same);
- a global array is a flat pointer, indexed row-major by its `Param.shape`;
  a LOCAL or SHARED array with literal extents is a native C array; a SHARED
  array with a launch-time extent is the block's dynamic shared memory
  (`extern __shared__`, row-major), and there may be at most one, because
  laying out two would be a decision;
- an `ir.Member` (#353) is a read through a typed view of its row:
  `reinterpret_cast<const fj_<record>*>(array)[row].<member>`, the row
  linearised as above over every extent but the last, which must be the
  record's width; each `Record` a kernel reads through is spelled once,
  ahead of the kernel, as `struct fj_<record>` with a `static_assert` that
  it is the unpadded row. A kernel with no `Member` takes none of this, so
  its text is unchanged;
- `Note` is a `//` comment beside the statement it precedes;
- a scalar parameter the body never reads -- one that exists only as an
  array's leading extent, which row-major indexing never needs -- is cast
  to `(void)` at the top of the body, so hipcc's `-Wextra` has no unused
  parameter to warn of (#156; nvcc's `-Wall` does not ask); likewise an
  ARRAY parameter the body never indexes, binds or reads (#407: the HGP
  J / K kernels keep one signature per target, and a class whose deposits
  need no bra shell index never reads `bra_sh`), after the scalars, under
  its own comment -- a kernel that reads every array has no such line, so
  its text is unchanged;
- an `Inline` is a `{ ... }` block headed `// fragment <name>`, its bindings
  `const` declarations, then the fragment's statements in place. Nothing is
  renamed: `Kernel` has already refused any name the block could shadow.

F32 (#338) is spelled from the value's type, never from a flag: `float` for
an F32 `Param` or `Decl`; `Fn.TO_F32` / `Fn.TO_F64` as `((float)x)` /
`((double)x)`, and a `TO_F32` of a float literal (which `ir` holds to be
exactly a float32) as the literal with C's `f` suffix, `0.5f`; `sqrtf` and
`expf` for an F32 argument; and an F32 `Weighted`'s int coefficients as
`2.0f`, so no double literal widens the sum (C would, silently). The types
come from `ir.expr_type`, the rules' one statement. The IR has no fused
multiply-add, so there is no `fmaf` to spell: whether `a*b + c` contracts is
the compiler's (`-fmad` / `-ffp-contract`, untouched, devtools/config.sh),
and test/test_f32_device.py measures what that leaves between card and
interpreter. A kernel that declares no F32 takes none of these paths, so
its text is byte-identical to before #338 (the goldens and
`kernels.json`'s `source_sha256` are the check).

A `Boys` statement has no vendor intrinsic, so the writer emits one fixed
device template, `fj_boys<M>` (`BOYS_DEVICE`), when the kernel holds one, and
spells the statement `fj_boys<top>(T, dst);` -- the spelling of every
`F_m(T)`, m = 0..top, in the way `sqrt` is the spelling of `SQRT`. Whether
it is accurate enough is a numerical question, and numerical questions are
the kernel tier's (#28), against `reference.boys`, which this module does
not import. Every kernel carries the same text (#301): the narrow and wide
spellings of #266, and `Kernel.wide_boys` that picked between them, are gone.
`emit/gpu.py` carries none: it writes the routine once, as the header
`eval_boys.h`, and its units include it and call `eval_boys` (`CWriter.BOYS`);
there the series loop is a fixed polynomial per order (gpu.py, "The Boys
series"), the rest of the text this one.

A `Rys` statement likewise: the writer emits `fj_rys<N>` and its constant
tables (`felizjueves.rys.roots.device_text`, the host rule's own operation
order and constants) when the kernel holds one, and spells the statement
`fj_rys<n>(T, nodes, weights);`. `emit/gpu.py` does not carry it yet and
refuses a kernel with one (`CWriter.RYS` is None there).

Names starting `fj_` are the writer's own and are refused in a kernel.
"""

from __future__ import annotations

import re
import textwrap

from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Barrier,
    Boys,
    Break,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
    For,
    If,
    Index,
    Inline,
    Intent,
    Kernel,
    LaneBroadcast,
    Lit,
    Member,
    Note,
    Param,
    Raw,
    Record,
    Rys,
    Scalar,
    Space,
    Stmt,
    Var,
    WarpReduce,
    Weighted,
    _exprs_of,
    _subexprs,
    expr_type,
    walk,
)
from felizjueves.performance.device import Arch
from felizjueves.rys.roots import device_text

INDENT = "    "
NOTE_WIDTH = 76
RESERVED = "fj_"

CTYPE = {Scalar.F64: "double", Scalar.F32: "float", Scalar.I32: "int"}
_CAST = {Fn.TO_F32: "float", Fn.TO_F64: "double"}

_BINARY = {Fn.ADD: "+", Fn.SUB: "-", Fn.MUL: "*", Fn.DIV: "/", Fn.LT: "<"}
_GEOMETRY = {
    Fn.THREAD_X: "threadIdx.x",
    Fn.THREAD_Y: "threadIdx.y",
    Fn.BLOCK_X: "blockIdx.x",
    Fn.BLOCK_DIM_X: "blockDim.x",
    Fn.BLOCK_DIM_Y: "blockDim.y",
}

# `fj_boys<M>(T, F)` fills `F[m] = F_m(T)` for m = 0..M (#301): every order
# a `Boys` statement asks for, from one exp(-T). Two branches, one switch:
#
# - From T = M + BOYS_SWITCH_PAD, F_0 = (1/2) sqrt(pi/T) erf(sqrt(T)) and the
#   upward recurrence F_(m+1) = ((2m+1) F_m - exp(-T)) / (2T). Both are exact
#   (unlike the bare asymptote, there is no truncation Q(m+1/2, T) to wait
#   out), so the switch is set by the subtraction alone, which cancels only
#   where T is near m: from T = m + 8 it costs nothing measurable.
# - Below it, the Kummer series for F_M alone, every term positive, summed
#   until it no longer changes the sum, then the downward recurrence
#   F_m = (2T F_(m+1) + exp(-T)) / (2m+1), stable for every T. The series
#   runs at most ~75 terms (M = 40), where the narrow routine it replaces
#   ran ~2T + 40 per order and overflowed near T = 709.
#
# M is a template argument, the kernel's top order, fixed when the kernel is
# built: both recurrences unroll, each 1/(2m+1) is a constant, and F is a
# local array the compiler keeps in registers once it is inlined. Measured in
# float64 against mpmath's incomplete gamma over T in [0, 1e6] before it was
# written (#301): worst (M + 3) ulp, the upward recurrence's accumulation.
# Its accuracy on the card is the kernel tier's question (test_kernel_device.py).
# The pad and pi are spelled as literals below; test_emit_cuda.py holds them
# to BOYS_SWITCH_PAD and repr(math.pi).
BOYS_SWITCH_PAD = 8
BOYS_DEVICE = """\
// F_m(T) = integral_0^1 u^(2m) exp(-T u^2) du ([JEngine, v2, eqs. 13, 18]),
// F[m] for every m in 0..M (#301). From T = M + 8: F_0 = (1/2) sqrt(pi/T)
// erf(sqrt(T)) and upward, F_(m+1) = ((2m+1) F_m - exp(-T)) / (2T). Below:
// the Kummer series F_M = exp(-T) SUM_k (2T)^k / ((2M+1)(2M+3)...(2M+2k+1)),
// every term positive, summed until it no longer changes the sum, and
// downward, F_m = (2T F_(m+1) + exp(-T)) / (2m+1). Its accuracy on
// reference.boys's domain is a kernel-tier question, not this file's.
template <int M>
__device__ __forceinline__ void fj_boys(double T, double (&F)[M + 1]) {
    const double e = exp(-T);
    if (T >= (double)(M + 8)) {
        const double h = 0.5 / T;
        double f = 0.5 * sqrt(3.141592653589793 / T) * erf(sqrt(T));
        F[0] = f;
#pragma unroll
        for (int m = 1; m <= M; ++m) {
            f = ((double)(2 * m - 1) * f - e) * h;
            F[m] = f;
        }
        return;
    }
    double term = 1.0 / (double)(2 * M + 1);
    double sum = term;
    for (int k = 1; k < 4096; ++k) {
        term = term * ((2.0 * T) / (double)(2 * M + 2 * k + 1));
        double next = sum + term;
        if (next == sum) {
            break;
        }
        sum = next;
    }
    double f = e * sum;
    F[M] = f;
#pragma unroll
    for (int m = M - 1; m >= 0; --m) {
        f = (2.0 * T * f + e) * (1.0 / (double)(2 * m + 1));
        F[m] = f;
    }
}
"""


class EmitError(ValueError):
    """The kernel cannot be spelled without the writer deciding something."""


class _Array:
    """What the writer needs to index an array: its extents and its layout."""

    __slots__ = ("flat", "shape")

    def __init__(self, shape: tuple[Expr, ...], flat: bool) -> None:
        self.shape = shape
        self.flat = flat


class CWriter:
    """Spell a `Kernel` as device C++; subclass per vendor for the hooks."""

    # The spelling of `Scalar.I32` -- declarations, loop counters, the
    # geometry casts and `fj_boys`'s integers. A backend may name the width
    # (`emit/gpu.py`: `int32_t`); the vendor backends keep `int`, so their
    # text, goldens and `source_sha256` pins do not move.
    INT = "int"

    # The name a `Boys` statement calls. The vendor backends carry the
    # routine in each unit under the writer's own prefix; `emit/gpu.py`
    # includes it from a header of its own and names it there.
    BOYS = "fj_boys"

    # The name a `Rys` statement calls; None where the backend does not
    # carry the routine (`emit/gpu.py`), which refuses the statement.
    RYS: str | None = "fj_rys"

    def __init__(self, arch: Arch) -> None:
        if not isinstance(arch, Arch):
            raise TypeError(f"emit takes a device.Arch, not {arch!r}")
        self.arch = arch

    # --- the translation unit ---------------------------------------------

    def kernel(self, k: Kernel) -> str:
        if not isinstance(k, Kernel):
            raise TypeError(f"emit takes a Kernel, not {k!r}")
        self._check_names(k)
        self._scopes: list[dict[str, _Array | None]] = [{}]
        self._types: list[dict[str, Scalar]] = [{}]
        self._lines: list[str] = []
        self._dynamic_shared = 0
        self._used: set[str] = set()
        for p in k.params:
            self._bind(p.name, _Array(p.shape, True) if p.array else None,
                       p.type)
        sig = ",\n".join(INDENT + self.param(p) for p in k.params)
        self._block(k.body, 1)
        unread = [p.name for p in k.params
                  if not p.array and p.name not in self._used]
        unread_arrays = [p.name for p in k.params
                         if p.array and p.name not in self._used]
        top: list[str] = []
        if unread:
            top += [
                f"{INDENT}// an extent only (a leading dimension, never read):\n",
                *(f"{INDENT}(void){n};\n" for n in unread)]
        if unread_arrays:  # #407: a fixed signature this class does not read
            top += [
                f"{INDENT}// an array this class never reads (one signature "
                "for every class):\n",
                *(f"{INDENT}(void){n};\n" for n in unread_arrays)]
        self._lines[:0] = top
        head = self.prologue(k)
        boys = self.boys(BOYS_DEVICE) if has_boys(k) else ""
        if boys:
            head += "\n" + boys
        if has_rys(k):
            head += "\n" + self.spell_int(device_text(prefix=self.RYS))
        records = _records(k)
        if records:
            head += "\n" + "".join(record_struct(r, self.ctype(r.scalar))
                                    for r in records)
        return (
            f"{head}\n"
            f"{self.signature(k, sig)}\n"
            "{\n" + "".join(self._lines) + "}\n"
        )

    def signature(self, k: Kernel, sig: str) -> str:
        """The text that opens the kernel, `sig` its spelled parameters."""
        return f'extern "C" __global__ void {k.name}(\n{sig})'

    def boys(self, text: str) -> str:
        """`fj_boys` as this unit carries it, `text` its fixed C; empty
        when the unit includes it instead."""
        return self.spell_int(text)

    def ctype(self, t: Scalar) -> str:
        return self.INT if t is Scalar.I32 else CTYPE[t]

    def spell_int(self, text: str) -> str:
        """`text` (fixed C, such as `fj_boys`) with `int` spelled `INT`."""
        return text if self.INT == "int" else re.sub(r"\bint\b", self.INT, text)

    def param(self, p: Param) -> str:
        t = self.ctype(p.type)
        if not p.array:
            return f"{t} {p.name}"
        const = "const " if p.intent is Intent.IN else ""
        return f"{const}{t}* {p.name}"

    # --- statements ---------------------------------------------------------

    def _emit(self, depth: int, text: str) -> None:
        self._lines.append(f"{INDENT * depth}{text}\n")

    def _block(self, body: tuple[Stmt, ...], depth: int) -> None:
        self._scopes.append({})
        self._types.append({})
        try:
            for s in body:
                self.stmt(s, depth)
        finally:
            self._scopes.pop()
            self._types.pop()

    def stmt(self, s: Stmt, depth: int) -> None:
        if isinstance(s, Note):
            for para in s.prov.splitlines() or [""]:
                for line in textwrap.wrap(para, NOTE_WIDTH) or [""]:
                    if line.endswith("\\"):
                        raise EmitError("a Note line ending in '\\' would "
                                        "continue the comment")
                    self._emit(depth, f"// {line}".rstrip())
        elif isinstance(s, Decl):
            self.decl(s, depth)
        elif isinstance(s, Assign):
            self._emit(depth, f"{self.lvalue(s.dst)} = {self.expr(s.value)};")
        elif isinstance(s, For):
            v = s.var
            self._emit(depth, f"for ({self.INT} {v} = {self.expr(s.start)}; "
                              f"{v} < {self.expr(s.stop)}; "
                              f"{v} += {self.expr(s.step)}) {{")
            self._scopes.append({v: None})
            self._types.append({v: Scalar.I32})
            try:
                self._block(s.body, depth + 1)
            finally:
                self._scopes.pop()
                self._types.pop()
            self._emit(depth, "}")
        elif isinstance(s, If):
            cond = self.expr(s.cond)
            if not (isinstance(s.cond, Call) and s.cond.fn in _BINARY):
                cond = f"({cond})"  # a binary Call is already parenthesised
            self._emit(depth, f"if {cond} {{")
            self._block(s.then, depth + 1)
            if s.orelse:
                self._emit(depth, "} else {")
                self._block(s.orelse, depth + 1)
            self._emit(depth, "}")
        elif isinstance(s, Break):
            self._emit(depth, "break;")
        elif isinstance(s, Barrier):
            self._emit(depth, "__syncthreads();")  # both vendors spell it so
        elif isinstance(s, WarpReduce):
            for line in self.warp_reduce(self.lvalue(s.dst), self.expr(s.value)):
                self._emit(depth, line)
        elif isinstance(s, LaneBroadcast):
            self._lane_broadcast(s, depth)
        elif isinstance(s, AtomicAdd):
            self._emit(depth, self.atomic_add(self.lvalue(s.dst),
                                              self.expr(s.value)))
        elif isinstance(s, Boys):
            if self._type(s.t) is not Scalar.F64 or self._lookup(s.dst) is None:
                raise EmitError(f"Boys into {s.dst!r}: an F64 T and a local "
                                f"array, or no {self.BOYS}")
            self._emit(depth,
                       f"{self.BOYS}<{s.top}>({self.expr(s.t)}, {s.dst});")
        elif isinstance(s, Rys):
            if self.RYS is None:
                raise EmitError(f"Rys into {s.nodes!r}: this backend does not "
                                "carry the device Rys rule")
            if (self._type(s.t) is not Scalar.F64
                    or self._lookup(s.nodes) is None
                    or self._lookup(s.weights) is None):
                raise EmitError(f"Rys into {s.nodes!r}: an F64 T and two "
                                f"local arrays, or no {self.RYS}")
            self._emit(depth, f"{self.RYS}<{s.n}>({self.expr(s.t)}, "
                       f"{s.nodes}, {s.weights});")
        elif isinstance(s, Inline):
            self.inline(s, depth)
        elif isinstance(s, Raw):
            for line in s.text.splitlines():
                self._emit(depth, line)
        else:  # pragma: no cover
            raise EmitError(f"not a Stmt: {s!r}")

    def _lane_broadcast(self, s: LaneBroadcast, depth: int) -> None:
        """#349's indexed shuffle, through the vendor hook."""
        self._emit(depth, self.lane_broadcast(
            self.lvalue(s.dst), self.expr(s.value), self.expr(s.source)))

    def inline(self, s: Inline, depth: int) -> None:
        """The fragment's statements in place, in a block of their own.

        Each binding is a `const` declaration ahead of the body: a scalar
        bind `const auto b = <expr>;`, an array bind `auto* const b = a;`
        (the host array, decayed to a pointer that indexes as it does). A
        bind bound to the host name it shares (`Var(b)`, which `Kernel`
        admits and nothing else) is the host's own `b`, so it gets no line.
        """
        frag = s.fragment
        self._emit(depth, "{")
        self._emit(depth + 1, f"// fragment {frag.name}")
        scope: dict[str, _Array | None] = {}
        types: dict[str, Scalar] = {}
        for name, e in s.bindings:
            t = self._type(e)
            if t is not None:
                types[name] = t
            if e == Var(name):
                self._used.add(name)
                scope[name] = self._lookup(name)
            elif name in frag.array_binds:
                self._used.add(e.name)
                arr = self._lookup(e.name)
                if arr is None:
                    raise EmitError(f"array bind {name!r} is bound to the "
                                    f"scalar {e.name!r}")
                scope[name] = arr
                self._emit(depth + 1, f"auto* const {name} = {e.name};")
            else:
                scope[name] = None
                self._emit(depth + 1, f"const auto {name} = {self.expr(e)};")
        self._scopes.append(scope)
        self._types.append(types)
        try:
            self._block(frag.body, depth + 1)
        finally:
            self._scopes.pop()
            self._types.pop()
        self._emit(depth, "}")

    def decl(self, s: Decl, depth: int) -> None:
        t = self.ctype(s.type)
        if not s.shape:
            if s.space is Space.SHARED:
                if s.init is not None:
                    raise EmitError(f"shared scalar {s.name!r} cannot be "
                                    "initialised in its declaration")
                self._bind(s.name, None, s.type)
                self._emit(depth, f"__shared__ {t} {s.name};")
                return
            init = "" if s.init is None else f" = {self.expr(s.init)}"
            self._bind(s.name, None, s.type)
            self._emit(depth, f"{t} {s.name}{init};")
            return
        static = all(isinstance(d, Lit) for d in s.shape)
        if static:
            dims = "".join(f"[{self.expr(d)}]" for d in s.shape)
            space = "__shared__ " if s.space is Space.SHARED else ""
            self._bind(s.name, _Array(s.shape, False), s.type)
            self._emit(depth, f"{space}{t} {s.name}{dims};")
            return
        if s.space is not Space.SHARED:
            raise EmitError(f"local array {s.name!r} has a launch-time extent")
        self._dynamic_shared += 1
        if self._dynamic_shared > 1:
            raise EmitError("more than one dynamically sized SHARED array: "
                            "laying them out would be a decision")
        self._bind(s.name, _Array(s.shape, True), s.type)
        dims = " x ".join(self.expr(d) for d in s.shape)
        self._emit(depth, f"// {s.name}: dynamic shared memory, row-major, "
                          f"{dims} {t}s")
        self._emit(depth, f"extern __shared__ {t} {s.name}[];")

    # --- expressions --------------------------------------------------------

    def lvalue(self, e: Var | Index) -> str:
        if isinstance(e, Member):
            raise EmitError(f"{e.array}: a Member is a read through a const "
                            "typed view, never a store target")
        return self.expr(e)

    def expr(self, e: Expr) -> str:
        if isinstance(e, Lit):
            return self.lit(e)
        if isinstance(e, Var):
            self._used.add(e.name)
            return e.name
        if isinstance(e, Index):
            return self.index(e)
        if isinstance(e, Weighted):
            return self.weighted(e)
        if isinstance(e, Call):
            return self.call(e)
        raise EmitError(f"not an Expr: {e!r}")  # pragma: no cover

    def lit(self, e: Lit) -> str:
        v = e.value
        if type(v) is int:
            text = str(v)
        else:
            if v != v or v in (float("inf"), float("-inf")):
                raise EmitError(f"literal {v!r} has no C spelling")
            text = repr(v)  # round-trips, and always carries '.' or 'e'
        return f"({text})" if text.startswith("-") else text

    def index(self, e: Index) -> str:
        arr = self._lookup(e.array)
        if arr is None:
            raise EmitError(f"{e.array!r} is a scalar, indexed as an array")
        self._used.add(e.array)
        if isinstance(e, Member):
            return self.member(e, arr)
        at = [self.expr(i) for i in e.at]
        if not arr.flat:
            return e.array + "".join(f"[{i}]" for i in at)
        if len(at) == 1:
            return f"{e.array}[{at[0]}]"
        if len(arr.shape) != len(at):
            raise EmitError(
                f"{e.array}: {len(at)} subscripts but a declared shape of "
                f"{len(arr.shape)} -- a flat array is linearised by its shape")
        flat = at[0]
        for d, i in zip(arr.shape[1:], at[1:], strict=True):
            flat = f"({flat} * {self.expr(d)} + {i})"
        return f"{e.array}[{flat}]"

    def member(self, e: Member, arr: _Array) -> str:
        """`reinterpret_cast<const fj_<rec>*>(array)[row].<member>` (#353):
        the row is the leading subscripts linearised row-major by the
        leading extents, as `index` linearises the whole; the last extent
        must be the record's width and the array's type its scalar, so the
        view is the layout the flat array already has."""
        rec = e.record
        width = arr.shape[-1] if arr.shape else None
        if len(arr.shape) != len(e.at) or not (
                isinstance(width, Lit) and width.value == len(rec.members)):
            raise EmitError(
                f"{e.array}: a {rec.name} view needs a declared shape whose "
                f"last extent is the literal {len(rec.members)}, one subscript "
                f"per extent; the shape is {arr.shape}")
        if self._type_of(e.array) is not rec.scalar:
            raise EmitError(f"{e.array} is {self._type_of(e.array)}, its "
                            f"{rec.name} view {rec.scalar}")
        at = [self.expr(i) for i in e.at[:-1]]
        row = at[0]
        for d, i in zip(arr.shape[1:-1], at[1:], strict=True):
            row = f"({row} * {self.expr(d)} + {i})"
        return (f"reinterpret_cast<const {RESERVED}{rec.name}*>({e.array})"
                f"[{row}].{e.member}")

    def weighted(self, e: Weighted) -> str:
        parts = []
        suffix = "f" if self._type(e) is Scalar.F32 else ""
        for t in e.terms:
            src = self.expr(t.src)
            factor = None if t.g is Factor.ONE else t.g.name_in_kernel
            coeff = None if t.c == 1 else repr(float(t.c)) + suffix
            if coeff and factor:
                parts.append(f"(({coeff} * {factor}) * {src})")
            elif coeff or factor:
                parts.append(f"({coeff or factor} * {src})")
            else:
                parts.append(src)
        out = parts[0]
        for p in parts[1:]:
            out = f"({out} + {p})"
        return out

    def call(self, e: Call) -> str:
        fn = e.fn
        if fn in _GEOMETRY:
            return f"({self.INT}){_GEOMETRY[fn]}"
        if fn is Fn.TO_F32 and isinstance(e.args[0], Lit) \
                and e.args[0].type is Scalar.F64:
            text = self.lit(e.args[0])  # ir: exactly a float32
            return f"({text[1:-1]}f)" if text.startswith("(") else f"{text}f"
        args = [self.expr(a) for a in e.args]
        if fn in _BINARY:
            return f"({args[0]} {_BINARY[fn]} {args[1]})"
        if fn in _CAST:
            return self.cast(_CAST[fn], args[0])
        f = "f" if fn in (Fn.SQRT, Fn.EXP) and \
            self._type(e.args[0]) is Scalar.F32 else ""
        if fn is Fn.SQRT:
            return f"sqrt{f}({args[0]})"
        if fn is Fn.EXP:
            return f"exp{f}({args[0]})"
        raise EmitError(f"no spelling for {fn}")  # pragma: no cover

    def cast(self, t: str, text: str) -> str:
        """`Fn.TO_F32` / `Fn.TO_F64` of `text`, already spelled."""
        return f"(({t}){text})"

    # --- names --------------------------------------------------------------

    def _bind(self, name: str, arr: _Array | None, type_: Scalar) -> None:
        self._scopes[-1][name] = arr
        self._types[-1][name] = type_

    def _type_of(self, name: str) -> Scalar | None:
        for scope in reversed(self._types):
            if name in scope:
                return scope[name]
        return None

    def _type(self, e: Expr) -> Scalar | None:
        return expr_type(e, self._type_of)

    def _lookup(self, name: str) -> _Array | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        raise EmitError(f"{name!r} is not bound")

    @staticmethod
    def _check_names(k: Kernel) -> None:
        bad = sorted({n for n in kernel_names(k) if n.startswith(RESERVED)})
        if bad:
            raise EmitError(f"names {bad} use the writer's {RESERVED!r} prefix")

    # --- the vendor hooks ---------------------------------------------------

    def prologue(self, k: Kernel) -> str:
        """The text before the kernel: a header comment and any includes."""
        raise NotImplementedError

    def warp_reduce(self, dst: str, value: str) -> list[str]:
        """Lines that set `dst` to the warp-wide sum of `value` on every lane."""
        raise NotImplementedError

    def atomic_add(self, dst: str, value: str) -> str:
        return f"atomicAdd(&{dst}, {value});"

    def lane_broadcast(self, dst: str, value: str, lane: str) -> str:
        """The line that sets `dst` to `value` as held by lane `lane` (#349)."""
        raise NotImplementedError


def kernel_names(k: Kernel) -> list[str]:
    """Every name `k` declares: itself, its parameters, its locals, its loop
    counters and what its fragments bind."""
    names = [k.name, *(p.name for p in k.params)]
    for s in walk(k.body):
        if isinstance(s, Decl):
            names.append(s.name)
        elif isinstance(s, For):
            names.append(s.var)
        elif isinstance(s, Inline):
            names.extend(s.fragment.binds)
    return names


def has_rys(k: Kernel) -> bool:
    """Whether `k` holds a `Rys` statement, so its unit needs the rule."""
    return any(isinstance(s, Rys) for s in walk(k.body))


def has_boys(k: Kernel) -> bool:
    """Whether `k` holds a `Boys` statement, so its unit needs the routine."""
    return any(isinstance(s, Boys) for s in walk(k.body))


def precision(k: Kernel) -> str:
    """The prologue's word on precision: what the kernel declares (#338)."""
    declared = {p.type for p in k.params} | {
        s.type for s in walk(k.body) if isinstance(s, Decl)}
    if Scalar.F32 in declared:
        return "fp32 and fp64 as declared"
    return "fp64 throughout"


def record_struct(rec: Record, t: str | None = None) -> str:
    """The C struct of one `Record` view (#353), and the size check that
    makes the view's layout the flat row's: one scalar type, so a struct of
    exactly `len(members)` of them has no padding and member `i` at element
    `i` (declaration order is layout order for a standard-layout struct).
    Natural alignment only: the view asks no more of a base pointer than
    the flat array does."""
    t, name = t or CTYPE[rec.scalar], RESERVED + rec.name
    members = "".join(f"{INDENT}{t} {m};\n" for m in rec.members)
    return (
        f"// {name}: a typed view of one packed row of {len(rec.members)} "
        f"{t}s (#353)\n"
        f"struct {name} {{\n{members}}};\n"
        f"static_assert(sizeof({name}) == {len(rec.members)} * sizeof({t}), "
        f'"{name} must be the packed row, unpadded");\n')


def _records(k: Kernel) -> tuple[Record, ...]:
    """Every `Record` a `Member` of `k` reads through, in first-use order;
    two records of one name must be one record."""
    out: dict[str, Record] = {}
    for e in _exprs(k):
        if isinstance(e, Member):
            seen = out.setdefault(e.record.name, e.record)
            if seen != e.record:
                raise EmitError(f"two Records named {e.record.name!r}: "
                                f"{seen} and {e.record}")
    return tuple(out.values())


def _exprs(k: Kernel):
    """Every expression in the kernel body, nested ones included."""
    for s in walk(k.body):
        for root in _exprs_of(s):
            if root is not None:
                yield from _subexprs(root)
