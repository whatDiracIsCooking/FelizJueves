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
- `Note` is a `//` comment beside the statement it precedes;
- an `Inline` is a `{ ... }` block headed `// fragment <name>`, its bindings
  `const` declarations, then the fragment's statements in place. Nothing is
  renamed: `Kernel` has already refused any name the block could shadow.

`Fn.BOYS` has no vendor intrinsic, so the writer emits one fixed device
routine, `fj_boys` (`BOYS_DEVICE`), when the kernel calls it -- the spelling
of `F_m(T)` in the way `sqrt` is the spelling of `SQRT`. Whether it is
accurate enough is a numerical question, and numerical questions are the
kernel tier's (#28), against `reference.boys`, which this module does not
import.

Names starting `fj_` are the writer's own and are refused in a kernel.
"""

from __future__ import annotations

import textwrap

from felizjueves.device import Arch
from felizjueves.ir import (
    Assign,
    AtomicAdd,
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
    Lit,
    Note,
    Param,
    Raw,
    Scalar,
    Space,
    Stmt,
    Var,
    WarpReduce,
    Weighted,
    _exprs_of,
    _subexprs,
    walk,
)

INDENT = "    "
NOTE_WIDTH = 76
RESERVED = "fj_"

CTYPE = {Scalar.F64: "double", Scalar.I32: "int"}

_BINARY = {Fn.ADD: "+", Fn.SUB: "-", Fn.MUL: "*", Fn.DIV: "/", Fn.LT: "<"}
_GEOMETRY = {
    Fn.THREAD_X: "(int)threadIdx.x",
    Fn.THREAD_Y: "(int)threadIdx.y",
    Fn.BLOCK_X: "(int)blockIdx.x",
    Fn.BLOCK_DIM_X: "(int)blockDim.x",
    Fn.BLOCK_DIM_Y: "(int)blockDim.y",
}

BOYS_DEVICE = """\
// F_m(T) = exp(-T) SUM_k (2T)^k / ((2m+1)(2m+3)...(2m+2k+1)), the Kummer
// series ([JEngine, v2, eqs. 13, 18]): every term positive, summed until it
// no longer changes the sum. Its accuracy on reference.boys's domain is a
// kernel-tier question, not this file's.
__device__ double fj_boys(int m, double T) {
    double term = 1.0 / (double)(2 * m + 1);
    double sum = term;
    for (int k = 1; k < 4096; ++k) {
        term = term * ((2.0 * T) / (double)(2 * m + 2 * k + 1));
        double next = sum + term;
        if (next == sum) {
            break;
        }
        sum = next;
    }
    return exp(-T) * sum;
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
        self._lines: list[str] = []
        self._dynamic_shared = 0
        for p in k.params:
            self._bind(p.name, _Array(p.shape, True) if p.array else None)
        sig = ",\n".join(INDENT + self.param(p) for p in k.params)
        self._block(k.body, 1)
        head = self.prologue(k)
        if any(isinstance(e, Call) and e.fn is Fn.BOYS for e in _exprs(k)):
            head += "\n" + BOYS_DEVICE
        return (
            f"{head}\n"
            f'extern "C" __global__ void {k.name}(\n{sig})\n'
            "{\n" + "".join(self._lines) + "}\n"
        )

    def param(self, p: Param) -> str:
        t = CTYPE[p.type]
        if not p.array:
            return f"{t} {p.name}"
        const = "const " if p.intent is Intent.IN else ""
        return f"{const}{t}* {p.name}"

    # --- statements ---------------------------------------------------------

    def _emit(self, depth: int, text: str) -> None:
        self._lines.append(f"{INDENT * depth}{text}\n")

    def _block(self, body: tuple[Stmt, ...], depth: int) -> None:
        self._scopes.append({})
        try:
            for s in body:
                self.stmt(s, depth)
        finally:
            self._scopes.pop()

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
            self._emit(depth, f"for (int {v} = {self.expr(s.start)}; "
                              f"{v} < {self.expr(s.stop)}; "
                              f"{v} += {self.expr(s.step)}) {{")
            self._scopes.append({v: None})
            try:
                self._block(s.body, depth + 1)
            finally:
                self._scopes.pop()
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
        elif isinstance(s, WarpReduce):
            for line in self.warp_reduce(self.lvalue(s.dst), self.expr(s.value)):
                self._emit(depth, line)
        elif isinstance(s, AtomicAdd):
            self._emit(depth, self.atomic_add(self.lvalue(s.dst),
                                              self.expr(s.value)))
        elif isinstance(s, Inline):
            self.inline(s, depth)
        elif isinstance(s, Raw):
            for line in s.text.splitlines():
                self._emit(depth, line)
        else:  # pragma: no cover
            raise EmitError(f"not a Stmt: {s!r}")

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
        for name, e in s.bindings:
            if e == Var(name):
                scope[name] = self._lookup(name)
            elif name in frag.array_binds:
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
        try:
            self._block(frag.body, depth + 1)
        finally:
            self._scopes.pop()
        self._emit(depth, "}")

    def decl(self, s: Decl, depth: int) -> None:
        t = CTYPE[s.type]
        if not s.shape:
            if s.space is Space.SHARED:
                if s.init is not None:
                    raise EmitError(f"shared scalar {s.name!r} cannot be "
                                    "initialised in its declaration")
                self._bind(s.name, None)
                self._emit(depth, f"__shared__ {t} {s.name};")
                return
            init = "" if s.init is None else f" = {self.expr(s.init)}"
            self._bind(s.name, None)
            self._emit(depth, f"{t} {s.name}{init};")
            return
        static = all(isinstance(d, Lit) for d in s.shape)
        if static:
            dims = "".join(f"[{self.expr(d)}]" for d in s.shape)
            space = "__shared__ " if s.space is Space.SHARED else ""
            self._bind(s.name, _Array(s.shape, False))
            self._emit(depth, f"{space}{t} {s.name}{dims};")
            return
        if s.space is not Space.SHARED:
            raise EmitError(f"local array {s.name!r} has a launch-time extent")
        self._dynamic_shared += 1
        if self._dynamic_shared > 1:
            raise EmitError("more than one dynamically sized SHARED array: "
                            "laying them out would be a decision")
        self._bind(s.name, _Array(s.shape, True))
        dims = " x ".join(self.expr(d) for d in s.shape)
        self._emit(depth, f"// {s.name}: dynamic shared memory, row-major, "
                          f"{dims} {t}s")
        self._emit(depth, f"extern __shared__ {t} {s.name}[];")

    # --- expressions --------------------------------------------------------

    def lvalue(self, e: Var | Index) -> str:
        return self.expr(e)

    def expr(self, e: Expr) -> str:
        if isinstance(e, Lit):
            return self.lit(e)
        if isinstance(e, Var):
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

    def weighted(self, e: Weighted) -> str:
        parts = []
        for t in e.terms:
            src = self.expr(t.src)
            factor = None if t.g is Factor.ONE else t.g.name_in_kernel
            coeff = None if t.c == 1 else repr(float(t.c))
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
            return _GEOMETRY[fn]
        args = [self.expr(a) for a in e.args]
        if fn in _BINARY:
            return f"({args[0]} {_BINARY[fn]} {args[1]})"
        if fn is Fn.SQRT:
            return f"sqrt({args[0]})"
        if fn is Fn.EXP:
            return f"exp({args[0]})"
        if fn is Fn.BOYS:
            return f"fj_boys({args[0]}, {args[1]})"
        raise EmitError(f"no spelling for {fn}")  # pragma: no cover

    # --- names --------------------------------------------------------------

    def _bind(self, name: str, arr: _Array | None) -> None:
        self._scopes[-1][name] = arr

    def _lookup(self, name: str) -> _Array | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        raise EmitError(f"{name!r} is not bound")

    @staticmethod
    def _check_names(k: Kernel) -> None:
        names = [k.name, *(p.name for p in k.params)]
        for s in walk(k.body):
            if isinstance(s, Decl):
                names.append(s.name)
            elif isinstance(s, For):
                names.append(s.var)
            elif isinstance(s, Inline):
                names.extend(s.fragment.binds)
        bad = sorted({n for n in names if n.startswith(RESERVED)})
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


def _exprs(k: Kernel):
    """Every expression in the kernel body, nested ones included."""
    for s in walk(k.body):
        for root in _exprs_of(s):
            if root is not None:
                yield from _subexprs(root)
