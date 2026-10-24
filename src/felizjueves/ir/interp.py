"""Execute a `Kernel` on the host: the compiler-free answer to "is it right?".

    interpret(kernel, **inputs) -> {out_param: ndarray}

An instruction set machine for ir/__init__.py and nothing more. It executes Decl,
Assign, For, If, Break, WarpReduce, LaneBroadcast, Barrier, AtomicAdd and
Inline; it skips Note; it
refuses Raw (opaque by definition). **It does not know what McMurchie-Davidson is**:
`Weighted` is `SUM_k c_k * g_k * src_k` in fp64, a `Factor` is the scalar of
that name, and no recurrence, index set or integral appears here. The moment
one did, it would share ancestry with the builder it checks and the pins in
test/test_interp.py would be worth nothing.

INTERPRETING IS NOT EMITTING. A kernel that interprets correctly can still
emit text that does not compile, or compiles to something slow; that is the
golden files' and the kernel tier's business. A green interpreter is never a
verified kernel.

THE MACHINE.

- A launch is `Launch(grid, block, lanes)`: `grid` blocks along x, `block =
  (dim_x, dim_y)` threads, and the emulated warp width `lanes` (a power of
  two). The IR deliberately does not know the width (ir/__init__.py); the device target
  does, so it is a launch parameter here.
- Threads are linearised x-fastest within a block (`x + y * dim_x`), as on
  both vendors' hardware, and cut into warps of `lanes` consecutive threads.
- Each thread runs sequentially as a coroutine; a warp's lanes are advanced to
  their next `WarpReduce` one after another, and the reduction is applied when
  every lane has arrived **at the same statement**. A lane that has finished,
  or is waiting at a different one, is divergence and raises, as does a
  reduction in a partial warp: on the device both are undefined behaviour,
  not a smaller sum.
- The reduction is the xor butterfly a shuffle reduction performs
  (`v[i] += v[i ^ o]` for o = lanes/2 .. 1), so every lane sees the same sum,
  rounded as the device would round it.
- A `LaneBroadcast` (#349) is the other warp collective, under the same
  arrival rule: lane `i` receives the `value` lane `src_i` evaluated, where
  `src_i` is lane `i`'s own `source`. A source lane outside `[0, lanes)`
  raises (the hardware would wrap it modulo the width; a builder that relies
  on that is wrong). The value moves unchanged, so no rounding happens.
- Warps of a block run one after another, sharing the block's SHARED arrays,
  each to completion or to its next `Barrier` (#351). At a barrier the block
  waits until every warp has arrived -- every thread at the SAME `Barrier`,
  else it raises (a finished thread, or one at another statement, is
  undefined behaviour on the device) -- and then the warps resume, again one
  after another. A kernel with no `Barrier` therefore runs in exactly the
  order it ran before #351. Shared memory is sequentially consistent; a race
  that a missing barrier would leave (a read of a row another warp has not
  yet written) shows as a read of a never-written element, which raises.

STRICTNESS, because a lax machine passes broken kernels:

- Array `Param`s must be passed as ndarrays of the declared dtype (float64 for
  F64, int32 for I32); IN arrays are read-only, OUT arrays are copied and the
  copies returned. Scalar `Param`s take a Python or 0-d numpy number.
- An array with a declared `Param.shape` must hold exactly prod(extents)
  elements, in any shape (flat is the contract: the backend linearises), and
  is indexed at the declared extents. An extent the bound I32 scalars and the
  launch cannot evaluate leaves that array unchecked, in
  `Interpreter.unchecked`.
- Every subscript is bounds-checked (no negative wrap); every read of a local
  or shared element, or of a scalar, that was never written raises.
- int is I32 and float is F64, as `Lit` says. Integer DIV truncates toward
  zero and overflow past 32 bits raises; storing a float into an I32 raises;
  an int stored into an F64 widens (exact). A `For` with step <= 0 raises.

F32 (#338) IS ROUNDED WHERE THE DEVICE ROUNDS IT, which is the whole of its
meaning: interpreted in fp64 and rounded only when it lands in a float32
array, an fp32 program would pass every test while computing something no
card computes. An F32 value is carried as a `F32` (a float subclass, so it
keeps its tag) holding a float32-exact value, and every operation that
produces one rounds to float32 -- each ADD SUB MUL DIV, SQRT and EXP, each
multiply and add of a `Weighted` term in the emitter's left-to-right order
(`(c * g) * src`, then the running sum), each `AtomicAdd`, each butterfly
step of a `WarpReduce`, each `TO_F32`, and each store into an F32 slot. That
is C's semantics on both device compilers (single-precision arithmetic is
evaluated in single precision, FLT_EVAL_METHOD 0), and it implies the
issue's "rounded at every assignment". An operation is computed exactly or
in fp64 and then rounded once: for + - * / and sqrt of float32 operands,
fp64 (53 bits >= 2 * 24 + 2) makes that double rounding innocuous, so the
result is the correctly rounded float32 the device produces. `exp` is not
correctly rounded on either device; the interpreter rounds the fp64 `exp`.
Where a card may still differ is FMA contraction (one rounding where the
interpreter has two) and `expf`'s last ulp: test/test_f32_device.py holds
the card to the interpreter bitwise where neither can occur, and to the
declared fp32 tolerance (test/oracle/tolerance.py) where they can.
`_round32` is the one rounding; the device test replaces it with the
identity to show the comparison fails when the rounding is absent.

F32 never meets F64 without a cast (`ir.expr_type`'s rules, which the
`Kernel` already checked); at run time the machine refuses it again rather
than guess. An I32 meeting an F32 converts to F32 (rounded), as in C.

`Boys` is the one special function. Its semantics, F_m(T), are INJECTED:
the caller passes `boys=` (the pins pass `reference.boys.boys`, quadrature),
each order of a `Boys` statement is one call of it, and a kernel that runs
a `Boys` without one raises. A kernel with a `Boys` is therefore checked on
everything *but* its Boys function: the writer's `fj_boys` (emit/_c.py) is
the kernel tier's, on the card. There is no
default, on purpose: a default would be an import of `reference/` from here,
and the interpreter must not share ancestry with the reference its own pins
compare against (test/test_layering.py).

FRAGMENTS. An `Inline` evaluates its bindings in the host's scope -- a scalar
bind is a read-only copy typed by its value, an array bind an alias of the
host array -- then runs the fragment's body with ONLY those names visible,
and drops its locals after. `interpret_fragment(frag, **bindings)` runs one
fragment alone, as a single thread with no kernel around it.

COUNTERS. `Interpreter.iterations[var]` is the number of `For` body entries
of the loop over `var`, summed over every thread of the launch. It is how a
test tells a screening `break` from a `continue`, which give the same answer.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    Fragment,
    If,
    Index,
    Inline,
    Intent,
    Kernel,
    LaneBroadcast,
    Lit,
    Note,
    Raw,
    Scalar,
    Space,
    Stmt,
    Var,
    WarpReduce,
    Weighted,
    is_f32,
)

_I32_MIN, _I32_MAX = -(2**31), 2**31 - 1
_DTYPE = {Scalar.F64: np.dtype(np.float64), Scalar.F32: np.dtype(np.float32),
          Scalar.I32: np.dtype(np.int32)}


class F32(float):
    """An F32 value: a float32-exact Python float that keeps its tag."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"F32({float(self)!r})"


def _round32(x: float) -> float:
    """Round an fp64 value to the nearest float32 (ties to even): THE rounding.

    The single seam the device test (#338) mutates to the identity, to show
    that the declared tolerance fails when this is absent.
    """
    return float(np.float32(x))


def _f32(x: int | float) -> F32:
    return F32(_round32(float(x)))


def _kind(v: Any, what: str) -> Scalar:
    t = type(v)
    if t is int:
        return Scalar.I32
    if t is F32:
        return Scalar.F32
    if t is float:
        return Scalar.F64
    raise InterpError(f"{what}: {v!r} is not a number")


def _mixed(what: str) -> InterpError:
    return InterpError(f"{what}: F32 meets F64 without TO_F32 / TO_F64")


def _arith(fn: Fn, a: int | float, b: int | float) -> int | float:
    """ADD SUB MUL DIV LT on two values, in the type C would compute them in."""
    ka, kb = _kind(a, fn.name), _kind(b, fn.name)
    if Scalar.F32 in (ka, kb):
        if Scalar.F64 in (ka, kb):
            raise _mixed(fn.name)
        x, y = float(_f32(a)), float(_f32(b))  # an I32 converts, rounded
        if fn is Fn.LT:
            return int(x < y)
        if fn is Fn.DIV and y == 0:
            raise InterpError("division by zero")
        op = {Fn.ADD: x + y, Fn.SUB: x - y, Fn.MUL: x * y}
        return _f32(op[fn] if fn in op else x / y)
    ints = ka is Scalar.I32 and kb is Scalar.I32
    if fn is Fn.LT:
        return int(a < b)
    if fn is Fn.ADD:
        return _check_i32(a + b) if ints else float(a) + float(b)
    if fn is Fn.SUB:
        return _check_i32(a - b) if ints else float(a) - float(b)
    if fn is Fn.MUL:
        return _check_i32(a * b) if ints else float(a) * float(b)
    if fn is Fn.DIV:
        if b == 0:
            raise InterpError("division by zero")
        if ints:
            q = abs(a) // abs(b)
            return _check_i32(q if (a < 0) == (b < 0) else -q)
        return float(a) / float(b)
    raise InterpError(f"no meaning for {fn}")  # pragma: no cover


class InterpError(RuntimeError):
    """The kernel did something the machine will not give a meaning to."""


@dataclass(frozen=True)
class Launch:
    """`grid` blocks along x, `block = (dim_x, dim_y)`, warp width `lanes`."""

    grid: int = 1
    block: tuple[int, int] = (1, 1)
    lanes: int = 32

    def __post_init__(self) -> None:
        if type(self.grid) is not int or self.grid < 1:
            raise ValueError(f"grid {self.grid!r} is not a positive int")
        if len(self.block) != 2 or any(type(b) is not int or b < 1
                                       for b in self.block):
            raise ValueError(f"block {self.block!r} is not two positive ints")
        if type(self.lanes) is not int or self.lanes < 1 or (
                self.lanes & (self.lanes - 1)):
            raise ValueError(f"lanes {self.lanes!r} is not a power of two")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class _Scalar:
    __slots__ = ("type", "value")

    def __init__(self, type_: Scalar, value: int | float | None = None) -> None:
        self.type = type_
        self.value = value


class _Array:
    __slots__ = ("type", "data", "written", "writable", "name")

    def __init__(self, name: str, type_: Scalar, data: np.ndarray,
                 written: np.ndarray | None, writable: bool) -> None:
        self.name = name
        self.type = type_
        self.data = data
        self.written = written  # None: every element is defined (a global)
        self.writable = writable


class _Break(Exception):
    pass


def _bound_scalar(v: Any, name: str) -> _Scalar:
    """A scalar bind's slot: typed by its value, as `Lit` types a literal."""
    v = _number(v)
    if type(v) is int:
        return _Scalar(Scalar.I32, _check_i32(v))
    if type(v) is F32:
        return _Scalar(Scalar.F32, v)
    if type(v) is float:
        return _Scalar(Scalar.F64, v)
    raise InterpError(f"bind {name!r}: {v!r} is not an int or float")


def _number(v: Any) -> Any:
    """A caller's number as the machine's: a numpy float32 is an `F32`."""
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v[()]
    if isinstance(v, np.float32):
        return F32(float(v))
    if isinstance(v, np.generic):
        return v.item()
    return v


def _check_i32(v: int) -> int:
    if not _I32_MIN <= v <= _I32_MAX:
        raise InterpError(f"I32 overflow: {v}")
    return v


def _coerce(type_: Scalar, v: Any, what: str) -> int | float:
    """Store `v` into a `type_` slot: int widens into F64 (exact) or F32
    (rounded); float never into I32; F32 and F64 never into each other."""
    k = _kind(v, what)
    if type_ is Scalar.F64:
        if k is Scalar.F32:
            raise _mixed(f"{what}: storing an F32 into an F64")
        return float(v)
    if type_ is Scalar.F32:
        if k is Scalar.F64:
            raise _mixed(f"{what}: storing an F64 into an F32")
        return _f32(v)  # rounded at every assignment
    if k is not Scalar.I32:
        raise InterpError(f"{what}: storing {type(v).__name__} {v!r} into an I32")
    return _check_i32(v)


# ---------------------------------------------------------------------------
# One thread
# ---------------------------------------------------------------------------


class _Thread:
    """One thread's execution of the body, as a coroutine.

    It yields `(stmt, value)` at a WarpReduce and is sent the warp's sum, and
    `(stmt, (value, source))` at a LaneBroadcast and is sent its source's value.
    """

    def __init__(self, interp: Interpreter, globals_: dict[str, Any],
                 shared: dict[str, _Array], geometry: dict[Fn, int]) -> None:
        self.interp = interp
        self.scopes: list[dict[str, Any]] = [globals_]
        self.shared = shared
        self.geometry = geometry

    # --- names ------------------------------------------------------------

    def lookup(self, name: str) -> Any:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise InterpError(f"{name!r} is not bound")

    def scalar(self, name: str) -> _Scalar:
        slot = self.lookup(name)
        if not isinstance(slot, _Scalar):
            raise InterpError(f"{name!r} is an array, used as a scalar")
        return slot

    def array(self, name: str) -> _Array:
        slot = self.lookup(name)
        if not isinstance(slot, _Array):
            raise InterpError(f"{name!r} is a scalar, indexed as an array")
        return slot

    def declare(self, name: str, slot: Any) -> None:
        scope = self.scopes[-1]
        if name in scope:
            raise InterpError(f"{name!r} declared twice in one scope")
        scope[name] = slot

    # --- expressions ------------------------------------------------------

    def subscript(self, e: Index) -> tuple[_Array, tuple[int, ...]]:
        arr = self.array(e.array)
        at = tuple(self.eval(i) for i in e.at)
        if any(type(i) is not int for i in at):
            raise InterpError(f"{e.array}: subscript {at} is not all I32")
        shape = arr.data.shape
        if len(at) != len(shape) or any(not 0 <= i < n for i, n in
                                        zip(at, shape, strict=True)):
            raise InterpError(f"{e.array}{list(at)} outside shape {shape}")
        return arr, at

    def load(self, e: Var | Index) -> int | float:
        if isinstance(e, Var):
            slot = self.scalar(e.name)
            if slot.value is None:
                raise InterpError(f"{e.name!r} read before it was written")
            return slot.value
        arr, at = self.subscript(e)
        if arr.written is not None and not arr.written[at]:
            raise InterpError(f"{e.array}{list(at)} read before it was written")
        v = arr.data[at]
        if arr.type is Scalar.F32:
            return F32(float(v))
        return float(v) if arr.type is Scalar.F64 else int(v)

    def store(self, dst: Var | Index, v: int | float) -> None:
        if isinstance(dst, Var):
            slot = self.scalar(dst.name)
            slot.value = _coerce(slot.type, v, dst.name)
            return
        arr, at = self.subscript(dst)
        if not arr.writable:
            raise InterpError(f"{dst.array} is an IN array, written to")
        arr.data[at] = _coerce(arr.type, v, dst.array)
        if arr.written is not None:
            arr.written[at] = True

    def eval(self, e: Expr) -> int | float:
        if isinstance(e, Lit):
            return e.value
        if isinstance(e, (Var, Index)):
            return self.load(e)
        if isinstance(e, Weighted):
            terms = [(t.c, None if t.g is Factor.ONE
                      else self.load(Var(t.g.name_in_kernel)), self.load(t.src))
                     for t in e.terms]
            kinds = {_kind(v, "a Weighted term") for _, g, src in terms
                     for v in (g, src) if v is not None}
            if Scalar.F32 in kinds:
                if Scalar.F64 in kinds:
                    raise _mixed("a Weighted sum")
                return self.weighted32(terms)
            acc = 0.0
            for c, g, src in terms:
                acc += float(c) * float(1.0 if g is None else g) * float(src)
            return acc
        if isinstance(e, Call):
            return self.call(e)
        raise InterpError(f"not an Expr: {e!r}")  # pragma: no cover

    @staticmethod
    def weighted32(terms: list) -> F32:
        """An F32 `Weighted`, rounded as the emitted `((c * g) * src)` terms
        summed left to right from the first are (an I32 converts, rounded)."""
        acc: F32 | None = None
        for c, g, src in terms:
            cg = _f32(c) if g is None else _f32(float(_f32(c)) * float(_f32(g)))
            term = _f32(float(cg) * float(_f32(src)))
            acc = term if acc is None else _f32(float(acc) + float(term))
        assert acc is not None
        return acc

    def call(self, e: Call) -> int | float:
        fn = e.fn
        if fn in self.geometry:
            return self.geometry[fn]
        args = [self.eval(a) for a in e.args]
        if fn is Fn.TO_F32:
            (x,) = args
            if _kind(x, fn.name) is Scalar.F32:
                raise InterpError("TO_F32 of an F32")
            return _f32(x)
        if fn is Fn.TO_F64:
            (x,) = args
            if _kind(x, fn.name) is Scalar.F64:
                raise InterpError("TO_F64 of an F64")
            return float(x)
        if fn is Fn.SQRT:
            (x,) = args
            if x < 0:
                raise InterpError(f"sqrt of {x!r}")
            return _f32(math.sqrt(x)) if type(x) is F32 else math.sqrt(x)
        if fn is Fn.EXP:
            (x,) = args
            return _f32(math.exp(x)) if type(x) is F32 else math.exp(float(x))
        a, b = args
        return _arith(fn, a, b)

    # --- statements -------------------------------------------------------

    def run(self, body: tuple[Stmt, ...]) -> Generator[Any, Any, None]:
        for s in body:
            yield from self.stmt(s)

    def block(self, body: tuple[Stmt, ...]) -> Generator[Any, Any, None]:
        self.scopes.append({})
        try:
            yield from self.run(body)
        finally:
            self.scopes.pop()

    def stmt(self, s: Stmt) -> Generator[Any, Any, None]:
        if isinstance(s, Note):
            return
        if isinstance(s, Decl):
            self.decl(s)
        elif isinstance(s, Assign):
            self.store(s.dst, self.eval(s.value))
        elif isinstance(s, AtomicAdd):
            self.store(s.dst, _arith(Fn.ADD, self.eval(s.dst), self.eval(s.value)))
        elif isinstance(s, If):
            yield from self.block(s.then if self.eval(s.cond) else s.orelse)
        elif isinstance(s, For):
            yield from self.loop(s)
        elif isinstance(s, Break):
            raise _Break
        elif isinstance(s, WarpReduce):
            total = yield (s, self.eval(s.value))
            self.store(s.dst, total)
        elif isinstance(s, LaneBroadcast):
            got = yield (s, (self.eval(s.value), self.eval(s.source)))
            self.store(s.dst, got)
        elif isinstance(s, Barrier):
            yield (s, None)  # the block resumes every thread together
        elif isinstance(s, Boys):
            self.boys(s)
        elif isinstance(s, Inline):
            yield from self.inline(s)
        elif isinstance(s, Raw):
            raise InterpError("Raw is opaque to the interpreter")
        else:  # pragma: no cover
            raise InterpError(f"not a Stmt: {s!r}")

    def boys(self, s: Boys) -> None:
        """`dst[m] = F_m(T)`, m = 0..top, each order from the injected
        evaluator (module docstring): what the `Boys` statement means, not
        how the writer's `fj_boys` gets there."""
        if self.interp.boys is None:
            raise InterpError("Boys run but no boys= evaluator injected")
        T = self.eval(s.t)
        if type(T) is F32:
            raise InterpError("Boys of an F32: there is no fp32 Boys")
        arr = self.array(s.dst)
        if arr.type is not Scalar.F64 or arr.data.shape != (s.top + 1,):
            raise InterpError(f"Boys into {s.dst!r}: not an F64 array of "
                              f"{s.top + 1}")
        for m in range(s.top + 1):
            self.store(Index(s.dst, (Lit(m),)),
                       float(self.interp.boys(m, float(T))))

    def inline(self, s: Inline) -> Generator[Any, Any, None]:
        """Bind the names in the host's scopes, then run the body in isolation.

        The fragment sees its binds and nothing else of the host (its scope
        chain is replaced, not extended), and its locals end with it.
        """
        frag = s.fragment
        binds: dict[str, Any] = {}
        for name, e in s.bindings:
            if name in frag.array_binds:
                binds[name] = self.array(e.name)  # an alias: writes reach the host
            else:
                binds[name] = _bound_scalar(self.eval(e), name)
        host, self.scopes = self.scopes, [binds, {}]
        try:
            yield from self.run(frag.body)
        finally:
            self.scopes = host

    def decl(self, s: Decl) -> None:
        if not s.shape:
            slot = _Scalar(s.type)
            if s.init is not None:
                slot.value = _coerce(s.type, self.eval(s.init), s.name)
            self.declare(s.name, slot)
            return
        shape = tuple(self.eval(d) for d in s.shape)
        if any(type(n) is not int or n < 1 for n in shape):
            raise InterpError(f"{s.name}: extent {shape} is not positive I32s")
        if s.space is Space.SHARED:
            arr = self.shared.get(s.name)
            if arr is None:
                arr = self.shared[s.name] = _Array(
                    s.name, s.type, np.zeros(shape, _DTYPE[s.type]),
                    np.zeros(shape, bool), True)
            elif arr.data.shape != shape or arr.type is not s.type:
                raise InterpError(f"shared {s.name!r} redeclared differently")
        else:
            arr = _Array(s.name, s.type, np.zeros(shape, _DTYPE[s.type]),
                         np.zeros(shape, bool), True)
        self.declare(s.name, arr)

    def loop(self, s: For) -> Generator[Any, Any, None]:
        self.scopes.append({})
        try:
            self.declare(s.var, _Scalar(Scalar.I32, _coerce(
                Scalar.I32, self.eval(s.start), s.var)))
            while self.eval(Var(s.var)) < self.eval(s.stop):
                self.interp.iterations[s.var] += 1
                try:
                    yield from self.block(s.body)
                except _Break:
                    break
                step = self.eval(s.step)
                if type(step) is not int or step <= 0:
                    raise InterpError(f"For {s.var}: step {step!r} is not > 0")
                self.store(Var(s.var), _check_i32(self.eval(Var(s.var)) + step))
        finally:
            self.scopes.pop()


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------


def _broadcast(arrived: list[tuple[Any, Any]]) -> list[Any]:
    """The indexed shuffle: lane `i` gets the value of lane `arrived[i][1]`."""
    n = len(arrived)
    out = []
    for i, (_, src) in enumerate(arrived):
        if type(src) is not int or not 0 <= src < n:
            raise InterpError(f"LaneBroadcast: lane {i} names source lane "
                              f"{src!r}, outside [0, {n})")
        out.append(arrived[src][0])
    return out


def _butterfly(values: list[float | int]) -> int | float:
    """The xor-shuffle reduction: every lane ends with the same sum."""
    v = list(values)
    o = len(v) // 2
    while o:
        v = [_arith(Fn.ADD, v[i], v[i ^ o]) for i in range(len(v))]
        o //= 2
    return v[0]


class Interpreter:
    """Run one `Kernel`; `iterations` counts `For` body entries by loop var."""

    def __init__(self, kernel: Kernel, *, launch: Launch | None = None,
                 boys: Callable[[int, float], float] | None = None) -> None:
        if not isinstance(kernel, Kernel):
            raise TypeError("interpret takes a Kernel")
        self.kernel = kernel
        self.launch = Launch() if launch is None else launch
        self.boys = boys
        self.iterations: Counter[str] = Counter()
        # Array params whose declared extents could not be evaluated, and why.
        self.unchecked: dict[str, str] = {}

    def _bind(self, inputs: dict[str, Any]) -> tuple[dict[str, Any], dict]:
        params = {p.name: p for p in self.kernel.params}
        extra = inputs.keys() - params.keys()
        missing = params.keys() - inputs.keys()
        if extra or missing:
            raise InterpError(f"inputs: missing {sorted(missing)}, "
                              f"unknown {sorted(extra)}")
        globals_: dict[str, Any] = {}
        outs: dict[str, np.ndarray] = {}
        for name, p in params.items():
            v = inputs[name]
            if p.array:
                if not isinstance(v, np.ndarray) or v.dtype != _DTYPE[p.type] \
                        or v.ndim < 1:
                    raise InterpError(
                        f"{name}: needs an ndarray of {_DTYPE[p.type]}")
                out = p.intent is Intent.OUT
                data = v.copy()
                if out:
                    outs[name] = data
                globals_[name] = _Array(name, p.type, data, None, out)
            else:
                v = _number(v)
                if p.type is Scalar.F32 and type(v) is float:
                    # A caller's Python float is F64 by `Lit`'s rule; an
                    # F32 parameter takes it only when it is already exact.
                    if not is_f32(v):
                        raise InterpError(f"{name}: {v!r} is not a float32; "
                                          "pass np.float32")
                    v = F32(v)
                globals_[name] = _Scalar(p.type, _coerce(p.type, v, name))

        # Shapes. A backend lays a global array out flat (row-major), so the
        # contract is the SIZE, prod(extents) == v.size, never v.shape ==
        # extents: a flat array of the right length is the right array. It is
        # then viewed at the declared extents, so every subscript is bounds-
        # checked per extent as the backend linearises it; an OUT array is
        # returned in the caller's shape. Extents are evaluated with the bound
        # I32 scalars and the launch's block dimensions; one that cannot be
        # (an F64 scalar, a division by zero) leaves its array unchecked, with
        # the reason in `self.unchecked`: unevaluable is not wrong.
        self.unchecked = {}
        env = {k: s for k, s in globals_.items()
               if isinstance(s, _Scalar) and s.type is Scalar.I32}
        bx, by = self.launch.block
        ev = _Thread(self, env, {}, {Fn.BLOCK_DIM_X: bx, Fn.BLOCK_DIM_Y: by})
        for name, p in params.items():
            if not p.array or not p.shape:
                continue
            arr = globals_[name]
            try:
                extents = tuple(ev.eval(d) for d in p.shape)
            except InterpError as e:
                self.unchecked[name] = str(e)
                continue
            if any(type(n) is not int for n in extents):
                self.unchecked[name] = f"extents {extents} are not all I32"
                continue
            need = math.prod(extents)
            if any(n < 0 for n in extents) or need != arr.data.size:
                raise InterpError(
                    f"{name}: declared extents {p.shape!r} evaluate to "
                    f"{extents} = {need} elements, but got shape "
                    f"{arr.data.shape} = {arr.data.size} elements")
            arr.data = arr.data.reshape(extents)  # a view: an OUT still aliases
        return globals_, outs

    def run(self, **inputs: Any) -> dict[str, np.ndarray]:
        globals_, outs = self._bind(inputs)
        bx, by = self.launch.block
        lanes = self.launch.lanes
        for block_x in range(self.launch.grid):
            shared: dict[str, _Array] = {}
            threads = []
            for tid in range(bx * by):
                geometry = {
                    Fn.THREAD_X: tid % bx, Fn.THREAD_Y: tid // bx,
                    Fn.BLOCK_X: block_x, Fn.BLOCK_DIM_X: bx, Fn.BLOCK_DIM_Y: by,
                }
                # Global arrays are shared; scalar params are per-thread copies.
                mine = {k: _Scalar(v.type, v.value) if isinstance(v, _Scalar)
                        else v for k, v in globals_.items()}
                t = _Thread(self, mine, shared, geometry)
                threads.append(t.run(self.kernel.body))
            warps = [_Warp(threads[w:w + lanes], full=len(threads) - w >= lanes)
                     for w in range(0, len(threads), lanes)]
            while True:  # one pass per Barrier the block meets
                at = [w.advance() for w in warps]
                if all(a is None for a in at):
                    break
                if len({id(a) for a in at}) != 1:
                    raise InterpError(
                        "a Barrier not reached by every thread of the block: "
                        "a warp finished, or waits at another statement")
        return outs


class _Warp:
    """One warp's lanes, advanced together (module docstring, THE MACHINE):
    to completion, or to the next `Barrier`, which `advance` returns."""

    def __init__(self, lanes: list[Generator], *, full: bool) -> None:
        self.lanes, self.full = lanes, full
        self.sends: list[Any] = [None] * len(lanes)

    def advance(self) -> Barrier | None:
        lanes = self.lanes
        while True:
            arrived = []
            for i, g in enumerate(lanes):
                try:
                    arrived.append(g.send(self.sends[i]))
                except StopIteration:
                    arrived.append(None)
            self.sends = [None] * len(lanes)
            if all(a is None for a in arrived):
                return None
            stmts = {id(a[0]) if a is not None else None for a in arrived}
            barrier = any(a is not None and isinstance(a[0], Barrier)
                          for a in arrived)
            if len(stmts) != 1:
                if barrier:
                    raise InterpError(
                        "a Barrier not reached by every thread of the block: "
                        "lanes of one warp are not all at it")
                what = next(type(a[0]).__name__ for a in arrived
                            if a is not None)
                raise InterpError(f"warp diverged at a {what}: lanes are "
                                  "not all at the same statement")
            if barrier:
                return arrived[0][0]
            what = type(arrived[0][0]).__name__
            if not self.full:
                raise InterpError(f"{what} in a partial warp")
            if isinstance(arrived[0][0], LaneBroadcast):
                self.sends = _broadcast([a[1] for a in arrived])
            else:
                total = _butterfly([a[1] for a in arrived])
                self.sends = [total] * len(lanes)


def interpret(kernel: Kernel, *, launch: Launch | None = None,
              boys: Callable[[int, float], float] | None = None,
              **inputs: Any) -> dict[str, np.ndarray]:
    """Execute every Stmt: Decl, Assign, For, If, Break, WarpReduce, AtomicAdd.

    Returns a copy of every OUT array after the launch. The warp reduction is
    emulated on the host (see the module docstring for the machine).
    """
    return Interpreter(kernel, launch=launch, boys=boys).run(**inputs)


class _Alone:
    """What a `_Thread` needs of its machine, for a fragment run alone."""

    def __init__(self, boys: Callable[[int, float], float] | None) -> None:
        self.boys = boys
        self.iterations: Counter[str] = Counter()


def interpret_fragment(fragment: Fragment, *,
                       boys: Callable[[int, float], float] | None = None,
                       **bindings: Any) -> dict[str, Any]:
    """Run one `Fragment` with no kernel around it: thread 0 of one block.

    `bindings` names every bind exactly once: a scalar bind takes a Python or
    numpy number (an int is I32, a float F64, as `Lit` types them, and a
    numpy float32 F32); an array bind an ndarray of float64, float32 or
    int32, which is copied, indexed at its own
    shape, and may be written. Returns the fragment's top-level locals (a
    scalar's value, or None if never written; an array's data) and every
    array bind's data after the run. A `WarpReduce` raises: alone, there is
    no warp.
    """
    if not isinstance(fragment, Fragment):
        raise TypeError("interpret_fragment takes a Fragment")
    extra = bindings.keys() - set(fragment.binds)
    missing = set(fragment.binds) - bindings.keys()
    if extra or missing:
        raise InterpError(f"bindings: missing {sorted(missing)}, "
                          f"unknown {sorted(extra)}")
    binds: dict[str, Any] = {}
    for name in fragment.binds:
        v = bindings[name]
        if name in fragment.array_binds:
            types = {d: t for t, d in _DTYPE.items()}
            if not isinstance(v, np.ndarray) or v.ndim < 1 or v.dtype not in types:
                raise InterpError(f"{name}: an array bind needs an ndarray of "
                                  "float64, float32 or int32")
            binds[name] = _Array(name, types[v.dtype], v.copy(), None, True)
        else:
            binds[name] = _bound_scalar(v, name)
    geometry = {Fn.THREAD_X: 0, Fn.THREAD_Y: 0, Fn.BLOCK_X: 0,
                Fn.BLOCK_DIM_X: 1, Fn.BLOCK_DIM_Y: 1}
    thread = _Thread(_Alone(boys), binds, {}, geometry)  # type: ignore[arg-type]
    local: dict[str, Any] = {}
    thread.scopes.append(local)
    for at in thread.run(fragment.body):
        if isinstance(at[0], Barrier):
            raise InterpError("Barrier in a fragment run alone: there is no "
                              "block")
        raise InterpError(f"{type(at[0]).__name__} in a fragment run alone: "
                          "there is no warp")
    out: dict[str, Any] = {}
    for name, slot in local.items():
        out[name] = slot.value if isinstance(slot, _Scalar) else slot.data.copy()
    for name in sorted(fragment.array_binds):
        out[name] = binds[name].data
    return out
