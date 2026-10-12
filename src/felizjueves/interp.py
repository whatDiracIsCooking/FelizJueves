"""Execute a `Kernel` on the host: the compiler-free answer to "is it right?".

    interpret(kernel, **inputs) -> {out_param: ndarray}

An instruction set machine for ir.py and nothing more. It executes Decl,
Assign, For, If, Break, WarpReduce, AtomicAdd and Inline; it skips Note; it
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
  two). The IR deliberately does not know the width (ir.py); the device target
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
- Warps of a block run one after another, sharing the block's SHARED arrays.
  The IR has no barrier, so communication through shared memory between
  threads is not modelled beyond sequential consistency.

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

`Fn.BOYS` is the one special function. Its semantics, F_m(T), are INJECTED:
the caller passes `boys=` (the pins pass `reference.boys.boys`, quadrature),
and a kernel that calls `Fn.BOYS` without one raises. A kernel that calls it
is therefore checked on everything *but* its Boys function. There is no
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
    Lit,
    Note,
    Raw,
    Scalar,
    Space,
    Stmt,
    Var,
    WarpReduce,
    Weighted,
)

_I32_MIN, _I32_MAX = -(2**31), 2**31 - 1
_DTYPE = {Scalar.F64: np.dtype(np.float64), Scalar.I32: np.dtype(np.int32)}


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
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    elif isinstance(v, np.generic):
        v = v.item()
    if type(v) is int:
        return _Scalar(Scalar.I32, _check_i32(v))
    if type(v) is float:
        return _Scalar(Scalar.F64, v)
    raise InterpError(f"bind {name!r}: {v!r} is not an int or float")


def _check_i32(v: int) -> int:
    if not _I32_MIN <= v <= _I32_MAX:
        raise InterpError(f"I32 overflow: {v}")
    return v


def _coerce(type_: Scalar, v: Any, what: str) -> int | float:
    """Store `v` into a `type_` slot: int widens into F64; float never into I32."""
    if type_ is Scalar.F64:
        if type(v) not in (int, float):
            raise InterpError(f"{what}: {v!r} is not a number")
        return float(v)
    if type(v) is not int:
        raise InterpError(f"{what}: storing {type(v).__name__} {v!r} into an I32")
    return _check_i32(v)


# ---------------------------------------------------------------------------
# One thread
# ---------------------------------------------------------------------------


class _Thread:
    """One thread's execution of the body, as a coroutine.

    It yields `(stmt, value)` at a WarpReduce and is sent the warp's sum.
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
            acc = 0.0
            for t in e.terms:
                g = 1.0 if t.g is Factor.ONE else self.load(Var(t.g.name_in_kernel))
                acc += float(t.c) * float(g) * float(self.load(t.src))
            return acc
        if isinstance(e, Call):
            return self.call(e)
        raise InterpError(f"not an Expr: {e!r}")  # pragma: no cover

    def call(self, e: Call) -> int | float:
        fn = e.fn
        if fn in self.geometry:
            return self.geometry[fn]
        args = [self.eval(a) for a in e.args]
        if fn is Fn.BOYS:
            m, T = args
            if self.interp.boys is None:
                raise InterpError("BOYS called but no boys= evaluator injected")
            if type(m) is not int:
                raise InterpError(f"BOYS order {m!r} is not an I32")
            return float(self.interp.boys(m, float(T)))
        if fn is Fn.SQRT:
            (x,) = args
            if x < 0:
                raise InterpError(f"sqrt of {x!r}")
            return math.sqrt(x)
        if fn is Fn.EXP:
            (x,) = args
            return math.exp(float(x))
        a, b = args
        ints = type(a) is int and type(b) is int
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
            self.store(s.dst, self.eval(s.dst) + self.eval(s.value))
        elif isinstance(s, If):
            yield from self.block(s.then if self.eval(s.cond) else s.orelse)
        elif isinstance(s, For):
            yield from self.loop(s)
        elif isinstance(s, Break):
            raise _Break
        elif isinstance(s, WarpReduce):
            total = yield (s, self.eval(s.value))
            self.store(s.dst, total)
        elif isinstance(s, Inline):
            yield from self.inline(s)
        elif isinstance(s, Raw):
            raise InterpError("Raw is opaque to the interpreter")
        else:  # pragma: no cover
            raise InterpError(f"not a Stmt: {s!r}")

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


def _butterfly(values: list[float | int]) -> int | float:
    """The xor-shuffle reduction: every lane ends with the same sum."""
    v = list(values)
    o = len(v) // 2
    while o:
        v = [v[i] + v[i ^ o] for i in range(len(v))]
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
                if isinstance(v, np.ndarray) and v.ndim == 0:
                    v = v.item()
                elif isinstance(v, np.generic):
                    v = v.item()
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
            for w in range(0, len(threads), lanes):
                self._warp(threads[w:w + lanes], full=len(threads) - w >= lanes)
        return outs

    def _warp(self, lanes: list[Generator], *, full: bool) -> None:
        sends: list[Any] = [None] * len(lanes)
        while True:
            arrived = []
            for i, g in enumerate(lanes):
                try:
                    arrived.append(g.send(sends[i]))
                except StopIteration:
                    arrived.append(None)
            if all(a is None for a in arrived):
                return
            stmts = {id(a[0]) if a is not None else None for a in arrived}
            if len(stmts) != 1:
                raise InterpError("warp diverged at a WarpReduce: lanes are "
                                  "not all at the same statement")
            if not full:
                raise InterpError("WarpReduce in a partial warp")
            total = _butterfly([a[1] for a in arrived])
            sends = [total] * len(lanes)


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
    numpy number (an int is I32, a float F64, as `Lit` types them); an array
    bind an ndarray of float64 or int32, which is copied, indexed at its own
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
                                  "float64 or int32")
            binds[name] = _Array(name, types[v.dtype], v.copy(), None, True)
        else:
            binds[name] = _bound_scalar(v, name)
    geometry = {Fn.THREAD_X: 0, Fn.THREAD_Y: 0, Fn.BLOCK_X: 0,
                Fn.BLOCK_DIM_X: 1, Fn.BLOCK_DIM_Y: 1}
    thread = _Thread(_Alone(boys), binds, {}, geometry)  # type: ignore[arg-type]
    local: dict[str, Any] = {}
    thread.scopes.append(local)
    for _ in thread.run(fragment.body):
        raise InterpError("WarpReduce in a fragment run alone: there is no warp")
    out: dict[str, Any] = {}
    for name, slot in local.items():
        out[name] = slot.value if isinstance(slot, _Scalar) else slot.data.copy()
    for name in sorted(fragment.array_binds):
        out[name] = binds[name].data
    return out
