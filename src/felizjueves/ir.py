"""The kernel IR: what a generated kernel IS, before any vendor spells it.

Types only. No passes, no emitters, no interpreter -- those are functions of a
`Kernel` and live elsewhere (`interp.py`, `emit/`). Everything is a frozen
dataclass over tuples, so a `Kernel` is hashable and two of them compare with
`==`: a pass is checked by structural equality on IR, never on text.

    Kernel(name, params, body)
    Stmt = Decl | Assign | For | If | Break | WarpReduce | AtomicAdd | Note | Raw
    Expr = Weighted | Index | Call | Lit | Var

THE ONE THING EXPRESSED NATIVELY is the recurrence instruction, in the reduction
form of `[JEngine, v2, eqs. 10-12]` (docs/implementation/j_engine.md):

    dst = SUM_k ( c_k * g_k * src_k )      c_k a positive int, g_k a `Factor`

`Factor` is the declared, closed vocabulary of `g_k`: `1` and `PQ[d]`. With
`c_k` in `{1, r_d - 1}` that is the two terms of
`[r]^(m) = PQ_d [r-1d]^(m+1) + (r_d - 1) [r-2d]^(m+1)`, and the cheap unit
component (one multiply; the second term vanishes) is a one-term `Weighted`.
A `g_k` that is an arbitrary string would be a `Raw` in disguise, so it raises.

CLOSED VOCABULARIES, all named here and nowhere else:

- `Scalar`: `F64` and `I32`. **The floating type is fp64 and that is an IR
  fact** -- the oracle (#20) is float64, and the claim made this milestone is
  "the algorithm is right"; fp32 would fold an accuracy question into it.
  Precision as a lever belongs in `Spec.levers`, not in a backend flag.
- `Factor`: the recurrence's `g_k`.
- `Fn`: what a `Call` may call -- the arithmetic, comparison and special
  functions the [p|q] contract (#25) needs, and the launch geometry
  (thread/block index and extent). Nothing "for later".
- `Space`: where a `Decl` lives, `LOCAL` (per thread) or `SHARED` (per block).
  The storage class is a decision, so it is an IR fact a pass can change and
  the interpreter can check; a backend that chose it would be deciding.

WHAT IS DELIBERATELY NOT HERE: the wavefront width, or anything else the two
vendors spell differently. `WarpReduce` says "sum across the warp"; how wide a
warp is, and how a shuffle is spelled, is the device target's (#37) and the
backend's business. The IR knowing an arch is how the seam #27 validates stops
being a seam. Likewise there are no pointers, templates or classes: an array
is a name you `Index`.

`Note(prov)` is metadata -- an equation or algorithm line a backend renders as
a comment, so a citation lives in one place. It never affects semantics: an
interpreter ignores it.

`Raw(text)` is the single escape hatch: explicitly marked, opaque to every
pass and to the interpreter, and **counted** (test/test_ir.py's census over
src/). Raise the count only with a reason.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class Scalar(Enum):
    """Element types. fp64 is the floating type, as an IR fact."""

    F64 = "f64"
    I32 = "i32"


class Factor(Enum):
    """The `g_k` of a `Weighted` term: `1`, or one component of `PQ`.

    A non-unit factor names a scalar the kernel must bind (a `Param` or `Decl`
    called `Factor.name_in_kernel`), which `Kernel` validates.
    """

    ONE = "1"
    PQ_X = "PQ_x"
    PQ_Y = "PQ_y"
    PQ_Z = "PQ_z"

    @property
    def name_in_kernel(self) -> str | None:
        return None if self is Factor.ONE else self.value

    @classmethod
    def pq(cls, d: int) -> Factor:
        """`PQ[d]` for d in 0, 1, 2 (x, y, z)."""
        return (cls.PQ_X, cls.PQ_Y, cls.PQ_Z)[d]


class Fn(Enum):
    """What a `Call` may call, with its arity. Closed.

    Each member is here because the [p|q] contract (#25) needs it:

    - ADD SUB MUL DIV: index arithmetic, `PQ = P - Q`, `theta^2 = ze/(z+e)`,
      the `D_max * I_p * I_q` bound, the `[p|q] D_q` contraction.
    - LT: the screening test `bound < tau`, whose `If` holds the `Break`.
    - SQRT: `theta` in the fundamental `(2/sqrt(pi)) theta (-2 theta^2)^m F_m`.
    - BOYS: `F_m(T)`, `Call(BOYS, (m, T))` -- the fundamentals' special
      function; the recurrences never reduce it.
    - THREAD_X THREAD_Y BLOCK_X BLOCK_DIM_X BLOCK_DIM_Y: launch geometry. The
      bra index is `THREAD_X + BLOCK_X * BLOCK_DIM_X`; kets are strip-mined
      from `THREAD_Y` with stride `BLOCK_DIM_Y`.
    """

    ADD = ("add", 2)
    SUB = ("sub", 2)
    MUL = ("mul", 2)
    DIV = ("div", 2)
    LT = ("lt", 2)
    SQRT = ("sqrt", 1)
    BOYS = ("boys", 2)
    THREAD_X = ("thread_x", 0)
    THREAD_Y = ("thread_y", 0)
    BLOCK_X = ("block_x", 0)
    BLOCK_DIM_X = ("block_dim_x", 0)
    BLOCK_DIM_Y = ("block_dim_y", 0)

    @property
    def arity(self) -> int:
        return self.value[1]


class Space(Enum):
    """Where a `Decl` lives: per thread, or shared by the block."""

    LOCAL = "local"
    SHARED = "shared"


class Intent(Enum):
    """Direction of a kernel parameter."""

    IN = "in"
    OUT = "out"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _ident(name: object, what: str) -> None:
    if not isinstance(name, str) or not _IDENT.match(name) or keyword.iskeyword(name):
        raise ValueError(f"{what} {name!r} is not an identifier")


def _tuple(obj: object, field: str, value: object) -> tuple:
    """Freeze a sequence field into a tuple, so the node stays hashable."""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{type(obj).__name__}.{field} must be a sequence")
    t = tuple(value)
    object.__setattr__(obj, field, t)
    return t


def _expr(value: object, what: str) -> None:
    if not isinstance(value, EXPR_TYPES):
        raise TypeError(f"{what} must be an Expr, got {type(value).__name__}")


def _lvalue(value: object, what: str) -> None:
    if not isinstance(value, (Var, Index)):
        raise TypeError(f"{what} must be a Var or Index, got {type(value).__name__}")


def _body(obj: object, field: str, value: object) -> None:
    for s in _tuple(obj, field, value):
        if not isinstance(s, STMT_TYPES):
            raise TypeError(
                f"{type(obj).__name__}.{field} holds {type(s).__name__}, not a Stmt"
            )


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Var:
    """A named scalar: a local, a loop variable, or a scalar `Param`."""

    name: str

    def __post_init__(self) -> None:
        _ident(self.name, "Var name")


@dataclass(frozen=True, eq=False)
class Lit:
    """A literal. Its type is its Python type: `int` is I32, `float` is F64.

    Equality includes the type, so `Lit(1) != Lit(1.0)`: an I32 is not an F64.
    """

    value: int | float

    def __post_init__(self) -> None:
        if type(self.value) not in (int, float):
            raise TypeError(f"Lit value {self.value!r} is not an int or float")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lit):
            return NotImplemented
        return type(self.value) is type(other.value) and self.value == other.value

    def __hash__(self) -> int:
        return hash((Lit, type(self.value), self.value))

    @property
    def type(self) -> Scalar:
        return Scalar.I32 if type(self.value) is int else Scalar.F64


@dataclass(frozen=True)
class Index:
    """`array[i0, i1, ...]`: an element of an array `Param` or array `Decl`."""

    array: str
    at: tuple[Expr, ...]

    def __post_init__(self) -> None:
        _ident(self.array, "Index array")
        if not _tuple(self, "at", self.at):
            raise ValueError("Index needs at least one subscript")
        for i in self.at:
            _expr(i, "an Index subscript")


@dataclass(frozen=True)
class Call:
    """`fn(args...)` with `fn` from the closed `Fn` vocabulary."""

    fn: Fn
    args: tuple[Expr, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fn, Fn):
            raise ValueError(f"Call fn {self.fn!r} is not in the Fn vocabulary")
        args = _tuple(self, "args", self.args)
        if len(args) != self.fn.arity:
            raise ValueError(
                f"{self.fn.name} takes {self.fn.arity} args, got {len(args)}"
            )
        for a in args:
            _expr(a, f"a {self.fn.name} argument")


@dataclass(frozen=True)
class Term:
    """One `c * g * src` of a `Weighted` sum."""

    c: int
    g: Factor
    src: Var | Index

    def __post_init__(self) -> None:
        if type(self.c) is not int or self.c < 1:
            # A zero term is dropped, not emitted: that is the unit-component
            # saving. The reduction form's coefficients are 1 and r_d - 1 >= 1.
            raise ValueError(f"Term coefficient {self.c!r} is not a positive int")
        if not isinstance(self.g, Factor):
            raise ValueError(f"Term factor {self.g!r} is not in the Factor vocabulary")
        _lvalue(self.src, "a Term source")


@dataclass(frozen=True)
class Weighted:
    """The recurrence instruction: `SUM_k c_k * g_k * src_k`.

    `[JEngine, v2, eqs. 10-12]` in reduction form. Interpreted in fp64.
    """

    terms: tuple[Term, ...]

    def __post_init__(self) -> None:
        terms = _tuple(self, "terms", self.terms)
        if not terms:
            raise ValueError("Weighted needs at least one term")
        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"Weighted holds {type(t).__name__}, not a Term")


Expr = Weighted | Index | Call | Lit | Var
EXPR_TYPES = (Weighted, Index, Call, Lit, Var)


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decl:
    """Declare `name` of `type`, a scalar when `shape` is (), else an array.

    `space` is LOCAL (per thread) or SHARED (per block); `init` optionally
    initialises a scalar.
    """

    name: str
    type: Scalar
    shape: tuple[Expr, ...] = ()
    space: Space = Space.LOCAL
    init: Expr | None = None

    def __post_init__(self) -> None:
        _ident(self.name, "Decl name")
        if not isinstance(self.type, Scalar):
            raise ValueError(f"Decl type {self.type!r} is not a Scalar")
        if not isinstance(self.space, Space):
            raise ValueError(f"Decl space {self.space!r} is not a Space")
        for d in _tuple(self, "shape", self.shape):
            _expr(d, "a Decl extent")
        if self.init is not None:
            if self.shape:
                raise ValueError("an array Decl takes no init")
            _expr(self.init, "a Decl init")


@dataclass(frozen=True)
class Assign:
    """`dst = value`."""

    dst: Var | Index
    value: Expr

    def __post_init__(self) -> None:
        _lvalue(self.dst, "Assign dst")
        _expr(self.value, "Assign value")


@dataclass(frozen=True)
class For:
    """`for var = start; var < stop; var += step`. All bounds are `Expr`.

    `var` is an I32 scoped to the body. A strip-mined ket loop is
    `For("q", THREAD_Y, n_kets, BLOCK_DIM_Y, ...)`.
    """

    var: str
    start: Expr
    stop: Expr
    step: Expr
    body: tuple[Stmt, ...]

    def __post_init__(self) -> None:
        _ident(self.var, "For var")
        _expr(self.start, "For start")
        _expr(self.stop, "For stop")
        _expr(self.step, "For step")
        _body(self, "body", self.body)


@dataclass(frozen=True)
class If:
    """`if cond: then else: orelse`."""

    cond: Expr
    then: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...] = ()

    def __post_init__(self) -> None:
        _expr(self.cond, "If cond")
        _body(self, "then", self.then)
        _body(self, "orelse", self.orelse)


@dataclass(frozen=True)
class Break:
    """Leave the innermost enclosing `For`. Valid only inside one.

    The screening exit: kets arrive sorted descending by bound, so the first
    one under `tau` ends the loop -- a `break`, not a `continue`.
    """


@dataclass(frozen=True)
class WarpReduce:
    """`dst = SUM over the lanes of this thread's warp of value`.

    Vendor-neutral on purpose: the lane count is the device target's fact,
    and the shuffle spelling (and whatever synchronisation it implies) is the
    backend's. Every lane sees the sum.
    """

    dst: Var | Index
    value: Expr

    def __post_init__(self) -> None:
        _lvalue(self.dst, "WarpReduce dst")
        _expr(self.value, "WarpReduce value")


@dataclass(frozen=True)
class AtomicAdd:
    """`dst += value`, atomically. `dst` is an element of an array."""

    dst: Index
    value: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.dst, Index):
            raise TypeError("AtomicAdd dst must be an Index")
        _expr(self.value, "AtomicAdd value")


@dataclass(frozen=True)
class Note:
    """Provenance -- an equation or algorithm line. A comment, never semantics."""

    prov: str

    def __post_init__(self) -> None:
        if not isinstance(self.prov, str) or not self.prov:
            raise ValueError("Note needs a non-empty provenance string")


@dataclass(frozen=True)
class Raw:
    """Backend text, verbatim. The one escape hatch -- and it is counted."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Raw needs non-empty text")


Stmt = Decl | Assign | For | If | Break | WarpReduce | AtomicAdd | Note | Raw
STMT_TYPES = (Decl, Assign, For, If, Break, WarpReduce, AtomicAdd, Note, Raw)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """A kernel argument: a scalar, or (when `array`) a global array.

    An OUT parameter is an array; there are no pointers, only names to index.
    """

    name: str
    type: Scalar
    array: bool = False
    intent: Intent = Intent.IN

    def __post_init__(self) -> None:
        _ident(self.name, "Param name")
        if not isinstance(self.type, Scalar):
            raise ValueError(f"Param type {self.type!r} is not a Scalar")
        if not isinstance(self.intent, Intent):
            raise ValueError(f"Param intent {self.intent!r} is not an Intent")
        if self.intent is Intent.OUT and not self.array:
            raise ValueError(f"OUT param {self.name!r} must be an array")


def walk(body: tuple[Stmt, ...]) -> Iterator[Stmt]:
    """Every statement in `body`, nested ones included, in source order."""
    for s in body:
        yield s
        if isinstance(s, For):
            yield from walk(s.body)
        elif isinstance(s, If):
            yield from walk(s.then)
            yield from walk(s.orelse)


def _exprs_of(s: Stmt) -> tuple[Expr | None, ...]:
    if isinstance(s, Decl):
        return (*s.shape, s.init)
    if isinstance(s, (Assign, WarpReduce, AtomicAdd)):
        return (s.dst, s.value)
    if isinstance(s, For):
        return (s.start, s.stop, s.step)
    if isinstance(s, If):
        return (s.cond,)
    return ()


def _subexprs(e: Expr) -> Iterator[Expr]:
    yield e
    if isinstance(e, Index):
        for i in e.at:
            yield from _subexprs(i)
    elif isinstance(e, Call):
        for a in e.args:
            yield from _subexprs(a)
    elif isinstance(e, Weighted):
        for t in e.terms:
            yield from _subexprs(t.src)


def _check_breaks(body: tuple[Stmt, ...], in_loop: bool) -> None:
    for s in body:
        if isinstance(s, Break) and not in_loop:
            raise ValueError("Break outside a For")
        if isinstance(s, For):
            _check_breaks(s.body, True)
        elif isinstance(s, If):
            _check_breaks(s.then, in_loop)
            _check_breaks(s.orelse, in_loop)


@dataclass(frozen=True)
class Kernel:
    """One device kernel: a name, its parameters, and an ordered body.

    Validated on construction: parameter names are unique, every `Break` sits
    inside a `For`, and every non-unit `Factor` a `Weighted` uses is bound by
    name to a `Param` or `Decl`. Frozen and hashable.
    """

    name: str
    params: tuple[Param, ...]
    body: tuple[Stmt, ...]

    def __post_init__(self) -> None:
        _ident(self.name, "Kernel name")
        params = _tuple(self, "params", self.params)
        for p in params:
            if not isinstance(p, Param):
                raise TypeError(f"Kernel params holds {type(p).__name__}, not Param")
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate Param names in {names}")
        _body(self, "body", self.body)
        _check_breaks(self.body, False)

        bound = set(names) | {s.name for s in walk(self.body) if isinstance(s, Decl)}
        for s in walk(self.body):
            for root in _exprs_of(s):
                if root is None:
                    continue
                for e in _subexprs(root):
                    if not isinstance(e, Weighted):
                        continue
                    for t in e.terms:
                        need = t.g.name_in_kernel
                        if need is not None and need not in bound:
                            raise ValueError(
                                f"Factor {t.g.name} needs a Param or Decl "
                                f"named {need!r}"
                            )
