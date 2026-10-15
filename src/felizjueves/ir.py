"""The kernel IR: what a generated kernel IS, before any vendor spells it.

Types only. No passes, no emitters, no interpreter -- those are functions of a
`Kernel` and live elsewhere (`interp.py`, `emit/`). Everything is a frozen
dataclass over tuples, so a `Kernel` is hashable and two of them compare with
`==`: a pass is checked by structural equality on IR, never on text.

    Kernel(name, params, body)
    Stmt = Decl | Assign | For | If | Break | WarpReduce | AtomicAdd | Note | Raw
         | Inline
    Expr = Weighted | Index | Call | Lit | Var
    Fragment(name, binds, body)   -- spliced into a kernel by Inline(fragment, bindings)

THE ONE THING EXPRESSED NATIVELY is the recurrence instruction, in the reduction
form of `[JEngine, v2, eqs. 10-12]` (docs/implementation/j_engine.md):

    dst = SUM_k ( c_k * g_k * src_k )      c_k a nonzero int, g_k a `Factor`

`Factor` is the declared, closed vocabulary of `g_k`: `1` and `PQ[d]`
(and, for the `E` fragments, `1/2zeta`, `X_PA[d]`, `X_PB[d]`; for the HGP
recurrences of `[Fock]` eqs 25, 26 and 28, #277, the scalars listed on
`Factor`). With `c_k` in `{1, r_d - 1}` that is the two terms of
`[r]^(m) = PQ_d [r-1d]^(m+1) + (r_d - 1) [r-2d]^(m+1)`, and the cheap unit
component (one multiply; the second term vanishes) is a one-term `Weighted`.
A `g_k` that is an arbitrary string would be a `Raw` in disguise, so it raises.
`c_k` is signed since #277 (eq 25's `-e_d` term); it is never zero -- a zero
term is dropped, not emitted.

CLOSED VOCABULARIES, all named here and nowhere else:

- `Scalar`: `F64`, `F32` and `I32`. **Every value's precision is an IR
  fact** -- declared on its `Param` or `Decl`, never a backend flag. Until
  #338 this line read "the floating type is fp64 and that is an IR fact": the
  oracle (#20) is float64, and the claim of the first milestones was "the
  algorithm is right", which fp32 would have folded an accuracy question
  into. That claim is now made (both engines, on both cards), and milestone
  13's mixed-precision levers (#346, #347, #355) need an fp32 value to be
  SPELLABLE -- with F64 the only floating type, mixed precision could not be
  written at all. So the fact was widened rather than dropped: what stays an
  IR fact is that precision is declared per value and changes only where the
  IR says so (the cast rule below). The F64/I32 rules are what they were, so
  no fp64 kernel is refused or spelled differently. Choosing fp32 for a
  kernel is still a lever's decision (`Spec.levers`), never the IR's.
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

PRECISION HYGIENE (#338). Every expression has a type (`expr_type`), and a
`Kernel` checks every statement against these rules when it is built:

- A float literal (`Lit(1.5)`) is F64 and an int literal I32, always: no
  literal takes its type from its context. An F32 constant is spelled
  `Call(Fn.TO_F32, (Lit(x),))`, and `x` must be exactly a float32, so the
  constant is the value written (the backends spell it `xf`).
- Arithmetic (`ADD SUB MUL DIV LT`): I32 with I32 is I32; I32 with a float
  type converts to that type, as C converts it; F64 with F64 is F64 and F32
  with F32 is F32. **F32 with F64 is refused**: C would silently widen the
  F32 operand, and an "fp32" computation would run in fp64 without saying so.
- A `Weighted`'s factor scalars and sources share one float type; F32 and
  F64 in one sum is refused. Its int coefficients take the sum's type.
- `SQRT` and `EXP` keep their argument's float type (an I32 argument is F64,
  as before); `BOYS` is fp64 only and refuses an F32 `T` -- there is no fp32
  Boys function.
- A store (`Assign`, a `Decl`'s init, `AtomicAdd`, `WarpReduce`) of F32 into
  F64, of F64 into F32, or of F32 into I32 is refused; an I32 into either
  float type converts. `For` bounds may not be F32.
- **The cast rule:** the only way between F32 and F64 is an explicit
  `Fn.TO_F32` / `Fn.TO_F64`, and a cast to the type its value already has is
  refused (it would say a precision changes where none does).

A name the checker cannot type (one a `Raw` introduces, or a fragment's bind
when the fragment is checked alone) is unknown, and no rule fires on an
unknown: the rules refuse what is known to mix, never what cannot be seen.
Every refusal involves F32, so the check cannot refuse a kernel that
declares none.

`Fragment` is the opposite of `Raw`: a block of IR whose binding contract is
checked by construction. Its `binds` are exactly the names its body uses
without declaring; `Inline` binds each to a host `Expr`, once; and the
`Kernel` refuses a fragment whose own names collide with the host's -- a
fragment is not a macro, so nothing is ever renamed. `walk()` descends into
an `Inline`, so every census here sees through the splice.
"""

from __future__ import annotations

import keyword
import re
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class Scalar(Enum):
    """Element types. A value's precision is an IR fact (#338): the module
    docstring's "PRECISION HYGIENE" says which may mix, and how."""

    F64 = "f64"
    F32 = "f32"
    I32 = "i32"


FLOATS = (Scalar.F64, Scalar.F32)


class PrecisionError(TypeError):
    """F32 and F64 meet where the IR has not said a precision changes."""


class Factor(Enum):
    """The `g_k` of a `Weighted` term. Closed; each member serves a contract.

    - ONE: the unit factor -- `(r_d - 1)` and `(t + 1)` ride in `c_k`.
    - PQ_X PQ_Y PQ_Z: `PQ[d]`, the [p|q] contract's (#25) Hermite-to-`[r]`
      recurrence, `[JEngine, v2, eqs. 10-12]`.
    - HALF_INV_ZETA, PA_X PA_Y PA_Z, PB_X PB_Y PB_Z: the `E` recurrence's
      `g_k` (#152; McMurchie & Davidson's report eqs. 2.20-2.21, entry 46 of
      docs/references.md): `E^{i+1,j}_t = (1/2zeta) E^{ij}_{t-1} + X_PA
      E^{ij}_t + (t+1) E^{ij}_{t+1}`, and the same with `X_PB` raising `j`.
      `1/2zeta` is one bound scalar, not a `DIV` per term; `X_PA` and `X_PB`
      are per axis, as `PQ` is.

    - The HGP recurrences of `[Fock]` §3.1 (#277), read off the built DAGs
      `hgp.vrr` (#273) and `hgp.hrr` (#274), not off the paper:
      - WP_X..WP_Z, WQ_X..WQ_Z, QC_X..QC_Z: eq 25's `W - P` and eq 26's
        `W - Q` and `Q - C`, per axis. The bra's `P - A` is `PA_X..` above,
        REUSED, not duplicated: it is `P - A` in both schemes, bound per
        primitive pair in the HGP kernel and inside the `E` fragment in the
        MMD one -- the evidence that the two vocabularies genuinely overlap
        rather than coincide.
      - INV2P, INV2Q, INV2PQ: `1/(2p)`, `1/(2q)`, `1/(2(p+q))`, each one bound
        scalar (as `1/2zeta` is), the `e_d`, `f_d` multipliers riding in `c`.
      - INV2P_ALPHA_OVER_P, INV2Q_ALPHA_OVER_Q: the DERIVED products
        `(1/2p)(alpha/p)` and `(1/2q)(alpha/q)` of eqs 25-26's `(m+1)`
        lowering term, each declared once per primitive quartet by the engine.
      - AB_X..AB_Z, CD_X..CD_Z: eq 28's `A - B` and its ket twin `C - D`, per
        axis. The HRR's transfer term is `ONE`; nothing else in eq 28 is new.

    WHY A DERIVED SCALAR, NOT A PRODUCT TERM (#277, decision 2). Flattened,
    eq 25 has one term whose `g` is a product of two named scalars,
    `-e_d * inv2p * alpha_over_p * src` (eq 26 likewise with `q`). Two ways to
    say it were open:

    (a) `Term` grows a tuple of factors, `g = (INV2P, ALPHA_OVER_P)`. Every
        consumer of a `Term` (the interpreter, the emitters' shared writer,
        `lower`'s costing and recycling, `Kernel`'s binding check) learns a
        product, the vocabulary stops being one scalar per term, and the
        multiply that forms the product moves inside the recurrence -- once
        per VRR term instead of once per primitive quartet.
    (b) The engine declares the product as its own scalar, bound once per
        primitive quartet, and `Term` stays single-factor.

    (b) is the choice. It keeps `Factor` a closed, finite, one-scalar-per-term
    vocabulary; it changes no existing `Term` consumer; and the product's cost
    is paid where it belongs, in the engine, once per quartet. Nothing in the
    built DAGs argues for (a): across every one of `hgp.vrr.CLASSES`' DAGs
    the product appears only as these two pairs, never in a third
    combination, so two derived members close it (test/test_ir.py checks
    that every term of every built HGP DAG is a `Term` of this vocabulary).
    `alpha/p` and `alpha/q` alone are deliberately NOT members: no DAG term
    has either as its whole `g`, so each would be vocabulary "for later";
    under (b) they are the engine's intermediates, not a `g_k`. A third
    product would be the trigger to revisit (a).

    THE SIGN LIVES IN `c`, NOT IN A FACTOR. The lowering term's coefficient
    is `-e_d` (`hgp.vrr.reduce`'s `mult`), so `Term.c` is signed; folding the
    sign into a negated derived scalar instead would keep `c >= 1` at the
    price of a factor that means "minus" -- a sign spelled as a name. And
    `WQ` names `W - Q` itself, which the engine forms from `W`: the
    vocabulary does not carry eq 26's printed `-PQ zeta/(zeta+eta)`, whose
    sign is the defect claim `fock/eq26-ket-vrr-sign`
    (data/literature/fock/claims.json; asserted by #291 on #279's test, not
    here).

    A non-unit factor names a scalar the kernel must bind (a `Param` or `Decl`
    called `Factor.name_in_kernel`), which `Kernel` validates.
    """

    ONE = "1"
    PQ_X = "PQ_x"
    PQ_Y = "PQ_y"
    PQ_Z = "PQ_z"
    HALF_INV_ZETA = "half_inv_zeta"
    PA_X = "PA_x"
    PA_Y = "PA_y"
    PA_Z = "PA_z"
    PB_X = "PB_x"
    PB_Y = "PB_y"
    PB_Z = "PB_z"
    # #277: the HGP recurrences, [Fock] eqs 25-26 (VRR) and 28 (HRR)
    WP_X = "WP_x"
    WP_Y = "WP_y"
    WP_Z = "WP_z"
    WQ_X = "WQ_x"
    WQ_Y = "WQ_y"
    WQ_Z = "WQ_z"
    QC_X = "QC_x"
    QC_Y = "QC_y"
    QC_Z = "QC_z"
    INV2P = "inv2p"
    INV2Q = "inv2q"
    INV2PQ = "inv2pq"
    INV2P_ALPHA_OVER_P = "inv2p_alpha_over_p"
    INV2Q_ALPHA_OVER_Q = "inv2q_alpha_over_q"
    AB_X = "AB_x"
    AB_Y = "AB_y"
    AB_Z = "AB_z"
    CD_X = "CD_x"
    CD_Y = "CD_y"
    CD_Z = "CD_z"

    @property
    def name_in_kernel(self) -> str | None:
        return None if self is Factor.ONE else self.value

    @classmethod
    def pq(cls, d: int) -> Factor:
        """`PQ[d]` for d in 0, 1, 2 (x, y, z)."""
        return (cls.PQ_X, cls.PQ_Y, cls.PQ_Z)[d]

    @classmethod
    def pa(cls, d: int) -> Factor:
        """`X_PA` on axis d (`P - A`), for d in 0, 1, 2 (x, y, z)."""
        return (cls.PA_X, cls.PA_Y, cls.PA_Z)[d]

    @classmethod
    def pb(cls, d: int) -> Factor:
        """`X_PB` on axis d (`P - B`), for d in 0, 1, 2 (x, y, z)."""
        return (cls.PB_X, cls.PB_Y, cls.PB_Z)[d]

    @classmethod
    def wp(cls, d: int) -> Factor:
        """`W - P` on axis d, eq 25 (#277)."""
        return (cls.WP_X, cls.WP_Y, cls.WP_Z)[d]

    @classmethod
    def wq(cls, d: int) -> Factor:
        """`W - Q` on axis d, eq 26 (#277)."""
        return (cls.WQ_X, cls.WQ_Y, cls.WQ_Z)[d]

    @classmethod
    def qc(cls, d: int) -> Factor:
        """`Q - C` on axis d, eq 26 (#277)."""
        return (cls.QC_X, cls.QC_Y, cls.QC_Z)[d]

    @classmethod
    def ab(cls, d: int) -> Factor:
        """`A - B` on axis d, eq 28's bra HRR (#277)."""
        return (cls.AB_X, cls.AB_Y, cls.AB_Z)[d]

    @classmethod
    def cd(cls, d: int) -> Factor:
        """`C - D` on axis d, eq 28's ket twin (#277)."""
        return (cls.CD_X, cls.CD_Y, cls.CD_Z)[d]


class Fn(Enum):
    """What a `Call` may call, with its arity. Closed.

    Each member is here because the [p|q] contract (#25) needs it:

    - ADD SUB MUL DIV: index arithmetic, `PQ = P - Q`, `theta^2 = ze/(z+e)`,
      the `D_max * I_p * I_q` bound, the `[p|q] D_q` contraction.
    - LT: the screening test `bound < tau`, whose `If` holds the `Break`.
    - SQRT: `theta` in the fundamental `(2/sqrt(pi)) theta (-2 theta^2)^m F_m`.
    - BOYS: `F_m(T)`, `Call(BOYS, (m, T))` -- the fundamentals' special
      function; the recurrences never reduce it.
    - EXP: `E^{00}_0 = exp(zeta X_PA X_PB)` (`= exp(-mu X_AB^2)`), the seed
      of each axis of the `E` recurrence (#152), formed from the pair
      `(X_PA, X_PB)` exactly as `ao/hermite.py` forms it.
    - THREAD_X THREAD_Y BLOCK_X BLOCK_DIM_X BLOCK_DIM_Y: launch geometry. The
      bra index is `THREAD_X + BLOCK_X * BLOCK_DIM_X`; kets are strip-mined
      from `THREAD_Y` with stride `BLOCK_DIM_Y`.
    - TO_F32 TO_F64: the cast rule (#338) -- the only way a value crosses
      between F32 and F64. A `Call`, not a node of its own, so every pass
      that walks a `Call`'s arguments already sees through it.
    """

    ADD = ("add", 2)
    SUB = ("sub", 2)
    MUL = ("mul", 2)
    DIV = ("div", 2)
    LT = ("lt", 2)
    SQRT = ("sqrt", 1)
    EXP = ("exp", 1)
    BOYS = ("boys", 2)
    THREAD_X = ("thread_x", 0)
    THREAD_Y = ("thread_y", 0)
    BLOCK_X = ("block_x", 0)
    BLOCK_DIM_X = ("block_dim_x", 0)
    BLOCK_DIM_Y = ("block_dim_y", 0)
    TO_F32 = ("to_f32", 1)
    TO_F64 = ("to_f64", 1)

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
        if self.fn is Fn.TO_F32 and isinstance(args[0], Lit) \
                and args[0].type is Scalar.F64 and not is_f32(args[0].value):
            raise PrecisionError(
                f"TO_F32 of the literal {args[0].value!r}, which is not a "
                "float32: write the float32 value, so the constant is what "
                "it says")


def is_f32(v: float) -> bool:
    """True when the float `v` is exactly a float32 (finite, in range)."""
    try:
        return struct.unpack("f", struct.pack("f", v))[0] == v
    except OverflowError:
        return False


@dataclass(frozen=True)
class Term:
    """One `c * g * src` of a `Weighted` sum: `c` a nonzero int (signed since
    #277), `g` one `Factor` -- one scalar, never a product (`Factor`'s
    docstring, "WHY A DERIVED SCALAR")."""

    c: int
    g: Factor
    src: Var | Index

    def __post_init__(self) -> None:
        if type(self.c) is not int or self.c == 0:
            # A zero term is dropped, not emitted: that is the unit-component
            # saving. MMD's reduction form has 1 and r_d - 1 >= 1; HGP's eq 25
            # (#277) adds the signed -e_d of its (m+1) lowering term.
            raise ValueError(f"Term coefficient {self.c!r} is not a nonzero int")
        if not isinstance(self.g, Factor):
            raise ValueError(f"Term factor {self.g!r} is not in the Factor vocabulary")
        _lvalue(self.src, "a Term source")


@dataclass(frozen=True)
class Weighted:
    """The recurrence instruction: `SUM_k c_k * g_k * src_k`.

    `[JEngine, v2, eqs. 10-12]` in reduction form. Interpreted in its
    terms' float type (`expr_type`): F32 when they are F32, else fp64.
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


# ---------------------------------------------------------------------------
# Fragments: a device-code block with a checked binding contract
# ---------------------------------------------------------------------------


def _free_uses(body: tuple[Stmt, ...]) -> tuple[dict[str, str], set[str], set[str]]:
    """What a fragment body uses without declaring it, and what it declares.

    Returns `(free, written, declared)`: `free` maps each name the body uses
    but never declared to `"scalar"` (a `Var` read, or a `Factor`'s name) or
    `"array"` (an `Index` read or written); `written` is the free names
    assigned as scalars; `declared` is every `Decl` name and `For` var. Scope
    is source order and block nesting, as the interpreter and C see it.
    """
    free: dict[str, str] = {}
    written: set[str] = set()
    declared: set[str] = set()
    scopes: list[set[str]] = [set()]

    def visible(name: str) -> bool:
        return any(name in sc for sc in scopes)

    def use(name: str, kind: str) -> None:
        if visible(name):
            return
        if free.setdefault(name, kind) != kind:
            raise ValueError(f"fragment uses {name!r} both as a scalar and as "
                             "an array")

    def read(e: Expr | None) -> None:
        if e is None:
            return
        for x in _subexprs(e):
            if isinstance(x, Var):
                use(x.name, "scalar")
            elif isinstance(x, Index):
                use(x.array, "array")
            elif isinstance(x, Weighted):
                for t in x.terms:
                    if t.g.name_in_kernel is not None:
                        use(t.g.name_in_kernel, "scalar")

    def write(dst: Var | Index) -> None:
        if isinstance(dst, Index):
            read(dst)
        elif not visible(dst.name):
            use(dst.name, "scalar")
            written.add(dst.name)

    def scan(stmts: tuple[Stmt, ...]) -> None:
        for s in stmts:
            if isinstance(s, Inline):
                raise ValueError("a Fragment cannot hold an Inline: one level, "
                                 "names only")
            if isinstance(s, Raw):
                raise ValueError("a Fragment cannot hold a Raw: its names "
                                 "cannot be checked against the binds")
            if isinstance(s, Decl):
                for d in s.shape:
                    read(d)
                read(s.init)
                scopes[-1].add(s.name)
                declared.add(s.name)
            elif isinstance(s, (Assign, WarpReduce)):
                read(s.value)
                write(s.dst)
            elif isinstance(s, AtomicAdd):
                read(s.value)
                read(s.dst)
            elif isinstance(s, For):
                read(s.start)
                read(s.stop)
                read(s.step)
                declared.add(s.var)
                scopes.append({s.var})
                scan(s.body)
                scopes.pop()
            elif isinstance(s, If):
                read(s.cond)
                for branch in (s.then, s.orelse):
                    scopes.append(set())
                    scan(branch)
                    scopes.pop()

    scan(body)
    return free, written, declared


@dataclass(frozen=True)
class Fragment:
    """A named block of device code whose free names are its `binds`, exactly.

    `binds` is the READ set of the body: every name the body uses without
    declaring it -- a `Var`, an indexed array, a `Factor`'s scalar -- is in
    `binds`, and every bind is used. A bind nothing reads is a stale
    contract, and raises. A bind is used either as a scalar (read-only: an
    assignment to one raises) or as an array (indexed, and may be written --
    the one way a fragment hands a result to its host, since its own locals
    end with it).

    One level, names only: no parameters, no return value, no `Inline` inside
    a fragment, and no `Raw` (whose names could not be checked). A `Break`
    must sit inside a `For` of the fragment's own.

    `array_binds` and `declared` (every `Decl` name and `For` var) are
    derived, for `Inline` and `Kernel` to check against; they take no part in
    equality.
    """

    name: str
    binds: tuple[str, ...]
    body: tuple[Stmt, ...]
    array_binds: frozenset[str] = field(init=False, repr=False, compare=False)
    declared: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _ident(self.name, "Fragment name")
        binds = _tuple(self, "binds", self.binds)
        for b in binds:
            _ident(b, "Fragment bind")
        if len(set(binds)) != len(binds):
            raise ValueError(f"duplicate binds in {list(binds)}")
        _body(self, "body", self.body)
        _check_breaks(self.body, False)
        free, written, declared = _free_uses(self.body)
        unbound = sorted(free.keys() - set(binds))
        if unbound:
            raise ValueError(f"fragment {self.name!r} reads {unbound} without "
                             "declaring or binding them")
        unused = [b for b in binds if b not in free]
        if unused:
            raise ValueError(f"fragment {self.name!r} binds {unused}, which "
                             "nothing reads: a stale contract")
        if written:
            raise ValueError(f"fragment {self.name!r} assigns to scalar binds "
                             f"{sorted(written)}: a scalar bind is read-only")
        clash = sorted(declared & set(binds))
        if clash:
            raise ValueError(f"fragment {self.name!r} declares {clash}, "
                             "which it also binds")
        object.__setattr__(self, "array_binds", frozenset(
            b for b in binds if free[b] == "array"))
        object.__setattr__(self, "declared", frozenset(declared))
        # Its own locals are typed even alone; its binds are typed by the
        # host, so here they are unknown and the Kernel checks them (#338).
        _check_types((), self.body, unknown=binds)


@dataclass(frozen=True)
class Inline:
    """Splice `fragment` here, each of its binds bound to an `Expr` of the host.

    `bindings` (a mapping, or `(name, expr)` pairs) binds every name in
    `fragment.binds` exactly once -- none missing, none extra -- and is
    frozen to `(name, expr)` pairs in the fragment's bind order. A scalar bind
    takes any `Expr`; an array bind takes a `Var` naming a host array (here a
    `Var` names the array to alias, not a scalar). The fragment's own names
    are checked against the host's when the `Kernel` is built: a collision
    raises, and nothing is renamed.
    """

    fragment: Fragment
    bindings: tuple[tuple[str, Expr], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fragment, Fragment):
            raise TypeError("Inline needs a Fragment, got "
                            f"{type(self.fragment).__name__}")
        raw = self.bindings
        pairs = list(raw.items()) if hasattr(raw, "items") else list(
            _tuple(self, "bindings", raw))
        given: dict[str, Expr] = {}
        for pair in pairs:
            name, e = pair
            if name in given:
                raise ValueError(f"Inline binds {name!r} twice")
            given[name] = e
        binds = self.fragment.binds
        missing = [b for b in binds if b not in given]
        extra = sorted(given.keys() - set(binds))
        if missing or extra:
            raise ValueError(f"Inline of {self.fragment.name!r}: missing "
                             f"bindings {missing}, extra {extra}")
        for b in binds:
            _expr(given[b], f"the binding of {b!r}")
            if b in self.fragment.array_binds and not isinstance(given[b], Var):
                raise TypeError(f"array bind {b!r} must be bound to a Var "
                                "naming a host array")
        object.__setattr__(self, "bindings",
                           tuple((b, given[b]) for b in binds))


Stmt = (Decl | Assign | For | If | Break | WarpReduce | AtomicAdd | Note | Raw
        | Inline)
STMT_TYPES = (Decl, Assign, For, If, Break, WarpReduce, AtomicAdd, Note, Raw,
              Inline)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """A kernel argument: a scalar, or (when `array`) a global array.

    An OUT parameter is an array; there are no pointers, only names to index.

    `shape` is an array's extents, row-major, when it has more than one
    subscript (`()` leaves a 1-D array's length to the caller). It is an IR
    fact because a backend lays a global array out flat and needs the
    trailing extents to linearise `P[bra, d]`; the interpreter checks every
    input against it at launch (its size must be the extents' product, and
    each subscript is bounds-checked per extent). Extents are
    evaluated at launch, so they may use only I32 literals, scalar `Param`s
    and ADD SUB MUL DIV.
    """

    name: str
    type: Scalar
    array: bool = False
    intent: Intent = Intent.IN
    shape: tuple[Expr, ...] = ()

    def __post_init__(self) -> None:
        _ident(self.name, "Param name")
        if not isinstance(self.type, Scalar):
            raise ValueError(f"Param type {self.type!r} is not a Scalar")
        if not isinstance(self.intent, Intent):
            raise ValueError(f"Param intent {self.intent!r} is not an Intent")
        if self.intent is Intent.OUT and not self.array:
            raise ValueError(f"OUT param {self.name!r} must be an array")
        shape = _tuple(self, "shape", self.shape)
        if shape and not self.array:
            raise ValueError(f"scalar param {self.name!r} takes no shape")
        for d in shape:
            _expr(d, "a Param extent")


def walk(body: tuple[Stmt, ...], *, into_inline: bool = True) -> Iterator[Stmt]:
    """Every statement in `body`, nested ones included, in source order.

    An `Inline` is yielded, then -- unless `into_inline` is false -- its
    fragment's body, so every census sees through the splice.
    """
    for s in body:
        yield s
        if isinstance(s, For):
            yield from walk(s.body, into_inline=into_inline)
        elif isinstance(s, If):
            yield from walk(s.then, into_inline=into_inline)
            yield from walk(s.orelse, into_inline=into_inline)
        elif isinstance(s, Inline) and into_inline:
            yield from walk(s.fragment.body, into_inline=into_inline)


def _exprs_of(s: Stmt) -> tuple[Expr | None, ...]:
    if isinstance(s, Decl):
        return (*s.shape, s.init)
    if isinstance(s, (Assign, WarpReduce, AtomicAdd)):
        return (s.dst, s.value)
    if isinstance(s, For):
        return (s.start, s.stop, s.step)
    if isinstance(s, If):
        return (s.cond,)
    if isinstance(s, Inline):
        return tuple(e for _, e in s.bindings)
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


_EXTENT_FNS = (Fn.ADD, Fn.SUB, Fn.MUL, Fn.DIV)


def _check_inlines(params: tuple[Param, ...], body: tuple[Stmt, ...]) -> None:
    """Hygiene: a fragment is not a macro, so a name clash raises.

    Host names are the `Param`s and every `Decl` name and `For` var outside
    any fragment. A fragment's own `Decl` name or `For` var that is also a
    host name raises. A bind may share a host name only when bound to that
    very name (`Var(b)`): anything else would shadow the host's `b` inside the
    splice. An array bind's `Var` must name a host array, and a scalar bind's
    binding must not be one.
    """
    host_stmts = list(walk(body, into_inline=False))
    host = {p.name for p in params}
    arrays = {p.name for p in params if p.array}
    for s in host_stmts:
        if isinstance(s, Decl):
            host.add(s.name)
            if s.shape:
                arrays.add(s.name)
        elif isinstance(s, For):
            host.add(s.var)
    for s in host_stmts:
        if not isinstance(s, Inline):
            continue
        frag = s.fragment
        clash = sorted(frag.declared & host)
        if clash:
            raise ValueError(
                f"fragment {frag.name!r} declares {clash}, which the host "
                "kernel also names: a fragment is not a macro, and nothing is "
                "renamed")
        for b, e in s.bindings:
            if b in host and e != Var(b):
                raise ValueError(
                    f"fragment {frag.name!r} binds {b!r}, a host name, to "
                    f"{e!r}: only the host's own {b!r} may be bound under "
                    "that name")
            if b in frag.array_binds:
                if e.name not in arrays:
                    raise ValueError(f"array bind {b!r} of {frag.name!r} is "
                                     f"bound to {e.name!r}, not a host array")
            elif isinstance(e, Var) and e.name in arrays:
                raise ValueError(f"scalar bind {b!r} of {frag.name!r} is "
                                 f"bound to the host array {e.name!r}")


# ---------------------------------------------------------------------------
# Precision hygiene (#338): the module docstring's rules, in one place
# ---------------------------------------------------------------------------

_ARITH = (Fn.ADD, Fn.SUB, Fn.MUL, Fn.DIV, Fn.LT)
_CASTS = {Fn.TO_F32: Scalar.F32, Fn.TO_F64: Scalar.F64}


def _mix(a: Scalar | None, b: Scalar | None, what: str) -> Scalar | None:
    """The type of an arithmetic on `a` and `b`, or a refusal."""
    if a is None or b is None:
        return None
    if a is b:
        return a
    if Scalar.I32 in (a, b):
        return b if a is Scalar.I32 else a  # the integer converts, as in C
    raise PrecisionError(f"{what} mixes F32 with F64: say which precision "
                         "with Fn.TO_F32 or Fn.TO_F64")


def expr_type(e: Expr, lookup: Callable[[str], Scalar | None]) -> Scalar | None:
    """The type of `e`, given each name's type by `lookup` (None: unknown).

    Raises `PrecisionError` where the module docstring's PRECISION HYGIENE
    refuses; returns None where an unknown name hides the answer.
    """
    if isinstance(e, Lit):
        return e.type
    if isinstance(e, Var):
        return lookup(e.name)
    if isinstance(e, Index):
        for i in e.at:
            expr_type(i, lookup)
        return lookup(e.array)
    if isinstance(e, Weighted):
        t: Scalar | None = Scalar.I32
        for term in e.terms:
            if term.g.name_in_kernel is not None:
                t = _mix(t, lookup(term.g.name_in_kernel), "a Weighted sum")
            t = _mix(t, expr_type(term.src, lookup), "a Weighted sum")
        return Scalar.F64 if t is Scalar.I32 else t
    if isinstance(e, Call):
        fn = e.fn
        args = [expr_type(a, lookup) for a in e.args]
        if fn in _ARITH:
            t = _mix(args[0], args[1], f"{fn.name}")
            return Scalar.I32 if fn is Fn.LT else t
        if fn in (Fn.SQRT, Fn.EXP):
            (a,) = args
            return Scalar.F64 if a is Scalar.I32 else a
        if fn is Fn.BOYS:
            if args[1] is Scalar.F32:
                raise PrecisionError("Fn.BOYS is fp64 only: there is no fp32 "
                                     "Boys function; TO_F64 its T")
            return Scalar.F64
        if fn in _CASTS:
            to = _CASTS[fn]
            if args[0] is to:
                raise PrecisionError(f"{fn.name} of a value that is already "
                                     f"{to.name}: a cast must change precision")
            return to
        return Scalar.I32  # launch geometry
    raise TypeError(f"not an Expr: {e!r}")  # pragma: no cover


def check_store(dst: Scalar | None, value: Scalar | None, what: str) -> None:
    """Refuse a store the hygiene rules refuse (F32 into anything but F32,
    F64 into F32). The F64/I32 cases are the interpreter's, as before."""
    if dst is None or value is None or dst is value:
        return
    if Scalar.F32 in (dst, value) and value is not Scalar.I32:
        raise PrecisionError(f"{what}: storing {value.name} into {dst.name} "
                             "needs Fn.TO_F32 or Fn.TO_F64")


def _check_types(params: tuple[Param, ...], body: tuple[Stmt, ...],
                 unknown: tuple[str, ...] = ()) -> None:
    """Walk `body` in scope order and apply the hygiene rules to every
    statement. `unknown` names are in scope with no known type."""
    scopes: list[dict[str, Scalar | None]] = [
        {**{n: None for n in unknown}, **{p.name: p.type for p in params}}]

    def lookup(name: str) -> Scalar | None:
        for sc in reversed(scopes):
            if name in sc:
                return sc[name]
        return None

    def ty(e: Expr | None) -> Scalar | None:
        return None if e is None else expr_type(e, lookup)

    def run(stmts: tuple[Stmt, ...], fresh: dict[str, Scalar | None]) -> None:
        scopes.append(fresh)
        try:
            for s in stmts:
                one(s)
        finally:
            scopes.pop()

    def one(s: Stmt) -> None:
        if isinstance(s, Decl):
            for d in s.shape:
                ty(d)
            check_store(s.type, ty(s.init), f"Decl {s.name!r}")
            scopes[-1][s.name] = s.type
        elif isinstance(s, (Assign, WarpReduce, AtomicAdd)):
            check_store(ty(s.dst), ty(s.value), f"{type(s).__name__} to {s.dst!r}")
        elif isinstance(s, For):
            for b in (s.start, s.stop, s.step):
                if ty(b) is Scalar.F32:
                    raise PrecisionError(f"For {s.var!r}: an F32 bound; "
                                         "loop bounds are I32")
            run(s.body, {s.var: Scalar.I32})
        elif isinstance(s, If):
            ty(s.cond)
            run(s.then, {})
            run(s.orelse, {})
        elif isinstance(s, Inline):
            binds: dict[str, Scalar | None] = {}
            for name, e in s.bindings:
                binds[name] = (lookup(e.name) if name in s.fragment.array_binds
                               else ty(e))
            host = scopes[:]
            scopes[:] = [binds]  # the fragment sees its binds and nothing else
            try:
                run(s.fragment.body, {})
            finally:
                scopes[:] = host

    run(body, {})


@dataclass(frozen=True)
class Kernel:
    """One device kernel: a name, its parameters, and an ordered body.

    Validated on construction: parameter names are unique, every `Break` sits
    inside a `For`, and every non-unit `Factor` a `Weighted` uses is bound by
    name to a `Param` or `Decl`, and every statement obeys the precision
    hygiene rules (#338, the module docstring). Frozen and hashable.

    `wide_boys` (#266) is a property of the kernel, not of a call: the domain
    its `Fn.BOYS` must hold on. False (the default, every J-chain kernel) is
    the narrow domain a quartet's `T` stays in; True asks the writer for the
    wide spelling, valid out to the asymptote's range, which a point-charge
    ket (`Target.V`) needs because nothing damps its `T`. The meaning of
    `F_m(T)` is the same either way -- only how far it must hold changes --
    so the interpreter, whose Boys is injected, does not read it. A kernel
    that sets it without calling `Fn.BOYS` is refused: the flag would widen
    nothing.
    """

    name: str
    params: tuple[Param, ...]
    body: tuple[Stmt, ...]
    wide_boys: bool = False

    def __repr__(self) -> str:
        # The dataclass's own spelling, with `wide_boys` shown only when set:
        # every kernel that existed before #266 keeps its repr byte for byte,
        # and with it every hash taken of one (test_launch.py's J plans).
        wide = ", wide_boys=True" if self.wide_boys else ""
        return (f"Kernel(name={self.name!r}, params={self.params!r}, "
                f"body={self.body!r}{wide})")

    def __post_init__(self) -> None:
        _ident(self.name, "Kernel name")
        if type(self.wide_boys) is not bool:
            raise TypeError(f"Kernel wide_boys must be a bool, not "
                            f"{self.wide_boys!r}")
        params = _tuple(self, "params", self.params)
        for p in params:
            if not isinstance(p, Param):
                raise TypeError(f"Kernel params holds {type(p).__name__}, not Param")
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate Param names in {names}")
        scalars = {p.name for p in params if not p.array}
        for p in params:
            for e in (x for d in p.shape for x in _subexprs(d)):
                ok = (isinstance(e, Lit) and e.type is Scalar.I32) or (
                    isinstance(e, Var) and e.name in scalars) or (
                    isinstance(e, Call) and e.fn in _EXTENT_FNS)
                if not ok:
                    raise ValueError(
                        f"Param {p.name!r} extent uses {e!r}: only I32 "
                        "literals, scalar Params and ADD SUB MUL DIV")
        _body(self, "body", self.body)
        _check_breaks(self.body, False)

        _check_inlines(params, self.body)
        _check_types(params, self.body)

        bound = set(names) | {s.name for s in walk(self.body) if isinstance(s, Decl)}
        bound |= {b for s in walk(self.body) if isinstance(s, Inline)
                  for b in s.fragment.binds}
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
        if self.wide_boys and not any(
                isinstance(e, Call) and e.fn is Fn.BOYS
                for s in walk(self.body) for root in _exprs_of(s)
                if root is not None for e in _subexprs(root)):
            raise ValueError(f"Kernel {self.name!r} sets wide_boys but never "
                             "calls Fn.BOYS")
