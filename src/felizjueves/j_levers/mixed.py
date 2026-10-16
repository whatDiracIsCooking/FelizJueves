"""The `mixed` lever (#346): an FP32 recurrence inside an FP64 boundary.

THE LITERAL KERNEL (`kernel._pq_kernel`) is fp64 throughout. Under `mixed`
the per-(bra, ket) pair region -- the `[0]^(m)` fundamentals, the Boys
argument `T`, every intermediate `[r]^(m)` and every target `[r]^(0)` -- is
F32, while everything that ACCUMULATES or CROSSES THE LAUNCH stays F64: the
kernel's parameters (positions, exponents, bounds, the density `D_q`, `J`),
the screen, the shared `J_p` accumulator, the warp reduction, the lane-0
write and its `passes.atomize`d form. That is the FP64-BOUNDARY INVARIANT,
and the IR enforces it rather than this module promising it: F32 never meets
F64 without an `Fn.TO_F32` / `Fn.TO_F64` (`ir`'s precision hygiene, #338),
so every crossing below is a cast the reader can find.

WHERE THE CASTS ARE, and nowhere else:

  PQ_d    = TO_F32(P_d - Q_d)               the difference in fp64, then
  theta   = TO_F32(sqrt(ze et / (ze + et)))  narrowed: an fp32 subtraction of
                                             two nearby centres would cancel
  F_m(T)  = TO_F32(BOYS(m, TO_F64(T)))      `Fn.BOYS` is fp64 only (`ir`
                                             refuses an F32 `T`): there is no
                                             fp32 Boys routine, so `T` is
                                             formed in fp32 and widened for
                                             the call, the value narrowed back
  J_p    += TO_F64([p+q]^(0)) * D_q         the contraction, in fp64

So "the fundamentals run in FP32" holds for `W_m = (2/sqrt(pi)) theta
(-2 theta^2)^m` and the product `W_m F_m`, not for the Boys evaluation
itself, which this lever does not narrow (a narrower `fj_boys` is a
separate change to `ir`, the emitters and `reference.boys`). The constants
`-2` and `2/sqrt(pi)` are F32 literals, the latter rounded to the nearest
float32 (`ir` holds a `TO_F32` literal to be exactly one).

COMPOSING WITH `fused` (#344): LEGAL, and neither requires the other. The
fused hook places each target's updates right after the `Decl` that forms
it and reads the target by its node name, in fp64. Under both levers the
F32 target is declared as `<name>f` and immediately widened into an F64
`Decl` of the node's own name, which `fused.fuse` then contracts and which
is dead after its updates: one widened target live at a time, the fused
lever's property kept. Without `fused`, the targets stay F32 until the
contraction block and are widened inside it (`widen`), so the whole set of
targets is held at half the footprint -- the point of the lever.

NOT BITWISE against the literal kernel, by construction; held to the same
reference at the fp32 tier (test/oracle/tolerance.py's `F32_REL`, the one
declared fp32 tolerance), never to a tier loosened for it. Without the lever
the builder's output is the literal kernel's, byte for byte.

Builder-side and vendor-neutral, like `kernel.py`: IR in, IR out.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import replace

from felizjueves.ir import (
    Call,
    Decl,
    Expr,
    Fn,
    Lit,
    Note,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.recurrence import PQ_PARAMS, THETA, Fundamental, node_name

__all__ = ["MIXED", "MIXED_NOTE", "WIDENED", "narrow", "operands", "prelude",
           "widen"]

MIXED = "mixed"
"""The lever's name in `Spec.levers` (declared in `levers.VOCABULARY`)."""

MIXED_NOTE = (
    "mixed (#346): PQ, theta, T, the [0]^(m) and every [r]^(m) are F32; "
    "the parameters, D_q, the J_p accumulator, the reduction and the write "
    "stay F64 (the FP64 boundary); F_m is evaluated in fp64 on TO_F64(T) "
    "and narrowed"
)

WIDENED = "f"
"""Under `fused` too, the F32 target `<node>` is declared as `<node>f` and
widened into an F64 `<node>` for the fused updates."""

# The pair operands the recurrence reads as `Factor`s or as its prelude's
# input: the kernel declares them in fp64; this lever narrows each once.
_NARROWED = (*PQ_PARAMS, THETA)


def _f32(x: float) -> Call:
    """An F32 constant: `x` rounded to the nearest float32, then `TO_F32`."""
    return Call(Fn.TO_F32, (Lit(struct.unpack("f", struct.pack("f", x))[0]),))


def _mul(a: Expr, b: Expr) -> Call:
    return Call(Fn.MUL, (a, b))


def _add(a: Expr, b: Expr) -> Call:
    return Call(Fn.ADD, (a, b))


def operands(stmts: Sequence[Stmt]) -> tuple[Stmt, ...]:
    """The kernel's fp64 operand declarations with `PQ_d` and `theta`
    narrowed: each `Decl` of one of them becomes F32, its fp64 value cast
    once. Raises when one is missing -- the recurrence would read an
    undeclared factor."""
    out, seen = [], set()
    for s in stmts:
        if isinstance(s, Decl) and s.name in _NARROWED:
            if s.type is not Scalar.F64 or s.init is None:
                raise ValueError(f"{s.name}: expected an initialised F64 Decl")
            s = replace(s, type=Scalar.F32,
                        init=Call(Fn.TO_F32, (s.init,)))
            seen.add(s.name)
        out.append(s)
    missing = sorted(set(_NARROWED) - seen)
    if missing:
        raise ValueError(f"mixed: operands {missing} are not declared")
    return tuple(out)


def prelude(ms: Sequence[int]) -> tuple[tuple[Stmt, ...], Fundamental]:
    """`recurrence.fundamental_prelude` in F32: `t2`, `T`, `s` and the `W{m}`
    F32 from the narrowed `theta` and `PQ_d`, and `[0]^(m) = W{m} *
    TO_F32(F_m(TO_F64(T)))`."""
    pqx, pqy, pqz = (Var(n) for n in PQ_PARAMS)
    theta = Var(THETA)
    top = max(ms, default=0)
    stmts: list[Stmt] = [
        Note(MIXED_NOTE),
        Decl("t2", Scalar.F32, init=_mul(theta, theta)),
        Decl("T", Scalar.F32, init=_mul(Var("t2"), _add(
            _add(_mul(pqx, pqx), _mul(pqy, pqy)), _mul(pqz, pqz)))),
    ]
    if top >= 1:  # as the literal prelude: no dead `s` at L = 0 (#62)
        stmts.append(Decl("s", Scalar.F32, init=_mul(_f32(-2.0), Var("t2"))))
    stmts.append(Decl("W0", Scalar.F32,
                      init=_mul(_f32(2.0 / math.sqrt(math.pi)), theta)))
    for m in range(1, top + 1):
        stmts.append(Decl(f"W{m}", Scalar.F32,
                          init=_mul(Var(f"W{m - 1}"), Var("s"))))

    def fundamental(m: int) -> Expr:
        boys = Call(Fn.BOYS, (Lit(m), Call(Fn.TO_F64, (Var("T"),))))
        return _mul(Var(f"W{m}"), Call(Fn.TO_F32, (boys,)))

    return tuple(stmts), fundamental


def narrow(body: Sequence[Stmt], targets: Sequence, *,
           widen_targets: bool) -> tuple[Stmt, ...]:
    """`recurrence.recurrence_body`'s output with every node `Decl` F32.

    `targets` are the batch's target nodes (`BatchDag.targets`). With
    `widen_targets` (the `fused` composition), each target is declared F32
    as `<node>f` and followed by an F64 `Decl` of `<node>` = `TO_F64` of it;
    a target is never a DAG parent (every parent has order m + 1), so the
    rename moves no reader."""
    names = {node_name(t) for t in targets}
    out: list[Stmt] = []
    for s in body:
        if not isinstance(s, Decl):
            out.append(s)
            continue
        if s.type is not Scalar.F64:
            raise ValueError(f"{s.name}: the recurrence declares {s.type.name}")
        if widen_targets and s.name in names:
            f32 = s.name + WIDENED
            out += [replace(s, name=f32, type=Scalar.F32),
                    Decl(s.name, Scalar.F64,
                         init=Call(Fn.TO_F64, (Var(f32),)))]
        else:
            out.append(replace(s, type=Scalar.F32))
    return tuple(out)


def widen(name: str) -> Expr:
    """How the (unfused) contraction reads an F32 target: `TO_F64` of it."""
    return Call(Fn.TO_F64, (Var(name),))
