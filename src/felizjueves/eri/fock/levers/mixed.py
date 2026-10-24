"""Mixed-precision fused F: the VRR, the boundary and the HRR in FP32 (#355,
milestone 13 lane H12).

The F instance of the precision lever, after #346's `mixed` (J) and #347's
`kmixed` (K), under a name of its own, `fmixed`: `mixed`'s `VOCABULARY`
entry is J's text and is in every J `mixed` unit's provenance `Note`.

WHAT IS FP32 (the issue's scope, the predecessor's ledger row 7):

    region "vrr"       the ket and bra primitive reads (P, PA / QC, zeta, U)
                       narrowed once each, the quartet geometry (PQ, ppq,
                       rho, theta, WP, WQ, the 1/2x factors), `pair`, the
                       Boys prelude (T, S) and every VRR node [e0|f0]^(m)
    region "boundary"  the contracted accumulators c_(e,f) (eq 27), in
                       registers or, under #350's `batched-vrr`, its local
                       `boundary[]` array (which reads `real(ctx,
                       "boundary")` for its type)
    region "hrr"       AB / CD narrowed once each, and every HRR node (eq 28)

WHAT STAYS FP64 (the FP64 boundary): region "fock" -- `G_ab`, the six-block
screen (`dmax`, `bound` against `tau`), `deg`, `s_j` / `s_k`, the bra's own
density block `d_ab`, the weighted densities `w`, the six Fock buffers
`f_*`, and the AtomicAdds into `F` -- and every parameter: the pair records,
`Dsh`, `D` and `F` are `const double*` / `double*` as in the literal kernel,
and the shared `|ab)` staging array stays F64 (each record value is
narrowed at its read). An ERI enters the digestion only through
`TO_F64(H...)` at the `eri` hook, so each multiply-add of the predigest is
`f += w * TO_F64(x)` in fp64.

BOYS STAYS FP64, as in J's and K's levers: `Boys` is fp64 only (the IR
refuses an F32 `T`, and there is no fp32 Boys routine), so the prelude's
`boys` array is filled from `TO_F64(T)` and the fundamental is
`S * TO_F32(boys[m])`, with `T` and `S` formed in fp32. The
constant `2/sqrt(pi)` is the nearest float32.

THE ACCUMULATION IS NOT KEPT FP64, AND THAT IS THE ISSUE'S CHOICE, NOT J'S
INVARIANT. #346 and #347 keep every accumulation fp64; #355 narrows the
boundary accumulation and the HRR on purpose, because the boundary (961
values at `(dd|dd)`) is what the literal kernel spills, and an fp64
boundary with an fp32 VRR would keep it. The error argument, per contracted
ERI x, with u = 2^-24 the fp32 unit roundoff:

  - the VRR at one primitive quartet is a fixed straight-line program of
    depth at most L + 1 sums of products of O(1) terms, so each boundary
    value carries a relative error of a small multiple of (L + c) u of the
    sum of |terms| -- the quantity `hgp_drive`'s absolute-value twin
    bounds;
  - the boundary accumulation sums K_AB K_CD primitive values from 0 in a
    fixed order: |error| <= K_AB K_CD u * sum |values| (recursive summation),
    so at most 128 x 128 x 6e-8 ~ 1e-3 of the |.| sum at the staging cap and
    81 x 6e-8 ~ 5e-6 at nine primitive pairs per side;
  - the HRR (eq 28) is a sum of l_b + l_d >= 0 transfers, each `x + AB_k y`
    on contracted values; its error is u times the |.| HRR of the |.|
    boundary, the same absolute-value twin, so cancellation in eq 28 is
    charged against the scale and not hidden by it.

So |x - x_exact| <= c(L, K) u * scale(x), with the digestion in fp64 adding
nothing at this tier, and `F32_REL = 2^29 * 1e-12 ~ 5.4e-4` of the same
scale_G (~ 9e3 u) is the bound the tests hold it to. The argument says the
HRR is safe in fp32 only because the comparison's scale is the |.| twin; a
scale of max |G| alone would not bound it.

THE HOOKS IT MOVES: `real` and `cast` (the regions' type and their SOURCES:
each record read and each float literal of an F32 region's scalar is
`TO_F32`-ed; a local of the region already has its type -- the seams'
contract, #368), `quartet_scalars` (the Boys prelude `T`, `S`, which the
default emits from `ctx.prelude` in fp64, retyped), `vrr` and `hrr` (every
F64 `Decl` of the recurrence retyped F32, each `boys` read narrowed as
above) and `eri` (`TO_F64` of the prior hook's value). Each wraps the hook
it replaces, so it narrows whatever another lever built there: #350's
grouped nest runs these same hooks on each group's context and reads the
boundary through `real(ctx, "boundary")`. Nothing here decides an
accumulation order: the arithmetic is the literal kernel's, operation for
operation, in the narrower type.

NOT BITWISE against the literal kernel, by construction: held to
`eri.fock.fock`'s every-ordered-quartet sweep at `F32_REL` of `scale_G`
(test/oracle/tolerance.py, never widened), and shown to MISS the fp64 tier
there (the build really is fp32). Without the lever the built kernel is the
literal one, byte for byte (test/test_f_seams.py's digests).

M9's FINDING (#347), TO BE TESTED BY THE CLOCK, NOT HERE: on the MMD side,
compute-only narrowing lost on the clock while winning on spill, the
operand traffic being the suspect. This lever narrows compute only -- the
density reads stay fp64 by the issue's scope -- and the F kernel reads six
density blocks per quartet, so the same regression is the prediction to
test when the arm is timed; nothing in this module measures it.

Like the seams, nothing here imports `ao/`, the interpreter, `reference/`,
a backend or a device.
"""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Callable

from felizjueves.eri.fock.levers.seams import BOYS_ARRAY, FContext, FHooks
from felizjueves.ir import (
    Boys,
    Call,
    Decl,
    Expr,
    Fn,
    Index,
    Lit,
    Note,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.spec import Spec

__all__ = ["F32_REGIONS", "NAME", "NOTE", "apply", "narrow_sources"]

NAME = "fmixed"
"""The lever's name in `Spec.levers` (declared in `levers.VOCABULARY`)."""

F32_REGIONS = frozenset(("vrr", "boundary", "hrr"))
"""The `FHooks.real` regions this lever makes F32; "fock" stays F64."""

NOTE = (
    "fmixed (#355): the VRR, the boundary accumulators and the HRR are F32 "
    "(each record read narrowed once); F_m is evaluated in fp64 on "
    "TO_F64(T) and narrowed. The screen, the densities, the six Fock "
    "buffers, the parameters and the atomics stay F64; each ERI enters the "
    "digestion as TO_F64(x)")


def _round32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _f32(e: Expr) -> Call:
    return Call(Fn.TO_F32, (e,))


def narrow_sources(e: Expr) -> Expr:
    """`e` with its fp64 SOURCES narrowed: each float literal becomes the
    nearest float32 under `TO_F32`, each array read (a record, a field, the
    shared staging) is `TO_F32`-ed, and a `Var` -- a local of an F32 region
    -- is left as it is. A `Var` that is in fact fp64 is not guessed at: the
    IR's hygiene checker refuses the mix when the kernel is built."""
    if isinstance(e, Lit):
        if isinstance(e.value, float):
            return _f32(Lit(_round32(e.value)))
        return e
    if isinstance(e, Var):
        return e
    if isinstance(e, Call):
        if e.fn in (Fn.TO_F32, Fn.TO_F64):
            raise ValueError(f"fmixed: a region source is already cast: {e!r}")
        return Call(e.fn, tuple(narrow_sources(a) for a in e.args))
    return _f32(e)  # an Index or any other read: an fp64 source


def _real(prior: Callable[[FContext, str], Scalar]
          ) -> Callable[[FContext, str], Scalar]:
    def real(ctx: FContext, region: str) -> Scalar:
        return Scalar.F32 if region in F32_REGIONS else prior(ctx, region)
    return real


def _cast(prior: Callable[[FContext, Expr, str], Expr]
          ) -> Callable[[FContext, Expr, str], Expr]:
    def cast(ctx: FContext, e: Expr, region: str) -> Expr:
        e = prior(ctx, e, region)
        return narrow_sources(e) if region in F32_REGIONS else e
    return cast


def _boys32(e: Expr) -> Expr:
    """Every read of the prelude's fp64 `boys[m]` in `e` as `TO_F32` of it."""
    if isinstance(e, Index) and e.array == BOYS_ARRAY:
        return _f32(e)
    if isinstance(e, Call):
        return Call(e.fn, tuple(_boys32(a) for a in e.args))
    return e


def _retype(stmts, what: str) -> tuple[Stmt, ...]:
    """A recurrence's statements with every F64 `Decl` F32 and its `boys`
    reads narrowed. Refuses a statement it does not know how to narrow."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Note):
            out.append(s)
        elif isinstance(s, Decl) and s.type is Scalar.F64 and not s.shape:
            out.append(dataclasses.replace(
                s, type=Scalar.F32,
                init=None if s.init is None else _boys32(s.init)))
        else:
            raise ValueError(f"fmixed: the {what} holds {s!r}; it narrows "
                             "scalar F64 Decls only")
    return tuple(out)


def _vrr(prior: Callable[[FContext], tuple[Stmt, ...]]
         ) -> Callable[[FContext], tuple[Stmt, ...]]:
    def vrr(ctx: FContext) -> tuple[Stmt, ...]:
        return _retype(prior(ctx), "VRR")
    return vrr


def _hrr(prior: Callable[[FContext], tuple[Stmt, ...]]
         ) -> Callable[[FContext], tuple[Stmt, ...]]:
    def hrr(ctx: FContext) -> tuple[Stmt, ...]:
        return _retype(prior(ctx), "HRR")
    return hrr


def _quartet_scalars(prior: Callable[[FContext], tuple[Stmt, ...]]
                     ) -> Callable[[FContext], tuple[Stmt, ...]]:
    """The Boys prelude (`ctx.prelude`: `T`, `S`), fp64 in the default,
    retyped F32 with its literal narrowed, and `NOTE` before it; its `boys`
    array stays fp64 and is filled from `TO_F64(T)` (there is no fp32 Boys).
    Refused when the prior hook did not emit the prelude."""
    def quartet_scalars(ctx: FContext) -> tuple[Stmt, ...]:
        pre = {s for s in ctx.prelude if isinstance(s, Decl) and not s.shape}
        out: list[Stmt] = []
        seen = 0
        for s in prior(ctx):
            if isinstance(s, Decl) and s in pre:
                if seen == 0:
                    out.append(Note(NOTE))
                seen += 1
                out.append(dataclasses.replace(
                    s, type=Scalar.F32, init=narrow_sources(s.init)))
            elif isinstance(s, Boys) and s in ctx.prelude:
                out.append(dataclasses.replace(
                    s, t=Call(Fn.TO_F64, (s.t,))))
            else:
                out.append(s)
        if seen != len(pre):
            raise ValueError(f"{ctx.name}: the Boys prelude is not in "
                             "quartet_scalars; fmixed narrows it there")
        return tuple(out)
    return quartet_scalars


def _eri(prior: Callable[[FContext, object], Expr]
         ) -> Callable[[FContext, object], Expr]:
    def eri(ctx: FContext, t) -> Expr:
        return Call(Fn.TO_F64, (prior(ctx, t),))
    return eri


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """`hooks` with the VRR, boundary and HRR regions F32 (module
    docstring), each moved hook wrapping the one it replaces."""
    return dataclasses.replace(
        hooks, real=_real(hooks.real), cast=_cast(hooks.cast),
        quartet_scalars=_quartet_scalars(hooks.quartet_scalars),
        vrr=_vrr(hooks.vrr), hrr=_hrr(hooks.hrr), eri=_eri(hooks.eri))
