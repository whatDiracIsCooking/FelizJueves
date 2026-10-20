"""Mixed-precision K, and the `D` / `E` operand narrowing (#347, lane M9).

The K twin of #346's `mixed` (`j_levers/mixed.py`), with the predecessor's
finding attached: on `jgen`, FP32 COMPUTE ALONE regressed the clock (+71% at
`(dp|dp)`) although it won on spill, and the measured cause was the FP64 `D`
and `E` operand traffic; narrowing those operands too flipped it to -73%,
and narrowing only one of them was noise. So this module builds BOTH POINTS,
as two levers, and nothing in between:

    kmixed      THE COMPUTE-ONLY ARM. The per-(bra, ket) pair region is F32
                exactly as in J's `mixed`: `PQ_d` and `theta` narrowed once
                (`TO_F32` of their fp64 values), the prelude `t2`, `T`, `s`,
                `W{m}`, and every `[r]^(m)` node. `F_m` is evaluated in fp64
                on `TO_F64(T)` and narrowed back (`Boys` is fp64 only: the
                IR refuses an F32 `T`, and there is no fp32 Boys routine; a
                narrower `fj_boys` is a separate change, as for J). The
                `E^ac` / `E^bd` blocks, `D_cd` and every parameter stay FP64.
    kmixed-de   THE FULLY NARROWED ARM: `kmixed`, plus the operands. `E` --
                both blocks are formed in F32 by the SAME fragment
                (`e_fragment.block_fragment`, every `Decl` F32, `exp` F32)
                from F32 inputs, into F32 local arrays `Eac` / `Ebd`; the
                fragments' per-record inputs `XPA XPC XQB XQD` arrive as
                `const float*` (the `E`-only operands: `zeta_ac` / `eta` feed
                `theta` too and stay fp64, narrowed in-kernel for `E`). `D` --
                `Dcd` arrives as `const float*`. All five are packed by the
                host as narrowed COPIES (`ao.pairs.narrowed`, one
                round-to-nearest per value; the fp64 arrays are untouched).

THE FP64 BOUNDARY, KEPT IN BOTH ARMS (J's invariant): everything that
ACCUMULATES or CROSSES THE LAUNCH stays F64 -- the screen, the dressings
`s_ac` / `s_bd`, the shared `I[p]` column and `K[a, b]` accumulator, the
stage-1 and stage-2 folds, the `D V` digest, the warp reduction and the
lane-0 (or atomized) write. An F32 value enters an fp64 expression only
through `Fn.TO_F64` at its read -- a target `[p+q]^(0)`, an `E` entry, a
`D_cd` -- and the IR's hygiene checker (#338) refuses any other crossing, so
the boundary is a property of the built `Kernel`, not this module's promise.
So `kmixed-de` narrows STORAGE and TRAFFIC (the five arrays' loads, the two
`E` blocks' registers) and the `E` arithmetic; it does not move a single
accumulation to fp32.

ALL OR NOTHING: there is no `D`-only or `E`-only arm (the issue: half
narrowing was noise), and one spec carrying both levers is refused -- they
are two points of one axis, not two composable changes.

THE HOOKS IT MOVES. `finish` (both arms): the ket loop's pair region
narrowed BY NAME -- every `Decl` of `PQ_d`, `theta`, the prelude and the
recurrence's nodes is replaced by its F32 form, each checked to be the
literal one first (a lever that changed the recurrence would be refused,
not silently narrowed), and every other read of a narrowed name anywhere in
the kernel is wrapped in `TO_F64`. By name and not by position, so it is
right on whatever the other hooks built: #341's `split` prunes unread nodes
BEFORE this `finish` runs (its own `finish` hands the pruned kernel on), and
#343's `interleave` reads the targets and `Dcd` from its own contraction
region. `kmixed-de` also moves `bind_e_ac` / `bind_e_bd` (the F32 fragment,
F32 binds, F32 array) and `e_ac` / `e_bd` (`TO_F64` of the entry), so every
contraction that reads `E` through the hooks -- the default `ket_stage` /
`bra_digest`, `interleave`'s -- reads a widened F32 entry; and its `finish`
moves the five parameters' narrowing to the host (below). It does not use
the `params` hook: a `Kernel` is type-checked whole when it is made, and a
`const float* Dcd` read by the default digest before `finish` widens it
would be refused, so the kernel is first built fp64-clean (the parameters
F64, the fragment binds `TO_F32` of them) and `finish` retypes the
parameters, drops those binds' casts and widens every `Dcd` read, in one
step.

COMPOSITION. With `split` (#341) and `interleave` (#343): LEGAL, by the
above, in either key order; the arithmetic of each is unchanged in fp64
and the narrowed region is the literal one's. A lever that REPLACES
`bind_e_ac` / `e_ac` after this one (#342's staged `E^ac`) would drop
`kmixed-de`'s `E` narrowing for that side; that pairing is not declared
here and is #342's to decide.

NOT BITWISE against the literal kernel, by construction: held to
`ao.k.k_quartet` at THE fp32 tier, `F32_REL` of the quartet's own scale
(test/oracle/tolerance.py), never a tier loosened for it, and shown to
MISS the fp64 tier (the build really is fp32). Without either lever the
built kernel is the literal one, byte for byte (test/test_k_seams.py).

Like the seams, nothing here imports `ao/`, the interpreter, `reference/`,
a backend or a device. It reuses `j_levers.mixed`'s F32 prelude and node
narrowing, so the two engines' fp32 recurrence is one spelling.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from felizjueves import e_fragment as ef
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Boys,
    Call,
    Decl,
    Expr,
    Fn,
    For,
    Fragment,
    If,
    Index,
    Inline,
    Kernel,
    Lit,
    Note,
    Scalar,
    Stmt,
    Var,
    WarpReduce,
    Weighted,
    _subexprs,
)
from felizjueves.j_levers import mixed as jmixed
from felizjueves.k_levers.seams import KET, KContext, KHooks, bindings
from felizjueves.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    recurrence_body,
)
from felizjueves.spec import Spec

__all__ = ["ARMS", "COMPUTE", "COMPUTE_NOTE", "FULL", "FULL_NOTE",
           "NARROWED_PARAMS", "apply", "f32_fragment", "narrow_finish"]

COMPUTE = "kmixed"
"""The compute-only arm's name in `Spec.levers`."""

FULL = "kmixed-de"
"""The fully narrowed arm's name: `kmixed` plus the `D` and `E` operands."""

ARMS = (COMPUTE, FULL)

NARROWED_PARAMS = ("XPA", "XPC", "XQB", "XQD", "Dcd")
"""The `const float*` parameters of `kmixed-de`: the `E` fragments' inputs
and the density. The host packs each as a narrowed copy."""

COMPUTE_NOTE = (
    "kmixed (#347): PQ, theta, T, the [0]^(m) and every [r]^(m) are F32; "
    "F_m is evaluated in fp64 on TO_F64(T) and narrowed. The E blocks, D_cd, "
    "the parameters, the I column, the K accumulator, the reduction and the "
    "write stay F64 (the FP64 boundary); each target is read through TO_F64")

FULL_NOTE = (
    "kmixed-de (#347): as kmixed, and the D and E operands narrowed -- "
    "XPA XPC XQB XQD and Dcd are const float*, packed by the host; E^ac and "
    "E^bd are formed in F32 into F32 arrays. Every accumulation stays F64: "
    "an E entry and a D_cd are read through TO_F64")


def _f64(e: Expr) -> Call:
    return Call(Fn.TO_F64, (e,))


def _f32(e: Expr) -> Call:
    return Call(Fn.TO_F32, (e,))


# ---------------------------------------------------------------------------
# The pair region, narrowed by name (both arms)
# ---------------------------------------------------------------------------


def _region(ctx: KContext) -> tuple[dict[str, Decl], dict[str, Decl]]:
    """`(literal, narrowed)`: the literal kernel's prelude and node `Decl`s
    of this batch by name, and their F32 forms (`j_levers.mixed`'s)."""
    order = list(ctx.order)
    ms = [m for (r, m) in order if not any(r)]
    lit_pre, lit_fun = fundamental_prelude(ms)
    lit_body = recurrence_body(ctx.dag, order, lit_fun)
    f32_pre, f32_fun = jmixed.prelude(ms)
    f32_body = jmixed.narrow(recurrence_body(ctx.dag, order, f32_fun),
                             ctx.dag.targets, widen_targets=False)

    def decls(stmts) -> dict[str, Decl]:
        return {s.name: s for s in stmts if isinstance(s, Decl)}

    literal = decls((*lit_pre, *lit_body))
    narrowed = decls((*f32_pre, *f32_body))
    assert literal.keys() == narrowed.keys()
    return literal, narrowed


def _widen(e: Expr, scalars: frozenset[str], arrays: frozenset[str]) -> Expr:
    """`e` with every read of a narrowed scalar or array wrapped in TO_F64."""
    if isinstance(e, Var):
        return _f64(e) if e.name in scalars else e
    if isinstance(e, Lit):
        return e
    if isinstance(e, Index):
        out = Index(e.array, tuple(_widen(i, scalars, arrays) for i in e.at))
        return _f64(out) if e.array in arrays else out
    if isinstance(e, Call):
        if e.fn is Fn.TO_F64:
            return e  # already widened at its read (an `e_ac` / `e_bd` hook)
        return Call(e.fn, tuple(_widen(a, scalars, arrays) for a in e.args))
    if isinstance(e, Weighted):
        names = {x.name for x in _subexprs(e) if isinstance(x, Var)}
        if names & scalars:
            raise ValueError("a Weighted outside the narrowed region reads "
                             f"{sorted(names & scalars)}: say its precision")
        return e
    raise TypeError(f"cannot widen {type(e).__name__}")


def _lvalue(dst, scalars: frozenset[str], arrays: frozenset[str]):
    """An assignment target: its subscripts widened, the target itself not."""
    if isinstance(dst, Index):
        return Index(dst.array, tuple(_widen(i, scalars, arrays)
                                      for i in dst.at))
    return dst


def _host_packed(e: Expr, packed: frozenset[str]) -> Expr:
    """A bind `TO_F32(P[...])` of a parameter the host now packs in F32:
    the `P[...]` alone (the cast moved to the host)."""
    if (isinstance(e, Call) and e.fn is Fn.TO_F32
            and isinstance(e.args[0], Index) and e.args[0].array in packed):
        return e.args[0]
    return e


def _rewrite(body: tuple[Stmt, ...], *, swap: Callable, scalars: frozenset[str],
             arrays: frozenset[str], packed: frozenset[str],
             in_loop: bool) -> tuple[Stmt, ...]:
    """`body` with the ket loop's narrowed `Decl`s swapped (`swap`) and every
    other statement's reads widened. An `Inline`'s binds are the hook's,
    already in the fragment's precision; only a `TO_F32` of a `packed`
    parameter is dropped there (the host narrowed it)."""
    out: list[Stmt] = []
    rec = dict(swap=swap, scalars=scalars, arrays=arrays, packed=packed)

    def w(e: Expr) -> Expr:
        return _widen(e, scalars, arrays)

    for s in body:
        if in_loop and isinstance(s, Decl) and (new := swap(s)) is not None:
            out.extend(new if isinstance(new, tuple) else (new,))
            continue
        if isinstance(s, Inline):
            s = dataclasses.replace(s, bindings=tuple(
                (b, _host_packed(e, packed)) for b, e in s.bindings))
        elif isinstance(s, Decl):
            s = dataclasses.replace(s, shape=tuple(map(w, s.shape)),
                                    init=None if s.init is None else w(s.init))
        elif isinstance(s, (Assign, WarpReduce, AtomicAdd)):
            s = dataclasses.replace(s, dst=_lvalue(s.dst, scalars, arrays),
                                    value=w(s.value))
        elif isinstance(s, Boys):
            s = dataclasses.replace(s, t=w(s.t))  # TO_F64 of the F32 `T`
        elif isinstance(s, For):
            s = dataclasses.replace(
                s, start=w(s.start), stop=w(s.stop), step=w(s.step),
                body=_rewrite(s.body, **rec, in_loop=in_loop or s.var == KET))
        elif isinstance(s, If):
            s = dataclasses.replace(
                s, cond=w(s.cond),
                then=_rewrite(s.then, **rec, in_loop=in_loop),
                orelse=_rewrite(s.orelse, **rec, in_loop=in_loop))
        out.append(s)
    return tuple(out)


def narrow_finish(ctx: KContext, kernel: Kernel, *, note: str,
                  operands_too: bool) -> Kernel:
    """`kernel` with its pair region F32 (module docstring, "THE HOOKS IT
    MOVES"): each operand, prelude and node `Decl` of the ket loop checked
    against the literal one and replaced by its F32 form, `note` placed
    before the prelude, and every other read of those names widened by
    `TO_F64`. With `operands_too` (`kmixed-de`), also: the
    `NARROWED_PARAMS` become F32 parameters, the fragments' `TO_F32` of
    them is dropped (the host packs them narrowed), and every read of `Dcd`
    is widened."""
    literal, narrowed = _region(ctx)
    operands = set(PQ_PARAMS) | {THETA}
    seen: set[str] = set()

    def swap(s: Decl):
        if s.name in operands:
            if s.type is not Scalar.F64 or s.init is None:
                raise ValueError(f"{s.name}: expected an initialised F64 Decl")
            seen.add(s.name)
            return dataclasses.replace(s, type=Scalar.F32, init=_f32(s.init))
        if s.name not in literal:
            return None
        if s != literal[s.name]:
            raise ValueError(
                f"{ctx.name}: {s.name} is not the literal recurrence's "
                "statement; kmixed narrows only the literal pair region")
        seen.add(s.name)
        new = narrowed[s.name]
        return (Note(note), new) if s.name == "t2" else new

    scalars = frozenset(operands | set(literal))
    packed = frozenset(NARROWED_PARAMS) if operands_too else frozenset()
    arrays = frozenset(("Dcd",)) if operands_too else frozenset()
    body = _rewrite(kernel.body, swap=swap, scalars=scalars, arrays=arrays,
                    packed=packed, in_loop=False)
    missing = sorted((operands | {"t2", "T"}) - seen)
    if missing:
        raise ValueError(f"{ctx.name}: the pair region's {missing} is not "
                         "in the ket loop")
    params = kernel.params
    if operands_too:
        absent = sorted(packed - {p.name for p in params})
        if absent:
            raise ValueError(f"kmixed-de: parameters {absent} are not there")
        params = tuple(dataclasses.replace(p, type=Scalar.F32)
                       if p.name in packed else p for p in params)
    return dataclasses.replace(kernel, params=params, body=body)


# ---------------------------------------------------------------------------
# The E operands, narrowed (kmixed-de)
# ---------------------------------------------------------------------------


def _decl32(s: Stmt) -> Stmt:
    if isinstance(s, Decl):
        if s.type is not Scalar.F64:
            raise ValueError(f"{s.name}: the fragment declares {s.type.name}")
        return dataclasses.replace(s, type=Scalar.F32)
    if isinstance(s, (For, If, Inline)):
        raise ValueError("f32_fragment narrows a straight-line fragment only")
    return s


def f32_fragment(frag: Fragment) -> Fragment:
    """`frag` in F32: every `Decl` F32, nothing else changed -- the same
    terms in the same order, named `<name>_f32`. Refused when the body holds
    a float literal (F64 by `Lit`'s rule: its precision would be a decision
    this function does not make) or anything but straight-line statements."""
    for s in frag.body:
        for root in (getattr(s, "init", None), getattr(s, "value", None)):
            if root is not None and any(
                    isinstance(x, Lit) and isinstance(x.value, float)
                    for x in _subexprs(root)):
                raise ValueError(f"{frag.name}: a float literal in {s!r}")
    return Fragment(frag.name + "_f32", frag.binds,
                    tuple(_decl32(s) for s in frag.body))


def _bind32(frag: Fragment, zeta: Expr, pa: str, pb: str, at: Expr
            ) -> dict[str, Expr]:
    """The fragment's binds, F32: each of the literal's fp64 binds narrowed
    once -- `zeta` and `1/2zeta` (formed in fp64) in-kernel for good; the
    `PA_d` / `PB_d` reads' casts are moved to the host by `narrow_finish`
    (the parameters become the host's narrowed copies)."""
    return {b: _f32(e) for b, e in bindings(frag, zeta, pa, pb, at).items()}


def _bind_e_ac(ctx: KContext) -> tuple[Stmt, ...]:
    frag = f32_fragment(ctx.frag_ac)
    return (
        Note("the undressed E^ac block of this bra record, once, IN F32 "
             "(kmixed-de, #347): e_fragment.block_fragment(la, lc) with "
             "every Decl F32, from the host's narrowed XPA / XPC; s_ac is "
             "the host's, fp64"),
        Decl("Eac", Scalar.F32, (Lit(len(ctx.rows_ac)), Lit(len(ctx.ps)))),
        Inline(frag, {**_bind32(frag, Var("ze"), "XPA", "XPC", ctx.bra),
                      ef.E_ARRAY: Var("Eac")}),
    )


def _bind_e_bd(ctx: KContext) -> tuple[Stmt, ...]:
    frag = f32_fragment(ctx.frag_bd)
    return (
        Note("the undressed E^bd block of this ket record, IN F32 "
             "(kmixed-de, #347): e_fragment.block_fragment(lb, ld) with "
             "every Decl F32, from the host's narrowed XQB / XQD; s_bd is "
             "the host's, fp64"),
        Decl("Ebd", Scalar.F32, (Lit(len(ctx.rows_bd)), Lit(len(ctx.qs)))),
        Inline(frag, {**_bind32(frag, Var("et"), "XQB", "XQD", Var(KET)),
                      ef.E_ARRAY: Var("Ebd")}),
    )


def _e_ac(ctx: KContext, rac: int, i: int) -> Expr:
    return _f64(Index("Eac", (Lit(rac), Lit(i))))


def _e_bd(ctx: KContext, rbd: int, j: int) -> Expr:
    return _f64(Index("Ebd", (Lit(rbd), Lit(j))))


# ---------------------------------------------------------------------------
# The lever
# ---------------------------------------------------------------------------


def _finish(prior: Callable[[KContext, Kernel], Kernel], *, note: str,
            operands_too: bool) -> Callable[[KContext, Kernel], Kernel]:
    def finish(ctx: KContext, kernel: Kernel) -> Kernel:
        return narrow_finish(ctx, prior(ctx, kernel), note=note,
                             operands_too=operands_too)
    return finish


def apply(hooks: KHooks, spec: Spec) -> KHooks:
    """`hooks` with `spec`'s arm: the pair region F32 (`kmixed`), and under
    `kmixed-de` the `D` and `E` operands too. One arm per spec."""
    arms = [a for a in ARMS if a in spec.levers]
    if len(arms) != 1:
        raise ValueError(
            f"{spec.key()}: kmixed and kmixed-de are two points of one "
            "precision axis (#347: all or nothing), not two levers to "
            "compose -- pull one")
    if arms[0] == COMPUTE:
        return dataclasses.replace(hooks, finish=_finish(
            hooks.finish, note=COMPUTE_NOTE, operands_too=False))
    return dataclasses.replace(
        hooks, finish=_finish(hooks.finish, note=FULL_NOTE, operands_too=True),
        bind_e_ac=_bind_e_ac, e_ac=_e_ac, bind_e_bd=_bind_e_bd, e_bd=_e_bd)
