"""Kernel -> Kernel passes: the levers, each checked by interpretation.

    atomize(kernel) -> Kernel
    split_axes(kernel, axes) -> Kernel

A pass takes a finished `Kernel` and returns another; it never reads a plan
and never writes text. It is checked the way a builder is -- by running both
kernels in the interpreter and comparing what they write -- not by reading
the output. `lower.recycle` is the same shape; this module holds the passes
that change what a kernel MEANS for its caller, where `lower` holds the ones
that only change how it is stored.

ATOMIZE. `kernel.build_pq_kernel` writes `J_p` with a plain `Assign` by lane
0, because a one-batch class has exactly one kernel writing each `J[bra, :]`.
`[JEngine, v2, Alg. 1]` line 10 presents that write as atomic-free; §5
corrects it: once a class is split, the partition is over the targets
`r = p + q`, not over `p`, so several kernels contribute to the same `J_p`
and the write must accumulate atomically. `atomize` makes that one change:

- each `Assign(J[bra, i], Jp{i})` under the lane-0 guard becomes
  `AtomicAdd(J[bra, i], Jp{i})`, same `dst`, same `value`;
- the lane-0 guard STAYS. `WarpReduce` gives every lane the sum, so an
  unguarded atomic would add it once per lane. The atomic is for cross-kernel
  accumulation; the warp reduction is still what makes the global traffic one
  write per bra index per block;
- the provenance `Note`s that said "plain write, NOT atomic" / "no atomics"
  are rewritten to say why the write is atomic now.

It REFUSES (raises) rather than no-ops: a kernel with no such write, a write
into the array outside the lane-0 guard, and a kernel that already atomics
into the array -- so `atomize(atomize(k))` raises. Applying a lever twice is a
caller's mistake, not a fixed point to be found.

An atomic is not ordered: on a card, fp64 atomics accumulate in hardware
order, so a batched `J` is not bit-reproducible run to run. The interpreter
runs blocks in order and cannot see that; a device comparison of a batched
class needs a tolerance for it. Onto a zeroed `J`, a single atomic add of `x`
is `x` exactly, which is what the pass's test asserts.

SPLIT_AXES. Every kernel but `S`, `T`, `V` and `SX` takes a vector per record
interleaved, one `[n, 3]` array (`XPA`, `XPB`, `P`, `Q`), read at a literal
axis: `XPA[k, 0]`. A consumer whose records are three flat arrays, one per
axis -- FreeBreakfast's `pa_x pa_y pa_z`, as the one-electron kernels take
them -- asks for that layout here, one array at a time and under names it
chooses: `split_axes(k, {"XPA": ("pa_x", "pa_y", "pa_z")})`. The pass makes
that one change and no other:

- the `[n, 3]` parameter becomes three 1-D parameters of extent `n`, in its
  place in the list, in axis order, with its type and intent;
- each `XPA[i, d]` becomes `pa_{d}[i]`, the same subscript `i`.

So the work per record is the same arithmetic on the same values, and the
kernel writes the same bits: its test runs both in the interpreter on one
input, laid out each way, and compares with `==`. It is a pass and not an
emitter option because a backend spells a `Kernel` and never rewrites one
(test_layering's `emit` rule); a consumer composes
`gpu.emit(split_axes(k, axes), ...)`, and `emit_template` of the same kernel.
#421 moved these kernels to per-axis arrays by building them so, inside one
by-value `PrimPairSoA` struct, and #440 reverted that; here the builders and
their interleaved default are untouched, there is no struct, and the arrays
stay plain positional parameters.

It REFUSES (raises) rather than guesses: a name that is not an array
parameter of extents `(n, 3)`; an axis subscript that is not a literal 0, 1
or 2; any other use of the array (a whole-array `Var`, as a fragment's array
bind takes it, or a `Member` read); a new name already used in the kernel;
and three names that are not distinct.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Decl,
    Fn,
    For,
    Fragment,
    If,
    Index,
    Inline,
    Kernel,
    Lit,
    Member,
    Note,
    Param,
    Stmt,
    Var,
    walk,
)

__all__ = ["ATOMIC_NOTE", "LANE0", "atomize", "split_axes"]

LANE0 = Call(Fn.LT, (Call(Fn.THREAD_Y), Lit(1)))
"""The lane-0 guard `THREAD_Y < 1` the write-back sits under."""

ATOMIC_NOTE = (
    "atomic write, by lane 0 only: [JEngine, v2, §5] corrects Alg. 1 line 10 "
    "-- a split class partitions r = p + q, not p, so several kernels write "
    "the same J_p and the write must accumulate. The lane-0 guard stays: "
    "every lane holds the warp sum, and an unguarded atomic would add it once "
    "per lane"
)

_PLAIN = "NOT atomic"
_NO_ATOMICS = "no atomics"


def _into(s: Stmt, array: str) -> bool:
    return isinstance(s, (Assign, AtomicAdd)) and isinstance(s.dst, Index) \
        and s.dst.array == array


def atomize(kernel: Kernel, *, array: str = "J") -> Kernel:
    """Lane 0's plain `J[bra, i] = Jp{i}` block becomes an AtomicAdd each.

    Raises `ValueError` when the kernel has no plain write into `array`, when
    such a write is not directly under the lane-0 guard, or when the kernel
    already atomics into `array`.
    """
    if not isinstance(kernel, Kernel):
        raise TypeError(f"atomize takes a Kernel, not {type(kernel).__name__}")
    stmts = list(walk(kernel.body))
    if any(isinstance(s, AtomicAdd) and _into(s, array) for s in stmts):
        raise ValueError(
            f"kernel {kernel.name!r} already atomics into {array!r}: "
            "atomize applies once")
    writes = [s for s in stmts if isinstance(s, Assign) and _into(s, array)]
    if not writes:
        raise ValueError(
            f"kernel {kernel.name!r} has no plain write into {array!r} to atomize")
    guarded = [s for g in stmts if isinstance(g, If) and g.cond == LANE0
               for s in g.then if isinstance(s, Assign) and _into(s, array)]
    if len(guarded) != len(writes):
        raise ValueError(
            f"kernel {kernel.name!r} writes {array!r} outside the lane-0 guard: "
            "every lane holds the warp sum, so an atomic there adds it per lane")

    def note(s: Note) -> Note:
        if _PLAIN in s.prov:
            return Note(ATOMIC_NOTE)
        if _NO_ATOMICS in s.prov:
            return Note(s.prov.replace(
                _NO_ATOMICS, "atomic write-back ([JEngine, v2, §5])"))
        return s

    def st(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for s in body:
            if isinstance(s, If) and s.cond == LANE0:
                out.append(If(s.cond, tuple(
                    AtomicAdd(w.dst, w.value)
                    if isinstance(w, Assign) and _into(w, array) else w
                    for w in st(s.then)), st(s.orelse)))
            elif isinstance(s, If):
                out.append(If(s.cond, st(s.then), st(s.orelse)))
            elif isinstance(s, Note):
                out.append(note(s))
            elif isinstance(s, For):
                out.append(For(s.var, s.start, s.stop, s.step, st(s.body)))
            else:
                out.append(s)
        return tuple(out)

    return Kernel(kernel.name, kernel.params, st(kernel.body))


def _names(kernel: Kernel) -> set[str]:
    """Every name the kernel binds: its params, its Decls, its fragments'."""
    names = {p.name for p in kernel.params}
    for s in walk(kernel.body):
        if isinstance(s, Decl):
            names.add(s.name)
        elif isinstance(s, Inline):
            names |= set(s.fragment.binds) | set(s.fragment.declared)
    return names


def split_axes(kernel: Kernel,
               axes: Mapping[str, tuple[str, str, str]]) -> Kernel:
    """Each `[n, 3]` array parameter in `axes` becomes three flat ones.

    `axes` maps the array's name to its three new names, x, y, z; every
    `array[i, d]` reads or writes `axes[array][d][i]` instead. Raises
    `ValueError` as the module docstring says, before building anything.
    """
    if not isinstance(kernel, Kernel):
        raise TypeError(f"split_axes takes a Kernel, not {type(kernel).__name__}")
    if not isinstance(axes, Mapping) or not axes:
        raise ValueError(f"split_axes takes a nonempty mapping, not {axes!r}")
    params = {p.name: p for p in kernel.params}
    taken = _names(kernel)
    new: list[str] = []
    for name, three in axes.items():
        p = params.get(name)
        if p is None or not p.array or len(p.shape) != 2 \
                or p.shape[1] != Lit(3):
            raise ValueError(f"kernel {kernel.name!r}: {name!r} is not an "
                             "array parameter of extents (n, 3)")
        if not isinstance(three, tuple) or len(three) != 3 \
                or len(set(three)) != 3:
            raise ValueError(f"{name!r} splits into three distinct names, "
                             f"not {three!r}")
        new.extend(three)
    clash = sorted((set(new) & taken) | {n for n in new if new.count(n) > 1})
    if clash:
        raise ValueError(f"kernel {kernel.name!r} already uses {clash}")

    def expr(e: object) -> object:
        if isinstance(e, Member) and e.array in axes:
            raise ValueError(f"{e.array!r} is read as a record, not by axis")
        if isinstance(e, Var) and e.name in axes:
            raise ValueError(f"{e.name!r} is used whole, not by axis")
        if isinstance(e, Index) and e.array in axes:
            if len(e.at) != 2 or not isinstance(e.at[1], Lit) \
                    or e.at[1].type is not Lit(0).type \
                    or e.at[1].value not in (0, 1, 2):
                raise ValueError(f"{e.array!r}'s axis subscript {e.at!r} is "
                                 "not a literal 0, 1 or 2")
            return Index(axes[e.array][e.at[1].value], (expr(e.at[0]),))
        if isinstance(e, Fragment):
            # hygienic: it names host arrays only through its array binds,
            # which are Vars in the Inline and refused above
            return e
        if isinstance(e, tuple):
            return tuple(expr(x) for x in e)
        if dataclasses.is_dataclass(e) and not isinstance(e, type):
            return type(e)(**{f.name: expr(getattr(e, f.name))
                              for f in dataclasses.fields(e)})
        return e

    out: list[Param] = []
    for p in kernel.params:
        if p.name in axes:
            out.extend(Param(n, p.type, array=True, intent=p.intent,
                             shape=(p.shape[0],)) for n in axes[p.name])
        else:
            out.append(p)
    return Kernel(kernel.name, tuple(out), expr(kernel.body))
