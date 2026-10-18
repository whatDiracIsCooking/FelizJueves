"""Kernel -> Kernel passes: the levers, each checked by interpretation.

    atomize(kernel) -> Kernel

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
"""

from __future__ import annotations

import dataclasses

from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Call,
    Fn,
    For,
    If,
    Index,
    Kernel,
    Lit,
    Note,
    Stmt,
    walk,
)

__all__ = ["ATOMIC_NOTE", "LANE0", "atomize"]

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

    # `replace`, so the call boundary (`bundled`, `wide_boys`) is kept
    return dataclasses.replace(kernel, body=st(kernel.body))
