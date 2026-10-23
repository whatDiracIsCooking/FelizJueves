"""The `fused` lever (#344): produce, consume, discard each `[r]^(0)`.

THE LITERAL KERNEL (`kernel._pq_kernel`) runs the whole recurrence region of
one (bra, ket) pair -- every node a LOCAL scalar, in `schedule`'s order --
and only then contracts

    J_p += SUM_q (-1)^|q| [p+q]^(0) D_q        [JEngine, v2, eqs. 7-9, 20-22]

one shared-memory update per bra function `p`. Every target `[r]^(0)`,
|r| <= L, is therefore live from the statement that forms it to the end of
the region: at the top of the ket loop the kernel holds all of them at once,
which is what spills from the middle of the subgrid up (`kernels.json`).

FUSED, each target is contracted the moment the recurrence forms it: right
after the `Decl` of `[r]^(0)`, one update

    Jsh[slot(p)][lane] = Jsh[slot(p)][lane] +/- [r]^(0) * D[k, j(q)]

per `(p, q)` pair of the batch with `p + q = r` (`kernel.batch_contraction`
names them; for a given `r` and `p` there is at most one `q`). A target is
never a parent -- every DAG parent of `[r]^(m)` has order `m + 1` -- so after
its updates nothing reads it again: ONE TARGET LIVE AT A TIME, the
intermediates `[r]^(m >= 1)` keeping exactly the live ranges they had.
Predecessor: `md/um26/jgen/emit/cuda.py`'s `fused=` (ledger row 7).

WHAT IT CHANGES, and what it does not. The same products are formed and the
same terms accumulated into the same shared slots; the sign is the literal
kernel's `(-1)^|q|`, spelled as an ADD or a SUB. The ORDER of accumulation
is not the same: the literal kernel adds a balanced tree of a slot's terms
to it once per ket, and the fused one adds them one at a time, in the order
the schedule forms their targets. So the lever is NOT bitwise against the
literal kernel; both are held to the same reference at the same tolerance,
and to each other by the summation bound (test/test_kernel_fused.py). The
shared-memory traffic grows from one read-modify-write per `p` per ket to
one per `(p, q)` pair; the register pressure is what it buys.

Everything else -- the screen, the prelude, the warp reductions, the lane-0
write and its `passes.atomize`d form for a split class -- is the literal
kernel's, untouched: the hook replaces only the recurrence-then-contract
tail of the ket loop.

Builder-side and vendor-neutral, like `kernel.py`: IR in, IR out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from felizjueves.ir import Assign, Call, Decl, Expr, Fn, Index, Note, Stmt, Var
from felizjueves.mmd.recurrence import node_name
from felizjueves.plan.targets import R

__all__ = ["FUSED", "FUSED_NOTE", "fuse"]

FUSED = "fused"
"""The lever's name in `Spec.levers` (declared in `levers.VOCABULARY`)."""

FUSED_NOTE = (
    "fused (#344): each [r]^(0) is contracted into J_p += (-1)^|q| [r]^(0) "
    "D_q (r = p + q, [JEngine, v2, eqs. 7-9, 20-22]) right after the "
    "recurrence forms it, and is dead from then on: one target live at a "
    "time; [ab|cd] is never formed"
)

# One batch's contraction, `kernel.batch_contraction`'s shape:
# ((i, p, ((j, q), ...)), ...) with `slot` the position in the tuple.
Contraction = Sequence[tuple[int, R, Sequence[tuple[int, R]]]]


def _updates(touched: Contraction) -> dict[str, list[tuple[int, int, bool]]]:
    """Per target's node name: `(slot, j, odd)` for every pair it feeds."""
    out: dict[str, list[tuple[int, int, bool]]] = {}
    for slot, (_, p, pairs) in enumerate(touched):
        for j, q in pairs:
            r = (p[0] + q[0], p[1] + q[1], p[2] + q[2])
            out.setdefault(node_name((r, 0)), []).append(
                (slot, j, sum(q) % 2 == 1))
    return out


def fuse(recurrence: Sequence[Stmt], touched: Contraction, *,
         jsh: Callable[[int], Index],
         density: Callable[[int], Expr]) -> tuple[Stmt, ...]:
    """`recurrence` with the contraction of `touched` interleaved into it.

    `recurrence` is `recurrence.recurrence_body`'s output (one `Decl` per
    node, in schedule order); `touched` is `kernel.batch_contraction`'s;
    `jsh(slot)` is this lane's shared accumulator for that slot and
    `density(j)` the current ket's `D_q`. Each target's updates follow its
    `Decl` directly, in slot order. Raises when a pair's target is not
    declared in `recurrence` -- the contraction would silently lose a term.
    """
    todo = _updates(touched)
    if not todo:
        raise ValueError("fuse: the batch contracts nothing")
    out: list[Stmt] = [Note(FUSED_NOTE)]
    for s in recurrence:
        out.append(s)
        if not isinstance(s, Decl) or s.name not in todo:
            continue
        for slot, j, odd in todo.pop(s.name):
            dst = jsh(slot)
            term = Call(Fn.MUL, (Var(s.name), density(j)))
            out.append(Assign(dst, Call(Fn.SUB if odd else Fn.ADD,
                                        (dst, term))))
    if todo:
        raise ValueError(f"fuse: targets {sorted(todo)} are contracted but "
                         "never formed by the recurrence")
    return tuple(out)
