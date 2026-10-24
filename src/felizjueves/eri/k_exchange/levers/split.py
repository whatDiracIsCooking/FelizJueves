"""The per-element split K kernel, `K[a_i, b_j]` (#341, milestone 13 lane M1).

`kernel_k`'s one thread mapping is warp-per-bra with the whole `K[a, b]`
block accumulated in shared memory (`shared_doubles`). This lever is the
register-relief mapping the predecessor ships beside it: ONE KERNEL PER
CARTESIAN ELEMENT `K[a_i, b_j]`, each carrying only that element's
accumulator, `n_a * n_b` kernels tiling the block. Per batch of the plan the
class's kernel fans out into those element kernels, so a split class
(`n_batches > 1`) is at most `n_batches * n_a * n_b` kernels (an element a
batch never reaches is DROPPED by `kernel_k.build_k_kernels`, never emitted
empty), each `passes.atomize`d as every batch kernel is.

THE TRADE, STATED. Every element kernel re-runs the `[p|q]` recurrence and
both `E` blocks for every (bra, ket) pair -- up to `n_a * n_b` times the
recurrence work of the block kernel -- to hold one accumulator instead of
`n_a * n_b`. Only what its element reads is kept: the stage-1 `I[p]` its
stage 2 reads (`KContext.of` drops the rest), so the shared footprint is
`|I rows read| + 1`, and the recurrence nodes those `I[p]` read (`prune`
below drops every other scalar of the ket loop, to a fixed point: an
unread local is an `-Wunused-variable` on hipcc, #92's filter, and work
nvcc would drop anyway). The trade is recorded in each kernel as a `Note`.

THE ARITHMETIC IS THE LITERAL KERNEL'S. Each element kernel performs exactly
the literal kernel's operations for its element, in the literal order: the
same recurrence nodes, the same stage-1 left folds, the same stage-2 fold
over `p`, the same `K += D V` over the `(b, d)` rows; `prune` removes only
declarations nothing reads. The union of the element kernels' writes is the
literal block BIT FOR BIT (test/test_k_lever_split.py), and with a batch
split each element's batches add their partials in the literal batch order.

THE HOOKS IT MOVES: `fan_out` (one variant per element of every variant the
hooks already fan out to, suffixed `_e{i}_{j}`) and, per variant, `keep`
(the one element), `init_acc` (the trade's `Note` before the zeroing) and
`finish` (`prune`, then whatever `finish` it was handed). Nothing else, so a
lever moving the contraction (#343's interleaved `I_p`, whose split arm is
#348) composes with it by construction.

THE INTERFACE (#348 builds on it): `elements(spec)` the `(i, j)` tiling in
emission order, `suffix(i, j)` a variant's name suffix, `keep_element(i, j)`
the `keep` predicate, `prune(kernel)`, and `apply(hooks, spec)`. Like the
seams, nothing here imports `ao/`, the interpreter, `reference/`, a backend
or a device.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable

from felizjueves.eri.k_exchange.levers.seams import (
    KET,
    KContext,
    KHooks,
    ncart,
    one_kernel,
)
from felizjueves.ir import (
    Decl,
    For,
    If,
    Kernel,
    Note,
    Stmt,
    Var,
    _exprs_of,
    _subexprs,
    walk,
)
from felizjueves.spec import Spec

__all__ = ["NAME", "TRADE", "apply", "elements", "keep_element", "prune",
           "suffix"]

NAME = "split"

TRADE = (
    "per-element split (#341): this kernel accumulates ONE element of the "
    "K[a,b] block; the class's n_a*n_b element kernels each re-run the "
    "[p|q] recurrence (the nodes their element reads) and both E blocks for "
    "every (bra, ket) pair -- recurrence work up to n_a*n_b times the block "
    "kernel's, traded for one accumulator in place of n_a*n_b")


def elements(spec: Spec) -> tuple[tuple[int, int], ...]:
    """Every `(a_i, b_j)` of the class's `K` block, row-major: emission order."""
    la, _, lb, _ = spec.am
    return tuple((i, j) for i in range(ncart(la)) for j in range(ncart(lb)))


def suffix(i: int, j: int) -> str:
    """The element kernel's name suffix, after any batch label."""
    return f"_e{i}_{j}"


def keep_element(i: int, j: int) -> Callable[[Spec, int, int], bool]:
    """The `keep` hook that keeps `K[a_i, b_j]` only."""
    def keep(spec: Spec, ia: int, ib: int) -> bool:
        return ia == i and ib == j
    return keep


def _reads(s: Stmt) -> list[str]:
    """Every variable name an expression of `s` mentions (a `Var` written by
    an `Assign` counts too: conservative, nothing written is pruned)."""
    return [e.name for root in _exprs_of(s) if root is not None
            for e in _subexprs(root) if isinstance(e, Var)]


def _prunable(s: Stmt) -> bool:
    return (isinstance(s, Decl) and not s.shape and s.init is not None)


def prune(kernel: Kernel) -> Kernel:
    """`kernel` with every scalar declaration of the ket loop's body that
    nothing reads removed, to a fixed point (a removed declaration's own
    reads are released). Pure expressions only, so no value anything reads
    changes: the arithmetic that remains is the arithmetic that was."""
    count = Counter(n for s in walk(kernel.body) for n in _reads(s))
    dead: set[str] = set()

    def loop_body(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        decls = {s.name: s for s in body if _prunable(s)}
        todo = [n for n in decls if count[n] == 0]
        while todo:
            n = todo.pop()
            if n in dead:
                continue
            dead.add(n)
            for r in _reads(decls[n]):
                count[r] -= 1
                if count[r] == 0 and r in decls:
                    todo.append(r)
        return tuple(s for s in body if not (_prunable(s) and s.name in dead))

    def rebuild(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for s in body:
            if isinstance(s, For) and s.var == KET:
                s = dataclasses.replace(s, body=loop_body(s.body))
            elif isinstance(s, For):
                s = dataclasses.replace(s, body=rebuild(s.body))
            elif isinstance(s, If):
                s = dataclasses.replace(s, then=rebuild(s.then),
                                        orelse=rebuild(s.orelse))
            out.append(s)
        return tuple(out)

    return dataclasses.replace(kernel, body=rebuild(kernel.body))


def _init_acc(prior: Callable[[KContext], tuple[Stmt, ...]]
              ) -> Callable[[KContext], tuple[Stmt, ...]]:
    def init_acc(ctx: KContext) -> tuple[Stmt, ...]:
        return (Note(TRADE), *prior(ctx))
    return init_acc


PLAIN_WRITE = (
    "plain write, NOT atomic: one batch, one warp per bra record, and one "
    "kernel per element (#341), so this kernel is the only writer of its "
    "element of K")


def _restate_write(kernel: Kernel) -> Kernel:
    """The one-batch write-back's `Note` says the kernel is the only writer
    of its ROW of `K`; an element kernel is the only writer of its ELEMENT.
    Restated, the write itself unchanged (a batch kernel's note, the
    partial-sum form, stays true as it is)."""
    def note(s: Stmt) -> Stmt:
        if isinstance(s, Note) and s.prov.startswith("plain write, NOT atomic"):
            return Note(PLAIN_WRITE)
        if isinstance(s, If):
            return dataclasses.replace(s, then=tuple(map(note, s.then)),
                                       orelse=tuple(map(note, s.orelse)))
        return s
    return dataclasses.replace(kernel, body=tuple(map(note, kernel.body)))


def _finish(prior: Callable[[KContext, Kernel], Kernel]
            ) -> Callable[[KContext, Kernel], Kernel]:
    def finish(ctx: KContext, kernel: Kernel) -> Kernel:
        return prior(ctx, _restate_write(prune(kernel)))
    return finish


def apply(hooks: KHooks, spec: Spec) -> KHooks:
    """`hooks` fanned out per element: each variant the prior `fan_out`
    yields becomes `n_a * n_b` variants, each keeping one element."""
    prior = hooks.fan_out

    def fan_out(spec: Spec, h: KHooks) -> tuple[tuple[str, KHooks], ...]:
        out: list[tuple[str, KHooks]] = []
        for pre, hv in prior(spec, h):
            for i, j in elements(spec):
                out.append((pre + suffix(i, j), dataclasses.replace(
                    hv, keep=keep_element(i, j), fan_out=one_kernel,
                    init_acc=_init_acc(hv.init_acc),
                    finish=_finish(hv.finish))))
        return tuple(out)

    return dataclasses.replace(hooks, fan_out=fan_out)
