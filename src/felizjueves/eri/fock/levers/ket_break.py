"""The monotone ket key and the early break for the fused F (#352, lane H9).

`kernel_f`'s six-block screen is a SKIP, never a break (pin 8,
`fock/skip-not-break`), because the six-block bound

    bound(P, Q) = G_ab (G_cd dmax(P, Q)),
    dmax(P, Q)  = max(2 Dsh_ab, 2 Dsh_cd, 1/2 Dsh_ac, 1/2 Dsh_ad,
                      1/2 Dsh_bc, 1/2 Dsh_bd)

is not monotone along `launch_f`'s batch sort (`G * dmax`): a screened ket
says nothing of the kets after it. (test/test_f_lever_ket_break.py exhibits
it: a concrete launch of H2O/STO-3G where the literal stream's bound rises
after a skip, so a `break` on the literal stream drops a kept quartet.)

#352 ASKS whether a ket key exists for which a `break` is sound. It does,
through a DOMINATING bound that separates the bra from the ket:

  * THE KEY is `G_cd` alone -- the ket's own Schwarz factor, the record value
    the kernel reads -- with the kets of every batch sorted descending by it,
    stably (`launch_f.f_launches(..., ket_break=True)`). This is `Target.K`'s
    precedent (`kernel_k`: kets sorted by `I_bd` alone, `NO_DENSITY_KEY`,
    because no density key can be formed), not `Target.J`'s product key.
  * THE DOMINATING FACTOR is one number per bra pair `P = (a, b)` and launch
    (ket batch `J`), formed on the host (`walk_f`, `M_ab` below):

        R_s  = max_t Dsh[s][t]                      (a row maximum)
        M_P  = max(|w_ab| Dsh_ab, |w_ac| R_a, |w_ad| R_a,
                   |w_bc| R_b, |w_bd| R_b, |w_cd| max_{(c,d) in J} Dsh_cd)

    Every cross block touches `a` or `b`, so `Dsh_ac <= R_a`, `Dsh_ad <=
    R_a`, `Dsh_bc <= R_b`, `Dsh_bd <= R_b`, and `Dsh_cd` is at most its
    batch maximum: `dmax(P, Q) <= M_P` FOR EVERY KET `Q` OF THE BATCH.

THE PROOF, IN FLOATING POINT. The kernel tests, per ket `k` of a thread's
strided sequence and BEFORE the six-block screen,

    brk_k = Gb (G_cd[k] Mb)      Break when brk_k < tau

in the six-block bound's own operation order. IEEE multiplication rounds
monotonically (`x <= y` gives `fl(c x) <= fl(c y)` for `c >= 0`) and every
term of `M_P` is the matching six-block term's `fl(|w| Dsh)` with its `Dsh`
replaced by a value at least as large, so `fl(|w| Dsh_b) <= M_P` for each of
the six blocks and hence `dmax(P, Q_j) <= M_P` EXACTLY, as floats. With the
kets sorted, `G_cd[j] <= G_cd[k]` for every later ket `j` of the thread, so

    bound(P, Q_j) = fl(Gb fl(G_cd[j] dmax_j)) <= fl(Gb fl(G_cd[k] M_P)) = brk_k

and `brk_k < tau` gives `bound(P, Q_j) < tau`: EVERY KET THE BREAK DROPS IS
ONE THE SIX-BLOCK SCREEN WOULD HAVE SKIPPED, ket `k` included. So on one
sorted stream the lever's kernel computes exactly the literal kernel's
quartets, in the literal order, and its `F` is the literal kernel's BIT FOR
BIT (test/test_f_lever_ket_break.py, interpreted, every F class through
L = 4 and H2O/STO-3G whole); the six-block skip stays, unchanged, after the
break, for the kets the separable bound is too loose to drop. Every skipped
quartet is still held to the exact ERIs (`walk_f` predicts the break,
`FWalk.kernel_broken`, and the test checks each against `ao.eri`).

THE STREAM MUST BE THE SORTED ONE. The proof needs `G_cd` non-increasing
along the ket batch; `walk_f` REFUSES a `ket-break` launch whose ket batch is
not, and the test runs the lever's kernel on an unsorted batch directly to
show the answer is then wrong (a kept quartet is dropped) -- the wrong-
answer test the issue asks for.

WHAT THE BREAK IS, AND IS NOT. It is PER THREAD: each thread leaves its own
strided loop, so a warp diverges where its lanes' kets straddle the cut; the
predecessor's ledger row 4a is the warp-uniform form, a vote this lever does
not add (it would only lose sound exits). It saves the loop's trailing
iterations -- the six-block screen's reads and products -- never quartet
work the literal kernel did: the literal kernel skipped those quartets too.
Whether that is faster on a card is unmeasured (milestone 13 measures
levers separately); the literal default stream and kernel are unchanged.

THE HOOKS IT MOVES: `bra_setup` (appends `Mb = M_ab[bra]`), `screen` (the
break, before whatever screen it was handed) and `params` (`M_ab[n_bra]`
right after `G_ab`). Like the seams, nothing here imports `ao/`, the
interpreter, `reference/`, a backend or a device.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from felizjueves.eri.fock.levers.seams import FContext, FHooks
from felizjueves.ir import Break, Call, Fn, If, Index, Note, Param, Scalar, Stmt, Var
from felizjueves.spec import Spec

__all__ = ["BOUND", "BREAK_NOTE", "MB", "NAME", "apply"]

NAME = "ket-break"
BOUND = "M_ab"  # per bra pair: the dominating density factor over the batch
MB = "Mb"  # its per-bra local
BRK = "brk"

BREAK_NOTE = (
    "#352: the early break. The kets are sorted descending by G_cd and Mb "
    "bounds every ket's six-block dmax, so brk = Gb (G_cd Mb) bounds the "
    "six-block bound of this ket and of every later ket of this thread: "
    "brk < tau ends the loop, dropping only kets the six-block screen would "
    "skip")


def _bra_setup(inner: Callable[[FContext], tuple[Stmt, ...]]):
    def bra_setup(ctx: FContext) -> tuple[Stmt, ...]:
        return (*inner(ctx),
                ctx.scalar(MB, Index(BOUND, (ctx.bra,)), "fock"))
    return bra_setup


def _screen(inner: Callable[[FContext], tuple[Stmt, ...]]):
    def screen(ctx: FContext) -> tuple[Stmt, ...]:
        brk = Call(Fn.MUL, (Var("Gb"), Call(Fn.MUL, (
            ctx.read("G_cd", ctx.k), Var(MB)))))
        return (Note(BREAK_NOTE),
                ctx.scalar(BRK, brk, "fock"),
                If(Call(Fn.LT, (Var(BRK), Var("tau"))), (Break(),)),
                *inner(ctx))
    return screen


def _params(inner: Callable[[FContext], tuple[Param, ...]]):
    def params(ctx: FContext) -> tuple[Param, ...]:
        out = list(inner(ctx))
        at = [p.name for p in out].index("G_ab") + 1
        out.insert(at, Param(BOUND, Scalar.F64, array=True,
                             shape=(Var("n_bra"),)))
        return tuple(out)
    return params


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """`hooks` with the break before the screen, `Mb` read per bra and
    `M_ab` in the signature."""
    return replace(hooks, bra_setup=_bra_setup(hooks.bra_setup),
                   screen=_screen(hooks.screen),
                   params=_params(hooks.params))
