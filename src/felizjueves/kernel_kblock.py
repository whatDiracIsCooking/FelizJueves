"""The exchange kernel with opt-UM's thread mapping: one block per `K_ab` (#481).

    build_kblock_kernel((la, lc, lb, ld) | spec) -> Kernel   n_batches = 1
    launch_shape(n_blk, lanes)          -> (n_blk, (1, lanes))
    shared_doubles(am) / shared_bytes(am, lanes)       the footprint

A SIBLING of `kernel_k`, not a replacement and not a lever. The arithmetic
is `kernel_k`'s, term for term -- the same `Target.K` class `am = (la, lc,
lb, ld)` (Hermite pairing, bra first), the same plan, DAG, schedule and
recurrence region, the same two `E` fragments, the same stage 1 / fused
stages 2 and 3 (`k_levers.seams`' default hooks, called through a
`KContext`), the same per-pair screen -- and only the MAPPING differs:

  kernel_k     one warp per BRA RECORD; the kets of one `b` shell
               strip-mined over the lanes; one `K[a, b]` row PER BRA RECORD
               written to an `n_rec x n_a n_b` scratch, which the consumer
               sums over a bra shell's records and scatters into `K_mn`.
  kernel_kblock  one block (one warp) per CONTRACTED SHELL BLOCK `K_AB`
               (Palethorpe, Stocks & Barca's opt-UM, [Fock, §4.3]): every
               lane walks the block's whole bra list `[ac|` (every record
               of every `(A, c)` pair, `c` any shell of the class's `lc`),
               and the lanes take distinct ket records `|bd]` (every record
               of every `(B, d)`), strip-mined over `THREAD_Y`. The `K_AB`
               accumulator is the lane's, in the same shared slots
               `kernel_k` uses, zeroed ONCE per block; one warp reduction at
               the end, and lane 0 adds the block into `K_mn` itself, at
               the AO offsets `(ko_a, ko_b)`: no per-record scratch, no
               scatter pass, `n_a n_b` atomics per block.

WHY NOT A LEVER (#481's first question). `k_levers.seams.KHooks` move parts
of the kernel BODY inside `kernel_k`'s fixed skeleton, and the skeleton
fixes three things this mapping changes: the launch shape (`bra = THREAD_X
+ BLOCK_X * BLOCK_DIM_X`, one warp per bra record, `If bra < n_bra`), the
parameters (one bra per warp, where this kernel takes a bra RANGE and a ket
RANGE per block) and the output (a per-record row of a scratch, where this
writes `K_mn`). None of the fifteen hooks reaches the bra index or the loop
around the ket loop, so a lever would have had to replace `params`,
`write_back` and `finish` at once and rebuild the body in `finish` -- a
second kernel wearing the first's name. What IS shared is shared by
import: `KContext` and the default hooks for every statement inside the
ket loop, and `kernel_k.bra_scalars` / `kernel_k.ket_body`, the bra
record's scalars and the per-pair body, so the two kernels cannot drift.

    params  n_blk n_bra n_ket n_dc n_ao (I32)  tau maxD (F64)
            bra0[n_blk] bra1[n_blk] ket0[n_blk] ket1[n_blk]
            ko_a[n_blk] ko_b[n_blk] (I32)                       per block
            P[n_bra, 3]  zeta_ac[n_bra]  XPA[n_bra, 3]  XPC[n_bra, 3]
            s_ac[n_bra]  Ip[n_bra]  dc[n_bra] (I32)             bra side
            Q[n_ket, 3]  eta[n_ket]  XQB[n_ket, 3]  XQD[n_ket, 3]
            s_bd[n_ket]  Iq[n_ket]  dd[n_ket] (I32)             ket side
            Dcd[n_dc, n_c n_d]                                  density
            K[n_ao, n_ao]  (OUT, ADDED INTO)

Block `blk` (`BLOCK_X`) contracts bras `[bra0, bra1)` against kets `[ket0,
ket1)` into `K[ko_a + i, ko_b + j]`. Every name of `kernel_k`'s parameter
list keeps its meaning there (`dc[bra] + dd[k]` is still the `Dcd` row, so
the host lays `Dcd` out exactly as for `kernel_k`).

THE GROUPING IS THE LAUNCH PLAN'S (`launch_kblock.py`): the bras of one
class grouped by first shell `A`, the kets by first shell `B`, each group
SORTED DESCENDING BY ITS BOUND (`Ip`, `Iq`), stably. A block is one
`(A group, B group)` pair; one launch holds each `(A, B)` once.

THE SCREEN, two `break`s on `kernel_k`'s bound `Ib * (Iq * maxD) < tau`:

- the per-pair one, inside the ket loop, `kernel_k`'s statement (the kets
  of a block sorted by `Iq`);
- one more, on the BRA loop: `Ib * (Iq[ket0] * maxD) < tau` -- `Iq[ket0]`
  is the block's largest ket bound, so every pair of this bra and of every
  later (smaller) bra is under `tau` too. It is uniform across the warp
  (no lane-dependent term), so the warp leaves the loop together. A block
  whose FIRST bra fails it does no integral at all: the Schwarz-level
  block skip, `G_A G_B max|D| < tau`, falls out of the sort.

THE BLOCK SCREEN OF [Fock, §4.3] (eqs 41-49, the density-decay fit
`|D_cd| <= H exp(-Lambda |CD|)` and the block test on `|AB| - R_a - R_b`)
is NOT part of this kernel's contract, by decision (#481's design
question): a block it skips is a block the host does not put in the launch,
and the per-block parameters above take any block list, so it is a LAUNCH
FILTER that needs no parameter here. Its in-loop tightenings `G_a -> G_[ac]`,
`G_b -> G_[bd]` are the two sorted `break`s above with the global `maxD`;
a per-block density bound would only replace `maxD`, a scalar, by a
per-block array. Not built; the gain measured first is the mapping alone.

THE WRITE. Lane 0 ADDS each reduced element, `AtomicAdd(K[ko_a + i, ko_b +
j])`. Within one launch no two blocks share an `(A, B)` (the plan's), so
no two atomics of a launch meet; the add is atomic only so that launches of
several classes onto one `K` need no ordering. With launches on one stream
the result is bit-reproducible. The caller zeroes `K` once.

ONE BATCH ONLY: `n_batches != 1` is refused, as is a kernel-body lever.
The partial-sum split adds into the same `K` and would need nothing new
here, but nothing measured asks for it (`batchreport` ships one batch).

SHARED MEMORY is `kernel_k`'s array and footprint, unchanged:
`Ksh[n_p + n_a n_b, BLOCK_DIM_Y]`, `shared_bytes = (n_p + n_a n_b) lanes 8`
-- the I column and the `K_AB` slots per lane (`kernel_k.shared_doubles`).

Interpreted with one lane and one bra record per block, the kernel is `==`
`kernel_k`'s row of that record bit for bit (the same statements in the same
order); with several bras it is that row's sum folded left in bra order, a
reassociation of the per-record rows (test/test_kernel_kblock.py).

Like `kernel_k`, nothing here imports `ao/`, the interpreter, `reference/`,
a backend or a device.
"""

from __future__ import annotations

from felizjueves.ir import (
    AtomicAdd,
    Break,
    Call,
    Decl,
    Expr,
    Fn,
    For,
    If,
    Index,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
    WarpReduce,
)
from felizjueves.k_levers.seams import DEFAULT_HOOKS, KET, KContext
from felizjueves.k_levers.seams import params as k_params
from felizjueves.kernel import builder_plan, split
from felizjueves.kernel_k import (
    NO_DENSITY_KEY,
    bra_scalars,
    k_class_spec,
    ket_body,
)
from felizjueves.kernel_k import shared_bytes as _k_shared_bytes
from felizjueves.kernel_k import shared_doubles as _k_shared_doubles
from felizjueves.spec import Spec, Target

__all__ = [
    "BLOCK_PARAMS",
    "PARAMS",
    "SORTED_GROUPS",
    "build_kblock_kernel",
    "kblock_name",
    "launch_shape",
    "shared_bytes",
    "shared_doubles",
]

BLOCK_PARAMS = ("bra0", "bra1", "ket0", "ket1", "ko_a", "ko_b")

PARAMS = ("n_blk", "n_bra", "n_ket", "n_dc", "n_ao", "tau", "maxD",
          *BLOCK_PARAMS,
          "P", "zeta_ac", "XPA", "XPC", "s_ac", "Ip", "dc",
          "Q", "eta", "XQB", "XQD", "s_bd", "Iq", "dd",
          "Dcd", "K")

SORTED_GROUPS = (
    "PRECONDITION: a block's bras [bra0, bra1) arrive sorted by Ip ALONE and "
    "its kets [ket0, ket1) by Iq ALONE, each descending. Both screens are "
    "breaks: once Ib * Iq * maxD < tau every later ket is under it, and once "
    "Ib * Iq[ket0] * maxD < tau every later bra is. Unsorted groups make a "
    "break a wrong answer, not a slow one (launch_kblock's plan sorts them)")

WRITE_NOTE = (
    "lane 0 ADDS the block into K_mn: one (A, B) per block and per launch, "
    "so no two atomics of a launch meet; atomic only so that launches of "
    "several classes onto one zeroed K need no ordering (#481)")


def kblock_name(am: tuple[int, int, int, int]) -> str:
    la, lc, lb, ld = am
    return f"kb_{la}{lc}_{lb}{ld}"


def _spec_of(cls: tuple[int, int, int, int] | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = k_class_spec(*cls)
    else:
        raise TypeError("build_kblock_kernel takes (la, lc, lb, ld) or a "
                        f"Spec, not {cls!r}")
    if spec.target is not Target.K:
        raise ValueError(f"{spec.key()} is not a Target.K class")
    if spec.n_batches != 1:
        raise ValueError(
            f"n_batches = {spec.n_batches}: the block kernel is built at one "
            "batch only (module docstring, 'ONE BATCH ONLY')")
    if spec.levers:
        raise ValueError(
            f"levers {spec.levers}: the K levers move kernel_k's skeleton; "
            "the block kernel takes none")
    return spec


def shared_doubles(am: tuple[int, int, int, int]) -> int:
    """`kernel_k.shared_doubles`: the I column and the `K_AB` slots."""
    return _k_shared_doubles(am)


def shared_bytes(am: tuple[int, int, int, int], lanes: int) -> int:
    """`kernel_k.shared_bytes`: the same array, the same footprint."""
    return _k_shared_bytes(am, lanes)


def launch_shape(n_blk: int, lanes: int) -> tuple[int, tuple[int, int]]:
    """`(n_blk, (1, lanes))`: one block, one warp, per `K_AB` block."""
    if n_blk < 1 or lanes < 1:
        raise ValueError(f"n_blk = {n_blk}, lanes = {lanes}: both >= 1")
    return n_blk, (1, lanes)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _params(ctx: KContext) -> tuple[Param, ...]:
    """`PARAMS`: the block ranges, then `kernel_k`'s record and density
    parameters unchanged, then `K` as the AO matrix."""
    nblk, nao = Var("n_blk"), Var("n_ao")
    k = {p.name: p for p in k_params(ctx)}
    out = (
        Param("n_blk", Scalar.I32), k["n_bra"], k["n_ket"], k["n_dc"],
        Param("n_ao", Scalar.I32), k["tau"], k["maxD"],
        *(Param(n, Scalar.I32, array=True, shape=(nblk,))
          for n in BLOCK_PARAMS),
        *(k[n] for n in PARAMS[7 + len(BLOCK_PARAMS):-1]),
        Param("K", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nao, nao)),
    )
    assert tuple(p.name for p in out) == PARAMS
    return out


def _write_back(ctx: KContext) -> tuple[Stmt, ...]:
    """One `WarpReduce` per `K[a, b]` of the block, then lane 0 adds each."""
    out: list[Stmt] = [
        Note("reduce K_AB across the warp, once per block: every lane sees "
             "the sum")]
    for s in ctx.k_outs:
        out += [Decl(f"Kab{s}", Scalar.F64),
                WarpReduce(Var(f"Kab{s}"), ctx.sh(ctx.kslot[s]))]
    out += [
        Note(WRITE_NOTE),
        If(_c(Fn.LT, ctx.ty, Lit(1)), tuple(
            AtomicAdd(Index("K", (_c(Fn.ADD, Var("ka"), Lit(s // ctx.nb)),
                                  _c(Fn.ADD, Var("kb"), Lit(s % ctx.nb)))),
                      Var(f"Kab{s}"))
            for s in ctx.k_outs)),
    ]
    return tuple(out)


def build_kblock_kernel(cls: tuple[int, int, int, int] | Spec, *,
                        name: str | None = None) -> Kernel:
    """The exchange kernel of one K class, opt-UM's mapping (module
    docstring). `cls` is `(la, lc, lb, ld)`, the Hermite pairing, bra first,
    or a one-batch, lever-free `Target.K` `Spec`."""
    spec = _spec_of(cls)
    la, lc, lb, ld = spec.am
    lp, lq = split(spec)
    (dag,) = builder_plan(spec).dags
    name = kblock_name(spec.am) if name is None else name
    hooks = DEFAULT_HOOKS
    ctx = KContext.of(spec, dag, name, None, hooks)
    assert ctx.n_shared == shared_doubles(spec.am)
    blk = Var("blk")

    per_bra: list[Stmt] = [
        *bra_scalars(ctx),
        Note("bra screen: Ib * Iq[ket0] * maxD < tau bounds every pair of "
             "this bra and of every later one (both groups sorted) -- so "
             "break; uniform across the warp"),
        If(_c(Fn.LT, _c(Fn.MUL, Var("Ib"), _c(Fn.MUL, Var("Iq0"),
                                              Var("maxD"))), Var("tau")),
           (Break(),)),
        *hooks.bind_e_ac(ctx),
        Note("the block's kets strip-mined over threadIdx.y, stride "
             "blockDim.y"),
        For(KET, _c(Fn.ADD, Var("k0"), ctx.ty), Var("k1"),
            Call(Fn.BLOCK_DIM_Y), ket_body(ctx)),
    ]
    per_blk: list[Stmt] = [
        *(Decl(v, Scalar.I32, init=Index(p, (blk,)))
          for v, p in (("b0", "bra0"), ("b1", "bra1"), ("k0", "ket0"),
                       ("k1", "ket1"), ("ka", "ko_a"), ("kb", "ko_b"))),
        Note("the block's largest ket bound: its kets are sorted, and a "
             "block in the launch has at least one"),
        Decl("Iq0", Scalar.F64, init=Index("Iq", (Var("k0"),))),
        *hooks.init_acc(ctx),
        Note("every lane walks the block's whole bra list"),
        For("bra", Var("b0"), Var("b1"), Lit(1), tuple(per_bra)),
        *_write_back(ctx),
    ]
    body: tuple[Stmt, ...] = (
        Note(f"K[a,b] += D[c,d] SUM_p E^ac_p SUM_q [p|q] E^bd_q, K class "
             f"(la,lc|lb,ld) = ({la},{lc}|{lb},{ld}), lp={lp} lq={lq} "
             f"(L={lp + lq}), one batch: kernel_k's staged contraction on "
             "opt-UM's mapping, one block per contracted K_AB block "
             "([Fock, §4.3]; #481)"),
        Note(SORTED_GROUPS),
        Note(NO_DENSITY_KEY),
        Note("launch: gridDim = n_blk, blockDim = (1, warp width)"),
        Decl("blk", Scalar.I32, init=Call(Fn.BLOCK_X)),
        hooks.shared_decl(ctx),
        If(_c(Fn.LT, blk, Var("n_blk")), tuple(per_blk)),
    )
    return Kernel(name, _params(ctx), body)
