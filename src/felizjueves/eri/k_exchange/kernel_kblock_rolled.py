"""The block exchange kernel, `E` from global memory, the contraction ROLLED (#571).

    build_kblock_rolled_kernel((la, lc, lb, ld) | spec) -> Kernel  n_batches = 1
    build_kblock_dense_kernel((la, lc, lb, ld) | spec)  -> Kernel  (#575)
    e_pattern(l1, l2)        -> (ptr, cols)   one E block's nonzero entries
    hpq_table(am)            -> [n_p][n_q]    the R index of p + q
    tables(am)               -> {name: ints}  every class-constant I32 table
    launch_shape(n_blk, lanes) / shared_doubles(am) / shared_bytes(am, lanes)

A SIBLING of `kernel_kblock`, opt-in, as `kernel_kblock` is beside
`kernel_k`. Its mapping, plan, screen and output are `kernel_kblock`'s
unchanged -- one block (one warp) per contracted `K_AB` block, every lane
walking the block's bras, the lanes taking its kets, the two `break`s on
`Ib * Iq * maxD < tau` with `maxD = dmax[blk]`, one warp reduction and lane
0 adding the block into `K[n_ao, n_ao]` (`launch_kblock.kb_launches` /
`walk_kb` drive it, `test/kblock_driver.py` launches it) -- and so is every
thread's quartet: `kernel_k.bra_scalars` and `kernel_k.ket_body`, the bra
scalars, the screen and the GENERATED recurrence region, shared by import.
Two things move, both through the K seams' hooks:

- `E` IS AN INPUT. Neither `E` fragment is inlined: `E^ac` and `E^bd` of
  every primitive pair arrive in global memory, `Eac[n_bra, n_eac]` and
  `Ebd[n_ket, n_ebd]`, NONZERO ENTRIES ONLY (`t_k <= a_k + c_k` on every
  axis), row-major over `e_pattern`'s CSR order (rows `cartesians(l1) x
  cartesians(l2)`, `l1` outer; within a row the Hermite `t` in
  `hermite_targets(l1 + l2)` order). They are the UNDRESSED `E`, the host's
  `mmd.hermite.e_block` (`launch_kblock.e_table`); `Ebd` carries the ket
  sign `(-1)^|q|`, a negation and so exact. The dressings `s_ac`, `s_bd`
  stay host scalars per record, as in `kernel_k`.
- THE CONTRACTION IS LOOPS. #569 measured the binding constraint at f: the
  compile, not local memory. The unrolled `E^ac_p [p|q] E^bd_q D_cd`
  products grow as `n_ac n_bd n_p n_q` statements (`kb_33_33` is 22 MB of
  source, still in `cicc` after 2.7 h). Here the recurrence's `[r]^(0)` are
  stored into one local `R[n_r]` (`n_r = (L+1)(L+2)(L+3)/6`, Hermite order
  of `hermite_targets(L)`), and the stages run as REAL loops over the class
  tables `tables(am)`:

      for (b, d):                                    rbd = b n_d + d
        for p:  Ksh[p] = SUM_{e in bd row}  R[hpq[p, bd_q[e]]] (s_bd Ebd[k, e])
        for (a, c):                                  rac = a n_c + c
          Ksh[n_p + a n_b + b] += Dcd[dk, c n_d + d]
                                  SUM_{e in ac row} (s_ac Eac[bra, e]) Ksh[ac_p[e]]

  `kernel_k`'s stage order (stage 1 into the shared `I` column, stages 2
  and 3 fused), every fold a left fold from `+0.0` in the same order and
  over the same nonzero terms. So no `n_ac x n_bd` block is ever live: per
  lane `R` (local), `I[n_p]` and `K_AB[n_a n_b]` (shared, `kernel_k`'s
  `Ksh` and footprint unchanged), and the kernel's text is the recurrence
  plus a constant number of loops, whatever the class. The issue's other
  order (`D` contracted into `E^bd` first) keeps an `n_q n_b n_c` block
  live per pair, 8,400 doubles at `(ff|ff)`: not chosen.

    params  n_blk n_bra n_ket n_dc n_ao (I32)  tau (F64)
            bra0[n_blk] bra1[n_blk] ket0[n_blk] ket1[n_blk]
            ko_a[n_blk] ko_b[n_blk] (I32)  dmax[n_blk] (F64)    per block
            P[n_bra, 3]  zeta_ac[n_bra]  s_ac[n_bra]  Ip[n_bra]
            dc[n_bra] (I32)                                     bra side
            Q[n_ket, 3]  eta[n_ket]  s_bd[n_ket]  Iq[n_ket]
            dd[n_ket] (I32)                                     ket side
            Eac[n_bra, n_eac]  Ebd[n_ket, n_ebd]                E, nonzero
            ac_ptr[n_ac + 1]  ac_p[n_eac]  bd_ptr[n_bd + 1]  bd_q[n_ebd]
            hpq[n_p, n_q] (I32)                                 the tables
            Dcd[n_dc, n_c n_d]                                  density
            K[n_ao, n_ao]  (OUT, ADDED INTO)

`kernel_kblock`'s list without `XPA XPC XQB XQD` (no `E` is formed here),
with the `E` values and the five class tables before `Dcd`. The tables are
constants of the class (`tables(am)`); the host uploads them once.

EXACTNESS. Interpreted, the kernel is `kernel_kblock`'s `K` BIT FOR BIT
when its `E` tables are `mmd.hermite.e_block`'s, which the inlined fragment
equals bit for bit (test/test_e_fragment.py): the same terms, in the same
order, with the sign moved from a `SUB` into `Ebd` (`acc - r (s e)` is
`acc + r (s (-e))` exactly). On a card the compilers' contraction into FMAs
differs between the unrolled and the rolled text, so there it is held to
rounding (test/test_kblock_rolled_ondevice.py).

THE LAYOUT IS RECORD-MAJOR by default, `Eac[bra, e]` as `XPA[bra, d]` is:
the lanes of a warp read one bra's row (a broadcast) and distinct kets' rows
of `Ebd` (strided by `n_ebd` doubles, one sector per lane). Opt-in,
`build_kblock_rolled_kernel(..., transposed=True)` (#576, `kbrt_*`) is
`ir.passes.transpose(kernel, ("Ebd",))` of it: `Ebd[n_ebd, n_ket]`, every
`Ebd[k, e]` read as `Ebd[e, k]`, so the warp's consecutive kets read
consecutive doubles; a relayout, the same terms in the same order, so
interpreted it is the default's `K` bit for bit. `Eac` stays record-major
(a broadcast). The default text is unchanged by the option. Measured on
the RTX 3080 (#576, `data/measured/kblock_rolled.json`'s `transpose_clock`):
0.94x the default's time at `(dd|dd)`, neutral at `(ff|ff)`, so the
default stays record-major.

THE DENSE-HERMITE FORM (#575), `build_kblock_dense_kernel` (`kbd_*`), a
third contraction between `kb`'s unrolled text and this rolled one: the
`(b, d)` and `(a, c)` row loops stay rolled, the Hermite `p` and `q` are
UNROLLED, and each row of `E` arrives DENSE over its Hermite index,
structural zeros kept (`Eac[n_bra, n_ac n_p]`, `Ebd[n_ket, n_bd n_q]`,
row-major, `launch_kblock.e_table(..., dense=True)`, the ket's signed).
Every `p + q` is then a constant of the text, so stage 1 reads the
recurrence's own named `[r]^(0)` (no `R`, no `hpq` / `bd_q` / `ac_p`,
none of the five tables), the row's `s_bd E^bd_q` are formed once into
`n_q` locals, and the only runtime-indexed reads are `E[record, row, t]`
and the `Ksh` slots:

    for (b, d):  eq_j = s_bd Ebd[k, rbd n_q + j]        j < n_q
                 Ksh[p] = SUM_j [p + q_j] eq_j           n_p n_q terms
      for (a, c):
                 Ksh[n_p + a n_b + b] += Dcd[dk, c n_d + d]
                     SUM_p (s_ac Eac[bra, rac n_p + p]) Ksh[p]   n_p terms

The text grows as `n_p n_q`, not `n_ac n_bd n_p n_q`: `kbd_33_33` is 0.5 MB
and compiles in seconds (`kb_33_33` is 22 MB and never finished). Its
parameters are `PARAMS` without `TABLES` (`DENSE_PARAMS`). The terms are
`kbr`'s, in `kbr`'s order, plus `[r] * (+-0)` for every structural zero,
which leaves a nonzero sum unchanged: interpreted it equals `kb` and `kbr`
BY VALUE. The price is those zeros' multiplies: 3.75x `kbr`'s contraction
terms at `(dd|dd)`, 4.4x at `(ff|ff)` (`(n_rows n_hermite) / nnz`).

ONE BATCH ONLY, no lever: `kernel_kblock.spec_of`'s refusals.

Like `kernel_kblock`, nothing here imports `ao/`, the interpreter,
`reference/`, a backend or a device; the `E` VALUES are the host's
(`launch_kblock.e_table`), only their structural pattern is here.
"""

from __future__ import annotations

from dataclasses import replace

from felizjueves.eri.k_exchange.kernel_k import bra_scalars, ket_body
from felizjueves.eri.k_exchange.kernel_kblock import (
    BLOCK_PARAMS,
    NO_DENSITY_KEY,
    SORTED_GROUPS,
    WRITE_NOTE,
    launch_shape,
    shared_bytes,
    shared_doubles,
    spec_of,
)
from felizjueves.eri.k_exchange.levers.seams import (
    DEFAULT_HOOKS,
    KET,
    KContext,
    absent,
    plus,
)
from felizjueves.ir import (
    Assign,
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
from felizjueves.ir.passes import transpose
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.kernel import builder_plan, split
from felizjueves.mmd.recurrence import node_name
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Spec

__all__ = [
    "DENSE_PARAMS",
    "TRANSPOSED",
    "E_PARAMS",
    "PARAMS",
    "TABLES",
    "build_kblock_dense_kernel",
    "build_kblock_rolled_kernel",
    "e_pattern",
    "hpq_table",
    "kblock_dense_name",
    "kblock_rolled_name",
    "launch_shape",
    "shared_bytes",
    "shared_doubles",
    "tables",
]

E_PARAMS = ("Eac", "Ebd")
TABLES = ("ac_ptr", "ac_p", "bd_ptr", "bd_q", "hpq")

PARAMS = ("n_blk", "n_bra", "n_ket", "n_dc", "n_ao", "tau",
          *BLOCK_PARAMS,
          "P", "zeta_ac", "s_ac", "Ip", "dc",
          "Q", "eta", "s_bd", "Iq", "dd",
          *E_PARAMS, *TABLES,
          "Dcd", "K")

DENSE_PARAMS = tuple(p for p in PARAMS if p not in TABLES)  # kbd (#575)

TRANSPOSED = ("Ebd",)  # stored [n_ebd, n_ket] by transposed=True (#576)

R = "R"  # the local [r]^(0) array, Hermite order of hermite_targets(L)

ROLLED_NOTE = (
    "the contraction ROLLED (#571): E^ac, E^bd from global memory (nonzero "
    "entries only, e_pattern's CSR order), [r]^(0) stored into R, and "
    "kernel_k's stages as loops over the class tables -- stage 1 into the "
    "shared I column per (b,d) row, then stages 2 and 3 fused per (a,c) "
    "row; no n_ac x n_bd block is live")

DENSE_NOTE = (
    "the contraction DENSE-HERMITE (#575): E^ac, E^bd from global memory, "
    "each (row) dense over its Hermite index, zeros kept; the (b,d) and "
    "(a,c) row loops rolled, p and q unrolled, so every p + q is a constant "
    "and [p+q] the recurrence's own named local -- no R array, no tables")


def kblock_rolled_name(am: tuple[int, int, int, int], *,
                       transposed: bool = False) -> str:
    la, lc, lb, ld = am
    return f"kbr{'t' if transposed else ''}_{la}{lc}_{lb}{ld}"


def e_pattern(l1: int, l2: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """One `E^{l1 l2}` block's nonzero entries as CSR: `ptr[row]` .. `ptr[row
    + 1]` index `cols`, the Hermite index of each entry within
    `hermite_targets(l1 + l2)`. Rows `ef.block_rows(l1, l2)` (`l1` outer)."""
    ts = hermite_targets(l1 + l2)
    ptr, cols = [0], []
    for x, y in ef.block_rows(l1, l2):
        cols += [i for i, t in enumerate(ts) if not absent(t, x, y)]
        ptr.append(len(cols))
    return tuple(ptr), tuple(cols)


def hpq_table(am: tuple[int, int, int, int]) -> tuple[tuple[int, ...], ...]:
    """`hpq[i][j]`: the index in `hermite_targets(L)` (`R`'s order) of `p + q`,
    `p = hermite_targets(la + lc)[i]`, `q = hermite_targets(lb + ld)[j]`."""
    la, lc, lb, ld = am
    at = {r: n for n, r in enumerate(hermite_targets(la + lc + lb + ld))}
    return tuple(tuple(at[plus(p, q)] for q in hermite_targets(lb + ld))
                 for p in hermite_targets(la + lc))


def tables(am: tuple[int, int, int, int]) -> dict[str, tuple[int, ...]]:
    """The five class-constant I32 tables, each flat (`hpq` row-major)."""
    la, lc, lb, ld = am
    ac_ptr, ac_p = e_pattern(la, lc)
    bd_ptr, bd_q = e_pattern(lb, ld)
    return dict(ac_ptr=ac_ptr, ac_p=ac_p, bd_ptr=bd_ptr, bd_q=bd_q,
                hpq=tuple(x for row in hpq_table(am) for x in row))


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _params(spec: Spec) -> tuple[Param, ...]:
    la, lc, lb, ld = spec.am
    nblk, nbra, nket = Var("n_blk"), Var("n_bra"), Var("n_ket")
    ac_ptr, ac_p = e_pattern(la, lc)
    bd_ptr, bd_q = e_pattern(lb, ld)
    n_p, n_q = len(hermite_targets(la + lc)), len(hermite_targets(lb + ld))

    def f64(n: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(n, Scalar.F64, array=True, intent=intent, shape=shape)

    def i32(n: str, *shape: Expr) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    out = (
        *(Param(n, Scalar.I32) for n in ("n_blk", "n_bra", "n_ket", "n_dc",
                                         "n_ao")),
        Param("tau", Scalar.F64),
        *(f64(n, nblk) if n == "dmax" else i32(n, nblk)
          for n in BLOCK_PARAMS),
        f64("P", nbra, Lit(3)), f64("zeta_ac", nbra), f64("s_ac", nbra),
        f64("Ip", nbra), i32("dc", nbra),
        f64("Q", nket, Lit(3)), f64("eta", nket), f64("s_bd", nket),
        f64("Iq", nket), i32("dd", nket),
        f64("Eac", nbra, Lit(len(ac_p))), f64("Ebd", nket, Lit(len(bd_q))),
        i32("ac_ptr", Lit(len(ac_ptr))), i32("ac_p", Lit(len(ac_p))),
        i32("bd_ptr", Lit(len(bd_ptr))), i32("bd_q", Lit(len(bd_q))),
        i32("hpq", Lit(n_p), Lit(n_q)),
        f64("Dcd", Var("n_dc"), Lit(_ncart(lc) * _ncart(ld))),
        f64("K", Var("n_ao"), Var("n_ao"), intent=Intent.OUT),
    )
    assert tuple(p.name for p in out) == PARAMS
    return out


def _ncart(l: int) -> int:  # noqa: E741
    return (l + 1) * (l + 2) // 2


def _store_r(ctx: KContext) -> tuple[Stmt, ...]:
    """In place of `bind_e_bd`: every `[r]^(0)` of the recurrence into `R`."""
    lp, lq = split(ctx.spec)
    ts = hermite_targets(lp + lq)
    held = {r for (r, m) in ctx.order if m == 0}
    missing = [r for r in ts if r not in held]
    assert not missing, f"the recurrence never forms {missing[:3]}..."
    return (
        Note("[r]^(0) into the local R, Hermite order, so the rolled stages "
             "can index it"),
        Decl(R, Scalar.F64, (Lit(len(ts)),)),
        *(Assign(Index(R, (Lit(i),)), Var(node_name((r, 0))))
          for i, r in enumerate(ts)),
    )


def _rolled_contract(ctx: KContext) -> tuple[Stmt, ...]:
    """`kernel_k`'s stage 1 and fused stages 2 and 3, as loops (docstring)."""
    na, nc, nb, nd = (_ncart(x) for x in ctx.spec.am)
    n_p = len(ctx.ps)
    ty = ctx.ty

    def sh(row: Expr) -> Index:
        return Index("Ksh", (row, ty))

    rbd, rac = Var("rbd"), Var("rac")
    stage1 = For("ip", Lit(0), Lit(n_p), Lit(1), (
        Decl("acc", Scalar.F64, init=Lit(0.0)),
        For("ie", Index("bd_ptr", (rbd,)),
            Index("bd_ptr", (_c(Fn.ADD, rbd, Lit(1)),)), Lit(1), (
                Assign(Var("acc"), _c(Fn.ADD, Var("acc"), _c(
                    Fn.MUL,
                    Index(R, (Index("hpq", (Var("ip"), Index(
                        "bd_q", (Var("ie"),)))),)),
                    _c(Fn.MUL, Var("sq"), Index("Ebd", (Var(KET),
                                                        Var("ie"))))))),)),
        Assign(sh(Var("ip")), Var("acc")),
    ))
    slot = _c(Fn.ADD, Lit(n_p), _c(Fn.ADD, _c(Fn.MUL, Var("xa"), Lit(nb)),
                                   Var("xb")))
    digest = For("xa", Lit(0), Lit(na), Lit(1), (For("xc", Lit(0), Lit(nc),
                                                     Lit(1), (
        Decl("rac", Scalar.I32, init=_c(Fn.ADD, _c(Fn.MUL, Var("xa"),
                                                   Lit(nc)), Var("xc"))),
        Decl("v", Scalar.F64, init=Lit(0.0)),
        For("ie", Index("ac_ptr", (rac,)),
            Index("ac_ptr", (_c(Fn.ADD, rac, Lit(1)),)), Lit(1), (
                Assign(Var("v"), _c(Fn.ADD, Var("v"), _c(
                    Fn.MUL,
                    _c(Fn.MUL, Var("sb"), Index("Eac", (ctx.bra, Var("ie")))),
                    sh(Index("ac_p", (Var("ie"),)))))),)),
        Assign(sh(slot), _c(Fn.ADD, sh(slot), _c(
            Fn.MUL, Index("Dcd", (Var("dk"), _c(Fn.ADD, _c(
                Fn.MUL, Var("xc"), Lit(nd)), Var("xd")))), Var("v")))),
    )),))
    return (
        Note(ROLLED_NOTE),
        For("xb", Lit(0), Lit(nb), Lit(1), (For("xd", Lit(0), Lit(nd), Lit(1), (
            Decl("rbd", Scalar.I32, init=_c(Fn.ADD, _c(Fn.MUL, Var("xb"),
                                                       Lit(nd)), Var("xd"))),
            Note("stage 1: I[p] = SUM_q [p+q] (-1)^|q| E^bd_q, the sign in "
                 "Ebd, into the shared column"),
            stage1,
            Note("stages 2 and 3 fused: K[a,b] += D[c,d] SUM_p E^ac_p I[p]"),
            digest,
        )),)),
    )


def _rolled_init(ctx: KContext) -> tuple[Stmt, ...]:
    n_p = len(ctx.ps)
    return (
        Note("zero the lane's K_AB slots, after its I column"),
        For("s", Lit(0), Lit(ctx.n_ab), Lit(1), (
            Assign(Index("Ksh", (_c(Fn.ADD, Lit(n_p), Var("s")), ctx.ty)),
                   Lit(0.0)),)),
    )


def _write_back(ctx: KContext) -> tuple[Stmt, ...]:
    """`kernel_kblock`'s reduction and lane-0 add, rolled over `(a, b)`."""
    n_p, nb = len(ctx.ps), ctx.nb
    na = ctx.n_ab // nb
    return (
        Note("reduce K_AB across the warp, once per block: every lane sees "
             "the sum"),
        Note(WRITE_NOTE),
        For("xa", Lit(0), Lit(na), Lit(1), (For("xb", Lit(0), Lit(nb), Lit(1), (
            Decl("Kab", Scalar.F64),
            WarpReduce(Var("Kab"), Index("Ksh", (_c(Fn.ADD, Lit(n_p), _c(
                Fn.ADD, _c(Fn.MUL, Var("xa"), Lit(nb)), Var("xb"))), ctx.ty))),
            If(_c(Fn.LT, ctx.ty, Lit(1)), (
                AtomicAdd(Index("K", (_c(Fn.ADD, Var("ka"), Var("xa")),
                                      _c(Fn.ADD, Var("kb"), Var("xb")))),
                          Var("Kab")),)),
        )),)),
    )


# ---------------------------------------------------------------------------
# The dense-Hermite form (#575): rows rolled, p and q unrolled over dense E
# ---------------------------------------------------------------------------


def kblock_dense_name(am: tuple[int, int, int, int]) -> str:
    la, lc, lb, ld = am
    return f"kbd_{la}{lc}_{lb}{ld}"


def _dense_params(spec: Spec) -> tuple[Param, ...]:
    """`kbr`'s list without the five tables, `E` dense: `Eac[n_bra, n_ac
    n_p]`, `Ebd[n_ket, n_bd n_q]`."""
    la, lc, lb, ld = spec.am
    n_p, n_q = len(hermite_targets(la + lc)), len(hermite_targets(lb + ld))
    width = {"Eac": _ncart(la) * _ncart(lc) * n_p,
             "Ebd": _ncart(lb) * _ncart(ld) * n_q}
    out = tuple(
        replace(p, shape=(p.shape[0], Lit(width[p.name])))
        if p.name in width else p
        for p in _params(spec) if p.name not in TABLES)
    assert tuple(p.name for p in out) == DENSE_PARAMS
    return out


def _no_r(ctx: KContext) -> tuple[Stmt, ...]:
    """In place of `bind_e_bd`: nothing. Every `p + q` is a constant of the
    unrolled text, so stage 1 reads the recurrence's named `[r]^(0)`."""
    return ()


def _dense_contract(ctx: KContext) -> tuple[Stmt, ...]:
    """`kbr`'s two stages with `p` and `q` unrolled over dense `E` rows
    (module docstring, THE DENSE-HERMITE FORM)."""
    na, nc, nb, nd = (_ncart(x) for x in ctx.spec.am)
    ps, qs = ctx.ps, ctx.qs
    n_p, n_q = len(ps), len(qs)
    ty = ctx.ty

    def sh(row: Expr) -> Index:
        return Index("Ksh", (row, ty))

    def e(j: int) -> str:
        return f"eq{j}"

    rbd, rac = Var("rbd"), Var("rac")
    stage1: list[Stmt] = [
        Decl("eb", Scalar.I32, init=_c(Fn.MUL, rbd, Lit(n_q))),
        Note("the (b,d) row of E^bd, dense over q, the sign in Ebd and the "
             "dressing s_bd applied once per entry"),
        *(Decl(e(j), Scalar.F64, init=_c(Fn.MUL, Var("sq"), Index(
            "Ebd", (Var(KET), _c(Fn.ADD, Var("eb"), Lit(j))))))
          for j in range(n_q)),
        Decl("acc", Scalar.F64),
    ]
    for i, p in enumerate(ps):
        stage1.append(Assign(Var("acc"), Lit(0.0)))
        stage1 += [Assign(Var("acc"), _c(Fn.ADD, Var("acc"), _c(
            Fn.MUL, Var(node_name((plus(p, q), 0))), Var(e(j)))))
            for j, q in enumerate(qs)]
        stage1.append(Assign(sh(Lit(i)), Var("acc")))
    slot = _c(Fn.ADD, Lit(n_p), _c(Fn.ADD, _c(Fn.MUL, Var("xa"), Lit(nb)),
                                   Var("xb")))
    ea = _c(Fn.MUL, rac, Lit(n_p))
    digest = For("xa", Lit(0), Lit(na), Lit(1), (For("xc", Lit(0), Lit(nc),
                                                     Lit(1), (
        Decl("rac", Scalar.I32, init=_c(Fn.ADD, _c(Fn.MUL, Var("xa"),
                                                   Lit(nc)), Var("xc"))),
        Decl("ea", Scalar.I32, init=ea),
        Decl("v", Scalar.F64, init=Lit(0.0)),
        *(Assign(Var("v"), _c(Fn.ADD, Var("v"), _c(
            Fn.MUL,
            _c(Fn.MUL, Var("sb"), Index("Eac", (ctx.bra, _c(
                Fn.ADD, Var("ea"), Lit(i))))),
            sh(Lit(i)))))
          for i in range(n_p)),
        Assign(sh(slot), _c(Fn.ADD, sh(slot), _c(
            Fn.MUL, Index("Dcd", (Var("dk"), _c(Fn.ADD, _c(
                Fn.MUL, Var("xc"), Lit(nd)), Var("xd")))), Var("v")))),
    )),))
    return (
        Note(DENSE_NOTE),
        For("xb", Lit(0), Lit(nb), Lit(1), (For("xd", Lit(0), Lit(nd), Lit(1), (
            Decl("rbd", Scalar.I32, init=_c(Fn.ADD, _c(Fn.MUL, Var("xb"),
                                                       Lit(nd)), Var("xd"))),
            Note(f"stage 1: I[p] = SUM_q [p+q] (-1)^|q| E^bd_q, {n_p} x "
                 f"{n_q} terms unrolled, into the shared column"),
            *stage1,
            Note(f"stages 2 and 3 fused: K[a,b] += D[c,d] SUM_p E^ac_p I[p], "
                 f"{n_p} terms unrolled per (a,c) row"),
            digest,
        )),)),
    )


def build_kblock_dense_kernel(cls: tuple[int, int, int, int] | Spec, *,
                              name: str | None = None) -> Kernel:
    """The dense-Hermite block exchange kernel of one K class (#575; module
    docstring, THE DENSE-HERMITE FORM). `cls` as `build_kblock_rolled_kernel`
    takes it."""
    spec = spec_of(cls)
    name = kblock_dense_name(spec.am) if name is None else name
    return _build(spec, name, _no_r, _dense_contract, _dense_params(spec),
                  "global memory DENSE over its Hermite index, the (a,c) and "
                  "(b,d) rows rolled and p, q unrolled (#575)")


def build_kblock_rolled_kernel(cls: tuple[int, int, int, int] | Spec, *,
                               name: str | None = None,
                               transposed: bool = False) -> Kernel:
    """The rolled block exchange kernel of one K class (module docstring).
    `cls` is `(la, lc, lb, ld)`, the Hermite pairing, bra first, or a
    one-batch, lever-free `Target.K` `Spec` (`kernel_kblock.spec_of`).
    `transposed=True` (#576) stores `Ebd` as `[n_ebd, n_ket]`
    (`ir.passes.transpose`, `kbrt_*`); the default is untouched."""
    spec = spec_of(cls)
    if name is None:
        name = kblock_rolled_name(spec.am, transposed=transposed)
    k = _build(spec, name, _store_r, _rolled_contract, _params(spec),
               "global memory and the contraction rolled (#571)")
    return transpose(k, TRANSPOSED) if transposed else k


def _build(spec: Spec, name: str, bind_e_bd, contract,
           params: tuple[Param, ...], how: str) -> Kernel:
    """The block kernel's skeleton around one contraction form: `kbr`'s and
    `kbd`'s differ only in `bind_e_bd`, `contract` and the parameters."""
    la, lc, lb, ld = spec.am
    lp, lq = split(spec)
    (dag,) = builder_plan(spec).dags
    hooks = replace(DEFAULT_HOOKS, bind_e_bd=bind_e_bd,
                    contract=contract, init_acc=_rolled_init)
    ctx = KContext.of(spec, dag, name, None, hooks)
    assert ctx.n_shared == shared_doubles(spec.am)
    assert ctx.islot == {i: i for i in range(len(ctx.ps))}
    blk = Var("blk")

    per_bra: list[Stmt] = [
        *bra_scalars(ctx),
        Note("bra screen: Ib * Iq[ket0] * maxD < tau bounds every pair of "
             "this bra and of every later one (both groups sorted) -- so "
             "break; uniform across the warp"),
        If(_c(Fn.LT, _c(Fn.MUL, Var("Ib"), _c(Fn.MUL, Var("Iq0"),
                                              Var("maxD"))), Var("tau")),
           (Break(),)),
        Note("the block's kets strip-mined over threadIdx.y, stride "
             "blockDim.y"),
        For(KET, _c(Fn.ADD, Var("k0"), ctx.ty), Var("k1"),
            Call(Fn.BLOCK_DIM_Y), ket_body(ctx)),
    ]
    per_blk: list[Stmt] = [
        *(Decl(v, Scalar.I32, init=Index(p, (blk,)))
          for v, p in (("b0", "bra0"), ("b1", "bra1"), ("k0", "ket0"),
                       ("k1", "ket1"), ("ka", "ko_a"), ("kb", "ko_b"))),
        Note("the block's density factor, read once (#484)"),
        Decl("maxD", Scalar.F64, init=Index("dmax", (blk,))),
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
             f"(L={lp + lq}), one batch: kernel_kblock's mapping with E from "
             + how),
        Note(SORTED_GROUPS),
        Note(NO_DENSITY_KEY),
        Note("launch: gridDim = n_blk, blockDim = (1, warp width)"),
        Decl("blk", Scalar.I32, init=Call(Fn.BLOCK_X)),
        hooks.shared_decl(ctx),
        If(_c(Fn.LT, blk, Var("n_blk")), tuple(per_blk)),
    )
    return Kernel(name, params, body)
