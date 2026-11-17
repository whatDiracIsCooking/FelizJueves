"""The full-ERI kernel, `Target.ERI`: every `(mn|ls)` of a class, packed.

    build_eri_kernel(spec | (la, lb, lc, ld)) -> Kernel   one per class
    eri_spec(la, lb, lc, ld)                  -> Spec     the class's key
    quartet_spec(cls)                         -> Spec     the Target.J quartet
    eri_plan(cls)                             -> (dag, order)
    launch_shape(n_task, threads)             -> (grid, block)
    contraction(am) / launch(head, per_task) / input_params(scalars)
                                              the pieces `kernel_dense` shares

`kernel_schwarz`'s kernel, widened from the diagonal of one pair against
itself to the whole block of a bra shell pair against a ket shell pair, by
McMurchie & Davidson's Hermite path (`eri/eri.py`, its witness):

    (ab|cd) = SUM_{bra rec i} SUM_{ket rec k} s_i s_k
              SUM_p E^{ab}_p(i) SUM_q (-1)^{|q|} [p+q]^(0)(P_i - Q_k) E^{cd}_q(k)

THE RECURRENCE is the `Target.J` quartet `(la lb|lc ld)`'s: `eri_plan` asks
`pipeline.plan` for that spec and `recurrence.fundamental_prelude` /
`recurrence_body` spell its one DAG in `schedule`'s order, with every
statement no element reads dropped by `kernel_schwarz._live`'s backward pass.

THE THREAD MAPPING: ONE THREAD PER TASK, a task being one (bra shell pair, ket
shell pair) of the class (`full_eri.packed.eri_classes`: the host chooses the
tasks, so the eight-fold symmetry and any screening are the staging's, not
the kernel's). The thread runs the double loop over the bra pair's records
`[brec0, brec1)` and the ket pair's `[krec0, krec1)` serially -- the bra `E`
block formed once per bra record, the ket's once per (bra, ket) -- and holds
one accumulator per element of the `n_ab x n_cd` block in registers.

THE CONTRACTION, per (bra, ket) record pair, kernel_k's staging:

    I_c[p]   = SUM_q +-[p+q]^(0) Eket[c, q]          per ket row c, every p
    acc_rc  += (s_i s_k) * SUM_p Ebra[r, p] I_c[p]    per element (r, c)

each sum a left fold in `hermite_targets` order over the structurally nonzero
`p` (of row `r`) and `q` (of row `c`), from its first term, odd `|q|`
subtracted. No prefactor: the dressing is in `s`, `(2/sqrt(pi)) theta` in
`[0]^(m)`. ROUND-OFF, NOT BITWISE, against `eri.eri` (the host's `[p|q]` is
`reference.pq`'s recursion and it dresses each `E` entry).

THE SCATTER, PLAIN STORES INTO THE PACKED MATRIX. With `pidx[m, n]` the
host's packed pair index (`packed.pair_index`, symmetric), element `(r, c)` =
`((a, b), (c, d))` goes to

    eri[pidx[oa + a, ob + b], pidx[oc + c, od + d]]  and its mirror,

the mirror being the bra-ket symmetry. A DIAGONAL shell pair (`oa == ob`,
possible only when `la == lb`) writes its lower triangle `a >= b` only, as
`kernel_schwarz`'s does, so `(mn|..)` and `(nm|..)` are one store; the ket
likewise. Every element of the packed matrix is owned by one task, so the
result is bit-reproducible and, the mirror storing the same value, exactly
symmetric -- which `kernel_gemv` relies on.

    params  n_task n_brec n_krec n_bpair n_kpair n_ao n_pp (I32)
            zeta_ab[n_brec] P[n_brec, 3] XPA XPB [n_brec, 3] s_ab[n_brec]
            zeta_cd[n_krec] Q[n_krec, 3] XQC XQD [n_krec, 3] s_cd[n_krec]
            brec0 brec1 off_a off_b [n_bpair]  krec0 krec1 off_c off_d
            [n_kpair]  task_bra task_ket [n_task]  pidx[n_ao, n_ao]  (I32)
            eri[n_pp, n_pp]  (OUT, each owned element written once)

Elements of tasks not launched are untouched: the caller zeroes first.

A builder: vendor-neutral IR. It imports neither `ao/` nor the interpreter
nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.kernel_schwarz import _live, _plus, _reads
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
    Factor,
    Fn,
    For,
    If,
    Index,
    Inline,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "OUT",
    "RECORD_ARRAYS",
    "build_eri_kernel",
    "eri_plan",
    "eri_spec",
    "launch_shape",
    "quartet_spec",
]

TASK = "task"
BRA = "ib"
KET = "ik"
OUT = "eri"

# The parameter names, in order (the driver checks a kernel against them).
SCALARS = ("n_task", "n_brec", "n_krec", "n_bpair", "n_kpair", "n_ao", "n_pp")
RECORD_ARRAYS = ("zeta_ab", "P", "XPA", "XPB", "s_ab",
                 "zeta_cd", "Q", "XQC", "XQD", "s_cd")
INDEX_ARRAYS = ("brec0", "brec1", "off_a", "off_b",
                "krec0", "krec1", "off_c", "off_d",
                "task_bra", "task_ket", "pidx")


def eri_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The quartet class `(la lb|lc ld)` as a generation key, `Target.ERI`."""
    return Spec(Operator.COULOMB, 0, (la, lb, lc, ld), Target.ERI)


def _spec(cls: Spec | tuple[int, int, int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else eri_spec(*cls)
    if spec.target is not Target.ERI:
        raise ValueError(f"{spec.key()} is not a Target.ERI key")
    return spec


def quartet_spec(cls: Spec | tuple[int, int, int, int]) -> Spec:
    """The `Target.J` quartet with the same `am`, whose plan the class runs."""
    return Spec(Operator.COULOMB, 0, _spec(cls).am, Target.J)


def eri_plan(cls: Spec | tuple[int, int, int, int]
             ) -> tuple[BatchDag, Order]:
    """`pipeline.plan(quartet_spec(cls))`'s one DAG and `schedule(dag)`."""
    (dag,) = plan(quartet_spec(cls)).dags
    return dag, schedule(dag)


def launch_shape(n_task: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per task."""
    if type(n_task) is not int or n_task < 1:
        raise ValueError(f"n_task = {n_task!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_task // threads), (threads, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _binds(frag, zeta: Expr, at: Expr, array: str, xa: str,
           xb: str) -> dict[str, Expr]:
    out: dict[str, Expr] = {ef.ZETA: zeta, ef.E_ARRAY: Var(array)}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), zeta)
    for d in range(3):
        out[Factor.pa(d).name_in_kernel] = Index(xa, (at, Lit(d)))
        out[Factor.pb(d).name_in_kernel] = Index(xb, (at, Lit(d)))
    return out


def _reach(row, ts) -> list[tuple[int, tuple[int, int, int]]]:
    a, b = row
    return [(j, t) for j, t in enumerate(ts)
            if all(t[d] <= a[d] + b[d] for d in range(3))]


def contraction(am: tuple[int, int, int, int]) -> tuple[Stmt, ...]:
    """One task's statements up to its scatter: the task's pair indices,
    record ranges and AO offsets (`tb tk b0 b1 k0 k1 oa ob oc od`), the
    accumulators `acc{r}_{c}` of the block `(ab|cd)` (`r` over
    `e_fragment.block_rows(la, lb)`, `c` over `block_rows(lc, ld)`, zeroed),
    and the bra-record loop that fills them (module docstring, THE
    CONTRACTION). `am` is any AO-order quartet: the plan is the `Target.J`
    quartet's with that `am`. Shared with `kernel_dense`, whose scatter
    differs."""
    la, lb, lc, ld = am
    (dag,) = plan(Spec(Operator.COULOMB, 0, am, Target.J)).dags
    order = schedule(dag)
    fbra, fket = ef.block_fragment(la, lb), ef.block_fragment(lc, ld)
    rows, cols = ef.block_rows(la, lb), ef.block_rows(lc, ld)
    tp, tq = hermite_targets(la + lb), hermite_targets(lc + ld)
    task, ib, ik = Var(TASK), Var(BRA), Var(KET)
    reach_r = [_reach(r, tp) for r in rows]
    reach_c = [_reach(c, tq) for c in cols]
    used_p = sorted({i for rr in reach_r for i, _ in rr})

    # --- the contraction of one (bra, ket) record pair ---------------------
    stage1: list[Stmt] = []
    for kc, rc in enumerate(reach_c):
        for i in used_p:
            p = tp[i]
            acc: Expr | None = None
            for j, q in rc:
                term = _c(Fn.MUL, Var(node_name((_plus(p, q), 0))),
                          Index("Eket", (Lit(kc), Lit(j))))
                acc = term if acc is None else _c(
                    Fn.SUB if sum(q) % 2 else Fn.ADD, acc, term)
            stage1.append(Decl(f"I{kc}_{i}", Scalar.F64, init=acc))
    stage2: list[Stmt] = []
    for r, rr in enumerate(reach_r):
        for kc in range(len(cols)):
            v: Expr | None = None
            for i, _ in rr:
                term = _c(Fn.MUL, Index("Ebra", (Lit(r), Lit(i))),
                          Var(f"I{kc}_{i}"))
                v = term if v is None else _c(Fn.ADD, v, term)
            a = Var(f"acc{r}_{kc}")
            stage2.append(Assign(a, _c(Fn.ADD, a, _c(Fn.MUL, Var("w"), v))))
    wanted: set[str] = set()
    for st in stage1:
        wanted |= _reads(st.init)

    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tower = _live((*prelude, *recurrence_body(dag, order, fundamental)),
                  wanted)

    ket_body: tuple[Stmt, ...] = (
        Decl("et", Scalar.F64, init=Index("zeta_cd", (ik,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                      Index("Q", (ik, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Note("theta^2 = zeta eta / (zeta + eta), j_engine.md 'Two defects in "
             "the source'"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ze"), Var("et")),
            _c(Fn.ADD, Var("ze"), Var("et"))))),
        Note(f"[p+q]^(0) by the plan's DAG for the quartet ({la},{lb}|{lc},"
             f"{ld}), statements no element reads dropped"),
        *tower,
        Note("the undressed E block of this ket record: "
             "e_fragment.block_fragment, inlined"),
        Decl("Eket", Scalar.F64, (Lit(len(cols)), Lit(len(tq)))),
        Inline(fket, _binds(fket, Var("et"), ik, "Eket", "XQC", "XQD")),
        Decl("w", Scalar.F64, init=_c(Fn.MUL, Var("sb"),
                                      Index("s_cd", (ik,)))),
        Note("stage 1, per ket row c: I_c[p] = SUM_q +-[p+q] Eket[c, q]"),
        *stage1,
        Note("stage 2, per element (r, c): acc += s_i s_k SUM_p Ebra[r, p] "
             "I_c[p]"),
        *stage2,
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("ze", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Note("the undressed E block of this bra record, once: "
             "e_fragment.block_fragment, inlined"),
        Decl("Ebra", Scalar.F64, (Lit(len(rows)), Lit(len(tp)))),
        Inline(fbra, _binds(fbra, Var("ze"), ib, "Ebra", "XPA", "XPB")),
        Note("the ket records of the task's ket shell pair"),
        For(KET, Var("k0"), Var("k1"), Lit(1), ket_body),
    )

    return (
        Decl("tb", Scalar.I32, init=Index("task_bra", (task,))),
        Decl("tk", Scalar.I32, init=Index("task_ket", (task,))),
        Decl("b0", Scalar.I32, init=Index("brec0", (Var("tb"),))),
        Decl("b1", Scalar.I32, init=Index("brec1", (Var("tb"),))),
        Decl("k0", Scalar.I32, init=Index("krec0", (Var("tk"),))),
        Decl("k1", Scalar.I32, init=Index("krec1", (Var("tk"),))),
        Decl("oa", Scalar.I32, init=Index("off_a", (Var("tb"),))),
        Decl("ob", Scalar.I32, init=Index("off_b", (Var("tb"),))),
        Decl("oc", Scalar.I32, init=Index("off_c", (Var("tk"),))),
        Decl("od", Scalar.I32, init=Index("off_d", (Var("tk"),))),
        *(Decl(f"acc{r}_{kc}", Scalar.F64, init=Lit(0.0))
          for r in range(len(rows)) for kc in range(len(cols))),
        Note("the bra records of the task's bra shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
    )


def launch(head: tuple[Stmt, ...], per_task: tuple[Stmt, ...]
           ) -> tuple[Stmt, ...]:
    """A kernel body: `head`'s notes, the task index, and `per_task` under
    `task < n_task`."""
    return (
        *head,
        Note("launch: one thread per (bra shell pair, ket shell pair) task"),
        Decl(TASK, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(TASK), Var("n_task")), per_task),
    )


def input_params(scalars: tuple[str, ...]) -> tuple[Param, ...]:
    """`scalars` (I32), then the record and pair arrays and the task list,
    in `kernel_eri`'s order -- everything up to its `pidx`."""
    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    nbr, nkr = Var("n_brec"), Var("n_krec")
    nbp, nkp, ntask = Var("n_bpair"), Var("n_kpair"), Var("n_task")
    return (
        *(Param(n, Scalar.I32) for n in scalars),
        f64("zeta_ab", (nbr,)), f64("P", (nbr, Lit(3))),
        f64("XPA", (nbr, Lit(3))), f64("XPB", (nbr, Lit(3))),
        f64("s_ab", (nbr,)),
        f64("zeta_cd", (nkr,)), f64("Q", (nkr, Lit(3))),
        f64("XQC", (nkr, Lit(3))), f64("XQD", (nkr, Lit(3))),
        f64("s_cd", (nkr,)),
        *(i32(n, (nbp,)) for n in INDEX_ARRAYS[:4]),
        *(i32(n, (nkp,)) for n in INDEX_ARRAYS[4:8]),
        i32("task_bra", (ntask,)), i32("task_ket", (ntask,)),
    )


def build_eri_kernel(cls: Spec | tuple[int, int, int, int], *,
                     name: str | None = None) -> Kernel:
    """The full-ERI kernel of one quartet class (module docstring)."""
    spec = _spec(cls)
    if spec.levers:
        raise ValueError(f"{spec.key()}: the full ERI kernel takes no levers")
    la, lb, lc, ld = spec.am

    # --- the scatter --------------------------------------------------------
    nb, nd = len(ef.cartesians(lb)), len(ef.cartesians(ld))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    def lower(n_first: int, n_second: int, diag: bool) -> list[int]:
        return [x * n_second + y for x in range(n_first)
                for y in range(n_second) if not (diag and y > x)]

    all_r = lower(len(ef.cartesians(la)), nb, False)
    low_r = lower(len(ef.cartesians(la)), nb, True)
    all_c = lower(len(ef.cartesians(lc)), nd, False)
    low_c = lower(len(ef.cartesians(lc)), nd, True)

    def stores(rs: list[int], cs: list[int]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for r in rs:
            for kc in cs:
                v = Var(f"acc{r}_{kc}")
                out += [Assign(Index(OUT, (Var(f"pr{r}"), Var(f"pc{kc}"))), v),
                        Assign(Index(OUT, (Var(f"pc{kc}"), Var(f"pr{r}"))), v)]
        return tuple(out)

    def differ(x: str, y: str) -> Expr:
        return _c(Fn.ADD, _c(Fn.LT, Var(x), Var(y)), _c(Fn.LT, Var(y), Var(x)))

    def ket_split(rs: list[int]) -> tuple[Stmt, ...]:
        if lc != ld:
            return stores(rs, all_c)
        return (If(differ("oc", "od"), stores(rs, all_c),
                   (Note("diagonal ket shell pair: its lower triangle only"),
                    *stores(rs, low_c))),)

    scatter: tuple[Stmt, ...]
    if la != lb:
        scatter = ket_split(all_r)
    else:
        scatter = (If(differ("oa", "ob"), ket_split(all_r),
                      (Note("diagonal bra shell pair: its lower triangle "
                            "only"), *ket_split(low_r))),)

    row_idx = [Decl(f"pr{r}", Scalar.I32, init=Index("pidx", (
        plus("oa", r // nb), plus("ob", r % nb)))) for r in all_r]
    col_idx = [Decl(f"pc{kc}", Scalar.I32, init=Index("pidx", (
        plus("oc", kc // nd), plus("od", kc % nd)))) for kc in all_c]

    per_task: tuple[Stmt, ...] = (
        *contraction(spec.am),
        Note("the packed pair index of every row and column"),
        *row_idx,
        *col_idx,
        Note("scatter: one plain store per element and its bra-ket mirror, "
             "no atomics -- each element belongs to one task"),
        *scatter,
    )
    body = launch((Note(f"full ERI, class ({la},{lb}|{lc},{ld}) at L = "
                        f"{la + lb + lc + ld}: eri[pidx(m,n), pidx(l,s)] = "
                        "(mn|ls) (eri/full_eri/packed.py, eri/eri.py)"),),
                  per_task)
    nao, npp = Var("n_ao"), Var("n_pp")
    params = (
        *input_params(SCALARS),
        Param("pidx", Scalar.I32, array=True, shape=(nao, nao)),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(npp, npp)),
    )
    return Kernel(name or f"eri_{la}{lb}{lc}{ld}", params, body)
