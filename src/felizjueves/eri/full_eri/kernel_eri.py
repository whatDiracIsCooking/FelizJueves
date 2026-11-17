"""The full-ERI kernel, `Target.ERI`: every `(mn|ls)` of a class, packed.

    build_eri_kernel(spec | (la, lb, lc, ld), lowstack=None) -> Kernel
                                              one per class
    eri_lowstack(cls)                         -> bool     the form's rule
    task_threads(cls, lowstack=None)          -> int      threads per task
    eri_spec(la, lb, lc, ld)                  -> Spec     the class's key
    quartet_spec(cls)                         -> Spec     the Target.J quartet
    eri_plan(cls)                             -> (dag, order)
    launch_shape(n_task, threads, rows=1)     -> (grid, block)
    contraction(am) / launch(head, per_task) / input_params(scalars)
    lowstack_body(am, element, ...) / lowstack_launch(head, am, per_item)
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

THE LOW-STACK FORM (every class with an f shell, `eri_lowstack`; forced by
`lowstack=True` at any class). The form above holds the whole `n_ab x
n_cd` block in accumulators and the bra and ket `E` blocks and every
`I_c[p]` in locals: an `sm_86` frame of 43.9 KB at (dd|dd), all of it
reserved for every resident thread, and (ff|ff) would hold 10,000
accumulators. The low-stack form holds no block. A thread takes `rows`
consecutive bra rows `(a, b)` of ONE TASK, serially (`lowstack_body`):
here ONE ROW, so `task_threads = n_a n_b` threads per task
(`launch_shape(n_task, threads, rows=n_a n_b)`); `kernel_dense` takes the
`n_b` rows of one Cartesian `a`. Chosen by measurement (RTX A6000): one
row a thread is up to 5x faster for the packed path's few tasks a class
((ff|ff), 55 tasks: 0.95 against 5.2 ms), `n_b` rows a thread is faster
once the card is full (the dense path's many) because `[t]^(0)` is formed
once per thread. The task is the FAST index (`chunk = item / n_task`,
`task = item - chunk n_task`): the lanes of a warp are consecutive tasks
on the same rows, so every rolled loop runs the same trip counts across
the warp. Per (bra, ket) record pair:

    [t]^(0), |t| <= L     the same plan DAG, once, into a local R
                          (hermite_targets order)
    per-axis E tables     e_fragment.axis_tables_fragment, the bra's per
                          bra record, the ket's per ket record from -XQC,
                          -XQD: E^{cd}_q(-X) = (-1)^{lc+ld-|q|} E^{cd}_q(X)
                          exactly, which folds the (-1)^{|q|} of [p+q]
                          into one class-wide sign
    per row r = (a, b):   Er = E^{ab}_p over the row's reach, expanded once;
                          G[q] = SUM_p Er[p] R[p+q] for every |q| <= lc+ld;
                          per column (c, d), v = SUM_q E^{cd}_q(-X) G[q]
                          (e_fragment.rolled_e_sum over the ket tables), and
                          the element += (+-) w v IN PLACE in `eri` (the
                          first record pair stores, so nothing is zeroed and
                          the value is a zeroed element's first add, bit for
                          bit)

so the frame is R, G, Er, the six tables and the Boys `F_m`: 1,251
doubles at (ff|ff) (455 + 84 + 27 + 6 x 112 + 13), no accumulator.
Ownership is the scatter's, element by element: a diagonal bra pair's rows
`rb > ra` and a diagonal ket pair's columns `d > c` are skipped, and a task
that pairs a shell pair with itself (possible only when the bra's pair
class is the ket's) stores element `(pr, pc)` only from the row with
`pc <= pr`; then each owned element's bra-ket mirror is copied from it.
So every element still has one owner and one value: plain stores, no
atomics, bit-reproducible and exactly symmetric. ROUND-OFF, not bitwise,
against the form above (the sums run in another order); the device bound's
analysis (test_eri_full_device) covers both.

A builder: vendor-neutral IR. It imports neither `ao/` nor the interpreter
nor `reference/`.
"""

from __future__ import annotations

from collections.abc import Callable

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
    "LOWSTACK_MIN_L",
    "OUT",
    "RECORD_ARRAYS",
    "build_eri_kernel",
    "eri_lowstack",
    "eri_plan",
    "eri_spec",
    "launch_shape",
    "quartet_spec",
    "task_threads",
]

TASK = "task"
INT32_MAX = 2 ** 31 - 1
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


def launch_shape(n_task: int, threads: int = 32, *, rows: int = 1
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: `rows` threads per task (`task_threads`:
    1, or the bra block's rows in the low-stack form)."""
    if type(n_task) is not int or n_task < 1:
        raise ValueError(f"n_task = {n_task!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    if type(rows) is not int or rows < 1:
        raise ValueError(f"rows = {rows!r} is not a positive int")
    if n_task * rows > INT32_MAX:
        raise ValueError(f"{n_task} tasks x {rows} rows passes int32 (the "
                         "kernel's thread index)")
    return -(-(n_task * rows) // threads), (threads, 1)


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
                     name: str | None = None,
                     lowstack: bool | None = None) -> Kernel:
    """The full-ERI kernel of one quartet class (module docstring).
    `lowstack` picks the form: `None` is `eri_lowstack(cls)`, `True` /
    `False` force either at any class; same name and parameters, and
    `task_threads` threads per task."""
    spec = _spec(cls)
    if spec.levers:
        raise ValueError(f"{spec.key()}: the full ERI kernel takes no levers")
    if lowstack is None:
        lowstack = eri_lowstack(spec)
    if lowstack:
        return _build_eri_lowstack(spec, name)
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


# ---------------------------------------------------------------------------
# The low-stack form: one thread per (task, bra Cartesian a)
# ---------------------------------------------------------------------------

CHUNK = "chunk"  # the thread's chunk of the task's bra rows
R_ARRAY = "R"  # [t]^(0), hermite_targets(L) order
G_ARRAY = "G"  # G[q] = SUM_p E^{ab}_p [p+q]^(0), hermite_targets(lc+ld)
BRA_ARRAYS = ("Ebx", "Eby", "Ebz")  # the bra's per-axis E tables
KET_ARRAYS = ("Ekx", "Eky", "Ekz")  # the ket's, from the negated X
LOWSTACK_MIN_L = 3  # the low-stack form needs a shell with l >= 3 (f)
ER_ARRAY = "Er"  # the row's E^{ab}_p over its reach


def eri_lowstack(cls: Spec | tuple[int, int, int, int]) -> bool:
    """THE SELECTION RULE: True when the class has a shell with
    `l >= LOWSTACK_MIN_L` (f), where `build_eri_kernel` and
    `kernel_dense.build_dense_kernel` emit the low-stack form by default.
    Every class through (dd|dd) keeps the one-thread-per-task form and its
    text. Takes a `Target.ERI` or `Target.ERI_DENSE` key, or `am`."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    return max(am) >= LOWSTACK_MIN_L


def _n_rows(am: tuple[int, int, int, int]) -> int:
    return len(ef.cartesians(am[0])) * len(ef.cartesians(am[1]))


def task_threads(cls: Spec | tuple[int, int, int, int],
                 lowstack: bool | None = None) -> int:
    """Threads per task of THIS kernel: 1 in the one-thread-per-task form,
    the bra block's `n_a n_b` rows in the low-stack form, one row each
    (`lowstack=None` is `eri_lowstack(cls)`; `kernel_dense.task_threads`
    is the dense kernel's)."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    if lowstack is None:
        lowstack = eri_lowstack(am)
    return _n_rows(am) if lowstack else 1


def _cart_decode(l: int, idx: Expr, tag: str  # noqa: E741
                 ) -> tuple[tuple[Stmt, ...], tuple[Var, Var, Var]]:
    """The components of `cartesians(l)[idx]`, `idx` an I32 expression:
    `u = #{k in 1..l : k (k+1) / 2 <= idx}`, `w = idx - u (u+1) / 2`, then
    `(l - u, u - w, w)` (`e_fragment.cartesian_loops`' order)."""
    u: Expr = Lit(0)
    for k in range(1, l + 1):
        step = _c(Fn.LT, Lit(k * (k + 1) // 2 - 1), idx)
        u = step if k == 1 else _c(Fn.ADD, u, step)
    uu, ww = Var(f"{tag}_u"), Var(f"{tag}_w")
    names = (Var(f"{tag}x"), Var(f"{tag}y"), Var(f"{tag}z"))
    return ((
        Decl(uu.name, Scalar.I32, init=u),
        Decl(ww.name, Scalar.I32, init=_c(Fn.SUB, idx, _c(
            Fn.DIV, _c(Fn.MUL, uu, _c(Fn.ADD, uu, Lit(1))), Lit(2)))),
        Decl(names[0].name, Scalar.I32, init=_c(Fn.SUB, Lit(l), uu)),
        Decl(names[1].name, Scalar.I32, init=_c(Fn.SUB, uu, ww)),
        Decl(names[2].name, Scalar.I32, init=ww),
    ), names)


def _table_binds(frag, zeta: Expr, at: Expr, xa: str, xb: str,
                 arrays: tuple[str, str, str], negate: bool
                 ) -> dict[str, Expr]:
    def x(name: str, d: int) -> Expr:
        v = Index(name, (at, Lit(d)))
        return _c(Fn.SUB, Lit(0.0), v) if negate else v
    out: dict[str, Expr] = {ef.ZETA: zeta}
    out.update({n: Var(a) for n, a in zip(ef.AXIS_ARRAYS, arrays,
                                         strict=True)})
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), zeta)
    for d in range(3):
        out[Factor.pa(d).name_in_kernel] = x(xa, d)
        out[Factor.pb(d).name_in_kernel] = x(xb, d)
    return out


# The builder's placement of one element: `element(ci, di)` -> (the output
# element `Index` of the row `(ra, rb)` and the column `(ci, di)`, the I32
# guard -- nonzero when the thread owns it -- or None), with `ra`, `rb` and
# whatever `row_decls` declared in scope.
Element = Callable[[Expr, Expr], tuple[Index, Expr | None]]


def _rows_loop(am: tuple[int, int, int, int], rows: int,
               body: tuple[Stmt, ...], row_decls: tuple[Stmt, ...],
               row_guard: Expr | None, tag: str) -> For:
    """`for r in [r0, r0 + rows)`: `ra`, `rb`, `row_decls`, then `body`
    under `row_guard`."""
    nb = len(ef.cartesians(am[1]))
    r = Var(f"{tag}r")
    inner: tuple[Stmt, ...] = (*row_decls, *(
        (If(row_guard, body),) if row_guard is not None else body))
    return For(r.name, Var("r0"), _c(Fn.ADD, Var("r0"), Lit(rows)), Lit(1), (
        Decl("ra", Scalar.I32, init=_c(Fn.DIV, r, Lit(nb))),
        Decl("rb", Scalar.I32, init=_c(Fn.SUB, r, _c(Fn.MUL, Var("ra"),
                                                     Lit(nb)))),
        *inner,
    ))


def _cols_loop(am: tuple[int, int, int, int], element: Element,
               value: Callable[[Index], Stmt]) -> For:
    """Every column `(ci, di)` the thread owns, `value(element)` at each."""
    nc, nd = len(ef.cartesians(am[2])), len(ef.cartesians(am[3]))
    ci, di = Var("ci"), Var("di")
    at, guard = element(ci, di)
    st = value(at)
    return For("ci", Lit(0), Lit(nc), Lit(1), (
        For("di", Lit(0), Lit(nd), Lit(1), (
            (If(guard, (st,)),) if guard is not None else (st,))),))


def lowstack_body(am: tuple[int, int, int, int], rows: int,
                  element: Element, row_decls: tuple[Stmt, ...] = (),
                  row_guard: Expr | None = None,
                  task_decls: tuple[Stmt, ...] = ()) -> tuple[Stmt, ...]:
    """The low-stack form's per-thread statements, after `lowstack_launch`
    declared `task` and `chunk`: the task's indices, its owned elements
    zeroed, then the record loops accumulating each element's
    `(+-) w SUM_q E^{cd}_q(-X) G[q]` in place (module docstring, THE
    LOW-STACK FORM). Shared with `kernel_dense`."""
    la, lb, lc, ld = am
    lp, lq = la + lb, lc + ld
    L = lp + lq
    (dag,) = plan(Spec(Operator.COULOMB, 0, am, Target.J)).dags
    order = schedule(dag)
    ts = hermite_targets(L)
    n_q = len(hermite_targets(lq))
    held = {r for (r, m) in order if m == 0}
    assert set(ts) <= held, "the recurrence never forms every [t]^(0)"
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tower = _live((*prelude, *recurrence_body(dag, order, fundamental)),
                  {node_name((t, 0)) for t in ts})
    fbra, fket = ef.axis_tables_fragment(la, lb), ef.axis_tables_fragment(
        lc, ld)
    ib, ik = Var(BRA), Var(KET)
    sub = lq % 2 == 1  # (-1)^{|q|} E^{cd}_q = (-1)^{lc+ld} E^{cd}_q(-X)

    def g_of(a, b):
        def f(q: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            t = (Var("e_tx"), Var("e_ty"), Var("e_tz"))
            at = tuple(_c(Fn.ADD, t[d], q[d]) for d in range(3))
            inner = (
                Assign(Var("g"), _c(Fn.ADD, Var("g"), _c(
                    Fn.MUL, Index(ER_ARRAY, (Var("e_i"),)),
                    Index(R_ARRAY, (ef.hermite_index(at),))))),
                Assign(Var("e_i"), _c(Fn.ADD, Var("e_i"), Lit(1))),
            )
            return (
                Decl("g", Scalar.F64, init=Lit(0.0)),
                Decl("e_i", Scalar.I32, init=Lit(0)),
                _reach_loops(a, b, inner),
                Assign(Index(G_ARRAY, (ef.hermite_index(q),)), Var("g")),
            )
        return f

    def expand(a, b) -> tuple[Stmt, ...]:
        base = [ef.axis_table_index(a[d], b[d], 0, lb, lp) for d in range(3)]
        t = (Var("e_tx"), Var("e_ty"), Var("e_tz"))

        def e(d: int) -> Index:
            return Index(BRA_ARRAYS[d], (_c(Fn.ADD, base[d], t[d]),))
        return (
            Decl("e_i", Scalar.I32, init=Lit(0)),
            _reach_loops(a, b, (
                Assign(Index(ER_ARRAY, (Var("e_i"),)), _c(
                    Fn.MUL, _c(Fn.MUL, e(0), e(1)), e(2))),
                Assign(Var("e_i"), _c(Fn.ADD, Var("e_i"), Lit(1))),
            )),
        )

    def column(c: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def cell(d: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            at, guard = element(ef.cartesian_index("c"),
                                ef.cartesian_index("d"))
            term = _c(Fn.MUL, Var("w"), Var("v"))
            work = (
                *ef.rolled_e_sum(lc, ld, c, d, lambda h: Index(G_ARRAY, (h,)),
                                 "v", tag="f", arrays=KET_ARRAYS),
                If(Var("first"), (Assign(at, _c(Fn.SUB, Lit(0.0), term)
                                         if sub else term),),
                   (Assign(at, _c(Fn.SUB if sub else Fn.ADD, at, term)),)),
            )
            return (If(guard, work),) if guard is not None else work
        return (ef.cartesian_loops(ld, ("dx", "dy", "dz"), cell, tag="d"),)

    dec_a, a = _cart_decode(la, Var("ra"), "a")
    dec_b, b = _cart_decode(lb, Var("rb"), "b")
    per_row: tuple[Stmt, ...] = (
        *dec_a, *dec_b,
        Note("the row's E^{ab}_p = (E_x E_y) E_z over its reach, once"),
        *expand(a, b),
        Note("G[q] = SUM_p E^{ab}_p [p+q]^(0) for every |q| <= lc + ld, "
             "a fold over the row's reach in its loop order"),
        ef.hermite_loops(lq, g_of(a, b), tag="q"),
        Note("per column (c, d): out += (+-) w SUM_q E^{cd}_q(-X) G[q], "
             "rolled over the ket tables, in place"),
        ef.cartesian_loops(lc, ("cx", "cy", "cz"), column, tag="c"),
    )
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
        Note(f"[t]^(0), |t| <= {L}, by the plan's DAG for the quartet ({la},"
             f"{lb}|{lc},{ld}), once per record pair, then into the local R "
             "(hermite_targets order) so the rolled sums can index it"),
        *tower,
        *(Assign(Index(R_ARRAY, (Lit(i),)), Var(node_name((t, 0))))
          for i, t in enumerate(ts)),
        Note("the ket record's per-axis tables from -XQC, -XQD: "
             "E^{cd}_q(-X) = (-1)^{lc+ld-|q|} E^{cd}_q(X), exactly, so "
             "SUM_q (-1)^{|q|} E^{cd}_q G[q] = (-1)^{lc+ld} SUM_q "
             "E^{cd}_q(-X) G[q]"),
        Inline(fket, _table_binds(fket, Var("et"), ik, "XQC", "XQD",
                                  KET_ARRAYS, True)),
        Decl("w", Scalar.F64, init=_c(Fn.MUL, Var("sb"),
                                      Index("s_cd", (ik,)))),
        Note("the first record pair stores (+-) w v, every later one adds "
             "it: the same value as a zeroed element's first add, bit for "
             "bit"),
        Decl("first", Scalar.I32, init=_c(Fn.MUL, _c(
            Fn.LT, ib, _c(Fn.ADD, Var("b0"), Lit(1))), _c(
            Fn.LT, ik, _c(Fn.ADD, Var("k0"), Lit(1))))),
        Note("the chunk's rows, serially"),
        _rows_loop(am, rows, per_row, row_decls, row_guard, "x"),
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("ze", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Note("the bra record's per-axis E^{ij}_t tables, once"),
        Inline(fbra, _table_binds(fbra, Var("ze"), ib, "XPA", "XPB",
                                  BRA_ARRAYS, False)),
        Note("the ket records of the task's ket shell pair"),
        For(KET, Var("k0"), Var("k1"), Lit(1), ket_body),
    )
    return (
        Decl("tb", Scalar.I32, init=Index("task_bra", (Var(TASK),))),
        Decl("tk", Scalar.I32, init=Index("task_ket", (Var(TASK),))),
        Decl("b0", Scalar.I32, init=Index("brec0", (Var("tb"),))),
        Decl("b1", Scalar.I32, init=Index("brec1", (Var("tb"),))),
        Decl("k0", Scalar.I32, init=Index("krec0", (Var("tk"),))),
        Decl("k1", Scalar.I32, init=Index("krec1", (Var("tk"),))),
        Decl("oa", Scalar.I32, init=Index("off_a", (Var("tb"),))),
        Decl("ob", Scalar.I32, init=Index("off_b", (Var("tb"),))),
        Decl("oc", Scalar.I32, init=Index("off_c", (Var("tk"),))),
        Decl("od", Scalar.I32, init=Index("off_d", (Var("tk"),))),
        *task_decls,
        Decl("r0", Scalar.I32, init=_c(Fn.MUL, Var(CHUNK), Lit(rows))),
        *(Decl(n, Scalar.F64, (Lit(ef.axis_table_extent(la, lb)),))
          for n in BRA_ARRAYS),
        *(Decl(n, Scalar.F64, (Lit(ef.axis_table_extent(lc, ld)),))
          for n in KET_ARRAYS),
        Decl(R_ARRAY, Scalar.F64, (Lit(len(ts)),)),
        Decl(G_ARRAY, Scalar.F64, (Lit(n_q),)),
        Decl(ER_ARRAY, Scalar.F64, (Lit(_max_reach(la, lb)),)),
        Note("the bra records of the task's bra shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
    )


def _max_reach(la: int, lb: int) -> int:
    """The most Hermite `p` any row `(a, b)` reaches: max PROD (a_k+b_k+1)."""
    return max((a[0] + b[0] + 1) * (a[1] + b[1] + 1) * (a[2] + b[2] + 1)
               for a, b in ef.block_rows(la, lb))


def _reach_loops(a, b, body: tuple[Stmt, ...]) -> For:
    """`p` over the row's reach, `p_k = 0..a_k + b_k`, x outer: the I32
    loop variables `e_tx`, `e_ty`, `e_tz`."""
    def top(d: int) -> Expr:
        return _c(Fn.ADD, _c(Fn.ADD, a[d], b[d]), Lit(1))
    return For("e_tx", Lit(0), top(0), Lit(1), (
        For("e_ty", Lit(0), top(1), Lit(1), (
            For("e_tz", Lit(0), top(2), Lit(1), body),)),))


def lowstack_launch(head: tuple[Stmt, ...], am: tuple[int, int, int, int],
                    rows: int, per_item: tuple[Stmt, ...]
                    ) -> tuple[Stmt, ...]:
    """A low-stack kernel body: `head`'s notes, the thread's task and chunk
    (`item = THREAD_X + BLOCK_X BLOCK_DIM_X`, `task = item / n`, `chunk =
    item - task n`, `n = task_threads`), and `per_item` under `task <
    n_task`."""
    if _n_rows(am) % rows:
        raise ValueError(f"{rows} rows a thread do not divide {am}'s "
                         f"{_n_rows(am)}")
    n = _n_rows(am) // rows
    item = Var("item")
    return (
        *head,
        Note(f"launch: one thread per (task, chunk of "
             f"{rows} bra rows), {n} a task"),
        Decl("item", Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        Decl(CHUNK, Scalar.I32, init=_c(Fn.DIV, item, Var("n_task"))),
        Decl(TASK, Scalar.I32, init=_c(Fn.SUB, item, _c(
            Fn.MUL, Var(CHUNK), Var("n_task")))),
        If(_c(Fn.LT, Var(CHUNK), Lit(n)), per_item),
    )


def _differ(x: str, y: str) -> Expr:
    return _c(Fn.ADD, _c(Fn.LT, Var(x), Var(y)), _c(Fn.LT, Var(y), Var(x)))


def _both(x: Expr | None, y: Expr | None) -> Expr | None:
    if x is None or y is None:
        return y if x is None else x
    return _c(Fn.MUL, x, y)


def _params() -> tuple[Param, ...]:
    nao, npp = Var("n_ao"), Var("n_pp")
    return (
        *input_params(SCALARS),
        Param("pidx", Scalar.I32, array=True, shape=(nao, nao)),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(npp, npp)),
    )


def _build_eri_lowstack(spec: Spec, name: str | None) -> Kernel:
    """`build_eri_kernel`'s low-stack form (module docstring, THE LOW-STACK
    FORM): the same parameters and output."""
    la, lb, lc, ld = spec.am
    pr = Var("pr")
    same = (la, lb) == (lc, ld)  # a task may pair a shell pair with itself

    def element(ci: Expr, di: Expr) -> tuple[Index, Expr | None]:
        pc = Index("pidx", (_c(Fn.ADD, Var("oc"), ci),
                            _c(Fn.ADD, Var("od"), di)))
        guard = None
        if lc == ld:  # a diagonal ket pair: its lower triangle
            guard = _c(Fn.ADD, _differ("oc", "od"), _c(
                Fn.LT, di, _c(Fn.ADD, ci, Lit(1))))
        if same:  # the same pair: the larger packed index owns the two
            guard = _both(guard, _c(Fn.ADD, Var("other"), _c(
                Fn.LT, pc, _c(Fn.ADD, pr, Lit(1)))))
        return Index(OUT, (pr, pc)), guard

    row_decls = (Decl("pr", Scalar.I32, init=Index("pidx", (
        _c(Fn.ADD, Var("oa"), Var("ra")), _c(Fn.ADD, Var("ob"),
                                             Var("rb"))))),)
    row_guard = None
    if la == lb:  # a diagonal bra pair: rows rb <= ra
        row_guard = _c(Fn.ADD, _differ("oa", "ob"), _c(
            Fn.LT, Var("rb"), _c(Fn.ADD, Var("ra"), Lit(1))))
    task_decls: tuple[Stmt, ...] = ()
    if same:
        task_decls = (
            Note("other: the task does not pair a shell pair with itself; "
                 "when it does, element (pr, pc) and its mirror are both "
                 "the task's, and the larger packed index owns the two"),
            Decl("other", Scalar.I32, init=_c(
                Fn.ADD, _differ("oa", "oc"), _differ("ob", "od"))))
    body = lowstack_body(spec.am, 1, element, row_decls, row_guard,
                         task_decls)

    def mirror(at: Index) -> Stmt:
        return Assign(Index(OUT, (at.at[1], at.at[0])), at)

    body += (Note("the bra-ket mirror of every owned element"),
             _rows_loop(spec.am, 1, (_cols_loop(spec.am, element, mirror),),
                        row_decls, row_guard, "m"))
    kbody = lowstack_launch(
        (Note(f"full ERI, class ({la},{lb}|{lc},{ld}) at L = "
              f"{la + lb + lc + ld}, LOW-STACK: eri[pidx(m,n), pidx(l,s)] = "
              "(mn|ls) (eri/full_eri/packed.py, eri/eri.py)"),),
        spec.am, 1, body)
    return Kernel(name or f"eri_{la}{lb}{lc}{ld}", _params(), kbody)
