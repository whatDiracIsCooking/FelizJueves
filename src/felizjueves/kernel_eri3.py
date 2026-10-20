"""The three-center DF integral kernel, `Target.ERI3`: `(mn|P)`.

    build_eri3_kernel(spec | (la, lb, lP)) -> Kernel      one per class
    eri3_spec(la, lb, lP)                  -> Spec        the class's key
    quartet_spec(spec | (la, lb, lP))      -> Spec        Target.J (la lb|lP 0)
    eri3_plan(spec | (la, lb, lP))         -> (dag, order)  the plan's answer
    live_targets(la, lb, lP)               -> the [r]^(0) the contraction reads
    launch_shape(n_cell, threads)          -> (grid, block)

`(mn|P) = INT INT m(r1) n(r1) P(r2) / r12`, `m`, `n` Cartesian AOs of an
ordinary basis and `P` a Cartesian function of an auxiliary one, by
McMurchie & Davidson's Hermite path with the ket's Hermite "pair" a single
Gaussian (`ao/eri3.py`'s module docstring, its witness):

    (mn|P) = SUM_{ij in mn} SUM_{k in P} s_ij s_k
             SUM_t E^{mn}_t(ij) SUM_tau (-1)^{|tau|} [t+tau]^(0)(P_ij - A_k)
             E^P_tau(k)

with `s_ij = c_i c_j (pi/p)^{3/2}` on each bra record (`ao.pairs.dress`'s
factor, `ao.eri3`'s `s_ab`), `s_k = c_k (pi/alpha_k)^{3/2}` on each aux
record (`ao.metric.aux_dress`) and `theta^2 = p alpha_k / (p + alpha_k)`.

AN ERI3 CLASS IS A TRIPLE THAT RUNS A QUARTET (`spec.py`, #441). The key is
the triple `(la, lb, lP)`; the recurrence is the `Target.J` quartet
`(la lb|lP 0)`'s at `L = la + lb + lP`: `eri3_plan` asks `pipeline.plan`
for THAT spec (the Schwarz and metric pattern, `kernel_metric.quartet_spec`),
and `recurrence.fundamental_prelude` / `recurrence_body` spell its one DAG in
`schedule`'s order, unchanged.

THE DEAD TARGETS. The bra block `E^{ab}_t` is structurally nonzero at every
`t_d <= a_d + b_d`, so the bra reaches every `|t| <= la + lb`; the
single-center `E^P_tau` is zero whenever `q_d - tau_d` is odd on some axis
(`e_fragment.center_absent`), so every live `tau` has `|tau| = lP (mod 2)`.
The contraction therefore reads `[r]^(0)` at exactly `|r| >= lP mod 2`, and,
when the bra is (ss), only at `|r| = lP (mod 2)` (`live_targets`; the
formula is pinned independently in the test). The DAG builds the whole
`|r| <= L` tower, so the other targets -- and every intermediate, fundamental
and Boys weight only they read -- are DROPPED by one backward liveness pass
(`_live`, `kernel_metric`'s rule restated), because a declared but unread
local fails #92's filter on both compilers. Dropping a dead `Decl` changes
no value that is kept.

THE KET SIGN IS ONE SIGN PER CLASS. Every live `tau` has the parity of `lP`,
so `(-1)^{|tau|} = (-1)^{lP}` on every term: the inner sum is spelled
unsigned and the class's sign is applied once, `acc - w v` for odd `lP`
(`acc + w v` for even). Negation is exact.

THE THREAD MAPPING: ONE THREAD PER CELL, a cell one (AO shell pair, aux
shell) of `ao.eri3.eri3_classes`' staging, `cell = THREAD_X + BLOCK_X *
BLOCK_DIM_X`, guarded by `cell < n_cell`. The thread walks its cell's two
record ranges serially -- bra `[ab_rec0, ab_rec1)` outer, the `mn` block
formed once per bra record; aux `[p_rec0, p_rec1)` inner, the tower and the
`P` block once per (bra, aux) -- and accumulates `nA x nB x nP`
accumulators, `acc{a}_{b}_{q}`, in registers (`(dd|g)` is 540: high classes
will spill; nothing here is measured).

BOTH `E` FRAGMENTS ARE INLINED: `e_fragment.block_fragment(la, lb)` into
`Ebra` (its `zeta` the record's `zeta_ab`, `half_inv_zeta = 0.5 / zeta`,
`PA_d` / `PB_d` the record's `XPA` / `XPB`) and
`e_fragment.center_fragment(lP)` into `Eket` (`half_inv_zeta = 0.5 /
alpha`); the ket fragment's locals shadow the bra's, as in `kernel_metric`,
and only the arrays are read. Every `Ebra[row, i]` and `Eket[q, j]` read is
a structurally nonzero entry, which the fragment writes.

THE CONTRACTION, per (bra, aux) record pair:

    u_iq   = SUM_tau [t_i + tau]^(0) Eket[q, tau]          live tau of row q
    v_abq  = SUM_t Ebra[(a,b), t] u_tq                     t_d <= a_d + b_d
    acc_abq = acc_abq +- (s_ij s_k) v_abq                  (-1)^{lP}

each sum a left fold in `hermite_targets` order from its first live term.
`u` is the inner sum `kernel_metric` spells inside every element, named once
per `(t, q)` because it does not depend on the bra row: a common
subexpression, the same operations in the same order, so the same rounding.
NO PREFACTOR: `(pi/p)^{3/2}`, `(pi/alpha)^{3/2}` and the coefficients are
inside the host's `s_ab` and `s`, `(2/sqrt(pi)) theta` inside `[0]^(m)` (the
prelude's `W0`).

ROUND-OFF, NOT BITWISE, against `ao.eri3`: the host's `[t|tau]` is
`reference.pq`'s fixed recursion, the kernel's the plan's MILD DAG, and the
host dresses each `E` entry where the kernel multiplies `s_ij s_k` once.

THE SCATTER, a plain store mirrored in `(m, n)`, into `[m][n][P]` (aux
fastest, `ao.eri3`'s layout): an off-diagonal shell pair (`off_a != off_b`)
writes element `(a, b, q)` to `eri3[off_a + a, off_b + b, off_p + q]` and
`eri3[off_b + b, off_a + a, off_p + q]`; a diagonal one (only when
`la == lb`) writes `a >= b`, mirrored. No atomics: each element belongs to
exactly one cell, so runs are bit-reproducible. A pair staged SWAPPED needs
no flag here: with the mirror, the transposed placement `ao.eri3.
place_staged` makes writes the same two elements.

THE WIDE BOYS (#266). A tight aux exponent far from a bra centre puts `T`
past the narrow series' overflow near 709, so the kernel sets
`Kernel.wide_boys`, as `V`, Schwarz and the metric do.

    params  n_cell n_bra_rec n_aux_rec n_ao n_aux (I32)
            zeta_ab[n_bra_rec]  P XPA XPB[n_bra_rec, 3]  s_ab[n_bra_rec]
            A[n_aux_rec, 3]  alpha[n_aux_rec]  s_aux[n_aux_rec]
            ab_rec0 ab_rec1 p_rec0 p_rec1 off_a off_b off_p [n_cell]  (I32)
            eri3[n_ao, n_ao, n_aux]  (OUT, written once per element)

The record and cell arrays are `ao.eri3.Eri3Class.arrays()`' by name,
interleaved `[n_rec, 3]` and passed as plain positional parameters, except
`s`, which is `s_aux` here (`PARAM_OF`): the prelude declares a local
`s = -2 theta^2` (`recurrence.fundamental_prelude`), which would shadow it.
Elements of cells NOT launched are untouched: a caller that launches a subset
zeroes first.

A builder: vendor-neutral IR. It imports neither `ao/` (its witness) nor the
interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves import e_fragment as ef
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
    Weighted,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
)
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "BRA",
    "CELL",
    "KET",
    "OUT",
    "PARAM_OF",
    "build_eri3_kernel",
    "eri3_name",
    "eri3_plan",
    "eri3_spec",
    "launch_shape",
    "live_targets",
    "quartet_spec",
]

CELL = "cell"  # one thread per (AO shell pair, aux shell)
BRA = "ib"  # the bra record, serial
KET = "ik"  # the aux record, serial inside the bra
OUT = "eri3"
S_AUX = "s_aux"  # the aux record's scale: `s` is a prelude local
# `Eri3Class.arrays()` key -> this kernel's param name, where they differ
PARAM_OF = {"s": S_AUX}

_DIAGONAL = (
    "diagonal AO shell pair (off_a == off_b): a >= b only, each mirrored "
    "once -- kernel_jab's diagonal rule, so (mn|P) and (nm|P) are one value")


def eri3_name(cls: Spec | tuple[int, int, int]) -> str:
    """The kernel name of a class: `eri3_` and its three `l`s, run
    together while every `l <= 9` (`eri3_220`) and `_`-separated once any
    is two digits (`eri3_0_11_0`), so `(0, 11, 0)` and `(0, 1, 10)` never
    share a name (#445)."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    sep = "_" if max(am) > 9 else ""
    return "eri3_" + sep.join(map(str, am))


def eri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` as a generation key, `Target.ERI3`."""
    return Spec(Operator.COULOMB, 0, (la, lb, lP), Target.ERI3)


def _spec(cls: Spec | tuple[int, int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else eri3_spec(*cls)
    if spec.target is not Target.ERI3:
        raise ValueError(f"{spec.key()} is not a Target.ERI3 key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the ERI3 kernel takes no levers")
    return spec


def quartet_spec(cls: Spec | tuple[int, int, int]) -> Spec:
    """The `Target.J` quartet `(la lb | lP 0)` whose recurrence the class
    runs -- the key the plan layer is asked for (module docstring)."""
    la, lb, lP = _spec(cls).am
    return Spec(Operator.COULOMB, 0, (la, lb, lP, 0), Target.J)


def eri3_plan(cls: Spec | tuple[int, int, int]) -> tuple[BatchDag, Order]:
    """`pipeline.plan(quartet_spec(cls))`'s one DAG and `schedule(dag)`."""
    (dag,) = plan(quartet_spec(cls)).dags
    return dag, schedule(dag)


def launch_shape(n_cell: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per cell, `ceil(n_cell/threads)`
    blocks; a partial last block is legal."""
    if type(n_cell) is not int or n_cell < 1:
        raise ValueError(f"n_cell = {n_cell!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_cell // threads), (threads, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _plus(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, int, int]:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def _bra_cols(a: tuple[int, int, int], b: tuple[int, int, int], L: int
              ) -> list[tuple[int, tuple[int, int, int]]]:
    """`(column, t)` of bra row `(a, b)`'s structurally nonzero entries, in
    `hermite_targets(L)` order -- the columns `block_fragment` writes."""
    return [(i, t) for i, t in enumerate(hermite_targets(L))
            if all(t[d] <= a[d] + b[d] for d in range(3))]


def _aux_cols(q: tuple[int, int, int], l: int  # noqa: E741
              ) -> list[tuple[int, tuple[int, int, int]]]:
    """`(column, tau)` of aux row `q`'s structurally nonzero entries, in
    `hermite_targets(l)` order -- the columns `center_fragment` writes."""
    return [(j, t) for j, t in enumerate(hermite_targets(l))
            if not ef.center_absent(q, t)]


def live_targets(la: int, lb: int, lP: int
                 ) -> frozenset[tuple[int, int, int]]:
    """Every `[r]^(0)` the contraction reads: `r = t + tau` over the live
    entries of the bra block and the single-center block (module
    docstring)."""
    L = la + lb
    out: set[tuple[int, int, int]] = set()
    for a, b in ef.block_rows(la, lb):
        for _, t in _bra_cols(a, b, L):
            for q in ef.cartesians(lP):
                for _, tau in _aux_cols(q, lP):
                    out.add(_plus(t, tau))
    return frozenset(out)


def _reads(e: object) -> set[str]:
    """Every name an expression reads: `Var`s, arrays, `Factor` scalars."""
    if isinstance(e, Var):
        return {e.name}
    if isinstance(e, Index):
        return {e.array}.union(*(_reads(i) for i in e.at))
    if isinstance(e, Call):
        return set().union(*(_reads(a) for a in e.args))
    if isinstance(e, Weighted):
        out: set[str] = set()
        for t in e.terms:
            out |= _reads(t.src)
            if t.g.name_in_kernel is not None:
                out.add(t.g.name_in_kernel)
        return out
    return set()  # Lit


def _live(body: tuple[Stmt, ...], wanted: set[str]) -> tuple[Stmt, ...]:
    """`body` (`Decl`s and `Note`s) without the `Decl`s that neither `wanted`
    nor a kept later `Decl` reads: one backward pass."""
    kept: list[Stmt] = []
    live = set(wanted)
    for st in reversed(body):
        if isinstance(st, Decl):
            if st.name not in live:
                continue
            live |= _reads(st.init)
        elif not isinstance(st, Note):
            raise TypeError(f"_live takes Decls and Notes, got "
                            f"{type(st).__name__}")
        kept.append(st)
    return tuple(reversed(kept))


def _bra_binds(frag, zeta: Expr, at: Expr) -> dict[str, Expr]:
    """The bra block's binds: `zeta`, `1/2zeta`, `PA_d`, `PB_d`, `E`."""
    out: dict[str, Expr] = {ef.ZETA: zeta, ef.E_ARRAY: Var("Ebra")}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), zeta)
    for d in range(3):
        out[Factor.pa(d).name_in_kernel] = Index("XPA", (at, Lit(d)))
        out[Factor.pb(d).name_in_kernel] = Index("XPB", (at, Lit(d)))
    return out


def _aux_binds(frag, alpha: Expr) -> dict[str, Expr]:
    """The single-center block's binds: `1/2alpha` (when read) and `E`."""
    out: dict[str, Expr] = {ef.E_ARRAY: Var("Eket")}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), alpha)
    return out


def _acc(a: int, b: int, q: int) -> str:
    return f"acc{a}_{b}_{q}"


def _u(i: int, q: int) -> str:
    return f"u{i}_{q}"


def build_eri3_kernel(cls: Spec | tuple[int, int, int], *,
                      name: str | None = None) -> Kernel:
    """The three-center kernel of one class (module docstring).

    `cls` is a `Target.ERI3` `Spec` or `(la, lb, lP)`; any class the
    fragments and the plan accept is built."""
    spec = _spec(cls)
    la, lb, lP = spec.am
    Lab = la + lb
    dag, order = eri3_plan(spec)
    fB, fP = ef.block_fragment(la, lb), ef.center_fragment(lP)
    rowsB, rowsP = ef.block_rows(la, lb), ef.cartesians(lP)
    nA, nB, nP = len(ef.cartesians(la)), len(ef.cartesians(lb)), len(rowsP)
    tsB = hermite_targets(Lab)
    odd = lP % 2 == 1  # (-1)^{|tau|}, one sign for the class
    cell, ib, ik = Var(CELL), Var(BRA), Var(KET)

    # --- the contraction of one (bra, aux) record pair ----------------------
    bra_cols = [_bra_cols(a, b, Lab) for a, b in rowsB]
    read_t = sorted({i for cols in bra_cols for i, _ in cols})
    inner: list[Stmt] = []
    for i in read_t:
        for q, rq in enumerate(rowsP):
            u: Expr | None = None
            for j, tau in _aux_cols(rq, lP):
                term = _c(Fn.MUL, Var(node_name((_plus(tsB[i], tau), 0))),
                          Index("Eket", (Lit(q), Lit(j))))
                u = term if u is None else _c(Fn.ADD, u, term)
            assert u is not None  # every row has a live column
            inner.append(Decl(_u(i, q), Scalar.F64, init=u))
    contract: list[Stmt] = []
    for row, cols in enumerate(bra_cols):
        a, b = divmod(row, nB)
        for q in range(nP):
            v: Expr | None = None
            for i, _ in cols:
                term = _c(Fn.MUL, Index("Ebra", (Lit(row), Lit(i))),
                          Var(_u(i, q)))
                v = term if v is None else _c(Fn.ADD, v, term)
            assert v is not None  # t = 0 is live in every bra row
            acc = Var(_acc(a, b, q))
            contract.append(Assign(acc, _c(
                Fn.SUB if odd else Fn.ADD, acc, _c(Fn.MUL, Var("w"), v))))
    wanted: set[str] = set()
    for st in inner:
        wanted |= _reads(st.init)

    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tower = _live((*prelude, *recurrence_body(dag, order, fundamental)),
                  wanted)

    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                      Index("A", (ik, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Note("theta^2 = p alpha / (p + alpha): the aux Gaussian is a "
             "Hermite pair at zeta = alpha, Q = A"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ze"), Var("aq")),
            _c(Fn.ADD, Var("ze"), Var("aq"))))),
        Note("[t+tau]^(0) by the plan's DAG for the quartet "
             f"({la},{lb}|{lP},0), the targets no live t + tau reaches (the "
             "center E's parity zeros) dropped with everything only they "
             "read"),
        *tower,
        Note("the undressed single-center E block of this aux record: "
             "e_fragment.center_fragment, inlined"),
        Decl("Eket", Scalar.F64, (Lit(nP), Lit(len(hermite_targets(lP))))),
        Inline(fP, _aux_binds(fP, Var("aq"))),
        Note("u_tq = SUM_tau [t+tau] Eket_q,tau over the live tau, once per "
             "(t, q): it does not depend on the bra row"),
        *inner,
        Decl("w", Scalar.F64, init=_c(Fn.MUL, Var("sb"),
                                      Index(S_AUX, (ik,)))),
        Note(f"per element (a, b, q): acc {'-' if odd else '+'}= s_ij s_k "
             "SUM_t Ebra_t u_tq over the structurally nonzero t; "
             f"(-1)^|tau| = (-1)^{lP} on every live tau"),
        *contract,
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("ze", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Note("the undressed E block of this bra record, once: "
             "e_fragment.block_fragment, inlined"),
        Decl("Ebra", Scalar.F64, (Lit(len(rowsB)), Lit(len(tsB)))),
        Inline(fB, _bra_binds(fB, Var("ze"), ib)),
        Note("the aux records of the cell"),
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )

    # --- the scatter --------------------------------------------------------
    def at(m: Expr, n: Expr, q: int) -> Index:
        return Index(OUT, (m, n, plus("op", q)))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    off_diag: list[Stmt] = []
    for a in range(nA):
        for b in range(nB):
            for q in range(nP):
                v = Var(_acc(a, b, q))
                off_diag += [Assign(at(plus("oa", a), plus("ob", b), q), v),
                             Assign(at(plus("ob", b), plus("oa", a), q), v)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    scatter: Stmt
    if la == lb:
        diag: list[Stmt] = [Note(_DIAGONAL)]
        for a in range(nA):
            for b in range(a + 1):
                for q in range(nP):
                    v = Var(_acc(a, b, q))
                    diag.append(Assign(at(plus("oa", a), plus("oa", b), q), v))
                    if a != b:
                        diag.append(Assign(
                            at(plus("oa", b), plus("oa", a), q), v))
        scatter = If(differ, tuple(off_diag), tuple(diag))
    else:
        scatter = If(differ, tuple(off_diag))

    per_cell: tuple[Stmt, ...] = (
        Decl("b0", Scalar.I32, init=Index("ab_rec0", (cell,))),
        Decl("b1", Scalar.I32, init=Index("ab_rec1", (cell,))),
        Decl("p0", Scalar.I32, init=Index("p_rec0", (cell,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (cell,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (cell,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (cell,))),
        Decl("op", Scalar.I32, init=Index("off_p", (cell,))),
        *(Decl(_acc(a, b, q), Scalar.F64, init=Lit(0.0))
          for a in range(nA) for b in range(nB) for q in range(nP)),
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        Note("scatter: one plain store per element and its (m, n) mirror, "
             "no atomics -- each element belongs to one cell"),
        scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"three-center DF integral, class (la, lb, lP) = ({la}, {lb}, "
             f"{lP}): (mn|P) by the quartet ({la},{lb}|{lP},0)'s recurrence "
             f"at L = {Lab + lP} (ao/eri3.py)"),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    ncell, nbra, naux_rec = Var("n_cell"), Var("n_bra_rec"), Var("n_aux_rec")
    nao, naux = Var("n_ao"), Var("n_aux")
    params = (
        Param("n_cell", Scalar.I32),
        Param("n_bra_rec", Scalar.I32),
        Param("n_aux_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_aux", Scalar.I32),
        f64("zeta_ab", (nbra,)),
        f64("P", (nbra, Lit(3))),
        f64("XPA", (nbra, Lit(3))),
        f64("XPB", (nbra, Lit(3))),
        f64("s_ab", (nbra,)),
        f64("A", (naux_rec, Lit(3))),
        f64("alpha", (naux_rec,)),
        f64(S_AUX, (naux_rec,)),
        *(i32(n, (ncell,)) for n in ("ab_rec0", "ab_rec1", "p_rec0",
                                     "p_rec1", "off_a", "off_b", "off_p")),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nao, nao, naux)),
    )
    return Kernel(name or eri3_name(spec), params, body, wide_boys=True)
