"""The three-center DF integral kernel, `Target.ERI3`: `(mn|P)`.

    build_eri3_kernel(spec | (la, lb, lP), lowstack=None) -> Kernel
    eri3_lowstack(spec | (la, lb, lP))     -> bool        the form's rule
    eri3_spec(la, lb, lP)                  -> Spec        the class's key
    quartet_spec(spec | (la, lb, lP))      -> Spec        Target.J (la lb|lP 0)
    eri3_plan(spec | (la, lb, lP))         -> (dag, order)  the plan's answer
    live_targets(la, lb, lP)               -> the [r]^(0) the contraction reads
    launch_shape(n_cell, threads)          -> (grid, block)

`(mn|P) = INT INT m(r1) n(r1) P(r2) / r12`, `m`, `n` Cartesian AOs of an
ordinary basis and `P` a Cartesian function of an auxiliary one, by
McMurchie & Davidson's Hermite path with the ket's Hermite "pair" a single
Gaussian (`eri/density_fit/eri3.py`'s module docstring, its witness):

    (mn|P) = SUM_{ij in mn} SUM_{k in P} s_ij s_k
             SUM_t E^{mn}_t(ij) SUM_tau (-1)^{|tau|} [t+tau]^(0)(P_ij - A_k)
             E^P_tau(k)

with `s_ij = c_i c_j (pi/p)^{3/2}` on each bra record (`ao.pairs.dress`'s
factor, `eri.density_fit.eri3`'s `s_ab`), `s_k = c_k (pi/alpha_k)^{3/2}` on each aux
record (`eri.density_fit.metric.aux_dress`) and `theta^2 = p alpha_k / (p + alpha_k)`.

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
shell) of `eri.density_fit.eri3.eri3_classes`' staging, `cell = THREAD_X + BLOCK_X *
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

ROUND-OFF, NOT BITWISE, against `eri.density_fit.eri3`: the host's `[t|tau]` is
`reference.pq`'s fixed recursion, the kernel's the plan's MILD DAG, and the
host dresses each `E` entry where the kernel multiplies `s_ij s_k` once.

THE SCATTER, a plain store mirrored in `(m, n)`, into `[m][n][P]` (aux
fastest, `eri.density_fit.eri3`'s layout): an off-diagonal shell pair (`off_a != off_b`)
writes element `(a, b, q)` to `eri3[off_a + a, off_b + b, off_p + q]` and
`eri3[off_b + b, off_a + a, off_p + q]`; a diagonal one (only when
`la == lb`) writes `a >= b`, mirrored. No atomics: each element belongs to
exactly one cell, so runs are bit-reproducible. A pair staged SWAPPED needs
no flag here: with the mirror, the transposed placement `eri.density_fit.eri3.
place_staged` makes writes the same two elements.

THE BOYS DOMAIN. A tight aux exponent far from a bra centre puts `T` far
past where the old narrow series overflowed (near 709); the one `fj_boys`
every kernel carries since #301 holds on all of `reference.boys`' domain.

THE LOW-STACK FORM (#606), the default from (df|s), (ff|s) and every aux-i
class. The form above holds `Ebra[nA nB][n_t(la+lb)]`, `Eket` and the
`nA nB nP` accumulators per thread, and nvcc keeps those blocks whole: #606
measured `sm_86` frames of 13,856 B at (dd|g), 32,040 B at (df|g) and
70,232 B at (ff|g) (the `E` blocks alone are 14,280, 28,680 and 69,000 B),
reserved for every resident thread (~7.3 GB of a 10 GB RTX 3080 at
(ff|g)). `build_eri3_kernel(cls)` therefore emits, whenever
`eri3_lowstack(cls)`, a kernel with the SAME name, parameters, launch and
output contract and neither block nor accumulator:

    per cell:    eri3[own (m, n, P)] = 0;  R[n_t(L)], U[n_t(la+lb)]  local
    per bra:     Ex Ey Ez = e_fragment.axis_tables_fragment(la, lb)
    per aux:     the tower, each live [r]^(0) stored into R as it forms
                     (recurrence.store_targets);
                 Cx Cy Cz = e_fragment.center_tables_fragment(lP)
                 for each aux row q            (cartesian_loops)
                     U[t] = SUM_tau R[t+tau] E^q_tau   every |t| <= la+lb,
                                               (rolled_center_sum)
                     for each bra row (a, b)   (cartesian_loops)
                         eri3[m, n, P] +-= s_ij s_k SUM_t E^{ab}_t U[t]
                                               (rolled_e_sum)
    per cell:    the (m, n) mirror, a plain copy

so the frame is `R`, `U` and the six tables (8,280 B at (ff|i), 6,336 B at
(ff|g)) plus whatever of the tower the compiler spills, and the text is the
tower and a fixed set of loops (124 KB at (ff|i) against 1.56 MB for
(ff|g) unrolled). Accumulating in place is legal because each element
belongs to exactly one cell, which zeroes its slots first: no atomics, and
runs are still bit-reproducible. A diagonal pair forms `a >= b` only, as
above. Interpreted, both forms agree with the host to the same bound and
with each other to `1e-14` of the class's max (test/test_kernel_df_lowstack
.py); the product `E_x (E_y (E_z U))` is in another association from the
block's, so only s and p are bitwise.

WHY THE RULE. `eri3_lowstack` is `la + lb + lP >= LOWSTACK_MIN_L = 10` or
the unrolled `E` blocks over (dd|g)'s, `E_DOUBLES_MAX = 1785` doubles:
the form that ships keeps its measured 13,856 B, the issue's budget, and no
class through f / i is built unrolled with larger blocks. On the ladder
`la <= lb <= 3`, `lP <= 6` that is 24 of 70 classes: every (df|.) and
(ff|.), every aux-i row, and (pf|h), (dd|h); nothing at `L <= 4`, so every
golden is unchanged. The `L >= 10` arm is for the tower, which grows
fastest (the `[p|q]` kernel's one-batch frame is 4,832 B at L = 10 and
12,496 B at L = 12, `kernels.json`). It was set from the text, and then
compiled once by hand (#606, no stored row). On `sm_86`, every low-stack
unit is at most 8,256 B of stack with at most 68 B of spill stores, and
compiles in at most 2.6 s. The unrolled (dd|g) is 15,992 B of stack with
27,276 B of spill stores, so the `E_DOUBLES_MAX` arm may be too loose
(three_center.md, "The low-stack form"). V's (df), which nvcc kept to
6,192 B with a 60 x 56 block (#586), says the mixed classes might fit
unrolled, so the rule errs to the smaller frame.

    params  n_cell n_bra_rec n_aux_rec n_ao n_aux (I32)
            zeta_ab[n_bra_rec]  P XPA XPB[n_bra_rec, 3]  s_ab[n_bra_rec]
            A[n_aux_rec, 3]  alpha[n_aux_rec]  s_aux[n_aux_rec]
            ab_rec0 ab_rec1 p_rec0 p_rec1 off_a off_b off_p [n_cell]  (I32)
            eri3[n_ao, n_ao, n_aux]  (OUT, written once per element)

The record and cell arrays are `eri.density_fit.eri3.Eri3Class.arrays()`' by name,
interleaved `[n_rec, 3]` and passed as plain positional parameters, except
`s`, which is `s_aux` here (`PARAM_OF`): the prelude declares a local
`s = -2 theta^2` (`recurrence.fundamental_prelude`), which would shadow it.
Elements of cells NOT launched are untouched: a caller that launches a subset
zeroes first.

A builder: vendor-neutral IR. It imports neither `ao/` (its witness) nor the
interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.ir import (
    Assign,
    Boys,
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
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
    store_targets,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "BRA",
    "CELL",
    "KET",
    "OUT",
    "PARAM_OF",
    "build_eri3_kernel",
    "E_DOUBLES_MAX",
    "eri3_e_doubles",
    "eri3_lowstack",
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
    """`body` (`Decl`s, `Note`s and the prelude's `Boys` fill) without the
    `Decl`s that neither `wanted` nor a kept later statement reads, and
    without a `Boys` whose array nothing kept reads: one backward pass."""
    kept: list[Stmt] = []
    live = set(wanted)
    for st in reversed(body):
        if isinstance(st, Decl):
            if st.name not in live:
                continue
            live |= _reads(st.init)
        elif isinstance(st, Boys):
            if st.dst not in live:
                continue
            live |= _reads(st.t)
        elif not isinstance(st, Note):
            raise TypeError(f"_live takes Decls, Notes and Boys, got "
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
                      name: str | None = None,
                      lowstack: bool | None = None) -> Kernel:
    """The three-center kernel of one class (module docstring).

    `cls` is a `Target.ERI3` `Spec` or `(la, lb, lP)`; any class the
    fragments and the plan accept is built. `lowstack` picks the form:
    `None` (the default) is `eri3_lowstack(cls)` (THE LOW-STACK FORM),
    `True` / `False` force either at any class. Both have the same name,
    parameters, launch and output contract."""
    spec = _spec(cls)
    if lowstack is None:
        lowstack = eri3_lowstack(spec)
    if lowstack:
        return _build_lowstack(spec, name)
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
    return Kernel(name or eri3_name(spec), _params(), body)


def _params() -> tuple[Param, ...]:
    """The parameter block (module docstring), one for both forms."""
    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    ncell, nbra, naux_rec = Var("n_cell"), Var("n_bra_rec"), Var("n_aux_rec")
    nao, naux = Var("n_ao"), Var("n_aux")
    return (
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


# ---------------------------------------------------------------------------
# The low-stack form (#606)
# ---------------------------------------------------------------------------

LOWSTACK_MIN_L = 10  # every class from L = la + lb + lP = 10 (#606)
# (dd|g)'s E blocks, 36 x 35 + 15 x 35 doubles (14,280 B): the largest the
# unrolled form keeps (#606 measured its frame at 13,856 B on sm_86)
E_DOUBLES_MAX = 1785
R_ARRAY = "R"  # [r]^(0) of one (bra, aux) pair, hermite_targets(L) order
U_ARRAY = "U"  # SUM_tau [t+tau] E^q_tau of one aux row q, hermite_targets(Lab)
AUX_TABLES = ("Cx", "Cy", "Cz")  # the aux record's single-center tables

LOWSTACK_NOTE = (
    "LOW-STACK (#606): per-axis E tables for the bra and the aux record in "
    "place of both E blocks, [r]^(0) into a local R as it forms, the "
    "contraction as loops through U[t] of one aux row, and the elements "
    "accumulated in place in the cell's own eri3 slots, mirrored at the end: "
    "no nA nB nP accumulators")


def eri3_e_doubles(cls: Spec | tuple[int, int, int]) -> int:
    """The doubles the UNROLLED form declares for its two `E` blocks,
    `Ebra[nA nB][n_t(la+lb)]` and `Eket[nP][n_t(lP)]`: the quantity its
    measured frame tracks (THE LOW-STACK FORM)."""
    la, lb, lP = _spec(cls).am
    return (len(ef.block_rows(la, lb)) * len(hermite_targets(la + lb))
            + len(ef.cartesians(lP)) * len(hermite_targets(lP)))


def eri3_lowstack(cls: Spec | tuple[int, int, int]) -> bool:
    """THE SELECTION RULE (#606): True when `la + lb + lP >=
    LOWSTACK_MIN_L`, or when the unrolled form's `E` blocks are larger
    than (dd|g)'s (`eri3_e_doubles`, `E_DOUBLES_MAX`); there
    `build_eri3_kernel` emits the low-stack form by default. False below,
    where the unrolled text is unchanged."""
    spec = _spec(cls)
    return (sum(spec.am) >= LOWSTACK_MIN_L
            or eri3_e_doubles(spec) > E_DOUBLES_MAX)


def _iadd(x: Expr, y: Expr) -> Expr:
    return _c(Fn.ADD, x, y)


def _build_lowstack(spec: Spec, name: str | None) -> Kernel:
    """`build_eri3_kernel`'s kernel with no `E` block and no accumulators
    (module docstring, THE LOW-STACK FORM): the same parameters, launch and
    output."""
    la, lb, lP = spec.am
    Lab, L = la + lb, la + lb + lP
    dag, order = eri3_plan(spec)
    fB, fP = ef.axis_tables_fragment(la, lb), ef.center_tables_fragment(lP)
    nA, nB, nP = (len(ef.cartesians(x)) for x in (la, lb, lP))
    odd = lP % 2 == 1
    cell, ib, ik = Var(CELL), Var(BRA), Var(KET)
    targets = live_targets(la, lb, lP)

    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tower = store_targets(_live((*prelude, *recurrence_body(
        dag, order, fundamental)), {node_name((r, 0)) for r in targets}),
        targets, L, R_ARRAY)

    def out_at(i: Expr, j: Expr, k: Expr, mirror: bool = False) -> Index:
        m, n = _iadd(Var("oa"), i), _iadd(Var("ob"), j)
        return Index(OUT, (n, m, _iadd(Var("op"), k)) if mirror
                     else (m, n, _iadd(Var("op"), k)))

    # --- one aux row q: U[t], then every (a, b) element of it ----------------
    def per_q(q: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def u_of(t: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            def r_at(tau: tuple[Expr, Expr, Expr]) -> Expr:
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _iadd(t[d], tau[d]) for d in range(3))),))
            return (*ef.rolled_center_sum(lP, q, r_at, "u", tag="c",
                                          arrays=AUX_TABLES),
                    Assign(Index(U_ARRAY, (ef.hermite_index(t),)), Var("u")))

        def row(a: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            def elem(b: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
                at = out_at(Var("ra"), Var("rb"), Var("rq"))
                update: tuple[Stmt, ...] = (
                    *ef.rolled_e_sum(la, lb, a, b,
                                     lambda h: Index(U_ARRAY, (h,)), "v"),
                    Assign(at, _c(Fn.SUB if odd else Fn.ADD, at,
                                  _c(Fn.MUL, Var("w"), Var("v")))),
                )
                head = (Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),)
                if la != lb:
                    return (*head, *update)
                return (*head, If(_iadd(Var("differ"), _c(
                    Fn.LT, Var("rb"), _iadd(Var("ra"), Lit(1)))), update))
            return (Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                    ef.cartesian_loops(lb, ("bx", "by", "bz"), elem, tag="b"))

        return (
            Decl("rq", Scalar.I32, init=ef.cartesian_index("q")),
            Note("U[t] = SUM_tau [t+tau]^(0) E^q_tau over the live tau of "
                 "this aux row, for every |t| <= la + lb: once per (t, q)"),
            ef.hermite_loops(Lab, u_of, tag="h"),
            Note(f"per element (a, b, q): eri3 {'-' if odd else '+'}= s_ij "
                 "s_k SUM_t E^{ab}_t U[t], rolled over the per-axis tables; "
                 f"(-1)^|tau| = (-1)^{lP} on every live tau"
                 + ("; a diagonal pair forms a >= b only" if la == lb
                    else "")),
            ef.cartesian_loops(la, ("ax", "ay", "az"), row, tag="a"),
        )

    aux_binds: dict[str, Expr] = {n: Var(n) for n in AUX_TABLES}
    if ef.HALF_INV_ZETA in fP.binds:
        aux_binds[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), Var("aq"))
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
             f"({la},{lb}|{lP},0), its dead targets dropped, each live one "
             "stored into R as it forms"),
        *tower,
        Note("the aux record's single-center E, per axis: "
             "e_fragment.center_tables_fragment, inlined"),
        *(Decl(n, Scalar.F64, (Lit(ef.center_table_extent(lP)),))
          for n in AUX_TABLES),
        Inline(fP, aux_binds),
        Decl("w", Scalar.F64, init=_c(Fn.MUL, Var("sb"),
                                      Index(S_AUX, (ik,)))),
        ef.cartesian_loops(lP, ("qx", "qy", "qz"), per_q, tag="q"),
    )

    bra_binds: dict[str, Expr] = {ef.ZETA: Var("ze"),
                                  **{n: Var(n) for n in ef.AXIS_ARRAYS}}
    if ef.HALF_INV_ZETA in fB.binds:
        bra_binds[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), Var("ze"))
    for d in range(3):
        bra_binds[Factor.pa(d).name_in_kernel] = Index("XPA", (ib, Lit(d)))
        bra_binds[Factor.pb(d).name_in_kernel] = Index("XPB", (ib, Lit(d)))
    bra_body: tuple[Stmt, ...] = (
        Decl("ze", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Note("the bra record's per-axis E^{ij}_t "
             "(e_fragment.axis_tables_fragment); no E^{ab}_t block"),
        *(Decl(n, Scalar.F64, (Lit(ef.axis_table_extent(la, lb)),))
          for n in ef.AXIS_ARRAYS),
        Inline(fB, bra_binds),
        Note("the aux records of the cell"),
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )

    # --- zero the cell's slots first, mirror them last ---------------------
    i, j, k = Var("i"), Var("j"), Var("k")

    def nest(ni: Expr, nj: Expr, body: tuple[Stmt, ...]) -> For:
        return For("i", Lit(0), ni, Lit(1), (For("j", Lit(0), nj, Lit(1), (
            For("k", Lit(0), Lit(nP), Lit(1), body),)),))

    zero = nest(Lit(nA), Lit(nB), (Assign(out_at(i, j, k), Lit(0.0)),))
    off_diag = (nest(Lit(nA), Lit(nB), (
        Assign(out_at(i, j, k, mirror=True), out_at(i, j, k)),)),)
    scatter: Stmt
    if la == lb:
        diag = (Note(_DIAGONAL), nest(Lit(nA), i, (
            Assign(out_at(i, j, k, mirror=True), out_at(i, j, k)),)))
        scatter = If(Var("differ"), off_diag, diag)
    else:
        scatter = If(Var("differ"), off_diag)

    per_cell: tuple[Stmt, ...] = (
        Decl("b0", Scalar.I32, init=Index("ab_rec0", (cell,))),
        Decl("b1", Scalar.I32, init=Index("ab_rec1", (cell,))),
        Decl("p0", Scalar.I32, init=Index("p_rec0", (cell,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (cell,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (cell,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (cell,))),
        Decl("op", Scalar.I32, init=Index("off_p", (cell,))),
        Decl("differ", Scalar.I32, init=_c(
            Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
            _c(Fn.LT, Var("ob"), Var("oa")))),
        Note(LOWSTACK_NOTE),
        Note("the cell's own elements, zeroed: no other cell writes them"),
        zero,
        *(Decl(n, Scalar.F64, (Lit(len(hermite_targets(x))),))
          for n, x in ((R_ARRAY, L), (U_ARRAY, Lab))),
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        Note("the (m, n) mirror, a plain copy: each element belongs to one "
             "cell, so no atomics"),
        scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"three-center DF integral, class (la, lb, lP) = ({la}, {lb}, "
             f"{lP}), LOW-STACK (#606): (mn|P) by the quartet "
             f"({la},{lb}|{lP},0)'s recurrence at L = {L} (ao/eri3.py)"),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )
    return Kernel(name or eri3_name(spec), _params(), body)
