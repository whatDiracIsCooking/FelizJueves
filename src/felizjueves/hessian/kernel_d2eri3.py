"""The three-center DF Hessian kernel: `Target.ERI3` at `derivative_order
= 2`.

    build_d2eri3_kernel(spec | (la, lb, lP)) -> Kernel        one per class
    d2eri3_name(spec | (la, lb, lP))         -> str           d2eri3_{la}{lb}{lP}
    d2eri3_spec(la, lb, lP)                  -> Spec          the class's d2 key
    d2eri3_plan(spec | (la, lb, lP))         -> (dag, order)  the tower two up
    launch_shape(n_cell, threads)            -> (grid, block)

Per cell (an AO shell pair crossed with an aux shell) of one class, staged
as `kernel_eri3` stages it (`eri.density_fit.eri3.eri3_classes`), the
cell's contribution to

    H[X, k, Y, l] += SUM_mnP Gam_mnP d2(mn|P) / dR_{X,k} dR_{Y,l}

(`felizjueves.hessian`; `hessian.host_eri3.eri3_hessian` is the reference,
by another route on the aux side). `Gam` is any `[n_ao, n_ao, n_aux]`
tensor (`eri3`'s layout); a DF-J skeleton Hessian passes `D (x) d`. The
name is `d2eri3_`, not `heri3_`: that is the HGP `(mn|P)`'s
(`eri.density_fit.kernel_hgp`).

THREE CENTRES, TWO RULES, each applied twice (`kernel_geri3`'s, one order
up):

- the AO pair's centres `A` and `B` by the Cartesian rule on the per-axis
  `E` tables TWO indices above the class (`axis_tables_fragment(la + 2,
  lb + 2)`), an operator per axis (`kernel_h1e.axis_ops` /
  `kernel_h1e.operated`: `dA`, `dB`, `dA dA`, `dA dB`, `dB dB`);
- the aux centre `C` by the Hermite shift: its single-center `E` does not
  depend on `C`, and `[r]^(0)` reads `P - C`, so `d/dC_l [r]^(0) =
  -[r + 1_l]^(0)` and `d2/dC_k dC_l [r]^(0) = +[r + 1_k + 1_l]^(0)`.

Per bra record, the weighted Hermite densities at each aux row `q`, for
every operator triple `ops` a component needs (the plain one, the six
first derivatives, the 21 bra-bra second derivatives):

    H^{ops}_q,t = SUM_r w_rq PROD_d (op_d E)^{a_d b_d}_{t_d}
                                              |t| <= la + lb + n_ops

then per aux record `u_t,q = SUM_tau [t + tau]^(0) Eket[q, tau]` (the d0
kernel's inner sum, `|t| <= la + lb + 2`) and, with `w = s_ij s_k` and the
class's sign `sigma = (-1)^{lP}`:

    hXY_kl += sigma w SUM_q SUM_t H^{XY,kl}_q,t u_t,q          X, Y in A, B
    hXC_kl -= sigma w SUM_q SUM_t H^{X_k}_q,t   u_{t + 1_l},q  X in A, B
    hCC_kl += sigma w SUM_q SUM_t H^{0}_q,t     u_{t + 1_k + 1_l},q

45 accumulators per cell (`AA`, `BB`, `CC` with `k <= l`; `AB`, `AC`,
`BC` all nine), and once per cell 81 `AtomicAdd`s: each value into
`H[X, k, Y, l]` and its mirror `H[Y, l, X, k]` (once on the diagonal of a
same-centre block). `H` is ZEROED BY THE CALLER. No centre is inferred from
the others, so `SUM_A H[A, k, B, l] = 0` is a check.

THE TOWER is the plan of the `Target.J` quartet `(la + 2, lb | lP, 0)` at
`L + 2` (`d2eri3_plan`), its targets no `u` reads dropped by
`kernel_gmetric.live_body`.

THE THREAD MAPPING, the weights and the parameter block are
`kernel_geri3`'s, `G[n_atom, 3]` replaced by `H[n_atom, 3, n_atom, 3]`:
one thread per cell, the bra records outer, the aux records inner. A cell
staged SWAPPED needs no flag (`kernel_geri3`'s reason). The arrays are
`Eri3Class.arrays()`' (`s` as `s_aux`) plus `gradient.host_df.eri3_records`'
exponents and atoms, unchanged from d1.

ROUND-OFF, NOT BITWISE, against `hessian.host_eri3` (another route on `C`,
the plan's MILD DAG against `reference.pq`, the per-axis factors against
moved 3-D Cartesians). The interpreted test states its bound
(test/test_kernel_d2eri3.py).

THE TEXT of the form above is straight-line, as `kernel_geri3`'s: every
operator triple times every aux row times every Hermite index, unrolled
(504 KB at (pp|p), 1.25 MB at (pd|p)).

THE LOW-STACK FORM (#694), the default from `L = la + lb + lP >=
LOWSTACK_MIN_L = 3` (`d2eri3_lowstack`; `lowstack=` forces either form,
the name, parameters, launch and `H` contract the same). It holds no
weighted density and no weight, and is `kernel_geri3`'s low-stack form
with the low-stack `d2v`'s operated factors (`kernel_h1e.operated_value`,
`contractions`, `rolled`):

    per cell:    45 accumulators q = 0;  R[n_t(L + 2)], U[n_t(la + lb + 2)],
                 F{op}[3 (la + lb + 3)] for op in "" A B AA AB BB   local
    per bra:     Ex Ey Ez = e_fragment.axis_tables_fragment(la + 2, lb + 2)
    per aux:     the tower, each read [r]^(0) stored into R as it forms;
                 Cx Cy Cz = e_fragment.center_tables_fragment(lP)
                 for each aux row q:  U[t] = SUM_tau R[t + tau] E^q_tau
                     for each bra row (a, b):  wr = w wt, wt from M
                         F{op}[d, t] = (op E)^{a_d b_d}_t, by loops
                         q += wr SUM_t PROD_d F{o_d}[d, t_d] U[t + up]
                              for the 28 operator triples and their shifts
    per cell:    hXY = s qXY, hXC = -s qX, hCC = s qC (s = (-1)^{lP}),
                 then the same 81 atomics

The sums are regrouped by element, so the two forms agree to rounding. The
floor is measured (data/measured/d2_df_lowstack.json, `devtools/measure.sh
d2-df-lowstack-compile`, its rule declared before any row): every unrolled
class at `L <= 2` is within #606's 14,000 B of `sm_86` stack, and at `L =
3` (sp|d), (sd|p), (pp|p) and (pd|s) are not. The low-stack frames are
RECORDED, not gated: the bra tables two indices up are 9.5 KB alone at
(ff|.), and the (ff|i) unit is 24,544 B of `sm_86` stack (8 KB of it
spill), 293 KB of text, 23 s of nvcc.

A builder: vendor-neutral IR. It imports neither `ao/` nor
`hessian.host_eri3` (its witness) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_metric import S_AUX
from felizjueves.gradient.kernel_g1e import Locals, axis_reader
from felizjueves.gradient.kernel_geri3 import cell_weights
from felizjueves.gradient.kernel_gmetric import (
    center_binds,
    center_cols,
    launch_shape,
    reads,
    tower,
)
from felizjueves.hessian.kernel_h1e import (
    ATOMIC_REASON,
    COMPONENTS,
    OPS,
    ZEROED_H,
    axis_ops,
    contractions,
    f_array,
    operated,
    operated_value,
    rolled,
)
from felizjueves.ir import (
    Assign,
    AtomicAdd,
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
    node_name,
    store_targets,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ACCUMULATORS",
    "LOWSTACK_MIN_L",
    "build_d2eri3_kernel",
    "d2eri3_lowstack",
    "d2eri3_name",
    "d2eri3_plan",
    "d2eri3_spec",
    "launch_shape",
]

CELL = "cell"  # one thread per (AO shell pair, aux shell)
BRA = "ib"  # the bra record, serial
KET = "ik"  # the aux record, serial inside the bra

Ops = tuple[str, str, str]
Ang = tuple[int, int, int]

# The cell's centre-centre components `(X, k, Y, l)`: `kernel_h1e`'s 21
# bra-bra ones, then the 18 bra-aux ones and the six aux-aux ones `k <= l`.
ACCUMULATORS: tuple[tuple[str, int, str, int], ...] = (
    *COMPONENTS,
    *((x, k, "C", l) for x in "AB" for k in range(3)
      for l in range(3)),  # noqa: E741
    *(("C", k, "C", l) for k in range(3) for l in range(k, 3)),  # noqa: E741
)


def d2eri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` at derivative order 2, `Target.ERI3`."""
    return Spec(Operator.COULOMB, 2, (la, lb, lP), Target.ERI3)


def _class(cls: Spec | tuple[int, int, int]) -> tuple[int, int, int]:
    spec = cls if isinstance(cls, Spec) else d2eri3_spec(*cls)
    if spec.target is not Target.ERI3 or spec.derivative_order != 2:
        raise ValueError(f"{spec.key()} is not a d2 Target.ERI3 key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the hessian kernels take no levers")
    la, lb, lP = spec.am
    return la, lb, lP


def d2eri3_name(cls: Spec | tuple[int, int, int]) -> str:
    """The kernel name: `d2eri3_` and the three `l`s, `_`-separated once
    any is two digits (`kernel_eri3.eri3_name`'s rule)."""
    am = _class(cls)
    sep = "_" if max(am) > 9 else ""
    return "d2eri3_" + sep.join(map(str, am))


def d2eri3_plan(cls: Spec | tuple[int, int, int]) -> tuple[BatchDag, Order]:
    """The plan of the quartet `(la + 2, lb | lP, 0)`: the tower at
    `L + 2`."""
    la, lb, lP = _class(cls)
    (dag,) = plan(Spec(Operator.COULOMB, 0, (la + 2, lb, lP, 0),
                       Target.J)).dags
    return dag, schedule(dag)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _fold(terms: list[Expr]) -> Expr:
    out = terms[0]
    for t in terms[1:]:
        out = _c(Fn.ADD, out, t)
    return out


def _prod(f: list[Expr]) -> Expr:
    return _mul(_mul(f[0], f[1]), f[2])


def _plus(*vs: tuple[int, ...]) -> Ang:
    return tuple(sum(v[d] for v in vs) for d in range(3))  # type: ignore[return-value]


def _units(*axes: int) -> Ang:
    return tuple(sum(a == d for a in axes) for d in range(3))  # type: ignore[return-value]


def _h(x: str, k: int, y: str, l: int) -> str:  # noqa: E741
    return f"h{x}{y}_{k}{l}"


def _density_name(ops: Ops, q: int, n: int) -> str:
    return "H_" + "_".join(o or "n" for o in ops) + f"_q{q}_{n}"


def build_d2eri3_kernel(cls: Spec | tuple[int, int, int], *,
                        name: str | None = None,
                        lowstack: bool | None = None) -> Kernel:
    """The three-center Hessian kernel of one class (module docstring).

    `cls` is a d2 `Target.ERI3` `Spec` or `(la, lb, lP)`. `lowstack` picks
    the form: `None` (the default) is `d2eri3_lowstack(cls)` (THE
    LOW-STACK FORM), `True` / `False` force it. The name, parameters,
    launch and `H` contract are the same in both."""
    la, lb, lP = _class(cls)
    if lowstack is None:
        lowstack = d2eri3_lowstack((la, lb, lP))
    if lowstack:
        return _build_lowstack((la, lb, lP), name)
    Lab = la + lb
    dag, order = d2eri3_plan((la, lb, lP))
    fP = ef.center_fragment(lP)
    rowsB, rowsP = ef.block_rows(la, lb), ef.cartesians(lP)
    nP = len(rowsP)
    odd = lP % 2 == 1  # (-1)^{|tau|}, one sign for the class
    ib, ik, cell = Var(BRA), Var(KET), Var(CELL)
    ts2 = hermite_targets(Lab + 2)
    index2 = {t: n for n, t in enumerate(ts2)}

    # --- the bra side: per-axis E two up, the weighted densities -----------
    pool = Locals()
    F = operated(axis_reader(la + 2, lb + 2), pool, "")
    plain: Ops = ("", "", "")
    first = {(x, k): axis_ops(x, k, "", k) for x in "AB" for k in range(3)}
    triples: list[Ops] = [plain, *first.values(),
                          *(axis_ops(*c) for c in COMPONENTS)]
    dens: list[Stmt] = []
    names: dict[tuple[Ops, int, Ang], str] = {}  # (ops, q, t) -> name
    for ops in dict.fromkeys(triples):
        top = Lab + sum(map(len, ops))
        for q in range(nP):
            for n, t in enumerate(hermite_targets(top)):
                terms = [_mul(Var(f"w{r}_{q}"),
                              _prod([F(ops[d], d, a[d], b[d], t[d])
                                     for d in range(3)]))
                         for r, (a, b) in enumerate(rowsB)
                         if all(t[d] <= a[d] + b[d] + len(ops[d])
                                for d in range(3))]
                if terms:
                    names[ops, q, t] = _density_name(ops, q, n)
                    dens.append(Decl(names[ops, q, t], Scalar.F64,
                                     init=_fold(terms)))

    # --- the aux side: u_t,q and the 45 sums -------------------------------
    def contract(ops: Ops, up: Ang) -> list[tuple[str, Ang, int]]:
        """`(density, t + up, q)` of every live term of `SUM_q SUM_t
        H^{ops}_q,t u_{t + up},q`."""
        return [(nm, _plus(t, up), q) for (o, q, t), nm in names.items()
                if o == ops]

    sums: list[tuple[str, Fn, list[tuple[str, Ang, int]]]] = []
    plus_s, minus_s = (Fn.SUB, Fn.ADD) if odd else (Fn.ADD, Fn.SUB)
    for comp in COMPONENTS:
        sums.append((_h(*comp), plus_s, contract(axis_ops(*comp), (0, 0, 0))))
    for x in "AB":
        for k in range(3):
            for l in range(3):  # noqa: E741
                sums.append((_h(x, k, "C", l), minus_s,
                             contract(first[x, k], _units(l))))
    for k in range(3):
        for l in range(k, 3):  # noqa: E741
            sums.append((_h("C", k, "C", l), plus_s,
                         contract(plain, _units(k, l))))
    need = sorted({(index2[t], q) for _, _, terms in sums
                   for _, t, q in terms}, key=lambda x: (x[1], x[0]))
    inner: list[Stmt] = [Decl(f"u{n}_{q}", Scalar.F64, init=_fold([
        _mul(Var(node_name((_plus(ts2[n], tau), 0))),
             Index("Eket", (Lit(q), Lit(j))))
        for j, tau in center_cols(rowsP[q], lP)])) for n, q in need]
    wanted: set[str] = set()
    for st in inner:
        wanted |= reads(st.init)
    contracted: list[Stmt] = []
    for acc, fn, terms in sums:
        g = Var(acc)
        body = _fold([_mul(Var(nm), Var(f"u{index2[t]}_{q}"))
                      for nm, t, q in terms])
        contracted.append(Assign(g, _c(fn, g, _mul(Var("w"), body))))

    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("A", (ik, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var(ef.ZETA), Var("aq")),
            _c(Fn.ADD, Var(ef.ZETA), Var("aq"))))),
        Note(f"[t+tau]^(0) by the plan's DAG for the quartet ({la + 2},{lb}|"
             f"{lP},0) at L + 2 = {Lab + lP + 2}, the targets no u reads "
             "dropped with everything only they read"),
        *tower(order, dag, wanted),
        Decl("Eket", Scalar.F64, (Lit(nP), Lit(len(hermite_targets(lP))))),
        Inline(fP, center_binds(fP, Var("aq"), "Eket")),
        Note("u_t,q = SUM_tau [t+tau] Eket_q,tau over the live tau"),
        *inner,
        Decl("w", Scalar.F64, init=_mul(Var("sb"), Index(S_AUX, (ik,)))),
        Note(f"hXY {'-' if odd else '+'}= w SUM H^XY_t u_t (the Cartesian "
             f"rule twice); hXC {'+' if odd else '-'}= w SUM H^X_t "
             f"u_(t+1_l) (rule and shift); hCC {'-' if odd else '+'}= w SUM "
             f"H0_t u_(t+1_k+1_l) (the shift twice); (-1)^|tau| = "
             f"(-1)^{lP} on every live tau"),
        *contracted,
    )
    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA),
                              ef.HALF_INV_ZETA: Var("hiz")}
    for d in range(3):
        binds[Factor.pa(d).name_in_kernel] = Index("XPA", (ib, Lit(d)))
        binds[Factor.pb(d).name_in_kernel] = Index("XPB", (ib, Lit(d)))
    for arr in ef.AXIS_ARRAYS:
        binds[arr] = Var(arr)
    n_tab = ef.axis_table_extent(la + 2, lb + 2)
    bra_body: tuple[Stmt, ...] = (
        Decl(ef.ZETA, Scalar.F64, init=Index("zeta_ab", (ib,))),
        Note("hiz = 1/(2 zeta), not `half_inv_zeta`: the aux fragment binds "
             "that name"),
        Decl("hiz", Scalar.F64, init=_c(Fn.DIV, Lit(0.5), Var(ef.ZETA))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Decl("ea2", Scalar.F64, init=_mul(Lit(2.0), Index("ea_bra", (ib,)))),
        Decl("eb2", Scalar.F64, init=_mul(Lit(2.0), Index("eb_ket", (ib,)))),
        Note(f"the undressed per-axis E^{{ij}}_t, i <= {la + 2}, j <= "
             f"{lb + 2}: two indices above the class on each side, for the "
             "derivative rule applied twice"),
        *(Decl(arr, Scalar.F64, (Lit(n_tab),)) for arr in ef.AXIS_ARRAYS),
        Inline(ef.axis_tables_fragment(la + 2, lb + 2), binds),
        Note("the operated factors, each declared where first read"),
        *pool.decls,
        Note("the weighted Hermite densities of this bra record, per "
             "operator triple and aux row"),
        *dens,
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )
    atom = {"A": Var("aa"), "B": Var("ab"), "C": Var("ap")}

    def at(x: str, k: int, y: str, l: int) -> Index:  # noqa: E741
        return Index("H", (atom[x], Lit(k), atom[y], Lit(l)))

    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for x, k, y, l in ACCUMULATORS:  # noqa: E741
        h = Var(_h(x, k, y, l))
        scatter.append(AtomicAdd(at(x, k, y, l), h))
        if (x, k) != (y, l):
            scatter.append(AtomicAdd(at(y, l, x, k), h))
    per_cell: tuple[Stmt, ...] = (
        Decl("b0", Scalar.I32, init=Index("ab_rec0", (cell,))),
        Decl("b1", Scalar.I32, init=Index("ab_rec1", (cell,))),
        Decl("p0", Scalar.I32, init=Index("p_rec0", (cell,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (cell,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (cell,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (cell,))),
        Decl("op", Scalar.I32, init=Index("off_p", (cell,))),
        Decl("aa", Scalar.I32, init=Index("atom_a", (cell,))),
        Decl("ab", Scalar.I32, init=Index("atom_b", (cell,))),
        Decl("ap", Scalar.I32, init=Index("atom_p", (cell,))),
        *cell_weights(la, lb, lP),
        *(Decl(_h(*c), Scalar.F64, init=Lit(0.0)) for c in ACCUMULATORS),
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"Hessian of the three-center DF integral, class (la, lb, lP) "
             f"= ({la}, {lb}, {lP}): H[X, k, Y, l] += SUM_mnP M_mnP "
             "d2(mn|P)/dR_{X,k} dR_{Y,l} (felizjueves.hessian)"),
        Note(ZEROED_H),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )
    return Kernel(name or d2eri3_name((la, lb, lP)), _params(), body)


def _params() -> tuple[Param, ...]:
    """The parameter list, the same in both forms (module docstring)."""

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    ncell, nbra, naux_rec = Var("n_cell"), Var("n_bra_rec"), Var("n_aux_rec")
    nao, naux, nat = Var("n_ao"), Var("n_aux"), Var("n_atom")
    return (
        Param("n_cell", Scalar.I32),
        Param("n_bra_rec", Scalar.I32),
        Param("n_aux_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_aux", Scalar.I32),
        Param("n_atom", Scalar.I32),
        f64("zeta_ab", (nbra,)),
        f64("P", (nbra, Lit(3))),
        f64("XPA", (nbra, Lit(3))),
        f64("XPB", (nbra, Lit(3))),
        f64("s_ab", (nbra,)),
        f64("ea_bra", (nbra,)),
        f64("eb_ket", (nbra,)),
        f64("A", (naux_rec, Lit(3))),
        f64("alpha", (naux_rec,)),
        f64(S_AUX, (naux_rec,)),
        *(i32(n, (ncell,)) for n in ("ab_rec0", "ab_rec1", "p_rec0",
                                     "p_rec1", "off_a", "off_b", "off_p",
                                     "atom_a", "atom_b", "atom_p")),
        f64("M", (nao, nao, naux)),
        Param("H", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3), nat, Lit(3))),
    )



# ---------------------------------------------------------------------------
# The low-stack form (#694)
# ---------------------------------------------------------------------------

# every class from L = la + lb + lP = 3 (#694): the smallest L with an
# inadmissible unrolled class (data/measured/d2_df_lowstack.json)
LOWSTACK_MIN_L = 3
R_ARRAY = "R"  # [r]^(0) of one (bra, aux) pair, hermite_targets(L + 2) order
U_ARRAY = "U"  # SUM_tau [t+tau] E^q_tau of one aux row q, |t| <= la + lb + 2
AUX_TABLES = ("Cx", "Cy", "Cz")  # the aux record's single-center tables

LOWSTACK_NOTE = (
    "LOW-STACK (#694): no weighted densities and no per-cell weights; per "
    "aux record [r]^(0) into a local R as it forms, per aux row U[t] = "
    "SUM_tau R[t + tau] E^q_tau, and per element (a, b, q) its weight from "
    "M, the operated per-axis factors F{op}[d, t] and the 28 contractions "
    "against U[t + up] as loops (the low-stack d2v's, kernel_h1e)")


def d2eri3_lowstack(cls: Spec | tuple[int, int, int]) -> bool:
    """THE SELECTION RULE (#694): True when `la + lb + lP >=
    LOWSTACK_MIN_L`; there `build_d2eri3_kernel` emits the low-stack form
    by default. False below, where the unrolled text is unchanged."""
    return sum(_class(cls)) >= LOWSTACK_MIN_L


def _iadd(x: Expr, y: Expr) -> Expr:
    return _c(Fn.ADD, x, y)


def _q_name(x: str, k: int, y: str, l: int) -> str:  # noqa: E741
    """The accumulator `kernel_h1e.contractions` names for the component
    `(x, k, y, l)`: `qXY_kl`, `qX_kl` (`y` the aux centre), `qC_kl`."""
    if x == y == "C":
        return f"qC_{k}{l}"
    if y == "C":
        return f"q{x}_{k}{l}"
    return "q" + _h(x, k, y, l)[1:]


def _build_lowstack(am: tuple[int, int, int], name: str | None) -> Kernel:
    """`build_d2eri3_kernel`'s kernel with no density and no weight held
    (module docstring, THE LOW-STACK FORM): the same parameters, launch and
    `H` contract."""
    la, lb, lP = am
    Lab = la + lb
    L2 = Lab + lP + 2  # the tower's order
    dag, order = d2eri3_plan(am)
    fB = ef.axis_tables_fragment(la + 2, lb + 2)
    fP = ef.center_tables_fragment(lP)
    odd = lP % 2 == 1  # (-1)^{|tau|}, one sign for the class
    cell, ib, ik = Var(CELL), Var(BRA), Var(KET)
    targets = frozenset(_plus(t, tau) for t in hermite_targets(Lab + 2)
                        for q in ef.cartesians(lP)
                        for _, tau in center_cols(q, lP))
    tw = store_targets(tower(order, dag, {node_name((r, 0)) for r in targets}),
                       targets, L2, R_ARRAY)
    lb_t, L_t = lb + 2, Lab + 4  # the bra record's tables
    width = Lab + 3  # one axis's operated factor, t <= a_d + b_d + 2
    contracted = contractions(la, lb)

    def m_at(i: Expr, j: Expr) -> Index:
        return Index("M", (i, j, _iadd(Var("op"), Var("rq"))))

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
                m, n = _iadd(Var("oa"), Var("ra")), _iadd(Var("ob"), Var("rb"))
                off = _iadd(m_at(m, n), m_at(n, m))
                if la != lb:
                    weight: tuple[Stmt, ...] = (
                        Decl("wt", Scalar.F64, init=off),)
                else:
                    weight = (
                        Decl("wt", Scalar.F64, init=Lit(0.0)),
                        If(Var("differ"), (Assign(Var("wt"), off),),
                           (Assign(Var("wt"), m_at(m, _iadd(
                               Var("oa"), Var("rb")))),)),
                    )
                out: list[Stmt] = [
                    Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),
                    *weight,
                    Decl("wr", Scalar.F64, init=_mul(Var("w"), Var("wt"))),
                    Note("F{op}[d, t] = (op E)^{a_d b_d}_t on each axis, "
                         "kernel_h1e.operated's rule with the element's "
                         "indices at run time"),
                ]
                for op in OPS:
                    for d in range(3):
                        top = _iadd(_iadd(a[d], b[d]), Lit(len(op) + 1))
                        out.append(For("q_t", Lit(0), top, Lit(1), (
                            *operated_value(op, d, a[d], b[d], Var("q_t"),
                                            "o", lb_t, L_t),
                            Assign(Index(f_array(op), (_iadd(
                                Lit(d * width), Var("q_t")),)), Var("o")),
                        )))
                out.append(Note("q += wr SUM_t PROD_d F{o_d}[d, t_d] "
                                "U[t + up], rolled over t per axis"))
                out += [rolled(ops, ups, a, b, width, src=U_ARRAY)
                        for ops, ups in contracted]
                return tuple(out)
            return (Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                    ef.cartesian_loops(lb, ("bx", "by", "bz"), elem, tag="b"))

        return (
            Decl("rq", Scalar.I32, init=ef.cartesian_index("q")),
            Note("U[t] = SUM_tau [t+tau]^(0) E^q_tau over the live tau of "
                 "this aux row, for every |t| <= la + lb + 2: once per (t, q)"),
            ef.hermite_loops(Lab + 2, u_of, tag="h"),
            ef.cartesian_loops(la, ("ax", "ay", "az"), row, tag="a"),
        )

    aux_binds: dict[str, Expr] = {n: Var(n) for n in AUX_TABLES}
    if ef.HALF_INV_ZETA in fP.binds:
        aux_binds[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), Var("aq"))
    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("A", (ik, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var("ze"), Var("aq")),
            _c(Fn.ADD, Var("ze"), Var("aq"))))),
        Note(f"[t+tau]^(0) by the plan's DAG for the quartet ({la + 2},{lb}|"
             f"{lP},0) at L + 2 = {L2}, the targets no U reads dropped, each "
             "live one stored into R as it forms"),
        *tw,
        Note("the aux record's single-center E, per axis: "
             "e_fragment.center_tables_fragment, inlined"),
        *(Decl(n, Scalar.F64, (Lit(ef.center_table_extent(lP)),))
          for n in AUX_TABLES),
        Inline(fP, aux_binds),
        Decl("w", Scalar.F64, init=_mul(Var("sb"), Index(S_AUX, (ik,)))),
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
        Decl("ea2", Scalar.F64, init=_mul(Lit(2.0), Index("ea_bra", (ib,)))),
        Decl("eb2", Scalar.F64, init=_mul(Lit(2.0), Index("eb_ket", (ib,)))),
        Note(f"the undressed per-axis E^{{ij}}_t, i <= {la + 2}, j <= "
             f"{lb + 2}: two indices above the class on each side, for the "
             "derivative rule applied twice"),
        *(Decl(n, Scalar.F64, (Lit(ef.axis_table_extent(la + 2, lb + 2)),))
          for n in ef.AXIS_ARRAYS),
        Inline(fB, bra_binds),
        Note("the aux records of the cell"),
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )
    atom = {"A": Var("aa"), "B": Var("ab"), "C": Var("ap")}

    def at(x: str, k: int, y: str, l: int) -> Index:  # noqa: E741
        return Index("H", (atom[x], Lit(k), atom[y], Lit(l)))

    plus_s = not odd  # hXY, hCC carry (-1)^{lP}; hXC the opposite
    signed: list[Stmt] = [Note(
        f"hXY = {'-' if odd else '+'}qXY (the Cartesian rule twice), hXC = "
        f"{'+' if odd else '-'}qX (rule and shift), hCC = "
        f"{'-' if odd else '+'}qC (the shift twice): (-1)^|tau| = "
        f"(-1)^{lP} on every live tau")]
    for x, k, y, l in ACCUMULATORS:  # noqa: E741
        q = Var(_q_name(x, k, y, l))
        pos = plus_s if (y != "C" or x == "C") else not plus_s
        signed.append(Decl(_h(x, k, y, l), Scalar.F64,
                           init=q if pos else _c(Fn.SUB, Lit(0.0), q)))
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for x, k, y, l in ACCUMULATORS:  # noqa: E741
        h = Var(_h(x, k, y, l))
        scatter.append(AtomicAdd(at(x, k, y, l), h))
        if (x, k) != (y, l):
            scatter.append(AtomicAdd(at(y, l, x, k), h))
    differ: tuple[Stmt, ...] = ()
    if la == lb:
        differ = (Decl("differ", Scalar.I32, init=_c(
            Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
            _c(Fn.LT, Var("ob"), Var("oa")))),)
    per_cell: tuple[Stmt, ...] = (
        Decl("b0", Scalar.I32, init=Index("ab_rec0", (cell,))),
        Decl("b1", Scalar.I32, init=Index("ab_rec1", (cell,))),
        Decl("p0", Scalar.I32, init=Index("p_rec0", (cell,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (cell,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (cell,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (cell,))),
        Decl("op", Scalar.I32, init=Index("off_p", (cell,))),
        Decl("aa", Scalar.I32, init=Index("atom_a", (cell,))),
        Decl("ab", Scalar.I32, init=Index("atom_b", (cell,))),
        Decl("ap", Scalar.I32, init=Index("atom_p", (cell,))),
        *differ,
        Note(LOWSTACK_NOTE),
        *(Decl(q, Scalar.F64, init=Lit(0.0))
          for _, ups in contracted for _, q in ups),
        *(Decl(n, Scalar.F64, (Lit(len(hermite_targets(x))),))
          for n, x in ((R_ARRAY, L2), (U_ARRAY, Lab + 2))),
        *(Decl(f_array(op), Scalar.F64, (Lit(3 * width),)) for op in OPS),
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        *signed,
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"Hessian of the three-center DF integral, class (la, lb, lP) "
             f"= ({la}, {lb}, {lP}), LOW-STACK (#694): H[X, k, Y, l] += "
             "SUM_mnP M_mnP d2(mn|P)/dR_{X,k} dR_{Y,l} (felizjueves.hessian)"),
        Note(ZEROED_H),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )
    return Kernel(name or d2eri3_name(am), _params(), body)
