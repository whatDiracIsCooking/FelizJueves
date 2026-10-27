"""The three-center DF gradient kernel: `Target.ERI3` at `derivative_order
= 1`.

    build_geri3_kernel(spec | (la, lb, lP)) -> Kernel        one per class
    geri3_name(spec | (la, lb, lP))         -> str           geri3_{la}{lb}{lP}
    geri3_spec(la, lb, lP)                  -> Spec          the class's d1 key
    geri3_plan(spec | (la, lb, lP))         -> (dag, order)  the tower one up
    launch_shape(n_cell, threads)           -> (grid, block)

Per cell (an AO shell pair crossed with an aux shell) of one class, staged
as `kernel_eri3` stages it (`eri.density_fit.eri3.eri3_classes`), the
cell's contribution to

    G[A, k] += SUM_mnP Gam_mnP d(mn|P) / dR_{A,k}

(`felizjueves.gradient`; `gradient.host_df.eri3_gradient` is the reference,
by another route on the aux side). `Gam` is any `[n_ao, n_ao, n_aux]`
tensor (`eri3`'s layout); a DF-J gradient passes `D (x) d`.

THREE CENTRES, TWO RULES (`host_df`'s module docstring):

- the AO pair's centres `A` and `B` by the Cartesian rule on the per-axis
  `E` tables one index above the class (`kernel_g1e`'s readers, #586's
  `axis_tables_fragment` at `(la + 1, lb + 1)`),
  `dA E^{ij}_t = 2 alpha E^{i+1,j}_t - i E^{i-1,j}_t`, `dB` likewise;
- the aux centre `C` by the Hermite shift: its single-center `E` does not
  depend on `C`, and `[r]^(0)` reads `P - C`, so
  `d/dC_k [t + tau]^(0) = -[t + tau + 1_k]^(0)`.

Per bra record, the weighted Hermite densities at each aux row `q` (the
weights `w_{r,q}` formed once per cell, `kernel_g1e.weights`' rule in
`(m, n)` at each aux function):

    H0_q,t   = SUM_r w_rq E^r_t                  |t| <= la + lb
    HA_k,q,t = SUM_r w_rq (dA_k E)^r_t           |t| <= la + lb + 1   (HB likewise)

then per aux record, `u_t,q = SUM_tau [t + tau]^(0) Eket[q, tau]` (the d0
kernel's inner sum, `|t| <= la + lb + 1`), and with `w = s_ij s_k` and the
class's sign `(-1)^{lP}`:

    gA_k +-= w SUM_q SUM_t HA_k,q,t u_t,q          (gB likewise)
    gC_k -+= w SUM_q SUM_t H0_q,t   u_{t + 1_k},q

and once per cell nine `AtomicAdd`s, `G[atom_a] += gA`, `G[atom_b] += gB`,
`G[atom_p] += gC`. `G` is ZEROED BY THE CALLER. No centre is inferred from
the others, so `SUM_A G[A] = 0` is a check.

THE TOWER is the plan of the `Target.J` quartet `(la + 1, lb | lP, 0)` at
`L + 1` (`geri3_plan`), its targets no `u` reads dropped by
`kernel_gmetric.live_body`.

THE THREAD MAPPING is `kernel_eri3`'s: one thread per cell, the bra records
outer, the aux records inner. A cell staged SWAPPED needs no flag: the weight
rule is symmetric in `(m, n)` and `atom_a` / `atom_b` follow the staged order
(`gradient.host_df.eri3_records`).

ROUND-OFF, NOT BITWISE, against `host_df` (another route on `C`, the plan's
MILD DAG against `reference.pq`, the weights folded in first).

    params  n_cell n_bra_rec n_aux_rec n_ao n_aux n_atom (I32)
            zeta_ab[n_bra_rec]  P XPA XPB[n_bra_rec, 3]  s_ab  ea_bra  eb_ket
            A[n_aux_rec, 3]  alpha[n_aux_rec]  s_aux[n_aux_rec]
            ab_rec0 ab_rec1 p_rec0 p_rec1 off_a off_b off_p
                atom_a atom_b atom_p [n_cell]  (I32)
            M[n_ao, n_ao, n_aux]                 the contracted tensor Gam
            G[n_atom, 3]  (OUT, read and AtomicAdd-ed)

The arrays are `Eri3Class.arrays()`' (`s` as `s_aux`) plus
`host_df.eri3_records`' exponents and atoms.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host_df`
(its witness) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_metric import S_AUX
from felizjueves.gradient.kernel_g1e import (
    ATOMIC_REASON,
    ZEROED_G,
    Locals,
    axis_reader,
    derivative_reader,
)
from felizjueves.gradient.kernel_gmetric import (
    center_binds,
    center_cols,
    launch_shape,
    reads,
    tower,
    unit,
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
from felizjueves.mmd.recurrence import PQ_PARAMS, THETA, node_name
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "build_geri3_kernel",
    "cell_weights",
    "geri3_name",
    "geri3_plan",
    "geri3_spec",
    "launch_shape",
]

CELL = "cell"  # one thread per (AO shell pair, aux shell)
BRA = "ib"  # the bra record, serial
KET = "ik"  # the aux record, serial inside the bra

_WEIGHTS = (
    "w_rq = M[a_i, b_j, P_q] + M[b_j, a_i, P_q] off the diagonal, "
    "M[a_i, a_j, P_q] on a diagonal shell pair: SUM_mnP M_mnP X_mnP "
    "exactly, for any M")


def geri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` at derivative order 1, `Target.ERI3`."""
    return Spec(Operator.COULOMB, 1, (la, lb, lP), Target.ERI3)


def _class(cls: Spec | tuple[int, int, int]) -> tuple[int, int, int]:
    spec = cls if isinstance(cls, Spec) else geri3_spec(*cls)
    if spec.target is not Target.ERI3 or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 Target.ERI3 key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb, lP = spec.am
    return la, lb, lP


def geri3_name(cls: Spec | tuple[int, int, int]) -> str:
    """The kernel name: `geri3_` and the three `l`s, `_`-separated once any
    is two digits (`kernel_eri3.eri3_name`'s rule)."""
    am = _class(cls)
    sep = "_" if max(am) > 9 else ""
    return "geri3_" + sep.join(map(str, am))


def geri3_plan(cls: Spec | tuple[int, int, int]) -> tuple[BatchDag, Order]:
    """The plan of the quartet `(la + 1, lb | lP, 0)`: the tower at
    `L + 1`."""
    la, lb, lP = _class(cls)
    (dag,) = plan(Spec(Operator.COULOMB, 0, (la + 1, lb, lP, 0),
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


def _plus(*vs: tuple[int, ...]) -> tuple[int, int, int]:
    return tuple(sum(v[d] for v in vs) for d in range(3))  # type: ignore[return-value]


def cell_weights(la: int, lb: int, lP: int) -> list[Stmt]:
    """`w{r}_{q}` per Cartesian pair `r` and aux row `q`, from `M`."""
    nb, nP = len(ef.cartesians(lb)), len(ef.cartesians(lP))
    rows = range(len(ef.block_rows(la, lb)))

    def m(i: Expr, j: Expr, q: int) -> Index:
        return Index("M", (i, j, _c(Fn.ADD, Var("op"), Lit(q))))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    def off(r: int, q: int) -> Expr:
        i, j = divmod(r, nb)
        return _c(Fn.ADD, m(plus("oa", i), plus("ob", j), q),
                  m(plus("ob", j), plus("oa", i), q))

    names = [(r, q) for r in rows for q in range(nP)]
    if la != lb:
        return [Note(_WEIGHTS), *(Decl(f"w{r}_{q}", Scalar.F64, init=off(r, q))
                                  for r, q in names)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    return [
        Note(_WEIGHTS),
        *(Decl(f"w{r}_{q}", Scalar.F64, init=Lit(0.0)) for r, q in names),
        If(differ,
           tuple(Assign(Var(f"w{r}_{q}"), off(r, q)) for r, q in names),
           tuple(Assign(Var(f"w{r}_{q}"), m(plus("oa", r // nb),
                                            plus("oa", r % nb), q))
                 for r, q in names)),
    ]


def build_geri3_kernel(cls: Spec | tuple[int, int, int], *,
                       name: str | None = None) -> Kernel:
    """The three-center gradient kernel of one class (module docstring).

    `cls` is a d1 `Target.ERI3` `Spec` or `(la, lb, lP)`."""
    la, lb, lP = _class(cls)
    Lab = la + lb
    dag, order = geri3_plan((la, lb, lP))
    fP = ef.center_fragment(lP)
    rowsB, rowsP = ef.block_rows(la, lb), ef.cartesians(lP)
    nP = len(rowsP)
    odd = lP % 2 == 1  # (-1)^{|tau|}, one sign for the class
    ib, ik, cell = Var(BRA), Var(KET), Var(CELL)
    ts0, ts1 = hermite_targets(Lab), hermite_targets(Lab + 1)
    index1 = {t: n for n, t in enumerate(ts1)}

    # --- the bra side: per-axis E one up, the weighted densities -------------
    E = axis_reader(la + 1, lb + 1)
    pool = Locals()
    dA = derivative_reader(E, pool, "A")
    dB = derivative_reader(E, pool, "B")

    def prod(f: list[Expr]) -> Expr:
        return _mul(_mul(f[0], f[1]), f[2])

    dens: list[Stmt] = []
    H0: dict[tuple[int, int], str] = {}  # (q, n over ts0) -> name
    HD: dict[tuple[str, int, int], str] = {}  # (side k, q, n over ts1)
    for q in range(nP):
        for n, t in enumerate(ts0):
            terms = [_mul(Var(f"w{r}_{q}"), prod([E(d, a[d], b[d], t[d])
                                                  for d in range(3)]))
                     for r, (a, b) in enumerate(rowsB)
                     if all(t[d] <= a[d] + b[d] for d in range(3))]
            if terms:
                H0[q, n] = f"H0q{q}_{n}"
                dens.append(Decl(H0[q, n], Scalar.F64, init=_fold(terms)))
    for side, D in (("A", dA), ("B", dB)):
        for k in range(3):
            for q in range(nP):
                for n, t in enumerate(ts1):
                    terms = []
                    for r, (a, b) in enumerate(rowsB):
                        top = [a[d] + b[d] + (d == k) for d in range(3)]
                        if any(t[d] > top[d] for d in range(3)):
                            continue
                        f = [D(d, a[d], b[d], t[d]) if d == k
                             else E(d, a[d], b[d], t[d]) for d in range(3)]
                        terms.append(_mul(Var(f"w{r}_{q}"), prod(f)))
                    if terms:
                        HD[f"{side}{k}", q, n] = f"H{side}{k}q{q}_{n}"
                        dens.append(Decl(HD[f"{side}{k}", q, n], Scalar.F64,
                                         init=_fold(terms)))

    # --- the aux side: u_t,q and the nine sums --------------------------------
    need: set[tuple[int, int]] = {(n, q) for (_, q, n) in HD}
    for (q, n) in H0:
        for k in range(3):
            need.add((index1[_plus(ts0[n], unit(k))], q))
    inner: list[Stmt] = []
    for n, q in sorted(need, key=lambda x: (x[1], x[0])):
        inner.append(Decl(f"u{n}_{q}", Scalar.F64, init=_fold([
            _mul(Var(node_name((_plus(ts1[n], tau), 0))),
                 Index("Eket", (Lit(q), Lit(j))))
            for j, tau in center_cols(rowsP[q], lP)])))
    wanted: set[str] = set()
    for st in inner:
        wanted |= reads(st.init)
    plus_ab, minus_ab = (Fn.SUB, Fn.ADD) if odd else (Fn.ADD, Fn.SUB)
    contract: list[Stmt] = []
    for side in "AB":
        for k in range(3):
            terms = [_mul(Var(HD[f"{side}{k}", q, n]), Var(f"u{n}_{q}"))
                     for (s, q, n) in HD if s == f"{side}{k}"]
            g = Var(f"g{side}{k}")
            contract.append(Assign(g, _c(plus_ab, g,
                                         _mul(Var("w"), _fold(terms)))))
    for k in range(3):
        terms = [_mul(Var(H0[q, n]),
                      Var(f"u{index1[_plus(ts0[n], unit(k))]}_{q}"))
                 for (q, n) in H0]
        g = Var(f"gC{k}")
        contract.append(Assign(g, _c(minus_ab, g,
                                     _mul(Var("w"), _fold(terms)))))

    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        *(Decl(nm, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                       Index("A", (ik, Lit(d)))))
          for d, nm in enumerate(PQ_PARAMS)),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var(ef.ZETA), Var("aq")),
            _c(Fn.ADD, Var(ef.ZETA), Var("aq"))))),
        Note(f"[t+tau]^(0) by the plan's DAG for the quartet ({la + 1},{lb}|"
             f"{lP},0) at L + 1 = {Lab + lP + 1}, the targets no u reads "
             "dropped with everything only they read"),
        *tower(order, dag, wanted),
        Decl("Eket", Scalar.F64, (Lit(nP), Lit(len(hermite_targets(lP))))),
        Inline(fP, center_binds(fP, Var("aq"), "Eket")),
        Note("u_t,q = SUM_tau [t+tau] Eket_q,tau over the live tau"),
        *inner,
        Decl("w", Scalar.F64, init=_mul(Var("sb"), Index(S_AUX, (ik,)))),
        Note(f"gA, gB {'-' if odd else '+'}= w SUM H(dE)_t u_t (the Cartesian "
             f"rule); gC {'+' if odd else '-'}= w SUM H0_t u_(t+1_k) (the "
             f"Hermite shift); (-1)^|tau| = (-1)^{lP} on every live tau"),
        *contract,
    )
    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA),
                              ef.HALF_INV_ZETA: Var("hiz")}
    for d in range(3):
        binds[Factor.pa(d).name_in_kernel] = Index("XPA", (ib, Lit(d)))
        binds[Factor.pb(d).name_in_kernel] = Index("XPB", (ib, Lit(d)))
    for arr in ef.AXIS_ARRAYS:
        binds[arr] = Var(arr)
    n_tab = ef.axis_table_extent(la + 1, lb + 1)
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
        Note(f"the undressed per-axis E^{{ij}}_t, i <= {la + 1}, j <= "
             f"{lb + 1}: one index above the class on each side, for the "
             "derivative rule"),
        *(Decl(arr, Scalar.F64, (Lit(n_tab),)) for arr in ef.AXIS_ARRAYS),
        Inline(ef.axis_tables_fragment(la + 1, lb + 1), binds),
        Note("the derivative factors, each declared where first read"),
        *pool.decls,
        Note("the weighted Hermite densities of this bra record, per aux row"),
        *dens,
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for side, atom in (("A", "aa"), ("B", "ab"), ("C", "ap")):
        for k in range(3):
            scatter.append(AtomicAdd(Index("G", (Var(atom), Lit(k))),
                                     Var(f"g{side}{k}")))
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
        *(Decl(f"g{s}{k}", Scalar.F64, init=Lit(0.0))
          for s in "ABC" for k in range(3)),
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"gradient of the three-center DF integral, class (la, lb, lP) "
             f"= ({la}, {lb}, {lP}): G[A, k] += SUM_mnP M_mnP d(mn|P)/dR_"
             "{A,k} (felizjueves.gradient)"),
        Note(ZEROED_G),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    ncell, nbra, naux_rec = Var("n_cell"), Var("n_bra_rec"), Var("n_aux_rec")
    nao, naux, nat = Var("n_ao"), Var("n_aux"), Var("n_atom")
    params = (
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
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3))),
    )
    return Kernel(name or geri3_name((la, lb, lP)), params, body)
