"""The uncontracted DF metric derivative: `Target.METRIC` at
`derivative_order = 1` with the kernel-body lever `uncontracted` (#693).

    build_ipmetric_kernel(spec | (lP, lQ)) -> Kernel       one per aux class
    ipmetric_name(spec | (lP, lQ))         -> str          ipmetric_{lP}{lQ}
    ipmetric_spec(lP, lQ)                  -> Spec         the class's key
    launch_shape(n_pair, threads)          -> (grid, block)

Per aux shell pair `(P, Q)` of one class, staged as `kernel_metric` stages it
(`eri.density_fit.metric.metric_classes`), the derivative integrals
themselves, in the ip1 form -- the derivative on the centre of the FIRST
index:

    M1[k, P, Q] = d(P|Q) / dA_k          A the centre of P

`[3, n_aux, n_aux]`, row-major. Not their contraction: `gmetric_` writes
`SUM_PQ W_PQ d(P|Q)/dR` into `G[atom, 3]`, which a gradient needs and the
DF Hessian's response terms do not -- `SUM_Q M1 d`, `SUM_Q M1 c_ij` and the
`M^x M^y` cross terms are each a gemm on this matrix (3 n_aux^2 doubles,
54 MB at 1500 aux functions). `gradient.host_df.metric_derivative` is the
reference, by another route, and libcint's `int2c2e_ip1` the witness:
libcint differentiates the ELECTRON coordinate, so `M1 = -int2c2e_ip1`.

THE OTHER CENTRE NEEDS NO SECOND KERNEL. `(P|Q)` reads `A - B` only, so
`d/dB_k = -d/dA_k`, and `(P | d_k Q) = -M1[k, P, Q]`; read with `Q` first,
that is `M1[k, Q, P] = d(Q|P)/dB_k = -M1[k, P, Q]`: the matrix is
ANTISYMMETRIC per axis. An off-diagonal pair writes `(P, Q)` and the
negated mirror `(Q, P)`; a diagonal pair (one shell, `off_p == off_q`)
forms every `(a, b)` of its block and mirrors nothing, element `(b, a)`
being its own. Atom `X`'s block of a perturbation is the rows `P` on `X`
and their mirror; that slicing is the caller's.

THE HERMITE SHIFT, `kernel_gmetric`'s: a single-center `E^a_t` does not
depend on where its Gaussian sits, so moving `P` moves only the Hermite
Gaussians, and

    M1[k, P, Q] = SUM_t E^P_t SUM_tau (-1)^{|tau|} [t + tau + 1_k]^(0) E^Q_tau

-- the tower of the quartet `(lP + 1, 0 | lQ, 0)` at `L + 1`
(`kernel_gmetric.gmetric_plan`), its targets no `t + tau + 1_k` reaches
dropped (`kernel_gmetric.tower`).

ONE FORM, THE ROLLED ONE, at every class: `kernel_gmetric`'s low-stack
form (#613) with its contraction undone. Per (P, Q) record pair the tower
is stored into a local `R[n_t(L + 1)]`; per Q row `b`, `U[t'] = SUM_tau
R[t' + tau] E^b_tau` for every `|t'| <= lP + 1` of its parity; per element
`(a, b)` and axis `k`, `M1[k, op + a, oq + b] +-= s_i s_k SUM_t E^a_t U[t +
1_k]`, rolled over the P record's single-center tables
(`e_fragment.rolled_center_sum`), accumulated IN PLACE in the pair's own
slots (zeroed first, `kernel_metric`'s low-stack rule) -- no `3 nP nQ`
accumulators, which at (i|i) would be 2,352 registers. The frame is `R`,
`U` and six tables at every class.

THE THREAD MAPPING is `kernel_metric`'s: one thread per aux shell pair,
both record ranges serially, bra outer. No atomics and no reduction: each
element belongs to exactly one aux shell pair, so it is written by plain
stores, bit-reproducible run to run. Elements of pairs NOT launched are
untouched: a caller that launches a subset of the classes zeroes first.

A pair staged SWAPPED (`lP <= lQ` puts the lower-l shell first) needs no
flag: `off_p` is the staged P's, and the mirror makes both orders.

ROUND-OFF, NOT BITWISE, against `host_df`: another route (the Cartesian
rule `2 alpha E^{a+1} - a E^{a-1}` on each centre separately) and the
plan's MILD DAG against `reference.pq`.

    params  n_pair n_rec n_aux (I32)
            A[n_rec, 3]  alpha[n_rec]  s_aux[n_rec]           per aux record
            p_rec0 p_rec1 q_rec0 q_rec1 off_p off_q [n_pair]  (I32)
            M1[3, n_aux, n_aux]  (OUT, read and written)

`kernel_metric`'s parameter block, `M1` in place of `metric`: no atoms and
no weight matrix.

THE NAME is `ipmetric_`, after libcint's `ip1`: not `gmetric_` (the
contraction), `hmetric_` (the HGP metric), `hgmetric_` (its gradient) nor
`d2metric_` (the Hessian).

A builder: vendor-neutral IR. It imports neither `ao/` nor
`gradient.host_df` (its witness) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_metric import S_AUX
from felizjueves.gradient.kernel_gmetric import (
    BRA,
    KET,
    PAIR,
    center_cols,
    gmetric_plan,
    launch_shape,
    tower,
)
from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Expr,
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
from felizjueves.mmd.recurrence import PQ_PARAMS, THETA, node_name, store_targets
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target, lever_provenance

__all__ = [
    "OUT",
    "UNCONTRACTED",
    "build_ipmetric_kernel",
    "ipmetric_name",
    "ipmetric_spec",
    "launch_shape",
]

UNCONTRACTED = "uncontracted"  # the kernel-body lever (levers.VOCABULARY)
OUT = "M1"  # [3, n_aux, n_aux]
R_ARRAY = "R"  # [r]^(0) of one (P, Q) record pair, hermite_targets(L + 1)
U_ARRAY = "U"  # SUM_tau [t'+tau] E^b_tau of one Q row b, |t'| <= lP + 1
BRA_TABLES = ("CPx", "CPy", "CPz")  # the P record's single-center tables
KET_TABLES = ("CQx", "CQy", "CQz")  # the Q record's

ROLLED_NOTE = (
    "ROLLED (#693, kernel_gmetric's low-stack form uncontracted): per-axis "
    "single-center E tables for both records, [r]^(0) into a local R as it "
    "forms, U[t'] of one Q row, and each element's three derivatives as "
    "rolled sums over the P tables through U[t + 1_k], accumulated in place "
    "in the pair's own M1 slots: no 3 nP nQ accumulators")

DIAGONAL = (
    "diagonal aux shell pair (off_p == off_q): every (a, b) of the block is "
    "formed, and nothing is mirrored -- element (b, a) is its own, "
    "M1[k, b, a] = -M1[k, a, b] by its own sum")


def ipmetric_spec(lP: int, lQ: int) -> Spec:
    """The aux class `(lP, lQ)` at derivative order 1, uncontracted."""
    return Spec(Operator.COULOMB, 1, (lP, lQ), Target.METRIC, (UNCONTRACTED,))


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else ipmetric_spec(*cls)
    if spec.target is not Target.METRIC or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 Target.METRIC key")
    if spec.levers != (UNCONTRACTED,):
        raise ValueError(f"{spec.key()}: the uncontracted metric derivative "
                         f"is the key with exactly the lever {UNCONTRACTED!r}"
                         " (the contraction is kernel_gmetric's)")
    lP, lQ = spec.am
    return lP, lQ


def ipmetric_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_ipmetric_kernel(cls)` gives its kernel, unbuilt."""
    lP, lQ = _class(cls)
    return f"ipmetric_{lP}{lQ}"


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _iadd(x: Expr, y: Expr) -> Expr:
    return _c(Fn.ADD, x, y)


def _plus(*vs: tuple[int, ...]) -> tuple[int, int, int]:
    return tuple(sum(v[d] for v in vs) for d in range(3))  # type: ignore[return-value]


def _table_binds(frag, alpha: Expr, arrays: tuple[str, str, str]
                 ) -> dict[str, Expr]:
    out: dict[str, Expr] = {n: Var(m)
                            for n, m in zip(ef.CENTER_ARRAYS, arrays,
                                            strict=True)}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), alpha)
    return out


def build_ipmetric_kernel(cls: Spec | tuple[int, int], *,
                          name: str | None = None) -> Kernel:
    """The uncontracted metric derivative kernel of one aux class (module
    docstring). `cls` is `ipmetric_spec(lP, lQ)` or `(lP, lQ)`."""
    lP, lQ = _class(cls)
    L1 = lP + lQ + 1  # the tower's order
    dag, order = gmetric_plan((lP, lQ))
    fP, fQ = ef.center_tables_fragment(lP), ef.center_tables_fragment(lQ)
    nP, nQ = len(ef.cartesians(lP)), len(ef.cartesians(lQ))
    odd = lQ % 2 == 1  # (-1)^{|tau|}, one sign for the class
    pair, ip, iq = Var(PAIR), Var(BRA), Var(KET)
    shifted = [t for t in hermite_targets(lP + 1)
               if sum(t) % 2 == (lP + 1) % 2]
    targets = frozenset(_plus(t, tau) for t in shifted
                        for b in ef.cartesians(lQ)
                        for _, tau in center_cols(b, lQ))
    tw = store_targets(tower(order, dag, {node_name((r, 0)) for r in targets}),
                       targets, L1, R_ARRAY)

    def out_at(k: int, i: Expr, j: Expr, mirror: bool = False) -> Index:
        m, n = _iadd(Var("op"), i), _iadd(Var("oq"), j)
        return Index(OUT, (Lit(k), *((n, m) if mirror else (m, n))))

    # --- one Q row b: U[t'], then every (a, b) element of it -----------------
    def per_b(b: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def u_of(t: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            def r_at(tau: tuple[Expr, Expr, Expr]) -> Expr:
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _iadd(t[d], tau[d]) for d in range(3))),))
            return (*ef.rolled_center_sum(lQ, b, r_at, "u", tag="c",
                                          arrays=KET_TABLES),
                    Assign(Index(U_ARRAY, (ef.hermite_index(t),)), Var("u")))

        def elem(a: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            out: list[Stmt] = [
                Decl("ra", Scalar.I32, init=ef.cartesian_index("a"))]
            for k in range(3):
                def x(t: tuple[Expr, Expr, Expr], k=k) -> Expr:
                    return Index(U_ARRAY, (ef.hermite_index(tuple(
                        _iadd(t[d], Lit(1)) if d == k else t[d]
                        for d in range(3))),))
                at = out_at(k, Var("ra"), Var("rb"))
                out += [
                    *ef.rolled_center_sum(lP, a, x, f"v{k}", tag=f"e{k}",
                                          arrays=BRA_TABLES),
                    Assign(at, _c(Fn.SUB if odd else Fn.ADD, at,
                                  _mul(Var("w"), Var(f"v{k}")))),
                ]
            return tuple(out)

        return (
            Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),
            Note("U[t'] = SUM_tau [t'+tau]^(0) E^b_tau over the live tau of "
                 "this Q row, for every |t'| <= lP + 1 of lP + 1's parity: "
                 "once per (t', b)"),
            ef.hermite_loops(lP + 1, u_of, tag="h", parity=True),
            Note(f"per element (a, b) and axis k: M1[k] "
                 f"{'-' if odd else '+'}= s_i s_k SUM_t E^a_t U[t + 1_k] over "
                 "the live t, rolled over the P tables (the Hermite shift); "
                 f"(-1)^|tau| = (-1)^{lQ} on every live tau"),
            ef.cartesian_loops(lP, ("ax", "ay", "az"), elem, tag="a"),
        )

    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (iq,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"A{d}"),
                                      Index("A", (iq, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var("ap"), Var("aq")),
            _c(Fn.ADD, Var("ap"), Var("aq"))))),
        Note(f"[t'+tau]^(0) by the plan's DAG for the quartet ({lP + 1},0|"
             f"{lQ},0) at L + 1 = {L1}, the targets no U reads dropped, each "
             "live one stored into R as it forms"),
        *tw,
        Note("the Q record's single-center E, per axis: "
             "e_fragment.center_tables_fragment, inlined"),
        *(Decl(n, Scalar.F64, (Lit(ef.center_table_extent(lQ)),))
          for n in KET_TABLES),
        Inline(fQ, _table_binds(fQ, Var("aq"), KET_TABLES)),
        Decl("w", Scalar.F64, init=_mul(Var("sp"), Index(S_AUX, (iq,)))),
        ef.cartesian_loops(lQ, ("bx", "by", "bz"), per_b, tag="b"),
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("ap", Scalar.F64, init=Index("alpha", (ip,))),
        *(Decl(f"A{d}", Scalar.F64, init=Index("A", (ip, Lit(d))))
          for d in range(3)),
        Decl("sp", Scalar.F64, init=Index(S_AUX, (ip,))),
        Note("the P record's single-center E, per axis: "
             "e_fragment.center_tables_fragment, inlined"),
        *(Decl(n, Scalar.F64, (Lit(ef.center_table_extent(lP)),))
          for n in BRA_TABLES),
        Inline(fP, _table_binds(fP, Var("ap"), BRA_TABLES)),
        Note("the Q records of the pair"),
        For(KET, Var("q0"), Var("q1"), Lit(1), ket_body),
    )

    # --- zero the pair's slots first, mirror them last ---------------------
    i, j = Var("i"), Var("j")

    def nest(body: tuple[Stmt, ...]) -> For:
        return For("i", Lit(0), Lit(nP), Lit(1), (
            For("j", Lit(0), Lit(nQ), Lit(1), body),))

    zero = nest(tuple(Assign(out_at(k, i, j), Lit(0.0)) for k in range(3)))
    mirror = nest(tuple(
        Assign(out_at(k, i, j, mirror=True),
               _c(Fn.SUB, Lit(0.0), out_at(k, i, j))) for k in range(3)))
    scatter: Stmt
    if lP == lQ:
        scatter = If(Var("differ"), (mirror,), (Note(DIAGONAL),))
    else:
        scatter = If(Var("differ"), (mirror,))

    per_pair: tuple[Stmt, ...] = (
        Decl("p0", Scalar.I32, init=Index("p_rec0", (pair,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (pair,))),
        Decl("q0", Scalar.I32, init=Index("q_rec0", (pair,))),
        Decl("q1", Scalar.I32, init=Index("q_rec1", (pair,))),
        Decl("op", Scalar.I32, init=Index("off_p", (pair,))),
        Decl("oq", Scalar.I32, init=Index("off_q", (pair,))),
        Decl("differ", Scalar.I32, init=_c(
            Fn.ADD, _c(Fn.LT, Var("op"), Var("oq")),
            _c(Fn.LT, Var("oq"), Var("op")))),
        Note(ROLLED_NOTE),
        Note("the pair's own elements, zeroed on every axis: no other pair "
             "writes them"),
        zero,
        *(Decl(n, Scalar.F64, (Lit(len(hermite_targets(x))),))
          for n, x in ((R_ARRAY, L1), (U_ARRAY, lP + 1))),
        Note("the P records of the pair, serially"),
        For(BRA, Var("p0"), Var("p1"), Lit(1), bra_body),
        Note("the mirror, negated: M1[k, Q, P] = d(Q|P)/dB_k = -M1[k, P, Q] "
             "([r]^(0) reads A - B); a plain store, each element belongs to "
             "one aux shell pair, so no atomics"),
        scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"uncontracted derivative of the DF metric, class (lP, lQ) = "
             f"({lP}, {lQ}): M1[k, P, Q] = d(P|Q)/dA_k, A the centre of P "
             "(felizjueves.gradient)"),
        Note(lever_provenance(UNCONTRACTED)),
        Note("launch: one thread per aux shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )
    return Kernel(name or ipmetric_name((lP, lQ)), _params(), body)


def _params() -> tuple[Param, ...]:
    """`kernel_metric`'s parameter block, `M1[3, n_aux, n_aux]` the output."""

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    npair, nrec, naux = Var("n_pair"), Var("n_rec"), Var("n_aux")
    return (
        Param("n_pair", Scalar.I32),
        Param("n_rec", Scalar.I32),
        Param("n_aux", Scalar.I32),
        f64("A", (nrec, Lit(3))),
        f64("alpha", (nrec,)),
        f64(S_AUX, (nrec,)),
        *(i32(n, (npair,)) for n in ("p_rec0", "p_rec1", "q_rec0", "q_rec1",
                                     "off_p", "off_q")),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(3), naux, naux)),
    )
