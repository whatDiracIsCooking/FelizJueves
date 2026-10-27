"""The DF metric Hessian kernel: `Target.METRIC` at `derivative_order = 2`.

    build_d2metric_kernel(spec | (lP, lQ)) -> Kernel        one per aux class
    d2metric_name(spec | (lP, lQ))         -> str           d2metric_{lP}{lQ}
    d2metric_spec(lP, lQ)                  -> Spec          the class's d2 key
    d2metric_plan(spec | (lP, lQ))         -> (dag, order)  the tower two up

Per aux shell pair `(P, Q)` of one class, staged as `kernel_metric` stages it
(`eri.density_fit.metric.metric_classes`), the pair's contribution to

    H[A, k, B, l] += SUM_PQ W_PQ d2(P|Q) / dR_{A,k} dR_{B,l}

(`felizjueves.hessian`; `hessian.host_df.metric_hessian` is the reference,
by another route). `W` is any `n_aux x n_aux` matrix; a DF-J energy
Hessian's metric term passes `-1/2 d d^T`, skeleton only.

The name is `d2metric_`, not `hmetric_`: that prefix is the HGP metric's
(`eri.density_fit.kernel_hgp`), as `kernel_h1e`'s `d2s_` is not `hs_`.

THE HERMITE SHIFT, TWICE. A single-center `E^a_t` does not depend on where
its Gaussian sits (`kernel_gmetric`'s docstring), so moving `P` moves only
the Hermite Gaussians, `d/dA_k Lambda_t(A) = Lambda_{t+1_k}`, and `[r]^(0)`
reads `A - B`:

    d2(P|Q)/dA_k dA_l = SUM_t E^P_t SUM_tau (-1)^{|tau|}
                        [t + tau + 1_k + 1_l]^(0) E^Q_tau  =: h_kl
    d2(P|Q)/dA_k dB_l = - h_kl,     d2(P|Q)/dB_k dB_l = + h_kl

`kernel_gmetric`'s contraction with the tower read two indices up -- no
raised `E`, no derivative table, and `h` symmetric in `k <-> l`, so six
contractions per pair, not nine. Exact, not translational invariance
assumed: `SUM_A H[A, k, B, l]` is zero by construction here, and is checked
against `host_df`, which differentiates both centres by the Cartesian rule.

THE TOWER is the plan of the `Target.J` quartet `(lP + 2, 0 | lQ, 0)` at
`L + 2` (`d2metric_plan`), its targets no `t + tau + 1_k + 1_l` reaches
dropped by `kernel_gmetric.live_body`. Every read target has the parity of
`L`.

PER RECORD PAIR, with `w_ab` the weights (`kernel_g1e.weights`' rule on
`W`) formed once per pair, and the bra's weighted Hermite density
`H_b,t = SUM_a w_ab Ebra[a, t]` once per bra record (`kernel_gmetric`'s),

    v_kl = SUM_b SUM_tau Eket[b, tau] SUM_t H_b,t [t + tau + 1_k + 1_l]^(0)
    h_kl = h_kl +- (s_i s_k) v_kl                  k <= l, (-1)^{lQ}

and once per pair, with `X = atom_p`, `Y = atom_q`, for all nine `(k, l)`
(`h_lk = h_kl`): `H[X, k, X, l] += h`, `H[Y, k, Y, l] += h`, `H[X, k, Y, l]
+= (0 - h)`, `H[Y, k, X, l] += (0 - h)`: 36 `AtomicAdd`s, because every
pair on an atom adds into its blocks. `H` is ZEROED BY THE CALLER.

THE THREAD MAPPING is `kernel_metric`'s and `kernel_gmetric`'s: one thread
per aux shell pair, both record ranges serially, bra outer. A diagonal pair
(`off_p == off_q`) takes every `(a, b)` at weight `W[op + a, op + b]` and
adds to one atom's block four times, summing to zero. A pair staged SWAPPED
needs no flag: the weight rule is symmetric in it and `atom_p` / `atom_q`
follow the staged order (`gradient.host_df.metric_records`).

ROUND-OFF, NOT BITWISE, against `host_df`: another route (above), the
plan's MILD DAG against `reference.pq`, and the weights folded in first.

    params  n_pair n_rec n_aux n_atom (I32)
            A[n_rec, 3]  alpha[n_rec]  s_aux[n_rec]                per record
            p_rec0 p_rec1 q_rec0 q_rec1 off_p off_q atom_p atom_q  [n_pair] (I32)
            M[n_aux, n_aux]                          the contracted matrix W
            H[n_atom, 3, n_atom, 3]  (OUT, read and AtomicAdd-ed)

`kernel_gmetric`'s parameter block, `H` in place of `G`.

A builder: vendor-neutral IR. It imports neither `ao/` nor `hessian.host_df`
(its witness) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_metric import S_AUX
from felizjueves.gradient.kernel_g1e import weights
from felizjueves.gradient.kernel_gmetric import (
    BRA,
    KET,
    PAIR,
    center_binds,
    center_cols,
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
    "ATOMIC_REASON",
    "KL",
    "ZEROED_H",
    "build_d2metric_kernel",
    "d2metric_name",
    "d2metric_plan",
    "d2metric_spec",
]

ATOMIC_REASON = (
    "AtomicAdd: every aux shell pair on an atom adds into that atom's "
    "blocks of H")

ZEROED_H = (
    "PRECONDITION: H is zeroed by the caller before the first hessian "
    "launch of a build; this kernel only adds")

# The six components of the symmetric `h_kl`, `k <= l`.
KL: tuple[tuple[int, int], ...] = tuple(
    (k, l) for k in range(3) for l in range(k, 3))  # noqa: E741


def d2metric_spec(lP: int, lQ: int) -> Spec:
    """The aux class `(lP, lQ)` at derivative order 2, `Target.METRIC`."""
    return Spec(Operator.COULOMB, 2, (lP, lQ), Target.METRIC)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else d2metric_spec(*cls)
    if spec.target is not Target.METRIC or spec.derivative_order != 2:
        raise ValueError(f"{spec.key()} is not a d2 Target.METRIC key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the hessian kernels take no levers")
    lP, lQ = spec.am
    return lP, lQ


def d2metric_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_d2metric_kernel(cls)` gives its kernel, unbuilt."""
    lP, lQ = _class(cls)
    return f"d2metric_{lP}{lQ}"


def d2metric_plan(cls: Spec | tuple[int, int]) -> tuple[BatchDag, Order]:
    """The plan of the quartet `(lP + 2, 0 | lQ, 0)`: the tower at `L + 2`."""
    lP, lQ = _class(cls)
    (dag,) = plan(Spec(Operator.COULOMB, 0, (lP + 2, 0, lQ, 0),
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


def _h(k: int, l: int) -> Var:  # noqa: E741
    return Var(f"h{min(k, l)}{max(k, l)}")


def build_d2metric_kernel(cls: Spec | tuple[int, int], *,
                         name: str | None = None) -> Kernel:
    """The metric Hessian kernel of one aux class (module docstring).

    `cls` is a d2 `Target.METRIC` `Spec` or `(lP, lQ)`."""
    lP, lQ = _class(cls)
    dag, order = d2metric_plan((lP, lQ))
    fP, fQ = ef.center_fragment(lP), ef.center_fragment(lQ)
    rowsP, rowsQ = ef.cartesians(lP), ef.cartesians(lQ)
    nP, nQ = len(rowsP), len(rowsQ)
    odd = lQ % 2 == 1  # (-1)^{|tau|}, one sign for the class
    ip, iq = Var(BRA), Var(KET)
    tsP = hermite_targets(lP)

    # --- the bra's weighted density, per bra record (kernel_gmetric's) ------
    dens: list[Stmt] = []
    Hcols: dict[int, list[int]] = {}  # b -> the columns i with an H_b,i
    for b in range(nQ):
        Hcols[b] = []
        for i, t in enumerate(tsP):
            terms = [_mul(Var(f"w{a * nQ + b}"), Index("Ebra", (Lit(a), Lit(i))))
                     for a, ra in enumerate(rowsP)
                     if not ef.center_absent(ra, t)]
            if terms:
                Hcols[b].append(i)
                dens.append(Decl(f"H{b}_{i}", Scalar.F64, init=_fold(terms)))

    # --- the six contractions per (bra, ket) record pair ---------------------
    contract: list[Stmt] = []
    for k, l in KL:  # noqa: E741
        up = _plus(unit(k), unit(l))
        outer: list[Expr] = []
        for b, rb in enumerate(rowsQ):
            for j, tau in center_cols(rb, lQ):
                inner = _fold([_mul(Var(f"H{b}_{i}"), Var(node_name(
                    (_plus(tsP[i], tau, up), 0)))) for i in Hcols[b]])
                outer.append(_mul(Index("Eket", (Lit(b), Lit(j))), inner))
        h = _h(k, l)
        contract.append(Assign(h, _c(Fn.SUB if odd else Fn.ADD, h,
                                     _mul(Var("w"), _fold(outer)))))
    wanted: set[str] = set()
    for st in contract:
        wanted |= reads(st.value)

    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (iq,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"A{d}"),
                                      Index("A", (iq, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var("ap"), Var("aq")),
            _c(Fn.ADD, Var("ap"), Var("aq"))))),
        Note(f"[t+tau+1_k+1_l]^(0) by the plan's DAG for the quartet "
             f"({lP + 2},0|{lQ},0) at L + 2 = {lP + lQ + 2}, the targets no "
             "live t + tau + 1_k + 1_l reaches dropped with everything only "
             "they read"),
        *tower(order, dag, wanted),
        Decl("Eket", Scalar.F64, (Lit(nQ), Lit(len(hermite_targets(lQ))))),
        Inline(fQ, center_binds(fQ, Var("aq"), "Eket")),
        Decl("w", Scalar.F64, init=_mul(Var("sp"), Index(S_AUX, (iq,)))),
        Note(f"h_kl {'-' if odd else '+'}= s_i s_k SUM_b SUM_tau Eket_b,tau "
             "SUM_t H_b,t [t+tau+1_k+1_l]: d2(P|Q)/dA_k dA_l by the Hermite "
             f"shift twice, k <= l; (-1)^|tau| = (-1)^{lQ} on every live tau"),
        *contract,
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("ap", Scalar.F64, init=Index("alpha", (ip,))),
        *(Decl(f"A{d}", Scalar.F64, init=Index("A", (ip, Lit(d))))
          for d in range(3)),
        Decl("sp", Scalar.F64, init=Index(S_AUX, (ip,))),
        Decl("Ebra", Scalar.F64, (Lit(nP), Lit(len(tsP)))),
        Inline(fP, center_binds(fP, Var("ap"), "Ebra")),
        Note("H_b,t = SUM_a w_ab Ebra_a,t: the weighted bra density, once "
             "per bra record"),
        *dens,
        For(KET, Var("q0"), Var("q1"), Lit(1), ket_body),
    )
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for X, Y, neg in (("aa", "aa", False), ("ab", "ab", False),
                      ("aa", "ab", True), ("ab", "aa", True)):
        for k in range(3):
            for l in range(3):  # noqa: E741
                v: Expr = _c(Fn.SUB, Lit(0.0), _h(k, l)) if neg else _h(k, l)
                scatter.append(AtomicAdd(Index("H", (Var(X), Lit(k), Var(Y),
                                                     Lit(l))), v))
    pair = Var(PAIR)
    per_pair: tuple[Stmt, ...] = (
        Decl("p0", Scalar.I32, init=Index("p_rec0", (pair,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (pair,))),
        Decl("q0", Scalar.I32, init=Index("q_rec0", (pair,))),
        Decl("q1", Scalar.I32, init=Index("q_rec1", (pair,))),
        Decl("oa", Scalar.I32, init=Index("off_p", (pair,))),
        Decl("ob", Scalar.I32, init=Index("off_q", (pair,))),
        Decl("aa", Scalar.I32, init=Index("atom_p", (pair,))),
        Decl("ab", Scalar.I32, init=Index("atom_q", (pair,))),
        *weights(lP, lQ),
        *(Decl(_h(*kl).name, Scalar.F64, init=Lit(0.0)) for kl in KL),
        Note("the P records of the pair, serially"),
        For(BRA, Var("p0"), Var("p1"), Lit(1), bra_body),
        Note("d2/dA dB = -h, d2/dB dB = +h: [r]^(0) reads A - B; h_lk = h_kl"),
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"hessian of the DF metric, class (lP, lQ) = ({lP}, {lQ}): "
             "H[A, k, B, l] += SUM_PQ W_PQ d2(P|Q)/dR_{A,k} dR_{B,l} "
             "(felizjueves.hessian)"),
        Note(ZEROED_H),
        Note("launch: one thread per aux shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    npair, nrec, naux, nat = (Var("n_pair"), Var("n_rec"), Var("n_aux"),
                              Var("n_atom"))
    params = (
        Param("n_pair", Scalar.I32),
        Param("n_rec", Scalar.I32),
        Param("n_aux", Scalar.I32),
        Param("n_atom", Scalar.I32),
        f64("A", (nrec, Lit(3))),
        f64("alpha", (nrec,)),
        f64(S_AUX, (nrec,)),
        *(i32(n, (npair,)) for n in ("p_rec0", "p_rec1", "q_rec0", "q_rec1",
                                     "off_p", "off_q", "atom_p", "atom_q")),
        f64("M", (naux, naux)),
        Param("H", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3), nat, Lit(3))),
    )
    return Kernel(name or d2metric_name((lP, lQ)), params, body)
