"""The DF metric gradient kernel: `Target.METRIC` at `derivative_order = 1`.

    build_gmetric_kernel(spec | (lP, lQ)) -> Kernel        one per aux class
    gmetric_name(spec | (lP, lQ))         -> str           gmetric_{lP}{lQ}
    gmetric_spec(lP, lQ)                  -> Spec          the class's d1 key
    gmetric_plan(spec | (lP, lQ))         -> (dag, order)  the tower one up
    launch_shape(n_pair, threads)         -> (grid, block)
    reads(expr) / live_body(body, wanted)                  the liveness pass

Per aux shell pair `(P, Q)` of one class, staged as `kernel_metric` stages it
(`eri.density_fit.metric.metric_classes`), the pair's contribution to

    G[A, k] += SUM_PQ W_PQ d(P|Q) / dR_{A,k}

(`felizjueves.gradient`; `gradient.host_df.metric_gradient` is the
reference, by another route). `W` is any `n_aux x n_aux` matrix; a DF-J
gradient passes `-1/2 d d^T` (`host_df`'s module docstring).

THE HERMITE SHIFT. A single-center `E^a_t` does not depend on where its
Gaussian sits (`e_fragment.center_fragment`, `X_PA = 0`), so moving `P`
moves only the Hermite Gaussians, `d/dA_k Lambda_t(A) = Lambda_{t+1_k}`:

    d(P|Q)/dA_k = SUM_t E^P_t SUM_tau (-1)^{|tau|} [t + tau + 1_k]^(0) E^Q_tau
    d(P|Q)/dB_k = - d(P|Q)/dA_k                    ([r]^(0) reads A - B)

the d0 kernel's contraction with the tower read one index up -- no raised
`E`, no derivative table. The second line is the same identity for the
other centre, not translational invariance assumed: it is exact, so
`SUM_A G[A]` is zero by construction here and is checked against
`host_df`, which differentiates both centres by the Cartesian rule.

THE TOWER is the plan of the `Target.J` quartet `(lP + 1, 0 | lQ, 0)` at
`L + 1` (`gmetric_plan`, `kernel_metric.metric_plan`'s pattern one order
up), its targets no `t + tau + 1_k` reaches dropped by `live_body`. Every
read target has the parity of `L + 1`.

PER RECORD PAIR, with `w_ab` the weights (`kernel_g1e.weights`' rule on
`W`, the class's pair `(lP, lQ)`), formed once per pair, and the bra's
weighted Hermite density formed once per bra record,

    H_b,t  = SUM_a w_ab Ebra[a, t]                          live t of P
    v_k    = SUM_b SUM_tau Eket[b, tau] SUM_t H_b,t [t + tau + 1_k]^(0)
    g_k    = g_k +- (s_i s_k) v_k                           (-1)^{lQ}

and once per pair `G[atom_p, k] += g_k`, `G[atom_q, k] += (0 - g_k)`: six
`AtomicAdd`s, because every pair on an atom adds into its row. `G` is ZEROED
BY THE CALLER.

THE THREAD MAPPING is `kernel_metric`'s: one thread per aux shell pair, both
record ranges serially, bra outer. A diagonal pair (`off_p == off_q`) takes
every `(a, b)` at weight `W[op + a, op + b]` and adds to one atom twice,
with opposite signs. A pair staged SWAPPED needs no flag: the weight rule
is symmetric in it and `atom_p` / `atom_q` follow the staged order
(`gradient.host_df.metric_records`).

ROUND-OFF, NOT BITWISE, against `host_df`: another route (above), the
plan's MILD DAG against `reference.pq`, and the weights folded in first.

    params  n_pair n_rec n_aux n_atom (I32)
            A[n_rec, 3]  alpha[n_rec]  s_aux[n_rec]                per record
            p_rec0 p_rec1 q_rec0 q_rec1 off_p off_q atom_p atom_q  [n_pair] (I32)
            M[n_aux, n_aux]                          the contracted matrix W
            G[n_atom, 3]  (OUT, read and AtomicAdd-ed)

The record and pair arrays are `MetricClass.arrays()`' (`s` as `s_aux`,
`kernel_metric.PARAM_OF`) plus `host_df.metric_records`' atoms.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host_df`
(its witness) nor the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_metric import S_AUX
from felizjueves.gradient.kernel_g1e import ATOMIC_REASON, ZEROED_G, weights
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Boys,
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
    Weighted,
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
    "build_gmetric_kernel",
    "center_binds",
    "center_cols",
    "gmetric_name",
    "gmetric_plan",
    "gmetric_spec",
    "launch_shape",
    "live_body",
    "reads",
    "tower",
    "unit",
]

PAIR = "pair"  # one thread per aux shell pair
BRA = "ip"  # the P record, serial
KET = "iq"  # the Q record, serial inside the bra


def gmetric_spec(lP: int, lQ: int) -> Spec:
    """The aux class `(lP, lQ)` at derivative order 1, `Target.METRIC`."""
    return Spec(Operator.COULOMB, 1, (lP, lQ), Target.METRIC)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else gmetric_spec(*cls)
    if spec.target is not Target.METRIC or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 Target.METRIC key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    lP, lQ = spec.am
    return lP, lQ


def gmetric_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_gmetric_kernel(cls)` gives its kernel, unbuilt."""
    lP, lQ = _class(cls)
    return f"gmetric_{lP}{lQ}"


def gmetric_plan(cls: Spec | tuple[int, int]) -> tuple[BatchDag, Order]:
    """The plan of the quartet `(lP + 1, 0 | lQ, 0)`: the tower at `L + 1`."""
    lP, lQ = _class(cls)
    (dag,) = plan(Spec(Operator.COULOMB, 0, (lP + 1, 0, lQ, 0),
                       Target.J)).dags
    return dag, schedule(dag)


def launch_shape(n: int, threads: int = 32) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per pair (or cell),
    `ceil(n/threads)` blocks; a partial last block is legal."""
    if type(n) is not int or n < 1:
        raise ValueError(f"n = {n!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n // threads), (threads, 1)


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


def unit(k: int) -> tuple[int, int, int]:
    """`1_k`: the Hermite index one up on axis `k`."""
    return tuple(int(d == k) for d in range(3))  # type: ignore[return-value]


def center_cols(a: tuple[int, int, int], l: int  # noqa: E741
                ) -> list[tuple[int, tuple[int, int, int]]]:
    """`(column, t)` of single-center row `a`'s structurally nonzero
    entries, in `hermite_targets(l)` order -- the columns
    `center_fragment` writes."""
    return [(j, t) for j, t in enumerate(hermite_targets(l))
            if not ef.center_absent(a, t)]


def reads(e: object) -> set[str]:
    """Every name an expression reads: `Var`s, arrays, `Factor` scalars
    (`kernel_metric`'s rule, restated public for the d1 kernels)."""
    if isinstance(e, Var):
        return {e.name}
    if isinstance(e, Index):
        return {e.array}.union(*(reads(i) for i in e.at))
    if isinstance(e, Call):
        return set().union(*(reads(a) for a in e.args))
    if isinstance(e, Weighted):
        out: set[str] = set()
        for t in e.terms:
            out |= reads(t.src)
            if t.g.name_in_kernel is not None:
                out.add(t.g.name_in_kernel)
        return out
    return set()  # Lit


def live_body(body: tuple[Stmt, ...], wanted: set[str]) -> tuple[Stmt, ...]:
    """`body` (`Decl`s, `Note`s and the prelude's `Boys` fill) without what
    neither `wanted` nor a kept later statement reads: one backward pass
    (`kernel_metric._live`'s rule, restated)."""
    kept: list[Stmt] = []
    live = set(wanted)
    for st in reversed(body):
        if isinstance(st, Decl):
            if st.name not in live:
                continue
            live |= reads(st.init)
        elif isinstance(st, Boys):
            if st.dst not in live:
                continue
            live |= reads(st.t)
        elif not isinstance(st, Note):
            raise TypeError(f"live_body takes Decls, Notes and Boys, got "
                            f"{type(st).__name__}")
        kept.append(st)
    return tuple(reversed(kept))


def tower(order: Order, dag: BatchDag, wanted: set[str]) -> tuple[Stmt, ...]:
    """The plan's prelude and recurrence, cut to what `wanted` reads."""
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    return live_body((*prelude, *recurrence_body(dag, order, fundamental)),
                     wanted)


def center_binds(frag, alpha: Expr, array: str) -> dict[str, Expr]:
    """One inlined single-center block's binds: `1/2alpha` and `E`."""
    out: dict[str, Expr] = {ef.E_ARRAY: Var(array)}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), alpha)
    return out


def build_gmetric_kernel(cls: Spec | tuple[int, int], *,
                         name: str | None = None) -> Kernel:
    """The metric gradient kernel of one aux class (module docstring).

    `cls` is a d1 `Target.METRIC` `Spec` or `(lP, lQ)`."""
    lP, lQ = _class(cls)
    dag, order = gmetric_plan((lP, lQ))
    fP, fQ = ef.center_fragment(lP), ef.center_fragment(lQ)
    rowsP, rowsQ = ef.cartesians(lP), ef.cartesians(lQ)
    nP, nQ = len(rowsP), len(rowsQ)
    odd = lQ % 2 == 1  # (-1)^{|tau|}, one sign for the class
    ip, iq = Var(BRA), Var(KET)
    tsP = hermite_targets(lP)

    # --- the bra's weighted density, per bra record -------------------------
    dens: list[Stmt] = []
    H: dict[int, list[int]] = {}  # b -> the columns i with an H_b,i
    for b in range(nQ):
        H[b] = []
        for i, t in enumerate(tsP):
            terms = [_mul(Var(f"w{a * nQ + b}"), Index("Ebra", (Lit(a), Lit(i))))
                     for a, ra in enumerate(rowsP)
                     if not ef.center_absent(ra, t)]
            if terms:
                H[b].append(i)
                dens.append(Decl(f"H{b}_{i}", Scalar.F64, init=_fold(terms)))

    # --- the contraction per (bra, ket) record pair --------------------------
    contract: list[Stmt] = []
    for k in range(3):
        outer: list[Expr] = []
        for b, rb in enumerate(rowsQ):
            for j, tau in center_cols(rb, lQ):
                inner = _fold([_mul(Var(f"H{b}_{i}"), Var(node_name(
                    (_plus(tsP[i], tau, unit(k)), 0)))) for i in H[b]])
                outer.append(_mul(Index("Eket", (Lit(b), Lit(j))), inner))
        g = Var(f"g{k}")
        contract.append(Assign(g, _c(Fn.SUB if odd else Fn.ADD, g,
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
        Note(f"[t+tau+1_k]^(0) by the plan's DAG for the quartet ({lP + 1},0|"
             f"{lQ},0) at L + 1 = {lP + lQ + 1}, the targets no live "
             "t + tau + 1_k reaches dropped with everything only they read"),
        *tower(order, dag, wanted),
        Decl("Eket", Scalar.F64, (Lit(nQ), Lit(len(hermite_targets(lQ))))),
        Inline(fQ, center_binds(fQ, Var("aq"), "Eket")),
        Decl("w", Scalar.F64, init=_mul(Var("sp"), Index(S_AUX, (iq,)))),
        Note(f"g_k {'-' if odd else '+'}= s_i s_k SUM_b SUM_tau Eket_b,tau "
             "SUM_t H_b,t [t+tau+1_k]: d(P|Q)/dA_k by the Hermite shift; "
             f"(-1)^|tau| = (-1)^{lQ} on every live tau"),
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
    for k in range(3):
        scatter.append(AtomicAdd(Index("G", (Var("aa"), Lit(k))), Var(f"g{k}")))
    for k in range(3):
        scatter.append(AtomicAdd(Index("G", (Var("ab"), Lit(k))),
                                 _c(Fn.SUB, Lit(0.0), Var(f"g{k}"))))
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
        *(Decl(f"g{k}", Scalar.F64, init=Lit(0.0)) for k in range(3)),
        Note("the P records of the pair, serially"),
        For(BRA, Var("p0"), Var("p1"), Lit(1), bra_body),
        Note("d/dB_k = -d/dA_k: [r]^(0) reads A - B"),
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"gradient of the DF metric, class (lP, lQ) = ({lP}, {lQ}): "
             "G[A, k] += SUM_PQ W_PQ d(P|Q)/dR_{A,k} (felizjueves.gradient)"),
        Note(ZEROED_G),
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
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3))),
    )
    return Kernel(name or gmetric_name((lP, lQ)), params, body)
