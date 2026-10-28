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

THE LOW-STACK FORM (#613), the default from `L = lP + lQ >= LOWSTACK_MIN_L
= 10` (`gmetric_lowstack`; `lowstack=` forces either form, the name,
parameters, launch and `G` contract the same): no `E` block, weight or
density held. Per (P, Q) record pair the tower is stored into a local
`R[n_t(L + 1)]`; per Q row `b`, `U[t'] = SUM_tau R[t' + tau] E^b_tau` for
every `|t'| <= lP + 1` of its parity; per element `(a, b)` its weight from
`M` and `g_k +-= w wt SUM_t E^a_t U[t + 1_k]`, rolled over the P record's
single-center tables (`e_fragment.rolled_center_sum`). Measured
(data/measured/df_grad_lowstack.json): every unrolled class through L = 9
is within 14,000 B of stack on `sm_86` and `gfx1200`, (g|i) and (h|h) are
not; the low-stack (i|i) is 7,872 B on `sm_86`.

CHUNKED FOLDS (#613). Every contraction fold longer than `FOLD_CHUNK`
terms is a chain of locals (`chunked_fold`), the same left fold bitwise:
at (ff|i) one fold of `kernel_geri3`'s unrolled form is 3,360 terms, whose
`Call` nest overflowed Python's default recursion limit in the IR checker.

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
    store_targets,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "FOLD_CHUNK",
    "LOWSTACK_MIN_L",
    "build_gmetric_kernel",
    "center_binds",
    "center_cols",
    "chunked_fold",
    "gmetric_lowstack",
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


FOLD_CHUNK = 128  # terms per link of a chunked left fold (#613)


def chunked_fold(terms: list[Expr], name: str
                 ) -> tuple[list[Stmt], Expr]:
    """`_fold(terms)` as a chain of F64 locals `{name}_c{n}` of at most
    `FOLD_CHUNK` terms each, and the last link's expression: the SAME left
    fold, operation for operation (each link starts from the one before),
    so the value is bitwise `_fold`'s, but no expression is deeper than one
    chunk. A fold of at most `FOLD_CHUNK` terms is `_fold`'s text exactly.
    At (ff|i) a single fold is 3,360 terms, which overflowed Python's
    default recursion limit in the IR checker (#613)."""
    if len(terms) <= FOLD_CHUNK:
        return [], _fold(terms)
    out: list[Stmt] = []
    acc = _fold(terms[:FOLD_CHUNK])
    for n, i in enumerate(range(FOLD_CHUNK, len(terms), FOLD_CHUNK)):
        out.append(Decl(f"{name}_c{n}", Scalar.F64, init=acc))
        acc = Var(f"{name}_c{n}")
        for t in terms[i:i + FOLD_CHUNK]:
            acc = _c(Fn.ADD, acc, t)
    return out, acc


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
                         name: str | None = None,
                         lowstack: bool | None = None) -> Kernel:
    """The metric gradient kernel of one aux class (module docstring).

    `cls` is a d1 `Target.METRIC` `Spec` or `(lP, lQ)`. `lowstack` picks
    the form: `None` (the default) is `gmetric_lowstack(cls)` (THE
    LOW-STACK FORM), `True` / `False` force it. The name, parameters,
    launch and `G` contract are the same in both."""
    lP, lQ = _class(cls)
    if lowstack is None:
        lowstack = gmetric_lowstack((lP, lQ))
    if lowstack:
        return _build_lowstack((lP, lQ), name)
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
        links, total = chunked_fold(outer, f"s{k}")
        contract += links
        contract.append(Assign(g, _c(Fn.SUB if odd else Fn.ADD, g,
                                     _mul(Var("w"), total))))
    wanted: set[str] = set()
    for st in contract:
        wanted |= reads(st.value if isinstance(st, Assign) else st.init)

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
    return Kernel(name or gmetric_name((lP, lQ)), _params(), body)


def _params() -> tuple[Param, ...]:
    """The parameter list, the same in both forms (module docstring)."""

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    npair, nrec, naux, nat = (Var("n_pair"), Var("n_rec"), Var("n_aux"),
                              Var("n_atom"))
    return (
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


# ---------------------------------------------------------------------------
# The low-stack form (#613)
# ---------------------------------------------------------------------------

# every class from L = lP + lQ = 10 (#613): the smallest L with an
# inadmissible unrolled class (data/measured/df_grad_lowstack.json)
LOWSTACK_MIN_L = 10
R_ARRAY = "R"  # [r]^(0) of one (P, Q) record pair, hermite_targets(L + 1)
U_ARRAY = "U"  # SUM_tau [t'+tau] E^b_tau of one Q row b, |t'| <= lP + 1
BRA_TABLES = ("CPx", "CPy", "CPz")  # the P record's single-center tables
KET_TABLES = ("CQx", "CQy", "CQz")  # the Q record's

LOWSTACK_NOTE = (
    "LOW-STACK (#613): per-axis single-center E tables for both records in "
    "place of both E blocks and the weighted density, [r]^(0) into a local "
    "R as it forms, U[t'] of one Q row, and each element's three terms as "
    "rolled sums over the P tables through U[t + 1_k], its weight read from "
    "M where it is used: no per-record density and no per-pair weight is "
    "held")


def gmetric_lowstack(cls: Spec | tuple[int, int]) -> bool:
    """THE SELECTION RULE (#613): True when `lP + lQ >= LOWSTACK_MIN_L`;
    there `build_gmetric_kernel` emits the low-stack form by default. False
    below, where the unrolled text is unchanged."""
    return sum(_class(cls)) >= LOWSTACK_MIN_L


def _iadd(x: Expr, y: Expr) -> Expr:
    return _c(Fn.ADD, x, y)


def _table_binds(frag, alpha: Expr, arrays: tuple[str, str, str]
                 ) -> dict[str, Expr]:
    out: dict[str, Expr] = {n: Var(m)
                            for n, m in zip(ef.CENTER_ARRAYS, arrays,
                                            strict=True)}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), alpha)
    return out


def _build_lowstack(am: tuple[int, int], name: str | None) -> Kernel:
    """`build_gmetric_kernel`'s kernel with no block, density or weight
    held (module docstring, THE LOW-STACK FORM): the same parameters,
    launch and `G` contract."""
    lP, lQ = am
    L1 = lP + lQ + 1  # the tower's order
    dag, order = gmetric_plan(am)
    fP, fQ = ef.center_tables_fragment(lP), ef.center_tables_fragment(lQ)
    odd = lQ % 2 == 1
    pair, ip, iq = Var(PAIR), Var(BRA), Var(KET)
    shifted = [t for t in hermite_targets(lP + 1)
               if sum(t) % 2 == (lP + 1) % 2]
    targets = frozenset(_plus(t, tau) for t in shifted
                        for b in ef.cartesians(lQ)
                        for _, tau in center_cols(b, lQ))
    tw = store_targets(tower(order, dag, {node_name((r, 0)) for r in targets}),
                       targets, L1, R_ARRAY)

    def per_b(b: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def u_of(t: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            def r_at(tau: tuple[Expr, Expr, Expr]) -> Expr:
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _iadd(t[d], tau[d]) for d in range(3))),))
            return (*ef.rolled_center_sum(lQ, b, r_at, "u", tag="c",
                                          arrays=KET_TABLES),
                    Assign(Index(U_ARRAY, (ef.hermite_index(t),)), Var("u")))

        def elem(a: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            m, n = _iadd(Var("oa"), Var("ra")), _iadd(Var("ob"), Var("rb"))
            off = _iadd(Index("M", (m, n)), Index("M", (n, m)))
            if lP != lQ:
                weight: tuple[Stmt, ...] = (Decl("wt", Scalar.F64, init=off),)
            else:
                weight = (
                    Decl("wt", Scalar.F64, init=Lit(0.0)),
                    If(Var("differ"), (Assign(Var("wt"), off),),
                       (Assign(Var("wt"), Index("M", (m, _iadd(
                           Var("oa"), Var("rb"))))),)),
                )
            out: list[Stmt] = [
                Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                *weight,
                Decl("ww", Scalar.F64, init=_mul(Var("w"), Var("wt"))),
            ]
            for k in range(3):
                def x(t: tuple[Expr, Expr, Expr], k=k) -> Expr:
                    return Index(U_ARRAY, (ef.hermite_index(tuple(
                        _iadd(t[d], Lit(1)) if d == k else t[d]
                        for d in range(3))),))
                out += [
                    *ef.rolled_center_sum(lP, a, x, f"v{k}", tag=f"e{k}",
                                          arrays=BRA_TABLES),
                    Assign(Var(f"g{k}"), _c(Fn.SUB if odd else Fn.ADD,
                                            Var(f"g{k}"),
                                            _mul(Var("ww"), Var(f"v{k}")))),
                ]
            return tuple(out)

        return (
            Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),
            Note("U[t'] = SUM_tau [t'+tau]^(0) E^b_tau over the live tau of "
                 "this Q row, for every |t'| <= lP + 1 of lP + 1's parity: "
                 "once per (t', b)"),
            ef.hermite_loops(lP + 1, u_of, tag="h", parity=True),
            Note(f"per element (a, b), weight wt from M: g_k "
                 f"{'-' if odd else '+'}= w wt SUM_t E^a_t U[t + 1_k] over "
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
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for k in range(3):
        scatter.append(AtomicAdd(Index("G", (Var("aa"), Lit(k))), Var(f"g{k}")))
    for k in range(3):
        scatter.append(AtomicAdd(Index("G", (Var("ab"), Lit(k))),
                                 _c(Fn.SUB, Lit(0.0), Var(f"g{k}"))))
    differ: tuple[Stmt, ...] = ()
    if lP == lQ:
        differ = (Decl("differ", Scalar.I32, init=_c(
            Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
            _c(Fn.LT, Var("ob"), Var("oa")))),)
    per_pair: tuple[Stmt, ...] = (
        Decl("p0", Scalar.I32, init=Index("p_rec0", (pair,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (pair,))),
        Decl("q0", Scalar.I32, init=Index("q_rec0", (pair,))),
        Decl("q1", Scalar.I32, init=Index("q_rec1", (pair,))),
        Decl("oa", Scalar.I32, init=Index("off_p", (pair,))),
        Decl("ob", Scalar.I32, init=Index("off_q", (pair,))),
        Decl("aa", Scalar.I32, init=Index("atom_p", (pair,))),
        Decl("ab", Scalar.I32, init=Index("atom_q", (pair,))),
        *differ,
        Note(LOWSTACK_NOTE),
        *(Decl(f"g{k}", Scalar.F64, init=Lit(0.0)) for k in range(3)),
        *(Decl(n, Scalar.F64, (Lit(len(hermite_targets(x))),))
          for n, x in ((R_ARRAY, L1), (U_ARRAY, lP + 1))),
        Note("the P records of the pair, serially"),
        For(BRA, Var("p0"), Var("p1"), Lit(1), bra_body),
        Note("d/dB_k = -d/dA_k: [r]^(0) reads A - B"),
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"gradient of the DF metric, class (lP, lQ) = ({lP}, {lQ}), "
             "LOW-STACK (#613): G[A, k] += SUM_PQ W_PQ d(P|Q)/dR_{A,k} "
             "(felizjueves.gradient)"),
        Note(ZEROED_G),
        Note("launch: one thread per aux shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )
    return Kernel(name or gmetric_name(am), _params(), body)
