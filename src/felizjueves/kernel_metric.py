"""The DF Coulomb metric kernel, `Target.METRIC`: `(P|Q)` of an aux basis.

    build_metric_kernel(spec | (lP, lQ))  -> Kernel      one per aux class
    metric_spec(lP, lQ)                   -> Spec        the class's key
    quartet_spec(spec | (lP, lQ))         -> Spec        Target.J (lP 0|lQ 0)
    metric_plan(spec | (lP, lQ))          -> (dag, order)  the plan's answer
    launch_shape(n_pair, threads)         -> (grid, block)

`(P|Q) = INT INT P(r1) Q(r2) / r12` of two Cartesian auxiliary shells, by
McMurchie & Davidson's Hermite path with a single Gaussian on each side
(`ao/metric.py`'s module docstring, its witness):

    (P|Q) = SUM_{i in P} SUM_{k in Q} s_i s_k
            SUM_t E^P_t(i) SUM_tau (-1)^{|tau|} [t+tau]^(0)(A_i - A_k) E^Q_tau(k)

with `s = c (pi/alpha)^{3/2}` on each aux record (`ao.metric.aux_dress`) and
`theta^2 = alpha_i alpha_k / (alpha_i + alpha_k)`.

A METRIC CLASS IS A PAIR THAT RUNS A QUARTET (`spec.py`, #410). The key is
the pair `(lP, lQ)`; the recurrence is the `Target.J` quartet `(lP 0|lQ 0)`'s
at `L = lP + lQ`: `metric_plan` asks `pipeline.plan` for THAT spec (the
Schwarz pattern, `kernel_schwarz.quartet_spec`), and
`recurrence.fundamental_prelude` / `recurrence_body` spell its one DAG in
`schedule`'s order, unchanged.

THE OPPOSITE-PARITY TARGETS ARE DEAD. The single-center `E^a_t` is zero
whenever `a_d - t_d` is odd on some axis (`e_fragment.center_absent`), so
every live `t` of a `P` row has `|t| = lP (mod 2)` and every live `tau` of a
`Q` row `|tau| = lQ (mod 2)`: the contraction reads `[r]^(0)` only at
`|r| = L (mod 2)`, and only at `|r| >= (lP mod 2) + (lQ mod 2)` (an odd-l
row has no live `t = 0`: `(p|p)` never reads `[0]`). Exactly those targets
are read (`live_targets`), but the DAG builds the whole `|r| <= L` tower,
so the others -- and every intermediate,
fundamental and Boys weight only they read -- are DROPPED by one backward
liveness pass (`_live`, `kernel_schwarz`'s rule restated), because a declared
but unread local fails #92's filter on both compilers. Dropping a dead
`Decl` changes no value that is kept. (The issue's "nothing is dead" holds
for the reachable targets only, not for the whole tower.)

THE KET SIGN IS ONE SIGN PER CLASS. Every live `tau` has the parity of `lQ`,
so `(-1)^{|tau|} = (-1)^{lQ}` on every term: the inner sum is spelled
unsigned and the class's sign is applied once, `acc - w v` for odd `lQ`
(`acc + w v` for even). Negation is exact, so this is the same rounding as
subtracting term by term.

THE THREAD MAPPING: ONE THREAD PER AUX SHELL PAIR `(P <= Q)`, `pair =
THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by `pair < n_pair`. The thread
runs the double loop over its pair's TWO record ranges serially -- bra
`[p_rec0, p_rec1)` outer, the `P` block formed once per bra record; ket
`[q_rec0, q_rec1)` inner, the tower and the `Q` block once per (bra, ket) --
and accumulates `nP x nQ` accumulators, `acc{a}_{b}`, in registers. No
reduction and no atomics: each element belongs to exactly one aux shell
pair, so it is written by a plain store, bit-reproducible run to run.

`E` IS `e_fragment.center_fragment(l)`, INLINED TWICE -- into `Ebra` at `lP`
and `Eket` at `lQ` (kernel_schwarz's pattern: the ket's fragment locals
shadow the bra's, and only the arrays are read). Every `Ebra[a, t]` and
`Eket[b, tau]` read is a structurally nonzero entry, which the fragment
writes. `half_inv_zeta` is bound to `0.5 / alpha` of the record.

THE CONTRACTION, per (bra, ket) and element `(a, b)`:

    v_ab  = SUM_t Ebra[a, t] * (SUM_tau [t+tau]^(0) Eket[b, tau])   live t, tau
    acc_ab = acc_ab +- (s_i s_k) * v_ab                             (-1)^{lQ}

each sum a left fold in `hermite_targets` order from its first live term.
NO PREFACTOR: `(pi/alpha)^{3/2}` and the coefficients are inside the host's
`s`, `(2/sqrt(pi)) theta` inside `[0]^(m)` (the prelude's `W0`); their
product is `2 pi^{5/2} / (alpha_i alpha_k sqrt(alpha_i + alpha_k))`.

ROUND-OFF, NOT BITWISE, against `ao.metric`: the host's `[t|tau]` is
`reference.pq`'s fixed recursion, the kernel's the plan's MILD DAG, and the
host dresses each `E` entry where the kernel multiplies `s_i s_k` once.

THE SCATTER, `kernel_jab`'s rule with a plain store: an off-diagonal pair
(`off_p != off_q`) writes element `(a, b)` to `metric[off_p + a, off_q + b]`
and its mirror; a diagonal one (only when `lP == lQ`) writes its lower
triangle `a >= b`, mirrored. A pair staged SWAPPED (`ao.metric.
metric_classes`) needs no flag here: with the mirror, the transposed
placement `place_staged` makes writes the same two elements.

THE BOYS DOMAIN. Tight aux exponents far apart (`alpha = 1000` against
`alpha = 1000` two bohr away is `T = 2000`) reach far past where the old
narrow series overflowed (near 709); the one `fj_boys` every kernel carries
since #301 holds on all of `reference.boys`' domain, so nothing here asks
for a wider one.

    params  n_pair n_rec n_aux (I32)
            A[n_rec, 3]  alpha[n_rec]  s_aux[n_rec]           per aux record
            p_rec0 p_rec1 q_rec0 q_rec1 off_p off_q [n_pair]  (I32)
            metric[n_aux, n_aux]  (OUT, written once per element)

The record and pair arrays are `ao.metric.MetricClass.arrays()`' by name,
except `s`, which is `s_aux` here (`PARAM_OF`): the prelude declares a local
`s = -2 theta^2` (`recurrence.fundamental_prelude`), which would shadow it.
Elements of pairs NOT launched are untouched: a caller that launches a subset
zeroes first.

A builder: vendor-neutral IR. It imports neither `ao/` (its witness) nor the
interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves import e_fragment as ef
from felizjueves.ir import (
    Assign,
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
    "KET",
    "OUT",
    "PAIR",
    "PARAM_OF",
    "build_metric_kernel",
    "launch_shape",
    "live_targets",
    "metric_plan",
    "metric_spec",
    "quartet_spec",
]

PAIR = "pair"  # one thread per aux shell pair
BRA = "ip"  # the P record, serial
KET = "iq"  # the Q record, serial inside the bra
OUT = "metric"
S_AUX = "s_aux"  # the dressed scale's param: `s` is a prelude local
# `MetricClass.arrays()` key -> this kernel's param name, where they differ
PARAM_OF = {"s": S_AUX}

_DIAGONAL = (
    "diagonal aux shell pair (off_p == off_q): the lower triangle a >= b "
    "only, each mirrored once -- kernel_jab's diagonal rule, so (P|Q) and "
    "(Q|P) are one value")


def metric_spec(lP: int, lQ: int) -> Spec:
    """The aux class `(lP, lQ)` as a generation key, `Target.METRIC`."""
    return Spec(Operator.COULOMB, 0, (lP, lQ), Target.METRIC)


def _spec(cls: Spec | tuple[int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else metric_spec(*cls)
    if spec.target is not Target.METRIC:
        raise ValueError(f"{spec.key()} is not a Target.METRIC key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the metric kernel takes no levers")
    return spec


def quartet_spec(cls: Spec | tuple[int, int]) -> Spec:
    """The `Target.J` quartet `(lP 0 | lQ 0)` whose recurrence the class
    runs -- the key the plan layer is asked for (module docstring)."""
    lP, lQ = _spec(cls).am
    return Spec(Operator.COULOMB, 0, (lP, 0, lQ, 0), Target.J)


def metric_plan(cls: Spec | tuple[int, int]) -> tuple[BatchDag, Order]:
    """`pipeline.plan(quartet_spec(cls))`'s one DAG and `schedule(dag)`."""
    (dag,) = plan(quartet_spec(cls)).dags
    return dag, schedule(dag)


def launch_shape(n_pair: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per aux shell pair,
    `ceil(n_pair/threads)` blocks; a partial last block is legal."""
    if type(n_pair) is not int or n_pair < 1:
        raise ValueError(f"n_pair = {n_pair!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_pair // threads), (threads, 1)


def live_targets(lP: int, lQ: int) -> frozenset[tuple[int, int, int]]:
    """Every `[r]^(0)` the contraction reads: `r = t + tau` over the live
    entries of the two single-center blocks (module docstring)."""
    out: set[tuple[int, int, int]] = set()
    for a in ef.cartesians(lP):
        for b in ef.cartesians(lQ):
            for t in _live_cols(a, lP):
                for tau in _live_cols(b, lQ):
                    out.add(_plus(t[1], tau[1]))
    return frozenset(out)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _plus(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, int, int]:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def _live_cols(a: tuple[int, int, int], l: int  # noqa: E741
               ) -> list[tuple[int, tuple[int, int, int]]]:
    """`(column, t)` of row `a`'s structurally nonzero entries, in
    `hermite_targets(l)` order -- the columns the fragment writes."""
    return [(j, t) for j, t in enumerate(hermite_targets(l))
            if not ef.center_absent(a, t)]


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


def _binds(frag, alpha: Expr, array: str) -> dict[str, Expr]:
    """One inlined center block's binds: `1/2alpha` (when read) and `E`."""
    out: dict[str, Expr] = {ef.E_ARRAY: Var(array)}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), alpha)
    return out


def _acc(a: int, b: int) -> str:
    return f"acc{a}_{b}"


def build_metric_kernel(cls: Spec | tuple[int, int], *,
                        name: str | None = None) -> Kernel:
    """The metric kernel of one aux class (module docstring).

    `cls` is a `Target.METRIC` `Spec` or `(lP, lQ)`; any `lP`, `lQ` the
    fragment and the plan accept is built."""
    spec = _spec(cls)
    lP, lQ = spec.am
    dag, order = metric_plan(spec)
    fP, fQ = ef.center_fragment(lP), ef.center_fragment(lQ)
    rowsP, rowsQ = ef.cartesians(lP), ef.cartesians(lQ)
    nP, nQ = len(rowsP), len(rowsQ)
    odd = lQ % 2 == 1  # (-1)^{|tau|}, one sign for the class
    pair, ip, iq = Var(PAIR), Var(BRA), Var(KET)

    # --- the contraction of one (bra, ket) record pair, per element --------
    contract: list[Stmt] = []
    for a, ra in enumerate(rowsP):
        for b, rb in enumerate(rowsQ):
            v: Expr | None = None
            for i, t in _live_cols(ra, lP):
                inner: Expr | None = None
                for j, tau in _live_cols(rb, lQ):
                    term = _c(Fn.MUL, Var(node_name((_plus(t, tau), 0))),
                              Index("Eket", (Lit(b), Lit(j))))
                    inner = term if inner is None else _c(Fn.ADD, inner, term)
                assert inner is not None  # every row has a live column
                term = _c(Fn.MUL, Index("Ebra", (Lit(a), Lit(i))), inner)
                v = term if v is None else _c(Fn.ADD, v, term)
            assert v is not None
            acc = Var(_acc(a, b))
            contract.append(Assign(acc, _c(
                Fn.SUB if odd else Fn.ADD, acc, _c(Fn.MUL, Var("w"), v))))
    wanted: set[str] = set()
    for st in contract:
        wanted |= _reads(st.value)

    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tower = _live((*prelude, *recurrence_body(dag, order, fundamental)),
                  wanted)

    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (iq,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"A{d}"),
                                      Index("A", (iq, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Note("theta^2 = alpha_P alpha_Q / (alpha_P + alpha_Q): a single "
             "Gaussian on each side is a Hermite pair at zeta = alpha"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ap"), Var("aq")),
            _c(Fn.ADD, Var("ap"), Var("aq"))))),
        Note("[t+tau]^(0) by the plan's DAG for the quartet "
             f"({lP},0|{lQ},0), the targets no live t + tau reaches (the "
             "center E's parity zeros) dropped with everything only they "
             "read"),
        *tower,
        Note("the undressed single-center E block of this Q record: "
             "e_fragment.center_fragment, inlined"),
        Decl("Eket", Scalar.F64, (Lit(nQ), Lit(len(hermite_targets(lQ))))),
        Inline(fQ, _binds(fQ, Var("aq"), "Eket")),
        Decl("w", Scalar.F64, init=_c(Fn.MUL, Var("sp"), Index(S_AUX, (iq,)))),
        Note(f"per element (a, b): acc {'-' if odd else '+'}= s_i s_k "
             "SUM_t Ebra_t SUM_tau [t+tau] Eket_tau over the live t and tau; "
             f"(-1)^|tau| = (-1)^{lQ} on every live tau"),
        *contract,
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("ap", Scalar.F64, init=Index("alpha", (ip,))),
        *(Decl(f"A{d}", Scalar.F64, init=Index("A", (ip, Lit(d))))
          for d in range(3)),
        Decl("sp", Scalar.F64, init=Index(S_AUX, (ip,))),
        Note("the undressed single-center E block of this P record, once: "
             "e_fragment.center_fragment, inlined"),
        Decl("Ebra", Scalar.F64, (Lit(nP), Lit(len(hermite_targets(lP))))),
        Inline(fP, _binds(fP, Var("ap"), "Ebra")),
        Note("the Q records of the pair"),
        For(KET, Var("q0"), Var("q1"), Lit(1), ket_body),
    )

    # --- the scatter --------------------------------------------------------
    def at(row: Expr, col: Expr) -> Index:
        return Index(OUT, (row, col))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    off_diag: list[Stmt] = []
    for a in range(nP):
        for b in range(nQ):
            v = Var(_acc(a, b))
            off_diag += [Assign(at(plus("op", a), plus("oq", b)), v),
                         Assign(at(plus("oq", b), plus("op", a)), v)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("op"), Var("oq")),
                _c(Fn.LT, Var("oq"), Var("op")))
    scatter: Stmt
    if lP == lQ:
        diag: list[Stmt] = [Note(_DIAGONAL)]
        for a in range(nP):
            for b in range(a + 1):
                v = Var(_acc(a, b))
                diag.append(Assign(at(plus("op", a), plus("op", b)), v))
                if a != b:
                    diag.append(Assign(at(plus("op", b), plus("op", a)), v))
        scatter = If(differ, tuple(off_diag), tuple(diag))
    else:
        scatter = If(differ, tuple(off_diag))

    per_pair: tuple[Stmt, ...] = (
        Decl("p0", Scalar.I32, init=Index("p_rec0", (pair,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (pair,))),
        Decl("q0", Scalar.I32, init=Index("q_rec0", (pair,))),
        Decl("q1", Scalar.I32, init=Index("q_rec1", (pair,))),
        Decl("op", Scalar.I32, init=Index("off_p", (pair,))),
        Decl("oq", Scalar.I32, init=Index("off_q", (pair,))),
        *(Decl(_acc(a, b), Scalar.F64, init=Lit(0.0))
          for a in range(nP) for b in range(nQ)),
        Note("the P records of the pair, serially"),
        For(BRA, Var("p0"), Var("p1"), Lit(1), bra_body),
        Note("scatter: one plain store per element and its mirror, no "
             "atomics -- each element belongs to one aux shell pair"),
        scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"DF Coulomb metric, class (lP, lQ) = ({lP}, {lQ}): (P|Q) by "
             f"the quartet ({lP},0|{lQ},0)'s recurrence at L = {lP + lQ} "
             "(ao/metric.py)"),
        Note("launch: one thread per aux shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    npair, nrec, naux = Var("n_pair"), Var("n_rec"), Var("n_aux")
    params = (
        Param("n_pair", Scalar.I32),
        Param("n_rec", Scalar.I32),
        Param("n_aux", Scalar.I32),
        f64("A", (nrec, Lit(3))),
        f64("alpha", (nrec,)),
        f64(S_AUX, (nrec,)),
        *(i32(n, (npair,)) for n in ("p_rec0", "p_rec1", "q_rec0", "q_rec1",
                                     "off_p", "off_q")),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(naux, naux)),
    )
    return Kernel(name or f"metric_{lP}{lQ}", params, body)
