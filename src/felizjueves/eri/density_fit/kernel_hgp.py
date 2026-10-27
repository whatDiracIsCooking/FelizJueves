"""The DF integrals on the HGP engine: `(P|Q)` and `(mn|P)` by VRR and HRR.

    build_hmetric_kernel(spec | (lP, lQ))     -> Kernel   `hmetric_{lP}{lQ}`
    build_heri3_kernel(spec | (la, lb, lP))   -> Kernel   `heri3_{la}{lb}{lP}`
    hmetric_spec(lP, lQ) / heri3_spec(la, lb, lP) -> Spec  (engine HGP)
    metric_class(spec) / eri3_class(spec)     -> the HGP class it builds
    HGP_DF_L_MAX                               -> 1: every shell s or p

The two density-fitting targets of milestones 15 and 16, `Target.METRIC` and
`Target.ERI3`, on the Head-Gordon-Pople recurrence of `[Fock]` §3.1 instead
of McMurchie & Davidson's Hermite path (`kernel_metric`, `kernel_eri3`).
Each is a SIBLING of its MMD kernel, not a lever: the same class (keyed with
`engine=Engine.HGP`, so the slot spells `metric.hgp` / `eri3.hgp`), the
same parameters in the same order, the same staging
(`eri.density_fit.metric.MetricClass.arrays()` /
`eri.density_fit.eri3.Eri3Class.arrays()`, `s` passed as `s_aux`), the same
thread mapping and the same plain mirrored stores into the same output. A
caller swaps one builder for the other and changes nothing else
(`metric_drive` / `eri3_drive` take `engine=`).

A DF CLASS IS AN HGP QUARTET WITH EXPONENT-0 PARTNERS. An auxiliary
Gaussian `P` is the primitive pair `(P, 1)` whose second member is an s
function of exponent 0 on the same centre: `p = alpha`, `P = A`, `K = 1`,
and the pair factor `c (pi/alpha)^{3/2}` is exactly the aux record's `s`
(`eri.density_fit.metric.aux_dress`). So

    (P|Q)  = (lP 0 | lQ 0)    the VRR alone (`b = d = 0`: no HRR)
    (mn|P) = (la lb | lP 0)   the VRR to `la <= |e| <= la + lb`, then the
                              bra HRR (eq 28) on contracted values

and on the aux side of both `QC = Q - C = 0` (and, for the metric, the
bra's `PA = P - A = 0` too). Every VRR term whose factor is one of those
zeros is DROPPED (`_without`): it is `0 * x`, exactly zero, and with it goes
every node, Boys order and quartet scalar only such a term read (one
backward liveness pass, `kernel_metric._live`, because a declared but
unread local fails #92's filter on both compilers). The DAG is
`hgp.vrr.build`'s, unchanged; only its zero terms are not spelled.

THE BRA ORIENTATION (`eri3_class`). `(mn|P) = (nm|P)`, so the class is built
with the HIGHER `l` first, `(max(la, lb), min(la, lb) | lP 0)`: the VRR's bra
range then starts at the higher `l` and the HRR moves units onto the LOWER
shell, which is nothing when it is s. The staging keeps `la <= lb`, so a
swapped build reads the record's `XPB` as its `PA` (`P - B`), takes `AB = B
- A`, and stores element `(a, b)` from the built `(b a|q)`. Through p every
class but `(pp|P)` therefore has no HRR at all.

THE QUARTET SCALARS, per (bra record, aux record), `eri/hgp_drive.py`'s
binding (`hgp.recurrence`'s module docstring, "WHAT THE SCALARS ARE") with
`q = alpha_k` and `Q = C = A_k`:

    PQ = P - A_k    rho = p q/(p + q)    theta = sqrt(rho)
    WP = -(q/(p+q)) PQ    WQ = +(p/(p+q)) PQ   (`W - Q` derived, #291)
    inv2p = 1/(2p)  inv2q = 1/(2q)  inv2pq = 1/(2(p+q))
    inv2p_alpha_over_p = inv2p (rho/p)    inv2q_alpha_over_q = inv2q (rho/q)
    pair = U_P U_Q

`U_Q` is the aux record's `s_aux`. For the metric `U_P` is the bra aux
record's `s_aux` (`p = alpha_i`, `P = A_i`); for `(mn|P)` it is the bra
record's `s_ab = c_i c_j (pi/p)^{3/2}` times `K_AB`, which the MMD staging
keeps out of `s_ab` (it rides in `E`'s seed there): it is formed per record
from the record's own `XPA`, `XPB`, as `e_fragment`'s seed is, `K_AB =
exp(p X_PA . X_PB)` -- `X_PA = -(b/p) AB` and `X_PB = (a/p) AB`, so `p X_PA .
X_PB = -(a b/p) |AB|^2`. The fundamental is `hgp.recurrence.
fundamental_prelude`'s, `[00|00]^(m) = pair (2/sqrt(pi)) theta F_m(T)`,
unchanged.

THE THREAD MAPPING IS THE MMD KERNEL'S: one thread per staged aux shell pair
(the metric) or per cell, an AO shell pair crossed with one aux shell
(`(mn|P)`), its two record ranges walked serially, bra outer. Per (bra, aux)
record pair the VRR runs and each `m = 0` boundary value is added into its
own register accumulator, `c{e}__{f}` (`hgp.vrr.targets` order): the
contraction completes BEFORE the HRR reads it, because eq 28 holds only on
contracted values (milestone 12, pin 5). After the loops the HRR runs once
per cell, `AB` the shell pair's, read off the cell's first bra record
(`AB = X_PB - X_PA`: every record of one shell pair has the same `A - B`).

THE SCATTER is the MMD kernel's, a plain store and its mirror (`kernel_jab`'s
diagonal rule, the lower triangle when the two shells are one), no atomics:
each element belongs to one thread, so runs are bit-reproducible. No sign:
HGP produces `(P|Q)` and `(mn|P)` themselves, where MMD folds `(-1)^{|tau|}`
into one sign per class.

ROUND-OFF, NOT BITWISE, against the hosts (`eri.density_fit.metric`,
`eri.density_fit.eri3`, McMurchie & Davidson on `reference.pq`) and against
the MMD kernels: another recurrence associates differently.

SCOPE. The builders accept any class the DAGs accept, but correctness is
tested for s and p shells only (`HGP_DF_L_MAX = 1`: the metric's (s|s),
(s|p), (p|p) and `(mn|P)`'s six classes `la <= lb <= 1`, `lP <= 1`), and the
CLI's ladders refuse more. Nothing is measured.

    params  exactly kernel_metric's / kernel_eri3's (their module docstrings)

A builder: vendor-neutral IR. It imports neither `ao/` (its witnesses), nor
the interpreter, nor `reference/`, nor `plan/` (the HGP engine has no
Hermite plan), nor a backend.
"""

from __future__ import annotations

from dataclasses import replace

from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import PAIR as PAIR_FACTOR
from felizjueves.hgp.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    hrr_body,
    hrr_name,
    vrr_body,
    vrr_name,
)
from felizjueves.hgp.vrr import AmClass, Cart
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
from felizjueves.spec import Engine, Operator, Spec, Target

__all__ = [
    "HGP_DF_L_MAX",
    "build_heri3_kernel",
    "build_hmetric_kernel",
    "eri3_class",
    "heri3_name",
    "heri3_spec",
    "hmetric_name",
    "hmetric_spec",
    "metric_class",
]

HGP_DF_L_MAX = 1  # the highest shell l the HGP DF kernels are tested for

# The MMD kernels' names, restated rather than imported (`kernel_metric`,
# `kernel_eri3` reach the plan layer, which an HGP builder must not):
# test/test_kernel_hgp_df.py holds the parameters equal to theirs.
PAIR, BRA, KET, OUT = "pair", "ip", "iq", "metric"  # the metric
CELL, BRA3, KET3, OUT3 = "cell", "ib", "ik", "eri3"  # the three-center
S_AUX = "s_aux"  # the aux record's `s`, as the MMD kernels spell it
_DIAGONAL = (
    "diagonal aux shell pair (off_p == off_q): the lower triangle a >= b "
    "only, each mirrored once -- kernel_jab's diagonal rule, so (P|Q) and "
    "(Q|P) are one value")
_DIAGONAL3 = (
    "diagonal AO shell pair (off_a == off_b): a >= b only, each mirrored "
    "once -- kernel_jab's diagonal rule, so (mn|P) and (nm|P) are one value")

# The quartet scalars that vanish on a single-centre (exponent-0 partner) side.
_QC = frozenset(Factor.qc(d) for d in range(3))
_PA = frozenset(Factor.pa(d) for d in range(3))


def hmetric_spec(lP: int, lQ: int) -> Spec:
    """The aux class `(lP, lQ)` on the HGP engine: `.../metric.hgp/...`."""
    return Spec(Operator.COULOMB, 0, (lP, lQ), Target.METRIC,
                engine=Engine.HGP)


def heri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` on the HGP engine: `.../eri3.hgp/...`."""
    return Spec(Operator.COULOMB, 0, (la, lb, lP), Target.ERI3,
                engine=Engine.HGP)


def _spec(cls, target: Target, make) -> Spec:
    spec = cls if isinstance(cls, Spec) else make(*cls)
    if spec.target is not target or spec.engine is not Engine.HGP:
        raise ValueError(f"{spec.key()} is not an HGP Target.{target.name} "
                         "key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the HGP DF kernels take no levers")
    return spec


def hmetric_name(cls: Spec | tuple[int, int]) -> str:
    """`hmetric_` and the two `l`s, `_`-separated once any is two digits."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    return "hmetric_" + ("_" if max(am) > 9 else "").join(map(str, am))


def heri3_name(cls: Spec | tuple[int, int, int]) -> str:
    """`heri3_` and the three `l`s, `_`-separated once any is two digits:
    `h` before `kernel_eri3.eri3_name`'s spelling (`heri3_011`)."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    return "heri3_" + ("_" if max(am) > 9 else "").join(map(str, am))


def metric_class(cls: Spec | tuple[int, int]) -> AmClass:
    """The HGP class the metric kernel builds: `(lP, 0, lQ, 0)`."""
    lP, lQ = _spec(cls, Target.METRIC, hmetric_spec).am
    return (lP, 0, lQ, 0)


def eri3_class(cls: Spec | tuple[int, int, int]) -> AmClass:
    """The HGP class the three-center kernel builds: the higher bra `l`
    first, `(max, min, lP, 0)` (module docstring, "THE BRA ORIENTATION")."""
    la, lb, lP = _spec(cls, Target.ERI3, heri3_spec).am
    return (max(la, lb), min(la, lb), lP, 0)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _digits(v: Cart) -> str:
    return "_".join(map(str, v))


def _acc(e: Cart, f: Cart) -> str:
    return f"c{_digits(e)}__{_digits(f)}"


def _expr_reads(e: object) -> set[str]:
    """Every name an expression reads: `Var`s, arrays, `Factor` scalars."""
    if isinstance(e, Var):
        return {e.name}
    if isinstance(e, Index):
        return {e.array}.union(*(_expr_reads(i) for i in e.at))
    if isinstance(e, Call):
        return set().union(*(_expr_reads(a) for a in e.args))
    if isinstance(e, Weighted):
        out: set[str] = set()
        for t in e.terms:
            out |= _expr_reads(t.src)
            if t.g.name_in_kernel is not None:
                out.add(t.g.name_in_kernel)
        return out
    return set()  # Lit


def _reads(body: tuple[Stmt, ...]) -> set[str]:
    """Every name the `Decl`, `Assign` and `Boys` statements of `body` read."""
    out: set[str] = set()
    for st in body:
        if isinstance(st, Decl):
            out |= _expr_reads(st.init)
        elif isinstance(st, Assign):
            out |= _expr_reads(st.value)
        elif isinstance(st, Boys):
            out |= _expr_reads(st.t)
    return out


def _live(body: tuple[Stmt, ...], wanted: set[str]) -> tuple[Stmt, ...]:
    """`body` (`Decl`s, `Note`s and the prelude's `Boys` fill) without the
    `Decl`s that neither `wanted` nor a kept later statement reads, and
    without a `Boys` whose array nothing kept reads: one backward pass
    (`kernel_metric._live`'s rule, restated)."""
    kept: list[Stmt] = []
    live = set(wanted)
    for st in reversed(body):
        if isinstance(st, Decl):
            if st.name not in live:
                continue
            live |= _expr_reads(st.init)
        elif isinstance(st, Boys):
            if st.dst not in live:
                continue
            live |= _expr_reads(st.t)
        elif not isinstance(st, Note):
            raise TypeError(f"_live takes Decls, Notes and Boys, got "
                            f"{type(st).__name__}")
        kept.append(st)
    return tuple(reversed(kept))


def _without(body: tuple[Stmt, ...], zero: frozenset[Factor]
             ) -> tuple[Stmt, ...]:
    """`body` with every `Weighted` term whose factor is in `zero` removed."""
    out: list[Stmt] = []
    for st in body:
        if isinstance(st, Decl) and isinstance(st.init, Weighted):
            terms = tuple(t for t in st.init.terms if t.g not in zero)
            if not terms:
                raise ValueError(f"{st.name}: every term is a zero factor")
            st = replace(st, init=Weighted(terms))
        out.append(st)
    return tuple(out)


def _quartet_scalars(p: Expr, q: Expr, ket_pair: Expr) -> tuple[Stmt, ...]:
    """Every scalar `hgp.recurrence.VRR_SCALARS` names but `PA`, `QC` and
    `PQ` (bound by the caller) for one primitive quartet of bra exponent `p`
    and aux exponent `q`, with `pair = up * ket_pair` (module docstring)."""
    stmts: list[Stmt] = [
        Decl("ipq", Scalar.F64, init=_c(Fn.DIV, Lit(1.0), _c(Fn.ADD, p, q))),
        Decl("rho", Scalar.F64, init=_c(Fn.MUL, _c(Fn.MUL, p, q), Var("ipq"))),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, Var("rho"))),
        Decl("wp_s", Scalar.F64, init=_c(Fn.SUB, Lit(0.0),
                                         _c(Fn.MUL, q, Var("ipq")))),
        Decl("wq_s", Scalar.F64, init=_c(Fn.MUL, p, Var("ipq"))),
    ]
    for d, n in enumerate(PQ_PARAMS):
        stmts.append(Decl(Factor.wp(d).value, Scalar.F64,
                          init=_c(Fn.MUL, Var("wp_s"), Var(n))))
    for d, n in enumerate(PQ_PARAMS):
        stmts.append(Decl(Factor.wq(d).value, Scalar.F64,
                          init=_c(Fn.MUL, Var("wq_s"), Var(n))))
    inv2p, inv2q = Factor.INV2P.value, Factor.INV2Q.value
    stmts += [
        Decl(inv2p, Scalar.F64, init=_c(Fn.DIV, Lit(0.5), p)),
        Decl(inv2q, Scalar.F64, init=_c(Fn.DIV, Lit(0.5), q)),
        Decl(Factor.INV2PQ.value, Scalar.F64,
             init=_c(Fn.MUL, Lit(0.5), Var("ipq"))),
        Decl(Factor.INV2P_ALPHA_OVER_P.value, Scalar.F64, init=_c(
            Fn.MUL, Var(inv2p), _c(Fn.DIV, Var("rho"), p))),
        Decl(Factor.INV2Q_ALPHA_OVER_Q.value, Scalar.F64, init=_c(
            Fn.MUL, Var(inv2q), _c(Fn.DIV, Var("rho"), q))),
        Decl(PAIR_FACTOR, Scalar.F64, init=_c(Fn.MUL, Var("up"), ket_pair)),
    ]
    return tuple(stmts)


def _ket_body(cls: AmClass, zero: frozenset[Factor], centre: str,
              ket: Var) -> tuple[tuple[Stmt, ...], list[tuple[Cart, Cart]]]:
    """One aux record's statements inside the bra loop: the quartet scalars,
    the prelude and the VRR of `cls` with its `zero` terms dropped and
    everything nothing kept reads gone with them, then one `c += V` per
    boundary value; and the boundary `(e, f)` in `hgp.vrr.targets` order.
    `centre` names the bra centre's three locals (`A`, `P`)."""
    dag = vrr.build(cls)
    prelude, fundamental = fundamental_prelude(sum(cls))
    bnd = [(e, f) for e, f, _ in vrr.targets(cls)]
    body: tuple[Stmt, ...] = (
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"{centre}{d}"),
                                      Index("A", (ket, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        *_quartet_scalars(Var("pe"), Var("aq"), Index(S_AUX, (ket,))),
        *prelude,
        *_without(vrr_body(dag, fundamental), zero),
    )
    accumulate = tuple(
        Assign(Var(_acc(e, f)), _c(Fn.ADD, Var(_acc(e, f)),
                                   Var(vrr_name((e, f, 0)))))
        for e, f in bnd)
    kept = _live(body, {vrr_name((e, f, 0)) for e, f in bnd})
    return (Decl("aq", Scalar.F64, init=Index("alpha", (ket,))),
            Note(f"[e0|f0]^(0) by the HGP VRR of {cls} ([Fock] eqs 25-26); "
                 "the single-centre side's terms in "
                 f"{', '.join(sorted(g.value for g in zero))} (exactly zero) "
                 "dropped with everything only they read"),
            *kept, *accumulate), bnd


def _thread(index: str, count: str, per: tuple[Stmt, ...], what: str
            ) -> tuple[Stmt, ...]:
    return (
        Note(f"launch: one thread per {what}"),
        Decl(index, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, Var(index), Var(count)), per),
    )


def _differ(a: str, b: str) -> Expr:
    return _c(Fn.ADD, _c(Fn.LT, Var(a), Var(b)), _c(Fn.LT, Var(b), Var(a)))


def _plus(base: str, k: int) -> Expr:
    return _c(Fn.ADD, Var(base), Lit(k))


def _f64(n: str, shape: tuple[Expr, ...]) -> Param:
    return Param(n, Scalar.F64, array=True, shape=shape)


def _i32(n: str, shape: tuple[Expr, ...]) -> Param:
    return Param(n, Scalar.I32, array=True, shape=shape)


def _metric_params() -> tuple[Param, ...]:
    """`kernel_metric`'s parameters, in its order."""
    npair, nrec, naux = Var("n_pair"), Var("n_rec"), Var("n_aux")
    return (
        Param("n_pair", Scalar.I32), Param("n_rec", Scalar.I32),
        Param("n_aux", Scalar.I32),
        _f64("A", (nrec, Lit(3))), _f64("alpha", (nrec,)),
        _f64(S_AUX, (nrec,)),
        *(_i32(n, (npair,)) for n in ("p_rec0", "p_rec1", "q_rec0",
                                      "q_rec1", "off_p", "off_q")),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(naux, naux)),
    )


def _eri3_params() -> tuple[Param, ...]:
    """`kernel_eri3`'s parameters, in its order."""
    ncell, nbra, naux_rec = Var("n_cell"), Var("n_bra_rec"), Var("n_aux_rec")
    return (
        *(Param(n, Scalar.I32) for n in ("n_cell", "n_bra_rec", "n_aux_rec",
                                         "n_ao", "n_aux")),
        _f64("zeta_ab", (nbra,)), _f64("P", (nbra, Lit(3))),
        _f64("XPA", (nbra, Lit(3))), _f64("XPB", (nbra, Lit(3))),
        _f64("s_ab", (nbra,)),
        _f64("A", (naux_rec, Lit(3))), _f64("alpha", (naux_rec,)),
        _f64(S_AUX, (naux_rec,)),
        *(_i32(n, (ncell,)) for n in ("ab_rec0", "ab_rec1", "p_rec0",
                                      "p_rec1", "off_a", "off_b", "off_p")),
        Param(OUT3, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Var("n_ao"), Var("n_ao"), Var("n_aux"))),
    )


def build_hmetric_kernel(cls: Spec | tuple[int, int], *,
                         name: str | None = None) -> Kernel:
    """The HGP metric kernel of one aux class (module docstring); its
    parameters, launch and output are `kernel_metric.build_metric_kernel`'s.
    """
    spec = _spec(cls, Target.METRIC, hmetric_spec)
    lP, lQ = spec.am
    rowsP, rowsQ = vrr.cartesians(lP), vrr.cartesians(lQ)
    ip = Var(BRA)
    ket_body, bnd = _ket_body(metric_class(spec), _PA | _QC, "A",
                              Var(KET))
    assert bnd == [(e, f) for e in rowsP for f in rowsQ]
    bra_body: tuple[Stmt, ...] = (
        Decl("pe", Scalar.F64, init=Index("alpha", (ip,))),
        *(Decl(f"A{d}", Scalar.F64, init=Index("A", (ip, Lit(d))))
          for d in range(3)),
        Note("U_P = s_aux of the bra record: c (pi/alpha)^(3/2), K = 1"),
        Decl("up", Scalar.F64, init=Index(S_AUX, (ip,))),
        Note("the Q records of the pair"),
        For(KET, Var("q0"), Var("q1"), Lit(1), ket_body),
    )

    def at(row: Expr, col: Expr) -> Index:
        return Index(OUT, (row, col))

    def val(a: int, b: int) -> Var:
        return Var(_acc(rowsP[a], rowsQ[b]))

    off_diag: list[Stmt] = []
    for a in range(len(rowsP)):
        for b in range(len(rowsQ)):
            off_diag += [Assign(at(_plus("op", a), _plus("oq", b)), val(a, b)),
                         Assign(at(_plus("oq", b), _plus("op", a)), val(a, b))]
    scatter: Stmt
    if lP == lQ:
        diag: list[Stmt] = [Note(_DIAGONAL)]
        for a in range(len(rowsP)):
            for b in range(a + 1):
                diag.append(Assign(at(_plus("op", a), _plus("op", b)),
                                   val(a, b)))
                if a != b:
                    diag.append(Assign(at(_plus("op", b), _plus("op", a)),
                                       val(a, b)))
        scatter = If(_differ("op", "oq"), tuple(off_diag), tuple(diag))
    else:
        scatter = If(_differ("op", "oq"), tuple(off_diag))

    per_pair: tuple[Stmt, ...] = (
        *(Decl(n, Scalar.I32, init=Index(arr, (Var(PAIR),)))
          for n, arr in (("p0", "p_rec0"), ("p1", "p_rec1"), ("q0", "q_rec0"),
                         ("q1", "q_rec1"), ("op", "off_p"), ("oq", "off_q"))),
        *(Decl(_acc(e, f), Scalar.F64, init=Lit(0.0)) for e, f in bnd),
        Note("the P records of the pair, serially; the boundary sums are "
             "the contracted (P|Q) themselves (b = d = 0: no HRR)"),
        For(BRA, Var("p0"), Var("p1"), Lit(1), bra_body),
        Note("scatter: one plain store per element and its mirror, no "
             "atomics -- each element belongs to one aux shell pair"),
        scatter,
    )
    body = (
        Note(f"DF Coulomb metric on the HGP engine, class (lP, lQ) = ({lP}, "
             f"{lQ}): (P|Q) = ({lP}0|{lQ}0) with exponent-0 partners "
             "(eri/density_fit/kernel_hgp.py)"),
        *_thread(PAIR, "n_pair", per_pair, "aux shell pair"),
    )
    return Kernel(name or hmetric_name(spec), _metric_params(), body)


def build_heri3_kernel(cls: Spec | tuple[int, int, int], *,
                       name: str | None = None) -> Kernel:
    """The HGP three-center kernel of one class (module docstring); its
    parameters, launch and output are `kernel_eri3.build_eri3_kernel`'s."""
    spec = _spec(cls, Target.ERI3, heri3_spec)
    la, lb, lP = spec.am
    hcls = eri3_class(spec)
    swap = la < lb  # built (b a|P): the record's B is the recurrence's A
    xa, xb = ("XPB", "XPA") if swap else ("XPA", "XPB")
    ib = Var(BRA3)
    ket_body, bnd = _ket_body(hcls, _QC, "P", Var(KET3))

    kdot: Expr | None = None
    for d in range(3):
        t = _c(Fn.MUL, Index("XPA", (ib, Lit(d))), Index("XPB", (ib, Lit(d))))
        kdot = t if kdot is None else _c(Fn.ADD, kdot, t)
    assert kdot is not None
    bra_scalars = _live((
        Decl("pe", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        *(Decl(Factor.pa(d).value, Scalar.F64, init=Index(xa, (ib, Lit(d))))
          for d in range(3)),
        Note("U_P = s_ab K_AB, K_AB = exp(p X_PA . X_PB): the MMD staging "
             "keeps K_AB out of s_ab (it rides in E's seed there)"),
        Decl("up", Scalar.F64, init=_c(Fn.MUL, Index("s_ab", (ib,)), _c(
            Fn.EXP, _c(Fn.MUL, Var("pe"), kdot)))),
    ), _reads(ket_body))
    bra_body: tuple[Stmt, ...] = (
        *bra_scalars,
        Note("the aux records of the cell"),
        For(KET3, Var("p0"), Var("p1"), Lit(1), ket_body),
    )

    # --- the HRR, once per cell, on the contracted boundary -----------------
    hdag = hrr.build(hcls, "bra")
    if set(bnd) != hdag.boundary:
        raise ValueError(f"{hcls}: the HRR boundary is not the VRR target set")
    nA, nB = len(vrr.cartesians(la)), len(vrr.cartesians(lb))
    rowsP = vrr.cartesians(lP)
    node: dict[tuple[int, int, int], str] = {}  # staged (a, b, q) -> its ERI
    for a, ca in enumerate(vrr.cartesians(la)):
        for b, cb in enumerate(vrr.cartesians(lb)):
            for q, cq in enumerate(rowsP):
                ab = (cb, ca) if swap else (ca, cb)
                node[(a, b, q)] = hrr_name((*ab, cq, (0, 0, 0)))
    hbody = _live((
        *(Decl(Factor.ab(d).value, Scalar.F64, init=_c(
            Fn.SUB, Index(xb, (Var("b0"), Lit(d))),
            Index(xa, (Var("b0"), Lit(d)))))
          for d in range(3)),
        *hrr_body(hdag, lambda e, f: Var(_acc(e, f))),
    ), set(node.values()))

    def at(m: Expr, n: Expr, q: int) -> Index:
        return Index(OUT3, (m, n, _plus("op", q)))

    off_diag: list[Stmt] = []
    for a in range(nA):
        for b in range(nB):
            for q in range(len(rowsP)):
                v = Var(node[(a, b, q)])
                off_diag += [Assign(at(_plus("oa", a), _plus("ob", b), q), v),
                             Assign(at(_plus("ob", b), _plus("oa", a), q), v)]
    scatter: Stmt
    if la == lb:
        diag: list[Stmt] = [Note(_DIAGONAL3)]
        for a in range(nA):
            for b in range(a + 1):
                for q in range(len(rowsP)):
                    v = Var(node[(a, b, q)])
                    diag.append(Assign(
                        at(_plus("oa", a), _plus("oa", b), q), v))
                    if a != b:
                        diag.append(Assign(
                            at(_plus("oa", b), _plus("oa", a), q), v))
        scatter = If(_differ("oa", "ob"), tuple(off_diag), tuple(diag))
    else:
        scatter = If(_differ("oa", "ob"), tuple(off_diag))

    per_cell: tuple[Stmt, ...] = (
        *(Decl(n, Scalar.I32, init=Index(arr, (Var(CELL),)))
          for n, arr in (("b0", "ab_rec0"), ("b1", "ab_rec1"),
                         ("p0", "p_rec0"), ("p1", "p_rec1"), ("oa", "off_a"),
                         ("ob", "off_b"), ("op", "off_p"))),
        *(Decl(_acc(e, f), Scalar.F64, init=Lit(0.0)) for e, f in bnd),
        Note("the bra records of the cell's shell pair, serially; the "
             "boundary sums complete before the HRR reads them"),
        For(BRA3, Var("b0"), Var("b1"), Lit(1), bra_body),
        Note(f"the bra HRR of {hcls} ([Fock] eq 28) on contracted values, "
             "AB = X_PB - X_PA of the cell's first record"
             + (" (built swapped: B - A)" if swap else "")
             if hdag.n_interior else
             f"no HRR transfer for {hcls} (the built bra's second shell is "
             "s): the contracted boundary values are the integrals"),
        *hbody,
        Note("scatter: one plain store per element and its (m, n) mirror, "
             "no atomics -- each element belongs to one cell"),
        scatter,
    )
    body = (
        Note(f"three-center DF integral on the HGP engine, class (la, lb, lP) "
             f"= ({la}, {lb}, {lP}): (mn|P) = {hcls} with an exponent-0 "
             "partner (eri/density_fit/kernel_hgp.py)"),
        *_thread(CELL, "n_cell", per_cell,
                 "(AO shell pair, aux shell) cell"),
    )
    return Kernel(name or heri3_name(spec), _eri3_params(), body)
