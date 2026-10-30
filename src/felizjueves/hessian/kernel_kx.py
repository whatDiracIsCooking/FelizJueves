"""The exchange derivative matrix kernels (#689): `Target.KX1` and
`Target.KXC` at `derivative_order = 1`, `kernel_gk`'s rolled form
scattering into an AO matrix instead of reducing into `G[atom]`.

    build_kx_kernel(spec | (la, lc, lb, ld), form=) -> Kernel  one per K class
    kx_name(spec | am, form=)    -> str     kx1_{la}{lc}_{lb}{ld} / kxc_...
    kx_spec(la, lc, lb, ld, form=) -> Spec  the class's d1 key
    kx_pq_spec(la, lc, lb, ld)   -> Spec    the d0 K class whose plan it runs
    launch_shape(n_bra, lanes)   -> (grid, (1, lanes))

THE MATRIX (`hessian.host_kx`'s module docstring has the whole account).
For atom `X`, axis `k`, `K^X_ab = SUM_cd D_cd d(ac|bd)/dR_{X,k}` (K's
pairing: bra record `(a, c)`, ket record `(b, d)`), assembled from two
forms, each the derivative on ONE bra function:

    kx1 (ip1):        K1[k, a, b]   += s_ac SUM_cd D_cd SUM_p (dA_k E^{ac})_p I_bd[p]
    kxc (contracted): KC_X[k, a, b] += s_ac SUM_cd D_cd SUM_p (dC_k E^{ac})_p I_bd[p]
    I_bd[p] = SUM_q (-1)^|q| [p+q]^(0) s_bd E^{bd}_q,   |p| <= la + lc + 1

`kx1` is every atom's at once (one launch per class, every bra record);
`kxc` is launched ONCE PER ATOM `X` with only the bra records whose `c`
sits on `X` (`host_kx.kx_records(..., atom=X)`), so the atoms together
cost one pass. The caller assembles `K^X = P_X K1 + (P_X K1)^T + KC_X +
KC_X^T` (`P_X` the rows on `X`; the transposes are the ket's two centres,
for a SYMMETRIC `D`). The RHF exchange part of `F^x` is `-1/2 K^X`: the
factor is the caller's.

THE PLAN IS `gk`'s: `kx_pq_spec = gradient.kernel_gk.gk_pq_spec`, the d0
K class with the bra raised by one (split `(lp + 1, lq)`), whose one DAG
forms every `[r]^(0)` with `|r| <= L + 1` either form reads (asserted).

THE ROLLED FORM, ONLY. #618's lessons apply from the start: no `E` block,
no straight-line fold. Per bra record the bra's per-axis tables one index
up on the MOVED side only (`(la + 1, lc)` for `kx1`, `(la, lc + 1)` for
`kxc`), COMPACT (`kernel_gk.tables_fragment`); per ket record `[r]^(0)`
into a local `R` as it forms (`recurrence.store_targets`), the ket's
compact tables with the sign `(-1)^|q|` folded into their odd-`t`
entries, then

    for every b row:  Y[3 n_a] = 0
        for every d row:  I[p] = s_bd SUM_q E^bd_q R[p + q]    |p| <= lp + 1
            for every (a, c) row:  w = M[c, d]
                Y[k, a] += w (2 e SUM E^{a'(+1_k)} I - n_k SUM E^{a'(-1_k)} I)
        X[k, a, b] += s_ac Y[k, a]        AtomicAdd, every a, every k

(`e` the moved function's exponent, `alpha` or `gamma`; `n_k` its
component on axis `k`), each `SUM` a loop nest over the tables
(`kernel_gk`'s `_table_sum`). `Y` is the one per-row local: `3 n_a`
doubles. No warp reduction: each lane holds its own ket record, so its
`(a, b)` block is its own, and lanes whose kets share a shell pair meet in
the atomics.

THE THREAD MAPPING IS `kernel_k`'s: one warp per bra record, `bra =
THREAD_X + BLOCK_X * BLOCK_DIM_X` with blockDim `(1, lanes)`, the kets
strip-mined over `THREAD_Y`. NO SCREEN (`gk`'s reason). No shared memory.

    params  n_bra n_ket n_ao (I32)
            P[n_bra, 3]  zeta_ac  XPA[n_bra, 3]  XPC[n_bra, 3]  s_ac
                ea_bra (kx1) | ec_bra (kxc)  [n_bra]             bra side
            off_a  off_c  [n_bra]  (I32)
            Q[n_ket, 3]  eta  XQB[n_ket, 3]  XQD[n_ket, 3]  s_bd  [n_ket]
            off_b  off_d  [n_ket]  (I32)                         ket side
            M[n_ao, n_ao]          the density D, read at AO offsets
            X[3, n_ao, n_ao]       (OUT, read and AtomicAdd-ed)

`X` is ZEROED BY THE CALLER; every launch only adds. Cartesian,
row-major; spherical is the caller's (`U X U^T`).

ROUND-OFF, NOT BITWISE, against `hessian.host_kx`, which differentiates
`E^{ac}` before the fold and runs `reference.pq`'s `[r]` (the kernel: the
plan's MILD recurrence); the stated bound is in test/test_kernel_kx.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor the host nor
the interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves.eri.k_exchange.levers.seams import KET, bindings
from felizjueves.gradient.kernel_gk import (
    I_ARRAY,
    KET_TABLES,
    R_ARRAY,
    _ket_signs,
    _table_sum,
    compact_extent,
    gk_pq_spec,
    tables_fragment,
)
from felizjueves.gradient.kernel_gk import launch_shape as _gk_launch_shape
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
from felizjueves.mmd.kernel import builder_plan
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    recurrence_body,
    store_targets,
)
from felizjueves.plan.schedule import schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "FORMS",
    "build_kx_kernel",
    "kx_name",
    "kx_pq_spec",
    "kx_spec",
    "launch_shape",
    "params",
]

# The two forms, by target: the moved bra function, its exponent's record
# array, and the kernel-name prefix.
FORMS: dict[Target, tuple[str, str, str]] = {
    Target.KX1: ("A", "ea_bra", "kx1"),
    Target.KXC: ("C", "ec_bra", "kxc"),
}

Y_ARRAY = "Y"  # one b row's 3 n_a sums, [k * n_a + a]

ZEROED_X = ("PRECONDITION: X is zeroed by the caller; every launch only "
            "adds")

NO_SCREEN = (
    "no screen: K's bound I_ac I_bd max|D| bounds the integral, not its "
    "derivative, so every ordered quartet of the launch is visited")

Am = tuple[int, int, int, int]


def kx_spec(la: int, lc: int, lb: int, ld: int, *,
            form: Target = Target.KX1) -> Spec:
    """The K class `(la, lc | lb, ld)`'s derivative matrix key, `form`
    `Target.KX1` (ip1) or `Target.KXC` (the contracted index)."""
    return Spec(Operator.COULOMB, 1, (la, lc, lb, ld), form)


def kx_pq_spec(la: int, lc: int, lb: int, ld: int) -> Spec:
    """The d0 K class whose plan either form runs: `gk`'s, the bra raised
    by one, split `(la + lc + 1, lb + ld)`."""
    return gk_pq_spec(la, lc, lb, ld)


def _class(cls: Spec | Am, form: Target | None) -> tuple[Am, Target]:
    if isinstance(cls, Spec):
        if form is not None and form is not cls.target:
            raise ValueError(f"{cls.key()}: form= {form} is not its target")
        spec = cls
    else:
        spec = kx_spec(*cls, form=Target.KX1 if form is None else form)
    if spec.target not in FORMS or spec.derivative_order != 1:
        raise ValueError(f"{spec.key()} is not a d1 Target.KX1 / KXC key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the derivative matrix kernels take "
                         "no levers")
    if spec.n_batches != 1:
        raise ValueError(f"{spec.key()}: the derivative matrix is built at "
                         "one batch only")
    la, lc, lb, ld = spec.am
    return (la, lc, lb, ld), spec.target


def kx_name(cls: Spec | Am, *, form: Target | None = None) -> str:
    """The name `build_kx_kernel(cls, form=form)` gives its kernel."""
    (la, lc, lb, ld), t = _class(cls, form)
    return f"{FORMS[t][2]}_{la}{lc}_{lb}{ld}"


def launch_shape(n_bra: int, lanes: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (1, lanes))`: one warp per bra record, `kernel_k`'s."""
    return _gk_launch_shape(n_bra, lanes)


def params(form: Target) -> tuple[str, ...]:
    """The parameter names of `form`'s kernel, in order."""
    return ("n_bra", "n_ket", "n_ao",
            "P", "zeta_ac", "XPA", "XPC", "s_ac", FORMS[form][1],
            "off_a", "off_c",
            "Q", "eta", "XQB", "XQD", "s_bd", "off_b", "off_d",
            "M", "X")


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _add(x: Expr, y: Expr) -> Call:
    return _c(Fn.ADD, x, y)


def build_kx_kernel(cls: Spec | Am, *, form: Target | None = None,
                    name: str | None = None) -> Kernel:
    """The derivative matrix kernel of one K class (module docstring).
    `cls` is a d1 `Target.KX1` / `KXC` `Spec`, or `(la, lc, lb, ld)` with
    `form=` naming the target (`Target.KX1` when omitted)."""
    (la, lc, lb, ld), t = _class(cls, form)
    side, expo, _ = FORMS[t]
    lp, lq = la + lc, lb + ld
    L1 = lp + lq + 1
    na = len(ef.cartesians(la))
    (dag,) = builder_plan(kx_pq_spec(la, lc, lb, ld)).dags
    order = tuple(schedule(dag))
    targets = frozenset(hermite_targets(L1))
    held = {r for (r, m) in order if m == 0}
    assert targets <= held, "the recurrence never forms every [r]^(0)"
    bra, ket = Var("bra"), Var(KET)
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    rec = store_targets((*prelude, *recurrence_body(dag, list(order),
                                                    fundamental)),
                        targets, L1, R_ARRAY)

    # the bra tables, one index up on the moved side only
    t1, t2 = (la + 1, lc) if side == "A" else (la, lc + 1)
    frag_ac = tables_fragment(t1, t2, ef.AXIS_ARRAYS)
    binds_ac = {k: v for k, v in bindings(frag_ac, Var("ze"), "XPA", "XPC",
                                          bra).items() if k in frag_ac.binds}
    for arr in ef.AXIS_ARRAYS:
        binds_ac[arr] = Var(arr)
    frag_bd = tables_fragment(lb, ld, KET_TABLES)
    binds_bd = {k: v for k, v in bindings(frag_bd, Var("et"), "XQB", "XQD",
                                          ket).items() if k in frag_bd.binds}
    for arr in KET_TABLES:
        binds_bd[arr] = Var(arr)

    def i_read(tt: tuple[Expr, Expr, Expr]) -> Expr:
        return Index(I_ARRAY, (ef.hermite_index(tt),))

    def moved(v: tuple[Expr, Expr, Expr], k: int, by: int):
        step = _add(v[k], Lit(1)) if by > 0 else _c(Fn.SUB, v[k], Lit(1))
        return tuple(step if d == k else v[d] for d in range(3))

    def stage1(b, d):
        def per_p(p):
            def r_at(q):
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _add(p[x], q[x]) for x in range(3))),))
            return (*_table_sum(KET_TABLES, ld, b, d, r_at, "s1", "k"),
                    Assign(Index(I_ARRAY, (ef.hermite_index(p),)),
                           _mul(Var("sq"), Var("s1"))))
        return ef.hermite_loops(lp + 1, per_p, tag="h")

    def row_ac(b, d):
        def row_a(a):
            def elem(c):
                out: list[Stmt] = [
                    Decl("rc", Scalar.I32, init=ef.cartesian_index("c")),
                    Decl("wt", Scalar.F64, init=Index("M", (
                        _add(Var("oc"), Var("rc")),
                        _add(Var("od"), Var("rd"))))),
                ]
                x = a if side == "A" else c
                for k in range(3):
                    up, lo = moved(x, k, 1), moved(x, k, -1)
                    ua, uc = (up, c) if side == "A" else (a, up)
                    wa, wc = (lo, c) if side == "A" else (a, lo)
                    hi, dn = f"h{k}", f"l{k}"
                    slot = Index(Y_ARRAY, (_add(Lit(k * na), Var("ra")),))
                    out += [
                        *_table_sum(ef.AXIS_ARRAYS, t2, ua, uc, i_read, hi,
                                    f"p{k}"),
                        Decl(dn, Scalar.F64, init=Lit(0.0)),
                        If(_c(Fn.LT, Lit(0), x[k]), (
                            *_table_sum(ef.AXIS_ARRAYS, t2, wa, wc, i_read,
                                        f"m{k}", f"q{k}"),
                            Assign(Var(dn), _mul(x[k], Var(f"m{k}"))),
                        )),
                        Assign(slot, _add(slot, _mul(Var("wt"), _c(
                            Fn.SUB, _mul(Var("e2"), Var(hi)), Var(dn))))),
                    ]
                return tuple(out)
            return (Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                    ef.cartesian_loops(lc, ("cx", "cy", "cz"), elem, tag="c"))
        return ef.cartesian_loops(la, ("ax", "ay", "az"), row_a, tag="a")

    def row_b(b):
        def row_d(d):
            return (
                Decl("rd", Scalar.I32, init=ef.cartesian_index("d")),
                Note(f"stage 1: I[p] = s_bd SUM_q (-1)^|q| E^bd_q [p+q], "
                     f"|p| <= {lp + 1}, the sign in the ket tables"),
                stage1(b, d),
                Note(f"per (a,c) row, w = D_cd from M: Y[k, a] += w (2 e "
                     f"SUM E^(+1_k) I - n_k SUM E^(-1_k) I), the derivative "
                     f"on {'a (ip1)' if side == 'A' else 'c (contracted)'}, "
                     "each a rolled sum over the bra tables"),
                row_ac(b, d),
            )
        scatter = For("sa", Lit(0), Lit(na), Lit(1), tuple(
            AtomicAdd(Index("X", (Lit(k), _add(Var("oa"), Var("sa")),
                                  _add(Var("ob"), Var("rb")))),
                      _mul(Var("sb"), Index(Y_ARRAY, (
                          _add(Lit(k * na), Var("sa")),))))
            for k in range(3)))
        return (
            Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),
            For("zy", Lit(0), Lit(3 * na), Lit(1), (
                Assign(Index(Y_ARRAY, (Var("zy"),)), Lit(0.0)),)),
            ef.cartesian_loops(ld, ("dx", "dy", "dz"), row_d, tag="d"),
            Note("scatter this b row: X[k, a, b] += s_ac Y[k, a], every a"),
            scatter,
        )

    pair: list[Stmt] = [
        *(Decl(n, Scalar.F64,
               init=_c(Fn.SUB, Var(f"P{d}"), Index("Q", (ket, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Decl("et", Scalar.F64, init=Index("eta", (ket,))),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _mul(Var("ze"), Var("et")),
            _add(Var("ze"), Var("et"))))),
        Note(f"[r]^(0) by the plan's DAG for the d0 K class ({la + 1},{lc}|"
             f"{lb},{ld}) at L + 1 = {L1}, each stored into R as it forms"),
        *rec,
        Note("the undressed per-axis E^bd of this ket record: "
             "e_fragment.axis_tables_fragment(lb, ld), inlined into Fx Fy Fz"),
        Inline(frag_bd, binds_bd),
        Note("the ket sign (-1)^|q|: every odd-t entry negated, per axis"),
        _ket_signs(lb, ld),
        Decl("sq", Scalar.F64, init=Index("s_bd", (ket,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (ket,))),
        Decl("od", Scalar.I32, init=Index("off_d", (ket,))),
        ef.cartesian_loops(lb, ("bx", "by", "bz"), row_b, tag="b"),
    ]
    head: list[Stmt] = [
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (bra, Lit(d))))
          for d in range(3)),
        Decl("ze", Scalar.F64, init=Index("zeta_ac", (bra,))),
        Decl("sb", Scalar.F64, init=Index("s_ac", (bra,))),
        Decl("e2", Scalar.F64, init=_mul(Lit(2.0), Index(expo, (bra,)))),
        Decl("oa", Scalar.I32, init=Index("off_a", (bra,))),
        Decl("oc", Scalar.I32, init=Index("off_c", (bra,))),
        Note(f"the undressed per-axis E^{{ac}}_t, i <= {t1}, j <= {t2}: one "
             f"index above the class on the moved side ({side}), COMPACT"),
        *(Decl(arr, Scalar.F64, (Lit(compact_extent(t1, t2)),))
          for arr in ef.AXIS_ARRAYS),
        Inline(frag_ac, binds_ac),
        Decl(R_ARRAY, Scalar.F64, (Lit(len(targets)),)),
        Decl(I_ARRAY, Scalar.F64, (Lit(len(hermite_targets(lp + 1))),)),
        *(Decl(arr, Scalar.F64, (Lit(compact_extent(lb, ld)),))
          for arr in KET_TABLES),
        Decl(Y_ARRAY, Scalar.F64, (Lit(3 * na),)),
    ]
    per_bra: list[Stmt] = [
        *head,
        Note("kets strip-mined over threadIdx.y, stride blockDim.y"),
        For(KET, Call(Fn.THREAD_Y), Var("n_ket"), Call(Fn.BLOCK_DIM_Y),
            tuple(pair)),
    ]
    what = ("K1[k, a, b] += s_ac SUM D_cd SUM_p (dA_k E^ac)_p I_bd[p], the "
            "ip1 form, every atom" if side == "A" else
            "KC_X[k, a, b] += s_ac SUM D_cd SUM_p (dC_k E^ac)_p I_bd[p], the "
            "contracted form, launched per atom X with the bra records whose "
            "c sits on X")
    body: tuple[Stmt, ...] = (
        Note(f"exchange derivative matrix, K class (la,lc|lb,ld) = "
             f"({la},{lc}|{lb},{ld}), ROLLED: {what}; [r] at L = {L1} "
             "(felizjueves.hessian)"),
        Note(ZEROED_X),
        Note(NO_SCREEN),
        Note("launch: blockDim = (1, warp width), one warp per bra record"),
        Decl("bra", Scalar.I32, init=_add(Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, bra, Var("n_bra")), tuple(per_bra)),
    )
    return Kernel(name or kx_name((la, lc, lb, ld), form=t), _params(t), body)


def _params(form: Target) -> tuple[Param, ...]:
    nbr, nk, nao = Var("n_bra"), Var("n_ket"), Var("n_ao")

    def f64(n: str, *shape: Expr) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, ext: Expr) -> Param:
        return Param(n, Scalar.I32, array=True, shape=(ext,))

    out = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("n_ao", Scalar.I32),
        f64("P", nbr, Lit(3)), f64("zeta_ac", nbr),
        f64("XPA", nbr, Lit(3)), f64("XPC", nbr, Lit(3)), f64("s_ac", nbr),
        f64(FORMS[form][1], nbr),
        i32("off_a", nbr), i32("off_c", nbr),
        f64("Q", nk, Lit(3)), f64("eta", nk), f64("XQB", nk, Lit(3)),
        f64("XQD", nk, Lit(3)), f64("s_bd", nk),
        i32("off_b", nk), i32("off_d", nk),
        f64("M", nao, nao),
        Param("X", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(3), nao, nao)),
    )
    assert tuple(p.name for p in out) == params(form)
    return out
