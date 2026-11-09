"""The DF metric Hessian on the HGP engine: `Target.METRIC` at d2, `engine=HGP`.

    build_hgd2metric_kernel(spec | (lP, lQ)) -> Kernel   hgd2metric_{lP}{lQ}
    hgd2metric_spec(lP, lQ)                  -> Spec     `.../d2/.../metric.hgp/...`
    hgd2metric_name(spec | (lP, lQ))         -> str
    hgd2metric_dag(spec | (lP, lQ))          -> VrrDag   the bra two up, level, two down
    HGD2METRIC_L_MAX                         -> 1: every shell s or p

The HGP twin of `kernel_d2metric` (as `gradient.kernel_hgmetric` is
`kernel_gmetric`'s): per aux shell pair `(P, Q)`, staged as the d0 kernels
stage it, the pair's contribution to

    H[A, k, B, l] += SUM_PQ W_PQ d2(P|Q) / dR_{A,k} dR_{B,l}

with `kernel_d2metric`'s parameters, launch (one thread per aux shell
pair), weights and `H` contract, so a caller swaps one builder for the
other and changes nothing else (`test/df_grad_launch.run_d2metric` takes
`engine=`). `hessian.host_df.metric_hessian` is the witness.

THE CARTESIAN RULE, TWICE, NOT THE HERMITE SHIFT. With `u = 2 alpha` the
bra PRIMITIVE's exponent (`kernel_hgmetric`'s rule `D_k a = u [a + 1_k] -
a_k [a - 1_k]` applied to its own output):

    D_l D_k a = u^2 [a + 1_k + 1_l]
              - u (a + 1_k)_l [a + 1_k - 1_l] - u a_k [a - 1_k + 1_l]
              + a_k (a - 1_k)_l [a - 1_k - 1_l]

(for `k = l` the two middle terms are both `[a]`: `- u (2 a_k + 1) [a]`).
`D_k D_l = D_l D_k`, so `h_kl = SUM_ab w_ab D_k D_l (a|b)` is symmetric
and six contractions per pair suffice. `(P|Q)` reads `A - B` alone, so

    d2/dA_k dA_l = h_kl,   d2/dA_k dB_l = - h_kl,   d2/dB_k dB_l = + h_kl

exactly (`kernel_d2metric`'s second line, the same 36 `AtomicAdd`s: `+h`
on the two same-centre blocks, `-h` on the mixed ones). `SUM_A H[A]` is
then zero to round-off, checked against `host_df`, which differentiates
both centres.

THE VRR, once per primitive pair: the closure of the bra two up, level and
two down -- every `(e, f)` at `m = 0` with `|e|` in `{lP + 2, lP, lP - 2}`
(the last for `lP >= 2`) and `|f| = lQ` -- in the DAG of `(lo, lP + 2 - lo
| lQ, 0)` (`hgd2metric_dag`), `kernel_hgmetric`'s `vrr_ket_body` with the
`PA = QC = 0` terms unspelled. The rows `|e| = lP +- 1` are formed only as
intermediates. The `u` and `u^2` belong to the bra primitive and are applied
inside the record loops, never on contracted values.

PER RECORD PAIR, with `w_ab` the weights (`kernel_hgmetric`'s rule),

    h_kl += u^2 SUM_ab w_ab [a + 1_k + 1_l | b]
          - u   SUM_ab w_ab ((a + 1_k)_l [a + 1_k - 1_l | b]
                             + a_k [a - 1_k + 1_l | b])
          +     SUM_ab w_ab a_k (a - 1_k)_l [a - 1_k - 1_l | b]

`H` is ZEROED BY THE CALLER.

ROUND-OFF, NOT BITWISE, against `host_df` and `kernel_d2metric`: another
recurrence and another derivative route.

SCOPE. The builder accepts any class; correctness is tested for every class
`lP + lQ <= 4` interpreted, and the CLI ladder (`--hgp-hess-metric-l`) and
goldens are the s and p classes (`HGD2METRIC_L_MAX = 1`, as
`kernel_hgp.HGP_DF_L_MAX`). One form, nothing measured, no card.

    params  exactly kernel_d2metric's (its module docstring)

A builder: vendor-neutral IR. It imports neither `ao/` nor `hessian.host_df`
(its witness), nor `plan/`, nor `mmd/`, nor the interpreter, nor
`reference/`, nor a backend.
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_hgp import (
    S_AUX,
    ZERO_PA,
    ZERO_QC,
    vrr_ket_body,
)
from felizjueves.gradient.kernel_hgmetric import pair_weights
from felizjueves.hgp import vrr
from felizjueves.hgp.recurrence import vrr_name
from felizjueves.hgp.vrr import Cart
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
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.spec import Engine, Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "HGD2METRIC_L_MAX",
    "KL",
    "ZEROED_H",
    "build_hgd2metric_kernel",
    "hgd2metric_dag",
    "hgd2metric_name",
    "hgd2metric_spec",
]

HGD2METRIC_L_MAX = 1  # the highest shell l the CLI ladder and goldens take

PAIR = "pair"  # one thread per aux shell pair
BRA = "ip"  # the P record, serial
KET = "iq"  # the Q record, serial inside the bra

# `kernel_d2metric`'s notes, restated (it reaches `mmd/` and `plan/`):
# test/test_kernel_hgd2metric.py holds them equal.
ATOMIC_REASON = (
    "AtomicAdd: every aux shell pair on an atom adds into that atom's "
    "blocks of H")
ZEROED_H = (
    "PRECONDITION: H is zeroed by the caller before the first hessian "
    "launch of a build; this kernel only adds")

# The six components of the symmetric `h_kl`, `k <= l`.
KL: tuple[tuple[int, int], ...] = tuple(
    (k, l) for k in range(3) for l in range(k, 3))  # noqa: E741


def hgd2metric_spec(lP: int, lQ: int) -> Spec:
    """The aux class `(lP, lQ)` at d2 on the HGP engine."""
    return Spec(Operator.COULOMB, 2, (lP, lQ), Target.METRIC,
                engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else hgd2metric_spec(*cls)
    if (spec.target is not Target.METRIC or spec.derivative_order != 2
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d2 HGP Target.METRIC key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the hessian kernels take no levers")
    lP, lQ = spec.am
    return lP, lQ


def hgd2metric_name(cls: Spec | tuple[int, int]) -> str:
    """`hgd2metric_` and the two `l`s, `_`-separated once any is two
    digits."""
    am = _class(cls)
    return "hgd2metric_" + ("_" if max(am) > 9 else "").join(map(str, am))


def _rows(lP: int) -> tuple[int, ...]:
    """The bra `|e|` the kernel reads: two up, level, two down."""
    return tuple(n for n in (lP + 2, lP, lP - 2) if n >= 0)


def hgd2metric_dag(cls: Spec | tuple[int, int]) -> vrr.VrrDag:
    """The VRR the kernel runs: the closure of every `(e, f)` at `m = 0`
    with `|e|` in `{lP + 2, lP, lP - 2}` and `|f| = lQ`."""
    lP, lQ = _class(cls)
    lo = min(_rows(lP))
    whole = vrr.build((lo, lP + 2 - lo, lQ, 0))
    keep = set(_rows(lP))
    group = [n for n in whole.targets if sum(n[0]) in keep]
    return vrr.group_dag(whole, group)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _fold(terms: list[Expr]) -> Expr:
    out = terms[0]
    for t in terms[1:]:
        out = _c(Fn.ADD, out, t)
    return out


def _step(a: Cart, k: int, by: int) -> Cart:
    return tuple(c + by * (d == k) for d, c in enumerate(a))  # type: ignore[return-value]


def _scaled(coef: int, x: Expr) -> Expr:
    return x if coef == 1 else _mul(Lit(float(coef)), x)


def _h(k: int, l: int) -> Var:  # noqa: E741
    return Var(f"h{min(k, l)}{max(k, l)}")


def _contract(lP: int, lQ: int) -> list[Stmt]:
    """`h_kl += u^2 S2 - u S1 + S0` per primitive pair (module docstring)."""
    rowsP, rowsQ = vrr.cartesians(lP), vrr.cartesians(lQ)
    nQ = len(rowsQ)

    def v(e: Cart, f: Cart) -> Var:
        return Var(vrr_name((e, f, 0)))

    out: list[Stmt] = []
    for k, l in KL:  # noqa: E741
        up: list[Expr] = []
        mid: list[Expr] = []
        down: list[Expr] = []
        for a, ra in enumerate(rowsP):
            for b, rb in enumerate(rowsQ):
                w = Var(f"w{a * nQ + b}")
                up.append(_mul(w, v(_step(_step(ra, k, 1), l, 1), rb)))
                # the two middle terms, merged on the Cartesian they share
                mids: dict[Cart, int] = {}
                c1 = _step(ra, k, 1)[l]
                if c1:
                    e = _step(_step(ra, k, 1), l, -1)
                    mids[e] = mids.get(e, 0) + c1
                if ra[k]:
                    e = _step(_step(ra, k, -1), l, 1)
                    mids[e] = mids.get(e, 0) + ra[k]
                for e, coef in mids.items():
                    mid.append(_scaled(coef, _mul(w, v(e, rb))))
                c0 = ra[k] * _step(ra, k, -1)[l]
                if c0:
                    e = _step(_step(ra, k, -1), l, -1)
                    down.append(_scaled(c0, _mul(w, v(e, rb))))
        d: Expr = _mul(Var("tp2"), _fold(up))
        if mid:
            d = _c(Fn.SUB, d, _mul(Var("tp"), _fold(mid)))
        if down:
            d = _c(Fn.ADD, d, _fold(down))
        out.append(Assign(_h(k, l), _c(Fn.ADD, _h(k, l), d)))
    return out


def _scatter() -> list[Stmt]:
    """The 36 atomics of one pair, `h_lk = h_kl` (`kernel_d2metric`'s)."""
    out: list[Stmt] = [Note(ATOMIC_REASON)]
    for X, Y, neg in (("aa", "aa", False), ("ab", "ab", False),
                      ("aa", "ab", True), ("ab", "aa", True)):
        for k in range(3):
            for l in range(3):  # noqa: E741
                val: Expr = (_c(Fn.SUB, Lit(0.0), _h(k, l)) if neg
                             else _h(k, l))
                out.append(AtomicAdd(Index("H", (Var(X), Lit(k), Var(Y),
                                                 Lit(l))), val))
    return out


def _params() -> tuple[Param, ...]:
    """`kernel_d2metric`'s parameters, in its order (restated)."""

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    npair, nrec, naux, nat = (Var("n_pair"), Var("n_rec"), Var("n_aux"),
                              Var("n_atom"))
    return (
        Param("n_pair", Scalar.I32), Param("n_rec", Scalar.I32),
        Param("n_aux", Scalar.I32), Param("n_atom", Scalar.I32),
        f64("A", (nrec, Lit(3))), f64("alpha", (nrec,)),
        f64(S_AUX, (nrec,)),
        *(i32(n, (npair,)) for n in ("p_rec0", "p_rec1", "q_rec0", "q_rec1",
                                     "off_p", "off_q", "atom_p", "atom_q")),
        f64("M", (naux, naux)),
        Param("H", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3), nat, Lit(3))),
    )


def build_hgd2metric_kernel(cls: Spec | tuple[int, int], *,
                            name: str | None = None) -> Kernel:
    """The HGP metric Hessian kernel of one aux class (module docstring);
    its parameters, launch and `H` contract are `kernel_d2metric`'s."""
    lP, lQ = _class(cls)
    dag = hgd2metric_dag((lP, lQ))

    ket_body: tuple[Stmt, ...] = (
        *vrr_ket_body(dag, ZERO_PA | ZERO_QC, "A", Var(KET)),
        Note("h_kl += u^2 SUM w [a+1_k+1_l|b] - u SUM w ((a+1_k)_l "
             "[a+1_k-1_l|b] + a_k [a-1_k+1_l|b]) + SUM w a_k (a-1_k)_l "
             "[a-1_k-1_l|b]: d2(P|Q)/dA_k dA_l by the Cartesian rule twice, "
             "per primitive, u = 2 alpha_P"),
        *_contract(lP, lQ),
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("pe", Scalar.F64, init=Index("alpha", (Var(BRA),))),
        *(Decl(f"A{d}", Scalar.F64, init=Index("A", (Var(BRA), Lit(d))))
          for d in range(3)),
        Note("U_P = s_aux of the bra record: c (pi/alpha)^(3/2), K = 1"),
        Decl("up", Scalar.F64, init=Index(S_AUX, (Var(BRA),))),
        Decl("tp", Scalar.F64, init=_mul(Lit(2.0), Var("pe"))),
        Decl("tp2", Scalar.F64, init=_mul(Var("tp"), Var("tp"))),
        Note("the Q records of the pair"),
        For(KET, Var("q0"), Var("q1"), Lit(1), ket_body),
    )
    pair = Var(PAIR)
    per_pair: tuple[Stmt, ...] = (
        *(Decl(n, Scalar.I32, init=Index(arr, (pair,)))
          for n, arr in (("p0", "p_rec0"), ("p1", "p_rec1"), ("q0", "q_rec0"),
                         ("q1", "q_rec1"), ("oa", "off_p"), ("ob", "off_q"),
                         ("aa", "atom_p"), ("ab", "atom_q"))),
        *pair_weights(lP, lQ),
        *(Decl(_h(*kl).name, Scalar.F64, init=Lit(0.0)) for kl in KL),
        Note("the P records of the pair, serially"),
        For(BRA, Var("p0"), Var("p1"), Lit(1), bra_body),
        Note("d2/dA dB = -h, d2/dB dB = +h: (P|Q) reads A - B"),
        *_scatter(),
    )
    body: tuple[Stmt, ...] = (
        Note(f"hessian of the DF metric on the HGP engine, class (lP, lQ) = "
             f"({lP}, {lQ}): H[A, k, B, l] += SUM_PQ W_PQ d2(P|Q)/dR_{{A,k}} "
             "dR_{B,l} (hessian/kernel_hgd2metric.py)"),
        Note(ZEROED_H),
        Note("launch: one thread per aux shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )
    return Kernel(name or hgd2metric_name((lP, lQ)), _params(), body)
