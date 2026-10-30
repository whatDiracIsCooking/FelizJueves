"""The DF metric gradient on the HGP engine: `Target.METRIC` at d1, `engine=HGP`.

    build_hgmetric_kernel(spec | (lP, lQ)) -> Kernel   hgmetric_{lP}{lQ}
    hgmetric_spec(lP, lQ)                  -> Spec     `.../d1/.../metric.hgp/...`
    hgmetric_name(spec | (lP, lQ))         -> str
    hgmetric_dag(spec | (lP, lQ))          -> VrrDag   the bra one up and down
    HGMETRIC_L_MAX                         -> 1: every shell s or p

The HGP twin of `kernel_gmetric`: per aux shell pair `(P, Q)`, staged as the
d0 kernels stage it, the pair's contribution to

    G[A, k] += SUM_PQ W_PQ d(P|Q) / dR_{A,k}

with the SAME parameters, launch, weights and `G` contract, so a caller
swaps one builder for the other and changes nothing else
(`test/df_grad_launch.run_metric` takes `engine=`). `gradient.host_df.
metric_gradient` is the witness.

THE CARTESIAN RULE, NOT THE HERMITE SHIFT. An HGP class has no Hermite
tower to shift; it has Cartesian bra indices, and the derivative of a
Cartesian Gaussian is two Cartesian Gaussians of the SAME exponent on the
same centre (`felizjueves.gradient`'s package docstring):

    d(a|b)/dA_k = 2 alpha [a + 1_k | b] - a_k [a - 1_k | b]     per primitive
    d(a|b)/dB_k = - d(a|b)/dA_k                    ((P|Q) reads A - B only)

so per primitive pair the kernel runs ONE VRR whose targets are the bra
raised and lowered by one, `(lP + 1 0|lQ 0)` and `(lP - 1 0|lQ 0)` (the
second absent for `lP = 0`): the closure of those targets in the DAG of
`(lP - 1, 2 | lQ, 0)` (`hgmetric_dag`; `hgp.vrr.group_dag`), so the `|e| =
lP` row is never formed. As in `kernel_hgp` (`vrr_ket_body`), an aux
Gaussian is an HGP pair with an exponent-0 partner, and every term in `PA`
or `QC` (both exactly zero) is not spelled. The `2 alpha` belongs to the
bra PRIMITIVE, so it is applied inside the record loops, never on the
contracted values. The second line is exact (the integral depends on `A -
B` alone), as in `kernel_gmetric`, so `SUM_A G[A]` is zero to round-off
and is checked against `host_df`, which differentiates both centres.

PER RECORD PAIR, with `w_ab` the weights (`kernel_g1e.weights`' rule on `W`,
restated: an HGP builder reaches neither `mmd/` nor `plan/`),

    g_k += 2 alpha_P SUM_ab w_ab [a + 1_k|b] - SUM_ab a_k w_ab [a - 1_k|b]

and once per pair `G[atom_p, k] += g_k`, `G[atom_q, k] += (0 - g_k)`: six
`AtomicAdd`s. `G` is ZEROED BY THE CALLER. The pair factor `U_P U_Q` is in
every VRR value already (the fundamental's `pair`), and it is the same for
every Cartesian component, so the raised and lowered values need no other
factor.

ROUND-OFF, NOT BITWISE, against `host_df` and `kernel_gmetric`: another
recurrence and another derivative route associate differently.

SCOPE. The builder accepts any class, but correctness is tested for every
class `lP + lQ <= 4` interpreted, and the CLI ladder (`--hgp-grad-metric-l`)
and goldens are the s and p classes (`HGMETRIC_L_MAX = 1`, as
`kernel_hgp.HGP_DF_L_MAX`). Nothing is measured.

    params  exactly kernel_gmetric's (its module docstring)

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host_df`
(its witnesses), nor `plan/`, nor `mmd/`, nor the interpreter, nor
`reference/`, nor a backend.
"""

from __future__ import annotations

from felizjueves.eri.density_fit.kernel_hgp import (
    S_AUX,
    ZERO_PA,
    ZERO_QC,
    vrr_ket_body,
)
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
    "HGMETRIC_L_MAX",
    "ZEROED_G",
    "build_hgmetric_kernel",
    "hgmetric_dag",
    "hgmetric_name",
    "hgmetric_spec",
]

HGMETRIC_L_MAX = 1  # the highest shell l the CLI ladder and goldens take

PAIR = "pair"  # one thread per aux shell pair
BRA = "ip"  # the P record, serial
KET = "iq"  # the Q record, serial inside the bra

# `kernel_g1e`'s notes, restated (it reaches `mmd/` and `plan/`):
# test/test_kernel_hgmetric.py holds them equal.
ATOMIC_REASON = (
    "AtomicAdd: every record of every pair on an atom adds into that atom's "
    "row of G")
ZEROED_G = (
    "PRECONDITION: G is zeroed by the caller before the first gradient "
    "launch of a build; this kernel only adds")
_WEIGHTS = (
    "w_r = M[a_i, b_j] + M[b_j, a_i] off the diagonal, M[a_i, a_j] on a "
    "diagonal shell pair: SUM_mn M_mn X_mn exactly, for any M")


def hgmetric_spec(lP: int, lQ: int) -> Spec:
    """The aux class `(lP, lQ)` at d1 on the HGP engine."""
    return Spec(Operator.COULOMB, 1, (lP, lQ), Target.METRIC,
                engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else hgmetric_spec(*cls)
    if (spec.target is not Target.METRIC or spec.derivative_order != 1
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d1 HGP Target.METRIC key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    lP, lQ = spec.am
    return lP, lQ


def hgmetric_name(cls: Spec | tuple[int, int]) -> str:
    """`hgmetric_` and the two `l`s, `_`-separated once any is two digits."""
    am = _class(cls)
    return "hgmetric_" + ("_" if max(am) > 9 else "").join(map(str, am))


def _lowered(lP: int) -> bool:
    return lP > 0  # an s bra has no `a - 1_k` term


def hgmetric_dag(cls: Spec | tuple[int, int]) -> vrr.VrrDag:
    """The VRR the kernel runs: the closure of every `(e, f)` at `m = 0`
    with `|e| = lP + 1` or `|e| = lP - 1` and `|f| = lQ`, inside the DAG of
    `(lP - 1, 2 | lQ, 0)` (`(0, 1 | lQ, 0)` for `lP = 0`)."""
    lP, lQ = _class(cls)
    lo = lP - 1 if _lowered(lP) else lP
    whole = vrr.build((lo, lP + 1 - lo, lQ, 0))
    group = [n for n in whole.targets if sum(n[0]) != lP]
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


def _weights(lP: int, lQ: int) -> list[Stmt]:
    """`w{a * nQ + b}` from `M` (`kernel_g1e.weights`' rule, restated)."""
    nP, nQ = len(vrr.cartesians(lP)), len(vrr.cartesians(lQ))

    def m(row: Expr, col: Expr) -> Index:
        return Index("M", (row, col))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    def off(r: int) -> Expr:
        i, j = divmod(r, nQ)
        return _c(Fn.ADD, m(plus("oa", i), plus("ob", j)),
                  m(plus("ob", j), plus("oa", i)))

    rows = range(nP * nQ)
    if lP != lQ:  # a diagonal pair has lP == lQ
        return [Note(_WEIGHTS),
                *(Decl(f"w{r}", Scalar.F64, init=off(r)) for r in rows)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    return [
        Note(_WEIGHTS),
        *(Decl(f"w{r}", Scalar.F64, init=Lit(0.0)) for r in rows),
        If(differ,
           tuple(Assign(Var(f"w{r}"), off(r)) for r in rows),
           tuple(Assign(Var(f"w{r}"), m(plus("oa", r // nQ),
                                        plus("oa", r % nQ))) for r in rows)),
    ]


def _params() -> tuple[Param, ...]:
    """`kernel_gmetric`'s parameters, in its order (restated)."""

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
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3))),
    )


def build_hgmetric_kernel(cls: Spec | tuple[int, int], *,
                          name: str | None = None) -> Kernel:
    """The HGP metric gradient kernel of one aux class (module docstring);
    its parameters, launch and `G` contract are `kernel_gmetric`'s."""
    lP, lQ = _class(cls)
    rowsP, rowsQ = vrr.cartesians(lP), vrr.cartesians(lQ)
    nQ = len(rowsQ)
    dag = hgmetric_dag((lP, lQ))

    def v(e: Cart, f: Cart) -> Var:
        return Var(vrr_name((e, f, 0)))

    contract: list[Stmt] = []
    for k in range(3):
        up = [_mul(Var(f"w{a * nQ + b}"), v(_step(ra, k, 1), rb))
              for a, ra in enumerate(rowsP) for b, rb in enumerate(rowsQ)]
        down = [_mul(Var(f"w{a * nQ + b}"), v(_step(ra, k, -1), rb))
                if ra[k] == 1 else
                _mul(Lit(float(ra[k])), _mul(Var(f"w{a * nQ + b}"),
                                             v(_step(ra, k, -1), rb)))
                for a, ra in enumerate(rowsP) if ra[k]
                for b, rb in enumerate(rowsQ)]
        d: Expr = _mul(Var("tp"), _fold(up))
        if down:
            d = _c(Fn.SUB, d, _fold(down))
        contract.append(Assign(Var(f"g{k}"), _c(Fn.ADD, Var(f"g{k}"), d)))

    ket_body: tuple[Stmt, ...] = (
        *vrr_ket_body(dag, ZERO_PA | ZERO_QC, "A", Var(KET)),
        Note("g_k += 2 alpha_P SUM_ab w_ab [a+1_k|b] - SUM_ab a_k w_ab "
             "[a-1_k|b]: d(P|Q)/dA_k by the Cartesian rule, per primitive"
             + ("" if _lowered(lP) else " (an s bra: no a - 1_k term)")),
        *contract,
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("pe", Scalar.F64, init=Index("alpha", (Var(BRA),))),
        *(Decl(f"A{d}", Scalar.F64, init=Index("A", (Var(BRA), Lit(d))))
          for d in range(3)),
        Note("U_P = s_aux of the bra record: c (pi/alpha)^(3/2), K = 1"),
        Decl("up", Scalar.F64, init=Index(S_AUX, (Var(BRA),))),
        Decl("tp", Scalar.F64, init=_mul(Lit(2.0), Var("pe"))),
        Note("the Q records of the pair"),
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
        *(Decl(n, Scalar.I32, init=Index(arr, (pair,)))
          for n, arr in (("p0", "p_rec0"), ("p1", "p_rec1"), ("q0", "q_rec0"),
                         ("q1", "q_rec1"), ("oa", "off_p"), ("ob", "off_q"),
                         ("aa", "atom_p"), ("ab", "atom_q"))),
        *_weights(lP, lQ),
        *(Decl(f"g{k}", Scalar.F64, init=Lit(0.0)) for k in range(3)),
        Note("the P records of the pair, serially"),
        For(BRA, Var("p0"), Var("p1"), Lit(1), bra_body),
        Note("d/dB_k = -d/dA_k: (P|Q) reads A - B"),
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"gradient of the DF metric on the HGP engine, class (lP, lQ) = "
             f"({lP}, {lQ}): G[A, k] += SUM_PQ W_PQ d(P|Q)/dR_{{A,k}} "
             "(gradient/kernel_hgmetric.py)"),
        Note(ZEROED_G),
        Note("launch: one thread per aux shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )
    return Kernel(name or hgmetric_name((lP, lQ)), _params(), body)
