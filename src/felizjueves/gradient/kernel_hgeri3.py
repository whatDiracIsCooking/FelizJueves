"""The three-center DF gradient kernel on the HGP engine: `Target.ERI3` at
`derivative_order = 1`, `engine=Engine.HGP`.

    build_hgeri3_kernel(spec | (la, lb, lP)) -> Kernel       one per class
    hgeri3_name(spec | (la, lb, lP))         -> str          hgeri3_{la}{lb}{lP}
    hgeri3_spec(la, lb, lP)                  -> Spec         the class's d1 key
    derivative_tables(spec | (la, lb, lP))   -> the HRR classes it runs

The HGP sibling of `kernel_geri3` (McMurchie & Davidson): the same class,
keyed with `engine=Engine.HGP` (``coulomb/d1/(la,lb|lP)/b1/eri3.hgp/
literal``), the same parameters in the same order, the same staging
(`Eri3Class.arrays()`, `s` as `s_aux`, plus `gradient.host_df.eri3_records`'
exponents and atoms), the same thread mapping and the same `G` contract,

    G[A, k] += SUM_mnP Gam_mnP d(mn|P) / dR_{A,k}

per cell (an AO shell pair crossed with an aux shell), nine `AtomicAdd`s
into a `G` ZEROED BY THE CALLER. A caller swaps one builder for the other
and changes nothing else (test/df_grad_launch.py's `run_eri3(engine=)`).

THE RULE, on every centre, the Cartesian one (`felizjueves.gradient`):

    d/dX_k (x y|q) = 2 xi  (x + 1_k, y | q) - x_k (x - 1_k, y | q)
    d/dY_k (x y|q) = 2 eta (x, y + 1_k | q) - y_k (x, y - 1_k | q)
    d/dC_k (x y|q) = 2 gam (x, y | q + 1_k) - q_k (x, y | q - 1_k)

`xi`, `eta`, `gam` the PRIMITIVE exponents of the built bra's two shells and
of the aux shell, so the factor `2 xi` cannot be applied after the
contraction: each raised class is contracted with its own primitive weight.
The aux Gaussian is the HGP pair `(q, 1)` with an exponent-0 partner
(`eri.density_fit.kernel_hgp`'s module docstring), and raising `q` on it is
raising an ordinary Gaussian; so all three centres take the same rule, no
centre is inferred from the others, and `SUM_A G[A] = 0` is a check.

THE CLASSES. The bra is built higher `l` first, `(x, y) = (max, min)`, as
`kernel_hgp.eri3_class` builds it (a swapped class reads the record's `XPB`
as its `PA`, `eb_ket` as `xi`, and its `B` centre as `X`). The derivative
HRR classes (`derivative_tables`), in that orientation:

    uX (x + 1, y | lP)  weight 2 xi      dX (x - 1, y | lP)  weight 1
    uY (x, y + 1 | lP)  weight 2 eta     dY (x, y - 1 | lP)  weight 1
    uC (x, y | lP + 1)  weight 2 gam     dC (x, y | lP - 1)  weight 1

(a lowered class with a negative `l` is absent: its factor `x_k` is zero).
Each is `hgp.hrr.build`'s DAG of `(x', y' | f', 0)` on a contracted
boundary. The HRR is linear and its coefficient `AB` the same for every
primitive pair of the shell pair, so it holds on a contraction weighted by
any per-primitive factor: one accumulator per (weight, boundary value),
`c{w}{e}__{f}`, completed before any HRR reads it (milestone 12, pin 5).

THE VRR runs ONCE per primitive quartet, over the union of every table's
boundary: `hgp.vrr.build` of the rectangle spanning it, cut to the union by
`hgp.vrr.group_dag` (so the Boys top is `L + 1`, not the rectangle's
`L + 2`), the aux side's `QC = 0` terms dropped and everything only they
read gone with them (`kernel_hgp`'s rule). The quartet scalars, the
prelude and `U_P = s_ab K_AB` are `kernel_hgp.build_heri3_kernel`'s.

THE CONTRACTION WITH `M`, after the HRRs: per staged element `(a, b, q)`
the weight `w = M[a, b, q] + M[b, a, q]` off the diagonal and `M[a, b, q]`
on a diagonal shell pair (`kernel_geri3.cell_weights`' rule, restated), and

    gX_k += w (uX(x + 1_k, y, q) - x_k dX(x - 1_k, y, q))

and likewise on `Y` and `C`, the built `X`, `Y` mapped back onto the staged
`A`, `B` by the swap.

ROUND-OFF, NOT BITWISE, against `gradient.host_df` (McMurchie & Davidson on
`reference.pq`) and against `kernel_geri3`: another recurrence.

SCOPE. The builder accepts any class the DAGs accept; correctness is
tested for s and p shells (`kernel_hgp.HGP_DF_L_MAX = 1`, the d0 HGP DF
kernels' scope: the six classes `la <= lb <= 1`, `lP <= 1`, whose
derivative classes reach d), and the CLI's ladder refuses more. No
low-stack form, nothing measured.

    params  exactly kernel_geri3's (its module docstring)

A builder: vendor-neutral IR. It imports neither `ao/` nor
`gradient.host_df` (its witness), nor the interpreter, nor `reference/`,
nor `plan/`, nor the MMD engine (`kernel_geri3`'s and `kernel_g1e`'s names
are restated, and test/test_kernel_hgeri3.py holds them equal).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from felizjueves.eri.density_fit.kernel_hgp import (
    _QC,
    S_AUX,
    _live,
    _quartet_scalars,
    _reads,
    _without,
)
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import (
    PQ_PARAMS,
    fundamental_prelude,
    hrr_body,
    hrr_name,
    vrr_body,
    vrr_name,
)
from felizjueves.hgp.vrr import ZERO, AmClass, Cart
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
    "ZEROED_G",
    "Table",
    "build_hgeri3_kernel",
    "derivative_tables",
    "hgeri3_name",
    "hgeri3_spec",
]

CELL, BRA, KET = "cell", "ib", "ik"  # kernel_geri3's mapping

# `kernel_g1e`'s two notes, restated (it reaches the MMD engine):
# test/test_kernel_hgeri3.py holds them equal.
ATOMIC_REASON = (
    "AtomicAdd: every record of every pair on an atom adds into that atom's "
    "row of G")
ZEROED_G = (
    "PRECONDITION: G is zeroed by the caller before the first gradient "
    "launch of a build; this kernel only adds")
_WEIGHTS = (
    "w_abq = M[a_i, b_j, P_q] + M[b_j, a_i, P_q] off the diagonal, "
    "M[a_i, a_j, P_q] on a diagonal shell pair: SUM_mnP M_mnP X_mnP "
    "exactly, for any M")

# The primitive weight of each kind of accumulator: a bra-record local
# (`2 xi`, `2 eta`), an aux-record local (`2 gam`), or none.
_WEIGHT = {"1": None, "X": "xi2", "Y": "eta2", "C": "gam2"}


class Table(NamedTuple):
    """One derivative HRR: its tag, the built class `(x', y', f', 0)`, and
    its accumulators' weight (`_WEIGHT`'s key)."""

    tag: str
    am: AmClass
    kind: str


def hgeri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` at derivative order 1 on the HGP engine:
    `.../d1/.../eri3.hgp/...`."""
    return Spec(Operator.COULOMB, 1, (la, lb, lP), Target.ERI3,
                engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int, int]) -> tuple[int, int, int]:
    spec = cls if isinstance(cls, Spec) else hgeri3_spec(*cls)
    if (spec.target is not Target.ERI3 or spec.derivative_order != 1
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d1 HGP Target.ERI3 key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb, lP = spec.am
    return la, lb, lP


def hgeri3_name(cls: Spec | tuple[int, int, int]) -> str:
    """`hgeri3_` and the three `l`s, `_`-separated once any is two digits
    (`kernel_geri3.geri3_name`'s rule, `h` for the engine)."""
    am = _class(cls)
    return "hgeri3_" + ("_" if max(am) > 9 else "").join(map(str, am))


def derivative_tables(cls: Spec | tuple[int, int, int]) -> tuple[Table, ...]:
    """The HRR classes the kernel runs, in the built orientation (module
    docstring, THE CLASSES): every raised one, and each lowered one whose
    `l` is not negative."""
    la, lb, lP = _class(cls)
    x, y = max(la, lb), min(la, lb)
    out = [Table("uX", (x + 1, y, lP, 0), "X")]
    if x:
        out.append(Table("dX", (x - 1, y, lP, 0), "1"))
    out.append(Table("uY", (x, y + 1, lP, 0), "Y"))
    if y:
        out.append(Table("dY", (x, y - 1, lP, 0), "1"))
    out.append(Table("uC", (x, y, lP + 1, 0), "C"))
    if lP:
        out.append(Table("dC", (x, y, lP - 1, 0), "1"))
    return tuple(out)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _digits(v: Cart) -> str:
    return "_".join(map(str, v))


def _acc(kind: str, e: Cart, f: Cart) -> str:
    return f"c{kind}{_digits(e)}__{_digits(f)}"


def _shift(v: Cart, k: int, by: int) -> Cart:
    return tuple(c + by * (i == k) for i, c in enumerate(v))  # type: ignore[return-value]


def _plus(base: str, k: int) -> Expr:
    return _c(Fn.ADD, Var(base), Lit(k))


def _vrr(tables: tuple[Table, ...]
         ) -> tuple[vrr.VrrDag, dict[str, list[tuple[Cart, Cart]]]]:
    """The one VRR every table's boundary is cut from, and each kind's
    boundary values in that DAG's target order."""
    want: dict[str, set[tuple[Cart, Cart]]] = {}
    for t in tables:
        want.setdefault(t.kind, set()).update(hrr.build(t.am, "bra").boundary)
    every = set().union(*want.values())
    es = [sum(e) for e, _ in every]
    fs = [sum(f) for _, f in every]
    rect = vrr.build((min(es), max(es) - min(es), min(fs), max(fs) - min(fs)))
    order = [n for n in rect.targets if (n[0], n[1]) in every]
    dag = vrr.group_dag(rect, order)
    return dag, {k: [(n[0], n[1]) for n in order if (n[0], n[1]) in v]
                 for k, v in want.items()}


def _cell_weights(la: int, lb: int, nP: int) -> list[Stmt]:
    """`w{a}_{b}_{q}` per staged element, from `M` (module docstring)."""
    nA, nB = len(vrr.cartesians(la)), len(vrr.cartesians(lb))
    names = [(a, b, q) for a in range(nA) for b in range(nB)
             for q in range(nP)]

    def m(i: Expr, j: Expr, q: int) -> Index:
        return Index("M", (i, j, _plus("op", q)))

    def off(a: int, b: int, q: int) -> Expr:
        return _c(Fn.ADD, m(_plus("oa", a), _plus("ob", b), q),
                  m(_plus("ob", b), _plus("oa", a), q))

    def w(a: int, b: int, q: int) -> str:
        return f"w{a}_{b}_{q}"

    if la != lb:
        return [Note(_WEIGHTS), *(Decl(w(*n), Scalar.F64, init=off(*n))
                                  for n in names)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    return [
        Note(_WEIGHTS),
        *(Decl(w(*n), Scalar.F64, init=Lit(0.0)) for n in names),
        If(differ,
           tuple(Assign(Var(w(*n)), off(*n)) for n in names),
           tuple(Assign(Var(w(a, b, q)), m(_plus("oa", a), _plus("oa", b), q))
                 for a, b, q in names)),
    ]


def _moved(s: str, el: tuple[Cart, Cart, Cart], k: int, by: int
           ) -> tuple[Cart, Cart, Cart]:
    """`el = (x, y, q)` with component `k` of side `s` moved by `by`."""
    i = "XYC".index(s)
    return tuple(_shift(v, k, by) if n == i else v  # type: ignore[return-value]
                 for n, v in enumerate(el))


def build_hgeri3_kernel(cls: Spec | tuple[int, int, int], *,
                        name: str | None = None) -> Kernel:
    """The HGP three-center gradient kernel of one class (module
    docstring); its parameters, launch and `G` contract are
    `kernel_geri3.build_geri3_kernel`'s. `cls` is a d1 HGP `Target.ERI3`
    `Spec` or `(la, lb, lP)`."""
    la, lb, lP = _class(cls)
    swap = la < lb  # built (b a|P): the record's B is the recurrence's X
    xa, xb = ("XPB", "XPA") if swap else ("XPA", "XPB")
    ex, ey = ("eb_ket", "ea_bra") if swap else ("ea_bra", "eb_ket")
    # the staged centre each built one is: X, Y -> A, B (B, A when swapped)
    side = {"X": "B" if swap else "A", "Y": "A" if swap else "B", "C": "C"}
    tables = derivative_tables((la, lb, lP))
    dag, bnd = _vrr(tables)
    top = max(n[2] for n in dag.terms if vrr.is_leaf(n))
    ib, ik = Var(BRA), Var(KET)

    # --- per aux record: the VRR and the weighted boundary sums -------------
    prelude, fundamental = fundamental_prelude(top)
    body: tuple[Stmt, ...] = (
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                      Index("A", (ik, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        *_quartet_scalars(Var("pe"), Var("aq"), Index(S_AUX, (ik,))),
        *prelude,
        *_without(vrr_body(dag, fundamental), _QC),
    )
    accumulate: list[Stmt] = []
    for kind, values in bnd.items():
        for e, f in values:
            v: Expr = Var(vrr_name((e, f, 0)))
            if _WEIGHT[kind] is not None:
                v = _mul(Var(_WEIGHT[kind]), v)
            accumulate.append(Assign(Var(_acc(kind, e, f)), _c(
                Fn.ADD, Var(_acc(kind, e, f)), v)))
    kept = _live(body, {vrr_name(n) for n in dag.targets})
    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        Decl("gam2", Scalar.F64, init=_mul(Lit(2.0), Var("aq"))),
        Note(f"[e0|f0]^(0) by the HGP VRR ([Fock] eqs 25-26), every "
             f"boundary value of the {len(tables)} derivative HRRs at once; "
             "the aux side's QC terms (exactly zero) dropped with everything "
             "only they read"),
        *kept,
        Note("the boundary sums, each class contracted with its own "
             "primitive weight (2 xi, 2 eta, 2 gam, or 1)"),
        *accumulate,
    )

    kdot: Expr | None = None
    for d in range(3):
        t = _mul(Index("XPA", (ib, Lit(d))), Index("XPB", (ib, Lit(d))))
        kdot = t if kdot is None else _c(Fn.ADD, kdot, t)
    assert kdot is not None
    bra_scalars = _live((
        Decl("pe", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        *(Decl(Factor.pa(d).value, Scalar.F64, init=Index(xa, (ib, Lit(d))))
          for d in range(3)),
        Note("U_P = s_ab K_AB, K_AB = exp(p X_PA . X_PB) (kernel_hgp's)"),
        Decl("up", Scalar.F64, init=_mul(Index("s_ab", (ib,)), _c(
            Fn.EXP, _mul(Var("pe"), kdot)))),
        Note("2 xi, 2 eta: the built bra's two primitive exponents"
             + (" (built swapped: the record's b first)" if swap else "")),
        Decl("xi2", Scalar.F64, init=_mul(Lit(2.0), Index(ex, (ib,)))),
        Decl("eta2", Scalar.F64, init=_mul(Lit(2.0), Index(ey, (ib,)))),
    ), _reads(ket_body))
    bra_body: tuple[Stmt, ...] = (
        *bra_scalars,
        Note("the aux records of the cell"),
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )

    # --- once per cell: the HRRs, then the contraction with M ---------------
    def table_name(tag: str) -> Callable[[hrr.Node], str]:
        return lambda n: tag + hrr_name(n)

    hrrs: list[Stmt] = [
        *(Decl(Factor.ab(d).value, Scalar.F64, init=_c(
            Fn.SUB, Index(xb, (Var("b0"), Lit(d))),
            Index(xa, (Var("b0"), Lit(d)))))
          for d in range(3))]
    for t in tables:
        hrrs.append(Note(f"{t.tag}: the bra HRR of {t.am} ([Fock] eq 28) on "
                         "the "
                         + ("unweighted" if t.kind == "1" else "weighted")
                         + " contracted boundary"))
        hrrs += hrr_body(hrr.build(t.am, "bra"),
                         lambda e, f, k=t.kind: Var(_acc(k, e, f)),
                         name=table_name(t.tag))

    rows = (vrr.cartesians(max(la, lb)), vrr.cartesians(min(la, lb)),
            vrr.cartesians(lP))
    contract: list[Stmt] = []
    wanted: set[str] = set()
    for i, cx in enumerate(rows[0]):
        for j, cy in enumerate(rows[1]):
            for q, cq in enumerate(rows[2]):
                a, b = (j, i) if swap else (i, j)
                w = Var(f"w{a}_{b}_{q}")
                el = (cx, cy, cq)
                for s in "XYC":
                    for k in range(3):
                        term: Expr = Var(table_name(f"u{s}")(
                            (*_moved(s, el, k, 1), ZERO)))
                        wanted.add(term.name)
                        n_k = el["XYC".index(s)][k]
                        if n_k:
                            low = Var(table_name(f"d{s}")(
                                (*_moved(s, el, k, -1), ZERO)))
                            wanted.add(low.name)
                            term = _c(Fn.SUB, term, _mul(Lit(float(n_k)), low))
                        g = Var(f"g{side[s]}{k}")
                        contract.append(Assign(g, _c(Fn.ADD, g,
                                                     _mul(w, term))))
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for s, atom in (("A", "aa"), ("B", "ab"), ("C", "ap")):
        for k in range(3):
            scatter.append(AtomicAdd(Index("G", (Var(atom), Lit(k))),
                                     Var(f"g{s}{k}")))

    cell = Var(CELL)
    per_cell: tuple[Stmt, ...] = (
        *(Decl(n, Scalar.I32, init=Index(arr, (cell,)))
          for n, arr in (("b0", "ab_rec0"), ("b1", "ab_rec1"),
                         ("p0", "p_rec0"), ("p1", "p_rec1"), ("oa", "off_a"),
                         ("ob", "off_b"), ("op", "off_p"), ("aa", "atom_a"),
                         ("ab", "atom_b"), ("ap", "atom_p"))),
        *(Decl(_acc(kind, e, f), Scalar.F64, init=Lit(0.0))
          for kind, values in bnd.items() for e, f in values),
        Note("the bra records of the cell's shell pair, serially; every "
             "boundary sum completes before an HRR reads it"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        Note("AB = X_PB - X_PA of the cell's first record"
             + (" (built swapped: B - A)" if swap else "")),
        *_live(tuple(hrrs), wanted),
        *_cell_weights(la, lb, len(rows[2])),
        *(Decl(f"g{s}{k}", Scalar.F64, init=Lit(0.0))
          for s in "ABC" for k in range(3)),
        Note("g += w (2 x up - n x down) per staged element, on all three "
             "centres (the Cartesian rule; the built X, Y mapped back onto "
             "the staged A, B)"),
        *contract,
        *scatter,
    )
    kernel_body: tuple[Stmt, ...] = (
        Note(f"gradient of the three-center DF integral on the HGP engine, "
             f"class (la, lb, lP) = ({la}, {lb}, {lP}): G[A, k] += SUM_mnP "
             "M_mnP d(mn|P)/dR_{A,k} (gradient/kernel_hgeri3.py)"),
        Note(ZEROED_G),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )
    return Kernel(name or hgeri3_name((la, lb, lP)), _params(), kernel_body)


def _params() -> tuple[Param, ...]:
    """`kernel_geri3`'s parameters, in its order (restated)."""

    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    ncell, nbra, naux_rec = Var("n_cell"), Var("n_bra_rec"), Var("n_aux_rec")
    nao, naux, nat = Var("n_ao"), Var("n_aux"), Var("n_atom")
    return (
        *(Param(n, Scalar.I32) for n in ("n_cell", "n_bra_rec", "n_aux_rec",
                                         "n_ao", "n_aux", "n_atom")),
        f64("zeta_ab", (nbra,)), f64("P", (nbra, Lit(3))),
        f64("XPA", (nbra, Lit(3))), f64("XPB", (nbra, Lit(3))),
        f64("s_ab", (nbra,)), f64("ea_bra", (nbra,)), f64("eb_ket", (nbra,)),
        f64("A", (naux_rec, Lit(3))), f64("alpha", (naux_rec,)),
        f64(S_AUX, (naux_rec,)),
        *(i32(n, (ncell,)) for n in ("ab_rec0", "ab_rec1", "p_rec0",
                                     "p_rec1", "off_a", "off_b", "off_p",
                                     "atom_a", "atom_b", "atom_p")),
        f64("M", (nao, nao, naux)),
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3))),
    )
