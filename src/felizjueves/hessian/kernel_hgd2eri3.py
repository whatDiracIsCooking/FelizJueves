"""The three-center DF Hessian on the HGP engine: `Target.ERI3` at
`derivative_order = 2`, `engine=Engine.HGP`.

    build_hgd2eri3_kernel(spec | (la, lb, lP)) -> Kernel   one per class
    hgd2eri3_name(spec | (la, lb, lP))         -> str      hgd2eri3_{la}{lb}{lP}
    hgd2eri3_spec(la, lb, lP)                  -> Spec     the class's d2 key
    hessian_tables(spec | (la, lb, lP))        -> the HRR classes it runs
    COMPONENTS                                 -> the 45 accumulated (X, k, Y, l)

The HGP twin of `kernel_d2eri3`, as `gradient.kernel_hgeri3` is
`kernel_geri3`'s, and that kernel one order up: the same class keyed with
`engine=Engine.HGP` (``coulomb/d2/(la,lb|lP)/b1/eri3.hgp/literal``),
`kernel_d2eri3`'s parameters in its order, `kernel_hgeri3`'s staging, thread
mapping (ONE THREAD PER CELL, `gradient.kernel_gmetric.launch_shape`, not
`kernel_d2eri3`'s one warp per cell) and weights, and the `H` contract

    H[X, k, Y, l] += SUM_mnP Gam_mnP d2(mn|P) / dR_{X,k} dR_{Y,l}

per cell, 81 `AtomicAdd`s into an `H` ZEROED BY THE CALLER.
`hessian.host_eri3.eri3_hessian` is the witness.

THE RULE, the Cartesian one on every centre, applied twice. With `D^s_k x =
u_s [x + 1_k] - x_k [x - 1_k]` on side `s` in `X`, `Y`, `C` (`u_X = 2 xi`,
`u_Y = 2 eta`, `u_C = 2 gam`, the PRIMITIVE exponents of the built bra's
shells and of the aux shell; the aux Gaussian an HGP pair with an
exponent-0 partner, so raising `q` is raising an ordinary Gaussian):

    same side  D^s_l D^s_k x = u_s^2 [x + 1_k + 1_l]
                             - u_s (x + 1_k)_l [x + 1_k - 1_l]
                             - u_s x_k [x - 1_k + 1_l]
                             + x_k (x - 1_k)_l [x - 1_k - 1_l]
    two sides  D^s_k D^t_l   = u_s u_t [s + 1_k, t + 1_l] - u_s t_l [s + 1_k, t - 1_l]
                             - u_t s_k [s - 1_k, t + 1_l] + s_k t_l [s - 1_k, t - 1_l]

Every term is a class `(x + dx, y + dy | lP + dc)` (`|d| <= 2` on one side,
`<= 1` on each of two) weighted by a monomial in `u_X, u_Y, u_C` of degree
at most two: a TABLE (`hessian_tables`) is one such `(class, weight)`, its
bra HRR (`hgp.hrr`) run on boundary sums contracted with that weight, one
accumulator per (weight, boundary value) as in `kernel_hgeri3` (the HRR is
linear and its `AB` the same for every primitive of the shell pair). No
centre is inferred from the others, so `SUM_A H[A, k, B, l] = 0` is a check.
`D^s_k D^s_l = D^s_l D^s_k`, so a same-centre block needs `k <= l` alone.

THE VRR runs ONCE per primitive quartet over the union of every table's
boundary, cut from the rectangle spanning it (`kernel_hgeri3`'s `_vrr`,
the aux side's `QC = 0` terms dropped).

THE CONTRACTION, after the HRRs: per staged element `(a, b, q)` with weight
`w` (`kernel_hgeri3`'s rule) and each of the 45 components `(X, k, Y, l)`
(`COMPONENTS`: `AA`, `BB`, `CC` with `k <= l`, `AB`, `AC`, `BC` all nine),
`hXkYl += w SUM_terms coef table(moved element)`, the built `X`, `Y` mapped
back onto the staged `A`, `B` by the swap. Once per cell each value goes into
`H[X, k, Y, l]` and, off a same-centre block's diagonal, its mirror
`H[Y, l, X, k]`: 81 atomics.

ROUND-OFF, NOT BITWISE, against `hessian.host_eri3` (moved 3-D Cartesians
through Hermite `E` on `reference.pq`) and `kernel_d2eri3`: another
recurrence.

SCOPE. The builder accepts any class the DAGs accept; correctness is
tested interpreted for every class `la + lb + lP <= 3` (4 `slow`), and the
CLI ladder (`--hgp-hess-eri3-l`) and goldens are the s and p classes
(`kernel_hgp.HGP_DF_L_MAX = 1`, the d0 and d1 HGP DF kernels' scope). One
form, nothing measured, no card.

    params  exactly kernel_d2eri3's (its module docstring)

A builder: vendor-neutral IR. It imports neither `ao/` nor
`hessian.host_eri3` (its witness), nor the interpreter, nor `reference/`,
nor `plan/`, nor the MMD engine (`kernel_d2eri3`'s parameters and
`kernel_h1e`'s notes are restated; test/test_kernel_hgd2eri3.py holds them
equal).
"""

from __future__ import annotations

from collections.abc import Callable

from felizjueves.eri.density_fit.kernel_hgp import (
    _QC,
    S_AUX,
    _live,
    _quartet_scalars,
    _reads,
    _without,
)
from felizjueves.gradient.kernel_hgeri3 import (
    CELL,
    Table,
    _acc,
    _cell_weights,
    _moved,
    _vrr,
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
from felizjueves.hgp.vrr import ZERO, Cart
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
    "COMPONENTS",
    "ZEROED_H",
    "build_hgd2eri3_kernel",
    "hessian_tables",
    "hgd2eri3_name",
    "hgd2eri3_spec",
]

BRA, KET = "ib", "ik"  # kernel_hgeri3's mapping

# `kernel_h1e`'s two notes, restated (it reaches the MMD engine):
# test/test_kernel_hgd2eri3.py holds them equal.
ATOMIC_REASON = (
    "AtomicAdd: every record of every pair on an atom adds into that atom's "
    "blocks of H")
ZEROED_H = (
    "PRECONDITION: H is zeroed by the caller before the first hessian "
    "launch of a build; this kernel only adds")

# The cell's 45 staged centre-centre components `(X, k, Y, l)`.
COMPONENTS: tuple[tuple[str, int, str, int], ...] = tuple(
    (X, k, Y, l)
    for X, Y in (("A", "A"), ("A", "B"), ("A", "C"), ("B", "B"), ("B", "C"),
                 ("C", "C"))
    for k in range(3) for l in range(3)  # noqa: E741
    if X != Y or k <= l)

# A side's primitive factor `u_s`, a bra-record local or an aux-record one.
_FACTOR = {"X": "xi2", "Y": "eta2", "C": "gam2"}

El = tuple[Cart, Cart, Cart]
# (weight kind, shift (dX, dY, dC)) -> one table
TableKey = tuple[str, tuple[int, int, int]]


def hgd2eri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` at derivative order 2 on the HGP engine:
    `.../d2/.../eri3.hgp/...`."""
    return Spec(Operator.COULOMB, 2, (la, lb, lP), Target.ERI3,
                engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int, int]) -> tuple[int, int, int]:
    spec = cls if isinstance(cls, Spec) else hgd2eri3_spec(*cls)
    if (spec.target is not Target.ERI3 or spec.derivative_order != 2
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d2 HGP Target.ERI3 key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the hessian kernels take no levers")
    la, lb, lP = spec.am
    return la, lb, lP


def hgd2eri3_name(cls: Spec | tuple[int, int, int]) -> str:
    """`hgd2eri3_` and the three `l`s, `_`-separated once any is two
    digits (`kernel_hgeri3.hgeri3_name`'s rule)."""
    am = _class(cls)
    return "hgd2eri3_" + ("_" if max(am) > 9 else "").join(map(str, am))


def _kind(sides: str) -> str:
    """The weight monomial's canonical spelling: `XYC` order, `1` for none."""
    return "".join(sorted(sides, key="XYC".index)) or "1"


def _shift(*moves: tuple[str, int]) -> tuple[int, int, int]:
    out = [0, 0, 0]
    for s, by in moves:
        out["XYC".index(s)] += by
    return tuple(out)  # type: ignore[return-value]


def _terms(el: El, s: str, k: int, t: str, l: int  # noqa: E741
           ) -> dict[tuple[TableKey, El], int]:
    """`D^s_k D^t_l` of the element `el`: each nonzero term's (table,
    moved element) and its integer coefficient (module docstring)."""
    i, j = "XYC".index(s), "XYC".index(t)
    out: dict[tuple[TableKey, El], int] = {}

    def add(kind: str, moves, e: El, coef: int) -> None:
        if coef:
            key = ((_kind(kind), _shift(*moves)), e)
            out[key] = out.get(key, 0) + coef

    if s == t:
        up = _moved(s, el, k, 1)
        add(s + s, ((s, 2),), _moved(s, up, l, 1), 1)
        add(s, (), _moved(s, up, l, -1), -up[i][l])
        down = _moved(s, el, k, -1)
        add(s, (), _moved(s, down, l, 1), -el[i][k])
        add("", ((s, -2),), _moved(s, down, l, -1), el[i][k] * down[i][l])
    else:
        su, sd = _moved(s, el, k, 1), _moved(s, el, k, -1)
        add(s + t, ((s, 1), (t, 1)), _moved(t, su, l, 1), 1)
        add(s, ((s, 1), (t, -1)), _moved(t, su, l, -1), -el[j][l])
        add(t, ((s, -1), (t, 1)), _moved(t, sd, l, 1), -el[i][k])
        add("", ((s, -1), (t, -1)), _moved(t, sd, l, -1), el[i][k] * el[j][l])
    return {key: c for key, c in out.items() if c}


def _tag(key: TableKey) -> str:
    kind, shift = key
    return f"t{kind}" + "".join(str(d + 2) for d in shift)


def _built(la: int, lb: int) -> tuple[bool, dict[str, str]]:
    """Whether the bra is built swapped, and each staged centre's built
    side (`kernel_hgeri3`'s orientation, inverted)."""
    swap = la < lb
    return swap, {"A": "Y" if swap else "X", "B": "X" if swap else "Y",
                  "C": "C"}


def _contraction(la: int, lb: int, lP: int):
    """Every element's per-component terms, in the built orientation:
    `[(a, b, q, component, {(table, moved): coef})]`."""
    swap, side = _built(la, lb)
    rows = (vrr.cartesians(max(la, lb)), vrr.cartesians(min(la, lb)),
            vrr.cartesians(lP))
    out = []
    for i, cx in enumerate(rows[0]):
        for j, cy in enumerate(rows[1]):
            for q, cq in enumerate(rows[2]):
                a, b = (j, i) if swap else (i, j)
                for comp in COMPONENTS:
                    X, k, Y, l = comp  # noqa: E741
                    terms = _terms((cx, cy, cq), side[X], k, side[Y], l)
                    out.append((a, b, q, comp, terms))
    return out


def hessian_tables(cls: Spec | tuple[int, int, int]) -> tuple[Table, ...]:
    """The HRR classes the kernel runs, in the built orientation, each
    with its weight kind (module docstring): every `(class, weight)` some
    nonzero term reads, in a fixed order."""
    la, lb, lP = _class(cls)
    x, y = max(la, lb), min(la, lb)
    keys = {key for *_, terms in _contraction(la, lb, lP)
            for key, _ in terms}
    order = sorted(keys, key=lambda k: (len(k[0].strip("1")), k[0], k[1]))
    return tuple(Table(_tag(k), (x + k[1][0], y + k[1][1], lP + k[1][2], 0),
                       k[0]) for k in order)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _fold(terms: list[Expr]) -> Expr:
    out = terms[0]
    for t in terms[1:]:
        out = _c(Fn.ADD, out, t)
    return out


def _h(comp: tuple[str, int, str, int]) -> str:
    X, k, Y, l = comp  # noqa: E741
    return f"h{X}{k}{Y}{l}"


def _canonical(X: str, k: int, Y: str, l: int  # noqa: E741
               ) -> tuple[str, int, str, int]:
    if X == Y:
        return (X, min(k, l), Y, max(k, l))
    if "ABC".index(X) > "ABC".index(Y):
        return (Y, l, X, k)
    return (X, k, Y, l)


def _scatter() -> list[Stmt]:
    """The 81 atomics of one cell: every `(X, k, Y, l)` of the nine
    centre pairs from its canonical accumulator (module docstring)."""
    out: list[Stmt] = [Note(ATOMIC_REASON)]
    atom = {"A": "aa", "B": "ab", "C": "ap"}
    for X in "ABC":
        for Y in "ABC":
            for k in range(3):
                for l in range(3):  # noqa: E741
                    out.append(AtomicAdd(
                        Index("H", (Var(atom[X]), Lit(k), Var(atom[Y]),
                                    Lit(l))),
                        Var(_h(_canonical(X, k, Y, l)))))
    return out


def _params() -> tuple[Param, ...]:
    """`kernel_d2eri3`'s parameters, in its order (restated)."""

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
        Param("H", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3), nat, Lit(3))),
    )


def build_hgd2eri3_kernel(cls: Spec | tuple[int, int, int], *,
                          name: str | None = None) -> Kernel:
    """The HGP three-center Hessian kernel of one class (module
    docstring); its parameters and `H` contract are
    `kernel_d2eri3.build_d2eri3_kernel`'s, its mapping
    `kernel_hgeri3`'s (one thread per cell)."""
    la, lb, lP = _class(cls)
    swap, _ = _built(la, lb)
    xa, xb = ("XPB", "XPA") if swap else ("XPA", "XPB")
    ex, ey = ("eb_ket", "ea_bra") if swap else ("ea_bra", "eb_ket")
    tables = hessian_tables((la, lb, lP))
    tag_of = {(t.kind, (t.am[0] - max(la, lb), t.am[1] - min(la, lb),
                        t.am[2] - lP)): t.tag for t in tables}
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
        weight: Expr | None = None
        for s in kind.strip("1"):
            f = Var(_FACTOR[s])
            weight = f if weight is None else _mul(weight, f)
        for e, f_ in values:
            v: Expr = Var(vrr_name((e, f_, 0)))
            if weight is not None:
                v = _mul(weight, v)
            accumulate.append(Assign(Var(_acc(kind, e, f_)), _c(
                Fn.ADD, Var(_acc(kind, e, f_)), v)))
    kept = _live(body, {vrr_name(n) for n in dag.targets})
    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        Decl("gam2", Scalar.F64, init=_mul(Lit(2.0), Var("aq"))),
        Note(f"[e0|f0]^(0) by the HGP VRR ([Fock] eqs 25-26), every "
             f"boundary value of the {len(tables)} second-derivative HRRs at "
             "once; the aux side's QC terms (exactly zero) dropped with "
             "everything only they read"),
        *kept,
        Note("the boundary sums, each contracted with its own primitive "
             "weight (a product of at most two of 2 xi, 2 eta, 2 gam, or 1)"),
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
                         f"the boundary sums of weight {t.kind}"))
        hrrs += hrr_body(hrr.build(t.am, "bra"),
                         lambda e, f, k=t.kind: Var(_acc(k, e, f)),
                         name=table_name(t.tag))

    contract: list[Stmt] = []
    wanted: set[str] = set()
    for a, b, q, comp, terms in _contraction(la, lb, lP):
        parts: list[Expr] = []
        for (key, moved), coef in terms.items():
            v = Var(table_name(tag_of[key])((*moved, ZERO)))
            wanted.add(v.name)
            parts.append(v if coef == 1 else _mul(Lit(float(coef)), v))
        if not parts:
            continue
        h = Var(_h(comp))
        contract.append(Assign(h, _c(Fn.ADD, h, _mul(
            Var(f"w{a}_{b}_{q}"), _fold(parts)))))

    cell = Var(CELL)
    nP = len(vrr.cartesians(lP))
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
        *_cell_weights(la, lb, nP),
        *(Decl(_h(c), Scalar.F64, init=Lit(0.0)) for c in COMPONENTS),
        Note("h += w SUM coef table(moved element) per staged element and "
             "component: the Cartesian rule twice, on all three centres (the "
             "built X, Y mapped back onto the staged A, B)"),
        *contract,
        *_scatter(),
    )
    kernel_body: tuple[Stmt, ...] = (
        Note(f"hessian of the three-center DF integral on the HGP engine, "
             f"class (la, lb, lP) = ({la}, {lb}, {lP}): H[X, k, Y, l] += "
             "SUM_mnP M_mnP d2(mn|P)/dR_{X,k} dR_{Y,l} "
             "(hessian/kernel_hgd2eri3.py)"),
        Note(ZEROED_H),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )
    return Kernel(name or hgd2eri3_name((la, lb, lP)), _params(),
                  kernel_body)
