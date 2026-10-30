"""The overlap gradient by the HGP route: `Target.S` at `derivative_order =
1` on `Engine.HGP`, built as `Kernel`s.

    build_hgs_kernel(spec | (la, lb)) -> Kernel   one per class
    hgs_name(spec | (la, lb))         -> str      hgs_{la}{lb}
    hgs_spec(la, lb)                  -> Spec     the class's d1 HGP key
    pair_arrays()                     -> names    its per-pair arrays
    launch_shape(n_pair, threads)     -> (grid, block)

The same contraction as `kernel_g1e`'s `gs_` (`felizjueves.gradient`'s
package docstring; `gradient.host.overlap_gradient` is the reference),

    G[A, k] += SUM_mn M_mn dS_mn / dR_{A,k}

on the recurrence of `one_electron.kernel_hgp`'s `hs_`: Obara & Saika's
overlap VRR per primitive pair, the contraction, then Head-Gordon & Pople's
HRR on contracted values. `[Fock]` eq 25 at `f = 0` with the `m + 1` terms
dropped, and eq 28's bra half, both `hgp/`'s and not rewritten
(`kernel_hgp`'s docstring, "THE RECURRENCES").

THE DERIVATIVE RULE is the package docstring's, on the Cartesian of the
differentiated centre:

    dA_k (a|b) = 2 alpha (a+1_k|b) - a_k (a-1_k|b)
    dB_k (a|b) = 2 beta  (a|b+1_k) - b_k (a|b-1_k)

`alpha` and `beta` are per PRIMITIVE, so -- as `kernel_hgp`'s `T` folds
`beta^n` -- each is folded into the boundary before the contraction:
three contracted tables, each with its own HRR,

    table 0   SUM c_a c_b         (e|0)   targets (a-1_k|b), (a|b-1_k)
    table 1   SUM c_a c_b 2 alpha (e|0)   targets (a+1_k|b)
    table 2   SUM c_a c_b 2 beta  (e|0)   targets (a|b+1_k)

(table 0 is absent at (ss)), one VRR to `la + lb + 1` per record feeding
all three. Then, per Cartesian pair `r = (a, b)` of the block, its weight
`w_r` from `M` (`kernel_g1e.weights`, `gs_`'s rule: `M[a, b] + M[b, a]`
off the diagonal, `M[a, a']` on a diagonal shell pair),

    gA_k = SUM_r w_r (t1(a+1_k|b) - a_k t0(a-1_k|b))
    gB_k = SUM_r w_r (t2(a|b+1_k) - b_k t0(a|b-1_k))

and `G[atom_a, k] += gA_k`, `G[atom_b, k] += gB_k`, six `AtomicAdd`s per
SHELL PAIR (`gs_` adds six per primitive pair). Both centres are
computed: no translational invariance is used (the package docstring), so
`SUM_A G[A] = 0` stays a check.

THE THREAD MAPPING is `kernel_hgp`'s: ONE THREAD PER CONTRACTED SHELL PAIR,
`pair = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by `pair < n_pair`, the
pair's records `[rec0, rec1)` serially, the boundary accumulated in
registers from `0.0`, the HRRs once. Unlike `hs_` it does not store: every
pair on an atom adds into that atom's row of `G`, so the adds are atomic
(`ATOMIC_REASON`) and `G` is ZEROED BY THE CALLER (`kernel_g1e.ZEROED_G`).

THE PARAMETER BLOCK, `kernel_hgp`'s record arrays (`S`'s) plus the two
exponents, its per-pair block plus the atoms, and `gs_`'s `M` and `G`:

    params  n_pair n_rec n_ao n_atom (I32)
            zeta_ab pa_x pa_y pa_z pb_x pb_y pb_z s_ab ea_bra eb_ket
                                                           [n_rec] per record
            rec0 rec1 off_a off_b atom_a atom_b  [n_pair]  (I32) per pair
            ab_x ab_y ab_z                       [n_pair]  A - B
            M[n_ao, n_ao]                                  read, any M
            G[n_atom, 3]                       (OUT, read and AtomicAdd-ed)

Staged by `gradient.host.pair_records` (the per-record arrays are
`host.records`', so `gs_` and `hgs_` read one primitive-pair sweep).

SCOPE: shells through f (`L_MAX = 3`), the classes `gs_` is validated
through; `kernel_hgp.contracted_tables`' machinery is built for any `l`.

ROUND-OFF, NOT BITWISE, against the host: another recurrence, another
association (`kernel_hgp`'s "ROUND-OFF"). test/test_kernel_hgs.py states
its bound.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
(its witness), nor the interpreter, nor `reference/`.
"""

from __future__ import annotations

from felizjueves.gradient.kernel_g1e import ZEROED_G, weights
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import hrr_body, hrr_name, vrr_body, vrr_name
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
from felizjueves.one_electron.kernel_hgp import (
    acc_name,
    boundary_union,
    factors,
    table_dags,
    vrr_dag,
)
from felizjueves.spec import OPERATOR_OF, Engine, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "L_MAX",
    "PAIR",
    "build_hgs_kernel",
    "hgs_name",
    "hgs_spec",
    "launch_shape",
    "pair_arrays",
]

L_MAX = 3  # through f: gs_'s validated scope
PAIR = "pair"  # the contracted shell pair: one thread each
_AXES = "xyz"
_REC = "kr"  # the record loop's variable, kernel_hgp's
# Each table's per-record weight: none, 2 alpha, 2 beta (module docstring).
_WEIGHT = {0: None, 1: "ea2", 2: "eb2"}

ATOMIC_REASON = (
    "AtomicAdd: every shell pair on an atom adds into that atom's row of G")


def hgs_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of the overlap gradient on `Engine.HGP`."""
    return Spec(OPERATOR_OF[Target.S], 1, (la, lb), Target.S,
                engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else hgs_spec(*cls)
    if (spec.target is not Target.S or spec.derivative_order != 1
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d1 HGP S key; the MMD "
                         "overlap gradient is kernel_g1e's gs_")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb = spec.am
    if max(la, lb) > L_MAX:
        raise ValueError(f"{spec.key()}: the HGP overlap gradient is built "
                         f"for shells through f only (l <= {L_MAX})")
    return la, lb


def hgs_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_hgs_kernel(cls)` gives its kernel, unbuilt."""
    la, lb = _class(cls)
    return f"hgs_{la}{lb}"


def pair_arrays() -> tuple[str, ...]:
    """The per-pair array parameters, in order."""
    return ("rec0", "rec1", "off_a", "off_b", "atom_a", "atom_b",
            "ab_x", "ab_y", "ab_z")


def launch_shape(n_pair: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per shell pair,
    `ceil(n_pair/threads)` blocks; a partial last block is legal."""
    if type(n_pair) is not int or n_pair < 1:
        raise ValueError(f"n_pair = {n_pair!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_pair // threads), (threads, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _fold(xs: list[Expr]) -> Expr:
    out = xs[0]
    for x in xs[1:]:
        out = _c(Fn.ADD, out, x)
    return out


def _shift(v: Cart, d: int, by: int) -> Cart:
    return tuple(c + by * (i == d) for i, c in enumerate(v))  # type: ignore[return-value]


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def _rows(la: int, lb: int) -> list[tuple[Cart, Cart]]:
    """The block's Cartesian pairs, `a` outer: `kernel_g1e.weights`' rows."""
    return [(a, b) for a in vrr.cartesians(la) for b in vrr.cartesians(lb)]


def _tables(la: int, lb: int) -> dict[int, tuple[hrr.Node, ...]]:
    """Each table's HRR targets (module docstring); an empty one left out."""
    lo: list[hrr.Node] = []
    up_a: list[hrr.Node] = []
    up_b: list[hrr.Node] = []
    for a, b in _rows(la, lb):
        for d in range(3):
            up_a.append((_shift(a, d, 1), b, ZERO, ZERO))
            up_b.append((a, _shift(b, d, 1), ZERO, ZERO))
            if a[d]:
                lo.append((_shift(a, d, -1), b, ZERO, ZERO))
            if b[d]:
                lo.append((a, _shift(b, d, -1), ZERO, ZERO))
    out = {0: lo, 1: up_a, 2: up_b}
    return {n: tuple(dict.fromkeys(t)) for n, t in out.items() if t}


def _hname(table: int, node: hrr.Node) -> str:
    return f"t{table}{hrr_name(node)}"


def _derivatives(la: int, lb: int) -> list[Stmt]:
    """`gA_k`, `gB_k` as one fold each over the block's rows."""
    out: list[Stmt] = []
    for side in "AB":
        for k in range(3):
            terms: list[Expr] = []
            for r, (a, b) in enumerate(_rows(la, lb)):
                if side == "A":
                    up = _hname(1, (_shift(a, k, 1), b, ZERO, ZERO))
                    n, lo = a[k], (_shift(a, k, -1), b, ZERO, ZERO)
                else:
                    up = _hname(2, (a, _shift(b, k, 1), ZERO, ZERO))
                    n, lo = b[k], (a, _shift(b, k, -1), ZERO, ZERO)
                v: Expr = Var(up)
                if n:
                    v = _c(Fn.SUB, v, _c(Fn.MUL, Lit(float(n)),
                                         Var(_hname(0, lo))))
                terms.append(_c(Fn.MUL, Var(f"w{r}"), v))
            out.append(Decl(f"g{side}{k}", Scalar.F64, init=_fold(terms)))
    return out


def build_hgs_kernel(cls: Spec | tuple[int, int], *,
                     name: str | None = None) -> Kernel:
    """The HGP overlap-gradient kernel of one class.

    `cls` is a d1 HGP `Target.S` `Spec`, or `(la, lb)`. See the module
    docstring.
    """
    la, lb = _class(cls)
    hrrs, bounds = table_dags(_tables(la, lb))
    vdag = vrr_dag(boundary_union(bounds), False)
    k = Var(_REC)

    vstmts = vrr_body(vdag, lambda m: Var("u"))
    used = factors(vstmts)
    accumulate: list[Stmt] = []
    for n, bnd in bounds.items():
        for e in bnd:
            x: Expr = Var(vrr_name((e, ZERO, 0)))
            if _WEIGHT[n] is not None:
                x = _c(Fn.MUL, Var(_WEIGHT[n]), x)
            accumulate.append(Assign(Var(acc_name(n, e)), _c(
                Fn.ADD, Var(acc_name(n, e)), x)))

    record: list[Stmt] = [Decl("zeta", Scalar.F64,
                               init=Index("zeta_ab", (k,)))]
    if Factor.INV2P.value in used:
        record.append(Decl(Factor.INV2P.value, Scalar.F64,
                           init=_c(Fn.DIV, Lit(0.5), Var("zeta"))))
    for d in range(3):
        if Factor.pa(d).value in used:
            record.append(Decl(Factor.pa(d).value, Scalar.F64,
                               init=Index(f"pa_{_AXES[d]}", (k,))))
    record += [
        Note("u = (0|0) = s_ab K_AB, K_AB = exp(zeta (P-A).(P-B)): s_ab = "
             "c_a c_b (pi/zeta)^{3/2} is the host's (ao.pairs.dress)"),
        Decl("u", Scalar.F64, init=_c(Fn.MUL, Index("s_ab", (k,)), _c(
            Fn.EXP, _c(Fn.MUL, Var("zeta"), _fold([
                _c(Fn.MUL, Index(f"pa_{a}", (k,)), Index(f"pb_{a}", (k,)))
                for a in _AXES]))))),
        Note("the tables' weights: 2 alpha (table 1) and 2 beta (table 2), "
             "the derivative rule's, per primitive"),
        Decl("ea2", Scalar.F64,
             init=_c(Fn.MUL, Lit(2.0), Index("ea_bra", (k,)))),
        Decl("eb2", Scalar.F64,
             init=_c(Fn.MUL, Lit(2.0), Index("eb_ket", (k,)))),
        *vstmts,
        *accumulate,
    ]

    hstmts: list[Stmt] = []
    for n, dag in hrrs.items():
        what = {0: "", 1: " 2 alpha", 2: " 2 beta"}[n]
        hstmts.append(Note(f"table {n}: SUM c_a c_b{what} (e|0)"))
        hstmts += hrr_body(dag, lambda e, f, n=n: Var(acc_name(n, e)),
                           name=lambda node, n=n: _hname(n, node))
    used_ab = factors(tuple(hstmts))

    pair = Var(PAIR)
    head: list[Stmt] = [
        Decl("r0", Scalar.I32, init=Index("rec0", (pair,))),
        Decl("r1", Scalar.I32, init=Index("rec1", (pair,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (pair,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (pair,))),
        Decl("aa", Scalar.I32, init=Index("atom_a", (pair,))),
        Decl("ab", Scalar.I32, init=Index("atom_b", (pair,))),
    ]
    for d in range(3):
        if Factor.ab(d).value in used_ab:
            head.append(Decl(Factor.ab(d).value, Scalar.F64,
                             init=Index(f"ab_{_AXES[d]}", (pair,))))
    head.append(Note("the contracted boundary (e|0) of each table, summed "
                     "from 0.0 over the pair's records"))
    for n, bnd in bounds.items():
        head += [Decl(acc_name(n, e), Scalar.F64, init=Lit(0.0)) for e in bnd]
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for side, atom in (("A", "aa"), ("B", "ab")):
        for d in range(3):
            scatter.append(AtomicAdd(Index("G", (Var(atom), Lit(d))),
                                     Var(f"g{side}{d}")))
    per_pair = (
        *head,
        Note("the pair's primitive pairs, serially: the VRR per record "
             "([Fock] eq 25 at f = 0, the overlap limit) to la + lb + 1"),
        For(_REC, Var("r0"), Var("r1"), Lit(1), tuple(record)),
        *hstmts,
        *weights(la, lb),
        Note("dA_k (a|b) = 2 alpha (a+1_k|b) - a_k (a-1_k|b), dB_k likewise"),
        *_derivatives(la, lb),
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"HGP gradient of the overlap, class (la, lb) = ({la}, {lb}): "
             "G[A, k] += SUM_mn M_mn dS_mn/dR_{A,k} "
             "(gradient/kernel_hgs.py)"),
        Note(ZEROED_G),
        Note("launch: one thread per contracted shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )

    npair, nr = Var("n_pair"), Var("n_rec")
    nao, nat = Var("n_ao"), Var("n_atom")
    params = [
        Param("n_pair", Scalar.I32),
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_atom", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        *(_f64(f"pa_{a}", (nr,)) for a in _AXES),
        *(_f64(f"pb_{a}", (nr,)) for a in _AXES),
        _f64("s_ab", (nr,)),
        _f64("ea_bra", (nr,)),
        _f64("eb_ket", (nr,)),
        *(_i32(n, (npair,)) for n in pair_arrays()[:6]),
        *(_f64(n, (npair,)) for n in pair_arrays()[6:]),
        _f64("M", (nao, nao)),
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3))),
    ]
    return Kernel(name or hgs_name((la, lb)), tuple(params), body)
