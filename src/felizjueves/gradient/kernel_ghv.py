"""The nuclear-attraction gradient on the HGP engine: `Target.V` at
`derivative_order = 1`, `Engine.HGP`, built as `Kernel`s.

    build_ghv_kernel(spec | (la, lb)) -> Kernel   one per class
    ghv_name(spec | (la, lb))         -> str      ghv_{la}{lb}, unbuilt
    ghv_spec(la, lb)                  -> Spec     nuclear/d1/(la,lb)/b1/v.hgp/...
    pair_arrays()                     -> names    its per-pair arrays
    launch_shape(n_pair, threads)     -> (grid, block)

The same contraction `kernel_g1e`'s `gv_` writes (`felizjueves.gradient`'s
package docstring; `gradient.host.potential_gradient` is the reference),

    G[X, k] += SUM_mn M_mn dV_mn / dR_{X,k},   V_mn = -SUM_C Z_C (m|1/r_C|n)

every nucleus's Hellmann-Feynman term included, by the route of
`one_electron.kernel_hgp`'s `hv_`: Obara & Saika's VRR per primitive pair
(`hgp.vrr`'s eq 25 at `f = 0`, its point-charge limit), the contraction,
then `hgp.hrr`'s eq 28 on contracted values. No `E`, no Hermite `[t]^(0)`:
nothing of MMD's. Shells through f (`L_MAX = 3`, `kernel_hgt`'s scope;
`hv_` itself is built for s and p), Cartesian. Unrolled at every class: at
(ff) 6,104 B of `sm_86` stack and 4,708 B of `gfx1200` scratch, under
#613's 14,000 B, so no low-stack form (a reading, not a measured row).

THE DERIVATIVE RULE, on a primitive Cartesian Gaussian of exponent `alpha`
on `A`, `d/dA_k |a) = 2 alpha |a + 1_k) - a_k |a - 1_k)`, so per nucleus

    dA_k (a|b) = SUM_prim 2 alpha (a + 1_k|b) - a_k (a - 1_k|b)
    dB_k (a|b) = SUM_prim 2 beta  (a|b + 1_k) - b_k (a|b - 1_k)
    dC_k (a|b) = -(dA_k + dB_k) (a|b)                 translational invariance

The exponent is per primitive and the HRR's `AB` is not, so the weight
`2 alpha` (`2 beta`) is folded in BEFORE the contraction, `kernel_hgp`'s
`T` rule (its `beta^n` tables): three contracted tables per nucleus,

    table 1   SUM 2 alpha (e|0)    HRR targets (a + 1_k|b)
    table 2   SUM 2 beta  (e|0)    HRR targets (a|b + 1_k)
    table 0   SUM (e|0)            HRR targets (a - 1_k|b), (a|b - 1_k)

each its own HRR (`kernel_hgp.table_dags`), all three on one VRR to
`la + lb + 1` (table 0 is empty at (ss)). `C`'s term is not a third
derivative: one nucleus's `(a|1/r_C|b)` depends on `A - C` and `B - C`
alone, so its three centre derivatives sum to zero EXACTLY, per nucleus,
and `dC = -(dA + dB)` of that nucleus's own tables -- no `[t + 1_k]`, no
field integral.

THE LOOPS, one thread per contracted shell pair (`kernel_hgp`'s mapping):

    w_r = M[oa+i, ob+j] + M[ob+j, oa+i]     off-diagonal shell pair
        = M[oa+i, oa+j]                     diagonal (off_a == off_b)
    for each nucleus C (serially):
        the three tables zeroed
        for each record of the pair: the seed, the VRR, the accumulation
            acc -= w (Z_C (e|0)^{(0)}), w = 2 alpha, 2 beta or 1
        the three HRRs on contracted values
        fA_k = SUM_r w_r dA_k (r),  fB_k likewise
        gA_k += fA_k,  gB_k += fB_k
        G[C, k] += -(fA_k + fB_k)                         AtomicAdd
    G[atom_a, k] += gA_k,  G[atom_b, k] += gB_k           AtomicAdd

So the record loop runs INSIDE the nucleus loop: the HRR needs contracted
values, and a contracted table is one nucleus's (`hv_` sums the nuclei into
one table before its HRR; the gradient cannot, its `C` row is per nucleus).
The seed is recomputed per nucleus, an `exp` per record per nucleus where
`hv_` takes one per record. The signs are `hv_`'s: `V`'s `-Z_C` enters as
`acc - Z u`. `G` is ATOMIC (every pair on an atom adds into its row) and
ZEROED BY THE CALLER before a build's first gradient launch.

THE PARAMETER BLOCK, `kernel_hgp`'s for `V` plus what a derivative needs
(`kernel_g1e`'s names), the nuclei the atoms (`gv_`'s rule):

    params  n_pair n_rec n_ao n_atom (I32)
            zeta_ab  p_x p_y p_z  pa_x pa_y pa_z  pb_x pb_y pb_z  s_ab
            ea_bra  eb_ket                              [n_rec]  per record
            rec0 rec1 off_a off_b atom_a atom_b  [n_pair]  (I32)  per pair
            ab_x ab_y ab_z  [n_pair]                    A - B
            M[n_ao, n_ao]                               read, never written
            Z[n_atom]  Cnuc[n_atom, 3]
            G[n_atom, 3]  (OUT, read and AtomicAdd-ed)

`s_ab` is `c_a c_b (pi/zeta)^{3/2}` without `K_AB` (`kernel_hgp`'s seed).
The records of pair `n` are `rec0[n] <= k < rec1[n]`, contiguous, in
`gradient.host.records`' order (`gradient.host.pair_records` stages
both blocks).

ROUND-OFF, NOT BITWISE, against the MMD host: another recurrence, another
association, and `C`'s row is a difference of `A`'s and `B`'s terms where
the host forms it from `[t + 1_k]`. The interpreted test states its bound
(test/test_kernel_ghv.py).

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
(its witness), nor the interpreter, nor `reference/`, nor anything of MMD's
(`kernel_g1e`'s weight rule is restated, not imported).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from felizjueves.hgp import hrr
from felizjueves.hgp.recurrence import hrr_body, hrr_name, vrr_body, vrr_name
from felizjueves.hgp.vrr import ZERO, Cart, cartesians
from felizjueves.ir import (
    Assign,
    AtomicAdd,
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
)
from felizjueves.one_electron.kernel_hgp import (
    PAIR,
    acc_name,
    boundary_union,
    factors,
    table_dags,
    vrr_dag,
)
from felizjueves.spec import OPERATOR_OF, Engine, Spec, Target

__all__ = [
    "L_MAX",
    "PAIR",
    "build_ghv_kernel",
    "ghv_name",
    "ghv_spec",
    "launch_shape",
    "pair_arrays",
]

L_MAX = 3  # shells through f: the scope this builder is built and checked for
_AXES = "xyz"
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_REC = "kr"  # the record loop's variable
_NUC = "c"  # the nucleus loop's variable
_BOYS = "boys"
_LOW, _UP_A, _UP_B = 0, 1, 2  # the three tables (module docstring)
_WEIGHT = {_LOW: None, _UP_A: "ea2", _UP_B: "eb2"}

_ATOMIC = (
    "AtomicAdd: every pair on an atom adds into that atom's row of G")
_ZEROED = (
    "PRECONDITION: G is zeroed by the caller before the first gradient "
    "launch of a build; this kernel only adds")
_WEIGHTS = (
    "w_r = M[a_i, b_j] + M[b_j, a_i] off the diagonal, M[a_i, a_j] on a "
    "diagonal shell pair: SUM_mn M_mn X_mn exactly, for any M")


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def ghv_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of `V`'s gradient on `Engine.HGP`."""
    return Spec(OPERATOR_OF[Target.V], 1, (la, lb), Target.V,
                engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else ghv_spec(*cls)
    if (spec.target is not Target.V or spec.derivative_order != 1
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d1 v.hgp key; the MMD V "
                         "gradient is kernel_g1e's gv_")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb = spec.am
    if max(la, lb) > L_MAX:
        raise ValueError(f"{spec.key()}: the HGP V gradient is built for "
                         f"shells through f only (l <= {L_MAX})")
    return la, lb


def ghv_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_ghv_kernel(cls)` gives its kernel, unbuilt."""
    la, lb = _class(cls)
    return f"ghv_{la}{lb}"


def pair_arrays() -> tuple[str, ...]:
    """The per-pair array parameters, in order."""
    return ("rec0", "rec1", "off_a", "off_b", "atom_a", "atom_b",
            "ab_x", "ab_y", "ab_z")


def launch_shape(n_pair: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per shell pair,
    `ceil(n_pair/threads)` blocks. No `WarpReduce`, so a partial last block
    is legal."""
    if type(n_pair) is not int or n_pair < 1:
        raise ValueError(f"n_pair = {n_pair!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_pair // threads), (threads, 1)


# ---------------------------------------------------------------------------
# Small spellings
# ---------------------------------------------------------------------------


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _sum(xs: list[Expr]) -> Expr:
    """`xs` as a left fold of ADDs from its first term."""
    out = xs[0]
    for x in xs[1:]:
        out = _c(Fn.ADD, out, x)
    return out


def _dot(a: Callable[[int], Expr], b: Callable[[int], Expr]) -> Expr:
    return _sum([_c(Fn.MUL, a(d), b(d)) for d in range(3)])


def _shift(v: Cart, d: int, by: int) -> Cart:
    return tuple(c + by * (i == d) for i, c in enumerate(v))  # type: ignore[return-value]


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def _plus(base: str, k: int) -> Expr:
    return _c(Fn.ADD, Var(base), Lit(k))


# ---------------------------------------------------------------------------
# The tables, the weights, the derivative terms
# ---------------------------------------------------------------------------


def _tables(la: int, lb: int) -> dict[int, tuple[hrr.Node, ...]]:
    """The module docstring's three tables' HRR targets (an empty one left
    out)."""
    na, nb = cartesians(la), cartesians(lb)
    out = {
        _LOW: tuple(t for a in na for b in nb for d in range(3) for t in (
            *(((_shift(a, d, -1), b, ZERO, ZERO),) if a[d] else ()),
            *(((a, _shift(b, d, -1), ZERO, ZERO),) if b[d] else ()))),
        _UP_A: tuple((_shift(a, d, 1), b, ZERO, ZERO) for a in na
                     for b in nb for d in range(3)),
        _UP_B: tuple((a, _shift(b, d, 1), ZERO, ZERO) for a in na
                     for b in nb for d in range(3)),
    }
    return {n: tuple(dict.fromkeys(t)) for n, t in out.items() if t}


def _hname(table: int) -> Callable[[hrr.Node], str]:
    return lambda n: f"t{table}{hrr_name(n)}"


def _weights(la: int, lb: int) -> list[Stmt]:
    """`w{r}` per Cartesian pair `r` (`a` outer), from `M` (`_WEIGHTS`;
    `kernel_g1e.weights`' rule, restated)."""
    nb = len(cartesians(lb))
    rows = range(len(cartesians(la)) * nb)

    def m(row: Expr, col: Expr) -> Index:
        return Index("M", (row, col))

    def off(r: int) -> Expr:
        i, j = divmod(r, nb)
        return _c(Fn.ADD, m(_plus("oa", i), _plus("ob", j)),
                  m(_plus("ob", j), _plus("oa", i)))

    if la != lb:  # a diagonal shell pair has la == lb
        return [Note(_WEIGHTS),
                *(Decl(f"w{r}", Scalar.F64, init=off(r)) for r in rows)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    return [
        Note(_WEIGHTS),
        *(Decl(f"w{r}", Scalar.F64, init=Lit(0.0)) for r in rows),
        If(differ,
           tuple(Assign(Var(f"w{r}"), off(r)) for r in rows),
           tuple(Assign(Var(f"w{r}"), m(_plus("oa", r // nb),
                                        _plus("oa", r % nb))) for r in rows)),
    ]


def _derivatives(la: int, lb: int) -> list[Stmt]:
    """`fA{k}`, `fB{k}`: `SUM_r w_r dA_k (r)` and its `B` twin over the three
    tables' HRR targets (module docstring)."""
    na, nb = cartesians(la), cartesians(lb)

    def h(table: int, a: Cart, b: Cart) -> Var:
        return Var(_hname(table)((a, b, ZERO, ZERO)))

    out: list[Stmt] = []
    for side in "AB":
        for k in range(3):
            terms: list[Expr] = []
            for i, a in enumerate(na):
                for j, b in enumerate(nb):
                    if side == "A":
                        d: Expr = h(_UP_A, _shift(a, k, 1), b)
                        n, lo = a[k], (_shift(a, k, -1), b)
                    else:
                        d = h(_UP_B, a, _shift(b, k, 1))
                        n, lo = b[k], (a, _shift(b, k, -1))
                    if n:
                        d = _c(Fn.SUB, d, _c(Fn.MUL, Lit(float(n)),
                                             h(_LOW, *lo)))
                    terms.append(_c(Fn.MUL, Var(f"w{i * len(nb) + j}"), d))
            out.append(Decl(f"f{side}{k}", Scalar.F64, init=_sum(terms)))
    return out


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def build_ghv_kernel(cls: Spec | tuple[int, int], *,
                     name: str | None = None) -> Kernel:
    """The HGP `V` gradient kernel of one class.

    `cls` is a d1 `Target.V` `Spec` on `Engine.HGP`, or `(la, lb)`. See the
    module docstring.
    """
    la, lb = _class(cls)
    hrrs, bounds = table_dags(_tables(la, lb))
    vdag = vrr_dag(boundary_union(bounds), True)
    top = max(n[2] for n in vdag.terms)

    def fundamental(m: int) -> Expr:
        return _c(Fn.MUL, Var("zw"), Index(_BOYS, (Lit(m),)))

    vstmts = vrr_body(vdag, fundamental)
    used = factors(vstmts)

    # --- per record, inside the nucleus loop -------------------------------
    k, c = Var(_REC), Var(_NUC)
    record: list[Stmt] = [Decl("zeta", Scalar.F64,
                               init=Index("zeta_ab", (k,)))]
    if Factor.INV2P.value in used:
        record.append(Decl(Factor.INV2P.value, Scalar.F64,
                           init=_c(Fn.DIV, Lit(0.5), Var("zeta"))))
    if Factor.INV2P_ALPHA_OVER_P.value in used:
        record.append(Note("the point-charge limit of eq 25: rho/p -> 1"))
        record.append(Decl(Factor.INV2P_ALPHA_OVER_P.value, Scalar.F64,
                           init=Var(Factor.INV2P.value)))
    for d in range(3):
        if Factor.pa(d).value in used:
            record.append(Decl(Factor.pa(d).value, Scalar.F64,
                               init=Index(f"pa_{_AXES[d]}", (k,))))
    record += [
        Note("u = (0|0) = s_ab K_AB, K_AB = exp(zeta (P-A).(P-B)): s_ab = "
             "c_a c_b (pi/zeta)^{3/2} is the host's (ao.pairs.dress)"),
        Decl("u", Scalar.F64, init=_c(Fn.MUL, Index("s_ab", (k,)), _c(
            Fn.EXP, _c(Fn.MUL, Var("zeta"), _dot(
                lambda d: Index(f"pa_{_AXES[d]}", (k,)),
                lambda d: Index(f"pb_{_AXES[d]}", (k,))))))),
        Note("the derivative rule's weights 2 alpha, 2 beta: folded in "
             "before the contraction, the exponents being per primitive"),
        Decl("ea2", Scalar.F64,
             init=_c(Fn.MUL, Lit(2.0), Index("ea_bra", (k,)))),
        Decl("eb2", Scalar.F64,
             init=_c(Fn.MUL, Lit(2.0), Index("eb_ket", (k,)))),
        Note("eq 25's point-charge limit: W -> C, so WP = C - P"),
        *(Decl(Factor.wp(d).value, Scalar.F64, init=_c(
            Fn.SUB, Index("Cnuc", (c, Lit(d))),
            Index(f"p_{_AXES[d]}", (k,)))) for d in range(3)),
        Decl(_BOYS, Scalar.F64, (Lit(top + 1),)),
        Boys(_BOYS, top, _c(Fn.MUL, Var("zeta"), _dot(
            lambda d: Var(Factor.wp(d).value),
            lambda d: Var(Factor.wp(d).value)))),
        Note("(0|0)^m = Z s_ab K_AB (2/sqrt(pi)) sqrt(zeta) F_m; the "
             "attractive sign enters as acc - w Z u"),
        Decl("zw", Scalar.F64, init=_c(Fn.MUL, Index("Z", (c,)), _c(
            Fn.MUL, _c(Fn.MUL, Var("u"), Lit(_TWO_OVER_SQRT_PI)),
            _c(Fn.SQRT, Var("zeta"))))),
        *vstmts,
    ]
    for n, bnd in bounds.items():
        for e in bnd:
            x: Expr = Var(vrr_name((e, ZERO, 0)))
            if _WEIGHT[n] is not None:
                x = _c(Fn.MUL, Var(_WEIGHT[n]), x)
            record.append(Assign(Var(acc_name(n, e)), _c(
                Fn.SUB, Var(acc_name(n, e)), x)))

    # --- per nucleus: the tables, the HRRs, the derivative terms -----------
    hstmts: list[Stmt] = []
    for n, dag in hrrs.items():
        hstmts.append(Note({
            _LOW: "table 0: SUM -Z_C (e|0), the a - 1_k and b - 1_k targets",
            _UP_A: "table 1: SUM -Z_C 2 alpha (e|0), the a + 1_k targets",
            _UP_B: "table 2: SUM -Z_C 2 beta (e|0), the b + 1_k targets",
        }[n]))
        hstmts += hrr_body(dag, lambda e, f, n=n: Var(acc_name(n, e)),
                           name=_hname(n))
    used_ab = factors(tuple(hstmts))
    nucleus: list[Stmt] = [
        Note("the contracted tables of nucleus c, summed from 0.0 over the "
             "pair's records"),
        *(Decl(acc_name(n, e), Scalar.F64, init=Lit(0.0))
          for n, bnd in bounds.items() for e in bnd),
        Note("the pair's primitive pairs, serially: the VRR per record "
             "([Fock] eq 25 at f = 0) to la + lb + 1"),
        For(_REC, Var("r0"), Var("r1"), Lit(1), tuple(record)),
        *hstmts,
        Note("SUM_r w_r dA_k (r) and dB_k: 2 alpha (a+1_k|b) - a_k "
             "(a-1_k|b), 2 beta (a|b+1_k) - b_k (a|b-1_k)"),
        *_derivatives(la, lb),
        *(Assign(Var(f"g{s}{d}"), _c(Fn.ADD, Var(f"g{s}{d}"),
                                     Var(f"f{s}{d}")))
          for s in "AB" for d in range(3)),
        Note("Hellmann-Feynman: dC = -(dA + dB) of nucleus c's own tables, "
             "translational invariance, exact per nucleus"),
        Note(_ATOMIC),
        *(AtomicAdd(Index("G", (c, Lit(d))), _c(
            Fn.SUB, Lit(0.0), _c(Fn.ADD, Var(f"fA{d}"), Var(f"fB{d}"))))
          for d in range(3)),
    ]

    # --- per pair ----------------------------------------------------------
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
    head += _weights(la, lb)
    head += [Decl(f"g{s}{d}", Scalar.F64, init=Lit(0.0))
             for s in "AB" for d in range(3)]
    per_pair = (
        *head,
        Note("the nuclei, serially: a contracted table is one nucleus's"),
        For(_NUC, Lit(0), Var("n_atom"), Lit(1), tuple(nucleus)),
        Note(_ATOMIC),
        *(AtomicAdd(Index("G", (Var(atom), Lit(d))), Var(f"g{s}{d}"))
          for s, atom in (("A", "aa"), ("B", "ab")) for d in range(3)),
    )
    body: tuple[Stmt, ...] = (
        Note(f"HGP nuclear-attraction gradient, class (la, lb) = ({la}, "
             f"{lb}): G[X, k] += SUM_mn M_mn dV_mn/dR_Xk, a VRR per "
             "primitive pair, the contraction, the HRR on contracted values "
             "(gradient/kernel_ghv.py)"),
        Note(_ZEROED),
        Note("launch: one thread per contracted shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )

    npair, nr, nao = Var("n_pair"), Var("n_rec"), Var("n_ao")
    nat = Var("n_atom")
    params: list[Param] = [Param("n_pair", Scalar.I32),
                           Param("n_rec", Scalar.I32),
                           Param("n_ao", Scalar.I32),
                           Param("n_atom", Scalar.I32),
                           _f64("zeta_ab", (nr,))]
    params += [_f64(f"{v}_{a}", (nr,)) for v in ("p", "pa", "pb")
               for a in _AXES]
    params += [_f64(n, (nr,)) for n in ("s_ab", "ea_bra", "eb_ket")]
    params += [_i32(n, (npair,)) for n in pair_arrays()[:6]]
    params += [_f64(n, (npair,)) for n in pair_arrays()[6:]]
    params += [_f64("M", (nao, nao)), _f64("Z", (nat,)),
               _f64("Cnuc", (nat, Lit(3))),
               Param("G", Scalar.F64, array=True, intent=Intent.OUT,
                     shape=(nat, Lit(3)))]
    return Kernel(name or ghv_name((la, lb)), tuple(params), body)
