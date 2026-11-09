"""The multipole Hessians by the HGP route: `Target.DIPOLE` and
`Target.QUADRUPOLE` at `derivative_order = 2` on `Engine.HGP`, built as
`Kernel`s.

    build_hd2mu_kernel(spec | (target, la, lb)) -> Kernel  one per class
    hd2mu_name(spec | (target, la, lb))   -> str   hd2mu_ / hd2quad_{la}{lb}
    hd2mu_spec(target, la, lb)            -> Spec  the class's d2 HGP key
    n_components(target)                  -> 3 or 6
    pair_arrays()                         -> names its per-pair arrays
    launch_shape(n_pair, threads)         -> (grid, block)

The same contraction as `kernel_d2mu`'s `d2mu_` / `d2quad_`
(`hessian.host_1e.multipole_hessian` is the reference), per component `c`
of the moment `<a| q (r - C)^(ex, ey, ez) |b>` -- x y z for the dipole,
the raw second moments xx xy xz yy yz zz for the quadrupole,
`kernel_mu`'s order --

    H[c, A, k, B, l] += SUM_mn M_mn d2MU^c_mn / dR_{A,k} dR_{B,l}

on the route of `gradient.kernel_hgmu` / `kernel_hgquad`: Obara & Saika's
overlap VRR per primitive pair, the contraction, Head-Gordon & Pople's
HRR on contracted values, and the operator moved onto the ket, each
`(r - C)_d = (r - B)_d + BC_d`, `BC = B - C`, so

    MU_d(a, b)  = (a|b+1_d) + BC_d (a|b)
    Q_de(a, b)  = (a|b+1_d+1_e) + BC_e (a|b+1_d) + BC_d (a|b+1_e)
                  + BC_d BC_e (a|b),

every `(.|.)` an overlap (`one_electron.kernel_hgp`'s docstring).

THE DERIVATIVE RULE acts on the basis function, twice, not on the
operator: `C` is a FIXED origin, not a nucleus, so there are no operator
terms and the blocks are centre-centre only (`AA`, `BB`, `AB`), as
`d2mu_`'s. The derivative is taken on the Gaussian first and the rewriting
above (exact at every `B`) applied to each of its terms afterwards, so
`BC` is never differentiated:

    dA_k X(a, b) = 2 alpha X(a+1_k, b) - a_k X(a-1_k, b)     dB_k likewise

applied twice gives, per block, terms `coef (2 alpha)^p (2 beta)^n X(a',
b')` with `p + n = 0, 1, 2`. `alpha` and `beta` are per PRIMITIVE, so --
as `kernel_hgs` does at d1 -- each weight is folded into the boundary
before the contraction: SIX contracted tables, each with its own HRR,

    table 0   SUM c_a c_b               (e|0)
    table 1   SUM c_a c_b 2 alpha       (e|0)
    table 2   SUM c_a c_b 2 beta        (e|0)
    table 3   SUM c_a c_b 4 alpha^2     (e|0)
    table 4   SUM c_a c_b 4 beta^2      (e|0)
    table 5   SUM c_a c_b 4 alpha beta  (e|0)

(a table no term reads is left out: table 0 at (ss)), one VRR to `la + lb
+ e + 2` per record feeding all of them. Then, per Cartesian pair `r =
(a, b)` of the block, its weight `w_r` from `M` (`kernel_g1e.weights`),

    hXY_kl^c = SUM_r w_r SUM_terms coef X^c_table(a', b')

each `X^c_n(a', b')` a straight-line local declared where first read
(`kernel_g1e.Locals`), and `H[c, A, k, A, l] += q hAA_kl^c` (`k <= l`
formed, mirrored), `BB` likewise, `H[c, A, k, B, l] += q hAB_kl^c` and
`H[c, B, l, A, k]` the same value: `36 n_comp` `AtomicAdd`s per SHELL
PAIR. Both centres are computed; translational invariance is not used.

THE THREAD MAPPING is `kernel_hgquad`'s: ONE THREAD PER CONTRACTED SHELL
PAIR, `pair = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by `pair <
n_pair`, the pair's records `[rec0, rec1)` serially, the boundary
accumulated in registers from `0.0`, the HRRs once. Every pair on an atom
adds into that atom's blocks, so the adds are atomic (`ATOMIC_REASON`) and
`H` is ZEROED BY THE CALLER (`kernel_h1e.ZEROED_H`).

THE PARAMETER BLOCK is `kernel_hgquad`'s, with the output
`H[n_comp, n_atom, 3, n_atom, 3]` in place of `G[n_comp, n_atom, 3]`:

    params  n_pair n_rec n_ao n_atom (I32)
            zeta_ab pa_x pa_y pa_z pb_x pb_y pb_z s_ab ea_bra eb_ket
                                                           [n_rec] per record
            rec0 rec1 off_a off_b atom_a atom_b  [n_pair]  (I32) per pair
            ab_x ab_y ab_z  b_x b_y b_z          [n_pair]  A - B, and B
            M[n_ao, n_ao]                                  read, any M
            C_x C_y C_z q   (F64 scalars: the origin and the charge, RUNTIME)
            H[n_comp, n_atom, 3, n_atom, 3]  (OUT, read and AtomicAdd-ed)

Staged by `gradient.host.pair_records(..., ket_centre=True)`. The origin
and the charge are runtime scalars, as at d0 and d1: neither is a key
field.

SCOPE: shells through f (`L_MAX = 3`), `d2mu_`'s and `hgquad_`'s
validated scope.

ROUND-OFF, NOT BITWISE, against the host: another recurrence, another
association. test/test_kernel_hd2mu.py states its bound.

A builder: vendor-neutral IR. It imports neither `ao/` nor
`hessian.host_1e` (its witness), nor the interpreter, nor `reference/`,
nor the MMD Hessian `kernel_d2mu`.
"""

from __future__ import annotations

from itertools import combinations

from felizjueves.gradient.kernel_g1e import Locals, weights
from felizjueves.hessian.kernel_h1e import COMPONENTS, ZEROED_H
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
    "HESSIAN_MOMENTS",
    "L_MAX",
    "PAIR",
    "ZEROED_H",
    "build_hd2mu_kernel",
    "hd2mu_name",
    "hd2mu_spec",
    "launch_shape",
    "n_components",
    "pair_arrays",
    "second",
]

L_MAX = 3  # through f: d2mu_'s and hgquad_'s validated scope
PAIR = "pair"  # the contracted shell pair: one thread each
HESSIAN_MOMENTS = (Target.DIPOLE, Target.QUADRUPOLE)
_ORDER = {Target.DIPOLE: 1, Target.QUADRUPOLE: 2}
_PREFIX = {Target.DIPOLE: "hd2mu", Target.QUADRUPOLE: "hd2quad"}
_AXES = "xyz"
_REC = "kr"  # the record loop's variable, kernel_hgp's
# Each table by its weight's powers (p, n) of (2 alpha, 2 beta), and the
# per-record local that holds the weight (module docstring).
_TABLE = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (2, 0): 3, (0, 2): 4, (1, 1): 5}
_WEIGHT = {0: None, 1: "ea2", 2: "eb2", 3: "eaa", 4: "ebb", 5: "eab"}
_WHAT = {0: "", 1: " 2 alpha", 2: " 2 beta", 3: " 4 alpha^2",
         4: " 4 beta^2", 5: " 4 alpha beta"}

ATOMIC_REASON = (
    "AtomicAdd: every shell pair on an atom adds into that atom's blocks "
    "of every component of H")


def _components(target: Target) -> tuple[tuple[int, ...], ...]:
    """Each component as the axes of its `(r - C)` factors, `cartesians(e)`
    order: x y z, or xx xy xz yy yz zz."""
    return tuple(tuple(d for d in range(3) for _ in range(c[d]))
                 for c in vrr.cartesians(_ORDER[target]))


def n_components(target: Target) -> int:
    """3 for the dipole, 6 for the quadrupole."""
    return len(_components(target))


def hd2mu_spec(target: Target, la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of the dipole or quadrupole at d2 on
    `Engine.HGP`."""
    return Spec(OPERATOR_OF[target], 2, (la, lb), target, engine=Engine.HGP)


def _class(cls: Spec | tuple[Target, int, int]) -> tuple[Target, int, int]:
    spec = cls if isinstance(cls, Spec) else hd2mu_spec(*cls)
    if (spec.target not in HESSIAN_MOMENTS or spec.derivative_order != 2
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d2 HGP dipole or quadrupole "
                         "key; the MMD moment Hessian is kernel_d2mu's")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the hessian kernels take no levers")
    la, lb = spec.am
    if max(la, lb) > L_MAX:
        raise ValueError(f"{spec.key()}: the HGP moment Hessian is built "
                         f"for shells through f only (l <= {L_MAX})")
    return spec.target, la, lb


def hd2mu_name(cls: Spec | tuple[Target, int, int]) -> str:
    """The name `build_hd2mu_kernel(cls)` gives its kernel, unbuilt:
    `hd2mu_{la}{lb}` (the dipole) or `hd2quad_{la}{lb}` (the quadrupole)."""
    target, la, lb = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}"


def pair_arrays() -> tuple[str, ...]:
    """The per-pair array parameters, in order: `kernel_hgquad`'s."""
    return ("rec0", "rec1", "off_a", "off_b", "atom_a", "atom_b",
            "ab_x", "ab_y", "ab_z", "b_x", "b_y", "b_z")


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


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


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


# A term of a derivative: (coefficient, table, a', b').
Term = tuple[int, int, Cart, Cart]


def _d(side: str, k: int,
       terms: list[tuple[int, int, int, Cart, Cart]]
       ) -> list[tuple[int, int, int, Cart, Cart]]:
    """The derivative rule on `side`'s centre, axis `k`, of terms `(coef,
    p, n, a', b')` (`p`, `n` the powers of `2 alpha`, `2 beta`)."""
    out = []
    for coef, p, n, a, b in terms:
        if side == "A":
            out.append((coef, p + 1, n, _shift(a, k, 1), b))
            if a[k]:
                out.append((-coef * a[k], p, n, _shift(a, k, -1), b))
        else:
            out.append((coef, p, n + 1, a, _shift(b, k, 1)))
            if b[k]:
                out.append((-coef * b[k], p, n, a, _shift(b, k, -1)))
    return out


def second(x: str, k: int, y: str, l: int, a: Cart, b: Cart  # noqa: E741
           ) -> list[Term]:
    """`d2/dX_k dY_l` of `X(a, b)` as terms `(coef, table, a', b')`, like
    terms merged, in first-reached order (the leading term's `coef` is 1)."""
    acc: dict[tuple[int, Cart, Cart], int] = {}
    for coef, p, n, a2, b2 in _d(y, l, _d(x, k, [(1, 0, 0, a, b)])):
        key = (_TABLE[(p, n)], a2, b2)
        acc[key] = acc.get(key, 0) + coef
    return [(c, t, a2, b2) for (t, a2, b2), c in acc.items() if c]


def _subsets(axes: tuple[int, ...]) -> list[tuple[int, ...]]:
    """The positions of `axes` taken as ket shifts, largest first: the
    other positions are `BC` factors (module docstring's term order)."""
    n = len(axes)
    return [s for r in range(n, -1, -1) for s in combinations(range(n), r)]


def _reads(a: Cart, b: Cart, axes: tuple[int, ...]) -> tuple[hrr.Node, ...]:
    """The overlaps `X^c(a, b)` reads, in its terms' order."""
    out = []
    for s in _subsets(axes):
        bs = b
        for i in s:
            bs = _shift(bs, axes[i], 1)
        out.append((a, bs, ZERO, ZERO))
    return tuple(out)


def _terms(la: int, lb: int) -> dict[tuple[str, int, str, int, int],
                                     list[Term]]:
    """Every block component's terms per row `r`, keyed `(x, k, y, l, r)`."""
    return {(*comp, r): second(*comp, a, b)
            for comp in COMPONENTS
            for r, (a, b) in enumerate(_rows(la, lb))}


def _tables(target: Target, la: int, lb: int,
            terms: dict[tuple[str, int, str, int, int], list[Term]]
            ) -> dict[int, tuple[hrr.Node, ...]]:
    """Each table's HRR targets: every overlap its `X`s read; an empty
    table left out."""
    at: dict[int, list[tuple[Cart, Cart]]] = {n: [] for n in _WEIGHT}
    for ts in terms.values():
        for _, t, a, b in ts:
            at[t].append((a, b))
    comps = _components(target)
    return {n: tuple(dict.fromkeys(
        node for a, b in dict.fromkeys(ps) for axes in comps
        for node in _reads(a, b, axes)))
        for n, ps in at.items() if ps}


def _hname(table: int, node: hrr.Node) -> str:
    return f"t{table}{hrr_name(node)}"


def _xname(table: int, c: int, a: Cart, b: Cart) -> str:
    return f"X{table}c{c}_" + "_".join(str(x) for x in (*a, *b))


def _h(c: int, x: str, k: int, y: str, l: int) -> str:  # noqa: E741
    return f"h{c}{x}{y}_{k}{l}"


def _values(target: Target, la: int, lb: int,
            terms: dict[tuple[str, int, str, int, int], list[Term]],
            pool: Locals) -> list[Stmt]:
    """`hXY_kl^c` as one fold each over the block's rows, the `X`s they
    read pooled (module docstring)."""
    comps = _components(target)

    def X(table: int, c: int, a: Cart, b: Cart) -> Var:
        axes = comps[c]

        def make() -> Expr:
            parts: list[Expr] = []
            for s, node in zip(_subsets(axes), _reads(a, b, axes),
                               strict=True):
                v: Expr = Var(_hname(table, node))
                for i in range(len(axes)):
                    if i not in s:
                        v = _mul(Var(f"BC_{_AXES[axes[i]]}"), v)
                parts.append(v)
            return _fold(parts)
        return pool.get(_xname(table, c, a, b), make)

    out: list[Stmt] = []
    for c in range(len(comps)):
        for comp in COMPONENTS:
            rows: list[Expr] = []
            for r in range(len(_rows(la, lb))):
                v: Expr | None = None
                for coef, t, a, b in terms[(*comp, r)]:
                    x: Expr = X(t, c, a, b)
                    if abs(coef) != 1:
                        x = _mul(Lit(float(abs(coef))), x)
                    if v is None:
                        v = x if coef > 0 else _mul(Lit(-1.0), x)
                    else:
                        v = _c(Fn.ADD if coef > 0 else Fn.SUB, v, x)
                assert v is not None
                rows.append(_mul(Var(f"w{r}"), v))
            out.append(Decl(_h(c, *comp), Scalar.F64, init=_fold(rows)))
    return out


def build_hd2mu_kernel(cls: Spec | tuple[Target, int, int], *,
                       name: str | None = None) -> Kernel:
    """The HGP Hessian kernel of one dipole or quadrupole pair class.

    `cls` is a d2 HGP `Target.DIPOLE` / `QUADRUPOLE` `Spec`, or `(target,
    la, lb)`. See the module docstring.
    """
    target, la, lb = _class(cls)
    ncomp = n_components(target)
    terms = _terms(la, lb)
    hrrs, bounds = table_dags(_tables(target, la, lb, terms))
    vdag = vrr_dag(boundary_union(bounds), False)
    k = Var(_REC)

    vstmts = vrr_body(vdag, lambda m: Var("u"))
    used = factors(vstmts)
    weighted = {_WEIGHT[n] for n in bounds if _WEIGHT[n] is not None}
    accumulate: list[Stmt] = []
    for n, bnd in bounds.items():
        for e in bnd:
            x: Expr = Var(vrr_name((e, ZERO, 0)))
            if _WEIGHT[n] is not None:
                x = _mul(Var(_WEIGHT[n]), x)
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
        Decl("u", Scalar.F64, init=_mul(Index("s_ab", (k,)), _c(
            Fn.EXP, _mul(Var("zeta"), _fold([
                _mul(Index(f"pa_{a}", (k,)), Index(f"pb_{a}", (k,)))
                for a in _AXES]))))),
        Note("the tables' weights, the derivative rule's twice, per "
             "primitive: 2 alpha, 2 beta, 4 alpha^2, 4 beta^2, 4 alpha beta"),
        Decl("ea2", Scalar.F64, init=_mul(Lit(2.0), Index("ea_bra", (k,)))),
        Decl("eb2", Scalar.F64, init=_mul(Lit(2.0), Index("eb_ket", (k,)))),
    ]
    for w, (p, q) in (("eaa", ("ea2", "ea2")), ("ebb", ("eb2", "eb2")),
                      ("eab", ("ea2", "eb2"))):
        if w in weighted:
            record.append(Decl(w, Scalar.F64, init=_mul(Var(p), Var(q))))
    record += [*vstmts, *accumulate]

    hstmts: list[Stmt] = []
    for n, dag in hrrs.items():
        hstmts.append(Note(f"table {n}: SUM c_a c_b{_WHAT[n]} (e|0)"))
        hstmts += hrr_body(dag, lambda e, f, n=n: Var(acc_name(n, e)),
                           name=lambda node, n=n: _hname(n, node))
    used_ab = factors(tuple(hstmts))

    pool = Locals()
    values = _values(target, la, lb, terms, pool)
    bc = sorted({d for axes in _components(target) for d in axes})

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
    head.append(Note("BC = B - C: the operator (r - C) moved onto the ket"))
    head += [Decl(f"BC_{_AXES[d]}", Scalar.F64, init=_c(
        Fn.SUB, Index(f"b_{_AXES[d]}", (pair,)), Var(f"C_{_AXES[d]}")))
        for d in bc]
    head.append(Note("the contracted boundary (e|0) of each table, summed "
                     "from 0.0 over the pair's records"))
    for n, bnd in bounds.items():
        head += [Decl(acc_name(n, e), Scalar.F64, init=Lit(0.0)) for e in bnd]

    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    atom = {"A": Var("aa"), "B": Var("ab")}
    for c in range(ncomp):
        for x, kk, y, l in COMPONENTS:  # noqa: E741
            v = _mul(Var("q"), Var(_h(c, x, kk, y, l)))
            scatter.append(AtomicAdd(Index("H", (
                Lit(c), atom[x], Lit(kk), atom[y], Lit(l))), v))
            if x != y or kk != l:  # the mirror: (B, l, A, k), or (X, l, X, k)
                scatter.append(AtomicAdd(Index("H", (
                    Lit(c), atom[y], Lit(l), atom[x], Lit(kk))), v))

    e = _ORDER[target]
    what = ("MU_d(a, b) = (a|b+1_d) + BC_d (a|b)" if e == 1 else
            "Q_de(a, b) = (a|b+1_d+1_e) + BC_e (a|b+1_d) + BC_d (a|b+1_e) "
            "+ BC_d BC_e (a|b)")
    per_pair = (
        *head,
        Note("the pair's primitive pairs, serially: the VRR per record "
             f"([Fock] eq 25 at f = 0, the overlap limit) to la + lb + {e + 2}"),
        For(_REC, Var("r0"), Var("r1"), Lit(1), tuple(record)),
        *hstmts,
        *weights(la, lb),
        Note(f"{what} per table, each declared where first read"),
        *pool.decls,
        Note("d2/dX_k dY_l by dA_k X(a, b) = 2 alpha X(a+1_k, b) - a_k "
             "X(a-1_k, b), dB_k likewise, twice; C is fixed, so no "
             "operator term"),
        *values,
        *scatter,
    )
    title = ("dipole, mu = <a| q (r - C) |b>" if e == 1
             else "quadrupole (raw second moments), <a| q (r - C)_d "
                  "(r - C)_e |b>, xx xy xz yy yz zz")
    body: tuple[Stmt, ...] = (
        Note(f"HGP hessian of the {title}, class (la, lb) = ({la}, {lb}): "
             "H[c, A, k, B, l] += SUM_mn M_mn d2MU^c_mn/dR_{A,k}dR_{B,l} "
             "(hessian/kernel_hd2mu.py); the origin C is fixed, not a "
             "nucleus"),
        Note(ZEROED_H),
        Note("launch: one thread per contracted shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
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
        *(Param(f"C_{a}", Scalar.F64) for a in _AXES),
        Param("q", Scalar.F64),
        Param("H", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(ncomp), nat, Lit(3), nat, Lit(3))),
    ]
    return Kernel(name or hd2mu_name((target, la, lb)), tuple(params), body)
