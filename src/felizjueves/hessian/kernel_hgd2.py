"""The one-electron nuclear Hessians by the HGP route: `Target.S`, `T` and
`V` at `derivative_order = 2` on `Engine.HGP`, built as `Kernel`s.

    build_hgd2_kernel(spec | (target, la, lb)) -> Kernel   one per class
    hgd2_name(spec | (target, la, lb))         -> str      hgd2{s,t,v}_{la}{lb}
    hgd2_spec(target, la, lb)                  -> Spec     the d2 HGP key
    tables(target, la, lb)                     -> {weight: HRR targets}
    pair_arrays()                              -> names    per-pair arrays
    launch_shape(n_pair, threads)              -> (grid, block)

The contraction `kernel_h1e`'s `d2s_`, `d2t_`, `d2v_` write
(`hessian.host_1e` is the reference),

    H[X, k, Y, l] += SUM_mn M_mn d2X_mn / dR_{X,k} dR_{Y,l}

on the route of `gradient.kernel_hgs`, `kernel_hgt` and `kernel_ghv`
(`one_electron.kernel_hgp`'s): Obara & Saika's VRR per primitive pair
(`hgp.vrr`'s eq 25 at `f = 0`: the overlap limit for `S` and `T`, the
point-charge limit for `V`), the contraction, then Head-Gordon & Pople's
HRR (`hgp.hrr`'s eq 28) on contracted values. Nothing of MMD's: no `E`, no
Hermite `[t]^(0)`. Shells through f (`L_MAX = 3`), Cartesian. The names are
`hgd2s_`, `hgd2t_`, `hgd2v_{la}{lb}`: not `hs_`, `ht_`, `hv_` (the d0 HGP
one-electron ladder's) nor `d2s_` (MMD's).

THE RULE TWICE. On a primitive Cartesian Gaussian of exponent `alpha` on
`A`, `D_k |a) = 2 alpha |a + 1_k) - a_k |a - 1_k)`, so

    D_l D_k |a) = 4 alpha^2 |a + 1_k + 1_l)
                  - 2 alpha [(a_l + d_kl) |a + 1_k - 1_l) + a_k |a - 1_k + 1_l)]
                  + a_k (a_l - d_kl) |a - 1_k - 1_l)

(`_second`, `_first` composed with itself, like terms merged), and
`D^A_k D^B_l` is the rule once on each index. `alpha` and `beta` are per
PRIMITIVE and the HRR's `AB` is not, so each power is folded into the
boundary before the contraction (`kernel_hgt`'s rule): one contracted
table per weight `(2 alpha)^p (2 beta)^q beta^n` that some term reads
(`tables`), each its own HRR on contracted values, all on ONE VRR per
record cut to the union of their boundaries (`la + lb + 2`; `T`'s ket two
more).

`S` AND `T` BY TRANSLATIONAL INVARIANCE. Each depends on `A - B` alone, so
`dA dB = -dA dA` and `dB dB = dA dA`: the kernel forms the six `h_kl =
SUM_r w_r D^A_k D^A_l (r)` (`h` symmetric) and adds `+h` to the `AA` and
`BB` blocks and `-h` to `AB` and `BA`, 36 `AtomicAdd`s per shell pair (a
one-centre pair adds nothing in the algebra: its four blocks are one).
`T`'s value at `(a', b)` is `kernel_hgt`'s kinetic combination of the
`beta^n` tables at the same derivative power `p`,

    T_p(a', b) = (2 lb + 3) (a'|b)_{p,1} - 2 SUM_d (a'|b + 2_d)_{p,2}
                 - 1/2 SUM_d b_d (b_d - 1) (a'|b - 2_d)_{p,0}

the kinetic operator commuting with a translation of the bra.

`V` PER NUCLEUS, ITS CENTRE BY INVARIANCE. One nucleus's `(a|1/r_C|b)`
depends on `A - C` and `B - C` alone, so its derivatives on `C` are
`dC = -(dA + dB)` EXACTLY, per nucleus. Per nucleus the kernel forms the
21 centre-centre sums `AA_kl`, `BB_kl` (symmetric, six each) and `AB_kl`
(nine) from six tables `(2 alpha)^p (2 beta)^q`, `p + q <= 2`, then

    H[A, k, C, l] += -(AA_kl + AB_kl)          and its transpose
    H[B, k, C, l] += -(AB_lk + BB_kl)          and its transpose
    H[C, k, C, l] += AA_kl + AB_kl + AB_lk + BB_kl

45 `AtomicAdd`s per nucleus, and accumulates `AA`, `BB`, `AB` over the
nuclei into 21 registers that the pair adds once, after the loop, into its
`AA`, `BB`, `AB` and `BA` blocks (36 more). The record loop runs INSIDE the
nucleus loop (`kernel_ghv`'s reason: a contracted table is one nucleus's).
The signs are `hv_`'s: `V`'s `-Z_C` enters as `acc - w Z u`.

THE WEIGHTS `w_r` from `M` are `kernel_g1e.weights`' (`M[a, b] + M[b, a]`
off the diagonal, `M[a, a']` on a diagonal shell pair: exact for any `M`).

THE THREAD MAPPING is `kernel_hgp`'s: ONE THREAD PER CONTRACTED SHELL PAIR,
`pair = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by `pair < n_pair`, its
records `[rec0, rec1)` serially. Every write is an `AtomicAdd` into `H`
(every pair on an atom adds into that atom's blocks), and `H` is ZEROED BY
THE CALLER (`kernel_h1e.ZEROED_H`).

THE PARAMETER BLOCK, the d1 HGP kernels' (`kernel_hgs`, `kernel_hgt`,
`kernel_ghv`), `G` replaced by `H`:

    params  n_pair n_rec n_ao n_atom (I32)
            zeta_ab  [p_x p_y p_z (V)]  pa_x pa_y pa_z  pb_x pb_y pb_z
            s_ab  ea_bra  eb_ket                    [n_rec]  per record
            rec0 rec1 off_a off_b atom_a atom_b  [n_pair]  (I32) per pair
            ab_x ab_y ab_z  [n_pair]                A - B
            M[n_ao, n_ao]                           read, never written
            [Z[n_atom]  Cnuc[n_atom, 3]  (V)]
            H[n_atom, 3, n_atom, 3]  (OUT, read and AtomicAdd-ed)

staged by `gradient.host.pair_records`.

DEAD CODE IS NEVER EMITTED (#92): a table only where a term reads it, a
weight or recurrence scalar only where a statement reads it.

ROUND-OFF, NOT BITWISE, against the host (another engine, another
association, blocks by invariance); test/test_kernel_hgd2.py states the
bound. Unrolled at every class: no low-stack form (none measured).

A builder: vendor-neutral IR. It imports neither `ao/` nor the hosts (its
witnesses), nor the interpreter, nor `reference/`, and of MMD's only
`kernel_g1e`'s weight rule and local pool and `kernel_h1e`'s precondition.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from felizjueves.gradient.kernel_g1e import Locals, weights
from felizjueves.hessian.kernel_h1e import ZEROED_H
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import hrr_body, hrr_name, vrr_body, vrr_name
from felizjueves.hgp.vrr import ZERO, Cart
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
    "TARGETS",
    "build_hgd2_kernel",
    "hgd2_name",
    "hgd2_spec",
    "launch_shape",
    "pair_arrays",
    "tables",
]

L_MAX = 3  # shells through f: the scope this builder is built and checked for
PAIR = "pair"  # the contracted shell pair: one thread each
TARGETS = (Target.S, Target.T, Target.V)
_PREFIX = {Target.S: "hgd2s", Target.T: "hgd2t", Target.V: "hgd2v"}
_AXES = "xyz"
_REC = "kr"  # the record loop's variable
_NUC = "c"  # V's nucleus loop's variable
_BOYS = "boys"
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)

ATOMIC_REASON = (
    "AtomicAdd: every shell pair on an atom adds into that atom's blocks "
    "of H")

# A contracted table's weight `(2 alpha)^p (2 beta)^q beta^n`.
Table = tuple[int, int, int]
# One term of a derivative: `coef * table(a', b')`.
Term = tuple[float, Table, Cart, Cart]

# The upper triangle of a symmetric 3 x 3, row-major.
_SYM = tuple((k, m) for k in range(3) for m in range(k, 3))
_ALL = tuple((k, m) for k in range(3) for m in range(3))


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def hgd2_spec(target: Target, la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of `target`'s Hessian on `Engine.HGP`."""
    return Spec(OPERATOR_OF[target], 2, (la, lb), target, engine=Engine.HGP)


def _class(cls: Spec | tuple[Target, int, int]) -> tuple[Target, int, int]:
    spec = cls if isinstance(cls, Spec) else hgd2_spec(*cls)
    if (spec.target not in TARGETS or spec.derivative_order != 2
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d2 HGP S, T or V key; the "
                         "MMD one-electron Hessians are kernel_h1e's")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the HGP Hessian kernels take no "
                         "levers")
    la, lb = spec.am
    if max(la, lb) > L_MAX:
        raise ValueError(f"{spec.key()}: the HGP one-electron Hessians are "
                         f"built for shells through f only (l <= {L_MAX})")
    return spec.target, la, lb


def hgd2_name(cls: Spec | tuple[Target, int, int]) -> str:
    """The name `build_hgd2_kernel(cls)` gives its kernel, unbuilt."""
    target, la, lb = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}"


def pair_arrays() -> tuple[str, ...]:
    """The per-pair array parameters, in order (every target's)."""
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


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _sum(xs: list[Expr]) -> Expr:
    """`xs` as a left fold of ADDs from its first term."""
    out = xs[0]
    for x in xs[1:]:
        out = _c(Fn.ADD, out, x)
    return out


def _dot(a: Callable[[int], Expr], b: Callable[[int], Expr]) -> Expr:
    return _sum([_mul(a(d), b(d)) for d in range(3)])


def _shift(v: Cart, d: int, by: int) -> Cart:
    return tuple(c + by * (i == d) for i, c in enumerate(v))  # type: ignore[return-value]


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def _digits(v: Cart) -> str:
    return "_".join(str(c) for c in v)


# ---------------------------------------------------------------------------
# The derivative rule, symbolically
# ---------------------------------------------------------------------------


def _first(v: Cart, k: int) -> list[tuple[float, int, Cart]]:
    """`D_k |v) = 2 alpha |v + 1_k) - v_k |v - 1_k)` as `(coef, power of
    2 alpha, v')`, a zero term left out."""
    out = [(1.0, 1, _shift(v, k, 1))]
    if v[k]:
        out.append((-float(v[k]), 0, _shift(v, k, -1)))
    return out


def _second(v: Cart, k: int, m: int) -> list[tuple[float, int, Cart]]:
    """`D_m D_k |v)`: `_first` twice, like terms merged, zeros left out."""
    acc: dict[tuple[int, Cart], float] = {}
    for c1, p1, v1 in _first(v, k):
        for c2, p2, v2 in _first(v1, m):
            key = (p1 + p2, v2)
            acc[key] = acc.get(key, 0.0) + c1 * c2
    return [(c, p, w) for (p, w), c in acc.items() if c != 0.0]


def _terms(block: str, a: Cart, b: Cart, k: int, m: int) -> list[Term]:
    """The terms of `D^X_k D^Y_m (a|b)`, `block` = `XY`."""
    if block == "AA":
        return [(c, (p, 0, 0), a2, b) for c, p, a2 in _second(a, k, m)]
    if block == "BB":
        return [(c, (0, q, 0), a, b2) for c, q, b2 in _second(b, k, m)]
    return [(ca * cb, (p, q, 0), a2, b2) for ca, p, a2 in _first(a, k)
            for cb, q, b2 in _first(b, m)]


def _kets(b: Cart, n: int) -> list[Cart]:
    """The `b'` the kinetic combination reads table `n` at: `b - 2_d`
    (`n = 0`, where `b_d >= 2`), `b` (`n = 1`), `b + 2_d` (`n = 2`)."""
    if n == 1:
        return [b]
    if n == 2:
        return [_shift(b, d, 2) for d in range(3)]
    return [_shift(b, d, -2) for d in range(3) if b[d] >= 2]


def _blocks(target: Target) -> tuple[str, ...]:
    """The centre-centre sums a target's kernel forms (module docstring)."""
    return ("AA", "BB", "AB") if target is Target.V else ("AA",)


def _components(block: str) -> tuple[tuple[int, int], ...]:
    """`AA` and `BB` symmetric (six), `AB` not (nine)."""
    return _ALL if block == "AB" else _SYM


def _sym(block: str, k: int, m: int) -> str:
    """A block component's name, a symmetric block read in its upper
    triangle."""
    if block != "AB" and k > m:
        k, m = m, k
    return f"{block}{k}{m}"


def tables(target: Target, la: int, lb: int
           ) -> dict[Table, tuple[hrr.Node, ...]]:
    """Each contracted table's HRR targets `(a'|b')` (module docstring), an
    empty one left out, sorted by weight."""
    if target not in TARGETS:
        raise ValueError(f"{target} has no HGP one-electron Hessian")
    out: dict[Table, list[hrr.Node]] = {}
    for a in vrr.cartesians(la):
        for b in vrr.cartesians(lb):
            for block in _blocks(target):
                for k, m in _components(block):
                    for _, (p, q, _n), a2, b2 in _terms(block, a, b, k, m):
                        if target is Target.T:
                            for n in range(3):
                                out.setdefault((p, q, n), []).extend(
                                    (a2, b3, ZERO, ZERO)
                                    for b3 in _kets(b2, n))
                        else:
                            out.setdefault((p, q, 0), []).append(
                                (a2, b2, ZERO, ZERO))
    return {t: tuple(dict.fromkeys(out[t])) for t in sorted(out) if out[t]}


def _tname(t: Table) -> str:
    return f"{t[0]}{t[1]}{t[2]}"


def _acc(t: Table, e: Cart) -> str:
    return f"acc{_tname(t)}_{_digits(e)}"


def _hname(t: Table, node: hrr.Node) -> str:
    return f"t{_tname(t)}{hrr_name(node)}"


# The per-record weight factors: a table's weight is the product of
# `ea2` (2 alpha) p times, `eb2` (2 beta) q times and `eb` (beta) n times.
_BASE = {"ea2": ("ea_bra", 2.0), "eb2": ("eb_ket", 2.0),
         "eb": ("eb_ket", 1.0)}


def _factors_of(t: Table) -> list[str]:
    p, q, n = t
    return ["ea2"] * p + ["eb2"] * q + ["eb"] * n


def _weight_name(t: Table) -> str | None:
    fs = _factors_of(t)
    if not fs:
        return None
    return fs[0] if len(fs) == 1 else "x" + _tname(t)


def _weight_decls(ts: list[Table], k: Var) -> list[Stmt]:
    """Each base factor and each product some table reads, once."""
    out: list[Stmt] = []
    bases = {f for t in ts for f in _factors_of(t)}
    for name, (arr, two) in _BASE.items():
        if name in bases:
            v: Expr = Index(arr, (k,))
            out.append(Decl(name, Scalar.F64,
                            init=_mul(Lit(two), v) if two != 1.0 else v))
    for t in ts:
        fs = _factors_of(t)
        if len(fs) > 1:
            x: Expr = Var(fs[0])
            for f in fs[1:]:
                x = _mul(x, Var(f))
            out.append(Decl(_weight_name(t), Scalar.F64, init=x))  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------
# The contracted sums
# ---------------------------------------------------------------------------


def _sums(target: Target, la: int, lb: int, pool: Locals,
          name: Callable[[str, int, int], str]) -> list[Stmt]:
    """Each block's `SUM_r w_r D D (r)` per component into `name(block, k,
    m)` (module docstring); `T`'s kinetic combinations pooled in `pool`."""
    na, nb = vrr.cartesians(la), vrr.cartesians(lb)

    def h(t: Table, a: Cart, b: Cart) -> Var:
        return Var(_hname(t, (a, b, ZERO, ZERO)))

    def value(t: Table, a: Cart, b: Cart) -> Expr:
        if target is not Target.T:
            return h(t, a, b)
        p = t[0]

        def make() -> Expr:  # kernel_hgt's kinetic combination at a'
            v: Expr = _mul(Lit(float(2 * lb + 3)), h((p, 0, 1), a, b))
            v = _c(Fn.SUB, v, _mul(Lit(2.0), _sum(
                [h((p, 0, 2), a, b2) for b2 in _kets(b, 2)])))
            low = [_mul(Lit(float(b[d] * (b[d] - 1))),
                        h((p, 0, 0), a, _shift(b, d, -2))) for d in range(3)
                   if b[d] >= 2]
            if low:
                v = _c(Fn.SUB, v, _mul(Lit(0.5), _sum(low)))
            return v
        return pool.get(f"tk{p}_{_digits(a)}__{_digits(b)}", make)

    out: list[Stmt] = []
    for block in _blocks(target):
        for k, m in _components(block):
            terms: list[Expr] = []
            for i, a in enumerate(na):
                for j, b in enumerate(nb):
                    parts: list[Expr] = []
                    for coef, t, a2, b2 in _terms(block, a, b, k, m):
                        v = value(t, a2, b2)
                        parts.append(v if coef == 1.0 else _mul(Lit(coef), v))
                    terms.append(_mul(Var(f"w{i * len(nb) + j}"),
                                      _sum(parts)))
            out.append(Decl(name(block, k, m), Scalar.F64, init=_sum(terms)))
    return out


def _h(x: Expr, k: int, y: Expr, m: int) -> Index:
    return Index("H", (x, Lit(k), y, Lit(m)))


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def build_hgd2_kernel(cls: Spec | tuple[Target, int, int], *,
                      name: str | None = None) -> Kernel:
    """The HGP one-electron Hessian kernel of one class.

    `cls` is a d2 `Target.S`, `T` or `V` `Spec` on `Engine.HGP`, or
    `(target, la, lb)`. See the module docstring.
    """
    target, la, lb = _class(cls)
    nuclear = target is Target.V
    hrrs, bounds = table_dags(tables(target, la, lb))  # type: ignore[arg-type]
    vdag = vrr_dag(boundary_union(bounds), nuclear)  # type: ignore[arg-type]

    # --- per record: the seed, the VRR, the weighted accumulation ----------
    k = Var(_REC)
    top = max(n[2] for n in vdag.terms)
    if nuclear:
        vstmts = vrr_body(vdag, lambda m: _mul(Var("zw"),
                                               Index(_BOYS, (Lit(m),))))
    else:
        vstmts = vrr_body(vdag, lambda m: Var("u"))
    used = factors(vstmts)
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
        Decl("u", Scalar.F64, init=_mul(Index("s_ab", (k,)), _c(
            Fn.EXP, _mul(Var("zeta"), _dot(
                lambda d: Index(f"pa_{_AXES[d]}", (k,)),
                lambda d: Index(f"pb_{_AXES[d]}", (k,))))))),
        Note("the tables' weights (2 alpha)^p (2 beta)^q beta^n, the rule "
             "applied twice: folded in before the contraction, the "
             "exponents being per primitive"),
        *_weight_decls(list(hrrs), k),  # type: ignore[arg-type]
    ]
    if nuclear:
        c = Var(_NUC)
        record += [
            Note("eq 25's point-charge limit: W -> C, so WP = C - P"),
            *(Decl(Factor.wp(d).value, Scalar.F64, init=_c(
                Fn.SUB, Index("Cnuc", (c, Lit(d))),
                Index(f"p_{_AXES[d]}", (k,)))) for d in range(3)),
            Decl(_BOYS, Scalar.F64, (Lit(top + 1),)),
            Boys(_BOYS, top, _mul(Var("zeta"), _dot(
                lambda d: Var(Factor.wp(d).value),
                lambda d: Var(Factor.wp(d).value)))),
            Note("(0|0)^m = Z s_ab K_AB (2/sqrt(pi)) sqrt(zeta) F_m; the "
                 "attractive sign enters as acc - w Z u"),
            Decl("zw", Scalar.F64, init=_mul(Index("Z", (c,)), _mul(
                _mul(Var("u"), Lit(_TWO_OVER_SQRT_PI)),
                _c(Fn.SQRT, Var("zeta"))))),
        ]
    record += vstmts
    for t, bnd in bounds.items():
        w = _weight_name(t)  # type: ignore[arg-type]
        for e in bnd:
            x: Expr = Var(vrr_name((e, ZERO, 0)))
            if w is not None:
                x = _mul(Var(w), x)
            acc = Var(_acc(t, e))  # type: ignore[arg-type]
            record.append(Assign(acc, _c(Fn.SUB if nuclear else Fn.ADD,
                                         acc, x)))

    # --- the HRRs on contracted values, the sums ---------------------------
    hstmts: list[Stmt] = []
    for t, dag in hrrs.items():
        p, q, n = t  # type: ignore[misc]
        hstmts.append(Note(f"table ({p}, {q}, {n}): SUM "
                           f"{'-Z_C ' if nuclear else ''}c_a c_b "
                           f"(2 alpha)^{p} (2 beta)^{q} beta^{n} (e|0)"))
        hstmts += hrr_body(dag, lambda e, f, t=t: Var(_acc(t, e)),
                           name=lambda node, t=t: _hname(t, node))
    used_ab = factors(tuple(hstmts))
    pool = Locals()
    sums = _sums(target, la, lb, pool,
                 lambda bl, kk, mm: "f" + _sym(bl, kk, mm))

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
    head += weights(la, lb)
    accs: list[Stmt] = [
        Decl(_acc(t, e), Scalar.F64, init=Lit(0.0))  # type: ignore[arg-type]
        for t, bnd in bounds.items() for e in bnd]
    kinetic: list[Stmt] = ([
        Note("T_p(a', b), kernel_hgt's kinetic combination of the beta^n "
             "tables, each declared where first read"), *pool.decls]
        if pool.decls else [])
    aa, ab = Var("aa"), Var("ab")

    def f(bl: str, kk: int, mm: int) -> Var:
        return Var("f" + _sym(bl, kk, mm))

    if not nuclear:
        scatter: list[Stmt] = [Note(
            ATOMIC_REASON + "; dA dB = -dA dA, dB dB = dA dA (translational "
            "invariance)")]
        for x, y, sign in ((aa, aa, 1), (ab, ab, 1), (aa, ab, -1),
                           (ab, aa, -1)):
            for kk, mm in _ALL:
                v: Expr = f("AA", kk, mm)
                scatter.append(AtomicAdd(_h(x, kk, y, mm), v if sign > 0
                                         else _mul(Lit(-1.0), v)))
        per_pair = (
            *head,
            Note("the contracted boundary (e|0) of every table, summed from "
                 "0.0 over the pair's records"),
            *accs,
            Note("the pair's primitive pairs, serially: the VRR per record "
                 "([Fock] eq 25 at f = 0, the overlap limit)"),
            For(_REC, Var("r0"), Var("r1"), Lit(1), tuple(record)),
            *hstmts,
            *kinetic,
            Note("fAA_km = SUM_r w_r D^A_k D^A_m (r), symmetric"),
            *sums,
            *scatter,
        )
    else:
        c = Var(_NUC)

        def g(bl: str, kk: int, mm: int) -> Var:
            return Var("g" + _sym(bl, kk, mm))

        comps = [(bl, kk, mm) for bl in _blocks(target)
                 for kk, mm in _components(bl)]
        nucleus: list[Stmt] = [
            Note("the contracted tables of nucleus c, summed from 0.0 over "
                 "the pair's records"),
            *accs,
            Note("the pair's primitive pairs, serially: the VRR per record "
                 "([Fock] eq 25 at f = 0, the point-charge limit)"),
            For(_REC, Var("r0"), Var("r1"), Lit(1), tuple(record)),
            *hstmts,
            Note("nucleus c's centre-centre sums: AA, BB symmetric, AB not"),
            *sums,
            *(Assign(g(*u), _c(Fn.ADD, g(*u), f(*u))) for u in comps),
            Note("dC = -(dA + dB) of nucleus c's own tables, translational "
                 "invariance, exact per nucleus"),
            Note(ATOMIC_REASON),
        ]
        for kk, mm in _ALL:
            xac = _c(Fn.SUB, Lit(0.0), _c(Fn.ADD, f("AA", kk, mm),
                                          f("AB", kk, mm)))
            xbc = _c(Fn.SUB, Lit(0.0), _c(Fn.ADD, f("AB", mm, kk),
                                          f("BB", kk, mm)))
            nucleus += [
                AtomicAdd(_h(aa, kk, c, mm), xac),
                AtomicAdd(_h(c, mm, aa, kk), xac),
                AtomicAdd(_h(ab, kk, c, mm), xbc),
                AtomicAdd(_h(c, mm, ab, kk), xbc),
            ]
        for kk, mm in _ALL:
            nucleus.append(AtomicAdd(_h(c, kk, c, mm), _sum([
                f("AA", kk, mm), f("AB", kk, mm), f("AB", mm, kk),
                f("BB", kk, mm)])))
        tail: list[Stmt] = [Note(ATOMIC_REASON)]
        for kk, mm in _ALL:
            tail += [
                AtomicAdd(_h(aa, kk, aa, mm), g("AA", kk, mm)),
                AtomicAdd(_h(ab, kk, ab, mm), g("BB", kk, mm)),
                AtomicAdd(_h(aa, kk, ab, mm), g("AB", kk, mm)),
                AtomicAdd(_h(ab, mm, aa, kk), g("AB", kk, mm)),
            ]
        per_pair = (
            *head,
            Note("the centre-centre sums over every nucleus"),
            *(Decl(g(*u).name, Scalar.F64, init=Lit(0.0)) for u in comps),
            Note("the nuclei, serially: a contracted table is one nucleus's"),
            For(_NUC, Lit(0), Var("n_atom"), Lit(1), tuple(nucleus)),
            *tail,
        )

    what = {Target.S: "overlap", Target.T: "kinetic energy",
            Target.V: "nuclear attraction"}[target]
    body: tuple[Stmt, ...] = (
        Note(f"HGP Hessian of the {what}, class (la, lb) = ({la}, {lb}): "
             "H[X, k, Y, l] += SUM_mn M_mn d2X_mn/dR_{X,k} dR_{Y,l} "
             "(hessian/kernel_hgd2.py)"),
        Note(ZEROED_H),
        Note("launch: one thread per contracted shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _mul(
            Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )

    npair, nr, nao, nat = (Var("n_pair"), Var("n_rec"), Var("n_ao"),
                           Var("n_atom"))
    params: list[Param] = [
        Param("n_pair", Scalar.I32), Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32), Param("n_atom", Scalar.I32),
        _f64("zeta_ab", (nr,)),
    ]
    if nuclear:
        params += [_f64(f"p_{a}", (nr,)) for a in _AXES]
    params += [_f64(f"{v}_{a}", (nr,)) for v in ("pa", "pb") for a in _AXES]
    params += [_f64(n, (nr,)) for n in ("s_ab", "ea_bra", "eb_ket")]
    params += [_i32(n, (npair,)) for n in pair_arrays()[:6]]
    params += [_f64(n, (npair,)) for n in pair_arrays()[6:]]
    params.append(_f64("M", (nao, nao)))
    if nuclear:
        params += [_f64("Z", (nat,)), _f64("Cnuc", (nat, Lit(3)))]
    params.append(Param("H", Scalar.F64, array=True, intent=Intent.OUT,
                        shape=(nat, Lit(3), nat, Lit(3))))
    return Kernel(name or hgd2_name((target, la, lb)), tuple(params), body)
