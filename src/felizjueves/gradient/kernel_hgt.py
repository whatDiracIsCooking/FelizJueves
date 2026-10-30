"""The kinetic-energy gradient by the HGP route: `Target.T` at
`derivative_order = 1` on `Engine.HGP`, built as a `Kernel`.

    build_hgt_kernel(spec | (la, lb)) -> Kernel   one per class
    hgt_name(spec | (la, lb))         -> str      hgt_{la}{lb}, unbuilt
    hgt_spec(la, lb)                  -> Spec     kinetic/d1/(la,lb)/b1/t.hgp
    launch_shape(n_pair, threads)     -> (grid, block)

The contraction `gradient.kernel_g1e`'s `gt_` writes,

    G[A, k] += SUM_mn M_mn dT_mn / dR_{A,k}

(`felizjueves.gradient`'s package docstring; `gradient.host.
kinetic_gradient` is the reference), on the second engine: the HGP
one-electron route of `one_electron.kernel_hgp` (`spec.py`, "AN HGP
ONE-ELECTRON CLASS"), whose VRR, HRR and kinetic combination it reuses.
Shells through f (`L_MAX = 3`), Cartesian.

THE DERIVATIVE IS FOLDED INTO THE CONTRACTED TABLES. Per primitive,
`d/dA_k` of the bra Gaussian is `2 alpha (a + 1_k| - a_k (a - 1_k|`, and
the kinetic operator commutes with a translation, so per primitive

    dT_ab / dA_k = 2 alpha T(a + 1_k, b) - a_k T(a - 1_k, b)

with `T(a', b)` `kernel_hgp`'s kinetic combination of overlap-type
integrals at the same `b`,

    T(a', b) = (2 lb + 3) (a'|b)_1 - 2 SUM_d (a'|b + 2_d)_2
               - 1/2 SUM_d b_d (b_d - 1) (a'|b - 2_d)_0

`(.|.)_n` weighted by `beta^n`. `alpha` and `beta` are per primitive, so
both weights are folded in BEFORE the contraction: six contracted tables
`(p, n)`, weighted `(2 alpha)^p beta^n` -- `p = 1` read at `a + 1_k`
(`la + 1`), `p = 0` at `a - 1_k` (`la - 1`, absent for an s bra), `n = 0`
absent while `lb <= 1` (`b_d <= 1`) -- each its own HRR on contracted
values (`hgp.hrr`'s eq 28), as `kernel_hgp`'s three `T` tables are. The
VRR (`hgp.vrr`'s eq 25 at `f = 0`, the overlap limit) runs once per
record to `la + lb + 3`, cut to the union of the tables' boundaries.

THE KET CENTRE BY TRANSLATIONAL INVARIANCE. `T_ab` depends on `A - B`
alone (no third centre), so `dT/dB_k = -dT/dA_k`, and the kernel adds
`-gA_k` to `B`'s row: exact in the algebra (the host differentiates `B`
on its own), so a one-centre pair adds `g` and `-g` to one atom.

    gA_k = SUM_r w_r [T_{(1)}(a_r + 1_k, b_r) - a_{r,k} T_{(0)}(a_r - 1_k, b_r)]
    w_r  = M[oa+i, ob+j] + M[ob+j, oa+i]    off-diagonal shell pair
         = M[oa+i, oa+j]                    diagonal (off_a == off_b)

(`kernel_g1e.weights`' rule, exact for any `M`), then
`G[atom_a, k] += gA_k` and `G[atom_b, k] += -1 * gA_k`, six `AtomicAdd`s
per shell pair: every pair on an atom adds into its row. `G` is ZEROED BY
THE CALLER before a build's first gradient launch.

THE THREAD MAPPING is `kernel_hgp`'s: ONE THREAD PER CONTRACTED SHELL PAIR,
`pair = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by `pair < n_pair`, its
records `[rec0, rec1)` run serially, the boundary accumulated in registers
from `0.0`, the HRR once on contracted values.

THE PARAMETER BLOCK, `kernel_hgp`'s `T` block with `ea_bra` (the bra
exponent) and the atoms in place of the output matrix:

    params  n_pair n_rec n_ao n_atom (I32)
            zeta_ab  pa_x pa_y pa_z  pb_x pb_y pb_z  s_ab  ea_bra  eb_ket
                                                         [n_rec] per record
            rec0 rec1 off_a off_b atom_a atom_b [n_pair] (I32) per pair
            ab_x ab_y ab_z  [n_pair]                     A - B
            M[n_ao, n_ao]                                the contracted matrix
            G[n_atom, 3]  (OUT, read and AtomicAdd-ed)

staged by `gradient.host.pair_records`.

DEAD CODE IS NEVER EMITTED (#92): a table only where a row reads it, a
record weight or recurrence scalar only where a term reads it.

ROUND-OFF, NOT BITWISE, against the host (another engine, another
association, `dB` by invariance); test/test_kernel_hgt.py states the bound.

A builder: vendor-neutral IR. It imports neither `ao/` nor
`gradient.host` (its witness), nor the interpreter, nor `reference/`.
"""

from __future__ import annotations

from felizjueves.gradient.kernel_g1e import ATOMIC_REASON, ZEROED_G, Locals, weights
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
    _boundary,
    _factors,
    _hrr_dag,
    _vrr_dag,
    boundary_union,
)
from felizjueves.spec import OPERATOR_OF, Engine, Spec, Target

__all__ = [
    "L_MAX",
    "PAIR",
    "build_hgt_kernel",
    "hgt_name",
    "hgt_spec",
    "launch_shape",
    "pair_arrays",
    "tables",
]

L_MAX = 3  # shells through f: the scope this builder is built and checked for
PAIR = "pair"  # the contracted shell pair: one thread each
_AXES = "xyz"
_REC = "kr"  # the record loop's variable

Table = tuple[int, int]  # (p, n): weight (2 alpha)^p beta^n

# each table's per-record weight, by name; None is unweighted
_WEIGHT: dict[Table, str | None] = {
    (0, 0): None, (0, 1): "eb", (0, 2): "ebb",
    (1, 0): "ea2", (1, 1): "ea2eb", (1, 2): "ea2ebb",
}


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def hgt_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of `T` at derivative order 1 on HGP."""
    return Spec(OPERATOR_OF[Target.T], 1, (la, lb), Target.T,
                engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else hgt_spec(*cls)
    if (spec.target is not Target.T or spec.derivative_order != 1
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not an HGP d1 T key; the MMD "
                         "kinetic gradient is kernel_g1e's gt_")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the HGP gradient kernels take no "
                         "levers")
    la, lb = spec.am
    if max(la, lb) > L_MAX:
        raise ValueError(f"{spec.key()}: the HGP kinetic gradient is built "
                         f"for shells through f only (l <= {L_MAX})")
    return la, lb


def hgt_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_hgt_kernel(cls)` gives its kernel: `hgt_{la}{lb}`."""
    la, lb = _class(cls)
    return f"hgt_{la}{lb}"


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
# The tables
# ---------------------------------------------------------------------------


def _shift(v: Cart, d: int, by: int) -> Cart:
    return tuple(c + by * (i == d) for i, c in enumerate(v))  # type: ignore[return-value]


def _bras(la: int, p: int) -> tuple[Cart, ...]:
    """The `a'` table `p` is read at: `a + 1_k` (`p = 1`), `a - 1_k`."""
    by = 1 if p else -1
    out = [_shift(a, k, by) for a in vrr.cartesians(la) for k in range(3)
           if p or a[k] > 0]
    return tuple(dict.fromkeys(out))


def _kets(lb: int, n: int) -> tuple[Cart, ...]:
    """The `b'` table `n` is read at: `b - 2_d`, `b`, `b + 2_d`."""
    nb = vrr.cartesians(lb)
    if n == 1:
        return tuple(nb)
    by = 2 if n == 2 else -2
    out = [_shift(b, d, by) for b in nb for d in range(3)
           if n == 2 or b[d] >= 2]
    return tuple(dict.fromkeys(out))


def tables(la: int, lb: int) -> dict[Table, tuple[hrr.Node, ...]]:
    """Each contracted table's HRR targets `(a'|b')` (module docstring), an
    empty one left out, in `_WEIGHT` order."""
    out = {}
    for t in _WEIGHT:
        p, n = t
        tg = tuple((a, b, ZERO, ZERO) for a in _bras(la, p)
                   for b in _kets(lb, n))
        if tg:
            out[t] = tg
    return out


def _digits(v: Cart) -> str:
    return "_".join(str(c) for c in v)


def _acc(t: Table, e: Cart) -> str:
    return f"acc{t[0]}{t[1]}_{_digits(e)}"


def _hname(t: Table, node: hrr.Node) -> str:
    return f"t{t[0]}{t[1]}{hrr_name(node)}"


# ---------------------------------------------------------------------------
# Small spellings
# ---------------------------------------------------------------------------


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


def _sum(xs: list[Expr]) -> Expr:
    out = xs[0]
    for x in xs[1:]:
        out = _c(Fn.ADD, out, x)
    return out


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def _gradient(la: int, lb: int, pool: Locals) -> list[Stmt]:
    """`gA_k` per axis (module docstring), the kinetic combinations pooled
    in `pool` as they are first read."""
    nb = vrr.cartesians(lb)

    def h(t: Table, a: Cart, b: Cart) -> Var:
        return Var(_hname(t, (a, b, ZERO, ZERO)))

    def kinetic(p: int, a: Cart, j: int) -> Var:
        b = nb[j]

        def make() -> Expr:  # kernel_hgp's T combination at a'
            v: Expr = _mul(Lit(float(2 * lb + 3)), h((p, 1), a, b))
            v = _c(Fn.SUB, v, _mul(Lit(2.0), _sum(
                [h((p, 2), a, _shift(b, d, 2)) for d in range(3)])))
            low = [_mul(Lit(float(b[d] * (b[d] - 1))),
                        h((p, 0), a, _shift(b, d, -2))) for d in range(3)
                   if b[d] >= 2]
            if low:
                v = _c(Fn.SUB, v, _mul(Lit(0.5), _sum(low)))
            return v
        return pool.get(f"tk{p}_{_digits(a)}__{j}", make)

    out: list[Stmt] = []
    for k in range(3):
        terms: list[Expr] = []
        for i, a in enumerate(vrr.cartesians(la)):
            for j in range(len(nb)):
                g: Expr = kinetic(1, _shift(a, k, 1), j)
                if a[k]:
                    g = _c(Fn.SUB, g, _mul(Lit(float(a[k])),
                                           kinetic(0, _shift(a, k, -1), j)))
                terms.append(_mul(Var(f"w{i * len(nb) + j}"), g))
        out.append(Decl(f"gA{k}", Scalar.F64, init=_sum(terms)))
    return out


def build_hgt_kernel(cls: Spec | tuple[int, int], *,
                     name: str | None = None) -> Kernel:
    """The HGP kinetic-energy gradient kernel of one class.

    `cls` is an HGP d1 `Target.T` `Spec`, or `(la, lb)`. See the module
    docstring.
    """
    la, lb = _class(cls)
    hrrs = {t: _hrr_dag(tg) for t, tg in tables(la, lb).items()}
    bounds = {t: _boundary(d) for t, d in hrrs.items()}
    vdag = _vrr_dag(boundary_union(bounds), nuclear=False)

    # --- per record: the seed, the VRR, the weighted accumulation ----------
    k = Var(_REC)
    vstmts = vrr_body(vdag, lambda m: Var("u"))
    used = _factors(vstmts)
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
            Fn.EXP, _mul(Var("zeta"), _sum([
                _mul(Index(f"pa_{a}", (k,)), Index(f"pb_{a}", (k,)))
                for a in _AXES]))))),
        Note("the tables' weights (2 alpha)^p beta^n: alpha the bra "
             "exponent, beta the ket's"),
        Decl("eb", Scalar.F64, init=Index("eb_ket", (k,))),
        Decl("ea2", Scalar.F64, init=_mul(Lit(2.0), Index("ea_bra", (k,)))),
    ]
    wanted = {_WEIGHT[t] for t in hrrs} - {None, "eb", "ea2"}
    products = {"ebb": ("eb", "eb"), "ea2eb": ("ea2", "eb"),
                "ea2ebb": ("ea2", "ebb")}
    if "ea2ebb" in wanted:
        wanted.add("ebb")
    for w in ("ebb", "ea2eb", "ea2ebb"):
        if w in wanted:
            x, y = products[w]
            record.append(Decl(w, Scalar.F64, init=_mul(Var(x), Var(y))))
    record += vstmts
    for t, bnd in bounds.items():
        w = _WEIGHT[t]
        for e in bnd:
            x: Expr = Var(vrr_name((e, ZERO, 0)))
            if w is not None:
                x = _mul(Var(w), x)
            record.append(Assign(Var(_acc(t, e)),
                                 _c(Fn.ADD, Var(_acc(t, e)), x)))

    # --- per pair: the HRRs on contracted values, the gradient, the scatter
    hstmts: list[Stmt] = []
    for t, dag in hrrs.items():
        hstmts.append(Note(f"table ({t[0]}, {t[1]}): SUM c_a c_b "
                           f"(2 alpha)^{t[0]} beta^{t[1]} (e|0)"))
        hstmts += hrr_body(dag, lambda e, f, t=t: Var(_acc(t, e)),
                           name=lambda node, t=t: _hname(t, node))
    pool = Locals()
    grad = _gradient(la, lb, pool)
    used_ab = _factors(tuple(hstmts))

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
    head.append(Note("the contracted boundary (e|0) of every table, summed "
                     "from 0.0 over the pair's records"))
    for t, bnd in bounds.items():
        head += [Decl(_acc(t, e), Scalar.F64, init=Lit(0.0)) for e in bnd]
    scatter: list[Stmt] = [Note(ATOMIC_REASON + "; dT/dB = -dT/dA "
                                "(translational invariance)")]
    for kk in range(3):
        scatter.append(AtomicAdd(Index("G", (Var("aa"), Lit(kk))),
                                 Var(f"gA{kk}")))
    for kk in range(3):
        scatter.append(AtomicAdd(Index("G", (Var("ab"), Lit(kk))),
                                 _mul(Lit(-1.0), Var(f"gA{kk}"))))
    per_pair = (
        *head,
        Note("the pair's primitive pairs, serially: the VRR per record "
             "([Fock] eq 25 at f = 0)"),
        For(_REC, Var("r0"), Var("r1"), Lit(1), tuple(record)),
        *hstmts,
        *weights(la, lb),
        Note("T(a', b) of each table pair, kernel_hgp's kinetic "
             "combination, each declared where first read"),
        *pool.decls,
        Note("gA_k = SUM_r w_r [T_(1)(a + 1_k, b) - a_k T_(0)(a - 1_k, b)]"),
        *grad,
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"HGP gradient of the kinetic energy, class (la, lb) = ({la}, "
             f"{lb}): G[A, k] += SUM_mn M_mn dT_mn/dR_{{A,k}} "
             "(gradient/kernel_hgt.py)"),
        Note(ZEROED_G),
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
        *(_f64(f"pa_{a}", (nr,)) for a in _AXES),
        *(_f64(f"pb_{a}", (nr,)) for a in _AXES),
        _f64("s_ab", (nr,)), _f64("ea_bra", (nr,)), _f64("eb_ket", (nr,)),
        *(_i32(n, (npair,)) for n in pair_arrays()[:6]),
        *(_f64(n, (npair,)) for n in pair_arrays()[6:]),
        _f64("M", (nao, nao)),
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(nat, Lit(3))),
    ]
    return Kernel(name or hgt_name((la, lb)), tuple(params), body)
