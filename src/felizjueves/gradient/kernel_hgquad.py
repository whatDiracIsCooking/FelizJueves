"""The quadrupole gradient by the HGP route: `Target.QUADRUPOLE` at
`derivative_order = 1` on `Engine.HGP`, built as `Kernel`s.

    build_hgquad_kernel(spec | (la, lb)) -> Kernel   one per class
    hgquad_name(spec | (la, lb))         -> str      hgquad_{la}{lb}
    hgquad_spec(la, lb)                  -> Spec     the class's d1 HGP key
    pair_arrays()                        -> names    its per-pair arrays
    launch_shape(n_pair, threads)        -> (grid, block)
    N_COMPONENTS                         -> 6

The same contraction as `kernel_gmu`'s `gquad_` (`gradient.host.
multipole_gradient` at `e = 2` is the reference), per component `c` of the
raw second moments `<a| q (r - C)_d (r - C)_e |b>` -- xx xy xz yy yz zz,
`kernel_mu`'s order and `kernel_hgp`'s `hquad_`'s --

    G[c, A, k] += SUM_mn M_mn dQ^c_mn / dR_{A,k}

on the recurrence of `one_electron.kernel_hgp`'s `hquad_`: Obara & Saika's
overlap VRR per primitive pair, the contraction, Head-Gordon & Pople's HRR
on contracted values, and the operator moved onto the ket,

    Q_de(a, b) = (a|b+1_d+1_e) + BC_e (a|b+1_d) + BC_d (a|b+1_e)
                 + BC_d BC_e (a|b),            BC = B - C,

every `(.|.)` an overlap (`kernel_hgp`'s docstring, "Each operator is a
combination"). `[Fock]` eq 25 at `f = 0` with the `m + 1` terms dropped,
and eq 28's bra half, both `hgp/`'s and not rewritten.

THE DERIVATIVE RULE acts on the basis function, not on the operator: `C` is
a FIXED origin, not a nucleus, so there is no operator term (no
Hellmann-Feynman analogue), and `SUM_A G[c, A, k]` is minus the origin
derivative -- a check (test/test_kernel_hgquad.py), not an assumption. The
derivative is taken on the Gaussian first, and the rewriting above (exact
at every `B`) applied to each of its terms afterwards, so `BC` is never
differentiated:

    dA_k Q_de(a, b) = 2 alpha Q_de(a+1_k, b) - a_k Q_de(a-1_k, b)
    dB_k Q_de(a, b) = 2 beta  Q_de(a, b+1_k) - b_k Q_de(a, b-1_k)

`alpha` and `beta` are per PRIMITIVE, so -- as `kernel_hgs` does -- each is
folded into the boundary before the contraction: three contracted tables,
each with its own HRR,

    table 0   SUM c_a c_b         (e|0)   Q at (a-1_k, b), (a, b-1_k)
    table 1   SUM c_a c_b 2 alpha (e|0)   Q at (a+1_k, b)
    table 2   SUM c_a c_b 2 beta  (e|0)   Q at (a, b+1_k)

(table 0 is absent at (ss)), each table's targets every overlap the `Q`s it
is read at reach, one VRR to `la + lb + 3` per record feeding all three.
Then, per Cartesian pair `r = (a, b)` of the block, its weight `w_r` from
`M` (`kernel_g1e.weights`, `gquad_`'s rule),

    gA_{c,k} = SUM_r w_r (Q1_c(a+1_k, b) - a_k Q0_c(a-1_k, b))
    gB_{c,k} = SUM_r w_r (Q2_c(a, b+1_k) - b_k Q0_c(a, b-1_k))

each `Qn_c(a', b')` a straight-line local declared where first read
(`kernel_g1e.Locals`), and `G[c, atom_a, k] += q gA_{c,k}`, `G[c, atom_b,
k] += q gB_{c,k}`: 36 `AtomicAdd`s per SHELL PAIR. Both centres are
computed; translational invariance is not used.

THE THREAD MAPPING is `kernel_hgs`'s: ONE THREAD PER CONTRACTED SHELL PAIR,
`pair = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by `pair < n_pair`, the
pair's records `[rec0, rec1)` serially, the boundary accumulated in
registers from `0.0`, the HRRs once. Every pair on an atom adds into that
atom's rows of `G`, so the adds are atomic (`ATOMIC_REASON`) and `G` is
ZEROED BY THE CALLER (`kernel_g1e.ZEROED_G`).

THE PARAMETER BLOCK, `kernel_hgs`'s plus the ket centre (`hquad_`'s `b_*`)
and the moment's origin and charge:

    params  n_pair n_rec n_ao n_atom (I32)
            zeta_ab pa_x pa_y pa_z pb_x pb_y pb_z s_ab ea_bra eb_ket
                                                           [n_rec] per record
            rec0 rec1 off_a off_b atom_a atom_b  [n_pair]  (I32) per pair
            ab_x ab_y ab_z  b_x b_y b_z          [n_pair]  A - B, and B
            M[n_ao, n_ao]                                  read, any M
            C_x C_y C_z q   (F64 scalars: the origin and the charge, RUNTIME)
            G[6, n_atom, 3]                    (OUT, read and AtomicAdd-ed)

Staged by `gradient.host.pair_records(..., ket_centre=True)`. The origin
and the charge are runtime scalars, as at d0 (`spec.py`, "A DIPOLE CLASS"):
neither is a key field.

SCOPE: shells through f (`L_MAX = 3`), `gquad_`'s and `hquad_`'s validated
scope; `kernel_hgp.table_dags`' machinery is built for any `l`.

ROUND-OFF, NOT BITWISE, against the host: another recurrence, another
association (`kernel_hgp`'s "ROUND-OFF"). test/test_kernel_hgquad.py states
its bound.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
(its witness), nor the interpreter, nor `reference/`.
"""

from __future__ import annotations

from felizjueves.gradient.kernel_g1e import ZEROED_G, Locals, weights
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
    "N_COMPONENTS",
    "PAIR",
    "build_hgquad_kernel",
    "hgquad_name",
    "hgquad_spec",
    "launch_shape",
    "pair_arrays",
]

L_MAX = 3  # through f: gquad_'s and hquad_'s validated scope
PAIR = "pair"  # the contracted shell pair: one thread each
_AXES = "xyz"
_REC = "kr"  # the record loop's variable, kernel_hgp's
# Each table's per-record weight: none, 2 alpha, 2 beta (module docstring).
_WEIGHT = {0: None, 1: "ea2", 2: "eb2"}
# The components, `cartesians(2)` order (xx xy xz yy yz zz), as the axis
# pairs (d, e), d <= e: kernel_hgp's and kernel_mu's order.
_QUAD = tuple(tuple(d for d in range(3) for _ in range(c[d]))
              for c in vrr.cartesians(2))
N_COMPONENTS = len(_QUAD)

ATOMIC_REASON = (
    "AtomicAdd: every shell pair on an atom adds into that atom's row of "
    "every component of G")


def hgquad_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of the quadrupole gradient on
    `Engine.HGP`."""
    return Spec(OPERATOR_OF[Target.QUADRUPOLE], 1, (la, lb),
                Target.QUADRUPOLE, engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else hgquad_spec(*cls)
    if (spec.target is not Target.QUADRUPOLE or spec.derivative_order != 1
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d1 HGP quadrupole key; the "
                         "MMD quadrupole gradient is kernel_gmu's gquad_")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb = spec.am
    if max(la, lb) > L_MAX:
        raise ValueError(f"{spec.key()}: the HGP quadrupole gradient is "
                         f"built for shells through f only (l <= {L_MAX})")
    return la, lb


def hgquad_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_hgquad_kernel(cls)` gives its kernel, unbuilt."""
    la, lb = _class(cls)
    return f"hgquad_{la}{lb}"


def pair_arrays() -> tuple[str, ...]:
    """The per-pair array parameters, in order."""
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


def _reads(a: Cart, b: Cart, d: int, e: int) -> tuple[hrr.Node, ...]:
    """The overlaps `Q_de(a, b)` reads, in its terms' order (module
    docstring)."""
    return ((a, _shift(_shift(b, d, 1), e, 1), ZERO, ZERO),
            (a, _shift(b, d, 1), ZERO, ZERO),
            (a, _shift(b, e, 1), ZERO, ZERO),
            (a, b, ZERO, ZERO))


def _at(la: int, lb: int) -> dict[int, list[tuple[Cart, Cart]]]:
    """Each table's `(a', b')`, the pairs its `Q` is read at (module
    docstring); an empty table left out."""
    out: dict[int, list[tuple[Cart, Cart]]] = {0: [], 1: [], 2: []}
    for a, b in _rows(la, lb):
        for k in range(3):
            out[1].append((_shift(a, k, 1), b))
            out[2].append((a, _shift(b, k, 1)))
            if a[k]:
                out[0].append((_shift(a, k, -1), b))
            if b[k]:
                out[0].append((a, _shift(b, k, -1)))
    return {n: list(dict.fromkeys(p)) for n, p in out.items() if p}


def _tables(la: int, lb: int) -> dict[int, tuple[hrr.Node, ...]]:
    """Each table's HRR targets: every overlap its `Q`s read."""
    return {n: tuple(dict.fromkeys(
        node for a, b in at for d, e in _QUAD for node in _reads(a, b, d, e)))
        for n, at in _at(la, lb).items()}


def _hname(table: int, node: hrr.Node) -> str:
    return f"t{table}{hrr_name(node)}"


def _qname(table: int, c: int, a: Cart, b: Cart) -> str:
    return f"Q{table}c{c}_" + "_".join(str(x) for x in (*a, *b))


def _derivatives(la: int, lb: int, pool: Locals) -> list[Stmt]:
    """`gA_{c,k}`, `gB_{c,k}` as one fold each over the block's rows, the
    `Q`s they read pooled (module docstring)."""

    def Q(table: int, c: int, a: Cart, b: Cart) -> Var:
        d, e = _QUAD[c]

        def make() -> Expr:
            t = [Var(_hname(table, n)) for n in _reads(a, b, d, e)]
            bd, be = (Var(f"BC_{_AXES[x]}") for x in (d, e))
            return _fold([t[0], _mul(be, t[1]), _mul(bd, t[2]),
                          _mul(_mul(bd, be), t[3])])
        return pool.get(_qname(table, c, a, b), make)

    out: list[Stmt] = []
    for c in range(N_COMPONENTS):
        for side in "AB":
            for k in range(3):
                terms: list[Expr] = []
                for r, (a, b) in enumerate(_rows(la, lb)):
                    if side == "A":
                        v: Expr = Q(1, c, _shift(a, k, 1), b)
                        n, lo = a[k], (_shift(a, k, -1), b)
                    else:
                        v = Q(2, c, a, _shift(b, k, 1))
                        n, lo = b[k], (a, _shift(b, k, -1))
                    if n:
                        v = _c(Fn.SUB, v, _mul(Lit(float(n)), Q(0, c, *lo)))
                    terms.append(_mul(Var(f"w{r}"), v))
                out.append(Decl(f"g{c}{side}{k}", Scalar.F64,
                                init=_fold(terms)))
    return out


def build_hgquad_kernel(cls: Spec | tuple[int, int], *,
                        name: str | None = None) -> Kernel:
    """The HGP quadrupole-gradient kernel of one class.

    `cls` is a d1 HGP `Target.QUADRUPOLE` `Spec`, or `(la, lb)`. See the
    module docstring.
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
        Note("the tables' weights: 2 alpha (table 1) and 2 beta (table 2), "
             "the derivative rule's, per primitive"),
        Decl("ea2", Scalar.F64, init=_mul(Lit(2.0), Index("ea_bra", (k,)))),
        Decl("eb2", Scalar.F64, init=_mul(Lit(2.0), Index("eb_ket", (k,)))),
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

    pool = Locals()
    derivatives = _derivatives(la, lb, pool)

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
        for d in range(3)]
    head.append(Note("the contracted boundary (e|0) of each table, summed "
                     "from 0.0 over the pair's records"))
    for n, bnd in bounds.items():
        head += [Decl(acc_name(n, e), Scalar.F64, init=Lit(0.0)) for e in bnd]
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for c in range(N_COMPONENTS):
        for side, atom in (("A", "aa"), ("B", "ab")):
            for d in range(3):
                scatter.append(AtomicAdd(
                    Index("G", (Lit(c), Var(atom), Lit(d))),
                    _mul(Var("q"), Var(f"g{c}{side}{d}"))))
    per_pair = (
        *head,
        Note("the pair's primitive pairs, serially: the VRR per record "
             "([Fock] eq 25 at f = 0, the overlap limit) to la + lb + 3"),
        For(_REC, Var("r0"), Var("r1"), Lit(1), tuple(record)),
        *hstmts,
        *weights(la, lb),
        Note("Q_de(a, b) = (a|b+1_d+1_e) + BC_e (a|b+1_d) + BC_d (a|b+1_e) "
             "+ BC_d BC_e (a|b) per table, each declared where first read"),
        *pool.decls,
        Note("dA_k Q(a, b) = 2 alpha Q(a+1_k, b) - a_k Q(a-1_k, b), "
             "dB_k likewise; C is fixed, so no operator term"),
        *derivatives,
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note("HGP gradient of the quadrupole (raw second moments) "
             "<a| q (r - C)_d (r - C)_e |b>, xx xy xz yy yz zz, class "
             f"(la, lb) = ({la}, {lb}): G[c, A, k] += SUM_mn M_mn "
             "dQ^c_mn/dR_{A,k} (gradient/kernel_hgquad.py); the origin C "
             "is fixed, not a nucleus"),
        Note(ZEROED_G),
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
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(N_COMPONENTS), nat, Lit(3))),
    ]
    return Kernel(name or hgquad_name((la, lb)), tuple(params), body)
