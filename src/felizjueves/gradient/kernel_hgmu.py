"""The dipole gradient by the HGP route: `Target.DIPOLE` at
`derivative_order = 1` on `Engine.HGP`, built as `Kernel`s.

    build_hgmu_kernel(spec | (la, lb)) -> Kernel   one per class
    hgmu_name(spec | (la, lb))         -> str      hgmu_{la}{lb}
    hgmu_spec(la, lb)                  -> Spec     the class's d1 HGP key
    pair_arrays()                      -> names    its per-pair arrays
    launch_shape(n_pair, threads)      -> (grid, block)

The same contraction as `kernel_gmu`'s `gmu_` (`felizjueves.gradient`'s
package docstring; `gradient.host.multipole_gradient` at `e = 1` is the
reference), per component `c` of `MU^c = <a| q (r - C)_c |b>` about a FIXED
origin `C`,

    G[c, A, k] += SUM_mn M_mn dMU^c_mn / dR_{A,k}

on the recurrence of `one_electron.kernel_hgp`'s `hmu_`: the operator moved
onto the ket, `(r - C)_c |b> = |b+1_c> + BC_c |b>`, `BC = B - C`, so

    MU^c(a, b) = q [(a|b+1_c) + BC_c (a|b)]

over contracted overlaps -- Obara & Saika's overlap VRR per primitive pair,
the contraction, Head-Gordon & Pople's HRR on contracted values (`[Fock]`
eq 25 at `f = 0` with the `m + 1` terms dropped, and eq 28's bra half,
`hgp/`'s and not rewritten: `kernel_hgp`'s "THE RECURRENCES").

THE DERIVATIVE RULE acts on the basis function, as `gmu_`'s and `hgs_`'s:

    dA_k MU^c(a, b) = 2 alpha MU^c(a+1_k, b) - a_k MU^c(a-1_k, b)
    dB_k MU^c(a, b) = 2 beta  MU^c(a, b+1_k) - b_k MU^c(a, b-1_k)

The moved operator keeps the ket's centre `B` whatever `b` is, so each
`MU^c` above is the same two-term combination of overlaps. `alpha` and
`beta` are per PRIMITIVE, so -- as `hgs_` -- each is folded in before the
contraction: three contracted tables, each with its own HRR,

    table 0   SUM c_a c_b         (e|0)   (a-1_k|b), (a|b-1_k), each with
                                          its b+1_c
    table 1   SUM c_a c_b 2 alpha (e|0)   (a+1_k|b), (a+1_k|b+1_c)
    table 2   SUM c_a c_b 2 beta  (e|0)   (a|b+1_k), (a|b+1_k+1_c)

(table 0 is absent at (ss)), one VRR to `la + lb + 2` per record feeding all
three. Then, per Cartesian pair `r = (a, b)` of the block, its weight `w_r`
from `M` (`kernel_g1e.weights`, `gs_`'s rule), with `m_n(a, b, c) =
t_n(a|b+1_c) + BC_c t_n(a|b)` declared once, before its first read,

    gA_{c,k} = SUM_r w_r (m_1(a+1_k, b, c) - a_k m_0(a-1_k, b, c))
    gB_{c,k} = SUM_r w_r (m_2(a, b+1_k, c) - b_k m_0(a, b-1_k, c))

and `G[c, atom_a, k] += q gA_{c,k}`, `G[c, atom_b, k] += q gB_{c,k}`,
eighteen `AtomicAdd`s per SHELL PAIR (`gmu_` adds eighteen per primitive
pair). THE ORIGIN IS NOT A NUCLEUS: `C` is fixed, so there is no operator
term and both centres are computed; `SUM_A G[c, A, k] = q delta_ck SUM_mn
M_mn S_mn` (the origin shift) is a check (test/test_kernel_hgmu.py), not an
assumption.

THE THREAD MAPPING is `hgs_`'s: ONE THREAD PER CONTRACTED SHELL PAIR, `pair
= THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by `pair < n_pair`, the pair's
records `[rec0, rec1)` serially, the boundary accumulated in registers from
`0.0`, the HRRs once. Every pair on an atom adds into that atom's row of
every component of `G`, so the adds are atomic (`ATOMIC_REASON`) and `G` is
ZEROED BY THE CALLER (`kernel_g1e.ZEROED_G`).

THE PARAMETER BLOCK, `hgs_`'s with the ket centre per pair (for `BC`, as
`hmu_` takes it) and `gmu_`'s origin and charge:

    params  n_pair n_rec n_ao n_atom (I32)
            zeta_ab pa_x pa_y pa_z pb_x pb_y pb_z s_ab ea_bra eb_ket
                                                           [n_rec] per record
            rec0 rec1 off_a off_b atom_a atom_b  [n_pair]  (I32) per pair
            ab_x ab_y ab_z  b_x b_y b_z          [n_pair]  A - B, and B
            M[n_ao, n_ao]                                  read, any M
            C_x C_y C_z q   (F64 scalars: the origin and the charge, RUNTIME)
            G[3, n_atom, 3]                    (OUT, read and AtomicAdd-ed)

Staged by `gradient.host.pair_records(..., ket_centre=True)`.

SCOPE: shells through f (`L_MAX = 3`), the classes `gmu_` and `hmu_` are
validated through.

ROUND-OFF, NOT BITWISE, against the host: another recurrence, another
association (`kernel_hgp`'s "ROUND-OFF"). test/test_kernel_hgmu.py states
its bound.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
(its witness), nor the interpreter, nor `reference/`.
"""

from __future__ import annotations

from collections.abc import Iterator

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
    "N_COMP",
    "PAIR",
    "build_hgmu_kernel",
    "hgmu_name",
    "hgmu_spec",
    "launch_shape",
    "pair_arrays",
]

L_MAX = 3  # through f: gmu_'s and hmu_'s validated scope
N_COMP = 3  # x, y, z
PAIR = "pair"  # the contracted shell pair: one thread each
_AXES = "xyz"
_REC = "kr"  # the record loop's variable, kernel_hgp's
# Each table's per-record weight: none, 2 alpha, 2 beta (module docstring).
_WEIGHT = {0: None, 1: "ea2", 2: "eb2"}

ATOMIC_REASON = (
    "AtomicAdd: every shell pair on an atom adds into that atom's row of "
    "every component of G")


def hgmu_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of the dipole gradient on `Engine.HGP`."""
    return Spec(OPERATOR_OF[Target.DIPOLE], 1, (la, lb), Target.DIPOLE,
                engine=Engine.HGP)


def _class(cls: Spec | tuple[int, int]) -> tuple[int, int]:
    spec = cls if isinstance(cls, Spec) else hgmu_spec(*cls)
    if (spec.target is not Target.DIPOLE or spec.derivative_order != 1
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d1 HGP DIPOLE key; the MMD "
                         "dipole gradient is kernel_gmu's gmu_")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the gradient kernels take no levers")
    la, lb = spec.am
    if max(la, lb) > L_MAX:
        raise ValueError(f"{spec.key()}: the HGP dipole gradient is built "
                         f"for shells through f only (l <= {L_MAX})")
    return la, lb


def hgmu_name(cls: Spec | tuple[int, int]) -> str:
    """The name `build_hgmu_kernel(cls)` gives its kernel, unbuilt."""
    la, lb = _class(cls)
    return f"hgmu_{la}{lb}"


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


def _arguments(la: int, lb: int) -> Iterator[tuple[int, Cart, Cart]]:
    """Every `(table, a', b')` the derivative rule reads `MU` at (module
    docstring)."""
    for k in range(3):
        for a, b in _rows(la, lb):
            yield 1, _shift(a, k, 1), b
            if a[k]:
                yield 0, _shift(a, k, -1), b
            yield 2, a, _shift(b, k, 1)
            if b[k]:
                yield 0, a, _shift(b, k, -1)


def _tables(la: int, lb: int) -> dict[int, tuple[hrr.Node, ...]]:
    """Each table's HRR targets (module docstring); an empty one left out:
    `(a'|b')` and `(a'|b'+1_c)` for every argument `_arguments` names."""
    out: dict[int, list[hrr.Node]] = {0: [], 1: [], 2: []}
    for n, a, b in _arguments(la, lb):
        out[n].append((a, b, ZERO, ZERO))
        out[n] += [(a, _shift(b, c, 1), ZERO, ZERO) for c in range(3)]
    return {n: tuple(dict.fromkeys(t)) for n, t in out.items() if t}


def _hname(table: int, node: hrr.Node) -> str:
    return f"t{table}{hrr_name(node)}"


def _digits(v: Cart) -> str:
    return "".join(str(x) for x in v)


def _derivatives(la: int, lb: int) -> list[Stmt]:
    """The moved-operator locals `m_n(a', b', c)`, each declared before its
    first read, and `gA_{c,k}`, `gB_{c,k}` as one fold each over the
    block's rows."""
    out: list[Stmt] = []
    seen: set[str] = set()

    def m(n: int, a: Cart, b: Cart, c: int) -> Var:
        name = f"m{n}_{_digits(a)}_{_digits(b)}_{_AXES[c]}"
        if name not in seen:
            seen.add(name)
            out.append(Decl(name, Scalar.F64, init=_c(
                Fn.ADD, Var(_hname(n, (a, _shift(b, c, 1), ZERO, ZERO))),
                _c(Fn.MUL, Var(f"BC_{_AXES[c]}"),
                   Var(_hname(n, (a, b, ZERO, ZERO)))))))
        return Var(name)

    for c in range(N_COMP):
        for side in "AB":
            for k in range(3):
                terms: list[Expr] = []
                for r, (a, b) in enumerate(_rows(la, lb)):
                    if side == "A":
                        v: Expr = m(1, _shift(a, k, 1), b, c)
                        n, lo = a[k], (_shift(a, k, -1), b)
                    else:
                        v = m(2, a, _shift(b, k, 1), c)
                        n, lo = b[k], (a, _shift(b, k, -1))
                    if n:
                        v = _c(Fn.SUB, v, _c(Fn.MUL, Lit(float(n)),
                                             m(0, *lo, c)))
                    terms.append(_c(Fn.MUL, Var(f"w{r}"), v))
                out.append(Decl(f"g{c}{side}{k}", Scalar.F64,
                                init=_fold(terms)))
    return out


def build_hgmu_kernel(cls: Spec | tuple[int, int], *,
                      name: str | None = None) -> Kernel:
    """The HGP dipole-gradient kernel of one class.

    `cls` is a d1 HGP `Target.DIPOLE` `Spec`, or `(la, lb)`. See the module
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
    head.append(Note("BC = B - C: the operator moved onto the ket, "
                     "(r - C)_c |b> = |b+1_c> + BC_c |b>"))
    head += [Decl(f"BC_{_AXES[d]}", Scalar.F64, init=_c(
        Fn.SUB, Index(f"b_{_AXES[d]}", (pair,)), Var(f"C_{_AXES[d]}")))
        for d in range(3)]
    head.append(Note("the contracted boundary (e|0) of each table, summed "
                     "from 0.0 over the pair's records"))
    for n, bnd in bounds.items():
        head += [Decl(acc_name(n, e), Scalar.F64, init=Lit(0.0)) for e in bnd]
    scatter: list[Stmt] = [Note(ATOMIC_REASON)]
    for c in range(N_COMP):
        for side, atom in (("A", "aa"), ("B", "ab")):
            for d in range(3):
                scatter.append(AtomicAdd(
                    Index("G", (Lit(c), Var(atom), Lit(d))),
                    _c(Fn.MUL, Var("q"), Var(f"g{c}{side}{d}"))))
    per_pair = (
        *head,
        Note("the pair's primitive pairs, serially: the VRR per record "
             "([Fock] eq 25 at f = 0, the overlap limit) to la + lb + 2"),
        For(_REC, Var("r0"), Var("r1"), Lit(1), tuple(record)),
        *hstmts,
        *weights(la, lb),
        Note("MU^c(a, b) = (a|b+1_c) + BC_c (a|b) per table; dA_k = "
             "2 alpha MU(a+1_k) - a_k MU(a-1_k), dB_k likewise"),
        *_derivatives(la, lb),
        *scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"HGP gradient of the dipole, mu = <a| q (r - C) |b>, class "
             f"(la, lb) = ({la}, {lb}): G[c, A, k] += SUM_mn M_mn "
             "dMU^c_mn/dR_{A,k} (gradient/kernel_hgmu.py); the origin C is "
             "fixed, not a nucleus"),
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
        *(Param(f"C_{a}", Scalar.F64) for a in _AXES),
        Param("q", Scalar.F64),
        Param("G", Scalar.F64, array=True, intent=Intent.OUT,
              shape=(Lit(N_COMP), nat, Lit(3))),
    ]
    return Kernel(name or hgmu_name((la, lb)), tuple(params), body)
