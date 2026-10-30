"""The fused Fock gradient kernel: `Target.F` at `derivative_order = 1` on
the HGP engine, `kernel_f`'s mapping differentiated.

    build_gf_kernel(spec | (la, lb, lc, ld)) -> Kernel   one per F class
    gf_name(spec | (la, lb, lc, ld))         -> str      gf_{la}{lb}{lc}{ld}
    gf_spec(la, lb, lc, ld)                  -> Spec     the class's d1 key
    derive_gradient_weights()                -> GradWeights, computed
    derivative_classes(cls)                  -> {position: (raised, lowered)}
    pack(pairs) / GPrimRec / GPairRec        what the kernel reads
    launch_shape(n_bra, threads)             -> (n_bra, (threads, 1))
    params_of(cls)                           -> the parameter names, in order

THE ENERGY. With `D` the total closed-shell density and `G = J - K/2`
(`kernel_f`'s), the two-electron energy is

    E_2e = 1/2 SUM_ab D_ab G_ab = SUM_abcd Gam_abcd (ab|cd),
    Gam_abcd = 1/2 D_ab D_cd - 1/4 D_ac D_bd

so `dE_2e/dR = dE_J/dR - 1/2 dE_K/dR` in `gradient.rhf`'s terms (its
`coulomb` and `exchange` terms, signed, summed): the whole two-electron
gradient of RHF in ONE kernel per class, as `kernel_f` builds the whole
`G` in one. `G[A, k] += dE_2e / dR_{A,k}` is what it writes.

THE WEIGHTS, DERIVED (`derive_gradient_weights`). The kernel visits ONE
representative per orbit of `kernel_f.orbit_group()` (the canonical half
and the degeneracy factor are `kernel_f`'s, unchanged), and each member's
integral is a re-indexing of the representative's, so its derivative is
too. Summing `Gam` over the eight images of `(a, b, c, d)` (a member
`(p, q, r, s)` contributes `1/2 D_pq D_rs - 1/4 D_pr D_qs`; `D`
symmetric) gives, per unique quartet and per component `(i, j, k, l)`,

    w_ijkl = deg (4 D_ab[i,j] D_cd[k,l] - D_ac[i,k] D_bd[j,l]
                                        - D_ad[i,l] D_bc[j,k])

the Coulomb coefficient 4 = 8 x 1/2, each exchange one -1 = 4 x -1/4
(four images pair `(a,c)` with `(b,d)`, four pair `(a,d)` with `(b,c)`),
and `deg = 1/|Stab|` (`kernel_f.degeneracy`) where images coincide. Then

    dE_2e / dR_{X,k} = SUM_unique SUM_ijkl w_ijkl d(ab|cd)_ijkl / dX_k

over the quartet's four centres `X = A, B, C, D` -- the representative
stands for its members, whose bra centres are its ket centres, so ALL
FOUR are differentiated (the bra-only shortcut of `gjab` / `gk` is a
property of a sweep over every ordered quartet, not of one orbit's
representative). No translational invariance is used: `D`'s derivative is
computed like the rest, so `SUM_A G[A] = 0` stays a check
(`felizjueves.gradient`'s package docstring).

THE DERIVATIVE, BY HGP. `d/dA_k phi_a = 2 alpha phi_{a+1_k} - a_k
phi_{a-1_k}` (the package docstring's rule), so

    d(ab|cd)/dA_k = (a+1_k b|cd)^{2 alpha} - a_k (a-1_k b|cd)

where `^{2 alpha}` is the contracted integral whose every primitive
quartet is weighted by `2 alpha` of its `a` primitive. Eq 28 holds on
contracted values with any primitive weights (`hgp/hrr.py`, "ITS INPUTS
ARE CONTRACTED VALUES": its coefficients `AB`, `CD` are the same for every
primitive of the pair, so a weighted primitive sum transfers as an
unweighted one does). So per primitive quartet the kernel runs ONE VRR,
over the union of every boundary the derivative classes read
(`derivative_classes`: per centre the class raised by one there and,
when its `l > 0`, the class lowered by one there), to Boys order `L + 1`,
and accumulates five contracted families:

    c0  every boundary value of the LOWERED classes, weight 1
    cA  the A-raised class's boundary, weight 2 alpha   (bra record e1)
    cB  the B-raised class's, 2 beta                    (bra record e2)
    cC  the C-raised class's, 2 gamma                   (ket record e1)
    cD  the D-raised class's, 2 delta                   (ket record e2)

After the nest, one HRR per derivative class (`hgp.recurrence.hrr_body`,
its locals prefixed `ap`, `am`, `bp`, ... per class), each on its own
family, and per ERI target `t` of the class, centre `X` and axis `k`:

    gX_k += w_t (H^{X+}[t + 1_k at X] - t_X,k H^{X-}[t - 1_k at X]).

THE NEST AND THE RECORD ARE `kernel_f`'s: one block per bra pair, one
thread per contracted quartet, the bra's primitive pairs staged in shared
memory by every thread (the IR has no barrier), ket primitives outer and
bra primitives inner, the canonical half `k < diag * bra` skipped, `deg`
from the two shell-index tests and the diagonal launch. The record
(`GREC`) is `kernel_f.REC` plus the pair's two primitive EXPONENTS (`e1`,
`e2`: `alpha`, `beta` on the bra; `gamma`, `delta` on the ket), which the
weights need and `kernel_f` does not; no bound `G_ab`. `AB` and `CD` are
always read (the B- and D-raised HRRs move a unit onto `b` and `d`
whatever `lb` and `ld`), and each side's atom indices (`bra_at`, `ket_at`, `n x 2`)
place the four centres in `G`.

NO SCREEN; `tau` is not a parameter. The F screen bounds an integral, not
its derivative; a gradient bound needs the factor one angular momentum
higher (#555's guardrail). Every unique quartet of a launch is visited.

THE WRITE. `gA_k`, `gB_k` (the bra's centres) are summed over the
thread's whole ket loop and added once, after it; `gC_k`, `gD_k` per
quartet: `6 + 6 x quartets` `AtomicAdd`s into `G[atom, k]`, `G`
`n_atom x 3`, zeroed by the caller before the first launch and carried
through every launch. No host-side `close`: a gradient has no transpose.

Nothing here imports `plan/` (the plan layer refuses an HGP class), `ao/`
or `gradient.host` (the witnesses), `reference/`, the interpreter, a
backend or a device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from felizjueves.eri.fock.kernel_f import MAX_PRIM_PAIRS, REC, orbit_group
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import (
    PAIR,
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    hrr_body,
    hrr_name,
    vrr_body,
    vrr_name,
)
from felizjueves.hgp.vrr import AmClass, Cart
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
    Space,
    Stmt,
    Var,
    Weighted,
)
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "AB_SH",
    "CENTRES",
    "G_OUT",
    "GPairRec",
    "GPrimRec",
    "GREC",
    "GradWeights",
    "MAX_PRIM_PAIRS",
    "Packed",
    "WEIGHTS",
    "build_gf_kernel",
    "derivative_classes",
    "derive_gradient_weights",
    "gf_name",
    "gf_spec",
    "launch_shape",
    "pack",
    "params_of",
]

GREC = (*REC, "e1", "e2")  # kernel_f's record, then the two exponents
AB_SH = "ab_sh"  # the staged |ab) record, Space.SHARED
G_OUT = "G"  # n_atom x 3, AtomicAdd-ed, zeroed by the caller
KET = "k"
KAB = "kab"
KCD = "kcd"
STAGE = "i"
CENTRES = "ABCD"  # the four centres, by AO position
_SH = ("sa", "sb", "sc", "sd")
_AO = ("oa", "ob", "oc", "od")
_AT = ("ta", "tb", "tc", "td")
_P, _X, _ZETA, _U, _E1, _E2 = 0, 3, 6, 7, 8, 9
HALF = "the canonical half: this unordered pair of pairs is met as (k, bra)"
STAGING = ("every thread copies the whole |ab) record, exponents included "
           "(the IR has no barrier; kernel_f's docstring, 'THE STAGING IS "
           "PER THREAD')")


# --- the weights, derived from the permutation orbit (module docstring) ------


@dataclass(frozen=True)
class GradWeights:
    """`w = deg (coulomb D_ab D_cd + exchange (D_ac D_bd + D_ad D_bc))`."""

    coulomb: float
    exchange: float


COULOMB_GAMMA = 0.5  # Gam_abcd = 1/2 D_ab D_cd - 1/4 D_ac D_bd
EXCHANGE_GAMMA = -0.25


def derive_gradient_weights() -> GradWeights:
    """Sum `Gam` over the eight images of a representative (module
    docstring, "THE WEIGHTS"), counting which density pairs each image's
    Coulomb and exchange terms read. Refuses anything the closed form
    assumes and the enumeration does not show."""
    group = orbit_group()
    if len(group) != 8:
        raise AssertionError(f"the ERI group has order {len(group)}, not 8")

    def pairing(g, x: tuple[int, int], y: tuple[int, int]) -> frozenset:
        return frozenset((frozenset((g[x[0]], g[x[1]])),
                          frozenset((g[y[0]], g[y[1]]))))

    coul: dict[frozenset, int] = {}
    exch: dict[frozenset, int] = {}
    for g in group:  # member (p, q, r, s) = positions (g0, g1, g2, g3)
        jk, kx = pairing(g, (0, 1), (2, 3)), pairing(g, (0, 2), (1, 3))
        coul[jk] = coul.get(jk, 0) + 1
        exch[kx] = exch.get(kx, 0) + 1
    ident = (0, 1, 2, 3)
    ab_cd = pairing(ident, (0, 1), (2, 3))
    ac_bd = pairing(ident, (0, 2), (1, 3))
    ad_bc = pairing(ident, (0, 3), (1, 2))
    if coul != {ab_cd: 8}:
        raise AssertionError(f"Coulomb terms read {coul}, not D_ab D_cd x 8")
    if exch != {ac_bd: 4, ad_bc: 4}:
        raise AssertionError(f"exchange terms read {exch}, not D_ac D_bd x 4 "
                             "and D_ad D_bc x 4")
    return GradWeights(COULOMB_GAMMA * coul[ab_cd],
                       EXCHANGE_GAMMA * exch[ac_bd])


WEIGHTS = derive_gradient_weights()  # GradWeights(coulomb=4.0, exchange=-1.0)


# --- the key, the name, the launch -------------------------------------------


def gf_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The d1 F class `(la lb | lc ld)`, AO order, HGP (F's native engine)."""
    return Spec(Operator.COULOMB, 1, (la, lb, lc, ld), Target.F)


def _class(cls: Spec | AmClass) -> AmClass:
    if isinstance(cls, Spec):
        if cls.target is not Target.F or cls.derivative_order != 1:
            raise ValueError(f"{cls.key()}: build_gf_kernel builds Target.F "
                             "at derivative order 1 only")
        if cls.levers:
            raise ValueError(f"{cls.key()}: no lever is built on the F "
                             f"gradient, not {cls.levers[0]!r}")
        return tuple(cls.am)  # type: ignore[return-value]
    am = tuple(cls)
    if len(am) != 4 or any(type(x) is not int or x < 0 for x in am):
        raise ValueError(f"class {cls!r} is not four ints >= 0 "
                         "(la, lb, lc, ld)")
    gf_spec(*am)  # the key exists, or Spec says why not
    return am  # type: ignore[return-value]


def gf_name(cls: Spec | AmClass) -> str:
    """`gf_{la}{lb}{lc}{ld}`, `_`-separated once any `l` is above 9."""
    am = _class(cls)
    sep = "_" if any(x > 9 for x in am) else ""
    return "gf_" + sep.join(map(str, am))


def launch_shape(n_bra: int, threads: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (threads, 1))`: `kernel_f`'s mapping."""
    for name, v in (("n_bra", n_bra), ("threads", threads)):
        if type(v) is not int or v < 1:
            raise ValueError(f"{name} = {v!r} is not a positive int")
    return n_bra, (threads, 1)


# --- the derivative classes and their union VRR ------------------------------


def _shift(cls: AmClass, pos: int, by: int) -> AmClass:
    return tuple(x + by * (i == pos) for i, x in enumerate(cls))  # type: ignore[return-value]


def derivative_classes(cls: Spec | AmClass
                       ) -> dict[int, tuple[AmClass, AmClass | None]]:
    """Per AO position (0..3 = A..D): the class raised by one there, and
    the class lowered by one there (`None` when that `l` is 0)."""
    am = _class(cls)
    return {p: (_shift(am, p, 1), _shift(am, p, -1) if am[p] else None)
            for p in range(4)}


def _union_vrr(roots: Sequence[vrr.Node]) -> vrr.VrrDag:
    """The VRR DAG reaching every root by `hgp.vrr.reduce`'s terms, in
    `order_key`'s order -- `hgp.vrr.build`'s pruned DAG, for a target set
    that is several classes' boundaries. Its `cls` records only its top
    order, `(L + 1, 0, 0, 0)`."""
    terms: dict = {}
    todo = list(roots)
    while todo:
        n = todo.pop()
        if n in terms:
            continue
        terms[n] = vrr.reduce(n)
        todo.extend(t.child for t in terms[n])
    order = tuple(sorted(terms, key=vrr.order_key))
    top = max(sum(e) + sum(f) for e, f, _ in roots)
    return vrr.VrrDag((top, 0, 0, 0), True, tuple(roots),
                      MappingProxyType(terms), order)


def _boundary(cls: AmClass) -> tuple[tuple[Cart, Cart], ...]:
    return tuple((e, f) for e, f, _ in vrr.targets(cls))


# --- the packer --------------------------------------------------------------

Vec = tuple[float, float, float]


@dataclass(frozen=True)
class GPrimRec:
    """`kernel_f.PrimRec` and the pair's two primitive exponents."""

    P: Vec
    X: Vec
    zeta: float
    U: float
    e1: float
    e2: float

    def row(self) -> tuple[float, ...]:
        return (*map(float, self.P), *map(float, self.X), float(self.zeta),
                float(self.U), float(self.e1), float(self.e2))


@dataclass(frozen=True)
class GPairRec:
    """One contracted shell pair: `sep = A - B` (or `C - D`), its primitive
    pairs, its two shell indices, AO offsets and atoms."""

    sep: Vec
    prims: tuple[GPrimRec, ...]
    sh: tuple[int, int]
    ao: tuple[int, int]
    at: tuple[int, int]


@dataclass(frozen=True)
class Packed:
    """One side, flat and row-major: `prim` is `n x degree x GREC`, `sep`
    `n x 3`, `sh`, `ao` and `at` `n x 2` ints."""

    n: int
    degree: int
    prim: tuple[float, ...]
    sep: tuple[float, ...]
    sh: tuple[int, ...]
    ao: tuple[int, ...]
    at: tuple[int, ...]


def pack(pairs: Sequence[GPairRec]) -> Packed:
    """Stage one side: `kernel_f.pack`'s refusals (empty, mixed degree,
    above `MAX_PRIM_PAIRS`)."""
    pairs = tuple(pairs)
    if not pairs:
        raise ValueError("an empty side has no pairs to stage")
    degrees = {len(p.prims) for p in pairs}
    if len(degrees) != 1:
        raise ValueError(f"degrees {sorted(degrees)}: a batch's pairs share "
                         "one contraction degree")
    (degree,) = degrees
    if not 1 <= degree <= MAX_PRIM_PAIRS:
        raise ValueError(f"degree {degree} is outside 1..{MAX_PRIM_PAIRS}, "
                         "the shared staging cap")
    for p in pairs:
        for what in ("sh", "ao", "at"):
            v = getattr(p, what)
            if len(v) != 2 or any(type(x) is not int or x < 0 for x in v):
                raise ValueError(f"pair {what} = {v!r} is not two "
                                 "non-negative ints")
    return Packed(len(pairs), degree,
                  tuple(x for p in pairs for r in p.prims for x in r.row()),
                  tuple(float(x) for p in pairs for x in p.sep),
                  tuple(x for p in pairs for x in p.sh),
                  tuple(x for p in pairs for x in p.ao),
                  tuple(x for p in pairs for x in p.at))


# --- the kernel ----------------------------------------------------------------


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _F(name: str, e: Expr) -> Decl:
    return Decl(name, Scalar.F64, init=e)


def _I(name: str, e: Expr) -> Decl:
    return Decl(name, Scalar.I32, init=e)


def _add_to(name: str, e: Expr) -> Assign:
    return Assign(Var(name), _c(Fn.ADD, Var(name), e))


def _factors(stmts: Sequence[Stmt]) -> set[str]:
    out: set[str] = set()
    for s in stmts:
        if isinstance(s, Decl) and isinstance(s.init, Weighted):
            out |= {t.g.name_in_kernel for t in s.init.terms
                    if t.g.name_in_kernel is not None}
    return out


def _acc(fam: str, e: Cart, f: Cart) -> str:
    """Family `fam`'s contracted accumulator of boundary `(e0|f0)`."""
    return f"c{fam}_" + "__".join("_".join(map(str, v)) for v in (e, f))


def _hname(tag: str):
    return lambda node: tag + hrr_name(node)


def _comp(v: Cart, l: int) -> int:  # noqa: E741
    return vrr.cartesians(l).index(v)


def _moved(t, pos: int, axis: int, by: int):
    return tuple(tuple(c + by * (q == pos and d == axis)
                       for d, c in enumerate(v)) for q, v in enumerate(t))


_DERIVED = ((Factor.INV2P_ALPHA_OVER_P.value, Factor.INV2P.value, "p"),
            (Factor.INV2Q_ALPHA_OVER_Q.value, Factor.INV2Q.value, "q"))


def _quartet_scalars(vneed: frozenset[str]) -> list[Stmt]:
    """`kernel_f`'s quartet scalars (`f_levers.seams.quartet_scalars`, the
    literal path), in its operation order, from the bra (`Pb`, `p`, `Ub`)
    and ket (`Qk`, `q`, `Uq`) primitive records."""
    wp = [Factor.wp(d).value for d in range(3)]
    wq = [Factor.wq(d).value for d in range(3)]
    out: list[Stmt] = [
        Note("quartet geometry ([Fock] §3.1): W - P and W - Q with the "
             "DERIVED sign"),
        *(_F(n, _c(Fn.SUB, Var(f"Pb{d}"), Var(f"Qk{d}")))
          for d, n in enumerate(PQ_PARAMS)),
        _F("ppq", _c(Fn.ADD, Var("p"), Var("q"))),
        _F("rho", _c(Fn.DIV, _c(Fn.MUL, Var("p"), Var("q")), Var("ppq"))),
        _F(THETA, _c(Fn.SQRT, Var("rho"))),
    ]
    if vneed & set(wp):
        out.append(_F("q_ppq", _c(Fn.DIV, Var("q"), Var("ppq"))))
    if vneed & set(wq):
        out.append(_F("p_ppq", _c(Fn.DIV, Var("p"), Var("ppq"))))
    for d in range(3):
        if wp[d] in vneed:
            out.append(_F(wp[d], _c(Fn.SUB, Lit(0.0), _c(
                Fn.MUL, Var("q_ppq"), Var(PQ_PARAMS[d])))))
        if wq[d] in vneed:
            out.append(_F(wq[d], _c(Fn.MUL, Var("p_ppq"), Var(PQ_PARAMS[d]))))
    for n, x in ((Factor.INV2P.value, "p"), (Factor.INV2Q.value, "q"),
                 (Factor.INV2PQ.value, "ppq")):
        if n in vneed:
            out.append(_F(n, _c(Fn.DIV, Lit(0.5), Var(x))))
    for n, inv2, x in _DERIVED:
        if n in vneed:
            out.append(_F(n, _c(Fn.MUL, Var(inv2), _c(
                Fn.DIV, Var("rho"), Var(x)))))
    out.append(_F(PAIR, _c(Fn.MUL, Var("Ub"), Var("Uq"))))
    return out


@dataclass(frozen=True)
class _Hrr:
    """One derivative class's HRR: the family it reads, its local-name
    tag (`ap` = A raised, `am` = A lowered, ...), its class and DAG."""

    fam: str
    tag: str
    cls: AmClass
    dag: hrr.HrrDag


def _plan(am: AmClass):
    """The derivative classes' HRRs, each family's boundary and the union
    VRR (module docstring, "THE DERIVATIVE, BY HGP")."""
    hrrs: list[_Hrr] = []
    fams: dict[str, list[tuple[Cart, Cart]]] = {"0": []}
    for pos, (up, down) in derivative_classes(am).items():
        X = CENTRES[pos]
        fams[X] = list(_boundary(up))
        hrrs.append(_Hrr(X, f"{X.lower()}p", up, hrr.build(up)))
        if down is not None:
            fams["0"] += [ef for ef in _boundary(down) if ef not in fams["0"]]
            hrrs.append(_Hrr("0", f"{X.lower()}m", down, hrr.build(down)))
    for h in hrrs:
        if set(_boundary(h.cls)) != h.dag.boundary:  # #274; refuse, never misread
            raise ValueError(f"{h.cls}: HRR boundary is not the VRR target set")
    roots = sorted({(e, f, 0) for bnd in fams.values() for e, f in bnd},
                   key=vrr.order_key)
    return (tuple(hrrs), {k: tuple(v) for k, v in fams.items()},
            _union_vrr(roots))


# each raised family's weight: 2 x the exponent of its centre's primitive
_WEIGHT = {"A": "Ea", "B": "Eb", "C": "Ec", "D": "Ed"}


def params_of(cls: Spec | AmClass) -> tuple[str, ...]:
    """The kernel's parameter names, in order."""
    la, lb, lc, ld = _class(cls)
    return ("n_bra", "n_ket", "k_ab", "k_cd", "n_atom", "n_ao", "diag",
            "bra_prim", "AB", *(("bra_sh",) if la == lb else ()), "bra_ao",
            "bra_at", "ket_prim", "CD", *(("ket_sh",) if lc == ld else ()),
            "ket_ao", "ket_at", "D",
            G_OUT)


def build_gf_kernel(cls: Spec | AmClass, *, name: str | None = None,
                    weights: GradWeights = WEIGHTS) -> Kernel:
    """The fused Fock gradient kernel of one F class (module docstring).
    `cls` is a d1 `Target.F` `Spec` or `(la, lb, lc, ld)` in AO order.
    `weights` is the derived `WEIGHTS` unless a test substitutes a wrong
    one to watch it fail."""
    am = _class(cls)
    if not isinstance(weights, GradWeights):
        raise TypeError(f"weights {weights!r} is not a GradWeights")
    la, lb, lc, ld = am
    hrrs, fams, vdag = _plan(am)
    prelude, fundamental = fundamental_prelude(vdag.cls[0])
    vbody = tuple(vrr_body(vdag, fundamental))
    vf = _factors(vbody)
    vneed = frozenset(vf | {i2 for n, i2, _ in _DERIVED if n in vf})
    hbodies = [tuple(hrr_body(h.dag, lambda e, f, fam=h.fam: Var(
        _acc(fam, e, f)), name=_hname(h.tag))) for h in hrrs]
    hneed = frozenset(set().union(*(_factors(b) for b in hbodies)))
    by_tag = {h.tag: h for h in hrrs}
    n = tuple(len(vrr.cartesians(x)) for x in am)
    deg_bra, deg_ket, deg_pq = la == lb, lc == ld, (la, lb) == (lc, ld)
    bra, k = Var("bra"), Var(KET)

    def bra_prim(f: int) -> Expr:
        return Index(AB_SH, (Var(KAB), Lit(f)))

    def ket_prim(f: int) -> Expr:
        return Index("ket_prim", (k, Var(KCD), Lit(f)))

    # -- the primitive nest: ket outer (global), bra inner (shared) ----------
    accumulate: list[Stmt] = [
        Note("eq 27 per family: c0 += [e0|f0]^(0) over the lowered classes' "
             "boundaries, cX += 2 x_X [e0|f0]^(0) over centre X's raised one"),
        *(_add_to(_acc("0", e, f), Var(vrr_name((e, f, 0))))
          for e, f in fams["0"]),
    ]
    for X in CENTRES:
        accumulate += [_add_to(_acc(X, e, f), _c(
            Fn.MUL, Var(_WEIGHT[X]), Var(vrr_name((e, f, 0)))))
            for e, f in fams[X]]
    per_kab = (
        Note("the bra primitive pair, from shared memory"),
        *(_F(f"Pb{d}", bra_prim(_P + d)) for d in range(3)),
        *(_F(Factor.pa(d).value, bra_prim(_X + d)) for d in range(3)
          if Factor.pa(d).value in vneed),
        _F("p", bra_prim(_ZETA)),
        _F("Ub", bra_prim(_U)),
        _F("Ea", _c(Fn.MUL, Lit(2.0), bra_prim(_E1))),
        _F("Eb", _c(Fn.MUL, Lit(2.0), bra_prim(_E2))),
        *_quartet_scalars(vneed),
        *prelude,
        *vbody,
        *accumulate,
    )
    per_kcd = (
        Note("the ket primitive pair, from global memory (this thread's |cd))"),
        *(_F(f"Qk{d}", ket_prim(_P + d)) for d in range(3)),
        *(_F(Factor.qc(d).value, ket_prim(_X + d)) for d in range(3)
          if Factor.qc(d).value in vneed),
        _F("q", ket_prim(_ZETA)),
        _F("Uq", ket_prim(_U)),
        _F("Ec", _c(Fn.MUL, Lit(2.0), ket_prim(_E1))),
        _F("Ed", _c(Fn.MUL, Lit(2.0), ket_prim(_E2))),
        For(KAB, Lit(0), Var("k_ab"), Lit(1), per_kab),
    )

    # -- the weights per quartet, then the digestion per ERI -----------------
    def dname(b: tuple[int, int], i: int, j: int, s: str = "d") -> str:
        return f"{s}_{CENTRES[b[0]].lower()}{CENTRES[b[1]].lower()}_{i}_{j}"

    def dload(b: tuple[int, int], i: int, j: int) -> Expr:
        return Index("D", (_c(Fn.ADD, Var(_AO[b[0]]), Lit(i)),
                           _c(Fn.ADD, Var(_AO[b[1]]), Lit(j))))

    def block(b: tuple[int, int], s: str | None = None) -> list[Stmt]:
        return [_F(dname(b, i, j, s or "d"), dload(b, i, j) if s is None
                   else _c(Fn.MUL, Var(s), dload(b, i, j)))
                for i in range(n[b[0]]) for j in range(n[b[1]])]

    flags = [f for f, on in (("hb", deg_bra), ("hk", deg_ket)) if on]
    deg_init: Expr = Var(flags[0]) if flags else Lit(1.0)
    if len(flags) == 2:
        deg_init = _c(Fn.MUL, Var(flags[0]), Var(flags[1]))
    setup: list[Stmt] = [
        Note("the degeneracy factor 1/|Stab| (kernel_f): 1/2 for a = b, for "
             "c = d, and for P = Q"),
        *(_half_if_same("hk", "sc", "sd") if deg_ket else ()),
        _F("deg", deg_init),
    ]
    if deg_pq:
        setup.append(If(_c(Fn.LT, k, _c(Fn.ADD, Var("kmin"), Var("diag"))),
                        (Note("P = Q: the diagonal launch at k == bra"),
                         Assign(Var("deg"), _c(Fn.MUL, Var("deg"),
                                               Lit(0.5))))))
    setup += [
        _F("s_j", _c(Fn.MUL, Lit(float(weights.coulomb)), Var("deg"))),
        _F("s_k", _c(Fn.MUL, Lit(float(weights.exchange)), Var("deg"))),
        Note("the density blocks the weights read: w_t = D_ab (s_j D_cd) + "
             "(s_k D_ac) D_bd + (s_k D_ad) D_bc, deg and the derived "
             "coefficients folded in once per quartet"),
        *block((2, 3), "s_j"), *block((0, 2), "s_k"), *block((0, 3), "s_k"),
        *block((1, 3)), *block((1, 2)),
        Note("this quartet's four centre sums, zeroed"),
        *(_F(f"g{X}_{a}", Lit(0.0)) for X in CENTRES for a in range(3)),
    ]
    digest: list[Stmt] = [Note(
        "per ERI t: w_t, then gX_k += w_t (H^{X+}[t + 1_k at X] - t_X,k "
        "H^{X-}[t - 1_k at X]) for every centre X and axis k")]
    for t in hrr.targets(am):
        i, j, kk, ll = (_comp(t[p], am[p]) for p in range(4))
        w = "w" + hrr_name(t)[1:]
        digest.append(_F(w, _c(Fn.ADD, _c(
            Fn.MUL, Var(dname((0, 1), i, j)), Var(dname((2, 3), kk, ll, "s_j"))),
            _c(Fn.ADD, _c(Fn.MUL, Var(dname((0, 2), i, kk, "s_k")),
                          Var(dname((1, 3), j, ll))),
               _c(Fn.MUL, Var(dname((0, 3), i, ll, "s_k")),
                  Var(dname((1, 2), j, kk)))))))
        for pos, X in enumerate(CENTRES):
            up = _hname(f"{X.lower()}p")
            down = _hname(f"{X.lower()}m")
            for a in range(3):
                term: Expr = Var(up(_moved(t, pos, a, 1)))
                if t[pos][a]:
                    term = _c(Fn.SUB, term, _c(
                        Fn.MUL, Lit(float(t[pos][a])),
                        Var(down(_moved(t, pos, a, -1)))))
                digest.append(_add_to(f"g{X}_{a}", _c(Fn.MUL, Var(w), term)))
    if set(by_tag) != {f"{X.lower()}{s}" for X, lx in zip(CENTRES, am,
                                                          strict=True)
                       for s in ("p", "m") if s == "p" or lx}:
        raise AssertionError(f"derivative classes {sorted(by_tag)}")

    work = (
        *(_F(Factor.cd(d).value, Index("CD", (k, Lit(d)))) for d in range(3)
          if Factor.cd(d).value in hneed),
        *(_I(_AO[s], Index("ket_ao", (k, Lit(s - 2)))) for s in (2, 3)),
        *(_I(_AT[s], Index("ket_at", (k, Lit(s - 2)))) for s in (2, 3)),
        Note(f"the contracted accumulators, zeroed per ket: {len(fams['0'])}"
             " lowered, " + ", ".join(f"{len(fams[X])} {X}-raised"
                                      for X in CENTRES)),
        *(_F(_acc(fam, e, f), Lit(0.0)) for fam in ("0", *CENTRES)
          for e, f in fams[fam]),
        Note("the K_CD x K_AB primitive nest: ket outer (global), bra inner "
             "(shared); one VRR over every derivative class's boundary"),
        For(KCD, Lit(0), Var("k_cd"), Lit(1), per_kcd),
        *(s for b in hbodies for s in b),
        *setup,
        *digest,
        Note("the bra's centres summed over the ket loop; C's and D's added "
             "now"),
        *(_add_to(f"s{X}_{a}", Var(f"g{X}_{a}")) for X in "AB"
          for a in range(3)),
        *(AtomicAdd(Index(G_OUT, (Var(_AT[p]), Lit(a))),
                    Var(f"g{CENTRES[p]}_{a}")) for p in (2, 3)
          for a in range(3)),
    )
    per_ket = (
        If(_c(Fn.LT, k, Var("kmin")), (Note(HALF),), (
            *(_I(_SH[s], Index("ket_sh", (k, Lit(s - 2)))) for s in (2, 3)
              if deg_ket),
            *work)),
    )
    per_bra = (
        *(_F(Factor.ab(d).value, Index("AB", (bra, Lit(d)))) for d in range(3)
          if Factor.ab(d).value in hneed),
        *(_I(_SH[s], Index("bra_sh", (bra, Lit(s)))) for s in (0, 1)
          if deg_bra),
        *(_I(_AO[s], Index("bra_ao", (bra, Lit(s)))) for s in (0, 1)),
        *(_I(_AT[s], Index("bra_at", (bra, Lit(s)))) for s in (0, 1)),
        *(_half_if_same("hb", "sa", "sb") if deg_bra else ()),
        Decl("kmin", Scalar.I32, init=_c(Fn.MUL, Var("diag"), bra)),
        Note("the bra's own density block, read once"),
        *block((0, 1)),
        *(_F(f"s{X}_{a}", Lit(0.0)) for X in "AB" for a in range(3)),
        Note(STAGING),
        For(STAGE, Lit(0), Var("k_ab"), Lit(1), tuple(
            Assign(Index(AB_SH, (Var(STAGE), Lit(f))),
                   Index("bra_prim", (bra, Var(STAGE), Lit(f))))
            for f in range(len(GREC)))),
        Note("one thread per contracted quartet: consecutive threads take "
             "consecutive |cd) ([Fock] §3.2)"),
        For(KET, Call(Fn.THREAD_X), Var("n_ket"), Call(Fn.BLOCK_DIM_X),
            per_ket),
        Note("flush the bra's centres: once per thread, after its ket loop"),
        *(AtomicAdd(Index(G_OUT, (Var(_AT[p]), Lit(a))),
                    Var(f"s{CENTRES[p]}_{a}")) for p in (0, 1)
          for a in range(3)),
    )
    body = (
        Note(f"dE_2e/dR by the HGP recurrence, F class ({la}{lb}|{lc}{ld}) "
             "in AO order at derivative order 1: kernel_f's mapping, one "
             "VRR to L + 1 over the derivative classes' boundaries, five "
             "weighted contracted families, one HRR per derivative class, "
             "the derived weights w = deg (4 D_ab D_cd - D_ac D_bd - D_ad "
             "D_bc)"),
        Note("launch: grid = n_bra, blockDim = (threads, 1); one block per "
             "bra shell pair; G zeroed by the caller, no close"),
        Decl("bra", Scalar.I32, init=Call(Fn.BLOCK_X)),
        Decl(AB_SH, Scalar.F64, (Lit(MAX_PRIM_PAIRS), Lit(len(GREC))),
             Space.SHARED),
        If(_c(Fn.LT, bra, Var("n_bra")), per_bra),
    )
    return Kernel(name or gf_name(am), _params(am), body)


def _half_if_same(flag: str, s: str, t: str) -> list[Stmt]:
    """`flag = 1/2` when shells `s == t`: neither `s < t` nor `t < s`."""
    return [_F(flag, Lit(0.5)),
            If(_c(Fn.LT, Var(s), Var(t)), (Assign(Var(flag), Lit(1.0)),)),
            If(_c(Fn.LT, Var(t), Var(s)), (Assign(Var(flag), Lit(1.0)),))]


def _params(am: AmClass) -> tuple[Param, ...]:
    la, lb, lc, ld = am
    nb, nk, kab, kcd = (Var(x) for x in ("n_bra", "n_ket", "k_ab", "k_cd"))
    nat, nao = Var("n_atom"), Var("n_ao")

    def f64(x: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(x, Scalar.F64, array=True, intent=intent, shape=shape)

    def i32(x: str, *shape: Expr) -> Param:
        return Param(x, Scalar.I32, array=True, shape=shape)

    out = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("k_ab", Scalar.I32), Param("k_cd", Scalar.I32),
        Param("n_atom", Scalar.I32), Param("n_ao", Scalar.I32),
        Param("diag", Scalar.I32),
        f64("bra_prim", nb, kab, Lit(len(GREC))), f64("AB", nb, Lit(3)),
        *((i32("bra_sh", nb, Lit(2)),) if la == lb else ()),
        i32("bra_ao", nb, Lit(2)), i32("bra_at", nb, Lit(2)),
        f64("ket_prim", nk, kcd, Lit(len(GREC))),
        f64("CD", nk, Lit(3)),
        *((i32("ket_sh", nk, Lit(2)),) if lc == ld else ()),
        i32("ket_ao", nk, Lit(2)), i32("ket_at", nk, Lit(2)),
        f64("D", nao, nao),
        f64(G_OUT, nat, Lit(3), intent=Intent.OUT),
    )
    assert tuple(p.name for p in out) == params_of(am)
    return out
