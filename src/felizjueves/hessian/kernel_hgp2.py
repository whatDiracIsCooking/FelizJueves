"""The two-electron Hessians on the HGP recurrence: `Target.F`, and `J` and
`K` on `Engine.HGP`, at `derivative_order = 2` -- `gradient.kernel_gf`'s
mapping differentiated twice.

    build_d2_kernel(spec)                 -> Kernel   one per class
    d2_name(spec)                         -> str      d2f_ / d2hj_ / d2hk_{..}
    d2f_spec / d2hj_spec / d2hk_spec(..)  -> Spec     the classes' d2 keys
    d2_spec(target, am)                   -> Spec     any of the three
    d2_weights(target)                    -> GradWeights, derived
    components() / second(t, comp)        -> the rule twice, symbolically
    derivative_terms(am)                  -> {(family, class)}: its HRRs
    params_of(spec)                       -> the parameter names, in order
    launch_shape(n_bra, threads)          -> (n_bra, (threads, 1))

THREE ENERGIES, ONE KERNEL SHAPE. With `D` symmetric and `(pq|rs)` an ERI
in AO order,

    E_J  = 1/2 SUM D_pq D_rs (pq|rs)                    Target.J (HGP)
    E_K  = 1/2 SUM D_pr D_qs (pq|rs)                    Target.K (HGP)
    E_2e = E_J - 1/2 E_K                                Target.F

and each kernel writes `H[X, k, Y, l] += d2 E / dR_{X,k} dR_{Y,l}` for a
FIXED `D` (the skeleton; no CPHF response: `felizjueves.hessian`'s WHAT A
HESSIAN KERNEL WRITES) -- the same `H` as `hessian.host.coulomb_hessian`,
`hessian.host.exchange_hessian` and `coulomb - 1/2 exchange`. As for
`hkx_`, the RHF factor of `E_K` is the caller's for `K`; `F` carries it.

THE CLASS IS THE ERI'S, IN AO ORDER, for all three: an F class is
`(la, lb | lc, ld)`; an HGP J class is the same quartet; an HGP K class
is K's pairing `(la, lc | lb, ld)` (`spec`: "A K class is (la, lc | lb,
ld) under BOTH engines: its ERI is (ac|bd)"), and that tuple is exactly
the ERI's AO order, so the kernel reads every `am` alike. The launch
stream is `gradient.interpreted.gf_launches`' (every unique quartet once;
`hessian.interpreted` runs it).

THE WEIGHTS, DERIVED (`d2_weights`) from `kernel_gf`'s orbit counts: over
the eight images of a representative the Coulomb pairing `D_ab D_cd` is
read 8 times and each exchange pairing (`D_ac D_bd`, `D_ad D_bc`) 4 times
(`gradient.kernel_gf.derive_gradient_weights` asserts the enumeration),
so per unique quartet and component

    w = deg (cJ D_ab D_cd + cK (D_ac D_bd + D_ad D_bc))
    F: (cJ, cK) = (8 x 1/2, 4 x -1/4) = (4, -1)   (kernel_gf's own)
    J: (8 x 1/2, 0) = (4, 0)       K: (0, 4 x 1/2) = (0, 2)

with `deg = 1/|Stab|` (`kernel_f`'s two shell tests and its diagonal
launch).

THE RULE, TWICE. `d/dX_k phi = 2x phi_{+1_k} - t_k phi_{-1_k}` on the
contracted integral, `x` the exponent of `X`'s primitive. Composing it
symbolically (`second`) gives, per ERI target `t`, centres `X`, `Y` at AO
positions `p <= q` and axes `k`, `l`:

    p != q:          [X+k Y+l]^{2x 2y} - t_Yl [X+k Y-l]^{2x}
                     - t_Xk [X-k Y+l]^{2y} + t_Xk t_Yl [X-k Y-l]
    p = q, k != l:   [X+k+l]^{4x^2} - t_l [X+k-l]^{2x}
                     - t_k [X-k+l]^{2x} + t_k t_l [X-k-l]
    p = q, k = l:    [X+2k]^{4x^2} - (2 t_k + 1) [X]^{2x}
                     + t_k (t_k - 1) [X-2k]

where `[..]^{w}` is the contracted integral with every primitive quartet
weighted by `w`. So the kernel reads up to FIFTEEN WEIGHT FAMILIES -- `1`,
`2x` per centre, `2x 2y` per unordered pair of centres, squares included
-- each over the boundaries of the shifted classes it feeds. As in `gf_`
the HRR (eq 28) holds on contracted values with any primitive weights, so
per primitive quartet ONE VRR runs, to Boys order `L + 2`, over the union
of every family's boundary; each family accumulates `w [e0|f0]^(0)`; and
after the nest one HRR per `(family, shifted class)` (`derivative_terms`)
turns its boundary into that class's ERIs.

NO TRANSLATIONAL INVARIANCE IS USED: all four centres are differentiated
(a representative stands for members whose bra centres are its ket
centres, `gf_`'s reason), so `SUM_X H[X, k, Y, l] = 0` and `H = H^T`
stay checks.

THE WRITE. Per centre pair `p < q` nine sums, each into `H[X, k, Y, l]`
and its transpose `H[Y, l, X, k]`; per `p = q` six (`k <= l`), the
off-diagonal mirrored -- 78 sums. The bra-bra ones (`AA`, `AB`, `BB`) are
carried over the thread's ket loop and added once after it (36
`AtomicAdd`s), the rest per quartet (108). `H[n_atom, 3, n_atom, 3]` is
ZEROED BY THE CALLER and carried through every launch.

THE MAPPING, THE RECORD, THE NEST, THE SCALARS are `kernel_gf`'s,
imported: one block per bra shell pair, one thread per contracted
quartet, ket primitives outer (global) and bra inner (shared), the
canonical half skipped, `GREC` with the pair's two exponents. NO SCREEN
(an integral's bound does not bound its second derivative) and NO LEVER.

A builder: vendor-neutral IR. It imports neither `ao/` nor the hosts nor
the interpreter nor `reference/` (test/test_layering.py, rule
`hgp2_builder`).
"""

from __future__ import annotations

from collections.abc import Sequence

from felizjueves.gradient import kernel_gf as GF
from felizjueves.gradient.kernel_gf import (
    AB_SH,
    CENTRES,
    GREC,
    KAB,
    KCD,
    KET,
    MAX_PRIM_PAIRS,
    STAGE,
    STAGING,
    GradWeights,
)
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import (
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
)
from felizjueves.spec import Engine, Operator, Spec, Target

__all__ = [
    "H_OUT",
    "HGP2_TARGETS",
    "build_d2_kernel",
    "components",
    "d2_name",
    "d2_spec",
    "d2_weights",
    "d2f_spec",
    "d2hj_spec",
    "d2hk_spec",
    "derivative_terms",
    "launch_shape",
    "params_of",
    "second",
]

H_OUT = "H"  # n_atom x 3 x n_atom x 3, AtomicAdd-ed, zeroed by the caller
_SH = ("sa", "sb", "sc", "sd")
_AO = ("oa", "ob", "oc", "od")
_AT = ("ta", "tb", "tc", "td")
_P, _X, _ZETA, _U, _E1, _E2 = 0, 3, 6, 7, 8, 9
_AXES = "xyz"
_PREFIX = {Target.F: "d2f_", Target.J: "d2hj_", Target.K: "d2hk_"}
HGP2_TARGETS = tuple(_PREFIX)
Fam = tuple[int, ...]  # sorted AO positions whose 2 x exponent weights it
Target4 = tuple[Cart, Cart, Cart, Cart]
Comp = tuple[int, int, int, int]  # (p, k, q, l): p <= q, k <= l when p == q


# --- the weights, derived (module docstring) -----------------------------------

_GAMMA = {Target.F: (GF.COULOMB_GAMMA, GF.EXCHANGE_GAMMA),
          Target.J: (0.5, 0.0), Target.K: (0.0, 0.5)}


def d2_weights(target: Target) -> GradWeights:
    """`(cJ, cK)` of the target's energy from `kernel_gf`'s orbit counts (8
    Coulomb images, 4 per exchange pairing), times its `Gam` fractions."""
    if target not in _GAMMA:
        raise ValueError(f"target {target} has no HGP two-electron Hessian")
    n_j = GF.WEIGHTS.coulomb / GF.COULOMB_GAMMA
    n_k = GF.WEIGHTS.exchange / GF.EXCHANGE_GAMMA
    if (n_j, n_k) != (8.0, 4.0):
        raise AssertionError(f"orbit counts {(n_j, n_k)}, not (8, 4)")
    gj, gk = _GAMMA[target]
    return GradWeights(n_j * gj, n_k * gk)


# --- the keys and names ----------------------------------------------------------

def d2f_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The d2 F class `(la lb | lc ld)`, AO order, HGP (F's native)."""
    return Spec(Operator.COULOMB, 2, (la, lb, lc, ld), Target.F)


def d2hj_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The d2 HGP J class `(la lb | lc ld)`."""
    return Spec(Operator.COULOMB, 2, (la, lb, lc, ld), Target.J,
                engine=Engine.HGP)


def d2hk_spec(la: int, lc: int, lb: int, ld: int) -> Spec:
    """The d2 HGP K class `(la lc | lb ld)`, K's pairing: the ERI (ac|bd)."""
    return Spec(Operator.COULOMB, 2, (la, lc, lb, ld), Target.K,
                engine=Engine.HGP)


_SPEC = {Target.F: d2f_spec, Target.J: d2hj_spec, Target.K: d2hk_spec}


def d2_spec(target: Target, am: Sequence[int]) -> Spec:
    """The d2 key of `target` (F, J or K) on HGP at class `am`."""
    if target not in _SPEC:
        raise ValueError(f"target {target} has no HGP two-electron Hessian")
    return _SPEC[target](*am)


def _class(spec: Spec) -> tuple[Target, AmClass]:
    if not isinstance(spec, Spec):
        raise TypeError(f"build_d2_kernel takes a Spec, not {spec!r}")
    if (spec.target not in _PREFIX or spec.derivative_order != 2
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d2 F, J or K key on "
                         "Engine.HGP")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the HGP two-electron Hessian takes "
                         f"no lever, not {spec.levers[0]!r}")
    if spec.n_batches != 1:
        raise ValueError(f"{spec.key()}: an HGP class has no Hermite targets "
                         "to batch; n_batches must be 1")
    return spec.target, tuple(spec.am)  # type: ignore[return-value]


def d2_name(spec: Spec) -> str:
    """`d2f_` / `d2hj_` / `d2hk_` and the four `l`s in the key's order,
    `_`-separated once any `l` is above 9."""
    target, am = _class(spec)
    sep = "_" if any(x > 9 for x in am) else ""
    return _PREFIX[target] + sep.join(map(str, am))


def launch_shape(n_bra: int, threads: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (threads, 1))`: `kernel_f`'s mapping."""
    return GF.launch_shape(n_bra, threads)


def params_of(spec: Spec) -> tuple[str, ...]:
    """The parameter names, in order: `gf_`'s, with `H` for `G`."""
    _, (la, lb, lc, ld) = _class(spec)
    return ("n_bra", "n_ket", "k_ab", "k_cd", "n_atom", "n_ao", "diag",
            "bra_prim", "AB", *(("bra_sh",) if la == lb else ()), "bra_ao",
            "bra_at", "ket_prim", "CD", *(("ket_sh",) if lc == ld else ()),
            "ket_ao", "ket_at", "D", H_OUT)


# --- the second derivative, composed symbolically -------------------------------

def _moved(t: Target4, pos: int, axis: int, by: int) -> Target4:
    return tuple(tuple(c + by * (q == pos and d == axis)
                       for d, c in enumerate(v)) for q, v in enumerate(t))  # type: ignore[return-value]


def _d1(terms: dict, pos: int, axis: int) -> dict:
    """The rule once at AO position `pos`, axis `axis`, on `{(family,
    target): coefficient}`: `+ (family + pos, t + 1_k)`, `- t_k (family,
    t - 1_k)`; like terms combined, zeros dropped."""
    out: dict = {}
    for (fam, t), c in terms.items():
        up = (tuple(sorted((*fam, pos))), _moved(t, pos, axis, 1))
        out[up] = out.get(up, 0) + c
        n = t[pos][axis]
        if n:
            dn = (fam, _moved(t, pos, axis, -1))
            out[dn] = out.get(dn, 0) - c * n
    return {k: v for k, v in out.items() if v}


def components() -> tuple[Comp, ...]:
    """The 78 `(p, k, q, l)` the kernel sums: 9 per centre pair `p < q`, 6
    per `p = q` (`k <= l`)."""
    return tuple((p, k, q, l) for p in range(4) for q in range(p, 4)
                 for k in range(3) for l in range(3)  # noqa: E741
                 if p < q or k <= l)


def second(t: Target4, comp: Comp) -> dict[tuple[Fam, Target4], int]:
    """`d/dX_k d/dY_l` of ERI target `t` as `{(family, shifted target):
    coefficient}` (module docstring, THE RULE, TWICE)."""
    p, k, q, l = comp  # noqa: E741
    return _d1(_d1({((), t): 1}, p, k), q, l)


def _cls_of(t: Target4) -> AmClass:
    return tuple(sum(v) for v in t)  # type: ignore[return-value]


def derivative_terms(am: Sequence[int]) -> tuple[tuple[Fam, AmClass], ...]:
    """Every `(family, class)` the class's second derivatives read, sorted:
    the HRRs the kernel runs."""
    am = tuple(am)
    seen: set = set()
    for t in hrr.targets(am):  # type: ignore[arg-type]
        for comp in components():
            for fam, u in second(t, comp):
                seen.add((fam, _cls_of(u)))
    return tuple(sorted(seen))


# --- the kernel --------------------------------------------------------------------

def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _F(name: str, e: Expr) -> Decl:
    return Decl(name, Scalar.F64, init=e)


def _I(name: str, e: Expr) -> Decl:
    return Decl(name, Scalar.I32, init=e)


def _add_to(name: str, e: Expr) -> Assign:
    return Assign(Var(name), _c(Fn.ADD, Var(name), e))


def _ftag(fam: Fam) -> str:
    return "w" + "".join(CENTRES[p].lower() for p in fam)


def _acc(fam: Fam, e: Cart, f: Cart) -> str:
    return f"c{_ftag(fam)}_" + "__".join("_".join(map(str, v)) for v in (e, f))


def _htag(fam: Fam, cls: AmClass) -> str:
    return f"{_ftag(fam)}{''.join(map(str, cls))}_"


def _hname(tag: str):
    return lambda node: tag + hrr_name(node)


def _comp(v: Cart, l: int) -> int:  # noqa: E741
    return vrr.cartesians(l).index(v)


def _boundary(cls: AmClass) -> tuple[tuple[Cart, Cart], ...]:
    return tuple((e, f) for e, f, _ in vrr.targets(cls))


def _h(comp: Comp) -> str:
    """This quartet's sum of component `comp`."""
    p, k, q, l = comp  # noqa: E741
    return f"h{CENTRES[p]}{CENTRES[q]}_{_AXES[k]}{_AXES[l]}"


def _s(comp: Comp) -> str:
    """The thread's carried sum of a bra-bra component."""
    return "s" + _h(comp)[1:]


def _wname(fam: Fam) -> str:
    return "W" + "".join(CENTRES[p].lower() for p in fam)


def build_d2_kernel(spec: Spec, *, name: str | None = None,
                    weights: GradWeights | None = None) -> Kernel:
    """The HGP two-electron Hessian kernel of one d2 F, J.hgp or K.hgp class
    (module docstring). `weights` is `d2_weights(spec.target)` unless a
    test substitutes a wrong one to watch it fail."""
    target, am = _class(spec)
    if weights is None:
        weights = d2_weights(target)
    if not isinstance(weights, GradWeights):
        raise TypeError(f"weights {weights!r} is not a GradWeights")
    la, lb, lc, ld = am
    pairs = derivative_terms(am)
    dags = {fc: hrr.build(fc[1]) for fc in pairs}
    fams: dict[Fam, list[tuple[Cart, Cart]]] = {}
    for fam, cls in pairs:
        bnd = _boundary(cls)
        if set(bnd) != dags[(fam, cls)].boundary:  # #274; refuse, never misread
            raise ValueError(f"{cls}: HRR boundary is not the VRR target set")
        have = fams.setdefault(fam, [])
        have += [ef for ef in bnd if ef not in have]
    roots = sorted({(e, f, 0) for bnd in fams.values() for e, f in bnd},
                   key=vrr.order_key)
    vdag = GF._union_vrr(roots)
    prelude, fundamental = fundamental_prelude(vdag.cls[0])
    vbody = tuple(vrr_body(vdag, fundamental))
    vf = GF._factors(vbody)
    vneed = frozenset(vf | {i2 for n, i2, _ in GF._DERIVED if n in vf})
    hbodies = [tuple(hrr_body(dags[(fam, cls)], lambda e, f, fam=fam: Var(
        _acc(fam, e, f)), name=_hname(_htag(fam, cls)))) for fam, cls in pairs]
    hneed = frozenset(set().union(*(GF._factors(b) for b in hbodies)))
    n = tuple(len(vrr.cartesians(x)) for x in am)
    deg_bra, deg_ket, deg_pq = la == lb, lc == ld, (la, lb) == (lc, ld)
    bra, k = Var("bra"), Var(KET)
    use_j, use_k = weights.coulomb != 0.0, weights.exchange != 0.0
    if not (use_j or use_k):
        raise ValueError("weights (0, 0): nothing to differentiate")

    def bra_prim(f: int) -> Expr:
        return Index(AB_SH, (Var(KAB), Lit(f)))

    def ket_prim(f: int) -> Expr:
        return Index("ket_prim", (k, Var(KCD), Lit(f)))

    # -- the primitive nest: ket outer (global), bra inner (shared) ----------
    wdecl: list[Stmt] = []
    for fam in fams:
        if len(fam) == 2:
            wdecl.append(_F(_wname(fam), _c(
                Fn.MUL, Var("E" + CENTRES[fam[0]].lower()),
                Var("E" + CENTRES[fam[1]].lower()))))
    accumulate: list[Stmt] = [Note(
        "eq 27 per family: c<w> += w [e0|f0]^(0) over the boundary of every "
        "class the family feeds; w = 1, 2x, or 2x 2y")]
    for fam, efs in fams.items():
        wv: Expr | None = (None if not fam else Var(
            "E" + CENTRES[fam[0]].lower()) if len(fam) == 1
            else Var(_wname(fam)))
        for e, f in efs:
            v: Expr = Var(vrr_name((e, f, 0)))
            accumulate.append(_add_to(_acc(fam, e, f),
                                      v if wv is None else _c(Fn.MUL, wv, v)))
    per_kab = (
        Note("the bra primitive pair, from shared memory"),
        *(_F(f"Pb{d}", bra_prim(_P + d)) for d in range(3)),
        *(_F(Factor.pa(d).value, bra_prim(_X + d)) for d in range(3)
          if Factor.pa(d).value in vneed),
        _F("p", bra_prim(_ZETA)),
        _F("Ub", bra_prim(_U)),
        _F("Ea", _c(Fn.MUL, Lit(2.0), bra_prim(_E1))),
        _F("Eb", _c(Fn.MUL, Lit(2.0), bra_prim(_E2))),
        Note("the two-centre family weights of this primitive quartet"),
        *wdecl,
        *GF._quartet_scalars(vneed),
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
    comps = components()
    bra_comps = [c for c in comps if c[2] < 2]
    ket_comps = [c for c in comps if c[2] >= 2]
    setup: list[Stmt] = [
        Note("the degeneracy factor 1/|Stab| (kernel_f): 1/2 for a = b, for "
             "c = d, and for P = Q"),
        *(GF._half_if_same("hk", "sc", "sd") if deg_ket else ()),
        _F("deg", deg_init),
    ]
    if deg_pq:
        setup.append(If(_c(Fn.LT, k, _c(Fn.ADD, Var("kmin"), Var("diag"))),
                        (Note("P = Q: the diagonal launch at k == bra"),
                         Assign(Var("deg"), _c(Fn.MUL, Var("deg"),
                                               Lit(0.5))))))
    if use_j:
        setup.append(_F("s_j", _c(Fn.MUL, Lit(float(weights.coulomb)),
                                  Var("deg"))))
    if use_k:
        setup.append(_F("s_k", _c(Fn.MUL, Lit(float(weights.exchange)),
                                  Var("deg"))))
    setup.append(Note(
        "the density blocks the weights read: w_t = "
        + " + ".join(x for x, on in (("D_ab (s_j D_cd)", use_j),
                                      ("(s_k D_ac) D_bd + (s_k D_ad) D_bc",
                                       use_k)) if on)
        + ", deg and the derived coefficients folded in once per quartet"))
    if use_j:
        setup += block((2, 3), "s_j")
    if use_k:
        setup += [*block((0, 2), "s_k"), *block((0, 3), "s_k"),
                  *block((1, 3)), *block((1, 2))]
    setup += [Note("this quartet's 78 component sums, zeroed"),
              *(_F(_h(c), Lit(0.0)) for c in comps)]
    digest: list[Stmt] = [Note(
        "per ERI t: w_t, then per centre pair and axis pair h += w_t x (the "
        "rule twice: SUM coefficient x the HRR value of its family's "
        "shifted class)")]
    for t in hrr.targets(am):
        i, j, kk, ll = (_comp(t[p], am[p]) for p in range(4))
        w = "w" + hrr_name(t)[1:]
        terms: list[Expr] = []
        if use_j:
            terms.append(_c(Fn.MUL, Var(dname((0, 1), i, j)),
                            Var(dname((2, 3), kk, ll, "s_j"))))
        if use_k:
            terms += [_c(Fn.MUL, Var(dname((0, 2), i, kk, "s_k")),
                         Var(dname((1, 3), j, ll))),
                      _c(Fn.MUL, Var(dname((0, 3), i, ll, "s_k")),
                         Var(dname((1, 2), j, kk)))]
        wexpr = terms[0]
        for x in terms[1:]:
            wexpr = _c(Fn.ADD, wexpr, x)
        digest.append(_F(w, wexpr))
        for comp in comps:
            expr: Expr | None = None
            for (fam, u), coef in sorted(second(t, comp).items()):
                val: Expr = Var(_hname(_htag(fam, _cls_of(u)))(u))
                if coef != 1:
                    val = _c(Fn.MUL, Lit(float(coef)), val)
                expr = val if expr is None else _c(Fn.ADD, expr, val)
            if expr is not None:
                digest.append(_add_to(_h(comp), _c(Fn.MUL, Var(w), expr)))

    def h_index(p: int, x: int, q: int, y: int) -> Index:
        return Index(H_OUT, (Var(_AT[p]), Lit(x), Var(_AT[q]), Lit(y)))

    def flush(c: Comp, v: Expr) -> list[Stmt]:
        p, x, q, y = c
        out: list[Stmt] = [AtomicAdd(h_index(p, x, q, y), v)]
        if p != q or x != y:
            out.append(AtomicAdd(h_index(q, y, p, x), v))
        return out

    work = (
        *(_F(Factor.cd(d).value, Index("CD", (k, Lit(d)))) for d in range(3)
          if Factor.cd(d).value in hneed),
        *(_I(_AO[s], Index("ket_ao", (k, Lit(s - 2)))) for s in (2, 3)),
        *(_I(_AT[s], Index("ket_at", (k, Lit(s - 2)))) for s in (2, 3)),
        Note(f"the contracted accumulators, zeroed per ket: {len(fams)} "
             f"weight families over {len(pairs)} shifted classes"),
        *(_F(_acc(fam, e, f), Lit(0.0)) for fam, efs in fams.items()
          for e, f in efs),
        Note("the K_CD x K_AB primitive nest: ket outer (global), bra inner "
             "(shared); one VRR to L + 2 over every family's boundary"),
        For(KCD, Lit(0), Var("k_cd"), Lit(1), per_kcd),
        *(s for b in hbodies for s in b),
        *setup,
        *digest,
        Note("the bra-bra sums carried over the ket loop; the pairs with a "
             "ket centre added now"),
        *(_add_to(_s(c), Var(_h(c))) for c in bra_comps),
        *(st for c in ket_comps for st in flush(c, Var(_h(c)))),
    )
    per_ket = (
        If(_c(Fn.LT, k, Var("kmin")), (Note(GF.HALF),), (
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
        *(GF._half_if_same("hb", "sa", "sb") if deg_bra else ()),
        Decl("kmin", Scalar.I32, init=_c(Fn.MUL, Var("diag"), bra)),
        *((Note("the bra's own density block, read once"), *block((0, 1)))
          if use_j else ()),
        *(_F(_s(c), Lit(0.0)) for c in bra_comps),
        Note(STAGING),
        For(STAGE, Lit(0), Var("k_ab"), Lit(1), tuple(
            Assign(Index(AB_SH, (Var(STAGE), Lit(f))),
                   Index("bra_prim", (bra, Var(STAGE), Lit(f))))
            for f in range(len(GREC)))),
        Note("one thread per contracted quartet: consecutive threads take "
             "consecutive |cd) ([Fock] §3.2)"),
        For(KET, Call(Fn.THREAD_X), Var("n_ket"), Call(Fn.BLOCK_DIM_X),
            per_ket),
        Note("flush the bra-bra sums: once per thread, after its ket loop"),
        *(st for c in bra_comps for st in flush(c, Var(_s(c)))),
    )
    what = {Target.F: "d2 E_2e (E_J - 1/2 E_K)", Target.J: "d2 E_J",
            Target.K: "d2 E_K"}[target]
    body = (
        Note(f"{what} / dR dR by the HGP recurrence, class "
             f"({la}{lb}|{lc}{ld}) in the ERI's AO order: kernel_gf's "
             "mapping, one VRR to L + 2 over every weight family's boundary, "
             "one HRR per (family, shifted class), the rule applied twice "
             "on all four centres (hessian/kernel_hgp2.py)"),
        Note("launch: grid = n_bra, blockDim = (threads, 1); one block per "
             "bra shell pair; H zeroed by the caller"),
        Decl("bra", Scalar.I32, init=Call(Fn.BLOCK_X)),
        Decl(AB_SH, Scalar.F64, (Lit(MAX_PRIM_PAIRS), Lit(len(GREC))),
             Space.SHARED),
        If(_c(Fn.LT, bra, Var("n_bra")), per_bra),
    )
    return Kernel(name or d2_name(spec), _params(spec), body)


def _params(spec: Spec) -> tuple[Param, ...]:
    la, lb, lc, ld = spec.am
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
        f64(H_OUT, nat, Lit(3), nat, Lit(3), intent=Intent.OUT),
    )
    assert tuple(p.name for p in out) == params_of(spec)
    return out
