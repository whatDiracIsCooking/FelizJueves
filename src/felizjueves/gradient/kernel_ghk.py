"""The exchange-energy gradient on the HGP recurrence: `Target.K` at
`derivative_order = 1` on `Engine.HGP`.

    build_ghk_kernel(spec | (la, lc, lb, ld)) -> Kernel   one per K class
    ghk_name(spec | (la, lc, lb, ld))         -> str      ghk_{la}{lc}_{lb}{ld}
    ghk_spec(la, lc, lb, ld)                  -> Spec     the class's d1 key
    vrr_class(am) / hrr_classes(am)           -> the classes it builds
    launch_shape(n_bra, threads)              -> (n_bra, (threads, 1))
    params_of(spec | am)                      -> the parameter names, in order

THE ENERGY AND THE SWEEP ARE `kernel_gk`'s (its docstring is the
specification): `E_K = 1/2 SUM_abcd D_ab D_cd (ac|bd)`, K's Hermite pairing
`am = (la, lc, lb, ld)`, and for a SYMMETRIC `D` the ket derivative equals
the bra derivative, so

    dE_K / dR_{X,k} = SUM_{ordered (a,c), (b,d) shell pairs}
                      SUM_abcd D_ab D_cd d/dX_k (ac|bd),    X = A, C

the same `G` as `kernel_gk` and `gradient.host.exchange_gradient`. The RHF
exchange energy is `-1/2 E_K`: the factor is the caller's.

THE DERIVATIVE, ON HGP. A primitive's centre derivative raises or lowers
the shell's angular momentum (`d/dA_k x^i e^{-a x^2} = 2a x^{i+1} - i
x^{i-1}`), so on contracted integrals

    d/dA_k (ac|bd) = (a+1_k c|bd)_{2 alpha} - a_k (a-1_k c|bd)
    d/dC_k (ac|bd) = (a c+1_k|bd)_{2 gamma} - c_k (a c-1_k|bd)

where `(..)_{2 alpha}` is the contraction with every primitive quartet
weighted by twice its `A` exponent. The HRR (eq 28) holds for any
contraction weights that depend on the primitive alone -- `AB` is the same
for every primitive pair -- so each of the three contractions is an
ordinary HGP boundary set, accumulated in registers inside the `K_AB x
K_CD` nest, and the HRR runs on it afterwards (milestone 12, pin 5):

    one VRR    `vrr_class(am) = (e_lo, la + lc + 1 - e_lo | lb, ld)`, `e_lo =
               max(la - 1, 0)`: its targets cover every boundary below
    T(e,f)     the ket-inner sum of `[e0|f0]^(0)` per bra primitive pair,
               zeroed per `kab` (`kernel_hj.hgp_nest`'s accumulators)
    after the ket loop, per bra primitive pair (exponents `alpha`,
    `gamma` read from `bra_exp`):
        UA += 2 alpha T   over the boundary of (la + 1, lc | lb, ld)
        UC += 2 gamma T   over the boundary of (la, lc + 1 | lb, ld)
        U0 +=         T   over the boundaries of (la - 1, lc | ..) and
                          (la, lc - 1 | ..), where those exist
    four HRRs  (la+1, lc), (la, lc+1) on UA / UC, and (la-1, lc),
               (la, lc-1) on U0 (`hrr_classes`), each on its own locals

so no derivative is formed on a primitive and nothing but the weights and
the deposit is new: the VRR, the quartet scalars, the staging and the nest
are `kernel_hj`'s, imported. A weight multiplies once per bra primitive
pair (the ket-inner `T` sum first), not once per quartet.

THE DEPOSIT. Per ERI target `(a c|b d)` of the class, `W = M[a, b]
M[c, d]` (both read at the records' AO offsets, `kernel_gk`'s addressing),
and per axis `k`

    gA_k += W ((a+1_k c|bd)_{2 alpha} - a_k (a-1_k c|bd))
    gC_k += W ((a c+1_k|bd)_{2 gamma} - c_k (a c-1_k|bd))

into thread-private sums carried over the thread's kets; after the ket
loop each thread adds its six sums into `G[atom_a, k]` and `G[atom_c, k]`
by `AtomicAdd`. `G` is ZEROED BY THE CALLER; every launch only adds.

THE MAPPING IS `kernel_hj`'s, [Fock] §3.2: one block per bra shell pair
(the record staged in shared memory, every thread copying it, #333), one
thread per contracted quartet, the kets strided over `threadIdx.x`, the
primitive nest bra outer and ket inner. One class mode only, the class's
own recurrence (`kernel_hj`'s `ordered`): the derivative is taken on the
bra's two centres, which an 8-fold orbit representative would move. No
lever.

NO SCREEN, as `kernel_gk`: the K bound bounds the integral, not its
derivative. A launch is one class, every ordered bra pair `(a, c)` of
`(la, lc)` against every ordered ket pair `(b, d)` of `(lb, ld)` of one
contraction degree each (`gradient.host.hgp_exchange_launches` packs them,
one launch per pair of degrees), so the class's launches over a basis sum
to its share of `dE_K`.

    params  n_bra n_ket k_ab k_cd n_ao n_atom (I32)
            bra_prim[n_bra, k_ab, REC]   kernel_f's record of (a, c)
            bra_exp[n_bra, k_ab, 2]      (alpha, gamma) per primitive pair
            AB[n_bra, 3]                 A - C, the bra HRR's separation
            bra_ao  bra_atom  [n_bra, 2] (I32)
            ket_prim[n_ket, k_cd, REC]   kernel_f's record of (b, d)
            CD[n_ket, 3]                 B - D, only when ld > 0
            ket_ao[n_ket, 2]             (I32)
            M[n_ao, n_ao]                the density D
            G[n_atom, 3]                 (OUT, read and AtomicAdd-ed)

`REC` is `kernel_f`'s `(P, P - A, zeta, U)` with `U = c_a c_c (pi /
zeta)^{3/2} K_ac`, `eri.hgp_drive.primitive_scalars`' binding.

ROUND-OFF, NOT BITWISE, against `gradient.host.exchange_gradient`, the MMD
host reference (another recurrence, another contraction order); the bound
is stated in test/test_kernel_ghk.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
nor the interpreter nor `reference/` (test/test_layering.py, rule
`ghk_builder`).
"""

from __future__ import annotations

from felizjueves.eri.fock.kernel_f import MAX_PRIM_PAIRS, REC
from felizjueves.eri.hjk import kernel_hj as HJ
from felizjueves.eri.hjk.kernel_hj import (
    _P,
    _U,
    _X,
    _ZETA,
    AB_SH,
    KAB,
    KCD,
    KET,
    STAGE,
    STAGING,
    acc_name,
)
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import boundary_order, hrr_body, hrr_name
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
    "AB_SH",
    "G_OUT",
    "MAX_PRIM_PAIRS",
    "REC",
    "build_ghk_kernel",
    "ghk_name",
    "ghk_spec",
    "hrr_classes",
    "launch_shape",
    "params_of",
    "vrr_class",
]

Am = tuple[int, int, int, int]
G_OUT = "G"
_AXES = "xyz"

# the three contracted boundary sets (module docstring)
_UA, _UC, _U0 = "UA", "UC", "U0"

NO_SCREEN = ("no screen: the K bound bounds the integral, not its "
             "derivative (kernel_gk's NO SCREEN); every ordered quartet of "
             "the launch is visited")


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _F(name: str, e: Expr) -> Decl:
    return Decl(name, Scalar.F64, init=e)


def _I(name: str, e: Expr) -> Decl:
    return Decl(name, Scalar.I32, init=e)


def _add(x: Expr, y: Expr) -> Call:
    return _c(Fn.ADD, x, y)


def _mul(x: Expr, y: Expr) -> Call:
    return _c(Fn.MUL, x, y)


# --- the key -----------------------------------------------------------------


def ghk_spec(la: int, lc: int, lb: int, ld: int) -> Spec:
    """The HGP K class `(la, lc | lb, ld)` at derivative order 1."""
    return Spec(Operator.COULOMB, 1, (la, lc, lb, ld), Target.K,
                engine=Engine.HGP)


def _class(cls: Spec | Am) -> Am:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = ghk_spec(*cls)
    else:
        raise TypeError("build_ghk_kernel takes (la, lc, lb, ld) or a Spec, "
                        f"not {cls!r}")
    if (spec.target is not Target.K or spec.derivative_order != 1
            or spec.engine is not Engine.HGP):
        raise ValueError(f"{spec.key()} is not a d1 Target.K key on "
                         "Engine.HGP")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the HGP exchange gradient takes no "
                         "levers")
    if spec.n_batches != 1:
        raise ValueError(f"{spec.key()}: the HGP exchange gradient has no "
                         "Hermite targets to batch; n_batches must be 1")
    la, lc, lb, ld = spec.am
    return la, lc, lb, ld


def ghk_name(cls: Spec | Am) -> str:
    """`ghk_{la}{lc}_{lb}{ld}`, `kernel_gk`'s spelling; `_`-separated once
    any `l` is above 9."""
    la, lc, lb, ld = _class(cls)
    if max(la, lc, lb, ld) > 9:
        return f"ghk_{la}_{lc}__{lb}_{ld}"
    return f"ghk_{la}{lc}_{lb}{ld}"


def launch_shape(n_bra: int, threads: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (threads, 1))`: one block per bra shell pair, `kernel_hj`'s."""
    return HJ.launch_shape(n_bra, threads)


def vrr_class(cls: Spec | Am) -> Am:
    """The one VRR the kernel runs: `(e_lo, la + lc + 1 - e_lo | lb, ld)`,
    `e_lo = max(la - 1, 0)`, whose targets are every `(e, f)` with `e_lo <=
    |e| <= la + lc + 1`, `lb <= |f| <= lb + ld` (module docstring)."""
    la, lc, lb, ld = _class(cls)
    lo = max(la - 1, 0)
    return lo, la + lc + 1 - lo, lb, ld


def hrr_classes(cls: Spec | Am) -> dict[str, tuple[str, Am]]:
    """The HRRs the kernel runs, by role: `"a+"` / `"c+"` on the weighted
    sets, `"a-"` / `"c-"` on the plain one, a lowered one only where its
    shell has a unit to lower. Each is `(boundary set tag, class)`."""
    la, lc, lb, ld = _class(cls)
    out = {"a+": (_UA, (la + 1, lc, lb, ld)), "c+": (_UC, (la, lc + 1, lb, ld))}
    if la:
        out["a-"] = (_U0, (la - 1, lc, lb, ld))
    if lc:
        out["c-"] = (_U0, (la, lc - 1, lb, ld))
    return out


def params_of(cls: Spec | Am) -> tuple[str, ...]:
    """The parameter names, in order; `CD` only when `ld > 0`."""
    ld = _class(cls)[3]
    return ("n_bra", "n_ket", "k_ab", "k_cd", "n_ao", "n_atom",
            "bra_prim", "bra_exp", "AB", "bra_ao", "bra_atom",
            "ket_prim", *(("CD",) if ld else ()), "ket_ao", "M", G_OUT)


# --- the pieces --------------------------------------------------------------


def _set_name(tag: str, e, f) -> str:
    return tag + acc_name(e, f)


_HRR_PREFIX = {"a+": "ha", "c+": "hc", "a-": "hm", "c-": "hn"}


def _hrr_local(role: str):
    pre = _HRR_PREFIX[role]

    def name(node) -> str:
        return pre + hrr_name(node)

    return name


def _shift(v, k: int, by: int):
    return tuple(c + by * (i == k) for i, c in enumerate(v))


def _ao(base: str, i: int) -> Expr:
    return _add(Var(base), Lit(i))


# --- the kernel --------------------------------------------------------------


def build_ghk_kernel(cls: Spec | Am, *, name: str | None = None) -> Kernel:
    """The HGP exchange-gradient kernel of one K class (module docstring)."""
    am = _class(cls)
    la, lc, lb, ld = am
    vam = vrr_class(am)
    bnd = boundary_order(vam)
    roles = hrr_classes(am)
    hdags = {r: hrr.build(c) for r, (_, c) in roles.items()}
    need: dict[str, set] = {_UA: set(), _UC: set(), _U0: set()}
    for r, (tag, _) in roles.items():
        need[tag] |= hdags[r].boundary
    if not set().union(*need.values()) <= set(bnd):
        raise ValueError(f"{am}: the VRR {vam} does not reach every HRR "
                         "boundary")
    sets = {tag: tuple(ef for ef in bnd if ef in s) for tag, s in need.items()}
    nest = HJ.hgp_nest(vam, bnd)
    vneed = nest.need
    hbodies = {r: tuple(hrr_body(
        hdags[r], lambda e, f, t=roles[r][0]: Var(_set_name(t, e, f)),
        name=_hrr_local(r))) for r in roles}
    hneed = set().union(*(HJ._factors(b) for b in hbodies.values()))
    bra, k = Var("bra"), Var(KET)

    def bra_prim(f: int) -> Expr:
        return Index(AB_SH, (Var(KAB), Lit(f)))

    def ket_prim(f: int) -> Expr:
        return Index("ket_prim", (k, Var(KCD), Lit(f)))

    # -- the primitive nest, bra outer ---------------------------------------
    per_kcd = (
        Note("the ket primitive pair, from global memory"),
        *(_F(f"Qk{d}", ket_prim(_P + d)) for d in range(3)),
        *(_F(Factor.qc(d).value, ket_prim(_X + d)) for d in range(3)
          if Factor.qc(d).value in vneed),
        _F("q", ket_prim(_ZETA)),
        _F("Uq", ket_prim(_U)),
        *nest.body,
    )
    weights = {_UA: Var("ta"), _UC: Var("tc"), _U0: None}
    fold: list[Stmt] = [
        Note("the bra primitive pair's weights: 2 alpha on the a-raised "
             "set, 2 gamma on the c-raised one, 1 on the lowered ones"),
        _F("ta", _mul(Lit(2.0), Index("bra_exp", (bra, Var(KAB), Lit(0))))),
        _F("tc", _mul(Lit(2.0), Index("bra_exp", (bra, Var(KAB), Lit(1))))),
    ]
    for tag, efs in sets.items():
        w = weights[tag]
        for e, f in efs:
            u, t = Var(_set_name(tag, e, f)), Var(acc_name(e, f))
            fold.append(Assign(u, _add(u, t if w is None else _mul(w, t))))
    per_kab = (
        Note("the bra primitive pair, from shared memory"),
        *(_F(f"Pb{d}", bra_prim(_P + d)) for d in range(3)),
        *(_F(Factor.pa(d).value, bra_prim(_X + d)) for d in range(3)
          if Factor.pa(d).value in vneed),
        _F("p", bra_prim(_ZETA)),
        _F("Ub", bra_prim(_U)),
        Note(f"the ket-inner sums T of the {len(bnd)} VRR targets, zeroed "
             "per bra primitive pair"),
        *(_F(acc_name(e, f), Lit(0.0)) for e, f in bnd),
        For(KCD, Lit(0), Var("k_cd"), Lit(1), per_kcd),
        *fold,
    )

    # -- the deposit ----------------------------------------------------------
    n = [len(vrr.cartesians(x)) for x in am]
    ca, cc, cb, cd_ = (vrr.cartesians(x) for x in am)
    deposit: list[Stmt] = [
        Note("the deposit: per ERI (ac|bd), W = M[a, b] M[c, d], and per "
             "axis gA += W d/dA_k (ac|bd), gC += W d/dC_k (ac|bd)"),
        *(_F(f"dab_{i}_{j}", Index("M", (_ao("oa", i), _ao("ob", j))))
          for i in range(n[0]) for j in range(n[2])),
        *(_F(f"dcd_{i}_{j}", Index("M", (_ao("oc", i), _ao("od", j))))
          for i in range(n[1]) for j in range(n[3])),
    ]
    up_a, up_c = _hrr_local("a+"), _hrr_local("c+")
    dn_a, dn_c = _hrr_local("a-"), _hrr_local("c-")
    for i, a in enumerate(ca):
        for kk, c in enumerate(cc):
            for j, b in enumerate(cb):
                for m, d in enumerate(cd_):
                    w = f"w_{i}_{kk}_{j}_{m}"
                    deposit.append(_F(w, _mul(Var(f"dab_{i}_{j}"),
                                              Var(f"dcd_{kk}_{m}"))))
                    for x in range(3):
                        da: Expr = Var(up_a((_shift(a, x, 1), c, b, d)))
                        if a[x]:
                            da = _c(Fn.SUB, da, _mul(Lit(float(a[x])), Var(
                                dn_a((_shift(a, x, -1), c, b, d)))))
                        dc: Expr = Var(up_c((a, _shift(c, x, 1), b, d)))
                        if c[x]:
                            dc = _c(Fn.SUB, dc, _mul(Lit(float(c[x])), Var(
                                dn_c((a, _shift(c, x, -1), b, d)))))
                        for side, dv in (("gA", da), ("gC", dc)):
                            g = Var(f"{side}{_AXES[x]}")
                            deposit.append(Assign(g, _add(g, _mul(Var(w),
                                                                  dv))))

    work = (
        *(_F(Factor.cd(dd).value, Index("CD", (k, Lit(dd)))) for dd in range(3)
          if Factor.cd(dd).value in hneed),
        _I("ob", Index("ket_ao", (k, Lit(0)))),
        _I("od", Index("ket_ao", (k, Lit(1)))),
        Note("the three contracted boundary sets (module docstring), zeroed "
             "per ket"),
        *(_F(_set_name(tag, e, f), Lit(0.0))
          for tag, efs in sets.items() for e, f in efs),
        Note("the K_AB x K_CD primitive nest: bra outer (shared), ket inner "
             "(global) -- kernel_hj's order"),
        For(KAB, Lit(0), Var("k_ab"), Lit(1), per_kab),
        *(st for r in roles for st in (
            Note(f"HRR {r}: {roles[r][1]} on {roles[r][0]}"), *hbodies[r])),
        *deposit,
    )
    per_bra = (
        *(_F(Factor.ab(dd).value, Index("AB", (bra, Lit(dd))))
          for dd in range(3) if Factor.ab(dd).value in hneed),
        _I("oa", Index("bra_ao", (bra, Lit(0)))),
        _I("oc", Index("bra_ao", (bra, Lit(1)))),
        Note(STAGING),
        For(STAGE, Lit(0), Var("k_ab"), Lit(1), tuple(
            Assign(Index(AB_SH, (Var(STAGE), Lit(f))),
                   Index("bra_prim", (bra, Var(STAGE), Lit(f))))
            for f in range(len(REC)))),
        *(_F(f"{s}{x}", Lit(0.0)) for s in ("gA", "gC") for x in _AXES),
        Note(NO_SCREEN),
        Note("one thread per contracted quartet: consecutive threads take "
             "consecutive |bd) ([Fock] §3.2)"),
        For(KET, Call(Fn.THREAD_X), Var("n_ket"), Call(Fn.BLOCK_DIM_X),
            work),
        Note("this thread's share of the bra's two centres, added once"),
        *(AtomicAdd(Index(G_OUT, (Index("bra_atom", (bra, Lit(p))),
                                  Lit(x))), Var(f"{s}{_AXES[x]}"))
          for p, s in ((0, "gA"), (1, "gC")) for x in range(3)),
    )
    body = (
        Note(f"dE_K/dR by the HGP recurrence, K class ({la}{lc}|{lb}{ld}): "
             f"one VRR {vam} inside the K_AB x K_CD nest, the raised and "
             "lowered HRRs after it, the bra derivative contracted with "
             "D_ab D_cd (gradient/kernel_ghk.py)"),
        Note("launch: grid = n_bra, blockDim = (threads, 1); one block per "
             "bra shell pair"),
        Decl("bra", Scalar.I32, init=Call(Fn.BLOCK_X)),
        Decl(AB_SH, Scalar.F64, (Lit(MAX_PRIM_PAIRS), Lit(len(REC))),
             Space.SHARED),
        If(_c(Fn.LT, bra, Var("n_bra")), per_bra),
    )
    return Kernel(name or ghk_name(am), _params(am), body)


def _params(am: Am) -> tuple[Param, ...]:
    nb, nk, kab, kcd = (Var(x) for x in ("n_bra", "n_ket", "k_ab", "k_cd"))
    nao, nat = Var("n_ao"), Var("n_atom")

    def f64(x: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(x, Scalar.F64, array=True, intent=intent, shape=shape)

    def i32(x: str, *shape: Expr) -> Param:
        return Param(x, Scalar.I32, array=True, shape=shape)

    out = (
        *(Param(x, Scalar.I32) for x in ("n_bra", "n_ket", "k_ab", "k_cd",
                                          "n_ao", "n_atom")),
        f64("bra_prim", nb, kab, Lit(len(REC))),
        f64("bra_exp", nb, kab, Lit(2)),
        f64("AB", nb, Lit(3)),
        i32("bra_ao", nb, Lit(2)), i32("bra_atom", nb, Lit(2)),
        f64("ket_prim", nk, kcd, Lit(len(REC))),
        *((f64("CD", nk, Lit(3)),) if am[3] else ()),
        i32("ket_ao", nk, Lit(2)),
        f64("M", nao, nao),
        f64(G_OUT, nat, Lit(3), intent=Intent.OUT),
    )
    assert tuple(p.name for p in out) == params_of(am)
    return out
