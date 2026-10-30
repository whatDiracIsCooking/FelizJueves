"""The Coulomb-energy gradient on the HGP recurrence: `Target.J` on
`Engine.HGP` at `derivative_order = 1`.

    build_ghj_kernel(spec | (la, lb, lc, ld)) -> Kernel   one per ordered class
    ghj_spec(la, lb, lc, ld)          -> Spec   the class's d1 key
    ghj_name(spec | class)            -> "ghj_{la}{lb}{lc}{ld}"
    vrr_dag(cls)                      -> the one VRR the kernel runs
    hrr_classes(cls)                  -> the HRR classes it runs, by role
    params_of(spec | class)           -> the parameter names, in order
    launch_shape(n_bra, threads)      -> (n_bra, (threads, 1))

THE QUANTITY. `E_J = 1/2 SUM D_ab D_cd (ab|cd)` over every ORDERED shell
quartet, and its ket derivative equals its bra derivative (the package
docstring), so

    dE_J / dR_{X,k} = SUM_ordered SUM_ijkl D_ab D_cd
                        (d/dA_k + d/dB_k) (a_i b_j | c_k d_l)

with `d/dA_k` deposited on `a`'s atom and `d/dB_k` on `b`'s. That is the
MMD gradient's split (`kernel_gjab`: the bra derivative alone, the whole
`dE_J`, no half, no factor 2), reached through the other recurrence. Every
ordered quartet is visited once, as `kernel_hj`'s ordered stream visits it;
summed over every ordered class the deposits are the whole `dE_J`.

THE DERIVATIVE BY THE CARTESIAN RULE, `d/dA_k x_A^i e^{-alpha x_A^2} =
2 alpha x_A^{i+1} e^{...} - i x_A^{i-1} e^{...}`:

    d/dA_k (a b|cd) = 2 alpha (a+1_k, b|cd)  -  a_k (a-1_k, b|cd)
    d/dB_k (a b|cd) = 2 beta  (a, b+1_k|cd)  -  b_k (a, b-1_k|cd)

The `2 alpha` / `2 beta` belong to the PRIMITIVE pair, so they cannot ride
on a contracted HRR value; they ride on the contraction instead. Eq 28 is
linear and its coefficients are geometry alone (`hgp.hrr`'s docstring), so
it holds for ANY per-primitive weighting of the boundary values, and

    (a+1_k, b|cd)_{2alpha} = HRR of SUM_prim 2 alpha [e0|f0]^(0)

is exact. So one VRR serves three contractions, accumulated side by side
in the `K_AB x K_CD` nest:

    C   += [e0|f0]^(0)            plain    -> HRR (la-1, lb), (la, lb-1)
    CA  += 2 alpha [e0|f0]^(0)    on a     -> HRR (la+1, lb)
    CB  += 2 beta  [e0|f0]^(0)    on b     -> HRR (la, lb+1)

and up to four HRRs run after the nest, each its own `hgp.hrr` DAG with
its own local prefix (`ROLES`): `HP` (a raised), `HM` (a lowered), `HQ`
(b raised), `HN` (b lowered); a lowered one is absent when its `l` is 0
(every term it would feed carries `a_k = 0`), and `C` with it when both
are. THE VRR (`vrr_dag`) is the closure, under `hgp.vrr.reduce`, of the
union of the sets' boundaries -- `|e|` from `la - 1` (or `la`) to
`la + lb + 1`, `|f|` from `lc` to `lc + ld` -- on the fundamental one Boys
order up (`L + 1`). Every node keeps `hgp.vrr.reduce`'s terms, so a value
the d0 class `(la lb|lc ld)` also reaches is the same expression as in
`kernel_hj`'s units.

THE DEPOSIT, per contracted quartet (one thread): with `w = D[a_i][b_j]
D[c_k][d_l]` read where the ordered quartet puts it (no symmetric `D`
assumed inside the quartet),

    gA_k += w (HP(a_i + 1_k, b_j|..) - a_ik HM(a_i - 1_k, b_j|..))
    gB_k += w (HQ(a_i, b_j + 1_k|..) - b_jk HN(a_i, b_j - 1_k|..))

then `G[atom_a][k] += gA_k`, `G[atom_b][k] += gB_k` by atomic add: six
`AtomicAdd`s per quartet. `G` is ZEROED BY THE CALLER; every quartet only
adds. The atom of each bra shell is a per-pair record, `bra_atom[n_bra,
2]` (the host's `gradient.host.shell_atoms`).

ONE CLASS MODE: ITS OWN. The d0 kernel's canonical mode shares one kernel
across an 8-fold orbit through a runtime permutation; a derivative on the
bra breaks the bra-ket and within-pair symmetries the orbit is made of
(the raised and lowered classes of `(ps|ss)` and `(sp|ss)` differ), so
every ordered class is its own unit -- `kernel_hj`'s `ordered` mode by
construction. The key carries no lever, and any lever is refused.

NO SCREEN. `kernel_hj`'s one-block bound `G_ab G_cd max|D_cd|` bounds an
ERI, not a derivative ERI, so it is not borrowed: every quartet is
computed, and the kernel takes no `tau`, `Dsh`, `G_ab` or `G_cd`
(`kernel_gk` takes the same position).

THE NEST AND THE RECORD ARE `kernel_hj`'s: one block per bra shell pair,
one thread per contracted quartet, `|ab)` staged in shared memory (every
thread copies it; the IR has no barrier), bra primitives outer and ket
inner, the quartet scalars in `eri/hgp_drive.py`'s operation order, the
record layout `kernel_f.REC`. The exponents the weights need ride beside
it, `bra_exp[n_bra, k_ab, 2] = (alpha, beta)` per primitive pair (`i`
outer, the record's order), staged with it into `ab_ex`.

ROUND-OFF, NOT BITWISE, against `gradient.host.coulomb_gradient` (MMD,
another recurrence and another contraction order); the bound is in
test/test_kernel_ghj.py.

A builder: vendor-neutral IR. It imports neither `ao/` nor `gradient.host`
nor the interpreter nor `reference/` (test/test_layering.py, rule
`ghj_builder`).
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

from felizjueves.eri.fock.kernel_f import MAX_PRIM_PAIRS, REC
from felizjueves.eri.hjk.kernel_hj import (
    _DERIVED,
    _F,
    _P,
    _U,
    _X,
    _ZETA,
    AB_SH,
    KAB,
    KCD,
    KET,
    STAGE,
    _c,
    _factors,
    _quartet_scalars,
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
    "AB_EX",
    "G_OUT",
    "ROLES",
    "build_ghj_kernel",
    "ghj_name",
    "ghj_spec",
    "hrr_classes",
    "launch_shape",
    "params_of",
    "vrr_dag",
]

G_OUT = "G"  # the gradient, n_atom x 3, AtomicAdd-ed
AB_EX = "ab_ex"  # the staged (alpha, beta) of each bra primitive pair
N_EX = 2

# The four HRRs (module docstring): role -> (local prefix, accumulator
# set, shift of (la, lb)).
ROLES: dict[str, tuple[str, str, tuple[int, int]]] = {
    "a+": ("HP", "CA", (1, 0)),
    "a-": ("HM", "C", (-1, 0)),
    "b+": ("HQ", "CB", (0, 1)),
    "b-": ("HN", "C", (0, -1)),
}
# The weighted sets: the `bra_exp` column whose double weights each.
_WEIGHT = {"CA": 0, "CB": 1}

STAGING = (
    "stage |ab) and its exponents in shared memory: every thread copies the "
    "whole record (the IR has no barrier, #333); the degree k_ab is at most "
    f"MAX_PRIM_PAIRS = {MAX_PRIM_PAIRS}, which kernel_f.pack enforces")


# --- the key -----------------------------------------------------------------


def ghj_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The HGP `J` class `(la lb | lc ld)` at derivative order 1."""
    return Spec(Operator.COULOMB, 1, (la, lb, lc, ld), Target.J,
                engine=Engine.HGP)


def _spec_of(cls: AmClass | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = ghj_spec(*cls)
    else:
        raise TypeError("build_ghj_kernel takes (la, lb, lc, ld) or a Spec, "
                        f"not {cls!r}")
    if (spec.target is not Target.J or spec.engine is not Engine.HGP
            or spec.derivative_order != 1):
        raise ValueError(f"{spec.key()}: the HGP J gradient kernel builds "
                         "Target.J on Engine.HGP at derivative order 1 only")
    if spec.levers:
        raise ValueError(f"{spec.key()}: lever {spec.levers[0]!r} is not read "
                         "by the HGP J gradient kernel (one class mode, its "
                         "own: the module docstring)")
    if spec.n_batches != 1:
        raise ValueError(f"{spec.key()}: n_batches must be 1")
    return spec


def ghj_name(cls: AmClass | Spec) -> str:
    """`ghj_{la}{lb}{lc}{ld}`, `_`-separated once any `l` is above 9."""
    am = _spec_of(cls).am
    sep = "_" if max(am) > 9 else ""
    return "ghj_" + sep.join(map(str, am))


def launch_shape(n_bra: int, threads: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (threads, 1))`: `kernel_hj`'s mapping."""
    for name, v in (("n_bra", n_bra), ("threads", threads)):
        if type(v) is not int or v < 1:
            raise ValueError(f"{name} = {v!r} is not a positive int")
    return n_bra, (threads, 1)


def params_of(cls: AmClass | Spec) -> tuple[str, ...]:
    """The parameter names, in order; `CD` only when `ld > 0` (the ket HRR
    reads it)."""
    ld = _spec_of(cls).am[3]
    return ("n_bra", "n_ket", "k_ab", "k_cd", "n_ao", "n_atom",
            "bra_prim", "bra_exp", "AB", "bra_atom", "bra_ao",
            "ket_prim", *(("CD",) if ld else ()), "ket_ao", "D", G_OUT)


# --- the recurrences ---------------------------------------------------------


def hrr_classes(cls: AmClass | Spec) -> dict[str, AmClass]:
    """Role -> the HRR class it runs (`ROLES`), a lowered role absent when
    its `l` is 0."""
    la, lb, lc, ld = _spec_of(cls).am
    out = {}
    for role, (_, _, (da, db)) in ROLES.items():
        if la + da >= 0 and lb + db >= 0:
            out[role] = (la + da, lb + db, lc, ld)
    return out


def _sets(cls: AmClass | Spec) -> dict[str, tuple[tuple[Cart, Cart], ...]]:
    """Accumulator set -> its boundary `(e, f)`: the union of the boundaries
    of the HRRs reading it, each in `hgp.vrr.targets` order, first seen
    first."""
    out: dict[str, dict[tuple[Cart, Cart], None]] = {}
    for role, am in hrr_classes(cls).items():
        seen = out.setdefault(ROLES[role][1], {})
        for e, f, _ in vrr.targets(am):
            seen[(e, f)] = None
    return {k: tuple(v) for k, v in out.items()}


def vrr_dag(cls: AmClass | Spec) -> vrr.VrrDag:
    """The one VRR (module docstring): the closure of every accumulator
    set's boundary under `hgp.vrr.reduce`, in `hgp.vrr.order_key` order."""
    spec = _spec_of(cls)
    tg = tuple(dict.fromkeys((e, f, 0) for bnd in _sets(spec).values()
                             for e, f in bnd))
    terms: dict[vrr.Node, tuple[vrr.Term, ...]] = {}
    todo = list(tg)
    while todo:
        n = todo.pop()
        if n not in terms:
            terms[n] = vrr.reduce(n)
            todo.extend(t.child for t in terms[n])
    order = tuple(sorted(terms, key=vrr.order_key))
    return vrr.VrrDag(tuple(spec.am), True, tg, MappingProxyType(terms),
                      order)


def acc_name(acc: str, e: Cart, f: Cart) -> str:
    """The contracted accumulator of `(e0|f0)` in set `acc`."""
    return acc + "_" + "__".join("_".join(map(str, v)) for v in (e, f))


def local_name(role: str) -> Callable[[hrr.Node], str]:
    """The local spelling of role `role`'s HRR nodes: its prefix in place
    of `hrr_name`'s `H`."""
    prefix = ROLES[role][0]
    return lambda n: prefix + hrr_name(n)[1:]


def _shift(v: Cart, k: int, by: int) -> Cart:
    return tuple(c + by * (i == k) for i, c in enumerate(v))  # type: ignore[return-value]


# --- the kernel --------------------------------------------------------------


def _deposit(am: AmClass) -> list[Stmt]:
    """`gA_k`, `gB_k` over every ERI of the class (module docstring), then
    the six `AtomicAdd`s."""
    carts = [vrr.cartesians(x) for x in am]
    n = [len(c) for c in carts]

    def ao(pos: int, i: int) -> Expr:
        return _c(Fn.ADD, Var(f"o{'abcd'[pos]}"), Lit(i))

    out: list[Stmt] = [
        Note("the deposit: gA_k += w (HP(a+1_k) - a_k HM(a-1_k)), gB_k += w "
             "(HQ(b+1_k) - b_k HN(b-1_k)), w = D_ab D_cd where the ordered "
             "quartet puts them"),
        *(_F(f"dab{i}_{j}", Index("D", (ao(0, i), ao(1, j))))
          for i in range(n[0]) for j in range(n[1])),
        *(_F(f"dcd{k}_{l}", Index("D", (ao(2, k), ao(3, l))))
          for k in range(n[2]) for l in range(n[3])),  # noqa: E741
        *(_F(f"g{s}{x}", Lit(0.0)) for s in "AB" for x in range(3)),
        _F("w", Lit(0.0)),
    ]
    sides = (("A", 0, "a+", "a-"), ("B", 1, "b+", "b-"))
    for i, a in enumerate(carts[0]):
        for j, b in enumerate(carts[1]):
            for k, c in enumerate(carts[2]):
                for l, d in enumerate(carts[3]):  # noqa: E741
                    out.append(Assign(Var("w"), _c(
                        Fn.MUL, Var(f"dab{i}_{j}"), Var(f"dcd{k}_{l}"))))
                    for s, pos, up, dn in sides:
                        for x in range(3):
                            q = [a, b, c, d]
                            q[pos] = _shift(q[pos], x, 1)
                            term: Expr = Var(local_name(up)(tuple(q)))
                            nx = (a, b)[pos][x]
                            if nx:
                                q[pos] = _shift(q[pos], x, -2)
                                lo: Expr = Var(local_name(dn)(tuple(q)))
                                if nx > 1:
                                    lo = _c(Fn.MUL, Lit(float(nx)), lo)
                                term = _c(Fn.SUB, term, lo)
                            g = Var(f"g{s}{x}")
                            out.append(Assign(g, _c(
                                Fn.ADD, g, _c(Fn.MUL, Var("w"), term))))
    out += [AtomicAdd(Index(G_OUT, (Var(f"at{s}"), Lit(x))), Var(f"g{s}{x}"))
            for s in "AB" for x in range(3)]
    return out


def build_ghj_kernel(cls: AmClass | Spec, *, name: str | None = None) -> Kernel:
    """The HGP `J` gradient kernel of one ordered class (module
    docstring). `cls` is a d1 `Target.J`, `Engine.HGP` `Spec`, or `(la, lb,
    lc, ld)`."""
    spec = _spec_of(cls)
    am: AmClass = tuple(spec.am)  # type: ignore[assignment]
    la, lb, lc, ld = am
    sets = _sets(spec)
    dag = vrr_dag(spec)
    prelude, fundamental = fundamental_prelude(sum(am) + 1)
    vbody = tuple(vrr_body(dag, fundamental))
    vf = _factors(vbody)
    vneed = frozenset(vf | {i2 for n, i2, _ in _DERIVED if n in vf})

    roles = hrr_classes(spec)
    hbody: list[Stmt] = []
    for role, hcls in roles.items():
        acc = ROLES[role][1]
        hd = hrr.build(hcls)
        if not hd.boundary <= frozenset(sets[acc]):  # refuse, never misread
            raise ValueError(f"{am} {role}: the HRR boundary is not in {acc}")
        hbody += [Note(f"the {role} HRR, class {hcls}, on {acc}"),
                  *hrr_body(hd, lambda e, f, acc=acc: Var(acc_name(acc, e, f)),
                            name=local_name(role))]
    hneed = frozenset(_factors(hbody))
    bra, k = Var("bra"), Var(KET)

    def bra_prim(f: int) -> Expr:
        return Index(AB_SH, (Var(KAB), Lit(f)))

    def ket_prim(f: int) -> Expr:
        return Index("ket_prim", (k, Var(KCD), Lit(f)))

    accumulate: list[Stmt] = [
        Note("the three contractions: C plain, CA weighted 2 alpha, CB 2 beta "
             "-- eq 28 is linear in its boundary, so a per-primitive weight "
             "rides through the HRR exactly")]
    for acc, bnd in sets.items():
        for e, f in bnd:
            v: Expr = Var(vrr_name((e, f, 0)))
            if acc in _WEIGHT:
                v = _c(Fn.MUL, Var("t" + acc), v)
            x = Var(acc_name(acc, e, f))
            accumulate.append(Assign(x, _c(Fn.ADD, x, v)))

    per_kcd = (
        Note("the ket primitive pair, from global memory"),
        *(_F(f"Qk{d}", ket_prim(_P + d)) for d in range(3)),
        *(_F(Factor.qc(d).value, ket_prim(_X + d)) for d in range(3)
          if Factor.qc(d).value in vneed),
        _F("q", ket_prim(_ZETA)),
        _F("Uq", ket_prim(_U)),
        *_quartet_scalars(vneed),
        *prelude,
        *vbody,
        *accumulate,
    )
    per_kab = (
        Note("the bra primitive pair and its exponents, from shared memory"),
        *(_F(f"Pb{d}", bra_prim(_P + d)) for d in range(3)),
        *(_F(Factor.pa(d).value, bra_prim(_X + d)) for d in range(3)
          if Factor.pa(d).value in vneed),
        _F("p", bra_prim(_ZETA)),
        _F("Ub", bra_prim(_U)),
        *(_F("t" + acc, _c(Fn.MUL, Lit(2.0),
                           Index(AB_EX, (Var(KAB), Lit(col)))))
          for acc, col in _WEIGHT.items()),
        For(KCD, Lit(0), Var("k_cd"), Lit(1), per_kcd),
    )
    accs = [(acc, e, f) for acc, bnd in sets.items() for e, f in bnd]
    per_ket = (
        *(_F(Factor.cd(d).value, Index("CD", (k, Lit(d)))) for d in range(3)
          if Factor.cd(d).value in hneed),
        Decl("oc", Scalar.I32, init=Index("ket_ao", (k, Lit(0)))),
        Decl("od", Scalar.I32, init=Index("ket_ao", (k, Lit(1)))),
        Note(f"the {len(accs)} contracted boundary accumulators, "
             f"{len(sets)} sets, zeroed per ket"),
        *(_F(acc_name(acc, e, f), Lit(0.0)) for acc, e, f in accs),
        Note("the K_AB x K_CD primitive nest: bra outer (shared), ket inner "
             "(global) -- ao/hgp_drive's (i, j, k, l) order"),
        For(KAB, Lit(0), Var("k_ab"), Lit(1), per_kab),
        *hbody,
        *_deposit(am),
    )
    per_bra = (
        *(_F(Factor.ab(d).value, Index("AB", (bra, Lit(d)))) for d in range(3)
          if Factor.ab(d).value in hneed),
        Decl("oa", Scalar.I32, init=Index("bra_ao", (bra, Lit(0)))),
        Decl("ob", Scalar.I32, init=Index("bra_ao", (bra, Lit(1)))),
        Decl("atA", Scalar.I32, init=Index("bra_atom", (bra, Lit(0)))),
        Decl("atB", Scalar.I32, init=Index("bra_atom", (bra, Lit(1)))),
        Note(STAGING),
        For(STAGE, Lit(0), Var("k_ab"), Lit(1), (
            *(Assign(Index(AB_SH, (Var(STAGE), Lit(f))),
                     Index("bra_prim", (bra, Var(STAGE), Lit(f))))
              for f in range(len(REC))),
            *(Assign(Index(AB_EX, (Var(STAGE), Lit(f))),
                     Index("bra_exp", (bra, Var(STAGE), Lit(f))))
              for f in range(N_EX)))),
        Note("one thread per contracted quartet: consecutive threads take "
             "consecutive |cd) ([Fock] §3.2)"),
        For(KET, Call(Fn.THREAD_X), Var("n_ket"), Call(Fn.BLOCK_DIM_X),
            per_ket),
    )
    body = (
        Note(f"dE_J/dR by the HGP recurrence, class ({la}{lb}|{lc}{ld}) at "
             "d1: one VRR (Boys order L + 1) into the contractions "
             f"{', '.join(sets)}, the HRRs {', '.join(roles)} after the nest, "
             "then the bra derivative on a's and b's atoms (G zeroed by the "
             "caller)"),
        Note("launch: grid = n_bra, blockDim = (threads, 1); one block per "
             "bra shell pair"),
        Decl("bra", Scalar.I32, init=Call(Fn.BLOCK_X)),
        Decl(AB_SH, Scalar.F64, (Lit(MAX_PRIM_PAIRS), Lit(len(REC))),
             Space.SHARED),
        Decl(AB_EX, Scalar.F64, (Lit(MAX_PRIM_PAIRS), Lit(N_EX)),
             Space.SHARED),
        If(_c(Fn.LT, bra, Var("n_bra")), per_bra),
    )
    return Kernel(name or ghj_name(spec), _params(spec), body)


def _params(spec: Spec) -> tuple[Param, ...]:
    ld = spec.am[3]
    nb, nk, kab, kcd = (Var(x) for x in ("n_bra", "n_ket", "k_ab", "k_cd"))
    nao, nat = Var("n_ao"), Var("n_atom")

    def f64(x: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(x, Scalar.F64, array=True, intent=intent, shape=shape)

    def i32(x: str, *shape: Expr) -> Param:
        return Param(x, Scalar.I32, array=True, shape=shape)

    out = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("k_ab", Scalar.I32), Param("k_cd", Scalar.I32),
        Param("n_ao", Scalar.I32), Param("n_atom", Scalar.I32),
        f64("bra_prim", nb, kab, Lit(len(REC))),
        f64("bra_exp", nb, kab, Lit(N_EX)),
        f64("AB", nb, Lit(3)), i32("bra_atom", nb, Lit(2)),
        i32("bra_ao", nb, Lit(2)),
        f64("ket_prim", nk, kcd, Lit(len(REC))),
        *((f64("CD", nk, Lit(3)),) if ld else ()),
        i32("ket_ao", nk, Lit(2)),
        f64("D", nao, nao),
        f64(G_OUT, nat, Lit(3), intent=Intent.OUT),
    )
    assert tuple(p.name for p in out) == params_of(spec)
    return out
