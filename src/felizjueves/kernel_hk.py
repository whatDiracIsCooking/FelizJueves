"""The HGP exchange kernel, split (`Target.K` on `Engine.HGP`, #405).

    build_hk_kernel(spec | (la, lc, lb, ld), *, eris=, density=) -> Kernel
    hk_class_spec(la, lc, lb, ld, *, ordered=False) -> Spec   the class's key
    hk_name(spec)                  -> "hk_..." (canonical) / "hk_o_..." (ordered)
    built_class(spec)              -> the class whose recurrence is built
    orientations(spec)             -> the `hgp.orbits.GROUP` indices it deposits
    params_of(spec, *, eris=False) -> the parameter names, in order
    launch_shape(n_bra, threads)   -> (n_bra, (threads, 1))   (kernel_hj's)
    kernel_counts(spec) / counted(kernel) -> statement counts, DAG vs built

The exchange matrix on the Head-Gordon-Pople recurrence of `[Fock]` §3.1,
SPLIT: the sibling of `kernel_hj.py` (#404), and the same kernel in every
respect but the deposit. Every ORDERED shell quartet is visited once (#403's
stream) and deposits

    K[a_i][b_j] += D[c_k][d_l] (a_i c_k | b_j d_l)

by atomic add -- that and nothing else: no weights, no degeneracy factor, no
mirror. It is `ao/k.py`'s ordered sweep (`exchange_matrix_reference`: every
ordered quadruple `(a, b, c, d)` at weight 1, `K[a][b]` its own sum, the
quartet `k_quartet(epair(a, c), epair(b, d), D_cd)`), one class at a time,
on another engine.

`am` IS K'S HERMITE PAIRING, `(la, lc | lb, ld)` (#196, #401): the ERI is
`(ac|bd)`, so the HGP bra pair is `(a, c)` and the ket pair `(b, d)`, exactly
MMD `K`'s Hermite pairs, and a K class is named the same under both engines.
In the ERI's own positions `(x0 x1 | x2 x3) = (a c | b d)`, `K` lands on
positions `(0, 2)` and reads `D` at `(1, 3)` (`K_OUTPUT`, `K_DENSITY`). A
class spelled in AO order `(la, lb, lc, ld)` is a DIFFERENT class whenever
`lb != lc` -- it builds another recurrence and deposits into other `K`
blocks -- and test/test_kernel_hk.py shows it.

THE NEST, THE STAGING, THE CONTRACTION, THE QUARTET SCALARS AND THE TWO
CLASS MODES ARE `kernel_hj`'s, restated statement for statement (its
docstring is the specification; the pieces are imported from it, not
copied): one block per bra shell pair, `kernel_f`'s staged `|ab)` record
copied by every thread (no barrier, #333), one thread per contracted
quartet, the `K_AB x K_CD` primitive nest BRA OUTER and ket inner with the
scalars in `ao/hgp_drive.py`'s operation order -- so the ERI tap is `==`
`hgp_drive.eri_block` bit for bit -- the contraction completed in registers
before the HRR (eq 28 needs contracted values), and:

  canonical (literal)  one kernel per orbit of the 8-fold ERI symmetry,
      built on `hgp.orbits.representative`, `hk_{rep}`. An ordered K quartet
      `q = (a, c, b, d)` of any member class is launched as the
      representative-ordered `x`, `x[j] = q[g[j]]`, `g =
      to_representative(class).shells`; `perm` is `g`'s index in
      `hgp.orbits.GROUP`. `q`'s position `i` is `x`'s `g^-1[i]`, so `K`
      lands on `(g^-1[0], g^-1[2])` with the density of `(g^-1[1],
      g^-1[3])`. One deposit per orientation the orbit's members reach,
      selected by `kernel_hj`'s exact `LT` membership chain; a `perm`
      outside them deposits nothing.
  ordered (`ordered` lever)  the class's own recurrence, the identity
      deposit, no `perm`: `hk_o_{class}`.

Names `_`-separate the four `l`s once any is above 9, as `hj`'s.

THE DEPOSIT READS `D` WHERE THE ORDERED QUARTET PUTS IT, `D[c][d]`, in every
orientation, and writes `K[a][b]` only from its own block; nothing assumes a
symmetric `D` and nothing mirrors `K`.

THE SCREEN (#403's K key): `G_ac G_bd max|D_cd|` -- in the stream's AO
spelling `Q_ab Q_cd max|D_bd|` -- ONE density block (the `Dsh` element at
the shells of the density positions), a SKIP (`If(bound < tau)` holds a
`Note`; no `Break`). At `tau = 0` nothing is skipped.

`density=` names the ordered positions whose `D` block the deposit reads,
`(1, 3)` by default; any other pair is a MUTATION a test builds to watch the
deposit fail (`kernel_hj`'s `density=`).

Nothing here imports `plan/`, `ao/` (the witnesses: `ao/hgp_drive.py` for
the ERIs, `ao/k.py` for `K`), `reference/`, the interpreter, a backend or a
device (test/test_layering.py, rule `hk_builder`); it reaches `kernel_hj`
for the shared pieces, and through it nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from felizjueves.hgp import hrr, orbits, vrr
from felizjueves.hgp.recurrence import (
    boundary_order,
    fundamental_prelude,
    hrr_body,
    hrr_name,
    vrr_body,
    vrr_name,
)
from felizjueves.hgp.vrr import AmClass
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
    walk,
)
from felizjueves.kernel_f import MAX_PRIM_PAIRS, REC
from felizjueves.kernel_hj import (
    _AO,
    _DERIVED,
    _F,
    _I,
    _P,
    _SH,
    _U,
    _X,
    _ZETA,
    AB_SH,
    KAB,
    KCD,
    KET,
    ORDERED,
    OUT,
    PERM,
    STAGE,
    STAGING,
    _ao,
    _c,
    _comps,
    _factors,
    _membership,
    _Orient,
    _quartet_scalars,
    _record,
    acc_name,
    inverse,
    launch_shape,
)
from felizjueves.spec import Engine, Operator, Spec, Target, lever_provenance

__all__ = [
    "AB_SH",
    "Counts",
    "KAB",
    "KCD",
    "KET",
    "K_DENSITY",
    "K_OUT",
    "K_OUTPUT",
    "MAX_PRIM_PAIRS",
    "ORDERED",
    "OUT",
    "PERM",
    "REC",
    "SKIP",
    "STAGE",
    "build_hk_kernel",
    "built_class",
    "counted",
    "hk_class_spec",
    "hk_name",
    "inverse",
    "kernel_counts",
    "launch_shape",
    "orientations",
    "params_of",
]

K_OUT = "K"  # the exchange matrix, AtomicAdd-ed
K_OUTPUT = (0, 2)  # K_ab: the ERI positions (a, b) of (a c | b d)
K_DENSITY = (1, 3)  # K_ab reads D_cd: the ERI positions (c, d)

SKIP = ("screen: skip this quartet, never break; G_ac G_bd max|D_cd| bounds "
        "the one block K reads (#403's K key)")


# --- the key -----------------------------------------------------------------


def hk_class_spec(la: int, lc: int, lb: int, ld: int, *,
                  ordered: bool = False) -> Spec:
    """The HGP `K` class `(la, lc | lb, ld)` -- K's Hermite pairing, the
    HGP bra `(a, c)` (#196, #401) -- one batch, carrying the `ordered`
    lever when asked."""
    return Spec(Operator.COULOMB, 0, (la, lc, lb, ld), Target.K,
                (ORDERED,) if ordered else (), engine=Engine.HGP)


def _spec_of(cls: AmClass | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = hk_class_spec(*cls)
    else:
        raise TypeError("build_hk_kernel takes (la, lc, lb, ld) or a Spec, "
                        f"not {cls!r}")
    if spec.target is not Target.K or spec.engine is not Engine.HGP:
        raise ValueError(f"{spec.key()}: the HGP K kernel builds Target.K on "
                         "Engine.HGP only")
    if len(spec.am) != 4:
        raise ValueError(f"{spec.key()}: a K class is a quartet")
    other = tuple(lv for lv in spec.levers if lv != ORDERED)
    if other:
        raise ValueError(f"{spec.key()}: lever {other[0]!r} is not read by the "
                         "HGP K kernel (only the class mode, 'ordered')")
    if spec.n_batches != 1:
        raise ValueError(f"{spec.key()}: the HGP K kernel has no Hermite "
                         "targets to batch; n_batches must be 1")
    return spec


def _ordered(spec: Spec) -> bool:
    return ORDERED in spec.levers


def built_class(cls: AmClass | Spec) -> AmClass:
    """The class whose VRR and HRR the kernel runs: the spec's own `am`
    under `ordered`, its orbit's representative otherwise."""
    spec = _spec_of(cls)
    am: AmClass = tuple(spec.am)  # type: ignore[assignment]
    return am if _ordered(spec) else orbits.representative(am)


def orientations(cls: AmClass | Spec) -> tuple[int, ...]:
    """The `hgp.orbits.GROUP` indices the kernel deposits through, sorted:
    `(0,)` under `ordered`; canonical, the `to_representative` permutation
    of each distinct member of the orbit (the `perm` values)."""
    spec = _spec_of(cls)
    if _ordered(spec):
        return (0,)
    return tuple(sorted({orbits.GROUP.index(orbits.to_representative(m).shells)
                         for m in orbits.orbit(tuple(spec.am))}))


def hk_name(cls: AmClass | Spec) -> str:
    """`hk_{built class}` canonical, `hk_o_{class}` ordered, `_`-separated
    once any `l` is above 9."""
    spec = _spec_of(cls)
    am = built_class(spec)
    sep = "_" if max(am) > 9 else ""
    return ("hk_o_" if _ordered(spec) else "hk_") + sep.join(map(str, am))


def params_of(cls: AmClass | Spec, *, eris: bool = False) -> tuple[str, ...]:
    """The kernel's parameter names, in order (`kernel_hj.params_of`'s, with
    `K` for `J`). `perm` only canonical; `AB` only when the built class's
    second shell is not s, `CD` only when its fourth is not; `g` only with
    the tap."""
    spec = _spec_of(cls)
    _, l1, _, l3 = built_class(spec)
    return ("n_bra", "n_ket", "k_ab", "k_cd", "n_sh", "n_ao",
            *(() if _ordered(spec) else (PERM,)), "tau",
            "bra_prim", *(("AB",) if l1 else ()), "G_ab", "bra_sh", "bra_ao",
            "ket_prim", *(("CD",) if l3 else ()), "G_cd", "ket_sh", "ket_ao",
            "Dsh", "D", K_OUT, *((OUT,) if eris else ()))


# --- the pieces --------------------------------------------------------------


def _orients(spec: Spec, density: tuple[int, int]) -> tuple[_Orient, ...]:
    """One deposit per orientation: `K` on the built positions holding the
    ordered quartet's `K_OUTPUT`, `D` read at its `density` positions."""
    out = []
    for g in orientations(spec):
        inv = inverse(orbits.GROUP[g])
        out.append(_Orient(g, (inv[K_OUTPUT[0]], inv[K_OUTPUT[1]]),
                           (inv[density[0]], inv[density[1]])))
    return tuple(out)


def _deposit(o: _Orient, n: Sequence[int], comps, targets) -> tuple[Stmt, ...]:
    """`K[jpos] += D[dpos] x` for every ERI, through a thread-private
    buffer, then one `AtomicAdd` per `K` element (`kernel_hj._deposit`'s
    shape, into `K`)."""
    (r0, r1), (t0, t1) = o.jpos, o.dpos

    def kbuf(i: int, j: int) -> str:
        return f"x{o.g}_{i}_{j}"

    def dval(k: int, l: int) -> str:  # noqa: E741
        return f"d{o.g}_{k}_{l}"

    out: list[Stmt] = [
        Note(f"orientation GROUP[{o.g}] = {orbits.GROUP[o.g]}: K lands on "
             f"built positions {o.jpos}, D read at {o.dpos}"),
        *(_F(dval(k, l), Index("D", (_ao(t0, k), _ao(t1, l))))
          for k in range(n[t0]) for l in range(n[t1])),  # noqa: E741
        *(_F(kbuf(i, j), Lit(0.0)) for i in range(n[r0]) for j in range(n[r1])),
    ]
    for t, idx in zip(targets, comps, strict=True):
        f = Var(kbuf(idx[r0], idx[r1]))
        out.append(Assign(f, _c(Fn.ADD, f, _c(
            Fn.MUL, Var(dval(idx[t0], idx[t1])), Var(hrr_name(t))))))
    out += [AtomicAdd(Index(K_OUT, (_ao(r0, i), _ao(r1, j))), Var(kbuf(i, j)))
            for i in range(n[r0]) for j in range(n[r1])]
    return tuple(out)


# --- the kernel --------------------------------------------------------------


def build_hk_kernel(cls: AmClass | Spec, *, name: str | None = None,
                    eris: bool = False,
                    density: tuple[int, int] = K_DENSITY) -> Kernel:
    """The HGP `K` kernel of one class (module docstring). `cls` is a
    `Target.K`, `Engine.HGP` `Spec` (or `(la, lc, lb, ld)`, canonical);
    `eris=True` adds the ERI tap `g[bra, ket, i]` in the built class's
    `hgp.hrr.targets` order; `density` is the ordered positions whose `D`
    block the deposit reads -- `(1, 3)`, or a test's mutation."""
    spec = _spec_of(cls)
    density = tuple(density)  # type: ignore[assignment]
    if (len(density) != 2 or density[0] == density[1]
            or any(type(x) is not int or not 0 <= x < 4 for x in density)):
        raise ValueError(f"density {density!r} is not two distinct positions")
    am = built_class(spec)
    ordered = _ordered(spec)
    l0, l1, l2, l3 = am
    vdag, hdag = vrr.build(am), hrr.build(am)
    bnd = boundary_order(am)
    if set(bnd) != hdag.boundary:  # #274 holds this; refuse, never misread
        raise ValueError(f"{am}: HRR boundary is not the VRR target set")
    prelude, fundamental = fundamental_prelude(sum(am))
    vbody = tuple(vrr_body(vdag, fundamental))
    hbody = tuple(hrr_body(hdag, lambda e, f: Var(acc_name(e, f))))
    vf = _factors(vbody)
    vneed = frozenset(vf | {i2 for n, i2, _ in _DERIVED if n in vf})
    hneed = frozenset(_factors(hbody))
    n = tuple(len(vrr.cartesians(x)) for x in am)
    orients = _orients(spec, density)
    shells = {p for o in orients for p in o.dpos}
    aos = {p for o in orients for p in (*o.jpos, *o.dpos)}
    bra, k = Var("bra"), Var(KET)
    if name is None:
        name = hk_name(spec)

    def bra_prim(f: int) -> Expr:
        return Index(AB_SH, (Var(KAB), Lit(f)))

    def ket_prim(f: int) -> Expr:
        return Index("ket_prim", (k, Var(KCD), Lit(f)))

    # -- the screen: G_bra G_ket Dsh[density pair] ---------------------------
    def dsh(o: _Orient) -> Expr:
        return Index("Dsh", (Var(_SH[o.dpos[0]]), Var(_SH[o.dpos[1]])))

    reads = {dsh(o) for o in orients}
    if len(reads) == 1:
        dmax: tuple[Stmt, ...] = (_F("dmax", dsh(orients[0])),)
    else:
        dmax = (_F("dmax", Lit(0.0)),
                *_membership([(o.g, (Assign(Var("dmax"), dsh(o)),))
                              for o in orients]))

    # -- the primitive nest, bra outer (kernel_hj's, statement for statement) -
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
        Note("eq 27: c += [e0|f0]^(0), every boundary value, contracted in "
             "registers before the HRR (pin 5)"),
        *(Assign(Var(acc_name(e, f)), _c(Fn.ADD, Var(acc_name(e, f)),
                                         Var(vrr_name((e, f, 0)))))
          for e, f in bnd),
    )
    per_kab = (
        Note("the bra primitive pair, from shared memory"),
        *(_F(f"Pb{d}", bra_prim(_P + d)) for d in range(3)),
        *(_F(Factor.pa(d).value, bra_prim(_X + d)) for d in range(3)
          if Factor.pa(d).value in vneed),
        _F("p", bra_prim(_ZETA)),
        _F("Ub", bra_prim(_U)),
        For(KCD, Lit(0), Var("k_cd"), Lit(1), per_kcd),
    )

    comps, targets = _comps(am), hdag.targets
    deposits = [(o.g, _deposit(o, n, comps, targets)) for o in orients]
    deposit = deposits[0][1] if ordered else _membership(deposits)
    tap: tuple[Stmt, ...] = ()
    if eris:
        tap = (Note("the ERI tap: every (x0 x1|x2 x3) of the built class, "
                    "hgp.hrr.targets order"),
               *(Assign(Index(OUT, (bra, k, Lit(i))), Var(hrr_name(t)))
                 for i, t in enumerate(targets)))
    work = (
        *(_F(Factor.cd(d).value, Index("CD", (k, Lit(d)))) for d in range(3)
          if Factor.cd(d).value in hneed),
        Note(f"the {len(bnd)} contracted boundary accumulators "
             "(hgp.vrr.target_set, the HRR's leaves), zeroed per ket"),
        *(_F(acc_name(e, f), Lit(0.0)) for e, f in bnd),
        Note("the K_AB x K_CD primitive nest: bra outer (shared), ket inner "
             "(global) -- ao/hgp_drive's (i, j, k, l) order"),
        For(KAB, Lit(0), Var("k_ab"), Lit(1), per_kab),
        *hbody,
        *tap,
        Note("the deposit: K_ab += D_cd (ac|bd), one ordered quartet, no "
             "weight, no mirror, by atomic add"),
        *deposit,
    )
    per_ket = (
        *(_I(_SH[s], _record("ket_sh", k, s)) for s in (2, 3) if s in shells),
        Note("the one-block screen: G_bra G_ket max|D| over the density block "
             "the deposit reads"),
        *dmax,
        _F("bound", _c(Fn.MUL, Var("Gb"), _c(
            Fn.MUL, Index("G_cd", (k,)), Var("dmax")))),
        If(_c(Fn.LT, Var("bound"), Var("tau")), (Note(SKIP),), (
            *(_I(_AO[s], _record("ket_ao", k, s)) for s in (2, 3)
              if s in aos),
            *work)),
    )
    per_bra = (
        *(_F(Factor.ab(d).value, Index("AB", (bra, Lit(d)))) for d in range(3)
          if Factor.ab(d).value in hneed),
        _F("Gb", Index("G_ab", (bra,))),
        *(_I(_SH[s], _record("bra_sh", bra, s)) for s in (0, 1) if s in shells),
        *(_I(_AO[s], _record("bra_ao", bra, s)) for s in (0, 1) if s in aos),
        Note(STAGING),
        For(STAGE, Lit(0), Var("k_ab"), Lit(1), tuple(
            Assign(Index(AB_SH, (Var(STAGE), Lit(f))),
                   Index("bra_prim", (bra, Var(STAGE), Lit(f))))
            for f in range(len(REC)))),
        Note("one thread per contracted quartet: consecutive threads take "
             "consecutive kets ([Fock] §3.2)"),
        For(KET, Call(Fn.THREAD_X), Var("n_ket"), Call(Fn.BLOCK_DIM_X),
            per_ket),
    )
    mode = ("ordered: its own class, the identity deposit" if ordered else
            f"canonical: the orbit representative, {len(orients)} "
            "orientation(s) selected by perm")
    body = (
        Note(f"(x0 x1|x2 x3) by the HGP recurrence, K class ({l0}{l1}|{l2}{l3}) "
             f"[{mode}]: VRR eqs 25-26 inside the K_AB x K_CD nest, HRR eq 28 "
             "after it, then K_ab += D_cd (ac|bd), split (#405)"),
        *(Note(lever_provenance(lv)) for lv in spec.kernel_body_levers()),
        Note("launch: grid = n_bra, blockDim = (threads, 1); one block per "
             "bra shell pair"),
        Decl("bra", Scalar.I32, init=Call(Fn.BLOCK_X)),
        Decl(AB_SH, Scalar.F64, (Lit(MAX_PRIM_PAIRS), Lit(len(REC))),
             Space.SHARED),
        If(_c(Fn.LT, bra, Var("n_bra")), per_bra),
    )
    return Kernel(name, _params(spec, am, hdag.n_eri, eris), body)


def _params(spec: Spec, am: AmClass, n_eri: int, eris: bool
            ) -> tuple[Param, ...]:
    _, l1, _, l3 = am
    nb, nk, kab, kcd = (Var(x) for x in ("n_bra", "n_ket", "k_ab", "k_cd"))
    nsh, nao = Var("n_sh"), Var("n_ao")

    def f64(x: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(x, Scalar.F64, array=True, intent=intent, shape=shape)

    def i32(x: str, *shape: Expr) -> Param:
        return Param(x, Scalar.I32, array=True, shape=shape)

    out = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("k_ab", Scalar.I32), Param("k_cd", Scalar.I32),
        Param("n_sh", Scalar.I32), Param("n_ao", Scalar.I32),
        *(() if _ordered(spec) else (Param(PERM, Scalar.I32),)),
        Param("tau", Scalar.F64),
        f64("bra_prim", nb, kab, Lit(len(REC))),
        *((f64("AB", nb, Lit(3)),) if l1 else ()),
        f64("G_ab", nb), i32("bra_sh", nb, Lit(2)), i32("bra_ao", nb, Lit(2)),
        f64("ket_prim", nk, kcd, Lit(len(REC))),
        *((f64("CD", nk, Lit(3)),) if l3 else ()),
        f64("G_cd", nk), i32("ket_sh", nk, Lit(2)), i32("ket_ao", nk, Lit(2)),
        f64("Dsh", nsh, nsh), f64("D", nao, nao),
        f64(K_OUT, nao, nao, intent=Intent.OUT),
        *((f64(OUT, nb, nk, Lit(n_eri), intent=Intent.OUT),) if eris else ()),
    )
    assert tuple(p.name for p in out) == params_of(spec, eris=eris)
    return out


# --- the counts --------------------------------------------------------------


@dataclass(frozen=True)
class Counts:
    """Statement counts of one HGP K kernel (`kernel_hj.Counts`' fields)."""

    vrr: int  # one Decl per VRR node, in the innermost loop
    boundary: int  # accumulators: one Decl and one += each
    hrr: int  # one Decl per HRR node, boundary reads included
    hrr_interior: int  # the two-term eq 28 updates among them
    eri: int  # each read once per orientation's deposit
    atomic: int  # AtomicAdd statements: K's elements, per orientation


def kernel_counts(cls: AmClass | Spec) -> Counts:
    """The counts the built class's DAGs predict, and the deposits' size."""
    spec = _spec_of(cls)
    am = built_class(spec)
    v, h = vrr.build(am), hrr.build(am)
    n = [len(vrr.cartesians(x)) for x in am]
    atomic = sum(n[o.jpos[0]] * n[o.jpos[1]]
                 for o in _orients(spec, K_DENSITY))
    return Counts(v.n_nodes, h.n_boundary, h.n_statements, h.n_interior,
                  h.n_eri, atomic)


def counted(kernel: Kernel) -> Counts:
    """The same counts, read off a built kernel by statement kind."""
    stmts = list(walk(kernel.body))
    decls = [s for s in stmts if isinstance(s, Decl)]
    vrr_n = sum(1 for s in decls if s.name.startswith("V"))
    acc = sum(1 for s in decls if s.name.startswith("C")
              and s.name not in ("CD_x", "CD_y", "CD_z"))
    hrr_d = [s for s in decls if s.name.startswith("H")]
    interior = sum(1 for s in hrr_d if isinstance(s.init, Weighted))
    reads: dict[str, int] = {}
    n_dep = sum(1 for s in stmts if isinstance(s, Note)
                and s.prov.startswith("orientation GROUP["))
    for s in stmts:
        if (isinstance(s, Assign) and isinstance(s.dst, Var)
                and s.dst.name.startswith("x")):
            x = s.value.args[1].args[1]  # x = x + d * eri
            reads[x.name] = reads.get(x.name, 0) + 1
    if any(c != n_dep for c in reads.values()):
        raise ValueError("an ERI is not read once per deposit")
    atomic = sum(1 for s in stmts if isinstance(s, AtomicAdd))
    return Counts(vrr_n, acc, len(hrr_d), interior, len(reads), atomic)
