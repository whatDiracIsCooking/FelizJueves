"""The fused F kernel's lever seams: `FHooks`, `FContext`, the defaults (#368).

`kernel_f.build_f_kernel` is a SKELETON: it lays out what no milestone-13
lever moves (the head notes, the lever provenance notes, the launch note, the
bra index and the `bra < n_bra` guard) and asks an `FHooks` for every part a
lever does move. Each hook's default below is the code `kernel_f.py` held
before #368, moved here statement for statement, so the literal kernel is
byte-identical (the 21 goldens per backend, test/golden/fock_digests.json,
and test/test_f_seams.py's digest of every class through L = 4, both
backends, with and without the tap, against the pre-#368 tree).

THE NEST, AS HOOKS (kernel_f's module docstring has the nest itself):

    shared_decl(ctx)          the shared |ab) staging array        (#351's
                              reduction buffer beside it)
    per_bra(ctx)              everything under `bra < n_bra`: bra_setup,
                              stage, ket_loop, flush_bra, in that order
      bra_setup(ctx)          AB, G_ab, the shell indices and AO offsets,
                              the a = b flag, kmin, d_ab, f_ab = 0
      stage(ctx)              the |ab) copy into shared memory
      ket_loop(ctx)           the strided loop over |cd), body `per_ket`
      per_ket(ctx)            the canonical-half skip around `screen`
      screen(ctx)             the six-block screen and its SKIP; its orelse
                              is `work` (#352: a sorted key and a break)
      work(ctx)               one contracted quartet: CD, boundary_init,
                              nest, contracted
      boundary_init(ctx)      the boundary accumulators' declaration and
                              zeroing (#350: a thread-private
                              `Space.LOCAL` `boundary[]` instead)
      nest(ctx)               the K_CD x K_AB primitive nest (#350: one
                              nest per boundary group)
      ket_prim(ctx)           the ket primitive pair's reads
      quartet_scalars(ctx)    the bra primitive reads, the quartet geometry
                              and scalars, `pair`, the Boys prelude
      vrr(ctx)                the VRR statements (#354: ETR; #357: a
                              per-group schedule). Read through `ctx.vbody`
      accumulate(ctx)         eq 27, c += [e0|f0]^(0) (#350: into the
                              staged boundary)
      boundary_read(ctx, e, f)
                              the HRR's read of one boundary value (#350)
      hrr(ctx)                the HRR statements, read through `ctx.hbody`
                              (#356 sliced HRR, #361 sub-slices)
      contracted(ctx)         after the nest: HRR, tap, digest -- the
                              region #356 slices per bra pair
      eri(ctx, t)             the value of ERI target `t` (tap and digest)
      tap(ctx)                the eris=True writes of `g`
      digest(ctx)             digest_setup, zero_buffers, predigest,
                              flush_ket
      digest_setup(ctx)       deg, s_j, s_k, the ket AO offsets, the
                              weighted densities
      zero_buffers(ctx)       the five ket-dependent buffers, zeroed
                              (#360: the reduced per-slice set)
      predigest(ctx)          one `predigest_eri` per target, hrr order
      predigest_eri(ctx, t, idx)
                              the six multiply-adds of one ERI (#359:
                              called at the ERI's production site)
      flush_ket(ctx)          the five ket buffers' AtomicAdd (#351, #358)
      flush_bra(ctx)          f_ab's one AtomicAdd flush per thread
                              (#351's block reduction, #358's scatter)
    field(ctx, arr, idx)      EVERY read of a packed pair record --
                              bra_prim, ab_sh, ket_prim, AB, CD, G_ab, G_cd,
                              bra_sh, bra_ao, ket_sh, ket_ao (#353's typed
                              struct views); D and Dsh are not records
    real(ctx, region)         the floating type of the kernel's own
                              scalars per region, "vrr", "boundary", "hrr"
                              or "fock" (#355's FP32 VRR/boundary/HRR)
    cast(ctx, e, region)      each such scalar's init, so a narrowing lever
                              casts its region's SOURCES -- the record reads
                              and the literals; a local of the region
                              already has its type. `ir`'s precision hygiene
                              refuses a `real` moved without its `cast`, at
                              build. The recurrence's own statements are
                              retyped by wrapping `vrr` / `hrr`
    params(ctx)               the parameter list (#353, #358)
    finish(ctx, kernel)       a last rewrite of the built `Kernel`

A hook receives the `FContext`, which carries everything derived from the
class -- the two DAGs, the boundary order, the Boys prelude, the digestion's
names and flags -- and `ctx.hooks`, so a default calls its siblings through
the COMPOSED set (a `field` override is seen by every default that reads a
record; a `boundary_read` override by the default `hrr`).

A LEVER IS A MODULE: `felizjueves/f_levers/<name>.py` exposing
`apply(hooks, spec) -> FHooks` (`dataclasses.replace` of the hooks it moves),
registered by name in `f_levers.REGISTRY`. `kernel_f` resolves a spec's
kernel-body levers through `f_levers.hooks_for` and never changes for one.

Like `kernel_f`, nothing here imports `plan/`, `ao/`, the interpreter,
`reference/`, a backend or a device; the `Kernel` it helps build is
arch-neutral.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from felizjueves.hgp import hrr as _hrr
from felizjueves.hgp import vrr as _vrr
from felizjueves.hgp.recurrence import (
    BOYS_ARRAY,
    PAIR,
    PQ_PARAMS,
    THETA,
    boundary_order,
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
from felizjueves.spec import Spec

__all__ = [
    "AB_SH",
    "AO_OF",
    "BLOCKS",
    "BOYS_ARRAY",
    "DEFAULT_HOOKS",
    "FOCK",
    "HALF",
    "KAB",
    "KCD",
    "KET",
    "MAX_PRIM_PAIRS",
    "OUT",
    "REC",
    "REGIONS",
    "SH_OF",
    "SIX_BLOCK",
    "SKIP",
    "STAGE",
    "STAGING",
    "FContext",
    "FHooks",
    "acc_name",
    "cart_names",
    "factors",
    "params_of",
]

MAX_PRIM_PAIRS = 128  # the shared staging cap (kernel_f's docstring)
REC = ("P_x", "P_y", "P_z", "X_x", "X_y", "X_z", "zeta", "U")
AB_SH = "ab_sh"  # the staged |ab) record, Space.SHARED
OUT = "g"  # the ERI tap (eris=True)
FOCK = "F"  # the two-electron Fock matrix, AtomicAdd-ed, closed on the host
KET = "k"  # the strided ket loop: one thread per contracted quartet
KCD, KAB = "kcd", "kab"  # the primitive loops, ket outer, bra inner
STAGE = "i"

_P, _X, _ZETA, _U = 0, 3, 6, 7  # field offsets in REC

# The six blocks, as unordered pairs of positions in (a, b, c, d) = (0, 1, 2,
# 3), each written in order; the kernel's buffer names follow them.
BLOCKS = ((0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3))
_LETTER = "abcd"
SH_OF = ("sa", "sb", "sc", "sd")  # the four shell indices
AO_OF = ("oa", "ob", "oc", "od")  # the four AO offsets

# `real`'s regions: the kernel's own floating scalars, by where they live.
REGIONS = ("vrr", "boundary", "hrr", "fock")

SKIP = (
    "screen: skip this quartet, never break ([Fock] §3.2; fock/skip-not-"
    "break). The six-block D_max is not monotone along the batch sort, so a "
    "screened quartet says nothing of the kets after it")

HALF = (
    "the canonical half: in a diagonal launch (diag = 1, one batch on both "
    "sides) the quartet (bra, k < bra) is the thread for (k, bra)'s; skip it, "
    "never break")

SIX_BLOCK = (
    "the six-block screen: dmax is the largest |w| Dsh over the six density "
    "blocks the quartet digests into (ab, cd, ac, ad, bc, bd), so G_ab G_cd "
    "dmax bounds every term it deposits")

STAGING = (
    "stage |ab) in shared memory: every thread copies the whole record (the "
    "IR has no barrier, so a cooperative load could not be read safely); "
    f"the degree k_ab is at most MAX_PRIM_PAIRS = {MAX_PRIM_PAIRS}, which "
    "kernel_f.pack enforces")


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def acc_name(e: Cart, f: Cart) -> str:
    """The contracted accumulator of boundary `(e0|f0)`: `C{e}__{f}`."""
    return "C" + "__".join("_".join(map(str, v)) for v in (e, f))


def factors(stmts: Sequence[Stmt]) -> set[str]:
    """Every non-unit factor name a `Weighted` in `stmts` reads."""
    out: set[str] = set()
    for s in stmts:
        if isinstance(s, Decl) and isinstance(s.init, Weighted):
            out |= {t.g.name_in_kernel for t in s.init.terms
                    if t.g.name_in_kernel is not None}
    return out


def params_of(cls: AmClass, *, eris: bool = False) -> tuple[str, ...]:
    """The kernel's parameter names, in order. `AB` only when `lb > 0` (the
    bra HRR reads it), `CD` only when `ld > 0`; `g` only with the tap."""
    _, lb, _, ld = cls
    return ("n_bra", "n_ket", "k_ab", "k_cd", "n_sh", "n_ao", "diag", "tau",
            "bra_prim", *(("AB",) if lb else ()), "G_ab", "bra_sh", "bra_ao",
            "ket_prim", *(("CD",) if ld else ()), "G_cd", "ket_sh", "ket_ao",
            "Dsh", "D", FOCK, *((OUT,) if eris else ()))


def cart_names(cls: AmClass) -> tuple[tuple[int, ...], ...]:
    """Per ERI target, its four component indices within their shells."""
    pos = [{c: i for i, c in enumerate(_vrr.cartesians(x))} for x in cls]
    return tuple(tuple(pos[s][v] for s, v in enumerate(t))
                 for t in _hrr.targets(cls))


# ---------------------------------------------------------------------------
# The context: what the skeleton derives from the class, once
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FContext:
    """One kernel's derived quantities, shared by every hook.

    `weights` is the digestion's `kernel_f.Weights` (read by attribute:
    `coulomb`, `exchange`, `degeneracy`, `of(block)`). `vbody` / `hbody` are
    the `vrr` / `hrr` hooks' statements, formed once on first read, and
    `vneed` / `hneed` the factors they read -- so a scalar the recurrence a
    lever substitutes does not read is not declared (#92's warning rule).
    """

    spec: Spec
    cls: AmClass
    name: str
    weights: object
    eris: bool
    hooks: FHooks
    vdag: object
    hdag: object
    bnd: tuple
    prelude: tuple
    fundamental: object
    extra: dict = field(default_factory=dict)  # a lever's own derived data

    @classmethod
    def of(cls, spec: Spec, am: AmClass, name: str, weights: object,
           eris: bool, hooks: FHooks) -> FContext:
        vdag, hdag = _vrr.build(am), _hrr.build(am)
        bnd = boundary_order(am)
        if set(bnd) != hdag.boundary:  # #274 holds this; refuse, never misread
            raise ValueError(f"{am}: HRR boundary is not the VRR target set")
        prelude, fundamental = fundamental_prelude(sum(am))
        return cls(spec=spec, cls=am, name=name, weights=weights, eris=eris,
                   hooks=hooks, vdag=vdag, hdag=hdag, bnd=bnd,
                   prelude=tuple(prelude), fundamental=fundamental)

    # --- the recurrences, through their hooks -------------------------------
    @cached_property
    def vbody(self) -> tuple[Stmt, ...]:
        return tuple(self.hooks.vrr(self))

    @cached_property
    def hbody(self) -> tuple[Stmt, ...]:
        return tuple(self.hooks.hrr(self))

    @cached_property
    def vneed(self) -> frozenset[str]:
        """The VRR's factors, plus the `1/(2x)` a derived product needs even
        where no term reads it alone."""
        need = factors(self.vbody)
        return frozenset(need | {inv2 for n, inv2, _ in _DERIVED
                                 if n in need})

    @cached_property
    def hneed(self) -> frozenset[str]:
        return frozenset(factors(self.hbody))

    # --- the class's shapes and the digestion's flags -----------------------
    @cached_property
    def ncart(self) -> tuple[int, ...]:
        return tuple(len(_vrr.cartesians(x)) for x in self.cls)

    @cached_property
    def comps(self) -> tuple[tuple[int, ...], ...]:
        return cart_names(self.cls)

    @property
    def deg_bra(self) -> bool:  # else a != b at build time
        return bool(self.weights.degeneracy and self.cls[0] == self.cls[1])

    @property
    def deg_ket(self) -> bool:
        return bool(self.weights.degeneracy and self.cls[2] == self.cls[3])

    @property
    def deg_pq(self) -> bool:  # else P != Q
        return bool(self.weights.degeneracy
                    and tuple(self.cls[:2]) == tuple(self.cls[2:]))

    # --- the names every part of the kernel shares --------------------------
    @property
    def bra(self) -> Var:
        return Var("bra")

    @property
    def k(self) -> Var:
        return Var(KET)

    def scalar(self, n: str, e: Expr, region: str) -> Decl:
        """A floating local of `region`, typed by the `real` hook, its init
        passed through the `cast` hook."""
        return Decl(n, self.hooks.real(self, region),
                    init=self.hooks.cast(self, e, region))

    @staticmethod
    def index(n: str, e: Expr) -> Decl:  # a shell index or an AO offset
        return Decl(n, Scalar.I32, init=e)

    @staticmethod
    def blk(b: tuple[int, int]) -> str:
        return _LETTER[b[0]] + _LETTER[b[1]]

    @staticmethod
    def comp(b: tuple[int, int]) -> tuple[int, int]:
        return tuple(sorted({0, 1, 2, 3} - set(b)))  # type: ignore[return-value]

    def elems(self, b: tuple[int, int]):
        return ((i, j) for i in range(self.ncart[b[0]])
                for j in range(self.ncart[b[1]]))

    def buf(self, b, i, j) -> str:
        return f"f_{self.blk(b)}_{i}_{j}"

    def dens(self, b, i, j) -> str:  # the weighted density element w deg D
        return f"w_{self.blk(b)}_{i}_{j}"

    @staticmethod
    def raw(i, j) -> str:  # the bra's own block, loaded once per bra
        return f"d_ab_{i}_{j}"

    @staticmethod
    def ao(b, i, j) -> tuple[Expr, Expr]:
        return (_c(Fn.ADD, Var(AO_OF[b[0]]), Lit(i)),
                _c(Fn.ADD, Var(AO_OF[b[1]]), Lit(j)))

    def one_half_if_same(self, flag: str, s: str, t: str) -> list[Stmt]:
        """`flag = 1/2` when shells `s == t`: neither `s < t` nor `t < s`."""
        return [self.scalar(flag, Lit(0.5), "fock"),
                If(_c(Fn.LT, Var(s), Var(t)), (Assign(Var(flag), Lit(1.0)),)),
                If(_c(Fn.LT, Var(t), Var(s)), (Assign(Var(flag), Lit(1.0)),))]

    def read(self, arr: str, *idx: Expr) -> Expr:
        """A packed pair-record read, through the `field` hook."""
        return self.hooks.field(self, arr, tuple(idx))


_DERIVED = ((Factor.INV2P_ALPHA_OVER_P.value, Factor.INV2P.value, "p"),
            (Factor.INV2Q_ALPHA_OVER_Q.value, Factor.INV2Q.value, "q"))


# ---------------------------------------------------------------------------
# The frozen defaults: kernel_f.py's code before #368, statement for statement
# ---------------------------------------------------------------------------


def field_read(ctx: FContext, arr: str, idx: tuple[Expr, ...]) -> Expr:
    """A positional read into the flat packed array: `arr[idx]`."""
    return Index(arr, idx)


def real(ctx: FContext, region: str) -> Scalar:
    """FP64 everywhere: the literal kernel has one floating type."""
    return Scalar.F64


def cast(ctx: FContext, e: Expr, region: str) -> Expr:
    """A regional scalar's init, unchanged: nothing crosses a precision."""
    return e


def shared_decl(ctx: FContext) -> tuple[Stmt, ...]:
    """`ab_sh[MAX_PRIM_PAIRS, REC]`, shared, a literal extent (8 KiB)."""
    return (Decl(AB_SH, Scalar.F64, (Lit(MAX_PRIM_PAIRS), Lit(len(REC))),
                 Space.SHARED),)


def per_bra(ctx: FContext) -> tuple[Stmt, ...]:
    """Under `bra < n_bra`: the bra's setup, the staging, the ket loop, the
    `f_ab` flush."""
    h = ctx.hooks
    return (*h.bra_setup(ctx), *h.stage(ctx), *h.ket_loop(ctx),
            *h.flush_bra(ctx))


def bra_setup(ctx: FContext) -> tuple[Stmt, ...]:
    bra, ab = ctx.bra, BLOCKS[0]
    return (
        *(ctx.scalar(Factor.ab(d).value, ctx.read("AB", bra, Lit(d)), "hrr")
          for d in range(3) if Factor.ab(d).value in ctx.hneed),
        ctx.scalar("Gb", ctx.read("G_ab", bra), "fock"),
        *(ctx.index(SH_OF[s], ctx.read("bra_sh", bra, Lit(s))) for s in (0, 1)),
        *(ctx.index(AO_OF[s], ctx.read("bra_ao", bra, Lit(s))) for s in (0, 1)),
        *(ctx.one_half_if_same("hb", "sa", "sb") if ctx.deg_bra else ()),
        Decl("kmin", Scalar.I32, init=_c(Fn.MUL, Var("diag"), bra)),
        Note("the bra's own density block, read once; f_ab spans the ket "
             "loop and is flushed once"),
        *(ctx.scalar(ctx.raw(i, j), Index("D", ctx.ao(ab, i, j)), "fock")
          for i, j in ctx.elems(ab)),
        *(ctx.scalar(ctx.buf(ab, i, j), Lit(0.0), "fock")
          for i, j in ctx.elems(ab)),
    )


def stage(ctx: FContext) -> tuple[Stmt, ...]:
    """Every thread copies the whole `|ab)` record (the IR has no barrier)."""
    return (
        Note(STAGING),
        For(STAGE, Lit(0), Var("k_ab"), Lit(1), tuple(
            Assign(Index(AB_SH, (Var(STAGE), Lit(f))),
                   ctx.read("bra_prim", ctx.bra, Var(STAGE), Lit(f)))
            for f in range(len(REC)))),
    )


def ket_loop(ctx: FContext) -> tuple[Stmt, ...]:
    return (
        Note("one thread per contracted quartet: consecutive threads take "
             "consecutive |cd) ([Fock] §3.2, pin 7)"),
        For(KET, Call(Fn.THREAD_X), Var("n_ket"), Call(Fn.BLOCK_DIM_X),
            tuple(ctx.hooks.per_ket(ctx))),
    )


def per_ket(ctx: FContext) -> tuple[Stmt, ...]:
    """The canonical half: skip `k < diag * bra`, else the screen."""
    return (If(_c(Fn.LT, ctx.k, Var("kmin")), (Note(HALF),),
               tuple(ctx.hooks.screen(ctx))),)


def screen(ctx: FContext) -> tuple[Stmt, ...]:
    """The six-block screen, a SKIP: `bound < tau` holds only a `Note`."""
    k, w = ctx.k, ctx.weights

    def weighted_block(b: tuple[int, int]) -> Expr:
        return _c(Fn.MUL, Lit(abs(w.of(b))),
                  Index("Dsh", (Var(SH_OF[b[0]]), Var(SH_OF[b[1]]))))

    out: list[Stmt] = [
        *(ctx.index(SH_OF[s], ctx.read("ket_sh", k, Lit(s - 2)))
          for s in (2, 3)),
        Note(SIX_BLOCK),
        ctx.scalar("dmax", weighted_block(BLOCKS[0]), "fock"),
    ]
    for b in BLOCKS[1:]:
        t = f"dmax_{ctx.blk(b)}"
        out += [ctx.scalar(t, weighted_block(b), "fock"),
                If(_c(Fn.LT, Var("dmax"), Var(t)),
                   (Assign(Var("dmax"), Var(t)),))]
    out += [
        ctx.scalar("bound", _c(Fn.MUL, Var("Gb"), _c(
            Fn.MUL, ctx.read("G_cd", k), Var("dmax"))), "fock"),
        If(_c(Fn.LT, Var("bound"), Var("tau")), (Note(SKIP),),
           tuple(ctx.hooks.work(ctx))),
    ]
    return tuple(out)


def work(ctx: FContext) -> tuple[Stmt, ...]:
    """One unscreened contracted quartet: CD, the boundary, the nest, then
    everything that reads the contracted values."""
    h = ctx.hooks
    return (
        *(ctx.scalar(Factor.cd(d).value, ctx.read("CD", ctx.k, Lit(d)), "hrr")
          for d in range(3) if Factor.cd(d).value in ctx.hneed),
        *h.boundary_init(ctx),
        *h.nest(ctx),
        *h.contracted(ctx),
    )


def boundary_init(ctx: FContext) -> tuple[Stmt, ...]:
    return (
        Note(f"the {len(ctx.bnd)} contracted boundary accumulators "
             "(hgp.vrr.target_set, the HRR's leaves), zeroed per ket"),
        *(ctx.scalar(acc_name(e, f), Lit(0.0), "boundary") for e, f in ctx.bnd),
    )


def nest(ctx: FContext) -> tuple[Stmt, ...]:
    """`for kcd: ket_prim; for kab: quartet_scalars, VRR, accumulate`."""
    h = ctx.hooks
    quartet = (*h.quartet_scalars(ctx), *ctx.vbody, *h.accumulate(ctx))
    per_kcd = (*h.ket_prim(ctx),
               For(KAB, Lit(0), Var("k_ab"), Lit(1), quartet))
    return (
        Note("the K_CD x K_AB primitive nest: ket outer (global), bra inner "
             "(shared)"),
        For(KCD, Lit(0), Var("k_cd"), Lit(1), per_kcd),
    )


def ket_prim(ctx: FContext) -> tuple[Stmt, ...]:
    def ket(f: int) -> Expr:
        return ctx.read("ket_prim", ctx.k, Var(KCD), Lit(f))

    return (
        Note("the ket primitive pair, from global memory (this thread's |cd))"),
        *(ctx.scalar(f"Qk{d}", ket(_P + d), "vrr") for d in range(3)),
        *(ctx.scalar(Factor.qc(d).value, ket(_X + d), "vrr") for d in range(3)
          if Factor.qc(d).value in ctx.vneed),
        ctx.scalar("q", ket(_ZETA), "vrr"),
        ctx.scalar("Uq", ket(_U), "vrr"),
    )


def quartet_scalars(ctx: FContext) -> tuple[Stmt, ...]:
    """The bra primitive pair, the quartet geometry and every scalar the
    VRR reads, `pair`, then the Boys prelude (kernel_f's docstring, "THE
    QUARTET SCALARS")."""
    vneed = ctx.vneed

    def sh(f: int) -> Expr:
        return ctx.read(AB_SH, Var(KAB), Lit(f))

    def scalar(n: str, e: Expr) -> Decl:
        return ctx.scalar(n, e, "vrr")

    wp = [Factor.wp(d).value for d in range(3)]
    wq = [Factor.wq(d).value for d in range(3)]
    out: list[Stmt] = [
        Note("the bra primitive pair, from shared memory"),
        *(scalar(f"Pb{d}", sh(_P + d)) for d in range(3)),
        *(scalar(Factor.pa(d).value, sh(_X + d)) for d in range(3)
          if Factor.pa(d).value in vneed),
        scalar("p", sh(_ZETA)),
        scalar("Ub", sh(_U)),
        Note("quartet geometry ([Fock] §3.1; hgp/recurrence.py, 'WHAT THE "
             "SCALARS ARE'): W - P and W - Q with the DERIVED sign"),
        *(scalar(n, _c(Fn.SUB, Var(f"Pb{d}"), Var(f"Qk{d}")))
          for d, n in enumerate(PQ_PARAMS)),
        scalar("ppq", _c(Fn.ADD, Var("p"), Var("q"))),
        scalar("rho", _c(Fn.DIV, _c(Fn.MUL, Var("p"), Var("q")), Var("ppq"))),
        scalar(THETA, _c(Fn.SQRT, Var("rho"))),
    ]
    if vneed & set(wp):
        out.append(scalar("q_ppq", _c(Fn.DIV, Var("q"), Var("ppq"))))
    if vneed & set(wq):
        out.append(scalar("p_ppq", _c(Fn.DIV, Var("p"), Var("ppq"))))
    for d in range(3):
        if wp[d] in vneed:
            out.append(scalar(wp[d], _c(Fn.SUB, Lit(0.0), _c(
                Fn.MUL, Var("q_ppq"), Var(PQ_PARAMS[d])))))
        if wq[d] in vneed:
            out.append(scalar(wq[d], _c(
                Fn.MUL, Var("p_ppq"), Var(PQ_PARAMS[d]))))
    for n, x in ((Factor.INV2P.value, "p"), (Factor.INV2Q.value, "q"),
                 (Factor.INV2PQ.value, "ppq")):
        if n in vneed:
            out.append(scalar(n, _c(Fn.DIV, Lit(0.5), Var(x))))
    for n, inv2, x in _DERIVED:
        if n in vneed:
            out.append(scalar(n, _c(Fn.MUL, Var(inv2), _c(
                Fn.DIV, Var("rho"), Var(x)))))
    out += [scalar(PAIR, _c(Fn.MUL, Var("Ub"), Var("Uq"))), *ctx.prelude]
    return tuple(out)


def vrr(ctx: FContext) -> tuple[Stmt, ...]:
    """`hgp.recurrence.vrr_body` over the ket cascade's DAG."""
    return tuple(vrr_body(ctx.vdag, ctx.fundamental))


def accumulate(ctx: FContext) -> tuple[Stmt, ...]:
    """eq 27: `c = c + [e0|f0]^(0)` per boundary value, in registers."""
    return (
        Note("eq 27: c += [e0|f0]^(0), every boundary value, contracted in "
             "registers before the HRR (pin 5)"),
        *(Assign(Var(acc_name(e, f)), _c(
            Fn.ADD, Var(acc_name(e, f)), Var(vrr_name((e, f, 0)))))
          for e, f in ctx.bnd),
    )


def boundary_read(ctx: FContext, e: Cart, f: Cart) -> Expr:
    """The contracted accumulator, a register."""
    return Var(acc_name(e, f))


def hrr(ctx: FContext) -> tuple[Stmt, ...]:
    """`hgp.recurrence.hrr_body`, whole, reading the boundary through
    `boundary_read`."""
    return tuple(hrr_body(ctx.hdag,
                          lambda e, f: ctx.hooks.boundary_read(ctx, e, f)))


def contracted(ctx: FContext) -> tuple[Stmt, ...]:
    """After the nest: the HRR on the sums, the tap, the digestion."""
    h = ctx.hooks
    return (*ctx.hbody, *h.tap(ctx), *h.digest(ctx))


def eri(ctx: FContext, t) -> Expr:
    """ERI target `t`'s value: the HRR's node."""
    return Var(hrr_name(t))


def tap(ctx: FContext) -> tuple[Stmt, ...]:
    if not ctx.eris:
        return ()
    return (
        Note("the ERI tap: every (ab|cd), hgp.hrr.targets order"),
        *(Assign(Index(OUT, (ctx.bra, ctx.k, Lit(i))), ctx.hooks.eri(ctx, t))
          for i, t in enumerate(ctx.hdag.targets)),
    )


def digest(ctx: FContext) -> tuple[Stmt, ...]:
    h = ctx.hooks
    return (*h.digest_setup(ctx), *h.zero_buffers(ctx), *h.predigest(ctx),
            *h.flush_ket(ctx))


def digest_setup(ctx: FContext) -> tuple[Stmt, ...]:
    """`deg`, the two weight scalars, the ket AO offsets and the weighted
    densities (kernel_f's docstring, "THE BUFFERS")."""
    k = ctx.k
    flags = [f for f, on in (("hb", ctx.deg_bra), ("hk", ctx.deg_ket)) if on]
    deg_init: Expr = Lit(1.0)
    for f in flags:
        deg_init = Var(f) if deg_init == Lit(1.0) else _c(
            Fn.MUL, deg_init, Var(f))
    wj, wk = float(ctx.weights.coulomb), float(ctx.weights.exchange)
    out: list[Stmt] = [
        Note("the degeneracy factor 1/|Stab| (module docstring): 1/2 for "
             "a = b, for c = d, and for P = Q"),
        *(ctx.one_half_if_same("hk", "sc", "sd") if ctx.deg_ket else ()),
        ctx.scalar("deg", deg_init, "fock"),
    ]
    if ctx.deg_pq:
        out.append(If(_c(Fn.LT, k, _c(Fn.ADD, Var("kmin"), Var("diag"))),
                      (Note("P = Q: the diagonal launch at k == bra"),
                       Assign(Var("deg"), _c(Fn.MUL, Var("deg"),
                                             Lit(0.5))))))
    out += [
        ctx.scalar("s_j", _c(Fn.MUL, Lit(wj), Var("deg")), "fock"),
        ctx.scalar("s_k", _c(Fn.MUL, Lit(wk), Var("deg")), "fock"),
        *(ctx.index(AO_OF[s], ctx.read("ket_ao", k, Lit(s - 2)))
          for s in (2, 3)),
        Note("the weighted densities: w = (fraction x in-order members) deg "
             "D, the weight folded in once per quartet"),
    ]
    for b in BLOCKS:
        s = "s_j" if b in BLOCKS[:2] else "s_k"
        for i, j in ctx.elems(b):
            src: Expr = (Var(ctx.raw(i, j)) if b == (0, 1)
                         else Index("D", ctx.ao(b, i, j)))
            out.append(ctx.scalar(ctx.dens(b, i, j), _c(Fn.MUL, Var(s), src),
                                  "fock"))
    return tuple(out)


def zero_buffers(ctx: FContext) -> tuple[Stmt, ...]:
    out: list[Stmt] = [Note("the five ket-dependent buffers, zeroed per "
                            "quartet")]
    for b in BLOCKS[1:]:
        out += [ctx.scalar(ctx.buf(b, i, j), Lit(0.0), "fock")
                for i, j in ctx.elems(b)]
    return tuple(out)


def predigest(ctx: FContext) -> tuple[Stmt, ...]:
    out: list[Stmt] = [Note("the predigest: six multiply-adds per ERI, one "
                            "per block, each reading the complementary "
                            "density")]
    for t, idx in zip(ctx.hdag.targets, ctx.comps, strict=True):
        out += ctx.hooks.predigest_eri(ctx, t, idx)
    return tuple(out)


def predigest_eri(ctx: FContext, t, idx: tuple[int, ...]) -> tuple[Stmt, ...]:
    """ERI `t` (components `idx`) into the six buffers, block order."""
    x = ctx.hooks.eri(ctx, t)
    out: list[Stmt] = []
    for b in BLOCKS:
        c = ctx.comp(b)
        f = Var(ctx.buf(b, idx[b[0]], idx[b[1]]))
        out.append(Assign(f, _c(Fn.ADD, f, _c(
            Fn.MUL, Var(ctx.dens(c, idx[c[0]], idx[c[1]])), x))))
    return tuple(out)


def flush_ket(ctx: FContext) -> tuple[Stmt, ...]:
    out: list[Stmt] = [Note("flush the five ket-dependent buffers: AtomicAdd, "
                            "the only atomics besides f_ab's one flush")]
    for b in BLOCKS[1:]:
        out += [AtomicAdd(Index(FOCK, ctx.ao(b, i, j)), Var(ctx.buf(b, i, j)))
                for i, j in ctx.elems(b)]
    return tuple(out)


def flush_bra(ctx: FContext) -> tuple[Stmt, ...]:
    ab = BLOCKS[0]
    return (
        Note("flush f_ab: once per thread, after its whole ket loop"),
        *(AtomicAdd(Index(FOCK, ctx.ao(ab, i, j)), Var(ctx.buf(ab, i, j)))
          for i, j in ctx.elems(ab)),
    )


def params(ctx: FContext) -> tuple[Param, ...]:
    """`params_of(cls, eris=)`, typed and shaped."""
    _, lb, _, ld = ctx.cls
    nb, nk, kab, kcd = (Var(n) for n in ("n_bra", "n_ket", "k_ab", "k_cd"))
    nsh, nao = Var("n_sh"), Var("n_ao")

    def f64(n: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(n, Scalar.F64, array=True, intent=intent, shape=shape)

    def i32(n: str, *shape: Expr) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    out = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("k_ab", Scalar.I32), Param("k_cd", Scalar.I32),
        Param("n_sh", Scalar.I32), Param("n_ao", Scalar.I32),
        Param("diag", Scalar.I32), Param("tau", Scalar.F64),
        f64("bra_prim", nb, kab, Lit(len(REC))),
        *((f64("AB", nb, Lit(3)),) if lb else ()),
        f64("G_ab", nb), i32("bra_sh", nb, Lit(2)), i32("bra_ao", nb, Lit(2)),
        f64("ket_prim", nk, kcd, Lit(len(REC))),
        *((f64("CD", nk, Lit(3)),) if ld else ()),
        f64("G_cd", nk), i32("ket_sh", nk, Lit(2)), i32("ket_ao", nk, Lit(2)),
        f64("Dsh", nsh, nsh), f64("D", nao, nao),
        f64(FOCK, nao, nao, intent=Intent.OUT),
        *((f64(OUT, nb, nk, Lit(ctx.hdag.n_eri), intent=Intent.OUT),)
          if ctx.eris else ()),
    )
    assert tuple(p.name for p in out) == params_of(ctx.cls, eris=ctx.eris)
    return out


def finish(ctx: FContext, kernel: Kernel) -> Kernel:
    """The built kernel, unchanged."""
    return kernel


@dataclass(frozen=True)
class FHooks:
    """Every point an F lever may intervene, each defaulting to the frozen
    path (module docstring). Compose with `dataclasses.replace`."""

    shared_decl: Callable[[FContext], tuple[Stmt, ...]] = shared_decl
    per_bra: Callable[[FContext], tuple[Stmt, ...]] = per_bra
    bra_setup: Callable[[FContext], tuple[Stmt, ...]] = bra_setup
    stage: Callable[[FContext], tuple[Stmt, ...]] = stage
    ket_loop: Callable[[FContext], tuple[Stmt, ...]] = ket_loop
    per_ket: Callable[[FContext], tuple[Stmt, ...]] = per_ket
    screen: Callable[[FContext], tuple[Stmt, ...]] = screen
    work: Callable[[FContext], tuple[Stmt, ...]] = work
    boundary_init: Callable[[FContext], tuple[Stmt, ...]] = boundary_init
    nest: Callable[[FContext], tuple[Stmt, ...]] = nest
    ket_prim: Callable[[FContext], tuple[Stmt, ...]] = ket_prim
    quartet_scalars: Callable[[FContext], tuple[Stmt, ...]] = quartet_scalars
    vrr: Callable[[FContext], tuple[Stmt, ...]] = vrr
    accumulate: Callable[[FContext], tuple[Stmt, ...]] = accumulate
    boundary_read: Callable[[FContext, Cart, Cart], Expr] = boundary_read
    hrr: Callable[[FContext], tuple[Stmt, ...]] = hrr
    contracted: Callable[[FContext], tuple[Stmt, ...]] = contracted
    eri: Callable[[FContext, object], Expr] = eri
    tap: Callable[[FContext], tuple[Stmt, ...]] = tap
    digest: Callable[[FContext], tuple[Stmt, ...]] = digest
    digest_setup: Callable[[FContext], tuple[Stmt, ...]] = digest_setup
    zero_buffers: Callable[[FContext], tuple[Stmt, ...]] = zero_buffers
    predigest: Callable[[FContext], tuple[Stmt, ...]] = predigest
    predigest_eri: Callable[[FContext, object, tuple[int, ...]],
                            tuple[Stmt, ...]] = predigest_eri
    flush_ket: Callable[[FContext], tuple[Stmt, ...]] = flush_ket
    flush_bra: Callable[[FContext], tuple[Stmt, ...]] = flush_bra
    field: Callable[[FContext, str, tuple[Expr, ...]], Expr] = field_read
    real: Callable[[FContext, str], Scalar] = real
    cast: Callable[[FContext, Expr, str], Expr] = cast
    params: Callable[[FContext], tuple[Param, ...]] = params
    finish: Callable[[FContext, Kernel], Kernel] = finish

    def is_default(self, hook: str) -> bool:
        """True when `hook` is still the frozen path's."""
        return getattr(self, hook) is getattr(DEFAULT_HOOKS, hook)


DEFAULT_HOOKS = FHooks()
