"""The fused Fock kernel of `[Fock]` §3.2, `Target.F`: the contraction nest,
the boundary, HRR after the loop (#282).

    build_f_kernel((la, lb, lc, ld) | spec) -> Kernel      n_batches = 1
    f_class_spec(la, lb, lc, ld)            -> Spec        the class's key
    pack(pairs)                             -> Packed      one side's staged record
    PrimRec(P, X, zeta, U) / PairRec(sep, G, prims)        what the packer takes
    launch_shape(n_bra, threads)            -> (n_bra, (threads, 1))
    kernel_counts(cls)                      -> statement counts, from the class

Engine #2's kernel, beside `kernel_k.py` (the MMD `K`) and `kernel_jab.py`.
It computes every contracted ERI `(ab|cd)` of one launch -- a bra batch and
a ket batch of `launch_f`'s pair stream (#285) -- and STOPS AT THE ERIs: it
writes them to `g[bra, ket, i]` in `hgp.hrr.targets` order. The six-block
digestion into `F` is #283's; the recurrence half is pinned bitwise against
the host here while the dangerous half is still unwritten.

THE NEST (milestone 12, pins 5 and 7; `[Fock]` §3.2's "Block Execution"):

    bra = blockIdx.x                         one block per bra shell pair
    ab_sh[MAX_PRIM_PAIRS, REC] <- |ab)       the bra's primitive pairs, SHARED
    for k = threadIdx.x; k < n_ket; k += blockDim.x:   one thread per
                                             contracted quartet, consecutive
                                             threads on consecutive |cd)
        bound = G_ab G_cd maxD
        if bound < tau: skip                 the six-block screen's SHAPE
        else:
            c_(e,f) = 0                      |boundary| accumulators (961 at
                                             (dd|dd)), zeroed per ket
            for kcd < K_CD:                  the ket's primitive pairs (global)
                for kab < K_AB:              the bra's (shared)
                    quartet geometry, F_m(T) -> the VRR (eqs 25-26)
                    c_(e,f) += [e0|f0]^(0)   eq 27, every boundary value
            the HRR (eq 28) on the c_(e,f)   AFTER the loops: contracted only
            g[bra, k, :] = (ab|cd)

THE CONTRACTION LOOP IS INSIDE, AND THAT IS FORCED (pin 5): eq 28 holds only
for contracted values (`hgp/hrr.py`, "ITS INPUTS ARE CONTRACTED VALUES"), so
the `K_AB K_CD` sum completes in registers before the HRR reads it. The
accumulators are the BOUNDED boundary set, `hgp.vrr.target_set` == the HRR's
leaves: 961 at `(dd|dd)`, not the 1,225 #282's text prints (35^2, the same
set with the lower bound `la <= |e|`, `lc <= |f|` dropped; #273's and #274's
comments, and `hgp/hrr.py`'s docstring, which shows that #274's 6,254 HRR
statements fit 961 and only 961).

THE SCREEN IS A SKIP, NEVER A BREAK (pin 8; `fock/skip-not-break`, pending,
scored in #291). `If(bound < tau)` holds no statement but a `Note`; the
quartet's work sits in its `orelse`; the kernel holds no `Break` anywhere.
The batch sort (`launch_f`: `G * dmax`) "does not rigorously prevent warp
divergence", because the six-block `D_max` is not monotone along any one
sort, so a screened quartet cannot speak for the kets after it -- the
inverse of `Target.J`'s and `Target.K`'s sorted early break. THE BOUND
ITSELF IS A PLACEHOLDER: `G_ab[bra] * G_cd[k] * maxD`, `maxD` one host
scalar (K's form). The six-block `D_max` -- the largest `|D|` over the six
density blocks a quartet digests into -- and its weights are #283's; it
replaces the one `Decl bound`, and the control flow around it is final.

THE STAGED RECORD (`pack`, `REC`). One row per primitive pair, `REC`
fields: `P` (3), `X` (3) = `P - A` on the bra (`Q - C` on the ket), the
exponent sum `zeta` (`p`, or `q`), and `U = c_a c_b (pi/p)^{3/2} K_ab`, the
pair's factor WITH its two contraction coefficients (normalisation already
in them, `ao/shell.py`'s convention): the seed's `pair = U_P U_Q`
(`hgp/recurrence.py`, "THE FUNDAMENTAL") is one multiply here, and nothing
in the kernel forms a contraction coefficient or a `(pi/p)^{3/2}`. Per
contracted pair: `sep` = `A - B` (`C - D`), read by the HRR only, and `G`,
read by the screen only. The rows of one pair are `launch_f.FPair.prims`'
order (descending bound, after pruning); every pair of a batch has the same
degree `K_AB` (the bin key, #285), so a side is `n_pair x K x REC` with no
padding and the degree is ONE runtime scalar per side (`k_ab`, `k_cd`).

THE SHARED STAGING CAP: `MAX_PRIM_PAIRS = 128`. `ab_sh` is a native shared
array of literal extent `MAX_PRIM_PAIRS x REC` (8 KiB), so its size is a
compile-time constant and the bra degree a runtime one; `pack` REFUSES a
side whose degree exceeds the cap -- on either side, because every batch is
the bra of its own diagonal launch (`launch_f`'s `I == J`). 128 holds an
uncontracted pair of 10-primitive shells (cc-pVTZ's heavy-atom `s`, 100).
The predecessor shipped this cap and had to LIFT IT LATER, when a basis
outgrew it: a larger basis is a refusal here, never an overflow.

THE STAGING IS PER THREAD, BECAUSE THE IR HAS NO BARRIER (ir.py: "The IR has
no barrier"). A cooperative load -- thread `t` writes rows `t, t + nt, ...`
-- needs a `__syncthreads()` before any thread reads a row it did not
write, and nothing in `ir.py` spells one; the interpreter would refuse the
read (a shared element read before it was written). So every thread copies
the WHOLE record: each shared element is written by every thread with the
same value, and each thread reads only after its own write, which is
correct with no barrier on both vendors and under the interpreter's
sequential consistency. It is a `K_AB REC` redundant copy per thread, not
the bandwidth saving §3.2 stages for; the cooperative load is a follow-up
that needs a barrier in `ir.py` (named for #287, which emits this kernel).

THE QUARTET SCALARS, per primitive quartet (`p`, `q` the two exponent sums;
`hgp/recurrence.py`, "WHAT THE SCALARS ARE"):

    PQ = P - Q    ppq = p + q    rho = p q / ppq    theta = sqrt(rho)
    WP = 0 - (q/ppq) PQ    WQ = (p/ppq) PQ          (W - P, W - Q: the
                                                     derived sign, #279)
    inv2p = 0.5/p   inv2q = 0.5/q   inv2pq = 0.5/ppq
    inv2p_alpha_over_p = inv2p (rho/p)   inv2q_alpha_over_q = inv2q (rho/q)
    pair = U_P U_Q,  then T and S by `hgp.recurrence.fundamental_prelude`

Only the scalars the class's VRR reads are declared (an unused local is a
warning in generated text, #92), and only the separations its HRR reads;
the `AB` and `CD` params exist only when the bra / ket HRR does, since an
unread array parameter would be one too. `(ss|ss)` reads none of the eight.

FRAGMENT: NOT USED, and why. `ir.Fragment` pays off where one block is
spliced at two sites (`kernel_k` inlines the `E` block twice). Here the VRR
body occurs once -- the innermost loop -- and the HRR once, after it; the
fused kernel has no second site (a split HGP `J`/`K`, pin 1's follow-up,
would be the first). Reuse is already at BUILD time: the statements come
from `hgp.recurrence.vrr_body` / `hrr_body`, as `kernel_k` takes J's
`recurrence_body`, so the kernel cannot drift from the standalone kernels
#279 pins. A Fragment would add only a per-class bind list (the factors a
class reads differ) and a hygiene check over ~2,300 + 6,254 names, and buy
nothing it does not already have.

THE STATEMENT COUNTS (`kernel_counts`) are read off the built kernel by the
test and cross-checked against `hgp.vrr` / `hgp.hrr`'s pinned node counts:
one `Decl` per VRR node (2,330 at `(dd|dd)`), one accumulator `Decl` and one
`+=` per boundary value (961), one `Decl` per HRR node counting the boundary
reads (`HrrDag.n_statements`: 6,254 = 961 reads + 5,293 two-term updates,
#274's figure), one write per ERI (1,296). The plan is re-read, never
re-derived.

So, interpreted, `g` is `==` the host path, bit for bit: per primitive
quartet the scalars above in this order, `hgp.recurrence.build_vrr_kernel`'s
boundary, the left fold `c = c + v` over `kcd` outer and `kab` inner from
`+0.0`, then `build_hrr_kernel` on the sums (test/test_kernel_f.py).

Nothing here imports `plan/` (pin 11: it keys on a `[JEngine]` row an HGP
class does not have), `ao/` (the witness), `reference/`, the interpreter, a
backend or a device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import (
    PAIR,
    PQ_PARAMS,
    THETA,
    boundary_order,
    fundamental_prelude,
    hgp_class,
    hrr_body,
    hrr_name,
    vrr_body,
    vrr_name,
)
from felizjueves.hgp.vrr import AmClass, Cart
from felizjueves.ir import (
    Assign,
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
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "AB_SH",
    "Counts",
    "MAX_PRIM_PAIRS",
    "OUT",
    "REC",
    "SKIP",
    "Packed",
    "PairRec",
    "PrimRec",
    "build_f_kernel",
    "counted",
    "f_class_spec",
    "kernel_counts",
    "launch_shape",
    "pack",
    "params_of",
]

MAX_PRIM_PAIRS = 128  # the shared staging cap (module docstring)
REC = ("P_x", "P_y", "P_z", "X_x", "X_y", "X_z", "zeta", "U")
AB_SH = "ab_sh"  # the staged |ab) record, Space.SHARED
OUT = "g"
KET = "k"  # the strided ket loop: one thread per contracted quartet
KCD, KAB = "kcd", "kab"  # the primitive loops, ket outer, bra inner
STAGE = "i"

_P, _X, _ZETA, _U = 0, 3, 6, 7  # field offsets in REC

SKIP = (
    "screen: skip this quartet, never break ([Fock] §3.2; fock/skip-not-"
    "break). The six-block D_max is not monotone along the batch sort, so a "
    "screened quartet says nothing of the kets after it")

BOUND_PLACEHOLDER = (
    "PLACEHOLDER bound G_ab G_cd maxD: the six-block D_max and its weights "
    "are #283's; they replace this one Decl, and the skip around it is final")

STAGING = (
    "stage |ab) in shared memory: every thread copies the whole record (the "
    "IR has no barrier, so a cooperative load could not be read safely); "
    f"the degree k_ab is at most MAX_PRIM_PAIRS = {MAX_PRIM_PAIRS}, which "
    "kernel_f.pack enforces")


def f_class_spec(la: int, lb: int, lc: int, ld: int) -> Spec:
    """The F class `(la lb | lc ld)` -- AO order, one batch (spec.py, #272)."""
    return Spec(Operator.COULOMB, 0, (la, lb, lc, ld), Target.F)


def _spec_of(cls: AmClass | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = f_class_spec(*cls)
    else:
        raise TypeError("build_f_kernel takes (la, lb, lc, ld) or a Spec, "
                        f"not {cls!r}")
    hgp_class(spec)  # refuses anything but Target.F
    return spec  # n_batches == 1 is Spec's own rule for F (#272)


def launch_shape(n_bra: int, threads: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (threads, 1))`: one block per bra shell pair, the kets
    strided over `threadIdx.x` (pin 7)."""
    for name, v in (("n_bra", n_bra), ("threads", threads)):
        if type(v) is not int or v < 1:
            raise ValueError(f"{name} = {v!r} is not a positive int")
    return n_bra, (threads, 1)


# --- the packer --------------------------------------------------------------

Vec = tuple[float, float, float]


@dataclass(frozen=True)
class PrimRec:
    """One primitive pair's staged row: `P`, `X = P - A` (bra) or `Q - C`
    (ket), the exponent sum `zeta`, and `U = c_a c_b (pi/zeta)^{3/2} K_ab`."""

    P: Vec
    X: Vec
    zeta: float
    U: float

    def row(self) -> tuple[float, ...]:
        return (*map(float, self.P), *map(float, self.X), float(self.zeta),
                float(self.U))


@dataclass(frozen=True)
class PairRec:
    """One contracted shell pair: `sep = A - B` (or `C - D`), its bound `G`,
    and its kept primitive pairs in `launch_f.FPair.prims` order."""

    sep: Vec
    G: float
    prims: tuple[PrimRec, ...]


@dataclass(frozen=True)
class Packed:
    """One side of a launch, flat and row-major: `prim` is `n x degree x REC`,
    `sep` is `n x 3`, `G` is `n`."""

    n: int
    degree: int
    prim: tuple[float, ...]
    sep: tuple[float, ...]
    G: tuple[float, ...]


def pack(pairs: Sequence[PairRec]) -> Packed:
    """Stage one side: refuses an empty side, a mixed degree (a batch is
    degree-uniform, #285) and a degree above `MAX_PRIM_PAIRS`."""
    pairs = tuple(pairs)
    if not pairs:
        raise ValueError("an empty side has no pairs to stage")
    degrees = {len(p.prims) for p in pairs}
    if len(degrees) != 1:
        raise ValueError(f"degrees {sorted(degrees)}: a batch's pairs share "
                         "one contraction degree K_AB (launch_f's bin key)")
    (degree,) = degrees
    if degree < 1:
        raise ValueError("a pair with no primitive pairs is deleted, not "
                         "staged")
    if degree > MAX_PRIM_PAIRS:
        raise ValueError(
            f"degree {degree} exceeds MAX_PRIM_PAIRS = {MAX_PRIM_PAIRS}, the "
            "shared staging cap (kernel_f's docstring): the predecessor had "
            "to lift exactly this cap later; lift it here, do not overflow it")
    prim = tuple(x for p in pairs for r in p.prims for x in r.row())
    sep = tuple(float(x) for p in pairs for x in p.sep)
    return Packed(len(pairs), degree, prim, sep,
                  tuple(float(p.G) for p in pairs))


# --- the kernel --------------------------------------------------------------


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _acc_name(e: Cart, f: Cart) -> str:
    """The contracted accumulator of boundary `(e0|f0)`: `C{e}__{f}`."""
    return "C" + "__".join("_".join(map(str, v)) for v in (e, f))


def _factors(stmts: Sequence[Stmt]) -> set[str]:
    """Every non-unit factor name a `Weighted` in `stmts` reads."""
    out: set[str] = set()
    for s in stmts:
        if isinstance(s, Decl) and isinstance(s.init, Weighted):
            out |= {t.g.name_in_kernel for t in s.init.terms
                    if t.g.name_in_kernel is not None}
    return out


def params_of(cls: AmClass) -> tuple[str, ...]:
    """The kernel's parameter names, in order. `AB` only when `lb > 0` (the
    bra HRR reads it), `CD` only when `ld > 0`."""
    _, lb, _, ld = cls
    return ("n_bra", "n_ket", "k_ab", "k_cd", "tau", "maxD",
            "bra_prim", *(("AB",) if lb else ()), "G_ab",
            "ket_prim", *(("CD",) if ld else ()), "G_cd", OUT)


def build_f_kernel(cls: AmClass | Spec, *, name: str | None = None) -> Kernel:
    """The fused Fock kernel of one F class, up to the ERIs (module
    docstring). `cls` is `(la, lb, lc, ld)` in AO order or a `Target.F`
    `Spec`."""
    spec = _spec_of(cls)
    cls = hgp_class(spec)
    la, lb, lc, ld = cls
    vdag, hdag = vrr.build(cls), hrr.build(cls)
    bnd = boundary_order(cls)
    if set(bnd) != hdag.boundary:  # #274 holds this; refuse, never misread
        raise ValueError(f"{cls}: HRR boundary is not the VRR target set")
    if name is None:
        name = "f_" + "".join(map(str, cls))

    prelude, fundamental = fundamental_prelude()
    vbody = vrr_body(vdag, fundamental)
    hbody = hrr_body(hdag, lambda e, f: Var(_acc_name(e, f)))
    vneed, hneed = _factors(vbody), _factors(hbody)

    bra, k = Var("bra"), Var(KET)

    def sh(f: int) -> Index:
        return Index(AB_SH, (Var(KAB), Lit(f)))

    def ket(f: int) -> Index:
        return Index("ket_prim", (k, Var(KCD), Lit(f)))

    # --- one primitive quartet: geometry, scalars, F_m(T), the VRR --------
    def scalar(n: str, e: Expr) -> Decl:
        return Decl(n, Scalar.F64, init=e)

    wp = [Factor.wp(d).value for d in range(3)]
    wq = [Factor.wq(d).value for d in range(3)]
    quartet: list[Stmt] = [
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
        quartet.append(scalar("q_ppq", _c(Fn.DIV, Var("q"), Var("ppq"))))
    if vneed & set(wq):
        quartet.append(scalar("p_ppq", _c(Fn.DIV, Var("p"), Var("ppq"))))
    for d in range(3):
        if wp[d] in vneed:
            quartet.append(scalar(wp[d], _c(Fn.SUB, Lit(0.0), _c(
                Fn.MUL, Var("q_ppq"), Var(PQ_PARAMS[d])))))
        if wq[d] in vneed:
            quartet.append(scalar(wq[d], _c(
                Fn.MUL, Var("p_ppq"), Var(PQ_PARAMS[d]))))
    derived = ((Factor.INV2P_ALPHA_OVER_P.value, Factor.INV2P.value, "p"),
               (Factor.INV2Q_ALPHA_OVER_Q.value, Factor.INV2Q.value, "q"))
    # a derived product needs its 1/(2x) even where no term reads it alone
    vneed = vneed | {inv2 for n, inv2, _ in derived if n in vneed}
    for n, x in ((Factor.INV2P.value, "p"), (Factor.INV2Q.value, "q"),
                 (Factor.INV2PQ.value, "ppq")):
        if n in vneed:
            quartet.append(scalar(n, _c(Fn.DIV, Lit(0.5), Var(x))))
    for n, inv2, x in derived:
        if n in vneed:
            quartet.append(scalar(n, _c(Fn.MUL, Var(inv2), _c(
                Fn.DIV, Var("rho"), Var(x)))))
    quartet += [
        scalar(PAIR, _c(Fn.MUL, Var("Ub"), Var("Uq"))),
        *prelude,
        *vbody,
        Note("eq 27: c += [e0|f0]^(0), every boundary value, contracted in "
             "registers before the HRR (pin 5)"),
        *(Assign(Var(_acc_name(e, f)), _c(
            Fn.ADD, Var(_acc_name(e, f)), Var(vrr_name((e, f, 0)))))
          for e, f in bnd),
    ]

    # --- one ket primitive pair: read it, then the bra primitives ---------
    per_kcd: list[Stmt] = [
        Note("the ket primitive pair, from global memory (this thread's |cd))"),
        *(scalar(f"Qk{d}", ket(_P + d)) for d in range(3)),
        *(scalar(Factor.qc(d).value, ket(_X + d)) for d in range(3)
          if Factor.qc(d).value in vneed),
        scalar("q", ket(_ZETA)),
        scalar("Uq", ket(_U)),
        For(KAB, Lit(0), Var("k_ab"), Lit(1), tuple(quartet)),
    ]

    # --- one contracted quartet: accumulate, HRR, write -------------------
    work: list[Stmt] = [
        *(scalar(Factor.cd(d).value, Index("CD", (k, Lit(d))))
          for d in range(3) if Factor.cd(d).value in hneed),
        Note(f"the {len(bnd)} contracted boundary accumulators "
             "(hgp.vrr.target_set, the HRR's leaves), zeroed per ket"),
        *(scalar(_acc_name(e, f), Lit(0.0)) for e, f in bnd),
        Note("the K_CD x K_AB primitive nest: ket outer (global), bra inner "
             "(shared)"),
        For(KCD, Lit(0), Var("k_cd"), Lit(1), tuple(per_kcd)),
        *hbody,
        Note("the ERIs, hgp.hrr.targets order; the six-block digestion "
             "replaces this write (#283)"),
        *(Assign(Index(OUT, (bra, k, Lit(i))), Var(hrr_name(t)))
          for i, t in enumerate(hdag.targets)),
    ]
    per_ket: list[Stmt] = [
        Note(BOUND_PLACEHOLDER),
        scalar("bound", _c(Fn.MUL, Var("Gb"), _c(
            Fn.MUL, Index("G_cd", (k,)), Var("maxD")))),
        If(_c(Fn.LT, Var("bound"), Var("tau")), (Note(SKIP),), tuple(work)),
    ]

    stage = For(STAGE, Lit(0), Var("k_ab"), Lit(1), tuple(
        Assign(Index(AB_SH, (Var(STAGE), Lit(f))),
               Index("bra_prim", (bra, Var(STAGE), Lit(f))))
        for f in range(len(REC))))
    per_bra: list[Stmt] = [
        *(scalar(Factor.ab(d).value, Index("AB", (bra, Lit(d))))
          for d in range(3) if Factor.ab(d).value in hneed),
        scalar("Gb", Index("G_ab", (bra,))),
        Note(STAGING),
        stage,
        Note("one thread per contracted quartet: consecutive threads take "
             "consecutive |cd) ([Fock] §3.2, pin 7)"),
        For(KET, Call(Fn.THREAD_X), Var("n_ket"), Call(Fn.BLOCK_DIM_X),
            tuple(per_ket)),
    ]
    body: tuple[Stmt, ...] = (
        Note(f"(ab|cd) by the HGP recurrence, F class ({la}{lb}|{lc}{ld}) in "
             "AO order: [Fock] §3.2's fused kernel up to the ERIs (#282); "
             "VRR eqs 25-26 inside the K_CD x K_AB nest, HRR eq 28 after it"),
        Note("launch: grid = n_bra, blockDim = (threads, 1); one block per "
             "bra shell pair"),
        Decl("bra", Scalar.I32, init=Call(Fn.BLOCK_X)),
        Decl(AB_SH, Scalar.F64, (Lit(MAX_PRIM_PAIRS), Lit(len(REC))),
             Space.SHARED),
        If(_c(Fn.LT, bra, Var("n_bra")), tuple(per_bra)),
    )

    nb, nk, kab, kcd = (Var(n) for n in ("n_bra", "n_ket", "k_ab", "k_cd"))

    def f64(n: str, *shape: Expr, intent: Intent = Intent.IN) -> Param:
        return Param(n, Scalar.F64, array=True, intent=intent, shape=shape)

    params = (
        Param("n_bra", Scalar.I32), Param("n_ket", Scalar.I32),
        Param("k_ab", Scalar.I32), Param("k_cd", Scalar.I32),
        Param("tau", Scalar.F64), Param("maxD", Scalar.F64),
        f64("bra_prim", nb, kab, Lit(len(REC))),
        *((f64("AB", nb, Lit(3)),) if lb else ()),
        f64("G_ab", nb),
        f64("ket_prim", nk, kcd, Lit(len(REC))),
        *((f64("CD", nk, Lit(3)),) if ld else ()),
        f64("G_cd", nk),
        f64(OUT, nb, nk, Lit(hdag.n_eri), intent=Intent.OUT),
    )
    assert tuple(p.name for p in params) == params_of(cls)
    return Kernel(name, params, body)


# --- the counts --------------------------------------------------------------


@dataclass(frozen=True)
class Counts:
    """Statement counts of one F kernel (module docstring, "THE STATEMENT
    COUNTS")."""

    vrr: int  # one Decl per VRR node, in the innermost loop
    boundary: int  # accumulators: one Decl and one += each
    hrr: int  # one Decl per HRR node, boundary reads included
    hrr_interior: int  # the two-term eq 28 updates among them
    eri: int  # one write each


def kernel_counts(cls: AmClass) -> Counts:
    """The counts the class's DAGs predict: `hgp.vrr` / `hgp.hrr`'s own."""
    v, h = vrr.build(cls), hrr.build(cls)
    return Counts(v.n_nodes, h.n_boundary, h.n_statements, h.n_interior,
                  h.n_eri)


def counted(kernel: Kernel) -> Counts:
    """The same counts, read off a built kernel by statement kind."""
    decls = [s for s in walk(kernel.body) if isinstance(s, Decl)]
    vrr_n = sum(1 for s in decls if s.name.startswith("V"))
    acc = sum(1 for s in decls if s.name.startswith("C")
              and s.name not in ("CD_x", "CD_y", "CD_z"))
    hrr_d = [s for s in decls if s.name.startswith("H")]
    interior = sum(1 for s in hrr_d if isinstance(s.init, Weighted))
    eri = sum(1 for s in walk(kernel.body) if isinstance(s, Assign)
              and isinstance(s.dst, Index) and s.dst.array == OUT)
    return Counts(vrr_n, acc, len(hrr_d), interior, eri)
