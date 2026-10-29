"""The HGP Coulomb kernel, split (`Target.J` on `Engine.HGP`, #404).

    build_hj_kernel(spec | (la, lb, lc, ld), *, eris=, density=) -> Kernel
    hj_class_spec(la, lb, lc, ld, *, ordered=False) -> Spec   the class's key
    hj_name(spec)                  -> "hj_..." (canonical) / "hj_o_..." (ordered)
    built_class(spec)              -> the class whose recurrence is built
    orientations(spec)             -> the `hgp.orbits.GROUP` indices it deposits
    params_of(spec, *, eris=False) -> the parameter names, in order
    launch_shape(n_bra, threads)   -> (n_bra, (threads, 1))
    kernel_counts(spec) / counted(kernel) -> statement counts, DAG vs built

The Coulomb matrix on the Head-Gordon-Pople recurrence of `[Fock]` §3.1,
SPLIT: every ORDERED shell quartet `(a, b, c, d)` is visited once (#403's
stream: no symmetry pruning) and deposits

    J[a_i][b_j] += D[c_k][d_l] (a_i b_j | c_k d_l)

by atomic add -- that and nothing else: no 8-fold weights, no degeneracy
factor, no canonical half, no host-side `close`. It is `eri/j_coulomb/j.py`'s ordered
sweep (`coulomb_matrix_reference`: every ordered bra and ket pair, weight
1, `J[a][b]` written only from its own block), one class at a time, on
another engine. The fused `F` (`kernel_f.py`) is the other diagonal cell of
`[Fock]`'s 2x2; this is the (HGP, split) cell.

THE NEST (`[Fock]` §3.2's mapping, as `kernel_f`):

    bra = blockIdx.x                          one block per bra shell pair
    ab_sh[MAX_PRIM_PAIRS, REC] <- |ab)        staged, every thread copies it
    for k = threadIdx.x; k < n_ket; k += blockDim.x:   one thread per
                                              contracted quartet
        bound = G_ab G_cd Dsh[density pair]   the ONE density block J reads
        if bound < tau: skip                  a skip, never a break
        else:
            c_(e,f) = 0                       the boundary accumulators
            for kab < K_AB:                   the bra's primitive pairs (shared)
                for kcd < K_CD:               the ket's (global)
                    quartet scalars, F_m(T), the VRR (eqs 25-26)
                    c_(e,f) += [e0|f0]^(0)    eq 27
            the HRR (eq 28) on the c_(e,f)    after the loops: contracted
            j_(i,j) = sum D_(k,l) x           the deposit, per orientation
            AtomicAdd j into J

THE CONTRACTION IS INSIDE THE KERNEL, and that is forced, as in `kernel_f`
(milestone 12, pin 5): eq 28 holds only on contracted values, so the `K^4`
sum completes in registers before the HRR reads it.

THE PRIMITIVE NEST IS BRA OUTER, KET INNER -- the opposite of `kernel_f`'s
-- and the quartet scalars are spelled in `eri/hgp_drive.py`'s operation
order (`W = (p P + q Q)/(p + q)` then `W - P`, `1/(2p)`, ...), so the tap
is `==` `hgp_drive.eri_block` bit for bit (#404's acceptance): the drive
sums its VRR outputs over `(i, j, k, l)` with the bra pair outer, and a
bra-outer nest over `ao.pairs`-ordered records (`i` outer) is the same
left fold from `+0.0`. The price is that the ket's primitive record is read
from global memory `K_AB` times per quartet instead of once; nothing here is
measured (#407 emits, nothing times), so the order is chosen for the pin.

THE STAGING IS PER THREAD, because the IR has no barrier (#333; kernel_f's
docstring, "THE STAGING IS PER THREAD"): every thread copies the whole
`|ab)` record into `ab_sh`. The record layout (`REC`), its packer
(`kernel_f.pack`, `PairRec`, `PrimRec`) and its cap (`MAX_PRIM_PAIRS`) are
`kernel_f`'s, reused as they are.

THE TWO CLASS MODES (#401's `ordered` lever, #402's orbits).

  canonical (literal)  ONE kernel per orbit of the 8-fold ERI symmetry,
      built on the orbit's REPRESENTATIVE (`hgp.orbits.representative`, the
      cheapest orientation of the HRR and VRR), named after it:
      `hj_{rep}`. An ordered quartet `q` of any member class is launched as
      the representative-ordered quartet `x`, `x[j] = q[g[j]]` with `g =
      to_representative(class).shells`, and the kernel deposits through the
      permutation: `q`'s position `i` is `x`'s position `g^-1[i]`, so `J`
      lands on the rep positions `(g^-1[0], g^-1[1])` with the density of
      `(g^-1[2], g^-1[3])`. The permutation is a runtime scalar, `perm`,
      the index of `g` in `hgp.orbits.GROUP` -- one per launch, since a
      launch holds one ordered class (#403's stream carries it). The kernel
      holds one deposit per orientation the orbit's members reach
      (`orientations`), selected by an exact membership chain of `LT`s (the
      IR has no `EQ`); a `perm` outside them deposits nothing.
  ordered (`ordered` lever)  one kernel per ordered class, its own
      recurrence, the identity deposit, no `perm`: `hj_o_{class}`. The
      spelling cannot collide with a canonical unit's.

Both names `_`-separate the four `l`s once any is above 9 (`hj_0_11_0_0`),
as `kernel_eri3.eri3_name` does, so no two classes share a name.

THE DEPOSIT READS `D` WHERE THE ORDERED QUARTET PUTS IT: `D[c][d]` at the
ordered quartet's own ket positions, in every orientation, so no index
assumes a symmetric `D` (a symmetric `D` is `eri/j_coulomb/j.py`'s contract, the
witness's). `J[a][b]` is written only from its own block; the sweep over
every ordered quartet makes `J` symmetric, and nothing here mirrors it.

THE SCREEN (#403's J key): `G_ab G_cd max|D_cd|`, ONE density block (the
`Dsh` element at the density pair's two shells), a SKIP (`If(bound < tau)`
holds a `Note`; the kernel holds no `Break`). At `tau = 0` nothing is
skipped.

`density=` names the ordered positions whose `D` block the deposit reads,
`(2, 3)` by default -- the Coulomb pairing. Any other pair is a MUTATION a
test builds to watch the deposit fail (`kernel_f`'s `weights=` precedent).

THE STATEMENT COUNTS (`kernel_counts`, `counted`): one `Decl` per VRR node,
one accumulator per boundary value, one `Decl` per HRR node (boundary reads
included, `HrrDag.n_statements`), every ERI read once per orientation's
deposit, and one `AtomicAdd` per `J` element per orientation -- read off the
built kernel and cross-checked against `hgp.vrr` / `hgp.hrr` by the test.

THE SEAM (`Nest`, `split_j_kernel`): the engine is the `K_CD` loop's body
after the ket record's reads -- the quartet scalars, the recurrence and the
`+=` of every boundary value `(e0|f0)` into its accumulator -- and the
record scalars it reads (`PA_*`, `QC_*`). `hgp_nest` is HGP's (eqs 25-27);
`eri/hjk/kernel_rj.py` passes the Rys rule and the 2D VRR (`Target.J` on
`Engine.RYS`). Everything else is this module's, so an `rj_` unit is a
drop-in for the `hj_` unit of the same class and mode.

Nothing here imports `plan/`, `ao/` (the witnesses: `eri/hgp_drive.py` for
the ERIs, `eri/j_coulomb/j.py` for `J`), `reference/`, the interpreter, a backend or
a device (test/test_layering.py, rule `hj_builder`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from felizjueves.eri.fock.kernel_f import MAX_PRIM_PAIRS, REC
from felizjueves.hgp import hrr, orbits, vrr
from felizjueves.hgp.recurrence import (
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
    walk,
)
from felizjueves.spec import Engine, Operator, Spec, Target, lever_provenance

__all__ = [
    "AB_SH",
    "COULOMB_DENSITY",
    "Counts",
    "J_OUT",
    "KAB",
    "KCD",
    "KET",
    "MAX_PRIM_PAIRS",
    "Nest",
    "NestOf",
    "ORDERED",
    "OUT",
    "PERM",
    "REC",
    "SKIP",
    "STAGE",
    "build_hj_kernel",
    "built_class",
    "hgp_nest",
    "counted",
    "hj_class_spec",
    "hj_name",
    "inverse",
    "kernel_counts",
    "launch_shape",
    "orientations",
    "params_of",
    "split_j_kernel",
]

ORDERED = "ordered"  # the class-mode lever (#401, levers.VOCABULARY)
AB_SH = "ab_sh"  # the staged |ab) record, Space.SHARED
OUT = "g"  # the ERI tap (eris=True), built-class targets order
J_OUT = "J"  # the Coulomb matrix, AtomicAdd-ed
PERM = "perm"  # the canonical kernel's orientation: an index into GROUP
KET = "k"  # the strided ket loop: one thread per contracted quartet
KAB, KCD = "kab", "kcd"  # the primitive loops, bra OUTER, ket inner
STAGE = "i"
COULOMB_DENSITY = (2, 3)  # J_ab reads D_cd: the ordered positions (c, d)

_P, _X, _ZETA, _U = 0, 3, 6, 7  # field offsets in REC (kernel_f's layout)
_SH = ("sa", "sb", "sc", "sd")  # shell index, per BUILT-class position
_AO = ("oa", "ob", "oc", "od")  # AO offset, per BUILT-class position

SKIP = ("screen: skip this quartet, never break; G_ab G_cd max|D_cd| bounds "
        "the one block J reads (#403's J key)")
STAGING = (
    "stage |ab) in shared memory: every thread copies the whole record (the "
    "IR has no barrier, #333); the degree k_ab is at most MAX_PRIM_PAIRS = "
    f"{MAX_PRIM_PAIRS}, which kernel_f.pack enforces")


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


# --- the key -----------------------------------------------------------------


def hj_class_spec(la: int, lb: int, lc: int, ld: int, *,
                  ordered: bool = False) -> Spec:
    """The HGP `J` class `(la lb | lc ld)` (bra `(a, b)`, as MMD `J`; #401),
    one batch, carrying the `ordered` lever when asked."""
    return Spec(Operator.COULOMB, 0, (la, lb, lc, ld), Target.J,
                (ORDERED,) if ordered else (), engine=Engine.HGP)


def _spec_of(cls: AmClass | Spec) -> Spec:
    if isinstance(cls, Spec):
        spec = cls
    elif isinstance(cls, tuple) and len(cls) == 4:
        spec = hj_class_spec(*cls)
    else:
        raise TypeError("build_hj_kernel takes (la, lb, lc, ld) or a Spec, "
                        f"not {cls!r}")
    if spec.target is not Target.J or spec.engine is not Engine.HGP:
        raise ValueError(f"{spec.key()}: the HGP J kernel builds Target.J on "
                         "Engine.HGP only")
    if len(spec.am) != 4:
        raise ValueError(f"{spec.key()}: a J class is a quartet")
    other = tuple(lv for lv in spec.levers if lv != ORDERED)
    if other:
        raise ValueError(f"{spec.key()}: lever {other[0]!r} is not read by the "
                         "HGP J kernel (only the class mode, 'ordered')")
    if spec.n_batches != 1:
        raise ValueError(f"{spec.key()}: the HGP J kernel has no Hermite "
                         "targets to batch; n_batches must be 1")
    return spec


def _ordered(spec: Spec) -> bool:
    return ORDERED in spec.levers


def built_class(cls: AmClass | Spec) -> AmClass:
    """The class whose VRR and HRR the kernel runs: the spec's own under
    `ordered`, its orbit's representative otherwise."""
    spec = _spec_of(cls)
    am: AmClass = tuple(spec.am)  # type: ignore[assignment]
    return am if _ordered(spec) else orbits.representative(am)


def orientations(cls: AmClass | Spec) -> tuple[int, ...]:
    """The `hgp.orbits.GROUP` indices the kernel deposits through, sorted:
    `(0,)` (the identity) under `ordered`; under the canonical mode, the
    permutation `to_representative` returns for each distinct member of
    the orbit -- the values #403's stream passes as `perm`."""
    spec = _spec_of(cls)
    if _ordered(spec):
        return (0,)
    return tuple(sorted({orbits.GROUP.index(orbits.to_representative(m).shells)
                         for m in orbits.orbit(tuple(spec.am))}))


def inverse(g: Sequence[int]) -> tuple[int, int, int, int]:
    """`g^-1`: an ordered quartet's position `i` sits on the built class's
    position `inverse(g)[i]`."""
    return tuple(tuple(g).index(i) for i in range(4))  # type: ignore[return-value]


def hj_name(cls: AmClass | Spec) -> str:
    """`hj_{built class}` canonical, `hj_o_{class}` ordered (module
    docstring), `_`-separated once any `l` is above 9."""
    spec = _spec_of(cls)
    am = built_class(spec)
    sep = "_" if max(am) > 9 else ""
    return ("hj_o_" if _ordered(spec) else "hj_") + sep.join(map(str, am))


def launch_shape(n_bra: int, threads: int) -> tuple[int, tuple[int, int]]:
    """`(n_bra, (threads, 1))`: one block per bra shell pair, the kets
    strided over `threadIdx.x`."""
    for name, v in (("n_bra", n_bra), ("threads", threads)):
        if type(v) is not int or v < 1:
            raise ValueError(f"{name} = {v!r} is not a positive int")
    return n_bra, (threads, 1)


def params_of(cls: AmClass | Spec, *, eris: bool = False) -> tuple[str, ...]:
    """The kernel's parameter names, in order. `perm` only in the canonical
    mode; `AB` only when the built class's `lb > 0` (the bra HRR reads it),
    `CD` only when its `ld > 0`; `g` only with the tap."""
    spec = _spec_of(cls)
    _, lb, _, ld = built_class(spec)
    return ("n_bra", "n_ket", "k_ab", "k_cd", "n_sh", "n_ao",
            *(() if _ordered(spec) else (PERM,)), "tau",
            "bra_prim", *(("AB",) if lb else ()), "G_ab", "bra_sh", "bra_ao",
            "ket_prim", *(("CD",) if ld else ()), "G_cd", "ket_sh", "ket_ao",
            "Dsh", "D", J_OUT, *((OUT,) if eris else ()))


# --- the pieces --------------------------------------------------------------


def acc_name(e: Cart, f: Cart) -> str:
    """The contracted accumulator of boundary `(e0|f0)` (kernel_f's name)."""
    return "C" + "__".join("_".join(map(str, v)) for v in (e, f))


def _factors(stmts: Sequence[Stmt]) -> set[str]:
    out: set[str] = set()
    for s in stmts:
        if isinstance(s, Decl) and isinstance(s.init, Weighted):
            out |= {t.g.name_in_kernel for t in s.init.terms
                    if t.g.name_in_kernel is not None}
    return out


def _comps(cls: AmClass) -> tuple[tuple[int, ...], ...]:
    """Per ERI target, its four component indices within their shells."""
    pos = [{c: i for i, c in enumerate(vrr.cartesians(x))} for x in cls]
    return tuple(tuple(pos[s][v] for s, v in enumerate(t))
                 for t in hrr.targets(cls))


def _F(name: str, e: Expr) -> Decl:
    return Decl(name, Scalar.F64, init=e)


def _I(name: str, e: Expr) -> Decl:
    return Decl(name, Scalar.I32, init=e)


def _record(arr: str, row: Expr, pos: int) -> Expr:
    """Shell index / AO offset of BUILT position `pos` (0, 1 bra; 2, 3 ket)."""
    return Index(arr, (row, Lit(pos % 2)))


def _ao(pos: int, i: int) -> Expr:
    return _c(Fn.ADD, Var(_AO[pos]), Lit(i))


def _quartet_scalars(vneed: frozenset[str]) -> list[Stmt]:
    """The scalars the VRR reads, in `eri/hgp_drive.primitive_scalars`'
    operation order (module docstring), from the bra (`Pb`, `p`, `Ub`) and
    ket (`Qk`, `q`, `Uq`) primitive records."""
    wp = [Factor.wp(d).value for d in range(3)]
    wq = [Factor.wq(d).value for d in range(3)]
    p, q = Var("p"), Var("q")
    out: list[Stmt] = [
        Note("quartet geometry in ao/hgp_drive's operation order: PQ = P - "
             "Q, W = (p P + q Q)/(p + q), W - P, and W - Q = p/(p + q) PQ "
             "(the derived sign, #279)"),
        *(_F(n, _c(Fn.SUB, Var(f"Pb{d}"), Var(f"Qk{d}")))
          for d, n in enumerate(PQ_PARAMS)),
        _F("ppq", _c(Fn.ADD, p, q)),
        _F("rho", _c(Fn.DIV, _c(Fn.MUL, p, q), Var("ppq"))),
        _F(THETA, _c(Fn.SQRT, Var("rho"))),
    ]
    for d in range(3):
        if wp[d] in vneed:
            out += [_F(f"W{d}", _c(Fn.DIV, _c(
                        Fn.ADD, _c(Fn.MUL, p, Var(f"Pb{d}")),
                        _c(Fn.MUL, q, Var(f"Qk{d}"))), Var("ppq"))),
                    _F(wp[d], _c(Fn.SUB, Var(f"W{d}"), Var(f"Pb{d}")))]
    if vneed & set(wq):
        out.append(_F("p_ppq", _c(Fn.DIV, p, Var("ppq"))))
    for d in range(3):
        if wq[d] in vneed:
            out.append(_F(wq[d], _c(Fn.MUL, Var("p_ppq"), Var(PQ_PARAMS[d]))))
    inv2 = {Factor.INV2P.value: p, Factor.INV2Q.value: q,
            Factor.INV2PQ.value: Var("ppq")}
    for n, x in inv2.items():
        if n in vneed:
            out.append(_F(n, _c(Fn.DIV, Lit(1.0), _c(Fn.MUL, Lit(2.0), x))))
    for n, i2, x in _DERIVED:
        if n in vneed:
            out.append(_F(n, _c(Fn.MUL, Var(i2), _c(Fn.DIV, Var("rho"),
                                                     Var(x)))))
    out.append(_F(PAIR, _c(Fn.MUL, Var("Ub"), Var("Uq"))))
    return out


_DERIVED = ((Factor.INV2P_ALPHA_OVER_P.value, Factor.INV2P.value, "p"),
            (Factor.INV2Q_ALPHA_OVER_Q.value, Factor.INV2Q.value, "q"))


def _membership(items: Sequence[tuple[int, tuple[Stmt, ...]]]
                ) -> tuple[Stmt, ...]:
    """`if perm == g: body` for each `(g, body)`, `g` ascending, spelled
    with `LT` alone (the IR has no `EQ`): exact, so a `perm` that is no
    listed `g` runs no body."""
    none = (Note("perm is not an orientation of this kernel's orbit"),)

    def chain(i: int, lo: int) -> tuple[Stmt, ...]:
        if i == len(items):
            return none
        g, body = items[i]
        hit = If(_c(Fn.LT, Var(PERM), Lit(g + 1)), body, chain(i + 1, g + 1))
        if g == lo:
            return (hit,)
        return (If(_c(Fn.LT, Var(PERM), Lit(g)), none, (hit,)),)

    return chain(0, 0)


@dataclass(frozen=True)
class _Orient:
    """One deposit: GROUP index `g`, the built positions `J` lands on and
    the built positions whose density it reads."""

    g: int
    jpos: tuple[int, int]
    dpos: tuple[int, int]


def _orients(spec: Spec, density: tuple[int, int]) -> tuple[_Orient, ...]:
    out = []
    for g in orientations(spec):
        inv = inverse(orbits.GROUP[g])
        out.append(_Orient(g, (inv[0], inv[1]),
                           (inv[density[0]], inv[density[1]])))
    return tuple(out)


def _deposit(o: _Orient, n: Sequence[int], comps, targets) -> tuple[Stmt, ...]:
    """`J[jpos] += D[dpos] x` for every ERI, through a thread-private
    buffer, then one `AtomicAdd` per `J` element."""
    (r0, r1), (t0, t1) = o.jpos, o.dpos

    def jbuf(i: int, j: int) -> str:
        return f"j{o.g}_{i}_{j}"

    def dval(k: int, l: int) -> str:  # noqa: E741
        return f"d{o.g}_{k}_{l}"

    out: list[Stmt] = [
        Note(f"orientation GROUP[{o.g}] = {orbits.GROUP[o.g]}: J lands on "
             f"built positions {o.jpos}, D read at {o.dpos}"),
        *(_F(dval(k, l), Index("D", (_ao(t0, k), _ao(t1, l))))
          for k in range(n[t0]) for l in range(n[t1])),  # noqa: E741
        *(_F(jbuf(i, j), Lit(0.0)) for i in range(n[r0]) for j in range(n[r1])),
    ]
    for t, idx in zip(targets, comps, strict=True):
        f = Var(jbuf(idx[r0], idx[r1]))
        out.append(Assign(f, _c(Fn.ADD, f, _c(
            Fn.MUL, Var(dval(idx[t0], idx[t1])), Var(hrr_name(t))))))
    out += [AtomicAdd(Index(J_OUT, (_ao(r0, i), _ao(r1, j))), Var(jbuf(i, j)))
            for i in range(n[r0]) for j in range(n[r1])]
    return tuple(out)


# --- the kernel --------------------------------------------------------------


@dataclass(frozen=True)
class Nest:
    """The primitive nest's ENGINE part (module docstring, "THE SEAM"):
    `need`, the record scalars (`PA_*` off the bra record, `QC_*` off the
    ket's) the body reads; `body`, the `K_CD` loop's statements after the
    ket record's reads, which must leave every boundary value `(e0|f0)` of
    the primitive quartet added into its accumulator `acc_name(e, f)`; and
    `note`, the kernel's opening `Note`, with `{cls}` and `{mode}` fields."""

    need: frozenset[str]
    body: tuple[Stmt, ...]
    note: str


NestOf = Callable[[AmClass, Sequence[tuple[Cart, Cart]]], Nest]


def hgp_nest(am: AmClass, bnd: Sequence[tuple[Cart, Cart]]) -> Nest:
    """The HGP engine's nest: the quartet scalars in `eri/hgp_drive.py`'s
    operation order, one Boys evaluation, the VRR (eqs 25-26), then eq 27's
    `+=` of every boundary value."""
    prelude, fundamental = fundamental_prelude(sum(am))
    vbody = tuple(vrr_body(vrr.build(am), fundamental))
    vf = _factors(vbody)
    vneed = frozenset(vf | {i2 for n, i2, _ in _DERIVED if n in vf})
    body = (
        *_quartet_scalars(vneed),
        *prelude,
        *vbody,
        Note("eq 27: c += [e0|f0]^(0), every boundary value, contracted in "
             "registers before the HRR (pin 5)"),
        *(Assign(Var(acc_name(e, f)), _c(Fn.ADD, Var(acc_name(e, f)),
                                         Var(vrr_name((e, f, 0)))))
          for e, f in bnd),
    )
    return Nest(vneed, body,
                "(ab|cd) by the HGP recurrence, J class ({cls}) [{mode}]: VRR "
                "eqs 25-26 inside the K_AB x K_CD nest, HRR eq 28 after it, "
                "then J_ab += D_cd (ab|cd), split (#404)")


def build_hj_kernel(cls: AmClass | Spec, *, name: str | None = None,
                    eris: bool = False,
                    density: tuple[int, int] = COULOMB_DENSITY) -> Kernel:
    """The HGP `J` kernel of one class (module docstring). `cls` is a
    `Target.J`, `Engine.HGP` `Spec` (or `(la, lb, lc, ld)`, canonical);
    `eris=True` adds the ERI tap `g[bra, ket, i]` in the built class's
    `hgp.hrr.targets` order; `density` is the ordered positions whose `D`
    block the deposit reads -- `(2, 3)`, or a test's mutation."""
    spec = _spec_of(cls)
    return split_j_kernel(spec, hgp_nest, name=name or hj_name(spec),
                          eris=eris, density=density)


def split_j_kernel(cls: AmClass | Spec, nest_of: NestOf, *, name: str,
                   eris: bool = False,
                   density: tuple[int, int] = COULOMB_DENSITY) -> Kernel:
    """The split `J` kernel of one HGP-keyed class, the primitive nest's
    engine part from `nest_of` (module docstring, "THE SEAM").
    `build_hj_kernel` is `split_j_kernel(spec, hgp_nest, ...)`; the Rys
    builder (`eri/hjk/kernel_rj.py`) passes its HGP twin's spec and its own
    nest. Everything else -- the staging, the screen, the HRR, the tap, the
    deposit, the parameters -- is this module's."""
    spec = _spec_of(cls)
    density = tuple(density)  # type: ignore[assignment]
    if (len(density) != 2 or density[0] == density[1]
            or any(type(x) is not int or not 0 <= x < 4 for x in density)):
        raise ValueError(f"density {density!r} is not two distinct positions")
    am = built_class(spec)
    ordered = _ordered(spec)
    la, lb, lc, ld = am
    hdag = hrr.build(am)
    bnd = boundary_order(am)
    if set(bnd) != hdag.boundary:  # #274 holds this; refuse, never misread
        raise ValueError(f"{am}: HRR boundary is not the VRR target set")
    nest = nest_of(am, bnd)
    vneed = nest.need
    hbody = tuple(hrr_body(hdag, lambda e, f: Var(acc_name(e, f))))
    hneed = frozenset(_factors(hbody))
    n = tuple(len(vrr.cartesians(x)) for x in am)
    orients = _orients(spec, density)
    shells = {p for o in orients for p in o.dpos}
    aos = {p for o in orients for p in (*o.jpos, *o.dpos)}
    bra, k = Var("bra"), Var(KET)

    def bra_prim(f: int) -> Expr:
        return Index(AB_SH, (Var(KAB), Lit(f)))

    def ket_prim(f: int) -> Expr:
        return Index("ket_prim", (k, Var(KCD), Lit(f)))

    # -- the screen: G_ab G_cd Dsh[density pair] ------------------------------
    def dsh(o: _Orient) -> Expr:
        return Index("Dsh", (Var(_SH[o.dpos[0]]), Var(_SH[o.dpos[1]])))

    reads = {dsh(o) for o in orients}
    if len(reads) == 1:
        dmax: tuple[Stmt, ...] = (_F("dmax", dsh(orients[0])),)
    else:
        dmax = (_F("dmax", Lit(0.0)),
                *_membership([(o.g, (Assign(Var("dmax"), dsh(o)),))
                              for o in orients]))

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
        tap = (Note("the ERI tap: every (ab|cd) of the built class, "
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
        Note("the deposit: J_ab += D_cd (ab|cd), one ordered quartet, no "
             "weight, no degeneracy, by atomic add"),
        *deposit,
    )
    per_ket = (
        *(_I(_SH[s], _record("ket_sh", k, s)) for s in (2, 3) if s in shells),
        Note("the one-block screen: G_ab G_cd max|D| over the density block "
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
             "consecutive |cd) ([Fock] §3.2)"),
        For(KET, Call(Fn.THREAD_X), Var("n_ket"), Call(Fn.BLOCK_DIM_X),
            per_ket),
    )
    mode = ("ordered: its own class, the identity deposit" if ordered else
            f"canonical: the orbit representative, {len(orients)} "
            "orientation(s) selected by perm")
    body = (
        Note(nest.note.format(cls=f"{la}{lb}|{lc}{ld}", mode=mode)),
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
    _, lb, _, ld = am
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
        *((f64("AB", nb, Lit(3)),) if lb else ()),
        f64("G_ab", nb), i32("bra_sh", nb, Lit(2)), i32("bra_ao", nb, Lit(2)),
        f64("ket_prim", nk, kcd, Lit(len(REC))),
        *((f64("CD", nk, Lit(3)),) if ld else ()),
        f64("G_cd", nk), i32("ket_sh", nk, Lit(2)), i32("ket_ao", nk, Lit(2)),
        f64("Dsh", nsh, nsh), f64("D", nao, nao),
        f64(J_OUT, nao, nao, intent=Intent.OUT),
        *((f64(OUT, nb, nk, Lit(n_eri), intent=Intent.OUT),) if eris else ()),
    )
    assert tuple(p.name for p in out) == params_of(spec, eris=eris)
    return out


# --- the counts --------------------------------------------------------------


@dataclass(frozen=True)
class Counts:
    """Statement counts of one HGP J kernel (module docstring)."""

    vrr: int  # one Decl per VRR node, in the innermost loop
    boundary: int  # accumulators: one Decl and one += each
    hrr: int  # one Decl per HRR node, boundary reads included
    hrr_interior: int  # the two-term eq 28 updates among them
    eri: int  # each read once per orientation's deposit
    atomic: int  # AtomicAdd statements: J's elements, per orientation


def kernel_counts(cls: AmClass | Spec) -> Counts:
    """The counts the built class's DAGs predict, and the deposits' size."""
    spec = _spec_of(cls)
    am = built_class(spec)
    v, h = vrr.build(am), hrr.build(am)
    n = [len(vrr.cartesians(x)) for x in am]
    atomic = sum(n[o.jpos[0]] * n[o.jpos[1]]
                 for o in _orients(spec, COULOMB_DENSITY))
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
                and s.dst.name.startswith("j")):
            x = s.value.args[1].args[1]  # j = j + d * x
            reads[x.name] = reads.get(x.name, 0) + 1
    if any(c != n_dep for c in reads.values()):
        raise ValueError("an ERI is not read once per deposit")
    atomic = sum(1 for s in stmts if isinstance(s, AtomicAdd))
    return Counts(vrr_n, acc, len(hrr_d), interior, len(reads), atomic)
