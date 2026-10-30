"""The one-electron kernels by the HGP route: `S`, `T`, `V`, the dipole and
the quadrupole.

    build_hgp_kernel(spec | (target, la, lb)) -> Kernel   one per class
    hgp_name(spec | (target, la, lb))         -> str      its name, unbuilt
    hgp_spec(target, la, lb)                  -> Spec     the class's key
    pair_arrays(target)                       -> names    its per-pair arrays
    launch_shape(n_pair, threads)             -> (grid, block)

`Target.S`, `T`, `V`, `DIPOLE` and `QUADRUPOLE` on `Engine.HGP` (`spec.py`,
"AN HGP ONE-ELECTRON CLASS"), Cartesian: `S`, `T` and `V` for s and p shells
(`L_MAX = 1`), the two moments through f (`MOMENT_L_MAX = 3`, `l_max`). The
MMD kernels (`kernel_s`, `kernel_t`, `kernel_v`, `kernel_mu`) are untouched;
this is the second engine's instance of the same five matrices, promoted
from the spike `experimental/hgp/one_electron/hgp1e.py`, whose route it
follows.

THE ROUTE: Head-Gordon & Pople's split on Obara & Saika's one-electron
recurrences. Per primitive pair, with `p = zeta`, `P`, `PA = P - A`:

    overlap  (e+1_d|0)   = PA_d (e|0) + e_d/(2p) (e-1_d|0)
             (0|0)       = c_a c_b (pi/p)^{3/2} K_AB
    nuclear  (e+1_d|0)^m = PA_d (e|0)^m + (C - P)_d (e|0)^{m+1}
                           + e_d/(2p) [(e-1_d|0)^m - (e-1_d|0)^{m+1}]
             (0|0)^m     = c_a c_b (2 pi/p) K_AB F_m(p |P - C|^2)

THEN the primitive pairs are SUMMED, and the HRR moves angular momentum from
`A` to `B` once per shell pair, on contracted values (`AB = A - B` carries no
exponent):

    (a|b+1_d) = (a+1_d|b) + AB_d (a|b)

Each operator is a combination of contracted overlap-type integrals:

    S     (a|b)                                                VRR to la+lb
    MU_d  q [(a|b+1_d) + (B_d - C_d) (a|b)]                    VRR to la+lb+1
    Q_de  q [(a|b+1_d+1_e) + BC_e (a|b+1_d) + BC_d (a|b+1_e)
             + BC_d BC_e (a|b)]                                VRR to la+lb+2
    T     (2 lb + 3) (a|b)_1 - 2 SUM_d (a|b+2_d)_2
          - 1/2 SUM_d b_d (b_d - 1) (a|b-2_d)_0                VRR to la+lb+2
    V     -SUM_C Z_C (a|b)^{(0)}_C                             VRR to la+lb

`T`'s subscript `n` is a contracted table weighted by `beta^n` (`beta` the
ket exponent): differentiating `B`'s Gaussian twice brings out powers of
`beta`, which is per primitive, so it is folded in before the contraction;
each table then runs its own HRR. (The spike's
`-1/2 SUM_d [... - 2 (2 b_d + 1) (a|b)_1 ...]` with its three `(a|b)_1`
terms collected: `SUM_d (2 b_d + 1) = 2 lb + 3`.) The moments move the
operator onto the ket: `(r - C)_d = (r - B)_d + BC_d`, `BC = B - C`, and
`(r - B)_d |b> = |b+1_d>`, applied once for `MU` and twice for `Q` (the
raw second moments, `kernel_mu.build_quad_kernel`'s six components xx, xy,
xz, yy, yz, zz in `cartesians(2)` order, not the traceless tensor). `V`
sums the nuclei into one table before the HRR, so its HRR also runs once
per shell pair.

THE RECURRENCES ARE `hgp/`'s, NOT REWRITTEN. The VRR is `hgp.vrr`'s eq 25
DAG at `f = 0` (`hgp.vrr.build((0, top, 0, 0), prune=False)`'s bra phase,
cut to the nodes the HRR boundary reaches), spelled by
`hgp.recurrence.vrr_body` unchanged. Two limits of eq 25, read off its
coefficients, give the two one-electron VRRs:

- NUCLEAR: the ket a unit point charge at `C`, so `q -> infinity`: `W -> C`
  (`WP = C - P`) and `rho / p -> 1` (`inv2p_alpha_over_p = inv2p`), with the
  leaf `(0|0)^m` above. The kernel binds those two scalars so; eq 25's own
  terms and signs do the rest.
- OVERLAP: the `m + 1` terms dropped (`hgp.vrr`'s `WP` and
  `inv2p*a_over_p` terms) and the DAG cut to `m = 0`: the ket a constant.

The HRR is `hgp.hrr.reduce` (eq 28's bra half, `c = d = 0`), closed over the
class's targets and spelled by `hgp.recurrence.hrr_body` (its `name=` hook
prefixes `T`'s three tables). `S` reads the targets `(a|b)`; `MU` the
targets `(a|b)` and `(a|b+1_d)` of one table, one DAG over their union, and
`Q` those and `(a|b+1_d+1_e)`;
`T` table 0 `(a|b-2_d)` (absent through p: `b_d <= 1`), table 1 `(a|b)`
and table 2 `(a|b+2_d)`.

DEAD CODE IS NEVER EMITTED. A declared but unread local is a `#177-D` on
nvcc and a `-Wunused-variable` on hipcc (#92's filter): the VRR is cut to
exactly the boundary its HRRs read, a table is accumulated only on the
boundary its own HRR reads, and a recurrence scalar (`PA_x`, `inv2p`,
`AB_x`, ...) is declared only when a term reads it (`(ss)` reads none).

THE THREAD MAPPING: ONE THREAD PER CONTRACTED SHELL PAIR, `pair = THREAD_X +
BLOCK_X * BLOCK_DIM_X`, guarded by `pair < n_pair` -- the Schwarz kernel's
mapping (`eri.kernel_schwarz`), because the HRR needs CONTRACTED values: the
thread runs its pair's record range `[rec0, rec1)` serially, the VRR per
record, the boundary accumulated in registers from `0.0`, then the HRR once.
NO ATOMICS: each element of the matrix belongs to one shell pair, so it is
written ONCE, by a plain store, and the result is bit-reproducible run to
run. The scatter is `kernel_jab`'s placement: an off-diagonal pair writes
its block and the mirror, a diagonal one (`off_a == off_b`) its lower
triangle, mirrored. Elements of pairs NOT launched are untouched.

NO PREFACTOR IS FORMED BEYOND THE SEED. `s_ab` is the host's `c_a c_b
(pi/zeta)^{3/2}` (`ao.pairs.dress`, the MMD kernels' `s_ab`), WITHOUT
`K_AB` (the MMD `E` seed carries it per axis); the HGP seed is the whole
`(0|0)`, so `K_AB = exp(zeta (PA . PB))` is formed per record -- `zeta
(P - A).(P - B) = -alpha beta / zeta |AB|^2` -- from the record arrays the
MMD kernels already take. `V`'s leaf is `s_ab K_AB (2/sqrt(pi)) sqrt(zeta)
F_m`, `(pi/zeta)^{3/2} (2/sqrt(pi)) sqrt(zeta) = 2 pi / zeta`; the attractive
sign enters as `acc - Z u`, `kernel_v`'s.

THE PARAMETER BLOCK, the MMD one-electron kernels' record arrays (flat, one
per axis, `kernel_s`'s names) plus a per-pair block:

    params  n_pair n_rec n_ao (I32)
            zeta_ab [p_x p_y p_z]  pa_x pa_y pa_z  pb_x pb_y pb_z  s_ab
            [eb_ket]                                    [n_rec] per record
            rec0 rec1 off_a off_b  [n_pair]  (I32)      per shell pair
            ab_x ab_y ab_z  [b_x b_y b_z]  [n_pair]     A - B  [and B]
            [n_nuc (I32)  Z[n_nuc]  Cnuc[n_nuc, 3]]     V only
            [C_x C_y C_z q (F64)]                       DIPOLE, QUADRUPOLE
            S | T | V [n_ao, n_ao]  or  MU[n_comp, n_ao, n_ao]  (OUT, stored)

`P` is `V`'s only (`P - C`); `eb_ket`, the ket primitive's exponent, `T`'s
only (`kernel_t`'s name); `B`, the ket shell's centre, the two moments' only
(`B - C`); `n_comp` is 3 for the dipole and 6 for the quadrupole
(`n_components`). `ab_*` is `A - B` exactly, from the centres, not from the record
arrays' `PB - PA`. `pair_arrays(target)` names the per-pair block; the
records of pair `n` are `rec0[n] <= k < rec1[n]`, contiguous, as
`launch.OneEClass.blocks` lays them out.

ROUND-OFF, NOT BITWISE, against the MMD host references
(`one_electron.overlap`, `kinetic`, `potential`, `multipole`): another
recurrence, another association. The tests state their bounds
(test/test_kernel_hgp.py), and hold every matrix to libcint's too.

A builder: vendor-neutral IR. It imports neither `ao/` (its witness) nor the
interpreter nor `reference/`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import MappingProxyType

from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.recurrence import hrr_body, hrr_name, vrr_body, vrr_name
from felizjueves.hgp.vrr import ZERO, Cart
from felizjueves.ir import (
    Assign,
    Boys,
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
    Weighted,
    walk,
)
from felizjueves.spec import OPERATOR_OF, Engine, Spec, Target

__all__ = [
    "L_MAX",
    "MOMENTS",
    "MOMENT_L_MAX",
    "OUT",
    "PAIR",
    "TARGETS",
    "build_hgp_kernel",
    "hgp_name",
    "hgp_spec",
    "factors",
    "l_max",
    "launch_shape",
    "n_components",
    "pair_arrays",
    "table_dags",
    "vrr_dag",
]

L_MAX = 1  # S, T, V: s and p shells, the scope they are built and checked for
MOMENT_L_MAX = 3  # the dipole and quadrupole: built and checked through f
TARGETS = (Target.S, Target.T, Target.V, Target.DIPOLE, Target.QUADRUPOLE)
MOMENTS = (Target.DIPOLE, Target.QUADRUPOLE)  # (r - C)^e on the ket, e = 1, 2
PAIR = "pair"  # the contracted shell pair: one thread each
OUT = {Target.S: "S", Target.T: "T", Target.V: "V", Target.DIPOLE: "MU",
       Target.QUADRUPOLE: "MU"}
_PREFIX = {Target.S: "hs", Target.T: "ht", Target.V: "hv",
           Target.DIPOLE: "hmu", Target.QUADRUPOLE: "hquad"}
# The quadrupole's components, `cartesians(2)` order (xx xy xz yy yz zz), as
# the axis pairs (d, e), d <= e: kernel_mu.build_quad_kernel's order.
_QUAD = tuple(tuple(d for d in range(3) for _ in range(c[d]))
              for c in vrr.cartesians(2))
_AXES = "xyz"
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)
_REC = "kr"  # the record loop's variable
_NUC = "c"  # the nucleus loop's variable
_BOYS = "boys"

_STORED = (
    "a plain store, no atomics: every element belongs to one shell pair and "
    "is written once (one thread per contracted shell pair), so the result "
    "is bit-reproducible; elements of pairs not launched are untouched")
_DIAGONAL = (
    "diagonal shell pair (off_a == off_b): the lower triangle i >= j only, "
    "each mirrored -- kernel_jab's placement")


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def hgp_spec(target: Target, la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` of `target` on `Engine.HGP`."""
    return Spec(OPERATOR_OF[target], 0, (la, lb), target, engine=Engine.HGP)


def _class(cls: Spec | tuple[Target, int, int]) -> tuple[Target, int, int]:
    spec = cls if isinstance(cls, Spec) else hgp_spec(*cls)
    if spec.target not in TARGETS:
        raise ValueError(f"{spec.key()}: the HGP one-electron kernels are "
                         f"{[t.value for t in TARGETS]}")
    if spec.engine is not Engine.HGP:
        raise ValueError(f"{spec.key()} is not an HGP key; the MMD kernel "
                         "of this target is kernel_s / kernel_t / kernel_v / "
                         "kernel_mu")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the HGP one-electron kernels take "
                         "no levers")
    la, lb = spec.am
    top = l_max(spec.target)
    if max(la, lb) > top:
        raise ValueError(
            f"{spec.key()}: the HGP one-electron kernels are built for "
            + ("s and p shells only" if top == 1 else "shells through f only")
            + f" (l <= {top})")
    return spec.target, la, lb


def l_max(target: Target) -> int:
    """The highest shell `l` `target`'s HGP kernel is built and checked
    for: `MOMENT_L_MAX` for the dipole and quadrupole, `L_MAX` else."""
    return MOMENT_L_MAX if target in MOMENTS else L_MAX


def n_components(target: Target) -> int:
    """The output's components: 3 for the dipole, 6 for the quadrupole,
    1 for `S`, `T`, `V`."""
    return {Target.DIPOLE: 3, Target.QUADRUPOLE: len(_QUAD)}.get(target, 1)


def hgp_name(cls: Spec | tuple[Target, int, int]) -> str:
    """The name `build_hgp_kernel(cls)` gives its kernel, unbuilt:
    `hs_{la}{lb}`, `ht_`, `hv_`, `hmu_`, `hquad_` (`hj_` / `hk_`'s
    spelling)."""
    target, la, lb = _class(cls)
    return f"{_PREFIX[target]}_{la}{lb}"


def pair_arrays(target: Target) -> tuple[str, ...]:
    """The per-pair array parameters of `target`'s kernel, in order."""
    if target not in TARGETS:
        raise ValueError(f"{target!r} is not an HGP one-electron target")
    out = ("rec0", "rec1", "off_a", "off_b", "ab_x", "ab_y", "ab_z")
    if target in MOMENTS:
        out += ("b_x", "b_y", "b_z")
    return out


def launch_shape(n_pair: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per shell pair,
    `ceil(n_pair/threads)` blocks. No `WarpReduce`, so a partial last block
    is legal."""
    if type(n_pair) is not int or n_pair < 1:
        raise ValueError(f"n_pair = {n_pair!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_pair // threads), (threads, 1)


# ---------------------------------------------------------------------------
# The recurrences: hgp.vrr / hgp.hrr DAGs, cut to what is read
# ---------------------------------------------------------------------------


def _shift(v: Cart, d: int, by: int) -> Cart:
    return tuple(c + by * (i == d) for i, c in enumerate(v))  # type: ignore[return-value]


def _hrr_dag(targets: tuple[hrr.Node, ...]) -> hrr.HrrDag:
    """`hgp.hrr`'s eq 28 closed over `targets` (all `(a, b | 0, 0)`), in
    `hrr.order_key`'s build order."""
    terms: dict[hrr.Node, tuple[hrr.Term, ...]] = {}
    todo = list(targets)
    while todo:
        n = todo.pop()
        if n not in terms:
            terms[n] = hrr.reduce(n)
            todo.extend(t.child for t in terms[n])
    la = min(sum(n[0]) for n in targets)
    lb = max(sum(n[1]) for n in targets)
    order = tuple(sorted(terms, key=hrr.order_key))
    return hrr.HrrDag((la, lb, 0, 0), "bra", targets,
                      MappingProxyType(terms), order)


def _boundary(dag: hrr.HrrDag) -> tuple[Cart, ...]:
    """The `e` of every `(e, 0 | 0, 0)` leaf `dag` reads, in VRR order."""
    return tuple(sorted((e for e, _ in dag.boundary),
                        key=lambda e: vrr.order_key((e, ZERO, 0))))


def vrr_dag(roots: tuple[Cart, ...], nuclear: bool) -> vrr.VrrDag:
    """`hgp.vrr`'s eq 25 at `f = 0`, cut to what `roots` reach: every Boys
    order for `V` (the point-charge limit), `m = 0` with the `m + 1` terms
    dropped for the overlap (module docstring)."""
    top = max(sum(e) for e in roots)
    terms = dict(vrr.build((0, top, 0, 0), prune=False).terms)
    if not nuclear:
        terms = {n: tuple(t for t in ts if t.child[2] == 0)
                 for n, ts in terms.items() if n[2] == 0}
    tg = tuple((e, ZERO, 0) for e in roots)
    keep = vrr.reachable(terms, tg)
    kept = {n: t for n, t in terms.items() if n in keep}
    return vrr.VrrDag((0, top, 0, 0), True, tg, MappingProxyType(kept),
                      tuple(sorted(kept, key=vrr.order_key)))


def _digits(v: Cart) -> str:
    return "_".join(str(c) for c in v)


def _table_name(table: int) -> Callable[[hrr.Node], str]:
    return lambda n: f"t{table}{hrr_name(n)}"


# ---------------------------------------------------------------------------
# Small spellings
# ---------------------------------------------------------------------------


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _sum(xs: list[Expr]) -> Expr:
    """`xs` as a left fold of ADDs from its first term."""
    out = xs[0]
    for x in xs[1:]:
        out = _c(Fn.ADD, out, x)
    return out


def _dot(a: Callable[[int], Expr], b: Callable[[int], Expr]) -> Expr:
    return _sum([_c(Fn.MUL, a(d), b(d)) for d in range(3)])


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def factors(body: tuple[Stmt, ...]) -> set[str]:
    """Every `Factor` scalar a `Weighted` of `body` reads."""
    out: set[str] = set()
    for s in walk(body):
        if isinstance(s, Decl) and isinstance(s.init, Weighted):
            out |= {t.g.name_in_kernel for t in s.init.terms
                    if t.g.name_in_kernel is not None}
    return out


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


def _tables(target: Target, la: int, lb: int
            ) -> dict[int, tuple[hrr.Node, ...]]:
    """Each contracted table's HRR targets (module docstring): one table for
    `S`, `V` and `MU`, three weighted by `beta^n` for `T` (an empty one
    left out); the moments one table, every `(a|b)` the operator moved
    onto the ket reaches."""
    na, nb = vrr.cartesians(la), vrr.cartesians(lb)
    if target is Target.T:
        out = {
            0: tuple((a, _shift(b, d, -2), ZERO, ZERO) for a in na
                     for b in nb for d in range(3) if b[d] >= 2),
            1: tuple((a, b, ZERO, ZERO) for a in na for b in nb),
            2: tuple((a, _shift(b, d, 2), ZERO, ZERO) for a in na
                     for b in nb for d in range(3)),
        }
        return {n: tuple(dict.fromkeys(t)) for n, t in out.items() if t}
    tg = tuple((a, b, ZERO, ZERO) for a in na for b in nb)
    if target in MOMENTS:
        tg += tuple((a, _shift(b, d, 1), ZERO, ZERO) for a in na for b in nb
                    for d in range(3))
    if target is Target.QUADRUPOLE:
        tg += tuple((a, _shift(_shift(b, d, 1), e, 1), ZERO, ZERO)
                    for a in na for b in nb for d, e in _QUAD)
    return {0: tuple(dict.fromkeys(tg))}


def _values(target: Target, la: int, lb: int,
            name: Callable[[int, hrr.Node], str]
            ) -> tuple[list[Stmt], dict[tuple[int, int, int], Expr]]:
    """The output elements: `(comp, i, j) -> Expr` over the HRR targets'
    locals, and the `Decl`s that form them."""
    na, nb = vrr.cartesians(la), vrr.cartesians(lb)
    stmts: list[Stmt] = []
    out: dict[tuple[int, int, int], Expr] = {}

    def h(table: int, a: Cart, b: Cart) -> Var:
        return Var(name(table, (a, b, ZERO, ZERO)))

    for i, a in enumerate(na):
        for j, b in enumerate(nb):
            if target in (Target.S, Target.V):
                out[(0, i, j)] = h(0, a, b)
            elif target is Target.T:
                v: Expr = _c(Fn.MUL, Lit(float(2 * lb + 3)), h(1, a, b))
                v = _c(Fn.SUB, v, _c(Fn.MUL, Lit(2.0), _sum(
                    [h(2, a, _shift(b, d, 2)) for d in range(3)])))
                low = [_c(Fn.MUL, Lit(float(b[d] * (b[d] - 1))),
                          h(0, a, _shift(b, d, -2))) for d in range(3)
                       if b[d] >= 2]
                if low:
                    v = _c(Fn.SUB, v, _c(Fn.MUL, Lit(0.5), _sum(low)))
                stmts.append(Decl(f"tv_{i}_{j}", Scalar.F64, init=v))
                out[(0, i, j)] = Var(f"tv_{i}_{j}")
            elif target is Target.QUADRUPOLE:
                for n, (d, e) in enumerate(_QUAD):
                    bd, be = (Var(f"BC_{_AXES[x]}") for x in (d, e))
                    v = _sum([
                        h(0, a, _shift(_shift(b, d, 1), e, 1)),
                        _c(Fn.MUL, be, h(0, a, _shift(b, d, 1))),
                        _c(Fn.MUL, bd, h(0, a, _shift(b, e, 1))),
                        _c(Fn.MUL, _c(Fn.MUL, bd, be), h(0, a, b))])
                    stmts.append(Decl(f"qu{n}_{i}_{j}", Scalar.F64,
                                      init=_c(Fn.MUL, Var("q"), v)))
                    out[(n, i, j)] = Var(f"qu{n}_{i}_{j}")
            else:
                for d in range(3):
                    v = _c(Fn.MUL, Var("q"), _c(
                        Fn.ADD, h(0, a, _shift(b, d, 1)),
                        _c(Fn.MUL, Var(f"BC_{_AXES[d]}"), h(0, a, b))))
                    stmts.append(Decl(f"mu{d}_{i}_{j}", Scalar.F64, init=v))
                    out[(d, i, j)] = Var(f"mu{d}_{i}_{j}")
    return stmts, out


def _scatter(target: Target, la: int, lb: int,
             values: dict[tuple[int, int, int], Expr]) -> Stmt:
    """`kernel_jab`'s placement, as plain stores (module docstring)."""
    out = OUT[target]
    na, nb = len(vrr.cartesians(la)), len(vrr.cartesians(lb))
    comps = n_components(target)

    def at(comp: int, row: Expr, col: Expr) -> Index:
        if target in MOMENTS:
            return Index(out, (Lit(comp), row, col))
        return Index(out, (row, col))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    off: list[Stmt] = [Note(_STORED)]
    diag: list[Stmt] = [Note(_DIAGONAL)]
    for comp in range(comps):
        for i in range(na):
            for j in range(nb):
                v = values[(comp, i, j)]
                off += [Assign(at(comp, plus("oa", i), plus("ob", j)), v),
                        Assign(at(comp, plus("ob", j), plus("oa", i)), v)]
                if la == lb and i >= j:
                    diag.append(Assign(at(comp, plus("oa", i),
                                          plus("oa", j)), v))
                    if i != j:
                        diag.append(Assign(at(comp, plus("oa", j),
                                              plus("oa", i)), v))
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    return If(differ, tuple(off), tuple(diag) if la == lb else ())


def build_hgp_kernel(cls: Spec | tuple[Target, int, int], *,
                     name: str | None = None) -> Kernel:
    """The HGP one-electron kernel of one class.

    `cls` is an HGP `Target.S` / `T` / `V` / `DIPOLE` / `QUADRUPOLE`
    `Spec`, or
    `(target, la, lb)`. See the module docstring.
    """
    target, la, lb = _class(cls)
    nuclear = target is Target.V
    hrrs, bounds = contracted_tables(target, la, lb)
    vdag = vrr_dag(boundary_union(bounds), nuclear)

    k = Var(_REC)
    # --- per record: the seed, the VRR, the accumulation -------------------
    if nuclear:
        top = max(n[2] for n in vdag.terms)

        def fundamental(m: int) -> Expr:
            return _c(Fn.MUL, Var("zw"), Index(_BOYS, (Lit(m),)))
    else:
        def fundamental(m: int) -> Expr:
            return Var("u")
    vstmts = vrr_body(vdag, fundamental)
    used = factors(vstmts)

    def read(e: Cart) -> Expr:
        return Var(vrr_name((e, ZERO, 0)))

    accumulate: list[Stmt] = []
    for n, bnd in bounds.items():
        w = {0: None, 1: Var("eb"), 2: Var("eb2")}[n] \
            if target is Target.T else None
        for e in bnd:
            x = read(e) if w is None else _c(Fn.MUL, w, read(e))
            accumulate.append(Assign(Var(acc_name(n, e)), _c(
                Fn.SUB if nuclear else Fn.ADD, Var(acc_name(n, e)), x)))

    record: list[Stmt] = [Decl("zeta", Scalar.F64,
                               init=Index("zeta_ab", (k,)))]
    if Factor.INV2P.value in used:
        record.append(Decl(Factor.INV2P.value, Scalar.F64,
                           init=_c(Fn.DIV, Lit(0.5), Var("zeta"))))
    if Factor.INV2P_ALPHA_OVER_P.value in used:
        record.append(Note("the point-charge limit of eq 25: rho/p -> 1"))
        record.append(Decl(Factor.INV2P_ALPHA_OVER_P.value, Scalar.F64,
                           init=Var(Factor.INV2P.value)))
    for d in range(3):
        if Factor.pa(d).value in used:
            record.append(Decl(Factor.pa(d).value, Scalar.F64,
                               init=Index(f"pa_{_AXES[d]}", (k,))))
    record += [
        Note("u = (0|0) = s_ab K_AB, K_AB = exp(zeta (P-A).(P-B)): s_ab = "
             "c_a c_b (pi/zeta)^{3/2} is the host's (ao.pairs.dress)"),
        Decl("u", Scalar.F64, init=_c(Fn.MUL, Index("s_ab", (k,)), _c(
            Fn.EXP, _c(Fn.MUL, Var("zeta"), _dot(
                lambda d: Index(f"pa_{_AXES[d]}", (k,)),
                lambda d: Index(f"pb_{_AXES[d]}", (k,))))))),
    ]
    if target is Target.T:
        record += [Decl("eb", Scalar.F64, init=Index("eb_ket", (k,))),
                   Note("the tables' weights beta^n: beta the ket exponent"),
                   Decl("eb2", Scalar.F64,
                        init=_c(Fn.MUL, Var("eb"), Var("eb")))]
    if nuclear:
        c = Var(_NUC)
        nucleus: list[Stmt] = [
            Note("eq 25's point-charge limit: W -> C, so WP = C - P"),
            *(Decl(Factor.wp(d).value, Scalar.F64, init=_c(
                Fn.SUB, Index("Cnuc", (c, Lit(d))),
                Index(f"p_{_AXES[d]}", (k,)))) for d in range(3)),
            Decl(_BOYS, Scalar.F64, (Lit(top + 1),)),
            Boys(_BOYS, top, _c(Fn.MUL, Var("zeta"), _dot(
                lambda d: Var(Factor.wp(d).value),
                lambda d: Var(Factor.wp(d).value)))),
            Note("(0|0)^m = Z s_ab K_AB (2 pi / (zeta (pi/zeta)^{3/2})) F_m; "
                 "the attractive sign enters as acc - Z u"),
            Decl("zw", Scalar.F64, init=_c(Fn.MUL, Index("Z", (c,)),
                                           Var("w0"))),
            *vstmts,
            *accumulate,
        ]
        record += [
            Note("w0 = s_ab K_AB (2/sqrt(pi)) sqrt(zeta) = c_a c_b K_AB "
                 "2 pi / zeta"),
            Decl("w0", Scalar.F64, init=_c(Fn.MUL, _c(
                Fn.MUL, Var("u"), Lit(_TWO_OVER_SQRT_PI)),
                _c(Fn.SQRT, Var("zeta")))),
            Note("the nuclei, serially"),
            For(_NUC, Lit(0), Var("n_nuc"), Lit(1), tuple(nucleus)),
        ]
    else:
        record += [*vstmts, *accumulate]

    what = {Target.S: "overlap S", Target.T: "kinetic T",
            Target.V: "nuclear attraction V",
            Target.DIPOLE: "dipole MU = <a| q (r - C) |b>",
            Target.QUADRUPOLE: ("quadrupole (raw second moments) "
                                "<a| q (r - C)_d (r - C)_e |b>")}[target]
    return assemble(
        target, la, lb, hrrs, bounds, tuple(record),
        name=name or hgp_name((target, la, lb)),
        title=(f"HGP {what}, class (la, lb) = ({la}, {lb}): a VRR per "
               "primitive pair, the contraction, the HRR on contracted "
               "values (one_electron/kernel_hgp.py)"),
        per_record=("the pair's primitive pairs, serially: the VRR per "
                    "record ([Fock] eq 25 at f = 0)"))


# ---------------------------------------------------------------------------
# The pair skeleton, shared with the Rys route (`one_electron.kernel_rys`)
# ---------------------------------------------------------------------------


def contracted_tables(target: Target, la: int, lb: int
                      ) -> tuple[dict[int, hrr.HrrDag],
                                 dict[int, tuple[Cart, ...]]]:
    """Each contracted table's HRR DAG, and the boundary `(e|0)` it reads
    (VRR order): what a route's per-record body must accumulate into
    `acc_name(table, e)` -- `S`, `V`, `MU` one table, `T` up to three
    (module docstring). Built for any `l`: `L_MAX` is the scope of this
    module's builder, not of this function."""
    return table_dags(_tables(target, la, lb))


def table_dags(tables: dict[int, tuple[hrr.Node, ...]]
               ) -> tuple[dict[int, hrr.HrrDag], dict[int, tuple[Cart, ...]]]:
    """`contracted_tables` for any tables: each table's HRR targets (all
    `(a, b | 0, 0)`) to its DAG and the boundary `(e|0)` it reads. The
    overlap gradient's tables (`gradient.kernel_hgs`) are not an operator's
    of this module."""
    hrrs = {n: _hrr_dag(tg) for n, tg in tables.items()}
    return hrrs, {n: _boundary(d) for n, d in hrrs.items()}


def boundary_union(bounds: dict[int, tuple[Cart, ...]]) -> tuple[Cart, ...]:
    """Every `e` any table reads, once, in VRR order."""
    return tuple(sorted({e for b in bounds.values() for e in b},
                        key=lambda e: vrr.order_key((e, ZERO, 0))))


def acc_name(table: int, e: Cart) -> str:
    """The contracted accumulator of `(e|0)` in table `table`."""
    return f"acc{table}_{_digits(e)}"


def assemble(target: Target, la: int, lb: int,
             hrrs: dict[int, hrr.HrrDag],
             bounds: dict[int, tuple[Cart, ...]],
             record: tuple[Stmt, ...], *, name: str, title: str,
             per_record: str) -> Kernel:
    """The kernel around a route's per-record body (module docstring, "THE
    THREAD MAPPING" and "THE PARAMETER BLOCK"): one thread per contracted
    shell pair, the accumulators `acc_name(n, e)` of `bounds` zeroed,
    `record` run over the pair's records `kr in [r0, r1)`, then the HRR of
    `hrrs` on the contracted values, the operator's combination and
    `kernel_jab`'s placement as plain stores. `record` reads the record
    arrays at `kr` and adds into the accumulators; `title` and `per_record`
    are its two notes."""
    nuclear = target is Target.V

    def hname(table: int, node: hrr.Node) -> str:
        return (_table_name(table)(node) if target is Target.T
                else hrr_name(node))

    # --- per pair: the HRR on contracted values, the values, the scatter ---
    hstmts: list[Stmt] = []
    for n, dag in hrrs.items():
        if target is Target.T:
            hstmts.append(Note(f"table {n}: SUM c_a c_b beta^{n} (e|0)"))
        hstmts += hrr_body(dag, lambda e, f, n=n: Var(acc_name(n, e)),
                           name=lambda node, n=n: hname(n, node))
    vals, values = _values(target, la, lb, hname)
    used_ab = factors(tuple(hstmts))

    pair = Var(PAIR)
    head: list[Stmt] = [
        Decl("r0", Scalar.I32, init=Index("rec0", (pair,))),
        Decl("r1", Scalar.I32, init=Index("rec1", (pair,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (pair,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (pair,))),
    ]
    for d in range(3):
        if Factor.ab(d).value in used_ab:
            head.append(Decl(Factor.ab(d).value, Scalar.F64,
                             init=Index(f"ab_{_AXES[d]}", (pair,))))
    if target in MOMENTS:
        head += [Decl(f"BC_{_AXES[d]}", Scalar.F64, init=_c(
            Fn.SUB, Index(f"b_{_AXES[d]}", (pair,)), Var(f"C_{_AXES[d]}")))
            for d in range(3)]
    head.append(Note("the contracted boundary (e|0), summed from 0.0 over the "
                     "pair's records"))
    for n, bnd in bounds.items():
        head += [Decl(acc_name(n, e), Scalar.F64, init=Lit(0.0)) for e in bnd]
    per_pair = (
        *head,
        Note(per_record),
        For(_REC, Var("r0"), Var("r1"), Lit(1), record),
        *hstmts,
        *vals,
        Note("scatter into the matrix and its mirror"),
        _scatter(target, la, lb, values),
    )
    body: tuple[Stmt, ...] = (
        Note(title),
        Note("launch: one thread per contracted shell pair"),
        Decl(PAIR, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, pair, Var("n_pair")), per_pair),
    )

    npair, nr, nao = Var("n_pair"), Var("n_rec"), Var("n_ao")
    params: list[Param] = [Param("n_pair", Scalar.I32),
                           Param("n_rec", Scalar.I32),
                           Param("n_ao", Scalar.I32),
                           _f64("zeta_ab", (nr,))]
    if nuclear:
        params += [_f64(f"p_{a}", (nr,)) for a in _AXES]
    params += [_f64(f"pa_{a}", (nr,)) for a in _AXES]
    params += [_f64(f"pb_{a}", (nr,)) for a in _AXES]
    params.append(_f64("s_ab", (nr,)))
    if target is Target.T:
        params.append(_f64("eb_ket", (nr,)))
    params += [_i32(n, (npair,)) for n in ("rec0", "rec1", "off_a", "off_b")]
    params += [_f64(n, (npair,)) for n in pair_arrays(target)[4:]]
    if nuclear:
        params += [Param("n_nuc", Scalar.I32),
                   _f64("Z", (Var("n_nuc"),)),
                   _f64("Cnuc", (Var("n_nuc"), Lit(3)))]
    if target in MOMENTS:
        params += [Param(f"C_{a}", Scalar.F64) for a in _AXES]
        params.append(Param("q", Scalar.F64))
        params.append(Param("MU", Scalar.F64, array=True, intent=Intent.OUT,
                            shape=(Lit(n_components(target)), nao, nao)))
    else:
        params.append(Param(OUT[target], Scalar.F64, array=True,
                            intent=Intent.OUT, shape=(nao, nao)))
    return Kernel(name, tuple(params), body)
