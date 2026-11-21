"""The three-center DF integral kernel, `Target.ERI3`: `(mn|P)`.

    build_eri3_kernel(spec | (la, lb, lP), lowstack=None, packed=False,
                      spherical_aux=False, spherical_bra=False) -> Kernel
    eri3_lowstack(spec | (la, lb, lP))     -> bool        the form's rule
    aux_harmonics(l)                       -> c[m][tau]   the aux columns
    eri3_spec(la, lb, lP)                  -> Spec        the class's key
    build_bra_transform_kernel(la, lb)     -> Kernel      sbra3_ (#747)
    quartet_spec(spec | (la, lb, lP))      -> Spec        Target.J (la lb|lP 0)
    eri3_plan(spec | (la, lb, lP))         -> (dag, order)  the plan's answer
    live_targets(la, lb, lP)               -> the [r]^(0) the contraction reads
    launch_shape(n_cell, threads)          -> (grid, block)

`(mn|P) = INT INT m(r1) n(r1) P(r2) / r12`, `m`, `n` Cartesian AOs of an
ordinary basis and `P` a Cartesian function of an auxiliary one, by
McMurchie & Davidson's Hermite path with the ket's Hermite "pair" a single
Gaussian (`eri/density_fit/eri3.py`'s module docstring, its witness):

    (mn|P) = SUM_{ij in mn} SUM_{k in P} s_ij s_k
             SUM_t E^{mn}_t(ij) SUM_tau (-1)^{|tau|} [t+tau]^(0)(P_ij - A_k)
             E^P_tau(k)

with `s_ij = c_i c_j (pi/p)^{3/2}` on each bra record (`ao.pairs.dress`'s
factor, `eri.density_fit.eri3`'s `s_ab`), `s_k = c_k (pi/alpha_k)^{3/2}` on each aux
record (`eri.density_fit.metric.aux_dress`) and `theta^2 = p alpha_k / (p + alpha_k)`.

AN ERI3 CLASS IS A TRIPLE THAT RUNS A QUARTET (`spec.py`, #441). The key is
the triple `(la, lb, lP)`; the recurrence is the `Target.J` quartet
`(la lb|lP 0)`'s at `L = la + lb + lP`: `eri3_plan` asks `pipeline.plan`
for THAT spec (the Schwarz and metric pattern, `kernel_metric.quartet_spec`),
and `recurrence.fundamental_prelude` / `recurrence_body` spell its one DAG in
`schedule`'s order, unchanged.

THE DEAD TARGETS. The bra block `E^{ab}_t` is structurally nonzero at every
`t_d <= a_d + b_d`, so the bra reaches every `|t| <= la + lb`; the
single-center `E^P_tau` is zero whenever `q_d - tau_d` is odd on some axis
(`e_fragment.center_absent`), so every live `tau` has `|tau| = lP (mod 2)`.
The contraction therefore reads `[r]^(0)` at exactly `|r| >= lP mod 2`, and,
when the bra is (ss), only at `|r| = lP (mod 2)` (`live_targets`; the
formula is pinned independently in the test). The DAG builds the whole
`|r| <= L` tower, so the other targets -- and every intermediate, fundamental
and Boys weight only they read -- are DROPPED by one backward liveness pass
(`_live`, `kernel_metric`'s rule restated), because a declared but unread
local fails #92's filter on both compilers. Dropping a dead `Decl` changes
no value that is kept.

THE KET SIGN IS ONE SIGN PER CLASS. Every live `tau` has the parity of `lP`,
so `(-1)^{|tau|} = (-1)^{lP}` on every term: the inner sum is spelled
unsigned and the class's sign is applied once, `acc - w v` for odd `lP`
(`acc + w v` for even). Negation is exact.

THE THREAD MAPPING: ONE THREAD PER CELL, a cell one (AO shell pair, aux
shell) of `eri.density_fit.eri3.eri3_classes`' staging, `cell = THREAD_X + BLOCK_X *
BLOCK_DIM_X`, guarded by `cell < n_cell`. The thread walks its cell's two
record ranges serially -- bra `[ab_rec0, ab_rec1)` outer, the `mn` block
formed once per bra record; aux `[p_rec0, p_rec1)` inner, the tower and the
`P` block once per (bra, aux) -- and accumulates `nA x nB x nP`
accumulators, `acc{a}_{b}_{q}`, in registers (`(dd|g)` is 540: high classes
will spill; nothing here is measured).

BOTH `E` FRAGMENTS ARE INLINED: `e_fragment.block_fragment(la, lb)` into
`Ebra` (its `zeta` the record's `zeta_ab`, `half_inv_zeta = 0.5 / zeta`,
`PA_d` / `PB_d` the record's `XPA` / `XPB`) and
`e_fragment.center_fragment(lP)` into `Eket` (`half_inv_zeta = 0.5 /
alpha`); the ket fragment's locals shadow the bra's, as in `kernel_metric`,
and only the arrays are read. Every `Ebra[row, i]` and `Eket[q, j]` read is
a structurally nonzero entry, which the fragment writes.

THE CONTRACTION, per (bra, aux) record pair:

    u_iq   = SUM_tau [t_i + tau]^(0) Eket[q, tau]          live tau of row q
    v_abq  = SUM_t Ebra[(a,b), t] u_tq                     t_d <= a_d + b_d
    acc_abq = acc_abq +- (s_ij s_k) v_abq                  (-1)^{lP}

each sum a left fold in `hermite_targets` order from its first live term.
`u` is the inner sum `kernel_metric` spells inside every element, named once
per `(t, q)` because it does not depend on the bra row: a common
subexpression, the same operations in the same order, so the same rounding.
NO PREFACTOR: `(pi/p)^{3/2}`, `(pi/alpha)^{3/2}` and the coefficients are
inside the host's `s_ab` and `s`, `(2/sqrt(pi)) theta` inside `[0]^(m)` (the
prelude's `W0`).

ROUND-OFF, NOT BITWISE, against `eri.density_fit.eri3`: the host's `[t|tau]` is
`reference.pq`'s fixed recursion, the kernel's the plan's MILD DAG, and the
host dresses each `E` entry where the kernel multiplies `s_ij s_k` once.

THE SCATTER, a plain store mirrored in `(m, n)`, into `[m][n][P]` (aux
fastest, `eri.density_fit.eri3`'s layout): an off-diagonal shell pair (`off_a != off_b`)
writes element `(a, b, q)` to `eri3[off_a + a, off_b + b, off_p + q]` and
`eri3[off_b + b, off_a + a, off_p + q]`; a diagonal one (only when
`la == lb`) writes `a >= b`, mirrored. No atomics: each element belongs to
exactly one cell, so runs are bit-reproducible. A pair staged SWAPPED needs
no flag here: with the mirror, the transposed placement `eri.density_fit.eri3.
place_staged` makes writes the same two elements.

THE BOYS DOMAIN. A tight aux exponent far from a bra centre puts `T` far
past where the old narrow series overflowed (near 709); the one `fj_boys`
every kernel carries since #301 holds on all of `reference.boys`' domain.

THE LOW-STACK FORM (#606), the default from (df|s), (ff|s) and every aux-i
class. The form above holds `Ebra[nA nB][n_t(la+lb)]`, `Eket` and the
`nA nB nP` accumulators per thread, and nvcc keeps those blocks whole: #606
measured `sm_86` frames of 13,856 B at (dd|g), 32,040 B at (df|g) and
70,232 B at (ff|g) (the `E` blocks alone are 14,280, 28,680 and 69,000 B),
reserved for every resident thread (~7.3 GB of a 10 GB RTX 3080 at
(ff|g)). `build_eri3_kernel(cls)` therefore emits, whenever
`eri3_lowstack(cls)`, a kernel with the SAME name, parameters, launch and
output contract and neither block nor accumulator:

    per cell:    eri3[own (m, n, P)] = 0;  R[n_t(L)], U[n_t(la+lb)]  local
    per bra:     Ex Ey Ez = e_fragment.axis_tables_fragment(la, lb)
    per aux:     the tower, each live [r]^(0) stored into R as it forms
                     (recurrence.store_targets);
                 Cx Cy Cz = e_fragment.center_tables_fragment(lP)
                 for each aux row q            (cartesian_loops)
                     U[t] = SUM_tau R[t+tau] E^q_tau   every |t| <= la+lb,
                                               (rolled_center_sum)
                     for each bra row (a, b)   (cartesian_loops)
                         eri3[m, n, P] +-= s_ij s_k SUM_t E^{ab}_t U[t]
                                               (rolled_e_sum)
    per cell:    the (m, n) mirror, a plain copy

so the frame is `R`, `U` and the six tables (8,280 B at (ff|i), 6,336 B at
(ff|g)) plus whatever of the tower the compiler spills, and the text is the
tower and a fixed set of loops (124 KB at (ff|i) against 1.56 MB for
(ff|g) unrolled). Accumulating in place is legal because each element
belongs to exactly one cell, which zeroes its slots first: no atomics, and
runs are still bit-reproducible. A diagonal pair forms `a >= b` only, as
above. Interpreted, both forms agree with the host to the same bound and
with each other to `1e-14` of the class's max (test/test_kernel_df_lowstack
.py); the product `E_x (E_y (E_z U))` is in another association from the
block's, so only s and p are bitwise.

WHY THE RULE. `eri3_lowstack` is `la + lb + lP >= LOWSTACK_MIN_L = 10` or
the unrolled `E` blocks over (pf|g)'s, `E_DOUBLES_MAX = 1575` doubles. It
is set from MEASURED rows (data/measured/df_lowstack.json, `devtools/
measure.sh df-lowstack-compile`, its rule declared before any row): every
class of the ladder in both forms where the gate is read, each unit
compiled alone under a 300 s budget. An unrolled form is admissible at
<= 14,000 B of stack (#606's ~14 KB) and within the budget on `sm_86` and
`gfx1200`. The unrolled (dd|g) is not (15,992 B of `sm_86` stack, 27,276 B
of spill stores), so it is gated; the largest admissible unrolled class is
(pf|g) (13,928 B on `sm_86`, 9,412 B of `gfx1200` scratch), whose blocks
are the bound. (df|g) unrolled is 33,992 B / 52 s on `sm_86`, and (ff|g)
unrolled does not compile within 300 s there (242 s and 34,660 B of
scratch on `gfx1200`). On the ladder `la <= lb <= 3`, `lP <= 6` that is
25 of 70 classes: every (df|.) and (ff|.), every aux-i row, and (dd|g),
(pf|h), (dd|h); nothing at `L <= 4`, so every golden is unchanged. The
`L >= 10` arm is for the tower, which grows fastest (the `[p|q]` kernel's
one-batch frame is 4,832 B at L = 10 and 12,496 B at L = 12,
`kernels.json`). Every low-stack unit of the ladder is at most 8,256 B of
`sm_86` stack (8,512 B of `gfx1200` scratch) with at most 68 B of `sm_86`
spill stores, and compiles in at most 5.1 s.

THE SPHERICAL-AUX OUTPUT (`spherical_aux=True`; FelizJueves#743,
FreeBreakfast#1736). The packed low-stack kernel with the aux index already
in real solid harmonics: `(ab|Q)`, `a`, `b` still Cartesian, `2 lP + 1`
columns per aux shell where the Cartesian form has `(lP+1)(lP+2)/2`.
Cartesian `P` is never formed. An aux primitive is single-centre, so by
Hobson's theorem a solid harmonic on it is a pure degree-`lP` Hermite
function,

    S_m(r_A) e^{-alpha r_A^2} = (2 alpha)^{-lP} SUM_{|tau| = lP} c[m][tau]
                                Lambda_tau(r_A),

and `E^m_tau = (2 alpha)^{-lP} c[m][tau]` at `|tau| = lP`, zero below:
the lower-degree Hermite terms every Cartesian row carries cancel in the
harmonic combination. So per (bra, aux) record pair the single-centre
tables and the `rolled_center_sum` are gone, and per column `m`

    U[t]        = SUM_{|tau| = lP} c[m][tau] [t+tau]^(0)      every |t| <= la+lb
    eri3[a,b,m] +-= s_ij s_k (2 alpha)^{-lP} SUM_t E^{ab}_t U[t]

with the class's one sign `(-1)^{lP}` as before. `c[m][tau]` is
`aux_harmonics(lP)`, a compile-time table spelled as literals: the `2 lP +
1` columns are unrolled, each `U` sum over that column's nonzero `tau`
only, and the `t` and `(a, b)` loops stay rolled, so no coefficient array
and no extra stack. `(2 alpha)^{-lP}` is folded into the record's `w`.

THE CONVENTION, which the consumer's transform must equal (FreeBreakfast's
`fb_basis::spherical_transform_matrix`, restricted to one shell's block):
column `m` runs `-lP ..= lP` (PySCF / libcint order), row `tau` over
`cartesians(lP)`; each column is the Schlegel-Frisch real solid harmonic
normalized under the Gaussian-moment product `<x^a, x^b> = PROD_d (a_d +
b_d - 1)!!` and then scaled by `sqrt((2 lP - 1)!!)`, the ratio between that
product and an AO normalized as `(lP, 0, 0)` on every component. s and p
are the identity: `p` columns stay `x, y, z`. So `c` is that transform's
block transposed, and the output is the Cartesian packed output contracted
by it on the aux index, to rounding (test/test_kernel_eri3_spherical.py).

The mode exists only in the low-stack form, at every class: the unrolled
form's measured rule (`E_DOUBLES_MAX`) is about its `E` blocks and
accumulators, and is untouched. Named `speri3_` (`speri3_333`); the packed
parameters, launch and cell arrays, with `n_aux` and `dst` counting
spherical aux columns. The square `eri3` has no such mode.

THE FULLY SPHERICAL OUTPUT (`spherical_bra=True` with `spherical_aux`;
FelizJueves#745). Both bra indices in real solid harmonics too: `(mu nu|Q)`,
`(2 la + 1)(2 lb + 1)(2 lP + 1)` elements a cell, at `dst + n_aux (mu (2 lb
+ 1) + nu) + m`. The low-stack form has no staged block to transform (it
accumulates straight into global memory, #606), and a cell's spherical
region is smaller than its Cartesian block, so neither "transform before
the store" nor an in-place pass over the accumulated block is open.
Instead, inside each unrolled aux column `m`, per primitive triple:

    V[a, b]      = SUM_t E^{ab}_t U[t]          every Cartesian (a, b), LOCAL
    T[b]         = SUM_a c^{la}[mu][a] V[a, b]  one mu at a time
    out[mu,nu,m] +-= w SUM_b c^{lb}[nu][b] T[b]

`c^l` is `aux_harmonics(l)`, the aux convention, spelled as literals over
its nonzero terms, `mu` and `nu` unrolled and the `T` row's `b` rolled.
The added stack is `V[nA nB]` and `T[nB]` (`V_ARRAY`, `T_ARRAY`), 880 B at
(ff|.), independent of `lP`; the global read-modify-writes per column and
primitive triple fall from `nA nB` to `(2 la + 1)(2 lb + 1)`. A diagonal
pair forms `a >= b` and mirrors within `V`, accumulates `mu >= nu`, and is
mirrored on spherical indices at the end. s and p are the identity: with
`la, lb <= 1` there is no `V` and no transform, and the body is
`speri3_`'s unchanged (`2 l + 1` is the Cartesian count there, so even the
slots are the same). Named `ssperi3_`; the packed parameters, `n_col`
counting spherical bra functions (`eri3.packed_columns(...,
spherical_bra=True)`). Boys, the tower, the bra tables, the aux columns,
the records, the cells and the launch are `speri3_`'s.

    params  n_cell n_bra_rec n_aux_rec n_ao n_aux (I32)
            zeta_ab[n_bra_rec]  P XPA XPB[n_bra_rec, 3]  s_ab[n_bra_rec]
            A[n_aux_rec, 3]  alpha[n_aux_rec]  s_aux[n_aux_rec]
            ab_rec0 ab_rec1 p_rec0 p_rec1 off_a off_b off_p [n_cell]  (I32)
            eri3[n_ao, n_ao, n_aux]  (OUT, written once per element)

The record and cell arrays are `eri.density_fit.eri3.Eri3Class.arrays()`' by name,
interleaved `[n_rec, 3]` and passed as plain positional parameters, except
`s`, which is `s_aux` here (`PARAM_OF`): the prelude declares a local
`s = -2 theta^2` (`recurrence.fundamental_prelude`), which would shadow it.
Elements of cells NOT launched are untouched: a caller that launches a subset
zeroes first.

A builder: vendor-neutral IR. It imports neither `ao/` (its witness) nor the
interpreter nor `reference/`.
"""

from __future__ import annotations

import math

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
    Inline,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
    Weighted,
)
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
    store_targets,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "BRA",
    "CELL",
    "KET",
    "OUT",
    "PACKED_CELL_ARRAYS",
    "PARAM_OF",
    "aux_harmonics",
    "build_bra_transform_kernel",
    "build_eri3_kernel",
    "bra_transform_name",
    "E_DOUBLES_MAX",
    "eri3_e_doubles",
    "eri3_lowstack",
    "eri3_name",
    "eri3_plan",
    "eri3_spec",
    "launch_shape",
    "live_targets",
    "quartet_spec",
]

CELL = "cell"  # one thread per (AO shell pair, aux shell)
BRA = "ib"  # the bra record, serial
KET = "ik"  # the aux record, serial inside the bra
OUT = "eri3"
S_AUX = "s_aux"  # the aux record's scale: `s` is a prelude local
# `Eri3Class.arrays()` key -> this kernel's param name, where they differ
PARAM_OF = {"s": S_AUX}
# THE PACKED FORM's cell arrays (`packed=True`), in parameter order: the two
# record ranges, the cell's destination offset into the packed output and its
# diagonal flag (`eri.density_fit.eri3.packed_cells`' keys)
PACKED_CELL_ARRAYS = ("ab_rec0", "ab_rec1", "p_rec0", "p_rec1", "dst",
                      "diag")

_DIAGONAL = (
    "diagonal AO shell pair (off_a == off_b): a >= b only, each mirrored "
    "once -- kernel_jab's diagonal rule, so (mn|P) and (nm|P) are one value")
_PACKED_DIAGONAL = (
    "diagonal AO shell pair (diag): a >= b formed, each mirrored once into "
    "the full square block -- the square kernel's values, bitwise")


def eri3_name(cls: Spec | tuple[int, int, int], *,
              packed: bool = False, spherical_aux: bool = False,
              spherical_bra: bool = False) -> str:
    """The kernel name of a class: `eri3_` and its three `l`s, run
    together while every `l <= 9` (`eri3_220`) and `_`-separated once any
    is two digits (`eri3_0_11_0`), so `(0, 11, 0)` and `(0, 1, 10)` never
    share a name (#445). The packed form's is `peri3_` (`peri3_220`), its
    spherical-aux mode's `speri3_` (`speri3_220`), and the fully spherical
    one's `ssperi3_` (`ssperi3_220`, #745)."""
    am = cls.am if isinstance(cls, Spec) else tuple(cls)
    sep = "_" if max(am) > 9 else ""
    head = ("ssperi3_" if spherical_bra else "speri3_" if spherical_aux
            else "peri3_" if packed else "eri3_")
    return head + sep.join(map(str, am))


def eri3_spec(la: int, lb: int, lP: int) -> Spec:
    """The class `(la, lb, lP)` as a generation key, `Target.ERI3`."""
    return Spec(Operator.COULOMB, 0, (la, lb, lP), Target.ERI3)


def _spec(cls: Spec | tuple[int, int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else eri3_spec(*cls)
    if spec.target is not Target.ERI3:
        raise ValueError(f"{spec.key()} is not a Target.ERI3 key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the ERI3 kernel takes no levers")
    return spec


def quartet_spec(cls: Spec | tuple[int, int, int]) -> Spec:
    """The `Target.J` quartet `(la lb | lP 0)` whose recurrence the class
    runs -- the key the plan layer is asked for (module docstring)."""
    la, lb, lP = _spec(cls).am
    return Spec(Operator.COULOMB, 0, (la, lb, lP, 0), Target.J)


def eri3_plan(cls: Spec | tuple[int, int, int]) -> tuple[BatchDag, Order]:
    """`pipeline.plan(quartet_spec(cls))`'s one DAG and `schedule(dag)`."""
    (dag,) = plan(quartet_spec(cls)).dags
    return dag, schedule(dag)


def launch_shape(n_cell: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per cell, `ceil(n_cell/threads)`
    blocks; a partial last block is legal."""
    if type(n_cell) is not int or n_cell < 1:
        raise ValueError(f"n_cell = {n_cell!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_cell // threads), (threads, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _plus(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, int, int]:
    return (p[0] + q[0], p[1] + q[1], p[2] + q[2])


def _bra_cols(a: tuple[int, int, int], b: tuple[int, int, int], L: int
              ) -> list[tuple[int, tuple[int, int, int]]]:
    """`(column, t)` of bra row `(a, b)`'s structurally nonzero entries, in
    `hermite_targets(L)` order -- the columns `block_fragment` writes."""
    return [(i, t) for i, t in enumerate(hermite_targets(L))
            if all(t[d] <= a[d] + b[d] for d in range(3))]


def _aux_cols(q: tuple[int, int, int], l: int  # noqa: E741
              ) -> list[tuple[int, tuple[int, int, int]]]:
    """`(column, tau)` of aux row `q`'s structurally nonzero entries, in
    `hermite_targets(l)` order -- the columns `center_fragment` writes."""
    return [(j, t) for j, t in enumerate(hermite_targets(l))
            if not ef.center_absent(q, t)]


def live_targets(la: int, lb: int, lP: int
                 ) -> frozenset[tuple[int, int, int]]:
    """Every `[r]^(0)` the contraction reads: `r = t + tau` over the live
    entries of the bra block and the single-center block (module
    docstring)."""
    L = la + lb
    out: set[tuple[int, int, int]] = set()
    for a, b in ef.block_rows(la, lb):
        for _, t in _bra_cols(a, b, L):
            for q in ef.cartesians(lP):
                for _, tau in _aux_cols(q, lP):
                    out.add(_plus(t, tau))
    return frozenset(out)


def _reads(e: object) -> set[str]:
    """Every name an expression reads: `Var`s, arrays, `Factor` scalars."""
    if isinstance(e, Var):
        return {e.name}
    if isinstance(e, Index):
        return {e.array}.union(*(_reads(i) for i in e.at))
    if isinstance(e, Call):
        return set().union(*(_reads(a) for a in e.args))
    if isinstance(e, Weighted):
        out: set[str] = set()
        for t in e.terms:
            out |= _reads(t.src)
            if t.g.name_in_kernel is not None:
                out.add(t.g.name_in_kernel)
        return out
    return set()  # Lit


def _live(body: tuple[Stmt, ...], wanted: set[str]) -> tuple[Stmt, ...]:
    """`body` (`Decl`s, `Note`s and the prelude's `Boys` fill) without the
    `Decl`s that neither `wanted` nor a kept later statement reads, and
    without a `Boys` whose array nothing kept reads: one backward pass."""
    kept: list[Stmt] = []
    live = set(wanted)
    for st in reversed(body):
        if isinstance(st, Decl):
            if st.name not in live:
                continue
            live |= _reads(st.init)
        elif isinstance(st, Boys):
            if st.dst not in live:
                continue
            live |= _reads(st.t)
        elif not isinstance(st, Note):
            raise TypeError(f"_live takes Decls, Notes and Boys, got "
                            f"{type(st).__name__}")
        kept.append(st)
    return tuple(reversed(kept))


def _bra_binds(frag, zeta: Expr, at: Expr) -> dict[str, Expr]:
    """The bra block's binds: `zeta`, `1/2zeta`, `PA_d`, `PB_d`, `E`."""
    out: dict[str, Expr] = {ef.ZETA: zeta, ef.E_ARRAY: Var("Ebra")}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), zeta)
    for d in range(3):
        out[Factor.pa(d).name_in_kernel] = Index("XPA", (at, Lit(d)))
        out[Factor.pb(d).name_in_kernel] = Index("XPB", (at, Lit(d)))
    return out


def _aux_binds(frag, alpha: Expr) -> dict[str, Expr]:
    """The single-center block's binds: `1/2alpha` (when read) and `E`."""
    out: dict[str, Expr] = {ef.E_ARRAY: Var("Eket")}
    if ef.HALF_INV_ZETA in frag.binds:
        out[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), alpha)
    return out


def _acc(a: int, b: int, q: int) -> str:
    return f"acc{a}_{b}_{q}"


def _u(i: int, q: int) -> str:
    return f"u{i}_{q}"


def build_eri3_kernel(cls: Spec | tuple[int, int, int], *,
                      name: str | None = None,
                      lowstack: bool | None = None,
                      packed: bool = False,
                      spherical_aux: bool = False,
                      spherical_bra: bool = False) -> Kernel:
    """The three-center kernel of one class (module docstring).

    `cls` is a `Target.ERI3` `Spec` or `(la, lb, lP)`; any class the
    fragments and the plan accept is built. `lowstack` picks the form:
    `None` (the default) is `eri3_lowstack(cls)` (THE LOW-STACK FORM),
    `True` / `False` force either at any class. Both have the same name,
    parameters, launch and output contract. `packed` writes each cell's
    whole block at its table's destination offset instead (THE PACKED
    OUTPUT): the same arithmetic in either form, named `peri3_`.
    `spherical_aux` (packed only) writes the aux index in real solid
    harmonics (THE SPHERICAL-AUX OUTPUT), always in the low-stack form,
    named `speri3_`; it refuses `packed=False` and `lowstack=False`.
    `spherical_bra` (with `spherical_aux`) writes both bra indices in real
    solid harmonics too (THE FULLY SPHERICAL OUTPUT, #745), named
    `ssperi3_`."""
    spec = _spec(cls)
    if spherical_bra and not spherical_aux:
        raise ValueError("spherical_bra is a mode of the spherical-aux "
                         "output: pass spherical_aux=True")
    if spherical_aux:
        if not packed:
            raise ValueError("spherical_aux is a mode of the packed output: "
                             "pass packed=True")
        if lowstack is False:
            raise ValueError("spherical_aux exists in the low-stack form "
                             "only (THE SPHERICAL-AUX OUTPUT)")
        return _build_lowstack(spec, name, packed=True, spherical=True,
                               spherical_bra=spherical_bra)
    if lowstack is None:
        lowstack = eri3_lowstack(spec)
    if lowstack:
        return _build_lowstack(spec, name, packed)
    la, lb, lP = spec.am
    Lab = la + lb
    dag, order = eri3_plan(spec)
    fB, fP = ef.block_fragment(la, lb), ef.center_fragment(lP)
    rowsB, rowsP = ef.block_rows(la, lb), ef.cartesians(lP)
    nA, nB, nP = len(ef.cartesians(la)), len(ef.cartesians(lb)), len(rowsP)
    tsB = hermite_targets(Lab)
    odd = lP % 2 == 1  # (-1)^{|tau|}, one sign for the class
    cell, ib, ik = Var(CELL), Var(BRA), Var(KET)

    # --- the contraction of one (bra, aux) record pair ----------------------
    bra_cols = [_bra_cols(a, b, Lab) for a, b in rowsB]
    read_t = sorted({i for cols in bra_cols for i, _ in cols})
    inner: list[Stmt] = []
    for i in read_t:
        for q, rq in enumerate(rowsP):
            u: Expr | None = None
            for j, tau in _aux_cols(rq, lP):
                term = _c(Fn.MUL, Var(node_name((_plus(tsB[i], tau), 0))),
                          Index("Eket", (Lit(q), Lit(j))))
                u = term if u is None else _c(Fn.ADD, u, term)
            assert u is not None  # every row has a live column
            inner.append(Decl(_u(i, q), Scalar.F64, init=u))
    contract: list[Stmt] = []
    for row, cols in enumerate(bra_cols):
        a, b = divmod(row, nB)
        for q in range(nP):
            v: Expr | None = None
            for i, _ in cols:
                term = _c(Fn.MUL, Index("Ebra", (Lit(row), Lit(i))),
                          Var(_u(i, q)))
                v = term if v is None else _c(Fn.ADD, v, term)
            assert v is not None  # t = 0 is live in every bra row
            acc = Var(_acc(a, b, q))
            contract.append(Assign(acc, _c(
                Fn.SUB if odd else Fn.ADD, acc, _c(Fn.MUL, Var("w"), v))))
    wanted: set[str] = set()
    for st in inner:
        wanted |= _reads(st.init)

    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tower = _live((*prelude, *recurrence_body(dag, order, fundamental)),
                  wanted)

    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                      Index("A", (ik, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Note("theta^2 = p alpha / (p + alpha): the aux Gaussian is a "
             "Hermite pair at zeta = alpha, Q = A"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ze"), Var("aq")),
            _c(Fn.ADD, Var("ze"), Var("aq"))))),
        Note("[t+tau]^(0) by the plan's DAG for the quartet "
             f"({la},{lb}|{lP},0), the targets no live t + tau reaches (the "
             "center E's parity zeros) dropped with everything only they "
             "read"),
        *tower,
        Note("the undressed single-center E block of this aux record: "
             "e_fragment.center_fragment, inlined"),
        Decl("Eket", Scalar.F64, (Lit(nP), Lit(len(hermite_targets(lP))))),
        Inline(fP, _aux_binds(fP, Var("aq"))),
        Note("u_tq = SUM_tau [t+tau] Eket_q,tau over the live tau, once per "
             "(t, q): it does not depend on the bra row"),
        *inner,
        Decl("w", Scalar.F64, init=_c(Fn.MUL, Var("sb"),
                                      Index(S_AUX, (ik,)))),
        Note(f"per element (a, b, q): acc {'-' if odd else '+'}= s_ij s_k "
             "SUM_t Ebra_t u_tq over the structurally nonzero t; "
             f"(-1)^|tau| = (-1)^{lP} on every live tau"),
        *contract,
    )
    bra_body: tuple[Stmt, ...] = (
        Decl("ze", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Note("the undressed E block of this bra record, once: "
             "e_fragment.block_fragment, inlined"),
        Decl("Ebra", Scalar.F64, (Lit(len(rowsB)), Lit(len(tsB)))),
        Inline(fB, _bra_binds(fB, Var("ze"), ib)),
        Note("the aux records of the cell"),
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )

    # --- the scatter --------------------------------------------------------
    if packed:
        return _packed_kernel(spec, name, bra_body, (nA, nB, nP), tuple(
            Decl(_acc(a, b, q), Scalar.F64, init=Lit(0.0))
            for a in range(nA) for b in range(nB) for q in range(nP)))

    def at(m: Expr, n: Expr, q: int) -> Index:
        return Index(OUT, (m, n, plus("op", q)))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    off_diag: list[Stmt] = []
    for a in range(nA):
        for b in range(nB):
            for q in range(nP):
                v = Var(_acc(a, b, q))
                off_diag += [Assign(at(plus("oa", a), plus("ob", b), q), v),
                             Assign(at(plus("ob", b), plus("oa", a), q), v)]
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    scatter: Stmt
    if la == lb:
        diag: list[Stmt] = [Note(_DIAGONAL)]
        for a in range(nA):
            for b in range(a + 1):
                for q in range(nP):
                    v = Var(_acc(a, b, q))
                    diag.append(Assign(at(plus("oa", a), plus("oa", b), q), v))
                    if a != b:
                        diag.append(Assign(
                            at(plus("oa", b), plus("oa", a), q), v))
        scatter = If(differ, tuple(off_diag), tuple(diag))
    else:
        scatter = If(differ, tuple(off_diag))

    per_cell: tuple[Stmt, ...] = (
        Decl("b0", Scalar.I32, init=Index("ab_rec0", (cell,))),
        Decl("b1", Scalar.I32, init=Index("ab_rec1", (cell,))),
        Decl("p0", Scalar.I32, init=Index("p_rec0", (cell,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (cell,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (cell,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (cell,))),
        Decl("op", Scalar.I32, init=Index("off_p", (cell,))),
        *(Decl(_acc(a, b, q), Scalar.F64, init=Lit(0.0))
          for a in range(nA) for b in range(nB) for q in range(nP)),
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        Note("scatter: one plain store per element and its (m, n) mirror, "
             "no atomics -- each element belongs to one cell"),
        scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"three-center DF integral, class (la, lb, lP) = ({la}, {lb}, "
             f"{lP}): (mn|P) by the quartet ({la},{lb}|{lP},0)'s recurrence "
             f"at L = {Lab + lP} (ao/eri3.py)"),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )
    return Kernel(name or eri3_name(spec), _params(), body)


def _packed_at(c: Expr, q: Expr) -> Index:
    """The packed output's element `q` of block column `c` of the cell
    whose destination offset is the local `od`: `od + n_aux c + q`, the
    aux index fastest (THE PACKED OUTPUT)."""
    return Index(OUT, (_c(Fn.ADD, Var("od"), _c(Fn.ADD, _c(
        Fn.MUL, Var("n_aux"), c), q)),))


def _cell_ranges(cell: Var) -> tuple[Stmt, ...]:
    return (Decl("b0", Scalar.I32, init=Index("ab_rec0", (cell,))),
            Decl("b1", Scalar.I32, init=Index("ab_rec1", (cell,))),
            Decl("p0", Scalar.I32, init=Index("p_rec0", (cell,))),
            Decl("p1", Scalar.I32, init=Index("p_rec1", (cell,))))


def _packed_kernel(spec: Spec, name: str | None, bra_body: tuple[Stmt, ...],
                   n: tuple[int, int, int],
                   accs: tuple[Stmt, ...]) -> Kernel:
    """The unrolled form's packed kernel (THE PACKED OUTPUT): its
    accumulators and bra loop unchanged, the scatter one plain store per
    element of the cell's whole block at `dst`; a diagonal pair (`diag`,
    read only where `la == lb`) stores `a >= b` and its mirror, the square
    kernel's values."""
    la, lb, lP = spec.am
    nA, nB, nP = n
    cell = Var(CELL)

    def at(a: int, b: int, q: int) -> Index:
        return _packed_at(Lit(a * nB + b), Lit(q))

    whole: list[Stmt] = [Assign(at(a, b, q), Var(_acc(a, b, q)))
                         for a in range(nA) for b in range(nB)
                         for q in range(nP)]
    flag: tuple[Stmt, ...] = ()
    scatter: tuple[Stmt, ...] = tuple(whole)
    if la == lb:
        diag: list[Stmt] = [Note(_PACKED_DIAGONAL)]
        for a in range(nA):
            for b in range(a + 1):
                for q in range(nP):
                    v = Var(_acc(a, b, q))
                    diag.append(Assign(at(a, b, q), v))
                    if a != b:
                        diag.append(Assign(at(b, a, q), v))
        flag = (Decl("dg", Scalar.I32, init=Index("diag", (cell,))),)
        scatter = (If(_c(Fn.LT, Var("dg"), Lit(1)), tuple(whole),
                      tuple(diag)),)
    per_cell: tuple[Stmt, ...] = (
        *_cell_ranges(cell),
        Decl("od", Scalar.I32, init=Index("dst", (cell,))),
        *flag,
        *accs,
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        Note("scatter: the cell's whole block at its destination offset, "
             "one plain store per element, no atomics -- each element "
             "belongs to one cell"),
        *scatter,
    )
    return Kernel(name or eri3_name(spec, packed=True), _params(True),
                  _packed_body(spec, False, per_cell))


def _packed_body(spec: Spec, lowstack: bool,
                 per_cell: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    la, lb, lP = spec.am
    cell = Var(CELL)
    return (
        Note(f"three-center DF integral, class (la, lb, lP) = ({la}, {lb}, "
             f"{lP}){', LOW-STACK (#606)' if lowstack else ''}, PACKED: "
             f"(mn|P) by the quartet ({la},{lb}|{lP},0)'s recurrence at L = "
             f"{la + lb + lP} (ao/eri3.py), each cell's whole block at its "
             "destination offset"),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )


def _params(packed: bool = False) -> tuple[Param, ...]:
    """The parameter block (module docstring), one for both forms; the
    packed one's (THE PACKED OUTPUT) when `packed`."""
    def f64(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.F64, array=True, shape=shape)

    def i32(n: str, shape: tuple[Expr, ...]) -> Param:
        return Param(n, Scalar.I32, array=True, shape=shape)

    ncell, nbra, naux_rec = Var("n_cell"), Var("n_bra_rec"), Var("n_aux_rec")
    nao, naux = Var("n_ao"), Var("n_aux")
    ncol = Var("n_col")
    cells = (PACKED_CELL_ARRAYS if packed else
             ("ab_rec0", "ab_rec1", "p_rec0", "p_rec1", "off_a", "off_b",
              "off_p"))
    return (
        Param("n_cell", Scalar.I32),
        Param("n_bra_rec", Scalar.I32),
        Param("n_aux_rec", Scalar.I32),
        Param("n_col" if packed else "n_ao", Scalar.I32),
        Param("n_aux", Scalar.I32),
        f64("zeta_ab", (nbra,)),
        f64("P", (nbra, Lit(3))),
        f64("XPA", (nbra, Lit(3))),
        f64("XPB", (nbra, Lit(3))),
        f64("s_ab", (nbra,)),
        f64("A", (naux_rec, Lit(3))),
        f64("alpha", (naux_rec,)),
        f64(S_AUX, (naux_rec,)),
        *(i32(n, (ncell,)) for n in cells),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=((_c(Fn.MUL, ncol, naux),) if packed
                     else (nao, nao, naux))),
    )


# ---------------------------------------------------------------------------
# The low-stack form (#606)
# ---------------------------------------------------------------------------

LOWSTACK_MIN_L = 10  # every class from L = la + lb + lP = 10 (#606)
# (pf|g)'s E blocks, 30 x 35 + 15 x 35 doubles (12,600 B): the largest the
# unrolled form keeps, its frame measured at 13,928 B on sm_86; (dd|g)'s
# 1,785 is over it, at 15,992 B (data/measured/df_lowstack.json, #606)
E_DOUBLES_MAX = 1575
R_ARRAY = "R"  # [r]^(0) of one (bra, aux) pair, hermite_targets(L) order
U_ARRAY = "U"  # SUM_tau [t+tau] E^q_tau of one aux row q, hermite_targets(Lab)
V_ARRAY = "Vab"  # one aux column's Cartesian bra block, [a nB + b] (#745)
T_ARRAY = "Tmu"  # one spherical mu's row of it, SUM_a c[mu][a] V[a, b] (#745)
AUX_TABLES = ("Cx", "Cy", "Cz")  # the aux record's single-center tables

LOWSTACK_NOTE = (
    "LOW-STACK (#606): per-axis E tables for the bra and the aux record in "
    "place of both E blocks, [r]^(0) into a local R as it forms, the "
    "contraction as loops through U[t] of one aux row, and the elements "
    "accumulated in place in the cell's own eri3 slots, mirrored at the end: "
    "no nA nB nP accumulators")
SPHERICAL_NOTE = (
    "SPHERICAL AUX (FelizJueves#743): eri3[dst + n_aux (a nB + b) + m], a b "
    "Cartesian, m = 0 .. 2 lP the real solid harmonic of order m - lP "
    "(-lP ..= lP, PySCF/libcint order; s and p the identity, p as x y z), "
    "each the Schlegel-Frisch harmonic normalized under the Gaussian-moment "
    "product and scaled by sqrt((2 lP - 1)!!): FreeBreakfast's "
    "spherical_transform_matrix block for one shell. n_aux and dst count "
    "spherical aux columns")
SPHERICAL_BRA_NOTE = (
    "FULLY SPHERICAL (FelizJueves#745): eri3[dst + n_aux (mu (2 lb + 1) + "
    "nu) + m], mu, nu and m all real solid harmonics in aux_harmonics' "
    "convention (s and p the identity); each column's Cartesian bra block "
    "is formed in a local Vab per primitive triple and transformed before "
    "the global add. n_col counts spherical bra functions")
_SPHERICAL_DIAGONAL = (
    "diagonal AO shell pair (diag): mu >= nu formed, each mirrored once "
    "into the full square spherical block")


def eri3_e_doubles(cls: Spec | tuple[int, int, int]) -> int:
    """The doubles the UNROLLED form declares for its two `E` blocks,
    `Ebra[nA nB][n_t(la+lb)]` and `Eket[nP][n_t(lP)]`: the quantity its
    measured frame tracks (THE LOW-STACK FORM)."""
    la, lb, lP = _spec(cls).am
    return (len(ef.block_rows(la, lb)) * len(hermite_targets(la + lb))
            + len(ef.cartesians(lP)) * len(hermite_targets(lP)))


def eri3_lowstack(cls: Spec | tuple[int, int, int]) -> bool:
    """THE SELECTION RULE (#606): True when `la + lb + lP >=
    LOWSTACK_MIN_L`, or when the unrolled form's `E` blocks are larger
    than (pf|g)'s (`eri3_e_doubles`, `E_DOUBLES_MAX`); there
    `build_eri3_kernel` emits the low-stack form by default. False below,
    where the unrolled text is unchanged."""
    spec = _spec(cls)
    return (sum(spec.am) >= LOWSTACK_MIN_L
            or eri3_e_doubles(spec) > E_DOUBLES_MAX)


def _iadd(x: Expr, y: Expr) -> Expr:
    return _c(Fn.ADD, x, y)


def _odd_double_factorial(n: int) -> int:
    """`n!!` of an odd `n >= -1`, `(-1)!! = 1`."""
    return math.prod(range(n, 0, -2))


def _monomial_gram(p: tuple[int, int, int], q: tuple[int, int, int]) -> int:
    """The Gaussian-moment product of two monomials, `PROD_d (p_d + q_d -
    1)!!`, zero when any sum is odd."""
    s = _plus(p, q)
    if any(x % 2 for x in s):
        return 0
    return math.prod(_odd_double_factorial(x - 1) for x in s)


def aux_harmonics(l: int) -> tuple[tuple[float, ...], ...]:  # noqa: E741
    """`c[m][tau]` of THE SPHERICAL-AUX OUTPUT: column `m` (`-l ..= l`,
    index `m + l`) of the real solid harmonics over the Cartesians `tau`
    of `cartesians(l)`, in the convention the module docstring states
    (FreeBreakfast's `solid_harmonic_block(l)` times `sqrt((2l-1)!!)`):
    the identity at s and p; above, the Schlegel-Frisch triple sum (IJQC
    54, 83 (1995)), each column normalized under `_monomial_gram`, then
    scaled. Host Python, a build-time table: the kernel spells it as
    literals."""
    if type(l) is not int or l < 0:
        raise ValueError(f"l = {l!r} is not a non-negative int")
    carts = ef.cartesians(l)
    n = len(carts)
    if l <= 1:
        return tuple(tuple(1.0 if i == j else 0.0 for i in range(n))
                     for j in range(n))
    comb, fact = math.comb, math.factorial
    scale = math.sqrt(float(_odd_double_factorial(2 * l - 1)))
    out = []
    for m in range(-l, l + 1):
        ma = abs(m)
        pre = (1.0 / (2.0 ** l * fact(l))) * math.sqrt(
            2.0 * fact(l + ma) * fact(l - ma) / (2.0 if m == 0 else 1.0))
        v0 = 0 if m >= 0 else 1  # the sine branch's half-integer offset
        col = [0.0] * n
        for t in range((l - ma) // 2 + 1):
            for u in range(t + 1):
                for two_v in range(v0, ma + 1, 2):
                    p = (2 * t + ma - 2 * u - two_v, 2 * u + two_v,
                         l - 2 * t - ma)
                    if min(p) < 0:
                        continue
                    sign = -1.0 if (t + (two_v - v0) // 2) % 2 else 1.0
                    col[carts.index(p)] += pre * (
                        sign * 0.25 ** t * comb(l, t) * comb(l - t, ma + t)
                        * comb(t, u) * comb(ma, two_v))
        norm2 = sum(col[i] * col[j] * _monomial_gram(carts[i], carts[j])
                    for i in range(n) for j in range(n)
                    if col[i] != 0.0 and col[j] != 0.0)
        inv = 1.0 / math.sqrt(norm2)
        out.append(tuple(c * inv * scale for c in col))
    return tuple(out)


def _shifted(t: tuple[Expr, Expr, Expr], tau: tuple[int, int, int]
             ) -> tuple[Expr, Expr, Expr]:
    """`t + tau` for a literal `tau`, a zero component left as it is."""
    return tuple(t[d] if tau[d] == 0 else _iadd(t[d], Lit(tau[d]))
                 for d in range(3))


def _scaled_w(lP: int) -> Expr:
    """`s_ij s_k (2 alpha)^{-lP}`: `sb`, the record's `s_aux`, then `lP`
    factors of the local `hq = 0.5 / alpha`, a left fold."""
    w: Expr = _c(Fn.MUL, Var("sb"), Index(S_AUX, (Var(KET),)))
    for _ in range(lP):
        w = _c(Fn.MUL, w, Var("hq"))
    return w


def _build_lowstack(spec: Spec, name: str | None,
                    packed: bool = False, spherical: bool = False,
                    spherical_bra: bool = False) -> Kernel:
    """`build_eri3_kernel`'s kernel with no `E` block and no accumulators
    (module docstring, THE LOW-STACK FORM): the same parameters, launch and
    output. `packed`: the packed output's (THE PACKED OUTPUT), each element
    accumulated in its packed slot by the same operations. `spherical`
    (packed only): the aux index in solid harmonics (THE SPHERICAL-AUX
    OUTPUT), the aux record's tables replaced by `aux_harmonics(lP)`.
    `spherical_bra` (spherical only): both bra indices in solid harmonics
    too (THE FULLY SPHERICAL OUTPUT), each column's Cartesian block formed
    in a local `V` and transformed into the output's spherical slots."""
    la, lb, lP = spec.am
    Lab, L = la + lb, la + lb + lP
    dag, order = eri3_plan(spec)
    fB, fP = ef.axis_tables_fragment(la, lb), ef.center_tables_fragment(lP)
    nA, nB = (len(ef.cartesians(x)) for x in (la, lb))
    nP = 2 * lP + 1 if spherical else len(ef.cartesians(lP))
    # the output block's bra extents: spherical when `spherical_bra`, where
    # s and p are the identity, so only a class with a d or higher bra
    # shell transforms (THE FULLY SPHERICAL OUTPUT)
    oA, oB = (2 * la + 1, 2 * lb + 1) if spherical_bra else (nA, nB)
    transform = spherical_bra and max(la, lb) >= 2
    odd = lP % 2 == 1
    cell, ib, ik = Var(CELL), Var(BRA), Var(KET)
    targets = live_targets(la, lb, lP)

    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    tower = store_targets(_live((*prelude, *recurrence_body(
        dag, order, fundamental)), {node_name((r, 0)) for r in targets}),
        targets, L, R_ARRAY)

    def out_at(i: Expr, j: Expr, k: Expr, mirror: bool = False) -> Index:
        if packed:  # block (i, j), or (j, i) for the mirror, at `od`
            r, c = (j, i) if mirror else (i, j)
            return _packed_at(_iadd(_c(Fn.MUL, r, Lit(oB)), c), k)
        m, n = _iadd(Var("oa"), i), _iadd(Var("ob"), j)
        return Index(OUT, (n, m, _iadd(Var("op"), k)) if mirror
                     else (m, n, _iadd(Var("op"), k)))

    # --- every (a, b) element of one aux column `rq`, from U[t] -------------
    def elements(rq: Expr) -> For:
        def row(a: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            def elem(b: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
                at = out_at(Var("ra"), Var("rb"), rq)
                v_sum = ef.rolled_e_sum(la, lb, a, b,
                                        lambda h: Index(U_ARRAY, (h,)), "v")
                update: tuple[Stmt, ...]
                if transform:  # the column's Cartesian block, local
                    update = (*v_sum, Assign(_v_at(Var("ra"), Var("rb")),
                                             Var("v")))
                    if la == lb:  # a diagonal pair's mirror, within V
                        update = (*update, If(_c(Fn.LT, Lit(0), Var("dg")), (
                            Assign(_v_at(Var("rb"), Var("ra")), Var("v")),)))
                else:
                    update = (*v_sum, Assign(at, _c(
                        Fn.SUB if odd else Fn.ADD, at,
                        _c(Fn.MUL, Var("w"), Var("v")))))
                head = (Decl("rb", Scalar.I32, init=ef.cartesian_index("b")),)
                if la != lb:
                    return (*head, *update)
                return (*head, If(_iadd(Var("differ"), _c(
                    Fn.LT, Var("rb"), _iadd(Var("ra"), Lit(1)))), update))
            return (Decl("ra", Scalar.I32, init=ef.cartesian_index("a")),
                    ef.cartesian_loops(lb, ("bx", "by", "bz"), elem, tag="b"))
        return ef.cartesian_loops(la, ("ax", "ay", "az"), row, tag="a")

    def _v_at(a: Expr, b: Expr) -> Index:
        return Index(V_ARRAY, (_iadd(_c(Fn.MUL, a, Lit(nB)), b),))

    element_note = Note(
        f"per element (a, b, q): eri3 {'-' if odd else '+'}= s_ij s_k SUM_t "
        "E^{ab}_t U[t], rolled over the per-axis tables; "
        f"(-1)^|tau| = (-1)^{lP} on every live tau"
        + ("; a diagonal pair forms a >= b only" if la == lb else ""))
    if transform:
        element_note = Note(
            "per Cartesian (a, b): Vab[a nB + b] = SUM_t E^{ab}_t U[t], "
            "rolled over the per-axis tables, LOCAL (no global add)"
            + ("; a diagonal pair forms a >= b only, mirrored within Vab"
               if la == lb else ""))

    # --- one column's V onto the spherical bra slots, unrolled -------------
    def to_spherical(m: int) -> tuple[Stmt, ...]:
        ca, cb = aux_harmonics(la), aux_harmonics(lb)
        out: list[Stmt] = [Note(
            f"the bra onto solid harmonics: for each mu, Tmu[b] = SUM_a "
            "c[mu][a] Vab[a nB + b]; then out[mu, nu, m] "
            f"{'-' if odd else '+'}= w SUM_b c[nu][b] Tmu[b] "
            "(aux_harmonics, nonzero terms only)"
            + ("; a diagonal pair accumulates mu >= nu only"
               if la == lb else ""))]
        tb = Var("tb")
        for mu in range(oA):
            t_row: Expr | None = None
            for a, c in enumerate(ca[mu]):
                if c != 0.0:
                    term = _c(Fn.MUL, Lit(c), _v_at(Lit(a), tb))
                    t_row = term if t_row is None else _iadd(t_row, term)
            assert t_row is not None  # every harmonic has a term
            out.append(For(tb.name, Lit(0), Lit(nB), Lit(1), (
                Assign(Index(T_ARRAY, (tb,)), t_row),)))
            for nu in range(oB):
                sv: Expr | None = None
                for b, c in enumerate(cb[nu]):
                    if c != 0.0:
                        term = _c(Fn.MUL, Lit(c), Index(T_ARRAY, (Lit(b),)))
                        sv = term if sv is None else _iadd(sv, term)
                assert sv is not None
                at = out_at(Lit(mu), Lit(nu), Lit(m))
                upd: Stmt = Assign(at, _c(Fn.SUB if odd else Fn.ADD, at,
                                          _c(Fn.MUL, Var("w"), sv)))
                if la == lb and nu > mu:
                    upd = If(Var("differ"), (upd,))
                out.append(upd)
        return tuple(out)

    # --- one Cartesian aux row q: U[t], then every (a, b) element of it ------
    def per_q(q: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
        def u_of(t: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            def r_at(tau: tuple[Expr, Expr, Expr]) -> Expr:
                return Index(R_ARRAY, (ef.hermite_index(tuple(
                    _iadd(t[d], tau[d]) for d in range(3))),))
            return (*ef.rolled_center_sum(lP, q, r_at, "u", tag="c",
                                          arrays=AUX_TABLES),
                    Assign(Index(U_ARRAY, (ef.hermite_index(t),)), Var("u")))

        return (
            Decl("rq", Scalar.I32, init=ef.cartesian_index("q")),
            Note("U[t] = SUM_tau [t+tau]^(0) E^q_tau over the live tau of "
                 "this aux row, for every |t| <= la + lb: once per (t, q)"),
            ef.hermite_loops(Lab, u_of, tag="h"),
            element_note,
            elements(Var("rq")),
        )

    # --- one solid-harmonic aux column m: U[t] from c[m], unrolled ----------
    def per_m(m: int, col: tuple[float, ...]) -> tuple[Stmt, ...]:
        terms = [(tau, c) for tau, c in zip(ef.cartesians(lP), col,
                                             strict=True) if c != 0.0]

        def u_of(t: tuple[Expr, Expr, Expr]) -> tuple[Stmt, ...]:
            u: Expr | None = None
            for tau, c in terms:
                term = _c(Fn.MUL, Lit(c), Index(R_ARRAY, (
                    ef.hermite_index(_shifted(t, tau)),)))
                u = term if u is None else _iadd(u, term)
            assert u is not None  # every harmonic has a term
            return (Assign(Index(U_ARRAY, (ef.hermite_index(t),)), u),)

        return (
            Note(f"aux column m = {m - lP}: U[t] = SUM_tau c[m][tau] "
                 f"[t+tau]^(0) over its {len(terms)} nonzero |tau| = {lP} "
                 "(aux_harmonics), for every |t| <= la + lb"),
            ef.hermite_loops(Lab, u_of, tag="h"),
            element_note,
            elements(Lit(m)),
            *(to_spherical(m) if transform else ()),
        )

    if spherical:
        aux_rows: tuple[Stmt, ...] = (
            Note("the aux record in solid harmonics (THE SPHERICAL-AUX "
                 "OUTPUT): no single-center tables; E^m_tau = (2 alpha)^-lP "
                 "c[m][tau] at |tau| = lP, the factor folded into w"),
            *((Decl("hq", Scalar.F64,
                    init=_c(Fn.DIV, Lit(0.5), Var("aq"))),) if lP else ()),
            Decl("w", Scalar.F64, init=_scaled_w(lP)),
            *(s for m, col in enumerate(aux_harmonics(lP))
              for s in per_m(m, col)),
        )
    else:
        aux_binds: dict[str, Expr] = {n: Var(n) for n in AUX_TABLES}
        if ef.HALF_INV_ZETA in fP.binds:
            aux_binds[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), Var("aq"))
        aux_rows = (
            Note("the aux record's single-center E, per axis: "
                 "e_fragment.center_tables_fragment, inlined"),
            *(Decl(n, Scalar.F64, (Lit(ef.center_table_extent(lP)),))
              for n in AUX_TABLES),
            Inline(fP, aux_binds),
            Decl("w", Scalar.F64, init=_c(Fn.MUL, Var("sb"),
                                          Index(S_AUX, (ik,)))),
            ef.cartesian_loops(lP, ("qx", "qy", "qz"), per_q, tag="q"),
        )
    ket_body: tuple[Stmt, ...] = (
        Decl("aq", Scalar.F64, init=Index("alpha", (ik,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                      Index("A", (ik, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        Note("theta^2 = p alpha / (p + alpha): the aux Gaussian is a "
             "Hermite pair at zeta = alpha, Q = A"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, _c(
            Fn.DIV, _c(Fn.MUL, Var("ze"), Var("aq")),
            _c(Fn.ADD, Var("ze"), Var("aq"))))),
        Note("[t+tau]^(0) by the plan's DAG for the quartet "
             f"({la},{lb}|{lP},0), its dead targets dropped, each live one "
             "stored into R as it forms"),
        *tower,
        *aux_rows,
    )

    bra_binds: dict[str, Expr] = {ef.ZETA: Var("ze"),
                                  **{n: Var(n) for n in ef.AXIS_ARRAYS}}
    if ef.HALF_INV_ZETA in fB.binds:
        bra_binds[ef.HALF_INV_ZETA] = _c(Fn.DIV, Lit(0.5), Var("ze"))
    for d in range(3):
        bra_binds[Factor.pa(d).name_in_kernel] = Index("XPA", (ib, Lit(d)))
        bra_binds[Factor.pb(d).name_in_kernel] = Index("XPB", (ib, Lit(d)))
    bra_body: tuple[Stmt, ...] = (
        Decl("ze", Scalar.F64, init=Index("zeta_ab", (ib,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index("P", (ib, Lit(d))))
          for d in range(3)),
        Decl("sb", Scalar.F64, init=Index("s_ab", (ib,))),
        Note("the bra record's per-axis E^{ij}_t "
             "(e_fragment.axis_tables_fragment); no E^{ab}_t block"),
        *(Decl(n, Scalar.F64, (Lit(ef.axis_table_extent(la, lb)),))
          for n in ef.AXIS_ARRAYS),
        Inline(fB, bra_binds),
        Note("the aux records of the cell"),
        For(KET, Var("p0"), Var("p1"), Lit(1), ket_body),
    )

    # --- zero the cell's slots first, mirror them last ---------------------
    i, j, k = Var("i"), Var("j"), Var("k")

    def nest(ni: Expr, nj: Expr, body: tuple[Stmt, ...]) -> For:
        return For("i", Lit(0), ni, Lit(1), (For("j", Lit(0), nj, Lit(1), (
            For("k", Lit(0), Lit(nP), Lit(1), body),)),))

    zero = nest(Lit(oA), Lit(oB), (Assign(out_at(i, j, k), Lit(0.0)),))
    if packed:
        cell_body: tuple[Stmt, ...] = (
            *_cell_ranges(cell),
            Decl("od", Scalar.I32, init=Index("dst", (cell,))),
            *((Decl("dg", Scalar.I32, init=Index("diag", (cell,))),
               Decl("differ", Scalar.I32, init=_c(Fn.LT, Var("dg"),
                                                  Lit(1))))
              if la == lb else ()),
            Note(LOWSTACK_NOTE.replace(
                "own eri3 slots, mirrored at the end",
                "own packed slots, a diagonal pair's mirrored at the end")),
            Note("the cell's own elements, zeroed: no other cell writes them"),
            zero,
            *(Decl(n, Scalar.F64, (Lit(len(hermite_targets(x))),))
              for n, x in ((R_ARRAY, L), (U_ARRAY, Lab))),
            *((Decl(V_ARRAY, Scalar.F64, (Lit(nA * nB),)),
               Decl(T_ARRAY, Scalar.F64, (Lit(nB),))) if transform else ()),
            Note("the bra records of the cell's shell pair, serially"),
            For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
            *((Note(_SPHERICAL_DIAGONAL if transform
                    else _PACKED_DIAGONAL), If(
                _c(Fn.LT, Lit(0), Var("dg")), (nest(Lit(oA), i, (Assign(
                    out_at(i, j, k, mirror=True), out_at(i, j, k)),)),)))
              if la == lb else ()),
        )
        body = _packed_body(spec, True, cell_body)
        if spherical:
            body = (Note(SPHERICAL_NOTE), *body)
        if spherical_bra:
            body = (Note(SPHERICAL_BRA_NOTE), *body)
        return Kernel(name or eri3_name(spec, packed=True,
                                        spherical_aux=spherical,
                                        spherical_bra=spherical_bra),
                      _params(True), body)
    off_diag = (nest(Lit(nA), Lit(nB), (
        Assign(out_at(i, j, k, mirror=True), out_at(i, j, k)),)),)
    scatter: Stmt
    if la == lb:
        diag = (Note(_DIAGONAL), nest(Lit(nA), i, (
            Assign(out_at(i, j, k, mirror=True), out_at(i, j, k)),)))
        scatter = If(Var("differ"), off_diag, diag)
    else:
        scatter = If(Var("differ"), off_diag)

    per_cell: tuple[Stmt, ...] = (
        Decl("b0", Scalar.I32, init=Index("ab_rec0", (cell,))),
        Decl("b1", Scalar.I32, init=Index("ab_rec1", (cell,))),
        Decl("p0", Scalar.I32, init=Index("p_rec0", (cell,))),
        Decl("p1", Scalar.I32, init=Index("p_rec1", (cell,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (cell,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (cell,))),
        Decl("op", Scalar.I32, init=Index("off_p", (cell,))),
        Decl("differ", Scalar.I32, init=_c(
            Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
            _c(Fn.LT, Var("ob"), Var("oa")))),
        Note(LOWSTACK_NOTE),
        Note("the cell's own elements, zeroed: no other cell writes them"),
        zero,
        *(Decl(n, Scalar.F64, (Lit(len(hermite_targets(x))),))
          for n, x in ((R_ARRAY, L), (U_ARRAY, Lab))),
        Note("the bra records of the cell's shell pair, serially"),
        For(BRA, Var("b0"), Var("b1"), Lit(1), bra_body),
        Note("the (m, n) mirror, a plain copy: each element belongs to one "
             "cell, so no atomics"),
        scatter,
    )
    body: tuple[Stmt, ...] = (
        Note(f"three-center DF integral, class (la, lb, lP) = ({la}, {lb}, "
             f"{lP}), LOW-STACK (#606): (mn|P) by the quartet "
             f"({la},{lb}|{lP},0)'s recurrence at L = {L} (ao/eri3.py)"),
        Note("launch: one thread per (AO shell pair, aux shell) cell"),
        Decl(CELL, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, cell, Var("n_cell")), per_cell),
    )
    return Kernel(name or eri3_name(spec), _params(), body)


# ---------------------------------------------------------------------------
# The bra transform of a spherical-aux packed output (#747)
# ---------------------------------------------------------------------------

BRA_SRC = "src"  # the speri3_ output read: Cartesian bra, spherical aux
BRA_PAIR_ARRAYS = ("src_col", "dst_col")  # per pair: its first column, each


def bra_transform_name(la: int, lb: int) -> str:
    """`sbra3_{la}{lb}`: the bra transform of the `(la, lb)` pairs."""
    sep = "_" if max(la, lb) > 9 else ""
    return f"sbra3_{la}{sep}{lb}"


def build_bra_transform_kernel(la: int, lb: int, *,
                               name: str | None = None) -> Kernel:
    """THE BRA TRANSFORM (#747): `speri3_`'s packed output, Cartesian bra and
    spherical aux, onto spherical bra slots -- the pass FreeBreakfast runs
    after `speri3_`, the other leg of #747's clock against `ssperi3_`.

    One thread per (pair, aux column `q`), `q` fastest, so a warp reads and
    writes consecutive `q`: the pair's `nA nB` Cartesian columns read once
    into a local `Vab` (`src[src_col n_aux + n_aux (a nB + b) + q]`), then,
    as `ssperi3_`, one `mu` at a time a row `Tmu[b] = SUM_a c^{la}[mu][a]
    Vab[a nB + b]` and `out[dst_col n_aux + n_aux (mu (2lb+1) + nu) + q] =
    SUM_b c^{lb}[nu][b] Tmu[b]` (`aux_harmonics`, literals, nonzero terms
    only), a plain store. Each element is read and written once: the
    traffic floor of any transform of that output, a GEMM's included. A
    diagonal pair's full square block in, its full square out.

        params  n_pair n_aux n_src_col n_dst_col (I32)
                src_col dst_col [n_pair] (I32)
                src[n_src_col n_aux]  out[n_dst_col n_aux] (OUT)
    """
    for x in (la, lb):
        if type(x) is not int or x < 0:
            raise ValueError(f"l = {x!r} is not a non-negative int")
    nA, nB = len(ef.cartesians(la)), len(ef.cartesians(lb))
    sA, sB = 2 * la + 1, 2 * lb + 1
    ca, cb = aux_harmonics(la), aux_harmonics(lb)
    gid, pr, q = Var("gid"), Var("pr"), Var("q")
    tb, i = Var("tb"), Var("i")

    def src_at(col: Expr) -> Index:
        return Index(BRA_SRC, (_iadd(Var("s0"), _c(Fn.MUL, Var("n_aux"),
                                                   col)),))

    def v_at(a: int, b: Expr) -> Index:
        return Index(V_ARRAY, (_iadd(Lit(a * nB), b),))

    rows: list[Stmt] = []
    for mu in range(sA):
        t_row: Expr | None = None
        for a, c in enumerate(ca[mu]):
            if c != 0.0:
                term = _c(Fn.MUL, Lit(c), v_at(a, tb))
                t_row = term if t_row is None else _iadd(t_row, term)
        assert t_row is not None
        rows.append(For(tb.name, Lit(0), Lit(nB), Lit(1), (
            Assign(Index(T_ARRAY, (tb,)), t_row),)))
        for nu in range(sB):
            sv: Expr | None = None
            for b, c in enumerate(cb[nu]):
                if c != 0.0:
                    term = _c(Fn.MUL, Lit(c), Index(T_ARRAY, (Lit(b),)))
                    sv = term if sv is None else _iadd(sv, term)
            assert sv is not None
            rows.append(Assign(Index(OUT, (_iadd(Var("d0"), _c(
                Fn.MUL, Var("n_aux"), Lit(mu * sB + nu))),)), sv))
    per: tuple[Stmt, ...] = (
        Decl("pr", Scalar.I32, init=_c(Fn.DIV, gid, Var("n_aux"))),
        Decl("q", Scalar.I32, init=_c(Fn.SUB, gid,
                                      _c(Fn.MUL, pr, Var("n_aux")))),
        Decl("s0", Scalar.I32, init=_iadd(_c(
            Fn.MUL, Index("src_col", (pr,)), Var("n_aux")), q)),
        Decl("d0", Scalar.I32, init=_iadd(_c(
            Fn.MUL, Index("dst_col", (pr,)), Var("n_aux")), q)),
        Note("the pair's Cartesian block at this aux column, read once"),
        Decl(V_ARRAY, Scalar.F64, (Lit(nA * nB),)),
        Decl(T_ARRAY, Scalar.F64, (Lit(nB),)),
        For(i.name, Lit(0), Lit(nA * nB), Lit(1), (
            Assign(Index(V_ARRAY, (i,)), src_at(i)),)),
        Note("onto solid harmonics, one mu at a time through Tmu "
             "(aux_harmonics, nonzero terms only)"),
        *rows,
    )
    ncol_s, ncol_d, naux = Var("n_src_col"), Var("n_dst_col"), Var("n_aux")
    params = (
        Param("n_pair", Scalar.I32), Param("n_aux", Scalar.I32),
        Param("n_src_col", Scalar.I32), Param("n_dst_col", Scalar.I32),
        *(Param(n, Scalar.I32, array=True, shape=(Var("n_pair"),))
          for n in BRA_PAIR_ARRAYS),
        Param(BRA_SRC, Scalar.F64, array=True,
              shape=(_c(Fn.MUL, ncol_s, naux),)),
        Param(OUT, Scalar.F64, array=True, intent=Intent.OUT,
              shape=(_c(Fn.MUL, ncol_d, naux),)),
    )
    body: tuple[Stmt, ...] = (
        Note(f"bra transform ({la}, {lb}) (#747): a spherical-aux packed "
             "output's Cartesian bra onto solid harmonics, one thread per "
             "(pair, aux column), the aux column fastest"),
        Decl("gid", Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, gid, _c(Fn.MUL, Var("n_pair"), naux)), per),
    )
    return Kernel(name or bra_transform_name(la, lb), params, body)
