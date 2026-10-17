"""The nuclear-attraction kernel, `Target.V`, built as a `Kernel` (#239).

    build_v_kernel(spec | (la, lb)) -> Kernel     one per pair class
    v_spec(la, lb)                  -> Spec       the class's key
    v_plan(spec | (la, lb))         -> (dag, order)  the plan layer's answer
    launch_shape(n_rec, threads)    -> (grid, block)

THE FIRST KERNEL OUTSIDE THE J CHAIN TO USE THE PLAN LAYER. `V_ab` is the J
engine with the ket replaced by a unit point charge (milestone 11's design
pin; `ao/potential.py`'s module docstring): per primitive-pair record `k` of
one pair class `(la, lb)`,

    theta = sqrt(zeta)                         a local, formed once
    per nucleus C (serially, `ir.For`):
        PQ = P - C                             a local per iteration
        [t]^(0) for |t| <= la + lb             recurrence_body, UNCHANGED
        u[r] = SUM_t E[r, t] [t]^(0)           the reachable columns only
        acc[r] = acc[r] - Z_C u[r]
    v[r] = s[k] acc[r]

then the scatter, `kernel_jab`'s rule unchanged (as in `kernel_s`).

THE RECURRENCE IS IMPORTED, NOT REWRITTEN. The `[t]^(0)` statements are
`recurrence.fundamental_prelude` and `recurrence.recurrence_body` over the
one DAG `plan.pipeline.plan(v_spec(la, lb))` returns, in `schedule(dag)`'s
order (`v_plan`). `dag.build` reads only `spec.total_am` and
`spec.n_batches`, so MILD, the sieve, the schedule and the cost model apply
at Hermite order `la + lb` with no edit to `plan/` -- the plan layer's first
non-J consumer. `n_batches = 1` is the only count `Spec` admits for `V`
(milestone 11's decision, `spec._PAIR_BATCH_REFUSAL`); the builder adds no
refusal of its own and relies on the key's.

`E` IS `e_fragment.block_fragment(la, lb)`, INLINED into a local array
`E[n_rows, n_cols]` -- the first consumer that reads EVERY column. Its
structurally zero entries (`t_d > a_d + b_d`) are never written, so they are
never read either: `u[r]` sums only the reachable columns of row `r`,
left to right in `hermite_order` order, starting from the first term (not
from a `0.0`).

NO PREFACTOR. `(pi/zeta)^{3/2}` is inside the host's `s` (`ao.pairs.dress`,
`ao.j.epair`'s `s`), and `(2/sqrt(pi)) sqrt(zeta)` is inside `[0]^(m)` (the
prelude's `W0`); their product is `2 pi / zeta`, the one-electron prefactor,
so the kernel forms neither. No `K_AB` is reinstated: the `E` seed carries it
per axis. The attractive sign enters as `acc - Z u`, a subtraction, which is
`acc + (-Z) u` exactly.

THE PARAMETER BLOCK:

    params  n_rec n_ao n_nuc (I32)
            zeta_ab  s_ab  XP YP ZP  XPA YPA ZPA  XPB YPB ZPB
                                                      [n_rec]  per record
            off_a  off_b  [n_rec]  (I32)
            Z[n_nuc]  Cnuc[n_nuc, 3]                  per nucleus
            V[n_ao, n_ao]  (OUT, read and AtomicAdd-ed)

`kernel_s`'s block, with three changes, each for a reason:

- `P` is added, as `XP YP ZP`: `PQ = P - C` needs the product centre, and
  `XPA` / `XPB` without `A` cannot give it.
- `s` is spelled `s_ab`: `recurrence.fundamental_prelude` declares a local
  `s` (`-2 theta^2`) inside the nucleus loop, and a local shadowing a host
  array of the same name is the hazard `zeta_ab` already avoids
  (`kernel_jab`'s hygiene note). The value is the same host scalar.
- `n_nuc`, `Z` and `Cnuc` are the nuclei, `(Z_C, C)` in the order the host
  gives them. `V` is ZEROED BY THE CALLER before the first `V` launch.

ONE THREAD PER RECORD, `rec = THREAD_X + BLOCK_X * BLOCK_DIM_X`, guarded by
`rec < n_rec`; the nuclei are SERIAL inside the record (the milestone's
mapping pin: a warp per record with the nuclei strip-mined is a follow-up).

THE WIDE BOYS (#266). The kernel sets `Kernel.wide_boys`, so its `fj_boys`
is `emit/_c.py`'s `BOYS_DEVICE_WIDE`: the Kummer series below T = 125 and
the asymptote from there, because `T = zeta |P - C|^2` reaches ~1.25e5 on
the tracked fixtures and the narrow series overflows near 709. Every J-chain
unit keeps the narrow text (its `kernels.json` pins); limits.json records
why there are two.

ROUND-OFF, NOT BITWISE, against `ao.potential.potential_block`. The host
recurses `[t]^(0)` along `reference.pq.one_centre`'s fixed rule; this kernel
runs the plan's MILD directions (and the sieve's re-parenting) in the
schedule's order, which is the point of having both. It also applies `s`
once per row after the nucleus sum rather than once per `E` entry. So the
two agree to rounding, and the test states its bound (test/test_kernel_v.py);
contrast `kernel_s` and `kernel_jab`, which are bitwise.

A builder: vendor-neutral IR. It imports neither `ao/` (its witness) nor the
interpreter nor `reference/`.
"""

from __future__ import annotations

from felizjueves import e_fragment as ef
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
    Inline,
    Intent,
    Kernel,
    Lit,
    Note,
    Param,
    Scalar,
    Stmt,
    Var,
)
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.schedule import Order, schedule
from felizjueves.plan.targets import hermite_targets
from felizjueves.prim_pair import AXES, bundled
from felizjueves.recurrence import (
    PQ_PARAMS,
    THETA,
    fundamental_prelude,
    node_name,
    recurrence_body,
)
from felizjueves.spec import Operator, Spec, Target

__all__ = [
    "ATOMIC_REASON",
    "NUC",
    "REC",
    "ZEROED_V",
    "build_v_kernel",
    "launch_shape",
    "v_plan",
    "v_spec",
]

REC = "rec"  # the record index: one thread per primitive-pair record

# The record arrays arrive in one `PrimPairSoA` (prim_pair.py); the field
# `s` is read as `s_ab` (the parameter block's note on the name).
BUNDLED = bundled(*(f"{a}P{c}" for c in ("", "A", "B") for a in AXES),
                  "off_a", "off_b", zeta="zeta_ab", s="s_ab")
NUC = "c"  # the nucleus index, serial inside the record

ATOMIC_REASON = (
    "AtomicAdd: several primitive records of one shell pair, and both mirror "
    "positions, add into the same V elements (kernel_jab's scatter rule)")

ZEROED_V = (
    "PRECONDITION: V is zeroed by the caller before the first V launch of a "
    "build; this kernel only adds")

_DIAGONAL = (
    "diagonal shell pair (off_a == off_b): the lower triangle i >= j only, "
    "each mirrored once -- kernel_jab's diagonal rule, so the diagonal is not "
    "double-counted")


def v_spec(la: int, lb: int) -> Spec:
    """The pair class `(la, lb)` as a generation key, `Target.V`."""
    return Spec(Operator.NUCLEAR, 0, (la, lb), Target.V)


def _spec(cls: Spec | tuple[int, int]) -> Spec:
    spec = cls if isinstance(cls, Spec) else v_spec(*cls)
    if spec.target is not Target.V:
        raise ValueError(f"{spec.key()} is not a Target.V key")
    if spec.levers:
        raise ValueError(f"{spec.key()}: the V kernel takes no levers")
    return spec


def v_plan(cls: Spec | tuple[int, int]) -> tuple[BatchDag, Order]:
    """The plan layer's answer for the class: `pipeline.plan(spec)`'s one
    DAG and `schedule(dag)`. Nothing here re-derives either."""
    spec = _spec(cls)
    (dag,) = plan(spec).dags
    return dag, schedule(dag)


def launch_shape(n_rec: int, threads: int = 32
                 ) -> tuple[int, tuple[int, int]]:
    """`(grid, (threads, 1))`: one thread per record, `ceil(n_rec/threads)`
    blocks. No `WarpReduce`, so a partial last block is legal."""
    if type(n_rec) is not int or n_rec < 1:
        raise ValueError(f"n_rec = {n_rec!r} is not a positive int")
    if type(threads) is not int or threads < 1:
        raise ValueError(f"threads = {threads!r} is not a positive int")
    return -(-n_rec // threads), (threads, 1)


def _c(fn: Fn, *args: Expr) -> Call:
    return Call(fn, args)


def _f64(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.F64, array=True, shape=shape)


def _i32(name: str, shape: tuple[Expr, ...]) -> Param:
    return Param(name, Scalar.I32, array=True, shape=shape)


def build_v_kernel(cls: Spec | tuple[int, int], *,
                   name: str | None = None) -> Kernel:
    """The nuclear-attraction kernel of one pair class.

    `cls` is a `Target.V` `Spec` or `(la, lb)`. See the module docstring.
    """
    spec = _spec(cls)
    la, lb = spec.am
    dag, order = v_plan(spec)
    frag = ef.block_fragment(la, lb)
    rows = ef.block_rows(la, lb)
    ts = hermite_targets(la + lb)
    nb = len(ef.cartesians(lb))
    rec, nuc = Var(REC), Var(NUC)

    # --- the record's inputs, and E^{ab}_t from the fragment ---------------
    binds: dict[str, Expr] = {ef.ZETA: Var(ef.ZETA),
                              ef.E_ARRAY: Var(ef.E_ARRAY)}
    head: list[Stmt] = [
        Decl(ef.ZETA, Scalar.F64, init=Index("zeta_ab", (rec,))),
    ]
    if ef.HALF_INV_ZETA in frag.binds:
        head.append(Decl(ef.HALF_INV_ZETA, Scalar.F64,
                         init=_c(Fn.DIV, Lit(0.5), Var(ef.ZETA))))
        binds[ef.HALF_INV_ZETA] = Var(ef.HALF_INV_ZETA)
    for d, a in enumerate(AXES):
        binds[Factor.pa(d).name_in_kernel] = Index(f"{a}PA", (rec,))
        binds[Factor.pb(d).name_in_kernel] = Index(f"{a}PB", (rec,))
    head += [
        Decl("sk", Scalar.F64, init=Index("s_ab", (rec,))),
        Decl("oa", Scalar.I32, init=Index("off_a", (rec,))),
        Decl("ob", Scalar.I32, init=Index("off_b", (rec,))),
        *(Decl(f"P{d}", Scalar.F64, init=Index(f"{a}P", (rec,)))
          for d, a in enumerate(AXES)),
        Note("the undressed block E^{ab}_t of this record, from the fragment: "
             "rows cartesians(la) x cartesians(lb), columns hermite_order"
             "(la + lb); a structurally zero entry is never written nor read"),
        Decl(ef.E_ARRAY, Scalar.F64, (Lit(len(rows)), Lit(len(ts)))),
        Inline(frag, binds),
        Note("theta = sqrt(zeta): the eta -> inf limit of theta^2 = "
             "zeta eta / (zeta + eta), a unit point-charge ket"),
        Decl(THETA, Scalar.F64, init=_c(Fn.SQRT, Var(ef.ZETA))),
        *(Decl(f"acc{r}", Scalar.F64, init=Lit(0.0)) for r in range(len(rows))),
    ]

    # --- per nucleus: PQ, [t]^(0) by the plan's recurrence, the contraction -
    prelude, fundamental = fundamental_prelude(
        [m for (r, m) in order if not any(r)])
    per_nuc: list[Stmt] = [
        Decl("Zc", Scalar.F64, init=Index("Z", (nuc,))),
        *(Decl(n, Scalar.F64, init=_c(Fn.SUB, Var(f"P{d}"),
                                      Index("Cnuc", (nuc, Lit(d)))))
          for d, n in enumerate(PQ_PARAMS)),
        *prelude,
        *recurrence_body(dag, order, fundamental),
        Note("u_r = SUM_t E[r, t] [t]^(0) over the reachable columns, and "
             "acc_r -= Z_C u_r: V's attractive sign, applied here only"),
    ]
    for r, (a, b) in enumerate(rows):
        terms = [_c(Fn.MUL, Index(ef.E_ARRAY, (Lit(r), Lit(col))),
                    Var(node_name((t, 0))))
                 for col, t in enumerate(ts)
                 if all(t[d] <= a[d] + b[d] for d in range(3))]
        u: Expr = terms[0]
        for term in terms[1:]:
            u = _c(Fn.ADD, u, term)
        per_nuc += [
            Decl(f"u{r}", Scalar.F64, init=u),
            Assign(Var(f"acc{r}"), _c(Fn.SUB, Var(f"acc{r}"),
                                      _c(Fn.MUL, Var("Zc"), Var(f"u{r}")))),
        ]
    nuclei = For(NUC, Lit(0), Var("n_nuc"), Lit(1), tuple(per_nuc))

    # --- v per Cartesian pair: the host's one scalar times the nucleus sum --
    values: list[Stmt] = [
        Note("V_ab = s acc: s = c_a c_b (pi/zeta)^{3/2} is the host's "
             "(ao.pairs.dress), and with W0's (2/sqrt(pi)) theta it makes "
             "2 pi / zeta; no prefactor and no K_AB is formed here"),
    ]
    for r in range(len(rows)):
        values.append(Decl(f"v{r}", Scalar.F64, init=_c(
            Fn.MUL, Var("sk"), Var(f"acc{r}"))))

    # --- the scatter: kernel_jab's rule ------------------------------------
    def at(row: Expr, col: Expr) -> Index:
        return Index("V", (row, col))

    def plus(base: str, k: int) -> Expr:
        return _c(Fn.ADD, Var(base), Lit(k))

    off_diag: list[Stmt] = [Note(ATOMIC_REASON)]
    for i in range(len(ef.cartesians(la))):
        for j in range(nb):
            vr = Var(f"v{i * nb + j}")
            off_diag += [AtomicAdd(at(plus("oa", i), plus("ob", j)), vr),
                         AtomicAdd(at(plus("ob", j), plus("oa", i)), vr)]
    scatter: Stmt
    differ = _c(Fn.ADD, _c(Fn.LT, Var("oa"), Var("ob")),
                _c(Fn.LT, Var("ob"), Var("oa")))
    if la == lb:
        diag: list[Stmt] = [Note(_DIAGONAL), Note(ATOMIC_REASON)]
        for i in range(nb):
            for j in range(i + 1):
                vr = Var(f"v{i * nb + j}")
                diag.append(AtomicAdd(at(plus("oa", i), plus("oa", j)), vr))
                if i != j:
                    diag.append(AtomicAdd(at(plus("oa", j), plus("oa", i)), vr))
        scatter = If(differ, tuple(off_diag), tuple(diag))
    else:
        scatter = If(differ, tuple(off_diag))

    per_rec = (*head,
               Note("the nuclei, serially inside the record (milestone 11's "
                    "mapping pin)"),
               nuclei, *values,
               Note("scatter into V_mn and its mirror V_nm"), scatter)
    body: tuple[Stmt, ...] = (
        Note(f"nuclear attraction, class (la, lb) = ({la}, {lb}): V_ab = "
             "-SUM_C Z_C SUM_t E^{ab}_t [t]^(0)(P - C, sqrt(zeta)), the J "
             f"engine's recurrence at Hermite order {la + lb} with the ket a "
             "unit point charge (ao/potential.py)"),
        Note(ZEROED_V),
        Note("launch: one thread per primitive-pair record"),
        Decl(REC, Scalar.I32, init=_c(Fn.ADD, Call(Fn.THREAD_X), _c(
            Fn.MUL, Call(Fn.BLOCK_X), Call(Fn.BLOCK_DIM_X)))),
        If(_c(Fn.LT, rec, Var("n_rec")), per_rec),
    )
    nr, nao, nn = Var("n_rec"), Var("n_ao"), Var("n_nuc")
    params = (
        Param("n_rec", Scalar.I32),
        Param("n_ao", Scalar.I32),
        Param("n_nuc", Scalar.I32),
        _f64("zeta_ab", (nr,)),
        _f64("s_ab", (nr,)),
        *(_f64(f"{a}P", (nr,)) for a in AXES),
        *(_f64(f"{a}PA", (nr,)) for a in AXES),
        *(_f64(f"{a}PB", (nr,)) for a in AXES),
        _i32("off_a", (nr,)),
        _i32("off_b", (nr,)),
        _f64("Z", (nn,)),
        _f64("Cnuc", (nn, Lit(3))),
        Param("V", Scalar.F64, array=True, intent=Intent.OUT, shape=(nao, nao)),
    )
    # wide_boys (#266): a point-charge ket has no eta to damp T = zeta |P-C|^2
    # (~1.25e5 on HF/cc-pVTZ), so this kernel asks for the wide fj_boys.
    return Kernel(name or f"v_{la}{lb}", params, body, wide_boys=True,
                  bundled=BUNDLED)
