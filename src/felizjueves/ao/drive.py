"""`J_p` from the INTERPRETED GENERATED KERNEL, as `ao.j`'s `jp=` seam (#87).

    KernelJP(boys=..., n_batches=1, lanes=1)    a `ao.j.JP`
    admitted_batches(am, candidates)            the n_batches the plan admits
    class_spec(am, n_batches)                   the Spec of a real class
    KernelDQ(lanes=32), KernelJAB(threads=32)   the dq= / jab= seams
    coulomb_matrix_chain(basis, D, *, bound, boys, ...) -> Chain
                                                the whole chain (#155)

`coulomb_matrix_chain` composes all three generated steps -- `Target.DQ`,
`[p|q]` and `Target.JAB` -- into one `J_mn`, interpreted, over milestone 6's
class-grouped staging (`ao.staged`, #143); its result reports which
generated kernel ran which step for which classes, and a step a kernel did
not cover raises. Its docstring names the one host round-trip (`Dmax` read
back to sort) a launch program would remove.

`ao.j.coulomb_matrix(basis, D, pq=..., jp=KernelJP(boys=...))` is then `J_mn`
with its ONLY shell-quartet-level step, `J_p = sum_k sum_q [p|q] D_q`, done by
`kernel.build_pq_kernels` run in `interp.Interpreter` -- the generator's own
output, with no compiler and no card. The ket pre-contraction, the bra
transform and the scatter stay `ao.j`'s (#86, already witnessed against
libcint), so a disagreement localises to the kernel.

THE CLASS IS THE BASIS'S. For a bra shell pair `(la, lb)` and a ket class
`(lc, ld)` the kernel is built from `Spec(COULOMB, 0, (la, lb, lc, ld), J,
(), n_batches)` -- never `pq_spec(L)`'s balanced split -- so `lp = la + lb`
and `lq = lc + ld`: `(p,p|s,s)` and `(s,s|p,p)` are different kernels with
different `n_p` / `n_q` (the issue's hazard).

THE BATCH AXIS. `n_batches` is asked of the plan for each class: a split the
plan refuses (#64: a batch with no target) is not built. With `strict=True`
(the default) a refused class raises; with `strict=False` it runs at one
batch and is recorded in `refused`, so one `J` can sweep a batch count
across every class while the caller sees exactly which cells ran (`ran`).
A split class's batch kernels run in `Plan.dags` order into ONE zeroed `J`,
`build_pq_kernels`' launch convention; `passes.atomize` is the builder's.

THE LAUNCH. One launch per (bra shell pair, ket class): `n_bra` = that bra
pair's primitive records, `n_ket` = every ket record of the class, `D` the
pre-contracted `D_q` in `hermite_targets(lq)` order (which `ao.hermite`'s
order is pinned equal to). `tau = 0` and `Ip = Iq = Dmax = 1`: the seam is
unscreened (`ao.j` refuses `jp` with `tau > 0`), so the bound is >= 0 and
never `< tau`, no ket is skipped, and the sorted-ket precondition -- which
exists only to make the `break` correct -- is vacuous.

BOYS IS INJECTED (`boys=`), as the interpreter requires; nothing here imports
`reference/`. Kernels are memoised per Spec (building is pure); results are
not.

`KernelJAB(threads=32)` (#154) is the same for `ao.j`'s `jab=` seam: the bra
transform and scatter of `kernel_jab.build_jab_kernel((la, lb))`, run in the
interpreter, one launch per unique bra shell pair with its records in order
(`row = 0..n_rec-1`, `Jp` the pair's own `J_p` rows), adding into the seam's
zeroed `J`. Interpreted, it is `==` to `ao.j.jab_from_host` (the kernel
module's docstring says why); `ran` is the set of `(la, lb)` launched.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np

from felizjueves.ao.density import (
    EPair,
    Matrix,
    checked_density,
    checked_tau,
    epair,
)
from felizjueves.ao.j import BraRec, KetRec
from felizjueves.ao.shell import Basis
from felizjueves.ao.staged import KetData, Staged, coulomb_matrix_staged
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import Kernel
from felizjueves.kernel import (
    bra_functions,
    build_pq_kernels,
    ket_functions,
    launch_shape,
)
from felizjueves.kernel_dq import build_dq_kernel, dq_shape
from felizjueves.kernel_dq import launch_shape as dq_launch_shape
from felizjueves.kernel_jab import build_jab_kernel, jab_launch_shape
from felizjueves.launch import (
    Bound,
    ClassLaunch,
    LaunchPlan,
    PairRec,
    _batches_for,
    _g,
    _sorted_desc,
)
from felizjueves.launch import _kernels as _plan_kernels
from felizjueves.launch import class_spec as launch_class_spec
from felizjueves.plan.pipeline import plan
from felizjueves.spec import Operator, Spec, Target

__all__ = ["BATCH_COUNTS", "KernelJAB", "KernelJP", "admitted_batches",
           "class_spec"]

BATCH_COUNTS = (1, 3, 6, 9)
Am = tuple[int, int, int, int]


def class_spec(am: Am, n_batches: int = 1) -> Spec:
    """The real class `(la, lb | lc, ld)` as a generation key."""
    return Spec(Operator.COULOMB, 0, tuple(am), Target.J, (), n_batches)


@cache
def admitted_batches(am: Am, candidates: tuple[int, ...] = BATCH_COUNTS
                     ) -> tuple[int, ...]:
    """The `n_batches` among `candidates` that `plan` admits for `am`."""
    out = []
    for n in candidates:
        try:
            plan(class_spec(am, n))
        except ValueError:
            continue
        out.append(n)
    return tuple(out)


@cache
def _kernels(spec: Spec) -> tuple[Kernel, ...]:
    return build_pq_kernels(spec)


class KernelJP:
    """A `ao.j.JP` backed by the interpreted `[p|q]` kernel (module docstring).

    `ran` is the set of `(am, n_batches)` cells launched; `refused` the
    `(am, n_batches)` the plan refused and that ran at one batch instead
    (only with `strict=False`).
    """

    def __init__(self, *, boys: Callable[[int, float], float],
                 n_batches: int = 1, lanes: int = 1,
                 strict: bool = True) -> None:
        self.boys = boys
        self.n_batches = n_batches
        self.lanes = lanes
        self.strict = strict
        self.ran: set[tuple[Am, int]] = set()
        self.refused: set[tuple[Am, int]] = set()

    def batches_for(self, am: Am) -> int:
        if self.n_batches in admitted_batches(am, (self.n_batches,)):
            return self.n_batches
        if self.strict:
            raise ValueError(f"the plan refuses n_batches = {self.n_batches} "
                             f"for class {am}")
        self.refused.add((am, self.n_batches))
        return 1

    def __call__(self, bra_am: tuple[int, int], ket_am: tuple[int, int],
                 bras: Sequence[BraRec], kets: Sequence[KetRec]) -> np.ndarray:
        am: Am = (*bra_am, *ket_am)
        n = self.batches_for(am)
        spec = class_spec(am, n)
        n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
        n_bra, n_ket = len(bras), len(kets)
        if n_ket == 0:
            return np.zeros((n_bra, n_p))
        args = dict(
            n_bra=n_bra, n_ket=n_ket, tau=0.0,
            P=np.array([b[0] for b in bras], dtype=np.float64),
            zeta=np.array([b[1] for b in bras], dtype=np.float64),
            Ip=np.ones(n_bra),
            Q=np.array([k[0] for k in kets], dtype=np.float64),
            eta=np.array([k[1] for k in kets], dtype=np.float64),
            Iq=np.ones(n_ket), Dmax=np.ones(n_ket),
            D=np.array([list(k[2]) for k in kets], dtype=np.float64),
        )
        if args["D"].shape != (n_ket, n_q):
            raise ValueError(f"D_q has shape {args['D'].shape}, class {am} "
                             f"wants ({n_ket}, {n_q})")
        grid, block = launch_shape(n_bra, self.lanes)
        J = np.zeros((n_bra, n_p))
        for k in _kernels(spec):
            it = Interpreter(k, launch=Launch(grid, block, self.lanes),
                             boys=self.boys)
            J = it.run(**args, J=J)["J"]
        self.ran.add((am, n))
        return J


# ---------------------------------------------------------------------------
# The bra transform and scatter, as `ao.j`'s `jab=` seam (#154)
# ---------------------------------------------------------------------------


@cache
def _jab_kernel(la: int, lb: int) -> Kernel:
    return build_jab_kernel((la, lb))


class KernelJAB:
    """A `ao.j.JAB` backed by the interpreted `Target.JAB` kernel (module
    docstring). `ran` is the set of bra pair classes `(la, lb)` launched."""

    def __init__(self, *, threads: int = 32) -> None:
        self.threads = threads
        self.ran: set[tuple[int, int]] = set()

    def __call__(self, bra: EPair, jps: Sequence[Sequence[float]],
                 offsets: tuple[int, int], J: list[list[float]]) -> None:
        n_rec = len(bra.recs)
        if len(jps) != n_rec:
            raise ValueError(f"{len(jps)} J_p rows for {n_rec} records")
        oa, ob = offsets
        n_ao = len(J)
        grid, block = jab_launch_shape(n_rec, self.threads)
        it = Interpreter(_jab_kernel(bra.la, bra.lb),
                         launch=Launch(grid, block, 1))
        out = it.run(
            n_rec=n_rec, n_ao=n_ao, n_rows=n_rec,
            zeta_ab=np.array([z for _, z, _ in bra.recs], dtype=np.float64),
            XPA=np.array([[P[d] - bra.A[d] for d in range(3)]
                          for P, _, _ in bra.recs], dtype=np.float64),
            XPB=np.array([[P[d] - bra.B[d] for d in range(3)]
                          for P, _, _ in bra.recs], dtype=np.float64),
            s=np.array([s for _, _, s in bra.recs], dtype=np.float64),
            off_a=np.full(n_rec, oa, dtype=np.int32),
            off_b=np.full(n_rec, ob, dtype=np.int32),
            row=np.arange(n_rec, dtype=np.int32),
            Jp=np.array([list(r) for r in jps], dtype=np.float64),
            J=np.array(J, dtype=np.float64),
        )["J"]
        for x in range(n_ao):
            J[x][:] = out[x].tolist()
        self.ran.add((bra.la, bra.lb))


# ---------------------------------------------------------------------------
# `D_q` from the interpreted pre-contraction kernel: `ao.j`'s `dq=` seam (#153)
# ---------------------------------------------------------------------------

__all__ += ["KernelDQ"]

DQItem = tuple[EPair, Sequence[float], float]  # (ket, dcd, w): one dq= call


@cache
def _dq_kernel(lc: int, ld: int) -> Kernel:
    return build_dq_kernel((lc, ld))


class KernelDQ:
    """A `ao.j.DQ` backed by the interpreted `Target.DQ` kernel
    (`kernel_dq.build_dq_kernel`): `D_q` per ket record, and `Dmax`.

    `ao.j.coulomb_matrix(basis, D, pq=..., dq=KernelDQ())` is then `J_mn`
    with the ket pre-contraction done by the generator's own kernel in
    `interp.Interpreter` -- no compiler, no card -- and everything else
    (`Q`, `eta`, `J_p`, the bra transform, the scatter) `ao.j`'s, so a
    disagreement localises to this kernel.

    `run(items)` is ONE launch over every record of several ket shell pairs
    of one class -- each item `(ket, dcd, w)` exactly as `dq=` receives it,
    `pair[k]` the item record `k` came from -- and returns `(D, Dmax)` in
    record order. `__call__` is the seam: one item, `D` as rows.

    THE WEIGHT AND THE DRESSING ARE THE HOST'S. `w` is whatever `ao.j`
    passed (2.0 off the diagonal, 1.0 on it) and `s` is `EPair`'s, from
    `pairs.dress`; both reach the kernel as per-record scalars, and nothing
    here decides either. The kernel receives `X_PA = P - A` and
    `X_PB = P - B`, formed here per axis with the subtraction in that order,
    as `ao.hermite.e_block` forms them.

    `ran` is the set of ket classes `(lc, ld)` launched. Kernels are
    memoised per class (building is pure); results are not.
    """

    def __init__(self, *, lanes: int = 32) -> None:
        self.lanes = lanes
        self.ran: set[tuple[int, int]] = set()

    def run(self, items: Sequence[DQItem]) -> tuple[np.ndarray, np.ndarray]:
        if not items:
            raise ValueError("no ket shell pairs to pre-contract")
        classes = {(ket.la, ket.lb) for ket, _, _ in items}
        if len(classes) != 1:
            raise ValueError(f"one launch is one ket class, got "
                             f"{sorted(classes)}")
        ((lc, ld),) = classes
        n_cd, n_q = dq_shape(lc, ld)
        eta, xpa, xpb, s, w, pair = [], [], [], [], [], []
        for i, (ket, dcd, wt) in enumerate(items):
            if len(dcd) != n_cd:
                raise ValueError(f"dcd has {len(dcd)} entries, class "
                                 f"{(lc, ld)} wants {n_cd}")
            for P, zeta, sk in ket.recs:
                eta.append(zeta)
                xpa.append([P[d] - ket.A[d] for d in range(3)])
                xpb.append([P[d] - ket.B[d] for d in range(3)])
                s.append(sk)
                w.append(float(wt))
                pair.append(i)
        n_rec = len(eta)
        if n_rec == 0:
            return np.zeros((0, n_q)), np.zeros(0)
        grid, block = dq_launch_shape(n_rec, self.lanes)
        out = Interpreter(_dq_kernel(lc, ld),
                          launch=Launch(grid, block, self.lanes)).run(
            n_rec=n_rec, n_pair=len(items),
            eta=np.array(eta, dtype=np.float64),
            XPA=np.array(xpa, dtype=np.float64),
            XPB=np.array(xpb, dtype=np.float64),
            s=np.array(s, dtype=np.float64), w=np.array(w, dtype=np.float64),
            pair=np.array(pair, dtype=np.int32),
            Dcd=np.array([[float(x) for x in dcd] for _, dcd, _ in items],
                         dtype=np.float64),
            D=np.zeros((n_rec, n_q)), Dmax=np.zeros(n_rec))
        self.ran.add((lc, ld))
        return out["D"], out["Dmax"]

    def __call__(self, ket: EPair, dcd: Sequence[float],
                 w: float) -> list[list[float]]:
        D, _ = self.run([(ket, dcd, w)])
        return D.tolist()


# ---------------------------------------------------------------------------
# The whole chain: `J_mn` with every arithmetic step generated (#155)
# ---------------------------------------------------------------------------

__all__ += ["Chain", "coulomb_matrix_chain"]

Rec = tuple[int, int, int]  # (shell i, shell j, primitive pair index)


@dataclass(frozen=True)
class Chain:
    """`coulomb_matrix_chain`'s result: `J`, and WHICH GENERATED KERNEL
    computed each arithmetic step, for which classes -- the report the
    acceptance is asserted on, rather than prose.

    `dq`   the ket pair classes `(lc, ld)` whose `D_q` and `Dmax` came from
           `Target.DQ` (`KernelDQ.run`, one launch per class), in launch order;
    `pq`   the `(lp, lq, n_batches)` classes whose `J_p` came from the
           `[p|q]` kernels (`Staged.ran`), in plan order;
    `jab`  the bra pair classes `(la, lb)` whose eq. 20 and scatter came from
           `Target.JAB`, one launch per class, in launch order;
    `n_dq_records`, `n_jab_records`  the records each step's launches
           covered (checked equal to the basis's unique-pair records);
    `plan` the launch plan the chain built, keyed on the kernel's `Dmax`;
    `staged`  `ao.staged`'s record of the buffers and `[p|q]` launches."""

    J: Matrix
    dq: tuple[tuple[int, int], ...]
    pq: tuple[tuple[int, int, int], ...]
    jab: tuple[tuple[int, int], ...]
    n_dq_records: int
    n_jab_records: int
    plan: LaunchPlan
    staged: Staged

    def steps(self) -> dict[str, tuple]:
        """Step name -> the classes a generated kernel ran it for."""
        return {"DQ": self.dq, "[p|q]": self.pq, "JAB": self.jab}


def _unique_pairs(basis: Basis) -> list[tuple[int, int]]:
    n = len(basis.shells)
    return [(a, b) for a in range(n) for b in range(a + 1)]


def coulomb_matrix_chain(basis: Basis, D: object, *, bound: Bound,
                         boys: Callable[[int, float], float],
                         n_batches: int = 1, tau: float = 0.0,
                         strict: bool = True,
                         resolve: Callable[[int, int], int] | None = None,
                         lanes: int = 1, dq_lanes: int = 32,
                         jab_threads: int = 32) -> Chain:
    """`J_mn` with every arithmetic step a GENERATED kernel, interpreted.

    For each ket pair class, `Target.DQ` (`KernelDQ`); for each `(lp, lq)`
    class, the `[p|q]` kernels; for each bra pair class, `Target.JAB`. The
    host owns only the pair list (unique shell pairs `a >= b`, their
    records, the weight `w`), the density check, the bounds (`bound(zeta,
    L)`, injected, as `launch.launches` takes it), the sort, the staging and
    the launch order -- and the staging and `[p|q]` launches ARE milestone
    6's (#143): `ao.staged.coulomb_matrix_staged`, through its `ket_data=`
    and `transform=` hooks, over a `launch.LaunchPlan`.

    THE ROUND-TRIP, named as the launch program's to remove. The sort needs
    `Dmax`, and `Dmax` is the DQ kernel's output: so the chain READS `Dmax`
    BACK from every DQ launch, keys each ket record `G * Dmax`, sorts each
    ket class on the host (`launch`'s stable descending sort), and gathers
    `D_q` into the class's buffer in that order before `[p|q]` launches.
    It is the chain's only device-to-host transfer before `J` itself
    (j_engine.md, "The round-trip"); a launch program that sorted on the
    device would remove it. The plan here is built from the KERNEL's `Dmax`,
    never from a host pre-contraction (`launch.launches` computes one; it is
    not called), and `ao.staged` checks every staged `Iq * Dmax` against
    the plan key it sorted by, bit for bit.

    A GAP RAISES. There is no `pq=`, no host `D_q` and no host transform to
    fall back to: a class whose kernel cannot be built or run raises out of
    the builder or the interpreter; a refused batch count raises with
    `strict=True` (`launch`'s contract; `strict=False` builds it at one
    batch and records it in `plan.refused`); and after the launches every
    ket record must have had a DQ row, every bra record a JAB launch, and
    every pair class a step, or this raises.

    The host still forms `P`, `zeta`, `X_PA`, `X_PB` and the dressing scalar
    `s` per record (`ao.density.epair`, `ao.pairs`): pair data and the contract's
    host-supplied inputs, not steps of `J` (j_engine.md, "What stays
    host-side"). `ao.pairs.primitive_pairs` also forms each record's dressed
    `E` block as a side effect of building the record; no value of it
    reaches `J` here.

    `tau` is the `[p|q]` kernel's screen (real, as in `ao.staged`); the
    libcint witness is at `tau = 0`.
    """
    rows = checked_density(basis, D)
    tau = checked_tau(tau)
    shells, off = basis.shells, basis.offsets
    pairs = _unique_pairs(basis)

    # --- the pair list, the records and their bounds (host) ----------------
    eps: dict[tuple[int, int], EPair] = {}
    gs: dict[tuple[int, int], list[float]] = {}
    for a, b in pairs:
        sa, sb = shells[a], shells[b]
        eps[(a, b)] = ep = epair(sa, sb)
        L = sa.l + sb.l
        gs[(a, b)] = [_g(bound, zeta, L) for _, zeta, _ in ep.recs]

    # --- step 2: D_q and Dmax, one DQ launch per ket pair class -------------
    kdq = KernelDQ(lanes=dq_lanes)
    by_class: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in pairs:
        by_class.setdefault((shells[a].l, shells[b].l), []).append((a, b))
    dq_rows: dict[Rec, tuple[np.ndarray, float]] = {}
    dq_order: list[tuple[int, int]] = []
    n_dq = 0
    for cls in sorted(by_class):
        members = by_class[cls]
        items = []
        for c, d in members:
            sc, sd = shells[c], shells[d]
            dcd = [rows[off[c] + i][off[d] + j]
                   for i in range(sc.n_cart) for j in range(sd.n_cart)]
            items.append((eps[(c, d)], dcd, 1.0 if c == d else 2.0))
        Dk, Dmax = kdq.run(items)  # THE ROUND-TRIP: Dmax read back
        k = 0
        for c, d in members:
            for prim in range(len(eps[(c, d)].recs)):
                dq_rows[(c, d, prim)] = (Dk[k], float(Dmax[k]))
                k += 1
        if k != len(Dk):
            raise ValueError(f"DQ class {cls}: {len(Dk)} rows for {k} "
                             "records")
        n_dq += k
        dq_order.append(cls)

    # --- the sort on the read-back Dmax, and the launch plan (host) --------
    cplan = _chain_plan(basis, pairs, gs, dq_rows, n_batches=n_batches,
                        tau=tau, strict=strict, resolve=resolve)

    def ket_data(recs: tuple[PairRec, ...], n_q: int) -> KetData:
        got = []
        for r in recs:  # the gather, in the plan's sorted order
            if (r.i, r.j, r.prim) not in dq_rows:
                raise ValueError(f"ket record {(r.i, r.j, r.prim)} has no "
                                 "D_q from a DQ launch")
            got.append(dq_rows[(r.i, r.j, r.prim)])
        P = [eps[(r.i, r.j)].recs[r.prim] for r in recs]
        return KetData(
            Q=np.array([p[0] for p in P], dtype=np.float64),
            eta=np.array([p[1] for p in P], dtype=np.float64),
            D=np.array([g[0] for g in got], dtype=np.float64).reshape(
                len(recs), n_q),
            Dmax=np.array([g[1] for g in got], dtype=np.float64))

    jab_order: list[tuple[int, int]] = []
    n_jab = [0]

    def transform(blocks, jps) -> Matrix:
        J, ran, n = _jab_launches(basis, eps, blocks, jps, jab_threads)
        jab_order.extend(ran)
        n_jab[0] = n
        return J

    st = coulomb_matrix_staged(cplan, basis, D, boys=boys, lanes=lanes,
                               ket_data=ket_data, transform=transform)

    # --- the report, and the gap check --------------------------------------
    n_rec = sum(len(ep.recs) for ep in eps.values())
    want_pair_classes = set(by_class)
    want_pq = {(cl.lp, cl.lq) for cl in cplan}
    want_lp = {shells[a].l + shells[b].l for a, b in pairs}
    gaps = []
    if set(dq_order) != want_pair_classes or n_dq != n_rec:
        gaps.append(f"DQ ran {sorted(dq_order)} over {n_dq} records; the "
                    f"basis has {sorted(want_pair_classes)} over {n_rec}")
    if set(jab_order) != want_pair_classes or n_jab[0] != n_rec:
        gaps.append(f"JAB ran {sorted(jab_order)} over {n_jab[0]} records; "
                    f"the basis has {sorted(want_pair_classes)} over {n_rec}")
    if {(lp, lq) for lp, lq, _ in st.ran} != want_pq or {
            lp for lp, _ in want_pq} != want_lp:
        gaps.append(f"[p|q] ran {sorted(st.ran)}; the plan has "
                    f"{sorted(want_pq)} and the bra classes {sorted(want_lp)}")
    if gaps:
        raise ValueError("the chain has a gap, and nothing covers it: "
                         + "; ".join(gaps))
    return Chain(st.J, tuple(dq_order), st.ran, tuple(jab_order), n_dq,
                 n_jab[0], cplan, st)


def _chain_plan(basis: Basis, pairs, gs, dq_rows, *, n_batches: int,
                tau: float, strict: bool, resolve) -> LaunchPlan:
    """`launch.launches`' plan, keyed on the DQ kernel's read-back `Dmax`:
    the same unique pairs, records, bra blocks, classes, batch counts and
    stable descending sort (`launch`'s own helpers), with no host
    pre-contraction. Product key only (`density_sort=True`)."""
    shells = basis.shells
    bras: dict[int, list[PairRec]] = {}
    blocks: dict[int, list[tuple[int, int, int, int]]] = {}
    kets: dict[int, list[PairRec]] = {}
    for a, b in pairs:
        L = shells[a].l + shells[b].l
        side = bras.setdefault(L, [])
        start = len(side)
        for k, g in enumerate(gs[(a, b)]):
            side.append(PairRec(a, b, k, L, g, g))
            kets.setdefault(L, []).append(
                PairRec(a, b, k, L, g, g * dq_rows[(a, b, k)][1]))
        blocks.setdefault(L, []).append((a, b, start, len(side)))
    refused: list[tuple[int, int, int]] = []
    out = []
    for lp in sorted(bras):
        for lq in sorted(kets):
            n = _batches_for(lp, lq, n_batches, strict=strict,
                             refused=refused, resolve=resolve)
            out.append(ClassLaunch(
                lp, lq, n, tuple(bras[lp]), tuple(_sorted_desc(kets[lq])),
                _plan_kernels(launch_class_spec(lp, lq, n)),
                tuple(blocks[lp])))
    return LaunchPlan(tuple(out), tau, n_batches, True, tuple(refused))


def _jab_launches(basis: Basis, eps, blocks, jps, threads: int):
    """Eq. 20 and the scatter: ONE `Target.JAB` launch per bra pair class
    `(la, lb)`, over every record of every bra shell pair of that class,
    each reading its row of its bra class's `J_p` buffer, all adding into
    one zeroed `J`. Returns `(J, classes launched, records covered)`."""
    shells, off = basis.shells, basis.offsets
    n_ao = basis.n_ao
    J = np.zeros((n_ao, n_ao))
    groups: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = {}
    for lp, blks in blocks.items():
        for a, b, start, stop in blks:
            groups.setdefault((shells[a].l, shells[b].l), []).append(
                (lp, a, b, start, stop))
    ran, n_cov = [], 0
    for la, lb in sorted(groups):
        members = groups[(la, lb)]
        lp = la + lb
        if any(m[0] != lp for m in members):
            raise ValueError(f"bra class {(la, lb)} filed under a J_p buffer "
                             f"other than lp = {lp}")
        Jp = jps[lp]
        recs, rows_, oa, ob = [], [], [], []
        for _, a, b, start, stop in members:
            ep = eps[(a, b)]
            if len(ep.recs) != stop - start:
                raise ValueError(f"bra block {(a, b)}: {stop - start} rows "
                                 f"for {len(ep.recs)} records")
            for k, rec in enumerate(ep.recs):
                recs.append((ep, rec))
                rows_.append(start + k)
                oa.append(off[a])
                ob.append(off[b])
        n_rec = len(recs)
        grid, block = jab_launch_shape(n_rec, threads)
        J = Interpreter(_jab_kernel(la, lb),
                        launch=Launch(grid, block, 1)).run(
            n_rec=n_rec, n_ao=n_ao, n_rows=Jp.shape[0],
            zeta_ab=np.array([r[1] for _, r in recs], dtype=np.float64),
            XPA=np.array([[r[0][d] - ep.A[d] for d in range(3)]
                          for ep, r in recs], dtype=np.float64),
            XPB=np.array([[r[0][d] - ep.B[d] for d in range(3)]
                          for ep, r in recs], dtype=np.float64),
            s=np.array([r[2] for _, r in recs], dtype=np.float64),
            off_a=np.array(oa, dtype=np.int32),
            off_b=np.array(ob, dtype=np.int32),
            row=np.array(rows_, dtype=np.int32),
            Jp=np.asarray(Jp, dtype=np.float64), J=J)["J"]
        ran.append((la, lb))
        n_cov += n_rec
    if not np.all(np.isfinite(J)):
        raise ValueError("J has a non-finite entry")
    return tuple(tuple(float(x) for x in r) for r in J), ran, n_cov
