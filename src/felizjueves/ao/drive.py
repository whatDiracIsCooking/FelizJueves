"""`J_p` from the INTERPRETED GENERATED KERNEL, as `ao.j`'s `jp=` seam (#87).

    KernelJP(boys=..., n_batches=1, lanes=1)    a `ao.j.JP`
    admitted_batches(am, candidates)            the n_batches the plan admits
    class_spec(am, n_batches)                   the Spec of a real class

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
from functools import cache

import numpy as np

from felizjueves.ao.j import BraRec, EPair, KetRec
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import Kernel
from felizjueves.kernel import (
    bra_functions,
    build_pq_kernels,
    ket_functions,
    launch_shape,
)
from felizjueves.kernel_jab import build_jab_kernel, jab_launch_shape
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
