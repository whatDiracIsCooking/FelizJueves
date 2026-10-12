"""`J_mn` from a LAUNCH PLAN, class-grouped and staged (#143; milestone 6).

    coulomb_matrix_staged(plan, basis, D, *, boys, lanes=1,
                          check_sorted=True)            -> Staged
    Staged    J, the buffers it built (counted), the cells that ran, and the
              interpreter's ket-loop iterations per class

The host side of residency, at interpreter level: the plan's shape replacing
`ao.drive.KernelJP`'s one launch per (bra shell pair, ket class). The
adapter beside it is UNCHANGED -- two paths, not one replaced.

THE BUFFERS, each built ONCE and counted as it is built (`Staged.builds`),
so the count a test asserts is measured, not assumed:

  bra side, per bra class `lp`   `P[n_bra, 3]`, `zeta`, `Ip` -- the plan's
      bra records in pair order (the kernel maps one warp per bra BY INDEX);
  ket side, per ket class `lq`   `Q[n_ket, 3]`, `eta`, `Iq`, `Dmax`,
      `D[n_ket, n_q]` in `hermite_targets(lq)` order -- the plan's ket records
      in the plan's sorted order. Every `ClassLaunch` with that `lq` must
      carry the SAME ket list (the plan sorts each ket class once), and it is
      refused otherwise, so one upload per ket class is sound;
  `J_p`, per bra class `lp`      `n_bra x n_p`, zeroed once, and EVERY batch
      kernel of every `(lp, lq)` class launched into it.

THE SCREEN IS REAL. `Ip` and `Iq` are the records' `G` -- the injected
`I_pair` -- and `Dmax` is `max_q |D_q|` of the weighted, dressed ket vector
(`ao.j`'s pre-contraction, called rather than restated), or with the plan's
`density_sort=False` the one global value it ranked by. `tau` is the plan's.
NOT the `jp=` seam's `Ip = Iq = Dmax = 1, tau = 0`: carried here by accident,
that would leave the screen and the break untested while everything stayed
green (#143's hazard). Every ket's `Iq * Dmax` is checked equal to its plan
`key`, BIT FOR BIT, so the order the kernel's `break` relies on is the order
the plan sorted by.

THE PRECONDITION. The kernel's screen is a `break`: kets must arrive sorted
descending by `Iq * Dmax`. The plan establishes it; `check_sorted=True` (the
default) refuses a ket list that is not, and `check_sorted=False` exists so a
test can inject an unsorted list and show that the answer then CHANGES.

ATOMICS ACROSS KET CLASSES. One `J_p` per bra class has a writer per ket
class, so the write-back must ADD. A split class's kernels are `atomize`d by
the builder already; a one-batch class's kernel keeps `[JEngine, v2, Alg. 1]`
line 10's plain write (the builder's deliberate asymmetry), which is correct
only for a buffer's SOLE writer -- so here it is `passes.atomize`d, exactly
the pass the builder applies to a split class. The builder is unchanged.

EQ. 20 AND THE SCATTER stay host-side (milestone (c)): each bra block
`(a, b, start, stop)` of the plan -- the record -> shell pair map -- takes
rows `start:stop` of its class's `J_p` through the dressed `E^{ab}_p` and is
scattered to `J[a][b]` and, mirrored, `J[b][a]`: the arithmetic of `ao.j`'s
`jp=` sweep, restated here because `ao.j` exposes it only inside that sweep.

THE HOOKS (#155), additive: without them every path above is unchanged.

  `ket_data(recs, n_q) -> KetData`  replaces the HOST pre-contraction as the
      source of one ket class's `Q`, `eta`, `D` and `Dmax`, in the order of
      `recs` (the plan's sorted order). The checks stay here: the shapes, and
      every `Iq * Dmax` equal to its plan key bit for bit -- so a hook whose
      `Dmax` is not the one the plan sorted by is refused, not run. With a
      hook, `density_sort=False`'s host global `Dmax` is not computed: the
      hook's `Dmax` is the one checked.
  `transform(blocks, jps) -> J`  replaces eq. 20 and the scatter: `blocks`
      the bra blocks `(a, b, start, stop)` per bra class `lp`, `jps` each bra
      class's finished `J_p` buffer.

`ao.drive.coulomb_matrix_chain` is their one caller: `D_q` and `Dmax` from
the generated pre-contraction, eq. 20 and the scatter from the generated
bra-transform kernel, and the staging, its checks and the `[p|q]` launches
this module's, unchanged.

BOYS IS INJECTED (`boys=`); nothing here imports `reference/`. Nothing is
memoised across calls: the buffers are built per call and dropped.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from felizjueves.ao.j import Matrix, _density, _kets
from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import AtomicAdd, Kernel, walk
from felizjueves.kernel import (
    KET,
    bra_functions,
    ket_functions,
    launch_shape,
)
from felizjueves.launch import ClassLaunch, LaunchPlan, PairRec, class_spec
from felizjueves.passes import atomize

__all__ = ["BufferSize", "KetData", "Staged", "coulomb_matrix_staged"]

_F64 = 8  # bytes per double


@dataclass(frozen=True)
class BufferSize:
    """One staged buffer set: `side` is "bra", "ket" or "J_p", `l` its class
    (`lp` or `lq`), `n_records` its rows, `n_fn` the Hermite width (`n_p` for
    `J_p`, `n_q` for the ket `D`; 0 for the bra side), `n_doubles` every
    double it holds, and `bytes` = 8 * `n_doubles`."""

    side: str
    l: int  # noqa: E741 -- the class's angular momentum, as everywhere here
    n_records: int
    n_fn: int
    n_doubles: int

    @property
    def bytes(self) -> int:
        return _F64 * self.n_doubles


@dataclass(frozen=True)
class Staged:
    """`coulomb_matrix_staged`'s result (module docstring).

    `builds` counts buffer builds by side ("bra", "ket", "J_p"); `sizes` the
    buffers, one `BufferSize` per build; `ran` the `(lp, lq, n_batches)`
    classes launched, in plan order; `iterations[(lp, lq)]` the ket-loop body
    entries (`kernel.KET`) summed over that class's batch kernels -- with
    `lanes = 1`, exactly the (bra record, ket record) pairs each kernel
    examined, the breaking ket included."""

    J: Matrix
    builds: Counter[str]
    sizes: tuple[BufferSize, ...]
    ran: tuple[tuple[int, int, int], ...]
    iterations: dict[tuple[int, int], int]


@dataclass(frozen=True)
class KetData:
    """What a `ket_data=` hook returns for one ket class (module docstring,
    "THE HOOKS"): per record, in the order of the records it was given,
    `Q[n, 3]`, `eta[n]`, `D[n, n_q]` and `Dmax[n]`."""

    Q: np.ndarray
    eta: np.ndarray
    D: np.ndarray
    Dmax: np.ndarray


KetDataHook = Callable[[tuple[PairRec, ...], int], KetData]
Transform = Callable[[dict[int, tuple[tuple[int, int, int, int], ...]],
                      dict[int, np.ndarray]], Matrix]


@dataclass(frozen=True)
class _Bras:
    P: np.ndarray
    zeta: np.ndarray
    Ip: np.ndarray


@dataclass(frozen=True)
class _Kets:
    recs: tuple[PairRec, ...]
    Q: np.ndarray
    eta: np.ndarray
    Iq: np.ndarray
    Dmax: np.ndarray
    D: np.ndarray


def _bra_buffers(basis: Basis, recs: tuple[PairRec, ...]) -> _Bras:
    sh = basis.shells
    prims = [primitive_pairs(sh[r.i], sh[r.j])[r.prim] for r in recs]
    return _Bras(
        P=np.array([p.P for p in prims], dtype=np.float64),
        zeta=np.array([p.zeta for p in prims], dtype=np.float64),
        Ip=np.array([r.G for r in recs], dtype=np.float64))


def _ket_vectors(basis: Basis, rows, recs) -> list:
    """`ao.j`'s ket records for `recs`, one `_kets` call per shell pair."""
    by_pair: dict[tuple[int, int], list] = {}
    out = []
    for r in recs:
        if r.shell_pair not in by_pair:
            w = 1.0 if r.i == r.j else 2.0
            by_pair[r.shell_pair] = _kets(basis, rows, [(r.i, r.j, w)],
                                          pq=_no_pq, tau=0.0)
        out.append(by_pair[r.shell_pair][r.prim])
    return out


def _ket_buffers(basis: Basis, rows, recs: tuple[PairRec, ...], n_q: int,
                 dglobal: float | None) -> _Kets:
    kts = _ket_vectors(basis, rows, recs)
    dmax = [max(abs(v) for v in k.dq) if dglobal is None else dglobal
            for k in kts]
    return _kets_checked(recs, n_q, KetData(
        Q=np.array([k.Q for k in kts], dtype=np.float64),
        eta=np.array([k.eta for k in kts], dtype=np.float64),
        D=np.array([list(k.dq) for k in kts], dtype=np.float64),
        Dmax=np.array(dmax, dtype=np.float64)))


def _kets_checked(recs: tuple[PairRec, ...], n_q: int, kd: KetData) -> _Kets:
    """One ket class's buffers from `kd` (rows in `recs`' order): the shapes
    checked, and every `Iq * Dmax` equal to its plan key, bit for bit."""
    n = len(recs)
    for name, got, want in (("D_q", np.shape(kd.D), (n, n_q)),
                            ("Q", np.shape(kd.Q), (n, 3)),
                            ("eta", np.shape(kd.eta), (n,)),
                            ("Dmax", np.shape(kd.Dmax), (n,))):
        if got != want:
            raise ValueError(f"{name} has shape {got}, the class wants "
                             f"{want}")
    for r, d in zip(recs, kd.Dmax, strict=True):
        if r.G * float(d) != r.key:
            raise ValueError(
                f"ket record {(r.i, r.j, r.prim)}: Iq * Dmax = "
                f"{r.G * float(d)!r} is not its plan key {r.key!r}; the "
                "kernel's break would read a different order from the one "
                "the plan sorted")
    return _Kets(
        recs=recs,
        Q=np.asarray(kd.Q, dtype=np.float64),
        eta=np.asarray(kd.eta, dtype=np.float64),
        Iq=np.array([r.G for r in recs], dtype=np.float64),
        Dmax=np.asarray(kd.Dmax, dtype=np.float64),
        D=np.asarray(kd.D, dtype=np.float64))


def _no_pq(p, q, PQv, theta) -> float:
    raise AssertionError("no [p|q] is evaluated building kets at tau = 0")


def _dglobal(plan: LaunchPlan, basis: Basis, rows) -> float | None:
    """`density_sort=False`'s one global `Dmax`: the largest per-record
    `Dmax` over every ket of the plan (launch.py's definition)."""
    if plan.density_sort:
        return None
    seen: dict[int, tuple[PairRec, ...]] = {}
    for cl in plan:
        seen.setdefault(cl.lq, cl.kets)
    return max((max(abs(v) for v in k.dq)
                for recs in seen.values()
                for k in _ket_vectors(basis, rows, recs)), default=0.0)


def _sorted_desc(recs: tuple[PairRec, ...]) -> bool:
    return all(a.key >= b.key for a, b in zip(recs, recs[1:], strict=False))


def _accumulating(k: Kernel) -> Kernel:
    """A kernel that ADDS into `J`: split kernels already do; a one-batch
    kernel's plain write is atomized (module docstring)."""
    if any(isinstance(s, AtomicAdd) for s in walk(k.body)):
        return k
    return atomize(k)


def coulomb_matrix_staged(plan: LaunchPlan, basis: Basis, D: object, *,
                          boys: Callable[[int, float], float], lanes: int = 1,
                          check_sorted: bool = True,
                          ket_data: KetDataHook | None = None,
                          transform: Transform | None = None) -> Staged:
    """`J_mn` from `plan` (built by `launch.launches` for this `basis` and
    `D`), its buffers staged per class and `J_p` from the interpreted
    generated kernels (module docstring; `ket_data=` and `transform=` are
    "THE HOOKS")."""
    rows = _density(basis, D)
    dglobal = None if ket_data is not None else _dglobal(plan, basis, rows)
    builds: Counter[str] = Counter()
    sizes: list[BufferSize] = []
    bras: dict[int, _Bras] = {}
    kets: dict[int, _Kets] = {}
    jps: dict[int, np.ndarray] = {}
    blocks: dict[int, tuple[tuple[int, int, int, int], ...]] = {}
    ran: list[tuple[int, int, int]] = []
    iterations: dict[tuple[int, int], int] = {}

    for cl in plan:
        spec = class_spec(cl.lp, cl.lq, cl.n_batches)
        n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
        n_bra, n_ket = len(cl.bras), len(cl.kets)
        if cl.lp not in bras:
            bras[cl.lp] = _bra_buffers(basis, cl.bras)
            blocks[cl.lp] = cl.bra_blocks
            builds["bra"] += 1
            sizes.append(BufferSize("bra", cl.lp, n_bra, 0, 5 * n_bra))
            jps[cl.lp] = np.zeros((n_bra, n_p))
            builds["J_p"] += 1
            sizes.append(BufferSize("J_p", cl.lp, n_bra, n_p, n_bra * n_p))
        elif blocks[cl.lp] != cl.bra_blocks:
            raise ValueError(f"bra class {cl.lp}: two launches carry "
                             "different bra lists; one J_p buffer cannot "
                             "serve both")
        if cl.lq not in kets:
            if check_sorted and not _sorted_desc(cl.kets):
                raise ValueError(
                    f"class ({cl.lp}, {cl.lq}): kets are not sorted "
                    "descending by key, and the kernel's screen is a break")
            kets[cl.lq] = (
                _ket_buffers(basis, rows, cl.kets, n_q, dglobal)
                if ket_data is None
                else _kets_checked(cl.kets, n_q, ket_data(cl.kets, n_q)))
            builds["ket"] += 1
            sizes.append(BufferSize("ket", cl.lq, n_ket, n_q,
                                    n_ket * (6 + n_q)))
        elif kets[cl.lq].recs != cl.kets:
            raise ValueError(f"ket class {cl.lq}: two launches carry "
                             "different ket lists; one upload cannot serve "
                             "both")
        iterations[(cl.lp, cl.lq)] = _launch(
            cl, bras[cl.lp], kets[cl.lq], jps[cl.lp], plan.tau, boys, lanes)
        ran.append((cl.lp, cl.lq, cl.n_batches))

    J = (_transform(basis, blocks, jps) if transform is None
         else transform(blocks, jps))
    return Staged(J, builds, tuple(sizes), tuple(ran), iterations)


def _launch(cl: ClassLaunch, b: _Bras, k: _Kets, J: np.ndarray, tau: float,
            boys, lanes: int) -> int:
    """Every batch kernel of `cl` into `J` (in place); the ket-loop count."""
    grid, block = launch_shape(len(cl.bras), lanes)
    args = dict(n_bra=len(cl.bras), n_ket=len(cl.kets), tau=tau,
                P=b.P, zeta=b.zeta, Ip=b.Ip,
                Q=k.Q, eta=k.eta, Iq=k.Iq, Dmax=k.Dmax, D=k.D)
    n = 0
    for kern in cl.kernels:
        it = Interpreter(_accumulating(kern), launch=Launch(grid, block, lanes),
                         boys=boys)
        J[...] = it.run(**args, J=J)["J"]
        n += it.iterations[KET]
    return n


def _transform(basis: Basis, blocks, jps) -> Matrix:
    """Eq. 20 per bra block, and the mirrored scatter (module docstring)."""
    shells, off = basis.shells, basis.offsets
    n = basis.n_ao
    J = [[0.0] * n for _ in range(n)]
    for lp, blks in blocks.items():
        jp = jps[lp]
        for a, b, start, stop in blks:
            sa, sb = shells[a], shells[b]
            recs = primitive_pairs(sa, sb)
            if len(recs) != stop - start:
                raise ValueError(f"bra block {(a, b)}: {stop - start} rows "
                                 f"for {len(recs)} primitive pairs")
            jab = [0.0] * len(recs[0].block)
            for rec, row_jp in zip(recs, jp[start:stop], strict=True):
                for r, row in enumerate(rec.block):
                    s = 0.0
                    for e, v in zip(row, row_jp, strict=True):
                        s += e * float(v)
                    jab[r] += s
            for i in range(sa.n_cart):
                for j in range(sb.n_cart):
                    v = jab[i * sb.n_cart + j]
                    J[off[a] + i][off[b] + j] = v
                    J[off[b] + j][off[a] + i] = v
    if not all(math.isfinite(x) for r in J for x in r):
        raise ValueError("J has a non-finite entry")
    return tuple(tuple(r) for r in J)
