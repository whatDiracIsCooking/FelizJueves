"""The J and K builds FROM A LAUNCH PLAN on the shared runner (#144, #206,
#659) -- harness.

    driver(plan, backend, workdir)          -> Driver   (runner + sequence)
    stage(plan, basis, D, driver)           -> {buffer: array}, the input
    compile_checked(tc, driver, out)        -> Path     (#92's warning filter,
                                               harness_driver's since #659)
    run(driver, binary, buffers, workdir)   -> Sequenced, one process
    read_back(driver, ran)                  -> [run][lp] -> J_p[n_bra, n_p]
    read_back_tail(driver, ran)             -> [run] -> J_mn (or a tail's)
    reps(ran)                               -> per timed build, its clocks
    compile_dq(classes, ...).run()          -> the DQ step, on the runner
    compile_jab / jab_step / run_jab        -> one JAB unit, on the runner
    k_driver / k_stage / k_run / k_scatter  -> K the same way (#206)

The device twin of `eri.j_coulomb.staged.coulomb_matrix_staged` (#143): the plan's
shape on the card, where `test_ao_j_ondevice.py` (#89) launched one process
per class. Several batch launches into ONE zeroed `dJ` per bra class, from
one class to a whole molecule. It is HARNESS, in `test/`, not a second
emitter in `src/`: the kernels are `emit/cuda.py` / `emit/hip.py`'s text,
unchanged, every unit in its own namespace in ONE runner
(`harness_driver.units_runner`); no `main()` is written here (#659).

WHAT ONE RUN OF THE SEQUENCE DOES (`run` runs it `RUNS` times, or `warmup`
+ `runs` in the timing mode, all in ONE process, so the fp64 atomics'
run-to-run spread is measured, not assumed):

  - one `J<lp>` buffer per BRA class `lp`, `n_bra(lp) x n_p(lp)`, zeroed
    ONCE per run by the first launch (inside the run's clock);
  - every `(lp, lq)` class's batch kernels, in plan order, launched into
    `J<lp>` -- several ket classes and several batches per buffer, so every
    kernel ADDS: a split class's kernels are `atomize`d by the builder, and a
    one-batch kernel's plain write (correct only for a sole writer) is
    `passes.atomize`d here, exactly as `eri.j_coulomb.staged._accumulating` does (#143's
    decision, carried to the card);
  - one synchronize, then every `J<lp>` read back, one copy each.

THE UPLOADS happen once, before the first run: ONE buffer and one copy per
bra class (`bra<lp>`: `P zeta Ip`) and per ket class (`ket<lq>`: `Q eta Iq
Dmax D`), the kernel's arguments `Ref`s into it -- one upload per ket
class, not one per (bra pair, ket class). The sequence (`Driver.steps`)
names every argument of every launch by the kernel's parameter list, and
the runner checks it against the compiled signature.

THE DATA is `eri.j_coulomb.staged`'s, not restated: `_bra_buffers` / `_ket_buffers`
build the very arrays the interpreted staged driver feeds `interp`, including
its bit-for-bit check that every ket's `Iq * Dmax` is its plan key (the
order the kernel's `break` reads). Eq. 20 and the scatter stay host-side,
`eri.j_coulomb.staged._transform`, as `jgen` does.

THE TIMING MODE (#146): `run(..., runs=REPS, warmup=WARMUP)`, every run
kept (`witness="each"`). One timed build is the region
`measured_store.BUILD_TIMED_REGION` names -- the per-run zeros, every
launch, one synchronize and the read-back, after a synchronize -- because
the sequence is run with runner.h's `timed_readback` (the read-back inside
each run's host clock) and `launch_events=False` (only the run's two
events, nothing between the launches): `reps` gives each build's wall,
device, read-back and event times. The uploads are timed once, apart
(`Sequenced.upload_ns`), and eq. 20 is host-side and not in it.

THE CHAIN (#157): every arithmetic step of `J_mn` a generated kernel on the
card, in THIS sequence -- not a second one. Two processes, because the plan
the driver is generated from is keyed on the DQ kernel's own `Dmax`:

  1. `compile_dq` -- every ket pair class's `Target.DQ` kernel (`dq_l_l`)
     on the shared runner (#659), one launch per class over all its
     records, `D_q` and `Dmax` read back:
     the chain's one round-trip (j_engine.md, "The round-trip"), as in
     `eri.j_coulomb.drive.coulomb_matrix_chain`, whose plan builder (`_chain_plan`)
     then keys and sorts the kets on the kernel's `Dmax`;
  2. `driver(plan, ..., chain=basis)` -- milestone 6's sequence, unchanged
     in its uploads and `[p|q]` launches, staged with the DQ kernel's `D_q` /
     `Dmax` through `eri.j_coulomb.staged`'s `ket_data=` hook (`stage(ket_data=)`,
     its plan-key check intact), and EXTENDED by one `Target.JAB` launch
     per bra pair class (`jab_tail`) after every `[p|q]` launch, on the same
     stream: each reads its bra class's `J<lp>` WHERE IT LIES ON THE CARD
     (no `J_p` comes back to be re-uploaded) and adds into one `Jmn`,
     zeroed by the first JAB launch and read back per run. `J_p` is read
     back too, as the witness of the run-to-run bound.

ANOTHER CHAIN STEP (#619): `driver(plan, ..., tail=Tail(...))` puts a step
that is not JAB in the same place -- the Coulomb gradient's `gjab`,
reading the raised plan's `J_p` where it lies (test/grad_driver.py,
`gjab_tail`). A `Tail` brings its kernels, its input buffers, its launches
(each `Jp` the buffer `jp_buffer(lp)`) and its output, read back per run
after the `J_p` buffers; `chain=` and `tail=` are exclusive.

ONE JAB UNIT ALONE (#157's part 1): `compile_jab`, `jab_step`, `run_jab`
-- one runner per unit over a given `J_p`, each run its own process.

THE EXCHANGE MATRIX (#206): `k_driver` is the same pattern for a K launch
plan (`launch_k.KLaunchPlan`, #204). Its section below, "K on the card",
states the layout; `k_stage` packs the input, `k_buffers` cuts it into the
sequence's buffers, `k_run` runs it, and `k_scatter` is the host-side sum
of the per-record blocks into `K_mn`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from conftest import Toolchain
from harness_driver import (  # noqa: F401 -- the compile path, re-exported (#659)
    FLAGS_VAR,
    Program,
    Ref,
    Sequenced,
    Step,
    Zeros,
    build_runner,
    chain_runners,
    compile_checked,
    compile_command,
    generated_warnings,
    run_case,
    run_chain,
    run_sequence,
    runner_program,
    units_runner,
)
from harness_driver import emit_unit as _unit
from test_kernel_device import BACKEND

from felizjueves.ao.density import checked_density, epair
from felizjueves.ao.shell import Basis, n_cartesians
from felizjueves.eri.j_coulomb.kernel_dq import PARAMS as DQ_PARAMS
from felizjueves.eri.j_coulomb.kernel_dq import build_dq_kernel, dq_shape
from felizjueves.eri.j_coulomb.kernel_dq import launch_shape as dq_launch_shape
from felizjueves.eri.j_coulomb.kernel_jab import build_jab_kernel, jab_launch_shape
from felizjueves.eri.j_coulomb.launch import LaunchPlan, class_spec
from felizjueves.eri.j_coulomb.staged import (
    KetDataHook,
    _accumulating,
    _bra_buffers,
    _dglobal,
    _ket_buffers,
    _kets_checked,
    _transform,
)
from felizjueves.eri.k_exchange.kernel_k import PARAMS as K_PARAMS
from felizjueves.eri.k_exchange.kernel_k import launch_shape as k_launch_shape
from felizjueves.eri.k_exchange.launch_k import KLaunchPlan
from felizjueves.eri.launch_common import PairRec
from felizjueves.ir import AtomicAdd, For, If, Kernel, Note, Stmt, walk
from felizjueves.ir.passes import atomize
from felizjueves.mmd.kernel import bra_functions, ket_functions, launch_shape
from felizjueves.performance import measured

RUNS = 2  # whole-plan runs per process: the run-to-run witness
# The kernel's signature, in the order every launch below passes it.
PARAMS = ("n_bra", "n_ket", "tau", "P", "zeta", "Ip", "Q", "eta", "Iq",
          "Dmax", "D", "J")


@dataclass(frozen=True)
class Launch:
    """One batch kernel of one class, into `J_p[lp]`."""

    lp: int
    lq: int
    kernel: Kernel


@dataclass(frozen=True)
class Tail:
    """A chain step after every `[p|q]` launch, on the same stream, reading
    each bra class's `J_p` where it lies on the card (#157's JAB, #619's
    `gjab`): its kernels (one unit each), its input buffers, its launches
    -- each `Jp` argument the buffer `jp_buffer(lp)`, and the first
    launch's `zero` its output -- and that output's name and shape, read
    back every run after the `J_p` buffers."""

    kernels: tuple[Kernel, ...]
    buffers: Mapping[str, np.ndarray]
    steps: tuple[Step, ...]
    out: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Driver:
    """The runner over every unit of a plan, and the sequence it launches:
    what `compile_checked` builds (`backend`, `path`, `units`), and the
    layout `stage` / `run` / `read_back` follow."""

    backend: str
    source: str
    path: Path  # the runner's own source
    units: tuple[Path, ...]  # every emitted unit, launch order, then headers
    bra: tuple[tuple[int, int, int], ...]  # (lp, n_bra, n_p), plan order
    ket: tuple[tuple[int, int, int], ...]  # (lq, n_ket, n_q), plan order
    launches: tuple[Launch, ...]  # the [p|q] launches, plan order
    tau: float
    steps: tuple[Step, ...]  # the whole sequence: [p|q], then the tail's
    tail: Tail | None = None

    @property
    def writers(self) -> Counter[int]:
        """Kernels writing into each bra class's `J_p` (atomics per element)."""
        return Counter(x.lp for x in self.launches)

    @property
    def outs(self) -> list[str]:
        """The buffers read back every run: each `J_p`, then the tail's."""
        return [jp_buffer(lp) for lp, _, _ in self.bra] + (
            [self.tail.out] if self.tail else [])


def jp_buffer(lp: int) -> str:
    """Bra class `lp`'s `J_p` buffer, by name (a tail's `Jp` argument)."""
    return f"J{lp}"


def _shapes(plan: LaunchPlan):
    bra: dict[int, tuple[int, int]] = {}
    ket: dict[int, tuple[int, int]] = {}
    for cl in plan:
        spec = class_spec(cl.lp, cl.lq, cl.n_batches)
        n_p, n_q = len(bra_functions(spec)), len(ket_functions(spec))
        if bra.setdefault(cl.lp, (len(cl.bras), n_p)) != (len(cl.bras), n_p):
            raise ValueError(f"bra class {cl.lp}: two shapes in one plan")
        if ket.setdefault(cl.lq, (len(cl.kets), n_q)) != (len(cl.kets), n_q):
            raise ValueError(f"ket class {cl.lq}: two shapes in one plan")
    return ([(lp, *bra[lp]) for lp in bra], [(lq, *ket[lq]) for lq in ket])


def bra_rows(plan: LaunchPlan) -> dict[int, int]:
    """Bra records per bra class `lp`: the rows of its `J_p` (a tail's
    `n_rows`)."""
    return {cl.lp: len(cl.bras) for cl in plan}


def driver(plan: LaunchPlan, backend: str, workdir: Path, *,
           chain: Basis | None = None, tail: Tail | None = None) -> Driver:
    """Emit every kernel of `plan` for `backend`, the runner over them, and
    the sequence that launches them in the plan's shape (module
    docstring). With `chain=` (the plan's basis, #157) the sequence also
    launches every bra pair class's `Target.JAB` kernel after the `[p|q]`
    launches (`jab_tail`), reading each `J_p` buffer where it lies on the
    card, and reads back `J_mn` ("THE CHAIN"). `tail=` is another step in
    its place (`Tail`, #619)."""
    if chain is not None and tail is not None:
        raise ValueError("chain= and tail= are the same slot")
    if chain is not None:
        tail = jab_tail(plan, chain)
    _, arch, _ = BACKEND[backend]
    bra, ket = _shapes(plan)
    nb = {lp: n for lp, n, _ in bra}
    nk = {lq: n for lq, n, _ in ket}
    lanes = {launch_shape(n, arch.wavefront)[1][1] for n in nb.values()}
    assert lanes == {arch.wavefront}, lanes
    launches, steps, kernels = [], [], []
    for cl in plan:
        n, m = nb[cl.lp], nk[cl.lq]
        b, k = f"bra{cl.lp}", f"ket{cl.lq}"
        for kern in map(_accumulating, cl.kernels):
            names = tuple(p.name for p in kern.params)
            if names != PARAMS:
                raise ValueError(f"{kern.name}: signature {names}, the driver "
                                 f"passes {PARAMS}")
            launches.append(Launch(cl.lp, cl.lq, kern))
            kernels.append(kern)
            steps.append(Step(kern, n, (1, arch.wavefront), {
                "n_bra": n, "n_ket": m, "tau": plan.tau,
                "P": Ref(b), "zeta": Ref(b, 3 * n), "Ip": Ref(b, 4 * n),
                "Q": Ref(k), "eta": Ref(k, 3 * m), "Iq": Ref(k, 4 * m),
                "Dmax": Ref(k, 5 * m), "D": Ref(k, 6 * m),
                "J": jp_buffer(cl.lp)},
                shared_bytes=measured.jsh_slots(kern) * arch.wavefront * 8))
    # every J_p zeroed ONCE per run, inside its clock, before any launch
    steps[0] = replace(steps[0], zero=tuple(jp_buffer(lp) for lp, _, _ in bra))
    if tail is not None:
        clash = set(tail.buffers) & ({f"bra{lp}" for lp in nb}
                                     | {f"ket{lq}" for lq in nk}
                                     | {jp_buffer(lp) for lp in nb})
        if clash or tail.out in tail.buffers:
            raise ValueError(f"the tail's buffers clash: {sorted(clash)}")
        kernels += tail.kernels
        steps += tail.steps
    seen: dict[str, Kernel] = {}
    for kern in kernels:
        if seen.setdefault(kern.name, kern) != kern:
            raise ValueError(f"two kernels named {kern.name}")
    units = [(_unit(kern, backend, workdir), [kern.name])
             for kern in seen.values()]
    prog = units_runner(backend, units, workdir, "driver")
    return Driver(backend, prog.source, prog.path, prog.units, tuple(bra),
                  tuple(ket), tuple(launches), plan.tau, tuple(steps), tail)


def stage(plan: LaunchPlan, basis: Basis, D: object, drv: Driver, *,
          ket_data: KetDataHook | None = None) -> dict[str, np.ndarray | Zeros]:
    """The sequence's buffers: per bra class `bra<lp>` = `P zeta Ip`, per
    ket class `ket<lq>` = `Q eta Iq Dmax D` -- ONE upload each, built by
    `eri.j_coulomb.staged`'s own buffer builders -- a zeroed `J<lp>` per
    bra class, and the tail's. `ket_data=` is `eri.j_coulomb.staged`'s hook
    (#155): a ket class's `Q eta D Dmax` from it -- the DQ kernel's, in the
    chain -- instead of the host pre-contraction, through the same
    `_kets_checked` (every `Iq * Dmax` its plan key, bit for bit)."""
    rows = checked_density(basis, D)
    dglobal = None if ket_data is not None else _dglobal(plan, basis, rows)
    bras, kets = {}, {}
    for cl in plan:
        if cl.lp not in bras:
            bras[cl.lp] = _bra_buffers(basis, cl.bras)
        if cl.lq not in kets:
            n_q = dict((lq, q) for lq, _, q in drv.ket)[cl.lq]
            kets[cl.lq] = (
                _ket_buffers(basis, rows, cl.kets, n_q, dglobal)
                if ket_data is None
                else _kets_checked(cl.kets, n_q, ket_data(cl.kets, n_q)))
        elif kets[cl.lq].recs != cl.kets:
            raise ValueError(f"ket class {cl.lq}: two ket lists, one upload")

    def flat(*parts) -> np.ndarray:
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel()
                               for p in parts])

    out: dict[str, np.ndarray | Zeros] = {}
    for lp, n, _ in drv.bra:
        b = bras[lp]
        out[f"bra{lp}"] = flat(b.P, b.zeta, b.Ip)
        assert out[f"bra{lp}"].size == 5 * n
    for lq, n, q in drv.ket:
        k = kets[lq]
        out[f"ket{lq}"] = flat(k.Q, k.eta, k.Iq, k.Dmax, k.D)
        assert out[f"ket{lq}"].size == n * (6 + q)
    for lp, n, p in drv.bra:
        out[jp_buffer(lp)] = Zeros((n, p))
    if drv.tail is not None:
        out |= dict(drv.tail.buffers)
        out[drv.tail.out] = Zeros(drv.tail.shape)
    return out


def run(drv: Driver, binary: Path, buffers: Mapping[str, object],
        workdir: Path, *, runs: int = RUNS, warmup: int = 0) -> Sequenced:
    """`warmup` untimed builds and `runs` timed ones, in ONE process (the
    runner, compiled from `drv`): every build the region
    `measured_store.BUILD_TIMED_REGION` names -- the `J_p` zeros (and the
    tail's), every launch, one synchronize, the read-back of every out
    buffer -- with only the run's two events (`launch_events=False`).
    Every timed build's outputs are kept (`witness="each"`: the kernels add
    with fp64 atomics, and the test bounds the spread). The caller holds
    the card (its device fixture)."""
    ran = run_sequence(binary, workdir, buffers, drv.steps, outs=drv.outs,
                       runs=runs, warmup=warmup, witness="each",
                       timed_readback=True, launch_events=False)
    assert ran.launched == [s.kernel.name for s in drv.steps], ran.launched
    return ran


def read_back(drv: Driver, ran: Sequenced) -> list[dict[int, np.ndarray]]:
    """`[run][lp]` -> that run's `J_p` of bra class `lp`."""
    per = {lp: ran.load(jp_buffer(lp)) for lp, _, _ in drv.bra}
    n = {len(v) for v in per.values()}
    assert len(n) == 1, n
    return [{lp: per[lp][r] for lp in per} for r in range(n.pop())]


def read_back_tail(drv: Driver, ran: Sequenced) -> list[np.ndarray]:
    """`[run]` -> that run's tail output (`J_mn` in the chain, #157)."""
    assert drv.tail is not None, "no tail"
    return ran.load(drv.tail.out)


def reps(ran: Sequenced) -> list[tuple[int, int, int, float]]:
    """Per timed build, `(wall_ns, device_ns, readback_ns, events_ns)`:
    `measured_store.BUILD_TIMED_REGION`'s (a)-(d), its (a)-(c), its (d),
    and the events from before (a) to after (b) -- the runner's
    `wall_ns + readback_ns`, `wall_ns`, `readback_ns`, `events_ns`, which
    share their clock readings, so the sum is the one interval."""
    assert len(ran.readback_ns) == len(ran.wall_ns), "not timed_readback"
    return [(w + r, w, r, e) for w, r, e in zip(
        ran.wall_ns, ran.readback_ns, ran.events_ns, strict=True)]


# --- the J chain on the card (#157) ----------------------------------------

DQ_LANES = 32  # one thread per record (kernel_dq.launch_shape); not numerical
JAB_THREADS = 32  # one thread per record (kernel_jab.jab_launch_shape)
JAB_PARAMS = ("n_rec", "n_ao", "n_rows", "zeta_ab", "XPA", "XPB", "s", "off_a",
              "off_b", "row", "Jp", "J")
DQ_DOUBLES = 10  # per record: eta XPA(3) XPB(3) s w pair
DQ_POISON = np.frombuffer(b"\xff" * 8, dtype="<f8")[0]  # a NaN, every byte 0xff
JAB_DOUBLES = 11  # per record: zeta_ab XPA(3) XPB(3) s off_a off_b row


def jab_rows(basis: Basis, members) -> tuple[np.ndarray, np.ndarray]:
    """`(data, ints)` for the records of `members`, each `(a, b, start)`: a
    shell pair and its first row of the `J_p` buffer, records in
    `primitive_pairs` order -- `eri.j_coulomb.drive.KernelJAB` / `_jab_launches`' own
    `zeta_ab`, `X_PA = P - A`, `X_PB = P - B`, `s`, offsets and rows."""
    shells, off = basis.shells, basis.offsets
    data, ints = [], []
    for a, b, start in members:
        ep = epair(shells[a], shells[b])
        for k, (P, zeta, s) in enumerate(ep.recs):
            data.append([zeta, *(P[d] - ep.A[d] for d in range(3)),
                         *(P[d] - ep.B[d] for d in range(3)), s])
            ints.append([off[a], off[b], start + k])
    return (np.array(data, dtype=np.float64).reshape(-1, 8),
            np.array(ints, dtype=np.float64).reshape(-1, 3))


def jab_records(plan: LaunchPlan, basis: Basis):
    """Per bra pair class `(la, lb)`, sorted: `(la, lb, lp, data, ints)` over
    every record of every bra shell pair of the class -- the grouping of
    `eri.j_coulomb.drive._jab_launches` (one launch per class, rows of `J_p[lp]` from
    the plan's bra blocks)."""
    shells = basis.shells
    blocks: dict[int, tuple] = {}
    for cl in plan:
        blocks.setdefault(cl.lp, cl.bra_blocks)
    groups: dict[tuple[int, int], list] = {}
    for lp, blks in blocks.items():
        for a, b, start, stop in blks:
            la, lb = shells[a].l, shells[b].l
            if la + lb != lp:
                raise ValueError(f"bra pair {(a, b)} filed under lp = {lp}")
            if stop - start != len(epair(shells[a], shells[b]).recs):
                raise ValueError(f"bra block {(a, b)}: {stop - start} rows")
            groups.setdefault((la, lb), []).append((a, b, start))
    return [(la, lb, la + lb, *jab_rows(basis, groups[(la, lb)]))
            for la, lb in sorted(groups)]



@dataclass(frozen=True)
class DQClass:
    """One ket pair class's DQ launch: its shell pairs (sweep order), per
    record `eta XPA XPB s w` and `pair`, and `Dcd` per shell pair --
    `eri.j_coulomb.drive.KernelDQ.run`'s arguments, built the same way."""

    lc: int
    ld: int  # ld
    members: tuple[tuple[int, int], ...]
    data: np.ndarray  # (n_rec, 9)
    pair: np.ndarray  # (n_rec,)
    dcd: np.ndarray  # (n_pair, n_cd)

    @property
    def cls(self) -> tuple[int, int]:
        return (self.lc, self.ld)

    @property
    def n_rec(self) -> int:
        return int(self.data.shape[0])


def dq_classes(basis: Basis, D: object) -> list[DQClass]:
    """Every ket pair class of `basis`, sorted, with its unique shell pairs
    `c >= d` in sweep order and the host's weight -- the grouping and the
    arguments of `eri.j_coulomb.drive.coulomb_matrix_chain`'s DQ step."""
    rows = checked_density(basis, D)
    shells, off = basis.shells, basis.offsets
    by: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c in range(len(shells)):
        for d in range(c + 1):
            by.setdefault((shells[c].l, shells[d].l), []).append((c, d))
    out = []
    for cls in sorted(by):
        data, pair, dcd = [], [], []
        for i, (c, d) in enumerate(by[cls]):
            sc, sd = shells[c], shells[d]
            ep = epair(sc, sd)
            w = 1.0 if c == d else 2.0
            dcd.append([float(rows[off[c] + x][off[d] + y])
                        for x in range(sc.n_cart) for y in range(sd.n_cart)])
            for P, zeta, sk in ep.recs:
                data.append([zeta, *(P[k] - ep.A[k] for k in range(3)),
                             *(P[k] - ep.B[k] for k in range(3)), sk, w])
                pair.append(i)
        out.append(DQClass(cls[0], cls[1], tuple(by[cls]),
                           np.array(data, dtype=np.float64),
                           np.array(pair, dtype=np.float64),
                           np.array(dcd, dtype=np.float64)))
    return out


def dq_kernel(dc: DQClass) -> Kernel:
    """The class's `Target.DQ` kernel, its signature checked."""
    kern = build_dq_kernel(dc.cls)
    names = tuple(p.name for p in kern.params)
    if names != DQ_PARAMS:
        raise ValueError(f"{kern.name}: signature {names}")
    return kern


def dq_args(dc: DQClass) -> dict:
    """Every argument of the class's launch by parameter name: per record
    `eta XPA XPB s w pair`, `Dcd` per shell pair, and `D` / `Dmax` POISONED
    (every byte 0xff, a NaN) and read and updated, so a record the kernel
    leaves unwritten reads back non-finite -- the runner restores the poison
    before every run."""
    d = dc.data
    n, (_, n_q) = dc.n_rec, dq_shape(*dc.cls)
    return {"n_rec": n, "n_pair": len(dc.members), "eta": d[:, 0],
            "XPA": d[:, 1:4].copy(), "XPB": d[:, 4:7].copy(), "s": d[:, 7],
            "w": d[:, 8], "pair": as_exact_int(dc.pair),
            "Dcd": np.ascontiguousarray(dc.dcd),
            "D": np.full((n, n_q), DQ_POISON), "Dmax": np.full(n, DQ_POISON)}


def as_exact_int(a) -> np.ndarray:
    """`a` (exact integers held as doubles) as int32, refusing any other."""
    got = np.asarray(a).astype(np.int32)
    if not np.array_equal(got, a):
        raise ValueError("an index is not an exact int32")
    return got


@dataclass(frozen=True)
class DQBinary:
    """Every DQ class's runner, compiled (#659: on the shared runner, one
    runner per class's unit, one launch per class)."""

    classes: tuple[DQClass, ...]
    kernels: tuple[Kernel, ...]
    binaries: tuple[Path, ...]
    workdir: Path

    def run(self, *, runs: int = RUNS, warmup: int = 0, tag: str = "run"
            ) -> tuple[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
                       dict[tuple[int, int], list[float]]]:
        """Per class `(D[n_rec, n_q], Dmax[n_rec])` in record order, and the
        event times of its `runs` timed launches. Every launch is from the
        poison, and the runner fails unless every run is bit-identical to
        the first (plain stores, one owner per record: no atomic). The
        caller holds the card (its device fixture)."""
        out, times = {}, {}
        for c, (dc, kern, binary) in enumerate(zip(
                self.classes, self.kernels, self.binaries, strict=True)):
            grid, block = dq_launch_shape(dc.n_rec, DQ_LANES)
            ran = run_case(binary, kern, self.workdir / f"{tag}{c}", grid,
                           block, dq_args(dc), runs=runs, warmup=warmup)
            out[dc.cls] = (np.load(ran.outs["D"]), np.load(ran.outs["Dmax"]))
            times[dc.cls] = ran.events_ns
        return out, times


def dq_runners(classes: list[DQClass], backend: str, workdir: Path, *,
               units: Mapping[str, Path] | None = None
               ) -> list[tuple[Kernel, Program]]:
    """Per class, its kernel and its runner's source (`units` as `_unit`'s:
    the tier's own files when it ran)."""
    out = []
    for dc in classes:
        kern = dq_kernel(dc)
        unit = _unit(kern, backend, workdir, units)
        out.append((kern, runner_program(backend, unit, [kern.name], workdir)))
    return out


def compile_dq(classes: list[DQClass], backend: str, tc: Toolchain,
               workdir: Path, *, units: Mapping[str, Path] | None = None,
               timeout: float = 1800.0) -> DQBinary:
    """Every class's runner compiled under #92's warning filter."""
    runners = dq_runners(classes, backend, workdir, units=units)
    bins = tuple(build_runner(tc, prog, workdir / f"runner_{c}_{kern.name}",
                              timeout=timeout)
                 for c, (kern, prog) in enumerate(runners))
    return DQBinary(tuple(classes), tuple(k for k, _ in runners), bins,
                    workdir)


def jab_kernel(cls: tuple[int, int]) -> Kernel:
    """The bra pair class's `Target.JAB` kernel, its signature checked."""
    kern = build_jab_kernel(cls)
    names = tuple(p.name for p in kern.params)
    if names != JAB_PARAMS:
        raise ValueError(f"{kern.name}: signature {names}")
    return kern


def jab_step(kern: Kernel, jp: np.ndarray, data: np.ndarray,
             ints: np.ndarray, n_ao: int) -> Step:
    """ONE bra pair class's JAB launch over a given `J_p` (#157's part 1):
    `data` per record `zeta_ab XPA XPB s`, `ints` `off_a off_b row` (exact
    as doubles), `J` carried (`harness_driver.run_chain`)."""
    jp = np.ascontiguousarray(jp, dtype=np.float64)
    n = int(data.shape[0])
    grid, block = jab_launch_shape(n, JAB_THREADS)
    return Step(kern, grid, block, {
        "n_rec": n, "n_ao": n_ao, "n_rows": int(jp.shape[0]),
        "zeta_ab": data[:, 0], "XPA": data[:, 1:4].copy(),
        "XPB": data[:, 4:7].copy(), "s": data[:, 7],
        "off_a": as_exact_int(ints[:, 0]), "off_b": as_exact_int(ints[:, 1]),
        "row": as_exact_int(ints[:, 2]), "Jp": jp})


def compile_jab(cls: tuple[int, int], backend: str, tc: Toolchain,
                workdir: Path, *, units: Mapping[str, Path] | None = None,
                timeout: float = 1800.0) -> tuple[Kernel, Path]:
    """The class's runner compiled under #92's warning filter."""
    kern = jab_kernel(cls)
    (prog,) = chain_runners([kern], backend, workdir, units=units).values()
    return kern, build_runner(tc, prog, workdir / f"runner_{kern.name}",
                              timeout=timeout)


def run_jab(binary: Path, step: Step, n_ao: int, workdir: Path, *,
            runs: int = RUNS) -> np.ndarray:
    """`[run][n_ao][n_ao]`: `runs` launches, each its own process into a
    zeroed `J` -- the kernel adds a shell pair's records with fp64 atomics,
    so the runs' spread is the test's to bound."""
    return np.array(run_chain({step.kernel.name: binary}, [step], "J",
                              np.zeros((n_ao, n_ao)), workdir, runs=runs))


def transform(basis: Basis, plan: LaunchPlan, jps: dict[int, np.ndarray]):
    """Eq. 20 and the mirrored scatter: `eri.j_coulomb.staged._transform`, host-side."""
    blocks = {}
    for cl in plan:
        blocks.setdefault(cl.lp, cl.bra_blocks)
    return np.array(_transform(basis, blocks, jps))


def jab_tail(plan: LaunchPlan, basis: Basis) -> Tail:
    """The chain's JAB step (#157): one `Target.JAB` launch per bra pair
    class (`jab_records`), each reading `J_p[lp]` where it lies on the card
    and adding into one `J_mn`, zeroed by the first JAB launch. Per class
    `jd<c>` = `zeta_ab XPA XPB s` and `ji<c>` = `off_a off_b row` (int32),
    one upload each."""
    rows = bra_rows(plan)
    kernels, steps, buffers = [], [], {}
    for c, (la, lb, lp, data, ints) in enumerate(jab_records(plan, basis)):
        kern = jab_kernel((la, lb))
        n = int(data.shape[0])
        d, i = f"jd{c}", f"ji{c}"
        buffers[d] = np.concatenate([data[:, 0], data[:, 1:4].ravel(),
                                     data[:, 4:7].ravel(), data[:, 7]])
        buffers[i] = as_exact_int(np.concatenate(
            [ints[:, 0], ints[:, 1], ints[:, 2]]))
        grid, (threads, _) = jab_launch_shape(n, JAB_THREADS)
        kernels.append(kern)
        steps.append(Step(kern, grid, threads, {
            "n_rec": n, "n_ao": basis.n_ao, "n_rows": rows[lp],
            "zeta_ab": Ref(d), "XPA": Ref(d, n), "XPB": Ref(d, 4 * n),
            "s": Ref(d, 7 * n), "off_a": Ref(i), "off_b": Ref(i, n),
            "row": Ref(i, 2 * n), "Jp": jp_buffer(lp), "J": "Jmn"}))
    steps[0] = replace(steps[0], zero=("Jmn",))
    return Tail(tuple(kernels), buffers, tuple(steps), "Jmn",
                (basis.n_ao, basis.n_ao))


# --- K on the card (#206) ----------------------------------------------------
#
# The whole-molecule exchange build from a K launch plan (`launch_k`, #204),
# with the kernels `kernel_k.build_k_kernels` builds (#198, #203) injected
# into the plan as `build=`. Harness, as the J driver above is: the kernels
# are `emit/cuda.py` / `emit/hip.py`'s text, unchanged, in one runner, and
# the sequence (`KDriver.steps`) is written here (#659: its own template,
# `K_DRIVER`, is gone). It has no timing mode: `RUNS` runs, every one kept.
#
# WHAT A K LAUNCH IS. The K kernel writes one `K[a, b]` block per bra record,
# `K[n_bra, n_a n_b]`, with `b` taken from the KET (`kernel_k`'s docstring:
# kets of several `d` shells may share a launch, but they must share the one
# `b` shell whose column of `K` the block is -- "the launch plan's business").
# A `KClassLaunch` carries every ket record of its ket class, every `b` shell
# at once, sorted by `I_bd`. So the driver launches each of the class's
# kernels once PER `b` SHELL, over that shell's kets: a SUBSEQUENCE of the
# class's sorted list, in the same order, so still sorted descending by `Iq`
# and the kernel's `break` stays sound.
#
# THE BUFFERS, uploaded once before the first run:
#
#   per bra pair class (la, lc)  one upload of its records in pair order:
#                                P(3n) zeta_ac(n) XPA(3n) XPC(3n) s_ac(n)
#                                Ip(n), and `dc` (n ints) apart;
#   per ket pair class (lb, ld)  one upload, its records grouped by b shell
#                                (b ascending, each group in the plan's
#                                order): Q(3n) eta(n) XQB(3n) XQD(3n)
#                                s_bd(n) Iq(n), and `dd` (n ints); a launch
#                                passes `Ref`s at its group's first ket;
#   per density class (lc, ld)   `Dcd[n_sh * n_sh, n_c n_d]`, row
#                                `c * n_sh + d` the (c, d) shell block of
#                                D (zero for a shell of another l, never
#                                read): so `dc = c * n_sh` and `dd = d`, and
#                                the kernel's `dc[bra] + dd[k]` names it.
#
# ONE K BUFFER, zeroed once per run (by the first launch) and read back once
# per run, holding one
# block per (bra pair class, b shell): `n_bra x n_a n_b`. Every kernel whose
# kets have that `b` ADDS into it -- ket classes of every `ld`, and every
# batch of a split class -- so a one-batch kernel's plain write is
# `passes.atomize`d here (`k_accumulating`), as `eri.j_coulomb.staged._accumulating`
# does for J; a split class's kernels are atomic already (#203).
#
# THE SCATTER is host-side (`k_scatter`): `K_mn[a][b] = SUM` of the rows of
# every bra record whose first shell is `a`, over the blocks of shell `b`,
# each element summed with `math.fsum` -- the launch plan's `bra_blocks`
# (`(a, c, start, stop)`) name the rows.

K_HARNESS_WRITE = (
    "atomic write, by lane 0 only: the K launch harness (test/plan_driver.py, "
    "#206) adds every kernel whose kets share one b shell -- ket classes of "
    "every ld, and every batch -- into ONE (bra pair class, b) block of one "
    "zeroed K buffer, so the write must accumulate; on a zeroed block the "
    "first add is exact. The lane-0 guard stays: every lane holds the warp sum")
K_BRA_DOUBLES = 12  # per bra record: P(3) zeta_ac XPA(3) XPC(3) s_ac Ip
K_KET_DOUBLES = 12  # per ket record: Q(3) eta XQB(3) XQD(3) s_bd Iq
_K_PLAIN = "plain write, NOT atomic"


def _renote(body: tuple[Stmt, ...], old: str, new: str) -> tuple[Stmt, ...]:
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Note) and old in s.prov:
            s = Note(new)
        elif isinstance(s, If):
            s = If(s.cond, _renote(s.then, old, new),
                   _renote(s.orelse, old, new))
        elif isinstance(s, For):
            s = replace(s, body=_renote(s.body, old, new))
        out.append(s)
    return tuple(out)


def k_accumulating(k: Kernel) -> Kernel:
    """A K kernel that ADDS into `K`: a split class's kernels already do; a
    one-batch kernel's plain write is atomized, its write note restated for
    this harness (`K_HARNESS_WRITE`) rather than left as `passes`' J note."""
    if any(isinstance(s, AtomicAdd) for s in walk(k.body)):
        return k
    return atomize(replace(k, body=_renote(k.body, _K_PLAIN, K_HARNESS_WRITE)),
                   array="K")


@dataclass(frozen=True)
class KBlock:
    """One (bra pair class, b shell) block of the K buffer: one row per bra
    record of the class, `n_a n_b` columns, at `offset` doubles."""

    bra: tuple[int, int]  # (la, lc)
    b: int  # the ket shell whose column of K the block is
    offset: int
    n_bra: int
    n_ab: int

    @property
    def size(self) -> int:
        return self.n_bra * self.n_ab


@dataclass(frozen=True)
class KLaunch:
    """One launch: `kernel` of class `am` (batch `batch`), over the `n_ket`
    kets of shell `b` from `start` in its ket class's upload, adding into
    block `block`."""

    am: tuple[int, int, int, int]
    batch: int
    b: int
    start: int
    n_ket: int
    block: int
    kernel: Kernel


@dataclass(frozen=True)
class KDriver:
    """The runner over every unit of a K plan, and the sequence it launches
    ("K on the card"): `compile_checked` builds it (`backend`, `path`,
    `units`)."""

    backend: str
    source: str
    path: Path
    units: tuple[Path, ...]
    tau: float
    dm: float
    n_sh: int
    bra: tuple[tuple[tuple[int, int], tuple[PairRec, ...]], ...]
    ket: tuple[tuple[tuple[int, int], tuple[PairRec, ...]], ...]  # by b
    dcd: tuple[tuple[int, int], ...]  # (lc, ld) per Dcd upload
    blocks: tuple[KBlock, ...]
    launches: tuple[KLaunch, ...]
    steps: tuple[Step, ...] = ()  # one per launch, the first zeroing K

    @property
    def writers(self) -> Counter[int]:
        """Launches adding into each block (atomic adds per element)."""
        return Counter(x.block for x in self.launches)

    @property
    def n_k(self) -> int:
        return sum(b.size for b in self.blocks)

    def dcd_doubles(self, lc: int, ld: int) -> int:
        return self.n_sh * self.n_sh * n_cartesians(lc) * n_cartesians(ld)

    @property
    def n_input(self) -> int:
        return 2 + sum((K_BRA_DOUBLES + 1) * len(r) for _, r in self.bra) + sum(
            (K_KET_DOUBLES + 1) * len(r) for _, r in self.ket) + sum(
            self.dcd_doubles(*x) for x in self.dcd)


def _k_groups(plan: KLaunchPlan):
    """Per bra class its records, per ket class its records grouped by b
    shell (each group in the plan's order) and the groups' `(start, n)`."""
    bra: dict[tuple[int, int], tuple[PairRec, ...]] = {}
    ket: dict[tuple[int, int], tuple[PairRec, ...]] = {}
    for cl in plan:
        bls, kls = (cl.am[0], cl.am[1]), (cl.am[2], cl.am[3])
        if bra.setdefault(bls, cl.bras) != cl.bras:
            raise ValueError(f"bra class {bls}: two record lists, one upload")
        if ket.setdefault(kls, cl.kets) != cl.kets:
            raise ValueError(f"ket class {kls}: two record lists, one upload")
    order: dict[tuple[int, int], tuple[PairRec, ...]] = {}
    segs: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    for kls, recs in ket.items():
        out: list[PairRec] = []
        segs[kls] = {}
        for b in sorted({r.i for r in recs}):
            grp = [r for r in recs if r.i == b]  # a subsequence: still sorted
            segs[kls][b] = (len(out), len(grp))
            out.extend(grp)
        order[kls] = tuple(out)
    return bra, order, segs


def k_driver(plan: KLaunchPlan, basis: Basis, backend: str,
             workdir: Path) -> KDriver:
    """Emit every kernel of the K plan for `backend`, the runner over them,
    and the sequence launching them per b shell into one K buffer ("K on
    the card"): buffers `kb<i>` / `kbi<i>` per bra class (12 n doubles,
    `dc` as int32), `kk<j>` / `kki<j>` per ket class, `kd<e>` per density
    class, and `K`, zeroed by the first launch."""
    _, arch, _ = BACKEND[backend]
    shells = basis.shells
    n_sh = len(shells)
    bra, ket, segs = _k_groups(plan)
    bra_cls, ket_cls = sorted(bra), sorted(ket)
    dcd_cls = sorted({(cl.am[1], cl.am[3]) for cl in plan})
    bslot = {x: i for i, x in enumerate(bra_cls)}
    kslot = {x: i for i, x in enumerate(ket_cls)}
    eslot = {x: i for i, x in enumerate(dcd_cls)}
    blocks: list[KBlock] = []
    bidx: dict[tuple[tuple[int, int], int], int] = {}
    launches, steps, units = [], [], []
    for cl in plan:
        if not cl.kernels:
            raise ValueError(f"K class {cl.am} carries no kernels: plan it "
                             "with build=")
        la, lc, lb, ld = cl.am
        bls, kls = (la, lc), (lb, ld)
        for b in segs[kls]:
            if (bls, b) not in bidx:
                if shells[b].l != lb:
                    raise ValueError(f"ket shell {b} filed under lb = {lb}")
                n_ab = n_cartesians(la) * n_cartesians(lb)
                bidx[bls, b] = len(blocks)
                blocks.append(KBlock(bls, b, sum(x.size for x in blocks),
                                     len(bra[bls]), n_ab))
        i, j, e = bslot[bls], kslot[kls], eslot[(lc, ld)]
        n, nk = len(bra[bls]), len(ket[kls])
        bp, bi, kp, ki = f"kb{i}", f"kbi{i}", f"kk{j}", f"kki{j}"
        for bt, k in enumerate(cl.kernels):
            kern = k_accumulating(k)
            names = tuple(p.name for p in kern.params)
            if names != K_PARAMS:
                raise ValueError(f"{kern.name}: signature {names}, the driver "
                                 f"passes {K_PARAMS}")
            units.append((_unit(kern, backend, workdir), [kern.name]))
            grid, (_, lanes) = k_launch_shape(n, arch.wavefront)
            assert grid == n and lanes == arch.wavefront, (grid, lanes)
            shared = measured.jsh_slots(kern) * lanes * 8
            for b, (s, m) in segs[kls].items():
                q = bidx[bls, b]
                launches.append(KLaunch(cl.am, bt, b, s, m, q, kern))
                steps.append(Step(kern, n, (1, lanes), {
                    "n_bra": n, "n_ket": m, "n_dc": n_sh * n_sh,
                    "tau": plan.tau, "maxD": plan.dm,
                    "P": Ref(bp), "zeta_ac": Ref(bp, 3 * n),
                    "XPA": Ref(bp, 4 * n), "XPC": Ref(bp, 7 * n),
                    "s_ac": Ref(bp, 10 * n), "Ip": Ref(bp, 11 * n),
                    "dc": Ref(bi), "Q": Ref(kp, 3 * s),
                    "eta": Ref(kp, 3 * nk + s), "XQB": Ref(kp, 4 * nk + 3 * s),
                    "XQD": Ref(kp, 7 * nk + 3 * s),
                    "s_bd": Ref(kp, 10 * nk + s), "Iq": Ref(kp, 11 * nk + s),
                    "dd": Ref(ki, s), "Dcd": Ref(f"kd{e}"),
                    "K": Ref("K", blocks[q].offset)}, shared_bytes=shared))
    steps[0] = replace(steps[0], zero=("K",))
    prog = units_runner(backend, units, workdir, "k_driver")
    return KDriver(backend, prog.source, prog.path, prog.units, plan.tau,
                   plan.dm, n_sh, tuple((x, bra[x]) for x in bra_cls),
                   tuple((x, ket[x]) for x in ket_cls), tuple(dcd_cls),
                   tuple(blocks), tuple(launches), tuple(steps))


def _k_rows(basis: Basis, recs: tuple[PairRec, ...], *,
            bra: bool) -> list[np.ndarray]:
    """One class's upload, in the order `k_buffers` cuts it: P(3n) zeta(n)
    X_PA(3n) X_PB(3n) s(n) I(n), then the n ints (`dc = c * n_sh` on the
    bra, `dd = d` on the ket) -- `test_kernel_k.k_args`' values: `X_PA =
    P - A`, `X_PB = P - B` of `epair(shells[i], shells[j])`, `s` its
    dressing, `I` the plan's bound `G`."""
    shells, n_sh = basis.shells, len(basis.shells)
    eps: dict[tuple[int, int], object] = {}
    data, ints = [], []
    for r in recs:
        ep = eps.get((r.i, r.j))
        if ep is None:
            ep = eps[r.i, r.j] = epair(shells[r.i], shells[r.j])
        P, zeta, s = ep.recs[r.prim]
        data.append([*P, zeta, *(P[d] - ep.A[d] for d in range(3)),
                     *(P[d] - ep.B[d] for d in range(3)), s, r.G])
        ints.append(r.j * n_sh if bra else r.j)
    a = np.array(data, dtype=np.float64).reshape(-1, K_BRA_DOUBLES)
    return [a[:, 0:3].ravel(), a[:, 3], a[:, 4:7].ravel(), a[:, 7:10].ravel(),
            a[:, 10], a[:, 11], np.array(ints, dtype=np.float64)]


def k_stage(drv: KDriver, basis: Basis, D: object) -> np.ndarray:
    """The packed input: `tau maxD | per bra class (12 + 1) n | per ket
    class (12 + 1) n | per density class Dcd`."""
    rows = np.array(checked_density(basis, D), dtype=np.float64)
    dm = max(abs(float(x)) for x in rows.ravel())
    if dm != drv.dm:
        raise ValueError(f"max|D| = {dm!r} is not the plan's {drv.dm!r}")
    shells, off, n_sh = basis.shells, basis.offsets, len(basis.shells)
    parts: list[np.ndarray] = [np.array([drv.tau, drv.dm])]
    for _, recs in drv.bra:
        parts += _k_rows(basis, recs, bra=True)
    for _, recs in drv.ket:
        parts += _k_rows(basis, recs, bra=False)
    for lc, ld in drv.dcd:
        nc, nd = n_cartesians(lc), n_cartesians(ld)
        blk = np.zeros((n_sh * n_sh, nc * nd))
        for c in range(n_sh):
            for d in range(n_sh):
                if shells[c].l == lc and shells[d].l == ld:
                    blk[c * n_sh + d] = rows[off[c]:off[c] + nc,
                                             off[d]:off[d] + nd].ravel()
        parts.append(blk.ravel())
    out = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
    assert out.size == drv.n_input, (out.size, drv.n_input)
    return out


def k_block(drv: KDriver, buf: np.ndarray, q: int) -> np.ndarray:
    """Block `q` of a K buffer, `[n_bra, n_a n_b]`."""
    b = drv.blocks[q]
    return buf[b.offset:b.offset + b.size].reshape(b.n_bra, b.n_ab)


def k_scatter(drv: KDriver, basis: Basis, buf: np.ndarray) -> np.ndarray:
    """`K_mn` from one run's K buffer: per block `(bra class, b)`, every bra
    record's row (first shell `a`) into `K[a][b]`, each element `fsum`med."""
    shells, off, n = basis.shells, basis.offsets, basis.n_ao
    recs = dict(drv.bra)
    terms: list[list[list[float]]] = [[[] for _ in range(n)] for _ in range(n)]
    for q, blk in enumerate(drv.blocks):
        X = k_block(drv, buf, q)
        nb = shells[blk.b].n_cart
        for row, r in enumerate(recs[blk.bra]):
            na = shells[r.i].n_cart
            assert na * nb == blk.n_ab
            for ia in range(na):
                for ib in range(nb):
                    terms[off[r.i] + ia][off[blk.b] + ib].append(
                        float(X[row, ia * nb + ib]))
    return np.array([[math.fsum(t) for t in r] for r in terms])



def k_buffers(drv: KDriver, staged: np.ndarray) -> dict[str, np.ndarray | Zeros]:
    """`k_stage`'s packed input cut into the sequence's buffers, one upload
    each, as `K_DRIVER`'s `main()` cut its uploads before #659: per bra
    class its 12 n doubles and its `dc` (int32), per ket class the same
    with `dd`, per density class its `Dcd`, and the zeroed `K`. `tau` and
    `maxD` are the launches' scalars, the plan's -- checked equal to the
    staged ones."""
    assert staged.size == drv.n_input, (staged.size, drv.n_input)
    assert (staged[0], staged[1]) == (drv.tau, drv.dm), staged[:2]
    out: dict[str, np.ndarray | Zeros] = {}
    h = 2
    for side, groups in (("kb", drv.bra), ("kk", drv.ket)):
        for c, (_, recs) in enumerate(groups):
            n = len(recs)
            out[f"{side}{c}"] = staged[h:h + 12 * n].copy()
            out[f"{side}i{c}"] = as_exact_int(staged[h + 12 * n:h + 13 * n])
            h += 13 * n
    for e, (lc, ld) in enumerate(drv.dcd):
        m = drv.dcd_doubles(lc, ld)
        out[f"kd{e}"] = staged[h:h + m].copy()
        h += m
    assert h == staged.size
    out["K"] = Zeros((drv.n_k,))
    return out


def k_run(drv: KDriver, binary: Path, staged: np.ndarray, workdir: Path, *,
          runs: int = RUNS) -> list[np.ndarray]:
    """`[run]` -> that run's whole K buffer (every block, in block order):
    `runs` builds in one process, each from a zeroed K, every run kept
    (`witness="each"`: the kernels add with fp64 atomics). The caller
    holds the card (its device fixture)."""
    ran = run_sequence(binary, workdir, k_buffers(drv, staged), drv.steps,
                       outs=["K"], runs=runs, witness="each")
    assert ran.launched == [s.kernel.name for s in drv.steps], ran.launched
    return ran.load("K")
