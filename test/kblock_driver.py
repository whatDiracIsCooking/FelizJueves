"""A device driver for the block exchange kernel and `kernel_k`, one upload (#481).

    kb_driver(plan, basis, backend, workdir)      -> KBDriver (runner + layout)
    kb_stage(drv, basis, D)                       -> np.ndarray, the packed input
    kb_steps(drv, staged, mode)                   -> (buffers, [Step], groups)
    class_args(drv, basis, D, c)                  -> one class's arrays, by name
    k_scatter(drv, basis, buf)                    -> K_mn from kernel_k's scratch
    ClassBinary.run / .timings / .case            -> one class's runner, its clocks

HARNESS, in `test/`, beside `plan_driver.k_driver` (#206), of whose layout
it is a copy cut to `launch_kblock`'s plan. The kernels are the generator's
text, unchanged; the `main()` is the shared runner's (#649; ported by #657),
over every unit (`harness_driver.units_runner`), and each mode a SEQUENCE
whose buffers are the packed input cut as the old `main()` cut its uploads
(per class its doubles `bp<c>` and its ints `ip<c>`, per density class its
`Dcd` `dp<e>`), every argument an offset into them (`Ref`), named from the
kernel's own parameter list.

ONE UPLOAD, TWO MAPPINGS. Per K class the driver uploads ONE copy of its
bra records (grouped by first shell, sorted by `Ip`), its ket records
(grouped by first shell, sorted by `Iq`: `launch_kblock`'s order), its
block table and its density class, and can launch on that same data either

  mode 0  `kernel_kblock`'s kernel, ONE launch per class over every block,
          adding into one zeroed `K[n_ao, n_ao]` -- no scratch, no scatter;
  mode 1  `kernel_k`'s kernel (one batch, its write `passes.atomize`d by
          `plan_driver.k_accumulating`, as #206's harness does), ONE launch
          per (class, b shell) over every bra record of the class and the
          kets of that `b`, adding into the (class, b) block of one zeroed
          scratch `[n_bra, n_a n_b]` -- #206's launch, unchanged; the sum
          over a bra shell's records and the scatter into `K_mn` stay on the
          host (`k_scatter`), as in #206.

`kernel_k` reads its bras by index and needs no order of them, and its kets
per `b` shell are exactly the plan's `b` groups, so the one upload serves
both: the comparison is the mapping and nothing else.

THE TIMING MODE: `ClassBinary.run(mode, ..., warmup=, reps=)`. Every run
zeroes the output (the first launch's `zero`, inside the run's time),
launches every class in plan order and synchronizes; every timed run's
output is kept (`run_classes` sums the first's and the last's). Each launch
is bracketed by a pair of GPU events, and `ClassBinary.timings` gives per
timed run each class's time (the sum of its launches') and the `total`
(memset to last launch). The correctness mode is `WARMUP 0 REPS 2`. Mode
1's host scatter is NOT in its time: the comparison favours `kernel_k` by
exactly that pass.

MODE 2, opt-in (`kb_driver(..., split=granularity)`, #548): each class's
`kernel_kblock_split` set at `R = 1`, every kernel on the block kernel's
arguments, into the same zeroed `K_mn` -- its parameters are the block
kernel's, `Dcd` and `K` with a leading extent of one.

THE ROLLED KERNEL, opt-in (`kb_driver(..., rolled=True)`, #571): MODE 0
launches `kernel_kblock_rolled`'s kernel in the block kernel's place, on the
same records (its `XPA XPC XQB XQD` simply not passed) plus what it reads
instead of forming `E`: each class's `E^ac` / `E^bd` tables per record
(`launch_kblock.e_table`, the ket's signed) appended to the class's doubles
after `dmax`, and its five class tables (`kernel_kblock_rolled.tables`)
appended to the class's ints. `with_k=False` (#569) leaves `kernel_k` out,
MODE 1 then refusing, so a class above `(dd|dd)` compiles one kernel.

THE DENSE-HERMITE KERNEL, opt-in (`kb_driver(..., rolled=True,
dense=True)`, #575): MODE 0 launches
`kernel_kblock_rolled.build_kblock_dense_kernel`'s kernel on the rolled
layout, its `E` tables DENSE (`e_table(..., dense=True)`, zeros kept) and
no class tables.

THE TRANSPOSED KET TABLE, opt-in (`kb_driver(..., rolled=True,
transposed=True)`, #576): MODE 0 launches
`build_kblock_rolled_kernel(..., transposed=True)` (`kbrt_*`), and the ket
`E` block is staged transposed, `Ebd[n_ebd, n_ket]` (entry-major, the
warp's consecutive kets adjacent), `class_args` returning it so; every
other array as `rolled=True` stages it.

THE COLUMN FORM, opt-in (`kb_driver(..., rolled=True, column=True)`,
#583): MODE 0 launches `build_kblock_rolled_kernel(..., column=True)`
(`kbrc_*`) on the rolled layout unchanged, its dynamic shared memory
`column_shared_doubles` (`n_p + n_a`) per lane; its `Kblk[n_a n_b]` is
static, the compiler's to count.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from harness_driver import (
    Ref,
    Sequenced,
    Step,
    Zeros,
    run_sequence,
    sequence_spec,
    units_runner,
    write_case,
)
from plan_driver import k_accumulating
from test_kernel_device import BACKEND

from felizjueves.ao.density import checked_density, epair
from felizjueves.ao.shell import Basis, n_cartesians
from felizjueves.eri.k_exchange.kernel_k import PARAMS as K_PARAMS
from felizjueves.eri.k_exchange.kernel_k import build_k_kernel
from felizjueves.eri.k_exchange.kernel_k import shared_doubles as k_shared_doubles
from felizjueves.eri.k_exchange.kernel_kblock import PARAMS as KB_PARAMS
from felizjueves.eri.k_exchange.kernel_kblock import shared_doubles as kb_shared_doubles
from felizjueves.eri.k_exchange.kernel_kblock_rolled import (
    DENSE_PARAMS,
    TABLES,
    build_kblock_dense_kernel,
    build_kblock_rolled_kernel,
    column_shared_doubles,
    e_pattern,
    tables,
)
from felizjueves.eri.k_exchange.kernel_kblock_rolled import (
    PARAMS as KBR_PARAMS,
)
from felizjueves.eri.k_exchange.kernel_kblock_split import (
    accumulators_per_density,
    build_kblock_split_kernels,
)
from felizjueves.eri.k_exchange.kernel_kblock_split import (
    shared_doubles as split_shared_doubles,
)
from felizjueves.eri.k_exchange.launch_kblock import KBLaunchPlan, e_table
from felizjueves.eri.launch_common import PairRec
from felizjueves.ir import Kernel

REC_DOUBLES = 12  # per record: P(3) zeta XPA(3) XPC(3) s I, then one int
BLK_INTS = 6  # per block: bra0 bra1 ket0 ket1 ko_a ko_b
BLK_DOUBLES = 1  # per block: dmax, the block's density factor (#484)
MODES = {"kb": 0, "k": 1}
SPLIT_MODE = {"kbs": 2}  # opt-in: kb_driver(..., split=granularity), #548


@dataclass(frozen=True)
class KBClass:
    """One class's layout: its records, blocks, scratch block per `b`."""

    am: tuple[int, int, int, int]
    n_bra: int
    n_ket: int
    n_blk: int
    n_ab: int
    kb: Kernel  # the block kernel, or with rolled= the rolled one (#571)
    k: Kernel | None  # None: built with_k=False (#569)
    kbs: tuple[Kernel, ...]  # the split set (#548); () without split=
    segs: tuple[tuple[int, int, int], ...]  # (b, ket0, ket1) per b group
    scratch: tuple[int, ...]  # offset of each segment's scratch block
    n_e: tuple[int, int] = (0, 0)  # rolled=: E entries per bra, per ket
    n_tab: int = 0  # rolled=: the class tables' ints
    # per mode, the class's launches (harness_driver.Step, #657), their
    # arrays named by buffer: `bp<c>` / `ip<c>` the class's doubles / ints,
    # `dp<e>` a density class, `K` or `S` the output
    launches: Mapping[str, tuple[Step, ...]] = field(default_factory=dict)

    @property
    def e_doubles(self) -> int:
        return self.n_e[0] * self.n_bra + self.n_e[1] * self.n_ket

    @property
    def ints(self) -> int:
        return self.n_bra + self.n_ket + BLK_INTS * self.n_blk + self.n_tab

    @property
    def doubles(self) -> int:
        return (REC_DOUBLES + 1) * (self.n_bra + self.n_ket) + (
            (BLK_DOUBLES + BLK_INTS) * self.n_blk) + self.e_doubles + self.n_tab


@dataclass(frozen=True)
class KBDriver:
    """The layout, and the runner over every unit (`source`, `path` and
    `units` are its `Program`'s, so `compile_checked` takes the driver)."""

    backend: str
    source: str
    path: Path
    units: tuple[Path, ...]
    tau: float
    dm: float
    n_sh: int
    n_ao: int
    classes: tuple[KBClass, ...]
    dcd: tuple[tuple[int, int], ...]
    n_scratch: int
    rolled: bool = False
    dense: bool = False  # with rolled: the dense-Hermite kernel (#575)
    transposed: bool = False  # with rolled: Ebd stored [n_ebd, n_ket] (#576)

    def dcd_doubles(self, lc: int, ld: int) -> int:
        return self.n_sh * self.n_sh * n_cartesians(lc) * n_cartesians(ld)

    @property
    def n_input(self) -> int:
        return 2 + sum(c.doubles for c in self.classes) + sum(
            self.dcd_doubles(*x) for x in self.dcd)

    @property
    def n_out(self) -> int:
        """Doubles per run: `K_mn`, then the scratch."""
        return self.n_ao * self.n_ao + self.n_scratch


def _segments(kets: tuple[PairRec, ...]) -> tuple[tuple[int, int, int], ...]:
    out, start = [], 0
    for i in range(1, len(kets) + 1):
        if i == len(kets) or kets[i].i != kets[start].i:
            out.append((kets[start].i, start, i))
            start = i
    return tuple(out)


def _launch(kern: Kernel, args: list, grid: int, shared: int, lanes: int
            ) -> Step:
    """One launch: `args` in the kernel's parameter order, `grid` blocks of
    `(1, lanes)`, `shared` doubles of dynamic shared memory per lane."""
    names = [p.name for p in kern.params]
    if len(names) != len(args):
        raise ValueError(f"{kern.name}: {len(args)} arguments for {names}")
    return Step(kern, grid, (1, lanes), dict(zip(names, args, strict=True)),
                shared_bytes=shared * lanes * 8)


def kb_driver(plan: KBLaunchPlan, basis: Basis, backend: str,
              workdir: Path, *, split: str | None = None,
              rolled: bool = False, dense: bool = False,
              transposed: bool = False, column: bool = False,
              with_k: bool = True) -> KBDriver:
    """Emit both kernels of every class of `plan` (whose `kernel` is the block
    kernel; `kernel_k`'s is built here) and the `main()` that launches either
    mapping on one upload (module docstring). With `split=` a granularity of
    `kernel_kblock_split` (#548), each class's split set (`R = 1`) too, and
    MODE 2 launches it on the block kernel's arguments into `K_mn`. With
    `rolled=True` (#571) MODE 0 is `kernel_kblock_rolled`'s kernel, built
    here, the plan's `kernel` unused; `with_k=False` leaves `kernel_k` out.
    With `dense=True` too (#575) MODE 0 is `build_kblock_dense_kernel`'s
    kernel instead, its `E` dense and no class tables. With
    `transposed=True` too (#576) MODE 0 is the rolled kernel with `Ebd`
    stored `[n_ebd, n_ket]`, staged so. With `column=True` too (#583)
    MODE 0 is the rolled kernel's column form (`kbrc_*`): the same
    parameters and staging, `Ksh` sized `column_shared_doubles` per lane."""
    if column and (not rolled or dense):
        raise ValueError("column= is a form of rolled= (not dense=): "
                         "pass rolled=True")
    if dense and not rolled:
        raise ValueError("dense= is a form of rolled=: pass both")
    if transposed and (not rolled or dense):
        raise ValueError("transposed= is a form of rolled= (not dense=): "
                         "pass rolled=True")
    if rolled and split is not None:
        raise ValueError("rolled= and split= are separate experiments")
    emit, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    n_sh = len(basis.shells)
    n_dc, lanes, n_ao = n_sh * n_sh, arch.wavefront, basis.n_ao
    dcd_cls = sorted({(c.am[1], c.am[3]) for c in plan})
    eslot = {x: i for i, x in enumerate(dcd_cls)}
    classes: list[KBClass] = []
    units: list[tuple[Path, list[str]]] = []
    scratch = 0
    for c, cl in enumerate(plan):
        if cl.kernel is None and not rolled:
            raise ValueError(f"K class {cl.am} carries no kernel: plan it "
                             "with build=")
        kb = (build_kblock_dense_kernel(cl.am) if dense else
              build_kblock_rolled_kernel(cl.am, transposed=transposed,
                                         column=column)
              if rolled else cl.kernel)
        k = k_accumulating(build_k_kernel(cl.am)) if with_k else None
        assert tuple(p.name for p in kb.params) == (
            DENSE_PARAMS if dense else KBR_PARAMS if rolled else KB_PARAMS)
        assert k is None or tuple(p.name for p in k.params) == K_PARAMS
        kbs = (() if split is None else
               build_kblock_split_kernels(cl.am, split, lanes=arch.wavefront))
        for ns, kern in ((f"fj_kb{c}", kb),
                         *(((f"fj_k{c}", k),) if with_k else ()),
                         *((f"fj_kbs{c}_{i}", x) for i, x in enumerate(kbs))):
            path = workdir / f"{ns}_{kern.name}.{ext}"
            path.write_text(emit(kern, arch))
            units.append((path, [kern.name]))
        la, lc, lb, ld = cl.am
        n_ab = n_cartesians(la) * n_cartesians(lb)
        segs = _segments(cl.kets)
        offs = []
        for _ in segs:
            offs.append(scratch)
            scratch += len(cl.bras) * n_ab
        n_e = (_dense_widths(cl.am) if dense else
               (len(e_pattern(la, lc)[1]), len(e_pattern(lb, ld)[1]))
               if rolled else (0, 0))
        n_tab = (sum(map(len, tables(cl.am).values()))
                 if rolled and not dense else 0)
        e = eslot[(lc, ld)]
        nb, nk, ng = len(cl.bras), len(cl.kets), len(cl.blocks)
        bp, ip, dp = f"bp{c}", f"ip{c}", f"dp{e}"
        bra_args = [Ref(bp, 0), Ref(bp, 3 * nb), Ref(bp, 4 * nb),
                    Ref(bp, 7 * nb), Ref(bp, 10 * nb), Ref(bp, 11 * nb),
                    Ref(ip, 0)]

        def ket_args(s: int, bp: str = bp, ip: str = ip, nb: int = nb,
                     nk: int = nk) -> list:
            kk = 12 * nb
            return [Ref(bp, kk + 3 * s), Ref(bp, kk + 3 * nk + s),
                    Ref(bp, kk + 4 * nk + 3 * s), Ref(bp, kk + 7 * nk + 3 * s),
                    Ref(bp, kk + 10 * nk + s), Ref(bp, kk + 11 * nk + s),
                    Ref(ip, nb + s)]

        kb_args = [ng, nb, nk, n_dc, n_ao, plan.tau,
                   *(Ref(ip, nb + nk + j * ng) for j in range(BLK_INTS)),
                   Ref(bp, 12 * (nb + nk)),
                   *bra_args, *ket_args(0), Ref(dp, 0), "K"]
        if rolled:  # #571: no XPA XPC XQB XQD; E and the tables before Dcd
            ea = 12 * (nb + nk) + ng
            tabs, at = [], nb + nk + BLK_INTS * ng
            for n in () if dense else TABLES:
                tabs.append(Ref(ip, at))
                at += len(tables(cl.am)[n])
            kb_args = [*kb_args[:6 + BLK_INTS + 1],
                       *(bra_args[i] for i in (0, 1, 4, 5, 6)),
                       *(ket_args(0)[i] for i in (0, 1, 4, 5, 6)),
                       Ref(bp, ea), Ref(bp, ea + n_e[0] * nb),
                       *tabs, Ref(dp, 0), "K"]
            assert len(kb_args) == len(DENSE_PARAMS if dense else KBR_PARAMS)
        launches = {"kb": (_launch(kb, kb_args, ng, column_shared_doubles(
            cl.am) if column else kb_shared_doubles(cl.am), lanes),)}
        if kbs:
            acc = accumulators_per_density(cl.am, split)
            launches["kbs"] = tuple(
                _launch(x, kb_args, ng, split_shared_doubles(cl.am, acc, 1),
                        lanes) for x in kbs)
        if with_k:
            launches["k"] = tuple(
                _launch(k, [nb, t - s, n_dc, plan.tau, plan.dm, *bra_args,
                            *ket_args(s), Ref(dp, 0), Ref("S", off)],
                        nb, k_shared_doubles(cl.am), lanes)
                for (_, s, t), off in zip(segs, offs, strict=True))
        classes.append(KBClass(cl.am, nb, nk, ng, n_ab, kb, k, kbs, segs,
                               tuple(offs), n_e, n_tab, launches))
    prog = units_runner(backend, units, workdir, "kb_driver")
    return KBDriver(backend, prog.source, prog.path, prog.units, plan.tau,
                    plan.dm, n_sh, n_ao, tuple(classes), tuple(dcd_cls),
                    scratch, rolled, dense, transposed)


def kb_steps(drv: KBDriver, staged: np.ndarray, mode: str
             ) -> tuple[dict, list[Step], list[list[int]]]:
    """Mode `mode`'s sequence on `staged` (`kb_stage`'s): the buffers cut
    from it as the old `main()` cut its uploads -- per class its doubles and
    its ints, per density class its `Dcd` -- every class's launches in plan
    order, the output zeroed by the first (inside the run's time, as the
    old `main()`'s memset was), and each class's launch indices."""
    if mode not in {**MODES, **SPLIT_MODE}:
        raise ValueError(f"mode {mode!r}: one of {sorted({**MODES, **SPLIT_MODE})}")
    assert staged.size == drv.n_input, (staged.size, drv.n_input)
    out = "S" if mode == "k" else "K"
    buffers: dict[str, object] = {
        out: Zeros((max(drv.n_scratch, 1),) if mode == "k"
                   else (drv.n_ao, drv.n_ao))}
    h = 2
    steps: list[Step] = []
    groups: list[list[int]] = []
    for c, x in enumerate(drv.classes):
        nd = x.doubles - x.ints
        buffers[f"bp{c}"] = staged[h:h + nd].astype(np.float64)
        buffers[f"ip{c}"] = staged[h + nd:h + x.doubles].astype(np.int32)
        h += x.doubles
        if mode not in x.launches:
            raise ValueError(f"mode {mode}: class {x.am} has no such kernel"
                             + (" (kernel_k not built, with_k=False)"
                                if mode == "k" else ""))
        groups.append(list(range(len(steps), len(steps) + len(
            x.launches[mode]))))
        steps += x.launches[mode]
    for e, (lc, ld) in enumerate(drv.dcd):
        n = drv.dcd_doubles(lc, ld)
        buffers[f"dp{e}"] = staged[h:h + n].astype(np.float64)
        h += n
    assert h == staged.size
    steps[0] = replace(steps[0], zero=(out,))
    return buffers, steps, groups


def _rows(basis: Basis, recs: tuple[PairRec, ...], *, bra: bool
          ) -> tuple[np.ndarray, np.ndarray]:
    """One side's doubles (P zeta XPA XPC s I, each contiguous) and ints
    (`dc = c n_sh` on the bra, `dd = d` on the ket): `plan_driver._k_rows`."""
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
    a = np.array(data, dtype=np.float64).reshape(-1, REC_DOUBLES)
    return (np.concatenate([a[:, 0:3].ravel(), a[:, 3], a[:, 4:7].ravel(),
                            a[:, 7:10].ravel(), a[:, 10], a[:, 11]]),
            np.array(ints, dtype=np.float64))


def _dense_widths(am: tuple[int, int, int, int]) -> tuple[int, int]:
    """#575's dense `E` per bra and per ket record: rows times Hermite."""
    la, lc, lb, ld = am

    def nh(l: int) -> int:  # noqa: E741
        return (l + 1) * (l + 2) * (l + 3) // 6

    return (n_cartesians(la) * n_cartesians(lc) * nh(la + lc),
            n_cartesians(lb) * n_cartesians(ld) * nh(lb + ld))


def _e_rows(basis: Basis, recs: tuple[PairRec, ...], *, bra: bool,
            dense: bool = False) -> np.ndarray:
    """One side's `E` tables (#571), record-major: `launch_kblock.e_table` of
    each record's primitive pair, the ket's signed; `dense` (#575) keeps the
    structural zeros."""
    shells = basis.shells
    eps: dict[tuple[int, int], object] = {}
    out = []
    for r in recs:
        ep = eps.get((r.i, r.j))
        if ep is None:
            ep = eps[r.i, r.j] = epair(shells[r.i], shells[r.j])
        P, zeta, _ = ep.recs[r.prim]
        out += e_table(ep.la, ep.lb, zeta, P, ep.A, ep.B, sign=not bra,
                       dense=dense)
    return np.array(out, dtype=np.float64)


def _dcd(drv: KBDriver, basis: Basis, rows: np.ndarray, lc: int, ld: int
         ) -> np.ndarray:
    shells, off, n_sh = basis.shells, basis.offsets, drv.n_sh
    nc, nd = n_cartesians(lc), n_cartesians(ld)
    blk = np.zeros((n_sh * n_sh, nc * nd))
    for c in range(n_sh):
        for d in range(n_sh):
            if shells[c].l == lc and shells[d].l == ld:
                blk[c * n_sh + d] = rows[off[c]:off[c] + nc,
                                         off[d]:off[d] + nd].ravel()
    return blk


def kb_stage(drv: KBDriver, plan: KBLaunchPlan, basis: Basis, D: object
             ) -> np.ndarray:
    """`tau maxD | per class: bra doubles, ket doubles, dmax[n_blk], bra
    ints, ket ints, block ints (bra0.. ko_b, each n_blk) | per density class
    Dcd`. `maxD` is `kernel_k`'s (mode 1); the block kernel reads `dmax`."""
    rows = np.array(checked_density(basis, D), dtype=np.float64)
    if float(np.max(np.abs(rows))) != drv.dm:
        raise ValueError("planned from another density")
    off = basis.offsets
    parts: list[np.ndarray] = [np.array([drv.tau, drv.dm])]
    for cl in plan:
        bd, bi = _rows(basis, cl.bras, bra=True)
        kd, ki = _rows(basis, cl.kets, bra=False)
        blk = np.array([[b.bra0 for b in cl.blocks], [b.bra1 for b in cl.blocks],
                        [b.ket0 for b in cl.blocks], [b.ket1 for b in cl.blocks],
                        [off[b.a] for b in cl.blocks],
                        [off[b.b] for b in cl.blocks]], dtype=np.float64)
        dmax = np.array([b.dmax for b in cl.blocks], dtype=np.float64)
        parts += [bd, kd, dmax]
        if drv.rolled:  # #571: the E tables, after dmax
            eb = _e_rows(basis, cl.kets, bra=False, dense=drv.dense)
            if drv.transposed:  # #576: Ebd[n_ebd, n_ket]
                eb = eb.reshape(len(cl.kets), -1).T.ravel()
            parts += [_e_rows(basis, cl.bras, bra=True, dense=drv.dense), eb]
        parts += [bi, ki, blk.ravel()]
        if drv.rolled and not drv.dense:  # #571: the class tables, last
            parts += [np.array(tables(cl.am)[n], dtype=np.float64)
                      for n in TABLES]
    for lc, ld in drv.dcd:
        parts.append(_dcd(drv, basis, rows, lc, ld).ravel())
    out = np.concatenate(parts)
    assert out.size == drv.n_input, (out.size, drv.n_input)
    return out


def class_args(drv: KBDriver, staged: np.ndarray, c: int) -> dict:
    """Class `c`'s block-kernel arguments sliced from the staged input as the
    generated `main()` slices them (`K` left out): the interpreted twin."""
    h = 2 + sum(x.doubles for x in drv.classes[:c])
    x = drv.classes[c]
    nb, nk, ng = x.n_bra, x.n_ket, x.n_blk
    b = staged[h:h + 12 * nb]
    k = staged[h + 12 * nb:h + 12 * (nb + nk)]
    dmax = staged[h + 12 * (nb + nk):h + 12 * (nb + nk) + ng]
    e0 = h + 12 * (nb + nk) + ng
    ev = staged[e0:e0 + x.e_doubles]
    ints = staged[e0 + x.e_doubles:h + x.doubles].astype(np.int32)
    e = 2 + sum(y.doubles for y in drv.classes)
    lc, ld = x.am[1], x.am[3]
    for y in drv.dcd[:drv.dcd.index((lc, ld))]:
        e += drv.dcd_doubles(*y)
    dcd = staged[e:e + drv.dcd_doubles(lc, ld)]
    blk = ints[nb + nk:nb + nk + BLK_INTS * ng]
    out = dict(
        n_blk=ng, n_bra=nb, n_ket=nk, n_dc=drv.n_sh * drv.n_sh,
        n_ao=drv.n_ao, tau=drv.tau,
        **{n: blk[j * ng:(j + 1) * ng] for j, n in enumerate(
            ("bra0", "bra1", "ket0", "ket1", "ko_a", "ko_b"))}, dmax=dmax,
        P=b[0:3 * nb].reshape(nb, 3), zeta_ac=b[3 * nb:4 * nb],
        XPA=b[4 * nb:7 * nb].reshape(nb, 3),
        XPC=b[7 * nb:10 * nb].reshape(nb, 3),
        s_ac=b[10 * nb:11 * nb], Ip=b[11 * nb:12 * nb], dc=ints[:nb],
        Q=k[0:3 * nk].reshape(nk, 3), eta=k[3 * nk:4 * nk],
        XQB=k[4 * nk:7 * nk].reshape(nk, 3),
        XQD=k[7 * nk:10 * nk].reshape(nk, 3),
        s_bd=k[10 * nk:11 * nk], Iq=k[11 * nk:12 * nk],
        dd=ints[nb:nb + nk],
        Dcd=dcd.reshape(drv.n_sh * drv.n_sh, -1))
    if not drv.rolled:
        return out
    for n in ("XPA", "XPC", "XQB", "XQD"):
        del out[n]
    ea, eb = x.n_e
    out["Eac"] = ev[:ea * nb].reshape(nb, ea)
    out["Ebd"] = (ev[ea * nb:].reshape(eb, nk) if drv.transposed  # #576
                  else ev[ea * nb:].reshape(nk, eb))
    if drv.dense:  # #575: no class tables
        return out
    at = nb + nk + BLK_INTS * ng
    for n in TABLES:
        m = len(tables(x.am)[n])
        out[n] = ints[at:at + m]
        at += m
    lq = x.am[2] + x.am[3]
    out["hpq"] = out["hpq"].reshape(-1, (lq + 1) * (lq + 2) * (lq + 3) // 6)
    return out


def k_scatter(drv: KBDriver, plan: KBLaunchPlan, basis: Basis,
              buf: np.ndarray) -> np.ndarray:
    """`K_mn` from mode 1's scratch: every bra record's row (first shell `a`)
    of every (class, b) block into `K[a][b]`, each element `fsum`med."""
    shells, off, n = basis.shells, basis.offsets, basis.n_ao
    terms: list[list[list[float]]] = [[[] for _ in range(n)] for _ in range(n)]
    for x, cl in zip(drv.classes, plan, strict=True):
        for (b, _, _), o in zip(x.segs, x.scratch, strict=True):
            X = buf[o:o + x.n_bra * x.n_ab].reshape(x.n_bra, x.n_ab)
            nb = shells[b].n_cart
            for row, r in enumerate(cl.bras):
                for ia in range(shells[r.i].n_cart):
                    for ib in range(nb):
                        terms[off[r.i] + ia][off[b] + ib].append(
                            float(X[row, ia * nb + ib]))
    return np.array([[math.fsum(t) for t in r] for r in terms])


@dataclass(frozen=True)
class ClassBinary:
    """One class's own runner: its one-class plan, layout, binary, input."""

    plan: KBLaunchPlan
    drv: KBDriver
    binary: Path
    staged: np.ndarray  # kb_stage's

    def run(self, mode: str, workdir: Path, *, warmup: int = 0,
            reps: int = 2) -> Sequenced:
        """Mode `mode`'s sequence `warmup + reps` times, every timed run's
        output kept. The caller holds the card (its device fixture)."""
        buffers, steps, _ = kb_steps(self.drv, self.staged, mode)
        return run_sequence(self.binary, workdir, buffers, steps,
                            outs=["S" if mode == "k" else "K"], runs=reps,
                            warmup=warmup, witness="each")

    def case(self, mode: str, workdir: Path, *, warmup: int = 0,
             reps: int = 1) -> list[str]:
        """The argv of one run of mode `mode` (its spec written into
        `workdir`): for a profiler wrapping the runner."""
        buffers, steps, _ = kb_steps(self.drv, self.staged, mode)
        spec, arrays = sequence_spec(buffers, steps, outs=[], runs=reps,
                                     warmup=warmup, witness="each")
        return [str(self.binary), str(write_case(workdir, spec, arrays))]

    def timings(self, mode: str, ran: Sequenced) -> dict[str, list[float]]:
        """`{"total" | class index: [ns per timed run]}`: the run's events,
        and per class the sum of its launches' (each kernel alone)."""
        _, _, groups = kb_steps(self.drv, self.staged, mode)
        out = {"total": list(ran.events_ns)}
        for c, g in enumerate(groups):
            out[str(c)] = [sum(r[i] for i in g) for r in ran.launch_ns]
        return out


def class_binaries(tc, plan: KBLaunchPlan, basis: Basis, D: object,
                   backend: str, workdir: Path, *, jobs: int | None = None,
                   split: str | None = None, rolled: bool = False,
                   dense: bool = False, transposed: bool = False,
                   column: bool = False, with_k: bool = True,
                   timeout: float = 1800.0) -> list[ClassBinary]:
    """One runner PER CLASS (`plan` cut to that one class), compiled in
    parallel, each staged. A whole molecule's 162 kernels in one translation
    unit leave one compiler thread at it for most of an hour; one class per
    unit compiles on every core, and times each class in its own process."""
    from concurrent.futures import ThreadPoolExecutor

    from plan_driver import compile_checked

    def one(c: int) -> ClassBinary:
        sub = replace(plan, classes=(plan.classes[c],))
        w = workdir / f"c{c}"
        drv = kb_driver(sub, basis, backend, w / "gen", split=split,
                        rolled=rolled, dense=dense, transposed=transposed,
                        column=column, with_k=with_k)
        binary = compile_checked(tc, drv, w / "kb_driver", timeout=timeout)
        return ClassBinary(sub, drv, binary, kb_stage(drv, sub, basis, D))

    n = jobs or max(1, min(len(plan), (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(one, range(len(plan))))


def run_classes(parts: list[ClassBinary], basis: Basis, mode: str,
                workdir: Path, *, warmup: int = 0, reps: int = 2
                ) -> tuple[list[np.ndarray], list[dict], bool]:
    """Every class's runner in `mode`, in plan order: `K_mn` of the first and
    the last timed run (each the sum over classes), every class's
    `ClassBinary.timings`, and whether each class's first and last runs
    were bit-identical."""
    n = basis.n_ao
    Ks = [np.zeros((n, n)), np.zeros((n, n))]
    outs, same = [], True
    for i, p in enumerate(parts):
        ran = p.run(mode, workdir / f"{mode}_{i}", warmup=warmup, reps=reps)
        runs = ran.load("S" if mode == "k" else "K")
        runs = [runs[0], runs[-1]]
        same &= runs[0].tobytes() == runs[1].tobytes()
        for j, r in enumerate(runs):
            Ks[j] += (r.reshape(n, n) if mode in ("kb", "kbs")
                      else k_scatter(p.drv, p.plan, basis, r))
        outs.append(p.timings(mode, ran))
    return Ks, outs, same
