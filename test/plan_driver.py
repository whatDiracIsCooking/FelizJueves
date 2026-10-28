"""A device driver `main()` generated FROM A LAUNCH PLAN (#144) -- harness.

    driver(plan, backend, workdir)          -> Driver   (source + layout)
    stage(plan, basis, D, driver)           -> np.ndarray, the packed input
    compile_checked(tc, driver, out)        -> Path     (#92's warning filter)
    read_back(driver, raw)                  -> [run][lp] -> J_p[n_bra, n_p]
    reps(stdout)                            -> the timing mode's rep lines

The device twin of `eri.j_coulomb.staged.coulomb_matrix_staged` (#143): the plan's
shape on the card, where `test_ao_j_ondevice.py` (#89) launched one process
per class. It extends `test_kernel_device.py`'s `BATCHED_HARNESS` pattern --
several batch launches into ONE zeroed `dJ`, each kernel's text `#include`d
in its own namespace, the same `PORTABLE` prelude and `VRAM_PROBE` -- from
one class to a whole molecule. It is HARNESS, in `test/`, not a second
emitter in `src/`: the kernels are `emit/cuda.py` / `emit/hip.py`'s text,
unchanged; only the `main()` around them is written here.

WHAT THE GENERATED `main()` DOES, per run (it runs the whole plan `RUNS`
times, so the fp64 atomics' run-to-run spread is measured, not assumed):

  - one `J_p` device buffer per BRA class `lp`, `n_bra(lp) x n_p(lp)`,
    allocated once and zeroed ONCE per run;
  - every `(lp, lq)` class's batch kernels, in plan order, launched into
    `J_p[lp]` -- several ket classes and several batches per buffer, so every
    kernel ADDS: a split class's kernels are `atomize`d by the builder, and a
    one-batch kernel's plain write (correct only for a sole writer) is
    `passes.atomize`d here, exactly as `eri.j_coulomb.staged._accumulating` does (#143's
    decision, carried to the card);
  - read back per bra class, after one synchronize.

THE UPLOADS happen once, before the first run: one contiguous device buffer
and ONE copy per bra class (`P zeta Ip`) and per ket class
(`Q eta Iq Dmax D`), the kernel's arguments being pointers into it -- one
upload per ket class, not one per (bra pair, ket class). The driver prints
its own counts (`fj-driver ...`) so the test asserts them as measured.

THE DATA is `eri.j_coulomb.staged`'s, not restated: `_bra_buffers` / `_ket_buffers`
build the very arrays the interpreted staged driver feeds `interp`, including
its bit-for-bit check that every ket's `Iq * Dmax` is its plan key (the
order the kernel's `break` reads). Eq. 20 and the scatter stay host-side,
`eri.j_coulomb.staged._transform`, as `jgen` does.

THE TIMING MODE (#146). `driver in out WARMUP REPS` runs WARMUP untimed
builds and REPS timed ones in the one process and prints a `rep` line per
timed build (`reps`); `out` is then the FIRST timed build's `J_p` and the
LAST's, the layout the correctness mode (no WARMUP/REPS: 0 and RUNS) writes.
One timed build is the region `measured_store.BUILD_TIMED_REGION` names:
the per-run memsets, every launch, one synchronize and the read-back, after
a synchronize; the uploads are timed once, apart (`upload_ns` on the counts
line), and eq. 20 is host-side and not in it.

THE LAYOUT is baked into the source from the plan (counts per class, launch
order, kernel names) and the input file carries only data; the driver
refuses a file whose size is not the plan's.

THE CHAIN (#157): every arithmetic step of `J_mn` a generated kernel on the
card, on THIS driver -- not a second one. Two processes, because the plan
the driver is generated from is keyed on the DQ kernel's own `Dmax`:

  1. `dq_program` -- every ket pair class's `Target.DQ` kernel (`dq_l_l`),
     one launch per class over all its records, `D_q` and `Dmax` read back:
     the chain's one round-trip (j_engine.md, "The round-trip"), as in
     `eri.j_coulomb.drive.coulomb_matrix_chain`, whose plan builder (`_chain_plan`)
     then keys and sorts the kets on the kernel's `Dmax`;
  2. `driver(plan, ..., chain=basis)` -- milestone 6's driver, unchanged in
     its uploads and `[p|q]` launches, staged with the DQ kernel's `D_q` /
     `Dmax` through `eri.j_coulomb.staged`'s `ket_data=` hook (`stage(ket_data=)`,
     its plan-key check intact), and EXTENDED by one `Target.JAB` launch
     per bra pair class (`jab_ll`) after every `[p|q]` launch, on the same
     stream: each reads its bra class's `J_p` buffer WHERE IT LIES ON THE
     CARD (no `J_p` comes back to be re-uploaded) and adds into one zeroed
     `J_mn`, read back per run. `J_p` is read back too, as the witness of the
     run-to-run bound. Without `chain=` the driver's text is milestone 6's,
     byte for byte (every chain slot of the template is empty).

`jab_program` is one JAB unit alone over a given `J_p` (#157's part 1); the
DQ part-1 harness is `dq_program` over one class. Both are harness, and the
emitted units they `#include` are the generator's text, unchanged.

THE EXCHANGE MATRIX (#206): `k_driver` is the same pattern for a K launch
plan (`launch_k.KLaunchPlan`, #204), in a template of its own (`K_DRIVER`),
so the J driver's text is untouched. Its section below, "K on the card",
states the layout; `k_stage` packs the input, `k_read_back` splits the runs,
and `k_scatter` is the host-side sum of the per-record blocks into `K_mn`.
"""

from __future__ import annotations

import math
import re
import shlex
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import devcache
import numpy as np
from conftest import Toolchain
from test_kernel_device import BACKEND, PORTABLE

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
from felizjueves.plan.targets import hermite_targets

RUNS = 2  # whole-plan runs per process: the run-to-run witness
# The kernel's signature, in the order every launch below passes it.
PARAMS = ("n_bra", "n_ket", "tau", "P", "zeta", "Ip", "Q", "eta", "Iq",
          "Dmax", "D", "J")


@dataclass(frozen=True)
class Launch:
    """One batch kernel of one class: `ns::name`, into `J_p[lp]`."""

    lp: int
    lq: int
    ns: str
    kernel: Kernel


@dataclass(frozen=True)
class Driver:
    """The generated translation unit and what it was generated from."""

    backend: str
    source: str
    path: Path  # the driver's own file
    units: tuple[Path, ...]  # the emitted kernel files it includes
    bra: tuple[tuple[int, int, int], ...]  # (lp, n_bra, n_p), plan order
    ket: tuple[tuple[int, int, int], ...]  # (lq, n_ket, n_q), plan order
    launches: tuple[Launch, ...]
    tau: float
    # The chain (#157): the JAB launches after every [p|q] launch, reading
    # `J_p[lp]` on the card and adding into one zeroed `J_mn` (n_ao x n_ao).
    # Empty for milestone 6's J_p-only driver, whose text is then unchanged.
    jab: tuple[JabLaunch, ...] = ()
    n_ao: int = 0

    @property
    def writers(self) -> Counter[int]:
        """Kernels writing into each bra class's `J_p` (atomics per element)."""
        return Counter(x.lp for x in self.launches)

    @property
    def n_input(self) -> int:
        return 1 + sum(5 * n for _, n, _ in self.bra) + sum(
            n * (6 + q) for _, n, q in self.ket) + sum(
            JAB_DOUBLES * x.n_rec for x in self.jab)


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


def driver(plan: LaunchPlan, backend: str, workdir: Path, *,
           chain: Basis | None = None) -> Driver:
    """Emit every kernel of `plan` for `backend` and the `main()` that
    launches them in the plan's shape (module docstring). With `chain=` (the
    plan's basis, #157) the driver also launches every bra pair class's
    `Target.JAB` kernel after the `[p|q]` launches, reading each `J_p`
    buffer where it lies on the card, and reads back `J_mn` ("THE CHAIN")."""
    emit, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    bra, ket = _shapes(plan)
    bslot = {lp: i for i, (lp, _, _) in enumerate(bra)}
    kslot = {lq: i for i, (lq, _, _) in enumerate(ket)}
    includes, units, launches, body = [], [], [], []
    for c, cl in enumerate(plan):
        for b, k in enumerate(cl.kernels):
            kern = _accumulating(k)
            names = tuple(p.name for p in kern.params)
            if names != PARAMS:
                raise ValueError(f"{kern.name}: signature {names}, the driver "
                                 f"passes {PARAMS}")
            ns = f"fj_c{c}_b{b}"
            path = workdir / f"{ns}_{kern.name}.{ext}"
            path.write_text(emit(kern, arch))
            units.append(path)
            includes.append(f'namespace {ns} {{\n#include "{path.name}"\n}}')
            launches.append(Launch(cl.lp, cl.lq, ns, kern))
            i, j = bslot[cl.lp], kslot[cl.lq]
            body.append(
                f"        // class ({cl.lp}, {cl.lq}), batch {b + 1} of "
                f"{len(cl.kernels)}\n"
                f"        {ns}::{kern.name}<<<dim3(nb[{i}]), dim3(1, LANES), "
                f"(size_t){measured.jsh_slots(kern)} * LANES * sizeof(double)>>>(\n"
                f"            nb[{i}], nk[{j}], tau, bp[{i}], bp[{i}] + 3 * nb[{i}], "
                f"bp[{i}] + 4 * nb[{i}],\n"
                f"            kp[{j}], kp[{j}] + 3 * nk[{j}], kp[{j}] + 4 * nk[{j}], "
                f"kp[{j}] + 5 * nk[{j}], kp[{j}] + 6 * nk[{j}], dJ[{i}]);\n"
                f"        CHECK(devGetLastError());\n"
                f"        ++launches;")
    lanes = {launch_shape(n, arch.wavefront)[1][1] for _, n, _ in bra}
    assert lanes == {arch.wavefront}, lanes
    jabs: list[JabLaunch] = []
    chain_text = dict.fromkeys(CHAIN_SLOTS, "")
    n_ao = 0
    if chain is not None:
        n_ao = chain.n_ao
        for c, (la, lb, lp, data, ints) in enumerate(jab_records(plan, chain)):
            kern = build_jab_kernel((la, lb))
            names = tuple(p.name for p in kern.params)
            if names != JAB_PARAMS:
                raise ValueError(f"{kern.name}: signature {names}, the driver "
                                 f"passes {JAB_PARAMS}")
            ns = f"fj_jab{c}"
            path = workdir / f"{ns}_{kern.name}.{ext}"
            path.write_text(emit(kern, arch))
            units.append(path)
            includes.append(f'namespace {ns} {{\n#include "{path.name}"\n}}')
            jabs.append(JabLaunch(la, lb, lp, ns, kern, data, ints))
        chain_text = _chain_text(jabs, bslot, n_ao)
    prologue = "#include <hip/hip_runtime.h>" if backend == "hip" else ""
    src = DRIVER.format(
        prologue=prologue, portable=PORTABLE, includes="\n".join(includes),
        **chain_text,
        n_b=len(bra), n_k=len(ket), lanes=arch.wavefront, runs=RUNS,
        nb=", ".join(str(n) for _, n, _ in bra),
        np_=", ".join(str(p) for _, _, p in bra),
        nk=", ".join(str(n) for _, n, _ in ket),
        nq=", ".join(str(q) for _, _, q in ket),
        n_launch=len(launches),
        vram_kernels=", ".join(f"(const void*){x.ns}::{x.kernel.name}"
                               for x in (*launches, *jabs)),
        launches="\n".join(body))
    path = workdir / f"driver.{ext}"
    path.write_text(src)
    return Driver(backend, src, path, tuple(units), tuple(bra), tuple(ket),
                  tuple(launches), plan.tau, tuple(jabs), n_ao)


def stage(plan: LaunchPlan, basis: Basis, D: object, drv: Driver, *,
          ket_data: KetDataHook | None = None) -> np.ndarray:
    """The packed input: `tau | per bra class P zeta Ip | per ket class
    Q eta Iq Dmax D [| per JAB class zeta_ab XPA XPB s off_a off_b row]`,
    built by `eri.j_coulomb.staged`'s own buffer builders. `ket_data=` is
    `eri.j_coulomb.staged`'s hook (#155): a ket class's `Q eta D Dmax` from it -- the
    DQ kernel's, in the chain -- instead of the host pre-contraction, through
    the same `_kets_checked` (every `Iq * Dmax` its plan key, bit for bit)."""
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
    parts = [np.array([plan.tau])]
    for lp, _, _ in drv.bra:
        b = bras[lp]
        parts += [b.P.ravel(), b.zeta, b.Ip]
    for lq, _, _ in drv.ket:
        k = kets[lq]
        parts += [k.Q.ravel(), k.eta, k.Iq, k.Dmax, k.D.ravel()]
    for x in drv.jab:
        parts += _jab_parts(x)
    out = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
    assert out.size == drv.n_input, (out.size, drv.n_input)
    return out


def read_back(drv: Driver, raw: np.ndarray) -> list[dict[int, np.ndarray]]:
    """`[run][lp]` -> that run's `J_p` of bra class `lp`. (A chain
    driver's `J_mn` follows the `J_p` runs: `read_back_j`.)"""
    per_run = sum(n * p for _, n, p in drv.bra)
    n_j = RUNS * drv.n_ao * drv.n_ao if drv.jab else 0
    assert raw.size == RUNS * per_run + n_j, (raw.size, RUNS, per_run, n_j)
    out, at = [], 0
    for _ in range(RUNS):
        run = {}
        for lp, n, p in drv.bra:
            run[lp] = raw[at:at + n * p].reshape(n, p)
            at += n * p
        out.append(run)
    return out


def read_back_j(drv: Driver, raw: np.ndarray) -> list[np.ndarray]:
    """A chain driver's `[run]` -> that run's `J_mn`, from the card's JAB."""
    assert drv.jab, "not a chain driver"
    n = drv.n_ao * drv.n_ao
    tail = raw[raw.size - RUNS * n:]
    return [tail[r * n:(r + 1) * n].reshape(drv.n_ao, drv.n_ao)
            for r in range(RUNS)]


# --- the J chain on the card (#157) ----------------------------------------

DQ_LANES = 32  # one thread per record (kernel_dq.launch_shape); not numerical
JAB_THREADS = 32  # one thread per record (kernel_jab.jab_launch_shape)
JAB_PARAMS = ("n_rec", "n_ao", "n_rows", "zeta_ab", "XPA", "XPB", "s", "off_a",
              "off_b", "row", "Jp", "J")
DQ_DOUBLES = 10  # per record: eta XPA(3) XPB(3) s w pair
JAB_DOUBLES = 11  # per record: zeta_ab XPA(3) XPB(3) s off_a off_b row


@dataclass(frozen=True)
class JabLaunch:
    """One bra pair class's `Target.JAB` launch, after every `[p|q]` launch,
    reading `J_p[lp]` on the card. `data` is per record `zeta_ab XPA XPB s`
    and `ints` `off_a off_b row` (exact as doubles), `eri.j_coulomb.drive`'s values."""

    la: int
    lb: int
    lp: int
    ns: str
    kernel: Kernel
    data: np.ndarray  # (n_rec, 8)
    ints: np.ndarray  # (n_rec, 3)

    @property
    def n_rec(self) -> int:
        return int(self.data.shape[0])


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


def _jab_parts(x: JabLaunch) -> list[np.ndarray]:
    """One JAB class's input, in the order the generated `main()` reads it."""
    return [x.data[:, 0], x.data[:, 1:4].ravel(), x.data[:, 4:7].ravel(),
            x.data[:, 7], x.ints[:, 0], x.ints[:, 1], x.ints[:, 2]]


CHAIN_SLOTS = ("jab_helpers", "jab_count", "jab_need", "jab_upload",
               "jab_launch", "jab_readback", "jab_keep", "jab_free",
               "jab_counts")

UPLOAD_INT = r"""
static int* upload_int(const double* h, size_t n) {
    std::vector<int> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (int)h[i];
    int* d = nullptr;
    if (devMalloc(&d, n * sizeof(int) + 8) != devSuccess) return nullptr;
    if (n && devMemcpy(d, v.data(), n * sizeof(int), devMemcpyH2D) != devSuccess)
        return nullptr;
    return d;
}
"""


def _jab_call(x: JabLaunch, c: int, jp: str, n_rows: str) -> str:
    n = x.n_rec
    grid, (threads, _) = jab_launch_shape(n, JAB_THREADS)
    return (f"        // bra pair class ({x.la}, {x.lb}): J_p[{x.lp}] -> J_mn\n"
            f"        {x.ns}::{x.kernel.name}<<<dim3({grid}), dim3({threads})>>>(\n"
            f"            {n}, NAO, {n_rows}, jd[{c}], jd[{c}] + {n}, "
            f"jd[{c}] + {4 * n}, jd[{c}] + {7 * n},\n"
            f"            ji[{c}], ji[{c}] + {n}, ji[{c}] + {2 * n}, {jp}, dJmn);\n"
            f"        CHECK(devGetLastError());\n")


def _chain_text(jabs: list[JabLaunch], bslot: dict[int, int],
                n_ao: int) -> dict[str, str]:
    """The DRIVER template's chain slots (#157), filled."""
    nj = len(jabs)
    calls = "".join(_jab_call(x, c, f"dJ[{bslot[x.lp]}]",
                              f"nb[{bslot[x.lp]}]") + "        ++jab_launches;\n"
                    for c, x in enumerate(jabs))
    return dict(
        jab_helpers=(f"\n// the chain (#157): one JAB launch per bra pair class"
                     f"\nstatic const int NJ = {nj}, NAO = {n_ao};\n"
                     f"static const int nj[NJ] = {{"
                     + ", ".join(str(x.n_rec) for x in jabs) + "};"
                     + UPLOAD_INT),
        jab_count=(f"    for (int c = 0; c < NJ; ++c) want += (size_t)"
                   f"{JAB_DOUBLES} * nj[c];\n"),
        jab_need="\n                          + (size_t)NAO * NAO * sizeof(double)",
        jab_upload=(
            "    int uploads_jab = 0, jab_launches = 0, j_memsets = 0, "
            "j_readbacks = 0;\n"
            "    double* jd[NJ];\n"
            "    int* ji[NJ];\n"
            "    for (int c = 0; c < NJ; ++c) {\n"
            "        jd[c] = upload(h, (size_t)8 * nj[c]);\n"
            "        h += (size_t)8 * nj[c];\n"
            "        ji[c] = upload_int(h, (size_t)3 * nj[c]);\n"
            "        h += (size_t)3 * nj[c];\n"
            "        if (!jd[c] || !ji[c]) { std::fprintf(stderr, \"jab upload "
            "%d failed\\n\", c); return 1; }\n"
            "        ++uploads_jab;\n"
            "    }\n"
            "    double* dJmn = nullptr;\n"
            "    CHECK(devMalloc(&dJmn, (size_t)NAO * NAO * sizeof(double)));\n"
            "    std::vector<double> Jmn((size_t)NAO * NAO), jfirst(Jmn.size()),"
            " jlast(Jmn.size());\n"),
        jab_launch=(
            "        // the chain (#157): every [p|q] launch above is on this "
            "stream, so\n"
            "        // each J_p row is complete; JAB reads it on the card and "
            "adds into J_mn\n"
            "        CHECK(devMemset(dJmn, 0, (size_t)NAO * NAO * sizeof(double)));"
            "\n        ++j_memsets;\n" + calls),
        jab_readback=(
            "        CHECK(devMemcpy(Jmn.data(), dJmn, Jmn.size() * sizeof(double),"
            "\n                        devMemcpyD2H));\n"
            "        ++j_readbacks;\n"),
        jab_keep=("        if (r == warmup) jfirst = Jmn;\n"
                  "        jlast = Jmn;\n"),
        jab_free=(
            "    out.insert(out.end(), jfirst.begin(), jfirst.end());\n"
            "    out.insert(out.end(), jlast.begin(), jlast.end());\n"
            "    for (int c = 0; c < NJ; ++c) {\n"
            "        CHECK(devFree(jd[c]));\n"
            "        CHECK(devFree(ji[c]));\n"
            "    }\n"
            "    CHECK(devFree(dJmn));\n"),
        jab_counts=(
            "    std::printf(\"fj-chain jab_classes %d uploads_jab %d jab_launches "
            "%d \"\n                \"j_memsets %d j_readbacks %d\\n\", NJ, "
            "uploads_jab,\n                jab_launches, j_memsets, "
            "j_readbacks);\n"),
    )


_CHAIN = re.compile(r"^fj-chain (.*)$", re.MULTILINE)


def chain_counts(stdout: str) -> dict[str, int]:
    """A chain driver's own JAB counts line, as `{name: value}`."""
    m = _CHAIN.search(stdout)
    assert m, f"no fj-chain line in:\n{stdout}"
    words = m.group(1).split()
    return {k: int(v) for k, v in zip(words[::2], words[1::2], strict=True)}


@dataclass(frozen=True)
class Program:
    """A generated harness `main()` and the emitted units it includes:
    what `compile_checked` compiles and filters (it reads `backend`, `path`,
    `units`)."""

    backend: str
    source: str
    path: Path
    units: tuple[Path, ...]


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


def _unit(kern: Kernel, backend: str, workdir: Path,
          units: Mapping[str, Path] | None) -> Path:
    """The emitted text of `kern`: `units[kern.name]` when the caller has the
    file (the tier's own, asserted byte-identical by the caller), else
    written into `workdir`."""
    if units is not None and kern.name in units:
        return units[kern.name]
    emit, arch, ext = BACKEND[backend]
    path = workdir / f"{kern.name}.{ext}"
    path.write_text(emit(kern, arch))
    return path


def dq_program(classes: list[DQClass], backend: str, workdir: Path, *,
               units: Mapping[str, Path] | None = None) -> Program:
    """A `main()` running every class's `Target.DQ` kernel once per run
    (RUNS runs), one launch per class over all its records, and writing
    each run's `D` then `Dmax` per class. The outputs are poisoned (all
    bytes 0xff, a NaN) before every run, so a record the kernel leaves
    unwritten reads back non-finite."""
    _, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    includes, paths, calls, kernels = [], [], [], []
    for c, dc in enumerate(classes):
        kern = build_dq_kernel(dc.cls)
        names = tuple(p.name for p in kern.params)
        if names != DQ_PARAMS:
            raise ValueError(f"{kern.name}: signature {names}")
        path = _unit(kern, backend, workdir, units)
        paths.append(path)
        ns = f"fj_dq{c}"
        includes.append(f'namespace {ns} {{\n#include "{path}"\n}}')
        kernels.append(f"(const void*){ns}::{kern.name}")
        n = dc.n_rec
        grid, (lanes, _) = dq_launch_shape(n, DQ_LANES)
        calls.append(
            f"        // ket pair class ({dc.lc}, {dc.ld})\n"
            f"        {ns}::{kern.name}<<<dim3({grid}), dim3({lanes}, 1)>>>(\n"
            f"            nr[{c}], npair[{c}], dd[{c}], dd[{c}] + {n}, "
            f"dd[{c}] + {4 * n}, dd[{c}] + {7 * n},\n"
            f"            dd[{c}] + {8 * n}, dp[{c}], dc[{c}], dD[{c}], dM[{c}]);\n"
            f"        CHECK(devGetLastError());\n"
            f"        ++launches;")
    nq = [dq_shape(*dc.cls)[1] for dc in classes]
    ncd = [dq_shape(*dc.cls)[0] for dc in classes]
    src = DQ_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes), upload_int=UPLOAD_INT,
        n_c=len(classes), runs=RUNS,
        nr=", ".join(str(dc.n_rec) for dc in classes),
        npair=", ".join(str(len(dc.members)) for dc in classes),
        ncd=", ".join(map(str, ncd)), nq=", ".join(map(str, nq)),
        vram_kernels=", ".join(kernels), launches="\n".join(calls))
    path = workdir / f"dq_program.{ext}"
    path.write_text(src)
    return Program(backend, src, path, tuple(paths))


def dq_stage(classes: list[DQClass]) -> np.ndarray:
    """`dq_program`'s input: per class `eta XPA XPB s w | pair | Dcd`."""
    parts = []
    for dc in classes:
        d = dc.data
        parts += [d[:, 0], d[:, 1:4].ravel(), d[:, 4:7].ravel(), d[:, 7],
                  d[:, 8], dc.pair, dc.dcd.ravel()]
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])


def dq_read_back(classes: list[DQClass], raw: np.ndarray
                 ) -> list[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """`[run][(lc, ld)]` -> `(D[n_rec, n_q], Dmax[n_rec])`, record order."""
    out, at = [], 0
    for _ in range(RUNS):
        run = {}
        for dc in classes:
            n, n_q = dc.n_rec, dq_shape(*dc.cls)[1]
            D = raw[at:at + n * n_q].reshape(n, n_q)
            at += n * n_q
            run[dc.cls] = (D, raw[at:at + n])
            at += n
        out.append(run)
    assert at == raw.size, (at, raw.size)
    return out


def jab_program(cls: tuple[int, int], n_rec: int, n_rows: int, n_ao: int,
                backend: str, workdir: Path, *,
                units: Mapping[str, Path] | None = None) -> Program:
    """A `main()` running ONE bra pair class's `Target.JAB` kernel over a
    given `J_p` buffer, RUNS times into a freshly zeroed `J_mn` each, and
    writing each run's `J_mn` (#157's part 1)."""
    _, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    kern = build_jab_kernel(cls)
    names = tuple(p.name for p in kern.params)
    if names != JAB_PARAMS:
        raise ValueError(f"{kern.name}: signature {names}")
    path = _unit(kern, backend, workdir, units)
    n_p = len(hermite_targets(sum(cls)))
    x = JabLaunch(cls[0], cls[1], sum(cls), "fj_jab0", kern,
                  np.zeros((n_rec, 8)), np.zeros((n_rec, 3)))
    src = JAB_PROGRAM.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, unit=path, upload_int=UPLOAD_INT,
        n_rec=n_rec, n_rows=n_rows, n_p=n_p, n_ao=n_ao, runs=RUNS,
        name=kern.name, launch=_jab_call(x, 0, "dJp", "NROWS"))
    main = workdir / f"jab_program.{ext}"
    main.write_text(src)
    return Program(backend, src, main, (path,))


def jab_stage(jp: np.ndarray, data: np.ndarray, ints: np.ndarray) -> np.ndarray:
    """`jab_program`'s input: `Jp | zeta_ab XPA XPB s | off_a off_b row`."""
    x = JabLaunch(0, 0, 0, "", None, data, ints)  # type: ignore[arg-type]
    return np.concatenate([np.asarray(jp, dtype=np.float64).ravel(),
                           *(np.asarray(p, dtype=np.float64)
                             for p in _jab_parts(x))])


def transform(basis: Basis, plan: LaunchPlan, jps: dict[int, np.ndarray]):
    """Eq. 20 and the mirrored scatter: `eri.j_coulomb.staged._transform`, host-side."""
    blocks = {}
    for cl in plan:
        blocks.setdefault(cl.lp, cl.bra_blocks)
    return np.array(_transform(basis, blocks, jps))


_COUNTS = re.compile(r"^fj-driver (.*)$", re.MULTILINE)


_REP = re.compile(r"^rep (\d+) wall_ns (\d+) device_ns (\d+) readback_ns "
                  r"(\d+) events_ns (\d+)$", re.MULTILINE)


def reps(stdout: str) -> list[tuple[int, int, int, float]]:
    """The timing mode's `rep` lines, in order, as (wall_ns, device_ns,
    readback_ns, events_ns); their indices must be 0, 1, ..."""
    got = [(int(m[1]), int(m[2]), int(m[3]), int(m[4]), float(m[5]))
           for m in _REP.finditer(stdout)]
    assert [g[0] for g in got] == list(range(len(got))), stdout
    return [g[1:] for g in got]


def counts(stdout: str) -> dict[str, int]:
    """The driver's own counts line, as `{name: value}`."""
    m = _COUNTS.search(stdout)
    assert m, f"no fj-driver line in:\n{stdout}"
    words = m.group(1).split()
    return {k: int(v) for k, v in zip(words[::2], words[1::2], strict=True)}


# --- compiling it: the tier's flags, and #92's warning filter -------------

FLAGS_VAR = {"cuda": "NVCC_FLAGS", "hip": "HIPCC_FLAGS"}
# kernel-tier.sh's two spellings of a located warning: nvcc's front end
# `file(LINE): warning ...`, clang/gcc `file:LINE[:COL]: warning: ...`.
_WARNING = r"(?:\((\d+)\): warning|:(\d+)(?::\d+)?: warning:?)\s*(.*)"


def compile_command(tc: Toolchain, drv: Driver, out: Path) -> list[str]:
    """The tier's compile command for the driver: devtools/config.sh's
    NVCC_FLAGS / HIPCC_FLAGS (the flag set the compile phase uses, warnings
    on) and the arch, linked to an executable."""
    flags = shlex.split(measured.config_value(FLAGS_VAR[drv.backend]))
    arch = (f"-arch=sm_{tc.arch}" if tc.name == "nvcc"
            else f"--offload-arch={tc.arch}")
    return [tc.path, *flags, arch, str(drv.path), "-o", str(out)]


def generated_warnings(drv: Driver, diagnostics: str) -> list[str]:
    """Every warning whose LOCATION is generated text -- the driver or one of
    the kernel units it includes -- as `file, line N: message` (#92: a
    warning in a vendor header is not the generator's and is not counted)."""
    hits: list[str] = []
    for f in (drv.path, *drv.units):
        for name in {str(f), f.name}:
            pat = re.compile(r"(?:^|[\s\"'/])" + re.escape(name) + _WARNING,
                             re.MULTILINE)
            for m in pat.finditer(diagnostics):
                hit = f"{f.name}, line {m.group(1) or m.group(2)}: {m.group(3)}"
                if hit not in hits:
                    hits.append(hit)
    return hits


def compile_checked(tc: Toolchain, drv: Driver | Program, out: Path, *,
                    env: Mapping[str, str] | None = None,
                    timeout: float = 1800.0) -> Path:
    """Compile the driver with the tier's flags; FAIL on an error, and on any
    warning located in generated text (#92), naming file and line.

    `compile_command`'s flags, run through the compiler cache (#186,
    test/devcache.py): the driver and every unit it includes are carried, by
    name, into a content-addressed directory and compiled there -- `-c`
    through sccache, then linked -- so the same plan in a fresh `tmp_path`
    is a hit. The compiler always runs; on a hit sccache replays the stored
    diagnostics, and the filter below reads them the same way (the carried
    units keep their names, which is what `generated_warnings` matches)."""
    cmd = compile_command(tc, drv, out)
    text, units = devcache.localize(drv.path.read_text(), drv.path.parent)
    got = devcache.compile_unit(cmd[0], cmd[1:-3], drv.path.name, text, units,
                                out, env=env, timeout=timeout)
    diag = got.diagnostics
    assert got.returncode == 0, (
        f"{tc.name} failed in {got.workdir}:\n"
        + "\n".join(shlex.join(c) for c in got.commands) + f"\n{diag}")
    warned = generated_warnings(drv, diag)
    assert not warned, (
        f"{tc.name}: {len(warned)} warning(s) in generated text (#92) -- fix "
        "the emitter or the driver, never a flag:\n  " + "\n  ".join(warned))
    assert out.exists() and out.stat().st_size > 0, f"{out} is empty"
    return out


# The driver. `{...}` are filled from the plan; everything else is C++.
DRIVER = r"""// generated from a launch plan by test/plan_driver.py (#144) -- harness.
{prologue}
{portable}
{includes}{jab_helpers}
#include <chrono>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devMemset hipMemset
#define devFree   hipFree
#define devEvent_t          hipEvent_t
#define devEventCreate      hipEventCreate
#define devEventRecord(e)   hipEventRecord(e, 0)
#define devEventElapsedTime hipEventElapsedTime
#else
#define devMemset cudaMemset
#define devFree   cudaFree
#define devEvent_t          cudaEvent_t
#define devEventCreate      cudaEventCreate
#define devEventRecord(e)   cudaEventRecord(e, 0)
#define devEventElapsedTime cudaEventElapsedTime
#endif
typedef std::chrono::steady_clock fj_clock;
static long long fj_ns(fj_clock::time_point a, fj_clock::time_point b) {{
    return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        b - a).count();
}}

static const int NB = {n_b}, NK = {n_k}, LANES = {lanes}, RUNS = {runs};
static const int nb[NB] = {{{nb}}};   // bra records per bra class
static const int np_[NB] = {{{np_}}};  // n_p per bra class
static const int nk[NK] = {{{nk}}};   // ket records per ket class
static const int nq[NK] = {{{nq}}};   // n_q per ket class

// usage: in out [warmup reps]. `in` is tau | per bra class P zeta Ip | per
// ket class Q eta Iq Dmax D; `out` is every bra class's J_p, in plan order,
// of the FIRST timed build and then of the LAST (warmup 0 and reps RUNS when
// not given: the correctness mode). One `rep` line per timed build (#146):
//   rep <i> wall_ns <memsets + launches + synchronize + read-back>
//       device_ns <memsets + launches + synchronize> readback_ns <D2H>
//       events_ns <GPU events, before the memsets to after the last launch>
int main(int argc, char** argv) {{
    if (argc != 3 && argc != 5) {{
        std::fprintf(stderr, "usage: in out [warmup reps]\n");
        return 2;
    }}
    const int warmup = argc == 5 ? std::atoi(argv[3]) : 0;
    const int reps = argc == 5 ? std::atoi(argv[4]) : RUNS;
    if (warmup < 0 || reps < 1) {{
        std::fprintf(stderr, "bad warmup/reps\n");
        return 2;
    }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = 1, jdoubles = 0;
    for (int i = 0; i < NB; ++i) {{
        want += (size_t)5 * nb[i];
        jdoubles += (size_t)nb[i] * np_[i];
    }}
    for (int j = 0; j < NK; ++j) want += (size_t)nk[j] * (6 + nq[j]);
{jab_count}    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the plan wants %zu\n",
                     in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double) + (size_t)8 * (NB + NK)
                          + jdoubles * sizeof(double){jab_need},
                      {{{vram_kernels}}})) return 0;
    const double tau = in[0];
    const double* h = in.data() + 1;
    int uploads_bra = 0, uploads_ket = 0, jp_buffers = 0, memsets = 0,
        launches = 0, readbacks = 0;
    // ONE upload per bra class and per ket class; the kernel's arguments
    // are pointers into it. Timed ONCE, apart from the builds (#146).
    CHECK(devSynchronize());
    const auto u0 = fj_clock::now();
    double* bp[NB];
    for (int i = 0; i < NB; ++i) {{
        const size_t n = (size_t)5 * nb[i];
        bp[i] = upload(h, n);
        if (!bp[i]) {{ std::fprintf(stderr, "bra upload %d failed\n", i); return 1; }}
        h += n;
        ++uploads_bra;
    }}
    double* kp[NK];
    for (int j = 0; j < NK; ++j) {{
        const size_t n = (size_t)nk[j] * (6 + nq[j]);
        kp[j] = upload(h, n);
        if (!kp[j]) {{ std::fprintf(stderr, "ket upload %d failed\n", j); return 1; }}
        h += n;
        ++uploads_ket;
    }}
{jab_upload}    CHECK(devSynchronize());
    const long long upload_ns = fj_ns(u0, fj_clock::now());
    // ONE J_p per bra class.
    double* dJ[NB];
    for (int i = 0; i < NB; ++i) {{
        CHECK(devMalloc(&dJ[i], (size_t)nb[i] * np_[i] * sizeof(double)));
        ++jp_buffers;
    }}
    devEvent_t ev[2];
    for (int e = 0; e < 2; ++e) CHECK(devEventCreate(&ev[e]));
    std::vector<double> first(jdoubles), last(jdoubles), J(jdoubles);
    for (int r = 0; r < warmup + reps; ++r) {{
        CHECK(devSynchronize());  // nothing earlier inside the clock
        const auto t0 = fj_clock::now();
        CHECK(devEventRecord(ev[0]));
        for (int i = 0; i < NB; ++i) {{  // zeroed ONCE per run
            CHECK(devMemset(dJ[i], 0, (size_t)nb[i] * np_[i] * sizeof(double)));
            ++memsets;
        }}
{launches}
{jab_launch}        CHECK(devEventRecord(ev[1]));
        CHECK(devSynchronize());
        const auto t1 = fj_clock::now();
        size_t at = 0;
        for (int i = 0; i < NB; ++i) {{  // read back per bra class
            const size_t n = (size_t)nb[i] * np_[i];
            CHECK(devMemcpy(J.data() + at, dJ[i], n * sizeof(double),
                            devMemcpyD2H));
            at += n;
            ++readbacks;
        }}
{jab_readback}        const auto t2 = fj_clock::now();
        if (r < warmup) continue;
        float ms = 0.f;
        CHECK(devEventElapsedTime(&ms, ev[0], ev[1]));
        std::printf("rep %d wall_ns %lld device_ns %lld readback_ns %lld "
                    "events_ns %.0f\n", r - warmup, fj_ns(t0, t2),
                    fj_ns(t0, t1), fj_ns(t1, t2), (double)ms * 1e6);
        if (r == warmup) first = J;
        last = J;
{jab_keep}    }}
    std::vector<double> out(first);
    out.insert(out.end(), last.begin(), last.end());
{jab_free}    for (int i = 0; i < NB; ++i) CHECK(devFree(bp[i]));
    for (int j = 0; j < NK; ++j) CHECK(devFree(kp[j]));
    for (int i = 0; i < NB; ++i) CHECK(devFree(dJ[i]));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-driver runs %d bra_classes %d ket_classes %d "
                "uploads_bra %d uploads_ket %d jp_buffers %d memsets %d "
                "launches %d readbacks %d plan_launches %d warmup %d reps %d "
                "upload_ns %lld\n", warmup + reps, NB, NK,
                uploads_bra, uploads_ket, jp_buffers, memsets, launches,
                readbacks, {n_launch}, warmup, reps, upload_ns);
{jab_counts}    return 0;
}}
"""


# The DQ step alone (#157): every ket pair class's pre-contraction, RUNS
# times; `{...}` from the classes, everything else C++.
DQ_PROGRAM = r"""// generated by test/plan_driver.py (#157) -- harness: the DQ step.
{prologue}
{portable}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devMemset hipMemset
#define devFree   hipFree
#define devEvent_t          hipEvent_t
#define devEventCreate      hipEventCreate
#define devEventRecord(e)   hipEventRecord(e, 0)
#define devEventElapsedTime hipEventElapsedTime
#else
#define devMemset cudaMemset
#define devFree   cudaFree
#define devEvent_t          cudaEvent_t
#define devEventCreate      cudaEventCreate
#define devEventRecord(e)   cudaEventRecord(e, 0)
#define devEventElapsedTime cudaEventElapsedTime
#endif
{includes}
{upload_int}
static const int NC = {n_c};
static const int nr[NC] = {{{nr}}};      // records per ket pair class
static const int npair[NC] = {{{npair}}};  // ket shell pairs per class
static const int ncd[NC] = {{{ncd}}};     // Cartesian pairs per shell pair
static const int nq[NC] = {{{nq}}};      // Hermite indices per record

// usage: in out [WARMUP REPS]. `in` is, per class, eta XPA XPB s w | pair |
// Dcd; `out` is, per run and per class, D[nr, nq] then Dmax[nr]. With WARMUP
// REPS: WARMUP untimed runs, then REPS timed ones, each printing `rep <r> dq
// events_ns <x>` (GPU events around every launch, after the poisoning), and
// `out` is the first timed run's then the last's; without, RUNS runs.
int main(int argc, char** argv) {{
    if (argc != 3 && argc != 5) {{
        std::fprintf(stderr, "usage: in out [WARMUP REPS]\n");
        return 2;
    }}
    const int warmup = argc == 5 ? std::atoi(argv[3]) : 0;
    const int reps = argc == 5 ? std::atoi(argv[4]) : {runs};
    if (warmup < 0 || reps < 2) {{
        std::fprintf(stderr, "bad WARMUP/REPS\n");
        return 2;
    }}
    const int RUNS = warmup + reps;
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = 0, outs = 0;
    for (int c = 0; c < NC; ++c) {{
        want += (size_t)10 * nr[c] + (size_t)npair[c] * ncd[c];
        outs += (size_t)nr[c] * (nq[c] + 1);
    }}
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the classes want %zu\n",
                     in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double) + outs * sizeof(double)
                          + (size_t)40 * NC,
                      {{{vram_kernels}}})) return 0;
    const double* h = in.data();
    double* dd[NC];
    int* dp[NC];
    double* dc[NC];
    double* dD[NC];
    double* dM[NC];
    int launches = 0;
    for (int c = 0; c < NC; ++c) {{
        dd[c] = upload(h, (size_t)9 * nr[c]);
        h += (size_t)9 * nr[c];
        dp[c] = upload_int(h, (size_t)nr[c]);
        h += (size_t)nr[c];
        dc[c] = upload(h, (size_t)npair[c] * ncd[c]);
        h += (size_t)npair[c] * ncd[c];
        if (!dd[c] || !dp[c] || !dc[c]) {{
            std::fprintf(stderr, "upload %d failed\n", c);
            return 1;
        }}
        CHECK(devMalloc(&dD[c], (size_t)nr[c] * nq[c] * sizeof(double)));
        CHECK(devMalloc(&dM[c], (size_t)nr[c] * sizeof(double)));
    }}
    std::vector<double> out;
    devEvent_t ev0, ev1;
    CHECK(devEventCreate(&ev0));
    CHECK(devEventCreate(&ev1));
    for (int r = 0; r < RUNS; ++r) {{
        for (int c = 0; c < NC; ++c) {{  // poisoned: 0xff.. is a NaN
            CHECK(devMemset(dD[c], 0xff, (size_t)nr[c] * nq[c] * sizeof(double)));
            CHECK(devMemset(dM[c], 0xff, (size_t)nr[c] * sizeof(double)));
        }}
        CHECK(devEventRecord(ev0));
{launches}
        CHECK(devEventRecord(ev1));
        CHECK(devSynchronize());
        if (argc == 5 && r >= warmup) {{
            float ms = 0.f;
            CHECK(devEventElapsedTime(&ms, ev0, ev1));
            std::printf("rep %d dq events_ns %.0f\n", r - warmup, (double)ms * 1e6);
        }}
        if (r < warmup || (r != warmup && r != RUNS - 1)) continue;
        for (int c = 0; c < NC; ++c) {{
            std::vector<double> D((size_t)nr[c] * nq[c]), M((size_t)nr[c]);
            CHECK(devMemcpy(D.data(), dD[c], D.size() * sizeof(double),
                            devMemcpyD2H));
            CHECK(devMemcpy(M.data(), dM[c], M.size() * sizeof(double),
                            devMemcpyD2H));
            out.insert(out.end(), D.begin(), D.end());
            out.insert(out.end(), M.begin(), M.end());
        }}
    }}
    for (int c = 0; c < NC; ++c) {{
        CHECK(devFree(dd[c]));
        CHECK(devFree(dp[c]));
        CHECK(devFree(dc[c]));
        CHECK(devFree(dD[c]));
        CHECK(devFree(dM[c]));
    }}
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    if (argc == 5) std::printf("fj-dq classes %d launches %d warmup %d reps %d\n",
                               NC, launches, warmup, reps);
    else std::printf("fj-dq classes %d launches %d runs %d\n", NC, launches, RUNS);
    return 0;
}}
"""

# One JAB unit alone over a given J_p (#157's part 1), RUNS times.
JAB_PROGRAM = r"""// generated by test/plan_driver.py (#157) -- harness: one JAB unit.
{prologue}
{portable}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devMemset hipMemset
#define devFree   hipFree
#else
#define devMemset cudaMemset
#define devFree   cudaFree
#endif
namespace fj_jab0 {{
#include "{unit}"
}}
{upload_int}
static const int NROWS = {n_rows}, NP = {n_p}, NAO = {n_ao}, RUNS = {runs};

// usage: in out. `in` is Jp[NROWS, NP] | zeta_ab XPA XPB s | off_a off_b row;
// `out` is each run's J[NAO, NAO], each from a zeroed J.
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    const size_t want = (size_t)NROWS * NP + (size_t)11 * {n_rec};
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, want %zu\n", in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double)
                          + (size_t)NAO * NAO * sizeof(double) + 64,
                      {{(const void*)fj_jab0::{name}}})) return 0;
    const double* h = in.data();
    double* dJp = upload(h, (size_t)NROWS * NP);
    h += (size_t)NROWS * NP;
    double* jd[1];
    int* ji[1];
    jd[0] = upload(h, (size_t)8 * {n_rec});
    h += (size_t)8 * {n_rec};
    ji[0] = upload_int(h, (size_t)3 * {n_rec});
    if (!dJp || !jd[0] || !ji[0]) {{
        std::fprintf(stderr, "upload failed\n");
        return 1;
    }}
    double* dJmn = nullptr;
    CHECK(devMalloc(&dJmn, (size_t)NAO * NAO * sizeof(double)));
    std::vector<double> J((size_t)NAO * NAO), out;
    int launches = 0;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dJmn, 0, J.size() * sizeof(double)));
{launch}        ++launches;
        CHECK(devSynchronize());
        CHECK(devMemcpy(J.data(), dJmn, J.size() * sizeof(double), devMemcpyD2H));
        out.insert(out.end(), J.begin(), J.end());
    }}
    CHECK(devFree(dJp));
    CHECK(devFree(jd[0]));
    CHECK(devFree(ji[0]));
    CHECK(devFree(dJmn));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-jab launches %d runs %d\n", launches, RUNS);
    return 0;
}}
"""


# --- K on the card (#206) ----------------------------------------------------
#
# The whole-molecule exchange build from a K launch plan (`launch_k`, #204),
# with the kernels `kernel_k.build_k_kernels` builds (#198, #203) injected
# into the plan as `build=`. Harness, as the J driver above is: the kernels
# are `emit/cuda.py` / `emit/hip.py`'s text, unchanged; only `main()` is
# written here, from its own template (`K_DRIVER`).
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
#                                passes pointers at its group's first ket;
#   per density class (lc, ld)   `Dcd[n_sh * n_sh, n_c n_d]`, row
#                                `c * n_sh + d` the (c, d) shell block of
#                                D (zero for a shell of another l, never
#                                read): so `dc = c * n_sh` and `dd = d`, and
#                                the kernel's `dc[bra] + dd[k]` names it.
#
# ONE K BUFFER, zeroed once per run and read back once per run, holding one
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
    """One launch: kernel `ns::kernel.name` of class `am` (batch `batch`),
    over the `n_ket` kets of shell `b` from `start` in its ket class's
    upload, adding into block `block`."""

    am: tuple[int, int, int, int]
    batch: int
    b: int
    start: int
    n_ket: int
    block: int
    ns: str
    kernel: Kernel


@dataclass(frozen=True)
class KDriver:
    """The generated K translation unit and what it was generated from."""

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
    """Emit every kernel of the K plan for `backend` and the `main()` that
    launches them per b shell into one K buffer ("K on the card")."""
    emit, arch, ext = BACKEND[backend]
    workdir.mkdir(parents=True, exist_ok=True)
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
    includes, units, launches, body, kernels = [], [], [], [], []
    for c, cl in enumerate(plan):
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
        for bt, k in enumerate(cl.kernels):
            kern = k_accumulating(k)
            names = tuple(p.name for p in kern.params)
            if names != K_PARAMS:
                raise ValueError(f"{kern.name}: signature {names}, the driver "
                                 f"passes {K_PARAMS}")
            ns = f"fj_k{c}_b{bt}"
            path = workdir / f"{ns}_{kern.name}.{ext}"
            path.write_text(emit(kern, arch))
            units.append(path)
            includes.append(f'namespace {ns} {{\n#include "{path.name}"\n}}')
            kernels.append(f"(const void*){ns}::{kern.name}")
            i, j, e = bslot[bls], kslot[kls], eslot[(lc, ld)]
            n_bra = len(bra[bls])
            grid, (_, lanes) = k_launch_shape(n_bra, arch.wavefront)
            assert grid == n_bra and lanes == arch.wavefront, (grid, lanes)
            shared = measured.jsh_slots(kern)
            for b, (s, m) in segs[kls].items():
                q = bidx[bls, b]
                launches.append(KLaunch(cl.am, bt, b, s, m, q, ns, kern))
                body.append(
                    f"        // class ({la}{lc}|{lb}{ld}), batch {bt + 1} of "
                    f"{len(cl.kernels)}, ket shell {b}: kets [{s}, {s + m}) "
                    f"of ket class {j} -> block {q}\n"
                    f"        {ns}::{kern.name}<<<dim3(nb[{i}]), dim3(1, LANES), "
                    f"(size_t){shared} * LANES * sizeof(double)>>>(\n"
                    f"            nb[{i}], {m}, NDC, tau, maxD,\n"
                    f"            bp[{i}], bp[{i}] + 3 * nb[{i}], "
                    f"bp[{i}] + 4 * nb[{i}], bp[{i}] + 7 * nb[{i}],\n"
                    f"            bp[{i}] + 10 * nb[{i}], bp[{i}] + 11 * nb[{i}], "
                    f"bi[{i}],\n"
                    f"            kp[{j}] + {3 * s}, kp[{j}] + 3 * nk[{j}] + {s}, "
                    f"kp[{j}] + 4 * nk[{j}] + {3 * s},\n"
                    f"            kp[{j}] + 7 * nk[{j}] + {3 * s}, "
                    f"kp[{j}] + 10 * nk[{j}] + {s}, kp[{j}] + 11 * nk[{j}] + {s}, "
                    f"ki[{j}] + {s},\n"
                    f"            dp[{e}], dK + {blocks[q].offset});\n"
                    f"        CHECK(devGetLastError());\n"
                    f"        ++launches;")
    drv_partial = KDriver(backend, "", workdir, (), plan.tau, plan.dm, n_sh,
                          tuple((x, bra[x]) for x in bra_cls),
                          tuple((x, ket[x]) for x in ket_cls),
                          tuple(dcd_cls), tuple(blocks), tuple(launches))
    src = K_DRIVER.format(
        prologue="#include <hip/hip_runtime.h>" if backend == "hip" else "",
        portable=PORTABLE, includes="\n".join(includes), upload_int=UPLOAD_INT,
        n_b=len(bra_cls), n_k=len(ket_cls), n_e=len(dcd_cls),
        n_dc=n_sh * n_sh, lanes=arch.wavefront, runs=RUNS,
        nb=", ".join(str(len(bra[x])) for x in bra_cls),
        nk=", ".join(str(len(ket[x])) for x in ket_cls),
        ne=", ".join(str(drv_partial.dcd_doubles(*x)) for x in dcd_cls),
        n_kbuf=drv_partial.n_k, n_launch=len(launches),
        vram_kernels=", ".join(kernels), launches="\n".join(body))
    path = workdir / f"k_driver.{ext}"
    path.write_text(src)
    return replace(drv_partial, source=src, path=path, units=tuple(units))


def _k_rows(basis: Basis, recs: tuple[PairRec, ...], *,
            bra: bool) -> list[np.ndarray]:
    """One class's upload, in the order `main()` reads it: P(3n) zeta(n)
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


def k_read_back(drv: KDriver, raw: np.ndarray) -> list[np.ndarray]:
    """`[run]` -> that run's whole K buffer (every block, in block order)."""
    assert raw.size == RUNS * drv.n_k, (raw.size, RUNS, drv.n_k)
    return [raw[r * drv.n_k:(r + 1) * drv.n_k] for r in range(RUNS)]


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


_KCOUNTS = re.compile(r"^fj-kdriver (.*)$", re.MULTILINE)


def k_counts(stdout: str) -> dict[str, int]:
    """The K driver's own counts line, as `{name: value}`."""
    m = _KCOUNTS.search(stdout)
    assert m, f"no fj-kdriver line in:\n{stdout}"
    words = m.group(1).split()
    return {k: int(v) for k, v in zip(words[::2], words[1::2], strict=True)}


# The K driver (#206). `{...}` are filled from the plan; everything else C++.
K_DRIVER = r"""// generated from a K launch plan by test/plan_driver.py (#206): harness.
{prologue}
{portable}
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define devMemset hipMemset
#define devFree   hipFree
#else
#define devMemset cudaMemset
#define devFree   cudaFree
#endif
{includes}
{upload_int}
static const int NB = {n_b}, NK = {n_k}, NE = {n_e}, NDC = {n_dc},
                 LANES = {lanes}, RUNS = {runs};
static const int nb[NB] = {{{nb}}};  // bra records per bra pair class
static const int nk[NK] = {{{nk}}};  // ket records per ket pair class
static const int ne[NE] = {{{ne}}};  // Dcd doubles per density class
static const size_t NKBUF = {n_kbuf};  // the one K buffer, every block

// usage: in out. `in` is tau maxD | per bra class P zeta_ac XPA XPC s_ac Ip
// | dc | per ket class (grouped by b shell) Q eta XQB XQD s_bd Iq | dd | per
// density class Dcd; `out` is each run's K buffer.
int main(int argc, char** argv) {{
    if (argc != 3) {{ std::fprintf(stderr, "usage: in out\n"); return 2; }}
    std::vector<double> in;
    if (!slurp(argv[1], in)) {{ std::fprintf(stderr, "bad input\n"); return 2; }}
    size_t want = 2;
    for (int i = 0; i < NB; ++i) want += (size_t)13 * nb[i];
    for (int j = 0; j < NK; ++j) want += (size_t)13 * nk[j];
    for (int e = 0; e < NE; ++e) want += (size_t)ne[e];
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the plan wants %zu\n",
                     in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double) + NKBUF * sizeof(double)
                          + (size_t)16 * (NB + NK + NE),
                      {{{vram_kernels}}})) return 0;
    const double tau = in[0], maxD = in[1];
    const double* h = in.data() + 2;
    int uploads_bra = 0, uploads_ket = 0, uploads_dcd = 0, k_buffers = 0,
        memsets = 0, launches = 0, readbacks = 0;
    double* bp[NB];
    int* bi[NB];
    for (int i = 0; i < NB; ++i) {{
        bp[i] = upload(h, (size_t)12 * nb[i]);
        h += (size_t)12 * nb[i];
        bi[i] = upload_int(h, (size_t)nb[i]);
        h += (size_t)nb[i];
        if (!bp[i] || !bi[i]) {{
            std::fprintf(stderr, "bra upload %d failed\n", i);
            return 1;
        }}
        ++uploads_bra;
    }}
    double* kp[NK];
    int* ki[NK];
    for (int j = 0; j < NK; ++j) {{
        kp[j] = upload(h, (size_t)12 * nk[j]);
        h += (size_t)12 * nk[j];
        ki[j] = upload_int(h, (size_t)nk[j]);
        h += (size_t)nk[j];
        if (!kp[j] || !ki[j]) {{
            std::fprintf(stderr, "ket upload %d failed\n", j);
            return 1;
        }}
        ++uploads_ket;
    }}
    double* dp[NE];
    for (int e = 0; e < NE; ++e) {{
        dp[e] = upload(h, (size_t)ne[e]);
        h += (size_t)ne[e];
        if (!dp[e]) {{ std::fprintf(stderr, "Dcd upload %d failed\n", e); return 1; }}
        ++uploads_dcd;
    }}
    // ONE K buffer: a block per (bra pair class, b shell).
    double* dK = nullptr;
    CHECK(devMalloc(&dK, NKBUF * sizeof(double)));
    ++k_buffers;
    std::vector<double> K(NKBUF), out;
    for (int r = 0; r < RUNS; ++r) {{
        CHECK(devMemset(dK, 0, NKBUF * sizeof(double)));  // zeroed ONCE per run
        ++memsets;
{launches}
        CHECK(devSynchronize());
        CHECK(devMemcpy(K.data(), dK, NKBUF * sizeof(double), devMemcpyD2H));
        ++readbacks;
        out.insert(out.end(), K.begin(), K.end());
    }}
    for (int i = 0; i < NB; ++i) {{ CHECK(devFree(bp[i])); CHECK(devFree(bi[i])); }}
    for (int j = 0; j < NK; ++j) {{ CHECK(devFree(kp[j])); CHECK(devFree(ki[j])); }}
    for (int e = 0; e < NE; ++e) CHECK(devFree(dp[e]));
    CHECK(devFree(dK));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-kdriver runs %d bra_classes %d ket_classes %d dcd_classes %d "
                "uploads_bra %d uploads_ket %d uploads_dcd %d k_buffers %d "
                "memsets %d launches %d readbacks %d plan_launches %d\n",
                RUNS, NB, NK, NE, uploads_bra, uploads_ket, uploads_dcd,
                k_buffers, memsets, launches, readbacks, {n_launch});
    return 0;
}}
"""
