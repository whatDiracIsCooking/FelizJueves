"""A device driver `main()` generated FROM A LAUNCH PLAN (#144) -- harness.

    driver(plan, backend, workdir)          -> Driver   (source + layout)
    stage(plan, basis, D, driver)           -> np.ndarray, the packed input
    compile_checked(tc, driver, out)        -> Path     (#92's warning filter)
    read_back(driver, raw)                  -> [run][lp] -> J_p[n_bra, n_p]
    reps(stdout)                            -> the timing mode's rep lines

The device twin of `ao.staged.coulomb_matrix_staged` (#143): the plan's
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
    `passes.atomize`d here, exactly as `ao.staged._accumulating` does (#143's
    decision, carried to the card);
  - read back per bra class, after one synchronize.

THE UPLOADS happen once, before the first run: one contiguous device buffer
and ONE copy per bra class (`P zeta Ip`) and per ket class
(`Q eta Iq Dmax D`), the kernel's arguments being pointers into it -- one
upload per ket class, not one per (bra pair, ket class). The driver prints
its own counts (`fj-driver ...`) so the test asserts them as measured.

THE DATA is `ao.staged`'s, not restated: `_bra_buffers` / `_ket_buffers`
build the very arrays the interpreted staged driver feeds `interp`, including
its bit-for-bit check that every ket's `Iq * Dmax` is its plan key (the
order the kernel's `break` reads). Eq. 20 and the scatter stay host-side,
`ao.staged._transform`, as `jgen` does.

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
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from conftest import Toolchain
from test_kernel_device import BACKEND, PORTABLE

from felizjueves import measured
from felizjueves.ao.j import _density
from felizjueves.ao.shell import Basis
from felizjueves.ao.staged import (
    _accumulating,
    _bra_buffers,
    _dglobal,
    _ket_buffers,
    _transform,
)
from felizjueves.ir import Kernel
from felizjueves.kernel import bra_functions, ket_functions, launch_shape
from felizjueves.launch import LaunchPlan, class_spec

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

    @property
    def writers(self) -> Counter[int]:
        """Kernels writing into each bra class's `J_p` (atomics per element)."""
        return Counter(x.lp for x in self.launches)

    @property
    def n_input(self) -> int:
        return 1 + sum(5 * n for _, n, _ in self.bra) + sum(
            n * (6 + q) for _, n, q in self.ket)


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


def driver(plan: LaunchPlan, backend: str, workdir: Path) -> Driver:
    """Emit every kernel of `plan` for `backend` and the `main()` that
    launches them in the plan's shape (module docstring)."""
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
    prologue = "#include <hip/hip_runtime.h>" if backend == "hip" else ""
    src = DRIVER.format(
        prologue=prologue, portable=PORTABLE, includes="\n".join(includes),
        n_b=len(bra), n_k=len(ket), lanes=arch.wavefront, runs=RUNS,
        nb=", ".join(str(n) for _, n, _ in bra),
        np_=", ".join(str(p) for _, _, p in bra),
        nk=", ".join(str(n) for _, n, _ in ket),
        nq=", ".join(str(q) for _, _, q in ket),
        n_launch=len(launches),
        vram_kernels=", ".join(f"(const void*){x.ns}::{x.kernel.name}"
                               for x in launches),
        launches="\n".join(body))
    path = workdir / f"driver.{ext}"
    path.write_text(src)
    return Driver(backend, src, path, tuple(units), tuple(bra), tuple(ket),
                  tuple(launches), plan.tau)


def stage(plan: LaunchPlan, basis: Basis, D: object, drv: Driver) -> np.ndarray:
    """The packed input: `tau | per bra class P zeta Ip | per ket class
    Q eta Iq Dmax D`, built by `ao.staged`'s own buffer builders."""
    rows = _density(basis, D)
    dglobal = _dglobal(plan, basis, rows)
    bras, kets = {}, {}
    for cl in plan:
        if cl.lp not in bras:
            bras[cl.lp] = _bra_buffers(basis, cl.bras)
        if cl.lq not in kets:
            n_q = dict((lq, q) for lq, _, q in drv.ket)[cl.lq]
            kets[cl.lq] = _ket_buffers(basis, rows, cl.kets, n_q, dglobal)
        elif kets[cl.lq].recs != cl.kets:
            raise ValueError(f"ket class {cl.lq}: two ket lists, one upload")
    parts = [np.array([plan.tau])]
    for lp, _, _ in drv.bra:
        b = bras[lp]
        parts += [b.P.ravel(), b.zeta, b.Ip]
    for lq, _, _ in drv.ket:
        k = kets[lq]
        parts += [k.Q.ravel(), k.eta, k.Iq, k.Dmax, k.D.ravel()]
    out = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
    assert out.size == drv.n_input, (out.size, drv.n_input)
    return out


def read_back(drv: Driver, raw: np.ndarray) -> list[dict[int, np.ndarray]]:
    """`[run][lp]` -> that run's `J_p` of bra class `lp`."""
    per_run = sum(n * p for _, n, p in drv.bra)
    assert raw.size == RUNS * per_run, (raw.size, RUNS, per_run)
    out, at = [], 0
    for _ in range(RUNS):
        run = {}
        for lp, n, p in drv.bra:
            run[lp] = raw[at:at + n * p].reshape(n, p)
            at += n * p
        out.append(run)
    return out


def transform(basis: Basis, plan: LaunchPlan, jps: dict[int, np.ndarray]):
    """Eq. 20 and the mirrored scatter: `ao.staged._transform`, host-side."""
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


def compile_checked(tc: Toolchain, drv: Driver, out: Path) -> Path:
    """Compile the driver with the tier's flags; FAIL on an error, and on any
    warning located in generated text (#92), naming file and line."""
    cmd = compile_command(tc, drv, out)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                          cwd=drv.path.parent)
    diag = done.stdout + done.stderr
    assert done.returncode == 0, f"{tc.name} failed:\n{shlex.join(cmd)}\n{diag}"
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
{includes}
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
    if (in.size() != want) {{
        std::fprintf(stderr, "input has %zu doubles, the plan wants %zu\n",
                     in.size(), want);
        return 2;
    }}
    if (fj_vram_probe(in.size() * sizeof(double) + (size_t)8 * (NB + NK)
                          + jdoubles * sizeof(double),
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
    CHECK(devSynchronize());
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
        CHECK(devEventRecord(ev[1]));
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
        const auto t2 = fj_clock::now();
        if (r < warmup) continue;
        float ms = 0.f;
        CHECK(devEventElapsedTime(&ms, ev[0], ev[1]));
        std::printf("rep %d wall_ns %lld device_ns %lld readback_ns %lld "
                    "events_ns %.0f\n", r - warmup, fj_ns(t0, t2),
                    fj_ns(t0, t1), fj_ns(t1, t2), (double)ms * 1e6);
        if (r == warmup) first = J;
        last = J;
    }}
    std::vector<double> out(first);
    out.insert(out.end(), last.begin(), last.end());
    for (int i = 0; i < NB; ++i) CHECK(devFree(bp[i]));
    for (int j = 0; j < NK; ++j) CHECK(devFree(kp[j]));
    for (int i = 0; i < NB; ++i) CHECK(devFree(dJ[i]));
    if (!spill(argv[2], out)) {{ std::fprintf(stderr, "write failed\n"); return 1; }}
    std::printf("fj-driver runs %d bra_classes %d ket_classes %d "
                "uploads_bra %d uploads_ket %d jp_buffers %d memsets %d "
                "launches %d readbacks %d plan_launches %d warmup %d reps %d "
                "upload_ns %lld\n", warmup + reps, NB, NK,
                uploads_bra, uploads_ket, jp_buffers, memsets, launches,
                readbacks, {n_launch}, warmup, reps, upload_ns);
    return 0;
}}
"""
