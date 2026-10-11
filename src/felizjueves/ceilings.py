"""Each card's two roofline ceilings, measured on this box (#109).

WHAT THIS IS. `device.Arch` carries `fp64_flops_per_s` and
`mem_bandwidth_bytes_per_s`, `0.0` meaning NOT CHARACTERISED; `device` reads
their values from `data/measured/ceilings.json`. This module is the other side:
the probe that produces them, the record's shape, and the refusal to record a
phase the card gate (`felizjueves.gate`) did not let through. The recording run
is `test/test_ceilings_device.py`'s opt-in test (its docstring has the command);
the probe source is here so that the record's `source_sha256` is of text the
repo tracks.

THE PROBES, generated-kernel adjacent rather than library calls -- one portable
translation unit (nvcc and hipcc both accept it) with two kernels:

- `fj_fma<C>`: every thread runs `C` independent fp64 FMA chains for FMA_ITERS
  steps (`fma(a, b, c)`, spelled, so nothing depends on contraction flags); 2
  flops per step per chain. Every chain's result is stored and checked
  finite and non-zero, so the compiler can neither fold nor drop the loop.
  The grid is the device's resident-thread capacity times OVERSUBSCRIBE, read
  from the device, not a constant. Both FMA_CHAINS configurations run and the
  faster median is the ceiling: measured on this box, 8 chains is the RTX
  3080's best (16 halves it) and 16 the RX 9060 XT's (8 is ~7% short), and a
  probe that under-uses a card under-reports its ceiling.
- `fj_triad`: STREAM's triad, `a[i] = b[i] + s * c[i]`, grid-stride, over three
  fp64 arrays of TRIAD_N elements each (256 MiB apiece, far past either card's
  last-level cache: the RTX 3080's 5 MiB L2, the RX 9060 XT's 32 MiB Infinity
  Cache). Bytes counted the STREAM way, 3 x 8 per element, no write-allocate
  traffic. Its every element is checked on the device (`b = 1, c = 2, s = 3`,
  so `a = 7` exactly) and a mismatch fails the run.

Each times WARMUP untimed and REPEATS timed launches with the vendor's device
events; the ceiling's value is the MEDIAN rate (the statistic a `Measured`
row's `runtime_ns` is), with the best, q1 and q3 in `raw`.

ACHIEVABLE, NOT PEAK. The triad measures the bandwidth a streaming kernel
achieves, which is what a generated kernel can be expected to see and is below
the DRAM interface's datasheet figure; the FMA probe likewise measures the fp64
rate an FMA-only stream reaches at whatever clock the card sustains. Both are
recorded as `kind: "achievable"`, and the ridge moves with them.

THE PUBLISHED FIGURES ARE A SANITY BOUND, NEVER A DEFAULT. `published` in the
data file holds each card's vendor peaks with their source; a measured value
above one is a bug in the probe (`check_published` refuses it), and the ratio
of each measured value to its published figure is recorded in `raw`, not
hidden. They never reach `device`: an uncharacterised card refuses to score,
it does not borrow a datasheet.
"""

from __future__ import annotations

import json
import socket
import statistics
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from felizjueves import gate
from felizjueves.device import CEILINGS_PATH, QUANTITIES, Arch

SCHEMA = 1
WARMUP = 5
REPEATS = 31
FMA_CHAINS = (8, 16)  # both are run; the faster median is the ceiling
FMA_ITERS = 1 << 14
OVERSUBSCRIBE = 4
TRIAD_N = 1 << 25  # 32 Mi doubles = 256 MiB per array

# The named command a ceiling is reproduced by (test/test_ceilings_device.py).
COMMAND = ("env FJ_CEILINGS_STORE=data/measured/ceilings.json uv run pytest "
           "test/test_ceilings_device.py -p no:xdist -s -rs "
           "-m 'gpu and {backend}' -k record")

PROBE_SOURCE = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define devMalloc            hipMalloc
#define devFree              hipFree
#define devMemcpy            hipMemcpy
#define devMemcpyD2H         hipMemcpyDeviceToHost
#define devSynchronize       hipDeviceSynchronize
#define devGetLastError      hipGetLastError
#define devGetErrorString    hipGetErrorString
#define devError_t           hipError_t
#define devSuccess           hipSuccess
#define devEvent_t           hipEvent_t
#define devEventCreate       hipEventCreate
#define devEventRecord       hipEventRecord
#define devEventSynchronize  hipEventSynchronize
#define devEventElapsedTime  hipEventElapsedTime
#else
#define devMalloc            cudaMalloc
#define devFree              cudaFree
#define devMemcpy            cudaMemcpy
#define devMemcpyD2H         cudaMemcpyDeviceToHost
#define devSynchronize       cudaDeviceSynchronize
#define devGetLastError      cudaGetLastError
#define devGetErrorString    cudaGetErrorString
#define devError_t           cudaError_t
#define devSuccess           cudaSuccess
#define devEvent_t           cudaEvent_t
#define devEventCreate       cudaEventCreate
#define devEventRecord       cudaEventRecord
#define devEventSynchronize  cudaEventSynchronize
#define devEventElapsedTime  cudaEventElapsedTime
#endif
#define CHECK(x) do { devError_t e_ = (x); if (e_ != devSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, devGetErrorString(e_)); return 1; } } while (0)
@VRAM_PROBE@
template <int C>
__global__ void fj_fma(int iters, double b, double c, double* out) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    double a[C];
#pragma unroll
    for (int k = 0; k < C; ++k) a[k] = 1.0 + 1e-3 * (double)(k + (i & 7));
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int k = 0; k < C; ++k) a[k] = fma(a[k], b, c);
    }
    double s = 0.0;
#pragma unroll
    for (int k = 0; k < C; ++k) s += a[k];
    out[i] = s;
}

static devEvent_t t0, t1;

// WARMUP untimed and REPEATS timed launches of `launch`, one `rep` line each.
template <class F>
static int timed(int warmup, int repeats, F launch) {
    for (int r = 0; r < warmup + repeats; ++r) {
        CHECK(devEventRecord(t0, 0));
        launch();
        CHECK(devEventRecord(t1, 0));
        CHECK(devGetLastError());
        CHECK(devEventSynchronize(t1));
        float ms = 0.0f;
        CHECK(devEventElapsedTime(&ms, t0, t1));
        if (r >= warmup) std::printf("rep %d ns %.0f\n", r - warmup, 1e6 * (double)ms);
    }
    return 0;
}

// One FMA configuration: its header, its timed launches, its result checked.
template <int C>
static int fma_config(long grid, int block, int sms, int warmup, int repeats,
                      double* x) {
    const int iters = @FMA_ITERS@;
    std::printf("probe fma chains %d iters %d grid %ld block %d sms %d "
                "flops_per_launch %.0f\n", C, iters, grid, block, sms,
                2.0 * C * (double)iters * (double)(grid * block));
    if (timed(warmup, repeats, [&] {
            fj_fma<C><<<grid, block>>>(iters, 0.999999, 1e-6, x); })) return 1;
    std::vector<double> h((size_t)grid * block);
    CHECK(devMemcpy(h.data(), x, h.size() * sizeof(double), devMemcpyD2H));
    for (double v : h) if (!(v == v) || v == 0.0) {
        std::fprintf(stderr, "fma: non-finite or zero result\n");
        return 1;
    }
    return 0;
}

__global__ void fj_fill(long n, double* x, double v) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) x[i] = v;
}

__global__ void fj_triad(long n, double* a, const double* b, const double* c,
                         double s) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) a[i] = b[i] + s * c[i];
}

__global__ void fj_count_not(long n, const double* a, double want,
                             unsigned long long* bad) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
        if (a[i] != want) atomicAdd(bad, 1ULL);
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "fma") && std::strcmp(argv[1], "triad"))) {
        std::fprintf(stderr, "usage: probe fma|triad\n");
        return 2;
    }
    const bool fma_mode = !std::strcmp(argv[1], "fma");
    const int warmup = @WARMUP@, repeats = @REPEATS@, block = 256;
    int sms = 0, per_sm = 0;
    CHECK(devDeviceGetAttribute(&sms, devAttrSMs, 0));
    CHECK(devDeviceGetAttribute(&per_sm, devAttrThreadsPerSM, 0));
    const long threads = (long)sms * per_sm * @OVERSUBSCRIBE@;
    const long grid = (threads + block - 1) / block;
    const long n = @TRIAD_N@L;
    const size_t bytes = fma_mode ? (size_t)grid * block * sizeof(double)
                                  : 3 * (size_t)n * sizeof(double) + 8;
    if (fj_vram_probe(bytes, {(const void*)fj_fma<@CHAINS_LO@>,
                              (const void*)fj_fma<@CHAINS_HI@>,
                              (const void*)fj_triad})) return 0;
    CHECK(devEventCreate(&t0));
    CHECK(devEventCreate(&t1));
    double *x = nullptr, *y = nullptr, *z = nullptr;
    unsigned long long* bad = nullptr;
    if (fma_mode) {
        CHECK(devMalloc(&x, (size_t)grid * block * sizeof(double)));
        if (fma_config<@CHAINS_LO@>(grid, block, sms, warmup, repeats, x)) return 1;
        if (fma_config<@CHAINS_HI@>(grid, block, sms, warmup, repeats, x)) return 1;
    } else {
        CHECK(devMalloc(&x, (size_t)n * sizeof(double)));
        CHECK(devMalloc(&y, (size_t)n * sizeof(double)));
        CHECK(devMalloc(&z, (size_t)n * sizeof(double)));
        CHECK(devMalloc(&bad, sizeof(unsigned long long)));
        fj_fill<<<grid, block>>>(n, x, 0.0);
        fj_fill<<<grid, block>>>(n, y, 1.0);
        fj_fill<<<grid, block>>>(n, z, 2.0);
        fj_fill<<<1, 1>>>(1, (double*)bad, 0.0);  // all-zero bits: 0ULL
        CHECK(devGetLastError());
        CHECK(devSynchronize());
        std::printf("probe triad n %ld grid %ld block %d sms %d "
                    "bytes_per_launch %.0f\n", n, grid, block, sms,
                    3.0 * sizeof(double) * (double)n);
        if (timed(warmup, repeats, [&] {
                fj_triad<<<grid, block>>>(n, x, y, z, 3.0); })) return 1;
        fj_count_not<<<grid, block>>>(n, x, 7.0, bad);
        CHECK(devGetLastError());
        unsigned long long hbad = 0;
        CHECK(devMemcpy(&hbad, bad, sizeof(hbad), devMemcpyD2H));
        if (hbad) {
            std::fprintf(stderr, "triad: %llu of %ld elements != 7\n", hbad, n);
            return 1;
        }
    }
    std::printf("probe ok\n");
    return 0;
}
"""

PROBE_OF = {"fp64_flops_per_s": "fma", "mem_bandwidth_bytes_per_s": "triad"}
UNIT_OF = {"fp64_flops_per_s": "flop/s", "mem_bandwidth_bytes_per_s": "B/s"}


def probe_source(vram_probe: str) -> str:
    """The probe translation unit, with the harness's VRAM probe spliced in."""
    text = PROBE_SOURCE
    for token, value in (("VRAM_PROBE", vram_probe),
                         ("CHAINS_LO", FMA_CHAINS[0]), ("CHAINS_HI", FMA_CHAINS[1]),
                         ("WARMUP", WARMUP), ("REPEATS", REPEATS),
                         ("OVERSUBSCRIBE", OVERSUBSCRIBE), ("TRIAD_N", TRIAD_N),
                         ("FMA_ITERS", FMA_ITERS)):
        text = text.replace(f"@{token}@", str(value))
    assert "@" not in text, "an unfilled token in the probe source"
    return text


@dataclass(frozen=True)
class ProbeRun:
    """One probe's stdout, parsed: the work per launch and each timed launch."""

    mode: str
    per_launch: float  # flops (fma) or bytes (triad)
    ns: tuple[float, ...]
    header: str

    def rates(self) -> list[float]:
        return [self.per_launch / (t * 1e-9) for t in self.ns]


def parse_probe(out: str) -> list[ProbeRun]:
    """Every configuration the probe ran: a header line, then its `rep` lines."""
    if "probe ok" not in out:
        raise ValueError(f"probe did not finish:\n{out}")
    runs: list[tuple[str, list[float]]] = []
    for ln in out.splitlines():
        if ln.startswith("probe ") and "per_launch" in ln:
            runs.append((ln, []))
        elif ln.startswith("rep ") and runs:
            runs[-1][1].append(float(ln.split()[3]))
    if not runs:
        raise ValueError(f"probe printed no configuration:\n{out}")
    parsed = []
    for header, ns in runs:
        if len(ns) != REPEATS or min(ns) <= 0:
            raise ValueError(f"expected {REPEATS} positive timings after "
                             f"{header!r}:\n{out}")
        f = header.split()
        parsed.append(ProbeRun(f[1], float(f[-1]), tuple(ns), header))
    return parsed


def best_config(runs: list[ProbeRun]) -> ProbeRun:
    """The configuration with the highest median rate: the ceiling is the
    best the probe reaches, and a configuration that under-uses the card is
    the probe's shortfall, not the card's."""
    return max(runs, key=lambda r: statistics.median(r.rates()))


@dataclass(frozen=True)
class Ceiling:
    """One measured ceiling, with a `Measured` row's provenance fields."""

    value: float  # the median rate over `repeats`
    unit: str
    kind: str  # "achievable" -- see the module docstring
    probe: str
    command: str
    compile: str
    source_sha256: str
    toolchain: str
    arch: str
    host: str
    gpu: str
    date: str
    repeats: int
    raw: dict[str, Any] = field(default_factory=dict)


class ProbeBug(AssertionError):
    """A measured value above the vendor's published peak: the probe is wrong."""


def check_published(quantity: str, value: float, published: Mapping) -> float:
    """The ratio of `value` to the published figure; ProbeBug if above it."""
    peak = float(published[quantity])
    if not peak > 0:
        raise ValueError(f"published {quantity} must be positive, got {peak}")
    ratio = value / peak
    if ratio > 1.0:
        raise ProbeBug(
            f"measured {quantity} {value:.4g} is above the published peak "
            f"{peak:.4g} (ratio {ratio:.3f}): a bug in the probe, not a fast "
            "card -- check the work count and the timer")
    return ratio


def host() -> str:
    """The hostname, marked when it is a container's -- `measured.host`'s rule."""
    name = socket.gethostname()
    return f"{name} (container)" if Path("/.dockerenv").exists() else name


def load(path: Path = CEILINGS_PATH) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r} != {SCHEMA}")
    return data


def save(data: dict, path: Path = CEILINGS_PATH) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")


def record(store: dict, arch: Arch, quantity: str,
           run: Callable[[], str], *, backend: str, provenance: Mapping[str, str],
           read: Callable[[str], int | None] | None = None,
           wait_s: float | None = None, cpu_busy: Callable[[], float] | None = None,
           sleep: Callable[[float], None] | None = None,
           ) -> tuple[dict, Ceiling | None, str]:
    """Run one probe phase behind the card gate; return (store, ceiling, why).

    REFUSES -- returns the store unchanged and `None` -- when the gate never
    reads the card (or the CPU) idle before the phase, or the reading just
    after it is non-idle or unparseable (`None`: UNKNOWN, never idle). A
    contended ceiling is not noise: it under-reports, and a low ridge pushes
    every verdict toward compute-bound. `provenance` supplies `probe`,
    `compile`, `source_sha256`, `toolchain`, `gpu`, `date`; `read`, `wait_s`,
    `cpu_busy` and `sleep` are injectable so a test can drive the refusal.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"{quantity!r} is not one of {QUANTITIES}")
    try:
        out, phase = gate.gated(run, backend, read=read, wait_s=wait_s,
                                cpu_busy=cpu_busy, sleep=sleep)
    except gate.Contended as refused:
        return store, None, f"REFUSED {arch.name} {quantity}: {refused}"
    runs = parse_probe(out)
    if any(r.mode != PROBE_OF[quantity] for r in runs):
        raise ValueError(f"{quantity} needs the {PROBE_OF[quantity]} probe, "
                         f"got {[r.mode for r in runs]}")
    published = store["published"][arch.name]
    for r in runs:  # every configuration is bounded, not just the chosen one
        check_published(quantity, max(r.rates()), published)
    parsed = best_config(runs)
    rates = sorted(parsed.rates())
    value = statistics.median(rates)
    q1, _, q3 = statistics.quantiles(rates, n=4, method="inclusive")
    ratio = check_published(quantity, max(rates), published)
    raw = {
        "probe_header": parsed.header,
        "configs_median": {r.header: statistics.median(r.rates()) for r in runs},
        "best": max(rates),
        "worst": min(rates),
        "q1": q1,
        "q3": q3,
        "warmup": WARMUP,
        "published": float(published[quantity]),
        "ratio_to_published": value / float(published[quantity]),
        "best_ratio_to_published": ratio,
        "gpu_util_before": phase.gpu_before,
        "gpu_util_after": phase.gpu_after,
        "cpu_busy_permille_before": round(1000 * phase.cpu_before),
    }
    ceiling = Ceiling(
        value=value, unit=UNIT_OF[quantity], kind="achievable",
        probe=provenance["probe"], command=COMMAND.format(backend=backend),
        compile=provenance["compile"], source_sha256=provenance["source_sha256"],
        toolchain=provenance["toolchain"], arch=arch.name, host=host(),
        gpu=provenance["gpu"], date=provenance["date"], repeats=len(rates),
        raw=raw)
    new = json.loads(json.dumps(store))
    new.setdefault("ceilings", {}).setdefault(arch.name, {})[quantity] = asdict(
        ceiling)
    return new, ceiling, (f"recorded {arch.name} {quantity} = {value:.4g} "
                          f"{ceiling.unit} ({value / float(published[quantity]):.1%}"
                          " of published)")
