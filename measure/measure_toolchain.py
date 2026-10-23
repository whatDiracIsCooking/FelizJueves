"""The profilers read a kernel whose counts are known (#110, #140).

Moved from test/test_toolchain.py, which keeps the compilers' and the cards'
floor: these are the same kind of floor for the counter rows, and nothing
here checks a kernel's answer. A kernel whose fp64 FLOPs and streamed bytes
are known before it runs is read through `felizjueves.performance.measured.profile` --
ncu on the RTX 3080, rocprofv3's counter mode on the RX 9060 XT -- and a
profiler that cannot measure SKIPS, naming why, rather than reporting a
zero. On the RX 9060 XT what the profiler CAN count is pinned instead
(#140): its own counter list carries no fp64 counter, and its byte counters
read 0 for traffic the probe certainly moves while a control pass counts --
the mechanism of gfx1200's final refusal (`measured_store.UNMEASURED`).

    devtools/measure.sh probes --backend cuda   # or hip

which also runs measure/measure_ceilings_device.py's published-peak bound.
Each test is `gpu` plus `cuda` or `hip` and skips without the card, the
compiler or the profiler; nothing is written.
"""

from __future__ import annotations

import subprocess

import pytest

# #110's witness for the counter parsers: a kernel whose counts are known
# before it runs. One fp64 FMA per element (2 flops, exactly), 16 bytes read
# and 8 written per element, over arrays of 128 MiB each -- past either card's
# last-level cache (the RTX 3080's 5 MiB L2, the RX 9060 XT's 32 MiB Infinity
# Cache), so the traffic reaches the counters below them.
COUNTER_N = 1 << 24
COUNTER_PROBE_SOURCE = r"""
#include <cstdio>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define devMalloc      hipMalloc
#define devMemset      hipMemset
#define devSynchronize hipDeviceSynchronize
#define devGetLastError hipGetLastError
#define devError_t     hipError_t
#define devSuccess     hipSuccess
#else
#define devMalloc      cudaMalloc
#define devMemset      cudaMemset
#define devSynchronize cudaDeviceSynchronize
#define devGetLastError cudaGetLastError
#define devError_t     cudaError_t
#define devSuccess     cudaSuccess
#endif
#define CHECK(x) do { devError_t e_ = (x); if (e_ != devSuccess) { \
    std::fprintf(stderr, "%s failed\n", #x); return 1; } } while (0)

constexpr long kN = 1L << 24;

__global__ void fj_counter_probe(double a, const double* x, double* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fma(a, x[i], y[i]);
}

int main() {
  double *x = nullptr, *y = nullptr;
  CHECK(devMalloc(&x, kN * sizeof(double)));
  CHECK(devMalloc(&y, kN * sizeof(double)));
  CHECK(devMemset(x, 0, kN * sizeof(double)));
  CHECK(devMemset(y, 0, kN * sizeof(double)));
  fj_counter_probe<<<(kN + 255) / 256, 256>>>(3.0, x, y, kN);
  CHECK(devGetLastError());
  CHECK(devSynchronize());
  std::printf("ok\n");
  return 0;
}
"""


# --- #110: the profilers read a kernel whose counts are known ----------------

STREAMED_READ, STREAMED_WRITTEN = 16 * COUNTER_N, 8 * COUNTER_N
# DECLARED before reading a counter: each byte count within 10% of what the
# kernel streams (DRAM sectors on NVIDIA and fabric requests on AMD may round
# up; neither may lose a tenth of the traffic). The FLOP count is exact.
BYTES_BAND = (0.9, 1.1)


def _counted(backend: str, binary, tmp_path):
    """The probe's one launch under `backend`'s profiler, behind the card
    gate. A profiler that cannot measure SKIPS, naming why -- could not
    measure is not a pass, and never a zero."""
    from felizjueves.performance import gate, measured

    try:
        gate.wait_card(backend)
    except gate.Contended as e:
        pytest.skip(f"card gate: {e}")
    try:
        launches, tool, command = measured.profile(
            backend, [str(binary)], kernel_regex="fj_counter_probe",
            workdir=tmp_path)
    except measured.CouldNotMeasure as e:
        pytest.skip(str(e))  # it says "could not measure" and why
    (c,) = launches
    assert c.kernel == "fj_counter_probe"
    assert tool.startswith(measured.PROFILERS[backend] + " ")
    assert command.startswith(measured.PROFILERS[backend] + " ")
    lo, hi = BYTES_BAND
    assert lo <= c.dram_bytes_read / STREAMED_READ <= hi, c
    assert lo <= c.dram_bytes_written / STREAMED_WRITTEN <= hi, c
    assert c.counter_ns > 0
    return c


@pytest.mark.cuda
@pytest.mark.gpu
def test_ncu_counts_the_probes_flops_and_bytes(
    nvcc, cuda_device, compile_device_source, tmp_path
) -> None:
    """ncu, through `measured.profile`, reads exactly 2 flops per element
    (dadd + dmul + 2 x dfma) and the streamed bytes within BYTES_BAND."""
    binary = compile_device_source(nvcc, COUNTER_PROBE_SOURCE)
    c = _counted("cuda", binary, tmp_path)
    assert c.fp64_flops == 2 * COUNTER_N, c


@pytest.mark.hip
@pytest.mark.gpu
def test_rocprofv3_profiles_the_probe_and_its_byte_counters_read_zero(
    hipcc, hip_device, compile_device_source, tmp_path
) -> None:
    """#140: rocprofv3's counter mode, through `measured.profile`, on the RX
    9060 XT. It loads and profiles the probe -- and every GL2C <-> EA byte
    counter reads 0 for the 384 MiB the probe streams, which `profile`
    refuses (CouldNotMeasure, `measured.ROCPROF_ZERO_BYTES`) rather than
    filing as a zero. A CONTROL pass in the same shape shows the collection
    itself works: SQ_WAVES is exactly the probe's wave count and
    GRBM_GUI_ACTIVE is non-zero, so the zero is the byte counters', not a
    failed run's. If the bytes ever count, this fails: gfx1200's UNMEASURED
    reason says they do not, and is then stale (fp64 stays uncounted)."""
    from felizjueves.performance import gate, measured, measured_store

    binary = compile_device_source(hipcc, COUNTER_PROBE_SOURCE)
    try:
        gate.wait_card("hip")
    except gate.Contended as e:
        pytest.skip(f"card gate: {e}")
    try:
        launches, _, _ = measured.profile(
            "hip", [str(binary)], kernel_regex="fj_counter_probe",
            workdir=tmp_path)
    except measured.CouldNotMeasure as e:
        if not str(e).startswith(measured.ROCPROF_ZERO_BYTES):
            pytest.skip(str(e))  # an image whose rocprofv3 cannot load, say
        assert "fj_counter_probe" in str(e)
    else:
        pytest.fail(f"rocprofv3's byte counters now count on gfx1200 "
                    f"({launches}): measured_store.UNMEASURED's byte clause "
                    "is stale -- re-read it, and test the bytes against "
                    "BYTES_BAND")
    # the shared phrase: the driver's refusal and the store's reason agree
    zero = "GL2C <-> EA byte counters all read 0"
    assert zero in measured.ROCPROF_ZERO_BYTES
    assert zero in measured_store.UNMEASURED["gfx1200"]

    control = ("SQ_WAVES", "GRBM_GUI_ACTIVE")
    out = tmp_path / "control"
    done = subprocess.run(
        [measured._which("rocprofv3"), "--pmc", *control,
         "--kernel-include-regex", "fj_counter_probe", "--output-format",
         "csv", "-d", str(out), "-o", "control", "--", str(binary)],
        capture_output=True, text=True, timeout=measured.ROCPROF_TIMEOUT_S,
        cwd=tmp_path)  # rocprofv3 drops .rocprofv3/ in its cwd
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    (csv_path,) = out.rglob("*counter_collection.csv")
    ((_, kernel, raw, ns),) = measured._rocprof_dispatches(csv_path.read_text())
    assert kernel == "fj_counter_probe" and ns > 0
    assert raw["SQ_WAVES"] == COUNTER_N // 32, raw  # wave32: every wave seen
    assert raw["GRBM_GUI_ACTIVE"] > 0, raw


@pytest.mark.hip
@pytest.mark.gpu
def test_rocprofv3_lists_byte_counters_and_no_fp64_counter_on_gfx1200(
    hip_device,
) -> None:
    """#140's refusal, pinned to its mechanism: the card's own counter list.
    rocprofv3 loads and lists the GL2C <-> EA byte counters the store would
    read (which read 0 here: the test above), and NO counter that counts
    fp64 operations -- not the ones measured_store.MISSING_COUNTERS names,
    nor any other F64 / FLOP counter. If one ever appears, gfx1200's
    UNMEASURED refusal is wrong and this fails: news, to be re-read."""
    from felizjueves.performance import measured, measured_store

    tool = measured._which("rocprofv3")
    if tool is None:
        pytest.skip("rocprofv3 not found: the card's counter list is unread")
    done = subprocess.run([tool, "--list-avail"], capture_output=True,
                          text=True, timeout=300)
    report = done.stdout + done.stderr
    assert done.returncode == 0, report[-2000:]
    assert "error while loading shared libraries" not in report, report[-2000:]
    avail = measured.parse_rocprof_avail(done.stdout)
    assert "gfx1200" in avail, sorted(avail)
    names = avail["gfx1200"]
    assert set(measured.ROCPROF_COUNTERS) <= names  # listed (they read 0)
    assert measured.fp64_counters(names) == [], measured.fp64_counters(names)
    missing = measured_store.MISSING_COUNTERS["gfx1200"]
    assert not set(missing) & names
    assert all(n in measured_store.UNMEASURED["gfx1200"] for n in missing)
