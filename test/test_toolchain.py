"""Does this machine compile and run device code, and does it get the right answer?

THE SUITE'S FLOOR, not its subject. Nothing here tests a generated kernel --
there is no generator yet (README.md, "What is here"). What these assert is the
thing every later test depends on and no later test can diagnose: that nvcc and
hipcc are present and can produce code for the architectures this project
targets, and that a kernel launched on each card agrees with a numpy reference.

The profilers are the same kind of floor (#110): a kernel whose fp64 FLOPs
and streamed bytes are known before it runs is read through
`felizjueves.measured.profile` -- ncu on the RTX 3080, rocprofv3's counter
mode on the RX 9060 XT -- and a profiler that cannot measure SKIPS, naming
why, rather than reporting a zero. On the RX 9060 XT what the profiler CAN
count is pinned instead (#140): its own counter list carries no fp64
counter, and its byte counters read 0 for traffic the probe certainly
moves while a control pass counts -- the mechanism of gfx1200's final
refusal (`measured_store.UNMEASURED`).

Read a green run honestly. On a bare host every test here SKIPS except the
host-compiler one, and that is the expected healthy state -- `devtools/doctor.sh`
says which half of the workflow you are on, and `devtools/kernel-tier.sh` is
what runs the device half.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

# One source, both backends. HIP's runtime API is a near rename of CUDA's, so a
# saxpy this small is portable with a macro pair -- which is itself worth
# asserting: the generated kernels will aim to be the same text under both
# compilers, and this is the smallest instance of that claim.
SAXPY_SOURCE = r"""
#include <cstdio>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define devMalloc      hipMalloc
#define devMemcpy      hipMemcpy
#define devMemcpyH2D   hipMemcpyHostToDevice
#define devMemcpyD2H   hipMemcpyDeviceToHost
#define devSynchronize hipDeviceSynchronize
#define devGetLastError hipGetLastError
#define devSuccess     hipSuccess
#else
#define devMalloc      cudaMalloc
#define devMemcpy      cudaMemcpy
#define devMemcpyH2D   cudaMemcpyHostToDevice
#define devMemcpyD2H   cudaMemcpyDeviceToHost
#define devSynchronize cudaDeviceSynchronize
#define devGetLastError cudaGetLastError
#define devSuccess     cudaSuccess
#endif

constexpr int kN = 256;

__global__ void saxpy(float a, const float* x, const float* y, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a * x[i] + y[i];
}

int main() {
  float hx[kN], hy[kN], hout[kN];
  for (int i = 0; i < kN; ++i) {
    hx[i] = static_cast<float>(i) * 0.5f;
    hy[i] = static_cast<float>(i) * -0.25f + 1.0f;
  }
  float *dx = nullptr, *dy = nullptr, *dout = nullptr;
  devMalloc(&dx, sizeof(hx));
  devMalloc(&dy, sizeof(hy));
  devMalloc(&dout, sizeof(hout));
  devMemcpy(dx, hx, sizeof(hx), devMemcpyH2D);
  devMemcpy(dy, hy, sizeof(hy), devMemcpyH2D);
  saxpy<<<(kN + 63) / 64, 64>>>(2.0f, dx, dy, dout, kN);
  if (devSynchronize() != devSuccess || devGetLastError() != devSuccess) {
    std::fprintf(stderr, "kernel launch failed\n");
    return 1;
  }
  devMemcpy(hout, dout, sizeof(hout), devMemcpyD2H);
  for (int i = 0; i < kN; ++i) std::printf("%.7e\n", hout[i]);
  return 0;
}
"""

DEVICE_ONLY_SOURCE = r"""
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
// Deliberately uses shared memory and a sync: the generated J-engine kernels
// hold their recurrence intermediates in shared memory, so a toolchain that
// cannot compile this cannot compile anything this project emits.
__global__ void stage(const float* in, float* out, int n) {
  __shared__ float tile[64];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  tile[threadIdx.x] = (i < n) ? in[i] : 0.0f;
  __syncthreads();
  if (i < n) out[i] = tile[63 - threadIdx.x];
}
"""


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


def parse_floats(stdout: str) -> np.ndarray:
    """One float32 per line of a device binary's stdout."""
    return np.array([float(line) for line in stdout.split()], dtype=np.float32)


def saxpy_reference() -> np.ndarray:
    """The host oracle for SAXPY_SOURCE, in float32 -- the precision matters.

    Computed in float32 rather than numpy's default float64 so the comparison
    below is against what the device was asked to compute, not against a more
    accurate number that would force a looser tolerance to pass.
    """
    i = np.arange(256, dtype=np.float32)
    x = i * np.float32(0.5)
    y = i * np.float32(-0.25) + np.float32(1.0)
    return np.float32(2.0) * x + y


def test_host_compiler_present() -> None:
    """A host compiler exists -- the one check that needs no GPU anything.

    Both are asserted because both are used: nvcc drives g++ as its host
    compiler, while hipcc IS clang.
    """
    found = {name: shutil.which(name) for name in ("g++", "clang++")}
    assert any(found.values()), f"no host C++ compiler on PATH: {found}"
    for name, path in found.items():
        if path:
            done = subprocess.run([path, "--version"], capture_output=True, text=True)
            assert done.returncode == 0, f"{name} present but will not run"


@pytest.mark.cuda
def test_nvcc_compiles_device_code(nvcc, compile_device_source) -> None:
    """nvcc emits a cubin for this project's target architecture."""
    out = compile_device_source(nvcc, DEVICE_ONLY_SOURCE, link=False)
    assert out.suffix == ".cubin" or out.stat().st_size > 0


@pytest.mark.hip
def test_hipcc_compiles_device_code(hipcc, compile_device_source) -> None:
    """hipcc emits an object for this project's gfx target.

    The failure this catches is a ROCm install whose device-libs carry no
    bitcode for HIP_ARCH -- which is invisible until something compiles FOR
    that target. docker/install-rocm.sh makes the same check at image build.
    """
    compile_device_source(hipcc, DEVICE_ONLY_SOURCE, link=False)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_kernel_matches_reference(
    nvcc, cuda_device, compile_device_source, run_device_binary
) -> None:
    """A launched CUDA kernel agrees with the numpy reference bit for bit.

    `rtol=0` is deliberate and is not a statement about kernels in general: a
    single fused multiply-add over exactly representable inputs has one
    correct float32 answer, so any difference here is a toolchain or
    passthrough fault rather than accumulated error. The generated kernels will
    need a real tolerance; this one must not hide behind one.
    """
    binary = compile_device_source(nvcc, SAXPY_SOURCE)
    got = parse_floats(run_device_binary(binary))
    assert got.size == 256, f"expected 256 values, got {got.size}"
    np.testing.assert_allclose(got, saxpy_reference(), rtol=0, atol=0)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_kernel_matches_reference(
    hipcc, hip_device, compile_device_source, run_device_binary
) -> None:
    """The same kernel, the same reference, on the AMD card.

    Sharing SAXPY_SOURCE with the CUDA test is the point: when the two
    backends disagree, the question is whether the SOURCE or the TOOLCHAIN
    differs, and a shared source removes the first possibility.
    """
    binary = compile_device_source(hipcc, SAXPY_SOURCE)
    got = parse_floats(run_device_binary(binary))
    assert got.size == 256, f"expected 256 values, got {got.size}"
    np.testing.assert_allclose(got, saxpy_reference(), rtol=0, atol=0)


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
    from felizjueves import gate, measured

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
    from felizjueves import gate, measured, measured_store

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
    from felizjueves import measured, measured_store

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
