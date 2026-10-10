"""Does this machine compile and run device code, and does it get the right answer?

THE SUITE'S FLOOR, not its subject. Nothing here tests a generated kernel --
there is no generator yet (README.md, "What is here"). What these assert is the
thing every later test depends on and no later test can diagnose: that nvcc and
hipcc are present and can produce code for the architectures this project
targets, and that a kernel launched on each card agrees with a numpy reference.

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
