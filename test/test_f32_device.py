"""F32 on the card (#338): the interpreter's fp32 rounding, witnessed by a DEVICE.

THE GUARDRAIL. If the interpreter did not round to float32 at every
operation and assignment, every fp64 test would still pass and an fp32
parity check against the interpreter's own idea of fp32 would too. So the
two parity programs of test/f32_programs.py are emitted, compiled with the
project's unmodified flags (devtools/config.sh: no fast-math, contraction
left at each compiler's default), launched on the card, and held to the
interpreter:

- `f32_exact` BITWISE. No multiply in it feeds an add, so no FMA can form,
  and every operation in it is correctly rounded on both compilers by
  default (IEEE `/` and `sqrtf`: nvcc's `-prec-div` / `-prec-sqrt`, clang's
  `-fhip-fp32-correctly-rounded-divide-sqrt`). One ulp of difference
  anywhere is a semantic defect in the interpreter or the emitter.
- `f32_contracting` at the declared fp32 tier (test/oracle/tolerance.py,
  `F32_REL * scale` per column, `scale` the column's largest |value|). Its
  `Weighted` recurrence is an FMA candidate on both compilers and `expf`
  is not correctly rounded on either, so the card may differ from the
  interpreter by a few ulps there -- far inside the tier.

THE MUTATION. Each test then re-interprets with `interp._round32` replaced by
the identity (fp64 throughout, rounded only into the float32 output) and
asserts that this interpreter MISSES THE CARD: not bitwise on `f32_exact`,
and beyond `F32_REL * scale` on the discriminating columns of both
programs. So the declared tolerance is shown, against a card, to be tight
enough to fail when the rounding is absent.

Run on each card (the GPU agent):
    devtools/kernel-tier.sh --backend cuda -- -k f32
    devtools/kernel-tier.sh --backend hip -- -k f32
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from f32_programs import (
    DISCRIMINATING,
    contracting_kernel,
    exact_kernel,
    inputs,
    launch,
    run,
)
from oracle.tolerance import F32_REL, assert_within

from felizjueves.emit import cuda, hip
from felizjueves.ir import Kernel, Scalar, interp
from felizjueves.performance.device import GFX1200, SM_86

BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}

HARNESS = r"""
#include <cstdio>
#include <vector>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define devMalloc hipMalloc
#define devMemcpy hipMemcpy
#define devMemcpyH2D hipMemcpyHostToDevice
#define devMemcpyD2H hipMemcpyDeviceToHost
#define devSynchronize hipDeviceSynchronize
#define devGetLastError hipGetLastError
#define devGetErrorString hipGetErrorString
#define devError_t hipError_t
#define devSuccess hipSuccess
#else
#define devMalloc cudaMalloc
#define devMemcpy cudaMemcpy
#define devMemcpyH2D cudaMemcpyHostToDevice
#define devMemcpyD2H cudaMemcpyDeviceToHost
#define devSynchronize cudaDeviceSynchronize
#define devGetLastError cudaGetLastError
#define devGetErrorString cudaGetErrorString
#define devError_t cudaError_t
#define devSuccess cudaSuccess
#endif
#define CHECK(x) do {{ devError_t e_ = (x); if (e_ != devSuccess) {{ \
    std::fprintf(stderr, "%s: %s\n", #x, devGetErrorString(e_)); \
    return 1; }} }} while (0)
#include "{kernel}"

template <class T> static bool load(const char* p, std::vector<T>& v) {{
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    bool ok = std::fread(v.data(), sizeof(T), v.size(), f) == v.size();
    std::fclose(f);
    return ok;
}}
template <class T> static bool save(const char* p, const std::vector<T>& v) {{
    FILE* f = std::fopen(p, "wb");
    if (!f) return false;
    bool ok = std::fwrite(v.data(), sizeof(T), v.size(), f) == v.size();
    std::fclose(f);
    return ok;
}}
template <class T> static T* put(const std::vector<T>& h) {{
    T* d = nullptr;
    if (devMalloc(&d, h.size() * sizeof(T)) != devSuccess) return nullptr;
    if (devMemcpy(d, h.data(), h.size() * sizeof(T), devMemcpyH2D) != devSuccess)
        return nullptr;
    return d;
}}

// argv: x.bin y.bin d.bin out.bin out64.bin
int main(int argc, char** argv) {{
    if (argc != 6) return 2;
    const int n = {n};
    std::vector<float> x(n), y(n), out((size_t)n * {cols}, 0.0f);
    std::vector<double> d(n), out64((size_t)n * {cols64} + 1, 0.0);
    if (!load(argv[1], x) || !load(argv[2], y) || !load(argv[3], d)) return 2;
    float* dx = put(x);
    float* dy = put(y);
    double* dd = put(d);
    float* dout = put(out);
    double* dout64 = put(out64);
    if (!dx || !dy || !dd || !dout || !dout64) return 1;
    {name}<<<dim3({grid}), dim3({block}, 1)>>>(n, dx, dy, dd, dout{out64_arg});
    CHECK(devGetLastError());
    CHECK(devSynchronize());
    CHECK(devMemcpy(out.data(), dout, out.size() * sizeof(float), devMemcpyD2H));
    CHECK(devMemcpy(out64.data(), dout64, out64.size() * sizeof(double),
                    devMemcpyD2H));
    out64.pop_back();
    return save(argv[4], out) && save(argv[5], out64) ? 0 : 1;
}}
"""


def _toolchain(backend: str, request: pytest.FixtureRequest):
    tc = request.getfixturevalue("nvcc" if backend == "cuda" else "hipcc")
    request.getfixturevalue(f"{backend}_device")
    return tc


def _on_card(backend: str, kernel: Kernel, data: dict, request,
             compile_device_source, run_device_binary,
             tmp_path: Path) -> dict[str, np.ndarray]:
    emit, arch, ext = BACKEND[backend]
    tc = _toolchain(backend, request)
    kpath = tmp_path / f"{kernel.name}.{ext}"
    kpath.write_text(emit(kernel, arch))
    params = {p.name: p for p in kernel.params}
    n = data["x"].size
    lc = launch(arch.wavefront, n)
    cols = params["out"].shape[1].value
    cols64 = params["out64"].shape[1].value if "out64" in params else 0
    binary = compile_device_source(tc, HARNESS.format(
        kernel=kpath, n=n, cols=cols, cols64=cols64, name=kernel.name,
        grid=lc.grid, block=lc.block[0],
        out64_arg=", dout64" if cols64 else ""))
    files = [tmp_path / f"{k}.bin" for k in ("x", "y", "d", "out", "out64")]
    for f, k in zip(files, ("x", "y", "d"), strict=False):
        data[k].tofile(f)
    run_device_binary(binary, *map(str, files))
    got = {"out": np.fromfile(files[3], np.float32).reshape(n, cols)}
    if cols64:
        got["out64"] = np.fromfile(files[4], np.float64).reshape(n, cols64)
    return got


def _column_errors(got: np.ndarray, want: np.ndarray) -> list[float]:
    """max |got - want| / scale per column, scale = the column's max |want|."""
    errs = []
    for j in range(want.shape[1]):
        scale = float(np.max(np.abs(want[:, j].astype(np.float64))))
        assert scale > 0, f"column {j} is all zero: it witnesses nothing"
        errs.append(float(np.max(np.abs(got[:, j].astype(np.float64)
                                        - want[:, j]))) / scale)
    return errs


def _unrounded(kernel: Kernel, data: dict, lanes: int,
               monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    with monkeypatch.context() as m:
        m.setattr(interp, "_round32", lambda x: x)
        return run(kernel, data, lanes)


def _exact(backend: str, request, compile_device_source, run_device_binary,
           tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel, data = exact_kernel(), inputs()
    lanes = BACKEND[backend][1].wavefront
    card = _on_card(backend, kernel, data, request, compile_device_source,
                    run_device_binary, tmp_path)
    want = run(kernel, data, lanes)
    for k in want:
        bad = np.argwhere(card[k] != want[k])
        assert bad.size == 0, (
            f"{backend} {k}: the card differs from the interpreter at "
            f"(element, column) {bad[:8].tolist()}: card "
            f"{card[k][tuple(bad[0])]!r}, interpreter {want[k][tuple(bad[0])]!r}")
    # The mutation: without the rounding the interpreter misses the card.
    cold = _unrounded(kernel, data, lanes, monkeypatch)
    assert not np.array_equal(cold["out"], card["out"])
    errs = _column_errors(cold["out"], card["out"])
    for j in DISCRIMINATING[kernel.name]:
        assert errs[j] > F32_REL, (j, errs[j])
    print(f"\n{backend} f32_exact: bitwise on {card['out'].size} + "
          f"{card['out64'].size} values; unrounded interpreter off by "
          f"{max(errs):.2e} of scale")


def _contracting(backend: str, request, compile_device_source,
                 run_device_binary, tmp_path: Path,
                 monkeypatch: pytest.MonkeyPatch) -> None:
    kernel, data = contracting_kernel(), inputs()
    lanes = BACKEND[backend][1].wavefront
    card = _on_card(backend, kernel, data, request, compile_device_source,
                    run_device_binary, tmp_path)["out"]
    want = run(kernel, data, lanes)["out"]
    for j in range(want.shape[1]):
        assert_within(card[:, j], want[:, j],
                      float(np.max(np.abs(want[:, j].astype(np.float64)))),
                      precision=Scalar.F32, what=f"{backend} column {j}")
    errs = _column_errors(card, want)
    ulps = max(errs) / 2.0**-23
    cold = _unrounded(kernel, data, lanes, monkeypatch)["out"]
    cold_errs = _column_errors(cold, card)
    for j in DISCRIMINATING[kernel.name]:
        assert cold_errs[j] > F32_REL, (j, cold_errs[j])
    print(f"\n{backend} f32_contracting: worst card-vs-interpreter "
          f"{max(errs):.2e} of scale (~{ulps:.1f} float32 ulps; tier "
          f"{F32_REL:.2e}); bitwise in {sum(e == 0 for e in errs)}/"
          f"{len(errs)} columns; unrounded interpreter off by "
          f"{max(cold_errs):.2e}")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_f32_exact_is_the_interpreter_bitwise(
        request, compile_device_source, run_device_binary, tmp_path,
        monkeypatch) -> None:
    _exact("cuda", request, compile_device_source, run_device_binary,
           tmp_path, monkeypatch)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_f32_exact_is_the_interpreter_bitwise(
        request, compile_device_source, run_device_binary, tmp_path,
        monkeypatch) -> None:
    _exact("hip", request, compile_device_source, run_device_binary,
           tmp_path, monkeypatch)


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_f32_contracting_meets_the_fp32_tier(
        request, compile_device_source, run_device_binary, tmp_path,
        monkeypatch) -> None:
    _contracting("cuda", request, compile_device_source, run_device_binary,
                 tmp_path, monkeypatch)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_f32_contracting_meets_the_fp32_tier(
        request, compile_device_source, run_device_binary, tmp_path,
        monkeypatch) -> None:
    _contracting("hip", request, compile_device_source, run_device_binary,
                 tmp_path, monkeypatch)
