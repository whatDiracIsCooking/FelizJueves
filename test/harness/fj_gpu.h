// fj_gpu.h -- the one place the test harness spells a vendor (#649).
//
// Every `cuda*` / `hip*` runtime call the harness makes goes through the
// inline functions below, so device_buffer.h and runner.h say each thing
// once and compile under nvcc and hipcc alike. The selector is the compiler's
// own: hipcc (and `clang -x hip`) defines __HIPCC__, a ROCm build defines
// __HIP_PLATFORM_AMD__; anything else is CUDA. It is the same test as
// include/felizjueves/backend.h's, spelled here so the harness needs no -I.
//
// Plain C++17 (C++20 compatible): no modules, no `import std`, nothing from
// FreeBreakfast. Header-only. Errors are reported by `check`, which throws
// std::runtime_error naming the call: the runner catches it in main() and
// exits 1, so a failed launch can never read as a written output.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define FJ_GPU_VENDOR "hip"
#else
#include <cuda_runtime.h>
#define FJ_GPU_VENDOR "cuda"
#endif

namespace fj {
namespace gpu {

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
using error_t = hipError_t;
using stream_t = hipStream_t;
inline const char* error_string(error_t e) { return hipGetErrorString(e); }
inline bool ok(error_t e) { return e == hipSuccess; }
inline error_t malloc(void** p, std::size_t n) { return hipMalloc(p, n); }
inline error_t free(void* p) { return hipFree(p); }
inline error_t memset(void* p, int v, std::size_t n) { return hipMemset(p, v, n); }
inline error_t to_device(void* d, const void* h, std::size_t n) {
    return hipMemcpy(d, h, n, hipMemcpyHostToDevice);
}
inline error_t to_host(void* h, const void* d, std::size_t n) {
    return hipMemcpy(h, d, n, hipMemcpyDeviceToHost);
}
inline error_t launch(const void* fn, dim3 grid, dim3 block, void** args,
                      std::size_t shared, stream_t stream) {
    return hipLaunchKernel(fn, grid, block, args, shared, stream);
}
inline error_t last_error() { return hipGetLastError(); }
inline error_t synchronize() { return hipDeviceSynchronize(); }
inline error_t device_count(int* n) { return hipGetDeviceCount(n); }
inline error_t set_device(int d) { return hipSetDevice(d); }
using event_t = hipEvent_t;
inline error_t event_create(event_t* e) { return hipEventCreate(e); }
inline error_t event_destroy(event_t e) { return hipEventDestroy(e); }
inline error_t event_record(event_t e) { return hipEventRecord(e, nullptr); }
inline error_t elapsed_ms(float* ms, event_t a, event_t b) { return hipEventElapsedTime(ms, a, b); }
#else
using error_t = cudaError_t;
using stream_t = cudaStream_t;
inline const char* error_string(error_t e) { return cudaGetErrorString(e); }
inline bool ok(error_t e) { return e == cudaSuccess; }
inline error_t malloc(void** p, std::size_t n) { return cudaMalloc(p, n); }
inline error_t free(void* p) { return cudaFree(p); }
inline error_t memset(void* p, int v, std::size_t n) { return cudaMemset(p, v, n); }
inline error_t to_device(void* d, const void* h, std::size_t n) {
    return cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
}
inline error_t to_host(void* h, const void* d, std::size_t n) {
    return cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
}
inline error_t launch(const void* fn, dim3 grid, dim3 block, void** args,
                      std::size_t shared, stream_t stream) {
    return cudaLaunchKernel(fn, grid, block, args, shared, stream);
}
inline error_t last_error() { return cudaGetLastError(); }
inline error_t synchronize() { return cudaDeviceSynchronize(); }
inline error_t device_count(int* n) { return cudaGetDeviceCount(n); }
inline error_t set_device(int d) { return cudaSetDevice(d); }
using event_t = cudaEvent_t;
inline error_t event_create(event_t* e) { return cudaEventCreate(e); }
inline error_t event_destroy(event_t e) { return cudaEventDestroy(e); }
inline error_t event_record(event_t e) { return cudaEventRecord(e, nullptr); }
inline error_t elapsed_ms(float* ms, event_t a, event_t b) { return cudaEventElapsedTime(ms, a, b); }
#endif

// Throw, naming `what`, unless `e` is success.
inline void check(error_t e, const char* what) {
    if (!ok(e)) {
        throw std::runtime_error(std::string(FJ_GPU_VENDOR) + ": " + what +
                                 ": " + error_string(e));
    }
}

}  // namespace gpu
}  // namespace fj
