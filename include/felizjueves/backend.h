/*
 * felizjueves/backend.h -- which vendor is compiling a felizjueves.emit.gpu unit
 *
 * Defines exactly one of FJ_SELECTED_CUDA / FJ_SELECTED_HIP, from the macros
 * the device compiler itself predefines in both its host and device passes:
 * __CUDACC__ (nvcc), __HIP__ (clang -x hip) or __HIPCC__ (hipcc). Nothing else
 * is read, so a build passes no backend flag: the compiler invoked IS the
 * backend. A plain C++ compiler defines none of them, and including this from
 * a host-only translation unit is an error rather than a silent guess.
 *
 * __CUDACC__ is tested first: hipcc on the NVIDIA platform drives nvcc, which
 * defines both, and the text it compiles is then CUDA's.
 *
 * FelizJueves depends on no GPU library. This header and runtime.h beside it
 * are the whole of the vendor-neutral layer the `gpu` backend's text needs
 * (src/felizjueves/emit/gpu.py).
 */

#pragma once

#if defined(__CUDACC__)
#define FJ_SELECTED_CUDA 1
#elif defined(__HIP__) || defined(__HIPCC__)
#define FJ_SELECTED_HIP 1
#else
#error "felizjueves/backend.h: no device compiler -- compile with nvcc, hipcc or clang -x hip"
#endif
