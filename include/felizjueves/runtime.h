/*
 * felizjueves/runtime.h -- what a felizjueves.emit.gpu unit needs from its vendor
 *
 * Every unit the `gpu` backend writes includes this header, so one text
 * compiles under nvcc and under hipcc / clang -x hip. The two vendors spell
 * almost everything a generated kernel says identically -- the thread
 * builtins, the qualifiers, __syncthreads, atomicAdd on double, libm, the
 * __shfl_*_sync shuffles -- and this header covers the three places they
 * do not:
 *
 *   the runtime     hipcc does not pre-include the HIP runtime the way nvcc
 *                   pre-includes CUDA's for a .cu, and the shuffles live there;
 *   FJ_LANE_MASK_T  the type of a _sync shuffle's lane mask: `unsigned int`
 *                   under CUDA, 64-bit under HIP (ROCm static-asserts it);
 *   FJ_WAVEFRONT    the lanes per warp / wavefront of the arch being compiled
 *                   for: 32 on every NVIDIA arch and on RDNA, 64 on CDNA.
 *                   Defaults to 32; pass -DFJ_WAVEFRONT=64 for a wave64 arch.
 *
 * FJ_WAVEFRONT is CHECKED, not adapted to. A unit whose text depends on the
 * width -- one with a warp reduction or a lane broadcast, whose shuffle width
 * and mask are fixed at emission -- static-asserts that FJ_WAVEFRONT is the
 * width it was emitted for, so a wave32 unit built for a wave64 arch fails to
 * compile instead of computing the wrong sum. A unit with no shuffle carries
 * no assert and compiles for either width.
 *
 * INCLUDE IT AT FILE SCOPE FIRST when a consumer #includes units inside
 * namespaces of their own (two units' `fj_boys` collide at namespace scope):
 * the vendor runtime #include must not land inside a namespace. Everything
 * this header defines is a macro, so each unit's own later #include of it is a
 * no-op wherever it sits.
 */

#pragma once

#include "backend.h"

#if defined(FJ_SELECTED_HIP)
#include <hip/hip_runtime.h>
#define FJ_LANE_MASK_T unsigned long long
#else
// nvcc pre-includes cuda_runtime.h, which declares the shuffles.
#define FJ_LANE_MASK_T unsigned int
#endif

#ifndef FJ_WAVEFRONT
#define FJ_WAVEFRONT 32
#endif

#if FJ_WAVEFRONT != 32 && FJ_WAVEFRONT != 64
#error "felizjueves/runtime.h: FJ_WAVEFRONT must be 32 or 64"
#endif

#if defined(FJ_SELECTED_CUDA) && FJ_WAVEFRONT != 32
#error "felizjueves/runtime.h: FJ_WAVEFRONT is 32 on every NVIDIA arch"
#endif
