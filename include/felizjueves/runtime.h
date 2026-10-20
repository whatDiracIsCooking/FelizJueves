/*
 * felizjueves/runtime.h -- what a felizjueves.emit.gpu unit needs from its vendor
 *
 * Every unit the `gpu` backend writes includes this header, so one text
 * compiles under nvcc and under hipcc / clang -x hip. The two vendors spell
 * almost everything a generated kernel says identically -- the thread
 * builtins, the qualifiers, __syncthreads, atomicAdd on double, libm, the
 * __shfl_*_sync shuffles -- and this header covers the three places they
 * do not, plus <stdint.h>, since a unit spells every integer int32_t:
 *
 *   the runtime     hipcc does not pre-include the HIP runtime the way nvcc
 *                   pre-includes CUDA's for a .cu, and the shuffles live there;
 *   FJ_LANE_MASK_T  the type of a _sync shuffle's lane mask: uint32_t
 *                   under CUDA, uint64_t under HIP (ROCm static-asserts
 *                   that it is 8 bytes);
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
 * UNLESS THE UNIT WAS EMITTED any_wavefront (felizjueves.emit.gpu): then
 * FJ_WAVEFRONT IS the width -- its shuffles read FJ_WAVEFRONT and
 * FJ_FULL_LANE_MASK below, and it carries no assert -- so the build's
 * -DFJ_WAVEFRONT must be the arch's true width, as WarpWraps's WWR_WARP_SIZE
 * must be. It is a configure-time -D because that is the one spelling the
 * same in both of HIP's compile passes.
 *
 *   FJ_FULL_LANE_MASK  every lane of an FJ_WAVEFRONT-wide warp, as an
 *                      FJ_LANE_MASK_T: the mask a fixed-width unit spells
 *                      as a literal.
 *
 * INCLUDE IT AT FILE SCOPE FIRST when a consumer #includes units inside
 * namespaces of their own (two units' `fj_boys` collide at namespace scope):
 * the vendor runtime #include must not land inside a namespace. Everything
 * this header defines is a macro, so each unit's own later #include of it is a
 * no-op wherever it sits.
 */

#pragma once

#include "backend.h"

#include <stdint.h>

#if defined(FJ_SELECTED_HIP)
#include <hip/hip_runtime.h>
#define FJ_LANE_MASK_T uint64_t
#else
// nvcc pre-includes cuda_runtime.h, which declares the shuffles.
#define FJ_LANE_MASK_T uint32_t
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

#if FJ_WAVEFRONT == 64
#define FJ_FULL_LANE_MASK (static_cast<FJ_LANE_MASK_T>(0xffffffffffffffffull))
#else
#define FJ_FULL_LANE_MASK (static_cast<FJ_LANE_MASK_T>(0xffffffffull))
#endif
