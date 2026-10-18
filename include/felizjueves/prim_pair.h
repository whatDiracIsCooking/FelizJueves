/*
 * felizjueves/prim_pair.h -- PrimPairSoA, the kernels' primitive-pair records
 *
 * A primitive-pair class reaches every one-electron kernel (S, T, V and the
 * cross-basis SX) as structure of arrays: one flat n_rec-long array per
 * scalar of a record, each Cartesian quantity split per axis (XPA, YPA,
 * ZPA, not one interleaved [n_rec][3] array), so the lanes of a warp read
 * consecutive doubles. Those arrays cross the call as ONE struct of
 * pointers passed by value:
 *
 *     s_00(int n_rec, int n_ao, PrimPairSoA pp, double* S)
 *
 * It is the superset the four families read; a kernel leaves the fields it
 * does not need unread, so a host fills one value per class and hands it to
 * every family. Since #421 the J chain's DQ and JAB kernels take it too,
 * and K two values of it (pp_ac and pp_bd: its bra and its ket records).
 * The fields (felizjueves/prim_pair.py is their definition):
 *
 *     zeta           alpha + beta
 *     s              c_a c_b (pi / zeta)^{3/2}
 *     eb             beta, the ket primitive's exponent (T)
 *     XP  YP  ZP     P, the product centre (V)
 *     XPA YPA ZPA    P - A
 *     XPB YPB ZPB    P - B
 *     off_a off_b    the two shells' first AO
 *
 * Plain C++, no vendor runtime (only <stdint.h>, for int32_t, the gpu
 * backend's one integer spelling): a host translation unit includes it to
 * fill the struct. The struct text below is emit._c.bundle_struct's, verbatim
 * (test_emit_gpu.py); edit the Python, never this copy alone.
 *
 * INCLUDE IT AT FILE SCOPE FIRST, as runtime.h, when a consumer #includes
 * units inside namespaces of their own: the first unit's #include would
 * otherwise declare the struct inside that unit's namespace, and every
 * later one would find its own include a no-op and the type undeclared.
 */

#pragma once

#include <stdint.h>

// PrimPairSoA: the record arrays, one pointer per field, passed by value
struct PrimPairSoA {
    const double* zeta;
    const double* s;
    const double* eb;
    const double* XP;
    const double* YP;
    const double* ZP;
    const double* XPA;
    const double* YPA;
    const double* ZPA;
    const double* XPB;
    const double* YPB;
    const double* ZPB;
    const int32_t* off_a;
    const int32_t* off_b;
};
