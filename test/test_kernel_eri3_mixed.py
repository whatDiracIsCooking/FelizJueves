"""The mixed-precision spherical three-center kernels, interpreted:
`build_eri3_kernel(cls, packed=True, spherical_aux=True, mixed=True)`, with
`spherical_bra` (`mssperi3_`) and with `rolled` (`mrssperi3_`).

THE FORM (#346's pattern, j_coulomb/levers/mixed.py): the tower `R`, the
column sums `U` and the `E . U` contraction are F32; `PQ` and `theta` are
formed in F64 and narrowed, `F_m(T)` is evaluated in F64 and narrowed, the
bra tables are narrowed once per bra record; each contraction is widened
before it meets `w`, and the bra transform, `w` and the accumulation in
the output stay F64. The interpreter rounds F32 where the device does.

THE ORACLE is the host's FP64 Cartesian block contracted on all three
indices (test_kernel_eri3_fully_spherical.py's `host`), at the repo's
declared FP32 tier: `F32_REL` (test/oracle/tolerance.py, the fp64 gate
scaled by 2^29, derived and never tuned) times the contracted scale. The
NaN fill and the diagonal symmetry are held exactly, as in the FP64 form.

SCOPE (minimal): (pp|f), (dd|d), (ff|f) on the fully spherical test bases,
and the rolled form bitwise the unrolled at (dd|d): both narrow the same
coefficient to the same float and fold the same terms in the same order.
The card run is test_eri3_mixed_device.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_kernel_eri3_fully_spherical as fs
from oracle.tolerance import F32_REL

from felizjueves.eri.density_fit.eri3_drive import launch_packed
from felizjueves.eri.density_fit.kernel_eri3 import (
    OUT,
    R_ARRAY,
    U_ARRAY,
    build_eri3_kernel,
)
from felizjueves.ir import Decl, Scalar, walk
from felizjueves.reference.boys import boys

CLASSES = [(1, 1, 3), (2, 2, 2), (3, 3, 3)]


def run(am, rolled=False):
    cls = fs._cls(am)
    cells, _, n_col, n_aux, _ = fs._layout(am)
    return launch_packed(cls, cells, np.full((n_col, n_aux), np.nan),
                         boys=boys, threads=5, spherical_aux=True,
                         spherical_bra=True, rolled=rolled, mixed=True)


def check_f32(am, got) -> float:
    """Every cell within `F32_REL` of the contracted scale, a diagonal
    pair's block symmetric, exactly the class's elements stored: the worst
    error in units of `F32_REL`."""
    cls = fs._cls(am)
    _, cols, _, n_aux, offs = fs._layout(am)
    got = got.reshape(-1)
    stored = np.zeros(got.shape, dtype=bool)
    worst = 0.0
    for k in range(cls.n_cell):
        want, bound, at = fs.host(am, k, cols, n_aux, offs)
        g = got[at]
        err = np.abs(g - want)
        assert np.all(err <= F32_REL * bound), (am, cls.cells[k])
        pos = bound > 0
        worst = max(worst, float(np.max(err[pos] / bound[pos], initial=0.0)))
        a_sh, b_sh, _ = cls.cells[k]
        if a_sh == b_sh:
            assert np.array_equal(g, g.transpose(1, 0, 2))
        stored[at] = True
    assert not np.isnan(got[stored]).any()
    assert np.isnan(got[~stored]).all()
    return worst / F32_REL


@pytest.mark.parametrize("am", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_the_mixed_block_is_the_contracted_host_at_the_fp32_tier(am):
    print(f"\n{am}: worst {check_f32(am, run(am)):.1e} of F32_REL")


def test_the_rolled_mixed_kernel_is_the_unrolled_one_bitwise():
    assert np.array_equal(run((2, 2, 2), rolled=True), run((2, 2, 2)),
                          equal_nan=True)


def test_the_fp32_region_is_the_tower_and_the_output_stays_fp64():
    k = build_eri3_kernel((3, 3, 6), packed=True, spherical_aux=True,
                          spherical_bra=True, mixed=True)
    assert k.name == "mssperi3_336"
    d = {s.name: s.type for s in walk(k.body) if isinstance(s, Decl)}
    assert d[R_ARRAY] is d[U_ARRAY] is Scalar.F32
    assert next(p for p in k.params if p.name == OUT).type is Scalar.F64
    with pytest.raises(ValueError, match="spherical_aux"):
        build_eri3_kernel((1, 1, 1), packed=True, mixed=True)
