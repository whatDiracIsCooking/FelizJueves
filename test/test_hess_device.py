"""The d2 hessian kernels ON A CARD: `H = SUM_mn M_mn d2X_mn/dR dR` for
`S`, `T` and `V` (its operator's centre-nucleus and nucleus-nucleus terms
included), against `hessian.host_1e` (generator `test/hess_driver.py`).

PER MOLECULE, H2O/STO-3G and H2O/cc-pVDZ (`slow`, for its input size and
its (dd) classes, L + 2 = 6), each fixture's own density as `M`. One
program per (molecule, target), every pair class of the basis in it,
`RUNS` runs into a zeroed `H`.

THE BOUND, stated before any run: `|H_dev - H_host| <= DEV_TOL
max(scale)`, `DEV_TOL = 1e-14`, `scale` the host's `absolute=True` twin
(test_kernel_h1e.py's scale: `max|H|` is not one, the same-atom blocks of
a core function cancel to round-off). The scale is large against `max|H|`
here -- on H2O/STO-3G with its density, 2.6e3x for `S`, 5.9e5x for `T`,
1.3e4x for `V` -- so the bound is ~45 eps of it, not the gradient's
1e-13 of `max|G|`: the interpreted kernels are within 0.35 eps of the
scale of the host, and on the card the atomics add the records of an
atom in an order the hardware chooses and the vendor's `exp` and Boys
differ from the interpreter's by a few ulps, every such rounding bounded
by the same scale. The runs are compared with each other too.

THE SESSION THAT BUILT THESE KERNELS HAD NO DEVICE COMPILER AND NO CARD:
this file is written and has NOT been run on either card.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from hess_driver import compile_hess, one_electron_launches
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import _toolchain

from felizjueves.hessian import host_1e as host
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Target

DEV_TOL = 1e-14  # of max(scale), ~45 eps (module docstring)
MOLECULES = [pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
             pytest.param("h2o_ccpvdz_j", id="h2o_ccpvdz",
                          marks=(pytest.mark.slow,))]
TARGETS = ("s", "t", "v")
CASES = [pytest.param(m.values[0], w, id=f"{m.id}-{w}", marks=m.marks)
         for m in MOLECULES for w in TARGETS]


@cache
def _case(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    return ob.from_fixture(fx), nuclei, np.asarray(fx.arrays["density"])


def _host(name: str, which: str, absolute: bool = False) -> np.ndarray:
    basis, nuclei, D = _case(name)
    if which == "s":
        return np.array(host.overlap_hessian(basis, D, nuclei,
                                             absolute=absolute))
    if which == "t":
        return np.array(host.kinetic_hessian(basis, D, nuclei,
                                             absolute=absolute))
    return np.array(host.potential_hessian(basis, D, nuclei,
                                           one_centre=one_centre,
                                           absolute=absolute))


def _check(backend: str, name: str, which: str, request, run_device_binary,
           tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    basis, nuclei, D = _case(name)
    target = {"s": Target.S, "t": Target.T, "v": Target.V}[which]
    b = compile_hess(one_electron_launches(basis, nuclei, target, D),
                     len(nuclei), backend, tc, tmp_path)
    got = b.run(run_device_binary)
    want = _host(name, which)
    scale = _host(name, which, absolute=True).max()
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    assert max(errs) <= DEV_TOL, (
        f"{backend} {name} {which}: |H - host| / scale = {errs} > "
        f"{DEV_TOL:g}")
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{backend} {name} hess {which}: |H - host| / scale per run "
          f"{errs}; run-to-run {spread:.2e}; bit-identical "
          f"{bool(np.array_equal(got[0], got[1]))}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "which"), CASES)
def test_cuda_hessian_matches_the_host(name, which, request,
                                       run_device_binary, tmp_path) -> None:
    _check("cuda", name, which, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "which"), CASES)
def test_hip_hessian_matches_the_host(name, which, request,
                                      run_device_binary, tmp_path) -> None:
    _check("hip", name, which, request, run_device_binary, tmp_path)
