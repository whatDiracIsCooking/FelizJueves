"""The d1 gradient kernels ON A CARD: `G = SUM_mn M_mn dX_mn/dR` for `S`,
`T` and `V` (Hellmann-Feynman included), and `dE_J/dR` from `gjab_`,
against `gradient.host` (generator `test/grad_driver.py`).

PER MOLECULE, H2O/STO-3G and H2O/cc-pVDZ (`slow`, for its input size and
its (dd) classes, L + 1 = 5), each fixture's own density as `M` (the
recorded symmetric Hilbert matrix; `S`'s `M` is the same matrix -- the
kernel contracts any `M`). One program per (molecule, target), every pair
class of the basis in it, `RUNS` runs into a zeroed `G`.

THE BOUND, stated before any run: `|G_dev - G_host| <= DEV_TOL max|G|`,
`DEV_TOL = 1e-13`. The interpreted kernels are within `kernel_g1e`'s /
`kernel_gjab`'s 64 eps of `max|G|` of the host (test_kernel_g1e.py,
test_kernel_gjab.py); on the card the atomics add the records of an atom
in an order the hardware chooses, and the vendor's `exp` and Boys differ
from the interpreter's by a few ulps -- `1e-13` (~450 eps) holds that with
room and would still catch any wrong term (a gradient term is >= 1e-6 of
`max|G|` on these inputs). The runs are compared with each other too.

`gjab`'s `Jp` is the host's at `L + 1` (grad_driver.py's docstring).

THE SESSION THAT BUILT THESE KERNELS HAD NO DEVICE COMPILER AND NO CARD:
this file is written and has NOT been run on either card.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from grad_driver import compile_grad, gjab_launches, one_electron_launches
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import _toolchain

from felizjueves.gradient import host
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Target

DEV_TOL = 1e-13  # of max|G| (module docstring)
MOLECULES = [pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
             pytest.param("h2o_ccpvdz_j", id="h2o_ccpvdz",
                          marks=(pytest.mark.slow,))]
TARGETS = ("s", "t", "v", "j")


@cache
def _case(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    return ob.from_fixture(fx), nuclei, np.asarray(fx.arrays["density"])


def _host(name: str, which: str) -> np.ndarray:
    basis, nuclei, D = _case(name)
    if which == "s":
        return np.array(host.overlap_gradient(basis, D, nuclei))
    if which == "t":
        return np.array(host.kinetic_gradient(basis, D, nuclei))
    if which == "v":
        return np.array(host.potential_gradient(basis, D, nuclei,
                                                one_centre=one_centre))
    return np.array(host.coulomb_gradient(basis, D, nuclei,
                                          one_centre=one_centre))


def _launches(name: str, which: str):
    basis, nuclei, D = _case(name)
    if which == "j":
        return gjab_launches(basis, nuclei, D, one_centre=one_centre)
    target = {"s": Target.S, "t": Target.T, "v": Target.V}[which]
    return one_electron_launches(basis, nuclei, target, D)


def _check(backend: str, name: str, which: str, request, run_device_binary,
           tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    _, nuclei, _ = _case(name)
    b = compile_grad(_launches(name, which), len(nuclei), backend, tc,
                     tmp_path)
    got = b.run(run_device_binary)
    want = _host(name, which)
    scale = np.abs(want).max()
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    assert max(errs) <= DEV_TOL, (
        f"{backend} {name} {which}: |G - host| / max|G| = {errs} > "
        f"{DEV_TOL:g}")
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{backend} {name} grad {which}: |G - host| / max|G| per run "
          f"{errs}; run-to-run {spread:.2e}; bit-identical "
          f"{bool(np.array_equal(got[0], got[1]))}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("which", TARGETS)
@pytest.mark.parametrize("name", MOLECULES)
def test_cuda_gradient_matches_the_host(name, which, request,
                                        run_device_binary, tmp_path) -> None:
    _check("cuda", name, which, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("which", TARGETS)
@pytest.mark.parametrize("name", MOLECULES)
def test_hip_gradient_matches_the_host(name, which, request,
                                       run_device_binary, tmp_path) -> None:
    _check("hip", name, which, request, run_device_binary, tmp_path)
