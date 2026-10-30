"""The HGP nuclear-attraction gradient on the card: `ghv_` (`V` at d1 on
`Engine.HGP`, s and p shells) against the MMD host and PySCF.

THE BUILD is `ghv_driver.gradient_on_card`, on the shared runner: one
runner per class's `kernel_ghv` unit, compiled with devtools/config.sh's
flags and FAILING on a warning located in generated text or a harness
header (#92); every class launched once into one `G` carried from class to
class from zero, the chain run `RUNS` times.

THE BOUNDS, stated before any run: the MMD host
(`gradient.host.potential_gradient`) within `DEV_TOL = 1e-13` of `max|G|`
-- test_grad_device.py's bound for `gv_`: the interpreted kernel is within
test_kernel_ghv.py's `HOST_TOL` (measured 1.6e-15), and on the card the
atomics add in an order the hardware chooses and the vendor's `exp` and
Boys differ from the interpreter's by a few ulps; PySCF's `potential` term
(test_rhf_gradient.py's witness) within its `ABS_TOL = 1e-10` absolute.
The runs are compared with each other too, and printed.

SCOPE: s and p shells only (`kernel_hgp.L_MAX = 1`): the synthetic
off-axis basis of test_kernel_hgp.py (every ordered class, a fourth
nucleus with no shell), and H2O STO-3G and 6-31G. NOT A MEASUREMENT.
"""

from __future__ import annotations

import numpy as np
import pytest
from ghv_driver import ghv_steps, gradient_on_card
from harness_driver import build_steps
from test_kernel_device import BACKEND, _toolchain
from test_kernel_ghv import H2O, _host, _m
from test_kernel_hgp import NUCLEI, SYNTHETIC
from test_rhf_gradient import ABS_TOL, _witness

DEV_TOL = 1e-13  # of max|G| (module docstring)


def _on_card(backend, basis, nuclei, M, request, tmp_path) -> np.ndarray:
    tc = _toolchain(backend, request)
    Gs = gradient_on_card(basis, nuclei, M, backend, tc, tmp_path)
    want = _host(basis, nuclei, M)
    scale = np.abs(want).max()
    errs = [float(np.abs(G - want).max() / scale) for G in Gs]
    spread = float(np.abs(Gs[0] - Gs[-1]).max() / scale)
    print(f"\n{backend} {BACKEND[backend][1].name} ghv: |G - host| / max|G| "
          f"per run {errs}; run-to-run {spread:.2e}; bit-identical "
          f"{bool(np.array_equal(Gs[0], Gs[-1]))}")
    assert max(errs) <= DEV_TOL
    return Gs


def _synthetic(backend, request, tmp_path) -> None:
    steps = ghv_steps(SYNTHETIC, NUCLEI, _m(SYNTHETIC.n_ao))
    assert [s.kernel.name for s in steps] == [
        "ghv_00", "ghv_01", "ghv_10", "ghv_11"]
    _on_card(backend, SYNTHETIC, NUCLEI, _m(SYNTHETIC.n_ao), request,
             tmp_path)


def _pyscf(backend, name, request, tmp_path) -> None:
    w, basis, nuclei = _witness(name)
    Gs = _on_card(backend, basis, nuclei, w.density, request, tmp_path)
    err = max(float(np.abs(G - w.terms["potential"]).max()) for G in Gs)
    print(f"{backend} {name}: |G - PySCF potential term| = {err:.2e}")
    assert err <= ABS_TOL


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_ghv_matches_the_mmd_host(request, tmp_path) -> None:
    _synthetic("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_ghv_matches_the_mmd_host(request, tmp_path) -> None:
    _synthetic("hip", request, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", H2O)
def test_cuda_ghv_matches_pyscf(name, request, tmp_path) -> None:
    _pyscf("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", H2O)
def test_hip_ghv_matches_pyscf(name, request, tmp_path) -> None:
    _pyscf("hip", name, request, tmp_path)


@pytest.mark.cuda
def test_cuda_ghv_program_compiles(nvcc, tmp_path) -> None:
    build_steps(nvcc, "cuda", ghv_steps(SYNTHETIC, NUCLEI, _m(SYNTHETIC.n_ao)),
                tmp_path)


@pytest.mark.hip
def test_hip_ghv_program_compiles(hipcc, tmp_path) -> None:
    build_steps(hipcc, "hip", ghv_steps(SYNTHETIC, NUCLEI, _m(SYNTHETIC.n_ao)),
                tmp_path)


def test_every_step_names_the_kernels_arguments() -> None:
    """Every launch's runner spec, built host-side: each parameter named
    once, of its own dtype, `G` the one output -- no compiler, no card."""
    from harness_driver import chain_specs

    steps = ghv_steps(SYNTHETIC, NUCLEI, _m(SYNTHETIC.n_ao))
    specs = chain_specs(steps, "G", np.zeros((len(NUCLEI), 3)))
    for s, spec in zip(steps, specs, strict=True):
        assert [a["name"] for a in spec["args"]] == [
            p.name for p in s.kernel.params]
        assert [a["name"] for a in spec["args"] if "out" in a] == ["G"]
