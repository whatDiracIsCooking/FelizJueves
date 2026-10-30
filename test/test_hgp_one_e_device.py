"""The HGP one-electron kernels on the card (#598): `S`, `T`, `V` and the
dipole for s and p shells against the MMD host and libcint.

THE BUILD is `hgp_one_e_driver.build_on_card`, on the shared runner (#658):
one runner per class's `kernel_hgp` unit over `one_electron_launch`'s plan
(`tau = 0`), compiled with devtools/config.sh's flags and FAILING on a
warning located in generated text or a harness header (#92); every class
launched once into one output carried from class to class from zero, each
launch `RUNS` times.

THE BOUNDS, stated before any run, test_kernel_hgp.py's own, unchanged: the
MMD host within `HOST_TOL = 1e-13` of `max|M|` (another recurrence, so not
bitwise), libcint within `LIBCINT_TOL = 1e-12` of `max|M|`. The device
`fj_boys`' own error is not added: a Boys error that mattered would fail
here.

THE RUN-TO-RUN WITNESS: plain stores, one owner per element -- the runner
fails unless every run of a class is bit-identical to its first, and every
matrix is asserted exactly symmetric.

SCOPE: s and p shells only (`kernel_hgp.L_MAX = 1`): the synthetic off-axis
basis (every ordered class (ss) (sp) (ps) (pp), four nuclei), and H2O
STO-3G and 6-31G, all s and p. NOT A MEASUREMENT.
"""

from __future__ import annotations

import numpy as np
import pytest
from harness_driver import build_chain
from hgp_one_e_driver import build_on_card, hgp_one_e_steps
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import BACKEND, _toolchain
from test_kernel_hgp import (
    H2O,
    HOST_TOL,
    LIBCINT_TOL,
    NUCLEI,
    ORIGIN,
    SYNTHETIC,
    ZERO,
    _h2o,
    _host,
    _rel,
)

from felizjueves.ao.pairs import overlap_bound
from felizjueves.eri.j_coulomb.launch import one_electron_launch
from felizjueves.one_electron import kernel_hgp as kh
from felizjueves.spec import Target

FIXTURE_ARRAY = {Target.S: "int1e_ovlp", Target.T: "int1e_kin",
                 Target.V: "int1e_nuc"}
TARGETS = [t.value for t in kh.TARGETS]


def _plan(basis, nuclei):
    return one_electron_launch(basis, nuclei, tau=0.0, bound=overlap_bound)


def _on_card(backend, basis, nuclei, target, request,
             tmp_path, origin=ORIGIN):
    tc = _toolchain(backend, request)
    plan = _plan(basis, nuclei)
    M = build_on_card(plan, target, basis, backend, tc, tmp_path,
                      origin=origin)
    assert np.array_equal(M, np.swapaxes(M, -1, -2)), "not symmetric"
    return M, plan


def _synthetic(backend, tname, request, tmp_path):
    target = Target(tname)
    M, plan = _on_card(backend, SYNTHETIC, NUCLEI, target, request, tmp_path)
    assert {(c.la, c.lb) for c in plan} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    rel = _rel(M, _host(SYNTHETIC, NUCLEI, target))
    print(f"\n{backend} {BACKEND[backend][1].name} {target.value} synthetic: "
          f"|M - MMD host| = {rel:.2e} of max (tol {HOST_TOL})")
    assert rel <= HOST_TOL


def _fixture(backend, tname, name, request, tmp_path):
    target = Target(tname)
    label = f"\n{backend} {BACKEND[backend][1].name} {target.value} {name}"
    if target is Target.DIPOLE:
        p = manifest.dipole_pair(name)
        basis = ob.from_fixture(p.dipole)
        assert max(s.l for s in basis.shells) <= kh.L_MAX
        for C, want in ((p.origin, p.r), (ZERO, p.r0)):
            M, _ = _on_card(backend, basis, (), target, request,
                            tmp_path / str(C == ZERO), origin=C)
            rel, host = _rel(M, want), _rel(M, _host(basis, (), target, C=C))
            print(f"{label} C={tuple(C)}: |MU - int1e_r_cart| = {rel:.2e}, "
                  f"|MU - MMD host| = {host:.2e} of max")
            assert rel <= LIBCINT_TOL
            assert host <= HOST_TOL
        return
    fx, basis, nuclei = _h2o(name)
    assert max(s.l for s in basis.shells) <= kh.L_MAX
    M, _ = _on_card(backend, basis, nuclei, target, request, tmp_path)
    rel = _rel(M, fx.arrays[FIXTURE_ARRAY[target]])
    host = _rel(M, _host(basis, nuclei, target))
    print(f"{label}: |M - {FIXTURE_ARRAY[target]}| = {rel:.2e}, "
          f"|M - MMD host| = {host:.2e} of max")
    assert rel <= LIBCINT_TOL
    assert host <= HOST_TOL


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("tname", TARGETS)
def test_cuda_hgp_one_e_matches_the_mmd_host(tname, request, tmp_path):
    _synthetic("cuda", tname, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("tname", TARGETS)
def test_hip_hgp_one_e_matches_the_mmd_host(tname, request, tmp_path):
    _synthetic("hip", tname, request, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", H2O)
@pytest.mark.parametrize("tname", TARGETS)
def test_cuda_hgp_one_e_matches_libcint(tname, name, request, tmp_path):
    _fixture("cuda", tname, name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", H2O)
@pytest.mark.parametrize("tname", TARGETS)
def test_hip_hgp_one_e_matches_libcint(tname, name, request, tmp_path):
    _fixture("hip", tname, name, request, tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize("tname", TARGETS)
def test_cuda_hgp_one_e_program_compiles(tname, nvcc, tmp_path):
    build_chain(nvcc, "cuda", hgp_one_e_steps(_plan(SYNTHETIC, NUCLEI),
                                              Target(tname), SYNTHETIC),
                tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("tname", TARGETS)
def test_hip_hgp_one_e_program_compiles(tname, hipcc, tmp_path):
    build_chain(hipcc, "hip", hgp_one_e_steps(_plan(SYNTHETIC, NUCLEI),
                                             Target(tname), SYNTHETIC),
                tmp_path)
