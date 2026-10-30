"""The Rys three-center DF integral `(mn|P)` on the card: every class
`la + lb + lP <= 4` against the host, and H2O/STO-3G x cc-pVTZ-jkfit
against the host and libcint's `int3c2e_cart` -- its L <= 4 classes, and
(`slow`) the whole tensor through (pp|g) at L = 6 -- on the RTX 3080 and
the RX 9060 XT.

THE BUILD is the MMD tests' harness with `engine=Engine.RYS`
(`eri3_driver.build_on_card`, on the shared runner): one runner over every
class's `kernel_reri3` unit (each carrying `fj_rys<N>`), compiled with
devtools/config.sh's flags and FAILING on a warning located in generated
text (#92), and one sequence launching every class once into one zeroed
`n_ao x n_ao x n_aux` tensor, `RUNS` times. The signatures are the MMD
kernel's (the harness checks them), so nothing else changes.

THE BOUNDS, per element, stated before any run: the interpreted Rys
kernel's (test_kernel_reri3.py), unchanged -- `|T_dev - T_host| <=
TOL_ULPS eps scale`, `TOL_ULPS = 16`, `scale` the MMD tests' tower-twin
absolute-value sum; against libcint `TOL = 1e-12` of the fixture's
`max |.|`. As in test_eri3_device.py, the device rule's own error is NOT
added (its device text run as host C++ is bitwise the host rule,
test_rys_roots.py): an error that mattered would fail here. An element no
launched cell owns must come back exactly 0.0.

THE RUN-TO-RUN WITNESS: plain stores, one owner per element -- the runner
fails unless every run is bit-identical, and the tensor is asserted exactly
symmetric in `(m, n)`.

NOT A MEASUREMENT: nothing is written to `data/measured/`.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build every program. The
device tests are `gpu` plus `cuda` or `hip`, behind the card lock and the
VRAM probe: without the card they skip, and kernel-tier.sh exits 2.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_kernel_eri3 as te
from eri3_driver import build_on_card, eri3_program
from plan_driver import compile_checked
from test_kernel_device import BACKEND, _toolchain
from test_kernel_hgp_df import EPS, TOL, TOL_ULPS
from tiers import FAST_L_MAX, by_l

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.spec import Engine

RYS = Engine.RYS
FIXTURE_L4 = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
              (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
              (1, 1, 0), (1, 1, 1), (1, 1, 2)]
FIXTURE_ALL = sorted(FIXTURE_L4 + [(0, 1, 4), (1, 1, 3), (1, 1, 4)])
# (name, the top L it launches): the synthetic pair at L <= 4, the fixture
# at L <= 4 and whole (L = 6, `slow` by #419's rule)
CASES = {"synthetic": FAST_L_MAX, "fixture": FAST_L_MAX, "fixture-all": 6}
PARAMS = [by_l(n, L, id=n) for n, L in CASES.items()]


def _bases(name: str) -> tuple[Basis, Basis]:
    if name == "synthetic":
        return te.SYNTHETIC, te.SYNTHETIC_AUX
    return te._basis(), te._aux()


def _classes(name: str):
    return [c for c in eri3_classes(*_bases(name))
            if c.la + c.lb + c.lP <= CASES[name]]


def _expected(name: str):
    return {"synthetic": te.CLASSES, "fixture": FIXTURE_L4,
            "fixture-all": FIXTURE_ALL}[name]


@cache
def _host_of(name: str):
    """`(want, scale, owned)` of the L <= 4 classes, `test_kernel_eri3`'s."""
    return te._host(*_bases(name), {c for c in _expected(name)
                                    if sum(c) <= FAST_L_MAX})


def _run(backend: str, name: str, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    basis, aux = _bases(name)
    classes = _classes(name)
    assert sorted((c.la, c.lb, c.lP) for c in classes) == _expected(name)
    T = build_on_card(classes, basis.n_ao, aux.n_ao, backend, tc, tmp_path,
                      engine=RYS)
    label = f"{backend} {BACKEND[backend][1].name} reri3 {name}"
    assert np.array_equal(T, T.transpose(1, 0, 2))
    want, scale, owned = _host_of(name)
    if name == "fixture-all":  # every element; the host's stops at L = 4
        ref = te._ts().tensor
        lib = float(np.max(np.abs(T - ref))) / float(np.max(np.abs(ref)))
        print(f"\n{label}: {len(classes)} classes; max |T - int3c2e_cart| "
              f"= {lib:.2e} of max (tol {TOL:g}); runs bit-identical")
        assert lib <= TOL, lib
    err = np.abs(T - want)
    pos = owned & (scale > 0)
    worst = float(np.max(err[pos] / (EPS * scale[pos])))
    print(f"\n{label}: {len(classes)} classes, "
          f"{sum(c.n_cell for c in classes)} cells; max |T - host| = "
          f"{err[owned].max():.2e}, worst {worst:.2f} eps scale (tol "
          f"{TOL_ULPS}); runs bit-identical (the runner's witness)")
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), worst
    if name == "fixture-all":
        return
    assert np.all(T[~owned] == 0.0)
    if name == "fixture":
        ref = te._ts().tensor
        lib = float(np.max(np.abs(T - ref)[owned])) / float(
            np.max(np.abs(ref)))
        print(f"{label}: max |T - int3c2e_cart| = {lib:.2e} of max over "
              f"the {int(owned.sum())} owned elements (tol {TOL:g})")
        assert lib <= TOL, lib


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", PARAMS)
def test_cuda_rys_eri3_matches_the_host_and_libcint(name, request,
                                                    tmp_path):
    _run("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", PARAMS)
def test_hip_rys_eri3_matches_the_host_and_libcint(name, request,
                                                   tmp_path):
    _run("hip", name, request, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`) and the fast-tier pieces
# ---------------------------------------------------------------------------


def _program(name: str, backend: str, tmp_path: Path):
    basis, aux = _bases(name)
    return eri3_program(_classes(name), basis.n_ao, aux.n_ao, backend,
                        tmp_path, engine=RYS)


@pytest.mark.cuda
@pytest.mark.parametrize("name", PARAMS)
def test_cuda_rys_eri3_programs_compile(name, nvcc, tmp_path):
    compile_checked(nvcc, _program(name, "cuda", tmp_path), tmp_path / "bin")


@pytest.mark.hip
@pytest.mark.parametrize("name", PARAMS)
def test_hip_rys_eri3_programs_compile(name, hipcc, tmp_path):
    compile_checked(hipcc, _program(name, "hip", tmp_path), tmp_path / "bin")


@pytest.mark.parametrize("name", list(CASES))
def test_the_programs_launch_the_rys_units_of_their_classes(name, tmp_path):
    prog = _program(name, "cuda", tmp_path)
    units = {p.name for p in prog.units if p.suffix == ".cu"}
    assert units == {f"reri3_{''.join(map(str, c))}.cu"
                     for c in _expected(name)}
    assert "::reri3_" in prog.source  # the runner launches the Rys units
    assert "::eri3_" not in prog.source and "::heri3_" not in prog.source
