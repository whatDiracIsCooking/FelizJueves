"""The one-electron derivative matrices (#687) ON A CARD: `ips_`, `ipt_`,
`ipv_` (the ip1 form, `X1[k, m, n] += dX_mn/dA_k`) and `iprv_` (`V`'s
operator term at each nucleus), against the host (`gradient.host`'s
`*_ip1` and `potential_rinv`) and libcint (`int1e_ipovlp` / `ipkin` /
`ipnuc`, negated, and `-Z_C (iprinv + iprinv^T)`), generator
`test/ip1e_driver.py`.

PER MOLECULE, #550's gradient fixtures H2O/STO-3G and H2O/cc-pVDZ (`slow`,
its (dd) classes at L + 1 = 5), Cartesian. One program per (molecule,
matrix), every pair class of the basis in it, `RUNS` runs into a zeroed
matrix; the rinv form once per nucleus.

THE BOUND, stated before any run: `|X_dev - X_host| <= DEV_TOL max|X_host|`,
`DEV_TOL = 1e-14` (~45 eps): interpreted, the kernels are within 2 eps of
the host (test_kernel_ip1e.py); on the card the atomics add a shell's
records in an order the hardware chooses and the vendor's `exp` and Boys
differ from the interpreter's by a few ulps. Against libcint, test
_kernel_ip1e.py's `LIBCINT_TOL = 1e-12` of `max|libcint|`. The runs are
compared with each other too.

NOT YET RUN ON A CARD: written device-less (2026-10-30), its compile-only
twins (`test_X_ip1e_runners_compile`) green under nvcc and hipcc.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from harness_driver import build_chain, chain_specs
from ip1e_driver import compile_ip1e, ip1e_runners, ip1e_steps
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import _toolchain

from felizjueves.gradient import host
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Target

DEV_TOL = 1e-14  # of max|X_host|, ~45 eps (module docstring)
LIBCINT_TOL = 1e-12  # of max|libcint| (test_kernel_ip1e.py)
MOLECULES = [pytest.param("h2o_sto3g_grad", id="h2o_sto3g"),
             pytest.param("h2o_ccpvdz_grad", id="h2o_ccpvdz",
                          marks=(pytest.mark.slow,))]
WHICH = {"s": (Target.S, "ip1", None), "t": (Target.T, "ip1", None),
         "v": (Target.V, "ip1", None),
         **{f"rinv{c}": (Target.V, "rinv", c) for c in range(3)}}
CASES = [pytest.param(m.values[0], w, id=f"{m.id}-{w}", marks=m.marks)
         for m in MOLECULES for w in WHICH]


@cache
def _case(name: str):
    fx = manifest.gradient_fixture(name)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    return ob.from_fixture(fx), nuclei, fx


def _host(name: str, which: str) -> np.ndarray:
    basis, nuclei, _ = _case(name)
    target, form, nuc = WHICH[which]
    if form == "rinv":
        return np.array(host.potential_rinv(basis, nuclei, nuc,
                                            one_centre=one_centre))
    if target is Target.S:
        return np.array(host.overlap_ip1(basis))
    if target is Target.T:
        return np.array(host.kinetic_ip1(basis))
    return np.array(host.potential_ip1(basis, nuclei, one_centre=one_centre))


def _libcint(name: str, which: str) -> np.ndarray:
    _, _, fx = _case(name)
    target, form, nuc = WHICH[which]
    a = fx.arrays
    if form == "rinv":
        ip = a["int1e_iprinv"][nuc]
        return -fx.atoms[nuc].charge * (ip + ip.transpose(0, 2, 1))
    return -a[{Target.S: "int1e_ipovlp", Target.T: "int1e_ipkin",
               Target.V: "int1e_ipnuc"}[target]]


def _steps(name: str, which: str):
    basis, nuclei, _ = _case(name)
    return basis, ip1e_steps(basis, nuclei, *WHICH[which])


def _check(backend: str, name: str, which: str, request,
           tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    basis, steps = _steps(name, which)
    got = compile_ip1e(steps, basis.n_ao, backend, tc, tmp_path).run()
    want, cint = _host(name, which), _libcint(name, which)
    scale, cscale = np.abs(want).max(), np.abs(cint).max()
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    cerrs = [float(np.abs(g - cint).max() / cscale) for g in got]
    assert max(errs) <= DEV_TOL, (
        f"{backend} {name} {which}: |X - host| / max = {errs} > {DEV_TOL:g}")
    assert max(cerrs) <= LIBCINT_TOL, (
        f"{backend} {name} {which}: |X - libcint| / max = {cerrs}")
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{backend} {name} ip1e {which}: |X - host| / max per run "
          f"{errs}; libcint {cerrs}; run-to-run {spread:.2e}; bit-identical "
          f"{bool(np.array_equal(got[0], got[1]))}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "which"), CASES)
def test_cuda_ip1e_matches_the_host(name, which, request, tmp_path) -> None:
    _check("cuda", name, which, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "which"), CASES)
def test_hip_ip1e_matches_the_host(name, which, request, tmp_path) -> None:
    _check("hip", name, which, request, tmp_path)


# --- No card: the runners compile, and every launch's spec is the kernel's ---


def _runners_compile(backend: str, tc, name: str, which: str,
                     tmp_path: Path) -> None:
    _, steps = _steps(name, which)
    bins = build_chain(tc, ip1e_runners(steps, backend, tmp_path), tmp_path)
    assert set(bins) == {s.kernel.name for s in steps}


@pytest.mark.cuda
@pytest.mark.parametrize(("name", "which"), CASES)
def test_cuda_ip1e_runners_compile(name, which, nvcc, tmp_path) -> None:
    _runners_compile("cuda", nvcc, name, which, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("name", "which"), CASES)
def test_hip_ip1e_runners_compile(name, which, hipcc, tmp_path) -> None:
    _runners_compile("hip", hipcc, name, which, tmp_path)


@pytest.mark.parametrize(("name", "which"), CASES)
def test_every_launch_names_the_kernels_arguments(name, which) -> None:
    """Every launch's runner spec, built host-side (harness_driver)."""
    basis, steps = _steps(name, which)
    out = "XC" if which.startswith("rinv") else "X1"
    n = basis.n_ao
    specs = chain_specs(steps, out, np.zeros((3, n, n)))
    for s, spec in zip(steps, specs, strict=True):
        assert [a["name"] for a in spec["args"]] == [
            p.name for p in s.kernel.params]
        assert [a["name"] for a in spec["args"] if "out" in a] == [out]
