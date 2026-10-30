"""The ip1 bra half of the Coulomb derivative matrix ON A CARD: `jx_`
(`Target.JX` at d1, #688), against `hessian.host_jx` and PySCF (generator
`test/hess_driver.py`, its one OUT `J1[3, n_ao, n_ao]`).

The inputs are test_kernel_jx.py's, so the card is held to what the
interpreted kernel is held to:

- PER BRA CLASS, every ordered class through (dd) on its three-atom s, p,
  d, f basis: one launch over every record of the class, `Jp` the host's
  (`gradient.host.coulomb_jp` at `L + 1`), against
  `host_jx.coulomb_ip1(..., pairs=)` of the class's pairs.
- THE WHOLE `J1` of H2O over STO-3G and cc-pVDZ (`slow`): one `jx` launch
  per bra class into one `J1`, the host's `Jp`, against PySCF's
  (`oracle.numeric`'s `jx1_*`); then, on STO-3G only, `J^X` assembled
  from it and the host's ket half
  (`host_jx.coulomb_derivative_matrix(parts="ket")`) against PySCF's
  `h1ao` Coulomb part (`jx_*`), and `SUM_X J^X = 0` (cc-pVDZ's host ket
  half is hours of pure Python; test_kernel_jx.py says why).

THE BOUNDS, stated before any run. Per class `|J1_dev - J1_host| <=
DEV_TOL max(scale)`, `DEV_TOL = 1e-14` (~45 eps), `scale` the host's
absolute twin -- test_hess_j_device.py's bound for the same reason (the
interpreted kernel is within 16 eps of it; the card's atomics reorder the
adds and its `exp` differs by ulps). Against PySCF, `oracle.numeric`'s
fp64 tier. The runs are compared with each other too.

NOT RUN ON EITHER CARD: written in a session with no card (both compilers
passed every unit through (ff), test_emit_jx.py). The on-card chain
(`DQ` -> the raised `[p|q]` -> `jx`, and `hdq` -> gather -> `[p|q]` ->
`jab` for the ket half) is interpreted only (test_kernel_jx.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from harness_driver import chain_specs
from hess_driver import HessLaunch, compile_hess, hess_steps
from oracle import numeric as golden
from test_kernel_device import _toolchain
from test_kernel_jx import (
    GOLDEN,
    _arrays,
    _assemble,
    _case,
    _classes,
    _host_j1,
    _host_jp,
)
from tiers import by_l

from felizjueves.gradient import host as gh
from felizjueves.hessian import host_jx
from felizjueves.hessian.kernel_jx import build_jx_kernel
from felizjueves.reference.pq import one_centre

DEV_TOL = 1e-14  # of max(scale), ~45 eps (module docstring)
CLASSES = [by_l((la, lb), la + lb + 1, id=f"{la}{lb}")
           for la in range(3) for lb in range(3)]
MOLECULES = [pytest.param("h2o_sto3g", id="h2o_sto3g",
                          marks=(pytest.mark.slow,)),
             pytest.param("h2o_ccpvdz", id="h2o_ccpvdz",
                          marks=(pytest.mark.slow,))]


def _launch(name: str, la: int, lb: int) -> HessLaunch:
    basis, nuclei, _ = _case(name)
    rec = gh.records(basis, nuclei, la, lb)
    n = len(rec["s_ab"])
    arrs = _arrays(rec) | {"Jp": _host_jp(name, la, lb, rec),
                           "row": np.arange(n, dtype=np.int32)}
    return HessLaunch(build_jx_kernel((la, lb)), n, arrs,
                      {"n_rec": n, "n_ao": basis.n_ao, "n_rows": n},
                      out="J1")


def _run(backend, name, classes, request, tmp_path: Path) -> np.ndarray:
    tc = _toolchain(backend, request)
    n = _case(name)[0].n_ao
    xs = [_launch(name, *c) for c in classes]
    got = compile_hess(xs, 0, backend, tc, tmp_path, n_out=3 * n * n).run()
    return got.reshape(len(got), 3, n, n)


def _report(got, want, scale, label: str, tol: float) -> None:
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    assert max(errs) <= tol, f"{label}: |dev - want| / scale = {errs} > {tol:g}"
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{label}: |dev - want| / scale per run {errs}; run-to-run "
          f"{spread:.2e}; bit-identical {bool(np.array_equal(got[0], got[1]))}")


def _check_class(backend, cls, request, tmp_path: Path) -> None:
    basis = _case("spdf")[0]
    pairs = tuple(gh.class_pairs(basis, *cls))
    got = _run(backend, "spdf", [cls], request, tmp_path)
    _report(got, _host_j1("spdf", pairs),
            float(_host_j1("spdf", pairs, absolute=True).max()),
            f"{backend} jx_{cls[0]}{cls[1]}", DEV_TOL)


def _check_molecule(backend, name, request, tmp_path: Path) -> None:
    basis, nuclei, D = _case(name)
    got = _run(backend, name, _classes(name), request, tmp_path)
    g = GOLDEN[name]
    ket = (np.array(host_jx.coulomb_derivative_matrix(
        basis, D, nuclei, one_centre=one_centre, parts="ket"))
        if name == "h2o_sto3g" else None)
    for r, J1 in enumerate(got):
        e1 = golden.compare(f"jx1_{g}", J1)
        print(f"\n{backend} {name} run {r}: J1 {e1:.2e} of max|PySCF|")
        if ket is None:
            continue
        JX = _assemble(name, J1, ket)
        ex = golden.compare(f"jx_{g}", JX)
        assert np.abs(JX.sum(axis=0)).max() <= 1e-13 * np.abs(JX).max()
        print(f"{backend} {name} run {r}: J^X {ex:.2e} of max|PySCF|")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASSES)
def test_cuda_jx_class_matches_the_host(cls, request, tmp_path) -> None:
    _check_class("cuda", cls, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASSES)
def test_hip_jx_class_matches_the_host(cls, request, tmp_path) -> None:
    _check_class("hip", cls, request, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_cuda_jx_molecule_matches_pyscf(name, request, tmp_path) -> None:
    _check_molecule("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_hip_jx_molecule_matches_pyscf(name, request, tmp_path) -> None:
    _check_molecule("hip", name, request, tmp_path)


@pytest.mark.parametrize("cls", CLASSES)
def test_every_jx_launch_names_the_kernels_arguments(cls) -> None:
    """The `jx` launch's runner spec, built host-side: `J1` the output."""
    n = _case("spdf")[0].n_ao
    (spec,) = chain_specs(hess_steps([_launch("spdf", *cls)]), "J1",
                          np.zeros(3 * n * n))
    assert [a["name"] for a in spec["args"] if "out" in a] == ["J1"]
