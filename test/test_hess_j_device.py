"""The Coulomb-energy Hessian kernels ON A CARD: `hdq_` (the ket derivative
densities, `Target.DQ` at d1) and `hjab_` (the skeleton `d2 E_J`,
`Target.JAB` at d2), against `hessian.host` (generator
`test/hess_driver.py`, its one OUT `D1` for `hdq_`, `H` for `hjab_`).

The inputs are test_kernel_hessian.py's, so the card is held to the same
host the interpreted kernels are held to:

- `hdq`, PER KET CLASS, every ordered class through (dd) on its three-atom
  s, p, d basis: one launch over every record of the class, `D1[6, n_rec,
  n_q1]`, against `host.ket_derivative_records`.
- `hjab`, PER BRA CLASS, the same classes and basis: one launch, `Jp2` and
  `Jp1` the host's (`gradient.host.coulomb_jp`), against
  `coulomb_hessian(..., pairs=)` of the class's pairs.
- THE WHOLE `H` of H2O/STO-3G: one `hjab` launch per pair class into one
  `H`, the host's `Jp`s, against the host's whole skeleton Hessian.

Every kernel launched is a pair class `l_a, l_b <= 2` (L <= 4); its `Jp`s
are the host's at L + 1 and L + 2.

THE BOUNDS, stated before any run. `hjab`: `|H_dev - H_host| <= DEV_TOL
max(scale)`, `DEV_TOL = 1e-14` (~45 eps), `scale` the host's absolute twin,
test_hess_device.py's bound for the same reason (the interpreted kernel is
within 16 eps of it; the card's atomics reorder the adds and its `exp`
differs by ulps). `hdq` has no atomics and no absolute twin:
`|D1_dev - D1_host| <= DQ_TOL max|D1|`, `DQ_TOL = 1e-13`, the gradient
tests' bound (interpreted it is bitwise; the card's `exp` differs by
ulps). The runs are compared with each other too.

RUN ON BOTH CARDS (2026-10-27), every case passed: `hdq` worst 4.98e-16
of `max|D1|` (`hdq_02`, the same on both), `hjab` per class 7.16e-17 of
the scale, the whole H2O/STO-3G `d2 E_J` 9.28e-17 on both; the per-class
launches are bit-identical run to run.

NOT COVERED: `Jp2` / `Jp1` are the host's, so the on-card chain (`hdq` ->
gather -> the raised `[p|q]` -> `hjab`) is interpreted only
(test_kernel_hessian.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hess_driver import HessLaunch, compile_hess
from test_kernel_device import _toolchain
from test_kernel_hessian import (
    CLASSES,
    SPD,
    _all_pairs,
    _arrays,
    _case,
    _density,
    _host,
    _host_jps,
    _ket_classes,
    _scale,
)

from felizjueves.gradient import host as gh
from felizjueves.hessian import host
from felizjueves.hessian.kernel_hdq import build_hdq_kernel, n_q1
from felizjueves.hessian.kernel_hjab import build_hjab_kernel

DEV_TOL = 1e-14  # of max(scale), ~45 eps (module docstring)
DQ_TOL = 1e-13  # of max|D1| (module docstring)
IDS = [f"{a}{b}" for a, b in CLASSES]


def _hdq_launch(lc: int, ld: int) -> tuple[HessLaunch, int]:
    basis, _ = SPD
    rec = gh.records(basis, SPD[1], lc, ld)
    n = len(rec["s_ab"])
    arrs = _arrays(rec, atoms=False) | {"M": _density(basis.n_ao)}
    return HessLaunch(build_hdq_kernel((lc, ld)), n, arrs,
                      {"n_rec": n, "n_ao": basis.n_ao}, out="D1"), n


def _hjab_launch(name: str, la: int, lb: int) -> HessLaunch:
    basis, nuclei = _case(name)
    rec = gh.records(basis, nuclei, la, lb)
    n = len(rec["s_ab"])
    jp2, jp1 = _host_jps(name, la, lb, rec)
    arrs = _arrays(rec) | {"Jp2": jp2, "Jp1": jp1,
                           "M": _density(basis.n_ao),
                           "row": np.arange(n, dtype=np.int32)}
    return HessLaunch(build_hjab_kernel((la, lb)), n, arrs, {
        "n_rec": n, "n_ao": basis.n_ao, "n_atom": len(nuclei),
        "n_rows": n})


def _report(got, want, scale, label: str, tol: float) -> None:
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    assert max(errs) <= tol, f"{label}: |dev - host| / scale = {errs} > {tol:g}"
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{label}: |dev - host| / scale per run {errs}; run-to-run "
          f"{spread:.2e}; bit-identical {bool(np.array_equal(got[0], got[1]))}")


def _check_hdq(backend, cls, request, run_device_binary, tmp_path: Path):
    tc = _toolchain(backend, request)
    x, n = _hdq_launch(*cls)
    size = 6 * n * n_q1(*cls)
    got = compile_hess([x], len(SPD[1]), backend, tc, tmp_path,
                       n_out=size).run(run_device_binary)
    want = np.array(host.ket_derivative_records(
        SPD[0], _density(SPD[0].n_ao), *cls)).ravel()
    assert want.size == size
    _report(got, want, np.abs(want).max(), f"{backend} hdq_{cls[0]}{cls[1]}",
            DQ_TOL)


def _check_hjab(backend, name, classes, request, run_device_binary,
                tmp_path: Path, label: str):
    tc = _toolchain(backend, request)
    basis, nuclei = _case(name)
    nat = len(nuclei)
    xs = [_hjab_launch(name, la, lb) for la, lb in classes]
    got = compile_hess(xs, nat, backend, tc, tmp_path).run(run_device_binary)
    pairs = [p for c in classes for p in gh.class_pairs(basis, *c)]
    if set(pairs) == set(_all_pairs(name)):
        pairs = None
    want = _host(name, pairs)
    _report(got, want, _scale(name, pairs or _all_pairs(name)),
            f"{backend} {label}", DEV_TOL)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_cuda_hdq_matches_the_host(cls, request, run_device_binary,
                                   tmp_path) -> None:
    _check_hdq("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_hip_hdq_matches_the_host(cls, request, run_device_binary,
                                  tmp_path) -> None:
    _check_hdq("hip", cls, request, run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_cuda_hjab_class_matches_the_host(cls, request, run_device_binary,
                                          tmp_path) -> None:
    _check_hjab("cuda", "spd", [cls], request, run_device_binary, tmp_path,
                f"hjab_{cls[0]}{cls[1]}")


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_hip_hjab_class_matches_the_host(cls, request, run_device_binary,
                                         tmp_path) -> None:
    _check_hjab("hip", "spd", [cls], request, run_device_binary, tmp_path,
                f"hjab_{cls[0]}{cls[1]}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
def test_cuda_h2o_coulomb_hessian_matches_the_host(
        request, run_device_binary, tmp_path) -> None:
    _check_hjab("cuda", "h2o_sto3g_j", _ket_classes("h2o_sto3g_j"), request,
                run_device_binary, tmp_path, "H2O/STO-3G d2 E_J")


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
def test_hip_h2o_coulomb_hessian_matches_the_host(
        request, run_device_binary, tmp_path) -> None:
    _check_hjab("hip", "h2o_sto3g_j", _ket_classes("h2o_sto3g_j"), request,
                run_device_binary, tmp_path, "H2O/STO-3G d2 E_J")
