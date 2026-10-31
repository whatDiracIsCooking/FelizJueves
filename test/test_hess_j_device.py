"""The Coulomb-energy Hessian kernels ON A CARD: `hdq_` (the ket derivative
densities, `Target.DQ` at d1) and `hjab_` (the skeleton `d2 E_J`,
`Target.JAB` at d2), against `hessian.host` (generator
`test/hess_driver.py`, its one OUT `D1` for `hdq_`, `H` for `hjab_`).

The inputs are test_kernel_hessian.py's, so the card is held to the same
host the interpreted kernels are held to:

- `hdq`, PER KET CLASS, every ordered class through (ff) -- through (dd)
  on its three-atom s, p, d basis, with an f shell on its two-atom s, p,
  d, f basis (`SPDF`, #690) --: one launch over every record of the
  class, `D1[6, n_rec, n_q1]`, against `host.ket_derivative_records`, in
  its default form (rolled from `lc + ld = 5`), and every class but (ff)
  in its other form too (the unrolled `hdq_33` is ~114 s of nvcc).
- `hjab`, PER BRA CLASS, the same classes and bases: one launch, `Jp2` and
  `Jp1` the host's (`gradient.host.coulomb_jp`), against
  `coulomb_hessian(..., pairs=)` of the class's pairs, in its default form
  (rolled from `la + lb = 3`), and through `la + lb = 3` in its other form
  (the unrolled (dd) is ~186 s of nvcc). This is the issue's "every class
  launches on the RTX 3080".
- THE WHOLE `H` of H2O/STO-3G: one `hjab` launch per pair class into one
  `H`, the host's `Jp`s, against the host's whole skeleton Hessian.

The on-card chain, with the card's own `Jp`s, is
test_hjab_chain_ondevice.py.

THE BOUNDS, stated before any run. `hjab`: `|H_dev - H_host| <= DEV_TOL
max(scale)`, `DEV_TOL = 1e-14` (~45 eps), `scale` the host's absolute twin,
test_hess_device.py's bound for the same reason (the interpreted kernel is
within 16 eps of it; the card's atomics reorder the adds and its `exp`
differs by ulps). `hdq` has no atomics and no absolute twin:
`|D1_dev - D1_host| <= DQ_TOL max|D1|`, `DQ_TOL = 1e-13`, the gradient
tests' bound (interpreted it is bitwise; the card's `exp` differs by
ulps). The runs are compared with each other too.

RUN ON BOTH CARDS (2026-10-27, through (dd), unrolled), every case
passed: `hdq` worst 4.98e-16 of `max|D1|` (`hdq_02`, the same on both),
`hjab` per class 7.16e-17 of the scale, the whole H2O/STO-3G `d2 E_J`
9.28e-17 on both; the per-class launches are bit-identical run to run.
Through (ff), both forms: see the tree-state file (#690).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from harness_driver import chain_specs
from hess_driver import HessLaunch, compile_hess, hess_steps
from test_kernel_device import _toolchain
from test_kernel_hessian import (
    CLASSES,
    _all_pairs,
    _arrays,
    _basis_of,
    _case,
    _density,
    _host,
    _host_jps,
    _ket_classes,
    _scale,
)
from tiers import by_l

from felizjueves.gradient import host as gh
from felizjueves.hessian import host
from felizjueves.hessian.kernel_hdq import build_hdq_kernel, hdq_lowstack, n_q1
from felizjueves.hessian.kernel_hjab import build_hjab_kernel, hjab_lowstack

DEV_TOL = 1e-14  # of max(scale), ~45 eps (module docstring)
DQ_TOL = 1e-13  # of max|D1| (module docstring)
COMPILE_BUDGET_S = 120.0  # the user's per-.cu budget, each unit alone
# split by L as test_kernel_hessian's interpreted twins are: `hdq` runs the
# ket class raised by one, `hjab` the [p|q] with the bra raised by two, so
# (dd)'s hjab is an L = 6 kernel (218 s of the CUDA PR gate on its own)
HDQ_CLASSES = [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}") for c in CLASSES]
HJAB_CLASSES = [by_l(c, sum(c) + 2, id=f"{c[0]}{c[1]}") for c in CLASSES]
# #690: each class's other form, where it compiles within the budget
HDQ_OTHER = [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}") for c in CLASSES
             if c != (3, 3)]
HJAB_OTHER = [by_l(c, sum(c) + 2, id=f"{c[0]}{c[1]}") for c in CLASSES
              if sum(c) <= 3]


def _hdq_launch(lc: int, ld: int, lowstack: bool | None = None
                ) -> tuple[HessLaunch, int]:
    basis, nuclei = _case(_basis_of((lc, ld)))
    rec = gh.records(basis, nuclei, lc, ld)
    n = len(rec["s_ab"])
    arrs = _arrays(rec, atoms=False) | {"M": _density(basis.n_ao)}
    return HessLaunch(build_hdq_kernel((lc, ld), lowstack=lowstack), n, arrs,
                      {"n_rec": n, "n_ao": basis.n_ao}, out="D1"), n


def _hjab_launch(name: str, la: int, lb: int,
                 lowstack: bool | None = None) -> HessLaunch:
    basis, nuclei = _case(name)
    rec = gh.records(basis, nuclei, la, lb)
    n = len(rec["s_ab"])
    jp2, jp1 = _host_jps(name, la, lb, rec)
    arrs = _arrays(rec) | {"Jp2": jp2, "Jp1": jp1,
                           "M": _density(basis.n_ao),
                           "row": np.arange(n, dtype=np.int32)}
    return HessLaunch(build_hjab_kernel((la, lb), lowstack=lowstack), n,
                      arrs, {
        "n_rec": n, "n_ao": basis.n_ao, "n_atom": len(nuclei),
        "n_rows": n})


def _report(got, want, scale, label: str, tol: float) -> None:
    errs = [float(np.abs(g - want).max() / scale) for g in got]
    assert max(errs) <= tol, f"{label}: |dev - host| / scale = {errs} > {tol:g}"
    spread = float(np.abs(got[0] - got[1]).max() / scale)
    print(f"\n{label}: |dev - host| / scale per run {errs}; run-to-run "
          f"{spread:.2e}; bit-identical {bool(np.array_equal(got[0], got[1]))}")


def _check_hdq(backend, cls, request, tmp_path: Path,
               lowstack: bool | None = None):
    tc = _toolchain(backend, request)
    basis, nuclei = _case(_basis_of(cls))
    x, n = _hdq_launch(*cls, lowstack=lowstack)
    size = 6 * n * n_q1(*cls)
    got = compile_hess([x], len(nuclei), backend, tc, tmp_path,
                       n_out=size, timeout=COMPILE_BUDGET_S).run()
    want = np.array(host.ket_derivative_records(
        basis, _density(basis.n_ao), *cls)).ravel()
    assert want.size == size
    form = "rolled" if (hdq_lowstack(cls) if lowstack is None
                        else lowstack) else "unrolled"
    _report(got, want, np.abs(want).max(),
            f"{backend} hdq_{cls[0]}{cls[1]} ({form})", DQ_TOL)


def _check_hjab(backend, name, classes, request, tmp_path: Path,
                label: str, lowstack: bool | None = None):
    tc = _toolchain(backend, request)
    basis, nuclei = _case(name)
    nat = len(nuclei)
    xs = [_hjab_launch(name, la, lb, lowstack) for la, lb in classes]
    got = compile_hess(xs, nat, backend, tc, tmp_path,
                       timeout=COMPILE_BUDGET_S).run()
    pairs = [p for c in classes for p in gh.class_pairs(basis, *c)]
    if set(pairs) == set(_all_pairs(name)):
        pairs = None
    want = _host(name, pairs)
    _report(got, want, _scale(name, pairs or _all_pairs(name)),
            f"{backend} {label}", DEV_TOL)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", HDQ_CLASSES)
def test_cuda_hdq_matches_the_host(cls, request, tmp_path) -> None:
    _check_hdq("cuda", cls, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", HDQ_CLASSES)
def test_hip_hdq_matches_the_host(cls, request, tmp_path) -> None:
    _check_hdq("hip", cls, request, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", HDQ_OTHER)
def test_cuda_hdq_other_form_matches_the_host(cls, request,
                                              tmp_path) -> None:
    _check_hdq("cuda", cls, request, tmp_path, not hdq_lowstack(cls))


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", HDQ_OTHER)
def test_hip_hdq_other_form_matches_the_host(cls, request,
                                             tmp_path) -> None:
    _check_hdq("hip", cls, request, tmp_path, not hdq_lowstack(cls))


def _label(cls, lowstack) -> str:
    low = hjab_lowstack(cls) if lowstack is None else lowstack
    return f"hjab_{cls[0]}{cls[1]} ({'rolled' if low else 'unrolled'})"


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", HJAB_CLASSES)
def test_cuda_hjab_class_matches_the_host(cls, request, tmp_path) -> None:
    _check_hjab("cuda", _basis_of(cls), [cls], request, tmp_path,
                _label(cls, None))


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", HJAB_CLASSES)
def test_hip_hjab_class_matches_the_host(cls, request, tmp_path) -> None:
    _check_hjab("hip", _basis_of(cls), [cls], request, tmp_path,
                _label(cls, None))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", HJAB_OTHER)
def test_cuda_hjab_other_form_matches_the_host(cls, request,
                                               tmp_path) -> None:
    low = not hjab_lowstack(cls)
    _check_hjab("cuda", _basis_of(cls), [cls], request, tmp_path,
                _label(cls, low), low)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", HJAB_OTHER)
def test_hip_hjab_other_form_matches_the_host(cls, request,
                                              tmp_path) -> None:
    low = not hjab_lowstack(cls)
    _check_hjab("hip", _basis_of(cls), [cls], request, tmp_path,
                _label(cls, low), low)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
def test_cuda_h2o_coulomb_hessian_matches_the_host(
        request, tmp_path) -> None:
    _check_hjab("cuda", "h2o_sto3g_j", _ket_classes("h2o_sto3g_j"), request,
                tmp_path, "H2O/STO-3G d2 E_J")


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
def test_hip_h2o_coulomb_hessian_matches_the_host(
        request, tmp_path) -> None:
    _check_hjab("hip", "h2o_sto3g_j", _ket_classes("h2o_sto3g_j"), request,
                tmp_path, "H2O/STO-3G d2 E_J")


@pytest.mark.parametrize("cls", HDQ_CLASSES)
def test_every_hdq_launch_names_the_kernels_arguments(cls) -> None:
    """The `hdq` launch's runner spec, built host-side: `D1` the output."""
    x, n = _hdq_launch(*cls)
    (spec,) = chain_specs(hess_steps([x]), "D1",
                          np.zeros(6 * n * n_q1(*cls)))
    assert [a["name"] for a in spec["args"] if "out" in a] == ["D1"]


@pytest.mark.parametrize("cls", HJAB_CLASSES)
def test_every_hjab_launch_names_the_kernels_arguments(cls) -> None:
    """The `hjab` launch's runner spec, built host-side: `H` the output."""
    name = _basis_of(cls)
    nat = len(_case(name)[1])
    (spec,) = chain_specs(hess_steps([_hjab_launch(name, *cls)]), "H",
                          np.zeros((nat, 3, nat, 3)))
    assert [a["name"] for a in spec["args"] if "out" in a] == ["H"]
