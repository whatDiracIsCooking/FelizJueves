"""The DF Coulomb metric `(P|Q)` on the card: every class `lP + lQ <= 4`
against the host, and the whole L <= 4-filtered H2O/cc-pVTZ-jkfit metric
against the host and libcint's `int2c2e_cart`, on the RTX 3080 and the
RX 9060 XT.

THE BUILD is `metric_driver.build_on_card`, on the shared runner (#654):
one runner per class's unit (`kernel_metric`), compiled with
devtools/config.sh's flags and FAILING on a warning located in generated
text or a harness header (#92); every class launched once into one
`n_aux x n_aux` matrix carried from class to class, from zero, each
launch `RUNS` times.

SCOPE, by the milestone's decision: correctness is tested at `lP + lQ <= 4`
only -- the nine classes `lP <= lQ`, (s|s) (s|p) (s|d) (s|f) (s|g) (p|p)
(p|d) (p|f) (d|d) (`test_kernel_metric.CLASSES`). Nothing above L = 4 is
built, compiled or launched here. PART 1 is `test_kernel_metric.SYNTHETIC`
(three centres, every l to g, swapped, diagonal and off-diagonal pairs,
one-centre odd-parity pairs whose blocks are exactly zero) at those nine
classes. PART 2 is the tracked set `h2o_ccpvtz_jkfit_metric` (O to g, H to
f; 45 shells, 166 functions), every pair of its nine L <= 4 classes, the
other elements left untouched (0.0).

THE DEVICE-VERSUS-HOST BOUND, per element, stated before any run: the
interpreted kernel's, from test_kernel_metric.py, unchanged,

    |M_dev - M_host| <= TOL_ULPS eps scale,   TOL_ULPS = 16, eps = 2^-52,

`scale` the absolute-value twin of the element's sum with every `[t+tau]`
replaced by its TOWER's absolute-value twin (`test_kernel_metric.pq_tower`,
whose docstring says why the per-target `|[t+tau]|` is not enough). The
device runs the same IR as the interpreter; what differs is the vendor's
FMA contraction (which only removes roundings) and the device `fj_boys`
(#301's one template) in place of `reference.boys`. The latter is held
within `BOYS_REL = 1e-13` relative by test_kernel_device.py, which this
bound does NOT add (Schwarz's device bound does): the card is held to the
interpreter's yardstick as it stands, so a Boys error that mattered would
fail here rather than be absorbed. An element whose scale is 0 must come
back exactly 0.0, and every element no launched pair owns exactly 0.0.

THE LIBCINT BOUND, Part 2: `|M_dev - int2c2e_cart| <= TOL max|M|`,
`TOL = 1e-12`, `max|M|` the whole fixture matrix's largest element
(~96.5) -- test_ao_metric.py's and test_kernel_metric.py's, unchanged.

THE RUN-TO-RUN WITNESS: plain stores, one owner per element -- the runner
fails unless every run of a class is bit-identical to its first, and the
matrix is asserted exactly symmetric.

NOT A MEASUREMENT: nothing is written to `data/measured/`.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build every class's
runner -- the same nine for both bases, since a runner depends on its class
alone. The device tests are `gpu` plus `cuda` or `hip`, behind the card
lock: without the card they skip, and kernel-tier.sh exits 2.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from metric_driver import build_on_card, metric_runners
from plan_driver import compile_checked
from test_kernel_device import BACKEND, _toolchain
from test_kernel_df_lowstack import METRIC_LADDER, auxi_metric
from test_kernel_metric import (
    CLASSES,
    EPS,
    SYNTHETIC,
    TOL,
    TOL_ULPS,
    _aux,
    _host,
    _ms,
)
from tiers import FAST_L_MAX

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.kernel_metric import metric_lowstack
from felizjueves.eri.density_fit.metric import metric_classes

L_CAP = FAST_L_MAX  # 4, the milestone's cap
BASES = ("synthetic", "fixture")


@cache
def _basis(name: str) -> Basis:
    return SYNTHETIC if name == "synthetic" else _aux()


def _classes(basis: Basis):
    return [c for c in metric_classes(basis) if c.lP + c.lQ <= L_CAP]


@cache
def _host_of(name: str):
    """`(want, scale, owned)` of the L <= 4 classes, `test_kernel_metric`'s."""
    return _host(_basis(name), set(CLASSES))


def _run(backend: str, name: str, request, tmp_path: Path) -> np.ndarray:
    tc = _toolchain(backend, request)
    basis = _basis(name)
    classes = _classes(basis)
    assert sorted((c.lP, c.lQ) for c in classes) == CLASSES
    M = build_on_card(classes, basis.n_ao, backend, tc, tmp_path)
    want, scale, owned = _host_of(name)
    err = np.abs(M - want)
    pos = owned & (scale > 0)
    worst = float(np.max(err[pos] / (EPS * scale[pos])))
    print(f"\n{backend} {BACKEND[backend][1].name} {name}: "
          f"{len(classes)} classes, {sum(c.n_pair for c in classes)} pairs; "
          f"max |M - host| = {err[owned].max():.2e}, worst {worst:.2f} eps "
          f"scale (tol {TOL_ULPS}); runs bit-identical (the runner's witness)")
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), worst
    assert np.all(M[~owned] == 0.0)
    assert np.array_equal(M, M.T)
    return M


# ---------------------------------------------------------------------------
# Part 1: every class at L <= 4, against the host
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_every_metric_class_through_l4_matches_the_host(
        request, tmp_path):
    _run("cuda", "synthetic", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_every_metric_class_through_l4_matches_the_host(
        request, tmp_path):
    _run("hip", "synthetic", request, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the L <= 4-filtered cc-pVTZ-jkfit metric, against host and libcint
# ---------------------------------------------------------------------------


def _part2(backend: str, request, tmp_path: Path) -> None:
    M = _run(backend, "fixture", request, tmp_path)
    _, _, owned = _host_of("fixture")
    ref = _ms().metric
    sm = float(np.max(np.abs(ref)))
    worst = float(np.max(np.abs(M - ref)[owned])) / sm
    print(f"{backend} fixture: max |M - int2c2e_cart| = {worst:.2e} of "
          f"max|M| = {sm:.4g} (tol {TOL:g}); {int(owned.sum())} of "
          f"{owned.size} elements owned")
    assert worst <= TOL, worst


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_metric_through_l4_matches_libcint(request, tmp_path):
    _part2("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_metric_through_l4_matches_libcint(request, tmp_path):
    _part2("hip", request, tmp_path)


# ---------------------------------------------------------------------------
# Part 3 (#606): the aux ladder through i, every class in its default form
# ---------------------------------------------------------------------------

AUXI_COMPILE_S = 300.0  # the user's per-compile budget (#606)


@cache
def _auxi_host():
    _, aux = auxi_metric()
    return _host(aux, set(METRIC_LADDER))


def _part3(backend: str, request, tmp_path: Path) -> None:
    """All 28 classes `lP <= lQ <= 6` on the aux-i set, each in the form
    the gate picks (8 low-stack), one program per `lQ`, against the host
    (16 eps of the tower twin) and libcint (TOL of max)."""
    tc = _toolchain(backend, request)
    ms, aux = auxi_metric()
    classes = metric_classes(aux)
    assert sorted((c.lP, c.lQ) for c in classes) == METRIC_LADDER
    M = np.zeros((aux.n_ao, aux.n_ao))
    for lQ in range(7):
        part = [c for c in classes if c.lQ == lQ]
        M += build_on_card(part, aux.n_ao, backend, tc, tmp_path / f"lQ{lQ}",
                           timeout=AUXI_COMPILE_S)  # disjoint: others 0.0
    want, scale, owned = _auxi_host()
    assert owned.all()
    err = np.abs(M - want)
    pos = scale > 0
    worst = float(np.max(err[pos] / (EPS * scale[pos])))
    lib = float(np.max(np.abs(M - ms.metric))) / float(np.max(np.abs(
        ms.metric)))
    print(f"\n{backend} {BACKEND[backend][1].name} aux-i: {len(classes)} "
          f"classes, {sum(c.n_pair for c in classes)} pairs, "
          f"{sum(metric_lowstack((c.lP, c.lQ)) for c in classes)} "
          f"low-stack; worst {worst:.2f} eps scale against the host (tol "
          f"{TOL_ULPS}); {lib:.2e} of max against int2c2e_cart (tol "
          f"{TOL:g}); runs bit-identical")
    assert np.all(err <= TOL_ULPS * EPS * scale), worst
    assert np.array_equal(M, M.T)
    assert lib <= TOL, lib


@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_metric_through_i_i_matches_libcint(request, tmp_path):
    _part3("cuda", request, tmp_path)


@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.hip
@pytest.mark.gpu
def test_hip_metric_through_i_i_matches_libcint(request, tmp_path):
    _part3("hip", request, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`) and the fast-tier pieces
# ---------------------------------------------------------------------------


def _compiles(tc, backend: str, tmp_path: Path) -> None:
    for kern, prog in metric_runners(_classes(_basis("synthetic")), backend,
                                     tmp_path):
        compile_checked(tc, prog, tmp_path / f"runner_{kern.name}")


@pytest.mark.cuda
def test_cuda_metric_runners_compile(nvcc, tmp_path):
    _compiles(nvcc, "cuda", tmp_path)


@pytest.mark.hip
def test_hip_metric_runners_compile(hipcc, tmp_path):
    _compiles(hipcc, "hip", tmp_path)


@pytest.mark.parametrize("name", BASES)
def test_the_device_metric_reaches_the_nine_classes_and_no_more(name):
    """Both bases launch exactly the nine `lP + lQ <= 4` classes -- the
    same nine runners; the fixture also has classes above the cap, which
    are left out."""
    basis = _basis(name)
    got = sorted((c.lP, c.lQ) for c in _classes(basis))
    assert got == CLASSES and len(got) == 9
    assert any(c.lP + c.lQ > L_CAP for c in metric_classes(basis))
