"""The three-center DF integral `(mn|P)` on the card: every class
`la + lb + lP <= 4` against the host, and the L <= 4-filtered
H2O/STO-3G x cc-pVTZ-jkfit tensor against the host and libcint's
`int3c2e_cart`, on the RTX 3080 and the RX 9060 XT.

THE BUILD is `eri3_driver.build_on_card`, on the shared runner (#657): one
runner per (AO basis, aux basis) over every class's unit (`kernel_eri3`),
compiled with devtools/config.sh's flags and FAILING on a warning located
in generated text or a harness header (#92), and one sequence launching
every class once into one zeroed `n_ao x n_ao x n_aux` tensor, `RUNS`
times.

SCOPE, by the milestone's decision: correctness is tested at
`la + lb + lP <= 4` only (`tiers.FAST_L_MAX`). Nothing above L = 4 is
built, compiled or launched here. PART 1 is `test_kernel_eri3.SYNTHETIC` x
`SYNTHETIC_AUX` (three centres, every l to g on both sides, swapped,
diagonal and off-diagonal pairs, one-centre triples) at every one of the 22
classes `la <= lb` (`test_kernel_eri3.CLASSES`). PART 2 is the tracked set
`h2o_sto3g_jkfit_3c` (H2O/STO-3G's 7 AOs, s and p; cc-pVTZ-jkfit's 166 aux
functions, O to g), every cell of its twelve L <= 4 classes; the elements of
the three classes above the cap, (sp|g), (pp|f) and (pp|g), are left
untouched (0.0).

THE DEVICE-VERSUS-HOST BOUND, per element, stated before any run: the
interpreted kernel's, from test_kernel_eri3.py, unchanged,

    |T_dev - T_host| <= TOL_ULPS eps scale,   TOL_ULPS = 16, eps = 2^-52,

`scale` the absolute-value twin of the element's sum with every `[t+tau]`
replaced by its TOWER's absolute-value twin (`test_kernel_metric.pq_tower`,
whose docstring says why the per-target `|[t+tau]|` is not enough). The
device runs the same IR as the interpreter; what differs is the vendor's
FMA contraction (which only removes roundings) and the device `fj_boys`
(#301's one template) in place of `reference.boys`. The latter is held
within `BOYS_REL = 1e-13` relative by test_kernel_device.py, which this
bound does NOT add, as test_metric_device.py does not: the card is held to
the interpreter's yardstick as it stands, so a Boys error that mattered
would fail here rather than be absorbed. An element whose scale is 0 must
come back exactly 0.0, and every element no launched cell owns exactly 0.0.

THE LIBCINT BOUND, Part 2: `|T_dev - int3c2e_cart| <= TOL max|(mn|P)|`
over the owned elements, `TOL = 1e-12`, `max|(mn|P)|` the whole fixture
tensor's largest element (~7.21) -- test_ao_eri3.py's and
test_kernel_eri3.py's, unchanged.

THE RUN-TO-RUN WITNESS: plain stores, one owner per element -- the runner
fails unless every run is bit-identical to its first, and the tensor is
asserted exactly symmetric in `(m, n)`.

NOT A MEASUREMENT: nothing is written to `data/measured/`.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build both runners.
The device tests are `gpu` plus `cuda` or `hip`, behind the card lock:
without the card they skip, and kernel-tier.sh exits 2.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from eri3_driver import build_on_card, eri3_program
from plan_driver import compile_checked
from test_kernel_device import BACKEND, _toolchain
from test_kernel_df_lowstack import ERI3_LADDER, auxi_3c
from test_kernel_eri3 import (
    CLASSES,
    EPS,
    SYNTHETIC,
    SYNTHETIC_AUX,
    TOL,
    TOL_ULPS,
    _aux,
    _basis,
    _host,
    _ts,
)
from tiers import FAST_L_MAX

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.eri.density_fit.kernel_eri3 import eri3_lowstack

L_CAP = FAST_L_MAX  # 4, the milestone's cap
BASES = ("synthetic", "fixture")
FIXTURE_CLASSES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
                   (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
                   (1, 1, 0), (1, 1, 1), (1, 1, 2)]


@cache
def _bases(name: str) -> tuple[Basis, Basis]:
    if name == "synthetic":
        return SYNTHETIC, SYNTHETIC_AUX
    return _basis(), _aux()


def _classes(name: str):
    return [c for c in eri3_classes(*_bases(name))
            if c.la + c.lb + c.lP <= L_CAP]


def _expected(name: str):
    return CLASSES if name == "synthetic" else FIXTURE_CLASSES


@cache
def _host_of(name: str):
    """`(want, scale, owned)` of the L <= 4 classes, `test_kernel_eri3`'s."""
    return _host(*_bases(name), set(_expected(name)))


def _run(backend: str, name: str, request, tmp_path: Path) -> np.ndarray:
    tc = _toolchain(backend, request)
    basis, aux = _bases(name)
    classes = _classes(name)
    assert sorted((c.la, c.lb, c.lP) for c in classes) == _expected(name)
    T = build_on_card(classes, basis.n_ao, aux.n_ao, backend, tc, tmp_path)
    want, scale, owned = _host_of(name)
    err = np.abs(T - want)
    pos = owned & (scale > 0)
    worst = float(np.max(err[pos] / (EPS * scale[pos])))
    print(f"\n{backend} {BACKEND[backend][1].name} {name}: "
          f"{len(classes)} classes, {sum(c.n_cell for c in classes)} cells; "
          f"max |T - host| = {err[owned].max():.2e}, worst {worst:.2f} eps "
          f"scale (tol {TOL_ULPS}); runs bit-identical (the runner's witness)")
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), worst
    assert np.all(T[~owned] == 0.0)
    assert np.array_equal(T, T.transpose(1, 0, 2))
    return T


# ---------------------------------------------------------------------------
# Part 1: every class at L <= 4 on synthetic records, against the host
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_every_eri3_class_through_l4_matches_the_host(
        request, tmp_path):
    _run("cuda", "synthetic", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_every_eri3_class_through_l4_matches_the_host(
        request, tmp_path):
    _run("hip", "synthetic", request, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the L <= 4-filtered fixture tensor, against host and libcint
# ---------------------------------------------------------------------------


def _part2(backend: str, request, tmp_path: Path) -> None:
    T = _run(backend, "fixture", request, tmp_path)
    _, _, owned = _host_of("fixture")
    ref = _ts().tensor
    st = float(np.max(np.abs(ref)))
    worst = float(np.max(np.abs(T - ref)[owned])) / st
    print(f"{backend} fixture: max |T - int3c2e_cart| = {worst:.2e} of "
          f"max|(mn|P)| = {st:.4g} (tol {TOL:g}); {int(owned.sum())} of "
          f"{owned.size} elements owned")
    assert worst <= TOL, worst


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_eri3_through_l4_matches_libcint(request,
                                              tmp_path):
    _part2("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_eri3_through_l4_matches_libcint(request,
                                             tmp_path):
    _part2("hip", request, tmp_path)


# ---------------------------------------------------------------------------
# Part 3 (#606): the aux ladder through i, every class in its default form
# ---------------------------------------------------------------------------

AUXI_COMPILE_S = 300.0  # the user's per-compile budget (#606)


@cache
def _auxi_host():
    _, basis, aux = auxi_3c()
    return _host(basis, aux, set(ERI3_LADDER))


def _part3(backend: str, request, tmp_path: Path) -> None:
    """All 70 classes of `la <= lb <= 3`, `lP <= 6` on the aux-i set, each
    in the form the gate picks (24 low-stack), one program per `lP` (so no
    program's compile passes the budget), against the host (the same
    16 eps of the tower twin) and libcint (TOL of max)."""
    tc = _toolchain(backend, request)
    ts, basis, aux = auxi_3c()
    classes = eri3_classes(basis, aux)
    assert sorted((c.la, c.lb, c.lP) for c in classes) == ERI3_LADDER
    T = np.zeros((basis.n_ao, basis.n_ao, aux.n_ao))
    for lP in range(7):
        part = [c for c in classes if c.lP == lP]
        T += build_on_card(part, basis.n_ao, aux.n_ao, backend, tc,
                           tmp_path / f"lP{lP}",
                           timeout=AUXI_COMPILE_S)  # disjoint: others 0.0
    want, scale, owned = _auxi_host()
    assert owned.all()
    err = np.abs(T - want)
    pos = scale > 0
    worst = float(np.max(err[pos] / (EPS * scale[pos])))
    lib = float(np.max(np.abs(T - ts.tensor))) / float(np.max(np.abs(
        ts.tensor)))
    print(f"\n{backend} {BACKEND[backend][1].name} aux-i: {len(classes)} "
          f"classes, {sum(c.n_cell for c in classes)} cells, "
          f"{sum(eri3_lowstack((c.la, c.lb, c.lP)) for c in classes)} "
          f"low-stack; worst "
          f"{worst:.2f} eps scale against the host (tol {TOL_ULPS}); "
          f"{lib:.2e} of max against int3c2e_cart (tol {TOL:g}); "
          "runs bit-identical")
    assert np.all(err <= TOL_ULPS * EPS * scale), worst
    assert np.array_equal(T, T.transpose(1, 0, 2))
    assert lib <= TOL, lib


@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_eri3_through_ff_i_matches_libcint(request,
                                                tmp_path):
    _part3("cuda", request, tmp_path)


@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.hip
@pytest.mark.gpu
def test_hip_eri3_through_ff_i_matches_libcint(request,
                                               tmp_path):
    _part3("hip", request, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`) and the fast-tier pieces
# ---------------------------------------------------------------------------


def _program(name: str, backend: str, tmp_path: Path):
    basis, aux = _bases(name)
    return eri3_program(_classes(name), basis.n_ao, aux.n_ao, backend,
                        tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize("name", BASES)
def test_cuda_eri3_programs_compile(name, nvcc, tmp_path):
    compile_checked(nvcc, _program(name, "cuda", tmp_path), tmp_path / "bin")


@pytest.mark.hip
@pytest.mark.parametrize("name", BASES)
def test_hip_eri3_programs_compile(name, hipcc, tmp_path):
    compile_checked(hipcc, _program(name, "hip", tmp_path), tmp_path / "bin")


@pytest.mark.parametrize("name", BASES)
def test_the_device_eri3_reaches_the_l4_classes_and_no_more(name):
    """Both programs launch exactly the `la + lb + lP <= 4` classes the
    bases reach (all 22 for the synthetic pair, twelve for the fixture);
    the fixture also has classes above the cap, which are left out."""
    got = sorted((c.la, c.lb, c.lP) for c in _classes(name))
    assert got == _expected(name)
    assert len(got) == (22 if name == "synthetic" else 12)
    assert any(c.la + c.lb + c.lP > L_CAP
               for c in eri3_classes(*_bases(name)))
