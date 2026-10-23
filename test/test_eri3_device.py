"""The three-center DF integral `(mn|P)` on the card: every class
`la + lb + lP <= 4` against the host, and the L <= 4-filtered
H2O/STO-3G x cc-pVTZ-jkfit tensor against the host and libcint's
`int3c2e_cart`, on the RTX 3080 and the RX 9060 XT.

THE BUILD is `eri3_driver.build_on_card`: one program per (AO basis, aux
basis), every class's kernel (`kernel_eri3`) launched once into one zeroed
`n_ao x n_ao x n_aux` tensor, `RUNS` times, compiled with
devtools/config.sh's flags and FAILING on a warning located in generated
text (#92).

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

THE RUN-TO-RUN WITNESS: plain stores, one owner per element -- the runs are
asserted bit-identical, and the tensor exactly symmetric in `(m, n)`.

NOT A MEASUREMENT: nothing is written to `data/measured/`.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`) build both programs.
The device tests are `gpu` plus `cuda` or `hip`, behind the card lock and
the VRAM probe: without the card they skip, and kernel-tier.sh exits 2.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from eri3_driver import build_on_card, eri3_program
from plan_driver import compile_checked
from test_kernel_device import BACKEND, _toolchain
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


def _run(backend: str, name: str, request, run_device_binary,
         tmp_path: Path) -> np.ndarray:
    tc = _toolchain(backend, request)
    basis, aux = _bases(name)
    classes = _classes(name)
    assert sorted((c.la, c.lb, c.lP) for c in classes) == _expected(name)
    Ts = build_on_card(classes, basis.n_ao, aux.n_ao, backend, tc, tmp_path,
                       run_device_binary)
    T = Ts[0]
    want, scale, owned = _host_of(name)
    err = np.abs(T - want)
    pos = owned & (scale > 0)
    worst = float(np.max(err[pos] / (EPS * scale[pos])))
    same = all(np.array_equal(T, t) for t in Ts[1:])
    print(f"\n{backend} {BACKEND[backend][1].name} {name}: "
          f"{len(classes)} classes, {sum(c.n_cell for c in classes)} cells; "
          f"max |T - host| = {err[owned].max():.2e}, worst {worst:.2f} eps "
          f"scale (tol {TOL_ULPS}); runs bit-identical {same}")
    assert same
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
        request, run_device_binary, tmp_path):
    _run("cuda", "synthetic", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_every_eri3_class_through_l4_matches_the_host(
        request, run_device_binary, tmp_path):
    _run("hip", "synthetic", request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the L <= 4-filtered fixture tensor, against host and libcint
# ---------------------------------------------------------------------------


def _part2(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    T = _run(backend, "fixture", request, run_device_binary, tmp_path)
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
def test_cuda_eri3_through_l4_matches_libcint(request, run_device_binary,
                                              tmp_path):
    _part2("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_eri3_through_l4_matches_libcint(request, run_device_binary,
                                             tmp_path):
    _part2("hip", request, run_device_binary, tmp_path)


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
