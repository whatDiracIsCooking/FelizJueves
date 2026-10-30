"""The HGP DF kernels on the card: `(P|Q)` and `(mn|P)` for s and p shells
against the host and libcint, on the RTX 3080 and the RX 9060 XT.

THE BUILD is the MMD tests' harnesses with `engine=Engine.HGP`
(`metric_driver.build_on_card`, `eri3_driver.build_on_card`): the same
staging and launches, every class's `kernel_hgp` unit launched once into
one zeroed output, `RUNS` times, on the shared runner -- `(P|Q)` one
runner per unit (#654), `(mn|P)` one runner over every unit (#657) -- compiled
with devtools/config.sh's flags and FAILING on a warning located in
generated text (#92). The signatures are the MMD kernels' (the harnesses
check them), so nothing else changes.

SCOPE: s and p shells only (`kernel_hgp.HGP_DF_L_MAX = 1`). The synthetic
bases are `test_kernel_hgp_df`'s (the MMD tests' cut to s and p); the
fixtures are the tracked sets cut the same way: cc-pVTZ-jkfit's s and p aux
shells for `(P|Q)` (#411), and H2O/STO-3G against them for `(mn|P)` (#442).

THE BOUNDS, per element, stated before any run: the interpreted HGP
kernel's, from test_kernel_hgp_df.py, unchanged -- `|X_dev - X_host| <=
TOL_ULPS eps scale`, `TOL_ULPS = 16`, `scale` the MMD tests' tower-twin
absolute-value sum; and against libcint `TOL = 1e-12` of the fixture's
`max |.|`. As in test_metric_device.py, the device `fj_boys`' own error is
NOT added: a Boys error that mattered would fail here. An element no
launched pair or cell owns must come back exactly 0.0.

THE RUN-TO-RUN WITNESS: plain stores, one owner per element -- the runs are
asserted bit-identical (by the runner, which fails otherwise),
and the output exactly symmetric.

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
import test_kernel_metric as tm
from eri3_driver import build_on_card as eri3_on_card
from eri3_driver import eri3_program
from metric_driver import build_on_card as metric_on_card
from metric_driver import metric_runners
from plan_driver import compile_checked
from test_kernel_device import BACKEND, _toolchain
from test_kernel_hgp_df import AUX, BASIS, BASIS_AUX, EPS, ERI3, METRIC, TOL, TOL_ULPS

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.eri.density_fit.kernel_hgp import HGP_DF_L_MAX
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.spec import Engine

HGP = Engine.HGP
NAMES = ("synthetic", "fixture")


def _sp(basis: Basis) -> tuple[Basis, list[int]]:
    """`basis` cut to its s and p shells, and the cut's AO indices."""
    keep = [k for k, s in enumerate(basis.shells) if s.l <= HGP_DF_L_MAX]
    idx = [basis.offsets[k] + a for k in keep
           for a in range(basis.shells[k].n_cart)]
    return Basis(tuple(basis.shells[k] for k in keep)), idx


@cache
def _metric_basis(name: str) -> Basis:
    return AUX if name == "synthetic" else _sp(tm._aux())[0]


@cache
def _eri3_bases(name: str) -> tuple[Basis, Basis]:
    if name == "synthetic":
        return BASIS, BASIS_AUX
    return te._basis(), _sp(te._aux())[0]


def _check(X, want, scale, owned, Xs, label):
    err = np.abs(X - want)
    pos = owned & (scale > 0)
    worst = float(np.max(err[pos] / (EPS * scale[pos])))
    same = all(np.array_equal(X, x) for x in Xs[1:])
    print(f"\n{label}: max |X - host| = {err[owned].max():.2e}, worst "
          f"{worst:.2f} eps scale (tol {TOL_ULPS}); runs bit-identical {same}")
    assert same
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), worst
    assert np.all(X[~owned] == 0.0)


def _metric(backend, name, request, tmp_path):
    tc = _toolchain(backend, request)
    aux = _metric_basis(name)
    classes = list(metric_classes(aux))
    assert sorted((c.lP, c.lQ) for c in classes) == METRIC
    M = metric_on_card(classes, aux.n_ao, backend, tc, tmp_path, engine=HGP)
    want, scale, owned = tm._host(aux, set(METRIC))
    label = f"{backend} {BACKEND[backend][1].name} hmetric {name}"
    _check(M, want, scale, owned, [M], label)  # the runner witnessed the runs
    assert np.array_equal(M, M.T)
    if name == "fixture":
        rows = _sp(tm._aux())[1]
        ref = tm._ms().metric[np.ix_(rows, rows)]
        worst = float(np.max(np.abs(M - ref))) / float(
            np.max(np.abs(tm._ms().metric)))
        print(f"{label}: max |M - int2c2e_cart| = {worst:.2e} of max|M|")
        assert worst <= TOL, worst


def _eri3(backend, name, request, tmp_path):
    tc = _toolchain(backend, request)
    basis, aux = _eri3_bases(name)
    classes = list(eri3_classes(basis, aux))
    assert sorted((c.la, c.lb, c.lP) for c in classes) == ERI3
    T = eri3_on_card(classes, basis.n_ao, aux.n_ao, backend, tc, tmp_path,
                     engine=HGP)
    want, scale, owned = te._host(basis, aux, set(ERI3))
    label = f"{backend} {BACKEND[backend][1].name} heri3 {name}"
    _check(T, want, scale, owned, [T], label)  # the runner witnessed the runs
    assert np.array_equal(T, T.transpose(1, 0, 2))
    if name == "fixture":
        cols = _sp(te._aux())[1]
        ref = te._ts().tensor[:, :, cols]
        worst = float(np.max(np.abs(T - ref))) / float(
            np.max(np.abs(te._ts().tensor)))
        print(f"{label}: max |T - int3c2e_cart| = {worst:.2e} of max|T|")
        assert worst <= TOL, worst


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", NAMES)
def test_cuda_hgp_metric_matches_the_host_and_libcint(
        name, request, tmp_path):
    _metric("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", NAMES)
def test_hip_hgp_metric_matches_the_host_and_libcint(
        name, request, tmp_path):
    _metric("hip", name, request, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", NAMES)
def test_cuda_hgp_eri3_matches_the_host_and_libcint(
        name, request, tmp_path):
    _eri3("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", NAMES)
def test_hip_hgp_eri3_matches_the_host_and_libcint(
        name, request, tmp_path):
    _eri3("hip", name, request, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`) and the fast-tier pieces
# ---------------------------------------------------------------------------


def _programs(name: str, backend: str, tmp_path: Path):
    """Every metric class's runner, then the `(mn|P)` runner."""
    aux = _metric_basis(name)
    for _, prog in metric_runners(list(metric_classes(aux)), backend,
                                  tmp_path / "m", engine=HGP):
        yield prog
    basis, aux3 = _eri3_bases(name)
    yield eri3_program(list(eri3_classes(basis, aux3)), basis.n_ao,
                       aux3.n_ao, backend, tmp_path / "e", engine=HGP)


@pytest.mark.cuda
@pytest.mark.parametrize("name", NAMES)
def test_cuda_hgp_df_programs_compile(name, nvcc, tmp_path):
    for k, prog in enumerate(_programs(name, "cuda", tmp_path)):
        compile_checked(nvcc, prog, tmp_path / f"bin{k}")


@pytest.mark.hip
@pytest.mark.parametrize("name", NAMES)
def test_hip_hgp_df_programs_compile(name, hipcc, tmp_path):
    for k, prog in enumerate(_programs(name, "hip", tmp_path)):
        compile_checked(hipcc, prog, tmp_path / f"bin{k}")


@pytest.mark.parametrize("name", NAMES)
def test_the_programs_launch_the_hgp_units_of_every_sp_class(name, tmp_path):
    progs = list(_programs(name, "cuda", tmp_path))
    units = {p.name for prog in progs for p in prog.units
             if p.suffix == ".cu"}
    assert units == ({f"hmetric_{a}{b}.cu" for a, b in METRIC}
                     | {f"heri3_{''.join(map(str, c))}.cu" for c in ERI3})
    *metric, eri3 = progs
    assert len(metric) == len(METRIC)
    for prog in metric:  # each runner launches its HGP unit, no MMD one
        assert "FJ_KERNEL(hmetric_" in prog.source
        assert "FJ_KERNEL(metric_" not in prog.source
    assert "::heri3_" in eri3.source  # the runner launches the HGP units
    assert "::eri3_" not in eri3.source  # and no MMD one
