"""The dense J- and K-layout ERI matrices on the card: every class at L <= 4
against the host, H2 and H2O/STO-3G's `J` and `K` against PySCF for a
symmetric and a non-symmetric `D`, and a union of row ranges against the
whole matrix.

THE BUILD is `eri_dense_driver.build_on_card`, on the shared runner (#649):
one runner per class's unit (`kernel_dense`, one layout), compiled with
devtools/config.sh's flags and FAILING on a warning located in generated text
or a harness header (#92); every class launched once into one row block
carried from class to class from zero, each chain `RUNS` times and every run
bit-identical to the first (plain stores, one owner per element).

THE DEVICE-VERSUS-HOST BOUND is test_kernel_eri_dense's `bound`, stated
before any run: per element `(2 gamma_N + delta) scale`, test_eri_full_device's
analysis of `kernel_eri.contraction`, which this kernel runs unchanged (FMA
contraction only removes roundings; `delta = BOYS_REL`). Elements no launched
class owns must come back exactly 0.0.

PART 1 is test_kernel_eri_full's SYNTHETIC basis, every ordered class at
L <= 4, both layouts. PART 2 is H2 and H2O/STO-3G whole, both layouts: the
matrix against the host; `J = M_J vec(D)` and `K = M_K vec(D)` (the host's
product: a dense GEMV is the BLAS's) against PySCF's `vj` / `vk` at `J_TOL`
for the fixture's `D`, and for `D + A` (`A` antisymmetric) `J` against `vj`
and `(K(D) + K(D^T)) / 2` against `vk`, as interpreted; and H2O's three row
ranges, each launched alone on the card, stacked bit for bit into the whole
matrix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from eri_dense_driver import build_on_card, dense_steps, vec_product
from harness_driver import build_steps
from oracle import manifest
from test_kernel_device import BACKEND, _toolchain
from test_kernel_eri_dense import (
    J_OF,
    MOLECULES,
    PARTITION,
    _basis,
    _classes,
    _fixture,
    check,
    nonsymmetric,
)
from test_kernel_eri_full import J_TOL, L_MAX

from felizjueves.eri.full_eri.dense import LAYOUTS, dense_classes, row_range


def _run(backend: str, name: str, layout: str, request, tmp_path: Path,
         shells=None) -> np.ndarray:
    tc = _toolchain(backend, request)
    basis = _basis(name)
    classes = _classes(basis, shells)
    m0, n_row = row_range(basis, shells)
    M = build_on_card(classes, basis.n_ao, layout, m0, n_row, backend, tc,
                      tmp_path / f"{layout}_{shells}")
    keep = frozenset(c.am for c in classes)
    worst = check(M, name, layout, keep, shells, zero_elsewhere=True)
    print(f"\n{backend} {BACKEND[backend][1].name} {name} {layout} shells "
          f"{shells or 'all'}: {len(classes)} classes; worst / bound "
          f"{worst:.2e}; runs bit-identical (the runner's witness)")
    return M


# ---------------------------------------------------------------------------
# Part 1: every ordered class at L <= 4, against the host
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
def test_cuda_every_dense_class_through_l4_matches_the_host(
        layout, request, tmp_path):
    _run("cuda", "synthetic", layout, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
def test_hip_every_dense_class_through_l4_matches_the_host(
        layout, request, tmp_path):
    _run("hip", "synthetic", layout, request, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the molecules -- J and K against PySCF, and the row ranges
# ---------------------------------------------------------------------------


def _part2(backend: str, name: str, request, tmp_path: Path) -> None:
    basis = _basis(name)
    assert len(_classes(basis)) == len(dense_classes(basis))
    fx, fj = _fixture(name), _fixture(J_OF[name])
    M = {lay: _run(backend, name, lay, request, tmp_path) for lay in LAYOUTS}
    D = nonsymmetric(fx.density)
    vj, vk = fj.arrays[manifest.J_ARRAY], fx.arrays[manifest.K_ARRAY]
    K, KT = (vec_product(M["k"], x) for x in (D, D.T))
    rels = {
        "J vs vj": vec_product(M["j"], fx.density) - vj,
        "K vs vk": vec_product(M["k"], fx.density) - vk,
        "J(D+A) vs vj": vec_product(M["j"], D) - vj,
        "(K(D+A) + K((D+A)^T))/2 vs vk": (K + KT) / 2 - vk,
    }
    for what, diff in rels.items():
        ref = vj if what.startswith("J") else vk
        rel = float(np.max(np.abs(diff)) / np.max(np.abs(ref)))
        print(f"{backend} {name}: {what}: {rel:.2e} (tol {J_TOL:g})")
        assert rel < J_TOL, what


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_cuda_dense_j_and_k_match_pyscf(name, request, tmp_path):
    _part2("cuda", name, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("name", MOLECULES)
def test_hip_dense_j_and_k_match_pyscf(name, request, tmp_path):
    _part2("hip", name, request, tmp_path)


def _ranges(backend: str, layout: str, request, tmp_path: Path) -> None:
    name = "h2o_sto3g_k"
    whole = _run(backend, name, layout, request, tmp_path)
    blocks = [_run(backend, name, layout, request, tmp_path, cut)
              for cut in PARTITION]
    assert np.array_equal(np.vstack(blocks), whole)
    print(f"{backend} {name} {layout}: rows of {PARTITION} stacked == the "
          "whole matrix, bitwise")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
def test_cuda_a_union_of_row_ranges_is_the_whole_matrix(
        layout, request, tmp_path):
    _ranges("cuda", layout, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS)
def test_hip_a_union_of_row_ranges_is_the_whole_matrix(
        layout, request, tmp_path):
    _ranges("hip", layout, request, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`)
# ---------------------------------------------------------------------------


def _programs(name: str):
    basis = _basis(name)
    m0, n_row = row_range(basis)
    return [s for lay in LAYOUTS
            for s in dense_steps(_classes(basis), basis.n_ao, lay, m0, n_row)]


@pytest.mark.cuda
@pytest.mark.parametrize("name", ("synthetic", *MOLECULES))
def test_cuda_dense_programs_compile(name, nvcc, tmp_path):
    build_steps(nvcc, "cuda", _programs(name), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("name", ("synthetic", *MOLECULES))
def test_hip_dense_programs_compile(name, hipcc, tmp_path):
    build_steps(hipcc, "hip", _programs(name), tmp_path)


def test_part1_reaches_every_ordering_at_l4_and_no_more():
    got = {c.am for c in _classes(_basis("synthetic"))}
    assert {(2, 0, 0, 0), (0, 0, 0, 2), (1, 1, 1, 1), (2, 1, 0, 1),
            (0, 2, 2, 0)} <= got
    assert all(sum(am) <= L_MAX for am in got)
