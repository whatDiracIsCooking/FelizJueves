"""The packed three-center kernel on the card (FreeBreakfast#1680): every
class's packed block equals the square kernel's block for the same cell,
BITWISE.

THE CONTRACT is `eri.density_fit.eri3`'s THE PACKED OUTPUT: each cell's
whole staged block at its table's destination offset `dst`, element `(a,
b, q)` at `dst + n_aux (a nB + b) + q`, the aux index fastest; a diagonal
shell pair (`diag = 1`) stores its full square block.

THE WITNESS is the square kernel (`build_eri3_kernel(cls)`, held to the
host and libcint by test_eri3_device.py) of the same class and form, built
and run on the same card: `eri3_driver.gather_packed` reads each cell's
block `T[off_a + a, off_b + b, off_p + q]` out of its tensor, and the
packed output must equal it with `np.array_equal` -- no tolerance. Both
kernels do the same arithmetic in the same order and every store is plain,
so there is no rounding to bound. The packed output starts NaN, so an
element no cell stores fails the comparison too, and each runner repeats
its sequence `RUNS` times bit-identically (the runner's witness).

PART 1 (fast, a card): every class `la + lb + lP <= 4` on
`test_kernel_eri3`'s synthetic bases (swapped, diagonal and off-diagonal
pairs), in BOTH forms, forced. PART 2 (`slow`, `high_l`): the aux-i set
(`test_kernel_df_lowstack.auxi_3c`: He s/p/d, Ne f; aux l = 0..6), all 70
classes `la <= lb <= 3`, `lP <= 6` -- the ladder FreeBreakfast emits -- in
the default form of each, and ALSO in the low-stack form wherever the
default is unrolled (the form FreeBreakfast's tools/emit/lowstack.py forces
on some of them), one program per (form, kernel, `lP`), each compile under
`COMPILE_S`, the programs compiled concurrently.

NOT A MEASUREMENT: nothing is written to `data/measured/`.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from eri3_driver import (
    build_packed_on_card,
    eri3_kernel,
    eri3_program,
    eri3_steps,
    gather_packed,
    packed_kernel,
    packed_layout,
    packed_program,
    packed_steps,
)
from harness_driver import build_runner, run_sequence
from plan_driver import RUNS, compile_checked
from test_kernel_device import BACKEND, _toolchain
from test_kernel_df_lowstack import ERI3_LADDER, auxi_3c
from test_kernel_eri3 import CLASSES, SYNTHETIC, SYNTHETIC_AUX

from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.eri.density_fit.kernel_eri3 import OUT, eri3_lowstack

FORMS = {"unrolled": False, "lowstack": True}
COMPILE_S = 300.0  # the user's per-compile budget (#606, FreeBreakfast#1680)


def _synthetic():
    classes = [c for c in eri3_classes(SYNTHETIC, SYNTHETIC_AUX)
               if c.la + c.lb + c.lP <= 4]
    assert sorted((c.la, c.lb, c.lP) for c in classes) == CLASSES
    return classes


def _square(classes, n_ao, n_aux, binary, workdir, lowstack):
    kernels = [eri3_kernel(c, lowstack=lowstack) for c in classes]
    buffers, steps = eri3_steps(classes, n_ao, n_aux, kernels)
    (T,) = run_sequence(binary, workdir, buffers, steps, outs=[OUT],
                        runs=RUNS).load(OUT)
    return T


def _packed(classes, cells, n_col, n_aux, binary, workdir, lowstack):
    kernels = [packed_kernel(c, lowstack) for c in classes]
    buffers, steps = packed_steps(classes, cells, n_col, n_aux, kernels)
    (B,) = run_sequence(binary, workdir, buffers, steps, outs=[OUT],
                        runs=RUNS).load(OUT)
    return B.reshape(n_col, n_aux)


def _compare(got, want, label):
    stored = ~np.isnan(want)
    print(f"\n{label}: {int(stored.sum())} packed elements, "
          f"{int(np.isnan(got[stored]).sum())} unstored, bitwise "
          f"{np.array_equal(got, want, equal_nan=True)}")
    assert np.array_equal(got, want, equal_nan=True), label


# ---------------------------------------------------------------------------
# Part 1: every class at L <= 4, both forms, on the synthetic bases
# ---------------------------------------------------------------------------


def _part1(backend: str, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    classes = _synthetic()
    n_ao, n_aux = SYNTHETIC.n_ao, SYNTHETIC_AUX.n_ao
    cells, n_col = packed_layout(classes, n_aux)
    for form, low in FORMS.items():
        wd = tmp_path / form
        sq = build_runner(tc, eri3_program(classes, n_ao, n_aux, backend,
                                           wd / "sq", lowstack=low),
                          wd / "sq_bin")
        T = _square(classes, n_ao, n_aux, sq, wd / "sq_case", low)
        B = build_packed_on_card(classes, cells, n_col, n_aux, backend, tc,
                                 wd / "pk", lowstack=low)
        # the bases also reach classes above L = 4, which nothing launches:
        # NaN in both, so the comparison also says which elements stored
        _compare(B, gather_packed(T, classes, cells, n_col, n_aux),
                 f"{backend} {BACKEND[backend][1].name} {form}: "
                 f"{len(classes)} classes")


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_packed_eri3_through_l4_is_the_square_bitwise(request, tmp_path):
    _part1("cuda", request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_packed_eri3_through_l4_is_the_square_bitwise(request, tmp_path):
    _part1("hip", request, tmp_path)


# ---------------------------------------------------------------------------
# Part 2: the 70 classes through (ff|i), on the aux-i set
# ---------------------------------------------------------------------------


def _part2(backend: str, request, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    _, basis, aux = auxi_3c()
    classes = eri3_classes(basis, aux)
    assert sorted((c.la, c.lb, c.lP) for c in classes) == ERI3_LADDER
    n_ao, n_aux = basis.n_ao, aux.n_ao
    cells, n_col = packed_layout(classes, n_aux)
    # (form name, lowstack, the classes run in it)
    runs = [("default", None, classes),
            ("lowstack", True,
             [c for c in classes if not eri3_lowstack((c.la, c.lb, c.lP))])]
    jobs = []  # (form, lP, kernel kind, classes, program dir)
    for form, low, part in runs:
        for lP in range(7):
            sub = [c for c in part if c.lP == lP]
            if not sub:
                continue
            for kind in ("square", "packed"):
                jobs.append((form, low, lP, kind, sub,
                             tmp_path / f"{form}_{kind}_lP{lP}"))

    def compile_job(job):
        form, low, lP, kind, sub, wd = job
        prog = (eri3_program(sub, n_ao, n_aux, backend, wd, lowstack=low)
                if kind == "square" else
                packed_program(sub, backend, wd, lowstack=low))
        return build_runner(tc, prog, wd / "bin", timeout=COMPILE_S)

    workers = max(1, min(len(jobs), int(os.environ.get("JOBS", "6"))))
    with ThreadPoolExecutor(workers) as pool:
        binaries = list(pool.map(compile_job, jobs))
    for form, low, part in runs:
        T = np.zeros((n_ao, n_ao, n_aux))
        B = np.full((n_col, n_aux), np.nan)
        for job, binary in zip(jobs, binaries, strict=True):
            if job[0] != form:
                continue
            _, _, lP, kind, sub, wd = job
            sub_cells = [cells[classes.index(c)] for c in sub]
            if kind == "square":
                T += _square(sub, n_ao, n_aux, binary, wd / "case", low)
            else:
                got = _packed(sub, sub_cells, n_col, n_aux, binary,
                              wd / "case", low)
                mine = ~np.isnan(got)
                assert np.isnan(B[mine]).all()  # disjoint across lP
                B[mine] = got[mine]
        want = gather_packed(T, part, [cells[classes.index(c)] for c in part],
                             n_col, n_aux)
        if form == "default":
            assert not np.isnan(B).any()  # every pair, every aux shell
        _compare(B, want, f"{backend} {BACKEND[backend][1].name} aux-i "
                 f"{form}: {len(part)} classes, "
                 f"{sum(c.n_cell for c in part)} cells")


@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_packed_eri3_through_ff_i_is_the_square_bitwise(request,
                                                             tmp_path):
    _part2("cuda", request, tmp_path)


@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.hip
@pytest.mark.gpu
def test_hip_packed_eri3_through_ff_i_is_the_square_bitwise(request,
                                                            tmp_path):
    _part2("hip", request, tmp_path)


# ---------------------------------------------------------------------------
# Compile-only companions (`cuda` / `hip`, no `gpu`)
# ---------------------------------------------------------------------------


def _program(backend: str, tmp_path: Path):
    return packed_program(_synthetic(), backend, tmp_path)


@pytest.mark.cuda
def test_cuda_packed_eri3_program_compiles(nvcc, tmp_path):
    compile_checked(nvcc, _program("cuda", tmp_path), tmp_path / "bin")


@pytest.mark.hip
def test_hip_packed_eri3_program_compiles(hipcc, tmp_path):
    compile_checked(hipcc, _program("hip", tmp_path), tmp_path / "bin")
