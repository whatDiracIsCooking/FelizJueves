"""The uncontracted `(mn|P)` d1 kernel (`xeri3_`, #692) ON A CARD: every
class of a system launched into one `X[3, n_ao, n_ao, n_blk]`, against
`gradient.host_df.eri3_ip1` (the arrays are `df_grad_launch`'s, the
interpreted tests'; the runner is `harness_driver`'s chain, `X` carried).

PART 1 (`slow`). H2O/STO-3G against the cc-pVTZ-JKFIT aux basis's s, p
and d shells (every class at L <= 4, `(pp|d)` the highest), two launches
per backend:

- `whole`: one block of every aux shell, every cell;
- `block`: a STRICT SUBSET, the aux shells 1..n-2 and every other cell of
  them, into an `X` filled with a sentinel -- the cells' slots must be the
  host's and every other entry the sentinel, untouched.

PART 2 (`slow` + `high_l`), THE LADDER THROUGH (ff|i): the aux-i
witness's system (`manifest.df_gradient`, `prim_spd_f` with l = 0..6 aux
shells), which reaches every class of `--grad-dfx-l 3 --grad-dfx-aux-l 6`;
one program per aux `l`, each compile under the user's 300 s budget
(`COMPILE_BUDGET_S`), as test_grad_df_device.py's Part 2.

THE BOUND, stated before any run: `|X_dev - X_host| <= DEV_TOL max|X|`,
`DEV_TOL = 1e-13`, test_grad_df_device.py's: the interpreted kernel is
within 16 eps of the host (test_kernel_xeri3.py), and the card's `exp` and
Boys differ by ulps. The stores are plain and each slot is one thread's, so
the `RUNS` runs must be BIT-IDENTICAL (no atomic reorders an add).

NOT RUN ON A CARD: the session that wrote this file had none. Its compile
twins (`test_*_xeri3_program_compiles`: Part 1's block runners built, no
card) passed under nvcc and hipcc, and its staging, run through the
interpreter in place of the card, matched the host to 1.2e-16 of max.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from df_grad_launch import xeri3_arrays
from grad_driver import THREADS
from harness_driver import Step, build_steps, chain_specs, run_chain
from oracle import basis as ob
from oracle import manifest
from plan_driver import RUNS
from test_kernel_device import _toolchain

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.gradient import host_df
from felizjueves.gradient.kernel_xeri3 import build_xeri3_kernel, launch_shape
from felizjueves.reference.pq import pq

DEV_TOL = 1e-13  # of max|X| (module docstring)
COMPILE_BUDGET_S = 300.0  # the user's per-compile budget (#613)
SENTINEL = 7.25
SET = "h2o_sto3g_jkfit_3c"


@cache
def _h2o():
    ts = manifest.three_center_set(SET)
    basis = ob.from_fixture(ts.parent)
    aux = Basis(tuple(s for s in ob.from_fixture(ts).shells if s.l <= 2))
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return basis, aux, nuclei


@cache
def _auxi():
    ts = manifest.df_gradient().three_center
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return basis, aux, nuclei


@cache
def _host(system: str) -> np.ndarray:
    basis, aux, _ = _h2o() if system == "h2o" else _auxi()
    return np.array(host_df.eri3_ip1(basis, aux, pq=pq))


def _launch(basis, aux, nuclei, shells: range, *, every: int = 1,
            lP: int | None = None
            ) -> tuple[list[Step], list[tuple[int, int, int]], int, int]:
    """One step per class with a cell in the block: the block's cells (every
    `every`-th of each class's), `lP` the aux `l` alone when given; returns
    the steps, the cells, `aux0` and `n_blk`."""
    aux0 = aux.offsets[shells.start]
    n_blk = sum(aux.shells[n].n_cart for n in shells)
    steps, cells = [], []
    for cls in eri3_classes(basis, aux):
        if lP is not None and cls.lP != lP:
            continue
        keep = [i for i, (_, _, n) in enumerate(cls.cells)
                if n in shells][::every]
        if not keep:
            continue
        cells += [(min(a, b), max(a, b), n)
                  for a, b, n in (cls.cells[i] for i in keep)]
        grid, block = launch_shape(len(keep), THREADS)
        steps.append(Step(
            build_xeri3_kernel((cls.la, cls.lb, cls.lP)), grid, block,
            {**xeri3_arrays(cls, basis, aux, nuclei, keep),
             "n_cell": len(keep), "n_bra_rec": cls.n_bra_rec,
             "n_aux_rec": cls.n_aux_rec, "n_ao": basis.n_ao,
             "n_blk": n_blk, "aux0": aux0}))
    return steps, cells, aux0, n_blk


def _owned(basis, aux, cells, aux0: int, n_blk: int) -> np.ndarray:
    out = np.zeros((3, basis.n_ao, basis.n_ao, n_blk), dtype=bool)
    off, aoff = basis.offsets, aux.offsets
    for i, j, n in cells:
        m = slice(off[i], off[i] + basis.shells[i].n_cart)
        q = slice(off[j], off[j] + basis.shells[j].n_cart)
        p = slice(aoff[n] - aux0, aoff[n] - aux0 + aux.shells[n].n_cart)
        out[:, m, q, p] = out[:, q, m, p] = True
    return out


def _run(backend: str, request, tmp_path: Path, system: str, shells: range,
         *, every: int = 1, lP: int | None = None,
         timeout: float = 1800.0) -> None:
    tc = _toolchain(backend, request)
    basis, aux, nuclei = _h2o() if system == "h2o" else _auxi()
    steps, cells, aux0, n_blk = _launch(basis, aux, nuclei, shells,
                                        every=every, lP=lP)
    X0 = np.full((3, basis.n_ao, basis.n_ao, n_blk), SENTINEL)
    bins = build_steps(tc, backend, steps, tmp_path, timeout=timeout)
    got = run_chain(bins, steps, "X", X0, tmp_path / "runs", runs=RUNS)
    owned = _owned(basis, aux, cells, aux0, n_blk)
    want = _host(system)[..., aux0:aux0 + n_blk]
    scale = float(np.abs(want[owned]).max())
    errs = [float(np.abs(x[owned] - want[owned]).max() / scale) for x in got]
    print(f"\n{backend} xeri3 {system} aux {shells}: |X - host| / max|X| "
          f"per run {errs}; bit-identical "
          f"{all(np.array_equal(got[0], x) for x in got)}")
    assert max(errs) <= DEV_TOL
    for x in got:
        assert np.all(x[~owned] == SENTINEL)
        assert np.array_equal(x, got[0])


def _block(system: str) -> range:
    aux = (_h2o() if system == "h2o" else _auxi())[1]
    return range(1, len(aux.shells) - 1)


PART1 = {"whole": (None, 1), "block": ("strict", 2)}


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("which", sorted(PART1))
def test_cuda_xeri3_matches_the_host(which, request, tmp_path) -> None:
    _part1("cuda", which, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("which", sorted(PART1))
def test_hip_xeri3_matches_the_host(which, request, tmp_path) -> None:
    _part1("hip", which, request, tmp_path)


def _part1(backend: str, which: str, request, tmp_path: Path) -> None:
    aux = _h2o()[1]
    shells = (range(len(aux.shells)) if PART1[which][0] is None
              else _block("h2o"))
    _run(backend, request, tmp_path, "h2o", shells, every=PART1[which][1])


@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_xeri3_program_compiles(nvcc, tmp_path) -> None:
    _compiles(nvcc, "cuda", tmp_path)


@pytest.mark.hip
@pytest.mark.slow
def test_hip_xeri3_program_compiles(hipcc, tmp_path) -> None:
    _compiles(hipcc, "hip", tmp_path)


def _compiles(tc, backend: str, tmp_path: Path) -> None:
    """Part 1's runners built, no card: the harness and every unit."""
    basis, aux, nuclei = _h2o()
    steps = _launch(basis, aux, nuclei, _block("h2o"), every=2)[0]
    bins = build_steps(tc, backend, steps, tmp_path)
    assert set(bins) == {s.kernel.name for s in steps}
    assert all(p.exists() for p in bins.values())


@pytest.mark.slow
def test_the_staged_block_is_a_strict_subset_and_covers_every_class(
        tmp_path) -> None:
    # Host-only: the program's input, no compiler, no card.
    basis, aux, nuclei = _h2o()
    whole = _launch(basis, aux, nuclei, range(len(aux.shells)))
    part = _launch(basis, aux, nuclei, _block("h2o"), every=2)
    assert max(sum(map(int, s.kernel.name[6:])) for s in whole[0]) == 4
    assert {s.kernel.name for s in whole[0]} == {
        f"xeri3_{c.la}{c.lb}{c.lP}" for c in eri3_classes(basis, aux)}
    assert 0 < len(part[1]) < len(whole[1]) and set(part[1]) < set(whole[1])
    assert 0 < part[3] < whole[3]
    for steps in (whole[0], part[0]):
        specs = chain_specs(steps, "X", np.zeros((3, basis.n_ao, basis.n_ao,
                                                  steps[0].args["n_blk"])))
        assert [s["kernel"] for s in specs] == [s.kernel.name for s in steps]


# --- Part 2: the ladder through (ff|i) ---------------------------------------


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.parametrize("aux_l", range(7), ids=lambda x: f"xeri3-{x}")
def test_cuda_xeri3_ladder_through_i(aux_l, request, tmp_path) -> None:
    _part2("cuda", aux_l, request, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.high_l
@pytest.mark.parametrize("aux_l", range(7), ids=lambda x: f"xeri3-{x}")
def test_hip_xeri3_ladder_through_i(aux_l, request, tmp_path) -> None:
    _part2("hip", aux_l, request, tmp_path)


def _part2(backend: str, aux_l: int, request, tmp_path: Path) -> None:
    aux = _auxi()[1]
    _run(backend, request, tmp_path, "auxi", range(len(aux.shells)),
         lP=aux_l, timeout=COMPILE_BUDGET_S)
