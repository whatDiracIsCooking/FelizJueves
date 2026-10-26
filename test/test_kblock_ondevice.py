"""The whole molecule's `K` on the card, block kernel and `kernel_k` (#481).

WHAT RUNS. `launch_kblock.kb_launches` plans the molecule with
`kernel_kblock.build_kblock_kernel` injected; test/kblock_driver.py writes
ONE DRIVER PER CLASS (`class_binaries`, compiled in parallel), uploads the
class once and runs it, in two processes of the same binary, in MODE 0 (the
block kernel: one launch per class into `K_mn`, no scratch, no scatter) and
MODE 1 (`kernel_k`, #206's launch per (class, b shell) into a per-record
scratch, scattered on the host). Both read the same bytes; the classes'
`K_mn` are summed on the host.

THE ASSERTIONS, each for both modes, tolerances INHERITED from
test_k_ondevice.py and not loosened:

  the walk     `launch_kblock.walk_kb` over the same plan: `|K - K_walk| <=
               ORACLE_TOL max|K_walk|` (1e-13);
  the host     `eri.k_exchange.k.exchange_matrix_and_scale` (the unique sweep), `|K -
               K0| <= ORACLE_TOL max|K0| + tau R`;
  libcint      the tracked `get_jk` fixture's `vk`, `|K - vk| <= KJ_TOL
               scale_K + tau R` (1e-12), the whole matrix;

`R` is the walk's `scale`, the record pairs covering each element: every
skipped pair moves an element by less than `tau` (#200's bound).
  run to run   MODE 0 is bit-identical across its two runs: within a launch
               no two blocks share an element and the launches serialize on
               one stream, so its atomics meet nothing (kernel_kblock's
               docstring). MODE 1's two runs are within #206's fp64-atomic
               bound, so only printed here.

THE PER-BLOCK FACTOR (#484): the `decay` cell plans test_ao_k_decay.py's
chain with `eri.k_exchange.k.density_decay`, so the block kernel screens each block
with its own `dmax`; MODE 1 (`kernel_k`) still takes the scalar `max|D|`
and skips fewer pairs, so there it is held to the host and not to the walk.

THE CELLS: H2O/STO-3G whole (every class through (pp|pp), L = 4) at tau 0
and at a tau that screens, two H2 6 bohr apart screened, and the decaying
chain with the per-block factor -- the PR gate's; H2O/cc-pVDZ whole (every
class through (dd|dd), L = 8) at tau 0 is `slow`, the kernel tier's
`--full`.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`): the H2O/STO-3G driver
compiled where no card is; a warning in generated text fails it (#92).
Without a card the `gpu` tests SKIP: green and empty.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache, partial
from pathlib import Path

import numpy as np
import pytest
import test_ao_k
from kblock_driver import MODES, class_binaries, kb_driver, run_classes
from oracle import manifest
from plan_driver import compile_checked
from test_ao_k_decay import TAU, chain
from test_k_ondevice import ORACLE_TOL
from test_kernel_device import BACKEND, _toolchain
from test_kernel_kblock import sf_basis
from test_launch_k import _case as _launch_case
from tiers import DEVICE_L_MAX

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.k_exchange.k import (
    density_decay,
    exchange_matrix_and_scale,
    record_bounds,
)
from felizjueves.eri.k_exchange.kernel_kblock import build_kblock_kernel
from felizjueves.eri.k_exchange.launch_kblock import KBLaunchPlan, kb_launches, walk_kb
from felizjueves.reference.pq import pq

KJ_TOL = test_ao_k.KJ_TOL


@dataclass(frozen=True)
class Cell:
    case: str
    tau: float
    decay: bool = False
    with_k: bool = True  # False: the block kernel only (MODE 0), #569
    lmax: int | None = None  # the plan cut to its classes with L <= lmax

    @property
    def id(self) -> str:
        return (f"{self.case}-tau{self.tau:g}" + ("-decay" if self.decay
                                                   else "")
                + ("" if self.with_k else "-kb")
                + ("" if self.lmax is None else f"-L{self.lmax}"))


FAST = (Cell("h2o_sto3g_k", 0.0), Cell("h2o_sto3g_k", 1e-8),
        Cell("two_h2", 1e-6), Cell("chain", TAU, decay=True))
SLOW = (Cell("h2o_ccpvdz_k", 0.0),)
# f shells (#569, E held factored): `spdf`, one shell of each l <= 3 on four
# centres -- every K class with every l <= 3, 256 of them, (ff|ff) among
# them -- the block kernel only (`kernel_k` is the same body; doubling the
# compile of 256 classes buys nothing); and `sf`, an s and an f shell, its
# 16 classes in both modes, so `kernel_k` runs (ff|ff) on the card too.
# Above tiers.DEVICE_L_MAX: `slow` and `high_l`, run by hand. The `-L9`
# cells cut the plan to its classes with L <= 9 (241 of spdf's 256, 20 of
# them with an (f, f) pair; 12 of sf's 16) and are held to the walk over
# that cut plan only: a class at L >= 10 did not finish compiling on nvcc
# within the budget (#569: kb_33_33 was still in cicc after 2.7 h, factored
# or not), so the whole-plan cells are written and NOT run here.
HIGH_L = (Cell("spdf", 0.0, with_k=False, lmax=9), Cell("sf", 0.0, lmax=9),
          Cell("spdf", 0.0, with_k=False), Cell("sf", 0.0))
HIGH_L_COMPILE_S = 6 * 3600.0
PARAMS = ([pytest.param(c, id=c.id) for c in FAST]
          + [pytest.param(c, id=c.id, marks=pytest.mark.slow) for c in SLOW]
          + [pytest.param(c, id=c.id, marks=(pytest.mark.slow,
                                             pytest.mark.high_l))
             for c in HIGH_L])


def _spdf() -> tuple[Basis, np.ndarray]:
    """An s, p, d and f shell on four centres, two primitives each (one
    contracted pair per shell pair, four records), and a symmetric density."""
    shells = (Shell(0, (0.0, 0.0, 0.0), (1.2, 0.4), (0.6, 0.5)),
              Shell(1, (0.5, -0.6, 0.3), (0.9, 0.35), (0.7, 0.4)),
              Shell(2, (-0.4, 0.7, 0.6), (0.8, 0.3), (0.5, 0.6)),
              Shell(3, (0.6, 0.5, -0.7), (0.7, 0.3), (0.45, 0.55)))
    basis = Basis(shells)
    n = basis.n_ao
    return basis, np.array([[1.0 / (i + j + 1) for j in range(n)]
                            for i in range(n)])


@cache
def _case(name: str):
    """(libcint fixture or None, basis, D)."""
    if name.endswith("_k"):
        fx, basis = test_ao_k._kfx(name)
        return fx, basis, np.array(fx.density)
    if name == "chain":
        return None, *chain()
    if name == "spdf":
        return None, *_spdf()
    if name == "sf":
        return None, *sf_basis()
    basis, D = _launch_case(name)
    return None, basis, D


@cache
def _plan(cell: Cell) -> KBLaunchPlan:
    _, basis, D = _case(cell.case)
    bound = partial(record_bounds, pq=pq)
    decay = (density_decay(basis, D, bound=bound, tau=cell.tau)
             if cell.decay else None)
    if decay is not None:
        assert any(b.dmax < decay.dm for c in kb_launches(
            basis, D, bound=bound, tau=cell.tau, decay=decay)
            for b in c.blocks), "no block's factor is under max|D|"
    plan = kb_launches(basis, D, bound=bound, tau=cell.tau,
                       build=build_kblock_kernel, decay=decay)
    if cell.lmax is not None:
        plan = replace(plan, classes=tuple(c for c in plan
                                           if c.L <= cell.lmax))
    return plan


@cache
def _host(case: str):
    _, basis, D = _case(case)
    return tuple(np.array(x) for x in exchange_matrix_and_scale(basis, D,
                                                                pq=pq))


@cache
def _walk(cell: Cell):
    _, basis, D = _case(cell.case)
    return walk_kb(_plan(cell), basis, D, pq=pq)


def _check(backend: str, cell: Cell, request, run_device_binary,
           workdir: Path) -> None:
    tc = _toolchain(backend, request)
    fx, basis, D = _case(cell.case)
    plan = _plan(cell)
    high = max(c.L for c in plan) > DEVICE_L_MAX
    parts = class_binaries(tc, plan, basis, D, backend, workdir,
                           with_k=cell.with_k,
                           # an f-pair class's unit compiles for an hour or
                           # more (#569): its source is the unrolled
                           # contraction, not E
                           timeout=HIGH_L_COMPILE_S if high else 1800)
    w = _walk(cell)
    Kw = np.array(w.K)
    K0, scale = _host(cell.case) if cell.lmax is None else (Kw, None)
    tau_scale = cell.tau * np.array(w.scale)  # #200: < tau per skipped pair
    if cell.tau > 0.0:
        assert w.n_skipped > 0, "the screen never fired"
    lines = []
    for mode in MODES if cell.with_k else ("kb",):
        Ks, _, same = run_classes(parts, basis, mode, run_device_binary,
                                  backend)
        if mode == "kb":
            assert same, "the block kernel's two runs differ: its atomics met"
        K = Ks[0]
        assert np.all(np.isfinite(K))
        rel_w = float(np.max(np.abs(K - Kw)) / np.max(np.abs(Kw)))
        if not (cell.decay and mode == "k"):  # kernel_k: the scalar max|D|
            assert rel_w <= ORACLE_TOL, f"{mode}: K vs walk_kb {rel_w:.2e}"
        big0 = float(np.max(np.abs(K0)))
        # a cut plan builds part of K: held to the walk over that plan only
        assert cell.lmax is None or fx is None
        assert cell.lmax is not None or np.all(
            np.abs(K - K0) <= ORACLE_TOL * big0 + tau_scale), (
            f"{mode}: K vs exchange_matrix "
            f"{float(np.max(np.abs(K - K0))) / big0:.2e}")
        if cell.decay and mode == "kb":  # within the skipped pairs' bounds
            assert np.all(np.abs(K - K0) <= ORACLE_TOL * big0
                          + np.array(w.bound)), f"{mode}: K vs the bound"
        lib = "none"
        if fx is not None:
            vk = np.array(fx.arrays[manifest.K_ARRAY])
            err = np.abs(K - vk)
            assert np.all(err <= KJ_TOL * scale + tau_scale), (
                f"{mode}: K vs libcint, worst "
                f"{float(np.max(err / scale)):.2e} of scale_K")
            lib = f"{float(np.max(err / scale)):.2e} of scale_K"
        lines.append(f"{mode}: vs walk {rel_w:.2e}, vs libcint {lib}, runs "
                     f"bit-identical {same}")
    arch = BACKEND[backend][1]
    print(f"\n{backend} {arch.name} {cell.id}: {len(plan)} classes, "
          f"{sum(len(c.blocks) for c in plan)} blocks, walk skipped "
          f"{w.n_skipped} of {w.n_pairs}; " + "; ".join(lines))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cell", PARAMS)
def test_cuda_kblock_from_the_launch_plan(cell, request, run_device_binary,
                                          tmp_path) -> None:
    _check("cuda", cell, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cell", PARAMS)
def test_hip_kblock_from_the_launch_plan(cell, request, run_device_binary,
                                         tmp_path) -> None:
    _check("hip", cell, request, run_device_binary, tmp_path)


def _compile_only(backend: str, tc, tmp_path: Path) -> None:
    cell = FAST[0]
    _, basis, _ = _case(cell.case)
    drv = kb_driver(_plan(cell), basis, backend, tmp_path / "gen")
    compile_checked(tc, drv, tmp_path / "kb_driver")


@pytest.mark.cuda
def test_cuda_kblock_driver_compiles(nvcc, tmp_path) -> None:
    _compile_only("cuda", nvcc, tmp_path)


@pytest.mark.hip
def test_hip_kblock_driver_compiles(hipcc, tmp_path) -> None:
    _compile_only("hip", hipcc, tmp_path)
