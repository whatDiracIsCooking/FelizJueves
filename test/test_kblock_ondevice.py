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

THE ROLLED KERNEL (#571): every cell again with `rolled=True`, MODE 0
`kernel_kblock_rolled`'s kernel on the same upload (`E` and the class
tables appended), `kernel_k` not built, held to the same walk, host and
libcint bounds as the block kernel, its two runs bit-identical. Its f cells
are the point of it: `spdf`, an s, p, d and f shell on four centres, two
primitives each -- every K class with every `l <= 3`, 256 of them,
`(ff|ff)` among them -- and `sf`, an s and an f shell, its 16 classes,
against the walk and the host (no libcint `vk` has an f shell yet, #570).
Above `tiers.DEVICE_L_MAX` they are `slow` and `high_l`, run by hand
(`-m "gpu and cuda and high_l"`); they compile in seconds, not hours.

THE DENSE-HERMITE KERNEL (#575): every rolled cell again with `dense=True`,
MODE 0 `kernel_kblock_rolled.build_kblock_dense_kernel`'s kernel on the
same upload with its `E` dense and no class tables, held to the same
bounds, its two runs bit-identical; its `sf` and `spdf` cells beside the
rolled kernel's.

THE TRANSPOSED KET TABLE (#576): every rolled cell again with
`transposed=True`, MODE 0 the rolled kernel with `Ebd` stored
`[n_ebd, n_ket]` (`kbrt_*`) and staged so, held to the same bounds, its
two runs bit-identical; its `sf` and `spdf` cells beside the others.

THE COLUMN FORM (#583): every rolled cell again with `column=True`, MODE 0
the rolled kernel with each `K_AB` column reduced across the warp per
(bra, ket strip) (`kbrc_*`), on the rolled upload unchanged, held to the
same bounds, its two runs bit-identical; its `sf` and `spdf` cells beside
the others, and ONE g cell, `gg`: a single g shell, two primitives -- the
one class `(gg|gg)`, which no other form launches (its `Ksh` is 99,840 B a
warp without the column cut, 47,880 B with it, under 48 KiB with no
opt-in) -- against `walk_kb` ONLY (no libcint `vk` has a g shell, and the
host sweep adds nothing the interpreted tests do not already hold). A
625-class s, p, d, f, g cell was tried first and dropped: about 45 minutes
of its run was the pure-Python host references, not the card.

COMPILE-ONLY COMPANIONS (`cuda` / `hip`, no `gpu`): the H2O/STO-3G driver
compiled where no card is, both kernels; a warning in generated text fails
it (#92). Without a card the `gpu` tests SKIP: green and empty.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    rolled: bool = False  # MODE 0 the rolled kernel, kernel_k unbuilt (#571)
    dense: bool = False  # with rolled: the dense-Hermite kernel (#575)
    transposed: bool = False  # with rolled: Ebd [n_ebd, n_ket] (#576)
    column: bool = False  # with rolled: the column form (#583)

    @property
    def id(self) -> str:
        return (f"{self.case}-tau{self.tau:g}"
                + ("-decay" if self.decay else "")
                + ("-dense" if self.dense else "-rolled-T" if self.transposed
                   else "-rolled-C" if self.column
                   else "-rolled" if self.rolled else ""))


FAST = (Cell("h2o_sto3g_k", 0.0), Cell("h2o_sto3g_k", 1e-8),
        Cell("two_h2", 1e-6), Cell("chain", TAU, decay=True))
SLOW = (Cell("h2o_ccpvdz_k", 0.0),)
ROLLED = tuple(Cell(c.case, c.tau, c.decay, rolled=True) for c in FAST)
ROLLED_SLOW = tuple(Cell(c.case, c.tau, c.decay, rolled=True) for c in SLOW)
DENSE = tuple(Cell(c.case, c.tau, c.decay, rolled=True, dense=True)
              for c in FAST)
DENSE_SLOW = tuple(Cell(c.case, c.tau, c.decay, rolled=True, dense=True)
                   for c in SLOW)
TRANSPOSED = tuple(Cell(c.case, c.tau, c.decay, rolled=True, transposed=True)
                   for c in FAST)
TRANSPOSED_SLOW = tuple(Cell(c.case, c.tau, c.decay, rolled=True,
                             transposed=True) for c in SLOW)
COLUMN = tuple(Cell(c.case, c.tau, c.decay, rolled=True, column=True)
               for c in FAST)
COLUMN_SLOW = tuple(Cell(c.case, c.tau, c.decay, rolled=True, column=True)
                    for c in SLOW)
HIGH_L = (Cell("sf", 0.0, rolled=True), Cell("spdf", 0.0, rolled=True),
          Cell("sf", 0.0, rolled=True, dense=True),
          Cell("spdf", 0.0, rolled=True, dense=True),
          Cell("sf", 0.0, rolled=True, transposed=True),
          Cell("spdf", 0.0, rolled=True, transposed=True),
          Cell("sf", 0.0, rolled=True, column=True),
          Cell("spdf", 0.0, rolled=True, column=True),
          Cell("gg", 0.0, rolled=True, column=True))
WALK_ONLY = frozenset({"gg"})  # #583: held to walk_kb alone
# The PR gate holds the forms a class can default to (`kb`, `kbr`, `kbrc`);
# the dense (#575) and transposed (#576) forms, measured and not chosen, run
# their PR-gate cells under `slow` (`--full`), ~100 s of the CUDA gate.
PARAMS = ([pytest.param(c, id=c.id) for c in (*FAST, *ROLLED, *COLUMN)]
          + [pytest.param(c, id=c.id, marks=pytest.mark.slow)
             for c in (*DENSE, *TRANSPOSED, *SLOW, *ROLLED_SLOW, *DENSE_SLOW,
                       *TRANSPOSED_SLOW, *COLUMN_SLOW)]
          + [pytest.param(c, id=c.id, marks=(pytest.mark.slow,
                                             pytest.mark.high_l))
             for c in HIGH_L])


def _f_basis(ls: tuple[int, ...]) -> tuple[Basis, np.ndarray]:
    """One shell per `l` of `ls` on its own centre, two primitives each, and
    a symmetric density (#571's f cells; #583's g cell)."""
    centres = ((0.0, 0.0, 0.0), (0.5, -0.6, 0.3), (-0.4, 0.7, 0.6),
               (0.6, 0.5, -0.7), (-0.5, -0.4, -0.6))
    prims = {0: ((1.2, 0.4), (0.6, 0.5)), 1: ((0.9, 0.35), (0.7, 0.4)),
             2: ((0.8, 0.3), (0.5, 0.6)), 3: ((0.7, 0.3), (0.45, 0.55)),
             4: ((0.6, 0.25), (0.4, 0.6))}
    basis = Basis(tuple(Shell(l, centres[i], *prims[l])
                        for i, l in enumerate(ls)))  # noqa: E741
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
        return None, *_f_basis((0, 1, 2, 3))
    if name == "sf":
        return None, *_f_basis((0, 3))
    if name == "gg":  # #583: (gg|gg) alone
        return None, *_f_basis((4,))
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
    return kb_launches(basis, D, bound=bound, tau=cell.tau,
                       build=build_kblock_kernel, decay=decay)


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
    assert cell.rolled or max(c.L for c in plan) <= DEVICE_L_MAX
    parts = class_binaries(tc, plan, basis, D, backend, workdir,
                           rolled=cell.rolled, dense=cell.dense,
                           transposed=cell.transposed, column=cell.column,
                           with_k=not cell.rolled,
                           # #583: a column unit compiles in seconds; a
                           # driver past five minutes is a failure, not a wait
                           **({"timeout": 300.0} if cell.column else {}))
    w = _walk(cell)
    Kw = np.array(w.K)
    walk_only = cell.case in WALK_ONLY
    K0, scale = (None, None) if walk_only else _host(cell.case)
    tau_scale = cell.tau * np.array(w.scale)  # #200: < tau per skipped pair
    if cell.tau > 0.0:
        assert w.n_skipped > 0, "the screen never fired"
    lines = []
    for mode in ("kb",) if cell.rolled else MODES:
        Ks, _, same = run_classes(parts, basis, mode, run_device_binary,
                                  backend)
        if mode == "kb":
            assert same, "the block kernel's two runs differ: its atomics met"
        mode = ("kbd" if cell.dense else "kbrt" if cell.transposed
                else "kbrc" if cell.column else "kbr" if cell.rolled
                else mode)  # the lines' and errors' name
        K = Ks[0]
        assert np.all(np.isfinite(K))
        rel_w = float(np.max(np.abs(K - Kw)) / np.max(np.abs(Kw)))
        if not (cell.decay and mode == "k"):  # kernel_k: the scalar max|D|
            assert rel_w <= ORACLE_TOL, f"{mode}: K vs walk_kb {rel_w:.2e}"
        if not walk_only:
            big0 = float(np.max(np.abs(K0)))
            assert np.all(np.abs(K - K0) <= ORACLE_TOL * big0 + tau_scale), (
                f"{mode}: K vs exchange_matrix "
                f"{float(np.max(np.abs(K - K0))) / big0:.2e}")
        if cell.decay and mode != "k":  # within the skipped pairs' bounds
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
    print(f"\n{backend} {arch.name} {cell.id}: {len(plan)} classes "
          f"(L <= {max(c.L for c in plan)}), "
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
    for rolled, dense, tr, col in ((False, False, False, False),
                                   (True, False, False, False),
                                   (True, True, False, False),
                                   (True, False, True, False),
                                   (True, False, False, True)):
        w = tmp_path / ("dense" if dense else "rolled-T" if tr
                        else "rolled-C" if col
                        else "rolled" if rolled else "kb")
        drv = kb_driver(_plan(cell), basis, backend, w / "gen", rolled=rolled,
                        dense=dense, transposed=tr, column=col)
        compile_checked(tc, drv, w / "kb_driver")


@pytest.mark.cuda
def test_cuda_kblock_driver_compiles(nvcc, tmp_path) -> None:
    _compile_only("cuda", nvcc, tmp_path)


@pytest.mark.hip
def test_hip_kblock_driver_compiles(hipcc, tmp_path) -> None:
    _compile_only("hip", hipcc, tmp_path)
