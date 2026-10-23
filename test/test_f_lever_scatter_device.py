"""The atomic-free F write-back (#358) through the compilers and on the cards.

THREE PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): every F class with L <= 4
   (milestone 13's cap), the literal kernel, block-fab's (#351, H7) and the
   scatter lever's each its own translation unit, and the class-agnostic
   `f_scatter` unit once, compiled with the tier's flags (test_f_lever_
   batched_vrr_device.py's `compile_unit`). A warning in generated text
   fails (#92). One reading is asserted, a property of the launch: the
   lever's static shared bytes equal block-fab's (the same two shared
   arrays; `shared_bytes`, the resource report). Registers and spills are
   the lever ledger's (`measured --remeasure ... --ledger`), not this
   file's. The scatter programs (per class and per molecule) are compiled
   too, so a spelling problem shows without the card.
2. ON THE CARD, PER CLASS (`gpu` + `cuda` / `hip`, `slow`): every F class
   with L <= 4, its launches in H2O/cc-pVDZ's plan at `tau = 0`
   (test_fock_ondevice.py's `_class_recs`: the same packed records, `diag`,
   `Dsh` and `D`), each relaunched with the lever's kernel at 32 and 128
   threads and followed by the scatter over its map, held by
   test_fock_ondevice.py's own `_check_class` -- `TOL = 1e-13` of `scale_G`
   against the restricted sweep, `2 TOL` against the literal launches
   interpreted -- and, the lever's point, the two runs of one binary
   BIT-IDENTICAL (`F` is deterministic: no atomic anywhere). `slow` because
   cc-pVDZ is not a small input: its host references cost 15-30 s a class.
3. ON THE CARD, PER MOLECULE: H2O/STO-3G at `tau = 0` and `tau = 0.1, n =
   4` (the PR tier's), and, `slow`, H2O/6-31G at `tau = 0` (every class
   L <= 4), every launch of the plan through the scatter path into one `F`,
   against walk_f's interpreted literal build at `2 TOL`, at `tau = 0`
   against the sweep at `TOL` and libcint at `LIBCINT_TOL`, the two runs
   bit-identical -- the PR tier's witness of the lever's determinism.

The write-back tail's clock (#358's "measured before the verdict") is not a
test: it is measure/measure_f_scatter_tail.py, run by `devtools/measure.sh
scatter-tail --backend cuda|hip`, and imports this file's helpers.

Run parts 2 and 3 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip --full -j 6 -- \\
        -k f_lever_scatter

(without `--full` only part 3's STO-3G cells are selected, nothing `slow`).
"""

from __future__ import annotations

import dataclasses
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_fock_ondevice as tfo
from fock_driver import (
    FRec,
    fock_scatter_program,
    fock_scatter_stage,
    interpret_scatter,
    run_on_card,
    scatter_map_np,
)
from plan_driver import compile_checked
from test_f_lever_batched_vrr_device import compile_unit, shared_bytes
from test_fock_build import LIBCINT_TOL, TOL, sweep, walk

from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock import kernel_f_scatter as FS
from felizjueves.eri.fock.levers import block_fab, scatter
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.ir.lever_ladder import unit_name

NAME = scatter.NAME
L4 = tuple(c for c in CLASSES if sum(c) <= 4)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in L4]
DEVICE_THREADS = (32, 128)
WAVEFRONT = 32  # both cards' (device.SM_86, device.GFX1200)
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, id="h2o_sto3g-tau0"),
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0",
                 marks=pytest.mark.slow),  # not a small input: the PR tier's cut
]


def _named(spec_levers, cls) -> Kernel:
    k = KF.build_f_kernel(KF.f_class_spec(*cls, spec_levers))
    return dataclasses.replace(k, name=unit_name(k.name, spec_levers))


def lever(cls) -> Kernel:
    return _named((NAME,), cls)


def as_lever(recs, threads: int | None = None) -> tuple[FRec, ...]:
    """Recorded literal launches relaunched with the lever's kernel (the
    same arguments; `threads` checked against block-fab's contract)."""
    out, kernels = [], {}
    for r in recs:
        cls = tuple(int(c) for c in r.name[2:6])
        k = kernels.setdefault(cls, lever(cls))
        t = r.threads if threads is None else threads
        grid, (t, _) = block_fab.launch_shape(r.grid, t, WAVEFRONT)
        out.append(FRec(k, grid, t, r.args))
    return tuple(out)


def _kernels(recs) -> list[Kernel]:
    out: dict[str, Kernel] = {}
    for r in recs:
        out.setdefault(r.name, r.kernel)
    return list(out.values())


def _build(backend: str, tc, kernels, tmp_path: Path, name: str, *,
           f_only: bool = False) -> Path:
    name = f"prog_{name}"
    prog = fock_scatter_program(kernels, backend, tmp_path, name=name,
                                f_only=f_only)
    return compile_checked(tc, prog, tmp_path / f"{name}_bin")


# ---------------------------------------------------------------------------
# 1. Compile-only: the units, the shared footprint, and the programs
# ---------------------------------------------------------------------------


def _compiles(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    compile_unit(backend, compiler, KF.build_f_kernel(cls),
                 tmp_path / "literal")
    h7 = shared_bytes(backend, compiler, _named((block_fab.NAME,), cls),
                      tmp_path / "block_fab")
    lev = shared_bytes(backend, compiler, lever(cls), tmp_path / "scatter")
    assert lev == h7, (h7, lev)  # ab_sh + fab_sh, both


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_scatter_kernels_compile(cls, nvcc, tmp_path) -> None:
    _compiles("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_scatter_kernels_compile(cls, hipcc, tmp_path) -> None:
    _compiles("hip", hipcc.path, cls, tmp_path)


@pytest.mark.cuda
def test_cuda_the_scatter_unit_compiles(nvcc, tmp_path) -> None:
    compile_unit("cuda", nvcc.path, FS.build_scatter_kernel(), tmp_path)


@pytest.mark.hip
def test_hip_the_scatter_unit_compiles(hipcc, tmp_path) -> None:
    compile_unit("hip", hipcc.path, FS.build_scatter_kernel(), tmp_path)


@cache
def _molecule_recs(name: str, tau: float, n: int) -> tuple[FRec, ...]:
    recs, _ = tfo._plan(name, tau, n)
    return as_lever(recs)


PROGRAM_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}")
      for c in L4),
    *(pytest.param(("molecule", *p.values), id=p.id, marks=p.marks)
      for p in MOLECULES),
]


def _programs(backend: str, tc, cell, tmp_path: Path) -> None:
    if cell[0] == "class":
        kernels, name = [lever(cell[1])], "".join(map(str, cell[1]))
    else:
        kernels, name = _kernels(_molecule_recs(*cell[1:])), cell[1]
    for f_only in (False, True):
        _build(backend, tc, kernels, tmp_path / str(f_only), name,
               f_only=f_only)


@pytest.mark.cuda
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_cuda_scatter_programs_compile(cell, nvcc, tmp_path) -> None:
    _programs("cuda", nvcc, cell, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_hip_scatter_programs_compile(cell, hipcc, tmp_path) -> None:
    _programs("hip", hipcc, cell, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card, per class
# ---------------------------------------------------------------------------


def _class(backend: str, cls, threads: int, request, run_device_binary,
           tmp_path: Path) -> None:
    tc = tfo._toolchain(backend, request)
    recs = as_lever(tfo._class_recs(cls), threads)
    kernels = [recs[0].kernel]
    binary = _build(backend, tc, kernels, tmp_path, kernels[0].name)
    basis, _, _ = tfo.molecule(tfo.PART1)
    Fs, _ = run_on_card(binary, fock_scatter_stage(recs, kernels),
                        basis.n_ao, tmp_path, run_device_binary,
                        backend=backend)
    w = tfo._check_class(backend, cls, Fs)
    print(f"\n{backend} {cls} {NAME} at {threads} threads on {tfo.PART1}: "
          f"{len(recs)} launches + their scatters; worst |F - sweep| / "
          f"scale {w['sweep']:.2e} (tol {tfo.TOL:g}), |F - literal interp| "
          f"/ scale {w['interp']:.2e} (tol {2 * tfo.TOL:g}), bit-identical "
          f"runs {bool(w['bitwise_runs'])}", flush=True)
    assert np.array_equal(Fs[0], Fs[1]), "F is not deterministic"


DEVICE_PARAMS = [pytest.param(c, t, id=f"{''.join(map(str, c))}-t{t}")
                 for c in L4 for t in DEVICE_THREADS]


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.slow  # H2O/cc-pVDZ's launches: not a small input
@pytest.mark.parametrize(("cls", "threads"), DEVICE_PARAMS)
def test_cuda_f_lever_scatter_class_matches_the_host(
        cls, threads, request, run_device_binary, tmp_path) -> None:
    """The lever's launches and scatters on the RTX 3080, against the host."""
    _class("cuda", cls, threads, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.slow  # H2O/cc-pVDZ's launches: not a small input
@pytest.mark.parametrize(("cls", "threads"), DEVICE_PARAMS)
def test_hip_f_lever_scatter_class_matches_the_host(
        cls, threads, request, run_device_binary, tmp_path) -> None:
    """The lever's launches and scatters on the RX 9060 XT."""
    _class("hip", cls, threads, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# 3. On the card, per molecule
# ---------------------------------------------------------------------------


def _molecule(backend: str, name: str, tau: float, n: int, request,
              run_device_binary, tmp_path: Path) -> None:
    tc = tfo._toolchain(backend, request)
    recs = _molecule_recs(name, tau, n)
    kernels = _kernels(recs)
    binary = _build(backend, tc, kernels, tmp_path, f"fock_{name}")
    basis, _, libcint = tfo.molecule(name)
    Fs, _ = run_on_card(binary, fock_scatter_stage(recs, kernels),
                        basis.n_ao, tmp_path, run_device_binary,
                        backend=backend)
    got = np.array(KF.close(Fs[0]))
    w_int, _ = walk(name, tau, n)  # the literal interpreted walk
    scale = sweep(name, block="scale")
    w = {"interp": tfo._worst(got, np.array(w_int.F), scale)}
    assert tfo._within(got, np.array(w_int.F), scale, 2 * TOL), w
    if tau == 0.0:
        w["sweep"] = tfo._worst(got, sweep(name), scale)
        w["libcint"] = tfo._worst(got, libcint, scale)
        assert tfo._within(got, sweep(name), scale, TOL), w
        assert tfo._within(got, libcint, scale, LIBCINT_TOL), w
    print(f"\n{backend} {name} tau={tau} n={n} via {NAME}: {len(recs)} "
          f"launches of {len(kernels)} classes; worst / scale "
          + ", ".join(f"{k} {v:.2e}" for k, v in w.items())
          + f"; runs bit-identical {bool(np.array_equal(Fs[0], Fs[1]))}",
          flush=True)
    assert np.array_equal(Fs[0], Fs[1]), "F is not deterministic"


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_cuda_f_lever_scatter_molecule_matches_walk_and_libcint(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _molecule("cuda", name, tau, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_f_lever_scatter_molecule_matches_walk_and_libcint(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _molecule("hip", name, tau, n, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells, the staged input, the vectorised map, the host path
# ---------------------------------------------------------------------------


def test_the_device_cells_are_l_at_most_4_and_cover_it() -> None:
    assert all(sum(c) <= 4 for c in L4)
    assert len(L4) == sum(1 for c in CLASSES if sum(c) <= 4)
    assert all(t % WAVEFRONT == 0 for t in DEVICE_THREADS)


def test_the_relaunched_records_differ_only_in_kernel_and_block() -> None:
    cls = (0, 0, 0, 1)
    for a, b in zip(tfo._class_recs(cls), as_lever(tfo._class_recs(cls), 128),
                    strict=True):
        assert (a.grid, a.args) == (b.grid, b.args)
        assert b.threads == 128 and b.kernel.name == "f_0001_lv_scatter"


@pytest.mark.parametrize("cls", [(0, 0, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
                         ids=str)
def test_the_vectorised_map_is_the_builders(cls) -> None:
    for r in tfo._class_recs(cls):
        a = r.args
        m0 = FS.scatter_map(cls, a["n_ket"], a["bra_ao"], a["ket_ao"])
        m1 = scatter_map_np(cls, a["n_ket"], a["bra_ao"], a["ket_ao"])
        assert (m0.n_el, m0.n_slot) == (m1.n_el, m1.n_slot)
        for f in ("el", "ptr", "src"):
            assert tuple(getattr(m0, f)) == tuple(int(x) for x in getattr(m1, f))


def test_the_staged_input_carries_each_launchs_map() -> None:
    recs = _molecule_recs("h2o_sto3g", 0.0, 1)
    kernels = _kernels(recs)
    data = fock_scatter_stage(recs, kernels)
    nl = int(data[0])
    assert nl == len(recs)
    heads = data[4:4 + 11 * nl].reshape(nl, 11)
    for h, r in zip(heads, recs, strict=True):
        cls = tuple(int(c) for c in r.name[2:6])
        m = FS.scatter_map(cls, r.args["n_ket"], r.args["bra_ao"],
                           r.args["ket_ao"])
        assert (int(h[9]), int(h[10])) == (m.n_el, m.n_slot)
        nb, nk = int(h[1]), int(h[2])
        assert int(h[8]) == 4 * nb + 4 * nk + 3 * m.n_el + 1 + m.n_slot


@pytest.mark.slow  # interprets the class's cc-pVDZ launches twice (~40 s)
def test_the_host_scatter_path_is_the_literal_interpreted_class() -> None:
    """`interpret_scatter` over one class's relaunched records (the device
    test's host twin) against test_fock_ondevice.py's references."""
    cls = (0, 0, 0, 1)
    recs = as_lever(tfo._class_recs(cls), 32)
    basis, _, _ = tfo.molecule(tfo.PART1)
    F = np.array(KF.close(interpret_scatter(recs, basis.n_ao)))
    F_int, G, scale = tfo._class_reference(cls)
    assert tfo._within(F, G, scale, TOL)
    assert tfo._within(F, F_int, scale, 2 * TOL)
