"""The atomic-free F write-back (#358) through the compilers and on the cards.

FOUR PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): every F class with L <= 4
   (milestone 13's cap), the literal kernel, block-fab's (#351, H7) and the
   scatter lever's each its own translation unit, and the class-agnostic
   `f_scatter` unit once, compiled with the tier's flags plus the
   measurement flag (`measured.Backend.command`). A warning in generated
   text fails (#92). The figures -- registers, spill, static shared bytes
   -- are PRINTED (`-s`) as observations, never store rows; the lever's
   static shared bytes ARE asserted equal to block-fab's (the same two
   shared arrays). The scatter programs (per class and per molecule) are
   compiled too, so a spelling problem shows without the card.
2. ON THE CARD, PER CLASS (`gpu` + `cuda` / `hip`): every F class with
   L <= 4, its launches in H2O/cc-pVDZ's plan at `tau = 0`
   (test_fock_ondevice.py's `_class_recs`: the same packed records, `diag`,
   `Dsh` and `D`), each relaunched with the lever's kernel at 32 and 128
   threads and followed by the scatter over its map, held by
   test_fock_ondevice.py's own `_check_class` -- `TOL = 1e-13` of `scale_G`
   against the restricted sweep, `2 TOL` against the literal launches
   interpreted -- and, the lever's point, the two runs of one binary
   BIT-IDENTICAL (`F` is deterministic: no atomic anywhere).
3. ON THE CARD, PER MOLECULE: H2O/STO-3G at `tau = 0` and `tau = 0.1, n =
   4`, H2O/6-31G at `tau = 0` (every class L <= 4), every launch of the plan
   through the scatter path into one `F`, against walk_f's interpreted
   literal build at `2 TOL`, at `tau = 0` against the sweep at `TOL` and
   libcint at `LIBCINT_TOL`, the two runs bit-identical.
4. THE WRITE-BACK TAIL, MEASURED BEFORE THE VERDICT (the issue's order;
   opt-in, ALONE, behind the gate): per class, one synthetic off-diagonal
   launch (`fock_driver.synthetic_launch`, 2048 x 128 contracted quartets,
   the F rows' wide input) timed under four programs -- the literal kernel,
   block-fab's, the scatter lever's F kernel ALONE (`f_only`: W zeroed and
   written, no scatter) and the whole scatter path -- and the fractions
   printed: block-fab's atomic tail `(t_H7 - t_Fonly) / t_H7` (what plain
   stores save over its atomics, the zeroing of W charged to the stores),
   the scatter kernel's share of the path `(t_path - t_Fonly) / t_path`,
   and the net `t_path / t_H7 - 1`. Printed only, never a store row (#362
   is the ledger), no threshold:

    FJ_WALLCLOCK_F_SCATTER=1 [FJ_WALLCLOCK_F_L="2 3 4"] \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- \\
        -k f_lever_scatter_tail -s

Run parts 2 and 3 as (each backend; nothing above L = 4 and nothing `slow`
is selected; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- \\
        -k "f_lever_scatter and not tail"
"""

from __future__ import annotations

import dataclasses
import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_fock_ondevice as tfo
from fock_driver import (
    FRec,
    fock_program,
    fock_scatter_program,
    fock_scatter_stage,
    fock_stage,
    interpret_scatter,
    max_boys_T,
    reps,
    run_on_card,
    scatter_map_np,
    synthetic_launch,
)
from plan_driver import compile_checked
from test_f_lever_block_fab_device import _compile
from test_fock_build import LIBCINT_TOL, TOL, sweep, walk
from test_kernel_device import (
    BACKEND,
    GPU_BUSY_MAX,
    REPEATS,
    STUDY_RETAKES,
    WARMUP,
    check_vram,
)

from felizjueves import gate
from felizjueves import kernel_f as KF
from felizjueves import kernel_f_scatter as FS
from felizjueves.f_levers import block_fab, scatter
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.lever_ladder import unit_name

NAME = scatter.NAME
L4 = tuple(c for c in CLASSES if sum(c) <= 4)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in L4]
DEVICE_THREADS = (32, 128)
WAVEFRONT = 32  # both cards' (device.SM_86, device.GFX1200)
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, id="h2o_sto3g-tau0"),
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0"),
]
TAIL = "FJ_WALLCLOCK_F_SCATTER"


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
# 1. Compile-only: the figures, and the programs
# ---------------------------------------------------------------------------


def _figures(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    lit = _compile(backend, compiler, KF.build_f_kernel(cls),
                   tmp_path / "literal")
    h7 = _compile(backend, compiler, _named((block_fab.NAME,), cls),
                  tmp_path / "block_fab")
    lev = _compile(backend, compiler, lever(cls), tmp_path / "scatter")
    assert lev[2] == h7[2], (h7, lev)  # ab_sh + fab_sh, both
    unit = ("registers/spill-store bytes/smem" if backend == "cuda"
            else "VGPRs/spilled registers/LDS")
    print(f"\n{backend} {cls} ({unit}): literal {lit[0]}/{lit[1]}/{lit[2]}, "
          f"block-fab {h7[0]}/{h7[1]}/{h7[2]} -> {NAME} "
          f"{lev[0]}/{lev[1]}/{lev[2]}; W slots per bra "
          f"{scatter.n_ab(cls)} + {scatter.ket_width(cls)} n_ket",
          flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_scatter_kernels_compile_with_their_figures(cls, nvcc,
                                                         tmp_path) -> None:
    _figures("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_scatter_kernels_compile_with_their_figures(cls, hipcc,
                                                        tmp_path) -> None:
    _figures("hip", hipcc.path, cls, tmp_path)


def _scatter_unit(backend: str, compiler: str, tmp_path: Path) -> None:
    f = _compile(backend, compiler, FS.build_scatter_kernel(), tmp_path)
    print(f"\n{backend} {FS.NAME}: {f[0]}/{f[1]}/{f[2]}", flush=True)
    assert f[1] == 0  # the scatter spills nothing


@pytest.mark.cuda
def test_cuda_the_scatter_unit_compiles(nvcc, tmp_path) -> None:
    _scatter_unit("cuda", nvcc.path, tmp_path)


@pytest.mark.hip
def test_hip_the_scatter_unit_compiles(hipcc, tmp_path) -> None:
    _scatter_unit("hip", hipcc.path, tmp_path)


@cache
def _molecule_recs(name: str, tau: float, n: int) -> tuple[FRec, ...]:
    recs, _ = tfo._plan(name, tau, n)
    return as_lever(recs)


PROGRAM_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}")
      for c in L4),
    *(pytest.param(("molecule", *p.values), id=p.id) for p in MOLECULES),
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
@pytest.mark.parametrize(("cls", "threads"), DEVICE_PARAMS)
def test_cuda_f_lever_scatter_class_matches_the_host(
        cls, threads, request, run_device_binary, tmp_path) -> None:
    """The lever's launches and scatters on the RTX 3080, against the host."""
    _class("cuda", cls, threads, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
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
# 4. The write-back tail, measured (opt-in, alone)
# ---------------------------------------------------------------------------


def _tail_classes(env=os.environ) -> tuple:
    text = env.get(tfo.WALLCLOCK_F_L, "").strip()
    Ls = {int(x) for x in text.split()} if text else {0, 1, 2, 3, 4}
    got = tuple(c for c in L4 if sum(c) in Ls)
    assert got, f"{tfo.WALLCLOCK_F_L}={text!r} selects no class with L <= 4"
    return got


def _timed(backend: str, binary: Path, data: np.ndarray, n_ao: int,
           label: str, run_device_binary, tmp_path: Path):
    work = tmp_path / "timing"
    work.mkdir(exist_ok=True)
    src = work / "timing.in"
    data.tofile(src)
    check_vram(binary, (str(src), str(work / "timing.out")), backend)
    for take in range(1 + STUDY_RETAKES):
        try:
            gate.wait_quiet(backend, say=lambda line: print(line, flush=True))
        except gate.Contended as e:
            print(f"  DISCARDED {label}: {e} (take {take + 1})", flush=True)
            continue
        Fs, out = run_on_card(binary, data, n_ao, work, run_device_binary,
                              backend=backend, timing=(WARMUP, REPEATS),
                              tag="timing")
        after = gate.after_reading(backend)
        if after is not None and after <= GPU_BUSY_MAX:
            break
        print(f"  DISCARDED {label}: card {after}% after (take {take + 1})",
              flush=True)
    else:
        pytest.fail(f"{label}: contended through {STUDY_RETAKES + 1} takes")
    wall, events = reps(out)
    assert len(events) == REPEATS and np.all(np.isfinite(Fs)), label
    return float(np.median(events)), float(np.median(wall)), Fs


def _tail(backend: str, request, run_device_binary, tmp_path: Path) -> None:
    if not os.environ.get(TAIL, "").strip():
        pytest.skip(f"the write-back tail is opt-in and must run alone, "
                    f"uncontended: set {TAIL}=1 (see the module docstring)")
    tc = tfo._toolchain(backend, request)
    for cls in _tail_classes():
        rec, basis = synthetic_launch(cls, side=tfo.WIDE_SIDE,
                                      seed=358_000 + sum(cls))
        assert max_boys_T(basis, tfo.WIDE_SIDE) <= tfo.BOYS_T_WITNESSED
        work = tmp_path / "".join(map(str, cls))
        arms = {}
        for arm, levs in (("literal", ()), ("block-fab", (block_fab.NAME,))):
            k = _named(levs, cls)
            r = FRec(k, rec.grid, rec.threads, rec.args)
            prog = fock_program([k], backend, work / arm, name=f"prog_{arm}")
            binary = compile_checked(tc, prog, work / arm / "bin")
            arms[arm] = (binary, fock_stage([r], [k]))
        (r,) = as_lever([rec])
        data = fock_scatter_stage([r], [r.kernel], mapper=scatter_map_np)
        for arm, f_only in (("f-only", True), ("path", False)):
            binary = _build(backend, tc, [r.kernel], work / arm, arm,
                            f_only=f_only)
            arms[arm] = (binary, data)
        t, bitwise = {}, None
        for arm, (binary, staged) in arms.items():
            ev, wall, Fs = _timed(backend, binary, staged, basis.n_ao,
                                  f"{cls} {arm}", run_device_binary, work)
            t[arm] = ev
            if arm == "path":
                bitwise = bool(np.array_equal(Fs[0], Fs[1]))
        print(f"\n{backend} {BACKEND[backend][1].name} {cls} tail, 2048 x 128 "
              f"wide, median events ms: " + ", ".join(
                  f"{a} {v / 1e6:.3f}" for a, v in t.items())
              + f"; block-fab's atomic tail (t_H7 - t_Fonly) / t_H7 "
              f"{(t['block-fab'] - t['f-only']) / t['block-fab']:+.4f}; "
              f"scatter share {(t['path'] - t['f-only']) / t['path']:.4f}; "
              f"net vs H7 {t['path'] / t['block-fab'] - 1:+.4f}; "
              f"path bit-identical runs {bitwise}", flush=True)
        assert bitwise, f"{cls}: the scatter path is not deterministic"


@pytest.mark.cuda
@pytest.mark.gpu
def test_cuda_f_lever_scatter_tail(request, run_device_binary,
                                   tmp_path) -> None:
    """The write-back tail on the RTX 3080 (opt-in, alone)."""
    _tail("cuda", request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
def test_hip_f_lever_scatter_tail(request, run_device_binary,
                                  tmp_path) -> None:
    """The write-back tail on the RX 9060 XT (opt-in, alone)."""
    _tail("hip", request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells, the staged input, the vectorised map, the host path
# ---------------------------------------------------------------------------


def test_the_device_cells_are_l_at_most_4_and_cover_it() -> None:
    assert all(sum(c) <= 4 for c in L4)
    assert len(L4) == sum(1 for c in CLASSES if sum(c) <= 4)
    assert all(t % WAVEFRONT == 0 for t in DEVICE_THREADS)
    assert all(sum(c) <= 4 for c in _tail_classes({}))
    with pytest.raises(AssertionError, match="selects no class"):
        _tail_classes({tfo.WALLCLOCK_F_L: "5"})


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
