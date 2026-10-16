"""The ETR lever (#354) through the device compilers and on the cards.

TWO PARTS, every F class with L = la + lb + lc + ld <= 4 (milestone 13's
cap; 13 of the 21):

1. COMPILE-ONLY (`cuda` / `hip`, no card): BOTH RECURRENCES ON EVERY CLASS
   -- the literal kernel (the ket cascade) and the ETR forced
   (`etr.forced(etr.ETR)`), each its own translation unit -- compiled with
   the tier's flags plus the measurement flag (`measured.Backend.command`:
   `-Xptxas -v` on nvcc, the AMD `kernel-resource-usage` remark on hipcc).
   A warning in generated text fails (#92's rule). The figures (registers,
   spill-store bytes, stack frame on nvcc; VGPRs, spilled VGPRs, scratch
   bytes on hipcc), cascade against ETR, are PRINTED (run with `-s`) beside
   the model's choice and margins (`etr.score`), and each line says whether
   the compiler's direction ((registers, spill, frame) lexicographically:
   the spill decides at the register cap) AGREES with the model -- the
   issue's "the disagreement is measured rather than assumed". Compiler
   readings, never store rows, no threshold (#362 is the ledger). The
   harness programs part 2 runs are compiled too.
2. ON THE CARD (`gpu` + `cuda` / `hip`): test_fock_ondevice.py's launches
   (`fock_driver.record`: launch_f's plan walked by walk_f), every kernel
   swapped:
   (a) per class, H2O/cc-pVDZ's launches of that class at `tau = 0`, the
   kernel the ETR FORCED (so the ETR runs on all 13 classes, the cascade
   ones included), against those launches interpreted with the literal
   kernel and `ao.fock`'s restricted sweep, with test_fock_ondevice's
   `_check_class` and its stated tolerances unchanged (`TOL` against the
   sweep; `2 TOL` against the literal interpreted launches -- the ETR and
   the cascade each within `TOL` of the sweep, test_f_lever_etr.py -- and
   between runs);
   (b) per molecule, the SHIPPED lever (`etr`, the rule's choice per class)
   on H2O/STO-3G at `tau = 0` and `tau = 0.1, n = 4` and H2O/6-31G at
   `tau = 0` (s and p shells only: every class L <= 4), every launch of
   the plan, against walk_f's interpreted literal build (`2 TOL`) and, at
   `tau = 0`, the sweep (`TOL`) and libcint (`LIBCINT_TOL`), by
   test_fock_ondevice's checks. Nothing is recorded.

Run (each backend; no F unit is on KERNEL_GEN_CMD's ladder, the driver
emits its own, so the generator is only capped):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- \\
        -s -k f_lever_etr

which selects parts 1 and 2 and nothing `slow` or above L = 4.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import test_fock_ondevice as fo
from fock_driver import fock_stage, run_on_card
from test_fock_build import LIBCINT_TOL, TOL, molecule, sweep, walk
from test_kernel_device import BACKEND, _toolchain

from felizjueves import kernel_f as KF
from felizjueves import measured
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import etr as E
from felizjueves.f_levers.seams import DEFAULT_HOOKS
from felizjueves.hgp.vrr import CLASSES
from felizjueves.ir import Kernel
from felizjueves.lever_ladder import unit_name

NAME = E.NAME
CAP = 4
CAPPED = tuple(c for c in CLASSES if sum(c) <= CAP)
CLASS_PARAMS = [pytest.param(c, id="".join(map(str, c))) for c in CAPPED]
MOLECULES = [
    pytest.param("h2o_sto3g", 0.0, 1, id="h2o_sto3g-tau0"),
    pytest.param("h2o_sto3g", 0.1, 4, id="h2o_sto3g-tau0.1"),
    pytest.param("h2o_631g", 0.0, 1, id="h2o_631g-tau0"),
]
_UNIT = re.compile(r"f_(\d)(\d)(\d)(\d)(_lv_etr)?(_forced)?")


def _literal_name(cls) -> str:
    return "f_" + "".join(map(str, cls))


@cache
def lever_kernel(cls) -> Kernel:
    """The SHIPPED lever's kernel of `cls` (the rule's recurrence), named as
    the CLI names its unit."""
    cls = tuple(cls)
    return KF.build_f_kernel(KF.f_class_spec(*cls, (NAME,)), name=unit_name(
        _literal_name(cls), (NAME,)))


@cache
def forced_kernel(cls) -> Kernel:
    """The ETR forced on `cls`: the shipped unit where the rule picks the
    ETR, else its own `_forced` unit."""
    cls = tuple(cls)
    if E.choice(cls) == E.ETR:
        return lever_kernel(cls)
    spec = KF.f_class_spec(*cls, (NAME,))
    return KF.build_f_kernel(spec, name=unit_name(
        _literal_name(cls), (NAME,)) + "_forced",
        hooks=E.forced(E.ETR)(DEFAULT_HOOKS, spec))


def _cls_of(kernel: Kernel) -> tuple[int, ...]:
    m = _UNIT.fullmatch(kernel.name)
    assert m, kernel.name
    return tuple(int(x) for x in m.groups()[:4])


def _swap(recs, which=lever_kernel) -> tuple:
    """`recs` with every literal kernel replaced by `which`'s."""
    return tuple(dataclasses.replace(r, kernel=which(_cls_of(r.kernel)))
                 for r in recs)


# ---------------------------------------------------------------------------
# 1. Compile-only: both recurrences on every class, against the model
# ---------------------------------------------------------------------------

EMIT = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}
WARNING = re.compile(r"^(?P<file>[^\s:]+):\d+:\d+: warning:", re.MULTILINE)


def _compile(backend: str, compiler: str, kern: Kernel, workdir: Path
             ) -> tuple[int, int, int]:
    """(registers or VGPRs, spill-store bytes or spilled VGPRs, stack frame
    or scratch bytes) of one unit."""
    be = measured.BACKENDS[backend]
    emit, arch = EMIT[backend]
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / f"{kern.name}.{be.ext}"
    obj = workdir / f"{kern.name}.o"
    src.write_text(emit(kern, arch))
    done = subprocess.run(be.command(compiler, src, obj), capture_output=True,
                          text=True, timeout=1800)
    report = done.stdout + done.stderr
    assert done.returncode == 0, f"{kern.name}:\n{report[-4000:]}"
    warned = [m[0] for m in WARNING.finditer(report)
              if Path(m["file"]).name == src.name]
    assert not warned, f"{kern.name}: warning in generated text (#92): {warned}"
    if backend == "cuda":
        raw = measured.parse_ptxas(report, kern.name)
        return (raw["registers"], raw["spill_store_bytes"],
                raw["stack_frame_bytes"])
    raw = measured.parse_amd_remarks(report, kern.name)
    return (raw["VGPRs"], raw["VGPRs Spill"],
            raw.get("ScratchSize", raw.get("Private Segment Size", -1)))


def _verdict(cascade: tuple, etr: tuple) -> str:
    """The compiler's opinion: (registers, spill, frame) lexicographically
    -- the spill decides once both sit at the register cap."""
    return "tie" if cascade == etr else (E.ETR if etr < cascade
                                         else E.CASCADE)


def _figures(backend: str, compiler: str, cls, tmp_path: Path) -> None:
    lit = _compile(backend, compiler, KF.build_f_kernel(cls),
                   tmp_path / "cascade")
    lev = _compile(backend, compiler, forced_kernel(cls), tmp_path / "etr")
    s = E.score(cls)
    unit = ("registers/spill-store bytes/stack frame" if backend == "cuda"
            else "VGPRs/spilled VGPRs/scratch bytes")
    by_regs = _verdict(lit, lev)
    agree = "tie" if by_regs == "tie" else (
        "agrees" if by_regs == E.choice(cls) else "DISAGREES")
    print(f"\n{backend} {cls} ({unit}) cascade {lit[0]}/{lit[1]}/{lit[2]} "
          f"etr {lev[0]}/{lev[1]}/{lev[2]}; model: choice {E.choice(cls)}, "
          f"peak {s.cascade_peak}->{s.etr_peak}, x {s.cascade_x}->{s.etr_x}; "
          f"the compiler's (regs, spill, frame) favour {by_regs} ({agree} "
          "with the model)",
          flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_etr_compiles_both_recurrences(cls, nvcc,
                                                    tmp_path) -> None:
    _figures("cuda", nvcc.path, cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_etr_compiles_both_recurrences(cls, hipcc,
                                                   tmp_path) -> None:
    _figures("hip", hipcc.path, cls, tmp_path)


def _program_kernels(cell) -> tuple[list[Kernel], str]:
    if cell[0] == "class":
        return [forced_kernel(cell[1])], fo._class_name(cell[1]) + "_etr"
    recs = _swap(fo._plan(*cell[1:])[0])
    return fo._kernels(recs), f"fock_{cell[1]}_etr"


PROGRAM_CELLS = [
    *(pytest.param(("class", c), id=f"class-{''.join(map(str, c))}")
      for c in CAPPED),
    *(pytest.param(("molecule", *p.values), id=p.id) for p in MOLECULES),
]


@pytest.mark.cuda
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_cuda_f_lever_etr_programs_compile(cell, nvcc, tmp_path) -> None:
    kernels, name = _program_kernels(cell)
    fo._build("cuda", nvcc, kernels, tmp_path, name)


@pytest.mark.hip
@pytest.mark.parametrize("cell", PROGRAM_CELLS)
def test_hip_f_lever_etr_programs_compile(cell, hipcc, tmp_path) -> None:
    kernels, name = _program_kernels(cell)
    fo._build("hip", hipcc, kernels, tmp_path, name)


# ---------------------------------------------------------------------------
# 2. On the card
# ---------------------------------------------------------------------------


def _per_class(backend: str, cls, request, run_device_binary,
               tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs = _swap(fo._class_recs(cls), forced_kernel)
    kernels = [recs[0].kernel]
    binary = fo._build(backend, tc, kernels, tmp_path, kernels[0].name)
    basis, _, _ = molecule(fo.PART1)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    w = fo._check_class(backend, cls, Fs)  # the literal's references
    print(f"\n{backend} {BACKEND[backend][1].name} {cls} ETR (forced; "
          f"shipped choice {E.choice(cls)}) on {fo.PART1}: {len(recs)} "
          f"launches; worst |F - sweep| / scale {w['sweep']:.2e} (tol "
          f"{TOL:g}), |F - literal interp| / scale {w['interp']:.2e} (tol "
          f"{2 * TOL:g}), runs {w['runs']:.2e}, bit-identical "
          f"{bool(w['bitwise_runs'])}")


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_cuda_f_lever_etr_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("cuda", cls, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_hip_f_lever_etr_each_class_matches_the_host(
        cls, request, run_device_binary, tmp_path) -> None:
    _per_class("hip", cls, request, run_device_binary, tmp_path)


def _per_molecule(backend: str, name: str, tau: float, n: int, request,
                  run_device_binary, tmp_path: Path) -> None:
    tc = _toolchain(backend, request)
    recs = _swap(fo._plan(name, tau, n)[0])
    kernels = fo._kernels(recs)
    binary = fo._build(backend, tc, kernels, tmp_path, f"fock_{name}_etr")
    basis, _, libcint = molecule(name)
    Fs, _ = run_on_card(binary, fock_stage(recs, kernels), basis.n_ao,
                        tmp_path, run_device_binary, backend=backend)
    got = [np.array(KF.close(F)) for F in Fs]
    w_int, _ = walk(name, tau, n)  # the literal kernels, interpreted
    F_int = np.array(w_int.F)
    scale = sweep(name, block="scale")
    w = {"interp": fo._worst(got[0], F_int, scale),
         "runs": fo._worst(got[0], got[1], scale)}
    assert fo._within(got[0], F_int, scale, 2 * TOL), (backend, name, tau, w)
    assert fo._within(got[1], got[0], scale, 2 * TOL), (backend, name, tau, w)
    if tau == 0.0:
        w["sweep"] = fo._worst(got[0], sweep(name), scale)
        w["libcint"] = fo._worst(got[0], libcint, scale)
        assert fo._within(got[0], sweep(name), scale, TOL), (backend, name, w)
        assert fo._within(got[0], libcint, scale, LIBCINT_TOL), (backend,
                                                                 name, w)
    print(f"\n{backend} {BACKEND[backend][1].name} {name} tau={tau} n={n} "
          f"{NAME}: {len(recs)} launches of {[k.name for k in kernels]}; "
          "worst / scale " + ", ".join(f"{k} {v:.2e}" for k, v in w.items()))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_cuda_f_lever_etr_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("cuda", name, tau, n, request, run_device_binary, tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tau", "n"), MOLECULES)
def test_hip_f_lever_etr_whole_molecule_matches(
        name, tau, n, request, run_device_binary, tmp_path) -> None:
    _per_molecule("hip", name, tau, n, request, run_device_binary, tmp_path)


# ---------------------------------------------------------------------------
# Fast tier: the cells stay under the cap; the swaps are the lever's kernels
# ---------------------------------------------------------------------------


def test_the_cells_are_l_at_most_4_and_the_swaps_are_the_levers() -> None:
    assert len(CAPPED) == 13 and all(sum(c) <= CAP for c in CAPPED)
    for p in MOLECULES:
        recs = _swap(fo._plan(*p.values)[0])
        assert {sum(_cls_of(r.kernel)) for r in recs} <= set(range(CAP + 1))
        assert all(r.kernel.name.endswith("_lv_etr") for r in recs)
        # H2O/STO-3G and 6-31G reach both outcomes of the rule
        assert {E.choice(_cls_of(r.kernel)) for r in recs} == {E.ETR,
                                                               E.CASCADE}
    k = lever_kernel((0, 1, 0, 1))
    lit = KF.build_f_kernel((0, 1, 0, 1))
    assert k.params == lit.params and k.name == "f_0101_lv_etr"
    assert dataclasses.replace(k, name=lit.name) == KF.build_f_kernel(
        KF.f_class_spec(0, 1, 0, 1, (NAME,)))
    for c in CAPPED:
        f = forced_kernel(c)
        assert f.name == ("f_" + "".join(map(str, c)) + "_lv_etr"
                          + ("" if E.choice(c) == E.ETR else "_forced"))
        assert f.params == KF.build_f_kernel(c).params
        assert _cls_of(f) == c
