"""`E^ac` staged in shared memory (#342) through the device compilers and on the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): a set of K classes with L <= 4
   (milestone 13's cap), the literal kernel and the lever's kernel each its
   own translation unit, compiled with the tier's flags plus the measurement
   flag (`measured.Backend.command`: `-Xptxas -v` on nvcc, the AMD
   `kernel-resource-usage` remark on hipcc). A warning in generated text
   fails (#92's rule). The figures -- registers, spill stores and static
   shared bytes (nvcc's `smem`), VGPRs and spilled registers and LDS bytes
   (hipcc) -- literal against lever, and the DIRECTION of the change per
   compiler, are PRINTED (run with `-s`) as observations: compiler readings,
   never store rows, no threshold (#362 is the ledger). The static shared
   bytes ARE asserted: the literal kernel has none, and the lever's are at
   most `stage_eac.staged_bytes(am)`, the declared footprint, on both
   compilers. AT MOST, not equal: nvcc may drop staged entries nothing
   reads, and -- the IR having no barrier, so nothing orders one lane's
   stores before another's loads -- a compiler may forward a lane's own
   stores and promote the block back to registers (hipcc does at the
   smallest classes, LDS 0). The declared figure is the bound a launch
   must budget for; what was allocated is printed beside it.
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4,
   at one batch and, where the plan admits it, three, plus `(dd|ss)` (the
   largest `E^ac` block within the cap), with the lever, launched by
   test_kernel_device.py's K harness (`_run_k`: its inputs, its host staging
   `ao.k._k_quartet` on `reference.pq`, its pin `K_REL = 1e-12` of the
   quartet scale, unchanged), at the harness's full warp -- so the
   barrier-free, identical-value staging is exercised by every lane of a
   real warp. `measured.cell_kernels` passes a kernel-body K lever to
   `build_k_kernels` (#339), renaming each unit `<name>_lv_stage_eac`.
   Nothing is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2 --klever-stage-eac-k 0 1 2 3 4" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_stage_eac
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path

import pytest

from felizjueves import measured
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.ir import Kernel
from felizjueves.k_levers import stage_eac
from felizjueves.kernel import k_spec
from felizjueves.kernel_k import build_k_kernels
from felizjueves.spec import Operator, Spec, Target

NAME = stage_eac.NAME


def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# The compile set, (am, n_batches): the default K class of each L = 1..4,
# the classes with a d shell on the bra (the E^ac block is largest there),
# at one batch; and two classes at three batches, the atomized form.
COMPILE_CELLS = ([(k_spec(L).am, 1) for L in range(1, 5)]
                 + [((2, 0, 2, 0), 1), ((2, 1, 1, 0), 1), ((2, 2, 0, 0), 1),
                    ((1, 0, 1, 0), 3), ((1, 1, 1, 1), 3)])
COMPILE_PARAMS = [pytest.param(am, n, id=f"{_id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _spec(am, n_batches: int = 1, lever: bool = True) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K,
                (NAME,) if lever else (), n_batches)


# ---------------------------------------------------------------------------
# 1. Compile-only: registers, spill and shared, literal vs lever
# ---------------------------------------------------------------------------

EMIT = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}
WARNING = re.compile(r"^(?P<file>[^\s:]+):\d+:\d+: warning:", re.MULTILINE)


def _compile(backend: str, compiler: str, kern: Kernel, workdir: Path
             ) -> tuple[int, int, int]:
    """(registers or VGPRs, spill bytes or spilled registers, static shared
    bytes) of one unit."""
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
                raw.get("smem_bytes", 0))
    raw = measured.parse_amd_remarks(report, kern.name)
    return (raw["VGPRs"], raw["VGPRs Spill"] + raw.get("SGPRs Spill", 0),
            raw.get("LDS Size", 0))


def _direction(lit: int, lev: int) -> str:
    return "=" if lit == lev else ("down" if lev < lit else "up")


def _figures(backend: str, compiler: str, am, n: int, tmp_path: Path
             ) -> None:
    literal = build_k_kernels(_spec(am, n, lever=False))
    lever = build_k_kernels(_spec(am, n))
    assert [k.name for k in literal] == [k.name for k in lever]
    unit = ("registers/spill-store bytes/smem" if backend == "cuda"
            else "VGPRs/spilled registers/LDS")
    rows = []
    for a, b in zip(literal, lever, strict=True):
        lit = _compile(backend, compiler, a, tmp_path / "literal")
        lev = _compile(backend, compiler, b, tmp_path / "lever")
        # the literal kernel has no static shared (Ksh is dynamic, reported
        # as 0), and the declared footprint BOUNDS what the compiler
        # allocates for the staged block: it may drop entries nothing reads
        # (nvcc) or, with no barrier to stop it, promote the whole block
        # back to registers (hipcc, small classes) -- never exceed it
        assert lit[2] == 0, lit
        assert 0 <= lev[2] <= stage_eac.staged_bytes(tuple(am)), (lit, lev)
        rows.append(f"{a.name} {lit[0]}/{lit[1]}/{lit[2]} -> "
                    f"{lev[0]}/{lev[1]}/{lev[2]} of "
                    f"{stage_eac.staged_bytes(tuple(am))} declared (regs "
                    f"{_direction(lit[0], lev[0])}, spill "
                    f"{_direction(lit[1], lev[1])})")
    print(f"\n{backend} {_id(am)} b{n} ({unit}), literal -> {NAME}: "
          + "; ".join(rows), flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_stage_eac_kernels_compile_with_their_figures(am, n, nvcc,
                                                           tmp_path) -> None:
    _figures("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_stage_eac_kernels_compile_with_their_figures(am, n, hipcc,
                                                          tmp_path) -> None:
    _figures("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the lever's kernels against the host staging
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    for L in range(5):
        for b in (1, 3):
            spec = dataclasses.replace(k_spec(L, n_batches=b), levers=(NAME,))
            try:
                build_k_kernels(spec)
            except ValueError:
                continue  # L = 0 at 3 batches: an empty batch, refused
            out.append(pytest.param(spec, id=f"L{L}-b{b}-k-stage-eac"))
    out.append(pytest.param(_spec((2, 2, 0, 0)), id="dd-ss-b1-k-stage-eac"))
    return out


DEVICE_CELLS = _device_cells()


def test_the_driver_builds_a_stage_eac_cell_as_build_k_kernels() -> None:
    for spec in (k_spec(2), k_spec(4, n_batches=3)):
        spec = dataclasses.replace(spec, levers=(NAME,))
        want = build_k_kernels(spec)
        got = measured.cell_kernels(spec)
        assert [k.name for k in got] == [f"{k.name}_lv_stage_eac"
                                         for k in want]
        assert [dataclasses.replace(k, name=w.name)
                for k, w in zip(got, want, strict=True)] == list(want)
        # the harness sizes the dynamic Ksh only; the staged block is static
        assert ([measured.jsh_slots(k) for k in got]
                == [measured.jsh_slots(k) for k in build_k_kernels(
                    dataclasses.replace(spec, levers=()))])


def _run(backend: str, spec: Spec, request, compile_device_source,
         run_device_binary, tmp_path) -> None:
    import test_kernel_device as tkd

    tkd._run_k(backend, spec, request, compile_device_source,
               run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_stage_eac_matches_the_host(spec, request,
                                                 compile_device_source,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """The lever's kernels on the RTX 3080, against the host."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_stage_eac_matches_the_host(spec, request,
                                                compile_device_source,
                                                run_device_binary,
                                                tmp_path) -> None:
    """The lever's kernels on the RX 9060 XT, against the host."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [s.values[0].total_am for s in DEVICE_CELLS] == [0, 1, 1, 2, 2, 3,
                                                            3, 4, 4, 4]
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)
