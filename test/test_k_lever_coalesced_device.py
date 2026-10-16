"""The coalesced K write-back lever (#345): the device compilers and the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): the coalesced kernel of a set of
   K classes with L <= 4 (milestone 13's cap), each its own translation
   unit, compiled beside the class's literal kernel with the tier's flags
   plus the measurement flag (`measured.Backend.command`: `-Xptxas -v` on
   nvcc, the AMD `kernel-resource-usage` remark on hipcc). A warning in
   generated text fails (#92's rule) -- the lever spells no warp sync, so
   there is none for hipcc to refuse. The figures (registers and spill-store
   bytes on nvcc, VGPRs and spilled registers on hipcc, literal against
   lever) are PRINTED (run with `-s`) as observations: compiler readings,
   never store rows, no threshold (#362 is the ledger).
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4
   at one batch with the lever (a split class is refused by the lever,
   `coalesced.SPLIT_REFUSAL`), through test_kernel_device.py's K harness
   (`_run_k`: its inputs, its host staging, its pin `K_REL = 1e-12` of the
   quartet scale, unchanged), which builds the cell from
   `measured.cell_kernels` and so compiles the lever's kernel as
   `<name>_lv_coalesced`. Nothing is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_coalesced
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
from felizjueves.kernel import k_spec
from felizjueves.kernel_k import build_k_kernel, build_k_kernels
from felizjueves.spec import Operator, Spec, Target

COALESCED = "coalesced"


def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# The default K class of each L = 1..4, and two with a d shell (the largest
# K[a, b] blocks within the cap: 36 and 18 elements), one batch each.
COMPILE_AMS = ([k_spec(L).am for L in range(1, 5)]
               + [(2, 0, 2, 0), (2, 1, 1, 0)])
COMPILE_PARAMS = [pytest.param(am, id=_id(am)) for am in COMPILE_AMS]


def _lever(am) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K, (COALESCED,))


# ---------------------------------------------------------------------------
# 1. Compile-only: registers and spill, literal against the lever
# ---------------------------------------------------------------------------

EMIT = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}
WARNING = re.compile(r"^(?P<file>[^\s:]+):\d+:\d+: warning:", re.MULTILINE)


def _compile(backend: str, compiler: str, kern: Kernel, workdir: Path
             ) -> tuple[int, int]:
    """(registers or VGPRs, spill bytes or spilled registers) of one unit."""
    be = measured.BACKENDS[backend]
    emit, arch = EMIT[backend]
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
        return raw["registers"], raw["spill_store_bytes"]
    raw = measured.parse_amd_remarks(report, kern.name)
    return raw["VGPRs"], raw["VGPRs Spill"] + raw.get("SGPRs Spill", 0)


def _figures(backend: str, compiler: str, am, tmp_path: Path) -> None:
    literal = build_k_kernel(tuple(am))
    lever = dataclasses.replace(build_k_kernel(_lever(am)),
                                name=f"{literal.name}_lv_{COALESCED}")
    (lit_dir := tmp_path / "literal").mkdir()
    (lv_dir := tmp_path / "lever").mkdir()
    lit = _compile(backend, compiler, literal, lit_dir)
    lv = _compile(backend, compiler, lever, lv_dir)
    unit = ("registers/spill-store bytes" if backend == "cuda"
            else "VGPRs/spilled registers")
    print(f"\n{backend} {_id(am)} b1 ({unit}): literal {lit[0]}/{lit[1]}, "
          f"coalesced {lv[0]}/{lv[1]}", flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize("am", COMPILE_PARAMS)
def test_cuda_coalesced_kernels_compile_with_their_figures(am, nvcc,
                                                           tmp_path) -> None:
    _figures("cuda", nvcc.path, am, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("am", COMPILE_PARAMS)
def test_hip_coalesced_kernels_compile_with_their_figures(am, hipcc,
                                                          tmp_path) -> None:
    _figures("hip", hipcc.path, am, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the coalesced kernel against the host staging
# ---------------------------------------------------------------------------

DEVICE_CELLS = [pytest.param(dataclasses.replace(k_spec(L),
                                                 levers=(COALESCED,)),
                             id=f"L{L}-b1-k-coalesced") for L in range(5)]


def test_the_driver_builds_a_coalesced_cell_as_build_k_kernels() -> None:
    for L in (2, 4):
        spec = dataclasses.replace(k_spec(L), levers=(COALESCED,))
        (want,) = build_k_kernels(spec)
        (got,) = measured.cell_kernels(spec)
        assert got.name == f"{want.name}_lv_{COALESCED}"
        assert dataclasses.replace(got, name=want.name) == want


def _run(backend: str, spec: Spec, request, compile_device_source,
         run_device_binary, tmp_path) -> None:
    import test_kernel_device as tkd

    tkd._run_k(backend, spec, request, compile_device_source,
               run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_coalesced_matches_the_host(spec, request,
                                                 compile_device_source,
                                                 run_device_binary,
                                                 tmp_path) -> None:
    """The coalesced K kernel on the RTX 3080, against the host."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_coalesced_matches_the_host(spec, request,
                                                compile_device_source,
                                                run_device_binary,
                                                tmp_path) -> None:
    """The coalesced K kernel on the RX 9060 XT, against the host."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [s.values[0].total_am for s in DEVICE_CELLS] == [0, 1, 2, 3, 4]
    assert all(sum(am) <= 4 for am in COMPILE_AMS)
