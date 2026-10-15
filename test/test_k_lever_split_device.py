"""The per-element split K lever (#341) through the device compilers and on the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): every element kernel of a set of
   K classes with L <= 4 (milestone 13's cap), each its own translation
   unit, compiled with the tier's flags plus the measurement flag
   (`measured.Backend.command`: `-Xptxas -v` on nvcc, the AMD
   `kernel-resource-usage` remark on hipcc), beside the class's literal block
   kernel. A warning in generated text fails (#92's rule). The figures --
   registers and spill (nvcc), VGPRs and scratch spill (hipcc), per kernel,
   and the worst split kernel against the block kernel -- are PRINTED (run
   with `-s`) as observations: compiler readings, never store rows, no
   threshold (#362 is the ledger; the spilling classes are #341's X4's).
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4,
   at one batch and, where the plan admits it, three, with the split lever:
   the element kernels launched in order onto one zeroed `K` by
   test_kernel_device.py's K harness (`_run_k`: its inputs, its host
   staging `ao.k._k_quartet` on `reference.pq`, its pin `K_REL = 1e-12` of
   the quartet scale, unchanged). The harness builds a cell from
   `measured.cell_kernels`, which refuses a K lever (#205's store has no
   lever rows); here it is pointed at `build_k_kernels` for the test's
   duration, so the lever's kernels are what is compiled and run. Nothing
   is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_split
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from felizjueves import measured
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.ir import Kernel
from felizjueves.kernel import k_spec
from felizjueves.kernel_k import build_k_kernel, build_k_kernels
from felizjueves.spec import Operator, Spec, Target

SPLIT = "split"

def _id(am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


# The compile set, (am, n_batches): the default K class of each L = 1..4 and
# two classes with a d shell (the element count is largest within the cap),
# at one batch; and two classes at three batches, the atomized form.
COMPILE_CELLS = ([(k_spec(L).am, 1) for L in range(1, 5)]
                 + [((2, 0, 2, 0), 1), ((2, 1, 1, 0), 1),
                    ((1, 0, 1, 0), 3), ((1, 1, 1, 1), 3)])
COMPILE_PARAMS = [pytest.param(am, n, id=f"{_id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _lever(am, n_batches: int = 1) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K, (SPLIT,),
                n_batches)


# ---------------------------------------------------------------------------
# 1. Compile-only: registers and spill, literal block vs every element kernel
# ---------------------------------------------------------------------------

EMIT = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}
WARNING = re.compile(r"^(?P<file>[^\s:]+):\d+:\d+: warning:", re.MULTILINE)


def _compile(backend: str, compiler: str, kern: Kernel, workdir: Path
             ) -> tuple[int, int]:
    """(registers or VGPRs, spill bytes or scratch spill) of one unit."""
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


def _worst(names, got) -> str:
    i = max(range(len(got)), key=lambda i: (got[i][1], got[i][0]))
    return f"{names[i]} {got[i][0]}/{got[i][1]}"


def _figures(backend: str, compiler: str, am, n: int, tmp_path: Path
             ) -> None:
    literal = (build_k_kernel(tuple(am)),) if n == 1 else build_k_kernels(
        Spec(Operator.COULOMB, 0, tuple(am), Target.K, (), n))
    elements = build_k_kernels(_lever(am, n))
    kernels = (*literal, *elements)
    with ThreadPoolExecutor(max_workers=6) as pool:
        got = list(pool.map(lambda k: _compile(backend, compiler, k,
                                               tmp_path), kernels))
    lit, split = got[:len(literal)], got[len(literal):]
    unit = ("registers/spill-store bytes" if backend == "cuda"
            else "VGPRs/spilled registers")
    print(f"\n{backend} {_id(am)} b{n} ({unit}): literal worst "
          f"{_worst([k.name for k in literal], lit)}; {len(split)} element "
          f"kernels, worst {_worst([k.name for k in elements], split)}; per "
          "element " + " ".join(f"{r}/{s}" for r, s in split), flush=True)
    assert len(split) == len(elements)


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_split_kernels_compile_with_their_figures(am, n, nvcc,
                                                       tmp_path) -> None:
    _figures("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_split_kernels_compile_with_their_figures(am, n, hipcc,
                                                      tmp_path) -> None:
    _figures("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the element kernels against the host staging
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    for L in range(5):
        for b in (1, 3):
            spec = dataclasses.replace(k_spec(L, n_batches=b), levers=(SPLIT,))
            try:
                build_k_kernels(spec)
            except ValueError:
                continue  # L = 0 at 3 batches: an empty batch, refused
            out.append(pytest.param(spec, id=f"L{L}-b{b}-k-split"))
    return out


DEVICE_CELLS = _device_cells()


def _cell_kernels(spec: Spec) -> tuple[Kernel, ...]:
    if spec.target is Target.K and spec.levers == (SPLIT,):
        return build_k_kernels(spec)
    raise AssertionError(f"{spec.key()}: not a split cell")


def _run(backend: str, spec: Spec, request, compile_device_source,
         run_device_binary, tmp_path, monkeypatch) -> None:
    import test_kernel_device as tkd

    monkeypatch.setattr(measured, "cell_kernels", _cell_kernels)
    tkd._run_k(backend, spec, request, compile_device_source,
               run_device_binary, tmp_path)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_split_matches_the_host(spec, request,
                                             compile_device_source,
                                             run_device_binary, tmp_path,
                                             monkeypatch) -> None:
    """The split cell's element kernels on the RTX 3080, against the host."""
    _run("cuda", spec, request, compile_device_source, run_device_binary,
         tmp_path, monkeypatch)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_split_matches_the_host(spec, request,
                                            compile_device_source,
                                            run_device_binary, tmp_path,
                                            monkeypatch) -> None:
    """The split cell's element kernels on the RX 9060 XT, against the host."""
    _run("hip", spec, request, compile_device_source, run_device_binary,
         tmp_path, monkeypatch)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [s.values[0].total_am for s in DEVICE_CELLS] == [0, 1, 1, 2, 2, 3,
                                                            3, 4, 4]
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)
