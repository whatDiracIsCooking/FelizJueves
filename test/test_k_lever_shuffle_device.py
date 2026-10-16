"""The register K write-back lever (#349): the device compilers and the cards.

TWO PARTS, the issue's acceptance on the device side:

1. COMPILE-ONLY (`cuda` / `hip`, no card): the lever's kernel of a set of K
   classes with L <= 4 (milestone 13's cap), each its own translation unit,
   compiled beside the class's literal kernel AND its #345 `coalesced`
   kernel with the tier's flags plus the measurement flag
   (`measured.Backend.command`). A warning in generated text fails (#92).
   The figures (registers and spill-store bytes on nvcc, VGPRs and spilled
   registers on hipcc; literal, coalesced, shuffle) are PRINTED (run with
   `-s`) as observations: compiler readings, never store rows, no threshold
   (#362 is the ledger). `(d s|d s)` (36 elements) is the INELIGIBLE class
   at wave 32 on both cards: its shuffle kernel is the coalesced one.
2. ON THE CARD (`gpu` + `cuda` / `hip`): `k_spec(L)`'s class for L = 0..4
   at one batch and, where the plan admits it, three, plus the ineligible
   `(d s|d s)` at one and three batches, with the lever, through
   test_kernel_device.py's K harness (`_run_k`, its pin `K_REL = 1e-12`,
   unchanged), which compiles the lever's kernel as `<name>_lv_shuffle`.
   Nothing is recorded.

Run part 2 as (each backend; the generator capped at L = 4):

    KERNEL_GEN_CMD="env PYTHONPATH=src python -m felizjueves.emit \\
        --L 0 1 2 3 4 --pair-l 2 --one-e-l 2" \\
        devtools/kernel-tier.sh --backend cuda|hip -j 6 -- -k k_lever_shuffle
"""

from __future__ import annotations

import dataclasses

import pytest
import test_k_lever_coalesced_device as cd

from felizjueves import measured
from felizjueves.kernel import k_spec
from felizjueves.kernel_k import build_k_kernels
from felizjueves.spec import Operator, Spec, Target

SHUFFLE = "shuffle"
COALESCED = "coalesced"
WIDE = (2, 0, 2, 0)

COMPILE_CELLS = cd.COMPILE_CELLS + [(WIDE, 3)]
COMPILE_PARAMS = [pytest.param(am, n, id=f"{cd._id(am)}-b{n}")
                  for am, n in COMPILE_CELLS]


def _spec(am, n_batches: int, lever: str | None) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K,
                (lever,) if lever else (), n_batches)


# ---------------------------------------------------------------------------
# 1. Compile-only: registers and spill, literal / coalesced / shuffle
# ---------------------------------------------------------------------------


def _figures(backend: str, compiler: str, am, n: int, tmp_path) -> None:
    arms = {arm: build_k_kernels(_spec(am, n, lever))
            for arm, lever in (("literal", None), (COALESCED, COALESCED),
                               (SHUFFLE, SHUFFLE))}
    names = [k.name for k in arms["literal"]]
    assert all([k.name for k in ks] == names for ks in arms.values())
    unit = ("registers/spill-store bytes" if backend == "cuda"
            else "VGPRs/spilled registers")
    got: dict[str, list] = {}
    for arm, kernels in arms.items():
        (d := tmp_path / arm).mkdir()
        got[arm] = [cd._compile(backend, compiler, k if arm == "literal" else
                                dataclasses.replace(k, name=f"{k.name}_lv_{arm}"),
                                d) for k in kernels]
    for i, name in enumerate(names):
        print(f"\n{backend} {name} b{n} ({unit}): " + ", ".join(
            f"{arm} {got[arm][i][0]}/{got[arm][i][1]}" for arm in arms),
            flush=True)


@pytest.mark.cuda
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_cuda_shuffle_kernels_compile_with_their_figures(am, n, nvcc,
                                                         tmp_path) -> None:
    _figures("cuda", nvcc.path, am, n, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize(("am", "n"), COMPILE_PARAMS)
def test_hip_shuffle_kernels_compile_with_their_figures(am, n, hipcc,
                                                        tmp_path) -> None:
    _figures("hip", hipcc.path, am, n, tmp_path)


# ---------------------------------------------------------------------------
# 2. On the card: the lever's kernels against the host staging
# ---------------------------------------------------------------------------


def _device_cells() -> list:
    out = []
    specs = [(f"L{L}-b{b}", k_spec(L, n_batches=b))
             for L in range(5) for b in (1, 3)]
    # Only k_spec(L)'s classes on the card: the shared K harness
    # (`_k_check`) builds its inputs from `total_am` alone, as k_spec(L)'s
    # class, so WIDE (ds|ds) at L = 4 gets a wrong-length input and the
    # harness refuses it ("size mismatch") before any launch (as at #342).
    # WIDE's fallback path stays covered interpreted and compile-only.
    for label, spec in specs:
        spec = dataclasses.replace(spec, levers=(SHUFFLE,))
        try:
            build_k_kernels(spec)
        except ValueError:
            continue  # L = 0 at 3 batches: an empty batch, refused
        out.append(pytest.param(spec, id=f"{label}-k-shuffle"))
    return out


DEVICE_CELLS = _device_cells()


def test_the_driver_builds_a_shuffle_cell_as_build_k_kernels() -> None:
    for spec in (k_spec(2), k_spec(4, n_batches=3), _spec(WIDE, 1, None)):
        spec = dataclasses.replace(spec, levers=(SHUFFLE,))
        want = build_k_kernels(spec)
        got = measured.cell_kernels(spec)
        assert [k.name for k in got] == [f"{k.name}_lv_{SHUFFLE}"
                                         for k in want]
        assert [dataclasses.replace(k, name=w.name)
                for k, w in zip(got, want, strict=True)] == list(want)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_cuda_k_lever_shuffle_matches_the_host(spec, request,
                                               compile_device_source,
                                               run_device_binary,
                                               tmp_path) -> None:
    """The shuffle K kernel on the RTX 3080, against the host."""
    cd._run("cuda", spec, request, compile_device_source, run_device_binary,
            tmp_path)


@pytest.mark.hip
@pytest.mark.gpu
@pytest.mark.parametrize("spec", DEVICE_CELLS)
def test_hip_k_lever_shuffle_matches_the_host(spec, request,
                                              compile_device_source,
                                              run_device_binary,
                                              tmp_path) -> None:
    """The shuffle K kernel on the RX 9060 XT, against the host."""
    cd._run("hip", spec, request, compile_device_source, run_device_binary,
            tmp_path)


def test_the_device_cells_are_l_at_most_4() -> None:
    assert [(s.values[0].am, s.values[0].n_batches)
            for s in DEVICE_CELLS] == [
        ((0, 0, 0, 0), 1), ((1, 0, 0, 0), 1), ((1, 0, 0, 0), 3),
        ((1, 0, 1, 0), 1), ((1, 0, 1, 0), 3), ((1, 1, 1, 0), 1),
        ((1, 1, 1, 0), 3), ((1, 1, 1, 1), 1), ((1, 1, 1, 1), 3)]
    assert all(sum(am) <= 4 for am, _ in COMPILE_CELLS)
