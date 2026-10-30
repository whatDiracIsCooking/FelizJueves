"""The HGP overlap-gradient kernels emitted for both vendors: golden texts at
(ss) (sp) (pp) and a compile of each unit through (ff) by nvcc (sm_86) and
hipcc (gfx1200).

THE LADDER is `hgt_`'s, `python -m felizjueves.emit --hgp-grad-l [LMAX]`,
which writes `hgs_{la}{lb}` (`kernel_hgs.build_hgs_kernel`, `Target.S` at d1
on `Engine.HGP`) beside `hgt_{la}{lb}` for every class; its classes, CLI
path and refusals are test_emit_hgp_grad.py's, which asserts both units.

THE GOLDENS are `test/golden/{cuda,hip}/hgs_{00,01,11}.{cu,hip}`, and
nothing else under that prefix. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hgs.py --write

THE COMPILE. Each unit of the ladder through (ff) is compiled alone (`-c`)
with the kernel tier's own flags and FAILS on any warning located in the
generated unit (#92's filter, test_emit_hgp_one_e.py's `_compile`). Needs
a compiler, never a card: the compile cases are `cuda` / `hip`, none is
`gpu`; above (pp) they are `slow` by L (`la + lb + 1`, test_kernel_hgs.py's).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_emit_hgp_one_e import BACKEND, GOLDEN, _no_tier_env
from test_emit_hgp_one_e import _compile as _compile_unit
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.gradient.kernel_hgs import L_MAX, build_hgs_kernel, hgs_name
from felizjueves.ir import Raw, walk

__all__ = ["_no_tier_env"]  # the autouse fixture, imported to apply here

UNITS = [(0, 0), (0, 1), (1, 1)]
LADDER = emit_cli.hgp_grad_classes(L_MAX)


def _id(unit) -> str:
    return hgs_name(unit)


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_hgs_kernel(unit), arch)


def write_goldens() -> None:
    """Rewrite every HGP overlap-gradient golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for unit in UNITS:
            (GOLDEN / backend / f"{_id(unit)}.{ext}").write_text(
                _text(unit, backend))


# --- the units ----------------------------------------------------------------


def test_no_raw_escape_hatch() -> None:
    for unit in LADDER:
        k = build_hgs_kernel(unit)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_the_golden_is_unchanged(unit, backend: str) -> None:
    path = GOLDEN / backend / f"{_id(unit)}.{BACKEND[backend][2]}"
    assert _text(unit, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_bare_ladder() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"hgs_*.{ext}")}
        assert files == {_id(u) for u in UNITS}


@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(unit, "cuda"), _text(unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_atomic_adds_no_shared_no_boys() -> None:
    for backend in BACKEND:
        for unit in UNITS:
            text = _text(unit, backend)
            assert text.count("atomicAdd") == 6
            assert "__shared__" not in text and "fj_boys<" not in text


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, unit, tmp_path: Path) -> None:
    _compile_unit(tc, backend, unit, tmp_path, build=build_hgs_kernel)


_COMPILE = [by_l(u, sum(u) + 1, id=_id(u)) for u in LADDER]


@pytest.mark.cuda
@pytest.mark.parametrize("unit", _COMPILE)
def test_nvcc_compiles_the_unit(unit, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", unit, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", _COMPILE)
def test_hipcc_compiles_the_unit(unit, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", unit, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hgs.py --write")
    write_goldens()
