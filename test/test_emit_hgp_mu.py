"""The HGP dipole and quadrupole kernels emitted for both vendors: golden
texts at (ss) through (dd), the `--hgp-mu-l` ladder, and a compile of every
class through (ff) by nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hgp-mu-l [LMAX]` writes `hmu_` and
`hquad_{la}{lb}` (`kernel_hgp.build_hgp_kernel`, `Target.DIPOLE` and
`QUADRUPOLE` on `Engine.HGP`) for every `la <= lb <= LMAX` -- 2 when the flag
is bare. It accepts 0..3 ONLY (the two moments are built and checked through
(ff)), exits 2 above that, is NEVER in the bare default and is not in
KERNEL_GEN_CMD. Its `hmu_` units through (pp) are `--hgp-one-e-l`'s.

THE GOLDENS are `test/golden/{cuda,hip}/{hmu,hquad}_{00,01,02,11,12,22}
.{cu,hip}`, and nothing else under those prefixes. Regenerate (and review
the diff) with

    PYTHONPATH=src uv run python test/test_emit_hgp_mu.py --write

THE COMPILE. Each unit through (ff) is compiled alone (`-c`), test_emit_hgp_
one_e.py's `_compile`: the kernel tier's flags, FAILING on any warning
located in the generated unit (#92's filter). Needs a compiler, never a card.
"""

from __future__ import annotations

import sys

import pytest
from test_emit_hgp_one_e import BACKEND, GOLDEN, _compile, _names, _no_tier_env, _run
from tiers import split_by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.ir import Raw, walk
from felizjueves.one_electron.kernel_hgp import MOMENTS, build_hgp_kernel, hgp_name

__all__ = ["_no_tier_env"]  # the autouse fixture, re-exported to this module

CLASSES = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
UNITS = [(t, la, lb) for la, lb in CLASSES for t in MOMENTS]
ALL = [(t, la, lb) for la, lb in emit_cli.hgp_mu_classes(3) for t in MOMENTS]
PREFIXES = ("hmu_", "hquad_")


def _id(unit) -> str:
    return hgp_name(unit)


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_hgp_kernel(unit), arch)


def write_goldens() -> None:
    """Rewrite every HGP moment golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for unit in UNITS:
            (GOLDEN / backend / f"{_id(unit)}.{ext}").write_text(
                _text(unit, backend))


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_through_f() -> None:
    assert emit_cli.hgp_mu_classes(2) == CLASSES
    assert emit_cli.hgp_mu_classes(0) == [(0, 0)]
    assert emit_cli.DEFAULT_HGP_MU_L == 2 and emit_cli.HGP_MU_L_MAX == 3
    assert [k.name for k in emit_cli.hgp_mu_kernels(2)] == [
        _id(u) for u in UNITS]
    assert len(ALL) == 20
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3: .* through f"):
            emit_cli.hgp_mu_classes(bad)


def test_no_raw_escape_hatch() -> None:
    for unit in UNITS:
        k = build_hgp_kernel(unit)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


def test_the_dipole_is_the_one_e_ladders_through_p() -> None:
    # one builder, one text: --hgp-one-e-l's hmu_ units are these
    one_e = {k.name: k for k in emit_cli.hgp_one_e_kernels(1)}
    for k in emit_cli.hgp_mu_kernels(1):
        if k.name.startswith("hmu_"):
            assert k == one_e[k.name]


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_the_golden_is_unchanged(unit, backend: str) -> None:
    path = GOLDEN / backend / f"{_id(unit)}.{BACKEND[backend][2]}"
    assert _text(unit, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_ladder() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for pre in PREFIXES
                 for p in (GOLDEN / backend).glob(f"{pre}*.{ext}")}
        assert files == {_id(u) for u in UNITS}


@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(unit, "cuda"), _text(unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_plain_stores_no_atomics_no_shared_no_boys() -> None:
    for backend in BACKEND:
        for unit in UNITS:
            text = _text(unit, backend)
            assert "atomicAdd" not in text and "__shared__" not in text
            assert "fj_boys<" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hgp-mu-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_id(u)}.{ext}" for u in UNITS}
    for u in UNITS:
        p = f"{_id(u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_it_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hgp-mu-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "hmu_00.cu", "hquad_00.cu"}


def test_the_bare_default_writes_none(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0") == 0
    assert not any(n.startswith(PREFIXES) for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--hgp-mu-l", "4"],
    ["--hgp-mu-l", "-1"],
    ["--hgp-mu-l", "x"],
    ["--hgp-mu-l", "1", "--class", "0", "0", "0", "0"],
    ["--hgp-mu-l", "1", "--shell-ls", "0"],
    ["--hgp-mu-l", "1", "--fock-l", "0"],
    ["--hgp-mu-l", "1", "--hgp-j-l", "0"],
    ["--hgp-mu-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_above_f_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hgp-mu-l", "4") == 2
    assert "shells through f only" in capsys.readouterr().err


# --- the compile (a compiler, no card) ----------------------------------------

_COMPILE = split_by_l(ALL, lambda u: u[1] + u[2], ids=_id)


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
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hgp_mu.py "
                 "--write")
    write_goldens()
