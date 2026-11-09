"""The Rys quadrupole emitted for both vendors: golden texts through the
fast tier's cut, the `--rys-quad-l` ladder, and a compile of each unit
through (ff) by nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --rys-quad-l [LMAX]` writes
`rquad_{la}{lb}` (`kernel_rys.build_rys_kernel` for `Target.QUADRUPOLE` on
`Engine.RYS`) for every `la <= lb <= LMAX` -- 2 when the flag is bare. It
accepts 0..3 ONLY (the kernel is built and checked through (ff)), exits 2
above that, is NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/rquad_{la}{lb}.{cu,hip}` for every
ladder class with `la + lb <= FAST_L_MAX` (the fast tier's cut), and nothing
else under that prefix; the classes above it, (df) and (ff), are pinned by
byte count and sha256 in `test/golden/rys_quad_digests.json` (`slow`, as
building them is). Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_rys_quad.py --write

THE COMPILE is test_emit_rys_one_e.py's (`_compile`: the tier's flags, a
warning in generated text fails, #92). Needs a compiler, never a card: the
compile cases are `cuda` / `hip`, none is `gpu`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from tiers import FAST_L_MAX, split_by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.ir import Raw, walk
from felizjueves.one_electron import kernel_hgp as kh
from felizjueves.one_electron.kernel_rys import (
    L_MAX,
    build_rys_kernel,
    rys_name,
)
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Target

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
DIGESTS = GOLDEN / "rys_quad_digests.json"
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = [(la, lb) for la in range(L_MAX + 1) for lb in range(la, L_MAX + 1)]
UNITS = [(Target.QUADRUPOLE, la, lb) for la, lb in CLASSES]
GOLDEN_UNITS = [u for u in UNITS if u[1] + u[2] <= FAST_L_MAX]
PREFIX = "rquad_"


def _id(unit) -> str:
    return rys_name(unit)


def _l(unit) -> int:
    return unit[1] + unit[2]


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_rys_kernel(unit), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_goldens() -> None:
    """Rewrite every Rys quadrupole golden and digest (module docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for unit in UNITS:
            text = _text(unit, backend)
            if unit in GOLDEN_UNITS:
                (GOLDEN / backend / f"{_id(unit)}.{ext}").write_text(text)
            else:
                digests[backend][_id(unit)] = _digest(text)
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_OUT", "KERNEL_TIER_ARCH"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_s_through_f() -> None:
    assert emit_cli.rys_quad_classes(3) == CLASSES
    assert emit_cli.rys_quad_classes(0) == [(0, 0)]
    assert emit_cli.RYS_QUAD_L_MAX == L_MAX
    assert emit_cli.DEFAULT_RYS_QUAD_L == 2
    assert [k.name for k in emit_cli.rys_quad_kernels(1)] == [
        "rquad_00", "rquad_01", "rquad_11"]
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3: .* through f"):
            emit_cli.rys_quad_classes(bad)


@pytest.mark.parametrize("unit", split_by_l(UNITS, _l, ids=_id))
def test_no_raw_escape_hatch(unit) -> None:
    k = build_rys_kernel(unit)
    assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", split_by_l(UNITS, _l, ids=_id))
def test_the_golden_is_unchanged(unit, backend: str) -> None:
    text = _text(unit, backend)
    if unit in GOLDEN_UNITS:
        path = GOLDEN / backend / f"{_id(unit)}.{BACKEND[backend][2]}"
        assert text == path.read_text()
    else:
        pinned = json.loads(DIGESTS.read_text())[backend][_id(unit)]
        assert _digest(text) == pinned


def test_the_golden_set_is_exactly_the_ladder() -> None:
    pinned = json.loads(DIGESTS.read_text())
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"{PREFIX}*.{ext}")}
        assert files == {_id(u) for u in GOLDEN_UNITS}
        assert set(pinned[backend]) == {
            _id(u) for u in UNITS if u not in GOLDEN_UNITS}


@pytest.mark.parametrize("unit", GOLDEN_UNITS, ids=_id)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(unit, "cuda"), _text(unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_plain_stores_six_components_no_node() -> None:
    for backend in BACKEND:
        for unit in GOLDEN_UNITS:
            text = _text(unit, backend)
            assert "atomicAdd" not in text and "__shared__" not in text
            assert "fj_boys" not in text and "fj_rys" not in text
            assert "qu5_0_0" in text and "qu6_" not in text


@pytest.mark.parametrize("unit", GOLDEN_UNITS, ids=_id)
def test_the_signature_is_hquads(unit) -> None:
    # a drop-in: hquad_'s launcher and staging serve rquad_ unchanged
    def sig(text: str) -> str:
        head = text[text.index("__global__"):text.index("{")]
        return head.split("(", 1)[1]

    for backend in BACKEND:
        emit, arch, _ = BACKEND[backend]
        assert sig(_text(unit, backend)) == sig(
            emit(kh.build_hgp_kernel(unit), arch))


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_it_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--L", "0", "--rys-quad-l") == 0
    units = [u for u in UNITS if max(u[1:]) <= 2]  # the bare flag's
    assert set(units) <= set(GOLDEN_UNITS)
    assert _names(tmp_path) == {f"pq_L0_p0_q0.{ext}"} | {
        f"{_id(u)}.{ext}" for u in units}
    for u in units:
        p = f"{_id(u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_it_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--rys-quad-l", "0",
                "--rys-one-e-l", "0", "--hgp-mu-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "rquad_00.cu", "rs_00.cu",
                                "rt_00.cu", "rv_00.cu", "rmu_00.cu",
                                "hmu_00.cu",
                                "hquad_00.cu"}


def test_the_bare_default_writes_none(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0") == 0
    assert not any(n.startswith(PREFIX) for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--rys-quad-l", "4"],
    ["--rys-quad-l", "-1"],
    ["--rys-quad-l", "x"],
    ["--rys-quad-l", "1", "--class", "0", "0", "0", "0"],
    ["--rys-quad-l", "1", "--shell-ls", "0"],
    ["--rys-quad-l", "1", "--fock-l", "0"],
    ["--rys-quad-l", "1", "--hgp-j-l", "0"],
    ["--rys-quad-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_above_f_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--rys-quad-l", "4") == 2
    assert "through f only" in capsys.readouterr().err


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, unit, tmp_path: Path) -> None:
    from test_emit_rys_one_e import _compile as compile_rys

    compile_rys(tc, backend, unit, tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize("unit", split_by_l(UNITS, _l, ids=_id))
def test_nvcc_compiles_the_unit(unit, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", unit, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", split_by_l(UNITS, _l, ids=_id))
def test_hipcc_compiles_the_unit(unit, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", unit, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_rys_quad.py "
                 "--write")
    write_goldens()
