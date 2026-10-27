"""The HGP one-electron kernels emitted for both vendors: golden texts for
`S`, `T`, `V` and the dipole at (ss) (sp) (pp), the `--hgp-one-e-l` ladder,
and a compile of each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hgp-one-e-l [LMAX]` writes
`hs_`, `ht_`, `hv_` and `hmu_{la}{lb}` (`kernel_hgp.build_hgp_kernel`, each
target on `Engine.HGP`) for every `la <= lb <= LMAX` -- 1 when the flag is
bare. It accepts 0..1 ONLY (the kernels are built and checked for s and p
shells), exits 2 above that, is NEVER in the bare default and is not in
KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{hs,ht,hv,hmu}_{00,01,11}.{cu,hip}`,
and nothing else under those prefixes. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hgp_one_e.py --write

THE COMPILE. Each unit is compiled alone (`-c`) with the kernel tier's own
flags (devtools/config.sh `NVCC_FLAGS` / `HIPCC_FLAGS`, warnings on) and
FAILS on any warning located in the generated unit (#92's filter). Needs a
compiler, never a card: the compile cases are `cuda` / `hip`, none is `gpu`.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest
from test_emit_mu import _WARNING, FLAGS_VAR  # #92's filter, one spelling

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.ir import Raw, walk
from felizjueves.one_electron.kernel_hgp import TARGETS, build_hgp_kernel, hgp_name
from felizjueves.performance.device import GFX1200, SM_86

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = [(0, 0), (0, 1), (1, 1)]
UNITS = [(t, la, lb) for la, lb in CLASSES for t in TARGETS]
PREFIXES = ("hs_", "ht_", "hv_", "hmu_")


def _id(unit) -> str:
    return hgp_name(unit)


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_hgp_kernel(unit), arch)


def write_goldens() -> None:
    """Rewrite every HGP one-electron golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for unit in UNITS:
            (GOLDEN / backend / f"{_id(unit)}.{ext}").write_text(
                _text(unit, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_s_and_p() -> None:
    assert emit_cli.hgp_one_e_classes(1) == CLASSES
    assert emit_cli.hgp_one_e_classes(0) == [(0, 0)]
    assert emit_cli.DEFAULT_HGP_ONE_E_L == emit_cli.HGP_ONE_E_L_MAX == 1
    assert [k.name for k in emit_cli.hgp_one_e_kernels(1)] == [
        _id(u) for u in UNITS]
    for bad in (-1, 2, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.1: .* s and p"):
            emit_cli.hgp_one_e_classes(bad)


def test_no_raw_escape_hatch() -> None:
    for unit in UNITS:
        k = build_hgp_kernel(unit)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


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


def test_plain_stores_no_atomics_no_shared() -> None:
    for backend in BACKEND:
        for unit in UNITS:
            text = _text(unit, backend)
            assert "atomicAdd" not in text and "__shared__" not in text
            assert ("fj_boys<" in text) == (unit[0].value == "v")


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hgp-one-e-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_id(u)}.{ext}" for u in UNITS}
    for u in UNITS:
        p = f"{_id(u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_it_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hgp-one-e-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "hs_00.cu", "ht_00.cu",
                                "hv_00.cu", "hmu_00.cu"}


def test_the_bare_default_writes_none(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0") == 0
    assert not any(n.startswith(PREFIXES) for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--hgp-one-e-l", "2"],
    ["--hgp-one-e-l", "-1"],
    ["--hgp-one-e-l", "x"],
    ["--hgp-one-e-l", "1", "--class", "0", "0", "0", "0"],
    ["--hgp-one-e-l", "1", "--shell-ls", "0"],
    ["--hgp-one-e-l", "1", "--fock-l", "0"],
    ["--hgp-one-e-l", "1", "--hgp-j-l", "0"],
    ["--hgp-one-e-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_above_p_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hgp-one-e-l", "2") == 2
    assert "s and p shells only" in capsys.readouterr().err


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, unit, tmp_path: Path) -> None:
    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_hgp_kernel(unit)
    main = f"{k.name}.{ext}"
    target = (f"-arch=sm_{tc.arch}" if tc.name == "nvcc"
              else f"--offload-arch={tc.arch}")
    flags = [*shlex.split(measured.config_value(FLAGS_VAR[backend])), target]
    got = devcache.compile_unit(tc.path, flags, main, emit(k, arch), {},
                                tmp_path / f"{k.name}.o", mode="-c",
                                timeout=1800)
    diag = got.diagnostics
    assert got.returncode == 0, (
        f"{tc.name} failed in {got.workdir}:\n"
        + "\n".join(shlex.join(c) for c in got.commands) + f"\n{diag}")
    warned = re.findall(r"(?:^|[\s\"'/])" + re.escape(main) + _WARNING, diag,
                        re.MULTILINE)
    assert not warned, (f"{tc.name}: warning(s) in generated text (#92) -- "
                        f"fix the emitter, never a flag: {warned}")


@pytest.mark.cuda
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_nvcc_compiles_the_unit(unit, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", unit, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_hipcc_compiles_the_unit(unit, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", unit, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hgp_one_e.py "
                 "--write")
    write_goldens()
