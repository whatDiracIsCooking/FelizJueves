"""The Rys one-electron kernels emitted for both vendors: golden texts for
`S`, `T` and `V` through (ff), the `--rys-one-e-l` ladder, and a compile of
each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --rys-one-e-l [LMAX]` writes
`rs_`, `rt_` and `rv_{la}{lb}` (`kernel_rys.build_rys_kernel`, each target
on `Engine.RYS`) for every `la <= lb <= LMAX` -- 3 when the flag is bare. It
accepts 0..3 ONLY (the kernels are built and checked through (ff)), exits 2
above that, is NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{rs,rt,rv}_{la}{lb}.{cu,hip}` for
every ladder class with `la + lb <= FAST_L_MAX` (the fast tier's cut), and
nothing else under those prefixes; the classes above it, (df) and (ff), are
pinned by byte count and sha256 in `test/golden/rys_one_e_digests.json`
(`slow`, as building them is). Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_rys_one_e.py --write

THE COMPILE is test_emit_hgp_one_e.py's (`_compile`: the tier's flags, a
warning in generated text fails, #92). Needs a compiler, never a card: the
compile cases are `cuda` / `hip`, none is `gpu`. NOT YET RUN: the session
that built the kernels had neither nvcc nor hipcc; every unit passed clang
18's CUDA front end (device side, `-fsyntax-only -Wall -Wextra`, a runtime
stub) with no diagnostic, a proxy.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

import pytest
from test_emit_mu import _WARNING, FLAGS_VAR  # #92's filter, one spelling
from tiers import FAST_L_MAX, split_by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.ir import Raw, walk
from felizjueves.one_electron.kernel_rys import (
    L_MAX,
    TARGETS,
    build_rys_kernel,
    rys_name,
)
from felizjueves.performance.device import GFX1200, SM_86

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
DIGESTS = GOLDEN / "rys_one_e_digests.json"
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = [(la, lb) for la in range(L_MAX + 1) for lb in range(la, L_MAX + 1)]
UNITS = [(t, la, lb) for la, lb in CLASSES for t in TARGETS]
GOLDEN_UNITS = [u for u in UNITS if u[1] + u[2] <= FAST_L_MAX]
PREFIXES = ("rs_", "rt_", "rv_")


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
    """Rewrite every Rys one-electron golden and digest (module
    docstring)."""
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
    assert emit_cli.rys_one_e_classes(3) == CLASSES
    assert emit_cli.rys_one_e_classes(0) == [(0, 0)]
    assert emit_cli.DEFAULT_RYS_ONE_E_L == emit_cli.RYS_ONE_E_L_MAX == L_MAX
    assert [k.name for k in emit_cli.rys_one_e_kernels(1)] == [
        _id(u) for u in UNITS if max(u[1:]) <= 1]
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3: .* through f"):
            emit_cli.rys_one_e_classes(bad)


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
        files = {p.stem for pre in PREFIXES
                 for p in (GOLDEN / backend).glob(f"{pre}*.{ext}")}
        assert files == {_id(u) for u in GOLDEN_UNITS}
        assert set(pinned[backend]) == {
            _id(u) for u in UNITS if u not in GOLDEN_UNITS}


@pytest.mark.parametrize("unit", GOLDEN_UNITS, ids=_id)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(unit, "cuda"), _text(unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_plain_stores_no_atomics_no_shared_the_rule_in_v() -> None:
    for backend in BACKEND:
        for unit in GOLDEN_UNITS:
            text = _text(unit, backend)
            assert "atomicAdd" not in text and "__shared__" not in text
            assert "fj_boys" not in text
            assert ("fj_rys<" in text) == (unit[0].value == "v")


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_it_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--rys-one-e-l", "2") == 0
    units = [u for u in UNITS if max(u[1:]) <= 2]
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_id(u)}.{ext}" for u in units}
    for u in units:
        p = f"{_id(u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


@pytest.mark.slow
def test_bare_writes_through_f(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--rys-one-e-l") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu"} | {
        f"{_id(u)}.cu" for u in UNITS}


def test_it_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--rys-one-e-l", "0",
                "--hgp-one-e-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "rs_00.cu", "rt_00.cu",
                                "rv_00.cu", "hs_00.cu", "ht_00.cu",
                                "hv_00.cu", "hmu_00.cu"}


def test_the_bare_default_writes_none(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0") == 0
    assert not any(n.startswith(PREFIXES) for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--rys-one-e-l", "4"],
    ["--rys-one-e-l", "-1"],
    ["--rys-one-e-l", "x"],
    ["--rys-one-e-l", "1", "--class", "0", "0", "0", "0"],
    ["--rys-one-e-l", "1", "--shell-ls", "0"],
    ["--rys-one-e-l", "1", "--fock-l", "0"],
    ["--rys-one-e-l", "1", "--fock-l", "0", "--rys"],
    ["--rys-one-e-l", "1", "--rys"],
    ["--rys-one-e-l", "1", "--hgp-j-l", "0"],
    ["--rys-one-e-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_above_f_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--rys-one-e-l", "4") == 2
    assert "through f only" in capsys.readouterr().err


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, unit, tmp_path: Path) -> None:
    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_rys_kernel(unit)
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
@pytest.mark.parametrize("unit", split_by_l(UNITS, _l, ids=_id))
def test_nvcc_compiles_the_unit(unit, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", unit, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", split_by_l(UNITS, _l, ids=_id))
def test_hipcc_compiles_the_unit(unit, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", unit, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_rys_one_e.py "
                 "--write")
    write_goldens()
