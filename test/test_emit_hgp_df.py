"""The HGP DF kernels emitted for both vendors: golden texts for the metric's
three and the three-center's six s/p classes, the `--hgp-metric-l` /
`--hgp-eri3-l` ladders, and a compile of each through nvcc (sm_86) and hipcc
(gfx1200).

THE LADDERS. `python -m felizjueves.emit --hgp-metric-l [LMAX]` writes
`hmetric_{lP}{lQ}` (`kernel_hgp.build_hmetric_kernel`) for every `lP <= lQ <=
LMAX`, and `--hgp-eri3-l [LMAX]` writes `heri3_{la}{lb}{lP}`
(`kernel_hgp.build_heri3_kernel`) for every `la <= lb <= LMAX`, `lP <= LMAX`.
LMAX bounds each SHELL and is accepted 0..`HGP_DF_L_MAX` = 1 only (bare: 1);
neither is in the bare default nor in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{hmetric,heri3}_*.{cu,hip}`.
Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hgp_df.py --write

THE COMPILE. Each unit is compiled alone (`-c`) with the kernel tier's own
flags and FAILS on any warning located in the generated unit (#92's filter).
Needs a compiler, never a card: the compile cases are `cuda` / `hip`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import test_emit_metric as tem

from felizjueves.emit import __main__ as emit_cli
from felizjueves.eri.density_fit.kernel_hgp import (
    HGP_DF_L_MAX,
    build_heri3_kernel,
    build_hmetric_kernel,
)
from felizjueves.ir import Raw, walk

GOLDEN, BACKEND = tem.GOLDEN, tem.BACKEND
METRIC = [(0, 0), (0, 1), (1, 1)]
ERI3 = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
UNITS = [("hmetric", c) for c in METRIC] + [("heri3", c) for c in ERI3]
BUILD = {"hmetric": build_hmetric_kernel, "heri3": build_heri3_kernel}


def _name(unit) -> str:
    kind, cls = unit
    return f"{kind}_{''.join(map(str, cls))}"


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(BUILD[unit[0]](unit[1]), arch)


def write_goldens() -> None:
    """Rewrite every HGP DF golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for unit in UNITS:
            (GOLDEN / backend / f"{_name(unit)}.{ext}").write_text(
                _text(unit, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladders ---------------------------------------------------------------


def test_the_ladders_through_p_are_the_nine_classes() -> None:
    assert emit_cli.DEFAULT_HGP_DF_L == HGP_DF_L_MAX == 1
    assert emit_cli.hgp_metric_classes(1) == METRIC
    assert emit_cli.hgp_eri3_classes(1) == ERI3
    assert emit_cli.hgp_metric_classes(0) == [(0, 0)]
    assert emit_cli.hgp_eri3_classes(0) == [(0, 0, 0)]
    for bad in (-1, 2, 1.0, True):
        with pytest.raises(ValueError, match="s and p shells only"):
            emit_cli.hgp_metric_classes(bad)
        with pytest.raises(ValueError, match="s and p shells only"):
            emit_cli.hgp_eri3_classes(bad)


def test_no_raw_escape_hatch() -> None:
    for unit in UNITS:
        k = BUILD[unit[0]](unit[1])
        assert k.name == _name(unit)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens ---------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", UNITS, ids=_name)
def test_the_hgp_df_golden_is_unchanged(unit, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(unit)}.{BACKEND[backend][2]}"
    assert _text(unit, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_nine_units() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for kind in BUILD
                 for p in (GOLDEN / backend).glob(f"{kind}_*.{ext}")}
        assert files == {_name(u) for u in UNITS}


@pytest.mark.parametrize("unit", [("hmetric", (1, 1)), ("heri3", (1, 1, 1))],
                         ids=_name)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(unit, "cuda"), _text(unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_store_is_plain_and_the_signature_is_the_mmd_kernels() -> None:
    for backend in BACKEND:
        for unit in (("hmetric", (1, 1)), ("heri3", (1, 1, 1))):
            text = _text(unit, backend)
            assert "atomicAdd" not in text and "__shared__" not in text
        mmd = tem._text((1, 1), backend)
        sig = mmd[mmd.index("metric_11("):mmd.index(")")]
        hgp = _text(("hmetric", (1, 1)), backend)
        assert hgp[hgp.index("hmetric_11(") + 1:hgp.index(")")] == sig


# --- the CLI path --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_both_flags_bare_write_the_nine_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hgp-metric-l",
                "--hgp-eri3-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(u)}.{ext}" for u in UNITS}
    for u in UNITS:
        p = f"{_name(u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_each_flag_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hgp-metric-l", "0",
                "--metric-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "hmetric_00.cu",
                                "metric_00.cu"}
    out = tmp_path / "e"
    assert _run(out, "hip", "--L", "0", "--hgp-eri3-l", "0") == 0
    assert _names(out) == {"pq_L0_p0_q0.hip", "heri3_000.hip"}


@pytest.mark.parametrize("argv", [
    ["--hgp-metric-l", "2"],
    ["--hgp-eri3-l", "2"],
    ["--hgp-metric-l", "-1"],
    ["--hgp-eri3-l", "x"],
    ["--hgp-metric-l", "1", "--class", "0", "0", "0", "0"],
    ["--hgp-eri3-l", "1", "--shell-ls", "0"],
    ["--hgp-metric-l", "1", "--fock-l", "0"],
    ["--hgp-eri3-l", "1", "--hgp-j-l", "0"],
    ["--hgp-metric-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


# --- the compile (a compiler, no card) -----------------------------------------


def _compile(tc, backend: str, unit, tmp_path: Path) -> None:
    import re
    import shlex

    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = BUILD[unit[0]](unit[1])
    main = f"{k.name}.{ext}"
    target = (f"-arch=sm_{tc.arch}" if tc.name == "nvcc"
              else f"--offload-arch={tc.arch}")
    flags = [*shlex.split(measured.config_value(tem.FLAGS_VAR[backend])),
             target]
    got = devcache.compile_unit(tc.path, flags, main, emit(k, arch), {},
                                tmp_path / f"{k.name}.o", mode="-c",
                                timeout=1800)
    diag = got.diagnostics
    assert got.returncode == 0, (
        f"{tc.name} failed in {got.workdir}:\n"
        + "\n".join(shlex.join(c) for c in got.commands) + f"\n{diag}")
    warned = re.findall(r"(?:^|[\s\"'/])" + re.escape(main) + tem._WARNING,
                        diag, re.MULTILINE)
    assert not warned, (f"{tc.name}: warning(s) in generated text (#92) -- "
                        f"fix the emitter, never a flag: {warned}")


@pytest.mark.cuda
@pytest.mark.parametrize("unit", UNITS, ids=_name)
def test_nvcc_compiles_the_hgp_df_unit(unit, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", unit, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", UNITS, ids=_name)
def test_hipcc_compiles_the_hgp_df_unit(unit, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", unit, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hgp_df.py --write")
    write_goldens()
