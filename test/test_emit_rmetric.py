"""The DF metric on Rys quadrature emitted for both vendors: golden texts for
the nine classes `lP + lQ <= 4`, the `--rys-metric-l` ladder, and a compile
of each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --rys-metric-l [LMAX]` writes
`rmetric_{lP}{lQ}` (`kernel_rmetric.build_rmetric_kernel`) for every
`lP <= lQ`, `lP + lQ <= LMAX`; LMAX is the class's TOTAL L, as
`--metric-l`'s, accepted 0..`RYS_METRIC_L_MAX` = 9 only (the Rys rule's
N <= 5; bare: 4). It is in neither the bare default nor KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/rmetric_*.{cu,hip}`, the nine
units at `L <= tiers.FAST_L_MAX`; above it (through (g|h)) the units are
built and emitted, `slow`. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_rmetric.py --write

THE COMPILE. Each unit of the whole ladder, (s|s) through (g|h), is
compiled alone (`-c`) with the kernel tier's own flags and FAILS on any
warning located in the generated unit (#92's filter); above
`tiers.FAST_L_MAX` the case is `slow`. Needs a compiler, never a card: the
compile cases are `cuda` / `hip`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import test_emit_metric as tem
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.eri.density_fit.kernel_rmetric import (
    RYS_METRIC_L_MAX,
    build_rmetric_kernel,
)
from felizjueves.ir import Raw, walk

GOLDEN, BACKEND = tem.GOLDEN, tem.BACKEND
CLASSES = tem.CLASSES  # the nine classes lP + lQ <= 4, the goldens
ABOVE = [(a, b) for a in range(10) for b in range(a, 10)
         if 4 < a + b <= RYS_METRIC_L_MAX]


def _name(cls) -> str:
    return f"rmetric_{cls[0]}{cls[1]}"


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_rmetric_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every Rys metric golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for cls in CLASSES:
            (GOLDEN / backend / f"{_name(cls)}.{ext}").write_text(
                _text(cls, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_OUT", "KERNEL_TIER_ARCH"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ----------------------------------------------------------------


def test_the_ladder_is_every_class_through_total_l() -> None:
    assert emit_cli.DEFAULT_RYS_METRIC_L == 4
    assert emit_cli.RYS_METRIC_L_MAX == RYS_METRIC_L_MAX == 9
    assert emit_cli.rys_metric_classes(4) == sorted(CLASSES)
    assert emit_cli.rys_metric_classes(0) == [(0, 0)]
    assert len(emit_cli.rys_metric_classes(9)) == len(CLASSES) + len(ABOVE)
    for bad in (-1, 10, 1.0, True):
        with pytest.raises(ValueError, match="stops at N = 5"):
            emit_cli.rys_metric_classes(bad)


@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_no_raw_escape_hatch_and_the_rule_is_carried(cls) -> None:
    k = build_rmetric_kernel(cls)
    assert k.name == _name(cls)
    assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name
    for backend in BACKEND:
        text = _text(cls, backend)
        assert "fj_rys<" in text and "fj_boys" not in text


# --- the goldens ---------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_the_rmetric_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_nine_units() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"rmetric_*.{ext}")}
        assert files == {_name(c) for c in CLASSES}


@pytest.mark.parametrize("cls", [by_l(c, sum(c), id=_name(c)) for c in ABOVE])
def test_above_the_goldens_every_class_emits_for_both_vendors(cls) -> None:
    for backend in BACKEND:
        assert f"{_name(cls)}(" in _text(cls, backend)


def test_the_vendors_differ_only_in_the_prologue() -> None:
    c, h = _text((1, 1), "cuda"), _text((1, 1), "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_store_is_plain_and_the_signature_is_the_mmd_kernels() -> None:
    for backend in BACKEND:
        text = _text((1, 1), backend)
        assert "atomicAdd" not in text and "__shared__" not in text
        mmd = tem._text((1, 1), backend)
        sig = mmd[mmd.index("metric_11("):mmd.index(")")]
        assert text[text.index("rmetric_11(") + 1:text.index(")")] == sig


# --- the CLI path --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_flag_writes_the_nine_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--rys-metric-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(c)}.{ext}" for c in CLASSES}
    for c in CLASSES:
        p = f"{_name(c)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_it_adds_its_units_beside_the_other_metric_ladders(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--rys-metric-l", "1",
                "--hgp-metric-l", "0", "--metric-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "rmetric_00.cu",
                                "rmetric_01.cu", "hmetric_00.cu",
                                "metric_00.cu"}


def test_the_bare_default_writes_none(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0") == 0
    assert not any(n.startswith("rmetric_") for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--rys-metric-l", "10"],
    ["--rys-metric-l", "-1"],
    ["--rys-metric-l", "x"],
    ["--rys-metric-l", "1", "--class", "0", "0", "0", "0"],
    ["--rys-metric-l", "1", "--shell-ls", "0"],
    ["--rys-metric-l", "1", "--fock-l", "0"],
    ["--rys-metric-l", "1", "--rys"],
    ["--rys-metric-l", "1", "--hgp-j-l", "0"],
    ["--rys-metric-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_past_the_rule_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--rys-metric-l", "10") == 2
    assert "N = 5 nodes" in capsys.readouterr().err


# --- the compile (a compiler, no card) -----------------------------------------


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    import re
    import shlex

    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_rmetric_kernel(cls)
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


LADDER = [by_l(c, sum(c), id=_name(c)) for c in CLASSES + ABOVE]


@pytest.mark.cuda
@pytest.mark.parametrize("cls", LADDER)
def test_nvcc_compiles_the_rmetric_unit(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", LADDER)
def test_hipcc_compiles_the_rmetric_unit(cls, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_rmetric.py "
                 "--write")
    write_goldens()
