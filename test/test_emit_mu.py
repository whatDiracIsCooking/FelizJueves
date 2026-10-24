"""The dipole kernel emitted for both vendors (#522): golden texts for the
six pair classes (ss) through (dd), the `--mu-l` ladder, and a compile of
each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --mu-l [LMAX]` writes `mu_{la}{lb}`
(`kernel_mu.build_mu_kernel`, `Target.DIPOLE`, every component x, y, z in
one fused kernel) for every `la <= lb <= LMAX` -- 2 when the flag is bare.
It accepts 0..2 ONLY (milestone 19's cap, user decision 2026-10-24: the
dipole is validated through (dd), L = 4), exits 2 above that, is NEVER in
the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/mu_{la}{lb}.{cu,hip}` for the six
classes (ss) (sp) (sd) (pp) (pd) (dd), and nothing above. Regenerate (and
review the diff) with

    PYTHONPATH=src uv run python test/test_emit_mu.py --write

THE COMPILE. Each of the six is compiled alone (`-c`) with the kernel tier's
own flags (devtools/config.sh `NVCC_FLAGS` / `HIPCC_FLAGS`, warnings on) and
FAILS on any warning located in the generated unit (#92's filter). Nothing
above (dd) is compiled. Needs a compiler, never a card: the compile cases
are `cuda` / `hip`, none is `gpu` (the card run is #523's).
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.ir import Raw, walk
from felizjueves.one_electron.kernel_mu import build_mu_kernel, mu_name
from felizjueves.performance.device import GFX1200, SM_86

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
L_MAX = 2  # the milestone's cap: each shell l <= 2, the class L <= 4
CLASSES = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def _name(cls) -> str:
    return mu_name(cls)


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_mu_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every dipole golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for cls in CLASSES:
            (GOLDEN / backend / f"{_name(cls)}.{ext}").write_text(
                _text(cls, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_through_dd_is_the_six_classes() -> None:
    assert emit_cli.mu_classes(L_MAX) == CLASSES
    assert emit_cli.DEFAULT_MU_L == emit_cli.MU_L_MAX == L_MAX
    assert emit_cli.mu_classes(0) == [(0, 0)]
    assert emit_cli.mu_classes(1) == [(0, 0), (0, 1), (1, 1)]


def test_the_ladder_refuses_above_dd() -> None:
    for bad in (-1, 3, 8, 1.0, True):
        with pytest.raises(ValueError,
                           match=r"outside 0\.\.2: .* through \(dd\) only"):
            emit_cli.mu_classes(bad)


def test_the_names_are_the_builders() -> None:
    for cls in CLASSES:
        assert build_mu_kernel(cls).name == _name(cls) == (
            "mu_" + "".join(map(str, cls)))


def test_no_raw_escape_hatch() -> None:
    for cls in CLASSES:
        k = build_mu_kernel(cls)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_the_mu_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_six_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"mu_*.{ext}")}
        assert files == {_name(c) for c in CLASSES}


@pytest.mark.parametrize("cls", [(0, 0), (1, 2), (2, 2)], ids=_name)
def test_the_vendors_differ_only_in_the_prologue(cls) -> None:
    c, h = _text(cls, "cuda"), _text(cls, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_every_component_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        text = _text((1, 1), backend)
        assert "atomicAdd" in text
        assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_mu_l_bare_writes_the_six_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--mu-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(c)}.{ext}" for c in CLASSES}
    for c in CLASSES:
        p = f"{_name(c)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_mu_l_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--mu-l", "1") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "mu_00.cu", "mu_01.cu",
                                "mu_11.cu"}


@pytest.mark.parametrize("argv", [
    ["--mu-l", "3"],
    ["--mu-l", "8"],
    ["--mu-l", "-1"],
    ["--mu-l", "x"],
    ["--mu-l", "1", "--class", "0", "0", "0", "0"],
    ["--mu-l", "1", "--shell-ls", "0"],
    ["--mu-l", "1", "--fock-l", "0"],
    ["--mu-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--mu-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_mu_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_mu_l_above_dd_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--mu-l", "3") == 2
    assert "validated through (dd) only" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# --- the compile (a compiler, no card) ----------------------------------------

_WARNING = r"(?:\((\d+)\): warning|:(\d+)(?::\d+)?: warning:?)\s*(.*)"
FLAGS_VAR = {"cuda": "NVCC_FLAGS", "hip": "HIPCC_FLAGS"}


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_mu_kernel(cls)
    assert max(cls) <= L_MAX  # never compile above the milestone's cap
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
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_nvcc_compiles_the_mu_class(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_hipcc_compiles_the_mu_class(cls, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_mu.py --write")
    write_goldens()
