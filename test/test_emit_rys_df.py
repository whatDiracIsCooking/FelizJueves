"""The Rys three-center kernel emitted for both vendors: golden texts for the
22 classes at L <= 4, the `--rys-eri3-l` ladder, and a compile of each
through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --rys-eri3-l [LMAX]` writes
`reri3_{la}{lb}{lP}` (`kernel_rys.build_reri3_kernel`, `Target.ERI3` on
`Engine.RYS`) for every `la <= lb` with `la + lb + lP <= LMAX` -- 4 when
the flag is bare, the classes of `--eri3-l 4`. LMAX is accepted
0..`RYS_ERI3_L_MAX` = 9 only (the Rys rule stops at N = 5); neither the
bare default nor KERNEL_GEN_CMD writes a `reri3_` unit.

THE GOLDENS are `test/golden/{cuda,hip}/reri3_{la}{lb}{lP}.{cu,hip}`, every
text tracked (each far under the 512 KiB hook). Each unit carries
`fj_rys<N>` (`rys.roots.device_text`). Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_rys_df.py --write

THE COMPILE. Each unit is compiled alone (`-c`) with the kernel tier's own
flags and FAILS on any warning located in the generated unit (#92's filter).
Needs a compiler, never a card: the compile cases are `cuda` / `hip`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import test_emit_eri3 as tee

from felizjueves.emit import __main__ as emit_cli
from felizjueves.eri.density_fit.kernel_rys import (
    RYS_ERI3_L_MAX,
    build_reri3_kernel,
    reri3_name,
)
from felizjueves.ir import Raw, walk

GOLDEN, BACKEND, CLASSES = tee.GOLDEN, tee.BACKEND, tee.CLASSES


def _name(cls) -> str:
    return "reri3_" + "".join(map(str, cls))


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_reri3_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every Rys three-center golden (module docstring)."""
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


# --- the ladder ----------------------------------------------------------------


def test_the_ladder_is_the_eri3_ladder_capped_at_the_rule() -> None:
    assert emit_cli.rys_eri3_classes(4) == CLASSES
    assert emit_cli.rys_eri3_classes(RYS_ERI3_L_MAX) == emit_cli.eri3_classes(
        RYS_ERI3_L_MAX)
    assert emit_cli.rys_eri3_classes(0) == [(0, 0, 0)]
    for bad in (-1, RYS_ERI3_L_MAX + 1, 1.0, True):
        with pytest.raises(ValueError, match="stops at N = 5"):
            emit_cli.rys_eri3_classes(bad)


def test_the_names_and_no_raw_escape_hatch() -> None:
    for cls in CLASSES:
        k = build_reri3_kernel(cls)
        assert k.name == reri3_name(cls) == _name(cls)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens ---------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_the_rys_eri3_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_22_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"reri3_*.{ext}")}
        assert files == {_name(c) for c in CLASSES}
        for c in CLASSES:
            p = GOLDEN / backend / f"{_name(c)}.{ext}"
            assert p.stat().st_size < tee.HOOK_BYTES, p


@pytest.mark.parametrize("cls", [(0, 0, 0), (1, 1, 1), (2, 2, 0)], ids=_name)
def test_the_vendors_differ_only_in_the_prologue(cls) -> None:
    c, h = _text(cls, "cuda"), _text(cls, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_unit_carries_the_rule_and_the_mmd_signature() -> None:
    for backend in BACKEND:
        text = _text((1, 1, 1), backend)
        assert "atomicAdd" not in text and "__shared__" not in text
        assert "fj_rys<2>(" in text and "fj_boys<" not in text
        mmd = tee._text((1, 1, 1), backend)
        sig = mmd[mmd.index("eri3_111(") + len("eri3_111("):]
        rys = text[text.index("reri3_111(") + len("reri3_111("):]
        assert rys[:rys.index(")")] == sig[:sig.index(")")]


# --- the CLI path --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_rys_eri3_l_bare_writes_the_22_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--rys-eri3-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(c)}.{ext}" for c in CLASSES}
    for c in CLASSES:
        p = f"{_name(c)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_rys_eri3_l_adds_its_units_beside_the_df_ladders(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--rys-eri3-l", "1",
                "--eri3-l", "0", "--hgp-eri3-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "eri3_000.cu",
                                "heri3_000.cu", "reri3_000.cu",
                                "reri3_001.cu", "reri3_010.cu"}


@pytest.mark.parametrize("argv", [
    ["--rys-eri3-l", "10"],
    ["--rys-eri3-l", "-1"],
    ["--rys-eri3-l", "x"],
    ["--rys-eri3-l", "1", "--rys"],
    ["--rys-eri3-l", "1", "--class", "0", "0", "0", "0"],
    ["--rys-eri3-l", "1", "--shell-ls", "0"],
    ["--rys-eri3-l", "1", "--fock-l", "0"],
    ["--rys-eri3-l", "1", "--hgp-j-l", "0"],
    ["--rys-eri3-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


# --- the compile (a compiler, no card) -----------------------------------------


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    import re
    import shlex

    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_reri3_kernel(cls)
    main = f"{k.name}.{ext}"
    target = (f"-arch=sm_{tc.arch}" if tc.name == "nvcc"
              else f"--offload-arch={tc.arch}")
    flags = [*shlex.split(measured.config_value(tee.FLAGS_VAR[backend])),
             target]
    got = devcache.compile_unit(tc.path, flags, main, emit(k, arch), {},
                                tmp_path / f"{k.name}.o", mode="-c",
                                timeout=1800)
    diag = got.diagnostics
    assert got.returncode == 0, (
        f"{tc.name} failed in {got.workdir}:\n"
        + "\n".join(shlex.join(c) for c in got.commands) + f"\n{diag}")
    warned = re.findall(r"(?:^|[\s\"'/])" + re.escape(main) + tee._WARNING,
                        diag, re.MULTILINE)
    assert not warned, (f"{tc.name}: warning(s) in generated text (#92) -- "
                        f"fix the emitter, never a flag: {warned}")


@pytest.mark.cuda
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_nvcc_compiles_the_rys_eri3_class(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_hipcc_compiles_the_rys_eri3_class(cls, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_rys_df.py --write")
    write_goldens()
