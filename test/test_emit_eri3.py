"""The three-center kernel emitted for both vendors (#445): golden texts for
the 22 classes at L <= 4, the `--eri3-l` ladder, and a compile of each
through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --eri3-l [LMAX]` writes
`eri3_{la}{lb}{lP}` (`kernel_eri3.build_eri3_kernel`, `Target.ERI3`) for
every `la <= lb` with `la + lb + lP <= LMAX` -- 4 when the flag is bare. It
accepts any LMAX >= 0 (the builder is arbitrary-L), is NEVER in the bare
default and is not in KERNEL_GEN_CMD; it is the `--metric-l` ladder's shape
(#415), one aux shell beside an AO pair.

THE GOLDENS are `test/golden/{cuda,hip}/eri3_{la}{lb}{lP}.{cu,hip}` for the
22 classes (ss|s) through (dd|s), the milestone's L <= 4 cap. The largest,
`eri3_220`, is under 60 KiB, far below the 512 KiB hook, so every text is
tracked and no digests file exists (`test_every_golden_is_under_the_hook`
says so, and would fail first if a class outgrew it). Regenerate (and
review the diff) with

    PYTHONPATH=src uv run python test/test_emit_eri3.py --write

THE COMPILE. Each of the 22 is compiled alone (`-c`) with the kernel tier's
own flags (devtools/config.sh `NVCC_FLAGS` / `HIPCC_FLAGS`, warnings on) and
FAILS on any warning located in the generated unit (#92's filter). Nothing
above L = 4 is compiled. Needs a compiler, never a card: the compile cases
are `cuda` / `hip`, none is `gpu` (the card run is #446).
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest

from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.ir import Raw, walk
from felizjueves.kernel_eri3 import build_eri3_kernel

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
L_MAX = 4  # the milestone's cap: goldens and compiles stop here
HOOK_BYTES = 512 * 1024  # the pre-commit hook's file cap
CLASSES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
           (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
           (0, 2, 0), (0, 2, 1), (0, 2, 2),
           (0, 3, 0), (0, 3, 1),
           (0, 4, 0),
           (1, 1, 0), (1, 1, 1), (1, 1, 2),
           (1, 2, 0), (1, 2, 1),
           (1, 3, 0),
           (2, 2, 0)]


def _name(cls) -> str:
    return "eri3_" + "".join(map(str, cls))


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_eri3_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every eri3 golden (module docstring)."""
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


def test_the_ladder_through_l4_is_the_22_classes() -> None:
    assert emit_cli.eri3_classes(L_MAX) == CLASSES
    assert emit_cli.DEFAULT_ERI3_L == L_MAX
    assert emit_cli.eri3_classes(0) == [(0, 0, 0)]
    assert emit_cli.eri3_classes(1) == [(0, 0, 0), (0, 0, 1), (0, 1, 0)]


def test_the_ladder_accepts_any_l_without_building() -> None:
    got = emit_cli.eri3_classes(16)
    assert all(la <= lb and la + lb + lp <= 16 for la, lb, lp in got)
    assert len(set(got)) == len(got) == sum(
        1 for la in range(17) for lb in range(la, 17)
        for lp in range(17) if la + lb + lp <= 16)
    assert (8, 8, 0) in got and (0, 0, 16) in got and (0, 16, 0) in got
    for bad in (-1, 1.0, True):
        with pytest.raises(ValueError):
            emit_cli.eri3_classes(bad)


def test_eri3_name_is_the_builders_name() -> None:
    for cls in CLASSES:
        assert emit_cli.eri3_name(cls) == build_eri3_kernel(cls).name


def test_no_raw_escape_hatch() -> None:
    for cls in CLASSES:
        k = build_eri3_kernel(cls)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_the_eri3_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_22_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"eri3_*.{ext}")}
        assert files == {_name(c) for c in CLASSES}
    assert not (GOLDEN / "eri3_digests.json").exists()  # none is pinned


def test_every_golden_is_under_the_hook() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        for cls in CLASSES:
            p = GOLDEN / backend / f"{_name(cls)}.{ext}"
            assert p.stat().st_size < HOOK_BYTES, p


@pytest.mark.parametrize("cls", [(0, 0, 0), (1, 1, 1), (2, 2, 0)], ids=_name)
def test_the_vendors_differ_only_in_the_prologue(cls) -> None:
    c, h = _text(cls, "cuda"), _text(cls, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_store_is_plain_with_no_shared_array() -> None:
    for backend in BACKEND:
        text = _text((1, 1, 1), backend)
        assert "atomicAdd" not in text
        assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_eri3_l_bare_writes_the_22_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--eri3-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(c)}.{ext}" for c in CLASSES}
    for c in CLASSES:
        p = f"{_name(c)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_eri3_l_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--eri3-l", "1") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "eri3_000.cu",
                                "eri3_001.cu", "eri3_010.cu"}


def test_eri3_l_beside_metric_l_writes_both(tmp_path) -> None:
    assert _run(tmp_path, "hip", "--L", "0", "--metric-l", "0",
                "--eri3-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.hip", "metric_00.hip",
                                "eri3_000.hip"}


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_eri3_unit(backend, tmp_path) -> None:
    assert _run(tmp_path, backend) == 0
    got = _names(tmp_path)
    assert len(got) == 37
    assert not any(n.startswith("eri3_") for n in got)


@pytest.mark.slow  # builds classes above L = 4 (tiers.FAST_L_MAX)
def test_eri3_l_above_the_cap_is_written_not_refused(tmp_path) -> None:
    # The emitter is arbitrary-L: L = 5 is written (never compiled here).
    assert _run(tmp_path, "hip", "--pair-l", "0", "--eri3-l", "5") == 0
    names = _names(tmp_path)
    assert {f"{_name(c)}.hip" for c in emit_cli.eri3_classes(5)} <= names
    for c in ("eri3_005.hip", "eri3_050.hip", "eri3_230.hip", "eri3_221.hip",
              "eri3_113.hip"):
        assert c in names


@pytest.mark.parametrize("argv", [
    ["--eri3-l", "-1"],
    ["--eri3-l", "x"],
    ["--eri3-l", "1", "--class", "0", "0", "0", "0"],
    ["--eri3-l", "1", "--shell-ls", "0"],
    ["--eri3-l", "1", "--fock-l", "0"],
    ["--eri3-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--eri3-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_an_eri3_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


# --- the compile (a compiler, no card) ----------------------------------------

_WARNING = r"(?:\((\d+)\): warning|:(\d+)(?::\d+)?: warning:?)\s*(.*)"
FLAGS_VAR = {"cuda": "NVCC_FLAGS", "hip": "HIPCC_FLAGS"}


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    import devcache

    from felizjueves import measured

    emit, arch, ext = BACKEND[backend]
    k = build_eri3_kernel(cls)
    assert sum(cls) <= L_MAX  # never compile above the milestone's cap
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
def test_nvcc_compiles_the_eri3_class(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_hipcc_compiles_the_eri3_class(cls, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_eri3.py --write")
    write_goldens()
