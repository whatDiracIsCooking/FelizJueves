"""The Coulomb derivative matrix's bra kernel emitted for both vendors
(#688): golden texts for the s and p classes, the `--jx-l` ladder, and a
compile of every ordered class through (ff) by nvcc (sm_86) and hipcc
(gfx1200).

THE LADDER. `python -m felizjueves.emit --jx-l [LMAX]` writes, for every
ORDERED bra class `(la, lb)` with each `l <= LMAX` (2 when the flag is
bare), `jx_{la}{lb}` (`hessian.kernel_jx`). It accepts 0..3 (the kernel is
validated through (ff), interpreted: test/test_kernel_jx.py), exits 2 above
that, is NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/jx_{la}{lb}.{cu,hip}` for (ss),
(sp), (ps) and (pp), 8 files. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_jx.py --write

THE COMPILE. Each unit is compiled alone (`-c`) with the kernel tier's own
flags and FAILS on any warning located in the generated unit (#92's
filter), through test_emit_grad.py's `compile_kernel`. Needs a compiler,
never a card; the compile sample keeps the first class of each backend
(`tiers.sample_compiles`), the rest are `slow`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_emit_grad import compile_kernel

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.hessian.kernel_jx import build_jx_kernel, jx_name
from felizjueves.ir import Raw, walk
from felizjueves.performance.device import GFX1200, SM_86

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = [(0, 0), (0, 1), (1, 0), (1, 1)]
ALL = [(la, lb) for la in range(4) for lb in range(4)]
PREFIX = "jx_"


def _id(c) -> str:
    return jx_name(c)


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_jx_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every jx golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for c in CLASSES:
            (GOLDEN / backend / f"{jx_name(c)}.{ext}").write_text(
                _text(c, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_ordered_class_through_lmax() -> None:
    assert emit_cli.jx_classes(1) == CLASSES
    assert emit_cli.DEFAULT_JX_L == 2 and emit_cli.JX_L_MAX == 3
    assert emit_cli.jx_classes(0) == [(0, 0)]
    assert emit_cli.jx_classes(3) == ALL


def test_the_ladder_refuses_above_ff() -> None:
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            emit_cli.jx_classes(bad)


def test_the_names_are_the_builders_and_no_raw_text() -> None:
    for c in ALL:
        k = build_jx_kernel(c)
        assert k.name == jx_name(c) == PREFIX + "".join(map(str, c))
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_id)
def test_the_jx_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{jx_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_s_and_p_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"jx_[0-9]*.{ext}")}
        assert files == {jx_name(c) for c in CLASSES}


def test_the_vendors_differ_only_in_the_prologue() -> None:
    c, h = _text((1, 1), "cuda"), _text((1, 1), "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_jx_adds_atomically_into_j1_and_nothing_else() -> None:
    for backend in BACKEND:
        t = _text((0, 1), backend)
        assert "atomicAdd(&J1[" in t and "__shared__" not in t
        assert "atom_a" not in t and " M[" not in t


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_jx_l_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--jx-l", "1") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{jx_name(c)}.{ext}" for c in CLASSES}
    for c in CLASSES:
        p = f"{jx_name(c)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_jx_l_bare_is_through_dd(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--jx-l") == 0
    assert _names(tmp_path) - {"pq_L0_p0_q0.cu"} == {
        f"jx_{a}{b}.cu" for a in range(3) for b in range(3)}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_jx_unit(backend, tmp_path) -> None:
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith(PREFIX) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_jx_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--jx-l" not in line


def test_the_jx_names_cannot_collide_with_any_other_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--mu-l",
                "--schwarz-l", "--sx-l", "--metric-l", "--eri3-l",
                "--grad-l", "--grad-k-l", "--hess-l") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIX) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--jx-l", "3") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 16 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--jx-l", "4"],
    ["--jx-l", "-1"],
    ["--jx-l", "x"],
    ["--jx-l", "1", "--class", "0", "0", "0", "0"],
    ["--jx-l", "1", "--shell-ls", "0"],
    ["--jx-l", "1", "--fock-l", "0"],
    ["--jx-l", "1", "--n-batches", "3"],
    ["--jx-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_jx_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_jx_l_above_ff_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--jx-l", "4") == 2
    err = capsys.readouterr().err
    assert "--jx-l 4 is outside 0..3" in err
    assert "validated through (ff) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("cls", ALL, ids=_id)
def test_nvcc_compiles_the_jx_unit(cls, nvcc, tmp_path) -> None:
    compile_kernel(nvcc, "cuda", build_jx_kernel(cls), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", ALL, ids=_id)
def test_hipcc_compiles_the_jx_unit(cls, hipcc, tmp_path) -> None:
    compile_kernel(hipcc, "hip", build_jx_kernel(cls), tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_jx.py --write")
    write_goldens()
