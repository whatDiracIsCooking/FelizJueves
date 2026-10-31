"""The dipole and quadrupole gradient kernels emitted for both vendors:
golden texts for (ss) through (dd), the `--grad-mu-l` ladder, and a compile
of each unit through (ff) by nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --grad-mu-l [LMAX]` writes
`gmu_{la}{lb}` and `gquad_{la}{lb}` (`gradient.kernel_gmu`, `Target.DIPOLE`
/ `QUADRUPOLE` at d1) for every `la <= lb <= LMAX` -- 2 when the flag is
bare. It accepts 0..3 (validated through (ff), interpreted:
test/test_kernel_gmu.py), exits 2 above, is NEVER in the bare default and
is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{gmu,gquad}_{la}{lb}.{cu,hip}` for
the six classes (ss) (sp) (sd) (pp) (pd) (dd), and nothing above.
Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_grad_mu.py --write

THE COMPILE. Each unit through (ff) is compiled alone (`-c`) with the
kernel tier's own flags and FAILS on any warning located in the generated
unit (#92's filter), as `test_emit_quad.py`'s are. Needs a compiler, never
a card: the compile cases are `cuda` / `hip`, none is `gpu`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_emit_mu import _WARNING, FLAGS_VAR  # the same #92 filter, flags
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.gradient.kernel_gmu import (
    GRADIENT_MOMENTS,
    build_gmu_kernel,
    gmu_name,
)
from felizjueves.ir import Raw, walk
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Target

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
UNITS = [(t, *c) for c in CLASSES for t in GRADIENT_MOMENTS]
ALL = [(t, la, lb) for la in range(4) for lb in range(la, 4)
       for t in GRADIENT_MOMENTS]
PREFIXES = ("gmu_", "gquad_")


def _name(target, la, lb) -> str:
    return gmu_name((target, la, lb))


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_gmu_kernel(unit), arch)


def write_goldens() -> None:
    """Rewrite every moment-gradient golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for u in UNITS:
            (GOLDEN / backend / f"{_name(*u)}.{ext}").write_text(
                _text(u, backend))


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
    assert emit_cli.grad_mu_classes(2) == CLASSES
    assert emit_cli.DEFAULT_GRAD_MU_L == 2 and emit_cli.GRAD_MU_L_MAX == 3
    assert emit_cli.grad_mu_classes(0) == [(0, 0)]
    assert len(emit_cli.grad_mu_classes(3)) == 10


def test_the_ladder_refuses_above_ff() -> None:
    for bad in (-1, 4, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            emit_cli.grad_mu_classes(bad)


def test_the_names_are_the_builders() -> None:
    for u in UNITS:
        prefix = "gmu_" if u[0] is Target.DIPOLE else "gquad_"
        assert build_gmu_kernel(u).name == _name(*u) == (
            prefix + "".join(map(str, u[1:])))


def test_no_raw_escape_hatch() -> None:
    for u in UNITS:
        k = build_gmu_kernel(u)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", UNITS, ids=lambda u: _name(*u))
def test_the_golden_is_unchanged(unit, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(*unit)}.{BACKEND[backend][2]}"
    assert _text(unit, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_twelve_units() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for prefix in PREFIXES
                 for p in (GOLDEN / backend).glob(f"{prefix}*.{ext}")}
        assert files == {_name(*u) for u in UNITS}


@pytest.mark.parametrize("unit", [UNITS[0], UNITS[-1]],
                         ids=lambda u: _name(*u))
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(unit, "cuda"), _text(unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_every_component_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        dip = _text((Target.DIPOLE, 1, 1), backend)
        quad = _text((Target.QUADRUPOLE, 1, 1), backend)
        assert "__shared__" not in dip + quad
        assert "atomicAdd(&G[((2 * n_atom" in dip
        assert "atomicAdd(&G[((3 * n_atom" not in dip
        assert "atomicAdd(&G[((5 * n_atom" in quad
        assert "atomicAdd(&G[((6 * n_atom" not in quad


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_grad_mu_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--grad-mu-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(*u)}.{ext}" for u in UNITS}
    for u in UNITS:
        p = f"{_name(*u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_grad_mu_l_sits_beside_grad_l(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--grad-l", "0",
                "--grad-mu-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "gs_00.cu", "gt_00.cu",
                                "gv_00.cu", "gjab_00.cu", "gmu_00.cu",
                                "gquad_00.cu"}


@pytest.mark.parametrize("argv", [
    ["--grad-mu-l", "4"],
    ["--grad-mu-l", "-1"],
    ["--grad-mu-l", "x"],
    ["--grad-mu-l", "1", "--class", "0", "0", "0", "0"],
    ["--grad-mu-l", "1", "--shell-ls", "0"],
    ["--grad-mu-l", "1", "--fock-l", "0"],
    ["--grad-mu-l", "1", "--hgp-j-l", "0"],
], ids=lambda a: "_".join(a))
def test_a_grad_mu_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_grad_mu_l_above_ff_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--grad-mu-l", "4") == 2
    assert "validated through (ff) only" in capsys.readouterr().err


def test_kernel_gen_cmd_asks_for_no_moment_gradient_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--grad-mu-l" not in line


def test_no_other_ladder_writes_a_moment_gradient_name(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--grad-l", "3", "--mu-l",
                "--quad-l") == 0
    assert not any(n.startswith(PREFIXES) for n in _names(tmp_path))


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, name: str, tmp_path: Path) -> None:
    import re
    import shlex

    import devcache

    from felizjueves.performance import measured

    (unit,) = [u for u in ALL if _name(*u) == name]
    emit, arch, ext = BACKEND[backend]
    k = build_gmu_kernel(unit)
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


_COMPILE = [by_l(_name(*u), u[1] + u[2] + 1, id=_name(*u)) for u in ALL]


@pytest.mark.cuda
@pytest.mark.parametrize("name", _COMPILE)
def test_nvcc_compiles_the_unit(name, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", name, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("name", _COMPILE)
def test_hipcc_compiles_the_unit(name, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", name, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_grad_mu.py "
                 "--write")
    write_goldens()
