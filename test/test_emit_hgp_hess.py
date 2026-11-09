"""The HGP one-electron Hessian kernels emitted for both vendors: golden
texts for `hgd2s_`, `hgd2t_`, `hgd2v_` at (ss) (sp) (pp), the
`--hgp-hess-l` ladder, and a compile of each unit through (ff) by nvcc
(sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hgp-hess-l [LMAX]` writes
`hgd2s_`, `hgd2t_` and `hgd2v_{la}{lb}` (`hessian.kernel_hgd2`, `S`, `T`,
`V` at d2 on `Engine.HGP`), class-major, for every `la <= lb <= LMAX` -- 1
when the flag is bare. It accepts 0..3 ONLY (the kernels are checked
through (ff), interpreted: test/test_kernel_hgd2.py), exits 2 above that,
is NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/hgd2{s,t,v}_{00,01,11}.{cu,hip}`,
and nothing else under those prefixes. Regenerate (and review the diff)
with

    PYTHONPATH=src uv run python test/test_emit_hgp_hess.py --write

THE COMPILE is test_emit_hgp_grad.py's, for every unit through (ff) (L for
the tier cut `la + lb + 2`, test_kernel_hgd2.py's; above the fast cut
`slow`, and `tiers.sample_compiles` keeps the first of each kind in the
sample): each unit alone (`-c`) with the kernel tier's own flags, FAILING
on any warning located in the generated unit (#92's filter). Needs a
compiler, never a card.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_emit_hgp_one_e import BACKEND, GOLDEN, _names, _no_tier_env, _run
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.hessian.kernel_hgd2 import (
    TARGETS,
    build_hgd2_kernel,
    hgd2_name,
)
from felizjueves.ir import Raw, walk
from felizjueves.spec import Target

__all__ = ["_no_tier_env"]  # the autouse fixture, imported to apply here

CLASSES = [(0, 0), (0, 1), (1, 1)]
UNITS = [(t, *c) for c in CLASSES for t in TARGETS]
ALL = [(t, *c) for c in emit_cli.hgp_hess_classes(3) for t in TARGETS]
PREFIXES = ("hgd2s_", "hgd2t_", "hgd2v_")


def _text(unit, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_hgd2_kernel(unit), arch)


def write_goldens() -> None:
    """Rewrite every HGP one-electron Hessian golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for u in UNITS:
            (GOLDEN / backend / f"{hgd2_name(u)}.{ext}").write_text(
                _text(u, backend))


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_through_f() -> None:
    assert emit_cli.hgp_hess_classes(1) == CLASSES
    assert emit_cli.hgp_hess_classes(0) == [(0, 0)]
    assert len(emit_cli.hgp_hess_classes(3)) == 10
    assert emit_cli.DEFAULT_HGP_HESS_L == 1
    assert emit_cli.HGP_HESS_L_MAX == 3
    assert [k.name for k in emit_cli.hgp_hess_kernels(1)] == [
        hgd2_name(u) for u in UNITS]
    for bad in (-1, 4, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3: .* through f"):
            emit_cli.hgp_hess_classes(bad)


def test_no_raw_escape_hatch() -> None:
    for u in UNITS:
        k = build_hgd2_kernel(u)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", UNITS, ids=hgd2_name)
def test_the_golden_is_unchanged(unit, backend: str) -> None:
    path = GOLDEN / backend / f"{hgd2_name(unit)}.{BACKEND[backend][2]}"
    assert _text(unit, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_ladder() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for prefix in PREFIXES
                 for p in (GOLDEN / backend).glob(f"{prefix}*.{ext}")}
        assert files == {hgd2_name(u) for u in UNITS}


@pytest.mark.parametrize("unit", [UNITS[0], UNITS[-1]], ids=hgd2_name)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(unit, "cuda"), _text(unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_atomics_and_no_shared() -> None:
    for backend in BACKEND:
        for u in UNITS:
            text = _text(u, backend)
            want = 36 + 45 if u[0] is Target.V else 36
            assert text.count("atomicAdd(") == want, hgd2_name(u)
            assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hgp-hess-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{hgd2_name(u)}.{ext}" for u in UNITS}
    for u in UNITS:
        p = f"{hgd2_name(u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_it_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hgp-grad-l", "0",
                "--hgp-hess-l", "0") == 0
    assert _names(tmp_path) == {
        "pq_L0_p0_q0.cu", "hgt_00.cu", "hgs_00.cu", "ghv_00.cu", "hgmu_00.cu",
        "hgquad_00.cu", "hgd2s_00.cu", "hgd2t_00.cu", "hgd2v_00.cu"}


def test_no_other_ladder_writes_an_hgp_hessian_name(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hgp-grad-l", "1",
                "--hess-one-e-l", "1", "--hgp-one-e-l", "1") == 0
    assert not any(n.startswith(PREFIXES) for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--hgp-hess-l", "4"],
    ["--hgp-hess-l", "-1"],
    ["--hgp-hess-l", "x"],
    ["--hgp-hess-l", "1", "--class", "0", "0", "0", "0"],
    ["--hgp-hess-l", "1", "--shell-ls", "0"],
    ["--hgp-hess-l", "1", "--fock-l", "0"],
    ["--hgp-hess-l", "1", "--hgp-j-l", "0"],
    ["--hgp-hess-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_above_f_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hgp-hess-l", "4") == 2
    assert "through f only" in capsys.readouterr().err


def test_kernel_gen_cmd_asks_for_no_hgp_hessian_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--hgp-hess-l" not in line


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, name: str, tmp_path: Path) -> None:
    import re
    import shlex

    import devcache
    from test_emit_mu import _WARNING, FLAGS_VAR

    from felizjueves.performance import measured

    (unit,) = [u for u in ALL if hgd2_name(u) == name]
    emit, arch, ext = BACKEND[backend]
    k = build_hgd2_kernel(unit)
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


_COMPILE = [by_l(hgd2_name(u), u[1] + u[2] + 2, id=hgd2_name(u))
            for u in ALL]


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
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hgp_hess.py "
                 "--write")
    write_goldens()
