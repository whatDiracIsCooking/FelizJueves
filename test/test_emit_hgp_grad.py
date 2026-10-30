"""The HGP kinetic-gradient kernels emitted for both vendors: golden texts
for `hgt_` at (ss) (sp) (pp), the `--hgp-grad-l` ladder, and a compile of
each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hgp-grad-l [LMAX]` writes
`hgt_{la}{lb}` (`kernel_hgt.build_hgt_kernel`, `Target.T` at d1 on
`Engine.HGP`), `hgs_{la}{lb}` (`kernel_hgs.build_hgs_kernel`, `S`; its
goldens and compiles are test_emit_hgs.py's) and `ghv_{la}{lb}`
(`kernel_ghv.build_ghv_kernel`, `V`; test_emit_ghv.py's) for every `la <=
lb <= LMAX`
-- 1 when the flag is bare. It
accepts 0..3 ONLY (the kernels are checked through (ff), interpreted),
exits 2 above that, is NEVER in the bare default and is not in
KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/hgt_{00,01,11}.{cu,hip}`, and
nothing else under that prefix. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hgp_grad.py --write

THE COMPILE is test_emit_hgp_one_e.py's, for every ladder class through
(ff) (above the fast cut `slow`, and `tiers.sample_compiles` keeps the
first of each kind in the sample): each unit alone (`-c`) with the
kernel tier's own flags, FAILING on any warning located in the generated
unit (#92's filter). Needs a compiler, never a card.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_emit_hgp_one_e import BACKEND, GOLDEN, _names, _no_tier_env, _run
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.gradient.kernel_ghv import ghv_name
from felizjueves.gradient.kernel_hgs import hgs_name
from felizjueves.gradient.kernel_hgt import build_hgt_kernel, hgt_name
from felizjueves.ir import Raw, walk

__all__ = ["_no_tier_env"]  # the autouse fixture, imported to apply here

CLASSES = [(0, 0), (0, 1), (1, 1)]
# every ladder class through (ff), compiled; L for the tier cut is
# la + lb + 1, test_kernel_hgt.py's
COMPILED = [by_l(c, sum(c) + 1, id=f"hgt_{c[0]}{c[1]}")
            for c in emit_cli.hgp_grad_classes(3)]
PREFIX = "hgt_"


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_hgt_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every HGP kinetic-gradient golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for cls in CLASSES:
            (GOLDEN / backend / f"{hgt_name(cls)}.{ext}").write_text(
                _text(cls, backend))


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_through_f() -> None:
    assert emit_cli.hgp_grad_classes(1) == CLASSES
    assert emit_cli.hgp_grad_classes(0) == [(0, 0)]
    assert len(emit_cli.hgp_grad_classes(3)) == 10
    assert emit_cli.DEFAULT_HGP_GRAD_L == 1
    assert emit_cli.HGP_GRAD_L_MAX == 3
    assert [k.name for k in emit_cli.hgp_grad_kernels(1)] == [
        n for c in CLASSES for n in (hgt_name(c), hgs_name(c), ghv_name(c))]
    for bad in (-1, 4, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3: .* through f"):
            emit_cli.hgp_grad_classes(bad)


def test_no_raw_escape_hatch() -> None:
    for cls in CLASSES:
        k = build_hgt_kernel(cls)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=hgt_name)
def test_the_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{hgt_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_ladder() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"{PREFIX}*.{ext}")}
        assert files == {hgt_name(c) for c in CLASSES}


@pytest.mark.parametrize("cls", CLASSES, ids=hgt_name)
def test_the_vendors_differ_only_in_the_prologue(cls) -> None:
    c, h = _text(cls, "cuda"), _text(cls, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_atomics_and_no_shared() -> None:
    for backend in BACKEND:
        for cls in CLASSES:
            text = _text(cls, backend)
            assert text.count("atomicAdd") == 6 and "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hgp-grad-l") == 0
    units = [n for c in CLASSES
             for n in (hgt_name(c), hgs_name(c), ghv_name(c))]
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{n}.{ext}" for n in units}
    for n in units:
        p = f"{n}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_it_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hgp-grad-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "hgt_00.cu", "hgs_00.cu",
                                "ghv_00.cu"}


def test_the_bare_default_writes_none(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0") == 0
    assert not any(n.startswith((PREFIX, "hgs_", "ghv_"))
                   for n in _names(tmp_path))


@pytest.mark.parametrize("argv", [
    ["--hgp-grad-l", "4"],
    ["--hgp-grad-l", "-1"],
    ["--hgp-grad-l", "x"],
    ["--hgp-grad-l", "1", "--class", "0", "0", "0", "0"],
    ["--hgp-grad-l", "1", "--shell-ls", "0"],
    ["--hgp-grad-l", "1", "--fock-l", "0"],
    ["--hgp-grad-l", "1", "--hgp-j-l", "0"],
    ["--hgp-grad-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_flag_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_above_f_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hgp-grad-l", "4") == 2
    assert "through f only" in capsys.readouterr().err


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    import re
    import shlex

    import devcache
    from test_emit_mu import _WARNING, FLAGS_VAR

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_hgt_kernel(cls)
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
@pytest.mark.parametrize("cls", COMPILED)
def test_nvcc_compiles_the_unit(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", COMPILED)
def test_hipcc_compiles_the_unit(cls, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hgp_grad.py "
                 "--write")
    write_goldens()
