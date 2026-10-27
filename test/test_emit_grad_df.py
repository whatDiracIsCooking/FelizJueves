"""The DF d1 gradient kernels emitted for both vendors: golden texts for
every class at L <= 4, the `--grad-df-l` ladder, and a compile of each
through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --grad-df-l [LMAX]` writes
`gmetric_{lP}{lQ}` for every `lP <= lQ`, `lP + lQ <= LMAX`, and
`geri3_{la}{lb}{lP}` for every `la <= lb`, `la + lb + lP <= LMAX` (4 when
the flag is bare: 9 + 22 units). It accepts 0..4 (the d0 kernels' cap,
where the gradients are validated, interpreted), exits 2 above that, is
NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{gmetric,geri3}_*.{cu,hip}`, the 31
units of the bare flag (the largest 232 KB). Regenerate (and review the
diff) with

    PYTHONPATH=src uv run python test/test_emit_grad_df.py --write

THE COMPILE. Each golden is compiled alone (`-c`) with the kernel tier's
own flags and FAILS on any warning located in the generated unit (#92's
filter). Needs a compiler, never a card. The session that wrote these
kernels had neither nvcc nor hipcc, so these cases have NOT been run; the
units passed clang 18's CUDA front end (device side, `-Wall -Wextra`, a
declaration stub for the runtime) with no diagnostic, a proxy, not this
test.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest
from test_emit_grad import _WARNING, BACKEND, FLAGS_VAR, GOLDEN, _names, _run

from felizjueves.emit import __main__ as emit_cli
from felizjueves.gradient.kernel_geri3 import build_geri3_kernel, geri3_name
from felizjueves.gradient.kernel_gmetric import (
    build_gmetric_kernel,
    gmetric_name,
)
from felizjueves.ir import Raw, walk

METRIC, ERI3 = emit_cli.grad_df_classes(4)
UNITS = [("gmetric", c) for c in METRIC] + [("geri3", c) for c in ERI3]
PREFIXES = ("gmetric_", "geri3_")


def _kernel(family: str, cls):
    return (build_gmetric_kernel(cls) if family == "gmetric"
            else build_geri3_kernel(cls))


def _name(family: str, cls) -> str:
    return gmetric_name(cls) if family == "gmetric" else geri3_name(cls)


def _id(u) -> str:
    return _name(*u)


def _text(family: str, cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(_kernel(family, cls), arch)


def write_goldens() -> None:
    """Rewrite every DF gradient golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for f, c in UNITS:
            (GOLDEN / backend / f"{_name(f, c)}.{ext}").write_text(
                _text(f, c, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_OUT", "KERNEL_TIER_ARCH"):
        monkeypatch.delenv(v, raising=False)


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_class_through_lmax() -> None:
    assert len(METRIC) == 9 and len(ERI3) == 22
    assert emit_cli.DEFAULT_GRAD_DF_L == 4 and emit_cli.GRAD_DF_L_MAX == 4
    assert emit_cli.grad_df_classes(0) == ([(0, 0)], [(0, 0, 0)])
    assert all(p <= q and p + q <= 4 for p, q in METRIC)
    assert all(a <= b and a + b + p <= 4 for a, b, p in ERI3)


def test_the_ladder_refuses_above_l4() -> None:
    for bad in (-1, 5, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.4"):
            emit_cli.grad_df_classes(bad)


def test_the_names_are_the_builders() -> None:
    for f, c in UNITS:
        assert _kernel(f, c).name == _name(f, c) == f + "_" + "".join(
            map(str, c))


def test_no_raw_escape_hatch() -> None:
    for f, c in UNITS:
        k = _kernel(f, c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_the_df_gradient_golden_is_unchanged(unit, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(*unit)}.{BACKEND[backend][2]}"
    assert _text(*unit, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_ladder() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for f in ("gmetric", "geri3")
                 for p in (GOLDEN / backend).glob(f"{f}_[0-9]*.{ext}")}
        assert files == {_name(*u) for u in UNITS}


@pytest.mark.parametrize("unit", [("gmetric", (1, 2)), ("geri3", (0, 1, 1))],
                         ids=_id)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(*unit, "cuda"), _text(*unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_every_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        for f, c in (("gmetric", (1, 1)), ("geri3", (1, 1, 1))):
            text = _text(f, c, backend)
            assert "atomicAdd(&G[" in text
            assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_grad_df_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--grad-df-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(*u)}.{ext}" for u in UNITS}
    for u in UNITS:
        p = f"{_name(*u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_grad_df_l_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--grad-df-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "gmetric_00.cu",
                                "geri3_000.cu"}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_df_gradient_unit(backend,
                                                     tmp_path) -> None:
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith(PREFIXES) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_df_gradient_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--grad-df-l" not in line


def test_the_names_cannot_collide_with_any_other_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--mu-l",
                "--schwarz-l", "--sx-l", "--metric-l", "--eri3-l",
                "--grad-l", "3") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIXES) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--grad-df-l") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 31 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--grad-df-l", "5"],
    ["--grad-df-l", "-1"],
    ["--grad-df-l", "x"],
    ["--grad-df-l", "1", "--class", "0", "0", "0", "0"],
    ["--grad-df-l", "1", "--shell-ls", "0"],
    ["--grad-df-l", "1", "--fock-l", "0"],
    ["--grad-df-l", "1", "--n-batches", "3"],
    ["--grad-df-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_grad_df_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_grad_df_l_above_l4_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--grad-df-l", "5") == 2
    err = capsys.readouterr().err
    assert "--grad-df-l 5 is outside 0..4" in err
    assert "validated at L <= 4 only" in err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_nvcc_compiles_the_df_gradient_unit(unit, nvcc, tmp_path) -> None:
    _compile_unit(nvcc, "cuda", unit, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", UNITS, ids=_id)
def test_hipcc_compiles_the_df_gradient_unit(unit, hipcc, tmp_path) -> None:
    _compile_unit(hipcc, "hip", unit, tmp_path)


def _compile_unit(tc, backend: str, unit, tmp_path: Path) -> None:
    """test_emit_grad.py's compile, on this file's kernel."""
    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = _kernel(*unit)
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


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_grad_df.py "
                 "--write")
    write_goldens()
