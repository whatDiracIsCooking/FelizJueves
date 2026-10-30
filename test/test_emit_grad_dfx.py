"""The uncontracted `(mn|P)` d1 kernel (`xeri3_`, #692) emitted for both
vendors: golden texts for the bare ladder, the `--grad-dfx-l` ladder, and a
compile of each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --grad-dfx-l [LMAX]
[--grad-dfx-aux-l AUXMAX]` writes `xeri3_{la}{lb}{lP}` for every `la <= lb
<= LMAX`, `lP <= AUXMAX` -- `--grad-df-l`'s three-center classes, its
limits (0..3, 0..6: through (ff|i), where the kernel is interpreted against
the host) and its defaults (1 and 2 bare: 9 units). It exits 2 above either,
is NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/xeri3_*.{cu,hip}`, the 9 units of
the bare ladder. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_grad_dfx.py --write

THE COMPILE. Each golden, and the top classes of the ladder ((dd|g), (ff|i),
(sf|i)), is compiled alone (`-c`) with the kernel tier's own flags and FAILS
on any warning located in the generated unit (#92's filter). Needs a
compiler, never a card. All 24 cases passed under CUDA 13.0's nvcc and
ROCm 7.2.4's hipcc on a device-less cloud container (2026-10-30).
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest
from test_emit_grad import _WARNING, BACKEND, FLAGS_VAR, GOLDEN, _names, _run

from felizjueves.emit import __main__ as emit_cli
from felizjueves.gradient.kernel_xeri3 import build_xeri3_kernel, xeri3_name
from felizjueves.ir import Raw, walk

UNITS = emit_cli.grad_dfx_classes(1, 2)  # the bare ladder
TOP = [(2, 2, 4), (3, 3, 6), (0, 3, 6)]
PREFIX = "xeri3_"


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_xeri3_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every `xeri3_` golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for c in UNITS:
            (GOLDEN / backend / f"{xeri3_name(c)}.{ext}").write_text(
                _text(c, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_OUT", "KERNEL_TIER_ARCH"):
        monkeypatch.delenv(v, raising=False)


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_the_df_gradients_three_center_classes() -> None:
    assert len(UNITS) == 9 and UNITS == emit_cli.grad_df_classes(1, 2)[1]
    assert emit_cli.grad_dfx_classes(0, 0) == [(0, 0, 0)]
    assert emit_cli.grad_dfx_classes(3, 6) == emit_cli.grad_df_classes(3, 6)[1]
    assert len(emit_cli.grad_dfx_classes(3, 6)) == 70


def test_the_ladder_refuses_above_f_and_i_in_its_own_flags() -> None:
    for bad in (-1, 4, 1.0, True):
        with pytest.raises(ValueError, match=r"--grad-dfx-l .* 0\.\.3"):
            emit_cli.grad_dfx_classes(bad)
    for bad in (-1, 7):
        with pytest.raises(ValueError, match=r"--grad-dfx-aux-l .* 0\.\.6"):
            emit_cli.grad_dfx_classes(1, bad)


def test_the_names_are_the_builders_and_no_raw_escape_hatch() -> None:
    for c in UNITS:
        k = build_xeri3_kernel(c)
        assert k.name == xeri3_name(c) == PREFIX + "".join(map(str, c))
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", UNITS, ids=xeri3_name)
def test_the_xeri3_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{xeri3_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_bare_ladder() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"{PREFIX}*.{ext}")}
        assert files == {xeri3_name(c) for c in UNITS}


def test_the_vendors_differ_only_in_the_prologue() -> None:
    c, h = _text((0, 1, 1), "cuda"), _text((0, 1, 1), "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_stores_are_plain_into_x_with_no_atomic_or_shared() -> None:
    for backend in BACKEND:
        text = _text((1, 1, 1), backend)
        assert "X[" in text
        assert "atomicAdd" not in text and "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_grad_dfx_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--grad-dfx-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{xeri3_name(c)}.{ext}" for c in UNITS}
    for c in UNITS:
        p = f"{xeri3_name(c)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_grad_dfx_aux_l_takes_the_aux_ladder_to_i(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--grad-dfx-l", "0",
                "--grad-dfx-aux-l", "6") == 0
    assert _names(tmp_path) - {"pq_L0_p0_q0.cu"} == {
        f"xeri3_00{p}.cu" for p in range(7)}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_xeri3_unit(backend, tmp_path) -> None:
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith(PREFIX) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_xeri3_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--grad-dfx-l" not in line


def test_the_names_cannot_collide_with_any_other_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--mu-l",
                "--schwarz-l", "--sx-l", "--metric-l", "--eri3-l",
                "--grad-l", "3", "--grad-k-l", "--grad-df-l",
                "--hess-eri3-l") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIX) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--grad-dfx-l") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 9 and not mine & others
    for taken in ("eri3_", "geri3_", "heri3_", "hgeri3_", "reri3_",
                  "d2eri3_"):
        assert not PREFIX.startswith(taken) and not taken.startswith(PREFIX)


@pytest.mark.parametrize("argv", [
    ["--grad-dfx-l", "4"],
    ["--grad-dfx-l", "-1"],
    ["--grad-dfx-l", "1", "--grad-dfx-aux-l", "7"],
    ["--grad-dfx-aux-l", "2"],
    ["--grad-dfx-l", "x"],
    ["--grad-dfx-l", "1", "--class", "0", "0", "0", "0"],
    ["--grad-dfx-l", "1", "--fock-l", "0"],
    ["--grad-dfx-l", "1", "--hgp-j-l", "1"],
],ids=lambda a: "_".join(a))
def test_a_grad_dfx_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_grad_dfx_l_above_f_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--grad-dfx-l", "4") == 2
    err = capsys.readouterr().err
    assert "--grad-dfx-l 4 is outside 0..3" in err
    assert "validated through orbital f only" in err
    assert _run(tmp_path / "out", "hip", "--grad-dfx-l", "--grad-dfx-aux-l",
                "7") == 2
    assert "validated through aux i only" in capsys.readouterr().err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("cls", UNITS + TOP, ids=xeri3_name)
def test_nvcc_compiles_the_xeri3_unit(cls, nvcc, tmp_path) -> None:
    _compile_unit(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", UNITS + TOP, ids=xeri3_name)
def test_hipcc_compiles_the_xeri3_unit(cls, hipcc, tmp_path) -> None:
    _compile_unit(hipcc, "hip", cls, tmp_path)


def _compile_unit(tc, backend: str, cls, tmp_path: Path) -> None:
    """test_emit_grad.py's compile, on this file's kernel."""
    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_xeri3_kernel(cls)
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
        sys.exit("usage: PYTHONPATH=src python test/test_emit_grad_dfx.py "
                 "--write")
    write_goldens()
