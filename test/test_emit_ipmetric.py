"""The uncontracted DF metric derivative (#693) emitted for both vendors:
golden texts for the bare ladder, the `--ip-metric-l` ladder through aux i,
and a compile of each class through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --ip-metric-l [LMAX]` writes
`ipmetric_{lP}{lQ}` for every `lP <= lQ <= LMAX` (2 when the flag is bare:
6 units). It accepts 0..6 (aux i, where the kernel is validated,
interpreted: test_kernel_ipmetric.py), exits 2 above that, is NEVER in the
bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/ipmetric_*.{cu,hip}`, the 6 units
of the bare flag. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_ipmetric.py --write

THE COMPILE. Every class of the ladder through (i|i), each alone (`-c`)
with the kernel tier's own flags, FAILING on any warning located in the
generated unit (#92's filter). Above `FAST_L_MAX` (`L = lP + lQ + 1`, the
order the tower reaches) a case is `slow`, and `tiers.sample_compiles`
keeps the first of its kind. Needs a compiler, never a card. All 56 ran
green in the device-less cloud container that wrote them (nvcc 13.0, hipcc
of ROCm 7.2.4, 2026-10-30); no unit has run on a card.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest
from test_emit_grad import _WARNING, BACKEND, FLAGS_VAR, GOLDEN, _names, _run
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.gradient.kernel_ipmetric import (
    build_ipmetric_kernel,
    ipmetric_name,
)
from felizjueves.ir import Raw, walk

CLASSES = emit_cli.ip_metric_classes(emit_cli.DEFAULT_IP_METRIC_L)
LADDER = emit_cli.ip_metric_classes(emit_cli.IP_METRIC_L_MAX)
PREFIX = "ipmetric_"


def _id(c) -> str:
    return ipmetric_name(c)


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_ipmetric_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every uncontracted metric derivative golden (module
    docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for c in CLASSES:
            (GOLDEN / backend / f"{ipmetric_name(c)}.{ext}").write_text(
                _text(c, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_OUT", "KERNEL_TIER_ARCH"):
        monkeypatch.delenv(v, raising=False)


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_class_through_lmax() -> None:
    assert len(CLASSES) == 6 and len(LADDER) == 28
    assert emit_cli.DEFAULT_IP_METRIC_L == 2
    assert emit_cli.IP_METRIC_L_MAX == 6
    assert emit_cli.ip_metric_classes(0) == [(0, 0)]
    # the aux ladder `--grad-df-l` emits gmetric_ for, the contraction's
    assert LADDER == emit_cli.grad_df_classes(3, 6)[0]


def test_the_ladder_refuses_above_aux_i() -> None:
    for bad in (-1, 7, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.6"):
            emit_cli.ip_metric_classes(bad)


def test_the_names_are_the_builders() -> None:
    for c in CLASSES:
        assert build_ipmetric_kernel(c).name == ipmetric_name(c) == (
            PREFIX + "".join(map(str, c)))


def test_no_raw_escape_hatch() -> None:
    for c in CLASSES:
        k = build_ipmetric_kernel(c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_id)
def test_the_ipmetric_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{ipmetric_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_bare_ladder() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"{PREFIX}*.{ext}")}
        assert files == {ipmetric_name(c) for c in CLASSES}


def test_the_vendors_differ_only_in_the_prologue() -> None:
    c, h = _text((1, 2), "cuda"), _text((1, 2), "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_no_atomic_and_no_shared_array() -> None:
    for backend in BACKEND:
        text = _text((1, 1), backend)
        assert "atomicAdd" not in text and "__shared__" not in text
        assert "M1[" in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_ip_metric_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--ip-metric-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{ipmetric_name(c)}.{ext}" for c in CLASSES}
    for c in CLASSES:
        p = f"{ipmetric_name(c)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_ip_metric_l_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--ip-metric-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "ipmetric_00.cu"}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_ipmetric_unit(backend, tmp_path) -> None:
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith(PREFIX) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_ipmetric_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--ip-metric-l" not in line


def test_the_names_cannot_collide_with_any_other_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--metric-l",
                "--hgp-metric-l", "--hgp-grad-metric-l", "--grad-df-l",
                "--hess-metric-l") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIX) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--ip-metric-l") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 6 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--ip-metric-l", "7"],
    ["--ip-metric-l", "-1"],
    ["--ip-metric-l", "x"],
    ["--ip-metric-l", "1", "--class", "0", "0", "0", "0"],
    ["--ip-metric-l", "1", "--shell-ls", "0"],
    ["--ip-metric-l", "1", "--fock-l", "0"],
    ["--ip-metric-l", "1", "--n-batches", "3"],
    ["--ip-metric-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_an_ip_metric_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_ip_metric_l_above_aux_i_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--ip-metric-l", "7") == 2
    err = capsys.readouterr().err
    assert "--ip-metric-l 7 is outside 0..6" in err
    assert "validated through aux i only" in err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c))
                                 for c in LADDER])
def test_nvcc_compiles_the_ipmetric_unit(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c))
                                 for c in LADDER])
def test_hipcc_compiles_the_ipmetric_unit(cls, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    """test_emit_grad.py's compile, on this file's kernel."""
    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_ipmetric_kernel(cls)
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
        sys.exit("usage: PYTHONPATH=src python test/test_emit_ipmetric.py "
                 "--write")
    write_goldens()
