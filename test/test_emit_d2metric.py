"""The DF metric Hessian kernels emitted for both vendors: golden texts for
every class at L <= 4, digests for the rest of the ladder through (i|i),
the `--hess-metric-l` ladder, and a compile of each through nvcc (sm_86)
and hipcc (gfx1200).

THE LADDER (#694: the aux `l` per shell, as `--grad-df-aux-l`). `python
-m felizjueves.emit --hess-metric-l [AUXMAX]` writes `d2metric_{lP}{lQ}`
for every `lP <= lQ <= AUXMAX` (2 when the flag is bare: 6 units). It
accepts 0..6 (every class through (i|i), interpreted against the host,
test_kernel_d2_df_ladder.py), exits 2 above that, is NEVER in the bare
default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/d2metric_*.{cu,hip}` for the 9
classes with `lP + lQ <= FILE_L_MAX = 4` (the largest ~36 KB), unchanged by
#694; every other class of the ladder, through (i|i), in its DEFAULT form
(the low-stack one from `kernel_d2metric.LOWSTACK_MIN_L` = 9), is pinned by
its byte count and sha256 in `test/golden/d2metric_digests.json`.
Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_d2metric.py --write

THE COMPILE. Each unit is compiled alone (`-c`) with the kernel tier's own
flags and FAILS on any warning located in the generated unit (#92's
filter). Needs a compiler, never a card. nvcc 13.0 (sm_86) compiled every
class of the ladder on 2026-10-30 (#694, a device-less container), and
hipcc 7.2.4 (gfx1200) the same day.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

import pytest
from test_emit_grad import _WARNING, BACKEND, FLAGS_VAR, GOLDEN, _names, _run
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.hessian.kernel_d2metric import (
    build_d2metric_kernel,
    d2metric_name,
)
from felizjueves.ir import Raw, walk

LADDER = emit_cli.hess_metric_classes(6)
FILE_L_MAX = 4  # classes above it are pinned by digest (module docstring)
CLASSES = [c for c in LADDER if sum(c) <= FILE_L_MAX]
BARE = emit_cli.hess_metric_classes(emit_cli.DEFAULT_HESS_METRIC_L)
DIGESTS = GOLDEN / "d2metric_digests.json"
PREFIX = "d2metric_"


def _id(c) -> str:
    return d2metric_name(c)


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_d2metric_kernel(cls), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_goldens() -> None:
    """Rewrite every DF metric Hessian golden and the digest file (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for c in LADDER:
            text = _text(c, backend)
            path = GOLDEN / backend / f"{d2metric_name(c)}.{ext}"
            if c in CLASSES:
                path.write_text(text)
            else:
                digests[backend][d2metric_name(c)] = _digest(text)
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


def _pinned(cls, backend: str, text: str) -> None:
    if cls in CLASSES:
        path = GOLDEN / backend / f"{d2metric_name(cls)}.{BACKEND[backend][2]}"
        assert text == path.read_text()
    else:
        pinned = json.loads(DIGESTS.read_text())[backend]
        assert _digest(text) == pinned[d2metric_name(cls)]


def _lparam(c):
    return by_l(c, sum(c), id=_id(c))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_OUT", "KERNEL_TIER_ARCH"):
        monkeypatch.delenv(v, raising=False)


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_class_through_auxmax() -> None:
    assert len(CLASSES) == 9 and len(LADDER) == 28 and len(BARE) == 6
    assert emit_cli.DEFAULT_HESS_METRIC_L == 2
    assert emit_cli.HESS_METRIC_L_MAX == 6
    assert emit_cli.hess_metric_classes(0) == [(0, 0)]
    # the d1 DF ladder's metric classes, aux l <= 6 (#613)
    assert LADDER == emit_cli.grad_df_classes(3, 6)[0]
    assert BARE == [(p, q) for p in range(3) for q in range(p, 3)]


def test_the_ladder_refuses_above_i() -> None:
    for bad in (-1, 7, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.6"):
            emit_cli.hess_metric_classes(bad)


def test_the_names_are_the_builders() -> None:
    for c in LADDER:
        assert build_d2metric_kernel(c).name == d2metric_name(c) == (
            PREFIX + "".join(map(str, c)))


def test_no_raw_escape_hatch() -> None:
    for c in LADDER:
        k = build_d2metric_kernel(c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", [_lparam(c) for c in LADDER])
def test_the_metric_hessian_golden_is_unchanged(cls, backend: str) -> None:
    _pinned(cls, backend, _text(cls, backend))


def test_the_golden_set_is_exactly_the_ladder() -> None:
    pinned = json.loads(DIGESTS.read_text())
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"{PREFIX}*.{ext}")}
        assert files == {d2metric_name(c) for c in CLASSES}
        assert set(pinned[backend]) == {d2metric_name(c) for c in LADDER
                                        if c not in CLASSES}


def test_the_vendors_differ_only_in_the_prologue() -> None:
    c, h = _text((1, 2), "cuda"), _text((1, 2), "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_every_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        text = _text((1, 1), backend)
        assert text.count("atomicAdd(&H[") == 36
        assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hess_metric_l_bare_writes_the_golden_units(backend,
                                                    tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hess-metric-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{d2metric_name(c)}.{ext}" for c in BARE}
    for c in BARE:
        p = f"{d2metric_name(c)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_hess_metric_l_adds_its_units_beside_the_ladder_flags(
        tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hess-metric-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "d2metric_00.cu"}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_metric_hessian_unit(backend,
                                                        tmp_path) -> None:
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith(PREFIX) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_metric_hessian_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--hess-metric-l" not in line


def test_the_names_cannot_collide_with_any_other_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--metric-l",
                "--hgp-metric-l", "--grad-df-l", "--hess-l",
                "--hess-one-e-l") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIX) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--hess-metric-l") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 6 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--hess-metric-l", "7"],
    ["--hess-metric-l", "-1"],
    ["--hess-metric-l", "x"],
    ["--hess-metric-l", "1", "--class", "0", "0", "0", "0"],
    ["--hess-metric-l", "1", "--shell-ls", "0"],
    ["--hess-metric-l", "1", "--fock-l", "0"],
    ["--hess-metric-l", "1", "--n-batches", "3"],
    ["--hess-metric-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_hess_metric_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_hess_metric_l_above_i_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hess-metric-l", "7") == 2
    err = capsys.readouterr().err
    assert "--hess-metric-l 7 is outside 0..6" in err
    assert "validated through aux i only" in err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("cls", [_lparam(c) for c in LADDER])
def test_nvcc_compiles_the_metric_hessian_unit(cls, nvcc, tmp_path) -> None:
    _compile_unit(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", [_lparam(c) for c in LADDER])
def test_hipcc_compiles_the_metric_hessian_unit(cls, hipcc, tmp_path) -> None:
    _compile_unit(hipcc, "hip", cls, tmp_path)


def _compile_unit(tc, backend: str, cls, tmp_path: Path) -> None:
    """test_emit_grad.py's compile, on this file's kernel."""
    import devcache

    from felizjueves.performance import measured

    emit, arch, ext = BACKEND[backend]
    k = build_d2metric_kernel(cls)
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
        sys.exit("usage: PYTHONPATH=src python test/test_emit_d2metric.py "
                 "--write")
    write_goldens()
