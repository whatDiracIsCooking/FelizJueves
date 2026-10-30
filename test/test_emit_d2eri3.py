"""The three-center DF Hessian kernel emitted for both vendors: golden texts
for every class at L <= 2, digests for every other class of the ladder
through (ff|i), the `--hess-eri3-l` ladder, and a compile of each through
nvcc (sm_86) and hipcc (gfx1200).

THE LADDER (#694: split as `--grad-df-l` is). `python -m felizjueves.emit
--hess-eri3-l [LMAX] [--hess-eri3-aux-l AUXMAX]` writes
`d2eri3_{la}{lb}{lP}` (`hessian.kernel_d2eri3`) for every `la <= lb <=
LMAX`, `lP <= AUXMAX` (1 and 2 when the flags are bare or AUXMAX omitted: 9
units). It accepts LMAX 0..3 and AUXMAX 0..6 (every class through (ff|i),
interpreted against the host, test_kernel_d2_df_ladder.py), exits 2 above
either and for `--hess-eri3-aux-l` alone, is NEVER in the bare default and
is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/d2eri3_*.{cu,hip}` for the seven
classes with `la + lb + lP <= FILE_L_MAX = 2` (the largest 206 KB); every
other class -- the 22 of the old total-L ladder (`la + lb + lP <= 4`, `lb`
to g) and the 70 of the new one, 71 in all -- is pinned instead by its byte count and
sha256 in `test/golden/d2eri3_digests.json`, in its DEFAULT form (the
low-stack one from `kernel_d2eri3.LOWSTACK_MIN_L`). Each class at or above
that floor that the old ladder pinned unrolled keeps that digest too, under
`<name>.unrolled`, so the unrolled text is pinned unchanged. Regenerate
(and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_d2eri3.py --write

THE COMPILE is test_emit_grad.py's: each unit alone (`-c`), the kernel
tier's flags, FAILING on a warning located in the generated unit (#92).
Needs a compiler, never a card. nvcc 13.0 (sm_86) compiled every pinned
class on 2026-10-30 (#694, a device-less container); hipcc has not been run
on these units.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_emit_grad import BACKEND, GOLDEN, _names, _run, compile_kernel
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.hessian.kernel_d2eri3 import (
    LOWSTACK_MIN_L,
    build_d2eri3_kernel,
    d2eri3_name,
)
from felizjueves.ir import Raw, walk

# the old total-L ladder's classes (`la + lb + lP <= 4`, `lb` to g), pinned
# before #694, and the new ladder's
OLD = [(a, b, p) for a in range(5) for b in range(a, 5) for p in range(5)
       if a + b + p <= 4]
LADDER = emit_cli.hess_eri3_classes(3, 6)
CLASSES = sorted(set(OLD) | set(LADDER))
BARE = emit_cli.hess_eri3_classes(emit_cli.DEFAULT_HESS_ERI3_L)
UNROLLED = [c for c in OLD if sum(c) >= LOWSTACK_MIN_L]  # `.unrolled` too
DIGESTS = GOLDEN / "d2eri3_digests.json"
FILE_L_MAX = 2  # classes above it are pinned by digest (module docstring)
PREFIX = "d2eri3_"


def _id(c) -> str:
    return d2eri3_name(c)


def _text(cls, backend: str, lowstack: bool | None = None) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_d2eri3_kernel(cls, lowstack=lowstack), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _as_file(cls) -> bool:
    return sum(cls) <= FILE_L_MAX


def write_goldens() -> None:
    """Rewrite every DF Hessian golden and the digest file (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for c in CLASSES:
            text = _text(c, backend)
            path = GOLDEN / backend / f"{d2eri3_name(c)}.{ext}"
            if _as_file(c):
                path.write_text(text)
            else:
                digests[backend][d2eri3_name(c)] = _digest(text)
                path.unlink(missing_ok=True)
        for c in UNROLLED:
            digests[backend][d2eri3_name(c) + ".unrolled"] = _digest(
                _text(c, backend, lowstack=False))
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


def _pinned(cls, backend: str, text: str) -> None:
    if _as_file(cls):
        path = GOLDEN / backend / f"{d2eri3_name(cls)}.{BACKEND[backend][2]}"
        assert text == path.read_text()
    else:
        pinned = json.loads(DIGESTS.read_text())[backend]
        assert _digest(text) == pinned[d2eri3_name(cls)]


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_OUT", "KERNEL_TIER_ARCH"):
        monkeypatch.delenv(v, raising=False)


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_class_through_lmax() -> None:
    assert len(OLD) == 22 and len(LADDER) == 70 and len(BARE) == 9
    assert len(CLASSES) == 70 + 1  # (sg|s): lb = 4 is the old ladder's only
    assert emit_cli.DEFAULT_HESS_ERI3_L == 1 and emit_cli.HESS_ERI3_L_MAX == 3
    assert (emit_cli.DEFAULT_HESS_ERI3_AUX_L == 2
            and emit_cli.HESS_ERI3_AUX_L_MAX == 6)
    assert emit_cli.hess_eri3_classes(0, 0) == [(0, 0, 0)]
    # the d1 DF ladder's three-center classes (#613)
    assert LADDER == emit_cli.grad_df_classes(3, 6)[1]
    assert BARE == [(a, b, p) for a in range(2) for b in range(a, 2)
                    for p in range(3)]


def test_the_ladder_refuses_above_f_and_i() -> None:
    for bad in (-1, 4, 1.0, True):
        with pytest.raises(ValueError, match=r"--hess-eri3-l .* 0\.\.3"):
            emit_cli.hess_eri3_classes(bad)
    for bad in (-1, 7, 1.0, True):
        with pytest.raises(ValueError, match=r"--hess-eri3-aux-l .* 0\.\.6"):
            emit_cli.hess_eri3_classes(1, bad)


def test_the_names_are_the_builders() -> None:
    for c in CLASSES:
        assert build_d2eri3_kernel(c).name == d2eri3_name(c) == (
            PREFIX + "".join(map(str, c)))


def test_no_raw_escape_hatch() -> None:
    for c in BARE:
        k = build_d2eri3_kernel(c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


def _gparam(c):
    return by_l(c, sum(c), id=_id(c))


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", [_gparam(c) for c in CLASSES])
def test_the_df_hessian_golden_is_unchanged(cls, backend: str) -> None:
    _pinned(cls, backend, _text(cls, backend))


@pytest.mark.slow  # builds the unrolled text, ~1.25 MB at (pd|p)
@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", UNROLLED, ids=_id)
def test_the_unrolled_text_above_the_floor_is_unchanged(cls, backend) -> None:
    pinned = json.loads(DIGESTS.read_text())[backend]
    assert _digest(_text(cls, backend, lowstack=False)) == pinned[
        d2eri3_name(cls) + ".unrolled"]


def test_the_golden_set_is_exactly_the_ladder() -> None:
    pinned = json.loads(DIGESTS.read_text())
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"{PREFIX}*.{ext}")}
        assert files == {d2eri3_name(c) for c in CLASSES if _as_file(c)}
        assert set(pinned[backend]) == {d2eri3_name(c) for c in CLASSES
                                        if not _as_file(c)} | {
            d2eri3_name(c) + ".unrolled" for c in UNROLLED}


def test_the_vendors_differ_only_in_the_prologue() -> None:
    c, h = _text((0, 1, 1), "cuda"), _text((0, 1, 1), "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_every_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        text = _text((0, 1, 1), backend)
        assert text.count("atomicAdd(&H[") == 81
        assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hess_eri3_l_bare_writes_the_pinned_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hess-eri3-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{d2eri3_name(c)}.{ext}" for c in BARE}
    for c in BARE:
        _pinned(c, backend,
                (tmp_path / f"{d2eri3_name(c)}.{ext}").read_text())


def test_hess_eri3_l_adds_its_units_beside_the_other_ladders(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--grad-df-l", "0",
                "--grad-df-aux-l", "0", "--hess-eri3-l", "0",
                "--hess-eri3-aux-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "gmetric_00.cu",
                                "geri3_000.cu", "d2eri3_000.cu"}


def test_the_bare_default_writes_no_df_hessian_unit(tmp_path) -> None:
    assert _run(tmp_path, "cuda") == 0
    assert not any(n.startswith(PREFIX) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_df_hessian_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--hess-eri3-l" not in line


def test_the_names_cannot_collide_with_the_hgp_df_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--hgp-metric-l", "--hgp-eri3-l",
                "--eri3-l", "--grad-df-l", "--hess-one-e-l") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIX) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--hess-eri3-l", "1") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 3 * 3 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--hess-eri3-l", "4"],
    ["--hess-eri3-l", "1", "--hess-eri3-aux-l", "7"],
    ["--hess-eri3-aux-l", "1"],
    ["--hess-eri3-l", "-1"],
    ["--hess-eri3-l", "x"],
    ["--hess-eri3-l", "1", "--class", "0", "0", "0", "0"],
    ["--hess-eri3-l", "1", "--shell-ls", "0"],
    ["--hess-eri3-l", "1", "--fock-l", "0"],
    ["--hess-eri3-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--hess-eri3-l", "1", "--n-batches", "3"],
    ["--hess-eri3-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_hess_eri3_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_hess_eri3_l_above_f_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hess-eri3-l", "4") == 2
    err = capsys.readouterr().err
    assert "--hess-eri3-l 4 is outside 0..3" in err
    assert "validated through orbital f only" in err


def test_hess_eri3_aux_l_above_i_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hess-eri3-l", "1",
                "--hess-eri3-aux-l", "7") == 2
    err = capsys.readouterr().err
    assert "--hess-eri3-aux-l 7 is outside 0..6" in err
    assert "validated through aux i only" in err


def test_hess_eri3_aux_l_alone_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "cuda", "--hess-eri3-aux-l", "1") == 2
    assert "needs --hess-eri3-l" in capsys.readouterr().err


def test_the_aux_flag_raises_the_aux_limit(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hess-eri3-l", "0",
                "--hess-eri3-aux-l", "1") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "d2eri3_000.cu",
                                "d2eri3_001.cu"}


# --- the compile (a compiler, no card) ----------------------------------------


def _cparam(c):
    return by_l(c, sum(c), id=_id(c))


@pytest.mark.cuda
@pytest.mark.parametrize("cls", [_cparam(c) for c in CLASSES])
def test_nvcc_compiles_the_df_hessian_unit(cls, nvcc, tmp_path: Path) -> None:
    compile_kernel(nvcc, "cuda", build_d2eri3_kernel(cls), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", [_cparam(c) for c in CLASSES])
def test_hipcc_compiles_the_df_hessian_unit(cls, hipcc,
                                            tmp_path: Path) -> None:
    compile_kernel(hipcc, "hip", build_d2eri3_kernel(cls), tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_d2eri3.py "
                 "--write")
    write_goldens()
