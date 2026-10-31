"""The exchange-Hessian kernel emitted for both vendors: golden texts for
the s / p classes, digests of every other unit through `(ff|ff)`, the
`--hess-k-l` ladder, and a compile of each unit through nvcc (sm_86) and
hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hess-k-l [LMAX]` writes
`hkx_{la}{lc}_{lb}{ld}` (`hessian.kernel_hkx`) for every K class, the
Hermite pairing, each `l <= LMAX` (1 when the flag is bare: 16 units), in
its DEFAULT form (unrolled below `LOWSTACK_MIN_L = 3`, rolled from it,
#691). It accepts 0..3 (validated through `(ff|ff)`), exits 2 above that,
is NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/hkx_{la}{lc}_{lb}{ld}.{cu,hip}`
for the 11 s / p classes below the floor (unrolled, their text unchanged
by #691) and for `ROLLED_GOLDEN`, one rolled unit to review; every other
default unit through `(ff|ff)` is pinned by size and sha256 in
`test/golden/hkx_digests.json` (`slow` above `L + 2 = 4`, `tiers.by_l`),
and so is the UNROLLED text of the five s / p classes from the floor,
under `<name>.unrolled` (the pre-#691 default, its digests unchanged).
Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hess_k.py --write

THE COMPILE is test_emit_grad.py's: each unit alone (`-c`), the kernel
tier's flags, FAILING on a warning located in the generated unit (#92),
killed at `COMPILE_BUDGET_S = 120` s a unit (data/measured/hkx.json's
declared budget). Needs a compiler, never a card. nvcc 13.0 and hipcc
7.2.4 compiled every default unit through (ff|ff) (2026-10-31, #691;
data/measured/hkx.json).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_emit_grad import BACKEND, GOLDEN, _run, compile_kernel
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.hessian.kernel_hkx import (
    LOWSTACK_MIN_L,
    build_hkx_kernel,
    hkx_lowstack,
    hkx_name,
)
from felizjueves.ir import Raw, walk

CLASSES = emit_cli.hess_k_classes(3)
SP = emit_cli.hess_k_classes(1)
ROLLED_GOLDEN = (1, 1, 1, 0)
GOLDEN_CLASSES = [c for c in SP if not hkx_lowstack(c)] + [ROLLED_GOLDEN]
DIGESTED = [c for c in CLASSES if c not in GOLDEN_CLASSES]
UNROLLED_PINNED = [c for c in SP if hkx_lowstack(c)]
DIGESTS = GOLDEN / "hkx_digests.json"
COMPILE_BUDGET_S = 120.0  # data/measured/hkx.json's declared budget


def _id(c) -> str:
    return hkx_name(c)


def _text(cls, backend: str, lowstack: bool | None = None) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_hkx_kernel(cls, lowstack=lowstack), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_goldens() -> None:
    """Rewrite every exchange-Hessian golden and the digest file (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        for c in GOLDEN_CLASSES:
            (GOLDEN / backend / f"{hkx_name(c)}.{ext}").write_text(
                _text(c, backend))
        digests[backend] = {hkx_name(c): _digest(_text(c, backend))
                            for c in DIGESTED}
        digests[backend] |= {
            f"{hkx_name(c)}.unrolled": _digest(_text(c, backend, False))
            for c in UNROLLED_PINNED}
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def _param(c):
    return by_l(c, sum(c) + 2, id=hkx_name(c))


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_k_class_through_lmax() -> None:
    assert len(SP) == 16 and SP[0] == (0, 0, 0, 0) and SP[-1] == (1, 1, 1, 1)
    assert len(CLASSES) == 256 and CLASSES[-1] == (3, 3, 3, 3)
    assert emit_cli.DEFAULT_HESS_K_L == 1 and emit_cli.HESS_K_L_MAX == 3
    assert emit_cli.hess_k_classes(0) == [(0, 0, 0, 0)]


def test_the_ladder_refuses_above_ff() -> None:
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            emit_cli.hess_k_classes(bad)


def test_the_floor_splits_the_s_p_classes() -> None:
    assert LOWSTACK_MIN_L == 3
    assert [c for c in SP if hkx_lowstack(c)] == [
        c for c in SP if sum(c) >= 3]
    assert all(hkx_lowstack(c) for c in CLASSES if max(c) > 1)


def test_the_names_are_the_builders() -> None:
    for c in SP:
        assert build_hkx_kernel(c).name == hkx_name(c) == (
            f"hkx_{c[0]}{c[1]}_{c[2]}{c[3]}")


def test_no_raw_escape_hatch() -> None:
    for c in SP:
        k = build_hkx_kernel(c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", GOLDEN_CLASSES, ids=_id)
def test_the_exchange_hessian_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{hkx_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", [_param(c) for c in DIGESTED])
def test_the_exchange_hessian_digest_is_unchanged(cls, backend: str) -> None:
    pinned = json.loads(DIGESTS.read_text())[backend]
    assert _digest(_text(cls, backend)) == pinned[hkx_name(cls)]


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", [_param(c) for c in UNROLLED_PINNED])
def test_the_unrolled_text_from_the_floor_is_pinned(cls, backend) -> None:
    pinned = json.loads(DIGESTS.read_text())[backend]
    assert _digest(_text(cls, backend, False)) == pinned[
        f"{hkx_name(cls)}.unrolled"]


def test_the_golden_set_is_exactly_the_recipe() -> None:
    pinned = json.loads(DIGESTS.read_text())
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"hkx_*.{ext}")}
        assert files == {hkx_name(c) for c in GOLDEN_CLASSES}
        assert set(pinned[backend]) == {hkx_name(c) for c in DIGESTED} | {
            f"{hkx_name(c)}.unrolled" for c in UNROLLED_PINNED}


def test_the_vendors_differ_only_in_the_prologue_and_the_lane_mask() -> None:
    # the warp reduction's mask is 32-bit on CUDA, 64-bit on HIP
    for cls in ((1, 0, 0, 1), ROLLED_GOLDEN):
        c, h = _text(cls, "cuda"), _text(cls, "hip")
        assert c.split("\n")[2:] == h.replace(
            "0xffffffffull", "0xffffffffu").split("\n")[3:]
        assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        for cls in ((1, 1, 0, 1), ROLLED_GOLDEN):
            text = _text(cls, backend)
            assert "atomicAdd(&H[" in text
            assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hess_k_l_bare_writes_the_s_p_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hess-k-l") == 0
    mine = {f"{hkx_name(c)}.{ext}" for c in SP}
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | mine
    pinned = json.loads(DIGESTS.read_text())[backend]
    for c in SP:
        text = (tmp_path / f"{hkx_name(c)}.{ext}").read_text()
        if c in GOLDEN_CLASSES:
            assert text == (GOLDEN / backend / f"{hkx_name(c)}.{ext}"
                            ).read_text()
        else:
            assert _digest(text) == pinned[hkx_name(c)]


def test_hess_k_l_adds_its_units_beside_the_other_ladders(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hess-l", "0",
                "--grad-k-l", "0", "--hess-k-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "hdq_00.cu", "hjab_00.cu",
                                "gk_00_00.cu", "hkx_00_00.cu"}


def test_the_bare_default_writes_no_exchange_hessian_unit(tmp_path) -> None:
    assert _run(tmp_path, "cuda") == 0
    assert not any(n.startswith("hkx_") for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_exchange_hessian_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--hess-k-l" not in line


@pytest.mark.parametrize("argv", [
    ["--hess-k-l", "4"],
    ["--hess-k-l", "-1"],
    ["--hess-k-l", "x"],
    ["--hess-k-l", "1", "--class", "0", "0", "0", "0"],
    ["--hess-k-l", "1", "--shell-ls", "0"],
    ["--hess-k-l", "1", "--fock-l", "0"],
    ["--hess-k-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--hess-k-l", "1", "--n-batches", "3"],
    ["--hess-k-l", "1", "--hgp-k-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_hess_k_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_hess_k_l_above_ff_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hess-k-l", "4") == 2
    err = capsys.readouterr().err
    assert "--hess-k-l 4 is outside 0..3" in err
    assert "validated through (ff|ff) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    compile_kernel(tc, backend, build_hkx_kernel(cls), tmp_path,
                   timeout=COMPILE_BUDGET_S)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES])
def test_nvcc_compiles_the_exchange_hessian_unit(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES])
def test_hipcc_compiles_the_exchange_hessian_unit(cls, hipcc,
                                                  tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hess_k.py "
                 "--write")
    write_goldens()
