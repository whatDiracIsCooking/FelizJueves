"""The Coulomb-Hessian kernels emitted for both vendors: golden texts for
the pair classes (ss), (sp) and (pp), digests of every other ladder unit
through (ff), the `--hess-l` ladder, and a compile of each unit through
nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hess-l [LMAX]` writes, for every
`la <= lb <= LMAX` (1 when the flag is bare), `hdq_{la}{lb}`
(`kernel_hdq`) and `hjab_{la}{lb}` (`kernel_hjab`), each in its DEFAULT
form (#690: `hjab` rolled from `la + lb = 3`, `hdq` from `lc + ld = 5`).
It accepts 0..3 (validated through (ff)), exits 2 above that, is NEVER in
the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{hdq,hjab}_{la}{lb}.{cu,hip}` for
the three classes (ss) (sp) (pp), 12 files, all below both floors, so
their text is unchanged by #690. Every other ladder unit through (ff) is
pinned by size and sha256 in `test/golden/hjab_digests.json` (`slow`
above the cut, `tiers.by_l`), and so is the UNROLLED text of `hjab_12`
and `hjab_22`, the pre-#690 default that ran on both cards, as
`<name>.unrolled`. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hess.py --write

THE COMPILE. Each ladder unit is compiled alone (`-c`) with the kernel
tier's own flags (devtools/config.sh `NVCC_FLAGS` / `HIPCC_FLAGS`,
warnings on) and FAILS on any warning located in the generated unit
(#92's filter), through test_emit_grad.py's `compile_kernel`, killed at
`COMPILE_BUDGET_S = 120` s (data/measured/hjab.json's declared budget).
Needs a compiler, never a card. nvcc 13.0 and hipcc 7.2.4 compiled every
unit (data/measured/hjab.json, #690).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_emit_grad import compile_kernel
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.hessian.kernel_hdq import (
    build_hdq_kernel,
    hdq_lowstack,
    hdq_name,
)
from felizjueves.hessian.kernel_hjab import (
    build_hjab_kernel,
    hjab_lowstack,
    hjab_name,
)
from felizjueves.ir import Raw, walk
from felizjueves.performance.device import GFX1200, SM_86

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = [(0, 0), (0, 1), (1, 1)]
FAMILIES = ("hdq", "hjab")
PREFIXES = ("hdq_", "hjab_")
_BUILD = {"hdq": build_hdq_kernel, "hjab": build_hjab_kernel}
_NAME = {"hdq": hdq_name, "hjab": hjab_name}
UNITS = [(f, c) for c in CLASSES for f in FAMILIES]
LADDER = [(f, c) for c in emit_cli.hess_classes(3) for f in FAMILIES]
DIGESTED = [u for u in LADDER if u not in UNITS]
UNROLLED_PINNED = [("hjab", (1, 2)), ("hjab", (2, 2))]
DIGESTS = GOLDEN / "hjab_digests.json"
COMPILE_BUDGET_S = 120.0  # data/measured/hjab.json's declared budget
# the Hermite order each unit's kernel reaches: hdq the ket one up, hjab
# the bra two up
_REACH = {"hdq": 1, "hjab": 2}


def _kernel(family: str, cls, lowstack: bool | None = None):
    return _BUILD[family](cls, lowstack=lowstack)


def _name(family: str, cls) -> str:
    return _NAME[family](cls)


def _id(u) -> str:
    return _name(*u)


def _text(family: str, cls, backend: str,
          lowstack: bool | None = None) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(_kernel(family, cls, lowstack), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _param(u):
    return by_l(u, sum(u[1]) + _REACH[u[0]], id=_id(u))


def write_goldens() -> None:
    """Rewrite every Hessian golden and the digest file (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        for f, c in UNITS:
            (GOLDEN / backend / f"{_name(f, c)}.{ext}").write_text(
                _text(f, c, backend))
        digests[backend] = {_name(*u): _digest(_text(*u, backend))
                            for u in DIGESTED}
        digests[backend] |= {
            f"{_name(*u)}.unrolled": _digest(_text(*u, backend, False))
            for u in UNROLLED_PINNED}
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_pair_class_through_lmax() -> None:
    assert emit_cli.hess_classes(1) == CLASSES
    assert emit_cli.DEFAULT_HESS_L == 1 and emit_cli.HESS_L_MAX == 3
    assert emit_cli.hess_classes(0) == [(0, 0)]
    assert len(emit_cli.hess_classes(2)) == 6
    assert len(emit_cli.hess_classes(3)) == 10


def test_the_ladder_refuses_above_ff() -> None:
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            emit_cli.hess_classes(bad)


def test_the_floors_split_the_ladder() -> None:
    # #690: hjab rolled from la + lb = 3, hdq from lc + ld = 5
    for c in emit_cli.hess_classes(3):
        assert hjab_lowstack(c) == (sum(c) >= 3), c
        assert hdq_lowstack(c) == (sum(c) >= 5), c
    assert not any(hjab_lowstack(c) or hdq_lowstack(c) for c in CLASSES)


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
def test_the_hessian_golden_is_unchanged(unit, backend: str) -> None:
    path = GOLDEN / backend / f"{_name(*unit)}.{BACKEND[backend][2]}"
    assert _text(*unit, backend) == path.read_text()


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", [_param(u) for u in DIGESTED])
def test_the_hessian_digest_is_unchanged(unit, backend: str) -> None:
    pinned = json.loads(DIGESTS.read_text())[backend]
    assert _digest(_text(*unit, backend)) == pinned[_name(*unit)]


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("unit", [_param(u) for u in UNROLLED_PINNED])
def test_the_unrolled_text_from_the_floor_is_pinned(unit, backend) -> None:
    pinned = json.loads(DIGESTS.read_text())[backend]
    assert _digest(_text(*unit, backend, False)) == pinned[
        f"{_name(*unit)}.unrolled"]


def test_the_golden_set_is_exactly_the_three_classes() -> None:
    pinned = json.loads(DIGESTS.read_text())
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for f in FAMILIES
                 for p in (GOLDEN / backend).glob(f"{f}_[0-9]*.{ext}")}
        assert files == {_name(*u) for u in UNITS}
        assert set(pinned[backend]) == {_name(*u) for u in DIGESTED} | {
            f"{_name(*u)}.unrolled" for u in UNROLLED_PINNED}


def test_the_rolled_forms_read_no_named_density() -> None:
    for backend in BACKEND:
        hj = _text("hjab", (1, 2), backend)
        hd = _text("hdq", (2, 3), backend)
        assert "ROLLED" in hj and "ROLLED" in hd
        assert "HAA00_0" not in hj and "HA0_0" not in hd
        assert "atomicAdd(&H[" in hj and "atomicAdd" not in hd


@pytest.mark.parametrize("unit", [("hdq", (0, 1)), ("hjab", (1, 1))],
                         ids=_id)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(*unit, "cuda"), _text(*unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_hjab_adds_atomically_and_hdq_stores_plainly() -> None:
    for backend in BACKEND:
        hj, hd = _text("hjab", (0, 1), backend), _text("hdq", (0, 1), backend)
        assert "atomicAdd(&H[" in hj and "atomicAdd" not in hd
        assert "D1[" in hd
        assert "__shared__" not in hj + hd


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hess_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hess-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(*u)}.{ext}" for u in UNITS}
    for u in UNITS:
        p = f"{_name(*u)}.{ext}"
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_hess_l_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hess-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "hdq_00.cu", "hjab_00.cu"}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_the_bare_default_writes_no_hessian_unit(backend, tmp_path) -> None:
    assert _run(tmp_path, backend) == 0
    assert not any(n.startswith(PREFIXES) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_hessian_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--hess-l" not in line


def test_the_hessian_names_cannot_collide_with_any_other_ladder(
        tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--mu-l",
                "--schwarz-l", "--sx-l", "--metric-l", "--eri3-l",
                "--grad-l", "--grad-k-l") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIXES) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--hess-l", "1") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 6 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--hess-l", "4"],
    ["--hess-l", "-1"],
    ["--hess-l", "x"],
    ["--hess-l", "1", "--class", "0", "0", "0", "0"],
    ["--hess-l", "1", "--shell-ls", "0"],
    ["--hess-l", "1", "--fock-l", "0"],
    ["--hess-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--hess-l", "1", "--n-batches", "3"],
    ["--hess-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_hess_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_hess_l_above_ff_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hess-l", "4") == 2
    err = capsys.readouterr().err
    assert "--hess-l 4 is outside 0..3" in err
    assert "validated through (ff) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("unit", [_param(u) for u in LADDER])
def test_nvcc_compiles_the_hessian_unit(unit, nvcc, tmp_path) -> None:
    compile_kernel(nvcc, "cuda", _kernel(*unit), tmp_path,
                   timeout=COMPILE_BUDGET_S)


@pytest.mark.hip
@pytest.mark.parametrize("unit", [_param(u) for u in LADDER])
def test_hipcc_compiles_the_hessian_unit(unit, hipcc, tmp_path) -> None:
    compile_kernel(hipcc, "hip", _kernel(*unit), tmp_path,
                   timeout=COMPILE_BUDGET_S)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hess.py --write")
    write_goldens()
