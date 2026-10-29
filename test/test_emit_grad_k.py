"""The exchange-gradient kernel emitted for both vendors: golden texts for
every K class with every shell s or p and for the four f classes at
`L + 1 = 4`, digests of every other default unit through `(ff|ff)`, the
`--grad-k-l` ladder, and a compile of each golden through nvcc (sm_86) and
hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --grad-k-l [LMAX]` writes
`gk_{la}{lc}_{lb}{ld}` (`gradient.kernel_gk`) for every K class, the
Hermite pairing, each `l <= LMAX` (1 when the flag is bare: 16 units),
each in its DEFAULT form (the rolled form from `L = LOWSTACK_MIN_L` and
for every class with an f shell, #618). It accepts 0..3 (validated
through `(ff|ff)`), exits 2 above that, is NEVER in the bare default and
is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/gk_{la}{lc}_{lb}{ld}.{cu,hip}` for
the 16 s / p classes (the unrolled form, unchanged by #618) and the four
classes with one f shell and nothing else (`L + 1 = 4`, the rolled form).
Every other default unit through `(ff|ff)` is pinned by size and sha256 in
`test/golden/gk_digests.json` (`slow` above `L + 1 = 4`, `tiers.by_l`).
Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_grad_k.py --write

THE COMPILE is test_emit_grad.py's: each golden alone (`-c`), the kernel
tier's flags, FAILING on a warning located in the generated unit (#92),
under the user's #618 budget of `COMPILE_BUDGET_S = 10` s a unit. Needs a
compiler, never a card.
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
from felizjueves.gradient.kernel_gk import build_gk_kernel, gk_name
from felizjueves.ir import Raw, walk

CLASSES = emit_cli.grad_k_classes(1)
F_GOLDENS = [(3, 0, 0, 0), (0, 3, 0, 0), (0, 0, 3, 0), (0, 0, 0, 3)]
GOLDEN_CLASSES = CLASSES + F_GOLDENS
DIGESTS = GOLDEN / "gk_digests.json"
DIGESTED = [c for c in emit_cli.grad_k_classes(3) if c not in GOLDEN_CLASSES]
COMPILE_BUDGET_S = 10.0  # the user's per-compile budget for #618


def _id(c) -> str:
    return gk_name(c)


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_gk_kernel(cls), arch)


def _digest(text: str) -> dict:
    b = text.encode()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def write_goldens() -> None:
    """Rewrite every exchange-gradient golden and digest (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        for c in GOLDEN_CLASSES:
            (GOLDEN / backend / f"{gk_name(c)}.{ext}").write_text(
                _text(c, backend))
        digests[backend] = {gk_name(c): _digest(_text(c, backend))
                            for c in DIGESTED}
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_k_class_through_lmax() -> None:
    assert len(CLASSES) == 16 and CLASSES[0] == (0, 0, 0, 0)
    assert CLASSES[-1] == (1, 1, 1, 1)
    assert emit_cli.DEFAULT_GRAD_K_L == 1 and emit_cli.GRAD_K_L_MAX == 3
    assert emit_cli.grad_k_classes(0) == [(0, 0, 0, 0)]
    assert len(emit_cli.grad_k_classes(2)) == 81
    assert len(emit_cli.grad_k_classes(3)) == 256


def test_the_ladder_refuses_above_ff() -> None:
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            emit_cli.grad_k_classes(bad)


def test_the_names_are_the_builders() -> None:
    for c in GOLDEN_CLASSES:
        assert build_gk_kernel(c).name == gk_name(c) == (
            f"gk_{c[0]}{c[1]}_{c[2]}{c[3]}")


def test_no_raw_escape_hatch() -> None:
    for c in GOLDEN_CLASSES:
        k = build_gk_kernel(c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens and the digests ------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", GOLDEN_CLASSES, ids=_id)
def test_the_exchange_gradient_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{gk_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_twenty_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"gk_*.{ext}")}
        assert files == {gk_name(c) for c in GOLDEN_CLASSES}


def test_the_digests_are_every_other_default_unit() -> None:
    d = json.loads(DIGESTS.read_text())
    for backend in BACKEND:
        assert set(d[backend]) == {gk_name(c) for c in DIGESTED}


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=gk_name(c))
                                 for c in DIGESTED])
def test_the_exchange_gradient_digest_is_unchanged(cls, backend: str) -> None:
    want = json.loads(DIGESTS.read_text())[backend][gk_name(cls)]
    assert _digest(_text(cls, backend)) == want


def test_the_vendors_differ_only_in_the_prologue_and_the_lane_mask() -> None:
    # the warp reduction's mask is 32-bit on CUDA, 64-bit on HIP
    for cls in ((1, 0, 1, 1), (3, 0, 0, 0)):
        c, h = _text(cls, "cuda"), _text(cls, "hip")
        assert c.split("\n")[2:] == h.replace(
            "0xffffffffull", "0xffffffffu").split("\n")[3:]
        assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        for cls in ((1, 1, 0, 1), (0, 0, 3, 0)):
            text = _text(cls, backend)
            assert "atomicAdd(&G[" in text
            assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_grad_k_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--grad-k-l") == 0
    mine = {f"{gk_name(c)}.{ext}" for c in CLASSES}
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | mine
    for p in mine:
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_grad_k_l_adds_its_units_beside_the_other_ladders(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--grad-l", "0",
                "--grad-k-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "gs_00.cu", "gt_00.cu",
                                "gv_00.cu", "gjab_00.cu", "gk_00_00.cu"}


def test_the_bare_default_writes_no_exchange_gradient_unit(tmp_path) -> None:
    assert _run(tmp_path, "cuda") == 0
    assert not any(n.startswith("gk_") for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_exchange_gradient_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--grad-k-l" not in line


def test_the_names_cannot_collide_with_any_other_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--mu-l",
                "--schwarz-l", "--sx-l", "--metric-l", "--eri3-l",
                "--grad-l", "3") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith("gk_") for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--grad-k-l", "1") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 16 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--grad-k-l", "4"],
    ["--grad-k-l", "-1"],
    ["--grad-k-l", "x"],
    ["--grad-k-l", "1", "--class", "0", "0", "0", "0"],
    ["--grad-k-l", "1", "--shell-ls", "0"],
    ["--grad-k-l", "1", "--fock-l", "0"],
    ["--grad-k-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--grad-k-l", "1", "--n-batches", "3"],
    ["--grad-k-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_grad_k_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_grad_k_l_above_ff_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--grad-k-l", "4") == 2
    err = capsys.readouterr().err
    assert "--grad-k-l 4 is outside 0..3" in err
    assert "validated through (ff|ff) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    compile_kernel(tc, backend, build_gk_kernel(cls), tmp_path,
                   timeout=COMPILE_BUDGET_S)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", GOLDEN_CLASSES, ids=_id)
def test_nvcc_compiles_the_exchange_gradient_unit(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", GOLDEN_CLASSES, ids=_id)
def test_hipcc_compiles_the_exchange_gradient_unit(cls, hipcc,
                                                   tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_grad_k.py "
                 "--write")
    write_goldens()
