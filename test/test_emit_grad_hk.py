"""The HGP exchange-gradient kernel emitted for both vendors: golden texts
for every K class with every shell s or p, the `--grad-hk-l` ladder, and a
compile of each golden (and of the d classes, `slow` above `FAST_L_MAX`)
through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --grad-hk-l [LMAX]` writes
`ghk_{la}{lc}_{lb}{ld}` (`gradient.kernel_ghk`) for every K class, the
Hermite pairing, each `l <= LMAX` (1 when the flag is bare: 16 units). It
accepts 0..2 (validated through `(dd|dd)`, interpreted), exits 2 above
that, is NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/ghk_{la}{lc}_{lb}{ld}.{cu,hip}`
for the 16 s / p classes. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_grad_hk.py --write

THE COMPILE is test_emit_grad.py's: each unit alone (`-c`), the kernel
tier's flags, FAILING on a warning located in the generated unit (#92),
each under `COMPILE_BUDGET_S`. Needs a compiler, never a card.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from test_emit_grad import BACKEND, GOLDEN, _run, compile_kernel
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.gradient.kernel_ghk import build_ghk_kernel, ghk_name
from felizjueves.ir import Raw, walk

CLASSES = emit_cli.grad_hk_classes(1)
D_CLASSES = [(2, 0, 0, 0), (0, 2, 0, 0), (1, 1, 1, 1), (2, 1, 1, 0),
             (2, 2, 0, 0), (2, 2, 1, 1)]
COMPILE_BUDGET_S = 600.0


def _id(c) -> str:
    return ghk_name(c)


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_ghk_kernel(cls), arch)


def write_goldens() -> None:
    """Rewrite every HGP exchange-gradient golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for c in CLASSES:
            (GOLDEN / backend / f"{ghk_name(c)}.{ext}").write_text(
                _text(c, backend))


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_k_class_through_lmax() -> None:
    assert len(CLASSES) == 16 and CLASSES[0] == (0, 0, 0, 0)
    assert CLASSES == emit_cli.grad_k_classes(1)
    assert emit_cli.DEFAULT_GRAD_HK_L == 1 and emit_cli.GRAD_HK_L_MAX == 2
    assert emit_cli.grad_hk_classes(0) == [(0, 0, 0, 0)]
    assert len(emit_cli.grad_hk_classes(2)) == 81


def test_the_ladder_refuses_above_dd() -> None:
    for bad in (-1, 3, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.2"):
            emit_cli.grad_hk_classes(bad)


def test_the_names_are_the_builders() -> None:
    for c in CLASSES:
        assert build_ghk_kernel(c).name == ghk_name(c) == (
            f"ghk_{c[0]}{c[1]}_{c[2]}{c[3]}")


def test_no_raw_escape_hatch() -> None:
    for c in CLASSES:
        k = build_ghk_kernel(c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_id)
def test_the_hgp_exchange_gradient_golden_is_unchanged(cls, backend) -> None:
    path = GOLDEN / backend / f"{ghk_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_sixteen_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"ghk_*.{ext}")}
        assert files == {ghk_name(c) for c in CLASSES}


def test_the_write_is_atomic_and_the_bra_is_staged() -> None:
    for backend in BACKEND:
        text = _text((1, 1, 0, 1), backend)
        assert text.count("atomicAdd(&G[") == 6
        assert "__shared__" in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_grad_hk_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--grad-hk-l") == 0
    mine = {f"{ghk_name(c)}.{ext}" for c in CLASSES}
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | mine
    for p in mine:
        assert (tmp_path / p).read_text() == (GOLDEN / backend / p).read_text()


def test_grad_hk_l_adds_its_units_beside_the_mmd_ladder(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--grad-k-l", "0",
                "--grad-hk-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "gk_00_00.cu",
                                "ghk_00_00.cu"}


def test_the_bare_default_writes_no_hgp_exchange_gradient_unit(
        tmp_path) -> None:
    assert _run(tmp_path, "cuda") == 0
    assert not any(n.startswith("ghk_") for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_hgp_exchange_gradient_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--grad-hk-l" not in line


@pytest.mark.parametrize("argv", [
    ["--grad-hk-l", "3"],
    ["--grad-hk-l", "-1"],
    ["--grad-hk-l", "x"],
    ["--grad-hk-l", "1", "--class", "0", "0", "0", "0"],
    ["--grad-hk-l", "1", "--shell-ls", "0"],
    ["--grad-hk-l", "1", "--fock-l", "0"],
    ["--grad-hk-l", "1", "--n-batches", "3"],
    ["--grad-hk-l", "1", "--hgp-k-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_grad_hk_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_grad_hk_l_above_dd_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--grad-hk-l", "3") == 2
    err = capsys.readouterr().err
    assert "--grad-hk-l 3 is outside 0..2" in err
    assert "validated through (dd|dd) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


UNITS = [*CLASSES, *(c for c in D_CLASSES if c not in CLASSES)]
COMPILED = [by_l(c, sum(c) + 1, id=ghk_name(c)) for c in UNITS]


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    compile_kernel(tc, backend, build_ghk_kernel(cls), tmp_path,
                   timeout=COMPILE_BUDGET_S)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", COMPILED)
def test_nvcc_compiles_the_hgp_exchange_gradient_unit(cls, nvcc,
                                                      tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", COMPILED)
def test_hipcc_compiles_the_hgp_exchange_gradient_unit(cls, hipcc,
                                                       tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_grad_hk.py "
                 "--write")
    write_goldens()
