"""The fused Fock gradient kernel emitted for both vendors: golden texts for
the `(ss|ss)` class, digests of every other class of milestone 12's ladder
through `(dd|dd)`, and the `--grad-f-l` ladder.

THE LADDER. `python -m felizjueves.emit --grad-f-l [LMAX]` writes
`gf_{la}{lb}{lc}{ld}` (`gradient.kernel_gf`) for every F class of
milestone 12's ladder (`emit.__main__.fock_classes`: AO order, canonical)
with each `l <= LMAX` (1 when the flag is bare: 6 units). It accepts 0..2
(validated through `(dd|dd)`, interpreted), exits 2 above that, is NEVER
in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/gf_0000.{cu,hip}`; every other
ladder unit is pinned by size and sha256 in `test/golden/gf_digests.json`
(`slow` above `L + 1 = 4`, `tiers.by_l`: the kernel builds the class one
order up). An unrolled `gf_` unit is large (`(pp|pp)`'s is ~590 KB of
CUDA), so one text golden is tracked and the rest are digests, as
`gk_digests.json` does. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_grad_f.py --write

THE COMPILE is test_emit_grad.py's: each ladder unit through `(pp|pp)`
(`COMPILED`, the default ladder) alone (`-c`), the kernel tier's flags,
FAILING on a warning located in the generated unit (#92). Needs a
compiler, never a card. The d classes are not compiled by a test: their
units are 1.3 to 11 MB of unrolled text.
"""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
from test_emit_grad import BACKEND, GOLDEN, _run, compile_kernel
from tiers import by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.gradient.kernel_gf import build_gf_kernel, gf_name

CLASSES = emit_cli.grad_f_classes(1)
GOLDEN_CLASSES = [(0, 0, 0, 0)]
DIGESTS = GOLDEN / "gf_digests.json"
DIGESTED = [c for c in emit_cli.grad_f_classes(2) if c not in GOLDEN_CLASSES]
COMPILED = CLASSES  # through (pp|pp): the bare ladder


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_gf_kernel(cls), arch)


def _digest(text: str) -> dict:
    b = text.encode()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def write_goldens() -> None:
    """Rewrite the fused Fock gradient's goldens and digests (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        for c in GOLDEN_CLASSES:
            (GOLDEN / backend / f"{gf_name(c)}.{ext}").write_text(
                _text(c, backend))
        digests[backend] = {gf_name(c): _digest(_text(c, backend))
                            for c in DIGESTED}
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_the_fock_ladder_through_lmax() -> None:
    assert CLASSES == [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1),
                       (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)]
    assert emit_cli.DEFAULT_GRAD_F_L == 1 and emit_cli.GRAD_F_L_MAX == 2
    assert emit_cli.grad_f_classes(0) == [(0, 0, 0, 0)]
    assert emit_cli.grad_f_classes(2) == emit_cli.fock_classes(2)
    assert len(emit_cli.grad_f_classes(2)) == 21


def test_the_ladder_refuses_above_dd() -> None:
    for bad in (-1, 3, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.2"):
            emit_cli.grad_f_classes(bad)


# --- the goldens and the digests ------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", GOLDEN_CLASSES, ids=gf_name)
def test_the_fock_gradient_golden_is_unchanged(cls, backend: str) -> None:
    path = GOLDEN / backend / f"{gf_name(cls)}.{BACKEND[backend][2]}"
    assert _text(cls, backend) == path.read_text()


def test_the_golden_set_is_exactly_ss_ss() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"gf_*.{ext}")}
        assert files == {gf_name(c) for c in GOLDEN_CLASSES}


def test_the_digests_are_every_other_ladder_unit() -> None:
    d = json.loads(DIGESTS.read_text())
    for backend in BACKEND:
        assert set(d[backend]) == {gf_name(c) for c in DIGESTED}


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=gf_name(c))
                                 for c in DIGESTED])
def test_the_fock_gradient_digest_is_unchanged(cls, backend: str) -> None:
    want = json.loads(DIGESTS.read_text())[backend][gf_name(cls)]
    assert _digest(_text(cls, backend)) == want


def test_the_vendors_differ_only_in_the_prologue() -> None:
    c, h = _text((0, 0, 0, 0), "cuda"), _text((0, 0, 0, 0), "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_write_is_atomic_into_g_with_one_shared_record() -> None:
    for backend in BACKEND:
        text = _text((0, 0, 0, 0), backend)
        assert text.count("atomicAdd(&G[") == 12
        assert text.count("__shared__") == 1


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_grad_f_l_zero_writes_the_golden_unit(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--grad-f-l", "0") == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        f"dq_0_0.{ext}", f"jab_00.{ext}", f"gf_0000.{ext}"}
    assert (tmp_path / f"gf_0000.{ext}").read_text() == (
        GOLDEN / backend / f"gf_0000.{ext}").read_text()


def test_the_bare_default_writes_no_fock_gradient_unit(tmp_path) -> None:
    assert _run(tmp_path, "cuda") == 0
    assert not any(p.name.startswith("gf_") for p in tmp_path.iterdir())


def test_kernel_gen_cmd_asks_for_no_fock_gradient_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--grad-f-l" not in line


@pytest.mark.parametrize("argv", [
    ["--grad-f-l", "3"],
    ["--grad-f-l", "-1"],
    ["--grad-f-l", "x"],
    ["--grad-f-l", "1", "--class", "0", "0", "0", "0"],
    ["--grad-f-l", "1", "--shell-ls", "0"],
    ["--grad-f-l", "1", "--fock-l", "0"],
    ["--grad-f-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--grad-f-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_grad_f_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_grad_f_l_above_dd_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--grad-f-l", "3") == 2
    err = capsys.readouterr().err
    assert "--grad-f-l 3 is outside 0..2" in err
    assert "validated through (dd|dd) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("cls", COMPILED, ids=gf_name)
def test_nvcc_compiles_the_fock_gradient_unit(cls, nvcc, tmp_path) -> None:
    compile_kernel(nvcc, "cuda", build_gf_kernel(cls), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", COMPILED, ids=gf_name)
def test_hipcc_compiles_the_fock_gradient_unit(cls, hipcc, tmp_path) -> None:
    compile_kernel(hipcc, "hip", build_gf_kernel(cls), tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_grad_f.py "
                 "--write")
    write_goldens()
