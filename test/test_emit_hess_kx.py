"""The exchange derivative matrix kernels emitted for both vendors (#689):
golden texts for a handful of classes in both forms, digests of every
other unit through `(ff|ff)`, the `--hess-kx-l` ladder, and a compile of
each golden through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hess-kx-l [LMAX]` writes
`kx1_{la}{lc}_{lb}{ld}` and `kxc_{la}{lc}_{lb}{ld}` (`hessian.kernel_kx`)
for every K class, the Hermite pairing, each `l <= LMAX` (1 when the flag
is bare: 16 classes, 32 units). It accepts 0..3 (validated through
`(ff|ff)`), exits 2 above that, is NEVER in the bare default and is not
in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{kx1,kxc}_{la}{lc}_{lb}{ld}.{cu,hip}`
for `GOLDEN_CLASSES` in both forms; every other unit through `(ff|ff)` is
pinned by size and sha256 in `test/golden/kx_digests.json` (`slow` above
`L + 1 = 4`, `tiers.by_l`). Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hess_kx.py --write

THE COMPILE is test_emit_grad.py's: each golden alone (`-c`), the kernel
tier's flags, FAILING on a warning located in the generated unit (#92),
under `COMPILE_BUDGET_S = 30` s a unit (data/measured/kx.json's declared
budget). `(ff|ff)` in both forms is compiled too (`slow`). Needs a
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
from felizjueves.hessian.kernel_kx import (
    FORMS,
    build_kx_kernel,
    kx_name,
    kx_spec,
)
from felizjueves.ir import Raw, walk
from felizjueves.spec import Target

GOLDEN_CLASSES = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 1),
                  (1, 1, 1, 1), (3, 0, 0, 0), (0, 3, 0, 0)]
GOLDEN_SPECS = [kx_spec(*c, form=f) for c in GOLDEN_CLASSES for f in FORMS]
DIGESTS = GOLDEN / "kx_digests.json"
DIGESTED = [kx_spec(*c, form=f) for c in emit_cli.hess_kx_classes(3)
            for f in FORMS if c not in GOLDEN_CLASSES]
COMPILE_BUDGET_S = 30.0  # data/measured/kx.json's declared budget


def _name(spec) -> str:
    # `kx_name` is the name the builder gives (it names its kernel with it,
    # and test_kernel_kx.py pins the two equal), without building the
    # kernel: this names every case's id at collection, 540 builds a worker.
    return kx_name(spec)


def _text(spec, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_kx_kernel(spec), arch)


def _digest(text: str) -> dict:
    b = text.encode()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def write_goldens() -> None:
    """Rewrite every golden and digest (module docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        for s in GOLDEN_SPECS:
            (GOLDEN / backend / f"{_name(s)}.{ext}").write_text(
                _text(s, backend))
        digests[backend] = {_name(s): _digest(_text(s, backend))
                            for s in DIGESTED}
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_k_class_through_lmax() -> None:
    assert emit_cli.DEFAULT_HESS_KX_L == 1 and emit_cli.HESS_KX_L_MAX == 3
    assert emit_cli.hess_kx_classes(0) == [(0, 0, 0, 0)]
    assert len(emit_cli.hess_kx_classes(1)) == 16
    assert len(emit_cli.hess_kx_classes(3)) == 256


def test_the_ladder_refuses_above_ff() -> None:
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            emit_cli.hess_kx_classes(bad)


def test_no_raw_escape_hatch() -> None:
    for s in GOLDEN_SPECS:
        k = build_kx_kernel(s)
        assert not any(isinstance(x, Raw) for x in walk(k.body)), k.name


# --- the goldens and the digests ------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("spec", GOLDEN_SPECS, ids=_name)
def test_the_derivative_matrix_golden_is_unchanged(spec, backend) -> None:
    path = GOLDEN / backend / f"{_name(spec)}.{BACKEND[backend][2]}"
    assert _text(spec, backend) == path.read_text()


def test_the_golden_set_is_exactly_the_golden_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for d in ("kx1", "kxc")
                 for p in (GOLDEN / backend).glob(f"{d}_*.{ext}")}
        assert files == {_name(s) for s in GOLDEN_SPECS}


def test_the_digests_are_every_other_unit() -> None:
    d = json.loads(DIGESTS.read_text())
    for backend in BACKEND:
        assert set(d[backend]) == {_name(s) for s in DIGESTED}


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("spec", [by_l(s, s.total_am + 1, id=_name(s))
                                  for s in DIGESTED])
def test_the_derivative_matrix_digest_is_unchanged(spec, backend) -> None:
    want = json.loads(DIGESTS.read_text())[backend][_name(spec)]
    assert _digest(_text(spec, backend)) == want


def test_the_vendors_differ_only_in_the_prologue() -> None:
    for s in (kx_spec(1, 0, 1, 1, form=Target.KX1),
              kx_spec(3, 0, 0, 0, form=Target.KXC)):
        c, h = _text(s, "cuda"), _text(s, "hip")
        assert c.split("\n")[2:] == h.split("\n")[3:]
        assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        for s in GOLDEN_SPECS:
            text = _text(s, backend)
            assert "atomicAdd(&X[" in text
            assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hess_kx_l_bare_writes_both_forms_of_every_s_p_class(
        backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hess-kx-l") == 0
    mine = {f"{_name(kx_spec(*c, form=f))}.{ext}"
            for c in emit_cli.hess_kx_classes(1) for f in FORMS}
    assert len(mine) == 32
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | mine
    for s in GOLDEN_SPECS:
        p = f"{_name(s)}.{ext}"
        if p in mine:
            assert (tmp_path / p).read_text() == (
                GOLDEN / backend / p).read_text()


def test_hess_kx_l_adds_its_units_beside_the_other_ladders(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--grad-k-l", "0",
                "--hess-kx-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "gk_00_00.cu",
                                "kx1_00_00.cu", "kxc_00_00.cu"}


def test_the_bare_default_writes_no_derivative_matrix_unit(tmp_path) -> None:
    assert _run(tmp_path, "cuda") == 0
    assert not any(n.startswith(("kx1_", "kxc_")) for n in _names(tmp_path))


def test_kernel_gen_cmd_asks_for_no_derivative_matrix_unit() -> None:
    from felizjueves.performance import measured
    line = measured.config_value("KERNEL_GEN_CMD")
    assert "felizjueves.emit" in line and "--hess-kx-l" not in line


def test_the_names_cannot_collide_with_any_other_ladder(tmp_path) -> None:
    assert _run(tmp_path / "a", "cuda", "--one-e-l", "3", "--mu-l",
                "--schwarz-l", "--sx-l", "--metric-l", "--eri3-l",
                "--grad-l", "3", "--grad-k-l", "1", "--hess-k-l", "1") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(("kx1_", "kxc_")) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--hess-kx-l", "1") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 32 and not mine & others


@pytest.mark.parametrize("argv", [
    ["--hess-kx-l", "4"],
    ["--hess-kx-l", "-1"],
    ["--hess-kx-l", "x"],
    ["--hess-kx-l", "1", "--class", "0", "0", "0", "0"],
    ["--hess-kx-l", "1", "--shell-ls", "0"],
    ["--hess-kx-l", "1", "--fock-l", "0"],
    ["--hess-kx-l", "1", "--n-batches", "3"],
    ["--hess-kx-l", "1", "--hgp-j-l", "1"],
], ids=lambda a: "_".join(a))
def test_a_hess_kx_l_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_hess_kx_l_above_ff_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hess-kx-l", "4") == 2
    err = capsys.readouterr().err
    assert "--hess-kx-l 4 is outside 0..3" in err
    assert "validated through (ff|ff) only" in err


# --- the compile (a compiler, no card) ----------------------------------------

COMPILED = [*GOLDEN_SPECS,
            *(by_l(kx_spec(3, 3, 3, 3, form=f), 13,
                   id=f"{f.value}_33_33") for f in FORMS)]


def _compile(tc, backend: str, spec, tmp_path: Path) -> None:
    compile_kernel(tc, backend, build_kx_kernel(spec), tmp_path,
                   timeout=COMPILE_BUDGET_S)


@pytest.mark.cuda
@pytest.mark.parametrize("spec", COMPILED, ids=_name)
def test_nvcc_compiles_the_derivative_matrix_unit(spec, nvcc,
                                                  tmp_path) -> None:
    _compile(nvcc, "cuda", spec, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("spec", COMPILED, ids=_name)
def test_hipcc_compiles_the_derivative_matrix_unit(spec, hipcc,
                                                   tmp_path) -> None:
    _compile(hipcc, "hip", spec, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hess_kx.py "
                 "--write")
    write_goldens()
