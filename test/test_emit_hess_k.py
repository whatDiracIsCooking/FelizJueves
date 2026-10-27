"""The exchange-Hessian kernel emitted for both vendors: golden texts for
every K class with every shell s or p, the `--hess-k-l` ladder, and a
compile of each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hess-k-l [LMAX]` writes
`hkx_{la}{lc}_{lb}{ld}` (`hessian.kernel_hkx`) for every K class, the
Hermite pairing, each `l <= LMAX` (1 when the flag is bare: 16 units).
It accepts 0..1 (validated through `(pp|pp)`, interpreted), exits 2 above
that, is NEVER in the bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/hkx_{la}{lc}_{lb}{ld}.{cu,hip}`
for the 16 s / p classes; a class whose text is over `GOLDEN_CAP` (512
KiB, the pre-commit `check-added-large-files --maxkb=512`) is pinned
instead by its byte count and sha256 in `test/golden/hkx_digests.json`
(`test_emit_fock.py`'s rule). Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_hess_k.py --write

THE COMPILE is test_emit_grad.py's: each unit alone (`-c`), the kernel
tier's flags, FAILING on a warning located in the generated unit (#92).
Needs a compiler, never a card. The session that wrote this kernel had
neither compiler, so these cases have NOT been run yet (clang 18's CUDA
front end, device side, `-Wall -Wextra` with a runtime stub, passed all
16 units: a proxy).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_emit_grad import BACKEND, GOLDEN, _run, compile_kernel

from felizjueves.emit import __main__ as emit_cli
from felizjueves.hessian.kernel_hkx import build_hkx_kernel, hkx_name
from felizjueves.ir import Raw, walk

CLASSES = emit_cli.hess_k_classes(1)
DIGESTS = GOLDEN / "hkx_digests.json"
GOLDEN_CAP = 512 * 1024  # bytes: .pre-commit-config.yaml's --maxkb=512


def _id(c) -> str:
    return hkx_name(c)


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_hkx_kernel(cls), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_goldens() -> None:
    """Rewrite every exchange-Hessian golden and the digest file (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for c in CLASSES:
            text = _text(c, backend)
            path = GOLDEN / backend / f"{hkx_name(c)}.{ext}"
            if len(text.encode()) > GOLDEN_CAP:
                digests[backend][hkx_name(c)] = _digest(text)
                path.unlink(missing_ok=True)
            else:
                path.write_text(text)
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
    assert emit_cli.DEFAULT_HESS_K_L == 1 and emit_cli.HESS_K_L_MAX == 1
    assert emit_cli.hess_k_classes(0) == [(0, 0, 0, 0)]


def test_the_ladder_refuses_above_pp() -> None:
    for bad in (-1, 2, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.1"):
            emit_cli.hess_k_classes(bad)


def test_the_names_are_the_builders() -> None:
    for c in CLASSES:
        assert build_hkx_kernel(c).name == hkx_name(c) == (
            f"hkx_{c[0]}{c[1]}_{c[2]}{c[3]}")


def test_no_raw_escape_hatch() -> None:
    for c in CLASSES:
        k = build_hkx_kernel(c)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_id)
def test_the_exchange_hessian_golden_is_unchanged(cls, backend: str) -> None:
    text = _text(cls, backend)
    pinned = json.loads(DIGESTS.read_text())[backend]
    if hkx_name(cls) in pinned:
        assert len(text.encode()) > GOLDEN_CAP
        assert _digest(text) == pinned[hkx_name(cls)]
    else:
        path = GOLDEN / backend / f"{hkx_name(cls)}.{BACKEND[backend][2]}"
        assert text == path.read_text()


def test_the_golden_set_is_exactly_the_sixteen_classes() -> None:
    pinned = json.loads(DIGESTS.read_text())
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"hkx_*.{ext}")}
        assert not files & set(pinned[backend])
        assert files | set(pinned[backend]) == {hkx_name(c) for c in CLASSES}


def test_the_vendors_differ_only_in_the_prologue_and_the_lane_mask() -> None:
    # the warp reduction's mask is 32-bit on CUDA, 64-bit on HIP
    c, h = _text((1, 0, 0, 1), "cuda"), _text((1, 0, 0, 1), "hip")
    assert c.split("\n")[2:] == h.replace(
        "0xffffffffull", "0xffffffffu").split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        text = _text((1, 1, 0, 1), backend)
        assert "atomicAdd(&H[" in text
        assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hess_k_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hess-k-l") == 0
    mine = {f"{hkx_name(c)}.{ext}" for c in CLASSES}
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | mine
    pinned = json.loads(DIGESTS.read_text())[backend]
    for c in CLASSES:
        text = (tmp_path / f"{hkx_name(c)}.{ext}").read_text()
        if hkx_name(c) in pinned:
            assert _digest(text) == pinned[hkx_name(c)]
        else:
            assert text == (GOLDEN / backend / f"{hkx_name(c)}.{ext}"
                            ).read_text()


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
    ["--hess-k-l", "2"],
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


def test_hess_k_l_above_pp_says_why(tmp_path, capsys) -> None:
    assert _run(tmp_path / "out", "hip", "--hess-k-l", "2") == 2
    err = capsys.readouterr().err
    assert "--hess-k-l 2 is outside 0..1" in err
    assert "validated through (pp|pp) only" in err


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    compile_kernel(tc, backend, build_hkx_kernel(cls), tmp_path)


def _cparam(c):
    # (pp|pp) is 1.6 MB of unrolled text
    return pytest.param(c, id=_id(c),
                        marks=(pytest.mark.slow,) if sum(c) > 2 else ())


@pytest.mark.cuda
@pytest.mark.parametrize("cls", [_cparam(c) for c in CLASSES])
def test_nvcc_compiles_the_exchange_hessian_unit(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", [_cparam(c) for c in CLASSES])
def test_hipcc_compiles_the_exchange_hessian_unit(cls, hipcc,
                                                  tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hess_k.py "
                 "--write")
    write_goldens()
