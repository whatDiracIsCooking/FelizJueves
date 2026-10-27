"""The d2 hessian kernels emitted for both vendors: golden texts for the
pair classes (ss) through (dd), the `--hess-l` ladder, and a compile of
each through nvcc (sm_86) and hipcc (gfx1200).

THE LADDER. `python -m felizjueves.emit --hess-l [LMAX]` writes, for every
`la <= lb <= LMAX` (2 when the flag is bare), `d2s_`, `d2t_` and
`d2v_{la}{lb}` (`hessian.kernel_h1e`). It accepts 0..3 (the kernels are
validated through (ff), interpreted), exits 2 above that, is NEVER in the
bare default and is not in KERNEL_GEN_CMD.

THE GOLDENS are `test/golden/{cuda,hip}/{d2s,d2t,d2v}_{la}{lb}.{cu,hip}` for
the six classes (ss) (sp) (sd) (pp) (pd) (dd); a unit over the repo's
512 KiB file cap (`d2v_22`, ~1 MB of unrolled Hermite densities) is pinned
instead by its byte count and sha256 in `test/golden/hessian_digests.json`,
as `fock_digests.json` pins the large F units. Regenerate (and review the
diff) with

    PYTHONPATH=src uv run python test/test_emit_hess.py --write

THE COMPILE. Each golden is compiled alone (`-c`) with the kernel tier's
own flags and FAILS on any warning located in the generated unit (#92's
filter; `test_emit_grad.compile_kernel`). Needs a compiler, never a card.
The session that wrote these kernels had neither compiler, so these cases
have NOT been run yet; the units passed clang 18's CUDA front end (device
side, `-Wall -Wextra`, a declaration stub for the runtime) with no
diagnostic, which is a proxy, not this test.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_emit_grad import compile_kernel

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.hessian.kernel_h1e import HESSIAN_1E, build_h1e_kernel, h1e_name
from felizjueves.ir import Raw, walk
from felizjueves.performance.device import GFX1200, SM_86

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
DIGESTS = GOLDEN / "hessian_digests.json"
CAP = 512 * 1024  # the check-added-large-files hook's --maxkb
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
FAMILY = dict(zip(("d2s", "d2t", "d2v"), HESSIAN_1E, strict=True))
PREFIXES = ("d2s_", "d2t_", "d2v_")
UNITS = [(f, c) for c in CLASSES for f in FAMILY]


def _kernel(family: str, cls):
    return build_h1e_kernel((FAMILY[family], *cls))


def _name(family: str, cls) -> str:
    return h1e_name((FAMILY[family], *cls))


def _id(u) -> str:
    return _name(*u)


def _text(family: str, cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(_kernel(family, cls), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_goldens() -> None:
    """Rewrite every hessian golden and the digest file (module
    docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for f, c in UNITS:
            text = _text(f, c, backend)
            path = GOLDEN / backend / f"{_name(f, c)}.{ext}"
            if len(text.encode()) > CAP:
                digests[backend][_name(f, c)] = _digest(text)
                path.unlink(missing_ok=True)
            else:
                path.write_text(text)
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def _pinned(backend: str) -> dict:
    return json.loads(DIGESTS.read_text())[backend]


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_every_pair_class_through_lmax() -> None:
    assert emit_cli.hess_classes(2) == CLASSES
    assert emit_cli.DEFAULT_HESS_L == 2 and emit_cli.HESS_L_MAX == 3
    assert emit_cli.hess_classes(0) == [(0, 0)]
    assert len(emit_cli.hess_classes(3)) == 10


def test_the_ladder_refuses_above_ff() -> None:
    for bad in (-1, 4, 8, 1.0, True):
        with pytest.raises(ValueError, match=r"outside 0\.\.3"):
            emit_cli.hess_classes(bad)


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
    text = _text(*unit, backend)
    name = _name(*unit)
    path = GOLDEN / backend / f"{name}.{BACKEND[backend][2]}"
    if name in _pinned(backend):
        assert not path.exists()
        assert _digest(text) == _pinned(backend)[name]
    else:
        assert len(text.encode()) <= CAP
        assert text == path.read_text()


def test_the_golden_set_is_exactly_the_six_classes() -> None:
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for f in FAMILY
                 for p in (GOLDEN / backend).glob(f"{f}_[0-9]*.{ext}")}
        assert files | set(_pinned(backend)) == {_name(*u) for u in UNITS}
        assert not files & set(_pinned(backend))


@pytest.mark.parametrize("unit", [("d2s", (1, 2)), ("d2v", (1, 1))], ids=_id)
def test_the_vendors_differ_only_in_the_prologue(unit) -> None:
    c, h = _text(*unit, "cuda"), _text(*unit, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_every_write_is_atomic_with_no_shared_array() -> None:
    for backend in BACKEND:
        for f in FAMILY:
            text = _text(f, (1, 1), backend)
            assert "atomicAdd(&H[" in text
            assert "__shared__" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hess_l_bare_writes_the_golden_units(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--pair-l", "0", "--hess-l") == 0
    assert _names(tmp_path) == {f"dq_0_0.{ext}", f"jab_00.{ext}"} | {
        f"{_name(*u)}.{ext}" for u in UNITS}
    pinned = _pinned(backend)
    for u in UNITS:
        name = _name(*u)
        text = (tmp_path / f"{name}.{ext}").read_text()
        if name in pinned:
            assert _digest(text) == pinned[name]
        else:
            assert text == (GOLDEN / backend / f"{name}.{ext}").read_text()


def test_hess_l_adds_its_units_beside_the_ladder_flags(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--L", "0", "--hess-l", "0") == 0
    assert _names(tmp_path) == {"pq_L0_p0_q0.cu", "d2s_00.cu", "d2t_00.cu",
                                "d2v_00.cu"}


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
                "--grad-l", "3", "--hgp-one-e-l") == 0
    others = _names(tmp_path / "a")
    assert not any(n.startswith(PREFIXES) for n in others)
    assert _run(tmp_path / "b", "cuda", "--L", "0", "--hess-l", "1") == 0
    mine = _names(tmp_path / "b") - {"pq_L0_p0_q0.cu"}
    assert len(mine) == 9 and not mine & others


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


def _slow(u):
    # d2v_22's ~1 MB of straight-line text, as F's large units are `slow`
    return pytest.param(u, id=_id(u), marks=(
        (pytest.mark.slow,) if u == ("d2v", (2, 2)) else ()))


@pytest.mark.cuda
@pytest.mark.parametrize("unit", [_slow(u) for u in UNITS])
def test_nvcc_compiles_the_hessian_unit(unit, nvcc, tmp_path) -> None:
    compile_kernel(nvcc, "cuda", _kernel(*unit), tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("unit", [_slow(u) for u in UNITS])
def test_hipcc_compiles_the_hessian_unit(unit, hipcc, tmp_path) -> None:
    compile_kernel(hipcc, "hip", _kernel(*unit), tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hess.py --write")
    write_goldens()
