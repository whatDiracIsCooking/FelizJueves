"""The fused Fock kernel on Rys quadrature emitted for both vendors: golden
texts for the 21 F classes, the `--fock-l` / `--fock-class` path with
`--rys`, and a compile of every class through nvcc (sm_86) and hipcc
(gfx1200).

THE GOLDENS are `test/golden/{cuda,hip}/fr_{la}{lb}{lc}{ld}.{cu,hip}`:
`kernel_frys.build_frys_kernel(cls)` spelled by each backend, one per class
of `hgp.vrr.CLASSES`; over `GOLDEN_CAP` (512 KiB) a class is pinned by its
byte count and sha256 in `test/golden/frys_digests.json` instead -- the cap
decides, never a list (test_emit_fock.py's rule). Regenerate (and review the
diff) with

    PYTHONPATH=src uv run python test/test_emit_frys.py --write

THE COMPILE is test_emit_fock.py's (`_compile`: the tier's flags, the
resource report, a warning in generated text fails, the 8,192 B of shared
staging). NOT YET RUN: the session that built the kernel had neither nvcc
nor hipcc; every unit passed clang 18's CUDA front end (device side,
`-fsyntax-only -Wall -Wextra`, a runtime stub) with no diagnostic, a proxy.
No compile time is measured, so the deferral to `slow` is by L (above
`FAST_L_MAX`), not by a recorded budget.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import test_emit_fock as TEF
from tiers import split_by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.eri.fock.kernel_frys import build_frys_kernel, frys_name
from felizjueves.hgp import vrr
from felizjueves.ir import Raw, walk

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
DIGESTS = GOLDEN / "frys_digests.json"
GOLDEN_CAP = TEF.GOLDEN_CAP
BACKEND = TEF.BACKEND
CLASSES = vrr.CLASSES


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_frys_kernel(cls), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_goldens() -> None:
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for cls in CLASSES:
            text = _text(cls, backend)
            path = GOLDEN / backend / f"{frys_name(cls)}.{ext}"
            if len(text.encode()) <= GOLDEN_CAP:
                path.write_text(text)
            else:
                path.unlink(missing_ok=True)
                digests[backend][frys_name(cls)] = _digest(text)
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", split_by_l(CLASSES, sum, ids=frys_name))
def test_the_frys_golden_is_unchanged(cls, backend: str) -> None:
    text = _text(cls, backend)
    path = GOLDEN / backend / f"{frys_name(cls)}.{BACKEND[backend][2]}"
    pinned = json.loads(DIGESTS.read_text())[backend]
    if len(text.encode()) <= GOLDEN_CAP:
        assert frys_name(cls) not in pinned
        assert text == path.read_text()
    else:
        assert not path.exists(), f"{path} is over the {GOLDEN_CAP} B cap"
        assert _digest(text) == pinned[frys_name(cls)]
    assert not any(isinstance(s, Raw)
                   for s in walk(build_frys_kernel(cls).body))


def test_the_golden_set_is_exactly_the_ladder() -> None:
    pinned = json.loads(DIGESTS.read_text())
    assert sorted(pinned) == sorted(BACKEND)
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"fr_*.{ext}")}
        assert files.isdisjoint(pinned[backend])
        assert files | set(pinned[backend]) == {frys_name(c) for c in CLASSES}


def test_the_vendors_differ_only_in_the_prologue() -> None:
    c, h = _text((1, 1, 1, 1), "cuda"), _text((1, 1, 1, 1), "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]


def test_the_staging_and_signature_are_kernel_fs() -> None:
    from felizjueves.eri.fock.kernel_f import build_f_kernel
    for backend in BACKEND:
        fr, f = _text((1, 1, 1, 1), backend), TEF._text((1, 1, 1, 1), backend)
        assert fr.count("__shared__") == 1 and "__syncthreads" not in fr
        sig = [line for line in fr.split("\n") if "__global__" in line]
        assert len(sig) == 1
        start = fr.index("(", fr.index("__global__"))
        fstart = f.index("(", f.index("__global__"))
        assert fr[start:fr.index(")", start)] == f[fstart:f.index(")", fstart)]
    assert build_f_kernel((1, 1, 1, 1)).params == \
        build_frys_kernel((1, 1, 1, 1)).params


# --- the CLI path -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_rys_writes_the_golden_text(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--rys", "--fock-l", "0",
                "--fock-class", "1", "1", "1", "1") == 0
    assert {p.name for p in tmp_path.iterdir()} == {f"fr_0000.{ext}",
                                                    f"fr_1111.{ext}"}
    for n in ("fr_0000", "fr_1111"):
        assert (tmp_path / f"{n}.{ext}").read_text() == \
            (GOLDEN / backend / f"{n}.{ext}").read_text()


@pytest.mark.parametrize("argv", [
    ["--rys"], ["--rys", "--L", "2"], ["--rys", "--one-e-l", "1"],
    ["--rys", "--fock-l", "3"],
], ids=lambda a: "_".join(a))
def test_a_rys_request_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, cls, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(TEF, "build_f_kernel", build_frys_kernel)
    TEF._compile(tc, backend, cls, tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", split_by_l(CLASSES, sum, ids=frys_name))
def test_nvcc_compiles_the_frys_class(cls, nvcc, tmp_path, monkeypatch) -> None:
    _compile(nvcc, "cuda", cls, tmp_path, monkeypatch)


@pytest.mark.hip
@pytest.mark.parametrize("cls", split_by_l(CLASSES, sum, ids=frys_name))
def test_hipcc_compiles_the_frys_class(cls, hipcc, tmp_path,
                                       monkeypatch) -> None:
    _compile(hipcc, "hip", cls, tmp_path, monkeypatch)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_frys.py --write")
    write_goldens()
