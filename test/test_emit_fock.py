"""The fused Fock kernel emitted for both vendors (#287): golden texts for
milestone 12's 21 classes, the `--fock-class` / `--fock-l` path, and a
compile of every class through nvcc (sm_86) and hipcc (gfx1200).

THE GOLDENS are `test/golden/{cuda,hip}/f_{la}{lb}{lc}{ld}.{cu,hip}`:
`kernel_f.build_f_kernel(cls)` spelled by each backend, one per class of
`hgp.vrr.CLASSES`. A class whose text is over `GOLDEN_CAP` (512 KiB, the
pre-commit `check-added-large-files --maxkb=512` the repo commits under) is
pinned instead by its byte count and sha256 in `test/golden/fock_digests.json`
-- today `(pd|pd)`, `(pd|dd)` and `(dd|dd)` on both backends (0.58, 1.07 and
1.99 MB). Which classes go which way is decided by the cap, never a list:
the test re-derives it. Regenerate (and review the diff) with

    PYTHONPATH=src uv run python test/test_emit_fock.py --write

THE COMPILE. Every class is compiled alone (`-c`) with the kernel tier's own
flags (devtools/config.sh `NVCC_FLAGS` / `HIPCC_FLAGS`, warnings on) plus
each compiler's resource report, and FAILS on any warning located in the
generated unit (#92's filter). It also pins the shared-memory footprint the
compilers report: ONE `ab_sh` per block, `MAX_PRIM_PAIRS x REC` doubles =
8,192 bytes, independent of `blockDim` (kernel_f's docstring, "THE STAGING
IS PER THREAD": every thread writes the same values into the same array, it
is not a copy per thread), against 48 KiB of static shared memory per block
on sm_86 and 64 KiB of LDS per workgroup on gfx1200. It fits, so the kernel
is emitted as built and #282's barrier-free staging is left as it is.

THE TIER BUDGET: a class whose single-unit compile took more than
`BUDGET_S` = 5 s on either backend is `slow`. `COMPILE_S` records the
measured seconds (nvcc 13.0 / hipcc ROCm 7.2.4, the tier's flags, one unit
at a time on this box, no cache), so the deferral is a measurement, not a
guess: `(pd|dd)` and `(dd|dd)`.

Needs a compiler, never a card: the compile cases are `cuda` / `hip`, none
is `gpu`. Running the kernels is #288's.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.hgp import vrr
from felizjueves.ir import Raw, walk
from felizjueves.kernel_f import MAX_PRIM_PAIRS, REC, build_f_kernel

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
DIGESTS = GOLDEN / "fock_digests.json"
GOLDEN_CAP = 512 * 1024  # bytes: .pre-commit-config.yaml's --maxkb=512
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
CLASSES = vrr.CLASSES
SHARED_BYTES = MAX_PRIM_PAIRS * len(REC) * 8  # 8,192

# Measured single-unit compile seconds (module docstring), (nvcc, hipcc).
COMPILE_S = {
    (0, 0, 0, 0): (0.7, 1.3), (0, 0, 0, 1): (0.8, 1.3), (0, 0, 0, 2): (0.8, 1.3),
    (0, 0, 1, 1): (0.8, 1.3), (0, 0, 1, 2): (0.8, 1.3), (0, 0, 2, 2): (0.9, 1.4),
    (0, 1, 0, 1): (0.8, 1.3), (0, 1, 0, 2): (0.8, 1.3), (0, 1, 1, 1): (0.8, 1.3),
    (0, 1, 1, 2): (0.9, 1.5), (0, 1, 2, 2): (1.1, 1.9), (0, 2, 0, 2): (1.0, 1.5),
    (0, 2, 1, 1): (1.0, 1.5), (0, 2, 1, 2): (1.2, 2.2), (0, 2, 2, 2): (2.0, 3.9),
    (1, 1, 1, 1): (1.0, 1.6), (1, 1, 1, 2): (1.3, 2.2), (1, 1, 2, 2): (2.2, 3.3),
    (1, 2, 1, 2): (2.4, 4.8), (1, 2, 2, 2): (5.5, 9.6), (2, 2, 2, 2): (15.0, 15.0),
}
BUDGET_S = 5.0
SLOW = frozenset(c for c, s in COMPILE_S.items() if max(s) > BUDGET_S)


def _name(cls) -> str:
    return "f_" + "".join(map(str, cls))


def _text(cls, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_f_kernel(cls), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_goldens() -> None:
    """Rewrite every fock golden and the digest file (module docstring)."""
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for cls in CLASSES:
            text = _text(cls, backend)
            path = GOLDEN / backend / f"{_name(cls)}.{ext}"
            if len(text.encode()) <= GOLDEN_CAP:
                path.write_text(text)
            else:
                path.unlink(missing_ok=True)
                digests[backend][_name(cls)] = _digest(text)
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


# --- the ladder ---------------------------------------------------------------


def test_the_ladder_is_milestone_12s_21_classes() -> None:
    assert len(CLASSES) == 21 and CLASSES[-1] == (2, 2, 2, 2)
    assert emit_cli.FOCK_CLASSES == CLASSES  # spelled there, held equal here
    assert all(max(c) <= emit_cli.FOCK_L_MAX for c in CLASSES)
    assert set(COMPILE_S) == set(CLASSES)


def test_the_deferred_classes_are_the_measured_ones() -> None:
    assert SLOW == {(1, 2, 2, 2), (2, 2, 2, 2)}


def test_no_raw_escape_hatch() -> None:
    for cls in CLASSES:
        k = build_f_kernel(cls)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_the_fock_golden_is_unchanged(cls, backend: str) -> None:
    text = _text(cls, backend)
    path = GOLDEN / backend / f"{_name(cls)}.{BACKEND[backend][2]}"
    pinned = json.loads(DIGESTS.read_text())[backend]
    if len(text.encode()) <= GOLDEN_CAP:
        assert _name(cls) not in pinned
        assert text == path.read_text()
    else:
        assert not path.exists(), f"{path} is over the {GOLDEN_CAP} B cap"
        assert _digest(text) == pinned[_name(cls)]


def test_the_golden_set_is_exactly_the_ladder() -> None:
    pinned = json.loads(DIGESTS.read_text())
    assert sorted(pinned) == sorted(BACKEND)
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"f_*.{ext}")}
        assert files.isdisjoint(pinned[backend])
        assert files | set(pinned[backend]) == {_name(c) for c in CLASSES}
        for p in (GOLDEN / backend).glob(f"f_*.{ext}"):
            assert p.stat().st_size <= GOLDEN_CAP, p


@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)],
                         ids=_name)
def test_the_vendors_differ_only_in_the_prologue(cls) -> None:
    c, h = _text(cls, "cuda"), _text(cls, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_staging_is_one_shared_array_per_block() -> None:
    for backend in BACKEND:
        text = _text((1, 1, 1, 1), backend)
        assert text.count("__shared__") == 1
        assert f"__shared__ double ab_sh[{MAX_PRIM_PAIRS}][{len(REC)}];" in text
        assert "__syncthreads" not in text  # #282's barrier-free staging


# --- the CLI path -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


@pytest.mark.parametrize(("lmax", "n"), [(0, 1), (1, 6), (2, 21)])
def test_fock_l_writes_the_ladder_through_lmax(lmax, n, tmp_path) -> None:
    assert _run(tmp_path, "hip", "--fock-l", str(lmax)) == 0
    want = [c for c in CLASSES if max(c) <= lmax]
    assert len(want) == n
    assert {p.name for p in tmp_path.iterdir()} == {
        f"{_name(c)}.hip" for c in want}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_fock_class_writes_the_golden_text(backend, tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--fock-class", "1", "1", "1", "1",
                "--fock-class", "0", "0", "0", "0") == 0
    assert {p.name for p in tmp_path.iterdir()} == {f"f_0000.{ext}",
                                                    f"f_1111.{ext}"}
    for n in ("f_0000", "f_1111"):
        assert (tmp_path / f"{n}.{ext}").read_text() == \
            (GOLDEN / backend / f"{n}.{ext}").read_text()


def test_fock_l_and_fock_class_write_the_union_in_ladder_order() -> None:
    assert emit_cli.fock_selection(0, [(2, 2, 2, 2), (0, 0, 0, 0)]) == [
        (0, 0, 0, 0), (2, 2, 2, 2)]
    assert emit_cli.fock_selection(None, [(0, 1, 0, 1)]) == [(0, 1, 0, 1)]


@pytest.mark.parametrize("argv", [
    ["--fock-l", "3"], ["--fock-l", "-1"],
    ["--fock-class", "1", "0", "0", "0"],  # not canonical: la > lb
    ["--fock-class", "1", "1", "0", "0"],  # bra pair after ket pair
    ["--fock-class", "0", "0", "0", "3"],  # off the ladder
    ["--fock-l", "1", "--L", "2"], ["--fock-l", "1", "--pair-l", "1"],
    ["--fock-l", "1", "--one-e-l", "1"],
    ["--fock-class", "0", "0", "0", "0", "--class", "0", "0", "0", "0"],
    ["--fock-l", "1", "--n-batches", "3"],
], ids=lambda a: "_".join(a))
def test_a_fock_request_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


# --- the compile (a compiler, no card) ----------------------------------------

_WARNING = r"(?:\((\d+)\): warning|:(\d+)(?::\d+)?: warning:?)\s*(.*)"
RESOURCE = {"cuda": ["-Xptxas", "-v"],
            "hip": ["-Rpass-analysis=kernel-resource-usage"]}
SHARED = {"cuda": r"(\d+) bytes smem", "hip": r"LDS Size \[bytes/block\]: (\d+)"}
FLAGS_VAR = {"cuda": "NVCC_FLAGS", "hip": "HIPCC_FLAGS"}


def _compile(tc, backend: str, cls, tmp_path: Path) -> None:
    import devcache

    from felizjueves import measured

    emit, arch, ext = BACKEND[backend]
    k = build_f_kernel(cls)
    main = f"{k.name}.{ext}"
    target = (f"-arch=sm_{tc.arch}" if tc.name == "nvcc"
              else f"--offload-arch={tc.arch}")
    flags = [*shlex.split(measured.config_value(FLAGS_VAR[backend])), target,
             *RESOURCE[backend]]
    got = devcache.compile_unit(tc.path, flags, main, emit(k, arch), {},
                                tmp_path / f"{k.name}.o", mode="-c",
                                timeout=1800)
    diag = got.diagnostics
    assert got.returncode == 0, (
        f"{tc.name} failed in {got.workdir}:\n"
        + "\n".join(shlex.join(c) for c in got.commands) + f"\n{diag}")
    warned = re.findall(r"(?:^|[\s\"'/])" + re.escape(main) + _WARNING, diag,
                        re.MULTILINE)
    assert not warned, (f"{tc.name}: warning(s) in generated text (#92) -- "
                        f"fix the emitter, never a flag: {warned}")
    shared = {int(x) for x in re.findall(SHARED[backend], diag)}
    assert shared == {SHARED_BYTES}, (shared, diag[-2000:])


def _cases():
    return [pytest.param(c, id=_name(c),
                         marks=[pytest.mark.slow] if c in SLOW else [])
            for c in CLASSES]


@pytest.mark.cuda
@pytest.mark.parametrize("cls", _cases())
def test_nvcc_compiles_the_fock_class(cls, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", cls, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cls", _cases())
def test_hipcc_compiles_the_fock_class(cls, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", cls, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_fock.py --write")
    write_goldens()
