"""The split J on Rys quadrature emitted for both vendors: golden texts for
every L <= 4 class in both class modes, digests above it through `(dd|dd)`,
the `--rys` flag on the `--hgp-j-l` / `--hgp-class` path, the measured
store's refusal of a `j.rys` row, and a compile of every unit through nvcc
(sm_86) and hipcc (gfx1200).

THE GOLDENS are `test/golden/{cuda,hip}/rj_*.{cu,hip}`: the units the CLI
path writes with `--rys` (`emit_cli.hgp_specs` on `"j"`, each key's Rys
twin built by `kernel_rj.build_rj_kernel`) for every ordered class at
L <= 4 with each shell `l <= 2` -- the set test_emit_hjk.py pins for
`hj_`: 13 canonical units (`rj_{rep}`) and 50 ordered (`rj_o_*`) per
backend. Above L = 4, through `(dd|dd)` (every shell `l <= 2`, the Rys
rule's reach), the canonical units are pinned by byte count and sha256 in
`test/golden/rj_digests.json` (`slow`: building them is above the fast
cut), as is any unit over `GOLDEN_CAP`. Regenerate (and review the diff)
with

    PYTHONPATH=src uv run python test/test_emit_rj.py --write

THE COMPILE is test_emit_hjk.py's (`_compile`: the tier's flags, the
resource report, a warning in generated text fails, the 8,192 B of shared
staging). NOT YET RUN by nvcc or hipcc: the session that built the kernel
had neither; every golden unit passed clang 18's CUDA front end (device
side, `-fsyntax-only -Wall -Wextra`, a runtime stub) with no diagnostic, a
proxy. The deferral to `slow` is by L, not by a recorded budget.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import test_emit_hjk as THK
from tiers import split_by_l

from felizjueves.emit import __main__ as emit_cli
from felizjueves.eri.hjk.kernel_hj import build_hj_kernel
from felizjueves.eri.hjk.kernel_rj import build_rj_kernel, rj_class_spec, rj_name
from felizjueves.hgp import orbits
from felizjueves.ir import Raw, walk
from felizjueves.performance import measured, measured_store
from felizjueves.performance.device import SM_86, MeasuredKey
from felizjueves.spec import Engine

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
DIGESTS = GOLDEN / "rj_digests.json"
GOLDEN_CAP = THK.GOLDEN_CAP
BACKEND = THK.BACKEND


def _rys(spec):
    return replace(spec, engine=Engine.RYS)


SPECS = [_rys(s) for ordered in (False, True)
         for s in emit_cli.hgp_specs("j", THK.CLASSES, ordered=ordered)]
HIGH = [_rys(s) for s in emit_cli.hgp_specs(
    "j", [c for c in orbits.ordered_classes(8) if sum(c) > 4])]
EVERY = split_by_l([*SPECS, *HIGH], lambda s: sum(s.am), ids=rj_name)


def _text(spec, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(build_rj_kernel(spec), arch)


def _digest(text: str) -> dict:
    data = text.encode()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _pinned(spec, text: str) -> bool:
    """Pinned by digest, not by file: above L = 4 or over the cap."""
    return sum(spec.am) > 4 or len(text.encode()) > GOLDEN_CAP


def _golden(spec, backend: str) -> Path:
    return GOLDEN / backend / f"{rj_name(spec)}.{BACKEND[backend][2]}"


def write_goldens() -> None:
    digests: dict[str, dict] = {}
    for backend, (_, _, ext) in BACKEND.items():
        digests[backend] = {}
        for p in (GOLDEN / backend).glob(f"rj_*.{ext}"):
            p.unlink()
        for spec in [*SPECS, *HIGH]:
            text = _text(spec, backend)
            if _pinned(spec, text):
                digests[backend][rj_name(spec)] = _digest(text)
            else:
                _golden(spec, backend).write_text(text)
    DIGESTS.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


# --- the set and the goldens --------------------------------------------------


def test_the_set_is_hjs_with_the_engine_swapped() -> None:
    assert len(SPECS) == 13 + 50
    assert all(s.engine is Engine.RYS for s in [*SPECS, *HIGH])
    assert [rj_name(s) for s in SPECS] == [
        "r" + THK._name(replace(s, engine=Engine.HGP))[1:] for s in SPECS]
    assert len({rj_name(s) for s in [*SPECS, *HIGH]}) == len(SPECS) + len(HIGH)
    assert max(sum(s.am) for s in HIGH) == 8


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("spec", EVERY)
def test_the_rj_golden_is_unchanged(spec, backend: str) -> None:
    text = _text(spec, backend)
    pinned = json.loads(DIGESTS.read_text())[backend]
    if _pinned(spec, text):
        assert not _golden(spec, backend).exists()
        assert _digest(text) == pinned[rj_name(spec)]
    else:
        assert rj_name(spec) not in pinned
        assert text == _golden(spec, backend).read_text()
    assert not any(isinstance(s, Raw) for s in walk(build_rj_kernel(spec).body))


def test_the_golden_set_is_exactly_the_specs() -> None:
    pinned = json.loads(DIGESTS.read_text())
    assert sorted(pinned) == sorted(BACKEND)
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for p in (GOLDEN / backend).glob(f"rj_*.{ext}")}
        assert files.isdisjoint(pinned[backend])
        assert files | set(pinned[backend]) == {
            rj_name(s) for s in [*SPECS, *HIGH]}


@pytest.mark.parametrize("ordered", [False, True])
def test_the_vendors_differ_only_in_the_prologue(ordered) -> None:
    spec = rj_class_spec(2, 1, 1, 0, ordered=ordered)
    c, h = _text(spec, "cuda"), _text(spec, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]


@pytest.mark.parametrize("ordered", [False, True])
def test_a_unit_is_a_drop_in_for_the_hj_unit(ordered) -> None:
    """The same parameters, so the same signature text; the rule carried in
    place of Boys; one shared staging array, no barrier."""
    spec = rj_class_spec(1, 1, 1, 1, ordered=ordered)
    hj = replace(spec, engine=Engine.HGP)
    assert build_rj_kernel(spec).params == build_hj_kernel(hj).params
    for backend, (emit, arch, _) in BACKEND.items():
        rj, h = _text(spec, backend), emit(build_hj_kernel(hj), arch)

        def sig(t: str) -> str:
            start = t.index("(", t.index("__global__"))
            return t[start:t.index(")", start)]

        assert sig(rj) == sig(h)
        assert "fj_rys<3>(" in rj and "fj_boys" not in rj
        assert rj.count("__shared__") == 1 and "__syncthreads" not in rj


# --- the CLI path -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_rys_hgp_j_l_1_writes_the_canonical_golden_units(backend,
                                                         tmp_path) -> None:
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--rys", "--hgp-j-l", "1") == 0
    want = {rj_name(s) for s in SPECS
            if not s.levers and max(s.am) <= 1}
    assert {p.stem for p in tmp_path.iterdir()} == want and len(want) == 6
    for p in tmp_path.iterdir():
        assert p.read_text() == (GOLDEN / backend / p.name).read_text()


def test_rys_hgp_class_writes_the_j_unit_alone(tmp_path) -> None:
    assert _run(tmp_path, "cuda", "--rys", "--ordered",
                "--hgp-class", "2", "1", "1", "0") == 0
    assert {p.name for p in tmp_path.iterdir()} == {"rj_o_2110.cu"}


@pytest.mark.parametrize("argv", [
    ["--rys", "--hgp-k-l", "1"], ["--rys", "--hgp-j-l", "3"],
    ["--rys", "--hgp-class", "3", "3", "2", "2"],
    ["--rys", "--hgp-j-l", "1", "--L", "2"],
], ids=lambda a: "_".join(a))
def test_a_rys_j_request_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_kernel_gen_cmd_asks_for_no_rys_unit() -> None:
    assert "--rys" not in measured.config_value("KERNEL_GEN_CMD")


# --- the measured store refuses a j.rys row -----------------------------------


def test_merge_refuses_a_rys_j_row() -> None:
    mk = MeasuredKey(rj_class_spec(1, 0, 0, 0), SM_86, "nvcc 13.0")
    assert "/j.rys/" in mk.key()
    row = measured_store.Measured(registers=40, **THK.PROV)
    with pytest.raises(ValueError, match=r"#407"):
        measured_store.merge({}, mk, row)


# --- the compile (a compiler, no card) ----------------------------------------


def _compile(tc, backend: str, spec, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(THK, "_kernel", build_rj_kernel)
    THK._compile(tc, backend, spec, tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize("spec", EVERY)
def test_nvcc_compiles_the_rj_unit(spec, nvcc, tmp_path, monkeypatch) -> None:
    _compile(nvcc, "cuda", spec, tmp_path, monkeypatch)


@pytest.mark.hip
@pytest.mark.parametrize("spec", EVERY)
def test_hipcc_compiles_the_rj_unit(spec, hipcc, tmp_path,
                                    monkeypatch) -> None:
    _compile(hipcc, "hip", spec, tmp_path, monkeypatch)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_rj.py --write")
    write_goldens()
