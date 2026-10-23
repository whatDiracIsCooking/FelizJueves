"""The HGP split J / K emitted for both vendors (#407): golden texts for
every L <= 4 class in both class modes, the `--hgp-j-l` / `--hgp-k-l` /
`--hgp-class` [`--ordered`] path, a compile of every golden unit through
nvcc (sm_86) and hipcc (gfx1200), and the measured store's refusal of an
HGP J / K row.

THE GOLDENS are `test/golden/{cuda,hip}/{hj,hk}_*.{cu,hip}`: the units the
CLI path writes (`emit_cli.hgp_specs`, built by `kernel_hj.build_hj_kernel`
/ `kernel_hk.build_hk_kernel`) for every ordered class at L <= 4 with each
shell `l <= 2` (`hgp.orbits.ordered_classes(4)`: 50 classes, the HGP
engine's built ceiling, d) -- in the CANONICAL mode one unit per orbit
representative (13, `hj_{rep}` / `hk_{rep}`), in the ORDERED mode one per
class (50, `hj_o_*` / `hk_o_*`): 126 units per backend. Every one is under
`GOLDEN_CAP` (512 KiB, the pre-commit large-file cap; the largest is about
100 KB), so no digest file is kept; `test_every_golden_is_under_the_cap`
fails the day one is not, and that unit is then pinned by digest as
test/golden/fock_digests.json pins F's. Regenerate (and review the diff)
with

    PYTHONPATH=src uv run python test/test_emit_hjk.py --write

THE COMPILE. Every golden unit is compiled alone (`-c`) with the kernel
tier's own flags (devtools/config.sh `NVCC_FLAGS` / `HIPCC_FLAGS`, warnings
on) plus each compiler's resource report, and FAILS on any warning located
in the generated unit (#92's filter). It pins the shared-memory footprint
too: `kernel_f`'s ONE staged `ab_sh` per block, `MAX_PRIM_PAIRS x REC`
doubles = 8,192 bytes, which both builders reuse. Every unit compiles in
under `BUDGET_S` = 5 s on both backends (measured on this box), so none is
`slow`; nothing above L = 4 is emitted or compiled here.

Needs a compiler, never a card: the compile cases are `cuda` / `hip`, none
is `gpu`. Running the kernels is #408's.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest

from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.eri.hjk.kernel_hj import build_hj_kernel, hj_class_spec, hj_name
from felizjueves.eri.hjk.kernel_hk import build_hk_kernel, hk_class_spec, hk_name
from felizjueves.hgp import orbits
from felizjueves.ir import Raw, walk
from felizjueves.kernel_f import MAX_PRIM_PAIRS, REC
from felizjueves.performance import measured, measured_store
from felizjueves.performance.device import GFX1200, SM_86, MeasuredKey
from felizjueves.spec import Target

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
GOLDEN_CAP = 512 * 1024  # bytes: .pre-commit-config.yaml's --maxkb=512
BACKEND = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
SHARED_BYTES = MAX_PRIM_PAIRS * len(REC) * 8  # 8,192
BUDGET_S = 5.0

CLASSES = orbits.ordered_classes(4)  # every L <= 4 class, each l <= 2
SPECS = [s for ordered in (False, True) for t in ("j", "k")
         for s in emit_cli.hgp_specs(t, CLASSES, ordered=ordered)]


def _kernel(spec):
    return (build_hj_kernel if spec.target is Target.J
            else build_hk_kernel)(spec)


def _name(spec) -> str:
    return (hj_name if spec.target is Target.J else hk_name)(spec)


def _text(spec, backend: str) -> str:
    emit, arch, _ = BACKEND[backend]
    return emit(_kernel(spec), arch)


def _golden(spec, backend: str) -> Path:
    return GOLDEN / backend / f"{_name(spec)}.{BACKEND[backend][2]}"


def write_goldens() -> None:
    """Rewrite every hj_ / hk_ golden (module docstring)."""
    for backend, (_, _, ext) in BACKEND.items():
        for p in [*(GOLDEN / backend).glob(f"hj_*.{ext}"),
                  *(GOLDEN / backend).glob(f"hk_*.{ext}")]:
            p.unlink()
        for spec in SPECS:
            text = _text(spec, backend)
            if len(text.encode()) > GOLDEN_CAP:
                sys.exit(f"{_name(spec)} is over the {GOLDEN_CAP} B cap: pin "
                         "it by digest (module docstring)")
            _golden(spec, backend).write_text(text)


# --- the set ------------------------------------------------------------------


def test_the_golden_set_is_13_orbits_and_50_ordered_classes_per_target():
    assert len(CLASSES) == 50
    reps = sorted(orbits.canonical_classes(4))
    assert len(reps) == 13
    for t, name, key in (("j", hj_name, hj_class_spec),
                         ("k", hk_name, hk_class_spec)):
        canon = emit_cli.hgp_specs(t, CLASSES)
        assert [tuple(s.am) for s in canon] == reps
        assert [name(s) for s in canon] == [name(key(*r)) for r in reps]
        ordered = emit_cli.hgp_specs(t, CLASSES, ordered=True)
        assert [tuple(s.am) for s in ordered] == list(CLASSES)
        assert all(name(s).startswith(f"h{t}_o_") for s in ordered)
    assert len(SPECS) == 2 * (13 + 50)
    assert len({_name(s) for s in SPECS}) == len(SPECS)


def test_a_canonical_unit_is_built_from_its_representatives_key():
    # Two members of one orbit are one unit, spelled one way.
    for t in ("j", "k"):
        a = emit_cli.hgp_specs(t, [(0, 1, 0, 0)])
        b = emit_cli.hgp_specs(t, [(0, 0, 1, 0), (0, 0, 0, 1)])
        assert a == b and len(a) == 1
        assert tuple(a[0].am) == orbits.representative((0, 1, 0, 0))


def test_no_raw_escape_hatch():
    for spec in SPECS:
        k = _kernel(spec)
        assert not any(isinstance(s, Raw) for s in walk(k.body)), k.name


# --- the goldens --------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKEND))
@pytest.mark.parametrize("spec", SPECS, ids=_name)
def test_the_hjk_golden_is_unchanged(spec, backend: str) -> None:
    assert _text(spec, backend) == _golden(spec, backend).read_text()


def test_the_golden_set_is_exactly_the_specs():
    for backend, (_, _, ext) in BACKEND.items():
        files = {p.stem for pre in ("hj_", "hk_")
                 for p in (GOLDEN / backend).glob(f"{pre}*.{ext}")}
        assert files == {_name(s) for s in SPECS}


def test_every_golden_is_under_the_cap():
    for backend, (_, _, ext) in BACKEND.items():
        for pre in ("hj_", "hk_"):
            for p in (GOLDEN / backend).glob(f"{pre}*.{ext}"):
                assert p.stat().st_size <= GOLDEN_CAP, p


@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 1, 1, 1), (2, 1, 1, 0)],
                         ids=lambda c: "".join(map(str, c)))
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("key", [hj_class_spec, hk_class_spec],
                         ids=["j", "k"])
def test_the_vendors_differ_only_in_the_prologue(cls, ordered, key) -> None:
    spec = key(*cls, ordered=ordered)
    c, h = _text(spec, "cuda"), _text(spec, "hip")
    assert c.split("\n")[2:] == h.split("\n")[3:]
    assert h.split("\n")[2] == "#include <hip/hip_runtime.h>"


def test_the_staging_is_one_shared_array_per_block():
    for backend in BACKEND:
        for key in (hj_class_spec, hk_class_spec):
            text = _text(key(1, 1, 1, 1), backend)
            assert text.count("__shared__") == 1
            assert (f"__shared__ double ab_sh[{MAX_PRIM_PAIRS}][{len(REC)}];"
                    in text)
            assert "__syncthreads" not in text


# --- the CLI path -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_tier_env(monkeypatch):
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)


def _run(out: Path, backend: str, *argv: str) -> int:
    return emit_cli.main(["--backend", backend, "--out", str(out), *argv])


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


@pytest.mark.parametrize("backend", sorted(BACKEND))
def test_hgp_j_l_1_writes_the_canonical_golden_units(backend, tmp_path):
    # LMAX = 1: each shell s or p, (pp|pp) at L = 4 the top.
    _, _, ext = BACKEND[backend]
    assert _run(tmp_path, backend, "--hgp-j-l", "1") == 0
    want = emit_cli.hgp_specs("j", [c for c in CLASSES if max(c) <= 1])
    assert len(want) == 6
    assert _names(tmp_path) == {f"{_name(s)}.{ext}" for s in want}
    for s in want:
        assert (tmp_path / f"{_name(s)}.{ext}").read_text() == \
            _golden(s, backend).read_text()


def test_hgp_k_l_1_ordered_writes_every_ordered_class(tmp_path):
    assert _run(tmp_path, "hip", "--hgp-k-l", "1", "--ordered") == 0
    want = [c for c in CLASSES if max(c) <= 1]
    assert len(want) == 16
    assert _names(tmp_path) == {
        f"{hk_name(hk_class_spec(*c, ordered=True))}.hip" for c in want}
    p = tmp_path / "hk_o_0101.hip"
    assert p.read_text() == (GOLDEN / "hip" / "hk_o_0101.hip").read_text()


def test_hgp_class_writes_both_targets_of_the_class(tmp_path):
    assert _run(tmp_path, "cuda", "--hgp-class", "0", "1", "0", "0") == 0
    rep = "".join(map(str, orbits.representative((0, 1, 0, 0))))
    assert _names(tmp_path) == {f"hj_{rep}.cu", f"hk_{rep}.cu"}
    for n in (f"hj_{rep}", f"hk_{rep}"):
        assert (tmp_path / f"{n}.cu").read_text() == \
            (GOLDEN / "cuda" / f"{n}.cu").read_text()


def test_hgp_class_ordered_writes_the_class_itself(tmp_path):
    assert _run(tmp_path, "cuda", "--ordered", "--hgp-class", "2", "0", "1",
                "1", "--hgp-class", "0", "1", "0", "0") == 0
    assert _names(tmp_path) == {"hj_o_2011.cu", "hk_o_2011.cu",
                                "hj_o_0100.cu", "hk_o_0100.cu"}


def test_the_ladder_and_named_classes_write_the_union_per_target():
    # The J ladder's (ss|ss) and the named (ss|ss) are one unit; each named
    # class is written on both targets.
    got = emit_cli.hgp_selection(0, None, [(0, 0, 0, 0), (1, 0, 0, 0)])
    rep = orbits.representative((1, 0, 0, 0))
    assert [(s.target, tuple(s.am)) for s in got] == [
        (Target.J, (0, 0, 0, 0)), (Target.J, rep),
        (Target.K, (0, 0, 0, 0)), (Target.K, rep)]


def test_an_lmax_above_2_is_accepted_and_nothing_is_emitted_above_l4(
        monkeypatch, tmp_path):
    # The parser and selection take any LMAX >= 0; the writer is stubbed,
    # so no IR is built above L = 4.
    seen = []
    monkeypatch.setattr(emit_cli, "generate_hgp", lambda b, a, out, specs: (
        seen.append(list(specs)) or []))
    assert _run(tmp_path, "cuda", "--hgp-j-l", "3", "--hgp-k-l", "5",
                "--ordered") == 0
    (specs,) = seen
    assert len(specs) == 4 ** 4 + 6 ** 4
    assert {_name(s) for s in specs} >= {"hj_o_3333", "hk_o_5555"}
    calls = []
    monkeypatch.setattr(emit_cli, "hgp_selection", lambda *a, **k: (
        calls.append((a, k)) or []))
    assert _run(tmp_path, "cuda", "--hgp-j-l", "7", "--hgp-class", "9", "0",
                "0", "11") == 0
    assert calls == [((7, None, [[9, 0, 0, 11]]), {"ordered": False})]


def test_a_name_above_nine_is_separated():
    assert hj_name(hj_class_spec(0, 11, 0, 0, ordered=True)) == "hj_o_0_11_0_0"
    assert hk_name(hk_class_spec(1, 10, 0, 0, ordered=True)) == "hk_o_1_10_0_0"


@pytest.mark.parametrize("argv", [
    ["--hgp-j-l", "-1"], ["--hgp-k-l", "-2"], ["--hgp-j-l", "x"],
    ["--hgp-class", "0", "0", "0"],
    ["--hgp-class", "0", "0", "0", "-1"],
    ["--ordered"],
    ["--ordered", "--L", "0"],
    ["--hgp-j-l", "1", "--L", "2"], ["--hgp-j-l", "1", "--pair-l", "1"],
    ["--hgp-k-l", "1", "--one-e-l", "1"], ["--hgp-k-l", "1", "--schwarz-l", "1"],
    ["--hgp-j-l", "1", "--sx-l", "1"], ["--hgp-j-l", "1", "--metric-l", "1"],
    ["--hgp-j-l", "1", "--eri3-l", "1"],
    ["--hgp-class", "0", "0", "0", "0", "--class", "0", "0", "0", "0"],
    ["--hgp-j-l", "1", "--shell-ls", "0", "1"],
    ["--hgp-j-l", "1", "--fock-l", "0"],
    ["--hgp-k-l", "1", "--fock-class", "0", "0", "0", "0"],
    ["--hgp-j-l", "1", "--klever-fused-pq", "1"],
    ["--hgp-j-l", "1", "--n-batches", "3"],
    ["--hgp-j-l", "1", "--admitted-only"],
], ids=lambda a: "_".join(a))
def test_an_hgp_request_it_cannot_honour_is_a_2(argv, tmp_path) -> None:
    assert _run(tmp_path / "out", "cuda", *argv) == 2
    assert not (tmp_path / "out").exists()


def test_kernel_gen_cmd_asks_for_no_hgp_unit():
    cfg = (ROOT / "devtools" / "config.sh").read_text()
    (line,) = [ln for ln in cfg.splitlines()
               if ln.lstrip().startswith("KERNEL_GEN_CMD=")]
    assert "felizjueves.emit" in line and "--hgp" not in line
    assert "--ordered" not in line


# --- the measured store refuses an HGP J / K row (#407's decision) ------------

PROV = {"toolchain": "nvcc 13.0", "arch": "sm_86", "host": "h", "gpu": "g",
        "date": "2026-10-21"}


@pytest.mark.parametrize("key", [hj_class_spec, hk_class_spec],
                         ids=["j", "k"])
@pytest.mark.parametrize("ordered", [False, True])
def test_merge_refuses_an_hgp_jk_row(key, ordered) -> None:
    mk = MeasuredKey(key(1, 0, 0, 0, ordered=ordered), SM_86, "nvcc 13.0")
    row = measured_store.Measured(registers=40, **PROV)
    with pytest.raises(ValueError, match=r"#407"):
        measured_store.merge({}, mk, row)


def test_load_refuses_an_hgp_jk_row(tmp_path) -> None:
    import json

    key = MeasuredKey(hj_class_spec(0, 0, 0, 0), SM_86, "nvcc 13.0").key()
    assert "/j.hgp/" in key
    doc = {"schema": measured_store.SCHEMA, "about": "x",
           "rows": {key: {"registers": 40, **PROV}}}
    path = tmp_path / "kernels.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=r"#407"):
        measured_store.load(path)


def test_the_measured_driver_refuses_an_hgp_jk_cell() -> None:
    for spec in (hj_class_spec(0, 0, 0, 0), hk_class_spec(0, 0, 0, 0)):
        with pytest.raises(ValueError, match=r"#407"):
            measured.cell_kernels(spec)


def test_an_f_row_is_still_admitted() -> None:
    from felizjueves.kernel_f import f_class_spec

    mk = MeasuredKey(f_class_spec(0, 0, 0, 0), SM_86, "nvcc 13.0")
    got = measured_store.merge(
        {}, mk, measured_store.Measured(registers=40, **PROV))
    assert got[mk.key()].target == "f"


# --- the compile (a compiler, no card) ----------------------------------------

_WARNING = r"(?:\((\d+)\): warning|:(\d+)(?::\d+)?: warning:?)\s*(.*)"
RESOURCE = {"cuda": ["-Xptxas", "-v"],
            "hip": ["-Rpass-analysis=kernel-resource-usage"]}
SHARED = {"cuda": r"(\d+) bytes smem", "hip": r"LDS Size \[bytes/block\]: (\d+)"}
FLAGS_VAR = {"cuda": "NVCC_FLAGS", "hip": "HIPCC_FLAGS"}


def _compile(tc, backend: str, spec, tmp_path: Path) -> None:
    import devcache

    emit, arch, ext = BACKEND[backend]
    k = _kernel(spec)
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


@pytest.mark.cuda
@pytest.mark.parametrize("spec", SPECS, ids=_name)
def test_nvcc_compiles_the_hjk_unit(spec, nvcc, tmp_path) -> None:
    _compile(nvcc, "cuda", spec, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("spec", SPECS, ids=_name)
def test_hipcc_compiles_the_hjk_unit(spec, hipcc, tmp_path) -> None:
    _compile(hipcc, "hip", spec, tmp_path)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_emit_hjk.py --write")
    write_goldens()
