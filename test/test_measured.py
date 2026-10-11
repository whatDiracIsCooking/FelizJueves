"""The measured store (data/measured/kernels.json) and the generator's CLI.

Fast tier, except the two `cuda`/`hip` re-measure checks at the bottom, which
need a compiler (no card): they recompile a few recorded classes and assert
the toolchain still reports what the store says -- when the toolchain is the
one the row was keyed on. A different compiler version is a different key,
and the check SKIPS naming it rather than comparing across compilers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from felizjueves import measured
from felizjueves.device import GFX1200, SM_86, MeasuredKey
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.kernel import build_pq_kernel, pq_spec

# --- the generator CLI ------------------------------------------------------


def test_the_cli_writes_one_unit_per_class_in_the_backend_spelling(tmp_path) -> None:
    assert emit_cli.main(["--backend", "cuda", "--out", str(tmp_path / "c"),
                          "--L", "0", "4"]) == 0
    assert emit_cli.main(["--backend", "hip", "--out", str(tmp_path / "h"),
                          "--L", "4"]) == 0
    assert sorted(p.name for p in (tmp_path / "c").iterdir()) == [
        "pq_L0_p0_q0.cu", "pq_L4_p2_q2.cu"]
    k = build_pq_kernel(4)
    assert (tmp_path / "c" / "pq_L4_p2_q2.cu").read_text() == cuda.emit(k, SM_86)
    assert (tmp_path / "h" / "pq_L4_p2_q2.hip").read_text() == hip.emit(k, GFX1200)


def test_the_cli_reads_the_tier_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_TIER_BACKEND", "hip")
    monkeypatch.setenv("KERNEL_TIER_ARCH", "gfx1200")
    monkeypatch.setenv("KERNEL_TIER_OUT", str(tmp_path))
    assert emit_cli.main(["--L", "1"]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["pq_L1_p1_q0.hip"]


def test_the_default_classes_are_l_0_to_16() -> None:
    assert emit_cli.DEFAULT_LS == tuple(range(17))


@pytest.mark.parametrize("argv", [
    ["--out", "x"],  # no backend
    ["--backend", "cuda"],  # no out
    ["--backend", "cuda", "--out", "x", "--arch", "gfx1200"],  # wrong vendor
    ["--backend", "cuda", "--out", "x", "--arch", "sm_90"],  # undeclared
    ["--bogus"],
])
def test_the_cli_exits_2_when_it_cannot_run(argv, monkeypatch, capsys) -> None:
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)
    assert emit_cli.main(argv) == 2


# --- parsing the toolchains' reports ----------------------------------------

PTXAS = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'pq_L12_p6_q6' for 'sm_86'
ptxas info    : Function properties for pq_L12_p6_q6
    12496 bytes stack frame, 18828 bytes spill stores, 40236 bytes spill loads
ptxas info    : Used 255 registers, used 0 barriers, 12496 bytes cumulative stack size
"""

# hipcc's remark lines, as ROCm 7.2.4 prints them (each ends with the tag).
AMD = "".join(
    f"k.hip:36:1: remark: {body} [-Rpass-analysis=kernel-resource-usage]\n"
    for body in (
        "Function Name: pq_L12_p6_q6",
        "    TotalSGPRs: 60",
        "    VGPRs: 192",
        "    ScratchSize [bytes/lane]: 5600",
        "    Dynamic Stack: False",
        "    Occupancy [waves/SIMD]: 8",
        "    SGPRs Spill: 0",
        "    VGPRs Spill: 5225",
        "    LDS Size [bytes/block]: 0",
    ))


def test_ptxas_is_read_for_the_named_entry() -> None:
    assert measured.parse_ptxas(PTXAS, "pq_L12_p6_q6") == {
        "registers": 255, "stack_frame_bytes": 12496,
        "spill_store_bytes": 18828, "spill_load_bytes": 40236}
    with pytest.raises(ValueError):
        measured.parse_ptxas(PTXAS, "pq_L4_p2_q2")


def test_the_amd_remark_is_read_for_the_named_kernel() -> None:
    got = measured.parse_amd_remarks(AMD, "pq_L12_p6_q6")
    assert got["VGPRs"] == 192 and got["VGPRs Spill"] == 5225
    assert got["ScratchSize"] == 5600 and got["SGPRs Spill"] == 0
    assert "Dynamic Stack" not in got  # not a number
    with pytest.raises(ValueError):
        measured.parse_amd_remarks(AMD, "other")


# --- the store --------------------------------------------------------------

STORE = json.loads(measured.STORE.read_text())
ROWS = measured.load()


def test_the_store_says_what_its_rows_are_keyed_on() -> None:
    h = STORE["header"]
    assert "MeasuredKey(spec, arch, toolchain).key()" in h["key"]
    assert "Table 4" in h["not_compared"]
    assert "--remeasure" in h["remeasure"]


def test_every_row_is_keyed_on_its_measured_key() -> None:
    assert ROWS, "no Measured rows recorded"
    by_name = {a.name: a for a in (SM_86, GFX1200)}
    for key, row in ROWS.items():
        L = int(row.kernel.split("_")[1].removeprefix("L"))
        assert build_pq_kernel(L).name == row.kernel
        mk = MeasuredKey(pq_spec(L), by_name[row.arch], row.toolchain)
        assert mk.key() == key


def test_every_row_carries_its_provenance() -> None:
    for key, row in ROWS.items():
        assert row.host and row.gpu and row.date and row.command, key
        assert row.gpu != "ABSENT", f"{key}: measured with no card visible"
        assert row.spill_unit and row.raw, key
        assert len(row.source_sha256) == 64, key
        # The flags are config.sh's, and none of them relaxes floating point.
        assert "fast-math" not in row.command and "fast_math" not in row.command


def test_the_recorded_source_is_what_the_generator_writes_today() -> None:
    """A row describes THIS emitter's text: regenerate-and-remeasure on a diff."""
    import hashlib

    for key, row in ROWS.items():
        L = int(row.kernel.split("_")[1].removeprefix("L"))
        emit, arch = (cuda.emit, SM_86) if row.arch == "sm_86" else (hip.emit, GFX1200)
        text = emit(build_pq_kernel(L), arch)
        assert hashlib.sha256(text.encode()).hexdigest() == row.source_sha256, (
            f"{key}: the emitted text changed since it was measured -- "
            "python -m felizjueves.measured --remeasure")


def test_both_cards_and_every_default_class_are_recorded() -> None:
    for arch in ("sm_86", "gfx1200"):
        kernels = {r.kernel for r in ROWS.values() if r.arch == arch}
        assert kernels == {build_pq_kernel(L).name for L in range(17)}, arch


def test_the_store_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    measured.save(ROWS, p)
    assert measured.load(p) == ROWS


def test_a_measured_row_refuses_a_negative_or_missing_field() -> None:
    row = next(iter(ROWS.values()))
    fields = {**row.__dict__}
    with pytest.raises(ValueError):
        measured.Measured(**{**fields, "registers": -1})
    with pytest.raises(ValueError):
        measured.Measured(**{**fields, "gpu": ""})


# --- #50: schema 1, optional measurements, the read path --------------------

PROV = {"toolchain": "nvcc 13.0", "arch": "sm_86", "host": "h", "gpu": "g",
        "date": "2026-10-11"}


def _key(L: int = 4, arch=SM_86, toolchain: str = "nvcc 13.0") -> MeasuredKey:
    return MeasuredKey(pq_spec(L), arch, toolchain)


# The rows #28 recorded before this schema existed, as they read in the
# header-only file. A row is pinned while its source_sha256 is the one it was
# measured from; a --remeasure after an emitter change may move the numbers,
# never drop the key.
LEGACY_KEYS = {
    f"{pq_spec(L).key()}@{arch}/{tc}" for L in range(17)
    for arch, tc in (("sm_86", "nvcc 13.0"), ("gfx1200", "hipcc / ROCm 7.2.4"))}
LEGACY = {  # key: (registers, spill_store, spill_load, source_sha256[:12])
    "coulomb/d0/(0,0|0,0)/b1/j/literal@gfx1200/hipcc / ROCm 7.2.4":
        (42, 0, 0, "da13d0e3ddc4"),
    "coulomb/d0/(6,0|6,0)/b1/j/literal@gfx1200/hipcc / ROCm 7.2.4":
        (192, 5225, 5225, "983a2d085abd"),
    "coulomb/d0/(8,0|8,0)/b1/j/literal@gfx1200/hipcc / ROCm 7.2.4":
        (192, 22376, 22376, "e45c2fa006ea"),
    "coulomb/d0/(0,0|0,0)/b1/j/literal@sm_86/nvcc 13.0": (50, 0, 0, "d1a96a489a52"),
    "coulomb/d0/(6,0|6,0)/b1/j/literal@sm_86/nvcc 13.0":
        (255, 18828, 40236, "04f0eaf2348f"),
    "coulomb/d0/(8,0|8,0)/b1/j/literal@sm_86/nvcc 13.0":
        (255, 70628, 246632, "30839efc3ee3"),
}


def test_the_rows_28_recorded_survive_the_schema_1_migration() -> None:
    assert len(LEGACY_KEYS) == 34
    missing = sorted(LEGACY_KEYS - set(ROWS))
    assert not missing, f"rows dropped by the migration or the loader: {missing}"
    for key, (regs, store, load, sha) in LEGACY.items():
        r = ROWS[key]
        if r.source_sha256.startswith(sha):
            assert (r.registers, r.spill_store, r.spill_load) == (regs, store, load)


def test_the_loader_keeps_every_field_of_every_row() -> None:
    # The failure mode the amendment names: a row (or a field of one) silently
    # dropped or defaulted by a loader change.
    assert STORE["schema"] == measured.SCHEMA == 1
    assert set(ROWS) == set(STORE["rows"])
    for key, row in ROWS.items():
        assert asdict(row) == STORE["rows"][key], key


def test_the_store_says_what_it_is_and_what_it_is_not() -> None:
    assert "MeasuredKey(spec, arch, toolchain).key()" in STORE["about"]
    assert "Table 4" in STORE["about"]


def test_a_row_written_and_read_back_compares_equal_through_its_key(
        tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    k = _key(4)
    row = measured.Measured(registers=40, compile_s=1.5, **PROV)
    measured.save(measured.merge({}, k, row), p)
    assert measured.load(p) == {k.key(): row}
    assert measured.rows(spec=k.spec, arch=k.arch, store=p) == {k.key(): row}


def test_a_timing_only_row_and_a_compile_only_row_are_both_legal(tmp_path) -> None:
    timing = measured.Measured(runtime_ns=1234.5, repeats=50, spread_ns=3.0, **PROV)
    compile_only = measured.Measured(registers=40, spill_store=0, spill_load=0,
                                     shared_bytes=0, **PROV)
    assert timing.registers is None and compile_only.runtime_ns is None
    p = tmp_path / "m.json"
    measured.save({_key(4).key(): timing, _key(6).key(): compile_only}, p)
    assert measured.load(p)[_key(4).key()] == timing


def test_a_row_that_measures_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="records nothing"):
        measured.Measured(**PROV)


@pytest.mark.parametrize("bad", [{"repeats": 0}, {"runtime_ns": -1.0},
                                 {"registers": 1.5}, {"compile_s": "1"}])
def test_a_bad_measurement_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        measured.Measured(**{"registers": 1, **PROV, **bad})


@pytest.mark.parametrize("missing", measured.PROVENANCE)
def test_a_row_missing_provenance_is_refused(missing, tmp_path) -> None:
    prov = {k: v for k, v in PROV.items() if k != missing}
    with pytest.raises(TypeError):
        measured.Measured(registers=1, **prov)
    with pytest.raises(ValueError):
        measured.Measured(registers=1, **{**PROV, missing: ""})
    # ...and in a file, refused with the path named.
    p = tmp_path / "m.json"
    measured.save({}, p)
    doc = json.loads(p.read_text())
    doc["rows"][_key().key()] = {"registers": 1, **prov}
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=str(p)):
        measured.load(p)


def _doc(tmp_path: Path, **change) -> Path:
    p = tmp_path / "m.json"
    measured.save({_key().key(): measured.Measured(registers=1, **PROV)}, p)
    doc = json.loads(p.read_text())
    for k, v in change.items():
        if v is None:
            doc.pop(k)
        else:
            doc[k] = v
    p.write_text(json.dumps(doc))
    return p


@pytest.mark.parametrize("change", [
    {"schema": None},  # #28's header-only form
    {"schema": 2},
    {"about": None},
    {"rows": []},
    {"rows": {_key().key(): {"registers": 1, "bogus": 2, **PROV}}},
    {"rows": {"not a key": {"registers": 1, **PROV}}},
    # a row filed under another toolchain's key than the one it records
    {"rows": {_key(toolchain="nvcc 13.1").key(): {"registers": 1, **PROV}}},
])
def test_a_malformed_or_foreign_store_is_refused_with_its_path(
        change, tmp_path) -> None:
    p = _doc(tmp_path, **change)
    with pytest.raises(ValueError, match=str(p)):
        measured.load(p)


def test_a_file_that_is_not_json_is_refused_with_its_path(tmp_path) -> None:
    p = tmp_path / "m.json"
    p.write_text("{ not json")
    with pytest.raises(ValueError, match=str(p)):
        measured.load(p)


def test_an_empty_store_is_not_a_missing_one(tmp_path) -> None:
    p = tmp_path / "m.json"
    with pytest.raises(FileNotFoundError):
        measured.load(p)
    measured.save({}, p)
    assert measured.load(p) == {}
    assert measured.rows(store=p) == {}


def test_the_same_key_twice_replaces_rather_than_duplicates() -> None:
    k = _key()
    first = measured.Measured(registers=40, **PROV)
    second = measured.Measured(registers=41, **PROV)
    got = measured.merge(measured.merge({}, k, first), k, second)
    assert got == {k.key(): second}


def test_two_toolchains_on_one_arch_and_spec_do_not_collide() -> None:
    a, b = _key(toolchain="nvcc 13.0"), _key(toolchain="nvcc 13.1")
    ra = measured.Measured(registers=40, **PROV)
    rb = measured.Measured(registers=38, **{**PROV, "toolchain": "nvcc 13.1"})
    got = measured.merge(measured.merge({}, a, ra), b, rb)
    assert len(got) == 2
    assert measured.rows(spec=a.spec, arch=SM_86, store=got) == got
    assert measured.rows(toolchain="nvcc 13.1", store=got) == {b.key(): rb}


def test_merge_refuses_a_row_filed_under_another_cards_key() -> None:
    with pytest.raises(ValueError):
        measured.merge({}, _key(arch=GFX1200),
                       measured.Measured(registers=1, **PROV))


def test_the_read_path_filters_the_real_store() -> None:
    spec = pq_spec(8)
    got = measured.rows(spec=spec)
    assert {r.arch for r in got.values()} == {"sm_86", "gfx1200"}
    assert all(k.startswith(spec.key() + "@") for k in got)
    nv = measured.rows(arch=SM_86)
    assert len(nv) == 17 and measured.rows(arch="sm_86") == nv
    assert measured.rows(spec=spec, arch=GFX1200).keys() == {
        k for k in got if "@gfx1200/" in k}


def test_the_n_batches_axis_reaches_the_builder() -> None:
    """measure() takes a Spec, so #52 can sweep n_batches; the builder refuses
    a split class today, and does so before anything is compiled."""
    b = measured.BACKENDS["cuda"]
    assert b.emit(pq_spec(4)) == b.emit(4)
    split = replace(pq_spec(4), n_batches=3)
    with pytest.raises(ValueError, match="n_batches"):
        measured.measure(b, split, compiler="/nonexistent", toolchain="nvcc 13.0",
                         gpu="g", workdir=Path("/nonexistent"))


def test_ptxas_smem_is_read_when_printed() -> None:
    text = PTXAS.replace("used 0 barriers,", "used 1 barriers, 384 bytes smem,")
    assert measured.parse_ptxas(text, "pq_L12_p6_q6")["smem_bytes"] == 384
    assert "smem_bytes" not in measured.parse_ptxas(PTXAS, "pq_L12_p6_q6")


def test_the_cli_prints_the_store_and_exits_2_on_a_bad_one(tmp_path, capsys) -> None:
    assert measured.main([]) == 0
    assert "registers=" in capsys.readouterr().out
    bad = tmp_path / "m.json"
    bad.write_text("{}")
    assert measured.main(["--store", str(bad)]) == 2


# --- re-measure: the recorded rows against the toolchain here ---------------

RECHECK_LS = (0, 4, 8)


def _recheck(backend_name: str, tmp_path: Path) -> None:
    b = measured.BACKENDS[backend_name]
    compiler = b.compiler()
    toolchain = b.toolchain(compiler)
    for L in RECHECK_LS:
        key = MeasuredKey(pq_spec(L), b.arch, toolchain).key()
        if key not in ROWS:
            pytest.skip(f"no row for {key}: this toolchain is not the recorded "
                        "one (a different key, not a regression)")
        _, row = measured.measure(b, L, compiler=compiler, toolchain=toolchain,
                                  gpu=ROWS[key].gpu, workdir=tmp_path)
        want = ROWS[key]
        assert (row.registers, row.spill_store, row.spill_load) == (
            want.registers, want.spill_store, want.spill_load), key


@pytest.mark.cuda
def test_nvcc_still_reports_the_recorded_rows(nvcc, tmp_path) -> None:
    _recheck("cuda", tmp_path)


@pytest.mark.hip
def test_hipcc_still_reports_the_recorded_rows(hipcc, tmp_path) -> None:
    _recheck("hip", tmp_path)
