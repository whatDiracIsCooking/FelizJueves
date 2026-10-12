"""The measured store (data/measured/kernels.json) and the generator's CLI.

Two halves, as the source is split (#94): the STORE -- the record, the file,
the read path, the subgrid and the verdicts -- is `measured_store`, and its
cases call nothing else; the DRIVER (`measured`: parsing what each compiler
prints, emitting a cell's units, compiling, the CLI) is reached only by the
cases that test it, and by the checks that a row describes the text the
generator writes today.

Fast tier, except the `cuda`/`hip` re-measure checks at the bottom, which
need a compiler (no card): they recompile a few recorded classes, and every
`measured_store.SUBGRID` cell, and assert the toolchain still reports what the store
says -- when the toolchain is the one the row was keyed on. A different
compiler version is a different key, and the check SKIPS naming it rather
than comparing across compilers. The subgrid's L >= 13 cells are also `slow`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from felizjueves import asym, gate, measured, measured_store
from felizjueves.device import GFX1200, SM_86, MeasuredKey
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.kernel import build_pq_kernel, build_pq_kernels, pq_spec
from felizjueves.lower import RECYCLE
from felizjueves.plan.batching import in_batch, partition
from felizjueves.plan.pipeline import plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import is_batch_count

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

STORE = json.loads(measured_store.STORE.read_text())
ROWS = measured_store.load()

# Every Spec a row may be keyed on: the one-batch classes, the subgrid, and
# the subgrid's recycle-lever arm (#55).
RECYCLE_CELLS = measured_store.lever_cells(RECYCLE)
SPECS = {s.key(): s for s in (*(pq_spec(L) for L in range(17)), *measured_store.SUBGRID,
                              *RECYCLE_CELLS, *asym.ASYM_CELLS)}  # + #113's arm
ARCHS = {a.name: a for a in (SM_86, GFX1200)}
BACKEND_OF = {"sm_86": "cuda", "gfx1200": "hip"}


def _spec_of(key: str):
    spec_key, _, _ = measured_store.split_key(key)
    assert spec_key in SPECS, f"{key}: keyed on a Spec no one declared"
    return SPECS[spec_key]


def test_the_store_says_what_its_rows_are_keyed_on() -> None:
    h = STORE["header"]
    assert "MeasuredKey(spec, arch, toolchain).key()" in h["key"]
    assert "Table 4" in h["not_compared"]
    assert "--remeasure" in h["remeasure"]


def test_every_row_is_keyed_on_its_measured_key() -> None:
    assert ROWS, "no Measured rows recorded"
    for key, row in ROWS.items():
        spec = _spec_of(key)
        assert row.kernel == ",".join(k.name for k in measured.cell_kernels(spec))
        mk = measured.measured_key(spec, ARCHS[row.arch], row.toolchain)
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
        spec = _spec_of(key)
        units = measured.BACKENDS[BACKEND_OF[row.arch]].units(spec)
        assert measured.cell_sha256(units) == row.source_sha256, (
            f"{key}: the emitted text changed since it was measured -- "
            "python -m felizjueves.measured --remeasure")
        if spec.n_batches == 1:  # the one-batch hash is the one text's
            emit, arch = (cuda.emit, SM_86) if row.arch == "sm_86" else (
                hip.emit, GFX1200)
            (k,) = measured.cell_kernels(spec)  # build_pq_kernel, lowered
            text = emit(k, arch)
            assert hashlib.sha256(text.encode()).hexdigest() == row.source_sha256


def test_both_cards_and_every_default_class_are_recorded() -> None:
    for arch in ("sm_86", "gfx1200"):
        kernels = {r.kernel for k, r in ROWS.items()
                   if r.arch == arch and _spec_of(k).n_batches == 1
                   and not _spec_of(k).levers}
        assert kernels == {build_pq_kernel(L).name for L in range(17)}, arch


# --- #52: the declared subgrid ----------------------------------------------


def test_the_subgrid_is_the_declared_l_by_n_batches_product() -> None:
    cells = [(s.total_am, s.n_batches) for s in measured_store.SUBGRID]
    # 12 is #60's: "9 batches suffice" cannot be judged from below 9
    assert cells == [(L, b) for L in (7, 10, 13, 16) for b in (1, 3, 6, 9, 12)]
    assert all(s == pq_spec(s.total_am, n_batches=s.n_batches)
               for s in measured_store.SUBGRID)


@pytest.mark.parametrize("spec", measured_store.SUBGRID, ids=lambda s: s.key())
def test_every_subgrid_cell_is_admitted_by_the_plan_layer(spec) -> None:
    # #64 made partition refuse an empty batch; a refused cell is not a cell.
    assert len(plan(spec).dags) == spec.n_batches
    assert len(build_pq_kernels(spec)) == spec.n_batches


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", measured_store.SUBGRID, ids=lambda s: s.key())
def test_every_subgrid_cell_is_recorded_on_both_cards(spec, arch) -> None:
    got = measured_store.rows(spec=spec, arch=arch, store=ROWS)
    assert len(got) == 1, f"{spec.key()}@{arch}: {len(got)} rows -- " \
        "python -m felizjueves.measured --remeasure --subgrid"
    (row,) = got.values()
    assert row.registers is not None and row.spill_store is not None
    assert row.compile_s is not None


def _per_batch(row, field: str) -> list[int]:
    return [row.raw[f"{name}:{field}"] for name in row.kernel.split(",")]


def test_a_split_cell_is_one_row_aggregated_over_its_batch_kernels() -> None:
    split = {k: r for k, r in ROWS.items() if _spec_of(k).n_batches > 1}
    assert split, "no split-cell rows recorded"
    for key, row in split.items():
        names = row.kernel.split(",")
        assert len(names) == _spec_of(key).n_batches, key
        if row.arch == "sm_86":
            regs = _per_batch(row, "registers")
            store = _per_batch(row, "spill_store_bytes")
            load = _per_batch(row, "spill_load_bytes")
        else:
            regs = _per_batch(row, "VGPRs")
            store = load = [v + s for v, s in zip(
                _per_batch(row, "VGPRs Spill"), _per_batch(row, "SGPRs Spill"),
                strict=True)]
        assert row.registers == max(regs), key
        assert (row.spill_store, row.spill_load) == (sum(store), sum(load)), key
        assert all(_per_batch(row, "jsh_dynamic_bytes")), key
        assert "<" in row.command and "_*." in row.command, key


def test_the_dynamic_shared_footprint_is_the_kernels_own_declaration() -> None:
    b = measured.BACKENDS["cuda"]
    (u,) = b.units(pq_spec(4))
    # (2,0|2,0): 10 bra functions p, blockDim.y = 32 lanes, 8-byte doubles
    assert u.jsh_dynamic_bytes == 10 * 32 * 8
    assert [x.jsh_dynamic_bytes for x in b.units(pq_spec(4, n_batches=3))] == [
        9 * 32 * 8] * 3


def test_the_store_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    measured_store.save(ROWS, p)
    assert measured_store.load(p) == ROWS


def test_the_tracked_store_is_rewritten_byte_for_byte(tmp_path: Path) -> None:
    """#94's hazard: the store holds measurements nobody wants to re-take, so
    load -> save must reproduce the tracked file exactly -- header, row order,
    number spelling and all. A refactor of either half that moves a byte here
    would move it in every `--remeasure` too."""
    p = tmp_path / "kernels.json"
    measured_store.save(ROWS, p)
    assert p.read_bytes() == measured_store.STORE.read_bytes()


def test_a_measured_row_refuses_a_negative_or_missing_field() -> None:
    row = next(iter(ROWS.values()))
    fields = {**row.__dict__}
    with pytest.raises(ValueError):
        measured_store.Measured(**{**fields, "registers": -1})
    with pytest.raises(ValueError):
        measured_store.Measured(**{**fields, "gpu": ""})


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
    assert STORE["schema"] == measured_store.SCHEMA == 2
    assert set(ROWS) == set(STORE["rows"])
    for key, row in ROWS.items():
        assert measured_store.as_record(row) == STORE["rows"][key], key
        # ...and a row without counters writes none of their fields (#110)
        assert set(asdict(row)) - set(STORE["rows"][key]) <= {
            *measured_store.COUNTERS, "counter_tool", "counter_unit"}, key


def test_the_store_says_what_it_is_and_what_it_is_not() -> None:
    assert "MeasuredKey(spec, arch, toolchain).key()" in STORE["about"]
    assert "Table 4" in STORE["about"]


def test_a_row_written_and_read_back_compares_equal_through_its_key(
        tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    k = _key(4)
    row = measured_store.Measured(registers=40, compile_s=1.5, **PROV)
    measured_store.save(measured_store.merge({}, k, row), p)
    assert measured_store.load(p) == {k.key(): row}
    assert measured_store.rows(spec=k.spec, arch=k.arch, store=p) == {k.key(): row}


UNKNOWN_CARD = dict.fromkeys(measured_store.GPU_UTIL_FIELDS)  # all null: unknown


def test_a_timing_only_row_and_a_compile_only_row_are_both_legal(tmp_path) -> None:
    timing = measured_store.Measured(runtime_ns=1234.5, repeats=50, spread_ns=3.0,
                                     raw=UNKNOWN_CARD, **PROV)
    compile_only = measured_store.Measured(registers=40, spill_store=0, spill_load=0,
                                           shared_bytes=0, **PROV)
    assert timing.registers is None and compile_only.runtime_ns is None
    p = tmp_path / "m.json"
    measured_store.save({_key(4).key(): timing, _key(6).key(): compile_only}, p)
    assert measured_store.load(p)[_key(4).key()] == timing


@pytest.mark.parametrize("raw", [
    {},  # no reading and no null: absent is not unknown
    {**UNKNOWN_CARD, "wallclock:near:gpu_util_after": 101},
    {**UNKNOWN_CARD, "wallclock:gpu_util_before": 12.5},
], ids=["absent", "over-100", "not-an-int"])
def test_a_timed_row_without_the_cards_state_is_refused(raw, tmp_path) -> None:
    # #93: a timed row records the card's state, or null for unknown.
    p = tmp_path / "m.json"
    timing = measured_store.Measured(runtime_ns=1, repeats=31, spread_ns=0, raw=raw,
                                     **PROV)
    measured_store.save({_key(4).key(): timing}, p)
    with pytest.raises(ValueError, match=str(p)):
        measured_store.load(p)


def test_a_row_that_measures_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="records nothing"):
        measured_store.Measured(**PROV)


@pytest.mark.parametrize("bad", [{"repeats": 0}, {"runtime_ns": -1.0},
                                 {"registers": 1.5}, {"compile_s": "1"}])
def test_a_bad_measurement_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        measured_store.Measured(**{"registers": 1, **PROV, **bad})


@pytest.mark.parametrize("missing", measured_store.PROVENANCE)
def test_a_row_missing_provenance_is_refused(missing, tmp_path) -> None:
    prov = {k: v for k, v in PROV.items() if k != missing}
    with pytest.raises(TypeError):
        measured_store.Measured(registers=1, **prov)
    with pytest.raises(ValueError):
        measured_store.Measured(registers=1, **{**PROV, missing: ""})
    # ...and in a file, refused with the path named.
    p = tmp_path / "m.json"
    measured_store.save({}, p)
    doc = json.loads(p.read_text())
    doc["rows"][_key().key()] = {"registers": 1, **prov}
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=str(p)):
        measured_store.load(p)


def _doc(tmp_path: Path, **change) -> Path:
    p = tmp_path / "m.json"
    measured_store.save({_key().key(): measured_store.Measured(registers=1, **PROV)}, p)
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
    {"schema": 1},  # before #93: timed rows without the card's state
    {"schema": 3},
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
        measured_store.load(p)


def test_a_file_that_is_not_json_is_refused_with_its_path(tmp_path) -> None:
    p = tmp_path / "m.json"
    p.write_text("{ not json")
    with pytest.raises(ValueError, match=str(p)):
        measured_store.load(p)


def test_an_empty_store_is_not_a_missing_one(tmp_path) -> None:
    p = tmp_path / "m.json"
    with pytest.raises(FileNotFoundError):
        measured_store.load(p)
    measured_store.save({}, p)
    assert measured_store.load(p) == {}
    assert measured_store.rows(store=p) == {}


def test_the_same_key_twice_replaces_rather_than_duplicates() -> None:
    k = _key()
    first = measured_store.Measured(registers=40, **PROV)
    second = measured_store.Measured(registers=41, **PROV)
    got = measured_store.merge(measured_store.merge({}, k, first), k, second)
    assert got == {k.key(): second}


def test_two_toolchains_on_one_arch_and_spec_do_not_collide() -> None:
    a, b = _key(toolchain="nvcc 13.0"), _key(toolchain="nvcc 13.1")
    ra = measured_store.Measured(registers=40, **PROV)
    rb = measured_store.Measured(registers=38, **{**PROV, "toolchain": "nvcc 13.1"})
    got = measured_store.merge(measured_store.merge({}, a, ra), b, rb)
    assert len(got) == 2
    assert measured_store.rows(spec=a.spec, arch=SM_86, store=got) == got
    assert measured_store.rows(toolchain="nvcc 13.1", store=got) == {b.key(): rb}


def test_merge_refuses_a_row_filed_under_another_cards_key() -> None:
    with pytest.raises(ValueError):
        measured_store.merge({}, _key(arch=GFX1200),
                             measured_store.Measured(registers=1, **PROV))


def test_the_read_path_filters_the_real_store() -> None:
    spec = pq_spec(8)
    got = measured_store.rows(spec=spec)
    assert {r.arch for r in got.values()} == {"sm_86", "gfx1200"}
    assert all(k.startswith(spec.key() + "@") for k in got)
    nv = measured_store.rows(arch=SM_86)
    # the one-batch classes L = 0..16, plus the subgrid's split cells, plus
    # the recycle arm of every subgrid cell (#55), plus #113's asym cells
    split = sum(s.n_batches > 1 for s in measured_store.SUBGRID)
    assert len(nv) == 17 + split + len(RECYCLE_CELLS) + len(asym.ASYM_CELLS)
    assert measured_store.rows(arch="sm_86") == nv
    assert measured_store.rows(spec=spec, arch=GFX1200).keys() == {
        k for k in got if "@gfx1200/" in k}


def test_the_n_batches_axis_reaches_the_builder() -> None:
    """measure() takes a Spec; a split cell is its build_pq_kernels units."""
    b = measured.BACKENDS["cuda"]
    assert b.units(pq_spec(4)) == b.units(4)
    assert [u.name for u in b.units(pq_spec(4))] == [build_pq_kernel(4).name]
    split = replace(pq_spec(4), n_batches=3)
    units = b.units(split)
    assert [u.name for u in units] == [k.name for k in build_pq_kernels(split)]
    assert [u.text for u in units] == [cuda.emit(k, SM_86)
                                       for k in build_pq_kernels(split)]


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


# --- #53: wall-clock rows ----------------------------------------------------

SHA = "a" * 64


def test_combine_keeps_the_other_side_when_both_saw_the_same_text() -> None:
    compiled = measured_store.Measured(registers=40, spill_store=0, spill_load=0,
                                       compile_s=1.0, kernel="k", spill_unit="u",
                                       command="nvcc -c", source_sha256=SHA,
                                       raw={"registers": 40}, **PROV)
    timed = measured_store.Measured(runtime_ns=1e6, repeats=31, spread_ns=1e3,
                                    kernel="k", source_sha256=SHA,
                                    raw={"wallclock:min_ns": 9},
                                    **{**PROV, "date": "2026-10-12"})
    got = measured_store.combine(compiled, timed)
    assert (got.registers, got.compile_s, got.runtime_ns, got.repeats,
            got.spread_ns) == (40, 1.0, 1e6, 31, 1e3)
    assert got.raw == {"registers": 40, "wallclock:min_ns": 9}
    assert got.command == "nvcc -c" and got.spill_unit == "u"
    assert got.date == "2026-10-12"  # the latest observation's provenance
    # symmetric: a re-measured compile row keeps the timing
    again = measured_store.combine(got, replace(compiled, registers=41))
    assert (again.registers, again.runtime_ns) == (41, 1e6)


def test_combine_drops_a_side_measured_on_other_text() -> None:
    compiled = measured_store.Measured(registers=40, source_sha256="b" * 64, **PROV)
    timed = measured_store.Measured(runtime_ns=1e6, repeats=31, spread_ns=1e3,
                                    source_sha256=SHA, **PROV)
    assert measured_store.combine(compiled, timed) == timed
    assert measured_store.combine(None, timed) == timed
    other = measured_store.Measured(registers=40, source_sha256=SHA,
                                    **{**PROV, "toolchain": "nvcc 13.1"})
    assert measured_store.combine(other, timed) == timed


def _timed(row) -> bool:
    return row.runtime_ns is not None


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", measured_store.SUBGRID, ids=lambda s: s.key())
def test_every_subgrid_cell_is_timed_on_both_cards(spec, arch) -> None:
    """#53's acceptance: runtime_ns, repeats and spread_ns, with provenance."""
    (row,) = measured_store.rows(spec=spec, arch=arch, store=ROWS).values()
    assert _timed(row), f"{spec.key()}@{arch}: no wall-clock -- run the " \
        "opt-in sweep in test/test_kernel_device.py (FJ_WALLCLOCK_STORE)"
    assert row.repeats is not None and row.repeats >= 2  # one timing is not one
    assert row.spread_ns is not None and row.gpu != "ABSENT"
    # the compile side survived beside it
    assert row.registers is not None and row.compile_s is not None


def test_a_wall_clock_row_is_its_own_quartiles() -> None:
    timed = {k: r for k, r in ROWS.items() if _timed(r)}
    assert timed, "no wall-clock rows recorded"
    for key, r in timed.items():
        w = {k.removeprefix("wallclock:"): v for k, v in r.raw.items()
             if k.startswith("wallclock:")}
        assert w["min_ns"] <= w["q1_ns"] <= r.runtime_ns <= w["q3_ns"] \
            <= w["max_ns"], key
        assert abs(r.spread_ns - (w["q3_ns"] - w["q1_ns"])) <= 1, key
        assert w["near:min_ns"] <= w["near:q1_ns"] <= w["near:median_ns"] \
            <= w["near:q3_ns"] <= w["near:max_ns"], key
        # the box was quiet: at most BUSY_MAX (10%) busy before each input
        assert w["cpu_busy_permille_before"] <= 100, key
        assert w["near:cpu_busy_permille_before"] <= 100, key
        names = r.kernel.split(",")
        assert all(f"{n}:events_median_ns" in w for n in names), key
        assert w["n_bra"] > 0 and w["n_ket"] > 0 and w["warmup"] > 0, key


# The card gate's threshold (the card at most 30% busy), read from the gate
# itself (#109 lifted it into felizjueves.gate) rather than spelled again here.
GPU_BUSY_MAX = gate.GPU_BUSY_MAX


def test_every_timed_row_records_the_cards_state_or_null() -> None:
    # #93's acceptance: a gpu_util reading, or an explicit null (unknown).
    timed = {k: r for k, r in ROWS.items() if _timed(r)}
    for key, r in timed.items():
        for f in measured_store.GPU_UTIL_FIELDS:
            assert f in r.raw, f"{key}: no {f}"
            v = r.raw[f]
            assert v is None or (type(v) is int and 0 <= v <= 100), (key, f, v)
    # The literal SUBGRID rows were re-timed behind the gate by #93: each
    # input's reading is KNOWN, and the gate let it through at <= 30%.
    for spec in measured_store.SUBGRID:
        for arch in ARCHS:
            (r,) = measured_store.rows(spec=spec, arch=arch, store=ROWS).values()
            for f in ("wallclock:gpu_util_before",
                      "wallclock:near:gpu_util_before"):
                v = r.raw[f]
                assert v is not None and v <= GPU_BUSY_MAX, (spec.key(), arch, f)


# --- #110: the counter fields, the counter store, and the profilers' output --
#
# Fast tier: no profiler is run here (the device witnesses are in
# test/test_toolchain.py), so these hold with ncu and rocprofv3 absent.

COUNTED = {"fp64_flops": 2_000, "dram_bytes_read": 1_600, "dram_bytes_written": 800,
           "counter_ns": 500.0}
COUNTER_PROV = {**PROV, "counter_tool": "ncu 2025.3.1.0",
                "counter_unit": measured.NV_COUNTER_UNIT, "command": "ncu ..."}


def _counter_row(**kw) -> measured_store.Measured:
    return measured_store.Measured(**{**COUNTED, **COUNTER_PROV, **kw})


def test_a_row_with_only_counter_fields_is_legal_and_derives_its_ratios() -> None:
    r = _counter_row()
    assert r.registers is None and r.runtime_ns is None
    assert r.dram_bytes == 2_400
    assert r.arithmetic_intensity == 2_000 / 2_400
    assert r.achieved_flops_per_s == 4e9  # 2000 FLOP in 500 ns
    # one side uncounted: the ratio is unknown, not zero
    amd = _counter_row(fp64_flops=None)
    assert amd.arithmetic_intensity is None and amd.achieved_flops_per_s is None
    assert amd.dram_bytes == 2_400
    with pytest.raises(ValueError, match="records nothing"):
        measured_store.Measured(**COUNTER_PROV)


@pytest.mark.parametrize("bad", [{"fp64_flops": -1}, {"dram_bytes_read": 1.5},
                                 {"counter_ns": -2.0}])
def test_a_bad_counter_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        _counter_row(**bad)


def test_every_counter_field_has_its_unit_and_the_header_says_what_they_are() -> None:
    units = measured_store.COUNTER_UNITS
    assert set(measured_store.COUNTERS) | {"arithmetic_intensity",
                                           "achieved_flops_per_s"} == set(units)
    for header in (STORE["header"], measured_store.ROOFLINE_HEADER):
        c = header["counters"]
        assert "ncu" in c and "rocprofv3" in c  # which tool reported them
        assert "NOT comparable to [JEngine, v2, Fig. 7]" in c
        assert "never runtime_ns" in c
        assert all(f in c for f in units)


def test_the_kernel_key_is_the_cell_key_and_the_batch() -> None:
    k = MeasuredKey(pq_spec(10, n_batches=9), GFX1200, "hipcc / ROCm 7.2.4")
    kk = measured_store.kernel_key(k, 3)
    assert kk == k.key() + "#3"
    assert measured_store.split_kernel_key(kk) == (k.key(), 3)
    assert measured_store.n_batches_of(k.key()) == 9
    for bad in (k.key(), k.key() + "#", k.key() + "#03", k.key() + "#-1"):
        with pytest.raises(ValueError):
            measured_store.split_kernel_key(bad)


# #112: the card's state every counter row carries (the gate's readings)
STATE = {"roofline:gpu_util_before": 3, "roofline:gpu_util_after": 0,
         "roofline:cpu_busy_permille_before": 40,
         "roofline:gate_discarded_phases": 0}


def _cell_rows(spec, flops=(100, 200, 300)) -> dict:
    k = MeasuredKey(spec, SM_86, "nvcc 13.0")
    return {measured_store.kernel_key(k, b): _counter_row(
        fp64_flops=f, kernel=f"k{b}", raw={measured.NCU_DFMA: f // 2, **STATE})
        for b, f in enumerate(flops)}


def test_a_counter_row_per_batch_kernel_is_written_and_read_back(tmp_path) -> None:
    spec = pq_spec(4, n_batches=3)
    rows = {**_cell_rows(spec), **_cell_rows(pq_spec(4), flops=(7,))}
    p = tmp_path / "roofline.json"
    measured_store.save_counters(rows, p)
    got = measured_store.load_counters(p)
    assert got == rows
    measured_store.save_counters(got, tmp_path / "again.json")
    assert (tmp_path / "again.json").read_bytes() == p.read_bytes()
    # the per-kernel rows, in plan order, and the cell DERIVED from them
    ks = measured_store.kernel_counters(got, MeasuredKey(spec, SM_86, "nvcc 13.0"))
    assert [r.kernel for r in ks] == ["k0", "k1", "k2"]
    cell = measured_store.cell_counters(got, MeasuredKey(spec, SM_86, "nvcc 13.0"))
    assert cell.fp64_flops == 600
    assert cell.dram_bytes_read == 3 * 1_600 and cell.counter_ns == 3 * 500.0
    assert cell.arithmetic_intensity == 600 / (3 * 2_400)
    assert cell.kernel == "k0,k1,k2"
    assert cell.raw == {f"#{b}:{k}": v for b, f in enumerate((100, 200, 300))
                        for k, v in {measured.NCU_DFMA: f // 2, **STATE}.items()}


def test_the_tracked_counter_store_reads_and_rewrites_byte_for_byte(tmp_path) -> None:
    got = measured_store.load_counters()
    p = tmp_path / "roofline.json"
    measured_store.save_counters(got, p)
    assert p.read_bytes() == measured_store.ROOFLINE.read_bytes()


@pytest.mark.parametrize("change", ["missing", "extra", "timed", "compiled",
                                    "uncounted", "two-tools", "no-command",
                                    "bare-key"])
def test_the_counter_store_refuses_what_is_not_a_complete_counter_cell(
        change, tmp_path) -> None:
    rows = _cell_rows(pq_spec(4, n_batches=3))
    keys = sorted(rows)
    if change == "missing":
        del rows[keys[1]]
    elif change == "extra":
        rows[keys[0][:-1] + "3"] = rows[keys[0]]
    elif change == "timed":  # a counter run's duration is not a timing
        rows[keys[0]] = replace(rows[keys[0]], runtime_ns=1.0)
    elif change == "compiled":
        rows[keys[0]] = replace(rows[keys[0]], registers=40)
    elif change == "uncounted":
        rows[keys[0]] = replace(rows[keys[0]], **dict.fromkeys(
            measured_store.COUNTERS), compile_s=1.0)
    elif change == "two-tools":
        rows[keys[2]] = replace(rows[keys[2]], counter_tool="ncu 2099.1")
    elif change == "no-command":
        rows[keys[0]] = replace(rows[keys[0]], command="")
    else:
        rows[keys[0][:-2]] = rows.pop(keys[0])
    p = tmp_path / "roofline.json"
    with pytest.raises(ValueError, match=str(p)):
        measured_store.save_counters(rows, p)
    # ...and a file holding it is refused on the way in, too
    ok = tmp_path / "ok.json"
    measured_store.save_counters(_cell_rows(pq_spec(4, n_batches=3)), ok)
    doc = json.loads(ok.read_text())
    doc["rows"] = {k: measured_store.as_record(v) for k, v in rows.items()}
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=str(p)):
        measured_store.load_counters(p)


def test_counter_rows_do_not_change_what_the_cell_store_reads(tmp_path) -> None:
    # The 40 literal rows mean one cell each; counter rows live elsewhere.
    assert measured_store.ROOFLINE != measured_store.STORE
    assert not any("#" in k for k in ROWS)


# ncu --csv --print-units base, as ncu 2025.3.1.0 printed it on this box's RTX
# 3080 for test_toolchain.py's counter probe: one row per metric per launch,
# every column as printed (rebuilt from parts only to fit the line length).
NCU_HEAD = ('"ID","Process ID","Process Name","Host Name","Kernel Name",'
            '"Context","Stream","Block Size","Grid Size","Device","CC",'
            '"Section Name","Metric Name","Metric Unit","Metric Value"')
NCU_LAUNCH = ('"0","108619","k_cuda","127.0.0.1",'
              '"fj_counter_probe(double, const double *, double *, long)","1",'
              '"7","(256, 1, 1)","(65536, 1, 1)","0","8.6",'
              '"Command line profiler metrics"')
NCU_CSV = "\n".join([
    "==PROF== Connected to process 108619 (k_cuda)", "ok",
    "==PROF== Disconnected from process 108619", NCU_HEAD,
    *(f'{NCU_LAUNCH},"{m}","{u}","{v}"' for m, u, v in [
        ("dram__bytes_read.sum", "byte", 271641472),
        ("dram__bytes_write.sum", "byte", 134105856),
        ("gpu__time_duration.sum", "ns", 600832),
        ("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", "inst", 0),
        ("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", "inst", 16777216),
        ("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum", "inst", 0)]),
]) + "\n"


def test_ncu_is_read_by_exact_counter_name() -> None:
    (c,) = measured.parse_ncu_csv(NCU_CSV)
    assert c.kernel == "fj_counter_probe"
    assert set(c.raw) == set(measured.NCU_METRICS)
    assert c.fp64_flops == 2 * 16777216  # an FMA is two flops
    assert (c.dram_bytes_read, c.dram_bytes_written) == (271641472, 134105856)
    assert c.counter_ns == 600832.0


def test_a_rescaled_or_missing_ncu_counter_is_refused() -> None:
    with pytest.raises(ValueError, match="Kbyte"):
        measured.parse_ncu_csv(NCU_CSV.replace('"byte","271641472"',
                                               '"Kbyte","265275.5"'))
    dropped = "\n".join(line for line in NCU_CSV.splitlines()
                        if "dram__bytes_write" not in line)
    with pytest.raises(ValueError, match="dram__bytes_write.sum"):
        measured.parse_ncu_csv(dropped)


# rocprofv3's counter_collection.csv: SYNTHETIC values, so the byte
# arithmetic is exercised on non-zero counts (on this box every byte counter
# reads 0: ROCPROF_PASS0/1 below are the RECORDED files, #140). The column
# names are the ones librocprofiler-sdk-tool 1.1.0 writes; the parser reads
# by name, so their order does not matter.
ROCPROF_HEAD = ("Correlation_Id,Dispatch_Id,Agent_Id,Queue_Id,Process_Id,Thread_Id,"
                "Grid_Size,Kernel_Id,Kernel_Name,Workgroup_Size,LDS_Block_Size,"
                "Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,Counter_Name,"
                "Counter_Value,Start_Timestamp,End_Timestamp")


def _rocprof(dispatch: int, kernel: str, counters: dict) -> str:
    return "\n".join(
        f'{dispatch},{dispatch},1,1,9,9,1024,3,"{kernel}",256,0,0,32,0,16,{n},{v},'
        f"1000,{1000 + 250 * dispatch}" for n, v in counters.items())


def test_rocprofv3_is_read_by_exact_counter_name_and_counts_no_flops() -> None:
    counters = {measured.ROCPROF_RD32: 10, measured.ROCPROF_RD64: 20,
                measured.ROCPROF_RD128: 30, measured.ROCPROF_WR: 50,
                measured.ROCPROF_WR64: 40}
    text = "\n".join([ROCPROF_HEAD, _rocprof(2, "pq_b1(double*)", counters),
                      _rocprof(1, "pq_b0", counters)]) + "\n"
    a, b = measured.parse_rocprof_csv(text)
    assert (a.kernel, b.kernel) == ("pq_b0", "pq_b1")  # dispatch order
    assert a.fp64_flops is None  # gfx1200 has no fp64 counter: unknown, not 0
    assert a.dram_bytes_read == 32 * 10 + 64 * 20 + 128 * 30
    assert a.dram_bytes_written == 64 * 40 + 32 * (50 - 40)
    assert (a.counter_ns, b.counter_ns) == (250.0, 500.0)
    assert set(a.raw) == set(measured.ROCPROF_COUNTERS)
    del counters[measured.ROCPROF_WR64]
    with pytest.raises(ValueError, match=measured.ROCPROF_WR64):
        measured.parse_rocprof_csv(ROCPROF_HEAD + "\n" + _rocprof(1, "k", counters))


# The probe of test/test_toolchain.py (2^24 doubles, 384 MiB streamed) under
# `measured.profile("hip", ...)`'s two passes, RECORDED on the RX 9060 XT
# (#140, rocprofv3 1.1.0, ROCm 7.2.4): every byte counter reads 0.
_ROCPROF_REC_HEAD = (
    '"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id",'
    '"Thread_Id","Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size",'
    '"LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count",'
    '"SGPR_Count","Counter_Name","Counter_Value","Start_Timestamp",'
    '"End_Timestamp"')
_PROBE = '"fj_counter_probe(double, double const*, double*, long)"'


def _rec(pid: int, counter: str, start: int, end: int) -> str:
    return (f'3,3,"Agent 1",1,{pid},{pid},16777216,16,{_PROBE},256,0,0,8,0,'
            f'128,"{counter}",0.00000000e+00,{start},{end}')


ROCPROF_PASS0 = "\n".join([_ROCPROF_REC_HEAD] + [
    _rec(35230, c, 411622396805623, 411622398883152)
    for c in ("GL2C_EA_RDREQ_128B_sum", "GL2C_EA_RDREQ_32B_sum",
              "GL2C_EA_RDREQ_64B_sum")]) + "\n"
ROCPROF_PASS1 = "\n".join([_ROCPROF_REC_HEAD] + [
    _rec(35236, c, 411622658835233, 411622660502673)
    for c in ("GL2C_EA_WRREQ_sum", "GL2C_EA_WRREQ_64B_sum")]) + "\n"


def test_rocprofv3_passes_fit_gfx1200s_gl2c_and_cover_every_counter() -> None:
    # #140: five GL2C counters in one pass abort rocprofv3 (error code 38)
    flat = [c for p in measured.ROCPROF_PASSES for c in p]
    assert sorted(flat) == sorted(measured.ROCPROF_COUNTERS)  # each once
    assert all(len(p) <= 4 for p in measured.ROCPROF_PASSES)
    cmds = measured.profile_commands("hip", "rocprofv3", ["./probe"],
                                     kernel_regex="k", outdir=Path("/o"))
    assert len(cmds) == len(measured.ROCPROF_PASSES)
    for i, (cmd, p) in enumerate(zip(cmds, measured.ROCPROF_PASSES,
                                     strict=True)):
        assert cmd[cmd.index("--pmc") + 1:cmd.index("--kernel-include-regex")] \
            == list(p)
        assert cmd[cmd.index("-d") + 1] == f"/o/pass{i}"


def test_the_recorded_passes_join_into_one_launch_of_zero_bytes() -> None:
    (c,) = measured.parse_rocprof_csv(ROCPROF_PASS0, ROCPROF_PASS1)
    assert c.kernel == "fj_counter_probe" and c.fp64_flops is None
    assert c.raw == dict.fromkeys(measured.ROCPROF_COUNTERS, 0)
    assert (c.dram_bytes_read, c.dram_bytes_written) == (0, 0)
    assert c.counter_ns == 411622398883152 - 411622396805623  # the read pass
    with pytest.raises(ValueError, match="different kernels"):
        measured.parse_rocprof_csv(ROCPROF_PASS0, ROCPROF_PASS1.replace(
            "fj_counter_probe(", "other("))
    with pytest.raises(ValueError, match="two passes"):
        measured.parse_rocprof_csv(ROCPROF_PASS0, ROCPROF_PASS0)


def _fake_rocprof(monkeypatch, passes: list[str], report: str = "") -> None:
    monkeypatch.setattr(measured, "_which", lambda tool: f"/x/{tool}")
    monkeypatch.setattr(measured, "counter_tool", lambda *a: "rocprofv3 1.1.0")
    calls = iter(passes)

    def run(cmd, **_):
        out = Path(cmd[cmd.index("-d") + 1])
        out.mkdir(parents=True)
        (out / "counters_counter_collection.csv").write_text(next(calls))
        return type("D", (), {"returncode": 0, "stdout": "", "stderr": report})()
    monkeypatch.setattr(measured.subprocess, "run", run)


def test_rocprofv3_byte_counters_that_all_read_zero_are_refused(
        monkeypatch, tmp_path) -> None:
    # #140: the recorded reading -- a zero that is not a measurement
    _fake_rocprof(monkeypatch, [ROCPROF_PASS0, ROCPROF_PASS1])
    with pytest.raises(measured.CouldNotMeasure) as e:
        measured.profile("hip", [str(tmp_path / "probe")],
                         kernel_regex="fj_counter_probe", workdir=tmp_path)
    assert str(e.value).startswith(measured.ROCPROF_ZERO_BYTES)
    assert "could not measure" in str(e.value)
    # a reading with bytes in it is a launch, with both passes recorded
    _fake_rocprof(monkeypatch, [ROCPROF_PASS0.replace(
        "0.00000000e+00,411622396805623", "7,411622396805623", 1),
        ROCPROF_PASS1], )
    (c,), tool, command = measured.profile(
        "hip", [str(tmp_path / "probe")], kernel_regex="fj_counter_probe",
        workdir=tmp_path / "again")
    assert c.dram_bytes_read == 128 * 7 and tool == "rocprofv3 1.1.0"
    assert command.count("rocprofv3 --pmc") == 2 and "<probe>" in command
    assert "<rocprofv3-out/pass1>" in command and str(tmp_path) not in command


def test_a_rocprofv3_pass_the_card_cannot_collect_is_refused(
        monkeypatch, tmp_path) -> None:
    _fake_rocprof(monkeypatch, [ROCPROF_PASS0], report=(
        "Could not construct profile cfg failed with error code 38: Request "
        "exceeds the capabilities of the hardware to collect"))
    with pytest.raises(measured.CouldNotMeasure, match="pass 0 .*exceeds"):
        measured.profile("hip", ["./probe"], kernel_regex="k",
                         workdir=tmp_path)


# `rocprofv3 --list-avail`, an excerpt RECORDED on the RX 9060 XT (#140,
# rocprofv3 1.1.0, ROCm 7.2.4): the PC-sampling block first, then the
# counter block, each opening with the agent's Name.
ROCPROF_AVAIL = """\
GPU     :\t0
Name    :\tgfx1200
configs :
   Method              :\thost_trap
   Unit                :\ttime

GPU:0
Name:gfx1200
Counter_Name        :\tGL2C_EA_RDREQ_32B_sum
Description         :\tNumber of 32-byte GL2C/EA read requests. Sum over GL2C instances.


Counter_Name        :\tGL2C_EA_RDREQ_64B_sum
Description         :\tNumber of 64-byte GL2C/EA read requests. Sum over GL2C instances.


Counter_Name        :\tSQ_INSTS_VALU
Description         :\tThe number of VALU (Vector ALU) instructions issued. \
The value is returned per-SE (aggregate of values in SIMDs in the SE). See AMD \
ISAs for more information on VALU instructions.


Counter_Name        :\tSQ_WAVE64_INSTS
Description         :\tNumber of instructions issued by wave64 waves. Skipped \
instructions are not counted. {emulated}
"""


def test_rocprofv3_list_avail_is_read_per_agent_and_fp64_counters_picked() -> None:
    got = measured.parse_rocprof_avail(ROCPROF_AVAIL)
    assert got == {"gfx1200": frozenset({
        "GL2C_EA_RDREQ_32B_sum", "GL2C_EA_RDREQ_64B_sum", "SQ_INSTS_VALU",
        "SQ_WAVE64_INSTS"})}
    # a byte counter's "64B" and a wave64 count are not fp64 counters ...
    assert measured.fp64_counters(got["gfx1200"]) == []
    # ... and every fp64 / FLOP spelling rocprofiler-sdk 1.1.0 uses (on CDNA) is
    cdna = {"SQ_INSTS_VALU_ADD_F64", "SQ_INSTS_VALU_MUL_F64",
            "SQ_INSTS_VALU_FMA_F64", "SQ_INSTS_VALU_TRANS_F64",
            "SQ_INSTS_VALU_MFMA_MOPS_F64", "SQ_INSTS_VALU_FLOPS_FP64",
            "TOTAL_64_OPS", "MfmaFlopsF64"}
    assert measured.fp64_counters(cdna | got["gfx1200"]) == sorted(cdna)
    with pytest.raises(ValueError, match="named no counter"):
        measured.parse_rocprof_avail("GPU:0\nName:gfx1200\n")


def test_counter_rows_file_one_row_per_batch_kernel_in_plan_order(tmp_path) -> None:
    b = measured.BACKENDS["cuda"]
    spec = pq_spec(4, n_batches=3)
    units = b.units(spec)
    launches = [measured.KernelCounters(u.name, {"x": i}, 2 * i + 2, 16, 8, 10.0)
                for i, u in reversed(list(enumerate(units)))]
    rows = measured.counter_rows(b, spec, units, launches, toolchain="nvcc 13.0",
                                 gpu="g", tool="ncu 2025.3.1.0", command="ncu ...")
    cell = MeasuredKey(spec, SM_86, "nvcc 13.0")
    assert list(rows) == [measured_store.kernel_key(cell, i) for i in range(3)]
    assert [r.kernel for r in rows.values()] == [u.name for u in units]
    assert [r.fp64_flops for r in rows.values()] == [2, 4, 6]
    assert all(r.runtime_ns is None for r in rows.values())
    p = tmp_path / "roofline.json"
    rows = {k: replace(r, raw={**r.raw, **STATE}) for k, r in rows.items()}
    measured_store.save_counters(rows, p)
    assert measured_store.cell_counters(measured_store.load_counters(p),
                                        cell).fp64_flops == 12
    with pytest.raises(ValueError, match="profiled 0 times"):
        measured.counter_rows(b, spec, units, launches[:2], toolchain="nvcc 13.0",
                              gpu="g", tool="t", command="c")


# --- #112: the roofline rows, the verdict, and the unmeasured card ----------


@pytest.mark.parametrize("change", ["no-state", "unknown-util", "unmeasured",
                                    "uncharacterised"])
def test_a_counter_row_without_card_state_or_on_a_card_without_a_ridge_is_refused(
        change, tmp_path, monkeypatch) -> None:
    rows = _cell_rows(pq_spec(4, n_batches=3))
    k0 = sorted(rows)[0]
    if change == "no-state":
        rows[k0] = replace(rows[k0], raw={measured.NCU_DFMA: 1})
    elif change == "unknown-util":  # UNKNOWN is never idle, and never recorded
        rows[k0] = replace(rows[k0], raw={
            **rows[k0].raw, "roofline:gpu_util_after": None})
    elif change == "unmeasured":  # gfx1200: no rows, by decision
        k = MeasuredKey(pq_spec(4), GFX1200, "hipcc / ROCm 7.2.4")
        rows = {measured_store.kernel_key(k, 0): _counter_row(
            fp64_flops=None, arch="gfx1200", toolchain="hipcc / ROCm 7.2.4",
            raw=STATE)}
    else:  # a card with no measured ceilings has no ridge, so no row
        bare = type(SM_86)(SM_86.vendor, SM_86.name, SM_86.wavefront)
        monkeypatch.setattr(measured_store, "by_name", lambda name: bare)
    with pytest.raises(ValueError, match=str(tmp_path)):
        measured_store.save_counters(rows, tmp_path / "roofline.json")


def test_the_unmeasured_card_is_recorded_with_its_reason_not_as_zero() -> None:
    assert set(measured_store.UNMEASURED) == {"gfx1200"}
    why = measured_store.UNMEASURED["gfx1200"]
    # #140: FINAL, with its mechanism -- the profiler loads (so "cannot
    # load" is no longer the reason) and the missing counter is named
    assert why.startswith("FINAL (#140): ")
    assert "now LOADS" in why and "cannot load" not in why
    assert "libdw.so.1" in why and "fp64" in why
    assert set(measured_store.MISSING_COUNTERS) == set(measured_store.UNMEASURED)
    for name in measured_store.MISSING_COUNTERS["gfx1200"]:
        assert name in why and measured.fp64_counters({name}) == [name]
    assert "gfx1200" not in measured_store.FP64_COUNTER_ARCHS
    # and the bytes: the driver's refusal and the store's reason share a phrase
    zero = "GL2C <-> EA byte counters all read 0"
    assert zero in why and zero in measured.ROCPROF_ZERO_BYTES
    assert ", ".join(measured_store.FP64_COUNTER_ARCHS) in why
    assert measured_store.ROOFLINE_HEADER["could_not_measure"] == \
        measured_store.UNMEASURED
    doc = json.loads(measured_store.ROOFLINE.read_text())
    assert doc["header"]["could_not_measure"] == measured_store.UNMEASURED
    assert not any("@gfx1200/" in k for k in doc["rows"])


def test_memory_bound_is_computed_against_the_measured_ridge() -> None:
    ridge = SM_86.ridge
    for intensity, bound in ((ridge * 0.5, True), (ridge * 2, False)):
        flops = 10**9
        r = _counter_row(fp64_flops=flops, dram_bytes_read=round(
            flops / intensity), dram_bytes_written=0, counter_ns=1e6)
        p = measured_store.roofline_point(r, "k")
        assert p.memory_bound is bound
        assert p.ridge == ridge and p.achieved_flops_per_s == 1e12
        assert p.of_peak == 1e12 / SM_86.fp64_flops_per_s
    # no FLOP count (gfx1200's counters): no point, never a zero
    with pytest.raises(ValueError, match="no measured intensity"):
        measured_store.roofline_point(_counter_row(fp64_flops=None))
    # per kernel AND per cell, read off one set of numbers
    rows = _cell_rows(pq_spec(4, n_batches=3), flops=(100, 20_000, 300))
    cell = MeasuredKey(pq_spec(4, n_batches=3), SM_86, "nvcc 13.0")
    ks = measured_store.kernel_roofline(rows, cell)
    assert [k.memory_bound for k in ks] == [
        100 / 2400 < ridge, 20_000 / 2400 < ridge, 300 / 2400 < ridge]
    c = measured_store.cell_roofline(rows, cell)
    assert c.intensity == 20_400 / (3 * 2_400)
    assert c.achieved_flops_per_s == 20_400 * 1e9 / (3 * 500.0)


COUNTER_ROWS = measured_store.load_counters()


def test_the_tracked_roofline_rows_cover_the_subgrid_on_sm_86_behind_the_gate() -> None:
    cells = {measured_store.split_kernel_key(k)[0] for k in COUNTER_ROWS}
    for spec in measured_store.SUBGRID:
        (cell,) = [c for c in cells if c.startswith(spec.key() + "@sm_86/")]
        ks = measured_store.kernel_counters(COUNTER_ROWS, cell)
        # the kernels profiled are the kernels the compile row names
        (compiled,) = measured_store.rows(spec=spec, arch=SM_86,
                                          store=ROWS).values()
        assert [r.kernel for r in ks] == compiled.kernel.split(",")
        for r in ks:
            assert r.counter_tool.startswith("ncu ")
            assert r.command.startswith("ncu ") and "--metrics" in r.command
            assert all(m in r.raw for m in measured.NCU_METRICS)
            assert r.toolchain == compiled.toolchain and r.gpu == compiled.gpu
            for f in ("roofline:gpu_util_before", "roofline:gpu_util_after"):
                assert r.raw[f] <= GPU_BUSY_MAX, (cell, f)
            assert r.raw["roofline:n_bra"] == compiled.raw["wallclock:n_bra"]
            assert r.raw["roofline:n_ket"] == compiled.raw["wallclock:n_ket"]
        # both verdicts compute; neither is stored
        assert len(measured_store.kernel_roofline(COUNTER_ROWS, cell)) == \
            spec.n_batches
        measured_store.cell_roofline(COUNTER_ROWS, cell)
    assert all(r.arch == "sm_86" for r in COUNTER_ROWS.values())


def test_a_profiler_that_cannot_measure_yields_no_row(monkeypatch, tmp_path) -> None:
    # Absent, refused the counters, or unable to load: CouldNotMeasure, never 0.
    monkeypatch.setattr(measured, "_which", lambda tool: None)
    with pytest.raises(measured.CouldNotMeasure, match="could not measure"):
        measured.profile("cuda", ["./a.out"], kernel_regex="fj_", workdir=tmp_path)

    class Done:
        returncode = 0
        stderr = ""

    for out, match in [
        ("==ERROR== ERR_NVGPUCTRPERM - The user does not have permission",
         "ERR_NVGPUCTRPERM"),
        ("./a.out: error while loading shared libraries: libdw.so.1: cannot open",
         "libdw.so.1"),
        ("==PROF== Disconnected\n", "profiled no launch"),
    ]:
        monkeypatch.setattr(measured, "_which", lambda tool: f"/x/{tool}")
        monkeypatch.setattr(measured.subprocess, "run",
                            lambda *a, _o=out, **k: type("D", (Done,),
                                                         {"stdout": _o})())
        with pytest.raises(measured.CouldNotMeasure, match=match):
            measured.profile("cuda", ["./a.out"], kernel_regex="fj_",
                             workdir=tmp_path)


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


def _recheck_cell(backend_name: str, spec, tmp_path: Path) -> None:
    b = measured.BACKENDS[backend_name]
    compiler = b.compiler()
    toolchain = b.toolchain(compiler)
    key = measured.measured_key(spec, b.arch, toolchain).key()
    if key not in ROWS:
        pytest.skip(f"no row for {key}: this toolchain is not the recorded "
                    "one (a different key, not a regression)")
    want = ROWS[key]
    _, row = measured.measure(b, spec, compiler=compiler, toolchain=toolchain,
                              gpu=want.gpu, workdir=tmp_path)
    assert row.source_sha256 == want.source_sha256, key
    # the stored row also carries the wall-clock side (#53): compare the
    # compiler's own numbers only
    compiled = {k: v for k, v in want.raw.items() if not k.startswith("wallclock:")}
    assert (row.registers, row.spill_store, row.spill_load, row.raw) == (
        want.registers, want.spill_store, want.spill_load, compiled), key


SUBGRID_PARAMS = [
    pytest.param(s, id=s.key(),
                 marks=[pytest.mark.slow] if s.total_am >= 13 else [])
    for s in measured_store.SUBGRID]


@pytest.mark.cuda
@pytest.mark.parametrize("spec", SUBGRID_PARAMS)
def test_nvcc_still_reports_the_subgrid(spec, nvcc, tmp_path) -> None:
    _recheck_cell("cuda", spec, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("spec", SUBGRID_PARAMS)
def test_hipcc_still_reports_the_subgrid(spec, hipcc, tmp_path) -> None:
    _recheck_cell("hip", spec, tmp_path)


# --- #60: do 9 batches suffice? ---------------------------------------------


def _cell(b: int, spill: int, wide: tuple[int, int], near: tuple[int, int],
          L: int = 13) -> tuple[str, measured_store.Measured]:
    """A synthetic compiled-and-timed row: wide/near are (median, IQR)."""
    key = MeasuredKey(pq_spec(L, n_batches=b), SM_86, "nvcc 13.0").key()
    return key, measured_store.Measured(
        registers=255, spill_store=spill, spill_load=spill,
        runtime_ns=wide[0], repeats=31, spread_ns=wide[1], **PROV,
        raw={"wallclock:near:median_ns": near[0], "wallclock:near:q1_ns": 0,
             "wallclock:near:q3_ns": near[1]})


def _verdict(nine, twelve) -> measured_store.TwelveVsNine:
    store = dict([_cell(9, *nine), _cell(12, *twelve)])
    return measured_store.nine_vs_twelve(store, "sm_86", 13)


def test_nine_suffices_when_twelve_is_no_better_on_anything() -> None:
    c = _verdict((100, (1000, 10), (500, 5)), (100, (995, 8), (497, 4)))
    assert not (c.fewer_spills or c.faster_wide or c.faster_near)
    assert c.nine_suffices


@pytest.mark.parametrize("twelve, which", [
    pytest.param((99, (1000, 10), (500, 5)), "fewer_spills", id="one-spill-less"),
    pytest.param((100, (989, 10), (500, 5)), "faster_wide", id="wide-beyond-iqr"),
    pytest.param((100, (1000, 10), (494, 5)), "faster_near", id="near-beyond-iqr"),
])
def test_any_one_win_for_twelve_means_nine_does_not_suffice(twelve, which) -> None:
    c = _verdict((100, (1000, 10), (500, 5)), twelve)
    assert getattr(c, which) and not c.nine_suffices


def test_a_wall_clock_win_is_judged_against_the_larger_iqr() -> None:
    # 20 faster, but 12's own spread is 25: within noise, not a win
    c = _verdict((100, (1000, 10), (500, 5)), (100, (980, 25), (500, 5)))
    assert not c.faster_wide and c.nine_suffices
    # exactly the spread is not beyond it
    c = _verdict((100, (1000, 10), (500, 5)), (100, (990, 10), (500, 5)))
    assert not c.faster_wide


def test_an_untimed_cell_cannot_be_judged() -> None:
    k9, r9 = _cell(9, 100, (1000, 10), (500, 5))
    k12, r12 = _cell(12, 100, (1000, 10), (500, 5))
    untimed = replace(r12, runtime_ns=None, repeats=None, spread_ns=None)
    with pytest.raises(KeyError):
        measured_store.nine_vs_twelve({k9: r9, k12: untimed}, "sm_86", 13)
    with pytest.raises(KeyError):
        measured_store.nine_vs_twelve({k9: r9}, "sm_86", 13)


# The verdict (#60), on the tracked rows. The definition is measured_store's
# docstring, committed before any 12-batch row existed; this pins what it
# said about the rows, cell by cell, so a remeasure that moves it is seen.
# claims.json's `jengine/nine-batches-suffice` record cites this test.
NINE_VS_TWELVE = {
    # (arch, L): (fewer_spills, faster_wide, faster_near) for b 12 against b 9
    ("sm_86", 7): (False, False, False),
    ("sm_86", 10): (False, False, False),
    ("sm_86", 13): (True, False, False),
    ("sm_86", 16): (True, False, False),
    ("gfx1200", 7): (False, False, False),
    ("gfx1200", 10): (False, False, False),
    ("gfx1200", 13): (False, False, False),
    ("gfx1200", 16): (False, False, True),
}


def test_nine_batches_do_not_suffice_on_either_card() -> None:
    got = {}
    for arch, L in NINE_VS_TWELVE:
        c = measured_store.nine_vs_twelve(ROWS, arch, L)
        got[arch, L] = (c.fewer_spills, c.faster_wide, c.faster_near)
    assert got == NINE_VS_TWELVE
    assert {(a, L) for a, L in got} == {
        (a, L) for a in ARCHS for L in measured_store.SUBGRID_LS}
    for arch in ARCHS:
        # 9 suffices on an arch iff at every subgrid L: on neither card
        assert not all(not any(v) for (a, _), v in got.items() if a == arch)


# --- #55: the recycle lever against the literal kernel ----------------------


def test_the_lever_cells_are_the_whole_subgrid_with_the_lever() -> None:
    # Not one class: every L and every batch count, so a sign flip is seen.
    assert [replace(s, levers=()) for s in RECYCLE_CELLS] == list(
        measured_store.SUBGRID)
    assert all(s.levers == (RECYCLE,) for s in RECYCLE_CELLS)
    with pytest.raises(ValueError, match="kernel-layer"):
        measured_store.lever_cells("no-sieve")  # a plan lever: not this A/B


def _ab(control: tuple[int, int], lever: tuple[int, int],
        near_c: tuple[int, int] = (500, 5),
        near_l: tuple[int, int] = (500, 5)) -> measured_store.LeverAB:
    """Synthetic literal + lever rows; (median, IQR) per arm and input."""
    spec = RECYCLE_CELLS[0]
    lit = MeasuredKey(replace(spec, levers=()), SM_86, "nvcc 13.0").key()
    lev = MeasuredKey(spec, SM_86, "nvcc 13.0").key()
    store = {
        lit: measured_store.Measured(registers=255, spill_store=10, spill_load=20,
                                     **PROV),
        lev: measured_store.Measured(
            registers=255, spill_store=10, spill_load=30,
            runtime_ns=lever[0], repeats=31, spread_ns=lever[1], **PROV,
            raw={"wallclock:control:median_ns": control[0],
                 "wallclock:control:q1_ns": 0,
                 "wallclock:control:q3_ns": control[1],
                 "wallclock:control:near:median_ns": near_c[0],
                 "wallclock:control:near:q1_ns": 0,
                 "wallclock:control:near:q3_ns": near_c[1],
                 "wallclock:near:median_ns": near_l[0],
                 "wallclock:near:q1_ns": 0, "wallclock:near:q3_ns": near_l[1]}),
    }
    return measured_store.lever_ab(store, "sm_86", spec)


def test_a_lever_is_judged_both_ways_against_the_larger_iqr() -> None:
    c = _ab((1000, 10), (995, 8))
    assert (c.wide_sign, c.near_sign, c.verdict) == ("neutral", "neutral",
                                                     "neutral")
    assert c.spill == (30, 40)  # reported, literal row then lever row
    assert _ab((1000, 10), (989, 10)).verdict == "win"
    assert _ab((1000, 10), (1011, 10)).verdict == "loss"
    assert _ab((1000, 10), (990, 10)).verdict == "neutral"  # exactly the IQR
    assert _ab((1000, 10), (980, 25)).verdict == "neutral"  # lever's own spread
    assert _ab((1000, 10), (989, 10), (500, 5), (506, 5)).verdict == "mixed"
    assert _ab((1000, 10), (1000, 10), (500, 5), (494, 5)).verdict == "win"


def test_a_lever_row_without_its_control_cannot_be_judged() -> None:
    spec = RECYCLE_CELLS[0]
    lev = MeasuredKey(spec, SM_86, "nvcc 13.0").key()
    lit = MeasuredKey(replace(spec, levers=()), SM_86, "nvcc 13.0").key()
    timed = measured_store.Measured(registers=1, spill_store=0, spill_load=0,
                                    runtime_ns=1, repeats=31, spread_ns=0, **PROV)
    compiled = measured_store.Measured(registers=1, spill_store=0, spill_load=0, **PROV)
    with pytest.raises(KeyError, match="control"):
        measured_store.lever_ab({lev: timed, lit: compiled}, "sm_86", spec)
    with pytest.raises(KeyError):
        measured_store.lever_ab({lit: compiled}, "sm_86", spec)


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", RECYCLE_CELLS, ids=lambda s: s.key())
def test_every_recycle_cell_is_compiled_and_timed_against_a_control(
        spec, arch) -> None:
    """#55's acceptance: two rows per (class, batch count, arch) differing
    only in `levers`, the lever's timed beside its same-session control."""
    (row,) = measured_store.rows(spec=spec, arch=arch, store=ROWS).values()
    (lit,) = measured_store.rows(spec=replace(spec, levers=()), arch=arch,
                                 store=ROWS).values()
    assert row.kernel == lit.kernel and row.source_sha256 != lit.source_sha256
    assert row.registers is not None and row.compile_s is not None
    assert _timed(row) and row.repeats == 31 and row.gpu != "ABSENT", (
        f"{spec.key()}@{arch}: FJ_WALLCLOCK_LEVER=recycle sweep not recorded")
    w = row.raw
    for p in ("wallclock:control:", "wallclock:control:near:"):
        assert w[f"{p}min_ns"] <= w[f"{p}q1_ns"] <= w[f"{p}median_ns"] \
            <= w[f"{p}q3_ns"] <= w[f"{p}max_ns"], (spec.key(), p)
        assert w[f"{p}cpu_busy_permille_before"] <= 100, (spec.key(), p)


# The verdict (#55), on the tracked rows. The definition is measured_store's
# docstring ("A LEVER A/B"), committed before any lever row existed; this pins
# what it said, cell by cell, so a remeasure that moves it is seen.
# (arch, L, n_batches): (wide sign, near sign) of recycle against literal.
# Every cell NEUTRAL on both inputs, except four gfx1200 cells at L <= 10
# where the rule calls the lever SLOWER by a hair over the larger IQR (e.g.
# L=7 b3 near 8.621 -> 8.753 ms, IQR 0.094) -- on machine code that both
# compilers emit IDENTICALLY for the two arms (the PR records the SASS / ISA
# comparison), so those four are the protocol's noise floor (the near-T
# control always runs first), not an effect of the lever. Pinned as the rule
# said them, not as they "should" read: the rule was declared first.
LEVER_AB: dict[tuple[str, int, int], tuple[str, str]] = {
    **{(a, s.total_am, s.n_batches): ("neutral", "neutral")
       for a in ARCHS for s in RECYCLE_CELLS},
    ("gfx1200", 7, 3): ("slower", "slower"),
    ("gfx1200", 7, 6): ("neutral", "slower"),
    ("gfx1200", 7, 9): ("neutral", "slower"),
    ("gfx1200", 10, 1): ("neutral", "slower"),
}


def test_the_recycle_verdict_is_what_the_rows_say() -> None:
    got = {}
    for arch in ARCHS:
        for spec in RECYCLE_CELLS:
            c = measured_store.lever_ab(ROWS, arch, spec)
            got[arch, c.L, c.n_batches] = (c.wide_sign, c.near_sign)
    assert got == LEVER_AB


def test_recycle_stays_opt_in() -> None:
    """The default changes only if recycle WINS at every cell on BOTH arches
    (measured_store.py). It wins at none, on either: `lower()` stays a no-op unless
    asked, and a Spec without the lever is built exactly as before."""
    verdicts = {k: measured_store.lever_ab(ROWS, k[0], replace(
        pq_spec(k[1], n_batches=k[2]), levers=(RECYCLE,))).verdict
        for k in LEVER_AB}
    assert "win" not in verdicts.values()
    assert "stays opt-in" in STORE["header"]["levers"]  # the data says so too
    for s in measured_store.SUBGRID[:5]:
        assert measured.cell_kernels(s) == build_pq_kernels(s)
    assert "double slot[" not in cuda.emit(build_pq_kernel(10), SM_86)


RECYCLE_PARAMS = [
    pytest.param(s, id=s.key(),
                 marks=[pytest.mark.slow] if s.total_am >= 13 else [])
    for s in RECYCLE_CELLS]


@pytest.mark.cuda
@pytest.mark.parametrize("spec", RECYCLE_PARAMS)
def test_nvcc_still_reports_the_recycle_cells(spec, nvcc, tmp_path) -> None:
    _recheck_cell("cuda", spec, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("spec", RECYCLE_PARAMS)
def test_hipcc_still_reports_the_recycle_cells(spec, hipcc, tmp_path) -> None:
    _recheck_cell("hip", spec, tmp_path)


# --- #113: the asymmetric split, a study the generator refuses ---------------


def test_the_study_rule_was_declared_before_its_rows() -> None:
    doc = asym.__doc__
    assert "DECLARED BEFORE ANY ASYMMETRIC ROW EXISTED" in doc
    for part in ("THE CELLS", "THE SPLIT", "THE NEIGHBOURS", "THE CONTROL",
                 "PER INPUT", "PER CELL", "PER ARCH", "THE CLAIM"):
        assert part in doc, part


def test_the_generator_still_refuses_an_asymmetric_count() -> None:
    """The study measures what Spec and partition refuse; it widens neither
    (the witness of jengine/batch-count-multiple-of-3 stays intact)."""
    for n in asym.ASYM_COUNTS:
        assert not is_batch_count(n)
        with pytest.raises(ValueError):
            pq_spec(13, n_batches=n)
        with pytest.raises(ValueError):
            partition(13, n)


def test_the_asym_cells_are_the_declared_set() -> None:
    assert [(c.L, c.n_batches) for c in asym.ASYM_CELLS] == [
        (L, n) for L in (13, 16) for n in (4, 5, 7, 8)]
    for c in asym.ASYM_CELLS:
        lo, hi = c.neighbours
        assert lo in measured_store.SUBGRID and hi in measured_store.SUBGRID
        assert lo.n_batches < c.n_batches < hi.n_batches == lo.n_batches + 3
    with pytest.raises(ValueError):
        asym.AsymCell(13, 6)  # a legal count is the literal arm's
    with pytest.raises(ValueError):
        asym.AsymCell(13, 2)


def test_the_split_is_minimax_then_the_next_largest() -> None:
    want = {4: (2, 1, 1), 5: (2, 2, 1), 7: (3, 2, 2), 8: (3, 3, 2)}
    for c in asym.ASYM_CELLS:
        assert c.ks == want[c.n_batches], c.key()
        assert len(c.batches) == c.n_batches == sum(c.ks)


@pytest.mark.parametrize("cell", asym.ASYM_CELLS, ids=lambda c: c.key())
def test_every_asym_cell_partitions_its_class(cell) -> None:
    asym.check_partition(cell.L, cell.batches)  # the study's own check
    owners = {r: [b for b in cell.batches if in_batch(r, *b)]
              for r in hermite_targets(cell.L)}
    assert all(len(v) == 1 for v in owners.values())
    assert {b for v in owners.values() for b in v} == set(cell.batches)


def test_a_bad_partition_is_refused_by_the_study() -> None:
    with pytest.raises(ValueError, match="2 batches"):
        asym.check_partition(4, (("xyz", 0, 4), ("x", 0, 4)))
    with pytest.raises(ValueError, match="0 batches"):
        asym.check_partition(4, (("x", 0, 4), ("y", 0, 4)))
    with pytest.raises(ValueError, match="no target"):
        asym.check_partition(4, (("x", 0, 4), ("y", 0, 0), ("y", 1, 4),
                                 ("z", 0, 4)))


@pytest.mark.parametrize("L,k", [(5, 2), (13, 2), (13, 3), (16, 3)])
def test_a_symmetric_split_built_by_the_study_is_the_literal_one(L, k) -> None:
    """For (k, k, k) the study's cut and builder reproduce build_pq_kernels
    byte for byte: the asymmetric arm differs from the literal one by the
    count alone."""
    assert asym.split_batches(L, (k, k, k)) == partition(L, 3 * k)
    got = asym.split_kernels(pq_spec(L), (k, k, k))
    want = build_pq_kernels(pq_spec(L, n_batches=3 * k))
    assert [cuda.emit(x, SM_86) for x in got] == [
        cuda.emit(x, SM_86) for x in want]


def test_an_asym_cell_is_keyed_as_its_own_arm() -> None:
    c = asym.AsymCell(13, 4)
    assert c.key() == "coulomb/d0/(7,0|6,0)/b4/j/asym"
    k = measured.measured_key(c, SM_86, "nvcc 13.0")
    assert k.key() == "coulomb/d0/(7,0|6,0)/b4/j/asym@sm_86/nvcc 13.0"
    assert measured_store.split_key(k.key())[1:] == ("sm_86", "nvcc 13.0")
    assert [x.name for x in measured.cell_kernels(c)] == [
        "pq_L13_p7_q6_x0_6", "pq_L13_p7_q6_x7_13", "pq_L13_p7_q6_y0_13",
        "pq_L13_p7_q6_z0_13"]
    # its rows sit beside the literal ones and are invisible to their readers
    lit = MeasuredKey(pq_spec(13, n_batches=3), SM_86, "nvcc 13.0")
    row = measured_store.Measured(registers=1, **PROV)
    store = measured_store.merge(measured_store.merge({}, lit, row), k, row)
    assert list(measured_store.rows(spec=pq_spec(13, n_batches=3),
                                    store=store)) == [lit.key()]


def _asym_store(lo: tuple, a: tuple, hi: tuple, *, spill=(30, 20, 10),
                cell=None) -> tuple[dict, object]:
    """Synthetic rows for one asymmetric cell: each arm is ((wide median,
    IQR), (near median, IQR)); the neighbours' stats are same-session
    controls in the asym row's raw, their literal rows compile-only."""
    cell = cell or asym.AsymCell(13, 4)
    nlo, nhi = cell.neighbours
    raw = {"wallclock:near:median_ns": a[1][0], "wallclock:near:q1_ns": 0,
           "wallclock:near:q3_ns": a[1][1]}
    for n, arm in ((nlo, lo), (nhi, hi)):
        p = asym.control_prefix(n)
        raw |= {f"{p}median_ns": arm[0][0], f"{p}q1_ns": 0,
                f"{p}q3_ns": arm[0][1], f"{p}near:median_ns": arm[1][0],
                f"{p}near:q1_ns": 0, f"{p}near:q3_ns": arm[1][1]}
    store = {}
    for s, sp in ((nlo, spill[0]), (nhi, spill[2])):
        store = measured_store.merge(store, MeasuredKey(s, SM_86, "nvcc 13.0"),
                                     measured_store.Measured(
                                         registers=255, spill_store=sp,
                                         spill_load=sp, **PROV))
    store = measured_store.merge(
        store, measured.measured_key(cell, SM_86, "nvcc 13.0"),
        measured_store.Measured(registers=255, spill_store=spill[1],
                                spill_load=spill[1], runtime_ns=a[0][0],
                                repeats=31, spread_ns=a[0][1], **PROV, raw=raw))
    return store, cell


def test_the_asym_arm_is_judged_against_its_best_neighbour() -> None:
    def verdict(lo, a, hi):
        store, cell = _asym_store(lo, a, hi)
        return asym.asym_vs_neighbours(store, "sm_86", cell)

    # slower than b3 on wide, between them on near (b6 the best there)
    c = verdict(((100, 2), (60, 1)), ((120, 2), (55, 1)), ((150, 2), (50, 1)))
    assert (c.wide_best, c.wide_sign) == (3, "slower")
    assert (c.near_best, c.near_sign) == (6, "slower")
    assert c.verdict == "worse"
    assert c.spill == (60, 40, 20) and c.registers == (255, 255, 255)
    # faster than the best neighbour on ONE input is a win, whatever the other
    c = verdict(((100, 2), (60, 1)), ((120, 2), (45, 1)), ((150, 2), (50, 1)))
    assert c.near_sign == "faster" and c.verdict == "win"
    # within the larger IQR on both: neutral
    c = verdict(((100, 2), (60, 1)), ((101, 3), (50, 4)), ((150, 2), (50, 1)))
    assert (c.wide_sign, c.near_sign, c.verdict) == ("neutral", "neutral",
                                                     "neutral")


def test_the_arch_and_claim_verdicts_follow_the_declared_rule() -> None:
    worse = (((100, 2), (60, 1)), ((120, 2), (65, 1)), ((150, 2), (70, 1)))
    win = (((100, 2), (60, 1)), ((90, 2), (65, 1)), ((150, 2), (70, 1)))
    tie = (((100, 2), (60, 1)), ((100, 2), (60, 1)), ((150, 2), (70, 1)))
    cells = (asym.AsymCell(13, 4), asym.AsymCell(13, 5))

    def store_of(a, b):
        s1, _ = _asym_store(*a, cell=cells[0])
        s2, _ = _asym_store(*b, cell=cells[1])
        return {**s1, **s2}

    assert asym.arch_verdict(store_of(worse, worse), "sm_86", cells) == "worse"
    assert asym.claim_holds(store_of(worse, worse), ["sm_86"], cells)
    assert asym.arch_verdict(store_of(worse, win), "sm_86", cells) == \
        "contradicted"
    assert asym.arch_verdict(store_of(worse, tie), "sm_86", cells) == \
        "not shown"
    assert not asym.claim_holds(store_of(worse, tie), ["sm_86"], cells)


def test_an_asym_row_without_both_controls_cannot_be_judged() -> None:
    store, cell = _asym_store(((100, 2), (60, 1)), ((120, 2), (65, 1)),
                              ((150, 2), (70, 1)))
    k = measured.measured_key(cell, SM_86, "nvcc 13.0").key()
    raw = {f: v for f, v in store[k].raw.items()
           if not f.startswith("wallclock:control:b6:")}
    store[k] = replace(store[k], raw=raw)
    with pytest.raises(KeyError, match="neighbours"):
        asym.asym_vs_neighbours(store, "sm_86", cell)
    with pytest.raises(KeyError):
        asym.asym_vs_neighbours({}, "sm_86", cell)


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("cell", asym.ASYM_CELLS, ids=lambda c: c.key())
def test_every_asym_cell_is_compiled_and_timed_against_both_neighbours(
        cell, arch) -> None:
    """#113's acceptance: the asymmetric row, compiled and timed behind the
    gate, with the card's state for itself AND each same-session control."""
    (r,) = measured_store.rows(spec=cell, arch=arch, store=ROWS).values()
    assert r.registers is not None and r.spill_store is not None
    assert _timed(r) and r.repeats == 31 and r.gpu != "ABSENT"
    assert r.kernel == ",".join(k.name for k in cell.kernels())
    assert r.raw["wallclock:gate_discarded_phases"] >= 0
    for n in cell.neighbours:
        p = asym.control_prefix(n)
        for q in ("", "near:"):
            assert r.raw[f"{p}{q}q1_ns"] <= r.raw[f"{p}{q}median_ns"] \
                <= r.raw[f"{p}{q}q3_ns"], (p, q)
            assert r.raw[f"{p}{q}cpu_busy_permille_before"] <= 100
            # the gate let it through quiet, and nothing took the card
            # during it (a phase read otherwise was discarded and retaken)
            for when in ("before", "after"):
                v = r.raw[f"{p}{q}gpu_util_{when}"]
                assert v is not None and v <= GPU_BUSY_MAX, (p, q, when, v)
    for f in measured_store.GPU_UTIL_FIELDS:
        assert r.raw[f] is not None and r.raw[f] <= GPU_BUSY_MAX, f


# What the declared rule (felizjueves.asym's docstring, committed before any
# asymmetric row existed) reads off the rows: (wide sign, near sign) of the
# asymmetric arm against its best same-session legal neighbour. Pinned as the
# rule's output, not tuned: a re-timing that moves a sign is news.
ASYM_VERDICT: dict[tuple[str, int, int], tuple[str, str]] = {
    **{("sm_86", L, n): ("slower", "slower") for L in (13, 16)
       for n in (4, 5, 7, 8)},
    ("gfx1200", 13, 4): ("slower", "neutral"),
    ("gfx1200", 13, 5): ("slower", "slower"),
    ("gfx1200", 13, 7): ("slower", "neutral"),
    ("gfx1200", 13, 8): ("slower", "slower"),
    ("gfx1200", 16, 4): ("slower", "neutral"),
    ("gfx1200", 16, 5): ("slower", "slower"),
    ("gfx1200", 16, 7): ("slower", "slower"),
    ("gfx1200", 16, 8): ("slower", "slower"),
}


def test_the_asym_verdict_is_what_the_rows_say() -> None:
    got = {(a, c.L, c.n_batches): (v.wide_sign, v.near_sign)
           for a in ARCHS for c in asym.ASYM_CELLS
           for v in [asym.asym_vs_neighbours(ROWS, a, c)]}
    assert got == ASYM_VERDICT


@pytest.mark.claim("jengine/symmetry-breaking-splits-are-worse")
def test_asymmetric_splits_are_worse_on_both_cards() -> None:
    """[JEngine, v2, §5]: a split breaking the three-way symmetry is worse.
    Under the rule declared before any asymmetric row existed, every
    asymmetric cell is slower than its best legal neighbour on at least one
    input and faster on none, on sm_86 AND on gfx1200. An asymmetric split
    winning anywhere -- or merely tying everywhere on one card -- fails it."""
    for arch in ARCHS:
        assert asym.arch_verdict(ROWS, arch) == asym.WORSE, arch
    assert asym.claim_holds(ROWS, tuple(ARCHS))
    assert "asserted" in STORE["header"]["asym"]


ASYM_PARAMS = [pytest.param(c, id=c.key(), marks=[pytest.mark.slow])
               for c in asym.ASYM_CELLS]


@pytest.mark.cuda
@pytest.mark.parametrize("cell", ASYM_PARAMS)
def test_nvcc_still_reports_the_asym_cells(cell, nvcc, tmp_path) -> None:
    _recheck_cell("cuda", cell, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("cell", ASYM_PARAMS)
def test_hipcc_still_reports_the_asym_cells(cell, hipcc, tmp_path) -> None:
    _recheck_cell("hip", cell, tmp_path)


# --- #146: the build store (data/measured/builds.json) ----------------------

BUILD_ARCHS = ("sm_86", "gfx1200")


def _build_row(**over) -> measured_store.BuildRow:
    raw = {"build:gpu_util_before": 3, "build:gpu_util_after": 0,
           "build:gpu_busy_max": 30, "build:cpu_busy_permille_before": 12,
           "build:gate_discarded_phases": 0, "build:upload_ns": 1234}
    raw.update(over.pop("raw", {}))
    kw = dict(fixture="h2o_ccpvdz_j", tau=1e-3, n_batches="dispatch:wide",
              density_sort=True, arm="baseline", runtime_ns=100_000,
              spread_ns=2_000, repeats=31, toolchain="nvcc 13.0",
              arch="sm_86", host="box", gpu="RTX", date="2026-10-12",
              command="nvcc <driver> -o <binary>", source_sha256="ab" * 32,
              raw={k: v for k, v in raw.items() if v is not None})
    kw.update(over)
    return measured_store.BuildRow(**kw)


def _build_doc(rows: dict) -> dict:
    return {"schema": measured_store.BUILD_SCHEMA, "about": "x",
            "rows": {k: asdict(r) for k, r in rows.items()}}


BUILDS = measured_store.load_builds()


def test_the_tracked_build_store_is_rewritten_byte_for_byte(tmp_path) -> None:
    """load -> save reproduces data/measured/builds.json exactly: header,
    row order, number spelling and all."""
    p = tmp_path / "builds.json"
    measured_store.save_builds(BUILDS, p)
    assert p.read_bytes() == measured_store.BUILDS.read_bytes()


def test_a_build_row_round_trips(tmp_path) -> None:
    rows = {r.key(): r for r in (
        _build_row(), _build_row(density_sort=False, arm="global"),
        _build_row(n_batches=3, tau=0.0, arch="gfx1200",
                   toolchain="hipcc / ROCm 7.2.4"))}
    p = tmp_path / "b.json"
    measured_store.save_builds(rows, p)
    assert measured_store.load_builds(p) == rows
    text = p.read_text()
    measured_store.save_builds(measured_store.load_builds(p), p)
    assert p.read_text() == text


def test_both_cards_have_the_baseline_build_rows_with_their_gate_state() -> None:
    """#146's acceptance: rows for both cards, each with the card's state as
    KNOWN gate readings (the loader refuses anything else), or the refusal
    recorded in the header."""
    from test_j_launched_ondevice import BUILD_CELLS
    doc = json.loads(measured_store.BUILDS.read_text())
    assert doc["header"]["could_not_measure"] == measured_store.BUILD_UNMEASURED
    assert doc["header"]["timed_region"] == measured_store.BUILD_TIMED_REGION
    for arch in BUILD_ARCHS:
        if arch in measured_store.BUILD_UNMEASURED:
            assert not [r for r in BUILDS.values() if r.arch == arch]
            continue
        on = {k.split("@")[0]: r for k, r in BUILDS.items() if r.arch == arch}
        want = {c.key(arch, "t").split("@")[0] for c in BUILD_CELLS}
        assert want <= set(on), (arch, sorted(want - set(on)))
        for r in on.values():
            for f in measured_store.BUILD_STATE_FIELDS:
                assert type(r.raw[f]) is int, (r.key(), f)
            assert r.raw["build:gpu_util_before"] <= r.raw["build:gpu_busy_max"]
            assert r.raw["build:gpu_util_after"] <= r.raw["build:gpu_busy_max"]
            assert r.raw["build:gpu_busy_max"] == gate.GPU_BUSY_MAX
            assert r.repeats >= 1
            assert r.raw["build:min_ns"] <= r.runtime_ns <= r.raw["build:max_ns"]
            assert r.raw["build:device_median_ns"] <= r.raw["build:max_ns"]


def test_build_key_spells_every_field() -> None:
    k = measured_store.build_key("h2o_sto3g_j", 0.0, 1, True, "baseline",
                                 "sm_86", "nvcc 13.0")
    assert k == "h2o_sto3g_j/tau=0.0/b=1/sort=density/baseline@sm_86/nvcc 13.0"
    assert measured_store.build_key(
        "h2o_ccpvdz_j", 1e-3, "dispatch:wide", False, "global", "gfx1200",
        "hipcc / ROCm 7.2.4") == ("h2o_ccpvdz_j/tau=0.001/b=dispatch:wide/"
                                  "sort=global/global@gfx1200/hipcc / ROCm 7.2.4")


@pytest.mark.parametrize("bad", [
    dict(n_batches=0), dict(n_batches="wide"), dict(n_batches="dispatch:"),
    dict(tau=1), dict(tau=-1.0), dict(density_sort=1), dict(arm=""),
    dict(arm="a/b"), dict(fixture="x@y"), dict(repeats=0),
    dict(runtime_ns=1.5), dict(spread_ns=-1), dict(host=""),
    dict(source_sha256=""), dict(command=""),
])
def test_a_malformed_build_row_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        _build_row(**bad)


@pytest.mark.parametrize("field", measured_store.BUILD_STATE_FIELDS)
def test_a_build_row_without_its_card_state_is_refused(field) -> None:
    """No row without a recorded card state is read as idle: a missing or
    null reading is refused, never defaulted."""
    with pytest.raises(ValueError, match=field):
        _build_row(raw={field: None})
    raw = {**_build_row().raw, field: None}
    with pytest.raises(ValueError):
        _build_row(raw=raw)


@pytest.mark.parametrize("field", ["build:gpu_util_before",
                                   "build:gpu_util_after"])
def test_a_contended_build_row_is_refused(field) -> None:
    """A phase over the gate's threshold is discarded, never recorded."""
    with pytest.raises(ValueError, match="contended"):
        _build_row(raw={field: 31})
    _build_row(raw={field: 30})  # at the threshold: let through


def test_the_build_loader_refuses(tmp_path, monkeypatch) -> None:
    p = tmp_path / "b.json"
    row = _build_row()

    def refused(doc, match):
        p.write_text(json.dumps(doc))
        with pytest.raises(ValueError, match=match):
            measured_store.load_builds(p)

    refused({**_build_doc({row.key(): row}), "schema": 99}, "schema")
    refused({**_build_doc({row.key(): row}), "about": ""}, "about")
    refused({**_build_doc({}), "rows": []}, "rows")
    refused(_build_doc({"elsewhere@sm_86/nvcc 13.0": row}), "records the key")
    doc = _build_doc({row.key(): row})
    doc["rows"][row.key()]["bogus"] = 1
    refused(doc, "unknown fields")
    doc = _build_doc({row.key(): row})
    doc["rows"][row.key()]["raw"]["build:gpu_util_after"] = None
    refused(doc, "build:gpu_util_after")
    del doc["rows"][row.key()]["raw"]["build:gpu_util_after"]
    refused(doc, "build:gpu_util_after")
    p.write_text("{")
    with pytest.raises(ValueError, match="not JSON"):
        measured_store.load_builds(p)
    with pytest.raises(FileNotFoundError):
        measured_store.load_builds(tmp_path / "absent.json")
    monkeypatch.setitem(measured_store.BUILD_UNMEASURED, "sm_86", "refused")
    refused(_build_doc({row.key(): row}), "by decision")


def test_save_builds_refuses_a_row_filed_under_another_key(tmp_path) -> None:
    row = _build_row()
    with pytest.raises(ValueError, match="filed at"):
        measured_store.save_builds({"x": row}, tmp_path / "b.json")


def test_an_arm_is_its_own_row() -> None:
    """#147's A/B: two arms of one cell are two rows, never one replaced."""
    a = _build_row()
    b = _build_row(density_sort=False, arm="global-dmax")
    store = measured_store.merge_build(measured_store.merge_build({}, a), b)
    assert set(store) == {a.key(), b.key()} and a.key() != b.key()
