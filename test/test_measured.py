"""The measured store (data/measured/kernels.json) and the generator's CLI.

Fast tier, except the `cuda`/`hip` re-measure checks at the bottom, which
need a compiler (no card): they recompile a few recorded classes, and every
`measured.SUBGRID` cell, and assert the toolchain still reports what the store
says -- when the toolchain is the one the row was keyed on. A different
compiler version is a different key, and the check SKIPS naming it rather
than comparing across compilers. The subgrid's L >= 13 cells are also `slow`.
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
from felizjueves.kernel import build_pq_kernel, build_pq_kernels, pq_spec
from felizjueves.lower import RECYCLE
from felizjueves.plan.pipeline import plan

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

# Every Spec a row may be keyed on: the one-batch classes, the subgrid, and
# the subgrid's recycle-lever arm (#55).
RECYCLE_CELLS = measured.lever_cells(RECYCLE)
SPECS = {s.key(): s for s in (*(pq_spec(L) for L in range(17)), *measured.SUBGRID,
                              *RECYCLE_CELLS)}
ARCHS = {a.name: a for a in (SM_86, GFX1200)}
BACKEND_OF = {"sm_86": "cuda", "gfx1200": "hip"}


def _spec_of(key: str):
    spec_key, _, _ = measured.split_key(key)
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
        mk = MeasuredKey(spec, ARCHS[row.arch], row.toolchain)
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
    cells = [(s.total_am, s.n_batches) for s in measured.SUBGRID]
    # 12 is #60's: "9 batches suffice" cannot be judged from below 9
    assert cells == [(L, b) for L in (7, 10, 13, 16) for b in (1, 3, 6, 9, 12)]
    assert all(s == pq_spec(s.total_am, n_batches=s.n_batches)
               for s in measured.SUBGRID)


@pytest.mark.parametrize("spec", measured.SUBGRID, ids=lambda s: s.key())
def test_every_subgrid_cell_is_admitted_by_the_plan_layer(spec) -> None:
    # #64 made partition refuse an empty batch; a refused cell is not a cell.
    assert len(plan(spec).dags) == spec.n_batches
    assert len(build_pq_kernels(spec)) == spec.n_batches


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", measured.SUBGRID, ids=lambda s: s.key())
def test_every_subgrid_cell_is_recorded_on_both_cards(spec, arch) -> None:
    got = measured.rows(spec=spec, arch=arch, store=ROWS)
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
    # the one-batch classes L = 0..16, plus the subgrid's split cells, plus
    # the recycle arm of every subgrid cell (#55)
    split = sum(s.n_batches > 1 for s in measured.SUBGRID)
    assert len(nv) == 17 + split + len(RECYCLE_CELLS)
    assert measured.rows(arch="sm_86") == nv
    assert measured.rows(spec=spec, arch=GFX1200).keys() == {
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
    compiled = measured.Measured(registers=40, spill_store=0, spill_load=0,
                                 compile_s=1.0, kernel="k", spill_unit="u",
                                 command="nvcc -c", source_sha256=SHA,
                                 raw={"registers": 40}, **PROV)
    timed = measured.Measured(runtime_ns=1e6, repeats=31, spread_ns=1e3,
                              kernel="k", source_sha256=SHA,
                              raw={"wallclock:min_ns": 9},
                              **{**PROV, "date": "2026-10-12"})
    got = measured.combine(compiled, timed)
    assert (got.registers, got.compile_s, got.runtime_ns, got.repeats,
            got.spread_ns) == (40, 1.0, 1e6, 31, 1e3)
    assert got.raw == {"registers": 40, "wallclock:min_ns": 9}
    assert got.command == "nvcc -c" and got.spill_unit == "u"
    assert got.date == "2026-10-12"  # the latest observation's provenance
    # symmetric: a re-measured compile row keeps the timing
    again = measured.combine(got, replace(compiled, registers=41))
    assert (again.registers, again.runtime_ns) == (41, 1e6)


def test_combine_drops_a_side_measured_on_other_text() -> None:
    compiled = measured.Measured(registers=40, source_sha256="b" * 64, **PROV)
    timed = measured.Measured(runtime_ns=1e6, repeats=31, spread_ns=1e3,
                              source_sha256=SHA, **PROV)
    assert measured.combine(compiled, timed) == timed
    assert measured.combine(None, timed) == timed
    other = measured.Measured(registers=40, source_sha256=SHA,
                              **{**PROV, "toolchain": "nvcc 13.1"})
    assert measured.combine(other, timed) == timed


def _timed(row) -> bool:
    return row.runtime_ns is not None


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", measured.SUBGRID, ids=lambda s: s.key())
def test_every_subgrid_cell_is_timed_on_both_cards(spec, arch) -> None:
    """#53's acceptance: runtime_ns, repeats and spread_ns, with provenance."""
    (row,) = measured.rows(spec=spec, arch=arch, store=ROWS).values()
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
    key = MeasuredKey(spec, b.arch, toolchain).key()
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
    for s in measured.SUBGRID]


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
          L: int = 13) -> tuple[str, measured.Measured]:
    """A synthetic compiled-and-timed row: wide/near are (median, IQR)."""
    key = MeasuredKey(pq_spec(L, n_batches=b), SM_86, "nvcc 13.0").key()
    return key, measured.Measured(
        registers=255, spill_store=spill, spill_load=spill,
        runtime_ns=wide[0], repeats=31, spread_ns=wide[1], **PROV,
        raw={"wallclock:near:median_ns": near[0], "wallclock:near:q1_ns": 0,
             "wallclock:near:q3_ns": near[1]})


def _verdict(nine, twelve) -> measured.TwelveVsNine:
    store = dict([_cell(9, *nine), _cell(12, *twelve)])
    return measured.nine_vs_twelve(store, "sm_86", 13)


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
        measured.nine_vs_twelve({k9: r9, k12: untimed}, "sm_86", 13)
    with pytest.raises(KeyError):
        measured.nine_vs_twelve({k9: r9}, "sm_86", 13)


# The verdict (#60), on the tracked rows. The definition is measured.py's
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
        c = measured.nine_vs_twelve(ROWS, arch, L)
        got[arch, L] = (c.fewer_spills, c.faster_wide, c.faster_near)
    assert got == NINE_VS_TWELVE
    assert {(a, L) for a, L in got} == {
        (a, L) for a in ARCHS for L in measured.SUBGRID_LS}
    for arch in ARCHS:
        # 9 suffices on an arch iff at every subgrid L: on neither card
        assert not all(not any(v) for (a, _), v in got.items() if a == arch)


# --- #55: the recycle lever against the literal kernel ----------------------


def test_the_lever_cells_are_the_whole_subgrid_with_the_lever() -> None:
    # Not one class: every L and every batch count, so a sign flip is seen.
    assert [replace(s, levers=()) for s in RECYCLE_CELLS] == list(measured.SUBGRID)
    assert all(s.levers == (RECYCLE,) for s in RECYCLE_CELLS)
    with pytest.raises(ValueError, match="kernel-layer"):
        measured.lever_cells("no-sieve")  # a plan lever: not this A/B


def _ab(control: tuple[int, int], lever: tuple[int, int],
        near_c: tuple[int, int] = (500, 5),
        near_l: tuple[int, int] = (500, 5)) -> measured.LeverAB:
    """Synthetic literal + lever rows; (median, IQR) per arm and input."""
    spec = RECYCLE_CELLS[0]
    lit = MeasuredKey(replace(spec, levers=()), SM_86, "nvcc 13.0").key()
    lev = MeasuredKey(spec, SM_86, "nvcc 13.0").key()
    store = {
        lit: measured.Measured(registers=255, spill_store=10, spill_load=20,
                               **PROV),
        lev: measured.Measured(
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
    return measured.lever_ab(store, "sm_86", spec)


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
    timed = measured.Measured(registers=1, spill_store=0, spill_load=0,
                              runtime_ns=1, repeats=31, spread_ns=0, **PROV)
    compiled = measured.Measured(registers=1, spill_store=0, spill_load=0, **PROV)
    with pytest.raises(KeyError, match="control"):
        measured.lever_ab({lev: timed, lit: compiled}, "sm_86", spec)
    with pytest.raises(KeyError):
        measured.lever_ab({lit: compiled}, "sm_86", spec)


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", RECYCLE_CELLS, ids=lambda s: s.key())
def test_every_recycle_cell_is_compiled_and_timed_against_a_control(
        spec, arch) -> None:
    """#55's acceptance: two rows per (class, batch count, arch) differing
    only in `levers`, the lever's timed beside its same-session control."""
    (row,) = measured.rows(spec=spec, arch=arch, store=ROWS).values()
    (lit,) = measured.rows(spec=replace(spec, levers=()), arch=arch,
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


# The verdict (#55), on the tracked rows. The definition is measured.py's
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
            c = measured.lever_ab(ROWS, arch, spec)
            got[arch, c.L, c.n_batches] = (c.wide_sign, c.near_sign)
    assert got == LEVER_AB


def test_recycle_stays_opt_in() -> None:
    """The default changes only if recycle WINS at every cell on BOTH arches
    (measured.py). It wins at none, on either: `lower()` stays a no-op unless
    asked, and a Spec without the lever is built exactly as before."""
    verdicts = {k: measured.lever_ab(ROWS, k[0], replace(
        pq_spec(k[1], n_batches=k[2]), levers=(RECYCLE,))).verdict
        for k in LEVER_AB}
    assert "win" not in verdicts.values()
    assert "stays opt-in" in STORE["header"]["levers"]  # the data says so too
    for s in measured.SUBGRID[:5]:
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
