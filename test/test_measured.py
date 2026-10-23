"""The measured store's code (`measured_store`) and the generator's CLI.

Two halves, as the source is split (#94): the STORE -- the record, the file,
the read path, the subgrid and the verdicts -- is `measured_store`, and its
cases call nothing else; the DRIVER (`measured`: parsing what each compiler
and profiler prints, emitting a cell's units, the CLI) is reached only by
the cases that test it.

EVERY CASE HERE IS ON ROWS IT BUILDS ITSELF, or on recorded tool output
spelled inline: nothing reads data/measured/. The checks of the TRACKED
stores -- their keys, provenance and byte-for-byte round trip, that each
row's `source_sha256` is today's emitted text, the declared verdicts as the
rows read them -- and the recompiles against the stored rows are
measure/measure_store.py's (`devtools/measure.sh check`, `recheck`). Every
check that builds, emits or plans a class above `tiers.FAST_L_MAX` is
`slow`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from felizjueves import asym, measured, measured_store
from felizjueves.device import GFX1200, SM_86, MeasuredKey
from felizjueves.emit import __main__ as emit_cli
from felizjueves.emit import cuda, hip
from felizjueves.ir.lower import LEVERS, RECYCLE
from felizjueves.mmd.kernel import build_pq_kernel, build_pq_kernels, k_spec, pq_spec
from felizjueves.mmd.kernel import split as hermite_split
from felizjueves.plan.batching import in_batch, partition
from felizjueves.plan.dag import build
from felizjueves.plan.pipeline import TUNED, plan, tuned_plan
from felizjueves.plan.sieve import NO_SIEVE
from felizjueves.plan.targets import hermite_targets
from felizjueves.spec import Operator, Spec, Target, is_batch_count


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

# Every Spec a row may be keyed on: the one-batch classes, the subgrid, and
# the subgrid's recycle-lever arm (#55), #113's arm, #162's sieve arm, and
# #163's tuned arm (the subgrid less its NOT_MEASURABLE 12-batch cells).
RECYCLE_CELLS = measured_store.lever_cells(RECYCLE)
SIEVE_CELLS = measured_store.SIEVE_CELLS
TUNED_CELLS = measured_store.TUNED_CELLS
ARCHS = {a.name: a for a in (SM_86, GFX1200)}


# --- #52: the declared subgrid ----------------------------------------------


def test_the_subgrid_is_the_declared_l_by_n_batches_product() -> None:
    cells = [(s.total_am, s.n_batches) for s in measured_store.SUBGRID]
    # 12 is #60's: "9 batches suffice" cannot be judged from below 9
    assert cells == [(L, b) for L in (7, 10, 13, 16) for b in (1, 3, 6, 9, 12)]
    assert all(s == pq_spec(s.total_am, n_batches=s.n_batches)
               for s in measured_store.SUBGRID)


@pytest.mark.slow  # the subgrid is L = 7..16 > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("spec", measured_store.SUBGRID, ids=lambda s: s.key())
def test_every_subgrid_cell_is_admitted_by_the_plan_layer(spec) -> None:
    # #64 made partition refuse an empty batch; a refused cell is not a cell.
    assert len(plan(spec).dags) == spec.n_batches
    assert len(build_pq_kernels(spec)) == spec.n_batches


def test_the_dynamic_shared_footprint_is_the_kernels_own_declaration() -> None:
    b = measured.BACKENDS["cuda"]
    (u,) = b.units(pq_spec(4))
    # (2,0|2,0): 10 bra functions p, blockDim.y = 32 lanes, 8-byte doubles
    assert u.jsh_dynamic_bytes == 10 * 32 * 8
    assert [x.jsh_dynamic_bytes for x in b.units(pq_spec(4, n_batches=3))] == [
        9 * 32 * 8] * 3


def test_a_measured_row_refuses_a_negative_or_missing_field() -> None:
    row = measured_store.Measured(registers=40, spill_store=0, spill_load=0,
                                  **PROV)
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
    good = tmp_path / "good.json"  # a store of its own: no tracked row read
    measured_store.save({_key().key(): measured_store.Measured(
        registers=40, **PROV)}, good)
    assert measured.main(["--store", str(good)]) == 0
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


# --- #110: the counter fields, the counter store, and the profilers' output --
#
# Fast tier: no profiler is run here (the device witnesses are in
# measure/measure_toolchain.py), so these hold with ncu and rocprofv3 absent.

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


# ncu --csv --print-units base, as ncu 2025.3.1.0 printed it on this box's RTX
# 3080 for measure_toolchain.py's counter probe: one row per metric per launch,
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


# The probe of measure/measure_toolchain.py (2^24 doubles, 384 MiB streamed) under
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


# --- #55: the recycle lever against the literal kernel ----------------------


def test_the_lever_cells_are_the_whole_subgrid_with_the_lever() -> None:
    # Not one class: every L and every batch count, so a sign flip is seen.
    assert [replace(s, levers=()) for s in RECYCLE_CELLS] == list(
        measured_store.SUBGRID)
    assert all(s.levers == (RECYCLE,) for s in RECYCLE_CELLS)
    # #161's lever, declared with the builder that honours it
    assert all(s.levers == ("tuned",)
               for s in measured_store.lever_cells("tuned"))


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


@pytest.mark.slow  # L = 7 and 10 > FAST_L_MAX (tiers.py)
def test_a_spec_without_recycle_is_built_exactly_as_before() -> None:
    for s in measured_store.SUBGRID[:5]:
        assert measured.cell_kernels(s) == build_pq_kernels(s)
    assert "double slot[" not in cuda.emit(build_pq_kernel(10), SM_86)


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
        assert c.n_batches == sum(c.ks)  # the batches: the `slow` test below


@pytest.mark.slow  # L = 13, 16 > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("cell", asym.ASYM_CELLS, ids=lambda c: c.key())
def test_every_asym_cell_partitions_its_class(cell) -> None:
    assert len(cell.batches) == cell.n_batches
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


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
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
    # its rows sit beside the literal ones and are invisible to their readers
    lit = MeasuredKey(pq_spec(13, n_batches=3), SM_86, "nvcc 13.0")
    row = measured_store.Measured(registers=1, **PROV)
    store = measured_store.merge(measured_store.merge({}, lit, row), k, row)
    assert list(measured_store.rows(spec=pq_spec(13, n_batches=3),
                                    store=store)) == [lit.key()]


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_an_asym_cell_builds_its_own_batch_kernels() -> None:
    assert [x.name for x in measured.cell_kernels(asym.AsymCell(13, 4))] == [
        "pq_L13_p7_q6_x0_6", "pq_L13_p7_q6_x7_13", "pq_L13_p7_q6_y0_13",
        "pq_L13_p7_q6_z0_13"]


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


# --- #147: the sort key, product against one global max|D| -----------------
#
# The rule is measured_store's docstring, THE SORT KEY, declared before any
# 'sort-ab' row existed. Synthetic rows first; the tracked rows after.

SK = measured_store
SK_CELL = SK.SORT_KEY_CELLS[0]


def _sort_pair(cell=SK_CELL, *, p=(100_000, 2_000, 50), g=(100_000, 2_000, 0),
               session=(7, 7), pairs=(1000, 1000), arch="sm_86",
               toolchain="nvcc 13.0") -> dict:
    """The two arms of `cell` as build rows: (median, IQR, n_skipped) each."""
    fixture, tau, n = cell
    out = {}
    for ds, (med, iqr, sk), s, npairs in zip(
            (True, False), (p, g), session, pairs, strict=True):
        r = _build_row(fixture=fixture, tau=tau, n_batches=n, density_sort=ds,
                       arm=SK.SORT_AB_ARM, runtime_ns=med, spread_ns=iqr,
                       arch=arch, toolchain=toolchain,
                       raw={SK.SESSION_FIELD: s, "build:n_pairs": npairs,
                            "build:n_skipped": sk})
        out[r.key()] = r
    return out


@pytest.mark.parametrize(("g", "clock", "verdict"), [
    ((104_001, 2_000, 0), SK.SLOWER, SK.WORSE),       # slower by > max IQR
    ((95_999, 2_000, 0), SK.FASTER, SK.BETTER),       # faster by > max IQR
    ((104_000, 2_000, 0), SK.NEUTRAL, SK.UNRESOLVED),  # == max IQR: neutral
    ((105_000, 4_000, 0), SK.NEUTRAL, SK.UNRESOLVED),  # the LARGER IQR
    ((100_000, 2_000, 50), SK.NEUTRAL, SK.SAME),      # same skips, same time
    ((104_001, 2_000, 50), SK.SLOWER, SK.WORSE),      # same skips, slower
])
def test_the_sort_key_verdict_follows_the_declared_rule(g, clock,
                                                        verdict) -> None:
    ab = SK.sort_key_ab(_sort_pair(p=(102_000, 2_000, 50), g=g), "sm_86",
                        SK_CELL)
    # p median 102_000: g - p is +2_001, -6_001, +2_000, +3_000 (under the
    # larger IQR, 4_000), -2_000, +2_001
    assert ab.clock_sign == clock and ab.verdict == verdict


def test_the_sort_key_arch_and_claim_verdicts() -> None:
    def store(per_cell, arch="sm_86"):
        out = {}
        for cell, g in zip(SK.SORT_KEY_CELLS, per_cell, strict=True):
            out.update(_sort_pair(cell, g=g, arch=arch))
        return out

    slow, fast, tie = (110_000, 2_000, 0), (90_000, 2_000, 0), \
        (100_000, 2_000, 0)
    assert SK.sort_key_arch_verdict(store([slow] * 4), "sm_86") == SK.WORSE
    assert SK.sort_key_arch_verdict(store([slow, slow, tie, slow]),
                                    "sm_86") == SK.NOT_SHOWN
    assert SK.sort_key_arch_verdict(store([slow, fast, tie, slow]),
                                    "sm_86") == SK.CONTRADICTED
    both = {**store([slow] * 4), **store([slow] * 4, arch="gfx1200")}
    assert SK.sort_key_claim_holds(both, ("sm_86", "gfx1200"))
    one = {**store([slow] * 4), **store([slow, tie, slow, slow],
                                        arch="gfx1200")}
    assert not SK.sort_key_claim_holds(one, ("sm_86", "gfx1200"))


def test_a_sort_key_pair_that_is_not_one_experiment_is_refused() -> None:
    with pytest.raises(KeyError, match="different sessions"):
        SK.sort_key_ab(_sort_pair(session=(7, 8)), "sm_86", SK_CELL)
    with pytest.raises(KeyError, match="more than the sort"):
        SK.sort_key_ab(_sort_pair(pairs=(1000, 999)), "sm_86", SK_CELL)
    lone = dict(list(_sort_pair().items())[:1])
    with pytest.raises(KeyError, match="0 rows"):
        SK.sort_key_ab(lone, "sm_86", SK_CELL)
    with pytest.raises(KeyError, match="0 rows"):
        SK.sort_key_ab(_sort_pair(), "gfx1200", SK_CELL)
    # the global key skipping MORE than the product is impossible: refused
    with pytest.raises(ValueError, match="impossible"):
        SK.sort_key_ab(_sort_pair(g=(1, 1, 51)), "sm_86", SK_CELL)


# --- #159: the A/B arms admit a plan lever ------------------------------------

NO_SIEVE_CELLS = measured_store.lever_cells(NO_SIEVE)


def test_the_declared_arms_and_their_kinds() -> None:
    assert dict(measured_store.ARMS) == {RECYCLE: measured_store.LOWERING,
                                         NO_SIEVE: measured_store.PLAN,
                                         TUNED: measured_store.PLAN}
    # every lowering lever is an arm, and the store's spelling of the plan
    # lever is the plan layer's (the store may not import plan/)
    assert {lv for lv, k in measured_store.ARMS.items()
            if k == measured_store.LOWERING} == set(LEVERS)
    assert measured_store.NO_SIEVE == NO_SIEVE
    with pytest.raises(TypeError):
        measured_store.ARMS["tuned"] = measured_store.PLAN  # read-only


def _plan_arm(spec=None, **kw) -> tuple[MeasuredKey, measured_store.Measured]:
    spec = NO_SIEVE_CELLS[0] if spec is None else spec
    return (MeasuredKey(spec, SM_86, "nvcc 13.0"),
            measured_store.Measured(registers=200, spill_store=0, spill_load=0,
                                    **PROV, **kw))


def test_a_plan_arm_round_trips_byte_for_byte_with_its_kind(
        tmp_path: Path) -> None:
    k, row = _plan_arm()
    assert not row.arm_kind
    store = measured_store.merge({}, k, row)  # merge stamps the kind
    assert store[k.key()].arm_kind == measured_store.PLAN
    p = tmp_path / "m.json"
    measured_store.save(store, p)
    doc = json.loads(p.read_text())
    assert doc["rows"][k.key()]["arm_kind"] == "plan"
    back = measured_store.load(p)
    assert back == store
    q = tmp_path / "again.json"
    measured_store.save(back, q)
    assert q.read_bytes() == p.read_bytes()
    # a recycle arm merged today is stamped too, as a lowering arm
    rk = MeasuredKey(RECYCLE_CELLS[0], SM_86, "nvcc 13.0")
    assert measured_store.merge({}, rk, row)[rk.key()].arm_kind == "lowering"
    # and a literal row carries none
    lk = MeasuredKey(measured_store.SUBGRID[0], SM_86, "nvcc 13.0")
    assert measured_store.merge({}, lk, row)[lk.key()].arm_kind == ""


def _write(tmp_path: Path, key: str, rec: dict) -> Path:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": measured_store.SCHEMA,
                             "about": measured_store.ABOUT,
                             "rows": {key: rec}}))
    return p


def test_the_loader_refuses_an_arm_kind_that_disagrees_with_its_key(
        tmp_path: Path) -> None:
    k, row = _plan_arm()
    rec = measured_store.as_record(row)  # no kind
    with pytest.raises(ValueError, match="plan arm and records no arm_kind"):
        measured_store.load(_write(tmp_path, k.key(), rec))
    with pytest.raises(ValueError, match="arm_kind 'lowering'"):
        measured_store.load(_write(tmp_path, k.key(),
                                   {**rec, "arm_kind": "lowering"}))
    lit = MeasuredKey(measured_store.SUBGRID[0], SM_86, "nvcc 13.0").key()
    with pytest.raises(ValueError, match="arm_kind 'plan'"):
        measured_store.load(_write(tmp_path, lit, {**rec, "arm_kind": "plan"}))
    with pytest.raises(ValueError, match="arm_kind"):
        measured_store.load(_write(tmp_path, k.key(),
                                   {**rec, "arm_kind": "tuning"}))
    with pytest.raises(ValueError, match="arm_kind"):
        measured_store.merge({}, k, replace(row, arm_kind="lowering"))


def test_a_spec_carrying_two_arms_is_refused_naming_both(
        tmp_path: Path) -> None:
    two = replace(measured_store.SUBGRID[0],
                  levers=tuple(sorted((NO_SIEVE, RECYCLE))))
    _, row = _plan_arm()
    for call in (lambda: measured_store.arm_of(two),
                 lambda: measured_store.lever_ab({}, "sm_86", two),
                 lambda: measured_store.merge({}, MeasuredKey(
                     two, SM_86, "nvcc 13.0"), row)):
        with pytest.raises(ValueError) as e:
            call()
        assert NO_SIEVE in str(e.value) and RECYCLE in str(e.value)
    key = MeasuredKey(two, SM_86, "nvcc 13.0").key()
    rec = {**measured_store.as_record(row), "arm_kind": "plan"}
    with pytest.raises(ValueError, match=r"no-sieve, recycle"):
        measured_store.load(_write(tmp_path, key, rec))


@pytest.mark.parametrize("lever", ("no_sieve", "literal", "asym", ""))
def test_a_lever_outside_the_declared_set_is_still_refused(lever) -> None:
    with pytest.raises(ValueError, match="not a declared arm lever"):
        measured_store.lever_cells(lever)
    if lever:  # Spec itself refuses an empty lever
        spec = replace(measured_store.SUBGRID[0], levers=(lever,))
        with pytest.raises(ValueError, match="not a declared arm lever"):
            measured_store.lever_ab({}, "sm_86", spec)
        with pytest.raises(ValueError, match="not a declared arm lever"):
            measured_store.merge({}, MeasuredKey(spec, SM_86, "nvcc 13.0"),
                                 _plan_arm()[1])


def test_a_literal_cell_is_a_control_not_an_arm() -> None:
    assert measured_store.arm_of(measured_store.SUBGRID[0]) is None
    with pytest.raises(ValueError, match="control"):
        measured_store.lever_ab({}, "sm_86", measured_store.SUBGRID[0])


def test_the_no_sieve_cells_are_the_whole_subgrid_unsieved() -> None:
    assert [replace(s, levers=()) for s in NO_SIEVE_CELLS] == list(
        measured_store.SUBGRID)
    assert all(s.levers == (NO_SIEVE,) for s in NO_SIEVE_CELLS)
    assert {measured_store.arm_kind(s) for s in NO_SIEVE_CELLS} == {"plan"}


def _dags(dags) -> tuple:
    """BatchDag compares by identity (eq=False): compare what it holds."""
    return tuple((d.batch, d.targets, dict(d.parents)) for d in dags)


@pytest.mark.slow  # L = 7 > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("spec", NO_SIEVE_CELLS[:5], ids=lambda s: s.key())
def test_the_driver_passes_a_plan_lever_to_the_builder(spec) -> None:
    """The driver strips only lower.LEVERS: a no-sieve cell's kernels are
    the builder's for the unsieved plan, asserted against `plan(spec)`
    directly. No compiler."""
    got = measured.cell_kernels(spec)
    assert got == build_pq_kernels(spec)
    literal = replace(spec, levers=())
    p = plan(spec)
    assert _dags(p.dags) == _dags(build(literal))  # MILD's DAGs, unsieved
    assert len(got) == len(p.dags) == spec.n_batches
    if _dags(p.dags) != _dags(plan(literal).dags):  # the sieve acted here...
        assert got != measured.cell_kernels(literal)  # ...so the kernel moves


@pytest.mark.slow  # L = 7 > FAST_L_MAX (tiers.py)
def test_the_sieve_changes_the_kernel_somewhere_on_the_subgrid() -> None:
    """A plan arm is not a lowering arm: its kernel differs from the
    control's (#162's premise), at least at one subgrid cell."""
    assert any(measured.cell_kernels(s)
               != measured.cell_kernels(replace(s, levers=()))
               for s in NO_SIEVE_CELLS[:5])


def _sieve_store(signs: dict[Spec, tuple[int, int]]) -> dict:
    """Synthetic sm_86 rows for no-sieve cells: (wide, near) lever medians
    against a control of (1000, 10) wide and (500, 5) near."""
    store = {}
    for spec, (w, n) in signs.items():
        lit = MeasuredKey(replace(spec, levers=()), SM_86, "nvcc 13.0")
        lev = MeasuredKey(spec, SM_86, "nvcc 13.0")
        store = measured_store.merge(store, lit, measured_store.Measured(
            registers=255, spill_store=0, spill_load=0, **PROV))
        store = measured_store.merge(store, lev, measured_store.Measured(
            registers=255, spill_store=0, spill_load=0, runtime_ns=w,
            repeats=31, spread_ns=10, **PROV,
            raw={"wallclock:control:median_ns": 1000,
                 "wallclock:control:q1_ns": 0, "wallclock:control:q3_ns": 10,
                 "wallclock:control:near:median_ns": 500,
                 "wallclock:control:near:q1_ns": 0,
                 "wallclock:control:near:q3_ns": 5,
                 "wallclock:near:median_ns": n,
                 "wallclock:near:q1_ns": 0, "wallclock:near:q3_ns": 5,
                 **UNKNOWN_CARD}))
    return store


def test_the_sieve_verdict_is_declared_before_its_rows() -> None:
    doc = measured_store.__doc__
    assert "THE SIEVE ARM" in doc
    assert "DECLARED BEFORE ANY `no-sieve` ROW EXISTED" in doc
    cells = NO_SIEVE_CELLS[:3]
    same, win, loss = (1000, 500), (900, 450), (1100, 550)
    mixed = (900, 550)

    def verdict(*arms) -> str:
        return measured_store.sieve_verdict(
            _sieve_store(dict(zip(cells, arms, strict=True))), "sm_86", cells)

    assert verdict(loss, same, same) == measured_store.SUPPORTED
    assert verdict(loss, loss, loss) == measured_store.SUPPORTED
    assert verdict(win, same, same) == measured_store.CONTRADICTED
    assert verdict(same, same, same) == measured_store.NOT_SHOWN
    assert verdict(win, loss, same) == measured_store.NOT_SHOWN
    assert verdict(mixed, loss, loss) == measured_store.NOT_SHOWN
    got = measured_store.sieve_ab(_sieve_store(dict(zip(
        cells, (loss, win, mixed), strict=True))), "sm_86", cells)
    assert [c.verdict for c in got.values()] == ["loss", "win", "mixed"]
    assert {c.kind for c in got.values()} == {"plan"}
    # the default moves only on a WIN at every cell on every arch asked
    allwin = _sieve_store(dict.fromkeys(cells, win))
    assert measured_store.sieve_may_leave_the_default(allwin, cells, ("sm_86",))
    assert not measured_store.sieve_may_leave_the_default(
        _sieve_store(dict(zip(cells, (win, win, same), strict=True))), cells,
        ("sm_86",))
    with pytest.raises(KeyError):  # gfx1200 not recorded: not judged
        measured_store.sieve_may_leave_the_default(allwin, cells)
    with pytest.raises(ValueError, match="not a no-sieve arm"):
        measured_store.sieve_ab(allwin, "sm_86", RECYCLE_CELLS[:1])
    with pytest.raises(KeyError):  # a cell not recorded is not judged
        measured_store.sieve_verdict(_sieve_store({cells[0]: win}), "sm_86")


@pytest.mark.slow  # L = 11 > FAST_L_MAX (tiers.py)
def test_the_l11_cell_is_one_plan_either_way_round() -> None:
    (l11,) = SIEVE_CELLS[20:]
    # [5|6] and [6|5] are one plan: the plan layer counts by L alone
    assert plan(l11).cost == plan(replace(l11, am=(5, 0, 6, 0))).cost


def test_the_sieve_verdict_reads_an_added_cell_by_the_same_rule() -> None:
    """THE SIEVE ARM: the rule is per cell, so L = 11 is judged unchanged."""
    l11 = SIEVE_CELLS[-1]
    same, win, loss = (1000, 500), (900, 450), (1100, 550)
    cells = (NO_SIEVE_CELLS[0], l11)
    store = _sieve_store({NO_SIEVE_CELLS[0]: same, l11: loss})
    got = measured_store.sieve_ab(store, "sm_86", cells)
    assert got[l11].verdict == "loss" and got[l11].L == 11
    assert measured_store.sieve_verdict(store, "sm_86", cells) == "supported"
    assert measured_store.sieve_verdict(
        _sieve_store({NO_SIEVE_CELLS[0]: same, l11: win}), "sm_86",
        cells) == "contradicted"
    with pytest.raises(KeyError):  # the default set needs every cell
        measured_store.sieve_verdict(store, "sm_86")


# --- #163: the tuned arm on both cards ----------------------------------------

TUNED_LEVER_CELLS = measured_store.lever_cells(TUNED)
NOT_MEASURABLE = measured_store.NOT_MEASURABLE[TUNED]


def test_the_tuned_spelling_is_the_plan_layers() -> None:
    assert measured_store.TUNED == TUNED
    assert measured_store.ARMS[TUNED] == measured_store.PLAN


@pytest.mark.slow  # the subgrid, L = 7..16 > FAST_L_MAX (tiers.py)
def test_the_not_measurable_cells_are_exactly_the_ones_the_lever_refuses(
        ) -> None:
    """Every `lever_cells(TUNED)` cell is planned: the ones `tuned_plan`
    refuses for want of a recorded choice are exactly NOT_MEASURABLE, each
    with that reason, and every other one builds (no cell silently lost)."""
    refused = {}
    for spec in TUNED_LEVER_CELLS:
        try:
            tuned_plan(spec)
        except ValueError as e:
            refused[spec] = str(e)
    assert set(refused) == set(NOT_MEASURABLE)
    assert {(s.total_am, s.n_batches) for s in NOT_MEASURABLE} == {
        (L, 12) for L in measured_store.SUBGRID_LS}
    for spec, err in refused.items():
        assert "no recorded choice" in err, err
        assert NOT_MEASURABLE[spec].startswith("no recorded choice")
        assert "never a fall back to the literal plan" in NOT_MEASURABLE[spec]
    with pytest.raises(ValueError, match="no recorded choice"):
        build_pq_kernels(next(iter(NOT_MEASURABLE)))


def test_the_tuned_cells_are_the_subgrid_less_the_not_measurable() -> None:
    assert TUNED_CELLS == measured_store.ab_cells(TUNED)
    assert TUNED_CELLS == tuple(s for s in TUNED_LEVER_CELLS
                                if s not in NOT_MEASURABLE)
    assert len(TUNED_CELLS) == 16 and len(NOT_MEASURABLE) == 4
    assert all(s.levers == (TUNED,) for s in TUNED_CELLS)
    assert {measured_store.arm_kind(s) for s in TUNED_CELLS} == {"plan"}
    # no other arm loses a cell to NOT_MEASURABLE (#55's and #162's unchanged)
    assert set(measured_store.NOT_MEASURABLE) == {TUNED}
    assert measured_store.tuned_ab.__defaults__ == (None,)


def test_the_tuned_verdict_is_declared_before_its_rows() -> None:
    doc = measured_store.__doc__
    assert "THE TUNED ARM" in doc
    assert "DECLARED BEFORE ANY `tuned` ROW EXISTED" in doc
    cells = TUNED_CELLS[:3]
    same, win, loss = (1000, 500), (900, 450), (1100, 550)
    mixed = (900, 550)

    def verdict(*arms) -> str:
        return measured_store.tuned_verdict(
            _sieve_store(dict(zip(cells, arms, strict=True))), "sm_86", cells)

    # the sieve's sign reversed: the search is SUPPORTED by a tuned WIN
    assert verdict(win, same, same) == measured_store.SUPPORTED
    assert verdict(win, win, win) == measured_store.SUPPORTED
    assert verdict(loss, same, same) == measured_store.CONTRADICTED
    assert verdict(same, same, same) == measured_store.NOT_SHOWN
    assert verdict(win, loss, same) == measured_store.NOT_SHOWN
    assert verdict(mixed, win, win) == measured_store.NOT_SHOWN
    got = measured_store.tuned_ab(_sieve_store(dict(zip(
        cells, (loss, win, mixed), strict=True))), "sm_86", cells)
    assert [c.verdict for c in got.values()] == ["loss", "win", "mixed"]
    assert {c.kind for c in got.values()} == {"plan"}
    assert {c.lever for c in got.values()} == {TUNED}
    allwin = _sieve_store(dict.fromkeys(cells, win))
    assert measured_store.tuned_may_become_the_default(allwin, cells,
                                                       ("sm_86",))
    assert not measured_store.tuned_may_become_the_default(
        _sieve_store(dict(zip(cells, (win, win, same), strict=True))), cells,
        ("sm_86",))
    with pytest.raises(KeyError):  # gfx1200 not recorded: not judged
        measured_store.tuned_may_become_the_default(allwin, cells)
    with pytest.raises(ValueError, match="not a tuned arm"):
        measured_store.tuned_ab(allwin, "sm_86", NO_SIEVE_CELLS[:1])
    with pytest.raises(ValueError, match="not measurable -- no recorded"):
        measured_store.tuned_ab(allwin, "sm_86", tuple(NOT_MEASURABLE)[:1])
    with pytest.raises(KeyError):  # a cell not recorded is not judged
        measured_store.tuned_verdict(_sieve_store({cells[0]: win}), "sm_86")


def test_the_remeasure_cli_names_the_unmeasurable_cells(capsys,
                                                        monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(measured, "remeasure",
                        lambda names, cells, j, store: seen.append(list(cells)))
    assert measured.main(["--remeasure", "--subgrid", "--lever", TUNED]) == 0
    assert seen == [list(TUNED_CELLS)]
    err = capsys.readouterr().err
    for spec in NOT_MEASURABLE:
        assert f"{spec.key()} NOT MEASURABLE -- no recorded choice" in err


# --- #201: the store admits a K cell ------------------------------------------
# Pure Spec keys and synthetic rows: no K kernel is built, planned or
# interpreted here, and no tracked file is written.


def _k_row(**kw) -> measured_store.Measured:
    return measured_store.Measured(registers=128, spill_store=0, spill_load=0,
                                   **PROV, **kw)


def test_k_subgrid_is_the_j_subgrid_twin_cell_for_cell() -> None:
    k_cells, j_cells = measured_store.K_SUBGRID, measured_store.SUBGRID
    assert len(k_cells) == len(j_cells) == 20
    assert k_cells == tuple(k_spec(L, n_batches=b)
                            for L in measured_store.SUBGRID_LS
                            for b in measured_store.SUBGRID_BATCHES)
    for k, j in zip(k_cells, j_cells, strict=True):
        assert k.target is Target.K and j.target is Target.J
        # the same L, Hermite split and batch count: the same plan and row
        assert (k.total_am, k.n_batches) == (j.total_am, j.n_batches)
        assert hermite_split(k) == hermite_split(j)
        assert k.key() != j.key()
    assert dict(measured_store.SUBGRIDS) == {Target.J: j_cells, Target.K: k_cells}
    with pytest.raises(TypeError):
        measured_store.SUBGRIDS[Target.DQ] = ()  # read-only


def test_a_k_cell_round_trips_through_the_counter_store(tmp_path: Path) -> None:
    rows = {kk: replace(r, target="k") for kk, r in
            _cell_rows(measured_store.K_SUBGRID[1], flops=(1, 2, 3)).items()}
    p = tmp_path / "roofline.json"
    measured_store.save_counters(rows, p)
    assert measured_store.load_counters(p) == rows
    # a K cell's counter rows without their target are refused
    with pytest.raises(ValueError, match="records no target"):
        measured_store.save_counters(
            {kk: replace(r, target="") for kk, r in rows.items()}, p)


def test_a_k_row_under_a_j_key_and_the_reverse_are_refused(
        tmp_path: Path) -> None:
    k = MeasuredKey(measured_store.K_SUBGRID[0], SM_86, "nvcc 13.0")
    j = MeasuredKey(measured_store.SUBGRID[0], SM_86, "nvcc 13.0")
    rec = measured_store.as_record(_k_row())  # no target
    # a K row under a J key
    with pytest.raises(ValueError, match="records target 'k'.*target is 'j'"):
        measured_store.load(_write(tmp_path, j.key(), {**rec, "target": "k"}))
    with pytest.raises(ValueError, match="records target 'k'"):
        measured_store.merge({}, j, _k_row(target="k"))
    # a J row under a K key: one that says so, and one that records nothing
    with pytest.raises(ValueError, match="records target 'j'.*target is 'k'"):
        measured_store.load(_write(tmp_path, k.key(), {**rec, "target": "j"}))
    with pytest.raises(ValueError, match="records target 'j'"):
        measured_store.merge({}, k, _k_row(target="j"))
    with pytest.raises(ValueError, match="'k' row and records no target"):
        measured_store.load(_write(tmp_path, k.key(), rec))
    # a target that is no Target at all
    with pytest.raises(ValueError, match="target = 'x'"):
        _k_row(target="x")
    with pytest.raises(ValueError, match="names no target"):
        measured_store.target_of(
            "coulomb/d0/(1,0|1,0)/b1/x/literal@sm_86/nvcc 13.0")


# --- #278: the store admits an F cell ----------------------------------------

_F_SPEC = Spec(Operator.COULOMB, 0, (2, 1, 1, 0), Target.F)
_GATE = dict.fromkeys(measured_store.GPU_UTIL_FIELDS, 3)


def _f_timed(**raw) -> measured_store.Measured:
    return measured_store.Measured(
        registers=96, spill_store=0, spill_load=0, runtime_ns=1234.5,
        repeats=31, spread_ns=10.0, raw={**_GATE, **raw}, **PROV)


def test_a_compile_only_f_row_needs_no_gate(tmp_path: Path) -> None:
    k = MeasuredKey(_F_SPEC, GFX1200, "hipcc / ROCm 7.2.4")
    row = measured_store.Measured(
        registers=64, spill_store=2, spill_load=2,
        spill_unit="registers (VGPR + SGPR spill counts)",
        **{**PROV, "arch": "gfx1200", "toolchain": k.toolchain})
    store = measured_store.merge({}, k, row)
    p = tmp_path / "m.json"
    measured_store.save(store, p)
    assert measured_store.load(p) == store


@pytest.mark.parametrize("field", measured_store.GPU_UTIL_FIELDS)
def test_a_timed_f_row_without_its_gate_readings_is_refused(
        tmp_path: Path, field: str) -> None:
    k = MeasuredKey(_F_SPEC, SM_86, "nvcc 13.0")
    absent = replace(_f_timed(), raw={f: v for f, v in _GATE.items()
                                      if f != field})
    null = _f_timed(**{field: None})
    for row in (absent, null):
        with pytest.raises(ValueError, match=f"timed F row.*{field}"):
            measured_store.merge({}, k, row)
        rec = {**measured_store.as_record(row), "target": "f"}
        with pytest.raises(ValueError, match=field):
            measured_store.load(_write(tmp_path, k.key(), rec))
    # a J row timed before #93 may still record null: unknown, not idle
    j = MeasuredKey(measured_store.SUBGRID[0], SM_86, "nvcc 13.0")
    rec = measured_store.as_record(null)
    assert measured_store.load(_write(tmp_path, j.key(), rec))[j.key()] == null


def test_an_f_row_records_its_target_and_names_a_legal_f_spec(
        tmp_path: Path) -> None:
    k = MeasuredKey(_F_SPEC, SM_86, "nvcc 13.0")
    rec = measured_store.as_record(_f_timed())  # no target
    with pytest.raises(ValueError, match="'f' row and records no target"):
        measured_store.load(_write(tmp_path, k.key(), rec))
    with pytest.raises(ValueError, match="records target 'k'.*target is 'f'"):
        measured_store.load(_write(tmp_path, k.key(), {**rec, "target": "k"}))
    with pytest.raises(ValueError, match="records target 'f'"):
        measured_store.merge({}, MeasuredKey(measured_store.SUBGRID[0], SM_86,
                                             "nvcc 13.0"),
                             replace(_f_timed(), target="f"))
    rec = {**rec, "target": "f"}
    base = "coulomb/d0/(2,1|1,0)/b{n}/f/{lv}@sm_86/nvcc 13.0"
    for key, match in (
            (base.format(n=3, lv="literal"), "names no legal F spec"),
            (base.format(n=1, lv="recycle"), "no lever is declared"),
            (base.format(n=1, lv="no-sieve"), "plan arm"),
            ("coulomb/d0/(2,1)/b1/f/literal@sm_86/nvcc 13.0", "whose key"),
            ("coulomb/d0/(02,1|1,0)/b1/f/literal@sm_86/nvcc 13.0",
             "does not spell")):
        with pytest.raises(ValueError, match=match):
            measured_store.load(_write(tmp_path, key, rec))
    # a plan arm is refused for an F cell even when it records its kind
    with pytest.raises(ValueError, match="no lever is declared"):
        measured_store.load(_write(tmp_path, base.format(n=1, lv="no-sieve"),
                                   {**rec, "arm_kind": measured_store.PLAN}))
    # and a legal one loads
    assert measured_store.load(_write(tmp_path, k.key(), rec))[k.key()] == \
        replace(_f_timed(), target="f")


# --- #205: the K rows, against J's at the same L ----------------------------


def test_the_k_rule_was_declared_in_the_store() -> None:
    doc = measured_store.__doc__
    assert "THE K ROWS (#205)" in doc
    assert "DECLARED BEFORE ANY SPLIT K ROW EXISTED" in doc
    for part in ("THE COMPILE SIDE", "THE CLOCK", "SPILL (`k_spill_curve`)",
                 "WALL-CLOCK (`k_gain`)", "A CELL THAT CANNOT BE MEASURED"):
        assert part in doc, part


def test_a_k_cell_is_built_by_kernel_k_and_takes_no_lever() -> None:
    from felizjueves.kernel_k import build_k_kernels
    s = k_spec(2, n_batches=3)
    assert measured.cell_kernels(s) == build_k_kernels(s)
    assert measured.cell_kernels(k_spec(1)) == build_k_kernels(k_spec(1))
    with pytest.raises(ValueError, match="no lever is declared for a K cell"):
        measured.cell_kernels(replace(k_spec(1), levers=(RECYCLE,)))
    # the unit's dynamic shared bytes are the K kernel's Ksh, per lane
    from felizjueves.kernel_k import shared_bytes
    (u,) = measured.BACKENDS["cuda"].units(k_spec(2))
    assert u.jsh_dynamic_bytes == shared_bytes(k_spec(2).am, 32)


def test_every_k_cell_is_the_twin_of_a_j_subgrid_cell() -> None:
    for k in measured_store.K_SUBGRID:
        j = measured_store.j_twin(k)
        assert j in measured_store.SUBGRID
        assert (j.total_am, j.n_batches) == (k.total_am, k.n_batches)
    with pytest.raises(ValueError, match="not a K cell"):
        measured_store.j_twin(measured_store.SUBGRID[0])


def test_the_remeasure_cli_reaches_the_k_subgrid(monkeypatch, capsys) -> None:
    seen = []
    monkeypatch.setattr(measured, "remeasure",
                        lambda names, cells, j, store, **kw: seen.append(
                            (list(cells), kw)))
    assert measured.main(["--remeasure", "--subgrid", "--target", "k"]) == 0
    assert seen[-1] == (list(measured_store.K_SUBGRID), {})
    assert measured.main(["--remeasure", "--subgrid", "--target", "k",
                          "--n-batches", "1", "--compile-timeout", "60"]) == 0
    assert seen[-1] == ([s for s in measured_store.K_SUBGRID
                         if s.n_batches == 1], {"timeout": 60.0})
    for bad in (["--remeasure", "--target", "k"],
                ["--remeasure", "--subgrid", "--target", "k", "--lever",
                 RECYCLE]):
        assert measured.main(bad) == 2


def test_a_compile_past_its_budget_gets_no_row_and_the_others_do(
        tmp_path: Path, monkeypatch) -> None:
    calls = []

    split = pq_spec(4, n_batches=3)
    units = measured.BACKENDS["cuda"].units(split)
    largest = max(units, key=lambda u: len(u.text)).name

    def fake(backend, unit, *, compiler, workdir, timeout):
        calls.append(unit.name)
        if unit.name.startswith("pq_L1") or unit.name == largest:
            raise measured.CompileTimeout(f"did not finish {unit.name}")
        return ({"registers": 8, "stack_frame_bytes": 0,
                 "spill_store_bytes": 0, "spill_load_bytes": 0}, 0.1, "nvcc x")

    monkeypatch.setattr(measured, "compile_unit", fake)
    monkeypatch.setattr(measured.Backend, "compiler", lambda self: "nvcc")
    monkeypatch.setattr(measured.Backend, "toolchain",
                        lambda self, c: "nvcc 0.0 (synthetic)")
    monkeypatch.setattr(measured.Backend, "gpu", lambda self: "g")
    p = tmp_path / "k.json"
    measured_store.save({}, p)
    with pytest.raises(RuntimeError, match="2 cells got no row") as e:
        measured.remeasure(["cuda"], [0, 1, 2, split], 1, p, timeout=5)
    assert f"did not finish {largest}" in str(e.value)
    got = measured_store.load(p)
    assert sorted(measured_store.split_key(k)[0] for k in got) == sorted(
        pq_spec(L).key() for L in (0, 2))
    # the split cell's LARGEST unit went first, every cell's before any
    # cell's second, and its failure cancelled the cell's other two units
    assert calls == ["pq_L0_p0_q0", "pq_L1_p1_q0", "pq_L2_p1_q1", largest]


def _kc(spec: Spec, spill: int, *, worst: int | None = None,
        arch: str = "sm_86") -> tuple[str, measured_store.Measured]:
    """A synthetic compiled row of `spec` whose cell spills `spill` bytes."""
    tc = "nvcc 13.0" if arch == "sm_86" else "hipcc / ROCm 7.2.4"
    key = MeasuredKey(spec, ARCHS[arch], tc).key()
    raw = ({"spill_store_bytes": spill // 2, "spill_load_bytes": spill // 2}
           if spec.n_batches == 1 else
           {f"k{i}:spill_store_bytes": v for i, v in enumerate(
               [worst if worst is not None else spill // 2, 0])}
           | {f"k{i}:spill_load_bytes": v for i, v in enumerate(
               [0, spill - (worst if worst is not None else spill // 2)])})
    return key, measured_store.Measured(
        registers=255, spill_store=spill // 2, spill_load=spill - spill // 2,
        toolchain=tc, arch=arch, host="h", gpu="g", date="2026-10-13",
        kernel="k0" if spec.n_batches == 1 else "k0,k1",
        target="k" if spec.target is Target.K else "", raw=raw)


def _curve(L: int, k: list[int], j: list[int],
           refused=None) -> measured_store.KSpillCurve:
    store = {}
    for s, sk, sj in zip([c for c in measured_store.K_SUBGRID
                          if c.total_am == L], k, j, strict=True):
        if sk is not None:
            store.update([_kc(s, sk)])
        store.update([_kc(measured_store.j_twin(s), sj)])
    return measured_store.k_spill_curve(store, "sm_86", L, refused=refused or {})


@pytest.mark.parametrize("k, j, verdict", [
    # K falls to a smaller fraction of its one-batch spill at every b
    pytest.param([100, 50, 20, 10, 5], [100, 80, 60, 40, 20], "faster",
                 id="faster"),
    pytest.param([100, 90, 80, 70, 60], [100, 50, 20, 10, 5], "slower",
                 id="slower"),
    pytest.param([100, 50, 80, 70, 60], [100, 60, 60, 40, 20], "mixed",
                 id="mixed"),
    # equal fractions are SAME, which is neither FASTER nor SLOWER
    pytest.param([100, 50, 20, 10, 5], [10, 5, 2, 1, 1], "mixed",
                 id="a-tie-is-not-faster"),
    pytest.param([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], "no spill", id="none"),
    # J never spills (flat at 1); K spilling at one batch and then less falls
    pytest.param([100, 50, 20, 10, 5], [0, 0, 0, 0, 0], "faster",
                 id="j-flat"),
    # K spilling only once split has RISEN (inf), slower than a flat J
    pytest.param([0, 10, 0, 0, 0], [0, 0, 0, 0, 0], "mixed", id="k-rises"),
])
def test_the_k_spill_curve_rule_on_synthetic_rows(k, j, verdict) -> None:
    c = _curve(13, k, j)
    assert c.batches == (1, 3, 6, 9, 12)
    assert (c.k, c.j) == (tuple(k), tuple(j))
    assert c.verdict == verdict


def test_a_refused_k_cell_is_named_and_not_judged() -> None:
    cells = [c for c in measured_store.K_SUBGRID if c.total_am == 16]
    c = _curve(16, [100, None, 20, 10, 5], [100, 80, 60, 40, 20],
               refused={cells[1]: "a mechanism"})
    assert (c.batches, c.refused, c.verdict) == ((1, 6, 9, 12), (3,), "faster")
    c = _curve(16, [None, 50, 20, 10, 5], [100, 80, 60, 40, 20],
               refused={cells[0]: "a mechanism"})
    assert (c.batches, c.verdict) == ((3, 6, 9, 12), "not judged")
    with pytest.raises(KeyError, match="0 rows"):
        _curve(16, [None, 50, 20, 10, 5], [100, 80, 60, 40, 20])


def test_the_worst_batch_kernel_is_read_off_raw_and_reported() -> None:
    s = [c for c in measured_store.K_SUBGRID if c.total_am == 7]
    _, row = _kc(s[1], 100, worst=70)
    assert measured_store.kernel_spills(row) == (70, 30)
    assert sum(measured_store.kernel_spills(row)) == 100
    # hipcc: one count per kernel, carried by both fields
    hip_row = measured_store.Measured(
        registers=192, spill_store=5, spill_load=5, toolchain="hipcc / ROCm 7.2.4",
        arch="gfx1200", host="h", gpu="g", date="2026-10-13", kernel="k0",
        target="k", raw={"VGPRs Spill": 4, "SGPRs Spill": 1})
    assert measured_store.kernel_spills(hip_row) == (10,)


def test_the_k_gain_is_read_against_the_same_session_j_twin() -> None:
    cells = [c for c in measured_store.K_SUBGRID if c.total_am == 10]
    store = {}
    for s, (kw, kn, jw, jn) in zip(cells, [
            (100, 50, 10, 5), (60, 40, 8, 6), (50, 30, 9, 7), (80, 45, 12, 8),
            (90, 55, 14, 9)], strict=True):
        key, row = _kc(s, 0)
        store[key] = replace(row, runtime_ns=kw, repeats=31, spread_ns=1, raw={
            **row.raw, "wallclock:near:median_ns": kn,
            "wallclock:control:median_ns": jw,
            "wallclock:control:near:median_ns": jn})
    g = measured_store.k_gain(store, "sm_86", 10, refused={})
    assert (g["wide"].k_gain, g["wide"].k_best) == (2.0, 6)
    assert (g["wide"].j_gain, g["wide"].j_best) == (1.25, 3)
    assert g["wide"].k_at_least_j
    assert (g["near"].k_gain, g["near"].j_gain) == (50 / 30, 1.0)
    del store[next(iter(store))]
    with pytest.raises(KeyError):
        measured_store.k_gain(store, "sm_86", 10, refused={})


def test_the_unlaunchable_k_cells_are_every_high_l_cell_on_both_cards() -> None:
    from felizjueves.kernel_k import shared_doubles
    high = {s for s in measured_store.K_SUBGRID if s.total_am >= 13}
    for arch, (smem, optin) in measured_store.K_SMEM_PER_BLOCK.items():
        assert set(measured_store.K_UNTIMED[arch]) == high, arch
        for s, why in measured_store.K_UNTIMED[arch].items():
            need = measured_store.K_MAX_SLOTS[(s.total_am, s.n_batches)] * 256
            assert need > smem and f"= {need} B" in why and f"{smem} B" in why
            assert ("opt-in to" in why) == (optin > smem), why
    # the one-batch cell's slots are kernel_k's own arithmetic, and a split
    # cell's largest batch kernel never holds more (Ksh is compacted)
    for L in (13, 16):
        one = measured_store.K_MAX_SLOTS[(L, 1)]
        assert one == shared_doubles(k_spec(L).am)
        assert all(measured_store.K_MAX_SLOTS[(L, b)] <= one
                   for b in measured_store.SUBGRID_BATCHES)
    # and every lower-L cell fits the smaller of the two cards' blocks
    for s in measured_store.K_SUBGRID:
        if s.total_am < 13:
            assert shared_doubles(s.am) * 256 <= min(
                v[0] for v in measured_store.K_SMEM_PER_BLOCK.values())


# --- #288: the F rows, the fused Fock kernel on both cards --------------------

# The cells recorded: #288's, the classes of total L <= 2 its default tier
# runs on the cards (test/test_fock_ondevice.py), and #290's, the sixteen
# above them (the deferred tier) -- the whole ladder, one row per (class,
# arch).
F_DEFAULT = measured.f_cells((0, 1, 2))
F_RECORDED = measured.f_cells()


def test_the_f_ladder_is_hgp_vrr_classes_and_every_class_is_recorded(
        ) -> None:
    from felizjueves.hgp.vrr import CLASSES
    assert measured.F_CLASSES == CLASSES == emit_cli.FOCK_CLASSES
    assert [s.am for s in measured.f_cells()] == list(CLASSES)
    assert [s.am for s in F_DEFAULT] == [
        (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 1, 1), (0, 1, 0, 1)]
    assert len(F_RECORDED) == 21  # #290: the deferred sixteen joined them
    for s in F_RECORDED:
        assert s.target is Target.F and s.n_batches == 1 and not s.levers


def test_an_f_cell_is_built_by_kernel_f_and_takes_no_lever() -> None:
    from felizjueves.kernel_f import build_f_kernel
    s = F_DEFAULT[1]
    assert measured.cell_kernels(s) == (build_f_kernel(s),)
    with pytest.raises(ValueError, match="no lever is declared for an F"):
        measured.cell_kernels(replace(s, levers=(RECYCLE,)))
    # its one shared array is static: no dynamic bytes at any arch
    for b in measured.BACKENDS.values():
        (u,) = b.units(s)
        assert u.name == "f_0001" and u.jsh_dynamic_bytes == 0


def test_the_f_target_refuses_what_an_f_cell_does_not_have() -> None:
    for argv in (["--remeasure", "--target", "f", "--subgrid"],
                 ["--remeasure", "--target", "f", "--lever", "recycle"],
                 ["--remeasure", "--target", "f", "--asym"],
                 ["--remeasure", "--target", "f", "--n-batches", "3"],
                 ["--remeasure", "--target", "f", "--L", "9"]):
        assert measured.main(argv) == 2, argv
