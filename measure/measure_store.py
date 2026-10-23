"""The TRACKED measured stores, checked: data/measured/kernels.json,
roofline.json and builds.json against the code that reads them and the
generator that wrote the text they describe.

Moved from test/test_measured.py, which keeps the store's and the driver's
CODE -- the record's refusals, the loaders, the parsers on recorded or
synthetic output, every declared verdict on synthetic rows. What is here
reads the tracked files: that every row is keyed on its MeasuredKey and
carries its provenance, that each file round-trips byte for byte, that the
recorded `source_sha256` is the text the generator writes today, that the
declared verdicts read what they were recorded to read off the rows, and the
three strict `xfail`s of #205's unfinished store (K_205_PAUSED). These are
checks of measurement artefacts, not of correctness: an emitter change that
moves a hash fails here, and is answered by re-measuring
(`devtools/measure.sh remeasure ...`), never by editing a test.

    devtools/measure.sh check            # the reads (no compiler, no card)
    devtools/measure.sh recheck --backend cuda   # the recompiles below

THE RECHECKS (`cuda` / `hip`, a compiler and no card) recompile a few
recorded classes, and every `measured_store.SUBGRID` cell and lever arm,
and assert the toolchain still reports what the store says -- when the
toolchain is the one the row was keyed on. A different compiler version is
a different key, and the check SKIPS naming it rather than comparing across
compilers. Every case that builds, emits or plans a class above
`tiers.FAST_L_MAX` is `slow` (`devtools/measure.sh check --full`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_measured import (
    _F_SPEC,
    ARCHS,
    F_DEFAULT,
    F_RECORDED,
    NOT_MEASURABLE,
    RECYCLE_CELLS,
    SIEVE_CELLS,
    SK,
    TUNED_CELLS,
    _f_timed,
    _k_row,
)
from tiers import is_fast

from felizjueves import asym, gate, measured, measured_store
from felizjueves.device import GFX1200, SM_86, MeasuredKey
from felizjueves.emit import cuda, hip
from felizjueves.ir.lower import RECYCLE
from felizjueves.kernel import build_pq_kernel, pq_spec
from felizjueves.plan.pipeline import TUNED
from felizjueves.plan.sieve import NO_SIEVE
from felizjueves.spec import Target

# #205 was paused mid-run (2026-10-13, the user's call) and its partial rows
# merged: the L = 7 and L = 10 K cells, compile-only on both cards, with no
# wall-clock and nothing yet for L = 13 and L = 16. These three tests state
# #205's finished store, so they fail on the partial one. strict=True: the
# moment #205 completes they XPASS, fail, and this marker must be removed.
K_205_PAUSED = pytest.mark.xfail(
    strict=True,
    reason="#205 paused: K rows for L = 13, 16 and all K wall-clock pending")


# --- the store --------------------------------------------------------------

STORE = json.loads(measured_store.STORE.read_text())
ROWS = measured_store.load()
SPECS = {s.key(): s for s in (*(pq_spec(L) for L in range(17)), *measured_store.SUBGRID,
                              *RECYCLE_CELLS, *asym.ASYM_CELLS, *SIEVE_CELLS,
                              *TUNED_CELLS, *measured_store.K_SUBGRID,
                              *measured.f_cells(),
                              *measured_store.ledger_cells())}  # #362

# #362: the lever ledger's rows, ADDED beside every row above -- each kernel
# lever arm at L <= 4 (arm_kind KERNEL), and the K literal classes at L <= 4
# it reads them against. The checks below that pin the store as it was
# before them read the store without these.
_K_LEDGER = {c.key() for c in measured_store.ledger_classes(Target.K)}
LEDGER_KEYS = frozenset(
    k for k, r in ROWS.items() if r.arm_kind == measured_store.KERNEL
    or measured_store.split_key(k)[0] in _K_LEDGER)
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


# The two whole-store checks build every row's cell. The fast tier builds
# the rows at L <= FAST_L_MAX (tiers.py); the `_above_l4` twins the rest,
# but for #205's K cells of L >= 13, which take minutes of Python to BUILD
# (their stage-2 digest unrolled: 40 to 520 MB of text a cell) and are
# test_a_high_l_k_row_hashes_todays_text's alone. Each runs one chunk of
# the store per (arch, target), so xdist spreads the building.
K_SLOW_L = 13
TARGETS = (Target.J, Target.K, Target.F)  # every target a row has
CHUNKS = [pytest.param(a, t, id=f"{a}-{t.value}") for a in ARCHS
          for t in TARGETS]


def _chunk(arch: str, target: Target):
    for key, row in ROWS.items():
        if row.arch == arch and measured_store.target_of(key) is target:
            yield key, row


def _fast(spec) -> bool:
    return is_fast(spec.total_am)


def _above_l4(spec) -> bool:
    return not _fast(spec) and not (
        getattr(spec, "target", Target.J) is Target.K
        and spec.total_am >= K_SLOW_L)  # an AsymCell has no target


def _check_keys(keep, arch: str, target: Target) -> None:
    for key, row in _chunk(arch, target):
        spec = _spec_of(key)
        if not keep(spec):
            continue
        assert row.kernel == ",".join(k.name for k in measured.cell_kernels(spec))
        mk = measured.measured_key(spec, ARCHS[row.arch], row.toolchain)
        assert mk.key() == key


@pytest.mark.parametrize("arch, target", CHUNKS)
def test_every_row_is_keyed_on_its_measured_key(arch, target) -> None:
    assert ROWS, "no Measured rows recorded"
    assert all(r.arch in ARCHS and measured_store.target_of(k) in TARGETS
               for k, r in ROWS.items())  # the chunks cover the store
    _check_keys(_fast, arch, target)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("arch, target", CHUNKS)
def test_every_row_above_l4_is_keyed_on_its_measured_key(arch, target) -> None:
    _check_keys(_above_l4, arch, target)


def test_every_row_carries_its_provenance() -> None:
    for key, row in ROWS.items():
        assert row.host and row.gpu and row.date and row.command, key
        assert row.gpu != "ABSENT", f"{key}: measured with no card visible"
        assert row.spill_unit and row.raw, key
        assert len(row.source_sha256) == 64, key
        # The flags are config.sh's, and none of them relaxes floating point.
        assert "fast-math" not in row.command and "fast_math" not in row.command


def _check_sources(keep, arch: str, target: Target) -> None:
    for key, row in _chunk(arch, target):
        spec = _spec_of(key)
        if not keep(spec):
            continue
        units = measured.BACKENDS[BACKEND_OF[row.arch]].units(spec)
        assert measured.cell_sha256(units) == row.source_sha256, (
            f"{key}: the emitted text changed since it was measured -- "
            "python -m felizjueves.measured --remeasure")
        # the one-unit hash is the one text's (#362: K's `split` lever
        # emits several units for one batch)
        if spec.n_batches == 1 and len(units) == 1:
            emit, arch = (cuda.emit, SM_86) if row.arch == "sm_86" else (
                hip.emit, GFX1200)
            (k,) = measured.cell_kernels(spec)  # build_pq_kernel, lowered
            text = emit(k, arch)
            assert hashlib.sha256(text.encode()).hexdigest() == row.source_sha256


@pytest.mark.parametrize("arch, target", CHUNKS)
def test_the_recorded_source_is_what_the_generator_writes_today(
        arch, target) -> None:
    """A row describes THIS emitter's text: regenerate-and-remeasure on a diff."""
    _check_sources(_fast, arch, target)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("arch, target", CHUNKS)
def test_the_recorded_source_above_l4_is_what_the_generator_writes_today(
        arch, target) -> None:
    _check_sources(_above_l4, arch, target)  # K at L >= 13: the high-L twin


def _check_default_classes(keep) -> None:
    for arch in ("sm_86", "gfx1200"):
        kernels = {}
        for k, r in ROWS.items():
            s = _spec_of(k)
            if (r.arch == arch and s.n_batches == 1 and not s.levers
                    and s.target is Target.J):
                kernels.setdefault(s.total_am, set()).add(r.kernel)
        assert set(kernels) == set(range(17)), arch
        for L in filter(keep, range(17)):
            assert kernels[L] == {build_pq_kernel(L).name}, (arch, L)


def test_both_cards_and_every_default_class_are_recorded() -> None:
    _check_default_classes(is_fast)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_both_cards_and_every_default_class_above_l4_are_recorded() -> None:
    _check_default_classes(lambda L: not is_fast(L))


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
    # and #205's K rows the same way
    assert any(_spec_of(k).target is Target.K for k in split)


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
        # ...and a row without counters writes none of their fields (#110),
        # nor an arm kind it does not record (#159), nor a target (#201)
        assert set(asdict(row)) - set(STORE["rows"][key]) <= {
            *measured_store.COUNTERS, "counter_tool", "counter_unit",
            "arm_kind", "target"}, key


def test_the_store_says_what_it_is_and_what_it_is_not() -> None:
    assert "MeasuredKey(spec, arch, toolchain).key()" in STORE["about"]
    assert "Table 4" in STORE["about"]


@K_205_PAUSED
def test_the_read_path_filters_the_real_store() -> None:
    spec = pq_spec(8)
    got = measured_store.rows(spec=spec)
    assert {r.arch for r in got.values()} == {"sm_86", "gfx1200"}
    assert all(k.startswith(spec.key() + "@") for k in got)
    nv = measured_store.rows(arch=SM_86)
    # the one-batch classes L = 0..16, plus the subgrid's split cells, plus
    # the recycle arm of every subgrid cell (#55), plus #113's asym cells,
    # plus #162's no-sieve arm (the subgrid and L = 11), plus #163's tuned
    # arm (the subgrid less its four 12-batch cells)
    split = sum(s.n_batches > 1 for s in measured_store.SUBGRID)
    # (and #205's K cells, each with a row or a final refusal on the card)
    k_rows = len(measured_store.K_SUBGRID) - len(
        measured_store.K_REFUSED.get("sm_86", {}))
    assert len(nv) == (17 + split + len(RECYCLE_CELLS) + len(asym.ASYM_CELLS)
                       + len(SIEVE_CELLS) + len(TUNED_CELLS) + k_rows)
    assert measured_store.rows(arch="sm_86") == nv
    assert measured_store.rows(spec=spec, arch=GFX1200).keys() == {
        k for k in got if "@gfx1200/" in k}


def _timed(row) -> bool:
    return row.runtime_ns is not None


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", measured_store.SUBGRID, ids=lambda s: s.key())
def test_every_subgrid_cell_is_timed_on_both_cards(spec, arch) -> None:
    """#53's acceptance: runtime_ns, repeats and spread_ns, with provenance."""
    (row,) = measured_store.rows(spec=spec, arch=arch, store=ROWS).values()
    assert _timed(row), f"{spec.key()}@{arch}: no wall-clock -- run the " \
        "opt-in sweep, devtools/measure.sh wallclock"
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
        # #363's lever clock reads the host clock only (no GPU events; the
        # store's header "lever_clock" says so); every other sweep keeps them
        if r.arm_kind != measured_store.KERNEL:
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


def test_the_tracked_counter_store_reads_and_rewrites_byte_for_byte(tmp_path) -> None:
    got = measured_store.load_counters()
    p = tmp_path / "roofline.json"
    measured_store.save_counters(got, p)
    assert p.read_bytes() == measured_store.ROOFLINE.read_bytes()


def test_counter_rows_do_not_change_what_the_cell_store_reads(tmp_path) -> None:
    # The 40 literal rows mean one cell each; counter rows live elsewhere.
    assert measured_store.ROOFLINE != measured_store.STORE
    assert not any("#" in k for k in ROWS)


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


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("cell", asym.ASYM_CELLS, ids=lambda c: c.key())
def test_every_asym_cell_is_compiled_and_timed_against_both_neighbours(
        cell, arch) -> None:
    """#113's acceptance: the asymmetric row, compiled and timed behind the
    gate, with the card's state for itself AND each same-session control."""
    (r,) = measured_store.rows(spec=cell, arch=arch, store=ROWS).values()
    assert r.registers is not None and r.spill_store is not None
    assert _timed(r) and r.repeats == 31 and r.gpu != "ABSENT"
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


@pytest.mark.slow  # L = 13, 16 > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("cell", asym.ASYM_CELLS, ids=lambda c: c.key())
def test_every_asym_row_names_its_cells_batch_kernels(cell, arch) -> None:
    (r,) = measured_store.rows(spec=cell, arch=arch, store=ROWS).values()
    assert r.kernel == ",".join(k.name for k in cell.kernels())


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


BUILDS = measured_store.load_builds()


def test_the_tracked_build_store_is_rewritten_byte_for_byte(tmp_path) -> None:
    """load -> save reproduces data/measured/builds.json exactly: header,
    row order, number spelling and all."""
    p = tmp_path / "builds.json"
    measured_store.save_builds(BUILDS, p)
    assert p.read_bytes() == measured_store.BUILDS.read_bytes()


def test_both_cards_have_the_baseline_build_rows_with_their_gate_state() -> None:
    """#146's acceptance: rows for both cards, each with the card's state as
    KNOWN gate readings (the loader refuses anything else), or the refusal
    recorded in the header."""
    from measure_j_launched_ondevice import BUILD_CELLS
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


def test_the_sort_key_cells_are_the_declared_set() -> None:
    """Two inputs x two taus, both arms of each, alternating which runs
    first; the chain is the two-H2 one whose T stays in fj_boys' domain."""
    from measure_j_launched_ondevice import SORT_KEY_BUILD_CELLS
    assert SK.SORT_KEY_CELLS == (
        ("h2o_ccpvdz_j", 1e-3, "dispatch:wide"),
        ("h2o_ccpvdz_j", 1e-10, "dispatch:wide"),
        ("h2chain2_sto3g", 1e-3, 1), ("h2chain2_sto3g", 1e-10, 1))
    cells = SORT_KEY_BUILD_CELLS
    assert len(cells) == 2 * len(SK.SORT_KEY_CELLS)
    assert {c.arm for c in cells} == {SK.SORT_AB_ARM}
    firsts = [cells[i].density_sort for i in range(0, len(cells), 2)]
    assert firsts == [True, False, True, False]
    for a, b in zip(cells[::2], cells[1::2], strict=True):
        assert (a.fixture, a.tau, a.n) == (b.fixture, b.tau, b.n)
        assert a.density_sort != b.density_sort
        assert a.n_batches == b.n_batches


# The tracked rows. What the declared rule reads off them -- the result, not
# a tuning target: a re-timing that moves a cell fails here, and the claim's
# reason and limits.json's record must move with it.
SORT_KEY_VERDICTS = {
    "sm_86": (SK.BETTER, SK.UNRESOLVED, SK.WORSE, SK.BETTER),
    "gfx1200": (SK.UNRESOLVED, SK.BETTER, SK.WORSE, SK.BETTER),
}


def test_every_sort_key_cell_has_both_arms_from_one_session() -> None:
    from test_launch_oracle import SORT_KEY_SKIPS
    for arch in BUILD_ARCHS:
        rows = [r for r in BUILDS.values()
                if r.arch == arch and r.arm == SK.SORT_AB_ARM]
        assert len(rows) == 2 * len(SK.SORT_KEY_CELLS), arch
        assert len({r.raw[SK.SESSION_FIELD] for r in rows}) == 1, arch
        assert len({(r.toolchain, r.host, r.gpu) for r in rows}) == 1, arch
        for cell in SK.SORT_KEY_CELLS:
            ab = SK.sort_key_ab(BUILDS, arch, cell)
            n, sk_p, sk_g = SORT_KEY_SKIPS[cell[:2]]
            # the rows' counts are the launch oracle's, recomputed there
            assert (ab.n_pairs, *ab.skipped) == (n, sk_p, sk_g), (arch, cell)


def test_the_sort_key_verdict_is_contradicted_on_both_cards() -> None:
    """#147 under the rule declared before its rows: the global key skips
    no pair anywhere, yet it is BETTER on the clock at some cell on each
    card, so the expectation ("the global key is worse") is CONTRADICTED on
    sm_86 AND gfx1200. Held cell by cell; claims.json's reason and
    limits.json's record quote these verdicts and are checked against them."""
    lim = json.loads((Path(measured_store.REPO) / "data" / "literature"
                      / "jengine" / "limits.json").read_text())["limits"]
    (rec,) = [x for x in lim if x["asserted_value"].get("claim")
              == "jengine/ket-sort-key-beats-the-global-max"]
    av = rec["asserted_value"]
    for arch, want in SORT_KEY_VERDICTS.items():
        got = tuple(SK.sort_key_ab(BUILDS, arch, c).verdict
                    for c in SK.SORT_KEY_CELLS)
        assert got == want, (arch, got)
        assert SK.sort_key_arch_verdict(BUILDS, arch) == SK.CONTRADICTED
        assert av["verdict"][arch] == SK.CONTRADICTED
        assert tuple(av["cells"][arch].values()) == want
        for c in SK.SORT_KEY_CELLS:
            assert SK.sort_key_ab(BUILDS, arch, c).skip_sign == SK.FEWER
    assert not SK.sort_key_claim_holds(BUILDS, BUILD_ARCHS)
    from felizjueves.literature.claims import load_claims
    (claim,) = [c for c in load_claims(Path(measured_store.REPO) / "data"
                                       / "literature" / "jengine"
                                       / "claims.json")
                if c.id == "jengine/ket-sort-key-beats-the-global-max"]
    assert claim.status.value == "unverifiable"
    assert "CONTRADICTED on BOTH cards" in claim.reason

# #55's recycle verdict as the tracked rows gave it BEFORE #159 generalised
# the arms (f95f87b): sha256 of the repr of every cell's LeverAB, wide sign,
# near sign and verdict, sm_86 then gfx1200, in lever_cells order.
RECYCLE_AB_SHA256 = (
    "7e5b8418d45749ee695bb5e4a9c65118e55bac4a2df1b98b0314f5114b2d8d60")


def test_recycle_rows_and_verdict_did_not_move() -> None:
    """#159's whole acceptance: the 40 recycle rows load unchanged (no kind
    stamped on them, the file written back byte for byte) and #55's verdict
    is byte-identical to the one before the generalisation."""
    keys = [MeasuredKey(s, a, t).key() for s in RECYCLE_CELLS
            for a, t in ((SM_86, "nvcc 13.0"), (GFX1200, "hipcc / ROCm 7.2.4"))]
    assert len(keys) == 40 and set(keys) <= set(ROWS)
    for k in keys:
        assert "arm_kind" not in STORE["rows"][k]
        assert ROWS[k].arm_kind == ""
        assert measured_store.as_record(ROWS[k]) == STORE["rows"][k]
    got = repr([(c, c.wide_sign, c.near_sign, c.verdict)
                for c in (measured_store.lever_ab(ROWS, a, s)
                          for a in ARCHS for s in RECYCLE_CELLS)])
    assert hashlib.sha256(got.encode()).hexdigest() == RECYCLE_AB_SHA256
    assert {measured_store.lever_ab(ROWS, "sm_86", s).kind
            for s in RECYCLE_CELLS} == {measured_store.LOWERING}


def test_the_store_file_round_trips_byte_for_byte(tmp_path: Path) -> None:
    p = tmp_path / "k.json"
    measured_store.save(ROWS, p)
    assert p.read_bytes() == measured_store.STORE.read_bytes()


# --- #162: the sieve A/B on both cards ----------------------------------------

# The 122 rows the store held before #162 (f95f87b..3191838): sha256 of their
# records, json.dumps(sort_keys=True). #162 only ADDS rows; this pins that
# every row it found -- #55's recycle rows among them -- is byte-identical.
PRE_SIEVE_ROWS = 122
PRE_SIEVE_SHA256 = (
    "01153b046c1e969b2d01fea2c0b86129333acb51fc9166184dc94c8b092cf997")


def test_the_sieve_ab_added_rows_and_moved_none() -> None:
    old = {k: v for k, v in STORE["rows"].items()
           if NO_SIEVE not in k and f"/{TUNED}@" not in k  # #163's, below
           and measured_store.target_of(k) is Target.J  # #205's K rows
           and k not in LEDGER_KEYS}  # #362's
    assert len(old) == PRE_SIEVE_ROWS
    got = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    assert got == PRE_SIEVE_SHA256
    new = {k for k in STORE["rows"] if NO_SIEVE in k}
    assert new == {MeasuredKey(s, a, ROWS[k].toolchain).key()
                   for s in SIEVE_CELLS for a in (SM_86, GFX1200)
                   for k in measured_store.rows(spec=replace(s, levers=()),
                                                arch=a, store=ROWS)}
    assert len(new) == 2 * len(SIEVE_CELLS) == 42


def test_the_sieve_cells_are_the_subgrid_and_l11() -> None:
    assert SIEVE_CELLS[:20] == measured_store.lever_cells(NO_SIEVE)
    (l11,) = SIEVE_CELLS[20:]
    assert l11 == pq_spec(11, levers=(NO_SIEVE,))
    assert (l11.total_am, l11.n_batches, l11.am) == (11, 1, (6, 0, 5, 0))
    # its literal control is the store's one-batch L = 11 row, on both cards
    for a in ARCHS:
        assert len(measured_store.rows(spec=pq_spec(11), arch=a,
                                       store=ROWS)) == 1
    # an arm with no added cell is its subgrid cells alone (#55 unchanged)
    assert measured_store.ab_cells(RECYCLE) == RECYCLE_CELLS
    assert measured_store.ab_cells(NO_SIEVE) == SIEVE_CELLS
    assert measured_store.sieve_ab.__defaults__ == (None,)


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", SIEVE_CELLS, ids=lambda s: s.key())
def test_every_sieve_cell_is_compiled_and_timed_against_a_control(
        spec, arch) -> None:
    """#162's acceptance: a no-sieve row per cell and card, compiled, timed
    behind the gate with its same-session literal control, the card's state
    known at every phase and the discarded phases counted."""
    (row,) = measured_store.rows(spec=spec, arch=arch, store=ROWS).values()
    (lit,) = measured_store.rows(spec=replace(spec, levers=()), arch=arch,
                                 store=ROWS).values()
    assert row.arm_kind == measured_store.PLAN
    assert STORE["rows"][MeasuredKey(spec, ARCHS[arch], row.toolchain).key()][
        "arm_kind"] == "plan"
    assert row.toolchain == lit.toolchain and row.gpu == lit.gpu != "ABSENT"
    assert row.kernel == lit.kernel  # the same batch kernels, by name
    assert row.registers is not None and row.compile_s is not None
    assert row.spill_store is not None and row.spill_load is not None
    assert _timed(row) and row.repeats == 31, (
        f"{spec.key()}@{arch}: FJ_WALLCLOCK_LEVER=no-sieve sweep not recorded")
    w = row.raw
    for p in ("wallclock:control:", "wallclock:control:near:"):
        assert w[f"{p}min_ns"] <= w[f"{p}q1_ns"] <= w[f"{p}median_ns"] \
            <= w[f"{p}q3_ns"] <= w[f"{p}max_ns"], (spec.key(), p)
        assert w[f"{p}cpu_busy_permille_before"] <= 100, (spec.key(), p)
    # every phase, both arms and both inputs, passed the gate clean: a known
    # reading at most GPU_BUSY_MAX before AND after (a phase that was not is
    # discarded and retaken, and counted)
    for p in ("wallclock:", "wallclock:near:", "wallclock:control:",
              "wallclock:control:near:"):
        for f in ("gpu_util_before", "gpu_util_after"):
            v = w[p + f]
            assert type(v) is int and 0 <= v <= GPU_BUSY_MAX, (spec.key(), p, f)
    d = w["wallclock:gate_discarded_phases"]
    assert type(d) is int and d >= 0
    assert w["wallclock:J_bitwise_repeatable"] in (0, 1)


def test_the_cli_prints_the_sieve_ab_and_its_verdict(capsys) -> None:
    assert measured.main(["--lever-ab", NO_SIEVE]) == 0
    out = capsys.readouterr().out
    for a in ARCHS:
        assert f"{a}: the sieve {measured_store.sieve_verdict(ROWS, a).upper()}" \
            in out
        assert f"{a} L=11 b1:" in out
    assert out.count(" -> no-sieve ") == 2 * len(SIEVE_CELLS)


SIEVE_PARAMS = [
    pytest.param(s, id=s.key(),
                 marks=[pytest.mark.slow] if s.total_am >= 13 else [])
    for s in SIEVE_CELLS]


@pytest.mark.cuda
@pytest.mark.parametrize("spec", SIEVE_PARAMS)
def test_nvcc_still_reports_the_sieve_cells(spec, nvcc, tmp_path) -> None:
    _recheck_cell("cuda", spec, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("spec", SIEVE_PARAMS)
def test_hipcc_still_reports_the_sieve_cells(spec, hipcc, tmp_path) -> None:
    _recheck_cell("hip", spec, tmp_path)


# The sieve A/B as the rule read the tracked rows (#162): (arch, L, n_batches)
# -> (wide sign, near sign) of no-sieve against its same-session sieved
# control. Pinned as the declared rule (measured_store, THE SIEVE ARM) said
# it, so a re-recording that moves it is seen; #164 scores the claim.
SIEVE_AB: dict[tuple[str, int, int], tuple[str, str]] = {
    **{(a, s.total_am, s.n_batches): ("neutral", "neutral")
       for a in ARCHS for s in SIEVE_CELLS},
    ("sm_86", 16, 1): ("slower", "slower"),
    ("sm_86", 16, 6): ("faster", "faster"),
    ("sm_86", 16, 12): ("faster", "neutral"),
    ("gfx1200", 7, 3): ("neutral", "slower"),
    ("gfx1200", 7, 6): ("neutral", "slower"),
    ("gfx1200", 7, 9): ("neutral", "slower"),
    ("gfx1200", 7, 12): ("neutral", "slower"),
    ("gfx1200", 10, 1): ("neutral", "faster"),
    ("gfx1200", 10, 3): ("neutral", "slower"),
    ("gfx1200", 11, 1): ("neutral", "slower"),
    ("gfx1200", 13, 1): ("slower", "slower"),
    ("gfx1200", 13, 3): ("neutral", "slower"),
    ("gfx1200", 16, 1): ("slower", "slower"),
    ("gfx1200", 16, 6): ("neutral", "faster"),
}


def test_the_sieve_ab_is_what_the_rows_say() -> None:
    got = {}
    for arch in ARCHS:
        for c in measured_store.sieve_ab(ROWS, arch).values():
            got[arch, c.L, c.n_batches] = (c.wide_sign, c.near_sign)
    assert got == SIEVE_AB
    # WINs and LOSSes on both cards: the declared rule says NOT SHOWN on
    # each, and the sieved plan stays the default
    for arch in ARCHS:
        assert measured_store.sieve_verdict(ROWS, arch) == "not shown"
    assert not measured_store.sieve_may_leave_the_default(ROWS)

# The 164 rows the store held before #163 (00e872c): sha256 of their records,
# json.dumps(sort_keys=True). #163 only ADDS rows; this pins that every row
# it found -- the literal controls, #55's and #162's arms -- is byte-identical.
PRE_TUNED_ROWS = 164
PRE_TUNED_SHA256 = (
    "ea45730e45f8f028e0585079d7eacac00083917b94d141581826081b1429e112")

# Where the search chose what the literal plan does, the tuned text IS the
# literal text (#161's eight unchanged cells, three of them on the subgrid).
TUNED_EQUALS_LITERAL = {(7, 3), (7, 6), (10, 9)}
# Where the recorded cut is not the literal (minimax) one, the batch kernels
# themselves are other kernels, named by their other ranges: on the subgrid,
# L = 7 at 9 batches only. Everywhere else the tuned arm moves only the seed
# order, inside the same batch kernels.
TUNED_OTHER_CUT = {(7, 9)}


def test_the_tuned_ab_added_rows_and_moved_none() -> None:
    old = {k: v for k, v in STORE["rows"].items() if f"/{TUNED}@" not in k
           and measured_store.target_of(k) is Target.J  # #205's K rows
           and k not in LEDGER_KEYS}  # #362's
    assert len(old) == PRE_TUNED_ROWS
    got = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    assert got == PRE_TUNED_SHA256
    new = {k for k in STORE["rows"] if f"/{TUNED}@" in k}
    assert new == {MeasuredKey(s, a, ROWS[k].toolchain).key()
                   for s in TUNED_CELLS for a in (SM_86, GFX1200)
                   for k in measured_store.rows(spec=replace(s, levers=()),
                                                arch=a, store=ROWS)}
    assert len(new) == 2 * len(TUNED_CELLS) == 32


def test_a_not_measurable_cell_has_no_row_and_no_literal_stand_in() -> None:
    """The four 12-batch cells are recorded as not measurable, never as a
    row -- least of all one of literal text under the tuned key."""
    for spec in NOT_MEASURABLE:
        for a in ARCHS:
            assert not measured_store.rows(spec=spec, arch=a, store=ROWS)


@pytest.mark.parametrize("arch", ("sm_86", "gfx1200"))
@pytest.mark.parametrize("spec", TUNED_CELLS, ids=lambda s: s.key())
def test_every_tuned_cell_is_compiled_and_timed_against_a_control(
        spec, arch) -> None:
    """#163's acceptance: a tuned row per measurable cell and card,
    compiled, timed behind the gate with its same-session literal control,
    the card's state known at every phase and the discarded phases counted."""
    (row,) = measured_store.rows(spec=spec, arch=arch, store=ROWS).values()
    (lit,) = measured_store.rows(spec=replace(spec, levers=()), arch=arch,
                                 store=ROWS).values()
    assert row.arm_kind == measured_store.PLAN
    assert STORE["rows"][MeasuredKey(spec, ARCHS[arch], row.toolchain).key()][
        "arm_kind"] == "plan"
    assert row.toolchain == lit.toolchain and row.gpu == lit.gpu != "ABSENT"
    cell = (spec.total_am, spec.n_batches)
    assert (row.kernel != lit.kernel) is (cell in TUNED_OTHER_CUT)
    assert len(row.kernel.split(",")) == len(lit.kernel.split(",")) \
        == spec.n_batches
    # the text moved exactly where the search chose other than literal
    same = cell in TUNED_EQUALS_LITERAL
    assert (row.source_sha256 == lit.source_sha256) is same
    assert row.registers is not None and row.compile_s is not None
    assert row.spill_store is not None and row.spill_load is not None
    assert _timed(row) and row.repeats == 31, (
        f"{spec.key()}@{arch}: FJ_WALLCLOCK_LEVER=tuned sweep not recorded")
    w = row.raw
    for p in ("wallclock:control:", "wallclock:control:near:"):
        assert w[f"{p}min_ns"] <= w[f"{p}q1_ns"] <= w[f"{p}median_ns"] \
            <= w[f"{p}q3_ns"] <= w[f"{p}max_ns"], (spec.key(), p)
        assert w[f"{p}cpu_busy_permille_before"] <= 100, (spec.key(), p)
    for p in ("wallclock:", "wallclock:near:", "wallclock:control:",
              "wallclock:control:near:"):
        for f in ("gpu_util_before", "gpu_util_after"):
            v = w[p + f]
            assert type(v) is int and 0 <= v <= GPU_BUSY_MAX, (spec.key(), p, f)
    d = w["wallclock:gate_discarded_phases"]
    assert type(d) is int and d >= 0
    assert w["wallclock:J_bitwise_repeatable"] in (0, 1)


def test_the_cli_prints_the_tuned_ab_its_verdict_and_the_unmeasurable(
        capsys) -> None:
    assert measured.main(["--lever-ab", TUNED]) == 0
    out = capsys.readouterr().out
    for a in ARCHS:
        assert (f"{a}: the search "
                f"{measured_store.tuned_verdict(ROWS, a).upper()}") in out
        for spec in NOT_MEASURABLE:
            assert f"{a} {spec.key()}: NOT MEASURABLE -- no recorded choice" \
                in out
    assert out.count(f" -> {TUNED} ") == 2 * len(TUNED_CELLS)


TUNED_PARAMS = [
    pytest.param(s, id=s.key(),
                 marks=[pytest.mark.slow] if s.total_am >= 13 else [])
    for s in TUNED_CELLS]


@pytest.mark.cuda
@pytest.mark.parametrize("spec", TUNED_PARAMS)
def test_nvcc_still_reports_the_tuned_cells(spec, nvcc, tmp_path) -> None:
    _recheck_cell("cuda", spec, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("spec", TUNED_PARAMS)
def test_hipcc_still_reports_the_tuned_cells(spec, hipcc, tmp_path) -> None:
    _recheck_cell("hip", spec, tmp_path)


# The tuned A/B as the rule read the tracked rows (#163): (arch, L, n_batches)
# -> (wide sign, near sign) of tuned against its same-session literal
# control. Pinned as the declared rule (measured_store, THE TUNED ARM) said
# it, so a re-recording that moves it is seen; #165/#166 read it.
#
# The three cells whose tuned text IS the literal text (TUNED_EQUALS_LITERAL)
# are the A/B's own null: on gfx1200 two of them, (7,3) and (7,6), read as
# SLOWER -- identical code judged apart by the rule, i.e. the protocol's noise
# floor on that card at L = 7 (#55 saw the same on identical machine code).
TUNED_AB: dict[tuple[str, int, int], tuple[str, str]] = {
    **{(a, s.total_am, s.n_batches): ("neutral", "neutral")
       for a in ARCHS for s in TUNED_CELLS},
    ("sm_86", 7, 9): ("slower", "neutral"),
    ("sm_86", 16, 1): ("neutral", "faster"),
    ("sm_86", 16, 3): ("faster", "faster"),
    ("sm_86", 16, 6): ("faster", "faster"),
    ("sm_86", 16, 9): ("neutral", "faster"),
    ("gfx1200", 7, 3): ("slower", "slower"),
    ("gfx1200", 7, 6): ("neutral", "slower"),
    ("gfx1200", 7, 9): ("slower", "slower"),
    ("gfx1200", 10, 1): ("neutral", "faster"),
    ("gfx1200", 10, 3): ("neutral", "slower"),
    ("gfx1200", 13, 1): ("neutral", "slower"),
    ("gfx1200", 16, 3): ("neutral", "faster"),
    ("gfx1200", 16, 6): ("neutral", "faster"),
    ("gfx1200", 16, 9): ("neutral", "faster"),
}
TUNED_VERDICT = {"sm_86": "not shown", "gfx1200": "not shown"}


def test_the_tuned_ab_is_what_the_rows_say() -> None:
    got = {}
    for arch in ARCHS:
        for c in measured_store.tuned_ab(ROWS, arch).values():
            got[arch, c.L, c.n_batches] = (c.wide_sign, c.near_sign)
    assert got == TUNED_AB
    # WINs at L = 16 and LOSSes at L = 7 (and on gfx1200 at L = 10 and 13)
    # on both cards: the declared rule says NOT SHOWN on each, and the
    # literal plan stays the default
    for arch in ARCHS:
        assert measured_store.tuned_verdict(ROWS, arch) == TUNED_VERDICT[arch]
    assert not measured_store.tuned_may_become_the_default(ROWS)


# #205: the 196 J rows the store held before any K row, as written, digested
# (sha256 of json.dumps(rows, sort_keys=True)) -- the K rows are ADDED.
J_ROWS_BEFORE_K = (196, "5327147d08704fbfcb9cad8134dc923111fce9e4d0b16b036e5ec33d"
                        "51f60c82")


def test_every_tracked_row_is_a_j_k_or_f_row_and_the_j_rows_did_not_move(
        ) -> None:
    # #201's acceptance on the tracked store: a J row records no target (its
    # record is read back as written, and the file round-trips byte for byte
    # in test_the_store_file_round_trips_byte_for_byte); #205's: the K rows
    # are added beside the J rows, every J row exactly as before, and every
    # K row is a K_SUBGRID cell that says so; #288's: so is every F row, a
    # cell of F_RECORDED.
    assert len(ROWS) == len(STORE["rows"])
    j_rows = {k: v for k, v in STORE["rows"].items()
              if measured_store.target_of(k) is Target.J
              and k not in LEDGER_KEYS}  # #362's are added beside them
    assert (len(j_rows), hashlib.sha256(json.dumps(
        j_rows, sort_keys=True).encode()).hexdigest()) == J_ROWS_BEFORE_K
    ledger = {c.key() for c in measured_store.ledger_cells()}
    for key, row in ROWS.items():
        if key in LEDGER_KEYS:  # #362: a ledger cell, its target stamped
            assert measured_store.split_key(key)[0] in ledger, key
            assert row.target == ("" if measured_store.target_of(key)
                                  is Target.J else
                                  measured_store.target_of(key).value), key
        elif measured_store.target_of(key) is Target.J:
            assert row.target == "" and "target" not in STORE["rows"][key], key
        elif measured_store.target_of(key) is Target.F:
            assert row.target == "f" == STORE["rows"][key]["target"], key
            assert any(key.startswith(s.key() + "@") for s in F_RECORDED), key
        else:
            assert measured_store.target_of(key) is Target.K, key
            assert row.target == "k" == STORE["rows"][key]["target"], key
            assert any(key.startswith(s.key() + "@")
                       for s in measured_store.K_SUBGRID), key
    assert measured_store.rows(store=ROWS, target=Target.J) == {
        k: ROWS[k] for k in (*j_rows, *(k for k in LEDGER_KEYS
                                        if measured_store.target_of(k)
                                        is Target.J))}
    for k, r in measured_store.load_counters().items():
        cell, _ = measured_store.split_kernel_key(k)
        assert measured_store.target_of(cell) is Target.J, k
        assert r.target == "", k


def test_a_k_cell_round_trips_through_the_store(tmp_path: Path) -> None:
    # a toolchain no tracked row is keyed on, so the K row is new (#205)
    k = MeasuredKey(measured_store.K_SUBGRID[0], SM_86, "nvcc 0.0 (synthetic)")
    j = MeasuredKey(measured_store.SUBGRID[0], SM_86, "nvcc 13.0")
    row = replace(_k_row(), toolchain=k.toolchain)
    assert not row.target
    store = measured_store.merge(ROWS, k, row)  # merge stamps the target
    assert store[k.key()].target == "k"
    assert store[j.key()] is ROWS[j.key()]  # the J twin is untouched
    p = tmp_path / "m.json"
    measured_store.save(store, p)
    doc = json.loads(p.read_text())
    assert doc["rows"][k.key()]["target"] == "k"
    assert {kk: v for kk, v in doc["rows"].items() if kk != k.key()} \
        == STORE["rows"]
    back = measured_store.load(p)
    assert back == store
    q = tmp_path / "again.json"
    measured_store.save(back, q)
    assert q.read_bytes() == p.read_bytes()
    # the read path: by spec, by target, and never through the J twin
    assert measured_store.rows(spec=k.spec, toolchain=k.toolchain,
                               store=back) == {k.key(): back[k.key()]}
    assert measured_store.rows(target=Target.K, store=back) == {
        k.key(): back[k.key()], **measured_store.rows(target=Target.K,
                                                      store=ROWS)}
    assert measured_store.rows(target=Target.J, store=back) == \
        measured_store.rows(target=Target.J, store=ROWS)
    assert measured_store.rows(spec=j.spec, arch="sm_86", store=back) == {
        j.key(): ROWS[j.key()]}
    # a J row merged today carries no target, as before #201
    assert measured_store.merge({}, j, _k_row())[j.key()].target == ""


def test_the_f_rule_was_declared_in_the_store() -> None:
    doc = measured_store.__doc__
    assert "THE F CELL (#278)" in doc
    for part in ("ITS KEY", "ITS TARGET IS ON THE ROW", "ITS UNITS",
                 "ITS GATE IS NOT OPTIONAL", "THE HEADER"):
        assert part in doc, part
    # the row shape did not change, so the schema did not move
    assert measured_store.SCHEMA == 2 == STORE["schema"]
    # the header's "f" entry is written by `save` once an F row exists --
    # since #288's rows, the tracked store's -- and is never in HEADER itself
    assert "f" not in measured_store.HEADER
    assert STORE["header"] == {**measured_store.HEADER,
                               "f": measured_store.F_ABOUT,
                               # #362's KERNEL rows bring #339's entry
                               "kernel_levers":
                                   measured_store.KERNEL_LEVERS_ABOUT,
                               # #363's timed KERNEL rows bring its entry
                               "lever_clock":
                                   measured_store.LEVER_CLOCK_ABOUT}


def test_a_synthetic_f_row_round_trips_through_the_store(
        tmp_path: Path) -> None:
    # #278's acceptance: written, read back, keyed. A toolchain no tracked
    # row is keyed on, so the F row is new and every tracked row untouched.
    k = MeasuredKey(_F_SPEC, SM_86, "nvcc 0.0 (synthetic)")
    assert k.key() == ("coulomb/d0/(2,1|1,0)/b1/f/literal@sm_86/"
                       "nvcc 0.0 (synthetic)")
    row = replace(_f_timed(), toolchain=k.toolchain)
    assert not row.target
    store = measured_store.merge(ROWS, k, row)  # merge stamps the target
    assert store[k.key()].target == "f"
    assert all(store[kk] is ROWS[kk] for kk in ROWS)
    p = tmp_path / "m.json"
    measured_store.save(store, p)
    doc = json.loads(p.read_text())
    assert doc["schema"] == STORE["schema"]
    assert doc["rows"][k.key()]["target"] == "f"
    assert {kk: v for kk, v in doc["rows"].items() if kk != k.key()} \
        == STORE["rows"]
    # the header gains its "f" entry, and only that
    assert doc["header"] == {**STORE["header"], "f": measured_store.F_ABOUT}
    back = measured_store.load(p)
    assert back == store
    q = tmp_path / "again.json"
    measured_store.save(back, q)
    assert q.read_bytes() == p.read_bytes()
    # the read path: by spec, by target, and the J and K reads unmoved
    assert measured_store.target_of(k.key()) is Target.F
    assert measured_store.rows(spec=_F_SPEC, store=back) == {
        k.key(): back[k.key()]}
    assert measured_store.rows(target=Target.F, store=back) == {
        k.key(): back[k.key()], **measured_store.rows(target=Target.F,
                                                      store=ROWS)}
    for t in (Target.J, Target.K):
        assert measured_store.rows(target=t, store=back) == \
            measured_store.rows(target=t, store=ROWS)
    # without the synthetic row the file is the tracked one again
    r = tmp_path / "without.json"
    measured_store.save({kk: v for kk, v in back.items() if kk != k.key()}, r)
    assert r.read_bytes() == measured_store.STORE.read_bytes()


@K_205_PAUSED
def test_every_k_cell_has_a_row_or_a_final_refusal_on_both_cards() -> None:
    # #205's acceptance: rows for every K_SUBGRID cell on both cards, or a
    # FINAL recorded refusal with its mechanism -- and never both.
    for arch in ("sm_86", "gfx1200"):
        refused = measured_store.K_REFUSED.get(arch, {})
        for s in measured_store.K_SUBGRID:
            got = measured_store.rows(spec=s, arch=arch, store=ROWS)
            if s in refused:
                assert refused[s].strip(), (arch, s.key())
                assert not any(r.registers is not None for r in got.values()), (
                    f"{s.key()} @ {arch} is refused yet compiled")
            else:
                assert len(got) == 1, f"{s.key()} @ {arch}: no row, no refusal"
                (r,) = got.values()
                assert r.registers is not None and r.source_sha256, s.key()


@K_205_PAUSED
def test_every_k_row_is_timed_beside_its_j_twin_or_recorded_untimed() -> None:
    for arch in measured_store.K_SMEM_PER_BLOCK:
        off = {**measured_store.K_REFUSED.get(arch, {}),
               **measured_store.K_UNTIMED[arch]}
        for s in measured_store.K_SUBGRID:
            for key, r in measured_store.rows(spec=s, arch=arch,
                                              store=ROWS).items():
                if s in off:
                    assert r.runtime_ns is None, key
                else:
                    assert r.runtime_ns is not None, f"{key}: not timed"
                    assert "wallclock:control:median_ns" in r.raw, key
                    assert "wallclock:control:near:median_ns" in r.raw, key
                    assert r.raw["wallclock:gate_discarded_phases"] >= 0


@pytest.mark.slow  # L = 7 > FAST_L_MAX (tiers.py)
def test_the_k_rows_hash_the_text_the_generator_writes_today() -> None:
    # the compile row's source_sha256 is measured.cell_sha256 of today's
    # emitted batch texts (cheap at L = 7; the rechecks below cover the rest)
    for arch, b in (("sm_86", measured.BACKENDS["cuda"]),
                    ("gfx1200", measured.BACKENDS["hip"])):
        for s in measured_store.K_SUBGRID:
            if s.total_am != 7:
                continue
            got = measured_store.rows(spec=s, arch=arch, store=ROWS)
            if not got:
                continue
            (r,) = got.values()
            assert r.source_sha256 == measured.cell_sha256(b.units(s)), s.key()


K_PARAMS = [
    pytest.param(s, id=s.key(),
                 marks=[pytest.mark.slow] if s.total_am >= 13 else [])
    for s in measured_store.K_SUBGRID]


@pytest.mark.cuda
@pytest.mark.parametrize("spec", K_PARAMS)
def test_nvcc_still_reports_the_k_cells(spec, nvcc, tmp_path) -> None:
    _recheck_k_cell("cuda", spec, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("spec", K_PARAMS)
def test_hipcc_still_reports_the_k_cells(spec, hipcc, tmp_path) -> None:
    _recheck_k_cell("hip", spec, tmp_path)


def _recheck_k_cell(backend_name: str, spec, tmp_path: Path) -> None:
    arch = measured.BACKENDS[backend_name].arch.name
    why = measured_store.K_REFUSED.get(arch, {}).get(spec)
    if why is not None:
        pytest.skip(f"REFUSED, recorded (measured_store.K_REFUSED): {why}")
    b = measured.BACKENDS[backend_name]
    compiler = b.compiler()
    toolchain = b.toolchain(compiler)
    key = measured.measured_key(spec, b.arch, toolchain).key()
    if key not in ROWS:
        pytest.skip(f"no row for {key}: this toolchain is not the recorded "
                    "one (a different key, not a regression)")
    want = ROWS[key]
    _, row = measured.measure(b, spec, compiler=compiler, toolchain=toolchain,
                              gpu=want.gpu, workdir=tmp_path,
                              timeout=K_RECHECK_TIMEOUT_S,
                              jobs=spec.n_batches)
    assert row.source_sha256 == want.source_sha256, key
    compiled = {k: v for k, v in want.raw.items() if not k.startswith("wallclock:")}
    assert (row.registers, row.spill_store, row.spill_load, row.raw) == (
        want.registers, want.spill_store, want.spill_load, compiled), key


K_RECHECK_TIMEOUT_S = 6 * 3600


@pytest.mark.slow
@pytest.mark.parametrize("spec", [s for s in measured_store.K_SUBGRID
                                  if s.total_am >= K_SLOW_L],
                         ids=lambda s: s.key())
def test_a_high_l_k_row_hashes_todays_text(spec) -> None:
    """The two whole-store reads, for the K cells of L >= 13 their
    `_above_l4` twins skip."""
    got = measured_store.rows(spec=spec, store=ROWS)
    if not got:
        pytest.skip(f"{spec.key()}: no row on either card (K_REFUSED)")
    for key, row in got.items():
        b = measured.BACKENDS[BACKEND_OF[row.arch]]
        units = b.units(spec)
        assert row.kernel == ",".join(u.name for u in units), key
        assert measured.measured_key(spec, b.arch, row.toolchain).key() == key
        assert measured.cell_sha256(units) == row.source_sha256, key


def _check_f_rows(keep) -> None:
    got = {k: r for k, r in measured_store.rows(target=Target.F,
                                                store=ROWS).items()
           if k not in LEDGER_KEYS}  # #362's lever arms, compile-only
    want = {measured.measured_key(s, by_arch, ROWS_TOOLCHAIN[by_arch.name])
            .key(): s for s in F_RECORDED for by_arch in (SM_86, GFX1200)}
    assert set(got) == set(want)
    for key, row in got.items():
        spec = want[key]
        if keep(spec.total_am):  # the emitted text: built per class
            b = measured.BACKENDS[BACKEND_OF[row.arch]]
            units = b.units(spec)
            assert row.kernel == units[0].name, key
            assert row.source_sha256 == measured.cell_sha256(units), key
        assert row.target == "f", key
        assert row.registers is not None and row.spill_store is not None, key
        assert row.shared_bytes == 8192, key  # ab_sh: 128 x 8 doubles
        assert row.raw["jsh_dynamic_bytes"] == 0, key
        # #290: every class ran on both cards, so every F row is timed, and
        # (#278) a timed F row is gated
        assert row.runtime_ns is not None, key
        assert all(row.raw[f] is not None
                   for f in measured_store.GPU_UTIL_FIELDS), key


def test_the_tracked_f_rows_are_the_recorded_cells_on_both_cards() -> None:
    _check_f_rows(is_fast)


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_tracked_f_rows_above_l4_are_the_recorded_cells_on_both_cards(
        ) -> None:
    _check_f_rows(lambda L: not is_fast(L))


ROWS_TOOLCHAIN = {"sm_86": "nvcc 13.0", "gfx1200": "hipcc / ROCm 7.2.4"}


# #290's sixteen (total L > 2) recompile only in the deferred tier: (dd|dd)
# is ~15 s through either compiler
F_PARAMS = [pytest.param(s, id=s.key(),
                         marks=[] if s in F_DEFAULT else [pytest.mark.slow])
            for s in F_RECORDED]


@pytest.mark.cuda
@pytest.mark.parametrize("spec", F_PARAMS)
def test_nvcc_still_reports_the_f_cells(spec, nvcc, tmp_path) -> None:
    _recheck_cell("cuda", spec, tmp_path)


@pytest.mark.hip
@pytest.mark.parametrize("spec", F_PARAMS)
def test_hipcc_still_reports_the_f_cells(spec, hipcc, tmp_path) -> None:
    _recheck_cell("hip", spec, tmp_path)
