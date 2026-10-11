"""The measured store (data/measured/kernels.json) and the generator's CLI.

Fast tier, except the two `cuda`/`hip` re-measure checks at the bottom, which
need a compiler (no card): they recompile a few recorded classes and assert
the toolchain still reports what the store says -- when the toolchain is the
one the row was keyed on. A different compiler version is a different key,
and the check SKIPS naming it rather than comparing across compilers.
"""

from __future__ import annotations

import json
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
