"""#339: the lever ladder as a registry, and the store's one lever-row path.

Beside the shipped vocabulary (`fused`, #344), the tests here declare a PROBE
lever legal where they need it (`monkeypatch.setitem`), the way a lever
issue declares its one entry -- and
nothing else is edited for it: the generator CLI's flags come from
`lever_ladder.flags()` (test/test_emit_cli.py drives them), and the store
records a lever row through `measured_store.lever_arm` and `merge`, keyed by
the full lever-carrying `Spec.key()`.

Fast tier: compiles nothing, launches nothing.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from felizjueves import levers
from felizjueves.emit import __main__ as emit_cli
from felizjueves.ir import lever_ladder
from felizjueves.kernel_f import f_class_spec
from felizjueves.kernel_k import build_k_kernels
from felizjueves.mmd.kernel import build_pq_kernels, k_spec, pq_spec
from felizjueves.performance import measured, measured_store
from felizjueves.performance.device import GFX1200, SM_86, MeasuredKey

PROBE = "probe"
PROV = dict(toolchain="nvcc 13.0", arch="sm_86", host="h", gpu="g",
            date="2026-10-15")


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> levers.Lever:
    lever = levers.Lever.of(PROBE, 339, "the ladder's witness", {"j", "k", "f"})
    monkeypatch.setitem(levers.VOCABULARY, PROBE, lever)
    return lever


@pytest.fixture
def probe_j_only(monkeypatch: pytest.MonkeyPatch) -> levers.Lever:
    lever = levers.Lever.of(PROBE, 339, "J only", {"j"},
                            k="the probe's own reason for K")
    monkeypatch.setitem(levers.VOCABULARY, PROBE, lever)
    return lever


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_every_family_has_a_builder_in_the_cli_and_a_target_slot() -> None:
    assert set(lever_ladder.FAMILIES) == set(emit_cli.LEVER_FAMILIES)
    assert set(lever_ladder.FAMILIES) <= set(levers.TARGET_SLOTS)
    assert {f.path for f in lever_ladder.FAMILIES.values()} == {
        lever_ladder.LADDER, lever_ladder.FOCK}
    for slot, fam in lever_ladder.FAMILIES.items():
        assert fam.slot == slot
        assert (fam.path == lever_ladder.FOCK) == (slot == "f")


def _probe(flags) -> list:
    return [lf for lf in flags if lf.lever == PROBE]


def test_the_shipped_vocabulary_s_ladders() -> None:
    # `fused` (#344) is legal on J alone, so it ladders on J alone.
    assert lever_ladder.ladders() == {
        name: tuple(s for s in lever_ladder.FAMILIES if s in lv.legal)
        for name, lv in levers.VOCABULARY.items()
        if lv.legal & set(lever_ladder.FAMILIES)}
    assert lever_ladder.ladders()["fused"] == ("j",)


def test_the_table_is_the_vocabulary_read_now(probe_j_only) -> None:
    assert lever_ladder.ladders()[PROBE] == ("j",)
    (lf,) = _probe(lever_ladder.flags())
    assert lf.flag == "--klever-probe-pq" and lf.family.slot == "j"


def test_a_lever_legal_on_every_family_gets_three_flags(probe) -> None:
    assert lever_ladder.ladders()[PROBE] == ("j", "k", "f")
    assert {lf.flag for lf in _probe(lever_ladder.flags())} == {
        "--klever-probe-pq", "--klever-probe-k", "--klever-probe-fock-l"}
    assert len({lf.dest for lf in lever_ladder.flags()}) == \
        len(lever_ladder.flags())


def test_a_lever_off_every_family_gets_no_flag(monkeypatch) -> None:
    before = lever_ladder.flags()
    monkeypatch.setitem(levers.VOCABULARY, "pairish", levers.Lever.of(
        "pairish", 339, "pair targets only", {"dq"}))
    assert "pairish" not in lever_ladder.ladders()
    assert lever_ladder.flags() == before


def test_two_levers_spelling_one_unit_suffix_are_refused(monkeypatch) -> None:
    for name in ("a-b", "a_b"):
        monkeypatch.setitem(levers.VOCABULARY, name, levers.Lever.of(
            name, 339, "collides", {"j"}))
    with pytest.raises(ValueError, match="_lv_a_b"):
        lever_ladder.ladders()


def test_the_unit_name_is_the_literal_name_without_a_kernel_lever() -> None:
    assert lever_ladder.unit_name("pq_L4_p2_q2", ()) == "pq_L4_p2_q2"
    assert lever_ladder.unit_name("pq_L4_p2_q2", ("x-y", "z")) == \
        "pq_L4_p2_q2_lv_x_y_lv_z"


# ---------------------------------------------------------------------------
# The store's one recording path
# ---------------------------------------------------------------------------


def test_a_kernel_lever_is_the_kernel_kind_and_arms_is_unmoved(probe) -> None:
    assert measured_store.kind_of(PROBE) == measured_store.KERNEL
    assert measured_store.kind_of("no-sieve") == measured_store.PLAN
    assert measured_store.kind_of("recycle") == measured_store.LOWERING
    assert PROBE not in measured_store.ARMS  # declared by the vocabulary
    with pytest.raises(ValueError, match="not a declared arm lever"):
        measured_store.kind_of("nonesuch")


def test_an_undeclared_lever_is_refused_without_its_entry() -> None:
    with pytest.raises(ValueError, match="not a declared arm lever"):
        measured_store.kind_of(PROBE)
    with pytest.raises(ValueError, match="not a declared arm lever"):
        measured_store.lever_arm(pq_spec(4), PROBE)


@pytest.mark.parametrize("cell", [
    pq_spec(4), measured_store.SUBGRID[3], measured_store.K_SUBGRID[2],
    k_spec(2), f_class_spec(0, 1, 1, 1)], ids=lambda s: s.key())
def test_lever_arm_names_every_target_s_cell_by_its_full_key(
        probe, cell) -> None:
    arm = measured_store.lever_arm(cell, PROBE)
    assert arm == replace(cell, levers=(PROBE,))
    assert arm.key() == cell.key().rsplit("/", 1)[0] + "/" + PROBE
    assert measured_store.arm_of(arm) == PROBE
    assert measured_store.arm_kind(arm) == measured_store.KERNEL


def test_lever_arm_refuses_with_the_legality_table_s_reason(
        probe_j_only) -> None:
    with pytest.raises(ValueError, match="the probe's own reason for K"):
        measured_store.lever_arm(k_spec(2), PROBE)
    with pytest.raises(ValueError, match="the fused F kernel is the HGP "
                       "engine"):  # F's default refusal (#340)
        measured_store.lever_arm(f_class_spec(0, 0, 0, 0), PROBE)
    with pytest.raises(ValueError, match="already carries no-sieve"):
        measured_store.lever_arm(pq_spec(4, ("no-sieve",)), PROBE)


def test_lever_cells_default_to_the_subgrid(probe) -> None:
    got = measured_store.lever_cells(PROBE)
    assert got == tuple(replace(s, levers=(PROBE,))
                        for s in measured_store.SUBGRID)
    assert measured_store.lever_cells(PROBE, measured_store.K_SUBGRID) == \
        tuple(replace(s, levers=(PROBE,)) for s in measured_store.K_SUBGRID)


def _row(**kw) -> measured_store.Measured:
    return measured_store.Measured(registers=96, spill_store=0, spill_load=0,
                                   **PROV, **kw)


@pytest.mark.parametrize("cell", [
    measured_store.SUBGRID[7], measured_store.K_SUBGRID[0],
    f_class_spec(0, 0, 1, 1)], ids=lambda s: s.key())
def test_a_lever_row_round_trips_byte_for_byte_with_its_kind(
        probe, cell, tmp_path: Path) -> None:
    key = MeasuredKey(measured_store.lever_arm(cell, PROBE), SM_86, "nvcc 13.0")
    store = measured_store.merge({}, key, _row())
    row = store[key.key()]
    assert row.arm_kind == measured_store.KERNEL  # merge stamps it
    p = tmp_path / "m.json"
    measured_store.save(store, p)
    doc = json.loads(p.read_text())
    assert doc["rows"][key.key()]["arm_kind"] == "kernel"
    assert doc["header"]["kernel_levers"] == measured_store.KERNEL_LEVERS_ABOUT
    back = measured_store.load(p)
    assert back == store
    q = tmp_path / "again.json"
    measured_store.save(back, q)
    assert q.read_bytes() == p.read_bytes()
    assert measured_store.rows(spec=key.spec, store=back) == store


def test_a_store_without_a_lever_row_has_no_kernel_levers_header(
        tmp_path: Path) -> None:
    key = MeasuredKey(pq_spec(3), GFX1200, "hipcc / ROCm 7.2.4")
    p = tmp_path / "m.json"
    row = measured_store.Measured(registers=1, **{
        **PROV, "arch": "gfx1200", "toolchain": "hipcc / ROCm 7.2.4"})
    measured_store.save(measured_store.merge({}, key, row), p)
    assert "kernel_levers" not in json.loads(p.read_text())["header"]


def test_a_kernel_lever_row_without_its_kind_is_refused_on_load(
        probe, tmp_path: Path) -> None:
    key = MeasuredKey(measured_store.lever_arm(pq_spec(2), PROBE), SM_86,
                      "nvcc 13.0").key()
    rec = measured_store.as_record(_row())
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": measured_store.SCHEMA, "about": "x",
                             "rows": {key: rec}}))
    with pytest.raises(ValueError, match="kernel arm and records no arm_kind"):
        measured_store.load(p)


def test_an_f_key_with_a_non_vocabulary_lever_is_still_refused(
        tmp_path: Path) -> None:
    key = "coulomb/d0/(0,0|0,0)/b1/f/no-sieve@sm_86/nvcc 13.0"
    rec = {**measured_store.as_record(_row()), "target": "f",
           "arm_kind": "plan"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": measured_store.SCHEMA, "about": "x",
                             "rows": {key: rec}}))
    with pytest.raises(ValueError, match="but a kernel-body one"):
        measured_store.load(p)


# ---------------------------------------------------------------------------
# The driver: the lever reaches the builder, and the units carry its name
# ---------------------------------------------------------------------------


def test_cell_kernels_names_a_lever_cell_s_units_as_the_ladder_does(
        probe) -> None:
    split = pq_spec(2, n_batches=3)  # L <= 2: a split cell, cheap to build
    j = measured_store.lever_arm(split, PROBE)
    lit = build_pq_kernels(split)
    assert len(lit) == 3
    got = measured.cell_kernels(j)
    assert [k.name for k in got] == [f"{k.name}_lv_probe" for k in lit]
    assert got[0] != replace(lit[0], name=got[0].name)  # the Note differs
    k = measured_store.lever_arm(k_spec(1, n_batches=3), PROBE)
    assert [x.name for x in measured.cell_kernels(k)] == [
        f"{x.name}_lv_probe" for x in build_k_kernels(k_spec(1, n_batches=3))]
    f = measured_store.lever_arm(f_class_spec(0, 0, 0, 0), PROBE)
    assert [x.name for x in measured.cell_kernels(f)] == ["f_0000_lv_probe"]
    # a literal cell, and a plan arm, keep the builder's names
    assert measured.cell_kernels(split) == lit


@pytest.mark.parametrize(("argv", "want"), [
    (["--L", "1", "2"], lambda: [pq_spec(1), pq_spec(2)]),
    (["--subgrid", "--subgrid-ls", "7", "--n-batches", "3"],
     lambda: [s for s in measured_store.SUBGRID
              if s.total_am == 7 and s.n_batches == 3]),
    (["--subgrid", "--target", "k", "--n-batches", "1"],
     lambda: [s for s in measured_store.K_SUBGRID if s.n_batches == 1]),
    (["--target", "f", "--L", "1"],
     lambda: measured.f_cells([1])),
])
def test_remeasure_with_a_kernel_lever_records_the_lever_cells(
        probe, monkeypatch, argv, want) -> None:
    seen = []
    monkeypatch.setattr(measured, "remeasure",
                        lambda names, cells, jobs, store: seen.append(cells))
    assert measured.main(["--remeasure", "--lever", PROBE, *argv]) == 0
    assert seen == [[measured_store.lever_arm(s, PROBE) for s in want()]]


def test_remeasure_refuses_an_illegal_lever_cell(probe_j_only, monkeypatch,
                                                 capsys) -> None:
    monkeypatch.setattr(measured, "remeasure", lambda *a, **k: pytest.fail())
    assert measured.main(["--remeasure", "--lever", PROBE, "--subgrid",
                          "--target", "k"]) == 2
    assert "the probe's own reason for K" in capsys.readouterr().err
    assert measured.main(["--remeasure", "--lever", "nonesuch"]) == 2
