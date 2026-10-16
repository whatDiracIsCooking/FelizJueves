"""The lever ledger: per-class register and spill deltas, both cards (#362).

All fast-tier (no compiler, no card):

- THE DATA IS THE DERIVATION. data/measured/lever_ledger.json equals
  `leverledger.ledger` recomputed from data/measured/kernels.json.
- A MISSING ROW FAILS, A REGRESSION DOES NOT. A vocabulary lever with no rows
  (a PROBE lever declared here) fails the report, naming it; a row whose
  delta is a regression is reported with its sign and fails nothing. There
  is no threshold to test against: the report has none (the guardrail).
- THE ROWS ARE TODAY'S TEXT. Every tracked ledger row's `source_sha256` is
  what the generator emits today (emission only, no compiler), so `measured
  --remeasure --ledger` on this tree recompiles nothing.
- A KERNEL COMPOSITION IS ONE ARM (`measured_store.is_composition`), and a
  plan or lowering lever still never composes.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from felizjueves import leverledger, levers, measured
from felizjueves import measured_store as store
from felizjueves.device import ARCHS, SM_86, MeasuredKey
from felizjueves.kernel import k_spec, pq_spec
from felizjueves.kernel_f import f_class_spec
from felizjueves.spec import Target

ROWS = store.load()
DOC = json.loads(leverledger.LEDGER.read_text())
PROBE = "probe"


def test_the_data_file_is_the_ledger_recomputed_from_the_store() -> None:
    assert DOC == leverledger.ledger(ROWS), (
        "rerun: PYTHONPATH=src uv run python -m felizjueves.leverledger "
        "--write (or measured --remeasure --ledger, which records the rows)")


def test_one_row_per_class_arm_and_arch_none_filtered() -> None:
    want = {(c.key(), arm, a.name) for arm in store.ledger_arms()
            for c in store.ledger_classes(store.arm_target(arm))
            for a in ARCHS}
    got = [(r["class"], r["arm"], r["arch"]) for r in DOC["rows"]]
    assert len(got) == len(set(got)) and set(got) == want
    # regressions and nulls are rows like any other
    signs = {leverledger.sign(r["delta"]["registers"]) for r in DOC["rows"]}
    assert {"+", "-", "0"} <= signs


def test_every_vocabulary_lever_is_in_an_arm_on_its_one_target() -> None:
    arms = store.ledger_arms()
    assert set(levers.VOCABULARY) == {lv for a in arms for lv in a.split("+")}
    for a in arms:
        t = store.arm_target(a)
        assert t.value in set.intersection(
            *(set(levers.VOCABULARY[lv].legal) for lv in a.split("+")))
    assert "interleave+split" in arms  # #348's arm, by declaration
    assert "batched-vrr+sliced-hrr" in arms and "sliced-hrr" not in arms


def test_a_vocabulary_lever_with_no_rows_fails_the_report(monkeypatch) -> None:
    monkeypatch.setitem(levers.VOCABULARY, PROBE, levers.Lever.of(
        PROBE, 362, "the ledger's witness", {"j"}))
    assert PROBE in store.ledger_arms()
    with pytest.raises(leverledger.MissingRows, match=PROBE) as e:
        leverledger.ledger(ROWS)
    # every class on both arches is named, and nothing else is missing
    lines = str(e.value).splitlines()[1:]
    assert len(lines) == len(store.ledger_classes(Target.J)) * len(ARCHS)
    assert all(ln.startswith(PROBE + " ") for ln in lines)


def test_a_missing_literal_row_fails_the_report() -> None:
    cls = pq_spec(2)
    rows = {k: r for k, r in ROWS.items()
            if not k.startswith(cls.key() + "@sm_86/")}
    with pytest.raises(leverledger.MissingRows, match="no literal row"):
        leverledger.ledger(rows)


def test_a_regression_is_a_row_and_fails_nothing() -> None:
    spec = store.lever_arm(k_spec(3), "split")
    key = next(k for k in ROWS if k.startswith(spec.key() + "@sm_86/"))
    lit = next(r for k, r in ROWS.items()
               if k.startswith(k_spec(3).key() + "@sm_86/"))
    bumped = {**ROWS, key: replace(ROWS[key], registers=lit.registers + 99,
                                   spill_store=lit.spill_store + 1000)}
    doc = leverledger.ledger(bumped)
    (row,) = [r for r in doc["rows"] if r["key"] == key]
    assert row["delta"]["registers"] == 99
    assert row["delta"]["spill_store"] == 1000
    assert row["baseline"].startswith(k_spec(3).key() + "@sm_86/")


def test_a_composition_is_read_against_its_sub_arms() -> None:
    for r in DOC["rows"]:
        parts = set(r["arm"].split("+"))
        subs = {a for a in store.ledger_arms() if set(a.split("+")) < parts}
        assert set(r.get("vs", {})) == subs, r["key"]
    (r,) = [r for r in DOC["rows"] if r["arm"] == "interleave+split"
            and r["class"] == k_spec(2).key() and r["arch"] == "sm_86"]
    assert set(r["vs"]) == {"interleave", "split"}


def test_a_flip_is_opposite_signs_on_the_two_arches() -> None:
    rows = [{"class": "c", "arm": "a", "arch": a,
             "delta": dict.fromkeys(leverledger.FIELDS, d)}
            for a, d in (("sm_86", 2), ("gfx1200", -1))]
    assert {f["field"] for f in leverledger.flips(rows)} == set(
        leverledger.FIELDS)
    rows[1]["delta"] = dict.fromkeys(leverledger.FIELDS, 0)
    assert leverledger.flips(rows) == []


def test_the_f_ladder_is_measured_s() -> None:
    assert store._F_LADDER == measured.F_CLASSES
    assert store.ledger_classes(Target.F) == tuple(
        f_class_spec(*c) for c in measured.F_CLASSES
        if sum(c) <= store.LEDGER_MAX_L)


def test_every_requires_entry_is_refused_alone_and_builds_composed() -> None:
    for lv, needs in store.REQUIRES.items():
        if not levers.is_kernel_lever(lv):
            continue  # declared ahead (#393-#397), inert until it lands
        cls = store.ledger_classes(store.arm_target(lv))[0]
        with pytest.raises(ValueError):
            measured.cell_kernels(store.lever_arm(cls, lv))
        assert measured.cell_kernels(
            store.lever_arm(cls, "+".join(sorted((lv, *needs)))))


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_the_ledger_rows_hash_the_text_the_generator_writes_today(
        backend: str) -> None:
    b = measured.BACKENDS[backend]
    tool = {r.arch: r.toolchain for r in ROWS.values()}[b.arch.name]
    assert measured.ledger_todo(b, tool, ROWS) == []


def test_a_kernel_composition_is_one_arm() -> None:
    spec = store.lever_arm(k_spec(2), "split+interleave")
    assert spec.levers == ("interleave", "split")
    assert store.arm_of(spec) == "interleave+split"
    assert store.arm_kind(spec) == store.KERNEL
    row = store.Measured(registers=1, toolchain="nvcc 0.0", arch="sm_86",
                         host="h", gpu="g", date="d")
    got = store.merge({}, MeasuredKey(spec, SM_86, "nvcc 0.0"), row)
    (r,) = got.values()
    assert r.arm_kind == store.KERNEL and r.target == "k"
    with pytest.raises(ValueError, match="arms at once"):
        store.arm_of(pq_spec(4, ("fused", "no-sieve")))


def test_the_ledger_flag_is_its_own_cell_set() -> None:
    for argv in (["--ledger"], ["--remeasure", "--ledger", "--subgrid"],
                 ["--remeasure", "--ledger", "--lever", "fused"],
                 ["--remeasure", "--ledger", "--asym"]):
        assert measured.main(argv) == 2, argv
