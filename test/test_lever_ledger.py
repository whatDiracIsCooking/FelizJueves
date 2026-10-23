"""The lever ledger's rules, without the tracked rows (#362).

No compiler, no card, no read of data/measured/:

- every vocabulary lever is in an arm on its one target, a flip is opposite
  signs on the two arches, and the F ladder is measured's;
- every `requires` entry is refused alone and builds composed;
- A KERNEL COMPOSITION IS ONE ARM (`measured_store.is_composition`), and a
  plan or lowering lever still never composes; the ledger flag is its own
  cell set.

That lever_ledger.json is `leverledger.ledger` recomputed from kernels.json,
that a missing row fails the report and a regression does not, and that the
tracked ledger rows hash today's text, are checks of measurement artefacts:
measure/measure_lever_ledger.py (`devtools/measure.sh check`).
"""

from __future__ import annotations

import pytest

from felizjueves import levers
from felizjueves.eri.fock.kernel_f import f_class_spec
from felizjueves.mmd.kernel import k_spec, pq_spec
from felizjueves.performance import leverledger, measured
from felizjueves.performance import measured_store as store
from felizjueves.performance.device import SM_86, MeasuredKey
from felizjueves.spec import Target


def test_every_vocabulary_lever_is_in_an_arm_on_its_one_target() -> None:
    arms = store.ledger_arms()
    assert set(store.ledger_levers()) == {
        lv for a in arms for lv in a.split("+")}
    # #401: a lever left out is one legal only on an engine slot (j.hgp,
    # k.hgp), which has no ledger class yet.
    for lv in set(levers.VOCABULARY) - set(store.ledger_levers()):
        assert levers.VOCABULARY[lv].legal <= set(levers.ENGINE_SLOTS), lv
    for a in arms:
        t = store.arm_target(a)
        assert t.value in set.intersection(
            *(set(levers.VOCABULARY[lv].legal) for lv in a.split("+")))
    assert "interleave+split" in arms  # #348's arm, by declaration
    assert "batched-vrr+sliced-hrr" in arms and "sliced-hrr" not in arms


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
