"""`PlanCost` against `Measured`, the rule on SYNTHETIC rows (#165): the
join, the pair classes, the rank statistic and the verdict where the right
answer is known; a cell with no clean clock phase reported, never imputed;
an absent arm given its recorded reason; and nothing importing the reading
(its own import rule is test/test_layering.py's `plan_analysis`).

Nothing here reads data/measured/. That plan_vs_kernel.json is
`planrank.report` recomputed from the tracked rows and the plan layer, and
the verdict pinned on those rows with every inversion named, are checks of
measurement artefacts: measure/measure_planrank.py, `devtools/measure.sh
check`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from felizjueves import measured_store as measured
from felizjueves import planrank
from felizjueves.device import ARCHS, MeasuredKey
from felizjueves.kernel import pq_spec
from felizjueves.plan.cost import PlanCost
from felizjueves.planrank import (
    CONCORDANT,
    DISCORDANT,
    INVERTS,
    MIXED,
    NO_CLAIM,
    NOISE_FLOOR,
    NOT_SHOWN,
    PREDICTS,
    UNRESOLVED,
)

SRC = Path(planrank.__file__).resolve().parent


# --- one pair ----------------------------------------------------------------


def test_classify_reads_the_measurement_first() -> None:
    # a measured tie makes no claim, whatever the plan says
    assert planrank.classify(-5, 0, False) == NO_CLAIM
    assert planrank.classify(0, 0, True) == NO_CLAIM
    # the same text measured apart is the protocol's floor, not the plan
    assert planrank.classify(0, 1, True) == NOISE_FLOOR
    # a plan tie where the measurement is strict is unresolved
    assert planrank.classify(0, -3, False) == UNRESOLVED
    # cheaper and better, cheaper and worse
    assert planrank.classify(-5, -1, False) == CONCORDANT
    assert planrank.classify(+5, +1, False) == CONCORDANT
    assert planrank.classify(-5, +1, False) == DISCORDANT
    assert planrank.classify(+5, -1, False) == DISCORDANT


def test_the_verdict_rule() -> None:
    def v(c: int, d: int) -> str:
        return planrank._verdict({CONCORDANT: c, DISCORDANT: d})
    assert (v(3, 0), v(0, 2), v(1, 1), v(0, 0)) == (
        PREDICTS, INVERTS, MIXED, NOT_SHOWN)


def test_the_cells_are_the_subgrid_and_every_added_cell() -> None:
    assert planrank.CELLS[:0] == ()
    got = {(c.total_am, c.n_batches) for c in planrank.CELLS}
    assert got == {(s.total_am, s.n_batches) for s in measured.SUBGRID} | {
        (11, 1)}
    assert all(c.levers == () for c in planrank.CELLS)
    assert len(planrank.CELLS) == 21


# --- the join, on synthetic rows ---------------------------------------------

SM = ARCHS[0]
CELL = pq_spec(13, n_batches=3)
TC = "tc"


def _gate(prefix: str, util: int | None = 5) -> dict:
    return {f"{prefix}gpu_util_before": util, f"{prefix}gpu_util_after": util}


def _row(spec: measured.Spec, spill: int, *, sha: str, timed: bool = True,
         arm: tuple[float, float] = (1000.0, 10.0),
         control: tuple[float, float] | None = (1000.0, 10.0),
         util: int | None = 5) -> measured.Measured:
    raw: dict = {}
    kw: dict = {}
    if timed:
        m, i = arm
        kw = {"runtime_ns": m, "spread_ns": i, "repeats": 31}
        raw = {**_gate("wallclock:", util), **_gate("wallclock:near:", util),
               "wallclock:near:median_ns": m, "wallclock:near:q1_ns": 0,
               "wallclock:near:q3_ns": i}
        if control is not None:
            cm, ci = control
            raw |= {**_gate("wallclock:control:"),
                    **_gate("wallclock:control:near:"),
                    "wallclock:control:median_ns": cm,
                    "wallclock:control:q1_ns": 0,
                    "wallclock:control:q3_ns": ci,
                    "wallclock:control:near:median_ns": cm,
                    "wallclock:control:near:q1_ns": 0,
                    "wallclock:control:near:q3_ns": ci}
    return measured.Measured(
        registers=255, spill_store=spill, spill_load=0, toolchain=TC,
        arch=SM.name, host="h", gpu="g", date="2026-10-13",
        source_sha256=sha, raw=raw,
        arm_kind="" if not spec.levers else measured.PLAN, **kw)


def _store(**rows: measured.Measured) -> dict:
    return {MeasuredKey(planrank.variant(CELL, v.replace("_", "-")), SM,
                        TC).key(): r for v, r in rows.items()}


COUNTS = {
    "literal": PlanCost(flops=1000, max_intermediates=100, max_targets=1),
    "no-sieve": PlanCost(flops=1100, max_intermediates=120, max_targets=1),
    "tuned": PlanCost(flops=990, max_intermediates=90, max_targets=1),
}


def _costs(spec):
    return COUNTS[spec.levers[0] if spec.levers else "literal"]


def _cell(store: dict) -> dict:
    return planrank.cell_row(store, CELL, SM.name, _costs)


def test_the_join_classifies_every_pair() -> None:
    store = _store(
        literal=_row(CELL, 50, sha="L", timed=False),
        # more intermediates, more spill; more FLOPs, slower
        no_sieve=_row(planrank.variant(CELL, "no-sieve"), 60, sha="N",
                      arm=(1200.0, 10.0)),
        # fewer intermediates, MORE spill (an inversion); fewer FLOPs, faster
        tuned=_row(planrank.variant(CELL, "tuned"), 70, sha="T",
                   arm=(800.0, 10.0)))
    r = _cell(store)
    cls = {(p["a"], p["b"]): p["class"] for p in r["spill"]["pairs"]}
    assert cls == {("literal", "no-sieve"): CONCORDANT,
                   ("literal", "tuned"): DISCORDANT,
                   ("no-sieve", "tuned"): DISCORDANT}
    # rho over the cell's three variants: intermediates 100/120/90 against
    # spill 50/60/70: ranks (2, 3, 1) against (1, 2, 3)
    assert r["spill"]["spearman"] == -0.5
    for w in planrank.INPUTS:
        assert [(p["variant"], p["sign"], p["class"]) for p in r["clock"][w]] == [
            ("no-sieve", measured.SLOWER, CONCORDANT),
            ("tuned", measured.FASTER, CONCORDANT)]
    s = planrank.summary([r], SM.name)
    assert s["spill"]["verdict"] == MIXED
    assert [(e["a"], e["b"]) for e in s["spill"]["inversions"]] == [
        ("literal", "tuned"), ("no-sieve", "tuned")]
    assert all(s["clock"][w]["verdict"] == PREDICTS for w in planrank.INPUTS)
    assert not s["predicts"]
    assert "MIXED (inversions: L13 b3 literal/tuned" in s["answer"]


def test_the_same_text_measured_apart_is_the_noise_floor() -> None:
    store = _store(
        literal=_row(CELL, 50, sha="same", timed=False),
        tuned=_row(planrank.variant(CELL, "tuned"), 50, sha="same",
                   arm=(1300.0, 10.0)))
    tuned = COUNTS["tuned"]
    COUNTS["tuned"] = COUNTS["literal"]
    try:
        r = _cell(store)
    finally:
        COUNTS["tuned"] = tuned
    (p,) = r["spill"]["pairs"]
    assert p["same_text"] and p["class"] == NO_CLAIM
    (c,) = [c for c in r["clock"]["wide"] if c["clean"]]
    assert c["same_text"] and c["sign"] == measured.SLOWER
    assert c["class"] == NOISE_FLOOR
    s = planrank.summary([r], SM.name)
    assert s["clock"]["wide"]["verdict"] == NOT_SHOWN
    assert len(s["clock"]["wide"]["noise_floor_pairs"]) == 1


@pytest.mark.parametrize("why, kw", [
    ("gpu_util_before = 45% > 30%", {"util": 45}),
    ("gpu_util_before unknown", {"util": None}),
    ("no same-session control", {"control": None}),
    ("no timed row", {"timed": False}),
])
def test_a_phase_that_is_not_clean_is_reported_never_imputed(why, kw) -> None:
    store = _store(
        literal=_row(CELL, 50, sha="L", timed=False),
        tuned=_row(planrank.variant(CELL, "tuned"), 40, sha="T",
                   arm=(500.0, 10.0), **kw))
    r = _cell(store)
    for w in planrank.INPUTS:
        tuned = [c for c in r["clock"][w] if c["variant"] == "tuned"]
        assert tuned == [{"variant": "tuned", "clean": False,
                          "reason": tuned[0]["reason"]}]
        assert why in tuned[0]["reason"]
        assert "class" not in tuned[0] and "sign" not in tuned[0]
    # the spill is compiler output and needs no clean phase
    assert [p["class"] for p in r["spill"]["pairs"]] == [CONCORDANT]
    s = planrank.summary([r], SM.name)
    assert s["clock"]["wide"]["n_pairs"] == 0
    assert s["clock"]["wide"]["clean"] == 0
    assert len(s["clock"]["wide"]["not_clean"]) == 2  # no-sieve: no row


def test_an_absent_arm_gives_its_recorded_reason() -> None:
    store = _store(literal=_row(CELL, 50, sha="L", timed=False))
    r = _cell(store)
    assert r["spill"]["pairs"] == [] and r["spill"]["spearman"] is None
    reasons = {c["variant"]: c["reason"] for c in r["clock"]["wide"]}
    assert reasons["tuned"].startswith("not recorded")
    twelve = pq_spec(13, n_batches=12)
    assert planrank._absent(twelve, "tuned") == (
        "not measurable: " + measured.NOT_MEASURABLE["tuned"][
            planrank.variant(twelve, "tuned")])
    assert "not an A/B cell of the 'tuned' arm" in planrank._absent(
        pq_spec(11), "tuned")


# --- reads both sides, feeds neither ------------------------------------------


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
            out |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return out


def test_nothing_imports_the_reading() -> None:
    users = sorted(str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
                   if p.name != "planrank.py"
                   and any(n.endswith("planrank") for n in _imports(p)))
    assert not users, f"planrank feeds {users}: it may feed neither side"


def test_the_reading_reaches_no_floor_and_no_driver() -> None:
    names = _imports(SRC / "planrank.py")
    assert not any(n.startswith(("felizjueves.literature",
                                 "felizjueves.measured.",
                                 "felizjueves.dispatch")) for n in names), names
    assert "felizjueves.measured" not in names
