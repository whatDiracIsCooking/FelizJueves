"""`PlanCost` against `Measured`: does the cheaper plan make a better kernel? (#165)

FAST TIER: reads recorded rows and counts plans on the host; compiles
nothing, launches nothing.

WHAT IS ASSERTED, AND WHAT IS NOT. That the plan's counts predict the kernel
is a RESULT, not a target: no test here asks for agreement. Instead

- the join, the pair classes and the rank statistic are checked on
  SYNTHETIC rows and counts, where the right answer is known;
- data/measured/plan_vs_kernel.json equals `planrank.report` recomputed from
  data/measured/kernels.json and the plan layer (the test_kernelrank.py /
  test_batchreport.py discipline): the record cannot drift from the rows;
- the verdict on the tracked rows is pinned (`PINNED`), with every
  inversion named, so a re-measurement that moves one is seen as a diff;
- a cell with no clean clock phase is reported, never imputed; and an
  uncharacterised card gets no verdict;
- nothing imports the module (it feeds neither side). Its own import rule
  is test/test_layering.py's `plan_analysis`.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from felizjueves import measured_store as measured
from felizjueves import planrank
from felizjueves.device import ARCHS, Arch, MeasuredKey, Uncharacterised, Vendor
from felizjueves.kernel import pq_spec
from felizjueves.plan.cost import PlanCost
from felizjueves.plan.pipeline import plan
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

DOC = json.loads(planrank.PLANRANK.read_text())
SRC = Path(planrank.__file__).resolve().parent


@pytest.fixture(scope="module")
def store() -> dict:
    return measured.load()


@pytest.fixture(scope="module")
def fresh(store) -> dict:
    return planrank.report(store)


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


def test_an_uncharacterised_card_gets_no_verdict(store) -> None:
    bare = Arch(Vendor.NVIDIA, "sm_86", 32)  # no measured ceilings
    with pytest.raises(Uncharacterised, match="not characterised"):
        planrank.verdict(store, bare)
    # a characterised card gets the recorded one
    assert planrank.verdict(store, "sm_86") == DOC["summary"]["sm_86"]


# --- the record: recomputed --------------------------------------------------


def test_the_data_file_is_the_report_recomputed(fresh) -> None:
    assert fresh == DOC, "rerun: PYTHONPATH=src uv run python -m " \
                         "felizjueves.planrank --write"
    assert [(r["arch"], r["L"], r["n_batches"]) for r in DOC["rows"]] == [
        (a.name, c.total_am, c.n_batches) for a in ARCHS
        for c in planrank.CELLS]


def test_the_counts_are_the_plan_layers_own(fresh) -> None:
    for r in fresh["rows"][:len(planrank.CELLS)]:
        cell = pq_spec(r["L"], n_batches=r["n_batches"])
        for v, e in r["variants"].items():
            if e["plan"] is None:
                continue
            c = plan(planrank.variant(cell, v)).cost
            assert e["plan"] == {"flops": c.flops,
                                 "max_intermediates": c.max_intermediates,
                                 "max_targets": c.max_targets}


def test_the_spill_side_is_the_stores_and_the_clock_side_lever_ab(
        store, fresh) -> None:
    for r in fresh["rows"]:
        cell = pq_spec(r["L"], n_batches=r["n_batches"])
        for w in planrank.INPUTS:
            for c in r["clock"][w]:
                if not c["clean"]:
                    continue
                ab = measured.lever_ab(store, r["arch"],
                                       planrank.variant(cell, c["variant"]))
                assert c["sign"] == (ab.wide_sign if w == "wide"
                                     else ab.near_sign)
                assert [*c["control_ns"], *c["arm_ns"]] == list(
                    ab.wide if w == "wide" else ab.near)
                lit = r["variants"]["literal"]["measured"]["spill"]
                arm = r["variants"][c["variant"]]["measured"]["spill"]
                assert ab.spill == (lit, arm)


# --- the verdict on the tracked rows -----------------------------------------

# (card, claim) -> (concordant, discordant, unresolved, no_claim,
#                   noise_floor, verdict)
PINNED = {
    ("sm_86", "spill"): (15, 17, 4, 17, 0, MIXED),
    ("sm_86", "wide"): (4, 2, 0, 31, 0, MIXED),
    ("sm_86", "near"): (5, 1, 0, 31, 0, MIXED),
    ("gfx1200", "spill"): (17, 13, 10, 13, 0, MIXED),
    ("gfx1200", "wide"): (3, 0, 0, 33, 1, PREDICTS),
    ("gfx1200", "near"): (10, 4, 1, 17, 5, MIXED),
}

# the clock's inversions: (L, n_batches, arm) where the plan's cheaper
# variant ran slower beyond the noise floor
CLOCK_INVERSIONS = {
    ("sm_86", "wide"): [(16, 6, "no-sieve"), (16, 12, "no-sieve")],
    ("sm_86", "near"): [(16, 6, "no-sieve")],
    ("gfx1200", "wide"): [],
    ("gfx1200", "near"): [(10, 1, "no-sieve"), (10, 3, "tuned"),
                          (13, 1, "tuned"), (16, 6, "no-sieve")],
}


def _claims(s: dict) -> dict:
    return {"spill": s["spill"], **s["clock"]}


def test_the_verdict_on_the_tracked_rows() -> None:
    got = {(a, k): (c[CONCORDANT], c[DISCORDANT], c[UNRESOLVED], c[NO_CLAIM],
                    c[NOISE_FLOOR], c["verdict"])
           for a, s in DOC["summary"].items() for k, c in _claims(s).items()}
    assert got == PINNED
    inv = {(a, w): [(e["L"], e["n_batches"], e["b"])
                    for e in s["clock"][w]["inversions"]]
           for a, s in DOC["summary"].items() for w in planrank.INPUTS}
    assert inv == CLOCK_INVERSIONS
    # on neither card do the counts predict the kernel
    assert {a: s["predicts"] for a, s in DOC["summary"].items()} == {
        "sm_86": False, "gfx1200": False}
    for a, s in DOC["summary"].items():
        assert s["answer"].startswith(
            f"On {a} the plan-layer counts do not predict the kernel.")


def test_clean_clock_phases_are_counted_and_the_rest_named() -> None:
    # per card and input: 21 no-sieve pairs and 16 tuned pairs are clean;
    # tuned's four 12-batch cells are not measurable and L = 11 is not one
    # of its cells -- 37 clean, 5 not, on both cards (74 and 10 in all)
    for s in DOC["summary"].values():
        for w in planrank.INPUTS:
            c = s["clock"][w]
            assert c["clean"] == c["n_pairs"] == 37
            assert [(e["L"], e["n_batches"], e["variant"])
                    for e in c["not_clean"]] == [
                (7, 12, "tuned"), (10, 12, "tuned"), (11, 1, "tuned"),
                (13, 12, "tuned"), (16, 12, "tuned")]
            assert all(e["reason"].startswith(("not measurable",
                                               "not recorded"))
                       for e in c["not_clean"])


def test_163s_notes_read_off_the_table() -> None:
    rows = {(r["arch"], r["L"], r["n_batches"]): r for r in DOC["rows"]}

    def clock(arch, L, b, w, v="tuned"):
        (c,) = [c for c in rows[(arch, L, b)]["clock"][w] if c["variant"] == v]
        return c
    # gfx1200 L7 b3/b6: the tuned text IS the literal text, yet the rule
    # reads a loss -- the noise floor, never an inversion
    for b in (3, 6):
        c = clock("gfx1200", 7, b, "near")
        assert c["same_text"] and c["sign"] == measured.SLOWER
        assert c["class"] == NOISE_FLOOR
    # sm_86 L7 b9: the one cell whose recorded cut differs -- more FLOPs,
    # slower on wide-T: the plan saw this loss coming
    c = clock("sm_86", 7, 9, "wide")
    assert not c["same_text"] and c["plan"] == [510, 539]
    assert c["sign"] == measured.SLOWER and c["class"] == CONCORDANT
    # tuned wins at L = 16 on both cards, and every such win is a FLOP saving
    for arch in ("sm_86", "gfx1200"):
        for b in (1, 3, 6, 9):
            for w in planrank.INPUTS:
                c = clock(arch, 16, b, w)
                assert c["plan"][1] < c["plan"][0]
                assert c["class"] in (CONCORDANT, NO_CLAIM)
    # and fewer intermediates did not mean less spill there on sm_86
    (p,) = [p for p in rows[("sm_86", 16, 1)]["spill"]["pairs"]
            if (p["a"], p["b"]) == ("literal", "tuned")]
    assert p["plan"] == [1639, 1601] and p["measured"] == [317260, 319284]
    assert p["class"] == DISCORDANT


def test_the_markdown_names_every_inversion() -> None:
    md = planrank.markdown(DOC)
    for s in DOC["summary"].values():
        for c in _claims(s).values():
            for e in c["inversions"]:
                assert f"L{e['L']} b{e['n_batches']} {e['a']}/{e['b']}" in md
    assert md.count("\n") == 1 + 3 * len(DOC["summary"])


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


def test_reading_leaves_the_store_and_the_plan_unchanged(store) -> None:
    before = {k: replace(r) for k, r in store.items()}
    planrank.report(store)
    assert store == before
    assert plan(pq_spec(7)).cost == planrank.plan_cost(pq_spec(7))
