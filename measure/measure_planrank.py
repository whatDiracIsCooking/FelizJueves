"""`PlanCost` against `Measured`: does the cheaper plan make a better kernel? (#165)

A CHECK OF MEASUREMENT ARTEFACTS, moved from test/ (`devtools/measure.sh
check`): reads the tracked rows and counts plans on the host; compiles
nothing, launches nothing. Every check that plans the tracked cells (L = 7
and up: `planrank.report`, `verdict` and `table`) is `slow`, above
`tiers.FAST_L_MAX` (`devtools/measure.sh check --full`).

WHAT IS ASSERTED, AND WHAT IS NOT. That the plan's counts predict the kernel
is a RESULT, not a target: no test here asks for agreement. Instead

- data/measured/plan_vs_kernel.json equals `planrank.report` recomputed from
  data/measured/kernels.json and the plan layer (the measure_kernelrank.py /
  measure_batchreport.py discipline): the record cannot drift from the rows;
- the verdict on the tracked rows is pinned (`PINNED`), with every
  inversion named, so a re-measurement that moves one is seen as a diff;
- an uncharacterised card gets no verdict on the tracked rows.

The join, the pair classes and the rank statistic on SYNTHETIC rows, where
the right answer is known, and the module's import discipline, stayed in
test/test_planrank.py: they are correctness.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from felizjueves.mmd.kernel import pq_spec
from felizjueves.performance import measured_store as measured
from felizjueves.performance import planrank
from felizjueves.performance.device import ARCHS, Arch, Uncharacterised, Vendor
from felizjueves.performance.planrank import (
    CONCORDANT,
    DISCORDANT,
    MIXED,
    NO_CLAIM,
    NOISE_FLOOR,
    PREDICTS,
    UNRESOLVED,
)
from felizjueves.plan.pipeline import plan

DOC = json.loads(planrank.PLANRANK.read_text())


@pytest.fixture(scope="module")
def store() -> dict:
    return measured.load()


@pytest.fixture(scope="module")
def fresh(store) -> dict:
    # plans every cell, L = 7..16: every user is `slow`
    return planrank.report(store)


def test_an_uncharacterised_card_gets_no_verdict(store) -> None:
    bare = Arch(Vendor.NVIDIA, "sm_86", 32)  # no measured ceilings
    with pytest.raises(Uncharacterised, match="not characterised"):
        planrank.verdict(store, bare)


@pytest.mark.slow  # plans every cell, L > FAST_L_MAX (tiers.py)
def test_a_characterised_card_gets_the_recorded_verdict(store) -> None:
    assert planrank.verdict(store, "sm_86") == DOC["summary"]["sm_86"]


# --- the record: recomputed --------------------------------------------------


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_the_data_file_is_the_report_recomputed(fresh) -> None:
    assert fresh == DOC, "rerun: PYTHONPATH=src uv run python -m " \
                         "felizjueves.performance.planrank --write"
    assert [(r["arch"], r["L"], r["n_batches"]) for r in DOC["rows"]] == [
        (a.name, c.total_am, c.n_batches) for a in ARCHS
        for c in planrank.CELLS]


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
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


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
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


@pytest.mark.slow  # L > FAST_L_MAX (tiers.py)
def test_reading_leaves_the_store_and_the_plan_unchanged(store) -> None:
    before = {k: replace(r) for k, r in store.items()}
    planrank.report(store)
    assert store == before
    assert plan(pq_spec(7)).cost == planrank.plan_cost(pq_spec(7))
