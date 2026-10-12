"""`KernelCost` against `Measured` (kernelrank.py), and data/measured/limits.json;
and the cross-read of the model, the clock and the counters (crossread.py, #115).

FAST TIER: reads recorded rows and builds kernels as IR; compiles nothing,
launches nothing. The whole module costs a few seconds (L = 16, b = 9 is the
largest IR the liveness sweep walks).

WHAT IS ASSERTED, AND WHAT IS NOT. The rank claim is a RESULT, not a target:
no test here asserts that the model ranks correctly. Instead

- every number data/measured/limits.json records -- the rank claim, the
  wall-clock claim, and each StructuralLimit's cells -- is recomputed from
  the store and `lower.kernel_cost` over the batch counts THE RECORD names,
  and must match: a record cannot drift from what the code and rows say;
- every mis-ranking over the CURRENT `measured.SUBGRID` (a discordant pair,
  or a spill win that is not the fastest) must be covered by a record of the
  same (arch, L, metric, aggregation) or (arch, L, input). A new one FAILS
  until it is written down with its mechanism -- that is the mechanism (#54).

WIDENING SUBGRID (#60's n_batches = 12). The recorded checks restrict to the
recorded batch counts, so they hold unchanged. The coverage checks run over
the whole SUBGRID: a b = 12 cell that creates a discordant pair in an
(arch, L, ...) no record covers fails here and needs a limits.json row (a
data edit, not a test edit); one inside an already-recorded group is covered.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from felizjueves import batchreport, crossread, kernelrank, ridge
from felizjueves import measured_store as measured
from felizjueves.device import ARCHS
from felizjueves.literature.limits import StructuralLimit, load_limits

LIMITS = Path(__file__).resolve().parents[1] / "data" / "measured" / "limits.json"


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(LIMITS.read_text())


@pytest.fixture(scope="module")
def store() -> dict:
    return measured.load()


# --- the arithmetic ----------------------------------------------------------


def test_ranks_average_ties() -> None:
    assert kernelrank.ranks([5, 0, 0, 0]) == [4.0, 2.0, 2.0, 2.0]
    assert kernelrank.ranks([3, 1, 2]) == [3.0, 1.0, 2.0]


def test_spearman() -> None:
    assert kernelrank.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert kernelrank.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert kernelrank.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    with pytest.raises(ValueError):
        kernelrank.spearman([1], [1])


def test_pairs_count_only_what_the_measurement_orders() -> None:
    # measured ties (b 3, 6, 9 all zero) make no claim; a model tie where the
    # measurement is strict is unresolved, never concordant.
    assert kernelrank.pairs([50, 0, 0, 0], [508, 0, 0, 0]) == {
        "concordant": 3, "discordant": 0, "unresolved": 0}
    assert kernelrank.pairs([9, 0, 0], [9, 3, 0]) == {
        "concordant": 2, "discordant": 0, "unresolved": 1}
    assert kernelrank.pairs([3, 2, 1], [1, 2, 3])["discordant"] == 3


# --- the declared budget -----------------------------------------------------


def test_the_budget_is_half_the_ceiling_each_compiler_spills_at(store) -> None:
    # The declaration is checkable against the rows: every spilling SUBGRID
    # row on an arch reports the ceiling as its register count (max over its
    # batch kernels), and none exceeds it.
    assert set(kernelrank.BUDGET) == {a.name for a in ARCHS} == set(
        kernelrank.CEILING)
    for arch in ARCHS:
        regs = [r.registers for s in measured.SUBGRID
                for r in measured.rows(s, arch, store=store).values()]
        spilling = [r.registers for s in measured.SUBGRID
                    for r in measured.rows(s, arch, store=store).values()
                    if r.spill_store]
        assert regs and spilling
        assert max(regs) == kernelrank.CEILING[arch.name]
        assert set(spilling) == {kernelrank.CEILING[arch.name]}, arch.name
        assert kernelrank.BUDGET[arch.name] == kernelrank.CEILING[arch.name] // 2


def test_measured_aggregation_matches_the_stores_own(store) -> None:
    # `sum` is the row's summed field; `max` from raw never exceeds it, and a
    # one-batch cell's two agree.
    for arch in ARCHS:
        for spec in measured.SUBGRID:
            (row,) = measured.rows(spec, arch, store=store).values()
            total = kernelrank.measured_spill(row, "sum")
            worst = kernelrank.measured_spill(row, "max")
            assert total == row.spill_store + row.spill_load
            assert worst <= total
            if spec.n_batches == 1:
                assert worst == total


# --- the record: recomputed, number for number -------------------------------


def test_the_limits_file_loads_as_structural_limits(doc) -> None:
    limits = load_limits(LIMITS)
    assert limits and all(isinstance(x, StructuralLimit) for x in limits)
    assert len(limits) == len(doc["limits"])


def _key(e: dict) -> tuple:
    return (e["arch"], e["L"], e.get("metric"), e.get("aggregation"),
            e.get("input"))


def _check(recorded: list[dict], fresh: list[dict]) -> None:
    by = {_key(e): e for e in fresh}
    for e in recorded:
        got = by[_key(e)]
        for k, v in e.items():
            assert got[k] == v, (_key(e), k, v, got[k])


def test_the_rank_claim_states_its_budget_and_n(doc) -> None:
    claim = doc["rank_claim"]
    assert claim["registers"] == kernelrank.BUDGET
    assert claim["ceiling"] == kernelrank.CEILING
    assert claim["n"] == len(claim["n_batches"]) >= 2
    subgrid_b = {s.n_batches for s in measured.SUBGRID}
    assert set(claim["n_batches"]) <= subgrid_b
    assert {e["registers"] for e in claim["cells"]} <= set(
        kernelrank.BUDGET.values())


def test_the_recorded_rank_claim_is_what_the_code_computes(doc, store) -> None:
    claim = doc["rank_claim"]
    fresh = kernelrank.rank_claim(store, claim["n_batches"])
    assert {_key(e) for e in claim["cells"]} == {_key(e) for e in fresh}
    _check(claim["cells"], fresh)


def test_the_recorded_wallclock_claim_is_what_the_rows_say(doc, store) -> None:
    claim = doc["wallclock_claim"]
    bs = doc["rank_claim"]["n_batches"]
    fresh = kernelrank.wallclock_claim(store, bs)
    assert {_key(e) for e in claim["cells"]} == {_key(e) for e in fresh}
    _check(claim["cells"], fresh)


def _is_cross_read(limit: dict) -> bool:
    return bool(limit["asserted_value"].get("cross_read"))


def test_every_limit_cell_is_recomputed_and_is_still_a_miss(doc, store) -> None:
    # the cross-read's records (#115) are checked in their own section below
    for limit in (x for x in doc["limits"] if not _is_cross_read(x)):
        value = limit["asserted_value"]
        bs = value["n_batches"]
        cells = value["cells"]
        assert cells, limit["claim"]
        if "registers" in value:
            fresh = kernelrank.rank_claim(store, bs)
            _check(cells, fresh)
            assert all(e["discordant"] > 0 for e in cells), limit["claim"]
        else:
            fresh = kernelrank.wallclock_claim(store, bs)
            _check(cells, fresh)
            assert not any(e["spill_win_is_fastest"] for e in cells)


# --- coverage over the CURRENT subgrid ---------------------------------------


def _covered(doc: dict) -> set[tuple]:
    return {_key(e) for lim in doc["limits"] if not _is_cross_read(lim)
            for e in lim["asserted_value"]["cells"]}


def test_every_mis_ranking_over_subgrid_has_a_record(doc, store) -> None:
    covered = _covered(doc)
    missing = [_key(e) for e in kernelrank.rank_claim(store)
               if e["discordant"] and _key(e) not in covered]
    assert not missing, f"mis-rankings with no StructuralLimit: {missing}"


def test_every_spill_win_that_is_not_fastest_has_a_record(doc, store) -> None:
    covered = _covered(doc)
    missing = [_key(e) for e in kernelrank.wallclock_claim(store)
               if not e["spill_win_is_fastest"] and _key(e) not in covered]
    assert not missing, f"spill wins not fastest, with no record: {missing}"


# --- the cross-read: three instruments, one subgrid (#115) -------------------
#
# crossread.py reads the model (above), the clock (batchreport, #56) and the
# counters' ridge crossing (ridge, #114) against each other per (L, arch).
# As above, agreement is not a target: the table is recomputed, the choice is
# shown to be the clock's alone, and every disagreement must be a record.

CROSSREAD_DOC = json.loads(crossread.CROSSREAD.read_text())
J_ENGINE = (measured.REPO / "docs" / "implementation" / "j_engine.md").read_text()


@pytest.fixture(scope="module")
def counters() -> dict:
    return measured.load_counters()


@pytest.fixture(scope="module")
def cross(store, counters) -> dict:
    return crossread.report(store, counters)


def _row(rows: list[dict], arch: str, L: int) -> dict:
    (r,) = [r for r in rows if r["arch"] == arch and r["L"] == L]
    return r


def test_the_crossread_file_is_the_table_recomputed(cross) -> None:
    assert cross == CROSSREAD_DOC, (
        "rerun `python -m felizjueves.crossread --write`")
    assert [(r["arch"], r["L"]) for r in cross["rows"]] == [
        (a.name, L) for a in ARCHS for L in measured.SUBGRID_LS]


def test_the_j_engine_table_is_the_data_verbatim() -> None:
    assert crossread.markdown(CROSSREAD_DOC) in J_ENGINE, (
        "paste `python -m felizjueves.crossread --markdown` into j_engine.md")


def test_first_compute_bound_reads_the_declared_answers() -> None:
    def c(answer: str, count: int | None) -> ridge.Crossing:
        return ridge.Crossing("sm_86", 16, ridge.CELL, answer, count, ())
    assert crossread.first_compute_bound(c(ridge.ALREADY, None)) == 1
    assert crossread.first_compute_bound(c(ridge.CROSSED, 6)) == 6
    assert crossread.first_compute_bound(c(ridge.NEVER, None)) is None
    assert crossread.first_compute_bound(
        c(ridge.COULD_NOT_MEASURE, None)) is None


def test_measured_outranks_cost_the_choice_is_the_clocks_alone(
        store, counters, cross) -> None:
    # The choice is the clock's best on each input, everywhere ...
    for r in cross["rows"]:
        for w in crossread.INPUTS:
            best = batchreport.recommend(store, r["arch"], r["L"], w).best
            assert r["choice"][w] == best.n_batches == r["clock"][w]["n_batches"]
    # ... and no model can move it: reverse the model's order and every pick
    # changes somewhere while no choice does.
    def reverse(spec, registers):
        return -crossread.kernelcost(spec, registers)
    flipped = crossread.table(store, counters, model=reverse)
    assert [r["choice"] for r in flipped] == [r["choice"] for r in cross["rows"]]
    assert [r["model"]["pick"] for r in flipped] != [
        r["model"]["pick"] for r in cross["rows"]]
    # where the model disagrees with the clock, the clock's count is chosen
    for r in cross["rows"]:
        for w, ok in r["agree"]["model_clock"].items():
            if not ok:
                assert r["choice"][w] not in r["model"]["pick"]


def test_an_unmeasured_card_has_null_crossing_flags_never_false(cross) -> None:
    for L in measured.SUBGRID_LS:
        r = _row(cross["rows"], "gfx1200", L)
        assert r["counters"] == {
            "could_not_measure": measured.UNMEASURED["gfx1200"]}
        ag = r["agree"]
        assert all(v is None for by in ag["clock_crossing"].values()
                   for v in by.values())
        assert all(v is None for v in ag["model_crossing"].values())
        # the model and the clock still meet on this card
        assert all(isinstance(v, bool) for v in ag["model_clock"].values())


def _cross_records(doc: dict) -> list[dict]:
    return [x for x in doc["limits"] if _is_cross_read(x)]


def _canon(d: dict) -> str:
    return json.dumps(d, sort_keys=True)


def test_every_cross_read_disagreement_has_a_record_and_is_still_one(
        doc, cross) -> None:
    fresh = {_canon(d) for d in crossread.disagreements(cross["rows"])}
    assert fresh, "the cross-read found no disagreement: re-read the records"
    recs = _cross_records(doc)
    assert recs
    recorded = set()
    for lim in recs:
        v = lim["asserted_value"]
        assert v["pair"] in crossread.PAIRS
        assert lim["mechanism"] and lim["evidence"] and lim["claim"]
        assert v["cells"], lim["claim"]
        for cell in v["cells"]:
            assert cell["pair"] == v["pair"]
            if "input" in v:
                assert cell["input"] == v["input"]
            assert _canon(cell) in fresh, ("no longer a disagreement", cell)
            recorded.add(_canon(cell))
    missing = sorted(fresh - recorded)
    assert not missing, f"cross-read disagreements with no record: {missing}"


def test_the_cross_read_on_the_tracked_rows(cross, store, counters) -> None:
    """The outcome the records and j_engine.md quote, recomputed. A
    re-measurement that moves one is news, not a test to edit."""
    rows = cross["rows"]
    sm = {L: _row(rows, "sm_86", L) for L in measured.SUBGRID_LS}
    first = {L: {v: sm[L]["counters"][v]["first_compute_bound"]
                 for v in ridge.VARIANTS} for L in sm}
    assert first == {L: dict.fromkeys(ridge.VARIANTS, 1 if L < 16 else 3)
                     for L in sm}
    # the clock: wide-T picks 1 everywhere; near-T splits from L 13
    assert {(r["arch"], r["L"]): (r["choice"]["wide"], r["choice"]["near"])
            for r in rows} == {
        ("sm_86", 7): (1, 1), ("sm_86", 10): (1, 1), ("sm_86", 13): (1, 3),
        ("sm_86", 16): (1, 9), ("gfx1200", 7): (1, 1), ("gfx1200", 10): (1, 3),
        ("gfx1200", 13): (1, 6), ("gfx1200", 16): (1, 12)}
    # clock = first compute-bound: only where nothing crosses
    for v in ridge.VARIANTS:
        assert {L: sm[L]["agree"]["clock_crossing"][v] for L in sm} == {
            7: {"wide": True, "near": True}, 10: {"wide": True, "near": True},
            13: {"wide": True, "near": False},
            16: {"wide": False, "near": False}}
    # the model never picks 1, and agrees with the clock once, at the edge
    assert all(1 not in r["model"]["pick"] for r in rows)
    agree = [(r["arch"], r["L"], w) for r in rows
             for w, ok in r["agree"]["model_clock"].items() if ok]
    assert agree == [("gfx1200", 16, "near")]
    assert _row(rows, "gfx1200", 16)["model"]["pick"] == [max(
        crossread._specs(16), key=lambda s: s.n_batches).n_batches]
    # the numbers the wide-T clock_crossing mechanism quotes (L 16, sm_86)
    c1, c3 = (measured.cell_counters(counters,
                                     ridge.cell_key(counters, "sm_86", 16, b))
              for b in (1, 3))
    assert round(c3.fp64_flops / c1.fp64_flops, 2) == 1.55
    assert round(c1.dram_bytes / c3.dram_bytes, 2) == 2.47
    wide = batchreport.recommend(store, "sm_86", 16, "wide")
    t = {x.n_batches: x.median_ns for x in wide.timings}
    assert (round(t[1] / 1e6, 3), round(t[3] / 1e6, 3)) == (156.851, 289.301)
    assert round(t[3] / t[1], 2) == 1.84
    near13 = {x.n_batches: x.median_ns
              for x in batchreport.recommend(store, "sm_86", 13, "near").timings}
    near16 = {x.n_batches: x.median_ns
              for x in batchreport.recommend(store, "sm_86", 16, "near").timings}
    assert (round(near13[1] / 1e6, 3), round(near13[3] / 1e6, 3)) == (
        32.598, 29.478)
    assert (round(near16[3] / 1e6, 3), round(near16[9] / 1e6, 3),
            round(near16[12] / 1e6, 3)) == (117.029, 98.776, 103.055)
    spill = {e["L"]: e["measured"] for e in kernelrank.rank_claim(store)
             if e["arch"] == "sm_86" and e["metric"] == "traffic"
             and e["aggregation"] == "sum"}
    assert spill[13][:2] == [86632, 59444]
    assert (spill[16][1], spill[16][3]) == (291916, 181224)
    assert [sm[L]["model"]["values"] for L in sm] == [
        [50, 0, 0, 0, 0], [2897, 0, 0, 0, 0], [11798, 5300, 0, 0, 0],
        [32827, 24651, 10912, 1845, 1574]]
    # and the records quote them
    text = " ".join(lim["mechanism"] + " " + lim["evidence"]
                    for lim in _cross_records(json.loads(LIMITS.read_text())))
    for s in ("1.55x (22.3 G -> 34.6 G)", "2.47x (56.3 GB -> 22.8 GB)",
              "1.84x (156.851 -> 289.301 ms)", "86632 -> 59444",
              "291916 -> 181224", "32.598 ms", "29.478 ms", "98.776 ms",
              "(50, 2897 and 11798 fp64 words)", "(24651 -> 1574)",
              "15 of 16 (L, arch, input) disagree"):
        assert s in text, s


# --- the separation ------------------------------------------------------------


@pytest.mark.parametrize("module", [kernelrank, crossread],
                         ids=lambda m: m.__name__)
def test_kernelrank_reaches_no_floor(module) -> None:
    # The rule lower.py states: nothing in KernelCost feeds a literature floor.
    # Its import half (plan/ and literature/ never import lower or these
    # modules) is test_layering.py's; here, that these modules import neither.
    src = Path(module.__file__).read_text()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    assert not any(n.startswith(("felizjueves.plan", "felizjueves.literature"))
                   for n in names), names
