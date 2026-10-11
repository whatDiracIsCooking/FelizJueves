"""`KernelCost` against `Measured` (kernelrank.py), and data/measured/limits.json.

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

from felizjueves import kernelrank
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


def test_every_limit_cell_is_recomputed_and_is_still_a_miss(doc, store) -> None:
    for limit in doc["limits"]:
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
    return {_key(e) for lim in doc["limits"] for e in lim["asserted_value"]["cells"]}


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


# --- the separation ------------------------------------------------------------


def test_kernelrank_reaches_no_floor() -> None:
    # The rule lower.py states: nothing in KernelCost feeds a literature floor.
    # Its import half (plan/ and literature/ never import lower or this
    # module) is test_layering.py's; here, that this module imports neither.
    src = Path(kernelrank.__file__).read_text()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    assert not any(n.startswith(("felizjueves.plan", "felizjueves.literature"))
                   for n in names), names
