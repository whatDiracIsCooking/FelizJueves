"""kernelrank.py's and crossread.py's arithmetic, on SYNTHETIC inputs (#54,
#115): average-tie ranks, Spearman, which pairs a measurement orders, the
first compute-bound count each declared ridge answer gives, and that neither
module reaches a scoring floor.

Nothing here reads data/measured/. The checks that data/measured/limits.json
and crossread.json are what the tracked rows recompute to, and that every
mis-ranking over the subgrid has its record, are checks of measurement
artefacts: measure/measure_kernelrank.py, `devtools/measure.sh check`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from felizjueves.performance import crossread, kernelrank, ridge

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


def test_first_compute_bound_reads_the_declared_answers() -> None:
    def c(answer: str, count: int | None) -> ridge.Crossing:
        return ridge.Crossing("sm_86", 16, ridge.CELL, answer, count, ())
    assert crossread.first_compute_bound(c(ridge.ALREADY, None)) == 1
    assert crossread.first_compute_bound(c(ridge.CROSSED, 6)) == 6
    assert crossread.first_compute_bound(c(ridge.NEVER, None)) is None
    assert crossread.first_compute_bound(
        c(ridge.COULD_NOT_MEASURE, None)) is None


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
