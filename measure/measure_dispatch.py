"""`dispatch` on the TRACKED rows (#145): the resolver recomputes
data/measured/batch_recommendation.json from data/measured/kernels.json --
the measure_batchreport.py pattern: the answer is never re-typed here --, the
default store is the tracked one, the tracked answer depends on the input and
the cards disagree, and both tracked cards are characterised (so the
refusal test/test_dispatch.py holds on synthetic rows is a distinct branch,
not every card's fate).

A CHECK OF MEASUREMENT ARTEFACTS, moved from test/test_dispatch.py, which
keeps the rule on synthetic rows: `devtools/measure.sh check` (no compiler,
no card).
"""

from __future__ import annotations

import json

import pytest
from test_dispatch import (
    _split,
)

from felizjueves.performance import batchreport, dispatch
from felizjueves.performance import measured_store as measured
from felizjueves.performance.device import (
    ARCHS,
    GFX1200,
    SM_86,
    Uncharacterised,
    by_name,
)

ROWS = measured.load()
DOC = json.loads(batchreport.RECOMMENDATION.read_text())


# --- refusal: an uncharacterised card gets no choice --------------------------


def test_the_tracked_cards_are_characterised() -> None:
    # so the refusal below is a distinct branch, not every card's fate
    assert all(a.characterised for a in ARCHS)


def test_a_characterised_card_nobody_timed_gets_no_choice() -> None:
    other = {k: r for k, r in ROWS.items() if r.arch != GFX1200.name}
    with pytest.raises(Uncharacterised, match="no wall-clock row"):
        dispatch.Resolver(GFX1200, "near", store=other)
    with pytest.raises(Uncharacterised):
        dispatch.choose(0, 0, GFX1200, "near", store={})


@pytest.mark.parametrize("which", dispatch.INPUTS)
def test_the_resolver_reproduces_the_recommendation_file(which) -> None:
    by_cell = {(c["arch"], c["L"]): c[which] for c in DOC["cells"]}
    assert set(by_cell) == {(a.name, L) for a in ARCHS
                            for L in measured.SUBGRID_LS}
    for name in {a for a, _ in by_cell}:
        arch = by_name(name)
        tab = dispatch.table(arch, which, store=ROWS)
        res = dispatch.Resolver(arch, which, store=ROWS)
        assert set(tab) == set(dispatch.measured_classes())
        for (lp, lq), c in tab.items():
            want = by_cell[(name, lp + lq)]
            assert c.measured
            assert (c.n_batches, c.median_ns, c.iqr_ns, c.repeats,
                    c.runner_up, c.margin_ns, c.resolved,
                    list(c.within_spread)) == (
                want["n_batches"], want["median_ns"], want["iqr_ns"],
                want["repeats"], want["runner_up"], want["margin_ns"],
                want["resolved"], want["within_spread"])
            assert res(lp, lq) == c.n_batches and res.choice(lp, lq) == c


def test_the_default_store_is_the_tracked_one() -> None:
    lp, lq = _split(16)
    assert (dispatch.choose(lp, lq, SM_86, "near")
            == dispatch.choose(lp, lq, SM_86, "near", store=ROWS))


def test_the_tracked_answer_is_input_dependent_and_the_cards_disagree() -> None:
    # The hazard the explicit axis exists for, read off the resolver.
    lp, lq = _split(16)
    wide = {a.name: dispatch.choose(lp, lq, a, "wide", store=ROWS).n_batches
            for a in ARCHS}
    near = {a.name: dispatch.choose(lp, lq, a, "near", store=ROWS).n_batches
            for a in ARCHS}
    assert set(wide.values()) == {1}
    assert near[SM_86.name] != near[GFX1200.name]
    assert all(n > 1 for n in near.values())
