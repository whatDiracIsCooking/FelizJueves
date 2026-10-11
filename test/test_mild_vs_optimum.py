"""MILD against the optimum: [JEngine, v2, §4.1]'s trade-off, measured here (#111).

§4.1 puts MILD 1-2% behind Johnson et al.'s near-optimal tree search on FLOPs
and 2-6% ahead on intermediates. That search is not here (limits.json says
so); plan/optimum.py (#106) is, so this module MEASURES the gap instead of
citing it, and data/literature/jengine/mild_vs_optimum.json is the record.

THE REFERENCE POINT, one for every cell: the frontier point with the FEWEST
FLOPs among those with EQUAL-OR-FEWER intermediates than MILD's (on a
frontier, the one with the most intermediates not exceeding MILD's). The exact
frontier always holds one, since MILD's own scheme is in the space. A gap is
signed, per axis, `100 * (MILD - reference) / reference`: positive means MILD
spends MORE (is behind), negative means it spends less (is ahead).

THE TWO RANGES.

- EXACT, L = 0..EXACT_MAX_L: the gap against `optimum(L)`. The claim test
  recomputes it (about 2 s) and pins it exactly, inside a stated band, so a
  worse MILD fails it (`test_a_degraded_mild_leaves_the_band`).
- BEAM, L = 6..16 (the published grid is 7..16; 6 is the hole between): the
  gap against `beam(L, BEAM_WIDTH)`, an achievable scheme, so an UPPER bound on
  the optimum and a LOWER bound on MILD's gap. It can prove MILD far from
  optimal and never close. A lower bound is recorded only where the beam's
  reference strictly dominates MILD; elsewhere the beam proves nothing and the
  row says so. L <= 10 is recomputed in the fast tier, L = 11..16 (about a
  minute) is `slow`.

Both MILD as built (`dag.build`, no sieve: the heuristic §4.1 is about) and
MILD followed by the sieve (`sieve.plan`, the pipeline's literal plan, seeded
in hermite_targets order) are measured; the claim is about the first.

Fast tier: no compiler, no card. `plan/cost.py` is read, never changed.
Rewrite the data with
`PYTHONPATH=src uv run python test/test_mild_vs_optimum.py --write`
(about 70 s; the slow test checks every cell it writes).
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from math import inf
from pathlib import Path
from unittest import mock

import pytest

import felizjueves.plan.dag as dag
from felizjueves.literature.claims import ClaimStatus, load_claims
from felizjueves.literature.limits import load_limits
from felizjueves.plan import optimum as opt
from felizjueves.plan.cost import cost
from felizjueves.plan.sieve import plan as sieved
from felizjueves.spec import Operator, Spec, Target

ROOT = Path(__file__).resolve().parents[1]
JENGINE = ROOT / "data" / "literature" / "jengine"
DATA = JENGINE / "mild_vs_optimum.json"
CLAIM = "jengine/mild-vs-exact-tradeoff"

EXACT_LS = range(opt.EXACT_MAX_L + 1)
BEAM_WIDTH = 16
BEAM_LS = range(opt.EXACT_MAX_L + 1, 17)
FAST_BEAM_LS = range(opt.EXACT_MAX_L + 1, 11)
SLOW_BEAM_LS = range(11, 17)
PUBLISHED_GRID = range(7, 17)

# The band the claim test holds MILD to, stated before the degraded variants
# were tried: no gap on either axis through L = 4, and at every exact L at
# most 1% behind on FLOPs and 7% behind on intermediates.
BAND = {"zero_through_L": 4, "flops_percent": [0.0, 1.0],
        "intermediates_percent": [0.0, 7.0]}

AXES = "xyz"


def _spec(L: int) -> Spec:
    return Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J)


def _point(c) -> list[int]:
    return [c.max_intermediates, c.flops]


def mild_point(L: int, direction=None) -> list[int]:
    """MILD's one-batch plan as built, no sieve. `direction` replaces
    `mild_direction` (only to show that a worse rule leaves the band)."""
    patch = (nullcontext() if direction is None
             else mock.patch.object(dag, "mild_direction", direction))
    with patch:
        return _point(cost(dag.build(_spec(L))))


def sieved_point(L: int) -> list[int]:
    return _point(cost(sieved(_spec(L))))


def reference(frontier, point) -> list[int] | None:
    """The fewest-FLOPs frontier point at equal-or-fewer intermediates."""
    ok = [p for p in frontier if p[0] <= point[0]]
    return list(min(ok, key=lambda p: (p[1], -p[0]))) if ok else None


def pct(a: int, b: int) -> float:
    if b == 0:
        if a != 0:
            raise ValueError(f"gap of {a} against a zero reference")
        return 0.0
    return round(100 * (a - b) / b, 3)


def gap(point, ref) -> dict[str, float]:
    return {"intermediates": pct(point[0], ref[0]),
            "flops": pct(point[1], ref[1])}


def direction_words(g: dict[str, float]) -> str:
    """The sign of each axis, in words: behind = MILD spends more."""
    def word(v: float) -> str:
        return "behind" if v > 0 else "ahead" if v < 0 else "level"
    i, f = word(g["intermediates"]), word(g["flops"])
    if i == f:
        return "equal to the reference" if i == "level" else f"{i} on both axes"
    return f"{f} on FLOPs, {i} on intermediates"


def exact_row(L: int, direction=None) -> dict:
    fr = opt.optimum(L)
    m = mild_point(L, direction)
    s = sieved_point(L)
    ref, sref = reference(fr, m), reference(fr, s)
    g = gap(m, ref)
    return {
        "L": L,
        "frontier": [list(p) for p in fr],
        "mild": m,
        "reference": ref,
        "gap_percent": g,
        "direction": direction_words(g),
        "mild_sieved": s,
        "sieved_reference": sref,
        "sieved_gap_percent": gap(s, sref),
    }


def _bound(point, fr) -> dict:
    """What the beam frontier `fr` proves about `point`."""
    ref = reference(fr, point)
    out: dict = {"reference": ref}
    if ref is None:
        out["proves"] = "nothing: every beam point has more intermediates"
        out["lower_bound_percent"] = None
        return out
    g = gap(point, ref)
    out["vs_reference_percent"] = g
    if ref[1] <= point[1] and ref != list(point):
        out["proves"] = ("the beam's reference dominates: "
                         "MILD is at least this far behind")
        out["lower_bound_percent"] = g
    else:
        out["proves"] = "nothing on FLOPs: the beam's reference spends more FLOPs"
        out["lower_bound_percent"] = None
    return out


def beam_row(L: int) -> dict:
    fr = opt.beam(L, BEAM_WIDTH)
    m, s = mild_point(L), sieved_point(L)
    return {
        "L": L,
        "in_published_grid": L in PUBLISHED_GRID,
        "beam_frontier": [list(p) for p in fr],
        "mild": m,
        "mild_bound": _bound(m, fr),
        "mild_sieved": s,
        "sieved_bound": _bound(s, fr),
    }


def in_band(rows) -> bool:
    for r in rows:
        g = r["gap_percent"]
        if r["L"] <= BAND["zero_through_L"]:
            if g != {"intermediates": 0.0, "flops": 0.0}:
                return False
        lo, hi = BAND["flops_percent"]
        if not lo <= g["flops"] <= hi:
            return False
        lo, hi = BAND["intermediates_percent"]
        if not lo <= g["intermediates"] <= hi:
            return False
    return True


def _record() -> dict:
    return json.loads(DATA.read_text())


# -- degraded rules: what the claim test must catch ---------------------------


def _masked(r):
    return [c if c else inf for c in r]


def ties_to_lowest_axis(r) -> str:
    """Min non-zero component, ties to the LOWEST axis (no cyclic rule)."""
    m = _masked(r)
    return AXES[m.index(min(m))]


def ties_to_highest_axis(r) -> str:
    m = _masked(r)
    lo = min(m)
    return AXES[max(i for i in range(3) if m[i] == lo)]


def largest_component(r) -> str:
    """Not MILD at all: reduce the LARGEST component."""
    return AXES[r.index(max(r))]


def ties_to_predecessor(r) -> str:
    """MILD's mirror image: ties to the cyclic PREDECESSOR, all-equal to x."""
    m = _masked(r)
    if m[0] == m[1] == m[2]:
        return "x"
    lo = min(m)
    for i in range(3):
        if m[i] == lo and m[(i + 1) % 3] == lo:
            return AXES[i]
    return AXES[m.index(lo)]


# -- the claim ----------------------------------------------------------------


@pytest.mark.claim("jengine/mild-vs-exact-tradeoff")
@pytest.mark.claim("jengine/mild-suboptimal-from-l5")  # equal to L = 4, behind at 5
def test_mild_against_the_exact_optimum() -> None:
    # Recomputed, not read back: every L the exact search reaches, MILD's gap
    # on each axis against the stated reference point, pinned to the record,
    # inside the stated band, with the direction in words.
    rec = _record()["exact"]
    rows = [exact_row(L) for L in EXACT_LS]
    assert rows == rec["rows"]
    assert BAND == rec["band"]
    assert in_band(rows)
    # The finding, spelled out: equal through L = 4; at L = 5 behind on BOTH
    # axes -- not §4.1's "behind on FLOPs, ahead on intermediates".
    assert [r["direction"] for r in rows] == (
        ["equal to the reference"] * 5 + ["behind on both axes"])
    assert rows[5]["mild"] == [51, 161] and rows[5]["reference"] == [48, 160]
    assert rows[5]["gap_percent"] == {"intermediates": 6.25, "flops": 0.625}
    # Ahead on intermediates cannot happen here: every exact frontier is a
    # single point, which no scheme in the space beats on either axis.
    assert all(len(r["frontier"]) == 1 for r in rows)


@pytest.mark.parametrize("rule", [
    ties_to_lowest_axis, ties_to_highest_axis, largest_component,
], ids=lambda f: f.__name__)
def test_a_degraded_mild_leaves_the_band(rule) -> None:
    # The claim test is a witness only if it fails for a worse MILD: each of
    # these moves the recorded gap outside the band (and off the pins).
    rows = [exact_row(L, rule) for L in EXACT_LS]
    assert not in_band(rows)
    assert rows != _record()["exact"]["rows"]


def test_the_mirror_tie_break_is_indistinguishable_by_cost() -> None:
    # A symmetry, not a blind spot: reversing the axis order maps MILD's
    # successor tie-break onto the predecessor one, and the targets and the
    # cost are invariant under it, so the gap cannot tell them apart.
    # test_plan_mild.py pins the tie-break itself.
    assert [exact_row(L, ties_to_predecessor) for L in EXACT_LS] == (
        _record()["exact"]["rows"])


# -- the beam, above the exact ceiling -----------------------------------------


def _beam_rows() -> dict[int, dict]:
    return {r["L"]: r for r in _record()["beam"]["rows"]}


@pytest.mark.parametrize("L", FAST_BEAM_LS)
def test_beam_bound_cheap_cells(L) -> None:
    assert beam_row(L) == _beam_rows()[L]


@pytest.mark.slow
@pytest.mark.parametrize("L", SLOW_BEAM_LS)
def test_beam_bound_every_cell(L) -> None:
    assert beam_row(L) == _beam_rows()[L]


def test_beam_record_shape_and_findings() -> None:
    beam = _record()["beam"]
    assert beam["width"] == BEAM_WIDTH
    rows = _beam_rows()
    assert sorted(rows) == list(BEAM_LS)
    proved = {L for L, r in rows.items()
              if r["mild_bound"]["lower_bound_percent"] is not None}
    # On the published grid the beam proves MILD behind at L = 11 and 12 only
    # (and at L = 6, outside it), there on BOTH axes: elsewhere its point has
    # fewer intermediates but more FLOPs.
    assert proved == {6, 11, 12}
    for L in proved:
        lb = rows[L]["mild_bound"]["lower_bound_percent"]
        assert lb["flops"] > 0 and lb["intermediates"] > 0
    # It proves nothing about MILD + sieve anywhere: the sieved plan has fewer
    # intermediates than the beam's point everywhere, and dominates it on the
    # grid.
    for L, r in rows.items():
        assert r["sieved_bound"]["lower_bound_percent"] is None
        (b,) = r["beam_frontier"]
        s = r["mild_sieved"]
        assert s[0] < b[0]
        if L in PUBLISHED_GRID:
            assert s[1] <= b[1]
    # Every lower bound is a dominating ACHIEVABLE point's gap, never a
    # negative number dressed as a bound.
    for r in rows.values():
        for key, pt in (("mild_bound", "mild"), ("sieved_bound", "mild_sieved")):
            b, p = r[key], r[pt]
            if b["lower_bound_percent"] is not None:
                assert b["reference"][0] <= p[0] and b["reference"][1] <= p[1]
                assert b["reference"] != p


# -- the record's own statements -----------------------------------------------


def test_the_record_states_its_search_space_and_reference() -> None:
    rec = _record()
    space = rec["search_space"].lower()
    for phrase in ("one reduction direction per materialised node",
                   "shared", "no algebraic rewriting",
                   "no recomputing a node to shorten a live range"):
        assert phrase in space
    assert "equal-or-fewer intermediates" in rec["reference_point"]
    assert rec["paper"] == {
        "source": "[JEngine, v2, §4.1]",
        "flops_percent_behind": [1, 2],
        "intermediates_percent_ahead": [2, 6],
    }
    assert rec["exact"]["max_L"] == opt.EXACT_MAX_L
    assert "reproduces" not in json.dumps(rec)


def test_the_claim_is_asserted_here() -> None:
    (c,) = [c for c in load_claims(JENGINE / "claims.json") if c.id == CLAIM]
    assert c.status is ClaimStatus.ASSERTED
    assert c.test == (
        "test/test_mild_vs_optimum.py::test_mild_against_the_exact_optimum")


def test_a_limit_says_the_papers_comparison_is_replaced() -> None:
    (lim,) = [lim for lim in load_limits(JENGINE / "limits.json")
              if lim.asserted_value.get("claim") == CLAIM]
    v = lim.asserted_value
    assert v["replaced_by"] == "data/literature/jengine/mild_vs_optimum.json"
    assert (v["flops_percent_behind"], v["intermediates_percent_ahead"]) == (
        [1, 2], [2, 6])
    assert "refs 59" in lim.source


# -- writing the record ---------------------------------------------------------


def _write() -> None:
    rec = _record() if DATA.exists() else {}
    rec["exact"] = {**rec.get("exact", {}), "max_L": opt.EXACT_MAX_L,
                    "band": BAND, "rows": [exact_row(L) for L in EXACT_LS]}
    rec["beam"] = {**rec.get("beam", {}), "width": BEAM_WIDTH,
                   "rows": [beam_row(L) for L in BEAM_LS]}
    DATA.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: PYTHONPATH=src python test/test_mild_vs_optimum.py --write")
    _write()
