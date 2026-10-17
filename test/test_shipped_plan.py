"""What the generator ships (#166): the LITERAL plan, on measured grounds.

`KERNEL_GEN_CMD` (`python -m felizjueves.emit`) builds every `[p|q]` kernel
from the literal plan -- the minimax cut and the sieve's fixed seed order,
`Spec.levers = ()` -- and not from the searched board's tuned plan
(#161's `tuned` lever), which the predecessor `jgen` ships. The decision is
read off measured wall-clock under the rule #163 declared before any tuned
row (`measured_store`, THE TUNED ARM, THE DEFAULT), never off a model:

- `tuned_may_become_the_default` is False and the search is NOT SHOWN on
  both cards (#163);
- the sieve may not leave the default either (#162, NOT SHOWN on both);
- `PlanCost` is not admitted as grounds: on neither card do the plan-layer
  counts predict the kernel (#165, `data/measured/plan_vs_kernel.json`).

This file holds the tree to that decision: the default really is literal
(the shipped path's text is the literal rows' text, and not the tuned rows'
where they differ), the `limits.json` record that states the refusal is
recomputed from the stores rather than re-typed, and `j_engine.md` carries
the sentence. That every `source_sha256` is the text the generator writes
today is `test_measured.py::test_the_recorded_source_is_what_the_generator_
writes_today` (every row) and `test_emit_cli.py::test_the_pq_ladder_still_
matches_the_measured_pins` (the CLI's own ladder); the goldens are
`test_emit_{cuda,hip,batched}.py`. Nothing is re-asserted here that those
already hold.

Fast tier: nothing here compiles anything. The shipped-path check emits
the subgrid's L = 7..16 classes and is `slow` (`tiers.FAST_L_MAX`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from felizjueves import measured_store
from felizjueves.emit import __main__ as emit_cli
from felizjueves.kernel import pq_spec
from felizjueves.measured_store import (
    LOSS,
    MIXED,
    NEUTRAL,
    NOT_MEASURABLE,
    NOT_SHOWN,
    TUNED,
    TUNED_CELLS,
    WIN,
)
from felizjueves.spec import Spec

ROOT = Path(__file__).resolve().parent.parent
LIMITS = ROOT / "data" / "literature" / "jengine" / "limits.json"
PLAN_VS_KERNEL = ROOT / "data" / "measured" / "plan_vs_kernel.json"
BATCH_RECOMMENDATION = ROOT / "data" / "measured" / "batch_recommendation.json"
JENGINE_MD = ROOT / "docs" / "implementation" / "j_engine.md"
ARCHS = ("sm_86", "gfx1200")
SUBJECT = "the shipped plan"

ROWS = measured_store.load()


def _cell(spec: Spec) -> str:
    return f"L{spec.total_am} b{spec.n_batches}"


def _one(spec: Spec, arch: str) -> measured_store.Measured:
    (row,) = measured_store.rows(spec=spec, arch=arch, store=ROWS).values()
    return row


def decision(store=ROWS) -> dict:
    """The refusal's `asserted_value`, recomputed from the stores."""
    pvk = json.loads(PLAN_VS_KERNEL.read_text())["summary"]
    rec = json.loads(BATCH_RECOMMENDATION.read_text())["cells"]
    unbuildable = {s.n_batches for s in NOT_MEASURABLE[TUNED]}
    cards = {}
    for a in ARCHS:
        ab = measured_store.tuned_ab(store, a)
        by = {v: [_cell(s) for s, x in ab.items() if x.verdict == v]
              for v in (WIN, LOSS, MIXED, NEUTRAL)}
        same_text = {s for s in TUNED_CELLS
                     if _one(s, a).source_sha256
                     == _one(replace(s, levers=()), a).source_sha256}
        cards[a] = {
            "tuned_verdict": measured_store.tuned_verdict(store, a),
            "tuned_win": by[WIN], "tuned_loss": by[LOSS],
            "tuned_mixed": by[MIXED], "tuned_neutral": len(by[NEUTRAL]),
            "tuned_loss_on_literal_text": [
                _cell(s) for s, x in ab.items()
                if x.verdict == LOSS and s in same_text],
            "sieve_verdict": measured_store.sieve_verdict(store, a),
            "plan_counts_predict_the_kernel": pvk[a]["predicts"],
        }
    some = ARCHS[0]
    return {
        "board": None, "rule": None, "claim": None,
        "subject": SUBJECT,
        "ships": "literal",
        "declined": TUNED,
        "rule_declared": "measured_store, THE TUNED ARM, THE DEFAULT (#163, "
                         "before any tuned row): tuned WINS at every "
                         "measurable cell on both cards",
        "tuned_may_become_the_default":
            measured_store.tuned_may_become_the_default(store),
        "sieve_may_leave_the_default":
            measured_store.sieve_may_leave_the_default(store),
        "cards": cards,
        "tuned_text_is_literal": [
            _cell(s) for s in TUNED_CELLS
            if all(_one(s, a).source_sha256
                   == _one(replace(s, levers=()), a).source_sha256
                   for a in ARCHS)],
        "tuned_cut_differs": [
            _cell(s) for s in TUNED_CELLS
            if _one(s, some).kernel != _one(replace(s, levers=()), some).kernel],
        "tuned_not_measurable": sorted(
            (_cell(s) for s in NOT_MEASURABLE[TUNED]),
            key=lambda c: int(c[1:].split()[0])),
        "recommended_count_tuned_cannot_build": [
            f"{c['arch']} L{c['L']} {w}-T b{c[w]['n_batches']}"
            for c in rec for w in ("wide", "near")
            if c[w]["n_batches"] in unbuildable],
    }


def _record() -> dict:
    (got,) = [r for r in json.loads(LIMITS.read_text())["limits"]
              if isinstance(r["asserted_value"], dict)
              and r["asserted_value"].get("subject") == SUBJECT]
    return got


# --- the decision, read off the rows -----------------------------------------


def test_the_declared_rule_keeps_literal_on_both_cards() -> None:
    """#163's rule, as the tracked rows read: tuned does not WIN at every
    measurable cell of either card, so the default does not move."""
    assert measured_store.tuned_may_become_the_default(ROWS) is False
    for a in ARCHS:
        assert measured_store.tuned_verdict(ROWS, a) == NOT_SHOWN, a
        assert measured_store.sieve_verdict(ROWS, a) == NOT_SHOWN, a
    assert measured_store.sieve_may_leave_the_default(ROWS) is False


def test_the_refusal_stands_without_the_noise_floor() -> None:
    """gfx1200's L7 b3 and b6 LOSSes are on text identical to literal (#163):
    the protocol's noise floor. Discount every such cell and tuned still
    LOSES somewhere on each card -- the refusal does not rest on a null."""
    for a, card in decision()["cards"].items():
        real = set(card["tuned_loss"]) - set(card["tuned_loss_on_literal_text"])
        assert real, a
        assert "L7 b9" in real, a  # the cell whose cut is not the minimax cut


def test_planrank_is_not_admitted_as_grounds() -> None:
    """#165: on neither card do the plan-layer counts predict the kernel, so
    a plan that COUNTS cheaper is not, by that, the one to ship."""
    pvk = json.loads(PLAN_VS_KERNEL.read_text())["summary"]
    for a in ARCHS:
        assert pvk[a]["predicts"] is False, a
        assert "do not predict the kernel" in pvk[a]["answer"], a


def test_the_limits_record_is_recomputed_from_the_stores() -> None:
    assert _record()["asserted_value"] == decision()


# --- the tree is self-consistent with it -------------------------------------


def test_the_default_spec_is_literal() -> None:
    assert Spec.__dataclass_fields__["levers"].default == ()
    for L in range(17):
        assert pq_spec(L).levers == ()
    assert "--lever" not in emit_cli.__doc__ and "tuned" not in emit_cli.__doc__


@pytest.mark.slow  # L = 7..16 > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_the_shipped_path_writes_literal_text_not_tuned(
        backend, tmp_path, monkeypatch) -> None:
    """`KERNEL_GEN_CMD`'s `[p|q]` units at the one-batch subgrid classes hash
    to the literal rows -- and, where the tuned text differs, not to the
    tuned rows. The shipped path writes the literal plan, by assertion."""
    for v in ("KERNEL_TIER_BACKEND", "KERNEL_TIER_ARCH", "KERNEL_TIER_OUT"):
        monkeypatch.delenv(v, raising=False)
    arch, ext = {"cuda": ("sm_86", "cu"), "hip": ("gfx1200", "hip")}[backend]
    Ls = sorted({s.total_am for s in TUNED_CELLS if s.n_batches == 1})
    assert emit_cli.main(["--backend", backend, "--out", str(tmp_path),
                          "--L", *map(str, Ls)]) == 0
    differs = 0
    for L in Ls:
        lit = _one(pq_spec(L), arch)
        tuned = _one(pq_spec(L, (TUNED,), 1), arch)
        sha = hashlib.sha256(
            (tmp_path / f"{lit.kernel}.{ext}").read_bytes()).hexdigest()
        assert sha == lit.source_sha256, (L, arch)
        if tuned.source_sha256 != lit.source_sha256:
            assert sha != tuned.source_sha256, (L, arch)
            differs += 1
    assert differs  # the check is not vacuous: tuned moves some shipped unit


def test_j_engine_md_says_what_ships_and_why() -> None:
    text = " ".join(JENGINE_MD.read_text().split())
    assert "The generator ships the **literal** plan" in text
    for cite in ("tuned_may_become_the_default", "plan_vs_kernel.json",
                 "sieve_may_leave_the_default", "batch_recommendation.json"):
        assert cite in text, cite
