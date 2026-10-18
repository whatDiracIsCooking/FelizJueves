"""The recommended batch count per (L, arch), from wall-clock only (#56).

Three things are held here, all without a compiler or a card -- a check
of measurement artefacts, moved from test/ (`devtools/measure.sh check`;
the rule on synthetic rows stayed, as test/test_batchreport.py):

- THE DATA IS THE DERIVATION. data/measured/batch_recommendation.json equals
  `batchreport.report` recomputed from data/measured/kernels.json, and the
  tables in docs/implementation/j_engine.md are `batchreport.markdown` of it,
  verbatim -- so the store, the data file and the doc cannot drift apart.
- WALL-CLOCK ONLY. Moving a row's spills or registers moves nothing; moving
  its median does. The import side is test/test_layering.py's.
- NOT A POLICY. `Spec.n_batches` still defaults to 1, and since #167 the
  data file's `about` says that 1 is shipped by decision, on these rows.

The pinned verdict (`PINNED`) is the result on the tracked rows, so a
re-timing that moves a recommendation is seen as a diff here.
"""

from __future__ import annotations

import dataclasses
import json

from felizjueves import batchreport
from felizjueves import measured_store as measured
from felizjueves.device import ARCHS
from felizjueves.kernel import pq_spec
from felizjueves.spec import Operator, Spec, Target

ROWS = measured.load()
DOC = json.loads(batchreport.RECOMMENDATION.read_text())
J_ENGINE = (measured.REPO / "docs" / "implementation" / "j_engine.md").read_text()


# --- the data, the derivation and the doc agree -------------------------------


def test_the_data_file_is_the_report_recomputed_from_the_store() -> None:
    assert DOC == batchreport.report(ROWS), (
        "rerun: PYTHONPATH=src uv run python -m felizjueves.batchreport --write")


def test_the_j_engine_tables_are_the_data_verbatim() -> None:
    assert batchreport.markdown(DOC) in J_ENGINE, (
        "paste `python -m felizjueves.batchreport --markdown` into j_engine.md")


def test_lever_rows_are_not_read() -> None:
    # #55's recycle arm is in the store; the report is of the literal rows only.
    levered = {k for k in ROWS if "/literal@" not in k}
    assert levered, "expected lever rows in the store (#55)"
    literal = {k: r for k, r in ROWS.items() if k not in levered}
    assert batchreport.report(literal) == batchreport.report(ROWS)


def test_every_subgrid_cell_on_both_arches_is_covered_and_none_dropped() -> None:
    got = {(c["arch"], c["L"]) for c in DOC["cells"]}
    assert got == {(a.name, L) for a in ARCHS for L in measured.SUBGRID_LS}
    assert len(DOC["cells"]) == len(got)
    assert DOC["header"]["dropped_cells"] == []
    assert DOC["header"]["subgrid"]["n_batches"] == list(batchreport.BATCHES)
    for c in DOC["cells"]:
        for w in batchreport.INPUTS:
            assert set(c[w]["all"]) == {str(b) for b in batchreport.BATCHES}
            assert c[w]["repeats"] > 1  # one timing is not a recommendation


def test_the_provenance_names_the_host_both_cards_and_both_toolchains() -> None:
    p = DOC["header"]["provenance"]
    assert p["sm_86"]["gpu"] == "NVIDIA GeForce RTX 3080"
    assert p["sm_86"]["toolchain"] == "nvcc 13.0"
    assert p["gfx1200"]["gpu"].endswith("gfx1200)")
    assert p["gfx1200"]["toolchain"] == "hipcc / ROCm 7.2.4"
    assert p["sm_86"]["host"] == p["gfx1200"]["host"]
    for arch in p.values():
        assert arch["host"] in J_ENGINE and arch["toolchain"] in J_ENGINE


# --- the verdict on the tracked rows ------------------------------------------

# (arch, L): (best on wide-T, best on near-T, near-T resolved)
PINNED = {
    ("sm_86", 7): (1, 1, True),
    ("sm_86", 10): (1, 1, True),
    ("sm_86", 13): (1, 3, True),
    ("sm_86", 16): (1, 9, True),
    ("gfx1200", 7): (1, 1, True),
    ("gfx1200", 10): (1, 3, False),
    ("gfx1200", 13): (1, 6, False),  # 3 before #93's re-timing; a tie either way
    ("gfx1200", 16): (1, 12, True),
}


def test_the_recommendation_on_the_tracked_rows() -> None:
    got = {(c["arch"], c["L"]): (c["wide"]["n_batches"], c["near"]["n_batches"],
                                 c["near"]["resolved"]) for c in DOC["cells"]}
    assert got == PINNED
    # wide-T: one batch, beyond its spread, everywhere
    assert all(c["wide"]["resolved"] and c["wide"]["within_spread"] == [1]
               for c in DOC["cells"])


def test_the_answer_depends_on_the_input_and_the_arches_disagree() -> None:
    # The finding, not a defect: the premise "the batch count with the best
    # median" has two answers, and at L = 16 the two cards give different ones.
    assert any(not c["agree"] for c in DOC["cells"])
    near16 = {c["arch"]: c["near"]["n_batches"] for c in DOC["cells"]
              if c["L"] == 16}
    assert near16["sm_86"] != near16["gfx1200"]


# --- not a policy -------------------------------------------------------------


def test_spec_n_batches_still_defaults_to_one() -> None:
    field = {f.name: f for f in dataclasses.fields(Spec)}["n_batches"]
    assert field.default == 1
    assert Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J).n_batches == 1
    assert pq_spec(16).n_batches == 1
    not_policy = DOC["header"]["not_policy"]
    assert "does not change" in not_policy
    assert "dispatch.py" in not_policy
    assert "never for the generator" in not_policy


def test_dispatch_has_a_home_nowhere_says_otherwise() -> None:
    """#229: dispatch lives in `dispatch.py` since #145, so no source or
    measured-data file may still say it has no home."""
    root = measured.REPO
    offenders = [
        str(f.relative_to(root))
        for base in (root / "src", root / "data" / "measured")
        for f in sorted(base.rglob("*"))
        if f.is_file() and f.suffix in {".py", ".json", ".md"}
        and "has no home" in f.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_one_batch_is_a_decision_and_its_grounds_are_the_rows() -> None:
    """#167: the shipped `n_batches = 1` is stated as a choice, and each of
    the grounds `ABOUT` names is read off the data file here rather than
    believed: wide-T picks 1 everywhere, near-T first splits at L = 13 on
    some card, and the cards disagree at L = 16 -- so following the rows
    would make the shipped unit set depend on the arch and the input."""
    about = " ".join(DOC["about"].split())
    assert about == " ".join(batchreport.ABOUT.split())
    assert "ships n_batches = 1 by decision (#167)" in about
    assert "NOT a policy" not in about and "future dispatch" not in about
    for phrase in ("wide-T picks 1 everywhere", "near-T splits from L = 13",
                   "the two cards disagree at L = 16", "per (arch, input)"):
        assert phrase in about, phrase
    cells = DOC["cells"]
    assert {c["wide"]["n_batches"] for c in cells} == {1}
    split_at = min(c["L"] for c in cells if c["near"]["n_batches"] != 1)
    # gfx1200's L = 10 near-T pick of 3 is unresolved against 1 (within its
    # spread), so the first RESOLVED split is the one ABOUT quotes
    resolved_split = min(c["L"] for c in cells
                         if c["near"]["n_batches"] != 1 and c["near"]["resolved"])
    assert split_at == 10 and resolved_split == 13
    near16 = {c["arch"]: c["near"]["n_batches"] for c in cells if c["L"] == 16}
    assert len(set(near16.values())) == 2
