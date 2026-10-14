"""The reading of [Fock] §3 (#271), held against its claims and its limits.

data/literature/fock/ is milestone 12's literature record, in the shape
test_claims.py holds jengine/ to: coverage.json is the unit-by-unit reading,
claims.json the scorable claims, limits.json what cannot be scored here. What
is tested is that the three cannot drift apart: each unit lists exactly the
records whose `source` names it, a unit not read lists none, the unit list is
pinned to the paper's structure, every record is declared with its scoring
rule, and the decisions the milestone pinned (no table cells, no [JEngine]
change) are visible in the files. Fast tier: no marker, no toolchain, no PySCF.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from felizjueves.literature.claims import ClaimKind, ClaimStatus, load_claims
from felizjueves.literature.limits import load_limits

ROOT = Path(__file__).resolve().parents[1]
FOCK = ROOT / "data" / "literature" / "fock"
CLAIMS = FOCK / "claims.json"
COVERAGE = FOCK / "coverage.json"
LIMITS = FOCK / "limits.json"

# Milestone 12 reads §3 only. Inside it: eqs. 25-29, the design decisions of
# §3.2 (numbered by the reading, in text order: the paper does not number
# them), and Fig. 1's four components (three labelled panels and the legend).
EQS = tuple(f"eq. {n}" for n in range(25, 30))
DECISIONS = tuple(f"§3.2 decision {n}" for n in range(1, 9))
FIG1 = ("Fig. 1 Kernel Launches", "Fig. 1 Block Launches",
        "Fig. 1 Block Execution", "Fig. 1 legend")
SECTION_3 = ("§3", "§3.1", *EQS[:4], "§3.2", *DECISIONS[:2], EQS[4],
             *DECISIONS[2:], *FIG1)

# The structure of arXiv:2407.21445v1, in the paper's order: every numbered
# section and subsection, with §3 opened into the units above.
UNITS = (
    "Abstract", "§1", "§2", "§2.1", "§2.2", "§2.3",
    *SECTION_3,
    "§4", "§4.1", "§4.2", "§4.2.1", "§4.3", "§4.3.1", "§4.3.2", "§4.3.3",
    "§4.4", "§5", "§6", "§6.1", "§6.1.1", "§6.2", "§6.3", "§6.4", "§6.4.1",
    "§6.4.2", "§6.4.3", "§7", "§8",
)
# Read beyond §3 because §3 depends on their notation or a §3 claim is stated
# there; every other unit outside §3 is not read.
READ_OUTSIDE = {"§1", "§2", "§2.1", "§2.2", "§2.3", "§4", "§4.3", "§4.3.1"}

# The issue's minimum: these five, with their kinds.
REQUIRED = {
    "fock/fused-six-block-digest": ClaimKind.SHAPE,
    "fock/symmetry-vs-screening": ClaimKind.SHAPE,
    "fock/skip-not-break": ClaimKind.SHAPE,
    "fock/predigest-bounds-atomics": ClaimKind.MONOTONICITY,
    "fock/prelaunch-screening-removes-launches": ClaimKind.SHAPE,
}

_PREFIX = "[Fock, "
_EQ_RANGE = re.compile(r"eqs\. (\d+)-(\d+)")
_ISSUE = re.compile(r"#\d+")
_QUOTE = re.compile(r"'[^']{3,}'")


def _coverage() -> list[dict]:
    return json.loads(COVERAGE.read_text())["units"]


def cited_units(source: str) -> set[str]:
    """The units a `source` names: tokens joined by ', ' or ' vs ', each a
    unit name, or `eqs. N-M` for the equation units in the range. Anything
    else -- a Table above all, since tables are not units -- is refused."""
    if not (source.startswith(_PREFIX) and source.endswith("]")):
        raise ValueError(f"{source!r} does not cite [Fock, ...]")
    out: set[str] = set()
    for tok in re.split(r", | vs ", source[len(_PREFIX):-1]):
        m = _EQ_RANGE.fullmatch(tok)
        names = ([f"eq. {n}" for n in range(int(m[1]), int(m[2]) + 1)]
                 if m else [tok])
        for name in names:
            if name not in UNITS:
                raise ValueError(f"{source!r}: {name!r} is not a unit")
            out.add(name)
    return out


# -- the reading ----------------------------------------------------------------


def test_the_reading_covers_every_unit_of_the_paper_in_order() -> None:
    units = _coverage()
    assert tuple(u["unit"] for u in units) == UNITS
    for u in units:
        assert set(u) == {"unit", "status", "claims", "note"}, u["unit"]
        assert u["status"] in {"read", "not-read"}, u["unit"]
        assert isinstance(u["note"], str) and u["note"].strip(), u["unit"]
        assert u["claims"] == sorted(set(u["claims"])), u["unit"]


def test_section_3_is_read_end_to_end_and_nothing_else_is_claimed_read() -> None:
    status = {u["unit"]: u["status"] for u in _coverage()}
    read = {name for name, s in status.items() if s == "read"}
    assert read == set(SECTION_3) | READ_OUTSIDE
    # the reading names each equation, each decision and each component
    assert len(EQS) == 5 and len(DECISIONS) == 8 and len(FIG1) == 4


def test_a_unit_not_read_lists_no_claims() -> None:
    for u in _coverage():
        if u["status"] == "not-read":
            assert not u["claims"], u["unit"]
            assert u["note"].startswith("Not read"), u["unit"]


def test_every_record_is_in_the_reading_and_every_unit_lists_its_records() -> None:
    cites: dict[str, set[str]] = {name: set() for name in UNITS}
    for c in load_claims(CLAIMS):
        placed = cited_units(c.source)
        assert placed, c.id
        for name in placed:
            cites[name].add(c.id)
    # both directions: a record whose unit does not list it, and a unit listing
    # a record that does not cite it (or does not exist), each fail here
    assert {u["unit"]: set(u["claims"]) for u in _coverage()} == cites


def test_a_note_names_only_records_that_exist() -> None:
    ids = {c.id for c in load_claims(CLAIMS)}
    for u in _coverage():
        named = set(re.findall(r"fock/[a-z0-9-]+[a-z0-9]", u["note"]))
        assert named <= ids, (u["unit"], sorted(named - ids))


def test_the_reading_records_its_text() -> None:
    read = json.loads(COVERAGE.read_text())["read"]
    assert "arXiv:2407.21445v1" in read["text"]
    assert "doi:10.1021/acs.jctc.4c00994" in read["text"]
    assert read["by"].startswith("#271")


# -- the records ----------------------------------------------------------------


def test_the_issue_minimum_is_declared_with_its_kinds() -> None:
    kinds = {c.id: c.kind for c in load_claims(CLAIMS)}
    assert {cid: kinds.get(cid) for cid in REQUIRED} == REQUIRED


def test_every_record_is_fock_and_cites_it() -> None:
    for c in load_claims(CLAIMS):
        assert c.id.startswith("fock/"), c.id
        assert c.source.startswith(_PREFIX), c.id


def test_every_pending_record_names_its_scoring_rule() -> None:
    claims = load_claims(CLAIMS)
    assert claims
    for c in claims:
        if c.status is ClaimStatus.PENDING:
            # the rule, and the issue whose work will score it
            assert "Scored by" in c.reason, c.id
            assert _ISSUE.search(c.reason), c.id


_NODE = re.compile(r"test/[A-Za-z0-9_/]+\.py::[A-Za-z0-9_]+")
_CARDS = ("RTX 3080", "RX 9060 XT", "A100")


def _defined(node: str) -> bool:
    path, name = node.split("::")
    src = (ROOT / path).read_text() if (ROOT / path).exists() else ""
    return re.search(rf"^def {name}\(", src, re.MULTILINE) is not None


def test_every_unverifiable_record_says_what_would_settle_it() -> None:
    # #291, the m5 discipline: an UNVERIFIABLE record names what would
    # settle it and on what hardware, and every test it names as holding
    # the reachable half exists (a rename shows up here, not as a stale
    # pointer).
    for c in load_claims(CLAIMS):
        if c.status is not ClaimStatus.UNVERIFIABLE:
            continue
        assert "would settle it" in c.reason, c.id
        assert any(card in c.reason for card in _CARDS), c.id
        for node in _NODE.findall(c.reason):
            assert _defined(node), (c.id, node)


def test_no_record_rests_on_the_ptxas_baseline() -> None:
    # data/measured/fock_baseline.json (#289) is a quality baseline of common
    # ancestry, never a numeric witness (limits.json): no record may cite it
    # or be held by its test.
    for c in load_claims(CLAIMS):
        text = f"{c.statement} {c.reason} {c.test}"
        assert "fock_baseline" not in text, c.id
        assert "ptxas" not in text.lower(), c.id


def test_section_3s_claims_are_scored() -> None:
    # #291 moved every record #271 declared off PENDING: the issue's five
    # minimum to the verdicts its scope stated in advance, and the defects
    # all asserted. A later record may arrive PENDING (with its rule, above);
    # these may not go back.
    status = {c.id: c.status for c in load_claims(CLAIMS)}
    A, U = ClaimStatus.ASSERTED, ClaimStatus.UNVERIFIABLE
    assert {cid: status[cid] for cid in REQUIRED} == {
        "fock/fused-six-block-digest": A,
        "fock/skip-not-break": A,
        "fock/predigest-bounds-atomics": A,
        "fock/prelaunch-screening-removes-launches": A,
        "fock/symmetry-vs-screening": U,
    }
    defects = {c.id for c in load_claims(CLAIMS) if c.kind is ClaimKind.DEFECT}
    assert len(defects) == 5
    assert all(status[cid] is A for cid in defects)


def test_every_statement_quotes_its_source_location() -> None:
    # In our vocabulary, and anchored: each statement names a unit it cites
    # and carries at least one quotation from the paper.
    for c in load_claims(CLAIMS):
        assert _QUOTE.search(c.statement), c.id
        anchors = {_anchor(name) for name in cited_units(c.source)}
        text = c.statement.lower()
        assert any(a.lower() in text for a in anchors), (c.id, sorted(anchors))


def _anchor(unit: str) -> str:
    """How a statement names a unit: a decision by its section, a Fig. 1
    component by its figure, anything else by its own name."""
    if unit in DECISIONS:
        return "§3.2"
    if unit in FIG1:
        return "Fig. 1"
    return unit


@pytest.mark.parametrize(
    "source, units",
    [
        pytest.param("[Fock, §3.2 decision 6]", {"§3.2 decision 6"}, id="decision"),
        pytest.param("[Fock, eq. 26 vs eq. 25]", {"eq. 25", "eq. 26"}, id="vs"),
        pytest.param("[Fock, §2.1 vs eqs. 25-26]", {"§2.1", "eq. 25", "eq. 26"},
                     id="eq-range"),
        pytest.param("[Fock, Fig. 1 Block Execution, §1]",
                     {"Fig. 1 Block Execution", "§1"}, id="figure"),
    ],
)
def test_a_source_resolves_to_its_units(source: str, units: set[str]) -> None:
    assert cited_units(source) == units


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("[Fock, Table 3]", id="table"),
        pytest.param("[Fock, eq. 14]", id="eq-not-a-unit"),
        pytest.param("[Fock, §3.2 decision 9]", id="no-such-decision"),
        pytest.param("[Fock, Fig. 1]", id="figure-not-a-component"),
        pytest.param("[JEngine, v2, §5]", id="other-paper"),
    ],
)
def test_a_source_that_cannot_be_placed_is_refused(source: str) -> None:
    with pytest.raises(ValueError):
        cited_units(source)


# -- the limits, and what the milestone pinned ----------------------------------


def test_the_limits_load_and_cite_fock() -> None:
    limits = load_limits(LIMITS)
    assert limits
    for lim in limits:
        assert lim.source.startswith(_PREFIX), lim.claim


def test_the_limits_name_what_cannot_be_scored() -> None:
    text = " ".join(f"{lim.claim} {lim.mechanism}" for lim in load_limits(LIMITS))
    # the tables are timings on cards this box does not have
    assert "A100" in text and "RTX 3080" in text and "RX 9060 XT" in text
    # §6 is whole-program wall-clock
    assert "§6" in text and "wall-clock" in text
    # and there is no published register table for the §3.2 kernel
    assert "no published register table for the §3.2 kernel" in text
    # no [JEngine] cell or floor moves
    assert "moves no [JEngine] cell, floor" in text


def test_no_table_is_transcribed() -> None:
    # Milestone 12 pin 2: one literature file, no table cells.
    assert sorted(p.name for p in FOCK.iterdir()) == [
        "README.md", "claims.json", "coverage.json", "limits.json"]
    assert not list(FOCK.glob("table*.json"))
