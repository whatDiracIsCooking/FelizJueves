"""The claims registry: the record type, the seed inventory, and the check that
keeps the inventory and the claim-marked tests in step, in both directions.

Nothing here tests a claim -- each ASSERTED record is discharged by the
marked test it names, elsewhere under test/. What is tested is that the inventory
cannot rot: an ASSERTED record must name a test that exists and carries its
mark, a marked test must discharge a registered record, and a record that is
not asserted must say why. And the inventory cannot be incomplete without
saying so: coverage.json records the section-by-section reading of the
source, and each unit of the paper must list exactly the records that cite it.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from conftest import scan_claim_marks

from felizjueves.literature.claims import (
    SCHEMA,
    Claim,
    ClaimKind,
    ClaimStatus,
    discover,
    load_claims,
    load_registry,
)

ROOT = Path(__file__).resolve().parents[1]
LITERATURE = ROOT / "data" / "literature"
CLAIMS = LITERATURE / "jengine" / "claims.json"

# The papers whose claims are inventoried, one directory each (#271). A fact
# about the tree, pinned so that a directory dropped or renamed shows up here
# rather than as a paper silently no longer checked.
PAPERS = {"fock", "jengine"}

SEED_IDS = {
    # #271: [Fock] §3's claims, declared PENDING before milestone 12 builds
    # anything; the first five are the issue's minimum, the rest the defects
    # its reading found.
    "fock/batch-magnitude-bound-direction",
    "fock/eq14-seed-inconsistent",
    "fock/eq26-ket-vrr-sign",
    "fock/eq29-screen-inequality-reversed",
    "fock/fig1-digestion-unweighted",
    "fock/fused-six-block-digest",
    "fock/predigest-bounds-atomics",
    "fock/prelaunch-screening-removes-launches",
    "fock/skip-not-break",
    "fock/symmetry-vs-screening",
    "jengine/alg1-line10-atomic-free",
    "jengine/batch-count-multiple-of-3",
    "jengine/batch-tie-break",
    "jengine/batching-trades-flops-for-intermediates",
    "jengine/eq17-1y-misprint",
    # #195: [JEngine]'s K claims, declared PENDING before any K kernel exists.
    # Neither direction of the registry check can see a claim nobody wrote
    # down; these three were such an omission, and are pinned here so they
    # cannot leave again unnoticed.
    "jengine/k-contraction-at-quartet-level",
    "jengine/k-recompute-has-more-merit",
    "jengine/k-splitting-redundant-hermite",
    "jengine/mild-keeps-rules-1-2-3-5",
    "jengine/mild-tie-break",
    "jengine/mild-vs-exact-tradeoff",
    "jengine/nine-batches-suffice",
    "jengine/rule-4-needs-the-sieve",
    "jengine/sieve-non-monotone-in-spills",
    "jengine/theta-undefined",
    "jengine/y12-example",
}


def _claim(**overrides: object) -> Claim:
    fields: dict[str, object] = {
        "id": "synthetic/c",
        "statement": "s",
        "source": "[Key, §1]",
        "kind": "shape",
        "status": "pending",
        "reason": "r",
    }
    fields.update(overrides)
    return Claim(**fields)  # type: ignore[arg-type]


# -- the registry, both ways ---------------------------------------------------


def test_every_paper_directory_is_discovered() -> None:
    # Found by listing data/literature/*/claims.json, not by a hard-coded
    # path: a second paper is checked by the same rule as the first (#271).
    found = discover(LITERATURE)
    assert set(found) == PAPERS
    assert found["jengine"] == CLAIMS
    assert set(load_registry(LITERATURE)) == PAPERS


@pytest.mark.parametrize("paper", sorted(PAPERS))
def test_every_asserted_claim_names_a_test_that_exists(
    paper: str, claim_marked_tests: dict[str, str],
) -> None:
    # Per paper: each paper's asserted records against exactly the marks that
    # carry its prefix, so adding a paper cannot weaken another's check.
    claims = load_registry(LITERATURE)[paper]
    asserted = {c.id: c.test for c in claims if c.status is ClaimStatus.ASSERTED}
    marked = {cid: node for cid, node in claim_marked_tests.items()
              if cid.startswith(f"{paper}/")}
    # both directions: an asserted claim without a marked test, and a marked
    # test without an asserted claim, each fail here
    assert set(asserted) == set(marked)
    # and the record names the very test that carries its mark
    assert asserted == marked


def test_every_claim_mark_belongs_to_a_paper(
    claim_marked_tests: dict[str, str],
) -> None:
    # The per-paper check above cannot see a mark whose prefix is no paper's.
    stray = sorted(cid for cid in claim_marked_tests
                   if cid.split("/", 1)[0] not in PAPERS)
    assert not stray


def test_seed_inventory_is_the_issue_9_list() -> None:
    # Grows as lanes add records; a removal or a rename shows up here.
    registry = load_registry(LITERATURE)
    assert SEED_IDS <= {c.id for claims in registry.values() for c in claims}


def test_no_claim_is_left_without_a_test_or_a_reason() -> None:
    for claims in load_registry(LITERATURE).values():
        for c in claims:
            if c.status is ClaimStatus.ASSERTED:
                assert c.test and not c.reason, c.id
            else:
                assert c.reason and not c.test, c.id


def test_the_registry_refuses_an_id_outside_its_paper(tmp_path: Path) -> None:
    for paper, cid in (("a", "a/x"), ("b", "a/y")):
        (tmp_path / paper).mkdir()
        (tmp_path / paper / "claims.json").write_text(
            json.dumps({"schema": SCHEMA, "claims": [_raw(cid)]}))
    assert set(discover(tmp_path)) == {"a", "b"}
    with pytest.raises(ValueError, match="not prefixed 'b/'"):
        load_registry(tmp_path)


def test_discovery_refuses_an_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no <paper>/claims.json"):
        discover(tmp_path)


def test_seed_claims_are_jengine_and_cite_it() -> None:
    for c in load_claims(CLAIMS):
        assert c.id.startswith("jengine/"), c.id
        assert c.source.startswith("[JEngine, v2, "), c.id


# -- the coverage record: the reading, held against the records (#120) ---------
#
# SEED_IDS can catch a removal and never an omission. coverage.json records the
# section-by-section reading of [JEngine, v2] instead: each unit of the paper
# lists exactly the records whose `source` cites it, so a record added without
# its unit, a unit claiming a record that does not cite it, and a unit dropped
# from the reading each fail here.

COVERAGE = CLAIMS.parent / "coverage.json"

# The structure of [JEngine, v2] (doi:10.26434/chemrxiv-2025-2lntb-v2), in the
# paper's order: a fact about the source, pinned so that the reading cannot
# quietly skip a unit. It changes only if the text of record does.
UNITS = (
    "Abstract",
    "§1", "§2", "§3.1", "§3.2", "§3.3", "§4", "§4.1", "§5", "§6",
    "§7", "§7.1", "§7.2", "§7.3", "§7.3.1", "§7.4", "§8", "§9", "§10",
    *(f"Alg. {n}" for n in range(1, 5)),
    *(f"Fig. {n}" for n in range(1, 12)),
)
N_EQS = 22  # eqs. 1-22, all in §3

_PREFIX = "[JEngine, v2, "
_SECTION = re.compile(r"§\d+(?:\.\d+)*")
_NUMBERED = re.compile(r"(Alg|Fig|eq)s?\. (\d+)(?:-(\d+))?")


def _coverage() -> list[dict]:
    return json.loads(COVERAGE.read_text())["units"]


def _eq_unit(units: list[dict]) -> dict[int, str]:
    out: dict[int, str] = {}
    for u in units:
        if "eqs" in u:
            lo, hi = u["eqs"]
            for n in range(lo, hi + 1):
                assert n not in out, f"eq. {n} in both {out[n]} and {u['unit']}"
                out[n] = u["unit"]
    return out


def cited_units(source: str, eq_unit: dict[int, str]) -> set[str]:
    """The units a `source` string cites: `§N(.M)`, `Alg. N`, `Fig(s). N(-M)`,
    `eq(s). N(-M)` (resolved to its section), joined by ', ' or ' vs '. A
    Table, or anything else unparseable, is refused: Tables are scoreboards."""
    if not (source.startswith(_PREFIX) and source.endswith("]")):
        raise ValueError(f"{source!r} does not cite [JEngine, v2, ...]")
    out: set[str] = set()
    for tok in re.split(r", | vs ", source[len(_PREFIX):-1]):
        if _SECTION.fullmatch(tok):
            out.add(tok)
            continue
        m = _NUMBERED.fullmatch(tok)
        if m is None:
            raise ValueError(f"{source!r}: cannot place {tok!r} in a unit")
        kind, lo = m[1], int(m[2])
        hi = int(m[3]) if m[3] else lo
        for n in range(lo, hi + 1):
            if kind == "eq":
                if n not in eq_unit:
                    raise ValueError(f"{source!r}: eq. {n} is in no unit")
                out.add(eq_unit[n])
            else:
                out.add(f"{kind}. {n}")
    return out


def test_the_reading_covers_every_unit_of_the_paper_in_order() -> None:
    units = _coverage()
    assert tuple(u["unit"] for u in units) == UNITS
    for u in units:
        assert set(u) <= {"unit", "claims", "note", "eqs"}, u["unit"]
        # every unit says what was read in it, claims or none
        assert isinstance(u["note"], str) and u["note"].strip(), u["unit"]
        assert u["claims"] == sorted(set(u["claims"])), u["unit"]
    # the equations partition 1..N_EQS, and only sections hold them
    eq_unit = _eq_unit(units)
    assert sorted(eq_unit) == list(range(1, N_EQS + 1))
    assert all(name.startswith("§") for name in eq_unit.values())


def test_every_record_is_in_the_reading_and_every_unit_lists_its_records() -> None:
    units = _coverage()
    eq_unit = _eq_unit(units)
    cites: dict[str, set[str]] = {name: set() for name in UNITS}
    for c in load_claims(CLAIMS):
        placed = cited_units(c.source, eq_unit)
        assert placed, c.id
        unknown = placed - set(UNITS)
        assert not unknown, f"{c.id} cites {sorted(unknown)}, not units of the paper"
        for name in placed:
            cites[name].add(c.id)
    # both directions: a record whose unit does not list it, and a unit listing
    # a record that does not cite it (or does not exist), each fail here
    assert {u["unit"]: set(u["claims"]) for u in units} == cites


def test_a_note_names_only_records_that_exist() -> None:
    ids = {c.id for c in load_claims(CLAIMS)}
    for u in _coverage():
        named = set(re.findall(r"jengine/[a-z0-9-]+[a-z0-9]", u["note"]))
        assert named <= ids, (u["unit"], sorted(named - ids))


# -- the status as the docs quote it (#116) -----------------------------------
#
# A status sentence carrying a stale tally is the failure mode this registry
# has already shown once (j_engine.md's "8 asserted" outlived its file). So
# j_engine.md carries the status table rendered from claims.json, verbatim,
# and any "N records: A asserted, U unverifiable, P pending" a doc spells is
# held to the file.

JENGINE_MD = ROOT / "docs" / "implementation" / "j_engine.md"
TALLY_DOCS = (JENGINE_MD, ROOT / "README.md", ROOT / ".claude" / "CLAUDE.md",
              CLAIMS.parent / "README.md")
_BEGIN, _END = "<!-- claims-status:begin -->", "<!-- claims-status:end -->"
_TALLY = re.compile(
    r"(\d+) records: (\d+) asserted, (\d+) unverifiable, (\d+) pending")
_ORDER = (ClaimStatus.ASSERTED, ClaimStatus.UNVERIFIABLE, ClaimStatus.PENDING)


def tally(claims) -> str:
    n = {s: sum(1 for c in claims if c.status is s) for s in _ORDER}
    return (f"{len(claims)} records: {n[ClaimStatus.ASSERTED]} asserted, "
            f"{n[ClaimStatus.UNVERIFIABLE]} unverifiable, "
            f"{n[ClaimStatus.PENDING]} pending")


def status_table(claims) -> str:
    """The block j_engine.md carries between its claims-status markers."""
    rows = [f"**{tally(claims)}** (recomputed from `claims.json` by "
            "`test/test_claims.py`).", "",
            "| status | claim | kind | held by |", "|---|---|---|---|"]
    for status in _ORDER:
        for c in (c for c in claims if c.status is status):
            held = (f"`{c.test}`" if status is ClaimStatus.ASSERTED
                    else "no test: the reason is in `claims.json`")
            rows.append(f"| {status.value} | `{c.id}` | {c.kind.value} | {held} |")
    return "\n".join(rows)


def test_j_engine_md_carries_the_status_table_verbatim() -> None:
    text = JENGINE_MD.read_text()
    assert text.count(_BEGIN) == text.count(_END) == 1
    block = text.split(_BEGIN)[1].split(_END)[0].strip()
    want = status_table(load_claims(CLAIMS))
    assert block == want, f"j_engine.md's claims table is stale; it should be:\n{want}"


def test_every_tally_a_doc_quotes_matches_the_file() -> None:
    want = tally(load_claims(CLAIMS))
    quoted = 0
    for doc in TALLY_DOCS:
        for m in _TALLY.finditer(doc.read_text()):
            quoted += 1
            assert m[0] == want, (doc.relative_to(ROOT).as_posix(), m[0], want)
    assert quoted  # the check is not vacuous: j_engine.md quotes one


# [Fock]'s tally (#291) is spelled with its key, "N [Fock] records: A
# asserted, U unverifiable, P pending", so it can never be read as the
# [JEngine] tally above (whose pattern needs the digits right before
# "records:"), and any doc that quotes one is held to the fock file. No doc
# is required to quote it: data/literature/fock/README.md says to recompute.

FOCK_CLAIMS = LITERATURE / "fock" / "claims.json"
FOCK_TALLY_DOCS = (*TALLY_DOCS, FOCK_CLAIMS.parent / "README.md",
                   *sorted((ROOT / "docs" / "implementation").glob("*.md")))
_FOCK_TALLY = re.compile(r"(\d+) `?\[Fock\]`? records: (\d+) asserted, "
                         r"(\d+) unverifiable, (\d+) pending")


def test_every_fock_tally_a_doc_quotes_matches_the_file() -> None:
    want = tally(load_claims(FOCK_CLAIMS))
    for doc in FOCK_TALLY_DOCS:
        for m in _FOCK_TALLY.finditer(_flat(doc)):
            got = f"{m[1]} records: {m[2]} asserted, {m[3]} unverifiable, " \
                  f"{m[4]} pending"
            assert got == want, (doc.relative_to(ROOT).as_posix(), m[0], want)
    # and the pattern does read the spelling it documents
    assert _FOCK_TALLY.fullmatch(want.replace(" records", " [Fock] records"))


# The same failure mode for limits.json: a doc that says "holds six
# `StructuralLimit`s" outlives the record that made it seven (#148). Every such
# count a doc spells, in words or digits, is held to the file.

LIMITS = CLAIMS.parent / "limits.json"
_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve "
          "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
          "twenty").split()
_TENS = {"twenty": 20, "thirty": 30}
# Hyphens are read with the word, so "twenty-one" is 21 and not "one" (#447).
_LIMIT_COUNT = re.compile(r"(?<![\w-])([\w-]+) `StructuralLimit`s")


def _count_word(word: str) -> int:
    """A count spelled in digits, as a word to twenty, or as a hyphenated
    compound ("twenty-one"); anything else raises, so a spelling this
    reader does not know fails loudly instead of being skipped."""
    word = word.lower()
    if word.isdigit():
        return int(word)
    if word in _WORDS:
        return _WORDS.index(word)
    tens, _, unit = word.partition("-")
    return _TENS[tens] + _WORDS.index(unit) if unit else _TENS[tens]


def test_the_count_reader_reads_compounds() -> None:
    assert [_count_word(w) for w in
            ("7", "nineteen", "Twenty", "twenty-one", "thirty-nine")] == \
        [7, 19, 20, 21, 39]
    assert _LIMIT_COUNT.search("holds twenty-one `StructuralLimit`s")[1] == \
        "twenty-one"


def test_every_limits_count_a_doc_spells_matches_the_file() -> None:
    n = len(json.loads(LIMITS.read_text())["limits"])
    quoted = 0
    for doc in TALLY_DOCS:
        for m in _LIMIT_COUNT.finditer(doc.read_text()):
            got = _count_word(m[1])
            quoted += 1
            assert got == n, (doc.relative_to(ROOT).as_posix(), m[0], n)
    assert quoted  # CLAUDE.md and the data README spell one each


# #174: jengine/sieve-underspecified's '~2%' stays UNVERIFIABLE after #172's
# wider search, and its reason now reads that search. What it says of the
# searched space -- the printed cells are all reached (so many exactly, the
# rest under), including every cell where the printed FLOPs lie below all 58
# rules -- is recomputed here from searched_cuts.json, the tables and
# seed_order_spread.json, and every doc that quotes the pass is held to it.
# (test_plan_search.py holds searched_cuts.json to the replay.)

SEARCHED_CUTS = ROOT / "data" / "scoreboards" / "jengine" / "searched_cuts.json"
SPREAD = CLAIMS.parent / "seed_order_spread.json"
_PASS_DOCS = (JENGINE_MD, ROOT / ".claude" / "CLAUDE.md")


def _flat(path: Path) -> str:
    return " ".join(path.read_text().split())


def test_the_sieve_claim_reads_the_searched_space() -> None:
    from felizjueves.literature.tables import load_table

    tables = [load_table(CLAIMS.parent / f"table{i}.json") for i in (1, 2, 3)]
    selected = {}
    for key, cell in json.loads(SEARCHED_CUTS.read_text())["cuts"].items():
        L, nb = map(int, key.split("/"))
        p = cell["frontier"][cell["selected"]]
        selected[L, nb] = (p["flops"], p["max_intermediates"], p["max_targets"])
    assert set(selected) == set(tables[0].cells)
    printed = {c: tuple(t.cells[c] for t in tables) for c in selected}
    assert all(s <= p for c in selected
               for s, p in zip(selected[c], printed[c], strict=True))
    exact = sum(selected[c] == printed[c] for c in selected)
    under = len(selected) - exact

    spread = json.loads(SPREAD.read_text())
    n_rules = len(spread["orders"])
    rows = {tuple(r["cell"]): r for r in spread["rows"]}
    below = [tuple(c) for c in spread["summary"]["flops"]["printed_outside_range"]]
    assert below
    for c in below:
        assert printed[c][0] < rows[c]["flops"]["min"], c  # below every rule
        assert selected[c][0] <= printed[c][0], c           # a climbed order reaches it

    (claim,) = [c for c in load_claims(CLAIMS)
                if c.id == "jengine/sieve-underspecified"]
    assert claim.status is ClaimStatus.UNVERIFIABLE
    reason = " ".join(claim.reason.split())
    assert "searched_cuts.json" in reason
    assert ("test/test_claims.py::test_the_sieve_claim_reads_the_searched_space"
            in reason)
    assert (f"At all {len(below)} cells where the printed FLOPs lie below "
            f"every one of the {n_rules} rules") in reason
    assert (f"in all {len(selected)} cells ({exact} exactly, {under} under)"
            in reason)
    mild = _flat(ROOT / "docs" / "implementation" / "mild.md")
    assert (f"At all {len(below)} cells where the printed FLOPs lie below "
            f"every one of the {n_rules} rules") in mild
    for doc in _PASS_DOCS:
        assert (f"in all {len(selected)} cells: {exact} land on the printed "
                f"triple exactly and {under} come in under it") in _flat(doc), doc
    assert (f"The {exact} and {under} are recomputed from `searched_cuts.json`"
            in _flat(JENGINE_MD))


# #167: milestone 6 (#148) stated parity with `jgen` with two qualifications,
# Table 4 declined and the tuned-plan wire absent. The wire now exists (#161)
# and was refused as the default on measured wall-clock (#166), so every doc
# that states the parity names ONE difference and the decision, and none
# carries the two-item wording any more. Both overclaims are refused: no doc
# may say the repo replicates `jgen`, and none may call the plan a gap.

_PARITY_DOCS = (JENGINE_MD, ROOT / "README.md", ROOT / ".claude" / "CLAUDE.md")
_PLAN_DOCS = (*_PARITY_DOCS, ROOT / "docs" / "implementation" / "mild.md",
              CLAIMS.parent / "README.md")
_STALE_PARITY = (
    "except the tuned-plan wire", "both qualifications",
    "in exactly two places", "literal plan, not the searched one",
    "reaches the emitter is the literal one", "the tuned-plan wire `jgen` has",
)


def test_the_jgen_difference_is_one_item_and_the_plan_a_decision() -> None:
    for doc in _PARITY_DOCS:
        text = _flat(doc)
        name = doc.relative_to(ROOT).as_posix()
        assert "with one difference: Table 4, declined" in text, name
        assert "#166" in text and "tuned" in text, name
    for doc in _PLAN_DOCS:
        text = _flat(doc)
        name = doc.relative_to(ROOT).as_posix()
        for stale in _STALE_PARITY:
            assert stale not in text, (name, stale)
        assert "replicates `jgen`" not in text.replace(
            '"this repo replicates `jgen`" is not true', ""), name
    # That the decision those sentences cite is the one the tracked store
    # reads is a check of the measured rows: measure/measure_claims.py.


@pytest.mark.parametrize(
    "source, units",
    [
        pytest.param("[JEngine, v2, §5]", {"§5"}, id="section"),
        pytest.param("[JEngine, v2, Alg. 1 vs §5]", {"Alg. 1", "§5"}, id="vs"),
        pytest.param("[JEngine, v2, eq. 13]", {"§3.2"}, id="eq"),
        pytest.param("[JEngine, v2, eqs. 20-22]", {"§3.3"}, id="eq-range"),
        pytest.param(
            "[JEngine, v2, §5, Figs. 5-6]", {"§5", "Fig. 5", "Fig. 6"}, id="figs"
        ),
    ],
)
def test_a_source_resolves_to_its_units(source: str, units: set[str]) -> None:
    assert cited_units(source, _eq_unit(_coverage())) == units


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("[JEngine, v2, Table 1]", id="table"),
        pytest.param("[JEngine, v2, eq. 99]", id="eq-out-of-range"),
        pytest.param("[JEngine, v2, p. 4]", id="unplaceable"),
        pytest.param("[JEngine, v1, §5]", id="not-v2"),
    ],
)
def test_a_source_that_cannot_be_placed_is_refused(source: str) -> None:
    with pytest.raises(ValueError):
        cited_units(source, _eq_unit(_coverage()))


# -- the record type -----------------------------------------------------------


def test_json_values_coerce_to_the_enums() -> None:
    c = _claim()
    assert c.kind is ClaimKind.SHAPE and c.status is ClaimStatus.PENDING
    assert c == _claim(kind=ClaimKind.SHAPE, status=ClaimStatus.PENDING)


def test_a_claim_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _claim().reason = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"status": "asserted", "reason": ""}, id="asserted-no-test"),
        pytest.param(
            {"status": "asserted", "test": "test/t.py::t"}, id="asserted-with-reason"
        ),
        pytest.param({"reason": ""}, id="pending-no-reason"),
        pytest.param({"reason": "  "}, id="pending-blank-reason"),
        pytest.param({"test": "test/t.py::t"}, id="pending-with-test"),
        pytest.param({"status": "unverifiable", "reason": ""}, id="unverif-no-reason"),
        pytest.param(
            {"status": "unverifiable", "test": "test/t.py::t"},
            id="unverif-with-test",
        ),
        pytest.param({"id": ""}, id="no-id"),
        pytest.param({"statement": " "}, id="no-statement"),
        pytest.param({"source": ""}, id="no-source"),
        pytest.param({"kind": "vibe"}, id="unknown-kind"),
        pytest.param({"status": "done"}, id="unknown-status"),
    ],
)
def test_the_status_pairing_is_enforced(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _claim(**overrides)


def test_each_status_accepts_its_pairing() -> None:
    _claim(status="asserted", reason="", test="test/t.py::t")
    _claim(status="pending", reason="r")
    _claim(status="unverifiable", reason="r")


# -- the loader ----------------------------------------------------------------


def _write(tmp_path: Path, claims: list[dict[str, object]], schema: int = SCHEMA):
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({"schema": schema, "claims": claims}))
    return path


def _raw(cid: str) -> dict[str, object]:
    return {
        "id": cid,
        "statement": "s",
        "source": "[Key, §1]",
        "kind": "shape",
        "status": "pending",
        "reason": "r",
    }


def test_loader_round_trips(tmp_path: Path) -> None:
    got = load_claims(_write(tmp_path, [_raw("a/x"), _raw("a/y")]))
    assert [c.id for c in got] == ["a/x", "a/y"]


def test_loader_rejects_a_wrong_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        load_claims(_write(tmp_path, [], schema=SCHEMA + 1))


def test_loader_rejects_a_duplicate_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_claims(_write(tmp_path, [_raw("a/x"), _raw("a/x")]))


def test_loader_rejects_an_unordered_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ordered"):
        load_claims(_write(tmp_path, [_raw("a/y"), _raw("a/x")]))


def test_loader_rejects_an_unknown_field(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        load_claims(_write(tmp_path, [{**_raw("a/x"), "owner": "me"}]))


# -- the collector -------------------------------------------------------------


def _tests(tmp_path: Path, source: str) -> Path:
    d = tmp_path / "test"
    d.mkdir()
    (d / "test_x.py").write_text(source)
    return d


def test_collector_finds_functions_and_methods(tmp_path: Path) -> None:
    d = _tests(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.claim('a/one')\n"
        "def test_one(): pass\n"
        "class TestK:\n"
        "    @pytest.mark.slow\n"
        "    @pytest.mark.claim('a/two')\n"
        "    def test_two(self): pass\n"
        "def test_unmarked(): pass\n",
    )
    assert scan_claim_marks(d) == {
        "a/one": "test/test_x.py::test_one",
        "a/two": "test/test_x.py::TestK::test_two",
    }


@pytest.mark.parametrize(
    "source, match",
    [
        pytest.param(
            "import pytest\n"
            "@pytest.mark.claim('a/one')\ndef test_a(): pass\n"
            "@pytest.mark.claim('a/one')\ndef test_b(): pass\n",
            "both",
            id="same-id-twice",
        ),
        pytest.param(
            "import pytest\nID = 'a/one'\n"
            "@pytest.mark.claim(ID)\ndef test_a(): pass\n",
            "literal",
            id="non-literal-id",
        ),
        pytest.param(
            "import pytest\n@pytest.mark.claim()\ndef test_a(): pass\n",
            "literal",
            id="no-id",
        ),
        pytest.param(
            "import pytest\npytestmark = pytest.mark.claim('a/one')\n",
            "not on a test function",
            id="module-pytestmark",
        ),
        pytest.param(
            "import pytest\n@pytest.mark.claim('a/one')\nclass TestK:\n"
            "    def test_a(self): pass\n",
            "not on a test function",
            id="on-a-class",
        ),
    ],
)
def test_collector_rejects_an_unpinnable_mark(
    tmp_path: Path, source: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        scan_claim_marks(_tests(tmp_path, source))
