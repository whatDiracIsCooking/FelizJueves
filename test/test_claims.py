"""The claims registry: the record type, the seed inventory, and the check that
keeps the inventory and the claim-marked tests in step, in both directions.

Nothing here tests a claim -- each ASSERTED record is discharged by the
marked test it names, elsewhere under test/. What is tested is that the inventory
cannot rot: an ASSERTED record must name a test that exists and carries its
mark, a marked test must discharge a registered record, and a record that is
not asserted must say why.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from conftest import scan_claim_marks

from felizjueves.literature.claims import (
    SCHEMA,
    Claim,
    ClaimKind,
    ClaimStatus,
    load_claims,
)

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "data" / "literature" / "jengine" / "claims.json"

SEED_IDS = {
    "jengine/alg1-line10-atomic-free",
    "jengine/batch-count-multiple-of-3",
    "jengine/batch-tie-break",
    "jengine/batching-trades-flops-for-intermediates",
    "jengine/eq17-1y-misprint",
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


def test_every_asserted_claim_names_a_test_that_exists(
    claim_marked_tests: dict[str, str],
) -> None:
    claims = load_claims(CLAIMS)
    asserted = {c.id: c.test for c in claims if c.status is ClaimStatus.ASSERTED}
    # both directions: an asserted claim without a marked test, and a marked
    # test without an asserted claim, each fail here
    assert set(asserted) == set(claim_marked_tests)
    # and the record names the very test that carries its mark
    assert asserted == claim_marked_tests


def test_seed_inventory_is_the_issue_9_list() -> None:
    # Grows as lanes add records; a removal or a rename shows up here.
    assert SEED_IDS <= {c.id for c in load_claims(CLAIMS)}


def test_no_claim_is_left_without_a_test_or_a_reason() -> None:
    for c in load_claims(CLAIMS):
        if c.status is ClaimStatus.ASSERTED:
            assert c.test and not c.reason, c.id
        else:
            assert c.reason and not c.test, c.id


def test_seed_claims_are_jengine_and_cite_it() -> None:
    for c in load_claims(CLAIMS):
        assert c.id.startswith("jengine/"), c.id
        assert c.source.startswith("[JEngine, v2, "), c.id


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
