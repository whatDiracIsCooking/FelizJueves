"""The source's non-numeric claims -- recorded as data, each with its status.

`tables` holds the published numbers and `rules` how to judge a candidate
against them; `limits` holds what a model structurally cannot predict. None of
them holds what the source asserts WITHOUT a number: a worked example's
membership, a stated tie-break, a split's shape, a direction of change, or a
defect in the source itself. Those are the checks that catch a reconstruction
that is structurally wrong and still scores well on totals, and they are
exactly the claims nobody maintains from memory -- so they are inventory, here,
rather than folklore.

A record's status says what holds us to it. ASSERTED names the test that does;
PENDING and UNVERIFIABLE name the reason none does. The pairing is enforced on
construction, and `test/test_claims.py` checks the registry against the
claim-marked tests in both directions, so neither side can drift alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SCHEMA = 1


class ClaimKind(StrEnum):
    MEMBERSHIP = "membership"  # a named set's members
    ORDERING = "ordering"  # a tie-break, a sort, a seed order
    SHAPE = "shape"  # a split's structure, a figure
    MONOTONICITY = "monotonicity"  # a direction, not a value
    DEFECT = "defect"  # the source states something wrong


class ClaimStatus(StrEnum):
    ASSERTED = "asserted"  # a test holds us to it
    PENDING = "pending"  # not testable yet; reason required
    UNVERIFIABLE = "unverifiable"  # out of this repo's reach; reason required


@dataclass(frozen=True)
class Claim:
    id: str  # "jengine/y12-example"
    statement: str  # what the source asserts, in OUR vocabulary
    source: str  # "[JEngine, v2, §5]"
    kind: ClaimKind
    status: ClaimStatus
    test: str = ""  # the test that holds us to it; ASSERTED requires it
    reason: str = ""  # why not asserted; PENDING/UNVERIFIABLE require it

    def __post_init__(self) -> None:
        for name in ("id", "statement", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Claim.{name} must be non-empty text")
        # Coerce, so a record built from JSON and one built in code compare
        # equal; an unknown value raises ValueError naming it.
        object.__setattr__(self, "kind", ClaimKind(self.kind))
        object.__setattr__(self, "status", ClaimStatus(self.status))
        for name in ("test", "reason"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"Claim.{name} must be text")
        has_test = bool(self.test.strip())
        has_reason = bool(self.reason.strip())
        if self.status is ClaimStatus.ASSERTED:
            if not has_test:
                raise ValueError(f"{self.id}: ASSERTED requires the test")
            if has_reason:
                raise ValueError(f"{self.id}: ASSERTED forbids a reason")
        else:
            if not has_reason:
                raise ValueError(f"{self.id}: {self.status.name} requires a reason")
            if has_test:
                raise ValueError(f"{self.id}: {self.status.name} forbids a test")


def load_claims(path: Path) -> tuple[Claim, ...]:
    """Load a claims file; reject a duplicate id or an unordered file.

    Ordered by id because several lanes append to the same file: an ordered
    file turns each addition into a one-record diff and a merge into a
    position, not a conflict over who appended last.
    """
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {raw.get('schema')!r}, expected {SCHEMA}")
    claims = tuple(Claim(**entry) for entry in raw["claims"])
    ids = [c.id for c in claims]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{path}: duplicate claim ids {dupes}")
    if ids != sorted(ids):
        raise ValueError(f"{path}: claims must be ordered by id")
    return claims
