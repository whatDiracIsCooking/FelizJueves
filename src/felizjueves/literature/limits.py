"""What a model will structurally be unable to predict -- recorded as data.

When a cost model cannot, by construction, reproduce a published number (a
measured timing depends on hardware the model does not describe, say), that is
a fact about the model and it belongs in a record, not in a test named
"cannot predict X": nobody maintains such a test from memory. A record says
what is claimed, why the model cannot reach it, what shows that, the value
taken from the source in its place, and which source.

Nothing fills this yet; the record type is here so the first such finding has
somewhere to go.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 1


@dataclass(frozen=True)
class StructuralLimit:
    claim: str  # what the model does not predict
    mechanism: str  # why, structurally, it cannot
    evidence: str  # what shows it
    asserted_value: object  # the value taken from the source instead
    source: str  # citation: [Key, table/eq/section]

    def __post_init__(self) -> None:
        for name in ("claim", "mechanism", "evidence", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"StructuralLimit.{name} must be non-empty text")


def load_limits(path: Path) -> tuple[StructuralLimit, ...]:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {raw.get('schema')!r}, expected {SCHEMA}")
    return tuple(StructuralLimit(**entry) for entry in raw["limits"])
