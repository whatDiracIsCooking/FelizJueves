"""What a model will structurally be unable to predict -- recorded as data.

When a cost model cannot, by construction, reproduce a published number (a
measured timing depends on hardware the model does not describe, say), that is
a fact about the model and it belongs in a record, not in a test named
"cannot predict X": nobody maintains such a test from memory. A record says
what is claimed, why the model cannot reach it, what shows that, the value
taken from the source in its place, and which source.

Two files hold records, each read with `load_limits` and each recomputed by the
tests that cite it:

- data/literature/jengine/limits.json: the `searched` board's residual against
  `[JEngine, v2, Tables 1-2]` (test/test_scoreboards.py, test_plan_search.py,
  test_plan_recovery.py);
- data/measured/limits.json: where `lower.KernelCost` mis-ranks the batch
  counts of `measured.SUBGRID` against the Measured rows, and where spill wins
  do not pick the fastest count (#54, #60; test/test_kernelrank.py). A record
  there is read by tests only and never feeds a literature floor.
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
