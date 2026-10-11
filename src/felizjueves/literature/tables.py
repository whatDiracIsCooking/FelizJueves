"""A table transcribed from a paper: its cells, and where they came from.

The JSON files under data/literature/ are the record; this module only reads
them and refuses a malformed one. A cell is keyed by (row value, column value)
-- for [JEngine] Tables 1-3, (L, batch count) -- never by position, so a
consumer cannot silently read the wrong row of a table that grows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA = 1

Cell = tuple[int, int]


@dataclass(frozen=True, eq=False)
class Table:
    id: str
    source: Mapping[str, Any]
    quantity: Mapping[str, Any]
    row_name: str
    row_values: tuple[int, ...]
    column_name: str
    column_values: tuple[int, ...]
    cells: Mapping[Cell, int]
    notes: tuple[str, ...]


def load_table(path: Path) -> Table:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {raw.get('schema')!r}, expected {SCHEMA}")
    rows = tuple(raw["rows"]["values"])
    cols = tuple(raw["columns"]["values"])
    grid = raw["cells"]
    if len(grid) != len(rows) or any(len(r) != len(cols) for r in grid):
        raise ValueError(f"{path}: cells are not {len(rows)} x {len(cols)}")
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        raise ValueError(f"{path}: duplicate row or column value")
    cells: dict[Cell, int] = {}
    for r, line in zip(rows, grid, strict=True):
        for c, v in zip(cols, line, strict=True):
            if type(v) is not int:
                raise ValueError(f"{path}: cell ({r}, {c}) = {v!r} is not an int")
            cells[(r, c)] = v
    return Table(
        id=raw["id"],
        source=MappingProxyType(raw["source"]),
        quantity=MappingProxyType(raw["quantity"]),
        row_name=raw["rows"]["name"],
        row_values=rows,
        column_name=raw["columns"]["name"],
        column_values=cols,
        cells=MappingProxyType(cells),
        notes=tuple(raw["notes"]),
    )
