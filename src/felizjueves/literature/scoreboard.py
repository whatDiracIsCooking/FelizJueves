"""A scoreboard: which rules, with which thresholds, apply to which tables.

TWO SCOREBOARDS, AND NEITHER MAY MOVE THE OTHER. `literal` will score the
paper's algorithms run as written, with the choices the paper leaves open fixed
arbitrarily; `searched` will score those same choices searched. Both read the
same tables. What keeps them apart, and exactly how far it reaches:

- THRESHOLDS. Each board's thresholds live in its own file, and
  `load_scoreboard` refuses a file that declares a different board than the
  one asked for. A rule is a frozen value: raising a floor makes a new rule
  (`Agreement.tightened`), and nothing here writes it anywhere -- writing it
  back to the board's own file is a deliberate, committed edit.
- CANDIDATES. A `Candidate` declares the board it was produced for, and
  `score` raises rather than judge one on another board. That check lives in
  `score` only: a rule's own `judge` does not look at `board`, so calling a
  rule directly bypasses it. Scoring goes through `score`.
- TABLES. They are read-only (`load_table` hands out mapping proxies).

The tables are passed in, not looked up: a board names the table ids it
scores, and the caller supplies those tables. A table the board names but the
caller did not supply is an error naming it; a missing candidate is a FAIL
verdict, like a missing cell, because a partial run cannot be told apart from
a regression.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from felizjueves.literature.rules import (
    Agreement,
    AtOrUnder,
    Candidate,
    ExactGivenInputs,
    Rule,
    Status,
    Verdict,
)
from felizjueves.literature.tables import Table

SCHEMA = 1

_RULES: dict[str, type[Rule]] = {
    "agreement": Agreement,
    "at_or_under": AtOrUnder,
    "exact_given_inputs": ExactGivenInputs,
}


@dataclass(frozen=True, eq=False)
class Scoreboard:
    name: str
    rules: Mapping[str, tuple[Rule, ...]]  # table id -> rules applied to it


def load_scoreboard(path: Path, name: str) -> Scoreboard:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {raw.get('schema')!r}, expected {SCHEMA}")
    if raw.get("scoreboard") != name:
        raise ValueError(
            f"{path} is scoreboard {raw.get('scoreboard')!r}, not {name!r}"
        )
    rules: dict[str, tuple[Rule, ...]] = {}
    for table_id, entries in raw["tables"].items():
        built = []
        for entry in entries:
            kwargs = dict(entry)
            kind = kwargs.pop("rule")
            if kind not in _RULES:
                raise ValueError(f"{path}: unknown rule {kind!r}")
            built.append(_RULES[kind](**kwargs))
        rules[table_id] = tuple(built)
    return Scoreboard(name, MappingProxyType(rules))


def score(
    board: Scoreboard,
    tables: Mapping[str, Table],
    candidates: Mapping[str, Candidate],
) -> tuple[Verdict, ...]:
    """Judge each candidate by every rule its table carries on this board."""
    foreign = sorted(
        f"{tid} (board {c.board!r})"
        for tid, c in candidates.items()
        if c.board != board.name
    )
    if foreign:
        raise ValueError(
            f"scoreboard {board.name!r} refuses candidates produced for another "
            f"board: {', '.join(foreign)}"
        )
    absent = sorted(set(board.rules) - set(tables))
    if absent:
        raise ValueError(
            f"scoreboard {board.name!r} scores tables that were not supplied: "
            f"{', '.join(absent)}"
        )
    verdicts: list[Verdict] = []
    for table_id, rules in board.rules.items():
        table = tables[table_id]
        candidate = candidates.get(table_id)
        if candidate is None:
            verdicts.extend(
                Verdict(rule.name, table_id, Status.FAIL, "no candidate supplied")
                for rule in rules
            )
            continue
        verdicts.extend(rule.judge(table, candidate) for rule in rules)
    return tuple(verdicts)
