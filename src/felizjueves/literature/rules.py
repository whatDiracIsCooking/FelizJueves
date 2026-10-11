"""Scoring rules: declared objects that judge a candidate against one table.

Three rules, each a frozen value that carries its own thresholds:

- `Agreement`        exact-cell count >= floor, and worst relative deviation
                     <= ceiling.
- `AtOrUnder`        the candidate is at or under the printed value in every
                     cell; reports by how much it is over, but never judges
                     the amount.
- `ExactGivenInputs` every cell exact -- but only for a candidate produced
                     from the recorded inputs; any other candidate is
                     INAPPLICABLE, not a failure.

A THRESHOLD NOBODY HAS MEASURED IS UNSET, NOT GUESSED. A slot left as None
makes the rule report what it measured with status UNSET: neither a pass nor a
fail. The first change that produces numbers fills the slot with what it
measured; after that a floor only rises and a ceiling only falls
(`Agreement.tightened` refuses the other direction), each move in its own
commit naming the cells that moved.

Equal is passing: a candidate exactly at the floor or the ceiling passes, so a
ratchet set to a measured value accepts that same measurement on the next run.

A candidate must cover exactly the table's cells. A missing or extra cell is a
FAIL, whatever the thresholds, because a partial run cannot be told apart from
a regression.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from felizjueves.literature.tables import Cell, Table


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNSET = "unset"  # a threshold this rule needs has not been measured yet
    INAPPLICABLE = "inapplicable"  # the candidate is outside the rule's scope


@dataclass(frozen=True, eq=False)
class Candidate:
    """Values for a table's cells, which scoreboard they were produced for, and
    the inputs that produced them.

    `board` is ownership, `inputs` is method: a candidate produced by the
    literal reconstruction says ``board="literal"``, and `score` refuses to
    judge it on any other board. The rules themselves ignore `board`; the
    check lives in `score`, the one entry point a scoreboard is read through.
    """

    cells: Mapping[Cell, float]
    board: str = field(kw_only=True)
    inputs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    rule: str
    table: str
    status: Status
    reason: str
    measured: Mapping[str, Any] = field(default_factory=dict)
    cells: tuple[Cell, ...] = ()  # the cells that decided a FAIL


def _coverage(name: str, table: Table, candidate: Candidate) -> Verdict | None:
    want, got = set(table.cells), set(candidate.cells)
    if want == got:
        return None
    off = tuple(sorted(want ^ got))
    return Verdict(
        name,
        table.id,
        Status.FAIL,
        f"{len(want - got)} missing and {len(got - want)} extra cells",
        cells=off,
    )


def _rel_dev(table: Table, cell: Cell, value: float) -> float:
    printed = table.cells[cell]
    if printed == 0:
        raise ValueError(
            f"{table.id} cell {cell} is printed as 0, so a relative deviation "
            "from it is undefined; score that table with a rule that does not "
            "divide by the printed value"
        )
    return abs(value - printed) / abs(printed)


@dataclass(frozen=True)
class Agreement:
    floor_exact: int | None = None
    ceiling_worst_rel_dev: float | None = None
    name: str = field(default="agreement", init=False)

    def judge(self, table: Table, candidate: Candidate) -> Verdict:
        if (bad := _coverage(self.name, table, candidate)) is not None:
            return bad
        devs = {c: _rel_dev(table, c, candidate.cells[c]) for c in table.cells}
        exact = sum(1 for c, p in table.cells.items() if candidate.cells[c] == p)
        worst = max(devs.values())
        measured = {"exact": exact, "worst_rel_dev": worst}
        failed = []
        if self.floor_exact is not None and exact < self.floor_exact:
            failed.append(f"{exact} exact cells < floor {self.floor_exact}")
        if (
            self.ceiling_worst_rel_dev is not None
            and worst > self.ceiling_worst_rel_dev
        ):
            failed.append(
                f"worst relative deviation {worst:.6g} > ceiling "
                f"{self.ceiling_worst_rel_dev:.6g}"
            )
        if failed:
            inexact = tuple(sorted(c for c, d in devs.items() if d > 0))
            return Verdict(
                self.name, table.id, Status.FAIL, "; ".join(failed), measured, inexact
            )
        if self.floor_exact is None or self.ceiling_worst_rel_dev is None:
            return Verdict(
                self.name, table.id, Status.UNSET, "threshold not yet measured",
                measured,
            )
        return Verdict(self.name, table.id, Status.PASS, "within thresholds", measured)

    def tightened(
        self, floor_exact: int | None = None, ceiling_worst_rel_dev: float | None = None
    ) -> Agreement:
        """A copy with a higher floor and/or lower ceiling; never the reverse."""
        floor, ceiling = self.floor_exact, self.ceiling_worst_rel_dev
        if floor_exact is not None:
            if floor is not None and floor_exact < floor:
                raise ValueError(f"floor may only rise: {floor} -> {floor_exact}")
            floor = floor_exact
        if ceiling_worst_rel_dev is not None:
            if ceiling is not None and ceiling_worst_rel_dev > ceiling:
                raise ValueError(
                    f"ceiling may only fall: {ceiling} -> {ceiling_worst_rel_dev}"
                )
            ceiling = ceiling_worst_rel_dev
        return Agreement(floor, ceiling)


def _overshoot(printed: float, value: float) -> float:
    """(value - printed) / |printed|; inf for a cell printed as 0, where any
    value over it is over by an unbounded fraction."""
    return (value - printed) / abs(printed) if printed else math.inf


@dataclass(frozen=True)
class AtOrUnder:
    """At or under the printed value in every cell. Nothing to ratchet.

    The MARGIN is reported, never judged: `measured` carries the worst
    relative overshoot across the table (0.0 when no cell is over) and, per
    cell over, its overshoot, in the same order as `cells`. There is no
    tolerance: a cell over by any amount is a FAIL. A margin is evidence for
    whatever discharges the FAIL (a StructuralLimit), not a way past it.
    """

    name: str = field(default="at_or_under", init=False)

    def judge(self, table: Table, candidate: Candidate) -> Verdict:
        if (bad := _coverage(self.name, table, candidate)) is not None:
            return bad
        over = tuple(
            sorted(c for c, p in table.cells.items() if candidate.cells[c] > p)
        )
        overshoot = tuple(
            (c, _overshoot(table.cells[c], candidate.cells[c])) for c in over
        )
        measured = {
            "over": len(over),
            "worst_overshoot": max((o for _, o in overshoot), default=0.0),
            "overshoot": overshoot,
        }
        if over:
            return Verdict(
                self.name, table.id, Status.FAIL,
                f"{len(over)} cells over the printed value", measured, over,
            )
        return Verdict(
            self.name, table.id, Status.PASS, "at or under in every cell", measured
        )


@dataclass(frozen=True, eq=False)
class ExactGivenInputs:
    # The inputs the paper's numbers were produced from (a batch split, say).
    # None until someone records them.
    inputs: Mapping[str, Any] | None = None
    name: str = field(default="exact_given_inputs", init=False)

    def __post_init__(self) -> None:
        if self.inputs is not None:
            object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))

    def judge(self, table: Table, candidate: Candidate) -> Verdict:
        if self.inputs is None:
            return Verdict(
                self.name, table.id, Status.UNSET, "recorded inputs not yet entered"
            )
        if dict(candidate.inputs) != dict(self.inputs):
            return Verdict(
                self.name, table.id, Status.INAPPLICABLE,
                "candidate was not produced from the recorded inputs",
            )
        if (bad := _coverage(self.name, table, candidate)) is not None:
            return bad
        wrong = tuple(
            sorted(c for c, p in table.cells.items() if candidate.cells[c] != p)
        )
        measured = {"inexact": len(wrong)}
        if wrong:
            return Verdict(
                self.name, table.id, Status.FAIL,
                f"{len(wrong)} cells differ from the printed value", measured, wrong,
            )
        return Verdict(self.name, table.id, Status.PASS, "every cell exact", measured)


# A closed set: a new kind of judgement is a new class here, added by name.
Rule = Agreement | AtOrUnder | ExactGivenInputs
