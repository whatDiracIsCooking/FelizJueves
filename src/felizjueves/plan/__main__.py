"""`python -m felizjueves.plan`: replay a scoreboard, or print one plan's report.

The project is a virtual uv project -- nothing is installed -- so `src/` has to
be put on the path by hand:

    PYTHONPATH=src uv run python -m felizjueves.plan replay literal
    PYTHONPATH=src uv run python -m felizjueves.plan replay searched
    PYTHONPATH=src uv run python -m felizjueves.plan report 12 6 [--no-sieve]
    PYTHONPATH=src uv run python -m felizjueves.plan search [--workers N]

REPLAY builds every cell the board's tables hold with `pipeline.plan`, scores
the candidates through `scoreboard.score`, and prints one verdict per
(table, rule). Host-side counting only: nothing compiles, nothing launches.

SEARCH is the searched board's build step (#172): it runs plan/search.py's
climb over every cell of the table grid, its climbs spread over `--workers`
processes (the record does not depend on how many), writes
searched_cuts.json, and prints its wall-clock. It judges nothing; run
`replay searched` after it. Exit 0 written, 2 could not run.

REPORT prints one Spec's `Plan.report()` as JSON: the cost and the per-batch
FLOP, intermediate and target counts.

EXIT STATUS, so that a script cannot read "nothing to judge" as a pass -- the
same rule as kernel-tier.sh's "never read a 2 as a pass":

    0  every verdict PASS (INAPPLICABLE is neutral: the rule does not apply)
    1  at least one FAIL
    2  could not run (bad arguments, or an input this reader cannot read)
    3  nothing FAILed, but something is UNSET -- or the board has no
       candidates at all yet

FAIL outranks UNSET. `report` judges nothing, and exits 0 or 2. Today
`replay literal` exits 3: every Agreement passes, Table 3's
exact_given_inputs passes from its recorded inputs (#14), and Tables 1-2's
are UNSET with a recorded refusal (#108, the rules-only doctrine): no rule
recovers the sieve's seed order, a per-cell fit is not admitted, and a
refusal scores UNSET, never PASS (literal.json's `refused`, and the evidence
in data/literature/jengine/limits.json). Table 3's Agreement line reads 40
exact because its candidate is the recovered one; its floor (38) is pinned to
the minimax reconstruction by test/test_scoreboards.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from felizjueves.literature.rules import Candidate, Status, Verdict
from felizjueves.literature.scoreboard import load_scoreboard, score
from felizjueves.literature.tables import Cell, Table, load_table
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.pipeline import plan
from felizjueves.plan.recovery import recovered_candidates
from felizjueves.plan.search import (
    CLIMB_BUDGET,
    Choice,
    build,
    dump_cuts,
    load_cuts,
    searched_plan,
)
from felizjueves.plan.sieve import NO_SIEVE
from felizjueves.spec import Operator, Spec, Target

EXIT_PASS, EXIT_FAIL, EXIT_COULD_NOT_RUN, EXIT_UNSET = 0, 1, 2, 3

DATA = Path(__file__).resolve().parents[3] / "data"
BOARDS = ("literal", "searched")

# Which PlanCost field each [JEngine] table prints.
QUANTITY = {
    "jengine/table1": "flops",
    "jengine/table2": "max_intermediates",
    "jengine/table3": "max_targets",
}
T3 = "jengine/table3"


def spec_for(total_am: int, n_batches: int, levers: tuple[str, ...] = ()) -> Spec:
    """A Spec indexes a table row through its total angular momentum alone
    (spec.py), so any split of L across the four shells names the same cell;
    (L, 0, 0, 0) is the one used here."""
    return Spec(Operator.COULOMB, 0, (total_am, 0, 0, 0), Target.J, levers, n_batches)


def literal_candidates(tables: Mapping[str, Table]) -> dict[str, Candidate]:
    """Every cell of every table, from the literal reconstruction's plan
    (levers=(): the paper's algorithm as written, sieve on)."""
    cells: set[Cell] = {c for t in tables.values() for c in t.cells}
    costs: dict[Cell, PlanCost] = {c: plan(spec_for(*c)).cost for c in sorted(cells)}
    out = {
        tid: Candidate(
            {c: getattr(costs[c], QUANTITY[tid]) for c in t.cells}, board="literal"
        )
        for tid, t in tables.items()
    }
    # #14: Table 3's inputs are recovered (plan/recovery.py) and recorded in
    # literal.json, so its candidate is produced from them and carries them;
    # exact_given_inputs then judges it. Tables 1-2 have no recorded inputs.
    if T3 in tables:
        out[T3] = recovered_candidates({T3: tables[T3]})[T3]
    return out


# -- the searched board's recorded cuts (#13) ----------------------------------
#
# #13 records the chosen cut and seed order per (L, n_batches) next to
# searched.json, in plan/search.py's schema. Without that file the board has
# no candidates. Everything that knows about the file is in this block.

SEARCHED_CUTS = "searched_cuts.json"


class UnreadableCuts(Exception):
    """The cuts file exists, but this reader does not know its schema."""


def read_searched_cuts(path: Path) -> dict[Cell, Choice] | None:
    """#13's (cut, seed order) per (L, n_batches); None while the file is
    absent.
    The schema is plan/search.py's (`load_cuts`); a file it refuses is
    UnreadableCuts."""
    if not path.exists():
        return None
    try:
        return load_cuts(path)
    except (ValueError, KeyError, TypeError) as e:
        raise UnreadableCuts(f"{path}: {e}") from e


def searched_candidates(
    tables: Mapping[str, Table], cuts: Mapping[Cell, Choice]
) -> dict[str, Candidate]:
    """One plan per cell, replayed from its recorded cut and seed order
    (search.searched_plan), scores every table. A cell with no record gets no
    candidate value, which the rule reports rather than this reader guessing
    one."""
    cells = {c for t in tables.values() for c in t.cells} & set(cuts)
    costs = {c: cost(searched_plan(*c, *cuts[c])) for c in sorted(cells)}
    return {
        tid: Candidate(
            {c: getattr(costs[c], QUANTITY[tid]) for c in t.cells if c in costs},
            board="searched",
        )
        for tid, t in tables.items()
    }


# -- replay --------------------------------------------------------------------


def _load_tables(data: Path, ids: Sequence[str]) -> dict[str, Table]:
    return {tid: load_table(data / "literature" / f"{tid}.json") for tid in ids}


def _pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def _print_verdicts(board: str, verdicts: Sequence[Verdict]) -> None:
    """One line per verdict. An at_or_under margin is printed in percent over
    the printed value -- the worst on the verdict's line, then one indented
    line per cell over -- since a list of float pairs is unreadable."""
    for v in verdicts:
        measured = dict(v.measured)
        per_cell = measured.pop("overshoot", ())
        if "worst_overshoot" in measured:
            measured["worst_overshoot"] = _pct(measured["worst_overshoot"])
        shown = json.dumps(measured) if measured else ""
        print(f"{board}  {v.table}  {v.rule}  {v.status.upper()}  {v.reason}"
              + (f"  {shown}" if shown else ""))
        for cell, over in per_cell:
            print(f"    {cell}  {_pct(over)}")


def exit_status(verdicts: Sequence[Verdict]) -> int:
    statuses = {v.status for v in verdicts}
    if Status.FAIL in statuses:
        return EXIT_FAIL
    if not statuses or Status.UNSET in statuses or Status.PASS not in statuses:
        return EXIT_UNSET
    return EXIT_PASS


def replay(board_name: str, data: Path = DATA) -> int:
    boards = data / "scoreboards" / "jengine"
    board = load_scoreboard(boards / f"{board_name}.json", board_name)
    tables = _load_tables(data, sorted(board.rules))
    if board_name == "literal":
        candidates = literal_candidates(tables)
    else:
        try:
            cuts = read_searched_cuts(boards / SEARCHED_CUTS)
        except UnreadableCuts as e:
            print(f"{board_name}: could not run: {e}", file=sys.stderr)
            return EXIT_COULD_NOT_RUN
        if cuts is None:
            print(f"{board_name}: no candidates yet -- {boards / SEARCHED_CUTS} "
                  "does not exist, so nothing was judged (UNSET)")
            return EXIT_UNSET
        candidates = searched_candidates(tables, cuts)
    verdicts = score(board, tables, candidates)
    _print_verdicts(board_name, verdicts)
    return exit_status(verdicts)


# -- the searched board's build step (#172) ------------------------------------

GRID = tuple((L, nb) for L in range(7, 17) for nb in (1, 3, 6, 9))


def build_searched(out: Path, workers: int, budget: int = CLIMB_BUDGET) -> int:
    t0 = time.monotonic()
    results = build(GRID, workers=workers, budget=budget)
    text = dump_cuts(results, budget=budget)
    out.write_text(text)
    wall = time.monotonic() - t0
    print(f"search: {len(results)} cells, "
          f"{sum(r.runs for r in results.values())} climbs "
          f"({sum(r.exhausted for r in results.values())} stopped by the "
          f"budget {budget}), {workers} workers, {wall:.0f} s wall-clock; "
          f"wrote {out}")
    return EXIT_PASS


# -- entry point ---------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m felizjueves.plan",
        description="Replay a [JEngine] scoreboard, or report one plan.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("replay", help="score a board's candidates; one verdict "
                        "per (table, rule)")
    rp.add_argument("board", choices=BOARDS)
    rp.add_argument("--data", type=Path, default=DATA,
                    help=f"the repo's data/ directory (default: {DATA})")
    sp = sub.add_parser("search", help="the searched board's build step: "
                        "climb every cell and write searched_cuts.json")
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--budget", type=int, default=CLIMB_BUDGET,
                    help=f"neighbours examined per run (default {CLIMB_BUDGET}, "
                    "the recorded one)")
    sp.add_argument("--out", type=Path,
                    default=DATA / "scoreboards" / "jengine" / SEARCHED_CUTS)
    pp = sub.add_parser("report", help="one plan's cost and per-batch counts, "
                        "as JSON")
    pp.add_argument("total_am", type=int, metavar="L")
    pp.add_argument("n_batches", type=int)
    pp.add_argument("--no-sieve", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "replay":
        return replay(args.board, args.data)
    if args.command == "search":
        if args.workers < 1 or args.budget < 0:
            print("search: could not run: --workers >= 1, --budget >= 0",
                  file=sys.stderr)
            return EXIT_COULD_NOT_RUN
        return build_searched(args.out, args.workers, args.budget)
    try:
        spec = spec_for(args.total_am, args.n_batches,
                        (NO_SIEVE,) if args.no_sieve else ())
        report = plan(spec).report()
    except ValueError as e:
        print(f"report: could not run: {e}", file=sys.stderr)
        return EXIT_COULD_NOT_RUN
    print(json.dumps(report, indent=2))
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
