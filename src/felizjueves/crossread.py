"""Three instruments, one subgrid: the cross-read (#115).

    python -m felizjueves.crossread             # the table, as JSON
    python -m felizjueves.crossread --write     # rewrite CROSSREAD
    python -m felizjueves.crossread --markdown  # the j_engine.md table

(with `PYTHONPATH=src uv run` in front, as for every entry point here.)

THE THREE INSTRUMENTS. The repo ranks a batch count three ways, each already
built and each recomputed by its own test. This module reads them against
each other on `measured_store.SUBGRID` and decides nothing new:

- THE MODEL (`kernelrank`, #54): `lower.KernelCost.traffic` summed over a
  cell's batch kernels, under the arch's declared register budget. That is
  kernelrank's headline aggregation. Its PICK is the set of batch counts at
  its minimum, ties kept. The model reads the IR and no row.
- THE CLOCK (`batchreport`, #56): the best-median `n_batches` on each of the
  two timed inputs (wide-T, near-T), with `resolved` and `within_spread` as
  batchreport defines them.
- THE COUNTERS (`ridge`, #112 / #114): the declared crossing rule's answer,
  both variants, and the FIRST COMPUTE-BOUND COUNT: 1 when the answer is
  ALREADY, the crossing count when CROSSED, none when NEVER. A card in
  `measured_store.UNMEASURED` (gfx1200 today) has no answer. Its counter
  column says *could not measure* and every flag that needs it is `null`.
  An absence is never a disagreement and never a zero.

THE FLAGS, per (arch, L):

- `model_clock[input]`: the clock's best is in the model's pick.
- `clock_crossing[variant][input]`: the clock's best IS the first
  compute-bound count. This is the paper's implicit mechanism (crossing the
  ridge is WHY the clock improves) made testable. `false` under NEVER, `null`
  when the card could not be measured. The counters were taken on the WIDE-T
  input only (roofline.json's header), so `same_input` marks the wide flag as
  the like-for-like one. The near flag compares a near-T clock against a
  wide-T verdict, and says so.
- `model_crossing[variant]`: the first compute-bound count is in the model's
  pick (`null` when the card could not be measured).

MEASURED OUTRANKS COST. `choice[input]` is the clock's best and nothing else.
It is computed without reading the model, and the test proves it by swapping
the model for its reverse and seeing no choice move. Where the model
disagrees, the model loses. The counters do not choose either. They explain
a choice or fail to, and a failure is a record.

AN ORDER, NOT A PREDICTION. Four L on two cards is a small sample (#54: over
n batch counts, Spearman "takes few values and a single pair flips it"). The
table reports which count each instrument names, and nothing here is a
prediction interval or a score averaged over cells.

A DISAGREEMENT IS A RECORD. data/measured/limits.json holds one
`StructuralLimit` per kind of disagreement found here (its asserted_value
carries `"cross_read": true` and the table rows it covers), each with a
mechanism. test/test_kernelrank.py recomputes every recorded row and fails
on a disagreement no record covers.

NOTHING HERE FEEDS A LITERATURE FLOOR. This module reads the three analysis
modules and the store. Nothing under `plan/` or `literature/` may import it
(test/test_layering.py), and it imports neither.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from felizjueves import batchreport, kernelrank, ridge
from felizjueves import measured_store as measured
from felizjueves.device import ARCHS
from felizjueves.spec import Spec

CROSSREAD = measured.REPO / "data" / "measured" / "crossread.json"
SCHEMA = 1

INPUTS = batchreport.INPUTS  # ("wide", "near")
COUNTER_INPUT = "wide"  # roofline.json's counter run: the wide-T input
MODEL_METRIC, MODEL_AGGREGATION = "traffic", "sum"

# (spec, register budget) -> the model's number for that cell. Injectable so
# the test can show that no `choice` reads it.
Model = Callable[[Spec, int], int]


def kernelcost(spec: Spec, registers: int) -> int:
    """kernelrank's headline: KernelCost.traffic summed over the batch kernels."""
    return kernelrank.model_cell(spec, registers).traffic_sum


ABOUT = ("The three ways this repo ranks a batch count (KernelCost's model, "
         "#54; the wall-clock recommendation, #56; the counters' ridge "
         "crossing, #112/#114) read against each other per (L, arch) of "
         "measured.SUBGRID (felizjueves.crossread, #115). The choice is the "
         "clock's, everywhere: Measured outranks Cost. Every disagreement is "
         "a StructuralLimit in data/measured/limits.json. An order, never a "
         "prediction. Recomputed by test/test_kernelrank.py.")


def _specs(L: int) -> list[Spec]:
    got = sorted((s for s in measured.SUBGRID if s.total_am == L),
                 key=lambda s: s.n_batches)
    if not got:
        raise ValueError(f"L={L} is not in SUBGRID")
    return got


def _order(bs: Sequence[int], xs: Sequence[float]) -> list[list[int]]:
    """Batch counts grouped by value, ascending: ties share a group."""
    out: list[list[int]] = []
    last = None
    for x, b in sorted(zip(xs, bs, strict=True)):
        if out and x == last:
            out[-1].append(b)
        else:
            out.append([b])
        last = x
    return out


def first_compute_bound(c: ridge.Crossing) -> int | None:
    """The smallest SUBGRID count at which the crossing variant holds: 1 when
    ALREADY, the crossing count when CROSSED, None when NEVER or unmeasured."""
    if c.answer == ridge.ALREADY:
        return 1
    if c.answer == ridge.CROSSED:
        return c.count
    return None


def row(store: Mapping[str, measured.Measured],
        counters: Mapping[str, measured.Measured], arch: str, L: int,
        model: Model = kernelcost) -> dict:
    """One (arch, L) of the table (module docstring)."""
    specs = _specs(L)
    bs = [s.n_batches for s in specs]
    budget = kernelrank.BUDGET[arch]
    values = [model(s, budget) for s in specs]
    order = _order(bs, values)
    pick = order[0]

    rec = {w: batchreport.recommend(store, arch, L, w) for w in INPUTS}
    clock = {w: {"n_batches": r.best.n_batches, "resolved": r.resolved,
                 "within_spread": list(r.within_spread),
                 "same_input": w == COUNTER_INPUT}
             for w, r in rec.items()}
    # Measured outranks Cost: the choice reads the clock and nothing else.
    choice = {w: r.best.n_batches for w, r in rec.items()}

    unmeasured = arch in measured.UNMEASURED
    cross = {v: ridge.crossing(counters, arch, L, v) for v in ridge.VARIANTS}
    first = {v: first_compute_bound(c) for v, c in cross.items()}
    counter_col = {v: {"answer": c.answer, "count": c.count,
                       "bound": [list(p) for p in c.bound],
                       "first_compute_bound": first[v]}
                   for v, c in cross.items()}

    def vs_first(v: str, b: int) -> bool | None:
        return None if unmeasured else first[v] == b

    return {
        "arch": arch, "L": L, "n_batches": bs,
        "model": {"metric": MODEL_METRIC, "aggregation": MODEL_AGGREGATION,
                  "registers": budget, "values": values, "order": order,
                  "pick": pick},
        "clock": clock,
        "counters": ({"could_not_measure": measured.UNMEASURED[arch]}
                     if unmeasured else counter_col),
        "choice": choice,
        "agree": {
            "model_clock": {w: choice[w] in pick for w in INPUTS},
            "clock_crossing": {v: {w: vs_first(v, choice[w]) for w in INPUTS}
                               for v in ridge.VARIANTS},
            "model_crossing": {v: (None if unmeasured else first[v] in pick)
                               for v in ridge.VARIANTS},
        },
    }


def table(store: Mapping[str, measured.Measured],
          counters: Mapping[str, measured.Measured],
          model: Model = kernelcost) -> list[dict]:
    """Every (arch, L) of SUBGRID, arches in ARCHS order, L ascending."""
    return [row(store, counters, a.name, L, model)
            for a in ARCHS for L in measured.SUBGRID_LS]


PAIRS = ("model_clock", "clock_crossing", "model_crossing")


def _disagreement(r: dict, pair: str, variant: str | None,
                  which: str | None) -> dict:
    d = {"arch": r["arch"], "L": r["L"], "pair": pair, "variant": variant,
         "input": which}
    if pair != "clock_crossing":
        d["model_pick"] = r["model"]["pick"]
    if which is not None:
        d["choice"] = r["choice"][which]
    if variant is not None:
        d["first_compute_bound"] = r["counters"][variant]["first_compute_bound"]
    return d


def disagreements(rows: Sequence[dict]) -> list[dict]:
    """Every flag that is False, with what each side named: (arch, L, pair,
    variant, input) -- variant or input None where the pair has none -- and
    the model's pick, the clock's choice and the first compute-bound count
    the pair compared. A null flag (could not measure) is not a
    disagreement."""
    out = []
    for r in rows:
        ag = r["agree"]
        out += [_disagreement(r, "model_clock", None, w)
                for w, ok in ag["model_clock"].items() if ok is False]
        out += [_disagreement(r, "clock_crossing", v, w)
                for v, by in ag["clock_crossing"].items()
                for w, ok in by.items() if ok is False]
        out += [_disagreement(r, "model_crossing", v, None)
                for v, ok in ag["model_crossing"].items() if ok is False]
    return out


def report(store: Mapping[str, measured.Measured],
           counters: Mapping[str, measured.Measured]) -> dict:
    return {
        "schema": SCHEMA,
        "about": ABOUT,
        "header": {
            "model": "kernelrank.model_cell(spec, kernelrank.BUDGET[arch])"
                     ".traffic_sum: KernelCost.traffic summed over the batch "
                     "kernels (#54's headline). order = batch counts grouped "
                     "by value, ascending; pick = the first group (ties kept).",
            "clock": "batchreport.recommend: best median per input, with "
                     "resolved and within_spread as #56 defines them.",
            "counters": "ridge.crossing, both variants (#114's declared "
                        "rule); first_compute_bound = 1 if compute-bound "
                        "already at 1 batch, the crossing count if crossed, "
                        "null if never. The counter run is the WIDE-T input "
                        "only, so clock.wide is the same-input comparison "
                        "and clock.near is not.",
            "choice": "The clock's best per input, and only that: Measured "
                      "outranks Cost. The model and the counters choose "
                      "nothing.",
            "agree": "model_clock: clock best in model pick. clock_crossing: "
                     "clock best == first compute-bound count. "
                     "model_crossing: first compute-bound count in model "
                     "pick. null = the card could not be measured "
                     "(measured_store.UNMEASURED): an absence, not a "
                     "disagreement.",
            "small_sample": "4 L x 2 cards x 5 batch counts. Each cell reports "
                            "which count each instrument names, never a "
                            "prediction interval or a pooled score.",
            "disagreements": "Each is covered by a StructuralLimit in "
                             "data/measured/limits.json whose asserted_value "
                             "has cross_read = true.",
            "regenerate": "PYTHONPATH=src uv run python -m "
                          "felizjueves.crossread --write",
        },
        "rows": table(store, counters),
    }


def _flag(x: bool | None) -> str:
    return "n/a" if x is None else ("yes" if x else "**no**")


def _crossing(r: dict) -> str:
    c = r["counters"]
    if "could_not_measure" in c:
        return "could not measure"
    cell, kern = c[ridge.CELL], c[ridge.PER_KERNEL]
    one = ("already at 1" if cell["answer"] == ridge.ALREADY
           else f"crossed at {cell['count']}" if cell["answer"] == ridge.CROSSED
           else cell["answer"])
    if kern["first_compute_bound"] != cell["first_compute_bound"]:
        one += f" (per kernel: {kern['first_compute_bound']})"
    return one


def markdown(doc: Mapping) -> str:
    """The j_engine.md table, from a report document (cell variant's flags;
    the per-kernel variant's are in the data file)."""
    out = ["| arch | L | model pick (traffic, summed) | clock wide | clock near "
           "| first compute-bound (counters, wide-T) | model = clock (wide, near) "
           "| clock = crossing (wide, near) | model ∋ crossing |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in doc["rows"]:
        ag = r["agree"]
        cc = ag["clock_crossing"][ridge.CELL]
        out.append(
            f"| {r['arch']} | {r['L']} | "
            f"{', '.join(str(b) for b in r['model']['pick'])} | "
            f"**{r['choice']['wide']}** | **{r['choice']['near']}** | "
            f"{_crossing(r)} | "
            f"{_flag(ag['model_clock']['wide'])}, "
            f"{_flag(ag['model_clock']['near'])} | "
            f"{_flag(cc['wide'])}, {_flag(cc['near'])} | "
            f"{_flag(ag['model_crossing'][ridge.CELL])} |")
    return "\n".join(out)


def write(store: Mapping[str, measured.Measured],
          counters: Mapping[str, measured.Measured],
          path: Path = CROSSREAD) -> dict:
    doc = report(store, counters)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m felizjueves.crossread")
    ap.add_argument("--write", action="store_true",
                    help=f"rewrite {CROSSREAD.relative_to(measured.REPO)}")
    ap.add_argument("--markdown", action="store_true",
                    help="print the j_engine.md table")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        store, counters = measured.load(), measured.load_counters()
        doc = write(store, counters) if args.write else report(store, counters)
    except (OSError, ValueError, KeyError) as e:
        print(f"crossread: {e}", file=sys.stderr)
        return 2
    print(markdown(doc) if args.markdown
          else json.dumps(doc, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
