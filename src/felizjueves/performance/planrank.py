"""`PlanCost` against `Measured`: does the cheaper plan make a better kernel? (#165)

    python -m felizjueves.performance.planrank              # the report, as JSON
    python -m felizjueves.performance.planrank --write      # rewrite PLANRANK
    python -m felizjueves.performance.planrank --markdown   # the per-card summary table

(with `PYTHONPATH=src uv run` in front, as for every entry point here.)

THE QUESTION neither `[JEngine]` nor `jgen` asks. The plan layer counts a
plan (`plan.cost.PlanCost`: FLOPs summed over batches, intermediates maxed
over batches) and ranks plans by those counts: the sieve exists because it
removes intermediates, the searched board exists because it finds fewer.
Do the counts predict what a device compiler and a card then do with the
kernel? Two claims, as the issue states them:

- SPILL: a plan with fewer `max_intermediates` spills less. Measured side:
  the row's `spill_store + spill_load` (the cell's whole spill, summed over
  its batch kernels -- bytes on sm_86, register counts on gfx1200; only the
  within-card comparison is made).
- CLOCK: a plan with fewer `flops` runs faster, on each of the two timed
  inputs (wide-T, near-T), judged apart.

THE PLAN VARIANTS are the three plans one cell can be built from: LITERAL
(the shipped default, sieved), `no-sieve` (#162's arm) and `tuned` (#163's
arm: the searched board's recorded cut and seed order). The cells are
`measured_store.SUBGRID` plus every arm's added cell (#162's L = 11, one
batch): 21 cells on each card.

THE COMPARISON IS WITHIN A CELL. Across cells the counts and the kernels
both grow with L, so a pooled ranking would agree for a reason that says
nothing about the plan; within one cell `(L, n_batches)` only the plan
moves. So the unit is a PAIR of variants of one cell on one card:

- SPILL pairs: every two variants of the cell that both have a compiled
  row -- (literal, no-sieve), (literal, tuned), (no-sieve, tuned). Compiler
  output, deterministic, so no tolerance: a measured difference of 0 is a
  tie.
- CLOCK pairs: (literal, v) for each arm v, and only from v's row: the arm
  was timed back to back with its same-session literal control (raw
  `wallclock:control:*`), which is the only comparison on this store that
  does not mix the box's drift into the effect. A no-sieve row and a tuned
  row were timed in different sessions, so that pair has NO clean clock
  phase and is never formed. The measured sign is `measured_store.lever_ab`'s
  (#60's symmetric rule: FASTER / SLOWER beyond the larger IQR, else
  NEUTRAL), unchanged.

A CLEAN CLOCK PHASE, per (arm, cell, card, input): the arm row exists, is
timed, carries its same-session control, and all four of its gate readings
on that input (the arm's and the control's, before and after) are known and
at most `gate.GPU_BUSY_MAX`. Anything else is reported as NOT CLEAN with its
reason -- `measured_store.NOT_MEASURABLE`'s for tuned's four 12-batch cells,
"not an A/B cell of the arm" for tuned at L = 11 -- and NEVER IMPUTED: no
literal row, no cross-session row and no neighbouring cell stands in.

EACH PAIR IS CLASSIFIED, `kernelrank.pairs`' reading extended by one class:

- NO_CLAIM: the measurement ties (equal spill; a NEUTRAL clock). Nothing to
  predict.
- NOISE_FLOOR: the two kernels are the SAME TEXT (equal `source_sha256`)
  and the measurement still orders them. Only a clock pair can do this: it
  is the protocol's noise floor, not a plan effect (#55 saw it on identical
  machine code; #163 on gfx1200's L = 7 tuned cells).
- UNRESOLVED: the plan ties (equal count) and the measurement does not.
- CONCORDANT: the plan's cheaper variant is the measured better one.
- DISCORDANT: the plan's cheaper variant is the measured WORSE one -- an
  INVERSION. Every inversion is named in the record.

THE VERDICT, per card and claim (and input, for the clock): PREDICTS iff
some pair is concordant and none discordant; INVERTS iff the reverse; MIXED
iff both; NOT SHOWN iff neither. The counts PREDICT THE KERNEL on a card iff
every claim there PREDICTS. This rule was written after #162's and #163's
rows were read (they are its input), so it decides nothing: it is a
reading, and the default stays #166's hand-recorded decision.

THE RANK STATISTIC is Spearman's rho (`kernelrank.spearman`: average ranks,
None when a side is constant), in kernelrank's shape -- per cell over its
variants (spill; n <= 3), and per card over the pairs' RELATIVE changes
(plan count change / the first variant's count, measured change / the first
variant's measurement): does a bigger plan saving buy a bigger kernel
saving? A spill pair whose first variant spills 0 and whose second does not
has no relative change and is left out of rho, named. AN ORDER, NOT A
PREDICTION: n is a handful of pairs per card, a single pair flips rho, and
nothing here is a prediction interval or a score averaged over cells
(#54's caveat, crossread's rule).

AN UNCHARACTERISED CARD GETS NO VERDICT. `verdict` refuses, through
`device.require_characterised`, a card without both measured ceilings --
the shape `Arch.ridge` refuses with -- rather than read rows nobody can
place.

READS BOTH SIDES, FEEDS NEITHER. This module reads `plan.pipeline` (the
counts) and `measured_store` (the rows). Nothing under `plan/` or
`literature/` may import it (test/test_layering.py's `plan_analysis` rule),
and it imports no literature table, no floor and no driver: a plan chosen by
measurement is #166's hand-recorded decision, never a loop this code closes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from felizjueves.performance import kernelrank
from felizjueves.performance import measured_store as measured
from felizjueves.performance.device import ARCHS, Arch, by_name, require_characterised
from felizjueves.performance.gate import GPU_BUSY_MAX
from felizjueves.plan.cost import PlanCost
from felizjueves.plan.pipeline import plan
from felizjueves.spec import Spec

PLANRANK = measured.REPO / "data" / "measured" / "plan_vs_kernel.json"
SCHEMA = 1

LITERAL = "literal"
VARIANTS = (LITERAL, measured.NO_SIEVE, measured.TUNED)
ARMS = VARIANTS[1:]
INPUTS = ("wide", "near")

# input -> (the arm's raw prefix, its control's raw prefix)
_PREFIX = {"wide": ("wallclock:", "wallclock:control:"),
           "near": ("wallclock:near:", "wallclock:control:near:")}

CONCORDANT, DISCORDANT, UNRESOLVED = "concordant", "discordant", "unresolved"
NO_CLAIM, NOISE_FLOOR = "no_claim", "noise_floor"
CLASSES = (CONCORDANT, DISCORDANT, UNRESOLVED, NO_CLAIM, NOISE_FLOOR)
PREDICTS, INVERTS, MIXED, NOT_SHOWN = "predicts", "inverts", "mixed", "not shown"

SPILL_METRIC, CLOCK_METRIC = "max_intermediates", "flops"

# spec -> its PlanCost, or ValueError when the plan layer refuses the cell.
# Injectable so the tests can join synthetic counts to synthetic rows.
Costs = Callable[[Spec], PlanCost]


def plan_cost(spec: Spec) -> PlanCost:
    return plan(spec).cost


def _cells() -> tuple[Spec, ...]:
    added = {replace(s, levers=()) for a in ARMS
             for s in measured.ADDED_CELLS.get(a, ())}
    return tuple(sorted({*measured.SUBGRID, *added},
                        key=lambda s: (s.total_am, s.n_batches)))


CELLS: tuple[Spec, ...] = _cells()


def variant(cell: Spec, v: str) -> Spec:
    """`cell` built from plan variant `v`."""
    if v not in VARIANTS:
        raise ValueError(f"variant {v!r} is not one of {VARIANTS}")
    return cell if v == LITERAL else replace(cell, levers=(v,))


# ---------------------------------------------------------------------------
# One pair
# ---------------------------------------------------------------------------


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def classify(plan_delta: float, measured_delta: float, same_text: bool) -> str:
    """One pair (a, b): `plan_delta` = count(b) - count(a), `measured_delta`
    = cost(b) - cost(a), lower being better on both sides (module
    docstring, EACH PAIR IS CLASSIFIED)."""
    m = _sign(measured_delta)
    if m == 0:
        return NO_CLAIM
    if same_text:
        return NOISE_FLOOR
    p = _sign(plan_delta)
    if p == 0:
        return UNRESOLVED
    return CONCORDANT if p == m else DISCORDANT


def _rel(delta: float, base: float) -> float | None:
    if base == 0:
        return 0.0 if delta == 0 else None
    return round(delta / base, 6)


# ---------------------------------------------------------------------------
# One cell on one card
# ---------------------------------------------------------------------------


def _row(store: Mapping[str, measured.Measured], spec: Spec,
         arch: str) -> measured.Measured | None:
    got = measured.rows(spec=spec, arch=arch, store=store)
    if len(got) > 1:
        raise ValueError(f"{spec.key()} @ {arch}: {len(got)} rows")
    return next(iter(got.values()), None)


def _absent(cell: Spec, v: str) -> str:
    """Why variant `v` of `cell` has no row: the arm's recorded reason."""
    spec = variant(cell, v)
    gone = measured.NOT_MEASURABLE.get(v, {})
    if spec in gone:
        return f"not measurable: {gone[spec]}"
    if v != LITERAL and spec not in measured.ab_cells(v):
        return (f"not recorded: L = {cell.total_am}, n_batches = "
                f"{cell.n_batches} is not an A/B cell of the {v!r} arm "
                f"(measured_store.ab_cells)")
    return "not recorded: no row in the store"


def _variant_entry(store, cell: Spec, v: str, arch: str,
                   costs: Costs) -> dict:
    spec = variant(cell, v)
    try:
        c = costs(spec)
        cost = {"flops": c.flops, "max_intermediates": c.max_intermediates,
                "max_targets": c.max_targets}
        refused = None
    except ValueError as e:
        cost, refused = None, str(e)
    r = _row(store, spec, arch)
    compiled = (r is not None and r.spill_store is not None
                and r.spill_load is not None)
    return {
        "key": spec.key(),
        "plan": cost,
        **({"plan_refused": refused} if refused else {}),
        "measured": ({"registers": r.registers,
                      "spill": r.spill_store + r.spill_load,
                      "source_sha256": r.source_sha256} if compiled else None),
        **({} if compiled else {"absent": _absent(cell, v)}),
    }


def _clean(r: measured.Measured | None, which: str) -> str | None:
    """None when the arm row has a clean clock phase on `which`, else why
    not (module docstring, A CLEAN CLOCK PHASE)."""
    if r is None or r.runtime_ns is None:
        return "no timed row"
    arm, ctl = _PREFIX[which]
    if f"{ctl}median_ns" not in r.raw:
        return "no same-session control"
    for p in (arm, ctl):
        for when in ("before", "after"):
            got = r.raw.get(f"{p}gpu_util_{when}")
            if got is None:
                return f"gate reading {p}gpu_util_{when} unknown"
            if got > GPU_BUSY_MAX:
                return (f"gate reading {p}gpu_util_{when} = {got}% > "
                        f"{GPU_BUSY_MAX}%")
    return None


def _spill_pairs(vs: Mapping[str, dict]) -> list[dict]:
    out = []
    for i, a in enumerate(VARIANTS):
        for b in VARIANTS[i + 1:]:
            ea, eb = vs[a], vs[b]
            if not (ea["plan"] and eb["plan"] and ea["measured"]
                    and eb["measured"]):
                continue
            pa, pb = ea["plan"][SPILL_METRIC], eb["plan"][SPILL_METRIC]
            ma, mb = ea["measured"]["spill"], eb["measured"]["spill"]
            same = ea["measured"]["source_sha256"] == eb["measured"][
                "source_sha256"] != ""
            out.append({
                "a": a, "b": b, "plan": [pa, pb], "measured": [ma, mb],
                "same_text": same,
                "plan_rel": _rel(pb - pa, pa), "measured_rel": _rel(mb - ma, ma),
                "class": classify(pb - pa, mb - ma, same),
            })
    return out


def _clock(store, cell: Spec, vs: Mapping[str, dict], arch: str,
           which: str) -> list[dict]:
    out = []
    for v in ARMS:
        r = _row(store, variant(cell, v), arch)
        why = vs[v].get("absent") if r is None else _clean(r, which)
        if why is None and not (vs[v]["measured"] and vs[LITERAL]["measured"]):
            why = "an arm or its literal control is not compiled"
        if why is None and not (vs[v]["plan"] and vs[LITERAL]["plan"]):
            why = "the plan layer refuses the arm or its control"
        if why is not None:
            out.append({"variant": v, "clean": False, "reason": why})
            continue
        ab = measured.lever_ab(store, arch, variant(cell, v))
        cm, ci, lm, li = ab.wide if which == "wide" else ab.near
        sign = ab.wide_sign if which == "wide" else ab.near_sign
        delta = {measured.FASTER: -1, measured.SLOWER: 1,
                 measured.NEUTRAL: 0}[sign]
        pl, pv = vs[LITERAL]["plan"][CLOCK_METRIC], vs[v]["plan"][CLOCK_METRIC]
        same = (vs[LITERAL]["measured"]["source_sha256"]
                == vs[v]["measured"]["source_sha256"] != "")
        out.append({
            "variant": v, "clean": True, "plan": [pl, pv],
            "control_ns": [cm, ci], "arm_ns": [lm, li], "sign": sign,
            "same_text": same,
            "plan_rel": _rel(pv - pl, pl), "measured_rel": _rel(lm - cm, cm),
            "class": classify(pv - pl, delta, same),
        })
    return out


def cell_row(store: Mapping[str, measured.Measured], cell: Spec, arch: str,
             costs: Costs = plan_cost) -> dict:
    """One (cell, card) of the table: every variant's counts and row, the
    spill pairs and the clock pairs per input."""
    vs = {v: _variant_entry(store, cell, v, arch, costs) for v in VARIANTS}
    spill = _spill_pairs(vs)
    have = [v for v in VARIANTS if vs[v]["plan"] and vs[v]["measured"]]
    rho = (kernelrank.spearman(
        [vs[v]["plan"][SPILL_METRIC] for v in have],
        [vs[v]["measured"]["spill"] for v in have]) if len(have) >= 2 else None)
    return {
        "arch": arch, "L": cell.total_am, "n_batches": cell.n_batches,
        "variants": vs,
        "spill": {"variants": have, "spearman": rho, "pairs": spill},
        "clock": {w: _clock(store, cell, vs, arch, w) for w in INPUTS},
    }


# ---------------------------------------------------------------------------
# Per card
# ---------------------------------------------------------------------------


def _verdict(counts: Mapping[str, int]) -> str:
    c, d = counts[CONCORDANT], counts[DISCORDANT]
    if c and not d:
        return PREDICTS
    if d and not c:
        return INVERTS
    return MIXED if c and d else NOT_SHOWN


def _named(r: dict, p: dict, *, clock: bool) -> dict:
    a, b = (LITERAL, p["variant"]) if clock else (p["a"], p["b"])
    out = {"L": r["L"], "n_batches": r["n_batches"], "a": a, "b": b,
           "plan": p["plan"]}
    out.update({"sign": p["sign"], "arm_ns": p["arm_ns"],
                "control_ns": p["control_ns"]} if clock
               else {"measured": p["measured"]})
    return out


def _claim(rows: Sequence[dict], pick: Callable[[dict], list[dict]], *,
           clock: bool) -> dict:
    counts = dict.fromkeys(CLASSES, 0)
    inv, noise, xs, ys, left_out = [], [], [], [], []
    for r in rows:
        for p in pick(r):
            if not p.get("clean", True):
                continue
            counts[p["class"]] += 1
            if p["class"] == DISCORDANT:
                inv.append(_named(r, p, clock=clock))
            if p["class"] == NOISE_FLOOR:
                noise.append(_named(r, p, clock=clock))
            if p["measured_rel"] is None:
                left_out.append(_named(r, p, clock=clock))
            else:
                xs.append(p["plan_rel"])
                ys.append(p["measured_rel"])
    return {"n_pairs": sum(counts.values()), **counts,
            "verdict": _verdict(counts),
            "spearman": kernelrank.spearman(xs, ys) if len(xs) >= 2 else None,
            "spearman_n": len(xs), "spearman_left_out": left_out,
            "inversions": inv, "noise_floor_pairs": noise}


def _not_clean(rows: Sequence[dict], which: str) -> list[dict]:
    return [{"L": r["L"], "n_batches": r["n_batches"], "variant": p["variant"],
             "reason": p["reason"]}
            for r in rows for p in r["clock"][which] if not p["clean"]]


def _sentence(arch: str, s: dict) -> str:
    def one(c: dict) -> str:
        inv = ", ".join(f"L{e['L']} b{e['n_batches']} {e['a']}/{e['b']}"
                        for e in c["inversions"])
        return c["verdict"].upper() + (f" (inversions: {inv})" if inv else "")
    clock = "; ".join(f"{w}-T {one(s['clock'][w])}" for w in INPUTS)
    yes = "predict" if s["predicts"] else "do not predict"
    return (f"On {arch} the plan-layer counts {yes} the kernel. Fewer "
            f"intermediates -> less spill: {one(s['spill'])}. Fewer FLOPs -> "
            f"faster: {clock}.")


def summary(rows: Sequence[dict], arch: str) -> dict:
    """The per-card verdict over `rows` (that card's cell rows)."""
    mine = [r for r in rows if r["arch"] == arch]
    spill = _claim(mine, lambda r: r["spill"]["pairs"], clock=False)
    clock = {w: {**_claim(mine, lambda r, w=w: r["clock"][w], clock=True),
                 "clean": sum(p["clean"] for r in mine for p in r["clock"][w]),
                 "not_clean": _not_clean(mine, w)}
             for w in INPUTS}
    s = {"spill": spill, "clock": clock,
         "predicts": all(c["verdict"] == PREDICTS
                         for c in (spill, *clock.values()))}
    s["answer"] = _sentence(arch, s)
    return s


def table(store: Mapping[str, measured.Measured],
          costs: Costs = plan_cost) -> list[dict]:
    """Every (cell, card): arches in ARCHS order, cells by (L, n_batches)."""
    return [cell_row(store, c, a.name, costs) for a in ARCHS for c in CELLS]


def verdict(store: Mapping[str, measured.Measured], arch: Arch | str,
            costs: Costs = plan_cost) -> dict:
    """The recorded answer for one card (module docstring, THE VERDICT).
    Raises `device.Uncharacterised` for a card without both measured
    ceilings: it gets no verdict."""
    a = require_characterised(by_name(arch) if isinstance(arch, str) else arch)
    return summary([cell_row(store, c, a.name, costs) for c in CELLS], a.name)


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

ABOUT = ("PlanCost across plan variants (literal, no-sieve, tuned) against "
         "the measured spills and, where a clean same-session clock phase "
         "exists, wall-clock, per (cell, card) "
         "(felizjueves.performance.planrank, #165). Does a plan with fewer "
         "counted intermediates spill less; does one "
         "with fewer FLOPs run faster? Reads both sides, feeds neither: "
         "nothing here reaches PlanCost or a literature floor. An order, "
         "never a prediction. Recomputed by test/test_planrank.py.")


def report(store: Mapping[str, measured.Measured],
           costs: Costs = plan_cost) -> dict:
    rows = table(store, costs)
    for a in ARCHS:  # an uncharacterised card gets no verdict
        require_characterised(a)
    return {
        "schema": SCHEMA,
        "about": ABOUT,
        "header": {
            "cells": [[c.total_am, c.n_batches] for c in CELLS],
            "variants": list(VARIANTS),
            "plan": "plan.pipeline.plan(spec).cost: flops summed over "
                    "batches, max_intermediates maxed over batches, of the "
                    "plan the kernel is built from (sieved unless no-sieve; "
                    "the recorded search if tuned).",
            "spill": "max_intermediates against the row's spill_store + "
                     "spill_load, over every two variants of a cell with a "
                     "compiled row. sm_86: bytes; gfx1200: spilled register "
                     "counts. Within a card only.",
            "clock": "flops against lever_ab's same-session sign (#60's "
                     "rule), for (literal, arm) only: a no-sieve row and a "
                     "tuned row were timed in different sessions, so that "
                     "pair is never formed. Per input, judged apart.",
            "clean": "an arm row timed against its same-session control, "
                     "its four gate readings on that input known and <= "
                     f"{GPU_BUSY_MAX}% (gate.GPU_BUSY_MAX). Anything else "
                     "is listed as not clean with its reason, never imputed.",
            "classes": "no_claim: the measurement ties. noise_floor: same "
                       "text (source_sha256), measured apart. unresolved: "
                       "the plan ties. concordant: the plan's cheaper "
                       "variant measures better. discordant: it measures "
                       "worse -- an inversion, named in summary.",
            "verdict": "per card and claim: predicts iff concordant > 0 and "
                       "discordant = 0; inverts iff the reverse; mixed iff "
                       "both; not shown iff neither. The counts predict the "
                       "kernel on a card iff every claim there predicts. "
                       "Written after #162's and #163's rows were read: a "
                       "reading, never a choice (#166 decides the default).",
            "spearman": "kernelrank.spearman: per cell over its variants "
                        "(spill), and per card over the pairs' relative "
                        "changes. A handful of points: an order, never a "
                        "prediction interval or a pooled score.",
            "regenerate": "PYTHONPATH=src uv run python -m "
                          "felizjueves.performance.planrank --write",
        },
        "summary": {a.name: summary(rows, a.name) for a in ARCHS},
        "rows": rows,
    }


def markdown(doc: Mapping) -> str:
    """One line per (card, claim): the verdict, the pair counts, rho and the
    inversions."""
    out = ["| card | claim | verdict | concordant | discordant | unresolved "
           "| no claim | noise floor | rho (n) | inversions |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for arch, s in doc["summary"].items():
        for name, c in (("intermediates -> spill", s["spill"]),
                        *((f"FLOPs -> clock, {w}-T", s["clock"][w])
                          for w in INPUTS)):
            inv = ", ".join(f"L{e['L']} b{e['n_batches']} {e['a']}/{e['b']}"
                            for e in c["inversions"]) or "none"
            rho = "n/a" if c["spearman"] is None else f"{c['spearman']:+.2f}"
            out.append(
                f"| {arch} | {name} | **{c['verdict']}** | {c[CONCORDANT]} | "
                f"{c[DISCORDANT]} | {c[UNRESOLVED]} | {c[NO_CLAIM]} | "
                f"{c[NOISE_FLOOR]} | {rho} ({c['spearman_n']}) | {inv} |")
    return "\n".join(out)


def write(store: Mapping[str, measured.Measured],
          path: Path = PLANRANK) -> dict:
    doc = report(store)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m felizjueves.performance.planrank")
    ap.add_argument("--write", action="store_true",
                    help=f"rewrite {PLANRANK.relative_to(measured.REPO)}")
    ap.add_argument("--markdown", action="store_true",
                    help="print the per-card summary table")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        store = measured.load()
        doc = write(store) if args.write else report(store)
    except (OSError, ValueError, KeyError) as e:
        print(f"planrank: {e}", file=sys.stderr)
        return 2
    print(markdown(doc) if args.markdown
          else json.dumps(doc, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
