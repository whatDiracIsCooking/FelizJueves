"""A recommended batch count per `(L, arch)`, from wall-clock only (#56).

    python -m felizjueves.performance.batchreport            # the report, as JSON
    python -m felizjueves.performance.batchreport --write    # rewrite RECOMMENDATION
    python -m felizjueves.performance.batchreport --markdown # the j_engine.md tables

(with `PYTHONPATH=src uv run` in front, as for every entry point here.)

WHAT IT IS. For every `(arch, L)` of `measured.SUBGRID`, and for each of the
two inputs the wall-clock sweep timed, the `n_batches` whose cell has the
lowest MEDIAN wall-clock, carried with its interquartile range and the number
of timed repeats behind it -- the same kind of statement `[JEngine, v2, §7]`
makes by bolding the runtime-minimising count in Tables 1-3, made about OUR
cards. It is not a reproduction of their bolded choice: their kernels, their
compiler, their A100 (data/literature/jengine/README.md).

WALL-CLOCK ONLY. The inputs are `runtime_ns` / `spread_ns` / `repeats` and
the raw `wallclock:near:*` keys of the `Measured` rows -- never spills,
registers or `lower.KernelCost` -- of the LITERAL rows, `measured.SUBGRID`'s
own specs (`levers = ()`); a lever arm's rows (#55's `.../recycle@...`) are
keyed on another Spec and never read. test/test_layering.py holds this module's
imports to `measured_store` (the store, not its driver: #94) and `device`,
so it cannot reach the cost model.

TWO INPUTS, TWO ANSWERS. The sweep (measure/measure_kernel_device.py) times every
cell on a WIDE-T input (`runtime_ns`, the row's number; T up to ~430, a long
Boys series) and a NEAR-T one (`wallclock:near:*`, T < ~2). Each batch
recomputes its fundamentals, so the Boys cost weighs on the batch-count
ranking and the two inputs need not agree -- on the tracked rows they do not.
"The batch count with the best median" is therefore input-dependent, and the
report gives both rather than choosing one: that dependence IS the result.
Both inputs were timed with the same protocol, so the row's `repeats` is the
repeat count of either.

THE RULE, per (arch, L, input), over the n_batches the subgrid declares:

- `best`: the lowest median. Its median, IQR and repeats are carried.
- `runner_up`: the second-lowest median; `margin_ns` = its median minus
  best's. `resolved` iff margin_ns > max(IQR(best), IQR(runner_up)) -- the
  test #60's NINE AGAINST TWELVE uses (measured_store.py), so "faster" means one
  thing in this repo.
- `within_spread`: every n_batches (best included) whose median exceeds
  best's by no more than max(IQR(best), IQR(that cell)). One entry means the
  best is distinguishable from every other count; more means it is not.
- `same_session`: whether best and runner_up were timed in one sweep
  (`SESSION`). #60 measured cross-session drift of up to 1.7% (10% on one
  noisy cell), which is IQR-scale, so a cross-session margin is weaker
  evidence. The tracked rows are all one session per arch (#93's re-timing,
  which replaced #53's b 1-9 and #60's b 12), so on them it is always true;
  the field stays for a store that mixes sweeps again.

A missing cell is not skipped: `recommend` raises, and the report cannot be
written over a subgrid with a gap. Every SUBGRID cell is timed on both arches
in the tracked store (no cell was dropped for compile cost, #52/#60).

NOT A POLICY. `Spec.n_batches` defaults to 1, and this does not change it.
Choosing a count per class is DISPATCH, and it lives in `dispatch.py` (#145):
it reads these rows (`recommend`'s best median, recomputed from the store)
per `(arch, class, input)` and is injected into `launch.launches(resolve=)`,
so it reaches only a launch plan -- never the generator, which ships one
batch by decision (#167). The table is kept per arch because the arches may
disagree. Nothing under `plan/` or `literature/` imports this module, and
nothing here feeds a literature floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from felizjueves.performance import measured_store as measured
from felizjueves.performance.device import ARCHS

RECOMMENDATION = measured.REPO / "data" / "measured" / "batch_recommendation.json"
SCHEMA = 1

INPUTS = ("wide", "near")
BATCHES = tuple(sorted({s.n_batches for s in measured.SUBGRID}))

# Which wall-clock sweep timed each n_batches of the tracked rows. #53 timed
# b 1, 3, 6, 9 (PR #75) and #60 b 12 (PR #77); #93 re-timed all five behind
# the card gate, ONE session per arch, replacing both. Declared, because the
# rows do not say.
SESSION: Mapping[int, str] = dict.fromkeys(BATCHES, "#93")

ABOUT = ("A recommended n_batches per (L, arch) of measured.SUBGRID, derived "
         "from wall-clock ONLY (felizjueves.performance.batchreport, #56), "
         "on each of the sweep's two inputs. The generator ships "
         "n_batches = 1 by decision "
         "(#167), and these rows are its grounds: wide-T picks 1 everywhere, "
         "near-T splits from L = 13 and the two cards disagree at L = 16, so "
         "following them (felizjueves.performance.dispatch, #145, which resolves per "
         "(arch, input)) would make the shipped unit set arch- and "
         "input-dependent, which it is not. The rows stay evidence: dispatch "
         "reads them for a launch plan. Not a reproduction of "
         "[JEngine, v2, §7]'s bolded counts (their A100). Recomputed from "
         "data/measured/kernels.json by test/test_batchreport.py.")


@dataclass(frozen=True)
class Timing:
    """One cell's wall-clock on one input."""

    n_batches: int
    median_ns: int
    iqr_ns: int
    repeats: int
    session: str


@dataclass(frozen=True)
class Recommendation:
    """The best-median n_batches at one (arch, L, input), with its evidence."""

    arch: str
    L: int
    input: str
    timings: tuple[Timing, ...]  # in n_batches order

    def _by_median(self) -> list[Timing]:
        return sorted(self.timings, key=lambda t: (t.median_ns, t.n_batches))

    @property
    def best(self) -> Timing:
        return self._by_median()[0]

    @property
    def runner_up(self) -> Timing:
        return self._by_median()[1]

    @property
    def margin_ns(self) -> int:
        return self.runner_up.median_ns - self.best.median_ns

    @property
    def resolved(self) -> bool:
        return self.margin_ns > max(self.best.iqr_ns, self.runner_up.iqr_ns)

    @property
    def within_spread(self) -> tuple[int, ...]:
        b = self.best
        return tuple(t.n_batches for t in self.timings
                     if t.median_ns - b.median_ns <= max(b.iqr_ns, t.iqr_ns))

    @property
    def same_session(self) -> bool:
        return self.best.session == self.runner_up.session


def _row(store: Mapping[str, measured.Measured], arch: str, L: int,
         b: int) -> measured.Measured:
    spec = next((s for s in measured.SUBGRID
                 if s.total_am == L and s.n_batches == b), None)
    if spec is None:
        raise KeyError(f"L={L}, n_batches={b} is not a SUBGRID cell")
    got = measured.rows(spec=spec, arch=arch, store=store)
    if len(got) != 1:
        raise KeyError(f"L={L}, n_batches={b} @ {arch}: {len(got)} rows")
    (r,) = got.values()
    return r


def timing(row: measured.Measured, which: str, n_batches: int) -> Timing:
    """`row`'s wall-clock on input `which`; KeyError when it was not timed."""
    if row.repeats is None:
        raise KeyError(f"{row.kernel}: not timed")
    if which == "wide":
        med, iqr = row.runtime_ns, row.spread_ns
    elif which == "near":
        raw = row.raw
        keys = ("wallclock:near:median_ns", "wallclock:near:q1_ns",
                "wallclock:near:q3_ns")
        if not all(k in raw for k in keys):
            raise KeyError(f"{row.kernel}: no near-T timing")
        med, iqr = raw[keys[0]], raw[keys[2]] - raw[keys[1]]
    else:
        raise ValueError(f"input {which!r} is not one of {INPUTS}")
    if med is None or iqr is None:
        raise KeyError(f"{row.kernel}: not timed on {which}")
    return Timing(n_batches, int(med), int(iqr), int(row.repeats),
                  SESSION[n_batches])


def recommend(store: Mapping[str, measured.Measured], arch: str, L: int,
              which: str) -> Recommendation:
    """The recommendation at (arch, L, input). KeyError on any untimed cell."""
    return Recommendation(arch, L, which, tuple(
        timing(_row(store, arch, L, b), which, b) for b in BATCHES))


def _provenance(store: Mapping[str, measured.Measured], arch: str) -> dict:
    """The one host / card / toolchain / date every timed subgrid row records."""
    seen = {(r.host, r.gpu, r.toolchain, r.date)
            for L in measured.SUBGRID_LS for b in BATCHES
            for r in [_row(store, arch, L, b)]}
    if len(seen) != 1:
        raise ValueError(f"{arch}: the timed subgrid rows disagree on "
                         f"provenance: {sorted(seen)}")
    host, gpu, toolchain, date = seen.pop()
    return {"host": host, "gpu": gpu, "toolchain": toolchain, "date": date}


def _uniform(store: Mapping[str, measured.Measured], what: str,
             get: Callable[[measured.Measured], object]) -> object:
    """The one value `get(row)` takes over every subgrid row on every arch."""
    seen = {get(_row(store, a.name, L, b)) for a in ARCHS
            for L in measured.SUBGRID_LS for b in BATCHES}
    if len(seen) != 1:
        raise ValueError(f"the timed subgrid rows disagree on {what}: "
                         f"{sorted(seen, key=repr)}")
    return seen.pop()


def report(store: Mapping[str, measured.Measured]) -> dict:
    """The whole recommendation document, as RECOMMENDATION records it."""
    warmup, repeats, n_bra, n_ket = (
        _uniform(store, what, get) for what, get in (
            ("warm-up", lambda r: r.raw.get("wallclock:warmup")),
            ("repeats", lambda r: r.repeats),
            ("n_bra", lambda r: r.raw.get("wallclock:n_bra")),
            ("n_ket", lambda r: r.raw.get("wallclock:n_ket"))))
    cells = []
    for arch in ARCHS:
        for L in measured.SUBGRID_LS:
            per = {w: recommend(store, arch.name, L, w) for w in INPUTS}
            cells.append({
                "arch": arch.name,
                "L": L,
                "agree": per["wide"].best.n_batches == per["near"].best.n_batches,
                **{w: {
                    "n_batches": r.best.n_batches,
                    "median_ns": r.best.median_ns,
                    "iqr_ns": r.best.iqr_ns,
                    "repeats": r.best.repeats,
                    "runner_up": r.runner_up.n_batches,
                    "margin_ns": r.margin_ns,
                    "resolved": r.resolved,
                    "within_spread": list(r.within_spread),
                    "same_session": r.same_session,
                    "all": {str(t.n_batches): [t.median_ns, t.iqr_ns]
                            for t in r.timings},
                } for w, r in per.items()},
            })
    return {
        "schema": SCHEMA,
        "about": ABOUT,
        "header": {
            "derived_from": "data/measured/kernels.json, the measured.SUBGRID "
                            "rows (levers=(), the literal kernel; lever rows "
                            "such as .../recycle@... are not read) and their "
                            "wall-clock fields only (runtime_ns, "
                            "spread_ns, repeats, raw wallclock:near:*)",
            "subgrid": {"L": list(measured.SUBGRID_LS),
                        "n_batches": list(BATCHES)},
            "dropped_cells": [],
            "inputs": {
                "wide": "runtime_ns / spread_ns: the sweep's wide-T input "
                        "(T up to ~430, the Boys series long)",
                "near": "raw wallclock:near:median_ns and q3 - q1: the "
                        "near-T input (T < ~2, the Boys series short)",
            },
            "rule": "best = lowest median; resolved iff runner-up median - "
                    "best median > max(IQR of the two) (#60's test); "
                    "within_spread = every n_batches whose median exceeds "
                    "best's by at most max(IQR of best, IQR of it).",
            "protocol": "test/test_kernel_device.py's opt-in wall-clock sweep: "
                        "host steady_clock over a cell's whole batched launch "
                        "sequence into one zeroed J, then one synchronize; "
                        f"{warmup} warm-up + {repeats} timed sequences per "
                        f"input; n_bra x n_ket = {n_bra} x {n_ket}, no "
                        "screening; the timed binary is the one checked "
                        "against reference/ first.",
            "sessions": {str(b): s for b, s in SESSION.items()},
            "cross_session": "Every n_batches of one arch was timed in ONE "
                             "session by #93, behind the card gate (each "
                             "row's raw wallclock:gpu_util_*), replacing #53's "
                             "(b 1-9) and #60's (b 12) timings, so every "
                             "margin is same-session. #60 had measured "
                             "cross-session drift of up to 1.7% (10% on one "
                             "noisy cell).",
            "provenance": {a.name: _provenance(store, a.name) for a in ARCHS},
            "compare": "Within one arch, across n_batches only. Absolute "
                       "times are not comparable across vendors (gfx1200 "
                       "rate-limits fp64).",
            "not_policy": "Spec.n_batches defaults to 1 and this does not "
                          "change it: choosing a count per class is dispatch, "
                          "which lives in dispatch.py (#145) and reads these "
                          "rows per (arch, class, input) for a launch plan "
                          "only, never for the generator, which ships one "
                          "batch by decision (#167).",
            "regenerate": "PYTHONPATH=src uv run python -m "
                          "felizjueves.performance.batchreport --write",
        },
        "cells": cells,
    }


def _ms(ns: int) -> str:
    return f"{ns / 1e6:.3f}"


def markdown(doc: Mapping) -> str:
    """The j_engine.md tables, one per input, from a report document."""
    out = []
    for w in INPUTS:
        out += [f"*{w}-T input:*", "",
                "| arch | L | best `n_batches` | median ms | IQR ms | repeats "
                "| runner-up (+ms) | resolved | within spread | same sweep |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for c in doc["cells"]:
            r = c[w]
            out.append(
                f"| {c['arch']} | {c['L']} | **{r['n_batches']}** | "
                f"{_ms(r['median_ns'])} | {_ms(r['iqr_ns'])} | {r['repeats']} | "
                f"{r['runner_up']} (+{_ms(r['margin_ns'])}) | "
                f"{'yes' if r['resolved'] else 'no'} | "
                f"{', '.join(str(b) for b in r['within_spread'])} | "
                f"{'yes' if r['same_session'] else 'no'} |")
        out.append("")
    return "\n".join(out).rstrip("\n")


def write(store: Mapping[str, measured.Measured],
          path: Path = RECOMMENDATION) -> dict:
    doc = report(store)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m felizjueves.performance.batchreport")
    ap.add_argument("--write", action="store_true",
                    help=f"rewrite {RECOMMENDATION.relative_to(measured.REPO)}")
    ap.add_argument("--markdown", action="store_true",
                    help="print the j_engine.md tables")
    ap.add_argument("--store", type=Path, default=measured.STORE)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        store = measured.load(args.store)
        doc = write(store) if args.write else report(store)
    except (OSError, ValueError, KeyError) as e:
        print(f"batchreport: {e}", file=sys.stderr)
        return 2
    print(markdown(doc) if args.markdown
          else json.dumps(doc, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
