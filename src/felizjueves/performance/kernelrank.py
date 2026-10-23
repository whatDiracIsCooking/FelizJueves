"""`KernelCost` against `Measured`: does the model RANK batch counts? (#54)

    python -m felizjueves.performance.kernelrank            # the report, as JSON

(with `PYTHONPATH=src uv run` in front, as for every entry point here.)

THE ONE QUESTION A MODEL OF THIS KIND CAN ANSWER. `lower.kernel_cost` reads
the IR; a `Measured` row is what nvcc or hipcc said about the compiled text.
Their units do not meet (fp64 words under a declared budget, against spill
BYTES on nvcc and spilled REGISTER COUNTS on hipcc), so the only claim is an
ORDER: within one `(L, arch)`, do the batch counts of `measured.SUBGRID` come
out in the same order under the model as in the rows? Spearman's rho (average
ranks for ties) and a pair count over the cell's batch counts. Four points
(or however many SUBGRID declares) is weak evidence by construction: rho
over n = 4 takes few values, and nothing here is a prediction interval.

THE REGISTER BUDGET IS DECLARED, NOT FITTED: `BUDGET`, in fp64 values per
thread. A value of the generated kernel is an fp64 scalar, two 32-bit
registers, so the budget is half the register ceiling the compiler is working
under: 255 on sm_86 (the ISA's per-thread maximum, which nvcc reports for
every spilling subgrid kernel) and 192 on gfx1200 (the VGPR count hipcc
settles at for every spilling subgrid kernel -- read from the rows, and
checked against them by the test, not an ISA maximum). Everything else the
kernel holds (indices, pointers, the Boys tail) is not priced: the model is
coarse and is not to be tuned to 16 cells (#54's hazard).

AGGREGATION -- a split cell is n_batches kernels and ONE row. The two sides
are aggregated the same way, and both ways are reported:

- `sum`: the cell's whole spill. Model: `traffic` summed over the batch
  kernels. Measured: the row's `spill_store + spill_load`, which `measured`
  already sums over the batch kernels. THIS IS THE HEADLINE CLAIM: it is the
  number the store carries, and the one a whole J build pays.
- `max`: the worst single launch. Model: the max per-kernel `traffic`.
  Measured: the max over the batch kernels of that kernel's store + load,
  read from `raw` (`<kernel>:spill_store_bytes` + `:spill_load_bytes` on
  nvcc; `2 x (VGPRs Spill + SGPRs Spill)` on hipcc, the factor matching the
  row's own duplication into both spill fields).

`peak_live` is ranked separately, aggregated as the MAX over the batch
kernels -- it is a per-launch quantity, like `registers`, which the store
also aggregates by max -- and so against the `max` column only. A sum of
per-kernel peaks prices nothing a launch sequence pays, and is not a claim.

WALL-CLOCK. Separately: do spill WINS predict wall-clock? For each
`(L, arch)`, rank the batch counts by the row's summed spill and by its
median wall-clock, on BOTH recorded inputs -- `runtime_ns` (the wide-T input)
and `raw["wallclock:near:median_ns"]` (the near-T input) -- and ask whether
the fewest-spill batch count is the fastest.

A MIS-RANKING IS A RECORD, NOT A TEST. data/measured/limits.json holds one
`literature.limits.StructuralLimit` per mis-ranking found here, with its
mechanism written out; measure/measure_kernelrank.py recomputes every number it
records, and fails on a mis-ranking over SUBGRID that no record covers.

NOTHING HERE FEEDS A LITERATURE FLOOR. This module reads `measured_store`
(the store, not its driver: #94) and `lower`; nothing under `plan/` or
`literature/` may import it (test/test_layering.py), and it imports neither.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from felizjueves.ir.lower import kernel_cost
from felizjueves.mmd.kernel import build_pq_kernels
from felizjueves.performance import measured_store as measured
from felizjueves.performance.device import ARCHS, Arch
from felizjueves.spec import Spec

# fp64 values per thread: half the 32-bit register ceiling (module docstring).
CEILING = {"sm_86": 255, "gfx1200": 192}
BUDGET = {arch: regs // 2 for arch, regs in CEILING.items()}

AGGREGATIONS = ("sum", "max")
INPUTS = {"wide": "runtime_ns", "near": "wallclock:near:median_ns"}


# ---------------------------------------------------------------------------
# The two sides of one cell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCell:
    """`KernelCost` of a cell's batch kernels, aggregated both ways."""

    registers: int
    traffic_sum: int
    traffic_max: int
    peak_live_max: int


@cache
def model_cell(spec: Spec, registers: int) -> ModelCell:
    costs = [kernel_cost(k, registers=registers) for k in build_pq_kernels(spec)]
    return ModelCell(
        registers=registers,
        traffic_sum=sum(c.traffic for c in costs),
        traffic_max=max(c.traffic for c in costs),
        peak_live_max=max(c.peak_live for c in costs),
    )


def _kernel_spill(row: measured.Measured, prefix: str) -> int:
    raw = row.raw
    if f"{prefix}spill_store_bytes" in raw:  # nvcc
        return raw[f"{prefix}spill_store_bytes"] + raw[f"{prefix}spill_load_bytes"]
    if f"{prefix}VGPRs Spill" in raw:  # hipcc: counts, in both fields
        return 2 * (raw[f"{prefix}VGPRs Spill"] + raw.get(f"{prefix}SGPRs Spill", 0))
    raise ValueError(f"row for {row.kernel!r}: no per-kernel spill for {prefix!r}")


def measured_spill(row: measured.Measured, aggregation: str) -> int:
    """The row's spill, aggregated as `aggregation` over its batch kernels."""
    if row.spill_store is None or row.spill_load is None:
        raise ValueError(f"row for {row.kernel!r} has no compile-side spill")
    if aggregation == "sum":
        return row.spill_store + row.spill_load
    if aggregation != "max":
        raise ValueError(f"aggregation {aggregation!r} is not one of {AGGREGATIONS}")
    names = row.kernel.split(",")
    if len(names) == 1:
        return _kernel_spill(row, "")
    return max(_kernel_spill(row, f"{k}:") for k in names)


def runtime(row: measured.Measured, which: str) -> float | None:
    """The row's median wall-clock on input `which` ("wide" | "near")."""
    if which == "wide":
        return row.runtime_ns
    got = row.raw.get(INPUTS[which])
    return None if got is None else float(got)


# ---------------------------------------------------------------------------
# Ranks
# ---------------------------------------------------------------------------


def ranks(xs: Sequence[float]) -> list[float]:
    """1-based ranks, ties given their average rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman's rho with average ranks; None when either side is constant
    (no order to agree with)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("spearman needs two equal-length sequences, n >= 2")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return round(sxy / (sxx * syy) ** 0.5, 6)


def pairs(model: Sequence[float], meas: Sequence[float]) -> dict[str, int]:
    """Every pair of batch counts the MEASUREMENT orders strictly, sorted by
    what the model says: `concordant` (same order), `discordant` (opposite),
    `unresolved` (the model ties them). Pairs the measurement ties make no
    claim and are not counted."""
    got = {"concordant": 0, "discordant": 0, "unresolved": 0}
    n = len(model)
    for i in range(n):
        for j in range(i + 1, n):
            m = (meas[i] > meas[j]) - (meas[i] < meas[j])
            if m == 0:
                continue
            s = (model[i] > model[j]) - (model[i] < model[j])
            got["unresolved" if s == 0 else
                "concordant" if s == m else "discordant"] += 1
    return got


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _cells(store: Mapping[str, measured.Measured], arch: Arch,
           batch_counts: Sequence[int] | None,
           ) -> dict[int, list[tuple[Spec, measured.Measured]]]:
    """SUBGRID's cells by L, in batch-count order, each with its one row on
    `arch` -- restricted to `batch_counts` when given."""
    by_l: dict[int, list[tuple[Spec, measured.Measured]]] = {}
    for spec in sorted(measured.SUBGRID, key=lambda s: (s.total_am, s.n_batches)):
        if batch_counts is not None and spec.n_batches not in batch_counts:
            continue
        got = list(measured.rows(spec, arch, store=store).values())
        if len(got) != 1:
            raise ValueError(f"{spec.key()} on {arch.name}: {len(got)} rows, "
                             "expected exactly one")
        by_l.setdefault(spec.total_am, []).append((spec, got[0]))
    return by_l


# (model metric, aggregation) -> the ModelCell field; the aggregation names
# the measured column it is ranked against (module docstring).
CLAIMS = {("traffic", "sum"): "traffic_sum", ("traffic", "max"): "traffic_max",
          ("peak_live", "max"): "peak_live_max"}


def rank_claim(store: Mapping[str, measured.Measured] | None = None,
               batch_counts: Sequence[int] | None = None) -> list[dict]:
    """One entry per (arch, L, model metric, aggregation): the batch counts,
    both sides' values in that order, rho, and the pair count."""
    store = measured.load() if store is None else store
    out = []
    for arch in ARCHS:
        budget = BUDGET[arch.name]
        for L, cells in sorted(_cells(store, arch, batch_counts).items()):
            for (metric, agg), field in CLAIMS.items():
                model = [getattr(model_cell(s, budget), field) for s, _ in cells]
                meas = [measured_spill(r, agg) for _, r in cells]
                out.append({
                    "arch": arch.name, "L": L, "metric": metric,
                    "aggregation": agg, "registers": budget,
                    "n_batches": [s.n_batches for s, _ in cells],
                    "model": model, "measured": meas,
                    "spearman": spearman(model, meas),
                    **pairs(model, meas),
                })
    return out


def wallclock_claim(store: Mapping[str, measured.Measured] | None = None,
                    batch_counts: Sequence[int] | None = None) -> list[dict]:
    """One entry per (arch, L, input): do fewer summed spills mean faster?
    Batch counts with no timing on that input are left out (and named)."""
    store = measured.load() if store is None else store
    out = []
    for arch in ARCHS:
        for L, cells in sorted(_cells(store, arch, batch_counts).items()):
            for which in INPUTS:
                timed = [(s, r) for s, r in cells if runtime(r, which) is not None]
                if len(timed) < 2:
                    continue
                spill = [measured_spill(r, "sum") for _, r in timed]
                time = [runtime(r, which) for _, r in timed]
                bs = [s.n_batches for s, _ in timed]
                least = min(spill)
                fewest = [b for b, x in zip(bs, spill, strict=True) if x == least]
                fastest = bs[time.index(min(time))]
                out.append({
                    "arch": arch.name, "L": L, "input": which,
                    "n_batches": bs, "spill_sum": spill, "median_ns": time,
                    "untimed": [s.n_batches for s, r in cells
                                if runtime(r, which) is None],
                    "fewest_spill": fewest, "fastest": fastest,
                    "spill_win_is_fastest": fastest in fewest,
                    "spearman": spearman(spill, time),
                })
    return out


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    doc = {"registers": BUDGET, "ceiling": CEILING,
           "rank": rank_claim(), "wallclock": wallclock_claim()}
    json.dump(doc, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
