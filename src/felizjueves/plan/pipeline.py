"""Pipeline steps 1-3 for one Spec, [JEngine, v2, §6]: split, MILD DAG, sieve.

This module composes the stages; it implements none of them.

    step 1  batching.partition     the targets, split into n_batches
    step 2  dag.build              one MILD DAG per batch
    step 3  sieve.sieve            Alg. 3 on each DAG, unless `no-sieve`

`sieve.plan(spec)` already runs those three in order and owns the `no-sieve`
lever, so `plan` here calls it and adds the two things a caller wants next:
the `PlanCost` ([JEngine, v2, Tables 1-3]) and a per-batch report.

THE PER-BATCH REPORT is docs/implementation/j_engine.md's "FLOP and
intermediate counts per batch... the only cost signal available without a
card". It is machine-readable on purpose (`Plan.report()` is plain JSON
types): an emitter will key on it later. Steps 4-5 -- the [p|q] assembly and
emission -- are not here, and nothing is counted for them.

THE `tuned` LEVER (#161) is the `jgen` wire: the plan the SEARCHED board
replayed, built instead of the literal one. `tuned_plan(spec)` reads the
cell's recorded choice -- one cut and one explicit seed order per batch --
from data/scoreboards/jengine/searched_cuts.json through `search.load_cuts`
and builds it with `search.searched_plan`, which passes each batch's order to
`sieve.sieve(dag, seed_order=...)`. Its `PlanCost` is therefore the board's
recorded counts for that cell: the plan the board scored, not a third thing.

THIS IS A USE OF searched_cuts.json FOR EMISSION, NOT A CLAIM ABOUT THE
PAPER'S CELLS. The board scores the searched plans against [JEngine] Tables
1-3; this lever only builds a kernel from the same choice so it can be
compiled and timed (#163). Nothing here says the paper's authors used that
cut or that order, and a re-selection of the file (a later search) changes
the tuned kernels with it -- by design, since the file is the input.

A CELL WITH NO RECORDED CHOICE IS REFUSED, never built as literal, so every
tuned kernel is traceable to a recorded search. `tuned` is an arm of its own:
it neither implies nor admits `no-sieve` (the recorded order is a SIEVE seed
order, so the pair is refused as two experiments at once), and any other
lever is refused here as it is everywhere else in the plan layer. With no
lever, `plan` is exactly `sieve.plan`: the default output does not move.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from felizjueves.plan.cost import PlanCost, batch_flops, cost
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.search import Choice, load_cuts, searched_plan
from felizjueves.plan.sieve import NO_SIEVE
from felizjueves.plan.sieve import plan as sieved_dags
from felizjueves.spec import Spec

TUNED = "tuned"
"""The `Spec.levers` name of the searched board's plan (#161)."""

SEARCHED_CUTS = (Path(__file__).resolve().parents[3] / "data" / "scoreboards"
                 / "jengine" / "searched_cuts.json")
"""The recorded choice per (L, n_batches), in `search.load_cuts`' schema."""


@cache
def _recorded(path: Path) -> dict[tuple[int, int], Choice]:
    return load_cuts(path)


def tuned_plan(spec: Spec, *, path: Path = SEARCHED_CUTS
               ) -> tuple[BatchDag, ...]:
    """The searched board's plan for `spec`'s cell (L, n_batches): its
    recorded cut and, per batch, its recorded seed order, sieved.

    `spec` must name `tuned` and no other lever. A cell `path` holds no
    choice for is refused: a tuned build never falls back to literal.
    """
    if TUNED not in spec.levers:
        raise ValueError(f"{spec.key()}: tuned_plan needs the {TUNED!r} lever")
    rest = tuple(lv for lv in spec.levers if lv != TUNED)
    if NO_SIEVE in rest:
        raise ValueError(
            f"{spec.key()}: {TUNED!r} with {NO_SIEVE!r} is two arms at once -- "
            "the recorded choice is a sieve seed order on a sieved plan")
    if rest:
        raise ValueError(f"no lever but {TUNED!r} is implemented on the tuned "
                         f"plan: {rest}")
    cell = (spec.total_am, spec.n_batches)
    choice = _recorded(path).get(cell)
    if choice is None:
        raise ValueError(
            f"{spec.key()}: no recorded choice for cell L={cell[0]}, "
            f"n_batches={cell[1]} in {path.name}; a tuned build is always a "
            "recorded search, never a fall back to the literal plan")
    cut, orders = choice
    return searched_plan(*cell, cut, orders)


@dataclass(frozen=True)
class BatchReport:
    direction: str  # "x", "y", "z", or "xyz" for the unsplit class
    a: int  # the batch covers targets whose max component lies in a..b
    b: int
    targets: int
    intermediates: int
    flops: int

    @classmethod
    def of(cls, dag: BatchDag) -> BatchReport:
        direction, a, b = dag.batch
        return cls(
            direction, a, b,
            targets=len(dag.targets),
            intermediates=len(dag.intermediates()),
            flops=batch_flops(dag),
        )


@dataclass(frozen=True, eq=False)
class Plan:
    spec: Spec
    dags: tuple[BatchDag, ...]  # one per batch, in partition order
    cost: PlanCost
    batches: tuple[BatchReport, ...]  # parallel to `dags`

    def report(self) -> dict[str, Any]:
        """The plan as JSON-ready values: the Spec key, the totals, the batches."""
        return {
            "spec": self.spec.key(),
            "total_am": self.spec.total_am,
            "n_batches": self.spec.n_batches,
            "levers": list(self.spec.levers),
            "cost": {
                "flops": self.cost.flops,
                "max_intermediates": self.cost.max_intermediates,
                "max_targets": self.cost.max_targets,
            },
            "batches": [
                {
                    "direction": b.direction,
                    "a": b.a,
                    "b": b.b,
                    "targets": b.targets,
                    "intermediates": b.intermediates,
                    "flops": b.flops,
                }
                for b in self.batches
            ],
        }


def plan(spec: Spec) -> Plan:
    """Steps 1-3 for `spec`, sieved unless it names `no-sieve`, and the
    searched board's recorded plan if it names `tuned` (`tuned_plan`); any
    other lever is refused by dag.build, as it is everywhere else in the
    plan layer."""
    dags = tuned_plan(spec) if TUNED in spec.levers else sieved_dags(spec)
    return Plan(spec, dags, cost(dags), tuple(BatchReport.of(d) for d in dags))
