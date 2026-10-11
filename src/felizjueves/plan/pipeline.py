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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from felizjueves.plan.cost import PlanCost, batch_flops, cost
from felizjueves.plan.dag import BatchDag
from felizjueves.plan.sieve import plan as sieved_dags
from felizjueves.spec import Spec


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
    """Steps 1-3 for `spec`, sieved unless it names `no-sieve`; any other lever
    is refused by dag.build, as it is everywhere else in the plan layer."""
    dags = sieved_dags(spec)
    return Plan(spec, dags, cost(dags), tuple(BatchReport.of(d) for d in dags))
