"""The plan's cost, counted from its DAGs: [JEngine, v2, Tables 1-3].

- `flops`             summed over batches        -> Table 1
- `max_intermediates` max over batches           -> Table 2
- `max_targets`       max over batches           -> Table 3

Counted BEFORE the sieve: the DAGs are plan/dag.py's, MILD with nothing pruned,
so Tables 1-2 (which are post-sieve) can only partly agree. The Boys function
is excluded ([JEngine, v2, §7.1]): a fundamental costs nothing here. The [p|q]
assembly ([JEngine, v2, eqs. 7-9], pipeline step 4) is not counted either.

TWO CONVENTIONS THE PAPER LEAVES OPEN, fixed here for the literal board.

1. FLOPS PER NODE: every operation the reduction form writes, except a
   multiply by 1. For [r]^(m) = PQ_d . [r - 1d]^(m+1) + (r_d - 1) . [r - 2d]^(m+1):

       r_d = 1    1 FLOP    PQ_d .            (the second term vanishes)
       r_d = 2    2 FLOPs   PQ_d . , +        (the coefficient r_d - 1 is 1)
       r_d >= 3   3 FLOPs   PQ_d . , (r_d - 1) . , +

   A multiply-add is two FLOPs, not one fused op. Evidence: pre-sieve, this
   reproduces 5 of Table 1's 40 cells exactly, among them the anchor
   (L = 7, 3 batches) = 425, and is at or above the printed value in every
   cell but (7, 9) and (10, 9), where the paper's partition is not ours
   (Table 3 already differs at (7, 9)). The readings "multiply-add = 1" and
   "multiply-add = 2, coefficient never counted" -- the latter is
   docs/implementation/mild.md's "one edge per multiply" -- reproduce no cell
   and fall below the printed post-sieve value in all 40 (368 against 424 at
   L = 7, one batch); "always count the coefficient" stays above it
   everywhere but reproduces no cell.

2. FUNDAMENTALS [0]^(m), m > 0, ARE INTERMEDIATES. They have m > 0 and are held
   live like any other; only their evaluation is the Boys function's. Evidence:
   counted, they reproduce 4 of Table 2's 40 cells and sit at or above the
   printed value in every cell but (7, 9); left out, they reproduce 2 and fall
   BELOW the printed post-sieve value in 16 -- e.g. 46 against 51 at (7, 3) --
   which no pass that only deletes intermediates could close.
"""

from __future__ import annotations

from dataclasses import dataclass

from felizjueves.plan.dag import BatchDag, is_fundamental


@dataclass(frozen=True)
class PlanCost:
    flops: int  # summed over batches -> Table 1
    max_intermediates: int  # max over batches -> Table 2
    max_targets: int  # max over batches -> Table 3


def node_flops(r_d: int) -> int:
    """Convention 1 above: 1, 2 or 3 FLOPs for r_d = 1, 2, >= 3."""
    if r_d < 1:
        raise ValueError(f"r_d = {r_d}: a reduced component is at least 1")
    return min(r_d, 3)


def batch_flops(dag: BatchDag) -> int:
    return sum(
        node_flops(dag.reduced_component(n))
        for n in dag.nodes()
        if not is_fundamental(n)
    )


def cost(dags: tuple[BatchDag, ...]) -> PlanCost:
    if not dags:
        raise ValueError("a plan has at least one batch")
    return PlanCost(
        flops=sum(batch_flops(d) for d in dags),
        max_intermediates=max(len(d.intermediates()) for d in dags),
        max_targets=max(len(d.targets) for d in dags),
    )
