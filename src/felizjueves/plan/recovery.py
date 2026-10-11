"""#14: recover the inputs behind [JEngine]'s printed cells (exact_given_inputs).

The literal board fixes every choice the paper leaves open arbitrarily. This
module searches for the choices the paper's OWN numbers came from, as rules --
never as a per-cell lookup of the answers. Nothing here changes a default:
the literal reconstruction (batching.MINIMAX_TARGETS, sieve.sieve in
`hermite_targets` order) still scores exactly as it did.

FOUND: THE CUT (Table 3, all 40 cells). [JEngine, v2, §5] says the cut
minimises the largest batch, but at 9 batches the paper prints one ABOVE the
minimax optimum at L = 7 (21 vs 20) and L = 15 (116 vs 115). The rule that
reproduces all 40 cells is TWO-ENDED NEAREST FILL, applied PER DIRECTION:

- each direction is cut on its OWN target counts by max component m (X, Y
  and Z do not share cut points; X holds the tied and all-equal targets, so
  its counts differ from Y's and Z's at small m);
- with T = (that direction's targets) / k, peel sub-batches alternately from
  the bottom (m = 0 upward) and the top (m = L downward); each peel takes
  the run of m values whose total is NEAREST T (a tie takes the shorter
  run), and is at least one m value;
- the last sub-batch is whatever remains in the middle.

At k = 2 that is one bottom peel; at k = 3, a bottom peel, a top peel and the
middle. The evidence it is the paper's rule and not a fit:

- Table 3: 40 of 40 (minimax: 38). At k = 2 every printed cell is reached by
  ONE shared cut only, and this rule finds it; at (7, 9) and (15, 9) it
  finds the non-minimax cut the printed value needs.
- Tables 1-2 move with it, and only the right way: at (7, 9) the printed
  FLOPs (560) and max intermediates (40) are reproduced, where minimax's cut
  gives 510 and 38; at (10, 9), which Table 3 cannot see because the max
  target count is 39 either way, the per-direction Y and Z cuts reproduce
  the printed 1669 FLOPs (shared cuts: 1613) and 95 intermediates. Neither
  table was an input to the rule.

Shared cut points computed from X's counts also give 40/40 on Table 3, but
not (10, 9) in Table 1; per-direction is the recorded input.

NOT FOUND: THE SIEVE'S INPUTS (Tables 1-2). With the recovered cut, the
printed post-sieve FLOPs and intermediates depend on the sieve's seed order
-- |targets|! permutations at one batch, 969! at L = 16 -- and the paper
states it did not optimise or fix that order ([JEngine, v2, §4.1]). The
bounded search over the structured family `seed_orders` is reproducible
here; what it reached, and the StructuralLimit it ends in, are in
data/literature/jengine/limits.json and test/test_plan_recovery.py.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from itertools import permutations

from felizjueves.literature.rules import Candidate
from felizjueves.literature.tables import Cell, Table
from felizjueves.plan.batching import ALL, AXES, Batch, target_counts
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.dag import BatchDag, batch_dag
from felizjueves.plan.sieve import sieve
from felizjueves.plan.targets import R, hermite_targets
from felizjueves.spec import is_batch_count

# What literal.json's exact_given_inputs records for Table 3. A candidate
# carries exactly this mapping when it was produced by `recovered_plan`.
CUT_INPUTS: Mapping[str, str] = {
    "cut": "two-ended-nearest-fill",
    "cut_scope": "per-direction",
}

SeedOrder = Callable[[int], Sequence[R]]  # L -> an order of hermite_targets(L)

QUANTITY = {
    "jengine/table1": "flops",
    "jengine/table2": "max_intermediates",
    "jengine/table3": "max_targets",
}


def _peel(weights: Sequence[int], target: Fraction) -> int:
    """How many leading entries of `weights` sum NEAREST `target`; at least
    one; a tie takes the shorter run."""
    acc = 0
    for n, w in enumerate(weights):
        if n and acc + w > target:
            return n if abs(acc - target) <= abs(acc + w - target) else n + 1
        acc += w
    return len(weights)


def two_ended_cuts(weights: Sequence[int], k: int) -> tuple[int, ...]:
    """Cut points c_1 < ... < c_{k-1} for one direction (batching.py's
    convention: sub-batch i covers c_{i-1} .. c_i - 1), by two-ended nearest
    fill on `weights[m]`, the direction's target count at max component m."""
    if k < 1:
        raise ValueError(f"k = {k}: at least one sub-batch")
    lo, hi = 0, len(weights)
    target = Fraction(sum(weights), k)
    cuts = []
    for i in range(k - 1):
        if i % 2 == 0:
            lo += _peel(weights[lo:hi], target)
            cuts.append(lo)
        else:
            hi -= _peel(weights[lo:hi][::-1], target)
            cuts.append(hi)
        if not lo < hi:
            raise ValueError(f"{k} sub-batches leave none for the middle")
    return tuple(sorted(cuts))


def recovered_batches(total_am: int, n_batches: int) -> tuple[Batch, ...]:
    """The batches under the recovered cut, direction-major like
    batching.partition, each direction cut on its own counts."""
    if not is_batch_count(n_batches):
        raise ValueError(f"n_batches {n_batches!r} is neither 1 nor 3k")
    if n_batches == 1:
        return ((ALL, 0, total_am),)
    counts = target_counts(total_am)
    out = []
    for d in AXES:
        edges = (0, *two_ended_cuts(counts[d], n_batches // 3), total_am + 1)
        out.extend((d, edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1))
    return tuple(out)


def recovered_plan(
    total_am: int, n_batches: int, *, seed_order: Sequence[R] | None = None
) -> tuple[BatchDag, ...]:
    """Pipeline steps 1-3 with the recovered cut: MILD DAG per batch, then the
    sieve, seeded in `seed_order` (default: the literal `hermite_targets`).
    An empty batch is kept and not sieved, as plan/search.py does."""
    dags = (batch_dag(total_am, b) for b in recovered_batches(total_am, n_batches))
    return tuple(sieve(d, seed_order=seed_order) if d.targets else d for d in dags)


def recovered_costs(
    cells: Sequence[Cell], seed_order: SeedOrder | None = None
) -> dict[Cell, PlanCost]:
    return {
        (L, nb): cost(recovered_plan(
            L, nb, seed_order=None if seed_order is None else seed_order(L)))
        for L, nb in cells
    }


def recovered_candidates(tables: Mapping[str, Table]) -> dict[str, Candidate]:
    """Every table's cells from `recovered_plan`, on the literal board. Only
    Table 3's inputs are recorded (CUT_INPUTS); Tables 1-2 carry the same
    mapping, which is true of how they were produced but is not, there, the
    paper's whole input (see limits.json)."""
    cells = sorted({c for t in tables.values() for c in t.cells})
    costs = recovered_costs(cells)
    return {
        tid: Candidate(
            {c: getattr(costs[c], QUANTITY[tid]) for c in t.cells},
            board="literal",
            inputs=dict(CUT_INPUTS),
        )
        for tid, t in tables.items()
    }


# -- the bounded seed-order search (Tables 1-2) ----------------------------------

def seed_orders() -> Iterator[tuple[str, SeedOrder]]:
    """The structured family searched: `hermite_targets` order and its
    reverse, and every sort by (|r| asc/desc, then the components in one of
    the 6 axis orders, all asc or all desc) -- 26 orders. Each value maps L to
    an order of hermite_targets(L)."""
    yield "hermite", hermite_targets
    yield "hermite-reversed", lambda L: hermite_targets(L)[::-1]
    for perm in permutations(range(3)):
        for n_sign in (1, -1):
            for c_sign in (1, -1):
                def order(L, perm=perm, n_sign=n_sign, c_sign=c_sign):
                    return sorted(
                        hermite_targets(L),
                        key=lambda r: (n_sign * sum(r),
                                       *(c_sign * r[p] for p in perm)),
                    )
                name = (f"|r| {'asc' if n_sign > 0 else 'desc'}, then "
                        f"{''.join(AXES[p] for p in perm)} "
                        f"{'asc' if c_sign > 0 else 'desc'}")
                yield name, order


def exact_counts(
    tables: Mapping[str, Table], costs: Mapping[Cell, PlanCost]
) -> dict[str, int]:
    return {
        tid: sum(getattr(costs[c], QUANTITY[tid]) == p for c, p in t.cells.items())
        for tid, t in tables.items()
    }
