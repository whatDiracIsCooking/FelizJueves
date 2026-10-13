"""The `searched` board's plans: search what the paper leaves open.

[JEngine, v2, §5] fixes WHAT the batch cut optimises -- minimax targets per
batch -- and leaves open WHICH cut is taken; [JEngine, v2, §4.1] leaves the
sieve's seed order open and says the authors did not optimise it. The literal
board fixes both arbitrarily (plan/batching.py: lowest minimax boundaries;
plan/sieve.py: `hermite_targets` order). This module searches both instead,
and scores the full plan, not the cut: a candidate is partition -> MILD DAG
per batch -> sieve -> PlanCost, so Tables 1 and 2 are the post-sieve numbers
the paper prints.

WHAT IS SEARCHED, per (L, n_batches), bounded and deterministic:

- THE CUT. Every contiguous cut `batching.partition` accepts: k - 1
  ascending cut points in 1..L, ONE set shared by X, Y and Z (the shape
  [JEngine, v2, Fig. 4] draws, and the only one `partition` builds), and
  leaving no batch empty (#64: a first cut at 1 empties Y_{0,0} and Z_{0,0},
  and `partition` refuses it). C(L, k - 1) - C(L - 1, k - 2) of them, at most
  C(16, 2) - 15 = 105; with 1 batch, or 3 (k = 1), the only cut is (). Only
  the FEASIBLE ones are searched further (largest batch within
  MINIMAX_TOLERANCE of the cell's minimax -- a cut's target count is its
  own, no seed order moves it); on the L = 7..16 grid that is one cut per
  cell but (7, 9), (8, 9) and (15, 9).
- THE SEED ORDER, per batch, as an explicit PERMUTATION of the batch's
  targets (#172). Until #172 a cell took one of plan/recovery.py's 58
  `seed_orders(widened=True)` rules (#168) for all its batches; those rules
  are now the STARTS of a climb, not the space:
    * STARTS: the 58 rules restricted to the batch's targets, one per
      distinct order (`batch_starts`). All are evaluated; the ones on the
      batch's frontier of them are climbed (`climb_starts`).
    * THE CLIMB: plan/order_search.py's `climb` (#170) -- every
      transposition and every single-element move, first improvement in a
      seeded random order, strictly better steps only, each start climbed
      once per weight of order_search's SWEEP (pure intermediates to pure
      FLOPs, in quarters) and each run stopped at a local optimum or after
      CLIMB_BUDGET neighbours examined; seed CLIMB_SEED. A batch's starts are
      climbed one at a time (`Unit`), each against the reference point of
      all of them, so the runs are exactly those of one `climb` over them,
      and the build step spreads them over processes without changing a
      byte of the result.
    * THE CELL: per batch, the frontier of every order its climbs
      evaluated; per cut, order_search's `product` of its batches'
      frontiers (max intermediates, summed FLOPs -- exact, not a heuristic;
      see order_search); over the feasible cuts, the frontier.
  Every rule plan on a feasible cut is evaluated, so the climbed frontier
  weakly dominates the whole 58-rule space there: the climb only adds to
  what #171's rule selects from. It is still a choice of this board only:
  #108's rules-only doctrine for the literal board's `exact_given_inputs`
  is untouched, since a climbed order is a per-cell search, not a rule.

WHAT IS NOT SEARCHED. Per-direction cuts (plan/recovery.py's cut is one; it
is the paper's recovered INPUT, judged on the literal board, not a choice
this board makes), orders more than CLIMB_BUDGET examined neighbours from a
climbed start (the factorial space beyond; #14's record searched random
orders and stopped there on purpose), the sieve's re-expansion choice and
sweep, and `no-sieve`. Everything else is the literal reconstruction, sieve
on.

ONE PLAN PER CELL, chosen in two steps (#171): the FRONTIER of the feasible
plans, then ONE point of it by a SELECTION RULE stated in #171 before the
board was re-read, and not tuned to it. #172 changes the space the frontier
is taken over, not the rule. The rule, verbatim:

    Among feasible plans (max targets within `MINIMAX_TOLERANCE` of the
    cell's minimax), take the Pareto frontier over (max_intermediates,
    FLOPs) and select the point minimising `intermediates /
    intermediates_min + flops / flops_min`, intermediates breaking ties.
    Equal weight because the paper prints the two in separate tables and
    trades them against each other without naming a price; normalised
    because the two quantities differ by an order of magnitude and an
    unnormalised sum would be a FLOPs-only rule wearing a disguise.

Spelled out as the code runs it:

    0. FEASIBLE: max targets per batch <= the cell's minimax + 1
       (MINIMAX_TOLERANCE). Minimax is the paper's own cut criterion; the
       tolerance admits a cut one target off it.
    1. FRONTIER (order_search's `frontier`, since #172 the only Pareto set
       in the package): the feasible plans whose point (max_intermediates,
       FLOPs) -- Tables 2 and 1 -- no other feasible plan's point weakly
       dominates; one plan per distinct point, the first met. Ascending in
       intermediates, strictly descending in FLOPs.
    2. SELECT (`select`): the frontier point with the smallest
       i / i_min + f / f_min, in exact rationals, with i_min and f_min the
       frontier's (equally, the feasible plans') fewest intermediates and
       fewest FLOPs; equal scores to the fewer intermediates.

The whole frontier is recorded beside the chosen point, so a reader with a
real exchange rate between the two quantities can re-select without
re-searching.

Why a frontier at all: neither lexicographic order works. Until #171 the
objective was intermediates first, then FLOPs (the per-batch live set is the
register-pressure bound batching exists to cut, [JEngine, v2, §5]); at
#168's cuts it left 9 cells of Table 1 over, and FLOPs first leaves 6 but
puts 2 back on Table 2. The tolerance is bounded, not dropped: without the
minimax constraint the cheapest-intermediates cut is a lopsided one that
puts the targets in one batch.

What a cell motivated, said plainly: the first search of #13 put targets
FIRST (tolerance 0), and its one large residual, (15, 9) Table 2 at +7.50%,
was an artefact of that -- the cut (6, 9) there is one target off minimax.
The tolerance was set to the smallest value that admits it (tolerance 2
picked identically under the old objective; 3 broke (9, 9) on Table 3).
The selection rule is #171's, quoted above, and was not changed after the
board was read. What #172 chose after reading the board, also said plainly:
the starts and the budget. Climbing EVERY distinct rule order with a budget
of 200 left (14, 1) and (15, 1) over Table 1, each with a frontier point at
or under all three tables that the rule passes over; climbing only the
batch's frontier starts, ten times deeper (order_search's DEFAULT_BUDGET),
is cheaper and is the space recorded.

An EMPTY batch (Y_{0,0} and Z_{0,0} whenever the first cut is 1; Fig. 4
does not draw them at L = 6) is not a candidate: `partition` refuses it
(#64), so the cuts that make one are not enumerated.

THE RECORD (schema 3, #172). Per (L, n_batches), data/scoreboards/jengine/
searched_cuts.json carries the cell's frontier -- every point's cut, the
start each batch's order was climbed from, its counts, and one explicit seed
order per batch -- and the index of the selected point. `load_cuts` and
`load_frontiers` refuse a recorded order that is not a permutation of its
batch's targets (`check_orders`, naming the cell and the batch), and a
replay (#15) builds the plan with `searched_plan(L, n_batches, cut,
orders)` without searching again. The build step that writes it is
`python -m felizjueves.plan search` (`build`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path

from felizjueves.plan.batching import (
    Batch,
    Counts,
    CutRule,
    empty_batches,
    partition,
    target_counts,
)
from felizjueves.plan.cost import PlanCost, cost
from felizjueves.plan.dag import BatchDag, batch_dag
from felizjueves.plan.order_search import (
    DEFAULT_BUDGET,
    NEIGHBOURHOOD,
    SWEEP,
    Evaluated,
    Point,
    climb,
    frontier,
    point,
    product,
)
from felizjueves.plan.recovery import SeedOrder, seed_orders
from felizjueves.plan.sieve import sieve
from felizjueves.plan.targets import R

Cuts = tuple[int, ...]
Order = tuple[R, ...]  # one batch's seed order: a permutation of its targets
Orders = tuple[Order, ...]  # one per batch, in partition order
CUTS_SCHEMA = 3

# How many targets over the cell's minimax a chosen cut may hold. See above.
MINIMAX_TOLERANCE = 1

LITERAL_SEED = "hermite"
SEED_ORDERS: Mapping[str, SeedOrder] = dict(seed_orders(widened=True))

# The climb (#172), declared: order_search.climb's neighbourhood and sweep,
# this budget per run, this seed.
CLIMB_BUDGET = DEFAULT_BUDGET
CLIMB_SEED = 172
CLIMB_SWEEP: tuple[Fraction, ...] = SWEEP

# #171's selection rule, verbatim from the issue, declared before the board
# was re-read. `select` is its code; the record quotes this text.
SELECTION_RULE = (
    "Among feasible plans (max targets within `MINIMAX_TOLERANCE` of the "
    "cell's minimax), take the Pareto frontier over (max_intermediates, "
    "FLOPs) and select the point minimising `intermediates / "
    "intermediates_min + flops / flops_min`, intermediates breaking ties. "
    "Equal weight because the paper prints the two in separate tables and "
    "trades them against each other without naming a price; normalised "
    "because the two quantities differ by an order of magnitude and an "
    "unnormalised sum would be a FLOPs-only rule wearing a disguise."
)


def fixed_cut(cuts: Cuts) -> CutRule:
    """A CutRule that ignores the counts and returns `cuts`. partition()
    validates them (k - 1 ascending points in 1..L, no batch empty)."""
    cuts = tuple(cuts)

    def rule(counts: Counts, k: int) -> Cuts:
        return cuts

    return rule


def candidate_cuts(total_am: int, n_batches: int) -> Iterator[Cuts]:
    """Every shared contiguous cut `partition` accepts -- none leaves a batch
    empty -- in lexicographic order; () when there is nothing to cut (1
    batch, or one sub-batch per direction)."""
    if n_batches == 1:
        yield ()
        return
    counts = target_counts(total_am)
    for cuts in combinations(range(1, total_am + 1), n_batches // 3 - 1):
        if not empty_batches(counts, cuts):
            yield cuts


def _seed_order(name: str) -> SeedOrder:
    try:
        return SEED_ORDERS[name]
    except KeyError:
        raise ValueError(
            f"seed order {name!r} is not one of recovery.seed_orders(widened=True)"
        ) from None


@cache
def _dags(total_am: int, n_batches: int, cuts: Cuts) -> tuple[BatchDag, ...]:
    """The unsieved MILD DAG of each batch of one cut (partition validates
    the cut). Cached: a DAG is immutable, and the climb sieves one thousands
    of times."""
    batches = partition(total_am, n_batches, cut=fixed_cut(cuts))
    return tuple(batch_dag(total_am, b) for b in batches)


def batch_targets(dag: BatchDag) -> frozenset[R]:
    return frozenset(r for r, _ in dag.targets)


def _batch_name(b: Batch) -> str:
    return f"{b[0]}{b[1]}..{b[2]}"


def check_orders(
    total_am: int, n_batches: int, cuts: Cuts, orders: Sequence[Sequence[R]]
) -> Orders:
    """`orders` as Orders, refused unless it holds one order per batch of
    this cut and each is a permutation of that batch's target set. The
    message names the cell and the batch."""
    dags = _dags(total_am, n_batches, tuple(cuts))
    cell = f"cell {total_am}/{n_batches}"
    if isinstance(orders, str) or len(orders) != len(dags):
        raise ValueError(f"{cell}: {len(orders)} seed orders for "
                         f"{len(dags)} batches")
    out = []
    for k, (d, o) in enumerate(zip(dags, orders, strict=True)):
        order = tuple(tuple(int(c) for c in r) for r in o)
        mine = batch_targets(d)
        where = f"{cell}, batch {k} ({_batch_name(d.batch)})"
        if len(set(order)) != len(order):
            raise ValueError(f"{where}: the seed order repeats a target")
        if set(order) != mine:
            raise ValueError(
                f"{where}: the seed order is not a permutation of the batch's "
                f"{len(mine)} targets ({len(order)} given, "
                f"{len(set(order) - mine)} not in the batch, "
                f"{len(mine - set(order))} missing)")
        out.append(order)
    return tuple(out)


def _sieved(dags: Iterable[BatchDag], orders: Iterable[Sequence[R]]
            ) -> tuple[BatchDag, ...]:
    return tuple(sieve(d, seed_order=o) for d, o in zip(dags, orders, strict=True))


def rule_order(total_am: int, dag: BatchDag, name: str) -> Order:
    """The named rule's order of `hermite_targets(L)`, restricted to the
    batch's targets: what `sieve(dag, seed_order=rule(L))` seeds with."""
    mine = batch_targets(dag)
    return tuple(r for r in _seed_order(name)(total_am) if r in mine)


def searched_plan(
    total_am: int, n_batches: int, cuts: Cuts,
    seed: str | Sequence[Sequence[R]] = LITERAL_SEED,
) -> tuple[BatchDag, ...]:
    """The full plan for one cut and seed: partition, MILD DAG per batch,
    sieve. `seed` is the name of one of `seed_orders(widened=True)`, every
    batch seeded in that rule's order, or (#172) one explicit permutation
    of each batch's targets, in partition order -- refused, naming the
    batch, unless it is one (`check_orders`)."""
    cuts = tuple(cuts)
    dags = _dags(total_am, n_batches, cuts)
    if isinstance(seed, str):
        return _sieved(dags, (rule_order(total_am, d, seed) for d in dags))
    return _sieved(dags, check_orders(total_am, n_batches, cuts, seed))


Plan = tuple[Cuts, str, PlanCost]  # a rule plan: cut points, rule name, cost


def feasible(c: PlanCost, minimax: int) -> bool:
    return c.max_targets <= minimax + MINIMAX_TOLERANCE


def reference(points: Iterator[Point] | tuple[Point, ...] | list[Point]) -> Point:
    """(intermediates_min, flops_min) over a frontier's points: the rule's
    normalisers. Over a frontier they equal the feasible plans' minima."""
    pts = list(points)
    if not pts:
        raise ValueError("an empty frontier has no reference point")
    return (min(p[0] for p in pts), min(p[1] for p in pts))


def score(p: Point, ref: Point) -> Fraction:
    """#171's selection score, in exact rationals:
    intermediates / intermediates_min + flops / flops_min."""
    return Fraction(p[0], ref[0]) + Fraction(p[1], ref[1])


def select(points: tuple[Point, ...] | list[Point]) -> int:
    """The index of the frontier point #171's rule selects: the smallest
    `score`, equal scores to the fewer intermediates."""
    ref = reference(points)
    return min(range(len(points)),
               key=lambda k: (score(points[k], ref), points[k][0]))


def all_plans(total_am: int, n_batches: int) -> Iterator[Plan]:
    """The rule space (#168), costed, in tie-break order: `candidate_cuts`
    outer, `seed_orders(widened=True)` inner."""
    for cuts in candidate_cuts(total_am, n_batches):
        for name in SEED_ORDERS:
            yield cuts, name, cost(searched_plan(total_am, n_batches, cuts, name))


def max_targets(total_am: int, n_batches: int, cuts: Cuts) -> int:
    """A cut's largest batch: Table 3's quantity, which no seed order moves
    (the sieve never deletes a target)."""
    return max(len(d.targets) for d in _dags(total_am, n_batches, tuple(cuts)))


def feasible_cuts(total_am: int, n_batches: int) -> tuple[Cuts, ...]:
    """The cuts within MINIMAX_TOLERANCE of the cell's minimax, in
    `candidate_cuts` order."""
    sizes = {c: max_targets(total_am, n_batches, c)
             for c in candidate_cuts(total_am, n_batches)}
    minimax = min(sizes.values())
    return tuple(c for c, n in sizes.items() if n <= minimax + MINIMAX_TOLERANCE)


# -- the climb ----------------------------------------------------------------


def _batch_point(dag: BatchDag, order: Sequence[R]) -> Point:
    return point(cost((sieve(dag, seed_order=order),)))


def batch_starts(total_am: int, dag: BatchDag) -> dict[str, Order]:
    """The 58 rules restricted to one batch, one start per DISTINCT order
    (the first rule, in `SEED_ORDERS` order, to give it names it)."""
    out: dict[str, Order] = {}
    seen: set[Order] = set()
    for name in SEED_ORDERS:
        o = rule_order(total_am, dag, name)
        if o not in seen:
            seen.add(o)
            out[name] = o
    return out


@dataclass(frozen=True)
class Unit:
    """One climb: one start of one batch of one cut of one cell, under every
    weight of the sweep, against the batch's reference point. Independent of
    every other unit, so units run in any order, in any process."""

    total_am: int
    n_batches: int
    cuts: Cuts
    batch: int  # index in partition order
    start: str
    ref: Point
    budget: int
    seed: int


@dataclass(frozen=True)
class UnitResult:
    unit: Unit
    frontier: tuple[Evaluated, ...]
    runs: int
    exhausted: int  # runs stopped by the budget, not at a local optimum
    examined: int  # neighbours examined over the runs, memo hits included


def run_unit(u: Unit) -> UnitResult:
    dag = _dags(u.total_am, u.n_batches, u.cuts)[u.batch]
    order = batch_starts(u.total_am, dag)[u.start]
    res = climb({u.start: order}, lambda o: _batch_point(dag, o),
                sweep=CLIMB_SWEEP, budget=u.budget, seed=u.seed, ref=u.ref)
    return UnitResult(u, res.frontier, len(res.runs),
                      sum(r.exhausted for r in res.runs),
                      sum(r.examined for r in res.runs))


def climb_starts(total_am: int, dag: BatchDag) -> tuple[tuple[str, ...], Point]:
    """The starts a batch is climbed from, and its reference point. Every
    distinct rule order (`batch_starts`) is evaluated; the reference is the
    best intermediates and the best FLOPs over them, as one `climb` over all
    of them would take it; and the climbed starts are the ones whose point
    no other start's weakly dominates -- the batch's 58-rule frontier, one
    start per point, the first in `SEED_ORDERS` order."""
    starts = batch_starts(total_am, dag)
    pts = {name: _batch_point(dag, o) for name, o in starts.items()}
    ref = (min(p[0] for p in pts.values()), min(p[1] for p in pts.values()))
    return frontier(list(starts), key=lambda name: pts[name]), ref


def units(total_am: int, n_batches: int, *, budget: int = CLIMB_BUDGET,
          seed: int = CLIMB_SEED) -> tuple[Unit, ...]:
    """Every climb the cell's search runs: feasible cuts, then batches, then
    `climb_starts`, in that order."""
    out = []
    for cuts in feasible_cuts(total_am, n_batches):
        for b, dag in enumerate(_dags(total_am, n_batches, cuts)):
            names, ref = climb_starts(total_am, dag)
            out += [Unit(total_am, n_batches, cuts, b, name, ref, budget, seed)
                    for name in names]
    return tuple(out)


@dataclass(frozen=True)
class SearchedPlan:
    """One plan of the climbed space: a cut, one seed order per batch, the
    start each order was climbed from, and its counts."""

    cuts: Cuts
    starts: tuple[str, ...]
    orders: Orders
    cost: PlanCost

    @property
    def point(self) -> Point:
        return point(self.cost)


@dataclass(frozen=True)
class SearchResult:
    total_am: int
    n_batches: int
    frontier: tuple[SearchedPlan, ...]  # the cell's Pareto set, see `assemble`
    selected: int  # the index of the chosen plan in `frontier`
    runs: int
    exhausted: int
    examined: int

    @property
    def chosen(self) -> SearchedPlan:
        return self.frontier[self.selected]

    @property
    def cuts(self) -> Cuts:
        return self.chosen.cuts

    @property
    def orders(self) -> Orders:
        return self.chosen.orders

    @property
    def cost(self) -> PlanCost:
        return self.chosen.cost


def assemble(total_am: int, n_batches: int,
             results: Iterable[UnitResult]) -> SearchResult:
    """The cell from its units' results: per batch, the frontier of every
    order its starts' climbs evaluated (starts in order, so a point keeps
    the first start's order); per cut, `product` of its batches' frontiers;
    over the feasible cuts in order, the frontier; and #171's `select`."""
    by_batch: dict[tuple[Cuts, int], list[Evaluated]] = {}
    rs = list(results)
    for r in rs:
        by_batch.setdefault((r.unit.cuts, r.unit.batch), []).extend(r.frontier)
    plans: list[SearchedPlan] = []
    for cuts in feasible_cuts(total_am, n_batches):
        dags = _dags(total_am, n_batches, cuts)
        n = max(len(d.targets) for d in dags)
        fronts = [frontier(by_batch[cuts, b], key=lambda e: e.point)
                  for b in range(len(dags))]
        for c in product(fronts, key=lambda e: e.point):
            plans.append(SearchedPlan(
                cuts, tuple(e.start for e in c.parts),
                tuple(e.order for e in c.parts),
                PlanCost(flops=c.point[1], max_intermediates=c.point[0],
                         max_targets=n)))
    front = frontier(plans, key=lambda p: p.point)
    k = select([p.point for p in front])
    return SearchResult(total_am, n_batches, front, k, sum(r.runs for r in rs),
                        sum(r.exhausted for r in rs), sum(r.examined for r in rs))


def search(total_am: int, n_batches: int, *, budget: int = CLIMB_BUDGET,
           seed: int = CLIMB_SEED,
           map_: Callable[..., Iterable[UnitResult]] = map) -> SearchResult:
    """One cell: run its `units` (through `map_`, e.g. a process pool's) and
    `assemble` them."""
    return assemble(total_am, n_batches,
                    map_(run_unit, units(total_am, n_batches, budget=budget,
                                         seed=seed)))


def build(cells: Sequence[tuple[int, int]], *, workers: int = 1,
          budget: int = CLIMB_BUDGET, seed: int = CLIMB_SEED
          ) -> dict[tuple[int, int], SearchResult]:
    """Every cell's search, its units spread over `workers` processes
    (largest batches first, for balance). The result does not depend on
    `workers`: each unit is a pure function of its fields."""
    todo = [u for c in cells for u in units(*c, budget=budget, seed=seed)]
    if workers <= 1:
        done = list(map(run_unit, todo))
    else:
        order = sorted(range(len(todo)), key=lambda i: -len(
            _dags(todo[i].total_am, todo[i].n_batches, todo[i].cuts)
            [todo[i].batch].targets))
        # spawn, not fork: the caller may be threaded (a test runner), and
        # a unit needs nothing of the parent but its fields.
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            got = list(pool.map(run_unit, [todo[i] for i in order], chunksize=1))
        done = [None] * len(todo)
        for i, r in zip(order, got, strict=True):
            done[i] = r
    out: dict[tuple[int, int], SearchResult] = {}
    for c in cells:
        out[c] = assemble(*c, (r for r in done
                               if (r.unit.total_am, r.unit.n_batches) == c))
    return out


# -- the record ---------------------------------------------------------------

Choice = tuple[Cuts, Orders]  # (cut points, one seed order per batch)


@dataclass(frozen=True)
class FrontierPoint:
    """One recorded frontier point: its plan and the counts it replays to."""

    cuts: Cuts
    starts: tuple[str, ...]
    orders: Orders
    max_intermediates: int
    flops: int
    max_targets: int

    @property
    def point(self) -> Point:
        return (self.max_intermediates, self.flops)

    @property
    def choice(self) -> Choice:
        return (self.cuts, self.orders)


def _compact(x) -> str:
    return json.dumps(x, separators=(",", ":"))


def _point_json(p: SearchedPlan) -> str:
    head = json.dumps({"cut": list(p.cuts), "start": list(p.starts),
                       "max_intermediates": p.cost.max_intermediates,
                       "flops": p.cost.flops,
                       "max_targets": p.cost.max_targets})
    orders = ",\n".join(f"        {_compact([list(r) for r in o])}"
                        for o in p.orders)
    return f'      {head[:-1]}, "orders": [\n{orders}\n      ]}}'


def space_record(budget: int = CLIMB_BUDGET, seed: int = CLIMB_SEED) -> dict:
    """The space the build step searched, as the record states it."""
    return {
        "cuts": ("every shared contiguous cut batching.partition accepts "
                 "whose largest batch is within minimax_tolerance of the "
                 "cell's minimax"),
        "starts": ("per batch, plan.recovery.seed_orders(widened=True)'s 58 "
                   "rules restricted to the batch's targets, one per distinct "
                   "order, named by the first rule that gives it; all are "
                   "evaluated, and those on the batch's frontier of them "
                   "(no other start's point weakly dominates theirs, one per "
                   "point) are climbed"),
        "neighbourhood": list(NEIGHBOURHOOD),
        "sweep": [str(w) for w in CLIMB_SWEEP],
        "budget": budget,
        "seed": seed,
        "climb": ("plan.order_search.climb, one start at a time against the "
                  "batch's reference point (the best intermediates and the "
                  "best flops over its starts): first-improvement, strictly "
                  "better steps only, each run ending at a local optimum or "
                  "after 'budget' neighbours examined"),
        "combine": ("per batch the frontier of every order its climbs "
                    "evaluated; per cut plan.order_search.product of its "
                    "batches' frontiers (max intermediates, summed flops); "
                    "over the feasible cuts the frontier; then 'select'"),
    }


def dump_cuts(results: Mapping[tuple[int, int], SearchResult], *,
              budget: int = CLIMB_BUDGET, seed: int = CLIMB_SEED) -> str:
    """searched_cuts.json: per "L/n_batches", the chosen cut, the start of
    each batch's order, the index of the chosen plan in the cell's frontier,
    the climb's tally, and the frontier -- each point's cut, starts, counts
    and one explicit seed order per batch, one batch per line; sorted by
    (L, nb)."""
    doc = {
        "schema": CUTS_SCHEMA,
        "about": (
            "The plan plan/search.py chose per (L, n_batches) for the searched "
            "board (#172): the shared cut points c_1 < ... < c_{k-1} that "
            "batching.partition takes, k = n_batches // 3, and one explicit "
            "sieve seed order per batch, in partition order -- a permutation "
            "of that batch's targets [rx, ry, rz], climbed from one of "
            "plan.recovery.seed_orders(widened=True) (named in 'start'). The "
            "chosen plan is frontier[selected]; replay any point with "
            "plan.search.searched_plan(L, n_batches, cut, orders); do not "
            "re-search. Each cell records its frontier -- the feasible plans' "
            "Pareto set over (max_intermediates, flops), ascending in "
            "max_intermediates -- so a reader with another exchange rate "
            "between the two can re-select without re-searching (#171). "
            "'space' is what was searched; the build step is "
            "`python -m felizjueves.plan search`."
        ),
        "objective": {
            "feasible": "max_targets <= minimax + minimax_tolerance",
            "minimax_tolerance": MINIMAX_TOLERANCE,
            "frontier": (
                "the feasible plans whose (max_intermediates, flops) no other "
                "feasible plan's weakly dominates, one plan per distinct point"
            ),
            "select": (
                "minimise max_intermediates / intermediates_min + "
                "flops / flops_min over the frontier (the minima are the "
                "frontier's), in exact rationals; equal scores to the fewer "
                "max_intermediates"
            ),
            "rule": SELECTION_RULE,
            "ties": (
                "one plan per point: the first met -- within a batch, starts "
                "in seed_orders(widened=True) order and each climb's "
                "evaluation order; across batches, order_search.product's "
                "nested order; across cuts, lowest cut points first"
            ),
        },
        "space": space_record(budget, seed),
    }
    blocks = []
    for L, nb in sorted(results):
        r = results[L, nb]
        head = json.dumps({"cut": list(r.cuts), "start": list(r.chosen.starts),
                           "selected": r.selected,
                           "climb": {"runs": r.runs, "exhausted": r.exhausted,
                                     "examined": r.examined}})
        pts = ",\n".join(_point_json(p) for p in r.frontier)
        blocks.append(f'    "{L}/{nb}": {head[:-1]}, "frontier": [\n{pts}\n    ]}}')
    head = json.dumps(doc, indent=2)
    return head[:-2] + ',\n  "cuts": {\n' + ",\n".join(blocks) + "\n  }\n}\n"


def _read(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != CUTS_SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, not {CUTS_SCHEMA}")
    return doc


def _frontier_of(key: str, cell: Mapping) -> tuple[FrontierPoint, ...]:
    """A cell's recorded frontier, every point's orders checked against its
    cut's batches, and its 'selected' point checked to be the cell's own cut
    and starts."""
    L, nb = (int(x) for x in key.split("/"))
    pts = []
    for p in cell["frontier"]:
        cuts = tuple(p["cut"])
        starts = tuple(p["start"])
        for s in starts:
            _seed_order(s)  # a start names one of the 58 rules
        orders = check_orders(L, nb, cuts, p["orders"])
        if len(starts) != len(orders):
            raise ValueError(f"cell {key}: {len(starts)} starts for "
                             f"{len(orders)} batches")
        pts.append(FrontierPoint(cuts, starts, orders, p["max_intermediates"],
                                 p["flops"], p["max_targets"]))
    k = cell["selected"]
    if not 0 <= k < len(pts) or (pts[k].cuts, pts[k].starts) != (
        tuple(cell["cut"]), tuple(cell["start"])
    ):
        raise ValueError(f"cell {key}: 'selected' {k} is not the recorded "
                         f"cut and starts' index in its frontier")
    return tuple(pts)


def load_frontiers(
    path: Path,
) -> dict[tuple[int, int], tuple[tuple[FrontierPoint, ...], int]]:
    """Per cell, the recorded frontier and the index of the selected point.
    Every recorded seed order is checked to be a permutation of its batch's
    targets (`check_orders`); a record that is not is refused."""
    doc = _read(path)
    out = {}
    for key, cell in doc["cuts"].items():
        L, nb = (int(x) for x in key.split("/"))
        out[L, nb] = (_frontier_of(key, cell), cell["selected"])
    return out


def load_cuts(path: Path) -> dict[tuple[int, int], Choice]:
    """(cut, one seed order per batch) per cell: its frontier's selected
    point, after `load_frontiers`' checks."""
    return {c: front[k].choice for c, (front, k) in load_frontiers(path).items()}
