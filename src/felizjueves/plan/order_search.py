"""A frontier and a neighbourhood search over orders (#170).

The machinery #171 and #172 consume, and nothing more: no board, no data
file, no objective decision. Its one importer is plan/search.py (#172),
which climbs each batch's seed order with it and takes the cell's frontier
with `product` (test_layering.py pins its edges and its importers).

THE POINT. A plan, a batch or an order is scored by the pair
(max_intermediates, FLOPs) -- Tables 2 and 1 -- both minimised. `Point` is
that pair; `point(PlanCost)` reads it off plan/cost.py's record, unchanged.

THE FRONTIER. `frontier(items, key)` is the Pareto set of a collection of
evaluated things (orders, plans, (cut, seed) pairs): every item whose point
no other item's point weakly dominates, ONE item per distinct point (the
first one met, so the caller's order is the tie-break), returned ascending in
intermediates and strictly descending in FLOPs -- plan/optimum.py's `pareto`
convention, which the test checks this against on bare points.

THE PRODUCT. A cell's cost over its batches is `max` intermediates and `sum`
FLOPs (plan/cost.py), and neither is separable: the batch that minimises one
need not minimise the other, so a cell's frontier is not the cell built from
each batch's best point. `product(frontiers, key)` combines one frontier per
batch into the cell's frontier, folding batch by batch and pruning to the
frontier after every fold. The pruning is exact, not a heuristic: `max` and
`sum` are both monotone, so a partial combination that another weakly
dominates stays dominated by it under every completion, and pruning each
batch's input to its own frontier is the same argument once more. The test
checks it against the frontier of the explicit cross product.

THE CLIMB. `climb(starts, evaluate, ...)` is a multi-start first-improvement
hill climb over an order -- a tuple of distinct elements, in practice a
batch's targets as the sieve's seed order (plan/sieve.py, lines 4-5).

- STARTS: named orders, in practice named rules (plan/recovery.py's
  `seed_orders`), all permutations of one element set.
- NEIGHBOURHOOD: every transposition (swap positions i < j) and every
  single-element move (take the element at i, reinsert it at j, |i - j| >= 2;
  a move by one place IS the adjacent transposition and is not counted
  twice): n(n - 1)/2 + (n - 1)(n - 2) neighbours of an order of n.
- SCALARISATIONS: each start is climbed once per weight w of a declared
  sweep, from w = 1 (pure intermediates) to w = 0 (pure FLOPs). An order is
  ranked by (w * i / i_ref + (1 - w) * f / f_ref, i, f), in exact rationals,
  with i_ref and f_ref the best intermediates and the best FLOPs over the
  starts (the two differ by an order of magnitude; unnormalised, every w but
  1 would be a FLOPs rule). At w = 1 that is (i, f) lexicographically, at
  w = 0 (f, i). A step is taken only to a STRICTLY better neighbour; a run
  ends at a local optimum or when it has examined `budget` neighbours.
- THE RESULT is everything evaluated, as a frontier -- not an incumbent.
  Choosing one point of it is a selection rule, and that is #171's decision,
  not this module's.

DETERMINISM, and no global state. Each run draws its neighbour order from its
own `random.Random`, seeded from the string "{seed}|{start name}|{w}", so a
run does not depend on which runs came before it and two calls with the same
arguments return the same result. `evaluate` must be a pure function of the
order: its results are memoised for the call, across runs, and a memo hit
counts against a run's budget exactly as a fresh evaluation does. Each pass
over the neighbourhood is a fresh uniformly random order of it, drawn lazily
(`_shuffled`, #172): a one-batch order at L = 16 has 1.4 million neighbours,
and shuffling them all before every step cost more than the steps.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from felizjueves.plan.cost import PlanCost

Point = tuple[int, int]  # (max_intermediates, FLOPs), both minimised
Order = tuple  # a tuple of distinct, hashable elements

# The declared sweep: pure intermediates to pure FLOPs, in quarters.
SWEEP: tuple[Fraction, ...] = tuple(Fraction(k, 4) for k in (4, 3, 2, 1, 0))
NEIGHBOURHOOD = ("transposition", "move")
DEFAULT_SEED = 170
DEFAULT_BUDGET = 2000


def point(c: PlanCost) -> Point:
    """A PlanCost as the pair this module ranks: (max_intermediates, FLOPs)."""
    return (c.max_intermediates, c.flops)


def dominates(a: Point, b: Point) -> bool:
    """`a` weakly dominates `b`: no worse on either axis."""
    return a[0] <= b[0] and a[1] <= b[1]


def frontier[T](items: Iterable[T], key: Callable[[T], Point]) -> tuple[T, ...]:
    """The items whose point no other item's point weakly dominates, one per
    distinct point (the first met), ascending in intermediates and strictly
    descending in FLOPs."""
    first: dict[Point, T] = {}
    for item in items:
        first.setdefault(tuple(key(item)), item)
    out: list[T] = []
    best_flops: int | None = None
    for p in sorted(first):
        if best_flops is None or p[1] < best_flops:
            out.append(first[p])
            best_flops = p[1]
    return tuple(out)


@dataclass(frozen=True)
class Combination:
    """One point of a cell: one item per batch, in batch order, and the
    cell's point -- max intermediates and summed FLOPs over the parts."""

    point: Point
    parts: tuple


def combine(points: Sequence[Point]) -> Point:
    """A cell's point from its batches' points: max intermediates, sum FLOPs."""
    if not points:
        raise ValueError("a cell has at least one batch")
    return (max(p[0] for p in points), sum(p[1] for p in points))


def product[T](
    frontiers: Sequence[Iterable[T]], key: Callable[[T], Point]
) -> tuple[Combination, ...]:
    """The cell's frontier from one collection of evaluated items per batch:
    the frontier of the cross product under `combine`, computed as a pruned
    fold (exact; see the module docstring). Among combinations with one point
    the first in nested batch order (earlier batches outer) is kept."""
    if not frontiers:
        raise ValueError("a cell has at least one batch")
    acc: tuple[Combination, ...] = (Combination((0, 0), ()),)
    for batch in frontiers:
        pruned = frontier(batch, key)
        if not pruned:
            raise ValueError("a batch has no evaluated item")
        acc = frontier(
            (
                Combination(combine((c.point, tuple(key(item)))), c.parts + (item,))
                for c in acc
                for item in pruned
            ),
            key=lambda c: c.point,
        )
    return acc


# -- the climb ----------------------------------------------------------------


@dataclass(frozen=True)
class Evaluated:
    """An evaluated order, and the start whose run first evaluated it."""

    order: Order
    point: Point
    start: str


@dataclass(frozen=True)
class Run:
    """One climb: a start under one weight, where it stopped, and why."""

    start: str
    weight: Fraction
    final: Evaluated
    steps: int  # improving moves taken
    examined: int  # neighbours examined, memo hits included
    exhausted: bool  # stopped by the budget, not at a local optimum


@dataclass(frozen=True)
class ClimbSpace:
    """What was searched, stated so a record can name it."""

    starts: tuple[str, ...]
    neighbourhood: tuple[str, ...]
    sweep: tuple[Fraction, ...]
    budget: int
    seed: int


@dataclass(frozen=True)
class ClimbResult:
    space: ClimbSpace
    frontier: tuple[Evaluated, ...]
    evaluated: int  # distinct orders evaluated
    runs: tuple[Run, ...]


Move = tuple[str, int, int]


def moves(n: int) -> list[Move]:
    """The neighbourhood of an order of n, as (kind, i, j) descriptors in a
    fixed order: transpositions i < j, then moves |i - j| >= 2."""
    out: list[Move] = [("transposition", i, j)
                       for i in range(n) for j in range(i + 1, n)]
    out += [("move", i, j)
            for i in range(n) for j in range(n) if abs(i - j) >= 2]
    return out


def n_moves(n: int) -> int:
    """len(moves(n)), without building it."""
    return n * (n - 1) // 2 + max(n - 1, 0) * max(n - 2, 0)


def move_at(n: int, k: int) -> Move:
    """moves(n)[k], by arithmetic: at n = 969 (L = 16, one batch) the list
    is 1.4 million moves, and a climb needs only the few it examines."""
    if not 0 <= k < n_moves(n):
        raise IndexError(f"move {k} of {n_moves(n)}")
    t = n * (n - 1) // 2
    if k < t:  # transpositions: row i holds n - 1 - i of them
        i = 0
        while k >= n - 1 - i:
            k -= n - 1 - i
            i += 1
        return ("transposition", i, i + 1 + k)
    k -= t  # moves: row i holds n - 1 - (0 < i) - (i < n - 1) of them
    i = 0
    while k >= (row := n - 1 - (i > 0) - (i < n - 1)):
        k -= row
        i += 1
    low = max(i - 1, 0)  # the j in 0..i-2
    return ("move", i, k if k < low else i + 2 + (k - low))


def _shuffled(rng: random.Random, n: int) -> Iterator[Move]:
    """The moves of an order of n in a uniformly random order, drawn lazily:
    a forward Fisher-Yates shuffle over the indices of `moves(n)`, holding
    only the positions it has swapped."""
    size = n_moves(n)
    swapped: dict[int, int] = {}
    for i in range(size):
        j = i + rng.randrange(size - i)
        here, there = swapped.get(i, i), swapped.get(j, j)
        swapped[j] = here
        yield move_at(n, there)


def apply(order: Order, move: Move) -> Order:
    kind, i, j = move
    o = list(order)
    if kind == "transposition":
        o[i], o[j] = o[j], o[i]
    elif kind == "move":
        o.insert(j, o.pop(i))
    else:
        raise ValueError(f"unknown move {kind!r}")
    return tuple(o)


def neighbours(order: Order) -> Iterator[Order]:
    """Every neighbour of `order`, in `moves` order."""
    for m in moves(len(order)):
        yield apply(order, m)


def _rank(w: Fraction, ref: Point) -> Callable[[Point], tuple]:
    i_ref, f_ref = (max(r, 1) for r in ref)

    def rank(p: Point) -> tuple:
        return (w * Fraction(p[0], i_ref) + (1 - w) * Fraction(p[1], f_ref),
                p[0], p[1])

    return rank


def climb[E](
    starts: Mapping[str, Sequence[E]],
    evaluate: Callable[[Order], Point],
    *,
    sweep: Sequence[Fraction] = SWEEP,
    budget: int = DEFAULT_BUDGET,
    seed: int = DEFAULT_SEED,
    ref: Point | None = None,
) -> ClimbResult:
    """Climb every start under every weight of `sweep` (module docstring) and
    return everything evaluated, as a frontier. Runs go start by start in
    `starts` order, weights in `sweep` order.

    `ref` is the scalarisation's normalising point; by default the best
    intermediates and the best FLOPs over `starts`. A caller that climbs a
    batch's starts one at a time (plan/search.py, #172, to spread them over
    processes) passes the reference of ALL the batch's starts, so each run
    is exactly the run one call over every start would make."""
    if not starts:
        raise ValueError("a climb needs at least one start")
    if budget < 0:
        raise ValueError(f"budget {budget} < 0")
    weights = tuple(Fraction(w) for w in sweep)
    if not weights or any(not 0 <= w <= 1 for w in weights):
        raise ValueError(f"a sweep is a non-empty sequence of weights in "
                         f"[0, 1]: {sweep!r}")
    orders = {name: tuple(o) for name, o in starts.items()}
    elements = None
    for name, o in orders.items():
        if len(set(o)) != len(o):
            raise ValueError(f"start {name!r} repeats an element")
        if elements is None:
            elements = frozenset(o)
        elif frozenset(o) != elements:
            raise ValueError(f"start {name!r} is not a permutation of the "
                             f"first start's elements")

    memo: dict[Order, Evaluated] = {}

    def score(order: Order, start: str) -> Evaluated:
        hit = memo.get(order)
        if hit is None:
            hit = memo[order] = Evaluated(order, tuple(evaluate(order)), start)
        return hit

    first = [score(o, name) for name, o in orders.items()]
    if ref is None:
        ref = (min(e.point[0] for e in first), min(e.point[1] for e in first))
    elif len(ref) != 2 or min(ref) < 0:
        raise ValueError(f"ref {ref!r} is not a point")
    ref = (ref[0], ref[1])
    n = len(next(iter(orders.values())))

    runs: list[Run] = []
    for name, o in orders.items():
        for w in weights:
            rank = _rank(w, ref)
            rng = random.Random(f"{seed}|{name}|{w}")
            here = memo[o]
            steps = examined = 0
            exhausted = False
            improved = True
            while improved:
                improved = False
                for m in _shuffled(rng, n):
                    if examined >= budget:
                        exhausted = True
                        break
                    examined += 1
                    there = score(apply(here.order, m), name)
                    if rank(there.point) < rank(here.point):
                        here, improved = there, True
                        steps += 1
                        break
            runs.append(Run(name, w, here, steps, examined, exhausted))

    return ClimbResult(
        space=ClimbSpace(tuple(orders), NEIGHBOURHOOD, weights, budget, seed),
        frontier=frontier(memo.values(), key=lambda e: e.point),
        evaluated=len(memo),
        runs=tuple(runs),
    )
