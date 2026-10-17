"""The plan layer's output does not depend on the hash seed (#169).

WHY OUT OF PROCESS. CPython reads PYTHONHASHSEED once, at interpreter start;
setting `os.environ` inside a test changes nothing already running. So each
seed is a fresh interpreter: this file, run as a script (`python
test/test_plan_determinism.py <probe>`), prints one probe's output as JSON,
and the test compares the bytes across seeds. The child gets a minimal
environment -- PYTHONHASHSEED, PYTHONPATH=src and nothing else -- so no
variable of the parent's run reaches it.

WHAT A HASH SEED CAN AND CANNOT REACH, said plainly. The seed salts the hash
of `str` and `bytes` (and so of any tuple holding one): a `Batch` is
`("x", a, b)`, and a seed-order or start name is a string, so a set or dict
iterated over either moves with it. It does NOT salt `int`: a `Node`,
`((rx, ry, rz), m)`, hashes the same in every interpreter, so a set of nodes
iterates in one order every run -- order-dependent, perhaps, but not
irreproducible, and this test cannot see it. What it pins is the hazard #169
names: an output that changes from one run to the next.

WHAT IS COMPARED. Not `PlanCost` alone: FLOPs are summed and the other two
are maxima, so a plan whose batches come back in another order, or whose sieve
picks another direction where two cost the same, can keep its totals. Each
probe prints, per cell, the `PlanCost` AND the plan itself in canonical form
(per batch, in the plan's order: its batch key and every node's parents,
sorted by node) -- a different DAG fails even where the counts agree.

THE PROBES, one per path through plan/ that a later issue makes load-bearing:

    literal      pipeline.plan(Spec) -- the shipped plan
    searched     search.searched_plan from the RECORD (searched_cuts.json),
                 read, not pinned: the record is re-written by other issues
    seed_orders  every rule of recovery.seed_orders(widened=True), as the
                 order it yields -- the widened rules sort on DAG features
    climb        order_search.climb over a small cell, RNG included: its
                 runs are seeded from strings, and a frontier over tuples
    search       search.build, the searched board's build step (#172), on
                 one small cell with a small budget: per-start climbs, the
                 per-batch frontiers, the product, the selection and the
                 record's text, serially and over a process pool

TO EXTEND: write a function
returning JSON-ready data, register it in PROBES, and it is run under every
seed in SEEDS by `test_a_probe_is_the_same_under_every_hash_seed`. Keep a
probe to a second or two -- it runs once per seed, in a fresh interpreter.

THE SIEVE IS IDEMPOTENT. Alg. 3 deletes and re-parents, so a second pass
finding nothing is not obvious; the plan layer applies it once and relies on
the result being a fixed point. Checked for every rule of
`seed_orders(widened=True)` at small L, in node set and in each node's
parents (its reduction direction).

TIERS. Every probe plans a class above FAST_L_MAX (test/tiers.py), so the
probe test is `slow`; the sieve's fixed point is split at the cut.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from tiers import is_fast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CUTS = ROOT / "data" / "scoreboards" / "jengine" / "searched_cuts.json"

# Three or more salts; 0 disables randomisation, the others are arbitrary.
SEEDS = ("0", "1", "169", "4294967295")

# A handful of (L, n_batches) cells: every batch count, low and high L, and
# both the shared-cut and three-cut shapes. Each must be a cell of the record.
CELLS = ((7, 1), (7, 9), (9, 6), (10, 3), (12, 9), (13, 6))


# -- the probes (run in the child) ---------------------------------------------

def _canonical(dags) -> list[Any]:
    """A plan as data: per batch, in plan order, its key and every node's
    parents, sorted by node."""
    return [
        {"batch": list(d.batch),
         "targets": sorted(d.targets),
         "parents": sorted((n, list(ps)) for n, ps in d.parents.items())}
        for d in dags
    ]


def _with_cost(dags) -> dict[str, Any]:
    from felizjueves.plan.cost import cost

    c = cost(dags)
    return {"cost": [c.flops, c.max_intermediates, c.max_targets],
            "plan": _canonical(dags)}


def probe_literal() -> dict[str, Any]:
    from felizjueves.plan.pipeline import plan
    from felizjueves.spec import Operator, Spec, Target

    out = {}
    for L, nb in CELLS:
        am = (L - 3 * (L // 4), L // 4, L // 4, L // 4)
        p = plan(Spec(Operator.COULOMB, 0, am, Target.J, n_batches=nb))
        out[f"{L}/{nb}"] = _with_cost(p.dags)
    return out


def probe_searched() -> dict[str, Any]:
    from felizjueves.plan.search import load_cuts, searched_plan

    record = load_cuts(CUTS)
    out = {}
    for L, nb in CELLS:
        cut, orders = record[L, nb]
        out[f"{L}/{nb}"] = {"choice": [list(cut), [list(o) for o in orders]],
                            **_with_cost(searched_plan(L, nb, cut, orders))}
    return out


def probe_search() -> dict[str, Any]:
    """The searched board's build step (#172) on one small cell, its RNG
    and its process split included: (7, 3), every start of every batch
    climbed with a budget of 4, once serially and once over two processes
    -- which must agree with each other as well as across hash seeds."""
    from felizjueves.plan.search import build, dump_cuts

    serial = dump_cuts(build([(7, 3)], workers=1, budget=4), budget=4)
    pooled = dump_cuts(build([(7, 3)], workers=2, budget=4), budget=4)
    assert serial == pooled, "the record depends on the worker count"
    return json.loads(serial)


def probe_seed_orders() -> dict[str, Any]:
    from felizjueves.plan.recovery import seed_orders

    return {name: {L: list(rule(L)) for L in (4, 7)}
            for name, rule in seed_orders(widened=True)}


def probe_climb() -> dict[str, Any]:
    """A small budgeted climb over one batch's seed order: (8, 3)'s first
    batch, started from four rules -- a cell where some runs take a step."""
    from felizjueves.plan.batching import partition
    from felizjueves.plan.cost import cost
    from felizjueves.plan.dag import batch_dag
    from felizjueves.plan.order_search import climb, point
    from felizjueves.plan.recovery import seed_orders
    from felizjueves.plan.sieve import sieve

    L, nb = 8, 3
    dag = batch_dag(L, partition(L, nb)[0])
    mine = {(r, 0) for r, _ in dag.targets}
    rules = dict(seed_orders())
    starts = {name: [r for r in rules[name](L) if (r, 0) in mine]
              for name in list(rules)[:4]}

    def evaluate(order) -> tuple[int, int]:
        return point(cost((sieve(dag, seed_order=order),)))

    res = climb(starts, evaluate, sweep=(Fraction(1), Fraction(0)), budget=40)
    return {"evaluated": res.evaluated,
            "frontier": [[list(e.order), list(e.point), e.start]
                         for e in res.frontier],
            "runs": [[r.start, str(r.weight), list(r.final.point), r.steps,
                      r.examined] for r in res.runs]}


PROBES: dict[str, Callable[[], Any]] = {
    "literal": probe_literal,
    "searched": probe_searched,
    "seed_orders": probe_seed_orders,
    "climb": probe_climb,
    "search": probe_search,
}


# -- the harness (run in the parent) -------------------------------------------

def run_probe(name: str, hash_seed: str) -> str:
    """`name`'s JSON output from a fresh interpreter with PYTHONHASHSEED set
    and nothing else of this process's environment."""
    env = {"PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(SRC)}
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), name],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, (
        f"probe {name} under PYTHONHASHSEED={hash_seed} failed:\n{done.stderr}")
    return done.stdout


@pytest.mark.slow  # every probe plans L > FAST_L_MAX (tiers.py)
@pytest.mark.parametrize("name", list(PROBES))
def test_a_probe_is_the_same_under_every_hash_seed(name: str) -> None:
    outputs = {s: run_probe(name, s) for s in SEEDS}
    first = outputs[SEEDS[0]]
    assert first.strip(), f"probe {name} printed nothing"
    # Compared as bytes, reported by key: the outputs are large, and a
    # string diff of them would cost more than the probe.
    differ = {
        s: sorted(k for k in json.loads(first)
                  if json.loads(out).get(k) != json.loads(first)[k])
        for s, out in outputs.items() if out != first
    }
    assert not differ, (
        f"probe {name} differs from its PYTHONHASHSEED={SEEDS[0]} output "
        f"(hash seed -> differing keys): {differ}")


def test_the_child_really_runs_under_the_seed_it_is_given() -> None:
    """The harness's own witness: a str hash moves with the seed in the
    child, so the seed reached it and a differing output would be seen."""
    code = "print(hash('felizjueves'))"
    hashes = {
        subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env={"PYTHONHASHSEED": s},
                       check=True).stdout
        for s in SEEDS[1:]
    }
    assert len(hashes) == len(SEEDS) - 1


def test_the_cells_are_in_the_record() -> None:
    # From the JSON; that the record loads (load_cuts builds every cell's
    # batches) is the slow twin.
    cells = {tuple(int(x) for x in k.split("/"))
             for k in json.loads(CUTS.read_text())["cuts"]}
    assert set(CELLS) <= cells


@pytest.mark.slow  # load_cuts builds every cell's batches: L > FAST_L_MAX
def test_the_cells_are_in_the_loaded_record() -> None:
    from felizjueves.plan.search import load_cuts

    assert set(CELLS) <= set(load_cuts(CUTS))


# -- the sieve is a fixed point ------------------------------------------------

# Small L, every batch count `partition` accepts there by default.
SIEVE_CELLS = ((4, 1), (4, 3), (4, 6), (4, 9), (5, 1), (5, 6), (6, 1), (6, 9))


@pytest.mark.parametrize(("L", "nb"), [
    pytest.param(L, nb, marks=() if is_fast(L) else pytest.mark.slow)
    for L, nb in SIEVE_CELLS
])
def test_the_sieve_is_idempotent_under_every_seed_order(L: int, nb: int) -> None:
    from felizjueves.plan.batching import partition
    from felizjueves.plan.dag import batch_dag
    from felizjueves.plan.recovery import seed_orders
    from felizjueves.plan.sieve import sieve

    dags = [batch_dag(L, b) for b in partition(L, nb)]
    for name, rule in seed_orders(widened=True):
        order = rule(L)
        for d in dags:
            once = sieve(d, seed_order=order)
            twice = sieve(once, seed_order=order)
            assert twice.nodes() == once.nodes(), (name, L, nb, d.batch)
            assert dict(twice.parents) == dict(once.parents), (
                name, L, nb, d.batch)


if __name__ == "__main__":
    json.dump(PROBES[sys.argv[1]](), sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
