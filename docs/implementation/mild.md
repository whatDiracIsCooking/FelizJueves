# The MD search tree: the MILD heuristic and the sieve

**Status: partly implemented.** MILD itself is: `mild_direction` in
`src/felizjueves/plan/mild.py`, with the tie-break table, rules 1, 2, 3 and 5,
and an exhibit of rule 4's absence tested in `test/test_plan_mild.py`. The
sieve is not implemented yet; that section is still a specification, the rule
the generator's pruning pass must follow, written so that a test can be read
against it.

This file covers steps 2 and 3 of the generator pipeline — choosing a reduction
direction for every node, and pruning the resulting graph. The engine around it
(the recurrences, the kernel, the batching) is [`j_engine.md`](j_engine.md).

Citations: `[JEngine, v2, …]` is entry 35 of [`../references.md`](../references.md),
indexed against the ChemRxiv v2 preprint; other works are cited by author and
DOI so they resolve regardless of that file's numbering.

## The problem

A target integral `[r]^(0)` is built by the McMurchie–Davidson recurrence,
which reduces one Cartesian component at a time and raises the auxiliary index
`m`. Each of the three components is a legal choice, so **a solution to the
recurrence is exactly an assignment of one reduction direction to every node** —
every target `[r]^(0)` and every intermediate `[r]^(m)`, `m > 0`. That
assignment induces a DAG; its node count is what sets register pressure, and its
edges set the FLOP count (1, 2 or 3 per node for `r_d` = 1, 2, ≥ 3, per `plan/cost.py` —
see the cost table in [`j_engine.md`](j_engine.md)).

Choosing the assignment that minimises the DAG is a tree-search problem. Exact
solutions are known only for `L ≤ 7` (Johnson, Gill & Pople 1991,
doi:10.1002/qua.560400610); we need `L` up to 16, so the direction comes from a
heuristic and the result is then pruned.

Two things make the search hard, and both matter when reading any cost number:

- **Intermediates are shared.** The cheapest way to build one target depends on
  which intermediates the *other* targets have already put in the graph, so the
  choices are not independent.
- **Fewer nodes is not fewer registers.** Register allocation depends on the
  *traversal order* of the DAG — on live ranges, not on node count. A smaller
  DAG can spill more. `[JEngine, v2, §4.1]` measures exactly this, and the
  relationship is non-monotone; see "What the sieve actually buys" below.

## MILD — reduce along the minimum non-zero component

The rule in one line: **reduce in the direction of the smallest non-zero
component of `r`, breaking ties round-robin.** `[JEngine, v2, Alg. 2]`, after
Ryu, Kim & Lee (entry 45 of [`../references.md`](../references.md)), who
introduced the minimum-component choice; "MILD" (minimum-L direction) is the
JEngine paper's name for it.

"Non-zero" is load-bearing: a zero component cannot be reduced at all, so zeros
are excluded from the minimum rather than winning it.

```python
AXES = ("x", "y", "z")

def mild_direction(r):
    """Reduction direction for a node. `r` is a 3-tuple of ints with sum > 0."""
    masked = [c if c else INF for c in r]      # a 0 component is not reducible
    if masked[0] == masked[1] == masked[2]:
        return "z"                             # three-way tie
    lo = min(masked)
    for i in (1, 2, 0):                        # on a tie, the cyclic successor
        if masked[i] == lo and masked[i - 1] == lo:
            return AXES[i]
    return AXES[masked.index(lo)]
```

`[0]^(m)` is a fundamental and is never reduced, hence the `sum > 0`
precondition — without it the all-`INF` branch would hand back a direction for
a node that has none.

### The tie-break, spelled out

The round-robin is a **cyclic successor**: of the two tied axes, take the one
whose cyclic predecessor is the other.

| tied components | direction |
|---|---|
| `rx = ry < rz` | `y` |
| `ry = rz < rx` | `z` |
| `rx = rz < ry` | `x` |
| `rx = ry = rz` | `z` |

Worth tabulating because it is asymmetric, it is easy to get backwards, and
**the batching pass uses the opposite convention on purpose** (cyclic
predecessor, all-equal → `x`; see `j_engine.md`). Two tie-breaks that differ by
what looks like a typo are a likely source of a silent generator bug, so both
belong in tests.

### Why the minimum, geometrically

Lay the targets out as a stack of ternary plots, one per total angular momentum
— a tetrahedron with the axial integrals `(0,0,j)` and permutations along its
edges. Then MILD is:

1. step toward the nearest **face** of the tetrahedron;
2. once on a face, step toward the nearest **edge**.

On an edge — an axial integral — only one component is non-zero and the
reduction is forced. So MILD drives work toward the faces and edges, where
nodes are shared by many targets, and away from the interior, where they are
not. `[JEngine, v2, §4.1]` lists the three consequences: heavy reuse of face
intermediates, removal of interior ones, and a unit component reduced wherever
possible — which is the cheapest case of the recurrence, one FLOP and one
parent.

## MILD against the exact rules

Johnson, Gill & Pople (doi:10.1002/qua.560400610) derived five rules that an
optimal tree obeys. `[JEngine, v2, §4.1]` claims MILD maintains 1, 2, 3 and 5
and leaves 4 to the sieve. That claim checks out, and *why* it does is the
useful part. Components below are named by magnitude rather than by axis:
`{0 1 j}` means "one component 0, one component 1, one component `j`", in any
axis order.

| Rule | Statement | Under MILD |
|---|---|---|
| 1 | Axial integrals `{0 0 j}` are present in every tree. | Not a choice — one non-zero component, reduction forced. Holds. |
| 2 | Axial children `{0 1 j}`, `j ≥ 2`: reduce the unit component. | Min non-zero is `1`. **Holds.** |
| 3 | `{1 1 j}`, `j ≥ 2`: never reduce the `j`. | Min non-zero is `1`, so a unit component is picked. **Holds** — and rule 3 does not say *which* unit component, so MILD's tie-break is a free choice here. |
| 4 | If the parent and grandparent of a minimum-cost reduction are already in the tree, reduce to them (common parents). | Context-dependent: a property of the graph built so far, while MILD is a pure function of `r` alone. **Cannot hold.** Deferred to the sieve. |
| 5 | `{0 2 j}`, `j ≥ 3`: reduce the `2`. | Min non-zero is `2`. **Holds.** |

Rules 2 and 5 are the pair worth staring at: they point at *different*
magnitudes — the `1` in one case, the `2` in the other — and a single "reduce
the minimum non-zero component" subsumes both. That is the appeal of MILD: one
stateless rule reproduces four hand-derived ones.

Rule 4 is the exception, and the reason is structural, not incidental.
Honouring it requires looking at the graph, so no per-node function can do it.
Hence the second pass.

Cost of the trade, per `[JEngine, v2, §4.1]`, against Johnson et al.'s
near-optimal tree search: **+1–2% FLOPs, −2–6% intermediates.** MILD is not a
cheap approximation of the exact search; on the metric that drives register
pressure it is slightly *better*, and it is `O(1)` per node.

## The sieve — pruning what MILD left behind

MILD is non-optimal in node count for `L ≥ 5`. The sieve, after Gill,
Head-Gordon & Pople 1989 (doi:10.1002/qua.560360831) and given as
`[JEngine, v2, Alg. 3]`, recovers rule 4 by walking the built graph and
deleting intermediates nothing truly needs.

Vocabulary, fixed here for the generator: a node's **parents** are the
intermediates it is formed from; its **children** are the nodes formed from it.
The graph is a DAG, and the recurrence only ever raises `m`, so parents sit at
higher `m` than their children.

```
I      <- every node MILD marked required
queue  <- parents of every target [r]^(0)        # seeded in target order
seen   <- {}

while queue:
    i = queue.popleft()
    if i in seen:
        continue
    seen.add(i)

    # can every child of i be re-expanded in some other direction, using
    # only nodes already in I \ {i}?
    if all(child.can_reexpand_without(i, I) for child in i.children):
        for child in i.children:
            child.reparent()                     # deviate from MILD here
        discard(i, and_any_now_unused_parents_of=i)
    else:
        queue.extend(i.parents)                  # i survives; keep walking up
```

Breadth-first from the targets upward. A node survives only if **some** child
cannot be re-expanded without it; the re-expansion in `can_reexpand_without` is
precisely where a child is permitted to deviate from MILD, and it is what makes
the sieve a rule-4 pass rather than a dead-code sweep.

Two properties to carry into the implementation:

- **The result depends on the seed order** of the targets. `[JEngine, v2, §4.1]`
  reports that this moves FLOPs and node count by at most ~2%, so they do not
  optimise it. We should still **fix** the order, so our output is
  deterministic and golden files are stable.
- **Deleting a node can orphan its parents**, so the delete step sweeps upward
  rather than just dropping `i`.

### What the sieve actually buys

Honest numbers, from `[JEngine, v2, §4.1]`: **~4% fewer spills on average, and
non-monotone** — the `[5|6]` class gets 3% *more* spills, `[8|8]` gets 11%
fewer. That is the "fewer nodes ≠ fewer registers" point from the top of this
file, measured.

The consequence for us is a design decision, not a footnote: **the sieve has to
be switchable.** The generator should emit a class with it and without it, and
the kernel tier should be able to compare — because for a given class on a
given architecture either may win, and neither of our two targets is the A100
those published numbers were measured on.
