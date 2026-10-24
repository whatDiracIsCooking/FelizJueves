"""The HGP vertical recurrence as a DAG over `(e, f, m)` (`[Fock]` eqs 25-26).

A node `(e, f, m)` is the primitive integral `[e, 0 | f, 0]^(m)`: `e` and `f`
the bra and ket Cartesian angular-momentum vectors, `m` the Boys order. The
recurrence, in build form (eq 25, and its ket twin eq 26):

    [e+1_d, 0 | f, 0]^(m) = PA_d [e,0|f,0]^(m) + WP_d [e,0|f,0]^(m+1)
                          + e_d/(2p)     ([e-1_d,0|f,0]^(m) - a/p [e-1_d,0|f,0]^(m+1))
                          + f_d/(2(p+q))  [e,0|f-1_d,0]^(m+1)

    [e, 0 | f+1_d, 0]^(m) = QC_d [e,0|f,0]^(m) + WQ_d [e,0|f,0]^(m+1)
                          + f_d/(2q)     ([e,0|f-1_d,0]^(m)
                                          - alpha/q [e,0|f-1_d,0]^(m+1))
                          + e_d/(2(p+q))  [e-1_d,0|f,0]^(m+1)

each lower term absent when its index is zero. `[0,0|0,0]^(m)` is Boys
output: a leaf, never reduced.

NOT GRADED BY `m` -- the structural fact this module records. MMD's `(r, m)`
graph (`plan/dag.py`) has every edge raise `m` by exactly one, which is what
makes its topological order free, its sieve (`plan/sieve.py`) well defined
and its recycling analysis exact. Here the leading `PA`/`QC` term (and the
`e_d/(2p)`, `f_d/(2q)` one) stays at the SAME `m`, so a node's order cannot
be read off its Boys index and the build order has to be constructed. That
is why the plan layer cannot absorb this DAG, and why `hgp/` does not import
it (test/test_layering.py, rule "hgp").

THE BUILD ORDER (`VrrDag.order`): bra first, with the ket held at zero --
every `[e,0|0,0]^(m)` by eq 25 with `f = 0`, so its `f_d/(2(p+q))` term is
absent -- then the ket by eq 26, by `|f|` ascending. The reason is `[Fock]`'s:
the ket recurrence couples to the bra indices through its `e_d/(2(p+q))`
term, which reads `[e-1_d,0|f,0]^(m+1)` -- a LOWER bra at the same ket
order -- so every bra value a ket step can read must exist before the ket
phase starts. Within one `|f|` no ket node reads another, and the bra phase
is ordered by `|e|` ascending. The order is a choice satisfying those
constraints, not the only one; `test/test_hgp_vrr.py` checks it is
topological.

THE TARGETS, and the explicit closure. A class `(la lb | lc ld)` needs, from
the VRR, exactly the boundary set of the horizontal recurrence (eq 28,
`hgp/hrr.py`, #274): every `(e, f)` at `m = 0` with
`la <= |e| <= la + lb` and `lc <= |f| <= lc + ld` (`target_set`). The DAG is
the closure of those targets under the reduction rule (`reduce`).

DEAD-NODE PRUNING, ON BY DEFAULT. `build(cls, prune=False)` returns the
forward build -- every `(e, f, m)` with `|e| <= la+lb`, `|f| <= lc+ld`,
`m <= la+lb+lc+ld - |e| - |f|`, which is closed under `reduce` -- and the
default keeps only the nodes some target reaches. In the predecessor this
was opt-in because a FLOP ratchet counted the lowered instructions; nothing
downstream of this DAG is ratcheted here, so the pruned DAG is the default.
At `(dd|dd)` the forward build has 35 x 35 = 1,225 nodes at `m = 0`
(`|e|, |f|` in 0..4) and the pruned DAG 1,089 (31 x 35 + 4): the targets
fix `|e| >= 2` on every ket-phase node at `m = 0`, because only the
`e_d/(2(p+q))` term lowers `e` there, and it raises `m`.

THE DIRECTION. Each non-leaf node is reduced along ONE axis `d`: the first
(in x, y, z order) with a nonzero component of `f` if `f != 0` (eq 26),
else of `e` (eq 25). That is a choice, and the pinned counts in the test
depend on it; changing it is a count diff, not a refactor.

THE LADDER (`CLASSES`). Milestone 12's 21 classes: every AO-order quartet
`(la lb | lc ld)` with each shell `l <= 2`, canonicalised `la <= lb`,
`lc <= ld` and bra pair `<=` ket pair lexicographically -- the 6 pairs
`(ss) (sp) (sd) (pp) (pd) (dd)` taken as unordered pairs of pairs,
6 x 7 / 2 = 21, topping out at `(dd|dd)`. The canonicalisation is
`a <= b` here (the predecessor's is `a >= b`; the join is milestone 12's).

Pure Python: no card, no compiler, no import from `plan/` or `reference/`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

Cart = tuple[int, int, int]
Node = tuple[Cart, Cart, int]  # (e, f, m): [e,0|f,0]^(m)
AmClass = tuple[int, int, int, int]  # (la, lb, lc, ld), AO order

ZERO: Cart = (0, 0, 0)

# The coefficient names, as `[Fock]` eqs 25-26 spell them. A product of two
# names is one coefficient (e.g. `inv2p*a_over_p`).
BRA = ("PA", "WP", "inv2p", "inv2p*a_over_p")
KET = ("QC", "WQ", "inv2q", "inv2q*alpha_over_q", "inv2pq")

_PAIRS = tuple((a, b) for a in range(3) for b in range(a, 3))
CLASSES: tuple[AmClass, ...] = tuple(
    bra + ket for i, bra in enumerate(_PAIRS) for ket in _PAIRS[i:])


def cartesians(l: int) -> tuple[Cart, ...]:  # noqa: E741
    """Every (i, j, k) with i + j + k = l: i descending, then j descending.

    Restated from `ao.shell.cartesians` (the AO layer is a witness, not an
    ancestor); test/test_hgp_vrr.py asserts the two agree."""
    if type(l) is not int or l < 0:
        raise ValueError(f"angular momentum {l!r} is not an int >= 0")
    return tuple(
        (i, j, l - i - j) for i in range(l, -1, -1) for j in range(l - i, -1, -1)
    )


def _check(cls: AmClass) -> AmClass:
    cls = tuple(cls)
    if len(cls) != 4 or any(type(x) is not int or x < 0 for x in cls):
        raise ValueError(f"class {cls!r} is not four ints >= 0 (la, lb, lc, ld)")
    return cls  # type: ignore[return-value]


def _range(lo: int, hi: int) -> tuple[Cart, ...]:
    return tuple(c for n in range(lo, hi + 1) for c in cartesians(n))


def target_set(cls: AmClass) -> frozenset[tuple[Cart, Cart]]:
    """The VRR's targets for `(la lb | lc ld)`: every `(e, f)`, at `m = 0`, with
    `la <= |e| <= la + lb` and `lc <= |f| <= lc + ld`.

    This is the boundary set the HRR (eq 28, #274) reads; #274 asserts its
    own boundary set equal to this one for every class in `CLASSES`."""
    la, lb, lc, ld = _check(cls)
    return frozenset((e, f) for e in _range(la, la + lb)
                     for f in _range(lc, lc + ld))


def targets(cls: AmClass) -> tuple[Node, ...]:
    """`target_set` as nodes at `m = 0`, in a fixed order (`e` major)."""
    la, lb, lc, ld = _check(cls)
    return tuple((e, f, 0) for e in _range(la, la + lb)
                 for f in _range(lc, lc + ld))


def _minus(v: Cart, d: int, k: int = 1) -> Cart:
    return tuple(c - k * (i == d) for i, c in enumerate(v))  # type: ignore[return-value]


@dataclass(frozen=True)
class Term:
    """One term of a node's recurrence: `mult * coef * child`."""

    coef: str
    mult: int
    child: Node


def is_leaf(node: Node) -> bool:
    """`[0,0|0,0]^(m)`: Boys-function output, never reduced."""
    return node[0] == ZERO and node[1] == ZERO


def reduce(node: Node) -> tuple[Term, ...]:
    """The terms that compute `node`: eq 26 when `f != 0`, else eq 25.

    Reduction form: the node is `[e',0|f'+1_d,0]` (or `[e'+1_d,...]`) and the
    terms read the lowered `f'` (`e'`). A leaf has no terms."""
    e, f, m = node
    if f != ZERO:
        d = next(i for i, c in enumerate(f) if c)
        g = _minus(f, d)
        out = [Term("QC", 1, (e, g, m)), Term("WQ", 1, (e, g, m + 1))]
        if g[d]:
            h = _minus(g, d)
            out += [Term("inv2q", g[d], (e, h, m)),
                    Term("inv2q*alpha_over_q", -g[d], (e, h, m + 1))]
        if e[d]:
            out.append(Term("inv2pq", e[d], (_minus(e, d), g, m + 1)))
        return tuple(out)
    if e != ZERO:
        d = next(i for i, c in enumerate(e) if c)
        g = _minus(e, d)
        out = [Term("PA", 1, (g, f, m)), Term("WP", 1, (g, f, m + 1))]
        if g[d]:
            h = _minus(g, d)
            out += [Term("inv2p", g[d], (h, f, m)),
                    Term("inv2p*a_over_p", -g[d], (h, f, m + 1))]
        return tuple(out)
    return ()


def order_key(node: Node) -> tuple:
    """The build order: bra phase (`f = 0`) by `|e|` ascending, then the ket
    phase by `|f|` ascending; ties by `|e|`, `m` descending, Cartesian order."""
    e, f, m = node
    return (f != ZERO, sum(f), sum(e), -m,
            tuple(-c for c in e), tuple(-c for c in f))


@dataclass(frozen=True, eq=False)
class VrrDag:
    """The VRR of one class: its targets, nodes, edges and build order."""

    cls: AmClass
    pruned: bool
    targets: tuple[Node, ...]
    terms: Mapping[Node, tuple[Term, ...]]  # every node, leaves -> ()
    order: tuple[Node, ...]

    @property
    def nodes(self) -> frozenset[Node]:
        return frozenset(self.terms)

    @property
    def n_nodes(self) -> int:
        return len(self.terms)

    @property
    def n_edges(self) -> int:
        """One edge per term: every term of a node reads a distinct child."""
        return sum(len(t) for t in self.terms.values())

    def edges(self) -> Iterable[tuple[Node, Node]]:
        """(parent, child) pairs, parent computed from child."""
        for n, ts in self.terms.items():
            for t in ts:
                yield n, t.child

    def at_m(self, m: int) -> frozenset[Node]:
        return frozenset(n for n in self.terms if n[2] == m)


def reachable(terms: Mapping[Node, tuple[Term, ...]],
              roots: Iterable[Node]) -> frozenset[Node]:
    """Every node some root reaches through `terms`, roots included."""
    seen: set[Node] = set()
    todo = list(roots)
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        todo.extend(t.child for t in terms[n])
    return frozenset(seen)


def _forward(cls: AmClass) -> dict[Node, tuple[Term, ...]]:
    la, lb, lc, ld = cls
    big_e, big_f = la + lb, lc + ld
    total = big_e + big_f
    return {(e, f, m): reduce((e, f, m))
            for e in _range(0, big_e) for f in _range(0, big_f)
            for m in range(total - sum(e) - sum(f) + 1)}


def build(cls: AmClass, prune: bool = True) -> VrrDag:
    """The VRR DAG of `(la lb | lc ld)`.

    `prune=False` is the forward build (every node with `|e| <= la+lb`,
    `|f| <= lc+ld`, `m <= L - |e| - |f|`); the default keeps only the nodes
    some target reaches."""
    cls = _check(cls)
    terms = _forward(cls)
    tg = targets(cls)
    if prune:
        keep = reachable(terms, tg)
        terms = {n: t for n, t in terms.items() if n in keep}
    order = tuple(sorted(terms, key=order_key))
    return VrrDag(cls, prune, tg, MappingProxyType(terms), order)


# ---------------------------------------------------------------------------
# Grouping the boundary along the build order (#350, the batched VRR)
# ---------------------------------------------------------------------------
#
# ADDITIVE: nothing above reads these, so `build`'s DAGs, their pinned counts
# (test/test_hgp_vrr.py) and hgp/cost.json are unchanged.
#
# THE CUT. `scheduled_targets(dag)` is the targets in the order the DAG's
# `order` produces them; `group_targets(dag, n)` cuts that sequence into `n`
# CONTIGUOUS runs of near-equal length (`round(g |T| / n)`), so each group is
# a stretch of the schedule, and the groups partition the targets.
#
# THE SUB-DAG. `group_dag(dag, group)` is the closure of one group's targets
# under the parent DAG's own terms -- every node some target of the group
# reaches -- as a `VrrDag` whose `targets` are the group and whose `order` is
# the PARENT's order filtered to the closure (a topological order, since a
# filter of one is). Every node keeps its parent's terms, so a node a group
# shares with another is the SAME expression in both: the recompute is exact
# and the kernel's values are the literal ones bit for bit. `order=` takes
# another order of the closure (#357's per-group schedule), checked
# complete and topological. `hgp.recurrence.vrr_body` builds a sub-DAG's
# statements unchanged.
#
# THE LIVE MODEL (`peak_live`). The register demand the group-count rule
# reads (eri/fock/levers/batched_vrr.py) is a STATEMENT-ORDER model: executing
# `order` one node per step, a value is live from the step that defines it
# to the last step that reads it, and a value in `live_out` (read after the
# phase: the targets, by the accumulation) to the end. The peak is the most
# values live at once. It models the emitted text, not a compiler's
# schedule (which may reorder); it is the quantity the rule is stated in,
# and nothing more is claimed for it.


def peak_live(order: Iterable, reads, live_out: Iterable = ()) -> int:
    """Peak simultaneous live values executing `order` (the text above).

    `reads(node)` is the nodes `node` reads; each must precede it in
    `order`. A node in `live_out` stays live to the end; any other is dead
    after its last reader (or at once, if nothing reads it)."""
    order = tuple(order)
    at = {n: i for i, n in enumerate(order)}
    if len(at) != len(order):
        raise ValueError("an order lists a node twice")
    last = dict(at)
    for i, n in enumerate(order):
        for c in reads(n):
            if at.get(c, i) >= i:
                raise ValueError(f"{n} reads {c}, not produced before it")
            last[c] = i
    end = len(order)
    for n in live_out:
        if n not in at:
            raise ValueError(f"live-out {n} is not in the order")
        last[n] = end
    delta = [0] * (end + 2)
    for n, i in at.items():
        delta[i] += 1
        delta[last[n] + 1] -= 1
    cur = best = 0
    for d in delta[:end + 1]:
        cur += d
        best = max(best, cur)
    return best


def vrr_peak(dag: VrrDag) -> int:
    """`peak_live` of `dag` in its own order, its targets live out."""
    return peak_live(dag.order, lambda n: (t.child for t in dag.terms[n]),
                     dag.targets)


def scheduled_targets(dag: VrrDag) -> tuple[Node, ...]:
    """`dag`'s targets in the order its `order` produces them."""
    tg = frozenset(dag.targets)
    return tuple(n for n in dag.order if n in tg)


def group_targets(dag: VrrDag, n: int) -> tuple[tuple[Node, ...], ...]:
    """The scheduled targets cut into `n` contiguous, near-equal runs.

    `1 <= n <= len(dag.targets)`; every run is non-empty and the runs
    partition the targets, in schedule order."""
    seq = scheduled_targets(dag)
    if type(n) is not int or not 1 <= n <= len(seq):
        raise ValueError(f"{dag.cls}: {n!r} groups of {len(seq)} targets; "
                         f"a group count is an int in 1..{len(seq)}")
    cut = [round(g * len(seq) / n) for g in range(n + 1)]
    return tuple(seq[a:b] for a, b in zip(cut, cut[1:], strict=False))


def group_dag(dag: VrrDag, group: Iterable[Node],
              order: Iterable[Node] | None = None) -> VrrDag:
    """The closure of `group` (a subset of `dag.targets`) as a `VrrDag`: the
    parent's terms, `targets = group`, and the parent's order filtered to
    the closure unless `order` (the closure, topologically) is given."""
    group = tuple(group)
    stray = set(group) - set(dag.targets)
    if not group or stray or len(set(group)) != len(group):
        raise ValueError(f"{dag.cls}: group {group!r} is not a non-empty "
                         "set of the DAG's targets")
    keep = reachable(dag.terms, group)
    terms = {n: t for n, t in dag.terms.items() if n in keep}
    if order is None:
        order = tuple(n for n in dag.order if n in keep)
    else:
        order = tuple(order)
        if len(order) != len(keep) or set(order) != keep:
            raise ValueError(f"{dag.cls}: the order is not the group's "
                             "closure")
        seen: set[Node] = set()
        for n in order:
            if any(t.child not in seen for t in terms[n]):
                raise ValueError(f"{dag.cls}: {n} precedes a node it reads")
            seen.add(n)
    return VrrDag(dag.cls, dag.pruned, group, MappingProxyType(terms), order)


def groups(cls: AmClass, n: int) -> tuple[VrrDag, ...]:
    """The pruned DAG of `cls` as `n` schedule-contiguous group sub-DAGs."""
    dag = build(cls)
    return tuple(group_dag(dag, g) for g in group_targets(dag, n))


# ---------------------------------------------------------------------------
# A dedicated schedule per group (#357, the per-group VRR schedule)
# ---------------------------------------------------------------------------
#
# ADDITIVE, like the grouping: nothing above reads these, so `build`'s DAGs,
# their pinned counts and hgp/cost.json are unchanged, and `group_dag`'s
# default (the parent's order filtered) is untouched.
#
# THE QUESTION. `group_dag` orders a group's closure by the PARENT's order
# filtered to it: a priority that reflects what the whole DAG needs, not
# what the group needs. `dedicated_order` schedules the closure FOR ITSELF:
# a greedy list schedule over the closure's own edges, its consumer counts
# taken INSIDE the closure (`consumers`), so a value whose last in-group
# reader has run is dead even where the parent DAG still reads it.
#
# THE LIST SCHEDULE (`list_schedule`). Ready = every unscheduled node whose
# children are all scheduled (initially the leaves). Each step emits the
# ready node of greatest GAIN = the values it frees, where a child is freed
# when this node is its last unscheduled in-closure reader and it is not a
# target (the targets are live out: the accumulation reads them after the
# VRR). Every node emitted adds one live value, so the gain is the net
# change plus one, and ranking by it ranks by the net change. Ties go to a
# secondary key, `tie` (`TIES`), then to the parent order's position, so the
# result is a function of the DAG alone:
#
#     "position"   none: the parent's position decides
#     "closeness"  the larger sum over its children of 1/remaining readers
#                  (prefer consuming values nearest their death)
#     "fan-in"     the more distinct children
#
# THE CHOICE (`dedicated_order`). Every candidate -- the parent's filtered
# order first, then one list schedule per tie in `TIES` order -- is scored by
# `order_peak` (`peak_live`, the targets live out: `vrr_peak`'s model), and
# the first of least peak wins. So the dedicated order is NEVER worse than
# the filtered one in that model, and is the filtered one where no list
# schedule is strictly better (a null result, recorded as one). The
# statements are the same nodes with the same terms in another topological
# order: no value changes and no sum is reassociated (each node is its own
# declaration, its terms in `reduce`'s order).

TIES: tuple[str, ...] = ("position", "closeness", "fan-in")


def consumers(dag: VrrDag) -> Mapping[Node, frozenset[Node]]:
    """Every node's readers INSIDE `dag` (its closure's own edges)."""
    out: dict[Node, set[Node]] = {n: set() for n in dag.terms}
    for n, ts in dag.terms.items():
        for t in ts:
            out[t.child].add(n)
    return MappingProxyType({n: frozenset(r) for n, r in out.items()})


def list_schedule(dag: VrrDag, tie: str = "position") -> tuple[Node, ...]:
    """The greedy live-minimising topological order of `dag` (text above),
    its secondary key `tie` (one of `TIES`)."""
    if tie not in TIES:
        raise ValueError(f"tie {tie!r} is not one of {TIES}")
    pos = {n: i for i, n in enumerate(dag.order)}
    kids = {n: tuple(dict.fromkeys(t.child for t in dag.terms[n]))
            for n in dag.terms}
    readers = consumers(dag)
    left = {n: len(readers[n]) for n in dag.terms}
    need = {n: len(kids[n]) for n in dag.terms}
    live_out = frozenset(dag.targets)
    ready = {n for n in dag.terms if not need[n]}

    def key(n: Node) -> tuple:
        gain = sum(1 for c in kids[n] if left[c] == 1 and c not in live_out)
        if tie == "closeness":
            second = -sum(1.0 / left[c] for c in kids[n])
        elif tie == "fan-in":
            second = -len(kids[n])
        else:
            second = 0
        return (-gain, second, pos[n])

    order: list[Node] = []
    while ready:
        n = min(ready, key=key)
        ready.remove(n)
        order.append(n)
        for c in kids[n]:
            left[c] -= 1
        for p in readers[n]:
            need[p] -= 1
            if not need[p]:
                ready.add(p)
    if len(order) != len(dag.terms):
        raise ValueError(f"{dag.cls}: the DAG has a cycle")
    return tuple(order)


def order_peak(dag: VrrDag, order: Iterable[Node]) -> int:
    """`peak_live` of `dag`'s nodes in `order`, its targets live out."""
    return peak_live(order, lambda n: (t.child for t in dag.terms[n]),
                     dag.targets)


def dedicated_order(dag: VrrDag) -> tuple[Node, ...]:
    """The first of least `order_peak` among `dag.order` and one
    `list_schedule` per tie in `TIES` (text above)."""
    cands = (tuple(dag.order), *(list_schedule(dag, t) for t in TIES))
    peaks = [order_peak(dag, c) for c in cands]
    return cands[peaks.index(min(peaks))]


def dedicated(dag: VrrDag) -> VrrDag:
    """`dag` (a group's closure, or a whole pruned DAG) in its
    `dedicated_order`, checked complete and topological by `group_dag`."""
    return group_dag(dag, dag.targets, order=dedicated_order(dag))
