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
