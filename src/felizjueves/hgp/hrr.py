"""The HGP horizontal recurrence as a DAG over `(a, b, c, d)` (`[Fock]` eq 28).

A node `(a, b, c, d)` is the CONTRACTED integral `(a b | c d)`: four Cartesian
angular-momentum vectors on the centres `A, B, C, D`. The recurrence, in build
form (eq 28, and its ket twin):

    (a, b+1_k | c, d) = (a+1_k, b | c, d) + AB_k (a, b | c, d)
    (a, b | c, d+1_k) = (a, b | c+1_k, d) + CD_k (a, b | c, d)

with `AB = A - B` and `CD = C - D`. Both follow from `(x - B_k) = (x - A_k) +
AB_k`, and neither reads an exponent: the coefficients are geometry alone.

TWO FACTS THAT MAKE THIS LAYER DIFFERENT FROM EVERYTHING ELSE IN THE TREE.

1. NO BOYS ORDER. A node has no `m` (`Node` is four Cartesians and nothing
   else; test/test_hgp_hrr.py asserts it structurally). The VRR's
   `(e, f, m)` (`hgp/vrr.py`) ends at `m = 0`; the HRR starts there.
2. ITS INPUTS ARE CONTRACTED VALUES. Eq 28 holds for a contracted integral
   because `AB` is the same for every primitive pair of the shell pair `(a,
   b)` -- but it does NOT hold for the primitive `[e,0|f,0]^(m)` a VRR node
   is, so the `K^4` contraction sum has to finish before the HRR runs. The
   HRR therefore sits OUTSIDE the primitive loops, once per contracted
   quartet, which is why the fused kernel accumulates the boundary set in
   registers (milestone 12, pin 5) and runs this DAG after the loop.

THE BOUNDARY SET (`HrrDag.boundary`). The leaves are the `(e, 0 | f, 0)` the
reductions bottom out at, and they are exactly the VRR's target set: every
`(e, f)` with `la <= |e| <= la + lb` and `lc <= |f| <= lc + ld`
(`hgp.vrr.target_set`; test/test_hgp_hrr.py asserts the equality for all 21
classes). The lower bound is real: the transfer term moves a unit from `b`
to `a` (or `d` to `c`) and the `AB` (`CD`) term drops one from `b` (`d`), so
no term lowers `a` and none raises `|a| + |b|`. At `(dd|dd)` that is
31 x 31 = 961 values -- not the 1,225 = 35 x 35 #274's text prints, which is
the same set with the lower bound dropped (#273's comment; the orchestrator
read 1,225 as a misprint).

THE DIRECTION, a choice the pinned counts depend on. A non-leaf node is
reduced along ONE side and ONE axis. The side: the KET first (`d != 0`), so
in build order the bra HRR runs first -- `(e0|f0) -> (ab|f0)` -- and the ket
HRR last -- `(ab|f0) -> (ab|cd)`. The axis: the first (x, y, z order) nonzero
component of `d` (else of `b`), as `hgp.vrr` chooses its axis. Under this
ladder's canonicalisation (bra pair `<=` ket pair) that side order is the
cheaper one in 9 of the 21 classes, equal in 11 and dearer in one, `(pp|pd)`;
`build(cls, first="ket")` is the other order, pinned beside it in the test.
Taking the LAST nonzero axis instead changed no count on the ladder when
this was written; that is not pinned.

THE STATEMENT COUNT (`HrrDag.n_statements`) counts EVERY node as one
statement: a boundary node is one read of a contracted accumulator, an
interior node one `x = y + AB_k z` (or `CD_k`). Under it `(dd|dd)` has 6,254
statements, #274's figure: 961 boundary reads plus 5,293 two-term updates
(`n_interior`). The 1,296 = 6^4 ERIs are the targets (`n_eri`). With the
issue's 1,225 boundary values the same convention would give 6,518, so the
issue's two figures are consistent only under the bounded set.

Pure Python: no card, no compiler, no import from `plan/` or `reference/`
(test/test_layering.py, rule "hgp").
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from felizjueves.hgp.vrr import ZERO, AmClass, Cart, cartesians, target_set

Node = tuple[Cart, Cart, Cart, Cart]  # (a, b, c, d): (a b | c d), contracted

AXES = "xyz"
# The coefficient names: the identity on the transferred term, and one
# component of a centre separation on the lowered one.
COEFS = ("1",) + tuple(f"AB_{k}" for k in AXES) + tuple(f"CD_{k}" for k in AXES)
SIDES = ("bra", "ket")


def _shift(v: Cart, k: int, by: int) -> Cart:
    return tuple(c + by * (i == k) for i, c in enumerate(v))  # type: ignore[return-value]


@dataclass(frozen=True)
class Term:
    """One term of a node's recurrence: `coef * child`."""

    coef: str
    child: Node


def is_boundary(node: Node) -> bool:
    """`(e, 0 | f, 0)`: a contracted VRR output, never reduced."""
    return node[1] == ZERO and node[3] == ZERO


def reduce(node: Node, first: str = "bra") -> tuple[Term, ...]:
    """The two terms that compute `node`, or `()` for a boundary node.

    `first` is the side whose HRR runs first in BUILD order, so the other
    side is reduced first here (module docstring, "THE DIRECTION")."""
    if first not in SIDES:
        raise ValueError(f"first={first!r} is not one of {SIDES}")
    a, b, c, d = node
    for side in (("ket", "bra") if first == "bra" else ("bra", "ket")):
        if side == "ket" and d != ZERO:
            k = next(i for i, x in enumerate(d) if x)
            dm = _shift(d, k, -1)
            return (Term("1", (a, b, _shift(c, k, 1), dm)),
                    Term(f"CD_{AXES[k]}", (a, b, c, dm)))
        if side == "bra" and b != ZERO:
            k = next(i for i, x in enumerate(b) if x)
            bm = _shift(b, k, -1)
            return (Term("1", (_shift(a, k, 1), bm, c, d)),
                    Term(f"AB_{AXES[k]}", (a, bm, c, d)))
    return ()


def targets(cls: AmClass) -> tuple[Node, ...]:
    """Every ERI of `(la lb | lc ld)`, `a` outermost, each in `cartesians`
    order: the 6^4 = 1,296 integrals of `(dd|dd)`."""
    target_set(cls)  # the class check, in one place
    la, lb, lc, ld = cls
    return tuple((a, b, c, d) for a in cartesians(la) for b in cartesians(lb)
                 for c in cartesians(lc) for d in cartesians(ld))


def order_key(node: Node, first: str = "bra") -> tuple:
    """A build order: boundary nodes, then the side run first by its lowered
    index's order ascending, then the other side likewise; ties by
    Cartesian order. Within one level no node reads another."""
    a, b, c, d = node
    lead, tail = (b, d) if first == "bra" else (d, b)
    return (tail != ZERO, sum(tail), sum(lead),
            tuple(-x for v in node for x in v))


@dataclass(frozen=True, eq=False)
class HrrDag:
    """The HRR of one class: its targets, nodes, edges, boundary and order."""

    cls: AmClass
    first: str
    targets: tuple[Node, ...]
    terms: Mapping[Node, tuple[Term, ...]]  # every node; boundary -> ()
    order: tuple[Node, ...]

    @property
    def nodes(self) -> frozenset[Node]:
        return frozenset(self.terms)

    @property
    def boundary(self) -> frozenset[tuple[Cart, Cart]]:
        """The `(e, f)` of every `(e, 0 | f, 0)` leaf: what the VRR must
        produce and the fused kernel accumulates over the primitive loops."""
        return frozenset((n[0], n[2]) for n, t in self.terms.items() if not t)

    @property
    def n_eri(self) -> int:
        return len(self.targets)

    @property
    def n_boundary(self) -> int:
        return sum(1 for t in self.terms.values() if not t)

    @property
    def n_interior(self) -> int:
        """Two-term updates, one per non-boundary node."""
        return sum(1 for t in self.terms.values() if t)

    @property
    def n_statements(self) -> int:
        """Every node, boundary reads included (module docstring)."""
        return len(self.terms)

    def edges(self) -> Iterable[tuple[Node, Node]]:
        """(parent, child) pairs, parent computed from child."""
        for n, ts in self.terms.items():
            for t in ts:
                yield n, t.child


def build(cls: AmClass, first: str = "bra") -> HrrDag:
    """The HRR DAG of `(la lb | lc ld)`: the closure of its ERIs under
    `reduce`, which ends exactly on the boundary set."""
    tg = targets(cls)
    terms: dict[Node, tuple[Term, ...]] = {}
    todo = list(tg)
    while todo:
        n = todo.pop()
        if n in terms:
            continue
        terms[n] = reduce(n, first)
        todo.extend(t.child for t in terms[n])
    order = tuple(sorted(terms, key=lambda n: order_key(n, first)))
    return HrrDag(tuple(cls), first, tg, MappingProxyType(terms), order)
