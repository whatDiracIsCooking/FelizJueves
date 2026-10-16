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

from felizjueves.hgp.vrr import (
    ZERO,
    AmClass,
    Cart,
    cartesians,
    peak_live,
    target_set,
)

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


# ---------------------------------------------------------------------------
# Slicing (#356, milestone 13 lane H2; additive: `build` and every count
# above are unchanged, test/test_hgp_hrr.py and hgp/cost.json pin them)
# ---------------------------------------------------------------------------
#
# A SLICE is the HRR of one bra pair `(a, b)`: the ERIs `(a b | c d)` for
# every `c, d`, and their closure under the parent DAG's own terms, in the
# parent's order filtered. `targets` lists `a` outermost, then `b`, so slice
# `s` is the contiguous run `targets[s n_cd : (s + 1) n_cd]`, and the slices,
# in `(a, b)` order, visit the targets in the parent's order. A node two
# slices share (a bra-HRR intermediate `(a' b' | f 0)`, a boundary leaf) is
# in both closures and is recomputed by both from the same terms.


def closure(terms: Mapping[Node, tuple[Term, ...]],
            roots: Iterable[Node]) -> frozenset[Node]:
    """Every node `roots` read, transitively, themselves included."""
    seen: set[Node] = set()
    todo = list(roots)
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo.extend(t.child for t in terms[n])
    return frozenset(seen)


def sub_dag(dag: HrrDag, targets: Iterable[Node],
            order: Iterable[Node] | None = None) -> HrrDag:
    """The closure of `targets` (a non-empty subset of `dag.targets`, kept
    in the given order) as an `HrrDag`: the parent's terms, and the
    parent's order filtered to the closure unless `order` (the closure,
    topologically) is given -- the shape #361's sub-slices reuse."""
    targets = tuple(targets)
    stray = set(targets) - set(dag.targets)
    if not targets or stray or len(set(targets)) != len(targets):
        raise ValueError(f"{dag.cls}: {targets!r} is not a non-empty set of "
                         "the DAG's targets")
    keep = closure(dag.terms, targets)
    terms = {n: t for n, t in dag.terms.items() if n in keep}
    if order is None:
        order = tuple(n for n in dag.order if n in keep)
    else:
        order = tuple(order)
        if len(order) != len(keep) or set(order) != keep:
            raise ValueError(f"{dag.cls}: the order is not the closure")
        seen: set[Node] = set()
        for n in order:
            if any(t.child not in seen for t in terms[n]):
                raise ValueError(f"{dag.cls}: {n} precedes a node it reads")
            seen.add(n)
    return HrrDag(dag.cls, dag.first, targets, MappingProxyType(terms), order)


def slice_targets(cls: AmClass) -> tuple[tuple[Node, ...], ...]:
    """`targets(cls)` cut per bra pair `(a, b)`, in `(a, b)` order: the
    `n_a n_b` contiguous runs of `n_c n_d` ERIs each."""
    tg = targets(cls)
    _, _, lc, ld = cls
    n_cd = len(cartesians(lc)) * len(cartesians(ld))
    return tuple(tg[i:i + n_cd] for i in range(0, len(tg), n_cd))


def slices(cls: AmClass, first: str = "bra") -> tuple[HrrDag, ...]:
    """The HRR of `cls` as one sub-DAG per bra pair (`slice_targets`)."""
    dag = build(cls, first)
    return tuple(sub_dag(dag, t) for t in slice_targets(cls))


def hrr_peak(dag: HrrDag) -> int:
    """Peak live values of `dag` in its own order, its ERIs live to the end
    (the digestion reads them after it): `hgp.vrr.peak_live`'s model."""
    return peak_live(dag.order, lambda n: (t.child for t in dag.terms[n]),
                     dag.targets)


# ---------------------------------------------------------------------------
# Sub-slicing (#361, milestone 13 lane H5; additive: `build`, `slices` and
# every count above are unchanged)
# ---------------------------------------------------------------------------
#
# A SUB-SLICE is a contiguous run of one slice's ERIs, in the slice's (and
# so the parent's) target order, and its closure under the parent DAG's
# terms, in the parent's order filtered (`sub_dag` unchanged). Contiguity in
# TARGET order is what keeps the digestion's visiting order the slice's: the
# sub-slices of a slice, in order, visit its ERIs exactly as the slice does.
# A node two sub-slices share (a bra-HRR `(a b | f 0)`, a leaf) is
# recomputed by both from the same terms, as #356's slices share theirs.


def runs(seq: Iterable[Node], n: int) -> tuple[tuple[Node, ...], ...]:
    """`seq` cut into `n` contiguous, near-equal, non-empty runs, in order
    (`hgp.vrr.group_targets`'s cut on a given sequence)."""
    seq = tuple(seq)
    if type(n) is not int or not 1 <= n <= len(seq):
        raise ValueError(f"{n!r} runs of {len(seq)} nodes; a run count is an "
                         f"int in 1..{len(seq)}")
    cut = [round(g * len(seq) / n) for g in range(n + 1)]
    return tuple(seq[a:b] for a, b in zip(cut, cut[1:], strict=False))


def sub_slices(cls: AmClass, n_sub: int, first: str = "bra"
               ) -> tuple[tuple[HrrDag, ...], ...]:
    """Each of `slices(cls)`'s slices cut into `n_sub` sub-DAGs of the
    WHOLE HRR (`sub_dag(build(cls), run)` per `runs(slice.targets, n_sub)`),
    in `(a, b)` order then run order."""
    dag = build(cls, first)
    return tuple(tuple(sub_dag(dag, r) for r in runs(t, n_sub))
                 for t in slice_targets(cls))
