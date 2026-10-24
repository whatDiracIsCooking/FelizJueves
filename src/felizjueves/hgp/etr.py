"""The electron-transfer relation (ETR) as a second VRR DAG over `(e, f, m)` (#354).

`hgp.vrr` builds the ket of `[e,0|f,0]^(0)` by eq 26, the ket twin of eq 25
(the KET CASCADE). Head-Gordon and Pople's alternative -- the ETR of
docs/references.md entry 41 (not read for this module: the relation is
derived below, not transcribed) -- raises the bra to `|e| <= la + lb + lc +
ld` by eq 25 alone and then TRANSFERS angular momentum from the bra to the
ket at `m = 0`:

    [e, 0 | f+1_d, 0]^(0) = (QC_d + (p/q) PA_d) [e,0|f,0]^(0)
                          + e_d/(2q)            [e-1_d,0|f,0]^(0)
                          + f_d/(2q)            [e,0|f-1_d,0]^(0)
                          - (p/q)               [e+1_d,0|f,0]^(0)

each lower term absent when its index is zero. DERIVED, NOT TRANSCRIBED:
translational invariance, `(d/dA + d/dB + d/dC + d/dD) (ab|cd) = 0`, with
`d/dA_d` of a Cartesian Gaussian `= 2 alpha (a + 1_d) - a_d (a - 1_d)` and
`b = d = 0`, gives

    2 p [e+1_d|f] + 2 beta AB_d [e|f] + 2 q [e|f+1_d] + 2 delta CD_d [e|f]
        - e_d [e-1_d|f] - f_d [e|f-1_d] = 0

(the `b` and `d` raises moved onto `a` and `c` by the HRR shift); and
`-beta AB_d = p PA_d`, `-delta CD_d = q QC_d` (`PA = P - A`, `QC = Q - C`,
`hgp.recurrence`'s convention). So the coefficient of `[e|f]` is `QC_d +
(p/q) PA_d`, which is ONE per-quartet scalar per axis (`etr_c`), and the
transfer term's is `-p/q`. Only `m = 0` appears on the ETR side: the
relation is exact for each Boys order separately, and the boundary is
`m = 0`.

THE SAME BOUNDARY, A DIFFERENT DAG. `targets(cls)` is `hgp.vrr.targets(cls)`
-- the HRR's leaves -- so the two DAGs are interchangeable under the HRR;
what differs is how the boundary is reached. The ETR's bra phase is eq 25's
nodes (`hgp.vrr.reduce` with `f = 0`, the cascade's own bra phase) up to
`|e| = la + lb + lc + ld`, where the cascade stops at `la + lb`; its ket
phase is the transfer above at `m = 0` only, where the cascade carries every
`m <= L - |e| - |f|` through eq 26. Which is cheaper is a property of the
class (`hgp.cost.etr_cost`, the arbitration in `eri/fock/levers/etr.py`).

THE DIRECTION: a ket node `[e|f]` is reduced along the first axis (x, y, z)
with `f_d > 0`, as the cascade's eq 26 nodes are. A bra node (`f = 0`) is
the cascade's eq 25 reduction, verbatim. A choice; the pinned counts in
test/test_f_lever_etr.py depend on it.

THE ORDER: `hgp.vrr.order_key` -- bra phase by `|e|`, then the ket phase by
`|f|` ascending -- is topological here too: every ETR term reads a node of
lower `|f|` (`[e|f]`, `[e-1_d|f]`, `[e+1_d|f]` at `|f|`, `[e|f-1_d]` below)
or a bra node.

PRUNED BY DEFAULT, as `hgp.vrr.build` is: the forward build is every bra
node `|e| <= L`, `m <= L - |e|` and every ket node `1 <= |f| <= lc + ld`,
`|e| <= L - |f|`, `m = 0` (closed under `reduce`); the default keeps the
nodes some target reaches.

ADDITIVE: `hgp.vrr` is read, never changed; its DAGs, their pinned counts
(test/test_hgp_vrr.py) and `hgp/cost.json` are untouched. The DAG is a
`hgp.vrr.VrrDag`, so `hgp.vrr.group_dag`, `peak_live` and `vrr_peak` apply
to it unchanged.

Pure Python: no card, no compiler, no import from `plan/`, `reference/` or
`ir` (test/test_layering.py, rule "hgp").
"""

from __future__ import annotations

from types import MappingProxyType

from felizjueves.hgp import vrr
from felizjueves.hgp.vrr import ZERO, AmClass, Node, Term, VrrDag

__all__ = ["ETR", "build", "is_transfer", "reduce", "targets"]

# The ETR's coefficient names: `etr_c` is `QC_d + (p/q) PA_d` on the reduced
# axis, `inv2q` is `1/(2q)` (both lowering terms), `p_over_q` is `p/q` (the
# transfer term, its sign in `mult`).
ETR = ("etr_c", "inv2q", "p_over_q")


def targets(cls: AmClass) -> tuple[Node, ...]:
    """The boundary: `hgp.vrr.targets(cls)`, the same nodes in the same order."""
    return vrr.targets(cls)


def is_transfer(node: Node) -> bool:
    """A ket-phase node (`f != 0`): computed by the ETR, at `m = 0`."""
    return node[1] != ZERO


def _plus(v, d: int):
    return tuple(c + (i == d) for i, c in enumerate(v))


def reduce(node: Node) -> tuple[Term, ...]:
    """The terms that compute `node`: the ETR when `f != 0` (only at
    `m = 0`), else eq 25 (`hgp.vrr.reduce`'s bra phase). Term order:
    `etr_c`, the `e_d` lowering, the `f_d` lowering, the transfer."""
    e, f, m = node
    if f == ZERO:
        return vrr.reduce(node)
    if m != 0:
        raise ValueError(f"{node}: the ETR is applied at m = 0 only")
    d = next(i for i, c in enumerate(f) if c)
    g = vrr._minus(f, d)
    out = [Term("etr_c", 1, (e, g, 0))]
    if e[d]:
        out.append(Term("inv2q", e[d], (vrr._minus(e, d), g, 0)))
    if g[d]:
        out.append(Term("inv2q", g[d], (e, vrr._minus(g, d), 0)))
    out.append(Term("p_over_q", -1, (_plus(e, d), g, 0)))
    return tuple(out)


def _forward(cls: AmClass) -> dict[Node, tuple[Term, ...]]:
    la, lb, lc, ld = cls
    total = la + lb + lc + ld
    bra = {(e, ZERO, m): reduce((e, ZERO, m))
           for e in vrr._range(0, total) for m in range(total - sum(e) + 1)}
    ket = {(e, f, 0): reduce((e, f, 0))
           for f in vrr._range(1, lc + ld)
           for e in vrr._range(0, total - sum(f))}
    return {**bra, **ket}


def build(cls: AmClass, prune: bool = True) -> VrrDag:
    """The ETR DAG of `(la lb | lc ld)`: the same targets as
    `hgp.vrr.build(cls)`, reached by eq 25 then the transfer.

    `prune=False` is the forward build (module docstring); the default
    keeps only the nodes some target reaches."""
    cls = vrr._check(cls)
    terms = _forward(cls)
    tg = targets(cls)
    if prune:
        keep = vrr.reachable(terms, tg)
        terms = {n: t for n, t in terms.items() if n in keep}
    order = tuple(sorted(terms, key=vrr.order_key))
    return VrrDag(cls, prune, tg, MappingProxyType(terms), order)
