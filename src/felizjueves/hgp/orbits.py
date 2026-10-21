"""ERI symmetry orbits of an ordered HGP class, and their cheapest orientation.

    orbit(cls)                       -> the distinct members of cls's orbit
    representative(cls)              -> the member the canonical mode builds
    rank_key(cls)                    -> (HRR statements, VRR nodes, cls)
    to_representative(cls)           -> Orientation: the shell and axis maps
    ordered_classes(lmax_total, lmax_shell=2)
    canonical_classes(lmax_total, lmax_shell=2)

A split build (#403-#405) visits every ORDERED quartet, so `(ps|ss)`,
`(sp|ss)`, `(ss|ps)` and `(ss|sp)` are four distinct visits. The real ERI's
8-fold permutational symmetry,

    (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba),

makes them one ORBIT, and the canonical class mode builds one kernel per
orbit, its REPRESENTATIVE, and deposits each ordered quartet through the
permutation `to_representative` returns. `GROUP` lists the eight elements as
shell-position tuples `g`: the class `g . cls` has `cls[g[j]]` in position
`j`. An orbit has 1, 2, 4 or 8 distinct members.

THE REPRESENTATIVE IS CHOSEN, NOT ASSUMED. The HGP HRR's cost depends on the
orientation (`hgp.hrr.build`, default `first="bra"`: `(dp|ss)` 16 boundary /
34 statements against `(pd|ss)` 19 / 60; `(ps|dp)` 48 / 102 against
`(sp|pd)` 76 / 256). So `representative` takes the orbit member with the
smallest `rank_key`:

    1. `hgp.hrr.build(m).n_statements` -- the HRR statements, boundary
       reads included (`HrrDag.n_statements`);
    2. `hgp.vrr.build(m).n_nodes` -- the pruned VRR's node count, the SECOND
       key, and it decides: at L <= 4 it splits the HRR tie in `(ds|ps)`
       (43 VRR nodes against `(ps|ds)`'s 52), `(fs|ps)` (85 against 122),
       `(dp|ps)` (103 against `(ps|dp)`'s 122) and `(ds|pp)` (116 against
       `(pp|ds)`'s 146), each time AGAINST the lexicographic order;
    3. the class tuple itself, lexicographically: the remaining ties are
       bra-ket mirrors with equal counts (`(ss|ps)` over `(ps|ss)`).

test/test_hgp_orbits.py pins every representative at L <= 4 with both counts.

NOT `Target.F`'S LADDER ORIENTATION. `hgp.vrr.CLASSES` canonicalises
`la <= lb`, `lc <= ld` and bra pair `<=` ket pair; the representative often
is not that member (`(ss|ps)`, not the ladder's `(ss|sp)`; `(ds|ps)`, not
`(sp|sd)`). `F` is left as it is (milestone 14, #402: recorded, not changed).

THE PERMUTATION (`Orientation`). A block is laid out `[a][b][c][d]`, each
shell's Cartesians in `cartesians` order, the first index outer (`ao.eri`'s
and `hgp.hrr.targets`' layout). To get the ordered quartet's block from the
representative's: evaluate the representative on the shells
`shells[j] for j in Orientation.shells` (the ordered quartet's shells, put in
the representative's positions), then `block_rep.transpose(axes)` is the
ordered block, `axes` the inverse of `shells`; `gather` is the same map on
flat indices (`ordered.flat[k] == rep.flat[gather[k]]`). Cartesian order
within a shell is untouched: a permutation moves whole shells. When several
group elements carry `cls` onto the representative (equal `l`s), the first
in `GROUP` order is the one returned, fixed so a deposit is reproducible.

THE SHELL CAP. `ordered_classes(lmax_total, lmax_shell=2)` is every
`(la, lb, lc, ld)` with `la + lb + lc + ld <= lmax_total` and each `l <=
lmax_shell`; the default cap is the HGP engine's built ceiling (d; `f` and
above are #330), which gives the 50 ordered classes and 13 orbits #402
counts at L <= 4. `lmax_shell=None` lifts it (70 and 17 at L <= 4).
Enumeration builds no DAG; only `representative` / `rank_key` /
`to_representative` / `canonical_classes` do, memoised per class.

Pure Python: no card, no compiler, no import from `plan/` or `reference/`
(test/test_layering.py, rule "hgp").
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache

from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.vrr import AmClass, cartesians

Perm = tuple[int, int, int, int]

GROUP: tuple[Perm, ...] = (
    (0, 1, 2, 3),  # (ab|cd)
    (1, 0, 2, 3),  # (ba|cd)
    (0, 1, 3, 2),  # (ab|dc)
    (1, 0, 3, 2),  # (ba|dc)
    (2, 3, 0, 1),  # (cd|ab)
    (3, 2, 0, 1),  # (dc|ab)
    (2, 3, 1, 0),  # (cd|ba)
    (3, 2, 1, 0),  # (dc|ba)
)


def _check(cls) -> AmClass:
    cls = tuple(cls)
    if len(cls) != 4 or any(type(x) is not int or x < 0 for x in cls):
        raise ValueError(f"class {cls!r} is not four ints >= 0 (la, lb, lc, ld)")
    return cls  # type: ignore[return-value]


def apply(g: Perm, cls: AmClass) -> AmClass:
    """`g . cls`: position `j` holds `cls[g[j]]`."""
    return tuple(cls[i] for i in g)  # type: ignore[return-value]


def orbit(cls: AmClass) -> tuple[AmClass, ...]:
    """The distinct members of `cls`'s 8-element orbit, sorted."""
    cls = _check(cls)
    return tuple(sorted({apply(g, cls) for g in GROUP}))


@cache
def rank_key(cls: AmClass) -> tuple[int, int, AmClass]:
    """`(HRR statements, VRR nodes, cls)` (module docstring)."""
    cls = _check(cls)
    return (hrr.build(cls).n_statements, vrr.build(cls).n_nodes, cls)


def representative(cls: AmClass) -> AmClass:
    """The orbit member with the smallest `rank_key`."""
    return min(orbit(cls), key=rank_key)


def _inverse(g: Perm) -> Perm:
    inv = [0] * 4
    for j, i in enumerate(g):
        inv[i] = j
    return tuple(inv)  # type: ignore[return-value]


@dataclass(frozen=True)
class Orientation:
    """How the ordered class `cls` is served by its representative `rep`.

    `shells[j]` is the ordered quartet's shell position that goes to the
    representative's position `j` (so `rep == apply(shells, cls)`); `axes`
    is its inverse, `ordered = rep_block.transpose(axes)`."""

    cls: AmClass
    rep: AmClass
    shells: Perm
    axes: Perm

    @property
    def gather(self) -> tuple[int, ...]:
        """Flat indices: `ordered.flat[k] == rep.flat[gather[k]]`, both
        blocks in `[a][b][c][d]` row-major order."""
        dims = [len(cartesians(x)) for x in self.rep]
        strides = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1]
        out = []
        for idx in itertools.product(*(range(len(cartesians(x)))
                                       for x in self.cls)):
            # ordered index i_k sits on rep axis inv(axes)[k] == shells^-1.
            out.append(sum(idx[self.shells[j]] * strides[j] for j in range(4)))
        return tuple(out)


@cache
def to_representative(cls: AmClass) -> Orientation:
    """The first `GROUP` element carrying `cls` onto `representative(cls)`."""
    cls = _check(cls)
    rep = representative(cls)
    g = next(g for g in GROUP if apply(g, cls) == rep)
    return Orientation(cls, rep, g, _inverse(g))


def ordered_classes(lmax_total: int, lmax_shell: int | None = 2
                    ) -> tuple[AmClass, ...]:
    """Every ordered `(la, lb, lc, ld)` with total `<= lmax_total` and each
    `l <= lmax_shell` (`None`: no cap), lexicographic. Builds no DAG."""
    for name, v in (("lmax_total", lmax_total), ("lmax_shell", lmax_shell)):
        if v is None and name == "lmax_shell":
            continue
        if type(v) is not int or v < 0:
            raise ValueError(f"{name}={v!r} is not an int >= 0")
    top = lmax_total if lmax_shell is None else min(lmax_total, lmax_shell)
    return tuple(c for c in itertools.product(range(top + 1), repeat=4)
                 if sum(c) <= lmax_total)


def canonical_classes(lmax_total: int, lmax_shell: int | None = 2
                      ) -> tuple[AmClass, ...]:
    """One representative per orbit of `ordered_classes`, sorted. Builds
    every member's DAGs (representative selection)."""
    return tuple(sorted({representative(c)
                         for c in ordered_classes(lmax_total, lmax_shell)}))
