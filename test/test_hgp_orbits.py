"""`hgp.orbits` (#402): the ERI symmetry orbits and their cheapest orientation.

THE COUNTS. At L <= 4 with the HGP shell cap (`l <= 2`, the default) there
are 50 ordered classes in 13 orbits, #402's figures; without the cap
(`lmax_shell=None`) 70 in 17. Every representative is pinned below with its
HRR statement count and its VRR node count, the two keys it was chosen by.

THE PERMUTATION is checked two ways, every ordered class at L <= 4 (all 70,
the cap lifted):

(a) EXACTLY, on a symbolic label tensor: an element of the ordered block is
    the quartet of `(shell position, Cartesian index)` labels, canonicalised
    under the 8-fold ERI symmetry -- the identity of the integral, with no
    arithmetic in it. The representative's label tensor, built on the
    ordered quartet's shells in the representative's positions and
    transposed by `Orientation.axes`, must equal the ordered one bitwise
    (and `gather` must agree with the transpose). A permutation outside the
    group fails it (the mutation below).

(b) NUMERICALLY, on `eri/eri.py`'s block: the representative's block on the
    permuted shells, transposed, against the ordered class's block, within

        |rep^T - ordered| <= 1e-12 * (scale_rep^T + scale_ordered),

    each `scale` `eri.eri`'s absolute-value twin -- test/test_hgp_eri.py's
    stated bound, the same Higham argument (two fp64 evaluations of the
    same sum in different orders each err by at most `gamma_n` of their
    twin). It is NOT bitwise, by measurement, and the user approved the
    tolerance in place of #402's "bitwise": on a (pd|sp) quartet on four
    centres, `eri.eri` agrees bitwise only for the identity and the ket swap
    (the bra swap and the four bra-ket swaps differ by up to 1.1e-16), and
    `eri.hgp_drive` only for the identity (the other seven by 6e-16 to
    1e-15): each orientation sums in a different order. Measured worst
    here when written (2026-10-21): 2.6e-16 of the summed scale, with 18 of
    the 70 classes bitwise.

Fast tier: L <= 4 (tiers.FAST_L_MAX) only. Enumeration at any L builds no
DAG (asserted); representative selection above L = 4 is `slow`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from felizjueves.ao.shell import Shell, cartesians
from felizjueves.eri.eri import eri_block_and_scale
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp.orbits import (
    GROUP,
    apply,
    canonical_classes,
    orbit,
    ordered_classes,
    rank_key,
    representative,
    to_representative,
)
from felizjueves.reference.pq import pq

TOL = 1e-12
ALL4 = ordered_classes(4, lmax_shell=None)

# representative -> (HRR statements, pruned VRR nodes), every orbit at L <= 4.
REPS = {
    (0, 0, 0, 0): (1, 1),
    (0, 0, 1, 0): (3, 5),
    (0, 0, 2, 0): (6, 15),
    (0, 0, 3, 0): (10, 35),
    (0, 0, 4, 0): (15, 70),
    (0, 0, 1, 1): (18, 15),
    (0, 0, 2, 1): (34, 35),
    (0, 0, 3, 1): (55, 70),
    (0, 0, 2, 2): (110, 70),
    (1, 0, 1, 0): (9, 18),
    (2, 0, 1, 0): (18, 43),
    (3, 0, 1, 0): (30, 85),
    (1, 0, 1, 1): (54, 52),
    (2, 1, 1, 0): (102, 103),
    (2, 0, 2, 0): (36, 116),
    (2, 0, 1, 1): (108, 116),
    (1, 1, 1, 1): (243, 146),
}
CAPPED = {r for r in REPS if max(r) <= 2}


# ---------------------------------------------------------------------------
# Enumeration and orbits
# ---------------------------------------------------------------------------


def test_the_counts_at_l4() -> None:
    capped = ordered_classes(4)
    assert len(capped) == 50
    assert len({orbit(c) for c in capped}) == 13
    assert len(ALL4) == 70
    assert len({orbit(c) for c in ALL4}) == 17
    assert set(capped) == {c for c in ALL4 if max(c) <= 2}
    assert ordered_classes(4, lmax_shell=2) == capped


def test_ordered_classes_is_every_tuple_under_the_caps() -> None:
    for lt, ls in [(0, 2), (3, 1), (5, 2), (6, None), (2, 7)]:
        top = lt if ls is None else min(lt, ls)
        want = tuple(c for c in itertools.product(range(lt + 1), repeat=4)
                     if sum(c) <= lt and max(c) <= top)
        assert ordered_classes(lt, ls) == want
    for bad in [(-1, 2), (4, -1), (4.0, 2), (4, "2")]:
        with pytest.raises(ValueError):
            ordered_classes(*bad)


def test_orbits_partition_the_ordered_classes() -> None:
    for c in ALL4:
        o = orbit(c)
        assert c in o
        assert len(o) in (1, 2, 4, 8)
        assert o == tuple(sorted(set(o)))
        for m in o:
            assert orbit(m) == o
    assert orbit((0, 1, 2, 3)) == tuple(sorted(apply(g, (0, 1, 2, 3))
                                               for g in GROUP))
    assert len(orbit((0, 1, 2, 3))) == 8
    assert orbit((1, 1, 1, 1)) == ((1, 1, 1, 1),)


def test_enumeration_builds_no_dag(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("a DAG was built")

    monkeypatch.setattr(hrr, "build", boom)
    monkeypatch.setattr(vrr, "build", boom)
    big = ordered_classes(12, lmax_shell=None)
    assert len(big) == 1820  # C(16, 4)
    assert len(ordered_classes(9)) == len(
        [c for c in itertools.product(range(3), repeat=4) if sum(c) <= 9])
    assert len(orbit((6, 0, 5, 1))) == 8


def test_bad_classes_are_refused() -> None:
    for bad in [(0, 0, 0), (0, 0, 0, -1), (0, 0, 0, 1.0)]:
        with pytest.raises(ValueError):
            orbit(bad)


# ---------------------------------------------------------------------------
# The representative
# ---------------------------------------------------------------------------


def test_every_representative_is_pinned_with_its_counts() -> None:
    assert canonical_classes(4, lmax_shell=None) == tuple(sorted(REPS))
    assert canonical_classes(4) == tuple(sorted(CAPPED))
    assert len(CAPPED) == 13
    for rep, (n_hrr, n_vrr) in REPS.items():
        assert rank_key(rep) == (n_hrr, n_vrr, rep)
        assert hrr.build(rep).n_statements == n_hrr
        assert vrr.build(rep).n_nodes == n_vrr
    for c in ALL4:
        rep = representative(c)
        assert rep in REPS
        assert rep in orbit(c)
        assert rank_key(rep) == min(rank_key(m) for m in orbit(c))


def test_the_issues_hrr_figures() -> None:
    """#402's text prints `(ds|ss)` / `(sd|ss)`; the counts are (dp|ss) and
    (pd|ss)'s ((ds|ss) is 6 / 6, (sd|ss) 10 / 25)."""
    for cls, (n_b, n_s) in {(2, 1, 0, 0): (16, 34), (1, 2, 0, 0): (19, 60),
                            (1, 0, 2, 1): (48, 102), (0, 1, 1, 2): (76, 256),
                            (2, 0, 0, 0): (6, 6), (0, 2, 0, 0): (10, 25)}.items():
        d = hrr.build(cls)
        assert (d.n_boundary, d.n_statements) == (n_b, n_s), cls
    assert representative((1, 2, 0, 0)) == (0, 0, 2, 1)
    assert representative((0, 1, 1, 2)) == (2, 1, 1, 0)


def test_the_vrr_key_decides_against_the_lexicographic_order() -> None:
    for rep, lex_first in {(2, 0, 1, 0): (1, 0, 2, 0),
                           (3, 0, 1, 0): (1, 0, 3, 0),
                           (2, 1, 1, 0): (1, 0, 2, 1),
                           (2, 0, 1, 1): (1, 1, 2, 0)}.items():
        assert rank_key(rep)[0] == rank_key(lex_first)[0]
        assert rank_key(rep)[1] < rank_key(lex_first)[1]
        assert lex_first < rep
        assert representative(lex_first) == rep


def test_the_representative_is_not_fs_ladder_orientation() -> None:
    """Recorded, not changed (module docstring)."""
    assert (0, 0, 0, 1) in vrr.CLASSES and representative((0, 0, 0, 1)) == (0, 0, 1, 0)
    assert (0, 1, 0, 2) in vrr.CLASSES and representative((0, 1, 0, 2)) == (2, 0, 1, 0)
    assert sum(r in vrr.CLASSES for r in CAPPED) < len(CAPPED)


@pytest.mark.slow
def test_representatives_above_l4() -> None:
    reps = canonical_classes(6)
    assert len(reps) == len({orbit(c) for c in ordered_classes(6)})
    for r in reps:
        assert representative(r) == r


# ---------------------------------------------------------------------------
# The permutation: (a) exact, on labels
# ---------------------------------------------------------------------------


def _canon(q):
    return min(apply(g, q) for g in GROUP)


def _labels(cls, shell_of):
    """The block of `cls` on shells `shell_of[j]` (positions of the ordered
    quartet), each element its canonical label quartet."""
    dims = [len(cartesians(x)) for x in cls]
    out = np.empty(dims, dtype=object)
    for idx in itertools.product(*(range(n) for n in dims)):
        out[idx] = _canon(tuple((shell_of[j], idx[j]) for j in range(4)))
    return out


@pytest.mark.parametrize("cls", ALL4, ids=str)
def test_the_permutation_on_labels_is_exact(cls) -> None:
    o = to_representative(cls)
    assert o.cls == cls and o.rep == representative(cls)
    assert apply(o.shells, cls) == o.rep
    assert o.shells in GROUP
    assert tuple(o.shells[i] for i in o.axes) == (0, 1, 2, 3)
    ordered = _labels(cls, (0, 1, 2, 3))
    rep = _labels(o.rep, o.shells)
    got = rep.transpose(o.axes)
    assert got.shape == ordered.shape
    assert all(a == b for a, b in zip(got.ravel(), ordered.ravel(), strict=True))
    # `gather` is the same map on flat indices.
    flat = rep.ravel()
    assert all(flat[k] == ordered.ravel()[i]
               for i, k in enumerate(o.gather))
    assert sorted(o.gather) == list(range(ordered.size))


def test_a_permutation_outside_the_group_fails_the_label_check() -> None:
    cls = (1, 1, 1, 1)
    bad = (0, 2, 1, 3)  # (ac|bd): not an ERI symmetry
    assert bad not in GROUP
    ordered = _labels(cls, (0, 1, 2, 3))
    rep = _labels(cls, bad)
    assert not all(a == b for a, b in zip(rep.transpose(bad).ravel(),
                                          ordered.ravel(), strict=True))


# ---------------------------------------------------------------------------
# The permutation: (b) numeric, on eri/eri.py's block
# ---------------------------------------------------------------------------

CENTRES = ((0.10, 0.20, -0.30), (0.90, -0.40, 0.25),
           (-0.50, 0.70, 0.60), (0.30, -0.80, 1.10))


def _shell(pos: int, l: int) -> Shell:  # noqa: E741
    return Shell(l, CENTRES[pos], (1.3 + 0.2 * pos, 0.4 + 0.1 * l),
                 (0.6, 0.5 - 0.05 * pos))


def _block(cls, shells):
    blk, scale = eri_block_and_scale((shells[0], shells[1]),
                                     (shells[2], shells[3]), pq=pq)
    dims = [len(cartesians(x)) for x in cls]
    return np.array(blk).reshape(dims), np.array(scale).reshape(dims)


@pytest.mark.parametrize("cls", ALL4, ids=str)
def test_the_permutation_on_ao_eri_blocks(cls) -> None:
    o = to_representative(cls)
    shells = [_shell(p, ang) for p, ang in enumerate(cls)]
    ordered, s_ord = _block(cls, shells)
    rep, s_rep = _block(o.rep, [shells[j] for j in o.shells])
    got, s_got = rep.transpose(o.axes), s_rep.transpose(o.axes)
    assert np.all(np.abs(got - ordered) <= TOL * (s_got + s_ord))
    assert np.array_equal(rep.ravel()[list(o.gather)], got.ravel())
    # The check has teeth: the values are not all one number.
    assert ordered.size == 1 or np.ptp(ordered) > 1e-6
