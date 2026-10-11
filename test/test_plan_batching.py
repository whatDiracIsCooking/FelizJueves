"""The plan layer's targets and batch partition: [JEngine, v2, Alg. 4] + minimax."""

from __future__ import annotations

from itertools import combinations

import pytest

from felizjueves.plan.batching import (
    ALL,
    AXES,
    MINIMAX_TARGETS,
    batch_direction,
    batch_size,
    empty_batches,
    in_batch,
    largest_batch,
    max_targets_per_batch,
    partition,
    target_counts,
)
from felizjueves.plan.pipeline import plan
from felizjueves.plan.targets import hermite_targets, n_targets
from felizjueves.spec import Operator, Spec, Target

# [JEngine, v2, §5]: L = 5, direction Y, a..b = 1..2
Y_12 = {
    (0, 1, 0),  # p_y
    (0, 1, 1),  # d_yz        ry == rz -- tie-break member
    (0, 2, 0),  # d_y2
    (1, 2, 0),  # f_xy2
    (0, 2, 1),  # f_y2z
    (1, 2, 1),  # g_xy2z
    (0, 2, 2),  # g_y2z2      ry == rz -- tie-break member
    (1, 2, 2),  # h_xy2z2     ry == rz -- tie-break member
}


def test_the_y_1_2_batch_is_the_papers_eight_targets() -> None:
    got = {r for r in hermite_targets(5) if in_batch(r, "y", 1, 2)}
    assert got == Y_12


def test_three_members_are_present_only_because_of_the_tie_break() -> None:
    # Under the opposite (successor) convention these sort into Z instead, so
    # a round-robin implemented backwards fails on exactly these three.
    assert {r for r in Y_12 if r[1] == r[2]} == {(0, 1, 1), (0, 2, 2), (1, 2, 2)}


@pytest.mark.parametrize(
    ("r", "direction"),
    [
        ((0, 0, 0), "x"),  # all-equal -> x
        ((2, 2, 2), "x"),
        ((3, 1, 0), "x"),
        ((0, 3, 1), "y"),
        ((1, 0, 3), "z"),
        ((2, 2, 0), "x"),  # {x,y} -> x
        ((0, 2, 2), "y"),  # {y,z} -> y
        ((2, 0, 2), "z"),  # {z,x} -> z
    ],
)
def test_ties_go_to_the_cyclic_predecessor(r, direction) -> None:
    assert batch_direction(r) == direction


@pytest.mark.parametrize("L", range(0, 21))
def test_the_enumeration_matches_the_closed_form(L: int) -> None:
    targets = hermite_targets(L)
    assert len(targets) == len(set(targets)) == n_targets(L)
    assert all(min(r) >= 0 and sum(r) <= L for r in targets)


def test_the_target_order_is_fixed() -> None:
    assert hermite_targets(2) == (
        (0, 0, 0),
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2),
    )


@pytest.mark.parametrize("L", range(7, 17))
@pytest.mark.parametrize("n_batches", (1, 3, 6, 9))
def test_the_partition_covers_every_target_exactly_once(L, n_batches) -> None:
    targets = hermite_targets(L)
    batches = partition(L, n_batches)
    assert len(batches) == n_batches
    seen = [r for b in batches for r in targets if in_batch(r, *b)]
    assert sorted(seen) == sorted(targets)  # disjoint AND covering


@pytest.mark.parametrize("bad", [0, 2, 4, 5, -3, 3.0, True])
def test_an_illegal_batch_count_raises(bad) -> None:
    with pytest.raises(ValueError):
        partition(10, bad)


def test_too_many_sub_batches_for_the_class_raises() -> None:
    # k = 3 sub-batches per direction, but at L = 1 the max component takes
    # only the values 0 and 1.
    with pytest.raises(ValueError, match="sub-batches per direction"):
        partition(1, 9)


# -- #64: which (L, n_batches) the plan layer admits ---------------------------

LEGAL_COUNTS = (1, *range(3, 22, 3))  # every legal n_batches up to 21

# Refused because a batch would come out EMPTY: Y_{0,0} and Z_{0,0}, always
# exactly those two. The first six are the cells of L = 0..16 x {1, 3, 6, 9}.
EMPTY = {
    (0, 3), (1, 6), (2, 6), (2, 9), (3, 9), (5, 9),
    (3, 12), (4, 12), (6, 12),
    (4, 15), (5, 15), (7, 15),
    (5, 18), (6, 18), (7, 18), (8, 18),
    (6, 21), (7, 21), (8, 21), (9, 21), (10, 21), (11, 21),
}


def _too_many(L: int, n_batches: int) -> bool:
    """More sub-batches per direction than the max component has values."""
    return n_batches > 1 and n_batches // 3 > L + 1


@pytest.mark.parametrize("L", range(0, 17))
@pytest.mark.parametrize("n_batches", LEGAL_COUNTS)
def test_the_feasibility_map_is_asserted(L: int, n_batches: int) -> None:
    # Every pair either partitions into n_batches non-empty batches covering
    # every target once, or is refused by name -- never an exception from a
    # later stage. The map is the artifact: EMPTY and _too_many ARE the rule.
    if _too_many(L, n_batches):
        with pytest.raises(ValueError, match="sub-batches per direction"):
            partition(L, n_batches)
    elif (L, n_batches) in EMPTY:
        with pytest.raises(
            ValueError,
            match=rf"L = {L}, n_batches = {n_batches} .*2 batches with no "
            r"targets: Y_\{0,0\}, Z_\{0,0\}",
        ):
            partition(L, n_batches)
    else:
        targets = hermite_targets(L)
        batches = partition(L, n_batches)
        assert len(batches) == n_batches
        assert all(batch_size(L, b) > 0 for b in batches)
        seen = [r for b in batches for r in targets if in_batch(r, *b)]
        assert sorted(seen) == sorted(targets)


@pytest.mark.parametrize(("L", "n_batches"), sorted(EMPTY))
def test_an_empty_batch_is_refused_at_the_partition_not_the_sieve(
    L: int, n_batches: int
) -> None:
    # #64: these reached sieve._target_order and died in `max()` three frames
    # below the cause. The full pipeline must now stop at the partition.
    spec = Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, n_batches=n_batches)
    with pytest.raises(ValueError, match="no targets") as info:
        plan(spec)
    assert "iterable argument is empty" not in str(info.value)
    assert info.traceback[-1].name == "partition"


def test_the_six_grid_cells_are_the_ones_the_issue_names() -> None:
    grid = {(L, b) for L in range(0, 17) for b in (1, 3, 6, 9)}
    assert grid & EMPTY == {(0, 3), (1, 6), (2, 6), (2, 9), (3, 9), (5, 9)}
    # [JEngine] Tables 1-3 are L = 7..16 x {1, 3, 6, 9}: all of it plans.
    assert not any(L >= 7 and _too_many(L, b) for L, b in grid)
    assert all(L < 7 for L, _ in grid & EMPTY)


def test_legality_is_not_monotone_in_either_argument() -> None:
    # Pinned so nobody "simplifies" the rule to a threshold: each minimax cut
    # is chosen on its own counts, and only a first cut at 1 empties a batch.
    with pytest.raises(ValueError, match="no targets"):
        partition(5, 9)
    assert len(partition(6, 9)) == 9  # larger L, same n_batches: plans
    assert len(partition(5, 12)) == 12  # same L, more batches: plans
    assert len(partition(4, 9)) == 9  # and smaller L plans too
    with pytest.raises(ValueError, match="no targets"):
        partition(4, 12)


def test_empty_batches_names_only_y_and_z_at_zero() -> None:
    counts = target_counts(7)
    assert empty_batches(counts, (1,)) == (("y", 0, 0), ("z", 0, 0))
    assert empty_batches(counts, (2,)) == ()
    for L in range(1, 17):
        c = target_counts(L)
        for cut in combinations(range(1, L + 1), 1):
            assert bool(empty_batches(c, cut)) == (cut[0] == 1)


def test_an_unknown_direction_raises() -> None:
    with pytest.raises(ValueError, match="direction"):
        in_batch((1, 0, 0), "w", 0, 1)


@pytest.mark.parametrize("L", range(7, 17))
def test_the_one_batch_count_is_the_closed_form(L: int) -> None:
    assert partition(L, 1) == ((ALL, 0, L),)
    assert max_targets_per_batch(L, 1) == n_targets(L)


@pytest.mark.parametrize("L", range(0, 17))
def test_the_direction_groups_differ_only_in_their_tie_class(L: int) -> None:
    targets = hermite_targets(L)
    groups = {d: [r for r in targets if batch_direction(r) == d] for d in AXES}
    # Strict-maximum members: the same set up to cyclic relabelling.
    strict = {
        d: sorted(
            r[i:] + r[:i]
            for r in groups[d]
            if sum(1 for c in r if c == max(r)) == 1
        )
        for i, d in enumerate(AXES)
    }
    assert strict["x"] == strict["y"] == strict["z"]
    # Two-way ties: each direction absorbs exactly one pair, equally many.
    pairs = {
        d: [r for r in groups[d] if sum(1 for c in r if c == max(r)) == 2]
        for d in AXES
    }
    assert len(pairs["x"]) == len(pairs["y"]) == len(pairs["z"])
    assert all(r[0] == r[1] for r in pairs["x"])
    assert all(r[1] == r[2] for r in pairs["y"])
    assert all(r[2] == r[0] for r in pairs["z"])
    # All-equal: only X, which is why X is the largest group at three batches.
    all_equal = [r for r in targets if r[0] == r[1] == r[2]]
    assert len(all_equal) == L // 3 + 1
    assert set(all_equal) <= set(groups["x"])
    assert len(groups["x"]) == len(groups["y"]) + len(all_equal)
    assert len(groups["y"]) == len(groups["z"])
    if (L, 3) in EMPTY:  # L = 0: Y and Z hold nothing, so 3 batches is refused
        assert not groups["y"] and not groups["z"]
        with pytest.raises(ValueError, match="no targets"):
            max_targets_per_batch(L, 3)
    else:
        assert max_targets_per_batch(L, 3) == len(groups["x"])


def test_figure_4s_cut_at_l_6() -> None:
    # [JEngine, v2, Fig. 4]: X_{0,2}, X_{3,6}, Y_{1,2}, Y_{3,6}, Z_{1,2},
    # Z_{3,6}. Y_{0,2} and Y_{1,2} hold the same targets (nothing in Y has a
    # maximum of 0), so the figure and the shared cut agree.
    assert partition(6, 6) == (
        ("x", 0, 2), ("x", 3, 6),
        ("y", 0, 2), ("y", 3, 6),
        ("z", 0, 2), ("z", 3, 6),
    )
    assert batch_size(6, ("y", 0, 0)) == batch_size(6, ("z", 0, 0)) == 0


@pytest.mark.parametrize("L", range(2, 11))
@pytest.mark.parametrize("k", (2, 3))
def test_the_cut_is_minimax_by_brute_force(L: int, k: int) -> None:
    # Every contiguous cut, measured by counting targets directly through
    # in_batch rather than through the per-magnitude counts the rule reads.
    targets = hermite_targets(L)

    def largest(cuts) -> int:
        edges = (0, *cuts, L + 1)
        return max(
            sum(1 for r in targets if in_batch(r, d, edges[i], edges[i + 1] - 1))
            for d in AXES
            for i in range(k)
        )

    sizes = {cuts: largest(cuts) for cuts in combinations(range(1, L + 1), k - 1)}
    best = min(sizes.values())
    if (L, 3 * k) in EMPTY:  # #64: the minimax cut starts at 1 -- refused
        with pytest.raises(ValueError, match="no targets"):
            max_targets_per_batch(L, 3 * k)
    else:
        assert max_targets_per_batch(L, 3 * k) == best
    # ... and among the optimal cuts, the lowest boundaries.
    chosen = MINIMAX_TARGETS(target_counts(L), k)
    assert chosen == min(c for c, s in sizes.items() if s == best)
    assert largest_batch(target_counts(L), chosen) == best


def test_the_cut_rule_is_a_seam() -> None:
    def highest(counts, k):
        L = len(counts["x"]) - 1
        return tuple(range(L - k + 2, L + 1))

    assert partition(8, 6, cut=highest) == (
        ("x", 0, 7), ("x", 8, 8),
        ("y", 0, 7), ("y", 8, 8),
        ("z", 0, 7), ("z", 8, 8),
    )
    with pytest.raises(ValueError, match="cut rule returned"):
        partition(8, 6, cut=lambda counts, k: (0,))
