"""`Spec`, the join key: frozen, canonical, and narrow on purpose."""

from __future__ import annotations

import dataclasses

import pytest

from felizjueves.spec import AM_LENGTH, Operator, Spec, Target


def _spec(**kw) -> Spec:
    base = dict(
        operator=Operator.COULOMB, derivative_order=0, am=(4, 4, 4, 4), target=Target.J
    )
    return Spec(**{**base, **kw})


def test_key_names_every_field() -> None:
    assert _spec().key() == "coulomb/d0/(4,4|4,4)/b1/j/literal"
    assert _spec(am=(10, 2, 3, 1), levers=("a", "b"), n_batches=9).key() == (
        "coulomb/d0/(10,2|3,1)/b9/j/a+b"
    )


def test_the_two_arms_of_a_lever_ab_are_two_keys() -> None:
    # #55: the recycle arm is told apart by its key, not by a note; levers
    # from two layers (the plan's no-sieve, the builder's recycle) compose.
    literal, lever = _spec(n_batches=3), _spec(levers=("recycle",), n_batches=3)
    assert literal.key() == "coulomb/d0/(4,4|4,4)/b3/j/literal"
    assert lever.key() == "coulomb/d0/(4,4|4,4)/b3/j/recycle"
    assert _spec(levers=("no-sieve", "recycle")).key().endswith("/no-sieve+recycle")
    with pytest.raises(ValueError):
        _spec(levers=("recycle", "no-sieve"))  # sorted, or refused


def test_the_tuned_arm_is_its_own_key() -> None:
    # #161: the searched board's plan is keyed apart from literal and from
    # no-sieve, so the store can tell the arms apart; sorted like any lever.
    assert _spec(levers=("tuned",), n_batches=3).key() == (
        "coulomb/d0/(4,4|4,4)/b3/j/tuned")
    keys = {_spec(levers=lv).key() for lv in ((), ("tuned",), ("no-sieve",))}
    assert len(keys) == 3
    assert _spec(levers=("recycle", "tuned")).key().endswith("/recycle+tuned")
    with pytest.raises(ValueError):
        _spec(levers=("tuned", "recycle"))  # sorted, or refused
    with pytest.raises(ValueError):
        _spec(levers=("tuned", "tuned"))  # unique, or refused


def test_the_batch_count_is_part_of_the_key() -> None:
    # A kernel is emitted per class AND per batch, so two batch counts are two
    # artifacts and must be two keys.
    assert _spec().n_batches == 1
    assert _spec(n_batches=3) != _spec(n_batches=6)
    assert len({_spec(n_batches=n).key() for n in (1, 3, 6, 9)}) == 4


def test_total_am_is_the_table_row() -> None:
    # The convention the next PR must honour: a Spec indexes a [JEngine] table
    # row by L = l1 + l2 + l3 + l4, and by nothing else.
    assert _spec(am=(4, 4, 4, 4)).total_am == 16
    assert _spec(am=(3, 0, 2, 2)).total_am == 7
    assert _spec(am=(4, 3, 0, 0)).total_am == _spec(am=(0, 0, 3, 4)).total_am


def test_equal_values_are_one_key() -> None:
    assert _spec() == _spec() and hash(_spec()) == hash(_spec())
    assert len({_spec(), _spec(), _spec(levers=("x",))}) == 2


def test_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _spec().derivative_order = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        {"operator": "coulomb"},
        {"target": "k"},
        {"derivative_order": 1},
        {"derivative_order": False},
        {"am": (4, 4, 4)},
        {"am": (4, 4, 4, -1)},
        {"levers": ("b", "a")},
        {"levers": ("a", "a")},
        {"levers": ("",)},
        {"n_batches": 0},
        {"n_batches": 2},
        {"n_batches": 4},
        {"n_batches": -3},
        {"n_batches": 3.0},
        {"n_batches": True},
        {"am": (4, 4)},  # J takes a quartet, not a pair
        {"am": (4,)},
        {"am": ()},
    ],
)
def test_illegal_values_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        _spec(**bad)


# #150: the pair targets. A pair class is (la, lb); a quartet (l1, l2 | l3, l4).

PAIR_TARGETS = (Target.DQ, Target.JAB)


def test_the_target_enum_is_widened_not_renamed() -> None:
    assert [t.value for t in Target] == ["j", "dq", "jab"]
    assert AM_LENGTH == {Target.J: 4, Target.DQ: 2, Target.JAB: 2}


@pytest.mark.parametrize("target", PAIR_TARGETS)
def test_a_pair_target_takes_two_shells_and_only_two(target: Target) -> None:
    assert _spec(target=target, am=(2, 1)).am == (2, 1)
    for am in ((4, 4, 4, 4), (2, 1, 0), (2,), (2, -1), (2, 1.0)):
        with pytest.raises(ValueError):
            _spec(target=target, am=am)


def test_j_takes_four_shells_and_only_four() -> None:
    assert _spec(am=(1, 0, 1, 0)).am == (1, 0, 1, 0)
    for am in ((1, 0), (1, 0, 1), (1, 0, 1, 0, 0)):
        with pytest.raises(ValueError):
            _spec(am=am)


@pytest.mark.parametrize("target", PAIR_TARGETS)
def test_total_am_is_the_sum_for_a_pair_too(target: Target) -> None:
    assert _spec(target=target, am=(3, 2)).total_am == 5
    assert _spec(target=target, am=(0, 0)).total_am == 0
    assert _spec(am=(3, 2, 0, 0)).total_am == 5  # same sum, different classes
    assert "NO published table row" in Spec.total_am.__doc__


@pytest.mark.parametrize("target", PAIR_TARGETS)
@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_a_pair_target_is_never_batched(target: Target, n: int) -> None:
    # A legal quartet batch count is still refused for a pair: batching
    # partitions a quartet's Hermite targets, and a pair has none.
    assert _spec(n_batches=n).n_batches == n
    with pytest.raises(ValueError, match="never batched"):
        _spec(target=target, am=(2, 1), n_batches=n)
    assert _spec(target=target, am=(2, 1)).n_batches == 1


def test_a_pair_key_is_spelled_without_a_bar() -> None:
    assert _spec(target=Target.DQ, am=(2, 1)).key() == "coulomb/d0/(2,1)/b1/dq/literal"
    assert _spec(target=Target.JAB, am=(0, 3), levers=("x",)).key() == (
        "coulomb/d0/(0,3)/b1/jab/x"
    )


def test_a_pair_key_never_collides_with_a_quartet_key() -> None:
    ls = range(4)
    quartets = {
        _spec(am=(a, b, c, d), levers=lv).key()
        for a in ls for b in ls for c in ls for d in ls for lv in ((), ("x",))
    }
    pairs = {
        _spec(target=t, am=(a, b), levers=lv).key()
        for t in PAIR_TARGETS for a in ls for b in ls for lv in ((), ("x",))
    }
    assert len(quartets) == 4**4 * 2 and len(pairs) == 2 * 4**2 * 2
    assert quartets.isdisjoint(pairs)
    # Not just a different target slot: the class field alone tells them apart.
    q_classes = {k.split("/")[2] for k in quartets}
    p_classes = {k.split("/")[2] for k in pairs}
    assert q_classes.isdisjoint(p_classes)
    assert all("|" in c for c in q_classes) and not any("|" in c for c in p_classes)


def test_a_quartet_key_is_byte_identical_to_before_the_widening() -> None:
    # The join key of 122 measured rows: these strings are pre-#150 literals.
    assert _spec(am=(0, 0, 0, 0)).key() == "coulomb/d0/(0,0|0,0)/b1/j/literal"
    assert _spec(am=(8, 0, 8, 0), n_batches=12, levers=("recycle",)).key() == (
        "coulomb/d0/(8,0|8,0)/b12/j/recycle"
    )
