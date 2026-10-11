"""`Spec`, the join key: frozen, canonical, and narrow on purpose."""

from __future__ import annotations

import dataclasses

import pytest

from felizjueves.spec import Operator, Spec, Target


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
    ],
)
def test_illegal_values_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        _spec(**bad)
