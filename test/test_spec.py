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
    assert _spec().key() == "coulomb/d0/(4,4|4,4)/j/literal"
    assert _spec(am=(10, 2, 3, 1), levers=("a", "b")).key() == (
        "coulomb/d0/(10,2|3,1)/j/a+b"
    )


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
    ],
)
def test_illegal_values_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        _spec(**bad)
