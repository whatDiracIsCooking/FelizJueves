"""`ao/shell.py`: the Cartesian order, the normalisation, and the validation (#80).

The norm is checked by Gauss-Hermite quadrature in numpy, never by the closed
form it implements.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from felizjueves.ao.shell import (
    Basis,
    Shell,
    cartesians,
    n_cartesians,
    primitive_norm,
)
from felizjueves.plan.targets import hermite_targets

# Written out by hand: the golden-stable order every later fixture depends on.
ORDER = {
    0: ((0, 0, 0),),
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    2: ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)),
    3: ((3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2),
        (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)),
    4: ((4, 0, 0), (3, 1, 0), (3, 0, 1), (2, 2, 0), (2, 1, 1), (2, 0, 2),
        (1, 3, 0), (1, 2, 1), (1, 1, 2), (1, 0, 3),
        (0, 4, 0), (0, 3, 1), (0, 2, 2), (0, 1, 3), (0, 0, 4)),
}


@pytest.mark.parametrize("l", range(9))
def test_cartesian_count(l: int) -> None:  # noqa: E741
    assert len(cartesians(l)) == (l + 1) * (l + 2) // 2 == n_cartesians(l)
    assert len(set(cartesians(l))) == len(cartesians(l))
    assert all(sum(c) == l and min(c) >= 0 for c in cartesians(l))


def test_cartesian_count_is_not_the_hermite_count() -> None:
    # The hazard: both are "the number of functions", and they differ.
    assert n_cartesians(2) == 6 and len(hermite_targets(2)) == 10


@pytest.mark.parametrize("l", sorted(ORDER))
def test_cartesian_order_literal(l: int) -> None:  # noqa: E741
    assert cartesians(l) == ORDER[l]


@pytest.mark.parametrize("l", range(9))
def test_the_two_orderings_agree(l: int) -> None:  # noqa: E741
    assert cartesians(l) == tuple(r for r in hermite_targets(l) if sum(r) == l)


def _axis_integral(i: int, a: float) -> float:
    """int x^(2i) exp(-a x^2) dx by Gauss-Hermite: x = u / sqrt(a)."""
    u, w = np.polynomial.hermite.hermgauss(20)  # exact to degree 39
    return float(np.sum(w * u ** (2 * i))) / a ** (i + 0.5)


def _self_overlap(c: tuple[int, int, int], alpha: float) -> float:
    n = primitive_norm(sum(c), alpha)
    return n * n * math.prod(_axis_integral(k, 2.0 * alpha) for k in c)


def _odd_df(k: int) -> int:
    return math.prod(range(1, 2 * k, 2))


@pytest.mark.parametrize("alpha", [0.05, 0.8, 1.0, 37.5, 4.0e3])
@pytest.mark.parametrize("l", range(9))
def test_normalised_axis_primitive_has_unit_self_overlap(l: int, alpha: float) -> None:  # noqa: E741
    for c in ((l, 0, 0), (0, l, 0), (0, 0, l)):
        assert _self_overlap(c, alpha) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("l", range(5))
def test_off_axis_components_follow_the_stated_ratio(l: int) -> None:  # noqa: E741
    # The convention is per-l, so (i, j, k) carries (2i-1)!!(2j-1)!!(2k-1)!!/(2l-1)!!.
    for c in cartesians(l):
        want = math.prod(_odd_df(k) for k in c) / _odd_df(l)
        assert _self_overlap(c, 0.7) == pytest.approx(want, rel=1e-12)


def test_shell_stores_tuples_of_floats() -> None:
    s = Shell(2, [0, 0.5, -1], [3.0, 0.4], [0.2, 0.9])
    assert s.centre == (0.0, 0.5, -1.0) and s.exponents == (3.0, 0.4)
    assert s.coefficients == (0.2, 0.9) and s.n_cart == 6
    assert not hasattr(s, "key") and not hasattr(s, "spec")


@pytest.mark.parametrize("args", [
    (1, (0, 0, 0), (1.0, 2.0), (1.0,)),            # mismatched lengths
    (-1, (0, 0, 0), (1.0,), (1.0,)),               # negative l
    (1.0, (0, 0, 0), (1.0,), (1.0,)),              # non-int l
    (True, (0, 0, 0), (1.0,), (1.0,)),             # bool is not an l
    (0, (0, math.nan, 0), (1.0,), (1.0,)),         # non-finite centre
    (0, (0, 0, math.inf), (1.0,), (1.0,)),         # non-finite centre
    (0, (0, 0), (1.0,), (1.0,)),                   # centre not 3-vector
    (0, (0, 0, 0), (0.0,), (1.0,)),                # exponent not positive
    (0, (0, 0, 0), (-1.0,), (1.0,)),               # exponent not positive
    (0, (0, 0, 0), (math.inf,), (1.0,)),           # non-finite exponent
    (0, (0, 0, 0), (1.0,), (math.nan,)),           # non-finite coefficient
    (0, (0, 0, 0), (), ()),                        # no primitives
    (0, "abc", (1.0,), (1.0,)),                    # centre not numbers
])
def test_malformed_shell_raises(args: tuple) -> None:
    with pytest.raises(ValueError):
        Shell(*args)


@pytest.mark.parametrize("l, alpha", [(-1, 1.0), (0, 0.0), (0, -2.0),
                                      (0, math.inf), (0, math.nan)])
def test_primitive_norm_rejects(l: int, alpha: float) -> None:  # noqa: E741
    with pytest.raises(ValueError):
        primitive_norm(l, alpha)


def test_cartesians_rejects_negative_l() -> None:
    with pytest.raises(ValueError):
        cartesians(-1)


def test_basis_offsets_and_n_ao() -> None:
    s = Shell(0, (0, 0, 0), (1.0,), (1.0,))
    p = Shell(1, (0, 0, 1), (1.0,), (1.0,))
    d = Shell(2, (1, 0, 0), (1.0,), (1.0,))
    b = Basis((s, p, d, s))
    assert b.offsets == (0, 1, 4, 10) and b.n_ao == 11
    assert Basis(()).n_ao == 0
    with pytest.raises(ValueError):
        Basis((s, "p"))
