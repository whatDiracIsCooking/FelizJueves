"""reference/boys.py: the Boys function by quadrature, against closed forms it
cannot have been derived from.

Closed forms: erf for F_0, 1/(2m+1) at T = 0, and the Kummer series summed in
50-digit decimal for every (m, T). The downward recurrence a kernel will use
appears here only as a consistency check ON the quadrature -- never as its
implementation, and never as the reference.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import pytest

from felizjueves.reference.boys import M_MAX, NODES, T_MAX, boys

# The grid the module docstring's measurement was taken on.
GRID_T = (0, 1e-6, 1e-3, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200,
          300, 500)
# Off the grid, so the docstring's figure is not a property of its own grid.
OFF_GRID_T = (0.25, 3.3, 7.7, 42.0, 123.4, 250.0, 420.0, 499.9)
# What the docstring claims for the default, with margin over the measured
# 1.4e-15: the bound, not the measurement.
CONVERGED = 1e-14
PLAN_M = range(0, 17)  # L <= 16 in [JEngine] Tables 1-3


def kummer(m: int, T: float) -> float:
    """F_m(T) = exp(-T) sum_k (2T)^k / ((2m+1)(2m+3)...(2m+2k+1)), 50 digits.

    All terms positive, so no cancellation; summed until a term falls below
    1e-40 of the total, past the peak at k ~ T.
    """
    with localcontext() as ctx:
        ctx.prec = 50
        t = Decimal(T)
        term = Decimal(1) / (2 * m + 1)
        total = term
        k = 0
        while True:
            k += 1
            term = term * 2 * t / (2 * m + 2 * k + 1)
            total += term
            if k > 2 * t and term < total * Decimal("1e-40"):
                break
        return float((-t).exp() * total)


def _worst(ms: range, Ts: tuple[float, ...], nodes: int = NODES) -> float:
    worst = 0.0
    for m in ms:
        for T in Ts:
            ref = kummer(m, T)
            worst = max(worst, abs(boys(m, float(T), nodes=nodes) - ref) / ref)
    return worst


def test_f0_is_the_error_function() -> None:
    for T in (1e-6, 0.1, 1.0, 10.0, 100.0):
        assert boys(0, T) == pytest.approx(
            math.sqrt(math.pi / (4 * T)) * math.erf(math.sqrt(T)), rel=1e-13
        )


def test_fm_at_zero_is_the_closed_form() -> None:
    for m in range(0, 40):
        assert boys(m, 0.0) == pytest.approx(1.0 / (2 * m + 1), rel=1e-14)


def test_kummer_series_agrees_with_the_other_closed_forms() -> None:
    """The series is the reference below, so it is checked against erf first."""
    for T in (1e-6, 0.1, 1.0, 10.0, 100.0, 500.0):
        erf_form = math.sqrt(math.pi / (4 * T)) * math.erf(math.sqrt(T))
        assert kummer(0, T) == pytest.approx(erf_form, rel=1e-14)
    for m in range(0, 41):
        assert kummer(m, 0.0) == 1.0 / (2 * m + 1)


def test_default_is_converged_on_the_documented_grid() -> None:
    assert _worst(range(0, M_MAX + 1), GRID_T) < CONVERGED
    assert _worst(PLAN_M, GRID_T) < CONVERGED


def test_default_is_converged_off_the_grid() -> None:
    assert _worst(range(0, M_MAX + 1), OFF_GRID_T) < CONVERGED


def test_64_nodes_is_not_converged_as_documented() -> None:
    """Keeps the docstring's warning against lowering the default honest."""
    ref = kummer(38, 500.0)
    assert abs(boys(38, 500.0, nodes=64) - ref) / ref > 1e-10


def test_quadrature_satisfies_the_downward_recurrence() -> None:
    """F_m = (2T F_{m+1} + exp(-T)) / (2m+1): a check ON the quadrature."""
    for T in GRID_T + OFF_GRID_T:
        for m in range(0, M_MAX):
            lhs = boys(m, float(T))
            rhs = (2 * T * boys(m + 1, float(T)) + math.exp(-T)) / (2 * m + 1)
            assert lhs == pytest.approx(rhs, rel=1e-13), (m, T)


def test_is_decreasing_in_m_and_in_t() -> None:
    for T in GRID_T:
        values = [boys(m, float(T)) for m in range(M_MAX + 1)]
        assert all(a > b for a, b in zip(values, values[1:], strict=False))
    for m in (0, 7, 16, 40):
        values = [boys(m, float(T)) for T in GRID_T]
        assert all(a > b for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize(
    ("m", "T"),
    [(-1, 1.0), (M_MAX + 1, 1.0), (1.0, 1.0), (0, -1e-3), (0, T_MAX * 1.01),
     (0, math.nan)],
)
def test_refuses_outside_the_measured_domain(m: object, T: float) -> None:
    with pytest.raises(ValueError):
        boys(m, T)  # type: ignore[arg-type]
