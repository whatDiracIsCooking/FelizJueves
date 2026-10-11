"""reference/boys.py: the Boys function by quadrature (and, from T_SWITCH, its
large-T asymptote), against closed forms it cannot have been derived from.

Closed forms: erf for F_0, 1/(2m+1) at T = 0, and the Kummer series summed in
50-digit decimal for every (m, T). The downward recurrence a kernel will use
appears here only as a consistency check ON the quadrature -- never as its
implementation, and never as the reference.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import pytest

from felizjueves.reference.boys import (
    M_MAX,
    T_MAX,
    T_SWITCH,
    asymptote,
    boys,
    quadrature,
)

# The grid the module docstring's quadrature measurement was taken on.
GRID_T = (0, 1e-6, 1e-3, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200,
          300, 500)
# Off the grid, so the docstring's figure is not a property of its own grid.
OFF_GRID_T = (0.25, 3.3, 7.7, 42.0, 123.4, 250.0, 420.0, 499.9)
# What the docstring claims for the default, with margin over the measured
# 1.4e-15: the bound, not the measurement.
CONVERGED = 1e-14
PLAN_M = range(0, 17)  # L <= 16 in [JEngine] Tables 1-3

# The asymptote's grids (#122), as the module docstring states them.
HIGH_T = (125, 150, 200, 300, 500, 750, 1000, 1250, 1500, 2000)
HIGH_OFF_GRID_T = (126.5, 333.3, 499.9, 612.0, 987.6, 1290.0, 1777.7, 1999.9)
# Where both branches are measured: T_SWITCH up to the quadrature grid's end.
OVERLAP_T = (125, 130, 150, 175, 200, 250, 300, 350, 400, 450, 500)
# T_SWITCH's criterion: the m = M_MAX truncation under this, on a 5-wide grid.
SWITCH_TRUNCATION = 1e-18
SWITCH_STEP = 5.0


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


def truncation(m: int, T: float) -> float:
    """The asymptote's relative truncation, (Gamma(m+1/2)/(2T^(m+1/2)) - F_m)/F_m,
    in 70-digit decimal: Q(m+1/2, T), with no float64 rounding in it."""
    with localcontext() as ctx:
        ctx.prec = 70
        t = Decimal(T)
        term = Decimal(1) / (2 * m + 1)
        total = term
        k = 0
        while True:
            k += 1
            term = term * 2 * t / (2 * m + 2 * k + 1)
            total += term
            if k > 2 * t and term < total * Decimal("1e-60"):
                break
        exact = (-t).exp() * total
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
        asym = (pi / t).sqrt() / 2
        for j in range(1, m + 1):
            asym = asym * (2 * j - 1) / (2 * t)
        return float((asym - exact) / exact)


def _worst(ms: range, Ts: tuple[float, ...], f=boys, ref=kummer) -> float:
    worst = 0.0
    for m in ms:
        for T in Ts:
            r = ref(m, T)
            worst = max(worst, abs(f(m, float(T)) - r) / r)
    return worst


def test_f0_is_the_error_function() -> None:
    for T in (1e-6, 0.1, 1.0, 10.0, 100.0, T_SWITCH, 1000.0, T_MAX):
        assert boys(0, T) == pytest.approx(
            math.sqrt(math.pi / (4 * T)) * math.erf(math.sqrt(T)), rel=1e-13
        )


def test_fm_at_zero_is_the_closed_form() -> None:
    for m in range(0, 40):
        assert boys(m, 0.0) == pytest.approx(1.0 / (2 * m + 1), rel=1e-14)


def test_kummer_series_agrees_with_the_other_closed_forms() -> None:
    """The series is the reference below, so it is checked against erf first."""
    for T in (1e-6, 0.1, 1.0, 10.0, 100.0, 500.0, T_MAX):
        erf_form = math.sqrt(math.pi / (4 * T)) * math.erf(math.sqrt(T))
        assert kummer(0, T) == pytest.approx(erf_form, rel=1e-14)
    for m in range(0, 41):
        assert kummer(m, 0.0) == 1.0 / (2 * m + 1)


def test_quadrature_is_converged_on_the_documented_grid() -> None:
    """All of GRID_T, past T_SWITCH included: the overlap's quadrature half."""
    assert _worst(range(0, M_MAX + 1), GRID_T, quadrature) < CONVERGED
    assert _worst(PLAN_M, GRID_T, quadrature) < CONVERGED


def test_quadrature_is_converged_off_the_grid() -> None:
    assert _worst(range(0, M_MAX + 1), OFF_GRID_T, quadrature) < CONVERGED


def test_64_nodes_is_not_converged_as_documented() -> None:
    """Keeps the docstring's warning against lowering the default honest."""
    ref = kummer(38, 500.0)
    assert abs(quadrature(38, 500.0, nodes=64) - ref) / ref > 1e-10


def test_switch_point_is_the_measured_one() -> None:
    """T_SWITCH is the first point of the 5-wide grid where the m = M_MAX
    truncation is under SWITCH_TRUNCATION; it is monotone in m and T, so
    every m <= M_MAX at every T >= T_SWITCH is under it too."""
    assert 0.0 < truncation(M_MAX, T_SWITCH) < SWITCH_TRUNCATION
    assert truncation(M_MAX, T_SWITCH - SWITCH_STEP) >= SWITCH_TRUNCATION
    assert all(truncation(m, T_SWITCH) <= truncation(M_MAX, T_SWITCH)
               for m in (0, 8, 16, 32))
    # Below the switch the asymptote is visibly wrong in float64: the switch
    # is not a formality.
    ref = kummer(M_MAX, 100.0)
    assert abs(asymptote(M_MAX, 100.0) - ref) / ref > 1e-12


def test_asymptote_is_converged_on_the_documented_grid() -> None:
    assert _worst(range(0, M_MAX + 1), HIGH_T, asymptote) < CONVERGED
    assert _worst(PLAN_M, HIGH_T, asymptote) < CONVERGED


def test_asymptote_is_converged_off_the_grid() -> None:
    assert _worst(range(0, M_MAX + 1), HIGH_OFF_GRID_T, asymptote) < CONVERGED


def test_branches_agree_across_the_overlap() -> None:
    """Quadrature and asymptote, both measured on [T_SWITCH, 500], agree to
    the sum of their rounding floors: the switch is no step."""
    assert _worst(range(0, M_MAX + 1), OVERLAP_T, quadrature,
                  ref=asymptote) < CONVERGED


def test_boys_routes_by_t_switch() -> None:
    below = math.nextafter(T_SWITCH, 0.0)
    for m in (0, 16, M_MAX):
        assert boys(m, below) == quadrature(m, below)
        assert boys(m, T_SWITCH) == asymptote(m, T_SWITCH)
        assert boys(m, T_MAX) == asymptote(m, T_MAX)
        assert boys(m, 1.0, nodes=64) == quadrature(m, 1.0, nodes=64)


def test_boys_is_converged_over_its_whole_domain() -> None:
    Ts = GRID_T + OFF_GRID_T + HIGH_T + HIGH_OFF_GRID_T
    assert _worst(range(0, M_MAX + 1), Ts) < CONVERGED


def test_quadrature_satisfies_the_downward_recurrence() -> None:
    """F_m = (2T F_{m+1} + exp(-T)) / (2m+1): a check ON the quadrature."""
    for T in GRID_T + OFF_GRID_T + HIGH_T + HIGH_OFF_GRID_T:
        for m in range(0, M_MAX):
            lhs = boys(m, float(T))
            rhs = (2 * T * boys(m + 1, float(T)) + math.exp(-T)) / (2 * m + 1)
            assert lhs == pytest.approx(rhs, rel=1e-13), (m, T)


def test_is_decreasing_in_m_and_in_t() -> None:
    Ts = sorted(set(GRID_T) | set(HIGH_T))
    for T in Ts:
        values = [boys(m, float(T)) for m in range(M_MAX + 1)]
        assert all(a > b for a, b in zip(values, values[1:], strict=False))
    for m in (0, 7, 16, 40):
        values = [boys(m, float(T)) for T in Ts]
        assert all(a > b for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize(
    ("m", "T"),
    [(-1, 1.0), (M_MAX + 1, 1.0), (1.0, 1.0), (0, -1e-3), (0, T_MAX * 1.01),
     (0, math.nextafter(T_MAX, math.inf)),
     (0, math.nan)],
)
def test_refuses_outside_the_measured_domain(m: object, T: float) -> None:
    with pytest.raises(ValueError):
        boys(m, T)  # type: ignore[arg-type]
