"""`rys.roots`: the fp64 Rys rule against a 40-digit route, and the device
routine's text against the host rule.

- THE RULE. `roots_weights(n, T)` against `rys.exact.roots_weights_exact`
  (moments in `decimal`, Gautschi's Chebyshev algorithm, Newton), which
  shares nothing with it but the measure: every `n` in 1..`N_MAX`, `T` from
  0 through `1e-300` to `1e6` and densely across `T_SWITCH`. Nodes and
  weights within `RULE_TOL` relative (measured worst 3.4e-15), and the
  moment identity `SUM w x^m = F_m(T)` for `m <= 2n - 1`.
- THE DEVICE TEXT. `device_text()`, the `fj_rys<N>` every Rys unit carries,
  compiled as HOST C++ (g++, `-ffp-contract=off`, the device qualifiers
  defined away) and run on the same grid: BITWISE the host rule here, since
  the text is the host rule's operation order and constants. That is the
  routine's arithmetic, not a card's: a card's `exp` and `sqrt` are its own,
  and the kernel tier is where a card number is made. Skipped without g++.
- `ir.Rys`: its checks, the writers' spelling, and `emit/gpu.py`'s refusal.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from decimal import Decimal

import numpy as np
import pytest

from felizjueves.emit import cuda, gpu, hip
from felizjueves.emit._c import EmitError
from felizjueves.eri.fock.kernel_frys import build_frys_kernel
from felizjueves.ir import Decl, Kernel, Lit, Param, Rys, Scalar, Space, Var
from felizjueves.performance.device import by_name
from felizjueves.rys import roots as R
from felizjueves.rys.exact import boys_exact, roots_weights_exact

RULE_TOL = 1e-14
GRID = sorted({0.0, 1e-300, 1e-10, 1e-3, 0.1, 0.5, 1.0, 2.0, 3.3, 5.0, 7.7,
               10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 79.0,
               math.nextafter(R.T_SWITCH, 0.0), R.T_SWITCH,
               math.nextafter(R.T_SWITCH, math.inf), 80.5, 100.0, 150.0,
               1e3, 1e4, 1e6})
NS = range(1, R.N_MAX + 1)


@pytest.mark.parametrize("n", NS)
def test_the_rule_meets_the_40_digit_route(n) -> None:
    worst = 0.0
    for T in GRID:
        xe, we = roots_weights_exact(n, T)
        x, w = R.roots_weights(n, T)
        assert list(x) == sorted(x)
        for got, want in ((x, xe), (w, we)):
            for a, b in zip(got, want, strict=True):
                worst = max(worst, abs(a - float(b)) / float(b))
    assert worst <= RULE_TOL, worst


@pytest.mark.parametrize("n", NS)
def test_the_moments_are_the_boys_function(n) -> None:
    for T in (0.0, 0.7, 12.0, 79.0, 80.0, 400.0):
        x, w = R.roots_weights(n, T)
        F = boys_exact(2 * n - 1, Decimal(T))
        for m in range(2 * n):
            got = sum(wi * xi ** m for xi, wi in zip(x, w, strict=True))
            assert abs(got - float(F[m])) <= 1e-14 * float(F[m]), (T, m)


def test_the_tail_the_switch_drops_is_below_rounding_at_n_max() -> None:
    """The Laguerre branch just above the switch, against the exact rule:
    its dropped tail `O(e^-T T^(2N - 3/2))` is invisible at `N_MAX`, and
    would not be at #468's T = 50."""
    n = R.N_MAX
    xe, we = roots_weights_exact(n, R.T_SWITCH)
    _, v = R.laguerre(n)
    w50 = [vi / (2.0 * math.sqrt(50.0)) for vi in v]
    _, we50 = roots_weights_exact(n, 50.0)
    assert max(abs(a - float(b)) / float(b) for a, b in
               zip(R.roots_weights(n, R.T_SWITCH)[1], we, strict=True)) < 1e-14
    assert max(abs(a - float(b)) / float(b) for a, b in
               zip(w50, we50, strict=True)) > 1e-11  # measured 5.6e-11


def test_refusals_and_n_roots() -> None:
    assert [R.n_roots(L) for L in range(9)] == [1, 1, 2, 2, 3, 3, 4, 4, 5]
    for bad in (0, R.N_MAX + 1):
        with pytest.raises(ValueError):
            R.roots_weights(bad, 1.0)
    for bad in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            R.roots_weights(2, bad)


_HARNESS = """\
#include <cmath>
#include <cstdio>
using std::exp;
using std::sqrt;
#define __device__
#define __forceinline__ inline
#define __constant__ static const
{text}
template <int N> void run(double T) {{
    double x[N], w[N];
    fj_rys<N>(T, x, w);
    for (int i = 0; i < N; ++i) printf("%a %a\\n", x[i], w[i]);
}}
int main() {{
    int n;
    double T;
    while (scanf("%d %lf", &n, &T) == 2) {{
        switch (n) {{
        case 1: run<1>(T); break;
        case 2: run<2>(T); break;
        case 3: run<3>(T); break;
        case 4: run<4>(T); break;
        case 5: run<5>(T); break;
        }}
    }}
}}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="no g++")
def test_the_device_text_is_the_host_rule_bitwise(tmp_path) -> None:
    assert R.N_MAX == 5  # the harness's switch
    src = tmp_path / "rys.cpp"
    src.write_text(_HARNESS.format(text=R.device_text()))
    exe = tmp_path / "rys"
    subprocess.run(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-ffp-contract=off", "-o", str(exe), str(src)],
                   check=True, capture_output=True)
    cases = [(n, T) for n in NS for T in GRID]
    out = subprocess.run([str(exe)], input="".join(
        f"{n} {T!r}\n" for n, T in cases), capture_output=True, text=True,
        check=True).stdout.split()
    got = iter(float.fromhex(v) for v in out)
    for n, T in cases:
        x, w = R.roots_weights(n, T)
        for i in range(n):
            assert (next(got), next(got)) == (x[i], w[i]), (n, T, i)


# --- ir.Rys ----------------------------------------------------------------------


def _rys_kernel(nodes_extent=3, t_type=Scalar.F64) -> Kernel:
    return Kernel("k", (Param("T", t_type),), (
        Decl("x", Scalar.F64, (Lit(nodes_extent),), Space.LOCAL),
        Decl("w", Scalar.F64, (Lit(3),), Space.LOCAL),
        Rys("x", "w", 3, Var("T")),
    ))


def test_ir_rys_checks_its_arrays() -> None:
    _rys_kernel()
    with pytest.raises(ValueError, match="extent 3"):
        _rys_kernel(nodes_extent=4)
    with pytest.raises(ValueError):
        Rys("x", "x", 3, Var("T"))
    with pytest.raises(ValueError):
        Rys("x", "w", 0, Var("T"))


def test_the_writers_spell_the_rule_and_gpu_refuses_it() -> None:
    k = build_frys_kernel((1, 1, 1, 1))
    for mod, arch in ((cuda, "sm_86"), (hip, "gfx1200")):
        text = mod.emit(k, by_name(arch))
        assert "fj_rys<3>(T, rys_x, rys_w);" in text
        assert "__forceinline__ void fj_rys(double T" in text
        assert "fj_boys" not in text
    with pytest.raises(EmitError, match="Rys"):
        gpu.emit(k, by_name("sm_86"))


def test_the_tables_are_the_host_constants() -> None:
    text = R.device_text()
    y, u = R.legendre_half()
    assert f"FJ_RYS_Y[{len(y)}]" in text and len(y) == R.M_LEGENDRE // 2
    assert all(repr(v) in text for v in (*y, *u))
    assert np.all(np.diff(y) < 0)  # descending t: the Newton guesses' order
