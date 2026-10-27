"""The DF kernels' low-stack forms (#606), `kernel_eri3` and `kernel_metric`,
interpreted.

THE GATE. `eri3_lowstack` / `metric_lowstack` pick the low-stack form from
`L = 10`, or where the unrolled form's two `E` blocks are larger than
(dd|g)'s (`E_DOUBLES_MAX`, the frame #606 measured at 13,856 B). On the
ladder FreeBreakfast emits (eri3 `la <= lb <= 3`, `lP <= 6`; the metric
`lP <= lQ <= 6`) that is the 24 and 8 classes pinned here, and none at
`L <= 4`, so every golden is untouched.

THE WITNESSES, test_kernel_eri3.py's and test_kernel_metric.py's, imported:
the host within `16 eps` of the tower twin's scale per element, the
launched cells' elements only, the mirror exact. And the unrolled kernel:
`|low - unrolled| <= 1e-14 max |unrolled|` on the class (#606's "today's
kernels at 1e-14"). The low-stack product `E_x (E_y (E_z U))` is in another
association than the block's `(E_x E_y) E_z`, so neither is bitwise beyond
s and p (pinned below).

SCOPE: every L <= 4 class with the form FORCED (s and p fast, the rest
`slow`, as the two modules), and a few gated classes by default (`slow`:
they are above `FAST_L_MAX`), on small bases with f AO shells and h and i
aux shells. The text of every gated class through (ff|i) and (i|i) is
built and bounded (`slow`) and compiled by none of these tests; nothing launches:
no libcint fixture reaches an h or i aux shell yet (#606's follow-up).
"""

from __future__ import annotations

import numpy as np
import pytest
import test_kernel_eri3 as te
import test_kernel_metric as tm
from tiers import FAST_L_MAX

from felizjueves.ao.shell import Basis, Shell
from felizjueves.emit import cuda
from felizjueves.eri.density_fit import kernel_eri3 as ke
from felizjueves.eri.density_fit import kernel_metric as km
from felizjueves.eri.density_fit.eri3_drive import eri3_tensor_kernel
from felizjueves.eri.density_fit.metric_drive import metric_matrix_kernel
from felizjueves.ir import AtomicAdd, Decl, Space, walk
from felizjueves.performance.device import SM_86
from felizjueves.reference.boys import boys

REL = 1e-14  # x max |unrolled| of the class, against the unrolled kernel
ERI3_LADDER = [(a, b, p) for a in range(4) for b in range(a, 4)
               for p in range(7)]
METRIC_LADDER = [(p, q) for p in range(7) for q in range(p, 7)]
ERI3_GATED = {(0, 0, 6), (0, 1, 6), (0, 2, 6), (0, 3, 6), (1, 1, 6),
              (1, 2, 6), (1, 3, 5), (1, 3, 6), (2, 2, 5), (2, 2, 6),
              (2, 3, 0), (2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4),
              (2, 3, 5), (2, 3, 6), (3, 3, 0), (3, 3, 1), (3, 3, 2),
              (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 3, 6)}
METRIC_GATED = {(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 5), (5, 6),
                (6, 6)}
# Every low-stack class declares at most this much local-array space per
# thread (eri3_336's R, U and tables: 8,280 B), against #606's ~14 KB.
ARRAY_BYTES_MAX = 8_800
SOURCE_BYTES_MAX = 128 * 1024  # (ff|g) unrolled is 1.56 MB

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
# f shells on two centres (so (ff|.) has diagonal and off-diagonal pairs), a
# d and a p shell (swapped mixed classes), contracted and single primitives
HIGH = Basis((
    Shell(3, A, (0.8,), (1.0,)),
    Shell(3, B, (1.1, 0.35), (0.6, 0.5)),
    Shell(2, A, (0.9,), (1.0,)),
    Shell(1, B, (0.7,), (1.0,)),
    Shell(0, A, (1.5,), (1.0,)),
))
HIGH_AUX = Basis((
    Shell(6, C, (1.6, 0.6), (0.7, 0.4)),
    Shell(5, B, (1.3,), (1.0,)),
    Shell(1, A, (0.9,), (1.0,)),
    Shell(0, C, (1.2,), (1.0,)),
    Shell(6, A, (2.0,), (1.0,)),
))


# --- the gate ------------------------------------------------------------------


def test_the_reference_is_dd_g_s_e_blocks_in_both_modules():
    assert ke.eri3_e_doubles((2, 2, 4)) == ke.E_DOUBLES_MAX == 36 * 35 + 15 * 35
    assert km.E_DOUBLES_MAX == ke.E_DOUBLES_MAX  # restated, not imported
    assert ke.LOWSTACK_MIN_L == km.LOWSTACK_MIN_L == 10
    assert not ke.eri3_lowstack((2, 2, 4))  # (dd|g) itself stays unrolled


def test_the_gate_on_the_ladder_is_pinned():
    assert {c for c in ERI3_LADDER if ke.eri3_lowstack(c)} == ERI3_GATED
    assert {c for c in METRIC_LADDER if km.metric_lowstack(c)} == METRIC_GATED
    # every (df|.) and (ff|.), and every aux-i row of the eri3 ladder
    assert {(2, 3, p) for p in range(7)} | {(3, 3, p) for p in range(7)} \
        <= ERI3_GATED
    assert {(a, b, 6) for a in range(4) for b in range(a, 4)} <= ERI3_GATED
    assert {(p, 6) for p in range(7)} <= METRIC_GATED


def test_no_class_at_or_below_the_fast_cut_is_gated():
    """So the L <= 4 goldens (test_emit_eri3.py, test_emit_metric.py) are
    the unrolled text, unchanged."""
    assert not any(ke.eri3_lowstack(c) for c in te.CLASSES)
    assert not any(km.metric_lowstack(c) for c in tm.CLASSES)
    assert max(sum(c) for c in te.CLASSES + tm.CLASSES) == FAST_L_MAX


# --- the forced form against the host and the unrolled kernel -----------------


def _against_unrolled(low, unrolled, label):
    s = float(np.max(np.abs(unrolled)))
    err = float(np.max(np.abs(low - unrolled)))
    print(f"\n{label}: max |low - unrolled| = {err / s:.2e} of max")
    assert err <= REL * s, label
    assert np.array_equal(low == 0.0, unrolled == 0.0)  # the same elements


def _eri3(basis, aux, cls, lowstack):
    return eri3_tensor_kernel(basis, aux, boys=boys, classes={cls},
                              threads=5, lowstack=lowstack)


def _metric(aux, cls, lowstack):
    return metric_matrix_kernel(aux, boys=boys, classes={cls}, threads=5,
                                lowstack=lowstack)


def _check_eri3(basis, aux, cls):
    low = _eri3(basis, aux, cls, True)
    assert low.ran == (cls,)
    want, scale, owned = te._host(basis, aux, {cls})
    te._check_host(low.T, want, scale, owned, f"{cls}, low-stack")
    _against_unrolled(low.T, _eri3(basis, aux, cls, False).T, f"{cls}")
    return low.T


def _check_metric(aux, cls):
    low = _metric(aux, cls, True)
    assert low.ran == (cls,)
    want, scale, owned = tm._host(aux, {cls})
    tm._check_host(low.M, want, scale, owned, f"{cls}, low-stack")
    _against_unrolled(low.M, _metric(aux, cls, False).M, f"{cls}")
    return low.M


@pytest.mark.parametrize("cls", [te._param(c) for c in te.CLASSES])
def test_the_forced_eri3_form_agrees_with_the_host(cls):
    _check_eri3(te.SYNTHETIC, te.SYNTHETIC_AUX, cls)


@pytest.mark.parametrize("cls", [tm._param(c) for c in tm.CLASSES])
def test_the_forced_metric_form_agrees_with_the_host(cls):
    _check_metric(tm.SYNTHETIC, cls)


def test_s_and_p_are_bitwise_and_d_is_not():
    """With no `E` product of three non-unit axes, the two forms do the same
    operations: (ss|s) and every metric class through (p|p) are bitwise;
    from d on the association differs (module docstring)."""
    assert np.array_equal(_eri3(te.SYNTHETIC, te.SYNTHETIC_AUX, (0, 0, 0),
                                True).T,
                          _eri3(te.SYNTHETIC, te.SYNTHETIC_AUX, (0, 0, 0),
                                False).T)
    for cls in ((0, 0), (0, 1), (1, 1)):
        assert np.array_equal(_metric(tm.SYNTHETIC, cls, True).M,
                              _metric(tm.SYNTHETIC, cls, False).M), cls
    assert not np.array_equal(_metric(tm.SYNTHETIC, (1, 2), True).M,
                              _metric(tm.SYNTHETIC, (1, 2), False).M)


def test_the_low_stack_form_zeroes_its_own_elements():
    """It accumulates into its slots, so it zeroes them first: a launch over
    a non-zero tensor leaves every launched element the integral and every
    other element as it was."""
    from felizjueves.eri.density_fit.eri3 import eri3_classes
    from felizjueves.eri.density_fit.eri3_drive import launch_class
    cls = next(c for c in eri3_classes(te.SYNTHETIC, te.SYNTHETIC_AUX)
               if (c.la, c.lb, c.lP) == (1, 1, 1))
    n, naux = te.SYNTHETIC.n_ao, te.SYNTHETIC_AUX.n_ao
    junk = np.full((n, n, naux), 7.25)
    got = launch_class(cls, junk, boys=boys, threads=5, lowstack=True)
    clean = launch_class(cls, np.zeros((n, n, naux)), boys=boys, threads=5,
                         lowstack=True)
    mask = clean != 0.0
    assert np.array_equal(got[mask], clean[mask])
    assert np.all(got[~mask & (got != 0.0)] == 7.25)


# --- gated classes by default (above the fast cut) ------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("cls", [(3, 3, 0), (2, 3, 1), (0, 0, 6), (1, 1, 6),
                                 (0, 1, 6), (2, 3, 6), (3, 3, 5),
                                 (3, 3, 6)],  # (ff|i): ~90 s interpreted
                         ids=ke.eri3_name)
def test_a_gated_eri3_class_by_default(cls):
    assert ke.eri3_lowstack(cls)
    T = _check_eri3(HIGH, HIGH_AUX, cls)
    assert np.any(T)


@pytest.mark.slow
@pytest.mark.parametrize("cls", [(0, 6), (1, 6), (5, 5), (5, 6), (6, 6)],
                         ids=str)
def test_a_gated_metric_class_by_default(cls):
    assert km.metric_lowstack(cls)
    M = _check_metric(HIGH_AUX, cls)
    assert np.any(M)


# --- the text of every gated class: no accumulators, a bounded frame -----------


def _array_bytes(kernel) -> int:
    total = 0
    for st in walk(kernel.body):
        if isinstance(st, Decl) and st.shape:
            assert st.space is Space.LOCAL, (kernel.name, st.name)
            n = 1
            for d in st.shape:
                n *= d.value
            total += 8 * n
    return total


def _bounded(kernel, blocks):
    names = {st.name for st in walk(kernel.body) if isinstance(st, Decl)}
    assert not any(n.startswith("acc") for n in names), kernel.name
    assert not names & set(blocks), kernel.name  # no E block
    assert not any(isinstance(st, AtomicAdd) for st in walk(kernel.body))
    arrays = _array_bytes(kernel)
    text = cuda.emit(kernel, SM_86)
    print(f"\n{kernel.name}: {arrays:,} B of local arrays, "
          f"{len(text) / 1024:.0f} KB of source")
    assert arrays <= ARRAY_BYTES_MAX, kernel.name
    assert len(text) <= SOURCE_BYTES_MAX, kernel.name


@pytest.mark.slow  # builds classes above FAST_L_MAX
@pytest.mark.parametrize("cls", sorted(ERI3_GATED), ids=ke.eri3_name)
def test_every_gated_eri3_text_is_bounded(cls):
    _bounded(ke.build_eri3_kernel(cls), ("Ebra", "Eket"))


@pytest.mark.slow
@pytest.mark.parametrize("cls", sorted(METRIC_GATED), ids=str)
def test_every_gated_metric_text_is_bounded(cls):
    _bounded(km.build_metric_kernel(cls), ("Ebra", "Eket"))


def test_the_forms_share_name_parameters_and_launch():
    for cls in ((1, 1, 1), (0, 2, 1)):
        a = ke.build_eri3_kernel(cls, lowstack=True)
        b = ke.build_eri3_kernel(cls, lowstack=False)
        assert (a.name, a.params) == (b.name, b.params)
    for cls in ((1, 2), (2, 2)):
        a = km.build_metric_kernel(cls, lowstack=True)
        b = km.build_metric_kernel(cls, lowstack=False)
        assert (a.name, a.params) == (b.name, b.params)
