"""`kernel_hgp`: the DF metric and three-center kernels on the HGP engine,
interpreted.

THE WITNESSES are the MMD path's: the hosts `eri.density_fit.metric` and
`eri.density_fit.eri3` (`class_block` with `reference.pq.pq` injected,
placed by `place_staged`) and, through the tracked sets (#411, #442),
libcint's `int2c2e_cart` and `int3c2e_cart`; and, as a cross-engine pin,
the MMD kernels themselves (`kernel_metric`, `kernel_eri3`), interpreted on
the same staging. The HGP kernels run through the same adapters
(`metric_drive`, `eri3_drive`, `engine=Engine.HGP`) with
`reference.boys.boys` injected: the same parameters, launch and output, so
nothing else changes.

ROUND-OFF, NOT BITWISE (`kernel_hgp`'s module docstring): another recurrence
associates differently. THE BOUNDS, per element:

- against the host, `|X - host| <= TOL_ULPS * eps * scale`, `TOL_ULPS =
  16`, `scale` the MMD tests' absolute-value twin over the tower twin
  (`test_kernel_metric.pq_tower`, `test_kernel_eri3._scale`) -- the MMD
  sum's yardstick, which holds the HGP kernel too here: measured worst at
  writing 2.89 eps of it (metric, (p|p)) and 3.20 (three-center, (pp|p)),
  both on the synthetic bases. An element whose scale is 0 must be exactly 0.
- against the MMD kernel, `32 eps` of the same scale: two roundings.
- against libcint, `TOL = 1e-12` of the fixture's `max |.|`, the MMD tests'
  bound.

SCOPE: s and p shells only (`kernel_hgp.HGP_DF_L_MAX = 1`), all fast: the
metric's (s|s), (s|p), (p|p) and the three-center's six classes `la <= lb
<= 1`, `lP <= 1`. Nothing compiles, nothing launches.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_kernel_eri3 as te
import test_kernel_metric as tm

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit import kernel_eri3, kernel_metric
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.eri.density_fit.eri3_drive import eri3_kernel, eri3_tensor_kernel
from felizjueves.eri.density_fit.kernel_hgp import (
    HGP_DF_L_MAX,
    build_heri3_kernel,
    build_hmetric_kernel,
    eri3_class,
    heri3_name,
    heri3_spec,
    hmetric_name,
    hmetric_spec,
    metric_class,
)
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.eri.density_fit.metric_drive import (
    metric_kernel,
    metric_matrix_kernel,
)
from felizjueves.hgp import vrr
from felizjueves.hgp.recurrence import hrr_name
from felizjueves.ir import Assign, AtomicAdd, Decl, Factor, Index, Weighted, walk
from felizjueves.reference.boys import boys
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL_ULPS = 16  # x eps x scale, against the host
CROSS_ULPS = 32  # x eps x scale, against the MMD kernel
TOL = 1e-12  # x the fixture's max |.|, against libcint
HGP = Engine.HGP

METRIC = [(0, 0), (0, 1), (1, 1)]
ERI3 = [(a, b, p) for a in range(2) for b in range(a, 2) for p in range(2)]


def _sp(basis: Basis) -> Basis:
    return Basis(tuple(s for s in basis.shells if s.l <= HGP_DF_L_MAX))


# The MMD tests' synthetic bases cut to their s and p shells: two shells of
# each l (diagonal and off-diagonal pairs), the higher l first on a centre
# (swapped pairs), one-centre pairs and triples, contracted and not.
AUX = _sp(tm.SYNTHETIC)
BASIS, BASIS_AUX = _sp(te.SYNTHETIC), _sp(te.SYNTHETIC_AUX)


def _worst(err, scale, owned) -> float:
    pos = owned & (scale > 0)
    return float(np.max(err[pos] / (EPS * scale[pos]))) if pos.any() else 0.0


def _check(X, want, scale, owned, mmd, label, mirror):
    err = np.abs(X - want)
    print(f"\n{label}: worst {_worst(err, scale, owned):.2f} eps scale "
          f"(host), {_worst(np.abs(X - mmd), scale, owned):.2f} (MMD kernel)")
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), label
    assert np.all(np.abs(X - mmd)[owned]
                  <= CROSS_ULPS * EPS * scale[owned]), label
    assert np.all(X[~owned] == 0.0)  # exactly the launched elements
    assert np.array_equal(X, mirror(X))  # the scatter mirrors one value


def test_the_bases_reach_every_class_swapped_and_diagonal():
    by = {(c.lP, c.lQ): c for c in metric_classes(AUX)}
    assert sorted(by) == METRIC
    for cls, c in by.items():
        if cls[0] == cls[1]:
            assert {p == q for p, q in c.pairs} == {True, False}, cls
        else:
            assert any(c.swapped), cls
    by3 = {(c.la, c.lb, c.lP): c for c in eri3_classes(BASIS, BASIS_AUX)}
    assert sorted(by3) == ERI3
    for cls, c in by3.items():
        pairs = {(a, b) for a, b, _ in c.cells}
        if cls[0] == cls[1]:
            assert {a == b for a, b in pairs} == {True, False}, cls
        else:
            assert any(c.swapped), cls


@pytest.mark.parametrize("cls", METRIC, ids=lambda c: f"{'sp'[c[0]]}|{'sp'[c[1]]}")
def test_the_metric_class_agrees_with_the_host_and_the_mmd_kernel(cls):
    run = metric_matrix_kernel(AUX, boys=boys, classes={cls}, threads=3,
                               engine=HGP)
    assert run.ran == (cls,)
    mmd = metric_matrix_kernel(AUX, boys=boys, classes={cls}, threads=3).M
    want, scale, owned = tm._host(AUX, {cls})
    _check(run.M, want, scale, owned, mmd, f"hmetric {cls}", np.transpose)


@pytest.mark.parametrize("cls", ERI3, ids=lambda c: "".join(map(str, c)))
def test_the_eri3_class_agrees_with_the_host_and_the_mmd_kernel(cls):
    run = eri3_tensor_kernel(BASIS, BASIS_AUX, boys=boys, classes={cls},
                             threads=3, engine=HGP)
    assert run.ran == (cls,)
    mmd = eri3_tensor_kernel(BASIS, BASIS_AUX, boys=boys, classes={cls},
                             threads=3).T
    want, scale, owned = te._host(BASIS, BASIS_AUX, {cls})
    _check(run.T, want, scale, owned, mmd, f"heri3 {cls}",
           lambda t: t.transpose(1, 0, 2))


def test_tight_exponents_past_the_narrow_boys_range():
    """`T` past 709 (the MMD tests' `TIGHT` sets): the one `fj_boys` holds."""
    run = metric_matrix_kernel(tm.TIGHT, boys=boys, engine=HGP)
    want, scale, owned = tm._host(tm.TIGHT, set(run.ran))
    mmd = metric_matrix_kernel(tm.TIGHT, boys=boys).M
    assert owned.all()
    _check(run.M, want, scale, owned, mmd, "hmetric tight", np.transpose)
    run3 = eri3_tensor_kernel(te.TIGHT, te.TIGHT_AUX, boys=boys, engine=HGP)
    want, scale, owned = te._host(te.TIGHT, te.TIGHT_AUX, set(run3.ran))
    mmd = eri3_tensor_kernel(te.TIGHT, te.TIGHT_AUX, boys=boys).T
    _check(run3.T, want, scale, owned, mmd, "heri3 tight",
           lambda t: t.transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# The fixtures: libcint
# ---------------------------------------------------------------------------


def test_the_sp_metric_of_the_jkfit_set_is_libcints():
    aux = tm._aux()
    keep = [k for k, s in enumerate(aux.shells) if s.l <= HGP_DF_L_MAX]
    sub = Basis(tuple(aux.shells[k] for k in keep))
    rows = [aux.offsets[k] + a for k in keep
            for a in range(aux.shells[k].n_cart)]
    run = metric_matrix_kernel(sub, boys=boys, engine=HGP)
    assert set(run.ran) == set(METRIC)
    ref = tm._ms().metric[np.ix_(rows, rows)]
    worst = float(np.max(np.abs(run.M - ref))) / float(
        np.max(np.abs(tm._ms().metric)))
    print(f"\nhmetric l <= 1: max |M - libcint| = {worst:.2e} of max|M|")
    assert worst <= TOL, worst
    assert np.linalg.eigvalsh(run.M).min() > 0


def test_the_sp_three_center_tensor_of_h2o_sto3g_is_libcints():
    aux = te._aux()
    keep = [k for k, s in enumerate(aux.shells) if s.l <= HGP_DF_L_MAX]
    sub = Basis(tuple(aux.shells[k] for k in keep))
    cols = [aux.offsets[k] + q for k in keep
            for q in range(aux.shells[k].n_cart)]
    run = eri3_tensor_kernel(te._basis(), sub, boys=boys, engine=HGP)
    assert set(run.ran) == set(ERI3)
    ref = te._ts().tensor[:, :, cols]
    worst = float(np.max(np.abs(run.T - ref))) / float(
        np.max(np.abs(te._ts().tensor)))
    print(f"\nheri3 l <= 1: max |T - libcint| = {worst:.2e} of max|(mn|P)|")
    assert worst <= TOL, worst


# ---------------------------------------------------------------------------
# What the kernels are built from
# ---------------------------------------------------------------------------


def _factors(kernel) -> set[Factor]:
    return {t.g for s in walk(kernel.body) if isinstance(s, Decl)
            and isinstance(s.init, Weighted) for t in s.init.terms}


def _hrr(kernel) -> set[str]:
    return {s.name for s in walk(kernel.body) if isinstance(s, Decl)
            and s.name.startswith("H")}


@pytest.mark.parametrize("cls", METRIC)
def test_the_metric_is_the_vrr_alone_without_its_zero_terms(cls):
    k = build_hmetric_kernel(cls)
    assert metric_class(cls) == (cls[0], 0, cls[1], 0)
    assert not _hrr(k)
    used = _factors(k)
    assert not used & {Factor.pa(d) for d in range(3)}  # P = A on the bra
    assert not used & {Factor.qc(d) for d in range(3)}  # Q = C on the ket
    assert not used & {Factor.ab(d) for d in range(3)}
    assert k.params == kernel_metric.build_metric_kernel(cls).params
    assert k.name == hmetric_name(cls) == f"hmetric_{cls[0]}{cls[1]}"


@pytest.mark.parametrize("cls", ERI3)
def test_the_eri3_is_the_vrr_then_the_bra_hrr_the_higher_l_first(cls):
    """Built `(max, min | lP 0)`: only `(pp|P)` runs an HRR, `QC` never
    appears, and `PA` does exactly when the built bra has a p shell."""
    la, lb, lP = cls
    k = build_heri3_kernel(cls)
    assert eri3_class(cls) == (max(la, lb), min(la, lb), lP, 0)
    used = _factors(k)
    assert not used & {Factor.qc(d) for d in range(3)}
    assert bool(used & {Factor.pa(d) for d in range(3)}) == (lb == 1)
    hrr = la == lb == 1
    assert bool(used & {Factor.ab(d) for d in range(3)}) == hrr
    # every stored ERI is an HRR node, of the built orientation
    built = {hrr_name((b, a, q, (0, 0, 0))) if la < lb
             else hrr_name((a, b, q, (0, 0, 0)))
             for a in vrr.cartesians(la) for b in vrr.cartesians(lb)
             for q in vrr.cartesians(lP)}
    assert built <= _hrr(k)
    assert k.params == kernel_eri3.build_eri3_kernel(cls).params
    assert k.name == heri3_name(cls) == "heri3_" + "".join(map(str, cls))


def test_the_scatter_is_plain_stores_and_the_keys_carry_the_engine():
    for k, out, n in ((build_hmetric_kernel((1, 1)), kernel_metric.OUT,
                       18 + 9),
                      (build_heri3_kernel((1, 1, 1)), kernel_eri3.OUT,
                       2 * 27 + 27)):
        assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
        assert len([s for s in walk(k.body) if isinstance(s, Assign)
                    and isinstance(s.dst, Index) and s.dst.array == out]) == n
    assert hmetric_spec(0, 1).key() == "coulomb/d0/(0,1)/b1/metric.hgp/literal"
    assert heri3_spec(0, 1, 1).key() == "coulomb/d0/(0,1|1)/b1/eri3.hgp/literal"
    assert metric_kernel(1, 1, HGP) is metric_kernel(1, 1, HGP)  # memoised
    assert eri3_kernel(0, 1, 1, HGP).name == "heri3_011"
    mmd = Spec(Operator.COULOMB, 0, (1, 1), Target.METRIC)
    with pytest.raises(ValueError, match="not an HGP Target.METRIC key"):
        build_hmetric_kernel(mmd)
    with pytest.raises(ValueError, match="not an HGP Target.ERI3 key"):
        build_heri3_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0), Target.ERI3))
    with pytest.raises(ValueError, match="unbatched by decision"):
        Spec(Operator.COULOMB, 0, (1, 1), Target.METRIC, n_batches=3,
             engine=HGP)
