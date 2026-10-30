"""`kernel_reri3`: the three-center DF integral on Rys quadrature, interpreted.

THE WITNESSES are the MMD path's: the host `eri.density_fit.eri3`
(`class_block` with `reference.pq.pq` injected, placed by `place_staged`)
and, through the tracked set `h2o_sto3g_jkfit_3c` (#442), libcint's
`int3c2e_cart`; and, as cross-engine pins, the MMD kernel (`kernel_eri3`)
and the HGP one (`kernel_hgp`), interpreted on the same staging. The Rys
kernel runs through the same adapter (`eri3_drive`, `engine=Engine.RYS`)
with `reference.boys.boys` and `rys.roots.roots_weights` injected: the same
parameters, launch and output, so nothing else changes.

ROUND-OFF, NOT BITWISE (`kernel_reri3`'s module docstring). THE BOUNDS, per
element, the HGP twin's (test_kernel_hgp_df.py), unchanged:

- against the host, `|X - host| <= TOL_ULPS * eps * scale`, `TOL_ULPS =
  16`, `scale` the MMD tests' tower-twin absolute-value sum
  (`test_kernel_eri3._scale`). Measured worst at writing 6.62 eps, (dd|s).
  An element whose scale is 0 must be exactly 0.
- against the MMD and HGP kernels, `CROSS_ULPS = 32` eps of the same scale.
- against libcint, `TOL = 1e-12` of the fixture's `max |.|`.

SCOPE: every class `la + lb + lP <= 4` against the host (the MMD tests'
`CLASSES` and their fast/slow split, `test_kernel_eri3._param`), and the
WHOLE fixture tensor against libcint -- H2O/STO-3G's s and p against every
cc-pVTZ-jkfit aux shell through g, so classes to L = 6, which the MMD
kernel's own tests stop short of (`slow`, by #419's rule); its aux s and p
columns are fast. Nothing compiles, nothing launches.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_kernel_eri3 as te
from test_kernel_hgp_df import CROSS_ULPS, EPS, TOL, TOL_ULPS

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit import kernel_eri3, kernel_hgp
from felizjueves.eri.density_fit.eri3_drive import eri3_kernel, eri3_tensor_kernel
from felizjueves.eri.density_fit.kernel_reri3 import (
    RYS_ERI3_L_MAX,
    build_reri3_kernel,
    reri3_name,
    reri3_spec,
)
from felizjueves.hgp import vrr
from felizjueves.hgp.recurrence import hrr_name
from felizjueves.ir import Assign, AtomicAdd, Boys, Decl, Index, Rys, Var, walk
from felizjueves.reference.boys import boys
from felizjueves.rys.roots import N_MAX, n_roots, roots_weights
from felizjueves.spec import Engine, Operator, Spec, Target

RYS, HGP = Engine.RYS, Engine.HGP
SP = [(a, b, p) for a in range(2) for b in range(a, 2) for p in range(2)]


def _run(basis, aux, *, engine=RYS, **kw):
    return eri3_tensor_kernel(basis, aux, boys=boys, engine=engine,
                              rys=roots_weights if engine is RYS else None,
                              **kw)


def _worst(err, scale, owned) -> float:
    pos = owned & (scale > 0)
    return float(np.max(err[pos] / (EPS * scale[pos]))) if pos.any() else 0.0


def _check(X, want, scale, owned, label, *others):
    err = np.abs(X - want)
    print(f"\n{label}: worst {_worst(err, scale, owned):.2f} eps scale (host)"
          + "".join(f", {_worst(np.abs(X - o), scale, owned):.2f} ({n})"
                    for n, o in others))
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), label
    for n, o in others:
        assert np.all(np.abs(X - o)[owned]
                      <= CROSS_ULPS * EPS * scale[owned]), (label, n)
    assert np.all(X[~owned] == 0.0)  # exactly the launched elements
    assert np.array_equal(X, X.transpose(1, 0, 2))  # the (m, n) mirror


@pytest.mark.parametrize("cls", [te._param(c) for c in te.CLASSES])
def test_every_class_through_l4_agrees_with_the_host(cls):
    run = _run(te.SYNTHETIC, te.SYNTHETIC_AUX, classes={cls}, threads=3)
    assert run.ran == (cls,)
    want, scale, owned = te._host(te.SYNTHETIC, te.SYNTHETIC_AUX, {cls})
    assert owned.any()
    _check(run.T, want, scale, owned, f"reri3 ({te._id(cls)})")


@pytest.mark.parametrize("cls", [(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
                         ids=te._id)
def test_the_sp_class_agrees_with_the_mmd_and_hgp_kernels(cls):
    from test_kernel_hgp_df import BASIS, BASIS_AUX

    run = _run(BASIS, BASIS_AUX, classes={cls}, threads=3)
    mmd = _run(BASIS, BASIS_AUX, classes={cls}, threads=3,
               engine=Engine.MMD).T
    hgp = _run(BASIS, BASIS_AUX, classes={cls}, threads=3, engine=HGP).T
    want, scale, owned = te._host(BASIS, BASIS_AUX, {cls})
    _check(run.T, want, scale, owned, f"reri3 ({te._id(cls)})",
           ("MMD kernel", mmd), ("HGP kernel", hgp))


def test_tight_exponents_past_the_narrow_boys_range():
    """`T` past 709 (the MMD tests' `TIGHT` sets): the Rys rule's Laguerre
    branch (`T >= rys.roots.T_SWITCH`)."""
    run = _run(te.TIGHT, te.TIGHT_AUX)
    want, scale, owned = te._host(te.TIGHT, te.TIGHT_AUX, set(run.ran))
    assert owned.all()
    _check(run.T, want, scale, owned, "reri3 tight")


# ---------------------------------------------------------------------------
# The fixture: libcint's int3c2e_cart over H2O/STO-3G x cc-pVTZ-jkfit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lmax", [
    pytest.param(1, id="aux-sp"),
    pytest.param(4, id="aux-spdfg", marks=pytest.mark.slow)])
def test_the_tensor_over_the_aux_basis_is_libcints(lmax):
    """H2O/STO-3G's AOs against the aux shells with `l <= lmax`: at 4 the
    whole fixture, every class through (pp|g) at L = 6."""
    aux = te._aux()
    keep = [k for k, s in enumerate(aux.shells) if s.l <= lmax]
    sub = Basis(tuple(aux.shells[k] for k in keep))
    cols = [aux.offsets[k] + q for k in keep
            for q in range(aux.shells[k].n_cart)]
    run = _run(te._basis(), sub)
    if lmax == 4:
        assert len(cols) == aux.n_ao and max(map(sum, run.ran)) == 6
    ref = te._ts().tensor[:, :, cols]
    worst = float(np.max(np.abs(run.T - ref))) / float(
        np.max(np.abs(te._ts().tensor)))
    print(f"\nreri3 aux l <= {lmax}: {len(run.ran)} classes, max |T - "
          f"libcint| = {worst:.2e} of max|(mn|P)|")
    assert worst <= TOL, worst
    assert np.array_equal(run.T, run.T.transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# What the kernel is built from
# ---------------------------------------------------------------------------


def _names(kernel) -> set[str]:
    return {s.name for s in walk(kernel.body) if isinstance(s, Decl)}


@pytest.mark.parametrize("cls", SP + [(0, 2, 2), (1, 2, 1)], ids=te._id)
def test_the_rule_replaces_the_boys_vrr_and_the_rest_is_hgps(cls):
    """One `Rys` of `N = L // 2 + 1` nodes per (bra, aux) record and no
    Boys; `QC` never formed (`D00 = x WQ`); the HRR nodes and the scatter
    are `kernel_hgp`'s, so the stored ERIs are the HGP kernel's names."""
    la, lb, lP = cls
    k = build_reri3_kernel(cls)
    rys = [s for s in walk(k.body) if isinstance(s, Rys)]
    assert [r.n for r in rys] == [n_roots(sum(cls))]
    assert not [s for s in walk(k.body) if isinstance(s, Boys)]
    names = _names(k)
    assert not {n for n in names if n.startswith("QC_")}
    assert bool({n for n in names if n.startswith("PA_")}) == (la + lb > 0)
    assert bool({n for n in names if n.startswith("AB_")}) == (min(la, lb)
                                                               > 0)
    built = {hrr_name((b, a, q, (0, 0, 0))) if la < lb
             else hrr_name((a, b, q, (0, 0, 0)))
             for a in vrr.cartesians(la) for b in vrr.cartesians(lb)
             for q in vrr.cartesians(lP)}
    stored = {s.value.name for s in walk(k.body) if isinstance(s, Assign)
              and isinstance(s.dst, Index) and s.dst.array == kernel_eri3.OUT
              and isinstance(s.value, Var)}
    assert stored == built
    assert k.params == kernel_eri3.build_eri3_kernel(cls).params
    assert k.params == kernel_hgp.build_heri3_kernel(cls).params
    assert k.name == reri3_name(cls) == "reri3_" + "".join(map(str, cls))


def test_the_scatter_is_plain_stores_and_the_key_carries_the_engine():
    k = build_reri3_kernel((1, 1, 1))
    assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    assert reri3_spec(0, 1, 1).key() == "coulomb/d0/(0,1|1)/b1/eri3.rys/literal"
    assert eri3_kernel(0, 1, 1, RYS) is eri3_kernel(0, 1, 1, RYS)  # memoised
    assert eri3_kernel(0, 1, 1, RYS).name == "reri3_011"
    assert reri3_name((10, 10, 0)) == "reri3_10_10_0"
    with pytest.raises(ValueError, match="not a Rys Target.ERI3 key"):
        build_reri3_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0), Target.ERI3))
    with pytest.raises(ValueError, match="not a Rys Target.ERI3 key"):
        build_reri3_kernel(kernel_hgp.heri3_spec(0, 0, 0))
    with pytest.raises(ValueError, match="unbatched by decision"):
        Spec(Operator.COULOMB, 0, (1, 1, 1), Target.ERI3, n_batches=3,
             engine=RYS)


def test_the_reach_is_the_rules():
    """`N = L // 2 + 1 <= N_MAX = 5`: L = 9 is the top, L = 10 a refusal
    (before any statement is built)."""
    assert RYS_ERI3_L_MAX == 9 and n_roots(RYS_ERI3_L_MAX) == N_MAX
    with pytest.raises(ValueError, match="stops at N = 5"):
        build_reri3_kernel((2, 4, 4))
    with pytest.raises(ValueError, match="stops at N = 5"):
        reri3_spec(0, 0, 10).key() and build_reri3_kernel((0, 0, 10))


@pytest.mark.slow
@pytest.mark.l_cut
def test_the_top_of_the_reach_builds():
    k = build_reri3_kernel((2, 3, 4))
    assert [s.n for s in walk(k.body) if isinstance(s, Rys)] == [5]
