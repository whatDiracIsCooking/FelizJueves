"""`kernel_rmetric`: the DF metric `(P|Q)` on Rys quadrature, interpreted.

THE WITNESSES are the HGP twin's (`test_kernel_hgp_df.py`): the host
`eri.density_fit.metric` (MMD on `reference.pq`, placed by `place_staged`),
libcint's `int2c2e_cart` through the tracked `cc-pvtz-jkfit` set (#411),
and, as cross-engine pins, the MMD kernel and the HGP twin interpreted on
the same staging. The Rys kernel runs through the same adapter
(`metric_drive`, `engine=Engine.RYS`) with `rys.roots.roots_weights`
injected as the rule (the device routine's own operation order, as
`test_kernel_rj.py` does) -- the same parameters, launch and output.

ROUND-OFF, NOT BITWISE. THE BOUNDS, per element, the HGP twin's:

- against the host, `16 eps` of the MMD tests' tower-twin scale
  (`test_kernel_metric.pq_tower`); an element whose scale is 0 must be 0.
  Measured worst at writing 3.69 eps ((s|p), the synthetic basis).
- against the MMD kernel (and, through p, the HGP twin), `32 eps` of it.
- against libcint, `1e-12` of the fixture's `max |M|`.

SCOPE: every class `lP <= lQ`, `lP + lQ <= RYS_METRIC_L_MAX = 9` (the
rule's `N <= 5`), on `REACH`, `test_kernel_metric.SYNTHETIC` (every l to g;
swapped, diagonal and off-diagonal pairs) plus one h shell -- `L <=
tiers.FAST_L_MAX` fast, above it `slow`; the whole jkfit fixture (O to
g, so (g|g)) `slow`. Nothing compiles, nothing launches.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_kernel_metric as tm
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit import kernel_hgp, kernel_metric
from felizjueves.eri.density_fit.kernel_rmetric import (
    RYS_METRIC_L_MAX,
    build_rmetric_kernel,
    rmetric_name,
    rmetric_spec,
)
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.eri.density_fit.metric_drive import (
    metric_kernel,
    metric_matrix_kernel,
)
from felizjueves.hgp import vrr
from felizjueves.ir import Assign, AtomicAdd, Boys, Decl, Index, Rys, Var, walk
from felizjueves.ir.interp import InterpError
from felizjueves.reference.boys import boys
from felizjueves.rys.roots import n_roots, roots_weights
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL_ULPS = 16  # x eps x scale, against the host
CROSS_ULPS = 32  # x eps x scale, against the MMD kernel / the HGP twin
TOL = 1e-12  # x the fixture's max |M|, against libcint
RYS = Engine.RYS

CLASSES = [(a, b) for a in range(6) for b in range(a, 6)
           if a + b <= RYS_METRIC_L_MAX]
# The MMD tests' synthetic basis and one h shell on its third centre, so the
# ladder's (s|h) .. (g|h) are reached; (h|h) is past the rule.
REACH = Basis((*tm.SYNTHETIC.shells, Shell(5, tm.C, (0.7,), (1.0,))))
SP = [(0, 0), (0, 1), (1, 1)]


def _run(basis, classes=None, **kw):
    return metric_matrix_kernel(basis, boys=boys, classes=classes, threads=3,
                                engine=RYS, rys=roots_weights, **kw)


def _worst(err, scale, owned) -> float:
    pos = owned & (scale > 0)
    return float(np.max(err[pos] / (EPS * scale[pos]))) if pos.any() else 0.0


def _check(M, want, scale, owned, label, *twins):
    err = np.abs(M - want)
    print(f"\n{label}: worst {_worst(err, scale, owned):.2f} eps scale "
          "(host)" + "".join(
              f", {_worst(np.abs(M - t), scale, owned):.2f} ({n})"
              for n, t in twins))
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), label
    for n, t in twins:
        assert np.all(np.abs(M - t)[owned]
                      <= CROSS_ULPS * EPS * scale[owned]), (label, n)
    assert np.all(M[~owned] == 0.0)  # exactly the launched elements
    assert np.array_equal(M, M.T)  # the scatter mirrors one value


def test_the_synthetic_basis_reaches_every_class_of_the_ladder():
    by = {(c.lP, c.lQ) for c in metric_classes(REACH)}
    assert by == set(CLASSES) | {(5, 5)}
    assert len(CLASSES) == 20 and max(map(sum, CLASSES)) == RYS_METRIC_L_MAX


@pytest.mark.parametrize("cls", [by_l(c, sum(c), id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_the_class_agrees_with_the_host_and_the_mmd_kernel(cls):
    run = _run(REACH, {cls})
    assert run.ran == (cls,)
    mmd = metric_matrix_kernel(REACH, boys=boys, classes={cls},
                               threads=3).M
    twins = [("MMD kernel", mmd)]
    if max(cls) <= kernel_hgp.HGP_DF_L_MAX:
        twins.append(("HGP twin", metric_matrix_kernel(
            REACH, boys=boys, classes={cls}, threads=3,
            engine=Engine.HGP).M))
    want, scale, owned = tm._host(REACH, {cls})
    _check(run.M, want, scale, owned, f"rmetric {cls}", *twins)


def test_tight_exponents_past_the_narrow_boys_range():
    """`T` = 2000 (the MMD tests' `TIGHT` set): the rule's Laguerre end."""
    run = _run(tm.TIGHT)
    want, scale, owned = tm._host(tm.TIGHT, set(run.ran))
    assert owned.all()
    mmd = metric_matrix_kernel(tm.TIGHT, boys=boys).M
    _check(run.M, want, scale, owned, "rmetric tight", ("MMD kernel", mmd))


def test_without_the_rule_the_interpreter_refuses():
    with pytest.raises(InterpError, match="rys="):
        metric_matrix_kernel(tm.SYNTHETIC, boys=boys, classes={(0, 0)},
                             engine=RYS)


# ---------------------------------------------------------------------------
# The fixture: libcint
# ---------------------------------------------------------------------------


def _check_fixture(basis, rows, run, label):
    want, scale, owned = tm._host(basis, set(run.ran))
    _check(run.M, want, scale, owned, f"{label}, against the host")
    ref = tm._ms().metric[np.ix_(rows, rows)]
    worst = float(np.max(np.abs(run.M - ref)[owned])) / float(
        np.max(np.abs(tm._ms().metric)))
    print(f"{label}: max |M - libcint| = {worst:.2e} of max|M|")
    assert worst <= TOL, worst


def test_the_sp_metric_of_the_jkfit_set_is_libcints():
    aux = tm._aux()
    keep = [k for k, s in enumerate(aux.shells) if s.l <= 1]
    sub = Basis(tuple(aux.shells[k] for k in keep))
    rows = [aux.offsets[k] + a for k in keep
            for a in range(aux.shells[k].n_cart)]
    run = _run(sub)
    assert set(run.ran) == set(SP)
    _check_fixture(sub, rows, run, "rmetric l <= 1")
    assert np.linalg.eigvalsh(run.M).min() > 0


@pytest.mark.slow  # d, f and g shells: every class through (g|g)
def test_the_whole_jkfit_metric_is_libcints():
    """Every class of the whole aux basis (O to g, H to f), (g|g) at L = 8
    included: the whole matrix against the host and libcint."""
    aux = tm._aux()
    run = _run(aux)
    assert set(run.ran) == {(c.lP, c.lQ) for c in metric_classes(aux)}
    assert max(map(sum, run.ran)) == 8  # (g|g)
    _check_fixture(aux, list(range(aux.n_ao)), run, "rmetric whole set")
    assert np.linalg.eigvalsh(run.M).min() > 0


# ---------------------------------------------------------------------------
# What the kernel is built from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [by_l(c, sum(c), id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_the_rule_has_n_nodes_and_no_boys_no_hrr_no_single_centre_zeros(cls):
    k = build_rmetric_kernel(cls)
    rules = [s for s in walk(k.body) if isinstance(s, Rys)]
    assert [r.n for r in rules] == [n_roots(sum(cls))]
    assert not [s for s in walk(k.body) if isinstance(s, Boys)]
    names = {s.name for s in walk(k.body) if isinstance(s, Decl)}
    assert not {n for n in names if n.startswith("H")}  # no HRR
    # P - A = Q - C = 0: never declared, never read
    reads = {s.init.name for s in walk(k.body) if isinstance(s, Decl)
             and isinstance(s.init, Var)}
    for ax in "xyz":
        assert f"PA_{ax}" not in names | reads
        assert f"QC_{ax}" not in names | reads
    assert k.params == kernel_metric.build_metric_kernel(cls).params
    assert k.name == rmetric_name(cls) == f"rmetric_{cls[0]}{cls[1]}"


def test_the_scatter_is_the_hgp_twins_and_the_key_carries_the_engine():
    k, h = build_rmetric_kernel((1, 1)), kernel_hgp.build_hmetric_kernel((1, 1))
    assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]

    def stores(kern):
        return [s for s in walk(kern.body) if isinstance(s, Assign)
                and isinstance(s.dst, Index) and s.dst.array == kernel_metric.OUT]
    assert stores(k) == stores(h) and len(stores(k)) == 18 + 9
    assert len(vrr.cartesians(1)) ** 2 == 9
    assert rmetric_spec(0, 1).key() == "coulomb/d0/(0,1)/b1/metric.rys/literal"
    assert metric_kernel(1, 1, RYS) is metric_kernel(1, 1, RYS)  # memoised
    assert metric_kernel(2, 3, RYS).name == "rmetric_23"
    with pytest.raises(ValueError, match="not a Rys Target.METRIC key"):
        build_rmetric_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.METRIC))
    with pytest.raises(ValueError, match="rys.roots stops at N = 5"):
        build_rmetric_kernel((5, 5))
    with pytest.raises(ValueError, match="unbatched by decision"):
        Spec(Operator.COULOMB, 0, (1, 1), Target.METRIC, n_batches=3,
             engine=RYS)
