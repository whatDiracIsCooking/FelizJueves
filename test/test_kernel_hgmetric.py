"""`kernel_hgmetric`: the DF metric gradient kernel on the HGP engine (d1),
interpreted.

THE WITNESSES are the MMD path's: `gradient.host_df.metric_gradient` (held
to finite differences of the libcint-witnessed `(P|Q)`,
test/test_gradient_host_df.py), and, as a cross-engine pin, the MMD kernel
`kernel_gmetric` interpreted on the same staging. The HGP kernel takes the
Cartesian rule on the bra by the HGP VRR (`2 alpha [a+1_k|b] - a_k
[a-1_k|b]`); the host takes the Cartesian rule on both centres through
Hermite `E`, the MMD kernel the Hermite shift: three routes. Same system,
weights and scale as test_kernel_gmetric.py (whose `AUX` reaches every class
`lP + lQ <= 4`, swapped and off-diagonal).

ROUND-OFF, NOT BITWISE: `|G - G_host| <= 16 eps max(scale)` and `|G -
G_mmd| <= 32 eps max(scale)` (two roundings), `scale` the split-centre scale
of test_kernel_gmetric.py. Measured at writing: worst 7.87 eps of the scale
against the host, at (d|d).

SCOPE: every class `lP <= lQ`, `lP + lQ <= 4` (`L` for the tier cut is
`lP + lQ + 1`, as test_kernel_gmetric.py's: `lP + lQ <= 3` fast, 4 `slow`).
Nothing compiles, nothing launches.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_kernel_gmetric as tg
from df_grad_launch import run_metric
from tiers import by_l

from felizjueves.gradient import host_df, kernel_g1e, kernel_gmetric
from felizjueves.gradient.kernel_hgmetric import (
    ATOMIC_REASON,
    HGMETRIC_L_MAX,
    ZEROED_G,
    build_hgmetric_kernel,
    hgmetric_dag,
    hgmetric_name,
    hgmetric_spec,
)
from felizjueves.hgp import vrr
from felizjueves.ir import AtomicAdd, Decl, Factor, Note, Raw, Weighted, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = tg.EPS
TOL = 16 * EPS  # of max(scale), against the host
CROSS = 32 * EPS  # of max(scale), against the MMD kernel
HGP = Engine.HGP


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=tg._id(c))
                                 for c in tg.CLASSES])
def test_one_launch_over_the_class_is_the_host_and_the_mmd_gradient(
        cls) -> None:
    G, ran = run_metric(tg.AUX, tg.NUCLEI, tg._w(), boys=boys,
                        classes={cls}, engine=HGP)
    assert ran == (cls,)
    pairs = tg._pairs(cls)
    want = np.array(host_df.metric_gradient(tg.AUX, tg._w(), tg.NUCLEI,
                                            pq=pq, pairs=pairs))
    scale = tg._scale(pairs)
    tg._close(G, want, scale, f"hgmetric {cls}")
    mmd, _ = run_metric(tg.AUX, tg.NUCLEI, tg._w(), boys=boys, classes={cls})
    assert np.abs(G - mmd).max() <= CROSS * scale, cls
    # d/dB = -d/dA, added pair by pair: the rows sum to round-off.
    assert np.abs(G.sum(axis=0)).max() <= 4 * EPS * scale


def test_the_whole_gradient_of_an_sp_aux_basis() -> None:
    sp = tg.Basis(tuple(s for s in tg.AUX.shells if s.l <= 1))
    W = np.random.default_rng(9).standard_normal((sp.n_ao, sp.n_ao))
    G, ran = run_metric(sp, tg.NUCLEI, W, boys=boys, engine=HGP)
    assert sorted(ran) == [(0, 0), (0, 1), (1, 1)]
    want = np.array(host_df.metric_gradient(sp, W, tg.NUCLEI, pq=pq))
    assert np.abs(G - want).max() <= TOL * np.abs(want).max() * 4


def test_g_is_added_to_not_overwritten() -> None:
    G0 = np.arange(9.0).reshape(3, 3)
    G, _ = run_metric(tg.AUX, tg.NUCLEI, tg._w(), boys=boys,
                      classes={(0, 1)}, G=G0.copy(), engine=HGP)
    want, _ = run_metric(tg.AUX, tg.NUCLEI, tg._w(), boys=boys,
                         classes={(0, 1)}, engine=HGP)
    assert np.abs(G - (G0 + want)).max() <= 8 * EPS * np.abs(G0).max()


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert hgmetric_spec(1, 2).key() == "coulomb/d1/(1,2)/b1/metric.hgp/literal"
    assert hgmetric_name((1, 2)) == "hgmetric_12"
    assert hgmetric_name((3, 10)) == "hgmetric_3_10"
    assert build_hgmetric_kernel(hgmetric_spec(1, 2)).name == "hgmetric_12"
    assert HGMETRIC_L_MAX == 1


def test_another_key_is_refused() -> None:
    for spec in (Spec(Operator.COULOMB, 0, (1, 0), Target.METRIC,
                      engine=HGP),
                 Spec(Operator.COULOMB, 1, (1, 0), Target.METRIC),
                 Spec(Operator.COULOMB, 1, (1, 0), Target.JAB)):
        with pytest.raises(ValueError, match="not a d1 HGP Target.METRIC"):
            build_hgmetric_kernel(spec)


@pytest.mark.parametrize("cls", [(0, 0), (0, 2), (1, 1), (2, 1)],
                         ids=tg._id)
def test_the_vrr_targets_are_the_bra_one_up_and_one_down(cls) -> None:
    lP, lQ = cls
    dag = hgmetric_dag(cls)
    want = {(e, f, 0) for n in (lP - 1, lP + 1) if n >= 0
            for e in vrr.cartesians(n) for f in vrr.cartesians(lQ)}
    assert set(dag.targets) == want
    # the |e| = lP row of the class itself is never formed
    assert not any(sum(e) == lP and m == 0 and sum(f) == lQ
                   for e, f, m in dag.terms if not vrr.is_leaf((e, f, m)))


def test_no_zero_factor_is_spelled() -> None:
    zero = {Factor.pa(d) for d in range(3)} | {Factor.qc(d) for d in range(3)}
    for cls in [(0, 0), (0, 1), (1, 1), (1, 2)]:
        for s in walk(build_hgmetric_kernel(cls).body):
            if isinstance(s, Decl) and isinstance(s.init, Weighted):
                assert not {t.g for t in s.init.terms} & zero, cls


def test_the_parameters_notes_and_writes_are_the_mmd_kernels() -> None:
    k, m = build_hgmetric_kernel((1, 1)), kernel_gmetric.build_gmetric_kernel(
        (1, 1))
    assert k.params == m.params
    assert (ATOMIC_REASON, ZEROED_G) == (kernel_g1e.ATOMIC_REASON,
                                         kernel_g1e.ZEROED_G)
    nodes = list(walk(build_hgmetric_kernel((1, 2)).body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 and all(a.dst.array == "G" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)
