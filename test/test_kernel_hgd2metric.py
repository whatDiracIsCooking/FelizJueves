"""`kernel_hgd2metric`: the DF metric Hessian kernel on the HGP engine (d2),
interpreted.

THE WITNESSES are the MMD path's: `hessian.host_df.metric_hessian` (held to
finite differences of `gradient.host_df.metric_gradient`,
test/test_hessian_host_df.py), and, as a cross-engine pin, the MMD kernel
`kernel_d2metric` interpreted on the same staging. The HGP kernel applies
the Cartesian rule twice on the bra through the HGP VRR; the host applies
it twice on each centre through Hermite `E`, the MMD kernel shifts the
Hermite tower twice: three routes. Same system, weights and scale as
test_kernel_d2metric.py (whose `AUX` reaches every class `lP + lQ <= 4`,
swapped and off-diagonal).

ROUND-OFF, NOT BITWISE: `|H - H_host| <= 16 eps max(twin)` and `|H -
H_mmd| <= 32 eps max(twin)` (two roundings), `twin` the host's
`absolute=True` twin over the same pairs. Measured at writing: worst 0.32
eps of the twin against the host, at (p|d).

SCOPE: every class `lP <= lQ`, `lP + lQ <= 4` (`L` for the tier cut is
`lP + lQ + 2`, as test_kernel_d2metric.py's: `lP + lQ <= 2` fast, 3 and 4
`slow`). Nothing compiles, nothing launches.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_kernel_d2metric as tm
from df_grad_launch import run_d2metric
from tiers import by_l

from felizjueves.ao.shell import Basis
from felizjueves.hessian import host_df, kernel_d2metric
from felizjueves.hessian.kernel_hgd2metric import (
    ATOMIC_REASON,
    HGD2METRIC_L_MAX,
    KL,
    ZEROED_H,
    build_hgd2metric_kernel,
    hgd2metric_dag,
    hgd2metric_name,
    hgd2metric_spec,
)
from felizjueves.hgp import vrr
from felizjueves.ir import AtomicAdd, Decl, Factor, Note, Raw, Weighted, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = tm.EPS
TOL = 16 * EPS  # of max(twin), against the host
CROSS = 32 * EPS  # of max(twin), against the MMD kernel
HGP = Engine.HGP


def _twin(pairs) -> float:
    return float(np.array(host_df.metric_hessian(
        tm.AUX, tm._w(), tm.NUCLEI, pq=pq, pairs=pairs,
        absolute=True)).max())


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=tm._id(c))
                                 for c in tm.CLASSES])
def test_one_launch_over_the_class_is_the_host_and_the_mmd_hessian(
        cls) -> None:
    H, ran = run_d2metric(tm.AUX, tm.NUCLEI, tm._w(), boys=boys,
                          classes={cls}, engine=HGP)
    assert ran == (cls,)
    pairs = tm._pairs(cls)
    want = np.array(host_df.metric_hessian(tm.AUX, tm._w(), tm.NUCLEI,
                                           pq=pq, pairs=pairs))
    scale = _twin(pairs)
    tm._close(H, want, scale, f"hgd2metric {cls}")
    mmd, _ = run_d2metric(tm.AUX, tm.NUCLEI, tm._w(), boys=boys,
                          classes={cls})
    assert np.abs(H - mmd).max() <= CROSS * scale, cls
    # d2/dA dB = -h, added pair by pair: the rows sum to round-off, H = H^T.
    assert np.abs(H.sum(axis=0)).max() <= 4 * EPS * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= 4 * EPS * scale


def test_the_whole_hessian_of_an_sp_aux_basis() -> None:
    sp = Basis(tuple(s for s in tm.AUX.shells if s.l <= 1))
    W = np.random.default_rng(9).standard_normal((sp.n_ao, sp.n_ao))
    H, ran = run_d2metric(sp, tm.NUCLEI, W, boys=boys, engine=HGP)
    assert sorted(ran) == [(0, 0), (0, 1), (1, 1)]
    want = np.array(host_df.metric_hessian(sp, W, tm.NUCLEI, pq=pq))
    scale = float(np.array(host_df.metric_hessian(
        sp, W, tm.NUCLEI, pq=pq, absolute=True)).max())
    tm._close(H, want, scale, "hgd2metric whole sp")


def test_h_is_added_to_not_overwritten() -> None:
    H0 = np.arange(81.0).reshape(3, 3, 3, 3)
    H, _ = run_d2metric(tm.AUX, tm.NUCLEI, tm._w(), boys=boys,
                        classes={(0, 1)}, H=H0.copy(), engine=HGP)
    want, _ = run_d2metric(tm.AUX, tm.NUCLEI, tm._w(), boys=boys,
                           classes={(0, 1)}, engine=HGP)
    assert np.abs(H - (H0 + want)).max() <= 8 * EPS * np.abs(H0).max()


def test_the_launcher_refuses_a_form_on_hgp() -> None:
    with pytest.raises(ValueError, match="one form"):
        run_d2metric(tm.AUX, tm.NUCLEI, tm._w(), boys=boys,
                     classes={(0, 0)}, engine=HGP, lowstack=True)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert hgd2metric_spec(1, 2).key() == \
        "coulomb/d2/(1,2)/b1/metric.hgp/literal"
    assert hgd2metric_name((1, 2)) == "hgd2metric_12"
    assert hgd2metric_name((3, 10)) == "hgd2metric_3_10"
    assert build_hgd2metric_kernel(hgd2metric_spec(1, 2)).name == \
        "hgd2metric_12"
    assert HGD2METRIC_L_MAX == 1


def test_another_key_is_refused() -> None:
    for spec in (Spec(Operator.COULOMB, 1, (1, 0), Target.METRIC,
                      engine=HGP),
                 Spec(Operator.COULOMB, 2, (1, 0), Target.METRIC),
                 Spec(Operator.COULOMB, 2, (1, 0), Target.JAB)):
        with pytest.raises(ValueError, match="not a d2 HGP Target.METRIC"):
            build_hgd2metric_kernel(spec)


@pytest.mark.parametrize("cls", [(0, 0), (0, 2), (1, 1), (2, 1), (3, 0)],
                         ids=tm._id)
def test_the_vrr_targets_are_the_bra_two_up_level_and_two_down(cls) -> None:
    lP, lQ = cls
    dag = hgd2metric_dag(cls)
    want = {(e, f, 0) for n in (lP - 2, lP, lP + 2) if n >= 0
            for e in vrr.cartesians(n) for f in vrr.cartesians(lQ)}
    assert set(dag.targets) == want


def test_no_zero_factor_is_spelled() -> None:
    zero = {Factor.pa(d) for d in range(3)} | {Factor.qc(d) for d in range(3)}
    for cls in [(0, 0), (0, 1), (1, 1), (1, 2)]:
        for s in walk(build_hgd2metric_kernel(cls).body):
            if isinstance(s, Decl) and isinstance(s.init, Weighted):
                assert not {t.g for t in s.init.terms} & zero, cls


def test_the_parameters_notes_and_writes_are_the_mmd_kernels() -> None:
    k = build_hgd2metric_kernel((1, 1))
    m = kernel_d2metric.build_d2metric_kernel((1, 1))
    assert k.params == m.params
    assert (ATOMIC_REASON, ZEROED_H, KL) == (
        kernel_d2metric.ATOMIC_REASON, kernel_d2metric.ZEROED_H,
        kernel_d2metric.KL)
    nodes = list(walk(build_hgd2metric_kernel((1, 2)).body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 36 and all(a.dst.array == "H" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_H in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_six_contractions_not_nine() -> None:
    names = {s.name for s in walk(build_hgd2metric_kernel((0, 1)).body)
             if isinstance(s, Decl)}
    assert {n for n in names if n.startswith("h") and len(n) == 3} == {
        f"h{k}{l}" for k, l in KL}  # noqa: E741
