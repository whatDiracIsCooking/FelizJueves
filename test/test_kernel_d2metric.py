"""`kernel_d2metric`: the DF metric Hessian kernel (d2), interpreted.

THE WITNESS is `hessian.host_df.metric_hessian` (itself held to finite
differences of `gradient.host_df.metric_gradient`,
test/test_hessian_host_df.py), which takes ANOTHER ROUTE: the Cartesian
rule `2 alpha E^{a+1} - a E^{a-1}` applied twice on each centre, where the
kernel shifts the Hermite tower twice (`[r] -> [r + 1_k + 1_l]`) and writes
the mixed and second-centre blocks as `-h` and `+h`. Per class, one launch
over every staged pair into a zeroed `H`, against the host over the same
pairs; and the whole Hessian of an aux basis, one launch per class.

ROUND-OFF, NOT BITWISE: `|H_kernel - H_host| <= TOL max(scale)`, `TOL = 16
eps`, `scale` the host's `absolute=True` twin over the same pairs (every
factor in `|.|`: the size of the terms before they cancel). `W` is a
non-symmetric matrix (the weight rule is exact for any `W`).

SCOPE: every class `lP <= lQ` with `lP + lQ <= 4` (the d0 kernel's cap).
`L` for the tier cut is `lP + lQ + 2`, the Hermite order the Hessian
reaches (test/tiers.py): `lP + lQ <= 2` fast, 3 and 4 `slow`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from df_grad_launch import run_d2metric
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.gradient.kernel_gmetric import center_cols, unit
from felizjueves.hessian import host_df
from felizjueves.hessian.kernel_d2metric import (
    ATOMIC_REASON,
    KL,
    ZEROED_H,
    build_d2metric_kernel,
    d2metric_name,
    d2metric_plan,
    d2metric_spec,
)
from felizjueves.ir import AtomicAdd, Decl, Intent, Note, Raw, walk
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.recurrence import node_name
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(scale): the module docstring
L_CAP = 4
CLASSES = [(p, q) for p in range(L_CAP + 1) for q in range(p, L_CAP + 1)
           if p + q <= L_CAP]

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
NUCLEI = ((8.0, A), (1.0, B), (1.0, C))
# test_kernel_gmetric.py's aux basis: every l to g on the three centres, the
# higher l first somewhere (so every lP < lQ class has pairs staged
# swapped), two s and two p shells (so the equal-l classes have
# off-diagonal pairs), contracted and not.
AUX = Basis((
    Shell(0, C, (2.0, 0.4), (0.7, 0.5)),
    Shell(1, A, (0.85,), (1.0,)),
    Shell(2, B, (1.4, 0.35), (0.6, 0.45)),
    Shell(3, B, (0.7,), (1.0,)),
    Shell(4, A, (1.2,), (1.0,)),
    Shell(0, A, (5.0,), (1.0,)),
    Shell(2, C, (0.9,), (1.0,)),
    Shell(1, B, (1.1, 0.5), (0.6, 0.5)),
))


def _id(c) -> str:
    return f"{'spdfg'[c[0]]}{'spdfg'[c[1]]}"


@cache
def _w() -> np.ndarray:
    return np.random.default_rng(5).standard_normal((AUX.n_ao, AUX.n_ao))


def _pairs(cls) -> list[tuple[int, int]]:
    sh = AUX.shells
    return [(i, j) for i in range(len(sh)) for j in range(i, len(sh))
            if tuple(sorted((sh[i].l, sh[j].l))) == cls]


def _close(got, want, scale: float, what: str) -> float:
    err = float(np.abs(got - want).max())
    print(f"\n{what}: {err / scale / EPS:.2f} eps of scale")
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"
    return err


def test_the_basis_reaches_every_class_swapped_and_off_diagonal() -> None:
    by = {(c.lP, c.lQ): c for c in metric_classes(AUX)}
    for cls in CLASSES:
        c = by[cls]
        assert any(p != q for p, q in c.pairs), cls
        if cls[0] != cls[1]:
            assert any(c.swapped), cls


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_id(c))
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_hessian(cls) -> None:
    H, ran = run_d2metric(AUX, NUCLEI, _w(), boys=boys, classes={cls})
    assert ran == (cls,)
    pairs = _pairs(cls)
    want = np.array(host_df.metric_hessian(AUX, _w(), NUCLEI, pq=pq,
                                           pairs=pairs))
    scale = float(np.array(host_df.metric_hessian(
        AUX, _w(), NUCLEI, pq=pq, pairs=pairs, absolute=True)).max())
    _close(H, want, scale, f"d2metric {cls}")
    # d2/dA dB = -d2/dA dA, added pair by pair: every row sums to round-off,
    # and H = H^T.
    assert np.abs(H.sum(axis=0)).max() <= 4 * EPS * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= 4 * EPS * scale


def test_the_whole_hessian_of_an_sp_aux_basis() -> None:
    sp = Basis(tuple(s for s in AUX.shells if s.l <= 1))
    W = np.random.default_rng(9).standard_normal((sp.n_ao, sp.n_ao))
    H, ran = run_d2metric(sp, NUCLEI, W, boys=boys)
    assert set(ran) == {(0, 0), (0, 1), (1, 1)}
    want = np.array(host_df.metric_hessian(sp, W, NUCLEI, pq=pq))
    scale = float(np.array(host_df.metric_hessian(
        sp, W, NUCLEI, pq=pq, absolute=True)).max())
    _close(H, want, scale, "d2metric whole sp")


def test_h_is_added_to_not_overwritten() -> None:
    H0 = np.arange(81.0).reshape(3, 3, 3, 3)
    H, _ = run_d2metric(AUX, NUCLEI, _w(), boys=boys, classes={(0, 1)},
                       H=H0.copy())
    want, _ = run_d2metric(AUX, NUCLEI, _w(), boys=boys, classes={(0, 1)})
    assert np.abs(H - (H0 + want)).max() <= 8 * EPS * np.abs(H0).max()


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert d2metric_spec(1, 2).key() == "coulomb/d2/(1,2)/b1/metric/literal"
    assert d2metric_name((1, 2)) == "d2metric_12"
    assert build_d2metric_kernel(d2metric_spec(1, 2)).name == "d2metric_12"


def test_a_d1_key_or_another_target_is_refused() -> None:
    with pytest.raises(ValueError, match="not a d2 Target.METRIC key"):
        build_d2metric_kernel(Spec(Operator.COULOMB, 1, (1, 0), Target.METRIC))
    with pytest.raises(ValueError, match="not a d2 Target.METRIC key"):
        build_d2metric_kernel(Spec(Operator.COULOMB, 2, (1, 0), Target.JAB))


def _declared(kernel) -> set[str]:
    return {s.name for s in walk(kernel.body) if isinstance(s, Decl)}


def _plus(*vs):
    return tuple(sum(v[d] for v in vs) for d in range(3))


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_id(c))
                                 for c in [(0, 0), (0, 1), (1, 1), (1, 2)]])
def test_the_tower_is_two_orders_up_and_keeps_exactly_the_shifted_targets(
        cls) -> None:
    lP, lQ = cls
    _, order = d2metric_plan(cls)
    assert max(sum(r) for r, _ in order) == lP + lQ + 2
    live = {_plus(t, tau, unit(k), unit(l))
            for a in ef.cartesians(lP) for b in ef.cartesians(lQ)
            for _, t in center_cols(a, lP) for _, tau in center_cols(b, lQ)
            for k, l in KL}  # noqa: E741
    assert all(sum(r) % 2 == (lP + lQ) % 2 for r in live)
    kept = {r for n in range(lP + lQ + 3) for r in hermite_targets(n)
            if node_name((r, 0)) in _declared(build_d2metric_kernel(cls))}
    assert kept == live


def test_the_parameters_and_the_one_output() -> None:
    k = build_d2metric_kernel((1, 1))
    assert [p.name for p in k.params] == [
        "n_pair", "n_rec", "n_aux", "n_atom", "A", "alpha", "s_aux",
        "p_rec0", "p_rec1", "q_rec0", "q_rec1", "off_p", "off_q", "atom_p",
        "atom_q", "M", "H"]
    out = [q for q in k.params if q.intent is Intent.OUT]
    assert [q.name for q in out] == ["H"] and len(out[0].shape) == 4


def test_every_write_is_an_atomic_add_into_h() -> None:
    nodes = list(walk(build_d2metric_kernel((1, 2)).body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 36 and all(a.dst.array == "H" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_H in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_six_contractions_not_nine() -> None:
    names = _declared(build_d2metric_kernel((0, 1)))
    assert {n for n in names if n.startswith("h") and len(n) == 3} == {
        f"h{k}{l}" for k, l in KL}  # noqa: E741
