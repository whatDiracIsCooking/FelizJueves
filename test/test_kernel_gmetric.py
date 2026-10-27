"""`kernel_gmetric`: the DF metric gradient kernel (d1), interpreted.

THE WITNESS is `gradient.host_df.metric_gradient` (itself held to finite
differences of the libcint-witnessed `(P|Q)`, test/test_gradient_host_df.py),
which takes ANOTHER ROUTE: the Cartesian rule `2 alpha E^{a+1} - a E^{a-1}`
on both centres, where the kernel shifts the Hermite tower (`[r] -> [r +
1_k]`) and writes the second centre as minus the first. Per class, one
launch over every staged pair into a zeroed `G`, against the host over the
same pairs; and the whole gradient of an aux basis, one launch per class.

ROUND-OFF, NOT BITWISE: `|G_kernel - G_host| <= TOL max(scale)`, `TOL =
16 eps`, `scale` the split-centre scale of test_kernel_g1e.py: the host's
`|G|` with EVERY SHELL ON A ROW OF ITS OWN (`atoms=`), folded back onto the
atoms -- the size of the terms before a one-centre pair cancels on its atom.
`W` is a non-symmetric matrix (the weight rule is exact for any `W`).

SCOPE: every class `lP <= lQ` with `lP + lQ <= 4` (the d0 kernel's cap).
`L` for the tier cut is `lP + lQ + 1`, the Hermite order the gradient
reaches (test/tiers.py): `lP + lQ <= 3` fast, 4 `slow`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from df_grad_launch import run_metric
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.gradient import host_df
from felizjueves.gradient.host import shell_atoms
from felizjueves.gradient.kernel_g1e import ATOMIC_REASON, ZEROED_G
from felizjueves.gradient.kernel_gmetric import (
    build_gmetric_kernel,
    center_cols,
    gmetric_name,
    gmetric_plan,
    gmetric_spec,
    launch_shape,
    unit,
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
# Every l to g on the three centres, the higher l first somewhere (so every
# lP < lQ class has pairs staged swapped), two s and two p shells (so the
# equal-l classes have off-diagonal pairs), contracted and not.
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


def _scale(pairs) -> float:
    nat = len(NUCLEI)
    real = shell_atoms(AUX, NUCLEI)
    split = tuple(nat + i for i in range(len(AUX.shells)))
    sc = np.zeros((nat, 3))
    for p in pairs:
        g = np.abs(np.array(host_df.metric_gradient(
            AUX, _w(), NUCLEI, pq=pq, pairs=[p], atoms=split)))
        for i, row in enumerate(split):
            sc[real[i]] += g[row]
    return float(sc.max())


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


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c))
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_gradient(cls) -> None:
    G, ran = run_metric(AUX, NUCLEI, _w(), boys=boys, classes={cls})
    assert ran == (cls,)
    pairs = _pairs(cls)
    want = np.array(host_df.metric_gradient(AUX, _w(), NUCLEI, pq=pq,
                                            pairs=pairs))
    _close(G, want, _scale(pairs), f"gmetric {cls}")
    # d/dB = -d/dA, added pair by pair: the rows sum to round-off.
    assert np.abs(G.sum(axis=0)).max() <= 4 * EPS * _scale(pairs)


def test_the_whole_gradient_of_an_sp_aux_basis() -> None:
    sp = Basis(tuple(s for s in AUX.shells if s.l <= 1))
    W = np.random.default_rng(9).standard_normal((sp.n_ao, sp.n_ao))
    G, _ = run_metric(sp, NUCLEI, W, boys=boys)
    want = np.array(host_df.metric_gradient(sp, W, NUCLEI, pq=pq))
    assert np.abs(G - want).max() <= TOL * np.abs(want).max() * 4


def test_g_is_added_to_not_overwritten() -> None:
    G0 = np.arange(9.0).reshape(3, 3)
    G, _ = run_metric(AUX, NUCLEI, _w(), boys=boys, classes={(0, 1)},
                      G=G0.copy())
    want, _ = run_metric(AUX, NUCLEI, _w(), boys=boys, classes={(0, 1)})
    assert np.abs(G - (G0 + want)).max() <= 8 * EPS * np.abs(G0).max()


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert gmetric_spec(1, 2).key() == "coulomb/d1/(1,2)/b1/metric/literal"
    assert gmetric_name((1, 2)) == "gmetric_12"
    assert build_gmetric_kernel(gmetric_spec(1, 2)).name == "gmetric_12"


def test_a_d0_key_or_another_target_is_refused() -> None:
    with pytest.raises(ValueError, match="not a d1 Target.METRIC key"):
        build_gmetric_kernel(Spec(Operator.COULOMB, 0, (1, 0), Target.METRIC))
    with pytest.raises(ValueError, match="not a d1 Target.METRIC key"):
        build_gmetric_kernel(Spec(Operator.COULOMB, 1, (1, 0), Target.JAB))


def _declared(kernel) -> set[str]:
    return {s.name for s in walk(kernel.body) if isinstance(s, Decl)}


@pytest.mark.parametrize("cls", [(0, 0), (0, 1), (1, 1), (1, 2)], ids=_id)
def test_the_tower_is_one_order_up_and_keeps_exactly_the_shifted_targets(
        cls) -> None:
    lP, lQ = cls
    _, order = gmetric_plan(cls)
    assert max(sum(r) for r, _ in order) == lP + lQ + 1
    live = {_plus(t, tau, unit(k))
            for a in ef.cartesians(lP) for b in ef.cartesians(lQ)
            for _, t in center_cols(a, lP) for _, tau in center_cols(b, lQ)
            for k in range(3)}
    assert all(sum(r) % 2 == (lP + lQ + 1) % 2 for r in live)
    kept = {r for r in hermite_targets(lP + lQ + 1)
            if node_name((r, 0)) in _declared(build_gmetric_kernel(cls))}
    assert kept == live


def _plus(*vs):
    return tuple(sum(v[d] for v in vs) for d in range(3))


def test_the_parameters_and_the_one_output() -> None:
    k = build_gmetric_kernel((1, 1))
    assert [p.name for p in k.params] == [
        "n_pair", "n_rec", "n_aux", "n_atom", "A", "alpha", "s_aux",
        "p_rec0", "p_rec1", "q_rec0", "q_rec1", "off_p", "off_q", "atom_p",
        "atom_q", "M", "G"]
    assert [q.name for q in k.params if q.intent is Intent.OUT] == ["G"]


def test_every_write_is_an_atomic_add_into_g() -> None:
    nodes = list(walk(build_gmetric_kernel((1, 2)).body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 and all(a.dst.array == "G" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
