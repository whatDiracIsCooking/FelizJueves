"""`kernel_ipmetric`: the uncontracted DF metric derivative (#693), interpreted.

    M1[k, P, Q] = d(P|Q)/dA_k            A the centre of P, [3, n_aux, n_aux]

THE WITNESSES.

- The host, `gradient.host_df.metric_derivative`, by ANOTHER ROUTE: the
  Cartesian rule `2 alpha E^{a+1} - a E^{a-1}` on each centre separately,
  where the kernel shifts the Hermite tower (`[r] -> [r + 1_k]`) and writes
  the other centre's derivative as the negated mirror. Every element a class
  owns, `|M1 - host| <= 16 eps` of the class's scale, and every element it
  does not own untouched. The scale is `max |host|` through L = 4 (on
  test_kernel_gmetric.py's basis, every class swapped and off-diagonal),
  and above it the host on `test_kernel_metric.pq_tower`, the
  absolute-value tower (test_kernel_df_grad_ladder.py's yardstick: at
  L = 9 to 13 the tower's own cancellation is what both routes round), on
  that file's two-shell basis per class.
- libcint, twice. `-int2c2e_ip1_cart` itself, per axis, over
  cc-pVDZ-JKFIT on H2O (the numerical golden store, #655: O to f, so every
  class through (f|f)), within the fp64 tier (`1e-12` of the golden's
  max); and through (i|i), the aux-i witness of #613
  (`oracle.manifest.df_gradient`: `int2c2e_ip1_cart` contracted per class
  with a closed-form `W`), against `M1` contracted the same way,

      G[A, k] = SUM_{X on A} SUM_Y (W_XY + W_YX) M1[k, X, Y]

  over the class's elements, within `1e-12` of the witness's largest
  class -- exact linear algebra on `M1`, so it holds every element the
  contraction weighs, not just their sum.

THE MIRROR IS EXACT: an off-diagonal pair's `(Q, P)` is `0 - (P, Q)`, a
negation, so `M1[k] = -M1[k]^T` bitwise there; a diagonal pair forms both
orders by its own sums, round-off apart.

SCOPE: every class `lP <= lQ <= 6`, FreeBreakfast's aux ladder. `L` for the
tier cut is `lP + lQ + 1`, the Hermite order the derivative reaches
(test/tiers.py, as test_kernel_gmetric.py): `lP + lQ <= 3` fast, the rest
`slow`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from df_grad_launch import run_ipmetric
from oracle import basis as ob
from oracle import manifest
from oracle import numeric as golden
from test_kernel_df_grad_ladder import _metric_basis
from test_kernel_gmetric import AUX
from test_kernel_metric import pq_tower
from tiers import by_l

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.metric import metric_classes
from felizjueves.gradient import host_df
from felizjueves.gradient.host import shell_atoms
from felizjueves.gradient.kernel_gmetric import build_gmetric_kernel
from felizjueves.gradient.kernel_ipmetric import (
    OUT,
    UNCONTRACTED,
    build_ipmetric_kernel,
    ipmetric_name,
    ipmetric_spec,
    launch_shape,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target, lever_provenance

EPS = 2.0 ** -52
TOL = 16 * EPS  # of the class's scale, against the host
LIBCINT_TOL = 1e-12  # of the aux-i witness's largest class max|G|
LADDER = [(p, q) for p in range(7) for q in range(p, 7)]
LOW = [c for c in LADDER if sum(c) <= 4]
HIGH = [c for c in LADDER if sum(c) >= 5]
GOLDEN = tuple(f"metric_ip1{x}_h2o_ccpvdz_jkfit_bse" for x in "xyz")


def _id(c) -> str:
    return ipmetric_name(c)


def _pairs(aux: Basis, cls) -> list[tuple[int, int]]:
    sh = aux.shells
    return [(i, j) for i in range(len(sh)) for j in range(i, len(sh))
            if tuple(sorted((sh[i].l, sh[j].l))) == tuple(cls)]


def _owned(aux: Basis, pairs) -> np.ndarray:
    """`[3, n, n]`: every element the pairs write, both orders."""
    sh, off = aux.shells, aux.offsets
    own = np.zeros((3, aux.n_ao, aux.n_ao), bool)
    for i, j in pairs:
        a = slice(off[i], off[i] + sh[i].n_cart)
        b = slice(off[j], off[j] + sh[j].n_cart)
        own[:, a, b] = own[:, b, a] = True
    return own


def _check(aux: Basis, cls, scale_pq) -> None:
    M1, ran = run_ipmetric(aux, boys=boys, classes={cls})
    assert ran == (cls,)
    pairs = _pairs(aux, cls)
    own = _owned(aux, pairs)
    want = np.array(host_df.metric_derivative(aux, pq=pq, pairs=pairs))
    scale = float(np.abs(np.array(host_df.metric_derivative(
        aux, pq=scale_pq, pairs=pairs))).max())
    err = float(np.abs(M1 - want)[own].max())
    print(f"\n{ipmetric_name(cls)}: {err / scale / EPS:.2f} eps of scale")
    assert err <= TOL * scale, f"{err / scale / EPS:.1f} eps of scale"
    assert np.all(M1[~own] == 0.0)
    # the mirror is a negation: bitwise on an off-diagonal pair's block
    sh, off = aux.shells, aux.offsets
    for i, j in pairs:
        if i == j:
            continue
        a = slice(off[i], off[i] + sh[i].n_cart)
        b = slice(off[j], off[j] + sh[j].n_cart)
        assert np.array_equal(M1[:, a, b], -M1[:, b, a].transpose(0, 2, 1))


# --- the host -------------------------------------------------------------------


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c)) for c in LOW])
def test_one_launch_over_the_class_is_the_host_derivative(cls) -> None:
    _check(AUX, cls, pq)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c))
                                 for c in HIGH])
def test_every_class_above_l4_is_the_host_derivative(cls) -> None:
    _check(_metric_basis(cls), cls, pq_tower)


def test_the_host_is_antisymmetric_and_libcint_signed() -> None:
    """`M1[k, Q, P] = -M1[k, P, Q]` by two independent sums (each centre by
    its own Cartesian rule), and the sign is the nuclear derivative's,
    `-int2c2e_ip1`: held to the golden on the s and p fitting shells."""
    sub, rows = _sp(GOLDEN[0])
    M1 = np.array(host_df.metric_derivative(sub, pq=pq))
    assert np.abs(M1 + M1.transpose(0, 2, 1)).max() <= 64 * EPS * np.abs(
        M1).max()
    for k, name in enumerate(GOLDEN):
        golden.compare(name, M1[k], index=(rows, rows))


def test_a_subset_launch_leaves_every_other_element_untouched() -> None:
    sentinel = np.full((3, AUX.n_ao, AUX.n_ao), 7.25)
    M1, _ = run_ipmetric(AUX, boys=boys, classes={(0, 1)}, M1=sentinel.copy())
    own = _owned(AUX, _pairs(AUX, (0, 1)))
    assert np.all(M1[~own] == 7.25) and not np.any(M1[own] == 7.25)


# --- libcint: -int2c2e_ip1 over cc-pVDZ-JKFIT ---------------------------------


@cache
def _sp(name: str) -> tuple[Basis, list[int]]:
    """`name`'s fitting shells with l <= 1, and their AOs' indices."""
    aux = golden.basis(name)
    keep = [k for k, s in enumerate(aux.shells) if s.l <= 1]
    rows = [aux.offsets[k] + a for k in keep
            for a in range(aux.shells[k].n_cart)]
    return Basis(tuple(aux.shells[k] for k in keep)), rows


def test_the_kernel_over_the_s_and_p_fitting_shells_is_libcints() -> None:
    sub, rows = _sp(GOLDEN[0])
    M1, ran = run_ipmetric(sub, boys=boys)
    assert ran == ((0, 0), (0, 1), (1, 1))
    for k, name in enumerate(GOLDEN):
        err = golden.compare(name, M1[k], index=(rows, rows))
        print(f"\n{name} (s, p): {err:.2e} of max")


@pytest.mark.slow  # the host through (f|f) on 131 aux functions
def test_the_host_over_cc_pvdz_jkfit_is_libcints() -> None:
    aux = golden.basis(GOLDEN[0])
    M1 = np.array(host_df.metric_derivative(aux, pq=pq))
    for k, name in enumerate(GOLDEN):
        err = golden.compare(name, M1[k])
        print(f"\n{name}: {err:.2e} of max")


@pytest.mark.slow  # every class through (f|f) interpreted, 131 functions
def test_the_kernel_over_cc_pvdz_jkfit_is_libcints() -> None:
    aux = golden.basis(GOLDEN[0])
    M1, ran = run_ipmetric(aux, boys=boys)
    assert max(map(sum, ran)) == 6
    for k, name in enumerate(GOLDEN):
        err = golden.compare(name, M1[k])
        print(f"\n{name}: {err:.2e} of max")


# --- libcint through (i|i): the aux-i witness, contracted ----------------------


@cache
def _witness():
    wit = manifest.df_gradient()
    ts = wit.three_center
    aux = ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return wit, aux, nuclei


def _contract(M1: np.ndarray, aux: Basis, nuclei, W: np.ndarray, cls
              ) -> np.ndarray:
    """`G[A, k] = SUM_{X on A} SUM_Y (W_XY + W_YX) M1[k, X, Y]` over the
    class's elements (module docstring)."""
    Wo = np.where(_owned(aux, _pairs(aux, cls))[0], W, 0.0)
    per = np.einsum("kxy,xy->xk", M1, Wo + Wo.T)
    atom = np.repeat(shell_atoms(aux, nuclei),
                     [s.n_cart for s in aux.shells])
    G = np.zeros((len(nuclei), 3))
    np.add.at(G, atom, per)
    return G


def _big(grads) -> float:
    return max(float(np.abs(g).max()) for g in grads.values())


def test_the_contraction_is_the_gradient_s_own() -> None:
    """The identity the libcint check rests on, held to `gmetric_` (the
    contracted kernel) on test_kernel_gmetric.py's basis."""
    from df_grad_launch import run_metric
    from test_kernel_gmetric import NUCLEI
    W = np.random.default_rng(3).standard_normal((AUX.n_ao, AUX.n_ao))
    for cls in [(0, 1), (1, 1), (1, 2)]:
        M1, _ = run_ipmetric(AUX, boys=boys, classes={cls})
        G, _ = run_metric(AUX, NUCLEI, W, boys=boys, classes={cls})
        got = _contract(M1, AUX, NUCLEI, W, cls)
        assert np.abs(got - G).max() <= 64 * EPS * np.abs(G).max(), cls


@pytest.mark.slow  # the host over every class of the aux-i set
def test_the_host_is_libcints_at_every_ladder_class() -> None:
    wit, aux, nuclei = _witness()
    assert sorted(wit.gmetric) == LADDER
    big = _big(wit.gmetric)
    worst = 0.0
    for cls in LADDER:
        M1 = np.array(host_df.metric_derivative(aux, pq=pq,
                                                pairs=_pairs(aux, cls)))
        g = _contract(M1, aux, nuclei, wit.w(), cls)
        worst = max(worst, float(np.abs(g - wit.gmetric[cls]).max()) / big)
    print(f"\nhost vs libcint through (i|i): {worst:.2e} of max")
    assert worst <= LIBCINT_TOL


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c)) for c in
                                 [(0, 0), (1, 1), (3, 4), (5, 6), (6, 6)]])
def test_the_kernel_is_libcints_on_the_aux_i_set(cls) -> None:
    wit, aux, nuclei = _witness()
    M1, ran = run_ipmetric(aux, boys=boys, classes={cls})
    assert ran == (cls,)
    g = _contract(M1, aux, nuclei, wit.w(), cls)
    err = float(np.abs(g - wit.gmetric[cls]).max()) / _big(wit.gmetric)
    print(f"\n{ipmetric_name(cls)} vs libcint: {err:.2e} of max")
    assert err <= LIBCINT_TOL


# --- the key and the kernel's shape ----------------------------------------------


def test_the_key_and_the_name() -> None:
    assert ipmetric_spec(1, 2).key() == (
        "coulomb/d1/(1,2)/b1/metric/uncontracted")
    assert ipmetric_name((1, 2)) == "ipmetric_12"
    assert build_ipmetric_kernel(ipmetric_spec(1, 2)).name == "ipmetric_12"
    assert ipmetric_spec(1, 2).kernel_body_levers() == (UNCONTRACTED,)


def test_the_form_is_refused_off_d1_and_off_the_mmd_metric() -> None:
    for order in (0, 2):
        with pytest.raises(ValueError, match="at derivative order"):
            Spec(Operator.COULOMB, order, (1, 2), Target.METRIC,
                 (UNCONTRACTED,))
    with pytest.raises(ValueError, match="#692"):
        Spec(Operator.COULOMB, 1, (1, 1, 2), Target.ERI3, (UNCONTRACTED,))
    with pytest.raises(ValueError, match="MMD engine only"):
        Spec(Operator.COULOMB, 1, (1, 2), Target.METRIC, (UNCONTRACTED,),
             engine=Engine.HGP)
    with pytest.raises(ValueError, match="lever 'uncontracted' on target s"):
        Spec(Operator.OVERLAP, 1, (1, 2), Target.S, (UNCONTRACTED,))


def test_each_builder_refuses_the_others_key() -> None:
    plain = Spec(Operator.COULOMB, 1, (1, 2), Target.METRIC)
    with pytest.raises(ValueError, match="exactly the lever"):
        build_ipmetric_kernel(plain)
    with pytest.raises(ValueError, match="not a d1 Target.METRIC key"):
        build_ipmetric_kernel(Spec(Operator.COULOMB, 0, (1, 2),
                                   Target.METRIC))
    with pytest.raises(ValueError, match="take no levers"):
        build_gmetric_kernel(ipmetric_spec(1, 2))


def test_the_parameters_and_the_one_output() -> None:
    k = build_ipmetric_kernel((1, 1))
    assert [p.name for p in k.params] == [
        "n_pair", "n_rec", "n_aux", "A", "alpha", "s_aux", "p_rec0",
        "p_rec1", "q_rec0", "q_rec1", "off_p", "off_q", OUT]
    (out,) = [q for q in k.params if q.intent is Intent.OUT]
    assert out.name == OUT and len(out.shape) == 3


def test_plain_stores_no_atomics_and_the_lever_s_provenance() -> None:
    nodes = list(walk(build_ipmetric_kernel((1, 2)).body))
    assert not any(isinstance(s, AtomicAdd | Raw) for s in nodes)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert lever_provenance(UNCONTRACTED) in notes


def test_the_classes_and_launch_are_the_d0_metric_s() -> None:
    assert {(c.lP, c.lQ) for c in metric_classes(AUX)} >= set(
        c for c in LOW if max(c) <= 4)
    assert launch_shape(65, 32) == (3, (32, 1))
