"""`hessian.kernel_hgp2`: the two-electron Hessians on HGP (`d2f_`, `d2hj_`,
`d2hk_`), interpreted.

THE WITNESSES, none of them the HGP recurrence:

- WHOLE BASES, THE MMD HOSTS. `hessian.interpreted.hgp2_hessian` over every
  launch of `gf_launches` against `hessian.host.coulomb_hessian` (J),
  `exchange_hessian` (K) and `coulomb - 1/2 exchange` (F) -- each held to
  finite differences of the gradient hosts by test_hessian_host.py --
  within `TOL` of `max|H|`, on synthetic three-atom bases where shells
  coincide on an atom (so `a = b`, `c = d` and `P = Q` all occur): s only
  (fast), then with a p and a d shell (`slow`: the kernel builds the class
  two orders up, `(pp|pp) + 2`, `(dd|dd) + 2`; the d case reaches every
  ladder class through (dd|dd)).
- LIBCINT. On H2O/STO-3G the numeric goldens `hjab_h2o_sto3g_bse` and
  `hkx_h2o_sto3g_bse` (`int2e_ipip1` / `_ipvip1` / `_ip1ip2` contracted,
  `oracle.numeric`) for J and K, at their stored density (`slow`).
- STRUCTURE. `H = H^T` and no net force on any whole basis (the kernel
  differentiates all four centres and assumes no translational
  invariance), and a wrong weight is seen.

Interpreter only: no compiler, no card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import numeric as golden

from felizjueves.ao.shell import Basis, Shell
from felizjueves.gradient.interpreted import gf_launches
from felizjueves.gradient.kernel_gf import GradWeights
from felizjueves.hessian import host
from felizjueves.hessian import kernel_hgp2 as H2
from felizjueves.hessian.interpreted import hgp2_hessian, hgp2_run
from felizjueves.ir import AtomicAdd, Raw, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import (
    HESSIAN_OFF_NATIVE,
    HESSIAN_TARGETS,
    Engine,
    Operator,
    Spec,
    Target,
)

TOL = 1e-13  # of max|H|; measured <= 1.8e-14 (F, the d basis)
NET_TOL = 1e-12
TARGETS = H2.HGP2_TARGETS
_NUC = ((1.0, (0.0, 0.0, 0.0)), (3.0, (0.3, -0.2, 1.4)),
        (2.0, (-0.9, 0.5, 0.2)))


def _synthetic(l2: int) -> Basis:
    return Basis((Shell(0, _NUC[0][1], (1.3, 0.4), (0.5, 0.6)),
                  Shell(0, _NUC[0][1], (0.25,), (0.7,)),
                  Shell(l2, _NUC[1][1], (0.9, 0.3), (0.4, 0.5)),
                  Shell(0, _NUC[2][1], (0.8,), (0.9,))))


def _sym(n: int, seed: int) -> np.ndarray:
    m = np.random.default_rng(seed).standard_normal((n, n))
    return (m + m.T) / 2


@cache
def _hosts(l2: int) -> tuple[np.ndarray, np.ndarray]:
    basis = _synthetic(l2)
    D = _sym(basis.n_ao, 3)
    n = len(_NUC)
    J, K = (np.array(f(basis, D, _NUC, one_centre=one_centre)).reshape(
        n, 3, n, 3) for f in (host.coulomb_hessian, host.exchange_hessian))
    return J, K


def _want(target: Target, l2: int) -> np.ndarray:
    J, K = _hosts(l2)
    return {Target.J: J, Target.K: K, Target.F: J - K / 2}[target]


# --- the weights, the keys, the names -----------------------------------------


def test_the_weights_are_derived_from_the_orbit() -> None:
    assert H2.d2_weights(Target.F) == GradWeights(4.0, -1.0)
    assert H2.d2_weights(Target.J) == GradWeights(4.0, 0.0)
    assert H2.d2_weights(Target.K) == GradWeights(0.0, 2.0)
    with pytest.raises(ValueError, match="no HGP two-electron Hessian"):
        H2.d2_weights(Target.S)


def test_the_keys_are_d2_on_hgp() -> None:
    assert H2.d2f_spec(1, 0, 0, 0).key() == "coulomb/d2/(1,0|0,0)/b1/f/literal"
    assert (H2.d2hj_spec(1, 0, 0, 0).key()
            == "coulomb/d2/(1,0|0,0)/b1/j.hgp/literal")
    assert (H2.d2hk_spec(1, 0, 2, 1).key()
            == "coulomb/d2/(1,0|2,1)/b1/k.hgp/literal")
    assert Target.F in HESSIAN_TARGETS
    assert HESSIAN_OFF_NATIVE == {(Target.DIPOLE, Engine.HGP),
                                  (Target.QUADRUPOLE, Engine.HGP),
                                  (Target.METRIC, Engine.HGP),
                                  (Target.ERI3, Engine.HGP),
                                  (Target.J, Engine.HGP),
                                  (Target.K, Engine.HGP)}
    with pytest.raises(ValueError, match="derivative order 2 on target j"):
        Spec(Operator.COULOMB, 2, (0, 0, 0, 0), Target.J)


def test_the_builder_refuses_what_it_does_not_build() -> None:
    with pytest.raises(ValueError, match="not a d2 F, J or K key"):
        H2.build_d2_kernel(Spec(Operator.COULOMB, 1, (0, 0, 0, 0), Target.F))
    with pytest.raises(ValueError, match="not a d2 F, J or K key"):
        H2.build_d2_kernel(Spec(Operator.COULOMB, 2, (0, 0), Target.JAB))
    with pytest.raises(TypeError, match="takes a Spec"):
        H2.build_d2_kernel((0, 0, 0, 0))
    with pytest.raises(TypeError, match="GradWeights"):
        H2.build_d2_kernel(H2.d2f_spec(0, 0, 0, 0), weights=(4.0, -1.0))
    with pytest.raises(ValueError, match="nothing to differentiate"):
        H2.build_d2_kernel(H2.d2f_spec(0, 0, 0, 0),
                           weights=GradWeights(0.0, 0.0))


def test_names_and_params() -> None:
    assert H2.d2_name(H2.d2f_spec(1, 0, 2, 1)) == "d2f_1021"
    assert H2.d2_name(H2.d2hj_spec(0, 0, 1, 1)) == "d2hj_0011"
    assert H2.d2_name(H2.d2hk_spec(10, 0, 0, 0)) == "d2hk_10_0_0_0"
    for t in TARGETS:
        s = H2.d2_spec(t, (1, 1, 0, 1))
        k = H2.build_d2_kernel(s)
        assert k.name == H2.d2_name(s)
        assert tuple(p.name for p in k.params) == H2.params_of(s)
        assert "bra_sh" in H2.params_of(s) and "ket_sh" not in H2.params_of(s)
    assert H2.launch_shape(5, 32) == (5, (32, 1))


def test_the_write_is_144_atomics_and_no_raw_text() -> None:
    assert len(H2.components()) == 78
    for t in TARGETS:
        for cls in ((0, 0, 0, 0), (1, 0, 1, 1)):
            k = H2.build_d2_kernel(H2.d2_spec(t, cls))
            stmts = list(walk(k.body))
            assert not any(isinstance(s, Raw) for s in stmts)
            assert sum(isinstance(s, AtomicAdd) for s in stmts) == 144


def test_the_rule_twice_is_the_closed_form() -> None:
    p1 = ((1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))  # a p_x on A
    up2 = ((3, 0, 0), *p1[1:])
    down0 = ((0,) * 3, *p1[1:])
    # dA_x dA_x: 4a^2 [+2] - 2a (2i + 1) [0] + i (i - 1) [-2], i = 1
    assert H2.second(p1, (0, 0, 0, 0)) == {((0, 0), up2): 1, ((0,), p1): -3}
    # dA_x dB_y on (p_x s|..): [A+x B+y]^{2a 2b} - 1 [A-x B+y]^{2b}
    up_ab = ((2, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0))
    dn_ab = ((0, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0))
    assert H2.second(p1, (0, 0, 1, 1)) == {((0, 1), up_ab): 1,
                                          ((1,), dn_ab): -1}
    assert ((), down0) not in H2.second(p1, (0, 0, 0, 0))
    # (ss|ss): the ten 2x 2y families' raised classes and, from the -(2t + 1)
    # term on a diagonal, each centre's 2x family on the class itself; the
    # unit family is never read (nothing to lower)
    terms = H2.derivative_terms((0, 0, 0, 0))
    assert len(terms) == 14
    assert {f for f, c in terms if len(f) == 1} == {(0,), (1,), (2,), (3,)}
    assert all(c == (0, 0, 0, 0) for f, c in terms if len(f) == 1)


# --- whole bases against the MMD hosts ----------------------------------------


def _whole(target: Target, l2: int) -> None:
    basis = _synthetic(l2)
    D = _sym(basis.n_ao, 3)
    got = hgp2_hessian(target, basis, _NUC, D, boys=boys)
    want = _want(target, l2)
    scale = np.abs(want).max()
    err = np.abs(got - want).max() / scale
    print(f"\n{target} l2={l2}: {err:.2e} of max|H| ({scale:.3g})")
    assert err <= TOL
    assert np.abs(got - got.transpose(2, 3, 0, 1)).max() <= NET_TOL * scale
    assert np.abs(got.sum(axis=0)).max() <= NET_TOL * scale


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_s_basis_is_the_host(target) -> None:
    _whole(target, 0)


@pytest.mark.slow
@pytest.mark.l_cut
@pytest.mark.parametrize("l2", (1, 2), ids=("p", "d"))
@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_p_and_d_bases_are_the_host(target, l2: int) -> None:
    _whole(target, l2)


def test_a_wrong_weight_is_seen() -> None:
    basis = _synthetic(0)
    D = _sym(basis.n_ao, 3)
    n = len(_NUC)
    H = np.zeros((n, 3, n, 3))
    for am, diag, pb, pk in gf_launches(basis, _NUC):
        s = H2.d2hj_spec(*am)
        H = hgp2_run(s, diag, pb, pk, n, D, H, boys=boys,
                     kernel=H2.build_d2_kernel(
                         s, weights=GradWeights(2.0, 0.0)))
    want = _want(Target.J, 0)
    assert np.abs(H - want / 2).max() <= TOL * np.abs(want).max()
    assert np.abs(H - want).max() > 0.1 * np.abs(want).max()


def test_an_asymmetric_density_is_refused() -> None:
    basis = _synthetic(0)
    D = _sym(basis.n_ao, 3)
    D[0, 1] += 1.0
    with pytest.raises(ValueError, match="not a symmetric"):
        hgp2_hessian(Target.F, basis, _NUC, D, boys=boys)


# --- H2O/STO-3G against libcint ------------------------------------------------

LIBCINT = {Target.J: "hjab_h2o_sto3g_bse", Target.K: "hkx_h2o_sto3g_bse"}


@pytest.mark.slow
@pytest.mark.l_cut
@pytest.mark.parametrize("target", sorted(LIBCINT), ids=str)
def test_h2o_sto3g_is_libcints(target) -> None:
    name = LIBCINT[target]
    nuclei = tuple((float(z), tuple(c)) for z, c in golden.atoms(name))
    basis, D = golden.basis(name), np.asarray(golden.density(name))
    H = hgp2_hessian(target, basis, nuclei, D, boys=boys)
    err = golden.compare(name, H.reshape(golden.want(name).shape))
    print(f"\n{target} {name}: {err:.2e} of max|H|")
