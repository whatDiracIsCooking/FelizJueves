"""`kernel_rk.py`: the split exchange kernel on Rys quadrature, interpreted.

The Rys kernel is `kernel_hk`'s split `K` with the primitive nest's engine
part replaced (the Rys rule and the 2D VRR into the HGP boundary;
`kernel_hk.split_k_kernel` with `kernel_rj.rys_nest`), so it is held to the
HGP kernel from both ends, on test/test_kernel_hk.py's bases, records,
launches and witnesses -- test/test_kernel_rj.py's checks, for `K`:

- THE SEAM. Outside the `K_CD` primitive loop the two kernels are the SAME
  IR, in both class modes: the same parameters, staging, screen, HRR, tap
  and deposit. Inside it, the same statements as `rj_`'s nest for the same
  built class.
- THE INTEGRALS. The ERI tap `g` against `eri/hgp_drive.py` (HGP) and
  `eri/eri.py` (MMD) on the same built-order shells, element by element
  within `TOL (s_hgp + s_mmd)` (test_kernel_rj.py's `check_tap`). Not
  bitwise: the boundary is a root sum, not a Boys ladder.
- THE DEPOSIT. `K` against `eri/k_exchange/k.py`'s `k_quartet` over the
  class's ordered quadruples (test_kernel_hk.py's `reference`, MMD), within
  `TOL (s_hgp + s_mmd)`, every element off the class's blocks exactly 0;
  the two class modes against each other; a wrong density block failing;
  the AO-order trap; a whole molecule from `launch_hjk`'s plan (screen
  `"k"`), walked by `walk_hjk` with this kernel injected (`build=`),
  against the full MMD sweep and libcint's `vk` -- test_hjk_build.py's
  checks at its tolerances.

The rule is injected as `rys=rys.roots.roots_weights`. Every class through
`L = 4` (all 70 ordered ones, K's `(la, lc, lb, ld)`) is fast; the classes
`5 <= L <= 8` with every shell `l <= 2` are `slow`. No compiler, no card,
no emission.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
import test_hjk_build as HB
import test_kernel_hk as THK
import test_kernel_rj as TRJ
from test_kernel_hj import basis_for, density
from tiers import split_by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.hjk import kernel_hk as HK
from felizjueves.eri.hjk import kernel_rj as RJ
from felizjueves.eri.hjk import kernel_rk as RK
from felizjueves.eri.hjk import walk_hjk as W
from felizjueves.ir import Boys, For, Note, Rys, walk
from felizjueves.ir.interp import InterpError
from felizjueves.rys.roots import T_SWITCH, roots_weights
from felizjueves.spec import Engine, Operator, Spec, Target

MODES = THK.MODES
TOL = THK.TOL
HIGH = TRJ.HIGH  # every K class 5 <= L <= 8 with every shell l <= 2
CLASSES = THK.CLASSES
LADDER = split_by_l([*CLASSES, *HIGH], sum, ids=lambda c: "".join(map(str, c)))
_outside_the_nest = TRJ._outside_the_nest


def _spec(cls, mode) -> Spec:
    return RK.rk_class_spec(*cls, ordered=mode == "ordered")


@cache
def _kernel(cls, mode, eris=False, density=HK.K_DENSITY):
    return RK.build_rk_kernel(_spec(cls, mode), eris=eris, density=density)


def launch(cls, mode, basis, D, **kw):
    """test_kernel_hk.py's launch on the Rys kernel, the rule injected."""
    return THK.launch(cls, mode, basis, D, kernel=_kernel, rys=roots_weights,
                      **kw)


def kernel_k(cls, mode, **kw):
    """test_kernel_hk.py's `kernel_k` (the kernel's `K` and the HGP twin of
    the orientation it ran) on the Rys kernel."""
    basis = basis_for(cls)
    D = density(basis.n_ao)
    K, g, bras, kets, gidx, _ = launch(cls, mode, basis, D, **kw)
    s = np.zeros_like(K)
    for b in bras:
        for k in kets:
            xs = (*b, *k)
            THK.deposit_scale(basis, D, xs, gidx, THK.hgp(basis, xs)[1], s)
    return K, s, g, bras, kets, gidx


# ---------------------------------------------------------------------------
# The key, the names, the refusals
# ---------------------------------------------------------------------------


def test_the_key_is_k_on_the_rys_engine_and_the_names_are_hks() -> None:
    s = RK.rk_class_spec(1, 0, 1, 1)
    assert (s.target, s.engine) == (Target.K, Engine.RYS)
    assert s.key() == "coulomb/d0/(1,0|1,1)/b1/k.rys/literal"
    o = RK.rk_class_spec(1, 0, 1, 1, ordered=True)
    assert o.key() == "coulomb/d0/(1,0|1,1)/b1/k.rys/ordered"
    for cls in CLASSES:
        for mode in MODES:
            hk = HK.hk_name(THK._spec(cls, mode))
            assert RK.rk_name(_spec(cls, mode)) == "r" + hk[1:]
    assert RK.build_rk_kernel(s).name == "rk_" + HK.hk_name(
        HK.hk_class_spec(1, 0, 1, 1))[3:]
    assert RK.build_rk_kernel(o).name == "rk_o_1011"
    assert RK.hgp_twin(o) == HK.hk_class_spec(1, 0, 1, 1, ordered=True)
    # never a J unit's name: the J and K Rys ladders write side by side
    assert not {RK.rk_name(_spec(c, m)) for c in CLASSES for m in MODES} & {
        RJ.rj_name(RJ.rj_class_spec(*c, ordered=m == "ordered"))
        for c in CLASSES for m in MODES}


def test_the_builder_refuses_hgp_j_a_lever_a_batch_and_past_the_rule() -> None:
    with pytest.raises(ValueError, match="Engine.RYS"):
        RK.build_rk_kernel(HK.hk_class_spec(0, 0, 0, 0))
    with pytest.raises(ValueError, match="Engine.RYS"):
        RK.build_rk_kernel(RJ.rj_class_spec(0, 0, 0, 0))
    with pytest.raises(ValueError, match="N = 5"):
        RK.build_rk_kernel((3, 3, 2, 2))  # L = 10: N = 6
    with pytest.raises(ValueError, match="no-sieve"):
        RK.build_rk_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.K,
                                ("no-sieve",), engine=Engine.RYS))
    with pytest.raises(ValueError, match="n_batches"):
        RK.build_rk_kernel(Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.K,
                                n_batches=3, engine=Engine.RYS))
    with pytest.raises(TypeError):
        RK.build_rk_kernel([0, 0, 0, 0])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The seam: kernel_hk's kernel outside the primitive nest, rj_'s inside
# ---------------------------------------------------------------------------


def _nest(body):
    """The `K_CD` loop's body."""
    (loop,) = [s for s in walk(body) if isinstance(s, For) and s.var == HK.KCD]
    return loop.body


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_outside_the_primitive_nest_it_is_kernel_hk(cls, mode) -> None:
    rk = _kernel(cls, mode, eris=True)
    hk = HK.build_hk_kernel(THK._spec(cls, mode), eris=True)
    assert rk.params == hk.params
    assert _outside_the_nest(rk.body) == _outside_the_nest(hk.body)
    assert not any(isinstance(s, Boys) for s in walk(rk.body))
    (rule,) = [s for s in walk(rk.body) if isinstance(s, Rys)]
    assert rule.n == sum(cls) // 2 + 1


@pytest.mark.parametrize("cls", split_by_l(
    [(0, 0, 0, 0), (1, 0, 2, 0), (1, 1, 1, 1), (2, 2, 1, 2)], sum, ids=str))
def test_inside_the_nest_it_is_the_rys_j_nest_of_the_built_class(cls) -> None:
    """The recurrence does not know the deposit: the `rk_o_` nest of class
    `x` is the `rj_o_` nest of the same built class `x`, statement for
    statement."""
    rk = _kernel(cls, "ordered")
    rj = RJ.build_rj_kernel(RJ.rj_class_spec(*cls, ordered=True))
    assert _nest(rk.body) == _nest(rj.body)


def test_the_head_note_names_k_and_the_rule() -> None:
    k = _kernel((1, 0, 1, 1), "ordered")
    head = k.body[0]
    assert isinstance(head, Note)
    assert head.prov.startswith("(x0 x1|x2 x3) by Rys quadrature, K class "
                                "(10|11) [ordered")
    assert "N = 2 nodes" in head.prov and "K_ab += D_cd (ac|bd)" in head.prov


def test_a_rys_without_its_rule_injected_raises() -> None:
    cls = (0, 0, 0, 0)
    basis = basis_for(cls)
    with pytest.raises(InterpError, match="rys="):
        THK.launch(cls, "ordered", basis, density(basis.n_ao),
                   kernel=_kernel)


# ---------------------------------------------------------------------------
# The integrals: against HGP and MMD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_the_tap_meets_hgp_and_mmd(cls, mode) -> None:
    basis = basis_for(cls)
    _, g, bras, kets, _, _ = launch(cls, mode, basis, density(basis.n_ao))
    worst = TRJ.check_tap(cls, basis, g, bras, kets)
    print(f"{cls} {mode}: worst {worst:.3g} of s_hgp + s_mmd against MMD")


@pytest.mark.parametrize("cls", split_by_l([(0, 0, 1, 1), (1, 1, 2, 2)], sum,
                                            ids=str))
def test_far_apart_pairs_take_the_large_t_branch(cls) -> None:
    """The built ket's shells moved 20 bohr away: every primitive quartet's
    `T` is past `T_SWITCH` (the Laguerre branch, recorded), and the tap
    still meets HGP and MMD."""
    assert not set(cls[:2]) & set(cls[2:])
    far = Basis(tuple(
        Shell(s.l, (s.centre[0], s.centre[1], s.centre[2] + 20.0),
              s.exponents, s.coefficients) if s.l in cls[2:] else s
        for s in basis_for(cls).shells))
    seen: list[float] = []

    def rule(n, T):
        seen.append(T)
        return roots_weights(n, T)

    _, g, bras, kets, _, _ = THK.launch(cls, "ordered", far,
                                        density(far.n_ao), kernel=_kernel,
                                        rys=rule)
    assert seen and min(seen) >= T_SWITCH
    TRJ.check_tap(cls, far, g, bras, kets)


# ---------------------------------------------------------------------------
# The deposit: against eri/k_exchange/k.py's quartets, the modes, a molecule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_the_deposit_is_ao_ks_quartets_for_the_class(cls, mode) -> None:
    want, s_mmd, mask = THK.reference(cls)
    K, s_hgp, *_ = kernel_k(cls, mode)
    assert np.all(K[~mask] == 0.0)
    err = np.abs(K - want)
    bound = TOL * (s_hgp + s_mmd)
    assert np.all(err[mask] <= bound[mask]), float(np.max(err / (bound + 1e-300)))
    assert np.max(np.abs(want[mask])) > 1e-3


@pytest.mark.parametrize("cls", split_by_l(CLASSES, sum,
                                            ids=lambda c: "".join(map(str, c))))
def test_canonical_and_ordered_agree(cls) -> None:
    Kc, sc, *_ = kernel_k(cls, "canonical")
    Ko, so, *_ = kernel_k(cls, "ordered")
    assert np.all(np.abs(Kc - Ko) <= TOL * (sc + so))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", [(0, 1, 0, 1), (1, 0, 0, 1), (1, 1, 2, 0)],
                         ids=str)
def test_a_wrong_density_block_fails(cls, mode) -> None:
    """test_kernel_hk.py's mutation on the Rys kernel: `K_ab` reading J's
    block `(2, 3)` or the bra pair's `(0, 1)` misses `eri/k_exchange/k.py`
    by far more than the tolerance."""
    want, s_mmd, mask = THK.reference(cls)
    for wrong in ((2, 3), (0, 1)):
        K, s_hgp, *_ = kernel_k(cls, mode, density_=wrong)
        excess = np.abs(K - want)[mask] / (TOL * (s_hgp + s_mmd))[mask]
        assert np.max(excess) > 1e3, (wrong, float(np.max(excess)))


@pytest.mark.parametrize("ao", THK.TRAP, ids=str)
def test_an_ao_order_am_builds_a_different_wrong_class(ao) -> None:
    """test_kernel_hk.py's AO-order trap on the Rys kernel: `(la, lb, lc,
    ld)` spelled as K's `am`, `lb != lc`, misses the class far beyond the
    bound and deposits into other blocks, in both modes, while the Hermite
    spelling passes."""
    la, lb, lc, ld = ao
    herm = (la, lc, lb, ld)
    assert lb != lc
    want, s_mmd, mask = THK.reference(herm)
    basis = basis_for(herm)
    D = density(basis.n_ao)
    for mode in MODES:
        Kw, *_ = launch(ao, mode, basis, D, eris=False)
        Kr, s_hgp, *_ = kernel_k(herm, mode)
        bound = TOL * (s_hgp + s_mmd)
        assert np.all(np.abs(Kr - want)[mask] <= bound[mask])
        excess = np.abs(Kw - want)[mask] / bound[mask]
        assert np.max(excess) > 1e3, (mode, float(np.max(excess)))
        assert np.any(Kw[~mask] != 0.0)


def test_the_screen_skips_one_ket_and_never_breaks() -> None:
    """The middle ket's `G_cd = 0` skips it (its tap row unwritten), the
    rest computed."""
    cls = (0, 1, 1, 1)
    basis = basis_for(cls)
    D = density(basis.n_ao)
    _, g, bras, kets, _, it = launch(cls, "canonical", basis, D, tau=1e-14,
                                     G={1: 0.0})
    _, gf, *_ = launch(cls, "canonical", basis, D)
    for ib in range(len(bras)):
        for ik in range(len(kets)):
            if ik == 1:
                assert np.all(g[ib, ik] == -7.0)
            else:
                assert np.array_equal(g[ib, ik], gf[ib, ik])
    assert it.iterations[HK.KET] == len(bras) * len(kets)
    assert it.iterations[HK.KAB] == len(bras) * (len(kets) - 1) * 4


BUILT: list[str] = []


def rys_of(spec: Spec):
    """`walk_hjk`'s `build=`: the Rys drop-in of the launch's HGP key."""
    k = RK.build_rk_kernel(replace(spec, engine=Engine.RYS))
    BUILT.append(k.name)
    return k


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", [
    "h2_sto3g", pytest.param("h2o_sto3g", marks=pytest.mark.slow)])
def test_a_whole_molecule_against_the_sweep_and_libcint(name, mode) -> None:
    """`launch_hjk`'s K plan at `tau = 0`, every launch on the Rys kernel
    (`walk_hjk(build=)`): the full ordered MMD sweep to `TOL S` and
    libcint's `vk` to `LIBCINT_TOL S` (test_hjk_build.py's `S`)."""
    basis, D, _, vk = HB.molecule(name)
    BUILT.clear()
    w = W.walk_hjk(HB.plan(name, 0.0, mode, "k"), basis, D, run=TRJ.run,
                   build=rys_of)
    assert len(BUILT) == len(w.kernels) > 0
    assert all(k.startswith("rk_") for k in BUILT)
    assert len(w.visited) == len(basis.shells) ** 4
    X = np.array(w.X)
    S = HB.sweep(name, "k", block="scale", mode=mode)
    want = HB.reference(name, "k")
    assert np.max(np.abs(want)) > 0.1
    host, lib = HB._worst(X, want, S), HB._worst(X, vk, S)
    print(f"{name} rys k {mode}: host {host:.3g} S, libcint {lib:.3g} S")
    assert host <= HB.TOL
    assert lib <= HB.LIBCINT_TOL
