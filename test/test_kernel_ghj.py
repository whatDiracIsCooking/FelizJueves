"""`kernel_ghj`: the Coulomb-energy gradient on the HGP recurrence
(`Target.J` on `Engine.HGP` at d1), interpreted.

THE WITNESS is `gradient.host.coulomb_gradient` -- the MMD route (the bra
`E` tables one index up against `J_p` one Hermite order up), another
recurrence and another contraction, held to finite differences of
`eri.j_coulomb.j.coulomb_matrix` (test/test_gradient_host.py) -- and, whole,
PySCF's `grad:j` (#550's fixtures, `manifest.gradient_fixture`).

- PER CLASS GROUP, every ordered class with each shell `l <= 2` on a
  three-atom basis (an s of two primitives, a p and a d on one atom, an s
  and a p on another, a d on the last): the kernel's deposits summed over
  the ordered classes of one unordered group `{la, lb} x {lc, ld}`
  against the host restricted to it -- its bra pairs (`pairs=`, both
  orders through the host's pair weights) and its kets (the unique ket
  pairs of `{lc, ld}` at weight 2 off the diagonal, `ao.density.make_kets`)
  -- which is exactly that set of ordered quartets.
- WHOLE, the s, p pair of test_gradient_host.py against the host, and
  H2O/STO-3G, H2O/STO-3G plus a d primitive and H2O/6-31G against PySCF's
  `grad:j` through `gradient.interpreted.interpreted_runners(coulomb="hgp")`
  (`slow`: the raised classes are above `FAST_L_MAX`).

THE TOLERANCE. `|G_kernel - G_host| <= TOL scale`, `TOL = 1e-12`,
test/test_hgp_eri.py's per-ERI cross-engine bound (HGP against MMD), here
carried through the linear contraction; `scale` is the host's per-shell
contribution (`atoms=`, each shell its own row) in absolute value, folded
back onto the atoms, so a sum whose two centres cancel on one atom is not
measured against its own small result. Against PySCF, test_rhf_gradient's
`ABS_TOL = 1e-10` absolute.

THE L CUT: a class is `slow` when its RAISED total, `la + lb + lc + ld +
1`, is above `FAST_L_MAX` -- the VRR it builds is one order up.
"""

from __future__ import annotations

from functools import cache
from itertools import product

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_gradient_host import PAIR_SP
from tiers import by_l

from felizjueves.ao.density import checked_density, make_kets
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.hjk import kernel_hj as HJ
from felizjueves.gradient import host, rhf
from felizjueves.gradient import kernel_ghj as GH
from felizjueves.gradient.interpreted import (
    ghj_class_gradient,
    ghj_gradient,
    interpreted_runners,
)
from felizjueves.hgp import hrr, vrr
from felizjueves.ir import AtomicAdd, Intent, Param, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Engine, Operator, Spec, Target

TOL = 1e-12
ABS_TOL = 1e-10

_A, _B, _C = (0.0, 0.1, -0.2), (1.4, -0.3, 0.5), (-0.6, 1.1, 0.8)
SPD = (Basis((Shell(0, _A, (1.3, 0.4), (0.6, 0.5)),
              Shell(1, _A, (0.8,), (0.9,)), Shell(2, _A, (0.7,), (1.0,)),
              Shell(0, _B, (0.9,), (0.7,)), Shell(1, _B, (0.6,), (1.1,)),
              Shell(2, _C, (0.5,), (0.8,)))),
       ((2.0, _A), (1.0, _B), (3.0, _C)))
_UNORDERED = [(a, b) for a in range(3) for b in range(a, 3)]
GROUPS = [by_l((bra, ket), sum(bra) + sum(ket) + 1,
               id=f"{bra[0]}{bra[1]}|{ket[0]}{ket[1]}")
          for bra, ket in product(_UNORDERED, _UNORDERED)]


def _density(n: int) -> np.ndarray:
    X = np.random.default_rng(5).standard_normal((n, n))
    return X @ X.T / n


# ---------------------------------------------------------------------------
# The key, the name, the refusals
# ---------------------------------------------------------------------------


def test_the_key_is_the_d0_hgp_j_key_at_d1() -> None:
    s = GH.ghj_spec(1, 0, 0, 0)
    assert (s.target, s.engine, s.derivative_order, s.am, s.levers) == (
        Target.J, Engine.HGP, 1, (1, 0, 0, 0), ())
    assert s.key() == "coulomb/d1/(1,0|0,0)/b1/j.hgp/literal"
    assert s.key().replace("/d1/", "/d0/") == HJ.hj_class_spec(1, 0, 0, 0).key()
    assert GH.ghj_name(s) == "ghj_1000" != GH.ghj_name((0, 1, 0, 0))
    assert GH.ghj_name((0, 11, 0, 0)) == "ghj_0_11_0_0"


def test_mmd_j_and_rys_j_k_stay_refused_at_d1() -> None:
    # HGP K at d1 is legal since gradient.kernel_ghk (test_kernel_ghk.py)
    with pytest.raises(ValueError, match="derivative order 1"):
        Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.J)
    with pytest.raises(ValueError, match="derivative order 1"):
        Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.K, engine=Engine.RYS)
    with pytest.raises(ValueError, match="derivative order 1"):
        Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.J, engine=Engine.RYS)
    with pytest.raises(ValueError, match="derivative order 2"):
        Spec(Operator.COULOMB, 2, (1, 0, 0, 0), Target.J, engine=Engine.HGP)


def test_the_builder_refuses_another_key_and_any_lever() -> None:
    with pytest.raises(ValueError, match="derivative order 1 only"):
        GH.build_ghj_kernel(HJ.hj_class_spec(0, 0, 0, 0))
    lever = Spec(Operator.COULOMB, 1, (0, 0, 0, 0), Target.J, ("ordered",),
                 engine=Engine.HGP)
    with pytest.raises(ValueError, match="lever 'ordered'"):
        GH.build_ghj_kernel(lever)
    with pytest.raises(TypeError):
        GH.build_ghj_kernel((0, 0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 1, 0),
                                 (1, 1, 0, 1)])
def test_the_recurrences_are_the_four_hrrs_on_one_vrr(cls) -> None:
    la, lb, lc, ld = cls
    roles = GH.hrr_classes(cls)
    assert roles["a+"] == (la + 1, lb, lc, ld)
    assert roles["b+"] == (la, lb + 1, lc, ld)
    assert ("a-" in roles) == (la > 0) and ("b-" in roles) == (lb > 0)
    dag = GH.vrr_dag(cls)
    # the VRR holds every HRR's boundary at m = 0, and is closed
    for c in roles.values():
        assert {(e, f, 0) for e, f in hrr.build(c).boundary} <= dag.nodes
    for n in dag.nodes:
        assert all(t.child in dag.nodes for t in vrr.reduce(n))
    assert max(n[2] for n in dag.nodes) == sum(cls) + 1


def test_the_kernel_takes_no_screen_and_adds_six_per_quartet() -> None:
    k = GH.build_ghj_kernel((1, 1, 1, 0))
    names = tuple(p.name for p in k.params)
    assert names == GH.params_of((1, 1, 1, 0))
    assert not {"tau", "Dsh", "G_ab", "G_cd", "CD"} & set(names)
    assert "CD" in GH.params_of((0, 0, 0, 1))
    G = next(p for p in k.params if p.name == GH.G_OUT)
    assert isinstance(G, Param) and G.intent is Intent.OUT
    adds = [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 and {a.dst.array for a in adds} == {GH.G_OUT}


# ---------------------------------------------------------------------------
# Per class group, against the host restricted to it
# ---------------------------------------------------------------------------


def _ordered(ls: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted({ls, ls[::-1]})


@cache
def _kets(ket: tuple[int, int]):
    basis, _ = SPD
    D = _density(basis.n_ao)
    sh = basis.shells
    pairs = [(c, d, 1.0 if c == d else 2.0) for c in range(len(sh))
             for d in range(c, len(sh))
             if sorted((sh[c].l, sh[d].l)) == list(ket)]
    return make_kets(basis, checked_density(basis, D), pairs,
                     pq=lambda *a: 0.0, tau=0.0)


def _host(bra, ket, atoms=None) -> np.ndarray:
    basis, nuclei = SPD
    sh = basis.shells
    pairs = [(i, j) for i in range(len(sh)) for j in range(i, len(sh))
             if sorted((sh[i].l, sh[j].l)) == list(bra)]
    return np.array(host.coulomb_gradient(
        basis, _density(basis.n_ao), nuclei, one_centre=one_centre,
        pairs=pairs, atoms=atoms, kets=_kets(ket)))


def _scale(bra, ket) -> float:
    basis, nuclei = SPD
    nat = len(nuclei)
    real = host.shell_atoms(basis, nuclei)
    rows = tuple(nat + i for i in range(len(basis.shells)))
    g = np.abs(_host(bra, ket, rows))
    sc = np.zeros((nat, 3))
    for i, row in enumerate(rows):
        sc[real[i]] += g[row]
    return float(sc.max())


@pytest.mark.parametrize("group", GROUPS)
def test_each_class_group_is_the_host_restricted_to_it(group) -> None:
    bra, ket = group
    basis, nuclei = SPD
    D = _density(basis.n_ao)
    classes = [b + k for b in _ordered(bra) for k in _ordered(ket)]
    G = sum(ghj_class_gradient(basis, nuclei, D, c, boys=boys)
            for c in classes)
    want = _host(bra, ket)
    sc = _scale(bra, ket)
    assert sc > 0.0, group
    err = float(np.abs(G - want).max())
    assert err <= TOL * sc, f"{group}: {err / sc:.2e} of scale"


# ---------------------------------------------------------------------------
# Whole, against the host and against PySCF
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_whole_gradient_of_an_sp_pair_is_the_hosts() -> None:
    basis, nuclei = PAIR_SP
    D = _density(basis.n_ao)
    G = ghj_gradient(basis, nuclei, D, boys=boys)
    want = np.array(host.coulomb_gradient(basis, D, nuclei,
                                          one_centre=one_centre))
    assert np.abs(G - want).max() <= TOL * np.abs(want).max()
    assert np.abs(G.sum(axis=0)).max() <= TOL * np.abs(G).max()


@cache
def _fixture(name: str):
    fx = manifest.gradient_fixture(name)
    nuclei = tuple((float(x.charge), x.coord) for x in fx.atoms)
    return ob.from_fixture(fx), nuclei, fx.density, fx.arrays["grad:j"]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["h2o_sto3g_grad", "h2o_sto3g_d_grad",
                                  "h2o_631g_grad"])
def test_the_hgp_coulomb_term_is_pyscfs(name: str) -> None:
    basis, nuclei, D, want = _fixture(name)
    run = interpreted_runners(boys=boys, coulomb="hgp")
    G = rhf.SIGN["coulomb"] * np.asarray(run.coulomb(basis, nuclei, D))
    err = float(np.abs(G - want).max())
    assert err <= ABS_TOL, f"{name}: {err:.2e} from PySCF"


def test_the_runners_refuse_an_unknown_coulomb_engine() -> None:
    with pytest.raises(ValueError, match="coulomb"):
        interpreted_runners(boys=boys, coulomb="rys")
