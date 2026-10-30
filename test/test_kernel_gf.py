"""`gradient.kernel_gf`: the fused Fock gradient (`Target.F` at d1, HGP),
interpreted.

THREE WITNESSES, none of them the HGP recurrence:

- PER CLASS, ANOTHER ENGINE. One unique quartet of four distinct shells on
  four distinct atoms, one launch (`diag = 0`, so `deg = 1`), against the
  derived weights contracted with derivative integrals from `eri.eri` (MMD
  through `reference.pq`): `d(ab|cd)/dA_k` is `eri_block` with the A shell
  raised to `l + 1`, its coefficients times `2 alpha`, minus `a_k` times
  the block with it lowered to `l - 1` -- the package docstring's rule,
  spelled on whole contracted shells. Every class of milestone 12's ladder
  in every orientation the stream launches, within `TOL = 1e-13` of the
  absolute-value twin of that sum; `slow` above `L + 1 = 4`
  (`tiers.by_l`). This holds the kernel's derivative and digestion, not
  the weights' derivation.
- WHOLE BASES, THE HOST. `gf_gradient` over every launch of `gf_launches`
  against `gradient.host`'s `coulomb_gradient - 1/2 exchange_gradient`
  (the MMD references, held to finite differences of libcint-witnessed
  `J` and `K` by test_gradient_host.py), within `TOL` of `max|G|`, on
  synthetic three-atom bases where shells coincide on an atom, so `a = b`,
  `c = d` and `P = Q` all occur: s only (fast), then with a p and with a d
  shell (`slow`). This holds the weights, `deg` and the canonical half.
- PYSCF. #550's gradient fixtures (`oracle.manifest.gradient_fixture`): the
  sum of their `grad:j` and `grad:k` terms, PySCF's two-electron gradient
  at its converged density, within `ABS_TOL = 1e-10` absolute (#565's
  bound) on H2O/STO-3G and H2O/STO-3G plus a d primitive on O (`slow`:
  `(pp|pp) + 1` and `(dd|dd) + 1`).

Every whole-basis `G` also carries no net force, to rounding: the kernel
differentiates all four centres and assumes no translational invariance,
so that is a check -- of structure only: it cannot see a wrong overall
factor, which the three witnesses above do.

Interpreter only: no compiler, no card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.emit import __main__ as emit_cli
from felizjueves.eri.eri import eri_block
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.gradient import host
from felizjueves.gradient import kernel_gf as GF
from felizjueves.gradient.interpreted import (
    gf_gradient,
    gf_launches,
    gf_pair,
    gf_run,
)
from felizjueves.ir import AtomicAdd, Raw, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, pq
from felizjueves.spec import GRADIENT_TARGETS, Engine, Operator, Spec, Target

TOL = 1e-13
ABS_TOL = 1e-10
NET_TOL = 1e-12
CLASSES = emit_cli.FOCK_CLASSES  # milestone 12's 21, through (dd|dd)
# Each class as the stream launches it AND transposed (bra and ket swapped),
# so every position carries every l the ladder reaches.
ORIENTED = sorted({c for a in CLASSES for c in (a, (*a[2:], *a[:2]))})

_NUCLEI = ((1.0, (0.0, 0.0, 0.0)), (3.0, (0.3, -0.2, 1.4)),
           (2.0, (-0.9, 0.5, 0.2)), (1.0, (0.4, 1.1, -0.6)))
_EXPS = ((1.3, 0.4), (0.9, 0.3), (0.8, 0.25), (1.1, 0.35))
_COEF = ((0.5, 0.6), (0.4, 0.5), (0.7, 0.3), (0.45, 0.55))


def _sym(n: int, seed: int) -> np.ndarray:
    m = np.random.default_rng(seed).standard_normal((n, n))
    return (m + m.T) / 2


# --- the weights, the key, the name -------------------------------------------


def test_the_weights_are_derived_from_the_orbit() -> None:
    assert GF.WEIGHTS == GF.GradWeights(coulomb=4.0, exchange=-1.0)
    assert GF.derive_gradient_weights() == GF.WEIGHTS
    # the energy's fractions are kernel_f's: G = J - K/2
    assert KF.COULOMB_FRACTION == 2 * GF.COULOMB_GAMMA
    assert KF.EXCHANGE_FRACTION == -4 * GF.EXCHANGE_GAMMA / 2


def test_the_key_is_f_at_order_one_on_hgp() -> None:
    s = GF.gf_spec(1, 0, 2, 1)
    assert s.key() == "coulomb/d1/(1,0|2,1)/b1/f/literal"
    assert s.engine is Engine.HGP and Target.F in GRADIENT_TARGETS
    assert s.key() == KF.f_class_spec(1, 0, 2, 1).key().replace("/d0/", "/d1/")
    with pytest.raises(ValueError, match="engine mmd on target f"):
        Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.F, engine=Engine.MMD)


def test_the_builder_refuses_what_it_does_not_build() -> None:
    with pytest.raises(ValueError, match="derivative order 1 only"):
        GF.build_gf_kernel(KF.f_class_spec(1, 0, 0, 0))
    with pytest.raises(ValueError, match="no lever"):
        GF.build_gf_kernel(Spec(Operator.COULOMB, 1, (0, 0, 0, 0), Target.F,
                                ("slim-fock",)))
    for bad in ((1, 0, 0), (1, 0, 0, -1), (1.0, 0, 0, 0)):
        with pytest.raises(ValueError, match="not four ints"):
            GF.build_gf_kernel(bad)
    with pytest.raises(TypeError, match="GradWeights"):
        GF.build_gf_kernel((0, 0, 0, 0), weights=KF.WEIGHTS)


def test_names_params_and_the_derivative_classes() -> None:
    assert GF.gf_name((1, 0, 2, 1)) == "gf_1021"
    assert GF.gf_name((10, 0, 0, 0)) == "gf_10_0_0_0"
    k = GF.build_gf_kernel((1, 1, 0, 1))
    assert k.name == "gf_1101"
    assert tuple(p.name for p in k.params) == GF.params_of((1, 1, 0, 1))
    assert "bra_sh" in GF.params_of((1, 1, 0, 1))
    assert "ket_sh" not in GF.params_of((1, 1, 0, 1))
    assert {"AB", "CD"} <= set(GF.params_of((0, 0, 0, 0)))
    assert GF.derivative_classes((1, 0, 2, 0)) == {
        0: ((2, 0, 2, 0), (0, 0, 2, 0)), 1: ((1, 1, 2, 0), None),
        2: ((1, 0, 3, 0), (1, 0, 1, 0)), 3: ((1, 0, 2, 1), None)}
    assert GF.launch_shape(5, 32) == (5, (32, 1))


def test_the_write_is_twelve_atomics_and_no_raw_text() -> None:
    for cls in ((0, 0, 0, 0), (1, 0, 1, 1)):
        k = GF.build_gf_kernel(cls)
        stmts = list(walk(k.body))
        assert not any(isinstance(s, Raw) for s in stmts)
        assert sum(isinstance(s, AtomicAdd) for s in stmts) == 12


# --- per class: one quartet against eri.eri's derivative integrals --------------


def _shells(cls) -> tuple[Shell, ...]:
    return tuple(Shell(lx, _NUCLEI[i][1], _EXPS[i], _COEF[i])
                 for i, lx in enumerate(cls))


def _moved(s: Shell, by: int) -> Shell:
    c = (tuple(2 * a * x for a, x in zip(s.exponents, s.coefficients,
                                         strict=True))
         if by > 0 else s.coefficients)
    return Shell(s.l + by, s.centre, s.exponents, c)


def _block(sh) -> np.ndarray:
    """`eri_block` as a 4-index array `[i, j, k, l]`."""
    b = np.array(eri_block((sh[0], sh[1]), (sh[2], sh[3]), pq=pq))
    return b.reshape([s.n_cart for s in sh])


def _quartet_reference(cls, D: np.ndarray, offs) -> tuple[np.ndarray, ...]:
    """`G[X, k] = SUM w d(ab|cd)/dX_k` and its absolute-value twin, `deg =
    1` (module docstring)."""
    sh = _shells(cls)
    sl = [slice(o, o + s.n_cart) for o, s in zip(offs, sh, strict=True)]
    w = (4 * np.einsum("ij,kl->ijkl", D[sl[0], sl[1]], D[sl[2], sl[3]])
         - np.einsum("ik,jl->ijkl", D[sl[0], sl[2]], D[sl[1], sl[3]])
         - np.einsum("il,jk->ijkl", D[sl[0], sl[3]], D[sl[1], sl[2]]))
    G, S = np.zeros((4, 3)), np.zeros((4, 3))
    for pos in range(4):
        up = _block([_moved(s, 1) if q == pos else s
                     for q, s in enumerate(sh)])
        lo = (_block([_moved(s, -1) if q == pos else s
                      for q, s in enumerate(sh)]) if cls[pos] else None)
        cu = {v: n for n, v in enumerate(cartesians(cls[pos] + 1))}
        cl = ({v: n for n, v in enumerate(cartesians(cls[pos] - 1))}
              if cls[pos] else {})
        for k in range(3):
            d = np.zeros(w.shape)
            a = np.zeros(w.shape)
            for idx in np.ndindex(*w.shape):
                v = cartesians(cls[pos])[idx[pos]]
                vu = tuple(c + (x == k) for x, c in enumerate(v))
                j = list(idx)
                j[pos] = cu[vu]
                d[idx] = up[tuple(j)]
                a[idx] = abs(up[tuple(j)])
                if v[k]:
                    vl = tuple(c - (x == k) for x, c in enumerate(v))
                    j[pos] = cl[vl]
                    d[idx] -= v[k] * lo[tuple(j)]
                    a[idx] += v[k] * abs(lo[tuple(j)])
            G[pos, k] = float(np.sum(w * d))
            S[pos, k] = float(np.sum(np.abs(w) * a))
    return G, S


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=GF.gf_name(c))
                                 for c in ORIENTED])
def test_each_class_is_the_derivative_integrals_contracted(cls) -> None:
    sh = _shells(cls)
    basis = Basis(sh)
    D = _sym(basis.n_ao, sum(cls))
    atoms = (0, 1, 2, 3)
    bra = GF.pack([gf_pair(basis, atoms, 0, 1)])
    ket = GF.pack([gf_pair(basis, atoms, 2, 3)])
    G = gf_run(cls, 0, bra, ket, 4, D, np.zeros((4, 3)), boys=boys)
    want, scale = _quartet_reference(cls, D, basis.offsets)
    assert np.max(np.abs(want)) > 0
    assert np.all(np.abs(G - want) <= TOL * np.maximum(scale, 1e-300))
    assert np.abs(G.sum(axis=0)).max() <= NET_TOL * np.abs(G).max()


def test_a_wrong_coulomb_weight_is_seen() -> None:
    """The per-class witness reads the weights the kernel is built with:
    substitute kernel_f's 2 (its in-order count) and the class fails."""
    cls = (0, 0, 0, 0)
    basis = Basis(_shells(cls))
    D = _sym(basis.n_ao, 7)
    kern = GF.build_gf_kernel(cls, weights=GF.GradWeights(2.0, -1.0))
    want, _ = _quartet_reference(cls, D, basis.offsets)
    atoms = (0, 1, 2, 3)
    bra = GF.pack([gf_pair(basis, atoms, 0, 1)])
    ket = GF.pack([gf_pair(basis, atoms, 2, 3)])
    G = gf_run(cls, 0, bra, ket, 4, D, np.zeros((4, 3)), boys=boys,
               kernel=kern)
    assert np.abs(G - want).max() > 1e-3 * np.abs(want).max()


# --- whole bases: the stream, the weights, deg and the canonical half -----------


def _synthetic(l2: int) -> tuple[Basis, tuple]:
    """Three atoms; two s shells on the first (so `a = b` and `P = Q` meet
    distinct shells on one atom), one shell of `l2` on the second, an s on
    the third."""
    nuc = _NUCLEI[:3]
    return Basis((Shell(0, nuc[0][1], (1.3, 0.4), (0.5, 0.6)),
                  Shell(0, nuc[0][1], (0.25,), (0.7,)),
                  Shell(l2, nuc[1][1], (0.9, 0.3), (0.4, 0.5)),
                  Shell(0, nuc[2][1], (0.8,), (0.9,)))), nuc


@cache
def _host(basis: Basis, nuclei, seed: int) -> tuple[np.ndarray, np.ndarray]:
    D = _sym(basis.n_ao, seed)
    G = (np.array(host.coulomb_gradient(basis, D, nuclei,
                                        one_centre=one_centre))
         - 0.5 * np.array(host.exchange_gradient(basis, D, nuclei,
                                                 one_centre=one_centre)))
    return D, G


def test_the_stream_meets_every_unique_quartet_once() -> None:
    basis, nuc = _synthetic(1)
    n_pair = len(basis.shells) * (len(basis.shells) + 1) // 2
    met = 0
    for am, diag, pb, pk in gf_launches(basis, nuc):
        assert am == (*(basis.shells[pb.sh[0]].l, basis.shells[pb.sh[1]].l),
                      *(basis.shells[pk.sh[0]].l, basis.shells[pk.sh[1]].l))
        assert am[0] <= am[1] and am[2] <= am[3]
        met += pb.n * (pb.n + 1) // 2 if diag else pb.n * pk.n
    assert met == n_pair * (n_pair + 1) // 2


@pytest.mark.parametrize("l2", [by_l(0, 1, id="s"), by_l(1, 5, id="p"),
                                by_l(2, 9, id="d")])
def test_the_whole_basis_is_the_host_j_minus_half_k(l2: int) -> None:
    basis, nuc = _synthetic(l2)
    D, want = _host(basis, nuc, 11 + l2)
    G = gf_gradient(basis, nuc, D, boys=boys)
    scale = np.abs(want).max()
    assert scale > 0
    assert np.abs(G - want).max() <= TOL * scale
    assert np.abs(G.sum(axis=0)).max() <= NET_TOL * scale


def test_an_asymmetric_density_is_refused() -> None:
    basis, nuc = _synthetic(0)
    D = np.arange(basis.n_ao ** 2, dtype=float).reshape(basis.n_ao, -1)
    with pytest.raises(ValueError, match="symmetric"):
        gf_gradient(basis, nuc, D, boys=boys)


# --- PySCF ----------------------------------------------------------------------


@pytest.mark.parametrize("name", [by_l("h2o_sto3g_grad", 5, id="h2o_sto3g"),
                                  by_l("h2o_sto3g_d_grad", 9,
                                       id="h2o_sto3g_d")])
def test_the_molecule_is_pyscfs_two_electron_gradient(name: str) -> None:
    fx = manifest.gradient_fixture(name)
    basis = ob.from_fixture(fx)
    nuclei = tuple((float(x.charge), x.coord) for x in fx.atoms)
    want = fx.arrays["grad:j"] + fx.arrays["grad:k"]
    G = gf_gradient(basis, nuclei, fx.density, boys=boys)
    err = float(np.abs(G - want).max())
    assert err <= ABS_TOL, f"{err:.2e} from PySCF, over {ABS_TOL:g}"
    assert np.abs(G.sum(axis=0)).max() <= NET_TOL * np.abs(G).max()
