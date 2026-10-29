"""`kernel_frys.py`: the fused Fock kernel on Rys quadrature, interpreted.

The Rys kernel is `kernel_f`'s skeleton with the primitive nest's body
replaced (the Rys rule and the 2D VRR into the HGP boundary), so it is held
to the HGP kernel from both ends:

- THE INTEGRALS. Its ERI tap `g` against the HGP host path
  (`test_kernel_f.host`, #279's pinned VRR and HRR kernels interpreted) and
  against `eri/eri.py`'s MMD blocks, both to `TOL` of the MMD absolute-value
  scale: three engines, two operation orders, one rounding bound. Not
  bitwise: the boundary is a different sum.
- THE DIGESTION. One unique quartet's closed deposit against
  `eri.fock.fock`'s every-ordered-quartet sweep (`test_kernel_f.sweep`), and
  whole molecules from `launch_f`'s plan against the full sweep and
  libcint's `vj - vk/2` -- the HGP kernel's own checks, at its tolerances.
- THE SEAM. Outside the `K_CD` primitive loop the two kernels are the SAME
  IR: the same parameters, staging, screen, HRR, tap and digestion.

The rule is injected as `rys=rys.roots.roots_weights` (the device routine's
operation order; test/test_rys_roots.py holds it to a 40-digit route and the
device text to it). Fast tier for `L <= FAST_L_MAX`, `slow` above.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
import test_kernel_f as TKF
from tiers import split_by_l

from felizjueves.ao.shell import Basis
from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock import kernel_frys as KR
from felizjueves.eri.fock.launch_f import f_launches
from felizjueves.hgp import hrr, vrr
from felizjueves.ir import Boys, For, Kernel, Note, Rys, walk
from felizjueves.ir.interp import InterpError, Interpreter, Launch
from felizjueves.rys.roots import T_SWITCH, roots_weights
from felizjueves.spec import Engine, Operator, Spec, Target

LADDER = split_by_l(vrr.CLASSES, sum, ids=str)
TOL, LIBCINT_TOL = TKF.TOL, TKF.LIBCINT_TOL


@cache
def _kernel(cls, weights=KF.WEIGHTS, eris=False) -> Kernel:
    return KR.build_frys_kernel(cls, weights=weights, eris=eris)


def fused(cls, basis, D, bras, kets, *, diag=0, tau=0.0, threads=2,
          weights=KF.WEIGHTS, eris=False, F0=None, rule=roots_weights):
    """`test_kernel_f.fused` on the Rys kernel: `(F unclosed, g, it)`."""
    k = _kernel(tuple(cls), weights, eris)
    pb, pk = KF.pack(bras), KF.pack(kets)
    n_eri = len(hrr.targets(cls))
    n_sh, n_ao = len(basis.shells), basis.n_ao
    D = np.asarray(D, dtype=float)
    Dsh = KF.density_blocks(D, basis.offsets, [s.n_cart for s in basis.shells])
    every = {
        "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree, "k_cd": pk.degree,
        "n_sh": n_sh, "n_ao": n_ao, "diag": int(diag), "tau": float(tau),
        "bra_prim": np.array(pb.prim), "AB": np.array(pb.sep),
        "G_ab": np.array(pb.G),
        "bra_sh": np.array(pb.sh, dtype=np.int32),
        "bra_ao": np.array(pb.ao, dtype=np.int32),
        "ket_prim": np.array(pk.prim), "CD": np.array(pk.sep),
        "G_cd": np.array(pk.G),
        "ket_sh": np.array(pk.sh, dtype=np.int32),
        "ket_ao": np.array(pk.ao, dtype=np.int32),
        "Dsh": np.array(Dsh, dtype=float).ravel(), "D": D.ravel(),
        KF.FOCK: (np.zeros(n_ao * n_ao) if F0 is None
                  else np.asarray(F0, dtype=float).ravel().copy()),
        KF.OUT: np.zeros(pb.n * pk.n * n_eri),
    }
    grid, block = KF.launch_shape(pb.n, threads)
    it = Interpreter(k, launch=Launch(grid, block, 32), rys=rule)
    out = it.run(**{n: every[n] for n in KF.params_of(cls, eris=eris)})
    F = out[KF.FOCK].reshape(n_ao, n_ao)
    g = out[KF.OUT].reshape(pb.n, pk.n, n_eri) if eris else None
    return F, g, it


# ---------------------------------------------------------------------------
# The key, the name, the refusals
# ---------------------------------------------------------------------------


def test_the_key_is_f_on_the_rys_engine() -> None:
    spec = KR.frys_class_spec(1, 0, 1, 1)
    assert spec.engine is Engine.RYS and spec.target is Target.F
    assert spec.key() == "coulomb/d0/(1,0|1,1)/b1/f.rys/literal"
    assert KR.build_frys_kernel(spec).name == "fr_1011"


@pytest.mark.parametrize("target", [
    t for t in Target
    # F here; S, T and V in one_electron/kernel_rys.py (test_kernel_rys_one_e)
    if t not in (Target.F, Target.S, Target.T, Target.V)])
def test_rys_is_refused_on_every_other_target(target) -> None:
    from felizjueves.spec import AM_LENGTH, OPERATOR_OF
    with pytest.raises(ValueError, match=f"engine rys on target {target}"):
        Spec(OPERATOR_OF[target], 0, (0,) * AM_LENGTH[target], target,
             engine=Engine.RYS)


def test_the_builder_refuses_hgp_a_lever_and_past_the_rule() -> None:
    with pytest.raises(ValueError, match="Engine.RYS|rys"):
        KR.build_frys_kernel(KF.f_class_spec(0, 0, 0, 0))
    with pytest.raises(ValueError, match="N = 5"):
        KR.build_frys_kernel((3, 3, 3, 3))  # L = 12: N = 7
    outer = Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.F, ("tuned",),
                 engine=Engine.RYS)  # a plan-layer lever: Spec admits it
    with pytest.raises(ValueError, match="no lever"):
        KR.build_frys_kernel(outer)
    with pytest.raises(TypeError):
        KR.build_frys_kernel((0, 0))


# ---------------------------------------------------------------------------
# The seam: the same kernel as kernel_f outside the primitive nest
# ---------------------------------------------------------------------------


def _outside_the_nest(body):
    """`body` with the `K_CD` loop's body and every `Note` dropped."""
    out = []
    for s in body:
        if isinstance(s, Note):
            continue
        if isinstance(s, For) and s.var == KF.KCD:
            out.append(For(s.var, s.start, s.stop, s.step, ()))
        elif isinstance(s, For):
            out.append(For(s.var, s.start, s.stop, s.step,
                           _outside_the_nest(s.body)))
        elif hasattr(s, "then"):
            out.append(type(s)(s.cond, _outside_the_nest(s.then),
                               _outside_the_nest(s.orelse)))
        else:
            out.append(s)
    return tuple(out)


@pytest.mark.parametrize("cls", LADDER)
def test_outside_the_primitive_nest_it_is_kernel_f(cls) -> None:
    fr, f = _kernel(cls, eris=True), KF.build_f_kernel(cls, eris=True)
    assert fr.params == f.params
    assert _outside_the_nest(fr.body) == _outside_the_nest(f.body)
    assert not any(isinstance(s, Boys) for s in walk(fr.body))
    (rule,) = [s for s in walk(fr.body) if isinstance(s, Rys)]
    assert rule.n == sum(cls) // 2 + 1


def test_a_rys_without_its_rule_injected_raises() -> None:
    cls = (0, 0, 0, 0)
    basis = Basis(tuple(TKF._shells(cls)))
    k = _kernel(cls)
    pb = KF.pack([TKF.records(basis, 0, 1)])
    pk = KF.pack([TKF.records(basis, 2, 3)])
    it = Interpreter(k, launch=Launch(1, (1, 1), 32))
    with pytest.raises(InterpError, match="rys="):
        it.run(n_bra=1, n_ket=1, k_ab=pb.degree, k_cd=pk.degree, n_sh=4,
               n_ao=4, diag=0, tau=0.0, bra_prim=np.array(pb.prim),
               G_ab=np.array(pb.G), bra_sh=np.array(pb.sh, dtype=np.int32),
               bra_ao=np.array(pb.ao, dtype=np.int32),
               ket_prim=np.array(pk.prim), G_cd=np.array(pk.G),
               ket_sh=np.array(pk.sh, dtype=np.int32),
               ket_ao=np.array(pk.ao, dtype=np.int32),
               Dsh=np.ones(16), D=np.ones(16), F=np.zeros(16))


# ---------------------------------------------------------------------------
# The integrals: against the HGP host path and the MMD blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", LADDER)
def test_one_contracted_quartet_against_hgp_and_mmd(cls) -> None:
    """All 21 classes on one contracted quartet of 2-primitive shells on
    four centres (16 primitive quartets): the tap against the HGP host path
    and `eri/eri.py`, each to `TOL` of the MMD absolute-value scale."""
    sh = TKF._shells(cls)
    basis = Basis(tuple(sh))
    bra, ket = TKF.records(basis, 0, 1), TKF.records(basis, 2, 3)
    _, g, _ = fused(cls, basis, TKF._density(basis.n_ao), [bra], [ket],
                    threads=1, eris=True)
    ref, scale = TKF.mmd(cls, sh)
    hgp = TKF.host(cls, bra, ket)
    assert np.all(np.abs(g[0, 0] - hgp) <= TOL * scale)
    assert np.all(np.abs(g[0, 0] - ref) <= TOL * scale), float(
        np.max(np.abs(g[0, 0] - ref) / scale))
    assert np.max(np.abs(ref)) > 0


@pytest.mark.parametrize("cls", split_by_l([(0, 1, 1, 1), (1, 1, 2, 2)], sum,
                                            ids=str))
def test_far_apart_pairs_take_the_large_t_branch(cls) -> None:
    """The ket moved 20 bohr away: every primitive quartet's `T` is past
    `T_SWITCH` (the Laguerre branch, recorded), and the tap still meets
    MMD."""
    far = tuple(tuple(x + (20.0 if i >= 2 and d == 2 else 0.0)
                      for d, x in enumerate(c))
                for i, c in enumerate(TKF.CENTRES))
    sh = TKF._shells(cls, far)
    basis = Basis(tuple(sh))
    bra, ket = TKF.records(basis, 0, 1), TKF.records(basis, 2, 3)
    seen: list[float] = []

    def rule(n, T):
        seen.append(T)
        return roots_weights(n, T)

    _, g, _ = fused(cls, basis, TKF._density(basis.n_ao), [bra], [ket],
                    threads=1, eris=True, rule=rule)
    assert len(seen) == 16 and min(seen) >= T_SWITCH
    ref, scale = TKF.mmd(cls, sh)
    assert np.all(np.abs(g[0, 0] - ref) <= TOL * scale)


# ---------------------------------------------------------------------------
# The digestion: against eri.fock.fock's sweep and libcint
# ---------------------------------------------------------------------------


def _check_launch(cls, basis, D, bra_ix, ket_ix, *, diag, threads=2):
    bras = [TKF.records(basis, *p) for p in bra_ix]
    kets = [TKF.records(basis, *p) for p in ket_ix]
    F, _, _ = fused(cls, basis, D, bras, kets, diag=diag, threads=threads)
    G = np.array(KF.close(F))
    want, scale = TKF.sweep(basis, D, TKF._members(bra_ix, ket_ix))
    assert np.max(np.abs(want)) > 0
    return TKF._worst(G, want, scale)


@pytest.mark.parametrize("cls", LADDER)
def test_one_orbit_equals_its_ordered_members_in_the_sweep(cls) -> None:
    basis = Basis(tuple(TKF._shells(cls)))
    D = TKF._density(basis.n_ao)
    assert _check_launch(cls, basis, D, [(0, 1)], [(2, 3)], diag=0) <= TOL


def test_a_diagonal_launch_with_every_degeneracy() -> None:
    """(pp|pp) on two p shells in one diagonal launch of its three canonical
    pairs: `a = b`, `c = d`, `(ab) = (cd)` and all three."""
    basis = Basis(tuple(TKF._same_l_shells(1, 2)))
    D = TKF._density(basis.n_ao)
    pairs = [(0, 0), (0, 1), (1, 1)]
    assert _check_launch((1, 1, 1, 1), basis, D, pairs, pairs,
                         diag=1) <= TOL


def fock_from_plan(basis, D, plan, *, threads=4):
    F = np.zeros((basis.n_ao, basis.n_ao))
    for L in plan:
        F, _, _ = fused(L.spec.am, basis, D,
                        TKF._side(basis, plan.batches[L.bra]),
                        TKF._side(basis, plan.batches[L.ket]),
                        diag=int(L.bra == L.ket), tau=plan.tau,
                        threads=threads, F0=F)
    return np.array(KF.close(F))


@pytest.mark.parametrize("name", [
    "h2_sto3g", pytest.param("h2o_sto3g", marks=pytest.mark.slow)])
def test_a_whole_molecule_against_the_sweep_and_libcint(name) -> None:
    """`launch_f`'s plan at `tau = 0`, every launch on the Rys kernel,
    closed: the full ordered sweep to `TOL` of scale and libcint's `vj -
    vk/2` to `LIBCINT_TOL`. The plan is the HGP kernel's: one plan serves
    both engines."""
    basis, D, libcint = TKF._molecule(name)
    plan = f_launches(basis, D, bound=TKF.BOUND, tau=0.0, n=1)
    G = fock_from_plan(basis, D, plan)
    want, scale = TKF.sweep(basis, D)
    assert TKF._worst(G, want, scale) <= TOL
    assert TKF._worst(G, libcint, scale) <= LIBCINT_TOL
