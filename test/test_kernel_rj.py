"""`kernel_rj.py`: the split Coulomb kernel on Rys quadrature, interpreted.

The Rys kernel is `kernel_hj`'s split `J` with the primitive nest's engine
part replaced (the Rys rule and the 2D VRR into the HGP boundary;
`kernel_hj.Nest`), so it is held to the HGP kernel from both ends, on
test/test_kernel_hj.py's bases, records, launches and witnesses:

- THE SEAM. Outside the `K_CD` primitive loop the two kernels are the SAME
  IR, in both class modes: the same parameters, staging, screen, HRR, tap
  and deposit.
- THE INTEGRALS. The ERI tap `g` against `eri/hgp_drive.py` (HGP) and
  `eri/eri.py` (MMD) on the same built-order shells, element by element
  within `TOL (s_hgp + s_mmd)`, test/test_hgp_eri.py's per-ERI cross-engine
  bound with both engines' absolute-value twins. Not bitwise: the boundary
  is a different sum (a root sum, not a Boys ladder).
- THE DEPOSIT. `J` against `eri/j_coulomb/j.py`'s sweep restricted to the class
  (test_kernel_hj.py's `reference`, MMD), within `TOL (s_hgp + s_mmd)`,
  every element off the class's blocks exactly 0; the two class modes
  against each other; a whole molecule from `launch_hjk`'s plan, walked by
  `walk_hjk` with this kernel injected (`build=`), against the full MMD
  sweep and libcint's `vj` -- test_hjk_build.py's checks at its
  tolerances.

The rule is injected as `rys=rys.roots.roots_weights` (test/test_rys_roots.py
holds it to a 40-digit route and the device text to it). Every class through
`L = 4` (all 70 ordered ones) is fast; the classes `5 <= L <= 8` with every
shell `l <= 2` are `slow`. No compiler, no card, no emission.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
import test_hjk_build as HB
import test_kernel_hj as THJ
from tiers import split_by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.eri import eri_block_and_scale as mmd_eri
from felizjueves.eri.hjk import kernel_hj as HJ
from felizjueves.eri.hjk import kernel_rj as RJ
from felizjueves.eri.hjk import walk_hjk as W
from felizjueves.hgp import orbits
from felizjueves.ir import Boys, For, Note, Rys, Scalar, walk
from felizjueves.ir.interp import InterpError, Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.rys.roots import T_SWITCH, roots_weights
from felizjueves.spec import Engine, Operator, Spec, Target

MODES = THJ.MODES
TOL = THJ.TOL
HIGH = [c for c in orbits.ordered_classes(8, lmax_shell=2) if sum(c) > 4]
CLASSES = THJ.CLASSES
LADDER = split_by_l([*CLASSES, *HIGH], sum, ids=lambda c: "".join(map(str, c)))


def _spec(cls, mode) -> Spec:
    return RJ.rj_class_spec(*cls, ordered=mode == "ordered")


@cache
def _kernel(cls, mode, eris=False, density=HJ.COULOMB_DENSITY):
    return RJ.build_rj_kernel(_spec(cls, mode), eris=eris, density=density)


def launch(cls, mode, basis, D, **kw):
    """test_kernel_hj.py's launch on the Rys kernel, the rule injected."""
    return THJ.launch(cls, mode, basis, D, kernel=_kernel, rys=roots_weights,
                      **kw)


def kernel_j(cls, mode, **kw):
    return THJ.kernel_j(cls, mode, kernel=_kernel, rys=roots_weights, **kw)


def test_the_high_ladder_is_every_shell_through_d() -> None:
    assert len(HIGH) == 81 - 50 and max(map(sum, HIGH)) == 8


# ---------------------------------------------------------------------------
# The key, the names, the refusals
# ---------------------------------------------------------------------------


def test_the_key_is_j_on_the_rys_engine_and_the_names_are_hjs() -> None:
    s = RJ.rj_class_spec(1, 0, 1, 1)
    assert (s.target, s.engine) == (Target.J, Engine.RYS)
    assert s.key() == "coulomb/d0/(1,0|1,1)/b1/j.rys/literal"
    o = RJ.rj_class_spec(1, 0, 1, 1, ordered=True)
    assert o.key() == "coulomb/d0/(1,0|1,1)/b1/j.rys/ordered"
    for cls in CLASSES:
        for mode in MODES:
            hj = HJ.hj_name(THJ._spec(cls, mode))
            assert RJ.rj_name(_spec(cls, mode)) == "r" + hj[1:]
    assert RJ.build_rj_kernel(s).name == "rj_" + HJ.hj_name(
        HJ.hj_class_spec(1, 0, 1, 1))[3:]
    assert RJ.build_rj_kernel(o).name == "rj_o_1011"
    assert RJ.hgp_twin(o) == HJ.hj_class_spec(1, 0, 1, 1, ordered=True)


def test_the_builder_refuses_hgp_a_lever_a_batch_and_past_the_rule() -> None:
    with pytest.raises(ValueError, match="Engine.RYS"):
        RJ.build_rj_kernel(HJ.hj_class_spec(0, 0, 0, 0))
    with pytest.raises(ValueError, match="Engine.RYS"):
        RJ.build_rj_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.F,
                                engine=Engine.RYS))
    with pytest.raises(ValueError, match="N = 5"):
        RJ.build_rj_kernel((3, 3, 2, 2))  # L = 10: N = 6
    with pytest.raises(ValueError, match="no-sieve"):
        RJ.build_rj_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.J,
                                ("no-sieve",), engine=Engine.RYS))
    with pytest.raises(ValueError, match="n_batches"):
        RJ.build_rj_kernel(Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.J,
                                n_batches=3, engine=Engine.RYS))
    with pytest.raises(TypeError):
        RJ.build_rj_kernel([0, 0, 0, 0])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The seam: kernel_hj's kernel outside the primitive nest
# ---------------------------------------------------------------------------


def _outside_the_nest(body):
    """`body` with the `K_CD` loop's body and every `Note` dropped."""
    out = []
    for s in body:
        if isinstance(s, Note):
            continue
        if isinstance(s, For) and s.var == HJ.KCD:
            out.append(For(s.var, s.start, s.stop, s.step, ()))
        elif isinstance(s, For):
            out.append(replace(s, body=_outside_the_nest(s.body)))
        elif hasattr(s, "then"):
            out.append(replace(s, then=_outside_the_nest(s.then),
                               orelse=_outside_the_nest(s.orelse)))
        else:
            out.append(s)
    return tuple(out)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_outside_the_primitive_nest_it_is_kernel_hj(cls, mode) -> None:
    rj = _kernel(cls, mode, eris=True)
    hj = HJ.build_hj_kernel(THJ._spec(cls, mode), eris=True)
    assert rj.params == hj.params
    assert _outside_the_nest(rj.body) == _outside_the_nest(hj.body)
    assert not any(isinstance(s, Boys) for s in walk(rj.body))
    (rule,) = [s for s in walk(rj.body) if isinstance(s, Rys)]
    assert rule.n == sum(cls) // 2 + 1


def test_a_rys_without_its_rule_injected_raises() -> None:
    cls = (0, 0, 0, 0)
    basis = THJ.basis_for(cls)
    with pytest.raises(InterpError, match="rys="):
        THJ.launch(cls, "ordered", basis, THJ.density(basis.n_ao),
                   kernel=_kernel)


# ---------------------------------------------------------------------------
# The integrals: against HGP and MMD
# ---------------------------------------------------------------------------


@cache
def mmd(basis, xs):
    """`eri.eri`'s block and twin on the built-order shells `xs`, flat in
    `hgp.hrr.targets` order (`[a][b][c][d]`, as `eri_block`'s)."""
    sh = basis.shells
    blk, scale = mmd_eri((sh[xs[0]], sh[xs[1]]), (sh[xs[2]], sh[xs[3]]),
                         pq=pq)
    return np.array(blk).ravel(), np.array(scale).ravel()


def check_tap(cls, basis, g, bras, kets) -> float:
    """Every ERI against MMD element by element within `TOL (s_hgp +
    s_mmd)`, and against HGP within `TOL` of the block's largest `s_hgp +
    s_mmd`: HGP's drive forms `W - P` as a difference, so where the four
    centres coincide its structural zeros come out as rounding (4.6e-34 at
    `(pp|pp)`, about half its own twin), which no element-wise bound on
    either twin covers; MMD and Rys leave them exactly 0."""
    worst = 0.0
    for ib, b in enumerate(bras):
        for ik, k in enumerate(kets):
            xs = (*b, *k)
            want, s_hgp = THJ.hgp(basis, xs)
            ref, s_mmd = mmd(basis, xs)
            scale = s_hgp + s_mmd
            err = np.abs(g[ib, ik] - ref)
            assert np.all(err <= TOL * scale), (b, k)
            assert np.all(np.abs(g[ib, ik] - want) <= TOL * np.max(scale))
            worst = max(worst, float(np.max(
                err / np.where(scale > 0, scale, 1.0))))
    return worst


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_the_tap_meets_hgp_and_mmd(cls, mode) -> None:
    """Every ERI of every launched quartet against `eri.eri` and
    `hgp_drive.eri_block` on the same built-order shells (`check_tap`)."""
    basis = THJ.basis_for(cls)
    _, g, bras, kets, _, _ = launch(cls, mode, basis, THJ.density(basis.n_ao))
    worst = check_tap(cls, basis, g, bras, kets)
    print(f"{cls} {mode}: worst {worst:.3g} of s_hgp + s_mmd against MMD")


@pytest.mark.parametrize("cls", split_by_l([(0, 0, 1, 1), (1, 1, 2, 2)], sum,
                                            ids=str))
def test_far_apart_pairs_take_the_large_t_branch(cls) -> None:
    """The ket's shells (their `l` not on the bra) moved 20 bohr away:
    every primitive quartet's `T` is past `T_SWITCH` (the Laguerre branch,
    recorded), and the tap still meets HGP and MMD."""
    assert not set(cls[:2]) & set(cls[2:])
    far = Basis(tuple(
        Shell(s.l, (s.centre[0], s.centre[1], s.centre[2] + 20.0),
              s.exponents, s.coefficients) if s.l in cls[2:] else s
        for s in THJ.basis_for(cls).shells))
    seen: list[float] = []

    def rule(n, T):
        seen.append(T)
        return roots_weights(n, T)

    _, g, bras, kets, _, _ = THJ.launch(cls, "ordered", far,
                                        THJ.density(far.n_ao), kernel=_kernel,
                                        rys=rule)
    assert seen and min(seen) >= T_SWITCH
    check_tap(cls, far, g, bras, kets)


# ---------------------------------------------------------------------------
# The deposit: against eri/j_coulomb/j.py's restricted sweep, the modes, a molecule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_the_deposit_is_ao_js_sweep_restricted_to_the_class(cls, mode) -> None:
    want, s_mmd, mask = THJ.reference(cls)
    J, s_hgp, *_ = kernel_j(cls, mode)
    assert np.all(J[~mask] == 0.0)
    err = np.abs(J - want)
    bound = TOL * (s_hgp + s_mmd)
    assert np.all(err[mask] <= bound[mask]), float(np.max(err / (bound + 1e-300)))
    assert np.max(np.abs(want[mask])) > 1e-3


@pytest.mark.parametrize("cls", split_by_l(CLASSES, sum,
                                            ids=lambda c: "".join(map(str, c))))
def test_canonical_and_ordered_agree(cls) -> None:
    Jc, sc, *_ = kernel_j(cls, "canonical")
    Jo, so, *_ = kernel_j(cls, "ordered")
    assert np.all(np.abs(Jc - Jo) <= TOL * (sc + so))


def test_the_screen_skips_one_ket_and_never_breaks() -> None:
    """test_kernel_hj.py's screen case on the Rys kernel: the middle ket's
    `G_cd = 0` skips it (its tap row unwritten), the rest computed."""
    cls = (0, 1, 1, 1)
    basis = THJ.basis_for(cls)
    D = THJ.density(basis.n_ao)
    _, g, bras, kets, _, it = launch(cls, "canonical", basis, D, tau=1e-14,
                                     G={1: 0.0})
    _, gf, *_ = launch(cls, "canonical", basis, D)
    for ib in range(len(bras)):
        for ik in range(len(kets)):
            if ik == 1:
                assert np.all(g[ib, ik] == -7.0)
            else:
                assert np.array_equal(g[ib, ik], gf[ib, ik])
    assert it.iterations[HJ.KET] == len(bras) * len(kets)
    assert it.iterations[HJ.KAB] == len(bras) * (len(kets) - 1) * 4


def run(kernel, grid, block, args):
    """test_hjk_build.py's executor with the Rys rule injected too."""
    a = {}
    for p in kernel.params:
        v = args[p.name]
        if p.array:
            v = np.array(v, dtype=np.int32 if p.type is Scalar.I32 else float)
        a[p.name] = v
    return Interpreter(kernel, launch=Launch(grid, block, 32), boys=boys,
                       rys=roots_weights).run(**a)


BUILT: list[str] = []


def rys_of(spec: Spec):
    """`walk_hjk`'s `build=`: the Rys drop-in of the launch's HGP key."""
    k = RJ.build_rj_kernel(replace(spec, engine=Engine.RYS))
    BUILT.append(k.name)
    return k


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", [
    "h2_sto3g", pytest.param("h2o_sto3g", marks=pytest.mark.slow)])
def test_a_whole_molecule_against_the_sweep_and_libcint(name, mode) -> None:
    """`launch_hjk`'s plan at `tau = 0`, every launch on the Rys kernel
    (`walk_hjk(build=)`): the full ordered MMD sweep to `TOL S` and libcint's
    `vj` to `LIBCINT_TOL S` (test_hjk_build.py's `S`). One plan serves both
    engines."""
    basis, D, vj, _ = HB.molecule(name)
    BUILT.clear()
    w = W.walk_hjk(HB.plan(name, 0.0, mode, "j"), basis, D, run=run,
                   build=rys_of)
    assert len(BUILT) == len(w.kernels) > 0
    assert all(k.startswith("rj_") for k in BUILT)
    assert len(w.visited) == len(basis.shells) ** 4
    X = np.array(w.X)
    S = HB.sweep(name, "j", block="scale", mode=mode)
    want = HB.reference(name, "j")
    assert np.max(np.abs(want)) > 0.1
    host, lib = HB._worst(X, want, S), HB._worst(X, vj, S)
    print(f"{name} rys j {mode}: host {host:.3g} S, libcint {lib:.3g} S")
    assert host <= HB.TOL
    assert lib <= HB.LIBCINT_TOL
