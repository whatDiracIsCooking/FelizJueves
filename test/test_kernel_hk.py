"""`kernel_hk.py`: the HGP exchange kernel, split (#405), interpreted.

Every K class `(la, lc | lb, ld)` with `L <= 4` -- all 70 ordered tuples,
every shell `l` up to 4 (`hgp.orbits.ordered_classes(4, lmax_shell=None)`,
read as K's Hermite `am`) -- in BOTH class modes: canonical (the orbit
representative's kernel, launched with the class's permutation) and
`ordered` (the class's own kernel). Nothing above L = 4 is built or run.

THE ERIs, BITWISE. The tap `g` (`eris=True`) is `==` `ao/hgp_drive.py`'s
`eri_block` on the same built-order shells, bit for bit: the nest and the
quartet scalars are `kernel_hj`'s (bra outer, the drive's operation order),
and the records are formed by the drive's own expressions
(`test_kernel_hj.records`).

THE DEPOSIT, AGAINST `ao/k.py` FOR THAT CLASS. The witness is
`ao.k.k_quartet` -- the MMD staged contraction `X[a][b] = sum_cd D_cd
(ac|bd)`, a different recurrence and a different contraction order -- summed
over exactly the class's ordered quadruples (`a` of `l = la`, `c` of `lc`,
`b` of `lb`, `d` of `ld`) into `K[a][b]`, which is the class's share of
`ao.k.exchange_matrix_reference` (every ordered quadruple at weight 1, no
symmetry). Every `K` element outside the class's (row `l = la`, column
`l = lb`) blocks must be exactly 0. A whole s/p build -- every class
together -- is held to `exchange_matrix_reference` itself.

NOT BITWISE, AND THE TOLERANCE, DERIVED (test_kernel_hj's, for K). The
comparison is

    |K_kernel - ref| <= 1e-12 (s_hgp + s_mmd),   element by element,

by the triangle inequality through the exact `K` of the class:
`|K_kernel - K| <= 1e-12 s_hgp` and `|ref - K| <= 1e-12 s_mmd`. Each `s` is
its engine's absolute-value ERI twin carried through `K`'s linear sum on
`|D|`: `s_hgp = SUM_q |D_cd| scale_hgp(ac|bd)` with `ao.hgp_drive`'s twin in
the orientation the kernel ran, `s_mmd` the same with `ao.eri`'s twin --
which `ao/k.py`'s docstring ("THE SCALE") and test_ao_k.py (to 1e-14
relative) state is exactly `k_quartet`'s own `K` scale, and test_ao_k.py
holds `k_quartet` to the `ao.eri` contraction within `1e-14` of it. `1e-12`
is test/test_hgp_eri.py's `TOL`: `gamma_n` for `n` about 9000 roundings, the
per-ERI cross-engine bound it pins (HGP against MMD within `1e-12
(scale_mmd + scale_hgp)`); the deposit adds at most `n_cd x n_quartets`
(<= 81 x 16 here) roundings per element, inside that budget. No bound is
widened: the same constant, carried through a linear map.

THE MODES AGREE: `K` and every ERI (the canonical block permuted through
`hgp.orbits`' `gather`) within `1e-12 (scale_canonical + scale_ordered)`.

THE AO-ORDER TRAP: `(la, lb, lc, ld)` spelled as K's `am` is a different
class whenever `lb != lc` -- another recurrence, other `K` blocks -- and its
`K` misses the class's reference by orders of magnitude beyond the bound.

CAN FAIL: a deposit reading the wrong density block (`density=` mutation:
J's `(2, 3)`, or `(0, 1)`) misses `ao/k.py` far beyond the tolerance.

Fast tier: every case here is `L <= FAST_L_MAX` (tiers.py). No compiler,
no card, no emission (#407 emits).
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from test_kernel_hj import (
    _sp_basis,
    basis_for,
    density,
    dsh,
    hgp,
    mmd_scale,
    ordered_quartets,
    pairs_of,
    records,
)
from tiers import split_by_l

from felizjueves.ao.density import epair
from felizjueves.eri.fock.kernel_f import pack
from felizjueves.eri.hjk import kernel_hk as HK
from felizjueves.eri.k_exchange.k import exchange_matrix_reference, k_quartet
from felizjueves.hgp import hrr, orbits, vrr
from felizjueves.hgp import recurrence as R
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Break,
    Call,
    Decl,
    Fn,
    For,
    If,
    Index,
    Note,
    Space,
    Var,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target, lever_provenance

CLASSES = orbits.ordered_classes(4, lmax_shell=None)  # K's (la, lc, lb, ld)
LADDER = split_by_l(CLASSES, sum, ids=lambda c: "".join(map(str, c)))
MODES = ("canonical", "ordered")

# test/test_hgp_eri.py's TOL: the per-ERI cross-engine bound (module
# docstring, "THE TOLERANCE"), carried through K's linear sum.
TOL = 1e-12


def test_the_ladder_is_every_class_through_l4() -> None:
    assert len(CLASSES) == 70 and max(map(sum, CLASSES)) == 4
    assert len({orbits.representative(c) for c in CLASSES}) == 17


def _spec(cls, mode) -> Spec:
    return HK.hk_class_spec(*cls, ordered=mode == "ordered")


# ---------------------------------------------------------------------------
# The key, the names, the refusals
# ---------------------------------------------------------------------------


def test_the_spec_is_hgp_k_in_hermite_order_and_names_cannot_collide() -> None:
    s = HK.hk_class_spec(1, 0, 0, 0)  # (la, lc | lb, ld)
    assert (s.target, s.engine, s.am, s.levers) == (
        Target.K, Engine.HGP, (1, 0, 0, 0), ())
    assert s.key() == "coulomb/d0/(1,0|0,0)/b1/k.hgp/literal"
    # the HGP K key's am is the MMD K key's: one name under both engines
    mmd = Spec(Operator.COULOMB, 0, (2, 1, 0, 1), Target.K)
    assert HK.hk_class_spec(2, 1, 0, 1).am == mmd.am
    o = HK.hk_class_spec(1, 0, 0, 0, ordered=True)
    assert o.key() == "coulomb/d0/(1,0|0,0)/b1/k.hgp/ordered"
    assert HK.hk_name(s) == "hk_0010" == HK.hk_name(HK.hk_class_spec(0, 0, 0, 1))
    assert HK.hk_name(o) == "hk_o_1000"
    canon = {HK.hk_name(_spec(c, "canonical")) for c in CLASSES}
    ordered = {HK.hk_name(_spec(c, "ordered")) for c in CLASSES}
    assert len(canon) == 17 and len(ordered) == 70 and not canon & ordered
    assert HK.hk_name(HK.hk_class_spec(0, 11, 0, 0, ordered=True)) == \
        "hk_o_0_11_0_0"
    names = {HK.hk_name(HK.hk_class_spec(*c, ordered=True))
             for c in ((0, 11, 0, 0), (0, 1, 10, 0), (0, 1, 1, 0))}
    assert len(names) == 3


def test_refusals() -> None:
    with pytest.raises(ValueError, match="Engine.HGP"):
        HK.build_hk_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.K))
    with pytest.raises(ValueError, match="Engine.HGP"):
        HK.build_hk_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.J,
                                engine=Engine.HGP))
    with pytest.raises(ValueError, match="n_batches"):
        HK.build_hk_kernel(Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.K,
                                n_batches=3, engine=Engine.HGP))
    with pytest.raises(ValueError, match="no-sieve"):
        HK.build_hk_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.K,
                                ("no-sieve",), engine=Engine.HGP))
    with pytest.raises(TypeError):
        HK.build_hk_kernel([0, 0, 0, 0])  # type: ignore[arg-type]
    for bad in ((2, 2), (0, 4), (1,)):
        with pytest.raises(ValueError, match="density"):
            HK.build_hk_kernel((0, 0, 0, 0), density=bad)


def test_params_follow_the_mode_the_hrr_and_the_tap() -> None:
    assert HK.PERM in HK.params_of((0, 0, 0, 0))
    assert HK.PERM not in HK.params_of(HK.hk_class_spec(0, 0, 0, 0,
                                                        ordered=True))
    assert "AB" not in HK.params_of((1, 0, 0, 0))  # builds (0,0,1,0)
    assert "AB" in HK.params_of(HK.hk_class_spec(0, 1, 0, 0, ordered=True))
    assert "CD" in HK.params_of(HK.hk_class_spec(0, 0, 0, 1, ordered=True))
    for mode in MODES:
        for eris in (False, True):
            s = _spec((0, 1, 1, 1), mode)
            k = HK.build_hk_kernel(s, eris=eris)
            assert tuple(p.name for p in k.params) == HK.params_of(s, eris=eris)
    assert HK.params_of((0, 0, 0, 0), eris=True)[-1] == HK.OUT
    assert HK.K_OUT in HK.params_of((0, 0, 0, 0))


def test_orientations_are_the_orbits_permutations() -> None:
    """Canonical: one `GROUP` index per distinct member's
    `to_representative`; ordered: the identity. Every canonical deposit
    lands K on one shell of each built pair (K crosses the pairs)."""
    for cls in CLASSES:
        got = HK.orientations(cls)
        want = sorted({orbits.GROUP.index(orbits.to_representative(m).shells)
                       for m in orbits.orbit(cls)})
        assert list(got) == want, cls
        assert HK.orientations(_spec(cls, "ordered")) == (0,)
        for g in got:
            inv = HK.inverse(orbits.GROUP[g])
            assert {inv[0] // 2, inv[2] // 2} == {0, 1}
            assert {inv[1] // 2, inv[3] // 2} == {0, 1}


# ---------------------------------------------------------------------------
# The counts and the shape
# ---------------------------------------------------------------------------


@cache
def _kernel(cls, mode, eris=False, density=HK.K_DENSITY):
    return HK.build_hk_kernel(_spec(cls, mode), eris=eris, density=density)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_statement_counts_are_the_built_classs_dags(cls, mode) -> None:
    spec = _spec(cls, mode)
    got = HK.counted(_kernel(cls, mode))
    assert got == HK.kernel_counts(spec)
    am = HK.built_class(spec)
    assert am == (cls if mode == "ordered" else orbits.representative(cls))
    v, h = vrr.build(am), hrr.build(am)
    assert got.vrr == v.n_nodes
    assert got.boundary == h.n_boundary == len(vrr.target_set(am))
    assert got.hrr == h.n_statements == h.n_boundary + h.n_interior
    assert got.hrr_interior == h.n_interior
    assert got.eri == h.n_eri == math.prod(len(vrr.cartesians(x)) for x in am)
    n = [len(vrr.cartesians(x)) for x in am]
    want_atomic = 0
    for g in HK.orientations(spec):
        inv = HK.inverse(orbits.GROUP[g])
        want_atomic += n[inv[0]] * n[inv[2]]
    assert got.atomic == want_atomic


def _find_for(body, var):
    return next(s for s in walk(body) if isinstance(s, For) and s.var == var)


@pytest.mark.parametrize("mode", MODES)
def test_the_nest_the_staging_the_screen_and_no_break(mode) -> None:
    cls = (1, 1, 1, 1)
    k = _kernel(cls, mode)
    sh = [s for s in walk(k.body) if isinstance(s, Decl)
          and s.space is Space.SHARED]
    assert [s.name for s in sh] == [HK.AB_SH]
    assert [d.value for d in sh[0].shape] == [HK.MAX_PRIM_PAIRS, len(HK.REC)]
    ket = _find_for(k.body, HK.KET)
    assert ket.start == Call(Fn.THREAD_X) and ket.step == Call(Fn.BLOCK_DIM_X)
    order = list(walk(k.body))
    assert order.index(_find_for(k.body, HK.STAGE)) < order.index(ket)
    kab = _find_for(ket.body, HK.KAB)
    kcd = _find_for(kab.body, HK.KCD)
    inner = {s.name for s in walk(kcd.body) if isinstance(s, Decl)}
    assert sum(n.startswith("V") for n in inner) == vrr.build(cls).n_nodes
    hrr_names = {R.hrr_name(t) for t in hrr.build(cls).terms}
    assert not hrr_names & {s.name for s in walk(kab.body)
                            if isinstance(s, Decl)}
    screens = [s for s in walk(k.body) if isinstance(s, If)
               and s.cond == Call(Fn.LT, (Var("bound"), Var("tau")))]
    assert len(screens) == 1
    assert all(isinstance(x, Note) for x in screens[0].then)
    assert _find_for(screens[0].orelse, HK.KAB)
    assert not any(isinstance(s, Break) for s in walk(k.body))
    assert all(s.dst.array == HK.K_OUT for s in walk(k.body)
               if isinstance(s, AtomicAdd))
    assert not any(isinstance(s, Assign) and isinstance(s.dst, Index)
                   and s.dst.array != HK.AB_SH for s in walk(k.body))


def _strip_tap(body):
    out = []
    for s in body:
        if isinstance(s, Assign) and isinstance(s.dst, Index) and \
                s.dst.array == HK.OUT:
            continue
        if isinstance(s, Note) and s.prov.startswith("the ERI tap"):
            continue
        if isinstance(s, For):
            s = replace(s, body=tuple(_strip_tap(s.body)))
        elif isinstance(s, If):
            s = replace(s, then=tuple(_strip_tap(s.then)),
                        orelse=tuple(_strip_tap(s.orelse)))
        out.append(s)
    return out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 0, 2, 0), (1, 1, 1, 1)],
                         ids=str)
def test_the_tap_is_the_shipped_kernel_plus_its_writes(cls, mode) -> None:
    shipped, tap = _kernel(cls, mode), _kernel(cls, mode, True)
    assert tuple(_strip_tap(tap.body)) == shipped.body
    assert tap.params[:-1] == shipped.params


# ---------------------------------------------------------------------------
# The launches
# ---------------------------------------------------------------------------


def launch(cls, mode, basis, D, *, tau=0.0, G=None, threads=2, eris=True,
           density_=HK.K_DENSITY, perm=None):
    """One launch of K class `cls` in `mode` over every ordered quartet of
    the built class in `basis`. Returns `(K, g, bras, kets, gidx, it)`:
    bras / kets the BUILT-order shell pairs, `gidx` the permutation."""
    spec = _spec(cls, mode)
    am = HK.built_class(spec)
    if mode == "ordered":
        gidx = 0
    else:
        gidx = orbits.GROUP.index(orbits.to_representative(cls).shells)
    bras = pairs_of(basis, am[0], am[1])
    kets = pairs_of(basis, am[2], am[3])
    k = _kernel(cls, mode, eris, tuple(density_))
    G = G or {}
    pb = pack([records(basis, *p) for p in bras])
    pk = pack([records(basis, *p, G=G.get(i, 1.0)) for i, p in enumerate(kets)])
    n_eri = len(hrr.targets(am))
    n_ao = basis.n_ao
    every = {
        "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree, "k_cd": pk.degree,
        "n_sh": len(basis.shells), "n_ao": n_ao,
        HK.PERM: gidx if perm is None else perm, "tau": float(tau),
        "bra_prim": np.array(pb.prim), "AB": np.array(pb.sep),
        "G_ab": np.array(pb.G), "bra_sh": np.array(pb.sh, dtype=np.int32),
        "bra_ao": np.array(pb.ao, dtype=np.int32),
        "ket_prim": np.array(pk.prim), "CD": np.array(pk.sep),
        "G_cd": np.array(pk.G), "ket_sh": np.array(pk.sh, dtype=np.int32),
        "ket_ao": np.array(pk.ao, dtype=np.int32),
        "Dsh": dsh(basis, D).ravel(), "D": np.asarray(D, float).ravel(),
        HK.K_OUT: np.zeros(n_ao * n_ao),
        HK.OUT: np.full(pb.n * pk.n * n_eri, -7.0),
    }
    grid, block = HK.launch_shape(pb.n, threads)
    it = Interpreter(k, launch=Launch(grid, block, 32), boys=boys)
    out = it.run(**{n: every[n] for n in HK.params_of(spec, eris=eris)})
    K = out[HK.K_OUT].reshape(n_ao, n_ao)
    g = out[HK.OUT].reshape(pb.n, pk.n, n_eri) if eris else None
    return K, g, bras, kets, gidx, it


def _block(flat, shells_l) -> np.ndarray:
    return flat.reshape([len(vrr.cartesians(x)) for x in shells_l])


def deposit_scale(basis, D, xs, gidx, scale_flat, out) -> None:
    """`out[a][b] += sum_cd |D_cd| scale(ac|bd)` for the ordered K quartet
    `q = (a, c, b, d)` the built-order `xs` stands for under `gidx`."""
    inv = HK.inverse(orbits.GROUP[gidx])
    q = tuple(xs[inv[i]] for i in range(4))
    ls = [basis.shells[x].l for x in xs]
    blk = _block(scale_flat, ls).transpose(inv)  # ordered [a][c][b][d]
    o, n = basis.offsets, [basis.shells[x].n_cart for x in q]
    Dcd = np.abs(D[o[q[1]]:o[q[1]] + n[1], o[q[3]]:o[q[3]] + n[3]])
    out[o[q[0]]:o[q[0]] + n[0], o[q[2]]:o[q[2]] + n[2]] += np.einsum(
        "ikjl,kl->ij", blk, Dcd)


# ---------------------------------------------------------------------------
# The ERIs, bitwise against ao/hgp_drive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_the_tap_is_hgp_drive_bitwise(cls, mode) -> None:
    basis = basis_for(cls)
    _, g, bras, kets, _, _ = launch(cls, mode, basis, density(basis.n_ao))
    for ib, b in enumerate(bras):
        for ik, k in enumerate(kets):
            want, _ = hgp(basis, (*b, *k))
            assert np.array_equal(g[ib, ik], want), (b, k)


# ---------------------------------------------------------------------------
# The deposit against ao/k.py, the class's quadruples
# ---------------------------------------------------------------------------


def _mask(basis, la, lb) -> np.ndarray:
    o = basis.offsets
    mask = np.zeros((basis.n_ao, basis.n_ao), dtype=bool)
    for a, b in pairs_of(basis, la, lb):
        sa, sb = basis.shells[a], basis.shells[b]
        mask[o[a]:o[a] + sa.n_cart, o[b]:o[b] + sb.n_cart] = True
    return mask


@cache
def reference(cls):
    """`(ref, s_mmd, mask)`: `ao.k.k_quartet` over the K class's ordered
    quadruples, its `ao.eri` twin, and the (row l = la, column l = lb)
    blocks (module docstring)."""
    basis = basis_for(cls)
    D = density(basis.n_ao)
    sh, o = basis.shells, basis.offsets
    ref = np.zeros((basis.n_ao, basis.n_ao))
    s = np.zeros_like(ref)
    for q in ordered_quartets(basis, cls):  # q = (a, c, b, d)
        a, c, b, d = q
        dcd = D[o[c]:o[c] + sh[c].n_cart, o[d]:o[d] + sh[d].n_cart].ravel()
        X = np.array(k_quartet(epair(sh[a], sh[c]), epair(sh[b], sh[d]), dcd,
                               pq=pq))
        ref[o[a]:o[a] + sh[a].n_cart, o[b]:o[b] + sh[b].n_cart] += X
        deposit_scale(basis, D, q, 0, mmd_scale(basis, q), s)
    return ref, s, _mask(basis, cls[0], cls[2])


def kernel_k(cls, mode, **kw):
    """The kernel's `K` and its HGP twin `s_hgp` (the orientation it ran)."""
    basis = basis_for(cls)
    D = density(basis.n_ao)
    K, g, bras, kets, gidx, it = launch(cls, mode, basis, D, **kw)
    s = np.zeros_like(K)
    for b in bras:
        for k in kets:
            xs = (*b, *k)
            deposit_scale(basis, D, xs, gidx, hgp(basis, xs)[1], s)
    return K, s, g, bras, kets, gidx


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_the_deposit_is_ao_ks_quartets_for_the_class(cls, mode) -> None:
    want, s_mmd, mask = reference(cls)
    K, s_hgp, *_ = kernel_k(cls, mode)
    assert np.all(K[~mask] == 0.0)  # K[a][b] only, from its own block
    assert np.all(want[~mask] == 0.0)
    err = np.abs(K - want)
    bound = TOL * (s_hgp + s_mmd)
    assert np.all(err[mask] <= bound[mask]), float(np.max(err / (bound + 1e-300)))
    assert np.max(np.abs(want[mask])) > 1e-3  # a non-trivial deposit


@pytest.mark.parametrize("cls", LADDER)
def test_canonical_and_ordered_agree(cls) -> None:
    Kc, sc, gc, bc, kc, gidx = kernel_k(cls, "canonical")
    Ko, so, go, bo, ko, _ = kernel_k(cls, "ordered")
    assert np.all(np.abs(Kc - Ko) <= TOL * (sc + so))
    basis = basis_for(cls)
    inv = HK.inverse(orbits.GROUP[gidx])
    gather = np.array(orbits.to_representative(cls).gather)
    pos_o = {(*b, *k): (i, j) for i, b in enumerate(bo)
             for j, k in enumerate(ko)}
    for ib, b in enumerate(bc):
        for ik, k in enumerate(kc):
            xs = (*b, *k)
            q = tuple(xs[inv[i]] for i in range(4))
            i, j = pos_o[q]
            mine = gc[ib, ik][gather]
            scale = hgp(basis, xs)[1][gather] + hgp(basis, q)[1]
            assert np.all(np.abs(mine - go[i, j]) <= TOL * scale), (xs, q)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", [(0, 1, 0, 1), (1, 0, 0, 1), (1, 1, 2, 0)],
                         ids=str)
def test_a_wrong_density_block_fails(cls, mode) -> None:
    """The mutation: K_ab reading J's `D` block (ERI positions (2, 3), the
    ket pair `b, d`) or the bra pair's `(0, 1)` misses `ao/k.py` by far
    more than the tolerance."""
    want, s_mmd, mask = reference(cls)
    for wrong in ((2, 3), (0, 1)):
        K, s_hgp, *_ = kernel_k(cls, mode, density_=wrong)
        excess = np.abs(K - want)[mask] / (TOL * (s_hgp + s_mmd))[mask]
        assert np.max(excess) > 1e3, (wrong, float(np.max(excess)))


TRAP = [(1, 0, 1, 0), (0, 1, 2, 0), (1, 2, 0, 1), (0, 2, 1, 1)]


@pytest.mark.parametrize("ao", TRAP, ids=str)
def test_an_ao_order_am_builds_a_different_wrong_class(ao) -> None:
    """The AO-order trap (#196, #401): shells `(la, lb, lc, ld)` with
    `lb != lc`. K's class is `(la, lc | lb, ld)`; spelling `am` in AO
    order builds a DIFFERENT class (another orbit, another kernel) whose
    `K` lands on other blocks and misses `ao/k.py` far beyond the bound,
    in both modes -- while the Hermite spelling passes."""
    la, lb, lc, ld = ao
    assert lb != lc
    herm = (la, lc, lb, ld)
    for mode in MODES:
        right, wrong = _spec(herm, mode), _spec(ao, mode)
        assert right.am != wrong.am
        # another kernel, or (same orbit, canonical: one kernel per orbit)
        # the same kernel launched through another permutation
        perm = {c: orbits.GROUP.index(orbits.to_representative(c).shells)
                for c in (herm, ao)}
        if mode == "ordered":
            assert HK.hk_name(right) != HK.hk_name(wrong)
        else:
            assert (HK.hk_name(right) != HK.hk_name(wrong)
                    or perm[herm] != perm[ao])
        want, s_mmd, mask = reference(herm)
        basis = basis_for(herm)
        assert basis == basis_for(ao)  # same shells, same l multiset
        D = density(basis.n_ao)
        Kw, *_ = launch(ao, mode, basis, D, eris=False)
        Kr, s_hgp, *_ = kernel_k(herm, mode)
        bound = TOL * (s_hgp + s_mmd)
        assert np.all(np.abs(Kr - want)[mask] <= bound[mask])
        excess = np.abs(Kw - want)[mask] / bound[mask]
        assert np.max(excess) > 1e3, (mode, float(np.max(excess)))
        assert np.any(Kw[~mask] != 0.0)  # it deposits into other blocks


# ---------------------------------------------------------------------------
# The screen, the membership chain, and a whole K
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_screen_skips_one_ket_and_never_breaks(mode) -> None:
    cls = (0, 1, 1, 1)
    basis = basis_for(cls)
    D = density(basis.n_ao)
    K, g, bras, kets, gidx, it = launch(cls, mode, basis, D, tau=1e-14,
                                        G={1: 0.0})
    assert len(kets) >= 3
    full, gf, *_ = launch(cls, mode, basis, D)
    for ib in range(len(bras)):
        for ik in range(len(kets)):
            if ik == 1:
                assert np.all(g[ib, ik] == -7.0)
            else:
                assert np.array_equal(g[ib, ik], gf[ib, ik])
    assert it.iterations[HK.KET] == len(bras) * len(kets)
    assert it.iterations[HK.KAB] == len(bras) * (len(kets) - 1) * 4
    s = np.zeros_like(K)
    for b in bras:
        k = kets[1]
        deposit_scale(basis, D, (*b, *k), gidx,
                      np.abs(hgp(basis, (*b, *k))[0]), s)
    assert np.max(np.abs(full - K)) > 0.0
    assert np.all(np.abs(full - K) <= s * (1 + 1e-12) + 1e-300)


def test_a_perm_outside_the_orbit_deposits_nothing() -> None:
    cls = (1, 0, 0, 0)
    assert HK.orientations(cls) == (0, 2, 4, 6)
    basis = basis_for(cls)
    D = density(basis.n_ao)
    for perm in (1, 3, 5, 7, 8):
        K, *_ = launch(cls, "canonical", basis, D, perm=perm, eris=False)
        assert np.all(K == 0.0), perm


@pytest.mark.parametrize("mode", MODES)
def test_every_class_together_is_the_whole_ordered_sweep(mode) -> None:
    """On an s/p basis (every class L <= 4): the sum of all 16 K classes'
    launches is `ao.k.exchange_matrix_reference` (every ordered quadruple,
    weight 1), within the same derived bound; symmetric with no mirror."""
    basis = _sp_basis()
    D = density(basis.n_ao, 7)
    want = np.array(exchange_matrix_reference(basis, D, pq=pq))
    K = np.zeros_like(want)
    s_h = np.zeros_like(want)
    s_m = np.zeros_like(want)
    for cls in orbits.ordered_classes(4, lmax_shell=1):
        Kc, g, bras, kets, gidx, _ = launch(cls, mode, basis, D, eris=False)
        K += Kc
        for b in bras:
            for k in kets:
                xs = (*b, *k)
                deposit_scale(basis, D, xs, gidx, hgp(basis, xs)[1], s_h)
                inv = HK.inverse(orbits.GROUP[gidx])
                q = tuple(xs[inv[i]] for i in range(4))
                deposit_scale(basis, D, q, 0, mmd_scale(basis, q), s_m)
    assert np.all(np.abs(K - want) <= TOL * (s_h + s_m))
    assert np.all(np.abs(K - K.T) <= TOL * (s_h + s_h.T))
    assert np.max(np.abs(K)) > 0.1


def test_the_head_notes_name_the_class_the_mode_and_the_lever() -> None:
    k = _kernel((1, 0, 0, 0), "ordered")
    notes = [s.prov for s in k.body if isinstance(s, Note)]
    assert re.search(r"K class \(10\|00\) \[ordered", notes[0])
    assert notes[1] == lever_provenance(HK.ORDERED)
    c = _kernel((1, 0, 0, 0), "canonical")
    cnotes = [s.prov for s in c.body if isinstance(s, Note)]
    assert re.search(r"K class \(00\|10\) \[canonical", cnotes[0])
    assert lever_provenance(HK.ORDERED) not in cnotes
