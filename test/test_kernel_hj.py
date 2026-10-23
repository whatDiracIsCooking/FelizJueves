"""`kernel_hj.py`: the HGP Coulomb kernel, split (#404), interpreted.

Every ordered class with `L = la + lb + lc + ld <= 4` -- all 70, every
shell `l` up to 4 (`hgp.orbits.ordered_classes(4, lmax_shell=None)`) -- in
BOTH class modes: canonical (the orbit representative's kernel, launched
with the class's permutation) and `ordered` (the class's own kernel).
Nothing above L = 4 is built or run here.

THE ERIs, BITWISE. The tap `g` (`eris=True`) is `==` `ao/hgp_drive.py`'s
`eri_block` on the same (built-order) shells, bit for bit: the kernel's nest
is bra outer / ket inner and its quartet scalars are spelled in the drive's
operation order (kernel_hj's docstring), and the records here are formed by
the drive's own expressions (`records`). The tap is asserted to be the
shipped kernel plus its writes.

THE DEPOSIT, AGAINST `ao/j.py`'s SWEEP RESTRICTED TO THE CLASS. The witness
is `ao.j.coulomb_matrix` -- the MMD J engine, a different recurrence and a
different contraction (`D_q`, `J_p`, the bra transform) -- restricted
through its own `jp=` seam: the injected `J_p` evaluator is
`jp_from_pq(pq)` for a (bra pair, ket class) whose UNORDERED angular
momenta are the class's `{la, lb}` and `{lc, ld}`, and zero otherwise, so
no integral of another class is formed. That restricted unique sweep holds,
on the `J` blocks whose row shell has `l = la` and column shell `l = lb`,
exactly the ordered quartets of `(la lb | lc ld)` and, when `lc != ld`, of
its ket mirror `(la lb | ld lc)` (the sweep's ket weight 2 is `(ab|cd) D_cd
+ (ab|dc) D_dc`; its bra mirror lands on the transposed blocks). The two
ket orientations deposit EQUAL sums in exact arithmetic -- `D` symmetric
and `(ab|cd) = (ab|dc)`, relabel `c <-> d` -- so the class's own deposit is
`f ref` with `f = 1/2` when `lc != ld` and `1` when `lc = ld` (a halving,
exact in fp64). Every other `J` element the kernel writes must be exactly 0.

NOT BITWISE, AND THE TOLERANCE, DERIVED. The kernel and the witness are two
programs, so the comparison is

    |J_kernel - f ref| <= 1e-12 (s_hgp + f s_mmd),   element by element,

by the triangle inequality through the exact `J` of the class:
`|J_kernel - J| <= 1e-12 s_hgp` and `|f ref - J| <= f 1e-12 s_mmd`. Each
`s` is its engine's absolute-value twin carried through the (linear) `J`
sum on `|D|`: `s_hgp = SUM_q |D_cd| scale_hgp(q)` over the class's ordered
quartets with `ao.hgp_drive`'s twin in the orientation the kernel ran,
`s_mmd` the same over the quartets the restricted sweep summed (the class
and its ket mirror) with `ao.eri`'s twin -- which `ao/j.py`'s docstring
states is exactly its own `J` scale ("THE SCALE"). `1e-12` is
test/test_hgp_eri.py's `TOL`: `gamma_n` for `n` about 9000 roundings, the
per-ERI cross-engine bound that file justifies and pins (HGP against MMD
within `1e-12 (scale_mmd + scale_hgp)`); the deposit adds at most `n_cd
x n_quartets` (<= 81 x 16 here) roundings per element, inside that budget.
No existing bound is widened: this is the same constant, carried through a
linear map by the triangle inequality.

THE MODES AGREE: per class, canonical against ordered, `J` and every ERI
(the canonical block permuted to the ordered layout by `hgp.orbits`'
`gather`), within `1e-12 (scale_canonical + scale_ordered)` -- two HGP
orientations of one integral, each within `gamma_n` of its own twin.

CAN FAIL: a deposit reading the wrong density block (`density=` mutation)
misses the sweep by orders of magnitude beyond the tolerance, in both modes.

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
from tiers import split_by_l

from felizjueves.ao.j import coulomb_matrix, coulomb_matrix_reference, jp_from_pq
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri import hgp_drive as HD
from felizjueves.eri.eri import eri_block_and_scale as mmd_eri
from felizjueves.eri.hjk import kernel_hj as HJ
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
from felizjueves.kernel_f import PairRec, PrimRec, pack
from felizjueves.mmd.hermite import hermite_order
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target, lever_provenance

CLASSES = orbits.ordered_classes(4, lmax_shell=None)
LADDER = split_by_l(CLASSES, sum, ids=lambda c: "".join(map(str, c)))
MODES = ("canonical", "ordered")

# test/test_hgp_eri.py's TOL: the per-ERI cross-engine bound (module
# docstring, "THE TOLERANCE"), carried through J's linear sum.
TOL = 1e-12


def test_the_ladder_is_every_class_through_l4() -> None:
    assert len(CLASSES) == 70 and max(map(sum, CLASSES)) == 4
    assert len({orbits.representative(c) for c in CLASSES}) == 17


def _spec(cls, mode) -> Spec:
    return HJ.hj_class_spec(*cls, ordered=mode == "ordered")


# ---------------------------------------------------------------------------
# The key, the names, the refusals
# ---------------------------------------------------------------------------


def test_the_spec_is_hgp_j_and_the_names_cannot_collide() -> None:
    s = HJ.hj_class_spec(1, 0, 0, 0)
    assert (s.target, s.engine, s.am, s.levers) == (
        Target.J, Engine.HGP, (1, 0, 0, 0), ())
    assert s.key() == "coulomb/d0/(1,0|0,0)/b1/j.hgp/literal"
    o = HJ.hj_class_spec(1, 0, 0, 0, ordered=True)
    assert o.key() == "coulomb/d0/(1,0|0,0)/b1/j.hgp/ordered"
    # canonical: one name per orbit, the representative's; ordered: per class
    assert HJ.hj_name(s) == "hj_0010" == HJ.hj_name(HJ.hj_class_spec(0, 0, 0, 1))
    assert HJ.hj_name(o) == "hj_o_1000"
    canon = {HJ.hj_name(_spec(c, "canonical")) for c in CLASSES}
    ordered = {HJ.hj_name(_spec(c, "ordered")) for c in CLASSES}
    assert len(canon) == 17 and len(ordered) == 70 and not canon & ordered
    # two-digit l: `_`-separated, so (0,11,0,0) and (0,1,10,0) differ
    assert HJ.hj_name(HJ.hj_class_spec(0, 11, 0, 0, ordered=True)) == \
        "hj_o_0_11_0_0"
    names = {HJ.hj_name(HJ.hj_class_spec(*c, ordered=True))
             for c in ((0, 11, 0, 0), (0, 1, 10, 0), (0, 1, 1, 0))}
    assert len(names) == 3


def test_refusals() -> None:
    with pytest.raises(ValueError, match="Engine.HGP"):
        HJ.build_hj_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.J))
    with pytest.raises(ValueError, match="Engine.HGP"):
        HJ.build_hj_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.F))
    with pytest.raises(ValueError, match="n_batches"):
        HJ.build_hj_kernel(Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.J,
                                n_batches=3, engine=Engine.HGP))
    with pytest.raises(ValueError, match="no-sieve"):
        HJ.build_hj_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.J,
                                ("no-sieve",), engine=Engine.HGP))
    with pytest.raises(TypeError):
        HJ.build_hj_kernel([0, 0, 0, 0])  # type: ignore[arg-type]
    for bad in ((2, 2), (0, 4), (1,)):
        with pytest.raises(ValueError, match="density"):
            HJ.build_hj_kernel((0, 0, 0, 0), density=bad)


def test_params_follow_the_mode_the_hrr_and_the_tap() -> None:
    assert HJ.PERM in HJ.params_of((0, 0, 0, 0))
    assert HJ.PERM not in HJ.params_of(HJ.hj_class_spec(0, 0, 0, 0,
                                                        ordered=True))
    # canonical (1,0,0,0) builds (0,0,1,0): no AB, no CD (ld = 0)
    assert "AB" not in HJ.params_of((1, 0, 0, 0))
    assert "AB" in HJ.params_of(HJ.hj_class_spec(0, 1, 0, 0, ordered=True))
    assert "CD" in HJ.params_of(HJ.hj_class_spec(0, 0, 0, 1, ordered=True))
    for mode in MODES:
        for eris in (False, True):
            s = _spec((0, 1, 1, 1), mode)
            k = HJ.build_hj_kernel(s, eris=eris)
            assert tuple(p.name for p in k.params) == HJ.params_of(s, eris=eris)
    assert HJ.params_of((0, 0, 0, 0), eris=True)[-1] == HJ.OUT


def test_orientations_are_the_orbits_permutations() -> None:
    """Canonical: one `GROUP` index per distinct member's
    `to_representative` (the `perm` #403's stream passes); ordered: the
    identity alone. Every canonical deposit lands J on one of the built
    class's two pairs."""
    for cls in CLASSES:
        got = HJ.orientations(cls)
        want = sorted({orbits.GROUP.index(orbits.to_representative(m).shells)
                       for m in orbits.orbit(cls)})
        assert list(got) == want, cls
        assert HJ.orientations(_spec(cls, "ordered")) == (0,)
        for g in got:
            inv = HJ.inverse(orbits.GROUP[g])
            assert {inv[0], inv[1]} in ({0, 1}, {2, 3})
            assert orbits.apply(orbits.GROUP[g],
                                orbits.apply(inv, (0, 1, 2, 3))) == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# The counts and the shape
# ---------------------------------------------------------------------------


@cache
def _kernel(cls, mode, eris=False, density=HJ.COULOMB_DENSITY):
    return HJ.build_hj_kernel(_spec(cls, mode), eris=eris, density=density)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_statement_counts_are_the_built_classs_dags(cls, mode) -> None:
    """Read off the BUILT kernel, equal to `kernel_counts` and to
    `hgp.vrr` / `hgp.hrr`'s own counts for the class the mode builds."""
    spec = _spec(cls, mode)
    got = HJ.counted(_kernel(cls, mode))
    assert got == HJ.kernel_counts(spec)
    am = HJ.built_class(spec)
    assert am == (cls if mode == "ordered" else orbits.representative(cls))
    v, h = vrr.build(am), hrr.build(am)
    assert got.vrr == v.n_nodes
    assert got.boundary == h.n_boundary == len(vrr.target_set(am))
    assert got.hrr == h.n_statements == h.n_boundary + h.n_interior
    assert got.hrr_interior == h.n_interior
    assert got.eri == h.n_eri == math.prod(len(vrr.cartesians(x)) for x in am)
    n = [len(vrr.cartesians(x)) for x in am]
    want_atomic = 0
    for g in HJ.orientations(spec):
        inv = HJ.inverse(orbits.GROUP[g])
        want_atomic += n[inv[0]] * n[inv[1]]
    assert got.atomic == want_atomic


def _find_for(body, var):
    return next(s for s in walk(body) if isinstance(s, For) and s.var == var)


@pytest.mark.parametrize("mode", MODES)
def test_the_nest_the_staging_the_screen_and_no_break(mode) -> None:
    cls = (1, 1, 1, 1)
    k = _kernel(cls, mode)
    sh = [s for s in walk(k.body) if isinstance(s, Decl)
          and s.space is Space.SHARED]
    assert [s.name for s in sh] == [HJ.AB_SH]
    assert [d.value for d in sh[0].shape] == [HJ.MAX_PRIM_PAIRS, len(HJ.REC)]
    assert any(isinstance(s, Decl) and s.name == "bra"
               and s.init == Call(Fn.BLOCK_X) for s in walk(k.body))
    ket = _find_for(k.body, HJ.KET)
    assert ket.start == Call(Fn.THREAD_X) and ket.step == Call(Fn.BLOCK_DIM_X)
    order = list(walk(k.body))
    assert order.index(_find_for(k.body, HJ.STAGE)) < order.index(ket)
    # bra outer, ket inner, every VRR node inside, no HRR in the loops
    kab = _find_for(ket.body, HJ.KAB)
    kcd = _find_for(kab.body, HJ.KCD)
    inner = {s.name for s in walk(kcd.body) if isinstance(s, Decl)}
    assert sum(n.startswith("V") for n in inner) == vrr.build(cls).n_nodes
    hrr_names = {R.hrr_name(t) for t in hrr.build(cls).terms}
    assert not hrr_names & {s.name for s in walk(kab.body)
                            if isinstance(s, Decl)}
    # the screen: one If(bound < tau), a Note in its then, the work after
    screens = [s for s in walk(k.body) if isinstance(s, If)
               and s.cond == Call(Fn.LT, (Var("bound"), Var("tau")))]
    assert len(screens) == 1
    assert all(isinstance(x, Note) for x in screens[0].then)
    assert _find_for(screens[0].orelse, HJ.KAB)
    assert not any(isinstance(s, Break) for s in walk(k.body))
    # atomics only into J, nothing else written globally but shared staging
    assert all(s.dst.array == HJ.J_OUT for s in walk(k.body)
               if isinstance(s, AtomicAdd))
    assert not any(isinstance(s, Assign) and isinstance(s.dst, Index)
                   and s.dst.array != HJ.AB_SH for s in walk(k.body))


def _strip_tap(body):
    out = []
    for s in body:
        if isinstance(s, Assign) and isinstance(s.dst, Index) and \
                s.dst.array == HJ.OUT:
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
# Records in ao/hgp_drive's operation order, and the launches
# ---------------------------------------------------------------------------

CENTRES = ((0.0, 0.0, 0.0), (0.4, -0.3, 0.5), (-0.2, 0.6, 0.1),
           (0.7, 0.2, -0.4), (-0.5, -0.1, 0.3), (0.1, 0.45, -0.6),
           (0.3, -0.55, -0.2), (-0.35, 0.25, 0.65))
EXPS = ((1.3, 0.35), (0.7, 2.2), (0.9, 0.3), (1.6, 0.5))
COEFS = ((0.6, 0.45), (0.8, -0.3), (0.5, 0.7), (0.9, 0.25))


@cache
def basis_for(cls) -> Basis:
    """Two shells of each `l` the class repeats, one of each it does not:
    so a repeated `l` meets both `a = b` and `a != b` pairs, and every pair
    side has degree 4 (two primitives per shell). Distinct centres."""
    shells, i = [], 0
    for l in sorted(set(cls)):  # noqa: E741
        for _ in range(2 if cls.count(l) > 1 else 1):
            shells.append(Shell(l, CENTRES[i], EXPS[i % 4], COEFS[i % 4]))
            i += 1
    return Basis(tuple(shells))


def density(n_ao: int, seed: int = 404) -> np.ndarray:
    """Symmetric, sign-mixed, no structure (ao/j.py requires symmetry)."""
    r = np.random.default_rng(seed).uniform(-1.0, 1.0, (n_ao, n_ao))
    return r + r.T


def records(basis: Basis, a: int, b: int, G: float = 1.0) -> PairRec:
    """The ordered pair `(a, b)`, primitive pairs `i` outer, every field by
    `ao.hgp_drive.primitive_scalars`' own expressions -- so the kernel's
    scalars are the drive's bit for bit."""
    sa, sb = basis.shells[a], basis.shells[b]
    A, B = sa.centre, sb.centre
    AB = tuple(A[x] - B[x] for x in range(3))
    prims = []
    for i, x in enumerate(sa.exponents):
        for j, y in enumerate(sb.exponents):
            p = x + y
            P = tuple((x * A[d] + y * B[d]) / p for d in range(3))
            kab = math.exp(-x * y / p * (AB[0] ** 2 + AB[1] ** 2 + AB[2] ** 2))
            U = (sa.coefficients[i] * sb.coefficients[j]
                 * (math.pi / p) ** 1.5 * kab)
            prims.append(PrimRec(P, tuple(P[d] - A[d] for d in range(3)), p, U))
    return PairRec(AB, G, tuple(prims), (a, b),
                   (basis.offsets[a], basis.offsets[b]))


def pairs_of(basis: Basis, la: int, lb: int) -> list[tuple[int, int]]:
    ls = [s.l for s in basis.shells]
    return [(a, b) for a in range(len(ls)) for b in range(len(ls))
            if ls[a] == la and ls[b] == lb]


def dsh(basis: Basis, D: np.ndarray) -> np.ndarray:
    o, n = basis.offsets, [s.n_cart for s in basis.shells]
    m = len(n)
    return np.array([[np.max(np.abs(D[o[s]:o[s] + n[s], o[t]:o[t] + n[t]]))
                      for t in range(m)] for s in range(m)])


def launch(cls, mode, basis, D, *, tau=0.0, G=None, threads=2, eris=True,
           density_=HJ.COULOMB_DENSITY, perm=None):
    """One launch of the class in `mode` over every ordered quartet of the
    class in `basis`. Returns `(J, g, bras, kets, gidx)`: bras / kets are
    the BUILT-order shell pairs launched, `gidx` the permutation."""
    spec = _spec(cls, mode)
    am = HJ.built_class(spec)
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
        HJ.PERM: gidx if perm is None else perm, "tau": float(tau),
        "bra_prim": np.array(pb.prim), "AB": np.array(pb.sep),
        "G_ab": np.array(pb.G), "bra_sh": np.array(pb.sh, dtype=np.int32),
        "bra_ao": np.array(pb.ao, dtype=np.int32),
        "ket_prim": np.array(pk.prim), "CD": np.array(pk.sep),
        "G_cd": np.array(pk.G), "ket_sh": np.array(pk.sh, dtype=np.int32),
        "ket_ao": np.array(pk.ao, dtype=np.int32),
        "Dsh": dsh(basis, D).ravel(), "D": np.asarray(D, float).ravel(),
        HJ.J_OUT: np.zeros(n_ao * n_ao),
        HJ.OUT: np.full(pb.n * pk.n * n_eri, -7.0),
    }
    grid, block = HJ.launch_shape(pb.n, threads)
    it = Interpreter(k, launch=Launch(grid, block, 32), boys=boys)
    out = it.run(**{n: every[n] for n in HJ.params_of(spec, eris=eris)})
    J = out[HJ.J_OUT].reshape(n_ao, n_ao)
    g = out[HJ.OUT].reshape(pb.n, pk.n, n_eri) if eris else None
    return J, g, bras, kets, gidx, it


@cache
def hgp(basis: Basis, xs: tuple[int, int, int, int]):
    """`ao.hgp_drive`'s block and twin on the built-order shells `xs`,
    flat in `hgp.hrr.targets` order."""
    sh = basis.shells
    blk, scale = HD.eri_block_and_scale((sh[xs[0]], sh[xs[1]]),
                                        (sh[xs[2]], sh[xs[3]]), boys=boys)
    return np.array(blk).ravel(), np.array(scale).ravel()


@cache
def mmd_scale(basis: Basis, q: tuple[int, int, int, int]) -> np.ndarray:
    sh = basis.shells
    return np.array(mmd_eri((sh[q[0]], sh[q[1]]), (sh[q[2]], sh[q[3]]),
                            pq=pq)[1]).ravel()


def _block(flat, shells_l) -> np.ndarray:
    return flat.reshape([len(vrr.cartesians(x)) for x in shells_l])


def deposit_scale(basis, D, xs, gidx, scale_flat, out) -> None:
    """`out[a][b] += |D_cd| scale` for the ordered quartet `xs` stands for
    under orientation `gidx` (the kernel's deposit, on the twin)."""
    inv = HJ.inverse(orbits.GROUP[gidx])
    q = tuple(xs[inv[i]] for i in range(4))
    ls = [basis.shells[x].l for x in xs]
    blk = _block(scale_flat, ls).transpose(inv)  # ordered [a][b][c][d]
    o, n = basis.offsets, [basis.shells[x].n_cart for x in q]
    Dcd = np.abs(D[o[q[2]]:o[q[2]] + n[2], o[q[3]]:o[q[3]] + n[3]])
    out[o[q[0]]:o[q[0]] + n[0], o[q[1]]:o[q[1]] + n[1]] += np.einsum(
        "ijkl,kl->ij", blk, Dcd)


def ordered_quartets(basis, cls):
    return [(a, b, c, d) for a, b in pairs_of(basis, cls[0], cls[1])
            for c, d in pairs_of(basis, cls[2], cls[3])]


# ---------------------------------------------------------------------------
# The ERIs, bitwise against ao/hgp_drive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_the_tap_is_hgp_drive_bitwise(cls, mode) -> None:
    """Every ERI of every launched quartet `==` `hgp_drive.eri_block` on the
    same built-order shells, bit for bit."""
    basis = basis_for(cls)
    _, g, bras, kets, _, _ = launch(cls, mode, basis, density(basis.n_ao))
    for ib, b in enumerate(bras):
        for ik, k in enumerate(kets):
            want, _ = hgp(basis, (*b, *k))
            assert np.array_equal(g[ib, ik], want), (b, k)


# ---------------------------------------------------------------------------
# The deposit against ao/j.py's sweep, restricted to the class
# ---------------------------------------------------------------------------


def restricted_jp(cls):
    """`ao.j`'s `jp=` seam limited to the class's unordered bra and ket
    angular momenta; every other (bra pair, ket class) contributes zero
    and forms no integral (module docstring)."""
    full = jp_from_pq(pq)
    bra_l, ket_l = sorted(cls[:2]), sorted(cls[2:])

    def jp(bra_am, ket_am, bras, kets):
        if sorted(bra_am) == bra_l and sorted(ket_am) == ket_l:
            return full(bra_am, ket_am, bras, kets)
        return [[0.0] * len(hermite_order(sum(bra_am))) for _ in bras]

    return jp


@cache
def reference(cls):
    """`(f ref, f s_mmd, mask)` on the class's `J` blocks (module
    docstring): `mask` is the (row l = la, column l = lb) blocks."""
    basis = basis_for(cls)
    D = density(basis.n_ao)
    ref = np.array(coulomb_matrix(basis, D, pq=pq, jp=restricted_jp(cls)))
    la, lb, lc, ld = cls
    mirror = (la, lb, ld, lc)
    s = np.zeros_like(ref)
    for c in {cls, mirror}:
        for q in ordered_quartets(basis, c):
            deposit_scale(basis, D, q, 0, mmd_scale(basis, q), s)
    mask = np.zeros(ref.shape, dtype=bool)
    o = basis.offsets
    for a, b in pairs_of(basis, la, lb):
        sa, sb = basis.shells[a], basis.shells[b]
        mask[o[a]:o[a] + sa.n_cart, o[b]:o[b] + sb.n_cart] = True
    f = 0.5 if lc != ld else 1.0
    return f * ref, f * s, mask


def kernel_j(cls, mode, **kw):
    """The kernel's `J` and its HGP twin `s_hgp` (the orientation it ran)."""
    basis = basis_for(cls)
    D = density(basis.n_ao)
    J, g, bras, kets, gidx, it = launch(cls, mode, basis, D, **kw)
    s = np.zeros_like(J)
    for b in bras:
        for k in kets:
            xs = (*b, *k)
            deposit_scale(basis, D, xs, gidx, hgp(basis, xs)[1], s)
    return J, s, g, bras, kets, gidx


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", LADDER)
def test_the_deposit_is_ao_js_sweep_restricted_to_the_class(cls, mode) -> None:
    want, s_mmd, mask = reference(cls)
    J, s_hgp, *_ = kernel_j(cls, mode)
    assert np.all(J[~mask] == 0.0)  # J[a][b] only, from its own block
    err = np.abs(J - want)
    bound = TOL * (s_hgp + s_mmd)
    assert np.all(err[mask] <= bound[mask]), float(np.max(err / (bound + 1e-300)))
    assert np.max(np.abs(want[mask])) > 1e-3  # a non-trivial deposit


@pytest.mark.parametrize("cls", LADDER)
def test_canonical_and_ordered_agree(cls) -> None:
    """`J` and every ERI, the canonical block permuted to the ordered
    layout through `hgp.orbits`' `gather`."""
    Jc, sc, gc, bc, kc, gidx = kernel_j(cls, "canonical")
    Jo, so, go, bo, ko, _ = kernel_j(cls, "ordered")
    assert np.all(np.abs(Jc - Jo) <= TOL * (sc + so))
    basis = basis_for(cls)
    inv = HJ.inverse(orbits.GROUP[gidx])
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
    """The mutation: J_ab reading `D_bd` (exchange's block) or `D_ab`
    misses the sweep by far more than the tolerance."""
    want, s_mmd, mask = reference(cls)
    for wrong in ((1, 3), (0, 1)):
        J, s_hgp, *_ = kernel_j(cls, mode, density_=wrong)
        excess = np.abs(J - want)[mask] / (TOL * (s_hgp + s_mmd))[mask]
        assert np.max(excess) > 1e3, (wrong, float(np.max(excess)))


# ---------------------------------------------------------------------------
# The screen, the membership chain, and a whole J
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_screen_skips_one_ket_and_never_breaks(mode) -> None:
    """The middle ket's `G_cd = 0` puts it under `tau`: its tap row is never
    written and its quartets leave `J` alone, while the ket after it is
    still computed; the ket loop is entered for every ket, the primitive
    loops only for the unscreened ones."""
    cls = (0, 1, 1, 1)
    basis = basis_for(cls)
    D = density(basis.n_ao)
    J, g, bras, kets, gidx, it = launch(cls, mode, basis, D, tau=1e-14,
                                        G={1: 0.0})
    assert len(kets) >= 3
    full, gf, *_ = launch(cls, mode, basis, D)
    for ib in range(len(bras)):
        for ik in range(len(kets)):
            if ik == 1:
                assert np.all(g[ib, ik] == -7.0)
            else:
                assert np.array_equal(g[ib, ik], gf[ib, ik])
    assert it.iterations[HJ.KET] == len(bras) * len(kets)
    assert it.iterations[HJ.KAB] == len(bras) * (len(kets) - 1) * 4
    s = np.zeros_like(J)
    for b in bras:
        k = kets[1]
        deposit_scale(basis, D, (*b, *k), gidx,
                      np.abs(hgp(basis, (*b, *k))[0]), s)
    assert np.max(np.abs(full - J)) > 0.0  # the skip moved J ...
    assert np.all(np.abs(full - J) <= s * (1 + 1e-12) + 1e-300)  # ... by it


def test_a_perm_outside_the_orbit_deposits_nothing() -> None:
    """The membership chain is exact: `(1,0,0,0)`'s kernel takes perms
    {0, 2, 4, 6}; perms 1, 3, 5 and 7 compute (at tau = 0) and deposit
    nothing, and 8 likewise."""
    cls = (1, 0, 0, 0)
    assert HJ.orientations(cls) == (0, 2, 4, 6)
    basis = basis_for(cls)
    D = density(basis.n_ao)
    for perm in (1, 3, 5, 7, 8):
        J, *_ = launch(cls, "canonical", basis, D, perm=perm, eris=False)
        assert np.all(J == 0.0), perm


def _sp_basis() -> Basis:
    return Basis(tuple(Shell(am, CENTRES[i], EXPS[i], COEFS[i])
                       for i, am in enumerate((0, 1, 0, 1))))


@pytest.mark.parametrize("mode", MODES)
def test_every_class_together_is_the_whole_ordered_sweep(mode) -> None:
    """On an s/p basis (every class L <= 4): the sum over all 16 ordered
    classes' launches is `ao.j.coulomb_matrix_reference`'s whole `J`
    (every ordered bra and ket pair, weight 1), within the same derived
    bound; and `J` comes out symmetric to that bound with no mirror."""
    basis = _sp_basis()
    D = density(basis.n_ao, 7)
    want = np.array(coulomb_matrix_reference(basis, D, pq=pq))
    J = np.zeros_like(want)
    s_h = np.zeros_like(want)
    s_m = np.zeros_like(want)
    for cls in orbits.ordered_classes(4, lmax_shell=1):
        Jc, g, bras, kets, gidx, _ = launch(cls, mode, basis, D, eris=False)
        J += Jc
        for b in bras:
            for k in kets:
                xs = (*b, *k)
                deposit_scale(basis, D, xs, gidx, hgp(basis, xs)[1], s_h)
                inv = HJ.inverse(orbits.GROUP[gidx])
                q = tuple(xs[inv[i]] for i in range(4))
                deposit_scale(basis, D, q, 0, mmd_scale(basis, q), s_m)
    assert np.all(np.abs(J - want) <= TOL * (s_h + s_m))
    assert np.all(np.abs(J - J.T) <= TOL * (s_h + s_h.T))
    assert np.max(np.abs(J)) > 0.1


def test_the_head_notes_name_the_class_the_mode_and_the_lever() -> None:
    """The head Note names the built class and its mode; the `ordered`
    lever's provenance line sits right under it, and only there."""
    k = _kernel((1, 0, 0, 0), "ordered")
    notes = [s.prov for s in k.body if isinstance(s, Note)]
    assert re.search(r"J class \(10\|00\) \[ordered", notes[0])
    assert notes[1] == lever_provenance(HJ.ORDERED)
    c = _kernel((1, 0, 0, 0), "canonical")
    cnotes = [s.prov for s in c.body if isinstance(s, Note)]
    assert re.search(r"J class \(00\|10\) \[canonical", cnotes[0])
    assert lever_provenance(HJ.ORDERED) not in cnotes
