"""`kernel_eri3`: the three-center DF integral kernel, `Target.ERI3`, interpreted.

THE WITNESS is `ao.eri3` (#443: `class_block` of each staged cell with
`reference.pq.pq` injected, placed by `place_staged`); the kernel runs over
`ao.eri3.eri3_classes`' staging through `ao.eri3_drive`, its Boys function
`reference.boys.boys`, injected into the interpreter. Through the tracked
schema-4 set `h2o_sto3g_jkfit_3c` (#442) the host is libcint's
`int3c2e_cart` (test/test_ao_eri3.py), so the fixture cases here also compare
the interpreted kernel against libcint directly.

ROUND-OFF, NOT BITWISE (`kernel_eri3`'s module docstring): the host's
`[t|tau]` is `reference.pq`'s recursion, the kernel's the plan's MILD DAG,
and the host dresses each `E` entry where the kernel multiplies `s_ij s_k`
once. No class's term order matches the host's, so none is held bitwise
(`test_no_class_is_bitwise_and_why` records it). THE BOUNDS:

- against the host, per element, `|T - host| <= TOL_ULPS * eps * scale`,
  `TOL_ULPS = 16`, `scale` the absolute-value twin of the same sum with
  every `[t+tau]` replaced by its TOWER's absolute-value twin
  (`test_kernel_metric.pq_tower`, imported, whose docstring says why the
  per-target `|[t+tau]|` is unsound). An element whose scale is 0 must be
  exactly 0.
- against libcint, `TOL = 1e-12` of the TENSOR scale `max |(mn|P)|` of the
  fixture, test_ao_eri3.py's bound for the host on the same data.

SCOPE: correctness is tested at `la + lb + lP <= 4` only (the milestone's
cap, `tiers.FAST_L_MAX`) -- every class `la <= lb` -- classes whose shells
are all s or p fast and anything with a `d` or higher shell `slow`, as
test_kernel_metric.py. One class above the cap is BUILT here (`slow`, by
#419's rule) and never interpreted. Nothing is emitted, nothing compiles,
nothing launches.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_kernel_metric import pq_tower
from tiers import FAST_L_MAX, by_l

from felizjueves import e_fragment as ef
from felizjueves.ao.eri3 import (
    class_block,
    eri3_block_and_scale,
    eri3_classes,
    place_staged,
)
from felizjueves.ao.eri3_drive import eri3_kernel, eri3_tensor_kernel
from felizjueves.ao.shell import Basis, Shell
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Decl,
    Fn,
    Index,
    Inline,
    Kernel,
    walk,
)
from felizjueves.kernel_eri3 import (
    OUT,
    build_eri3_kernel,
    eri3_plan,
    eri3_spec,
    launch_shape,
    live_targets,
    quartet_spec,
)
from felizjueves.plan.pipeline import plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.recurrence import node_name
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL_ULPS = 16  # x eps x scale, per element, against the host
TOL = 1e-12  # x max |(mn|P)| of the fixture, against libcint
SET = "h2o_sto3g_jkfit_3c"
L_CAP = FAST_L_MAX  # 4
CLASSES = [(a, b, p) for a in range(L_CAP + 1) for b in range(a, L_CAP + 1)
           for p in range(L_CAP + 1) if a + b + p <= L_CAP]


def _id(c: tuple[int, int, int]) -> str:
    return f"{'spdfg'[c[0]]}{'spdfg'[c[1]]}|{'spdfg'[c[2]]}"


def _param(c):
    p = by_l(c, sum(c), id=_id(c))
    if max(c) <= 1:
        return p
    return pytest.param(*p.values, id=p.id, marks=(*p.marks, pytest.mark.slow))


A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
# The AO side: every l to g, the higher l FIRST on each centre (so every
# la < lb class has cells staged swapped), two shells of each l <= 2 (so every
# la == lb class has a diagonal and an off-diagonal pair), one-centre pairs,
# and contracted and single primitives.
SYNTHETIC = Basis((
    Shell(4, A, (0.9,), (1.0,)),
    Shell(2, A, (1.1, 0.3), (0.5, 0.6)),
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(0, A, (3.4, 0.62, 0.17), (0.15, 0.53, 0.44)),
    Shell(3, B, (0.6, 0.2), (0.7, 0.4)),
    Shell(2, B, (0.8,), (1.0,)),
    Shell(0, B, (1.3, 0.21), (0.6, 0.47)),
    Shell(1, B, (0.9,), (1.0,)),
))
# The aux side: every l to g, on all three centres between them (so every
# class has a one-centre triple, an AO pair and an aux shell on one centre,
# and two- and three-centre ones), contracted and not.
SYNTHETIC_AUX = Basis((
    Shell(0, C, (2.0, 0.4), (0.7, 0.5)),
    Shell(1, A, (0.85,), (1.0,)),
    Shell(2, B, (1.4, 0.35), (0.6, 0.45)),
    Shell(3, B, (0.7,), (1.0,)),
    Shell(4, A, (1.2,), (1.0,)),
    Shell(0, A, (5.0,), (1.0,)),
    Shell(0, B, (3.0,), (1.0,)),
    Shell(1, B, (1.1,), (1.0,)),
))
# A tight s aux primitive two bohr from a tight s pair: T = (2000 * 1000 /
# 3000) * 4 ~ 2667, past the narrow Boys series' overflow near 709.
TIGHT = Basis((
    Shell(0, A, (1000.0, 1.0), (0.4, 0.6)),
    Shell(1, A, (1000.0, 0.8), (0.5, 0.5)),
))
TIGHT_AUX = Basis((
    Shell(0, (0.0, 0.0, 2.0), (1000.0, 0.7), (0.5, 0.5)),
    Shell(1, (0.0, 0.0, 2.0), (1000.0,), (1.0,)),
))


@cache
def _ts() -> manifest.ThreeCenterSet:
    return manifest.three_center_set(SET)


@cache
def _basis() -> Basis:
    return ob.from_fixture(_ts().parent)


@cache
def _aux() -> Basis:
    return ob.from_fixture(_ts())


def _scale(sA, sB, sP):
    """The staged block's scale: `eri3_block_and_scale`'s |.| sum over the
    tower twin (`pq_tower`)."""
    return eri3_block_and_scale(sA, sB, sP, pq=pq_tower)[1]


def _host(basis, aux, ran):
    """The host's `(mn|P)` and its scale on the classes `ran`, placed by
    `place_staged` (0.0 elsewhere), and the mask of those elements."""
    n, naux = basis.n_ao, aux.n_ao
    want = [[[0.0] * naux for _ in range(n)] for _ in range(n)]
    scale = [[[0.0] * naux for _ in range(n)] for _ in range(n)]
    owned = [[[0.0] * naux for _ in range(n)] for _ in range(n)]
    sh, ash = basis.shells, aux.shells
    for c in eri3_classes(basis, aux):
        if (c.la, c.lb, c.lP) not in ran:
            continue
        for k, (a, b, p) in enumerate(c.cells):
            blk = class_block(c, k, pq=pq)
            place_staged(c, k, blk, want)
            place_staged(c, k, _scale(sh[a], sh[b], ash[p]), scale)
            place_staged(c, k, [[[1.0] * len(r) for r in pl] for pl in blk],
                         owned)
    return np.array(want), np.array(scale), np.array(owned) == 1.0


def _check_host(T, want, scale, owned, label):
    err = np.abs(T - want)
    pos = owned & (scale > 0)
    worst = float(np.max(err[pos] / (EPS * scale[pos]))) if pos.any() else 0.0
    print(f"\n{label}: max |T - host| = {err.max():.2e}, worst "
          f"{worst:.2f} eps scale")
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), label
    assert np.all(T[~owned] == 0.0)  # exactly the launched cells' elements
    assert np.array_equal(T, T.transpose(1, 0, 2))  # the (m, n) mirror


def test_the_synthetic_bases_reach_every_class_swapped_and_diagonal():
    by = {(c.la, c.lb, c.lP): c for c in eri3_classes(SYNTHETIC,
                                                      SYNTHETIC_AUX)}
    for cls in CLASSES:
        c = by[cls]
        pairs = {(a, b) for a, b, _ in c.cells}
        if cls[0] == cls[1]:
            assert any(a == b for a, b in pairs), cls
            assert any(a != b for a, b in pairs), cls
        else:
            assert any(c.swapped), cls
        # a one-centre triple: AO pair and aux shell on one centre
        sh, ash = SYNTHETIC.shells, SYNTHETIC_AUX.shells
        assert any(sh[a].centre == sh[b].centre == ash[p].centre
                   for a, b, p in c.cells), cls


@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES])
def test_one_launch_over_the_class_agrees_with_the_host(cls):
    run = eri3_tensor_kernel(SYNTHETIC, SYNTHETIC_AUX, boys=boys,
                             classes={cls}, threads=5)
    assert run.ran == (cls,)
    want, scale, owned = _host(SYNTHETIC, SYNTHETIC_AUX, {cls})
    assert owned.any()
    _check_host(run.T, want, scale, owned, f"({_id(cls)}), synthetic")


def _block(T, basis, aux, i, j, k):
    off, sh, aoff, ash = basis.offsets, basis.shells, aux.offsets, aux.shells
    return T[off[i]:off[i] + sh[i].n_cart, off[j]:off[j] + sh[j].n_cart,
             aoff[k]:aoff[k] + ash[k].n_cart]


@pytest.mark.parametrize("cls", [_param((0, 1, 1)), _param((1, 2, 0))])
def test_a_swapped_pair_lands_in_basis_order(cls):
    """A pair `(i, j)`, `i < j`, with `l_i > l_j` is staged `(j, i)`; the
    kernel's block at the basis order's `[off_i][off_j]` is `(ij|P)` of the
    basis-order shells to the bound, and its mirror the transpose, exactly."""
    sh, ash = SYNTHETIC.shells, SYNTHETIC_AUX.shells
    run = eri3_tensor_kernel(SYNTHETIC, SYNTHETIC_AUX, boys=boys,
                             classes={cls})
    c = next(c for c in eri3_classes(SYNTHETIC, SYNTHETIC_AUX)
             if (c.la, c.lb, c.lP) == cls)
    kk = next(k for k, (b, a, p) in enumerate(c.cells) if c.swapped[k]
              and sh[a].centre != sh[b].centre)  # two centres: nonzero
    j, i, p = c.cells[kk]
    assert i < j and sh[i].l > sh[j].l
    want, _ = eri3_block_and_scale(sh[i], sh[j], ash[p], pq=pq)
    scale = np.array(_scale(sh[i], sh[j], ash[p]))
    got = _block(run.T, SYNTHETIC, SYNTHETIC_AUX, i, j, p)
    assert got.shape == (sh[i].n_cart, sh[j].n_cart, ash[p].n_cart)
    assert np.all(np.abs(got - np.array(want)) <= TOL_ULPS * EPS * scale)
    assert np.array_equal(_block(run.T, SYNTHETIC, SYNTHETIC_AUX, j, i, p),
                          got.transpose(1, 0, 2))
    assert np.any(got)


@pytest.mark.parametrize("cls", [_param((1, 1, 0)), _param((1, 1, 2))])
def test_a_diagonal_pair_is_its_own_symmetric_block(cls):
    """A shell against itself writes `a >= b`, mirrored: the block is exactly
    symmetric in `(m, n)` and `(mm|P)` of the host to the bound."""
    sh, ash = SYNTHETIC.shells, SYNTHETIC_AUX.shells
    run = eri3_tensor_kernel(SYNTHETIC, SYNTHETIC_AUX, boys=boys,
                             classes={cls})
    c = next(c for c in eri3_classes(SYNTHETIC, SYNTHETIC_AUX)
             if (c.la, c.lb, c.lP) == cls)
    a, _, p = next(t for t in c.cells if t[0] == t[1]
                   and sh[t[0]].centre != ash[t[2]].centre)
    got = _block(run.T, SYNTHETIC, SYNTHETIC_AUX, a, a, p)
    want, _ = eri3_block_and_scale(sh[a], sh[a], ash[p], pq=pq)
    scale = np.array(_scale(sh[a], sh[a], ash[p]))
    assert np.array_equal(got, got.transpose(1, 0, 2))
    assert np.all(np.abs(got - np.array(want)) <= TOL_ULPS * EPS * scale)
    assert np.any(got)


def test_a_tight_triple_past_the_narrow_boys_range():
    """`T` reaches ~2667 here, past where the old narrow series overflowed
    (near 709): the one `fj_boys` (#301) holds there."""
    c = next(c for c in eri3_classes(TIGHT, TIGHT_AUX)
             if (c.la, c.lb, c.lP) == (0, 0, 0))
    T = max((c.zeta_ab[i] * c.alpha[k] / (c.zeta_ab[i] + c.alpha[k]))
            * sum((c.P[i][d] - c.A[k][d]) ** 2 for d in range(3))
            for i in range(c.n_bra_rec) for k in range(c.n_aux_rec))
    assert T > 709.0
    run = eri3_tensor_kernel(TIGHT, TIGHT_AUX, boys=boys)
    want, scale, owned = _host(TIGHT, TIGHT_AUX, set(run.ran))
    assert owned.all()
    _check_host(run.T, want, scale, owned, "tight")


def test_no_class_is_bitwise_and_why():
    """The host and the kernel agree to rounding, not bitwise, even at
    (ss|s): the host forms `((s_ab E) [0]) (s E_P)` per (bra, aux) pair
    and the kernel `(s_ab s_k) (E ([0] E_P))`, with `[0]` the reference's
    Boys path against the prelude's. Recorded so a change that makes them
    bitwise is noticed (and the issue's "bitwise where the term order
    matches" is answered: nowhere)."""
    run = eri3_tensor_kernel(SYNTHETIC, SYNTHETIC_AUX, boys=boys,
                             classes={(0, 0, 0)})
    want, scale, owned = _host(SYNTHETIC, SYNTHETIC_AUX, {(0, 0, 0)})
    _check_host(run.T, want, scale, owned, "(ss|s)")
    assert not np.array_equal(run.T[owned], want[owned])


# ---------------------------------------------------------------------------
# The fixture: libcint's int3c2e_cart over H2O/STO-3G x cc-pVTZ-jkfit
# ---------------------------------------------------------------------------


_LMAX = [pytest.param(1, id="sp"),
         pytest.param(2, id="spd", marks=pytest.mark.slow)]


@pytest.mark.parametrize("lmax", _LMAX)
def test_the_tensor_over_a_capped_aux_basis_is_libcints(lmax):
    """Every class of H2O/STO-3G's AOs (s, p) against the aux shells with
    `l <= lmax` (so every triple is at `L <= 2 + lmax <= 4`): the whole
    tensor against the host and against the same columns of the fixture."""
    aux = _aux()
    keep = [k for k, s in enumerate(aux.shells) if s.l <= lmax]
    sub = Basis(tuple(aux.shells[k] for k in keep))
    cols = [aux.offsets[k] + q for k in keep
            for q in range(aux.shells[k].n_cart)]
    basis = _basis()
    assert 2 * max(s.l for s in basis.shells) + lmax <= L_CAP
    run = eri3_tensor_kernel(basis, sub, boys=boys)
    want, scale, owned = _host(basis, sub, set(run.ran))
    assert owned.all()
    _check_host(run.T, want, scale, owned, f"aux l <= {lmax}, host")
    ref = _ts().tensor[:, :, cols]
    st = float(np.max(np.abs(_ts().tensor)))
    worst = float(np.max(np.abs(run.T - ref))) / st
    print(f"aux l <= {lmax}: max |T - libcint| = {worst:.2e} of max|(mn|P)|")
    assert worst <= TOL, worst


# ---------------------------------------------------------------------------
# What the kernel is built from
# ---------------------------------------------------------------------------


def _parity_targets(la: int, lb: int, lP: int) -> set[tuple[int, int, int]]:
    """Every `|r| <= L` with `|r| >= lP mod 2` (an odd-`lP` aux row has no
    live `tau = 0`), and, when the bra is (ss) (only `t = 0`), `|r| = lP
    (mod 2)` -- derived from the parity of the single-center `E` alone,
    independently of the builder (`kernel_eri3`'s module docstring)."""
    L = la + lb + lP
    return {r for r in hermite_targets(L) if sum(r) >= lP % 2
            and (la + lb > 0 or (sum(r) - lP) % 2 == 0)}


_KEPT = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1),
         (1, 1, 1), (0, 2, 1), (1, 2, 1), (0, 1, 3), (2, 2, 0), (0, 0, 4)]


@pytest.mark.parametrize("cls", [by_l(c, sum(c), id=_id(c)) for c in _KEPT])
def test_the_recurrence_is_the_quartets_plan_minus_the_unread_targets(cls):
    """Every recurrence `Decl` is the J quartet `(la lb|lP 0)`'s node, the
    plan is that quartet's, and the kept target set is exactly what the
    contraction reads: every `[r]^(0)` reachable from a live `t + tau`
    (`_parity_targets`), no other."""
    la, lb, lP = cls
    assert quartet_spec(cls) == Spec(Operator.COULOMB, 0, (la, lb, lP, 0),
                                     Target.J)
    dag, _ = eri3_plan(cls)
    (want_dag,) = plan(quartet_spec(cls)).dags  # BatchDag compares by id
    assert (dag.batch, dag.targets, dag.parents) == (
        want_dag.batch, want_dag.targets, want_dag.parents)
    k = build_eri3_kernel(cls)
    decls = {s.name for s in walk(k.body) if isinstance(s, Decl)}
    kept = {r for r, m in dag.targets if m == 0
            and node_name((r, 0)) in decls}
    assert kept == set(live_targets(la, lb, lP)) == _parity_targets(*cls)
    assert decls & ({node_name(n) for n in dag.parents}
                    | {node_name(n) for n in dag.targets})


def test_both_e_blocks_are_inlined_bra_then_ket():
    k = build_eri3_kernel((1, 2, 1))
    inl = [s for s in walk(k.body, into_inline=False) if isinstance(s, Inline)]
    assert [s.fragment for s in inl] == [ef.block_fragment(1, 2),
                                         ef.center_fragment(1)]
    assert [dict(s.bindings)[ef.E_ARRAY].name for s in inl] == [
        "Ebra", "Eket"]


def test_the_scatter_is_plain_stores_and_the_key_is_a_triple():
    k = build_eri3_kernel((1, 1, 1))
    assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    stores = [s for s in walk(k.body) if isinstance(s, Assign)
              and isinstance(s.dst, Index) and s.dst.array == OUT]
    # (pp|p): 27 elements, off-diagonal 2 x 27, diagonal (6 + 3 mirrored) x 3
    assert len(stores) == 54 + 27
    assert len([s for s in walk(build_eri3_kernel((0, 1, 2)).body)
                if isinstance(s, Assign) and isinstance(s.dst, Index)
                and s.dst.array == OUT]) == 2 * 18  # no diagonal branch
    assert eri3_spec(1, 2, 3).key() == "coulomb/d0/(1,2|3)/b1/eri3/literal"
    assert eri3_kernel(1, 1, 1) is eri3_kernel(1, 1, 1)  # memoised
    with pytest.raises(ValueError, match="by decision"):
        Spec(Operator.COULOMB, 0, (1, 1, 1), Target.ERI3, n_batches=3)
    with pytest.raises(ValueError, match="not a Target.ERI3 key"):
        build_eri3_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.METRIC))
    with pytest.raises(ValueError):
        launch_shape(0)
    assert launch_shape(33, 32) == (2, (32, 1))


def test_the_ket_sign_is_one_sign_per_class():
    """Every live `tau` of an `lP` row has `|tau| = lP (mod 2)`, so the
    accumulate is `acc - w v` exactly when `lP` is odd."""
    for cls, fn in (((0, 0, 1), Fn.SUB), ((1, 1, 2), Fn.ADD),
                    ((0, 0, 0), Fn.ADD), ((0, 1, 3), Fn.SUB)):
        k = build_eri3_kernel(cls)
        accs = [s for s in walk(k.body) if isinstance(s, Assign)
                and not isinstance(s.dst, Index)
                and s.dst.name.startswith("acc")]
        assert accs and {s.value.fn for s in accs} == {fn}, cls


@pytest.mark.slow  # L = 5 > FAST_L_MAX (tiers.py): built, never interpreted
def test_a_class_above_the_cap_builds():
    """The builder is arbitrary-L: (pd|d) at L = 5 builds, its kept targets
    the parity set. NOT checked numerically (the milestone's L <= 4 cap)."""
    k = build_eri3_kernel((1, 2, 2))
    assert isinstance(k, Kernel) and k.name == "eri3_122"
    decls = {s.name for s in walk(k.body) if isinstance(s, Decl)}
    assert {r for r in hermite_targets(5)
            if node_name((r, 0)) in decls} == _parity_targets(1, 2, 2)
