"""`kernel_metric`: the DF Coulomb metric kernel, `Target.METRIC`, interpreted.

THE WITNESS is `ao.metric` (#412: `class_block` of each staged aux shell pair
with `reference.pq.pq` injected, placed by `place_staged`); the kernel runs
over `ao.metric.metric_classes`' staging through `ao.metric_drive`, its Boys
function `reference.boys.boys`, injected into the interpreter. Through the
tracked cc-pVTZ-jkfit set (#411) the host is libcint's `int2c2e_cart`
(test/test_ao_metric.py), so the whole-matrix cases here also compare the
interpreted kernel against libcint directly.

ROUND-OFF, NOT BITWISE (`kernel_metric`'s module docstring): the host's
`[t|tau]` is `reference.pq`'s recursion, the kernel's the plan's MILD DAG,
and the host dresses each `E` entry where the kernel multiplies `s_i s_k`
once. THE BOUNDS:

- against the host, per element, `|M - host| <= TOL_ULPS * eps * scale`,
  `TOL_ULPS = 16`, `scale` the absolute-value twin of the same sum with
  every `[t+tau]` replaced by its TOWER's absolute-value twin (`pq_tower`,
  whose docstring says why the per-target `|[t+tau]|` is not enough). That
  scale was changed during #414: `16 eps` of the per-target scale was tried
  first and failed at 122.9 and 229.3 eps on the fixture's (p|p) blocks,
  where host and kernel are equally far from the exact value (mpmath, the
  PR). An element whose scale is 0 (one centre, odd total parity: every
  Hermite term vanishes) must be exactly 0. Measured worst at writing: 2.60
  eps scale on the fixture, 1.78 on the synthetic basis.
- against libcint, `TOL = 1e-12` of the MATRIX scale `max |M|` of the
  fixture (~96.5), test_ao_metric.py's bound for the host on the same data.

SCOPE: correctness is tested at `lP + lQ <= 4` only (the milestone's cap,
`tiers.FAST_L_MAX`) -- the nine classes `lP <= lQ` -- s/p classes fast and
anything with a `d` or higher shell `slow`, as test_ao_metric.py. The
emitter accepts any L; one class above the cap is BUILT here (`slow`, by
#419's rule) and never interpreted. Nothing compiles, nothing launches.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import FAST_L_MAX, by_l

from felizjueves import e_fragment as ef
from felizjueves.ao.metric import (
    class_block,
    metric_block_and_scale,
    metric_classes,
    place_staged,
)
from felizjueves.ao.metric_drive import metric_kernel, metric_matrix_kernel
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
from felizjueves.kernel_metric import (
    OUT,
    build_metric_kernel,
    launch_shape,
    live_targets,
    metric_plan,
    metric_spec,
    quartet_spec,
)
from felizjueves.plan.pipeline import plan
from felizjueves.plan.targets import hermite_targets
from felizjueves.recurrence import node_name
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import fundamental, pq
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL_ULPS = 16  # x eps x scale, per element, against the host
TOL = 1e-12  # x max |M| of the fixture, against libcint
SET = "h2o_ccpvtz_jkfit_metric"
L_CAP = FAST_L_MAX  # 4
CLASSES = [(a, b) for a in range(L_CAP + 1) for b in range(a, L_CAP + 1)
           if a + b <= L_CAP]
FAST_CLASSES = {(0, 0), (0, 1), (1, 1)}


def _param(c):
    p = by_l(c, sum(c), id=f"{'spdfg'[c[0]]}|{'spdfg'[c[1]]}")
    if c in FAST_CLASSES:
        return p
    return pytest.param(*p.values, id=p.id, marks=(*p.marks, pytest.mark.slow))


A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
# Every l to g on three centres, the higher l FIRST on each centre (so every
# lP < lQ class has pairs staged swapped), two shells of each l <= 2 (so every
# lP == lQ class has a diagonal and an off-diagonal pair), one-centre pairs of
# odd total parity (exactly-zero blocks), and contracted and single
# primitives.
SYNTHETIC = Basis((
    Shell(4, A, (0.9,), (1.0,)),
    Shell(2, A, (1.1, 0.3), (0.5, 0.6)),
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(0, A, (3.4, 0.62, 0.17), (0.15, 0.53, 0.44)),
    Shell(3, B, (0.6, 0.2), (0.7, 0.4)),
    Shell(2, B, (0.8,), (1.0,)),
    Shell(0, B, (1.3, 0.21), (0.6, 0.47)),
    Shell(1, B, (0.9,), (1.0,)),
    Shell(1, C, (0.55, 0.12), (0.8, 0.3)),
    Shell(0, C, (2.0,), (1.0,)),
))
# Two tight s primitives two bohr apart: T = (1000 * 1000 / 2000) * 4 = 2000,
# past the narrow Boys series' overflow near 709.
TIGHT = Basis((
    Shell(0, A, (1000.0, 1.0), (0.4, 0.6)),
    Shell(1, (0.0, 0.0, 2.0), (1000.0, 0.8), (0.5, 0.5)),
))


@cache
def _ms() -> manifest.MetricSet:
    return manifest.metric_set(SET)


@cache
def _aux() -> Basis:
    return ob.from_fixture(_ms())


def pq_tower(p, q, R, theta):
    """The absolute-value twin of `[p+q]^(0)`'s TOWER: the same raising
    recursion, `[r + 1_d]^(m) = r_d [r - 1_d]^(m+1) + X_d [r]^(m+1)`, over
    `|X_d|` and `|[0]^(m)|` (`reference.pq.fundamental`), so every term is
    added as its absolute value. Its coefficients are nonnegative, so it is
    the same polynomial whichever axis each step raises: it bounds the terms
    of `reference.pq`'s recursion AND of the plan's MILD DAG alike.

    WHY NOT THE PER-TARGET `|[t+tau]|`. `metric_block_and_scale`'s scale
    takes each `[t+tau]` as one atom, so it is blind to the cancellation
    INSIDE the tower that forms it. For (p|p), `[2,0,0]^(0) = X^2 [0]^(2) +
    [0]^(1)`, two terms of opposite sign (`[0]^(m)` carries `(-2 theta^2)^m`):
    with one primitive on each side and the two centres ~1.8 bohr apart the
    sum has one term, the per-target scale is just `|value|`, and an
    honest few-ulp rounding of each tower term reads as 123-229 eps of it.
    Host and kernel are then equally far from the exact value (mpmath, #414's
    PR), so that reading measures the yardstick, not the kernel."""
    X = tuple(abs(x) for x in R)

    @cache
    def tower(r, m):
        if any(x < 0 for x in r):
            return 0.0
        if r == (0, 0, 0):
            return abs(fundamental(m, R, theta))
        d = next(k for k in range(3) if r[k] > 0)
        r1 = list(r)
        r1[d] -= 1
        r2 = list(r1)
        r2[d] -= 1
        return (r1[d] * tower(tuple(r2), m + 1)
                + X[d] * tower(tuple(r1), m + 1))

    return tower((p[0] + q[0], p[1] + q[1], p[2] + q[2]), 0)


def _scale(sP, sQ):
    """The block's scale: `metric_block_and_scale`'s |.| sum over the
    tower twin (`pq_tower`)."""
    return np.array(metric_block_and_scale(sP, sQ, pq=pq_tower)[1])


def _host(basis, ran):
    """The host's `(P|Q)` and its scale on the classes `ran`, placed by
    `place_staged` (0.0 elsewhere), and the mask of those elements."""
    n = basis.n_ao
    want, scale = [[0.0] * n for _ in range(n)], [[0.0] * n for _ in range(n)]
    owned = np.zeros((n, n), bool)
    for c in metric_classes(basis):
        if (c.lP, c.lQ) not in ran:
            continue
        for k, (p, q) in enumerate(c.pairs):
            blk = class_block(c, k, pq=pq)
            sc = _scale(basis.shells[p], basis.shells[q])
            place_staged(c, k, blk, want)
            place_staged(c, k, sc, scale)
            op, oq = c.off_p[k], c.off_q[k]
            owned[op:op + len(blk), oq:oq + len(blk[0])] = True
    return np.array(want), np.array(scale), owned | owned.T


def _check_host(M, want, scale, owned, label):
    err = np.abs(M - want)
    pos = owned & (scale > 0)
    worst = float(np.max(err[pos] / (EPS * scale[pos]))) if pos.any() else 0.0
    print(f"\n{label}: max |M - host| = {err.max():.2e}, worst "
          f"{worst:.2f} eps scale")
    assert np.all(err[owned] <= TOL_ULPS * EPS * scale[owned]), label
    assert np.all(M[~owned] == 0.0)  # exactly the launched pairs' elements
    assert np.array_equal(M, M.T)  # the scatter mirrors one value


def test_the_synthetic_basis_reaches_every_class_swapped_and_diagonal():
    by = {(c.lP, c.lQ): c for c in metric_classes(SYNTHETIC)}
    for cls in CLASSES:
        c = by[cls]
        if cls[0] == cls[1]:
            assert any(p == q for p, q in c.pairs), cls
            assert any(p != q for p, q in c.pairs), cls
        else:
            assert any(c.swapped), cls


@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES])
def test_one_launch_over_the_class_agrees_with_the_host(cls):
    run = metric_matrix_kernel(SYNTHETIC, boys=boys, classes={cls},
                               threads=3)
    assert run.ran == (cls,)
    want, scale, owned = _host(SYNTHETIC, {cls})
    _check_host(run.M, want, scale, owned,
                f"({'spdfg'[cls[0]]}|{'spdfg'[cls[1]]}), synthetic")


def _block(M, basis, i, j):
    off, sh = basis.offsets, basis.shells
    return M[off[i]:off[i] + sh[i].n_cart, off[j]:off[j] + sh[j].n_cart]


@pytest.mark.parametrize("cls", [_param((0, 1)), _param((1, 3))])
def test_a_swapped_pair_lands_in_basis_order(cls):
    """A pair `(i, j)`, `i < j`, with `l_i > l_j` is staged `(j, i)`; the
    kernel's block at the basis order's `[off_i][off_j]` is `(i|j)` of the
    basis-order shells, and its mirror the transpose, exactly."""
    sh = SYNTHETIC.shells
    run = metric_matrix_kernel(SYNTHETIC, boys=boys, classes={cls})
    c = next(c for c in metric_classes(SYNTHETIC) if (c.lP, c.lQ) == cls)
    k = next(k for k, (q, p) in enumerate(c.pairs) if c.swapped[k]
             and sh[p].centre != sh[q].centre)  # two centres: nonzero
    j, i = c.pairs[k]
    assert i < j and sh[i].l > sh[j].l
    want = np.array(metric_block_and_scale(sh[i], sh[j], pq=pq)[0])
    scale = _scale(sh[i], sh[j])
    got = _block(run.M, SYNTHETIC, i, j)
    assert got.shape == (sh[i].n_cart, sh[j].n_cart)
    assert np.all(np.abs(got - want) <= TOL_ULPS * EPS * scale)
    assert np.array_equal(_block(run.M, SYNTHETIC, j, i), got.T)
    assert np.any(got)


@pytest.mark.parametrize("cls", [_param((1, 1)), _param((2, 2))])
def test_a_diagonal_pair_is_its_own_symmetric_block(cls):
    """A shell against itself writes its lower triangle, mirrored: the block
    is exactly symmetric and `(P|P)` of the host to the bound."""
    sh = SYNTHETIC.shells
    run = metric_matrix_kernel(SYNTHETIC, boys=boys, classes={cls})
    c = next(c for c in metric_classes(SYNTHETIC) if (c.lP, c.lQ) == cls)
    p = next(p for p, q in c.pairs if p == q)
    got = _block(run.M, SYNTHETIC, p, p)
    want = np.array(metric_block_and_scale(sh[p], sh[p], pq=pq)[0])
    scale = _scale(sh[p], sh[p])
    assert np.array_equal(got, got.T)
    assert np.all(np.abs(got - want) <= TOL_ULPS * EPS * scale)
    assert np.all(np.diag(got) > 0)  # (P|P) > 0


def test_a_tight_pair_past_the_narrow_boys_range():
    """`T` reaches 2000 here, past where the old narrow series overflowed
    (near 709): the one `fj_boys` (#301) holds there."""
    c = next(c for c in metric_classes(TIGHT) if (c.lP, c.lQ) == (0, 1))
    T = max((c.alpha[i] * c.alpha[k] / (c.alpha[i] + c.alpha[k]))
            * sum((c.A[i][d] - c.A[k][d]) ** 2 for d in range(3))
            for i in range(c.n_rec) for k in range(c.n_rec))
    assert T > 709.0
    run = metric_matrix_kernel(TIGHT, boys=boys)
    want, scale, owned = _host(TIGHT, set(run.ran))
    assert owned.all()
    _check_host(run.M, want, scale, owned, "tight")


# ---------------------------------------------------------------------------
# The fixture: libcint's int2c2e_cart over cc-pVTZ-jkfit, L <= 4 filtered
# ---------------------------------------------------------------------------


def _check_fixture(basis, rows, run, label):
    want, scale, owned = _host(basis, set(run.ran))
    _check_host(run.M, want, scale, owned, f"{label}, against the host")
    ref = _ms().metric[np.ix_(rows, rows)]
    sm = float(np.max(np.abs(_ms().metric)))
    worst = float(np.max(np.abs(run.M - ref)[owned])) / sm
    print(f"{label}: max |M - libcint| = {worst:.2e} of max|M|")
    assert worst <= TOL, worst


def test_the_sp_sub_basis_matrix_is_libcints():
    """The aux shells with `l <= 1`, every class of them, the whole matrix."""
    aux = _aux()
    keep = [k for k, s in enumerate(aux.shells) if s.l <= 1]
    sub = Basis(tuple(aux.shells[k] for k in keep))
    rows = [aux.offsets[k] + a for k in keep
            for a in range(aux.shells[k].n_cart)]
    run = metric_matrix_kernel(sub, boys=boys)
    assert set(run.ran) == {(0, 0), (0, 1), (1, 1)}
    _check_fixture(sub, rows, run, "l <= 1")
    assert np.linalg.eigvalsh(run.M).min() > 0


@pytest.mark.slow  # d, f and g shells (the issue's rule, as test_ao_metric)
def test_every_l_le_4_class_of_the_whole_fixture_is_libcints():
    """Every class `lP + lQ <= 4` over the WHOLE aux basis (O to g, H to f):
    all nine classes, every element they own against the host and libcint,
    every other element untouched."""
    aux = _aux()
    run = metric_matrix_kernel(aux, boys=boys, classes=set(CLASSES))
    assert sorted(run.ran) == CLASSES
    _check_fixture(aux, list(range(aux.n_ao)), run, "L <= 4")


# ---------------------------------------------------------------------------
# What the kernel is built from
# ---------------------------------------------------------------------------


def _parity_targets(lP: int, lQ: int) -> set[tuple[int, int, int]]:
    """`|r| = L (mod 2)` and `|r| >= (lP mod 2) + (lQ mod 2)`: an odd-l row
    has no live `t = 0` (`kernel_metric`'s module docstring)."""
    L = lP + lQ
    return {r for r in hermite_targets(L) if (sum(r) - L) % 2 == 0
            and sum(r) >= lP % 2 + lQ % 2}


@pytest.mark.parametrize("cls", [by_l(c, sum(c), id=f"{c[0]}{c[1]}")
                                 for c in [(0, 0), (0, 1), (1, 1), (0, 3),
                                           (1, 2), (2, 2), (0, 4)]])
def test_the_recurrence_is_the_quartets_plan_minus_the_other_parity(cls):
    """Every recurrence `Decl` is the J quartet `(lP 0|lQ 0)`'s node, and the
    kept target set is exactly what the contraction reads: every `[r]^(0)`
    reachable from a live `t + tau` (`_parity_targets`), no other."""
    lP, lQ = cls
    assert quartet_spec(cls) == Spec(Operator.COULOMB, 0, (lP, 0, lQ, 0),
                                     Target.J)
    dag, _ = metric_plan(cls)
    (want_dag,) = plan(quartet_spec(cls)).dags  # BatchDag compares by id
    assert (dag.batch, dag.targets, dag.parents) == (
        want_dag.batch, want_dag.targets, want_dag.parents)
    k = build_metric_kernel(cls)
    decls = {s.name for s in walk(k.body) if isinstance(s, Decl)}
    kept = {r for r, m in dag.targets if m == 0
            and node_name((r, 0)) in decls}
    assert kept == set(live_targets(lP, lQ)) == _parity_targets(lP, lQ)
    assert decls & ({node_name(n) for n in dag.parents}
                    | {node_name(n) for n in dag.targets})


def test_both_e_blocks_are_the_center_fragment_inlined():
    k = build_metric_kernel((1, 2))
    inl = [s for s in walk(k.body, into_inline=False) if isinstance(s, Inline)]
    assert [s.fragment for s in inl] == [ef.center_fragment(1),
                                         ef.center_fragment(2)]
    assert [dict(s.bindings)[ef.E_ARRAY].name for s in inl] == [
        "Ebra", "Eket"]


def test_the_scatter_is_plain_stores_and_the_key_is_a_pair():
    k = build_metric_kernel((1, 1))
    assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    stores = [s for s in walk(k.body) if isinstance(s, Assign)
              and isinstance(s.dst, Index) and s.dst.array == OUT]
    # (p|p): 9 elements, off-diagonal 2 x 9, diagonal 6 + 3 mirrored = 9
    assert len(stores) == 18 + 9
    assert len([s for s in walk(build_metric_kernel((0, 2)).body)
                if isinstance(s, Assign) and isinstance(s.dst, Index)
                and s.dst.array == OUT]) == 2 * 6  # no diagonal branch
    assert metric_spec(1, 2).key() == "coulomb/d0/(1,2)/b1/metric/literal"
    assert metric_kernel(1, 1) is metric_kernel(1, 1)  # memoised
    with pytest.raises(ValueError, match="unbatched by decision"):
        Spec(Operator.COULOMB, 0, (1, 1), Target.METRIC, n_batches=3)
    with pytest.raises(ValueError, match="not a Target.METRIC key"):
        build_metric_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.SCHWARZ))
    with pytest.raises(ValueError):
        launch_shape(0)
    assert launch_shape(33, 32) == (2, (32, 1))


def test_the_ket_sign_is_one_sign_per_class():
    """Every live `tau` of an `lQ` row has `|tau| = lQ (mod 2)`, so the
    accumulate is `acc - w v` exactly when `lQ` is odd."""
    for cls, fn in (((0, 1), Fn.SUB), ((1, 2), Fn.ADD), ((0, 0), Fn.ADD)):
        k = build_metric_kernel(cls)
        accs = [s for s in walk(k.body) if isinstance(s, Assign)
                and not isinstance(s.dst, Index)
                and s.dst.name.startswith("acc")]
        assert accs and {s.value.fn for s in accs} == {fn}, cls


@pytest.mark.slow  # L = 5 > FAST_L_MAX (tiers.py): built, never interpreted
def test_a_class_above_the_cap_builds():
    """The emitter is arbitrary-L: (d|f) at L = 5 builds, its kept targets the
    parity set. NOT checked numerically (the milestone's L <= 4 cap)."""
    k = build_metric_kernel((2, 3))
    assert isinstance(k, Kernel) and k.name == "metric_23"
    decls = {s.name for s in walk(k.body) if isinstance(s, Decl)}
    assert {r for r in hermite_targets(5)
            if node_name((r, 0)) in decls} == _parity_targets(2, 3)
