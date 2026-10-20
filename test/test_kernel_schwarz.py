"""`kernel_schwarz`: the Schwarz kernel, `Target.SCHWARZ`, interpreted.

THE WITNESS is `ao.schwarz` (the component diagonal of `ao.eri.eri_block` with
`reference.pq.pq` injected), placed by `kernel_jab`'s scatter rule; the
kernel's records are `ao.schwarz.schwarz_classes`', and its Boys function is
`reference.boys.boys`, injected into the interpreter. Through the fixtures the
host is libcint's (test/test_ao_schwarz.py), so the whole-matrix cases here
also compare the interpreted kernel against libcint directly.

ROUND-OFF, NOT BITWISE (`kernel_schwarz`'s module docstring): the host's
`[p|q]` is `reference.pq`'s recursion, the kernel's the plan's MILD DAG, and
the host dresses each `E` entry where the kernel multiplies `s_i s_k` once.
The bound is per element on the SQUARE, `|Q^2 - (mn|mn)| <= TOL_ULPS * eps *
scale`, `scale` being `ao.eri`'s absolute-value twin of the same sum -- the
running-error scale `test_kernel_v.py` argues for, `TOL_ULPS = 16` stated
before measuring. Squaring a rounded root adds two roundings of `Q^2 <=
scale`, inside that.

SCOPE: correctness is tested through L = 2 (la + lb) <= 4 -- the classes
(ss), (sp), (pp) and (sd), and their swapped twins through the staging. The
ladder builds above that (`--schwarz-l` up to (gg) at L = 16); the build-only
checks here touch no class above L = 8. Nothing compiles, nothing launches.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import split_by_l

from felizjueves import e_fragment as ef
from felizjueves.ao.schwarz import (
    _place,
    diagonal_and_scale,
    diagonal_matrix,
    scale_matrix,
    schwarz_classes,
)
from felizjueves.ao.shell import Basis, Shell
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import Assign, AtomicAdd, Decl, Index, Inline, walk
from felizjueves.kernel_schwarz import (
    OUT,
    build_schwarz_kernel,
    launch_shape,
    quartet_spec,
    schwarz_plan,
    schwarz_spec,
)
from felizjueves.plan.pipeline import plan
from felizjueves.recurrence import node_name
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL_ULPS = 16  # x eps scale, on the square: see the module docstring
L_MAX = 4  # correctness is tested for 2 (la + lb) <= L_MAX
CLASSES = [(la, lb) for la in range(3) for lb in range(la, 3)
           if 2 * (la + lb) <= L_MAX]

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
# s, p and d shells on three centres, the d and p shells BEFORE the s shells
# on A (so (ps) and (ds) pairs are staged swapped), two shells of each l
# against itself (a diagonal and an off-diagonal pair per la == lb class), and
# one tight pair two bohr apart, whose T = theta^2 |P_i - P_k|^2 passes the
# narrow Boys series' range.
SYNTHETIC = Basis((
    Shell(2, A, (1.1, 0.3), (0.5, 0.6)),
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(0, A, (3.4, 0.62, 0.17), (0.15, 0.53, 0.44)),
    Shell(0, B, (1.3, 0.21), (0.6, 0.47)),
    Shell(1, B, (0.9,), (1.0,)),
    Shell(2, C, (0.7,), (1.0,)),
    Shell(0, C, (2.0,), (1.0,)),
    Shell(1, C, (0.55, 0.12), (0.8, 0.3)),
))
TIGHT = Basis((
    Shell(0, A, (1000.0, 1.0), (0.4, 0.6)),
    Shell(1, (0.0, 0.0, 2.0), (1000.0, 0.8), (0.5, 0.5)),
))


@cache
def _fixture(name: str):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


def _arrays(c):
    return {k: np.array(v, dtype=np.int32 if k in c.PAIR_ARRAYS else float)
            for k, v in c.arrays().items()}


def _launch(c, n_ao, Q, threads=3):
    """One launch of class `c` over every staged pair, into `Q`."""
    grid, block = launch_shape(c.n_pair, threads)
    return Interpreter(build_schwarz_kernel((c.la, c.lb)),
                       launch=Launch(grid, block, 1), boys=boys).run(
        n_pair=c.n_pair, n_rec=c.n_rec, n_ao=n_ao, **{OUT: Q},
        **_arrays(c))[OUT]


def _build(basis, classes=None):
    """`Q` from the interpreted kernels of `classes` (every class at
    L <= L_MAX when None), and the classes launched."""
    n = basis.n_ao
    Q = np.zeros((n, n))
    done = []
    for c in schwarz_classes(basis):
        if (classes is None and 2 * (c.la + c.lb) <= L_MAX) or (
                classes is not None and (c.la, c.lb) in classes):
            Q = _launch(c, n, Q)
            done.append(c)
    return Q, done


def _owned(basis, done):
    sh, off = basis.shells, basis.offsets
    m = np.zeros((basis.n_ao, basis.n_ao), bool)
    for c in done:
        for a, b in c.pairs:
            m[off[a]:off[a] + sh[a].n_cart, off[b]:off[b] + sh[b].n_cart] = True
    return m | m.T


def _host_on(basis, done):
    """`diagonal_matrix` and `scale_matrix` on the launched classes' shell
    pairs only, 0.0 elsewhere: `_check` reads only those elements, and the
    whole of SYNTHETIC reaches (dd|dd) at L = 8 > FAST_L_MAX (tiers.py)."""
    sh = basis.shells
    keep = {frozenset((id(sh[a]), id(sh[b]))) for c in done for a, b in c.pairs}

    def part(k):
        def value(sa, sb):
            if frozenset((id(sa), id(sb))) not in keep:
                return [0.0] * (sa.n_cart * sb.n_cart)
            return diagonal_and_scale(sa, sb, pq=pq)[k]
        return np.array(_place(basis, value))

    return part(0), part(1)


def _check(Q, want, scale, owned, label):
    err = np.abs(Q[owned] ** 2 - want[owned])
    worst = float(np.max(err / (EPS * scale[owned])))
    print(f"\n{label}: max |Q^2 - (mn|mn)| = {err.max():.2e}, worst "
          f"{worst:.2f} eps scale")
    assert np.all(err <= TOL_ULPS * EPS * scale[owned]), label
    assert np.all(Q[~owned] == 0.0)  # exactly the launched pairs' elements
    assert np.array_equal(Q, Q.T)  # the scatter mirrors one value


def test_the_synthetic_basis_reaches_every_class_swapped_and_diagonal():
    by = {(c.la, c.lb): c for c in schwarz_classes(SYNTHETIC)}
    for cls in CLASSES:
        c = by[cls]
        if cls[0] == cls[1]:
            assert any(a == b for a, b in c.pairs), cls
            assert any(a != b for a, b in c.pairs), cls
        else:
            assert any(a > b for a, b in c.pairs), cls  # staged swapped


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_one_launch_over_the_class_agrees_with_the_host(cls):
    Q, done = _build(SYNTHETIC, {cls})
    assert [(c.la, c.lb) for c in done] == [cls]
    want, scale = _host_on(SYNTHETIC, done)
    _check(Q, want, scale, _owned(SYNTHETIC, done),
           f"({cls[0]}{cls[1]}), synthetic")


def test_a_tight_pair_past_the_narrow_boys_range():
    """`T` reaches ~2000 here, past where the old narrow series overflowed
    (near 709): the one `fj_boys` (#301) holds there."""
    c = next(c for c in schwarz_classes(TIGHT) if (c.la, c.lb) == (0, 1))
    T = max((c.zeta_ab[i] * c.zeta_ab[k] / (c.zeta_ab[i] + c.zeta_ab[k]))
            * sum((c.P[i][d] - c.P[k][d]) ** 2 for d in range(3))
            for i in range(c.n_rec) for k in range(c.n_rec))
    assert T > 709.0
    Q, done = _build(TIGHT)
    _check(Q, np.array(diagonal_matrix(TIGHT, pq=pq)),
           np.array(scale_matrix(TIGHT, pq=pq)), _owned(TIGHT, done), "tight")


@pytest.mark.parametrize("name", ("h2_sto3g_schwarz", "h2o_sto3g_schwarz",
                                  "h2o_631g_schwarz"))
def test_the_whole_matrix_against_the_host_and_libcint(name):
    fx = _fixture(name)
    basis = ob.from_fixture(fx)
    Q, done = _build(basis)
    assert {(c.la, c.lb) for c in done} == {
        (c.la, c.lb) for c in schwarz_classes(basis)}  # s and p: all of it
    owned = np.ones((basis.n_ao, basis.n_ao), bool)
    _check(Q, np.array(diagonal_matrix(basis, pq=pq)),
           np.array(scale_matrix(basis, pq=pq)), owned, name)
    ref = fx.arrays[manifest.SCHWARZ_ARRAY]
    rel = float(np.max(np.abs(Q - np.sqrt(ref)) / np.sqrt(ref)))
    print(f"{name}: max |Q - sqrt(libcint)| / sqrt(libcint) = {rel:.2e}")
    assert rel < 1e-12


# ---------------------------------------------------------------------------
# What the kernel is built from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", split_by_l(
    [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)],
    lambda c: 2 * sum(c), ids=lambda c: f"{c[0]}{c[1]}"))
def test_the_recurrence_is_the_quartets_plan_minus_dead_targets(cls):
    """Every recurrence `Decl` is the J quartet `(la lb|la lb)`'s node, and
    the kept target set is exactly what the diagonal rows read: `[r]` with
    `r_d <= 2 (a_d + b_d)` on every axis for some row `(a, b)`."""
    la, lb = cls
    assert quartet_spec(cls) == Spec(Operator.COULOMB, 0, (la, lb, la, lb),
                                     Target.J)
    dag, _ = schwarz_plan(cls)
    (want_dag,) = plan(quartet_spec(cls)).dags  # BatchDag compares by id
    assert (dag.batch, dag.targets, dag.parents) == (
        want_dag.batch, want_dag.targets, want_dag.parents)
    k = build_schwarz_kernel(cls)
    decls = {s.name for s in walk(k.body) if isinstance(s, Decl)}
    nodes = {node_name(n) for n in dag.parents} | {
        node_name(n) for n in dag.targets}
    rows = ef.block_rows(la, lb)
    want = {r for r, m in dag.targets if m == 0
            and any(all(r[d] <= 2 * (a[d] + b[d]) for d in range(3))
                    for a, b in rows)}
    kept = {r for r, m in dag.targets if m == 0
            and node_name((r, 0)) in decls}
    assert kept == want
    if cls == (0, 1):  # (sx|sy) would read [1,1,0]; no diagonal row does
        assert node_name(((1, 1, 0), 0)) not in decls
    assert decls & nodes  # the tower is there


@pytest.mark.slow  # (12|12) at L = 6 > FAST_L_MAX (tiers.py)
def test_both_e_blocks_are_the_block_fragment_inlined():
    k = build_schwarz_kernel((1, 2))
    inl = [s for s in walk(k.body, into_inline=False) if isinstance(s, Inline)]
    assert [s.fragment for s in inl] == [ef.block_fragment(1, 2)] * 2
    assert [dict(s.bindings)[ef.E_ARRAY].name for s in inl] == [
        "Ebra", "Eket"]


def test_the_scatter_is_plain_stores_and_the_key_is_a_pair():
    k = build_schwarz_kernel((1, 1))
    assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    stores = [s for s in walk(k.body) if isinstance(s, Assign)
              and isinstance(s.dst, Index) and s.dst.array == OUT]
    # (pp): 9 rows, off-diagonal 2 x 9, diagonal 6 + 3 mirrored = 9
    assert len(stores) == 18 + 9
    spec = schwarz_spec(2, 1)
    assert spec.key() == "coulomb/d0/(2,1)/b1/schwarz/literal"
    with pytest.raises(ValueError, match="unbatched by decision"):
        Spec(Operator.COULOMB, 0, (1, 1), Target.SCHWARZ, n_batches=3)
    with pytest.raises(ValueError, match="not a Target.SCHWARZ key"):
        build_schwarz_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.DQ))
    with pytest.raises(ValueError):
        launch_shape(0)
