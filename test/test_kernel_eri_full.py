"""`full_eri`: the packed full ERI (`Target.ERI`) and `J` by GEMV, interpreted.

THE WITNESS is `eri.full_eri.packed` (`eri.eri.eri_block` with
`reference.pq.pq` injected, placed by the kernel's scatter), and through the
fixtures PySCF's `get_jk` Coulomb matrix: `J` from the interpreted GEMV over
the interpreted packed ERI is held to the fixture's `vj` directly.

ROUND-OFF, NOT BITWISE, for the ERI (`kernel_eri`'s module docstring): per
element `|ERI - host| <= TOL_ULPS * eps * scale`, `scale` being `eri.eri`'s
absolute-value twin of the same sum, `TOL_ULPS = 16` as test_kernel_schwarz's.
The GEMV is bitwise a left fold, so it is held `==` to one.

COVERAGE: every element must be written. The output starts as NaN, and an
element no launched task owns must come back NaN, every owned one finite.

SCOPE: classes with L <= FAST_L_MAX = 4 (tiers.py) on a synthetic s/p/d
basis, and H2 and H2O in STO-3G whole. Nothing compiles, nothing launches.
"""

from __future__ import annotations

import itertools
import math
from functools import cache

import numpy as np
import pytest
from eri_full_driver import interpret, interpret_gemv
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.full_eri import kernel_eri, kernel_gemv
from felizjueves.eri.full_eri.packed import (
    N_AO_MAX,
    coulomb_from_packed,
    eri_classes,
    n_packed,
    owned,
    pack_density,
    packed_eri_and_scale,
    pair_index,
    unpack,
)
from felizjueves.ir import AtomicAdd, Decl, Inline, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.recurrence import node_name
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL_ULPS = 16
L_MAX = 4  # FAST_L_MAX: correctness here is for la + lb + lc + ld <= 4
J_TOL = 1e-12  # x max |vj|

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
# The p shell BEFORE the s shell on A (so (ps) pairs are staged swapped), two
# s and two p shells (diagonal and off-diagonal pairs of each la == lb
# class), contracted shells, and one d shell.
SYNTHETIC = Basis((
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(0, A, (3.4, 0.62, 0.17), (0.15, 0.53, 0.44)),
    Shell(0, B, (1.3, 0.21), (0.6, 0.47)),
    Shell(1, B, (0.9,), (1.0,)),
    Shell(2, C, (0.7,), (1.0,)),
    Shell(0, C, (2.0,), (1.0,)),
))
SYNTHETIC_CLASSES = sorted(c.am for c in eri_classes(SYNTHETIC)
                           if sum(c.am) <= L_MAX)
MOLECULES = ("h2_sto3g_j", "h2o_sto3g_j")


@cache
def _fixture(name: str):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


def _nan(n_ao: int) -> np.ndarray:
    n = n_packed(n_ao)
    return np.full((n, n), np.nan)


def _check(eri, basis, keep, label):
    want, scale = (np.array(x) for x in packed_eri_and_scale(
        basis, pq=pq, keep=keep))
    mine = np.array(owned(basis, keep))
    assert np.all(np.isfinite(eri[mine])), label
    assert np.all(np.isnan(eri[~mine])), label
    err = np.abs(eri[mine] - want[mine])
    worst = float(np.max(err / (EPS * np.maximum(scale[mine], 1e-300))))
    print(f"\n{label}: max |ERI - host| = {err.max():.2e}, worst "
          f"{worst:.2f} eps scale")
    assert np.all(err <= TOL_ULPS * EPS * scale[mine]), label
    assert np.array_equal(np.nan_to_num(eri), np.nan_to_num(eri).T)


# ---------------------------------------------------------------------------
# The staging
# ---------------------------------------------------------------------------


def test_every_unordered_pair_of_shell_pairs_is_one_task():
    sh = SYNTHETIC.shells
    seen = []
    for c in eri_classes(SYNTHETIC):
        la, lb, lc, ld = c.am
        assert la <= lb and lc <= ld and (la, lb) <= (lc, ld)
        for tb, tk in zip(c.task_bra, c.task_ket, strict=True):
            p, q = c.bra.pairs[tb], c.ket.pairs[tk]
            assert (sh[p[0]].l, sh[p[1]].l, sh[q[0]].l, sh[q[1]].l) == c.am
            seen.append(frozenset((frozenset(p), frozenset(q))))
    n = len(sh)
    pairs = [frozenset((i, j)) for i in range(n) for j in range(i, n)]
    want = {frozenset((p, q)) for p, q in
            itertools.combinations_with_replacement(pairs, 2)}
    assert len(seen) == len(set(seen)) == len(want)
    assert set(seen) == want


def test_the_synthetic_basis_reaches_swapped_and_diagonal_pairs():
    # every quartet of pair classes with L <= 4 the basis can form
    pcs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    assert SYNTHETIC_CLASSES == sorted(
        (*x, *y) for i, x in enumerate(pcs) for y in pcs[i:]
        if sum(x) + sum(y) <= L_MAX)
    by = {(c.bra.la, c.bra.lb): c.bra for c in eri_classes(SYNTHETIC)}
    assert any(a > b for a, b in by[(0, 1)].pairs)  # staged swapped
    for cls in ((0, 0), (1, 1)):
        assert any(a == b for a, b in by[cls].pairs)
        assert any(a != b for a, b in by[cls].pairs)


def test_the_packing_and_the_cap():
    pidx = pair_index(4)
    assert sorted({pidx[m][n] for m in range(4) for n in range(4)}) == list(
        range(n_packed(4)))
    assert all(pidx[m][n] == pidx[n][m] for m in range(4) for n in range(4))
    D = [[1.0, 2.0], [3.0, 4.0]]
    assert pack_density(D) == [1.0, 5.0, 4.0]
    assert unpack([1.0, 2.0, 3.0], 2) == [[1.0, 2.0], [2.0, 3.0]]
    big = Basis(tuple(Shell(0, (float(i), 0.0, 0.0), (1.0,), (1.0,))
                      for i in range(N_AO_MAX + 1)))
    with pytest.raises(ValueError, match="N_AO_MAX"):
        eri_classes(big)


# ---------------------------------------------------------------------------
# The kernel, interpreted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", SYNTHETIC_CLASSES,
                         ids=lambda c: "".join(map(str, c)))
def test_one_launch_over_the_class_agrees_with_the_host(cls):
    classes = [c for c in eri_classes(SYNTHETIC) if c.am == cls]
    eri = interpret(classes, SYNTHETIC.n_ao, init=_nan(SYNTHETIC.n_ao))
    _check(eri, SYNTHETIC, {cls}, f"({''.join(map(str, cls))}), synthetic")


@pytest.mark.parametrize("name", MOLECULES)
def test_the_whole_matrix_and_j_by_gemv_against_pyscf(name):
    fx = _fixture(name)
    basis = ob.from_fixture(fx)
    classes = eri_classes(basis)
    assert max(sum(c.am) for c in classes) <= L_MAX  # s and p: all of it
    eri = interpret(classes, basis.n_ao, init=_nan(basis.n_ao))
    _check(eri, basis, None, name)
    J = interpret_gemv(eri, fx.density)
    host = np.array(coulomb_from_packed(eri.tolist(), fx.density))
    assert np.array_equal(J, host)  # the GEMV is the host's left fold
    ref = fx.arrays[manifest.J_ARRAY]
    rel = float(np.max(np.abs(J - ref)) / np.max(np.abs(ref)))
    print(f"{name}: max |J_gemv - vj| / max |vj| = {rel:.2e}")
    assert rel < J_TOL
    assert np.array_equal(J, J.T)


def test_the_gemv_is_a_left_fold_by_column():
    rng = np.random.default_rng(7)
    n = 11
    M = rng.standard_normal((n, n))
    M = M + M.T
    x = rng.standard_normal(n)
    grid, block = kernel_gemv.launch_shape(n, 4)
    got = Interpreter(kernel_gemv.build_gemv_kernel(),
                      launch=Launch(grid, block, 1)).run(
        n_pp=n, eri=M, Dp=x, Jp=np.zeros(n))[kernel_gemv.OUT]
    want = []
    for i in range(n):
        acc = 0.0
        for k in range(n):
            acc = acc + float(M[k, i]) * float(x[k])
        want.append(acc)
    assert got.tolist() == want


# ---------------------------------------------------------------------------
# What the kernel is built from, and its key
# ---------------------------------------------------------------------------


def test_the_recurrence_is_the_j_quartets_and_both_e_blocks_are_inlined():
    cls = (0, 1, 1, 1)
    dag, _ = kernel_eri.eri_plan(cls)
    assert kernel_eri.quartet_spec(cls) == Spec(Operator.COULOMB, 0, cls,
                                                Target.J)
    k = kernel_eri.build_eri_kernel(cls)
    decls = {s.name for s in walk(k.body) if isinstance(s, Decl)}
    nodes = {node_name(n) for n in dag.parents} | {
        node_name(n) for n in dag.targets}
    assert decls & nodes
    inl = [s for s in walk(k.body, into_inline=False) if isinstance(s, Inline)]
    assert [s.fragment for s in inl] == [ef.block_fragment(0, 1),
                                         ef.block_fragment(1, 1)]
    assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    assert k.name == "eri_0111"


def test_the_key_and_its_refusals():
    spec = kernel_eri.eri_spec(0, 1, 1, 1)
    assert spec.key() == "coulomb/d0/(0,1|1,1)/b1/eri/literal"
    assert math.isclose(spec.total_am, 3)
    with pytest.raises(ValueError, match="unbatched by decision"):
        Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.ERI, n_batches=3)
    with pytest.raises(ValueError):
        Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.ERI,
             engine=Engine.HGP)
    with pytest.raises(ValueError, match="takes no levers"):
        kernel_eri.build_eri_kernel(Spec(Operator.COULOMB, 0, (1, 1, 1, 1),
                                         Target.ERI, ("recycle",)))
    with pytest.raises(ValueError, match="not a Target.ERI key"):
        kernel_eri.build_eri_kernel(Spec(Operator.COULOMB, 0, (1, 1, 1, 1),
                                         Target.J))
    with pytest.raises(ValueError):
        kernel_eri.launch_shape(0)
    with pytest.raises(ValueError):
        kernel_gemv.launch_shape(0)
