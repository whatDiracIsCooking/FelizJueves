"""`full_eri`: the packed full ERI (`Target.ERI`), `J` and `K` by GEMV, interpreted.

THE WITNESS is `eri.full_eri.packed` (`eri.eri.eri_block` with
`reference.pq.pq` injected, placed by the kernel's scatter), and through the
fixtures PySCF's `get_jk` Coulomb matrix: `J` from the interpreted GEMV over
the interpreted packed ERI is held to the fixture's `vj` directly.

ROUND-OFF, NOT BITWISE, for the ERI (`kernel_eri`'s module docstring): per
element `|ERI - host| <= TOL_ULPS * eps * scale`, `scale` being `eri.eri`'s
absolute-value twin of the same sum, `TOL_ULPS = 16` as test_kernel_schwarz's.
The GEMV is bitwise a left fold, so it is held `==` to one.

`K` BY GEMV: `kernel_kperm` gathers the exchange-ordered `X_pp` from
`ERI_pp`, held `==` to the host's `exchange_packed` (the same two loads, one
add, one exact halving) and bitwise symmetric; `K` is then the same GEMV
kernel over `X_pp`, held to PySCF's `vk` at `J_TOL`, and the host's packed
`K` to a dense `einsum` over the unpacked tensor for a random symmetric `D`.

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
from eri_full_driver import interpret, interpret_gemv, interpret_kperm
from oracle import basis as ob
from oracle import manifest
from tiers import split_by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.full_eri import (
    kernel_dense,
    kernel_eri,
    kernel_gemv,
    kernel_kperm,
)
from felizjueves.eri.full_eri.packed import (
    N_AO_MAX,
    coulomb_from_packed,
    eri_classes,
    exchange_from_packed,
    exchange_packed,
    n_packed,
    owned,
    pack_density,
    packed_eri_and_scale,
    pair_index,
    pair_of,
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
K_MOLECULES = ("h2_sto3g_k", "h2o_sto3g_k")
# SYNTHETIC without its d shell: every class at L <= 4, the whole tensor
SP = Basis(tuple(s for s in SYNTHETIC.shells if s.l < 2))


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
# K by GEMV: the exchange-ordered matrix, and K over it
# ---------------------------------------------------------------------------


def _dense(eri: np.ndarray, n_ao: int) -> np.ndarray:
    """`(mn|ls)` as an `n_ao^4` tensor from `ERI_pp`."""
    pidx = np.array(pair_index(n_ao))
    return eri[pidx[:, :, None, None], pidx[None, None, :, :]]


def test_pair_of_inverts_the_packing():
    pidx = pair_index(5)
    pm, pn = pair_of(5)
    assert [pidx[m][n] for m, n in zip(pm, pn, strict=True)] == list(
        range(n_packed(5)))
    assert all(m >= n for m, n in zip(pm, pn, strict=True))


def test_k_by_the_packed_gemv_is_the_dense_exchange():
    eri = np.array(packed_eri_and_scale(SP, pq=pq)[0])
    rng = np.random.default_rng(11)
    D = rng.standard_normal((SP.n_ao, SP.n_ao))
    D = D + D.T
    X = np.array(exchange_packed(eri.tolist()))
    assert np.array_equal(X, X.T)
    K = np.array(exchange_from_packed(X.tolist(), D.tolist()))
    want = np.einsum("mlns,ls->mn", _dense(eri, SP.n_ao), D)
    err = float(np.max(np.abs(K - want)) / np.max(np.abs(want)))
    print(f"\nSP basis: max |K_gemv - K_dense| / max |K| = {err:.2e}")
    assert err < 1e-14
    assert np.array_equal(K, K.T)
    # and J's packed GEMV over the same tensor is the dense Coulomb sum
    J = np.array(coulomb_from_packed(eri.tolist(), D.tolist()))
    wj = np.einsum("mnls,ls->mn", _dense(eri, SP.n_ao), D)
    assert np.max(np.abs(J - wj)) < 1e-14 * np.max(np.abs(wj))


def test_the_permute_is_the_host_bitwise_and_exactly_symmetric():
    eri = np.array(packed_eri_and_scale(SP, pq=pq)[0])
    X = interpret_kperm(eri)
    assert np.array_equal(X, np.array(exchange_packed(eri.tolist())))
    assert np.array_equal(X, X.T)


@pytest.mark.parametrize("name", K_MOLECULES)
def test_k_by_gemv_against_pyscf(name):
    fx = _fixture(name)
    basis = ob.from_fixture(fx)
    classes = eri_classes(basis)
    assert max(sum(c.am) for c in classes) <= L_MAX
    eri = interpret(classes, basis.n_ao, init=_nan(basis.n_ao))
    _check(eri, basis, None, name)
    X = interpret_kperm(eri)
    assert np.array_equal(X, np.array(exchange_packed(eri.tolist())))
    K = interpret_gemv(X, fx.density)
    host = np.array(exchange_from_packed(X.tolist(), fx.density))
    assert np.array_equal(K, host)
    ref = fx.arrays[manifest.K_ARRAY]
    rel = float(np.max(np.abs(K - ref)) / np.max(np.abs(ref)))
    print(f"{name}: max |K_gemv - vk| / max |vk| = {rel:.2e}")
    assert rel < J_TOL
    assert np.array_equal(K, K.T)


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
    with pytest.raises(ValueError):
        kernel_kperm.launch_shape(0)
    with pytest.raises(ValueError, match="n_ao"):
        exchange_packed([[0.0] * 2] * 2)
    assert kernel_kperm.build_kperm_kernel().name == "eri_kperm"


# ---------------------------------------------------------------------------
# The low-stack form (f, and forced at any class)
# ---------------------------------------------------------------------------

U = 2.0 ** -53
BOYS_REL = 1e-13  # test_kernel_device.BOYS_REL, the card's Boys tolerance
D_ = (0.9, 1.3, -0.6)
# One shell of each l, s to f, the s and p contracted: every one of the 55
# classes through (ff|ff), one pair per pair class (so (ll) pairs are
# diagonal, and an (ll|ll) task pairs a shell pair with itself).
F_BASIS = Basis((
    Shell(3, A, (0.8,), (1.0,)),
    Shell(2, B, (0.9,), (1.0,)),
    Shell(1, C, (1.4, 0.35), (0.45, 0.66)),
    Shell(0, D_, (2.6, 0.5), (0.3, 0.8)),
))
F_CLASSES = sorted(c.am for c in eri_classes(F_BASIS))
# Two f shells: off-diagonal (ff) pairs, and (ff|ff) tasks that pair an
# off-diagonal shell pair with itself.
FF_BASIS = Basis((Shell(3, A, (0.8,), (1.0,)),
                  Shell(3, C, (1.1,), (1.0,))))


def _gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


def gamma_n(c, tb: int, tk: int) -> int:
    """test_eri_full_device's `N` for one task: every term's roundings."""
    from felizjueves.mmd.hermite import hermite_order

    la, lb, lc, ld = c.am
    lp, lq = la + lb, lc + ld
    kb = c.bra.rec1[tb] - c.bra.rec0[tb]
    kk = c.ket.rec1[tk] - c.ket.rec0[tk]
    return ((4 * lp + 8) + (4 * lq + 8) + (3 * (lp + lq) + 12)
            + len(hermite_order(lp)) + len(hermite_order(lq)) + 3 + kb * kk)


def packed_bound(basis: Basis, keep=None) -> np.ndarray:
    """Each element's `2 gamma_N + delta` (test_eri_full_device's analysis,
    which the low-stack form keeps: the same E entries, one [p+q]^(0) at
    order L, the two folds over n_p and n_q terms -- now nested over the
    row's reach and the axes -- the products by E and w, and the record
    fold), placed as the kernel places it."""
    sh, off = basis.shells, basis.offsets
    n = basis.n_ao
    pidx = np.array(pair_index(n))
    tol = np.zeros((n_packed(n),) * 2)
    for c in eri_classes(basis):
        if keep is not None and c.am not in keep:
            continue
        for tb, tk in zip(c.task_bra, c.task_ket, strict=True):
            t = 2 * _gamma(gamma_n(c, tb, tk)) + BOYS_REL
            a_, b_ = c.bra.pairs[tb]
            c_, d_ = c.ket.pairs[tk]
            rows = pidx[off[a_]:off[a_] + sh[a_].n_cart,
                        off[b_]:off[b_] + sh[b_].n_cart].ravel()
            cols = pidx[off[c_]:off[c_] + sh[c_].n_cart,
                        off[d_]:off[d_] + sh[d_].n_cart].ravel()
            tol[np.ix_(rows, cols)] = t
            tol[np.ix_(cols, rows)] = t
    return tol


@cache
def packed_host(name: str, keep: frozenset | None = None):
    """`(ERI_pp, scale, owned, tol)` of the host over a named basis."""
    basis = {"f": F_BASIS, "ff": FF_BASIS, "synthetic": SYNTHETIC}[name]
    want, scale = (np.array(x) for x in packed_eri_and_scale(
        basis, pq=pq, keep=keep))
    return (want, scale, np.array(owned(basis, keep)),
            packed_bound(basis, keep) * scale)


def check_bound(eri: np.ndarray, name: str, keep: frozenset,
                *, zero_elsewhere: bool = False) -> float:
    """`eri` against the host within the analytic bound, every owned element
    written (NaN, or 0.0, elsewhere), exactly symmetric; the worst ratio."""
    want, _, mine, tol = packed_host(name, keep)
    assert np.all(np.isfinite(eri[mine])), name
    if zero_elsewhere:
        assert np.all(eri[~mine] == 0.0), name
    else:
        assert np.all(np.isnan(eri[~mine])), name
    err = np.abs(eri[mine] - want[mine])
    worst = float(np.max(err / np.maximum(tol[mine], 1e-300)))
    assert np.all(err <= tol[mine]), (name, worst)
    assert np.array_equal(np.nan_to_num(eri), np.nan_to_num(eri).T)
    return worst


def _ids(c) -> str:
    return "".join(map(str, c))


def test_the_low_stack_rule_and_the_sp_text():
    assert kernel_eri.LOWSTACK_MIN_L == 2
    for c in F_CLASSES:
        assert kernel_eri.eri_lowstack(c) is (max(c) >= 2)
        assert kernel_eri.eri_lowstack(kernel_eri.eri_spec(*c)) is \
            kernel_eri.eri_lowstack(c)
        assert kernel_eri.task_threads(c, False) == 1
        assert kernel_eri.task_threads(c) == (
            kernel_eri.task_threads(c, True) if max(c) >= 2 else 1)
    # the default is the low-stack text from d, the one-thread-per-task
    # text through (pp|pp)
    for c in SYNTHETIC_CLASSES + [(2, 2, 2, 2)]:
        assert kernel_eri.build_eri_kernel(c) == kernel_eri.build_eri_kernel(
            c, lowstack=max(c) >= 2)
    with pytest.raises(ValueError, match="rows"):
        kernel_eri.launch_shape(4, 32, rows=0)
    with pytest.raises(ValueError, match="int32"):
        kernel_eri.launch_shape(2 ** 30, 32, rows=10)
    assert kernel_eri.launch_shape(5, 4, rows=3) == (4, (4, 1))


def _local_doubles(k) -> int:
    return sum(math.prod(x.value for x in s.shape) for s in walk(k.body)
               if isinstance(s, Decl) and s.shape)


def test_the_ff_ff_low_stack_frame_is_its_arrays():
    """No accumulator block: the frame is R (455), G (84), the row's E
    (27), six per-axis tables (6 x 112) and the Boys F_m (13), the same at
    (ff|ff) for the packed and the dense kernels."""
    k = kernel_eri.build_eri_kernel((3, 3, 3, 3))
    assert _local_doubles(k) == 455 + 84 + 27 + 6 * 112 + 13
    for lay in ("j", "k"):
        assert _local_doubles(kernel_dense.build_dense_kernel(
            (3, 3, 3, 3), lay)) == _local_doubles(k)
    assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    assert kernel_eri.task_threads((3, 3, 3, 3)) == 100
    assert kernel_dense.task_threads((3, 3, 3, 3)) == 10
    assert kernel_dense.task_threads((3, 2, 0, 0)) == 10
    assert kernel_dense.task_threads((2, 2, 2, 2)) == 6
    assert kernel_dense.task_threads((1, 1, 1, 1)) == 1


@pytest.mark.parametrize("cls", SYNTHETIC_CLASSES, ids=_ids)
def test_the_forced_low_stack_form_through_l4_agrees_with_the_host(cls):
    classes = [c for c in eri_classes(SYNTHETIC) if c.am == cls]
    eri = interpret(classes, SYNTHETIC.n_ao, init=_nan(SYNTHETIC.n_ao),
                    lowstack=True)
    worst = check_bound(eri, "synthetic", frozenset({cls}))
    print(f"\n({_ids(cls)}) low-stack, synthetic: worst / bound "
          f"{worst:.2e}")


def _by_l(classes):
    return split_by_l(classes, sum, _ids)


@pytest.mark.parametrize("cls", _by_l(F_CLASSES))
def test_every_class_through_ff_ff_low_stack_agrees_with_the_host(cls):
    """Every class of the s-to-f basis in the low-stack form (the default
    for each with a d or f shell, forced below), one launch, against
    `eri.eri` within the analytic bound."""
    classes = [c for c in eri_classes(F_BASIS) if c.am == cls]
    eri = interpret(classes, F_BASIS.n_ao, init=_nan(F_BASIS.n_ao),
                    lowstack=True)
    worst = check_bound(eri, "f", frozenset({cls}))
    print(f"\n({_ids(cls)}) low-stack, s-to-f: worst / bound {worst:.2e}")


@pytest.mark.slow
def test_ff_ff_over_two_f_shells_agrees_with_the_host():
    """Off-diagonal (ff) pairs, and tasks pairing one with itself: the
    larger packed index owns each element and its mirror."""
    (c,) = eri_classes(FF_BASIS)
    assert c.am == (3, 3, 3, 3) and c.n_task == 6
    eri = interpret([c], FF_BASIS.n_ao, init=_nan(FF_BASIS.n_ao))
    worst = check_bound(eri, "ff", frozenset({c.am}))
    print(f"\n(ff|ff), two f shells: worst / bound {worst:.2e}")
