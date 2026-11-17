"""`kernel_dense`: the dense J- and K-layout ERI (`Target.ERI_DENSE`), interpreted.

THE WITNESS is `eri.full_eri.dense` (`eri.eri.eri_block_and_scale` of every
ordered quartet, `reference.pq.pq` injected, placed by the kernel's rule),
and through the fixtures PySCF's `get_jk`: `J = M_J vec(D)` and `K = M_K
vec(D)` from the interpreted matrices are held to the fixtures' `vj` and
`vk` directly.

THE BOUND, stated before any run and the same one the card is held to
(test_eri_dense_device's `bound`): per element

    |M - host| <= (2 gamma_N + delta) scale,
    N = (4 L_p + 8) + (4 L_q + 8) + (3 L + 12) + n_p + n_q + 3 + K_b K_k,

test_eri_full_device's analysis of the same contraction (`kernel_dense` runs
`kernel_eri.contraction` unchanged; only the placement differs, and a store
rounds nothing), `scale` being `eri.eri`'s absolute-value twin, `K_b K_k`
the owning task's primitive quartets. The interpreter is IEEE double with no
FMA, one case of that analysis.

A NON-SYMMETRIC `D`, which the layouts support and the packed path does not:
`D = D_fx + A`, `A` antisymmetric. `(mn|ls) = (mn|sl)`, so `J(A) = 0`
exactly and `J(D)` must be the fixture's `vj`; `K(D)^T = K(D^T)`, so
`(K(D) + K(D^T)) / 2` must be its `vk`, while `K(D) - K(D^T) = 2 K(A)` is
held to a dense `einsum` of the host tensor. With the oracle group
installed, `test_a_nonsymmetric_density_against_pyscf_today` asks PySCF's
`get_jk(hermi=0)` for the whole `J(D)` and `K(D)` (`oracle`).

THE ROW RANGE: a partition of the shells into ranges, each launched alone,
stacks bit for bit into the one launch over every shell; every element is
written (the output starts as NaN).

SCOPE: classes with L <= FAST_L_MAX = 4 (tiers.py) on test_kernel_eri_full's
synthetic s/p/d basis, every ORDERING (the class is the ordered quartet),
and H2 and H2O in STO-3G whole. Nothing compiles, nothing launches.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np
import pytest
from eri_dense_driver import interpret, vec_product
from oracle import basis as ob
from oracle import manifest
from test_kernel_device import BOYS_REL
from test_kernel_eri_full import F_BASIS, J_TOL, L_MAX, SYNTHETIC
from tiers import split_by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.full_eri import kernel_dense, kernel_eri
from felizjueves.eri.full_eri.dense import (
    LAYOUTS,
    col_index,
    dense_classes,
    dense_eri_and_scale,
    owned,
    pair_class,
    row_index,
    row_range,
)
from felizjueves.ir import AtomicAdd, Inline, walk
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.hermite import hermite_order
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target

U = 2.0 ** -53
MOLECULES = ("h2_sto3g_k", "h2o_sto3g_k")
J_OF = {"h2_sto3g_k": "h2_sto3g_j", "h2o_sto3g_k": "h2o_sto3g_j"}
SYNTHETIC_CLASSES = sorted(c.am for c in dense_classes(SYNTHETIC)
                           if sum(c.am) <= L_MAX)
# H2O/STO-3G's five shells (O 1s, 2s, 2p, H 1s, H 1s) cut three ways
PARTITION = ((0, 1), (1, 3), (3, 5))


def _gamma(n: int) -> float:
    return n * U / (1.0 - n * U)


@cache
def _fixture(name: str):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


@cache
def _basis(name: str) -> Basis:
    if name in ("synthetic", "f"):
        return SYNTHETIC if name == "synthetic" else F_BASIS
    return ob.from_fixture(_fixture(name))


def _classes(basis: Basis, shells=None):
    return [c for c in dense_classes(basis, shells) if sum(c.am) <= L_MAX]


def bound(basis: Basis, layout: str, keep=None, shells=None) -> np.ndarray:
    """Each element's `2 gamma_N + delta` (module docstring), placed as the
    kernel places it in the local row block; 0 where nothing is kept."""
    sh = basis.shells
    n = basis.n_ao
    m0, n_row = row_range(basis, shells)
    out = np.zeros((n_row, n * n))
    for c in dense_classes(basis, shells):
        if keep is not None and c.am not in keep:
            continue
        la, lb, lc, ld = c.am
        lp, lq = la + lb, lc + ld
        L = lp + lq
        for tb, tk in zip(c.task_bra, c.task_ket, strict=True):
            kb = c.bra.rec1[tb] - c.bra.rec0[tb]
            kk = c.ket.rec1[tk] - c.ket.rec0[tk]
            N = ((4 * lp + 8) + (4 * lq + 8) + (3 * L + 12)
                 + len(hermite_order(lp)) + len(hermite_order(lq)) + 3
                 + kb * kk)
            i, j = c.bra.pairs[tb]
            k, q = c.ket.pairs[tk]
            for e in itertools.product(
                    *(range(off, off + sh[s].n_cart) for s, off in (
                        (i, c.bra.off_a[tb]), (j, c.bra.off_b[tb]),
                        (k, c.ket.off_a[tk]), (q, c.ket.off_b[tk])))):
                out[row_index(n, layout, *e) - m0 * n,
                    col_index(n, layout, *e)] = 2 * _gamma(N) + BOYS_REL
    return out


@cache
def host(name: str, layout: str, keep: frozenset | None = None,
         shells: tuple[int, int] | None = None):
    """`(M, scale, owned, tol)` of the host, as arrays."""
    basis = _basis(name)
    want, scale = (np.array(x) for x in dense_eri_and_scale(
        basis, layout, pq=pq, shells=shells, keep=keep))
    mine = np.array(owned(basis, layout, shells, keep))
    return want, scale, mine, bound(basis, layout, keep, shells) * scale


def check(M: np.ndarray, name: str, layout: str, keep=None, shells=None,
          *, zero_elsewhere: bool = False) -> float:
    """`M` against the host within the bound, every owned element written
    (and every other NaN, or 0.0 with `zero_elsewhere`); the worst ratio."""
    want, _, mine, tol = host(name, layout, keep, shells)
    assert np.all(np.isfinite(M[mine])), name
    if zero_elsewhere:
        assert np.all(M[~mine] == 0.0), name
    else:
        assert np.all(np.isnan(M[~mine])), name
    err = np.abs(M[mine] - want[mine])
    worst = float(np.max(err / np.maximum(tol[mine], 1e-300)))
    assert np.all(err <= tol[mine]), (name, layout, worst)
    return worst


def nonsymmetric(D) -> np.ndarray:
    """`D + A`, `A[i][j] = (i - j) / (i + j + 2)`: antisymmetric, no RNG."""
    D = np.asarray(D, dtype=np.float64)
    i, j = np.indices(D.shape)
    return D + (i - j) / (i + j + 2.0)


def _dense(basis: Basis) -> np.ndarray:
    """`(mn|ls)` as an `n_ao^4` tensor from the host's J layout."""
    n = basis.n_ao
    return np.array(dense_eri_and_scale(basis, "j", pq=pq)[0]).reshape(
        n, n, n, n)


@cache
def _full(name: str, layout: str, shells=None) -> np.ndarray:
    basis = _basis(name)
    m0, n_row = row_range(basis, shells)
    return interpret(dense_classes(basis, shells), basis.n_ao, layout, m0,
                     n_row, init=np.full((n_row, basis.n_ao ** 2), np.nan))


# ---------------------------------------------------------------------------
# The staging
# ---------------------------------------------------------------------------


def test_every_ordered_shell_quartet_is_one_task():
    sh = SYNTHETIC.shells
    seen = []
    for c in dense_classes(SYNTHETIC):
        for tb, tk in zip(c.task_bra, c.task_ket, strict=True):
            q = (*c.bra.pairs[tb], *c.ket.pairs[tk])
            assert tuple(sh[s].l for s in q) == c.am
            seen.append(q)
    n = len(sh)
    assert sorted(seen) == list(itertools.product(range(n), repeat=4))


def test_every_ordering_is_its_own_class():
    got = {c.am for c in dense_classes(SYNTHETIC)}
    assert got == set(itertools.product((0, 1, 2), repeat=4))
    assert (1, 0, 0, 0) in SYNTHETIC_CLASSES and (0, 0, 0, 1) in \
        SYNTHETIC_CLASSES and (2, 0, 1, 1) in SYNTHETIC_CLASSES


def test_a_partition_of_the_shells_partitions_the_tasks_and_the_rows():
    n_sh = len(SYNTHETIC.shells)
    cuts = ((0, 2), (2, 3), (3, n_sh))
    whole = {(c.am, c.bra.pairs[b], c.ket.pairs[k])
             for c in dense_classes(SYNTHETIC)
             for b, k in zip(c.task_bra, c.task_ket, strict=True)}
    parts = [{(c.am, c.bra.pairs[b], c.ket.pairs[k])
              for c in dense_classes(SYNTHETIC, cut)
              for b, k in zip(c.task_bra, c.task_ket, strict=True)}
             for cut in cuts]
    assert sum(map(len, parts)) == len(whole) and set().union(*parts) == whole
    n = SYNTHETIC.n_ao
    rows = [row_range(SYNTHETIC, cut) for cut in cuts]
    assert [m0 * n for m0, _ in rows] == [0, *itertools.accumulate(
        r for _, r in rows)][:-1]
    assert sum(r for _, r in rows) == n * n
    # the bra's pairs are those whose first shell is in the range, every
    # second shell; the ket's are every ordered pair of its class
    bra = pair_class(SYNTHETIC, 0, 1, (2, 3))
    assert bra.pairs == ((2, 0), (2, 3))
    assert pair_class(SYNTHETIC, 1, 0).pairs == ((0, 1), (0, 2), (0, 5),
                                                (3, 1), (3, 2), (3, 5))


def test_the_int32_bound_is_per_launch_and_there_is_no_n_ao_cap():
    def s_shells(n: int) -> Basis:
        return Basis(tuple(Shell(0, (float(i), 0.0, 0.0), (1.0,), (1.0,))
                           for i in range(n)))
    big = s_shells(1000)  # past the packed path's N_AO_MAX = 150
    assert row_range(big, (0, 2)) == (0, 2000)  # 2000 * 10^6 < 2^31
    assert row_range(big, (998, 1000)) == (998, 2000)
    with pytest.raises(ValueError, match="int32"):
        row_range(big, (0, 3))  # 3000 * 10^6 > 2^31
    with pytest.raises(ValueError, match="int32"):
        row_range(big)
    with pytest.raises(ValueError, match="nonempty range"):
        row_range(big, (5, 5))


def test_the_layouts_place_one_element_two_ways():
    n = 7
    assert (row_index(n, "j", 1, 2, 3, 4), col_index(n, "j", 1, 2, 3, 4)) \
        == (1 * n + 2, 3 * n + 4)
    assert (row_index(n, "k", 1, 2, 3, 4), col_index(n, "k", 1, 2, 3, 4)) \
        == (1 * n + 3, 2 * n + 4)
    with pytest.raises(ValueError, match="layout"):
        row_index(n, "x", 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# The kernel, interpreted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("cls", SYNTHETIC_CLASSES,
                         ids=lambda c: "".join(map(str, c)))
def test_one_launch_over_the_class_agrees_with_the_host(cls, layout):
    classes = [c for c in dense_classes(SYNTHETIC) if c.am == cls]
    n = SYNTHETIC.n_ao
    M = interpret(classes, n, layout, 0, n * n,
                  init=np.full((n * n, n * n), np.nan))
    worst = check(M, "synthetic", layout, frozenset({cls}))
    print(f"\n({''.join(map(str, cls))}) {layout}: worst / bound "
          f"{worst:.2e}")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name", MOLECULES)
def test_the_whole_matrix_against_the_host(name, layout):
    assert max(sum(c.am) for c in dense_classes(_basis(name))) <= L_MAX
    worst = check(_full(name, layout), name, layout)
    print(f"\n{name} {layout}: worst / bound {worst:.2e}")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_union_of_row_ranges_is_the_whole_matrix_bitwise(layout):
    name = "h2o_sto3g_k"
    basis = _basis(name)
    assert len(basis.shells) == PARTITION[-1][1]
    blocks = [_full(name, layout, cut) for cut in PARTITION]
    for cut, blk in zip(PARTITION, blocks, strict=True):
        check(blk, name, layout, shells=cut)  # each block on its own
    assert np.array_equal(np.vstack(blocks), _full(name, layout))


@pytest.mark.parametrize("name", MOLECULES)
def test_j_and_k_by_one_product_against_pyscf(name):
    fx, fj = _fixture(name), _fixture(J_OF[name])
    assert np.array_equal(fx.density, fj.density)
    for layout, ref in (("j", fj.arrays[manifest.J_ARRAY]),
                        ("k", fx.arrays[manifest.K_ARRAY])):
        got = vec_product(_full(name, layout), fx.density)
        rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
        print(f"\n{name}: max |{layout.upper()} - v{layout}| / max |v"
              f"{layout}| = {rel:.2e}")
        assert rel < J_TOL


@pytest.mark.parametrize("name", MOLECULES)
def test_a_nonsymmetric_density(name):
    fx, fj = _fixture(name), _fixture(J_OF[name])
    basis = _basis(name)
    D = nonsymmetric(fx.density)
    assert not np.allclose(D, D.T)
    vj, vk = fj.arrays[manifest.J_ARRAY], fx.arrays[manifest.K_ARRAY]
    J = vec_product(_full(name, "j"), D)
    K, KT = (vec_product(_full(name, "k"), x) for x in (D, D.T))
    rj = float(np.max(np.abs(J - vj)) / np.max(np.abs(vj)))
    rk = float(np.max(np.abs((K + KT) / 2 - vk)) / np.max(np.abs(vk)))
    eri = _dense(basis)
    KA = np.einsum("mlns,ls->mn", eri, (D - D.T) / 2)
    ra = float(np.max(np.abs((K - KT) / 2 - KA)) / np.max(np.abs(KA)))
    wj = np.einsum("mnls,ls->mn", eri, D)
    wk = np.einsum("mlns,ls->mn", eri, D)
    print(f"\n{name}, D + A: |J - vj| {rj:.2e}, |(K(D) + K(D^T))/2 - vk| "
          f"{rk:.2e}, |K(A) - einsum| {ra:.2e} (relative to the max)")
    assert rj < J_TOL and rk < J_TOL and ra < J_TOL
    assert np.max(np.abs(KA)) > 1e-3 * np.max(np.abs(vk))  # not vacuous
    assert np.max(np.abs(J - wj)) < J_TOL * np.max(np.abs(wj))
    assert np.max(np.abs(K - wk)) < J_TOL * np.max(np.abs(wk))


@pytest.mark.oracle
@pytest.mark.parametrize("name", MOLECULES)
def test_a_nonsymmetric_density_against_pyscf_today(name):
    """`get_jk(hermi=0)` on the fixture's Mole: `vk_il = SUM_jk (ij|kl)
    D_jk` is `K_mn = SUM_ls (ml|ns) D_ls` for any `D`."""
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    from oracle import generate
    from pyscf import scf

    mol = generate._mole(generate.CASES[name])
    D = nonsymmetric(_fixture(name).density)
    vj, vk = (np.asarray(x) for x in scf.hf.get_jk(mol, D, hermi=0))
    for layout, ref in (("j", vj), ("k", vk)):
        got = vec_product(_full(name, layout), D)
        rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
        print(f"\n{name}, D + A: max |{layout.upper()} - PySCF v{layout}| / "
              f"max = {rel:.2e}")
        assert rel < J_TOL


# ---------------------------------------------------------------------------
# What the kernel is built from, and its key
# ---------------------------------------------------------------------------


def test_both_layouts_run_kernel_eris_contraction_and_differ_in_the_scatter():
    cls = (1, 0, 1, 1)
    kj = kernel_dense.build_dense_kernel(cls, "j")
    kk = kernel_dense.build_dense_kernel(cls, "k")
    assert (kj.name, kk.name) == ("eri_dj_1011", "eri_dk_1011")
    head = kernel_eri.contraction(cls)
    for k in (kj, kk):
        inl = [s for s in walk(k.body, into_inline=False)
               if isinstance(s, Inline)]
        assert [s.fragment for s in inl] == [ef.block_fragment(1, 0),
                                             ef.block_fragment(1, 1)]
        assert not [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
        task = k.body[-1].then
        assert task[:len(head)] == head
        assert [p.name for p in k.params][-1] == kernel_dense.OUT
    assert kj.body != kk.body


def test_the_key_and_its_refusals():
    spec = kernel_dense.dense_spec(1, 0, 1, 1)
    assert spec.key() == "coulomb/d0/(1,0|1,1)/b1/eri_dense/literal"
    with pytest.raises(ValueError, match="unbatched by decision"):
        Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.ERI_DENSE,
             n_batches=3)
    with pytest.raises(ValueError):
        Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.ERI_DENSE,
             engine=Engine.HGP)
    with pytest.raises(ValueError, match="takes no levers"):
        kernel_dense.build_dense_kernel(Spec(
            Operator.COULOMB, 0, (1, 1, 1, 1), Target.ERI_DENSE,
            ("recycle",)), "j")
    with pytest.raises(ValueError, match="not a Target.ERI_DENSE key"):
        kernel_dense.build_dense_kernel(kernel_eri.eri_spec(0, 0, 0, 0), "j")
    with pytest.raises(ValueError, match="layout"):
        kernel_dense.build_dense_kernel((0, 0, 0, 0), "jk")
    with pytest.raises(ValueError):
        kernel_dense.launch_shape(0)


# ---------------------------------------------------------------------------
# The low-stack form (f, and forced at any class)
# ---------------------------------------------------------------------------

F_ORDERED = sorted(c.am for c in dense_classes(F_BASIS))
# The fast tier's forced low-stack sample: each kind of ordering once
FORCED_FAST = {(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (1, 0, 1, 1),
               (1, 1, 1, 1), (2, 1, 0, 1), (0, 2, 2, 0), (2, 0, 1, 1)}


def _ids(c) -> str:
    return "".join(map(str, c))


def test_the_dense_low_stack_rule_and_the_sp_text():
    assert len(F_ORDERED) == 256
    assert {c for c in F_ORDERED if kernel_eri.eri_lowstack(
        kernel_dense.dense_spec(*c))} == {c for c in F_ORDERED
                                          if max(c) >= 2}
    for c in (*SYNTHETIC_CLASSES, (3, 0, 0, 0), (0, 1, 3, 2)):
        for lay in LAYOUTS:
            k = kernel_dense.build_dense_kernel(c, lay)
            assert k.name == f"eri_d{lay}_" + _ids(c)
            assert k == kernel_dense.build_dense_kernel(
                c, lay, lowstack=max(c) >= 2)
    assert kernel_dense.launch_shape(5, 4, rows=3) == (4, (4, 1))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("cls", [
    pytest.param(c, id=_ids(c), marks=() if c in FORCED_FAST
                 else (pytest.mark.slow,))
    for c in SYNTHETIC_CLASSES])
def test_the_forced_low_stack_form_through_l4_agrees_with_the_host(
        cls, layout):
    classes = [c for c in dense_classes(SYNTHETIC) if c.am == cls]
    n = SYNTHETIC.n_ao
    M = interpret(classes, n, layout, 0, n * n,
                  init=np.full((n * n, n * n), np.nan), lowstack=True)
    worst = check(M, "synthetic", layout, frozenset({cls}))
    print(f"\n({_ids(cls)}) {layout} low-stack: worst / bound {worst:.2e}")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("cls", split_by_l(F_ORDERED, sum, _ids))
def test_every_ordered_class_through_ff_ff_low_stack_agrees_with_the_host(
        cls, layout):
    """Every ordering of the s-to-f basis in the low-stack form (the
    default with a d or f shell, forced below), both layouts, against
    `eri.eri` placed by the kernel's rule, within the analytic bound."""
    classes = [c for c in dense_classes(F_BASIS) if c.am == cls]
    n = F_BASIS.n_ao
    M = interpret(classes, n, layout, 0, n * n,
                  init=np.full((n * n, n * n), np.nan), lowstack=True)
    worst = check(M, "f", layout, frozenset({cls}))
    print(f"\n({_ids(cls)}) {layout} low-stack, s-to-f: worst / bound "
          f"{worst:.2e}")
