"""`kernel_dq`: the ket pre-contraction kernel, `D_q` and `Dmax` (#153).

The kernel is checked against the HOST pre-contraction it replaces --
`ao.j.dq_from_host` (the `dq=` seam's reference) and `ao.j._kets` (the sweep's
own) -- interpreted, on real primitive-pair records. The reference is the host
path, so the f class needs no `J` fixture: `hf_ccpvtz` (s-f on F, s-d on H,
the F s shells a general contraction of 8 primitives) reaches every ordered
ket pair class (ss) through (ff), with its tracked density.

EXACT EQUALITY, NOT A TOLERANCE. The kernel performs the host's operations in
the host's order (`acc_q += x * (s * E[r, q])`, `r` outer, from `+0.0`; then
`w * acc_q`; `kernel_dq`'s module docstring says why the absent structural
zeros and the host's `D_cd = 0` skip change no bit), and the inlined `E`
fragment is bitwise `ao.hermite.e_block` (test/test_e_fragment.py). So
`D_q` and `Dmax` are asserted `==`. It does NOT factor `s * w` out of the sum.

THE WEIGHT. `w = 2` off the diagonal is a host scalar; the kernel holds no
literal that could be a weight, and its output is linear in the `w` it is
given -- covered on `hf_ccpvtz` and on H2O/STO-3G, both of which have
off-diagonal ket pairs.

THE CHAIN. `ao.j.coulomb_matrix(..., dq=KernelDQ())` against the host path
(`==`, which the exactness above implies) and against the tracked libcint
`get_jk` fixtures at test_ao_j.py's derived `1e-12 scale_J`, inherited and not
loosened: H2 and H2O/STO-3G fast, H2O/cc-pVDZ (d) `slow`. No PySCF needed.

Fast through (pp); the classes with a d or f shell are `slow` (the issue's
split). Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
import test_emit_batched
import test_emit_cuda
import test_emit_hip
import test_measured
from oracle import basis as ob
from oracle import manifest
from tiers import split_by_l

from felizjueves import e_fragment as ef
from felizjueves import kernel_dq
from felizjueves.ao.drive import KernelDQ
from felizjueves.ao.hermite import hermite_order
from felizjueves.ao.j import (
    _carts,
    _kets,
    _no_pq,
    coulomb_matrix,
    coulomb_matrix_and_scale,
    dq_from_host,
    epair,
)
from felizjueves.interp import Launch, interpret
from felizjueves.ir import (
    AtomicAdd,
    Inline,
    Lit,
    Raw,
    WarpReduce,
    _exprs_of,
    _subexprs,
    walk,
)
from felizjueves.kernel_dq import build_dq_kernel, dq_shape, dq_spec
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

LMAX = 3
CLASSES = [(lc, ld) for lc in range(LMAX + 1) for ld in range(LMAX + 1)]
CLASS_PARAMS = [pytest.param(c, id=f"{c[0]}{c[1]}",
                             marks=() if max(c) <= 1 else pytest.mark.slow)
                for c in CLASSES]
CLASS_GRID = split_by_l(CLASSES, sum, ids=lambda c: f"{c[0]}-{c[1]}")
JK_TOL = 1e-12  # vs libcint, times scale_J: test_ao_j.py's, inherited


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx)


def _ket_pairs(name: str):
    """Every unique ket shell pair `c >= d` of the case: `(c, d, w, ket,
    dcd)` with the host's weight and the host's row order of `D_cd`."""
    fx, basis = _fixture(name)
    D, off = fx.density, basis.offsets
    out = []
    for c, sc in enumerate(basis.shells):
        for d in range(c + 1):
            sd = basis.shells[d]
            dcd = [float(D[off[c] + i][off[d] + j])
                   for i in range(sc.n_cart) for j in range(sd.n_cart)]
            out.append((c, d, 1.0 if c == d else 2.0, epair(sc, sd), dcd))
    return out


def _by_class(name: str):
    groups: dict[tuple[int, int], list] = {}
    for item in _ket_pairs(name):
        ket = item[3]
        groups.setdefault((ket.la, ket.lb), []).append(item)
    return groups


# ---------------------------------------------------------------------------
# The layout: the same two helpers the host uses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASS_GRID)
def test_rows_and_columns_are_the_hosts(cls) -> None:
    lc, ld = cls
    assert list(ef.block_rows(lc, ld)) == _carts(lc, ld)
    assert hermite_targets(lc + ld) == hermite_order(lc + ld)
    n_cd, n_q = dq_shape(lc, ld)
    assert (n_cd, n_q) == (len(_carts(lc, ld)), len(hermite_order(lc + ld)))
    k = build_dq_kernel((lc, ld))
    shape = {p.name: p.shape for p in k.params}
    assert shape["Dcd"][1] == Lit(n_cd) and shape["D"][1] == Lit(n_q)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_the_key_is_a_pair_class() -> None:
    assert dq_spec(2, 1) == Spec(Operator.COULOMB, 0, (2, 1), Target.DQ)
    assert build_dq_kernel(dq_spec(2, 1)) == build_dq_kernel((2, 1))
    assert build_dq_kernel((2, 1)).name == "dq_2_1"
    assert tuple(p.name for p in build_dq_kernel((0, 0)).params) == \
        kernel_dq.PARAMS
    with pytest.raises(ValueError, match="not a Target.DQ"):
        build_dq_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.JAB))
    with pytest.raises(ValueError, match="no lever"):
        build_dq_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.DQ,
                             ("recycle",)))
    with pytest.raises(TypeError):
        build_dq_kernel(4)  # type: ignore[arg-type]


@pytest.mark.parametrize("cls", CLASS_GRID)
def test_the_kernel_decides_no_weight_and_no_dressing(cls) -> None:
    """The only floating literals are `0.0` (accumulators, `|x|`, `max`) and
    the `0.5` of `1/2eta`: no `2.0` that could be a weight, no `pi`, no
    contraction coefficient. `w` and `s` are read from their params."""
    lc, ld = cls
    k = build_dq_kernel((lc, ld))
    lits = {e.value for s in walk(k.body) for root in _exprs_of(s)
            if root is not None for e in _subexprs(root)
            if isinstance(e, Lit) and isinstance(e.value, float)}
    assert lits <= {0.0, 0.5}
    if lc + ld == 0:
        assert lits == {0.0}  # (ss) raises nothing: no 1/2eta
    stmts = list(walk(k.body))
    assert not any(isinstance(s, (Raw, AtomicAdd, WarpReduce)) for s in stmts)
    (inl,) = [s for s in stmts if isinstance(s, Inline)]
    assert inl.fragment == ef.block_fragment(lc, ld)  # E inlined, undressed
    assert "never decides" in kernel_dq.WEIGHT_IS_THE_HOSTS


# ---------------------------------------------------------------------------
# Interpreted against the host: every class (ss) through (ff), hf_ccpvtz
# ---------------------------------------------------------------------------


def test_hf_ccpvtz_reaches_every_class_with_both_weights() -> None:
    groups = _by_class("hf_ccpvtz")
    assert set(groups) == set(CLASSES)
    _, basis = _fixture("hf_ccpvtz")
    assert any(len(s.exponents) > 1 and s.l == 0 for s in basis.shells)
    assert {w for items in groups.values() for _, _, w, _, _ in items} == {
        1.0, 2.0}


@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_one_launch_per_class_is_the_host_pre_contraction(cls) -> None:
    """Every ket shell pair of the class in ONE launch (`pair[k]` selects its
    row of `Dcd`), against `dq_from_host` and `_kets`, bitwise; `Dmax` is
    `max_q |D_q|` of the host's weighted, dressed row, bitwise."""
    fx, basis = _fixture("hf_ccpvtz")
    rows = [[float(x) for x in r] for r in fx.density]
    items = _by_class("hf_ccpvtz")[cls]
    kd = KernelDQ()
    D, Dmax = kd.run([(ket, dcd, w) for _, _, w, ket, dcd in items])
    want = [row for _, _, w, ket, dcd in items
            for row in dq_from_host(ket, dcd, w)]
    sweep = [k.dq for c, d, w, _, _ in items
             for k in _kets(basis, rows, [(c, d, w)], pq=_no_pq, tau=0.0)]
    assert want == sweep  # the two host forms agree (#151)
    assert D.tolist() == [list(r) for r in want]  # bitwise
    assert Dmax.tolist() == [max(abs(v) for v in r) for r in want]
    assert kd.ran == {cls}
    # Dmax is over the weighted, dressed D_q, not the density's largest
    # element: the two differ on the class's records.
    dens = [max(abs(x) for x in dcd) for _, _, _, ket, dcd in items
            for _ in ket.recs]
    assert any(m != x for m, x in zip(Dmax, dens, strict=True) if x)


@pytest.mark.parametrize("lanes", [1, 4, 32])
def test_the_lane_mapping_is_not_numerical(lanes: int) -> None:
    items = [(ket, dcd, w) for _, _, w, ket, dcd in
             _by_class("hf_ccpvtz")[(1, 0)]]
    D, Dmax = KernelDQ(lanes=lanes).run(items)
    D1, Dmax1 = KernelDQ(lanes=1).run(items)
    assert D.tolist() == D1.tolist() and Dmax.tolist() == Dmax1.tolist()


def test_the_guard_leaves_spare_lanes_idle() -> None:
    # 3 records on a block of 32: the 29 spare threads write nothing.
    (_, _, w, ket, dcd) = _by_class("h2_sto3g_j")[(0, 0)][0]
    k = build_dq_kernel((0, 0))
    recs = ket.recs[:3]
    args = dict(
        n_rec=3, n_pair=1,
        eta=np.array([z for _, z, _ in recs]),
        XPA=np.array([[P[d] - ket.A[d] for d in range(3)] for P, _, _ in recs]),
        XPB=np.array([[P[d] - ket.B[d] for d in range(3)] for P, _, _ in recs]),
        s=np.array([s for _, _, s in recs]), w=np.full(3, w),
        pair=np.zeros(3, np.int32), Dcd=np.array([dcd]),
        D=np.full((3, 1), np.nan), Dmax=np.full(3, np.nan))
    out = interpret(k, launch=Launch(1, (32, 1), 32), **args)
    assert not np.isnan(out["D"]).any() and not np.isnan(out["Dmax"]).any()


# ---------------------------------------------------------------------------
# The weight: w = 2 off the diagonal, a host scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["h2o_sto3g_j", "hf_ccpvtz"])
def test_the_weight_is_the_hosts_and_the_kernel_is_linear_in_it(name) -> None:
    """Through the seam, per unique ket pair with the host's own `w`, the
    kernel equals the host; and the off-diagonal pairs it is given `w = 2`
    for come out exactly twice what `w = 1` gives -- the factor is the
    host's, and a kernel that 'decided' 1 there would be off by 2x."""
    kd = KernelDQ()
    weights = set()
    for c, d, w, ket, dcd in _ket_pairs(name):
        if name == "hf_ccpvtz" and max(ket.la, ket.lb) > 1:
            continue  # the classes above (pp) run in the slow cases above
        weights.add(w)
        assert kd(ket, dcd, w) == [list(r) for r in dq_from_host(ket, dcd, w)]
        if c != d:
            one, m1 = kd.run([(ket, dcd, 1.0)])
            two, m2 = kd.run([(ket, dcd, 2.0)])
            assert (two == 2.0 * one).all() and (m2 == 2.0 * m1).all()
            if np.any(one):
                assert not np.array_equal(one, two)
    assert weights == {1.0, 2.0}


def test_bad_launches_are_refused() -> None:
    items = _by_class("h2o_sto3g_j")
    ss = items[(0, 0)][0]
    ps = items[(1, 0)][0]
    kd = KernelDQ()
    with pytest.raises(ValueError, match="one ket class"):
        kd.run([(ss[3], ss[4], ss[2]), (ps[3], ps[4], ps[2])])
    with pytest.raises(ValueError, match="dcd has"):
        kd.run([(ps[3], ps[4][:-1], ps[2])])
    with pytest.raises(ValueError, match="no ket shell pairs"):
        kd.run([])


# ---------------------------------------------------------------------------
# The chain: the whole J through the dq= seam, against libcint
# ---------------------------------------------------------------------------

CHAIN = [pytest.param("h2_sto3g_j", id="h2_sto3g_j"),
         pytest.param("h2o_sto3g_j", id="h2o_sto3g_j"),
         pytest.param("h2o_ccpvdz_j", id="h2o_ccpvdz_j",
                      marks=pytest.mark.slow)]


@pytest.mark.parametrize("name", CHAIN)
def test_j_through_the_dq_kernel_matches_the_host_and_libcint(name) -> None:
    fx, basis = _fixture(name)
    D = fx.density
    kd = KernelDQ()
    Jk = coulomb_matrix(basis, D, pq=pq, dq=kd)
    host, scale = coulomb_matrix_and_scale(basis, D, pq=pq)
    assert Jk == host  # exact: the kernel's D_q are the host's, bitwise
    reached = {(a.l, b.l) for i, a in enumerate(basis.shells)
               for b in basis.shells[:i + 1]}
    assert kd.ran == reached
    ref = fx.arrays[manifest.J_ARRAY]
    err = np.abs(np.array(Jk) - ref)
    scale = np.array(scale)
    assert np.all(scale > 0)
    bad = err > JK_TOL * scale
    assert not bad.any(), (f"{name}: {int(bad.sum())} elements exceed "
                           f"1e-12 scale_J; worst {np.max(err / scale):.2e}")


# ---------------------------------------------------------------------------
# The [p|q] kernel is untouched
# ---------------------------------------------------------------------------


def test_the_pq_goldens_are_untouched() -> None:
    """Its goldens, one batch and split, on both backends: the text the
    generator writes for `Target.J` is what it wrote before this kernel."""
    test_emit_cuda.test_the_golden_kernel_is_unchanged()
    test_emit_hip.test_the_golden_kernel_is_unchanged()
    for backend in sorted(test_emit_batched.BACKENDS):
        for i in range(test_emit_batched.N_BATCHES):
            test_emit_batched.test_the_batched_golden_is_unchanged(backend, i)


@pytest.mark.slow
def test_every_recorded_pq_source_sha256_is_untouched() -> None:
    """Every `source_sha256` row of data/measured/kernels.json still hashes
    the text the generator writes today (~50 s: every recorded cell is
    re-emitted). The fast tier runs the same check in test_measured.py;
    here it is the issue's named assertion."""
    test_measured.test_the_recorded_source_is_what_the_generator_writes_today()
