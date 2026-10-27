"""`kernel_dq`: the ket pre-contraction kernel, `D_q` and `Dmax` (#153).

The kernel is checked against the HOST pre-contraction it replaces --
`eri.j_coulomb.j.dq_from_host` (the `dq=` seam's reference) and `ao.density.make_kets`
(the sweep's
own) -- interpreted, on real primitive-pair records. The reference is the host
path, so the f class needs no `J` fixture: `hf_ccpvtz` (s-f on F, s-d on H,
the F s shells a general contraction of 8 primitives) reaches every ordered
ket pair class (ss) through (ff), with its tracked density.

EXACT EQUALITY, NOT A TOLERANCE. The kernel performs the host's operations in
the host's order (`acc_q += x * (s * E[r, q])`, `r` outer, from `+0.0`; then
`w * acc_q`; `kernel_dq`'s module docstring says why the absent structural
zeros and the host's `D_cd = 0` skip change no bit), and the inlined `E`
fragment is bitwise `mmd.hermite.e_block` (test/test_e_fragment.py). So
`D_q` and `Dmax` are asserted `==`. It does NOT factor `s * w` out of the sum.

THE WEIGHT. `w = 2` off the diagonal is a host scalar; the kernel holds no
literal that could be a weight, and its output is linear in the `w` it is
given -- covered on `hf_ccpvtz` and on H2O/STO-3G, both of which have
off-diagonal ket pairs.

THE CHAIN. `eri.j_coulomb.j.coulomb_matrix(..., dq=KernelDQ())` against the host path
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
from oracle import basis as ob
from oracle import manifest
from tiers import by_l, split_by_l

from felizjueves.ao.density import epair, make_kets
from felizjueves.emit import cuda, hip
from felizjueves.eri.j_coulomb import kernel_dq
from felizjueves.eri.j_coulomb.drive import KernelDQ
from felizjueves.eri.j_coulomb.j import (
    cart_pairs,
    coulomb_matrix,
    coulomb_matrix_and_scale,
    dq_from_host,
)
from felizjueves.eri.j_coulomb.kernel_dq import build_dq_kernel, dq_shape, dq_spec
from felizjueves.ir import (
    AtomicAdd,
    Decl,
    Inline,
    Lit,
    Raw,
    Var,
    WarpReduce,
    _exprs_of,
    _subexprs,
    walk,
)
from felizjueves.ir.interp import Launch, interpret
from felizjueves.ir.passes import split_axes, transpose
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.hermite import hermite_order
from felizjueves.performance.device import GFX1200, SM_86
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


def _no_pq(p, q, PQv, theta) -> float:
    raise AssertionError("no [p|q] is evaluated building kets at tau = 0")


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
    assert list(ef.block_rows(lc, ld)) == cart_pairs(lc, ld)
    assert hermite_targets(lc + ld) == hermite_order(lc + ld)
    n_cd, n_q = dq_shape(lc, ld)
    assert (n_cd, n_q) == (len(cart_pairs(lc, ld)), len(hermite_order(lc + ld)))
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
    want = (ef.axis_tables_fragment(lc, ld) if kernel_dq.dq_lowstack((lc, ld))
            else ef.block_fragment(lc, ld))
    assert inl.fragment == want  # E inlined, undressed (#586: per axis at f)
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
             for k in make_kets(basis, rows, [(c, d, w)], pq=_no_pq, tau=0.0)]
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

# That every recorded `source_sha256` row of data/measured/kernels.json still
# hashes today's text is a check of the measured store, not of this kernel:
# measure/measure_kernel_dq.py (`devtools/measure.sh check`).


# ---------------------------------------------------------------------------
# density="ao": the AO density read in place, the same bits as the gather
# ---------------------------------------------------------------------------


def _records(pairs, off) -> dict:
    """Per-record arrays of every (c, d, w, ket) in `pairs`, both variants'
    addressing: `pair` (the item index) and `off_c` / `off_d`."""
    a: dict[str, list] = {n: [] for n in (
        "eta", "XPA", "XPB", "s", "w", "pair", "off_c", "off_d")}
    for i, (c, d, w, ket) in enumerate(pairs):
        for P, zeta, sk in ket.recs:
            a["eta"].append(zeta)
            a["XPA"].append([P[x] - ket.A[x] for x in range(3)])
            a["XPB"].append([P[x] - ket.B[x] for x in range(3)])
            a["s"].append(sk)
            a["w"].append(w)
            a["pair"].append(i)
            a["off_c"].append(off[c])
            a["off_d"].append(off[d])
    ints = ("pair", "off_c", "off_d")
    return {n: np.array(v, dtype=np.int32 if n in ints else np.float64)
            for n, v in a.items()}


def _both(cls, pairs, dcds, density_ao) -> tuple[dict, dict]:
    """`D`, `Dmax` from the gathered kernel and from the AO-density one."""
    lc, ld = cls
    _, basis = _fixture("hf_ccpvtz")
    a = _records(pairs, basis.offsets)
    n_rec, n_q = len(a["eta"]), dq_shape(lc, ld)[1]
    common = dict(n_rec=n_rec, eta=a["eta"], XPA=a["XPA"], XPB=a["XPB"],
                  s=a["s"], w=a["w"])
    launch = Launch(*kernel_dq.launch_shape(n_rec, 32), 32)

    def out() -> dict:
        return dict(D=np.zeros((n_rec, n_q)), Dmax=np.zeros(n_rec))

    gathered = interpret(build_dq_kernel(cls), launch=launch, **common,
                         n_pair=len(pairs), pair=a["pair"],
                         Dcd=np.array(dcds, dtype=np.float64), **out())
    ao = interpret(build_dq_kernel(cls, density="ao"), launch=launch,
                   **common, n_ao=density_ao.shape[0], off_c=a["off_c"],
                   off_d=a["off_d"], Dao=density_ao, **out())
    return gathered, ao


@pytest.mark.parametrize("cls", CLASS_PARAMS)
def test_the_ao_density_is_the_gather_bitwise(cls) -> None:
    """Every ket shell pair of the class, one launch each way, against each
    other and against `dq_from_host`, `==`."""
    fx, _ = _fixture("hf_ccpvtz")
    items = _by_class("hf_ccpvtz")[cls]
    dense = np.array(fx.density, dtype=np.float64)
    gathered, ao = _both(cls, [(c, d, w, ket) for c, d, w, ket, _ in items],
                         [dcd for *_, dcd in items], dense)
    want = [row for _, _, w, ket, dcd in items
            for row in dq_from_host(ket, dcd, w)]
    assert ao["D"].tolist() == gathered["D"].tolist() == [list(r) for r in want]
    assert ao["Dmax"].tolist() == gathered["Dmax"].tolist()


def test_a_pair_stored_swapped_reads_its_own_orientation() -> None:
    """A (d, c) record of an (s|p) pair stored as (p, s)'s mirror -- a
    consumer that keeps every class la <= lb (FreeBreakfast's mmd) stores
    (p|s) so -- reads D[off_d + i][off_c + j] with no transposed copy: the
    AO kernel equals the host on the transposed gather, `==`."""
    fx, basis = _fixture("hf_ccpvtz")
    dense = np.array(fx.density, dtype=np.float64)
    off = basis.offsets
    swapped = []
    for c, d, w, _, _ in _by_class("hf_ccpvtz")[(1, 0)]:
        sd, sc = basis.shells[d], basis.shells[c]
        dcd_t = [float(dense[off[d] + i][off[c] + j])
                 for i in range(sd.n_cart) for j in range(sc.n_cart)]
        swapped.append((d, c, w, epair(sd, sc), dcd_t))
    assert swapped and all(ket.la == 0 and ket.lb == 1
                           for _, _, _, ket, _ in swapped)
    gathered, ao = _both((0, 1), [(d, c, w, ket) for d, c, w, ket, _ in swapped],
                         [dcd for *_, dcd in swapped], dense)
    want = [row for _, _, w, ket, dcd in swapped
            for row in dq_from_host(ket, dcd, w)]
    assert ao["D"].tolist() == gathered["D"].tolist() == [list(r) for r in want]


def test_the_ao_variant_is_its_own_parameter_block() -> None:
    k = build_dq_kernel((2, 1), density="ao")
    assert k.name == "dq_2_1_ao" == kernel_dq.dq_name((2, 1), density="ao")
    assert tuple(p.name for p in k.params) == kernel_dq.AO_PARAMS
    shape = {p.name: p.shape for p in k.params}
    assert shape["Dao"] == (Var("n_ao"), Var("n_ao"))
    assert "Dcd" not in shape and "pair" not in shape
    # the default is untouched, and named as before
    assert build_dq_kernel((2, 1), density="pairs") == build_dq_kernel((2, 1))
    assert build_dq_kernel((2, 1)).name == "dq_2_1"
    with pytest.raises(ValueError, match="density is one of"):
        build_dq_kernel((2, 1), density="dense")


# ---------------------------------------------------------------------------
# The low-stack form (#586, stage 2): the default at l_c + l_d >= 6, the
# text below it unchanged, the same bits
# ---------------------------------------------------------------------------

# sha256 of the unrolled DQ text of every class below L = 6 with l <= 3 --
# through (dd), and the mixed f classes (sf), (pf), (df) -- cuda and hip,
# both densities, as emitted at 534a8e7 (before stage 2): the low-stack form
# must not move a byte of it.
D_DIGESTS = {
    (0, 0): ("2764fd37ca241f6c82618334231923465eef95e201bb6a9a368eea9b982d44b6",
             "57c9a84ce87013865598285e004e3294d0b1b09808f663cfafa2b716d53f9ccc"),
    (0, 1): ("a949f8f4c776f8fc76a089040ba7faf9f73fb574a46198239676fc7777b6379c",
             "a6c7024b673cc7b91e6e522d2ddf9af4c2456519f4245046183772d1f5ed761c"),
    (0, 2): ("97f94d56b6f78eecb287c8d06931da68670c34581debe5319a35f11eee36b7bb",
             "8a426c3240037c414ab897800947f0fe3cb5e2fc619770186f17f257814de293"),
    (0, 3): ("76b66ca12a2e518c2ef729de1802d635c08980e2513776f93ed5eb52129cc9f0",
             "3eaabeed7e225a38fbcfe90f62d086ee380121d918a6044bf8c0d14a43b209c1"),
    (1, 0): ("30c9a87169405a6ea24a00b1ec6efa60e0664d31e9e907c13d57b40d8a3cedb0",
             "0991f263e73167fc7edf9c30cfdfd57aa365b57ac03a64bc4f4c72b069cb2cd4"),
    (1, 1): ("e4b0d79c37f5ea11224a73f3dbd1e1079cc98d737c4c5c24c0bb2d66bf8001a8",
             "fc53549485cdee57aaa7dfe64fd6fb95be29ab1d6480720afa5c5fa7b06af6b1"),
    (1, 2): ("661f1ec3002d78561e426aa70131db69c19608b98fb71f85b266047231ad10c5",
             "ff767069aa9dccdfa28fcd8968827e27092c703a1311420259ffabef059b23a7"),
    (1, 3): ("ba0f64a2fbbd7cbf5390c03eaedba502bcff0f71acae0d67c64c14d979f66a01",
             "1c13120c2cf0e83152734391e1ab75af65895962fb5c8805d79957c14f6e669a"),
    (2, 0): ("e5cbd306c18cfc4b27ac643af575c4b41c863d0becbeadf83d9fffa6300c631b",
             "6ee6b07e63055e6fec0b9e209c7e9c239c565d57f0583d8c1953f55fe3d0d809"),
    (2, 1): ("76124940859fa08b633fb2e782590882f8be38b4a6e005cab2581f8e22cb1e33",
             "8668ad101a83b3a0cece914407bc02d77fc1e547fa59fd57fb8fecfd09d58cc0"),
    (2, 2): ("91f7cabfc356c9bc0503b360cbb29d7731ba4bfb320a136ea80e4555d10b7936",
             "e6dd3d10ce90d163fcbb505c65bee6378e2a3c29e3ef259eff1f904a2424f180"),
    (2, 3): ("ccabae2b8fb51f644c76af24560585d4dd60a4bc9ac2286bf63931f489939d88",
             "36af6154485f6066ba749485cb070a102e022fe9ea8a71182c003b2c0da9c832"),
    (3, 0): ("d51bf77e0a4a761708e9226a1a814614a13f3f0e537539fbfc1714ec6dded209",
             "655b86a50d5f74422551cabdc90074a0b6db18e4ce9c184c154743ecc3d11ba6"),
    (3, 1): ("bf107373ab830515a559d7ba18ad01924459a59c4cd8956b7f1b379327c9bbba",
             "4438a5ace4e53eaada2b4fa883265f8eddb21bad3347eda3490efc135224b117"),
    (3, 2): ("c5d0501e3659f4bfb719df433c1921acd244ff304fe287acd5874282439882a5",
             "48cdfcfa9e52c4009e1b9a90df1d3917d76bd524c95c4a2e89b92366d6ca171c"),
}
D_DIGESTS_AO = {
    (0, 0): ("554bbdbbc6640fff4c36a6444da3e61a2ef5d42b9a795bfbefe7c260368fc252",
             "769796da5bba549b227309b3b1814bd2b1cac2cd0bcbcecbd3b49e9621179831"),
    (0, 1): ("bd05726cd2648683857e3107d726b029c36b61f4e6dbe9ea6451a09d10a451bf",
             "86f010bf1a1dfc49bc6ea871a055078749f00b7139b29039adf4010d61537147"),
    (0, 2): ("c324385dd6348b052fa6d78d81611b2b30c0c434b50d0855f57b772680f32e8c",
             "d84aa283615cf82a2fba4d8f7d65444575a2f0ffa02342192a294f697d5fb128"),
    (0, 3): ("d06d62f0376a85047f1336516f4760e06b294bbe092a6565a46194210e0d36e2",
             "763dc195d94a568b93c546dc0d1d39db91229b0ce456ccfd4cae3273b4d5c586"),
    (1, 0): ("18e4fb24e414108a540cabcfd2b178d73fe85abdde7143716b7f80624a6aa2dd",
             "ac4d9e1571ba2b2150756b3942ea13d338a5b2e64328a3ec4ef0008642e25965"),
    (1, 1): ("59661e9d7c655bc22559391236fea088dc1831c7c975a5b45c6e2acac0f1f841",
             "2c9eda10efc9ac0eca4e672064d1b98e58de767352d4cd86b0169306d672e1a2"),
    (1, 2): ("0ff358e1d33209125e7d0097b5f9247c6730bf22075c6bd4b4537bf56652fe80",
             "41f6cc982663ad2a398a378eb6b5599555af75d297c8f95086c06a0dcd0001d1"),
    (1, 3): ("d95066755bab975ea94dd0d5ca2b534a48b65247fb3fba24392b681aad9d12b6",
             "7cdffa5b3a6f847a87bdec78f12fccd3e81c7bc5ac2ebbee51f5cd6b24ba65e0"),
    (2, 0): ("326318e576d2b5e467b218da170c6faece5e295e5a5e6624ca056991c977d2fa",
             "20f2c16c95bfae5f07c9b1d7a65ed1c063890701fb67347a3246d91b312fdccb"),
    (2, 1): ("858c968e9115d2f2b5b8fade69f958b6a5da661276fe9bea80ed0356c893d629",
             "30431dd05a17f79d4fa2f4610a2d6437885def54f47b2c7b9d802ae749a02bef"),
    (2, 2): ("1f413f25b8b11a27dc09557a2d7424f755aeba91e2a65062d261f4300ddc8608",
             "a1817819ae47e2a5e2d62f989c37719cb52f04c9d08489b077813ba3aa99adbb"),
    (2, 3): ("3b3810ab05e68876b2a37842c8ccc76e689766093817faea67b37a5e431b8288",
             "a16b0c473a06c5f0086a8a344dd1e9b12382a2967830871535ed7bcb67c7195a"),
    (3, 0): ("e0c3a4113e77a8c67c3763cf8c9d21303ae098df4075351436ca078a129eab14",
             "729591831f74858d35aa2a4a8c250c033ddf9683ad5dc047dc313196164ab99e"),
    (3, 1): ("1684e2e8e9111f52bab9ed9c43e2e84fa894b2a39560b4c1449b811bdf2cefd8",
             "bb3fba2472a540f3af9c032fdb7065b31b40a9a496c7219b4eb18caf0fd96adb"),
    (3, 2): ("40c3526387a83ae90a62e4d91fff62235405012778c692e1fb759c46e42989c2",
             "2d4e7587015a903beb7e6ae141b81309e89b1ba7cf216ebbf1d12642138b88a0"),
}

# #586's budget: at most 8 KB of stack per thread at f on sm_86. The local
# arrays are the whole of what the low-stack kernel can spill by design, so
# they are held to it here, without a compiler (the measured frames are in
# docs/implementation/j_engine.md).
STACK_BUDGET = 8192
F_CLASSES = [(0, 3), (1, 3), (2, 3), (3, 3), (3, 0), (3, 1), (3, 2)]


def test_the_selection_rule_is_pair_l_6_and_above() -> None:
    # (ff), and past f (gd), (gf), ...; the mixed f classes stay unrolled
    assert kernel_dq.LOWSTACK_MIN_PAIR_L == 6
    for lc in range(5):
        for ld in range(5):
            assert kernel_dq.dq_lowstack((lc, ld)) is (lc + ld >= 6)
            assert kernel_dq.dq_lowstack(dq_spec(lc, ld)) is \
                kernel_dq.dq_lowstack((lc, ld))


@pytest.mark.parametrize("density", kernel_dq.DENSITIES)
@pytest.mark.parametrize("cls", split_by_l(sorted(D_DIGESTS), sum,
                                           ids=lambda c: f"{c[0]}{c[1]}"))
def test_the_text_below_l6_is_byte_identical(cls, density) -> None:
    import hashlib
    k = build_dq_kernel(cls, density=density)
    assert k == build_dq_kernel(cls, density=density, lowstack=False)
    got = tuple(hashlib.sha256(t.encode()).hexdigest()
                for t in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)))
    want = D_DIGESTS_AO if density == "ao" else D_DIGESTS
    assert got == want[cls]


@pytest.mark.parametrize("density", kernel_dq.DENSITIES)
@pytest.mark.parametrize("cls", split_by_l([(1, 1), (2, 3), (3, 2), (3, 3)], sum,
                                           ids=lambda c: f"{c[0]}{c[1]}"))
def test_both_forms_share_the_name_parameters_and_launch(cls, density) -> None:
    lo = build_dq_kernel(cls, density=density, lowstack=True)
    hi = build_dq_kernel(cls, density=density, lowstack=False)
    assert lo.name == hi.name == kernel_dq.dq_name(cls, density=density)
    assert lo.params == hi.params
    want = kernel_dq.AO_PARAMS if density == "ao" else kernel_dq.PARAMS
    assert tuple(p.name for p in lo.params) == want
    assert build_dq_kernel(cls, density=density) == (
        lo if kernel_dq.dq_lowstack(cls) else hi)
    rec = [s for s in lo.body if isinstance(s, Decl) and s.name == "k"]
    assert rec and rec == [s for s in hi.body
                           if isinstance(s, Decl) and s.name == "k"]


def _local_doubles(k) -> dict[str, int]:
    return {s.name: int(np.prod([d.value for d in s.shape]))
             for s in walk(k.body) if isinstance(s, Decl) and s.shape}


@pytest.mark.parametrize("cls", split_by_l(F_CLASSES, sum,
                                           ids=lambda c: f"{c[0]}{c[1]}"))
def test_the_lowstack_form_holds_no_e_block(cls) -> None:
    # every f class, forced where the rule keeps the unrolled form
    lc, ld = cls
    for density in kernel_dq.DENSITIES:
        k = build_dq_kernel(cls, density=density, lowstack=True)
        arrays = _local_doubles(k)
        ext = ef.axis_table_extent(lc, ld)
        assert arrays == {"Ex": ext, "Ey": ext, "Ez": ext,
                           "acc": dq_shape(lc, ld)[1]}
        assert 8 * sum(arrays.values()) <= STACK_BUDGET
        (inl,) = [s for s in walk(k.body, into_inline=False)
                  if isinstance(s, Inline)]
        assert inl.fragment == ef.axis_tables_fragment(lc, ld)


@pytest.mark.slow  # builds (ff), L = 6 > FAST_L_MAX (tiers.py)
def test_the_ff_lowstack_arrays_are_420_doubles() -> None:
    # 3 x 112 axis-table entries and 84 accumulators: 3,360 B, all of
    # sm_86's measured frame for dq_33.
    assert sum(_local_doubles(build_dq_kernel((3, 3))).values()) == 420


def _other_form(cls) -> bool:
    return not kernel_dq.dq_lowstack(cls)


@pytest.mark.parametrize("cls", split_by_l(CLASSES, sum,
                                           ids=lambda c: f"{c[0]}{c[1]}"))
def test_the_other_form_is_the_host_pre_contraction_bitwise(cls) -> None:
    """Every class through (ff) in the form `build_dq_kernel` does NOT choose
    by default -- the low-stack form through d, the unrolled one at f --
    both densities, every ket shell pair of the class in one launch,
    against `dq_from_host`, `==`: with the default-form tests above, both
    forms are held bitwise at every class."""
    fx, basis = _fixture("hf_ccpvtz")
    items = _by_class("hf_ccpvtz")[cls]
    pairs = [(c, d, w, ket) for c, d, w, ket, _ in items]
    a = _records(pairs, basis.offsets)
    n_rec, n_q = len(a["eta"]), dq_shape(*cls)[1]
    common = dict(n_rec=n_rec, eta=a["eta"], XPA=a["XPA"], XPB=a["XPB"],
                  s=a["s"], w=a["w"])
    launch = Launch(*kernel_dq.launch_shape(n_rec, 32), 32)
    want = [list(row) for _, _, w, ket, dcd in items
            for row in dq_from_host(ket, dcd, w)]
    dense = np.array(fx.density, dtype=np.float64)
    for density, extra in (
            ("pairs", dict(n_pair=len(pairs), pair=a["pair"],
                           Dcd=np.array([d for *_, d in items]))),
            ("ao", dict(n_ao=dense.shape[0], off_c=a["off_c"],
                        off_d=a["off_d"], Dao=dense))):
        k = build_dq_kernel(cls, density=density, lowstack=_other_form(cls))
        out = interpret(k, launch=launch, **common, **extra,
                        D=np.full((n_rec, n_q), np.nan),
                        Dmax=np.full(n_rec, np.nan))
        assert out["D"].tolist() == want, density
        assert out["Dmax"].tolist() == [max(abs(v) for v in r) for r in want]


@pytest.mark.parametrize("cls", [by_l((1, 2), 3, id="12"),
                                 by_l((3, 3), 6, id="33")])
def test_the_freebreakfast_spelling_takes_both_forms(cls) -> None:
    # free-breakfast's tools/emit_coulomb_kernels.py: density="ao", the
    # record vectors split per axis, D transposed q-major. The passes take
    # the low-stack form as they take the unrolled one, and the parameters
    # still match.
    axes = {"XPA": ("pa_x", "pa_y", "pa_z"), "XPB": ("pb_x", "pb_y", "pb_z")}

    def fbf(lowstack: bool):
        k = build_dq_kernel(cls, density="ao", name="dq", lowstack=lowstack)
        return transpose(split_axes(k, axes), ("D",))

    lo, hi = fbf(True), fbf(False)
    assert lo.params == hi.params
    assert "D[(q * n_rec + k)] = dq;" in cuda.emit(lo, SM_86)


# The (ff) default text, both backends: the low-stack form's golden.
LOWSTACK_GOLDEN_CLASS = (3, 3)
GOLDEN = test_emit_cuda.GOLDEN.parent.parent


@pytest.mark.slow  # emits (ff), L = 6 > FAST_L_MAX (tiers.py)
def test_the_lowstack_cuda_golden_is_unchanged() -> None:
    # Regenerate (never hand-edit), the default form at (ff):
    #   PYTHONPATH=src uv run python -c "from felizjueves.eri.j_coulomb.kernel_dq \
    #   import build_dq_kernel as b; from felizjueves.performance.device import \
    #   SM_86; from felizjueves.emit.cuda import emit; print(emit(b((3, 3)), \
    #   SM_86), end='')" > test/golden/cuda/dq_3_3.cu
    text = cuda.emit(build_dq_kernel(LOWSTACK_GOLDEN_CLASS), SM_86)
    assert text == (GOLDEN / "cuda" / "dq_3_3.cu").read_text()


@pytest.mark.slow  # emits (ff)
def test_the_lowstack_hip_golden_is_unchanged() -> None:
    #   ... the same with GFX1200 and felizjueves.emit.hip, into
    #   test/golden/hip/dq_3_3.hip
    text = hip.emit(build_dq_kernel(LOWSTACK_GOLDEN_CLASS), GFX1200)
    assert text == (GOLDEN / "hip" / "dq_3_3.hip").read_text()
