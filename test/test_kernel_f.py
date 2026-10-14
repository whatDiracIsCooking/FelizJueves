"""`kernel_f.py` (#282): the fused Fock kernel up to the ERIs, interpreted.

THE HOST PATH it is held to BITWISE is built here from #279's pinned pieces,
with no fused kernel in it: per primitive quartet (ket primitive outer, bra
inner) the scalars `kernel_f`'s docstring lists, computed in Python floats in
the same operation order; `interpret(hgp.recurrence.build_vrr_kernel)` for
the boundary `[e0|f0]^(0)`; the left fold `c = c + v` from `+0.0`; and
`interpret(build_hrr_kernel)` on the sums. `test_hgp_recurrence.py` holds
those two kernels to a direct recursion of eqs 25-26 and 28 and to
`ao/eri.py`, so a bitwise match here says the fused nest -- the staging, the
loops, the accumulators, HRR after the loop -- adds nothing and drops
nothing. The integrals are also checked against `ao/eri.py`'s MMD path on
the same contracted shells (a different engine; a rounding tolerance).

Fast tier for `L <= 4`, `slow` above (#282). No compiler, no card, no
emission (#287): the interpreter is the whole check at this stage.
"""

from __future__ import annotations

import math
import time
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves import kernel_f as KF
from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.ao.pairs import primitive_schwarz_bounds
from felizjueves.ao.shell import Shell
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp import recurrence as R
from felizjueves.interp import Interpreter, Launch, interpret
from felizjueves.ir import (
    Assign,
    Break,
    Call,
    Decl,
    Fn,
    For,
    If,
    Note,
    Space,
    Var,
    walk,
)
from felizjueves.launch_f import f_launches
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, pq
from felizjueves.spec import Operator, Spec, Target

CLASSES = vrr.CLASSES


def _speed(cls):
    """Fast for `L <= 4`, `slow` above (#282)."""
    return pytest.param(cls, id=str(cls),
                        marks=(pytest.mark.slow,) if sum(cls) > 4 else ())


LADDER = [_speed(c) for c in CLASSES]

# ---------------------------------------------------------------------------
# The pinned statement counts: the plan, not silently re-derived
# ---------------------------------------------------------------------------

# (vrr nodes, boundary accumulators, HRR statements incl. boundary reads,
#  HRR two-term updates, ERIs) per class, as built when #282 landed. The
# boundary is the BOUNDED set (961 at (dd|dd), not #282's printed 1,225 --
# #273/#274's comments), and the HRR column is #274's convention: 6,254 at
# (dd|dd) = 961 boundary reads + 5,293 updates.
COUNTS = {
    (0, 0, 0, 0): (1, 1, 1, 0, 1),
    (0, 0, 0, 1): (5, 4, 7, 3, 3),
    (0, 0, 0, 2): (15, 10, 25, 15, 6),
    (0, 0, 1, 1): (15, 9, 18, 9, 9),
    (0, 0, 1, 2): (35, 19, 60, 41, 18),
    (0, 0, 2, 2): (70, 31, 110, 79, 36),
    (0, 1, 0, 1): (21, 16, 37, 21, 9),
    (0, 1, 0, 2): (61, 40, 115, 75, 18),
    (0, 1, 1, 1): (61, 36, 90, 54, 27),
    (0, 1, 1, 2): (141, 76, 256, 180, 54),
    (0, 1, 2, 2): (281, 124, 454, 330, 108),
    (0, 2, 0, 2): (155, 100, 340, 240, 36),
    (0, 2, 1, 1): (155, 90, 279, 189, 54),
    (0, 2, 1, 2): (355, 190, 721, 531, 108),
    (0, 2, 2, 2): (705, 310, 1249, 939, 216),
    (1, 1, 1, 1): (146, 81, 243, 162, 81),
    (1, 1, 1, 2): (336, 171, 711, 540, 162),
    (1, 1, 2, 2): (671, 279, 1269, 990, 324),
    (1, 2, 1, 2): (696, 361, 1878, 1517, 324),
    (1, 2, 2, 2): (1381, 589, 3282, 2693, 648),
    (2, 2, 2, 2): (2330, 961, 6254, 5293, 1296),
}


def test_the_count_table_covers_the_ladder() -> None:
    assert tuple(COUNTS) == CLASSES and len(CLASSES) == 21


@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_emitted_statement_counts_are_the_dags(cls) -> None:
    """Read off the BUILT kernel by statement kind, equal to the pinned table
    and to `hgp.vrr` / `hgp.hrr`'s own node counts (#273, #274)."""
    k = KF.build_f_kernel(cls)
    got = KF.counted(k)
    assert tuple(vars(got).values()) == COUNTS[cls]
    assert got == KF.kernel_counts(cls)
    v, h = vrr.build(cls), hrr.build(cls)
    assert got.vrr == v.n_nodes
    assert got.boundary == h.n_boundary == len(vrr.target_set(cls))
    assert got.hrr == h.n_statements == h.n_boundary + h.n_interior
    assert got.eri == h.n_eri
    # and the eq 27 update: one `c = c + v` per boundary value
    acc = [s for s in walk(k.body) if isinstance(s, Assign)
           and isinstance(s.dst, Var) and s.dst.name.startswith("C")]
    assert len(acc) == h.n_boundary


def test_dd_dd_is_961_boundary_values_not_1225() -> None:
    """The orchestrator's reading of #282's 1,225 (35^2) as a misprint for
    the bounded set's 31^2: only 961 fits #274's 6,254 HRR statements."""
    c = KF.counted(KF.build_f_kernel((2, 2, 2, 2)))
    assert (c.boundary, c.hrr, c.hrr_interior) == (961, 6254, 5293)
    assert c.boundary == 31 * 31 != 35 * 35


def test_dd_dd_builds_and_its_construction_time_is_immaterial() -> None:
    t = time.perf_counter()
    k = KF.build_f_kernel((2, 2, 2, 2))
    dt = time.perf_counter() - t
    assert k.name == "f_2222"
    # ~0.2 s when written; the bound only says "not material" (#282 asks
    # that a material one be reported)
    assert dt < 10.0, dt


# ---------------------------------------------------------------------------
# The shape: nest, staging, screen, HRR after the loop
# ---------------------------------------------------------------------------


def _find_for(body, var):
    return next(s for s in walk(body) if isinstance(s, For) and s.var == var)


def test_the_nest_is_ket_then_kcd_then_kab_and_hrr_runs_after_it() -> None:
    cls = (1, 1, 1, 1)
    k = KF.build_f_kernel(cls)
    ket = _find_for(k.body, KF.KET)
    assert ket.start == Call(Fn.THREAD_X) and ket.step == Call(Fn.BLOCK_DIM_X)
    kcd = _find_for(ket.body, KF.KCD)
    kab = _find_for(kcd.body, KF.KAB)
    inner = list(walk(kab.body))
    vrr_in = {s.name for s in inner if isinstance(s, Decl)
              and s.name.startswith("V")}
    assert len(vrr_in) == vrr.build(cls).n_nodes  # every VRR node, inside
    hrr_names = {R.hrr_name(n) for n in hrr.build(cls).terms}
    assert not hrr_names & {s.name for s in walk(kcd.body)
                            if isinstance(s, Decl)}  # no HRR in the loops
    # in the quartet's work: accumulators, then the nest, then the HRR
    screen = next(s for s in walk(ket.body) if isinstance(s, If))
    work = screen.orelse
    i_loop = next(i for i, s in enumerate(work) if isinstance(s, For))
    decls_before = [s.name for s in work[:i_loop] if isinstance(s, Decl)]
    decls_after = [s.name for s in work[i_loop + 1:] if isinstance(s, Decl)]
    assert sum(n.startswith("C") and not n.startswith("CD_")
               for n in decls_before) == 81
    assert set(decls_after) == hrr_names


def test_the_bra_record_is_staged_in_shared_memory_under_the_cap() -> None:
    k = KF.build_f_kernel((0, 1, 0, 1))
    sh = [s for s in walk(k.body) if isinstance(s, Decl)
          and s.space is Space.SHARED]
    assert [s.name for s in sh] == [KF.AB_SH]
    assert [d.value for d in sh[0].shape] == [KF.MAX_PRIM_PAIRS, len(KF.REC)]
    stage = _find_for(k.body, KF.STAGE)
    ket = _find_for(k.body, KF.KET)
    order = list(walk(k.body))
    assert order.index(stage) < order.index(ket)  # staged before any ket


def _the_screen(k):
    ket = _find_for(k.body, KF.KET)
    ifs = [s for s in ket.body if isinstance(s, If)]
    assert len(ifs) == 1
    return ket, ifs[0]


@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)],
                         ids=str)
def test_fock_skip_not_break_the_screen_is_a_skip(cls) -> None:
    """`fock/skip-not-break` (pending; #291 scores it, so no claim mark): the
    in-kernel screen guards the quartet's work and exits nothing. Its `then`
    holds only a `Note`, its `orelse` the whole quartet, and there is no
    `Break` anywhere in the kernel -- the inverse of `Target.K`'s sorted
    early break, built beside it in the same IR."""
    k = KF.build_f_kernel(cls)
    ket, screen = _the_screen(k)
    assert screen.cond == Call(Fn.LT, (Var("bound"), Var("tau")))
    assert all(isinstance(s, Note) for s in screen.then)
    assert not any(isinstance(s, Break) for s in walk(screen.then))
    assert not any(isinstance(s, Break) for s in walk(screen.orelse))
    assert not any(isinstance(s, Break) for s in walk(k.body))
    assert _find_for(screen.orelse, KF.KCD)  # the work is the orelse
    # the contrast, in the same IR: Target.K's screen is a break
    from felizjueves.kernel_k import build_k_kernel
    kk = build_k_kernel((1, 0, 0, 0))
    assert any(isinstance(s, Break) for s in walk(kk.body))


def test_a_non_f_class_is_refused() -> None:
    with pytest.raises(ValueError, match="Target.F"):
        KF.build_f_kernel(Spec(Operator.COULOMB, 0, (1, 0, 0, 0), Target.K))
    with pytest.raises(TypeError):
        KF.build_f_kernel([0, 0, 0, 0])  # type: ignore[arg-type]


def test_params_follow_the_hrr() -> None:
    assert "AB" not in KF.params_of((0, 0, 1, 1))
    assert "CD" in KF.params_of((0, 0, 1, 1))
    assert "AB" in KF.params_of((0, 1, 0, 0))
    assert "CD" not in KF.params_of((0, 1, 0, 0))
    k = KF.build_f_kernel((0, 0, 0, 0))
    assert tuple(p.name for p in k.params) == KF.params_of((0, 0, 0, 0))


# ---------------------------------------------------------------------------
# The packer and its cap
# ---------------------------------------------------------------------------


def _prim(z=1.0):
    return KF.PrimRec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), z, 1.0)


def test_the_packer_refuses_past_the_cap_and_a_mixed_degree() -> None:
    at_cap = KF.PairRec((0.0, 0.0, 0.0), 1.0, (_prim(),) * KF.MAX_PRIM_PAIRS)
    assert KF.pack([at_cap]).degree == KF.MAX_PRIM_PAIRS
    over = KF.PairRec((0.0, 0.0, 0.0), 1.0,
                      (_prim(),) * (KF.MAX_PRIM_PAIRS + 1))
    with pytest.raises(ValueError, match="MAX_PRIM_PAIRS"):
        KF.pack([over])
    with pytest.raises(ValueError, match="degree"):
        KF.pack([KF.PairRec((0.0,) * 3, 1.0, (_prim(),)),
                 KF.PairRec((0.0,) * 3, 1.0, (_prim(), _prim()))])
    with pytest.raises(ValueError, match="empty"):
        KF.pack([])


def test_the_packed_layout_is_row_major() -> None:
    a = KF.PrimRec((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), 7.0, 8.0)
    b = KF.PrimRec((9.0, 10.0, 11.0), (12.0, 13.0, 14.0), 15.0, 16.0)
    p = KF.pack([KF.PairRec((0.1, 0.2, 0.3), 0.5, (a, b))])
    assert (p.n, p.degree) == (1, 2)
    assert p.prim == tuple(float(x) for x in range(1, 17))
    assert p.sep == (0.1, 0.2, 0.3) and p.G == (0.5,)


# ---------------------------------------------------------------------------
# Records from shells, the host path, and the fused kernel interpreted
# ---------------------------------------------------------------------------


def _sub(u, v):
    return tuple(a - b for a, b in zip(u, v, strict=True))


def records(sa: Shell, sb: Shell, G: float = 1.0, keep=None) -> KF.PairRec:
    """One contracted pair's record, primitive pairs in
    `ao.pairs.primitive_pairs` order (i outer) or the indices `keep`."""
    A, B = sa.centre, sb.centre
    AB = _sub(A, B)
    ab2 = sum(x * x for x in AB)
    prims = []
    for a, ca in zip(sa.exponents, sa.coefficients, strict=True):
        for b, cb in zip(sb.exponents, sb.coefficients, strict=True):
            p = a + b
            P = tuple((a * x + b * y) / p for x, y in zip(A, B, strict=True))
            U = ca * cb * (math.pi / p) ** 1.5 * math.exp(-a * b / p * ab2)
            prims.append(KF.PrimRec(P, _sub(P, A), p, U))
    if keep is not None:
        prims = [prims[i] for i in keep]
    return KF.PairRec(AB, G, tuple(prims))


def _scalars(b: KF.PrimRec, k: KF.PrimRec, AB, CD) -> dict:
    """The fused kernel's per-quartet scalars, in ITS operation order."""
    p, q = b.zeta, k.zeta
    PQ = tuple(x - y for x, y in zip(b.P, k.P, strict=True))
    ppq = p + q
    rho = (p * q) / ppq
    q_ppq, p_ppq = q / ppq, p / ppq
    inv2p, inv2q = 0.5 / p, 0.5 / q
    s = {"inv2p": inv2p, "inv2q": inv2q, "inv2pq": 0.5 / ppq,
         "inv2p_alpha_over_p": inv2p * (rho / p),
         "inv2q_alpha_over_q": inv2q * (rho / q),
         "theta": math.sqrt(rho), "pair": b.U * k.U}
    for d, ax in enumerate("xyz"):
        s[f"PA_{ax}"] = b.X[d]
        s[f"QC_{ax}"] = k.X[d]
        s[f"WP_{ax}"] = 0.0 - q_ppq * PQ[d]
        s[f"WQ_{ax}"] = p_ppq * PQ[d]
        s[f"PQ_{ax}"] = PQ[d]
        s[f"AB_{ax}"] = AB[d]
        s[f"CD_{ax}"] = CD[d]
    return s


def _fspec(cls) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(cls), Target.F)


@cache
def _vrr_kernel(cls):
    return R.build_vrr_kernel(_fspec(cls))


@cache
def _hrr_kernel(cls):
    return R.build_hrr_kernel(_fspec(cls))


def host(cls, bra: KF.PairRec, ket: KF.PairRec) -> np.ndarray:
    """The host path (module docstring): #279's two kernels, the fold."""
    n = len(R.boundary_order(cls))
    c = np.zeros(n)
    for kp in ket.prims:  # ket outer, as the fused nest
        for bp in bra.prims:
            s = _scalars(bp, kp, bra.sep, ket.sep)
            v = interpret(_vrr_kernel(cls), boys=boys,
                          **{x: s[x] for x in R.VRR_SCALARS},
                          out=np.zeros(n))["out"]
            c = c + v
    s = _scalars(bra.prims[0], ket.prims[0], bra.sep, ket.sep)
    return interpret(_hrr_kernel(cls), **{x: s[x] for x in R.HRR_SCALARS},
                     boundary=c, out=np.zeros(len(hrr.targets(cls))))["out"]


def fused(cls, bras, kets, *, tau=0.0, maxD=1.0, threads=2, g0=None):
    """Interpret the fused kernel over a launch; returns `(g, interpreter)`."""
    k = KF.build_f_kernel(cls)
    pb, pk = KF.pack(bras), KF.pack(kets)
    n_eri = len(hrr.targets(cls))
    every = {
        "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree, "k_cd": pk.degree,
        "tau": float(tau), "maxD": float(maxD),
        "bra_prim": np.array(pb.prim), "AB": np.array(pb.sep),
        "G_ab": np.array(pb.G),
        "ket_prim": np.array(pk.prim), "CD": np.array(pk.sep),
        "G_cd": np.array(pk.G),
        KF.OUT: (np.zeros(pb.n * pk.n * n_eri) if g0 is None else g0.copy()),
    }
    names = KF.params_of(cls)
    grid, block = KF.launch_shape(pb.n, threads)
    it = Interpreter(k, launch=Launch(grid, block, 32), boys=boys)
    out = it.run(**{n: every[n] for n in names})[KF.OUT]
    return out.reshape(pb.n, pk.n, n_eri), it


def mmd(cls, shells) -> tuple[np.ndarray, np.ndarray]:
    blk, scale = eri_block_and_scale((shells[0], shells[1]),
                                     (shells[2], shells[3]), pq=pq)
    return np.array(blk).ravel(), np.array(scale).ravel()


CENTRES = ((0.0, 0.0, 0.0), (0.4, -0.3, 0.5), (-0.2, 0.6, 0.1),
           (0.7, 0.2, -0.4))
EXPS = ((1.3, 0.35), (0.7, 2.2), (0.9, 0.3), (1.6, 0.5))
COEFS = ((0.6, 0.45), (0.8, -0.3), (0.5, 0.7), (0.9, 0.25))

# The same bound and reason as test_hgp_recurrence.py's TOL (two
# exact-arithmetic-equal recurrences in different operation orders, the
# HRR's cancellation unseen by the |.| scale): 1e-13 of scale.
TOL = 1e-13


def _shells(cls, centres=CENTRES):
    return [Shell(am, c, e, k) for am, c, e, k in
            zip(cls, centres, EXPS, COEFS, strict=True)]


@pytest.mark.parametrize("cls", LADDER)
def test_one_contracted_quartet_is_the_host_path_bitwise(cls) -> None:
    """All 21 classes on one contracted quartet of 2-primitive shells
    (`K_AB = K_CD = 4`, 16 primitive quartets): `==` the host path, and
    against `ao/eri.py`'s MMD integrals to rounding."""
    sh = _shells(cls)
    bra, ket = records(sh[0], sh[1]), records(sh[2], sh[3])
    g, _ = fused(cls, [bra], [ket], threads=1)
    want = host(cls, bra, ket)
    assert np.array_equal(g[0, 0], want)
    ref, scale = mmd(cls, sh)
    err = np.abs(g[0, 0] - ref)
    assert np.all(err <= TOL * scale), float(np.max(err / scale))


def _moved(dx):
    return tuple(tuple(x + dx[i % 3] * (j + 1) for i, x in enumerate(c))
                 for j, c in enumerate(CENTRES))


@pytest.mark.parametrize("cls", [_speed((0, 1, 1, 1)), _speed((1, 2, 1, 2))])
def test_a_multi_quartet_ket_loop_is_bitwise_and_the_screen_skips(cls) -> None:
    """Two bra pairs (two blocks) by three ket pairs strided over two
    threads. The MIDDLE ket is screened (`G_cd = 0 < tau`): its `g` row is
    untouched and the ket after it is still computed -- a break would have
    left it -- while the ket loop is entered for every ket and the primitive
    loops only for the unscreened ones (`fock/skip-not-break`, behaviour)."""
    geoms = [CENTRES, _moved((0.3, -0.2, 0.1)), _moved((-0.5, 0.4, 0.2))]
    shs = [_shells(cls, g) for g in geoms]
    bras = [records(shs[0][0], shs[0][1]), records(shs[1][0], shs[1][1])]
    kets = [records(shs[i][2], shs[i][3], G=(0.0 if i == 1 else 1.0))
            for i in range(3)]
    g0 = np.full(len(bras) * len(kets) * len(hrr.targets(cls)), -7.0)
    g, it = fused(cls, bras, kets, tau=1e-12, threads=2, g0=g0)
    for ib, b in enumerate(bras):
        for ik, k in enumerate(kets):
            if ik == 1:
                assert np.all(g[ib, ik] == -7.0)  # skipped: never written
            else:
                assert np.array_equal(g[ib, ik], host(cls, b, k))
    assert it.iterations[KF.KET] == len(bras) * len(kets)
    assert it.iterations[KF.KCD] == len(bras) * 2 * len(kets[0].prims)


# ---------------------------------------------------------------------------
# The record is consumable from launch_f's pair stream (#285)
# ---------------------------------------------------------------------------

BOUND = partial(primitive_schwarz_bounds, one_centre=one_centre)


def test_a_launch_from_the_pair_stream_runs_against_mmd() -> None:
    """H2O/STO-3G at `tau = 0`: one launch of `f_launches`' (sp|sp) class
    (`K_AB = K_CD = 9`), each side packed from its batch's `FPair.prims`
    indices -- multi-quartet, both sides, real shells -- bitwise against the
    host path and to rounding against `ao/eri.py`."""
    fx = manifest.load(manifest.FIXTURES / "h2o_sto3g_j.json")
    basis, D = ob.from_fixture(fx), np.array(fx.density)
    plan = f_launches(basis, D, bound=BOUND, tau=0.0, n=1)
    launch = next(L for L in plan if L.spec.am == (0, 1, 0, 1))
    cls = launch.spec.am
    sh = basis.shells

    def side(batch):
        return [records(sh[p.a], sh[p.b], G=p.G, keep=p.prims)
                for p in batch.pairs]

    bb, kb = plan.batches[launch.bra], plan.batches[launch.ket]
    bras, kets = side(bb), side(kb)
    assert KF.pack(bras).degree == bb.degree == 9
    g, _ = fused(cls, bras, kets, threads=4)
    for ib, (b, pb) in enumerate(zip(bras, bb.pairs, strict=True)):
        for ik, (k, pk) in enumerate(zip(kets, kb.pairs, strict=True)):
            assert np.array_equal(g[ib, ik], host(cls, b, k))
            ref, scale = mmd(cls, [sh[pb.a], sh[pb.b], sh[pk.a], sh[pk.b]])
            assert np.all(np.abs(g[ib, ik] - ref) <= TOL * scale)
