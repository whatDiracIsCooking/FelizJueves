"""`kernel_kblock`: the exchange kernel on opt-UM's mapping, interpreted (#481).

The witness is `kernel_k` itself: the same statements per (bra, ket) pair
(`kernel_k.ket_body`, the default hooks) in another loop nest. So

- with ONE LANE and ONE BRA RECORD in a block, the block kernel's `K_AB` is
  `kernel_k`'s row of that record BIT FOR BIT -- the kernel added into a
  zeroed `K` at the block's AO offsets, every other element untouched;
- with several bra records it is the rows' sum, folded left through one
  accumulator: a reassociation, so to rounding;
- with more lanes, the butterfly reassociates the per-lane partial sums.

Then the whole plan: `launch_kblock`'s plan staged as test/kblock_driver.py
stages it for the card, each class's kernel interpreted once over all its
blocks into one `K`, against `launch_kblock.walk_kb` -- at tau = 0 on a
synthetic s, s, p basis (several `a` and `b` shells per class) and screened
on two H2, where both `break`s fire.

Every class with L <= `tiers.FAST_L_MAX` is fast; the classes above it
through `(dd|dd)` are `slow`. Fast tier: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
import itertools
from functools import partial

import numpy as np
import pytest
from kblock_driver import class_args, kb_driver, kb_stage
from test_k_ondevice import ORACLE_TOL, _sp_basis
from test_kernel_k import _dcd, _pairs, k_args, run_k
from test_launch_k import _case as _launch_case
from tiers import by_l

from felizjueves.eri.k_exchange import kernel_k
from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_k import k_class_spec
from felizjueves.eri.k_exchange.kernel_kblock import (
    PARAMS,
    build_kblock_kernel,
    kblock_name,
    launch_shape,
    shared_bytes,
    shared_doubles,
)
from felizjueves.eri.k_exchange.launch_kblock import kb_launches, walk_kb
from felizjueves.ir import Assign, AtomicAdd, Index, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

ALL = [am for am in itertools.product(range(3), repeat=4)]
CLASSES = [by_l(am, sum(am), id="({}{}|{}{})".format(*("spd"[x] for x in am)))
           for am in ALL]


def _na_nb(am: Am) -> tuple[int, int]:
    return (am[0] + 1) * (am[0] + 2) // 2, (am[2] + 1) * (am[2] + 2) // 2


def run_kb(am: Am, bra, kets, dcd_rows, *, lanes: int = 1, tau: float = 0.0,
           maxD: float = 1.0, Ip=None, Iq=None, n_bra: int | None = None
           ) -> np.ndarray:
    """One block over bras `[0, n_bra)` and every ket, into `K` at AO
    offsets `(0, n_a)` of an `n_a + n_b` square: the block kernel's `K`."""
    a = k_args(am, bra, kets, dcd_rows, tau=tau, maxD=maxD, Ip=Ip, Iq=Iq)
    na, nb = _na_nb(am)
    n = na + nb
    blk = {k: np.array([v], np.int32) for k, v in dict(
        bra0=0, bra1=a["n_bra"] if n_bra is None else n_bra, ket0=0,
        ket1=a["n_ket"], ko_a=0, ko_b=na).items()}
    blk["dmax"] = np.array([a["maxD"]])  # #484: the block's own factor
    args = {**{k: v for k, v in a.items() if k not in ("K", "maxD")}, **blk,
            "n_blk": 1, "n_ao": n, "K": np.zeros((n, n))}
    grid, block = launch_shape(1, lanes)
    return Interpreter(build_kblock_kernel(am),
                       launch=Launch(grid, block, lanes),
                       boys=boys).run(**args)["K"]


def _block(am: Am, K: np.ndarray) -> np.ndarray:
    na, nb = _na_nb(am)
    out = K[:na, na:na + nb]
    rest = K.copy()
    rest[:na, na:na + nb] = 0.0
    assert not np.any(rest), "an element outside the block was written"
    return out.ravel()


@pytest.mark.parametrize("am", CLASSES)
def test_one_bra_record_is_kernel_ks_row_bitwise(am: Am) -> None:
    """An uncontracted bra (one record), a contracted ket (four), one lane:
    the block is `kernel_k`'s row, bit for bit, added at its offsets."""
    bra, ket = _pairs(am, contracted=False)
    assert len(bra.recs) == 1 and len(ket.recs) == 4
    dcd = _dcd(am[1], am[3])
    row, _ = run_k(am, bra, [(ket, 0)], [dcd])
    got = _block(am, run_kb(am, bra, [(ket, 0)], [dcd]))
    assert got.tolist() == row[0].tolist()
    assert np.any(got != 0.0)


@pytest.mark.parametrize("am", CLASSES)
def test_several_bra_records_sum_kernel_ks_rows(am: Am) -> None:
    """A contracted bra (four records): the block is the sum of `kernel_k`'s
    four rows, to rounding (one accumulator across the bras)."""
    bra, ket = _pairs(am)
    dcd = _dcd(am[1], am[3], shift=-0.1)
    rows, _ = run_k(am, bra, [(ket, 0)], [dcd])
    got = _block(am, run_kb(am, bra, [(ket, 0)], [dcd]))
    want = rows.sum(axis=0)
    assert np.max(np.abs(got - want)) <= 1e-14 * np.max(np.abs(rows).sum(0))


@pytest.mark.parametrize("lanes", [4, 32])
def test_more_lanes_reassociate_only(lanes: int) -> None:
    am = (1, 0, 1, 1)
    bra, ket = _pairs(am)
    dcd = _dcd(0, 1)
    a = _block(am, run_kb(am, bra, [(ket, 0)], [dcd], lanes=lanes))
    b = _block(am, run_kb(am, bra, [(ket, 0)], [dcd]))
    assert np.max(np.abs(a - b)) <= 1e-15 * np.max(np.abs(b))
    one = dataclasses.replace(ket, recs=ket.recs[:1])
    a = _block(am, run_kb(am, bra, [(one, 0)], [dcd], lanes=lanes))
    b = _block(am, run_kb(am, bra, [(one, 0)], [dcd]))
    assert a.tolist() == b.tolist()  # the other lanes add +0.0, exactly


def test_the_bra_break_ends_the_block() -> None:
    """Bras sorted by Ip: once `Ib * Iq[ket0] * maxD < tau`, the rest of the
    block's bras are not contracted -- the same `K` as a block of the bras
    before it; and the per-pair break inside the ket loop is `kernel_k`'s."""
    am = (1, 0, 1, 0)
    bra, ket = _pairs(am)
    dcd = _dcd(0, 0)
    Ip, Iq = [4.0, 3.0, 1e-9, 1e-10], [2.0, 1.5, 1.0, 0.5]
    got = run_kb(am, bra, [(ket, 0)], [dcd], tau=1e-6, Ip=Ip, Iq=Iq)
    want = run_kb(am, bra, [(ket, 0)], [dcd], Ip=Ip, Iq=Iq, n_bra=2)
    assert got.tolist() == want.tolist()
    # a ket-level skip: Ib Iq maxD < tau for the last ket only
    got = run_kb(am, bra, [(ket, 0)], [dcd], tau=1.0, Ip=[4.0] * 4,
                 Iq=[2.0, 1.5, 1.0, 0.2])
    row, _ = run_k(am, bra, [(ket, 0)], [dcd], tau=1.0, Ip=[4.0] * 4,
                   Iq=[2.0, 1.5, 1.0, 0.2])
    full, _ = run_k(am, bra, [(ket, 0)], [dcd], Ip=[4.0] * 4)
    assert not np.allclose(row, full)  # the ket screen fired
    assert np.max(np.abs(_block(am, got) - row.sum(0))) <= 1e-14 * np.max(
        np.abs(row).sum(0))


def test_the_kernel_shape() -> None:
    k = build_kblock_kernel((2, 1, 1, 2))
    assert k.name == kblock_name((2, 1, 1, 2)) == "kb_21_12"
    assert tuple(p.name for p in k.params) == PARAMS
    stmts = list(walk(k.body))
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 * 3 and all(s.dst.array == "K" for s in adds)
    assert not any(isinstance(s, Assign) and isinstance(s.dst, Index)
                   and s.dst.array == "K" for s in stmts)
    for am in ALL:
        assert shared_doubles(am) == kernel_k.shared_doubles(am)
        assert shared_bytes(am, 32) == kernel_k.shared_bytes(am, 32)
    assert launch_shape(7, 32) == (7, (1, 32))
    with pytest.raises(ValueError):
        launch_shape(0, 32)


def test_the_refusals() -> None:
    with pytest.raises(ValueError, match="one batch"):
        build_kblock_kernel(dataclasses.replace(k_class_spec(1, 1, 1, 1),
                                                n_batches=3))
    with pytest.raises(ValueError, match="levers"):
        build_kblock_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.K,
                                 ("stage-eac",)))
    with pytest.raises(ValueError, match="not a Target.K"):
        build_kblock_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target.J))
    with pytest.raises(TypeError):
        build_kblock_kernel((0, 0, 0))


# ---------------------------------------------------------------------------
# The whole plan, staged as for the card, interpreted
# ---------------------------------------------------------------------------


def _interp_K(plan, drv, staged, *, lanes: int) -> np.ndarray:
    K = np.zeros((drv.n_ao, drv.n_ao))
    for c, cl in enumerate(plan):
        args = class_args(drv, staged, c)
        grid, block = launch_shape(args["n_blk"], lanes)
        K = Interpreter(cl.kernel, launch=Launch(grid, block, lanes),
                        boys=boys).run(**args, K=K)["K"]
    return K


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_interpreted_plan_is_the_walk_with_p_shells(lanes, tmp_path):
    basis, D = _sp_basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq), tau=0.0,
                       build=build_kblock_kernel)
    assert any(len(c.blocks) > 1 for c in plan)
    assert max(c.L for c in plan) == 4
    drv = kb_driver(plan, basis, "cuda", tmp_path)
    K = _interp_K(plan, drv, kb_stage(drv, plan, basis, D), lanes=lanes)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    big = float(np.max(np.abs(Kw)))
    assert big > 0.0 and float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * big


def test_the_interpreted_plan_is_the_walk_screened(tmp_path):
    """Two H2 6 bohr apart, the hilbert density, a tau at which both breaks
    fire: the interpreted plan against the walk over the same plan."""
    basis, D = _launch_case("two_h2")
    tau = 1e-6
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=tau, build=build_kblock_kernel)
    w = walk_kb(plan, basis, D, pq=pq)
    assert w.n_skipped > 0
    (c,) = plan
    assert any((r.G * c.kets[b.ket0].G) * plan.dm < tau
               for b in c.blocks for r in c.bras[b.bra0:b.bra1])
    drv = kb_driver(plan, basis, "hip", tmp_path)
    K = _interp_K(plan, drv, kb_stage(drv, plan, basis, D), lanes=1)
    Kw = np.array(w.K)
    assert float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * float(
        np.max(np.abs(Kw)))


# ---------------------------------------------------------------------------
# The per-block density factor (#484)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("am", [(0, 0, 0, 0), (1, 0, 1, 1), (1, 1, 0, 1)])
@pytest.mark.parametrize("tau", [0.0, 1.0])
def test_dmax_max_d_is_the_scalar_screen_bitwise(am: Am, tau: float) -> None:
    """`dmax[blk] = maxD`: the block is `kernel_k`'s row under the scalar
    `maxD`, bit for bit, unscreened and with the ket screen firing."""
    bra, ket = _pairs(am, contracted=False)
    dcd = _dcd(am[1], am[3])
    Ip, Iq, maxD = [4.0], [2.0, 1.5, 1.0, 0.2], 0.7
    row, _ = run_k(am, bra, [(ket, 0)], [dcd], tau=tau, maxD=maxD, Ip=Ip,
                   Iq=Iq)
    got = _block(am, run_kb(am, bra, [(ket, 0)], [dcd], tau=tau, maxD=maxD,
                            Ip=Ip, Iq=Iq))
    assert got.tolist() == row[0].tolist()
    if tau > 0.0:  # the ket screen fired: Ib Iq maxD = 0.56 < 1
        full, _ = run_k(am, bra, [(ket, 0)], [dcd], maxD=maxD, Ip=Ip, Iq=Iq)
        assert row[0].tolist() != full[0].tolist()


def test_the_block_reads_its_own_factor() -> None:
    """A factor under the scalar's, in one block: the screen tightens to it
    (`Ib Iq dmax` against `tau`) and to nothing else."""
    am = (1, 0, 1, 0)
    bra, ket = _pairs(am, contracted=False)
    dcd = _dcd(0, 0)
    kw = dict(tau=1.0, Ip=[4.0], Iq=[2.0, 1.5, 1.0, 0.2])
    loose = run_kb(am, bra, [(ket, 0)], [dcd], maxD=1.0, **kw)
    tight = run_kb(am, bra, [(ket, 0)], [dcd], maxD=0.2, **kw)
    # 4 Iq 0.2 < 1 for Iq < 1.25: two kets survive, against three
    row2, _ = run_k(am, bra, [(ket, 0)], [dcd], maxD=0.2, **kw)
    assert _block(am, tight).tolist() == row2[0].tolist()
    assert _block(am, tight).tolist() != _block(am, loose).tolist()


def test_the_interpreted_plan_is_the_walk_with_the_decay(tmp_path):
    """The chain of test_ao_k_decay.py, its per-block factors staged as
    `dmax`: the interpreted plan against the walk over the same plan, and
    the factor skipping more than `max|D|`."""
    from test_ao_k_decay import TAU, chain

    from felizjueves.eri.k_exchange.k import density_decay

    basis, D = chain()
    bound = partial(record_bounds, pq=pq)
    dec = density_decay(basis, D, bound=bound, tau=TAU)
    plan = kb_launches(basis, D, bound=bound, tau=TAU,
                       build=build_kblock_kernel, decay=dec)
    (c,) = plan
    assert any(b.dmax < plan.dm for b in c.blocks)
    w = walk_kb(plan, basis, D, pq=pq)
    drv = kb_driver(plan, basis, "cuda", tmp_path)
    staged = kb_stage(drv, plan, basis, D)
    assert class_args(drv, staged, 0)["dmax"].tolist() == [
        b.dmax for b in c.blocks]
    K = _interp_K(plan, drv, staged, lanes=1)
    Kw = np.array(w.K)
    assert float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * float(
        np.max(np.abs(Kw)))
