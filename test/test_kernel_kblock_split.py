"""`kernel_kblock_split`: the block kernel split per element, interpreted (#548).

The witness is `kernel_kblock` itself. Each kernel of a split set keeps the
block kernel's per-slot fold (its stage-1 and stage-2 term lists filtered by
`KHooks.keep`, never refolded), so

- the set, launched into one zeroed `K` on the block kernel's arguments,
  is the block kernel's `K` BIT FOR BIT, for each granularity, one or
  several bras, one lane or four;
- a stack of `R` densities is `R` one-density builds, bit for bit each;
- the whole plan, `launch_kblock`'s, staged as test/kblock_driver.py stages
  it for the card and run through the split sets, is the block kernel's
  plan bit for bit, so `walk_kb`'s `K` is unchanged to the block kernel's
  own bar (on two H2 with both `break`s firing, and on H2O/STO-3G).

The declared shared footprint is pinned to #548's table. Every class is
`(dd|dd)` or below; the `(dd|dd)` cases are `slow` (above `FAST_L_MAX`).
Fast tier otherwise: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import numpy as np
import pytest
from kblock_driver import class_args, kb_driver, kb_stage
from test_k_ondevice import ORACLE_TOL
from test_kernel_k import _dcd, _pairs, k_args
from test_launch_k import _case as _launch_case
from tiers import by_l

from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_k import k_class_spec
from felizjueves.eri.k_exchange.kernel_kblock import PARAMS as KB_PARAMS
from felizjueves.eri.k_exchange.kernel_kblock import build_kblock_kernel
from felizjueves.eri.k_exchange.kernel_kblock_split import (
    GRANULARITIES,
    PARAMS,
    SHARED_LIMIT,
    accumulators_per_density,
    build_kblock_split_kernels,
    launch_shape,
    max_stack,
    shared_bytes,
    shared_doubles,
    split_name,
)
from felizjueves.eri.k_exchange.launch_kblock import kb_launches, walk_kb
from felizjueves.ir import AtomicAdd, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]
DDDD = (2, 2, 2, 2)

CLASSES = [by_l(am, sum(am), id="({}{}|{}{})".format(*("spd"[x] for x in am)))
           for am in [(1, 0, 1, 0), (1, 1, 1, 1), (0, 1, 2, 0), DDDD]]


def _na_nb(am: Am) -> tuple[int, int]:
    return (am[0] + 1) * (am[0] + 2) // 2, (am[2] + 1) * (am[2] + 2) // 2


def _run(kernels, am: Am, bra, kets, dcds, *, lanes: int = 1) -> np.ndarray:
    """Every kernel of `kernels`, one block over every bra and ket, into one
    zeroed `K[R, n, n]` at AO offsets `(0, n_a)`: the set's `K`."""
    a = k_args(am, bra, kets, [dcds[0]])
    na, nb = _na_nb(am)
    n = na + nb
    blk = {k: np.array([v], np.int32) for k, v in dict(
        bra0=0, bra1=a["n_bra"], ket0=0, ket1=a["n_ket"], ko_a=0,
        ko_b=na).items()}
    blk["dmax"] = np.array([a["maxD"]])
    stack = np.stack([k_args(am, bra, kets, [d])["Dcd"] for d in dcds])
    K = np.zeros((len(dcds), n, n))
    base = {**{k: v for k, v in a.items() if k not in ("K", "maxD", "Dcd")},
            **blk, "n_blk": 1, "n_ao": n}
    grid, block = launch_shape(1, lanes)
    for kern in kernels:
        args = {**base, "Dcd": stack if kern.name.startswith("kbs_")
                else stack[0], "K": K if kern.name.startswith("kbs_")
                else K[0]}
        out = Interpreter(kern, launch=Launch(grid, block, lanes),
                          boys=boys).run(**args)["K"]
        K = out if out.ndim == 3 else out[None]
    return K


@pytest.mark.parametrize("granularity", GRANULARITIES)
@pytest.mark.parametrize("am", CLASSES)
def test_the_set_is_the_block_kernel_bitwise(am: Am, granularity: str) -> None:
    """A contracted bra and ket (four records each): the split set's `K` is
    the block kernel's, every bit, and nonzero."""
    bra, ket = _pairs(am)
    dcd = _dcd(am[1], am[3], shift=-0.1)
    want = _run([build_kblock_kernel(am)], am, bra, [(ket, 0)], [dcd])
    got = _run(build_kblock_split_kernels(am, granularity), am, bra,
               [(ket, 0)], [dcd])
    assert np.any(want != 0.0)
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("granularity", ["column", "element"])
def test_more_lanes_still_bitwise(granularity: str) -> None:
    """Four lanes: the butterfly reduces each slot alone, as the block
    kernel's does, so the split set is still its `K` bit for bit."""
    am = (1, 1, 1, 0)
    bra, ket = _pairs(am)
    dcd = _dcd(1, 0)
    want = _run([build_kblock_kernel(am)], am, bra, [(ket, 0)], [dcd], lanes=4)
    got = _run(build_kblock_split_kernels(am, granularity), am, bra,
               [(ket, 0)], [dcd], lanes=4)
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("granularity", GRANULARITIES)
@pytest.mark.parametrize("am", [(1, 1, 1, 1), pytest.param(
    DDDD, marks=pytest.mark.slow, id="(dd|dd)")])
def test_a_stack_is_r_one_density_builds(am: Am, granularity: str) -> None:
    """`R = 3` densities in one sweep: density `r`'s `K` is the `R = 1` set
    on `Dcd[r]`, bit for bit."""
    bra, ket = _pairs(am)
    dcds = [_dcd(am[1], am[3], shift=s) for s in (-0.1, 0.0, 0.37)]
    got = _run(build_kblock_split_kernels(am, granularity, R=3), am, bra,
               [(ket, 0)], dcds)
    for r, d in enumerate(dcds):
        one = _run(build_kblock_split_kernels(am, granularity), am, bra,
                   [(ket, 0)], [d])
        assert got[r].tolist() == one[0].tolist()
    assert got[0].tolist() != got[2].tolist()


# ---------------------------------------------------------------------------
# The declared footprint and the refusals
# ---------------------------------------------------------------------------


def test_the_shared_table_of_548() -> None:
    """#548's table at `(dd|dd)`: slots per lane and the largest `R` at
    wave32 and wave64, for each granularity."""
    rows = {"block": (36, 4, 1), "column": (6, 26, 10), "element": (1, 157, 61)}
    for g, (acc, r32, r64) in rows.items():
        assert accumulators_per_density(DDDD, g) == acc
        for R in (1, 2, 55):
            assert shared_doubles(DDDD, acc, R) == 35 + acc * R
        assert max_stack(DDDD, g, 32) == r32
        assert max_stack(DDDD, g, 64) == r64
        assert shared_bytes(DDDD, acc, r64, 64) <= SHARED_LIMIT
        assert shared_bytes(DDDD, acc, r64 + 1, 64) > SHARED_LIMIT
    assert shared_bytes(DDDD, 36, 1, 32) == 18_176  # kb_22_22's smem
    # the block kernel cannot carry two densities at wave64; the split
    # clears R = 55 on both widths
    assert max_stack(DDDD, "block", 64) == 1
    assert min(max_stack(DDDD, "element", w) for w in (32, 64)) >= 55


def test_the_refusals() -> None:
    with pytest.raises(ValueError, match="48 KiB") as e:
        build_kblock_split_kernels(DDDD, "block", R=2, lanes=64)
    assert "at most R = 1" in str(e.value)
    with pytest.raises(ValueError, match="one batch"):
        build_kblock_split_kernels(dataclasses.replace(
            k_class_spec(1, 1, 1, 1), n_batches=3), "element")
    with pytest.raises(ValueError, match="levers"):
        build_kblock_split_kernels(Spec(Operator.COULOMB, 0, (0, 0, 0, 0),
                                        Target.K, ("stage-eac",)), "element")
    with pytest.raises(ValueError, match="granularity"):
        build_kblock_split_kernels((1, 0, 1, 0), "row")
    with pytest.raises(ValueError, match="R ="):
        build_kblock_split_kernels((1, 0, 1, 0), "element", R=0)


def test_the_set_shape() -> None:
    am = (2, 1, 1, 2)
    for g, n in (("block", 1), ("column", 3), ("element", 18)):
        ks = build_kblock_split_kernels(am, g, R=2)
        assert len(ks) == n
        acc = accumulators_per_density(am, g)
        for k in ks:
            assert tuple(p.name for p in k.params) == PARAMS == KB_PARAMS
            shp = {p.name: p.shape for p in k.params}
            assert shp["Dcd"][0].value == shp["K"][0].value == 2
            adds = [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
            assert len(adds) == 2 * acc
    assert [k.name for k in build_kblock_split_kernels(am, "column")] == [
        "kbs_21_12_c1_0", "kbs_21_12_c1_1", "kbs_21_12_c1_2"]
    assert split_name(am, "element", 4, (5, 2)) == "kbs_21_12_e4_5_2"


# ---------------------------------------------------------------------------
# The whole plan through the split sets, interpreted
# ---------------------------------------------------------------------------


def _plan_K(plan, drv, staged, build) -> np.ndarray:
    K = np.zeros((drv.n_ao, drv.n_ao))
    for c, cl in enumerate(plan):
        args = class_args(drv, staged, c)
        grid, block = launch_shape(args["n_blk"], 1)
        for kern in build(cl.am):
            out = Interpreter(kern, launch=Launch(grid, block, 1),
                              boys=boys).run(**{**args, "K": K})["K"]
            K = out.reshape(K.shape)
    return K


@pytest.mark.parametrize("case,tau", [("two_h2", 1e-6), pytest.param(
    "h2o_sto3g_j", 0.0, marks=pytest.mark.slow)])
def test_the_plan_through_the_split_is_the_block_kernels(case, tau, tmp_path):
    """`launch_kblock`'s plan, each class's split set in place of its block
    kernel: `K` is the block kernel's plan's bit for bit, and so `walk_kb`'s
    to the block kernel's own bar."""
    basis, D = _launch_case(case)
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=tau, build=build_kblock_kernel)
    w = walk_kb(plan, basis, D, pq=pq)
    drv = kb_driver(plan, basis, "cuda", tmp_path)
    staged = kb_stage(drv, plan, basis, D)
    want = _plan_K(plan, drv, staged, lambda am: [build_kblock_kernel(am)])
    Kw = np.array(w.K)
    for g in ("column", "element"):
        got = _plan_K(plan, drv, staged,
                      partial(build_kblock_split_kernels, granularity=g))
        assert got.tolist() == want.tolist()
    assert float(np.max(np.abs(want - Kw))) <= ORACLE_TOL * float(
        np.max(np.abs(Kw)))
