"""`kernel_kblock_rolled`: `E` from global memory, the contraction rolled (#571).

The witnesses, interpreted, each class planned on its own basis (one
contracted shell per distinct `l` of the class, on its own centre) and
staged as test/kblock_driver.py stages it for the card (`rolled=True`):

- `kernel_kblock`, BIT FOR BIT, every class with every `l <= 2`: the same
  terms in the same order (the module docstring's EXACTNESS), its inlined
  `E` fragment against the host's `launch_kblock.e_table`;
- `launch_kblock.walk_kb`, to `ORACLE_TOL`, every class with an f shell
  (`kernel_kblock` above `(dd|dd)` is the 22 MB text #569 could not compile,
  and interpreting it buys nothing the walk does not);
- the whole plan, as test_kernel_kblock.py holds the block kernel to it: the
  s, s, p basis at one and four lanes, and two H2 screened.

Plus the pieces: `e_pattern` against `e_table`, the tables, the shape and
the refusals, and the text's size (the point of the issue: the contraction's
text no longer grows with the class).

Every class with L <= `tiers.FAST_L_MAX` is fast, above it `slow` (and
`high_l` above `tiers.DEVICE_L_MAX`). Fast tier: no compiler, no card, no
PySCF.
"""

from __future__ import annotations

import dataclasses
import itertools
from functools import cache, partial

import numpy as np
import pytest
from kblock_driver import class_args, kb_driver, kb_stage
from test_k_ondevice import ORACLE_TOL, _sp_basis
from test_kernel_device import BACKEND
from test_launch_k import _case as _launch_case
from tiers import FAST_L_MAX, by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_k import k_class_spec
from felizjueves.eri.k_exchange.kernel_kblock import (
    build_kblock_kernel,
    launch_shape,
    shared_doubles,
)
from felizjueves.eri.k_exchange.kernel_kblock_rolled import (
    PARAMS,
    TABLES,
    build_kblock_rolled_kernel,
    e_pattern,
    hpq_table,
    kblock_rolled_name,
    tables,
)
from felizjueves.eri.k_exchange.launch_kblock import e_table, kb_launches, walk_kb
from felizjueves.ir import AtomicAdd, For, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd.hermite import e_block, hermite_order
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

SPD = list(itertools.product(range(3), repeat=4))
WITH_F = [am for am in itertools.product(range(4), repeat=4) if 3 in am]


def _id(am: Am) -> str:
    return "({}{}|{}{})".format(*("spdf"[x] for x in am))


CENTRES = ((0.0, 0.0, 0.0), (0.5, -0.6, 0.3), (-0.4, 0.7, 0.6),
           (0.6, 0.5, -0.7))


@cache
def _class_case(am: Am):
    """One contracted shell per distinct `l` of `am`, two primitives each at
    L <= FAST_L_MAX (four records per shell pair) and one above it (the
    interpreter's cost), and a symmetric density; the plan cut to `am`."""
    two = sum(am) <= FAST_L_MAX
    shells = tuple(
        Shell(l, CENTRES[l], (1.1 - 0.1 * l, 0.4) if two else (0.9 - 0.1 * l,),
              (0.6, 0.5) if two else (1.0,))
        for l in sorted(set(am)))  # noqa: E741
    basis = Basis(shells)
    n = basis.n_ao
    D = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq), tau=0.0,
                       build=build_kblock_kernel if max(am) <= 2 else None)
    plan = dataclasses.replace(plan, classes=tuple(c for c in plan
                                                   if c.am == am))
    assert len(plan) == 1
    return basis, D, plan


def _interp(plan, basis, D, tmp_path, *, rolled: bool, lanes: int = 1
            ) -> np.ndarray:
    drv = kb_driver(plan, basis, "cuda", tmp_path / ("r" if rolled else "b"),
                    rolled=rolled, with_k=False)
    staged = kb_stage(drv, plan, basis, D)
    K = np.zeros((drv.n_ao, drv.n_ao))
    for c in range(len(plan)):
        args = class_args(drv, staged, c)
        grid, block = launch_shape(args["n_blk"], lanes)
        K = Interpreter(drv.classes[c].kb, launch=Launch(grid, block, lanes),
                        boys=boys).run(**args, K=K)["K"]
    return K


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am)) for am in SPD])
def test_the_rolled_kernel_is_the_block_kernel_bitwise(am: Am, tmp_path):
    basis, D, plan = _class_case(am)
    got = _interp(plan, basis, D, tmp_path, rolled=True)
    want = _interp(plan, basis, D, tmp_path, rolled=False)
    assert np.any(got != 0.0)
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am)) for am in WITH_F])
def test_the_rolled_kernel_is_the_walk_with_an_f_shell(am: Am, tmp_path):
    basis, D, plan = _class_case(am)
    got = _interp(plan, basis, D, tmp_path, rolled=True)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    big = float(np.max(np.abs(Kw)))
    assert big > 0.0 and float(np.max(np.abs(got - Kw))) <= ORACLE_TOL * big


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_interpreted_plan_is_the_walk_with_p_shells(lanes, tmp_path):
    """test_kernel_kblock.py's s, s, p basis, every class: the rolled plan
    against the walk, and, at one lane, the block kernel's bitwise."""
    basis, D = _sp_basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq), tau=0.0,
                       build=build_kblock_kernel)
    assert any(len(c.blocks) > 1 for c in plan)
    K = _interp(plan, basis, D, tmp_path, rolled=True, lanes=lanes)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    big = float(np.max(np.abs(Kw)))
    assert big > 0.0 and float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * big
    if lanes == 1:
        assert K.tobytes() == _interp(plan, basis, D, tmp_path,
                                      rolled=False).tobytes()


def test_the_interpreted_plan_is_the_walk_screened(tmp_path):
    """Two H2 6 bohr apart at a tau where both breaks fire."""
    basis, D = _launch_case("two_h2")
    tau = 1e-6
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=tau, build=build_kblock_kernel)
    w = walk_kb(plan, basis, D, pq=pq)
    assert w.n_skipped > 0
    K = _interp(plan, basis, D, tmp_path, rolled=True)
    Kw = np.array(w.K)
    assert float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * float(
        np.max(np.abs(Kw)))
    assert K.tobytes() == _interp(plan, basis, D, tmp_path,
                                  rolled=False).tobytes()


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l1, l2", [(a, b) for a in range(4) for b in range(4)])
def test_e_table_is_e_blocks_nonzero_entries_in_pattern_order(l1, l2):
    from felizjueves.ao.shell import cartesians

    P, A, B, zeta = (0.1, -0.2, 0.3), (0.4, 0.0, -0.5), (-0.3, 0.6, 0.2), 1.3
    ptr, cols = e_pattern(l1, l2)
    got, signed = (e_table(l1, l2, zeta, P, A, B, sign=s) for s in (False,
                                                                   True))
    assert len(got) == len(cols) == ptr[-1]
    ts = hermite_order(l1 + l2)
    assert ts == hermite_targets(l1 + l2)
    rows = [(x, y) for x in cartesians(l1) for y in cartesians(l2)]
    assert len(ptr) == len(rows) + 1
    for r, (x, y) in enumerate(rows):
        full = e_block(x, y, zeta, P, A, B)
        for e in range(ptr[r], ptr[r + 1]):
            t = ts[cols[e]]
            assert got[e] == full[cols[e]]
            assert signed[e] == (-1.0) ** sum(t) * got[e]
        dropped = set(range(len(ts))) - set(cols[ptr[r]:ptr[r + 1]])
        assert all(full[i] == 0.0 for i in dropped)


def test_the_tables():
    am = (2, 1, 1, 2)
    t = tables(am)
    assert tuple(t) == TABLES
    lp, lq = 3, 3
    hp, hq, hr = (hermite_targets(x) for x in (lp, lq, lp + lq))
    h = hpq_table(am)
    assert len(t["hpq"]) == len(hp) * len(hq)
    for i, p in enumerate(hp):
        for j, q in enumerate(hq):
            assert hr[h[i][j]] == tuple(a + b for a, b in zip(p, q,
                                                              strict=True))
            assert t["hpq"][i * len(hq) + j] == h[i][j]


def test_the_kernel_shape():
    k = build_kblock_rolled_kernel((2, 1, 1, 2))
    assert k.name == kblock_rolled_name((2, 1, 1, 2)) == "kbr_21_12"
    assert tuple(p.name for p in k.params) == PARAMS
    stmts = list(walk(k.body))
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert len(adds) == 1 and adds[0].dst.array == "K"  # rolled over (a, b)
    emit, arch, _ = BACKEND["cuda"]
    text = emit(k, arch)
    assert all(f"{n}[" in text for n in ("Eac", "Ebd", *TABLES))
    assert not {"XPA", "XPC", "XQB", "XQD"} & {p.name for p in k.params}
    # zeroing; bra, ket; (b, d, p, e) and (a, c, e); the write-back (a, b)
    assert sum(isinstance(s, For) for s in stmts) == 12
    ksh = next(s for s in stmts if getattr(s, "name", None) == "Ksh")
    assert ksh.shape[0].value == shared_doubles((2, 1, 1, 2))


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am)) for am in
                                ((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3))])
def test_the_text_does_not_grow_with_the_contraction(am: Am):
    """The text past the recurrence is a constant number of loops: `(ff|ff)`
    under 2 MB (#571's checkpoint; `kb_33_33` is 22 MB), and every unit at
    most a fixed multiple of its `[r]` count."""
    emit, arch, _ = BACKEND["cuda"]
    text = emit(build_kblock_rolled_kernel(am), arch)
    assert len(text) < 2_000_000
    n_r = len(hermite_targets(sum(am)))
    assert len(text) < 400 * n_r + 20_000


def test_the_refusals():
    with pytest.raises(ValueError, match="one batch"):
        build_kblock_rolled_kernel(dataclasses.replace(
            k_class_spec(1, 1, 1, 1), n_batches=3))
    with pytest.raises(ValueError, match="levers"):
        build_kblock_rolled_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0),
                                        Target.K, ("stage-eac",)))
    with pytest.raises(ValueError, match="not a Target.K"):
        build_kblock_rolled_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0),
                                        Target.J))
