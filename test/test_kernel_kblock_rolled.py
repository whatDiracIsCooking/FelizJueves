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

And the transposed ket table (#576, `transposed=True`, `kbrt_*`): a
relayout, so the default rolled kernel's `K` BIT FOR BIT on every case
above (every class through `(ff|ff)`, the s, s, p plan at one and four
lanes, two H2 screened), staged as the driver stages it for the card; its
shape; and the default text pinned to its digest before the option
existed.

And the column form (#583, `column=True`, `kbrc_*`): each `K_AB` column
reduced across the warp per (bra, ket strip), so REASSOCIATED, held to
`walk_kb` within `ORACLE_TOL` (never bitwise against `kb` / `kbr`) on
every class through `(ff|ff)` at four lanes, the s, s, p plan and two H2
screened at one and four; its shape (every `WarpReduce` under no
lane-dependent `If`, the per-lane screen the walk's predicate), the
footprint the issue tabulates, and `transposed=True` composing with it.

Every class with L <= `tiers.FAST_L_MAX` is fast, above it `slow` (and
`high_l` above `tiers.DEVICE_L_MAX`). Fast tier: no compiler, no card, no
PySCF.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from functools import cache, partial
from pathlib import Path

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
    shared_bytes,
    shared_doubles,
)
from felizjueves.eri.k_exchange.kernel_kblock_rolled import (
    DENSE_PARAMS,
    KBLK,
    PARAMS,
    TABLES,
    TRANSPOSED,
    build_kblock_dense_kernel,
    build_kblock_rolled_kernel,
    column_block_doubles,
    column_shared_bytes,
    column_shared_doubles,
    e_pattern,
    hpq_table,
    kblock_dense_name,
    kblock_rolled_name,
    tables,
)
from felizjueves.eri.k_exchange.launch_kblock import e_table, kb_launches, walk_kb
from felizjueves.ir import (
    AtomicAdd,
    Call,
    Decl,
    For,
    If,
    Index,
    Space,
    Var,
    WarpReduce,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.ir.passes import transpose
from felizjueves.mmd.hermite import e_block, hermite_order
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

Am = tuple[int, int, int, int]

SPD = list(itertools.product(range(3), repeat=4))
WITH_F = [am for am in itertools.product(range(4), repeat=4) if 3 in am]


def _id(am: Am) -> str:
    return "({}{}|{}{})".format(*("spdfg"[x] for x in am))


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


def _interp(plan, basis, D, tmp_path, *, rolled: bool, lanes: int = 1,
            dense: bool = False, transposed: bool = False,
            column: bool = False) -> np.ndarray:
    drv = kb_driver(plan, basis, "cuda", tmp_path / (
        ("c" if column else "") + (
            "d" if dense else "t" if transposed else "r" if rolled else "b")),
                    rolled=rolled or dense or transposed or column,
                    dense=dense, transposed=transposed, column=column,
                    with_k=False)
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


# every shell pair through (f, f), `slow` above FAST_L_MAX by la + lb
PAIRS = [by_l((a, b), a + b, f"{a}-{b}") for a in range(4) for b in range(4)]


@pytest.mark.parametrize("pair", PAIRS)
def test_e_table_is_e_blocks_nonzero_entries_in_pattern_order(pair):
    from felizjueves.ao.shell import cartesians

    l1, l2 = pair
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


# ---------------------------------------------------------------------------
# The dense-Hermite form (#575): rows rolled, p and q unrolled over dense E
# ---------------------------------------------------------------------------
#
# Interpreted, `kbd` adds exactly `kbr`'s terms in `kbr`'s order plus the
# structural zeros' `[r] * (+-0)`, which leave a sum unchanged unless it is
# itself zero (then at most its sign moves). So it is held to `kb` (and so
# to `kbr`) by VALUE, `==`, every class with every `l <= 2` -- stronger than
# the issue's "to rounding" -- and to the walk within `ORACLE_TOL` with an f
# shell, as `kbr` is.


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am)) for am in SPD])
def test_the_dense_kernel_is_the_block_kernel_by_value(am: Am, tmp_path):
    basis, D, plan = _class_case(am)
    got = _interp(plan, basis, D, tmp_path, rolled=True, dense=True)
    want = _interp(plan, basis, D, tmp_path, rolled=False)
    assert np.any(got != 0.0)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am)) for am in WITH_F])
def test_the_dense_kernel_is_the_walk_with_an_f_shell(am: Am, tmp_path):
    basis, D, plan = _class_case(am)
    got = _interp(plan, basis, D, tmp_path, rolled=True, dense=True)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    big = float(np.max(np.abs(Kw)))
    assert big > 0.0 and float(np.max(np.abs(got - Kw))) <= ORACLE_TOL * big


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_dense_plan_is_the_walk_with_p_shells(lanes, tmp_path):
    basis, D = _sp_basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq), tau=0.0,
                       build=build_kblock_kernel)
    K = _interp(plan, basis, D, tmp_path, rolled=True, dense=True,
                lanes=lanes)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    big = float(np.max(np.abs(Kw)))
    assert big > 0.0 and float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * big
    if lanes == 1:
        assert np.array_equal(K, _interp(plan, basis, D, tmp_path,
                                         rolled=False))


def test_the_dense_plan_is_the_walk_screened(tmp_path):
    basis, D = _launch_case("two_h2")
    tau = 1e-6
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=tau, build=build_kblock_kernel)
    w = walk_kb(plan, basis, D, pq=pq)
    assert w.n_skipped > 0
    K = _interp(plan, basis, D, tmp_path, rolled=True, dense=True)
    Kw = np.array(w.K)
    assert float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * float(
        np.max(np.abs(Kw)))
    assert np.array_equal(K, _interp(plan, basis, D, tmp_path, rolled=False))


@pytest.mark.parametrize("pair", PAIRS)
def test_the_dense_e_table_is_e_block_whole_and_signed(pair):
    from felizjueves.ao.shell import cartesians

    l1, l2 = pair
    P, A, B, zeta = (0.1, -0.2, 0.3), (0.4, 0.0, -0.5), (-0.3, 0.6, 0.2), 1.3
    ts = hermite_order(l1 + l2)
    got, signed = (e_table(l1, l2, zeta, P, A, B, sign=s, dense=True)
                   for s in (False, True))
    rows = [(x, y) for x in cartesians(l1) for y in cartesians(l2)]
    assert len(got) == len(rows) * len(ts)
    for r, (x, y) in enumerate(rows):
        full = e_block(x, y, zeta, P, A, B)
        row = got[r * len(ts):(r + 1) * len(ts)]
        assert row == tuple(full)
        assert signed[r * len(ts):(r + 1) * len(ts)] == tuple(
            -v if sum(t) % 2 else v for t, v in zip(ts, full, strict=True))
    # the nonzero entries, in order, are the rolled kernel's table
    sparse = e_table(l1, l2, zeta, P, A, B)
    ptr, cols = e_pattern(l1, l2)
    assert sparse == tuple(got[r * len(ts) + cols[e]]
                           for r in range(len(rows))
                           for e in range(ptr[r], ptr[r + 1]))


def test_the_dense_kernel_shape():
    am = (2, 1, 1, 2)
    k = build_kblock_dense_kernel(am)
    assert k.name == kblock_dense_name(am) == "kbd_21_12"
    assert tuple(p.name for p in k.params) == DENSE_PARAMS
    assert not set(TABLES) & set(DENSE_PARAMS)
    width = {p.name: p.shape[1].value for p in k.params
             if p.name in ("Eac", "Ebd")}
    n3 = len(hermite_targets(3))
    assert width == {"Eac": 6 * 3 * n3, "Ebd": 3 * 6 * n3}
    stmts = list(walk(k.body))
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert len(adds) == 1 and adds[0].dst.array == "K"
    # zeroing; bra, ket; (b, d) and (a, c); the write-back (a, b): p, q
    # and the entries are unrolled
    assert sum(isinstance(s, For) for s in stmts) == 9
    emit, arch, _ = BACKEND["cuda"]
    text = emit(k, arch)
    assert "R[" not in text and "hpq" not in text
    ksh = next(s for s in stmts if getattr(s, "name", None) == "Ksh")
    assert ksh.shape[0].value == shared_doubles(am)


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am)) for am in
                                ((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3))])
def test_the_dense_text_grows_with_n_p_n_q_not_the_rows(am: Am):
    """Stage 1 is one body of `n_p n_q` multiply-adds, stage 2 one of `n_p`:
    the text past the recurrence is linear in `n_p n_q + n_q + n_p`, never
    in `n_ac n_bd` (`kb_33_33` is 22 MB)."""
    emit, arch, _ = BACKEND["cuda"]
    text = emit(build_kblock_dense_kernel(am), arch)
    rolled = emit(build_kblock_rolled_kernel(am), arch)
    la, lc, lb, ld = am
    n_p, n_q = (len(hermite_targets(x)) for x in (la + lc, lb + ld))
    assert len(text) < len(rolled) + 80 * (n_p * n_q + n_p + n_q)
    assert len(text) < 2_000_000


def test_the_dense_refusals():
    with pytest.raises(ValueError, match="one batch"):
        build_kblock_dense_kernel(dataclasses.replace(
            k_class_spec(1, 1, 1, 1), n_batches=3))
    with pytest.raises(ValueError, match="rolled="):
        kb_driver(_class_case((0, 0, 0, 0))[2], _class_case((0, 0, 0, 0))[0],
                  "cuda", Path("/nonexistent"), dense=True)


# ---------------------------------------------------------------------------
# The transposed ket table (#576): Ebd[n_ebd, n_ket], a relayout
# ---------------------------------------------------------------------------

# the default rolled text, sha256 of its emission at 4429f09 (before #576);
# (dd|dd)'s cuda digest is #574's profiled `kbr_22_22` row's `source_sha256`
DEFAULT_SHA256 = {
    ("cuda", (1, 1, 1, 1)):
        "06237dd4d9576795887c2a3a15797edc537328dca8d146b23f7ed1b4026d5ba0",
    ("cuda", (2, 2, 2, 2)):
        "a36fa7c0d52031bb4119ad3563340b90b9f21ac4ba81cce218497e1fe72e9ca2",
    ("cuda", (3, 3, 3, 3)):
        "708c077db66da2020d0240394813ea0badca3331606fd3ccdf6a8a1a8a33b1a2",
    ("hip", (1, 1, 1, 1)):
        "90ba94ae0150da8fa3254f85b59726d7af0d972c62f7d66f2dc82b95bc0916ad",
    ("hip", (2, 2, 2, 2)):
        "8be6e600a6ee17a0be01ffcbeb586f6e72520d0c4889e50c63f4a90499c1bdd5",
    ("hip", (3, 3, 3, 3)):
        "8ce0737e06b7d30d4db0a9860a68585fbbd54a777882d888e8489bef0b5aff22",
}


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am))
                                for am in SPD + WITH_F])
def test_the_transposed_kernel_is_the_rolled_kernel_bitwise(am: Am, tmp_path):
    basis, D, plan = _class_case(am)
    got = _interp(plan, basis, D, tmp_path, rolled=True, transposed=True)
    want = _interp(plan, basis, D, tmp_path, rolled=True)
    assert np.any(got != 0.0)
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_transposed_plan_is_the_rolled_plan_bitwise(lanes, tmp_path):
    basis, D = _sp_basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq), tau=0.0,
                       build=build_kblock_kernel)
    got = _interp(plan, basis, D, tmp_path, rolled=True, transposed=True,
                  lanes=lanes)
    want = _interp(plan, basis, D, tmp_path, rolled=True, lanes=lanes)
    assert np.any(got != 0.0) and got.tobytes() == want.tobytes()


def test_the_transposed_plan_is_the_rolled_plan_screened(tmp_path):
    basis, D = _launch_case("two_h2")
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=1e-6, build=build_kblock_kernel)
    assert walk_kb(plan, basis, D, pq=pq).n_skipped > 0
    got = _interp(plan, basis, D, tmp_path, rolled=True, transposed=True)
    want = _interp(plan, basis, D, tmp_path, rolled=True)
    assert got.tobytes() == want.tobytes()


def test_the_transposed_kernel_shape():
    am = (2, 1, 1, 2)
    k = build_kblock_rolled_kernel(am, transposed=True)
    d = build_kblock_rolled_kernel(am)
    assert k.name == kblock_rolled_name(am, transposed=True) == "kbrt_21_12"
    assert TRANSPOSED == ("Ebd",)
    assert k == dataclasses.replace(transpose(d, TRANSPOSED), name=k.name)
    shapes = {p.name: p.shape for p in k.params}
    dflt = {p.name: p.shape for p in d.params}
    assert tuple(p.name for p in k.params) == PARAMS
    assert shapes["Ebd"] == dflt["Ebd"][::-1]
    assert {n: v for n, v in shapes.items() if n != "Ebd"} == {
        n: v for n, v in dflt.items() if n != "Ebd"}
    emit, arch, _ = BACKEND["cuda"]
    text = emit(k, arch)
    assert "Ebd[(ie * n_ket + k)]" in text  # a warp's kets, adjacent
    assert "Eac[(bra * " in text  # the bra side untouched: a broadcast


@pytest.mark.parametrize("case", [
    by_l((b, am), sum(am), f"{b}-{_id(am)}") for b, am in DEFAULT_SHA256])
def test_the_default_rolled_text_is_unchanged(case):
    backend, am = case
    emit, arch, _ = BACKEND[backend]
    for k in (build_kblock_rolled_kernel(am),
              build_kblock_rolled_kernel(am, transposed=False),
              build_kblock_rolled_kernel(am, transposed=False,
                                         column=False)):  # #583
        assert hashlib.sha256(emit(k, arch).encode()).hexdigest() == \
            DEFAULT_SHA256[backend, am]


def test_the_transposed_refusals():
    basis, _, plan = _class_case((0, 0, 0, 0))
    for kw in ({"transposed": True}, {"transposed": True, "rolled": True,
                                      "dense": True}):
        with pytest.raises(ValueError, match="transposed="):
            kb_driver(plan, basis, "cuda", Path("/nonexistent"), **kw)


# ---------------------------------------------------------------------------
# The column form (#583): each K_AB column reduced across the warp in-kernel
# ---------------------------------------------------------------------------
#
# Per (bra, ket strip, b) the lane's column `col[a]` is reduced across the
# warp and lane 0 adds it into the block's one `Kblk`: the sum is
# REASSOCIATED against `kb` / `kbr` (one fold per strip, not one per lane
# over the whole block), so it is held to the walk within `ORACLE_TOL`,
# never bitwise. At four lanes, so every reduction sums real lanes and the
# last strip of a block is ragged.

# the issue's table: bytes per 32-lane block after, Ksh + Kblk
COLUMN_BYTES = {(2, 2, 2, 2): 10_784, (3, 3, 3, 3): 24_864,
                (4, 4, 4, 4): 47_880}


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am))
                                for am in SPD + WITH_F])
def test_the_column_kernel_is_the_walk(am: Am, tmp_path):
    basis, D, plan = _class_case(am)
    got = _interp(plan, basis, D, tmp_path, rolled=True, column=True,
                  lanes=4)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    big = float(np.max(np.abs(Kw)))
    assert big > 0.0 and float(np.max(np.abs(got - Kw))) <= ORACLE_TOL * big


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_column_plan_is_the_walk_with_p_shells(lanes, tmp_path):
    basis, D = _sp_basis()
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq), tau=0.0,
                       build=build_kblock_kernel)
    K = _interp(plan, basis, D, tmp_path, rolled=True, column=True,
                lanes=lanes)
    Kw = np.array(walk_kb(plan, basis, D, pq=pq).K)
    big = float(np.max(np.abs(Kw)))
    assert big > 0.0 and float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * big


@pytest.mark.parametrize("lanes", [1, 4])
def test_the_column_plan_is_the_walk_screened(lanes, tmp_path):
    """Two H2 6 bohr apart at a tau where the bra break, the strip break
    and the per-lane zero all fire."""
    basis, D = _launch_case("two_h2")
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=1e-6, build=build_kblock_kernel)
    w = walk_kb(plan, basis, D, pq=pq)
    assert w.n_skipped > 0
    K = _interp(plan, basis, D, tmp_path, rolled=True, column=True,
                lanes=lanes)
    Kw = np.array(w.K)
    assert float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * float(
        np.max(np.abs(Kw)))


def test_the_column_screen_skips_the_walks_pairs(tmp_path):
    """The screened column form is the screened walk, and the unscreened one
    is not: the agreement is the screen's (the per-lane predicate is the
    walk's per-pair test, `test_the_column_kernel_shape`), not an accident
    of pairs too small to see."""
    basis, D = _launch_case("two_h2")
    plan = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                       tau=1e-6, build=build_kblock_kernel)
    w = walk_kb(plan, basis, D, pq=pq)
    assert 0 < w.n_skipped < w.n_pairs
    K = _interp(plan, basis, D, tmp_path, rolled=True, column=True, lanes=4)
    Kw = np.array(w.K)
    assert float(np.max(np.abs(K - Kw))) <= ORACLE_TOL * float(
        np.max(np.abs(Kw)))
    # the unscreened build differs from the screened walk by more than the
    # tolerance, so the agreement above is the screen's, not an accident
    open_ = kb_launches(basis, D, bound=partial(record_bounds, pq=pq),
                        tau=0.0, build=build_kblock_kernel)
    K0 = _interp(open_, basis, D, tmp_path / "open", rolled=True,
                 column=True, lanes=4)
    assert float(np.max(np.abs(K0 - Kw))) > ORACLE_TOL * float(
        np.max(np.abs(Kw)))


def _collectives_under(body, conds=()):
    """Every `WarpReduce` with the `If` conditions it sits under."""
    for s in body:
        if isinstance(s, WarpReduce):
            yield s, conds
        elif isinstance(s, For):
            yield from _collectives_under(s.body, conds)
        elif isinstance(s, If):
            yield from _collectives_under(s.then, (*conds, s.cond))
            yield from _collectives_under(s.orelse, (*conds, s.cond))


def _reads(e) -> set[str]:
    """The names an expression reads (`Var`s, array names, Call kinds)."""
    out: set[str] = set()
    if isinstance(e, Var):
        out.add(e.name)
    elif isinstance(e, Index):
        out.add(e.array)
        for x in e.at:
            out |= _reads(x)
    elif isinstance(e, Call):
        out.add(e.fn.name)
        for x in e.args:
            out |= _reads(x)
    return out


def test_the_column_kernel_shape():
    am = (2, 1, 1, 2)
    k = build_kblock_rolled_kernel(am, column=True)
    assert k.name == kblock_rolled_name(am, column=True) == "kbrc_21_12"
    assert tuple(p.name for p in k.params) == PARAMS
    stmts = list(walk(k.body))
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert len(adds) == 1 and adds[0].dst.array == "K"
    shared = {s.name: s for s in stmts if isinstance(s, Decl)
              and s.space is Space.SHARED}
    assert set(shared) == {"Ksh", KBLK}
    assert shared["Ksh"].shape[0].value == column_shared_doubles(am) == \
        len(hermite_targets(3)) + 6
    assert [d.value for d in shared[KBLK].shape] == [
        column_block_doubles(am)] == [6 * 3]
    # one reduction per (bra, strip, b, a), under no lane-dependent If:
    # only the block guard `blk < n_blk`
    ((red, conds),) = list(_collectives_under(k.body))
    assert red.value.array == "Ksh"
    assert all(_reads(c) <= {"blk", "n_blk", "LT"} for c in conds), conds
    # the per-lane screen is the walk's per-pair predicate, the bound at k
    text = BACKEND["cuda"][0](k, BACKEND["cuda"][1])
    assert "if ((Ib * (Iq[k] * maxD)) < tau) {" in text
    assert "if ((Ib * (Iq[kk] * maxD)) < tau) {" in text


@pytest.mark.parametrize("am", list(COLUMN_BYTES))
def test_the_column_footprint_is_the_issues(am: Am):
    """`(n_p + n_a)` doubles per lane plus one `n_a n_b` copy per block:
    `(gg|gg)` under 48 KiB with no opt-in."""
    assert column_shared_bytes(am, 32) == COLUMN_BYTES[am]
    assert column_shared_bytes(am, 32) < shared_bytes(am, 32)
    assert column_shared_bytes((4, 4, 4, 4), 32) < 48 * 1024 < shared_bytes(
        (4, 4, 4, 4), 32)


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am)) for am in
                                ((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3),
                                 (4, 4, 4, 4))])
def test_the_column_text_stays_small(am: Am):
    emit, arch, _ = BACKEND["cuda"]
    text = emit(build_kblock_rolled_kernel(am, column=True), arch)
    n_r = len(hermite_targets(sum(am)))
    assert len(text) < 400 * n_r + 20_000


@pytest.mark.parametrize("am", [by_l(am, sum(am), _id(am)) for am in
                                ((1, 1, 1, 1), (2, 1, 1, 2), (2, 2, 2, 2))])
def test_the_column_kernel_transposed_is_itself_bitwise(am: Am, tmp_path):
    """`transposed=True` composes: a relayout of the column form, its `K`
    bit for bit."""
    basis, D, plan = _class_case(am)
    kt = build_kblock_rolled_kernel(am, column=True, transposed=True)
    assert kt.name == f"kbrtc_{am[0]}{am[1]}_{am[2]}{am[3]}"
    got = _interp(plan, basis, D, tmp_path, rolled=True, column=True,
                  transposed=True, lanes=4)
    want = _interp(plan, basis, D, tmp_path, rolled=True, column=True,
                   lanes=4)
    assert np.any(got != 0.0) and got.tobytes() == want.tobytes()


def test_the_column_refusals():
    basis, _, plan = _class_case((0, 0, 0, 0))
    for kw in ({"column": True}, {"column": True, "rolled": True,
                                  "dense": True}):
        with pytest.raises(ValueError, match="column="):
            kb_driver(plan, basis, "cuda", Path("/nonexistent"), **kw)
