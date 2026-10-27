"""`kernel_jab`: the bra transform and scatter, `Target.JAB`, interpreted (#154).

THE WITNESS is `eri.j_coulomb.j.jab_from_host`, the host's own bra transform and
scatter through the `jab=` seam (#151), which `coulomb_matrix` is `==` to
itself with (test_ao_j.py). Against it:

- PER CLASS, every bra pair class (ss) through (ff), all 16 ordered
  `(la, lb)`, on the tracked HF/cc-pVTZ basis (the one fixture whose shells
  reach f): for each class its off-diagonal shell pair with the most
  primitive records and, in every `la == lb` class, its DIAGONAL pair with
  the most records (`a == b`, where a naive mirror double-counts). `J_p` is
  a seeded random matrix -- the step is linear in `J_p`, and the question is
  the transform and the scatter, not where `J_p` came from. Fast through
  (pp), `slow` above.
- ONE LAUNCH PER CLASS: every record of every shell pair of the class in one
  launch, rows into one shared `Jp` buffer (`n_rows`), diagonal and
  off-diagonal records mixed -- the shape a whole-class device launch has.
- THE CHAIN: `coulomb_matrix(..., jab=KernelJAB())` against the host path
  (`==`) and against the libcint fixtures at test_ao_j.py's derived
  tolerance, inherited, on H2 and H2O/STO-3G (fast) and H2O/cc-pVDZ (slow).

EXACT, NOT A TOLERANCE. Interpreted with the records in order, the kernel
performs the host's additions in the host's order (kernel_jab's docstring),
so `np.array_equal` is the assertion. The device's fp64-atomic tolerance
belongs where a card is involved (#156/#157), not here.

Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_layering import _graph, _under
from tiers import by_l, split_by_l

from felizjueves.ao.density import epair
from felizjueves.emit import cuda, hip
from felizjueves.eri.j_coulomb import kernel_dq
from felizjueves.eri.j_coulomb.drive import KernelJAB
from felizjueves.eri.j_coulomb.j import (
    coulomb_matrix,
    coulomb_matrix_and_scale,
    jab_from_host,
)
from felizjueves.eri.j_coulomb.kernel_jab import (
    ATOMIC_REASON,
    LOWSTACK_MIN_PAIR_L,
    ZEROED_J,
    build_jab_kernel,
    jab_launch_shape,
    jab_lowstack,
    jab_name,
    jab_spec,
)
from felizjueves.ir import Assign, AtomicAdd, Decl, If, Index, Inline, Note, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.ir.passes import split_axes
from felizjueves.mmd import e_fragment as ef
from felizjueves.mmd.hermite import hermite_order
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

JK_TOL = 1e-12  # vs libcint, times scale_J: test_ao_j.py's, inherited
LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]


def _cls_param(cls):
    marks = () if max(cls) <= 1 else (pytest.mark.slow,)
    return pytest.param(cls, id=f"{cls[0]}{cls[1]}", marks=marks)


@cache
def _case(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return fx, ob.from_fixture(fx)


def _pairs_of(basis, cls):
    """Unique shell pairs `a >= b` of class `cls`, in sweep order."""
    sh = basis.shells
    return [(a, b) for a in range(len(sh)) for b in range(a + 1)
            if (sh[a].l, sh[b].l) == cls]


def _n_rec(basis, a, b) -> int:
    sh = basis.shells
    return len(sh[a].exponents) * len(sh[b].exponents)


def _jps(n_rec: int, n_p: int, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n_rec, n_p)).tolist()


def _zero(n: int) -> list[list[float]]:
    return [[0.0] * n for _ in range(n)]


def _both(basis, a, b, seed, threads=3):
    """J from the host and from the kernel, for shell pair (a, b) alone."""
    sh, off = basis.shells, basis.offsets
    bra = epair(sh[a], sh[b])
    jps = _jps(len(bra.recs), len(hermite_order(bra.la + bra.lb)), seed)
    host, kern = _zero(basis.n_ao), _zero(basis.n_ao)
    jab_from_host(bra, jps, (off[a], off[b]), host)
    k = KernelJAB(threads=threads)
    k(bra, jps, (off[a], off[b]), kern)
    assert k.ran == {(bra.la, bra.lb)}
    return np.array(host), np.array(kern)


def _block(basis, a, b) -> np.ndarray:
    """True on the elements a shell pair's scatter owns: block and mirror."""
    sh, off = basis.shells, basis.offsets
    m = np.zeros((basis.n_ao, basis.n_ao), bool)
    m[off[a]:off[a] + sh[a].n_cart, off[b]:off[b] + sh[b].n_cart] = True
    return m | m.T


# ---------------------------------------------------------------------------
# Per class, one shell pair at a time
# ---------------------------------------------------------------------------


def test_hf_ccpvtz_reaches_every_class_and_a_diagonal_in_each_square_one():
    _, basis = _case("hf_ccpvtz")
    for cls in CLASSES:
        pairs = _pairs_of(basis, cls)
        # HF/cc-pVTZ has one f shell, so (ff) has only its diagonal pair.
        assert any(a != b for a, b in pairs) == (cls != (LMAX, LMAX)), cls
        if cls[0] == cls[1]:
            assert any(a == b for a, b in pairs), cls
    # Accumulation over records is exercised: some pairs have many.
    assert max(_n_rec(basis, a, b) for a, b in _pairs_of(basis, (0, 0))) > 32


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_the_kernel_equals_the_host_per_class(cls):
    _, basis = _case("hf_ccpvtz")
    pairs = _pairs_of(basis, cls)
    picks = []
    if cls != (LMAX, LMAX):  # one f shell: (ff)'s only pair is diagonal
        picks.append(max((p for p in pairs if p[0] != p[1]),
                         key=lambda p: _n_rec(basis, *p)))
    if cls[0] == cls[1]:
        picks.append(max((p for p in pairs if p[0] == p[1]),
                         key=lambda p: _n_rec(basis, *p)))
    for seed, (a, b) in enumerate(picks):
        host, kern = _both(basis, a, b, seed + 10 * sum(cls))
        assert np.array_equal(kern, host), f"{cls} pair {(a, b)}"
        owned = _block(basis, a, b)
        # The scatter writes exactly the pair's block and its mirror...
        assert np.all(kern[~owned] == 0.0)
        assert np.all(kern[owned] != 0.0), f"{cls} {(a, b)}: a zero element"
        # ...and is symmetric there, the diagonal included.
        assert np.array_equal(kern, kern.T)


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES if c[0] == c[1]])
def test_the_diagonal_is_not_double_counted(cls):
    """`a == b`: the naive full block plus its mirror doubles the diagonal
    elements (and the lower-triangle pairs); the kernel equals the host's
    one write. The control shows the case would see a double count."""
    _, basis = _case("hf_ccpvtz")
    a, b = max((p for p in _pairs_of(basis, cls) if p[0] == p[1]),
               key=lambda p: _n_rec(basis, *p))
    sh, off = basis.shells, basis.offsets
    bra = epair(sh[a], sh[b])
    jps = _jps(len(bra.recs), len(hermite_order(2 * cls[0])), 7)
    host = _zero(basis.n_ao)
    jab_from_host(bra, jps, (off[a], off[b]), host)
    k = KernelJAB()
    kern = _zero(basis.n_ao)
    k(bra, jps, (off[a], off[b]), kern)
    host, kern = np.array(host), np.array(kern)
    assert np.array_equal(kern, host)
    o, n = off[a], sh[a].n_cart
    d = np.diag(kern)[o:o + n]
    assert np.all(d != 0.0)
    # The control: a naive scatter adds the diagonal twice.
    naive = kern.copy()
    naive[o:o + n, o:o + n] += np.diag(d)
    assert not np.array_equal(naive, host)


# ---------------------------------------------------------------------------
# One launch over a whole class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_one_launch_over_the_whole_class(cls):
    """Every record of every shell pair of `cls` in ONE launch, with a
    shared `Jp` buffer whose rows are given by `row` (here reversed, so
    `row[k] != k`), diagonal and off-diagonal pairs mixed: `==` to the host
    applied pair by pair, in sweep order, into the same zeroed `J`."""
    _, basis = _case("hf_ccpvtz")
    sh, off = basis.shells, basis.offsets
    pairs = _pairs_of(basis, cls)
    n_p = len(hermite_order(sum(cls)))
    host = _zero(basis.n_ao)
    recs, offs, rows_jp = [], [], []
    for seed, (a, b) in enumerate(pairs):
        bra = epair(sh[a], sh[b])
        jps = _jps(len(bra.recs), n_p, 100 + seed)
        jab_from_host(bra, jps, (off[a], off[b]), host)
        for (P, zeta, s), jp in zip(bra.recs, jps, strict=True):
            recs.append((P, zeta, s, bra.A, bra.B))
            offs.append((off[a], off[b]))
            rows_jp.append(jp)
    n_rec = len(recs)
    row = np.arange(n_rec, dtype=np.int32)[::-1].copy()
    Jp = np.zeros((n_rec, n_p))
    Jp[row] = np.array(rows_jp)
    grid, block = jab_launch_shape(n_rec, 5)
    got = Interpreter(build_jab_kernel(jab_spec(*cls)),
                      launch=Launch(grid, block, 1)).run(
        n_rec=n_rec, n_ao=basis.n_ao, n_rows=n_rec,
        zeta_ab=np.array([r[1] for r in recs]),
        XPA=np.array([[r[0][d] - r[3][d] for d in range(3)] for r in recs]),
        XPB=np.array([[r[0][d] - r[4][d] for d in range(3)] for r in recs]),
        s=np.array([r[2] for r in recs]),
        off_a=np.array([o[0] for o in offs], dtype=np.int32),
        off_b=np.array([o[1] for o in offs], dtype=np.int32),
        row=row, Jp=Jp, J=np.zeros((basis.n_ao, basis.n_ao)))["J"]
    assert np.array_equal(got, np.array(host))
    assert np.count_nonzero(got) == np.count_nonzero(
        np.logical_or.reduce([_block(basis, a, b) for a, b in pairs]))


# ---------------------------------------------------------------------------
# The chain: J through the seam, against the host and libcint
# ---------------------------------------------------------------------------

CHAIN = [pytest.param("h2_sto3g_j", id="h2_sto3g_j"),
         pytest.param("h2o_sto3g_j", id="h2o_sto3g_j"),
         pytest.param("h2o_ccpvdz_j", id="h2o_ccpvdz_j",
                      marks=pytest.mark.slow)]


@pytest.mark.parametrize("name", CHAIN)
def test_the_chain_through_the_jab_seam(name):
    fx, basis = _case(name)
    k = KernelJAB()
    J = np.array(coulomb_matrix(basis, fx.density, pq=pq, jab=k))
    # The host path, with no hook: test_ao_j.py pins it `==` to the
    # `jab=jab_from_host` path, so this is the seam's own reference.
    R, scale = (np.array(x) for x in
                coulomb_matrix_and_scale(basis, fx.density, pq=pq))
    ref = fx.arrays[manifest.J_ARRAY]
    sh = basis.shells
    assert k.ran == {(sh[a].l, sh[b].l) for a in range(len(sh))
                     for b in range(a + 1)}
    assert np.array_equal(J, R)
    worst = float(np.max(np.abs(J - ref) / scale))
    print(f"\n{name}: max|J - get_jk|/scale_J = {worst:.2e}")
    assert np.all(np.abs(J - ref) <= JK_TOL * scale), worst


def test_the_seam_is_unscreened():
    fx, basis = _case("h2_sto3g_j")
    with pytest.raises(ValueError, match="unscreened"):
        coulomb_matrix(basis, fx.density, pq=pq, tau=1e-10, jab=KernelJAB())


# ---------------------------------------------------------------------------
# The contract, structurally
# ---------------------------------------------------------------------------


def test_the_key_is_a_pair_class():
    assert jab_spec(2, 1) == Spec(Operator.COULOMB, 0, (2, 1), Target.JAB)
    assert build_jab_kernel(jab_spec(2, 1)) == build_jab_kernel((2, 1))
    with pytest.raises(ValueError, match="JAB"):
        build_jab_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.DQ))
    with pytest.raises(ValueError, match="JAB"):
        build_jab_kernel(Spec(Operator.COULOMB, 0, (1, 0, 1, 0), Target.J))
    with pytest.raises(ValueError, match="levers"):
        build_jab_kernel(Spec(Operator.COULOMB, 0, (1, 1), Target.JAB,
                              ("recycle",)))


@pytest.mark.parametrize(
    "cls", split_by_l(CLASSES, sum, ids=lambda c: f"{c[0]}{c[1]}"))
def test_every_write_to_J_is_an_atomic_add_with_its_reason(cls):
    if jab_lowstack(cls):  # the unrolled form's shape, forced at f (#586)
        k = build_jab_kernel(cls, lowstack=False)
    else:
        k = build_jab_kernel(cls)
    stmts = list(walk(k.body))
    notes = {s.prov for s in stmts if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_J in notes
    assert not [s for s in stmts if isinstance(s, Assign)
                and isinstance(s.dst, Index) and s.dst.array == "J"]
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert {s.dst.array for s in adds} == {"J"}
    na, nb = len(ef.cartesians(cls[0])), len(ef.cartesians(cls[1]))
    (scatter,) = [s for s in stmts if isinstance(s, If)
                  and any(isinstance(t, AtomicAdd) for t in s.then)]
    off = [s for s in scatter.then if isinstance(s, AtomicAdd)]
    diag = [s for s in scatter.orelse if isinstance(s, AtomicAdd)]
    assert len(off) == 2 * na * nb
    # The diagonal branch exists only where a diagonal pair can: n^2 adds,
    # one per element of the block (n(n+1)/2 lower, n(n-1)/2 mirrored).
    assert len(diag) == (na * na if cls[0] == cls[1] else 0)


def test_the_builder_reaches_neither_ao_nor_the_interpreter():
    """The kernel is checked against `eri.j_coulomb.j` in the interpreter: neither may
    be its ancestor, nor `reference/` (test_layering.py holds the rule)."""
    graph = _graph()
    seen, todo = set(), ["eri.j_coulomb.kernel_jab"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "mmd.e_fragment" in seen
    for banned in ("ao", "ir.interp", "reference", "emit"):
        assert not [e for e in seen if _under(e, banned)], banned


# ---------------------------------------------------------------------------
# The low-stack form (#586, stage 2): the default at l_a + l_b >= 6, the
# text below it unchanged, the same bits
# ---------------------------------------------------------------------------

# sha256 of the unrolled JAB text of every class below L = 6 with l <= 3 --
# through (dd), and the mixed f classes (sf), (pf), (df) -- cuda and hip, as
# emitted at 534a8e7 (before stage 2): the low-stack form must not move a
# byte of it.
D_DIGESTS = {
    (0, 0): ("f76d925076b5bbfb155a71145e2c7d3a5272b8a10ed0de523784c86134896d44",
             "558fb6e7423d646d4277649ca721684fe519b3d3145fcdee9913c6edfa9fd782"),
    (0, 1): ("cd5a69157f87aa469d544ad5c7211b8448adc48714ce83bab98a2ec9140fb977",
             "e3c17c557f0a5f4839fe8b28304d3eb7a2dbe1ae86143cd463a3db5f06192ebd"),
    (0, 2): ("78dd7b4115f9dd7469cae60e5fe151ef0aaee7cf70ef6ea2f84c2942796139de",
             "8621ca9c8e5c26595dc82170fdafaed144dee07f3b1d80cf81e69641a54eb4d0"),
    (0, 3): ("bdcd778a393bc57a342310243a6af3275d0b75a4b5fde1a4d8c9845026573d85",
             "64623e635d1c058f483fc83a9288331671b0b863c7ef3bdea4eeea62aa626d97"),
    (1, 0): ("e6dc53b551a0fca67bec475a668abe0c99807e1b21d0bcf0de7d840afd999b4e",
             "97a19b12ef6631143a71134d89ff2c45f0080383e95d4f51aab66c607e4a50a5"),
    (1, 1): ("a53d0fdbd46249eb993706d8294c02324fc85593440cbdf88c16867b7ec775b2",
             "6b7c4553560db4baa3351e6e4e058dc34f951c7df49ecb1f6d1c8968aed6d462"),
    (1, 2): ("627bb79f320dc0f72a5af4484a42fe94e24ed28a9db1b3f8c04084e0e68db48a",
             "35ffc24e225bff1f36640ecb4d6cdcc0d5fecb7ba820d424be255bf24353c423"),
    (1, 3): ("08f495433e818c445b85724ea8f251475cd2a04f62b3fe070ccddad5c62caa08",
             "b8fedacbc49dc92d9d71d457d5a0af8a0ddda3733606818b02f9ffdf89e9661f"),
    (2, 0): ("7199de59caac64b5fb31e308a64ed857f94dc04cf7e92946b4f95c25239827cf",
             "e759cf66beafb38a07579c2b381b3fa51c7ae97c3a17001652595bfd9155b6bd"),
    (2, 1): ("095127064af18df4d61dccbe4f5c82f5c6e83dc2da56610bf530eba3f3c166e6",
             "938e209d765a224cb188dd7be831cedcd57a941b5ded2a674175afa28fe62a54"),
    (2, 2): ("baf4f145a3b8c4a0aa2b3ba22dc440ec8fa265437e70c7732033e5ee64c06e1b",
             "1838df3e9120cbfb0c2fa28b7afe703c948bf811eac11809e70178adb0a191d5"),
    (2, 3): ("9db32dbbe78f98f53e3a451fd71e585a94a020c1b0596e3b43c9d9b5a71866c4",
             "33d7a7bc124ccfd0dfbf8306eb72d038e7d2de5e7e1a852e5780db295551ced6"),
    (3, 0): ("5592b783b21b9910c33be046eac4721858e8f0f127b512fb974eb8aea3a16cb2",
             "2f6c52e41216b73faaff8bd8627f181eeabf9532901259173aefd7f9bee8ad79"),
    (3, 1): ("cdb0a839d6cfc6050622f612be344eba0694ac42df88c095825151641add8a68",
             "c5d43e4ed39dddd90041c94d2380420a61578259b3c32782c56cb2c2ae72cd0c"),
    (3, 2): ("dda52ae37ca851f27d361a5f6febfc402b7289a29384a705207f39e040981f35",
             "d387cb36ab0ff780551ffbaa54e57fa421fdaa5e10e948e631c76ec44d1afca4"),
}

# #586's budget: at most 8 KB of stack per thread at f on sm_86, held here
# on the kernel's local arrays (the measured frames: j_engine.md).
STACK_BUDGET = 8192
F_CLASSES = [(0, 3), (1, 3), (2, 3), (3, 3), (3, 0), (3, 1), (3, 2)]


def _ids(c) -> str:
    return f"{c[0]}{c[1]}"


def test_the_selection_rule_is_pair_l_6_and_above():
    # one rule, DQ's: (ff), and past f (gd), (gf), ...
    assert LOWSTACK_MIN_PAIR_L == kernel_dq.LOWSTACK_MIN_PAIR_L == 6
    for la in range(5):
        for lb in range(5):
            assert jab_lowstack((la, lb)) is (la + lb >= 6)
            assert jab_lowstack((la, lb)) is kernel_dq.dq_lowstack((la, lb))
            assert jab_lowstack(jab_spec(la, lb)) is jab_lowstack((la, lb))


@pytest.mark.parametrize("cls", split_by_l(sorted(D_DIGESTS), sum, ids=_ids))
def test_the_text_below_l6_is_byte_identical(cls):
    import hashlib
    k = build_jab_kernel(cls)
    assert k == build_jab_kernel(cls, lowstack=False)
    got = tuple(hashlib.sha256(t.encode()).hexdigest()
                for t in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)))
    assert got == D_DIGESTS[cls]


@pytest.mark.parametrize("cls", split_by_l([(1, 1), (2, 3), (3, 2), (3, 3)],
                                           sum, ids=_ids))
def test_both_forms_share_the_name_parameters_and_launch(cls):
    lo = build_jab_kernel(cls, lowstack=True)
    hi = build_jab_kernel(cls, lowstack=False)
    assert lo.name == hi.name == jab_name(cls)
    assert lo.params == hi.params
    assert build_jab_kernel(cls) == (lo if jab_lowstack(cls) else hi)
    rec = [s for s in lo.body if isinstance(s, Decl) and s.name == "rec"]
    assert rec and rec == [s for s in hi.body
                           if isinstance(s, Decl) and s.name == "rec"]
    # free-breakfast's spelling (tools/emit_coulomb_kernels.py): the record
    # vectors split per axis, the same parameters either form
    axes = {"XPA": ("pa_x", "pa_y", "pa_z"), "XPB": ("pb_x", "pb_y", "pb_z")}
    assert split_axes(lo, axes).params == split_axes(hi, axes).params


def _local_doubles(k) -> dict[str, int]:
    return {s.name: int(np.prod([d.value for d in s.shape]))
             for s in walk(k.body) if isinstance(s, Decl) and s.shape}


@pytest.mark.parametrize("cls", split_by_l(F_CLASSES, sum, ids=_ids))
def test_the_lowstack_form_holds_no_e_block(cls):
    # every f class, forced where the rule keeps the unrolled form
    la, lb = cls
    k = build_jab_kernel(cls, lowstack=True)
    arrays = _local_doubles(k)
    ext = ef.axis_table_extent(la, lb)
    assert arrays == {"Ex": ext, "Ey": ext, "Ez": ext}
    assert 8 * sum(arrays.values()) <= STACK_BUDGET
    (inl,) = [s for s in walk(k.body, into_inline=False)
              if isinstance(s, Inline)]
    assert inl.fragment == ef.axis_tables_fragment(la, lb)
    # every write to J is still an AtomicAdd with its reason: one element
    # and its mirror per Cartesian pair, in place
    stmts = list(walk(k.body))
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert len(adds) == 2 and {s.dst.array for s in adds} == {"J"}
    assert ATOMIC_REASON in {s.prov for s in stmts if isinstance(s, Note)}
    assert not [s for s in stmts if isinstance(s, Assign)
                and isinstance(s.dst, Index) and s.dst.array == "J"]


@pytest.mark.slow  # builds (ff), L = 6 > FAST_L_MAX (tiers.py)
def test_the_ff_lowstack_arrays_are_336_doubles():
    # 3 x 112 axis-table entries: 2,688 B, all of sm_86's measured frame.
    assert sum(_local_doubles(build_jab_kernel((3, 3))).values()) == 336


def _other_form(cls) -> bool:
    return not jab_lowstack(cls)


@pytest.mark.parametrize("cls", [_cls_param(c) for c in CLASSES])
def test_the_other_form_equals_the_host_in_one_launch(cls):
    """Every class through (ff) in the form `build_jab_kernel` does NOT
    choose by default -- the low-stack form through d, the unrolled one at
    f -- over every record of the class in one launch, diagonal and
    off-diagonal pairs mixed, `==` to the host: with the default-form tests
    above, both forms are held bitwise at every class."""
    _, basis = _case("hf_ccpvtz")
    sh, off = basis.shells, basis.offsets
    pairs = _pairs_of(basis, cls)
    n_p = len(hermite_order(sum(cls)))
    host = _zero(basis.n_ao)
    recs, offs, rows_jp = [], [], []
    for seed, (a, b) in enumerate(pairs):
        bra = epair(sh[a], sh[b])
        jps = _jps(len(bra.recs), n_p, 586 + seed)
        jab_from_host(bra, jps, (off[a], off[b]), host)
        for (P, zeta, s), jp in zip(bra.recs, jps, strict=True):
            recs.append((P, zeta, s, bra.A, bra.B))
            offs.append((off[a], off[b]))
            rows_jp.append(jp)
    n_rec = len(recs)
    grid, block = jab_launch_shape(n_rec, 7)
    got = Interpreter(build_jab_kernel(cls, lowstack=_other_form(cls)),
                      launch=Launch(grid, block, 1)).run(
        n_rec=n_rec, n_ao=basis.n_ao, n_rows=n_rec,
        zeta_ab=np.array([r[1] for r in recs]),
        XPA=np.array([[r[0][d] - r[3][d] for d in range(3)] for r in recs]),
        XPB=np.array([[r[0][d] - r[4][d] for d in range(3)] for r in recs]),
        s=np.array([r[2] for r in recs]),
        off_a=np.array([o[0] for o in offs], dtype=np.int32),
        off_b=np.array([o[1] for o in offs], dtype=np.int32),
        row=np.arange(n_rec, dtype=np.int32), Jp=np.array(rows_jp),
        J=np.zeros((basis.n_ao, basis.n_ao)))["J"]
    assert np.array_equal(got, np.array(host))


@pytest.mark.parametrize("cls", [by_l((0, 1), 1, id="01"),
                                 by_l((1, 3), 4, id="13")])
def test_a_record_with_equal_offsets_writes_nothing_off_the_square(cls):
    # la != lb has no diagonal branch: the unrolled form's If(differ, ...)
    # with no else, so a record with off_a == off_b adds nothing. The
    # low-stack form keeps that, bit for bit.
    n_p = len(hermite_order(sum(cls)))
    args = dict(n_rec=1, n_ao=12, n_rows=1, zeta_ab=np.array([1.3]),
                XPA=np.array([[0.1, -0.2, 0.3]]),
                XPB=np.array([[-0.4, 0.5, 0.2]]), s=np.array([0.7]),
                off_a=np.array([2], np.int32), off_b=np.array([2], np.int32),
                row=np.array([0], np.int32),
                Jp=np.linspace(-1, 1, n_p)[None, :])
    for lowstack in (False, True):
        J = Interpreter(build_jab_kernel(cls, lowstack=lowstack),
                        launch=Launch(1, (1, 1), 1)).run(
            J=np.zeros((12, 12)), **args)["J"]
        assert not J.any()


# The (ff) default text, both backends: the low-stack form's golden.
LOWSTACK_GOLDEN_CLASS = (3, 3)
GOLDEN = Path(__file__).resolve().parent / "golden"


@pytest.mark.slow  # emits (ff), L = 6 > FAST_L_MAX (tiers.py)
def test_the_lowstack_cuda_golden_is_unchanged():
    # Regenerate (never hand-edit), the default form at (ff):
    #   PYTHONPATH=src uv run python -c "from felizjueves.eri.j_coulomb.kernel_jab \
    #   import build_jab_kernel as b; from felizjueves.performance.device import \
    #   SM_86; from felizjueves.emit.cuda import emit; print(emit(b((3, 3)), \
    #   SM_86), end='')" > test/golden/cuda/jab_33.cu
    text = cuda.emit(build_jab_kernel(LOWSTACK_GOLDEN_CLASS), SM_86)
    assert text == (GOLDEN / "cuda" / "jab_33.cu").read_text()


@pytest.mark.slow  # emits (ff)
def test_the_lowstack_hip_golden_is_unchanged():
    #   ... the same with GFX1200 and felizjueves.emit.hip, into
    #   test/golden/hip/jab_33.hip
    text = hip.emit(build_jab_kernel(LOWSTACK_GOLDEN_CLASS), GFX1200)
    assert text == (GOLDEN / "hip" / "jab_33.hip").read_text()
