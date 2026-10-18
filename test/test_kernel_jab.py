"""`kernel_jab`: the bra transform and scatter, `Target.JAB`, interpreted (#154).

THE WITNESS is `ao.j.jab_from_host`, the host's own bra transform and
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

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_layering import _graph, _under
from tiers import split_by_l

from felizjueves import e_fragment as ef
from felizjueves.ao.drive import KernelJAB
from felizjueves.ao.hermite import hermite_order
from felizjueves.ao.j import (
    coulomb_matrix,
    coulomb_matrix_and_scale,
    epair,
    jab_from_host,
)
from felizjueves.interp import Interpreter, Launch
from felizjueves.ir import Assign, AtomicAdd, If, Index, Note, walk
from felizjueves.kernel_jab import (
    ATOMIC_REASON,
    ZEROED_J,
    build_jab_kernel,
    jab_launch_shape,
    jab_spec,
)
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
    """The kernel is checked against `ao.j` in the interpreter: neither may
    be its ancestor, nor `reference/` (test_layering.py holds the rule)."""
    graph = _graph()
    seen, todo = set(), ["kernel_jab"]
    while todo:
        for e in graph.get(todo.pop(), set()):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    assert "e_fragment" in seen
    for banned in ("ao", "interp", "reference", "emit"):
        assert not [e for e in seen if _under(e, banned)], banned
