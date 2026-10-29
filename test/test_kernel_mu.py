"""`kernel_mu`: the fused dipole kernel, `Target.DIPOLE`, interpreted (#521).

THE WITNESS is `one_electron.multipole.multipole_block` (#520), the host's
dipole block `<a| q (r - C) |b>` by the Hermite route. The kernel's records
are `ao.pairs.primitive_pairs`' `(zeta, P, P - A, P - B)` and the host's one
scalar `s` per record (`ao.density.epair`), its origin and charge runtime
scalars, and its scatter `kernel_jab`'s rule, per component. Against it:

- PER CLASS, every ordered pair class (ss) through (dd) -- milestone 19's
  cap: nothing above L = 4, every shell l <= 2 -- on the tracked
  `hf_ccpvtz_spd_dipole` basis (HF/cc-pVTZ without f, #520), in ONE launch
  over every record of every shell pair `i <= j` of the class, diagonal and
  off-diagonal pairs mixed: `np.array_equal` with `multipole_block` placed
  by `kernel_jab`'s rule (kernel_mu's docstring, "WHY `==`"), at both of
  the fixture's origins, `q = 1`, `q = -1` and a non-unit `q`.
- THE WHOLE MATRIX, one launch per class into one zeroed `MU`, against
  `multipole_matrix`: the off-diagonal shell pairs' elements `==` (both read
  the same block entry), the diagonal shell pairs' within `DIAG_BOUND`
  (below).
- LIBCINT DIRECTLY, not only through the host: the kernel interpreted over
  `one_electron_launch`'s plan (its `OneEClass.arrays(Target.DIPOLE)`), and
  `walk_one_electron`'s `MU` from the same plan, against `int1e_r_cart` at
  both origins for H2O/STO-3G and 6-31G (#517), and per shell-pair block
  against `hf_ccpvtz_spd_dipole` (#520), within `LIBCINT_TOL` of the
  largest `|r|`.
- `with_overlap`: component 0 bitwise `kernel_s`'s `S`, components 1..3
  bitwise the plain kernel's.
- THE ORIGIN SHIFT on the interpreted output: `MU(C') = MU(C) + q (C - C') S`.
- AT f (#627), on `hf_ccpvtz_dipole` (the same molecule, f shells kept):
  every ordered class with an f, (sf) through (ff), per class bitwise the
  host block at both origins, the charge an exact factor and `with_overlap`
  `kernel_s` bitwise; the plan's launches per shell-pair block against
  libcint at both origins, and the origin shift against its `S`. A class
  above L = 4 is `slow` (tiers.FAST_L_MAX), and so is each whole-basis
  check, which launches (ff).

THE DIAGONAL BOUND. `one_electron.one_electron.matrix` reads a diagonal
shell pair's UPPER triangle, `kernel_jab`'s rule keeps the LOWER;
`<a_i|x|a_j>` and `<a_j|x|a_i>` are one integral reached along different
`E` recurrence paths, so they agree to rounding, not by construction. A
dipole element is a sum `E_0 X_PC + E_1` per axis, so it can cancel, and a
bound relative to the element itself is not sound; the bound is
`DIAG_BOUND * eps * max|MU_k|` per component `k` (`eps = 2^-52`): a few
roundings of the three-factor product, the two-term axis sum and the record
sum, against the component's scale. Measured worst on the spd basis: 0
(printed by the test) -- a diagonal shell pair has `P - A = P - B`, so the
two paths happen to agree there -- but 0 is an observation, not a
guarantee, so the bound stays stated.

Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_kernel_s import _launch as _s_launch
from test_kernel_v import _records
from tiers import split_by_l

from felizjueves.ao.pairs import overlap_bound
from felizjueves.emit import cuda, hip
from felizjueves.eri.j_coulomb.launch import one_electron_launch, walk_one_electron
from felizjueves.ir import AtomicAdd, If, Inline, Note, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd import e_fragment as ef
from felizjueves.one_electron.kernel_mu import (
    ATOMIC_REASON,
    MU,
    ZEROED_MU,
    build_mu_kernel,
    launch_shape,
    mu_name,
    mu_spec,
    n_components,
)
from felizjueves.one_electron.multipole import multipole_block, multipole_matrix
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Operator, Spec, Target

LMAX = 2  # milestone 19: every shell l <= 2, so (ss) through (dd)
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
EPS = 2.0 ** -52
DIAG_BOUND = 16  # x eps max|MU_k|: see the module docstring
LIBCINT_TOL = 1e-12  # of max|r|: the acceptance's bound
SHIFT_TOL = 1e-14  # of max|MU|: test_ao_multipole.py's, on the host
ZERO = (0.0, 0.0, 0.0)
H2O = ("h2o_sto3g", "h2o_631g")


# #627: the ordered classes with an f, on the f witness
F_LMAX = 3
F_CLASSES = [(la, lb) for la in range(F_LMAX + 1) for lb in range(F_LMAX + 1)
             if F_LMAX in (la, lb)]
F_TIERED = split_by_l(F_CLASSES, sum, ids=lambda c: f"{c[0]}{c[1]}")


@cache
def _spd():
    d = manifest.dipole_integrals()
    return d, ob.from_fixture(d.fixture)


@cache
def _f():
    d = manifest.dipole_integrals(manifest.F_DIPOLE)
    return d, ob.from_fixture(d.fixture)


def _pairs_of(basis, cls):
    """Shell pairs `i <= j` of class `cls`, in `one_electron.matrix` order."""
    sh = basis.shells
    return [(i, j) for i in range(len(sh)) for j in range(i, len(sh))
            if (sh[i].l, sh[j].l) == cls]


def _run(kernel, rec, n_ao, MUin, C, q, threads=5):
    n_rec = len(rec["s_ab"])
    grid, block = launch_shape(n_rec, threads)
    return Interpreter(kernel, launch=Launch(grid, block, 1)).run(
        n_rec=n_rec, n_ao=n_ao, C_x=float(C[0]), C_y=float(C[1]),
        C_z=float(C[2]), q=float(q), MU=MUin, **rec)[MU]


def _launch(cls, basis, pairs, MUin, C, q=1.0, with_overlap=False):
    """One launch of class `cls` over every record of `pairs`, into `MU`."""
    return _run(build_mu_kernel(mu_spec(*cls), with_overlap=with_overlap),
                _records(basis, pairs), basis.n_ao, MUin, C, q)


def _zeros(basis, with_overlap=False):
    n = basis.n_ao
    return np.zeros((n_components(with_overlap), n, n))


def _host(basis, pairs, C, q=1.0):
    """`multipole_block` placed into a fresh `MU` by `kernel_jab`'s rule."""
    sh, off = basis.shells, basis.offsets
    M = _zeros(basis)
    for i, j in pairs:
        blk = multipole_block(sh[i], sh[j], C, q=q)
        for k in range(3):
            for r in range(sh[i].n_cart):
                for c in range(sh[j].n_cart):
                    if i == j and c > r:
                        continue  # diagonal: the lower triangle, mirrored
                    M[k, off[i] + r, off[j] + c] = blk[k][r][c]
                    M[k, off[j] + c, off[i] + r] = blk[k][r][c]
    return M


def _whole(basis, C, q=1.0, with_overlap=False):
    M = _zeros(basis, with_overlap)
    sh = basis.shells
    for cls in sorted({(a.l, b.l) for k, a in enumerate(sh) for b in sh[k:]}):
        M = _launch(cls, basis, _pairs_of(basis, cls), M, C, q, with_overlap)
    return M


def _from_plan(basis, C, q=1.0):
    """The kernel interpreted over `one_electron_launch`'s plan: each class
    launched with the plan's own record arrays."""
    plan = one_electron_launch(basis, (), tau=0.0, bound=overlap_bound)
    M = _zeros(basis)
    for c in plan:
        rec = {k: np.asarray(v, dtype=np.int32 if k.startswith("off")
                             else float)
               for k, v in c.arrays(Target.DIPOLE).items()}
        M = _run(build_mu_kernel(c.spec(Target.DIPOLE)), rec, basis.n_ao,
                 M, C, q)
    return M, plan


def _rel(got, want) -> float:
    want = np.asarray(want)
    return float(np.max(np.abs(np.asarray(got) - want))
                 / np.max(np.abs(want)))


def _diag_mask(basis) -> np.ndarray:
    sh, off = basis.shells, basis.offsets
    m = np.zeros((basis.n_ao, basis.n_ao), bool)
    for i, s in enumerate(sh):
        m[off[i]:off[i] + s.n_cart, off[i]:off[i] + s.n_cart] = True
    return m


# ---------------------------------------------------------------------------
# The kernel's shape
# ---------------------------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert mu_spec(2, 1) == Spec(Operator.DIPOLE, 0, (2, 1), Target.DIPOLE)
    assert mu_spec(2, 1).key() == "dipole/d0/(2,1)/b1/dipole/literal"
    assert mu_name((2, 1)) == build_mu_kernel((2, 1)).name == "mu_21"
    assert mu_name((0, 2), with_overlap=True) == "mu_02_s"
    assert build_mu_kernel((0, 2), with_overlap=True).name == "mu_02_s"
    assert n_components() == 3 and n_components(True) == 4


def test_the_parameter_block() -> None:
    k = build_mu_kernel((1, 2))
    assert [p.name for p in k.params] == [
        "n_rec", "n_ao", "zeta_ab", "p_x", "p_y", "p_z", "pa_x", "pa_y",
        "pa_z", "pb_x", "pb_y", "pb_z", "s_ab", "off_a", "off_b", "C_x",
        "C_y", "C_z", "q", "MU"]
    mu = k.params[-1]
    assert mu.array and len(mu.shape) == 3 and mu.shape[0].value == 3
    assert build_mu_kernel((1, 2), with_overlap=True).params[-1].shape[0] \
        .value == 4
    assert not any(p.array for p in k.params if p.name in
                   ("C_x", "C_y", "C_z", "q"))


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_one_inline_and_one_atomic_add_per_value_and_mirror(cls) -> None:
    """The fragment is inlined ONCE (the E tables built once for every
    component), and the scatter is kernel_jab's rule per component: each
    value's off-diagonal branch adds twice; a diagonal class's diagonal
    branch adds each of the n^2 elements of each component exactly once."""
    la, lb = cls
    k = build_mu_kernel(cls)
    stmts = list(walk(k.body))
    assert sum(isinstance(s, Inline) for s in stmts) == 1
    notes = {s.prov for s in stmts if isinstance(s, Note)}
    assert {ATOMIC_REASON, ZEROED_MU} <= notes
    na, nb = len(ef.cartesians(la)), len(ef.cartesians(lb))
    ifs = [s for s in walk(k.body, into_inline=True)
           if isinstance(s, If) and s.then and isinstance(s.then[0], AtomicAdd)]
    assert len(ifs) == 3 * na * nb
    assert all(len(s.then) == 2 for s in ifs)
    dests = set()
    for s in ifs:
        dests |= {(a.dst.at[0].value, str(a.dst.at[1]), str(a.dst.at[2]))
                  for a in s.orelse}
    if la == lb:
        assert sum(len(s.orelse) for s in ifs) == 3 * na * na
        assert len(dests) == 3 * na * na
    else:
        assert all(s.orelse == () for s in ifs)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_both_backends_emit_every_class(cls) -> None:
    for w in (False, True):
        k = build_mu_kernel(cls, with_overlap=w)
        for text in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)):
            assert f"void {k.name}(" in text
            assert text.count("atomicAdd") > 0


def test_the_refusals() -> None:
    with pytest.raises(ValueError, match="not a Target.DIPOLE key"):
        build_mu_kernel(Spec(Operator.OVERLAP, 0, (1, 1), Target.S))
    with pytest.raises(ValueError, match="positive int"):
        launch_shape(0)
    with pytest.raises(ValueError, match="from_order"):
        ef.multipole_fragment(1, 1, 1, from_order=2)
    assert launch_shape(33, 32) == (2, (32, 1))


# ---------------------------------------------------------------------------
# Per class: one launch over every record of the class, bitwise
# ---------------------------------------------------------------------------


def test_the_spd_basis_reaches_every_class_in_both_orders() -> None:
    _, basis = _spd()
    assert max(s.l for s in basis.shells) == LMAX
    for cls in CLASSES:
        pairs = _pairs_of(basis, cls)
        assert pairs, cls
        if cls[0] == cls[1]:
            assert any(i == j for i, j in pairs), cls


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_one_launch_over_the_class_equals_the_host_block(cls) -> None:
    d, basis = _spd()
    pairs = _pairs_of(basis, cls)
    for C in (d.origin, ZERO):
        got = _launch(cls, basis, pairs, _zeros(basis), C)
        want = _host(basis, pairs, C)
        assert np.array_equal(got, want), tuple(C)
        assert np.array_equal(got, got.transpose(0, 2, 1))


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_the_charge_is_an_exact_factor(cls) -> None:
    d, basis = _spd()
    pairs = _pairs_of(basis, cls)
    plus = _launch(cls, basis, pairs, _zeros(basis), d.origin, 1.0)
    minus = _launch(cls, basis, pairs, _zeros(basis), d.origin, -1.0)
    assert np.array_equal(minus, -plus)
    q = 0.75
    got = _launch(cls, basis, pairs, _zeros(basis), d.origin, q)
    assert np.array_equal(got, _host(basis, pairs, d.origin, q=q))


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_the_overlap_component_is_kernel_s_bitwise(cls) -> None:
    d, basis = _spd()
    pairs = _pairs_of(basis, cls)
    n = basis.n_ao
    got = _launch(cls, basis, pairs, _zeros(basis, True), d.origin,
                  with_overlap=True)
    S = _s_launch(cls, basis, pairs, np.zeros((n, n)))
    assert np.array_equal(got[0], S)
    plain = _launch(cls, basis, pairs, _zeros(basis), d.origin)
    assert np.array_equal(got[1:], plain)


# ---------------------------------------------------------------------------
# The whole matrix
# ---------------------------------------------------------------------------


def test_the_whole_matrix_against_multipole_matrix() -> None:
    d, basis = _spd()
    sh = basis.shells
    every = [(i, j) for i in range(len(sh)) for j in range(i, len(sh))]
    diag = _diag_mask(basis)
    worst = 0.0
    for C in (d.origin, ZERO):
        M = _whole(basis, C)
        assert np.array_equal(M, _host(basis, every, C))  # bitwise
        R = np.array(multipole_matrix(basis, C))
        for k in range(3):
            assert np.array_equal(M[k][~diag], R[k][~diag])
            err = np.abs(M[k] - R[k])[diag]
            scale = np.max(np.abs(R[k]))
            worst = max(worst, float(err.max() / (EPS * scale)))
            assert np.all(err <= DIAG_BOUND * EPS * scale), (tuple(C), k)
    print(f"\nmax|MU - multipole_matrix| on diagonal shell pairs = "
          f"{worst:.2f} eps max|MU_k|")


# ---------------------------------------------------------------------------
# libcint, directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", H2O)
def test_the_plan_launched_and_walked_is_libcints_int1e_r_cart(name) -> None:
    p = manifest.dipole_pair(name)
    basis = ob.from_fixture(p.dipole)
    assert max(s.l for s in basis.shells) <= LMAX
    for C, want in ((p.origin, p.r), (ZERO, p.r0)):
        got, plan = _from_plan(basis, C)
        assert _rel(got, want) <= LIBCINT_TOL, (name, "kernel", tuple(C))
        walked = np.array(walk_one_electron(plan, basis, Target.DIPOLE,
                                            origin=C))
        assert _rel(walked, want) <= LIBCINT_TOL, (name, "walk", tuple(C))
        assert np.array_equal(walked, np.array(multipole_matrix(basis, C)))


def test_every_block_of_hf_spd_is_libcints() -> None:
    d, basis = _spd()
    sh, off = basis.shells, basis.offsets
    for C, want in ((d.origin, d.r), (ZERO, d.r0)):
        got, _ = _from_plan(basis, C)
        scale = float(np.max(np.abs(want)))
        for i, sa in enumerate(sh):
            for j, sb in enumerate(sh):
                blk = (slice(None), slice(off[i], off[i] + sa.n_cart),
                       slice(off[j], off[j] + sb.n_cart))
                assert np.max(np.abs(got[blk] - want[blk])) \
                    <= LIBCINT_TOL * scale, (i, j, tuple(C))


# ---------------------------------------------------------------------------
# The origin shift, on the interpreted output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [1.0, 0.75])
def test_the_origin_shift_on_the_interpreted_output(q) -> None:
    """`MU(C') = MU(C) + q (C - C') S`, every term interpreted: `q S` is
    the `with_overlap` launch's component 0."""
    d, basis = _spd()
    C, C2 = np.asarray(d.origin), np.array((-1.1, 0.25, 0.6))
    at_c = _whole(basis, C, q, with_overlap=True)
    at_c2 = _whole(basis, C2, q)
    qS = at_c[0]
    want = at_c[1:] + (C - C2)[:, None, None] * qS[None]
    assert _rel(at_c2, want) <= SHIFT_TOL
    # and against libcint's overlap, the shift from the origin to 0
    mu0 = _whole(basis, ZERO)
    assert _rel(_whole(basis, C), mu0 - C[:, None, None] * d.ovlp[None]) \
        <= LIBCINT_TOL


# ---------------------------------------------------------------------------
# At f (#627): the f witness
# ---------------------------------------------------------------------------


def test_the_f_basis_reaches_every_f_class_in_both_orders() -> None:
    _, basis = _f()
    assert max(s.l for s in basis.shells) == F_LMAX
    assert len(F_CLASSES) == 7
    for cls in F_CLASSES:
        pairs = _pairs_of(basis, cls)
        assert pairs, cls
        if cls[0] == cls[1]:
            assert any(i == j for i, j in pairs), cls


@pytest.mark.parametrize("cls", F_TIERED)
def test_at_f_one_launch_over_the_class_equals_the_host_block(cls) -> None:
    d, basis = _f()
    pairs = _pairs_of(basis, cls)
    for C in (d.origin, ZERO):
        got = _launch(cls, basis, pairs, _zeros(basis), C)
        assert np.array_equal(got, _host(basis, pairs, C)), tuple(C)
        assert np.array_equal(got, got.transpose(0, 2, 1))


@pytest.mark.parametrize("cls", F_TIERED)
def test_at_f_the_charge_is_an_exact_factor(cls) -> None:
    d, basis = _f()
    pairs = _pairs_of(basis, cls)
    plus = _launch(cls, basis, pairs, _zeros(basis), d.origin, 1.0)
    minus = _launch(cls, basis, pairs, _zeros(basis), d.origin, -1.0)
    assert np.array_equal(minus, -plus)


@pytest.mark.parametrize("cls", F_TIERED)
def test_at_f_the_overlap_component_is_kernel_s_bitwise(cls) -> None:
    d, basis = _f()
    pairs = _pairs_of(basis, cls)
    n = basis.n_ao
    got = _launch(cls, basis, pairs, _zeros(basis, True), d.origin,
                  with_overlap=True)
    assert np.array_equal(got[0], _s_launch(cls, basis, pairs,
                                            np.zeros((n, n))))
    plain = _launch(cls, basis, pairs, _zeros(basis), d.origin)
    assert np.array_equal(got[1:], plain)


@pytest.mark.slow  # the plan launches (ff), L = 6
def test_at_f_every_block_of_the_plan_is_libcints() -> None:
    d, basis = _f()
    sh, off = basis.shells, basis.offsets
    worst = 0.0
    for C, want in ((d.origin, d.r), (ZERO, d.r0)):
        got, plan = _from_plan(basis, C)
        assert {c.spec(Target.DIPOLE).am for c in plan} >= set(F_CLASSES)
        scale = float(np.max(np.abs(want)))
        for i, sa in enumerate(sh):
            for j, sb in enumerate(sh):
                blk = (slice(None), slice(off[i], off[i] + sa.n_cart),
                       slice(off[j], off[j] + sb.n_cart))
                err = float(np.max(np.abs(got[blk] - want[blk])))
                worst = max(worst, err / scale)
                assert err <= LIBCINT_TOL * scale, (i, j, tuple(C))
        walked = np.array(walk_one_electron(plan, basis, Target.DIPOLE,
                                            origin=C))
        assert _rel(walked, want) <= LIBCINT_TOL, ("walk", tuple(C))
    print(f"\nf witness: max|MU - int1e_r_cart| = {worst:.2e} of max|r|")


@pytest.mark.slow  # every class through (ff), L = 6
def test_at_f_the_origin_shift_on_the_interpreted_output() -> None:
    d, basis = _f()
    C, C2 = np.asarray(d.origin), np.array((-1.1, 0.25, 0.6))
    at_c = _whole(basis, C, 1.0, with_overlap=True)
    at_c2 = _whole(basis, C2, 1.0)
    want = at_c[1:] + (C - C2)[:, None, None] * at_c[0][None]
    assert _rel(at_c2, want) <= SHIFT_TOL
    mu0 = _whole(basis, ZERO)
    assert _rel(at_c[1:], mu0 - C[:, None, None] * d.ovlp[None]) \
        <= LIBCINT_TOL
