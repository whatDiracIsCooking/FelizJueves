"""`kernel_mu` at `e = 2`: the fused quadrupole kernel, `Target.QUADRUPOLE`,
interpreted.

THE WITNESS is `one_electron.multipole.multipole_block(..., e=2)`, the host's
second-moment block `<a| q (r - C)_d (r - C)_e |b>` by the Hermite route,
components xx, xy, xz, yy, yz, zz. The kernel is the dipole's (#521) with the
fragment at `e = 2`: the same records, runtime origin and charge, and
`kernel_jab`'s scatter per component. Against it:

- PER CLASS, every ordered pair class (ss) through (dd) -- the dipole's cap:
  nothing above L = 4, every shell l <= 2 -- on the tracked
  `hf_ccpvtz_spd_dipole` basis, in ONE launch over every record of every
  shell pair `i <= j` of the class: `np.array_equal` with `multipole_block`
  placed by `kernel_jab`'s rule (kernel_mu's docstring, "WHY `==`": at
  `e = 2` the moments `M^2_0 = X_PC X_PC + 1/(2 zeta)`, `M^2_1 = X_PC + X_PC`
  and `M^2_2 = 2` are formed in the host's association), at both origins,
  `q = 1`, `q = -1` and a non-unit `q`.
- THE WHOLE MATRIX, one launch per class into one zeroed `MU`, against
  `multipole_matrix(e=2)`: the off-diagonal shell pairs' elements `==`, the
  diagonal shell pairs' within `DIAG_BOUND` eps of the component's scale
  (test_kernel_mu.py's bound and its reason).
- LIBCINT DIRECTLY: the kernel interpreted over `one_electron_launch`'s plan
  (`OneEClass.arrays(Target.QUADRUPOLE)`), and `walk_one_electron`'s
  quadrupole from the same plan, against `int1e_rr_cart` at both origins
  (`manifest.quadrupole_integrals`, on `hf_ccpvtz_spd_dipole`'s inputs),
  whole and per shell-pair block, within `LIBCINT_TOL` of the largest
  `|rr|`.
- THE ORIGIN SHIFT on the interpreted output, every term a kernel:
  `Q_de(C') = Q_de(C) + delta_d mu_e(C) + delta_e mu_d(C) + delta_d delta_e
  q S`, `delta = C - C'`, with `mu` and `q S` from the dipole kernel's
  `with_overlap` launch.

Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_kernel_mu import _diag_mask, _pairs_of, _rel, _run, _spd
from test_kernel_v import _records

from felizjueves.ao.pairs import overlap_bound
from felizjueves.emit import cuda, hip
from felizjueves.eri.j_coulomb.launch import one_electron_launch, walk_one_electron
from felizjueves.ir import AtomicAdd, If, Inline, walk
from felizjueves.mmd import e_fragment as ef
from felizjueves.one_electron.kernel_mu import (
    ORDER,
    build_mu_kernel,
    build_quad_kernel,
    mu_spec,
    n_components,
    quad_name,
    quad_spec,
)
from felizjueves.one_electron.multipole import multipole_block, multipole_matrix
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.spec import Engine, Operator, Spec, Target

LMAX = 2  # the dipole's cap: every shell l <= 2, so (ss) through (dd)
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
NCOMP = 6
EPS = 2.0 ** -52
DIAG_BOUND = 16  # x eps max|Q_k|: test_kernel_mu.py's bound
LIBCINT_TOL = 1e-12  # of max|rr|: measured 1.1e-15
SHIFT_TOL = 1e-14  # of max|Q|
ZERO = (0.0, 0.0, 0.0)
PAIRS = [(d, e) for d in range(3) for e in range(d, 3)]  # xx xy xz yy yz zz


def _zeros(basis, ncomp=NCOMP):
    n = basis.n_ao
    return np.zeros((ncomp, n, n))


def _launch(cls, basis, pairs, MUin, C, q=1.0):
    """One quadrupole launch of class `cls` over every record of `pairs`."""
    return _run(build_quad_kernel(cls), _records(basis, pairs), basis.n_ao,
                MUin, C, q)


def _host(basis, pairs, C, q=1.0):
    """`multipole_block(e=2)` placed into a fresh `MU` by `kernel_jab`'s
    rule."""
    sh, off = basis.shells, basis.offsets
    M = _zeros(basis)
    for i, j in pairs:
        blk = multipole_block(sh[i], sh[j], C, e=2, q=q)
        for k in range(NCOMP):
            for r in range(sh[i].n_cart):
                for c in range(sh[j].n_cart):
                    if i == j and c > r:
                        continue  # diagonal: the lower triangle, mirrored
                    M[k, off[i] + r, off[j] + c] = blk[k][r][c]
                    M[k, off[j] + c, off[i] + r] = blk[k][r][c]
    return M


def _classes_of(basis):
    sh = basis.shells
    return sorted({(a.l, b.l) for k, a in enumerate(sh) for b in sh[k:]})


def _whole(basis, C, q=1.0):
    M = _zeros(basis)
    for cls in _classes_of(basis):
        M = _launch(cls, basis, _pairs_of(basis, cls), M, C, q)
    return M


def _whole_dipole_with_s(basis, C, q=1.0):
    """The dipole kernel's `with_overlap` launches: `[q S, mu_x, mu_y,
    mu_z]`."""
    M = _zeros(basis, 4)
    for cls in _classes_of(basis):
        M = _run(build_mu_kernel(cls, with_overlap=True),
                 _records(basis, _pairs_of(basis, cls)), basis.n_ao, M, C, q)
    return M


def _from_plan(basis, C, q=1.0):
    plan = one_electron_launch(basis, (), tau=0.0, bound=overlap_bound)
    M = _zeros(basis)
    for c in plan:
        rec = {k: np.asarray(v, dtype=np.int32 if k.startswith("off")
                             else float)
               for k, v in c.arrays(Target.QUADRUPOLE).items()}
        M = _run(build_mu_kernel(c.spec(Target.QUADRUPOLE)), rec,
                 basis.n_ao, M, C, q)
    return M, plan


# ---------------------------------------------------------------------------
# The kernel's shape
# ---------------------------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert quad_spec(2, 1) == Spec(Operator.QUADRUPOLE, 0, (2, 1),
                                   Target.QUADRUPOLE)
    assert quad_spec(2, 1).key() == "quadrupole/d0/(2,1)/b1/quadrupole/literal"
    assert quad_name((2, 1)) == build_quad_kernel((2, 1)).name == "quad_21"
    assert build_mu_kernel(quad_spec(2, 1)).name == "quad_21"
    assert ORDER == {Target.DIPOLE: 1, Target.QUADRUPOLE: 2}
    assert n_components(e=2) == NCOMP
    assert ef.multipole_components(2) == ((2, 0, 0), (1, 1, 0), (1, 0, 1),
                                          (0, 2, 0), (0, 1, 1), (0, 0, 2))


def test_the_parameter_block_is_the_dipoles() -> None:
    q, mu = build_quad_kernel((1, 2)), build_mu_kernel((1, 2))
    assert [p.name for p in q.params] == [p.name for p in mu.params]
    assert q.params[-1].shape[0].value == NCOMP


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_one_inline_and_one_atomic_add_per_value_and_mirror(cls) -> None:
    la, lb = cls
    k = build_quad_kernel(cls)
    assert sum(isinstance(s, Inline) for s in walk(k.body)) == 1
    na, nb = len(ef.cartesians(la)), len(ef.cartesians(lb))
    ifs = [s for s in walk(k.body, into_inline=True)
           if isinstance(s, If) and s.then and isinstance(s.then[0], AtomicAdd)]
    assert len(ifs) == NCOMP * na * nb
    assert {s.then[0].dst.at[0].value for s in ifs} == set(range(NCOMP))
    if la == lb:
        assert sum(len(s.orelse) for s in ifs) == NCOMP * na * na
    else:
        assert all(s.orelse == () for s in ifs)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_both_backends_emit_every_class(cls) -> None:
    k = build_quad_kernel(cls)
    for text in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)):
        assert f"void {k.name}(" in text
        assert "half_inv_zeta" in text  # M^2_0 reads 1 / (2 zeta)


def test_the_refusals() -> None:
    with pytest.raises(ValueError, match="dipole's alone"):
        build_mu_kernel(quad_spec(1, 1), with_overlap=True)
    with pytest.raises(ValueError, match="not a Target.QUADRUPOLE key"):
        build_quad_kernel(mu_spec(1, 1))
    with pytest.raises(ValueError, match="not a Target.DIPOLE key"):
        build_quad_kernel(Spec(Operator.OVERLAP, 0, (1, 1), Target.S))
    with pytest.raises(ValueError, match="quadrupole is built on"):
        Spec(Operator.QUADRUPOLE, 0, (1, 1), Target.QUADRUPOLE,
             engine=Engine.HGP)
    with pytest.raises(ValueError):
        Spec(Operator.QUADRUPOLE, 0, (1, 1), Target.QUADRUPOLE, n_batches=3)
    with pytest.raises(ValueError):
        Spec(Operator.DIPOLE, 0, (1, 1), Target.QUADRUPOLE)


# ---------------------------------------------------------------------------
# Per class: one launch over every record of the class, bitwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_one_launch_over_the_class_equals_the_host_block(cls) -> None:
    d, basis = _spd()
    pairs = _pairs_of(basis, cls)
    for C in (d.origin, ZERO):
        got = _launch(cls, basis, pairs, _zeros(basis), C)
        assert np.array_equal(got, _host(basis, pairs, C)), tuple(C)
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
        R = np.array(multipole_matrix(basis, C, e=2))
        for k in range(NCOMP):
            assert np.array_equal(M[k][~diag], R[k][~diag])
            err = np.abs(M[k] - R[k])[diag]
            scale = np.max(np.abs(R[k]))
            worst = max(worst, float(err.max() / (EPS * scale)))
            assert np.all(err <= DIAG_BOUND * EPS * scale), (tuple(C), k)
    print(f"\nmax|Q - multipole_matrix(e=2)| on diagonal shell pairs = "
          f"{worst:.2f} eps max|Q_k|")


# ---------------------------------------------------------------------------
# libcint, directly
# ---------------------------------------------------------------------------


def test_the_plan_launched_and_walked_is_libcints_int1e_rr_cart() -> None:
    w = manifest.quadrupole_integrals()
    basis = ob.from_fixture(w.parent.fixture)
    sh, off = basis.shells, basis.offsets
    for C, want in ((w.origin, w.rr), (ZERO, w.rr0)):
        got, plan = _from_plan(basis, C)
        assert _rel(got, want) <= LIBCINT_TOL, ("kernel", tuple(C))
        walked = np.array(walk_one_electron(plan, basis, Target.QUADRUPOLE,
                                            origin=C))
        assert _rel(walked, want) <= LIBCINT_TOL, ("walk", tuple(C))
        assert np.array_equal(walked,
                              np.array(multipole_matrix(basis, C, e=2)))
        scale = float(np.max(np.abs(want)))
        for i, sa in enumerate(sh):
            for j, sb in enumerate(sh):
                blk = (slice(None), slice(off[i], off[i] + sa.n_cart),
                       slice(off[j], off[j] + sb.n_cart))
                assert np.max(np.abs(got[blk] - want[blk])) \
                    <= LIBCINT_TOL * scale, (i, j, tuple(C))


def test_the_walk_refuses_a_quadrupole_without_an_origin() -> None:
    _, basis = _spd()
    plan = one_electron_launch(basis, (), tau=0.0, bound=overlap_bound)
    with pytest.raises(ValueError, match="quadrupole needs its origin"):
        walk_one_electron(plan, basis, Target.QUADRUPOLE)


# ---------------------------------------------------------------------------
# The origin shift, on the interpreted output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [1.0, 0.75])
def test_the_origin_shift_on_the_interpreted_output(q) -> None:
    """`Q_de(C') = Q_de(C) + delta_d mu_e + delta_e mu_d + delta_d delta_e
    q S`, `delta = C - C'`, `mu` and `q S` at `C` from the dipole kernel."""
    d, basis = _spd()
    C, C2 = np.asarray(d.origin), np.array((-1.1, 0.25, 0.6))
    dl = C - C2
    dip = _whole_dipole_with_s(basis, C, q)
    qS, mu = dip[0], dip[1:]
    at_c, at_c2 = _whole(basis, C, q), _whole(basis, C2, q)
    want = np.array([at_c[k] + dl[a] * mu[b] + dl[b] * mu[a]
                     + dl[a] * dl[b] * qS for k, (a, b) in enumerate(PAIRS)])
    assert _rel(at_c2, want) <= SHIFT_TOL
