"""`ao.metric`: the two-center Coulomb metric `(P|Q)`, host-side, against libcint.

The oracle is the tracked cc-pVTZ-jkfit metric set (#411,
`manifest.metric_set`): libcint's `int2c2e_cart` and `int1e_ovlp` over
H2O/cc-pVDZ's auxiliary basis, 45 shells, 166 functions, O to g and H to f.

EVERY TEST STAYS AT `lP + lQ <= 4` (the milestone's cap, `tiers.FAST_L_MAX`):
no block of a class above it is computed here, so the basis-wide matrix is
built only over a sub-basis whose shells keep every pair inside the cap.
Within that, s/p classes are fast and anything with a `d` or higher shell is
`slow` (the issue's rule).

THE WITNESSES, in order:

1. THE NORMALISATION, first: the aux overlap `ao.overlap.overlap_block` of the
   loaded shells against the fixture's `int1e_ovlp`, every shell pair with
   `lP + lQ <= 4`. If the coefficients carried a second normalisation this is
   where it shows, as a clean per-l constant.
2. EVERY BLOCK of every class `lP + lQ <= 4` (both orders) against
   `int2c2e_cart`.
3. THE s|s CLOSED FORM, `c_a c_b 2 pi^{5/2} / (a b sqrt(a + b)) F_0`, and the
   dressing identity `(2/sqrt(pi)) theta s_a s_b / (c_a c_b) = that lambda`.
4. THE SYMMETRY `(P|Q) = (Q|P)^T` across every swapped class.
5. THE STAGING (`metric_classes`): every pair once, `lP <= lQ`, the swap flag
   and the scatter's transpose, each shell's records `aux_records`', the
   staged block `metric_block` of the staged shells bitwise.

THE TOLERANCES, stated and not tuned, all relative to a scale:

- `TOL = 1e-12` of the MATRIX scale `max |M|` (the fixture's whole
  `int2c2e_cart`, ~96.5; for the overlap, `max |S|`): `test_ao_j.py`'s and
  `test_ao_schwarz.py`'s factor for the same Hermite arithmetic. Measured
  worst at writing: 2.6e-14 for `(P|Q)`, 8.0e-16 for the overlap.
- and the same `1e-12` of the BLOCK scale, the largest element of
  `metric_block_and_scale`'s absolute-value twin, so that a small (diffuse
  or distant) block is not hidden under the matrix's largest element.
  Measured worst at writing: 1.4e-13.
- The s|s closed form, which has no cancellation, element by element at
  `1e-13` relative (test_oracle_metric_fixture.py's bound for the same form).
- The dressing identity at `4 eps` relative (a handful of roundings).
- The symmetry at `TOL` of the matrix scale: `(Q|P)` evaluates `[t|tau]` at
  `-R`, a different rounding, equal exactly, not bitwise.

Fast tier unless marked: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import FAST_L_MAX, by_l

from felizjueves.ao.shell import Basis, cartesians
from felizjueves.eri.density_fit.metric import (
    MetricClass,
    aux_dress,
    aux_records,
    center_e,
    class_block,
    metric_block,
    metric_block_and_scale,
    metric_classes,
    metric_matrix,
    place_staged,
    scale_matrix,
)
from felizjueves.mmd.hermite import hermite_order
from felizjueves.one_electron.overlap import overlap_block
from felizjueves.reference.pq import fundamental, pq

SET = "h2o_ccpvtz_jkfit_metric"
TOL = 1e-12  # x the matrix scale, and x the block scale (module docstring)
SS_TOL = 1e-13  # relative, per element, the s|s closed form
EPS = np.finfo(float).eps
L_CAP = FAST_L_MAX  # 4: nothing above lP + lQ = 4 is computed here
# The nine classes lP <= lQ at L <= 4; s/p fast, d+ slow (the issue's rule).
CLASSES = [(a, b) for a in range(L_CAP + 1) for b in range(a, L_CAP + 1)
           if a + b <= L_CAP]
FAST_CLASSES = {(0, 0), (0, 1), (1, 1)}


def _param(c):
    # d+ is `slow` per the issue; by_l marks L > 4 too, which never occurs.
    p = by_l(c, sum(c), id=f"{'spdfg'[c[0]]}|{'spdfg'[c[1]]}")
    if c in FAST_CLASSES:
        return p
    return pytest.param(*p.values, id=p.id, marks=(*p.marks, pytest.mark.slow))


@cache
def _ms() -> manifest.MetricSet:
    return manifest.metric_set(SET)


@cache
def _aux() -> Basis:
    return ob.from_fixture(_ms())


def _scale_m() -> float:
    return float(np.max(np.abs(_ms().metric)))


def _sub(a: np.ndarray, i: int, j: int) -> np.ndarray:
    aux = _aux()
    off, sh = aux.offsets, aux.shells
    return a[off[i]:off[i] + sh[i].n_cart, off[j]:off[j] + sh[j].n_cart]


def _pairs(cls: tuple[int, int]) -> list[tuple[int, int]]:
    """Every ORDERED shell pair (i, j) whose (l_i, l_j) sorts to `cls`."""
    sh = _aux().shells
    return [(i, j) for i in range(len(sh)) for j in range(len(sh))
            if tuple(sorted((sh[i].l, sh[j].l))) == cls]


def test_the_classes_are_the_nine_and_the_basis_reaches_them():
    assert CLASSES == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1),
                       (1, 2), (1, 3), (2, 2)]
    for c in CLASSES:
        assert _pairs(c), c


# --- 1. the normalisation witness: the aux overlap --------------------------


@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES])
def test_the_aux_overlap_is_libcints(cls):
    S = _ms().ovlp
    scale = float(np.max(np.abs(S)))
    worst = 0.0
    for i, j in _pairs(cls):
        sh = _aux().shells
        got = np.array(overlap_block(sh[i], sh[j]))
        worst = max(worst, float(np.max(np.abs(got - _sub(S, i, j)))) / scale)
    print(f"aux overlap {cls}: worst {worst:.2e} of max|S|")
    assert worst <= TOL, worst


# --- 2. every block of every class against int2c2e_cart ---------------------


@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES])
def test_every_block_is_libcints_int2c2e(cls):
    M = _ms().metric
    sm = _scale_m()
    sh = _aux().shells
    worst_m = worst_b = 0.0
    for i, j in _pairs(cls):
        blk, sc = metric_block_and_scale(sh[i], sh[j], pq=pq)
        blk, sc = np.array(blk), np.array(sc)
        err = float(np.max(np.abs(blk - _sub(M, i, j))))
        assert blk.shape == (sh[i].n_cart, sh[j].n_cart)
        assert np.all(sc >= np.abs(blk))
        assert np.array_equal(np.array(metric_block(sh[i], sh[j], pq=pq)),
                              blk)
        worst_m = max(worst_m, err / sm)
        if not np.any(sc):
            # one centre and an odd total parity (s|p, p|d, s|f on one
            # atom): every Hermite term vanishes, so the block is exactly 0
            assert not np.any(blk) and sh[i].centre == sh[j].centre
            continue
        worst_b = max(worst_b, err / float(np.max(sc)))
    print(f"(P|Q) {cls}: worst {worst_m:.2e} of max|M|, "
          f"{worst_b:.2e} of the block scale")
    assert worst_m <= TOL, worst_m
    assert worst_b <= TOL, worst_b


def _l_le(lmax: int) -> tuple[Basis, list[int]]:
    keep = [k for k, s in enumerate(_aux().shells) if s.l <= lmax]
    return Basis(tuple(_aux().shells[k] for k in keep)), keep


def _fixture_rows(keep: list[int]) -> list[int]:
    aux = _aux()
    return [aux.offsets[k] + a for k in keep
            for a in range(aux.shells[k].n_cart)]


@pytest.mark.parametrize("lmax", [pytest.param(1, id="sp"),
                                  pytest.param(2, id="spd",
                                               marks=pytest.mark.slow)])
def test_the_matrix_over_a_capped_sub_basis_is_libcints(lmax):
    """`metric_matrix` over the aux shells with `l <= lmax` (every pair then
    at `lP + lQ <= 2 lmax <= 4`) against the same rows and columns of the
    fixture; exactly symmetric, and its scale bounds it."""
    sub, keep = _l_le(lmax)
    assert 2 * lmax <= L_CAP
    rows = _fixture_rows(keep)
    want = _ms().metric[np.ix_(rows, rows)]
    got = np.array(metric_matrix(sub, pq=pq))
    assert got.shape == want.shape == (sub.n_ao, sub.n_ao)
    assert np.array_equal(got, got.T)
    scale = np.array(scale_matrix(sub, pq=pq))
    assert np.all(scale >= np.abs(got))
    worst = float(np.max(np.abs(got - want))) / _scale_m()
    print(f"metric_matrix l <= {lmax}: worst {worst:.2e} of max|M|")
    assert worst <= TOL, worst
    assert np.linalg.eigvalsh(got).min() > 0


# --- 3. the s|s closed form and the dressing --------------------------------


def _boys0(t: float) -> float:
    return 1.0 if t == 0.0 else 0.5 * math.sqrt(math.pi / t) * math.erf(
        math.sqrt(t))


def test_every_s_s_block_is_the_closed_form():
    sh = _aux().shells
    worst = 0.0
    for i, j in _pairs((0, 0)):
        si, sj = sh[i], sh[j]
        r2 = sum((x - y) ** 2 for x, y in zip(si.centre, sj.centre,
                                              strict=True))
        want = 0.0
        for a, ca in zip(si.exponents, si.coefficients, strict=True):
            for b, cb in zip(sj.exponents, sj.coefficients, strict=True):
                want += (ca * cb * 2.0 * math.pi ** 2.5
                         / (a * b * math.sqrt(a + b))
                         * _boys0(a * b / (a + b) * r2))
        got = metric_block(si, sj, pq=pq)[0][0]
        worst = max(worst, abs(got - want) / abs(want))
    assert worst <= SS_TOL, worst


@pytest.mark.parametrize(("a", "b"), [(0.3, 7.0), (1.0, 1.0), (40.0, 0.05)])
def test_the_dressing_times_the_prelude_is_lambda(a, b):
    """`s = c (pi/alpha)^{3/2}` on each side and `(2/sqrt(pi)) theta` in the
    evaluator's `[0]^(0)` make McMurchie & Davidson's lambda."""
    theta = math.sqrt(a * b / (a + b))
    got = aux_dress(1.0, alpha=a) * aux_dress(1.0, alpha=b) * fundamental(
        0, (0.0, 0.0, 0.0), theta)
    want = 2.0 * math.pi ** 2.5 / (a * b * math.sqrt(a + b))
    assert abs(got - want) <= 4 * EPS * want


def test_the_record_is_one_per_primitive_and_carries_no_normalisation():
    for s in _aux().shells:
        recs = aux_records(s)
        assert [r.i for r in recs] == list(range(len(s.exponents)))
        for r, a, c in zip(recs, s.exponents, s.coefficients, strict=True):
            assert r.A == s.centre and r.alpha == a
            assert r.s == c * (math.pi / a) ** 1.5  # bitwise: aux_dress's
    with pytest.raises(ValueError):
        aux_dress(1.0, alpha=0.0)


@pytest.mark.parametrize("l", range(L_CAP + 1))
def test_the_center_e_is_the_x_pa_zero_recurrence(l):  # noqa: E741
    """`E^{i+1}_t = (1/2a) E^i_{t-1} + (t+1) E^i_{t+1}`, `E^0_0 = 1`, per
    axis, and the odd-parity entries exactly zero."""
    alpha = 0.7

    def axis(i: int, t: int) -> float:
        if t < 0 or t > i:
            return 0.0
        if i == 0:
            return 1.0
        return 0.5 / alpha * axis(i - 1, t - 1) + (t + 1) * axis(i - 1, t + 1)

    e = center_e(l, alpha)
    for a, row in zip(cartesians(l), e, strict=True):
        for t, v in zip(hermite_order(l), row, strict=True):
            want = axis(a[0], t[0]) * axis(a[1], t[1]) * axis(a[2], t[2])
            assert v == pytest.approx(want, rel=4 * EPS, abs=0.0)
            if any((a[d] - t[d]) % 2 for d in range(3)):
                assert v == 0.0


# --- 4. the symmetry across a swapped class ---------------------------------


@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES if c[0] < c[1]])
def test_the_swapped_block_is_the_transpose(cls):
    sh = _aux().shells
    sm = _scale_m()
    worst = 0.0
    for i, j in _pairs(cls):
        if sh[i].l > sh[j].l:
            continue
        pq_ = np.array(metric_block(sh[i], sh[j], pq=pq))
        qp = np.array(metric_block(sh[j], sh[i], pq=pq))
        assert qp.shape == pq_.T.shape
        worst = max(worst, float(np.max(np.abs(qp - pq_.T))) / sm)
    assert worst <= TOL, worst


# --- 5. the staging ----------------------------------------------------------


def test_every_pair_is_staged_once_lp_le_lq_with_its_swap():
    aux = _aux()
    sh, off = aux.shells, aux.offsets
    classes = metric_classes(aux)
    assert [(c.lP, c.lQ) for c in classes] == sorted(
        {tuple(sorted((a.l, b.l))) for a in sh for b in sh})
    seen = []
    for c in classes:
        assert c.lP <= c.lQ
        assert len({len(getattr(c, k)) for k in (
            "pairs", "swapped", *MetricClass.PAIR_ARRAYS)}) == 1
        assert c.n_rec == len(c.A) == len(c.s) == c.rec1[-1]
        assert len(set(c.shells)) == len(c.shells)  # each shell once
        # the record table is the shells' records, contiguous, in order
        assert c.rec0[0] == 0 and all(
            c.rec1[k] == c.rec0[k + 1] for k in range(len(c.shells) - 1))
        for n, r0, r1 in zip(c.shells, c.rec0, c.rec1, strict=True):
            assert [(c.A[r], c.alpha[r], c.s[r]) for r in range(r0, r1)] == [
                (r.A, r.alpha, r.s) for r in aux_records(sh[n])]
        for k, (p, q) in enumerate(c.pairs):
            assert (sh[p].l, sh[q].l) == (c.lP, c.lQ)
            i, j = (q, p) if c.swapped[k] else (p, q)
            assert i <= j
            assert c.swapped[k] == (sh[i].l > sh[j].l)
            assert (c.off_p[k], c.off_q[k]) == (off[p], off[q])
            P, Q = c.records(k)
            assert P == aux_records(sh[p]) and Q == aux_records(sh[q])
            seen.append((i, j))
    n = len(sh)
    assert sorted(seen) == [(i, j) for i in range(n) for j in range(i, n)]
    assert any(any(c.swapped) for c in classes)  # the basis has swaps
    arrays = classes[0].arrays()
    assert list(arrays) == [*MetricClass.RECORD_ARRAYS,
                            *MetricClass.PAIR_ARRAYS]


def test_the_staged_scatter_transposes_a_swap():
    """`place_staged` with a marker block: a swapped pair's element (a, b)
    lands at basis-order `[off_q + b][off_p + a]`, and every element is
    mirrored. Pure placement: no integral computed."""
    aux = _aux()
    for c in metric_classes(aux):
        for k, (p, q) in enumerate(c.pairs):
            np_, nq = len(cartesians(c.lP)), len(cartesians(c.lQ))
            blk = [[1.0 + a * nq + b for b in range(nq)] for a in range(np_)]
            out = [[0.0] * aux.n_ao for _ in range(aux.n_ao)]
            place_staged(c, k, blk, out)
            o = np.array(out)
            assert np.array_equal(o, o.T)
            op, oq = c.off_p[k], c.off_q[k]
            for a in range(np_):
                for b in range(nq):
                    if p == q and b > a:
                        continue
                    r, col = ((oq + b, op + a) if c.swapped[k]
                              else (op + a, oq + b))
                    assert o[r][col] == blk[a][b]
                    if p != q:  # the basis order's upper triangle
                        assert r < col


@pytest.mark.parametrize("lmax", [pytest.param(1, id="sp"),
                                  pytest.param(2, id="spd",
                                               marks=pytest.mark.slow)])
def test_the_staged_build_is_the_matrix(lmax):
    """Every staged pair's `class_block` (from the class's records) is
    `metric_block` of the staged shells bitwise; placed by `place_staged`,
    the whole is libcint's over the capped sub-basis."""
    sub, keep = _l_le(lmax)
    out = [[0.0] * sub.n_ao for _ in range(sub.n_ao)]
    swapped = 0
    for c in metric_classes(sub):
        assert c.lP + c.lQ <= L_CAP
        for k, (p, q) in enumerate(c.pairs):
            blk = class_block(c, k, pq=pq)
            assert blk == metric_block(sub.shells[p], sub.shells[q], pq=pq)
            place_staged(c, k, blk, out)
            swapped += c.swapped[k]
    assert swapped
    rows = _fixture_rows(keep)
    want = _ms().metric[np.ix_(rows, rows)]
    got = np.array(out)
    assert np.array_equal(got, got.T)
    worst = float(np.max(np.abs(got - want))) / _scale_m()
    assert worst <= TOL, worst
