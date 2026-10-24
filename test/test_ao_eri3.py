"""`eri.density_fit.eri3`: the three-center integral `(mn|P)`, host-side, against
libcint.

The oracle is the tracked schema-4 set `h2o_sto3g_jkfit_3c` (#442,
`manifest.three_center_set`): libcint's `int3c2e_cart` over H2O/STO-3G's 7
AOs (s and p only) and its cc-pVTZ-jkfit auxiliary basis (45 shells, 166
functions, O to g, H to f), `[m][n][P]`.

EVERY TEST STAYS AT `la + lb + lP <= 4` (`tiers.FAST_L_MAX`): no block of a
class above it is computed here, so the whole tensor is built only over an
aux sub-basis that keeps every triple inside the cap (the bra reaches p, so
aux `l <= 2`). Within that, a class with an s or p aux shell is fast and one
with a d or higher aux shell is `slow` (the issue's rule; the bra is s/p
throughout).

THE WITNESSES, in order:

1. EVERY BLOCK of every class `L <= 4`, both bra orders, against
   `int3c2e_cart`.
2. THE WHOLE TENSOR over the capped aux sub-basis against the same slice of
   the fixture, and its scale bounding it.
3. THE (ss|s) CLOSED FORM (derived in its test's docstring).
4. THE SYMMETRY: `eri3_tensor` is `(m, n)`-symmetric EXACTLY (the mirror is
   applied), and the block `(ba|P)` is `(ab|P)` transposed to rounding (its
   `E^{ba}` rounds differently from `E^{ab}`; libcint's own s1 tensor is
   symmetric to 4.4e-16 only, test_oracle_three_center_fixture.py).
5. THE STAGING (`eri3_classes`): every (AO pair `i <= j`, aux shell) cell
   exactly once, `la <= lb`, the swap flag and the scatter's transpose, the
   records `primitive_pairs`' and `aux_records`', and the staged block
   `eri3_block` of the staged shells bitwise.

THE INTERNAL TWIN IS NOT BUILT. `eri.eri`'s quartet `(ab|c0)` with `0` an s
shell of exponent 0 and coefficient 1 would be this integral algebraically
(`eta = alpha`, `Q = A_P`, `K_CD = 1`), but `ao.shell.Shell` refuses an
exponent that is not > 0, and #443 says to drop the check rather than widen
`shell`; the refusal itself is pinned below, so a `Shell` that starts
accepting 0 makes the twin constructible and this file says so.

THE TOLERANCES, stated and not tuned, all relative to a scale:

- `TOL = 1e-12` of the TENSOR scale `max |(mn|P)|` of the fixture (the
  issue's bound; `test_ao_metric.py`'s factor for the same Hermite
  arithmetic), and the same `1e-12` of the BLOCK scale (the largest element
  of `eri3_block_and_scale`'s absolute-value twin), so that a small block is
  not hidden under the tensor's largest element -- over the elements that
  are not zero by parity on one centre (`_parity_zero`: there the bra's
  `P_ij` is the centre only to rounding, so the computed element is input
  noise the twin cannot bound; libcint's is exactly 0, asserted, and ours
  is held to the tensor-scale bound like every other element).
- The (ss|s) closed form, which has no cancellation, element by element at
  `1e-13` relative (`test_ao_metric.py`'s bound for its s|s form).
- The block swap at `TOL` of the tensor scale (equal exactly, not bitwise).

Fast tier unless marked: no compiler, no card, no PySCF.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import FAST_L_MAX

from felizjueves.ao.pairs import dress, primitive_pairs
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.eri.density_fit.eri3 import (
    Eri3Class,
    class_block,
    eri3_block,
    eri3_block_and_scale,
    eri3_classes,
    eri3_tensor,
    place_staged,
    scale_tensor,
)
from felizjueves.eri.density_fit.metric import aux_records
from felizjueves.reference.pq import pq

SET = "h2o_sto3g_jkfit_3c"
TOL = 1e-12  # x the tensor scale, and x the block scale (module docstring)
SS_TOL = 1e-13  # relative, per element, the (ss|s) closed form
L_CAP = FAST_L_MAX  # 4: nothing above la + lb + lP = 4 is computed here
# Every class la <= lb <= 1 (the bra is s/p) with L <= 4.
CLASSES = [(a, b, p) for a in range(2) for b in range(a, 2)
           for p in range(L_CAP + 1) if a + b + p <= L_CAP]


def _id(c: tuple[int, int, int]) -> str:
    return f"{'spdfg'[c[0]]}{'spdfg'[c[1]]}|{'spdfg'[c[2]]}"


def _param(c: tuple[int, int, int]):
    # an s/p aux shell is fast, d and above slow (the issue's rule)
    marks = () if c[2] <= 1 else (pytest.mark.slow,)
    return pytest.param(c, id=_id(c), marks=marks)


@cache
def _ts() -> manifest.ThreeCenterSet:
    return manifest.three_center_set(SET)


@cache
def _basis() -> Basis:
    return ob.from_fixture(_ts().parent)


@cache
def _aux() -> Basis:
    return ob.from_fixture(_ts())


def _scale_t() -> float:
    return float(np.max(np.abs(_ts().tensor)))


def _sub(i: int, j: int, k: int) -> np.ndarray:
    b, x = _basis(), _aux()
    oi, oj, ok = b.offsets[i], b.offsets[j], x.offsets[k]
    return _ts().tensor[oi:oi + b.shells[i].n_cart, oj:oj + b.shells[j].n_cart,
                        ok:ok + x.shells[k].n_cart]


def _triples(cls: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Every ORDERED bra pair (i, j) whose sorted l's are `cls[:2]`, crossed
    with every aux shell of `l = cls[2]`."""
    sh, ash = _basis().shells, _aux().shells
    return [(i, j, k) for i in range(len(sh)) for j in range(len(sh))
            if tuple(sorted((sh[i].l, sh[j].l))) == cls[:2]
            for k in range(len(ash)) if ash[k].l == cls[2]]


def test_the_classes_and_the_bases_reach_them():
    assert CLASSES == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
                       (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
                       (1, 1, 0), (1, 1, 1), (1, 1, 2)]
    assert _basis().n_ao == _ts().tensor.shape[0] == 7
    assert _aux().n_ao == _ts().tensor.shape[2] == 166
    for c in CLASSES:
        assert _triples(c), c


# --- 1. every block of every class against int3c2e_cart ---------------------


def _parity_zero(sA: Shell, sB: Shell, sP: Shell) -> np.ndarray:
    """The elements `[a][b][q]` that are 0 in exact arithmetic: all three
    centres one point and some axis with an odd `a_d + b_d + q_d` (the
    integrand is odd along it about that point). The block-scale bound does
    not hold there: the bra's `P_ij = (a A + b A)/(a + b)` is `A` only to
    rounding, so `R = P_ij - A_k` is rounding noise the |.| twin cannot see,
    and the computed element is that noise (worst seen ~1e-16 absolute)."""
    out = np.zeros((sA.n_cart, sB.n_cart, sP.n_cart), dtype=bool)
    if not sA.centre == sB.centre == sP.centre:
        return out
    for x, a in enumerate(cartesians(sA.l)):
        for y, b in enumerate(cartesians(sB.l)):
            for z, q in enumerate(cartesians(sP.l)):
                out[x, y, z] = any((a[d] + b[d] + q[d]) % 2 for d in range(3))
    return out


@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES])
def test_every_block_is_libcints_int3c2e(cls):
    st = _scale_t()
    sh, ash = _basis().shells, _aux().shells
    worst_t = worst_b = 0.0
    zero = 0
    for i, j, k in _triples(cls):
        blk, sc = eri3_block_and_scale(sh[i], sh[j], ash[k], pq=pq)
        blk, sc = np.array(blk), np.array(sc)
        assert blk.shape == (sh[i].n_cart, sh[j].n_cart, ash[k].n_cart)
        assert np.all(sc >= np.abs(blk))
        assert np.array_equal(
            np.array(eri3_block(sh[i], sh[j], ash[k], pq=pq)), blk)
        want = _sub(i, j, k)
        diff = np.abs(blk - want)
        worst_t = max(worst_t, float(np.max(diff)) / st)
        live = ~_parity_zero(sh[i], sh[j], ash[k])
        # a parity-zero element is 0 in exact arithmetic and libcint's is 0
        assert not np.any(want[~live])
        zero += int(np.count_nonzero(~live))
        if np.any(live) and np.any(sc[live]):
            worst_b = max(worst_b, float(np.max(diff[live]))
                          / float(np.max(sc[live])))
    print(f"(mn|P) {_id(cls)}: worst {worst_t:.2e} of max|(mn|P)|, "
          f"{worst_b:.2e} of the block scale, {zero} parity-zero elements")
    assert worst_t <= TOL, worst_t
    assert worst_b <= TOL, worst_b


# --- 2. the whole tensor over a capped aux sub-basis ------------------------


def _aux_le(lmax: int) -> tuple[Basis, list[int]]:
    keep = [k for k, s in enumerate(_aux().shells) if s.l <= lmax]
    return Basis(tuple(_aux().shells[k] for k in keep)), keep


def _aux_cols(keep: list[int]) -> list[int]:
    aux = _aux()
    return [aux.offsets[k] + q for k in keep
            for q in range(aux.shells[k].n_cart)]


_LMAX = [pytest.param(1, id="sp"),
         pytest.param(2, id="spd", marks=pytest.mark.slow)]


@pytest.mark.parametrize("lmax", _LMAX)
def test_the_tensor_over_a_capped_aux_basis_is_libcints(lmax):
    """`eri3_tensor` over the aux shells with `l <= lmax` (every triple then
    at `L <= 2 + lmax <= 4`) against the same columns of the fixture;
    `(m, n)`-symmetric exactly, `[m][n][P]`, and its scale bounds it."""
    sub, keep = _aux_le(lmax)
    assert 2 * max(s.l for s in _basis().shells) + lmax <= L_CAP
    want = _ts().tensor[:, :, _aux_cols(keep)]
    got = np.array(eri3_tensor(_basis(), sub, pq=pq))
    assert got.shape == want.shape == (_basis().n_ao, _basis().n_ao,
                                       sub.n_ao)
    assert np.array_equal(got, got.transpose(1, 0, 2))
    scale = np.array(scale_tensor(_basis(), sub, pq=pq))
    assert np.all(scale >= np.abs(got))
    worst = float(np.max(np.abs(got - want))) / _scale_t()
    print(f"eri3_tensor aux l <= {lmax}: worst {worst:.2e} of max|(mn|P)|")
    assert worst <= TOL, worst


# --- 3. the (ss|s) closed form ----------------------------------------------


def _boys0(t: float) -> float:
    return 1.0 if t == 0.0 else 0.5 * math.sqrt(math.pi / t) * math.erf(
        math.sqrt(t))


def test_every_ss_s_block_is_the_closed_form():
    """For s primitives `a` on `A`, `b` on `B` and an aux `k` on `C`, the
    Gaussian product theorem gives `exp(-a r_A^2) exp(-b r_B^2) = K_AB
    exp(-p r_P^2)`, `p = a + b`, `P = (a A + b B)/p`, `K_AB = exp(-(a b/p)
    |A - B|^2)`; the Coulomb integral of two s Gaussians of exponents `p`
    and `k` a distance `R` apart is `2 pi^{5/2} / (p k sqrt(p + k))
    F_0(p k / (p + k) R^2)` (McMurchie & Davidson's `lambda`, the Boys
    function's definition). So

        (ss|s) = sum c_a c_b c_k K_AB 2 pi^{5/2} / (p k sqrt(p + k))
                 F_0(p k / (p + k) |P - C|^2),

    the coefficients multiplying unnormalised primitives (`ao/shell.py`).
    `F_0(T) = (1/2) sqrt(pi/T) erf(sqrt T)`, `F_0(0) = 1`."""
    sh, ash = _basis().shells, _aux().shells
    worst = 0.0
    n = 0
    for i, j, k in _triples((0, 0, 0)):
        sa, sb, sc = sh[i], sh[j], ash[k]
        ab2 = sum((x - y) ** 2 for x, y in zip(sa.centre, sb.centre,
                                               strict=True))
        want = 0.0
        for a, ca in zip(sa.exponents, sa.coefficients, strict=True):
            for b, cb in zip(sb.exponents, sb.coefficients, strict=True):
                p = a + b
                P = [(a * x + b * y) / p
                     for x, y in zip(sa.centre, sb.centre, strict=True)]
                kab = math.exp(-a * b / p * ab2)
                pc2 = sum((x - y) ** 2 for x, y in zip(P, sc.centre,
                                                       strict=True))
                for g, cg in zip(sc.exponents, sc.coefficients, strict=True):
                    want += (ca * cb * cg * kab * 2.0 * math.pi ** 2.5
                             / (p * g * math.sqrt(p + g))
                             * _boys0(p * g / (p + g) * pc2))
        got = eri3_block(sa, sb, sc, pq=pq)[0][0][0]
        worst = max(worst, abs(got - want) / abs(want))
        n += 1
    n_s = sum(1 for s in sh if s.l == 0)
    assert n == n_s * n_s * sum(1 for s in ash if s.l == 0)
    assert worst <= SS_TOL, worst


# --- 4. the (m, n) symmetry -------------------------------------------------


@pytest.mark.parametrize("cls", [_param(c) for c in CLASSES])
def test_the_swapped_bra_block_is_the_transpose(cls):
    """`(ba|P)` against `(ab|P)` with `m, n` transposed, every class (the
    diagonal classes too: a pair `i != j` of equal `l`)."""
    sh, ash = _basis().shells, _aux().shells
    worst = 0.0
    for i, j, k in _triples(cls):
        if i >= j:
            continue
        ab = np.array(eri3_block(sh[i], sh[j], ash[k], pq=pq))
        ba = np.array(eri3_block(sh[j], sh[i], ash[k], pq=pq))
        assert ba.shape == ab.transpose(1, 0, 2).shape
        worst = max(worst, float(np.max(np.abs(ba - ab.transpose(1, 0, 2)))))
    assert worst / _scale_t() <= TOL, worst


# --- 5. the staging ---------------------------------------------------------


def test_every_cell_is_staged_once_la_le_lb_with_its_swap():
    basis, aux = _basis(), _aux()
    sh, off, ash, aoff = (basis.shells, basis.offsets, aux.shells,
                          aux.offsets)
    classes = eri3_classes(basis, aux)
    assert [(c.la, c.lb, c.lP) for c in classes] == sorted(
        {(*sorted((a.l, b.l)), p.l) for a in sh for b in sh for p in ash})
    seen = []
    for c in classes:
        assert c.la <= c.lb
        assert len({len(getattr(c, k)) for k in (
            "cells", "swapped", *Eri3Class.CELL_ARRAYS)}) == 1
        assert c.n_bra_rec == len(c.P) == len(c.s_ab) == c.bra_rec1[-1]
        assert c.n_aux_rec == len(c.A) == len(c.s) == c.aux_rec1[-1]
        # each table holds every entry once, contiguous, in order
        for owners, r0s, r1s in ((c.bra_pairs, c.bra_rec0, c.bra_rec1),
                                 (c.aux_shells, c.aux_rec0, c.aux_rec1)):
            assert len(set(owners)) == len(owners)
            assert r0s[0] == 0 and all(r1s[k] == r0s[k + 1]
                                       for k in range(len(owners) - 1))
        for (a, b), r0, r1 in zip(c.bra_pairs, c.bra_rec0, c.bra_rec1,
                                  strict=True):
            pps = primitive_pairs(sh[a], sh[b])
            assert [(c.zeta_ab[r], c.P[r]) for r in range(r0, r1)] == [
                (pp.zeta, pp.P) for pp in pps]
            assert [c.s_ab[r] for r in range(r0, r1)] == [
                dress((1.0,), ca=sh[a].coefficients[pp.i],
                      cb=sh[b].coefficients[pp.j], zeta=pp.zeta)[0]
                for pp in pps]
        for n, r0, r1 in zip(c.aux_shells, c.aux_rec0, c.aux_rec1,
                             strict=True):
            assert [(c.A[r], c.alpha[r], c.s[r]) for r in range(r0, r1)] == [
                (r.A, r.alpha, r.s) for r in aux_records(ash[n])]
        for k, (a, b, n) in enumerate(c.cells):
            assert (sh[a].l, sh[b].l, ash[n].l) == (c.la, c.lb, c.lP)
            i, j = (b, a) if c.swapped[k] else (a, b)
            assert i <= j
            assert c.swapped[k] == (sh[i].l > sh[j].l)
            assert (c.off_a[k], c.off_b[k], c.off_p[k]) == (off[a], off[b],
                                                            aoff[n])
            bras, auxs = c.records(k)
            assert auxs == aux_records(ash[n])
            assert [(x.P, x.zeta, x.block) for x in bras] == [
                (x.P, x.zeta, x.block) for x in primitive_pairs(sh[a], sh[b])]
            seen.append((i, j, n))
    ns, na = len(sh), len(ash)
    assert sorted(seen) == [(i, j, n) for i in range(ns)
                            for j in range(i, ns) for n in range(na)]
    assert any(any(c.swapped) for c in classes)  # the basis has swaps
    assert list(classes[0].arrays()) == [*Eri3Class.BRA_ARRAYS,
                                         *Eri3Class.AUX_ARRAYS,
                                         *Eri3Class.CELL_ARRAYS]


def test_the_staged_scatter_transposes_a_swap():
    """`place_staged` with a marker block: a swapped cell's element (a, b, q)
    lands at basis-order `[off_b + b][off_a + a][off_p + q]`, mirrored in
    `(m, n)`. Pure placement: no integral computed."""
    basis, aux = _basis(), _aux()
    n, naux = basis.n_ao, aux.n_ao
    for c in eri3_classes(basis, aux):
        na, nb, nq = (len(cartesians(c.la)), len(cartesians(c.lb)),
                      len(cartesians(c.lP)))
        blk = [[[1.0 + (a * nb + b) * nq + q for q in range(nq)]
                for b in range(nb)] for a in range(na)]
        for k in range(0, c.n_cell, 7):  # a sample: placement only
            pa, pb, _ = c.cells[k]
            out = [[[0.0] * naux for _ in range(n)] for _ in range(n)]
            place_staged(c, k, blk, out)
            o = np.array(out)
            assert np.array_equal(o, o.transpose(1, 0, 2))
            oa, ob, op = c.off_a[k], c.off_b[k], c.off_p[k]
            for a in range(na):
                for b in range(nb):
                    if pa == pb and b > a:
                        continue
                    m, nn = ((ob + b, oa + a) if c.swapped[k]
                             else (oa + a, ob + b))
                    if pa != pb:  # the basis order's upper triangle
                        assert m < nn
                    for q in range(nq):
                        assert o[m][nn][op + q] == blk[a][b][q]


@pytest.mark.parametrize("lmax", _LMAX)
def test_the_staged_build_is_the_tensor(lmax):
    """Every staged cell's `class_block` (from the class's records) is
    `eri3_block` of the staged shells bitwise; placed by `place_staged`, the
    whole is libcint's over the capped aux sub-basis, and equal to
    `eri3_tensor` bitwise wherever no swap was staged."""
    basis = _basis()
    sub, keep = _aux_le(lmax)
    out = [[[0.0] * sub.n_ao for _ in range(basis.n_ao)]
           for _ in range(basis.n_ao)]
    swapped = 0
    for c in eri3_classes(basis, sub):
        assert c.la + c.lb + c.lP <= L_CAP
        for k, (a, b, n) in enumerate(c.cells):
            blk = class_block(c, k, pq=pq)
            assert blk == eri3_block(basis.shells[a], basis.shells[b],
                                     sub.shells[n], pq=pq)
            place_staged(c, k, blk, out)
            swapped += c.swapped[k]
    assert swapped
    got = np.array(out)
    assert np.array_equal(got, got.transpose(1, 0, 2))
    want = _ts().tensor[:, :, _aux_cols(keep)]
    worst = float(np.max(np.abs(got - want))) / _scale_t()
    assert worst <= TOL, worst
    # no bra pair of this basis with i < j has l_i > l_j except (p, s) ones;
    # every unswapped element is eri3_tensor's arithmetic exactly
    ref = np.array(eri3_tensor(basis, sub, pq=pq))
    sh, off = basis.shells, basis.offsets
    for i in range(len(sh)):
        for j in range(i, len(sh)):
            if sh[i].l > sh[j].l:
                continue
            si = slice(off[i], off[i] + sh[i].n_cart)
            sj = slice(off[j], off[j] + sh[j].n_cart)
            assert np.array_equal(got[si, sj], ref[si, sj])


# --- the internal twin: not constructible -----------------------------------


def test_the_internal_twin_is_refused_by_shell():
    """`(ab|c0)` with `0` an exponent-0 s shell would be `(ab|c)` (module
    docstring), but `Shell` refuses the exponent, so the twin is dropped
    (#443). If this starts passing a zero exponent, build the twin."""
    with pytest.raises(ValueError, match="> 0"):
        Shell(0, (0.0, 0.0, 0.0), (0.0,), (1.0,))
