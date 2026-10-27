"""Second nuclear derivatives of S, T, V and (ab|cd) by McMurchie-Davidson, and
of the nuclear repulsion.

Built on the gradient spike (``experimental/mmd/gradient``): its ``mmd.py``
(shells, ``E``, ``Pair``, ``d_centre``, Boys, ``R``) and ``ints.py``'s
8-fold scatter are reused unchanged. Every second derivative is formed the way
the first ones are there: ``d/dA_k`` of a Cartesian Gaussian is
``2a g_{i+1} - i g_{i-1}`` (``mmd.d_centre``), applied twice, so a second
derivative integral is up to four ordinary integrals with the functions' ``l``
moved by two in all, read off the same ``E`` and ``R`` tables. ``V``'s
operator moves with its nucleus: ``d/dC_k R_tuv(p, P - C) = -R_{tuv+e_k}``,
so ``d2/dC_k dC_l`` is ``+R_{tuv+e_k+e_l}`` and a mixed centre-nucleus term is
one function derivative times one ``R`` shift.

No translational invariance is used: every pair of centres (and nuclei, for
``V``) is computed, so ``sum_A d2X/dR_A dR_B = 0`` is a check (``check.py``),
not an assumption. The ``(A, B)`` block of two different centres is formed
once and written to both ``[A, k, B, l]`` and ``[B, l, A, k]``, so the
integral tensors are symmetric by construction; the hessian's response part
is not, and its symmetry is checked.

Second-derivative arrays are ``d2X[A, k, B, l, ...]``: d/dR_{A,k} d/dR_{B,l}
of the whole integral ``X``, every centre on each atom summed.
"""

from __future__ import annotations

import functools
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradient"))

from ints import _scatter  # noqa: E402
from mmd import Pair, ao_offsets, cart, d_centre, hermite_r, shift  # noqa: E402


def _d(expo, v, k):
    return [(c, w) for c, w in d_centre(expo, v, k) if c != 0.0]


def _d2(expo, v, k, l):  # noqa: E741
    """d2/dA_k dA_l of one function, as ``[(coefficient, ang)]``."""
    return [(c1 * c2, w) for c1, u in _d(expo, v, k) for c2, w in _d(expo, u, l)]


# --- one-electron ----------------------------------------------------------------


def one_electron2(shells, charges, coords):
    """``d2S, d2T, d2V``, each ``[natm, 3, natm, 3, nao, nao]``.

    ``V = -sum_C Z_C <a| 1/|r - C| |b>``; ``d2V`` carries the operator terms,
    nucleus-nucleus and centre-nucleus."""
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    natm = len(charges)
    off = ao_offsets(shells)
    nao = off[-1]
    out = [np.zeros((natm, 3, natm, 3, nao, nao)) for _ in range(3)]
    kl9 = list(itertools.product(range(3), range(3)))

    for i, sa in enumerate(shells):
        for j in range(i, len(shells)):
            sb = shells[j]
            A, B = sa.atom, sb.atom
            ca, cb = cart(sa.l), cart(sb.l)
            blk = (natm, 3, natm, 3, len(ca), len(cb))
            gS, gT, gV = (np.zeros(blk) for _ in range(3))
            n = sa.l + sb.l + 3  # Hermite extent: t <= la + lb + 2
            for (a, ka), (b, kb) in itertools.product(
                zip(sa.exps, sa.coeffs, strict=True),
                zip(sb.exps, sb.coeffs, strict=True),
            ):
                pr = Pair(a, b, sa.centre, sb.centre, sa.l + 2, sb.l + 4)
                p = pr.p
                w = ka * kb
                s_pref = (math.pi / p) ** 1.5
                # R up to la + lb + 2, padded so a shift of two stays in range.
                R = hermite_r(n - 1, np.full(natm, p), pr.P[None, :] - coords)
                R = np.pad(R, ((0, 2), (0, 2), (0, 2), (0, 0)))
                v_fac = -charges * (2.0 * math.pi / p)
                herm = functools.cache(lambda ia, ib, pr=pr, n=n: pr.herm(ia, ib, n))

                def s1(ia, ib, pr=pr, s_pref=s_pref):
                    return s_pref * pr.ov(ia, ib)

                def t1(ia, ib, pr=pr, s_pref=s_pref, b=b):
                    out = 0.0
                    for k in range(3):
                        j_ = ib[k]
                        out += 4.0 * b * b * pr.ov(ia, shift(ib, k, 2))
                        out -= 2.0 * b * (2 * j_ + 1) * pr.ov(ia, ib)
                        if j_ >= 2:
                            out += j_ * (j_ - 1) * pr.ov(ia, shift(ib, k, -2))
                    return -0.5 * s_pref * out

                def vc(ia, ib, axes=(), herm=herm, R=R, v_fac=v_fac, n=n):
                    """Per nucleus ``[natm]``: ``V`` differentiated on each
                    nucleus along ``axes`` (the operator term)."""
                    s = [0, 0, 0]
                    for k in axes:
                        s[k] += 1
                    Rs = R[s[0] : s[0] + n, s[1] : s[1] + n, s[2] : s[2] + n]
                    sign = -1.0 if len(axes) % 2 else 1.0
                    return sign * v_fac * np.einsum("tuv,tuvc->c", herm(ia, ib), Rs)

                def v1(ia, ib, vc=vc):
                    return vc(ia, ib).sum()

                for x, ia in enumerate(ca):
                    for y, ib in enumerate(cb):
                        for g, op in ((gS, s1), (gT, t1), (gV, v1)):
                            for k, l in kl9:  # noqa: E741
                                aa = sum(c * op(u, ib) for c, u in _d2(a, ia, k, l))
                                bb = sum(c * op(ia, u) for c, u in _d2(b, ib, k, l))
                                ab = sum(
                                    c1 * c2 * op(u, v)
                                    for c1, u in _d(a, ia, k)
                                    for c2, v in _d(b, ib, l)
                                )
                                g[A, k, A, l, x, y] += w * aa
                                g[B, k, B, l, x, y] += w * bb
                                g[A, k, B, l, x, y] += w * ab
                                g[B, l, A, k, x, y] += w * ab
                        # V's operator: centre-nucleus and nucleus-nucleus.
                        for k, l in kl9:  # noqa: E741
                            aC = sum(c * vc(u, ib, (l,)) for c, u in _d(a, ia, k))
                            bC = sum(c * vc(ia, u, (l,)) for c, u in _d(b, ib, k))
                            cc = vc(ia, ib, (k, l))
                            for C in range(natm):
                                gV[A, k, C, l, x, y] += w * aC[C]
                                gV[C, l, A, k, x, y] += w * aC[C]
                                gV[B, k, C, l, x, y] += w * bC[C]
                                gV[C, l, B, k, x, y] += w * bC[C]
                                gV[C, k, C, l, x, y] += w * cc[C]

            si = slice(off[i], off[i + 1])
            sj = slice(off[j], off[j + 1])
            for M, g in zip(out, (gS, gT, gV), strict=True):
                M[..., si, sj] = g
                M[..., sj, si] = np.swapaxes(g, -1, -2)
    return tuple(out)


# --- two-electron ----------------------------------------------------------------

# A pair's Hermite stack, ``H[comp, ab, t, u, v]``:
#   0            the pair
#   1 + k        d/dA_k              4 + k        d/dB_k
#   7 + 3k + l   d2/dA_k dA_l        16 + 3k + l  d2/dA_k dB_l
#   25 + 3k + l  d2/dB_k dB_l
NCOMP = 34


def _first(pos: int, k: int) -> int:
    """The stack component of ``d/d(centre pos)_k``, ``pos`` 0 (A) or 1 (B)."""
    return 1 + 3 * pos + k


def _second(p: int, q: int, k: int, l: int) -> int:  # noqa: E741
    """The component of ``d/d(p)_k d/d(q)_l`` within one pair."""
    if p == q:
        return (7 if p == 0 else 25) + 3 * k + l
    return 16 + (3 * k + l if p == 0 else 3 * l + k)


def _pair_data2(sa, sb):
    """Per primitive pair of ``(sa, sb)``: ``p``, ``P`` and the 34-component
    Hermite stack with the contraction coefficients folded in."""
    ca, cb = cart(sa.l), cart(sb.l)
    n = sa.l + sb.l + 3
    ps, Ps, Hs = [], [], []
    for (a, ka), (b, kb) in itertools.product(
        zip(sa.exps, sa.coeffs, strict=True), zip(sb.exps, sb.coeffs, strict=True)
    ):
        pr = Pair(a, b, sa.centre, sb.centre, sa.l + 2, sb.l + 2)
        H = np.zeros((NCOMP, len(ca) * len(cb), n, n, n))
        for m, (ia, ib) in enumerate(itertools.product(ca, cb)):
            H[0, m] = pr.herm(ia, ib, n)
            for k in range(3):
                for c, u in _d(a, ia, k):
                    H[_first(0, k), m] += c * pr.herm(u, ib, n)
                for c, u in _d(b, ib, k):
                    H[_first(1, k), m] += c * pr.herm(ia, u, n)
                for l in range(3):  # noqa: E741
                    for c, u in _d2(a, ia, k, l):
                        H[_second(0, 0, k, l), m] += c * pr.herm(u, ib, n)
                    for c, u in _d2(b, ib, k, l):
                        H[_second(1, 1, k, l), m] += c * pr.herm(ia, u, n)
                    for c1, u in _d(a, ia, k):
                        for c2, v in _d(b, ib, l):
                            H[_second(0, 1, k, l), m] += c1 * c2 * pr.herm(u, v, n)
        ps.append(pr.p)
        Ps.append(pr.P)
        Hs.append(H * (ka * kb))
    return np.array(ps), np.array(Ps), np.array(Hs), n


def _quartet2(bra, ket):
    """Contracted, for every derivative split of total order <= 2:
    ``Mb[comp_bra, ab, cd]`` (ket undifferentiated), ``Mk[ab, comp_ket, cd]``
    (bra undifferentiated) and ``Mx[comp_bra, ab, comp_ket, cd]`` for one
    first derivative on each side (comps 1..6, stored 0..5)."""
    pb, Pb, Hb, nb = bra
    pk, Pk, Hk, nk = ket
    nq = len(pk)
    nab, ncd = Hb.shape[2], Hk.shape[2]
    t = np.arange(nk)
    sign = (-1.0) ** (t[:, None, None] + t[None, :, None] + t[None, None, :])
    Hk = (Hk * sign).reshape(nq, NCOMP, ncd, nk**3)
    tt = np.add.outer(np.arange(nb), np.arange(nk))
    idx = (
        tt.reshape(nb, 1, 1, nk, 1, 1),
        tt.reshape(1, nb, 1, 1, nk, 1),
        tt.reshape(1, 1, nb, 1, 1, nk),
    )
    # Total order 2 needs R to (la+lb) + (lc+ld) + 2; the stacks are padded to
    # +2 on each side, and every R entry past that meets a zero E.
    L = (nb - 1) + (nk - 1) - 2
    Mb = np.zeros((NCOMP, nab, ncd))
    Mk = np.zeros((nab, NCOMP, ncd))
    Mx = np.zeros((6, nab, 6, ncd))
    for ip in range(len(pb)):
        p = pb[ip]
        alpha = p * pk / (p + pk)
        lam = 2.0 * math.pi**2.5 / (p * pk * np.sqrt(p + pk))
        R = hermite_r(L, alpha, Pb[ip][None, :] - Pk)
        R = np.pad(R, ((0, 2), (0, 2), (0, 2), (0, 0)))[idx]
        Y = np.tensordot(
            Hb[ip].reshape(NCOMP * nab, nb**3), R.reshape(nb**3, nk**3, nq), axes=1
        )
        Y = (Y * lam).reshape(NCOMP, nab, nk**3, nq)
        Mb += np.einsum("cxmq,qym->cxy", Y, Hk[:, 0])
        Mk += np.einsum("xmq,qcym->xcy", Y[0], Hk)
        Mx += np.einsum("cxmq,qdym->cxdy", Y[1:7], Hk[:, 1:7])
    return Mb, Mk, Mx


def two_electron2(shells, natm: int):
    """``d2(ab|cd)`` as ``[natm, 3, natm, 3] + [nao]*4``. One sweep over the
    unique shell quartets; every ordered pair of the four centres formed."""
    off = ao_offsets(shells)
    nao = off[-1]
    out = np.zeros((natm, 3, natm, 3) + (nao,) * 4)
    pairs = [(i, j) for i in range(len(shells)) for j in range(i, len(shells))]
    data = {ij: _pair_data2(shells[ij[0]], shells[ij[1]]) for ij in pairs}
    kl9 = list(itertools.product(range(3), range(3)))
    for P, (i, j) in enumerate(pairs):
        for k, l in pairs[P:]:  # noqa: E741
            Mb, Mk, Mx = _quartet2(data[(i, j)], data[(k, l)])
            ijkl = (i, j, k, l)
            shape = tuple(len(cart(shells[s].l)) for s in ijkl)
            at = [shells[s].atom for s in ijkl]
            g = np.zeros((natm, 3, natm, 3) + shape)
            for p, q in itertools.product(range(4), range(4)):
                for kk, ll in kl9:
                    if p < 2 and q < 2:
                        blk = Mb[_second(p, q, kk, ll)]
                    elif p >= 2 and q >= 2:
                        blk = Mk[:, _second(p - 2, q - 2, kk, ll)]
                    elif p < 2:
                        blk = Mx[_first(p, kk) - 1, :, _first(q - 2, ll) - 1]
                    else:
                        blk = Mx[_first(q, ll) - 1, :, _first(p - 2, kk) - 1]
                    g[at[p], kk, at[q], ll] += blk.reshape(shape)
            _scatter(out, g, ijkl, off)
    return out


# --- nuclear repulsion -------------------------------------------------------------


def nuclear_repulsion2(charges, coords):
    """``d2 E_nuc / dR_{A,k} dR_{B,l}`` as ``[natm, 3, natm, 3]``."""
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    natm = len(charges)
    H = np.zeros((natm, 3, natm, 3))
    for A in range(natm):
        for B in range(A + 1, natm):
            r = coords[A] - coords[B]
            d = np.linalg.norm(r)
            blk = 3.0 * np.outer(r, r) / d**5 - np.eye(3) / d**3
            blk *= charges[A] * charges[B]
            H[A, :, A, :] += blk
            H[B, :, B, :] += blk
            H[A, :, B, :] -= blk
            H[B, :, A, :] -= blk
    return H
