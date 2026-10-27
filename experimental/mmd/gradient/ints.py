"""S, T, V and (ab|cd) and their nuclear derivatives by McMurchie-Davidson, and
the J, K and RHF gradient built on them.

Every derivative is analytic and formed the same way: d/dA_k of a Cartesian
Gaussian is ``2a g_{i+1} - i g_{i-1}`` (``mmd.d_centre``), so a derivative
integral is two ordinary integrals with one function's ``l`` moved by one, and
both are read off the same ``E`` and ``R`` tables as the integral itself. The
one exception is ``V``'s operator: ``1/|r - C|`` moves with nucleus ``C``, and
``d/dC_k R_tuv(p, P - C) = -R_{tuv + e_k}`` (Hellmann-Feynman), one more
Hermite index on the same ``R``. No translational invariance is used: the
derivative on every centre is computed, so ``sum_atoms d/dR = 0`` is a check
(``check.py``), not an assumption.

Derivative arrays are ``dX[atom, k, ...]``: the derivative of the whole
integral ``X`` with respect to nucleus ``atom``'s coordinate ``k``, every
centre on that atom summed.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from mmd import Pair, ao_offsets, cart, d_centre, hermite_r, shift

# --- one-electron ----------------------------------------------------------------


def _d(expo, v, k):
    return [(c, w) for c, w in d_centre(expo, v, k) if c != 0.0]


def one_electron(shells, charges, coords):
    """``S, T, V`` and ``dS, dT, dV`` (each ``[natm, 3, nao, nao]``).

    ``V = -sum_C Z_C <a| 1/|r - C| |b>``; ``dV`` includes the operator term."""
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    natm = len(charges)
    off = ao_offsets(shells)
    nao = off[-1]
    S, T, V = (np.zeros((nao, nao)) for _ in range(3))
    dS, dT, dV = (np.zeros((natm, 3, nao, nao)) for _ in range(3))

    for i, sa in enumerate(shells):
        for j in range(i, len(shells)):
            sb = shells[j]
            ca, cb = cart(sa.l), cart(sb.l)
            shape = (len(ca), len(cb))
            bS, bT, bV = (np.zeros(shape) for _ in range(3))
            gS, gT, gV = (np.zeros((natm, 3) + shape) for _ in range(3))
            n = sa.l + sb.l + 2  # Hermite extent: t <= la + lb + 1
            for (a, ka), (b, kb) in itertools.product(
                zip(sa.exps, sa.coeffs, strict=True),
                zip(sb.exps, sb.coeffs, strict=True),
            ):
                pr = Pair(a, b, sa.centre, sb.centre, sa.l + 1, sb.l + 3)
                p = pr.p
                w = ka * kb
                s_pref = (math.pi / p) ** 1.5
                # R up to la + lb + 2: a bra/ket derivative or the nucleus one.
                R = hermite_r(n, np.full(natm, p), pr.P[None, :] - coords)
                v_fac = -charges * (2.0 * math.pi / p)

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

                def v1(ia, ib, pr=pr, R=R, v_fac=v_fac, n=n):
                    H = pr.herm(ia, ib, n)
                    return v_fac @ np.einsum("tuv,tuvc->c", H, R[:n, :n, :n])

                def v_nuc(ia, ib, pr=pr, R=R, v_fac=v_fac, n=n):
                    """d/dC_k for every nucleus: ``[natm, 3]``."""
                    H = pr.herm(ia, ib, n)
                    Rs = (
                        R[1 : n + 1, :n, :n],
                        R[:n, 1 : n + 1, :n],
                        R[:n, :n, 1 : n + 1],
                    )
                    return -np.stack(
                        [v_fac * np.einsum("tuv,tuvc->c", H, r) for r in Rs], axis=1
                    )

                for x, ia in enumerate(ca):
                    for y, ib in enumerate(cb):
                        for blk, g, op in ((bS, gS, s1), (bT, gT, t1), (bV, gV, v1)):
                            blk[x, y] += w * op(ia, ib)
                            for k in range(3):
                                dA = sum(c * op(v, ib) for c, v in _d(a, ia, k))
                                dB = sum(c * op(ia, v) for c, v in _d(b, ib, k))
                                g[sa.atom, k, x, y] += w * dA
                                g[sb.atom, k, x, y] += w * dB
                        gV[:, :, x, y] += w * v_nuc(ia, ib)

            si = slice(off[i], off[i + 1])
            sj = slice(off[j], off[j + 1])
            for M, blk in ((S, bS), (T, bT), (V, bV)):
                M[si, sj] = blk
                M[sj, si] = blk.T
            for M, g in ((dS, gS), (dT, gT), (dV, gV)):
                M[:, :, si, sj] = g
                M[:, :, sj, si] = g.transpose(0, 1, 3, 2)
    return S, T, V, dS, dT, dV


# --- two-electron ----------------------------------------------------------------

_PERMS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def _pair_data(sa, sb, deriv: bool):
    """Per primitive pair of ``(sa, sb)``: ``p``, ``P`` and the Hermite stack
    ``H[comp, ab, t, u, v]`` with the contraction coefficients folded in;
    ``comp`` 0 is the function pair, 1..3 its d/dA_k, 4..6 its d/dB_k."""
    ca, cb = cart(sa.l), cart(sb.l)
    n = sa.l + sb.l + (2 if deriv else 1)
    ncomp = 7 if deriv else 1
    ps, Ps, Hs = [], [], []
    for (a, ka), (b, kb) in itertools.product(
        zip(sa.exps, sa.coeffs, strict=True), zip(sb.exps, sb.coeffs, strict=True)
    ):
        pr = Pair(a, b, sa.centre, sb.centre, sa.l + 1, sb.l + 1)
        H = np.zeros((ncomp, len(ca) * len(cb), n, n, n))
        for m, (ia, ib) in enumerate(itertools.product(ca, cb)):
            H[0, m] = pr.herm(ia, ib, n)
            if deriv:
                for k in range(3):
                    for c, v in _d(a, ia, k):
                        H[1 + k, m] += c * pr.herm(v, ib, n)
                    for c, v in _d(b, ib, k):
                        H[4 + k, m] += c * pr.herm(ia, v, n)
        ps.append(pr.p)
        Ps.append(pr.P)
        Hs.append(H * (ka * kb))
    return np.array(ps), np.array(Ps), np.array(Hs), n


def _quartet(bra, ket, deriv: bool):
    """``M1[comp_bra, ab, cd]`` (the ket undifferentiated) and, with ``deriv``,
    ``M2[ab, comp_ket, cd]`` (the bra undifferentiated), contracted."""
    pb, Pb, Hb, nb = bra
    pk, Pk, Hk, nk = ket
    nq = len(pk)
    ncomp, nab = Hb.shape[1], Hb.shape[2]
    ncd = Hk.shape[2]
    t = np.arange(nk)
    sign = (-1.0) ** (t[:, None, None] + t[None, :, None] + t[None, None, :])
    Hk = (Hk * sign).reshape(nq, ncomp, ncd, nk**3)
    tt = np.add.outer(np.arange(nb), np.arange(nk))
    idx = (
        tt.reshape(nb, 1, 1, nk, 1, 1),
        tt.reshape(1, nb, 1, 1, nk, 1),
        tt.reshape(1, 1, nb, 1, 1, nk),
    )
    L = (nb - 1) + (nk - 1)
    M1 = np.zeros((ncomp * nab, ncd))
    M2 = np.zeros((nab, ncomp, ncd)) if deriv else None
    for ip in range(len(pb)):
        p = pb[ip]
        alpha = p * pk / (p + pk)
        lam = 2.0 * math.pi**2.5 / (p * pk * np.sqrt(p + pk))
        R6 = hermite_r(L, alpha, Pb[ip][None, :] - Pk)[idx]
        Y = np.tensordot(
            Hb[ip].reshape(ncomp * nab, nb**3), R6.reshape(nb**3, nk**3, nq), axes=1
        )
        Y *= lam
        M1 += np.einsum("kmq,qlm->kl", Y, Hk[:, 0])
        if deriv:
            M2 += np.einsum("kmq,qclm->kcl", Y[:nab], Hk)
    return M1.reshape(ncomp, nab, ncd), M2


def _scatter(out, X, shells_ijkl, off):
    """Write block ``X[..., a, b, c, d]`` of shells ``(i, j, k, l)`` into all
    eight places the permutational symmetry of ``(ab|cd)`` puts it."""
    lead = X.ndim - 4
    for perm in _PERMS:
        sl = tuple(slice(off[shells_ijkl[m]], off[shells_ijkl[m] + 1]) for m in perm)
        axes = tuple(range(lead)) + tuple(lead + m for m in perm)
        out[(Ellipsis,) + sl] = X.transpose(axes)


def two_electron(shells, natm: int, deriv: bool = True):
    """``(ab|cd)`` as ``[nao]*4`` and, with ``deriv``, ``d(ab|cd)/dR`` as
    ``[natm, 3] + [nao]*4``. One sweep over the unique shell quartets."""
    off = ao_offsets(shells)
    nao = off[-1]
    eri = np.zeros((nao,) * 4)
    deri = np.zeros((natm, 3) + (nao,) * 4) if deriv else None
    pairs = [(i, j) for i in range(len(shells)) for j in range(i, len(shells))]
    data = {ij: _pair_data(shells[ij[0]], shells[ij[1]], deriv) for ij in pairs}
    for P, (i, j) in enumerate(pairs):
        for k, l in pairs[P:]:  # noqa: E741
            M1, M2 = _quartet(data[(i, j)], data[(k, l)], deriv)
            shape = tuple(len(cart(shells[s].l)) for s in (i, j, k, l))
            ijkl = (i, j, k, l)
            _scatter(eri, M1[0].reshape(shape), ijkl, off)
            if not deriv:
                continue
            g = np.zeros((natm, 3) + shape)
            for kk in range(3):
                g[shells[i].atom, kk] += M1[1 + kk].reshape(shape)
                g[shells[j].atom, kk] += M1[4 + kk].reshape(shape)
                g[shells[k].atom, kk] += M2[:, 1 + kk].reshape(shape)
                g[shells[l].atom, kk] += M2[:, 4 + kk].reshape(shape)
            _scatter(deri, g, ijkl, off)
    return eri, deri


# --- J, K and the RHF gradient ---------------------------------------------------


def coulomb(eri, D):
    """``J_mn = sum_ls (mn|ls) D_ls``."""
    return np.einsum("mnls,ls->mn", eri, D)


def exchange(eri, D):
    """``K_mn = sum_ls (ml|sn) D_ls`` (PySCF's ``get_jk`` convention)."""
    return np.einsum("mlsn,ls->mn", eri, D)


def coulomb_deriv(deri, D):
    """``dJ[A, k, m, n] = sum_ls d(mn|ls)/dR_Ak D_ls``: ``D`` held fixed."""
    return np.einsum("akmnls,ls->akmn", deri, D)


def exchange_deriv(deri, D):
    """``dK[A, k, m, n] = sum_ls d(ml|sn)/dR_Ak D_ls``: ``D`` held fixed."""
    return np.einsum("akmlsn,ls->akmn", deri, D)


def nuclear_repulsion(charges, coords):
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    e, g = 0.0, np.zeros_like(coords)
    for A in range(len(charges)):
        for B in range(len(charges)):
            if A == B:
                continue
            r = coords[A] - coords[B]
            d = np.linalg.norm(r)
            if A < B:
                e += charges[A] * charges[B] / d
            g[A] -= charges[A] * charges[B] * r / d**3
    return e, g


def rhf_energy(S, T, V, eri, D, charges, coords):
    """``E = Tr D h + 1/2 Tr D J - 1/4 Tr D K + E_nuc``, ``D`` the total density."""
    h = T + V
    J, K = coulomb(eri, D), exchange(eri, D)
    return (
        np.sum(D * h)
        + 0.5 * np.sum(D * J)
        - 0.25 * np.sum(D * K)
        + nuclear_repulsion(charges, coords)[0]
    )


def rhf_gradient(shells, charges, coords, D, W):
    """The closed-shell RHF gradient ``[natm, 3]`` and its parts.

    ``D`` is the total (spin-summed) density and ``W`` the energy-weighted one,
    ``2 sum_occ e_i C_i C_i^T``; both are taken as given (converged), so the
    orbital response is the ``-Tr W dS`` term alone."""
    natm = len(charges)
    _, _, _, dS, dT, dV = one_electron(shells, charges, coords)
    _, deri = two_electron(shells, natm)
    parts = {
        "nuc": nuclear_repulsion(charges, coords)[1],
        "T": np.einsum("akmn,mn->ak", dT, D),
        "V": np.einsum("akmn,mn->ak", dV, D),
        "J": 0.5 * np.einsum("akmn,mn->ak", coulomb_deriv(deri, D), D),
        "K": -0.25 * np.einsum("akmn,mn->ak", exchange_deriv(deri, D), D),
        "S": -np.einsum("akmn,mn->ak", dS, W),
    }
    return sum(parts.values()), parts
