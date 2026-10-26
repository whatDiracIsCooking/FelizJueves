"""Rys quadrature ERIs: the 2D integrals per axis, one primitive quartet, and a
contracted Cartesian (ab|cd) block, generic in (la, lb, lc, ld).

The settled form (``README.md`` records it and what was checked):

    B00 = x / (2 (zeta + eta))
    B10 = (1 - x eta / (zeta + eta)) / (2 zeta)
    B01 = (1 - x zeta / (zeta + eta)) / (2 eta)
    C00 = (P - A) + x (W - P)
    D00 = (Q - C) + x (W - Q)

    I(0,0,0,0)   = 1
    I(i+1,0,k,0) = C00 I(i,0,k,0) + i B10 I(i-1,0,k,0) + k B00 I(i,0,k-1,0)
    I(i,0,k+1,0) = D00 I(i,0,k,0) + k B01 I(i,0,k-1,0) + i B00 I(i-1,0,k,0)
    I(i,j+1,k,l) = I(i+1,j,k,l) + (A - B) I(i,j,k,l)
    I(i,j,k,l+1) = I(i,j,k+1,l) + (C - D) I(i,j,k,l)

with the per-quartet scale on the weight,

    W_n * (2/sqrt(pi)) sqrt(rho) (pi/zeta)^(3/2) K_AB (pi/eta)^(3/2) K_CD.
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from roots import _PI, roots_weights, roots_weights_exact_decimal

# The form's signs and factors, as multipliers. Every entry is 1 in the settled
# form; ``check.py`` flips one at a time to show which classes would catch it.
FORM = {
    "B00": 1.0,  # B00 = FORM * x / (2 (zeta + eta))
    "B10.x": 1.0,  # B10 = (1 - FORM * x eta / (zeta + eta)) / (2 zeta)
    "B01.x": 1.0,  # B01 = (1 - FORM * x zeta / (zeta + eta)) / (2 eta)
    "C00.PA": 1.0,  # C00 = FORM * (P - A) + x (W - P)
    "C00.x": 1.0,  # C00 = (P - A) + FORM * x (W - P)
    "D00.QC": 1.0,  # D00 = FORM * (Q - C) + x (W - Q)
    "D00.x": 1.0,  # D00 = (Q - C) + FORM * x (W - Q)
    "AB": 1.0,  # I(i,j+1) = I(i+1,j) + FORM * (A - B) I(i,j)
    "CD": 1.0,  # I(k,l+1) = I(k+1,l) + FORM * (C - D) I(k,l)
}


def cart_list(am: int) -> list[tuple[int, int, int]]:
    """libcint's Cartesian order: lx descending, then ly descending."""
    return [
        (lx, ly, am - lx - ly)
        for lx in range(am, -1, -1)
        for ly in range(am - lx, -1, -1)
    ]


def ncart(am: int) -> int:
    return (am + 1) * (am + 2) // 2


def _two_d(x, la, lb, lc, ld, AB, CD, PA, QC, WP, WQ, zeta, eta):
    """I[n, i, j, k, l] for one axis, nodes x[n]: VRR to (i + j, 0 | k + l, 0),
    then the two transfer (horizontal) steps."""
    nn = len(x)
    lab, lcd = la + lb, lc + ld
    s = zeta + eta
    f = FORM
    b00 = f["B00"] * x / (2.0 * s)
    b10 = (1.0 - f["B10.x"] * x * eta / s) / (2.0 * zeta)
    b01 = (1.0 - f["B01.x"] * x * zeta / s) / (2.0 * eta)
    c00 = f["C00.PA"] * PA + f["C00.x"] * x * WP
    d00 = f["D00.QC"] * QC + f["D00.x"] * x * WQ
    AB, CD = f["AB"] * AB, f["CD"] * CD

    g = np.zeros((nn, lab + 1, lcd + 1))
    g[:, 0, 0] = 1.0
    for i in range(lab):
        g[:, i + 1, 0] = c00 * g[:, i, 0] + (i * b10 * g[:, i - 1, 0] if i else 0.0)
    for k in range(lcd):
        for i in range(lab + 1):
            v = d00 * g[:, i, k]
            if k:
                v = v + k * b01 * g[:, i, k - 1]
            if i:
                v = v + i * b00 * g[:, i - 1, k]
            g[:, i, k + 1] = v

    # Bra transfer: h[n, i, j, k] for i + j <= lab.
    h = np.zeros((nn, lab + 1, lb + 1, lcd + 1))
    h[:, :, 0, :] = g
    for j in range(lb):
        for i in range(lab - j):
            h[:, i, j + 1, :] = h[:, i + 1, j, :] + AB * h[:, i, j, :]
    h = h[:, : la + 1]
    # Ket transfer.
    out = np.zeros((nn, la + 1, lb + 1, lcd + 1, ld + 1))
    out[..., 0] = h
    for l_ in range(ld):
        for k in range(lcd - l_):
            out[..., k, l_ + 1] = out[..., k + 1, l_] + CD * out[..., k, l_]
    return out[:, :, :, : lc + 1, :]


def prim_quartet(ls, centres, exps, t_log: list | None = None) -> np.ndarray:
    """[ab|cd] over Cartesian components for one primitive quartet, with no
    contraction coefficient or normalisation: shape (nc_a, nc_b, nc_c, nc_d)."""
    la, lb, lc, ld = ls
    A, B, C, D = (np.asarray(c, dtype=float) for c in centres)
    a, b, c, d = exps
    zeta, eta = a + b, c + d
    rho = zeta * eta / (zeta + eta)
    P = (a * A + b * B) / zeta
    Q = (c * C + d * D) / eta
    W = (zeta * P + eta * Q) / (zeta + eta)
    k_ab = np.exp(-a * b / zeta * np.dot(A - B, A - B))
    k_cd = np.exp(-c * d / eta * np.dot(C - D, C - D))
    T = rho * np.dot(P - Q, P - Q)
    if t_log is not None:
        t_log.append(T)
    n = (la + lb + lc + ld) // 2 + 1
    x, w = roots_weights(n, T)
    w = (
        w
        * (2.0 / np.sqrt(np.pi))
        * np.sqrt(rho)
        * (np.pi / zeta) ** 1.5
        * k_ab
        * (np.pi / eta) ** 1.5
        * k_cd
    )
    axes = [
        _two_d(x, la, lb, lc, ld, A[q] - B[q], C[q] - D[q], P[q] - A[q],
               Q[q] - C[q], W[q] - P[q], W[q] - Q[q], zeta, eta)
        for q in range(3)
    ]
    ca, cb, cc, cd = (cart_list(v) for v in ls)
    out = np.empty((len(ca), len(cb), len(cc), len(cd)))
    for ia, ea in enumerate(ca):
        for ib, eb in enumerate(cb):
            for ic, ec in enumerate(cc):
                for id_, ed in enumerate(cd):
                    prod = w.copy()
                    for q in range(3):
                        prod = prod * axes[q][:, ea[q], eb[q], ec[q], ed[q]]
                    out[ia, ib, ic, id_] = prod.sum()
    return out


@dataclass(frozen=True)
class Shell:
    """A contracted Cartesian shell; ``coeffs`` already carry normalisation."""

    am: int
    centre: tuple[float, float, float]
    exps: tuple[float, ...]
    coeffs: tuple[float, ...]


# libcint's s/p common factor (CINTcommon_fac_sp), which int2e applies.
COMMON_FAC = {0: 0.5 / np.sqrt(np.pi), 1: np.sqrt(3.0 / (4.0 * np.pi))}


def shells_of(mol) -> list[Shell]:
    """Every shell of a PySCF ``Mole`` with libcint's normalisation, read from
    the Mole: ``_libcint_ctr_coeff`` (the radial normalisation times the
    contraction) and ``COMMON_FAC``. s and p only: d and above have no common
    factor here, and the roots stop at N = 3, i.e. (pp|pp)."""
    out = []
    for ib in range(mol.nbas):
        am = int(mol.bas_angular(ib))
        if am not in COMMON_FAC:
            raise ValueError(f"shell {ib} has l = {am}; this spike stops at p")
        coeff = mol._libcint_ctr_coeff(ib)
        assert coeff.shape[1] == 1, "generally contracted shell"
        out.append(
            Shell(
                am,
                tuple(float(v) for v in mol.bas_coord(ib)),
                tuple(float(v) for v in mol.bas_exp(ib)),
                tuple(float(v) * COMMON_FAC[am] for v in coeff[:, 0]),
            )
        )
    return out


def contracted_block(sa: Shell, sb: Shell, sc: Shell, sd: Shell, t_log=None):
    """(ab|cd) for four contracted shells: the coefficient-weighted sum of
    ``prim_quartet`` over every primitive quartet."""
    ls = (sa.am, sb.am, sc.am, sd.am)
    centres = (sa.centre, sb.centre, sc.centre, sd.centre)
    out = np.zeros(tuple(ncart(v) for v in ls))
    for a, ka in zip(sa.exps, sa.coeffs, strict=True):
        for b, kb in zip(sb.exps, sb.coeffs, strict=True):
            for c, kc in zip(sc.exps, sc.coeffs, strict=True):
                for d, kd in zip(sd.exps, sd.coeffs, strict=True):
                    out += ka * kb * kc * kd * prim_quartet(
                        ls, centres, (a, b, c, d), t_log
                    )
    return out


# --- the same quadrature at 50 digits, to referee a disagreement ---------------


def prim_quartet_exact(ls, centres, exps, digits: int = 50) -> np.ndarray:
    """``prim_quartet`` in ``decimal`` on the exact float inputs, with the
    exact-route roots: a ground truth for rounding, not for the form (it
    uses the same recurrence). Returned rounded to float64."""
    with decimal.localcontext() as ctx:
        ctx.prec = digits + 10
        la, lb, lc, ld = ls
        A, B, C, D = ([Decimal(float(v)) for v in c] for c in centres)
        a, b, c, d = (Decimal(float(v)) for v in exps)
        zeta, eta = a + b, c + d
        s = zeta + eta
        rho = zeta * eta / s
        P = [(a * A[q] + b * B[q]) / zeta for q in range(3)]
        Q = [(c * C[q] + d * D[q]) / eta for q in range(3)]
        W = [(zeta * P[q] + eta * Q[q]) / s for q in range(3)]
        ab2 = sum((A[q] - B[q]) ** 2 for q in range(3))
        cd2 = sum((C[q] - D[q]) ** 2 for q in range(3))
        T = rho * sum((P[q] - Q[q]) ** 2 for q in range(3))
        n = (la + lb + lc + ld) // 2 + 1
        xs, ws = roots_weights_exact_decimal(n, T, digits)
        pi = _PI
        k_abcd = (-a * b / zeta * ab2 - c * d / eta * cd2).exp()
        scale = (
            2 / pi.sqrt() * rho.sqrt() * (pi / zeta) ** Decimal(1.5)
            * (pi / eta) ** Decimal(1.5) * k_abcd
        )
        lab, lcd = la + lb, lc + ld

        def axis(x, q):
            b00 = x / (2 * s)
            b10 = (1 - x * eta / s) / (2 * zeta)
            b01 = (1 - x * zeta / s) / (2 * eta)
            c00 = (P[q] - A[q]) + x * (W[q] - P[q])
            d00 = (Q[q] - C[q]) + x * (W[q] - Q[q])
            g = {(0, 0): Decimal(1)}
            for i in range(lab):
                g[i + 1, 0] = c00 * g[i, 0] + (i * b10 * g[i - 1, 0] if i else 0)
            for k in range(lcd):
                for i in range(lab + 1):
                    v = d00 * g[i, k]
                    if k:
                        v += k * b01 * g[i, k - 1]
                    if i:
                        v += i * b00 * g[i - 1, k]
                    g[i, k + 1] = v
            h = {(i, 0, k): g[i, k] for i in range(lab + 1) for k in range(lcd + 1)}
            for j in range(lb):
                for i in range(lab - j):
                    for k in range(lcd + 1):
                        h[i, j + 1, k] = h[i + 1, j, k] + (A[q] - B[q]) * h[i, j, k]
            o = {(i, j, k, 0): h[i, j, k] for (i, j, k) in h}
            for l_ in range(ld):
                for k in range(lcd - l_):
                    for i in range(la + 1):
                        for j in range(lb + 1):
                            o[i, j, k, l_ + 1] = (
                                o[i, j, k + 1, l_] + (C[q] - D[q]) * o[i, j, k, l_]
                            )
            return o

        axes = [[axis(x, q) for q in range(3)] for x in xs]
        ca, cb, cc, cd = (cart_list(v) for v in ls)
        out = np.empty((len(ca), len(cb), len(cc), len(cd)))
        for ia, ea in enumerate(ca):
            for ib, eb in enumerate(cb):
                for ic, ec in enumerate(cc):
                    for id_, ed in enumerate(cd):
                        tot = Decimal(0)
                        for nn, w in enumerate(ws):
                            prod = w
                            for q in range(3):
                                prod *= axes[nn][q][ea[q], eb[q], ec[q], ed[q]]
                            tot += prod
                        out[ia, ib, ic, id_] = float(tot * scale)
        return out
