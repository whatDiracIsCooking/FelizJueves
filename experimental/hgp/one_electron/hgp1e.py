"""S, T, V and the dipole by the HGP route: a VRR on the bra centre per
primitive pair, the contraction, then the HRR on contracted values.

    shells_of(mol)                     -> [Shell], libcint's normalisation, s and p
    overlap(shells)                    -> S[n_ao, n_ao]
    kinetic(shells)                    -> T[n_ao, n_ao]
    nuclear(shells, charges, coords)   -> V[n_ao, n_ao]
    dipole(shells, origin)             -> MU[3, n_ao, n_ao], <a| r - C |b>

THE ROUTE (Head-Gordon & Pople's split, on Obara & Saika's one-electron
recurrences). Per primitive pair, with `p = a + b`, `P = (aA + bB)/p`,
`K = exp(-ab/p |AB|^2)`, the VRR builds `(e|0)` for every Cartesian `e` on
`A` up to the order the target needs:

    overlap  (e+1_k|0)   = PA_k (e|0) + e_k/(2p) (e-1_k|0)
             (0|0)       = (pi/p)^{3/2} K
    nuclear  (e+1_k|0)^m = PA_k (e|0)^m - PC_k (e|0)^{m+1}
                           + e_k/(2p) [(e-1_k|0)^m - (e-1_k|0)^{m+1}]
             (0|0)^m     = (2 pi/p) K F_m(p |PC|^2)

The coefficients `c_a c_b` are applied there and the primitive pairs summed:
from here on every value is contracted. The HRR moves angular momentum from
`A` to `B`, and since `AB = A - B` does not depend on an exponent it runs
once per shell pair, on contracted values:

    (a|b+1_k) = (a+1_k|b) + AB_k (a|b)

Each operator is then a combination of contracted overlap-type integrals:

    S    (a|b)                                        VRR to la + lb
    MU_k (a|b+1_k) + (B_k - C_k) (a|b)                VRR to la + lb + 1
    T    -1/2 SUM_k [ b_k(b_k-1) (a|b-2_k)_0
                     - 2(2b_k+1) (a|b)_1 + 4 (a|b+2_k)_2 ]   VRR to la + lb + 2
    V    -SUM_C Z_C (a|b)^{(0)}_C                      VRR to la + lb, m to la + lb

The kinetic `_n` subscripts are three contracted VRR tables weighted by
`b^n` (`b` the ket exponent): differentiating `B`'s Gaussian twice brings
out powers of `b`, which is per primitive, so it is folded into the weight
before the contraction, and the three tables then share the HRR. For `V`
the nuclei are summed before the contraction ends, so the HRR runs once
per shell pair for all of them.

`FORM` holds every sign of the recurrences as a multiplier (all 1), so
`check.py` can flip each in turn and show the oracle catches it.

Plain numpy. Cartesian, libcint's order. s and p shells only (refused above;
the recurrences are general in `l`, the normalisation here is not).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np

L_MAX = 1  # p
COMMON_FAC = {0: 0.5 / math.sqrt(math.pi), 1: math.sqrt(3.0 / (4.0 * math.pi))}

# Every sign in the recurrences, as a multiplier; check.py flips each.
FORM = {
    "vrr_PA": 1.0,  # +PA_k (e|0)^m
    "vrr_PC": 1.0,  # -PC_k (e|0)^{m+1}
    "vrr_2p": 1.0,  # +e_k/(2p) (e-1_k|0)^m
    "vrr_2p_m1": 1.0,  # -e_k/(2p) (e-1_k|0)^{m+1}
    "hrr_AB": 1.0,  # +AB_k (a|b)
    "mu_BC": 1.0,  # +(B_k - C_k) (a|b)
    "t_mid": 1.0,  # -2(2b_k+1) (a|b)_1
}


@dataclass(frozen=True)
class Shell:
    """A contracted Cartesian shell; ``coeffs`` already carry normalisation."""

    am: int
    centre: tuple[float, float, float]
    exps: tuple[float, ...]
    coeffs: tuple[float, ...]


@cache
def cart(l: int) -> tuple[tuple[int, int, int], ...]:  # noqa: E741
    """Cartesian exponents of order ``l`` in libcint's order (x, y, z at p)."""
    return tuple(
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    )


def ncart(l: int) -> int:  # noqa: E741
    return (l + 1) * (l + 2) // 2


def shells_of(mol) -> list[Shell]:
    """Every shell of a PySCF ``Mole``: ``_libcint_ctr_coeff`` times libcint's
    s/p common factor (``CINTcommon_fac_sp``). s and p only."""
    out = []
    for ib in range(mol.nbas):
        am = int(mol.bas_angular(ib))
        if am > L_MAX:
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


def boys(m_max: int, T: float) -> np.ndarray:
    """``F_0..F_{m_max}(T)``: the series at ``m_max`` and the downward
    recurrence below ``T = 30``, ``erf`` and the upward recurrence above."""
    F = np.empty(m_max + 1)
    e = math.exp(-T)
    if T < 30.0:
        term = 1.0 / (2 * m_max + 1)
        s = term
        k = 0
        while term > 1e-17 * s:
            k += 1
            term *= 2.0 * T / (2 * m_max + 2 * k + 1)
            s += term
        F[m_max] = e * s
        for m in range(m_max - 1, -1, -1):
            F[m] = (2.0 * T * F[m + 1] + e) / (2 * m + 1)
    else:
        F[0] = 0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T))
        for m in range(m_max):
            F[m + 1] = ((2 * m + 1) * F[m] - e) / (2.0 * T)
    return F


def _lower(e) -> int:
    """The axis a recurrence steps down: the first nonzero one."""
    for k in range(3):
        if e[k]:
            return k
    raise ValueError("e = 0 has no axis to step down")


def _minus(e, k, n=1):
    return tuple(v - n if i == k else v for i, v in enumerate(e))


def _plus(e, k, n=1):
    return tuple(v + n if i == k else v for i, v in enumerate(e))


def vrr_overlap(order: int, PA, p: float, seed: float) -> dict:
    """``(e|0)`` for every Cartesian ``e`` with ``|e| <= order``."""
    t = {(0, 0, 0): seed}
    for n in range(1, order + 1):
        for e in cart(n):
            k = _lower(e)
            em = _minus(e, k)
            v = FORM["vrr_PA"] * PA[k] * t[em]
            if em[k]:
                v += FORM["vrr_2p"] * em[k] / (2.0 * p) * t[_minus(em, k)]
            t[e] = v
    return t


def vrr_nuclear(order: int, PA, PC, p: float, seed) -> dict:
    """``(e|0)^{(0)}`` for every ``|e| <= order`` from ``seed[m]``,
    ``m = 0..order``. Each entry is held as a vector over ``m``, one shorter
    per step up; the auxiliary index is gone at the end."""
    t = {(0, 0, 0): np.asarray(seed, dtype=float)}
    for n in range(1, order + 1):
        for e in cart(n):
            k = _lower(e)
            em = _minus(e, k)
            a = t[em]
            v = FORM["vrr_PA"] * PA[k] * a[:-1] - FORM["vrr_PC"] * PC[k] * a[1:]
            if em[k]:
                b = t[_minus(em, k)]
                v = v + em[k] / (2.0 * p) * (
                    FORM["vrr_2p"] * b[:-2] - FORM["vrr_2p_m1"] * b[1:-1]
                )
            t[e] = v
    return {e: float(v[0]) for e, v in t.items()}


def hrr(table: dict, AB, a, b) -> float:
    """``(a|b)`` from the contracted ``(e|0)`` table by the HRR."""

    @cache
    def go(a, b):
        if not any(b):
            return table[a]
        k = _lower(b)
        bm = _minus(b, k)
        return go(_plus(a, k), bm) + FORM["hrr_AB"] * AB[k] * go(a, bm)

    return go(tuple(a), tuple(b))


def _pairs(sa: Shell, sb: Shell):
    """Per primitive pair: ``(c_a c_b, b, p, P, PA, K)``."""
    A = np.array(sa.centre)
    B = np.array(sb.centre)
    AB2 = float((A - B) @ (A - B))
    for a, ca in zip(sa.exps, sa.coeffs, strict=True):
        for b, cb in zip(sb.exps, sb.coeffs, strict=True):
            p = a + b
            P = (a * A + b * B) / p
            yield ca * cb, b, p, P, P - A, math.exp(-a * b / p * AB2)


def _accumulate(acc: dict, w: float, t: dict) -> None:
    for e, v in t.items():
        acc[e] = acc.get(e, 0.0) + w * v


def _contracted(sa, sb, order, powers=(0,)):
    """The contracted overlap VRR tables, one per power ``n`` of the ket
    exponent: ``SUM c_a c_b b^n (e|0)``."""
    out = {n: {} for n in powers}
    for c, b, p, _P, PA, K in _pairs(sa, sb):
        t = vrr_overlap(order, PA, p, (math.pi / p) ** 1.5 * K)
        for n in powers:
            _accumulate(out[n], c * b**n, t)
    return out


def _AB(sa, sb):
    return np.array(sa.centre) - np.array(sb.centre)


def overlap_block(sa: Shell, sb: Shell) -> np.ndarray:
    t = _contracted(sa, sb, sa.am + sb.am)[0]
    AB = _AB(sa, sb)
    return np.array([[hrr(t, AB, a, b) for b in cart(sb.am)] for a in cart(sa.am)])


def kinetic_block(sa: Shell, sb: Shell) -> np.ndarray:
    tabs = _contracted(sa, sb, sa.am + sb.am + 2, powers=(0, 1, 2))
    AB = _AB(sa, sb)
    out = np.zeros((ncart(sa.am), ncart(sb.am)))
    for i, a in enumerate(cart(sa.am)):
        for j, b in enumerate(cart(sb.am)):
            v = 0.0
            for k in range(3):
                bk = b[k]
                if bk >= 2:
                    v += bk * (bk - 1) * hrr(tabs[0], AB, a, _minus(b, k, 2))
                v -= FORM["t_mid"] * 2.0 * (2 * bk + 1) * hrr(tabs[1], AB, a, b)
                v += 4.0 * hrr(tabs[2], AB, a, _plus(b, k, 2))
            out[i, j] = -0.5 * v
    return out


def dipole_block(sa: Shell, sb: Shell, origin) -> np.ndarray:
    t = _contracted(sa, sb, sa.am + sb.am + 1)[0]
    AB = _AB(sa, sb)
    BC = np.array(sb.centre) - np.asarray(origin, dtype=float)
    out = np.zeros((3, ncart(sa.am), ncart(sb.am)))
    for i, a in enumerate(cart(sa.am)):
        for j, b in enumerate(cart(sb.am)):
            s = hrr(t, AB, a, b)
            for k in range(3):
                out[k, i, j] = hrr(t, AB, a, _plus(b, k)) + FORM["mu_BC"] * BC[k] * s
    return out


def nuclear_block(sa: Shell, sb: Shell, charges, coords) -> np.ndarray:
    order = sa.am + sb.am
    acc: dict = {}
    for c, _b, p, P, PA, K in _pairs(sa, sb):
        for Z, Cn in zip(charges, coords, strict=True):
            PC = P - np.asarray(Cn, dtype=float)
            seed = 2.0 * math.pi / p * K * boys(order, p * float(PC @ PC))
            _accumulate(acc, -float(Z) * c, vrr_nuclear(order, PA, PC, p, seed))
    AB = _AB(sa, sb)
    return np.array([[hrr(acc, AB, a, b) for b in cart(sb.am)] for a in cart(sa.am)])


def _offsets(shells):
    off = [0]
    for s in shells:
        off.append(off[-1] + ncart(s.am))
    return off


def _matrix(shells, block, ncomp=None):
    """Every ORDERED shell pair, no symmetry used, so (sp) and (ps) are both
    evaluated and the symmetry of the result is a check, not a construction."""
    off = _offsets(shells)
    n = off[-1]
    M = np.zeros((n, n) if ncomp is None else (ncomp, n, n))
    for i, sa in enumerate(shells):
        for j, sb in enumerate(shells):
            M[..., off[i] : off[i + 1], off[j] : off[j + 1]] = block(sa, sb)
    return M


def overlap(shells) -> np.ndarray:
    return _matrix(shells, overlap_block)


def kinetic(shells) -> np.ndarray:
    return _matrix(shells, kinetic_block)


def nuclear(shells, charges, coords) -> np.ndarray:
    return _matrix(shells, lambda a, b: nuclear_block(a, b, charges, coords))


def dipole(shells, origin) -> np.ndarray:
    return _matrix(shells, lambda a, b: dipole_block(a, b, origin), ncomp=3)
