"""McMurchie-Davidson primitives for the gradient spike: shells, the Hermite
expansion coefficients ``E``, the Boys function and the Hermite Coulomb
integrals ``R``.

Everything here is generic in angular momentum; only ``shells_of`` stops at p,
because the libcint common factor below is written for s and p only. A
gradient raises one function's ``l`` by one (and the kinetic operator by two
more on the ket), so the integrals themselves run to f internally.

Per axis, for exponents ``a`` on ``A`` and ``b`` on ``B``, ``p = a + b``:

    E^{00}_0    = exp(-ab/p X_AB^2)
    E^{i+1,j}_t = 1/(2p) E^{ij}_{t-1} + X_PA E^{ij}_t + (t+1) E^{ij}_{t+1}
    E^{i,j+1}_t = 1/(2p) E^{ij}_{t-1} + X_PB E^{ij}_t + (t+1) E^{ij}_{t+1}

and, with ``R^n_{000}(p, PC) = (-2p)^n F_n(p |PC|^2)``,

    R^n_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + X_PC R^{n+1}_{t,u,v}

(the same for ``u`` with ``Y_PC`` and ``v`` with ``Z_PC``); ``R_tuv = R^0_tuv``
is ``d^t/dP_x^t d^u/dP_y^u d^v/dP_z^v`` of ``F_0(p |P - C|^2)``.

numpy only; PySCF is needed by ``shells_of`` (it reads a ``Mole``) and by
``check.py``, nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# libcint's s/p common factor (CINTcommon_fac_sp): with it and
# ``Mole._libcint_ctr_coeff`` every Cartesian s and p function is normalised.
COMMON_FAC = {0: 0.5 / math.sqrt(math.pi), 1: math.sqrt(3.0 / (4.0 * math.pi))}
LMAX = 1

Ang = tuple[int, int, int]


@dataclass(frozen=True)
class Shell:
    """A contracted Cartesian shell; ``coeffs`` carry the normalisation."""

    l: int  # noqa: E741
    centre: tuple[float, float, float]
    exps: tuple[float, ...]
    coeffs: tuple[float, ...]
    atom: int


def cart(l: int) -> list[Ang]:  # noqa: E741
    """Cartesian components in libcint's order (xx, xy, xz, yy, ...)."""
    return [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]


def ncart(l: int) -> int:  # noqa: E741
    return (l + 1) * (l + 2) // 2


def ao_offsets(shells: list[Shell]) -> list[int]:
    out, n = [], 0
    for s in shells:
        out.append(n)
        n += ncart(s.l)
    out.append(n)
    return out


def shells_of(mol) -> list[Shell]:
    """Every shell of a PySCF ``Mole`` (``cart=True``) with libcint's
    normalisation. s and p only."""
    out = []
    for ib in range(mol.nbas):
        am = int(mol.bas_angular(ib))
        if am > LMAX:
            raise ValueError(f"shell {ib} has l = {am}; this spike stops at p")
        coeff = mol._libcint_ctr_coeff(ib)
        if coeff.shape[1] != 1:
            raise ValueError(f"shell {ib} is generally contracted")
        out.append(
            Shell(
                am,
                tuple(float(v) for v in mol.bas_coord(ib)),
                tuple(float(v) for v in mol.bas_exp(ib)),
                tuple(float(v) * COMMON_FAC[am] for v in coeff[:, 0]),
                int(mol.bas_atom(ib)),
            )
        )
    return out


def displaced(shells: list[Shell], atom: int, axis: int, h: float) -> list[Shell]:
    """``shells`` with every shell on ``atom`` moved by ``h`` along ``axis``."""
    out = []
    for s in shells:
        if s.atom == atom:
            c = list(s.centre)
            c[axis] += h
            s = replace(s, centre=tuple(c))
        out.append(s)
    return out


# --- E -------------------------------------------------------------------------


def e_coeffs(imax: int, jmax: int, a: float, b: float, xa: float, xb: float):
    """``E[i, j, t]`` for one axis, ``i <= imax``, ``j <= jmax``."""
    p = a + b
    xab = xa - xb
    xpa = -b / p * xab
    xpb = a / p * xab
    h = 0.5 / p
    n = imax + jmax + 1
    E = np.zeros((imax + 1, jmax + 1, n + 1))  # one spare t, always zero
    E[0, 0, 0] = math.exp(-a * b / p * xab * xab)
    for i in range(imax + 1):
        for j in range(jmax + 1):
            if i == 0 and j == 0:
                continue
            src, x = (E[i - 1, j], xpa) if i > 0 else (E[i, j - 1], xpb)
            for t in range(i + j + 1):
                v = x * src[t] + (t + 1) * src[t + 1]
                if t > 0:
                    v += h * src[t - 1]
                E[i, j, t] = v
    return E[:, :, :n]


class Pair:
    """One primitive pair: ``E`` on all three axes up to ``(imax, jmax)``."""

    def __init__(self, a, b, A, B, imax, jmax):
        self.a, self.b = a, b
        self.p = a + b
        self.P = (a * np.asarray(A) + b * np.asarray(B)) / self.p
        self.E = [e_coeffs(imax, jmax, a, b, A[k], B[k]) for k in range(3)]

    def ov(self, ia: Ang, ib: Ang) -> float:
        """The overlap ``<ia|ib>`` without ``(pi/p)^{3/2}``; 0 for a negative
        exponent (the ``- i g_{i-1}`` term of a derivative at ``i = 0``)."""
        if min(ia) < 0 or min(ib) < 0:
            return 0.0
        Ex, Ey, Ez = self.E
        return Ex[ia[0], ib[0], 0] * Ey[ia[1], ib[1], 0] * Ez[ia[2], ib[2], 0]

    def herm(self, ia: Ang, ib: Ang, n: int) -> np.ndarray:
        """The 3D Hermite expansion ``E_tuv`` of ``|ia><ib|`` padded to (n,n,n)."""
        out = np.zeros((n, n, n))
        if min(ia) < 0 or min(ib) < 0:
            return out
        Ex, Ey, Ez = self.E
        ex = Ex[ia[0], ib[0], : ia[0] + ib[0] + 1]
        ey = Ey[ia[1], ib[1], : ia[1] + ib[1] + 1]
        ez = Ez[ia[2], ib[2], : ia[2] + ib[2] + 1]
        out[: len(ex), : len(ey), : len(ez)] = np.einsum("i,j,k->ijk", ex, ey, ez)
        return out


def shift(v: Ang, k: int, d: int) -> Ang:
    w = list(v)
    w[k] += d
    return tuple(w)


def d_centre(expo: float, v: Ang, k: int) -> list[tuple[float, Ang]]:
    """d/dA_k of ``x_A^i exp(-a x_A^2)``: ``2a g_{i+1} - i g_{i-1}``."""
    return [(2.0 * expo, shift(v, k, +1)), (-float(v[k]), shift(v, k, -1))]


# --- Boys and R ----------------------------------------------------------------

_SERIES_T = 30.0
_SERIES_TERMS = 200
_erf = np.vectorize(math.erf, otypes=[float])


def boys(nmax: int, T) -> np.ndarray:
    """``F_n(T)`` for ``n = 0..nmax``, shape ``(nmax+1,) + T.shape``.

    Below ``T = 30``: the series at ``nmax``, then downward recurrence. Above:
    ``F_0`` from ``erf`` and upward recurrence, stable there."""
    T = np.asarray(T, dtype=float)
    out = np.empty((nmax + 1,) + T.shape)
    small = T < _SERIES_T
    if small.any():
        t = T[small]
        e = np.exp(-t)
        term = np.full_like(t, 1.0 / (2 * nmax + 1))
        total = term.copy()
        for k in range(1, _SERIES_TERMS):
            term = term * (2.0 * t) / (2 * nmax + 2 * k + 1)
            total += term
        f = total * e
        out[nmax][small] = f
        for n in range(nmax - 1, -1, -1):
            f = (2.0 * t * f + e) / (2 * n + 1)
            out[n][small] = f
    big = ~small
    if big.any():
        t = T[big]
        e = np.exp(-t)
        f = 0.5 * np.sqrt(np.pi / t) * _erf(np.sqrt(t))
        out[0][big] = f
        for n in range(nmax):
            f = ((2 * n + 1) * f - e) / (2.0 * t)
            out[n + 1][big] = f
    return out


def hermite_r(L: int, p, pc) -> np.ndarray:
    """``R_tuv(p, PC)`` for ``t + u + v <= L``, shape ``(L+1, L+1, L+1, N)``,
    vectorised over ``N`` (``p`` shape ``(N,)``, ``pc`` shape ``(N, 3)``)."""
    p = np.asarray(p, dtype=float)
    pc = np.asarray(pc, dtype=float)
    X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
    F = boys(L, p * np.einsum("ij,ij->i", pc, pc))
    prev = None
    for n in range(L, -1, -1):
        cur = np.zeros((L + 1, L + 1, L + 1, len(p)))
        cur[0, 0, 0] = (-2.0 * p) ** n * F[n]
        for tot in range(1, L - n + 1):
            for t in range(tot + 1):
                for u in range(tot - t + 1):
                    v = tot - t - u
                    if t > 0:
                        val = X * prev[t - 1, u, v]
                        if t > 1:
                            val = val + (t - 1) * prev[t - 2, u, v]
                    elif u > 0:
                        val = Y * prev[t, u - 1, v]
                        if u > 1:
                            val = val + (u - 1) * prev[t, u - 2, v]
                    else:
                        val = Z * prev[t, u, v - 1]
                        if v > 1:
                            val = val + (v - 1) * prev[t, u, v - 2]
                    cur[t, u, v] = val
        prev = cur
    return prev
