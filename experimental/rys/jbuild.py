"""The Coulomb matrix on the Rys ERIs: J_ab = SUM_cd (ab|cd) D_cd.

One sweep over the unique shell quartets: bra pair p = (i, j) with i <= j,
ket pair q = (k, l) with k <= l, and p <= q in pair order. Each block
``(ij|kl)`` is evaluated once and digested both ways,

    J_ij += SUM_cd (ij|cd) w_kl D_cd        J_kl += SUM_ab (ab|kl) w_ij D_ab  (p != q)

with ``w = 1`` on a diagonal pair (k = l) and 2 off it, because the sweep
visits (k, l) and never (l, k). D enters only through its symmetric part,
which loses nothing: (ab|cd) = (ab|dc), so J of any D is J of (D + D^T) / 2.
The J_ji blocks are J_ij transposed.

Schwarz screen, optional (``tau = 0`` skips nothing): with
Q_p = sqrt(max_ab (ab|ab)) over the pair's diagonal block, |(ab|cd)| <= Q_p Q_q
for every element, so a quartet moves no element of J_p by more than
Q_p Q_q SUM_cd |w D_cd|, nor of J_q by more than Q_p Q_q SUM_ab |w D_ab|. A
quartet is skipped when the larger of the two is under ``tau``; each skipped
quartet's two bounds are summed into ``JBuild.bound``, elementwise, so
|J(tau) - J(0)| <= bound holds with no tolerance but rounding.

The ERI evaluator is injected (``eri=``, default ``rys2d.contracted_block``):
``check_j.py`` passes a cached one to walk a tau ladder on one set of blocks.
s and p shells only, as ``rys2d``: the roots stop at N = 3, i.e. (pp|pp).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from rys2d import Shell, contracted_block, ncart

Eri = Callable[[Shell, Shell, Shell, Shell], np.ndarray]


def ao_offsets(shells: Sequence[Shell]) -> np.ndarray:
    """AO offset of each shell, Cartesian, with the total at the end
    (PySCF's ``ao_loc_nr(cart=True)``)."""
    return np.concatenate([[0], np.cumsum([ncart(s.am) for s in shells])])


def shell_pairs(n_shells: int) -> list[tuple[int, int]]:
    """The unique shell pairs (i, j), i <= j, in sweep order."""
    return [(i, j) for i in range(n_shells) for j in range(i, n_shells)]


def schwarz(shells: Sequence[Shell], eri: Eri = contracted_block) -> np.ndarray:
    """Q[i, j] = sqrt(max_ab (ab|ab)) over shells i and j, symmetric."""
    n = len(shells)
    q = np.zeros((n, n))
    for i, j in shell_pairs(n):
        blk = eri(shells[i], shells[j], shells[i], shells[j])
        na, nb = blk.shape[:2]
        diag = blk.reshape(na * nb, na * nb).diagonal()
        q[i, j] = q[j, i] = np.sqrt(max(diag.max(), 0.0))
    return q


@dataclass(frozen=True)
class JBuild:
    """One build: ``J`` and what the screen did."""

    J: np.ndarray
    n_quartets: int  # unique shell quartets examined
    n_skipped: int
    bound: np.ndarray  # elementwise bound on |J - J(tau = 0)|; 0 when nothing skipped


def coulomb_matrix(
    shells: Sequence[Shell],
    D: np.ndarray,
    tau: float = 0.0,
    eri: Eri = contracted_block,
) -> JBuild:
    """J_ab = SUM_cd (ab|cd) D_cd over ``shells``' Cartesian AOs."""
    if tau < 0.0:
        raise ValueError(f"tau = {tau} is negative")
    loc = ao_offsets(shells)
    n_ao = int(loc[-1])
    D = np.asarray(D, dtype=float)
    if D.shape != (n_ao, n_ao):
        raise ValueError(f"D is {D.shape}, the basis has {n_ao} AOs")
    Ds = 0.5 * (D + D.T)
    pairs = shell_pairs(len(shells))
    sl = [slice(loc[i], loc[i + 1]) for i in range(len(shells))]
    wD = [Ds[sl[i], sl[j]] * (1.0 if i == j else 2.0) for i, j in pairs]
    dsum = [np.abs(w).sum() for w in wD]
    Q = schwarz(shells, eri) if tau > 0.0 else None
    Jp = [np.zeros(w.shape) for w in wD]
    Bp = [np.zeros(w.shape) for w in wD]
    skipped = 0
    for ip, (i, j) in enumerate(pairs):
        for iq in range(ip, len(pairs)):
            k, l_ = pairs[iq]
            if Q is not None:
                qq = Q[i, j] * Q[k, l_]
                to_p, to_q = qq * dsum[iq], qq * dsum[ip]
                if max(to_p, to_q) < tau:
                    skipped += 1
                    Bp[ip] += to_p
                    if iq != ip:
                        Bp[iq] += to_q
                    continue
            blk = eri(shells[i], shells[j], shells[k], shells[l_])
            Jp[ip] += np.einsum("abcd,cd->ab", blk, wD[iq])
            if iq != ip:
                Jp[iq] += np.einsum("abcd,ab->cd", blk, wD[ip])
    J = np.zeros((n_ao, n_ao))
    bound = np.zeros((n_ao, n_ao))
    for (i, j), jb, bb in zip(pairs, Jp, Bp, strict=True):
        J[sl[i], sl[j]] = jb
        J[sl[j], sl[i]] = jb.T
        bound[sl[i], sl[j]] = bb
        bound[sl[j], sl[i]] = bb.T
    n = len(pairs)
    return JBuild(J, n * (n + 1) // 2, skipped, bound)


def coulomb_energy(D: np.ndarray, J: np.ndarray) -> float:
    """E_J = (1/2) Tr(D J)."""
    return 0.5 * float(np.einsum("ab,ab->", D, J))
