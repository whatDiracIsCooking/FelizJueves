"""The exchange matrix on the Rys ERIs: K_ad = SUM_bc (ab|cd) D_bc.

PySCF's convention (``get_jk``, ``einsum('ijkl,jk->il')``), for any D,
symmetric or not.

One sweep over the unique shell quartets, as ``jbuild``: bra pair (i, j) with
i <= j, ket pair (k, l) with k <= l, bra <= ket in pair order. Each block is
evaluated once and digested once per **distinct** shell quartet of its 8-fold
orbit, (ij|kl) (ji|kl) (ij|lk) (ji|lk) (kl|ij) (lk|ij) (kl|ji) (lk|ji), the
block transposed to match, each at weight 1:

    K[x0, x3] += SUM_bc blk'[a, b, c, d] D[x1, x2]_bc    for blk' over (x0 x1|x2 x3)

Every ordered shell quartet lies in exactly one orbit and is distinct once in
it, so this is the dense SUM over every ordered quartet with no weights to
derive: a degenerate quartet (i = j, k = l or ij = kl) simply has fewer
distinct members. Unlike J, D is not symmetrised: K of a non-symmetric D is not
K of its symmetric part.

Schwarz screen, optional (``tau = 0`` skips nothing): |(ab|cd)| <= Q_ij Q_kl
for every element, so the member (x0 x1|x2 x3) moves no element of the
K[x0, x3] block by more than Q_ij Q_kl SUM|D[x1, x2]|. The density block is
the exchange one (bra-ket mixed), not J's pair block. A quartet is skipped
when every member's bound is under ``tau``; each skipped member's bound is
summed into ``KBuild.bound``, elementwise, so |K(tau) - K(0)| <= bound holds
with no tolerance but rounding. It is a ``continue``, not a sorted ``break``.

The ERI evaluator is injected (``eri=``, default ``rys2d.contracted_block``).
s and p shells only, as ``rys2d``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jbuild import Eri, ao_offsets, schwarz, shell_pairs
from rys2d import Shell, contracted_block

# The 8-fold orbit as axis permutations: member m's shell quartet is
# (q[m[0]], ..., q[m[3]]) and its block is blk.transpose(m).
ORBIT = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def orbit_members(quartet: tuple[int, int, int, int]) -> list[tuple[int, ...]]:
    """The axis permutations giving each distinct shell quartet of
    ``quartet``'s orbit once, first occurrence in ``ORBIT`` order."""
    seen: set[tuple[int, ...]] = set()
    members = []
    for m in ORBIT:
        member = tuple(quartet[p] for p in m)
        if member not in seen:
            seen.add(member)
            members.append(m)
    return members


@dataclass(frozen=True)
class KBuild:
    """One build: ``K`` and what the screen did."""

    K: np.ndarray
    n_quartets: int  # unique shell quartets examined
    n_skipped: int
    bound: np.ndarray  # elementwise bound on |K - K(tau = 0)|; 0 when nothing skipped


def exchange_matrix(
    shells: Sequence[Shell],
    D: np.ndarray,
    tau: float = 0.0,
    eri: Eri = contracted_block,
) -> KBuild:
    """K_ad = SUM_bc (ab|cd) D_bc over ``shells``' Cartesian AOs."""
    if tau < 0.0:
        raise ValueError(f"tau = {tau} is negative")
    loc = ao_offsets(shells)
    n_ao = int(loc[-1])
    D = np.asarray(D, dtype=float)
    if D.shape != (n_ao, n_ao):
        raise ValueError(f"D is {D.shape}, the basis has {n_ao} AOs")
    n_sh = len(shells)
    sl = [slice(loc[i], loc[i + 1]) for i in range(n_sh)]
    dsum = np.array(
        [[np.abs(D[sl[b], sl[c]]).sum() for c in range(n_sh)] for b in range(n_sh)]
    )
    Q = schwarz(shells, eri) if tau > 0.0 else None
    K = np.zeros((n_ao, n_ao))
    bound = np.zeros((n_ao, n_ao))
    pairs = shell_pairs(n_sh)
    skipped = 0
    for ip, (i, j) in enumerate(pairs):
        for iq in range(ip, len(pairs)):
            quartet = (i, j, *pairs[iq])
            members = [
                (m, tuple(quartet[p] for p in m)) for m in orbit_members(quartet)
            ]
            if Q is not None:
                qq = Q[i, j] * Q[quartet[2], quartet[3]]
                bounds = [qq * dsum[x[1], x[2]] for _, x in members]
                if max(bounds) < tau:
                    skipped += 1
                    for (_, x), b in zip(members, bounds, strict=True):
                        bound[sl[x[0]], sl[x[3]]] += b
                    continue
            blk = eri(*(shells[s] for s in quartet))
            for m, x in members:
                K[sl[x[0]], sl[x[3]]] += np.einsum(
                    "abcd,bc->ad", blk.transpose(m), D[sl[x[1]], sl[x[2]]]
                )
    n = len(pairs)
    return KBuild(K, n * (n + 1) // 2, skipped, bound)


def exchange_energy(D: np.ndarray, K: np.ndarray) -> float:
    """E_K = (1/2) Tr(D K), PySCF's ``einsum('ij,ji', D, K)`` / 2."""
    return 0.5 * float(np.einsum("ab,ba->", D, K))
