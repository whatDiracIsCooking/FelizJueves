"""The fused Fock build on the Rys ERIs: G = J - K/2 from one evaluation per block.

    J_ab = SUM_cd (ab|cd) D_cd        K_ad = SUM_bc (ab|cd) D_bc

PySCF's conventions (``get_jk``, ``hermi=0``), for any D, symmetric or not, and
G = J - EXCHANGE_FRACTION K, closed-shell restricted (D the total density; the
1/2 is 1/N_spin, derived in ``src/felizjueves/eri/fock/fock.py``).

``jbuild`` and ``kbuild`` each sweep the unique shell quartets and each
evaluate every block, so G from them costs two evaluations per quartet. This
is the fused scheme of the HGP route (``[Fock]`` §3.2, ``eri/fock/kernel_f.py``):
one sweep, each block evaluated **once** and digested into both J and K. The
digestion is ``kbuild``'s: once per **distinct** shell quartet of the 8-fold
orbit (``kbuild.ORBIT``), the block transposed to match, each at weight 1,

    J[x0, x1] += SUM_cd blk'[a, b, c, d] D[x2, x3]_cd
    K[x0, x3] += SUM_bc blk'[a, b, c, d] D[x1, x2]_bc    for blk' over (x0 x1|x2 x3).

Every ordered shell quartet lies in exactly one orbit and is distinct once in
it, so both are the dense sums over every ordered quartet, with no weights to
derive: the Coulomb half needs neither ``jbuild``'s factor 2 on an
off-diagonal pair nor its symmetrised D, because it no longer relies on
(ab|cd) = (ab|dc) to fold the member (ij|lk) into (ij|kl). That costs up to
eight Coulomb digestions per block where ``jbuild`` makes two; the integrals,
not the digestion, are the expense here.

Schwarz screen, optional (``tau = 0`` skips nothing): |(ab|cd)| <= Q_ij Q_kl
for every element, so the member (x0 x1|x2 x3) moves no element of the
J[x0, x1] block by more than Q_ij Q_kl SUM|D[x2, x3]| and no element of the
K[x0, x3] block by more than Q_ij Q_kl SUM|D[x1, x2]|. A quartet's bound on G
is those, the K ones times EXCHANGE_FRACTION, summed per target shell block
(a J and a K term, or two members, may land on the same block). The quartet
is skipped when that summed bound is under ``tau`` on every block; each
skipped member's J and K bounds are summed into ``FBuild.bound_J`` and
``bound_K``, so |J(tau) - J(0)| <= bound_J, |K(tau) - K(0)| <= bound_K and
|G(tau) - G(0)| <= bound_G = bound_J + EXCHANGE_FRACTION bound_K hold
elementwise with no tolerance but rounding. The screen is on G, the quantity
built: a quartet J alone would skip can be kept for its exchange, and the
reverse. It is a ``continue``, not a sorted ``break``.

The ERI evaluator is injected (``eri=``, default ``rys2d.contracted_block``).
s and p shells only, as ``rys2d``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jbuild import Eri, ao_offsets, schwarz, shell_pairs
from kbuild import orbit_members
from rys2d import Shell, contracted_block

EXCHANGE_FRACTION = 0.5  # 1 / N_spin: closed-shell restricted


@dataclass(frozen=True)
class FBuild:
    """One build: G, its J and K, and what the screen did."""

    G: np.ndarray
    J: np.ndarray
    K: np.ndarray
    n_quartets: int  # unique shell quartets examined
    n_skipped: int  # the other n_quartets - n_skipped blocks were evaluated once
    bound_J: np.ndarray  # elementwise bound on |J - J(tau = 0)|
    bound_K: np.ndarray  # elementwise bound on |K - K(tau = 0)|

    @property
    def bound_G(self) -> np.ndarray:
        """Elementwise bound on |G - G(tau = 0)|; 0 when nothing skipped."""
        return self.bound_J + EXCHANGE_FRACTION * self.bound_K


def fock_matrix(
    shells: Sequence[Shell],
    D: np.ndarray,
    tau: float = 0.0,
    eri: Eri = contracted_block,
) -> FBuild:
    """G = J - K/2 over ``shells``' Cartesian AOs, each unique block
    evaluated once."""
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
    J = np.zeros((n_ao, n_ao))
    K = np.zeros((n_ao, n_ao))
    bound_J = np.zeros((n_ao, n_ao))
    bound_K = np.zeros((n_ao, n_ao))
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
                bj = [qq * dsum[x[2], x[3]] for _, x in members]
                bk = [qq * dsum[x[1], x[2]] for _, x in members]
                on_g: dict[tuple[int, int], float] = {}
                for (_, x), cj, ck in zip(members, bj, bk, strict=True):
                    on_g[x[0], x[1]] = on_g.get((x[0], x[1]), 0.0) + cj
                    on_g[x[0], x[3]] = (
                        on_g.get((x[0], x[3]), 0.0) + EXCHANGE_FRACTION * ck
                    )
                if max(on_g.values()) < tau:
                    skipped += 1
                    for (_, x), cj, ck in zip(members, bj, bk, strict=True):
                        bound_J[sl[x[0]], sl[x[1]]] += cj
                        bound_K[sl[x[0]], sl[x[3]]] += ck
                    continue
            blk = eri(*(shells[s] for s in quartet))
            for m, x in members:
                b = blk.transpose(m)
                J[sl[x[0]], sl[x[1]]] += np.einsum(
                    "abcd,cd->ab", b, D[sl[x[2]], sl[x[3]]]
                )
                K[sl[x[0]], sl[x[3]]] += np.einsum(
                    "abcd,bc->ad", b, D[sl[x[1]], sl[x[2]]]
                )
    n = len(pairs)
    return FBuild(
        J - EXCHANGE_FRACTION * K, J, K, n * (n + 1) // 2, skipped, bound_J, bound_K
    )


def fock_energy(D: np.ndarray, G: np.ndarray) -> float:
    """The two-electron energy E_2 = (1/2) Tr(D G), the two-electron part of
    PySCF's ``energy_elec`` (``einsum('ij,ji', D, G)`` / 2)."""
    return 0.5 * float(np.einsum("ab,ba->", D, G))
