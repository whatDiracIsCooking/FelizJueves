"""The density-fitting integrals ``(P|Q)`` and ``(mn|P)`` and their nuclear
derivatives by McMurchie-Davidson, and the DF-J, DF-K and DF-RHF gradient
built on them.

An auxiliary function is a pair with a unit s function: ``|P) = |P s_0)``, the
partner an s Gaussian of exponent 0 and coefficient 1 on the same centre, so
``p = alpha_P``, ``P = C``, ``E^{i0}_t`` is the single-centre expansion and
``E^{00}_0 = 1``. The four-centre machinery of ``ints.py`` then reads both
integrals unchanged: ``(P|Q) = (P s_0|Q s_0)`` and ``(mn|P) = (mn|P s_0)``,
the prefactor ``2 pi^{5/2} / (p q sqrt(p + q))`` included. The derivative on
the partner is zero (``d/dB`` of ``exp(0)``, ``mmd.d_centre`` at ``b = 0``),
so the aux side's derivative is its ``d/dA`` component alone, and as in
``ints.py`` every centre's derivative is computed, none inferred.

With ``gamma_P = sum_mn (P|mn) D_mn``, ``c = M^{-1} gamma`` and
``C^P = sum_Q M^{-1}_PQ (Q|..)``, ``D`` the total density held fixed:

    E_J = 1/2 gamma^T c
    dE_J = c^T dgamma - 1/2 c^T dM c
    E_K = -1/4 sum D_mn D_ls (ml|sn),  (ml|sn) ~ sum_PQ (ml|P) M^{-1}_PQ (Q|sn)
    dE_K = -1/4 [2 sum_P sum_ml d(ml|P) (D C^P D)_ml - sum_PQ dM_PQ Tr(C^P D C^Q D)]

Derivative arrays are ``dX[atom, k, ...]`` as in ``ints.py``.
"""

from __future__ import annotations

import numpy as np
from ints import _pair_data, _quartet, nuclear_repulsion, one_electron
from mmd import Shell, ao_offsets, cart


def _unit(s: Shell) -> Shell:
    """The exponent-0, coefficient-1 s partner that makes ``s`` a pair."""
    return Shell(0, s.centre, (0.0,), (1.0,), s.atom)


def _aux_data(aux, deriv):
    return [_pair_data(s, _unit(s), deriv) for s in aux]


def metric(aux, natm: int, deriv: bool = True):
    """``(P|Q)`` as ``[naux, naux]`` and, with ``deriv``, ``d(P|Q)/dR`` as
    ``[natm, 3, naux, naux]``. One sweep over ``P <= Q``, mirrored."""
    off = ao_offsets(aux)
    n = off[-1]
    M = np.zeros((n, n))
    dM = np.zeros((natm, 3, n, n)) if deriv else None
    data = _aux_data(aux, deriv)
    for i, si in enumerate(aux):
        for j in range(i, len(aux)):
            sj = aux[j]
            M1, M2 = _quartet(data[i], data[j], deriv)
            a, b = slice(off[i], off[i + 1]), slice(off[j], off[j + 1])
            M[a, b] = M1[0]
            M[b, a] = M1[0].T
            if not deriv:
                continue
            g = np.zeros((natm, 3) + M1[0].shape)
            for k in range(3):
                g[si.atom, k] += M1[1 + k]
                g[sj.atom, k] += M2[:, 1 + k]
            dM[:, :, a, b] = g
            dM[:, :, b, a] = g.transpose(0, 1, 3, 2)
    return M, dM


def three_center(shells, aux, natm: int, deriv: bool = True):
    """``(mn|P)`` as ``[nao, nao, naux]`` and, with ``deriv``, ``d(mn|P)/dR``
    as ``[natm, 3, nao, nao, naux]``. One sweep over the AO shell pairs
    ``i <= j`` and every aux shell, mirrored in ``(m, n)``."""
    off, xoff = ao_offsets(shells), ao_offsets(aux)
    nao, naux = off[-1], xoff[-1]
    T = np.zeros((nao, nao, naux))
    dT = np.zeros((natm, 3, nao, nao, naux)) if deriv else None
    adata = _aux_data(aux, deriv)
    for i, si in enumerate(shells):
        for j in range(i, len(shells)):
            sj = shells[j]
            bra = _pair_data(si, sj, deriv)
            a, b = slice(off[i], off[i + 1]), slice(off[j], off[j + 1])
            shape_ab = (len(cart(si.l)), len(cart(sj.l)))
            for x, sx in enumerate(aux):
                M1, M2 = _quartet(bra, adata[x], deriv)
                c = slice(xoff[x], xoff[x + 1])
                blk = M1[0].reshape(shape_ab + (-1,))
                T[a, b, c] = blk
                T[b, a, c] = blk.transpose(1, 0, 2)
                if not deriv:
                    continue
                g = np.zeros((natm, 3) + blk.shape)
                for k in range(3):
                    g[si.atom, k] += M1[1 + k].reshape(blk.shape)
                    g[sj.atom, k] += M1[4 + k].reshape(blk.shape)
                    g[sx.atom, k] += M2[:, 1 + k].reshape(blk.shape)
                dT[:, :, a, b, c] = g
                dT[:, :, b, a, c] = g.transpose(0, 1, 3, 2, 4)
    return T, dT


# --- DF-J, DF-K and the DF-RHF gradient ------------------------------------------


def df_eri(eri3, M):
    """``(mn|ls) ~ sum_PQ (mn|P) M^{-1}_PQ (Q|ls)``, dense."""
    return np.einsum("mnP,PQ,lsQ->mnls", eri3, np.linalg.inv(M), eri3)


def df_coulomb(eri3, M, D):
    """``J_mn = sum_P (mn|P) c_P``, ``c = M^{-1} gamma``."""
    c = np.linalg.solve(M, np.einsum("mnP,mn->P", eri3, D))
    return np.einsum("mnP,P->mn", eri3, c)


def df_exchange(eri3, M, D):
    """``K_mn = sum_ls (ml|sn)_DF D_ls``, PySCF's ``get_k`` convention."""
    C = np.linalg.solve(M, eri3.reshape(-1, M.shape[0]).T)  # C[P, m*l]
    C = C.reshape((M.shape[0],) + eri3.shape[:2])
    return np.einsum("mlP,Psn,ls->mn", eri3, C, D)


def df_j_gradient(eri3, deri3, M, dM, D):
    """``dE_J [natm, 3]``, ``E_J = 1/2 Tr D J_DF``."""
    c = np.linalg.solve(M, np.einsum("mnP,mn->P", eri3, D))
    dgamma = np.einsum("akmnP,mn->akP", deri3, D)
    return dgamma @ c - 0.5 * np.einsum("P,akPQ,Q->ak", c, dM, c)


def df_k_gradient(eri3, deri3, M, dM, D):
    """``dE_K [natm, 3]``, ``E_K = -1/4 Tr D K_DF``."""
    naux = M.shape[0]
    C = np.linalg.solve(M, eri3.reshape(-1, naux).T).reshape((naux,) + eri3.shape[:2])
    DCD = np.einsum("mn,Pns,sl->Pml", D, C, D)
    term3 = np.einsum("akmlP,Pml->ak", deri3, DCD)
    CD = np.einsum("Pmn,nl->Pml", C, D)
    G = np.einsum("Pml,Qlm->PQ", CD, CD)
    term2 = np.einsum("akPQ,PQ->ak", dM, G)
    return -0.25 * (2.0 * term3 - term2)


def df_rhf_energy(S, T, V, eri3, M, D, charges, coords):
    h = T + V
    J, K = df_coulomb(eri3, M, D), df_exchange(eri3, M, D)
    return (
        np.sum(D * h)
        + 0.5 * np.sum(D * J)
        - 0.25 * np.sum(D * K)
        + nuclear_repulsion(charges, coords)[0]
    )


def df_rhf_gradient(shells, aux, charges, coords, D, W):
    """The closed-shell DF-RHF gradient ``[natm, 3]`` and its parts, the
    auxiliary-basis response (``dM`` and the aux centre of ``d(mn|P)``)
    included; ``D`` and ``W`` as in ``ints.rhf_gradient``."""
    natm = len(charges)
    _, _, _, dS, dT, dV = one_electron(shells, charges, coords)
    eri3, deri3 = three_center(shells, aux, natm)
    M, dM = metric(aux, natm)
    parts = {
        "nuc": nuclear_repulsion(charges, coords)[1],
        "T": np.einsum("akmn,mn->ak", dT, D),
        "V": np.einsum("akmn,mn->ak", dV, D),
        "J": df_j_gradient(eri3, deri3, M, dM, D),
        "K": df_k_gradient(eri3, deri3, M, dM, D),
        "S": -np.einsum("akmn,mn->ak", dS, W),
    }
    return sum(parts.values()), parts
