"""The closed-shell RHF nuclear hessian: second-derivative integrals
(``ints2.py``), the gradient spike's first-derivative integrals, and the
coupled-perturbed Hartree-Fock (CPHF) response.

The gradient at every geometry, with ``D`` and ``W`` the converged densities
(``experimental/mmd/gradient/ints.py``'s ``rhf_gradient``), is

    g_x = sum D h^x + 1/2 sum D D (mn||ls)^x - sum W S^x + E_nuc^x,
    (mn||ls) = (mn|ls) - 1/2 (ml|sn).

Differentiating it once more, ``D`` and ``W`` now moving with the nuclei:

    H_xy = E_nuc^xy + sum D h^xy + 1/2 sum D D (mn||ls)^xy - sum W S^xy   (skeleton)
         + sum D^y F^x - sum W^y S^x                                      (response)

with ``F^x = h^x + sum_ls (mn||ls)^x D_ls`` the skeleton Fock derivative
(``D`` held fixed). ``D^y`` and ``W^y`` come from the orbital response
``C^y = C U^y``: orthonormality gives ``U_pq + U_qp + S^y_pq = 0`` (MO basis),
the occupied-occupied block is fixed at ``U_ij = -1/2 S^y_ij`` (``D`` and
``W`` do not depend on that choice), and keeping ``F_ai = 0`` gives the CPHF
equations for the virtual-occupied block,

    (e_a - e_i) U_ai + sum_bj [4 (ai|bj) - (ab|ij) - (aj|bi)] U_bj
        = -(F^y_ai - e_i S^y_ai + G[D_S]_ai),       D_S = -2 C_o S^y_oo C_o^T,

``G[X] = J[X] - 1/2 K[X]``, solved densely for every perturbation at once.
Then ``D^y = 2 (C^y_o C_o^T + C_o C^y_o^T)`` and, from ``W = 2 C_o F_oo C_o^T``,

    W^y = 2 (C^y_o F_oo C_o^T + C_o F_oo C^y_o^T + C_o dF_oo C_o^T),
    dF_oo = C^y_o^T F C_o + C_o^T F C^y_o + C_o^T (F^y + G[D^y]) C_o.

Hessian arrays are ``H[A, k, B, l]``; PySCF's ``(A, B, k, l)`` is
``H.transpose(0, 2, 1, 3)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradient"))

from ints import (  # noqa: E402
    coulomb,
    coulomb_deriv,
    exchange,
    exchange_deriv,
    one_electron,
    two_electron,
)
from ints2 import nuclear_repulsion2, one_electron2, two_electron2  # noqa: E402


def _g(eri, X):
    """``G[X] = J[X] - 1/2 K[X]`` for every leading index of ``X[..., n, n]``."""
    return np.einsum("mnls,...ls->...mn", eri, X) - 0.5 * np.einsum(
        "mlsn,...ls->...mn", eri, X
    )


def cphf(C, e, nocc, eri, F0, F1, S1):
    """The density responses ``D1, W1`` (each shaped like ``F1``) and ``U``.

    ``C``, ``e``: MO coefficients and energies, occupied first; ``F0``: the
    AO Fock matrix; ``F1``, ``S1``: ``[..., nao, nao]`` skeleton Fock and
    overlap derivatives, one per perturbation."""
    lead = F1.shape[:-2]
    nao, nmo = C.shape
    F1 = F1.reshape(-1, nao, nao)
    S1 = S1.reshape(-1, nao, nao)
    nx = len(F1)
    o, v = slice(0, nocc), slice(nocc, nmo)
    Co, eo, ev = C[:, o], e[o], e[v]
    nv = nmo - nocc

    eri_mo = np.einsum("mp,nq,ls,tr,mnlt->pqsr", C, C, C, C, eri, optimize=True)
    A = (
        4.0 * eri_mo[v, o, v, o]
        - eri_mo[v, v, o, o].transpose(0, 2, 1, 3)
        - eri_mo[v, o, v, o].transpose(0, 3, 2, 1)
    ).reshape(nv * nocc, nv * nocc)
    A += np.diag(np.subtract.outer(ev, eo).ravel())

    F1mo = np.einsum("mp,xmn,nq->xpq", C, F1, C)
    S1mo = np.einsum("mp,xmn,nq->xpq", C, S1, C)
    U = np.zeros((nx, nmo, nocc))
    U[:, o] = -0.5 * S1mo[:, o, o]
    DS = -2.0 * np.einsum("mi,xij,nj->xmn", Co, S1mo[:, o, o], Co)
    GS = np.einsum("mp,xmn,nq->xpq", C, _g(eri, DS), C)
    rhs = -(F1mo[:, v, o] - S1mo[:, v, o] * eo + GS[:, v, o])
    U[:, v] = np.linalg.solve(A, rhs.reshape(nx, -1).T).T.reshape(nx, nv, nocc)

    C1 = np.einsum("mp,xpi->xmi", C, U)
    D1 = 2.0 * (np.einsum("xmi,ni->xmn", C1, Co) + np.einsum("mi,xni->xmn", Co, C1))
    dF = F1 + _g(eri, D1)
    Foo = np.diag(eo)
    dFoo = (
        np.einsum("xmi,mn,nj->xij", C1, F0, Co)
        + np.einsum("mi,mn,xnj->xij", Co, F0, C1)
        + np.einsum("mi,xmn,nj->xij", Co, dF, Co)
    )
    W1 = 2.0 * (
        np.einsum("xmi,ij,nj->xmn", C1, Foo, Co)
        + np.einsum("mi,ij,xnj->xmn", Co, Foo, C1)
        + np.einsum("mi,xij,nj->xmn", Co, dFoo, Co)
    )
    shape = lead + (nao, nao)
    return D1.reshape(shape), W1.reshape(shape), U.reshape(lead + (nmo, nocc))


def rhf_hessian(shells, charges, coords, C, e, nocc):
    """The closed-shell RHF hessian ``[natm, 3, natm, 3]`` and its parts.

    ``C``, ``e``: converged MO coefficients and energies (occupied first),
    ``nocc`` doubly occupied orbitals. Parts: ``nuc``, ``T``, ``V``, ``J``,
    ``K``, ``S`` (the skeleton) and ``response`` (``D^y F^x - W^y S^x``)."""
    charges = np.asarray(charges, dtype=float)
    coords = np.asarray(coords, dtype=float)
    natm = len(charges)
    S, T, V, dS, dT, dV = one_electron(shells, charges, coords)
    eri, deri = two_electron(shells, natm)
    d2S, d2T, d2V = one_electron2(shells, charges, coords)
    d2eri = two_electron2(shells, natm)

    Co = C[:, :nocc]
    D = 2.0 * Co @ Co.T
    W = 2.0 * (Co * e[:nocc]) @ Co.T
    F0 = T + V + coulomb(eri, D) - 0.5 * exchange(eri, D)
    F1 = dT + dV + coulomb_deriv(deri, D) - 0.5 * exchange_deriv(deri, D)
    D1, W1, _ = cphf(C, e, nocc, eri, F0, F1, dS)

    dd = np.tensordot(d2eri, D, axes=([6, 7], [0, 1]))  # sum_ls d2(mn|ls) D_ls
    dx = np.einsum("aibjmlsn,ls->aibjmn", d2eri, D)  # sum_ls d2(ml|sn) D_ls
    parts = {
        "nuc": nuclear_repulsion2(charges, coords),
        "T": np.einsum("akblmn,mn->akbl", d2T, D),
        "V": np.einsum("akblmn,mn->akbl", d2V, D),
        "J": 0.5 * np.einsum("akblmn,mn->akbl", dd, D),
        "K": -0.25 * np.einsum("akblmn,mn->akbl", dx, D),
        "S": -np.einsum("akblmn,mn->akbl", d2S, W),
        "response": np.einsum("akmn,blmn->akbl", F1, D1)
        - np.einsum("akmn,blmn->akbl", dS, W1),
    }
    return sum(parts.values()), parts
