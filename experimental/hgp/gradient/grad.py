"""Analytic nuclear derivatives of ``(ab|cd)``, ``J``, ``K`` and ``G = J - K/2``
by the HGP engine.

    quartet(sa, sb, sc, sd)                 -> (ab|cd), d(ab|cd)/d(centre)
    two_electron(shells, atoms, natm)       -> (ab|cd), d(ab|cd)/dR, dense
    coulomb, exchange, fock                 -> J, K, G from (ab|cd) and D
    coulomb_deriv, exchange_deriv, fock_deriv
                                            -> dJ, dK, dG from d(ab|cd)/dR and D
    two_electron_gradient(shells, atoms, natm, D)
                                            -> the J, K and G energy gradients,
                                               fused: no four-index tensor

Shells are ``felizjueves.ao.shell.Shell`` (any ``l``; ``atoms[i]`` is the atom
shell ``i`` sits on). Derivative arrays are ``dX[atom, k, ...]``, the
derivative with respect to the ``k``-th coordinate of the nucleus ``atom``.

THE RULE. For a primitive Cartesian Gaussian on centre ``A``,

    d/dA_k [x_A^i exp(-a x_A^2)] = 2a g_{i+1_k} - i g_{i-1_k},

so the derivative of ``(ab|cd)`` on the centre of ``a`` is two ordinary
integrals of the classes ``(la+1 lb|lc ld)`` and ``(la-1 lb|lc ld)``. The
first carries ``2a`` per PRIMITIVE, which is what makes it an HGP integral
and not an MMD one: ``[Fock]`` §3.1's split contracts the ``K^4`` primitive
quartets between the VRR and the HRR (eq 28 holds only on contracted values,
milestone 12's pin 5), so ``2a`` cannot be applied after the HRR. It is
folded into the contraction instead, as ``experimental/hgp/one_electron``
folds the kinetic ket's ``b^n``: every VRR value is linear in the seed scale
``pair`` (``eri.hgp_drive``, "THE BINDING"), so the raised class runs the
ordinary VRR with ``pair`` multiplied by ``2a`` lane by lane, the lanes are
summed, and the ordinary HRR (whose ``AB``, ``CD`` carry no exponent) runs
once on that weighted contracted boundary. The lowered class is the
ordinary contracted integral. Each centre is done this way: four raised and
up to four lowered classes per shell quartet, and translational invariance
is NOT used to infer one centre from the other three (``check.py`` tests it).

THE KERNELS are ``hgp.recurrence``'s VRR and HRR as ``eri.hgp_drive`` builds
them (``hgp_drive._kernels``, memoised per class), run over every primitive
quartet at once by ``lanes.compile_kernel``; the binding is
``hgp_drive.primitive_scalars`` spelled for arrays (``_lane_scalars``), and
``check.py`` holds the whole route to ``hgp_drive.eri_block`` on the
interpreter. No recurrence is restated and no ``src/`` file is touched.

THE SWEEPS. ``two_electron`` evaluates each unique shell quartet
(``i <= j``, ``k <= l``, ``ij <= kl``) once and assigns its value and its
atom-resolved derivative over the 8-fold orbit (each ``d(ab|cd)/dR`` has the
same symmetry as ``(ab|cd)``, ``dR`` being a nuclear coordinate).
``two_electron_gradient`` is the fused form, ``eri.fock``'s digestion read for
a derivative: the same unique sweep, each distinct ordered member of the
orbit contracted straight into

    dE_J = 1/2 SUM D_ab D_cd d(ab|cd),   dE_K = -1/4 SUM D_ac D_bd d(ab|cd),

``dE_G = dE_J + dE_K`` (the two-electron part of the RHF gradient with ``D``
the total density, ``G = J - K/2``), so nothing of size ``nao^4`` is formed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache

import numpy as np
from lanes import compile_kernel

from felizjueves.ao.shell import Shell, cartesians, n_cartesians
from felizjueves.eri.hgp_drive import _kernels
from felizjueves.hgp import recurrence as R

__all__ = ["block", "coulomb", "coulomb_deriv", "exchange", "exchange_deriv",
           "FORM", "fock", "fock_deriv", "offsets", "quartet", "two_electron",
           "two_electron_gradient"]

_AXES = "xyz"
# Every factor of the derivative rule and the digestion, as a multiplier, so
# check.py can perturb each one alone and show a table catch it.
FORM = {"raise": 2.0, "lower": 1.0, "J": 0.5, "K": -0.25}
# The 8-fold orbit of (a b|c d), as position permutations.
ORBIT = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
         (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0))


def offsets(shells: Sequence[Shell]) -> tuple[int, ...]:
    out = [0]
    for s in shells:
        out.append(out[-1] + s.n_cart)
    return tuple(out)


def _lane_scalars(q: tuple[Shell, Shell, Shell, Shell]) -> dict[str, np.ndarray]:
    """``hgp_drive.primitive_scalars`` for every primitive quartet of ``q``
    at once, lanes in its ``i, j, k, l`` order (``l`` fastest)."""
    ex = [np.array(s.exponents) for s in q]
    cf = [np.array(s.coefficients) for s in q]
    a, b, c, d = np.meshgrid(*ex, indexing="ij")
    ca, cb, cc, cd = np.meshgrid(*cf, indexing="ij")
    a, b, c, d = (x.ravel() for x in (a, b, c, d))
    ca, cb, cc, cd = (x.ravel() for x in (ca, cb, cc, cd))
    A, B, C, D = (np.array(s.centre) for s in q)
    p, qq = a + b, c + d
    P = (np.outer(a, A) + np.outer(b, B)) / p[:, None]
    Q = (np.outer(c, C) + np.outer(d, D)) / qq[:, None]
    PQ = P - Q
    W = (p[:, None] * P + qq[:, None] * Q) / (p + qq)[:, None]
    rho = p * qq / (p + qq)
    AB, CD = A - B, C - D
    kab = np.exp(-a * b / p * float(AB @ AB))
    kcd = np.exp(-c * d / qq * float(CD @ CD))
    s: dict[str, np.ndarray] = {}
    for x, ax in enumerate(_AXES):
        s[f"PA_{ax}"] = P[:, x] - A[x]
        s[f"WP_{ax}"] = W[:, x] - P[:, x]
        s[f"QC_{ax}"] = Q[:, x] - C[x]
        s[f"WQ_{ax}"] = p / (p + qq) * PQ[:, x]
        s[f"PQ_{ax}"] = PQ[:, x]
    s["inv2p"] = 1.0 / (2.0 * p)
    s["inv2q"] = 1.0 / (2.0 * qq)
    s["inv2pq"] = 1.0 / (2.0 * (p + qq))
    s["inv2p_alpha_over_p"] = 1.0 / (2.0 * p) * (rho / p)
    s["inv2q_alpha_over_q"] = 1.0 / (2.0 * qq) * (rho / qq)
    s[R.THETA] = np.sqrt(rho)
    s[R.PAIR] = ((ca * cb * (math.pi / p) ** 1.5 * kab)
                 * (cc * cd * (math.pi / qq) ** 1.5 * kcd))
    s["_exp"] = (a, b, c, d)  # per lane, for the derivative weights
    return s


@cache
def _compiled(cls: tuple[int, int, int, int]):
    vk, hk = _kernels(cls, "bra", False)
    return compile_kernel(vk), compile_kernel(hk)


def block(q: tuple[Shell, Shell, Shell, Shell], ls: tuple[int, int, int, int],
          s: dict[str, np.ndarray], weight: np.ndarray | None = None
          ) -> np.ndarray:
    """The contracted HGP integral of the class ``ls`` on ``q``'s centres and
    primitives (``s``, from ``_lane_scalars``), each lane's ``pair`` times
    ``weight``: ``[n_cart(ls[0]), ..., n_cart(ls[3])]``. VRR on every lane,
    the lanes summed, the HRR once (module docstring)."""
    vrr, hrr = _compiled(ls)
    pair = s[R.PAIR] if weight is None else s[R.PAIR] * weight
    rows = vrr({**s, R.PAIR: pair}, {})
    boundary = [float(np.sum(r)) for r in rows]
    geo = {}
    for x, ax in enumerate(_AXES):
        geo[f"AB_{ax}"] = q[0].centre[x] - q[1].centre[x]
        geo[f"CD_{ax}"] = q[2].centre[x] - q[3].centre[x]
    out = hrr(geo, {R.BOUNDARY: boundary})
    return np.array(out, dtype=float).reshape(
        tuple(n_cartesians(x) for x in ls))


@cache
def _shift_maps(l: int, k: int):  # noqa: E741
    """For each Cartesian of ``l``: its index in ``l + 1`` raised along
    ``k``, its index in ``l - 1`` lowered along ``k`` (0 if it cannot be),
    and the lowering factor ``i_k``."""
    up_c = {c: n for n, c in enumerate(cartesians(l + 1))}
    dn_c = {c: n for n, c in enumerate(cartesians(l - 1))} if l else {}
    up, dn, fac = [], [], []
    for c in cartesians(l):
        r = list(c)
        r[k] += 1
        up.append(up_c[tuple(r)])
        if c[k]:
            r[k] -= 2
            dn.append(dn_c[tuple(r)])
        else:
            dn.append(0)
        fac.append(float(c[k]))
    return np.array(up), np.array(dn), np.array(fac)


def quartet(sa: Shell, sb: Shell, sc: Shell, sd: Shell
            ) -> tuple[np.ndarray, np.ndarray]:
    """``(ab|cd)`` ``[na, nb, nc, nd]`` and its derivative on each of the four
    centres, ``[4, 3, na, nb, nc, nd]`` (position, axis): per position one
    raised class, weighted by twice that position's exponent, and one lowered
    class (module docstring, "THE RULE")."""
    q = (sa, sb, sc, sd)
    ls = tuple(x.l for x in q)
    s = _lane_scalars(q)
    val = block(q, ls, s)
    der = np.zeros((4, 3) + val.shape)
    for pos in range(4):
        up_ls = tuple(x + (i == pos) for i, x in enumerate(ls))
        up = block(q, up_ls, s, FORM["raise"] * s["_exp"][pos])
        dn = None
        if ls[pos]:
            dn = block(q, tuple(x - (i == pos) for i, x in enumerate(ls)), s)
        for k in range(3):
            iu, idn, fac = _shift_maps(ls[pos], k)
            d = np.take(up, iu, axis=pos)
            if dn is not None:
                shape = [1, 1, 1, 1]
                shape[pos] = len(fac)
                d = d - FORM["lower"] * fac.reshape(shape) * np.take(dn, idn, axis=pos)
            der[pos, k] = d
    return val, der


def _unique(n: int):
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    for x, (i, j) in enumerate(pairs):
        for k, l in pairs[x:]:  # noqa: E741
            yield i, j, k, l


def _by_atom(der: np.ndarray, at: Sequence[int], natm: int) -> np.ndarray:
    """``[4, 3, ...]`` per position -> ``[natm, 3, ...]`` per nucleus."""
    out = np.zeros((natm,) + der.shape[1:])
    for pos in range(4):
        out[at[pos]] += der[pos]
    return out


def two_electron(shells: Sequence[Shell], atoms: Sequence[int], natm: int,
                 *, deriv: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """``(ab|cd)`` ``[nao]*4`` and ``d(ab|cd)/dR`` ``[natm, 3] + [nao]*4``,
    dense: one evaluation per unique shell quartet, assigned over its orbit.
    With ``deriv=False`` only the integrals (one class per quartet)."""
    off = offsets(shells)
    nao = off[-1]
    eri = np.zeros((nao,) * 4)
    deri = np.zeros((natm, 3) + (nao,) * 4) if deriv else None
    for idx in _unique(len(shells)):
        q = tuple(shells[i] for i in idx)
        if deriv:
            val, der = quartet(*q)
            g = _by_atom(der, [atoms[i] for i in idx], natm)
        else:
            val = block(q, tuple(x.l for x in q), _lane_scalars(q))
        for perm in ORBIT:
            sl = tuple(slice(off[idx[p]], off[idx[p] + 1]) for p in perm)
            eri[sl] = val.transpose(perm)
            if deriv:
                deri[(slice(None), slice(None)) + sl] = g.transpose(
                    (0, 1) + tuple(2 + p for p in perm))
    return eri, deri


def two_electron_gradient(shells: Sequence[Shell], atoms: Sequence[int],
                          natm: int, D: np.ndarray) -> dict[str, np.ndarray]:
    """``{"J", "K", "G"}``, each ``[natm, 3]``: ``1/2 SUM D dJ``,
    ``-1/4 SUM D dK`` and their sum, by the fused sweep (module docstring):
    every distinct ordered member of each unique quartet's orbit contracted
    with its density blocks as it forms, nothing of size ``nao^4`` kept."""
    off = offsets(shells)
    gJ, gK = np.zeros((natm, 3)), np.zeros((natm, 3))
    for idx in _unique(len(shells)):
        _, der = quartet(*(shells[i] for i in idx))
        g = _by_atom(der, [atoms[i] for i in idx], natm)
        seen = set()
        for perm in ORBIT:
            m = tuple(idx[p] for p in perm)
            if m in seen:
                continue
            seen.add(m)
            gm = g.transpose((0, 1) + tuple(2 + p for p in perm))
            a, b, c, d = (slice(off[i], off[i + 1]) for i in m)
            gJ += FORM["J"] * np.einsum("xyabcd,ab,cd->xy", gm, D[a, b], D[c, d])
            gK += FORM["K"] * np.einsum("xyabcd,ac,bd->xy", gm, D[a, c], D[b, d])
    return {"J": gJ, "K": gK, "G": gJ + gK}


def coulomb(eri: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.einsum("mnls,ls->mn", eri, D)


def exchange(eri: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.einsum("mlsn,ls->mn", eri, D)


def fock(eri: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``G = J - K/2``, ``eri.fock.fock``'s two-electron Fock matrix."""
    return coulomb(eri, D) - 0.5 * exchange(eri, D)


def coulomb_deriv(deri: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.einsum("xymnls,ls->xymn", deri, D)


def exchange_deriv(deri: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.einsum("xymlsn,ls->xymn", deri, D)


def fock_deriv(deri: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``dG = dJ - dK/2`` at fixed ``D``."""
    return coulomb_deriv(deri, D) - 0.5 * exchange_deriv(deri, D)
