"""Analytic second nuclear derivatives of ``(ab|cd)``, ``J``, ``K`` and
``G = J - K/2`` by the HGP engine, and their energy hessians.

    quartet2(sa, sb, sc, sd)                -> d2(ab|cd)/d(centre) d(centre)
    two_electron2(shells, atoms, natm)      -> d2(ab|cd)/dR dR, dense
    coulomb_deriv2, exchange_deriv2, fock_deriv2
                                            -> d2J, d2K, d2G at fixed D
    two_electron_hessian(shells, atoms, natm, D)
                                            -> the J, K and G energy hessians,
                                               fused: no four-index tensor

Built on ``experimental/hgp/gradient`` (its ``grad.py`` and ``lanes.py``
imported unchanged, by path): the same lanes, the same ``block`` (the VRR
over every primitive quartet with a per-lane weight on ``pair``, the lanes
summed, the HRR once on the contracted boundary), the same 8-fold orbit.
Shells are ``felizjueves.ao.shell.Shell``; ``atoms[i]`` is the atom shell
``i`` sits on. Second-derivative arrays are ``d2X[A, k, B, l, ...]``,
d/dR_{A,k} d/dR_{B,l} of the whole ``X``, every centre on each atom summed.

THE RULE, TWICE. One derivative on the centre of a primitive Cartesian
Gaussian ``g_i = x_A^i exp(-a x_A^2)`` is

    d/dA_k g_i = 2a g_{i+1_k} - i_k g_{i-1_k},

a linear map ``M_k = 2a U_k + L_k`` from the Cartesians of ``l`` to those of
``l + 1`` (``U_k``, coefficient 1) and ``l - 1`` (``L_k``, coefficient
``-i_k``). A second derivative on one centre is the product ``M_k M_m``:

    d2/dA_k dA_m g = (2a)^2 U_k U_m g[l+2] + 2a (U_k L_m + L_k U_m) g[l]
                   + L_k L_m g[l-2],

and on two different centres of the quartet the product of one ``M`` on
each: four classes ``(l_p +- 1, l_q +- 1)`` with weights
``(2a_p)^{0,1} (2a_q)^{0,1}``. So every second derivative is a handful of
ordinary contracted integrals of shifted classes, each weighted by a product
of exponents PER PRIMITIVE. HGP contracts between the VRR and the HRR
(``[Fock]`` eq 28 holds only on contracted values), so, as in the gradient
spike, the weight is folded into the seed ``pair`` lane by lane (every VRR
value is linear in it, ``eri.hgp_drive``, "THE BINDING") and the HRR runs
unchanged on the weighted contracted boundary. The class of a term fixes its
weight: at a centre carrying ``n`` derivatives a shift ``s`` has weight
``(2a)^{(s + n)/2}``. Per shell quartet that is at most 12 same-centre
blocks (four centres, ``s in {+2, 0, -2}``) and 24 cross-centre ones (six
centre pairs, ``(+-1, +-1)``), each evaluated once and mapped onto all nine
``(k, m)`` by the ``U`` / ``L`` matrices. All 16 ordered centre pairs are
filled (the ten with ``p <= q`` computed, the other six their transposes);
translational invariance is NOT used to infer any centre (``check.py``
tests it).

THE SWEEPS are the gradient spike's. ``two_electron2`` evaluates each unique
shell quartet once and assigns its atom-resolved second derivative over the
8-fold orbit (a nuclear coordinate leaves the orbit symmetry intact).
``two_electron_hessian`` is the fused form, ``eri.fock``'s digestion read
for a second derivative: each distinct ordered member of each orbit is
contracted as it forms into

    H_J = 1/2 SUM D_ab D_cd d2(ab|cd),   H_K = -1/4 SUM D_ac D_bd d2(ab|cd),

``H_G = H_J + H_K``, the two-electron skeleton of the RHF hessian (``D``
held fixed; the density response is ``hessian.py``'s), and nothing of size
``nao^4`` is formed.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradient"))

from grad import ORBIT, _lane_scalars, _unique, block, offsets  # noqa: E402

from felizjueves.ao.shell import Shell, cartesians  # noqa: E402

__all__ = ["coulomb_deriv2", "exchange_deriv2", "fock_deriv2", "FORM",
           "quartet2", "terms", "two_electron2", "two_electron_hessian"]

# Every factor of the rule and the digestion, as a multiplier, so check.py
# can perturb each one alone and show a table catch it.
FORM = {"raise": 2.0, "lower": 1.0, "J": 0.5, "K": -0.25}


@cache
def _up(l: int, k: int) -> np.ndarray:  # noqa: E741
    """``U_k``: ``[n_cart(l), n_cart(l + 1)]``, ``g_i -> g_{i+1_k}``."""
    to = {c: n for n, c in enumerate(cartesians(l + 1))}
    out = np.zeros((len(cartesians(l)), len(to)))
    for n, c in enumerate(cartesians(l)):
        r = list(c)
        r[k] += 1
        out[n, to[tuple(r)]] = 1.0
    return out


@cache
def _down_unit(l: int, k: int) -> np.ndarray:  # noqa: E741
    """``L_k`` without ``FORM["lower"]``: ``[n_cart(l), n_cart(l - 1)]``,
    ``g_i -> -i_k g_{i-1_k}``."""
    to = {c: n for n, c in enumerate(cartesians(l - 1))}
    out = np.zeros((len(cartesians(l)), len(to)))
    for n, c in enumerate(cartesians(l)):
        if c[k]:
            r = list(c)
            r[k] -= 1
            out[n, to[tuple(r)]] = -float(c[k])
    return out


def _one(l: int, k: int) -> dict[int, np.ndarray]:  # noqa: E741
    """One derivative along ``k`` of the Cartesians of ``l``:
    ``{shift: matrix}``, the raise weight ``2a`` apart (it is per lane)."""
    out = {+1: _up(l, k)}
    if l:
        out[-1] = FORM["lower"] * _down_unit(l, k)
    return out


def _same(l: int, k: int, m: int) -> dict[int, np.ndarray]:  # noqa: E741
    """``d2/dA_k dA_m`` on one centre as ``{shift: matrix}``, the two
    paths to shift 0 summed (their weight is the same, ``2a``)."""
    out: dict[int, np.ndarray] = {}
    for s1, m1 in _one(l, k).items():
        for s2, m2 in _one(l + s1, m).items():
            out[s1 + s2] = out.get(s1 + s2, 0) + m1 @ m2
    return out


def terms(ls: tuple[int, ...], p: int, q: int, k: int, m: int):
    """``d2/dX_{p,k} dX_{q,m}`` of the class ``ls`` as
    ``[(shifts, powers, {position: matrix})]``: per term the shifted class
    (``ls + shifts``), the power of ``2a`` on each position, and the matrix
    taking that class's Cartesians back to ``ls``'s on each moved one."""
    out = []
    n = len(ls)
    if p == q:
        for s, mat in _same(ls[p], k, m).items():
            sh, w = [0] * n, [0] * n
            sh[p], w[p] = s, (s + 2) // 2
            out.append((tuple(sh), tuple(w), {p: mat}))
        return out
    for sp, mp in _one(ls[p], k).items():
        for sq, mq in _one(ls[q], m).items():
            sh, w = [0] * n, [0] * n
            sh[p], sh[q] = sp, sq
            w[p], w[q] = (sp + 1) // 2, (sq + 1) // 2
            out.append((tuple(sh), tuple(w), {p: mp, q: mq}))
    return out


def _apply(blk: np.ndarray, mats: dict[int, np.ndarray]) -> np.ndarray:
    for pos, mat in mats.items():
        blk = np.moveaxis(np.tensordot(mat, blk, axes=([1], [pos])), 0, pos)
    return blk


def quartet2(sa: Shell, sb: Shell, sc: Shell, sd: Shell) -> np.ndarray:
    """``d2(ab|cd)`` on each pair of the four centres,
    ``[4, 3, 4, 3, na, nb, nc, nd]`` (position, axis, position, axis): the
    ten ``p <= q`` from at most 36 weighted shifted blocks, each evaluated
    once (module docstring), the other six their transposes."""
    q = (sa, sb, sc, sd)
    ls = tuple(x.l for x in q)
    s = _lane_scalars(q)
    two_a = [FORM["raise"] * e for e in s["_exp"]]
    blocks: dict[tuple, np.ndarray] = {}

    def get(sh, w):
        if (sh, w) not in blocks:
            weight = np.ones_like(two_a[0])
            for pos, n in enumerate(w):
                weight = weight * two_a[pos] ** n
            cls = tuple(x + d for x, d in zip(ls, sh, strict=True))
            blocks[(sh, w)] = block(q, cls, s, weight)
        return blocks[(sh, w)]

    shape = tuple(len(cartesians(x)) for x in ls)
    out = np.zeros((4, 3, 4, 3) + shape)
    for p in range(4):
        for pq in range(p, 4):
            for k in range(3):
                for m in range(3):
                    acc = np.zeros(shape)
                    for sh, w, mats in terms(ls, p, pq, k, m):
                        acc += _apply(get(sh, w), mats)
                    out[p, k, pq, m] = acc
                    if pq != p:
                        out[pq, m, p, k] = acc
    return out


def _by_atom(der: np.ndarray, at: Sequence[int], natm: int) -> np.ndarray:
    """``[4, 3, 4, 3, ...]`` per position pair -> ``[natm, 3, natm, 3, ...]``
    per pair of nuclei."""
    out = np.zeros((natm, 3, natm, 3) + der.shape[4:])
    for p in range(4):
        for q in range(4):
            out[at[p], :, at[q], :] += der[p, :, q, :]
    return out


def _axes(perm):
    return (0, 1, 2, 3) + tuple(4 + p for p in perm)


def two_electron2(shells: Sequence[Shell], atoms: Sequence[int], natm: int
                  ) -> np.ndarray:
    """``d2(ab|cd)/dR dR`` ``[natm, 3, natm, 3] + [nao]*4``, dense: one
    evaluation per unique shell quartet, assigned over its orbit."""
    off = offsets(shells)
    nao = off[-1]
    out = np.zeros((natm, 3, natm, 3) + (nao,) * 4)
    for idx in _unique(len(shells)):
        g = _by_atom(quartet2(*(shells[i] for i in idx)),
                     [atoms[i] for i in idx], natm)
        for perm in ORBIT:
            sl = tuple(slice(off[idx[p]], off[idx[p] + 1]) for p in perm)
            out[(slice(None),) * 4 + sl] = g.transpose(_axes(perm))
    return out


def two_electron_hessian(shells: Sequence[Shell], atoms: Sequence[int],
                         natm: int, D: np.ndarray) -> dict[str, np.ndarray]:
    """``{"J", "K", "G"}``, each ``[natm, 3, natm, 3]``: ``1/2 SUM D D d2J``,
    ``-1/4 SUM D D d2K`` and their sum, by the fused sweep (module
    docstring): every distinct ordered member of each unique quartet's orbit
    contracted with its density blocks as it forms, nothing of size
    ``nao^4`` kept."""
    off = offsets(shells)
    hJ = np.zeros((natm, 3, natm, 3))
    hK = np.zeros((natm, 3, natm, 3))
    for idx in _unique(len(shells)):
        g = _by_atom(quartet2(*(shells[i] for i in idx)),
                     [atoms[i] for i in idx], natm)
        seen = set()
        for perm in ORBIT:
            mem = tuple(idx[p] for p in perm)
            if mem in seen:
                continue
            seen.add(mem)
            gm = g.transpose(_axes(perm))
            a, b, c, d = (slice(off[i], off[i + 1]) for i in mem)
            hJ += FORM["J"] * np.einsum("xkylabcd,ab,cd->xkyl", gm,
                                        D[a, b], D[c, d])
            hK += FORM["K"] * np.einsum("xkylabcd,ac,bd->xkyl", gm,
                                        D[a, c], D[b, d])
    return {"J": hJ, "K": hK, "G": hJ + hK}


def coulomb_deriv2(d2eri: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``d2J[A, k, B, l, m, n] = SUM_ls d2(mn|ls) D_ls`` at fixed ``D``."""
    return np.tensordot(d2eri, D, axes=([6, 7], [0, 1]))


def exchange_deriv2(d2eri: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``d2K[A, k, B, l, m, n] = SUM_ls d2(ml|sn) D_ls`` at fixed ``D``."""
    return np.einsum("AkBlmusn,us->AkBlmn", d2eri, D, optimize=True)


def fock_deriv2(d2eri: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``d2G = d2J - d2K/2`` at fixed ``D``."""
    return coulomb_deriv2(d2eri, D) - 0.5 * exchange_deriv2(d2eri, D)
