"""The density-fitting tensors ``(P|Q)`` and ``(mn|P)`` by the HGP engine.

    metric(aux, *, boys)                    -> DF((P|Q), its |.| twin), [naux, naux]
    three_center(basis, aux, *, boys)       -> DF((mn|P), its |.| twin),
                                               [nao, nao, naux]
    metric_block(sP, sQ, *, boys)           -> one ordered aux shell pair
    three_center_block(sA, sB, sP, *, boys) -> one AO shell pair x one aux shell

s and p shells only, on both bases (``LMAX = 1``; above it ``ValueError``).

THE ROUTE. Nothing about the recurrence is restated here: every value comes
out of ``felizjueves.eri.hgp_drive.eri_block_and_scale``, i.e. ``hgp.vrr``'s
VRR (``[Fock]`` eqs 25-26) per primitive quartet, summed, then ``hgp.hrr``'s
HRR (eq 28) once on contracted values, both run as IR in ``ir.interp``. A
two- or three-center integral is a four-center one with a unit s partner:

    (P|Q)  = (P u_P | Q u_Q),    (mn|P) = (m n | P u_P),

``u_X`` an s Gaussian of exponent 0 and coefficient 1 on ``X``'s centre
(``experimental/mmd/gradient/df.py``'s ``_unit``, the same device). In
``hgp_drive.primitive_scalars`` that partner gives ``p = alpha``, ``P = A``
exactly (``(alpha A + 0 B) / alpha``), ``K_AB = exp(0) = 1``, and a pair
factor ``c (pi/alpha)^{3/2}`` -- ``eri.density_fit.metric.aux_dress``'s
``s``, with no normalisation added (it is in ``c``). With ``lb = 0`` (or
``ld = 0``) the HRR on that side is the identity, and its ``AB`` (``CD``) is
the zero vector, so nothing divides by the zero exponent anywhere.

``ao.shell.Shell`` refuses exponent 0 (``eri/density_fit/eri3.py``: "the
internal twin is not built"), so the partner is ``Unit``, a duck-typed shell
carrying only what ``hgp_drive`` reads (``l``, ``centre``, ``exponents``,
``coefficients``, ``n_cart``). ``src/`` is not touched.

THE LAYOUT, ``eri.density_fit``'s: ``(P|Q)`` rows and columns in the aux
basis's AO order; ``(mn|P)`` as ``[m][n][P]``, aux fastest. ``metric`` sweeps
aux shell pairs ``i <= j`` and mirrors; ``three_center`` sweeps AO shell
pairs ``i <= j`` against every aux shell and mirrors in ``(m, n)``. Both
tensors are therefore exactly symmetric where they should be.

THE SCALE is ``hgp_drive``'s absolute-value twin (every coefficient and
scalar replaced by its ``|.|``), placed alike: a bound on what the sum's
rounding can reach, so a check against it is about rounding, not about how
much the sum happened to cancel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from felizjueves.ao.shell import Shell
from felizjueves.eri.hgp_drive import eri_block_and_scale

__all__ = ["DF", "LMAX", "Unit", "metric", "metric_block", "offsets",
           "three_center", "three_center_block"]

LMAX = 1  # s and p, on both bases
Boys = Callable[[int, float], float]


@dataclass(frozen=True)
class Unit:
    """The exponent-0, coefficient-1 s partner on ``centre`` (module
    docstring). Only the attributes ``hgp_drive`` reads."""

    centre: tuple[float, float, float]
    l: int = 0  # noqa: E741
    exponents: tuple[float, ...] = (0.0,)
    coefficients: tuple[float, ...] = (1.0,)
    n_cart: int = 1


@dataclass(frozen=True)
class DF:
    """A tensor and its absolute-value twin, the same shape."""

    value: np.ndarray
    scale: np.ndarray


def _check(shells: Sequence[Shell], what: str) -> None:
    for i, s in enumerate(shells):
        if s.l > LMAX:
            raise ValueError(f"{what} shell {i} has l = {s.l}; this spike "
                             f"stops at p (LMAX = {LMAX})")


def offsets(shells: Sequence[Shell]) -> tuple[int, ...]:
    """The AO offset of every shell, and the total last."""
    out = [0]
    for s in shells:
        out.append(out[-1] + s.n_cart)
    return tuple(out)


def metric_block(sP: Shell, sQ: Shell, *, boys: Boys
                 ) -> tuple[np.ndarray, np.ndarray]:
    """``(P|Q)`` of one ordered aux shell pair, ``[nP, nQ]``, and its twin:
    the HGP quartet ``(P u|Q u)``."""
    _check((sP, sQ), "aux")
    blk, sc = eri_block_and_scale((sP, Unit(sP.centre)),
                                  (sQ, Unit(sQ.centre)), boys=boys)
    return np.array(blk), np.array(sc)


def three_center_block(sA: Shell, sB: Shell, sP: Shell, *, boys: Boys
                       ) -> tuple[np.ndarray, np.ndarray]:
    """``(ab|P)`` of one ordered AO shell pair and one aux shell,
    ``[na, nb, nP]``, and its twin: the HGP quartet ``(a b|P u)``."""
    _check((sA, sB), "AO")
    _check((sP,), "aux")
    blk, sc = eri_block_and_scale((sA, sB), (sP, Unit(sP.centre)), boys=boys)
    shape = (sA.n_cart, sB.n_cart, sP.n_cart)
    return np.array(blk).reshape(shape), np.array(sc).reshape(shape)


def metric(aux: Sequence[Shell], *, boys: Boys) -> DF:
    """``(P|Q)`` over every aux function, ``[naux, naux]``: one HGP block
    per aux shell pair ``i <= j``, mirrored."""
    _check(aux, "aux")
    off = offsets(aux)
    n = off[-1]
    M, S = np.zeros((n, n)), np.zeros((n, n))
    for i, si in enumerate(aux):
        for j in range(i, len(aux)):
            blk, sc = metric_block(si, aux[j], boys=boys)
            a, b = slice(off[i], off[i + 1]), slice(off[j], off[j + 1])
            M[a, b], S[a, b] = blk, sc
            M[b, a], S[b, a] = blk.T, sc.T
    return DF(M, S)


def three_center(basis: Sequence[Shell], aux: Sequence[Shell], *,
                 boys: Boys) -> DF:
    """``(mn|P)`` as ``[nao, nao, naux]``: one HGP block per AO shell pair
    ``i <= j`` and aux shell, mirrored in ``(m, n)``."""
    _check(basis, "AO")
    _check(aux, "aux")
    off, xoff = offsets(basis), offsets(aux)
    nao, naux = off[-1], xoff[-1]
    T, S = np.zeros((nao, nao, naux)), np.zeros((nao, nao, naux))
    for i, si in enumerate(basis):
        for j in range(i, len(basis)):
            a, b = slice(off[i], off[i + 1]), slice(off[j], off[j + 1])
            for k, sk in enumerate(aux):
                blk, sc = three_center_block(si, basis[j], sk, boys=boys)
                c = slice(xoff[k], xoff[k + 1])
                T[a, b, c], S[a, b, c] = blk, sc
                T[b, a, c] = blk.transpose(1, 0, 2)
                S[b, a, c] = sc.transpose(1, 0, 2)
    return DF(T, S)
