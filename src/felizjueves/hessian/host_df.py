"""The host reference of `kernel_d2metric`: the contracted nuclear Hessian
of the density-fitting Coulomb metric `(P|Q)`.

    metric_hessian(aux, W, nuclei, *, pq, pairs=, atoms=, absolute=) -> H

`H` is `n_atom x 3 x n_atom x 3`, `H[A][k][B][l] = SUM_PQ W_PQ d2(P|Q) /
dR_{A,k} dR_{B,l}` (the package docstring) -- over every aux shell pair
`i <= j`, or over `pairs=` only -- every centre of every pair on atom `A`
(and `B`) summed. `W` is any `n_aux x n_aux` matrix, weighted per
Cartesian pair by `gradient.host.pair_weights` (`W[p, q] + W[q, p]` off a
diagonal shell pair, `W[p, q]` on it), exact for any `W` against the
symmetric `(P|Q)`. A DF-J energy Hessian's metric term passes `W = -1/2 d
d^T` (`gradient.host_df`'s module docstring), skeleton only: `d` is not
differentiated (its response is the caller's).

THE ROUTE IS NOT THE KERNEL'S. The kernel moves the Hermite Gaussians
(`[r]^(0) -> [r + 1_k + 1_l]^(0)`, `kernel_d2metric`'s docstring). This
reference is `hessian.host_1e`'s: each aux FUNCTION is differentiated as
a list of `(coefficient, Cartesian)` -- `d_centre(a, v, k) = [(2a, v +
1_k), (-v_k, v - 1_k)]`, applied twice for `d2` -- and every term is the
ordinary `(u|v)` of the moved Cartesians on the single-center `E`
(`e_axis(i, 0, alpha, 0, 0)`, two indices above the class):

    H[A, k, A, l] += s w SUM c (u|b)          over d2_A(a; k, l)
    H[B, k, B, l] += s w SUM c (a|u)          over d2_B(b; k, l)
    H[A, k, B, l] += s w SUM c c' (u|u')      over d_A(a; k) x d_B(b; l)
    H[B, l, A, k] += the same                 (one value, both places)

per aux record pair (`s = s_P s_Q`, the metric's record weights) and
Cartesian pair `(a, b)` of weight `w`. The two routes are the same number
by `SUM_t (2 alpha E^{a+1}_t - a E^{a-1}_t) Lambda_t = SUM_t E^a_t
Lambda_{t+1_k}` applied twice, so agreement checks both.

`absolute=True` is the same sum with every factor in `|.|` (coefficients,
`E`, `[t|tau]`, `w`, `s`): a round-off scale for the kernel's interpreted
test, the size of the terms before they cancel.

NO TRANSLATIONAL INVARIANCE IS USED: both centres are differentiated on
their own, so `SUM_A H[A, k, B, l] = 0` and `H = H^T` are checks
(test/test_hessian_host_df.py), as is a central finite difference of
`gradient.host_df.metric_gradient`.

`[t|tau]` is INJECTED (`pq=`, `reference.pq.pq` in the tests), as in
`gradient.host_df`. Pure Python floats.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from felizjueves.ao.shell import Basis, cartesians
from felizjueves.eri.density_fit.metric import aux_records
from felizjueves.gradient.host import pair_weights, shell_atoms
from felizjueves.hessian.host_1e import Hess, d2_centre, d_centre, zeros
from felizjueves.mmd.hermite import e_axis, hermite_order
from felizjueves.one_electron.potential import Nucleus

Ang = tuple[int, int, int]
Vec = tuple[float, float, float]
PQ = Callable[[Ang, Ang, Vec, float], float]
Terms = list[tuple[float, Ang]]

_KL = tuple((k, l) for k in range(3) for l in range(3))  # noqa: E741


def _at(e: Sequence[float], t: int) -> float:
    return e[t] if t < len(e) else 0.0


def _rows(aux: Basis, nuclei: Sequence[Nucleus],
          atoms: Sequence[int] | None) -> tuple[int, ...]:
    if atoms is None:
        return shell_atoms(aux, nuclei)
    atoms = tuple(int(a) for a in atoms)
    if len(atoms) != len(aux.shells) or min(atoms, default=0) < 0:
        raise ValueError("atoms= needs one non-negative row per shell")
    return atoms


class _Record:
    """One aux record pair's `(u|v)` of any two Cartesians at most two
    above the class: `SUM_t SUM_tau E^u_t E^v_tau [t|tau]`, cached."""

    def __init__(self, ap: float, aq: float, lp: int, lq: int, R: Vec,
                 pq: PQ, f) -> None:
        self.eP = [e_axis(x, 0, ap, 0.0, 0.0) for x in range(lp + 3)]
        self.eQ = [e_axis(x, 0, aq, 0.0, 0.0) for x in range(lq + 3)]
        theta = math.sqrt(ap * aq / (ap + aq))
        self.m = {(t, tau): f(pq(t, tau, R, theta))
                  for t in hermite_order(lp + 2)
                  for tau in hermite_order(lq + 2)}
        self.f = f
        self.cache: dict[tuple[Ang, Ang], float] = {}

    def __call__(self, u: Ang, v: Ang) -> float:
        key = (u, v)
        if key not in self.cache:
            f, eP, eQ = self.f, self.eP, self.eQ
            acc = 0.0
            for t in hermite_order(sum(u)):
                et = (_at(eP[u[0]], t[0]) * _at(eP[u[1]], t[1])
                      * _at(eP[u[2]], t[2]))
                if not et:
                    continue
                for tau in hermite_order(sum(v)):
                    eu = (_at(eQ[v[0]], tau[0]) * _at(eQ[v[1]], tau[1])
                          * _at(eQ[v[2]], tau[2]))
                    if eu:
                        acc += f(et) * f(eu) * self.m[(t, tau)]
            self.cache[key] = acc
        return self.cache[key]

    def terms(self, ta: Terms, tb: Terms) -> float:
        f = self.f
        return sum(f(c1) * f(c2) * self(u, v) for c1, u in ta for c2, v in tb)


def metric_hessian(aux: Basis, W: Sequence[Sequence[float]],
                   nuclei: Sequence[Nucleus], *, pq: PQ,
                   pairs: Sequence[tuple[int, int]] | None = None,
                   atoms: Sequence[int] | None = None,
                   absolute: bool = False) -> Hess:
    """`H[A][k][B][l] = SUM_PQ W_PQ d2(P|Q) / dR_{A,k} dR_{B,l}` (module
    docstring)."""
    f = abs if absolute else float
    sh = aux.shells
    rows = _rows(aux, nuclei, atoms)
    H = zeros(max(len(nuclei), max(rows, default=-1) + 1))
    n = aux.n_ao
    Wr = [[float(W[i][j]) for j in range(n)] for i in range(n)]
    if pairs is None:
        pairs = [(i, j) for i in range(len(sh)) for j in range(i, len(sh))]
    for i, j in pairs:
        if not 0 <= i <= j < len(sh):
            raise ValueError(f"aux shell pair ({i}, {j}) is not i <= j")
        sP, sQ = sh[i], sh[j]
        A, B = rows[i], rows[j]
        w = pair_weights(Wr, i, j, aux)
        cart = [(a, b) for a in cartesians(sP.l) for b in cartesians(sQ.l)]
        for p in aux_records(sP):
            for q in aux_records(sQ):
                R = (p.A[0] - q.A[0], p.A[1] - q.A[1], p.A[2] - q.A[2])
                X = _Record(p.alpha, q.alpha, sP.l, sQ.l, R, pq, f)
                s = f(p.s) * f(q.s)
                for wr, (a, b) in zip(w, cart, strict=True):
                    if not wr:
                        continue
                    sw = s * f(wr)
                    for k, l in _KL:  # noqa: E741
                        aa = X.terms(d2_centre(p.alpha, a, k, l), [(1.0, b)])
                        bb = X.terms([(1.0, a)], d2_centre(q.alpha, b, k, l))
                        ab = X.terms(d_centre(p.alpha, a, k),
                                     d_centre(q.alpha, b, l))
                        H[A][k][A][l] += sw * aa
                        H[B][k][B][l] += sw * bb
                        H[A][k][B][l] += sw * ab
                        H[B][l][A][k] += sw * ab
    return H
