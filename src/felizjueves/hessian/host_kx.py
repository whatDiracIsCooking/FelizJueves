"""The host reference of the exchange derivative matrix `K^X` (#689): the
exchange part of the derivative Fock matrix the RHF Hessian's CPHF
right-hand side needs, as an AO matrix per perturbation.

    exchange_derivative_ip1(basis, D, nuclei, ...)        -> K1[3][n][n]
    exchange_derivative_contracted(basis, D, nuclei, atom, ...)
                                                          -> KC[3][n][n]
    exchange_derivative(basis, D, nuclei, atom, ...)      -> K^X[3][n][n]
    kx_records(basis, nuclei, am, *, atom=None)           -> kernel_kx's arrays

THE MATRIX. For atom `X`, axis `k`, with K's pairing (bra `(a, c)`, ket
`(b, d)`, `K_ab = SUM_cd D_cd (ac|bd)`),

    K^X_ab = SUM_cd D_cd d(ac|bd) / dR_{X,k}

Four centres move. `D_cd` straddles bra and ket, so there is no
derivative density to reuse (`hessian.kernel_hkx`'s reason) and two
FORMS are built, each the derivative on ONE bra function:

- ip1, the derivative on the matrix index `a`, all atoms at once:

      K1[k, a, b] = SUM_cd D_cd (dA_k a c|b d)          A the centre of a

- contracted, the derivative on the contracted index `c`, per atom:

      KC_X[k, a, b] = SUM_{c on X} SUM_d D_cd (a dC_k c|b d)

The ket's two centres are the bra's by symmetry: `(ac|b d_k d) = (b d_k
d|a c)`, so with a SYMMETRIC `D` the `d` term is `KC_X` transposed and
the `b` term is `K1`'s rows on `X` transposed. Hence

    K^X = P_X K1 + (P_X K1)^T + KC_X + KC_X^T

`P_X` keeping the rows of `a` on `X`. This is PySCF's `rhf_hess.make_h1`
exchange part times `-2` (its `vk1` is `KC_X^T`, its `vk2` restricted to
the rows on `X` is `P_X K1`, `h1 -= 1/2 (vk + vk^T)`), and `K1` is
`grad.rhf.get_jk`'s `vk` (both `int2e_ip1`, the electron-coordinate
derivative, which is minus the centre's; the signs are taken here).

THE SWEEP is `gradient.host.exchange_gradient`'s: every ORDERED bra shell
pair `(sa, sc)` against every ordered ket pair `(sb, sd)` (or
`quartets=`), per (bra record, ket record) `[r]^(0)` at `P - Q`, `|r| <=
L + 1`, per `(b,d)` row `I[p] = SUM_q (-1)^|q| [p+q]^(0) s_bd E^bd_q`
over `|p| <= lp + 1`, then per `(a,c)` row and axis `SUM_p (dY_k
E^{ac})_p I[p]` times `D_cd s_ac` into `X[k][a][b]` (`Y = A` or `C`). No
pair weight. Cartesian, row-major. `absolute=True` takes every factor in
`|.|`, both terms of the derivative rule added: the round-off scale.

`[t]^(0)` is INJECTED (`one_centre=`); tests pass
`reference.pq.one_centre`. Pure Python floats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from felizjueves.ao.density import checked_density
from felizjueves.ao.pairs import OneCentre
from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.gradient import host as ghost
from felizjueves.mmd.hermite import hermite_order
from felizjueves.one_electron.potential import Nucleus

Matrix = Sequence[Sequence[float]]
Deriv = list[list[list[float]]]  # X[k][a][b]
Quartets = Sequence[tuple[int, int, int, int]] | None

FORMS = ("ip1", "contracted")


def zeros(n: int) -> Deriv:
    return [[[0.0] * n for _ in range(n)] for _ in range(3)]


def _quartet(X: Deriv, side: int, sa: Shell, sc: Shell, sb: Shell,
             sd: Shell, oa: int, ob: int, Dcd, *, one_centre: OneCentre,
             absolute: bool) -> None:
    """One ordered shell quartet's `SUM_cd D_cd dY (ac|bd)` into `X`, `Y`
    the bra's `side` (0: `A`, the ip1 form; 1: `C`, the contracted one).
    `Dcd[k][l]` is the `(c, d)` density block."""
    f = abs if absolute else float
    lp, lq = sa.l + sc.l, sb.l + sd.l
    ps, qs = hermite_order(lp + 1), hermite_order(lq)
    rows_ac = [(a, c) for a in cartesians(sa.l) for c in cartesians(sc.l)]
    rows_bd = [(b, d) for b in cartesians(sb.l) for d in cartesians(sd.l)]
    nc, nd = sc.n_cart, sd.n_cart
    kets = []
    for eta, Q, s_bd, _, _ in ghost._records(sb, sd):
        ax = ghost._Axes(eta, Q, sb.centre, sd.centre, sb.l, sd.l)
        kets.append((eta, Q, [
            [s_bd * v for v in ghost._hermite([ax.e(x, b[x], e[x])
                                               for x in range(3)], qs)]
            for b, e in rows_bd]))
    for zeta, P, s_ac, alpha, gamma in ghost._records(sa, sc):
        ax = ghost._Axes(zeta, P, sa.centre, sc.centre, sa.l + 1, sc.l + 1)
        two = 2.0 * (alpha, gamma)[side]
        dE = []  # per (a,c) row, per axis: over `ps`
        for a, c in rows_ac:
            e = [ax.e(x, a[x], c[x]) for x in range(3)]
            per_k = []
            for k in range(3):
                n = (a, c)[side][k]
                hi = [a[k], c[k]]
                hi[side] += 1
                if absolute:
                    dk = [two * abs(x) for x in ax.e(k, *hi)]
                    fk = [[abs(x) for x in v] for v in e]
                else:
                    dk = [two * x for x in ax.e(k, *hi)]
                    fk = list(e)
                if n:
                    lo = [a[k], c[k]]
                    lo[side] -= 1
                    for t, x in enumerate(ax.e(k, *lo)):
                        dk[t] += n * abs(x) if absolute else -n * x
                fk[k] = dk
                per_k.append(ghost._hermite(fk, ps))
            dE.append(per_k)
        for eta, Q, ebd in kets:
            theta = math.sqrt(zeta * eta / (zeta + eta))
            PQv = (P[0] - Q[0], P[1] - Q[1], P[2] - Q[2])
            R = {r: f(one_centre(r, 0, PQv, theta))
                 for r in hermite_order(lp + lq + 1)}
            for rbd in range(len(rows_bd)):
                ib, id_ = divmod(rbd, nd)
                col = []
                for p in ps:
                    acc = 0.0
                    for q, e in zip(qs, ebd[rbd], strict=True):
                        x = R[(p[0] + q[0], p[1] + q[1], p[2] + q[2])] * f(e)
                        acc += x if absolute or sum(q) % 2 == 0 else -x
                    col.append(acc)
                for rac in range(len(rows_ac)):
                    ia, ic = divmod(rac, nc)
                    w = f(s_ac) * f(Dcd[ic][id_])
                    for k in range(3):
                        v = 0.0
                        for x, y in zip(dE[rac][k], col, strict=True):
                            v += x * y
                        X[k][oa + ia][ob + ib] += w * v


def _sweep(basis: Basis, D: Matrix, nuclei: Sequence[Nucleus], side: int,
           keep, *, one_centre: OneCentre, quartets: Quartets,
           absolute: bool) -> Deriv:
    ghost._check_nuclei(nuclei)
    rows = checked_density(basis, D)
    sh, off = basis.shells, basis.offsets
    n = len(sh)
    if quartets is None:
        quartets = [(i, k, j, m) for i in range(n) for k in range(n)
                    for j in range(n) for m in range(n)]
    X = zeros(basis.n_ao)
    for i, k, j, m in quartets:
        if not all(0 <= x < n for x in (i, k, j, m)):
            raise ValueError(f"shell quartet {(i, k, j, m)} is out of range")
        if not keep(i, k):
            continue
        Dcd = [[rows[off[k] + a][off[m] + b] for b in range(sh[m].n_cart)]
               for a in range(sh[k].n_cart)]
        _quartet(X, side, sh[i], sh[k], sh[j], sh[m], off[i], off[j], Dcd,
                 one_centre=one_centre, absolute=absolute)
    return X


def exchange_derivative_ip1(basis: Basis, D: Matrix,
                            nuclei: Sequence[Nucleus], *,
                            one_centre: OneCentre, quartets: Quartets = None,
                            absolute: bool = False) -> Deriv:
    """`K1[k][a][b] = SUM_cd D_cd (dA_k a c|b d)` (module docstring), over
    every ordered shell quartet or `quartets=` `(i, k, j, l)` (bra `(i,
    k)`, ket `(j, l)`). `D` must be symmetric."""
    return _sweep(basis, D, nuclei, 0, lambda i, k: True,
                  one_centre=one_centre, quartets=quartets,
                  absolute=absolute)


def exchange_derivative_contracted(basis: Basis, D: Matrix,
                                   nuclei: Sequence[Nucleus], atom: int, *,
                                   one_centre: OneCentre,
                                   quartets: Quartets = None,
                                   absolute: bool = False) -> Deriv:
    """`KC_X[k][a][b] = SUM_{c on X} SUM_d D_cd (a dC_k c|b d)`, `X =
    atom` (module docstring): the quartets whose bra `c` shell sits on
    `atom` (`gradient.host.shell_atoms`)."""
    at = ghost.shell_atoms(basis, nuclei)
    if not 0 <= atom < len(nuclei):
        raise ValueError(f"atom {atom} is out of range")
    return _sweep(basis, D, nuclei, 1, lambda i, k: at[k] == atom,
                  one_centre=one_centre, quartets=quartets,
                  absolute=absolute)


def exchange_derivative(basis: Basis, D: Matrix, nuclei: Sequence[Nucleus],
                        atom: int, *, one_centre: OneCentre,
                        k1: Deriv | None = None) -> Deriv:
    """The whole `K^X = P_X K1 + (P_X K1)^T + KC_X + KC_X^T` of atom `X`
    (module docstring). `k1=` reuses an `exchange_derivative_ip1` already
    computed (it is every atom's)."""
    if k1 is None:
        k1 = exchange_derivative_ip1(basis, D, nuclei, one_centre=one_centre)
    kc = exchange_derivative_contracted(basis, D, nuclei, atom,
                                        one_centre=one_centre)
    at = ghost.shell_atoms(basis, nuclei)
    on = [False] * basis.n_ao
    for s, o in enumerate(basis.offsets):
        for r in range(basis.shells[s].n_cart):
            on[o + r] = at[s] == atom
    n = basis.n_ao
    out = zeros(n)
    for k in range(3):
        for a in range(n):
            for b in range(n):
                v = kc[k][a][b] + kc[k][b][a]
                if on[a]:
                    v += k1[k][a][b]
                if on[b]:
                    v += k1[k][b][a]
                out[k][a][b] = v
    return out


def kx_records(basis: Basis, nuclei: Sequence[Nucleus],
               am: tuple[int, int, int, int], *,
               atom: int | None = None) -> tuple[dict, dict]:
    """`kernel_kx`'s per-record arrays of K class `am`: `gradient.host.
    exchange_records`'s, every bra pair of `exchange_pairs(la, lc)` (or,
    with `atom=`, only those whose `c` shell sits on `atom`: one `kxc_`
    launch) against every ket pair. The bra's atom columns are dropped
    (the kernel writes a matrix)."""
    bra, ket = ghost.exchange_records(basis, nuclei, am)
    if atom is not None:
        keep = [i for i, a in enumerate(bra["atom_c"]) if a == atom]
        bra = {k: [v[i] for i in keep] for k, v in bra.items()}
    for k in ("atom_a", "atom_c"):
        del bra[k]
    return bra, ket
