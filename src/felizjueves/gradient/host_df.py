"""The host references of the density-fitting d1 kernels: the contracted
nuclear gradients of the Coulomb metric `(P|Q)` and the three-center
integral `(mn|P)`.

    metric_gradient(aux, W, nuclei, *, pq)         -> G  SUM_PQ W_PQ d(P|Q)/dR
    metric_derivative(aux, *, pq)                  -> M1 d(P|Q)/dA_P, uncontracted
    eri3_gradient(basis, aux, Gam, nuclei, *, pq)  -> G  SUM_mnP Gam_mnP d(mn|P)/dR
    metric_records(cls, aux, nuclei)               -> gmetric's extra arrays
    eri3_records(cls, basis, aux, nuclei)          -> geri3's extra arrays
    eri3_ip1(basis, aux, *, pq)                    -> X  (nabla m n|P), uncontracted

`G` is `n_atom x 3`, as in `gradient.host`: the derivative of every function
on atom `A`, summed. A DF-J energy gradient is the two together,

    E_J = SUM_mnP D_mn (mn|P) d_P - 1/2 SUM_PQ d_P (P|Q) d_Q,
    (P|Q) d = SUM_mn (P|mn) D_mn,
    dE_J/dR = eri3_gradient(Gam = D (x) d) + metric_gradient(W = -1/2 d d^T)

(stationary in `d`, so `d` is not differentiated);
test/test_gradient_host_df.py holds that sum to a finite difference of
`E_J` itself. `W` need not be symmetric and `Gam` need not be symmetric in
`(m, n)`: the weights are the one-electron rule of
`gradient.host.pair_weights` -- `W[p, q] + W[q, p]` off a diagonal shell
pair, `W[p, q]` on it; for `(mn|P)` the same in `(m, n)` at each `P` --
exact for any matrix against an integral symmetric in the pair.

THE ROUTE IS THE CARTESIAN RULE ON EVERY CENTRE, the `E`-table route of
`gradient.host`: `d/dA_k` of a Cartesian Gaussian is `2 alpha G(a + 1_k) -
a_k G(a - 1_k)`, so

- an aux function (single centre, `X_PA = 0`): `dE^a_t = 2 alpha E^{a+1}_t
  - a E^{a-1}_t` on the moved axis, the single-center `E` (`e_axis(i, 0,
  alpha, 0, 0)`) one index higher -- for both sides of `(P|Q)` and the aux
  side of `(mn|P)`;
- an AO pair: `dA E^{ij}_t = 2 alpha E^{i+1,j}_t - i E^{i-1,j}_t` and `dB`
  likewise (`gradient.host`'s rule, restated here on `mmd.hermite.e_axis`).

The KERNELS take a different route for every single-centre side: the
single-center `E` does not depend on the centre, so `d/dA_k` of an aux
function is the Hermite shift `[r]^(0) -> [r + 1_k]^(0)`
(`gradient.kernel_gmetric`, `gradient.kernel_geri3`). The two routes are
the same number by `SUM_t (2 alpha E^{a+1}_t - a E^{a-1}_t) Lambda_t =
SUM_t E^a_t Lambda_{t+1_k}`, so agreement checks both, not a restatement.

NO TRANSLATIONAL INVARIANCE IS USED: every centre's derivative is computed
on its own, so `SUM_A G[A] = 0` is a check.

THE SWEEP is basis order, unstaged: aux shell pairs `i <= j` row-major for
`(P|Q)`; AO shell pairs `i <= j` row-major, aux shells inner, for `(mn|P)`
(`eri.density_fit.eri3._place`'s order). `pairs=` / `cells=` restrict it.
`atoms=` / `aux_atoms=` give each shell a row of `G` of the caller's
choosing (a test passes every shell a row of its own to see the size of the
terms before they cancel on one atom); by default each shell sits on the
nucleus whose centre it equals EXACTLY (`gradient.host.shell_atoms`).

`[t|tau]` is INJECTED (`pq=`, `reference.pq.pq` in the tests), as in
`eri.density_fit`. Pure Python floats: `src/felizjueves` has no runtime
dependencies.

THE STAGED TERMS (#620). The `Gam` and `W` of a DF-RHF two-electron
gradient, staged here so a consumer that builds its own (FreeBreakfast,
with dense GPU linear algebra) has something to hold them to:

    df_j_stage(T, M, D)              -> DfStage  E_J, Gam^J, W^J
    df_k_stage(T, M, C, *, occ)      -> DfStage  E_K, Gam^K, W^K
    df_term(basis, aux, stage, nuclei, *, pq)    -> DfTerm  the two contractions
    df_rhf_gradient(basis, aux, C, nuclei, *, pq, occ)  -> DfRhfGradient

`T = (mn|P)` (`eri3_tensor`), `M = (P|Q)` (`metric_matrix`), `D = occ C
C^T` with `C` the occupied coefficients `[n_ao][n_occ]` (`occ = 2`: the
closed-shell total density). With `(ml|ns) ~ SUM_PQ (ml|P) M^-1_PQ (Q|ns)`,

    E_J = 1/2 SUM D_mn D_ls (mn|ls),   b_P = SUM_mn D_mn (mn|P),  M d = b,
          Gam^J_mnP = D_mn d_P,        W^J_PQ = -1/2 d_P d_Q;
    E_K = 1/2 SUM D_mn D_ls (ml|ns),   X_ijP = SUM_ml C_mi C_lj (ml|P),
                                       M d_ij = X_ij,
          Gam^K_mlP = occ^2 SUM_ij C_mi C_lj d_ijP,
          W^K_PQ    = -1/2 occ^2 SUM_ij d_ijP d_ijQ,

each `dE/dR = eri3_gradient(Gam) + metric_gradient(W)`, `d` not
differentiated (the fitted energy is stationary in it). `E_K` is
`gradient.md`'s `1/2 SUM D_ab D_cd (ac|bd)`, so the closed-shell
two-electron energy is `E_J - 1/2 E_K` and its gradient `G_J - 1/2 G_K`;
the signs and halves of any other energy are the caller's.
`df_rhf_gradient` returns every term on its own. `M` is solved by a
Cholesky factorisation, once per stage; nothing is screened, and a metric
that is not positive definite is refused, not regularised.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from felizjueves.ao.pairs import primitive_pairs
from felizjueves.ao.shell import Basis, cartesians
from felizjueves.eri.density_fit.eri3 import Eri3Class, eri3_tensor
from felizjueves.eri.density_fit.metric import (
    MetricClass,
    aux_records,
    metric_matrix,
)
from felizjueves.gradient.host import pair_weights, shell_atoms
from felizjueves.mmd.hermite import e_axis, hermite_order
from felizjueves.one_electron.potential import Nucleus

Ang = tuple[int, int, int]
Vec = tuple[float, float, float]
Grad = list[list[float]]
PQ = Callable[[Ang, Ang, Vec, float], float]


def _at(f: Sequence[float], t: int) -> float:
    return f[t] if t < len(f) else 0.0


def _hermite(factors: Sequence[Sequence[float]], ts: Sequence[Ang]
             ) -> list[float]:
    """`PROD_d f_d[t_d]` over `ts`, zero where an axis has no such entry
    (`gradient.host`'s, restated: a private name is not imported)."""
    return [_at(factors[0], t[0]) * _at(factors[1], t[1])
            * _at(factors[2], t[2]) for t in ts]


def _deriv(lo: Sequence[float] | None, hi: Sequence[float], c2: float,
           n: int) -> list[float]:
    """`c2 * hi - n * lo` entry by entry (`hi` one entry longer)."""
    out = [c2 * x for x in hi]
    if lo is not None and n:
        for t, x in enumerate(lo):
            out[t] -= n * x
    return out


class _Center:
    """One aux primitive's single-center `E^i_t` per axis, `i <= l + 1`
    (centre-free), and its derivative rows."""

    def __init__(self, alpha: float, l: int) -> None:  # noqa: E741
        self.alpha = alpha
        self.tab = [e_axis(i, 0, alpha, 0.0, 0.0) for i in range(l + 2)]

    def e(self, a: Ang) -> list[Sequence[float]]:
        return [self.tab[a[d]] for d in range(3)]

    def d(self, a: Ang, k: int) -> list[Sequence[float]]:
        f = self.e(a)
        i = a[k]
        f[k] = _deriv(self.tab[i - 1] if i else None, self.tab[i + 1],
                      2.0 * self.alpha, i)
        return f


class _Pair:
    """One AO primitive pair's per-axis `E^{ij}_t`, one index above the
    class on each side, and its two derivative rows."""

    def __init__(self, zeta: float, P: Vec, A: Vec, B: Vec, la: int, lb: int,
                 alpha: float, beta: float) -> None:
        self.alpha, self.beta = alpha, beta
        self.tab = [[[e_axis(i, j, zeta, P[d] - A[d], P[d] - B[d])
                      for j in range(lb + 2)] for i in range(la + 2)]
                    for d in range(3)]

    def e(self, a: Ang, b: Ang) -> list[Sequence[float]]:
        return [self.tab[d][a[d]][b[d]] for d in range(3)]

    def dA(self, a: Ang, b: Ang, k: int) -> list[Sequence[float]]:
        f = self.e(a, b)
        i, j, tk = a[k], b[k], self.tab[k]
        f[k] = _deriv(tk[i - 1][j] if i else None, tk[i + 1][j],
                      2.0 * self.alpha, i)
        return f

    def dB(self, a: Ang, b: Ang, k: int) -> list[Sequence[float]]:
        f = self.e(a, b)
        i, j, tk = a[k], b[k], self.tab[k]
        f[k] = _deriv(tk[i][j - 1] if j else None, tk[i][j + 1],
                      2.0 * self.beta, j)
        return f


def _add(G: Grad, row: int, g: Sequence[float], s: float) -> None:
    for k in range(3):
        G[row][k] += s * g[k]


def _rows(basis: Basis, nuclei: Sequence[Nucleus],
          atoms: Sequence[int] | None) -> tuple[int, ...]:
    if atoms is None:
        return shell_atoms(basis, nuclei)
    atoms = tuple(int(a) for a in atoms)
    if len(atoms) != len(basis.shells) or min(atoms, default=0) < 0:
        raise ValueError("atoms= needs one non-negative row per shell")
    return atoms


def _grid(n: int) -> Grad:
    return [[0.0, 0.0, 0.0] for _ in range(n)]


def _contract(left: Sequence[float], m: Sequence[Sequence[float]],
              right: Sequence[float]) -> float:
    """`SUM_t left_t SUM_tau m[t][tau] right_tau`."""
    out = 0.0
    for lt, row in zip(left, m, strict=True):
        if lt:
            out += lt * sum(x * y for x, y in zip(row, right, strict=True))
    return out


def metric_gradient(aux: Basis, W: Sequence[Sequence[float]],
                    nuclei: Sequence[Nucleus], *, pq: PQ,
                    pairs: Sequence[tuple[int, int]] | None = None,
                    atoms: Sequence[int] | None = None) -> Grad:
    """`G[A][k] = SUM_PQ W_PQ d(P|Q) / dR_{A,k}` (module docstring)."""
    sh = aux.shells
    rows = _rows(aux, nuclei, atoms)
    G = _grid(max(len(nuclei), max(rows, default=-1) + 1))
    n = aux.n_ao
    Wr = [[float(W[i][j]) for j in range(n)] for i in range(n)]
    if pairs is None:
        pairs = [(i, j) for i in range(len(sh)) for j in range(i, len(sh))]
    for i, j in pairs:
        if not 0 <= i <= j < len(sh):
            raise ValueError(f"aux shell pair ({i}, {j}) is not i <= j")
        sP, sQ = sh[i], sh[j]
        w = pair_weights(Wr, i, j, aux)
        rows_pq = [(a, b) for a in cartesians(sP.l) for b in cartesians(sQ.l)]
        ts, taus = hermite_order(sP.l + 1), hermite_order(sQ.l + 1)
        for p in aux_records(sP):
            cp = _Center(p.alpha, sP.l)
            for q in aux_records(sQ):
                cq = _Center(q.alpha, sQ.l)
                R = (p.A[0] - q.A[0], p.A[1] - q.A[1], p.A[2] - q.A[2])
                theta = math.sqrt(p.alpha * q.alpha / (p.alpha + q.alpha))
                m = [[pq(t, tau, R, theta) for tau in taus] for t in ts]
                gA, gB = [0.0] * 3, [0.0] * 3
                for wr, (a, b) in zip(w, rows_pq, strict=True):
                    if not wr:
                        continue
                    ea, eb = _hermite(cp.e(a), ts), _hermite(cq.e(b), taus)
                    for k in range(3):
                        gA[k] += wr * _contract(
                            _hermite(cp.d(a, k), ts), m, eb)
                        gB[k] += wr * _contract(
                            ea, m, _hermite(cq.d(b, k), taus))
                s = p.s * q.s
                _add(G, rows[i], gA, s)
                _add(G, rows[j], gB, s)
    return G


def _cell_weights(Gam, basis: Basis, aux: Basis, i: int, j: int, n: int
                  ) -> list[list[float]]:
    """`w[r][q]` of cell `(i, j | n)`, `i <= j`, rows `a` outer: the
    one-electron rule in `(m, n)` at each aux function."""
    sa, sb, sp = basis.shells[i], basis.shells[j], aux.shells[n]
    oa, ob, op = basis.offsets[i], basis.offsets[j], aux.offsets[n]
    out = []
    for r in range(sa.n_cart):
        for c in range(sb.n_cart):
            if i == j:
                out.append([float(Gam[oa + r][oa + c][op + q])
                            for q in range(sp.n_cart)])
            else:
                out.append([float(Gam[oa + r][ob + c][op + q])
                            + float(Gam[ob + c][oa + r][op + q])
                            for q in range(sp.n_cart)])
    return out


def eri3_gradient(basis: Basis, aux: Basis, Gam, nuclei: Sequence[Nucleus],
                  *, pq: PQ,
                  cells: Sequence[tuple[int, int, int]] | None = None,
                  atoms: Sequence[int] | None = None,
                  aux_atoms: Sequence[int] | None = None) -> Grad:
    """`G[A][k] = SUM_mnP Gam_mnP d(mn|P) / dR_{A,k}` (module docstring);
    `Gam` is indexed `[m][n][P]`, `eri3_tensor`'s layout."""
    sh, ash = basis.shells, aux.shells
    rows = _rows(basis, nuclei, atoms)
    arows = _rows(aux, nuclei, aux_atoms)
    G = _grid(max(len(nuclei), max((*rows, *arows), default=-1) + 1))
    if cells is None:
        cells = [(i, j, n) for i in range(len(sh)) for j in range(i, len(sh))
                 for n in range(len(ash))]
    for i, j, n in cells:
        if not (0 <= i <= j < len(sh) and 0 <= n < len(ash)):
            raise ValueError(f"cell ({i}, {j} | {n}) is not i <= j over "
                             "the two bases")
        sa, sb, sp = sh[i], sh[j], ash[n]
        w = _cell_weights(Gam, basis, aux, i, j, n)
        rows_ab = [(a, b) for a in cartesians(sa.l) for b in cartesians(sb.l)]
        cq = cartesians(sp.l)
        ts, taus = hermite_order(sa.l + sb.l + 1), hermite_order(sp.l + 1)
        for pp in primitive_pairs(sa, sb):
            alpha, beta = sa.exponents[pp.i], sb.exponents[pp.j]
            sab = (sa.coefficients[pp.i] * sb.coefficients[pp.j]
                   * (math.pi / pp.zeta) ** 1.5)
            ab = _Pair(pp.zeta, pp.P, sa.centre, sb.centre, sa.l, sb.l,
                       alpha, beta)
            for c in aux_records(sp):
                cc = _Center(c.alpha, sp.l)
                R = (pp.P[0] - c.A[0], pp.P[1] - c.A[1], pp.P[2] - c.A[2])
                theta = math.sqrt(pp.zeta * c.alpha / (pp.zeta + c.alpha))
                m = [[pq(t, tau, R, theta) for tau in taus] for t in ts]
                eq = [_hermite(cc.e(x), taus) for x in cq]
                gA, gB, gC = [0.0] * 3, [0.0] * 3, [0.0] * 3
                for r, (a, b) in enumerate(rows_ab):
                    e_ab = _hermite(ab.e(a, b), ts)
                    for k in range(3):
                        dA = _hermite(ab.dA(a, b, k), ts)
                        dB = _hermite(ab.dB(a, b, k), ts)
                        for q, x in enumerate(cq):
                            wq = w[r][q]
                            if not wq:
                                continue
                            gA[k] += wq * _contract(dA, m, eq[q])
                            gB[k] += wq * _contract(dB, m, eq[q])
                            gC[k] += wq * _contract(
                                e_ab, m, _hermite(cc.d(x, k), taus))
                s = sab * c.s
                _add(G, rows[i], gA, s)
                _add(G, rows[j], gB, s)
                _add(G, arows[n], gC, s)
    return G


def eri3_ip1(basis: Basis, aux: Basis, *, pq: PQ,
             cells: Sequence[tuple[int, int, int]] | None = None
             ) -> list[list[list[list[float]]]]:
    """The UNCONTRACTED d1 integrals in libcint's `int3c2e_ip1` form (#692):

        X[k][m][n][P] = (nabla_k m n | P) = -d(mn|P) / dA_k,

    `A` the centre of `m`, indexed `[3][n_ao][n_ao][n_aux]`. Over `cells`
    (default every `i <= j` row-major, aux inner, as `eri3_gradient`), each
    writing its `(m, n)` slot by the `dA` rule and its `(n, m)` slot by the
    `dB` rule; a diagonal shell pair `i == j` writes the `dA` slot of every
    `(m, n)` of the shell, which is all of them. An entry no listed cell
    owns is 0. The aux centre needs no table of its own: by translational
    invariance of each integral,

        (mn | nabla_k P) = -X[k][m][n][P] - X[k][n][m][P]

    (`int3c2e_ip2`), which the caller forms."""
    sh, ash = basis.shells, aux.shells
    nao, naux = basis.n_ao, aux.n_ao
    X = [[[[0.0] * naux for _ in range(nao)] for _ in range(nao)]
         for _ in range(3)]
    if cells is None:
        cells = [(i, j, n) for i in range(len(sh)) for j in range(i, len(sh))
                 for n in range(len(ash))]
    for i, j, n in cells:
        if not (0 <= i <= j < len(sh) and 0 <= n < len(ash)):
            raise ValueError(f"cell ({i}, {j} | {n}) is not i <= j over "
                             "the two bases")
        sa, sb, sp = sh[i], sh[j], ash[n]
        oa, ob, op = basis.offsets[i], basis.offsets[j], aux.offsets[n]
        ca, cb, cq = cartesians(sa.l), cartesians(sb.l), cartesians(sp.l)
        ts, taus = hermite_order(sa.l + sb.l + 1), hermite_order(sp.l)
        for pp in primitive_pairs(sa, sb):
            alpha, beta = sa.exponents[pp.i], sb.exponents[pp.j]
            sab = (sa.coefficients[pp.i] * sb.coefficients[pp.j]
                   * (math.pi / pp.zeta) ** 1.5)
            ab = _Pair(pp.zeta, pp.P, sa.centre, sb.centre, sa.l, sb.l,
                       alpha, beta)
            for c in aux_records(sp):
                cc = _Center(c.alpha, sp.l)
                R = (pp.P[0] - c.A[0], pp.P[1] - c.A[1], pp.P[2] - c.A[2])
                theta = math.sqrt(pp.zeta * c.alpha / (pp.zeta + c.alpha))
                m = [[pq(t, tau, R, theta) for tau in taus] for t in ts]
                eq = [_hermite(cc.e(x), taus) for x in cq]
                s = sab * c.s
                for ia, a in enumerate(ca):
                    for ib, b in enumerate(cb):
                        for k in range(3):
                            dA = _hermite(ab.dA(a, b, k), ts)
                            dB = None if i == j else _hermite(
                                ab.dB(a, b, k), ts)
                            for q, e in enumerate(eq):
                                X[k][oa + ia][ob + ib][op + q] -= (
                                    s * _contract(dA, m, e))
                                if dB is not None:
                                    X[k][ob + ib][oa + ia][op + q] -= (
                                        s * _contract(dB, m, e))
    return X


def metric_derivative(aux: Basis, *, pq: PQ,
                      pairs: Sequence[tuple[int, int]] | None = None
                      ) -> list[Matrix]:
    """`M1[k][P][Q] = d(P|Q) / dA_k`, `A` the centre of `P`: the
    uncontracted d1 metric (#693, `gradient.kernel_ipmetric`'s witness),
    `[3][n_aux][n_aux]`, every element of every pair in `pairs` (default
    all `i <= j`), zero elsewhere.

    Each centre is differentiated on its own, by the Cartesian rule: an
    off-diagonal pair `(i, j)` writes `d/dA` of `(P_a|Q_b)` at `[a][b]` and
    `d/dB` of it at `[b][a]` (there `Q_b` is the moved first index); a
    diagonal pair writes `d/dA` at every `[a][b]` (its `[b][a]` is the
    element `(b, a)`'s own). So `M1[k][Q][P] = -M1[k][P][Q]`, which the
    kernel writes as its mirror, is a CHECK here, not an assumption.
    libcint's `int2c2e_ip1` differentiates the electron coordinate:
    `M1 = -int2c2e_ip1`."""
    sh, off = aux.shells, aux.offsets
    n = aux.n_ao
    M1 = [[[0.0] * n for _ in range(n)] for _ in range(3)]
    if pairs is None:
        pairs = [(i, j) for i in range(len(sh)) for j in range(i, len(sh))]
    for i, j in pairs:
        if not 0 <= i <= j < len(sh):
            raise ValueError(f"aux shell pair ({i}, {j}) is not i <= j")
        sP, sQ = sh[i], sh[j]
        ts, taus = hermite_order(sP.l + 1), hermite_order(sQ.l + 1)
        for p in aux_records(sP):
            cp = _Center(p.alpha, sP.l)
            for q in aux_records(sQ):
                cq = _Center(q.alpha, sQ.l)
                R = (p.A[0] - q.A[0], p.A[1] - q.A[1], p.A[2] - q.A[2])
                theta = math.sqrt(p.alpha * q.alpha / (p.alpha + q.alpha))
                m = [[pq(t, tau, R, theta) for tau in taus] for t in ts]
                s = p.s * q.s
                for x, a in enumerate(cartesians(sP.l)):
                    ea = _hermite(cp.e(a), ts)
                    for y, b in enumerate(cartesians(sQ.l)):
                        eb = _hermite(cq.e(b), taus)
                        for k in range(3):
                            M1[k][off[i] + x][off[j] + y] += s * _contract(
                                _hermite(cp.d(a, k), ts), m, eb)
                            if i != j:
                                M1[k][off[j] + y][off[i] + x] += (
                                    s * _contract(ea, m, _hermite(
                                        cq.d(b, k), taus)))
    return M1


def metric_records(cls: MetricClass, aux: Basis,
                   nuclei: Sequence[Nucleus]) -> dict[str, list[int]]:
    """The `gmetric` kernel's arrays beyond `MetricClass.arrays()`: the atom
    of each staged pair's two shells, `atom_p` / `atom_q`, staged order."""
    at = shell_atoms(aux, nuclei)
    return {"atom_p": [at[p] for p, _ in cls.pairs],
            "atom_q": [at[q] for _, q in cls.pairs]}


def eri3_records(cls: Eri3Class, basis: Basis, aux: Basis,
                 nuclei: Sequence[Nucleus]) -> dict[str, list]:
    """The `geri3` kernel's arrays beyond `Eri3Class.arrays()`: per bra
    record the two primitive exponents `ea_bra` / `eb_ket` (the derivative
    rule's `2 alpha`, `2 beta`), in `primitive_pairs` order of each staged
    pair; per cell the atoms `atom_a`, `atom_b` (staged order) and
    `atom_p`."""
    at, aat = shell_atoms(basis, nuclei), shell_atoms(aux, nuclei)
    out: dict[str, list] = {"ea_bra": [], "eb_ket": []}
    for a, b in cls.bra_pairs:
        sa, sb = basis.shells[a], basis.shells[b]
        for pp in primitive_pairs(sa, sb):
            out["ea_bra"].append(sa.exponents[pp.i])
            out["eb_ket"].append(sb.exponents[pp.j])
    if len(out["ea_bra"]) != cls.n_bra_rec:
        raise ValueError("the class's bra records are not this basis's")
    out["atom_a"] = [at[a] for a, _, _ in cls.cells]
    out["atom_b"] = [at[b] for _, b, _ in cls.cells]
    out["atom_p"] = [aat[n] for _, _, n in cls.cells]
    return out


# --- the staged DF-J and DF-K terms (#620; module docstring) -----------------

Matrix = list[list[float]]
Tensor = list[list[list[float]]]


@dataclass(frozen=True)
class DfStage:
    """One fitted energy and the weights of its gradient: `Gam[m][n][P]`
    for `eri3_gradient`, `W[P][Q]` for `metric_gradient`."""

    energy: float
    Gam: Tensor
    W: Matrix


@dataclass(frozen=True)
class DfTerm:
    """A stage contracted: `G = eri3 + metric`, each `n_atom x 3`."""

    stage: DfStage
    eri3: Grad
    metric: Grad
    G: Grad


@dataclass(frozen=True)
class DfRhfGradient:
    """The closed-shell DF two-electron gradient, `G = J.G - 1/2 K.G`."""

    J: DfTerm
    K: DfTerm
    G: Grad


def _cholesky(M: Sequence[Sequence[float]]) -> Matrix:
    """The lower factor `L L^T = M`; refuses a matrix that is not
    positive definite."""
    n = len(M)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = float(M[i][j]) - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if not s > 0.0:
                    raise ValueError(f"the metric is not positive definite "
                                     f"(pivot {i}: {s!r})")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _solve(L: Matrix, b: Sequence[float]) -> list[float]:
    """`x` of `L L^T x = b`."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def _sizes(T, M) -> tuple[int, int]:
    n_ao, n_aux = len(T), len(M)
    if any(len(r) != n_aux for r in M) or any(
            len(r) != n_ao or any(len(c) != n_aux for c in r) for r in T):
        raise ValueError("T is not [n_ao][n_ao][n_aux] over M's n_aux")
    return n_ao, n_aux


def df_j_stage(T, M, D) -> DfStage:
    """`E_J`, `Gam^J = D (x) d`, `W^J = -1/2 d d^T` (module docstring)."""
    n_ao, n_aux = _sizes(T, M)
    b = [sum(float(D[m][n]) * float(T[m][n][p]) for m in range(n_ao)
             for n in range(n_ao)) for p in range(n_aux)]
    d = _solve(_cholesky(M), b)
    return DfStage(
        0.5 * sum(x * y for x, y in zip(b, d, strict=True)),
        [[[float(D[m][n]) * d[p] for p in range(n_aux)] for n in range(n_ao)]
         for m in range(n_ao)],
        [[-0.5 * d[p] * d[q] for q in range(n_aux)] for p in range(n_aux)])


def df_k_stage(T, M, C, *, occ: float = 2.0) -> DfStage:
    """`E_K`, `Gam^K`, `W^K` of `D = occ C C^T`, `C` the occupied
    coefficients `[n_ao][n_occ]` (module docstring)."""
    n_ao, n_aux = _sizes(T, M)
    n_occ = len(C[0]) if n_ao else 0
    if len(C) != n_ao or any(len(r) != n_occ for r in C):
        raise ValueError("C is not [n_ao][n_occ] over T's n_ao")
    Cf = [[float(x) for x in r] for r in C]
    # Half-transformed: H[i][n][P] = SUM_m C_mi (mn|P).
    H = [[[sum(Cf[m][i] * float(T[m][n][p]) for m in range(n_ao))
           for p in range(n_aux)] for n in range(n_ao)] for i in range(n_occ)]
    L = _cholesky(M)
    pairs = [(i, j) for i in range(n_occ) for j in range(n_occ)]
    X = {(i, j): [sum(Cf[n][j] * H[i][n][p] for n in range(n_ao))
                  for p in range(n_aux)] for i, j in pairs}
    d = {ij: _solve(L, X[ij]) for ij in pairs}
    o2 = occ * occ
    energy = 0.5 * o2 * sum(x * y for ij in pairs
                            for x, y in zip(X[ij], d[ij], strict=True))
    # Back: E[i][n][P] = SUM_j C_nj d_ijP, Gam_mnP = occ^2 SUM_i C_mi E_inP.
    E = [[[sum(Cf[n][j] * d[i, j][p] for j in range(n_occ))
           for p in range(n_aux)] for n in range(n_ao)] for i in range(n_occ)]
    Gam = [[[o2 * sum(Cf[m][i] * E[i][n][p] for i in range(n_occ))
             for p in range(n_aux)] for n in range(n_ao)] for m in range(n_ao)]
    W = [[-0.5 * o2 * sum(d[ij][p] * d[ij][q] for ij in pairs)
          for q in range(n_aux)] for p in range(n_aux)]
    return DfStage(energy, Gam, W)


def _sum(a: Grad, b: Grad, s: float = 1.0) -> Grad:
    return [[x + s * y for x, y in zip(ra, rb, strict=True)]
            for ra, rb in zip(a, b, strict=True)]


def df_term(basis: Basis, aux: Basis, stage: DfStage,
            nuclei: Sequence[Nucleus], *, pq: PQ) -> DfTerm:
    """`stage`'s gradient: `eri3_gradient(Gam) + metric_gradient(W)`."""
    e = eri3_gradient(basis, aux, stage.Gam, nuclei, pq=pq)
    m = metric_gradient(aux, stage.W, nuclei, pq=pq)
    return DfTerm(stage, e, m, _sum(e, m))


def df_rhf_gradient(basis: Basis, aux: Basis, C, nuclei: Sequence[Nucleus],
                    *, pq: PQ, occ: float = 2.0) -> DfRhfGradient:
    """The DF two-electron gradient of `E_J - 1/2 E_K` at `D = occ C C^T`,
    `C` the occupied coefficients `[n_ao][n_occ]`, every term returned on
    its own (module docstring). `T` and `M` are computed once."""
    T = eri3_tensor(basis, aux, pq=pq)
    M = metric_matrix(aux, pq=pq)
    n_ao, n_occ = len(C), len(C[0]) if len(C) else 0
    D = [[occ * sum(float(C[m][i]) * float(C[n][i]) for i in range(n_occ))
          for n in range(n_ao)] for m in range(n_ao)]
    j = df_term(basis, aux, df_j_stage(T, M, D), nuclei, pq=pq)
    k = df_term(basis, aux, df_k_stage(T, M, C, occ=occ), nuclei, pq=pq)
    return DfRhfGradient(j, k, _sum(j.G, k.G, -0.5))
