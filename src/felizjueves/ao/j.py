"""The Coulomb matrix `J_mn`, through the J-engine split -- by two sweeps.

    coulomb_matrix(basis, D, *, pq, tau=0.0)            -> J   (unique pairs)
    coulomb_matrix_reference(basis, D, *, pq, tau=0.0)  -> J   (all ordered)
    coulomb_matrix_and_scale(basis, D, *, pq)           -> (J, scale)
    coulomb_matrix_screened(basis, D, *, pq, tau, order="sorted")
                                                        -> Screened (#88)

`J_mn = sum_{ls} (mn|ls) D_ls`, never by forming `(mn|ls)`. For a bra
primitive pair (index record) of an ordered shell pair `(a, b)` and a ket
primitive pair `k` of `(c, d)` ([JEngine, v2, eqs. 20-22]; j_engine.md, "The
Coulomb contraction"):

    D_q  = w sum_{cd} E^{cd}_q D_cd     ket pre-contraction, per ket record
    J_p  = sum_k sum_q [p|q] D_q        the ONLY shell-quartet-level work
    J_ab = sum_p E^{ab}_p J_p           bra transform, per bra record

with `E` the DRESSED coefficients of `ao.pairs.primitive_pairs`, both sides:
`(pi/zeta)^{3/2}` is on the bra `E` and `(pi/eta)^{3/2}` on the ket `E` --
so on `D_q` -- once each, by `pairs.dress`, and this module multiplies no
constant in. `[p|q]` carries `(2/sqrt(pi)) theta` and the `(-1)^{|q|}` sign
inside the injected evaluator, exactly as in `ao/eri.py`, whose oracle
comparison per class is what makes the pieces trustworthy one level down.

THE TWO SWEEPS, and why there are two (the off-diagonal weight is the
order-unity error this design is most likely to ship; the predecessor project
shipped it as 1 instead of 2 in a path "verified" against itself):

  `coulomb_matrix_reference`  every ORDERED ket shell pair `(c, d)` at weight
      `w = 1`, and every ordered bra shell pair `(a, b)` writing only its own
      block `J[a][b]`. No symmetry is used anywhere -- `J[b][a]` is computed,
      not copied -- so it is slow and it is the internal check.
  `coulomb_matrix`  unique ket pairs `c >= d` (shell index) with `w = 2`
      off the diagonal (`c != d`) and `w = 1` on it, because
      `(ab|cd) D_cd + (ab|dc) D_dc = 2 (ab|cd) D_cd` for a symmetric `D`; and
      unique bra pairs `a >= b`, each block scattered to `J[a][b]` and,
      transposed, to `J[b][a]`.

They must agree (test/test_ao_j.py). Do not "optimise" the reference sweep
away once it is green: it exists to catch the weight and nothing else.
The weight is folded INTO `D_q` -- what a driver would ship the kernel as its
ket `D` -- so the screen's `Dmax` sees it too.

THE DENSITY MUST BE SYMMETRIC, and an asymmetric one is REJECTED
(`ValueError`), not symmetrised: the unique sweep's `w = 2` is valid only for
`D_cd = D_dc`, and symmetrising silently would hide a caller's transposed
index. Exact equality is required -- a density that is symmetric "to
rounding" is the caller's to symmetrise. Both sweeps reject, so they have the
same contract.

THE SCREEN, `tau`. A (bra record, ket record) pair is skipped when
`I_bra * (I_ket * Dmax_ket) < tau` -- the kernel's own product, with
`I = sum_p I_p` the pair-factor-free Schwarz bound of `ao/pairs.py` and
`Dmax = max_q |D_q|` over the weighted, dressed ket vector. The bound is
evaluated through the injected `pq` as `I_p = sqrt([p|p](PQ = 0,
theta = sqrt(zeta/2)))`: `[p|p] = (-1)^{|p|} [2p]^{(0)}`, which is exactly
`pairs.hermite_bound`'s radicand, so no sign is computed here (a test pins
the two equal). This is the UNSORTED `continue` form; the sorted `break` is
the kernel's. AT `tau = 0.0` (THE DEFAULT) NO BOUND IS EVALUATED AND NOTHING
CAN BE SKIPPED: the product is >= 0, and the branch is not even taken -- so a
comparison at `tau = 0` cannot have an error hidden by screening.

THE SCALE (`coulomb_matrix_and_scale`, unique sweep, `tau = 0`): the same
sum with every factor replaced by its absolute value -- `|E^{ab}_p|`,
`|[p|q]|`, and `|D_q|` replaced by `w sum_{cd} |E^{cd}_q| |D_cd|` -- which is
exactly `sum_{ls} scale_{mnls} |D_ls|` for `ao.eri`'s per-element ERI scale.
So a per-ERI bound `|err| <= t scale_eri` gives `|err J| <= t scale_J` by the
triangle inequality: the J tolerance is inherited, not chosen.

THE SORTED `break` (#88, `coulomb_matrix_screened`) -- the kernel's form, on
real bounds. The ket records are sorted DESCENDING by `I_ket * Dmax`
(`_Ket.bound`, exactly the factor the screen multiplies by `I_bra`), with a
STABLE sort, and each bra record stops at its first ket under `tau`. That is
exact, not approximate: `I_bra >= 0`, and a rounded product with a fixed
non-negative factor is monotone non-decreasing in the other, so every ket
after the first rejected one is rejected too. `order="unsorted"` is the
`continue` form in the original ket order. The two must give the SAME `J`
to 0, and to make that a statement about WHICH terms were summed rather than
about summation order, each `J_p` is accumulated over its kets with
`math.fsum` (correctly rounded, hence independent of order); the per-ket
term itself is computed identically on both paths. `Screened.n_skipped`
counts the (bra record, ket record) index pairs not summed.

`Screened.scale` is the error claim's stated scale, independent of `tau`:
per element, `n_ket * sum_{bra records} sum_p |E^{ab}_p|`. Each skipped pair
has `|sum_q [p|q] D_q| <= I_p I_ket Dmax <= I_bra I_ket Dmax < tau` for every
`p` (ao/pairs.py), so it moves `J_ab` by at most `tau sum_p |E^{ab}_p|`, and
a bra record skips at most `n_ket` kets: `|J(tau) - J(0)| <= tau * scale`.

MEMOISE NOTHING. The ket records are built once per call and dropped; a cache
keyed on a shell pair is a driver's concern, and one here would make #88's
tau ladder measure the cache.

Pure Python floats; `D` is anything indexable as `D[m][n]` (a numpy array
works) and `J` is a tuple of rows. `src/felizjueves` has no runtime
dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from felizjueves.ao.hermite import hermite_order
from felizjueves.ao.pairs import PrimitivePair, primitive_pairs
from felizjueves.ao.shell import Basis

Vec = tuple[float, float, float]
Ang = tuple[int, int, int]
PQ = Callable[[Ang, Ang, Vec, float], float]
Matrix = tuple[tuple[float, ...], ...]

_ORIGIN = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _Ket:
    """One ket index record: `Q`, `eta`, the weighted dressed `D_q` (and its
    |.| twin), its Hermite indices, and the screen's `I_ket * Dmax`."""

    Q: Vec
    eta: float
    qs: tuple[Ang, ...]
    dq: tuple[float, ...]
    dq_abs: tuple[float, ...]
    bound: float  # I_ket * Dmax; 0.0 (unused) when tau == 0


def _density(basis: Basis, D: object) -> list[list[float]]:
    n = basis.n_ao
    try:
        rows = [[float(D[i][j]) for j in range(n)] for i in range(n)]  # type: ignore[index]
        extra = len(D) != n or any(len(D[i]) != n for i in range(n))  # type: ignore[arg-type,index]
    except (TypeError, IndexError, KeyError):
        raise ValueError(f"D is not an {n} x {n} matrix") from None
    if extra:
        raise ValueError(f"D is not an {n} x {n} matrix")
    if not all(math.isfinite(x) for r in rows for x in r):
        raise ValueError("D has a non-finite entry")
    for i in range(n):
        for j in range(i):
            if rows[i][j] != rows[j][i]:
                raise ValueError(
                    f"D is not symmetric: D[{i}][{j}] = {rows[i][j]!r} but "
                    f"D[{j}][{i}] = {rows[j][i]!r}. An asymmetric density is "
                    "rejected, not symmetrised (ao/j.py's module docstring)")
    return rows


def _tau(tau: float) -> float:
    if not (isinstance(tau, int | float) and math.isfinite(tau) and tau >= 0.0):
        raise ValueError(f"tau = {tau!r} is not a finite float >= 0")
    return float(tau)


def pair_bound_pq(zeta: float, L: int, *, pq: PQ) -> float:
    """`I = sum_p sqrt([p|p](PQ = 0, theta = sqrt(zeta/2)))` over
    `hermite_order(L)` -- `ao.pairs.pair_bound` through the `[p|q]` evaluator
    (module docstring). Raises on a negative radicand, never `abs`es it."""
    theta = math.sqrt(zeta / 2.0)
    out = []
    for p in hermite_order(L):
        s = pq(p, p, _ORIGIN, theta)
        if s < 0.0:
            raise ValueError(
                f"[p|p](0) = {s!r} < 0 at p = {p}, zeta = {zeta!r}: a diagonal "
                "of a positive-definite form, so the evaluator's sign "
                "convention is wrong")
        out.append(math.sqrt(s))
    return math.fsum(out)


def _kets(basis: Basis, D: list[list[float]], pairs, *, pq: PQ,
          tau: float) -> list[_Ket]:
    """The ket records of the shell pairs `pairs` = [(c, d, w)], `D_q`
    pre-contracted with weight `w`."""
    shells, off = basis.shells, basis.offsets
    out = []
    for c, d, w in pairs:
        sc, sd = shells[c], shells[d]
        # D_cd over this pair's Cartesian pairs, c outer: the block's row order.
        dcd = [D[off[c] + i][off[d] + j]
               for i in range(sc.n_cart) for j in range(sd.n_cart)]
        qs = hermite_order(sc.l + sd.l)
        for k in primitive_pairs(sc, sd):
            dq = [0.0] * len(qs)
            da = [0.0] * len(qs)
            for x, row in zip(dcd, k.block, strict=True):
                if x:
                    ax = abs(x)
                    for t, e in enumerate(row):
                        dq[t] += x * e
                        da[t] += ax * abs(e)
            dq = tuple(w * v for v in dq)
            da = tuple(w * v for v in da)
            bound = 0.0
            if tau > 0.0:
                bound = (pair_bound_pq(k.zeta, k.L, pq=pq)
                         * max(abs(v) for v in dq))
            out.append(_Ket(k.P, k.zeta, qs, dq, da, bound))
    return out


def _bra_block(bra_recs: Sequence[PrimitivePair], kets: Sequence[_Ket], *,
               pq: PQ, tau: float, want_scale: bool):
    """`J_ab` (rows `ab`, a outer) and its |.| twin, for one bra shell pair."""
    nrow = len(bra_recs[0].block)
    jab = [0.0] * nrow
    jab_abs = [0.0] * nrow
    for bra in bra_recs:
        ps = hermite_order(bra.L)
        zeta, P = bra.zeta, bra.P
        jp = [0.0] * len(ps)
        jp_abs = [0.0] * len(ps)
        ibra = pair_bound_pq(zeta, bra.L, pq=pq) if tau > 0.0 else 0.0
        for k in kets:
            if tau > 0.0 and ibra * k.bound < tau:
                continue
            theta = math.sqrt(zeta * k.eta / (zeta + k.eta))
            PQv = (P[0] - k.Q[0], P[1] - k.Q[1], P[2] - k.Q[2])
            for ip, p in enumerate(ps):
                acc = 0.0
                acc_abs = 0.0
                for q, dq, da in zip(k.qs, k.dq, k.dq_abs, strict=True):
                    v = pq(p, q, PQv, theta)
                    acc += v * dq
                    if want_scale:
                        acc_abs += abs(v) * da
                jp[ip] += acc
                jp_abs[ip] += acc_abs
        for r, row in enumerate(bra.block):
            s = 0.0
            sa = 0.0
            for e, v, va in zip(row, jp, jp_abs, strict=True):
                s += e * v
                if want_scale:
                    sa += abs(e) * va
            jab[r] += s
            jab_abs[r] += sa
    return jab, jab_abs


def _sweep(basis: Basis, D: object, *, pq: PQ, tau: float, ordered: bool,
           want_scale: bool) -> tuple[Matrix, Matrix]:
    rows = _density(basis, D)
    tau = _tau(tau)
    n_sh = len(basis.shells)
    if ordered:
        pairs = [(a, b, 1.0) for a in range(n_sh) for b in range(n_sh)]
    else:
        pairs = [(a, b, 1.0 if a == b else 2.0)
                 for a in range(n_sh) for b in range(a + 1)]
    kets = _kets(basis, rows, pairs, pq=pq, tau=tau)
    n = basis.n_ao
    J = [[0.0] * n for _ in range(n)]
    S = [[0.0] * n for _ in range(n)]
    shells, off = basis.shells, basis.offsets
    for a, b, _ in pairs:  # the bra pairs: the same set, unweighted
        sa, sb = shells[a], shells[b]
        jab, jab_abs = _bra_block(primitive_pairs(sa, sb), kets, pq=pq,
                                  tau=tau, want_scale=want_scale)
        for i in range(sa.n_cart):
            for j in range(sb.n_cart):
                r = i * sb.n_cart + j
                J[off[a] + i][off[b] + j] = jab[r]
                S[off[a] + i][off[b] + j] = jab_abs[r]
                if not ordered:  # the mirror: identical on the diagonal pair
                    J[off[b] + j][off[a] + i] = jab[r]
                    S[off[b] + j][off[a] + i] = jab_abs[r]
    return (tuple(tuple(r) for r in J), tuple(tuple(r) for r in S))


def coulomb_matrix(basis: Basis, D: object, *, pq: PQ,
                   tau: float = 0.0) -> Matrix:
    """`J_mn` by unique bra and ket shell pairs, ket off-diagonal weight 2."""
    return _sweep(basis, D, pq=pq, tau=tau, ordered=False,
                  want_scale=False)[0]


def coulomb_matrix_reference(basis: Basis, D: object, *, pq: PQ,
                             tau: float = 0.0) -> Matrix:
    """`J_mn` by EVERY ordered bra and ket shell pair, weight 1, no symmetry.
    The internal check on `coulomb_matrix`'s weight; keep it."""
    return _sweep(basis, D, pq=pq, tau=tau, ordered=True, want_scale=False)[0]


def coulomb_matrix_and_scale(basis: Basis, D: object, *,
                             pq: PQ) -> tuple[Matrix, Matrix]:
    """`coulomb_matrix` at `tau = 0`, and its absolute-value twin `scale`
    (module docstring) -- the size of what was added, per element."""
    return _sweep(basis, D, pq=pq, tau=0.0, ordered=False, want_scale=True)


# --- the screen as an error claim (#88): the sorted break -----------------


@dataclass(frozen=True)
class Screened:
    """`coulomb_matrix_screened`'s result (module docstring, "THE SORTED
    `break`"): `J`, the `tau`-independent error scale, and the count of
    (bra record, ket record) index pairs, total and skipped."""

    J: Matrix
    scale: Matrix
    n_pairs: int
    n_skipped: int

    @property
    def skip_fraction(self) -> float:
        return self.n_skipped / self.n_pairs


ORDERS = ("sorted", "unsorted")


def _sorted_kets(kets: Sequence[_Ket]) -> list[_Ket]:
    """Descending by `bound` (= `I_ket * Dmax`), STABLE: `sorted` is stable,
    and `reverse=True` keeps equal keys in their original order."""
    return sorted(kets, key=lambda k: k.bound, reverse=True)


def _bra_block_screened(bra_recs: Sequence[PrimitivePair],
                        kets: Sequence[_Ket], *, pq: PQ, tau: float,
                        brk: bool):
    nrow = len(bra_recs[0].block)
    jab = [0.0] * nrow
    sab = [0.0] * nrow
    skipped = 0
    n_ket = len(kets)
    for bra in bra_recs:
        ps = hermite_order(bra.L)
        zeta, P = bra.zeta, bra.P
        terms: list[list[float]] = [[] for _ in ps]
        ibra = pair_bound_pq(zeta, bra.L, pq=pq)
        for i, k in enumerate(kets):
            if ibra * k.bound < tau:
                if brk:  # sorted: every later ket is under tau too
                    skipped += n_ket - i
                    break
                skipped += 1
                continue
            theta = math.sqrt(zeta * k.eta / (zeta + k.eta))
            PQv = (P[0] - k.Q[0], P[1] - k.Q[1], P[2] - k.Q[2])
            for ip, p in enumerate(ps):
                acc = 0.0
                for q, dq in zip(k.qs, k.dq, strict=True):
                    acc += pq(p, q, PQv, theta) * dq
                terms[ip].append(acc)
        jp = [math.fsum(t) for t in terms]  # order-independent: see docstring
        for r, row in enumerate(bra.block):
            s = 0.0
            for e, v in zip(row, jp, strict=True):
                s += e * v
            jab[r] += s
            sab[r] += n_ket * math.fsum(abs(e) for e in row)
    return jab, sab, skipped


def coulomb_matrix_screened(basis: Basis, D: object, *, pq: PQ, tau: float,
                            order: str = "sorted") -> Screened:
    """`J_mn` by the unique sweep, screened at `tau` as the kernel's sorted
    `break` (`order="sorted"`) or as the unsorted `continue`
    (`order="unsorted"`); module docstring, "THE SORTED `break`". The bound
    is evaluated at every `tau`, 0 included (it is the sort key); at
    `tau = 0` nothing is skipped, since the product is >= 0."""
    if order not in ORDERS:
        raise ValueError(f"order = {order!r} is not one of {ORDERS}")
    rows = _density(basis, D)
    tau = _tau(tau)
    n_sh = len(basis.shells)
    pairs = [(a, b, 1.0 if a == b else 2.0)
             for a in range(n_sh) for b in range(a + 1)]
    # `_kets` evaluates the bound only when its tau is > 0; this one is
    # passed to switch it on and is used for nothing else.
    kets = _kets(basis, rows, pairs, pq=pq, tau=1.0)
    if order == "sorted":
        kets = _sorted_kets(kets)
    n = basis.n_ao
    J = [[0.0] * n for _ in range(n)]
    S = [[0.0] * n for _ in range(n)]
    shells, off = basis.shells, basis.offsets
    n_pairs = n_skipped = 0
    for a, b, _ in pairs:
        sa, sb = shells[a], shells[b]
        bras = primitive_pairs(sa, sb)
        jab, sab, sk = _bra_block_screened(bras, kets, pq=pq, tau=tau,
                                           brk=order == "sorted")
        n_pairs += len(bras) * len(kets)
        n_skipped += sk
        for i in range(sa.n_cart):
            for j in range(sb.n_cart):
                r = i * sb.n_cart + j
                J[off[a] + i][off[b] + j] = J[off[b] + j][off[a] + i] = jab[r]
                S[off[a] + i][off[b] + j] = S[off[b] + j][off[a] + i] = sab[r]
    return Screened(tuple(tuple(r) for r in J), tuple(tuple(r) for r in S),
                    n_pairs, n_skipped)
