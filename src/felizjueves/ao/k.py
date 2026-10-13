"""The exchange matrix `K_mn`, by the staged contraction -- by two sweeps.

    exchange_matrix(basis, D, *, pq, tau=0.0, k=None)   -> K   (unique)
    exchange_matrix_reference(basis, D, *, pq, tau=0.0) -> K   (all ordered)
    exchange_matrix_and_scale(basis, D, *, pq)          -> (K, scale)
    k_from_eris(basis, D, *, pq)                        -> K   via ao.eri
    k_quartet(bra_pair, ket_pair, D_cd, *, pq)          the seam's reference
    exchange_matrix_screened(basis, D, *, pq, tau, order="sorted")
                                                        -> KScreened (#200)
    record_bounds(pair, *, pq)                          -> I per record

`K_ab = sum_{cd} D_cd (ac|bd)`, never by forming `(ac|bd)`. The BRA pair
record is `(a, c)` and the KET pair record is `(b, d)` -- the two Hermite
pairs of the integral, not the AO quartet order -- so a K class is
`am = (la, lc, lb, ld)`, bra first: `(lp, lq) = (la + lc, lb + ld)`. (A class
spelled in AO order `(la, lb, lc, ld)` has the same `L` and the wrong split,
and plans a different kernel that still returns a plausible number; the
milestone's design pin.) Per primitive quartet (a bra record of the ordered
shell pair `(sa, sc)`, a ket record of `(sb, sd)`):

    stage 1 (ket)     I[p]    = sum_q [p|q] E^{bd}_q        per (b,d) row
    stage 2 (bra)     V(a,c)  = sum_p E^{ac}_p I[p]         per (a,c) row
    stage 3 (digest)  K[a,b] += D[c,d] V(a,c)

STAGES 2 AND 3 ARE FUSED INSIDE THE `(b,d)` LOOP and `V` is never an array:
`I` is one column of `n_p` values, rebuilt per `(b,d)` row, and each `V` is
digested the moment it is formed. That is the device kernel's shape
(`kernel_k.py`, #198), so the host is its bit-for-bit witness; it is forced
by the shared-memory budget, not chosen (the milestone description). `E` is
the DRESSED coefficient of `ao.pairs.primitive_pairs` on both sides, formed
as `s * E` with `s` from `pairs.dress` (via `ao.j.epair`, exactly as `ao.j`'s
seams form it), so `(pi/zeta)^{3/2}` rides the bra `E` and `(pi/eta)^{3/2}`
the ket `E`, once each, and this module multiplies no constant in. `[p|q]`
carries `(2/sqrt(pi)) theta` and `(-1)^{|q|}` inside the injected evaluator,
as in `ao/eri.py` and `ao/j.py`.

THE TWO SWEEPS, and why K's weights are not J's. `ao/j.py` uses two
symmetries: the ket pair `(c, d)` is SUMMED, so `(ab|cd) D_cd + (ab|dc) D_dc
= 2 (ab|cd) D_cd` gives the ket weight `w = 2` off the diagonal; and `J` is
symmetric, so a bra pair `a >= b` is mirrored. In K neither pair is summed:
each record holds one OUTPUT index (`a`, `b`) and one SUMMED index (`c`,
`d`), so swapping inside a pair, `(ac|bd) = (ca|bd)`, moves the term to a
different element (`K_cb`, weighted by `D_ad`) and buys nothing a
per-element contraction can use without forming `V`. The one symmetry the
staged form CAN use is the bra-ket swap: write `X(a,c|b,d)` for
`k_quartet`'s block, `X[i][j] = sum_{kl} D_{c_k d_l} (a_i c_k|b_j d_l)`.
Then, for a symmetric `D`,

    X(b,d|a,c) = X(a,c|b,d)^T      [(b_j d_l|a_i c_k) = (a_i c_k|b_j d_l),
                                    and D_{d_l c_k} = D_{c_k d_l}]

so the ordered quadruples `(a,b,c,d)` fall into orbits
`{(a,b,c,d), (b,a,d,c)}` of size 2, with fixed points `a = b, c = d`. The
unique sweep visits one quadruple per orbit, and its WEIGHTS are where the
two members of the orbit land:

  `a > b`, every `(c, d)`   `X` to block `K[a][b]` and `X^T` to `K[b][a]`,
      each at weight 1: the partner `(b,a,d,c)` is a different block, and
      it is the transpose. (This is J's bra mirror, and it is the WHOLE of
      `K`'s symmetry, not an extra on top of a ket weight.)
  `a = b`, `c > d`          `X + X^T` to `K[a][a]`: both members land in the
      SAME block, transposed -- NOT `2 X`. `X` is not symmetric here (a test
      shows it on a `p` shell), so J's scalar `w = 2` would be wrong by
      `X - X^T`; the scalar 2 is the order-unity error this design is most
      likely to ship.
  `a = b`, `c = d`          `X` once, at weight 1: a fixed point, its own
      partner, and symmetric by the same identity.

  `exchange_matrix_reference`  every ORDERED quadruple at weight 1, every
      block `K[a][b]` its own sum, no symmetry anywhere -- `K[b][a]` is
      computed, not copied. Slow, and the internal check.
  `exchange_matrix`  the unique sweep above.

They must agree to rounding (test/test_ao_k.py). The check exists to catch a
WEIGHT error -- J's `w = 2` carried over (a `c > d` ket counted twice, or a
diagonal block's `X + X^T` collapsed to `2 X`), or an orbit visited twice or
not at all -- and nothing else: it cannot see an error both sweeps share
(the contraction itself, a dressing, a convention), which is what
`k_from_eris` and the libcint fixture (#199) are for. Do not "optimise" the
reference away once it is green.

THE DENSITY MUST BE SYMMETRIC, and an asymmetric one is REJECTED
(`ValueError`, `ao.j`'s check, verbatim), not symmetrised: the transpose
identity above holds only for `D_cd = D_dc`. Every entry point rejects, so
they share one contract.

`k_from_eris` is a THIRD index path: every ordered quadruple's full ERI
block from `ao.eri.eri_block((sa, sc), (sb, sd))`, contracted with `D_cd`
element by element. It shares the `[p|q]` evaluator and the dressed `E`
with the sweeps and nothing else -- no staging, no fusing, no symmetry.

THE SCALE (`exchange_matrix_and_scale`, unique sweep): the same sum with
every factor in absolute value -- `|D_cd|`, `|E^{ac}_p|`, `|[p|q]|`,
`|E^{bd}_q|` -- and the same weights (`|X|^T` is the partner's scale, since
`|[p|q]|` is bra-ket symmetric). So `scale_K[a][b] = sum_{cd} |D_cd|
scale_eri(ac|bd)` for `ao.eri`'s per-element ERI scale, and a per-ERI bound
`|err| <= t scale_eri` gives `|err K| <= t scale_K` by the triangle
inequality: the K tolerance is inherited, not chosen, exactly as
`ao.j.coulomb_matrix_and_scale`'s is.

THE SCREEN (#200). A (bra record, ket record) pair -- one primitive pair
of `(sa, sc)` against one of `(sb, sd)` -- is skipped when

    (I_ac * I_bd) * Dm < tau,      Dm = max_{mn} |D_mn|  (the whole density)

with `I` the per-record scalar `record_bounds` returns:

    I_ac = max_{i in a} sum_{k in c} sum_p |E^{a_i c_k}_p| I_p(zeta)

`I_p = sqrt([p|p](PQ = 0, theta = sqrt(zeta/2)))` is `ao.pairs.hermite_bound`
-- the pair-factor-free Schwarz factor #84/#85 proved for J -- evaluated
through the injected `pq` exactly as `ao.j.pair_bound_pq` does (a test pins
it equal to `hermite_bound` under `reference.pq.one_centre`), and `E` is the
DRESSED coefficient, the same `s * E` the contraction uses. THE PROOF that
it bounds: the record pair's contribution to `K[a_i][b_j]` is

    X_ij = sum_{kl} D_{c_k d_l} sum_{pq} E^{ik}_p [p|q] E^{jl}_q

and `|[p|q]| <= I_p I_q` (ao/pairs.py: Cauchy-Schwarz on the Coulomb form,
the pair factors cancelling), so

    |X_ij| <= sum_{kl} |D_kl| (sum_p |E^{ik}_p| I_p) (sum_q |E^{jl}_q| I_q)
           <= Dm (sum_k sum_p |E^{ik}_p| I_p) (sum_l sum_q |E^{jl}_q| I_q)
           <= Dm I_ac I_bd.

The dressing rides `I` here where in J it rides `Dmax`: J's `D_q` is a
pre-contraction that carries the ket's `E`, and K has none (next
paragraph). test/test_ao_k_screening.py asserts the inequality over
randomised records rather than taking it on authority.

THE KEY THE J ENGINE USES IS UNAVAILABLE FOR K. J sorts its kets by
`I_ket * Dmax_ket` because `D_q` belongs to the ket alone. In K the density
element `D_cd` STRADDLES the two records -- `c` is the bra's summed index,
`d` the ket's -- so no ket-side quantity tracks it, and no ket order
follows the density. [Fock, §4.3.1] (docs/references.md entry 23, doi
10.1021/acs.jctc.4c00994; the sentence as the milestone description quotes
it) is prior art for the same observation: "Unlike the J scheme, we cannot
order our shell pairs based on the magnitude of the density matrix." So
K's density factor is the bra- and ket-independent `Dm`, and the kets are
sorted descending by `I_bd` ALONE, with a STABLE sort. That is what makes
the sorted `break` exact: for one bra record `I_ac >= 0` and `Dm >= 0` are
fixed, and the rounded `(I_ac * x) * Dm` is non-decreasing in `x`, so every
ket after the first rejected one is rejected too. Sorting by the J key
misapplied -- `I_bd * max|D_bd|`, the ket pair's own density block, J's
`Dmax` shape -- orders the kets by something the screen does not test, and
its `break` skips kets the `continue` keeps (a test shows it on real
bounds). The break's precondition is the Schwarz-only order. (A tighter
density factor that stays bra-side, the max over `D`'s `c` rows, would keep
the break exact too; it is not taken, and [Fock]'s density-decay bound is
out of this milestone's scope.)

  `exchange_matrix(tau=)`, `exchange_matrix_reference(tau=)`  the unsorted
      `continue`, inside the quartet contraction; the two sweeps skip the
      same set, the product being symmetric under the bra-ket swap.
  `exchange_matrix_screened(tau=, order=)`  the unique sweep per bra record
      over its kets: `order="sorted"` the kernel's `break`, `"unsorted"` the
      `continue`, returning `KScreened`. Each element is accumulated with
      `math.fsum` over its per-record-pair terms, so the two orders give the
      SAME `K` to 0 when they skip the same set -- a statement about WHICH
      terms were summed, not their order (#88's `coulomb_matrix_screened`).

AT `tau = 0.0` NOTHING IS SKIPPED AND NO BOUND IS EVALUATED, in every entry
point: the branch is not taken, no `I` is formed, and the sorted form keeps
the unsorted order (there is no key without a bound, and `fsum` makes the
order immaterial). So a comparison at `tau = 0` cannot have an error hidden
by screening. `exchange_matrix_and_scale` is unscreened, and the `k=` seam
with `tau > 0` is REFUSED: screening is not a hook's business.

`KScreened.scale` is the stated error scale, independent of `tau`: per
element `(m, n)`, the number of ORDERED record pairs whose block covers it,
`R(shell of m) * R(shell of n)` with `R(x) = sum_c n_records(x, c)`. Each
skipped record pair, with its bra-ket partner, removes from an element at
most one term per ordered record pair, each bounded by `I_ac I_bd Dm <
tau`, so `|K(tau) - K(0)| <= tau * scale` element by element.

THE `K` SEAM, `k=` (the shape #151 set for `dq=` / `jab=`). `exchange_matrix`
takes an optional `k(bra, ket, dcd) -> block`, called once per unique
quadruple in sweep order, with `bra = epair(sa, sc)` and `ket = epair(sb,
sd)` (`ao.j.EPair`: the angular momenta, the centres, and per primitive pair
`(P, zeta, s)`), `dcd` the density block `D_cd` over `cartesians(lc) x
cartesians(ld)`, c outer, and returning `X` as `n_a` rows of `n_b` values.
The quadruple list, the density's extraction and symmetry check, the
weights and the scatter stay this module's, so a disagreement through `k=`
localises to the hook. `k_quartet` with its `pq` bound is the host path
through the seam (`functools.partial(k_quartet, pq=pq)`), and with it
`exchange_matrix` is `==` to itself without it. The seam is unscreened:
`k` with `tau > 0` is refused.

MEMOISE NOTHING across calls. Pure Python floats; `D` is anything
indexable as `D[m][n]`, and `K` is a tuple of rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from felizjueves.ao.eri import eri_block
from felizjueves.ao.hermite import e_block, hermite_order
from felizjueves.ao.j import EPair, _density, _tau, epair
from felizjueves.ao.shell import Basis, cartesians

Vec = tuple[float, float, float]
Ang = tuple[int, int, int]
PQ = Callable[[Ang, Ang, Vec, float], float]
Matrix = tuple[tuple[float, ...], ...]
Block = list[list[float]]
KQ = Callable[[EPair, EPair, Sequence[float]], Sequence[Sequence[float]]]
Skip = Callable[[int, int], bool]

_ORIGIN = (0.0, 0.0, 0.0)


def _hermite_bounds_pq(zeta: float, L: int, *, pq: PQ) -> list[float]:
    """`I_p = sqrt([p|p](PQ = 0, theta = sqrt(zeta/2)))` over
    `hermite_order(L)`: `ao.pairs.hermite_bound` through the `[p|q]`
    evaluator, as `ao.j.pair_bound_pq` forms it. Raises on a negative
    radicand, never `abs`es it."""
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
    return out


def record_bounds(pair: EPair, *, pq: PQ) -> tuple[float, ...]:
    """K's per-record Schwarz scalar, one per record of `pair` (module
    docstring, "THE SCREEN"): `I = max_{i in a} sum_{k in c} sum_p
    |E^{a_i c_k}_p| I_p(zeta)`, the dressed `E` of `pair = epair(sa, sc)` --
    rows over the FIRST shell (the output index), summed over the second."""
    out = []
    L = pair.la + pair.lb
    cb = cartesians(pair.lb)
    for rec in pair.recs:
        ips = _hermite_bounds_pq(rec[1], L, pq=pq)
        rows = []
        for a in cartesians(pair.la):
            terms = []
            for c in cb:
                e = _dressed(rec, a, c, pair.A, pair.B)
                terms.extend(abs(x) * y for x, y in zip(e, ips, strict=True))
            rows.append(math.fsum(terms))
        out.append(max(rows))
    return tuple(out)


def _dm(rows: list[list[float]]) -> float:
    """`Dm = max_{mn} |D_mn|`, the screen's density factor."""
    return max(abs(x) for r in rows for x in r)


def _skip(ib: Sequence[float], ik: Sequence[float], dm: float,
          tau: float) -> Skip:
    """The screen's test on record indices: `(I_ac * I_bd) * Dm < tau`."""
    return lambda i, j: (ib[i] * ik[j]) * dm < tau


def _dressed(rec: tuple[Vec, float, float], x: Ang, y: Ang, X: Vec,
             Y: Vec) -> list[float]:
    """One dressed `E` row, `s * E^{xy}`, as `pairs.dress` forms it."""
    P, zeta, s = rec
    return [s * e for e in e_block(x, y, zeta, P, X, Y)]


def _k_quartet(bra: EPair, ket: EPair, dcd: Sequence[float], *, pq: PQ,
               want_scale: bool,
               skip: Skip | None = None) -> tuple[Block, Block]:
    """The staged contraction of one shell quartet, and its |.| twin;
    `skip(i, j)` true drops bra record `i` against ket record `j` (the
    unsorted `continue`)."""
    ca, cc = cartesians(bra.la), cartesians(bra.lb)
    cb, cd = cartesians(ket.la), cartesians(ket.lb)
    na, nc, nb, nd = len(ca), len(cc), len(cb), len(cd)
    if len(dcd) != nc * nd:
        raise ValueError(f"D_cd has {len(dcd)} entries, not {nc} x {nd}")
    ps = hermite_order(bra.la + bra.lb)
    qs = hermite_order(ket.la + ket.lb)
    K = [[0.0] * nb for _ in range(na)]
    S = [[0.0] * nb for _ in range(na)]
    for ibr, brec in enumerate(bra.recs):  # one bra pair record (a, c)
        P, zeta, _ = brec
        e_ac = [_dressed(brec, a, c, bra.A, bra.B) for a in ca for c in cc]
        for ikr, krec in enumerate(ket.recs):  # one ket pair record (b, d)
            if skip is not None and skip(ibr, ikr):
                continue
            Q, eta, _ = krec
            theta = math.sqrt(zeta * eta / (zeta + eta))
            PQv = (P[0] - Q[0], P[1] - Q[1], P[2] - Q[2])
            m = [[pq(p, q, PQv, theta) for q in qs] for p in ps]
            for ib, b in enumerate(cb):
                for id_, d in enumerate(cd):  # the (b, d) loop
                    e_bd = _dressed(krec, b, d, ket.A, ket.B)
                    # stage 1: one column I[p], rebuilt per (b, d) row
                    col = []
                    col_abs = []
                    for row in m:
                        acc = 0.0
                        acc_abs = 0.0
                        for v, e in zip(row, e_bd, strict=True):
                            acc += v * e
                            if want_scale:
                                acc_abs += abs(v) * abs(e)
                        col.append(acc)
                        col_abs.append(acc_abs)
                    # stages 2 and 3, fused: V(a, c) digested as it is formed
                    for ia in range(na):
                        for ic in range(nc):
                            e = e_ac[ia * nc + ic]
                            v = 0.0
                            for x, y in zip(e, col, strict=True):
                                v += x * y
                            dv = dcd[ic * nd + id_]
                            K[ia][ib] += dv * v
                            if want_scale:
                                va = 0.0
                                for x, y in zip(e, col_abs, strict=True):
                                    va += abs(x) * y
                                S[ia][ib] += abs(dv) * va
    return K, S


def k_quartet(bra_pair: EPair, ket_pair: EPair, D_cd: Sequence[float], *,
              pq: PQ) -> list[list[float]]:
    """`X[a][b] = sum_{cd} D_cd (ac|bd)` of one shell quartet by the staged
    contraction (module docstring): `bra_pair = epair(sa, sc)`, `ket_pair =
    epair(sb, sd)`, `D_cd` flat over `cartesians(lc) x cartesians(ld)`, c
    outer. Rows over `cartesians(la)`, columns over `cartesians(lb)`."""
    return _k_quartet(bra_pair, ket_pair, [float(x) for x in D_cd], pq=pq,
                      want_scale=False)[0]


def _dcd(basis: Basis, rows: list[list[float]], c: int, d: int) -> list[float]:
    sc, sd = basis.shells[c], basis.shells[d]
    oc, od = basis.offsets[c], basis.offsets[d]
    return [rows[oc + i][od + j] for i in range(sc.n_cart)
            for j in range(sd.n_cart)]


def _add(K: list[list[float]], blk: Sequence[Sequence[float]], oa: int,
         ob: int, *, transpose: bool) -> None:
    for i, row in enumerate(blk):
        for j, v in enumerate(row):
            if transpose:
                K[ob + j][oa + i] += v
            else:
                K[oa + i][ob + j] += v


def _quadruples(n_sh: int, *, ordered: bool):
    """`(a, b, c, d)` -- every ordered one, or one per bra-ket orbit."""
    for a in range(n_sh):
        for b in range(n_sh if ordered else a + 1):
            for c in range(n_sh):
                for d in range(n_sh):
                    if not ordered and a == b and c < d:
                        continue  # the partner (a, a, d, c) is visited
                    yield a, b, c, d


def _sweep(basis: Basis, D: object, *, pq: PQ, ordered: bool,
           want_scale: bool, k: KQ | None = None,
           tau: float = 0.0) -> tuple[Matrix, Matrix]:
    rows = _density(basis, D)
    tau = _tau(tau)
    if k is not None and tau > 0.0:
        raise ValueError(f"tau = {tau!r}: the k seam is unscreened; only "
                         "tau = 0 is accepted with k (ao/k.py's module "
                         "docstring)")
    shells, off = basis.shells, basis.offsets
    n_sh = len(shells)
    eps = {(x, y): epair(shells[x], shells[y])
           for x in range(n_sh) for y in range(n_sh)}
    # the bound, only when it can fire: at tau = 0 none is evaluated
    bnd = ({xy: record_bounds(ep, pq=pq) for xy, ep in eps.items()}
           if tau > 0.0 else None)
    dm = _dm(rows) if tau > 0.0 else 0.0
    n = basis.n_ao
    K = [[0.0] * n for _ in range(n)]
    S = [[0.0] * n for _ in range(n)]
    for a, b, c, d in _quadruples(n_sh, ordered=ordered):
        dcd = _dcd(basis, rows, c, d)
        if k is None:
            skip = (None if bnd is None
                    else _skip(bnd[a, c], bnd[b, d], dm, tau))
            X, A = _k_quartet(eps[a, c], eps[b, d], dcd, pq=pq,
                              want_scale=want_scale, skip=skip)
        else:
            got = k(eps[a, c], eps[b, d], dcd)
            na, nb = shells[a].n_cart, shells[b].n_cart
            if len(got) != na or any(len(r) != nb for r in got):
                raise ValueError(
                    f"k returned the wrong shape for quartet "
                    f"{(shells[a].l, shells[c].l, shells[b].l, shells[d].l)}")
            X, A = [[float(v) for v in r] for r in got], None
        oa, ob = off[a], off[b]
        _add(K, X, oa, ob, transpose=False)
        if want_scale:
            _add(S, A, oa, ob, transpose=False)
        if ordered or (a == b and c == d):
            continue  # weight 1, once: ordered, or a fixed point
        # the partner (b, a, d, c): X^T, to K[b][a] -- or, when a == b, to
        # the same block, making it X + X^T (never 2 X)
        _add(K, X, oa, ob, transpose=True)
        if want_scale:
            _add(S, A, oa, ob, transpose=True)
    return tuple(tuple(r) for r in K), tuple(tuple(r) for r in S)


def exchange_matrix(basis: Basis, D: object, *, pq: PQ, tau: float = 0.0,
                    k: KQ | None = None) -> Matrix:
    """`K_mn` by one quadruple per bra-ket orbit, with K's weights (module
    docstring); `k` replaces the quartet contraction (the seam). `tau > 0`
    screens as the unsorted `continue` ("THE SCREEN"); not with `k`."""
    return _sweep(basis, D, pq=pq, ordered=False, want_scale=False, k=k,
                  tau=tau)[0]


def exchange_matrix_reference(basis: Basis, D: object, *, pq: PQ,
                              tau: float = 0.0) -> Matrix:
    """`K_mn` by EVERY ordered quadruple, weight 1, no symmetry. The internal
    check on `exchange_matrix`'s weights; keep it. `tau > 0` screens as the
    unsorted `continue`, the same skipped set as `exchange_matrix`'s."""
    return _sweep(basis, D, pq=pq, ordered=True, want_scale=False,
                  tau=tau)[0]


def exchange_matrix_and_scale(basis: Basis, D: object, *,
                              pq: PQ) -> tuple[Matrix, Matrix]:
    """`exchange_matrix` and its absolute-value twin `scale` (module
    docstring) -- the size of what was added, per element."""
    return _sweep(basis, D, pq=pq, ordered=False, want_scale=True)


def k_from_eris(basis: Basis, D: object, *, pq: PQ) -> Matrix:
    """`K_mn = sum_{ls} D_ls (ml|ns)` from `ao.eri.eri_block` of every ordered
    quadruple -- the full ERI block, contracted element by element."""
    rows = _density(basis, D)
    shells, off = basis.shells, basis.offsets
    n_sh = len(shells)
    n = basis.n_ao
    K = [[0.0] * n for _ in range(n)]
    for a, b, c, d in _quadruples(n_sh, ordered=True):
        sa, sb, sc, sd = shells[a], shells[b], shells[c], shells[d]
        blk = eri_block((sa, sc), (sb, sd), pq=pq)  # rows (a c), cols (b d)
        nc, nd = sc.n_cart, sd.n_cart
        for i in range(sa.n_cart):
            for j in range(sb.n_cart):
                acc = 0.0
                for kk in range(nc):
                    row = blk[i * nc + kk]
                    for ll in range(nd):
                        acc += rows[off[c] + kk][off[d] + ll] * row[j * nd + ll]
                K[off[a] + i][off[b] + j] += acc
    return tuple(tuple(r) for r in K)


# --- the screen as an error claim (#200): the sorted break ---------------


@dataclass(frozen=True)
class KScreened:
    """`exchange_matrix_screened`'s result, `ao.j.Screened`'s shape for K
    (module docstring, "THE SCREEN"): `K`, the `tau`-independent error scale,
    the count of (bra record, ket record) pairs of the unique sweep, total,
    skipped and examined (whose screen test was evaluated: 0 at `tau = 0`),
    and the largest and the `math.fsum` of every skipped bound
    `I_ac I_bd Dm`."""

    K: Matrix
    scale: Matrix
    n_pairs: int
    n_skipped: int
    n_examined: int
    max_skipped: float
    dropped: float

    @property
    def skip_fraction(self) -> float:
        return self.n_skipped / self.n_pairs


ORDERS = ("sorted", "unsorted")

KetKey = Callable[[int, int, float], float]  # (b, d, I_bd) -> sort key


def _schwarz_key(b: int, d: int, i_bd: float) -> float:
    """K's ket order: `I_bd` ALONE (module docstring)."""
    return i_bd


def _screened(basis: Basis, D: object, *, pq: PQ, tau: float, order: str,
              ket_key: KetKey = _schwarz_key) -> KScreened:
    """The unique sweep, per bra record over its kets in `order`. `ket_key`
    is the sort key: `_schwarz_key`, except where a test shows what another
    key does to the break (module docstring)."""
    if order not in ORDERS:
        raise ValueError(f"order = {order!r} is not one of {ORDERS}")
    rows = _density(basis, D)
    tau = _tau(tau)
    shells, off = basis.shells, basis.offsets
    n_sh = len(shells)
    eps = {(x, y): epair(shells[x], shells[y])
           for x in range(n_sh) for y in range(n_sh)}
    screen = tau > 0.0  # at tau = 0: no bound, no key, no sort, no skip
    bnd = ({xy: record_bounds(ep, pq=pq) for xy, ep in eps.items()}
           if screen else {})
    dm = _dm(rows) if screen else 0.0
    dcds = {(x, y): _dcd(basis, rows, x, y)
            for x in range(n_sh) for y in range(n_sh)}
    n = basis.n_ao
    terms: list[list[list[float]]] = [[[] for _ in range(n)] for _ in range(n)]
    n_pairs = n_skipped = n_examined = 0
    dropped: list[float] = []
    for a in range(n_sh):
        oa = off[a]
        for c in range(n_sh):
            bra = eps[a, c]
            kets = [(b, d, j) for b in range(a + 1) for d in range(n_sh)
                    if not (b == a and d > c)  # its partner is visited
                    for j in range(len(eps[b, d].recs))]
            if screen and order == "sorted":  # descending, stable
                kets.sort(key=lambda t: ket_key(t[0], t[1],
                                                bnd[t[0], t[1]][t[2]]),
                          reverse=True)
            n_ket = len(kets)
            for i, brec in enumerate(bra.recs):
                one_bra = replace(bra, recs=(brec,))
                n_pairs += n_ket
                for idx, (b, d, j) in enumerate(kets):
                    if screen:
                        n_examined += 1
                        if (bnd[a, c][i] * bnd[b, d][j]) * dm < tau:
                            if order == "sorted":  # every later ket too
                                rest = kets[idx:]
                                n_skipped += len(rest)
                                dropped.extend(
                                    (bnd[a, c][i] * bnd[y, z][w]) * dm
                                    for y, z, w in rest)
                                break
                            n_skipped += 1
                            dropped.append((bnd[a, c][i] * bnd[b, d][j]) * dm)
                            continue
                    ket = eps[b, d]
                    X = _k_quartet(one_bra, replace(ket, recs=(ket.recs[j],)),
                                   dcds[c, d], pq=pq, want_scale=False)[0]
                    ob = off[b]
                    mirror = not (a == b and c == d)  # else a fixed point
                    for ii, row in enumerate(X):
                        for jj, v in enumerate(row):
                            terms[oa + ii][ob + jj].append(v)
                            if mirror:  # the partner: X^T (X + X^T if a == b)
                                terms[ob + jj][oa + ii].append(v)
    K = tuple(tuple(math.fsum(t) for t in r) for r in terms)
    # the scale: ordered record pairs covering (m, n) = R(sh m) * R(sh n)
    R = [sum(len(eps[x, y].recs) for y in range(n_sh)) for x in range(n_sh)]
    sh_of = [x for x in range(n_sh) for _ in range(shells[x].n_cart)]
    S = tuple(tuple(float(R[sh_of[m]] * R[sh_of[v]]) for v in range(n))
              for m in range(n))
    return KScreened(K, S, n_pairs, n_skipped, n_examined,
                     max(dropped, default=0.0), math.fsum(dropped))


def exchange_matrix_screened(basis: Basis, D: object, *, pq: PQ, tau: float,
                             order: str = "sorted") -> KScreened:
    """`K_mn` by the unique sweep, screened at `tau` as the kernel's sorted
    `break` (`order="sorted"`: kets descending by `I_bd` alone, stable) or as
    the unsorted `continue` (`order="unsorted"`); module docstring, "THE
    SCREEN". At `tau = 0` no bound is evaluated and nothing is skipped."""
    return _screened(basis, D, pq=pq, tau=tau, order=order)
