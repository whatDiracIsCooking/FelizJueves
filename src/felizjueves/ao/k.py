"""The exchange matrix `K_mn`, by the staged contraction -- by two sweeps.

    exchange_matrix(basis, D, *, pq, tau=0.0, k=None)   -> K   (unique)
    exchange_matrix_reference(basis, D, *, pq, tau=0.0) -> K   (all ordered)
    exchange_matrix_and_scale(basis, D, *, pq)          -> (K, scale)
    k_from_eris(basis, D, *, pq)                        -> K   via ao.eri
    k_quartet(bra_pair, ket_pair, D_cd, *, pq)          the seam's reference

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

THE SCREEN. `tau` is in every sweep's signature for #200 (K's Schwarz bound,
proved, and the sort by `I_bd` alone); it is not built here. `tau = 0.0` is
accepted and screens nothing; any `tau > 0` is REFUSED (`ValueError`) rather
than silently ignored.

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
`exchange_matrix` is `==` to itself without it. The seam is unscreened.

MEMOISE NOTHING across calls. Pure Python floats; `D` is anything
indexable as `D[m][n]`, and `K` is a tuple of rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

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


def _no_screen(tau: float) -> None:
    if _tau(tau) != 0.0:
        raise ValueError(
            f"tau = {tau!r}: K's screen is not built (#200); only tau = 0 is "
            "accepted (ao/k.py's module docstring)")


def _dressed(rec: tuple[Vec, float, float], x: Ang, y: Ang, X: Vec,
             Y: Vec) -> list[float]:
    """One dressed `E` row, `s * E^{xy}`, as `pairs.dress` forms it."""
    P, zeta, s = rec
    return [s * e for e in e_block(x, y, zeta, P, X, Y)]


def _k_quartet(bra: EPair, ket: EPair, dcd: Sequence[float], *, pq: PQ,
               want_scale: bool) -> tuple[Block, Block]:
    """The staged contraction of one shell quartet, and its |.| twin."""
    ca, cc = cartesians(bra.la), cartesians(bra.lb)
    cb, cd = cartesians(ket.la), cartesians(ket.lb)
    na, nc, nb, nd = len(ca), len(cc), len(cb), len(cd)
    if len(dcd) != nc * nd:
        raise ValueError(f"D_cd has {len(dcd)} entries, not {nc} x {nd}")
    ps = hermite_order(bra.la + bra.lb)
    qs = hermite_order(ket.la + ket.lb)
    K = [[0.0] * nb for _ in range(na)]
    S = [[0.0] * nb for _ in range(na)]
    for brec in bra.recs:  # one bra pair record (a, c)
        P, zeta, _ = brec
        e_ac = [_dressed(brec, a, c, bra.A, bra.B) for a in ca for c in cc]
        for krec in ket.recs:  # one ket pair record (b, d)
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
           want_scale: bool, k: KQ | None = None) -> tuple[Matrix, Matrix]:
    rows = _density(basis, D)
    shells, off = basis.shells, basis.offsets
    n_sh = len(shells)
    eps = {(x, y): epair(shells[x], shells[y])
           for x in range(n_sh) for y in range(n_sh)}
    n = basis.n_ao
    K = [[0.0] * n for _ in range(n)]
    S = [[0.0] * n for _ in range(n)]
    for a, b, c, d in _quadruples(n_sh, ordered=ordered):
        dcd = _dcd(basis, rows, c, d)
        if k is None:
            X, A = _k_quartet(eps[a, c], eps[b, d], dcd, pq=pq,
                              want_scale=want_scale)
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
    docstring); `k` replaces the quartet contraction (the seam)."""
    _no_screen(tau)
    return _sweep(basis, D, pq=pq, ordered=False, want_scale=False, k=k)[0]


def exchange_matrix_reference(basis: Basis, D: object, *, pq: PQ,
                              tau: float = 0.0) -> Matrix:
    """`K_mn` by EVERY ordered quadruple, weight 1, no symmetry. The internal
    check on `exchange_matrix`'s weights; keep it."""
    _no_screen(tau)
    return _sweep(basis, D, pq=pq, ordered=True, want_scale=False)[0]


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
