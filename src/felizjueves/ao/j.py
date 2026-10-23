"""The Coulomb matrix `J_mn`, through the J-engine split -- by two sweeps.

    coulomb_matrix(basis, D, *, pq, tau=0.0, jp=None, dq=None, jab=None)
                                                        -> J   (unique pairs)
    coulomb_matrix_reference(basis, D, *, pq, tau=0.0)  -> J   (all ordered)
    coulomb_matrix_and_scale(basis, D, *, pq)           -> (J, scale)
    coulomb_matrix_screened(basis, D, *, pq, tau, order="sorted")
                                                        -> Screened (#88)
    jp_from_pq(pq), dq_from_host, jab_from_host         the seams' references

The helpers other builds share -- the density and `tau` checks, `EPair` /
`epair`, the ket records (`Ket`, `make_kets`) and `pair_bound_pq`, which
`make_kets` evaluates -- live in `ao/density.py` (#503) and are imported
back here; their contracts are still stated below.

`J_mn = sum_{ls} (mn|ls) D_ls`, never by forming `(mn|ls)`. For a bra
primitive pair (index record) of an ordered shell pair `(a, b)` and a ket
primitive pair `k` of `(c, d)` ([JEngine, v2, eqs. 20-22]; j_engine.md, "Why
this is a J engine and not an ERI engine"):

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
(`Ket.bound`, exactly the factor the screen multiplies by `I_bra`), with a
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

`Screened` is THE screening record, shared with `launch.walk` (#142, the
launch oracle) so one type serves both paths: `n_pairs` considered,
`n_skipped` skipped, `n_examined` the kets whose screen test was evaluated
(the break's witness: `n_pairs` under the `continue`, fewer under the
`break`), `max_skipped` the largest skipped bound `I_bra * I_ket * Dmax`
(`0.0` when nothing is skipped) and `dropped` the `math.fsum` of every
skipped bound -- order-independent, so the break, the continue and the
oracle's per-class walk record the same value for the same skipped set.

`Screened.scale` is the error claim's stated scale, independent of `tau`:
per element, `n_ket * sum_{bra records} sum_p |E^{ab}_p|`. Each skipped pair
has `|sum_q [p|q] D_q| <= I_p I_ket Dmax <= I_bra I_ket Dmax < tau` for every
`p` (ao/pairs.py), so it moves `J_ab` by at most `tau sum_p |E^{ab}_p|`, and
a bra record skips at most `n_ket` kets: `|J(tau) - J(0)| <= tau * scale`.

THE `J_p` SEAM, `jp=` (#87). `coulomb_matrix` takes an optional `jp`, a
`JP` evaluator of the ONLY shell-quartet-level step, called once per (bra
shell pair, ket class) with the bra pair's angular momenta `(la, lb)`, the
ket class `(lc, ld)`, the bra records as `(P, zeta)` and the ket records
of every ket shell pair of that class as `(Q, eta, D_q)` -- `D_q` already
weighted and dressed, in `hermite_order(lc + ld)` -- and returning `J_p`
per bra record in `hermite_order(la + lb)`. The ket pre-contraction, the
bra transform and the scatter are this module's either way (unless their
own seams, below, replace them), so a `J` built
through a `jp` that disagrees localises the disagreement to the `jp`.
Without `jp` the `pq` path above runs unchanged; `jp_from_pq(pq)` is the
same contraction through the seam (a test pins the two equal). The seam is
UNSCREENED: `jp` with `tau > 0` is refused -- screening is the kernel's
sorted `break` and the `continue` above, not a `jp`'s business.

THE `D_q` AND `J_ab` SEAMS, `dq=` and `jab=` (#151). The same discipline as
`jp=`, one seam per step of the chain, so a generated pre-contraction and a
generated bra transform are each substituted alone and a disagreement
localises to one of them (j_engine.md, "The J chain as device code"). Both
take an `EPair` (`epair(sa, sb)`): one shell pair's angular momenta
`(la, lb)`, centres `(A, B)` and, per primitive pair in `primitive_pairs`
order (i outer), the record `(P, zeta, s)` -- `P` and `zeta` exactly as
`PrimitivePair` has them, and `s = c_a c_b (pi/zeta)^{3/2}` obtained from
`pairs.dress` itself (as `dress((1.0,), ...)[0]`, the scale times 1.0,
exactly), so the dressing stays one function's and a hook receives it as a
scalar.

  `dq(ket, dcd, w) -> D_q per ket record`, called once per unique ket shell
      pair `(c, d)`, `c >= d`, in sweep order, with `dcd` the density block
      `D_cd` over `cartesians(lc) x cartesians(ld)`, c outer, and `w` the
      weight (2.0 off the diagonal, 1.0 on it) -- decided HERE, never by the
      hook. It returns one row per record, each over `hermite_order(lc+ld)`:
      `D_q = w sum_cd D_cd s E^{cd}_q`, `E` undressed. The module keeps the
      pair list, the weight, the density's extraction and symmetry check,
      `Q` and `eta`, and everything downstream.
  `jab(bra, jps, (off_a, off_b), J) -> None`, called once per unique bra
      shell pair `(a, b)`, `a >= b`, in sweep order, with `jps` the `J_p`
      row per bra record over `hermite_order(la+lb)` (from `pq`, or from
      `jp`), the two shells' AO offsets, and `J` the module's own n x n list
      of rows, ZEROED before the first call. The hook ADDS
      `J_ab = sum_{records} sum_p s E^{ab}_p J_p` into `J`: off the diagonal
      (`off_a != off_b`) at `J[off_a+i][off_b+j]` and at its mirror
      `J[off_b+j][off_a+i]`; on the diagonal (`off_a == off_b`) only the
      block's lower triangle `i >= j`, each at `(i, j)` and, for `i != j`,
      at `(j, i)` -- so every element of `J` is written by exactly one call,
      once. That is not a convention chosen for a kernel: it is what the
      host scatter's last write leaves on a diagonal block (row-major, so
      element `(x, y)` ends holding `J_ab[max(x, y), min(x, y)]`), and so
      the form that reproduces the host bit for bit. The module keeps the
      pair list, the `J_p` and the zeroing of `J`.

`dq_from_host` and `jab_from_host` are each step's host arithmetic through
its seam, in the host's own operation order -- the analogues of
`jp_from_pq` -- and with them `coulomb_matrix` is `==` to itself without
them (test/test_ao_j.py), on the `pq` path and on the `jp` path. Without
either hook every path runs exactly the operations it ran before they
existed. Both seams are UNSCREENED, like `jp=`: either with `tau > 0` is
refused.

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

from felizjueves.ao.density import (
    PQ,
    Ang,
    EPair,
    Ket,
    Matrix,
    Vec,
    checked_density,
    checked_tau,
    epair,
    make_kets,
    pair_bound_pq,
)
from felizjueves.ao.hermite import e_block, hermite_order
from felizjueves.ao.pairs import PrimitivePair, primitive_pairs
from felizjueves.ao.shell import Basis, cartesians

BraRec = tuple[Vec, float]  # (P, zeta)
KetRec = tuple[Vec, float, Sequence[float]]  # (Q, eta, D_q)
JP = Callable[[tuple[int, int], tuple[int, int], Sequence[BraRec],
               Sequence[KetRec]], Sequence[Sequence[float]]]

DQ = Callable[[EPair, Sequence[float], float], Sequence[Sequence[float]]]
JAB = Callable[[EPair, Sequence[Sequence[float]], tuple[int, int],
                list[list[float]]], object]


def _record_jp(bra: PrimitivePair, kets: Sequence[Ket], *, pq: PQ,
               tau: float, want_scale: bool):
    """`J_p` (and its |.| twin) of one bra record, over `kets` in order."""
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
    return jp, jp_abs


def _bra_block(bra_recs: Sequence[PrimitivePair], kets: Sequence[Ket], *,
               pq: PQ, tau: float, want_scale: bool):
    """`J_ab` (rows `ab`, a outer) and its |.| twin, for one bra shell pair."""
    nrow = len(bra_recs[0].block)
    jab = [0.0] * nrow
    jab_abs = [0.0] * nrow
    for bra in bra_recs:
        jp, jp_abs = _record_jp(bra, kets, pq=pq, tau=tau,
                                want_scale=want_scale)
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


def jp_from_pq(pq: PQ) -> JP:
    """The `pq` contraction as a `JP`: `J_p = sum_k sum_q [p|q] D_q`."""

    def jp(bra_am: tuple[int, int], ket_am: tuple[int, int],
           bras: Sequence[BraRec], kets: Sequence[KetRec]) -> list[list[float]]:
        ps, qs = hermite_order(sum(bra_am)), hermite_order(sum(ket_am))
        out = []
        for P, zeta in bras:
            row = [0.0] * len(ps)
            for Q, eta, dq in kets:
                theta = math.sqrt(zeta * eta / (zeta + eta))
                PQv = (P[0] - Q[0], P[1] - Q[1], P[2] - Q[2])
                for ip, p in enumerate(ps):
                    acc = 0.0
                    for q, d in zip(qs, dq, strict=True):
                        acc += pq(p, q, PQv, theta) * d
                    row[ip] += acc
            out.append(row)
        return out

    return jp


def _no_pq(p: Ang, q: Ang, PQv: Vec, theta: float) -> float:
    raise AssertionError("no [p|q] is evaluated building kets at tau = 0")


def cart_pairs(la: int, lb: int) -> list[tuple[Ang, Ang]]:
    """The Cartesian pairs of `(la, lb)`, `a` outer: a block's row order."""
    return [(a, b) for a in cartesians(la) for b in cartesians(lb)]


def dq_from_host(ket: EPair, dcd: Sequence[float],
                 w: float) -> list[tuple[float, ...]]:
    """The host pre-contraction as a `DQ`: `make_kets`' arithmetic in its
    order, the dressed `E` re-formed as `s * E` exactly as `pairs.dress` forms it."""
    carts = cart_pairs(ket.la, ket.lb)
    n_q = len(hermite_order(ket.la + ket.lb))
    out = []
    for P, zeta, s in ket.recs:
        dq = [0.0] * n_q
        for x, (c, d) in zip(dcd, carts, strict=True):
            if x:
                for t, e in enumerate(e_block(c, d, zeta, P, ket.A, ket.B)):
                    dq[t] += x * (s * e)
        out.append(tuple(w * v for v in dq))
    return out


def jab_from_host(bra: EPair, jps: Sequence[Sequence[float]],
                  offsets: tuple[int, int], J: list[list[float]]) -> None:
    """The host bra transform and scatter as a `JAB`: the transform in the
    sweeps' order, then the one-write-per-element scatter (module
    docstring), added into the zeroed `J`."""
    carts = cart_pairs(bra.la, bra.lb)
    jab = [0.0] * len(carts)
    for (P, zeta, s), jp_row in zip(bra.recs, jps, strict=True):
        for r, (a, b) in enumerate(carts):
            acc = 0.0
            for e, v in zip(e_block(a, b, zeta, P, bra.A, bra.B), jp_row,
                            strict=True):
                acc += (s * e) * v
            jab[r] += acc
    oa, ob = offsets
    nb = len(cartesians(bra.lb))
    for i in range(len(cartesians(bra.la))):
        for j in range(nb):
            if oa == ob and i < j:
                continue  # the diagonal pair: its lower triangle only
            v = jab[i * nb + j]
            J[oa + i][ob + j] += v
            if oa + i != ob + j:
                J[ob + j][oa + i] += v


def _ket_records(basis: Basis, rows: list[list[float]], c: int, d: int,
                 w: float, dq: DQ | None) -> list[Ket]:
    """One ket shell pair's records: `make_kets` without a hook; with one, `D_q`
    from `dq`, and `Q`, `eta` and the Hermite indices still this module's."""
    if dq is None:
        return make_kets(basis, rows, [(c, d, w)], pq=_no_pq, tau=0.0)
    shells, off = basis.shells, basis.offsets
    sc, sd = shells[c], shells[d]
    recs = primitive_pairs(sc, sd)
    dcd = [rows[off[c] + i][off[d] + j]
           for i in range(sc.n_cart) for j in range(sd.n_cart)]
    qs = hermite_order(sc.l + sd.l)
    got = dq(epair(sc, sd, recs), dcd, w)
    if len(got) != len(recs) or any(len(g) != len(qs) for g in got):
        raise ValueError(f"dq returned the wrong shape for ket "
                         f"{(sc.l, sd.l)}")
    zero = (0.0,) * len(qs)
    return [Ket(k.P, k.zeta, qs, tuple(float(v) for v in g), zero, 0.0)
            for k, g in zip(recs, got, strict=True)]


def _seam_sweep(basis: Basis, D: object, *, pq: PQ, jp: JP | None,
                dq: DQ | None, jab: JAB | None) -> Matrix:
    """The unique sweep through the seams: the same kets, weights, bra
    transform and mirror as `_sweep(ordered=False)`. With `jp`, the kets are
    grouped by class and `J_p` comes from it; without, `J_p` is `pq`'s, per
    record over every ket in sweep order, exactly as `_bra_block` forms it."""
    rows = checked_density(basis, D)
    shells, off = basis.shells, basis.offsets
    n_sh = len(shells)
    pairs = [(a, b, 1.0 if a == b else 2.0)
             for a in range(n_sh) for b in range(a + 1)]
    all_kets: list[Ket] = []
    groups: dict[tuple[int, int], list[KetRec]] = {}
    for c, d, w in pairs:
        ks = _ket_records(basis, rows, c, d, w, dq)
        all_kets.extend(ks)
        groups.setdefault((shells[c].l, shells[d].l), []).extend(
            (k.Q, k.eta, k.dq) for k in ks)
    n = basis.n_ao
    J = [[0.0] * n for _ in range(n)]
    for a, b, _ in pairs:
        sa, sb = shells[a], shells[b]
        recs = primitive_pairs(sa, sb)
        n_p = len(hermite_order(sa.l + sb.l))
        if jp is None:
            jps = [_record_jp(r, all_kets, pq=pq, tau=0.0,
                              want_scale=False)[0] for r in recs]
        else:
            jps = [[0.0] * n_p for _ in recs]
            bras = [(r.P, r.zeta) for r in recs]
            for ket_am, kets in groups.items():
                got = jp((sa.l, sb.l), ket_am, bras, kets)
                if len(got) != len(recs) or any(len(g) != n_p for g in got):
                    raise ValueError(f"jp returned the wrong shape for bra "
                                     f"{(sa.l, sb.l)}, ket {ket_am}")
                for acc, g in zip(jps, got, strict=True):
                    for i, v in enumerate(g):
                        acc[i] += float(v)
        if jab is not None:
            jab(epair(sa, sb, recs), jps, (off[a], off[b]), J)
            continue
        jab_blk = [0.0] * len(recs[0].block)
        for rec, jp_row in zip(recs, jps, strict=True):
            for r, row in enumerate(rec.block):
                s = 0.0
                for e, v in zip(row, jp_row, strict=True):
                    s += e * v
                jab_blk[r] += s
        for i in range(sa.n_cart):
            for j in range(sb.n_cart):
                v = jab_blk[i * sb.n_cart + j]
                J[off[a] + i][off[b] + j] = v
                J[off[b] + j][off[a] + i] = v
    return tuple(tuple(r) for r in J)


def _sweep(basis: Basis, D: object, *, pq: PQ, tau: float, ordered: bool,
           want_scale: bool) -> tuple[Matrix, Matrix]:
    rows = checked_density(basis, D)
    tau = checked_tau(tau)
    n_sh = len(basis.shells)
    if ordered:
        pairs = [(a, b, 1.0) for a in range(n_sh) for b in range(n_sh)]
    else:
        pairs = [(a, b, 1.0 if a == b else 2.0)
                 for a in range(n_sh) for b in range(a + 1)]
    kets = make_kets(basis, rows, pairs, pq=pq, tau=tau)
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
                   tau: float = 0.0, jp: JP | None = None,
                   dq: DQ | None = None, jab: JAB | None = None) -> Matrix:
    """`J_mn` by unique bra and ket shell pairs, ket off-diagonal weight 2;
    `jp`, `dq` and `jab` each replace one step (the seams, above). With none
    of them it is the `pq` path, unchanged."""
    if jp is None and dq is None and jab is None:
        return _sweep(basis, D, pq=pq, tau=tau, ordered=False,
                      want_scale=False)[0]
    if checked_tau(tau) != 0.0:
        given = ", ".join(n for n, h in (("jp", jp), ("dq", dq),
                                          ("jab", jab)) if h is not None)
        raise ValueError(f"tau = {tau!r}: the {given} seam is unscreened; "
                         f"only tau = 0 is accepted with {given}")
    return _seam_sweep(basis, D, pq=pq, jp=jp, dq=dq, jab=jab)


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
    """`coulomb_matrix_screened`'s and `launch.walk`'s result (module
    docstring, "THE SORTED `break`"): `J`, the `tau`-independent error scale,
    the count of (bra record, ket record) index pairs, total, skipped and
    examined, and the largest and the total skipped bound."""

    J: Matrix
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


def _sorted_kets(kets: Sequence[Ket]) -> list[Ket]:
    """Descending by `bound` (= `I_ket * Dmax`), STABLE: `sorted` is stable,
    and `reverse=True` keeps equal keys in their original order."""
    return sorted(kets, key=lambda k: k.bound, reverse=True)


def _bra_block_screened(bra_recs: Sequence[PrimitivePair],
                        kets: Sequence[Ket], *, pq: PQ, tau: float,
                        brk: bool):
    nrow = len(bra_recs[0].block)
    jab = [0.0] * nrow
    sab = [0.0] * nrow
    skipped = examined = 0
    dropped: list[float] = []  # the skipped bounds I_bra * I_ket * Dmax
    n_ket = len(kets)
    for bra in bra_recs:
        ps = hermite_order(bra.L)
        zeta, P = bra.zeta, bra.P
        terms: list[list[float]] = [[] for _ in ps]
        ibra = pair_bound_pq(zeta, bra.L, pq=pq)
        for i, k in enumerate(kets):
            examined += 1
            if ibra * k.bound < tau:
                if brk:  # sorted: every later ket is under tau too
                    skipped += n_ket - i
                    dropped.extend(ibra * x.bound for x in kets[i:])
                    break
                skipped += 1
                dropped.append(ibra * k.bound)
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
    return jab, sab, skipped, examined, dropped


def coulomb_matrix_screened(basis: Basis, D: object, *, pq: PQ, tau: float,
                            order: str = "sorted") -> Screened:
    """`J_mn` by the unique sweep, screened at `tau` as the kernel's sorted
    `break` (`order="sorted"`) or as the unsorted `continue`
    (`order="unsorted"`); module docstring, "THE SORTED `break`". The bound
    is evaluated at every `tau`, 0 included (it is the sort key); at
    `tau = 0` nothing is skipped, since the product is >= 0."""
    if order not in ORDERS:
        raise ValueError(f"order = {order!r} is not one of {ORDERS}")
    rows = checked_density(basis, D)
    tau = checked_tau(tau)
    n_sh = len(basis.shells)
    pairs = [(a, b, 1.0 if a == b else 2.0)
             for a in range(n_sh) for b in range(a + 1)]
    # `make_kets` evaluates the bound only when its tau is > 0; this one is
    # passed to switch it on and is used for nothing else.
    kets = make_kets(basis, rows, pairs, pq=pq, tau=1.0)
    if order == "sorted":
        kets = _sorted_kets(kets)
    n = basis.n_ao
    J = [[0.0] * n for _ in range(n)]
    S = [[0.0] * n for _ in range(n)]
    shells, off = basis.shells, basis.offsets
    n_pairs = n_skipped = n_examined = 0
    dropped: list[float] = []
    for a, b, _ in pairs:
        sa, sb = shells[a], shells[b]
        bras = primitive_pairs(sa, sb)
        jab, sab, sk, ex, dr = _bra_block_screened(
            bras, kets, pq=pq, tau=tau, brk=order == "sorted")
        n_pairs += len(bras) * len(kets)
        n_skipped += sk
        n_examined += ex
        dropped.extend(dr)
        for i in range(sa.n_cart):
            for j in range(sb.n_cart):
                r = i * sb.n_cart + j
                J[off[a] + i][off[b] + j] = J[off[b] + j][off[a] + i] = jab[r]
                S[off[a] + i][off[b] + j] = S[off[b] + j][off[a] + i] = sab[r]
    return Screened(tuple(tuple(r) for r in J), tuple(tuple(r) for r in S),
                    n_pairs, n_skipped, n_examined, max(dropped, default=0.0),
                    math.fsum(dropped))
