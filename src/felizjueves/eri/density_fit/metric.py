"""The two-center Coulomb metric `(P|Q)` of an auxiliary basis, host-side.

    aux_dress(c, *, alpha)                 -> s = c (pi/alpha)^{3/2}, the ONE place
    center_e(l, alpha)                     -> the undressed single-center E block
    aux_records(shell)                     -> one AuxRec per primitive
    metric_block_and_scale(sP, sQ, *, pq)  -> ((P|Q) block, its |.| twin)
    metric_block(sP, sQ, *, pq)            -> (P|Q) of one ordered aux shell pair
    metric_matrix(aux, *, pq)              -> (P|Q) over every aux function
    scale_matrix(aux, *, pq)               -> the scale of every element, placed alike
    metric_classes(aux)                    -> the staging, per class (lP, lQ)
    class_block(cls, k, *, pq)             -> staged pair k's block, from its records
    place_staged(cls, k, block, out)       -> the scatter, transposing a swap

`(P|Q) = INT INT P(r1) Q(r2) / r12` (Milestone 15, #412; `Target.METRIC`,
spec.py). Aux shells are ordinary contracted Cartesian `Shell`s
(`ao/shell.py`), loaded from a fixture by `test/oracle/basis.from_fixture`.

AN AUX PRIMITIVE IS A HERMITE "PAIR" OF ONE GAUSSIAN. McMurchie & Davidson's
`E` (docs/references.md, entry 46, eq. 2.20) of a primitive pair at
`zeta = alpha`, `P = A`, `X_PA = 0` and no second Gaussian (so no `K_AB`) is
the single-center expansion

    x^i exp(-alpha x^2) = sum_t E^i_t Lambda_t(x; alpha, A),
    E^{i+1}_t = (1/2alpha) E^i_{t-1} + (t+1) E^i_{t+1},   E^0_0 = 1,

`E^i_t = 0` whenever `i - t` is odd (`e_fragment.center_fragment`, #410, the
kernel's spelling of the same recurrence). Here it is `mmd.hermite.e_block`
at `(a, (0, 0, 0))` with `A = B = P`: the seed is `exp(alpha * 0 * 0) = 1`,
the `X_PA` term contributes `0 * E`, and `X_PB` is never read (`j = 0`), so
it is the single-center `E` exactly; test_e_fragment.py pins the fragment to
`e_axis` at `X_PA = 0` bitwise.

THE INTEGRAL. For aux primitives `i` of `P` (centre `A`) and `k` of `Q`
(centre `B`),

    (P|Q) = sum_{i in P} sum_{k in Q} s_i s_k
            sum_t sum_tau E^P_t(i) [t|tau] E^Q_tau(k),
    [t|tau] = (-1)^{|tau|} [t+tau]^(0)(A - B, theta),
    theta^2 = alpha_i alpha_k / (alpha_i + alpha_k),

with `[t|tau]` the INJECTED evaluator's (`pq(t, tau, PQ, theta)`, as in
`eri.eri`; `ao` imports neither `reference/` nor `plan/`), so the sign and
`(2/sqrt(pi)) theta` are the evaluator's, inside its `[0]^(m)`.

WHERE THE DRESSING GOES, decided once, as `ao.pairs.dress` decides it for a
primitive pair. `aux_dress` multiplies in, and nothing else here does,

    s = c (pi/alpha)^{3/2},

`c` the shell's coefficient. Then

    (2/sqrt(pi)) theta (pi/alpha_i)^{3/2} (pi/alpha_k)^{3/2}
        = 2 pi^{5/2} / (alpha_i alpha_k sqrt(alpha_i + alpha_k)),

McMurchie & Davidson's `lambda` for two Hermite Gaussians, `ao.pairs.dress`'s
split with a single Gaussian on each side (test_ao_metric.py asserts it).

THE NORMALISATION IS IN `c`, NOT A DRESSING. A `Shell`'s coefficients
multiply UNNORMALISED primitives (`ao/shell.py`), so the primitive
normalisation `N` -- written `s = c N (pi/alpha)^{3/2}` in #412 -- is already
in `c` (put there by `test/oracle/basis.py` from libcint's), and multiplying
`ao.shell.primitive_norm` in here as well would normalise twice. The aux
overlap from the same shells against the fixture's `int1e_ovlp` is the
witness (test_ao_metric.py, first). And `eri.eri` with an exponent-0 dummy
`s` shell is NOT this integral's route: its primitive normalisation goes to
0 with the exponent.

THE SCALE. `scale` is the same sum with every term replaced by its absolute
value (as `eri.eri`'s), so a tolerance relative to it is a statement about
rounding, not about how much the sum happened to cancel.

THE LAYOUT. A block's rows run over `cartesians(lP)`, columns over
`cartesians(lQ)` -- AO order on each side. `metric_matrix` places every aux
shell pair `i <= j` in basis order at `[off_i + a][off_j + b]` and its
mirror; a DIAGONAL shell pair writes its lower triangle `a >= b` only,
mirrored (`eri.schwarz`'s and `kernel_jab`'s rule), so the matrix is exactly
symmetric and `(P|Q)`, `(Q|P)` -- equal exactly, not always bitwise -- are
one value.

THE STAGING (`metric_classes`), shaped for the metric kernel (#414: one
thread per aux shell pair, a loop over each shell's record range) and the
`(mn|P)` follow-up (whose aux side is the same `AuxRec`). Every aux shell
pair `P <= Q` by index joins the class `(lP, lQ)` with `lP <= lQ`: a pair
whose first shell has the higher `l` is staged SWAPPED, `(j, i)`, and the
flag recorded, so the scatter writes the staged block TRANSPOSED into
`[off_i][off_j]` (`place_staged`; with the mirror, either placement fills the
same elements). Each class holds one record table -- every shell it touches,
once, its primitives contiguous -- with `rec0`/`rec1` per shell, so a pair is
two record ranges, `[p_rec0, p_rec1)` and `[q_rec0, q_rec1)`, into it. A
record is `(A, alpha, s)`: the kernel forms `E` itself
(`e_fragment.center_fragment`) and multiplies `s` in.

Plain tuples and floats, like the rest of `ao/`: `src/felizjueves` has no
runtime dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from felizjueves.ao.shell import Basis, Shell, cartesians
from felizjueves.mmd.hermite import e_block, hermite_order

Vec = tuple[float, float, float]
Ang = tuple[int, int, int]
PQ = Callable[[Ang, Ang, Vec, float], float]  # eri.eri's, restated
Block = tuple[tuple[float, ...], ...]
Matrix = list[list[float]]

_S: Ang = (0, 0, 0)


def aux_dress(c: float, *, alpha: float) -> float:
    """`s = c (pi/alpha)^{3/2}` -- the ONLY place a dressing multiplies into an
    aux record (module docstring). No normalisation: it is in `c`."""
    if not alpha > 0.0:
        raise ValueError(f"alpha = {alpha!r} is not > 0")
    return c * (math.pi / alpha) ** 1.5


def center_e(l: int, alpha: float) -> Block:  # noqa: E741
    """The undressed single-center `E^{a}_t` of one aux primitive: rows
    `cartesians(l)`, columns `hermite_order(l)` (`X_PA = 0`; module doc).
    Centre-free: the expansion does not depend on where it sits."""
    o = (0.0, 0.0, 0.0)
    return tuple(e_block(a, _S, alpha, o, o, o) for a in cartesians(l))


@dataclass(frozen=True)
class AuxRec:
    """One aux primitive as a Hermite record: centre `A` (= `P`), exponent
    `alpha` (= `zeta`), dressed scale `s`, and `i` its index in the shell."""

    i: int
    A: Vec
    alpha: float
    s: float


def aux_records(shell: Shell) -> tuple[AuxRec, ...]:
    """One `AuxRec` per primitive of `shell`, in primitive order."""
    return tuple(AuxRec(i, shell.centre, a, aux_dress(c, alpha=a))
                 for i, (a, c) in enumerate(zip(shell.exponents,
                                                shell.coefficients,
                                                strict=True)))


def _matmul(a: Sequence[Sequence[float]],
            b: Sequence[Sequence[float]]) -> list[list[float]]:
    out = []
    for row in a:
        acc = [0.0] * len(b[0])
        for x, brow in zip(row, b, strict=True):
            if x:
                for k, y in enumerate(brow):
                    acc[k] += x * y
        out.append(acc)
    return out


def _dressed(s: float, e: Block) -> list[list[float]]:
    return [[s * x for x in row] for row in e]


def _block(lP: int, precs: Sequence[AuxRec], lQ: int,
           qrecs: Sequence[AuxRec], pq: PQ, want_scale: bool):
    """The sum over the two record ranges (module docstring), `s E` formed
    per record, then `E_P [t|tau] E_Q^T` per primitive pair."""
    ts, taus = hermite_order(lP), hermite_order(lQ)
    np_, nq = len(cartesians(lP)), len(cartesians(lQ))
    total = [[0.0] * nq for _ in range(np_)]
    scale = [[0.0] * nq for _ in range(np_)] if want_scale else None
    qe = [(r, _dressed(r.s, center_e(lQ, r.alpha))) for r in qrecs]
    for p in precs:
        ep = _dressed(p.s, center_e(lP, p.alpha))
        for q, eq in qe:
            R = (p.A[0] - q.A[0], p.A[1] - q.A[1], p.A[2] - q.A[2])
            theta = math.sqrt(p.alpha * q.alpha / (p.alpha + q.alpha))
            m = [[pq(t, tau, R, theta) for tau in taus] for t in ts]
            eq_t = [list(c) for c in zip(*eq, strict=True)]
            blk = _matmul(ep, _matmul(m, eq_t))
            for r_into, r_add in zip(total, blk, strict=True):
                for k, v in enumerate(r_add):
                    r_into[k] += v
            if scale is not None:
                ab = _matmul([[abs(x) for x in r] for r in ep],
                             _matmul([[abs(x) for x in r] for r in m],
                                     [[abs(x) for x in r] for r in eq_t]))
                for r_into, r_add in zip(scale, ab, strict=True):
                    for k, v in enumerate(r_add):
                        r_into[k] += v
    return total, scale


def metric_block_and_scale(sP: Shell, sQ: Shell, *, pq: PQ
                           ) -> tuple[Block, Block]:
    """`(P|Q)` of the ordered aux shell pair `(sP, sQ)` and its |.| twin."""
    total, scale = _block(sP.l, aux_records(sP), sQ.l, aux_records(sQ), pq,
                          True)
    return (tuple(tuple(r) for r in total), tuple(tuple(r) for r in scale))


def metric_block(sP: Shell, sQ: Shell, *, pq: PQ) -> Block:
    """`(P|Q)`: rows `cartesians(sP.l)`, columns `cartesians(sQ.l)`."""
    return tuple(tuple(r) for r in _block(sP.l, aux_records(sP), sQ.l,
                                          aux_records(sQ), pq, False)[0])


def _place(aux: Basis, value: Callable[[Shell, Shell], Block]) -> Matrix:
    sh, off = aux.shells, aux.offsets
    n = aux.n_ao
    out = [[0.0] * n for _ in range(n)]
    for i in range(len(sh)):
        for j in range(i, len(sh)):
            blk = value(sh[i], sh[j])
            for a in range(sh[i].n_cart):
                for b in range(sh[j].n_cart):
                    if i == j and b > a:
                        continue  # diagonal: the lower triangle, mirrored
                    v = blk[a][b]
                    out[off[i] + a][off[j] + b] = v
                    out[off[j] + b][off[i] + a] = v
    return out


def metric_matrix(aux: Basis, *, pq: PQ) -> Matrix:
    """`(P|Q)` for every pair of aux functions, `n_aux x n_aux`, exactly
    symmetric. It computes EVERY shell pair's class, so its cost is the
    basis's highest `lP + lQ`; a caller bounds that by the basis it passes."""
    return _place(aux, lambda sP, sQ: metric_block(sP, sQ, pq=pq))


def scale_matrix(aux: Basis, *, pq: PQ) -> Matrix:
    """The scale of every `(P|Q)`, placed as `metric_matrix`."""
    return _place(aux, lambda sP, sQ: metric_block_and_scale(sP, sQ, pq=pq)[1])


@dataclass(frozen=True)
class MetricClass:
    """Every aux shell pair of one class `(lP, lQ)`, `lP <= lQ`, staged.

    `pairs[k] = (p, q)` is the shell pair as staged (`l_p <= l_q`), and
    `swapped[k]` says it is `(j, i)` of the basis pair `i < j`, so its
    staged block lands TRANSPOSED in the basis order's upper triangle
    (`place_staged`). Its records are
    `[p_rec0[k], p_rec1[k])` and `[q_rec0[k], q_rec1[k])` of the class's
    record table (`A`, `alpha`, `s`), which holds every shell the class
    touches once (`shells`, with `rec0` / `rec1` per shell)."""

    lP: int
    lQ: int
    pairs: tuple[tuple[int, int], ...]
    swapped: tuple[bool, ...]
    p_rec0: tuple[int, ...]
    p_rec1: tuple[int, ...]
    q_rec0: tuple[int, ...]
    q_rec1: tuple[int, ...]
    off_p: tuple[int, ...]
    off_q: tuple[int, ...]
    shells: tuple[int, ...]
    rec0: tuple[int, ...]
    rec1: tuple[int, ...]
    A: tuple[Vec, ...]
    alpha: tuple[float, ...]
    s: tuple[float, ...]

    RECORD_ARRAYS = ("A", "alpha", "s")
    PAIR_ARRAYS = ("p_rec0", "p_rec1", "q_rec0", "q_rec1", "off_p", "off_q")

    @property
    def n_pair(self) -> int:
        return len(self.pairs)

    @property
    def n_rec(self) -> int:
        return len(self.alpha)

    def records(self, k: int) -> tuple[tuple[AuxRec, ...], tuple[AuxRec, ...]]:
        """Pair `k`'s two record ranges as `AuxRec`s (`i` the primitive's
        index in its shell)."""
        def rng(r0: int, r1: int) -> tuple[AuxRec, ...]:
            return tuple(AuxRec(r - r0, self.A[r], self.alpha[r], self.s[r])
                         for r in range(r0, r1))
        return (rng(self.p_rec0[k], self.p_rec1[k]),
                rng(self.q_rec0[k], self.q_rec1[k]))

    def arrays(self) -> dict[str, list]:
        """The record and pair arrays, in that order, as lists."""
        return {k: [list(v) if isinstance(v, tuple) else v
                    for v in getattr(self, k)]
                for k in (*self.RECORD_ARRAYS, *self.PAIR_ARRAYS)}


def metric_classes(aux: Basis) -> tuple[MetricClass, ...]:
    """Every aux shell pair `i <= j` of `aux`, staged (`l_p <= l_q`) into its
    class, classes sorted by `(lP, lQ)`, pairs in basis order within one,
    records in first-touch order of the shells."""
    sh, off = aux.shells, aux.offsets
    cols: dict[tuple[int, int], dict[str, list]] = {}
    where: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    keys = ("pairs", "swapped", *MetricClass.PAIR_ARRAYS, "shells", "rec0",
            "rec1", *MetricClass.RECORD_ARRAYS)

    def recs_of(c: dict[str, list], w: dict[int, tuple[int, int]],
                n: int) -> tuple[int, int]:
        if n not in w:
            r0 = len(c["alpha"])
            for r in aux_records(sh[n]):
                c["A"].append(r.A)
                c["alpha"].append(r.alpha)
                c["s"].append(r.s)
            w[n] = (r0, len(c["alpha"]))
            c["shells"].append(n)
            c["rec0"].append(r0)
            c["rec1"].append(w[n][1])
        return w[n]

    for i in range(len(sh)):
        for j in range(i, len(sh)):
            swap = sh[i].l > sh[j].l
            p, q = (j, i) if swap else (i, j)
            cls = (sh[p].l, sh[q].l)
            c = cols.setdefault(cls, {k: [] for k in keys})
            w = where.setdefault(cls, {})
            c["pairs"].append((p, q))
            c["swapped"].append(swap)
            pr = recs_of(c, w, p)
            qr = recs_of(c, w, q)
            c["p_rec0"].append(pr[0])
            c["p_rec1"].append(pr[1])
            c["q_rec0"].append(qr[0])
            c["q_rec1"].append(qr[1])
            c["off_p"].append(off[p])
            c["off_q"].append(off[q])
    return tuple(MetricClass(lP, lQ, **{k: tuple(v) for k, v in c.items()})
                 for (lP, lQ), c in sorted(cols.items()))


def class_block(cls: MetricClass, k: int, *, pq: PQ) -> Block:
    """Staged pair `k`'s block from the class's own records (rows
    `cartesians(lP)`, columns `cartesians(lQ)`): what the kernel computes
    for that thread, and `metric_block` of the staged shells exactly."""
    precs, qrecs = cls.records(k)
    return tuple(tuple(r) for r in _block(cls.lP, precs, cls.lQ, qrecs, pq,
                                          False)[0])


def place_staged(cls: MetricClass, k: int, block: Sequence[Sequence[float]],
                 out: Matrix) -> None:
    """Scatter staged pair `k`'s block (rows `cartesians(lP)`, columns
    `cartesians(lQ)`) into `out` by the swap: unswapped at `[off_p][off_q]`,
    swapped TRANSPOSED at `[off_q][off_p]` -- the basis order's upper
    triangle either way -- then mirrored; a diagonal pair writes `a >= b`."""
    p, q = cls.pairs[k]
    op, oq = cls.off_p[k], cls.off_q[k]
    for a, row in enumerate(block):
        for b, v in enumerate(row):
            if p == q and b > a:
                continue
            r, c = (oq + b, op + a) if cls.swapped[k] else (op + a, oq + b)
            out[r][c] = v
            out[c][r] = v
