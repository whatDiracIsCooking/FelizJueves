"""The fused Fock build's pair stream, as a value (#285; [Fock] §3.2).

    f_launches(basis, D, *, bound, tau, n, ket_break=False, scatter=False)
                                                  -> FPlan
    launch_footprint(plan, launch, basis)         -> FFootprint (#358)
    FPlan      iterable of FLaunch, plus batches / screened / deleted / counts
    FBatch(la, lb, degree, mag, magnitude, pairs) one bin of contracted pairs
    FPair(a, b, la, lb, n_prim, prims, prim_G, G, dmax, key, mag)
    FLaunch(bra, ket, spec)                       one batch pair, one kernel
    magnitudes(tau, n) / exponent(i, n)           eq. 29, numeric / symbolic
    magnitude_index(G, mags)                      the UPPER-bound assignment
    prelaunch_screened(i, j, n)                   i + j >= n
    printed_magnitude_index / printed_prelaunch_screened   the text, as printed

The host side `[Fock]` §3.2 asks for before any fused kernel runs (decisions
2, 3, 4 and 8 of this repo's reading, data/literature/fock/coverage.json):
which contracted shell pairs exist, which of their primitive pairs are kept,
how the pairs are binned, which batch pairs are launched, and in which order
the pairs of a batch sit. THE PLAN IS DATA (milestone 6's idiom, `launch.py`):
no text is emitted, nothing is compiled, launched or interpreted, and no ERI
is evaluated -- the bound is the only thing called, and it is INJECTED
(`bound(sa, sb) -> G_[ab] per primitive pair`; tests pass
`functools.partial(ao.pairs.primitive_schwarz_bounds, one_centre=...)`), so
this module reaches no `reference/`.

THE PAIRS (decision 4). One CONTRACTED pair per unordered shell pair, written
in AO order with `la <= lb` -- `sp`, never `ps` -- and, when `la == lb`, the
lower shell index first. So a pair class is one of `(0,0) (0,1) (0,2) (1,1)
(1,2) (2,2)` on milestone 12's ladder, and a launch pairs two of them.

THE BOUND, AND WHICH ONE THE SCREENS READ. `G_[ab]` per primitive pair is
`ao.pairs.primitive_schwarz_bound` (the J engine's proved `I_p`, read through
the dressed `E`: ao/pairs.py, "THE CONTRACTED SCHWARZ BOUND OF [Fock]"), and
`FPair.G` is its sum over ALL `K_a K_b` primitive pairs -- BEFORE pruning.
That `G` bounds every element of the exact contracted block and of every
pruned one (a subset sum is smaller), so a screen on it is sound against the
answer, not only against what the kernel computes. The sort key and the
magnitude read the same `G`.

PRUNING (decision 8). A contracted pair's primitive pairs are ordered
descending by `G_[ab]` and the longest tail summing below

    eps = tau / (2 Gmax),     Gmax = max over the basis' pairs of G

is DELETED (`ao.pairs.prune`): the kept set is a prefix of the bound order,
recorded in that order (`FPair.prims`, indices into
`ao.pairs.primitive_pairs(shells[a], shells[b])`). Then for any quartet the
pruned block differs from the exact one by at most (ao/pairs.py)
`t_ab G_cd + G_ab' t_cd < eps Gmax + Gmax eps = tau` per element, `t` the
deleted sums: PRUNING MOVES NO ERI BY `tau` OR MORE. A pair whose every
primitive pair is deleted (`G < eps`, so `G_ab G_cd < tau / 2` against every
partner) is an insignificant contracted pair and is DELETED whole
(`FPlan.deleted`): it is in no batch. At `tau = 0`, `eps = 0` and nothing is
deleted.

THE BIN KEY (decisions 1-2): `(la, lb, degree, mag)`, with `degree = K_AB =
len(prims)`, the primitive pairs KEPT -- `K_a K_b` when nothing is pruned
(Fig. 1's 1, 3, 9). Because the degree is in the key, every pair of a batch
has exactly `degree` primitive pairs, so the staged record
(`FBatch.packed`) is `len(pairs)` rows of `degree` real entries: no padding,
no sentinel. Batches are ordered by `(la, lb, degree, mag)`; inside one the
pairs sit descending by `key = G * dmax`, `dmax` the largest `|D_mn|` over
the pair's density block, stably (pair order on ties: `launch._sorted_desc`,
the J plan's sort) -- [Fock]'s "sorted by G_ab times the largest element of
the pair's density block" (decision 2), the ordering Fig. 1 shades
"decreasing size".

THE MAGNITUDES (eq. 29). `m_i = tau^(i/n)`, `i = 0 .. n-1`, so `m_i m_j =
tau^((i+j)/n)`, which is `<= tau` EXACTLY when `i + j >= n` (`exponent` holds
the exponents as fractions, so a test reads that symbolically). A pair goes
to the magnitude that BOUNDS IT FROM ABOVE:

    mag = the largest i in 1 .. n-1 with G < m_i, else 0

so every pair of a batch with `mag >= 1` has `G < m_mag` (the smallest
magnitude above `G`); batch 0 holds every `G >= m_1`, including `G > 1 = m_0`
(a tight core pair's `G` exceeds 1), and `m_0` bounds nothing -- harmless,
because no batch pair with a `mag = 0` side satisfies `i + j >= n`. The pairs
below `m_{n-1}` are all in batch `n - 1`. THE PRE-LAUNCH SCREEN: the batch
pair `(I, J)` is removed before launch when `mag_I + mag_J >= n`; then for
every quartet it holds `G_ab G_cd < m_i m_j <= tau`, so every ERI of it is
`< tau` in magnitude -- the screen is rigorous, against the exact answer.
The plan CHECKS the last step in floating point: each removed batch pair's
largest product `max G_I * max G_J` must be `< tau`, or the plan is refused
rather than an unsound one returned. The screen reads no density (it cannot:
the in-kernel six-block screen is what does, decision 7, #282/#287).

THE PRINTED TEXT, AND WHY IT IS NOT WHAT IS BUILT. Two readings of §3.2 are
defect records in data/literature/fock/claims.json (#271), both asserted by
#291; this module provides the printed forms so that a test can show the
difference, and uses neither:

  `fock/batch-magnitude-bound-direction`: the text assigns a pair "the
     largest batch magnitude that is less than its G_ab value"
     (`printed_magnitude_index`), a LOWER bound -- under it a batch pair
     the eq. 29 screen removes can hold a quartet with `G_ab G_cd >= tau`
     (indeed an ERI `>= tau`). Fig. 1 labels the magnitudes "Upper bound";
     the assignment above is the upper-bound one.
  `fock/eq29-screen-inequality-reversed`: the text says the batches with
     `i + j <= n` are screened (`printed_prelaunch_screened`). Read as
     removal it removes the LARGEST batches (at `n = 2`, every batch pair);
     Fig. 1 launches every pair holding a magnitude-1 batch and omits
     exactly `i + j = n`. The screen above is `i + j >= n`, the one that
     matches the figure and the bound.

THE LAUNCHES (decision 4). Batch pairs `(I, J)`, `I <= J` in batch order --
the upper triangle of the class-quartet matrix, so with pair classes `P(I) <=
P(J)` (batches are ordered by class first) -- each not removed by the
pre-launch screen, one `FLaunch` each, naming its kernel's `Spec`:
`Target.F`, `am = (la_I, lb_I, la_J, lb_J)` in AO order (spec.py, #272), one
batch. On milestone 12's ladder (every `l <= 2`) those are the 21 classes.
`I == J` is a launch too: which quartets of it a kernel computes (the
canonical half) is the digestion's business (#283), not the plan's.

THE MONOTONE KET KEY (#352), opt-in: `ket_break=True`. Everything above
is unchanged except the order inside a batch and the launches' `Spec`: the
pairs sit descending by `G` ALONE (`FPair.key = G`, stably, the same
`launch._sorted_desc`), and every launch names its class with the
`KET_BREAK` lever, whose kernel ends a thread's ket loop at the first ket
whose separable bound `G_ab (G_cd M_ab)` is under `tau` (f_levers/
ket_break.py has the dominating factor `M_ab` and the proof). `G * dmax`
cannot be that key: the six-block `dmax` is not monotone along it (pin 8),
and `G` is the only factor of the bound that every ket carries alone --
`Target.K`'s precedent (`launch_k`: kets sorted by `I_bd` alone). The bins,
the magnitudes, pruning and eq. 29's pre-launch screen read `G`, not the
order, so they are the default stream's exactly. The default
(`ket_break=False`) stream is byte-for-byte the one before #352.

THE ATOMIC-FREE WRITE-BACK (#358), opt-in: `scatter=True`. The stream is
the default's (or `ket_break`'s) exactly -- pairs, bins, order, screens --
and every launch names its class with the `SCATTER` lever too, whose
kernel stores its contributions to a per-launch slot buffer `W` and leaves
`F` to a second kernel (f_levers/scatter.py, kernel_f_scatter.py). Such a
launch needs host buffers the default does not, so the plan DECLARES them,
from the plan and the basis alone (`launch_footprint`), derived here
independently of the kernel's layout (a test holds the two equal):

    n_slot  = n_bra (n_a n_b + n_ket S5),  S5 = n_c n_d + n_a n_c + n_a n_d
                                                + n_b n_c + n_b n_d
    n_el   <= n_bra n_a n_b + n_ket n_c n_d + |A_bra| |A_ket|

`n_slot` the slots of `W` -- per bra its block-reduced `f_ab` and per
quartet its five ket-dependent blocks, the only units with one writer
without an atomic -- and `n_el` the elements of `F` the scatter writes:
the bra blocks, the ket blocks, and the cross blocks, whose rows are the
AOs `A_bra` of the bra batch's shells and columns the AOs `A_ket` of the
ket batch's. A buffer smaller than either is a wrong answer, never a
dropped term, so the kernel declares `W`'s extent and the host's map
refuses a footprint past the bound. With `scatter=False` nothing here
changes.

THE COUNTS. `FPlan` counts what fired -- pairs deleted, primitive pairs
pruned, batch pairs removed before launch -- and nothing more: on this
repo's compact fixtures those counts are facts about the fixtures, not about
the screen (milestone 12 pin 9; data/literature/fock/limits.json). How much
an extended molecule saves is a follow-up.

Plain tuples and floats; `D` is anything indexable as `D[m][n]`, symmetric
(`ao.j`'s density contract, shared).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction

from felizjueves.ao.density import checked_density as _density
from felizjueves.ao.density import checked_tau as _tau
from felizjueves.ao.pairs import prune
from felizjueves.ao.shell import Basis, Shell
from felizjueves.launch import _nonneg, _sorted_desc
from felizjueves.spec import Operator, Spec, Target

Bound = Callable[[Shell, Shell], Sequence[float]]

# #352: the kernel-body lever a `ket_break=True` stream's launches name; the
# same string as `f_levers.ket_break.NAME` (this module may not import the
# kernel's levers; test/test_f_lever_ket_break.py pins the two equal).
KET_BREAK = "ket-break"
# #358: the atomic-free write-back's lever, `f_levers.scatter.NAME` restated
# (test/test_f_lever_scatter.py pins the two equal).
SCATTER = "scatter"


# --- eq. 29 ------------------------------------------------------------------


def _n(n: object) -> int:
    if type(n) is not int or n < 1:
        raise ValueError(f"n = {n!r} is not an int >= 1 (the number of "
                         "eq. 29 magnitudes)")
    return n


def exponent(i: int, n: int) -> Fraction:
    """The exponent of `m_i = tau^(i/n)`, exactly: `Fraction(i, n)`."""
    _n(n)
    if type(i) is not int or not 0 <= i < n:
        raise ValueError(f"i = {i!r} is not a magnitude index in [0, {n})")
    return Fraction(i, n)


def magnitudes(tau: float, n: int) -> tuple[float, ...]:
    """Eq. 29's batch magnitudes `tau^(i/n)`, `i = 0 .. n-1` (`m_0 = 1`)."""
    tau = _tau(tau)
    return tuple(tau ** (i / _n(n)) for i in range(n))


def magnitude_index(G: float, mags: Sequence[float]) -> int:
    """The UPPER-bound assignment (module docstring): the largest `i >= 1`
    with `G < mags[i]`, else 0."""
    G = _nonneg(G, "G")
    for i in range(len(mags) - 1, 0, -1):
        if G < mags[i]:
            return i
    return 0


def prelaunch_screened(i: int, j: int, n: int) -> bool:
    """Eq. 29's screen as built: `m_i m_j <= tau` exactly when `i + j >= n`."""
    return exponent(i, n) + exponent(j, n) >= 1


def printed_magnitude_index(G: float, mags: Sequence[float]) -> int:
    """§3.2 AS PRINTED (not used here): "the largest batch magnitude that is
    less than its G_ab value" -- the smallest `i` with `mags[i] < G`, a
    LOWER bound; the last batch when none is (`fock/batch-magnitude-bound-
    direction`)."""
    G = _nonneg(G, "G")
    for i, m in enumerate(mags):
        if m < G:
            return i
    return len(mags) - 1


def printed_prelaunch_screened(i: int, j: int, n: int) -> bool:
    """§3.2 AS PRINTED (not used here): "for all i and j such that i+j <= n,
    the batches will be screened" (`fock/eq29-screen-inequality-reversed`)."""
    return exponent(i, n) + exponent(j, n) <= 1


# --- the plan ----------------------------------------------------------------


@dataclass(frozen=True)
class FPair:
    """One contracted shell pair (module docstring, "THE PAIRS").

    `a, b` the shells in AO order (`la <= lb`; `a < b` on a tie of `l`),
    `n_prim = K_a K_b`, `prims` the KEPT primitive-pair indices into
    `ao.pairs.primitive_pairs(shells[a], shells[b])` in descending-bound
    order and `prim_G` their `G_[ab]`, `G` the bound summed over ALL
    `n_prim` (before pruning), `dmax` the largest `|D|` of the pair's density
    block, `key = G * dmax`, `mag` its eq. 29 magnitude index.
    """

    a: int
    b: int
    la: int
    lb: int
    n_prim: int
    prims: tuple[int, ...]
    prim_G: tuple[float, ...]
    G: float
    dmax: float
    key: float
    mag: int

    @property
    def degree(self) -> int:
        """`K_AB`: the primitive pairs kept."""
        return len(self.prims)

    @property
    def shell_pair(self) -> tuple[int, int]:
        return (self.a, self.b)


@dataclass(frozen=True)
class FBatch:
    """One bin `(la, lb, degree, mag)`, its pairs descending by `key`."""

    la: int
    lb: int
    degree: int
    mag: int
    magnitude: float
    pairs: tuple[FPair, ...]

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("an empty batch is not a batch")
        for p in self.pairs:
            if (p.la, p.lb, p.degree, p.mag) != (self.la, self.lb, self.degree,
                                                 self.mag):
                raise ValueError(f"pair {p.shell_pair} is not of batch "
                                 f"{(self.la, self.lb, self.degree, self.mag)}")

    @property
    def cls(self) -> tuple[int, int]:
        return (self.la, self.lb)

    @property
    def packed(self) -> tuple[tuple[tuple[int, int, int], ...], ...]:
        """The staged record: one row per pair, `(a, b, prim)` per kept
        primitive pair -- every row exactly `degree` long, no padding."""
        return tuple(tuple((p.a, p.b, k) for k in p.prims) for p in self.pairs)

    @property
    def g_max(self) -> float:
        return max(p.G for p in self.pairs)


@dataclass(frozen=True)
class FLaunch:
    """The batch pair `(bra, ket)` (indices into `FPlan.batches`, `bra <=
    ket`) and the `Spec` of the kernel that runs it."""

    bra: int
    ket: int
    spec: Spec


@dataclass(frozen=True)
class FPlan:
    """The pair stream (module docstring). Iterates its launches."""

    tau: float
    n: int
    magnitudes: tuple[float, ...]
    eps: float
    g_max: float
    batches: tuple[FBatch, ...]
    launches: tuple[FLaunch, ...]
    screened: tuple[tuple[int, int], ...]
    deleted: tuple[tuple[int, int], ...]
    n_prim_total: int

    def __iter__(self) -> Iterator[FLaunch]:
        return iter(self.launches)

    def __len__(self) -> int:
        return len(self.launches)

    @property
    def pairs(self) -> tuple[FPair, ...]:
        return tuple(p for b in self.batches for p in b.pairs)

    @property
    def n_pairs_total(self) -> int:
        return len(self.pairs) + len(self.deleted)

    @property
    def n_prim_kept(self) -> int:
        return sum(p.degree for p in self.pairs)

    @property
    def n_prim_pruned(self) -> int:
        return self.n_prim_total - self.n_prim_kept

    def counts(self) -> dict[str, int]:
        """What fired, as plain numbers (limits.json records these)."""
        nb = len(self.batches)
        return {
            "pairs": self.n_pairs_total,
            "pairs_deleted": len(self.deleted),
            "primitive_pairs": self.n_prim_total,
            "primitive_pairs_pruned": self.n_prim_pruned,
            "batches": nb,
            "batch_pairs": nb * (nb + 1) // 2,
            "batch_pairs_screened": len(self.screened),
            "launches": len(self.launches),
        }


def f_class_spec(bra: FBatch, ket: FBatch, *,
                 levers: tuple[str, ...] = ()) -> Spec:
    """`Target.F`, `am = (la, lb, lc, ld)` in AO order, one batch; `levers`
    the kernel-body levers its kernel is built with (none by default)."""
    return Spec(Operator.COULOMB, 0, (bra.la, bra.lb, ket.la, ket.lb),
                Target.F, levers=levers)


def _canonical(shells: Sequence[Shell], i: int, j: int) -> tuple[int, int]:
    return (i, j) if (shells[i].l, i) <= (shells[j].l, j) else (j, i)


def f_launches(basis: Basis, D: object, *, bound: Bound, tau: float,
               n: int, ket_break: bool = False,
               scatter: bool = False) -> FPlan:
    """The pair stream of `basis` and the symmetric density `D` (module
    docstring). `bound(sa, sb)` returns `G_[ab]` per primitive pair of the
    ordered shell pair, in `primitive_pairs` order; injected. `ket_break`:
    the monotone ket key and its lever (module docstring, #352). `scatter`:
    every launch names the atomic-free write-back's lever (#358)."""
    if type(ket_break) is not bool:
        raise ValueError(f"ket_break = {ket_break!r} is not a bool")
    if type(scatter) is not bool:
        raise ValueError(f"scatter = {scatter!r} is not a bool")
    rows = _density(basis, D)
    tau = _tau(tau)
    n = _n(n)
    mags = magnitudes(tau, n)
    shells, off = basis.shells, basis.offsets

    raw = []  # (a, b, per-primitive bounds, G), in unique-pair order
    for i in range(len(shells)):
        for j in range(i, len(shells)):
            a, b = _canonical(shells, i, j)
            sa, sb = shells[a], shells[b]
            g = tuple(bound(sa, sb))
            want = len(sa.exponents) * len(sb.exponents)
            if len(g) != want:
                raise ValueError(f"bound(shell {a}, shell {b}) gave {len(g)} "
                                 f"values, not K_a K_b = {want}")
            g = tuple(_nonneg(x, f"bound(shell {a}, shell {b})") for x in g)
            raw.append((a, b, g, math.fsum(g)))
    g_max = max((r[3] for r in raw), default=0.0)
    eps = tau / (2.0 * g_max) if tau > 0.0 and g_max > 0.0 else 0.0

    kept_pairs, deleted, n_prim_total = [], [], 0
    for a, b, g, G in raw:
        n_prim_total += len(g)
        keep = prune(g, eps=eps)
        if not keep:
            deleted.append((a, b))
            continue
        sa, sb = shells[a], shells[b]
        dmax = max(abs(rows[m][k])
                   for m in range(off[a], off[a] + sa.n_cart)
                   for k in range(off[b], off[b] + sb.n_cart))
        kept_pairs.append(FPair(a, b, sa.l, sb.l, len(g), keep,
                                tuple(g[k] for k in keep), G, dmax,
                                G if ket_break else G * dmax,
                                magnitude_index(G, mags)))

    bins: dict[tuple[int, int, int, int], list[FPair]] = {}
    for p in kept_pairs:
        bins.setdefault((p.la, p.lb, p.degree, p.mag), []).append(p)
    batches = tuple(
        FBatch(*k, mags[k[3]], tuple(_sorted_desc(bins[k])))
        for k in sorted(bins))

    levers = tuple(sorted(((KET_BREAK,) if ket_break else ())
                          + ((SCATTER,) if scatter else ())))
    launches, screened = [], []
    for bi, bra in enumerate(batches):
        for ki in range(bi, len(batches)):
            ket = batches[ki]
            if prelaunch_screened(bra.mag, ket.mag, n):
                if not bra.g_max * ket.g_max < tau:
                    raise ValueError(
                        f"batch pair ({bi}, {ki}) at magnitudes ({bra.mag}, "
                        f"{ket.mag}) holds G_ab G_cd = "
                        f"{bra.g_max * ket.g_max!r} >= tau = {tau!r}: eq. "
                        "29's magnitudes rounded past the bound, and the "
                        "screen would not be sound")
                screened.append((bi, ki))
            else:
                launches.append(FLaunch(bi, ki, f_class_spec(
                    bra, ket, levers=levers)))

    return FPlan(tau, n, mags, eps, g_max, batches, tuple(launches),
                 tuple(screened), tuple(deleted), n_prim_total)


# --- #358: the atomic-free write-back's host buffers, declared ---------------


@dataclass(frozen=True)
class FFootprint:
    """One launch's host buffers for the `scatter` lever (module docstring,
    "THE ATOMIC-FREE WRITE-BACK"): `n_slot` slots of `W`, and at most
    `n_elem_bound` elements of `F` written by the scatter; `bra_aos` and
    `ket_aos` the AOs of the two sides' shells, sorted."""

    n_bra: int
    n_ket: int
    n_slot: int
    bra_aos: tuple[int, ...]
    ket_aos: tuple[int, ...]
    n_elem_bound: int


def launch_footprint(plan: FPlan, launch: FLaunch,
                     basis: Basis) -> FFootprint:
    """`launch`'s declared footprint, from the plan and `basis` alone."""
    if launch not in plan.launches:
        raise ValueError(f"{launch} is not one of the plan's launches")
    bra, ket = plan.batches[launch.bra], plan.batches[launch.ket]
    shells, off = basis.shells, basis.offsets
    na, nb = shells[bra.pairs[0].a].n_cart, shells[bra.pairs[0].b].n_cart
    nc, nd = shells[ket.pairs[0].a].n_cart, shells[ket.pairs[0].b].n_cart
    s5 = nc * nd + na * nc + na * nd + nb * nc + nb * nd

    def aos(batch: FBatch) -> tuple[int, ...]:
        return tuple(sorted({off[s] + i for p in batch.pairs
                             for s in (p.a, p.b)
                             for i in range(shells[s].n_cart)}))

    A, B = aos(bra), aos(ket)
    n_bra, n_ket = len(bra.pairs), len(ket.pairs)
    return FFootprint(n_bra, n_ket, n_bra * (na * nb + n_ket * s5), A, B,
                      n_bra * na * nb + n_ket * nc * nd + len(A) * len(B))
