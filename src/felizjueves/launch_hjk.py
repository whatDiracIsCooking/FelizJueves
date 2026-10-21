"""The split HGP build's ordered pair stream and its two one-block screens (#403).

    hjk_launches(basis, D, *, bound, tau, mode, screen)  -> HPlan
    HPlan     iterable of HLaunch, plus batches / visits / skipped / counts
    HBatch(la, lb, degree, pairs)       one bin of ORDERED contracted pairs
    HPair(a, b, la, lb, degree, G)      one ordered shell pair
    HLaunch(bra, ket, ordered, orientation)    one batch pair, one kernel
    HVisit(launch, quartet, key, skipped)      one ordered shell quartet
    j_key(...) / k_key(...)             the two screens' keys
    MODES = ("canonical", "ordered"), SCREENS = ("j", "k")

A sibling of `launch_f.py` for the SPLIT build (milestone 14), in which a `J`
kernel and a `K` kernel each read one density block. THE PLAN IS DATA
(milestone 6's idiom): no text is emitted, nothing is compiled, launched or
interpreted, and no ERI is evaluated -- the Schwarz bound is the only thing
called, and it is INJECTED exactly as `launch_f`'s is (`bound(sa, sb) ->
G_[ab] per primitive pair`, in `ao.pairs.primitive_pairs` order; tests pass
`functools.partial(ao.pairs.primitive_schwarz_bounds, one_centre=...)`), so
this module reaches no `reference/`.

NO SYMMETRY PRUNING. `launch_f` keeps one contracted pair per UNORDERED shell
pair and launches the upper triangle of batch pairs, because the fused
kernel's digestion deposits the 8-fold orbit itself. Here every ORDERED shell
pair `(a, b)` is a pair (so `(sp)` and `(ps)` both occur), and every ORDERED
batch pair `(I, J)` is a launch: the stream visits every ordered quartet of
shell pairs `(ab|cd)` -- `n_shell^4` of them -- exactly once.

THE PAIRS AND THEIR BOUND. `HPair.G = Q_ab` is the injected per-primitive
bound summed over ALL `K_a K_b` primitive pairs of the ordered pair, so it
bounds every element of the contracted block (launch_f, "THE BOUND"), and
`Q_ab Q_cd >= |(ij|kl)|` for every element of `(ab|cd)`. Nothing is pruned:
the degree is `K_a K_b` (pruning is `launch_f`'s eq. 29 machinery, which the
split build does not adopt here), and a degree above `MAX_PRIM_PAIRS = 128`
(`kernel_f`'s shared staging cap, restated; a test pins the two equal) is
REFUSED, as `kernel_f` refuses it.

THE BATCHES. Degree-uniform, as in `launch_f`: one bin per `(la, lb,
degree)`, bins ordered by that key; inside one the pairs sit descending by
`G`, stably (pair order -- `a` outer, `b` inner -- on ties). `G` and not a
density-weighted key, because `K`'s key crosses the bra and the ket (it
reads `D_bd`), so no per-pair density factor is the screen's; and since both
screens SKIP and never break, the order inside a batch is not load-bearing.

THE TWO MODES. Every ordered batch pair `(I, J)` has one ordered class
`(la_I, lb_I, la_J, lb_J)`; the mode decides which kernel serves it.

  ordered    the launch's kernel class is that ordered class itself; the
             orientation is the identity.
  canonical  the launch is routed to its orbit REPRESENTATIVE's class
             (`hgp.orbits.representative`), carrying the permutation
             (`hgp.orbits.to_representative`, an `Orientation`): the kernel
             runs on the shells `quartet[j] for j in orientation.shells`
             (`HVisit.kernel_quartet`), and `block.transpose(orientation.axes)`
             is the ordered quartet's block.

Both modes launch the SAME batch pairs and visit the SAME ordered quartets;
only the kernel class each visit is routed to, and the permutation it
carries, differ. So canonical mode builds one kernel per orbit, ordered mode
one per ordered class.

THE TWO SCREENS, ONE BLOCK EACH. A split kernel reads one density block, so
each screen bounds the contribution through that block alone:

    J_ab += D_cd (ab|cd)    screen "j": key = Q_ab Q_cd max|D_cd|
    K_ac += D_bd (ab|cd)    screen "k": key = Q_ab Q_cd max|D_bd|

`max|D_xy|` the largest `|D_mn|` over shell pair `(x, y)`'s density block.
A quartet is SKIPPED when `key < tau` -- then every element's contribution
`|D_kl (ij|kl)|` (J) or `|D_jl (ij|kl)|` (K) is `<= key < tau` -- and every
other quartet is visited: SKIP, NEVER BREAK (as `launch_f`), so the order of
the kets is not part of the screen's soundness. At `tau = 0` nothing is
skipped (`key >= 0`). J's key applied to a K build is UNSOUND: a zero (or
small) `D_cd` block does not bound `D_bd` (test_hjk_pairs.py shows a quartet
it skips that K needs).

WHAT IS NOT HERE. No `Spec`: the split HGP target's key is #401's
(`Spec.engine`), so a launch names its kernel by class (`HLaunch.cls`), not
by `Spec`. No eq. 29 magnitudes, no pruning, no pre-launch screen, no sort
key that can break. No import of `launch_f`'s or `launch`'s privates
(#328): the density and `tau` checks and the stable sort are restated here.

Plain tuples and floats; `D` is anything indexable as `D[m][n]`, symmetric
(`ao.j`'s density contract, restated).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from felizjueves.ao.shell import Basis, Shell
from felizjueves.hgp.orbits import Orientation, to_representative

Bound = Callable[[Shell, Shell], Sequence[float]]
Quartet = tuple[int, int, int, int]

MODES = ("canonical", "ordered")
SCREENS = ("j", "k")

# kernel_f's shared staging cap, restated (this module may not import a
# kernel builder); test/test_hjk_pairs.py pins the two equal.
MAX_PRIM_PAIRS = 128

IDENTITY = (0, 1, 2, 3)


# --- the contracts, restated (#328: no private borrowed) ---------------------


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
                raise ValueError(f"D is not symmetric: D[{i}][{j}] = "
                                 f"{rows[i][j]!r} but D[{j}][{i}] = "
                                 f"{rows[j][i]!r}")
    return rows


def _tau(tau: object) -> float:
    if not (isinstance(tau, int | float) and not isinstance(tau, bool)
            and math.isfinite(tau) and tau >= 0.0):
        raise ValueError(f"tau = {tau!r} is not a finite float >= 0")
    return float(tau)


def _nonneg(x: object, what: str) -> float:
    if not (isinstance(x, int | float) and not isinstance(x, bool)
            and math.isfinite(x) and x >= 0.0):
        raise ValueError(f"{what} = {x!r} is not a finite float >= 0")
    return float(x)


# --- the keys ----------------------------------------------------------------


def j_key(q_ab: float, q_cd: float, dmax_cd: float) -> float:
    """J's one-block key: `Q_ab Q_cd max|D_cd|`."""
    return q_ab * q_cd * dmax_cd


def k_key(q_ab: float, q_cd: float, dmax_bd: float) -> float:
    """K's one-block key: `Q_ab Q_cd max|D_bd|`."""
    return q_ab * q_cd * dmax_bd


# --- the plan ----------------------------------------------------------------


@dataclass(frozen=True)
class HPair:
    """One ORDERED contracted shell pair `(a, b)`: `degree = K_a K_b`, `G =
    Q_ab` the injected bound summed over every primitive pair."""

    a: int
    b: int
    la: int
    lb: int
    degree: int
    G: float

    @property
    def shell_pair(self) -> tuple[int, int]:
        return (self.a, self.b)


@dataclass(frozen=True)
class HBatch:
    """One bin `(la, lb, degree)`, its pairs descending by `G`, stably."""

    la: int
    lb: int
    degree: int
    pairs: tuple[HPair, ...]

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("an empty batch is not a batch")
        for p in self.pairs:
            if (p.la, p.lb, p.degree) != (self.la, self.lb, self.degree):
                raise ValueError(f"pair {p.shell_pair} is not of batch "
                                 f"{(self.la, self.lb, self.degree)}")

    @property
    def cls(self) -> tuple[int, int]:
        return (self.la, self.lb)


@dataclass(frozen=True)
class HLaunch:
    """The ordered batch pair `(bra, ket)` (indices into `HPlan.batches`),
    its ordered class, and the `Orientation` routing it to the class its
    kernel is built for (`cls`; the identity in ordered mode)."""

    bra: int
    ket: int
    ordered: tuple[int, int, int, int]
    orientation: Orientation

    @property
    def cls(self) -> tuple[int, int, int, int]:
        """The class of the kernel that runs this launch."""
        return self.orientation.rep


@dataclass(frozen=True)
class HVisit:
    """One ordered shell quartet `(a, b, c, d)` of launch `launch` (an index
    into `HPlan.launches`), its screen key, and whether it is skipped."""

    launch: int
    quartet: Quartet
    shells: Quartet  # the orientation's `shells`: kernel slot j <- quartet[shells[j]]
    key: float
    skipped: bool

    @property
    def kernel_quartet(self) -> Quartet:
        """The shells in the kernel class's positions."""
        return tuple(self.quartet[j] for j in self.shells)  # type: ignore[return-value]


@dataclass(frozen=True)
class HPlan:
    """The ordered pair stream (module docstring). Iterates its launches."""

    tau: float
    mode: str
    screen: str
    batches: tuple[HBatch, ...]
    launches: tuple[HLaunch, ...]
    visits: tuple[HVisit, ...]

    def __iter__(self) -> Iterator[HLaunch]:
        return iter(self.launches)

    def __len__(self) -> int:
        return len(self.launches)

    @property
    def pairs(self) -> tuple[HPair, ...]:
        return tuple(p for b in self.batches for p in b.pairs)

    @property
    def skipped(self) -> tuple[HVisit, ...]:
        return tuple(v for v in self.visits if v.skipped)

    @property
    def classes(self) -> tuple[tuple[int, int, int, int], ...]:
        """The distinct kernel classes the launches name, sorted."""
        return tuple(sorted({x.cls for x in self.launches}))

    def counts(self) -> dict[str, int]:
        """What the stream holds, as plain numbers."""
        return {
            "pairs": len(self.pairs),
            "batches": len(self.batches),
            "launches": len(self.launches),
            "classes": len(self.classes),
            "quartets": len(self.visits),
            "quartets_skipped": len(self.skipped),
        }


def _identity(cls: tuple[int, int, int, int]) -> Orientation:
    return Orientation(cls, cls, IDENTITY, IDENTITY)


def hjk_launches(basis: Basis, D: object, *, bound: Bound, tau: float,
                 mode: str, screen: str) -> HPlan:
    """The ordered pair stream of `basis` and the symmetric density `D`
    (module docstring). `bound(sa, sb)` returns `G_[ab]` per primitive pair
    of the ordered shell pair, injected; `mode` one of `MODES`, `screen` one
    of `SCREENS`."""
    if mode not in MODES:
        raise ValueError(f"mode = {mode!r} is not one of {MODES}")
    if screen not in SCREENS:
        raise ValueError(f"screen = {screen!r} is not one of {SCREENS}")
    rows = _density(basis, D)
    tau = _tau(tau)
    shells, off = basis.shells, basis.offsets
    ns = len(shells)

    G: dict[tuple[int, int], float] = {}
    dmax: dict[tuple[int, int], float] = {}
    pairs: list[HPair] = []
    for a in range(ns):
        for b in range(ns):
            sa, sb = shells[a], shells[b]
            g = tuple(bound(sa, sb))
            want = len(sa.exponents) * len(sb.exponents)
            if len(g) != want:
                raise ValueError(f"bound(shell {a}, shell {b}) gave {len(g)} "
                                 f"values, not K_a K_b = {want}")
            if want > MAX_PRIM_PAIRS:
                raise ValueError(
                    f"shell pair ({a}, {b}) has degree {want} > "
                    f"MAX_PRIM_PAIRS = {MAX_PRIM_PAIRS}, the kernel's shared "
                    "staging cap: refused, as kernel_f refuses it")
            g = tuple(_nonneg(x, f"bound(shell {a}, shell {b})") for x in g)
            G[a, b] = math.fsum(g)
            dmax[a, b] = max(abs(rows[m][k])
                             for m in range(off[a], off[a] + sa.n_cart)
                             for k in range(off[b], off[b] + sb.n_cart))
            pairs.append(HPair(a, b, sa.l, sb.l, want, G[a, b]))

    bins: dict[tuple[int, int, int], list[HPair]] = {}
    for p in pairs:
        bins.setdefault((p.la, p.lb, p.degree), []).append(p)
    batches = tuple(
        HBatch(*k, tuple(sorted(bins[k], key=lambda p: -p.G)))  # stable
        for k in sorted(bins))

    launches: list[HLaunch] = []
    visits: list[HVisit] = []
    for bi, bra in enumerate(batches):
        for ki, ket in enumerate(batches):
            ordered = (bra.la, bra.lb, ket.la, ket.lb)
            orient = (to_representative(ordered) if mode == "canonical"
                      else _identity(ordered))
            li = len(launches)
            launches.append(HLaunch(bi, ki, ordered, orient))
            for p in bra.pairs:
                for q in ket.pairs:
                    a, b, c, d = p.a, p.b, q.a, q.b
                    key = (j_key(p.G, q.G, dmax[c, d]) if screen == "j"
                           else k_key(p.G, q.G, dmax[b, d]))
                    visits.append(HVisit(li, (a, b, c, d), orient.shells,
                                         key, key < tau))

    return HPlan(tau, mode, screen, batches, tuple(launches), tuple(visits))
