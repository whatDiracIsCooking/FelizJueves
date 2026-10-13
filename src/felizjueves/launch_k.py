"""The K launch plan, as a value -- and whether `ClassLaunch` is J-shaped (#204).

    k_launches(basis, D, *, bound, key, tau, n_batches=1, strict=True,
               build=None)                         -> KLaunchPlan
    KLaunchPlan   iterable of KClassLaunch, plus tau / dm / n_batches / refused
    KClassLaunch(am, n_batches, bras, kets, kernels, bra_blocks)
    walk_k(plan, basis, D, *, pq, order="sorted", check_sorted=True)
                                                   -> ao.k.KScreened
    exchange_launched(basis, D, *, pq, tau, n_batches=1, strict=True)
                                                   -> ao.k.KScreened

`launch.py`'s counterpart for the exchange matrix `K_ab = sum_cd D_cd
(ac|bd)` (ao/k.py): which K class kernels run, over which bra and ket index
records, in which order -- and nothing else. THE PLAN IS DATA: no text is
emitted, nothing compiled or launched, no `[p|q]` evaluated (the bound is
the only thing called, and it is injected).

THE ANSWER (#204's question): OUTCOME 2. A `KClassLaunch` sits BESIDE
`ClassLaunch`, and the shared part is extracted into `launch.py` and
imported from it, unchanged in behaviour (the J plans are pinned by a
fingerprint taken before this module existed, test/test_launch.py):

  SHARED   `launch.PairRec` (one index record: shell pair, primitive index,
           pair momentum, bound `G`, sort `key`); `launch._sorted_desc` (the
           stable descending sort by `key` -- the injected key is written
           into `PairRec.key`, so the one sort serves both); `launch._nonneg`
           (a bound's value refused unless a finite float >= 0); and
           `launch._batches_for` (a class's batch count, asked of the plan,
           `strict` raising or recording -- admission depends on
           `(lp, lq, n_batches)` only, and test/test_launch_k.py pins that the
           plan admits a K `Spec` exactly when it admits the J class of the
           same split).

  NOT SHARED, because `ClassLaunch` is J-shaped in four places, none of them
  the sort key:

  1. THE CLASS IDENTITY. `ClassLaunch` is keyed `(lp, lq)` because the J
     kernel reads only the two Hermite sums: one `build_pq_kernels` serves
     every shell class with those sums (test/test_launch.py pins it). A K
     kernel inlines the `E` fragments of BOTH pairs (`kernel_k.py`, #198:
     `block_fragment(la, lc)` and `block_fragment(lb, ld)`), so it is a
     function of the quartet `am = (la, lc, lb, ld)` (Hermite order, bra
     first: `spec.py`, #196), and several K classes share one `(lp, lq)`.
     A `KClassLaunch` is keyed on `am`.
  2. THE KET SET. J's plan sweeps unique pairs `a >= b` on both sides, the
     ket's off-diagonal weight `w = 2` folded into `D_q`: J's symmetry is
     INSIDE a pair, so every bra walks the same ket list. K has no symmetry
     inside a pair (each record holds one output and one summed index); its
     one symmetry is the bra-ket swap (ao/k.py, "THE TWO SWEEPS"), and the
     unique sweep it gives hands each bra shell pair `(a, c)` a TRIANGULAR
     ket set (`b < a`, or `b = a` and `d <= c`) that one per-class ket list
     cannot hold. So the K plan is the ORDERED sweep --
     `ao.k.exchange_matrix_reference`'s set: bras and kets every ORDERED
     shell pair, every quadruple once at weight 1, nothing mirrored -- the
     exact analogue of J's plan, which uses no bra-ket symmetry either. It
     visits every orbit of the unique sweep twice (`n_quartets` below
     counts it); the bra-ket swap is left unused, as it is in J.
  3. THE BRA BLOCKS. `ClassLaunch.bra_blocks` is eq. 20's map: the bra rows
     a back-transform turns into `J[a][b]`, mirrored. K has no eq. 20 (the
     digestion is in the kernel, stages 2 and 3 fused) and its output block
     `K[a][b]` takes `a` from the bra and `b` from the KET: here
     `bra_blocks` is `(a, c, start, stop)`, the bra shell pair's rows, and
     the scatter is keyed by both sides.
  4. THE KEY'S FACTORS. J's ket key IS the screen's ket factor, `G * Dmax`,
     the density riding the ket (`ao.j._Ket.bound`; ao/staged.py checks it
     bit for bit). K's density factor `Dm = max|D|` is GLOBAL -- `D_cd`
     straddles the records (ao/k.py, "THE KEY THE J ENGINE USES IS
     UNAVAILABLE FOR K") -- so it is carried once, on `KLaunchPlan.dm`, and
     the screen is ao.k's `(I_ac * I_bd) * Dm < tau`, associated as ao.k
     associates it. The ket key is INJECTED (`key(b, d, I_bd) -> float`,
     ao.k's `KetKey`); the sound one is `I_bd` alone, and `walk_k` refuses
     (`check_sorted`) a ket list that is not non-increasing in `G`, the
     factor the `break` reads -- so the J key misapplied is refused, not run.

  Plus a fifth, not a shape but a dependency: J's kernels are built here
  (`kernel.build_pq_kernels`); K's builder (`kernel_k`, #198) is not this
  plan's to import, so it arrives INJECTED as `build(spec) -> kernels`, one
  call per `Spec` per plan. Without it a class carries no kernels, and its
  `spec` names what would be built.

  Why not outcome 1 (`ClassLaunch` generalised): a `ClassLaunch` carrying a
  K class would be accepted by ao/staged.py's J driver -- one ket upload per
  `lq`, one `J_p` per `lp`, eq. 20 over `bra_blocks` -- and run as J. Every
  field it would need changed (the identity, the block map's meaning, the
  ket set, the key's factors) changes what that consumer reads; generalising
  it is a second type wearing the first's name. Outcome 3 did not occur:
  nothing of m6's plan moved.

THE RECORDS. `PairRec(i, j, prim, L, G, key)` with `(i, j)` an ORDERED shell
pair -- `(a, c)` on the bra, `(b, d)` on the ket -- and `prim` the index into
`ao.j.epair(shells[i], shells[j]).recs` (`primitive_pairs` order, i outer).
`G` is the injected `bound(epair) -> per-record I` (tests pass
`functools.partial(ao.k.record_bounds, pq=pq)`), the SAME function on both
sides (rows over the first shell, the output index). A bra's `key` is `G`;
a ket's is `key(b, d, G)`. Bras stay in pair order (`a` outer, `c` inner,
`prim` ascending): the kernel maps one warp per bra record BY INDEX.

THE CLASSES. One `KClassLaunch` per `am = (la, lc, lb, ld)` with records on
both sides, sorted by `am`; each ket class `(lb, ld)` is sorted ONCE,
descending by `key`, stably, and every class with that ket class carries the
same list. `tau` is carried, validated; the plan itself skips nothing.

THE WALK (`walk_k`), the plan in Python (#142's oracle, for K): per class,
per bra record, down the class's kets -- `order="sorted"` breaks at the
first `(G_bra * G_ket) * dm < tau`, `"unsorted"` continues past it -- each
surviving pair contributing ao.k's own staged contraction of that one record
pair (`ao.k._k_quartet`, the host's arithmetic) to the block `K[a][b]`, and
every element summed with `math.fsum`. So break and continue give the same
`K` to 0 when they skip the same set, and against `ao.k.exchange_matrix` /
`exchange_matrix_reference` it agrees to rounding (the unique sweep's
mirrored `X^T` is the ordered sweep's partner block computed directly). The
result is `ao.k.KScreened`; its `scale` is counted FROM THE PLAN (ordered
record pairs covering each element), and equals ao.k's `R(a) R(b)`. At
`tau = 0` no screen test is evaluated. `plan` must have been planned from
the same `basis` and `D` (`dm` is recomputed and checked).

The L <= 2 RULE of milestone 10 is the tests' (test/test_launch_k.py): the
plan evaluates no quartet; the walk evaluates every class it is handed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace

from felizjueves.ao.j import EPair, _density, _tau, epair
from felizjueves.ao.k import (
    PQ,
    KetKey,
    KScreened,
    _dcd,
    _dm,
    _k_quartet,
    record_bounds,
)
from felizjueves.ao.shell import Basis
from felizjueves.ir import Kernel
from felizjueves.launch import PairRec, _batches_for, _nonneg, _sorted_desc
from felizjueves.spec import Operator, Spec, Target

__all__ = ["KBound", "KBuild", "KClassLaunch", "KLaunchPlan", "ORDERS",
           "exchange_launched", "k_class_spec", "k_launches", "walk_k"]

KBound = Callable[[EPair], Sequence[float]]  # one I >= 0 per record
KBuild = Callable[[Spec], Sequence[Kernel]]
Am = tuple[int, int, int, int]

ORDERS = ("sorted", "unsorted")


def k_class_spec(am: Am, n_batches: int = 1) -> Spec:
    """The generation key of K class `am = (la, lc, lb, ld)` (Hermite order,
    bra first: `spec.py`, #196)."""
    return Spec(Operator.COULOMB, 0, tuple(am), Target.K, (), n_batches)


@dataclass(frozen=True)
class KClassLaunch:
    """Every kernel of one K class `am = (la, lc, lb, ld)`, over its bra
    records `(a, c)` in pair order and its ket records `(b, d)` sorted
    descending by `key`. `bra_blocks` is `(a, c, start, stop)` per bra shell
    pair: `bras[start:stop]`. `kernels` is `()` when no builder was
    injected (module docstring)."""

    am: Am
    n_batches: int
    bras: tuple[PairRec, ...]
    kets: tuple[PairRec, ...]
    kernels: tuple[Kernel, ...]
    bra_blocks: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not self.bras or not self.kets:
            raise ValueError(
                f"K class {self.am} has {len(self.bras)} bra and "
                f"{len(self.kets)} ket records: a class with records on one "
                "side only is absent, not empty")
        if any(r.L != self.lp for r in self.bras) or any(
                r.L != self.lq for r in self.kets):
            raise ValueError(f"a record outside K class {self.am}")

    @property
    def lp(self) -> int:
        return self.am[0] + self.am[1]

    @property
    def lq(self) -> int:
        return self.am[2] + self.am[3]

    @property
    def L(self) -> int:
        return self.lp + self.lq

    @property
    def spec(self) -> Spec:
        return k_class_spec(self.am, self.n_batches)


@dataclass(frozen=True)
class KLaunchPlan:
    """`k_launches()`'s value: the K class launches in `am` order, and what
    they were planned with -- `dm = max|D|`, the screen's density factor.
    `refused` is `(la, lc, lb, ld, n_batches)` per class built at one batch
    (`strict=False`)."""

    classes: tuple[KClassLaunch, ...]
    tau: float
    dm: float
    n_batches: int
    refused: tuple[tuple[int, int, int, int, int], ...]

    def __iter__(self) -> Iterator[KClassLaunch]:
        return iter(self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def n_quartets(self) -> int:
        """(bra record, ket record) pairs the plan covers, unscreened."""
        return sum(len(c.bras) * len(c.kets) for c in self.classes)


def k_launches(basis: Basis, D: object, *, bound: KBound, key: KetKey,
               tau: float, n_batches: int = 1, strict: bool = True,
               build: KBuild | None = None) -> KLaunchPlan:
    """The launch plan of `K` for `basis` and the symmetric density `D`
    (module docstring): `bound(epair) -> I per record` and the ket sort
    `key(b, d, I_bd)` injected; `build(spec)` the K kernel builder, or
    none."""
    rows = _density(basis, D)
    tau = _tau(tau)
    dm = _dm(rows)
    shells = basis.shells
    n_sh = len(shells)

    bras: dict[tuple[int, int], list[PairRec]] = {}
    blocks: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    kets: dict[tuple[int, int], list[PairRec]] = {}
    for x in range(n_sh):
        for y in range(n_sh):  # every ORDERED pair, x outer
            ls = (shells[x].l, shells[y].l)
            L = ls[0] + ls[1]
            ep = epair(shells[x], shells[y])
            gs = tuple(bound(ep))
            if len(gs) != len(ep.recs):
                raise ValueError(f"bound gave {len(gs)} values for the "
                                 f"{len(ep.recs)} records of pair {(x, y)}")
            gs = tuple(_nonneg(g, f"bound(pair {(x, y)})[{k}]")
                       for k, g in enumerate(gs))
            side = bras.setdefault(ls, [])
            start = len(side)
            side.extend(PairRec(x, y, k, L, g, g) for k, g in enumerate(gs))
            blocks.setdefault(ls, []).append((x, y, start, len(side)))
            kets.setdefault(ls, []).extend(
                PairRec(x, y, k, L, g,
                        _nonneg(key(x, y, g), f"key{(x, y, g)}"))
                for k, g in enumerate(gs))
    sorted_kets = {ls: tuple(_sorted_desc(rs)) for ls, rs in kets.items()}

    refused: list[tuple[int, int, int, int, int]] = []
    built: dict[Spec, tuple[Kernel, ...]] = {}  # one build per Spec per plan
    out = []
    for bls in sorted(bras):
        for kls in sorted(sorted_kets):
            am = (*bls, *kls)
            lp, lq = bls[0] + bls[1], kls[0] + kls[1]
            r: list[tuple[int, int, int]] = []
            n = _batches_for(lp, lq, n_batches, strict=strict, refused=r)
            refused.extend((*am, nb) for *_, nb in r)
            kernels: tuple[Kernel, ...] = ()
            if build is not None:
                spec = k_class_spec(am, n)
                if spec not in built:
                    built[spec] = tuple(build(spec))
                    if not built[spec]:
                        raise ValueError(f"build({spec.key()}) returned no "
                                         "kernels")
                kernels = built[spec]
            out.append(KClassLaunch(am, n, tuple(bras[bls]), sorted_kets[kls],
                                    kernels, tuple(blocks[bls])))
    return KLaunchPlan(tuple(out), tau, dm, n_batches, tuple(refused))


# --- the walk: the plan in Python ---------------------------------------------


def _non_increasing_g(recs: tuple[PairRec, ...]) -> bool:
    return all(a.G >= b.G for a, b in zip(recs, recs[1:], strict=False))


def walk_k(plan: KLaunchPlan, basis: Basis, D: object, *, pq: PQ,
           order: str = "sorted", check_sorted: bool = True) -> KScreened:
    """`K` and its screening record by walking `plan` (module docstring, "THE
    WALK"). `order="sorted"` is the kernel's `break`; `check_sorted=False`
    lets a test walk kets the `break` is not sound on and show the answer
    move."""
    if order not in ORDERS:
        raise ValueError(f"order = {order!r} is not one of {ORDERS}")
    rows = _density(basis, D)
    if _dm(rows) != plan.dm:
        raise ValueError(f"max|D| = {_dm(rows)!r} is not the plan's dm "
                         f"{plan.dm!r}: planned from another density")
    tau, dm = plan.tau, plan.dm
    screen = tau > 0.0
    shells, off = basis.shells, basis.offsets
    sides = sorted({(r.i, r.j) for c in plan for r in (*c.bras, *c.kets)})
    eps = {xy: epair(shells[xy[0]], shells[xy[1]]) for xy in sides}
    one = {(x, y, k): replace(ep, recs=(rec,))
           for (x, y), ep in eps.items() for k, rec in enumerate(ep.recs)}

    n = basis.n_ao
    terms: list[list[list[float]]] = [[[] for _ in range(n)] for _ in range(n)]
    cover = [[0] * len(shells) for _ in shells]  # (a, b) -> record pairs
    n_pairs = n_skipped = n_examined = 0
    dropped: list[float] = []
    for c in plan:
        if screen and order == "sorted" and check_sorted and (
                not _non_increasing_g(c.kets)):
            raise ValueError(
                f"K class {c.am}: kets are not non-increasing in G = I_bd, "
                "the factor the screen reads, and the kernel's screen is a "
                "break")
        for br in c.bras:
            a, cc = br.i, br.j
            bra = one[a, cc, br.prim]
            n_pairs += len(c.kets)
            for kr in c.kets:
                cover[a][kr.i] += 1
            for idx, kr in enumerate(c.kets):
                if screen:
                    n_examined += 1
                    if (br.G * kr.G) * dm < tau:
                        if order == "sorted":  # every later ket too
                            rest = c.kets[idx:]
                            n_skipped += len(rest)
                            dropped.extend((br.G * r.G) * dm for r in rest)
                            break
                        n_skipped += 1
                        dropped.append((br.G * kr.G) * dm)
                        continue
                b, d = kr.i, kr.j
                X = _k_quartet(bra, one[b, d, kr.prim], _dcd(basis, rows, cc, d),
                               pq=pq, want_scale=False)[0]
                oa, ob = off[a], off[b]
                for ii, row in enumerate(X):
                    for jj, v in enumerate(row):
                        terms[oa + ii][ob + jj].append(v)
    K = tuple(tuple(math.fsum(t) for t in r) for r in terms)
    sh_of = [x for x in range(len(shells)) for _ in range(shells[x].n_cart)]
    S = tuple(tuple(float(cover[sh_of[m]][sh_of[v]]) for v in range(n))
              for m in range(n))
    return KScreened(K, S, n_pairs, n_skipped, n_examined,
                     max(dropped, default=0.0), math.fsum(dropped))


def _schwarz(b: int, d: int, i_bd: float) -> float:
    """K's ket key: `I_bd` alone (ao/k.py, "THE SCREEN")."""
    return i_bd


def exchange_launched(basis: Basis, D: object, *, pq: PQ, tau: float,
                      n_batches: int = 1, strict: bool = True) -> KScreened:
    """`walk_k(k_launches(...), basis, D, pq=pq)` with ao.k's own bound
    (`record_bounds` through `pq`) and the Schwarz-only key -- the host
    sweep's screen, so the walk skips exactly what
    `exchange_matrix_reference(tau=)` skips."""
    def bound(ep: EPair) -> Sequence[float]:
        return record_bounds(ep, pq=pq)

    return walk_k(k_launches(basis, D, bound=bound, key=_schwarz, tau=tau,
                             n_batches=n_batches, strict=strict),
                  basis, D, pq=pq)
