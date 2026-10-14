"""The fused Fock build's pair stream WALKED (#286): `launch_f`'s plan and
`kernel_f`'s kernel, together, in Python, before any card runs them.

    walk_f(plan, basis, D, *, run, threads=4) -> FWalk
    FWalk(F, launched, kernel_screened, prelaunch_screened, deleted,
          n_prim_pruned, n_launches)
    pair_record(basis, a, b, *, G, prims)     -> kernel_f.PairRec
    quartets(P_set, Q_set, *, diagonal)       the unordered quartets met

Milestone 6's launch oracle (`launch.walk`) applied to the fused kernel.
This is the first place the symmetry (the canonical half and the degeneracy
factor, #283), the screens (pruning and eq. 29's pre-launch screen, #285;
the six-block in-kernel skip, #283), the batching (#285) and the digestion
(#283) run together; `test/test_fock_build.py` holds the result to
`ao.fock`'s every-ordered-quartet sweep and to libcint's `vj - vk/2`.

INDEPENDENT OF THE PLAN BUILDER (m6's discipline). This module does NOT
import `launch_f` -- not even for its types: the plan is read by duck type
(`plan.tau`, `plan.batches`, `plan.launches`, `plan.screened`,
`plan.deleted`; each batch's `la, lb, degree, pairs`; each pair's `a, b,
G, prims`; each launch's `bra, ket, spec`). Nothing is re-planned and no
screen decision is asked of the builder. What the walker can check from
`basis` alone it CHECKS, and refuses the plan (`ValueError`) otherwise:

  * the pairs: every unordered shell pair of `basis`, written canonically
    (`la <= lb`, the lower index first on a tie), is in exactly one batch
    or in `plan.deleted` -- none missing, none twice;
  * each batch: its pairs are of its class `(la, lb)`, every one with
    exactly `degree` kept primitive pairs, each a distinct valid index into
    the ordered shell pair's `K_a K_b` primitive pairs;
  * the batch pairs: `launches` and `screened` together are every `(I, J)`
    with `I <= J` over the batches, each exactly once;
  * each launch names `Target.F` and the class `(la_I, lb_I, la_J, lb_J)`.

Then the unique quartets -- unordered pairs `{P, Q}` of canonical shell
pairs -- are partitioned three ways and the partition is asserted: those of
a launch (`launched`), those of a batch pair removed before launch
(`prelaunch_screened`), and those touching a deleted pair (`deleted`).

THE RECORDS ARE REBUILT FROM `basis` (`pair_record`): per kept primitive
pair index `k = i K_b + j` (`ao.pairs.primitive_pairs`' order, `i` outer),
`P`, `X = P - A`, `zeta = alpha + beta` and `U = c_a c_b (pi/zeta)^{3/2}
exp(-alpha beta / zeta |AB|^2)` -- `kernel_f`'s staged record (its module
docstring, "THE STAGED RECORD"), from the shells' own exponents,
coefficients and centres. The plan contributes only the indices and each
pair's bound `G` (the kernel's screen input), as `launch.walk` reads the
plan's `G` and `key`.

THE KERNEL RUNS INJECTED (`run`), so this module reaches no interpreter, no
backend and no `reference/` (Boys belongs to whatever executes): per launch
the walker builds the class's kernel once per call (`kernel_f.
build_f_kernel`), packs the two sides (`kernel_f.pack`), and calls

    run(kernel, grid, block, args) -> {name: flat sequence}

with `args` every parameter by name (`kernel_f.params_of`), arrays flat
and row-major as plain tuples, and `F` the running two-electron matrix
carried from launch to launch; `run` returns at least `F`. After the last
launch the host closes once: `F <- F + F^T` (`kernel_f.close`). `diag = 1`
exactly when the launch is a batch with itself (`bra == ket`), as
`kernel_f` requires.

THE IN-KERNEL SCREEN, PREDICTED (`kernel_screened`). For each launched
quartet the walker evaluates the six-block bound the kernel documents --
`bound = G_ab (G_cd dmax)`, `dmax` the largest `|w_b| Dsh_b` over
`kernel_f.BLOCKS` with `kernel_f.WEIGHTS` -- in the launch's orientation
(the bra batch's pair first) and in the kernel's operation order, and
records the quartet when `bound < tau`. It is a PREDICTION of a skip the
kernel makes on its own; the test holds it to the answer (the kept
quartets alone reproduce the walk's `F`) and holds every skipped quartet,
of all three kinds, to the exact ERIs.

Plain tuples and floats; `D` is `ao.j`'s density contract (square,
symmetric, finite), shared.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from felizjueves import kernel_f as KF
from felizjueves.ao.j import _density
from felizjueves.ao.shell import Basis
from felizjueves.ir import Kernel
from felizjueves.spec import Target

ShellPair = tuple[int, int]
Quartet = tuple[ShellPair, ShellPair]  # (P, Q), P <= Q, each canonical
Run = Callable[[Kernel, int, tuple[int, int], Mapping[str, object]],
               Mapping[str, Sequence[float]]]

THREADS = 4  # the default block width; the answer does not depend on it


@dataclass(frozen=True)
class FWalk:
    """The walk's result: `F` closed (`n_ao x n_ao`), and the unique
    quartets by fate (module docstring). `kernel_screened` is a subset of
    `launched`; the other three partition every unique quartet."""

    F: tuple[tuple[float, ...], ...]
    launched: frozenset[Quartet]
    kernel_screened: frozenset[Quartet]
    prelaunch_screened: frozenset[Quartet]
    deleted: frozenset[Quartet]
    n_prim_pruned: int
    n_launches: int

    @property
    def computed(self) -> frozenset[Quartet]:
        """The quartets whose ERIs reach `F`."""
        return self.launched - self.kernel_screened

    @property
    def skipped(self) -> frozenset[Quartet]:
        """Every quartet whose ERIs do not reach `F`, all three screens."""
        return self.kernel_screened | self.prelaunch_screened | self.deleted


def _canonical(basis: Basis, i: int, j: int) -> ShellPair:
    sh = basis.shells
    return (i, j) if (sh[i].l, i) <= (sh[j].l, j) else (j, i)


def _quartet(P: ShellPair, Q: ShellPair) -> Quartet:
    return (P, Q) if P <= Q else (Q, P)


def quartets(Ps: Sequence[ShellPair], Qs: Sequence[ShellPair], *,
             diagonal: bool) -> list[Quartet]:
    """The unique quartets a batch pair meets: every `{P, Q}`, `P` from the
    first side and `Q` from the second; on the diagonal (one batch with
    itself) each unordered pair once, `P = Q` included."""
    if diagonal:
        return [_quartet(Ps[i], Ps[j]) for i in range(len(Ps))
                for j in range(i, len(Ps))]
    return [_quartet(P, Q) for P in Ps for Q in Qs]


def pair_record(basis: Basis, a: int, b: int, *, G: float,
                prims: Iterable[int]) -> KF.PairRec:
    """`kernel_f`'s staged record of the ordered shell pair `(a, b)`, from
    `basis`'s shells (module docstring, "THE RECORDS"): the kept primitive
    pairs `prims`, in that order, indices `i K_b + j`."""
    sa, sb = basis.shells[a], basis.shells[b]
    A, B = sa.centre, sb.centre
    AB = tuple(u - v for u, v in zip(A, B, strict=True))
    ab2 = sum(x * x for x in AB)
    kb = len(sb.exponents)
    recs = []
    for k in prims:
        i, j = divmod(k, kb)
        x, ca = sa.exponents[i], sa.coefficients[i]
        y, cb = sb.exponents[j], sb.coefficients[j]
        p = x + y
        P = tuple((x * u + y * v) / p for u, v in zip(A, B, strict=True))
        U = ca * cb * (math.pi / p) ** 1.5 * math.exp(-x * y / p * ab2)
        recs.append(KF.PrimRec(P, tuple(u - v for u, v in zip(P, A,
                                                             strict=True)),
                               p, U))
    return KF.PairRec(AB, float(G), tuple(recs), (a, b),
                      (basis.offsets[a], basis.offsets[b]))


def _refuse(msg: str) -> ValueError:
    return ValueError(f"walk_f refuses the plan: {msg}")


def _check_pairs(plan: object, basis: Basis) -> int:
    """The pair and batch checks (module docstring); returns the primitive
    pairs pruned, counted from `basis`."""
    sh = basis.shells
    want = {_canonical(basis, i, j)
            for i in range(len(sh)) for j in range(i, len(sh))}
    seen: list[ShellPair] = [tuple(x) for x in plan.deleted]
    n_pruned = sum(len(sh[a].exponents) * len(sh[b].exponents)
                   for a, b in seen if (a, b) in want)
    for bi, batch in enumerate(plan.batches):
        if not batch.pairs:
            raise _refuse(f"batch {bi} is empty")
        for p in batch.pairs:
            P = (p.a, p.b)
            seen.append(P)
            if P not in want:
                raise _refuse(f"batch {bi}: {P} is not a canonical shell pair "
                              "of the basis")
            if (sh[p.a].l, sh[p.b].l) != (batch.la, batch.lb):
                raise _refuse(f"batch {bi} is class {(batch.la, batch.lb)}, "
                              f"its pair {P} is {(sh[p.a].l, sh[p.b].l)}")
            n = len(sh[p.a].exponents) * len(sh[p.b].exponents)
            prims = tuple(p.prims)
            if (len(prims) != batch.degree or len(set(prims)) != len(prims)
                    or any(type(k) is not int or not 0 <= k < n
                           for k in prims)):
                raise _refuse(f"batch {bi}: pair {P}'s primitive pairs "
                              f"{prims} are not {batch.degree} distinct "
                              f"indices into its {n}")
            n_pruned += n - len(prims)
    if len(seen) != len(set(seen)) or set(seen) != want:
        missing = sorted(want - set(seen))
        twice = sorted({P for P in seen if seen.count(P) > 1})
        raise _refuse(f"the pairs are not every canonical shell pair once "
                      f"(missing {missing}, twice {twice})")
    return n_pruned


def _check_batch_pairs(plan: object) -> None:
    batches = plan.batches
    nb = len(batches)
    want = {(i, j) for i in range(nb) for j in range(i, nb)}
    got = [(L.bra, L.ket) for L in plan.launches]
    got += [tuple(x) for x in plan.screened]
    if len(got) != len(set(got)) or set(got) != want:
        bad = sorted(x for x in set(got) if got.count(x) > 1 or x not in want)
        raise _refuse(f"launches and screened are not every batch pair I <= J "
                      f"once (missing {sorted(want - set(got))}, extra or "
                      f"twice {bad})")
    for L in plan.launches:
        b, k = batches[L.bra], batches[L.ket]
        cls = (b.la, b.lb, k.la, k.lb)
        if L.spec.target is not Target.F or tuple(L.spec.am) != cls:
            raise _refuse(f"launch ({L.bra}, {L.ket}) names {L.spec}, not "
                          f"Target.F {cls}")


def walk_f(plan: object, basis: Basis, D: object, *, run: Run,
           threads: int = THREADS) -> FWalk:
    """Walk `plan` (module docstring): check it against `basis`, run every
    launch through the injected `run`, close `F`, and record every unique
    quartet's fate. `plan` must have been planned from `basis` and `D`."""
    rows = _density(basis, D)
    tau = float(plan.tau)
    n_pruned = _check_pairs(plan, basis)
    _check_batch_pairs(plan)
    sh, n_sh, n_ao = basis.shells, len(basis.shells), basis.n_ao
    batches = plan.batches
    side = [[(p.a, p.b) for p in b.pairs] for b in batches]

    # The three-way partition of every unique quartet.
    deleted_pairs = {tuple(x) for x in plan.deleted}
    canon = sorted({_canonical(basis, i, j)
                    for i in range(n_sh) for j in range(i, n_sh)})
    every = {_quartet(canon[i], canon[j]) for i in range(len(canon))
             for j in range(i, len(canon))}
    deleted = frozenset(q for q in every
                        if q[0] in deleted_pairs or q[1] in deleted_pairs)
    prelaunch = frozenset(
        q for i, j in plan.screened
        for q in quartets(side[i], side[j], diagonal=i == j))
    launched: set[Quartet] = set()
    for L in plan.launches:
        launched.update(quartets(side[L.bra], side[L.ket],
                                 diagonal=L.bra == L.ket))
    parts = (launched, prelaunch, deleted)
    if (sum(map(len, parts)) != len(every)
            or set().union(*parts) != every):
        raise _refuse("launched, pre-launch screened and deleted quartets "
                      "do not partition the unique quartets")

    Dsh = KF.density_blocks(rows, basis.offsets, [s.n_cart for s in sh])
    wabs = [abs(KF.WEIGHTS.of(b)) for b in KF.BLOCKS]

    def six_block(Pb: ShellPair, Gb: float, Qk: ShellPair, Gk: float) -> bool:
        s = (*Pb, *Qk)
        dmax = wabs[0] * Dsh[s[0] * n_sh + s[1]]
        for w, (u, v) in zip(wabs[1:], KF.BLOCKS[1:], strict=True):
            t = w * Dsh[s[u] * n_sh + s[v]]
            if dmax < t:
                dmax = t
        return Gb * (Gk * dmax) < tau

    records = [[pair_record(basis, p.a, p.b, G=p.G, prims=p.prims)
                for p in b.pairs] for b in batches]
    kernels: dict[tuple[int, ...], Kernel] = {}
    F: tuple[float, ...] = (0.0,) * (n_ao * n_ao)
    kscreen: set[Quartet] = set()
    D_flat = tuple(x for r in rows for x in r)
    for L in plan.launches:
        diag = int(L.bra == L.ket)
        bras, kets = records[L.bra], records[L.ket]
        for ib, br in enumerate(bras):
            for ik in range(ib if diag else 0, len(kets)):
                kt = kets[ik]
                if six_block(br.sh, br.G, kt.sh, kt.G):
                    kscreen.add(_quartet(br.sh, kt.sh))
        am = tuple(L.spec.am)
        if am not in kernels:
            kernels[am] = KF.build_f_kernel(L.spec)
        pb, pk = KF.pack(bras), KF.pack(kets)
        args = {
            "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree,
            "k_cd": pk.degree, "n_sh": n_sh, "n_ao": n_ao, "diag": diag,
            "tau": tau, "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G,
            "bra_sh": pb.sh, "bra_ao": pb.ao, "ket_prim": pk.prim,
            "CD": pk.sep, "G_cd": pk.G, "ket_sh": pk.sh, "ket_ao": pk.ao,
            "Dsh": Dsh, "D": D_flat, KF.FOCK: F,
        }
        grid, block = KF.launch_shape(pb.n, threads)
        out = run(kernels[am], grid, block,
                  {n: args[n] for n in KF.params_of(am)})
        F = tuple(float(x) for x in out[KF.FOCK])
        if len(F) != n_ao * n_ao:
            raise ValueError(f"run returned F of {len(F)} elements, not "
                             f"{n_ao * n_ao}")
    square = [F[i * n_ao:(i + 1) * n_ao] for i in range(n_ao)]
    return FWalk(KF.close(square), frozenset(launched), frozenset(kscreen),
                 prelaunch, deleted, n_pruned,
                 len(plan.launches))
