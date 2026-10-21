"""The split HGP build's ordered pair stream WALKED (#406): `launch_hjk`'s
plan and `kernel_hj` / `kernel_hk`'s kernels, together, in Python, before
any card runs them.

    walk_hjk(plan, basis, D, *, run, threads=4) -> HJKWalk
    HJKWalk(X, screen, mode, visited, kernel_screened, n_launches, kernels)
    kernel_spec(screen, mode, ordered_class) -> the launch's `Spec` (#401)
    pair_record(basis, a, b, *, G)            -> kernel_f.PairRec
    kernel_pairs(launch, bra, ket)            -> the kernel's two sides

Milestone 6's launch oracle (`launch.walk`), and #286's `walk_f`, applied to
the SPLIT HGP build (milestone 14): the plan's `screen` decides whether the
walk assembles the whole Coulomb matrix (`"j"`: every launch runs
`kernel_hj`, `J_ab += D_cd (ab|cd)`) or the whole exchange matrix (`"k"`:
`kernel_hk`, `K_ac += D_bd (ab|cd)`), and its `mode` whether each launch
runs its own ordered class's kernel (`"ordered"`, the `ordered` lever) or
its orbit representative's through a runtime `perm` (`"canonical"`, the
literal). test/test_hjk_build.py holds the result to `ao/j.py`'s and
`ao/k.py`'s every-ordered-quartet sweeps and to libcint's `vj` / `vk`.

INDEPENDENT OF THE PLAN BUILDER (m6's discipline, as `walk_f`). This module
does NOT import `launch_hjk`: the plan is read by duck type (`plan.tau`,
`plan.mode`, `plan.screen`, `plan.batches`, `plan.launches`; each batch's
`la, lb, degree, pairs`; each pair's `a, b, G`; each launch's `bra, ket,
ordered, orientation` and the orientation's `rep, shells`). Nothing is
re-planned and the plan's own skip decisions (`plan.visits`) are not read.
What the walker can check from `basis` alone it CHECKS, and refuses the plan
(`ValueError`) otherwise:

  * the pairs: every ORDERED shell pair `(a, b)` of `basis` is in exactly
    one batch, of its class `(l_a, l_b)`, at the full degree `K_a K_b` (the
    split stream prunes nothing, launch_hjk "THE PAIRS");
  * the launches: every ORDERED batch pair `(I, J)` exactly once, each
    naming the ordered class `(la_I, lb_I, la_J, lb_J)`;
  * the routing: in `ordered` mode the orientation is the identity; in
    `canonical` mode its `rep` is the class `kernel_hj` / `kernel_hk`
    builds for that ordered class (`built_class`), its `shells` carry the
    ordered class onto it, and its `GROUP` index is one of the kernel's
    `orientations` -- so the `perm` passed is one the kernel deposits.

THE BRIDGE TO `Spec` (#401, #403). `launch_hjk` was written before
`Spec.engine` and names a launch's kernel by class. The walker forms the key
(`kernel_spec`): `kernel_hj.hj_class_spec(*ordered)` for J and
`kernel_hk.hk_class_spec(*ordered)` for K, with the `ordered` lever in
ordered mode. For K the stream's ordered class `(la, lb, lc, ld)` of the
ERI `(ab|cd)` IS `kernel_hk`'s `am`: K's Hermite pairing `(la, lc | lb, ld)`
names the ERI `(a c | b d)` of `K_ab += D_cd (ac|bd)`, and relabelling
`b <-> c` is the stream's `K_ac += D_bd (ab|cd)` -- the same positions
`(0, 2)` for the output and `(1, 3)` for the density. One kernel is built
per distinct `Spec` (so one per orbit in canonical mode, one per ordered
class in ordered mode) and reused across launches.

THE KERNEL'S TWO SIDES (`kernel_pairs`). A kernel runs its BUILT class: the
ordered quartet `q = (a, b, c, d)` of a launch is presented as `x`, `x[j] =
q[shells[j]]`. Every element of `hgp.orbits.GROUP` keeps the bra and ket
pairs whole (it swaps within a pair and/or the two pairs), so the kernel's
bra side is one batch's pairs, each read in the built order, and its ket
side the other's: `n_bra x n_ket` combinations, each exactly one ordered
quartet of `(I, J)`. The kernel's per-pair records are rebuilt from
`basis` (`pair_record`, `kernel_f`'s staged record in `ao/hgp_drive.py`'s
expressions), each carrying the plan's `G` of the ORDERED pair it is (the
plan bounds every ordered pair, `(b, a)` as well as `(a, b)`).

THE KERNEL RUNS INJECTED (`run`), so this module reaches no interpreter, no
backend and no `reference/`:

    run(kernel, grid, block, args) -> {name: flat sequence}

with `args` every parameter of the kernel by name, arrays flat and
row-major as plain tuples, and the output matrix (`J` or `K`) carried from
launch to launch; `run` returns at least that matrix. Nothing is closed or
mirrored after the last launch: the split kernels write each element from
its own blocks (kernel_hj, "THE DEPOSIT").

THE IN-KERNEL SCREEN, PREDICTED (`kernel_screened`). For every visited
ordered quartet the walker evaluates the kernel's one-block bound in the
kernel's operation order -- `bound = G_bra (G_ket dmax)`, `G_bra` / `G_ket`
the kernel side's records, `dmax` the `Dsh` element of the density pair
(`(c, d)` for J, `(b, d)` for K) -- and records the quartet when `bound <
tau`. It is a PREDICTION of a skip the kernel makes on its own; the test
holds every predicted skip to the exact ERIs and the walked matrix to the
sweep restricted to the kept quartets.

Plain tuples and floats; `D` is `ao.j`'s density contract (square,
symmetric, finite), restated here rather than borrowed (#328).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from felizjueves import kernel_f as KF
from felizjueves import kernel_hj as HJ
from felizjueves import kernel_hk as HK
from felizjueves.ao.shell import Basis
from felizjueves.hgp.orbits import GROUP, apply
from felizjueves.ir import Kernel
from felizjueves.spec import Spec

Quartet = tuple[int, int, int, int]  # an ORDERED shell quartet (a, b, c, d)
ShellPair = tuple[int, int]
Run = Callable[[Kernel, int, tuple[int, int], Mapping[str, object]],
               Mapping[str, Sequence[float]]]

THREADS = 4  # the default block width; the answer does not depend on it
MODES = ("canonical", "ordered")  # launch_hjk's, restated (a test pins them)
SCREENS = ("j", "k")
IDENTITY = (0, 1, 2, 3)

# Per screen: the ordered positions whose density block the deposit reads,
# the output matrix's parameter name, and the builder's module.
DENSITY: dict[str, ShellPair] = {"j": HJ.COULOMB_DENSITY, "k": HK.K_DENSITY}
OUTPUT: dict[str, str] = {"j": HJ.J_OUT, "k": HK.K_OUT}
_BUILDER = {"j": HJ, "k": HK}


@dataclass(frozen=True)
class HJKWalk:
    """The walk's result: `X` the whole `J` (screen `"j"`) or `K` (`"k"`),
    `n_ao x n_ao`; `visited` every ordered quartet a launch met (all
    `n_shell^4` of them, each once); `kernel_screened` the subset the
    kernel's one-block screen skips (module docstring); `kernels` the
    distinct kernel names built, sorted."""

    X: tuple[tuple[float, ...], ...]
    screen: str
    mode: str
    visited: frozenset[Quartet]
    kernel_screened: frozenset[Quartet]
    n_launches: int
    kernels: tuple[str, ...]

    @property
    def computed(self) -> frozenset[Quartet]:
        """The ordered quartets whose ERIs reach `X`."""
        return self.visited - self.kernel_screened


def _refuse(msg: str) -> ValueError:
    return ValueError(f"walk_hjk refuses the plan: {msg}")


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
                raise ValueError(f"D is not symmetric: D[{i}][{j}] != "
                                 f"D[{j}][{i}]")
    return rows


def kernel_spec(screen: str, mode: str, ordered: Quartet) -> Spec:
    """The `Spec` of the kernel serving the ordered class `ordered` (module
    docstring, "THE BRIDGE"): `Target.J` / `Target.K` on `Engine.HGP`, the
    `ordered` lever in ordered mode."""
    if screen not in SCREENS:
        raise ValueError(f"screen = {screen!r} is not one of {SCREENS}")
    if mode not in MODES:
        raise ValueError(f"mode = {mode!r} is not one of {MODES}")
    make = HJ.hj_class_spec if screen == "j" else HK.hk_class_spec
    return make(*ordered, ordered=mode == "ordered")


def pair_record(basis: Basis, a: int, b: int, *, G: float) -> KF.PairRec:
    """`kernel_f`'s staged record of the ordered shell pair `(a, b)`, every
    primitive pair (`i` outer), each field by `ao/hgp_drive.py`'s own
    expressions (`P = (x A + y B)/p`, `X = P - A`, `U = c_a c_b (pi/p)^{3/2}
    exp(-x y / p |AB|^2)`)."""
    sa, sb = basis.shells[a], basis.shells[b]
    A, B = sa.centre, sb.centre
    AB = tuple(A[d] - B[d] for d in range(3))
    ab2 = AB[0] ** 2 + AB[1] ** 2 + AB[2] ** 2
    recs = []
    for i, x in enumerate(sa.exponents):
        for j, y in enumerate(sb.exponents):
            p = x + y
            P = tuple((x * A[d] + y * B[d]) / p for d in range(3))
            U = (sa.coefficients[i] * sb.coefficients[j]
                 * (math.pi / p) ** 1.5 * math.exp(-x * y / p * ab2))
            recs.append(KF.PrimRec(P, tuple(P[d] - A[d] for d in range(3)),
                                   p, U))
    return KF.PairRec(AB, float(G), tuple(recs), (a, b),
                      (basis.offsets[a], basis.offsets[b]))


def kernel_pairs(shells: Sequence[int], bra: Sequence[ShellPair],
                 ket: Sequence[ShellPair]
                 ) -> tuple[list[ShellPair], list[ShellPair]]:
    """The kernel's bra and ket sides (module docstring, "THE KERNEL'S TWO
    SIDES"): position `j` of the built class takes the ordered quartet's
    position `shells[j]`; each side is one batch's pairs in that order."""
    s = tuple(shells)
    side = {0: bra, 2: ket}

    def built(j0: int, j1: int) -> list[ShellPair]:
        base = min(s[j0], s[j1]) // 2 * 2
        if {s[j0], s[j1]} != {base, base + 1}:
            raise ValueError(f"orientation {s} splits a shell pair")
        return [(P[s[j0] - base], P[s[j1] - base]) for P in side[base]]

    return built(0, 1), built(2, 3)


def _check_pairs(plan: object, basis: Basis) -> dict[ShellPair, float]:
    sh = basis.shells
    ns = len(sh)
    want = {(a, b) for a in range(ns) for b in range(ns)}
    seen: list[ShellPair] = []
    G: dict[ShellPair, float] = {}
    for bi, batch in enumerate(plan.batches):
        if not batch.pairs:
            raise _refuse(f"batch {bi} is empty")
        for p in batch.pairs:
            P = (p.a, p.b)
            seen.append(P)
            if P not in want:
                raise _refuse(f"batch {bi}: {P} is not a shell pair of the "
                              "basis")
            if (sh[p.a].l, sh[p.b].l) != (batch.la, batch.lb):
                raise _refuse(f"batch {bi} is class {(batch.la, batch.lb)}, "
                              f"its pair {P} is {(sh[p.a].l, sh[p.b].l)}")
            n = len(sh[p.a].exponents) * len(sh[p.b].exponents)
            if batch.degree != n:
                raise _refuse(f"batch {bi}: pair {P} has K_a K_b = {n} "
                              f"primitive pairs, the batch's degree is "
                              f"{batch.degree} (the split stream prunes "
                              "nothing)")
            G[P] = float(p.G)
    if len(seen) != len(set(seen)) or set(seen) != want:
        missing = sorted(want - set(seen))
        twice = sorted({P for P in seen if seen.count(P) > 1})
        raise _refuse(f"the pairs are not every ordered shell pair once "
                      f"(missing {missing}, twice {twice})")
    return G


def _check_launches(plan: object, screen: str, mode: str) -> list[Spec]:
    batches = plan.batches
    nb = len(batches)
    want = {(i, j) for i in range(nb) for j in range(nb)}
    got = [(L.bra, L.ket) for L in plan.launches]
    if len(got) != len(set(got)) or set(got) != want:
        bad = sorted(x for x in set(got) if got.count(x) > 1 or x not in want)
        raise _refuse(f"the launches are not every ordered batch pair once "
                      f"(missing {sorted(want - set(got))}, extra or twice "
                      f"{bad})")
    mod = _BUILDER[screen]
    specs = []
    for L in plan.launches:
        b, k = batches[L.bra], batches[L.ket]
        cls = (b.la, b.lb, k.la, k.lb)
        if tuple(L.ordered) != cls:
            raise _refuse(f"launch ({L.bra}, {L.ket}) names class "
                          f"{tuple(L.ordered)}, not {cls}")
        o = L.orientation
        spec = kernel_spec(screen, mode, cls)
        shells, rep = tuple(o.shells), tuple(o.rep)
        if mode == "ordered" and (shells != IDENTITY or rep != cls):
            raise _refuse(f"launch ({L.bra}, {L.ket}) is ordered mode but "
                          f"carries orientation {shells} onto {rep}")
        if (rep != mod.built_class(spec) or apply(shells, cls) != rep
                or GROUP.index(shells) not in mod.orientations(spec)):
            raise _refuse(f"launch ({L.bra}, {L.ket}): orientation {shells} "
                          f"onto {rep} is not how {mod.__name__} serves "
                          f"{cls} ({mode})")
        specs.append(spec)
    return specs


def walk_hjk(plan: object, basis: Basis, D: object, *, run: Run,
             threads: int = THREADS) -> HJKWalk:
    """Walk `plan` (module docstring): check it against `basis`, run every
    launch through the injected `run` on the screen's kernel, and record
    every ordered quartet's fate. `plan` must have been planned from `basis`
    and `D`."""
    screen, mode = plan.screen, plan.mode
    if screen not in SCREENS or mode not in MODES:
        raise _refuse(f"screen {screen!r} / mode {mode!r} is not one of "
                      f"{SCREENS} / {MODES}")
    rows = _density(basis, D)
    tau = float(plan.tau)
    G = _check_pairs(plan, basis)
    specs = _check_launches(plan, screen, mode)
    sh, n_sh, n_ao = basis.shells, len(basis.shells), basis.n_ao
    mod, out_name = _BUILDER[screen], OUTPUT[screen]
    d0, d1 = DENSITY[screen]
    side = [[(p.a, p.b) for p in b.pairs] for b in plan.batches]

    Dsh = KF.density_blocks(rows, basis.offsets, [s.n_cart for s in sh])
    D_flat = tuple(x for r in rows for x in r)
    records: dict[ShellPair, KF.PairRec] = {}

    def record(P: ShellPair) -> KF.PairRec:
        if P not in records:
            records[P] = pair_record(basis, *P, G=G[P])
        return records[P]

    kernels: dict[str, Kernel] = {}
    X: tuple[float, ...] = (0.0,) * (n_ao * n_ao)
    visited: list[Quartet] = []
    kscreen: set[Quartet] = set()
    for L, spec in zip(plan.launches, specs, strict=True):
        shells = tuple(L.orientation.shells)
        axes = tuple(shells.index(i) for i in range(4))  # q[i] = x[axes[i]]
        kb, kk = kernel_pairs(shells, side[L.bra], side[L.ket])
        bras, kets = [record(P) for P in kb], [record(P) for P in kk]
        for br in bras:
            for kt in kets:
                x = (*br.sh, *kt.sh)
                q = tuple(x[axes[i]] for i in range(4))
                visited.append(q)  # type: ignore[arg-type]
                dmax = Dsh[q[d0] * n_sh + q[d1]]
                if br.G * (kt.G * dmax) < tau:
                    kscreen.add(q)  # type: ignore[arg-type]
        name = mod.hj_name(spec) if screen == "j" else mod.hk_name(spec)
        if name not in kernels:
            kernels[name] = (HJ.build_hj_kernel(spec) if screen == "j"
                             else HK.build_hk_kernel(spec))
        kernel = kernels[name]
        pb, pk = KF.pack(bras), KF.pack(kets)
        args = {
            "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree,
            "k_cd": pk.degree, "n_sh": n_sh, "n_ao": n_ao,
            HJ.PERM: GROUP.index(shells), "tau": tau,
            "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G,
            "bra_sh": pb.sh, "bra_ao": pb.ao, "ket_prim": pk.prim,
            "CD": pk.sep, "G_cd": pk.G, "ket_sh": pk.sh, "ket_ao": pk.ao,
            "Dsh": Dsh, "D": D_flat, out_name: X,
        }
        grid, block = mod.launch_shape(pb.n, threads)
        out = run(kernel, grid, block,
                  {p.name: args[p.name] for p in kernel.params})
        X = tuple(float(v) for v in out[out_name])
        if len(X) != n_ao * n_ao:
            raise ValueError(f"run returned {out_name} of {len(X)} elements, "
                             f"not {n_ao * n_ao}")
    every = {(a, b, c, d) for a in range(n_sh) for b in range(n_sh)
             for c in range(n_sh) for d in range(n_sh)}
    if len(visited) != len(every) or set(visited) != every:
        raise _refuse("the launches do not visit every ordered quartet once")
    square = tuple(X[i * n_ao:(i + 1) * n_ao] for i in range(n_ao))
    return HJKWalk(square, screen, mode, frozenset(visited),
                   frozenset(kscreen), len(plan.launches),
                   tuple(sorted(kernels)))
