"""#352: the monotone ket key for the fused F, and the early break it buys.

The question the issue asks -- is there a ket key that makes a `break` sound
in the fused F kernel's ket loop? -- is answered YES here, with the proof in
f_levers/ket_break.py and its pieces held one by one:

  * THE PIN'S REASON IS REAL (the six-block bound itself is not monotone):
    on H2O/STO-3G a literal launch's six-block bound rises after a ket it
    screens, so a `break` on the literal stream drops a quartet the screen
    keeps; and two bras order the same two kets oppositely, so NO single ket
    order makes the six-block bound monotone for every bra of a launch. A
    sound break therefore needs a DOMINATING bound, not a better sort.
  * THE DOMINATING BOUND: `M_ab` (walk_f.dominating_factors) is at least the
    kernel's six-block `dmax` of its bra with every ket of the batch, as
    floats, on every launch of the fixtures.
  * THE LEVER'S KERNEL computes exactly the literal kernel's quartets: on a
    G-sorted batch its `F` is the literal kernel's BIT FOR BIT, for every F
    class with L <= 4, while its ket loop runs fewer iterations (the break
    fired); and on H2O/STO-3G and 6-31G whole, walked, its `F` is the literal
    walk's of the same stream bit for bit, every quartet it breaks on is one
    the six-block screen skips, and each such quartet is below `tau` in the
    EXACT ERIs (`ao.eri`).
  * THE STREAM MUST BE SORTED (the wrong-answer test): `walk_f` refuses a
    `ket-break` launch whose ket batch is not descending in `G`; forced past
    that refusal, the lever's kernel drops a quartet whose exact terms reach
    `tau` and `F` leaves the screened-sweep bound.
  * THE DEFAULT IS UNCHANGED: `f_launches` without `ket_break` is the stream
    it was, and `ket_break=True` moves only the order inside each batch and
    the launches' lever.

Every class here has L <= 4 (milestone 13's cap). Interpreter only: no
compiler, no card. H2O/6-31G is not a small input: everything on it -- the
rising-bound counterexample, the wrong-answer test, its walks -- is `slow`,
so the fast tier holds the lever on H2O/STO-3G (the walks bit for bit, the
broken quartets against the exact ERIs) and per class.
"""

from __future__ import annotations

import difflib
import itertools
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from test_fock_build import BOUND, TOL, _six_terms, members, molecule, run, sweep

from felizjueves import kernel_f as KF
from felizjueves import launch_f, levers, walk_f
from felizjueves.ao.pairs import primitive_schwarz_bounds
from felizjueves.ao.shell import Basis, Shell
from felizjueves.device import GFX1200, SM_86
from felizjueves.emit import cuda, hip
from felizjueves.f_levers import REGISTRY, hooks_for, ket_break
from felizjueves.hgp import vrr
from felizjueves.ir import Break, For, If, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.launch_f import f_launches
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

NAME = ket_break.NAME
CLASSES = [c for c in vrr.CLASSES if sum(c) <= 4]  # the milestone's cap
SLOW = pytest.mark.slow
# H2O/6-31G's walk is 9-17 s interpreted: not a small input, so `slow`
WALKS = [("h2o_sto3g", 0.1, 4), ("h2o_sto3g", 0.1, 1),
         pytest.param("h2o_631g", 1e-3, 4, marks=SLOW)]


def _spec(cls, levers=()) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(cls), Target.F, levers=levers)


def _plain(plan):
    """The same stream, every launch's lever removed: the literal kernel on
    the G-sorted order."""
    return replace(plan, launches=tuple(
        replace(L, spec=replace(L.spec, levers=())) for L in plan.launches))


@cache
def _walks(name, tau, n):
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=n, ket_break=True)
    return (walk_f.walk_f(plan, basis, D, run=run),
            walk_f.walk_f(_plain(plan), basis, D, run=run), plan)


# ---------------------------------------------------------------------------
# The vocabulary, the registry, the names restated across layers
# ---------------------------------------------------------------------------


def test_the_lever_is_f_only_registered_and_its_names_agree() -> None:
    lever = levers.VOCABULARY[NAME]
    assert lever.legal == {"f"} and lever.issue == 352
    assert "already breaks" in lever.refusals["j"]
    assert "NO_DENSITY_KEY" in lever.refusals["k"]
    assert REGISTRY[NAME] is ket_break.apply
    # restated in launch_f and walk_f, which may not import the lever
    assert launch_f.KET_BREAK == walk_f.KET_BREAK == NAME
    assert walk_f.BREAK_ARG == ket_break.BOUND
    for t in ("j", "k"):
        with pytest.raises(ValueError, match="already breaks"):
            Spec(Operator.COULOMB, 0, (0, 0, 0, 0), Target(t), levers=(NAME,))


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_the_kernel_has_one_break_in_the_ket_loop_and_one_more_param(
        cls) -> None:
    lit = KF.build_f_kernel(_spec(cls))
    k = KF.build_f_kernel(_spec(cls, (NAME,)))
    assert not hooks_for(_spec(cls, (NAME,))).is_default("screen")
    assert not any(isinstance(s, Break) for s in walk(lit.body))
    assert sum(isinstance(s, Break) for s in walk(k.body)) == 1
    loops = [s for s in walk(k.body) if isinstance(s, For)
             and s.var == KF.KET]
    assert len(loops) == 1
    assert sum(isinstance(s, Break) for s in walk(loops[0].body)) == 1
    names = [p.name for p in k.params]
    want = list(KF.params_of(cls))
    want.insert(want.index("G_ab") + 1, ket_break.BOUND)
    assert names == want
    # the break sits before the six-block screen, inside the canonical half
    ifs = [s for s in walk(loops[0].body) if isinstance(s, If)
           and any(isinstance(t, Break) for t in s.then)]
    assert len(ifs) == 1 and ifs[0].then == (Break(),)


GOLDEN = Path(__file__).resolve().parent / "golden"
GOLDEN_AM = (0, 1, 0, 1)  # both HRRs run, P = Q decided at run time
BACKENDS = {"cuda": (cuda.emit, SM_86, "cu"), "hip": (hip.emit, GFX1200, "hip")}
# Regenerate (only when the lever's text is meant to change):
#   PYTHONPATH=src python -c "from pathlib import Path; \
#   from felizjueves.device import SM_86, GFX1200; \
#   from felizjueves.emit import cuda, hip; \
#   from felizjueves.kernel_f import build_f_kernel; \
#   from felizjueves.spec import Operator, Spec, Target; \
#   k = build_f_kernel(Spec(Operator.COULOMB, 0, (0, 1, 0, 1), Target.F, \
#   levers=('ket-break',))); \
#   Path('test/golden/cuda/f_ket_break/f_0101.cu').write_text(cuda.emit(k, SM_86)); \
#   Path('test/golden/hip/f_ket_break/f_0101.hip').write_text(hip.emit(k, GFX1200))"


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_the_ket_break_goldens_are_unchanged(backend) -> None:
    emit, arch, ext = BACKENDS[backend]
    k = KF.build_f_kernel(_spec(GOLDEN_AM, (NAME,)))
    path = GOLDEN / backend / "f_ket_break" / f"{k.name}.{ext}"
    assert emit(k, arch) == path.read_text()


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_the_lever_text_only_adds_lines_to_the_literal(backend, cls) -> None:
    """The literal unit's every line survives, in order: the lever adds the
    `M_ab` parameter, its provenance comment, `Mb` and the break."""
    emit, arch, _ = BACKENDS[backend]
    lit = emit(KF.build_f_kernel(_spec(cls)), arch).splitlines()
    brk = emit(KF.build_f_kernel(_spec(cls, (NAME,))), arch).splitlines()
    ops = difflib.SequenceMatcher(a=lit, b=brk, autojunk=False).get_opcodes()
    assert all(op in ("equal", "insert") for op, *_ in ops)
    added = [ln.strip() for op, _, _, j1, j2 in ops if op == "insert"
             for ln in brk[j1:j2]]
    assert "break;" in added and "double Mb = M_ab[bra];" in added
    assert "const double* M_ab," in added


# ---------------------------------------------------------------------------
# The pin's reason: the six-block bound itself is not monotone
# ---------------------------------------------------------------------------


def _bounds(basis, D, plan):
    """Per launch, per bra, the kernel's six-block bound of every ket, in
    the kernel's operation order (walk_f's `six_block`)."""
    sh, n_sh = basis.shells, len(basis.shells)
    Dsh = KF.density_blocks(np.asarray(D).tolist(), basis.offsets,
                            [s.n_cart for s in sh])
    w = [abs(KF.WEIGHTS.of(b)) for b in KF.BLOCKS]

    def dmax(P, Q):
        s = (*P, *Q)
        return max(wi * Dsh[s[u] * n_sh + s[v]]
                   for wi, (u, v) in zip(w, KF.BLOCKS, strict=True))

    out = {}
    for L in plan.launches:
        bras, kets = plan.batches[L.bra].pairs, plan.batches[L.ket].pairs
        out[(L.bra, L.ket)] = [
            [p.G * (q.G * dmax(p.shell_pair, q.shell_pair)) for q in kets]
            for p in bras]
    return out, Dsh


@pytest.mark.slow  # H2O/6-31G's plan against its exact ERIs: 17 s
def test_the_literal_streams_six_block_bound_rises_after_a_screened_ket(
        ) -> None:
    """A concrete counterexample: H2O/6-31G, `tau = 0.03`, `n = 1`. In the
    literal stream (`G * dmax`) some launch has a bra whose six-block bound
    is under `tau` at one ket and AT or above it at a later one -- a
    `break` there would drop a quartet the screen keeps, and its exact terms
    reach `tau`. (H2O/STO-3G's bound rises too, at `tau = 0`, but pruning
    separates every rising pair before it can straddle a `tau`.)"""
    name, tau = "h2o_631g", 0.03
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=1)
    bounds, _ = _bounds(basis, D, plan)
    found = []
    for (bi, ki), rows in bounds.items():
        for ib, row in enumerate(rows):
            for i, j in itertools.combinations(range(len(row)), 2):
                if row[i] < tau <= row[j]:
                    found.append((bi, ki, ib, i, j))
    assert found, "the literal six-block bound was monotone on this fixture"
    # one of them is significant in the exact ERIs: the break is wrong
    worst = 0.0
    for bi, ki, ib, _, j in found:
        P = plan.batches[bi].pairs[ib].shell_pair
        Q = plan.batches[ki].pairs[j].shell_pair
        worst = max(worst, _six_terms(name, walk_f._quartet(P, Q)))
    assert worst >= tau


def test_no_ket_order_makes_the_six_block_bound_monotone_for_every_bra(
        ) -> None:
    """Two bras of one launch order two kets OPPOSITELY by the six-block
    bound (H2O/STO-3G, `tau = 0`, `n = 1`): whatever order the kets are
    sorted in, one bra sees its bound rise. So the break needs a bound that
    dominates the six blocks, not a better key for them."""
    basis, D, _ = molecule("h2o_sto3g")
    plan = f_launches(basis, D, bound=BOUND, tau=0.0, n=1)
    bounds, _ = _bounds(basis, D, plan)
    conflicts = 0
    for rows in bounds.values():
        for r1, r2 in itertools.combinations(rows, 2):
            for i, j in itertools.combinations(range(len(r1)), 2):
                if (r1[i] - r1[j]) * (r2[i] - r2[j]) < 0:
                    conflicts += 1
    assert conflicts > 0


# ---------------------------------------------------------------------------
# The dominating bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "tau", "n"), WALKS)
def test_m_ab_dominates_every_kets_six_block_dmax_as_floats(
        name, tau, n) -> None:
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=n, ket_break=True)
    bounds, Dsh = _bounds(basis, D, plan)
    n_sh = len(basis.shells)
    for L in plan.launches:
        bras, kets = plan.batches[L.bra].pairs, plan.batches[L.ket].pairs
        M = walk_f.dominating_factors(Dsh, n_sh, [p.shell_pair for p in bras],
                                      [q.shell_pair for q in kets])
        rows = bounds[(L.bra, L.ket)]
        for ib, p in enumerate(bras):
            for ik in range(len(kets)):
                # bound(P, Q_j) <= Gb (G_k M) for every k <= j: the proof
                for k in range(ik + 1):
                    assert rows[ib][ik] <= p.G * (kets[k].G * M[ib])


# ---------------------------------------------------------------------------
# The kernel, per class: bitwise the literal kernel on a sorted batch
# ---------------------------------------------------------------------------

_EXPS = ((1.4, 0.3), (0.8, 2.1), (0.5, 1.7))
_COEFS = ((0.6, 0.45), (0.8, -0.3), (0.7, 0.5))


def _basis(cls) -> Basis:
    """Three shells of each angular momentum the class reads, spread along
    a line so the pairs' `G` spans orders of magnitude."""
    rng = np.random.default_rng(352)
    shells = []
    for l in sorted(set(cls)):  # noqa: E741
        for i in range(3):
            c = tuple(float(x) for x in rng.uniform(-0.3, 0.3, 3) + 1.4 * i)
            shells.append(Shell(l, c, _EXPS[i], _COEFS[i]))
    return Basis(tuple(shells))


def _pairs(basis, la, lb):
    idx = [i for i, s in enumerate(basis.shells) if s.l == la]
    jdx = [i for i, s in enumerate(basis.shells) if s.l == lb]
    if la == lb:
        return [(idx[i], idx[j]) for i in range(len(idx))
                for j in range(i, len(idx))]
    return [(a, b) for a in idx for b in jdx]


def _rec(basis, a, b):
    sa, sb = basis.shells[a], basis.shells[b]
    G = float(sum(primitive_schwarz_bounds(sa, sb, one_centre=one_centre)))
    n = len(sa.exponents) * len(sb.exponents)
    return walk_f.pair_record(basis, a, b, G=G, prims=range(n))


def _run(cls, basis, D, bras, kets, *, levers, tau, threads=2):
    k = KF.build_f_kernel(_spec(cls, levers))
    pb, pk = KF.pack(bras), KF.pack(kets)
    n_sh, n_ao = len(basis.shells), basis.n_ao
    Dsh = KF.density_blocks(D.tolist(), basis.offsets,
                            [s.n_cart for s in basis.shells])
    args = {
        "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree, "k_cd": pk.degree,
        "n_sh": n_sh, "n_ao": n_ao, "diag": 0, "tau": tau,
        "bra_prim": pb.prim, "AB": pb.sep, "G_ab": pb.G, "bra_sh": pb.sh,
        "bra_ao": pb.ao, "ket_prim": pk.prim, "CD": pk.sep, "G_cd": pk.G,
        "ket_sh": pk.sh, "ket_ao": pk.ao, "Dsh": Dsh, "D": D.ravel(),
        KF.FOCK: np.zeros(n_ao * n_ao),
        ket_break.BOUND: walk_f.dominating_factors(
            Dsh, n_sh, [b.sh for b in bras], [q.sh for q in kets]),
    }
    a = {}
    for p in k.params:
        v = args[p.name]
        if p.array:
            v = np.array(v, dtype=np.int32 if p.type.name == "I32" else float)
        a[p.name] = v
    it = Interpreter(k, launch=Launch(*KF.launch_shape(pb.n, threads), 32),
                     boys=boys)
    return it.run(**a)[KF.FOCK], it.iterations[KF.KET]


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: "".join(map(str, c)))
def test_on_a_sorted_batch_the_lever_is_the_literal_kernel_bitwise(
        cls) -> None:
    """One off-diagonal launch of the class: every bra pair by every ket
    pair of a 3-shells-per-l basis (bra and ket pairs disjoint when the
    classes coincide), kets sorted descending by `G`, `tau` between the
    bounds so both the break and the six-block skip fire. The lever's `F`
    is the literal kernel's bit for bit, and its ket loop is shorter."""
    basis = _basis(cls)
    la, lb, lc, ld = cls
    bra_ix = _pairs(basis, la, lb)
    ket_ix = [q for q in _pairs(basis, lc, ld) if q not in bra_ix[:2]]
    bras = [_rec(basis, *p) for p in bra_ix[:2]]
    kets = sorted((_rec(basis, *q) for q in ket_ix), key=lambda r: r.G,
                  reverse=True)
    r = np.random.default_rng(1).uniform(-1, 1, (basis.n_ao, basis.n_ao))
    D = r + r.T
    Dsh = KF.density_blocks(D.tolist(), basis.offsets,
                            [s.n_cart for s in basis.shells])
    M = walk_f.dominating_factors(Dsh, len(basis.shells),
                                  [b.sh for b in bras], [q.sh for q in kets])
    # tau at the median break bound: half of every bra's kets break
    tau = float(np.median([b.G * (q.G * m) for b, m in zip(bras, M,
                                                          strict=True)
                           for q in kets]))
    # one thread, so a break before a thread's last ket saves an iteration
    # (the strided case is the walks' below, at walk_f's 4 threads)
    F_lit, n_lit = _run(cls, basis, D, bras, kets, levers=(), tau=tau,
                        threads=1)
    F_brk, n_brk = _run(cls, basis, D, bras, kets, levers=(NAME,), tau=tau,
                        threads=1)
    assert np.max(np.abs(F_lit)) > 0
    assert np.array_equal(F_lit, F_brk)
    assert n_brk < n_lit  # the break fired: trailing iterations left


# ---------------------------------------------------------------------------
# Whole molecules, walked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "tau", "n"), WALKS)
def test_the_walked_lever_is_the_literal_walk_of_the_same_stream_bitwise(
        name, tau, n) -> None:
    wb, wl, plan = _walks(name, tau, n)
    assert all(NAME in L.spec.levers for L in plan.launches)
    assert np.array_equal(np.array(wb.F), np.array(wl.F))
    assert wb.kernel_broken  # the break fired
    assert not wl.kernel_broken
    # every quartet the break drops is one the six-block screen skips
    assert wb.kernel_broken <= wl.kernel_screened
    assert wb.kernel_screened == wl.kernel_screened
    assert wb.launched == wl.launched


@pytest.mark.parametrize(("name", "tau", "n"), WALKS)
def test_every_broken_quartet_is_below_tau_in_the_exact_eris(
        name, tau, n) -> None:
    wb, _, plan = _walks(name, tau, n)
    for q in wb.kernel_broken:
        assert _six_terms(name, q) < tau, q
    # and F within the screened sweep's bound (test_fock_build's)
    F = np.array(wb.F)
    pruned = {p.shell_pair for p in plan.pairs if p.degree < p.n_prim}
    touched = members(q for q in wb.computed if q[0] in pruned
                      or q[1] in pruned)
    bound = (TOL * sweep(name, block="scale")
             + tau * sweep(name, touched, block="one"))
    assert np.all(np.abs(F - sweep(name, members(wb.computed))) <= bound)


# ---------------------------------------------------------------------------
# The stream must be the sorted one: refused, and wrong when forced
# ---------------------------------------------------------------------------


def _reversed(plan):
    return replace(plan, batches=tuple(
        replace(b, pairs=b.pairs[::-1]) for b in plan.batches))


@pytest.mark.slow  # two H2O/6-31G walks interpreted: 5 s
def test_an_unsorted_stream_is_refused_and_forced_gives_a_wrong_f(
        monkeypatch) -> None:
    """H2O/6-31G, `tau = 0.03`, `n = 1`, every batch reversed (ascending
    `G`). (H2O/STO-3G's batches are too short for a reversed break to reach
    a kept ket: every ket it drops there the six-block screen skips too.)"""
    name, tau, n = "h2o_631g", 0.03, 1
    basis, D, _ = molecule(name)
    plan = _reversed(f_launches(basis, D, bound=BOUND, tau=tau, n=n,
                                ket_break=True))
    with pytest.raises(ValueError, match="not sorted descending by G"):
        walk_f.walk_f(plan, basis, D, run=run)
    # the plain kernel on the reversed stream is still right (a skip is
    # order-free); force the lever past the refusal and it is not
    w_ok = walk_f.walk_f(_plain(plan), basis, D, run=run)
    monkeypatch.setattr(walk_f, "_check_break_order", lambda plan: None)
    w_bad = walk_f.walk_f(plan, basis, D, run=run)
    dropped = w_bad.kernel_broken - w_ok.kernel_screened
    assert dropped, "the forced break dropped nothing the screen keeps"
    assert max(_six_terms(name, q) for q in dropped) >= tau
    # the plain walk is test_fock_build's answer (F within rounding and
    # pruning of the sweep over the quartets the six-block screen keeps);
    # the forced break leaves it by more than tau -- a deposit no screen of
    # this build may drop. On the SORTED stream the two are bitwise equal.
    pruned = {p.shell_pair for p in plan.pairs if p.degree < p.n_prim}
    touched = members(q for q in w_ok.computed if q[0] in pruned
                      or q[1] in pruned)
    bound = (TOL * sweep(name, block="scale")
             + tau * sweep(name, touched, block="one"))
    kept = sweep(name, members(w_ok.computed))
    F_ok, F_bad = np.array(w_ok.F), np.array(w_bad.F)
    assert np.all(np.abs(F_ok - kept) <= bound)
    assert np.max(np.abs(F_bad - F_ok)) >= tau


def test_the_walker_reads_the_break_off_the_launchs_spec() -> None:
    """A stream sorted by `G` whose launches name no lever walks as the
    literal kernel; the lever only on some launches is checked per launch."""
    basis, D, _ = molecule("h2o_sto3g")
    plan = f_launches(basis, D, bound=BOUND, tau=0.1, n=4, ket_break=True)
    w = walk_f.walk_f(_plain(plan), basis, D, run=run)
    assert not w.kernel_broken
    # the literal stream with the lever on its launches: refused wherever
    # its G * dmax order is not G's (H2O/6-31G's; STO-3G's happens to be)
    basis, D, _ = molecule("h2o_631g")
    lit = f_launches(basis, D, bound=BOUND, tau=0.03, n=1)
    lev = replace(lit, launches=tuple(
        replace(L, spec=replace(L.spec, levers=(NAME,)))
        for L in lit.launches))
    unsorted = any(any(a.G < b.G for a, b in itertools.pairwise(B.pairs))
                   for B in lit.batches)
    assert unsorted
    with pytest.raises(ValueError, match="not sorted descending by G"):
        walk_f.walk_f(lev, basis, D, run=run)


# ---------------------------------------------------------------------------
# The default stream is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "tau", "n"),
                         [("h2_sto3g", 0.0, 1), *WALKS])
def test_the_default_stream_is_unchanged_and_the_key_moves_only_order(
        name, tau, n) -> None:
    basis, D, _ = molecule(name)
    lit = f_launches(basis, D, bound=BOUND, tau=tau, n=n)
    assert lit == f_launches(basis, D, bound=BOUND, tau=tau, n=n,
                             ket_break=False)
    assert all(p.key == p.G * p.dmax for p in lit.pairs)
    assert all(not L.spec.levers for L in lit.launches)
    srt = f_launches(basis, D, bound=BOUND, tau=tau, n=n, ket_break=True)
    assert all(p.key == p.G for p in srt.pairs)
    for a, b in zip(lit.batches, srt.batches, strict=True):
        assert (a.la, a.lb, a.degree, a.mag) == (b.la, b.lb, b.degree, b.mag)
        assert set(a.pairs) == {replace(p, key=p.G * p.dmax)
                                for p in b.pairs}
        assert all(x.G >= y.G for x, y in itertools.pairwise(b.pairs))
    assert lit.screened == srt.screened and lit.deleted == srt.deleted
    assert [(L.bra, L.ket) for L in lit] == [(L.bra, L.ket) for L in srt]
    assert all(L.spec == replace(M.spec, levers=())
               and M.spec.levers == (NAME,) for L, M in zip(lit, srt,
                                                           strict=True))
    with pytest.raises(ValueError, match="not a bool"):
        f_launches(basis, D, bound=BOUND, tau=tau, n=n, ket_break=1)
