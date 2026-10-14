"""`kernel_f.py`: the fused Fock kernel (#282) and its six-block predigest
with the 8-fold weights, derived (#283), interpreted.

THE RECURRENCE HALF (#282) is held BITWISE to a host path built here from
#279's pinned pieces, with no fused kernel in it: per primitive quartet (ket
primitive outer, bra inner) the scalars `kernel_f`'s docstring lists,
computed in Python floats in the same operation order;
`interpret(hgp.recurrence.build_vrr_kernel)` for the boundary
`[e0|f0]^(0)`; the left fold `c = c + v` from `+0.0`; and
`interpret(build_hrr_kernel)` on the sums. It reads the ERIs through the
kernel's TAP (`eris=True`), which is asserted to be the shipped body plus
the `g` writes and nothing else. The integrals are also checked against
`ao/eri.py`'s MMD path (a different engine; a rounding tolerance).

THE DIGESTION HALF (#283) is held to `ao.fock`'s sweep over EVERY ORDERED
quartet at weight 1 with no symmetry (milestone 12, pin 6) -- never to a
re-typing of the weights. Per class, one unique quartet's closed deposit
(`F + F^T`) equals the sweep restricted, through its injected `eri`, to the
ordered members of that quartet's orbit -- members enumerated here by
brute force over shell labels, not from `kernel_f.orbit_group`. Every
degeneracy case runs: `a = b`, `c = d`, `(ab) = (cd)`, the first two
together, and all three. The sweep's ERIs come from `ao.hgp_drive` (HGP,
interpreted per ordered member), so the only shared ancestry is the
recurrence the bitwise half already pins. Whole molecules run from
`launch_f`'s plan against the full sweep and libcint's `vj - vk/2`.

CAN FAIL: the predecessor's bug (Coulomb weight 1 for 2), the exchange
weight's sign, a dropped degeneracy factor, and Fig. 1's unit weights each
miss the sweep by orders of magnitude beyond the tolerance.

TOLERANCE, stated and not tuned: `1e-13` of `scale_G = scale_J + scale_K /
2`, element by element, where the scales are the same restricted sweep run
on `|D|` with `hgp_drive`'s absolute-value twin -- this file's ERI bound
(the same integrals in two operation orders) carried through the
digestion's sums by the triangle inequality. Against libcint, `1e-12` of
the same scale (test_ao_fock.py's LIBCINT_TOL).

Fast tier for `L <= 4`, `slow` above (#282, #283). No compiler, no card,
no emission (#287): the interpreter is the whole check at this stage.
"""

from __future__ import annotations

import itertools
import math
import re
import time
from dataclasses import astuple, replace
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves import kernel_f as KF
from felizjueves.ao import fock
from felizjueves.ao import hgp_drive as HD
from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.ao.pairs import primitive_schwarz_bounds
from felizjueves.ao.shell import Basis, Shell
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp import recurrence as R
from felizjueves.interp import Interpreter, Launch, interpret
from felizjueves.ir import (
    Assign,
    AtomicAdd,
    Break,
    Call,
    Decl,
    Fn,
    For,
    If,
    Index,
    Kernel,
    Note,
    Space,
    Var,
    walk,
)
from felizjueves.launch_f import f_launches
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre, pq
from felizjueves.spec import Operator, Spec, Target

CLASSES = vrr.CLASSES


def _speed(cls):
    """Fast for `L <= 4`, `slow` above (#282)."""
    return pytest.param(cls, id=str(cls),
                        marks=(pytest.mark.slow,) if sum(cls) > 4 else ())


LADDER = [_speed(c) for c in CLASSES]

# ---------------------------------------------------------------------------
# The pinned statement counts: the plan, not silently re-derived
# ---------------------------------------------------------------------------

# (vrr nodes, boundary accumulators, HRR statements incl. boundary reads,
#  HRR two-term updates, ERIs) per class, as built when #282 landed. The
# boundary is the BOUNDED set (961 at (dd|dd), not #282's printed 1,225 --
# #273/#274's comments), and the HRR column is #274's convention: 6,254 at
# (dd|dd) = 961 boundary reads + 5,293 updates.
COUNTS = {
    (0, 0, 0, 0): (1, 1, 1, 0, 1),
    (0, 0, 0, 1): (5, 4, 7, 3, 3),
    (0, 0, 0, 2): (15, 10, 25, 15, 6),
    (0, 0, 1, 1): (15, 9, 18, 9, 9),
    (0, 0, 1, 2): (35, 19, 60, 41, 18),
    (0, 0, 2, 2): (70, 31, 110, 79, 36),
    (0, 1, 0, 1): (21, 16, 37, 21, 9),
    (0, 1, 0, 2): (61, 40, 115, 75, 18),
    (0, 1, 1, 1): (61, 36, 90, 54, 27),
    (0, 1, 1, 2): (141, 76, 256, 180, 54),
    (0, 1, 2, 2): (281, 124, 454, 330, 108),
    (0, 2, 0, 2): (155, 100, 340, 240, 36),
    (0, 2, 1, 1): (155, 90, 279, 189, 54),
    (0, 2, 1, 2): (355, 190, 721, 531, 108),
    (0, 2, 2, 2): (705, 310, 1249, 939, 216),
    (1, 1, 1, 1): (146, 81, 243, 162, 81),
    (1, 1, 1, 2): (336, 171, 711, 540, 162),
    (1, 1, 2, 2): (671, 279, 1269, 990, 324),
    (1, 2, 1, 2): (696, 361, 1878, 1517, 324),
    (1, 2, 2, 2): (1381, 589, 3282, 2693, 648),
    (2, 2, 2, 2): (2330, 961, 6254, 5293, 1296),
}

# The AtomicAdd statements per class, RECORDED as built by #283 (the evidence
# for `fock/predigest-bounds-atomics`, asserted by #291): exactly the six
# blocks' elements. Beside the ERI count (COUNTS' last column): the atomics
# EXCEED the ERIs up to (0,1,1,2) -- 6 against 1 at (ss|ss) -- cross at
# (0,1,2,2) / (1,1,1,1), and are 216 against 1,296 at (dd|dd).
ATOMICS = {
    (0, 0, 0, 0): 6, (0, 0, 0, 1): 12, (0, 0, 0, 2): 21, (0, 0, 1, 1): 22,
    (0, 0, 1, 2): 37, (0, 0, 2, 2): 61, (0, 1, 0, 1): 22, (0, 1, 0, 2): 37,
    (0, 1, 1, 1): 36, (0, 1, 1, 2): 57, (0, 1, 2, 2): 87, (0, 2, 0, 2): 61,
    (0, 2, 1, 1): 57, (0, 2, 1, 2): 87, (0, 2, 2, 2): 126, (1, 1, 1, 1): 54,
    (1, 1, 1, 2): 81, (1, 1, 2, 2): 117, (1, 2, 1, 2): 117, (1, 2, 2, 2): 162,
    (2, 2, 2, 2): 216,
}


def test_the_count_tables_cover_the_ladder() -> None:
    assert tuple(COUNTS) == CLASSES and len(CLASSES) == 21
    assert tuple(ATOMICS) == CLASSES


@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_emitted_statement_counts_are_the_dags(cls) -> None:
    """Read off the BUILT kernel by statement kind, equal to the pinned table
    and to `hgp.vrr` / `hgp.hrr`'s own node counts (#273, #274)."""
    k = KF.build_f_kernel(cls)
    got = KF.counted(k)
    assert astuple(got) == (*COUNTS[cls], ATOMICS[cls])
    assert got == KF.kernel_counts(cls)
    v, h = vrr.build(cls), hrr.build(cls)
    assert got.vrr == v.n_nodes
    assert got.boundary == h.n_boundary == len(vrr.target_set(cls))
    assert got.hrr == h.n_statements == h.n_boundary + h.n_interior
    assert got.eri == h.n_eri
    # and the eq 27 update: one `c = c + v` per boundary value
    acc = [s for s in walk(k.body) if isinstance(s, Assign)
           and isinstance(s.dst, Var) and s.dst.name.startswith("C")]
    assert len(acc) == h.n_boundary


def test_dd_dd_is_961_boundary_values_not_1225() -> None:
    """The orchestrator's reading of #282's 1,225 (35^2) as a misprint for
    the bounded set's 31^2: only 961 fits #274's 6,254 HRR statements."""
    c = KF.counted(KF.build_f_kernel((2, 2, 2, 2)))
    assert (c.boundary, c.hrr, c.hrr_interior) == (961, 6254, 5293)
    assert c.boundary == 31 * 31 != 35 * 35


def test_dd_dd_builds_and_its_construction_time_is_immaterial() -> None:
    t = time.perf_counter()
    k = KF.build_f_kernel((2, 2, 2, 2))
    dt = time.perf_counter() - t
    assert k.name == "f_2222"
    # ~0.4 s with the digestion; the bound only says "not material" (#282
    # asks that a material one be reported)
    assert dt < 10.0, dt


# ---------------------------------------------------------------------------
# The shape: nest, staging, the half, the screen, HRR after the loop
# ---------------------------------------------------------------------------


def _find_for(body, var):
    return next(s for s in walk(body) if isinstance(s, For) and s.var == var)


BOUND_LT_TAU = Call(Fn.LT, (Var("bound"), Var("tau")))


def _the_half_and_screen(k):
    """The ket loop's one statement is the canonical-half `If`; the screen
    is the one `If(bound < tau)` in its `orelse`."""
    ket = _find_for(k.body, KF.KET)
    assert len(ket.body) == 1 and isinstance(ket.body[0], If)
    half = ket.body[0]
    screens = [s for s in half.orelse if isinstance(s, If)
               and s.cond == BOUND_LT_TAU]
    assert len(screens) == 1
    return ket, half, screens[0]


def test_the_nest_is_ket_then_kcd_then_kab_and_hrr_runs_after_it() -> None:
    cls = (1, 1, 1, 1)
    k = KF.build_f_kernel(cls)
    ket = _find_for(k.body, KF.KET)
    assert ket.start == Call(Fn.THREAD_X) and ket.step == Call(Fn.BLOCK_DIM_X)
    kcd = _find_for(ket.body, KF.KCD)
    kab = _find_for(kcd.body, KF.KAB)
    inner = list(walk(kab.body))
    vrr_in = {s.name for s in inner if isinstance(s, Decl)
              and s.name.startswith("V")}
    assert len(vrr_in) == vrr.build(cls).n_nodes  # every VRR node, inside
    hrr_names = {R.hrr_name(n) for n in hrr.build(cls).terms}
    assert not hrr_names & {s.name for s in walk(kcd.body)
                            if isinstance(s, Decl)}  # no HRR in the loops
    # in the quartet's work: accumulators, then the nest, then the HRR, then
    # the digestion (which declares no H name)
    _, _, screen = _the_half_and_screen(k)
    work = screen.orelse
    i_loop = next(i for i, s in enumerate(work) if isinstance(s, For))
    decls_before = [s.name for s in work[:i_loop] if isinstance(s, Decl)]
    decls_after = [s.name for s in work[i_loop + 1:] if isinstance(s, Decl)]
    assert sum(n.startswith("C") and not n.startswith("CD_")
               for n in decls_before) == 81
    assert {n for n in decls_after if n.startswith("H")} == hrr_names
    last_h = max(i for i, s in enumerate(work) if isinstance(s, Decl)
                 and s.name in hrr_names)
    first_atomic = min(i for i, s in enumerate(work)
                       if isinstance(s, AtomicAdd))
    assert last_h < first_atomic  # the digestion after the HRR


def test_the_bra_record_is_staged_in_shared_memory_under_the_cap() -> None:
    k = KF.build_f_kernel((0, 1, 0, 1))
    sh = [s for s in walk(k.body) if isinstance(s, Decl)
          and s.space is Space.SHARED]
    assert [s.name for s in sh] == [KF.AB_SH]
    assert [d.value for d in sh[0].shape] == [KF.MAX_PRIM_PAIRS, len(KF.REC)]
    stage = _find_for(k.body, KF.STAGE)
    ket = _find_for(k.body, KF.KET)
    order = list(walk(k.body))
    assert order.index(stage) < order.index(ket)  # staged before any ket


_BREAK = re.compile(r"^\s*break;", re.MULTILINE)
_SCREEN = re.compile(r"if \(bound < tau\) \{\n((?:\s*//[^\n]*\n)*)\s*\} else \{")


def _kernel_text(kernel: Kernel, backend: str) -> str:
    """The emitted `__global__` function alone: the device `fj_boys` before
    it has a `break` of its own (the series' convergence exit)."""
    from felizjueves.device import GFX1200, SM_86
    from felizjueves.emit import cuda, hip
    emit, arch = {"cuda": (cuda.emit, SM_86), "hip": (hip.emit, GFX1200)}[backend]
    text = emit(kernel, arch)
    assert text.count("__global__") == 1
    return text[text.index("__global__"):]


@pytest.mark.claim("fock/skip-not-break")
@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)],
                         ids=str)
def test_fock_skip_not_break_the_screen_is_a_skip(cls) -> None:
    """`fock/skip-not-break`: the in-kernel screen guards the quartet's work
    and exits nothing. In the IR its `then` holds only a `Note`, its
    `orelse` the whole quartet, and there is no `Break` anywhere in the
    kernel; in the EMITTED text, on both vendors, the screen is `if (bound <
    tau) { <comments> } else { ... }` and the kernel has no `break;`. The
    contrast in the same test: `Target.J`'s emitted `[p|q]` kernel takes a
    `break;` on its screen, and `Target.K`'s IR has a `Break` -- the sorted
    early exits this kernel inverts. The canonical half test is a skip of
    the same shape."""
    k = KF.build_f_kernel(cls)
    ket, half, screen = _the_half_and_screen(k)
    assert half.cond == Call(Fn.LT, (Var(KF.KET), Var("kmin")))
    for s in (half, screen):
        assert all(isinstance(x, Note) for x in s.then)
    assert not any(isinstance(s, Break) for s in walk(k.body))
    assert _find_for(screen.orelse, KF.KCD)  # the work is the orelse
    for backend in ("cuda", "hip"):
        text = _kernel_text(k, backend)
        assert not _BREAK.search(text), backend
        assert len(_SCREEN.findall(text)) == 1, backend
    # the contrast: Target.J's emitted screen is a break, Target.K's IR too
    from felizjueves.kernel import build_pq_kernel
    from felizjueves.kernel_k import build_k_kernel
    for backend in ("cuda", "hip"):
        jtext = _kernel_text(build_pq_kernel(4), backend)
        assert re.search(r"if \(bound < tau\) \{\n\s*break;", jtext), backend
    kk = build_k_kernel((1, 0, 0, 0))
    assert any(isinstance(s, Break) for s in walk(kk.body))


def _dsh_reads(stmts) -> set[tuple[str, str]]:
    return {(s.init.args[1].at[0].name, s.init.args[1].at[1].name)
            for s in stmts if isinstance(s, Decl) and s.name.startswith("dmax")}


def test_the_screen_reads_six_density_blocks_weighted() -> None:
    """The structural half of `fock/symmetry-vs-screening` (UNVERIFIABLE,
    #291: its reason names this test, unmarked, as what IS held): the bound
    reads SIX `Dsh` blocks, `ab cd ac ad bc bd`, each times its block's
    `|w|`, where `Target.J` reads one (`maxD * I_p * I_q`)."""
    k = KF.build_f_kernel((1, 1, 1, 1))
    _, half, _ = _the_half_and_screen(k)
    reads = _dsh_reads(half.orelse)
    assert reads == {("sa", "sb"), ("sc", "sd"), ("sa", "sc"), ("sa", "sd"),
                     ("sb", "sc"), ("sb", "sd")}
    w = {s.init.args[0].value for s in half.orelse
         if isinstance(s, Decl) and s.name.startswith("dmax")}
    assert w == {abs(KF.WEIGHTS.coulomb), abs(KF.WEIGHTS.exchange)}


def test_six_two_index_outputs_and_atomics_only_into_them() -> None:
    """The six outputs on one mixed class: every `AtomicAdd` writes `F` at
    one of the six blocks' offset pairs, each block gets exactly its
    elements, and nothing else is written to global memory but the tap's
    `g` (absent from the shipped kernel). The ladder-wide check, with the
    thread mapping, is `fock/fused-six-block-digest`'s test below."""
    cls = (0, 1, 1, 2)
    k = KF.build_f_kernel(cls)
    n = [len(vrr.cartesians(x)) for x in cls]
    by_block: dict[tuple[str, str], int] = {}
    for s in walk(k.body):
        if isinstance(s, AtomicAdd):
            assert s.dst.array == KF.FOCK
            key = tuple(e.args[0].name for e in s.dst.at)
            by_block[key] = by_block.get(key, 0) + 1
    o = ("oa", "ob", "oc", "od")
    assert by_block == {(o[i], o[j]): n[i] * n[j] for i, j in KF.BLOCKS}
    assert not any(isinstance(s, Assign) and isinstance(s.dst, Index)
                   and s.dst.array not in (KF.AB_SH,) for s in walk(k.body))
    assert KF.OUT not in {p.name for p in k.params}


@pytest.mark.claim("fock/fused-six-block-digest")
def test_one_thread_per_contracted_quartet_evaluates_every_eri_into_six_blocks(
) -> None:
    """`fock/fused-six-block-digest`, on the built IR of all 21 classes (the
    builds are cheap: ~0.4 s at `(dd|dd)`, so no class is deferred):

    * the thread mapping is one block per bra pair (`bra = blockIdx.x`,
      `launch_shape`'s grid is the bra count) and one thread per contracted
      quartet: the ket loop starts at `threadIdx.x` and strides by
      `blockDim.x`, and the whole `K_CD x K_AB` primitive nest sits inside
      it, so a quartet's contraction never leaves its thread;
    * every ERI of the class -- each of the `n_a n_b n_c n_d` targets of
      `hgp.hrr.targets` -- is declared inside that thread's screened work,
      after the nest, and read by the digestion;
    * exactly six two-index outputs are written: every `AtomicAdd` lands
      in `F` at one of the six blocks' offset pairs, each block takes
      exactly its `n_i n_j` elements, and nothing else reaches global
      memory."""
    o = ("oa", "ob", "oc", "od")
    for cls in CLASSES:
        k = _kernel(cls)
        n = [len(vrr.cartesians(x)) for x in cls]
        assert any(isinstance(s, Decl) and s.name == "bra"
                   and s.init == Call(Fn.BLOCK_X) for s in walk(k.body)), cls
        assert KF.launch_shape(7, 32) == (7, (32, 1))
        ket = _find_for(k.body, KF.KET)
        assert ket.start == Call(Fn.THREAD_X), cls
        assert ket.step == Call(Fn.BLOCK_DIM_X), cls
        assert sum(isinstance(s, For) and s.var == KF.KET
                   for s in walk(k.body)) == 1, cls
        _, _, screen = _the_half_and_screen(k)
        work = screen.orelse
        kcd = _find_for(work, KF.KCD)
        assert _find_for(kcd.body, KF.KAB), cls  # the contraction, in-thread
        eris = {R.hrr_name(t) for t in hrr.targets(cls)}
        assert len(eris) == math.prod(n) == COUNTS[cls][4], cls
        declared = {s.name for s in walk(work) if isinstance(s, Decl)}
        assert eris <= declared, cls
        in_nest = {s.name for s in walk(kcd.body) if isinstance(s, Decl)}
        assert not eris & in_nest, cls  # after the nest, on contracted sums
        read = {e.name for s in walk(work) for e in _vars(s)}
        assert eris <= read, cls  # every ERI reaches the digestion
        by_block: dict[tuple[str, str], int] = {}
        for s in walk(k.body):
            if isinstance(s, AtomicAdd):
                assert s.dst.array == KF.FOCK, cls
                key = tuple(e.args[0].name for e in s.dst.at)
                by_block[key] = by_block.get(key, 0) + 1
        assert by_block == {(o[i], o[j]): n[i] * n[j]
                            for i, j in KF.BLOCKS}, cls
        assert not any(isinstance(s, Assign) and isinstance(s.dst, Index)
                       and s.dst.array != KF.AB_SH for s in walk(k.body)), cls


def _vars(stmt) -> list[Var]:
    """Every `Var` read by one statement's expressions (not its children's
    bodies, which `walk` visits on its own)."""
    from felizjueves.ir import _exprs_of, _subexprs
    return [x for e in _exprs_of(stmt) if e is not None
            for x in _subexprs(e) if isinstance(x, Var)]


def test_a_non_f_class_is_refused() -> None:
    with pytest.raises(ValueError, match="Target.F"):
        KF.build_f_kernel(Spec(Operator.COULOMB, 0, (1, 0, 0, 0), Target.K))
    with pytest.raises(TypeError):
        KF.build_f_kernel([0, 0, 0, 0])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        KF.build_f_kernel((0, 0, 0, 0), weights=(2.0, -0.5))  # type: ignore


def test_params_follow_the_hrr_and_the_tap() -> None:
    assert "AB" not in KF.params_of((0, 0, 1, 1))
    assert "CD" in KF.params_of((0, 0, 1, 1))
    assert "AB" in KF.params_of((0, 1, 0, 0))
    assert "CD" not in KF.params_of((0, 1, 0, 0))
    assert KF.OUT not in KF.params_of((0, 0, 0, 0))
    assert KF.params_of((0, 0, 0, 0), eris=True)[-1] == KF.OUT
    for eris in (False, True):
        k = KF.build_f_kernel((0, 0, 0, 0), eris=eris)
        assert tuple(p.name for p in k.params) == KF.params_of(
            (0, 0, 0, 0), eris=eris)


def _strip_tap(body):
    out = []
    for s in body:
        if isinstance(s, Assign) and isinstance(s.dst, Index) and \
                s.dst.array == KF.OUT:
            continue
        if isinstance(s, Note) and s.prov.startswith("the ERI tap"):
            continue
        if isinstance(s, For):
            s = replace(s, body=tuple(_strip_tap(s.body)))
        elif isinstance(s, If):
            s = replace(s, then=tuple(_strip_tap(s.then)),
                        orelse=tuple(_strip_tap(s.orelse)))
        out.append(s)
    return out


@pytest.mark.parametrize("cls", [(0, 0, 0, 0), (0, 1, 1, 2), (1, 1, 1, 1)],
                         ids=str)
def test_the_tap_is_the_shipped_kernel_plus_its_writes(cls) -> None:
    """What the bitwise tests read through `g` is the shipped kernel: strip
    the tap's writes and its Note, and the bodies are equal."""
    shipped = KF.build_f_kernel(cls)
    tap = KF.build_f_kernel(cls, eris=True)
    assert tuple(_strip_tap(tap.body)) == shipped.body
    assert tap.params[:-1] == shipped.params


# ---------------------------------------------------------------------------
# The weights and the degeneracy factor, derived
# ---------------------------------------------------------------------------


def test_the_weights_are_derived_from_the_orbit() -> None:
    """`derive_weights` enumerates the group: Coulomb 2 (fraction 1 x two
    in-order members), exchange -1/2 (fraction 1/2 x one) -- and the
    fractions are ao.fock's, the witness's, not a restatement that could
    drift."""
    assert len(KF.orbit_group()) == 8
    assert KF.derive_weights() == KF.WEIGHTS == KF.Weights(2.0, -0.5, True)
    assert KF.COULOMB_FRACTION == fock.COULOMB_FRACTION
    assert KF.EXCHANGE_FRACTION == fock.EXCHANGE_FRACTION
    assert KF.N_SPIN == fock.N_SPIN
    assert KF.WEIGHTS.coulomb != 1.0  # the predecessor's number


def _canonical_quartets(labels):
    pairs = [(a, b) for a in labels for b in labels if a <= b]
    return [(*p, *q) for p in pairs for q in pairs if p <= q]


def test_the_degeneracy_factor_is_one_over_the_stabiliser() -> None:
    """The kernel's three flags, `(1/2 if a = b)(1/2 if c = d)(1/2 if P =
    Q)`, against `1/|Stab|` by enumeration over every canonical quartet of
    three shell labels; and the docstring's orbit sizes, per case, by brute
    force over the ordered quartets."""
    for a, b, c, d in _canonical_quartets(range(3)):
        flags = ((0.5 if a == b else 1.0) * (0.5 if c == d else 1.0)
                 * (0.5 if (a, b) == (c, d) else 1.0))
        assert KF.degeneracy(a, b, c, d) == flags, (a, b, c, d)
    for q, size in (((0, 1, 2, 3), 8), ((0, 0, 1, 2), 4), ((0, 1, 2, 2), 4),
                    ((0, 1, 0, 1), 4), ((0, 0, 1, 1), 2), ((0, 0, 0, 0), 1)):
        assert len(_members([(q[0], q[1])], [(q[2], q[3])])) == size
        assert KF.degeneracy(*q) == size / 8


# ---------------------------------------------------------------------------
# The packer, its cap, and the host's half
# ---------------------------------------------------------------------------


def _prim(z=1.0):
    return KF.PrimRec((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), z, 1.0)


def _pr(prims, sh=(0, 1), ao=(0, 1)):
    return KF.PairRec((0.0, 0.0, 0.0), 1.0, tuple(prims), sh, ao)


def test_the_packer_refuses_past_the_cap_and_a_mixed_degree() -> None:
    at_cap = _pr((_prim(),) * KF.MAX_PRIM_PAIRS)
    assert KF.pack([at_cap]).degree == KF.MAX_PRIM_PAIRS
    over = _pr((_prim(),) * (KF.MAX_PRIM_PAIRS + 1))
    with pytest.raises(ValueError, match="MAX_PRIM_PAIRS"):
        KF.pack([over])
    with pytest.raises(ValueError, match="degree"):
        KF.pack([_pr((_prim(),)), _pr((_prim(), _prim()))])
    with pytest.raises(ValueError, match="empty"):
        KF.pack([])
    with pytest.raises(ValueError, match="non-negative"):
        KF.pack([_pr((_prim(),), sh=(0, -1))])


def test_the_packed_layout_is_row_major() -> None:
    a = KF.PrimRec((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), 7.0, 8.0)
    b = KF.PrimRec((9.0, 10.0, 11.0), (12.0, 13.0, 14.0), 15.0, 16.0)
    p = KF.pack([KF.PairRec((0.1, 0.2, 0.3), 0.5, (a, b), (2, 3), (4, 7))])
    assert (p.n, p.degree) == (1, 2)
    assert p.prim == tuple(float(x) for x in range(1, 17))
    assert p.sep == (0.1, 0.2, 0.3) and p.G == (0.5,)
    assert p.sh == (2, 3) and p.ao == (4, 7)


def test_density_blocks_and_close() -> None:
    D = [[1.0, -3.0, 0.5], [-3.0, 2.0, 0.25], [0.5, 0.25, -4.0]]
    assert KF.density_blocks(D, (0, 1), (1, 2)) == (1.0, 3.0, 3.0, 4.0)
    F = [[1.0, 2.0], [3.0, 4.0]]
    assert KF.close(F) == ((2.0, 5.0), (5.0, 8.0))
    with pytest.raises(ValueError, match="square"):
        KF.close([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# Records from shells, the host path, and the fused kernel interpreted
# ---------------------------------------------------------------------------


def _sub(u, v):
    return tuple(a - b for a, b in zip(u, v, strict=True))


def records(basis: Basis, a: int, b: int, G: float = 1.0,
            keep=None) -> KF.PairRec:
    """The contracted pair `(a, b)` of `basis`: primitive pairs in
    `ao.pairs.primitive_pairs` order (i outer) or the indices `keep`."""
    sa, sb = basis.shells[a], basis.shells[b]
    A, B = sa.centre, sb.centre
    AB = _sub(A, B)
    ab2 = sum(x * x for x in AB)
    prims = []
    for x, ca in zip(sa.exponents, sa.coefficients, strict=True):
        for y, cb in zip(sb.exponents, sb.coefficients, strict=True):
            p = x + y
            P = tuple((x * u + y * v) / p for u, v in zip(A, B, strict=True))
            U = ca * cb * (math.pi / p) ** 1.5 * math.exp(-x * y / p * ab2)
            prims.append(KF.PrimRec(P, _sub(P, A), p, U))
    if keep is not None:
        prims = [prims[i] for i in keep]
    return KF.PairRec(AB, G, tuple(prims), (a, b),
                      (basis.offsets[a], basis.offsets[b]))


def _scalars(b: KF.PrimRec, k: KF.PrimRec, AB, CD) -> dict:
    """The fused kernel's per-quartet scalars, in ITS operation order."""
    p, q = b.zeta, k.zeta
    PQ = tuple(x - y for x, y in zip(b.P, k.P, strict=True))
    ppq = p + q
    rho = (p * q) / ppq
    q_ppq, p_ppq = q / ppq, p / ppq
    inv2p, inv2q = 0.5 / p, 0.5 / q
    s = {"inv2p": inv2p, "inv2q": inv2q, "inv2pq": 0.5 / ppq,
         "inv2p_alpha_over_p": inv2p * (rho / p),
         "inv2q_alpha_over_q": inv2q * (rho / q),
         "theta": math.sqrt(rho), "pair": b.U * k.U}
    for d, ax in enumerate("xyz"):
        s[f"PA_{ax}"] = b.X[d]
        s[f"QC_{ax}"] = k.X[d]
        s[f"WP_{ax}"] = 0.0 - q_ppq * PQ[d]
        s[f"WQ_{ax}"] = p_ppq * PQ[d]
        s[f"PQ_{ax}"] = PQ[d]
        s[f"AB_{ax}"] = AB[d]
        s[f"CD_{ax}"] = CD[d]
    return s


def _fspec(cls) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(cls), Target.F)


@cache
def _vrr_kernel(cls):
    return R.build_vrr_kernel(_fspec(cls))


@cache
def _hrr_kernel(cls):
    return R.build_hrr_kernel(_fspec(cls))


@cache
def _kernel(cls, weights=KF.WEIGHTS, eris=False) -> Kernel:
    return KF.build_f_kernel(cls, weights=weights, eris=eris)


def host(cls, bra: KF.PairRec, ket: KF.PairRec) -> np.ndarray:
    """The host path (module docstring): #279's two kernels, the fold."""
    n = len(R.boundary_order(cls))
    c = np.zeros(n)
    for kp in ket.prims:  # ket outer, as the fused nest
        for bp in bra.prims:
            s = _scalars(bp, kp, bra.sep, ket.sep)
            v = interpret(_vrr_kernel(cls), boys=boys,
                          **{x: s[x] for x in R.VRR_SCALARS},
                          out=np.zeros(n))["out"]
            c = c + v
    s = _scalars(bra.prims[0], ket.prims[0], bra.sep, ket.sep)
    return interpret(_hrr_kernel(cls), **{x: s[x] for x in R.HRR_SCALARS},
                     boundary=c, out=np.zeros(len(hrr.targets(cls))))["out"]


def fused(cls, basis, D, bras, kets, *, diag=0, tau=0.0, threads=2,
          weights=KF.WEIGHTS, eris=False, F0=None, g0=None, Dsh=None):
    """Interpret the fused kernel over one launch. Returns `(F, g, it)`:
    `F` NOT closed (`n_ao x n_ao`), `g` the tap (`None` without it)."""
    k = _kernel(tuple(cls), weights, eris)
    pb, pk = KF.pack(bras), KF.pack(kets)
    n_eri = len(hrr.targets(cls))
    n_sh, n_ao = len(basis.shells), basis.n_ao
    D = np.asarray(D, dtype=float)
    if Dsh is None:
        Dsh = KF.density_blocks(D, basis.offsets,
                                [s.n_cart for s in basis.shells])
    every = {
        "n_bra": pb.n, "n_ket": pk.n, "k_ab": pb.degree, "k_cd": pk.degree,
        "n_sh": n_sh, "n_ao": n_ao, "diag": int(diag), "tau": float(tau),
        "bra_prim": np.array(pb.prim), "AB": np.array(pb.sep),
        "G_ab": np.array(pb.G),
        "bra_sh": np.array(pb.sh, dtype=np.int32),
        "bra_ao": np.array(pb.ao, dtype=np.int32),
        "ket_prim": np.array(pk.prim), "CD": np.array(pk.sep),
        "G_cd": np.array(pk.G),
        "ket_sh": np.array(pk.sh, dtype=np.int32),
        "ket_ao": np.array(pk.ao, dtype=np.int32),
        "Dsh": np.array(Dsh, dtype=float).ravel(), "D": D.ravel(),
        KF.FOCK: (np.zeros(n_ao * n_ao) if F0 is None
                  else np.asarray(F0, dtype=float).ravel().copy()),
        KF.OUT: (np.zeros(pb.n * pk.n * n_eri) if g0 is None else g0.copy()),
    }
    names = KF.params_of(cls, eris=eris)
    grid, block = KF.launch_shape(pb.n, threads)
    it = Interpreter(k, launch=Launch(grid, block, 32), boys=boys)
    out = it.run(**{n: every[n] for n in names})
    F = out[KF.FOCK].reshape(n_ao, n_ao)
    g = out[KF.OUT].reshape(pb.n, pk.n, n_eri) if eris else None
    return F, g, it


def mmd(cls, shells) -> tuple[np.ndarray, np.ndarray]:
    blk, scale = eri_block_and_scale((shells[0], shells[1]),
                                     (shells[2], shells[3]), pq=pq)
    return np.array(blk).ravel(), np.array(scale).ravel()


CENTRES = ((0.0, 0.0, 0.0), (0.4, -0.3, 0.5), (-0.2, 0.6, 0.1),
           (0.7, 0.2, -0.4))
EXPS = ((1.3, 0.35), (0.7, 2.2), (0.9, 0.3), (1.6, 0.5))
COEFS = ((0.6, 0.45), (0.8, -0.3), (0.5, 0.7), (0.9, 0.25))

# The same bound and reason as test_hgp_recurrence.py's TOL (two
# exact-arithmetic-equal recurrences in different operation orders, the
# HRR's cancellation unseen by the |.| scale): 1e-13 of scale.
TOL = 1e-13
LIBCINT_TOL = 1e-12  # test_ao_fock.py's, against the fixtures


def _shells(cls, centres=CENTRES):
    return [Shell(am, c, e, k) for am, c, e, k in
            zip(cls, centres, EXPS, COEFS, strict=True)]


def _density(n_ao: int, seed: int = 283) -> np.ndarray:
    """A symmetric, sign-mixed density with no structure to hide behind."""
    r = np.random.default_rng(seed).uniform(-1.0, 1.0, (n_ao, n_ao))
    return r + r.T


@pytest.mark.parametrize("cls", LADDER)
def test_one_contracted_quartet_is_the_host_path_bitwise(cls) -> None:
    """All 21 classes on one contracted quartet of 2-primitive shells
    (`K_AB = K_CD = 4`, 16 primitive quartets): the tap's `g` `==` the host
    path, and against `ao/eri.py`'s MMD integrals to rounding."""
    sh = _shells(cls)
    basis = Basis(tuple(sh))
    bra, ket = records(basis, 0, 1), records(basis, 2, 3)
    _, g, _ = fused(cls, basis, _density(basis.n_ao), [bra], [ket],
                    threads=1, eris=True)
    want = host(cls, bra, ket)
    assert np.array_equal(g[0, 0], want)
    ref, scale = mmd(cls, sh)
    err = np.abs(g[0, 0] - ref)
    assert np.all(err <= TOL * scale), float(np.max(err / scale))


def _moved(dx):
    return tuple(tuple(x + dx[i % 3] * (j + 1) for i, x in enumerate(c))
                 for j, c in enumerate(CENTRES))


@pytest.mark.parametrize("cls", [_speed((0, 1, 1, 1)), _speed((1, 2, 1, 2))])
def test_a_multi_quartet_ket_loop_is_bitwise_and_the_screen_skips(cls) -> None:
    """Two bra pairs (two blocks) by three ket pairs strided over two
    threads. The MIDDLE ket is screened (`G_cd = 0 < tau`): its `g` row is
    untouched and the ket after it is still computed -- a break would have
    left it -- while the ket loop is entered for every ket and the primitive
    loops only for the unscreened ones (`fock/skip-not-break`, behaviour)."""
    geoms = [CENTRES, _moved((0.3, -0.2, 0.1)), _moved((-0.5, 0.4, 0.2))]
    shells = [s for g in geoms for s in _shells(cls, g)]
    basis = Basis(tuple(shells))
    bras = [records(basis, 0, 1), records(basis, 4, 5)]
    kets = [records(basis, 4 * i + 2, 4 * i + 3, G=(0.0 if i == 1 else 1.0))
            for i in range(3)]
    g0 = np.full(len(bras) * len(kets) * len(hrr.targets(cls)), -7.0)
    _, g, it = fused(cls, basis, _density(basis.n_ao), bras, kets,
                     tau=1e-12, threads=2, g0=g0, eris=True)
    for ib, b in enumerate(bras):
        for ik, k in enumerate(kets):
            if ik == 1:
                assert np.all(g[ib, ik] == -7.0)  # skipped: never written
            else:
                assert np.array_equal(g[ib, ik], host(cls, b, k))
    assert it.iterations[KF.KET] == len(bras) * len(kets)
    assert it.iterations[KF.KCD] == len(bras) * 2 * len(kets[0].prims)


# ---------------------------------------------------------------------------
# The digestion against ao.fock's ordered sweep
# ---------------------------------------------------------------------------

HGP_BOTH = partial(HD.eri_block_and_scale, boys=boys)


def _members(bras, kets) -> frozenset:
    """Every ORDERED shell quartet the launch's unique quartets stand for,
    by brute force over labels: bra an ordering of one pair and ket an
    ordering of the other, for each (bra pair, ket pair) met."""
    out = set()
    for P in bras:
        for Q in kets:
            for X, Y in ((P, Q), (Q, P)):
                for p, q in {X, X[::-1]}:
                    for r, s in {Y, Y[::-1]}:
                        out.add((p, q, r, s))
    return frozenset(out)


def sweep(basis: Basis, D, members=None) -> tuple[np.ndarray, np.ndarray]:
    """`ao.fock`'s sweep, every ordered quartet at weight 1, with its `eri`
    returning zeros outside `members` (all quartets when `None`). Returns
    `(G, scale_G)`: `G = J - K/2` and the same sweep on `|D|` with the
    absolute-value twin, `scale_J + scale_K / 2`."""
    idx = {id(s): i for i, s in enumerate(basis.shells)}
    memo: dict = {}

    def both(bra, ket):
        q = (idx[id(bra[0])], idx[id(bra[1])], idx[id(ket[0])],
             idx[id(ket[1])])
        if q not in memo:
            if members is not None and q not in members:
                z = [[0.0] * (ket[0].n_cart * ket[1].n_cart)
                     for _ in range(bra[0].n_cart * bra[1].n_cart)]
                memo[q] = (z, z)
            else:
                memo[q] = HGP_BOTH(bra, ket)
        return memo[q]

    D = np.asarray(D, dtype=float)
    J, K = fock.coulomb_exchange_reference(
        basis, D.tolist(), eri=lambda b, k: both(b, k)[0])
    Js, Ks = fock.coulomb_exchange_reference(
        basis, np.abs(D).tolist(), eri=lambda b, k: both(b, k)[1])
    f = fock.EXCHANGE_FRACTION
    return (np.array(J) - f * np.array(K), np.array(Js) + f * np.array(Ks))


def _worst(got, want, scale) -> float:
    return float(np.max(np.abs(got - want) / np.where(scale > 0, scale, 1.0)))


def _check_launch(cls, basis, D, bra_ix, ket_ix, *, diag, weights=KF.WEIGHTS,
                  threads=2):
    """Run one launch, close it, and return `(worst, G, scale)` against the
    sweep restricted to the launch's orbits."""
    bras = [records(basis, *p) for p in bra_ix]
    kets = [records(basis, *p) for p in ket_ix]
    F, _, it = fused(cls, basis, D, bras, kets, diag=diag, threads=threads,
                     weights=weights)
    G = np.array(KF.close(F))
    want, scale = sweep(basis, D, _members(bra_ix, ket_ix))
    assert np.max(np.abs(want)) > 0  # the comparison is not of zeros
    return _worst(G, want, scale), G, want, scale, it


@pytest.mark.parametrize("cls", LADDER)
def test_one_orbit_equals_its_ordered_members_in_the_sweep(cls) -> None:
    """All 21 classes, four distinct shells: one unique quartet's closed
    deposit `==` the sweep over its 8 ordered members to `TOL` of scale; and
    the predecessor's Coulomb weight (1 for 2) FAILS the same comparison."""
    basis = Basis(tuple(_shells(cls)))
    D = _density(basis.n_ao)
    worst, *_ = _check_launch(cls, basis, D, [(0, 1)], [(2, 3)], diag=0)
    assert worst <= TOL, worst
    bad, *_ = _check_launch(cls, basis, D, [(0, 1)], [(2, 3)], diag=0,
                            weights=KF.Weights(1.0, -0.5))
    assert bad > 1e6 * TOL, bad


def _same_l_shells(l, n):  # noqa: E741
    return [Shell(l, CENTRES[i], EXPS[i], COEFS[i]) for i in range(n)]


# (class, shells' l, bra pairs, ket pairs, diag): every degeneracy case, and
# a shell shared across bra and ket with no degeneracy (orbit still 8).
COINCIDENCE = {
    "shared-shell": ((1, 1, 1, 1), 3, [(0, 1)], [(0, 2)], 0),
    "a=b": ((1, 1, 1, 1), 3, [(0, 0)], [(1, 2)], 0),
    "c=d": ((1, 1, 1, 1), 3, [(0, 1)], [(2, 2)], 0),
    "ab=cd": ((1, 1, 1, 1), 3, [(0, 1)], [(0, 1)], 1),
    "a=b,c=d": ((1, 1, 1, 1), 3, [(0, 0)], [(1, 1)], 0),
    "all-three": ((1, 1, 1, 1), 1, [(0, 0)], [(0, 0)], 1),
    "diagonal-batch": ((1, 1, 1, 1), 2, [(0, 0), (0, 1), (1, 1)],
                       [(0, 0), (0, 1), (1, 1)], 1),
    "ss-all-three": ((0, 0, 0, 0), 1, [(0, 0)], [(0, 0)], 1),
    "ss-diagonal-batch": ((0, 0, 0, 0), 3, [(0, 0), (0, 2), (1, 2), (2, 2)],
                          [(0, 0), (0, 2), (1, 2), (2, 2)], 1),
    "dd-diagonal-batch": ((2, 2, 2, 2), 2, [(0, 0), (0, 1), (1, 1)],
                          [(0, 0), (0, 1), (1, 1)], 1),
}


@pytest.mark.parametrize("case", [
    pytest.param(c, id=c, marks=(pytest.mark.slow,)
                 if sum(COINCIDENCE[c][0]) > 4 else ()) for c in COINCIDENCE])
def test_every_degeneracy_case_equals_the_sweep(case) -> None:
    """`a = b`, `c = d`, `(ab) = (cd)`, the first two together, and all
    three, each run explicitly (and a diagonal batch, where the canonical
    half decides which of `(bra, k)` and `(k, bra)` runs): `==` the sweep
    over the orbits' ordered members. Dropping the degeneracy factor fails
    every case that has one."""
    cls, n, bra_ix, ket_ix, diag = COINCIDENCE[case]
    basis = Basis(tuple(_same_l_shells(cls[0], n)))
    D = _density(basis.n_ao)
    worst, *_ = _check_launch(cls, basis, D, bra_ix, ket_ix, diag=diag)
    assert worst <= TOL, worst
    if case != "shared-shell":
        bad, *_ = _check_launch(cls, basis, D, bra_ix, ket_ix, diag=diag,
                                weights=KF.Weights(2.0, -0.5, False))
        assert bad > 1e6 * TOL, (case, bad)


def test_a_mixed_class_with_both_pairs_degenerate() -> None:
    """`(ss|pp)` with `a = b` and `c = d` (different `l`, so `P != Q` by
    build-time fact) -- the bra and ket flags without the diagonal one."""
    basis = Basis((Shell(0, CENTRES[0], EXPS[0], COEFS[0]),
                   Shell(1, CENTRES[1], EXPS[1], COEFS[1])))
    D = _density(basis.n_ao)
    worst, *_ = _check_launch((0, 0, 1, 1), basis, D, [(0, 0)], [(1, 1)],
                              diag=0)
    assert worst <= TOL, worst


def test_the_canonical_half_runs_each_unique_quartet_once() -> None:
    """A diagonal launch of three pairs over two threads: the ket loop is
    entered for all 9 (bra, k), the primitive nest only for the 6 with
    `k >= bra`."""
    basis = Basis(tuple(_same_l_shells(0, 2)))
    pairs = [(0, 0), (0, 1), (1, 1)]
    *_, it = _check_launch((0, 0, 0, 0), basis, _density(basis.n_ao), pairs,
                           pairs, diag=1)
    assert it.iterations[KF.KET] == 9
    # each unscreened quartet runs the K_CD = 2 x 2 = 4 ket primitive pairs
    assert it.iterations[KF.KCD] == 6 * 4


@pytest.mark.parametrize("wrong", ["coulomb-1", "exchange-sign",
                                   "no-degeneracy", "fig1-printed"])
def test_each_wrong_weight_is_caught(wrong) -> None:
    """The mutation test: the predecessor's bug (Coulomb 1 for 2), a flipped
    exchange sign, a dropped degeneracy factor and Fig. 1's printed unit
    weights each miss the sweep by far more than the tolerance, on the
    all-three-coincident diagonal batch where every factor is live."""
    weights = {"coulomb-1": KF.Weights(1.0, -0.5),
               "exchange-sign": KF.Weights(2.0, 0.5),
               "no-degeneracy": KF.Weights(2.0, -0.5, False),
               "fig1-printed": KF.Weights(1.0, 1.0, False)}[wrong]
    cls, n, bra_ix, ket_ix, diag = COINCIDENCE["diagonal-batch"]
    basis = Basis(tuple(_same_l_shells(cls[0], n)))
    D = _density(basis.n_ao)
    bad, *_ = _check_launch(cls, basis, D, bra_ix, ket_ix, diag=diag,
                            weights=weights)
    assert bad > 1e6 * TOL, (wrong, bad)


# ---------------------------------------------------------------------------
# The screen: six blocks, weighted, skip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", KF.BLOCKS, ids=lambda b: "abcd"[b[0]]
                         + "abcd"[b[1]])
def test_each_of_the_six_blocks_can_keep_a_quartet(block) -> None:
    """With `G_ab = G_cd = 1` and `tau = 1`, a `Dsh` that is zero except on
    ONE block keeps the quartet exactly when that block's `|w| Dsh >= tau`:
    0.6 keeps it through a Coulomb block (`2 x 0.6`) and not through an
    exchange one (`0.6 / 2`); 2.5 keeps it through any. All zero skips it,
    and a skipped quartet deposits nothing."""
    cls = (0, 0, 0, 0)
    basis = Basis(tuple(_same_l_shells(0, 4)))
    D = _density(basis.n_ao)
    bras, kets = [records(basis, 0, 1)], [records(basis, 2, 3)]

    def run(v):
        Dsh = np.zeros((4, 4))
        Dsh[block[0], block[1]] = Dsh[block[1], block[0]] = v
        F, _, it = fused(cls, basis, D, bras, kets, tau=1.0, Dsh=Dsh)
        return F, it.iterations.get(KF.KCD, 0)

    F0, n0 = run(0.0)
    assert n0 == 0 and not F0.any()
    coulomb = block in KF.BLOCKS[:2]
    assert (run(0.6)[1] > 0) == coulomb
    assert run(2.5)[1] > 0


# ---------------------------------------------------------------------------
# The AtomicAdd count against the ERI count
# ---------------------------------------------------------------------------


@pytest.mark.claim("fock/predigest-bounds-atomics")
def test_predigest_bounds_atomics_recorded() -> None:
    """`fock/predigest-bounds-atomics`, a COUNT of statements in the built
    IR, not a timing: per class the `AtomicAdd`s are EXACTLY the six blocks'
    elements, never the `n_a n_b n_c n_d` ERIs; and their ratio to the ERIs
    falls strictly whenever a class grows in any shell (componentwise larger
    Cartesian counts), from 6 at `(ss|ss)` to 1/6 at `(dd|dd)`. Below the
    crossing the atomics outnumber the ERIs (ATOMICS' comment); the claim
    held is the scaling, not a saving at every class."""
    ratio = {}
    for cls in CLASSES:
        c = KF.counted(_kernel(cls))
        assert c.atomic == KF.six_block_size(cls) == ATOMICS[cls]
        ratio[cls] = c.atomic / c.eri
    n = {c: [len(vrr.cartesians(x)) for x in c] for c in CLASSES}
    for c1, c2 in itertools.permutations(CLASSES, 2):
        if all(x <= y for x, y in zip(n[c1], n[c2], strict=True)):
            assert ratio[c2] < ratio[c1], (c1, c2)
    assert ratio[(0, 0, 0, 0)] == 6.0 and ratio[(2, 2, 2, 2)] == 1 / 6


# ---------------------------------------------------------------------------
# Whole molecules, from launch_f's pair stream
# ---------------------------------------------------------------------------

BOUND = partial(primitive_schwarz_bounds, one_centre=one_centre)


def _side(basis, batch):
    return [records(basis, p.a, p.b, G=p.G, keep=p.prims)
            for p in batch.pairs]


def fock_from_plan(basis, D, plan, *, weights=KF.WEIGHTS, threads=4):
    """Every launch of `plan` interpreted into one `F`, then closed."""
    F = np.zeros((basis.n_ao, basis.n_ao))
    for L in plan:
        F, _, _ = fused(L.spec.am, basis, D, _side(basis, plan.batches[L.bra]),
                        _side(basis, plan.batches[L.ket]),
                        diag=int(L.bra == L.ket), tau=plan.tau,
                        threads=threads, weights=weights, F0=F)
    return np.array(KF.close(F))


def _molecule(name):
    pair = manifest.fock_pair(name)  # asserts vj and vk share their inputs
    basis = ob.from_fixture(pair.j)
    return basis, np.array(pair.j.density), pair.two_electron()


@pytest.mark.parametrize("name", [
    "h2_sto3g", pytest.param("h2o_sto3g", marks=pytest.mark.slow)])
def test_a_whole_molecule_against_the_sweep_and_libcint(name) -> None:
    """The plan at `tau = 0`, every launch interpreted, closed: against the
    FULL ordered sweep to `TOL` of scale, and libcint's `vj - vk/2` to
    `LIBCINT_TOL`. H2/STO-3G is one diagonal launch of three pairs -- `a =
    b`, `c = d`, `(ab) = (cd)` and all three in one fixture."""
    basis, D, libcint = _molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=0.0, n=1)
    G = fock_from_plan(basis, D, plan)
    want, scale = sweep(basis, D)
    assert _worst(G, want, scale) <= TOL
    assert _worst(G, libcint, scale) <= LIBCINT_TOL


@pytest.mark.claim("fock/fig1-digestion-unweighted")
def test_fig1_digestion_unweighted_disagrees_with_the_sweep() -> None:
    """`fock/fig1-digestion-unweighted`: Fig. 1's panel as printed -- six
    buffers, every one at weight 1, no degeneracy factor -- run over the
    unique quartets of
    H2/STO-3G (a fixture with diagonal quartets) disagrees with
    `fock_build_reference`'s sweep, while the derived weights agree."""
    basis, D, _ = _molecule("h2_sto3g")
    plan = f_launches(basis, D, bound=BOUND, tau=0.0, n=1)
    want, scale = sweep(basis, D)
    # this file's `sweep` IS fock_build_reference, unrestricted
    assert np.array_equal(want, np.array(fock.fock_build_reference(
        basis, D.tolist(), eri=partial(HD.eri_block, boys=boys))))
    printed = fock_from_plan(basis, D, plan,
                             weights=KF.Weights(1.0, 1.0, False))
    derived = fock_from_plan(basis, D, plan)
    assert _worst(derived, want, scale) <= TOL
    assert _worst(printed, want, scale) > 1e6 * TOL


def test_a_launch_from_the_pair_stream_runs_against_mmd() -> None:
    """H2O/STO-3G at `tau = 0`: one launch of `f_launches`' (sp|sp) class
    (`K_AB = K_CD = 9`), each side packed from its batch's `FPair.prims`
    indices -- multi-quartet, both sides, real shells -- the tap bitwise
    against the host path and to rounding against `ao/eri.py`."""
    fx = manifest.load(manifest.FIXTURES / "h2o_sto3g_j.json")
    basis, D = ob.from_fixture(fx), np.array(fx.density)
    plan = f_launches(basis, D, bound=BOUND, tau=0.0, n=1)
    launch = next(L for L in plan if L.spec.am == (0, 1, 0, 1))
    cls = launch.spec.am
    sh = basis.shells
    bb, kb = plan.batches[launch.bra], plan.batches[launch.ket]
    bras, kets = _side(basis, bb), _side(basis, kb)
    assert KF.pack(bras).degree == bb.degree == 9
    _, g, _ = fused(cls, basis, D, bras, kets, threads=4, eris=True,
                    diag=int(launch.bra == launch.ket))
    for ib, (b, pb) in enumerate(zip(bras, bb.pairs, strict=True)):
        for ik, (k, pk) in enumerate(zip(kets, kb.pairs, strict=True)):
            if launch.bra == launch.ket and ik < ib:
                continue  # the canonical half: never computed, never written
            assert np.array_equal(g[ib, ik], host(cls, b, k))
            ref, scale = mmd(cls, [sh[pb.a], sh[pb.b], sh[pk.a], sh[pk.b]])
            assert np.all(np.abs(g[ib, ik] - ref) <= TOL * scale)
