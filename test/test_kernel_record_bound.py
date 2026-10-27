"""kernel_record_bound.py: K's dressed record bound, interpreted (#499).

THE WITNESS is `eri.k_exchange.k.record_bounds(epair(sa, sb), pq=reference.pq.pq)`: the
host forms each `I_p` from the `[p|q]` recurrence at `PQ = 0`, the dressed
`E` as `s * e_block`, and sums each row with `fsum`; the kernel inlines
`e_fragment`'s `E`, uses `kernel_pair_bound`'s closed form one `p` at a time
and a plain left-to-right sum. Every term is non-negative, so they agree to
rounding, not bitwise: the bar is `REL_TOL = 16 eps` of the bound, per
record, `PAIR_BOUND`'s bar; a record whose host bound is exactly `0.0` (the
pair factor underflowed) must be exactly `0.0` here too. Measured when
written: worst 2.82 eps on H2O/cc-pVDZ, 2.39 eps on the H10 chain.

THE INPUTS, as a consumer stages an ordered record of `(sa, sb)`: `zeta_ab`,
`XPA = P - A` (A the FIRST shell's centre), `XPB = P - B`, and `s_ab = c_a
c_b (pi/zeta)^{3/2}` (`ao.density.epair`'s `s`).

Cases: every ordered class `la, lb <= 3` on contracted shells with signed
coefficients, `la + lb` above `tiers.FAST_L_MAX` in the slow tier; and every
record of every ordered shell pair of H2O/cc-pVDZ and of #486's H10 chain in
cc-pVDZ (both orientations, since every ordered pair is a record set); and,
slow, every record of CO/cc-pVTZ, whose f shells reach (ff).

THE LOW-STACK FORM (#586, stage 3), the default from f up: its selection
rule, the d text byte-identical (sha256 from 534a8e7, before stage 3), the
two forms' equal name, parameters and launch, its local arrays inside the
8 KB budget, and BOTH forms held to the host at the same `16 eps` at every
class through (ff), and to each other.
"""

from __future__ import annotations

import hashlib
import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import split_by_l

from felizjueves.ao.density import epair
from felizjueves.ao.pairs import hermite_bound, pair_bound
from felizjueves.ao.shell import Shell
from felizjueves.emit import cuda, hip
from felizjueves.eri.k_exchange import kernel_record_bound as krb
from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_record_bound import (
    LOWSTACK_MIN_L,
    PARAMS,
    build_record_bound_kernel,
    record_bound_lowstack,
    record_bound_name,
    record_bound_spec,
)
from felizjueves.eri.kernel_pair_bound import (
    hermite_bound_coefficients,
    pair_bound_coefficients,
)
from felizjueves.ir import Decl, Inline, walk
from felizjueves.ir.interp import Launch, interpret
from felizjueves.mmd import e_fragment as ef
from felizjueves.performance.device import GFX1200, SM_86
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.pq import one_centre, pq
from felizjueves.spec import Operator, Spec, Target

EPS = float(np.finfo(np.float64).eps)
REL_TOL = 16 * EPS

LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
CLASS_GRID = split_by_l(CLASSES, sum, ids=lambda c: f"{c[0]}{c[1]}")

MOLECULES = ("h2o_ccpvdz_k", "h10_chain_ccpvdz_k")
F_MOLECULES = ("co_ccpvtz_k",)  # #570's f case: classes through (ff)


# ---------------------------------------------------------------------------
# Staging and the two paths
# ---------------------------------------------------------------------------


Records = dict[tuple[int, int], tuple[dict[str, np.ndarray], np.ndarray]]


def stage(pairs) -> Records:
    """Every record of the ordered shell pairs `pairs`, grouped by ordered
    class `(la, lb)`: the kernel's inputs, and the host's bound per record."""
    by: dict[tuple[int, int], list] = {}
    for sa, sb in pairs:
        ep = epair(sa, sb)
        for (P, zeta, s), want in zip(ep.recs, record_bounds(ep, pq=pq),
                                      strict=True):
            by.setdefault((sa.l, sb.l), []).append((
                zeta, [P[d] - sa.centre[d] for d in range(3)],
                [P[d] - sb.centre[d] for d in range(3)], s, want))
    out: Records = {}
    for cls, rows in by.items():
        args = {"zeta_ab": np.array([r[0] for r in rows]),
                "XPA": np.array([r[1] for r in rows]),
                "XPB": np.array([r[2] for r in rows]),
                "s_ab": np.array([r[3] for r in rows])}
        out[cls] = args, np.array([r[4] for r in rows])
    return out


def run(cls: tuple[int, int], args: dict[str, np.ndarray],
        threads: int = 32, lowstack: bool | None = None) -> np.ndarray:
    """One interpreted launch; `lowstack` is `build_record_bound_kernel`'s
    (None: the class's default form)."""
    n = len(args["zeta_ab"])
    grid, block = krb.launch_shape(n, threads)
    out = interpret(build_record_bound_kernel(cls, lowstack=lowstack),
                    launch=Launch(grid, block, threads), n_rec=n,
                    **{k: v.copy() for k, v in args.items()},
                    bound=np.full(n, np.nan))
    return out["bound"]


def worst(got: np.ndarray, want: np.ndarray) -> float:
    """The worst relative error over the records, in units of the bound,
    after checking that every host zero is a kernel zero."""
    assert not np.isnan(got).any()
    assert np.all(want >= 0.0)
    zero = want == 0.0
    assert np.all(got[zero] == 0.0)
    if zero.all():
        return 0.0
    return float((np.abs(got - want)[~zero] / want[~zero]).max())


def _shell(rng: np.random.Generator, l: int, centre) -> Shell:  # noqa: E741
    n = 3
    exps = np.sort(rng.uniform(0.08, 30.0, n))[::-1]
    coefs = rng.uniform(-1.0, 1.0, n)
    return Shell(l, tuple(centre), tuple(exps), tuple(coefs))


@cache
def synthetic(cls: tuple[int, int]) -> Records:
    """Contracted shells with signed coefficients: one pair on one centre,
    and three at 0.4..3 bohr apart, in the class's order."""
    la, lb = cls
    rng = np.random.default_rng(499 + 17 * la + lb)
    pairs = []
    for k in range(4):
        A = rng.uniform(-1.0, 1.0, 3)
        B = A if k == 0 else A + rng.normal(size=3) / 1.7 * (0.4 + k)
        pairs.append((_shell(rng, la, A), _shell(rng, lb, B)))
    return stage(pairs)


@cache
def molecule(name: str) -> Records:
    basis = ob.from_fixture(manifest.load(manifest.FIXTURES / f"{name}.json"))
    return stage([(a, b) for a in basis.shells for b in basis.shells])


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_the_key_is_an_ordered_pair_class() -> None:
    assert record_bound_spec(2, 1) == Spec(Operator.COULOMB, 0, (2, 1),
                                           Target.RECORD_BOUND)
    assert record_bound_spec(2, 1).key() == \
        "coulomb/d0/(2,1)/b1/record_bound/literal"
    assert build_record_bound_kernel(record_bound_spec(2, 1)) == \
        build_record_bound_kernel((2, 1))
    assert build_record_bound_kernel((2, 1)).name == "record_bound_21" == \
        record_bound_name((2, 1))
    assert tuple(p.name for p in build_record_bound_kernel((0, 0)).params) \
        == PARAMS == ("n_rec", "zeta_ab", "XPA", "XPB", "s_ab", "bound")


def test_what_the_key_refuses() -> None:
    with pytest.raises(ValueError, match="not a Target.RECORD_BOUND"):
        build_record_bound_kernel(
            Spec(Operator.COULOMB, 0, (1, 1), Target.PAIR_BOUND))
    with pytest.raises(ValueError, match="never batched"):
        Spec(Operator.COULOMB, 0, (1, 1), Target.RECORD_BOUND, (), 3)
    with pytest.raises(ValueError):
        Spec(Operator.OVERLAP, 0, (1, 1), Target.RECORD_BOUND)
    with pytest.raises(TypeError):
        build_record_bound_kernel(4)  # type: ignore[arg-type]


def test_the_text_depends_on_the_order_of_the_pair() -> None:
    # Unlike PAIR_BOUND, whose text is a function of la + lb: the max runs
    # over the FIRST shell's Cartesians, so (1, 0) is not (0, 1).
    assert build_record_bound_kernel((1, 0), name="b") != \
        build_record_bound_kernel((0, 1), name="b")


# ---------------------------------------------------------------------------
# The per-p coefficients
# ---------------------------------------------------------------------------


def test_the_hermite_coefficients_are_the_pair_bounds_split_by_p() -> None:
    for L in range(9):
        cs = hermite_bound_coefficients(L)
        ts = hermite_targets(L)
        assert len(cs) == len(ts) and all(c > 0.0 for c in cs)
        for n, d in enumerate(pair_bound_coefficients(L)):
            part = math.fsum(c for c, t in zip(cs, ts, strict=True)
                             if sum(t) == n)
            assert abs(part - d) <= 4 * EPS * d, (L, n)
    with pytest.raises(ValueError):
        hermite_bound_coefficients(-1)


@pytest.mark.parametrize("zeta", [1e-3, 0.3, 2.0, 47.0, 1e5])
def test_each_coefficient_is_its_own_hermite_bound(zeta: float) -> None:
    x = math.sqrt(zeta)
    for L in range(5):
        for c, t in zip(hermite_bound_coefficients(L), hermite_targets(L),
                        strict=True):
            host = hermite_bound(zeta, t, one_centre=one_centre)
            assert abs(c * x ** sum(t) * math.sqrt(x) - host) <= \
                REL_TOL * host, (L, t)


# ---------------------------------------------------------------------------
# Interpreted against the host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASS_GRID)
def test_the_kernel_is_the_hosts_record_bounds(cls: tuple[int, int]) -> None:
    ((got_cls, (args, want)),) = synthetic(cls).items()
    assert got_cls == cls and np.all(want > 0.0)
    w = worst(run(cls, args), want)
    assert w <= REL_TOL, (cls, w / EPS)


@pytest.mark.parametrize("name", MOLECULES)
def test_every_record_of_a_molecule_is_the_hosts(name: str) -> None:
    recs = molecule(name)
    assert all(la <= 2 and lb <= 2 for la, lb in recs)
    w = max(worst(run(cls, args), want) for cls, (args, want) in recs.items())
    print(f"\n{name}: worst {w / EPS:.2f} eps over "
          f"{sum(len(v[1]) for v in recs.values())} records")
    assert w <= REL_TOL, (name, w / EPS)


def test_the_orientation_changes_the_value() -> None:
    # A (p, s) record maxes over three rows of one term each; the same
    # primitive pair as (s, p) sums the three in one row.
    a = Shell(1, (0.0, 0.0, 0.0), (1.3,), (1.0,))
    b = Shell(0, (0.3, -0.4, 1.1), (0.7,), (1.0,))
    (ps, (pa, wa)), = stage([(a, b)]).items()
    (sp, (pb, wb)), = stage([(b, a)]).items()
    assert (ps, sp) == ((1, 0), (0, 1))
    ga, gb = run(ps, pa), run(sp, pb)
    assert worst(ga, wa) <= REL_TOL and worst(gb, wb) <= REL_TOL
    assert gb[0] > ga[0] * (1 + 1e-6)


def _against_pair_bound(name: str) -> tuple[int, float, float]:
    """`(n above, worst ratio, worst ratio over pairs > FAR bohr apart)`:
    the kernel's dressed bound over `PAIR_BOUND`'s undressed one, per
    record, every ordered shell pair of `name`."""
    basis = ob.from_fixture(manifest.load(manifest.FIXTURES / f"{name}.json"))
    above, top, far = 0, 0.0, 0.0
    for sa in basis.shells:
        for sb in basis.shells:
            for (_, zeta, _), i in zip(epair(sa, sb).recs, run_pair(sa, sb),
                                       strict=True):
                r = i / pair_bound(zeta, sa.l + sb.l, one_centre=one_centre)
                above += r > 1.0
                top = max(top, r)
                if math.dist(sa.centre, sb.centre) > FAR:
                    far = max(far, r)
    return above, top, far


FAR = 15.0  # bohr; the H10 chain is 21.6 end to end


def test_the_dressed_bound_is_not_the_pair_bound() -> None:
    # What K's screen is proved against (eri.k_exchange.k, "THE SCREEN") and what
    # PAIR_BOUND gives differ both ways (FreeBreakfast#132's table): on
    # H2O/cc-pVDZ a dressed record exceeds the undressed bound (by 1.2, an
    # s shell with itself), and on the H10 chain the dressed bound
    # of a far pair is a small fraction of the undressed one, which is a
    # function of zeta alone and does not decay with |AB|.
    above, top, _ = _against_pair_bound("h2o_ccpvdz_k")
    assert above >= 1 and 1.1 < top < 1.3, (above, top)
    _, _, far = _against_pair_bound("h10_chain_ccpvdz_k")
    print(f"\nH10 chain: worst dressed / undressed over pairs > {FAR} bohr "
          f"apart: {far:.3g}")
    assert 0.0 < far < 1e-6  # measured 1.12e-07


def run_pair(sa: Shell, sb: Shell) -> np.ndarray:
    ((cls, (args, _)),) = stage([(sa, sb)]).items()
    return run(cls, args)


def test_the_lane_mapping_is_not_numerical() -> None:
    ((cls, (args, _)),) = synthetic((2, 1)).items()
    one = run(cls, args, threads=1)
    for threads in (4, 32):
        assert run(cls, args, threads=threads).tolist() == one.tolist()


def test_the_guard_leaves_spare_lanes_idle() -> None:
    # 9 records on a block of 32: the spare threads write nothing past
    # n_rec (the interpreter bounds-checks every write).
    ((cls, (args, want)),) = stage([(
        Shell(0, (0.0, 0.0, 0.0), (3.0, 1.0, 0.2), (0.2, 0.5, 0.4)),
        Shell(1, (0.0, 0.0, 1.4), (2.0, 0.5, 0.1), (0.3, 0.6, -0.2)),
    )]).items()
    assert len(want) == 9
    assert worst(run(cls, args, threads=32), want) <= REL_TOL


@pytest.mark.slow
@pytest.mark.parametrize("name", F_MOLECULES)
def test_every_record_of_an_f_molecule_is_the_hosts(name: str) -> None:
    recs = molecule(name)
    assert max(max(c) for c in recs) == 3 and (3, 3) in recs
    w = max(worst(run(cls, args), want) for cls, (args, want) in recs.items())
    print(f"\n{name}: worst {w / EPS:.2f} eps over "
          f"{sum(len(v[1]) for v in recs.values())} records")
    assert w <= REL_TOL, (name, w / EPS)


# ---------------------------------------------------------------------------
# The low-stack form (#586, stage 3): from f up by default, the d text
# unchanged
# ---------------------------------------------------------------------------

# sha256 of the unrolled text through (dd), cuda and hip, as emitted at
# 534a8e7 (before stage 3): the low-stack form must not move a byte of it.
D_DIGESTS = {
    (0, 0): ("45d148d8b87197939371d8a1aa67ee949d45c0bd898a4b221e7a243a0ac4e397",
             "bb4e5ee6f24dfcd2bf80cb51ea102ddd05632a350ce5724d820cdb869410f956"),
    (0, 1): ("da309ac0b6529e6a348b54b50f52c80e19e3ad5fae6ffb42fd1e2eed4d22ba19",
             "1899a9c52e2c4e459a99c5ae1fa24de800690853189f5daf27640979e766c08b"),
    (0, 2): ("fcab6f3f5283489e545311c86190e260bad6135b16291f73c929d6fbb93f74bd",
             "799b5b1d26ffefae1118af086c26d21cf1fde9963198d1ba50d013dee3625609"),
    (1, 0): ("28330731794ee934a573634a39eea2721b204e83b9daa5c09fd273da555deead",
             "acf207f768c7224bba1d19907af8c6c76958cca933ff32862f67f36f027d20a1"),
    (1, 1): ("09a25dec9fd14396de3996245d54c014b184534c332bc1aa3b8c4b5f99a2003e",
             "5c13942bc21b2ee8fc0ca92bd261f77b9e6c0ad7b571c64cba46b491a1220a8d"),
    (1, 2): ("42cb39958371d150d087637a53528101f0c41713e540be1916e19aba73618f42",
             "3edee60f4a92d01e6fd9bcdeccda291188394d5c9e62750d008b4d2c3d1c3391"),
    (2, 0): ("d610038db3a7ea78781bee4842641460eaa55263864b1cd8912681d5f97e0d84",
             "2c041144768f32721a110f050851827cfd617939198993bc0c78b7f9ed015b7b"),
    (2, 1): ("43e24b372c47e341dda8937b21a65ea33283c9e17f9dc385135d32e16c67a685",
             "ed94087a5d2ab34d61976508d1164bcedadb8bc861f4a02f8d0578d2b318a4b4"),
    (2, 2): ("f897e3a2d496e11d91ea958c0f401dff727a6b1d09c9a8052d7206998893aca8",
             "f93a6e7d78f376827b96fe120a6f65995491dd75d0f17dd38f153ea2f55152f9"),
}

# #586's budget: at most 8 KB of stack per thread on sm_86 for every f
# unit. The kernel's local arrays are the whole of what it can spill by
# design, so they are held to it here, without a compiler (the measured
# frames are in kernel_record_bound.py's docstring).
STACK_BUDGET = 8192

F_CLASSES = [(0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]


def test_the_selection_rule_is_f_and_above() -> None:
    assert LOWSTACK_MIN_L == 3
    for la in range(5):
        for lb in range(5):
            assert record_bound_lowstack((la, lb)) is (max(la, lb) >= 3)
            assert record_bound_lowstack(record_bound_spec(la, lb)) is \
                record_bound_lowstack((la, lb))


@pytest.mark.parametrize("cls", sorted(D_DIGESTS),
                         ids=lambda c: f"{c[0]}{c[1]}")
def test_the_d_text_is_byte_identical(cls: tuple[int, int]) -> None:
    k = build_record_bound_kernel(cls)
    assert k == build_record_bound_kernel(cls, lowstack=False)
    got = tuple(hashlib.sha256(t.encode()).hexdigest()
                for t in (cuda.emit(k, SM_86), hip.emit(k, GFX1200)))
    assert got == D_DIGESTS[cls]


@pytest.mark.parametrize("cls", [(1, 0), (2, 2), *F_CLASSES],
                         ids=lambda c: f"{c[0]}{c[1]}")
def test_both_forms_share_the_name_parameters_and_launch(
        cls: tuple[int, int]) -> None:
    # The ABI free-breakfast's launcher binds: the same kernel name, the
    # same parameters (names, types, intents, shapes, in order), the same
    # one-thread-per-record index and guard.
    lo = build_record_bound_kernel(cls, lowstack=True)
    hi = build_record_bound_kernel(cls, lowstack=False)
    assert lo.name == hi.name == record_bound_name(cls)
    assert lo.params == hi.params
    assert tuple(p.name for p in lo.params) == PARAMS
    assert build_record_bound_kernel(cls) == \
        (lo if record_bound_lowstack(cls) else hi)
    assert lo.body[2] == hi.body[2] and lo.body[2].name == krb.REC
    assert lo.body[3].cond == hi.body[3].cond
    assert lo.body[1:3] == hi.body[1:3]
    assert build_record_bound_kernel(cls, lowstack=True, name="x").name == "x"


def _local_doubles(k) -> dict[str, int]:
    return {s.name: math.prod(d.value for d in s.shape)
            for s in walk(k.body) if isinstance(s, Decl) and s.shape}


@pytest.mark.parametrize("cls", F_CLASSES, ids=lambda c: f"{c[0]}{c[1]}")
def test_the_lowstack_form_holds_no_e_block(cls: tuple[int, int]) -> None:
    la, lb = cls
    k = build_record_bound_kernel(cls)
    arrays = _local_doubles(k)
    ext = ef.axis_table_extent(la, lb)
    assert arrays == {"Ex": ext, "Ey": ext, "Ez": ext,
                      "I": len(hermite_targets(la + lb))}
    assert ef.E_ARRAY not in arrays
    assert 8 * sum(arrays.values()) <= STACK_BUDGET
    (inl,) = [s for s in walk(k.body, into_inline=False)
              if isinstance(s, Inline)]
    assert inl.fragment == ef.axis_tables_fragment(la, lb)
    # the unrolled form, for contrast, holds the whole block
    rows, cols = ef.block_shape(la, lb)
    assert _local_doubles(build_record_bound_kernel(cls, lowstack=False)) \
        == {ef.E_ARRAY: rows * cols}


def test_the_ff_lowstack_arrays_are_420_doubles() -> None:
    # 3 x 112 axis-table entries and 84 I_p: 3,360 B, sm_86's whole frame
    # at (ff) (measured), against the unrolled form's 76,608 B.
    assert sum(_local_doubles(build_record_bound_kernel((3, 3))).values()) \
        == 420


@pytest.mark.parametrize("cls", CLASS_GRID)
def test_the_other_form_is_the_hosts_record_bounds(
        cls: tuple[int, int]) -> None:
    """Every class through (ff) in the form `build_record_bound_kernel`
    does NOT choose by default -- the low-stack form through d, the unrolled
    one at f -- against the host at the same `REL_TOL`: with the
    default-form test above, both forms are held at every class."""
    ((_, (args, want)),) = synthetic(cls).items()
    other = not record_bound_lowstack(cls)
    w = worst(run(cls, args, lowstack=other), want)
    assert w <= REL_TOL, (cls, w / EPS)


@pytest.mark.parametrize("cls", CLASS_GRID)
def test_the_two_forms_agree_with_each_other(cls: tuple[int, int]) -> None:
    ((_, (args, _)),) = synthetic(cls).items()
    lo = run(cls, args, lowstack=True)
    hi = run(cls, args, lowstack=False)
    assert worst(lo, hi) <= REL_TOL, (cls, worst(lo, hi) / EPS)


def test_the_lowstack_form_keeps_an_exact_zero() -> None:
    # A record whose pair factor underflows: every E entry is +0.0, so the
    # bound must be exactly 0.0 in both forms, as the host's is.
    a = Shell(3, (0.0, 0.0, 0.0), (40.0,), (1.0,))
    b = Shell(3, (0.0, 0.0, 9.0), (40.0,), (-1.0,))
    ((cls, (args, want)),) = stage([(a, b)]).items()
    assert cls == (3, 3) and want.tolist() == [0.0]
    for lowstack in (True, False):
        assert run(cls, args, lowstack=lowstack).tolist() == [0.0]


GOLDEN = Path(__file__).resolve().parent / "golden"
LOWSTACK_GOLDEN_CLASS = (3, 3)


def test_the_lowstack_cuda_golden_is_unchanged() -> None:
    # Regenerate (never hand-edit), the default form at (ff):
    #   PYTHONPATH=src uv run python -c "from felizjueves.eri.k_exchange.\
    #   kernel_record_bound import build_record_bound_kernel as b; from \
    #   felizjueves.performance.device import SM_86; from \
    #   felizjueves.emit.cuda import emit; print(emit(b((3, 3)), SM_86), \
    #   end='')" > test/golden/cuda/record_bound_33.cu
    text = cuda.emit(build_record_bound_kernel(LOWSTACK_GOLDEN_CLASS), SM_86)
    assert text == (GOLDEN / "cuda" / "record_bound_33.cu").read_text()


def test_the_lowstack_hip_golden_is_unchanged() -> None:
    #   ... the same with GFX1200 and felizjueves.emit.hip, into
    #   test/golden/hip/record_bound_33.hip
    text = hip.emit(build_record_bound_kernel(LOWSTACK_GOLDEN_CLASS), GFX1200)
    assert text == (GOLDEN / "hip" / "record_bound_33.hip").read_text()
