"""kernel_record_bound.py: K's dressed record bound, interpreted (#499).

THE WITNESS is `ao.k.record_bounds(epair(sa, sb), pq=reference.pq.pq)`: the
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
cc-pVDZ (both orientations, since every ordered pair is a record set).
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import split_by_l

from felizjueves.ao.density import epair
from felizjueves.ao.pairs import hermite_bound, pair_bound
from felizjueves.ao.shell import Shell
from felizjueves.eri.k_exchange import kernel_record_bound as krb
from felizjueves.eri.k_exchange.k import record_bounds
from felizjueves.eri.k_exchange.kernel_record_bound import (
    PARAMS,
    build_record_bound_kernel,
    record_bound_name,
    record_bound_spec,
)
from felizjueves.eri.kernel_pair_bound import (
    hermite_bound_coefficients,
    pair_bound_coefficients,
)
from felizjueves.ir.interp import Launch, interpret
from felizjueves.plan.targets import hermite_targets
from felizjueves.reference.pq import one_centre, pq
from felizjueves.spec import Operator, Spec, Target

EPS = float(np.finfo(np.float64).eps)
REL_TOL = 16 * EPS

LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
CLASS_GRID = split_by_l(CLASSES, sum, ids=lambda c: f"{c[0]}{c[1]}")

MOLECULES = ("h2o_ccpvdz_k", "h10_chain_ccpvdz_k")


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
        threads: int = 32) -> np.ndarray:
    n = len(args["zeta_ab"])
    grid, block = krb.launch_shape(n, threads)
    out = interpret(build_record_bound_kernel(cls),
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
    # What K's screen is proved against (ao.k, "THE SCREEN") and what
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
