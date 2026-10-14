"""`hgp/recurrence.py` (#279): the HGP VRR and HRR as IR, interpreted.

Three kinds of witness, from the inside out:

1. A DIRECT recursion of `[Fock]` eqs 25-26 and eq 28, written here from the
   equations (memoised over the index tuple; no DAG, no build order, no IR),
   reducing along the same axis and summing the terms in the same order as
   the DAG does -- so `interpret(build_vrr_kernel(spec))` and
   `interpret(build_hrr_kernel(spec))` must agree with it BITWISE, all 21
   classes. This checks the IR spelling, not the recurrence.
2. The fundamental's relation to FJ's one fundamental, `reference/pq.py`'s
   `[0]^(m)`, asserted pointwise (and `[Fock]`'s `R0_m = pref F_m(T)` with it).
3. The INTEGRALS: VRR then HRR, interpreted, against `ao/eri.py`'s MMD path
   (evaluator `reference.pq.pq`, itself witnessed against libcint) on one
   primitive quartet with four distinct centres, all 21 classes. This is what
   checks the recurrence, and it is where the two `[Fock]` defects show:
   eq 26's printed `(m+1)` sign and eq 14's printed seed each fail it, while
   the derived ones pass (`fock/eq26-ket-vrr-sign` and
   `fock/eq14-seed-inconsistent`, pending; #291 scores them, so these tests
   carry no `claim` mark here). The contracted, cross-engine pin with
   libcint fixtures is #281's.

Fast tier: no compiler, no card. Every class here interprets in well under a
second, so none is `slow`.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest

from felizjueves.ao.eri import eri_block_and_scale
from felizjueves.ao.shell import Shell
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp import recurrence as R
from felizjueves.interp import interpret
from felizjueves.ir import (
    Assign,
    Decl,
    Factor,
    Index,
    Intent,
    Kernel,
    Lit,
    Param,
    Scalar,
    Var,
)
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import fundamental as fj_fundamental
from felizjueves.reference.pq import pq as fj_pq
from felizjueves.reference.pq import theta_of
from felizjueves.spec import Operator, Spec, Target

CLASSES = vrr.CLASSES
TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


def fspec(cls) -> Spec:
    return Spec(Operator.COULOMB, 0, tuple(cls), Target.F)


# ---------------------------------------------------------------------------
# One primitive quartet: the scalars the module docstring defines
# ---------------------------------------------------------------------------

EXPS = (1.3, 0.7, 0.9, 1.6)
CENTRES = ((0.0, 0.0, 0.0), (0.4, -0.3, 0.5), (-0.2, 0.6, 0.1),
           (0.7, 0.2, -0.4))
SPREAD = ((0.3, 2.1, 5.0, 0.15), ((0.0, 0.0, 0.0), (1.1, 0.0, -0.3),
                                  (0.5, -1.9, 0.8), (-0.6, 0.4, 2.0)))
GEOMS = {"compact": (EXPS, CENTRES), "spread": SPREAD}


def _sub(u, v):
    return tuple(a - b for a, b in zip(u, v, strict=True))


def quartet(exps=EXPS, centres=CENTRES, *, ket_sign: float = 1.0) -> dict:
    """Every scalar of both standalone kernels for one primitive quartet.

    `ket_sign = -1` binds `WQ` with `[Fock]` eq 26's PRINTED sign,
    `-p/(p+q) PQ`, instead of the derived `W - Q = +p/(p+q) PQ`."""
    a, b, c, d = exps
    A, B, C, D = centres
    p, q = a + b, c + d
    P = tuple((a * x + b * y) / p for x, y in zip(A, B, strict=True))
    Q = tuple((c * x + d * y) / q for x, y in zip(C, D, strict=True))
    PQ = _sub(P, Q)
    W = tuple((p * x + q * y) / (p + q) for x, y in zip(P, Q, strict=True))
    rho = p * q / (p + q)
    AB, CD = _sub(A, B), _sub(C, D)
    kab = math.exp(-a * b / p * sum(x * x for x in AB))
    kcd = math.exp(-c * d / q * sum(x * x for x in CD))
    s = {}
    for k, ax in enumerate("xyz"):
        s[f"PA_{ax}"] = P[k] - A[k]
        s[f"WP_{ax}"] = W[k] - P[k]
        s[f"QC_{ax}"] = Q[k] - C[k]
        s[f"WQ_{ax}"] = ket_sign * p / (p + q) * PQ[k]
        s[f"PQ_{ax}"] = PQ[k]
        s[f"AB_{ax}"] = AB[k]
        s[f"CD_{ax}"] = CD[k]
    s["inv2p"] = 1.0 / (2.0 * p)
    s["inv2q"] = 1.0 / (2.0 * q)
    s["inv2pq"] = 1.0 / (2.0 * (p + q))
    s["inv2p_alpha_over_p"] = 1.0 / (2.0 * p) * (rho / p)
    s["inv2q_alpha_over_q"] = 1.0 / (2.0 * q) * (rho / q)
    s["theta"] = theta_of(p, q)
    s["pair"] = (math.pi / p) ** 1.5 * kab * (math.pi / q) ** 1.5 * kcd
    s["_meta"] = dict(p=p, q=q, kab=kab, kcd=kcd, W=W, P=P, Q=Q)
    return s


def vrr_inputs(s: dict) -> dict:
    return {n: s[n] for n in R.VRR_SCALARS}


def hrr_inputs(s: dict) -> dict:
    return {n: s[n] for n in R.HRR_SCALARS}


def run_vrr(cls, s) -> np.ndarray:
    k = R.build_vrr_kernel(fspec(cls))
    n = len(R.boundary_order(cls))
    return interpret(k, boys=boys, **vrr_inputs(s), out=np.zeros(n))["out"]


def run_hrr(cls, s, bnd, first="bra") -> np.ndarray:
    k = R.build_hrr_kernel(fspec(cls), first=first)
    return interpret(k, **hrr_inputs(s), boundary=np.asarray(bnd),
                     out=np.zeros(len(hrr.targets(cls))))["out"]


# ---------------------------------------------------------------------------
# 1. The direct recursions, written from the equations
# ---------------------------------------------------------------------------


def _low(v, d, k=1):
    return tuple(x - k * (i == d) for i, x in enumerate(v))


def seed(s: dict, m: int) -> float:
    """`[00|00]^(m) = pair ((2/sqrt(pi)) theta) F_m(T)`, in the kernel's
    operation order (`R.fundamental_prelude`)."""
    th = s["theta"]
    T = (th * th) * ((s["PQ_x"] * s["PQ_x"] + s["PQ_y"] * s["PQ_y"])
                     + s["PQ_z"] * s["PQ_z"])
    return (s["pair"] * (TWO_OVER_SQRT_PI * th)) * boys(m, T)


def direct_vrr(s: dict, seed_of=seed):
    """`[e0|f0]^(m)` by eqs 25 (f = 0) and 26 (f != 0), reduced along the
    first nonzero axis of `f`, else of `e` -- `hgp.vrr`'s documented choice."""
    ax = "xyz"

    @cache
    def v(e, f, m):
        if not any(e) and not any(f):
            return seed_of(s, m)
        terms = []  # (c, g, value), in eq order
        if any(f):
            d = next(i for i, x in enumerate(f) if x)
            g = _low(f, d)
            terms += [(1, s[f"QC_{ax[d]}"], v(e, g, m)),
                      (1, s[f"WQ_{ax[d]}"], v(e, g, m + 1))]
            if g[d]:
                h = _low(g, d)
                terms += [(g[d], s["inv2q"], v(e, h, m)),
                          (-g[d], s["inv2q_alpha_over_q"], v(e, h, m + 1))]
            if e[d]:
                terms.append((e[d], s["inv2pq"], v(_low(e, d), g, m + 1)))
        else:
            d = next(i for i, x in enumerate(e) if x)
            g = _low(e, d)
            terms += [(1, s[f"PA_{ax[d]}"], v(g, f, m)),
                      (1, s[f"WP_{ax[d]}"], v(g, f, m + 1))]
            if g[d]:
                h = _low(g, d)
                terms += [(g[d], s["inv2p"], v(h, f, m)),
                          (-g[d], s["inv2p_alpha_over_p"], v(h, f, m + 1))]
        acc = 0.0
        for c, gk, x in terms:
            acc += float(c) * float(gk) * float(x)
        return acc

    return v


def direct_hrr(s: dict, bnd: dict, first="bra"):
    """`(ab|cd)` by eq 28: `(a, b+1_k|..) = (a+1_k, b|..) + AB_k (a, b|..)`
    and its ket twin, the side `first` NOT names reduced first, axis the first
    nonzero component; the boundary `(e0|f0)` read from `bnd`."""
    ax = "xyz"

    def up(v, k):
        return tuple(x + (i == k) for i, x in enumerate(v))

    @cache
    def h(a, b, c, d):
        order = ("ket", "bra") if first == "bra" else ("bra", "ket")
        for side in order:
            if side == "ket" and any(d):
                k = next(i for i, x in enumerate(d) if x)
                dm = _low(d, k)
                terms = [(1.0, h(a, b, up(c, k), dm)),
                         (s[f"CD_{ax[k]}"], h(a, b, c, dm))]
                break
            if side == "bra" and any(b):
                k = next(i for i, x in enumerate(b) if x)
                bm = _low(b, k)
                terms = [(1.0, h(up(a, k), bm, c, d)),
                         (s[f"AB_{ax[k]}"], h(a, bm, c, d))]
                break
        else:
            return bnd[(a, c)]
        acc = 0.0
        for g, x in terms:
            acc += 1.0 * float(g) * float(x)
        return acc

    return h


@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_vrr_kernel_is_eqs_25_26_bitwise(cls) -> None:
    s = quartet()
    got = run_vrr(cls, s)
    v = direct_vrr(s)
    want = np.array([v(e, f, 0) for e, f in R.boundary_order(cls)])
    assert np.array_equal(got, want), np.max(np.abs(got - want))


@pytest.mark.parametrize("first", hrr.SIDES)
@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_hrr_kernel_is_eq_28_bitwise_fed_by_the_vrr(cls, first) -> None:
    s = quartet()
    bnd = run_vrr(cls, s)
    got = run_hrr(cls, s, bnd, first)
    h = direct_hrr(s, dict(zip(R.boundary_order(cls), bnd, strict=True)),
                   first)
    want = np.array([h(*n) for n in hrr.targets(cls)])
    assert np.array_equal(got, want), np.max(np.abs(got - want))


# ---------------------------------------------------------------------------
# 2. One fundamental: the relation to reference/pq.py's [0]^(m)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", range(9))
@pytest.mark.parametrize("geom", [
    (EXPS, CENTRES),
    SPREAD,
    ((4.0, 4.0, 0.5, 0.5), ((0.0, 0.0, 0.0),) * 4),  # P = Q: T = 0
], ids=("compact", "spread", "one-centre"))
def test_the_seed_is_fjs_fundamental_with_the_factors_placed(m, geom) -> None:
    s = quartet(*geom)
    meta = s["_meta"]
    p, q = meta["p"], meta["q"]
    th = s["theta"]
    pqv = (s["PQ_x"], s["PQ_y"], s["PQ_z"])
    T = th * th * sum(x * x for x in pqv)
    ours = seed(s, m)
    # [00|00]^(m) = U_P U_Q (-2 theta^2)^(-m) [0]^(m)  (FJ's placement)
    via_fj = s["pair"] * fj_fundamental(m, pqv, th) / (-2.0 * th * th) ** m
    # [Fock] sec. 3.1: R0_m = sqrt(4 pi^5) Kab Kcd / sqrt(p^2 q^2 (p+q)) F_m
    pref = (math.sqrt(4.0 * math.pi ** 5) * meta["kab"] * meta["kcd"]
            / math.sqrt(p * p * q * q * (p + q)))
    via_fock = pref * boys(m, T)
    # A handful of roundings of each side (the powers of (-2 theta^2) and the
    # pair factors), relative: these are three spellings of one number.
    assert ours == pytest.approx(via_fj, rel=1e-14, abs=0.0)
    assert ours == pytest.approx(via_fock, rel=1e-14, abs=0.0)


def test_the_ssss_kernel_output_is_the_seed_bitwise() -> None:
    s = quartet()
    assert run_vrr((0, 0, 0, 0), s)[0] == seed(s, 0)


# ---------------------------------------------------------------------------
# 3. The integrals: HGP against ao/eri.py's MMD path, one primitive quartet
# ---------------------------------------------------------------------------


def mmd_block(cls, exps=EXPS, centres=CENTRES):
    """`(ab|cd)` and its scale by `ao/eri.py` on unnormalised single-primitive
    shells (coefficient 1 multiplies the bare Cartesian Gaussian,
    `ao/shell.py`), flattened in `hgp.hrr.targets` order (a outermost)."""
    shells = [Shell(am, ctr, (x,), (1.0,))
              for am, ctr, x in zip(cls, centres, exps, strict=True)]
    blk, scale = eri_block_and_scale((shells[0], shells[1]),
                                     (shells[2], shells[3]), pq=fj_pq)
    return np.array(blk).ravel(), np.array(scale).ravel()


def hgp_block(cls, s) -> np.ndarray:
    return run_hrr(cls, s, run_vrr(cls, s))


# The comparison is between two exact-arithmetic-equal recurrences with
# different operation orders; HRR's `AB`/`CD` transfers cancel, which MMD's
# |.|-scale does not see, so the bound is stated against that scale: 1e-13
# (about 450 eps) is 25x the worst error measured on the ladder over both
# geometries when this was written (4.1e-15 of scale, (dd|dd), "spread"),
# and 10^11 below the eq 26 / eq 14 defects it must separate (> 1e-2).
TOL = 1e-13


@pytest.mark.parametrize("geom", GEOMS)
@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_hgp_matches_mmd_on_a_primitive_quartet(cls, geom) -> None:
    got = hgp_block(cls, quartet(*GEOMS[geom]))
    want, scale = mmd_block(cls, *GEOMS[geom])
    assert got.shape == want.shape
    err = np.abs(got - want)
    assert np.all(err <= TOL * scale), float(np.max(err / scale))


@pytest.mark.parametrize("cls", [(0, 0, 0, 1), (0, 1, 0, 1), (1, 1, 1, 1),
                                 (2, 2, 2, 2)], ids=str)
def test_eq26s_printed_ket_sign_fails_and_the_derived_one_passes(cls) -> None:
    """`fock/eq26-ket-vrr-sign`: with `P != Q`, binding `WQ = -p/(p+q) PQ`
    (eq 26 as printed) gets the integrals wrong; `W - Q = +p/(p+q) PQ`
    (derived) gets them right. The IR is the same kernel both times: the
    sign is a property of the bound value."""
    want, scale = mmd_block(cls)
    derived, printed = quartet(), quartet(ket_sign=-1.0)
    assert any(abs(derived[f"PQ_{a}"]) > 0.1 for a in "xyz")  # P != Q
    ok = np.abs(hgp_block(cls, derived) - want)
    bad = np.abs(hgp_block(cls, printed) - want)
    assert np.all(ok <= TOL * scale)
    assert np.max(bad / scale) > 1e-2  # wrong, not rounding


def _seed_harness(cls) -> Kernel:
    """The VRR of `cls` with its seeds read from an input array `seed[m]`,
    so a printed seed can be fed in place of the derived one."""
    dag = vrr.build(cls)
    body = list(R.vrr_body(dag, lambda m: Index("seed", (Lit(m),))))
    for i, (e, f) in enumerate(R.boundary_order(cls)):
        body.append(Assign(Index("out", (Lit(i),)), Var(R.vrr_name((e, f, 0)))))
    scalars = [n for n in R.VRR_SCALARS if n not in (*R.PQ_PARAMS, "theta",
                                                     "pair")]
    params = (*(Param(n, Scalar.F64) for n in scalars),
              Param("seed", Scalar.F64, array=True),
              Param("out", Scalar.F64, array=True, intent=Intent.OUT))
    return Kernel("seed_harness", params, tuple(body))


def _with_seed(cls, s, seeds) -> np.ndarray:
    k = _seed_harness(cls)
    ins = {p.name: s[p.name] for p in k.params if p.name not in ("seed", "out")}
    bnd = interpret(k, **ins, seed=np.array(seeds),
                    out=np.zeros(len(R.boundary_order(cls))))["out"]
    return run_hrr(cls, s, bnd)


def test_eq14s_printed_seed_fails_and_the_derived_one_passes() -> None:
    """`fock/eq14-seed-inconsistent`: eq 14's `U_P U_Q theta^m F_m(T)` misses
    `(ss|ss)` (no `(2/sqrt(pi)) theta`); with that factor restored it matches
    `(ss|ss)` but still misses `(sp|ss)`, one bra VRR step that reads
    `m = 1` (its `theta^m`). The derived seed matches both. The "spread"
    quartet has `theta` and `(2/sqrt(pi)) theta` both far from 1, so neither
    defect can hide behind a unit factor."""
    s = quartet(*SPREAD)
    th, pair = s["theta"], s["pair"]
    T = th * th * sum(s[f"PQ_{a}"] ** 2 for a in "xyz")
    assert abs(th - 1.0) > 0.1 and abs(TWO_OVER_SQRT_PI * th - 1.0) > 0.1
    derived = [seed(s, m) for m in range(3)]
    eq14 = [pair * th ** m * boys(m, T) for m in range(3)]
    repaired = [TWO_OVER_SQRT_PI * th * x for x in eq14]
    for cls, eq14_ok, repaired_ok in (((0, 0, 0, 0), False, True),
                                      ((0, 1, 0, 0), False, False)):
        want, scale = mmd_block(cls, *SPREAD)

        def close(seeds, cls=cls, want=want, scale=scale):
            return bool(np.all(np.abs(_with_seed(cls, s, seeds) - want)
                               <= TOL * scale))

        assert close(derived)
        assert close(eq14) is eq14_ok
        assert close(repaired) is repaired_ok


# ---------------------------------------------------------------------------
# Structure, refusals, and "can this comparison fail"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_one_decl_per_dag_node_in_build_order(cls) -> None:
    v = vrr.build(cls)
    h = hrr.build(cls)
    vd = [s for s in R.build_vrr_kernel(fspec(cls)).body
          if isinstance(s, Decl) and s.name.startswith("V")]
    hd = [s for s in R.build_hrr_kernel(fspec(cls)).body
          if isinstance(s, Decl)]
    assert [d.name for d in vd] == [R.vrr_name(n) for n in v.order]
    assert [d.name for d in hd] == [R.hrr_name(n) for n in h.order]
    assert len(hd) == h.n_statements
    assert len({d.name for d in vd}) == v.n_nodes


def test_the_two_kernels_bind_every_hgp_factor() -> None:
    hgp = {f for f in Factor if f.value in R.VRR_SCALARS + R.HRR_SCALARS}
    names = {p.name for p in R.build_vrr_kernel(fspec((1, 1, 1, 1))).params}
    names |= {p.name for p in R.build_hrr_kernel(fspec((1, 1, 1, 1))).params}
    assert {f.value for f in hgp} <= names
    assert set(R.VRR_FACTOR) == set(vrr.BRA) | set(vrr.KET)
    assert set(R.HRR_FACTOR) == set(hrr.COEFS)


def test_a_non_f_spec_is_refused() -> None:
    with pytest.raises(ValueError, match="Target.F"):
        R.build_vrr_kernel(Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.J))
    with pytest.raises(ValueError, match="Target.F"):
        R.build_hrr_kernel(Spec(Operator.COULOMB, 0, (1, 1, 1, 1), Target.K))


def test_a_malformed_dag_is_refused() -> None:
    from dataclasses import replace
    from types import MappingProxyType

    dag = vrr.build((1, 1, 1, 1))
    node = next(n for n in dag.order if len(dag.terms[n]) == 5)
    ts = list(dag.terms[node])
    ts[1] = vrr.Term(ts[1].coef, 1, ts[0].child)  # WQ reading the same m
    bad = replace(dag, terms=MappingProxyType({**dag.terms, node: tuple(ts)}))
    with pytest.raises(ValueError, match="not eq 26"):
        R.vrr_reduction(bad, node)

    hdag = hrr.build((1, 1, 1, 1))
    hn = next(n for n in hdag.order if hdag.terms[n])
    t0, t1 = hdag.terms[hn]
    hbad = replace(hdag, terms=MappingProxyType(
        {**hdag.terms, hn: (t0, hrr.Term(t1.coef, t0.child))}))
    with pytest.raises(ValueError, match="not eq 28"):
        R.hrr_reduction(hbad, hn)


def test_perturbing_one_coefficient_breaks_the_integral_pin() -> None:
    """The ERI comparison can fail: nudge `inv2pq` and `(pp|pp)` misses."""
    cls = (1, 1, 1, 1)
    want, scale = mmd_block(cls)
    s = quartet()
    s["inv2pq"] *= 1.0 + 1e-6
    assert np.max(np.abs(hgp_block(cls, s) - want) / scale) > TOL
