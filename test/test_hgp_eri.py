"""The cross-engine pin (#281): HGP against `ao/eri.py`, 21 classes, CONTRACTED.

`(ab|cd)` for a contracted shell quartet two ways --

    MMD   `ao.eri.eri_block_and_scale` (evaluator `reference.pq.pq`): the
          Hermite path, witnessed against libcint by test/test_ao_eri.py
    HGP   `ao.hgp_drive.eri_block_and_scale` (Boys `reference.boys.boys`):
          `hgp.recurrence`'s VRR kernel interpreted per primitive quartet,
          the boundary summed over all `K^4`, then its HRR kernel once

-- and they must agree, element by element, at the CONTRACTED AO QUARTET
(not at an intermediate: MMD's Hermite tower and HGP's `[e0|f0]` boundary are
different quantities). This is the first pair inside `src/` with
independent ancestry: a disagreement localises to the engine that moved, and
it needs neither the fused kernel, the digestion nor a card. #279 pinned the
same comparison on ONE primitive quartet; here it is real basis shells, with
the contraction sum completing before the HRR runs (pin 5).

THE MOLECULES, the tracked fixtures' converted bases (libcint's conventions,
#84), so every shell is a real basis shell with its real contraction:

    hf_ccpvtz     HF / cc-pVTZ: d on both atoms, F s contracted 8 deep
    h2o_ccpvdz_j  H2O / cc-pVDZ: three centres, d on O only
    prim_spdf     He, Ne: one uncontracted primitive per l on each, the
                  second atom off every axis

THE QUARTET per (molecule, class), a rule and not a choice: among every
ordered shell quartet whose `l`s are the class, the one with the most
distinct centres, then the most primitive quartets, then the first in
shell-index order. Every class reaches TWO OR MORE CENTRES on hf_ccpvtz
and prim_spdf, and on h2o_ccpvdz_j in all but `(dd|dd)`, whose d shells all
sit on O -- so all 21 classes are pinned on at least two molecules with
distinct centres (asserted below). Every class but `(dd|dd)` is contracted
in at least one of them; `(dd|dd)` is a single primitive quartet in every
basis here (Dunning d shells are uncontracted), also asserted, so the claim
is not wider than the data.

THE TOLERANCE, stated and justified, not tuned:

    |hgp - mmd| <= 1e-12 * (scale_mmd + scale_hgp),   element by element,

each `scale` its own engine's absolute-value twin -- `SUM |term|` through
every product and sum it formed (`ao.eri`'s, and `ao.hgp_drive`'s
`absolute_kernel` run on |scalars|). An fp64 evaluation of a sum of
products with at most `n` sequential roundings on every path errs by at most
`gamma_n = n u / (1 - n u)` of that twin (Higham, Thm 3.1, recursive
summation and products), so by the triangle inequality the two differ by at
most `gamma_n (scale_mmd + scale_hgp)`, plus the error of `F_m(T)` itself,
which both engines take from the same injected `reference.boys.boys`.
`1e-12` is `gamma_n` for `n` about 9000 (`u = 2^-53`), and
`test_the_rounding_budget_covers_every_quartet` asserts, per pinned quartet,
that the HGP path's count -- the `K^4`-term contraction sum (worst here
1,536) plus the VRR and HRR depth, `3 x 5` roundings per level (a level is
one `Weighted` of at most five `c * g * src` terms) -- stays under it. The
HGP scale is NEEDED, not decoration: the HRR's `AB`/`CD` transfers cancel,
which MMD's scale cannot see (on H2O/cc-pVDZ `(pd|pd)` the difference is
1.04e-13 of the MMD scale alone). The same `1e-12` is test/test_ao_eri.py's
libcint bound, and it is used for the libcint comparison here, against the
same summed scale (libcint's own rounding approximated by MMD's twin, as
there). Measured worst on this box when written (2026-10-14, printed with
`-s`): HGP against MMD 5.4e-16 of the summed scale (HF/cc-pVTZ `(ss|sd)`),
HGP against libcint 2.9e-14 (H2O/6-31G `(pp|ss)`, shells (3, 4, 5, 7)).

THE OUTSIDE WITNESS. The HGP block is also held to libcint directly, on every
quartet of `h2o_631g_eri` and `hf_ccpvtz_eri` with every shell `l <= 2` --
all 21 classes on HF/cc-pVTZ and the six s/p ones on H2O/6-31G (asserted).

CAN IT FAIL. One HGP coefficient perturbed (`inv2pq`, by 1 + 1e-6) misses by
far more than the tolerance; and the claims' scorer half (#291 scores them,
so nothing here carries a `claim` mark): with eq 26's printed `WQ` sign the
contracted ket VRR misses, with the derived one it passes
(`fock/eq26-ket-vrr-sign`, on contracted quartets with `P != Q`). Eq 14's
seed (`fock/eq14-seed-inconsistent`) needs a seed the kernel's prelude does
not spell, so its contracted twin would be a second harness; #279's
primitive-quartet test stands for it.

Fast tier: no compiler, no card. The whole file runs in about ten seconds
(the slowest quartet, HF/cc-pVTZ `(dd|dd)`, a fraction of a second), so
nothing is `slow`.
"""

from __future__ import annotations

import itertools
import math
from functools import cache, partial

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao import hgp_drive
from felizjueves.ao.eri import eri_block_and_scale as mmd_eri
from felizjueves.ao.fock import coulomb_exchange_reference
from felizjueves.ao.hgp_drive import absolute_kernel, f_spec
from felizjueves.ao.hgp_drive import eri_block as hgp_block
from felizjueves.ao.hgp_drive import eri_block_and_scale as hgp_eri
from felizjueves.ao.shell import cartesians
from felizjueves.hgp import hrr, vrr
from felizjueves.hgp import recurrence as R
from felizjueves.ir import Decl, For, Lit, Weighted
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq

TOL = 1e-12
U = 2.0 ** -53
ROUNDINGS_PER_LEVEL = 3 * 5  # one Weighted: <= 5 terms, c*g*src then +
CLASSES = vrr.CLASSES
MOLECULES = ("hf_ccpvtz", "h2o_ccpvdz_j", "prim_spdf")
LIBCINT = ("h2o_631g_eri", "hf_ccpvtz_eri")
CLS_NAME = "spd"


def _name(cls) -> str:
    return "({}{}|{}{})".format(*(CLS_NAME[x] for x in cls))


@cache
def _fixture(name: str):
    return manifest.load(manifest.FIXTURES / f"{name}.json")


@cache
def _shells(name: str):
    return ob.from_fixture(_fixture(name)).shells


def _n_prim(shells) -> int:
    return math.prod(len(s.exponents) for s in shells)


@cache
def pick(name: str, cls) -> tuple[int, int, int, int]:
    """The module docstring's rule: most distinct centres, then most
    primitive quartets, then first in shell-index order."""
    sh = _shells(name)
    best = None
    for q in itertools.product(range(len(sh)), repeat=4):
        if tuple(sh[s].l for s in q) != cls:
            continue
        key = (len({sh[s].centre for s in q}), _n_prim([sh[s] for s in q]))
        if best is None or key > best[0]:
            best = (key, q)
    assert best is not None, (name, cls)
    return best[1]


def _quartet(name, q):
    sh = _shells(name)
    return (sh[q[0]], sh[q[1]]), (sh[q[2]], sh[q[3]])


@cache
def both(name: str, q, first: str = "bra"):
    """(hgp, mmd, scale_hgp + scale_mmd), flat in `hgp.hrr.targets` order."""
    bra, ket = _quartet(name, q)
    h, hs = hgp_eri(bra, ket, boys=boys, first=first)
    m, ms = mmd_eri(bra, ket, pq=pq)
    return (np.array(h).ravel(), np.array(m).ravel(),
            np.array(hs).ravel() + np.array(ms).ravel())


def worst(got, want, scale) -> float:
    err = np.abs(got - want)
    # An element both engines form from all-zero terms is zero in both.
    assert np.all(err[scale == 0.0] == 0.0)
    return float(np.max(np.divide(err, scale, out=np.zeros_like(err),
                                  where=scale > 0)))


# ---------------------------------------------------------------------------
# The layout, and the coverage the module docstring claims
# ---------------------------------------------------------------------------


def test_the_hgp_targets_are_ao_eris_layout() -> None:
    """`hgp.hrr.targets` is `a, b, c, d` nested in `ao.shell.cartesians`
    order: the flat order `eri_block`'s rows/columns and `ao.eri`'s agree."""
    for cls in CLASSES:
        want = tuple(itertools.product(*(cartesians(x) for x in cls)))
        assert hrr.targets(cls) == want
        for x in cls:
            assert vrr.cartesians(x) == cartesians(x)


def test_every_class_is_pinned_on_two_molecules_with_distinct_centres() -> None:
    for cls in CLASSES:
        multi = [m for m in MOLECULES
                 if len({s.centre for s in sum(_quartet(m, pick(m, cls)), ())})
                 >= 2]
        assert len(multi) >= 2, (cls, multi)
        if cls != (2, 2, 2, 2):
            assert len(multi) == 3, (cls, multi)
    # (dd|dd) on H2O/cc-pVDZ: every d shell is on O, one centre.
    assert {s.centre for s in sum(_quartet("h2o_ccpvdz_j",
                                           pick("h2o_ccpvdz_j", (2, 2, 2, 2))),
                                  ())}.__len__() == 1


def test_every_class_but_dddd_is_contracted_somewhere() -> None:
    for cls in CLASSES:
        n = max(_n_prim(sum(_quartet(m, pick(m, cls)), ())) for m in MOLECULES)
        assert (n > 1) is (cls != (2, 2, 2, 2)), (cls, n)
    # (dd|dd) has no contracted d shell to reach in any of these bases.
    for m in MOLECULES:
        assert all(len(s.exponents) == 1 for s in _shells(m) if s.l == 2)


def test_the_rounding_budget_covers_every_quartet() -> None:
    """`TOL` is `gamma_n` with `n` above every pinned quartet's rounding
    count (module docstring, "THE TOLERANCE")."""
    n_max = TOL / U  # gamma_n ~ n u for n u << 1
    assert 9000 < n_max < 9100
    for m in MOLECULES:
        for cls in CLASSES:
            shells = sum(_quartet(m, pick(m, cls)), ())
            depth = sum(cls) + cls[1] + cls[3]  # VRR |e|+|f|, then HRR lb+ld
            n = _n_prim(shells) + ROUNDINGS_PER_LEVEL * depth
            assert n < n_max, (m, cls, n)


# ---------------------------------------------------------------------------
# The pin: all 21 classes, contracted, on each molecule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=_name)
@pytest.mark.parametrize("mol", MOLECULES)
def test_hgp_matches_mmd_on_contracted_quartets(mol, cls) -> None:
    q = pick(mol, cls)
    got, want, scale = both(mol, q)
    w = worst(got, want, scale)
    print(f"\n{mol} {_name(cls)} {q}: max err/scale = {w:.2e}")
    assert w <= TOL, (mol, cls, q, w)


@pytest.mark.parametrize("cls", CLASSES, ids=_name)
def test_the_hrr_side_order_does_not_change_the_integrals(cls) -> None:
    """`first="ket"` (`hgp.hrr.build`'s other order) on HF/cc-pVTZ."""
    q = pick("hf_ccpvtz", cls)
    ket_first, want, scale = both("hf_ccpvtz", q, "ket")
    assert worst(ket_first, want, scale) <= TOL


def test_the_scale_bounds_the_block() -> None:
    """The |.|-twin is >= |value| everywhere, and strictly larger somewhere
    on a class with a ket VRR and an HRR (it sees cancellation)."""
    for cls in ((1, 1, 1, 1), (1, 2, 1, 2), (2, 2, 2, 2)):
        bra, ket = _quartet("hf_ccpvtz", pick("hf_ccpvtz", cls))
        h, hs = (np.array(x) for x in hgp_eri(bra, ket, boys=boys))
        assert np.all(hs >= np.abs(h))
        assert np.any(hs > 1.5 * np.abs(h))


def test_absolute_kernel_flips_only_the_signed_coefficients() -> None:
    k = R.build_vrr_kernel(f_spec((1, 1, 1, 1)))
    a = absolute_kernel(k)
    assert len(a.body) == len(k.body)
    signed = [t for s in k.body if isinstance(s, Decl)
              and isinstance(s.init, Weighted) for t in s.init.terms
              if t.c < 0]
    assert signed  # eq 25's -e_d rides in c (#277)
    assert all(t.c > 0 for s in a.body if isinstance(s, Decl)
               and isinstance(s.init, Weighted) for t in s.init.terms)


def test_absolute_kernel_refuses_a_statement_it_does_not_know() -> None:
    from dataclasses import replace
    k = R.build_vrr_kernel(f_spec((0, 0, 0, 0)))
    loop = For("i", Lit(0), Lit(1), Lit(1), (k.body[-1],))
    bad = replace(k, body=(*k.body, loop))
    with pytest.raises(ValueError, match="not a recurrence statement"):
        absolute_kernel(bad)


# ---------------------------------------------------------------------------
# The outside witness: libcint's int2e
# ---------------------------------------------------------------------------


def _libcint_params():
    out = []
    for name in LIBCINT:
        sh = _shells(name)
        for q in sorted(manifest.eri_blocks(_fixture(name))):
            ls = tuple(sh[s].l for s in q)
            if max(ls) <= 2:
                out.append(pytest.param(name, q, id=f"{name}-{_name(ls)}-{q}"))
    return out


LIBCINT_PARAMS = _libcint_params()


def _canonical(ls) -> frozenset:
    """A class up to the 8-fold permutations: the fixture's representative
    is in its own shell order, the ladder's is `la <= lb`, bra <= ket."""
    return frozenset((tuple(sorted(ls[:2])), tuple(sorted(ls[2:]))))


def test_the_libcint_comparison_reaches_every_class() -> None:
    seen: dict[str, set] = {name: set() for name in LIBCINT}
    for p in LIBCINT_PARAMS:
        name, q = p.values
        seen[name].add(_canonical([_shells(name)[s].l for s in q]))
    assert seen["hf_ccpvtz_eri"] == {_canonical(c) for c in CLASSES}
    assert seen["h2o_631g_eri"] == {_canonical(c) for c in CLASSES
                                    if max(c) <= 1}
    assert len(seen["h2o_631g_eri"]) == 6


@pytest.mark.parametrize("name,q", LIBCINT_PARAMS)
def test_hgp_matches_libcint(name, q) -> None:
    ref = manifest.eri_blocks(_fixture(name))[q].ravel()
    got, _, scale = both(name, q)
    w = worst(got, ref, scale)
    print(f"\n{name} {q}: max err/scale vs libcint = {w:.2e}")
    assert w <= TOL, (name, q, w)


# ---------------------------------------------------------------------------
# Can it fail
# ---------------------------------------------------------------------------


def _perturbed(monkeypatch, key: str, fn) -> None:
    real = hgp_drive.primitive_scalars

    def wrapped(*args):
        s = real(*args)
        for ax in ("x", "y", "z") if key.endswith("_") else ("",):
            s[key + ax] = fn(s[key + ax])
        return s

    monkeypatch.setattr(hgp_drive, "primitive_scalars", wrapped)


def _plain(name: str, cls):
    """(hgp value, mmd, summed scale), the hgp value recomputed now."""
    q = pick(name, cls)
    bra, ket = _quartet(name, q)
    _, want, scale = both(name, q)
    return np.array(hgp_block(bra, ket, boys=boys)).ravel(), want, scale


def test_perturbing_one_hgp_coefficient_breaks_the_pin(monkeypatch) -> None:
    for cls in ((1, 1, 1, 1), (0, 2, 1, 2)):
        got, want, scale = _plain("hf_ccpvtz", cls)
        assert worst(got, want, scale) <= TOL
    _perturbed(monkeypatch, "inv2pq", lambda v: v * (1.0 + 1e-6))
    for cls in ((1, 1, 1, 1), (0, 2, 1, 2)):
        got, want, scale = _plain("hf_ccpvtz", cls)
        assert worst(got, want, scale) > 1e3 * TOL, cls


@pytest.mark.parametrize("cls", [(0, 0, 0, 1), (1, 1, 1, 1), (0, 2, 1, 2),
                                 (2, 2, 2, 2)], ids=_name)
def test_eq26s_printed_ket_sign_fails_contracted(monkeypatch, cls) -> None:
    """`fock/eq26-ket-vrr-sign`, on real contracted shells with `P != Q`:
    the derived `W - Q = +p/(p+q) PQ` passes, eq 26's printed `-p/(p+q) PQ`
    misses by orders of magnitude more than rounding."""
    got, want, scale = _plain("hf_ccpvtz", cls)
    assert worst(got, want, scale) <= TOL
    bra, ket = _quartet("hf_ccpvtz", pick("hf_ccpvtz", cls))
    s = hgp_drive.primitive_scalars(*bra, *ket, 0, 0, 0, 0)
    assert max(abs(s[f"PQ_{a}"]) for a in "xyz") > 1e-2  # P != Q
    _perturbed(monkeypatch, "WQ_", lambda v: -v)
    got, want, scale = _plain("hf_ccpvtz", cls)
    assert worst(got, want, scale) > 1e-2


# ---------------------------------------------------------------------------
# The signature: injectable into ao.fock's sweep (#286 makes the swap)
# ---------------------------------------------------------------------------


def test_the_evaluator_drops_into_ao_focks_sweep() -> None:
    """`partial(eri_block, boys=boys)` fills `ao.fock`'s `eri=` slot, the
    one `partial(ao.eri.eri_block, pq=pq)` fills: H2/STO-3G, `J` and `K`
    through both engines, within `TOL` of the sweep run on `|D|` with the
    summed scale (the triangle inequality carries the per-ERI bound)."""
    fx = _fixture("h2_sto3g_j")
    basis = ob.from_fixture(fx)
    D = np.array(fx.density)
    hj, hk = (np.array(x) for x in coulomb_exchange_reference(
        basis, D, eri=partial(hgp_block, boys=boys)))
    mj, mk = (np.array(x) for x in coulomb_exchange_reference(
        basis, D, eri=lambda b, k: mmd_eri(b, k, pq=pq)[0]))

    def scale(b, k):
        hs = np.array(hgp_eri(b, k, boys=boys)[1])
        ms = np.array(mmd_eri(b, k, pq=pq)[1])
        return tuple(map(tuple, hs + ms))

    sj, sk = (np.array(x) for x in coulomb_exchange_reference(
        basis, np.abs(D), eri=scale))
    assert np.all(np.abs(hj - mj) <= TOL * sj)
    assert np.all(np.abs(hk - mk) <= TOL * sk)
    assert np.max(np.abs(hj)) > 0.1
