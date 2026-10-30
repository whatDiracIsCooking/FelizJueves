"""`kernel_ghk`: the exchange-energy gradient on the HGP recurrence
(`Target.K` at d1 on `Engine.HGP`), interpreted.

THE WITNESS is `gradient.host.exchange_gradient`, the MMD host reference
(held to finite differences of `eri.k_exchange.k.exchange_matrix`,
test/test_gradient_host.py, and to libcint on CO/cc-pVTZ,
test/test_k_gradient_witness.py). The two share nothing of the arithmetic:
the host differentiates the MMD `E^{ac}` and runs `reference.pq`'s
`[r]^(0)`; the kernel runs one HGP VRR, accumulates the ket-inner sums per
bra primitive pair into three contracted sets (weighted `2 alpha`, `2
gamma`, 1) and runs the raised and lowered bra classes' HRRs on them.

- PER CLASS, every launch of `host.hgp_exchange_launches(am)` into one
  `G`, against `exchange_gradient(quartets=host.class_quartets)`: every K
  class with every shell s or p on test_gradient_host.py's two-atom s, p
  basis, and every class with a d shell through `(dd|dd)` on
  test_kernel_gk.py's two-atom s, p, d basis (classes above `FAST_L_MAX`
  `slow`, by `tiers.by_l` on the raised `L + 1`, as test_kernel_gk.py).
- THE WHOLE `dE_K` of the s, p basis: the 16 classes into one `G`, against
  the host's every-quartet sweep, translationally invariant.
- MIXED CONTRACTION DEGREES: a basis whose shells carry one, two and three
  primitives, so a class is cut into several `(k_ab, k_cd)` launches.
- THREADS: one, two and 32 threads per block give the same `G` to
  round-off (a thread's sums over its kets, then its atomic adds).
- AGAINST libcint (`slow`, `high_l` off): `(dd|dd)` on CO/cc-pVTZ, every
  ordered quartet of the class, against the tracked `int2e_ip1_cart`
  witness's class, within `1e-12` of the whole `max|G|` (that test's
  gate).

ROUND-OFF: `|G_kernel - G_host| <= TOL max(scale)`, `TOL = 16 eps`,
`scale` the host's `absolute=True` twin -- test_kernel_gk.py's gate,
unchanged for the other engine. Measured: at most 4.4 eps of the scale
(at `(ss|ss)`), 0.4 eps at `(dd|dd)`, every class through `(dd|dd)`.

Fast tier: every case with `L + 1 <= FAST_L_MAX`. No compiler, no card.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np
import pytest
from test_gradient_host import PAIR_SP
from test_kernel_gk import SPD
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.gradient import host
from felizjueves.gradient.kernel_ghk import (
    build_ghk_kernel,
    ghk_name,
    ghk_spec,
    hrr_classes,
    launch_shape,
    params_of,
    vrr_class,
)
from felizjueves.hgp import vrr
from felizjueves.ir import AtomicAdd, Break, Decl, Intent, Space, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS
_INTS = ("bra_ao", "bra_atom", "ket_ao")

_A, _B = (0.0, 0.1, -0.2), (1.4, -0.3, 0.5)
MIXED = (Basis((Shell(0, _A, (1.3, 0.4, 0.15), (0.3, 0.5, 0.4)),
                Shell(1, _A, (0.8,), (1.1,)),
                Shell(0, _B, (0.6, 2.1), (0.7, 0.2)),
                Shell(1, _B, (1.7, 0.45), (0.5, 0.6)))),
         ((3.0, _A), (1.0, _B)))


def _case(name: str):
    return {"pair_sp": PAIR_SP, "spd": SPD, "mixed": MIXED}[name]


@cache
def _density(name: str) -> np.ndarray:
    n = _case(name)[0].n_ao
    X = np.random.default_rng(5).standard_normal((n, n))
    return X @ X.T / n


def run_class(basis: Basis, nuclei, M: np.ndarray, am, G: np.ndarray, *,
              threads: int = 2) -> np.ndarray:
    """Every launch of class `am` over `basis` into `G` (zeroed by the
    caller), interpreted; returns the new `G`."""
    k = build_ghk_kernel(am)
    names = params_of(am)
    G = np.asarray(G, float).ravel()
    for la in host.hgp_exchange_launches(basis, nuclei, am):
        arrays = {n: np.array(la[n], dtype=np.int32 if n in _INTS else float)
                  for n in names if n in la and not n.startswith(("n_", "k_"))}
        grid, block = launch_shape(la["n_bra"], threads)
        G = Interpreter(k, launch=Launch(grid, block, 32), boys=boys).run(
            n_bra=la["n_bra"], n_ket=la["n_ket"], k_ab=la["k_ab"],
            k_cd=la["k_cd"], n_ao=basis.n_ao, n_atom=len(nuclei),
            M=np.asarray(M, float).ravel(), G=G, **arrays)["G"]
    return G.reshape(len(nuclei), 3)


def _launch(name: str, am, *, threads: int = 2) -> np.ndarray:
    basis, nuclei = _case(name)
    return run_class(basis, nuclei, _density(name), am,
                     np.zeros((len(nuclei), 3)), threads=threads)


def _host(name: str, quartets=None, *, absolute: bool = False) -> np.ndarray:
    basis, nuclei = _case(name)
    return np.array(host.exchange_gradient(
        basis, _density(name), nuclei, one_centre=one_centre,
        quartets=quartets, absolute=absolute))


def _classes(lmax: int, *, need: int = 0):
    return [c for c in itertools.product(range(lmax + 1), repeat=4)
            if max(c) >= need]


def _param(name: str, am):
    return by_l((name, am), sum(am) + 1,
                id=f"{name}-{am[0]}{am[1]}_{am[2]}{am[3]}")


CASES = [*(_param("pair_sp", am) for am in _classes(1)),
         *(_param("spd", am) for am in _classes(2, need=2))]


# --- the key and the structure ----------------------------------------------


def test_the_key_is_k_at_order_one_on_hgp() -> None:
    s = ghk_spec(2, 1, 1, 0)
    assert s == Spec(Operator.COULOMB, 1, (2, 1, 1, 0), Target.K,
                     engine=Engine.HGP)
    assert s.key() == "coulomb/d1/(2,1|1,0)/b1/k.hgp/literal"
    assert s.key() == Spec(Operator.COULOMB, 1, (2, 1, 1, 0),
                           Target.K).key().replace("/k/", "/k.hgp/")
    assert ghk_name(s) == ghk_name((2, 1, 1, 0)) == "ghk_21_10"
    assert build_ghk_kernel((2, 1, 1, 0)).name == "ghk_21_10"


def test_refusals() -> None:
    with pytest.raises(ValueError, match="not a d1 Target.K key on"):
        build_ghk_kernel(Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.K))
    with pytest.raises(ValueError, match="not a d1 Target.K key on"):
        build_ghk_kernel(Spec(Operator.COULOMB, 0, (1, 0, 0, 0), Target.K,
                              engine=Engine.HGP))
    with pytest.raises(ValueError, match="no levers"):
        build_ghk_kernel(Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.K,
                              ("ordered",), engine=Engine.HGP))
    with pytest.raises(TypeError):
        build_ghk_kernel((1, 0, 0))


def test_the_vrr_reaches_every_boundary_of_the_hrrs() -> None:
    assert vrr_class((0, 0, 0, 0)) == (0, 1, 0, 0)
    assert vrr_class((2, 1, 1, 2)) == (1, 3, 1, 2)
    assert set(hrr_classes((0, 0, 1, 1))) == {"a+", "c+"}
    assert hrr_classes((1, 2, 0, 1)) == {
        "a+": ("UA", (2, 2, 0, 1)), "c+": ("UC", (1, 3, 0, 1)),
        "a-": ("U0", (0, 2, 0, 1)), "c-": ("U0", (1, 1, 0, 1))}
    for am in _classes(2):
        targets = vrr.target_set(vrr_class(am))
        for _, cls in hrr_classes(am).values():
            assert vrr.target_set(cls) <= targets, am


@pytest.mark.parametrize("am", [(0, 0, 0, 0), (1, 1, 0, 1), (2, 0, 1, 2)],
                         ids=lambda a: "".join(map(str, a)))
def test_the_mapping_staging_and_write(am) -> None:
    k = build_ghk_kernel(am)
    stmts = list(walk(k.body))
    assert not any(isinstance(s, Break) for s in stmts)
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 and {a.dst.array for a in adds} == {"G"}
    shared = [s for s in stmts if isinstance(s, Decl)
              and s.space is Space.SHARED]
    assert [s.name for s in shared] == ["ab_sh"]
    assert [p.name for p in k.params] == list(params_of(am))
    assert [p.name for p in k.params if p.intent is Intent.OUT] == ["G"]
    assert ("CD" in params_of(am)) == bool(am[3])


# --- against the host --------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_one_class_is_the_host_gradient(case) -> None:
    name, am = case
    basis, _ = _case(name)
    q = host.class_quartets(basis, am)
    assert q, am
    G = _launch(name, am)
    sc = float(_host(name, q, absolute=True).max())
    assert sc > 0.0
    err = float(np.abs(G - _host(name, q)).max())
    assert err <= TOL * sc, f"{am}: {err / sc / EPS:.2f} eps of scale"


def test_the_whole_exchange_gradient_of_the_sp_basis() -> None:
    basis, nuclei = PAIR_SP
    G = np.zeros((len(nuclei), 3))
    for am in _classes(1):
        G = run_class(basis, nuclei, _density("pair_sp"), am, G)
    sc = float(_host("pair_sp", absolute=True).max())
    assert float(np.abs(G - _host("pair_sp")).max()) <= TOL * sc
    assert float(np.abs(G.sum(axis=0)).max()) <= TOL * sc


@pytest.mark.parametrize("am", [(0, 0, 0, 0), (1, 0, 0, 1), (0, 1, 1, 1)],
                         ids=lambda a: "".join(map(str, a)))
def test_mixed_degrees_are_cut_into_launches(am) -> None:
    basis, nuclei = MIXED
    launches = host.hgp_exchange_launches(basis, nuclei, am)
    assert len({(x["k_ab"], x["k_cd"]) for x in launches}) == len(launches)
    assert len(launches) > 1
    assert sum(x["n_bra"] * x["n_ket"] for x in launches) == len(
        host.class_quartets(basis, am))
    q = host.class_quartets(basis, am)
    G = _launch("mixed", am)
    sc = float(_host("mixed", q, absolute=True).max())
    err = float(np.abs(G - _host("mixed", q)).max())
    assert err <= TOL * sc, f"{am}: {err / sc / EPS:.2f} eps of scale"


@pytest.mark.parametrize("threads", [1, 32])
def test_the_thread_count_changes_only_the_round_off(threads) -> None:
    am = (1, 1, 0, 1)
    sc = float(_host("pair_sp", host.class_quartets(PAIR_SP[0], am),
                     absolute=True).max())
    err = float(np.abs(_launch("pair_sp", am, threads=threads)
                       - _launch("pair_sp", am)).max())
    assert err <= TOL * sc


# --- against libcint ---------------------------------------------------------


@pytest.mark.slow
def test_dd_dd_on_co_ccpvtz_is_the_libcint_witness() -> None:
    from oracle import basis as ob
    from oracle import manifest

    w = manifest.k_gradient()
    fx = w.parent
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms)
    basis = ob.from_fixture(fx)
    am = (2, 2, 2, 2)
    G = run_class(basis, nuclei, np.asarray(fx.density), am,
                  np.zeros((len(nuclei), 3)), threads=8)
    scale = float(np.abs(w.whole).max())
    err = float(np.abs(G - w.classes[am]).max()) / scale
    assert err <= 1e-12, f"{err:.3e} of max|G|"
    print(f"\nghk {am} against libcint: {err:.2e} of max|G|")
