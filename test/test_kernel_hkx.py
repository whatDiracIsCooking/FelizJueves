"""`kernel_hkx`: the exchange-energy Hessian kernel (`Target.K` at d2),
interpreted.

THE WITNESS is `hessian.host.exchange_hessian` (held to finite differences
of `gradient.host.exchange_gradient`, test/test_hessian_host.py). The two
share the records and the rule and nothing of the arithmetic's order: the
host differentiates `E^{ac}` and `E^{bd}` before the fold and runs
`reference.pq`'s `[r]^(0)`; the kernel folds the SHIFTED bra rows once and
digests them into each component, one accumulator per power of `2 alpha`,
`2 gamma`, and runs the plan's MILD recurrence two bra orders up
(`hkx_pq_spec`).

- PER CLASS, one launch over every ordered bra record of `(la, lc)`
  against every ordered ket record of `(lb, ld)`, the kets grouped by
  atom pair (`host.exchange_hessian_records`), against
  `exchange_hessian(quartets=gradient.host.class_quartets)`: every K class
  with every shell s or p on test_gradient_host.py's two-atom s, p basis
  (four ket segments per class there; classes above `FAST_L_MAX` `slow`,
  by `tiers.by_l` on the raised `L + 2`), and on a three-atom s, p basis
  with two primitives per shell (up to nine segments, contracted records;
  all but `(ss|ss)` `slow`, ~25 s a class interpreted).
- THE WHOLE `d2 E_K` of the s, p basis: the 16 class launches into one
  `H` (`slow`), against the host's every-quartet sweep, symmetric and
  translationally invariant.
- LANES: one lane, four and 32 per warp give the same `H` to round-off
  (the warp reductions reassociate the per-lane sums).

ROUND-OFF: `|H_kernel - H_host| <= TOL max(scale)`, `TOL = 16 eps`,
`scale` the host's `absolute=True` twin (every factor in `|.|`, both terms
of each derivative rule added). Measured: at most ~1 eps of the scale on
every class through `(pp|pp)`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from test_gradient_host import PAIR_SP
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.gradient import host as gh
from felizjueves.hessian import host
from felizjueves.hessian.kernel_hkx import (
    PARAMS,
    _derivative_terms,
    build_hkx_kernel,
    hkx_name,
    hkx_pq_spec,
    hkx_spec,
    launch_shape,
)
from felizjueves.ir import AtomicAdd, For, Intent, WarpReduce, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd.kernel import split
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS

_A, _B, _C = (0.0, 0.1, -0.2), (1.4, -0.3, 0.5), (-0.6, 1.1, 0.8)
TRI_SP = (Basis((Shell(1, _A, (1.1, 0.4), (0.6, 0.5)),
                 Shell(0, _B, (0.9, 0.3), (0.7, 0.4)),
                 Shell(1, _C, (0.8, 0.35), (0.5, 0.6)),
                 Shell(0, _A, (1.3, 0.5), (0.4, 0.8)))),
          ((2.0, _A), (1.0, _B), (3.0, _C)))


def _case(name: str):
    return {"pair_sp": PAIR_SP, "tri_sp": TRI_SP}[name]


@cache
def _density(name: str) -> np.ndarray:
    n = _case(name)[0].n_ao
    X = np.random.default_rng(11).standard_normal((n, n))
    return X @ X.T / n


def _launch(name: str, am, H: np.ndarray, lanes: int = 4) -> np.ndarray:
    basis, nuclei = _case(name)
    bra, ket, seg = host.exchange_hessian_records(basis, nuclei, am)
    arrays = {k: np.array(v, dtype=np.int32
                          if k.startswith(("off", "atom", "seg"))
                          else np.float64)
              for k, v in {**bra, **ket, **seg}.items()}
    n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
    grid, block = launch_shape(n_bra, lanes)
    return Interpreter(build_hkx_kernel(am), launch=Launch(grid, block, lanes),
                       boys=boys).run(
        n_bra=n_bra, n_ket=n_ket, n_seg=len(seg["seg_b"]), n_ao=basis.n_ao,
        n_atom=len(nuclei), M=_density(name), H=H, **arrays)["H"]


def _host(name: str, quartets=None, *, absolute: bool = False) -> np.ndarray:
    basis, nuclei = _case(name)
    n = len(nuclei)
    return np.array(host.exchange_hessian(
        basis, _density(name), nuclei, one_centre=one_centre,
        quartets=quartets, absolute=absolute)).reshape(n, 3, n, 3)


def _classes(lmax: int):
    r = range(lmax + 1)
    return [(a, b, c, d) for a in r for b in r for c in r for d in r]


def _param(name: str, am):
    return by_l((name, am), sum(am) + 2,
                id=f"{name}-{am[0]}{am[1]}_{am[2]}{am[3]}")


def _tri(am):
    # three atoms, two primitives a shell: ~25 s a class interpreted, so
    # only (ss|ss) is in the fast tier
    return pytest.param(("tri_sp", am),
                        id=f"tri_sp-{am[0]}{am[1]}_{am[2]}{am[3]}",
                        marks=(pytest.mark.slow,) if any(am) else ())


CASES = [*(_param("pair_sp", am) for am in _classes(1)),
         *(_tri(am) for am in _classes(1))]


@pytest.mark.parametrize("case", CASES)
def test_one_launch_over_the_class_is_the_host_hessian(case) -> None:
    name, am = case
    basis, nuclei = _case(name)
    q = gh.class_quartets(basis, am)
    assert q, am
    n = len(nuclei)
    H = _launch(name, am, np.zeros((n, 3, n, 3)))
    want = _host(name, q)
    sc = float(_host(name, q, absolute=True).max())
    assert sc > 0.0
    err = float(np.abs(H - want).max())
    assert err <= TOL * sc, f"{am}: {err / sc / EPS:.2f} eps of scale"


@pytest.mark.slow
def test_the_classes_sum_to_the_whole_exchange_hessian() -> None:
    _, nuclei = PAIR_SP
    n = len(nuclei)
    H = np.zeros((n, 3, n, 3))
    for am in _classes(1):
        H = _launch("pair_sp", am, H)
    want = _host("pair_sp")
    sc = float(_host("pair_sp", absolute=True).max())
    assert np.abs(H - want).max() <= TOL * sc
    M = H.reshape(3 * n, 3 * n)
    assert np.abs(M - M.T).max() <= 1e-13 * np.abs(M).max()
    assert np.abs(H.sum(axis=0)).max() <= 1e-13 * np.abs(M).max()


@pytest.mark.parametrize("lanes", [1, 32])
def test_the_lane_count_is_a_reassociation(lanes: int) -> None:
    am = (1, 0, 0, 1)
    z = np.zeros((2, 3, 2, 3))
    ref = _launch("pair_sp", am, z.copy())
    got = _launch("pair_sp", am, z.copy(), lanes=lanes)
    q = gh.class_quartets(PAIR_SP[0], am)
    sc = float(_host("pair_sp", q, absolute=True).max())
    assert np.abs(got - ref).max() <= TOL * sc


# --- the derivative terms -----------------------------------------------------


def test_the_twice_applied_rule_on_one_axis() -> None:
    # dA_x dA_x on a = (1, 0, 0): (2a)^2 [a + 2x] - 3 (2a) [a] (the lowered
    # row's own - 0 term absent), + 1 * 0 [a - 2x] absent
    t = _derivative_terms((1, 0, 0), (0, 0, 0), (("A", 0), ("A", 0)))
    assert t == {((2, 0, 0), (0, 0, 0), 2, 0): 1,
                 ((0, 0, 0), (0, 0, 0), 1, 0): -3}
    # a = (2, 0, 0): ... + a (a - 1) = 2 on [a - 2x]
    t = _derivative_terms((2, 0, 0), (0, 0, 0), (("A", 0), ("A", 0)))
    assert t[((-2, 0, 0), (0, 0, 0), 0, 0)] == 2
    assert t[((0, 0, 0), (0, 0, 0), 1, 0)] == -5


def test_mixed_centres_are_a_product_and_commute() -> None:
    a, c = (1, 0, 1), (0, 1, 0)
    ac = _derivative_terms(a, c, (("A", 2), ("C", 1)))
    ca = _derivative_terms(a, c, (("C", 1), ("A", 2)))
    assert ac == ca == {
        ((0, 0, 1), (0, 1, 0), 1, 1): 1,
        ((0, 0, 1), (0, -1, 0), 1, 0): -1,
        ((0, 0, -1), (0, 1, 0), 0, 1): -1,
        ((0, 0, -1), (0, -1, 0), 0, 0): 1,
    }


# --- the kernel's shape -------------------------------------------------------


def test_the_key_the_name_and_the_plan_class() -> None:
    assert hkx_spec(2, 1, 1, 0).key() == "coulomb/d2/(2,1|1,0)/b1/k/literal"
    assert hkx_name((1, 0, 1, 1)) == build_hkx_kernel((1, 0, 1, 1)).name \
        == "hkx_10_11"
    pqs = hkx_pq_spec(2, 1, 1, 0)
    assert pqs.target is Target.K and pqs.derivative_order == 0
    assert split(pqs) == (5, 1)


@pytest.mark.parametrize("spec", [
    Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.K),
    Spec(Operator.COULOMB, 2, (1, 0), Target.JAB),
    Spec(Operator.COULOMB, 2, (1, 0, 0, 0), Target.K, n_batches=3),
], ids=["d1", "jab", "batched"])
def test_what_is_not_a_one_batch_d2_k_key_is_refused(spec) -> None:
    with pytest.raises(ValueError, match="d2 Target.K key|one batch only"):
        build_hkx_kernel(spec)


def test_the_parameters_the_one_output_and_the_adds() -> None:
    k = build_hkx_kernel((1, 0, 1, 1))
    assert tuple(p.name for p in k.params) == PARAMS
    assert [p.name for p in k.params if p.intent is Intent.OUT] == ["H"]
    stmts = list(walk(k.body))
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    # 36 bra-ket per segment, 36 bra-bra (AA, CC, AC and its mirror)
    assert len(adds) == 72 and all(a.dst.array == "H" for a in adds)
    assert sum(isinstance(s, WarpReduce) for s in stmts) == 36 + 21
    loops = [s.var for s in stmts if isinstance(s, For)]
    assert loops == ["sg", "k"]
