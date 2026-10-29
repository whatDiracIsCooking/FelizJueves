"""`kernel_gk`: the exchange-energy gradient kernel (`Target.K` at d1),
interpreted.

THE WITNESS is `gradient.host.exchange_gradient` (held to finite
differences of `eri.k_exchange.k.exchange_matrix`,
test/test_gradient_host.py). The two share the records and the rule and
nothing of the arithmetic's order: the host differentiates `E^{ac}`
before its fold and runs `reference.pq`'s `[r]^(0)`; the kernel folds the
RAISED rows `V(a +- 1_k, c)` once and digests them into each axis, and
runs the plan's MILD recurrence one bra order up (`gk_pq_spec`).

- PER CLASS, one launch over every ordered bra record of `(la, lc)`
  against every ordered ket record of `(lb, ld)`
  (`host.exchange_records`), against `exchange_gradient(quartets=
  host.class_quartets)`: every K class with every shell s or p on
  test_gradient_host.py's two-atom s, p basis, and every class with a d
  shell through `(dd|dd)` on a two-atom s, p, d basis (classes above
  `FAST_L_MAX` `slow`, by `tiers.by_l` on the raised `L + 1`).
- THE WHOLE `dE_K` of the s, p basis: the 16 class launches into one `G`,
  against the host's every-quartet sweep, translationally invariant.
- LANES: one lane, four and 32 per warp give the same `G` to round-off
  (the warp reduction is a reassociation of the per-lane sums).

ROUND-OFF: `|G_kernel - G_host| <= TOL max(scale)`, `TOL = 16 eps`,
`scale` the host's `absolute=True` twin (every factor in `|.|`, both terms
of the derivative rule added). Measured: at most ~0.7 eps of the scale
on every class through `(dd|dd)`.

THE ROLLED FORM (#618, `build_gk_kernel(..., lowstack=True)`, the default
from `L = LOWSTACK_MIN_L` and for every class with an f shell):

- THE DIFFERENTIAL, through `(dd|dd)`: every K class with every `l <= 2`,
  the rolled form against the unrolled one (validated above) on the same
  records, within `TOL` of the class's `absolute=True` scale -- not
  bitwise: the rolled form folds each derivative term over the bra's
  compact tables per `(a, c)` row where the unrolled one folds each raised
  row once, and runs stage 1 per axis. Fast at `L + 1 <= 4`, `slow` above.
- THE f WITNESSES (`slow`): `(ff|ff)`, `(fs|fs)` and `(fd|pd)` on a
  two-atom, single-primitive s, p, d, f basis (`SPDF`), against
  `host.exchange_gradient` over the launched shell pairs' quartets, within
  `TOL` of the twin. `(ff|ff)` launches the off-centre bra pair `(f_A,
  f_B)` against the two off-centre ket pairs only (the rolled loops are
  generic over the class: f adds no new code path; a whole `(ff|ff)` class
  is ~16 interpreted quartets at minutes each).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from test_gradient_host import PAIR_SP
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.gradient import host
from felizjueves.gradient.kernel_gk import (
    LOWSTACK_MIN_L,
    PARAMS,
    UNROLLED_L_MAX,
    build_gk_kernel,
    compact_extent,
    compact_offset,
    gk_lowstack,
    gk_name,
    gk_pq_spec,
    gk_spec,
    launch_shape,
)
from felizjueves.ir import AtomicAdd, Intent, WarpReduce, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd.kernel import split
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS

_A, _B = (0.0, 0.1, -0.2), (1.4, -0.3, 0.5)
SPD = (Basis((Shell(0, _A, (1.1,), (0.7,)), Shell(2, _A, (0.6,), (1.3,)),
              Shell(1, _B, (0.7,), (1.2,)), Shell(2, _B, (0.9,), (0.7,)))),
       ((2.0, _A), (1.0, _B)))


SPDF = (Basis((Shell(0, _A, (1.1,), (0.7,)), Shell(2, _A, (0.6,), (1.3,)),
               Shell(3, _A, (0.8,), (0.9,)), Shell(1, _B, (0.7,), (1.2,)),
               Shell(2, _B, (0.9,), (0.7,)), Shell(3, _B, (1.0,), (1.1,)))),
        ((2.0, _A), (1.0, _B)))


def _case(name: str):
    return {"pair_sp": PAIR_SP, "spd": SPD, "spdf": SPDF}[name]


@cache
def _density(name: str) -> np.ndarray:
    n = _case(name)[0].n_ao
    X = np.random.default_rng(5).standard_normal((n, n))
    return X @ X.T / n


def _only(rec: dict, keys: tuple[str, str], offs) -> dict:
    """The records of `rec` whose AO offsets `(rec[keys[0]], rec[keys[1]])`
    are in `offs` (a shell pair's records, by its two shells' offsets)."""
    keep = [i for i, o in enumerate(zip(rec[keys[0]], rec[keys[1]],
                                        strict=True)) if o in offs]
    return {k: [v[i] for i in keep] for k, v in rec.items()}


def _launch(name: str, am, G: np.ndarray, lanes: int = 4, *,
            lowstack: bool | None = None, bra_pairs=None,
            ket_pairs=None) -> np.ndarray:
    """One launch over the class's records (or only those of the shell
    pairs `bra_pairs` / `ket_pairs`, `(i, k)` shell indices)."""
    basis, nuclei = _case(name)
    bra, ket = host.exchange_records(basis, nuclei, am)
    off = basis.offsets
    if bra_pairs is not None:
        bra = _only(bra, ("off_a", "off_c"),
                    {(off[i], off[k]) for i, k in bra_pairs})
    if ket_pairs is not None:
        ket = _only(ket, ("off_b", "off_d"),
                    {(off[j], off[m]) for j, m in ket_pairs})
    arrays = {k: np.array(v, dtype=np.int32 if k.startswith(("off", "atom"))
                          else np.float64) for k, v in {**bra, **ket}.items()}
    n_bra, n_ket = len(bra["s_ac"]), len(ket["s_bd"])
    grid, block = launch_shape(n_bra, lanes)
    return Interpreter(build_gk_kernel(am, lowstack=lowstack),
                       launch=Launch(grid, block, lanes), boys=boys).run(
        n_bra=n_bra, n_ket=n_ket, n_ao=basis.n_ao, n_atom=len(nuclei),
        M=_density(name), G=G, **arrays)["G"]


def _host(name: str, quartets=None, *, absolute: bool = False) -> np.ndarray:
    basis, nuclei = _case(name)
    return np.array(host.exchange_gradient(
        basis, _density(name), nuclei, one_centre=one_centre,
        quartets=quartets, absolute=absolute))


def _classes(lmax: int, *, need: int = 0):
    r = range(lmax + 1)
    return [c for c in ((a, b, c, d) for a in r for b in r for c in r
                        for d in r) if max(c) >= need]


def _param(name: str, am):
    return by_l((name, am), sum(am) + 1,
                id=f"{name}-{am[0]}{am[1]}_{am[2]}{am[3]}")


CASES = [*(_param("pair_sp", am) for am in _classes(1)),
         *(_param("spd", am) for am in _classes(2, need=2))]


@pytest.mark.parametrize("case", CASES)
def test_one_launch_over_the_class_is_the_host_gradient(case) -> None:
    name, am = case
    basis, nuclei = _case(name)
    q = host.class_quartets(basis, am)
    assert q, am
    G = _launch(name, am, np.zeros((len(nuclei), 3)), lowstack=False)
    want = _host(name, q)
    sc = float(_host(name, q, absolute=True).max())
    assert sc > 0.0
    err = float(np.abs(G - want).max())
    assert err <= TOL * sc, f"{am}: {err / sc / EPS:.2f} eps of scale"


@pytest.mark.parametrize("case", CASES)
def test_the_rolled_form_is_the_unrolled_one(case) -> None:
    # the differential (module docstring): the unrolled form is the
    # validated reference through (dd|dd)
    name, am = case
    basis, nuclei = _case(name)
    want = _launch(name, am, np.zeros((len(nuclei), 3)), lowstack=False)
    got = _launch(name, am, np.zeros((len(nuclei), 3)), lowstack=True)
    sc = float(_host(name, host.class_quartets(basis, am),
                     absolute=True).max())
    err = float(np.abs(got - want).max())
    assert err <= TOL * sc, f"{am}: {err / sc / EPS:.2f} eps of scale"


# (class, bra shell pairs, ket shell pairs) on SPDF: shells 0 s_A, 1 d_A,
# 2 f_A, 3 p_B, 4 d_B, 5 f_B (module docstring, THE f WITNESSES)
F_WITNESSES = [
    ((3, 3, 3, 3), [(2, 5)], [(2, 5), (5, 2)]),
    ((3, 0, 3, 0), None, None),
    ((3, 2, 1, 2), None, None),
]


@pytest.mark.slow
@pytest.mark.parametrize(("am", "bras", "kets"), F_WITNESSES,
                         ids=lambda v: "".join(map(str, v))
                         if isinstance(v, tuple) else None)
def test_an_f_witness_is_the_host_gradient(am, bras, kets) -> None:
    basis, nuclei = SPDF
    pairs = host.exchange_pairs
    bras = bras or pairs(basis, am[0], am[1])
    kets = kets or pairs(basis, am[2], am[3])
    q = [(i, k, j, m) for i, k in bras for j, m in kets]
    G = _launch("spdf", am, np.zeros((len(nuclei), 3)), bra_pairs=bras,
                ket_pairs=kets)
    want = _host("spdf", q)
    sc = float(_host("spdf", q, absolute=True).max())
    assert sc > 0.0 and float(np.abs(want).max()) > 1e-6 * sc
    err = float(np.abs(G - want).max())
    assert err <= TOL * sc, f"{am}: {err / sc / EPS:.2f} eps of scale"


@pytest.mark.parametrize("lanes", [1, 32])
def test_the_rolled_lane_count_is_a_reassociation(lanes: int) -> None:
    am = (1, 0, 1, 1)
    ref = _launch("pair_sp", am, np.zeros((2, 3)), lowstack=True)
    got = _launch("pair_sp", am, np.zeros((2, 3)), lanes=lanes,
                  lowstack=True)
    q = host.class_quartets(PAIR_SP[0], am)
    sc = float(_host("pair_sp", q, absolute=True).max())
    assert np.abs(got - ref).max() <= TOL * sc


def test_the_classes_sum_to_the_whole_exchange_gradient() -> None:
    basis, nuclei = PAIR_SP
    G = np.zeros((len(nuclei), 3))
    for am in _classes(1):
        G = _launch("pair_sp", am, G)
    want = _host("pair_sp")
    sc = float(_host("pair_sp", absolute=True).max())
    assert np.abs(G - want).max() <= TOL * sc
    assert np.abs(G.sum(axis=0)).max() <= 1e-13 * np.abs(G).max()


@pytest.mark.parametrize("lanes", [1, 32])
def test_the_lane_count_is_a_reassociation(lanes: int) -> None:
    am = (1, 1, 1, 0)
    ref = _launch("pair_sp", am, np.zeros((2, 3)))
    got = _launch("pair_sp", am, np.zeros((2, 3)), lanes=lanes)
    q = host.class_quartets(PAIR_SP[0], am)
    sc = float(_host("pair_sp", q, absolute=True).max())
    assert np.abs(got - ref).max() <= TOL * sc


# --- the kernel's shape -------------------------------------------------------


def test_the_key_the_name_and_the_plan_class() -> None:
    assert gk_spec(2, 1, 1, 0).key() == "coulomb/d1/(2,1|1,0)/b1/k/literal"
    assert gk_name((2, 1, 1, 0)) == build_gk_kernel((2, 1, 1, 0)).name \
        == "gk_21_10"
    pqs = gk_pq_spec(2, 1, 1, 0)
    assert pqs.target is Target.K and pqs.derivative_order == 0
    assert split(pqs) == (4, 1)


@pytest.mark.parametrize("spec", [
    Spec(Operator.COULOMB, 0, (1, 0, 0, 0), Target.K),
    Spec(Operator.COULOMB, 1, (1, 0), Target.JAB),
    Spec(Operator.COULOMB, 1, (1, 0, 0, 0), Target.K, n_batches=3),
], ids=["d0", "jab", "batched"])
def test_what_is_not_a_one_batch_d1_k_key_is_refused(spec) -> None:
    with pytest.raises(ValueError, match="d1 Target.K key|one batch only"):
        build_gk_kernel(spec)


def test_the_parameters_the_one_output_and_the_six_adds() -> None:
    k = build_gk_kernel((1, 0, 1, 1))
    assert tuple(p.name for p in k.params) == PARAMS
    assert [p.name for p in k.params if p.intent is Intent.OUT] == ["G"]
    stmts = list(walk(k.body))
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 and all(a.dst.array == "G" for a in adds)
    assert sum(isinstance(s, WarpReduce) for s in stmts) == 6


def test_an_s_shell_has_no_lowered_row() -> None:
    # `- a_k E^{a-1_k}` needs a_k >= 1: no l accumulator for an s shell
    text = repr(build_gk_kernel((0, 0, 1, 1)).body)
    assert "'lA0'" not in text and "'lC0'" not in text
    text = repr(build_gk_kernel((1, 1, 0, 0)).body)
    assert "'lA0'" in text and "'lC0'" in text


# --- the rolled form's shape and gate (#618) ------------------------------------


def test_the_gate_is_an_l_floor_and_any_f_shell() -> None:
    assert (LOWSTACK_MIN_L, UNROLLED_L_MAX) == (5, 2)
    r = range(4)
    for am in ((a, b, c, d) for a in r for b in r for c in r for d in r):
        assert gk_lowstack(am) == (sum(am) >= 5 or max(am) == 3), am


def test_the_rolled_form_keeps_the_name_parameters_and_six_adds() -> None:
    # the unrolled form is built at an s / p class only: never at f (#618)
    want = build_gk_kernel((1, 0, 1, 1), lowstack=False).params
    for am in ((1, 0, 1, 1), (2, 2, 1, 0), (3, 3, 3, 3)):
        k = build_gk_kernel(am, lowstack=True)
        assert k.name == gk_name(am)
        assert tuple(p.name for p in k.params) == PARAMS
        assert k.params == want
        stmts = list(walk(k.body))
        adds = [s for s in stmts if isinstance(s, AtomicAdd)]
        assert len(adds) == 6 and all(a.dst.array == "G" for a in adds)
        assert sum(isinstance(s, WarpReduce) for s in stmts) == 6


def test_below_the_floor_the_default_is_the_unrolled_text() -> None:
    for am in ((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 0, 0), (2, 0, 1, 1)):
        assert build_gk_kernel(am) == build_gk_kernel(am, lowstack=False)
    for am in ((2, 2, 1, 0), (3, 0, 0, 0)):
        assert build_gk_kernel(am) == build_gk_kernel(am, lowstack=True)


def test_the_compact_tables_hold_every_reachable_entry_once() -> None:
    for l1, l2 in ((0, 0), (1, 3), (4, 4), (3, 1)):
        slots = [compact_offset(i, j, l2) + t for i in range(l1 + 1)
                 for j in range(l2 + 1) for t in range(i + j + 1)]
        assert sorted(slots) == list(range(compact_extent(l1, l2)))
    assert compact_extent(4, 4) == 125
