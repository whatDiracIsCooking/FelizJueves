"""The `Engine` protocol (#292): both engines satisfy it, every member is
justified by both, and the cross-engine pin walks `engines()`.

THE CROSS-ENGINE CHECK is test/test_hgp_eri.py's (#281), lifted off its
hand-maintained pair: every PAIR of `engines()` must agree on `(ab|cd)` of a
contracted shell quartet, element by element, within

    |a - b| <= 1e-12 * (scale_a + scale_b),

each scale its own engine's absolute-value twin (`eri_and_scale`) -- the
same bound, the same quartet rule (`test_hgp_eri.pick`: most distinct
centres, then most primitive quartets, then shell-index order) and the same
justification (that module's docstring, "THE TOLERANCE"; its rounding-budget
test covers these quartets). Here it runs every class of milestone 12's
ladder on HF/cc-pVTZ, and each engine against libcint on H2O/6-31G's
quartets with every shell `l <= 1`, the libcint side's yardstick the summed
scale of every engine as there. A third engine joins every check by joining
`engines()`.

Fast tier: no compiler, no card.
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import typing
from functools import cache

import numpy as np
import pytest
from oracle import manifest
from test_hgp_eri import CLASSES, TOL, _fixture, _name, _quartet, _shells, pick
from tiers import split_by_l

import felizjueves.engine as E
from felizjueves.eri.fock.fock import coulomb_exchange_reference
from felizjueves.ir import Kernel
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

ENGINES = E.engines(pq=pq, boys=boys)
PAIRS = tuple(itertools.combinations(ENGINES, 2))


def _ids(x) -> str:
    return x.name if hasattr(x, "name") else "-".join(e.name for e in x)


@cache
def _eri(engine_name: str, mol: str, q) -> tuple[np.ndarray, np.ndarray]:
    (engine,) = (e for e in ENGINES if e.name == engine_name)
    blk, scale = engine.eri_and_scale(*_quartet(mol, q))
    return np.array(blk).ravel(), np.array(scale).ravel()


def _worst(got, want, scale) -> float:
    err = np.abs(got - want)
    assert np.all(err[scale == 0.0] == 0.0)
    return float(np.max(np.divide(err, scale, out=np.zeros_like(err),
                                  where=scale > 0)))


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


def test_both_engines_satisfy_the_protocol() -> None:
    assert len(ENGINES) == 2
    assert [e.name for e in ENGINES] == ["mmd", "hgp"]
    assert len({e.target for e in ENGINES}) == len(ENGINES)
    for e in ENGINES:
        assert isinstance(e, E.Engine), e


def test_every_member_is_justified_by_both_implementations() -> None:
    """`MEMBERS` is exactly the protocol's surface, and each member names two
    DIFFERENT callables, one per engine, that exist."""
    surface = set(typing.get_type_hints(E.Engine)) | {
        n for n, v in vars(E.Engine).items()
        if inspect.isfunction(v) and not n.startswith("_")}
    assert surface == set(E.MEMBERS)
    for member, (mmd, hgp) in E.MEMBERS.items():
        assert mmd != hgp, member
        for ref in (mmd, hgp):
            mod, attr = ref.split(":")
            assert callable(getattr(
                importlib.import_module(f"felizjueves.{mod}"), attr)), ref


def test_the_push_back_down_list_is_in_the_docstring() -> None:
    """Pin 12's primary report lives where someone will find it: the six
    items #292 names, and `pair_bound`'s demotion, are in engine.py's
    docstring -- and none of them is a protocol member."""
    doc = E.__doc__
    assert "PUSHED BACK DOWN" in doc
    for item in ("n_batches", "MILD", "sieve", "(r, m)", "total_am",
                 "plan/cost.py", "`pair_bound`"):
        assert item in doc, item
    for e in ENGINES:
        for private in ("n_batches", "plan", "pair_bound", "launch_shape"):
            assert not hasattr(e, private), (e.name, private)


@pytest.mark.parametrize("engine", ENGINES, ids=_ids)
def test_kernel_builds_its_own_target_and_refuses_the_others(engine) -> None:
    k = engine.kernel(Spec(Operator.COULOMB, 0, (0, 1, 0, 1), engine.target))
    assert isinstance(k, Kernel)
    for other in ENGINES:
        if other.target is not engine.target:
            with pytest.raises(ValueError, match="builds Target"):
                engine.kernel(Spec(Operator.COULOMB, 0, (0, 1, 0, 1),
                                   other.target))
    # K shares J's quartet shape and plan; the MMD builder alone would plan
    # it as J, so the protocol's guard is what refuses it.
    if engine.target is not Target.K:
        with pytest.raises(ValueError, match="builds Target"):
            engine.kernel(Spec(Operator.COULOMB, 0, (0, 1, 0, 1), Target.K))
    with pytest.raises(TypeError):
        engine.kernel((0, 1, 0, 1))


@pytest.mark.parametrize("engine", ENGINES, ids=_ids)
def test_eri_is_the_first_half_of_eri_and_scale(engine) -> None:
    bra, ket = _quartet("hf_ccpvtz", pick("hf_ccpvtz", (1, 1, 0, 2)))
    blk, scale = engine.eri_and_scale(bra, ket)
    assert engine.eri(bra, ket) == blk
    assert np.all(np.array(scale) >= np.abs(np.array(blk)))


# ---------------------------------------------------------------------------
# The cross-engine pin, over engines()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", split_by_l(CLASSES, sum, ids=_name))
@pytest.mark.parametrize("pair", PAIRS, ids=_ids)
def test_every_pair_of_engines_agrees_on_contracted_quartets(pair, cls) -> None:
    a, b = pair
    q = pick("hf_ccpvtz", cls)
    ga, sa = _eri(a.name, "hf_ccpvtz", q)
    gb, sb = _eri(b.name, "hf_ccpvtz", q)
    w = _worst(ga, gb, sa + sb)
    assert w <= TOL, (a.name, b.name, cls, q, w)


def _libcint_params():
    name = "h2o_631g_eri"
    sh = _shells(name)
    return [pytest.param(q, id=f"{name}-{q}")
            for q in sorted(manifest.eri_blocks(_fixture(name)))
            if max(sh[s].l for s in q) <= 1]


@pytest.mark.parametrize("q", _libcint_params())
@pytest.mark.parametrize("engine", ENGINES, ids=_ids)
def test_every_engine_matches_libcint(engine, q) -> None:
    name = "h2o_631g_eri"
    ref = manifest.eri_blocks(_fixture(name))[q].ravel()
    got, _ = _eri(engine.name, name, q)
    scale = sum(_eri(e.name, name, q)[1] for e in ENGINES)
    assert _worst(got, ref, scale) <= TOL, (engine.name, q)


@pytest.mark.parametrize("pair", PAIRS, ids=_ids)
def test_every_engine_fills_the_fock_witness_slot(pair) -> None:
    """`engine.eri` is `ao.fock`'s `eri=` -- the module above the engines
    takes either through the protocol: H2/STO-3G's `J` and `K` agree within
    `TOL` of the sweep run on `|D|` with the summed scale."""
    from oracle import basis as ob
    fx = _fixture("h2_sto3g_j")
    basis = ob.from_fixture(fx)
    D = np.array(fx.density)
    a, b = pair
    aj, ak = (np.array(x) for x in coulomb_exchange_reference(
        basis, D, eri=a.eri))
    bj, bk = (np.array(x) for x in coulomb_exchange_reference(
        basis, D, eri=b.eri))

    def scale(bra, ket):
        return tuple(map(tuple, np.array(a.eri_and_scale(bra, ket)[1])
                         + np.array(b.eri_and_scale(bra, ket)[1])))

    sj, sk = (np.array(x) for x in coulomb_exchange_reference(
        basis, np.abs(D), eri=scale))
    assert np.all(np.abs(aj - bj) <= TOL * sj)
    assert np.all(np.abs(ak - bk) <= TOL * sk)
    assert np.max(np.abs(aj)) > 0.1


# ---------------------------------------------------------------------------
# The widened protocol (#406): `j` and `k`, the split build's kernels
# ---------------------------------------------------------------------------

SPLIT = {Target.J: "j", Target.K: "k"}


@pytest.mark.parametrize("engine", ENGINES, ids=_ids)
@pytest.mark.parametrize("target", list(SPLIT), ids=lambda t: t.value)
def test_j_and_k_build_this_engines_split_kernel(engine, target) -> None:
    """`j` / `k` take a `Target.J` / `Target.K` spec on the engine's OWN
    recurrence (`Spec.engine`, #401) and build exactly the kernel the
    member's named callable builds; the other engine's spec, the other split
    target and `F` are refused, as is a non-Spec."""
    member = getattr(engine, SPLIT[target])
    own = E.RECURRENCE[engine.name]
    spec = Spec(Operator.COULOMB, 0, (0, 1, 0, 1), target, engine=own)
    k = member(spec)
    assert isinstance(k, Kernel)
    side = [e.name for e in ENGINES].index(engine.name)
    mod, attr = E.MEMBERS[SPLIT[target]][side].split(":")
    built = getattr(importlib.import_module(f"felizjueves.{mod}"), attr)(spec)
    assert k == built
    for other in ENGINES:
        if other.name != engine.name:
            with pytest.raises(ValueError, match="builds Target"):
                member(Spec(Operator.COULOMB, 0, (0, 1, 0, 1), target,
                            engine=E.RECURRENCE[other.name]))
    wrong = Target.K if target is Target.J else Target.J
    with pytest.raises(ValueError, match="builds Target"):
        member(Spec(Operator.COULOMB, 0, (0, 1, 0, 1), wrong, engine=own))
    with pytest.raises(ValueError, match="builds Target"):
        member(Spec(Operator.COULOMB, 0, (0, 1, 0, 1), Target.F))
    with pytest.raises(TypeError):
        member((0, 1, 0, 1))


def test_hgp_j_and_k_are_the_split_kernels_beside_f() -> None:
    """HGP builds three targets: `kernel` its native fused `F`, unchanged by
    the widening, and `j` / `k` the split `hj_` / `hk_` kernels (#404,
    #405) in both class modes."""
    from felizjueves.eri.hjk import kernel_hj, kernel_hk
    (hgp,) = (e for e in ENGINES if e.name == "hgp")
    assert hgp.target is Target.F
    assert hgp.kernel(Spec(Operator.COULOMB, 0, (0, 1, 0, 1),
                           Target.F)).name.startswith("f_")
    for ordered in (False, True):
        j = hgp.j(kernel_hj.hj_class_spec(1, 0, 0, 0, ordered=ordered))
        k = hgp.k(kernel_hk.hk_class_spec(1, 0, 0, 0, ordered=ordered))
        assert j.name == ("hj_o_1000" if ordered else "hj_0010")
        assert k.name == kernel_hk.hk_name(
            kernel_hk.hk_class_spec(1, 0, 0, 0, ordered=ordered))
        assert k.name.startswith("hk_o_" if ordered else "hk_")
    assert set(E.RECURRENCE) == {e.name for e in ENGINES}
