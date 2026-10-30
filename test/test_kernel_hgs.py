"""`kernel_hgs`: the overlap gradient on the HGP engine (`hgs_{la}{lb}`, `S`
at d1 on `Engine.HGP`), interpreted.

THE WITNESS is `gradient.host.overlap_gradient`, test_kernel_g1e.py's
(itself held to finite differences of the libcint-witnessed `S`,
test/test_gradient_host.py), which shares neither the kernel's recurrence
(McMurchie & Davidson's `E`, not Obara & Saika's VRR and the HGP HRR) nor
its contraction order (per record, not on contracted values):

- PER CLASS, every ordered pair class (ss) through (ff) on HF/cc-pVTZ, one
  launch over every shell pair `i <= j` of the class (staged by
  `host.pair_records`) into a zeroed `G`, `M` non-symmetric -- against the
  host over the same pairs, within `TOL = 64 eps` of test_kernel_g1e.py's
  split-centre `scale` (its docstring says why `max|G|` is not a scale).
  Not `gs_`'s 16: the HRR runs on CONTRACTED values, its `AB` terms
  cancelling across the pair's primitives where `gs_`'s per-record `E`
  does not. Measured (2026-10-30): worst 17.3 eps at (sd), every other
  class <= 9.8; (pf) and (ff) exactly zero, as the host, by parity on
  HF's one-centre pairs.
- THE WHOLE GRADIENT of H2O/STO-3G and CO/cc-pVTZ (f on both atoms, every
  class), one launch per class into one `G`, the same bound, and
  `SUM_A G[A] = 0` (translational invariance: the kernel computes both
  centres, so it is a check, not an assumption).
- AGAINST `gs_`: the same classes interpreted by `kernel_g1e`, the same
  bound -- the two engines' kernels agree with each other, not only with
  the host.

And the kernel's shape: the key and name, the refusals, the parameter
block equal to the staging, six atomic adds into `G` per shell pair.

`L` for the tier cut is `la + lb + 1`, as test_kernel_g1e.py's. Nothing
compiles and nothing launches on a card.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_g1e import EPS, _fixture, _launch, _m, _scale
from tiers import by_l

from felizjueves.gradient import host
from felizjueves.gradient.kernel_g1e import ZEROED_G
from felizjueves.gradient.kernel_hgs import (
    ATOMIC_REASON,
    L_MAX,
    build_hgs_kernel,
    hgs_name,
    hgs_spec,
    launch_shape,
    pair_arrays,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, Scalar, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.spec import Engine, Operator, Spec, Target

CLASSES = [(la, lb) for la in range(L_MAX + 1) for lb in range(L_MAX + 1)]
TOL = 64 * EPS  # of max(scale): the module docstring


def _close(got, want, scale: float, what: str) -> None:
    if scale == 0.0:  # every term zero by parity (a one-centre p-f overlap)
        assert np.all(got == 0.0) and np.all(want == 0.0), what
        return
    err = np.abs(got - want).max()
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"


def _arrays(basis, nuclei, la, lb, kern):
    rec = host.pair_records(basis, nuclei, la, lb)
    types = {p.name: p.type for p in kern.params if p.array}
    return {k: np.array(v, dtype=np.int32 if types[k] is Scalar.I32
                        else float) for k, v in rec.items() if k in types}


def launch_hgs(basis, nuclei, M, la, lb, G, threads=3):
    """One `hgs_` launch over every shell pair of class `(la, lb)`."""
    kern = build_hgs_kernel((la, lb))
    arrs = _arrays(basis, nuclei, la, lb, kern)
    n = len(arrs["rec0"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1)).run(
        n_pair=n, n_rec=len(arrs["s_ab"]), n_ao=basis.n_ao,
        n_atom=len(nuclei), M=M, G=G, **arrs)["G"]


def _classes(basis):
    return sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                   for b in basis.shells[k:]})


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_gradient(cls) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = host.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    G = launch_hgs(basis, nuclei, M, *cls, np.zeros((len(nuclei), 3)))
    scale = _scale(Target.S, basis, nuclei, M, pairs)
    _close(G, np.array(host.overlap_gradient(basis, M, nuclei, pairs=pairs)),
           scale, f"hgs {cls}")
    # and the MMD kernel of the same class, interpreted
    gs = _launch(Target.S, basis, nuclei, M, *cls, np.zeros((len(nuclei), 3)))
    _close(G, gs, scale, f"hgs {cls} against gs")


@pytest.mark.parametrize("name", [
    pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
    pytest.param("co_ccpvtz_j", id="co_ccpvtz", marks=(pytest.mark.slow,))])
def test_the_whole_gradient_is_the_host_gradient(name) -> None:
    basis, nuclei = _fixture(name)
    M = _m(basis.n_ao)
    G = np.zeros((len(nuclei), 3))
    for la, lb in _classes(basis):
        G = launch_hgs(basis, nuclei, M, la, lb, G)
    pairs = [(i, j) for i in range(len(basis.shells))
             for j in range(i, len(basis.shells))]
    _close(G, np.array(host.overlap_gradient(basis, M, nuclei)),
           _scale(Target.S, basis, nuclei, M, pairs), f"hgs {name}")
    assert np.abs(G.sum(axis=0)).max() <= 1e-13 * np.abs(G).max()


def test_g_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G0 = np.arange(9.0).reshape(3, 3)
    G = launch_hgs(basis, nuclei, M, 0, 1, G0.copy())
    want = launch_hgs(basis, nuclei, M, 0, 1, np.zeros((3, 3)))
    assert np.abs(G - (G0 + want)).max() <= 4 * EPS * np.abs(G0).max()
    assert not np.array_equal(G, want)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    assert hgs_spec(2, 1).key() == "overlap/d1/(2,1)/b1/s.hgp/literal"
    assert hgs_name((2, 1)) == "hgs_21"
    assert build_hgs_kernel(hgs_spec(2, 1)).name == "hgs_21"
    assert len({hgs_name(c) for c in CLASSES}) == len(CLASSES)


@pytest.mark.parametrize("spec", [
    Spec(Operator.OVERLAP, 0, (1, 0), Target.S, engine=Engine.HGP),
    Spec(Operator.OVERLAP, 1, (1, 0), Target.S),
    Spec(Operator.KINETIC, 0, (1, 0), Target.T, engine=Engine.HGP),
], ids=lambda s: s.key())
def test_another_key_is_refused(spec) -> None:
    with pytest.raises(ValueError, match="not a d1 HGP S key"):
        build_hgs_kernel(spec)


def test_above_f_is_refused() -> None:
    with pytest.raises(ValueError, match="through f only"):
        build_hgs_kernel((L_MAX + 1, 0))


def test_the_spec_admits_s_on_hgp_at_order_one_only() -> None:
    assert Spec(Operator.OVERLAP, 1, (1, 0), Target.S,
                engine=Engine.HGP).key() == "overlap/d1/(1,0)/b1/s.hgp/literal"
    with pytest.raises(ValueError, match="derivative order 1 on target v.hgp"):
        Spec(Operator.NUCLEAR, 1, (1, 0), Target.V, engine=Engine.HGP)
    with pytest.raises(ValueError, match="derivative order 2 on target s.hgp"):
        Spec(Operator.OVERLAP, 2, (1, 0), Target.S, engine=Engine.HGP)


def test_the_parameters_are_the_staging() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    k = build_hgs_kernel((1, 1))
    names = [p.name for p in k.params]
    assert names == [
        "n_pair", "n_rec", "n_ao", "n_atom", "zeta_ab", "pa_x", "pa_y",
        "pa_z", "pb_x", "pb_y", "pb_z", "s_ab", "ea_bra", "eb_ket",
        *pair_arrays(), "M", "G"]
    assert [q.name for q in k.params if q.intent is Intent.OUT] == ["G"]
    staged = set(host.pair_records(basis, nuclei, 1, 1))
    assert {p.name for p in k.params if p.array} - {"M", "G"} <= staged


@pytest.mark.parametrize("cls", [(0, 0), (1, 2), (3, 3)], ids=str)
def test_every_write_is_an_atomic_add_into_g(cls) -> None:
    k = build_hgs_kernel(cls)
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 and all(a.dst.array == "G" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
