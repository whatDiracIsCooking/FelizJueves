"""`kernel_hgmu`: the dipole gradient on the HGP engine (`hgmu_{la}{lb}`,
`DIPOLE` at d1 on `Engine.HGP`), interpreted.

THE WITNESS is `gradient.host.multipole_gradient` at `e = 1`,
test_kernel_gmu.py's (itself held to libcint's `int1e_irp_cart`, to central
differences of `multipole_matrix` and to the origin shift), which shares
neither the kernel's recurrence (McMurchie & Davidson's `E` and Hermite
moments, not Obara & Saika's VRR, the HGP HRR and the operator moved onto
the ket) nor its contraction order (per record, not on contracted values):

- PER CLASS, every ordered pair class (ss) through (ff) on HF/cc-pVTZ, one
  launch over every shell pair `i <= j` of the class (staged by
  `host.pair_records(..., ket_centre=True)`) into a zeroed `G`, `M`
  non-symmetric, the origin `C` off every atom -- against the host over the
  same pairs, within `TOL = 16 eps` of test_kernel_gmu.py's split-centre
  `scale` per component, `gmu_`'s gate. Measured (2026-11-09): worst 6.3
  eps at (dp), the rest <= 6.0.
- AGAINST `gmu_`: the same classes interpreted by `kernel_gmu`, the same
  bound -- the two engines' kernels agree with each other.
- THE WHOLE GRADIENT of H2O/STO-3G and CO/cc-pVTZ (`slow`), one launch per
  class into one `G`, the same bound (measured 5.4e-16 / 2.4e-14 of
  `max|G|`), and the ORIGIN SHIFT: `C` is not a
  nucleus, so `SUM_A G[c, A, k] = q delta_ck SUM M S` -- the kernel computes
  both centres, so it is a check, not an assumption.

And the kernel's shape: the key and name, the refusals, the parameter block
equal to the staging, eighteen atomic adds into `G` per shell pair.

`L` for the tier cut is `la + lb + 1`, as test_kernel_gmu.py's. Nothing
compiles and nothing launches on a card.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_gmu import EPS, ORIGIN, Q, _fixture, _host, _m, _scale
from test_kernel_gmu import _launch as _launch_gmu
from tiers import by_l

from felizjueves.gradient import host
from felizjueves.gradient.kernel_g1e import ZEROED_G
from felizjueves.gradient.kernel_hgmu import (
    ATOMIC_REASON,
    L_MAX,
    N_COMP,
    build_hgmu_kernel,
    hgmu_name,
    hgmu_spec,
    launch_shape,
    pair_arrays,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, Scalar, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.one_electron.overlap import overlap_matrix
from felizjueves.spec import Engine, Operator, Spec, Target

CLASSES = [(la, lb) for la in range(L_MAX + 1) for lb in range(L_MAX + 1)]
TOL = 16 * EPS  # of each component's scale: the module docstring


def _close(got, want, scale: np.ndarray, what: str) -> None:
    for c, s in enumerate(scale):
        if s == 0.0:  # every term zero by parity
            assert np.all(got[c] == 0.0) and np.all(want[c] == 0.0), what
            continue
        err = np.abs(got[c] - want[c]).max()
        assert err <= TOL * s, (
            f"{what} component {c}: {err / s / EPS:.1f} eps of scale")


def _arrays(basis, nuclei, la, lb, kern, pairs=None):
    rec = host.pair_records(basis, nuclei, la, lb, pairs=pairs,
                            ket_centre=True)
    types = {p.name: p.type for p in kern.params if p.array}
    return {k: np.array(v, dtype=np.int32 if types[k] is Scalar.I32
                        else float) for k, v in rec.items() if k in types}


def launch_hgmu(basis, nuclei, M, la, lb, G, threads=3, C=ORIGIN, q=Q):
    """One `hgmu_` launch over every shell pair of class `(la, lb)`."""
    kern = build_hgmu_kernel((la, lb))
    arrs = _arrays(basis, nuclei, la, lb, kern)
    n = len(arrs["rec0"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1)).run(
        n_pair=n, n_rec=len(arrs["s_ab"]), n_ao=basis.n_ao,
        n_atom=len(nuclei), M=M, G=G, C_x=C[0], C_y=C[1], C_z=C[2], q=q,
        **arrs)["G"]


def _classes(basis):
    return sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                   for b in basis.shells[k:]})


def _zero(nuclei) -> np.ndarray:
    return np.zeros((N_COMP, len(nuclei), 3))


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_gradient(cls) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = host.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    G = launch_hgmu(basis, nuclei, M, *cls, _zero(nuclei))
    scale = _scale(Target.DIPOLE, basis, nuclei, M, pairs)
    _close(G, _host(Target.DIPOLE, basis, nuclei, M, pairs=pairs), scale,
           f"hgmu {cls}")
    # and the MMD kernel of the same class, interpreted
    gmu = _launch_gmu(Target.DIPOLE, basis, nuclei, M, *cls, _zero(nuclei))
    _close(G, gmu, scale, f"hgmu {cls} against gmu")


@pytest.mark.parametrize("name", [
    pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
    pytest.param("co_ccpvtz_j", id="co_ccpvtz", marks=(pytest.mark.slow,))])
def test_the_whole_gradient_is_the_host_gradient(name) -> None:
    basis, nuclei = _fixture(name)
    M = _m(basis.n_ao)
    G = _zero(nuclei)
    for la, lb in _classes(basis):
        G = launch_hgmu(basis, nuclei, M, la, lb, G)
    pairs = [(i, j) for i in range(len(basis.shells))
             for j in range(i, len(basis.shells))]
    _close(G, _host(Target.DIPOLE, basis, nuclei, M),
           _scale(Target.DIPOLE, basis, nuclei, M, pairs), f"hgmu {name}")
    # the origin shift: SUM_A G[c, A, k] = q delta_ck SUM M S
    S = float(np.sum(M * np.array(overlap_matrix(basis))))
    err = np.abs(G.sum(axis=1) - Q * S * np.eye(3)).max()
    assert err <= 1e-13 * np.abs(G).max(), f"origin shift {err:.2e}"


def test_the_charge_is_an_exact_factor() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    neg = launch_hgmu(basis, nuclei, M, 1, 1, _zero(nuclei), q=-1.0)
    pos = launch_hgmu(basis, nuclei, M, 1, 1, _zero(nuclei), q=1.0)
    assert np.array_equal(neg, -pos)


def test_g_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G0 = np.arange(27.0).reshape(3, 3, 3)
    G = launch_hgmu(basis, nuclei, M, 0, 1, G0.copy())
    want = launch_hgmu(basis, nuclei, M, 0, 1, _zero(nuclei))
    assert np.abs(G - (G0 + want)).max() <= 4 * EPS * np.abs(G0).max()
    assert not np.array_equal(G, want)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    assert hgmu_spec(2, 1).key() == "dipole/d1/(2,1)/b1/dipole.hgp/literal"
    assert hgmu_name((2, 1)) == "hgmu_21"
    assert build_hgmu_kernel(hgmu_spec(2, 1)).name == "hgmu_21"
    assert len({hgmu_name(c) for c in CLASSES}) == len(CLASSES)


@pytest.mark.parametrize("spec", [
    Spec(Operator.DIPOLE, 0, (1, 0), Target.DIPOLE, engine=Engine.HGP),
    Spec(Operator.DIPOLE, 1, (1, 0), Target.DIPOLE),
    Spec(Operator.OVERLAP, 1, (1, 0), Target.S, engine=Engine.HGP),
], ids=lambda s: s.key())
def test_another_key_is_refused(spec) -> None:
    with pytest.raises(ValueError, match="not a d1 HGP DIPOLE key"):
        build_hgmu_kernel(spec)


def test_above_f_is_refused() -> None:
    with pytest.raises(ValueError, match="through f only"):
        build_hgmu_kernel((L_MAX + 1, 0))


def test_the_spec_admits_the_dipole_on_hgp_at_order_one_only() -> None:
    assert Spec(Operator.DIPOLE, 1, (1, 0), Target.DIPOLE,
                engine=Engine.HGP).key() == \
        "dipole/d1/(1,0)/b1/dipole.hgp/literal"
    with pytest.raises(ValueError,
                       match="derivative order 1 on target quadrupole.hgp"):
        Spec(Operator.QUADRUPOLE, 1, (1, 0), Target.QUADRUPOLE,
             engine=Engine.HGP)
    with pytest.raises(ValueError,
                       match="derivative order 2 on target dipole.hgp"):
        Spec(Operator.DIPOLE, 2, (1, 0), Target.DIPOLE, engine=Engine.HGP)


def test_the_parameters_are_the_staging() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    k = build_hgmu_kernel((1, 1))
    names = [p.name for p in k.params]
    assert names == [
        "n_pair", "n_rec", "n_ao", "n_atom", "zeta_ab", "pa_x", "pa_y",
        "pa_z", "pb_x", "pb_y", "pb_z", "s_ab", "ea_bra", "eb_ket",
        *pair_arrays(), "M", "C_x", "C_y", "C_z", "q", "G"]
    assert [q.name for q in k.params if q.intent is Intent.OUT] == ["G"]
    staged = set(host.pair_records(basis, nuclei, 1, 1, ket_centre=True))
    assert {p.name for p in k.params if p.array} - {"M", "G"} <= staged
    assert not {"b_x", "b_y", "b_z"} & set(
        host.pair_records(basis, nuclei, 1, 1))


@pytest.mark.parametrize("cls", [(0, 0), (1, 2), (3, 3)], ids=str)
def test_every_write_is_an_atomic_add_into_g(cls) -> None:
    k = build_hgmu_kernel(cls)
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 18 and all(a.dst.array == "G" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
