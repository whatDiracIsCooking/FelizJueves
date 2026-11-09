"""`kernel_hgd2`: the one-electron Hessians on the HGP engine (`hgd2s_`,
`hgd2t_`, `hgd2v_{la}{lb}`, `S`, `T`, `V` at d2 on `Engine.HGP`),
interpreted.

THE WITNESS is `hessian.host_1e` (held to central finite differences of
the gradient host, test/test_hessian_host_1e.py), test_kernel_h1e.py's,
which shares neither the kernel's recurrence (moved Cartesians on MMD's
`E`, not Obara & Saika's VRR and the HGP HRR) nor its contraction order
(per record, not on contracted values), nor its blocks (the host forms
every centre's directly; the kernel takes `S`'s and `T`'s `B` blocks and
`V`'s nucleus blocks by translational invariance):

- PER CLASS, every ordered pair class (ss) through (ff) on HF/cc-pVTZ, one
  launch over every shell pair `i <= j` of the class (staged by
  `gradient.host.pair_records`) into a zeroed `H`, `M` non-symmetric --
  against the host over the same pairs, within `TOL` of the host's
  `absolute=True` twin (test_kernel_h1e.py's scale: the terms' size before
  they cancel; `max|H|` is not one, a one-centre pair cancelling to
  round-off). `TOL = 16 eps`, `d2s_`'s gate. Measured (2026-11-09): every
  class within 2.02 eps of the scale (`hgd2t_30`); `S` within 0.57, `V`
  within 0.22. And AGAINST `d2s_` / `d2t_` / `d2v_` interpreted on the
  same pairs, the same bound: the two engines' kernels agree with each
  other, not only with the host.
- THE WHOLE HESSIAN of H2O/STO-3G (and CO/cc-pVTZ, f on both atoms, every
  class, `slow`), one launch per class into one `H`, the same bound, and
  `H = H^T` and `SUM_A H[A] = 0` (translational invariance of the whole).
  Measured: within 0.18 eps (H2O/STO-3G) and 0.08 eps (CO/cc-pVTZ).

And the kernel's shape: the key and the names, the refusals, the parameter
block equal to the staging, 36 atomic adds into `H` per pair (`V`: 36 + 45
per nucleus), the tables each weight reads.

`L` for the tier cut is `la + lb + 2`, test_kernel_h1e.py's. Nothing
compiles and nothing launches on a card.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_h1e import EPS, _fixture, _host, _m, _zeros
from test_kernel_h1e import _launch as _launch_mmd
from tiers import by_l

from felizjueves.gradient import host as grad
from felizjueves.hessian.kernel_h1e import ZEROED_H
from felizjueves.hessian.kernel_hgd2 import (
    ATOMIC_REASON,
    L_MAX,
    TARGETS,
    build_hgd2_kernel,
    hgd2_name,
    hgd2_spec,
    launch_shape,
    pair_arrays,
    tables,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, Scalar, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.spec import Engine, Operator, Spec, Target

CLASSES = [(la, lb) for la in range(L_MAX + 1) for lb in range(L_MAX + 1)]
TOL = 16 * EPS  # of max(scale): the module docstring


def _close(got, want, scale: float, what: str) -> None:
    if scale == 0.0:  # every term zero by parity (a one-centre s-f overlap)
        assert np.abs(got).max() == 0.0 and np.abs(want).max() == 0.0, what
        return
    err = np.abs(got - want).max()
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"


def launch_hgd2(target, basis, nuclei, M, la, lb, H, threads=3):
    """One `hgd2` launch over every shell pair of class `(la, lb)`."""
    kern = build_hgd2_kernel((target, la, lb))
    rec = grad.pair_records(basis, nuclei, la, lb)
    types = {p.name: p.type for p in kern.params if p.array}
    arrs = {k: np.array(v, dtype=np.int32 if types[k] is Scalar.I32
                        else float) for k, v in rec.items() if k in types}
    if target is Target.V:
        arrs["Z"] = np.array([z for z, _ in nuclei])
        arrs["Cnuc"] = np.array([c for _, c in nuclei])
    n = len(arrs["rec0"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1), boys=boys).run(
        n_pair=n, n_rec=len(arrs["s_ab"]), n_ao=basis.n_ao,
        n_atom=len(nuclei), M=M, H=H, **arrs)["H"]


def _classes(basis):
    return sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                   for b in basis.shells[k:]})


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_hessian(cls, target) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = grad.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    H = launch_hgd2(target, basis, nuclei, M, *cls, _zeros(nuclei))
    scale = _host(target, basis, nuclei, M, pairs, absolute=True).max()
    _close(H, _host(target, basis, nuclei, M, pairs), scale,
           f"hgd2 {target} {cls}")
    # and the MMD kernel of the same class, interpreted
    mmd = _launch_mmd(target, basis, nuclei, M, *cls, _zeros(nuclei))
    _close(H, mmd, scale, f"hgd2 {target} {cls} against d2")


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("name", [
    pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
    pytest.param("co_ccpvtz_j", id="co_ccpvtz", marks=(pytest.mark.slow,))])
def test_the_whole_hessian_is_the_host_hessian(name, target) -> None:
    basis, nuclei = _fixture(name)
    M = _m(basis.n_ao)
    H = _zeros(nuclei)
    for la, lb in _classes(basis):
        H = launch_hgd2(target, basis, nuclei, M, la, lb, H)
    _close(H, _host(target, basis, nuclei, M),
           _host(target, basis, nuclei, M, absolute=True).max(),
           f"hgd2 {target} {name}")
    scale = np.abs(H).max()
    assert np.abs(H.sum(axis=0)).max() <= 1e-10 * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= 1e-13 * scale


def test_h_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    H0 = np.arange(81.0).reshape(3, 3, 3, 3)
    for t in TARGETS:
        H = launch_hgd2(t, basis, nuclei, M, 0, 1, H0.copy())
        want = launch_hgd2(t, basis, nuclei, M, 0, 1, _zeros(nuclei))
        assert np.abs(H - (H0 + want)).max() <= 8 * EPS * np.abs(H0).max()
        assert not np.array_equal(H, want)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    assert hgd2_spec(Target.S, 2, 1).key() == \
        "overlap/d2/(2,1)/b1/s.hgp/literal"
    assert hgd2_spec(Target.T, 2, 1).key() == \
        "kinetic/d2/(2,1)/b1/t.hgp/literal"
    assert hgd2_spec(Target.V, 2, 1).key() == \
        "nuclear/d2/(2,1)/b1/v.hgp/literal"
    for t, prefix in zip(TARGETS, ("hgd2s", "hgd2t", "hgd2v"), strict=True):
        assert hgd2_name((t, 2, 1)) == f"{prefix}_21"
        assert build_hgd2_kernel(hgd2_spec(t, 2, 1)).name == f"{prefix}_21"
    names = {hgd2_name((t, *c)) for t in TARGETS for c in CLASSES}
    assert len(names) == len(TARGETS) * len(CLASSES)


@pytest.mark.parametrize("spec", [
    Spec(Operator.OVERLAP, 2, (1, 0), Target.S),
    Spec(Operator.OVERLAP, 1, (1, 0), Target.S, engine=Engine.HGP),
    Spec(Operator.KINETIC, 0, (1, 0), Target.T, engine=Engine.HGP),
    Spec(Operator.NUCLEAR, 2, (1, 0), Target.V),
], ids=lambda s: s.key())
def test_another_key_is_refused(spec) -> None:
    with pytest.raises(ValueError, match="not a d2 HGP S, T or V key"):
        build_hgd2_kernel(spec)


def test_above_f_is_refused() -> None:
    with pytest.raises(ValueError, match="through f only"):
        build_hgd2_kernel((Target.S, L_MAX + 1, 0))


def test_the_spec_admits_s_t_v_on_hgp_at_order_two() -> None:
    for t in TARGETS:
        assert hgd2_spec(t, 1, 0).engine is Engine.HGP
    with pytest.raises(ValueError, match="derivative order 2 on target v.rys"):
        Spec(Operator.NUCLEAR, 2, (1, 0), Target.V, engine=Engine.RYS)
    with pytest.raises(ValueError,
                       match="derivative order 2 on target eri3.rys"):
        Spec(Operator.COULOMB, 2, (1, 0, 0), Target.ERI3, engine=Engine.RYS)


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_parameters_are_the_staging(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    k = build_hgd2_kernel((target, 1, 1))
    names = [p.name for p in k.params]
    v = target is Target.V
    assert names == [
        "n_pair", "n_rec", "n_ao", "n_atom", "zeta_ab",
        *(["p_x", "p_y", "p_z"] if v else []),
        "pa_x", "pa_y", "pa_z", "pb_x", "pb_y", "pb_z",
        "s_ab", "ea_bra", "eb_ket", *pair_arrays(), "M",
        *(["Z", "Cnuc"] if v else []), "H"]
    out = [q for q in k.params if q.intent is Intent.OUT]
    assert [q.name for q in out] == ["H"] and len(out[0].shape) == 4
    staged = set(grad.pair_records(basis, nuclei, 1, 1))
    assert ({p.name for p in k.params if p.array}
            - {"M", "H", "Z", "Cnuc"}) <= staged


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("cls", [(0, 0), (1, 2), (3, 3)], ids=str)
def test_every_write_is_an_atomic_add_into_h(cls, target) -> None:
    k = build_hgd2_kernel((target, *cls))
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == (36 + 45 if target is Target.V else 36)
    assert all(a.dst.array == "H" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ZEROED_H in notes
    assert any(n.startswith(ATOMIC_REASON) for n in notes)
    assert not any(isinstance(s, Raw) for s in nodes)


def test_the_tables_are_the_weights_the_rule_reads() -> None:
    # S: (2 alpha)^p, p = 0, 1, 2 (no p = 0 table below a d bra: no
    # a - 1_k - 1_l)
    assert list(tables(Target.S, 0, 0)) == [(1, 0, 0), (2, 0, 0)]
    assert list(tables(Target.S, 1, 0)) == [(1, 0, 0), (2, 0, 0)]
    assert list(tables(Target.S, 2, 0)) == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    # T: (2 alpha)^p beta^n
    assert list(tables(Target.T, 0, 0)) == [(1, 0, 1), (1, 0, 2),
                                            (2, 0, 1), (2, 0, 2)]
    assert len(tables(Target.T, 1, 2)) == 6 and len(tables(Target.T, 2, 2)) == 9
    # V: (2 alpha)^p (2 beta)^q, p + q <= 2
    assert list(tables(Target.V, 1, 1)) == [
        (0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0)]
    with pytest.raises(ValueError):
        tables(Target.DIPOLE, 0, 0)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
