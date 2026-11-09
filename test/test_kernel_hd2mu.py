"""`kernel_hd2mu`: the dipole and quadrupole Hessians on the HGP engine
(`hd2mu_` / `hd2quad_{la}{lb}`, `Target.DIPOLE` / `QUADRUPOLE` at d2 on
`Engine.HGP`), interpreted.

THE WITNESS is `hessian.host_1e.multipole_hessian`, test_kernel_d2mu.py's
(itself held to central differences of the libcint-witnessed d1 host and
to the origin shift), which shares neither the kernel's recurrence
(McMurchie & Davidson's `E` and Hermite moments, not Obara & Saika's VRR,
the HGP HRR and the operator moved onto the ket) nor its contraction order
(per record, not on contracted values):

- PER CLASS, every ordered pair class (ss) through (ff) on HF/cc-pVTZ, one
  launch over every shell pair `i <= j` of the class (staged by
  `gradient.host.pair_records(..., ket_centre=True)`) into a zeroed `H`,
  `M` non-symmetric, `C` off every nucleus, `q = -1` -- against the host
  over the same pairs, within `TOL = 16 eps` of the host's `absolute=True`
  twin per component (test_kernel_d2mu.py's scale and bound). Measured
  (2026-11-09): worst 6.0 eps at the quadrupole's (fd), 4.6 at its (ps);
  the dipole's worst 3.3 at (ds).
- AGAINST `d2mu_` / `d2quad_`: the same classes interpreted by
  `kernel_d2mu`, the same bound -- the two engines' kernels agree with each
  other, not only with the host.
- THE WHOLE HESSIAN of H2O/STO-3G (and CO/cc-pVTZ, `slow`), one launch per
  class into one `H`, the same bound, its symmetry `H[c, A, k, B, l] =
  H[c, B, l, A, k]`, and the origin shift: `SUM_B H^c[A, k, B, l]` is the
  d1 host's (test_kernel_d2mu.py's identity) -- the kernel computes every
  block, so it is a check, not an assumption.

And the kernel's shape: the key and names, the refusals, the parameter
block equal to the staging, `36 n_comp` atomic adds into `H` per shell
pair, the charge an exact factor, the derivative rule's terms.

`L` for the tier cut is `la + lb + 2`, test_kernel_d2mu.py's. Nothing
compiles and nothing launches on a card.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_d2mu import EPS, ORIGIN, QUAD, Q, _close, _fixture, _host, _m
from test_kernel_d2mu import _launch as _launch_d2mu
from tiers import by_l

from felizjueves.gradient import host as grad
from felizjueves.hessian.kernel_hd2mu import (
    ATOMIC_REASON,
    HESSIAN_MOMENTS,
    L_MAX,
    ZEROED_H,
    build_hd2mu_kernel,
    hd2mu_name,
    hd2mu_spec,
    launch_shape,
    n_components,
    pair_arrays,
    second,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, Scalar, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.spec import Engine, Operator, Spec, Target

TARGETS = HESSIAN_MOMENTS
CLASSES = [(la, lb) for la in range(L_MAX + 1) for lb in range(L_MAX + 1)]


def _zeros(target, nuclei) -> np.ndarray:
    n = len(nuclei)
    return np.zeros((n_components(target), n, 3, n, 3))


def _arrays(basis, nuclei, la, lb, kern):
    rec = grad.pair_records(basis, nuclei, la, lb, ket_centre=True)
    types = {p.name: p.type for p in kern.params if p.array}
    return {k: np.array(v, dtype=np.int32 if types[k] is Scalar.I32
                        else float) for k, v in rec.items() if k in types}


def launch_hd2mu(target, basis, nuclei, M, la, lb, H, threads=3, q=Q):
    """One `hd2mu_` / `hd2quad_` launch over every shell pair of class
    `(la, lb)`."""
    kern = build_hd2mu_kernel((target, la, lb))
    arrs = _arrays(basis, nuclei, la, lb, kern)
    n = len(arrs["rec0"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1)).run(
        n_pair=n, n_rec=len(arrs["s_ab"]), n_ao=basis.n_ao,
        n_atom=len(nuclei), M=M, H=H, C_x=ORIGIN[0], C_y=ORIGIN[1],
        C_z=ORIGIN[2], q=q, **arrs)["H"]


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
    H = launch_hd2mu(target, basis, nuclei, M, *cls, _zeros(target, nuclei))
    scale = _host(target, basis, nuclei, M, pairs, absolute=True)
    _close(H, _host(target, basis, nuclei, M, pairs), scale,
           f"hgp {target} {cls}")
    # and the MMD kernel of the same class, interpreted
    mmd = _launch_d2mu(target, basis, nuclei, M, *cls,
                       _zeros(target, nuclei))
    _close(H, mmd, scale, f"hgp {target} {cls} against d2mu")


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("name", [
    pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
    pytest.param("co_ccpvtz_j", id="co_ccpvtz", marks=(pytest.mark.slow,))])
def test_the_whole_hessian_is_the_host_hessian(name, target) -> None:
    basis, nuclei = _fixture(name)
    M = _m(basis.n_ao)
    H = _zeros(target, nuclei)
    for la, lb in _classes(basis):
        H = launch_hd2mu(target, basis, nuclei, M, la, lb, H)
    _close(H, _host(target, basis, nuclei, M),
           _host(target, basis, nuclei, M, absolute=True),
           f"hgp {target} {name}")
    scale = np.abs(H).max()
    assert np.abs(H - H.transpose(0, 3, 4, 1, 2)).max() <= 1e-13 * scale
    # the origin shift (test_kernel_d2mu.py's identity), from the d1 hosts
    got = H.sum(axis=3)  # [comp, A, k, l]
    want = np.zeros_like(got)
    if target is Target.DIPOLE:
        gs = np.array(grad.overlap_gradient(basis, M, nuclei))
        for c in range(3):
            want[c, :, :, c] = Q * gs
    else:
        gmu = np.array(grad.multipole_gradient(basis, M, nuclei, ORIGIN,
                                               e=1, q=1.0))
        for c, (a, b) in enumerate(QUAD):
            want[c, :, :, a] += Q * gmu[b]
            want[c, :, :, b] += Q * gmu[a]
    err = np.abs(got - want).max()
    assert err <= 1e-12 * scale, f"origin shift: {err:.2e}"


def test_h_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    H0 = np.arange(243.0).reshape(3, 3, 3, 3, 3)
    H = launch_hd2mu(Target.DIPOLE, basis, nuclei, M, 0, 1, H0.copy())
    want = launch_hd2mu(Target.DIPOLE, basis, nuclei, M, 0, 1,
                        _zeros(Target.DIPOLE, nuclei))
    assert np.abs(H - (H0 + want)).max() <= 8 * EPS * np.abs(H0).max()
    assert not np.array_equal(H, want)


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_charge_is_an_exact_factor(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    up = launch_hd2mu(target, basis, nuclei, M, 1, 1,
                      _zeros(target, nuclei), q=1.0)
    dn = launch_hd2mu(target, basis, nuclei, M, 1, 1,
                      _zeros(target, nuclei), q=-1.0)
    assert np.array_equal(dn, -up) and np.abs(up).max() > 0.0


# --- the derivative rule ----------------------------------------------------


def test_the_second_derivative_terms() -> None:
    p, s = (1, 0, 0), (0, 0, 0)
    # d2/dA_x dA_x of (p_x|s): 4a^2 (f_xxx|s) - 2a (2 + 1) (p_x|s)
    assert second("A", 0, "A", 0, p, s) == [
        (1, 3, (3, 0, 0), s), (-3, 1, p, s)]
    # d2/dA_x dB_y of (p_x|s): 4ab (d_xx|p_y) - 2b (s|p_y)
    assert second("A", 0, "B", 1, p, s) == [
        (1, 5, (2, 0, 0), (0, 1, 0)), (-1, 2, s, (0, 1, 0))]
    # d2/dB_y dB_y of (s|p_y): 4b^2 (s|f_yyy) - 2b 3 (s|p_y)
    assert second("B", 1, "B", 1, s, (0, 1, 0)) == [
        (1, 4, s, (0, 3, 0)), (-3, 2, s, (0, 1, 0))]
    # the s-s AB block reads no unweighted table
    assert {t for _, t, _, _ in second("A", 2, "B", 2, s, s)} == {5}


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    for t, prefix in zip(TARGETS, ("hd2mu", "hd2quad"), strict=True):
        assert hd2mu_spec(t, 2, 1).key() == (
            f"{t.value}/d2/(2,1)/b1/{t.value}.hgp/literal")
        assert hd2mu_name((t, 2, 1)) == f"{prefix}_21"
        assert build_hd2mu_kernel(hd2mu_spec(t, 2, 1)).name == f"{prefix}_21"
    assert len({hd2mu_name((t, *c)) for t in TARGETS for c in CLASSES}) == \
        2 * len(CLASSES)


@pytest.mark.parametrize("spec", [
    Spec(Operator.DIPOLE, 1, (1, 0), Target.DIPOLE, engine=Engine.HGP),
    Spec(Operator.QUADRUPOLE, 0, (1, 0), Target.QUADRUPOLE,
         engine=Engine.HGP),
    Spec(Operator.DIPOLE, 2, (1, 0), Target.DIPOLE),
    Spec(Operator.OVERLAP, 1, (1, 0), Target.S, engine=Engine.HGP),
], ids=lambda s: s.key())
def test_another_key_is_refused(spec) -> None:
    with pytest.raises(ValueError,
                       match="not a d2 HGP dipole or quadrupole key"):
        build_hd2mu_kernel(spec)


def test_above_f_is_refused() -> None:
    with pytest.raises(ValueError, match="through f only"):
        build_hd2mu_kernel((Target.DIPOLE, L_MAX + 1, 0))


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_parameters_are_the_staging(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    k = build_hd2mu_kernel((target, 1, 1))
    names = [p.name for p in k.params]
    assert names == [
        "n_pair", "n_rec", "n_ao", "n_atom", "zeta_ab", "pa_x", "pa_y",
        "pa_z", "pb_x", "pb_y", "pb_z", "s_ab", "ea_bra", "eb_ket",
        *pair_arrays(), "M", "C_x", "C_y", "C_z", "q", "H"]
    out = [q for q in k.params if q.intent is Intent.OUT]
    assert [q.name for q in out] == ["H"] and len(out[0].shape) == 5
    assert out[0].shape[0].value == n_components(target)
    staged = set(grad.pair_records(basis, nuclei, 1, 1, ket_centre=True))
    assert {p.name for p in k.params if p.array} - {"M", "H"} <= staged


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("cls", [(0, 0), (1, 2), (3, 3)], ids=str)
def test_every_write_is_an_atomic_add_into_h(cls, target) -> None:
    k = build_hd2mu_kernel((target, *cls))
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 36 * n_components(target)
    assert all(a.dst.array == "H" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_H in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
