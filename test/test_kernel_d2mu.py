"""`kernel_d2mu`: the dipole and quadrupole hessian kernels (d2), interpreted,
and their host reference, `hessian.host_1e.multipole_hessian`.

THE HOST is held to
- central finite differences of `gradient.host.multipole_gradient` (itself
  held to libcint's `int1e_irp` / `int1e_irrp`, test/test_kernel_gmu.py),
  every atom moved with its shells, on H2O/STO-3G (`O(h^2)`, ~1e-8 of
  `max|H|`);
- the origin shift: `C` is not a nucleus, so moving every atom by `h` on
  axis `l` is moving `C` by `-h`: `SUM_B H^c[A, k, B, l] = q delta_cl
  G_S[A, k]` for the dipole (`G_S` the overlap gradient), and `q (delta_cl
  G_MU^d[A, k] + delta_dl G_MU^c[A, k])` for the quadrupole's `cd` (`G_MU`
  the dipole gradient at `q = 1`) -- both d1 hosts libcint-witnessed;
- the charge as an exact factor.

THE KERNEL is held to the host per class, one launch over every record of
every shell pair `i <= j` of that class into a zeroed `H`, every ordered
class (ss) through (ff) on HF/cc-pVTZ, `M` non-symmetric, within `TOL = 16
eps` of `scale` per component, `scale` the host's `absolute=True` twin
(every factor in `|.|`, test/test_kernel_h1e.py's): `max|H|` is not a
scale, a one-centre pair's blocks cancel on one atom. And the whole
H2O/STO-3G Hessian, one launch per class.

`L` for the tier cut is `la + lb + 2`, the Hermite order a hessian
reaches, as for `d2s` (test/tiers.py).
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves.ao.shell import Basis
from felizjueves.gradient import host as grad
from felizjueves.gradient.kernel_gmu import ORDER
from felizjueves.hessian import host_1e as host
from felizjueves.hessian.kernel_d2mu import (
    ATOMIC_REASON,
    HESSIAN_MOMENTS,
    ZEROED_H,
    build_d2mu_kernel,
    d2mu_name,
    d2mu_spec,
    launch_shape,
    n_components,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(scale): the module docstring
TARGETS = HESSIAN_MOMENTS
LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
ORIGIN = (0.3, -0.2, 0.5)
Q = -1.0  # an electron's charge: q is an exact factor of every value
QUAD = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]  # cartesians(2)


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


def _m(n: int) -> np.ndarray:
    return np.random.default_rng(7).standard_normal((n, n))


def _host(target, basis, nuclei, M, pairs=None, absolute=False, q=Q
          ) -> np.ndarray:
    return np.array(host.multipole_hessian(
        basis, M, nuclei, ORIGIN, e=ORDER[target], q=q, pairs=pairs,
        absolute=absolute))


def _zeros(target, nuclei) -> np.ndarray:
    n = len(nuclei)
    return np.zeros((n_components(target), n, 3, n, 3))


def _launch(target, basis, nuclei, M, la, lb, H, threads=7):
    rec = grad.records(basis, nuclei, la, lb)
    kern = build_d2mu_kernel((target, la, lb))
    want = {p.name for p in kern.params if p.array}
    arrs = {k: np.array(v, dtype=np.int32 if k.startswith(("off", "atom"))
                        else float) for k, v in rec.items() if k in want}
    n = len(rec["s_ab"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1)).run(
        n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), M=M, H=H,
        C_x=ORIGIN[0], C_y=ORIGIN[1], C_z=ORIGIN[2], q=Q, **arrs)["H"]


def _close(got, want, scale: np.ndarray, what: str) -> None:
    for c in range(len(scale)):
        s = scale[c].max()
        if s == 0.0:  # every term zero by parity
            assert np.all(got[c] == 0.0) and np.all(want[c] == 0.0), what
            continue
        err = np.abs(got[c] - want[c]).max()
        assert err <= TOL * s, (
            f"{what} component {c}: {err / s / EPS:.1f} eps of scale")


# --- the host ----------------------------------------------------------------


def _moved(basis: Basis, nuclei, A: int, k: int, h: float):
    atoms = grad.shell_atoms(basis, nuclei)
    shells = []
    for s, at in zip(basis.shells, atoms, strict=True):
        if at == A:
            c = list(s.centre)
            c[k] += h
            s = replace(s, centre=tuple(c))
        shells.append(s)
    nuc = tuple((z, tuple(x + h * (j == A and d == k)
                          for d, x in enumerate(c)))
                for j, (z, c) in enumerate(nuclei))
    return Basis(tuple(shells)), nuc


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_host_is_finite_differences_of_the_gradient(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    e, h = ORDER[target], 1e-4
    H = _host(target, basis, nuclei, M)
    fd = np.zeros_like(H)
    for B in range(len(nuclei)):
        for l in range(3):  # noqa: E741
            up, dn = (np.array(grad.multipole_gradient(
                b, M, n, ORIGIN, e=e, q=Q))
                for b, n in (_moved(basis, nuclei, B, l, s)
                             for s in (h, -h)))
            fd[:, :, :, B, l] = (up - dn) / (2 * h)
    err = np.abs(H - fd).max() / np.abs(H).max()
    assert err <= 1e-7, f"{target}: {err:.2e} of max|H|"


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_atoms_sum_to_the_origin_shift(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    H = _host(target, basis, nuclei, M).sum(axis=3)  # [comp, A, k, l]
    want = np.zeros_like(H)
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
    assert np.abs(H - want).max() <= 1e-13 * np.abs(H).max()


def test_the_charge_is_an_exact_factor() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    for t in TARGETS:
        assert np.array_equal(_host(t, basis, nuclei, M, q=-1.0),
                              -_host(t, basis, nuclei, M, q=1.0))


def test_the_host_refuses_a_bad_order_or_origin() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    with pytest.raises(ValueError, match="is not an int"):
        host.multipole_hessian(basis, M, nuclei, ORIGIN, e=-1)
    with pytest.raises(ValueError, match="three finite floats"):
        host.multipole_hessian(basis, M, nuclei, (0.0, float("nan"), 0.0))


# --- the kernel, interpreted ------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_hessian(cls, target) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = grad.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    H = _launch(target, basis, nuclei, M, *cls, _zeros(target, nuclei))
    _close(H, _host(target, basis, nuclei, M, pairs),
           _host(target, basis, nuclei, M, pairs, absolute=True),
           f"{target} {cls}")


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_whole_hessian_of_h2o_is_the_host_hessian(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    H = _zeros(target, nuclei)
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        H = _launch(target, basis, nuclei, M, la, lb, H)
    _close(H, _host(target, basis, nuclei, M),
           _host(target, basis, nuclei, M, absolute=True),
           f"{target} H2O/STO-3G")
    scale = np.abs(H).max()
    assert np.abs(H - H.transpose(0, 3, 4, 1, 2)).max() <= 1e-13 * scale


def test_h_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    H0 = np.arange(243.0).reshape(3, 3, 3, 3, 3)
    H = _launch(Target.DIPOLE, basis, nuclei, M, 0, 1, H0.copy())
    want = _launch(Target.DIPOLE, basis, nuclei, M, 0, 1,
                   _zeros(Target.DIPOLE, nuclei))
    assert np.abs(H - (H0 + want)).max() <= 8 * EPS * np.abs(H0).max()
    assert not np.array_equal(H, want)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    for t, prefix in zip(TARGETS, ("d2mu", "d2quad"), strict=True):
        key = d2mu_spec(t, 2, 1).key()
        assert key == f"{t.value}/d2/(2,1)/b1/{t.value}/literal"
        assert d2mu_name((t, 2, 1)) == f"{prefix}_21"
        assert build_d2mu_kernel(d2mu_spec(t, 2, 1)).name == f"{prefix}_21"


def test_a_d0_or_d1_key_another_target_an_hgp_key_or_a_lever_is_refused(
        ) -> None:
    for order in (0, 1):
        with pytest.raises(ValueError, match="not a d2 dipole or quadrupole"):
            build_d2mu_kernel(Spec(Operator.DIPOLE, order, (1, 0),
                                   Target.DIPOLE))
    with pytest.raises(ValueError, match="not a d2 dipole or quadrupole"):
        build_d2mu_kernel(Spec(Operator.OVERLAP, 2, (1, 0), Target.S))
    # the HGP moment's Hessian is kernel_hd2mu's, not this builder's
    with pytest.raises(ValueError, match="not an MMD key"):
        build_d2mu_kernel(Spec(Operator.DIPOLE, 2, (1, 0), Target.DIPOLE,
                               engine=Engine.HGP))
    with pytest.raises(ValueError, match="take no levers"):
        build_d2mu_kernel(Spec(Operator.DIPOLE, 2, (1, 0), Target.DIPOLE,
                               levers=("tuned",)))


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_parameters_and_the_one_output(target) -> None:
    k = build_d2mu_kernel((target, 1, 1))
    names = [p.name for p in k.params]
    assert names == [
        "n_rec", "n_ao", "n_atom", "zeta_ab", "pa_x", "pa_y", "pa_z",
        "pb_x", "pb_y", "pb_z", "p_x", "p_y", "p_z", "s_ab", "ea_bra",
        "eb_ket", "off_a", "off_b", "atom_a", "atom_b", "M",
        "C_x", "C_y", "C_z", "q", "H"]
    out = [q for q in k.params if q.intent is Intent.OUT]
    assert [q.name for q in out] == ["H"] and len(out[0].shape) == 5
    assert out[0].shape[0].value == n_components(target)


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_every_write_is_an_atomic_add_into_h(target) -> None:
    k = build_d2mu_kernel((target, 1, 2))
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert adds and all(a.dst.array == "H" for a in adds)
    assert len(adds) == 36 * n_components(target)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_H in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
