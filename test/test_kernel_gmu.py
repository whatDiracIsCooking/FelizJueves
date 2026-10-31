"""`kernel_gmu`: the dipole and quadrupole gradient kernels (d1), interpreted,
and their host reference, `gradient.host.multipole_gradient`.

THE HOST is held to
- libcint, live (`oracle`: PySCF): `int1e_irp_cart` (the dipole) and
  `int1e_irrp_cart` (the quadrupole, its xx xy xz yy yz zz of nine), `<a|
  (r - C) nabla |b>` about a common origin, contracted with a non-symmetric
  `M` per atom -- `dMU_mn/dR_{A,k} = -irp[k]_mn [n on A] - irp[k]_nm [m on
  A]` -- on H2O/cc-pVTZ (s through f), within 1e-12 of `max|G|` (measured
  2.0e-15 / 3.0e-15);
- central finite differences of `one_electron.multipole.multipole_matrix`,
  every atom moved with its shells, on H2O/STO-3G (`O(h^2)`, ~1e-9);
- the origin shift: `C` is not a nucleus, so moving every atom by `h` on
  axis `k` is moving `C` by `-h`: `SUM_A G^c[A, k] = q delta_ck SUM M S`
  for the dipole, and `q (delta_ck SUM M MU^d + delta_dk SUM M MU^c)` for
  the quadrupole's `cd` -- the libcint-witnessed `S` and `MU`.

THE KERNEL is held to the host per class, one launch over every record of
every shell pair `i <= j` of that class into a zeroed `G`, every ordered
class (ss) through (ff) on HF/cc-pVTZ, `M` non-symmetric, within `TOL = 16
eps` of `scale`: `test_kernel_g1e.py`'s, per component -- the sum over the
class's pairs of `|G|` of each pair alone with every shell on a row of its
own, folded back onto the atoms (a one-centre pair's `dA` and `dB` cancel
on one atom). And the whole H2O/STO-3G gradient, one launch per class.

`L` for the tier cut is `la + lb + 1`, as for `gs` (test/tiers.py).
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
from felizjueves.gradient import host
from felizjueves.gradient.kernel_gmu import (
    ATOMIC_REASON,
    GRADIENT_MOMENTS,
    ORDER,
    ZEROED_G,
    build_gmu_kernel,
    gmu_name,
    gmu_spec,
    launch_shape,
    n_components,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.one_electron.multipole import multipole_matrix
from felizjueves.one_electron.overlap import overlap_matrix
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(scale): the module docstring
LIBCINT_TOL = 1e-12
TARGETS = GRADIENT_MOMENTS
LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
ORIGIN = (0.3, -0.2, 0.5)
Q = -1.0  # an electron's charge: q is an exact factor of every value


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


def _m(n: int) -> np.ndarray:
    return np.random.default_rng(5).standard_normal((n, n))


def _host(target, basis, nuclei, M, pairs=None, atoms=None, C=ORIGIN,
          q=Q) -> np.ndarray:
    return np.array(host.multipole_gradient(
        basis, M, nuclei, C, e=ORDER[target], q=q, pairs=pairs, atoms=atoms))


def _launch(target, basis, nuclei, M, la, lb, G, threads=7):
    rec = host.records(basis, nuclei, la, lb)
    kern = build_gmu_kernel((target, la, lb))
    want = {p.name for p in kern.params if p.array}
    arrs = {k: np.array(v, dtype=np.int32 if k.startswith(("off", "atom"))
                        else float) for k, v in rec.items() if k in want}
    n = len(rec["s_ab"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1)).run(
        n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), M=M, G=G,
        C_x=ORIGIN[0], C_y=ORIGIN[1], C_z=ORIGIN[2], q=Q, **arrs)["G"]


def _scale(target, basis, nuclei, M, pairs) -> np.ndarray:
    """The module docstring's `scale`, its largest element per component."""
    nat = len(nuclei)
    real = host.shell_atoms(basis, nuclei)
    split = tuple(nat + i for i in range(len(basis.shells)))
    sc = np.zeros((n_components(target), nat, 3))
    for p in pairs:
        g = np.abs(_host(target, basis, nuclei, M, [p], atoms=split))
        sc += g[:, :nat]
        for i, row in enumerate(split):
            sc[:, real[i]] += g[:, row]
    return sc.reshape(len(sc), -1).max(axis=1)


def _close(got, want, scale: np.ndarray, what: str) -> None:
    for c, s in enumerate(scale):
        if s == 0.0:  # every term zero by parity
            assert np.all(got[c] == 0.0) and np.all(want[c] == 0.0), what
            continue
        err = np.abs(got[c] - want[c]).max()
        assert err <= TOL * s, (
            f"{what} component {c}: {err / s / EPS:.1f} eps of scale")


# --- the host ----------------------------------------------------------------


def _moved(basis: Basis, nuclei, A: int, k: int, h: float) -> Basis:
    atoms = host.shell_atoms(basis, nuclei)
    shells = []
    for s, at in zip(basis.shells, atoms, strict=True):
        if at == A:
            c = list(s.centre)
            c[k] += h
            s = replace(s, centre=tuple(c))
        shells.append(s)
    return Basis(tuple(shells))


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_host_is_finite_differences_of_the_moments(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    e, h = ORDER[target], 1e-4
    G = _host(target, basis, nuclei, M)
    fd = np.zeros_like(G)
    for A in range(len(nuclei)):
        for k in range(3):
            up, dn = (np.array(multipole_matrix(
                _moved(basis, nuclei, A, k, s), ORIGIN, e, Q))
                for s in (h, -h))
            fd[:, A, k] = np.einsum("mn,cmn->c", M, (up - dn) / (2 * h))
    err = np.abs(G - fd).max() / np.abs(G).max()
    assert err <= 1e-8, f"{target}: {err:.2e} of max|G|"


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_atoms_sum_to_the_origin_shift(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G = _host(target, basis, nuclei, M).sum(axis=1)  # [comp, k]
    S = float(np.sum(M * np.array(overlap_matrix(basis))))
    mu = np.einsum("mn,cmn->c", M,
                   np.array(multipole_matrix(basis, ORIGIN, 1, 1.0)))
    want = np.zeros_like(G)
    if target is Target.DIPOLE:
        want[:, :] = Q * S * np.eye(3)
    else:
        comps = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        for c, (a, b) in enumerate(comps):
            want[c, a] += Q * mu[b]
            want[c, b] += Q * mu[a]
    assert np.abs(G - want).max() <= 1e-13 * np.abs(G).max()


def test_the_charge_is_an_exact_factor() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    for t in TARGETS:
        assert np.array_equal(_host(t, basis, nuclei, M, q=-1.0),
                              -_host(t, basis, nuclei, M, q=1.0))


H2O = "O 0 0 0; H 1.43 0.21 -1.07; H -1.38 0.35 -1.12"


@pytest.mark.oracle
@pytest.mark.slow  # the host over H2O/cc-pVTZ, ~20 s per target
@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_host_is_libcints_irp_on_h2o_ccpvtz(target) -> None:
    gto = pytest.importorskip("pyscf.gto")
    mol = gto.M(atom=H2O, unit="Bohr", basis="cc-pvtz", cart=True)
    assert max(mol.bas_angular(i) for i in range(mol.nbas)) == 3
    basis = ob.from_mole(mol)
    nuclei = [(float(mol.atom_charge(i)), tuple(mol.atom_coord(i)))
              for i in range(mol.natm)]
    name, keep = {Target.DIPOLE: ("int1e_irp_cart", [0, 1, 2]),
                  Target.QUADRUPOLE: ("int1e_irrp_cart",
                                      [0, 1, 2, 4, 5, 8])}[target]
    with mol.with_common_orig(ORIGIN):
        irp = mol.intor(name).reshape(-1, 3, mol.nao, mol.nao)[keep]
    M = _m(mol.nao)
    want = np.zeros((len(keep), mol.natm, 3))
    for A, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        for c in range(len(keep)):
            for k in range(3):
                d = np.zeros((mol.nao, mol.nao))
                d[:, p0:p1] = -irp[c, k][:, p0:p1]
                want[c, A, k] = Q * np.sum(M * (d + d.T))
    got = _host(target, basis, nuclei, M)
    err = np.abs(got - want).max() / np.abs(want).max()
    print(f"\n{target} host against libcint: {err:.2e} of max|G|")
    assert err <= LIBCINT_TOL


# --- the kernel, interpreted ------------------------------------------------------


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_gradient(cls, target) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = host.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    G = _launch(target, basis, nuclei, M, *cls,
                np.zeros((n_components(target), len(nuclei), 3)))
    _close(G, _host(target, basis, nuclei, M, pairs),
           _scale(target, basis, nuclei, M, pairs), f"{target} {cls}")


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_whole_gradient_of_h2o_is_the_host_gradient(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G = np.zeros((n_components(target), len(nuclei), 3))
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        G = _launch(target, basis, nuclei, M, la, lb, G)
    pairs = [(i, j) for i in range(len(basis.shells))
             for j in range(i, len(basis.shells))]
    _close(G, _host(target, basis, nuclei, M),
           _scale(target, basis, nuclei, M, pairs), f"{target} H2O/STO-3G")


def test_g_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G0 = np.arange(27.0).reshape(3, 3, 3)
    G = _launch(Target.DIPOLE, basis, nuclei, M, 0, 1, G0.copy())
    want = _launch(Target.DIPOLE, basis, nuclei, M, 0, 1, np.zeros((3, 3, 3)))
    assert np.abs(G - (G0 + want)).max() <= 4 * EPS * np.abs(G0).max()
    assert not np.array_equal(G, want)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    for t, prefix in zip(TARGETS, ("gmu", "gquad"), strict=True):
        key = gmu_spec(t, 2, 1).key()
        assert key == f"{t.value}/d1/(2,1)/b1/{t.value}/literal"
        assert gmu_name((t, 2, 1)) == f"{prefix}_21"
        assert build_gmu_kernel(gmu_spec(t, 2, 1)).name == f"{prefix}_21"


def test_a_d0_key_another_target_an_hgp_key_or_a_lever_is_refused() -> None:
    with pytest.raises(ValueError, match="not a d1 dipole or quadrupole"):
        build_gmu_kernel(Spec(Operator.DIPOLE, 0, (1, 0), Target.DIPOLE))
    with pytest.raises(ValueError, match="not a d1 dipole or quadrupole"):
        build_gmu_kernel(Spec(Operator.OVERLAP, 1, (1, 0), Target.S))
    # the HGP moment has no gradient: the key itself is refused
    with pytest.raises(ValueError, match="first derivative"):
        Spec(Operator.DIPOLE, 1, (1, 0), Target.DIPOLE, engine=Engine.HGP)
    with pytest.raises(ValueError, match="take no levers"):
        build_gmu_kernel(Spec(Operator.DIPOLE, 1, (1, 0), Target.DIPOLE,
                              levers=("tuned",)))


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_parameters_and_the_one_output(target) -> None:
    k = build_gmu_kernel((target, 1, 1))
    names = [p.name for p in k.params]
    assert names == [
        "n_rec", "n_ao", "n_atom", "zeta_ab", "pa_x", "pa_y", "pa_z",
        "pb_x", "pb_y", "pb_z", "p_x", "p_y", "p_z", "s_ab", "ea_bra",
        "eb_ket", "off_a", "off_b", "atom_a", "atom_b", "M",
        "C_x", "C_y", "C_z", "q", "G"]
    out = [q for q in k.params if q.intent is Intent.OUT]
    assert [q.name for q in out] == ["G"]
    assert out[0].shape[0].value == n_components(target)


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_every_write_is_an_atomic_add_into_g(target) -> None:
    k = build_gmu_kernel((target, 1, 2))
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert adds and all(a.dst.array == "G" for a in adds)
    assert len(adds) == 6 * n_components(target)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
