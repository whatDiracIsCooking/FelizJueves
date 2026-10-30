"""`kernel_xeri3`: the UNCONTRACTED three-center d1 kernel (#692),
interpreted, and its host reference `gradient.host_df.eri3_ip1`.

THE QUANTITY. `X[k, m, n, P] = (nabla_k m n|P)`, libcint's `int3c2e_ip1`,
over one aux block; the aux centre is the caller's, `(mn|nabla_k P) =
-X[k, m, n, P] - X[k, n, m, P]` (`int3c2e_ip2`).

THE WITNESSES, three:

- the host, `host_df.eri3_ip1` (the same Cartesian rule on the AO side,
  `reference.pq` against the plan's MILD DAG): every class `la <= lb`,
  `la + lb + lP <= 4` on test_kernel_geri3.py's bases (swapped, diagonal and
  off-diagonal pairs, contracted and not), one launch per class, within
  `TOL = 16 eps` of the class's `max|X|`; every class of FreeBreakfast's
  ladder above it (`la <= lb <= 3`, `lP <= 6`, test_kernel_df_grad_ladder.py's
  one-primitive bases) within `TOL` of the TOWER-TWIN scale (`max|X|` of the
  host run on `test_kernel_metric.pq_tower`, #606's yardstick for the
  tower's own cancellation at high L);
- the contracted d1 host, `host_df.eri3_gradient`, which takes ANOTHER ROUTE
  on the aux centre (the Cartesian rule on its single-center `E`): `X`
  contracted with a random `Gam` onto the atoms -- `-X` on `m`'s, `-X^T` on
  `n`'s, `X + X^T` on the aux shell's (the identity) -- is that gradient;
- libcint: the host against `int3c2e_ip1_cart` and the identity against
  `int3c2e_ip2_cart` element by element, live (`oracle`: PySCF), on
  H2O/cc-pVDZ with cc-pVDZ-JKFIT and on the aux-i witness's system
  (`prim_spd_f` with AUX_I: orbital f, aux i, every ladder class), within
  `LIBCINT_TOL = 1e-13` of `max|ip1|` (measured 3.5e-14 and 2.2e-14); and
  without PySCF, through the tracked aux-i witness
  (`oracle.manifest.df_gradient`, libcint's ip1 / ip2 contracted per class),
  the host contracted per class (`slow`, ~1 min).

THE BLOCK. A launch over a strict subset of the cells of a strict subset of
the aux shells writes those cells' slots of the block's columns and leaves
every other entry of `X` as it was passed in.

`L` for the tier cut is `la + lb + lP + 1`, the Hermite order the
derivative reaches: up to 3 fast, 4 and above `slow`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from df_grad_launch import run_xeri3
from oracle import basis as ob
from oracle import manifest
from test_kernel_geri3 import AUX, BASIS, NUCLEI
from test_kernel_metric import pq_tower
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.gradient import host_df
from felizjueves.gradient.host import shell_atoms
from felizjueves.gradient.kernel_geri3 import build_geri3_kernel, geri3_spec
from felizjueves.gradient.kernel_xeri3 import (
    build_xeri3_kernel,
    xeri3_name,
    xeri3_spec,
)
from felizjueves.ir import AtomicAdd, Intent, Raw, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of the class's max|X| (or its tower-twin scale)
LIBCINT_TOL = 1e-13  # of max|ip1|
L_CAP = 4
CLASSES = [(a, b, p) for a in range(L_CAP + 1) for b in range(a, L_CAP + 1)
           for p in range(L_CAP + 1) if a + b + p <= L_CAP]
LADDER = [(a, b, p) for a in range(4) for b in range(a, 4) for p in range(7)]


def _id(c) -> str:
    return xeri3_name(c)


def _cells(basis, aux, cls) -> list[tuple[int, int, int]]:
    sh, ash = basis.shells, aux.shells
    return [(i, j, n) for i in range(len(sh)) for j in range(i, len(sh))
            for n in range(len(ash))
            if (*sorted((sh[i].l, sh[j].l)), ash[n].l) == cls]


def _host(basis, aux, cells, *, tower=pq) -> np.ndarray:
    return np.array(host_df.eri3_ip1(basis, aux, pq=tower, cells=cells))


def _close(got, want, scale: float, what: str) -> None:
    err = float(np.abs(got - want).max())
    print(f"\n{what}: {err / scale / EPS:.2f} eps of scale")
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"


# --- the host, every class ------------------------------------------------------


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c))
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_tensor(cls) -> None:
    X, ran = run_xeri3(BASIS, AUX, NUCLEI, boys=boys, classes={cls})
    assert ran == (cls,)
    want = _host(BASIS, AUX, _cells(BASIS, AUX, cls))
    assert np.abs(want).max() > 0
    _close(X, want, float(np.abs(want).max()), xeri3_name(cls))


A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)


def _ladder_bases(cls):
    """test_kernel_df_grad_ladder.py's: `la` at A, `lb` at B, the higher
    listed first (so `la < lb` is staged swapped; `la == lb` has a diagonal
    and an off-diagonal pair), the aux shell at C, one primitive each."""
    la, lb, lP = cls
    sh = [Shell(lb, B, (1.1,), (1.0,)), Shell(la, A, (0.8,), (1.0,))]
    return Basis(tuple(sh)), Basis((Shell(lP, C, (1.6,), (1.0,)),))


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c))
                                 for c in LADDER if sum(c) >= 5])
def test_every_ladder_class_above_l4_is_the_host_tensor(cls) -> None:
    basis, aux = _ladder_bases(cls)
    X, ran = run_xeri3(basis, aux, NUCLEI, boys=boys, classes={cls})
    assert ran == (cls,)
    cells = _cells(basis, aux, cls)
    scale = float(np.abs(_host(basis, aux, cells, tower=pq_tower)).max())
    _close(X, _host(basis, aux, cells), scale, xeri3_name(cls))


# --- the block ------------------------------------------------------------------


def test_an_aux_block_over_a_subset_of_its_cells_writes_those_alone() -> None:
    block = range(1, 4)  # the aux p, d and f shells; s and g left out
    assert 0 < len(block) < len(AUX.shells)
    classes = {c for c in CLASSES if sum(c) <= 3}
    inside = [c for c in _cells_of(BASIS, AUX, classes) if c[2] in block]
    cells = inside[::2]  # a strict subset of the block's cells
    assert 0 < len(cells) < len(inside)
    aux0 = AUX.offsets[block.start]
    n_blk = sum(AUX.shells[n].n_cart for n in block)
    sentinel = 7.25
    X0 = np.full((3, BASIS.n_ao, BASIS.n_ao, n_blk), sentinel)
    X, ran = run_xeri3(BASIS, AUX, NUCLEI, boys=boys, classes=classes,
                       aux_shells=block, cells=cells, X=X0.copy())
    assert ran
    want = _host(BASIS, AUX, cells)[..., aux0:aux0 + n_blk]
    owned = np.zeros(X.shape, dtype=bool)
    off, aoff = BASIS.offsets, AUX.offsets
    for i, j, n in cells:
        si, sj, sn = BASIS.shells[i], BASIS.shells[j], AUX.shells[n]
        m = slice(off[i], off[i] + si.n_cart)
        q = slice(off[j], off[j] + sj.n_cart)
        p = slice(aoff[n] - aux0, aoff[n] - aux0 + sn.n_cart)
        owned[:, m, q, p] = owned[:, q, m, p] = True
    assert np.all(X[~owned] == sentinel)
    _close(X[owned], want[owned], float(np.abs(want).max()), "the block")


def _cells_of(basis, aux, classes) -> list[tuple[int, int, int]]:
    return [c for cls in sorted(classes) for c in _cells(basis, aux, cls)]


# --- the contracted host: another route on the aux centre ------------------------


@cache
def _h2o(aux_l: int):
    ts = manifest.three_center_set("h2o_sto3g_jkfit_3c")
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return basis, Basis(tuple(s for s in aux.shells if s.l <= aux_l)), nuclei


def _contract(X, Gam, basis, aux, nuclei) -> np.ndarray:
    """`SUM_mnP Gam_mnP d(mn|P)/dR` from `X`: `-X` on `m`'s atom, `-X^T`
    on `n`'s, `X + X^T` (the aux identity) on `P`'s."""
    at = np.repeat(shell_atoms(basis, nuclei),
                   [s.n_cart for s in basis.shells])
    aat = np.repeat(shell_atoms(aux, nuclei), [s.n_cart for s in aux.shells])
    G = np.zeros((len(nuclei), 3))
    gm = -np.einsum("kmnp,mnp->mk", X, Gam)
    gn = -np.einsum("knmp,mnp->nk", X, Gam)
    gp = np.einsum("kmnp,mnp->pk", X + X.transpose(0, 2, 1, 3), Gam)
    np.add.at(G, at, gm)
    np.add.at(G, at, gn)
    np.add.at(G, aat, gp)
    return G


def _sp():
    """test_kernel_geri3.py's bases cut to their s and p shells."""
    return (Basis(tuple(s for s in BASIS.shells if s.l <= 1)),
            Basis(tuple(s for s in AUX.shells if s.l <= 1)), NUCLEI)


@pytest.mark.parametrize("system", ["sp", pytest.param(
    "h2o", marks=pytest.mark.slow)])
def test_the_kernel_contracted_is_the_host_gradient(system: str) -> None:
    """The geri3 bases' s and p shells (fast), and H2O/STO-3G against the
    cc-pVTZ-JKFIT aux basis's s and p shells (`slow`, ~1 min), every class
    of the system."""
    basis, aux, nuclei = _sp() if system == "sp" else _h2o(1)
    Gam = np.random.default_rng(11).standard_normal(
        (basis.n_ao, basis.n_ao, aux.n_ao))
    X, ran = run_xeri3(basis, aux, nuclei, boys=boys)
    assert max(sum(c) for c in ran) == 3  # (pp|p)
    want = np.array(host_df.eri3_gradient(basis, aux, Gam, nuclei, pq=pq))
    got = _contract(X, Gam, basis, aux, nuclei)
    err = np.abs(got - want).max()
    assert err <= 64 * EPS * np.abs(want).max(), err / np.abs(want).max()


# --- libcint ----------------------------------------------------------------------


@cache
def _witness():
    wit = manifest.df_gradient()
    ts = wit.three_center
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return wit, basis, aux, nuclei


@pytest.mark.slow  # the host over every class of the aux-i system, ~1 min
def test_the_host_contracted_is_libcints_at_every_ladder_class() -> None:
    wit, basis, aux, nuclei = _witness()
    assert sorted(wit.eri3) == LADDER
    big = max(float(np.abs(g).max()) for g in wit.eri3.values())
    worst = 0.0
    for cls in LADDER:
        X = _host(basis, aux, _cells(basis, aux, cls))
        g = _contract(X, wit.gam(), basis, aux, nuclei)
        worst = max(worst, float(np.abs(g - wit.eri3[cls]).max()) / big)
    print(f"\nhost ip1, contracted, against libcint: {worst:.2e} of max|G|")
    assert worst <= 1e-12  # test_kernel_df_grad_ladder.py's LIBCINT_TOL


def _libcint(mol, auxmol):
    from pyscf import df
    shape = (3, mol.nao, mol.nao, auxmol.nao)
    return tuple(df.incore.aux_e2(mol, auxmol, intor=f"int3c2e_{i}_cart",
                                  aosym="s1", comp=3).reshape(shape)
                 for i in ("ip1", "ip2"))


def _against_libcint(mol, auxmol, what: str) -> None:
    basis, aux = ob.from_mole(mol), ob.from_mole(auxmol)
    ip1, ip2 = _libcint(mol, auxmol)
    X = _host(basis, aux, None)
    e1 = float(np.abs(X - ip1).max() / np.abs(ip1).max())
    e2 = float(np.abs(-X - X.transpose(0, 2, 1, 3) - ip2).max()
               / np.abs(ip2).max())
    print(f"\n{what}: ip1 {e1:.2e}, ip2 by the identity {e2:.2e} of max")
    assert e1 <= LIBCINT_TOL and e2 <= LIBCINT_TOL


H2O = "O 0 0 0; H 1.43 0.21 -1.07; H -1.38 0.35 -1.12"


@pytest.mark.oracle
@pytest.mark.slow  # the host over every cell, ~30 s
def test_the_host_is_libcints_ip1_on_h2o_ccpvdz_jkfit() -> None:
    gto = pytest.importorskip("pyscf.gto")
    mol = gto.M(atom=H2O, unit="Bohr", basis="cc-pvdz", cart=True)
    aux = gto.M(atom=H2O, unit="Bohr", basis="cc-pvdz-jkfit", cart=True)
    assert max(aux.bas_angular(i) for i in range(aux.nbas)) == 3
    _against_libcint(mol, aux, "H2O/cc-pVDZ, cc-pVDZ-JKFIT")


@pytest.mark.oracle
@pytest.mark.slow  # the host over every cell, ~1 min
def test_the_host_is_libcints_ip1_through_orbital_f_and_aux_i() -> None:
    pytest.importorskip("pyscf")
    from oracle.generate import AUX_I, CASES, _mole
    from pyscf import df, gto
    mol = _mole(CASES["prim_spd_f"])
    aux = df.make_auxmol(mol, {
        s: gto.basis.parse(AUX_I, symb=s)
        for s in sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})})
    assert max(mol.bas_angular(i) for i in range(mol.nbas)) == 3
    assert max(aux.bas_angular(i) for i in range(aux.nbas)) == 6
    _against_libcint(mol, aux, "prim_spd_f, AUX_I")


# --- the kernel's shape -----------------------------------------------------------


def test_the_key_and_the_name() -> None:
    s = xeri3_spec(0, 1, 2)
    assert s.key() == "coulomb/d1u/(0,1|2)/b1/eri3/literal"
    assert s.key() == geri3_spec(0, 1, 2).key().replace("/d1/", "/d1u/")
    assert xeri3_name((0, 1, 2)) == "xeri3_012"
    assert xeri3_name((0, 11, 0)) == "xeri3_0_11_0"
    assert build_xeri3_kernel(s).name == "xeri3_012"


def test_a_contracted_key_or_another_target_is_refused() -> None:
    with pytest.raises(ValueError, match="not an uncontracted d1"):
        build_xeri3_kernel(geri3_spec(0, 1, 2))
    with pytest.raises(ValueError, match="not an uncontracted d1"):
        build_xeri3_kernel(Spec(Operator.COULOMB, 0, (1, 0, 0), Target.ERI3))
    with pytest.raises(ValueError, match="kernel_xeri3 builds it"):
        build_geri3_kernel(xeri3_spec(0, 1, 2))


def test_the_spec_admits_the_form_on_mmd_eri3_at_d1_alone() -> None:
    for d in (0, 2):
        with pytest.raises(ValueError, match="uncontracted form is built"):
            Spec(Operator.COULOMB, d, (0, 1, 2), Target.ERI3,
                 uncontracted=True)
    with pytest.raises(ValueError, match="uncontracted on d1 target "
                                         "eri3.hgp"):
        Spec(Operator.COULOMB, 1, (0, 1, 2), Target.ERI3, engine=Engine.HGP,
             uncontracted=True)
    with pytest.raises(ValueError, match="uncontracted on d1 target metric"):
        Spec(Operator.COULOMB, 1, (1, 2), Target.METRIC, uncontracted=True)
    with pytest.raises(ValueError, match="not a bool"):
        Spec(Operator.COULOMB, 1, (0, 1, 2), Target.ERI3, uncontracted=1)


def test_the_parameters_and_the_one_output() -> None:
    k = build_xeri3_kernel((0, 1, 1))
    assert [p.name for p in k.params] == [
        "n_cell", "n_bra_rec", "n_aux_rec", "n_ao", "n_blk", "aux0",
        "zeta_ab", "P", "XPA", "XPB", "s_ab", "ea_bra", "eb_ket", "A",
        "alpha", "s_aux", "ab_rec0", "ab_rec1", "p_rec0", "p_rec1", "off_a",
        "off_b", "off_p", "X"]
    assert [q.name for q in k.params if q.intent is Intent.OUT] == ["X"]


def test_no_atomic_and_no_raw_escape_hatch() -> None:
    for cls in ((0, 0, 0), (1, 1, 1), (1, 2, 3)):
        nodes = list(walk(build_xeri3_kernel(cls).body))
        assert not any(isinstance(s, (AtomicAdd, Raw)) for s in nodes)


def test_the_staging_reaches_swapped_and_diagonal_cells() -> None:
    by = {(c.la, c.lb, c.lP): c for c in eri3_classes(BASIS, AUX)}
    assert any(by[(0, 1, 0)].swapped)
    assert any(a == b for a, b, _ in by[(1, 1, 0)].cells)


def test_the_host_refuses_a_cell_that_is_not_i_le_j() -> None:
    with pytest.raises(ValueError, match="is not i <= j"):
        host_df.eri3_ip1(BASIS, AUX, pq=pq, cells=[(1, 0, 0)])
