"""`kernel_geri3`: the three-center DF gradient kernel (d1), interpreted.

THE WITNESS is `gradient.host_df.eri3_gradient` (itself held to finite
differences of the libcint-witnessed `(mn|P)`, test/test_gradient_host_df.py),
which takes ANOTHER ROUTE on the aux centre: the Cartesian rule on its
single-center `E`, where the kernel shifts the Hermite tower. The AO pair's
centres take the same rule in both, on `mmd.hermite.e_axis` against the
kernel's per-axis fragment. Per class, one launch over every staged cell
into a zeroed `G`, against the host over the same cells; then the whole
gradient of H2O/STO-3G against the cc-pVTZ-JKFIT aux basis's s and p shells
(`slow`), and translational invariance, which nothing in the kernel assumes.

ROUND-OFF, NOT BITWISE: `|G_kernel - G_host| <= TOL max(scale)`, `TOL =
16 eps`, `scale` the split-centre scale (test_kernel_g1e.py): the host's
`|G|` with every AO and aux shell on a row of its own, folded back onto the
atoms. `Gam` is a random tensor, not symmetric in `(m, n)`.

SCOPE: every class `la <= lb` with `la + lb + lP <= 4` (the d0 kernel's
cap). `L` for the tier cut is `la + lb + lP + 1`, the Hermite order the
gradient reaches: up to 3 fast, 4 `slow`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from df_grad_launch import run_eri3
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.density_fit.eri3 import eri3_classes
from felizjueves.gradient import host_df
from felizjueves.gradient.host import shell_atoms
from felizjueves.gradient.kernel_g1e import ATOMIC_REASON, ZEROED_G
from felizjueves.gradient.kernel_geri3 import (
    build_geri3_kernel,
    geri3_name,
    geri3_plan,
    geri3_spec,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(scale): the module docstring
L_CAP = 4
CLASSES = [(a, b, p) for a in range(L_CAP + 1) for b in range(a, L_CAP + 1)
           for p in range(L_CAP + 1) if a + b + p <= L_CAP]

A = (0.0, 0.0, 0.0)
B = (0.3, -0.7, 1.1)
C = (-1.2, 0.4, 0.25)
NUCLEI = ((8.0, A), (1.0, B), (1.0, C))
# The AO side: every l to g, the higher l first on each centre (so every
# la < lb class has cells staged swapped), two shells of each l <= 1 (so the
# la == lb classes through (pp) have diagonal and off-diagonal pairs), one-
# and two-centre pairs, contracted and not.
BASIS = Basis((
    Shell(4, A, (0.9,), (1.0,)),
    Shell(2, A, (1.1, 0.3), (0.5, 0.6)),
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(0, A, (3.4, 0.62), (0.15, 0.53)),
    Shell(3, B, (0.6,), (0.7,)),
    Shell(2, B, (0.8,), (1.0,)),
    Shell(0, B, (1.3, 0.21), (0.6, 0.47)),
    Shell(1, B, (0.9,), (1.0,)),
))
# The aux side: every l to g, on all three centres between them.
AUX = Basis((
    Shell(0, C, (2.0, 0.4), (0.7, 0.5)),
    Shell(1, A, (0.85,), (1.0,)),
    Shell(2, B, (1.4,), (0.6,)),
    Shell(3, C, (0.7,), (1.0,)),
    Shell(4, B, (1.2,), (1.0,)),
))


def _id(c) -> str:
    return f"{'spdfg'[c[0]]}{'spdfg'[c[1]]}|{'spdfg'[c[2]]}"


def _gam(basis, aux, seed=5) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(
        (basis.n_ao, basis.n_ao, aux.n_ao))


def _cells(basis, aux, cls) -> list[tuple[int, int, int]]:
    sh, ash = basis.shells, aux.shells
    return [(i, j, n) for i in range(len(sh)) for j in range(i, len(sh))
            for n in range(len(ash))
            if (*sorted((sh[i].l, sh[j].l)), ash[n].l) == cls]


def _scale(basis, aux, nuclei, Gam, cells) -> float:
    nat, ns = len(nuclei), len(basis.shells)
    real, areal = shell_atoms(basis, nuclei), shell_atoms(aux, nuclei)
    split = tuple(nat + i for i in range(ns))
    asplit = tuple(nat + ns + i for i in range(len(aux.shells)))
    sc = np.zeros((nat, 3))
    for c in cells:
        g = np.abs(np.array(host_df.eri3_gradient(
            basis, aux, Gam, nuclei, pq=pq, cells=[c], atoms=split,
            aux_atoms=asplit)))
        for i, row in enumerate(split):
            sc[real[i]] += g[row]
        for i, row in enumerate(asplit):
            sc[areal[i]] += g[row]
    return float(sc.max())


def _close(got, want, scale: float, what: str) -> None:
    err = float(np.abs(got - want).max())
    print(f"\n{what}: {err / scale / EPS:.2f} eps of scale")
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"


def test_the_bases_reach_every_class_swapped_and_one_centre() -> None:
    by = {(c.la, c.lb, c.lP): c for c in eri3_classes(BASIS, AUX)}
    sh, ash = BASIS.shells, AUX.shells
    for cls in CLASSES:
        c = by[cls]
        if cls[0] != cls[1]:
            assert any(c.swapped), cls
        else:
            assert any(a == b for a, b, _ in c.cells), cls
        assert any(sh[a].centre != ash[p].centre for a, _, p in c.cells), cls


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c))
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_gradient(cls) -> None:
    Gam = _gam(BASIS, AUX)
    G, ran = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={cls})
    assert ran == (cls,)
    cells = _cells(BASIS, AUX, cls)
    want = np.array(host_df.eri3_gradient(BASIS, AUX, Gam, NUCLEI, pq=pq,
                                          cells=cells))
    scale = _scale(BASIS, AUX, NUCLEI, Gam, cells)
    _close(G, want, scale, f"geri3 {cls}")
    # Three centres, each differentiated on its own: invariance is a check.
    assert np.abs(G.sum(axis=0)).max() <= TOL * scale


@cache
def _h2o():
    ts = manifest.three_center_set("h2o_sto3g_jkfit_3c")
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return basis, Basis(tuple(s for s in aux.shells if s.l <= 1)), nuclei


@pytest.mark.slow
def test_the_whole_gradient_of_h2o_against_an_sp_aux_basis() -> None:
    basis, aux, nuclei = _h2o()
    Gam = _gam(basis, aux, seed=11)
    G, ran = run_eri3(basis, aux, nuclei, Gam, boys=boys)
    assert max(sum(c) for c in ran) == 3  # (pp|p)
    want = np.array(host_df.eri3_gradient(basis, aux, Gam, nuclei, pq=pq))
    err = np.abs(G - want).max()
    assert err <= 64 * EPS * np.abs(want).max(), err / np.abs(want).max()
    assert np.abs(G.sum(axis=0)).max() <= 1e-13 * np.abs(G).max()


def test_g_is_added_to_not_overwritten() -> None:
    Gam = _gam(BASIS, AUX)
    G0 = np.arange(9.0).reshape(3, 3)
    G, _ = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={(0, 1, 0)},
                    G=G0.copy())
    want, _ = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys,
                       classes={(0, 1, 0)})
    assert np.abs(G - (G0 + want)).max() <= 8 * EPS * np.abs(G0).max()


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert geri3_spec(0, 1, 2).key() == "coulomb/d1/(0,1|2)/b1/eri3/literal"
    assert geri3_name((0, 1, 2)) == "geri3_012"
    assert geri3_name((0, 11, 0)) == "geri3_0_11_0"
    assert build_geri3_kernel(geri3_spec(0, 1, 2)).name == "geri3_012"


def test_a_d0_key_or_another_target_is_refused() -> None:
    with pytest.raises(ValueError, match="not a d1 Target.ERI3 key"):
        build_geri3_kernel(Spec(Operator.COULOMB, 0, (1, 0, 0), Target.ERI3))
    with pytest.raises(ValueError, match="not a d1 Target.ERI3 key"):
        build_geri3_kernel(Spec(Operator.COULOMB, 1, (1, 0), Target.METRIC))


def test_the_tower_is_one_order_up() -> None:
    _, order = geri3_plan((1, 1, 1))
    assert max(sum(r) for r, _ in order) == 4


def test_the_parameters_and_the_one_output() -> None:
    k = build_geri3_kernel((0, 1, 1))
    assert [p.name for p in k.params] == [
        "n_cell", "n_bra_rec", "n_aux_rec", "n_ao", "n_aux", "n_atom",
        "zeta_ab", "P", "XPA", "XPB", "s_ab", "ea_bra", "eb_ket", "A",
        "alpha", "s_aux", "ab_rec0", "ab_rec1", "p_rec0", "p_rec1", "off_a",
        "off_b", "off_p", "atom_a", "atom_b", "atom_p", "M", "G"]
    assert [q.name for q in k.params if q.intent is Intent.OUT] == ["G"]


def test_every_write_is_an_atomic_add_into_g() -> None:
    nodes = list(walk(build_geri3_kernel((1, 1, 1)).body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 9 and all(a.dst.array == "G" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)
