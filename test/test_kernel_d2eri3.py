"""`kernel_d2eri3`: the three-center DF Hessian kernel (d2), interpreted.

THE WITNESS is `hessian.host_eri3.eri3_hessian` (itself held to finite
differences of the d1 `(mn|P)` host gradient, test/test_hessian_host_eri3.py),
which takes ANOTHER ROUTE on every centre: it differentiates each 3-D
Cartesian as a list of moved Cartesians, the aux one included, where the
kernel operates per axis on the AO pair and shifts the Hermite tower for the
aux centre. Per class, one launch over every staged cell into a zeroed `H`,
against the host over the same cells; then the whole Hessian of H2O/STO-3G
against the cc-pVTZ-JKFIT aux basis's s and p shells (`slow`), and
translational invariance and symmetry, which nothing in the kernel assumes.

ROUND-OFF, NOT BITWISE: `|H_kernel - H_host| <= TOL max(twin)`, `TOL =
16 eps`, `twin` the host's `absolute=True` sum (every factor in `|.|`), as
test_kernel_h1e.py. Measured <= 0.40 eps through L = 4. `Gam` is a random
tensor, not symmetric in `(m, n)`.

SCOPE: every class `la <= lb` with `la + lb + lP <= 4` (the d0 kernel's
cap, test_kernel_geri3.py's bases). `L` for the tier cut is `la + lb + lP +
2`, the Hermite order the Hessian reaches: up to 2 fast, 3 and 4 `slow`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from df_grad_launch import run_d2eri3
from oracle import basis as ob
from oracle import manifest
from test_kernel_geri3 import AUX, BASIS, CLASSES, NUCLEI, _cells, _gam, _id
from tiers import by_l

from felizjueves.ao.shell import Basis
from felizjueves.hessian import host_eri3
from felizjueves.hessian.kernel_d2eri3 import (
    ACCUMULATORS,
    build_d2eri3_kernel,
    d2eri3_name,
    d2eri3_plan,
    d2eri3_spec,
)
from felizjueves.hessian.kernel_h1e import ATOMIC_REASON, ZEROED_H
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(twin): the module docstring


def _close(got, want, scale: float, what: str) -> None:
    err = float(np.abs(got - want).max())
    print(f"\n{what}: {err / scale / EPS:.2f} eps of the twin")
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps"


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=_id(c))
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_hessian(cls) -> None:
    Gam = _gam(BASIS, AUX)
    H, ran = run_d2eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={cls})
    assert ran == (cls,)
    cells = _cells(BASIS, AUX, cls)
    want = np.array(host_eri3.eri3_hessian(BASIS, AUX, Gam, NUCLEI, pq=pq,
                                         cells=cells))
    twin = np.array(host_eri3.eri3_hessian(BASIS, AUX, Gam, NUCLEI, pq=pq,
                                         cells=cells, absolute=True))
    scale = float(twin.max())
    _close(H, want, scale, f"d2eri3 {cls}")
    # Three centres, each differentiated on its own: invariance is a check.
    assert np.abs(H.sum(axis=0)).max() <= TOL * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= TOL * scale


@cache
def _h2o():
    ts = manifest.three_center_set("h2o_sto3g_jkfit_3c")
    basis, aux = ob.from_fixture(ts.parent), ob.from_fixture(ts)
    nuclei = tuple((float(a.charge), tuple(a.coord)) for a in ts.parent.atoms)
    return basis, Basis(tuple(s for s in aux.shells if s.l <= 1)), nuclei


@pytest.mark.slow
def test_the_whole_hessian_of_h2o_against_an_sp_aux_basis() -> None:
    basis, aux, nuclei = _h2o()
    Gam = _gam(basis, aux, seed=11)
    H, ran = run_d2eri3(basis, aux, nuclei, Gam, boys=boys)
    assert max(sum(c) for c in ran) == 3  # (pp|p)
    want = np.array(host_eri3.eri3_hessian(basis, aux, Gam, nuclei, pq=pq))
    twin = np.array(host_eri3.eri3_hessian(basis, aux, Gam, nuclei, pq=pq,
                                         absolute=True))
    _close(H, want, float(twin.max()), "d2eri3 H2O")
    assert np.abs(H.sum(axis=0)).max() <= 1e-12 * np.abs(H).max()


def test_h_is_added_to_not_overwritten() -> None:
    Gam = _gam(BASIS, AUX)
    H0 = np.arange(81.0).reshape(3, 3, 3, 3)
    H, _ = run_d2eri3(BASIS, AUX, NUCLEI, Gam, boys=boys,
                      classes={(0, 1, 0)}, H=H0.copy())
    want, _ = run_d2eri3(BASIS, AUX, NUCLEI, Gam, boys=boys,
                         classes={(0, 1, 0)})
    assert np.abs(H - (H0 + want)).max() <= 8 * EPS * np.abs(H0).max()


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert d2eri3_spec(0, 1, 2).key() == "coulomb/d2/(0,1|2)/b1/eri3/literal"
    assert d2eri3_name((0, 1, 2)) == "d2eri3_012"
    assert d2eri3_name((0, 11, 0)) == "d2eri3_0_11_0"
    assert build_d2eri3_kernel(d2eri3_spec(0, 1, 2)).name == "d2eri3_012"


def test_a_d1_key_or_another_target_is_refused() -> None:
    with pytest.raises(ValueError, match="not a d2 Target.ERI3 key"):
        build_d2eri3_kernel(Spec(Operator.COULOMB, 1, (1, 0, 0), Target.ERI3))
    with pytest.raises(ValueError, match="not a d2 Target.ERI3 key"):
        build_d2eri3_kernel(Spec(Operator.COULOMB, 2, (1, 0), Target.JAB))


def test_the_tower_is_two_orders_up() -> None:
    _, order = d2eri3_plan((1, 1, 1))
    assert max(sum(r) for r, _ in order) == 5


def test_the_parameters_and_the_one_output() -> None:
    k = build_d2eri3_kernel((0, 1, 1))
    assert [p.name for p in k.params] == [
        "n_cell", "n_bra_rec", "n_aux_rec", "n_ao", "n_aux", "n_atom",
        "zeta_ab", "P", "XPA", "XPB", "s_ab", "ea_bra", "eb_ket", "A",
        "alpha", "s_aux", "ab_rec0", "ab_rec1", "p_rec0", "p_rec1", "off_a",
        "off_b", "off_p", "atom_a", "atom_b", "atom_p", "M", "H"]
    assert [q.name for q in k.params if q.intent is Intent.OUT] == ["H"]


def test_every_write_is_an_atomic_add_into_h() -> None:
    assert len(ACCUMULATORS) == 45
    nodes = list(walk(build_d2eri3_kernel((1, 1, 1)).body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 81 and all(a.dst.array == "H" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_H in notes
    assert not any(isinstance(s, Raw) for s in nodes)
