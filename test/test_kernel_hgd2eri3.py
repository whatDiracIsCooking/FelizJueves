"""`kernel_hgd2eri3`: the three-center DF Hessian kernel (d2) on the HGP
engine, interpreted.

THE WITNESSES are `kernel_d2eri3`'s: `hessian.host_eri3.eri3_hessian` (held
to finite differences of the d1 `(mn|P)` host gradient,
test/test_hessian_host_eri3.py), on test_kernel_geri3.py's bases (every l to
g, swapped and diagonal cells, one- and two-centre pairs) and its random
`Gam`; and the MMD kernel `kernel_d2eri3` interpreted on the same staging.
Translational invariance and `H = H^T` are checks: the kernel differentiates
all three centres and writes every block from its own accumulator.

ROUND-OFF, NOT BITWISE: within `TOL = 16 eps` of the host's `absolute=True`
twin, `2 TOL` against the MMD kernel (both round). Measured at writing:
<= 0.33 eps of the twin through L = 3.

SCOPE: every class `la <= lb` with `la + lb + lP <= 4`, as for the MMD
kernel (the CLI's ladder stops at s and p shells, `HGP_DF_L_MAX`). `L` for
the tier cut is `la + lb + lP + 2`, the order the Hessian reaches.
"""

from __future__ import annotations

import numpy as np
import pytest
from df_grad_launch import run_d2eri3
from test_kernel_d2eri3 import _h2o
from test_kernel_geri3 import AUX, BASIS, CLASSES, NUCLEI, _cells, _gam, _id
from tiers import by_l

from felizjueves.hessian import host_eri3, kernel_d2eri3, kernel_h1e
from felizjueves.hessian import kernel_hgd2eri3 as kh
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(twin): the module docstring
HGP = Engine.HGP


def _cases():
    return [by_l(c, sum(c) + 2, id=_id(c)) for c in CLASSES]


def _close(got, want, scale: float, tol: float, what: str) -> None:
    err = float(np.abs(got - want).max())
    print(f"\n{what}: {err / scale / EPS:.2f} eps of the twin")
    assert err <= tol * scale, f"{what}: {err / scale / EPS:.1f} eps"


def _twin(Gam, cells) -> float:
    return float(np.array(host_eri3.eri3_hessian(
        BASIS, AUX, Gam, NUCLEI, pq=pq, cells=cells, absolute=True)).max())


@pytest.mark.parametrize("cls", _cases())
def test_one_launch_over_the_class_is_the_host_hessian(cls) -> None:
    Gam = _gam(BASIS, AUX)
    H, ran = run_d2eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={cls},
                        engine=HGP)
    assert ran == (cls,)
    cells = _cells(BASIS, AUX, cls)
    want = np.array(host_eri3.eri3_hessian(BASIS, AUX, Gam, NUCLEI, pq=pq,
                                           cells=cells))
    scale = _twin(Gam, cells)
    _close(H, want, scale, TOL, f"hgd2eri3 {cls}")
    # Three centres, each differentiated on its own: invariance is a check.
    assert np.abs(H.sum(axis=0)).max() <= TOL * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= TOL * scale


@pytest.mark.parametrize("cls", _cases())
def test_the_hgp_and_mmd_kernels_agree(cls) -> None:
    Gam = _gam(BASIS, AUX, seed=7)
    got, _ = run_d2eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={cls},
                        engine=HGP)
    want, _ = run_d2eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={cls})
    _close(got, want, _twin(Gam, _cells(BASIS, AUX, cls)), 2 * TOL,
           f"hgd2eri3 vs d2eri3 {cls}")


@pytest.mark.slow
def test_the_whole_hessian_of_h2o_against_an_sp_aux_basis() -> None:
    basis, aux, nuclei = _h2o()
    Gam = _gam(basis, aux, seed=11)
    H, ran = run_d2eri3(basis, aux, nuclei, Gam, boys=boys, engine=HGP)
    assert max(sum(c) for c in ran) == 3  # (pp|p)
    want = np.array(host_eri3.eri3_hessian(basis, aux, Gam, nuclei, pq=pq))
    twin = np.array(host_eri3.eri3_hessian(basis, aux, Gam, nuclei, pq=pq,
                                           absolute=True))
    _close(H, want, float(twin.max()), TOL, "hgd2eri3 H2O")


def test_h_is_added_to_not_overwritten() -> None:
    Gam = _gam(BASIS, AUX)
    H0 = np.arange(81.0).reshape(3, 3, 3, 3)
    H, _ = run_d2eri3(BASIS, AUX, NUCLEI, Gam, boys=boys,
                      classes={(0, 1, 0)}, H=H0.copy(), engine=HGP)
    want, _ = run_d2eri3(BASIS, AUX, NUCLEI, Gam, boys=boys,
                         classes={(0, 1, 0)}, engine=HGP)
    assert np.abs(H - (H0 + want)).max() <= 8 * EPS * np.abs(H0).max()


def test_the_launcher_refuses_a_form_on_hgp() -> None:
    with pytest.raises(ValueError, match="one form"):
        run_d2eri3(BASIS, AUX, NUCLEI, _gam(BASIS, AUX), boys=boys,
                   classes={(0, 0, 0)}, engine=HGP, lowstack=True)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert kh.hgd2eri3_spec(0, 1, 2).key() == \
        "coulomb/d2/(0,1|2)/b1/eri3.hgp/literal"
    assert kh.hgd2eri3_name((0, 1, 2)) == "hgd2eri3_012"
    assert kh.hgd2eri3_name((0, 11, 0)) == "hgd2eri3_0_11_0"
    assert kh.build_hgd2eri3_kernel(kh.hgd2eri3_spec(0, 1, 2)).name == \
        "hgd2eri3_012"


def test_another_key_is_refused() -> None:
    for spec in (Spec(Operator.COULOMB, 1, (1, 0, 0), Target.ERI3,
                      engine=HGP),
                 Spec(Operator.COULOMB, 2, (1, 0, 0), Target.ERI3),
                 Spec(Operator.COULOMB, 2, (1, 0), Target.METRIC)):
        with pytest.raises(ValueError, match="not a d2 HGP Target.ERI3 key"):
            kh.build_hgd2eri3_kernel(spec)


def test_the_tables_of_the_smallest_class() -> None:
    # (ss|s): every raised-twice class, each single-raised pair, and the
    # level class under each side's own weight (the k = l middle term).
    tables = kh.hessian_tables((0, 0, 0))
    got = {(t.kind, t.am) for t in tables}
    assert got == {
        ("X", (0, 0, 0, 0)), ("Y", (0, 0, 0, 0)), ("C", (0, 0, 0, 0)),
        ("XX", (2, 0, 0, 0)), ("YY", (0, 2, 0, 0)), ("CC", (0, 0, 2, 0)),
        ("XY", (1, 1, 0, 0)), ("XC", (1, 0, 1, 0)), ("YC", (0, 1, 1, 0))}
    assert len({t.tag for t in tables}) == len(tables)


def test_no_table_has_a_negative_l() -> None:
    for cls in ((0, 1, 0), (1, 1, 1), (0, 2, 1)):
        for t in kh.hessian_tables(cls):
            assert min(t.am) >= 0, (cls, t)


def test_the_parameters_and_the_one_output_are_d2eri3s() -> None:
    for cls in ((0, 1, 1), (1, 1, 0)):
        k = kh.build_hgd2eri3_kernel(cls)
        mmd = kernel_d2eri3.build_d2eri3_kernel(cls)
        assert k.params == mmd.params
        assert [q.name for q in k.params if q.intent is Intent.OUT] == ["H"]


def test_the_restated_notes_are_kernel_h1es() -> None:
    assert kh.ATOMIC_REASON == kernel_h1e.ATOMIC_REASON
    assert kh.ZEROED_H == kernel_h1e.ZEROED_H
    assert len(kh.COMPONENTS) == len(kernel_d2eri3.ACCUMULATORS) == 45
    assert set(kh.COMPONENTS) == set(kernel_d2eri3.ACCUMULATORS)


def test_every_write_is_an_atomic_add_into_h() -> None:
    nodes = list(walk(kh.build_hgd2eri3_kernel((1, 1, 1)).body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 81 and all(a.dst.array == "H" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert kh.ATOMIC_REASON in notes and kh.ZEROED_H in notes
    assert not any(isinstance(s, Raw) for s in nodes)
