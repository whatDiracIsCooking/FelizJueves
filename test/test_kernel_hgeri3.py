"""`kernel_hgeri3`: the three-center DF gradient kernel (d1) on the HGP
engine, interpreted.

THE WITNESSES are `kernel_geri3`'s: `gradient.host_df.eri3_gradient` (held
to finite differences of the libcint-witnessed `(mn|P)`), on
test_kernel_geri3.py's bases (every l to g, swapped and diagonal cells, one-
and two-centre pairs) and its random `Gam`; the MMD kernel interpreted on the
same staging; and libcint, the tracked DF gradient witness
(`oracle.manifest.df_gradient`, `int3c2e_ip1_cart` / `int3c2e_ip2_cart`).
Translational invariance is a check: the kernel differentiates all three
centres.

ROUND-OFF, NOT BITWISE: within `TOL = 16 eps` of the split-centre scale
(test_kernel_geri3.py's), `2 TOL` against the MMD kernel (both round), and
`1e-12` of the witness's largest class `max|G|` against libcint.

SCOPE: every class `la <= lb` with `la + lb + lP <= 4`, as for the MMD
kernel (the CLI's ladder stops at s and p shells, `HGP_DF_L_MAX`). `L` for
the tier cut is `la + lb + lP + 1`, the order the gradient reaches.
"""

from __future__ import annotations

import numpy as np
import pytest
from df_grad_launch import run_eri3
from test_kernel_df_grad_ladder import LIBCINT_TOL, _big, _witness
from test_kernel_geri3 import AUX, BASIS, CLASSES, NUCLEI, _cells, _gam, _id
from test_kernel_geri3 import _scale as _split_scale
from tiers import by_l

from felizjueves.gradient import host_df
from felizjueves.gradient import kernel_g1e as g1e
from felizjueves.gradient import kernel_hgeri3 as kh
from felizjueves.gradient.kernel_geri3 import build_geri3_kernel
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import pq
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(scale): the module docstring
HGP = Engine.HGP


def _cases():
    return [by_l(c, sum(c) + 1, id=_id(c)) for c in CLASSES]


def _close(got, want, scale: float, tol: float, what: str) -> None:
    err = float(np.abs(got - want).max())
    print(f"\n{what}: {err / scale / EPS:.2f} eps of scale")
    assert err <= tol * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"


@pytest.mark.parametrize("cls", _cases())
def test_one_launch_over_the_class_is_the_host_gradient(cls) -> None:
    Gam = _gam(BASIS, AUX)
    G, ran = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={cls},
                      engine=HGP)
    assert ran == (cls,)
    cells = _cells(BASIS, AUX, cls)
    want = np.array(host_df.eri3_gradient(BASIS, AUX, Gam, NUCLEI, pq=pq,
                                          cells=cells))
    scale = _split_scale(BASIS, AUX, NUCLEI, Gam, cells)
    _close(G, want, scale, TOL, f"hgeri3 {cls}")
    # Three centres, each differentiated on its own: invariance is a check.
    assert np.abs(G.sum(axis=0)).max() <= TOL * scale


@pytest.mark.parametrize("cls", _cases())
def test_the_hgp_and_mmd_kernels_agree(cls) -> None:
    Gam = _gam(BASIS, AUX, seed=7)
    got, _ = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={cls},
                      engine=HGP)
    want, _ = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={cls})
    scale = _split_scale(BASIS, AUX, NUCLEI, Gam, _cells(BASIS, AUX, cls))
    _close(got, want, scale, 2 * TOL, f"hgeri3 vs geri3 {cls}")


# The witness's classes the s / p ladder reaches, and a d class on each
# side. Not (ss|s): its one AO s shell makes that class's G exactly zero.
_LIBCINT = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (0, 0, 2),
            (0, 2, 1), (1, 1, 2)]


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=_id(c))
                                 for c in _LIBCINT])
def test_a_class_is_libcints(cls) -> None:
    wit, basis, aux, nuclei = _witness()
    G, ran = run_eri3(basis, aux, nuclei, wit.gam(), boys=boys,
                      classes={cls}, engine=HGP)
    assert ran == (cls,) and np.abs(wit.eri3[cls]).max() > 0
    w = float(np.abs(G - wit.eri3[cls]).max()) / _big(wit.eri3)
    print(f"\n{kh.hgeri3_name(cls)} vs libcint: {w:.2e} of max")
    assert w <= LIBCINT_TOL


def test_g_is_added_to_not_overwritten() -> None:
    Gam = _gam(BASIS, AUX)
    G0 = np.arange(9.0).reshape(3, 3)
    G, _ = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys, classes={(0, 1, 0)},
                    G=G0.copy(), engine=HGP)
    want, _ = run_eri3(BASIS, AUX, NUCLEI, Gam, boys=boys,
                       classes={(0, 1, 0)}, engine=HGP)
    assert np.abs(G - (G0 + want)).max() <= 8 * EPS * np.abs(G0).max()


def test_the_launcher_refuses_a_form_on_hgp() -> None:
    with pytest.raises(ValueError, match="one form"):
        run_eri3(BASIS, AUX, NUCLEI, _gam(BASIS, AUX), boys=boys,
                 classes={(0, 0, 0)}, engine=HGP, lowstack=True)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert kh.hgeri3_spec(0, 1, 2).key() == \
        "coulomb/d1/(0,1|2)/b1/eri3.hgp/literal"
    assert kh.hgeri3_name((0, 1, 2)) == "hgeri3_012"
    assert kh.hgeri3_name((0, 11, 0)) == "hgeri3_0_11_0"
    assert kh.build_hgeri3_kernel(kh.hgeri3_spec(0, 1, 2)).name == \
        "hgeri3_012"


def test_another_key_is_refused() -> None:
    for spec in (Spec(Operator.COULOMB, 0, (1, 0, 0), Target.ERI3,
                      engine=HGP),
                 Spec(Operator.COULOMB, 1, (1, 0, 0), Target.ERI3),
                 Spec(Operator.COULOMB, 1, (1, 0), Target.METRIC)):
        with pytest.raises(ValueError, match="not a d1 HGP Target.ERI3 key"):
            kh.build_hgeri3_kernel(spec)


def test_hgp_is_legal_at_d1_on_eri3_alone() -> None:
    with pytest.raises(ValueError, match="derivative order 1 on target "
                                         "metric.hgp"):
        Spec(Operator.COULOMB, 1, (1, 0), Target.METRIC, engine=HGP)
    with pytest.raises(ValueError, match="derivative order 2 on target "
                                         "eri3.hgp"):
        Spec(Operator.COULOMB, 2, (1, 0, 0), Target.ERI3, engine=HGP)


def test_the_derivative_tables() -> None:
    # The built orientation is the higher l first: (sp|s) builds (p s|s).
    assert kh.derivative_tables((0, 1, 0)) == (
        kh.Table("uX", (2, 0, 0, 0), "X"), kh.Table("dX", (0, 0, 0, 0), "1"),
        kh.Table("uY", (1, 1, 0, 0), "Y"), kh.Table("uC", (1, 0, 1, 0), "C"))
    assert [t.tag for t in kh.derivative_tables((1, 1, 1))] == [
        "uX", "dX", "uY", "dY", "uC", "dC"]
    assert [t.tag for t in kh.derivative_tables((0, 0, 0))] == [
        "uX", "uY", "uC"]


def test_the_parameters_and_the_one_output_are_geri3s() -> None:
    for cls in ((0, 1, 1), (1, 1, 0)):
        k, mmd = kh.build_hgeri3_kernel(cls), build_geri3_kernel(cls)
        assert k.params == mmd.params
        assert [q.name for q in k.params if q.intent is Intent.OUT] == ["G"]


def test_the_restated_notes_are_kernel_g1es() -> None:
    assert kh.ATOMIC_REASON == g1e.ATOMIC_REASON
    assert kh.ZEROED_G == g1e.ZEROED_G


def test_every_write_is_an_atomic_add_into_g() -> None:
    nodes = list(walk(kh.build_hgeri3_kernel((1, 1, 1)).body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 9 and all(a.dst.array == "G" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert kh.ATOMIC_REASON in notes and kh.ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)
