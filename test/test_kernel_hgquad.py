"""`kernel_hgquad`: the quadrupole gradient on the HGP engine
(`hgquad_{la}{lb}`, `Target.QUADRUPOLE` at d1 on `Engine.HGP`), interpreted.

THE WITNESS is `gradient.host.multipole_gradient` at `e = 2`,
test_kernel_gmu.py's (itself held to libcint's `int1e_irrp_cart` live, to
central differences of the libcint-witnessed `multipole_matrix`, and to the
origin shift), which shares neither the kernel's recurrence (McMurchie &
Davidson's `E` and Hermite moments, not Obara & Saika's VRR, the HGP HRR
and the operator moved onto the ket) nor its contraction order (per
record, not on contracted values):

- PER CLASS, every ordered pair class (ss) through (ff) on HF/cc-pVTZ, one
  launch over every shell pair `i <= j` of the class (staged by
  `host.pair_records`) into a zeroed `G`, `M` non-symmetric, `C` off every
  nucleus, `q = -1` -- against the host over the same pairs, within `TOL =
  64 eps` of test_kernel_gmu.py's split-centre `scale` per component
  (test_kernel_hgs.py's bound and its reason: the HRR on CONTRACTED values).
  Measured (2026-11-09): worst 24.1 eps at (fd), 13.5 at (fp), every other
  class <= 10.3.
- AGAINST `gquad_`: the same classes interpreted by `kernel_gmu`, the same
  bound -- the two engines' kernels agree with each other, not only with
  the host.
- THE WHOLE GRADIENT of H2O/STO-3G and CO/cc-pVTZ (`slow`; f on both
  atoms, every class), one launch per class into one `G`, the same bound,
  and the origin shift: `C` is fixed, so `SUM_A G^{de}[A, k] = q (delta_dk
  SUM M MU^e + delta_ek SUM M MU^d)` with the libcint-witnessed `MU` --
  the kernel computes both centres, so it is a check, not an assumption.

And the kernel's shape: the key and name, the refusals, the parameter
block equal to the staging, 36 atomic adds into `G` per shell pair, the
charge an exact factor.

`L` for the tier cut is `la + lb + 1`, as test_kernel_gmu.py's. Nothing
compiles and nothing launches on a card.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_gmu import EPS, ORIGIN, Q, _fixture, _host, _m, _scale
from test_kernel_gmu import _launch as _launch_gquad
from tiers import by_l

from felizjueves.gradient import host
from felizjueves.gradient.kernel_g1e import ZEROED_G
from felizjueves.gradient.kernel_hgquad import (
    ATOMIC_REASON,
    L_MAX,
    N_COMPONENTS,
    build_hgquad_kernel,
    hgquad_name,
    hgquad_spec,
    launch_shape,
    pair_arrays,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, Scalar, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.one_electron.multipole import multipole_matrix
from felizjueves.spec import Engine, Operator, Spec, Target

CLASSES = [(la, lb) for la in range(L_MAX + 1) for lb in range(L_MAX + 1)]
TOL = 64 * EPS  # of max(scale): the module docstring
QUAD = Target.QUADRUPOLE


def _close(got, want, scale: np.ndarray, what: str) -> None:
    for c, s in enumerate(scale):
        if s == 0.0:  # every term zero by parity
            assert np.all(got[c] == 0.0) and np.all(want[c] == 0.0), what
            continue
        err = np.abs(got[c] - want[c]).max()
        assert err <= TOL * s, (
            f"{what} component {c}: {err / s / EPS:.1f} eps of scale")


def _arrays(basis, nuclei, la, lb, kern):
    rec = host.pair_records(basis, nuclei, la, lb, ket_centre=True)
    types = {p.name: p.type for p in kern.params if p.array}
    return {k: np.array(v, dtype=np.int32 if types[k] is Scalar.I32
                        else float) for k, v in rec.items() if k in types}


def launch_hgquad(basis, nuclei, M, la, lb, G, threads=3, q=Q):
    """One `hgquad_` launch over every shell pair of class `(la, lb)`."""
    kern = build_hgquad_kernel((la, lb))
    arrs = _arrays(basis, nuclei, la, lb, kern)
    n = len(arrs["rec0"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1)).run(
        n_pair=n, n_rec=len(arrs["s_ab"]), n_ao=basis.n_ao,
        n_atom=len(nuclei), M=M, G=G, C_x=ORIGIN[0], C_y=ORIGIN[1],
        C_z=ORIGIN[2], q=q, **arrs)["G"]


def _classes(basis):
    return sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                   for b in basis.shells[k:]})


def _zero(nuclei) -> np.ndarray:
    return np.zeros((N_COMPONENTS, len(nuclei), 3))


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_gradient(cls) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = host.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    G = launch_hgquad(basis, nuclei, M, *cls, _zero(nuclei))
    scale = _scale(QUAD, basis, nuclei, M, pairs)
    _close(G, _host(QUAD, basis, nuclei, M, pairs), scale, f"hgquad {cls}")
    # and the MMD kernel of the same class, interpreted
    gq = _launch_gquad(QUAD, basis, nuclei, M, *cls, _zero(nuclei))
    _close(G, gq, scale, f"hgquad {cls} against gquad")


@pytest.mark.parametrize("name", [
    pytest.param("h2o_sto3g_j", id="h2o_sto3g"),
    pytest.param("co_ccpvtz_j", id="co_ccpvtz", marks=(pytest.mark.slow,))])
def test_the_whole_gradient_is_the_host_gradient(name) -> None:
    basis, nuclei = _fixture(name)
    M = _m(basis.n_ao)
    G = _zero(nuclei)
    for la, lb in _classes(basis):
        G = launch_hgquad(basis, nuclei, M, la, lb, G)
    pairs = [(i, j) for i in range(len(basis.shells))
             for j in range(i, len(basis.shells))]
    _close(G, _host(QUAD, basis, nuclei, M),
           _scale(QUAD, basis, nuclei, M, pairs), f"hgquad {name}")
    # the origin shift (module docstring), against the witnessed MU
    mu = np.einsum("mn,cmn->c", M,
                   np.array(multipole_matrix(basis, ORIGIN, 1, 1.0)))
    want = np.zeros((N_COMPONENTS, 3))
    comps = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for c, (a, b) in enumerate(comps):
        want[c, a] += Q * mu[b]
        want[c, b] += Q * mu[a]
    err = np.abs(G.sum(axis=1) - want).max()
    assert err <= 1e-13 * np.abs(G).max(), f"origin shift: {err:.2e}"


def test_g_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G0 = np.arange(54.0).reshape(N_COMPONENTS, 3, 3)
    G = launch_hgquad(basis, nuclei, M, 0, 1, G0.copy())
    want = launch_hgquad(basis, nuclei, M, 0, 1, _zero(nuclei))
    assert np.abs(G - (G0 + want)).max() <= 4 * EPS * np.abs(G0).max()
    assert not np.array_equal(G, want)


def test_the_charge_is_an_exact_factor() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    up = launch_hgquad(basis, nuclei, M, 1, 1, _zero(nuclei), q=1.0)
    dn = launch_hgquad(basis, nuclei, M, 1, 1, _zero(nuclei), q=-1.0)
    assert np.array_equal(dn, -up) and np.abs(up).max() > 0.0


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    assert hgquad_spec(2, 1).key() == (
        "quadrupole/d1/(2,1)/b1/quadrupole.hgp/literal")
    assert hgquad_name((2, 1)) == "hgquad_21"
    assert build_hgquad_kernel(hgquad_spec(2, 1)).name == "hgquad_21"
    assert len({hgquad_name(c) for c in CLASSES}) == len(CLASSES)


@pytest.mark.parametrize("spec", [
    Spec(Operator.QUADRUPOLE, 0, (1, 0), Target.QUADRUPOLE,
         engine=Engine.HGP),
    Spec(Operator.QUADRUPOLE, 1, (1, 0), Target.QUADRUPOLE),
    Spec(Operator.OVERLAP, 1, (1, 0), Target.S, engine=Engine.HGP),
], ids=lambda s: s.key())
def test_another_key_is_refused(spec) -> None:
    with pytest.raises(ValueError, match="not a d1 HGP quadrupole key"):
        build_hgquad_kernel(spec)


def test_above_f_is_refused() -> None:
    with pytest.raises(ValueError, match="through f only"):
        build_hgquad_kernel((L_MAX + 1, 0))


def test_the_spec_admits_the_quadrupole_on_hgp_at_order_one_only() -> None:
    d0 = Spec(Operator.QUADRUPOLE, 0, (1, 0), Target.QUADRUPOLE,
              engine=Engine.HGP)
    assert hgquad_spec(1, 0).key() == d0.key().replace("/d0/", "/d1/")
    # the quadrupole on HGP at d2 stays refused
    with pytest.raises(ValueError, match="derivative order 2 on target "
                                         "quadrupole.hgp"):
        Spec(Operator.QUADRUPOLE, 2, (1, 0), Target.QUADRUPOLE,
             engine=Engine.HGP)


def test_the_parameters_are_the_staging() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    k = build_hgquad_kernel((1, 1))
    names = [p.name for p in k.params]
    assert names == [
        "n_pair", "n_rec", "n_ao", "n_atom", "zeta_ab", "pa_x", "pa_y",
        "pa_z", "pb_x", "pb_y", "pb_z", "s_ab", "ea_bra", "eb_ket",
        *pair_arrays(), "M", "C_x", "C_y", "C_z", "q", "G"]
    out = [q for q in k.params if q.intent is Intent.OUT]
    assert [q.name for q in out] == ["G"]
    assert out[0].shape[0].value == N_COMPONENTS == 6
    staged = set(host.pair_records(basis, nuclei, 1, 1, ket_centre=True))
    assert {p.name for p in k.params if p.array} - {"M", "G"} <= staged
    assert not {"b_x", "b_y", "b_z"} & set(host.pair_records(
        basis, nuclei, 1, 1))  # opt-in: the other HGP d1 kernels take none


def test_the_staged_b_is_the_ket_centre() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    pairs = host.class_pairs(basis, 0, 1)
    rec = host.pair_records(basis, nuclei, 0, 1, pairs=pairs, ket_centre=True)
    for n, (_, j) in enumerate(pairs):
        assert tuple(rec[f"b_{a}"][n] for a in "xyz") == \
            basis.shells[j].centre


@pytest.mark.parametrize("cls", [(0, 0), (1, 2), (3, 3)], ids=str)
def test_every_write_is_an_atomic_add_into_g(cls) -> None:
    k = build_hgquad_kernel(cls)
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 * N_COMPONENTS
    assert all(a.dst.array == "G" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_G in notes
    assert not any(isinstance(s, Raw) for s in nodes)


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
