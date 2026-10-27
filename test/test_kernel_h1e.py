"""`kernel_h1e`: the `S`, `T` and `V` hessian kernels (d2), interpreted.

THE WITNESS is `hessian.host_1e` (itself held to finite differences of the
gradient host, test/test_hessian_host_1e.py), by another route: the host
differentiates each 3-D function as a list of moved Cartesians, the
kernel each axis's factor. Per class, one launch over every record of
every shell pair `i <= j` of that class, into a zeroed `H`, against
`host.*_hessian(..., pairs=)` of the same pairs.

- PER CLASS, every ordered pair class (ss) through (ff) on HF/cc-pVTZ (the
  one fixture whose shells reach f, both orientations occurring with
  `i <= j`), `M` non-symmetric (the weight rule is exact for any `M`).
- THE WHOLE HESSIAN, one launch per class into one `H`, on H2O/STO-3G.

ROUND-OFF, NOT BITWISE: `|H_kernel - H_host| <= TOL max(scale)` with
`TOL = 16 eps`, `scale` the host's `absolute=True` twin (every factor in
`|.|`): the size of the terms before they cancel. `max|H|` is not a scale
-- on a one-centre pair (F's own p-f) the `dA dA`, `dA dB` and `dB dB`
blocks cancel exactly on one atom, so the result is round-off. Measured:
worst 0.35 eps of the scale (`d2v_12`). An order-unity error is ~1 of the
scale.

`L` for the tier cut is `la + lb + 2`, the Hermite order a hessian
reaches (test/tiers.py): (sp), (pp) and below are fast, the rest `slow`.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves.gradient import host as grad
from felizjueves.hessian import host_1e as host
from felizjueves.hessian.kernel_h1e import (
    ATOMIC_REASON,
    COMPONENTS,
    ZEROED_H,
    axis_ops,
    build_h1e_kernel,
    h1e_name,
    h1e_spec,
    launch_shape,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of max(scale): the module docstring
TARGETS = (Target.S, Target.T, Target.V)
LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


def _m(n: int) -> np.ndarray:
    return np.random.default_rng(3).standard_normal((n, n))


def _launch(target, basis, nuclei, M, la, lb, H, threads=7):
    rec = grad.records(basis, nuclei, la, lb)
    kern = build_h1e_kernel((target, la, lb))
    want = {p.name for p in kern.params if p.array}
    arrs = {k: np.array(v, dtype=np.int32 if k.startswith(("off", "atom"))
                        else float) for k, v in rec.items() if k in want}
    if target is Target.V:
        arrs["Z"] = np.array([z for z, _ in nuclei])
        arrs["Cnuc"] = np.array([c for _, c in nuclei])
    n = len(rec["s_ab"])
    grid, block = launch_shape(n, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1), boys=boys).run(
        n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), M=M, H=H,
        **arrs)["H"]


def _host(target, basis, nuclei, M, pairs=None, absolute=False
          ) -> np.ndarray:
    kw = dict(pairs=pairs, absolute=absolute)
    if target is Target.S:
        h = host.overlap_hessian(basis, M, nuclei, **kw)
    elif target is Target.T:
        h = host.kinetic_hessian(basis, M, nuclei, **kw)
    else:
        h = host.potential_hessian(basis, M, nuclei, one_centre=one_centre,
                                   **kw)
    return np.array(h)


def _close(got, want, scale: float, what: str) -> None:
    if scale == 0.0:  # every term zero by parity (a one-centre s-f overlap)
        assert np.all(got == 0.0) and np.all(want == 0.0), what
        return
    err = np.abs(got - want).max()
    assert err <= TOL * scale, f"{what}: {err / scale / EPS:.1f} eps of scale"


def _zeros(nuclei) -> np.ndarray:
    n = len(nuclei)
    return np.zeros((n, 3, n, 3))


@pytest.mark.parametrize("target", TARGETS, ids=str)
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_hessian(cls, target) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = grad.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    H = _launch(target, basis, nuclei, M, *cls, _zeros(nuclei))
    scale = _host(target, basis, nuclei, M, pairs, absolute=True).max()
    _close(H, _host(target, basis, nuclei, M, pairs), scale,
           f"{target} {cls}")


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_whole_hessian_of_h2o_is_the_host_hessian(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    H = _zeros(nuclei)
    classes = sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                      for b in basis.shells[k:]})
    for la, lb in classes:
        H = _launch(target, basis, nuclei, M, la, lb, H)
    _close(H, _host(target, basis, nuclei, M),
           _host(target, basis, nuclei, M, absolute=True).max(),
           f"{target} H2O/STO-3G")
    scale = np.abs(H).max()
    assert np.abs(H.sum(axis=0)).max() <= 1e-10 * scale
    assert np.abs(H - H.transpose(2, 3, 0, 1)).max() <= 1e-13 * scale


def test_h_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    H0 = np.arange(81.0).reshape(3, 3, 3, 3)
    H = _launch(Target.S, basis, nuclei, M, 0, 1, H0.copy())
    want = _launch(Target.S, basis, nuclei, M, 0, 1, _zeros(nuclei))
    assert np.abs(H - (H0 + want)).max() <= 8 * EPS * np.abs(H0).max()
    assert not np.array_equal(H, want)


# --- the kernel's shape -------------------------------------------------------


def test_the_key_and_the_names() -> None:
    for t, prefix in zip(TARGETS, ("d2s", "d2t", "d2v"), strict=True):
        assert h1e_spec(t, 2, 1).key().split("/")[1] == "d2"
        assert h1e_name((t, 2, 1)) == f"{prefix}_21"
        assert build_h1e_kernel(h1e_spec(t, 2, 1)).name == f"{prefix}_21"


def test_a_d0_or_d1_key_or_another_target_is_refused() -> None:
    for order in (0, 1):
        with pytest.raises(ValueError, match="not a d2 S, T or V key"):
            build_h1e_kernel(Spec(Operator.OVERLAP, order, (1, 0), Target.S))
    with pytest.raises(ValueError, match="not a d2 S, T or V key"):
        build_h1e_kernel(Spec(Operator.COULOMB, 1, (1, 0), Target.JAB))


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_the_parameters_and_the_one_output(target) -> None:
    k = build_h1e_kernel((target, 1, 1))
    names = [p.name for p in k.params]
    head = ["n_rec", "n_ao", "n_atom", "zeta_ab", "pa_x", "pa_y", "pa_z",
            "pb_x", "pb_y", "pb_z"]
    p = ["p_x", "p_y", "p_z"] if target is Target.V else []
    tail = ["s_ab", "ea_bra", "eb_ket", "off_a", "off_b", "atom_a", "atom_b",
            "M"]
    nuc = ["Z", "Cnuc"] if target is Target.V else []
    assert names == [*head, *p, *tail, *nuc, "H"]
    out = [q for q in k.params if q.intent is Intent.OUT]
    assert [q.name for q in out] == ["H"] and len(out[0].shape) == 4


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_every_write_is_an_atomic_add_into_h(target) -> None:
    k = build_h1e_kernel((target, 1, 2))
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert adds and all(a.dst.array == "H" for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_H in notes
    assert not any(isinstance(s, Raw) for s in nodes)
    # 36 centre adds per record; V adds 36 centre-nucleus and 9
    # nucleus-nucleus ones per nucleus inside its loop.
    assert len(adds) == (36 + 45 if target is Target.V else 36)


def test_the_components_and_their_axis_operators() -> None:
    assert len(COMPONENTS) == 21
    assert axis_ops("A", 0, "A", 0) == ("AA", "", "")
    assert axis_ops("A", 1, "B", 1) == ("", "AB", "")
    assert axis_ops("A", 2, "B", 0) == ("B", "", "A")
    assert axis_ops("B", 0, "A", 0) == ("AB", "", "")


def test_launch_shape_refuses_a_bad_count() -> None:
    assert launch_shape(65, 32) == (3, (32, 1))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            launch_shape(bad)
