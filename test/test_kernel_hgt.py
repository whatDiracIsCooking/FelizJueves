"""`kernel_hgt`: the kinetic-energy gradient on `Engine.HGP` (`T` at d1),
interpreted. Shells through f (`kernel_hgt.L_MAX = 3`).

THE WITNESSES, two, neither of which shares the kernel's recurrence:

- THE MMD HOST, `gradient.host.kinetic_gradient` (itself held to finite
  differences of the libcint-witnessed `T`, test/test_gradient_host.py):
  per class, one launch over every shell pair `i <= j` of that class into
  a zeroed `G`, against the host over the same pairs -- every ordered pair
  class (ss) through (ff) on HF/cc-pVTZ, `M` non-symmetric. The scale is
  test_kernel_g1e.py's split-centre scale (its `_scale`: every shell on a
  row of its own, so a one-centre pair, whose `A` and `B` terms cancel on
  one atom, is held to the size of its terms, not to their round-off).
  The bound is LOOSER than the MMD kernels' 16 eps: another engine, and
  one whose kinetic combination `(2 lb + 3) (a|b)_1 - 2 SUM_d (a|b +
  2_d)_2` and HRR on contracted values cancel (`kernel_hgp`'s own d0 test
  allows 1e-13 of max). Measured worst 37.2 eps of the scale at (sd), 35.4
  at (dd), 15.9 at (ss), so `TOL = 64 eps`; an order-unity error is ~1 of
  the scale.
- LIBCINT, through #550's RHF gradient fixtures: `-2 SUM_{m on A}
  int1e_ipkin[k, m, n] D_mn` (test_rhf_gradient.py's `kinetic` term), the
  whole gradient of H2O/STO-3G, H2O/6-31G and the d case
  `h2o_sto3g_d_grad`, one launch per class into one `G`, within `1e-12`
  absolute (the fixtures' `GRADIENT_TOL`).

And the kernel's shape: the key and name, the refusals, the parameter
block equal to the staging's names, every local read (#92's filter), the
six atomics, and translational invariance of the whole gradient.

`L` for the tier cut is `la + lb + 1`, as test_kernel_g1e.py's. Nothing
compiles and nothing launches on a card.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel_g1e import EPS, _fixture, _m, _scale
from test_rhf_gradient import _witness
from tiers import by_l

from felizjueves.gradient import host
from felizjueves.gradient import kernel_hgt as kt
from felizjueves.ir import (
    AtomicAdd,
    Call,
    Decl,
    Index,
    Intent,
    Raw,
    Var,
    Weighted,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.spec import Engine, Operator, Spec, Target

TOL = 64 * EPS  # of max(scale): the module docstring
LIBCINT_TOL = 1e-12  # absolute: manifest's GRADIENT_TOL
CLASSES = [(la, lb) for la in range(kt.L_MAX + 1) for lb in range(kt.L_MAX + 1)]
WITNESSES = ("h2o_sto3g_grad", "h2o_631g_grad", "h2o_sto3g_d_grad")


def _launch(basis, nuclei, M, la, lb, G, pairs=None, threads=3):
    arr = host.pair_records(basis, nuclei, la, lb, pairs=pairs)
    kern = kt.build_hgt_kernel((la, lb))
    want = {p.name for p in kern.params if p.array}
    arrs = {k: np.array(v, dtype=np.int32 if k.startswith(
        ("off", "atom", "rec")) else float)
        for k, v in arr.items() if k in want}
    n_pair = len(arr["rec0"])
    grid, block = kt.launch_shape(n_pair, threads)
    return Interpreter(kern, launch=Launch(grid, block, 1)).run(
        n_pair=n_pair, n_rec=len(arr["s_ab"]), n_ao=basis.n_ao,
        n_atom=len(nuclei), M=M, G=G, **arrs)["G"]


def _classes(basis) -> list[tuple[int, int]]:
    return sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                   for b in basis.shells[k:]})


# ---------------------------------------------------------------------------
# Against the host and against libcint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_gradient(cls) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    pairs = host.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    G = _launch(basis, nuclei, M, *cls, np.zeros((len(nuclei), 3)))
    want = np.array(host.kinetic_gradient(basis, M, nuclei, pairs=pairs))
    scale = _scale(Target.T, basis, nuclei, M, pairs)
    if scale == 0.0:  # every term zero by parity (a one-centre p-f pair)
        assert np.all(G == 0.0) and np.all(want == 0.0), cls
        return
    err = np.abs(G - want).max()
    assert err <= TOL * scale, f"{cls}: {err / scale / EPS:.1f} eps of scale"


@pytest.mark.parametrize("name", WITNESSES)
def test_the_whole_gradient_is_libcints(name) -> None:
    w, basis, nuclei = _witness(name)
    G = np.zeros((len(nuclei), 3))
    for la, lb in _classes(basis):
        G = _launch(basis, nuclei, w.density, la, lb, G)
    want = w.terms["kinetic"]
    assert np.abs(G - want).max() <= LIBCINT_TOL
    # translational invariance: B's row is -A's, pair by pair
    assert np.abs(G.sum(axis=0)).max() <= 1e-14 * np.abs(G).max()


def test_g_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    M = _m(basis.n_ao)
    G0 = np.arange(9.0).reshape(3, 3)
    G = _launch(basis, nuclei, M, 0, 1, G0.copy())
    want = _launch(basis, nuclei, M, 0, 1, np.zeros((3, 3)))
    assert np.abs(G - (G0 + want)).max() <= 8 * EPS * np.abs(G0).max()
    assert not np.array_equal(G, want)


# ---------------------------------------------------------------------------
# The key, the name, the refusals, the shape
# ---------------------------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert kt.hgt_spec(1, 0).key() == "kinetic/d1/(1,0)/b1/t.hgp/literal"
    assert kt.hgt_name((2, 3)) == "hgt_23"
    assert kt.build_hgt_kernel((1, 2)).name == "hgt_12"
    names = {kt.hgt_name(c) for c in CLASSES}
    assert len(names) == len(CLASSES)


def test_the_refusals() -> None:
    with pytest.raises(ValueError, match="through f only"):
        kt.build_hgt_kernel((4, 0))
    with pytest.raises(ValueError, match="not an HGP d1 T key"):
        kt.build_hgt_kernel(Spec(Operator.KINETIC, 1, (1, 0), Target.T))
    with pytest.raises(ValueError, match="not an HGP d1 T key"):
        kt.build_hgt_kernel(Spec(Operator.KINETIC, 0, (1, 0), Target.T,
                                 engine=Engine.HGP))
    # the other one-electron gradients stay refused on HGP
    with pytest.raises(ValueError, match="derivative order 1 on target s.hgp"):
        Spec(Operator.OVERLAP, 1, (1, 0), Target.S, engine=Engine.HGP)
    with pytest.raises(ValueError, match="derivative order 1 on target v.hgp"):
        Spec(Operator.NUCLEAR, 1, (1, 0), Target.V, engine=Engine.HGP)
    with pytest.raises(ValueError, match="derivative order 2 on target t.hgp"):
        Spec(Operator.KINETIC, 2, (1, 0), Target.T, engine=Engine.HGP)
    with pytest.raises(ValueError, match="positive int"):
        kt.launch_shape(0)


def test_the_parameter_block_is_the_stagings() -> None:
    basis, nuclei = _fixture("h2o_sto3g_j")
    staged = set(host.pair_records(basis, nuclei, 1, 1))
    k = kt.build_hgt_kernel((1, 1))
    arrays = {p.name for p in k.params if p.array}
    # the staging's P is the MMD V gradient's, not read here
    assert arrays - {"M", "G"} == staged - {"p_x", "p_y", "p_z"}
    assert set(kt.pair_arrays()) <= arrays
    (g,) = [p for p in k.params if p.name == "G"]
    assert g.intent is Intent.OUT


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_every_local_is_read_and_six_atomics(cls) -> None:
    k = kt.build_hgt_kernel(cls)
    stmts = list(walk(k.body))
    assert not any(isinstance(s, Raw) for s in stmts)
    assert sum(isinstance(s, AtomicAdd) for s in stmts) == 6
    read: set[str] = set()

    def scan(e) -> None:
        if isinstance(e, Var):
            read.add(e.name)
        elif isinstance(e, Index):
            read.add(e.array)
            for x in e.at:
                scan(x)
        elif isinstance(e, Call):
            for x in e.args:
                scan(x)
        elif isinstance(e, Weighted):  # a recurrence term's factor
            for t in e.terms:
                if t.g.name_in_kernel is not None:
                    read.add(t.g.name_in_kernel)
                scan(t.src)

    for s in stmts:  # test_kernel_hgp.py's scan
        for f in ("init", "value", "cond", "start", "stop", "step", "t"):
            v = getattr(s, f, None)
            if v is not None and not isinstance(v, (int, str)):
                scan(v)
        dst = getattr(s, "dst", None)
        if isinstance(dst, Index):
            scan(dst)
    declared = {s.name for s in stmts if isinstance(s, Decl)}
    assert declared <= read, sorted(declared - read)


def test_the_tables() -> None:
    # (2 alpha)^p beta^n: p = 0 absent for an s bra, n = 0 for lb <= 1
    assert set(kt.tables(0, 1)) == {(1, 1), (1, 2)}
    assert set(kt.tables(1, 1)) == {(0, 1), (0, 2), (1, 1), (1, 2)}
    assert set(kt.tables(1, 2)) == {(0, 0), (0, 1), (0, 2),
                                    (1, 0), (1, 1), (1, 2)}
