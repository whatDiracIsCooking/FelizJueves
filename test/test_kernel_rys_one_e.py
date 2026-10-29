"""`kernel_rys`: the Rys one-electron kernels, `S`, `T` and `V` on
`Engine.RYS`, interpreted. Every shell through f (`kernel_rys.L_MAX = 3`).

The rule is injected as `rys=rys.roots.roots_weights` (the device routine's
operation order; test/test_rys_roots.py holds it to a 40-digit route and
the `fj_rys` text run as host C++ to it bitwise).

THE WITNESSES, three, none of which shares the kernels' route:

- THE MMD HOST, `one_electron.overlap` / `kinetic` / `potential`'s whole
  matrices (McMurchie & Davidson's `E`, the Boys function for `V`), within
  `HOST_TOL` of the matrix's largest element. Not bitwise: another
  recurrence, another association, and a quadrature in place of the Boys
  function. On SYNTHETIC bases, contracted, on three off-axis centres under
  four nuclei (every `PA`, `AB`, `P - C` component nonzero), every ordered
  class launched once over all its shell pairs, diagonal and off-diagonal
  mixed: s, p, d (fast: every class L <= 4) and s through f (`slow`).
- THE HGP KERNELS (`kernel_hgp`, s and p), the same skeleton on another
  per-record route, within `HOST_TOL`; and their parameter blocks equal,
  so a `rs_` unit is a drop-in for the `hs_` unit of its class.
- LIBCINT, through the tracked fixtures, within `LIBCINT_TOL` of the
  largest element: `int1e_ovlp`, `int1e_kin`, `int1e_nuc` on H2O/STO-3G and
  6-31G (s and p, fast), H2O/cc-pVDZ (d) and `prim_spdf` (one primitive per
  l = 0..3, f) `slow` -- the kernels interpreted over `one_electron_launch`'s
  plan, staged by `OneEClass.hgp_arrays`.

And the kernel's shape: the key and name, the refusals (g, an HGP key, the
dipole, `SX`, a lever), the parameter block equal to the staging's names, no
atomics, a `Rys` statement in `V` alone with `N = (la + lb) // 2 + 1` nodes,
and every local read (a declared but unread local is a warning on both
compilers, #92's filter).

Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l, split_by_l

from felizjueves.ao.pairs import overlap_bound
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.j_coulomb.launch import one_electron_launch
from felizjueves.ir import (
    AtomicAdd,
    Boys,
    Call,
    Decl,
    Index,
    Rys,
    Var,
    Weighted,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.one_electron import kernel_hgp as kh
from felizjueves.one_electron import kernel_rys as kr
from felizjueves.one_electron.kinetic import kinetic_matrix
from felizjueves.one_electron.overlap import overlap_matrix
from felizjueves.one_electron.potential import potential_matrix
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.rys.roots import roots_weights
from felizjueves.spec import Engine, Operator, Spec, Target

CLASSES = [(la, lb) for la in range(kr.L_MAX + 1)
           for lb in range(kr.L_MAX + 1)]
HOST_TOL = 1e-13  # of max|M|: two routes, a few dozen roundings apart
LIBCINT_TOL = 1e-12  # of max|M|: the one-electron kernels' acceptance bound
TARGETS = [pytest.param(t, id=t.value) for t in kr.TARGETS]

A = (0.05, -0.02, 0.11)
B = (0.94, 0.13, -0.31)
C = (-0.41, 0.88, -0.27)
NUCLEI = ((7.0, A), (1.0, B), (1.0, C), (2.0, (0.37, -0.79, -0.42)))


def _synthetic(lmax: int) -> Basis:
    """Contracted and uncontracted shells of every l <= lmax on the three
    centres, mixed exponents."""
    shells: list[Shell] = []
    for ell in range(lmax + 1):
        shells += [Shell(ell, A, (5.4, 1.2, 0.31), (0.15, 0.53, 0.44)),
                   Shell(ell, B, (0.9,), (1.0,)),
                   Shell(ell, C, (2.2, 0.55), (0.4, 0.8))]
    return Basis(tuple(shells))


def _arrays(c, target, basis):
    return {k: np.asarray(v, dtype=np.int32 if k.startswith(("off", "rec"))
                          else float)
            for k, v in c.hgp_arrays(target, basis).items()}


def _build(basis, nuclei, target, *, engine=Engine.RYS, threads=3):
    """`target`'s matrix from the Rys (or HGP) kernels, interpreted over
    `one_electron_launch`'s plan (tau = 0): one launch per class."""
    plan = one_electron_launch(basis, nuclei, tau=0.0, bound=overlap_bound)
    n = basis.n_ao
    out = kh.OUT[target]
    M = np.zeros((n, n))
    for c in plan:
        extra: dict = {}
        if target is Target.V:
            extra = dict(n_nuc=len(plan.nuclei), Z=np.array(plan.Z),
                         Cnuc=np.array(plan.Cnuc))
        grid, block = kh.launch_shape(len(c.blocks), threads)
        if engine is Engine.RYS:
            k = kr.build_rys_kernel(kr.rys_spec(target, c.la, c.lb))
        else:
            k = kh.build_hgp_kernel(kh.hgp_spec(target, c.la, c.lb))
        M = Interpreter(k, launch=Launch(grid, block, 1), boys=boys,
                        rys=roots_weights).run(
            n_pair=len(c.blocks), n_rec=len(c.recs), n_ao=n,
            **_arrays(c, target, basis), **extra, **{out: M})[out]
    return M


def _host(basis, nuclei, target):
    if target is Target.S:
        return np.array(overlap_matrix(basis))
    if target is Target.T:
        return np.array(kinetic_matrix(basis))
    return np.array(potential_matrix(basis, nuclei, one_centre=one_centre))


def _rel(got, want) -> float:
    want = np.asarray(want)
    return float(np.max(np.abs(np.asarray(got) - want))
                 / np.max(np.abs(want)))


@cache
def _fixture(name):
    fx = manifest.one_e_fixture(name)
    return fx, ob.from_fixture(fx), tuple((float(a.charge), a.coord)
                                          for a in fx.atoms)


# ---------------------------------------------------------------------------
# The key, the name, the refusals
# ---------------------------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert kr.rys_spec(Target.S, 1, 0).key() == \
        "overlap/d0/(1,0)/b1/s.rys/literal"
    assert kr.rys_spec(Target.V, 2, 3).key() == \
        "nuclear/d0/(2,3)/b1/v.rys/literal"
    names = {kr.rys_name((t, la, lb)) for t in kr.TARGETS
             for la, lb in CLASSES}
    assert len(names) == 3 * len(CLASSES)
    assert kr.rys_name((Target.T, 1, 3)) == "rt_13"
    assert kr.rys_name((Target.V, 0, 0)) == "rv_00"
    for t in kr.TARGETS:
        assert kr.build_rys_kernel((t, 1, 0)).name == \
            kr.rys_name((t, 1, 0))


def test_the_refusals() -> None:
    with pytest.raises(ValueError, match="through f only"):
        kr.build_rys_kernel((Target.S, 4, 0))
    with pytest.raises(ValueError, match="not a Rys key"):
        kr.build_rys_kernel(Spec(Operator.OVERLAP, 0, (0, 0), Target.S))
    with pytest.raises(ValueError, match="not a Rys key"):
        kr.build_rys_kernel(kh.hgp_spec(Target.T, 0, 0))
    with pytest.raises(ValueError, match="engine rys on target dipole"):
        kr.rys_spec(Target.DIPOLE, 0, 0)
    with pytest.raises(ValueError, match="engine rys on target sx"):
        kr.rys_spec(Target.SX, 0, 0)
    with pytest.raises(ValueError, match="Rys one-electron kernels are"):
        kr.build_rys_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0),
                                 Target.F, engine=Engine.RYS))


@pytest.mark.parametrize("target", TARGETS)
def test_a_drop_in_for_the_hgp_unit(target) -> None:
    # The same parameter block as kernel_hgp's unit of the class, in order,
    # and every staged array a parameter.
    plan = one_electron_launch(_synthetic(1), NUCLEI, tau=0.0,
                               bound=overlap_bound)
    staged = set(next(iter(plan)).hgp_arrays(target, _synthetic(1)))
    for la, lb in ((0, 0), (0, 1), (1, 1)):
        k = kr.build_rys_kernel((target, la, lb))
        assert k.params == kh.build_hgp_kernel((target, la, lb)).params
    k = kr.build_rys_kernel((target, 2, 2))
    arrays = {p.name for p in k.params if p.array}
    assert staged == arrays - {kh.OUT[target], "Z", "Cnuc"}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("cls", split_by_l(CLASSES, sum, ids=str))
def test_no_atomics_the_rule_in_v_and_every_local_read(target, cls) -> None:
    k = kr.build_rys_kernel((target, *cls))
    stmts = list(walk(k.body))
    assert not any(isinstance(s, (AtomicAdd, Boys)) for s in stmts)
    rules = [s for s in stmts if isinstance(s, Rys)]
    if target is Target.V:
        assert [r.n for r in rules] == [sum(cls) // 2 + 1]
    else:
        assert not rules
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
        elif isinstance(e, Weighted):
            for t in e.terms:
                if t.g.name_in_kernel is not None:
                    read.add(t.g.name_in_kernel)
                scan(t.src)

    for s in stmts:
        for f in ("init", "value", "cond", "start", "stop", "step", "t"):
            v = getattr(s, f, None)
            if v is not None and not isinstance(v, (int, str)):
                scan(v)
        dst = getattr(s, "dst", None)
        if isinstance(dst, Index):
            scan(dst)
        if isinstance(s, Rys):
            read |= {s.nodes, s.weights}
    declared = {s.name for s in stmts if isinstance(s, Decl)}
    assert declared <= read, sorted(declared - read)


def test_ss_reads_no_recurrence_scalar() -> None:
    # (ss) has no 1D step and no HRR step: no PA, inv2p, node or AB. (Not
    # T: its table 2 reads (a|b+2_d), so (ss) steps twice and transfers.)
    for t in (Target.S, Target.V):
        k = kr.build_rys_kernel((t, 0, 0))
        names = {s.name for s in walk(k.body) if isinstance(s, Decl)}
        assert not names & {"PA_x", "inv2p", "rys_xr", "B10", "q_ppq",
                            "AB_x", "AB_y", "AB_z"}, names


# ---------------------------------------------------------------------------
# Against the MMD host and the HGP kernels, on synthetic off-axis bases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lmax", [by_l(2, 4, "spd"), by_l(3, 6, "spdf")])
@pytest.mark.parametrize("target", TARGETS)
def test_the_whole_matrix_is_the_mmd_hosts(target, lmax) -> None:
    basis = _synthetic(lmax)
    got = _build(basis, NUCLEI, target)
    assert _rel(got, _host(basis, NUCLEI, target)) <= HOST_TOL
    # every element written once by a plain store: symmetric by construction
    assert np.array_equal(got, got.T)


@pytest.mark.parametrize("target", TARGETS)
def test_the_whole_matrix_is_the_hgp_kernels(target) -> None:
    basis = _synthetic(kh.L_MAX)
    assert _rel(_build(basis, NUCLEI, target),
                _build(basis, NUCLEI, target, engine=Engine.HGP)) <= HOST_TOL


def test_the_far_nucleus_reaches_the_laguerre_branch() -> None:
    # A nucleus far enough that zeta |P - C|^2 crosses rys.roots.T_SWITCH
    # (the Laguerre constants) for every pair, against the host.
    basis = _synthetic(2)
    far = ((3.0, (40.0, -25.0, 31.0)),)
    assert _rel(_build(basis, far, Target.V),
                _host(basis, far, Target.V)) <= HOST_TOL


# ---------------------------------------------------------------------------
# Against libcint, through the tracked fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "h2o_sto3g_j", "h2o_631g_j", pytest.param("h2o_ccpvdz_j", marks=(
        pytest.mark.slow,)), pytest.param("prim_spdf", marks=(
            pytest.mark.slow,))])
@pytest.mark.parametrize(("target", "array"), [
    (Target.S, "int1e_ovlp"), (Target.T, "int1e_kin"),
    (Target.V, "int1e_nuc")], ids=["s", "t", "v"])
def test_s_t_v_are_libcints(name, target, array) -> None:
    fx, basis, nuclei = _fixture(name)
    assert max(s.l for s in basis.shells) <= kr.L_MAX
    assert _rel(_build(basis, nuclei, target), fx.arrays[array]) \
        <= LIBCINT_TOL
