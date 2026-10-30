"""`kernel_hgp`: the HGP one-electron kernels, `S`, `T`, `V`, the dipole and
the quadrupole on `Engine.HGP`, interpreted. `S`, `T`, `V` for s and p shells
(`kernel_hgp.L_MAX = 1`), the two moments through f
(`kernel_hgp.MOMENT_L_MAX = 3`).

THE WITNESSES, two, neither of which shares the kernels' recurrence:

- THE MMD HOST, `one_electron.overlap` / `kinetic` / `potential` /
  `multipole`'s whole matrices (McMurchie & Davidson's `E`, the other engine),
  within `HOST_TOL` of the matrix's largest element. Not bitwise: another
  recurrence, another association (kernel_hgp's docstring, "ROUND-OFF").
  On a SYNTHETIC basis of s and p shells, contracted, on three off-axis
  centres under four nuclei (every `PA`, `AB`, `P - C` component nonzero),
  every ordered class (ss) (sp) (ps) (pp) launched once over all its shell
  pairs, diagonal and off-diagonal mixed.
- LIBCINT, through the tracked fixtures, within `LIBCINT_TOL` of the
  largest element: `int1e_ovlp`, `int1e_kin`, `int1e_nuc` on H2O/STO-3G and
  6-31G (all s and p), and `int1e_r_cart` at the fixtures' non-zero origin
  and at 0 -- the kernels interpreted over `one_electron_launch`'s plan,
  staged by `OneEClass.hgp_arrays`.

THE MOMENTS ABOVE p. The dipole and the quadrupole (raw second moments,
xx xy xz yy yz zz) are held, through (dd) on the fast tier and (ff) on the
slow one (`tiers.FAST_L_MAX`), to the MMD host's `multipole_matrix(e=1, 2)`
on a synthetic s/p/d(/f) off-axis basis within `HOST_TOL`, and to libcint:
the dipole's `int1e_r_cart` on `hf_ccpvtz_spd_dipole` (through d) and
`hf_ccpvtz_dipole` (through f), the quadrupole's `int1e_rr_cart` on
`hf_ccpvtz_spd_quadrupole` (through d; no f witness is tracked), both at
the fixtures' origin and at 0, within `LIBCINT_TOL`. And the quadrupole's
origin shift, every term an HGP kernel.

And the kernel's shape: the key and name, the refusals (d, an MMD key, `SX`,
a lever), the parameter block equal to the staging's names, no atomics, the
recurrence statements `hgp.recurrence`'s own, and every local read (a
declared but unread local is a warning on both compilers, #92's filter).

Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves.ao.pairs import overlap_bound
from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.j_coulomb.launch import one_electron_launch
from felizjueves.ir import (
    AtomicAdd,
    Boys,
    Call,
    Decl,
    Index,
    Var,
    Weighted,
    walk,
)
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.one_electron import kernel_hgp as kh
from felizjueves.one_electron.kinetic import kinetic_matrix
from felizjueves.one_electron.multipole import multipole_matrix
from felizjueves.one_electron.overlap import overlap_matrix
from felizjueves.one_electron.potential import potential_matrix
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Engine, Operator, Spec, Target

CLASSES = [(la, lb) for la in range(2) for lb in range(2)]
HOST_TOL = 1e-13  # of max|M|: two engines, a few dozen roundings apart
LIBCINT_TOL = 1e-12  # of max|M|: the one-electron kernels' acceptance bound
H2O = ("h2o_sto3g", "h2o_631g")
ZERO = (0.0, 0.0, 0.0)

A = (0.05, -0.02, 0.11)
B = (0.94, 0.13, -0.31)
C = (-0.41, 0.88, -0.27)
SYNTHETIC = Basis((
    Shell(0, A, (5.4, 1.2, 0.31), (0.15, 0.53, 0.44)),
    Shell(1, A, (1.9, 0.45), (0.38, 0.71)),
    Shell(0, B, (1.3, 0.21), (0.6, 0.47)),
    Shell(1, B, (0.9,), (1.0,)),
    Shell(1, C, (2.2, 0.55), (0.4, 0.8)),
    Shell(0, C, (0.7,), (1.0,)),
))
NUCLEI = ((7.0, A), (1.0, B), (1.0, C), (2.0, (0.37, -0.79, -0.42)))
ORIGIN = (0.3, -0.7, 1.1)
MOMENTS = [t.value for t in kh.MOMENTS]


def _with(*more: Shell) -> Basis:
    """`SYNTHETIC` with `more` appended: d (and f) shells for the moments."""
    return Basis((*SYNTHETIC.shells, *more))


SYNTHETIC_D = _with(Shell(2, B, (1.1, 0.35), (0.5, 0.6)),
                    Shell(2, C, (0.8,), (1.0,)))
SYNTHETIC_F = _with(Shell(2, B, (1.1, 0.35), (0.5, 0.6)),
                    Shell(3, A, (0.9, 0.3), (0.45, 0.7)),
                    Shell(3, C, (0.6,), (1.0,)))


def _arrays(c, target, basis):
    return {k: np.asarray(v, dtype=np.int32 if k.startswith(("off", "rec"))
                          else float)
            for k, v in c.hgp_arrays(target, basis).items()}


def _build(basis, nuclei, target, C=ORIGIN, q=1.0, threads=3):
    """`target`'s matrix from the HGP kernels, interpreted over
    `one_electron_launch`'s plan (tau = 0): one launch per class."""
    plan = one_electron_launch(basis, nuclei, tau=0.0, bound=overlap_bound)
    n = basis.n_ao
    out = kh.OUT[target]
    nc = kh.n_components(target)
    M = np.zeros((nc, n, n) if target in kh.MOMENTS else (n, n))
    for c in plan:
        extra: dict = {}
        if target is Target.V:
            extra = dict(n_nuc=len(plan.nuclei), Z=np.array(plan.Z),
                         Cnuc=np.array(plan.Cnuc))
        if target in kh.MOMENTS:
            extra = dict(C_x=float(C[0]), C_y=float(C[1]), C_z=float(C[2]),
                         q=float(q))
        grid, block = kh.launch_shape(len(c.blocks), threads)
        k = kh.build_hgp_kernel(kh.hgp_spec(target, c.la, c.lb))
        M = Interpreter(k, launch=Launch(grid, block, 1), boys=boys).run(
            n_pair=len(c.blocks), n_rec=len(c.recs), n_ao=n,
            **_arrays(c, target, basis), **extra, **{out: M})[out]
    return M


def _host(basis, nuclei, target, C=ORIGIN, q=1.0):
    if target is Target.S:
        return np.array(overlap_matrix(basis))
    if target is Target.T:
        return np.array(kinetic_matrix(basis))
    if target is Target.V:
        return np.array(potential_matrix(basis, nuclei, one_centre=one_centre))
    e = 2 if target is Target.QUADRUPOLE else 1
    return np.array(multipole_matrix(basis, C, e=e, q=q))


def _rel(got, want) -> float:
    want = np.asarray(want)
    return float(np.max(np.abs(np.asarray(got) - want))
                 / np.max(np.abs(want)))


@cache
def _h2o(name):
    fx = manifest.one_e_fixture(f"{name}_j")
    return fx, ob.from_fixture(fx), tuple((float(a.charge), a.coord)
                                          for a in fx.atoms)


# ---------------------------------------------------------------------------
# The key, the name, the refusals
# ---------------------------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert kh.hgp_spec(Target.S, 1, 0).key() == \
        "overlap/d0/(1,0)/b1/s.hgp/literal"
    assert kh.hgp_spec(Target.DIPOLE, 0, 1).key() == \
        "dipole/d0/(0,1)/b1/dipole.hgp/literal"
    assert kh.hgp_spec(Target.QUADRUPOLE, 2, 3).key() == \
        "quadrupole/d0/(2,3)/b1/quadrupole.hgp/literal"
    names = {kh.hgp_name((t, la, lb)) for t in kh.TARGETS for la, lb in CLASSES}
    assert len(names) == 20
    assert kh.hgp_name((Target.T, 1, 1)) == "ht_11"
    assert kh.hgp_name((Target.DIPOLE, 0, 1)) == "hmu_01"
    assert kh.hgp_name((Target.QUADRUPOLE, 3, 2)) == "hquad_32"
    assert [kh.l_max(t) for t in kh.TARGETS] == [1, 1, 1, 3, 3]
    assert [kh.n_components(t) for t in kh.TARGETS] == [1, 1, 1, 3, 6]
    for t in kh.TARGETS:
        k = kh.build_hgp_kernel((t, 1, 0))
        assert k.name == kh.hgp_name((t, 1, 0))


def test_the_refusals() -> None:
    with pytest.raises(ValueError, match="s and p shells only"):
        kh.build_hgp_kernel((Target.S, 2, 0))
    for t in kh.MOMENTS:
        with pytest.raises(ValueError, match="shells through f only"):
            kh.hgp_name((t, 4, 0))
    with pytest.raises(ValueError, match="not an HGP key"):
        kh.build_hgp_kernel(Spec(Operator.OVERLAP, 0, (0, 0), Target.S))
    with pytest.raises(ValueError, match="engine hgp on target sx"):
        kh.hgp_spec(Target.SX, 0, 0)
    with pytest.raises(ValueError, match="HGP one-electron kernels are"):
        kh.build_hgp_kernel(Spec(Operator.COULOMB, 0, (0, 0, 0, 0),
                                 Target.J, engine=Engine.HGP))
    with pytest.raises(ValueError, match="positive int"):
        kh.launch_shape(0)


@pytest.mark.parametrize("target", kh.TARGETS, ids=lambda t: t.value)
def test_the_parameter_block_is_the_stagings(target) -> None:
    # Every array `OneEClass.hgp_arrays` stages is a parameter of the kernel,
    # and every array parameter but the output and the nuclei is staged.
    plan = one_electron_launch(SYNTHETIC, NUCLEI, tau=0.0, bound=overlap_bound)
    staged = set(next(iter(plan)).hgp_arrays(target, SYNTHETIC))
    k = kh.build_hgp_kernel((target, 1, 1))
    arrays = {p.name for p in k.params if p.array}
    assert staged == arrays - {kh.OUT[target], "Z", "Cnuc"}
    assert set(kh.pair_arrays(target)) <= staged


@pytest.mark.parametrize("target", kh.TARGETS, ids=lambda t: t.value)
@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_no_atomics_and_every_local_is_read(target, cls) -> None:
    k = kh.build_hgp_kernel((target, *cls))
    stmts = list(walk(k.body))
    assert not any(isinstance(s, AtomicAdd) for s in stmts)
    assert any(isinstance(s, Boys) for s in stmts) == (target is Target.V)
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
        if isinstance(s, Boys):
            read.add(s.dst)
    declared = {s.name for s in stmts if isinstance(s, Decl)}
    assert declared <= read, sorted(declared - read)


def test_ss_reads_no_recurrence_scalar() -> None:
    # (ss) has no VRR step and no HRR step: no PA, inv2p or AB is declared.
    for t in (Target.S, Target.V):
        k = kh.build_hgp_kernel((t, 0, 0))
        names = {s.name for s in walk(k.body) if isinstance(s, Decl)}
        assert not names & {"PA_x", "inv2p", "AB_x", "AB_y", "AB_z"}, names


# ---------------------------------------------------------------------------
# Against the MMD host, on the synthetic off-axis basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", kh.TARGETS, ids=lambda t: t.value)
@pytest.mark.parametrize("cls", [(0, 0), (1, 1), (1, 0)], ids=str)
def test_the_vrr_reaches_the_operators_ket_shift(target, cls) -> None:
    # S and V read up to (a|b); MU also (a|b+1_d); T and Q (a|b+2): the
    # VRR's top is la + lb plus the shift the operator puts on the ket.
    _, bounds = kh.contracted_tables(target, *cls)
    e = {Target.DIPOLE: 1, Target.QUADRUPOLE: 2, Target.T: 2}.get(target, 0)
    assert max(sum(b) for bnd in bounds.values() for b in bnd) == sum(cls) + e


@pytest.mark.parametrize("target", kh.TARGETS, ids=lambda t: t.value)
def test_the_whole_matrix_is_the_mmd_hosts(target) -> None:
    got = _build(SYNTHETIC, NUCLEI, target)
    want = _host(SYNTHETIC, NUCLEI, target)
    assert _rel(got, want) <= HOST_TOL
    # every element written once by a plain store: the result is symmetric
    # by construction (each pair's block and its mirror are one value)
    assert np.array_equal(got, np.swapaxes(got, -1, -2))


@pytest.mark.parametrize("case", [
    by_l((t, SYNTHETIC_D), 4, id=f"{t}-d") for t in MOMENTS] + [
    by_l((t, SYNTHETIC_F), 6, id=f"{t}-f") for t in MOMENTS])
def test_the_moments_above_p_are_the_mmd_hosts(case) -> None:
    target, basis = Target(case[0]), case[1]
    got = _build(basis, (), target)
    assert _rel(got, _host(basis, (), target)) <= HOST_TOL
    assert np.array_equal(got, np.swapaxes(got, -1, -2))


@pytest.mark.parametrize("target", MOMENTS)
def test_the_charge_is_an_exact_factor(target) -> None:
    target = Target(target)
    one = _build(SYNTHETIC_D, (), target, q=1.0)
    assert np.array_equal(_build(SYNTHETIC_D, (), target, q=-1.0), -one)


def test_the_origin_shift() -> None:
    # MU(C') = MU(C) + q (C - C') S
    S = _build(SYNTHETIC, (), Target.S)
    a = _build(SYNTHETIC, (), Target.DIPOLE, C=ORIGIN)
    b = _build(SYNTHETIC, (), Target.DIPOLE, C=ZERO)
    for d in range(3):
        assert _rel(b[d], a[d] + ORIGIN[d] * S) <= HOST_TOL


def test_the_quadrupoles_origin_shift() -> None:
    # Q_de(C') = Q_de(C) + delta_d MU_e(C) + delta_e MU_d(C)
    #            + delta_d delta_e S,  delta = C - C': every term HGP's but
    # S (the HGP S stops at p), which is the MMD host's
    S = _host(SYNTHETIC_D, (), Target.S)
    mu = _build(SYNTHETIC_D, (), Target.DIPOLE, C=ORIGIN)
    a = _build(SYNTHETIC_D, (), Target.QUADRUPOLE, C=ORIGIN)
    b = _build(SYNTHETIC_D, (), Target.QUADRUPOLE, C=ZERO)
    for n, (d, e) in enumerate(kh._QUAD):
        want = (a[n] + ORIGIN[d] * mu[e] + ORIGIN[e] * mu[d]
                + ORIGIN[d] * ORIGIN[e] * S)
        assert _rel(b[n], want) <= HOST_TOL, (d, e)


# ---------------------------------------------------------------------------
# Against libcint, through the tracked fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", H2O)
@pytest.mark.parametrize(("target", "array"), [
    (Target.S, "int1e_ovlp"), (Target.T, "int1e_kin"),
    (Target.V, "int1e_nuc")], ids=["s", "t", "v"])
def test_s_t_v_are_libcints(name, target, array) -> None:
    fx, basis, nuclei = _h2o(name)
    assert max(s.l for s in basis.shells) <= kh.L_MAX
    assert _rel(_build(basis, nuclei, target), fx.arrays[array]) \
        <= LIBCINT_TOL


@pytest.mark.parametrize("name", H2O)
def test_the_dipole_is_libcints_int1e_r_cart(name) -> None:
    p = manifest.dipole_pair(name)
    basis = ob.from_fixture(p.dipole)
    assert max(s.l for s in basis.shells) <= kh.L_MAX
    for C, want in ((p.origin, p.r), (ZERO, p.r0)):
        assert _rel(_build(basis, (), Target.DIPOLE, C=C), want) \
            <= LIBCINT_TOL, tuple(C)


@pytest.mark.parametrize("case", [
    by_l(manifest.SPD_DIPOLE, 4, id="spd"),
    by_l(manifest.F_DIPOLE, 6, id="f")])
def test_the_dipole_above_p_is_libcints_int1e_r_cart(case) -> None:
    d = manifest.dipole_integrals(case)
    basis = ob.from_fixture(d.fixture)
    assert max(s.l for s in basis.shells) <= kh.MOMENT_L_MAX
    for C, want in ((d.origin, d.r), (ZERO, d.r0)):
        assert _rel(_build(basis, (), Target.DIPOLE, C=C), want) \
            <= LIBCINT_TOL, tuple(C)


def test_the_quadrupole_is_libcints_int1e_rr_cart() -> None:
    # hf_ccpvtz_spd_quadrupole: the six upper-triangle components of
    # int1e_rr_cart, xx xy xz yy yz zz, at the parent's origin and at 0
    w = manifest.quadrupole_integrals()
    basis = ob.from_fixture(w.parent.fixture)
    for C, want in ((w.origin, w.rr), (ZERO, w.rr0)):
        assert _rel(_build(basis, (), Target.QUADRUPOLE, C=C), want) \
            <= LIBCINT_TOL, tuple(C)
