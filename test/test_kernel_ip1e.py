"""`kernel_ip1e`: the one-electron derivative MATRICES (#687) -- `S`, `T`,
`V` at d1 in the ip1 form (`ips_`, `ipt_`, `ipv_`) and `V`'s per-nucleus
operator term (`iprv_`), interpreted -- and their host reference,
`gradient.host`'s `overlap_ip1`, `kinetic_ip1`, `potential_ip1` and
`potential_rinv`.

THE HOST AGAINST LIBCINT, through #550's gradient fixtures (H2O/STO-3G and
H2O/cc-pVDZ, Cartesian): `X1 = -int1e_ipovlp`, `-int1e_ipkin`,
`-int1e_ipnuc` (libcint differentiates in the electron's coordinate), and
nucleus `C`'s operator term `-Z_C (iprinv_C + iprinv_C^T)`, element by
element, within `LIBCINT_TOL = 1e-12` of the array's `max|.|` (measured:
<= 2.8e-15 on cc-pVDZ, `V` worst; the fixture is libcint's arithmetic in
another association). Not a finite difference: the fixture IS the
derivative integrals.

THE KERNEL AGAINST THE HOST, per class: one launch over every record of
every shell pair `i <= j` of that class into a zeroed matrix, against the
host over the same pairs (`pairs=`). Every ordered pair class (ss) through
(ff) on HF/cc-pVTZ (the one fixture whose shells reach f, both
orientations occurring with `i <= j`), every form, `iprv` at each nucleus;
then the WHOLE matrices of H2O/STO-3G and cc-pVDZ, one launch per class,
against libcint. Within `TOL = 16 eps` of the class's `max|X_host|` (no
contraction here, so no cancellation for a scale to hide: an entry is one
row of one block, summed over its pair's primitives). Measured: `ips` and
`ipt` bitwise; `ipv` and `iprv` within 2 eps.

INVARIANCE (the issue's check): for `V`, `X1 + X1^T + SUM_C XC(C) = 0` --
the derivative of `V` over every centre and every nucleus together is the
derivative of a rigid translation -- on the host and on the kernel's
matrices, within `INV_TOL = 1e-13` of `max|X1|` (measured: 3.2e-16 on
cc-pVDZ). `S` and `T` have no operator term, so for them the identity is
`X1 + X1^T = 0` alone (both centres moved together), held on the host.

`L` for the tier cut is `la + lb + 1`, the Hermite order a d1 kernel
reaches (test/tiers.py, as test_kernel_g1e.py): (sd), (pp) and below are
fast, the rest `slow`. Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from tiers import by_l

from felizjueves import levers
from felizjueves.gradient import host
from felizjueves.hessian.kernel_ip1e import (
    ATOMIC_REASON,
    IP1E,
    LEVER,
    ZEROED_X,
    build_ip1e_kernel,
    ip1e_name,
    ip1e_spec,
    launch_shape,
    out_array,
)
from felizjueves.ir import AtomicAdd, Intent, Note, Raw, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Engine, Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS  # of the class's max|X_host| (module docstring)
LIBCINT_TOL = 1e-12  # of max|libcint| (module docstring)
INV_TOL = 1e-13  # of max|X1| (module docstring)
LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
H2O = ("h2o_sto3g_grad", "h2o_ccpvdz_grad")
_IDS = {(Target.S, "ip1"): "ips", (Target.T, "ip1"): "ipt",
        (Target.V, "ip1"): "ipv", (Target.V, "rinv"): "iprv"}


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


@cache
def _gradient_fixture(name: str):
    return manifest.gradient_fixture(name)


def _classes(basis) -> list[tuple[int, int]]:
    return sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                   for b in basis.shells[k:]})


def launch(target, form, basis, nuclei, la, lb, X, *, nuc=None, threads=7
           ) -> np.ndarray:
    """One launch of the class's kernel over every record of the class,
    interpreted, adding into `X` (`[3, n_ao, n_ao]`)."""
    rec = host.records(basis, nuclei, la, lb)
    kern = build_ip1e_kernel((target, la, lb, form))
    want = {p.name for p in kern.params if p.array}
    arrs = {k: np.array(v, dtype=np.int32 if k.startswith("off") else float)
            for k, v in rec.items() if k in want}
    scalars = {}
    if target is Target.V:
        arrs["Z"] = np.array([z for z, _ in nuclei])
        arrs["Cnuc"] = np.array([c for _, c in nuclei])
        scalars["n_atom"] = len(nuclei)
    if form == "rinv":
        scalars["nuc"] = nuc
    n = len(rec["s_ab"])
    grid, block = launch_shape(n, threads)
    out = out_array((target, la, lb, form))
    return Interpreter(kern, launch=Launch(grid, block, 1), boys=boys).run(
        n_rec=n, n_ao=basis.n_ao, **{out: X}, **scalars, **arrs)[out]


def whole(target, form, basis, nuclei, *, nuc=None) -> np.ndarray:
    """Every class of `basis` launched once into one zeroed matrix."""
    X = np.zeros((3, basis.n_ao, basis.n_ao))
    for la, lb in _classes(basis):
        X = launch(target, form, basis, nuclei, la, lb, X, nuc=nuc)
    return X


def _host(target, form, basis, nuclei, *, nuc=None, pairs=None
          ) -> np.ndarray:
    if form == "rinv":
        X = host.potential_rinv(basis, nuclei, nuc, one_centre=one_centre,
                                pairs=pairs)
    elif target is Target.S:
        X = host.overlap_ip1(basis, pairs=pairs)
    elif target is Target.T:
        X = host.kinetic_ip1(basis, pairs=pairs)
    else:
        X = host.potential_ip1(basis, nuclei, one_centre=one_centre,
                               pairs=pairs)
    return np.array(X)


def _libcint(target, form, name: str, nuc=None) -> np.ndarray:
    fx = _gradient_fixture(name)
    a = fx.arrays
    if form == "rinv":
        ip = a["int1e_iprinv"][nuc]
        return -fx.atoms[nuc].charge * (ip + ip.transpose(0, 2, 1))
    return -a[{Target.S: "int1e_ipovlp", Target.T: "int1e_ipkin",
               Target.V: "int1e_ipnuc"}[target]]


def _close(got, want, scale: float, tol: float, what: str) -> None:
    err = float(np.abs(got - want).max())
    assert err <= tol * scale, f"{what}: {err / scale:.3g} of the scale"


def _cases():
    """Every (form, nucleus) case: one per ip1 target, one per nucleus of
    the molecule for rinv (H2O: three)."""
    for target, form in IP1E:
        for nuc in (range(3) if form == "rinv" else (None,)):
            yield pytest.param(target, form, nuc,
                               id=_IDS[target, form]
                               + ("" if nuc is None else f"-C{nuc}"))


# ---------------------------------------------------------------------------
# The host against libcint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [by_l(n, 1 + 2 * (1 if "sto3g" in n
                                                       else 2), id=n)
                                  for n in H2O])
@pytest.mark.parametrize(("target", "form", "nuc"), list(_cases()))
def test_the_host_matrix_is_libcints(target, form, nuc, name) -> None:
    basis, nuclei = _fixture(name)
    want = _libcint(target, form, name, nuc)
    got = _host(target, form, basis, nuclei, nuc=nuc)
    _close(got, want, float(np.abs(want).max()), LIBCINT_TOL,
           f"{name} {_IDS[target, form]}")


@pytest.mark.parametrize("name", [by_l(n, 1 + 2 * (1 if "sto3g" in n
                                                       else 2), id=n)
                                  for n in H2O])
def test_the_host_v_is_translationally_invariant(name) -> None:
    basis, nuclei = _fixture(name)
    X1 = _host(Target.V, "ip1", basis, nuclei)
    tot = X1 + X1.transpose(0, 2, 1)
    for c in range(len(nuclei)):
        tot += _host(Target.V, "rinv", basis, nuclei, nuc=c)
    assert np.abs(tot).max() <= INV_TOL * np.abs(X1).max()


@pytest.mark.parametrize("target", [Target.S, Target.T], ids=str)
def test_the_host_s_and_t_need_no_operator_term(target) -> None:
    basis, nuclei = _fixture("h2o_sto3g_grad")
    X1 = _host(target, "ip1", basis, nuclei)
    tot = X1 + X1.transpose(0, 2, 1)
    assert np.abs(tot).max() <= INV_TOL * np.abs(X1).max()


def test_the_rinv_nucleus_is_checked() -> None:
    basis, nuclei = _fixture("h2o_sto3g_grad")
    for bad in (-1, 3, 1.0, None):
        with pytest.raises(ValueError, match="nucleus"):
            host.potential_rinv(basis, nuclei, bad, one_centre=one_centre)


# ---------------------------------------------------------------------------
# The kernel against the host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("target", "form", "nuc"), [
    p for p in _cases() if p.values[2] in (None, 0)])
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_class_is_the_host_matrix(cls, target, form,
                                                      nuc) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    la, lb = cls
    pairs = host.class_pairs(basis, la, lb)
    assert pairs, cls
    X = np.zeros((3, basis.n_ao, basis.n_ao))
    got = launch(target, form, basis, nuclei, la, lb, X, nuc=nuc)
    want = _host(target, form, basis, nuclei, nuc=nuc, pairs=pairs)
    scale = float(np.abs(want).max())
    if scale == 0.0:  # every entry zero by parity: F's one-centre (pf), (ff)
        assert np.all(got == 0.0), cls
        return
    _close(got, want, scale, TOL, f"{_IDS[target, form]}_{la}{lb}")


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_rinv_at_the_other_nucleus_is_the_host_matrix(cls) -> None:
    basis, nuclei = _fixture("hf_ccpvtz")
    la, lb = cls
    pairs = host.class_pairs(basis, la, lb)
    X = np.zeros((3, basis.n_ao, basis.n_ao))
    got = launch(Target.V, "rinv", basis, nuclei, la, lb, X, nuc=1)
    want = _host(Target.V, "rinv", basis, nuclei, nuc=1, pairs=pairs)
    _close(got, want, float(np.abs(want).max()), TOL, f"iprv_{la}{lb} C1")


@pytest.mark.parametrize("name", [by_l(n, 1 + 2 * (1 if "sto3g" in n
                                                       else 2), id=n)
                                  for n in H2O])
@pytest.mark.parametrize(("target", "form", "nuc"), list(_cases()))
def test_the_whole_matrix_is_libcints(target, form, nuc, name) -> None:
    basis, nuclei = _fixture(name)
    want = _libcint(target, form, name, nuc)
    got = whole(target, form, basis, nuclei, nuc=nuc)
    _close(got, want, float(np.abs(want).max()), LIBCINT_TOL,
           f"{name} {_IDS[target, form]}")


def test_the_kernels_v_is_translationally_invariant() -> None:
    basis, nuclei = _fixture("h2o_sto3g_grad")
    X1 = whole(Target.V, "ip1", basis, nuclei)
    tot = X1 + X1.transpose(0, 2, 1)
    for c in range(len(nuclei)):
        tot += whole(Target.V, "rinv", basis, nuclei, nuc=c)
    assert np.abs(tot).max() <= INV_TOL * np.abs(X1).max()


def test_x_is_added_to_not_overwritten() -> None:
    basis, nuclei = _fixture("h2o_sto3g_grad")
    once = whole(Target.S, "ip1", basis, nuclei)
    X = once.copy()
    for la, lb in _classes(basis):
        X = launch(Target.S, "ip1", basis, nuclei, la, lb, X)
    # the second pass adds onto the first's sums: round-off, not bitwise
    assert np.abs(X - 2.0 * once).max() <= 4 * EPS * np.abs(once).max()


# ---------------------------------------------------------------------------
# The key, the names, the refusals, the shape
# ---------------------------------------------------------------------------


def test_the_key_and_the_names() -> None:
    assert ip1e_spec(Target.S, 2, 1).key() == (
        "overlap/d1/(2,1)/b1/s/uncontracted")
    assert ip1e_spec(Target.V, 0, 1, "rinv").key() == (
        "nuclear/d1/(0,1)/b1/v/rinv")
    for (t, form), prefix in _IDS.items():
        assert ip1e_name((t, 2, 1, form)) == f"{prefix}_21"
        assert build_ip1e_kernel(ip1e_spec(t, 2, 1, form)).name == (
            f"{prefix}_21")
    # The contraction's key is untouched: no lever, `literal`.
    assert Spec(Operator.OVERLAP, 1, (2, 1), Target.S).key() == (
        "overlap/d1/(2,1)/b1/s/literal")


def test_the_prefixes_are_the_registry_s_and_collide_with_none() -> None:
    from registry import UNITS
    from registry import _unit as unit
    assert {UNITS[p] for p in ("ips", "ipt", "ipv", "iprv")} == {
        ("mmd", "s", 1), ("mmd", "t", 1), ("mmd", "v", 1)}
    # #700's `ipmetric_` stays the metric's, not read as one of these.
    assert unit("ipmetric_12") == (("mmd", "metric", 1), 3)
    assert unit("iprv_12") == (("mmd", "v", 1), 3)


def test_each_form_s_lever_is_legal_on_its_targets_at_d1_only() -> None:
    assert LEVER == {"ip1": "uncontracted", "rinv": "rinv"}
    for target, form in IP1E:
        assert ip1e_spec(target, 1, 1, form).levers == (LEVER[form],)
    assert {"s", "t", "v"} <= levers.VOCABULARY["uncontracted"].legal
    assert levers.VOCABULARY["rinv"].legal == {"v"}
    for name in ("uncontracted", "rinv"):
        assert levers.VOCABULARY[name].orders == {1}


_OP = {Target.S: Operator.OVERLAP, Target.T: Operator.KINETIC,
       Target.V: Operator.NUCLEAR}


@pytest.mark.parametrize(("target", "lever", "deriv", "engine", "match"), [
    (Target.S, "rinv", 1, None, "lever 'rinv' on target s"),
    (Target.T, "rinv", 1, None, "lever 'rinv' on target t"),
    (Target.S, "uncontracted", 0, None, "at derivative order 0"),
    (Target.V, "uncontracted", 2, None, "at derivative order 2"),
    (Target.V, "rinv", 0, None, "at derivative order 0"),
    (Target.S, "uncontracted", 1, Engine.HGP, "on target s.hgp"),
    (Target.V, "rinv", 1, Engine.HGP, "on target v.hgp"),
], ids=lambda v: getattr(v, "value", v))
def test_spec_refuses_a_form_s_lever_elsewhere(target, lever, deriv, engine,
                                               match) -> None:
    with pytest.raises(ValueError, match=match):
        Spec(_OP[target], deriv, (1, 0), target, (lever,), engine=engine)


def test_a_contraction_key_or_two_forms_are_refused() -> None:
    with pytest.raises(ValueError, match="exactly one of the levers"):
        build_ip1e_kernel(Spec(Operator.OVERLAP, 1, (1, 0), Target.S))
    with pytest.raises(ValueError, match="exactly one of the levers"):
        build_ip1e_kernel(Spec(Operator.NUCLEAR, 1, (1, 0), Target.V,
                               ("rinv", "uncontracted")))
    with pytest.raises(ValueError, match="not one of"):
        ip1e_spec(Target.S, 1, 0, "ip2")
    # the metric's uncontracted key is kernel_ipmetric's, not this builder's
    with pytest.raises(ValueError, match="exactly one of the levers"):
        build_ip1e_kernel(Spec(Operator.COULOMB, 1, (1, 0), Target.METRIC,
                               ("uncontracted",)))


@pytest.mark.parametrize(("target", "form"), IP1E,
                         ids=lambda v: getattr(v, "value", v))
def test_the_parameters_and_the_one_output(target, form) -> None:
    k = build_ip1e_kernel((target, 1, 1, form))
    names = [p.name for p in k.params]
    v = target is Target.V
    want = ["n_rec", "n_ao", *(["n_atom"] if v else []),
            *(["nuc"] if form == "rinv" else []), "zeta_ab",
            "pa_x", "pa_y", "pa_z", "pb_x", "pb_y", "pb_z",
            *(["p_x", "p_y", "p_z"] if v else []), "s_ab",
            *(["ea_bra", "eb_ket"] if form == "ip1" else []),
            "off_a", "off_b", *(["Z", "Cnuc"] if v else []),
            "XC" if form == "rinv" else "X1"]
    assert names == want
    assert [q.name for q in k.params if q.intent is Intent.OUT] == [want[-1]]


@pytest.mark.parametrize(("target", "form"), IP1E,
                         ids=lambda v: getattr(v, "value", v))
@pytest.mark.parametrize("cls", [(1, 2), (2, 2)], ids=["12", "22"])
def test_every_write_is_an_atomic_add_into_the_matrix(target, form,
                                                      cls) -> None:
    k = build_ip1e_kernel((target, *cls, form))
    out = out_array((target, *cls, form))
    nodes = list(walk(k.body))
    adds = [s for s in nodes if isinstance(s, AtomicAdd)]
    assert adds and all(a.dst.array == out for a in adds)
    notes = {s.prov for s in nodes if isinstance(s, Note)}
    assert ATOMIC_REASON in notes and ZEROED_X in notes
    assert not any(isinstance(s, Raw) for s in nodes)
    # Two blocks per axis: (m, n), and (n, m) off the diagonal; S and T
    # unroll the rows, V and rinv roll them.
    rows = 1 if target is Target.V else 6 * (6 if cls == (2, 2) else 3)
    assert len(adds) == 2 * 3 * rows
