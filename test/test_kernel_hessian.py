"""`kernel_hdq` and `kernel_hjab`: the Coulomb-energy Hessian kernels
(`Target.DQ` at d1, `Target.JAB` at d2), interpreted.

THE WITNESS is `hessian.host` (held to finite differences of the gradient,
test/test_hessian_host.py).

- `hdq`, PER KET CLASS, every ordered class through (dd) on a three-atom
  basis (one s, p and d shell on one atom, an s and a p on another, a p, s
  and d on the last, one primitive each), and every class with an f shell
  on a two-atom basis with an s, p, d and f shell on each (`SPDF`): one
  launch over every record of the class against
  `host.ket_derivative_records`, BIT FOR BIT, in BOTH forms (#690: the
  rolled form is the same sums in the same order).
- `hjab`, PER BRA CLASS, the same classes and bases: one launch, `Jp2` and
  `Jp1` the host's (`gradient.host.coulomb_jp` over the kets and over
  `host.coulomb_dkets`), against `coulomb_hessian(..., pairs=)` of the
  class's pairs, in the default form (rolled from `la + lb = 3`) and,
  through `la + lb = 4`, in the other form too.
- THE CHAIN: every `J` from the INTERPRETED d0 `[p|q]` kernel
  (`eri.j_coulomb.drive.KernelJP`) -- `Jp2` of `hjab_pq2_spec`'s class
  over the J chain's kets, `Jp1` of `hjab_pq1_spec`'s class over `hdq`'s
  own output gathered per atom and axis (the host's step) -- then `hjab`:
  the whole skeleton `d2 E_J` of the s, p basis of test_gradient_host.py
  and of H2O/STO-3G (`slow`), against the host.

ROUND-OFF: `|H_kernel - H_host| <= TOL max(scale)`, `TOL = 16 eps`,
`scale` the host's absolute twin (`coulomb_hessian(..., absolute=True)`:
every `E` entry, derivative term, weight, `s` and `J_t` in `|.|`), the size
of what cancels -- a diagonal shell pair's `AA`, `AB` and `BB` terms, each
carrying `(2 alpha)^2`, land on one atom and cancel to a far smaller
value, which a scale built from the result alone misses (it read 22 eps at
(dd)). The chain's `J`s are the plan's MILD recurrence, not
`reference.pq`'s, so it is held to `CHAIN_TOL = 64 eps` of the same scale.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_gradient_host import PAIR_SP
from tiers import by_l

from felizjueves.ao.shell import Basis, Shell
from felizjueves.eri.j_coulomb.drive import KernelJP
from felizjueves.gradient import host as gh
from felizjueves.hessian import host
from felizjueves.hessian.kernel_hdq import (
    build_hdq_kernel,
    hdq_lowstack,
    hdq_name,
    hdq_spec,
    n_q1,
)
from felizjueves.hessian.kernel_hjab import (
    build_hjab_kernel,
    hjab_lowstack,
    hjab_name,
    hjab_pq1_spec,
    hjab_pq2_spec,
    hjab_spec,
    launch_shape,
    n_p1,
    n_p2,
)
from felizjueves.ir import Assign, AtomicAdd, For, Intent, walk
from felizjueves.ir.interp import Interpreter, Launch
from felizjueves.mmd.hermite import hermite_order
from felizjueves.mmd.kernel import split
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Operator, Spec, Target

EPS = 2.0 ** -52
TOL = 16 * EPS
CHAIN_TOL = 64 * EPS
LMAX = 3
CLASSES = [(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)]
# the other form of each class through la + lb = 4 (the unrolled (df) and
# (ff) are megabytes of straight-line text)
OTHER_FORM = [c for c in CLASSES if sum(c) <= 4]

_A, _B, _C = (0.0, 0.1, -0.2), (1.4, -0.3, 0.5), (-0.6, 1.1, 0.8)
SPD = (Basis((Shell(0, _B, (0.9,), (0.8,)), Shell(1, _B, (0.7,), (1.2,)),
              Shell(0, _A, (1.1,), (0.7,)), Shell(1, _A, (0.8,), (0.9,)),
              Shell(2, _A, (0.6,), (1.3,)),
              Shell(1, _C, (1.3,), (0.5,)), Shell(0, _C, (0.4,), (1.0,)),
              Shell(2, _C, (0.9,), (0.7,)))),
       ((2.0, _A), (1.0, _B), (3.0, _C)))
# #690: an s, p, d and f shell on each of two atoms, one primitive each, so
# every ordered class through (ff) has a shell pair (A's shells first)
_EXP = (1.1, 0.8, 0.6, 0.5)
SPDF = (Basis(tuple(Shell(l, X, (_EXP[l] * (1.0 + 0.2 * n),), (1.0,))
                    for n, X in enumerate((_A, _B))
                    for l in range(4))),  # noqa: E741
        ((2.0, _A), (1.0, _B)))


def _basis_of(cls) -> str:
    """The per-class tests' basis: `SPD` through d, `SPDF` with an f."""
    return "spd" if max(cls) <= 2 else "spdf"


@cache
def _case(name: str):
    if name == "spd":
        return SPD
    if name == "spdf":
        return SPDF
    if name == "pair_sp":
        return PAIR_SP
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


@cache
def _density(n: int) -> np.ndarray:
    X = np.random.default_rng(5).standard_normal((n, n))
    return X @ X.T / n


@cache
def _kets(name: str):
    basis, _ = _case(name)
    return gh.coulomb_kets(basis, _density(basis.n_ao))


@cache
def _dkets(name: str):
    basis, nuclei = _case(name)
    return host.coulomb_dkets(basis, _density(basis.n_ao), nuclei)


def _arrays(rec, *, atoms: bool = True):
    out = dict(
        zeta_ab=np.array(rec["zeta_ab"]),
        XPA=np.array([rec[f"pa_{a}"] for a in "xyz"]).T.copy(),
        XPB=np.array([rec[f"pb_{a}"] for a in "xyz"]).T.copy(),
        s=np.array(rec["s_ab"]), ea_bra=np.array(rec["ea_bra"]),
        eb_ket=np.array(rec["eb_ket"]))
    for k in ("off_a", "off_b") + (("atom_a", "atom_b") if atoms else ()):
        out[k] = np.array(rec[k], dtype=np.int32)
    return out


def _bras(rec) -> list:
    return [((rec["p_x"][r], rec["p_y"][r], rec["p_z"][r]), rec["zeta_ab"][r])
            for r in range(len(rec["s_ab"]))]


def _hdq(name: str, lc: int, ld: int, lowstack: bool | None = None
         ) -> tuple[dict, np.ndarray]:
    """One interpreted `hdq` launch over every record of ket class `(lc,
    ld)`: the records and `D1[6, n_rec, n_q1]`."""
    basis, nuclei = _case(name)
    rec = gh.records(basis, nuclei, lc, ld)
    n = len(rec["s_ab"])
    grid, block = launch_shape(n, 3)
    D1 = np.full((6, n, n_q1(lc, ld)), np.nan)  # every element is stored
    D1 = Interpreter(build_hdq_kernel((lc, ld), lowstack=lowstack),
                     launch=Launch(grid, block, 1)).run(
        n_rec=n, n_ao=basis.n_ao, M=_density(basis.n_ao), D1=D1,
        **_arrays(rec, atoms=False))["D1"]
    return rec, D1


def _host_jps(name: str, la: int, lb: int, rec):
    nat = len(_case(name)[1])
    bras = _bras(rec)
    jp2 = np.array([gh.coulomb_jp(P, z, la + lb + 2, _kets(name),
                                  one_centre=one_centre) for P, z in bras])
    dk = _dkets(name)
    jp1 = np.zeros((nat, 3, len(bras), n_p1(la, lb)))
    for (C, l), ks in dk.items():  # noqa: E741
        if ks:
            jp1[C, l] = [gh.coulomb_jp(P, z, la + lb + 1, ks,
                                       one_centre=one_centre)
                         for P, z in bras]
    return jp2, jp1


def _lq(n_q: int) -> int:
    return next(L for L in range(32) if len(hermite_order(L)) == n_q)


def _chain_jps(name: str, la: int, lb: int, rec):
    """`Jp2` and `Jp1` from the interpreted `[p|q]` kernel, one ket class
    at a time; `Jp1`'s kets are `hdq`'s output gathered per atom and axis
    (module docstring)."""
    basis, nuclei = _case(name)
    nat = len(nuclei)
    bras = _bras(rec)
    jp = KernelJP(boys=boys)
    jp2 = np.zeros((len(bras), n_p2(la, lb)))
    by_lq: dict[int, list] = {}
    for k in _kets(name):
        by_lq.setdefault(_lq(len(k.qs)), []).append((k.Q, k.eta, k.dq))
    for lq, kets in sorted(by_lq.items()):
        assert split(hjab_pq2_spec(la, lb, lq, 0)) == (la + lb + 2, lq)
        jp2 += jp((la + 2, lb), (lq, 0), bras, kets)
    jp1 = np.zeros((nat, 3, len(bras), n_p1(la, lb)))
    for lc, ld in _ket_classes(name):
        krec, D1 = _hdq(name, lc, ld)
        lq = lc + ld
        assert split(hjab_pq1_spec(la, lb, lc, ld)) == (la + lb + 1, lq + 1)
        for C in range(nat):
            for l in range(3):  # noqa: E741
                kets = []
                for r in range(len(krec["s_ab"])):
                    Q = (krec["p_x"][r], krec["p_y"][r], krec["p_z"][r])
                    for side, at in ((0, "atom_a"), (1, "atom_b")):
                        if krec[at][r] == C:
                            kets.append((Q, krec["zeta_ab"][r],
                                         D1[3 * side + l, r]))
                if kets:
                    jp1[C, l] += jp((la + 1, lb), (lq + 1, 0), bras, kets)
    return jp2, jp1


def _ket_classes(name: str) -> list[tuple[int, int]]:
    basis, _ = _case(name)
    sh = basis.shells
    return sorted({(sh[i].l, sh[j].l) for i in range(len(sh))
                   for j in range(i, len(sh))})


def _launch(name: str, la: int, lb: int, H, *, chain: bool = False,
            lowstack: bool | None = None):
    basis, nuclei = _case(name)
    rec = gh.records(basis, nuclei, la, lb)
    n = len(rec["s_ab"])
    jp2, jp1 = (_chain_jps if chain else _host_jps)(name, la, lb, rec)
    grid, block = launch_shape(n, 3)
    return Interpreter(build_hjab_kernel((la, lb), lowstack=lowstack),
                       launch=Launch(grid, block, 1)).run(
        n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), n_rows=n, Jp2=jp2,
        Jp1=jp1, M=_density(basis.n_ao), H=H,
        row=np.arange(n, dtype=np.int32), **_arrays(rec))["H"]


def _host(name: str, pairs=None) -> np.ndarray:
    basis, nuclei = _case(name)
    nat = len(nuclei)
    return np.array(host.coulomb_hessian(
        basis, _density(basis.n_ao), nuclei, one_centre=one_centre,
        pairs=pairs, kets=_kets(name), dkets=_dkets(name))).reshape(
            nat, 3, nat, 3)


def _scale(name: str, pairs) -> float:
    """`max` of the host's absolute twin over `pairs` (module docstring)."""
    basis, nuclei = _case(name)
    return float(np.abs(np.array(host.coulomb_hessian(
        basis, _density(basis.n_ao), nuclei, one_centre=one_centre,
        pairs=pairs, kets=_kets(name), dkets=_dkets(name),
        absolute=True))).max())


def _all_pairs(name: str):
    n = len(_case(name)[0].shells)
    return [(i, j) for i in range(n) for j in range(i, n)]


@pytest.mark.parametrize("form", ["default", "other"])
@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_hdq_is_the_hosts_ket_derivative_densities_bit_for_bit(
        cls, form) -> None:
    name = _basis_of(cls)
    basis, _ = _case(name)
    low = None if form == "default" else not hdq_lowstack(cls)
    _, D1 = _hdq(name, *cls, lowstack=low)
    want = np.array(host.ket_derivative_records(
        basis, _density(basis.n_ao), *cls))
    assert D1.shape == want.shape and not np.isnan(D1).any()
    assert np.array_equal(D1, want), cls


def _check_class(cls, lowstack: bool | None) -> None:
    name = _basis_of(cls)
    basis, nuclei = _case(name)
    pairs = gh.class_pairs(basis, *cls)
    assert pairs, cls
    nat = len(nuclei)
    H = _launch(name, *cls, np.zeros((nat, 3, nat, 3)), lowstack=lowstack)
    want = _host(name, pairs)
    sc = _scale(name, pairs)
    assert sc > 0.0
    err = np.abs(H - want).max()
    assert err <= TOL * sc, f"{cls}: {err / sc / EPS:.1f} eps of scale"


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_bra_class_is_the_host_hessian(cls) -> None:
    _check_class(cls, None)


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 2, id=f"{c[0]}{c[1]}")
                                 for c in OTHER_FORM])
def test_the_other_form_is_the_host_hessian_too(cls) -> None:
    _check_class(cls, not hjab_lowstack(cls))


@pytest.mark.parametrize("name", [
    pytest.param("pair_sp", id="pair_sp"),
    pytest.param("h2o_sto3g_j", id="h2o_sto3g", marks=(pytest.mark.slow,))])
def test_the_chain_through_the_pq_kernel_is_the_coulomb_hessian(name) -> None:
    basis, nuclei = _case(name)
    nat = len(nuclei)
    H = np.zeros((nat, 3, nat, 3))
    for la, lb in _ket_classes(name):
        H = _launch(name, la, lb, H, chain=True)
    want = _host(name)
    sc = _scale(name, _all_pairs(name))
    err = np.abs(H - want).max()
    assert err <= CHAIN_TOL * sc, f"{name}: {err / sc / EPS:.1f} eps of scale"
    Hm = H.reshape(3 * nat, 3 * nat)
    assert np.abs(Hm - Hm.T).max() <= 1e-13 * np.abs(Hm).max()
    assert np.abs(H.sum(axis=0)).max() <= 1e-13 * np.abs(Hm).max()


# --- the kernels' shape -------------------------------------------------------


def test_the_keys_the_names_and_the_pq_classes() -> None:
    assert hjab_spec(2, 1).key() == "coulomb/d2/(2,1)/b1/jab/literal"
    assert hdq_spec(2, 1).key() == "coulomb/d1/(2,1)/b1/dq/literal"
    assert hjab_name((2, 1)) == build_hjab_kernel((2, 1)).name == "hjab_21"
    assert hdq_name((1, 0)) == build_hdq_kernel((1, 0)).name == "hdq_10"
    for pqs, want in ((hjab_pq2_spec(2, 1, 1, 0), (5, 1)),
                      (hjab_pq1_spec(2, 1, 1, 0), (4, 2))):
        assert pqs.target is Target.J and pqs.derivative_order == 0
        assert split(pqs) == want
    assert n_p2(2, 1) == len(hermite_order(5))
    assert n_p1(2, 1) == len(hermite_order(4))
    assert n_q1(1, 1) == len(hermite_order(3))


def test_the_wrong_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="not a d2 Target.JAB key"):
        build_hjab_kernel(Spec(Operator.COULOMB, 1, (1, 0), Target.JAB))
    with pytest.raises(ValueError, match="not a d1 Target.DQ key"):
        build_hdq_kernel(Spec(Operator.COULOMB, 0, (1, 0), Target.DQ))


def test_the_parameters_and_the_one_output() -> None:
    k = build_hjab_kernel((1, 2), lowstack=False)
    assert [p.name for p in k.params] == [
        "n_rec", "n_ao", "n_atom", "n_rows", "zeta_ab", "XPA", "XPB", "s",
        "ea_bra", "eb_ket", "off_a", "off_b", "row", "atom_a", "atom_b",
        "Jp2", "Jp1", "M", "H"]
    assert [p.name for p in k.params if p.intent is Intent.OUT] == ["H"]
    shapes = {p.name: p.shape for p in k.params}
    assert shapes["Jp2"][1].value == n_p2(1, 2)
    assert shapes["Jp1"][3].value == n_p1(1, 2)
    loops = [s for s in walk(k.body) if isinstance(s, For)]
    assert len(loops) == 1 and loops[0].stop.name == "n_atom"
    adds = [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    inner = [s for s in walk(loops[0].body) if isinstance(s, AtomicAdd)]
    assert len(adds) == 36 + 18 and len(inner) == 18
    assert all(a.dst.array == "H" for a in adds)

    r = build_hjab_kernel((1, 2))  # rolled, the default from la + lb = 3
    assert r.params == k.params
    adds = [s for s in walk(r.body) if isinstance(s, AtomicAdd)]
    atom = [s for s in walk(r.body) if isinstance(s, For)
            and getattr(s.stop, "name", None) == "n_atom"]
    assert len(adds) == 36 + 2 and len(atom) == 1
    assert len([s for s in walk(atom[0].body)
                if isinstance(s, AtomicAdd)]) == 2

    d = build_hdq_kernel((1, 2))
    assert [p.name for p in d.params] == [
        "n_rec", "n_ao", "zeta_ab", "XPA", "XPB", "s", "ea_bra", "eb_ket",
        "off_a", "off_b", "M", "D1"]
    assert [p.name for p in d.params if p.intent is Intent.OUT] == ["D1"]
    stores = [s for s in walk(d.body) if isinstance(s, Assign)
              and getattr(s.dst, "array", None) == "D1"]
    assert len(stores) == 6 * n_q1(1, 2)
    assert not any(isinstance(s, AtomicAdd) for s in walk(d.body))
    dr = build_hdq_kernel((1, 2), lowstack=True)
    assert dr.params == d.params
    assert len([s for s in walk(dr.body) if isinstance(s, Assign)
                and getattr(s.dst, "array", None) == "D1"]) == 1
    assert not any(isinstance(s, AtomicAdd) for s in walk(dr.body))
