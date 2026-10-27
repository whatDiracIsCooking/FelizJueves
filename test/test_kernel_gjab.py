"""`kernel_gjab`: the Coulomb-energy gradient kernel (`Target.JAB` at d1),
interpreted.

THE WITNESS is `gradient.host.coulomb_gradient` (held to finite
differences of `eri.j_coulomb.j.coulomb_matrix`, test/test_gradient_host.py).

- PER BRA CLASS, every ordered class through (ff) on a three-atom basis
  with one s, p, d and f shell on one atom, an s and a p shell on one
  other and an s, a p and a d (after the f, so (fd) occurs) on the last,
  one primitive each (so the host's `J_p` at `L + 1`, up to
  Hermite order 7, stays cheap): one launch over every record of the
  class, `Jp` the host's (`host.coulomb_jp`), against
  `coulomb_gradient(..., pairs=)` of the same pairs.
- THE CHAIN: `Jp` from the INTERPRETED d0 `[p|q]` kernel of
  `gjab_pq_spec`'s class -- the bra raised by one, an ordinary `Target.J`
  class (`eri.j_coulomb.drive.KernelJP`) -- over the J chain's kets, then
  `gjab`: the whole `dE_J` of the s, p basis of test_gradient_host.py and
  of H2O/STO-3G (`slow`), against the host.

ROUND-OFF: `|G_kernel - G_host| <= TOL max(scale)`, `TOL = 16 eps`,
`scale` test_kernel_g1e.py's (each shell on its own row, each pair alone,
`|.|`, folded back). The chain's `Jp` is the plan's MILD recurrence, not
`reference.pq`'s, so it adds that difference: the chain is held to
`CHAIN_TOL = 64 eps` of the same scale.
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
from felizjueves.gradient import host
from felizjueves.gradient.kernel_gjab import (
    build_gjab_kernel,
    gjab_name,
    gjab_pq_spec,
    gjab_spec,
    launch_shape,
    n_p1,
)
from felizjueves.ir import AtomicAdd, Intent, walk
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

_A, _B, _C = (0.0, 0.1, -0.2), (1.4, -0.3, 0.5), (-0.6, 1.1, 0.8)
SPDF = (Basis((Shell(0, _B, (0.9,), (0.8,)), Shell(1, _B, (0.7,), (1.2,)),
               Shell(0, _A, (1.1,), (0.7,)), Shell(1, _A, (0.8,), (0.9,)),
               Shell(2, _A, (0.6,), (1.3,)), Shell(3, _A, (0.5,), (0.6,)),
               Shell(1, _C, (1.3,), (0.5,)), Shell(0, _C, (0.4,), (1.0,)),
               Shell(2, _C, (0.9,), (0.7,)))),
        ((2.0, _A), (1.0, _B), (3.0, _C)))


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


def _case(name: str):
    return {"spdf": SPDF, "pair_sp": PAIR_SP}.get(name) or _fixture(name)


def _density(n: int) -> np.ndarray:
    X = np.random.default_rng(5).standard_normal((n, n))
    return X @ X.T / n


@cache
def _kets(name: str):
    basis, _ = _case(name)
    return host.coulomb_kets(basis, _density(basis.n_ao))


def _arrays(rec):
    return dict(
        zeta_ab=np.array(rec["zeta_ab"]),
        XPA=np.array([rec[f"pa_{a}"] for a in "xyz"]).T.copy(),
        XPB=np.array([rec[f"pb_{a}"] for a in "xyz"]).T.copy(),
        s=np.array(rec["s_ab"]), ea_bra=np.array(rec["ea_bra"]),
        eb_ket=np.array(rec["eb_ket"]),
        **{k: np.array(rec[k], dtype=np.int32)
           for k in ("off_a", "off_b", "atom_a", "atom_b")})


def _host_jp(name: str, la: int, lb: int, rec) -> np.ndarray:
    return np.array([host.coulomb_jp(
        (rec["p_x"][r], rec["p_y"][r], rec["p_z"][r]), rec["zeta_ab"][r],
        la + lb + 1, _kets(name), one_centre=one_centre)
        for r in range(len(rec["s_ab"]))])


def _chain_jp(name: str, la: int, lb: int, rec) -> np.ndarray:
    """`Jp` at `L + 1` from the interpreted `[p|q]` kernel of
    `gjab_pq_spec`'s class, one ket class at a time (module docstring)."""
    bras = [((rec["p_x"][r], rec["p_y"][r], rec["p_z"][r]), rec["zeta_ab"][r])
            for r in range(len(rec["s_ab"]))]
    by_lq: dict[int, list] = {}
    for k in _kets(name):
        lq = next(L for L in range(16) if len(hermite_order(L)) == len(k.qs))
        by_lq.setdefault(lq, []).append((k.Q, k.eta, k.dq))
    jp = KernelJP(boys=boys)
    out = np.zeros((len(bras), n_p1(la, lb)))
    for lq, kets in sorted(by_lq.items()):
        assert split(gjab_pq_spec(la, lb, lq, 0)) == (la + lb + 1, lq)
        out += jp((la + 1, lb), (lq, 0), bras, kets)
    return out


def _launch(name: str, la: int, lb: int, G, *, chain: bool = False):
    basis, nuclei = _case(name)
    rec = host.records(basis, nuclei, la, lb)
    n = len(rec["s_ab"])
    jp = (_chain_jp if chain else _host_jp)(name, la, lb, rec)
    grid, block = launch_shape(n, 3)
    return Interpreter(build_gjab_kernel((la, lb)),
                       launch=Launch(grid, block, 1)).run(
        n_rec=n, n_ao=basis.n_ao, n_atom=len(nuclei), n_rows=n, Jp=jp,
        M=_density(basis.n_ao), G=G, row=np.arange(n, dtype=np.int32),
        **_arrays(rec))["G"]


def _host(name: str, pairs=None, atoms=None) -> np.ndarray:
    basis, nuclei = _case(name)
    return np.array(host.coulomb_gradient(
        basis, _density(basis.n_ao), nuclei, one_centre=one_centre,
        pairs=pairs, atoms=atoms, kets=_kets(name)))


def _scale(name: str, pairs) -> float:
    basis, nuclei = _case(name)
    nat = len(nuclei)
    real = host.shell_atoms(basis, nuclei)
    rows = tuple(nat + i for i in range(len(basis.shells)))
    sc = np.zeros((nat, 3))
    for p in pairs:
        g = np.abs(_host(name, [p], rows))
        for i, row in enumerate(rows):
            sc[real[i]] += g[row]
    return float(sc.max())


def _all_pairs(name: str):
    n = len(_case(name)[0].shells)
    return [(i, j) for i in range(n) for j in range(i, n)]


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_bra_class_is_the_host_gradient(cls) -> None:
    basis, nuclei = SPDF
    pairs = host.class_pairs(basis, *cls)
    assert pairs, cls
    G = _launch("spdf", *cls, np.zeros((len(nuclei), 3)))
    want = _host("spdf", pairs)
    sc = _scale("spdf", pairs)
    assert sc > 0.0
    err = np.abs(G - want).max()
    assert err <= TOL * sc, f"{cls}: {err / sc / EPS:.1f} eps of scale"


@pytest.mark.parametrize("name", [
    pytest.param("pair_sp", id="pair_sp"),
    pytest.param("h2o_sto3g_j", id="h2o_sto3g", marks=(pytest.mark.slow,))])
def test_the_chain_through_the_pq_kernel_one_order_up_is_de_j(name) -> None:
    basis, nuclei = _case(name)
    G = np.zeros((len(nuclei), 3))
    for la, lb in sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                          for b in basis.shells[k:]}):
        G = _launch(name, la, lb, G, chain=True)
    want = _host(name)
    sc = _scale(name, _all_pairs(name))
    err = np.abs(G - want).max()
    assert err <= CHAIN_TOL * sc, f"{name}: {err / sc / EPS:.1f} eps of scale"
    assert np.abs(G.sum(axis=0)).max() <= 1e-13 * np.abs(G).max()


# --- the kernel's shape -------------------------------------------------------


def test_the_key_the_name_and_the_pq_class() -> None:
    assert gjab_spec(2, 1).key() == "coulomb/d1/(2,1)/b1/jab/literal"
    assert gjab_name((2, 1)) == build_gjab_kernel((2, 1)).name == "gjab_21"
    pqs = gjab_pq_spec(2, 1, 1, 0)
    assert pqs.target is Target.J and pqs.derivative_order == 0
    assert split(pqs) == (4, 1)
    assert n_p1(2, 1) == len(hermite_order(4))


def test_a_d0_jab_key_is_refused() -> None:
    with pytest.raises(ValueError, match="not a d1 Target.JAB key"):
        build_gjab_kernel(Spec(Operator.COULOMB, 0, (1, 0), Target.JAB))


def test_the_parameters_and_the_one_output() -> None:
    k = build_gjab_kernel((1, 2))
    assert [p.name for p in k.params] == [
        "n_rec", "n_ao", "n_atom", "n_rows", "zeta_ab", "XPA", "XPB", "s",
        "ea_bra", "eb_ket", "off_a", "off_b", "row", "atom_a", "atom_b", "Jp",
        "M", "G"]
    assert [p.name for p in k.params if p.intent is Intent.OUT] == ["G"]
    jp = next(p for p in k.params if p.name == "Jp")
    assert jp.shape[1].value == n_p1(1, 2)
    adds = [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 and all(a.dst.array == "G" for a in adds)
