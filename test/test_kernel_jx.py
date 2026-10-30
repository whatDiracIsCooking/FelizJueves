"""`kernel_jx` and `host_jx`: the Coulomb derivative matrix `J^X` (#688),
its ip1 bra half `J1` as a generated kernel (`Target.JX` at d1), its ket
half as the ordinary J chain over `hdq`'s derivative kets.

THE WITNESS is PySCF, through the numerical golden store
(`oracle.numeric`, `jx1_*` and `jx_*`): `J1` is `grad.rhf.get_jk`'s `vj`
(libcint's `int2e_ip1` contracted `ijkl,lk->ij`) and `J^X` the Coulomb part
of `hessian.rhf`'s `h1ao`, H2O over STO-3G and cc-pVDZ, for the J
fixtures' recorded densities.

- THE HOST (`hessian.host_jx`): `coulomb_ip1` and
  `coulomb_derivative_matrix` against the STO-3G goldens, `coulomb_ip1`
  against the cc-pVDZ one (`slow`), and `SUM_X J^X = 0` (translational
  invariance) on both goldens and on the host. The whole `J^X` over
  cc-pVDZ is not recomputed here: its ket half is nine J builds over ~10k
  derivative kets in pure Python, hours; the golden is tracked for a
  faster witness to use.
- PER BRA CLASS, every ordered class through (ff) on test_kernel_gjab.py's
  three-atom s, p, d, f basis: one interpreted `jx` launch over every record
  of the class, `Jp` the host's (`gradient.host.coulomb_jp` at `L + 1`),
  against `coulomb_ip1(..., pairs=)` of the same pairs.
- THE CHAIN, interpreted end to end: `J1` from `jx` over `Jp` of the
  INTERPRETED d0 `[p|q]` kernel of `jx_pq_spec`'s class (the bra raised by
  one, `gjab`'s); the ket half from `hdq`'s `D1` slabs gathered per atom and
  axis (`hjab`'s gather), the d0 `[p|q]` of `(la, lb | lc + ld + 1, 0)` over
  them and the d0 `jab`; assembled as the host does -- on the s, p pair of
  test_gradient_host.py against the host, and on H2O/STO-3G (`slow`)
  against PySCF.

ROUND-OFF: the kernel against the host `<= TOL max(scale)`, `TOL = 16 eps`,
`scale` the host's absolute twin (`coulomb_ip1(..., absolute=True)`: every
factor in `|.|`), the size of what cancels in a derivative; the chain's `J`s
are the plan's MILD recurrence, not `reference.pq`'s, so it is held to
`CHAIN_TOL = 64 eps`. Against PySCF, `oracle.numeric.compare`'s fp64 tier.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import numeric as golden
from test_gradient_host import PAIR_SP
from test_kernel_gjab import SPDF
from tiers import by_l

from felizjueves.eri.j_coulomb.drive import KernelJP
from felizjueves.eri.j_coulomb.kernel_jab import build_jab_kernel
from felizjueves.gradient import host as gh
from felizjueves.hessian import host as hh
from felizjueves.hessian import host_jx
from felizjueves.hessian.kernel_hdq import build_hdq_kernel, n_q1
from felizjueves.hessian.kernel_jx import (
    build_jx_kernel,
    jx_name,
    jx_pq_spec,
    jx_spec,
    launch_shape,
    n_p1,
)
from felizjueves.ir import AtomicAdd, For, If, Intent, walk
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
GOLDEN = {"h2o_sto3g": "h2o_sto3g_bse", "h2o_ccpvdz": "h2o_ccpvdz_bse"}


def _density(n: int) -> np.ndarray:
    X = np.random.default_rng(5).standard_normal((n, n))
    return X @ X.T / n


@cache
def _case(name: str):
    """`(basis, nuclei, D)`: a synthetic basis with a random density, or a
    golden's file, molecule and recorded density."""
    if name in GOLDEN:
        g = f"jx_{GOLDEN[name]}"
        nuclei = tuple((float(z), tuple(c)) for z, c in golden.atoms(g))
        return golden.basis(g), nuclei, np.asarray(golden.density(g))
    basis, nuclei = {"spdf": SPDF, "pair_sp": PAIR_SP}[name]
    return basis, nuclei, _density(basis.n_ao)


@cache
def _kets(name: str):
    basis, _, D = _case(name)
    return gh.coulomb_kets(basis, D)


@cache
def _host_j1(name: str, pairs=None, absolute: bool = False) -> np.ndarray:
    basis, _, D = _case(name)
    return np.array(host_jx.coulomb_ip1(
        basis, D, one_centre=one_centre, pairs=pairs, kets=_kets(name),
        absolute=absolute))


@cache
def _host_jx(name: str) -> np.ndarray:
    basis, nuclei, D = _case(name)
    return np.array(host_jx.coulomb_derivative_matrix(
        basis, D, nuclei, one_centre=one_centre, kets=_kets(name),
        J1=_host_j1(name).tolist()))


def _arrays(rec, *, exponents: bool = True):
    out = dict(
        zeta_ab=np.array(rec["zeta_ab"]),
        XPA=np.array([rec[f"pa_{a}"] for a in "xyz"]).T.copy(),
        XPB=np.array([rec[f"pb_{a}"] for a in "xyz"]).T.copy(),
        s=np.array(rec["s_ab"]),
        **{k: np.array(rec[k], dtype=np.int32) for k in ("off_a", "off_b")})
    if exponents:
        out.update(ea_bra=np.array(rec["ea_bra"]),
                   eb_ket=np.array(rec["eb_ket"]))
    return out


def _bras(rec) -> list:
    return [((rec["p_x"][r], rec["p_y"][r], rec["p_z"][r]), rec["zeta_ab"][r])
            for r in range(len(rec["s_ab"]))]


def _lq(n_q: int) -> int:
    return next(L for L in range(32) if len(hermite_order(L)) == n_q)


def _classes(name: str) -> list[tuple[int, int]]:
    sh = _case(name)[0].shells
    return sorted({(sh[i].l, sh[j].l) for i in range(len(sh))
                   for j in range(i, len(sh))})


def _host_jp(name: str, la: int, lb: int, rec) -> np.ndarray:
    return np.array([gh.coulomb_jp(P, z, la + lb + 1, _kets(name),
                                   one_centre=one_centre)
                     for P, z in _bras(rec)])


def _chain_jp(name: str, la: int, lb: int, rec) -> np.ndarray:
    """`Jp` at `L + 1` from the interpreted `[p|q]` of `jx_pq_spec`'s
    class, one ket class at a time."""
    bras = _bras(rec)
    by_lq: dict[int, list] = {}
    for k in _kets(name):
        by_lq.setdefault(_lq(len(k.qs)), []).append((k.Q, k.eta, k.dq))
    jp = KernelJP(boys=boys)
    out = np.zeros((len(bras), n_p1(la, lb)))
    for lq, kets in sorted(by_lq.items()):
        assert split(jx_pq_spec(la, lb, lq, 0)) == (la + lb + 1, lq)
        out += jp((la + 1, lb), (lq, 0), bras, kets)
    return out


def _launch_jx(name: str, la: int, lb: int, J1, *, chain: bool = False):
    basis, nuclei, _ = _case(name)
    rec = gh.records(basis, nuclei, la, lb)
    n = len(rec["s_ab"])
    jp = (_chain_jp if chain else _host_jp)(name, la, lb, rec)
    grid, block = launch_shape(n, 3)
    return Interpreter(build_jx_kernel((la, lb)),
                       launch=Launch(grid, block, 1)).run(
        n_rec=n, n_ao=basis.n_ao, n_rows=n, Jp=jp, J1=J1,
        row=np.arange(n, dtype=np.int32), **_arrays(rec))["J1"]


def _chain_j1(name: str) -> np.ndarray:
    n = _case(name)[0].n_ao
    J1 = np.zeros((3, n, n))
    for la, lb in _classes(name):
        J1 = _launch_jx(name, la, lb, J1, chain=True)
    return J1


def _hdq(name: str, lc: int, ld: int):
    basis, nuclei, D = _case(name)
    rec = gh.records(basis, nuclei, lc, ld)
    n = len(rec["s_ab"])
    grid, block = launch_shape(n, 3)
    D1 = np.full((6, n, n_q1(lc, ld)), np.nan)
    D1 = Interpreter(build_hdq_kernel((lc, ld)),
                     launch=Launch(grid, block, 1)).run(
        n_rec=n, n_ao=basis.n_ao, M=D, D1=D1, **_arrays(rec))["D1"]
    return rec, D1


def _chain_ket(name: str) -> np.ndarray:
    """The ket half, `[X, k, m, n]`: per atom and axis, `hdq`'s slabs of
    that atom gathered into kets (`hjab`'s gather), the d0 `[p|q]` of every
    bra class against them, then the d0 `jab` into a zeroed `J`."""
    basis, nuclei, _ = _case(name)
    nat, n = len(nuclei), basis.n_ao
    dkets: dict[tuple[int, int], dict[int, list]] = {}
    for lc, ld in _classes(name):
        krec, D1 = _hdq(name, lc, ld)
        for r in range(len(krec["s_ab"])):
            Q = (krec["p_x"][r], krec["p_y"][r], krec["p_z"][r])
            for side, at in ((0, "atom_a"), (1, "atom_b")):
                for k in range(3):
                    dkets.setdefault((krec[at][r], k), {}).setdefault(
                        lc + ld + 1, []).append(
                            (Q, krec["zeta_ab"][r], D1[3 * side + k, r]))
    jp = KernelJP(boys=boys)
    out = np.zeros((nat, 3, n, n))
    for la, lb in _classes(name):
        rec = gh.records(basis, nuclei, la, lb)
        nr = len(rec["s_ab"])
        bras = _bras(rec)
        grid, block = launch_shape(nr, 3)
        jab = build_jab_kernel((la, lb))
        for (X, k), by_lq in sorted(dkets.items()):
            p = np.zeros((nr, len(hermite_order(la + lb))))
            for lq, kets in sorted(by_lq.items()):
                p += jp((la, lb), (lq, 0), bras, kets)
            out[X, k] = Interpreter(jab, launch=Launch(grid, block, 1)).run(
                n_rec=nr, n_ao=n, n_rows=nr, Jp=p, J=out[X, k],
                row=np.arange(nr, dtype=np.int32),
                **_arrays(rec, exponents=False))["J"]
    return out


def _assemble(name: str, J1: np.ndarray, ket: np.ndarray) -> np.ndarray:
    """The bra half from `J1` (X's rows plus the transpose), plus `ket`."""
    basis, nuclei, _ = _case(name)
    at = gh.shell_atoms(basis, nuclei)
    ao_atom = np.array([at[s] for s, sh in enumerate(basis.shells)
                        for _ in range(sh.n_cart)])
    out = ket.copy()
    for X in range(len(nuclei)):
        bra = np.where((ao_atom == X)[None, :, None], J1, 0.0)
        out[X] += bra + bra.transpose(0, 2, 1)
    return out


# --- the host, held to PySCF ---------------------------------------------------


def test_the_host_is_pyscf() -> None:
    name = "h2o_sto3g"
    g = GOLDEN[name]
    e1 = golden.compare(f"jx1_{g}", _host_j1(name))
    ex = golden.compare(f"jx_{g}", _host_jx(name))
    print(f"\n{name}: J1 {e1:.2e}, J^X {ex:.2e} of max|want|")


@pytest.mark.slow  # 1629 bra records against 1629 kets, pure Python
def test_the_host_j1_over_cc_pvdz_is_pyscf() -> None:
    """cc-pVDZ holds `J1` alone: the host's ket half there is nine J builds
    over ~10k derivative kets in pure Python, hours (the module
    docstring); the whole `J^X` is held to PySCF on STO-3G."""
    e1 = golden.compare(f"jx1_{GOLDEN['h2o_ccpvdz']}", _host_j1("h2o_ccpvdz"))
    print(f"\nh2o_ccpvdz: J1 {e1:.2e} of max|want|")


@pytest.mark.parametrize("g", GOLDEN.values())
def test_the_golden_sums_to_zero_over_atoms(g) -> None:
    """Translational invariance: `SUM_X d(mn|ls)/dR_X = 0`, so `SUM_X J^X
    = 0`; and each `J^X` is symmetric."""
    want = golden.want(f"jx_{g}")
    scale = float(np.abs(want).max())
    assert np.abs(want.sum(axis=0)).max() <= 1e-13 * scale
    assert np.array_equal(want, want.transpose(0, 1, 3, 2))


def test_the_host_sums_to_zero_over_atoms() -> None:
    JX = _host_jx("pair_sp")
    assert np.abs(JX.sum(axis=0)).max() <= 1e-13 * np.abs(JX).max()
    basis, nuclei, D = _case("pair_sp")
    halves = [np.array(host_jx.coulomb_derivative_matrix(
        basis, D, nuclei, one_centre=one_centre, parts=p))
        for p in ("bra", "ket")]
    assert np.abs(halves[0] + halves[1] - JX).max() <= 1e-15 * np.abs(JX).max()


def test_the_derivative_kets_are_hdqs_slabs() -> None:
    """The host's ket half reads `hessian.host.coulomb_dkets`, whose records
    are `hdq`'s (test_kernel_hessian.py holds `hdq` to them bit for bit)."""
    basis, nuclei, D = _case("pair_sp")
    dk = hh.coulomb_dkets(basis, D, nuclei)
    assert sorted(dk) == [(X, k) for X in range(len(nuclei)) for k in range(3)]


# --- the kernel, held to the host ---------------------------------------------


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in CLASSES])
def test_one_launch_over_the_bra_class_is_the_host_j1(cls) -> None:
    basis, _, _ = _case("spdf")
    pairs = tuple(gh.class_pairs(basis, *cls))
    assert pairs, cls
    n = basis.n_ao
    J1 = _launch_jx("spdf", *cls, np.zeros((3, n, n)))
    want = _host_j1("spdf", pairs)
    sc = float(_host_j1("spdf", pairs, absolute=True).max())
    assert sc > 0.0
    err = np.abs(J1 - want).max()
    assert err <= TOL * sc, f"{cls}: {err / sc / EPS:.1f} eps of scale"


def test_the_chain_is_the_host_j1_and_jx() -> None:
    name = "pair_sp"
    J1 = _chain_j1(name)
    sc = float(_host_j1(name, absolute=True).max())
    err = np.abs(J1 - _host_j1(name)).max()
    assert err <= CHAIN_TOL * sc, f"J1: {err / sc / EPS:.1f} eps of scale"
    JX = _assemble(name, J1, _chain_ket(name))
    want = _host_jx(name)
    err = np.abs(JX - want).max()
    assert err <= CHAIN_TOL * float(np.abs(want).max()), (
        f"J^X: {err / np.abs(want).max() / EPS:.1f} eps of max|J^X|")
    assert np.abs(JX.sum(axis=0)).max() <= 1e-13 * np.abs(JX).max()


@pytest.mark.slow  # every class of H2O/STO-3G through the interpreter
def test_the_chain_on_h2o_is_pyscf() -> None:
    name = "h2o_sto3g"
    J1 = _chain_j1(name)
    e1 = golden.compare(f"jx1_{GOLDEN[name]}", J1)
    JX = _assemble(name, J1, _chain_ket(name))
    ex = golden.compare(f"jx_{GOLDEN[name]}", JX)
    assert np.abs(JX.sum(axis=0)).max() <= 1e-13 * np.abs(JX).max()
    print(f"\nchain: J1 {e1:.2e}, J^X {ex:.2e} of max|want|")


# --- the kernel's shape -------------------------------------------------------


def test_the_key_the_name_and_the_pq_class() -> None:
    assert jx_spec(2, 1).key() == "coulomb/d1/(2,1)/b1/jx/literal"
    assert jx_name((2, 1)) == build_jx_kernel((2, 1)).name == "jx_21"
    pqs = jx_pq_spec(2, 1, 1, 0)
    assert pqs.target is Target.J and pqs.derivative_order == 0
    assert split(pqs) == (4, 1)
    assert n_p1(2, 1) == len(hermite_order(4))


def test_a_jab_key_is_refused() -> None:
    with pytest.raises(ValueError, match="not a Target.JX key"):
        build_jx_kernel(Spec(Operator.COULOMB, 1, (1, 0), Target.JAB))


def test_the_parameters_and_the_one_output() -> None:
    k = build_jx_kernel((1, 2))
    assert [p.name for p in k.params] == [
        "n_rec", "n_ao", "n_rows", "zeta_ab", "XPA", "XPB", "s", "ea_bra",
        "eb_ket", "off_a", "off_b", "row", "Jp", "J1"]
    assert [p.name for p in k.params if p.intent is Intent.OUT] == ["J1"]
    shapes = {p.name: p.shape for p in k.params}
    assert shapes["Jp"][1].value == n_p1(1, 2)
    assert shapes["J1"][0].value == 3
    adds = [s for s in walk(k.body) if isinstance(s, AtomicAdd)]
    assert len(adds) == 6 and all(a.dst.array == "J1" for a in adds)
    assert any(isinstance(s, For) for s in walk(k.body))


def test_a_diagonal_pair_writes_the_a_term_alone() -> None:
    """`la == lb`: the B term's three adds sit under the off-diagonal test;
    `la != lb` builds no such branch."""
    def guarded(k) -> list:
        return [s for s in walk(k.body) if isinstance(s, If)
                and getattr(s.cond, "name", None) == "offd"]

    same = guarded(build_jx_kernel((2, 2)))
    assert len(same) == 1
    assert len([s for s in walk(same[0].then) if isinstance(s, AtomicAdd)]) == 3
    assert not guarded(build_jx_kernel((2, 1)))
