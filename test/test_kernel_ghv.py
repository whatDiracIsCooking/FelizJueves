"""`kernel_ghv`: the nuclear-attraction gradient on the HGP engine
(`ghv_{la}{lb}`, `V` at d1 on `Engine.HGP`), interpreted. Shells through f
(`kernel_ghv.L_MAX = 3`).

THE WITNESSES, neither of which shares the kernel's recurrence:

- THE MMD HOST, `gradient.host.potential_gradient` (itself held to finite
  differences of the libcint-witnessed `V`, test/test_gradient_host.py):
  per class, one launch over every shell pair `i <= j` of that class into a
  zeroed `G`, against the host over the same pairs, `M` non-symmetric (the
  weight rule is exact for any `M`). Every ordered class (ss) through (ff)
  on HF/cc-pVTZ (the one fixture whose shells reach f, both orientations
  occurring with `i <= j`), and the s, p classes on the SYNTHETIC off-axis
  basis of test_kernel_hgp.py (three centres, each a nucleus, plus a FOURTH
  NUCLEUS with no shell, so a Hellmann-Feynman row lands where no basis
  function is). Within `TOL = 16 eps` of test_kernel_g1e.py's SPLIT-CENTRE
  SCALE (its docstring: `max|G|` is no scale where a one-centre pair's
  `dA` and `dB` cancel on one atom -- HF's own (df) is 1.05e-13 of
  `max|G|` and 1.4 eps of the scale): another recurrence, and `C`'s row a
  difference of `A`'s and `B`'s terms where the host forms it from
  `[t + 1_k]` (measured: worst 6.0 eps, at (dd)). L for the tier cut is
  `la + lb + 1`, test_kernel_g1e.py's: (sd), (pp) and below are fast.
- PYSCF, through #550's gradient fixtures: the `potential` term of
  test_rhf_gradient.py's witness (`grad:h` less its kinetic half, libcint's
  `int1e_ipnuc` and `int1e_iprinv` contracted with the fixture's density),
  on H2O/STO-3G and 6-31G (all s and p), within that module's `ABS_TOL =
  1e-10` absolute (measured: 7.9e-14 and 4.9e-14).

Plus translational invariance (the rows sum to zero, `1e-13` of `max|G|`)
and the kernel's shape: the key and name, the refusals, the parameter block
equal to the staging's names, every local read (#92's filter), `G` only
AtomicAdd-ed, the recurrence statements `hgp.recurrence`'s.

Nothing compiles and nothing launches on a card.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest
from test_kernel_g1e import EPS, TOL, _scale
from test_kernel_hgp import NUCLEI, SYNTHETIC
from test_rhf_gradient import ABS_TOL, _witness
from tiers import by_l

from felizjueves.gradient import host
from felizjueves.gradient import kernel_ghv as kg
from felizjueves.ir import (
    Assign,
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
from felizjueves.reference.boys import boys
from felizjueves.reference.pq import one_centre
from felizjueves.spec import Engine, Operator, Spec, Target

CLASSES = [(la, lb) for la in range(2) for lb in range(2)]  # s and p
ALL = [(la, lb) for la in range(4) for lb in range(4)]  # through (ff)
HOST_TOL = 1e-13  # of max|G|, the whole s, p builds (module docstring)
H2O = ("h2o_sto3g_grad", "h2o_631g_grad")
_I32 = ("rec0", "rec1", "off_a", "off_b", "atom_a", "atom_b")


@cache
def _fixture(name: str):
    fx = manifest.load(manifest.FIXTURES / f"{name}.json")
    return (ob.from_fixture(fx),
            tuple((float(a.charge), tuple(a.coord)) for a in fx.atoms))


def _m(n: int) -> np.ndarray:
    return np.random.default_rng(5).standard_normal((n, n))


def _arrays(basis, nuclei, la, lb, pairs=None) -> dict[str, np.ndarray]:
    rec = host.pair_records(basis, nuclei, la, lb, pairs=pairs)
    return {k: np.asarray(v, dtype=np.int32 if k in _I32 else float)
            for k, v in rec.items()}


def launch(basis, nuclei, M, la, lb, G, *, pairs=None, threads=3
           ) -> np.ndarray:
    """One `ghv_{la}{lb}` launch over the class's shell pairs (or `pairs`),
    interpreted, adding into `G`."""
    arrs = _arrays(basis, nuclei, la, lb, pairs)
    n_pair = len(arrs["rec0"])
    grid, block = kg.launch_shape(n_pair, threads)
    return Interpreter(kg.build_ghv_kernel((la, lb)),
                       launch=Launch(grid, block, 1), boys=boys).run(
        n_pair=n_pair, n_rec=len(arrs["s_ab"]), n_ao=basis.n_ao,
        n_atom=len(nuclei), M=M, Z=np.array([z for z, _ in nuclei]),
        Cnuc=np.array([c for _, c in nuclei], dtype=float), G=G,
        **arrs)["G"]


def whole(basis, nuclei, M) -> np.ndarray:
    """Every class of `basis` launched once into one zeroed `G`."""
    G = np.zeros((len(nuclei), 3))
    for la, lb in sorted({(a.l, b.l) for k, a in enumerate(basis.shells)
                          for b in basis.shells[k:]}):
        G = launch(basis, nuclei, M, la, lb, G)
    return G


def _host(basis, nuclei, M, pairs=None) -> np.ndarray:
    return np.array(host.potential_gradient(basis, M, nuclei,
                                            one_centre=one_centre,
                                            pairs=pairs))


def _rel(got, want) -> float:
    return float(np.abs(got - want).max() / np.abs(want).max())


# ---------------------------------------------------------------------------
# The key, the name, the refusals, the shape
# ---------------------------------------------------------------------------


def test_the_key_and_the_name() -> None:
    assert kg.ghv_spec(1, 0).key() == "nuclear/d1/(1,0)/b1/v.hgp/literal"
    assert kg.ghv_spec(1, 0).key() == Spec(
        Operator.NUCLEAR, 0, (1, 0), Target.V, engine=Engine.HGP).key(
        ).replace("/d0/", "/d1/")
    assert [kg.ghv_name(c) for c in CLASSES] == [
        "ghv_00", "ghv_01", "ghv_10", "ghv_11"]
    for c in CLASSES:
        assert kg.build_ghv_kernel(kg.ghv_spec(*c)).name == kg.ghv_name(c)


def test_the_refusals() -> None:
    with pytest.raises(ValueError, match="shells through f only"):
        kg.build_ghv_kernel((4, 0))
    with pytest.raises(ValueError, match="not a d1 v.hgp key"):
        kg.build_ghv_kernel(Spec(Operator.NUCLEAR, 1, (0, 0), Target.V))
    with pytest.raises(ValueError, match="not a d1 v.hgp key"):
        kg.build_ghv_kernel(Spec(Operator.NUCLEAR, 0, (0, 0), Target.V,
                                 engine=Engine.HGP))
    with pytest.raises(ValueError, match="derivative order 1 on target "
                                         "v.rys"):
        Spec(Operator.NUCLEAR, 1, (0, 0), Target.V, engine=Engine.RYS)
    with pytest.raises(ValueError, match="positive int"):
        kg.launch_shape(0)


def test_the_parameter_block_is_the_stagings() -> None:
    staged = set(_arrays(SYNTHETIC, NUCLEI, 1, 1))
    k = kg.build_ghv_kernel((1, 1))
    assert staged == {p.name for p in k.params if p.array} - {
        "M", "Z", "Cnuc", "G"}
    assert set(kg.pair_arrays()) <= staged
    assert [p.name for p in k.params if not p.array] == [
        "n_pair", "n_rec", "n_ao", "n_atom"]


def _read(stmts) -> set[str]:
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
    return read


@pytest.mark.parametrize("cls", [by_l(c, sum(c) + 1, id=f"{c[0]}{c[1]}")
                                 for c in ALL])
def test_every_local_is_read_and_g_is_only_added_to(cls) -> None:
    stmts = list(walk(kg.build_ghv_kernel(cls).body))
    declared = {s.name for s in stmts if isinstance(s, Decl)}
    assert declared <= _read(stmts), sorted(declared - _read(stmts))
    adds = [s for s in stmts if isinstance(s, AtomicAdd)]
    assert {s.dst.array for s in adds} == {"G"}
    assert len(adds) == 9  # three per nucleus, six per pair
    assert not any(isinstance(s, Assign) and isinstance(s.dst, Index)
                   for s in stmts)
    assert sum(isinstance(s, Boys) for s in stmts) == 1


def test_ss_has_no_lowered_table() -> None:
    # (ss): no a - 1_k or b - 1_k target, so table 0 is absent; table 2's
    # (0|1_k) is one HRR step from (1_k|0), so AB is read even here.
    names = {s.name for s in walk(kg.build_ghv_kernel((0, 0)).body)
             if isinstance(s, Decl)}
    assert not any(n.startswith("acc0_") for n in names), names
    assert {"PA_x", "WP_x", "AB_x", "ea2", "eb2"} <= names


# ---------------------------------------------------------------------------
# Against the MMD host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "cls"), [
    *(pytest.param("hf_ccpvtz", c, id=f"hf_ccpvtz-{c[0]}{c[1]}",
                   marks=by_l(c, sum(c) + 1).marks) for c in ALL),
    *(pytest.param("synthetic", c, id=f"synthetic-{c[0]}{c[1]}")
      for c in CLASSES)])
def test_one_launch_over_the_class_is_the_host_gradient(name, cls) -> None:
    basis, nuclei = ((SYNTHETIC, NUCLEI) if name == "synthetic"
                     else _fixture(name))
    pairs = host.class_pairs(basis, *cls)
    assert pairs, cls
    M = _m(basis.n_ao)
    G = launch(basis, nuclei, M, *cls, np.zeros((len(nuclei), 3)))
    want = _host(basis, nuclei, M, pairs)
    scale = _scale(Target.V, basis, nuclei, M, pairs)
    err = float(np.abs(G - want).max())
    assert err <= TOL * scale, f"{name} {cls}: {err / scale / EPS:.1f} eps"
    assert np.abs(G.sum(axis=0)).max() <= TOL * scale


def test_the_bare_nucleus_takes_its_hellmann_feynman_row() -> None:
    # NUCLEI[3] carries no shell: its row is the Hellmann-Feynman term alone,
    # and it is not zero.
    M = _m(SYNTHETIC.n_ao)
    G = whole(SYNTHETIC, NUCLEI, M)
    want = _host(SYNTHETIC, NUCLEI, M)
    assert _rel(G, want) <= HOST_TOL
    assert np.abs(G[3]).max() > 1e-3 * np.abs(G).max()


def test_g_is_added_to_not_overwritten() -> None:
    M = _m(SYNTHETIC.n_ao)
    G0 = np.arange(12.0).reshape(4, 3)
    G = launch(SYNTHETIC, NUCLEI, M, 0, 1, G0.copy())
    want = launch(SYNTHETIC, NUCLEI, M, 0, 1, np.zeros((4, 3)))
    assert np.abs(G - (G0 + want)).max() <= 1e-14 * np.abs(G).max()


# ---------------------------------------------------------------------------
# Against PySCF, through #550's gradient fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", H2O)
def test_the_potential_term_is_pyscfs(name) -> None:
    w, basis, nuclei = _witness(name)
    assert max(s.l for s in basis.shells) <= kg.L_MAX
    G = whole(basis, nuclei, w.density)
    err = float(np.abs(G - w.terms["potential"]).max())
    assert err <= ABS_TOL, f"{name}: {err:.2e} from PySCF"
    assert _rel(G, _host(basis, nuclei, w.density)) <= HOST_TOL
