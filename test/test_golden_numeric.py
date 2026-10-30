"""The numerical golden store (#655, `oracle.numeric`): the manifest against
its recipe and its files, the comparison helper, and the host paths against
the seeded arrays.

THE STORE CHECKS (fast, no PySCF): `check_manifest` finds nothing -- every
recipe entry has its record, every record its file, every file its sha256,
shape and dtype, every key a registry case, every referenced basis file its
`provenance.json` sha256 and every referenced density its fixture's -- and it
DOES find a tampered record, a stray file and a missing one. The store stays
inside the README's budget, and no npy entry is a copy of a fixture array.

THE HOST PATHS, against the npy entries through `compare` -- the helper a
device test calls -- at the fp64 tier (1e-12 of the golden's largest
element): our `coulomb_matrix` over `numeric.basis` (the `data/` file read
with no PySCF) and the fixture's density, and our `metric_matrix` over the
fitting file's s and p shells (every pair at lP + lQ <= 2, inside
`FAST_L_MAX`) against those rows and columns of the golden, and (#693)
`host_df.metric_derivative` over the same shells against each axis of the
uncontracted d1 metric. H2O/STO-3G and
the s/p metric blocks are fast; 6-31G and cc-pVDZ's J are `slow`, by
test_ao_j.py's cost rule. Measured worst at writing (2026-10-30, the store's
pyscf 2.14.0 / libcint 6.1.3): J 1.2e-15 (STO-3G), 3.0e-15 (6-31G),
2.1e-15 (cc-pVDZ); metric s/p 6.3e-15.

THE ORACLE TEST (`-m oracle`, PySCF): each npy entry rebuilt in memory is
the tracked `.npy` bit for bit.
"""

from __future__ import annotations

import json
import shutil
from functools import cache

import numpy as np
import pytest
from oracle import manifest
from oracle import numeric as golden

from felizjueves.ao.shell import Basis
from felizjueves.eri.density_fit.metric import metric_matrix
from felizjueves.eri.j_coulomb.j import coulomb_matrix
from felizjueves.gradient import host_df
from felizjueves.ir import Scalar
from felizjueves.reference.pq import pq

# The README's budget: one array under half the repo's 512 KiB
# check-added-large-files cap, the whole store (arrays + manifest) under 1 MiB.
FILE_BUDGET = 256 * 1024
STORE_BUDGET = 1024 * 1024

NPY = golden.entries("npy")
J_FAST = ("j_h2o_sto3g_bse",)
J_SLOW = ("j_h2o_631g_bse", "j_h2o_ccpvdz_bse")
METRICS = tuple(n for n in NPY if golden.entry(n).what == "metric")
IP1 = tuple(n for n in NPY if golden.entry(n).what.startswith("metric.d1."))


def test_the_manifest_the_recipe_and_the_files_agree():
    bad = golden.check_manifest()
    assert not bad, "\n".join(bad)


def test_the_seed_is_j_and_the_metric_both_kinds():
    kinds = {(golden.entry(n).quantity, golden.entry(n).kind)
             for n in golden.entries()}
    assert kinds == {("j", "npy"), ("j", "fixture"), ("metric", "npy"),
                     ("metric", "fixture")}
    assert set(J_FAST + J_SLOW) | set(METRICS) | set(IP1) == set(NPY)
    # #693's three axes of one uncontracted d1 metric
    assert sorted(golden.entry(n).component for n in IP1) == ["x", "y", "z"]


def test_the_store_stays_inside_its_budget():
    sizes = {p.name: p.stat().st_size for p in golden.STORE.iterdir()
             if p.suffix in (".npy", ".json")}
    over = {k: v for k, v in sizes.items() if v > FILE_BUDGET}
    assert not over, f"over {FILE_BUDGET} bytes each: {over}"
    assert sum(sizes.values()) <= STORE_BUDGET, sizes


def test_no_npy_entry_copies_a_fixture_array():
    """The store holds an array only where test/oracle/fixtures/ does not:
    each npy entry differs from every indexed fixture array of its shape."""
    held = [golden.want(n) for n in golden.entries("fixture")]
    for name in NPY:
        a = golden.want(name)
        same = [b for b in held if b.shape == a.shape]
        assert all(not np.array_equal(a, b) for b in same), name


@pytest.mark.parametrize("name", golden.entries("fixture"))
def test_a_fixture_entry_loads_the_fixtures_array(name):
    e = golden.entry(name)
    a = golden.want(name)
    if e.fixture is not None:
        fx = manifest.load(manifest.FIXTURES / f"{e.fixture}.json")
        assert np.array_equal(a, fx.arrays[e.array])
    else:
        assert np.array_equal(a, manifest.metric_set(e.metric_set).metric)
    assert not a.flags.writeable


@pytest.mark.parametrize("name", NPY)
def test_an_npy_entrys_inputs_are_its_fixtures(name):
    e = golden.entry(name)
    rec = golden.record(name)
    basis = golden.basis(name)
    assert basis.n_ao == golden.want(name).shape[0] == rec["shape"][0]
    assert len(golden.atoms(name)) == len(
        manifest.load(manifest.FIXTURES / f"{e.molecule}.json").atoms)
    if e.quantity == "j":
        assert golden.density(name).shape == tuple(rec["shape"])


# --- the comparison helper --------------------------------------------------


def test_compare_passes_the_golden_itself_and_fails_a_perturbation():
    w = golden.want("j_h2o_ccpvdz_bse")
    assert golden.compare("j_h2o_ccpvdz_bse", w.copy()) == 0.0
    off = w.copy()
    off[3, 5] += 1e-10 * float(np.max(np.abs(w)))
    with pytest.raises(AssertionError, match="F64 tier"):
        golden.compare("j_h2o_ccpvdz_bse", off)


def test_compare_refuses_a_shape_and_crosses_no_tier():
    w = golden.want("j_h2o_sto3g_bse")
    with pytest.raises(AssertionError, match="shape"):
        golden.compare("j_h2o_sto3g_bse", w[:-1])
    with pytest.raises(TypeError, match="tiers do not cross"):
        golden.compare("j_h2o_sto3g_bse", w.astype(np.float32))
    assert golden.compare("j_h2o_sto3g_bse", w.astype(np.float32),
                          precision=Scalar.F32) <= 1e-7


def test_compare_judges_a_block_against_the_whole_arrays_scale():
    name = "metric_h2o_ccpvdz_jkfit_bse"
    w = golden.want(name)
    rows = [0, 3, 7]
    block = w[np.ix_(rows, rows)].copy()
    assert golden.compare(name, block, index=(rows, rows)) == 0.0
    block[0, 0] += 2e-12 * float(np.max(np.abs(w)))
    with pytest.raises(AssertionError):
        golden.compare(name, block, index=(rows, rows))


def test_an_unknown_entry_is_refused_by_name():
    with pytest.raises(golden.GoldenError, match="no numeric golden"):
        golden.want("j_nothing")


# --- check_manifest finds what it should ------------------------------------


@pytest.fixture
def store_copy(tmp_path, monkeypatch):
    dst = tmp_path / "numeric"
    shutil.copytree(golden.STORE, dst)
    monkeypatch.setattr(golden, "STORE", dst)
    monkeypatch.setattr(golden, "MANIFEST", dst / "manifest.json")
    golden._manifest.cache_clear()
    golden.want.cache_clear()
    yield dst
    golden._manifest.cache_clear()
    golden.want.cache_clear()


def _edit(store, name, **fields):
    raw = json.loads((store / "manifest.json").read_text())
    raw["entries"][name].update(fields)
    (store / "manifest.json").write_text(json.dumps(raw))
    golden._manifest.cache_clear()


def test_check_manifest_finds_a_wrong_sha_shape_and_density(store_copy):
    _edit(store_copy, "j_h2o_sto3g_bse", sha256="0" * 64, shape=[7, 8])
    _edit(store_copy, "j_h2o_631g_bse",
          density={"fixture": "h2o_631g_j", "sha256": "0" * 64})
    bad = "\n".join(golden.check_manifest())
    assert "j_h2o_sto3g_bse: j_h2o_sto3g_bse.npy's sha256" in bad
    assert "the manifest says float64 [7, 8]" in bad
    assert "j_h2o_631g_bse: density is" in bad


def test_check_manifest_finds_a_wrong_key_and_a_wrong_basis(store_copy):
    _edit(store_copy, "metric_h2o_ccpvdz_jkfit_bse", key=["mmd", "metric", 0, 4])
    raw = json.loads((store_copy / "manifest.json").read_text())
    raw["entries"]["j_h2o_ccpvdz_bse"]["basis"]["sha256"] = "0" * 64
    (store_copy / "manifest.json").write_text(json.dumps(raw))
    golden._manifest.cache_clear()
    bad = "\n".join(golden.check_manifest())
    assert "metric_h2o_ccpvdz_jkfit_bse: key is" in bad
    assert "j_h2o_ccpvdz_bse: basis is" in bad


def test_check_manifest_finds_a_stray_and_a_missing_file(store_copy):
    (store_copy / "j_h2o_sto3g_bse.npy").rename(store_copy / "stray.npy")
    bad = "\n".join(golden.check_manifest())
    assert "j_h2o_sto3g_bse.npy is missing" in bad
    assert "files no entry names: ['stray.npy']" in bad


def test_check_manifest_finds_an_entry_the_recipe_lacks(store_copy):
    raw = json.loads((store_copy / "manifest.json").read_text())
    raw["entries"]["extra"] = raw["entries"]["j_h2o_sto3g"]
    (store_copy / "manifest.json").write_text(json.dumps(raw))
    golden._manifest.cache_clear()
    assert any("not the recipe's" in m for m in golden.check_manifest())


# --- the host paths against the npy entries ----------------------------------


@pytest.mark.parametrize("name", [
    *[pytest.param(n, id=n) for n in J_FAST],
    *[pytest.param(n, id=n, marks=pytest.mark.slow) for n in J_SLOW]])
def test_host_j_over_the_basis_file_is_the_golden(name):
    J = np.array(coulomb_matrix(golden.basis(name), golden.density(name),
                                pq=pq, tau=0.0))
    err = golden.compare(name, J)
    print(f"\n{name}: max |J - golden| / max |golden| = {err:.2e}")


@cache
def _sp(name: str) -> tuple[Basis, list[int]]:
    """`name`'s fitting shells with l <= 1, and their AOs' indices."""
    aux = golden.basis(name)
    keep = [k for k, s in enumerate(aux.shells) if s.l <= 1]
    rows = [aux.offsets[k] + a for k in keep
            for a in range(aux.shells[k].n_cart)]
    return Basis(tuple(aux.shells[k] for k in keep)), rows


@pytest.mark.parametrize("name", METRICS)
def test_host_metric_over_the_s_and_p_fitting_shells_is_the_golden(name):
    sub, rows = _sp(name)
    got = np.array(metric_matrix(sub, pq=pq))
    err = golden.compare(name, got, index=(rows, rows))
    print(f"\n{name} (s, p): max |M - golden| / max |golden| = {err:.2e}")


@pytest.mark.parametrize("name", IP1)
def test_host_metric_derivative_over_the_s_and_p_fitting_shells_is_the_golden(
        name):
    # #693: `host_df.metric_derivative` (each centre by the Cartesian rule),
    # the axis of this entry, against -int2c2e_ip1 on those rows and columns
    sub, rows = _sp(name)
    k = "xyz".index(golden.entry(name).component)
    got = np.array(host_df.metric_derivative(sub, pq=pq)[k])
    err = golden.compare(name, got, index=(rows, rows))
    print(f"\n{name} (s, p): max |M1 - golden| / max |golden| = {err:.2e}")


# --- the oracle: regenerated bit for bit -------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", NPY)
def test_numeric_golden_is_what_pyscf_computes_today(name):
    """Rebuild and compare the whole `.npy`, bytes and all. A failure is
    news about PySCF/libcint (versions in the message), not proof the array
    was wrong."""
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    from oracle import generate

    msg = generate.numeric_drift(name)
    assert msg is None, msg
