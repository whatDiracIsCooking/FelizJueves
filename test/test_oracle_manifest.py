"""The oracle's fixtures: they load without PySCF, refuse corruption by name,
and -- with the `oracle` group installed -- are what PySCF computes today.

Everything but the last test is fast-tier and needs numpy only: that is the
rule `test/oracle/` exists to establish (a comparison AGAINST a tracked
fixture is fast-tier; PySCF is needed only to regenerate one). The last test
is `oracle`-marked, so `FAST_TEST_ARGS` keeps it off the push gate, and it
skips -- says so, and proves nothing -- where PySCF is absent.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from oracle import manifest
from oracle.manifest import FixtureError, load

ROOT = Path(__file__).resolve().parents[1]
SEED = manifest.FIXTURES / "h2_sto3g.json"


# --- the tracked fixtures, with no PySCF -----------------------------------


def test_there_is_a_seed_fixture():
    assert SEED in manifest.tracked()


@pytest.mark.parametrize("path", manifest.tracked(), ids=lambda p: p.stem)
def test_tracked_fixture_loads_and_is_canonical(path):
    fx = load(path)
    assert fx.name == path.stem
    # Byte-for-byte what the one serialiser writes: full-precision floats, and
    # no hand edit the generator would undo.
    assert manifest.dumps(json.loads(path.read_text())) == path.read_text()
    schema = json.loads(path.read_text())["schema"]
    if fx.metric is not None:  # #411: schema 3, its two row blocks only
        assert sorted(fx.arrays) == sorted(manifest.METRIC_ARRAYS)
        assert schema == manifest.METRIC_SCHEMA
    elif fx.three_center is not None:  # #442: schema 4, its one array
        assert sorted(fx.arrays) == [fx.three_center.array]
        assert schema == manifest.THREE_CENTER_SCHEMA
    else:
        assert set(fx.arrays) >= set(manifest.REQUIRED_ARRAYS)
        assert schema == manifest.AO_SCHEMA
    for a in fx.arrays.values():
        assert not a.flags.writeable


def test_seed_fixture_is_h2_sto3g():
    fx = load(SEED)
    assert fx.provenance.basis == "sto-3g" and fx.provenance.cart is True
    assert [a.symbol for a in fx.atoms] == ["H", "H"]
    assert fx.atoms[1].coord == (0.0, 0.0, 1.4)
    assert fx.nao == 2 and fx.ao_loc == (0, 1, 2)
    assert all(s.l == 0 and len(s.exponents) == 3 for s in fx.shells)
    d = fx.density
    assert np.array_equal(d, d.T)
    # The recorded density is its recipe's, exactly -- an input, not a result.
    i = np.arange(fx.nao, dtype=np.float64)
    assert np.array_equal(d, 1.0 / (i[:, None] + i[None, :] + 1.0))
    s = fx.arrays["ovlp"]
    assert np.array_equal(s, s.T)


def test_manifest_imports_numpy_and_stdlib_only():
    """The loader must not need PySCF, or the fast tier would."""
    tree = ast.parse(Path(manifest.__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    third_party = mods - {"__future__", "json", "math", "sys", "dataclasses",
                          "pathlib", "typing"}
    assert third_party == {"numpy"}


def test_pyscf_is_kept_out_of_dev_and_the_gate():
    """`oracle` is its own group, reachable from nothing `uv sync` installs,
    and the push gate's marker expression excludes it."""
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text())[
        "dependency-groups"]
    assert any(str(r).startswith("pyscf") for r in groups["oracle"])

    def reach(name, seen=()):
        for item in groups[name]:
            if isinstance(item, dict):
                yield from reach(item["include-group"], (*seen, name))
            else:
                yield item

    assert not [r for r in reach("dev") if str(r).startswith("pyscf")]
    assert all(not (isinstance(i, dict) and i["include-group"] == "oracle")
               for g in groups.values() for i in g)
    config = (ROOT / "devtools" / "config.sh").read_text()
    fast = re.search(r"^FAST_TEST_ARGS=.*$", config, re.M).group(0)
    assert "not oracle" in fast


# --- corruption fails the loader, naming the file --------------------------


def _write(tmp_path, mutate):
    raw = json.loads(SEED.read_text())
    mutate(raw)
    path = tmp_path / "h2_sto3g.json"
    path.write_text(json.dumps(raw))
    return path


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r["provenance"].pop("libcint"), "missing key(s) ['libcint']"),
        (lambda r: r["provenance"].pop("date"), "missing key(s) ['date']"),
        (lambda r: r["arrays"]["density"]["values"].pop(), "truncated"),
        (lambda r: r["arrays"]["ovlp"].update(values=[1.0]), "truncated"),
        (lambda r: r["arrays"]["density"]["values"].__setitem__(1, 0.25),
         "not exactly symmetric"),
        (lambda r: r["arrays"].pop("density"), "missing ['density']"),
        (lambda r: r.pop("ao_loc"), "missing key(s) ['ao_loc']"),
        (lambda r: r.update(ao_loc=[0, 2, 2]), "ao_loc gives shell 0"),
        (lambda r: r["shells"][0]["coefficients"].pop(), "one row per primitive"),
        (lambda r: r["provenance"].update(cart="yes"), "cart must be"),
        (lambda r: r.update(schema=5), "schema 5"),
    ],
    ids=["no-libcint", "no-date", "density-short", "ovlp-short", "asymmetric",
         "no-density", "no-ao_loc", "bad-ao_loc", "ragged-shell", "cart-str",
         "schema"],
)
def test_corrupted_fixture_fails_naming_the_file(tmp_path, mutate, says):
    path = _write(tmp_path, mutate)
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value)
    assert says in str(e.value)


MATRIX_FIXTURES = {manifest.J_ARRAY: manifest.FIXTURES / "h2_sto3g_j.json",
                   manifest.K_ARRAY: manifest.FIXTURES / "h2_sto3g_k.json"}


def test_the_matrix_arrays_are_named_and_tracked():
    """`vj` (#86) and `vk` (#199) are distinct names, and each has a tracked
    fixture that carries it with the converted basis its AOs are in."""
    assert (manifest.J_ARRAY, manifest.K_ARRAY) == ("vj", "vk")
    for name, path in MATRIX_FIXTURES.items():
        fx = load(path)
        assert name in fx.arrays and fx.converted is not None
        assert fx.arrays[name].shape == (fx.nao, fx.nao)


@pytest.mark.parametrize("array", [manifest.J_ARRAY, manifest.K_ARRAY])
@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r, a: r.pop("converted"), "needs the converted basis"),
        (lambda r, a: r["arrays"][a].update(shape=[1, 4]), "has shape (1, 4)"),
    ],
    ids=["no-converted", "wrong-shape"],
)
def test_a_bad_matrix_array_fails_naming_the_file(tmp_path, array, mutate,
                                                  says):
    """The shape check covers `vk` exactly as it covers `vj`."""
    raw = json.loads(MATRIX_FIXTURES[array].read_text())
    mutate(raw, array)
    path = tmp_path / MATRIX_FIXTURES[array].name
    path.write_text(json.dumps(raw))
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value)
    assert f"arrays.{array} " in str(e.value) and says in str(e.value)


def test_truncated_file_fails_naming_the_file(tmp_path):
    path = tmp_path / "h2_sto3g.json"
    path.write_text(SEED.read_text()[:-200])
    with pytest.raises(FixtureError, match=re.escape(str(path))):
        load(path)


def test_manifest_cli_reports_each_fixture(capsys):
    assert manifest.main() == 0
    out = capsys.readouterr().out
    assert "h2_sto3g: nao=2, generated " in out


# --- #240: schema 2 and the one-electron arrays ----------------------------

ONE_E_FIXTURE = manifest.FIXTURES / "h2o_ccpvdz_j.json"  # has a d shell


def test_a_schema_1_file_is_refused_naming_the_bump(tmp_path):
    """Not a KeyError on a missing array: the message says what changed."""
    path = _write(tmp_path, lambda r: r.update(schema=1))
    with pytest.raises(FixtureError) as e:
        load(path)
    msg = str(e.value)
    assert str(path) in msg and "schema 1" in msg
    assert "bump to 2 (#240)" in msg and "regenerate" in msg


def test_the_one_e_arrays_are_named_and_shaped_by_cartesian_aos():
    assert (manifest.AO_SCHEMA, manifest.METRIC_SCHEMA,
            manifest.THREE_CENTER_SCHEMA, manifest.SCHEMA) == (2, 3, 4, 4)
    assert manifest.SCHEMAS == (2, 3, 4)
    assert manifest.ONE_E_ARRAYS == ("int1e_ovlp", "int1e_kin", "int1e_nuc",
                                     "hcore")
    fx = load(ONE_E_FIXTURE)
    assert fx.provenance.cart is True and fx.converted is not None
    # A d shell is 6 Cartesian AOs, not 5: H2O/cc-pVDZ is 25, not 24.
    assert any(s.l == 2 for s in fx.converted.shells)
    assert fx.nao == 25
    for name in manifest.ONE_E_ARRAYS:
        a = fx.arrays[name]
        assert a.shape == (25, 25)
        # libcint computes both triangles, so symmetric to rounding only
        assert np.max(np.abs(a - a.T)) <= 1e-14 * np.max(np.abs(a)), name
    assert np.array_equal(fx.arrays["int1e_ovlp"], fx.arrays["ovlp"])


def test_every_converted_fixture_carries_the_one_e_arrays_or_its_twin_does():
    """The five HF/cc-pVTZ ERI files and the three `_schwarz` files are the
    declared exceptions (`Case.record_one_e`), and a twin -- same molecule,
    basis, Cartesian flag (`hf_ccpvtz`, the `_j` cases) -- carries the four
    for them."""
    from oracle import generate

    fixtures = [load(p) for p in manifest.tracked()]

    def key(fx):
        p = fx.provenance
        return (p.mol["atom"], p.mol["unit"], p.basis, p.cart)

    carriers = {key(fx) for fx in fixtures
                if manifest.OVLP_ARRAY in fx.arrays}
    skipped = set()
    for fx in fixtures:
        has = [n for n in manifest.ONE_E_ARRAYS if n in fx.arrays]
        if fx.converted is None:
            assert not has, fx.name
            continue
        if generate.CASES[fx.name].record_one_e:
            assert has == list(manifest.ONE_E_ARRAYS), fx.name
        else:
            assert not has and key(fx) in carriers, fx.name
            skipped.add(fx.name)
    assert skipped == {"hf_ccpvtz_eri", *(f"hf_ccpvtz_eri_l{L}"
                                          for L in (9, 10, 11, 12)),
                       "h2_sto3g_schwarz", "h2o_sto3g_schwarz",
                       "h2o_631g_schwarz",
                       "h2o_sto3g_x_631g", "h2o_sto3g_x_ccpvdz"}


def test_the_cross_overlap_fixtures_name_their_rows_and_fit_them():
    """Every `int1e_ovlp_cross:<rows>` array names a tracked fixture with a
    converted basis, and is `(rows' nao, nao)`; `cross_overlap` refuses one
    whose rows are absent or the wrong size."""
    paths = manifest.cross_fixtures()
    assert {p.stem for p in paths} == {
        "h2o_sto3g_x_631g", "h2o_sto3g_x_ccpvdz", "h2o_sto3g_x_631g_moved"}
    for p in paths:
        fx = load(p)
        rows, S = manifest.cross_overlap(fx)
        assert rows.name == "h2o_sto3g_j"
        assert S.shape == (rows.nao, fx.nao)


def test_a_cross_overlap_array_that_cannot_be_read_is_refused(tmp_path):
    import json
    import shutil

    src = manifest.FIXTURES / "h2o_sto3g_x_631g.json"
    raw = json.loads(src.read_text())
    (name,) = [n for n in raw["arrays"]
               if n.startswith(manifest.CROSS_PREFIX)]
    # absent rows
    bad = dict(raw, arrays={**raw["arrays"]})
    bad["arrays"][manifest.CROSS_PREFIX + "nope"] = bad["arrays"].pop(name)
    path = tmp_path / src.name
    path.write_text(json.dumps(bad))
    with pytest.raises(manifest.FixtureError, match="no fixture"):
        manifest.cross_overlap(load(path))
    # wrong row count: rows of another size beside it
    shutil.copy(manifest.FIXTURES / "h2o_631g_j.json", tmp_path)
    bad["arrays"] = {**raw["arrays"]}
    bad["arrays"][manifest.CROSS_PREFIX + "h2o_631g_j"] = bad["arrays"].pop(
        name)
    path.write_text(json.dumps(bad))
    with pytest.raises(manifest.FixtureError, match="expected"):
        manifest.cross_overlap(load(path))
    # wrong column count is refused at load
    bad["arrays"] = {**raw["arrays"]}
    bad["arrays"][name] = {"shape": [7, 2], "values": [0.0] * 14}
    path.write_text(json.dumps(bad))
    with pytest.raises(manifest.FixtureError, match="expected"):
        load(path)


# --- #411: schema 3 and the metric pieces, both directions -----------------

PIECE = manifest.FIXTURES / "h2o_ccpvtz_jkfit_metric_rows0_43.json"


def test_schema_2_3_and_4_files_load_side_by_side():
    """Every AO fixture is still schema 2, unchanged; the metric pieces are
    schema 3 and the only files that are, and the three-center pieces
    (#442) schema 4 and the only files that are."""
    by_schema = {}
    for p in manifest.tracked():
        by_schema.setdefault(json.loads(p.read_text())["schema"], set()).add(
            p.stem)
    assert set(by_schema) == {2, 3, 4}
    assert all(load(manifest.FIXTURES / f"{n}.json").metric is not None
               for n in by_schema[3])
    assert by_schema[3] == {p.stem for p in manifest.tracked()
                            if p.stem.startswith("h2o_ccpvtz_jkfit_metric")}
    assert all(load(manifest.FIXTURES / f"{n}.json").three_center is not None
               for n in by_schema[4])
    assert by_schema[4] == {"h2o_sto3g_jkfit_3c_cols0_166"}
    for n in by_schema[2]:
        fx = load(manifest.FIXTURES / f"{n}.json")
        assert fx.metric is None and fx.three_center is None


def test_a_schema_2_fixture_still_requires_its_density(tmp_path):
    path = _write(tmp_path, lambda r: r["arrays"].pop("density"))
    with pytest.raises(FixtureError, match=re.escape("missing ['density']")):
        load(path)


@pytest.mark.parametrize("schema", [3, 4])
def test_a_schema_3_or_4_ao_fixture_still_requires_its_density(tmp_path,
                                                                schema):
    """Schemas 3 and 4 lift the density requirement for a metric or
    three-center piece only."""
    ok = _write(tmp_path, lambda r: r.update(schema=schema))
    assert load(ok).metric is None and load(ok).three_center is None
    path = _write(tmp_path, lambda r: (r.update(schema=schema),
                                       r["arrays"].pop("density")))
    with pytest.raises(FixtureError, match=re.escape("missing ['density']")):
        load(path)


def test_a_metric_piece_loads_without_density_and_refuses_one():
    fx = load(PIECE)
    assert fx.metric is not None and fx.converted is None
    assert "density" not in fx.arrays
    with pytest.raises(FixtureError, match="records no density"):
        _ = fx.density


def _write_piece(tmp_path, mutate, src=PIECE):
    raw = json.loads(src.read_text())
    mutate(raw)
    path = tmp_path / src.name
    path.write_text(json.dumps(raw))
    return path


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r.update(schema=2), "a metric piece is schema 3"),
        (lambda r: r["arrays"].update(density={"shape": [1], "values": [1.0]}),
         "exactly the arrays"),
        (lambda r: r["arrays"].pop(manifest.AUX_OVLP_ARRAY),
         "exactly the arrays"),
        (lambda r: r["metric"].update(rows=[0, 42]), "shell boundaries"),
        (lambda r: r["metric"].update(rows=[0, 81]), "is named"),
        (lambda r: r["metric"].pop("parent"), "exactly the keys"),
        (lambda r: r.update(converted=r["metric"]["converted"]),
         "metric.converted, not in converted"),
        (lambda r: r["arrays"][manifest.METRIC_ARRAY].update(
            shape=[43, 165], values=[0.0] * 43 * 165), "has shape (43, 165)"),
    ],
    ids=["schema-2", "with-density", "one-array", "mid-shell", "misnamed",
         "no-parent", "top-converted", "narrow"],
)
def test_a_bad_metric_piece_fails_naming_the_file(tmp_path, mutate, says):
    path = _write_piece(tmp_path, mutate)
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value) and says in str(e.value)


def test_an_ao_fixture_cannot_carry_a_metric_array(tmp_path):
    path = _write(tmp_path, lambda r: r["arrays"].update(
        {manifest.METRIC_ARRAY: r["arrays"]["ovlp"]}))
    with pytest.raises(FixtureError, match="belong to a metric piece only"):
        load(path)


# --- #442: schema 4 and the three-center pieces, both directions ----------

PIECE_3C = manifest.FIXTURES / "h2o_sto3g_jkfit_3c_cols0_166.json"
ARRAY_3C = manifest.THREE_CENTER_PREFIX + "h2o_sto3g_j"


def test_a_three_center_piece_loads_without_density_and_refuses_one():
    fx = load(PIECE_3C)
    assert fx.three_center is not None and fx.metric is None
    assert fx.converted is None and "density" not in fx.arrays
    assert fx.three_center.array == ARRAY_3C
    assert fx.arrays[ARRAY_3C].shape == (7, 7, 166)
    with pytest.raises(FixtureError, match="records no density"):
        _ = fx.density


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r.update(schema=3), "a three-center piece is schema 4"),
        (lambda r: r.update(schema=2), "a three-center piece is schema 4"),
        (lambda r: r.update(schema=1), "schema 1"),
        (lambda r: r["arrays"].update(density={"shape": [1], "values": [1.0]}),
         "exactly the array"),
        (lambda r: r["three_center"].update(cols=[0, 12]), "shell boundaries"),
        (lambda r: r["three_center"].update(cols=[0, 10]), "is named"),
        (lambda r: r["three_center"].pop("parent"), "exactly the keys"),
        (lambda r: r["three_center"].update(parent="h2o_631g_j"),
         "exactly the array"),
        (lambda r: r.update(converted=r["three_center"]["converted"]),
         "three_center.converted, not in converted"),
        (lambda r: r.update(metric=r["three_center"]), "not both"),
        (lambda r: r["arrays"][ARRAY_3C].update(
            shape=[7, 6, 166], values=[0.0] * 7 * 6 * 166),
         "has shape (7, 6, 166)"),
    ],
    ids=["schema-3", "schema-2", "schema-1", "with-density", "mid-shell",
         "misnamed", "no-parent", "other-parent", "top-converted",
         "also-metric", "not-square"],
)
def test_a_bad_three_center_piece_fails_naming_the_file(tmp_path, mutate,
                                                        says):
    path = _write_piece(tmp_path, mutate, src=PIECE_3C)
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value) and says in str(e.value)


def test_an_ao_fixture_cannot_carry_a_three_center_array(tmp_path):
    path = _write(tmp_path, lambda r: r["arrays"].update(
        {manifest.THREE_CENTER_PREFIX + "x": r["arrays"]["ovlp"]}))
    with pytest.raises(FixtureError, match="belong to a three-center piece"):
        load(path)


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: (r.pop("converted"), r["arrays"].pop("vj")),
         "arrays.int1e_ovlp needs the converted basis"),
        # cart=True is load-bearing: read as spherical, the d shell's 6
        # Cartesian AOs are refused as the 5 a spherical d would have.
        (lambda r: r["provenance"].update(cart=False),
         "ao_loc gives shell 4 6 AOs, its l and contractions need 5"),
        (lambda r: r["arrays"]["hcore"].update(
            shape=[24, 24], values=[0.0] * 576), "arrays.hcore has shape"),
        (lambda r: r["arrays"].pop("int1e_nuc"), "all four or none"),
        (lambda r: r["arrays"]["int1e_ovlp"]["values"].__setitem__(0, 0.5),
         "arrays.int1e_ovlp is not exactly arrays.ovlp"),
    ],
    ids=["no-converted", "not-cart", "wrong-shape", "three-of-four",
         "ovlp-mismatch"],
)
def test_a_bad_one_e_array_fails_naming_the_file(tmp_path, mutate, says):
    raw = json.loads(ONE_E_FIXTURE.read_text())
    mutate(raw)
    path = tmp_path / ONE_E_FIXTURE.name
    path.write_text(json.dumps(raw))
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value)
    assert says in str(e.value)


# --- the drift check: needs PySCF ------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("path", manifest.tracked(), ids=lambda p: p.stem)
def test_fixture_is_what_pyscf_computes_today(path):
    """Regenerate in memory and compare byte-for-byte on the numbers.

    A failure here is news about PySCF/libcint, not proof the fixture was
    wrong: a fixture records what libcint computed on a date. The message
    names both versions and that date.
    """
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    from oracle import generate

    msg = generate.drift(path.stem)
    assert msg is None, msg
    # And the arrays, compared as doubles rather than as text.
    fresh = generate.build(path.stem)
    fx = load(path)
    for name, a in fresh["arrays"].items():
        got = np.array(a["values"]).reshape(a["shape"])
        assert np.array_equal(got, fx.arrays[name]), name
