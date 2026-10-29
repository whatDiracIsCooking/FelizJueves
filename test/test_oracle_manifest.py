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
    elif fx.gradient is not None:  # #550: schema 5, its arrays exactly
        assert sorted(fx.arrays) == sorted(("density",
                                            *manifest.GRADIENT_ARRAYS))
        assert schema == manifest.GRADIENT_SCHEMA
    elif fx.density_free:  # #520: no density, by its recipe
        assert "density" not in fx.arrays
        assert schema == manifest.AO_SCHEMA
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
                          "pathlib", "typing", "hashlib"}
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
        (lambda r: r.update(schema=6), "schema 6"),
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
            manifest.THREE_CENTER_SCHEMA, manifest.GRADIENT_SCHEMA,
            manifest.SCHEMA) == (2, 3, 4, 5, 5)
    assert manifest.SCHEMAS == (2, 3, 4, 5)
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
    """The five HF/cc-pVTZ ERI files, the three `_schwarz` files and the
    three `_dipole` files (#517) are the declared exceptions
    (`Case.record_one_e`), and a twin -- same molecule, basis, Cartesian
    flag (`hf_ccpvtz`, the `_j` cases) -- carries the four for them."""
    from oracle import generate

    fixtures = [load(p) for p in manifest.tracked()]

    def key(fx):
        p = fx.provenance
        return (p.mol["atom"], p.mol["unit"], p.basis, p.cart)

    carriers = {key(fx) for fx in fixtures
                if manifest.OVLP_ARRAY in fx.arrays}
    skipped = set()
    pieces = {f"co_ccpvtz_j{s}" for s in manifest.ONE_E_PIECE_SUFFIXES}
    for fx in fixtures:
        has = [n for n in manifest.ONE_E_ARRAYS if n in fx.arrays]
        if fx.gradient is not None:  # #550: exactly its own arrays
            assert not has, fx.name
            continue
        if fx.converted is None:
            assert not has, fx.name
            continue
        if fx.name in pieces:  # #570: a half each (THE ONE-ELECTRON PIECES)
            continue
        if fx.name == "co_ccpvtz_j":  # #570: its two pieces carry the four
            whole = manifest.one_e_fixture(fx.name)
            assert not has and all(n in whole.arrays
                                   for n in manifest.ONE_E_ARRAYS)
            skipped.add(fx.name)
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
                       "h2o_sto3g_x_631g", "h2o_sto3g_x_ccpvdz",
                       "h2o_sto3g_dipole", "h2o_631g_dipole",
                       "h2o_ccpvdz_dipole", "co_ccpvtz_j", "co_ccpvtz_k"}


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


def test_schema_2_3_4_and_5_files_load_side_by_side():
    """Every AO fixture is still schema 2, unchanged; the metric pieces are
    schema 3 and the only files that are, the three-center pieces (#442)
    schema 4 and the only files that are, and the gradient fixtures (#550)
    schema 5 and the only files that are."""
    by_schema = {}
    for p in manifest.tracked():
        by_schema.setdefault(json.loads(p.read_text())["schema"], set()).add(
            p.stem)
    assert set(by_schema) == {2, 3, 4, 5}
    assert by_schema[5] == set(GRADIENT_FIXTURES)
    assert all(load(manifest.FIXTURES / f"{n}.json").metric is not None
               for n in by_schema[3])
    assert by_schema[3] == {p.stem for p in manifest.tracked()
                            if p.stem.startswith(("h2o_ccpvtz_jkfit_metric",
                                                  "prim_spd_f_auxi_metric"))}
    assert all(load(manifest.FIXTURES / f"{n}.json").three_center is not None
               for n in by_schema[4])
    assert by_schema[4] == {"h2o_sto3g_jkfit_3c_cols0_166"} | {
        f"prim_spd_f_auxi_3c_cols{c0}_{c1}" for c0, c1 in
        ((0, 35), (35, 84), (84, 119), (119, 140), (140, 168))}
    for n in by_schema[2]:
        fx = load(manifest.FIXTURES / f"{n}.json")
        assert fx.metric is None and fx.three_center is None
        assert fx.gradient is None


def test_a_schema_2_fixture_still_requires_its_density(tmp_path):
    path = _write(tmp_path, lambda r: r["arrays"].pop("density"))
    with pytest.raises(FixtureError, match=re.escape("missing ['density']")):
        load(path)


@pytest.mark.parametrize("schema", [3, 4, 5])
def test_a_schema_3_or_4_ao_fixture_still_requires_its_density(tmp_path,
                                                                schema):
    """Schemas 3 and 4 lift the density requirement for a metric or
    three-center piece only, and schema 5 lifts it for nothing."""
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


# --- #570: the one-electron pieces ------------------------------------------

PIECE_OWNER = "co_ccpvtz_j"


def test_the_one_e_pieces_reassemble_into_their_owner():
    """`one_e_fixture` merges exactly the two halves into the owner, which
    carries none of the four itself; each piece is density-free and holds
    its half and nothing else of the four; a fixture that carries the four
    comes back as loaded."""
    owner = load(manifest.FIXTURES / f"{PIECE_OWNER}.json")
    assert not any(n in owner.arrays for n in manifest.ONE_E_ARRAYS)
    whole = manifest.one_e_fixture(PIECE_OWNER)
    assert set(whole.arrays) == set(owner.arrays) | set(manifest.ONE_E_ARRAYS)
    for half, suffix in zip(manifest.ONE_E_HALVES,
                            manifest.ONE_E_PIECE_SUFFIXES, strict=True):
        piece = load(manifest.FIXTURES / f"{PIECE_OWNER}{suffix}.json")
        assert piece.density_free
        assert [n for n in manifest.ONE_E_ARRAYS if n in piece.arrays] \
            == list(half)
        for n in half:
            assert np.array_equal(whole.arrays[n], piece.arrays[n])
    assert np.array_equal(whole.density, owner.density)
    own = manifest.one_e_fixture("h2o_sto3g_j")
    plain = load(manifest.FIXTURES / "h2o_sto3g_j.json")
    assert set(own.arrays) == set(plain.arrays)
    assert all(np.array_equal(own.arrays[n], plain.arrays[n])
               for n in plain.arrays)
    assert PIECE_OWNER in manifest.one_e_fixtures()
    assert not any(n.endswith(s) for n in manifest.one_e_fixtures()
                   for s in manifest.ONE_E_PIECE_SUFFIXES)


def _pieces(tmp_path, mutate_st=None):
    """The owner and both pieces copied to `tmp_path`, `_st` mutated."""
    for suffix in ("", *manifest.ONE_E_PIECE_SUFFIXES):
        src = manifest.FIXTURES / f"{PIECE_OWNER}{suffix}.json"
        raw = json.loads(src.read_text())
        if suffix == "_st" and mutate_st is not None:
            mutate_st(raw)
        (tmp_path / src.name).write_text(json.dumps(raw))
    return tmp_path / f"{PIECE_OWNER}_st.json"


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r["arrays"].update(int1e_nuc=r["arrays"]["int1e_kin"]),
         "carries exactly ['int1e_ovlp', 'int1e_kin']"),
        (lambda r: r["arrays"].pop("int1e_kin"),
         "carries exactly ['int1e_ovlp', 'int1e_kin']"),
        (lambda r: (r["arrays"].update(density={"shape": [70, 70],
                                                "values": [0.0] * 4900}),
                    r.update(density_recipe="hilbert: x")),
         "is density-free"),
    ],
    ids=["extra-array", "half-missing", "with-density"],
)
def test_a_bad_one_e_piece_fails_naming_the_file(tmp_path, mutate, says):
    path = _pieces(tmp_path, mutate)
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value) and says in str(e.value)


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r["atoms"][0]["coord"].__setitem__(0, 0.5), "atoms"),
        (lambda r: r["provenance"].update(basis="cc-pvdz"),
         "provenance.basis"),
        (lambda r: (r["arrays"]["ovlp"]["values"].__setitem__(1, 0.5),
                    r["arrays"]["int1e_ovlp"]["values"].__setitem__(1, 0.5)),
         "arrays.ovlp"),
    ],
    ids=["moved-atom", "other-basis", "other-ovlp"],
)
def test_a_piece_of_other_inputs_is_refused_naming_both(tmp_path, mutate,
                                                        says):
    path = _pieces(tmp_path, mutate)
    with pytest.raises(FixtureError) as e:
        manifest.one_e_fixture(PIECE_OWNER, tmp_path)
    assert str(path) in str(e.value) and says in str(e.value)
    assert "not a one-electron piece" in str(e.value)


def test_a_missing_one_e_piece_is_refused(tmp_path):
    _pieces(tmp_path)
    (tmp_path / f"{PIECE_OWNER}_vh.json").unlink()
    with pytest.raises(FixtureError, match=f"{PIECE_OWNER}_vh"):
        manifest.one_e_fixture(PIECE_OWNER, tmp_path)


# --- #517: the dipole fixtures, siblings of the `_j` files ----------------

DIPOLE_MOLECULES = ("h2o_631g", "h2o_ccpvdz", "h2o_sto3g")
DIPOLE_SRC = manifest.FIXTURES / "h2o_sto3g_dipole.json"
# libcint computes <m|r - C|n> directly, not as r_0 - C S: the two agree to
# rounding of the largest |r_0| (measured 5.0e-16 of it at worst).
SHIFT_TOL = 1e-14
# dip_moment contracts PySCF's intor_symmetric copy of the same integrals.
DIP_TOL = 1e-13


def test_the_dipole_fixtures_are_the_three_h2o_j_siblings():
    assert manifest.dipole_pairs() == list(DIPOLE_MOLECULES)
    assert manifest.DIPOLE_COMPONENTS == ("x", "y", "z")
    from oracle import generate

    for name in DIPOLE_MOLECULES:
        p = manifest.dipole_pair(name)
        fd, fj = p.dipole, p.j
        assert fd.name == f"{name}_dipole" and fj.name == f"{name}_j"
        assert sorted(fd.arrays) == sorted(
            {"density", "ovlp", *manifest.DIPOLE_ARRAYS})
        assert fd.converted == fj.converted
        assert np.array_equal(fd.density, fj.density)
        n = fd.nao
        assert p.r.shape == p.r0.shape == (3, n, n)
        assert p.dip.shape == p.dip0.shape == (3,)
        assert p.origin.tolist() == list(generate.DIPOLE_ORIGIN)
        # non-zero and non-symmetric: X_PC is exercised on every axis
        assert len({abs(c) for c in p.origin}) == 3 and all(p.origin)
        # each component is symmetric in (m, n), to rounding
        for r in (p.r, p.r0):
            assert np.max(np.abs(r - r.transpose(0, 2, 1))) <= (
                SHIFT_TOL * np.max(np.abs(r)))


@pytest.mark.parametrize("name", DIPOLE_MOLECULES)
def test_the_dipole_component_order_is_x_y_z(name):
    """Pinned twice. Against the origin: `r_C[k] = r_0[k] - C_k S`, and C's
    components differ, so a permuted axis fails. Against the geometry: an
    s AO on a hydrogen has `<s|r|s> = R_H <s|s>` (with `C = 0`)."""
    p = manifest.dipole_pair(name)
    S = p.dipole.arrays["ovlp"]
    scale = np.max(np.abs(p.r0))
    shifted = p.r0 - p.origin[:, None, None] * S[None]
    assert np.max(np.abs(p.r - shifted)) <= SHIFT_TOL * scale
    swapped = p.r[[1, 0, 2]]
    assert np.max(np.abs(swapped - shifted)) > 1e3 * SHIFT_TOL * scale
    h_s = [i for i, lab in enumerate(p.dipole.ao_labels)
           if lab.split()[1] == "H" and lab.split()[2].endswith("s")]
    assert h_s
    for i in h_s:
        atom = p.dipole.atoms[int(p.dipole.ao_labels[i].split()[0])]
        for k in range(3):
            assert abs(p.r0[k, i, i] - atom.coord[k] * S[i, i]) <= (
                SHIFT_TOL * scale), (i, k)


@pytest.mark.parametrize("name", DIPOLE_MOLECULES)
def test_the_recorded_dip_moment_is_nuclear_minus_electronic(name):
    """`dip_moment` from the recorded integrals and nuclei, at both origins,
    and the two origins related by the charge left over:
    `mu_C = mu_0 - (sum Z - Tr(D S)) C`."""
    p = manifest.dipole_pair(name)
    D, S = p.density, p.dipole.arrays["ovlp"]
    Z = p.dipole.arrays[manifest.DIP_CHARGES_ARRAY]
    R = p.dipole.arrays[manifest.DIP_COORDS_ARRAY]
    for dip, ints, c in ((p.dip, p.r, p.origin), (p.dip0, p.r0, np.zeros(3))):
        nuc = Z @ (R - c[None, :])
        el = np.einsum("xij,ji->x", ints, D)
        scale = np.max(np.abs(nuc)) + np.max(np.abs(el))
        assert np.max(np.abs(dip - (nuc - el))) <= DIP_TOL * scale
    q = Z.sum() - np.trace(D @ S)
    assert abs(q) > 1.0  # the hilbert density is not neutral: C matters
    scale = np.max(np.abs(p.dip0)) + abs(q) * np.max(np.abs(p.origin))
    assert np.max(np.abs(p.dip - (p.dip0 - q * p.origin))) <= DIP_TOL * scale


def _sym_set(r, i, j, v):
    n = r["nao"]
    vals = r["arrays"]["density"]["values"]
    vals[i * n + j] = vals[j * n + i] = v


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: _sym_set(r, 0, 1, 0.25), "density differ"),
        (lambda r: r["shells"][0]["exponents"].__setitem__(0, 130.0),
         "shells differ"),
        (lambda r: r["provenance"]["mol"].update(charge=1),
         "provenance.mol differ"),
        (lambda r: r["provenance"].update(basis="6-31g"),
         "provenance.basis differ"),
        (lambda r: r["arrays"]["ovlp"]["values"].__setitem__(0, 0.5),
         "arrays.ovlp differ"),
        (lambda r: r.update(density_recipe="rhf: something else"),
         "density_recipe differ"),
    ],
    ids=["density", "shells", "mol", "basis", "ovlp", "recipe"],
)
def test_a_dipole_file_off_its_j_sibling_is_refused(tmp_path, mutate, says):
    import shutil

    shutil.copy(manifest.FIXTURES / "h2o_sto3g_j.json", tmp_path)
    raw = json.loads(DIPOLE_SRC.read_text())
    mutate(raw)
    (tmp_path / DIPOLE_SRC.name).write_text(json.dumps(raw))
    with pytest.raises(FixtureError) as e:
        manifest.dipole_pair("h2o_sto3g", tmp_path)
    msg = str(e.value)
    assert "not a dipole pair" in msg and says in msg
    assert "h2o_sto3g_j.json" in msg and DIPOLE_SRC.name in msg


def test_check_dipole_refuses_the_wrong_halves():
    p = manifest.dipole_pair("h2o_sto3g")
    k = load(manifest.FIXTURES / "h2o_sto3g_k.json")
    with pytest.raises(FixtureError, match="the first carries no vj"):
        manifest.check_dipole(k, p.dipole)
    with pytest.raises(FixtureError, match="the second carries no"):
        manifest.check_dipole(p.j, p.j)
    other = manifest.dipole_pair("h2o_631g")
    with pytest.raises(FixtureError, match="not a dipole pair: shells"):
        manifest.check_dipole(p.j, other.dipole)


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r["arrays"].pop(manifest.DIP0_ARRAY),
         "all seven or none"),
        (lambda r: r["arrays"][manifest.R_ARRAY].update(
            shape=[7, 7, 3]), "expected (3, 7, 7)"),
        (lambda r: r["arrays"][manifest.DIP_ARRAY].update(
            shape=[1, 3]), "expected (3,)"),
        (lambda r: r["arrays"][manifest.DIPOLE_ORIGIN_ARRAY].update(
            values=[0.37, 0.0, 1.23]), "must be non-zero"),
        (lambda r: r["arrays"][manifest.DIPOLE_ORIGIN_ARRAY].update(
            values=[0.5, -0.5, 1.0]), "pairwise distinct"),
        (lambda r: r["arrays"][manifest.DIP_CHARGES_ARRAY]["values"]
         .__setitem__(0, 7.0), "not the atoms' charges"),
        (lambda r: r["arrays"][manifest.DIP_COORDS_ARRAY]["values"]
         .__setitem__(3, 1.0), "not the atoms' coordinates"),
        (lambda r: r.pop("converted"), "need the converted basis"),
    ],
    ids=["six-of-seven", "r-shape", "dip-shape", "origin-zero",
         "origin-symmetric", "charges", "coords", "no-converted"],
)
def test_a_bad_dipole_array_fails_naming_the_file(tmp_path, mutate, says):
    path = _write_piece(tmp_path, mutate, src=DIPOLE_SRC)
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value) and says in str(e.value)


# --- #520: the density-free dipole fixture ---------------------------------

SPD_SRC = manifest.FIXTURES / f"{manifest.SPD_DIPOLE}.json"


def test_the_spd_dipole_fixture_loads_and_holds_nothing_above_d():
    from oracle import generate

    d = manifest.dipole_integrals()
    fx = d.fixture
    assert fx.name == "hf_ccpvtz_spd_dipole" and fx.density_free
    assert sorted(fx.arrays) == sorted(
        {"ovlp", *manifest.ONE_E_ARRAYS, *manifest.DIPOLE_INT_ARRAYS})
    hf = load(manifest.FIXTURES / "hf_ccpvtz.json")
    # hf_ccpvtz's molecule, its basis with the f shells gone: F 4s3p2d,
    # H 3s2p1d, 40 Cartesian AOs
    assert fx.atoms == hf.atoms and fx.provenance.mol == hf.provenance.mol
    assert fx.shells == tuple(s for s in hf.shells if s.l <= 2)
    assert max(s.l for s in hf.shells) == 3
    assert fx.nao == 40
    assert max(s.l for s in fx.converted.shells) == manifest.DIPOLE_SPD_L_MAX
    assert {fx.converted.shells[i].atom for i, s in
            enumerate(fx.converted.shells) if s.l == 2} == {0, 1}
    assert fx.provenance.basis == "cc-pvtz, shells with l <= 2 only"
    assert d.origin.tolist() == list(generate.DIPOLE_ORIGIN)
    assert d.r.shape == d.r0.shape == (3, 40, 40)
    assert np.array_equal(d.ovlp, fx.arrays["ovlp"])
    # the component order, against the origin
    shifted = d.r0 - d.origin[:, None, None] * d.ovlp[None]
    scale = np.max(np.abs(d.r0))
    assert np.max(np.abs(d.r - shifted)) <= SHIFT_TOL * scale
    assert np.max(np.abs(d.r[[1, 0, 2]] - shifted)) > 1e3 * SHIFT_TOL * scale
    assert SPD_SRC.stat().st_size < 512 * 1024


def test_a_density_free_fixture_has_no_density_to_give():
    fx = manifest.dipole_integrals().fixture
    with pytest.raises(FixtureError, match="records no density"):
        fx.density  # noqa: B018


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r["arrays"].update(
            density={"shape": [40, 40], "values": [0.0] * 1600}),
         "says no density, and arrays carries ['density']"),
        (lambda r: r["arrays"].update(
            {manifest.DIP_ARRAY: {"shape": [3], "values": [1.0, 2.0, 3.0]}}),
         "says no density, and arrays carries"),
        (lambda r: r["arrays"].update(
            vj={"shape": [40, 40], "values": [0.0] * 1600}),
         "says no density, and arrays carries ['vj']"),
        (lambda r: r["arrays"].pop(manifest.R0_ARRAY),
         "a density-free fixture: exactly"),
        (lambda r: r["arrays"].update(
            {manifest.DIP_CHARGES_ARRAY: {"shape": [2],
                                          "values": [9.0, 1.0]}}),
         "a density-free fixture: exactly"),
        (lambda r: r.update(density_recipe="hilbert: something"),
         "arrays is missing ['density']"),
    ],
    ids=["density", "dip-moment", "vj", "two-of-three", "charges",
         "recipe"],
)
def test_a_bad_density_free_fixture_fails_naming_the_file(tmp_path, mutate,
                                                          says):
    path = _write_piece(tmp_path, mutate, src=SPD_SRC)
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value) and says in str(e.value)


def test_dipole_integrals_refuses_a_shell_above_d(tmp_path):
    raw = json.loads(SPD_SRC.read_text())
    for c in raw["converted"]["shells"]:
        if c["l"] == 2:
            c["l"] = 3  # a converted f shell, with a d's AOs: refused first
            break
    (tmp_path / SPD_SRC.name).write_text(json.dumps(raw))
    with pytest.raises(FixtureError):
        manifest.dipole_integrals(fixtures=tmp_path)
    # the l check itself, on a fixture that loads: hf_ccpvtz has f shells,
    # so it is refused by the l cap once it otherwise qualifies
    hf = load(manifest.FIXTURES / "hf_ccpvtz.json")
    spd = manifest.dipole_integrals().fixture
    from dataclasses import replace

    with_f = replace(spd, shells=hf.shells)
    with pytest.raises(FixtureError, match="l = 3, above DIPOLE_SPD_L_MAX"):
        manifest.check_dipole_integrals(with_f)
    conv_f = replace(spd, converted=hf.converted)
    with pytest.raises(FixtureError, match="l = 3, above DIPOLE_SPD_L_MAX"):
        manifest.check_dipole_integrals(conv_f)


def test_dipole_integrals_refuses_the_wrong_kind_of_fixture():
    with pytest.raises(FixtureError, match="it records a density"):
        manifest.check_dipole_integrals(
            manifest.dipole_pair("h2o_sto3g").dipole)
    with pytest.raises(FixtureError, match="metric or three-center"):
        manifest.check_dipole_integrals(load(PIECE))
    spd = manifest.dipole_integrals().fixture
    from dataclasses import replace

    bare = replace(spd, arrays={k: v for k, v in spd.arrays.items()
                                if k != manifest.OVLP_ARRAY})
    with pytest.raises(FixtureError, match="carries no \\['int1e_ovlp'\\]"):
        manifest.check_dipole_integrals(bare)
    with pytest.raises(FixtureError, match="converted basis"):
        manifest.check_dipole_integrals(replace(spd, converted=None))


# --- #550: schema 5 and the gradient fixtures --------------------------------

# H2O/cc-pVDZ (d shells: J/K classes to L = 8) is read by `slow` tests only.
GRADIENT_FIXTURES = {"h2o_sto3g_grad": "h2o_sto3g_j",
                     "h2o_631g_grad": "h2o_631g_j",
                     "h2o_ccpvdz_grad": "h2o_ccpvdz_j",
                     "nh2f_631g_grad": None}
GRAD_SLOW = {"h2o_ccpvdz_grad"}
GRAD = manifest.FIXTURES / "h2o_sto3g_grad.json"


def _grad_params():
    return [pytest.param(n, marks=(pytest.mark.slow,) if n in GRAD_SLOW
                         else ()) for n in sorted(GRADIENT_FIXTURES)]


def test_the_gradient_fixtures_are_tracked_with_their_siblings():
    from oracle import generate

    assert manifest.gradient_fixtures() == sorted(GRADIENT_FIXTURES)
    for name, sib in GRADIENT_FIXTURES.items():
        fx = load(manifest.FIXTURES / f"{name}.json")
        assert fx.gradient == manifest.GradientRecord(sib)
        case = generate.CASES[name]
        assert case.record_gradient and case.gradient_sibling == sib
        assert fx.density_recipe == generate.RHF_DENSITY_RECIPE
        # the PySCF details that each bit once (#550): Cartesian AOs, and
        # libcint's common_fac 1/sqrt(4 pi), sqrt(3 / (4 pi)), then 1
        assert fx.provenance.cart is True
        cf = fx.converted.common_factor
        assert np.allclose(cf[:2], (1 / np.sqrt(4 * np.pi),
                                    np.sqrt(3 / (4 * np.pi))), rtol=1e-15)
        assert all(c == 1.0 for c in cf[2:])


@pytest.mark.parametrize("name", _grad_params())
def test_a_gradient_fixture_recomputes_from_its_stored_arrays(name):
    """THE assertions, each naming the claim it carries.

    - each per-atom term, `density` and `W` recomputed from the stored
      integrals, orbitals and `vj1` / `vk1` by the docstring's sign
      conventions: carries the SIGNS and the MULTIPLICITY FACTORS (2 on J,
      -1 on K, 2 on the Pulay term) -- and nothing else here does;
    - the terms sum to PySCF's own total: carries completeness (no term
      missing or counted twice);
    - the total sums to zero over atoms: STRUCTURAL only. A wrong factor on
      a term scales a sum that vanishes per quartet, and it still vanishes,
      so a green net force is never the factor's warrant."""
    fx = manifest.gradient_fixture(name)  # all of the below, and the sibling
    tol = manifest.GRADIENT_TOL
    got = manifest.gradient_recompute(fx)
    for term in (*manifest.GRADIENT_TERMS, "density", "dme"):
        assert np.max(np.abs(got[term] - fx.arrays[term])) <= tol, term
    total = fx.arrays[manifest.GRAD_TOTAL]
    terms = sum(fx.arrays[n] for n in manifest.GRADIENT_TERMS)
    assert np.max(np.abs(terms - total)) <= tol
    assert np.max(np.abs(total.sum(axis=0))) <= tol
    # a real force on every atom, so a sign error is visible
    assert np.min(np.abs(total)) > 1e-3


def test_the_net_force_cannot_see_a_wrong_factor():
    """The guardrail, demonstrated: doubling `vj1` (a J factor of 4, not 2)
    changes `grad:j` and leaves the net J force zero -- only the per-term
    recompute against PySCF's `grad:j` catches it."""
    fx = manifest.gradient_fixture("h2o_sto3g_grad")
    from dataclasses import replace

    bad = replace(fx, arrays={**fx.arrays, "vj1": 2 * fx.arrays["vj1"]})
    j = manifest.gradient_recompute(bad)["grad:j"]
    assert np.max(np.abs(j - fx.arrays["grad:j"])) > 1e-2
    assert np.max(np.abs(j.sum(axis=0))) <= manifest.GRADIENT_TOL
    with pytest.raises(FixtureError, match="arrays.grad:j differs"):
        manifest.check_gradient(bad, load(manifest.FIXTURES /
                                          "h2o_sto3g_j.json"))


def test_the_low_symmetry_molecule_has_no_zero_component():
    """NH2F, C1: no atom on an axis and no plane of symmetry, so every
    Cartesian component of every term on every atom is non-zero."""
    fx = manifest.gradient_fixture("nh2f_631g_grad")
    assert [a.symbol for a in fx.atoms] == ["N", "F", "H", "H"]
    assert {s.l for s in fx.shells} == {0, 1}
    for n in (*manifest.GRADIENT_TERMS, manifest.GRAD_TOTAL):
        assert np.min(np.abs(fx.arrays[n])) > 1e-4, n


def _grad_copy(tmp_path, mutate=None, mutate_j=None):
    """h2o_sto3g_grad and its sibling in `tmp_path`, each mutated."""
    out = []
    for src, m in ((GRAD, mutate),
                   (manifest.FIXTURES / "h2o_sto3g_j.json", mutate_j)):
        raw = json.loads(src.read_text())
        if m is not None:
            m(raw)
        (tmp_path / src.name).write_text(json.dumps(raw))
        out.append(tmp_path / src.name)
    return out


def _bump(r, name, k=0, by=1e-9):
    r["arrays"][name]["values"][k] += by


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: r["arrays"].pop("vk1"), "missing arrays ['vk1']"),
        (lambda r: r["arrays"].pop("grad:total"),
         "missing arrays ['grad:total']"),
        (lambda r: r["arrays"].pop("int1e_iprinv"),
         "missing arrays ['int1e_iprinv']"),
        (lambda r: r["arrays"]["int1e_iprinv"].update(
            shape=[1, 3, 7, 21]), "arrays.int1e_iprinv has shape (1, 3, 7, 21)"),
        (lambda r: r["arrays"].update(ovlp=r["arrays"]["dme"]),
         "carries no other arrays, not ['ovlp']"),
        (lambda r: r.update(schema=4), "a gradient fixture is schema 5"),
        (lambda r: r.update(gradient={}), "exactly the keys ['sibling']"),
        (lambda r: r["gradient"].update(sibling="h2o_sto3g_k"),
         "sibling must be a `_j`"),
        (lambda r: r.pop("converted"), "the converted basis"),
    ],
    ids=["no-vk1", "no-total", "no-iprinv", "iprinv-shape", "extra-array",
         "schema-4", "bad-record", "k-sibling", "no-converted"],
)
def test_a_bad_gradient_fixture_fails_naming_the_file(tmp_path, mutate, says):
    path, _ = _grad_copy(tmp_path, mutate)
    with pytest.raises(FixtureError) as e:
        load(path)
    assert str(path) in str(e.value) and says in str(e.value)


def test_gradient_arrays_outside_a_gradient_fixture_are_refused(tmp_path):
    grad = json.loads(GRAD.read_text())
    path = _write(tmp_path, lambda r: r["arrays"].update(
        {"vj1": grad["arrays"]["vj1"]}))
    with pytest.raises(FixtureError, match="belong to a gradient fixture"):
        load(path)


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (lambda r: _bump(r, "grad:h"), "arrays.grad:h differs"),
        (lambda r: _bump(r, "grad:nuc", 4), "arrays.grad:nuc differs"),
        # a stored W off its orbitals: the Pulay term built on it is first
        (lambda r: _bump(r, "dme", 3), "arrays.grad:s differs"),
        (lambda r: _bump(r, "mo_coeff", 2), "arrays.density differs"),
        (lambda r: _bump(r, "grad:total"), "sum to grad:total only"),
    ],
    ids=["term-h", "term-nuc", "dme", "orbitals", "total"],
)
def test_a_gradient_fixture_off_its_own_arrays_is_refused(tmp_path, mutate,
                                                          says):
    _grad_copy(tmp_path, mutate)
    with pytest.raises(FixtureError) as e:
        manifest.gradient_fixture("h2o_sto3g_grad", tmp_path)
    assert says in str(e.value)


def _move_atom(r, by=1e-9):
    r["atoms"][1]["coord"][0] += by


@pytest.mark.parametrize(
    ("mutate", "mutate_j", "says"),
    [
        (None, lambda r: r["converted"]["shells"][0]["coefficients"]
         .__setitem__(0, 1.0), "converted.shells differ"),
        (None, lambda r: r["provenance"].update(basis="6-31g"),
         "provenance.basis differ"),
        (None, _move_atom, "atoms differ"),
        (None, lambda r: r["provenance"]["mol"].update(charge=1),
         "provenance.mol differ"),
    ],
    ids=["converted", "basis", "atoms", "mol"],
)
def test_a_gradient_fixture_off_its_sibling_is_refused_naming_both(
        tmp_path, mutate, mutate_j, says):
    path, jpath = _grad_copy(tmp_path, mutate, mutate_j)
    with pytest.raises(FixtureError) as e:
        manifest.gradient_fixture("h2o_sto3g_grad", tmp_path)
    assert str(path) in str(e.value) and str(jpath) in str(e.value)
    assert says in str(e.value)


def test_a_gradient_fixture_without_its_sibling_is_refused(tmp_path):
    path, jpath = _grad_copy(tmp_path)
    jpath.unlink()
    with pytest.raises(FixtureError, match="no fixture h2o_sto3g_j.json"):
        manifest.gradient_fixture("h2o_sto3g_grad", tmp_path)
    fx = load(path)
    with pytest.raises(FixtureError, match="the sibling given is None"):
        manifest.check_gradient(fx)
    other = load(manifest.FIXTURES / "h2o_631g_j.json")
    with pytest.raises(FixtureError, match="not 'h2o_631g_j'"):
        manifest.check_gradient(fx, other)
    with pytest.raises(FixtureError, match="not a gradient fixture"):
        manifest.gradient_fixture("h2o_sto3g_j")


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


@pytest.mark.oracle
def test_the_recorded_gradient_is_the_energy_s_slope():
    """#550, a witness independent of PySCF's gradient code: the recorded
    H2O/STO-3G `grad:total` against a central difference of converged RHF
    energies, one Cartesian component per atom. Carries the overall sign
    and the factors end to end (the net force cannot)."""
    pytest.importorskip("pyscf", reason="oracle group not installed: "
                        "uv sync --group oracle")
    from oracle import generate
    from pyscf import gto, scf

    fx = manifest.gradient_fixture("h2o_sto3g_grad")
    case = generate.CASES["h2o_sto3g_grad"]
    coords = np.array([a.coord for a in fx.atoms])
    h = 1e-4

    def energy(xyz):
        mol = gto.M(atom=[(a.symbol, tuple(c)) for a, c in
                          zip(fx.atoms, xyz, strict=True)],
                    unit="Bohr", basis=case.basis, cart=True, verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        return mf.kernel()

    for atom, x in ((0, 2), (1, 0), (2, 1)):
        plus, minus = coords.copy(), coords.copy()
        plus[atom, x] += h
        minus[atom, x] -= h
        fd = (energy(plus) - energy(minus)) / (2 * h)
        got = fx.arrays[manifest.GRAD_TOTAL][atom, x]
        assert abs(fd - got) <= 1e-6, (atom, x, fd, got)
