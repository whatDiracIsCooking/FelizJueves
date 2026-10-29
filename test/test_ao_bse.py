"""`ao.bse` and `data/basis/`, `data/aux_basis/` (#650): FreeBreakfast's basis
files, read here, agree shell for shell with the basis the libcint fixtures
were written over.

The fixtures were generated with PySCF's OWN library (`basis="sto-3g"` and so
on, `test/oracle/generate.py`), not with these files, so this is a comparison
between two copies of each set. Each fixture records both PySCF's file data
(`shells`) and our converted shells (`converted`), and both are checked:

- STRUCTURE is exact: per atom, the same contracted functions in libcint's
  order, each with the same angular momentum and number of primitives.
- STO-3G and 6-31G VALUES agree to PySCF's rounding. PySCF's tables and BSE's
  carry these Gaussian/GAMESS numbers to different digits: worst seen
  9.3e-7 relative on an exponent and 5.2e-7 absolute on a coefficient (F and
  N in 6-31G, `nh2f_631g_grad`), so the bounds are 2e-6 and 1e-6. Not a
  rounding of one to the other's digits -- tested, and it is not -- so the
  bound is a measured one, stated.
- cc-pVDZ and cc-pVTZ agree EXACTLY, but only after
  `oracle.basis.free_primitive_reduced`: BSE's general contractions carry a
  component on each exponent that also stands alone, and PySCF's do not.
  Same span, different AOs; `test_bse_cc_sets_are_not_pyscfs_as_read` pins
  that the difference is real, so nobody "fixes" the reduction away.
- the converted shells, `oracle.basis.from_bse` against the fixture's
  `converted` record: 1e-14 relative where the file data agree exactly, and
  where they agree only to rounding, 2e-5 of the shell's largest coefficient
  (the normalisation turns 5e-7 absolute into at most 1.25e-5 of it).
- cc-pVTZ-JKFIT, the DF fixtures' aux basis, agrees to 1e-14 with no
  reduction: FreeBreakfast took it from PySCF's own table.

Everything here reads JSON: no compiler, no card, no PySCF. `from_bse` was
held to `from_mole` over a live PySCF Mole built from these same files, for
seven orbital sets on H2O, at 6e-16 relative (by hand, 2026-10-29; the
`oracle`-marked test below repeats it).
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from oracle import basis as ob
from oracle import manifest

from felizjueves.ao import bse

EXP_REL = 2e-6
COEF_ABS = 1e-6
CONVERTED_ROUNDED = 2e-5
CONVERTED_EXACT = 1e-14

# (fixture, BSE name, the file data agree exactly)
ORBITAL_CASES = (
    ("h2_sto3g", "STO-3G", False),
    ("h2o_sto3g_j", "STO-3G", False),
    ("h2o_631g_j", "6-31G", False),
    ("nh2f_631g_grad", "6-31G", False),
    ("h2o_ccpvdz_j", "cc-pVDZ", True),
    ("co_ccpvtz_j", "cc-pVTZ", True),
)
AUX_CASES = ("h2o_ccpvtz_jkfit_metric", "h2o_sto3g_jkfit_3c")

Function = tuple[int, tuple[float, ...], tuple[float, ...]]


def _fixture(name: str) -> manifest.Fixture:
    return manifest.load(manifest.FIXTURES / f"{name}.json")


def _atoms(fx: manifest.Fixture) -> list[tuple[int, tuple[float, float, float]]]:
    return [(a.charge, a.coord) for a in fx.atoms]


def _fixture_functions(fx: manifest.Fixture, atom: int) -> list[Function]:
    """PySCF's file data on `atom`, one entry per contracted function."""
    out = []
    for s in fx.shells:
        if s.atom != atom:
            continue
        for col in zip(*s.coefficients, strict=True):
            keep = [(a, x) for a, x in zip(s.exponents, col, strict=True) if x]
            out.append((s.l, tuple(a for a, _ in keep), tuple(x for _, x in keep)))
    return out


def _bse_functions(cs: tuple[bse.Contraction, ...]) -> list[Function]:
    return [(c.l, c.exponents, row)
            for c in ob.free_primitive_reduced(cs) for row in c.coefficients]


# --- the files ------------------------------------------------------------


def test_the_imported_sets_are_all_there() -> None:
    assert {p.name for p in bse.files(bse.ORBITAL)} == {
        "sto-3g.json", "6-31g.json", "6-31gstar.json", "6-31gstarstar.json",
        "def2-SVP.json", "def2-TZVP.json", "cc-pVDZ.json", "cc-pVTZ.json",
        "aug-cc-pVDZ.json", "aug-cc-pVTZ.json", "minao.json"}
    assert {p.name for p in bse.files(bse.AUXILIARY)} == {
        "cc-pVDZ-JKFIT.json", "cc-pVTZ-JKFIT.json", "def2-universal-JKFIT.json",
        "cc-pVDZ-acCD.json", "cc-pVDZ-acCD-6-6.json", "cc-pVTZ-acCD-6-6.json",
        "aug-cc-pVDZ-acCD-6-6.json", "aug-cc-pVTZ-acCD-6-6.json",
        "def2-SVP-acCD-6-6.json", "def2-TZVP-acCD-6-6.json",
        "6-31gstarstar-acCD-6-6.json"}
    for d in (bse.ORBITAL, bse.AUXILIARY):
        assert (d / "README.md").is_file()


@pytest.mark.parametrize("directory", [bse.ORBITAL, bse.AUXILIARY],
                         ids=["basis", "aux_basis"])
def test_every_file_is_freebreakfasts_bytes(directory) -> None:
    """`provenance.json` names the FreeBreakfast commit the files were copied
    from and each file's sha256 there; an edit here, or a refresh that did
    not update the record, fails."""
    rec = json.loads((directory / bse.PROVENANCE).read_text())
    assert len(rec["commit"]) == 40 and rec["path"] == f"data/{directory.name}/"
    assert set(rec["files"]) == {p.name for p in bse.files(directory)}
    for p in bse.files(directory):
        entry = rec["files"][p.name]
        data = p.read_bytes()
        assert (len(data), hashlib.sha256(data).hexdigest()) == (
            entry["bytes"], entry["sha256"]), p.name
        assert bse.load(p).name == entry["name"]


@pytest.mark.parametrize("directory, lmax", [(bse.ORBITAL, 4), (bse.AUXILIARY, 6)],
                         ids=["basis", "aux_basis"])
def test_every_file_parses_within_its_ceiling(directory, lmax) -> None:
    """Orbital sets stop at g (aug-cc-pVTZ), fitting sets at i (the DF
    kernels' aux ceiling, FreeBreakfast's `kMetricMaxMomentum`)."""
    for p in bse.files(directory):
        b = bse.load(p)
        assert b.elements, p.name
        top = max(c.l for cs in b.elements.values() for c in cs)
        assert top <= lmax, (p.name, top)
        for z, cs in b.elements.items():
            for c in cs:
                assert c.function_type in bse.FUNCTION_TYPES, (p.name, z)
                assert all(len(r) == len(c.exponents) for r in c.coefficients)


def test_find_matches_the_document_name_not_the_file_name() -> None:
    assert bse.find("6-31G*").name == "6-31gstar.json"
    assert bse.find("6-31g**").name == "6-31gstarstar.json"
    assert bse.find("CC_PVDZ") == bse.find("cc-pVDZ") == bse.find("ccpvdz")
    assert bse.find("ccpvdzjkfit", bse.AUXILIARY).name == "cc-pVDZ-JKFIT.json"
    with pytest.raises(KeyError, match="cc-pVDZ-JKFIT"):
        bse.find("cc-pVDZ-JKFIT")  # a fitting set is not an orbital set
    with pytest.raises(KeyError, match="Z = 19"):
        bse.named("cc-pVDZ").contractions(19)  # BSE has no cc-pVDZ K


def test_a_fused_shell_is_split_and_a_general_one_is_not() -> None:
    o = bse.named("6-31G").contractions(8)
    assert [(c.l, len(c.exponents), c.n_contracted) for c in o] == [
        (0, 6, 1), (0, 3, 1), (1, 3, 1), (0, 1, 1), (1, 1, 1)]
    assert o[1].exponents == o[2].exponents
    o = bse.named("cc-pVDZ").contractions(8)
    assert [(c.l, len(c.exponents), c.n_contracted) for c in o] == [
        (0, 9, 3), (1, 4, 2), (2, 1, 1)]
    assert o[0].coefficients[2] == (0.0,) * 8 + (1.0,)


def test_the_reader_refuses_what_it_cannot_read(tmp_path) -> None:
    doc = {"molssi_bse_schema": {}, "name": "x", "elements": {"1": {
        "electron_shells": [{"function_type": "gto", "angular_momentum": [0, 1],
                             "exponents": ["1.0"], "coefficients": [["1.0"]]}]}}}
    with pytest.raises(ValueError, match="angular momenta"):
        bse.parse(doc, tmp_path / "x.json")
    doc["elements"]["1"]["electron_shells"][0]["function_type"] = "sto"
    with pytest.raises(ValueError, match="function_type"):
        bse.parse(doc, tmp_path / "x.json")
    p = tmp_path / "y.json"
    p.write_text(json.dumps({"name": "y"}))
    with pytest.raises(ValueError, match="not a BSE"):
        bse.load(p)


# --- against the libcint fixtures -----------------------------------------


@pytest.mark.parametrize("fixture, name, exact", ORBITAL_CASES,
                         ids=[c[0] for c in ORBITAL_CASES])
def test_bse_agrees_with_the_fixtures_shell_for_shell(fixture, name, exact) -> None:
    fx = _fixture(fixture)
    assert bse.canonical(fx.provenance.basis) == bse.canonical(name)
    basis = bse.named(name)
    for i, atom in enumerate(fx.atoms):
        want = _fixture_functions(fx, i)
        got = _bse_functions(basis.contractions(atom.charge))
        assert [(f[0], len(f[1])) for f in got] == [(f[0], len(f[1])) for f in want], (
            fixture, atom.symbol)
        for (_, ea, ca), (_, eb, cb) in zip(got, want, strict=True):
            if exact:
                assert ea == eb and ca == cb, (fixture, atom.symbol)
                continue
            for a, b in zip(ea, eb, strict=True):
                assert math.isclose(a, b, rel_tol=EXP_REL), (fixture, atom.symbol, a, b)
            for a, b in zip(ca, cb, strict=True):
                assert abs(a - b) <= COEF_ABS, (fixture, atom.symbol, a, b)


@pytest.mark.parametrize("fixture, name, exact",
                         [c for c in ORBITAL_CASES if c[0] != "h2_sto3g"],
                         ids=[c[0] for c in ORBITAL_CASES if c[0] != "h2_sto3g"])
def test_from_bse_agrees_with_the_converted_shells(fixture, name, exact) -> None:
    fx = _fixture(fixture)
    assert fx.converted is not None
    got = ob.from_bse(bse.named(name), _atoms(fx), reduce=True)
    want = fx.converted.shells
    assert len(got.shells) == len(want)
    for a, b in zip(got.shells, want, strict=True):
        assert (a.l, a.centre, len(a.exponents)) == (b.l, b.centre, len(b.exponents))
        top = max(abs(x) for x in b.coefficients)
        for x, y in zip(a.coefficients, b.coefficients, strict=True):
            if exact:
                assert abs(x - y) <= CONVERTED_EXACT * abs(y), (fixture, x, y)
            else:
                assert abs(x - y) <= CONVERTED_ROUNDED * top, (fixture, x, y)


def _columns(cs, lq: int, exps: list[float]) -> np.ndarray:
    """Every contracted function at `lq`, a row over the shared `exps`."""
    return np.array([[dict(zip(c.exponents, r, strict=True)).get(a, 0.0)
                      for a in exps]
                     for c in cs if c.l == lq for r in c.coefficients])


def test_bse_cc_sets_are_not_pyscfs_as_read() -> None:
    """Without the reduction, cc-pVDZ's O s is three functions over nine
    primitives in BSE and two over eight plus one in PySCF -- but the AO
    count and the span are the same."""
    fx = _fixture("h2o_ccpvdz_j")
    raw = ob.from_bse(bse.named("cc-pVDZ"), _atoms(fx))
    reduced = ob.from_bse(bse.named("cc-pVDZ"), _atoms(fx), reduce=True)
    assert raw.n_ao == reduced.n_ao == fx.nao
    assert len(raw.shells[0].exponents) == 9 != len(fx.converted.shells[0].exponents)
    for z in (8, 1):
        cs = bse.named("cc-pVDZ").contractions(z)
        for lq in {c.l for c in cs}:
            exps = sorted({a for c in cs if c.l == lq for a in c.exponents})
            a = _columns(cs, lq, exps)
            b = _columns(ob.free_primitive_reduced(cs), lq, exps)
            assert a.shape == b.shape
            assert np.linalg.matrix_rank(np.vstack([a, b])) == a.shape[0], (z, lq)


@pytest.mark.parametrize("set_name", AUX_CASES)
def test_jkfit_agrees_with_the_df_fixtures_aux_basis(set_name) -> None:
    rec = (manifest.metric_set(set_name) if set_name.endswith("_metric")
           else manifest.three_center_set(set_name))
    fx = rec.pieces[0]
    assert bse.canonical(fx.provenance.basis) == bse.canonical("cc-pVTZ-JKFIT")
    got = ob.from_bse(bse.named("cc-pVTZ-JKFIT", bse.AUXILIARY), _atoms(fx))
    want = rec.converted.shells
    assert got.n_ao == fx.nao and len(got.shells) == len(want)
    for a, b in zip(got.shells, want, strict=True):
        assert (a.l, a.centre, a.exponents) == (b.l, b.centre, b.exponents)
        for x, y in zip(a.coefficients, b.coefficients, strict=True):
            assert abs(x - y) <= CONVERTED_EXACT * abs(y)


# --- against a live PySCF -------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["STO-3G", "6-31G", "6-31G**", "cc-pVDZ",
                                  "cc-pVTZ", "def2-SVP", "aug-cc-pVDZ"])
def test_from_bse_is_from_mole_over_the_same_file(name) -> None:
    gto = pytest.importorskip("pyscf.gto")
    basis = bse.named(name)

    def pyscf_format(z: int) -> list:
        return [[c.l, *([a, *col] for a, col in
                        zip(c.exponents, zip(*c.coefficients, strict=True),
                            strict=True))]
                for c in basis.contractions(z)]

    fx = _fixture("h2o_ccpvdz_j")
    mol = gto.M(atom=fx.provenance.mol["atom"], unit="Bohr", cart=True, verbose=0,
                basis={"O": pyscf_format(8), "H": pyscf_format(1)})
    want = ob.from_mole(mol)
    got = ob.from_bse(basis, _atoms(fx))
    assert len(got.shells) == len(want.shells)
    for a, b in zip(got.shells, want.shells, strict=True):
        assert (a.l, a.centre, a.exponents) == (b.l, b.centre, b.exponents)
        for x, y in zip(a.coefficients, b.coefficients, strict=True):
            assert abs(x - y) <= CONVERTED_EXACT * abs(y)
