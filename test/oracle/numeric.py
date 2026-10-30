"""The numerical golden store (#655): its recipe, schema, loader and the one
comparison helper -- NUMPY ONLY.

    entries()                    -> every entry's name, in recipe order
    want(name)                   -> the golden array (read-only float64)
    compare(name, got, ...)      -> assert `got` is the golden; the error/scale
    atoms(name), basis(name),    -> an npy entry's inputs, PySCF-free, for a
    density(name)                   test that stages them itself

WHAT IT IS. `test/golden/numeric/` holds libcint/PySCF arrays, one plain
float64 `.npy` per array, for a curated set of molecule x basis x quantity
cases, each keyed by the registry case (`registry.Key`) its whole build
reaches -- the TOP class, `(engine, quantity, deriv, L)`, of every class the
molecule's build runs. `manifest.json` beside them records how each was
produced. It is NOT the text goldens (`test/golden/{cuda,hip,gpu}/` and the
`*_digests.json`), which hold what the generator EMITS and stay a
compile-free check; nothing here is generated kernel text.

TWO KINDS OF ENTRY, so nothing is held twice:

- `npy`: an array in the store, computed by `oracle.generate --numeric`
  (the only module that imports PySCF) over a BASIS FILE FROM `data/basis/`
  or `data/aux_basis/` (#650) -- read as the file states it, in libcint's
  shell order and with NO free-primitive reduction (`oracle.basis.from_bse`,
  `reduce=False`), which is the form FreeBreakfast reads -- on the molecule,
  and with the density, of a tracked `test/oracle/fixtures/` case. The
  molecule and the density are REFERENCED, never copied: the manifest names
  the fixture and the density's sha256 (`manifest.density_sha256`), and the
  fast test holds both to the fixture as it is now.
- `fixture`: an array that `test/oracle/fixtures/` already holds (a `vj`, a
  metric set), indexed here under its registry case and loaded from there.
  Its regeneration stays `generate.py`'s and its own `oracle` test's; the
  store adds only the index, not a second copy of 11 MB of JSON.

THE RECIPE is `ENTRIES`, below: what each entry is, in code. The manifest is
what the generator WROTE from it: the recipe's fields plus the array's shape,
dtype, sha256 and provenance (versions, date, host, command) and the basis
file's sha256 as `provenance.json` records it. `check_manifest` holds the
three -- recipe, manifest, files -- to one another; the fast test calls it.

REGENERATION. `PYTHONPATH=test uv run --group oracle python -m
oracle.generate --numeric --regenerate [NAME ...]` rewrites the arrays and the
manifest; `--check` (and the `oracle`-marked test) rebuilds each array in
memory and requires it BIT FOR BIT equal to the tracked file. A drift is
news about PySCF until proven otherwise, as for every fixture.

THE COMPARISON. `compare` is `oracle.tolerance.assert_within` against the
golden: `max |got - want| <= REL[precision] * scale`, `scale` the golden's
largest `|want|` unless the caller states a tighter one, over the whole
array or over `index` (`np.ix_` rows and columns, for a test that builds a
sub-basis). A float32 result is judged at the fp32 tier, never the fp64 one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from felizjueves.ao import bse
from felizjueves.ao.shell import Basis
from felizjueves.ir import Scalar

from . import manifest as fixtures
from .tolerance import assert_within

STORE = Path(__file__).resolve().parents[1] / "golden" / "numeric"
MANIFEST = STORE / "manifest.json"
SCHEMA = 1

COMMAND = ("PYTHONPATH=test uv run --group oracle python -m oracle.generate "
           "--numeric --regenerate")

# What an npy entry's array is, by quantity; the generator dispatches on it.
QUANTITIES = {
    "j": ("PySCF's scf.hf.get_jk Coulomb matrix (hermi=1, no vhfopt) for "
          "the referenced density, J_mn = sum_ls (mn|ls) D_ls, nao x nao, "
          "cross-checked against einsum(int2e, D) before it is written"),
    "metric": ("libcint's two-center Coulomb metric (P|Q), int2c2e_cart, "
               "naux x naux, checked equal to int2c2e (cart=True) before it "
               "is written"),
    # #693: the uncontracted d1 metric, one axis per entry (the whole
    # [3, naux, naux] is over FILE_BUDGET at cc-pVDZ-JKFIT's 131 functions).
    **{f"metric.d1.{x}": (
        f"the uncontracted metric derivative on axis {x}, M1[{k}, P, Q] = "
        f"d(P|Q)/dA_{x} with A the centre of P: MINUS libcint's "
        f"int2c2e_ip1_cart[{k}] (which differentiates the electron "
        "coordinate), naux x naux, checked equal to int2c2e_ip1 (cart=True) "
        "and antisymmetric to 1e-13 of its max before it is written")
       for k, x in enumerate("xyz")},
    # #688: the Coulomb derivative matrix, its ip1 bra half and the whole.
    "jx1": ("PySCF's grad.rhf.get_jk vj for the referenced density, "
            "J1[k, m, n] = sum_ls (d/dA_k m n|ls) D_ls with A the centre of m "
            "(-int2e_ip1 contracted ijkl,lk->ij), 3 x nao x nao, "
            "cross-checked against einsum(int2e_ip1_cart, D) before it is "
            "written"),
    "jx": ("PySCF's hessian.rhf h1ao Coulomb part for the referenced "
           "density, J^X[X, k, m, n] = sum_ls D_ls d(mn|ls)/dR_{X,k} (vj1 + "
           "vj1^T, the ket half, plus vj2 on X's rows and its transpose, "
           "the bra half, from hessian.rhf._get_jk), natm x 3 x nao x nao, "
           "cross-checked against the same sum from the full int2e_ip1_cart "
           "tensor before it is written"),
}

# The basis file's directory, by registry quantity: an orbital set for J and
# J^X, a fitting set for the metric.
DIRECTORY = {"j": bse.ORBITAL, "metric": bse.AUXILIARY, "jx": bse.ORBITAL}

# The quantities whose array is built for the molecule fixture's density.
WITH_DENSITY = frozenset({"j", "jx1", "jx"})


class GoldenError(ValueError):
    """The store disagrees with its recipe or its files; the message says
    which entry and which field."""


@dataclass(frozen=True)
class Entry:
    """One golden array. `key` is the registry case `(engine, quantity,
    deriv, L)` its build reaches. An npy entry names a `basis` file and the
    tracked `molecule` fixture whose geometry (and, for `j`, density) it
    takes; a fixture entry names the tracked `fixture` and `array`, or a
    `metric_set`. `form` names WHICH array of the key's quantity an npy
    entry holds when there is more than one (`jx1`, the ip1 bra half of
    the `jx` registry quantity, and `jx`, the whole); None derives it from
    the key (`what`)."""

    name: str
    key: tuple[str, str, int, int]
    description: str
    basis: str | None = None  # a BSE name `bse.find` resolves
    molecule: str | None = None  # a tracked fixture's stem
    fixture: str | None = None
    array: str | None = None
    metric_set: str | None = None
    component: str | None = None  # an npy derivative entry's axis, x y z
    form: str | None = None

    @property
    def kind(self) -> str:
        return "npy" if self.basis is not None else "fixture"

    @property
    def quantity(self) -> str:
        return self.key[1]

    @property
    def what(self) -> str:
        """The `QUANTITIES` key the generator dispatches on: `form` when an
        entry names one (#688's `jx1` / `jx`, two arrays of one registry
        quantity); else the quantity, `.d<n>` at a derivative order,
        `.<axis>` for one component."""
        if self.form is not None:
            return self.form
        q = self.quantity if not self.key[2] else (
            f"{self.quantity}.d{self.key[2]}")
        return q if self.component is None else f"{q}.{self.component}"


_H2O = "the H2O geometry every H2O fixture shares"

ENTRIES: tuple[Entry, ...] = (
    # J over FreeBreakfast's orbital files, on the #86 fixtures' molecule and
    # recorded density. STO-3G and 6-31G are BSE's rounding, not PySCF's
    # library's (test_ao_bse.ORBITAL_CASES); cc-pVDZ is the file's general
    # contraction UNREDUCED, so different AOs from h2o_ccpvdz_j's -- each a
    # different array from the fixture's `vj`, none a copy.
    Entry("j_h2o_sto3g_bse", ("mmd", "j", 0, 4),
          f"J of {_H2O}, in data/basis/sto-3g.json, for h2o_sto3g_j's density",
          basis="STO-3G", molecule="h2o_sto3g_j"),
    Entry("j_h2o_631g_bse", ("mmd", "j", 0, 4),
          f"J of {_H2O}, in data/basis/6-31g.json, for h2o_631g_j's density",
          basis="6-31G", molecule="h2o_631g_j"),
    Entry("j_h2o_ccpvdz_bse", ("mmd", "j", 0, 8),
          f"J of {_H2O}, in data/basis/cc-pVDZ.json as the file states it "
          "(general contraction, no free-primitive reduction), for "
          "h2o_ccpvdz_j's density",
          basis="cc-pVDZ", molecule="h2o_ccpvdz_j"),
    # The DF metric over two fitting files no fixture covers.
    Entry("metric_h2o_ccpvdz_jkfit_bse", ("mmd", "metric", 0, 6),
          f"(P|Q) of data/aux_basis/cc-pVDZ-JKFIT.json on {_H2O} (O to f)",
          basis="cc-pVDZ-JKFIT", molecule="h2o_sto3g_j"),
    Entry("metric_h2o_def2_universal_jkfit_bse", ("mmd", "metric", 0, 8),
          f"(P|Q) of data/aux_basis/def2-universal-JKFIT.json on {_H2O} "
          "(O to g)",
          basis="def2-universal-JKFIT", molecule="h2o_sto3g_j"),
    # The uncontracted d1 metric (#693) over cc-pVDZ-JKFIT, one axis each.
    *(Entry(f"metric_ip1{x}_h2o_ccpvdz_jkfit_bse", ("mmd", "metric", 1, 6),
            f"d(P|Q)/dA_{x} of data/aux_basis/cc-pVDZ-JKFIT.json on {_H2O} "
            "(O to f), A the centre of P",
            basis="cc-pVDZ-JKFIT", molecule="h2o_sto3g_j", component=x)
      for x in "xyz"),
    # What test/oracle/fixtures/ already holds, indexed, not copied.
    Entry("j_h2_sto3g", ("mmd", "j", 0, 0), "h2_sto3g_j's vj",
          fixture="h2_sto3g_j", array=fixtures.J_ARRAY),
    Entry("j_h2o_sto3g", ("mmd", "j", 0, 4), "h2o_sto3g_j's vj",
          fixture="h2o_sto3g_j", array=fixtures.J_ARRAY),
    Entry("j_h2o_631g", ("mmd", "j", 0, 4), "h2o_631g_j's vj",
          fixture="h2o_631g_j", array=fixtures.J_ARRAY),
    Entry("j_h2o_ccpvdz", ("mmd", "j", 0, 8), "h2o_ccpvdz_j's vj",
          fixture="h2o_ccpvdz_j", array=fixtures.J_ARRAY),
    Entry("j_h10_chain_ccpvdz", ("mmd", "j", 0, 8),
          "h10_chain_ccpvdz_j's vj (an RHF density)",
          fixture="h10_chain_ccpvdz_j", array=fixtures.J_ARRAY),
    Entry("j_co_ccpvtz", ("mmd", "j", 0, 12), "co_ccpvtz_j's vj",
          fixture="co_ccpvtz_j", array=fixtures.J_ARRAY),
    Entry("metric_h2o_ccpvtz_jkfit", ("mmd", "metric", 0, 8),
          "the h2o_ccpvtz_jkfit_metric set's int2c2e_cart",
          metric_set="h2o_ccpvtz_jkfit_metric"),
    Entry("metric_prim_spd_f_auxi", ("mmd", "metric", 0, 12),
          "the prim_spd_f_auxi_metric set's int2c2e_cart (aux to i)",
          metric_set="prim_spd_f_auxi_metric"),
    # #688: the Coulomb derivative matrix of the RHF Hessian's CPHF
    # right-hand side, over the J entries' files, molecule and density: its
    # ip1 bra half J1 (kernel_jx's output) and the whole J^X (host_jx). Keyed
    # by the top JX bra class the build reaches, (pp) and (dd).
    Entry("jx1_h2o_sto3g_bse", ("mmd", "jx", 1, 2),
          f"J1 (ip1 bra half of J^X) of {_H2O}, in data/basis/sto-3g.json, "
          "for h2o_sto3g_j's density",
          basis="STO-3G", molecule="h2o_sto3g_j", form="jx1"),
    Entry("jx_h2o_sto3g_bse", ("mmd", "jx", 1, 2),
          f"J^X (both halves) of {_H2O}, in data/basis/sto-3g.json, for "
          "h2o_sto3g_j's density",
          basis="STO-3G", molecule="h2o_sto3g_j", form="jx"),
    Entry("jx1_h2o_ccpvdz_bse", ("mmd", "jx", 1, 4),
          f"J1 (ip1 bra half of J^X) of {_H2O}, in data/basis/cc-pVDZ.json "
          "as the file states it, for h2o_ccpvdz_j's density",
          basis="cc-pVDZ", molecule="h2o_ccpvdz_j", form="jx1"),
    Entry("jx_h2o_ccpvdz_bse", ("mmd", "jx", 1, 4),
          f"J^X (both halves) of {_H2O}, in data/basis/cc-pVDZ.json as the "
          "file states it, for h2o_ccpvdz_j's density",
          basis="cc-pVDZ", molecule="h2o_ccpvdz_j", form="jx"),
)

BY_NAME = {e.name: e for e in ENTRIES}


def entries(kind: str | None = None) -> list[str]:
    """Every entry's name, in recipe order; only `kind`'s if given."""
    return [e.name for e in ENTRIES if kind is None or e.kind == kind]


def entry(name: str) -> Entry:
    try:
        return BY_NAME[name]
    except KeyError:
        raise GoldenError(f"no numeric golden {name!r}; known: "
                          f"{entries()}") from None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basis_file(e: Entry) -> Path:
    """The `data/` file an npy entry's basis is read from."""
    assert e.basis is not None
    return bse.find(e.basis, DIRECTORY[e.quantity])


@cache
def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())


def record(name: str) -> dict[str, Any]:
    """The manifest's record of `name`."""
    got = _manifest()["entries"].get(name)
    if got is None:
        raise GoldenError(f"{MANIFEST}: no record of {name!r} -- regenerate "
                          f"with {COMMAND} {name}")
    return got


@cache
def _molecule(name: str) -> fixtures.Fixture:
    return fixtures.load(fixtures.FIXTURES / f"{name}.json")


def atoms(name: str) -> list[tuple[int, tuple[float, float, float]]]:
    """An npy entry's atoms, (Z, centre in Bohr) each: its molecule
    fixture's."""
    e = entry(name)
    if e.molecule is None:
        raise GoldenError(f"{name}: a fixture entry; its inputs are the "
                          f"fixture {e.fixture or e.metric_set}'s")
    return [(a.charge, a.coord) for a in _molecule(e.molecule).atoms]


def basis(name: str) -> Basis:
    """An npy entry's shells in OUR convention, from the `data/` file with
    no PySCF -- the same functions, in the same AO order, as the Mole the
    generator built (it checks that before it writes)."""
    from .basis import from_bse

    e = entry(name)
    return from_bse(bse.load(basis_file(e)), atoms(name), reduce=False)


def density(name: str) -> np.ndarray:
    """A `j` or `jx` npy entry's density: its molecule fixture's recorded
    one."""
    e = entry(name)
    if e.kind != "npy" or e.what not in WITH_DENSITY:
        raise GoldenError(f"{name}: only a j or jx npy entry has a density")
    return _molecule(e.molecule).density


@cache
def want(name: str) -> np.ndarray:
    """The golden array, read-only float64: the npy file, or the fixture's."""
    e = entry(name)
    if e.fixture is not None:
        a = _molecule(e.fixture).arrays[e.array]
    elif e.metric_set is not None:
        a = fixtures.metric_set(e.metric_set).metric
    else:
        rec = record(name)
        a = np.load(STORE / rec["file"], allow_pickle=False)
        if a.dtype != np.float64 or list(a.shape) != rec["shape"]:
            raise GoldenError(f"{name}: {rec['file']} is {a.dtype} "
                              f"{a.shape}, the manifest says float64 "
                              f"{rec['shape']}")
    a = np.array(a, dtype=np.float64)
    a.setflags(write=False)
    return a


def compare(name: str, got: np.ndarray, *, precision: Scalar = Scalar.F64,
            scale: float | None = None,
            index: Sequence[Sequence[int]] | None = None) -> float:
    """Assert `got` is the golden `name` to `precision`'s tier
    (`oracle.tolerance`), relative to `scale` (default: the golden's largest
    `|want|` over the WHOLE array, so a sub-block is not judged against its
    own small elements); over `index` (one sequence of indices per axis,
    `np.ix_`) when given. Returns the error as a fraction of `scale`."""
    full = want(name)
    w = full if index is None else full[np.ix_(*index)]
    got = np.asarray(got)
    if got.shape != w.shape:
        raise AssertionError(f"{name}: got shape {got.shape}, the golden "
                             f"{'block ' if index is not None else ''}is "
                             f"{w.shape}")
    if scale is None:
        scale = float(np.max(np.abs(full)))
    return assert_within(got, w, scale, precision=precision,
                         what=f"numeric golden {name}")


# -- the manifest and the files, held to the recipe ---------------------------

# The fields an npy record carries; `provenance` is the generator's.
NPY_FIELDS = frozenset({"key", "kind", "description", "quantity", "molecule",
                        "basis", "density", "file", "shape", "dtype",
                        "sha256", "provenance"})
FIXTURE_FIELDS = frozenset({"key", "kind", "description", "source"})
PROVENANCE_FIELDS = frozenset({"pyscf", "libcint", "numpy", "date", "host",
                               "command", "cart"})


def recipe_record(e: Entry) -> dict[str, Any]:
    """What the manifest records of `e` that follows from the recipe and the
    tracked inputs alone -- everything but the array's own fields and the
    provenance. The generator writes it; `check_manifest` recomputes it."""
    out: dict[str, Any] = {"key": list(e.key), "kind": e.kind,
                           "description": e.description}
    if e.kind == "fixture":
        out["source"] = ({"fixture": e.fixture, "array": e.array}
                         if e.fixture is not None
                         else {"metric_set": e.metric_set})
        return out
    path = basis_file(e)
    prov = json.loads((path.parent / bse.PROVENANCE).read_text())
    mol = _molecule(e.molecule)
    out["quantity"] = QUANTITIES[e.what]
    out["molecule"] = {"fixture": e.molecule, "mol": mol.provenance.mol}
    out["basis"] = {
        "file": path.relative_to(bse.DATA.parent).as_posix(),
        "name": bse.load(path).name,
        "sha256": prov["files"][path.name]["sha256"],
        "provenance": (path.parent / bse.PROVENANCE).relative_to(
            bse.DATA.parent).as_posix(),
        "commit": prov["commit"],
        "form": ("the file's contractions in libcint's order (by l, file "
                 "order within one), no free-primitive reduction; read "
                 "Cartesian (cart=True)"),
    }
    out["density"] = ({"fixture": e.molecule,
                       "sha256": fixtures.density_sha256(mol.density)}
                      if e.what in WITH_DENSITY else None)
    out["file"] = f"{e.name}.npy"
    return out


def check_manifest() -> list[str]:
    """Every disagreement between the recipe, the manifest and the files, as
    one message each; empty when they agree."""
    from registry import REGISTRY, Key

    bad: list[str] = []
    raw = _manifest()
    if raw.get("schema") != SCHEMA:
        bad.append(f"manifest schema {raw.get('schema')!r}, not {SCHEMA}")
    recs = raw.get("entries", {})
    if list(recs) != entries():
        bad.append(f"manifest entries {list(recs)} are not the recipe's "
                   f"{entries()}, in order")
    for e in ENTRIES:
        if Key(*e.key) not in REGISTRY:
            bad.append(f"{e.name}: {e.key} is not a registry case")
        rec = recs.get(e.name)
        if rec is None:
            continue
        fields = NPY_FIELDS if e.kind == "npy" else FIXTURE_FIELDS
        if set(rec) != fields:
            bad.append(f"{e.name}: fields {sorted(rec)}, want {sorted(fields)}")
            continue
        for k, v in recipe_record(e).items():
            if rec[k] != v:
                bad.append(f"{e.name}: {k} is {rec[k]!r}, the recipe and "
                           f"the tracked inputs give {v!r}")
        if e.kind == "fixture":
            continue
        if set(rec["provenance"]) != PROVENANCE_FIELDS:
            bad.append(f"{e.name}: provenance fields "
                       f"{sorted(rec['provenance'])}")
        path = STORE / rec["file"]
        if not path.is_file():
            bad.append(f"{e.name}: {rec['file']} is missing")
            continue
        if sha256(path) != rec["sha256"]:
            bad.append(f"{e.name}: {rec['file']}'s sha256 is not the "
                       "manifest's")
        a = np.load(path, allow_pickle=False)
        if (rec["dtype"] != "float64" or a.dtype != np.float64
                or list(a.shape) != rec["shape"]):
            bad.append(f"{e.name}: {rec['file']} is {a.dtype} {list(a.shape)}"
                       f", the manifest says {rec['dtype']} {rec['shape']}")
        if not a.flags.c_contiguous or not np.all(np.isfinite(a)):
            bad.append(f"{e.name}: {rec['file']} is not a finite C-order "
                       "array")
    tracked = {f"{n}.npy" for n in entries("npy")}
    stray = {p.name for p in STORE.glob("*.npy")} - tracked
    if stray:
        bad.append(f"files no entry names: {sorted(stray)}")
    return bad


__all__ = ["COMMAND", "DIRECTORY", "ENTRIES", "MANIFEST", "QUANTITIES",
           "SCHEMA", "STORE", "WITH_DENSITY", "Entry", "GoldenError", "atoms",
           "basis", "basis_file", "check_manifest", "compare", "density", "entries",
           "entry", "record", "recipe_record", "sha256", "want"]
