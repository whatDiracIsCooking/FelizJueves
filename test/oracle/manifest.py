"""The fixture schema, its loader, and the provenance record -- NUMPY ONLY.

Nothing here imports PySCF, so a fast-tier test can load and validate every
tracked fixture on a box where the `oracle` group was never installed. The
writer is `generate.py`; this module only defines what a fixture IS and refuses
anything that is not one, with a message naming the file.

THE SHAPE (schema 2, 3 or 4), one JSON object per file under `fixtures/`:

    schema       2, 3 or 4 (#240: schema 1 had no one-electron arrays; a
                 schema-1 file is refused by name, not read as one with them
                 absent). Schema 3 (#411) adds one optional key, `metric`
                 (THE METRIC PIECES, below), and schema 4 (#442) another,
                 `three_center` (THE THREE-CENTER PIECES); an AO fixture is
                 still written as schema 2 and a metric piece as schema 3,
                 unchanged, and a `metric` key in a schema-2 file or a
                 `three_center` key in a file below schema 4 is refused.
    name         the file's stem
    description  free text: what the case is for
    provenance   {pyscf, libcint, numpy, date, host, command, mol, basis, cart}
                 -- exactly these keys. Versions, an ISO date, the host (marked
                 "(container)" inside one, as `measured` marks it), the command
                 that wrote the file, the molecule as PySCF was given it
                 ({atom, unit, charge, spin}), the basis name, and whether the
                 AOs are Cartesian.
    atoms        [{symbol, charge, coord}]          coord in Bohr
    shells       [{atom, l, exponents, coefficients}]
                 the basis as PySCF's library parsed it (`Mole._basis`), in
                 libcint's shell order; coefficients[k][c] is primitive k of
                 contraction c, NOT normalised -- normalisation conventions are
                 #84's to pin, so the raw numbers are what is recorded.
    nao          number of AOs
    ao_labels    PySCF's `ao_labels()`, one per AO
    ao_loc       shell offsets into the AO list, len(shells) + 1 entries
    arrays       {name: {shape, values}} -- values flattened row-major
                 `density` (nao x nao, exactly symmetric) is required.
                 `int2e:i,j,k,l` (OPTIONAL, #85; needs `converted`) is
                 libcint's Cartesian `(ij|kl)` block for converted shells
                 i, j, k, l, shape (n_i, n_j, n_k, n_l), AOs in `cartesians`
                 order -- one shell quartet, never the whole tensor
                 `vj` (OPTIONAL, #86; needs `converted`) is PySCF's
                 `get_jk` Coulomb matrix (nao x nao) for `density`, AO order
                 `vk` (OPTIONAL, #199; needs `converted`) is PySCF's
                 `get_jk` exchange matrix (nao x nao) for `density`, AO
                 order: `vk_il = sum_jk (ij|kl) D_jk`, which is our
                 `K_ab = sum_cd D_cd (ac|bd)`
                 `int1e_ovlp`, `int1e_kin`, `int1e_nuc`, `hcore` (OPTIONAL,
                 #240; each needs `converted`, all four or none) are
                 libcint's `S`, `T`, `V` (point nuclei, `-Z_C` folded in)
                 and PySCF's `get_hcore` (`T + V`), nao x nao, AO order.
                 `int1e_ovlp` is the same `intor` call as `ovlp` and must
                 equal it exactly; the legacy `ovlp` stays in every file.
                 `int1e_ovlp_cross:<rows>` (OPTIONAL; needs `converted`) is
                 libcint's rectangular overlap `<m|n>`, shape
                 (rows' nao, nao): `m` an AO of the tracked fixture `<rows>`
                 (which must carry `converted` and the same `cart`), `n` of
                 this one -- the oracle of `S_{B1,B2}` (`Target.SX`);
                 `cross_overlap(fx)` loads the partner and checks it.
    density_recipe  how the recorded density was made: "hilbert: ..."
                 (1/(i+j+1), every case but one pair) or "rhf: ..."
                 (#486, PySCF's converged RHF density, recorded and
                 from then on the input; oracle.generate.case_density)
    converted    OPTIONAL. {common_factor, shells}: the basis in OUR convention
                 (`ao.shell.Shell`, one entry per contraction, in AO order) as
                 `oracle.basis` converts it -- coefficients are libcint's
                 normalised `_libcint_ctr_coeff` times `common_factor(l)`, to
                 multiply unnormalised Cartesian primitives -- and
                 `common_factor`, libcint's `common_fac` per l = 0.. as PySCF's
                 data yields it. Present only where a case records it; it lets
                 a later test build our shells with no PySCF. Cartesian only.

THE METRIC PIECES (#411, schema 3). An auxiliary (density-fitting) basis's
two-centre Coulomb metric `(P|Q)` -- libcint's `int2c2e_cart` -- and its
overlap `int1e_ovlp`, nao_aux x nao_aux, are each over the repo's 512 KiB
`check-added-large-files` cap on their own (166 x 166 for cc-pVTZ-jkfit on
H2O: ~556 KB for the metric alone), so, as #121 split HF/cc-pVTZ's ERIs by
L, the WHOLE matrices are split across files by row block. A piece carries

    metric       {set, rows, parent, converted}: `set` the name every piece
                 of one matrix shares, `rows` = [r0, r1) the aux AOs whose
                 rows it holds (r0, r1 on shell boundaries, `ao_loc`),
                 `parent` the tracked AO fixture whose molecule the aux basis
                 was put on (`df.make_auxmol(parent, basis)`), and
                 `converted` the aux basis in our convention
                 ({common_factor, shells}, as the top-level slot). It is kept
                 HERE and not in the top-level `converted` slot because it is
                 not an AO basis: every consumer that walks the converted
                 fixtures as AO bases (`Fixture.converted`) must not meet it.
    arrays       exactly `int2c2e_cart:rows` and `int1e_ovlp:rows`, each
                 (r1 - r0, nao), full width -- NO `density` (a piece is not
                 an AO fixture; `Fixture.density` refuses one).

and its stem is `<set>_rows<r0>_<r1>`. The rest of the shape (provenance,
atoms, shells, nao, ao_labels, ao_loc) is the aux basis's, the same in every
piece. `metric_set(set)` reassembles the whole matrices and, the way
`fock_pair` does, refuses a missing piece (a row gap), an overlapping one,
and a piece whose inputs (provenance but its date / host / command, atoms,
shells, nao, ao_labels, ao_loc, parent, converted) differ from the first's,
by name, never by a tolerance; the reassembled matrices must be symmetric to
1e-14 of their largest element (libcint computes both triangles).
`metric_sets()` names every tracked set.

THE THREE-CENTER PIECES (#442, schema 4). libcint's three-center Coulomb
integral `(mn|P)` -- `int3c2e_cart`, through `df.incore.aux_e2(mol, auxmol,
aosym="s1")` -- with `m, n` AOs of a tracked AO fixture (the `parent`) and
`P` an AO of the auxiliary basis `df.make_auxmol(parent, basis)`. A piece
carries

    three_center {set, cols, parent, converted}: `set` the name every piece
                 of one tensor shares, `cols` = [c0, c1) the aux AOs whose
                 columns it holds (on aux shell boundaries, `ao_loc`),
                 `parent` the tracked AO fixture whose molecule and basis are
                 the bra side, and `converted` the aux basis in our
                 convention (kept here, not top-level, for the metric's
                 reason).
    arrays       exactly `int3c2e_cart:<parent>`, shape
                 (parent's nao, parent's nao, c1 - c0) -- NO `density`.

and its stem is `<set>_cols<c0>_<c1>`. The rest of the shape (provenance,
atoms, shells, nao, ao_labels, ao_loc) is the AUX basis's, as a metric
piece's is. `three_center_set(set)` reassembles the whole
(nao_parent, nao_parent, nao_aux) tensor and refuses a column gap, an
overlapping or duplicate piece, a piece whose inputs differ from the
first's (as `metric_set` compares them), and a parent that is absent, has
no converted basis, or does not match what the piece carries: its atoms,
`provenance.mol` and `cart`, and the array's AO width as the parent's
`nao`. The tensor must be symmetric in `m, n` to 1e-14 of its largest
element -- NOT bitwise: libcint computes both triangles separately (1,998 of
the 8,134 entries of the tracked set differ from their transpose, by at most
4.4e-16). `three_center_sets()` names every tracked set.

THE FOCK PAIR (#280, milestone 12 pin 10). No fixture records the
two-electron Fock matrix; it is COMBINED, never stored, from a `<m>_j` /
`<m>_k` pair, which `fock_pair(m)` loads and `check_pair` asserts share
their inputs before anything is combined: the raw `shells`, the `density`
(exactly), `ao_loc`, `converted.common_factor` and `provenance.mol` (the
issue's five), and also `provenance.basis` / `cart`, `atoms`, `nao`,
`ao_labels`, the converted shells, the density recipe, and every array the
two files both carry (`ovlp`, the one-electron matrices) -- each exactly.
A pair regenerated at different times from different molecules would pass
numerically on a small case and be meaningless, so a mismatch is a
`FixtureError` naming both files and the field, never a tolerance. The
combined quantity is `FockPair.two_electron() = vj - vk / 2`: PySCF's RHF
`vj - 0.5 vk` for the total closed-shell density, `FOCK_EXCHANGE_FRACTION`
= 1/2 (derived in `felizjueves.ao.fock`, which this module must not import;
test/test_oracle_fock.py asserts the two constants agree). `fock_pairs()`
names every tracked molecule that has both halves.

FULL PRECISION. Floats are written by `json`, i.e. by `repr`, which is the
shortest string that round-trips the double exactly, so "the numbers match"
can mean `==` and not a tolerance. `dumps` is the one serialiser, and a
tracked fixture is byte-for-byte `dumps(load(...))`.

`python -m oracle.manifest` (PYTHONPATH=test) loads every tracked fixture and
prints one line each; devtools/doctor.sh's oracle row runs exactly that.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 4  # the newest this loader reads (#442: the three-center pieces)
AO_SCHEMA = 2  # what an AO fixture is written as, unchanged since #240
METRIC_SCHEMA = 3  # what a metric piece is written as, unchanged since #411
THREE_CENTER_SCHEMA = 4  # what a three-center piece is written as (#442)
SCHEMAS = (AO_SCHEMA, METRIC_SCHEMA, THREE_CENTER_SCHEMA)
FIXTURES = Path(__file__).resolve().parent / "fixtures"

PROVENANCE_KEYS = (
    "pyscf", "libcint", "numpy", "date", "host", "command", "mol", "basis", "cart",
)
MOL_KEYS = ("atom", "unit", "charge", "spin")
TOP_KEYS = (
    "schema", "name", "description", "provenance", "atoms", "shells", "nao",
    "ao_labels", "ao_loc", "arrays", "density_recipe",
)
OPTIONAL_KEYS = ("converted", "metric", "three_center")
REQUIRED_ARRAYS = ("density",)
ERI_PREFIX = "int2e:"  # an ERI block's array name: int2e:i,j,k,l (#85)
J_ARRAY = "vj"  # PySCF get_jk's Coulomb matrix for `density` (#86)
K_ARRAY = "vk"  # PySCF get_jk's exchange matrix for `density` (#199)
# #240: libcint's one-electron matrices and PySCF's core Hamiltonian.
OVLP_ARRAY = "int1e_ovlp"
KIN_ARRAY = "int1e_kin"
NUC_ARRAY = "int1e_nuc"
HCORE_ARRAY = "hcore"
SCHWARZ_ARRAY = "int2e_diag"  # libcint's (mn|mn) for every AO pair
# libcint's <m|n>, m an AO of the named fixture (rows), n of this one
CROSS_PREFIX = "int1e_ovlp_cross:"
ONE_E_ARRAYS = (OVLP_ARRAY, KIN_ARRAY, NUC_ARRAY, HCORE_ARRAY)
# #411: a metric piece's two row blocks (THE METRIC PIECES)
METRIC_ARRAY = "int2c2e_cart:rows"
AUX_OVLP_ARRAY = "int1e_ovlp:rows"
METRIC_ARRAYS = (METRIC_ARRAY, AUX_OVLP_ARRAY)
METRIC_KEYS = ("converted", "parent", "rows", "set")
ROWS_SEP = "_rows"
# #442: a three-center piece's one array, `int3c2e_cart:<parent>`
THREE_CENTER_PREFIX = "int3c2e_cart:"
THREE_CENTER_KEYS = ("cols", "converted", "parent", "set")
COLS_SEP = "_cols"


class FixtureError(ValueError):
    """A fixture file that is not a valid fixture. The message names the file."""


@dataclass(frozen=True)
class Provenance:
    """Who computed a fixture, with what, and from which inputs."""

    pyscf: str
    libcint: str
    numpy: str
    date: str
    host: str
    command: str
    mol: dict[str, Any]
    basis: str
    cart: bool


@dataclass(frozen=True)
class Atom:
    symbol: str
    charge: int
    coord: tuple[float, float, float]  # Bohr


@dataclass(frozen=True)
class Shell:
    """One contracted shell, as PySCF's basis library gives it (unnormalised)."""

    atom: int
    l: int  # noqa: E741 -- the angular momentum's own name
    exponents: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]  # [primitive][contraction]


@dataclass(frozen=True)
class ConvertedShell:
    """One contraction in our convention (`felizjueves.ao.shell.Shell`'s fields
    plus the atom it sits on). Plain data: this module must not import `src`."""

    atom: int
    l: int  # noqa: E741
    centre: tuple[float, float, float]  # Bohr
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class Converted:
    common_factor: tuple[float, ...]  # indexed by l
    shells: tuple[ConvertedShell, ...]


@dataclass(frozen=True)
class MetricPiece:
    """A schema-3 file's `metric` record: rows [r0, r1) of set `set`."""

    set: str
    rows: tuple[int, int]
    parent: str
    converted: Converted


@dataclass(frozen=True)
class ThreeCenterPiece:
    """A schema-4 file's `three_center` record: aux columns [c0, c1) of set
    `set`, the bra side the tracked fixture `parent`."""

    set: str
    cols: tuple[int, int]
    parent: str
    converted: Converted

    @property
    def array(self) -> str:
        return f"{THREE_CENTER_PREFIX}{self.parent}"


@dataclass(frozen=True)
class Fixture:
    path: Path
    name: str
    description: str
    provenance: Provenance
    atoms: tuple[Atom, ...]
    shells: tuple[Shell, ...]
    nao: int
    ao_labels: tuple[str, ...]
    ao_loc: tuple[int, ...]
    arrays: dict[str, np.ndarray]  # read-only
    density_recipe: str
    converted: Converted | None = None
    metric: MetricPiece | None = None
    three_center: ThreeCenterPiece | None = None

    @property
    def density(self) -> np.ndarray:
        if self.metric is not None:
            raise _fail(self.path, "a metric piece records no density")
        if self.three_center is not None:
            raise _fail(self.path, "a three-center piece records no density")
        return self.arrays["density"]


# --- serialisation ---------------------------------------------------------


def encode_array(a: np.ndarray) -> dict[str, Any]:
    """`{shape, values}`, row-major, every value a Python float (full repr)."""
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "values": [float(x) for x in a.ravel()]}


def dumps(obj: dict[str, Any]) -> str:
    """The one serialiser: what a tracked fixture's bytes are."""
    return json.dumps(obj, indent=1, allow_nan=False) + "\n"


# --- loading and validation ------------------------------------------------


def _fail(path: Path, msg: str) -> FixtureError:
    return FixtureError(f"{path}: {msg}")


def _is_num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _floats(path: Path, where: str, xs: Any, n: int | None = None) -> tuple:
    if not isinstance(xs, list) or not all(_is_num(x) for x in xs):
        raise _fail(path, f"{where} must be a list of numbers")
    if n is not None and len(xs) != n:
        raise _fail(path, f"{where} has {len(xs)} entries, expected {n}")
    if not all(math.isfinite(x) for x in xs):
        raise _fail(path, f"{where} has a non-finite value")
    return tuple(float(x) for x in xs)


def _provenance(path: Path, p: Any) -> Provenance:
    if not isinstance(p, dict):
        raise _fail(path, "provenance must be an object")
    missing = [k for k in PROVENANCE_KEYS if k not in p]
    if missing:
        raise _fail(path, f"provenance is missing key(s) {missing}")
    extra = sorted(set(p) - set(PROVENANCE_KEYS))
    if extra:
        raise _fail(path, f"provenance has unknown key(s) {extra}")
    for k in ("pyscf", "libcint", "numpy", "date", "host", "command", "basis"):
        if not isinstance(p[k], str) or not p[k]:
            raise _fail(path, f"provenance.{k} must be a non-empty string")
    if not isinstance(p["cart"], bool):
        raise _fail(path, "provenance.cart must be true or false")
    mol = p["mol"]
    if not isinstance(mol, dict) or sorted(mol) != sorted(MOL_KEYS):
        raise _fail(path, f"provenance.mol must have exactly the keys {MOL_KEYS}")
    return Provenance(**{k: p[k] for k in PROVENANCE_KEYS})


def _arrays(path: Path, raw: Any, required: tuple[str, ...] = REQUIRED_ARRAYS,
            ) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict):
        raise _fail(path, "arrays must be an object")
    missing = [k for k in required if k not in raw]
    if missing:
        raise _fail(path, f"arrays is missing {missing}")
    out = {}
    for name, a in raw.items():
        if not isinstance(a, dict) or sorted(a) != ["shape", "values"]:
            raise _fail(path, f"arrays.{name} must be {{shape, values}}")
        shape = a["shape"]
        if not (isinstance(shape, list) and shape
                and all(isinstance(d, int) and d > 0 for d in shape)):
            raise _fail(path, f"arrays.{name}.shape must be positive integers")
        need = math.prod(shape)
        values = a["values"]
        if isinstance(values, list) and len(values) != need:
            raise _fail(path, f"arrays.{name} is truncated or padded: "
                              f"{len(values)} values for shape {shape} "
                              f"(needs {need})")
        vals = _floats(path, f"arrays.{name}.values", values, need)
        arr = np.array(vals, dtype=np.float64).reshape(shape)
        arr.setflags(write=False)
        out[name] = arr
    return out


def validate(path: Path, raw: Any) -> Fixture:
    """Turn a parsed JSON object into a `Fixture`, or raise naming `path`."""
    path = Path(path)
    if not isinstance(raw, dict):
        raise _fail(path, "a fixture must be a JSON object")
    missing = [k for k in TOP_KEYS if k not in raw]
    if missing:
        raise _fail(path, f"missing key(s) {missing}")
    extra = sorted(set(raw) - set(TOP_KEYS) - set(OPTIONAL_KEYS))
    if extra:
        raise _fail(path, f"unknown key(s) {extra}")
    if raw["schema"] == 1:
        raise _fail(path, f"schema 1, this loader reads {SCHEMA}: the bump to "
                          "2 (#240) added the one-electron arrays -- "
                          "regenerate it (python -m oracle.generate "
                          "--regenerate)")
    if raw["schema"] not in SCHEMAS:
        raise _fail(path, f"schema {raw['schema']!r}, this loader reads "
                          f"{list(SCHEMAS)}")
    if "metric" in raw and raw["schema"] < METRIC_SCHEMA:
        raise _fail(path, f"a metric piece is schema {METRIC_SCHEMA} (#411), "
                          f"and this file says schema {raw['schema']}")
    if "three_center" in raw and raw["schema"] < THREE_CENTER_SCHEMA:
        raise _fail(path, f"a three-center piece is schema "
                          f"{THREE_CENTER_SCHEMA} (#442), and this file says "
                          f"schema {raw['schema']}")
    if "metric" in raw and "three_center" in raw:
        raise _fail(path, "a file is a metric piece or a three-center piece, "
                          "not both")
    if raw["name"] != path.stem:
        raise _fail(path, f"name {raw['name']!r} is not the file's stem")

    prov = _provenance(path, raw["provenance"])

    atoms = []
    for i, a in enumerate(raw["atoms"] if isinstance(raw["atoms"], list) else []):
        if not isinstance(a, dict) or sorted(a) != ["charge", "coord", "symbol"]:
            raise _fail(path, f"atoms[{i}] must be {{symbol, charge, coord}}")
        atoms.append(Atom(a["symbol"], int(a["charge"]),
                          _floats(path, f"atoms[{i}].coord", a["coord"], 3)))
    if not atoms:
        raise _fail(path, "atoms must be a non-empty list")

    shells = []
    for i, s in enumerate(raw["shells"] if isinstance(raw["shells"], list) else []):
        keys = ["atom", "coefficients", "exponents", "l"]
        if not isinstance(s, dict) or sorted(s) != keys:
            raise _fail(path, f"shells[{i}] must be {{atom, l, exponents, "
                              "coefficients}")
        if not (isinstance(s["atom"], int) and 0 <= s["atom"] < len(atoms)):
            raise _fail(path, f"shells[{i}].atom is not an atom index")
        if not (isinstance(s["l"], int) and s["l"] >= 0):
            raise _fail(path, f"shells[{i}].l must be a non-negative integer")
        exps = _floats(path, f"shells[{i}].exponents", s["exponents"])
        if not exps or min(exps) <= 0:
            raise _fail(path, f"shells[{i}].exponents must be positive")
        coefs = s["coefficients"]
        if not isinstance(coefs, list) or len(coefs) != len(exps):
            raise _fail(path, f"shells[{i}].coefficients needs one row per "
                              f"primitive ({len(exps)})")
        rows = [_floats(path, f"shells[{i}].coefficients[{k}]", r)
                for k, r in enumerate(coefs)]
        if len({len(r) for r in rows}) != 1 or not rows[0]:
            raise _fail(path, f"shells[{i}].coefficients is ragged")
        shells.append(Shell(s["atom"], s["l"], exps, tuple(rows)))
    if not shells:
        raise _fail(path, "shells must be a non-empty list")

    nao = raw["nao"]
    if not (isinstance(nao, int) and nao > 0):
        raise _fail(path, "nao must be a positive integer")
    ao_loc = raw["ao_loc"]
    if not (isinstance(ao_loc, list) and all(isinstance(x, int) for x in ao_loc)
            and len(ao_loc) == len(shells) + 1 and ao_loc[0] == 0
            and ao_loc[-1] == nao):
        raise _fail(path, f"ao_loc must be {len(shells) + 1} integer offsets "
                          f"from 0 to nao={nao}")
    for i, sh in enumerate(shells):
        ncomp = (sh.l + 1) * (sh.l + 2) // 2 if prov.cart else 2 * sh.l + 1
        width = ao_loc[i + 1] - ao_loc[i]
        if width != ncomp * len(sh.coefficients[0]):
            raise _fail(path, f"ao_loc gives shell {i} {width} AOs, its l and "
                              "contractions need "
                              f"{ncomp * len(sh.coefficients[0])}")
    labels = raw["ao_labels"]
    if not (isinstance(labels, list) and len(labels) == nao
            and all(isinstance(x, str) for x in labels)):
        raise _fail(path, f"ao_labels must be {nao} strings")

    for k in ("description", "density_recipe"):
        if not isinstance(raw[k], str) or not raw[k]:
            raise _fail(path, f"{k} must be a non-empty string")
    if "metric" in raw:
        if "converted" in raw:
            raise _fail(path, "a metric piece keeps its aux basis in "
                              "metric.converted, not in converted")
        arrays = _arrays(path, raw["arrays"], required=())
        metric = _metric(path, raw, prov, len(atoms), shells, nao,
                         tuple(ao_loc), arrays)
        return Fixture(path, raw["name"], raw["description"], prov,
                       tuple(atoms), tuple(shells), nao, tuple(labels),
                       tuple(ao_loc), arrays, raw["density_recipe"], None,
                       metric)
    if "three_center" in raw:
        if "converted" in raw:
            raise _fail(path, "a three-center piece keeps its aux basis in "
                              "three_center.converted, not in converted")
        arrays = _arrays(path, raw["arrays"], required=())
        tc = _three_center(path, raw, prov, len(atoms), shells, nao,
                           tuple(ao_loc), arrays)
        return Fixture(path, raw["name"], raw["description"], prov,
                       tuple(atoms), tuple(shells), nao, tuple(labels),
                       tuple(ao_loc), arrays, raw["density_recipe"], None,
                       None, tc)

    arrays = _arrays(path, raw["arrays"])
    d = arrays["density"]
    if d.shape != (nao, nao):
        raise _fail(path, f"density has shape {d.shape}, expected ({nao}, {nao})")
    if not np.array_equal(d, d.T):
        raise _fail(path, "density is not exactly symmetric")
    stray = sorted(n for n in arrays if n in METRIC_ARRAYS)
    if stray:
        raise _fail(path, f"arrays {stray} belong to a metric piece only")
    stray = sorted(n for n in arrays if n.startswith(THREE_CENTER_PREFIX))
    if stray:
        raise _fail(path, f"arrays {stray} belong to a three-center piece "
                          "only")

    converted = (_converted(path, raw["converted"], prov, len(atoms), shells,
                            nao)
                 if "converted" in raw else None)
    _check_eri(path, arrays, converted)
    one_e = [n for n in ONE_E_ARRAYS if n in arrays]
    if one_e and len(one_e) != len(ONE_E_ARRAYS):
        raise _fail(path, f"arrays has {one_e} of the one-electron arrays "
                          f"{list(ONE_E_ARRAYS)}: all four or none")
    for name in (J_ARRAY, K_ARRAY, SCHWARZ_ARRAY, *ONE_E_ARRAYS):
        if name not in arrays:
            continue
        if converted is None:
            raise _fail(path, f"arrays.{name} needs the converted basis "
                              "its AOs are in")
        if arrays[name].shape != (nao, nao):
            raise _fail(path, f"arrays.{name} has shape "
                              f"{arrays[name].shape}, expected "
                              f"({nao}, {nao})")
    if SCHWARZ_ARRAY in arrays:
        q = arrays[SCHWARZ_ARRAY]
        # (mn|mn) and (nm|nm) are equal exactly but computed apart, so the
        # record is symmetric to rounding, not bitwise
        if not np.all(np.abs(q - q.T) <= 1e-14 * np.max(np.abs(q))):
            raise _fail(path, f"arrays.{SCHWARZ_ARRAY} is not symmetric to "
                              "1e-14 of its largest element")
        if np.any(q < 0.0):
            raise _fail(path, f"arrays.{SCHWARZ_ARRAY} has a negative "
                              "(mn|mn)")
    for name, a in arrays.items():
        if not name.startswith(CROSS_PREFIX):
            continue
        rows = name[len(CROSS_PREFIX):]
        if not rows or rows != Path(rows).name or rows == raw["name"]:
            raise _fail(path, f"arrays.{name} must name another fixture "
                              "(its stem) as its rows")
        if converted is None:
            raise _fail(path, f"arrays.{name} needs the converted basis "
                              "its columns are in")
        if a.ndim != 2 or a.shape[1] != nao:
            raise _fail(path, f"arrays.{name} has shape {a.shape}, expected "
                              f"(rows' nao, {nao})")
    if one_e and not np.array_equal(arrays[OVLP_ARRAY], arrays["ovlp"]):
        raise _fail(path, f"arrays.{OVLP_ARRAY} is not exactly arrays.ovlp")

    return Fixture(path, raw["name"], raw["description"], prov, tuple(atoms),
                   tuple(shells), nao, tuple(labels), tuple(ao_loc), arrays,
                   raw["density_recipe"], converted)


def _converted(path: Path, c: Any, prov: Provenance, n_atoms: int,
               shells: list[Shell], nao: int) -> Converted:
    if not isinstance(c, dict) or sorted(c) != ["common_factor", "shells"]:
        raise _fail(path, "converted must be {common_factor, shells}")
    if not prov.cart:
        raise _fail(path, "converted is Cartesian only, and provenance.cart "
                          "is false")
    common = _floats(path, "converted.common_factor", c["common_factor"])
    out = []
    raw = c["shells"] if isinstance(c["shells"], list) else []
    for i, s in enumerate(raw):
        keys = ["atom", "centre", "coefficients", "exponents", "l"]
        if not isinstance(s, dict) or sorted(s) != keys:
            raise _fail(path, f"converted.shells[{i}] must be {{atom, l, "
                              "centre, exponents, coefficients}")
        if not (isinstance(s["atom"], int) and 0 <= s["atom"] < n_atoms):
            raise _fail(path, f"converted.shells[{i}].atom is not an atom index")
        if not (isinstance(s["l"], int) and 0 <= s["l"] < len(common)):
            raise _fail(path, f"converted.shells[{i}].l has no common_factor")
        centre = _floats(path, f"converted.shells[{i}].centre", s["centre"], 3)
        exps = _floats(path, f"converted.shells[{i}].exponents", s["exponents"])
        if not exps or min(exps) <= 0:
            raise _fail(path, f"converted.shells[{i}].exponents must be positive")
        coefs = _floats(path, f"converted.shells[{i}].coefficients",
                        s["coefficients"], len(exps))
        out.append(ConvertedShell(s["atom"], s["l"], centre, exps, coefs))
    want = sum(len(sh.coefficients[0]) for sh in shells)
    if len(out) != want:
        raise _fail(path, f"converted has {len(out)} shells, the basis has "
                          f"{want} contractions")
    n = sum((s.l + 1) * (s.l + 2) // 2 for s in out)
    if n != nao:
        raise _fail(path, f"converted shells give {n} AOs, nao is {nao}")
    return Converted(common, tuple(out))


def _metric(path: Path, raw: dict[str, Any], prov: Provenance, n_atoms: int,
            shells: list[Shell], nao: int, ao_loc: tuple[int, ...],
            arrays: dict[str, np.ndarray]) -> MetricPiece:
    m = raw["metric"]
    if not isinstance(m, dict) or sorted(m) != sorted(METRIC_KEYS):
        raise _fail(path, f"metric must have exactly the keys "
                          f"{sorted(METRIC_KEYS)}")
    for k in ("set", "parent"):
        if not isinstance(m[k], str) or not m[k] or m[k] != Path(m[k]).name:
            raise _fail(path, f"metric.{k} must be a fixture-style name")
    rows = m["rows"]
    if not (isinstance(rows, list) and len(rows) == 2
            and all(isinstance(r, int) and not isinstance(r, bool)
                    for r in rows)
            and 0 <= rows[0] < rows[1] <= nao):
        raise _fail(path, f"metric.rows must be [r0, r1] with "
                          f"0 <= r0 < r1 <= nao={nao}")
    r0, r1 = rows
    if r0 not in ao_loc or r1 not in ao_loc:
        raise _fail(path, f"metric.rows [{r0}, {r1}) does not fall on shell "
                          "boundaries (ao_loc)")
    want = f"{m['set']}{ROWS_SEP}{r0}_{r1}"
    if raw["name"] != want:
        raise _fail(path, f"a metric piece of set {m['set']!r}, rows "
                          f"[{r0}, {r1}) is named {want!r}")
    if sorted(arrays) != sorted(METRIC_ARRAYS):
        raise _fail(path, f"a metric piece carries exactly the arrays "
                          f"{sorted(METRIC_ARRAYS)}, not {sorted(arrays)}")
    for name in METRIC_ARRAYS:
        if arrays[name].shape != (r1 - r0, nao):
            raise _fail(path, f"arrays.{name} has shape "
                              f"{arrays[name].shape}, expected "
                              f"({r1 - r0}, {nao})")
    converted = _converted(path, m["converted"], prov, n_atoms, shells, nao)
    return MetricPiece(m["set"], (r0, r1), m["parent"], converted)


def _three_center(path: Path, raw: dict[str, Any], prov: Provenance,
                  n_atoms: int, shells: list[Shell], nao: int,
                  ao_loc: tuple[int, ...], arrays: dict[str, np.ndarray],
                  ) -> ThreeCenterPiece:
    t = raw["three_center"]
    if not isinstance(t, dict) or sorted(t) != sorted(THREE_CENTER_KEYS):
        raise _fail(path, f"three_center must have exactly the keys "
                          f"{sorted(THREE_CENTER_KEYS)}")
    for k in ("set", "parent"):
        if not isinstance(t[k], str) or not t[k] or t[k] != Path(t[k]).name:
            raise _fail(path, f"three_center.{k} must be a fixture-style name")
    cols = t["cols"]
    if not (isinstance(cols, list) and len(cols) == 2
            and all(isinstance(c, int) and not isinstance(c, bool)
                    for c in cols)
            and 0 <= cols[0] < cols[1] <= nao):
        raise _fail(path, f"three_center.cols must be [c0, c1] with "
                          f"0 <= c0 < c1 <= nao={nao}")
    c0, c1 = cols
    if c0 not in ao_loc or c1 not in ao_loc:
        raise _fail(path, f"three_center.cols [{c0}, {c1}) does not fall on "
                          "shell boundaries (ao_loc)")
    want = f"{t['set']}{COLS_SEP}{c0}_{c1}"
    if raw["name"] != want:
        raise _fail(path, f"a three-center piece of set {t['set']!r}, cols "
                          f"[{c0}, {c1}) is named {want!r}")
    name = f"{THREE_CENTER_PREFIX}{t['parent']}"
    if sorted(arrays) != [name]:
        raise _fail(path, f"a three-center piece carries exactly the array "
                          f"{name!r}, not {sorted(arrays)}")
    a = arrays[name]
    if a.ndim != 3 or a.shape[0] != a.shape[1] or a.shape[2] != c1 - c0:
        raise _fail(path, f"arrays.{name} has shape {a.shape}, expected "
                          f"(parent's nao, parent's nao, {c1 - c0})")
    converted = _converted(path, t["converted"], prov, n_atoms, shells, nao)
    return ThreeCenterPiece(t["set"], (c0, c1), t["parent"], converted)


def _eri_quartet(name: str) -> tuple[int, ...] | None:
    """`int2e:i,j,k,l` -> (i, j, k, l); None for any other array name."""
    if not name.startswith(ERI_PREFIX):
        return None
    try:
        q = tuple(int(x) for x in name[len(ERI_PREFIX):].split(","))
    except ValueError:
        return ()
    return q if len(q) == 4 and min(q) >= 0 else ()


def _check_eri(path: Path, arrays: dict[str, np.ndarray],
               converted: Converted | None) -> None:
    for name, a in arrays.items():
        q = _eri_quartet(name)
        if q is None:
            continue
        if not q:
            raise _fail(path, f"arrays.{name} is not int2e:i,j,k,l")
        if converted is None:
            raise _fail(path, f"arrays.{name} needs the converted basis its "
                              "indices name")
        if max(q) >= len(converted.shells):
            raise _fail(path, f"arrays.{name} names a shell the converted "
                              "basis does not have")
        want = tuple((converted.shells[s].l + 1) * (converted.shells[s].l + 2)
                     // 2 for s in q)
        if a.shape != want:
            raise _fail(path, f"arrays.{name} has shape {a.shape}, its shells "
                              f"need {want}")


def eri_blocks(fx: Fixture) -> dict[tuple[int, int, int, int], np.ndarray]:
    """Every `int2e:i,j,k,l` block of `fx`, keyed by its converted-shell
    quartet (validated on load)."""
    return {q: a for name, a in fx.arrays.items()
            if (q := _eri_quartet(name))}


def load(path: Path | str) -> Fixture:
    """Read and validate one fixture file. Every failure names the file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except OSError as e:
        raise _fail(path, f"cannot read: {e}") from e
    except json.JSONDecodeError as e:
        raise _fail(path, f"not valid JSON (truncated?): {e}") from e
    return validate(path, raw)


def tracked() -> list[Path]:
    """Every tracked fixture file, sorted."""
    return sorted(FIXTURES.glob("*.json"))


# --- the cross-basis overlap ------------------------------------------------


def cross_overlap(fx: Fixture) -> tuple[Fixture, np.ndarray]:
    """`(rows, S)` of a fixture carrying `int1e_ovlp_cross:<rows>`: the row
    fixture, loaded from beside `fx`, and the `rows.nao x fx.nao` array.
    Refuses a row fixture that is absent, has no converted basis, differs in
    `cart`, or whose `nao` is not the array's row count."""
    names = [n for n in fx.arrays if n.startswith(CROSS_PREFIX)]
    if len(names) != 1:
        raise _fail(fx.path, f"carries {len(names)} {CROSS_PREFIX}* arrays, "
                             "not one")
    (name,) = names
    rpath = fx.path.parent / f"{name[len(CROSS_PREFIX):]}.json"
    if not rpath.exists():
        raise _fail(fx.path, f"arrays.{name}: no fixture {rpath.name}")
    rows = load(rpath)
    S = fx.arrays[name]
    if rows.converted is None:
        raise _fail(fx.path, f"arrays.{name}: {rpath.name} has no converted "
                             "basis")
    if rows.provenance.cart != fx.provenance.cart:
        raise _fail(fx.path, f"arrays.{name}: {rpath.name} differs in cart")
    if S.shape != (rows.nao, fx.nao):
        raise _fail(fx.path, f"arrays.{name} has shape {S.shape}, expected "
                             f"({rows.nao}, {fx.nao})")
    return rows, S


def cross_fixtures() -> list[Path]:
    """Every tracked fixture carrying a cross-basis overlap, sorted."""
    return [p for p in tracked()
            if any(n.startswith(CROSS_PREFIX) for n in load(p).arrays)]


# --- the metric pieces (#411) -----------------------------------------------


@dataclass(frozen=True)
class MetricSet:
    """A metric set `metric_set` reassembled: the WHOLE matrices."""

    name: str
    pieces: tuple[Fixture, ...]  # in row order
    metric: np.ndarray  # libcint's int2c2e_cart, nao x nao, read-only
    ovlp: np.ndarray  # libcint's int1e_ovlp, nao x nao, read-only
    converted: Converted  # the aux basis (oracle.basis.from_fixture reads it)

    @property
    def path(self) -> Path:
        return self.pieces[0].path

    @property
    def nao(self) -> int:
        return self.pieces[0].nao

    @property
    def parent(self) -> str:
        assert self.pieces[0].metric is not None
        return self.pieces[0].metric.parent


# Provenance a piece may differ in from its siblings: when and where it was
# written, and the command naming it. Everything else is an input.
_PIECE_FREE = ("date", "host", "command")


def _inputs(fx: Fixture, record: str = "metric",
            ) -> tuple[tuple[str, Any], ...]:
    rec = getattr(fx, record)
    assert rec is not None
    pv = fx.provenance
    return (
        *((f"provenance.{k}", getattr(pv, k)) for k in PROVENANCE_KEYS
          if k not in _PIECE_FREE),
        ("atoms", fx.atoms), ("shells", fx.shells), ("nao", fx.nao),
        ("ao_labels", fx.ao_labels), ("ao_loc", fx.ao_loc),
        (f"{record}.parent", rec.parent),
        (f"{record}.converted", rec.converted),
    )


def metric_set(name: str, fixtures: Path = FIXTURES) -> MetricSet:
    """Load every `<name>_rows*` piece and reassemble the whole matrices.

    Refuses, naming the files: no piece; a piece of another set; two pieces
    whose rows overlap; a row gap (a missing piece) at the start, between
    two pieces or at the end; a piece whose inputs differ from the first's;
    and reassembled matrices that are not symmetric to 1e-14 of their
    largest element. Every input comparison is exact."""
    paths = sorted(fixtures.glob(f"{name}{ROWS_SEP}*.json"))
    if not paths:
        raise FixtureError(f"{fixtures}: no metric piece {name}{ROWS_SEP}*")
    pieces = []
    for p in paths:
        fx = load(p)
        if fx.metric is None or fx.metric.set != name:
            raise _fail(p, f"is not a piece of metric set {name!r}")
        pieces.append(fx)
    pieces.sort(key=lambda f: f.metric.rows)  # type: ignore[union-attr]
    first = pieces[0]
    ref = _inputs(first)
    for fx in pieces[1:]:
        for (field, a), (_, b) in zip(ref, _inputs(fx), strict=True):
            if a != b:
                raise FixtureError(f"{first.path} / {fx.path}: not one metric "
                                   f"set: {field} differ")
    end = 0
    for fx in pieces:
        r0, r1 = fx.metric.rows  # type: ignore[union-attr]
        if r0 < end:
            raise _fail(fx.path, f"rows [{r0}, {r1}) overlap a piece "
                                 f"ending at row {end}")
        if r0 > end:
            raise _fail(fx.path, f"rows [{end}, {r0}) of set {name!r} are "
                                 "in no piece (one is missing)")
        end = r1
    if end != first.nao:
        raise FixtureError(f"{fixtures}: rows [{end}, {first.nao}) of set "
                           f"{name!r} are in no piece (one is missing)")
    out = {}
    for array in METRIC_ARRAYS:
        a = np.vstack([fx.arrays[array] for fx in pieces])
        if not np.all(np.abs(a - a.T) <= 1e-14 * np.max(np.abs(a))):
            raise FixtureError(f"{first.path}: set {name!r}'s {array} is not "
                               "symmetric to 1e-14 of its largest element")
        a.setflags(write=False)
        out[array] = a
    assert first.metric is not None
    return MetricSet(name, tuple(pieces), out[METRIC_ARRAY],
                     out[AUX_OVLP_ARRAY], first.metric.converted)


def metric_sets(fixtures: Path = FIXTURES) -> list[str]:
    """Every tracked metric set's name, sorted."""
    names = {fx.metric.set for p in sorted(fixtures.glob("*.json"))
             if (fx := load(p)).metric is not None}
    return sorted(names)


# --- the three-center pieces (#442) ----------------------------------------


@dataclass(frozen=True)
class ThreeCenterSet:
    """A three-center set `three_center_set` reassembled: the WHOLE tensor."""

    name: str
    pieces: tuple[Fixture, ...]  # in column order
    parent: Fixture  # the bra side's AO fixture
    tensor: np.ndarray  # int3c2e_cart, (parent nao, parent nao, aux nao)
    converted: Converted  # the aux basis (oracle.basis.from_fixture reads it)

    @property
    def path(self) -> Path:
        return self.pieces[0].path

    @property
    def nao(self) -> int:
        """The aux basis's AO count."""
        return self.pieces[0].nao


def three_center_set(name: str, fixtures: Path = FIXTURES) -> ThreeCenterSet:
    """Load every `<name>_cols*` piece and reassemble the whole tensor
    (`assemble_three_center` refuses, naming the files)."""
    paths = sorted(fixtures.glob(f"{name}{COLS_SEP}*.json"))
    return assemble_three_center(name, [load(p) for p in paths], fixtures)


def assemble_three_center(name: str, pieces: list[Fixture],
                          fixtures: Path = FIXTURES) -> ThreeCenterSet:
    """Reassemble already-loaded pieces of set `name`, the parent read from
    `fixtures`. Refuses: no piece; a piece of another set; two pieces over
    the same columns (a duplicate) or overlapping ones; a column gap at the
    start, between two pieces or at the end; a piece whose inputs differ
    from the first's; a parent that is absent, has no converted basis, or
    differs from what the pieces carry (atoms, provenance.mol and cart, the
    array's AO width as its nao); and a tensor not symmetric in m, n to
    1e-14 of its largest element. Every input comparison is exact."""
    if not pieces:
        raise FixtureError(f"{fixtures}: no three-center piece "
                           f"{name}{COLS_SEP}*")
    for fx in pieces:
        if fx.three_center is None or fx.three_center.set != name:
            raise _fail(fx.path, f"is not a piece of three-center set "
                                 f"{name!r}")
    pieces = sorted(pieces, key=lambda f: f.three_center.cols)  # type: ignore[union-attr]
    first = pieces[0]
    ref = _inputs(first, "three_center")
    for fx in pieces[1:]:
        for (field, a), (_, b) in zip(ref, _inputs(fx, "three_center"),
                                      strict=True):
            if a != b:
                raise FixtureError(f"{first.path} / {fx.path}: not one "
                                   f"three-center set: {field} differ")
    end, prev = 0, None
    for fx in pieces:
        c0, c1 = fx.three_center.cols  # type: ignore[union-attr]
        if prev == (c0, c1):
            raise _fail(fx.path, f"cols [{c0}, {c1}) is a duplicate piece")
        if c0 < end:
            raise _fail(fx.path, f"cols [{c0}, {c1}) overlap a piece "
                                 f"ending at column {end}")
        if c0 > end:
            raise _fail(fx.path, f"cols [{end}, {c0}) of set {name!r} are "
                                 "in no piece (one is missing)")
        end, prev = c1, (c0, c1)
    if end != first.nao:
        raise FixtureError(f"{fixtures}: cols [{end}, {first.nao}) of set "
                           f"{name!r} are in no piece (one is missing)")
    tc = first.three_center
    assert tc is not None
    ppath = fixtures / f"{tc.parent}.json"
    if not ppath.exists():
        raise _fail(first.path, f"three_center.parent: no fixture "
                                f"{ppath.name}")
    parent = load(ppath)
    a = np.concatenate([fx.arrays[tc.array] for fx in pieces], axis=2)
    checks = (
        ("atoms", first.atoms, parent.atoms),
        ("provenance.mol", first.provenance.mol, parent.provenance.mol),
        ("provenance.cart", first.provenance.cart, parent.provenance.cart),
        ("nao", a.shape[0], parent.nao),
    )
    for field, mine, theirs in checks:
        if mine != theirs:
            raise FixtureError(f"{first.path} / {ppath}: not its parent: "
                               f"{field} differ")
    if parent.converted is None:
        raise _fail(first.path, f"three_center.parent {ppath.name} has no "
                                "converted basis")
    # libcint computes both triangles: symmetric to rounding, not bitwise
    if not np.all(np.abs(a - a.transpose(1, 0, 2))
                  <= 1e-14 * np.max(np.abs(a))):
        raise FixtureError(f"{first.path}: set {name!r}'s {tc.array} is not "
                           "symmetric in m, n to 1e-14 of its largest element")
    a.setflags(write=False)
    return ThreeCenterSet(name, tuple(pieces), parent, a, tc.converted)


def three_center_sets(fixtures: Path = FIXTURES) -> list[str]:
    """Every tracked three-center set's name, sorted."""
    names = {fx.three_center.set for p in sorted(fixtures.glob("*.json"))
             if (fx := load(p)).three_center is not None}
    return sorted(names)


# --- the Fock pair (#280) ----------------------------------------------------

J_SUFFIX, K_SUFFIX = "_j", "_k"
# G = J - K / N_SPIN for the total closed-shell density (felizjueves.ao.fock's
# derivation; PySCF's RHF vj - 0.5 vk). Not imported: this module stays numpy.
FOCK_EXCHANGE_FRACTION = 0.5


@dataclass(frozen=True)
class FockPair:
    """A `<name>_j` / `<name>_k` fixture pair that `check_pair` accepted."""

    name: str
    j: Fixture
    k: Fixture

    @property
    def density(self) -> np.ndarray:
        return self.j.density

    @property
    def vj(self) -> np.ndarray:
        return self.j.arrays[J_ARRAY]

    @property
    def vk(self) -> np.ndarray:
        return self.k.arrays[K_ARRAY]

    def two_electron(self) -> np.ndarray:
        """`G = vj - vk / 2`, PySCF's two-electron Fock matrix for the pair's
        (total, closed-shell) density. `F = hcore + G`."""
        g = self.vj - FOCK_EXCHANGE_FRACTION * self.vk
        g.setflags(write=False)
        return g


def _pair_fail(fj: Fixture, fk: Fixture, msg: str) -> FixtureError:
    return FixtureError(f"{fj.path} / {fk.path}: not a Fock pair: {msg}")


def check_pair(fj: Fixture, fk: Fixture) -> None:
    """Refuse, naming both files and the field, unless `fj` (carrying `vj`)
    and `fk` (carrying `vk`) were computed from the same inputs. Every
    comparison is exact (module docstring: THE FOCK PAIR)."""
    if J_ARRAY not in fj.arrays:
        raise _pair_fail(fj, fk, f"the first carries no {J_ARRAY}")
    if K_ARRAY not in fk.arrays:
        raise _pair_fail(fj, fk, f"the second carries no {K_ARRAY}")
    if fj.converted is None or fk.converted is None:
        raise _pair_fail(fj, fk, "both must record the converted basis")
    pj, pk = fj.provenance, fk.provenance
    fields = (  # the issue's five first, so the message names the first rot
        ("shells", fj.shells, fk.shells),
        ("ao_loc", fj.ao_loc, fk.ao_loc),
        ("converted.common_factor", fj.converted.common_factor,
         fk.converted.common_factor),
        ("provenance.mol", pj.mol, pk.mol),
        ("provenance.basis", pj.basis, pk.basis),
        ("provenance.cart", pj.cart, pk.cart),
        ("atoms", fj.atoms, fk.atoms),
        ("nao", fj.nao, fk.nao),
        ("ao_labels", fj.ao_labels, fk.ao_labels),
        ("converted.shells", fj.converted.shells, fk.converted.shells),
        ("density_recipe", fj.density_recipe, fk.density_recipe),
    )
    for name, a, b in fields[:3]:
        if a != b:
            raise _pair_fail(fj, fk, f"{name} differ")
    if not np.array_equal(fj.density, fk.density):
        raise _pair_fail(fj, fk, "density differ")
    for name, a, b in fields[3:]:
        if a != b:
            raise _pair_fail(fj, fk, f"{name} differ")
    shared = sorted(set(fj.arrays) & set(fk.arrays) - {"density"})
    for name in shared:
        if not np.array_equal(fj.arrays[name], fk.arrays[name]):
            raise _pair_fail(fj, fk, f"arrays.{name} differ")


def fock_pair(name: str, fixtures: Path = FIXTURES) -> FockPair:
    """Load `<name>_j` and `<name>_k` and check them as a pair."""
    fj = load(fixtures / f"{name}{J_SUFFIX}.json")
    fk = load(fixtures / f"{name}{K_SUFFIX}.json")
    check_pair(fj, fk)
    return FockPair(name, fj, fk)


def fock_pairs() -> list[str]:
    """Every tracked molecule with both a `_j` and a `_k` fixture, sorted.
    By file name only; `fock_pair` is what checks one."""
    stems = {p.stem for p in tracked()}
    return sorted(s[:-len(J_SUFFIX)] for s in stems
                  if s.endswith(J_SUFFIX)
                  and s[:-len(J_SUFFIX)] + K_SUFFIX in stems)


def main(argv: list[str] | None = None) -> int:
    """Load every tracked fixture; one line each. Exit 1 on the first bad one."""
    paths = [Path(p) for p in (argv or [])] or tracked()
    if not paths:
        print(f"no fixtures under {FIXTURES}")
        return 1
    for p in paths:
        try:
            f = load(p)
        except FixtureError as e:
            print(f"BAD {e}")
            return 1
        pv = f.provenance
        print(f"{f.name}: nao={f.nao}, generated {pv.date} on {pv.host} "
              f"(pyscf {pv.pyscf}, libcint {pv.libcint})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
