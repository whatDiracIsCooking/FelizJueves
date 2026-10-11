"""The fixture schema, its loader, and the provenance record -- NUMPY ONLY.

Nothing here imports PySCF, so a fast-tier test can load and validate every
tracked fixture on a box where the `oracle` group was never installed. The
writer is `generate.py`; this module only defines what a fixture IS and refuses
anything that is not one, with a message naming the file.

THE SHAPE (schema 1), one JSON object per file under `fixtures/`:

    schema       1
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
                 `density` (nao x nao, exactly symmetric) is required
    density_recipe  how the recorded density was made
    converted    OPTIONAL. {common_factor, shells}: the basis in OUR convention
                 (`ao.shell.Shell`, one entry per contraction, in AO order) as
                 `oracle.basis` converts it -- coefficients are libcint's
                 normalised `_libcint_ctr_coeff` times `common_factor(l)`, to
                 multiply unnormalised Cartesian primitives -- and
                 `common_factor`, libcint's `common_fac` per l = 0.. as PySCF's
                 data yields it. Present only where a case records it; it lets
                 a later test build our shells with no PySCF. Cartesian only.

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

SCHEMA = 1
FIXTURES = Path(__file__).resolve().parent / "fixtures"

PROVENANCE_KEYS = (
    "pyscf", "libcint", "numpy", "date", "host", "command", "mol", "basis", "cart",
)
MOL_KEYS = ("atom", "unit", "charge", "spin")
TOP_KEYS = (
    "schema", "name", "description", "provenance", "atoms", "shells", "nao",
    "ao_labels", "ao_loc", "arrays", "density_recipe",
)
OPTIONAL_KEYS = ("converted",)
REQUIRED_ARRAYS = ("density",)


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

    @property
    def density(self) -> np.ndarray:
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


def _arrays(path: Path, raw: Any) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict):
        raise _fail(path, "arrays must be an object")
    missing = [k for k in REQUIRED_ARRAYS if k not in raw]
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
    if raw["schema"] != SCHEMA:
        raise _fail(path, f"schema {raw['schema']!r}, this loader reads {SCHEMA}")
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

    arrays = _arrays(path, raw["arrays"])
    d = arrays["density"]
    if d.shape != (nao, nao):
        raise _fail(path, f"density has shape {d.shape}, expected ({nao}, {nao})")
    if not np.array_equal(d, d.T):
        raise _fail(path, "density is not exactly symmetric")
    for k in ("description", "density_recipe"):
        if not isinstance(raw[k], str) or not raw[k]:
            raise _fail(path, f"{k} must be a non-empty string")

    converted = (_converted(path, raw["converted"], prov, len(atoms), shells,
                            nao)
                 if "converted" in raw else None)

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
