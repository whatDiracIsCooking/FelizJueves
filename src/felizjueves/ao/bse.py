"""Basis Set Exchange JSON, read as the file states it (#650).

`data/basis/` and `data/aux_basis/` hold FreeBreakfast's basis sets verbatim,
in the BSE's JSON schema (`molssi_bse_schema`), so a test here can run on the
same sets as the consumer does. This module READS them and does nothing else:
no normalisation, no centring, no change of convention. A `Contraction` holds
the file's exponents and coefficients exactly, and mapping them onto
`ao.shell`'s convention is the test oracle's job (`test/oracle/basis.py`'s
`from_bse`), as `ao/shell.py` says every such mapping is.

WHAT THE READER DOES TO THE FILE'S LAYOUT, and nothing more:

- a fused shell (`"angular_momentum": [0, 1]`, Pople's SP) is split into one
  `Contraction` per angular momentum, each with its own row of coefficients,
  in the file's order;
- a general contraction (one angular momentum, several coefficient rows, as
  cc-pVDZ's) stays ONE `Contraction` with several columns -- zero entries kept,
  since dropping a primitive from one column but not another is a choice;
- `function_type` is recorded, not acted on. This tree is Cartesian
  throughout, so a `gto_spherical` d read here is six functions, not five --
  exactly what the libcint fixtures' `cart=True` does with the same tag. Both
  FreeBreakfast's relabellings (6-31G from K, 6-31G* and ** from Li:
  `data/basis/README.md`) are therefore invisible to a Cartesian reader.

An element's contractions are returned in file order. libcint's order (by
angular momentum, file order within one) is the oracle's to impose.

`find(name)` matches the document's `name` field, not the file name, with
`-`, `_` and spaces dropped and case ignored -- FreeBreakfast's rule, so
`cc-pVDZ`, `ccpvdz` and `CC_PVDZ` are one set and `6-31G*` is
`6-31gstar.json`.

Plain tuples and floats, like `ao.shell`; `json` from the standard library.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

# The repository's data/, from src/felizjueves/ao/bse.py.
DATA = Path(__file__).resolve().parents[3] / "data"
ORBITAL = DATA / "basis"
AUXILIARY = DATA / "aux_basis"

FUNCTION_TYPES = frozenset({"gto", "gto_spherical", "gto_cartesian"})

# Where each directory records its files' origin and bytes; not a basis.
PROVENANCE = "provenance.json"


def files(directory: Path) -> tuple[Path, ...]:
    """Every basis document in `directory`, sorted: its JSON but the
    provenance record."""
    return tuple(p for p in sorted(Path(directory).glob("*.json"))
                 if p.name != PROVENANCE)


def _floats(where: str, xs: object) -> tuple[float, ...]:
    if not isinstance(xs, list) or not xs:
        raise ValueError(f"{where}: {xs!r} is not a non-empty list")
    try:
        out = tuple(float(x) for x in xs)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {xs!r} is not a list of numbers") from None
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{where}: {xs!r} has a non-finite entry")
    return out


@dataclass(frozen=True)
class Contraction:
    """One angular momentum of one BSE shell, as the file states it.

    `coefficients[c][p]` is contraction c's coefficient on primitive p -- the
    BSE's row layout, one row per contracted function.
    """

    l: int  # noqa: E741 -- the name the issue and the literature use
    exponents: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    function_type: str

    @property
    def n_contracted(self) -> int:
        return len(self.coefficients)


@dataclass(frozen=True)
class BasisSet:
    """A BSE document: its `name`, and each element's contractions."""

    name: str
    path: Path
    elements: Mapping[int, tuple[Contraction, ...]]

    def contractions(self, z: int) -> tuple[Contraction, ...]:
        """Element `z`'s contractions in file order, SP split."""
        try:
            return self.elements[z]
        except KeyError:
            raise KeyError(f"{self.name} ({self.path.name}) defines no "
                           f"element Z = {z}") from None


def _shell(where: str, raw: object) -> tuple[Contraction, ...]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a shell is not an object")
    ftype = raw.get("function_type")
    if ftype not in FUNCTION_TYPES:
        raise ValueError(f"{where}: function_type {ftype!r} is not one of "
                         f"{sorted(FUNCTION_TYPES)}")
    ls = raw.get("angular_momentum")
    if (not isinstance(ls, list) or not ls
            or not all(type(l) is int and l >= 0 for l in ls)):  # noqa: E741
        raise ValueError(f"{where}: angular_momentum {ls!r}")
    exps = _floats(f"{where} exponents", raw.get("exponents"))
    if not all(a > 0 for a in exps):
        raise ValueError(f"{where}: exponents {exps!r} must all be > 0")
    rows = raw.get("coefficients")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{where}: no coefficients")
    coeffs = tuple(_floats(f"{where} coefficients[{i}]", r)
                   for i, r in enumerate(rows))
    if any(len(r) != len(exps) for r in coeffs):
        raise ValueError(f"{where}: {len(exps)} exponents, but a coefficient "
                         "row of another length")
    if len(ls) == 1:
        return (Contraction(ls[0], exps, coeffs, ftype),)
    # A fused shell: one row per angular momentum, nothing general about it.
    if len(coeffs) != len(ls):
        raise ValueError(f"{where}: angular momenta {ls} but {len(coeffs)} "
                         "coefficient rows")
    return tuple(Contraction(lq, exps, (r,), ftype)
                 for lq, r in zip(ls, coeffs, strict=True))


def parse(doc: object, path: Path) -> BasisSet:
    """The `BasisSet` a decoded BSE document describes."""
    if not isinstance(doc, dict) or "molssi_bse_schema" not in doc:
        raise ValueError(f"{path}: not a BSE JSON document")
    name = doc.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: no name")
    raw = doc.get("elements")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: no elements")
    elements: dict[int, tuple[Contraction, ...]] = {}
    for key, element in raw.items():
        z = int(key)
        shells = element.get("electron_shells") if isinstance(element, dict) else None
        if not isinstance(shells, list) or not shells:
            raise ValueError(f"{path}: Z = {z} has no electron_shells")
        elements[z] = tuple(c for i, s in enumerate(shells)
                            for c in _shell(f"{path.name} Z={z} shell {i}", s))
    return BasisSet(name, path, MappingProxyType(dict(sorted(elements.items()))))


def load(path: str | Path) -> BasisSet:
    """Read one BSE JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return parse(json.load(f), path)


def canonical(name: str) -> str:
    """FreeBreakfast's name rule: `-`, `_` and spaces dropped, case folded."""
    return "".join(ch for ch in name if ch not in "-_ ").casefold()


@cache
def _index(directory: Path) -> Mapping[str, Path]:
    out: dict[str, Path] = {}
    for path in files(directory):
        with path.open(encoding="utf-8") as f:
            name = json.load(f).get("name")
        if not isinstance(name, str):
            raise ValueError(f"{path}: no name")
        key = canonical(name)
        if key in out:
            raise ValueError(f"{path} and {out[key]} are both {name!r}")
        out[key] = path
    return MappingProxyType(out)


def find(name: str, directory: Path = ORBITAL) -> Path:
    """The file in `directory` whose document is named `name`."""
    index = _index(Path(directory))
    try:
        return index[canonical(name)]
    except KeyError:
        raise KeyError(f"no basis named {name!r} in {directory}; there are "
                       f"{sorted(index)}") from None


@cache
def named(name: str, directory: Path = ORBITAL) -> BasisSet:
    """`load(find(name, directory))`, cached: the documents are read-only."""
    return load(find(name, directory))


__all__ = ["AUXILIARY", "DATA", "FUNCTION_TYPES", "ORBITAL", "PROVENANCE",
           "BasisSet", "Contraction", "canonical", "files", "find", "load",
           "named", "parse"]
