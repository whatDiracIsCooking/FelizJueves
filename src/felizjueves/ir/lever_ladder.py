"""The lever ladder as a registry: which flags a kernel-body lever gets (#339).

`python -m felizjueves.emit` writes ONE LADDER PER KERNEL FAMILY (#156). A
kernel-body lever (`levers.VOCABULARY`, #337) is one more axis over those
ladders: the same class, built with the lever pulled. Milestone 13 adds such
levers one issue at a time, and if each one edited the CLI's argument
handling, every lever issue would collide with every other in one file. So
the lever ladder is DATA, read by the CLI:

- THE TABLE IS `levers.VOCABULARY`, keyed by lever name. A lever issue adds
  its ONE entry there (`Lever.of(name, issue, what, legal)`) and nothing
  here or in `emit/__main__.py`: `ladders()` reads the vocabulary at call
  time, and the CLI builds one flag per `(lever, family)` from it.
- THE FAMILIES are `FAMILIES`, keyed by target slot: the targets a lever
  ladder can be written for, each with its flag suffix, the CLI path it
  belongs to and the argument it takes. A lever gets a family's flag exactly
  when the family's target is in its `legal` set -- so the legality table
  #337 already requires is the only declaration a lever makes.

      j  `--klever-<name>-pq L [L ...]`    the [p|q] ladder: `pq_spec(L)`
                                          with the lever, one batch
      k  `--klever-<name>-k L [L ...]`     the K ladder: `k_spec(L)` with the
                                          lever, one batch (K has no literal
                                          ladder in the CLI; only its lever
                                          arms are written here)
      f  `--klever-<name>-fock-l LMAX`     the F path: every class of
                                          `--fock-l LMAX` with the lever

  `j` and `k` are LADDER flags: they ADD units beside whichever J ladders
  the bare/`--L`/`--pair-l` selection writes, as `--one-e-l` does. `f` is a
  FOCK flag: it belongs to the separate F path, beside `--fock-l` /
  `--fock-class` (#340 declares F's levers). A lever ladder is NEVER in the
  no-flag default, so the 37 J-chain units and their pinned `source_sha256`
  (data/measured/kernels.json) never move.

- A LEVER UNIT'S NAME is the literal kernel's name with `_lv_<lever>`
  appended per kernel-body lever (`unit_name`; a lever name's
  non-identifier characters become `_`), so a lever unit and its literal
  twin can share an output directory and a binary. The measured driver
  names a kernel-body lever row's units the same way, so a store row's
  `source_sha256` is the emitted file's.

A leaf beside `levers`: it imports nothing else from the package, and names
no builder -- the CLI maps each family to its builder.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from felizjueves import levers as _levers

__all__ = [
    "FAMILIES",
    "FOCK",
    "LADDER",
    "PREFIX",
    "Family",
    "LadderFlag",
    "flags",
    "ladders",
    "unit_name",
]

LADDER, FOCK = "ladder", "fock"  # the CLI path a family's flag belongs to
PREFIX = "--klever-"  # every lever flag starts so: none collides with a ladder's


@dataclass(frozen=True)
class Family:
    """One target a lever ladder is written for."""

    slot: str  # the target's key slot (`levers.TARGET_SLOTS`)
    suffix: str  # the flag is `--klever-<name>-<suffix>`
    path: str  # LADDER or FOCK
    metavar: str  # what the flag takes: "L" (a list) or "LMAX" (one)
    what: str


FAMILIES: Mapping[str, Family] = MappingProxyType({
    "j": Family("j", "pq", LADDER, "L",
                "the [p|q] ladder: pq_spec(L) with the lever, one batch"),
    "k": Family("k", "k", LADDER, "L",
                "the K ladder: k_spec(L) with the lever, one batch"),
    "f": Family("f", "fock-l", FOCK, "LMAX",
                "the F path: every --fock-l LMAX class with the lever"),
})


@dataclass(frozen=True)
class LadderFlag:
    """One `(lever, family)` flag of the CLI."""

    lever: str
    family: Family

    @property
    def flag(self) -> str:
        return f"{PREFIX}{self.lever}-{self.family.suffix}"

    @property
    def dest(self) -> str:
        return "lever__" + _ident(self.lever) + "__" + self.family.slot


def _ident(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name)


def ladders() -> dict[str, tuple[str, ...]]:
    """`{lever: the family slots it ladders on}`, for every kernel-body lever
    in `levers.VOCABULARY` (read now, not at import), in `FAMILIES` order.
    A lever legal on no family gets no flag; two levers whose names spell
    the same identifier are refused (their units would collide)."""
    out: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}
    for name in sorted(_levers.VOCABULARY):
        lever = _levers.VOCABULARY[name]
        ident = _ident(name)
        if ident in seen:
            raise ValueError(f"levers {seen[ident]!r} and {name!r} spell one "
                             f"unit suffix _lv_{ident}")
        seen[ident] = name
        fams = tuple(s for s in FAMILIES if s in lever.legal)
        if fams:
            out[name] = fams
    return out


def flags() -> tuple[LadderFlag, ...]:
    """Every lever flag the CLI offers, lever-major."""
    return tuple(LadderFlag(name, FAMILIES[s])
                 for name, fams in ladders().items() for s in fams)


def unit_name(name: str, kernel_body_levers: Iterable[str]) -> str:
    """The unit name of a kernel built with `kernel_body_levers` (in key
    order): `name` itself when there are none -- so no literal unit, and no
    plan or lowering arm, is renamed -- else `name_lv_<lever>...`."""
    return name + "".join(f"_lv_{_ident(lv)}" for lv in kernel_body_levers)
