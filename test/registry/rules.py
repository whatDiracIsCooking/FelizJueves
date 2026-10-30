"""The registry's record types: a case, its key, and the three rules a device
test function resolves by (`__init__.py`'s docstring has the whole story)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .cost import Cost

ENGINES = frozenset({"mmd", "hgp", "rys"})
TIERS = frozenset({"interpreted", "compile", "gpu"})
# The two cards on this box, by the backend that drives each.
BACKEND_CARD = {"cuda": "rtx3080", "hip": "rx9060xt"}
CARDS = frozenset(BACKEND_CARD.values())


class Key(NamedTuple):
    engine: str
    quantity: str
    deriv: int
    L: int


@dataclass(frozen=True)
class Case:
    engine: str
    quantity: str
    deriv: int
    L: int
    cards: frozenset[str]
    tiers: frozenset[str]
    # Seconds per tier and backend, from the cost store (#652, cost.py);
    # None only for a case the store does not cover yet.
    cost_s: Cost | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"engine {self.engine!r} not in {sorted(ENGINES)}")
        if not self.tiers <= TIERS:
            raise ValueError(f"tiers {sorted(self.tiers)} not in {sorted(TIERS)}")
        if not self.cards <= CARDS:
            raise ValueError(f"cards {sorted(self.cards)} not in {sorted(CARDS)}")
        if self.cards and "gpu" not in self.tiers:
            raise ValueError(f"{self.key}: ran on {sorted(self.cards)} "
                             "with no gpu tier")

    @property
    def key(self) -> Key:
        return Key(self.engine, self.quantity, self.deriv, self.L)


def ladder(engine: str, quantity: str, deriv: int, *,
           compile: Iterable[int] = (), gpu: Iterable[int] = (),
           interpreted: Iterable[int] | None = None,
           cards: dict[str, Iterable[int]] | None = None) -> list[Case]:
    """One case per L of one (engine, quantity, deriv): `compile` and `gpu`
    the Ls with a test of that tier, `interpreted` those interpreted on the
    host (every L of the other two, by default), and `cards` each card's Ls
    it has run on."""
    compile, gpu = set(compile), set(gpu)
    interp = compile | gpu if interpreted is None else set(interpreted)
    ran = {card: set(Ls) for card, Ls in (cards or {}).items()}
    return [Case(engine, quantity, deriv, L,
                 frozenset(c for c, Ls in ran.items() if L in Ls),
                 frozenset(t for t, Ls in (("interpreted", interp),
                                           ("compile", compile), ("gpu", gpu))
                           if L in Ls))
            for L in sorted(compile | gpu | interp)]


@dataclass(frozen=True)
class Allow:
    """Covers no case, and says why: an allowlist entry."""

    reason: str


@dataclass(frozen=True)
class Classes:
    """One class per collected case, read from its parameters; the fields are
    the default when the parameters do not name them. `also` names further
    quantities the same case covers at the same L (a pair program that runs
    both `dq` and `jab`). `otherwise` is what a case whose parameters name
    no class is -- a function that mixes class cells with molecule cells;
    None makes such a case an error."""

    engine: str
    quantity: str
    deriv: int = 0
    also: tuple[str, ...] = ()
    otherwise: Allow | None = None


@dataclass(frozen=True)
class Span:
    """Every collected case covers each of these classes."""

    keys: tuple[Key, ...]

    def __add__(self, other: Span) -> Span:
        return Span(self.keys + other.keys)


def span(engine: str, quantity: str, deriv: int, Ls: Iterable[int]) -> Span:
    return Span(tuple(Key(engine, quantity, deriv, L) for L in Ls))


@dataclass(frozen=True)
class ById:
    """A rule per case, chosen by the case's id; an id it does not name is
    an error, so a new case cannot slip past the registry."""

    by_id: dict[str, Classes | Span | Allow]


Rule = Classes | Span | Allow | ById

# The allowlist's recurring reasons.
MOLECULE = ("a whole molecule's build: it spans every class the basis "
            "carries, so it is no one L")
PROGRAM = ("one program over a basis's classes, compiled or run as a unit: "
           "no one L")
MACHINERY = ("the generator's gpu-backend machinery (linking, ABI, "
             "wavefront width), not one family's class")
