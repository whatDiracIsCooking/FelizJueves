"""Dispatch: the batch count of a class, chosen from measured wall-clock (#145).

    choose(lp, lq, arch, input, *, store=None)   -> Choice
    Resolver(arch, input, *, store=None)         (lp, lq) -> n_batches
    table(arch, input, *, store=None)            {(lp, lq): Choice}, measured
    INPUTS                                       ("wide", "near")

`launch.launches(..., resolve=Resolver(arch, input))` builds each class at
the count chosen here; `launch` never imports this module (the policy is
injected, test/test_layering.py). `choose` is the one-cell entry point, for a
test or a device phase that wants the dispatched count of one class (#144).

NOT PARITY. The predecessor's `jgen` chose its counts its own way; this is
this repo's measured choice on THIS box's two cards, from #93's timings of
`measured_store.SUBGRID`, and claims nothing about `jgen` or about
`[JEngine, v2, §7]`'s bolded counts (their A100).

MEASURED ROWS ONLY. The choice is `batchreport.recommend`'s best median --
the rule `data/measured/batch_recommendation.json` records, recomputed from
the `Measured` rows of the store (`measured_store.load()` by default), never
from the file and never from a model: no `lower.KernelCost`, no byte or
FLOP proxy (test/test_layering.py holds this module off `lower`,
`kernelrank` and `plan`). #54's limits record rows where the model picked
and the hardware disagreed; the clock arbitrates.

THE INPUT AXIS IS EXPLICIT. The sweep timed a wide-T input and a near-T one,
and they disagree: wide-T picks 1 everywhere, near-T splits from L = 13, and
at L = 16 the two cards pick different counts. `input` is therefore a
required argument, "wide" or "near", with no default, and the two are never
averaged or merged: which Boys regime a molecule's classes sit in is the
caller's to say.

THE RULE, per (arch, class, input):

- the class is MEASURED iff it is the split of a SUBGRID cell -- `pq_spec(L)`
  measures `(ceil(L/2), floor(L/2))` only. Another split of the same L (its
  bra and ket do different work) is NOT read off that row: it has no row.
- a measured class gets the count with the lowest median over the subgrid's
  counts; a TIE goes to the fewer batches (fewer launches, no atomics) --
  `batchreport.Recommendation`'s order, `(median, n_batches)`. Its IQR,
  runner-up, margin, `resolved` and `within_spread` come with it: a choice
  inside its spread is still a choice, and says so.
- a class with no row gets 1 -- `Spec.n_batches`'s default, the generator's
  own -- with `measured = False` and the reason.
- a measured class whose cells are partly untimed is REFUSED (KeyError,
  `batchreport.recommend`'s): a best over an incomplete set is not the best.

REFUSALS, the `Arch.ridge` shape: an `Arch` without both measured ceilings
(`0.0` = not characterised) gets no choice -- `device.Uncharacterised` is
raised, never a default of 1. Nor does a characterised card with no timed
row at all: a card nobody timed is not characterised for dispatch either,
and falling back to 1 on every class would pass for a choice.

The chosen count is still ASKED OF THE PLAN by `launch` (#64's admission),
exactly as a fixed count is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from felizjueves import batchreport
from felizjueves import measured_store as measured
from felizjueves.device import Arch, Uncharacterised, require_characterised

__all__ = ["FALLBACK", "INPUTS", "Choice", "Resolver", "choose",
           "measured_classes", "table"]

INPUTS = batchreport.INPUTS
FALLBACK = 1  # Spec.n_batches' default: the count a class with no row gets

Store = Mapping[str, measured.Measured]


def measured_classes() -> dict[tuple[int, int], int]:
    """`{(lp, lq): L}` for every class a SUBGRID cell measures."""
    out = {}
    for s in measured.SUBGRID:
        l1, l2, l3, l4 = s.am
        out[(l1 + l2, l3 + l4)] = s.total_am
    return out


@dataclass(frozen=True)
class Choice:
    """The dispatched count of class `(lp, lq)` on `arch` for `input`, with
    its evidence. `measured = False` is the fallback: no row, `reason` says
    so, and every timing field is None."""

    lp: int
    lq: int
    arch: str
    input: str
    n_batches: int
    measured: bool
    reason: str
    median_ns: int | None = None
    iqr_ns: int | None = None
    repeats: int | None = None
    runner_up: int | None = None
    margin_ns: int | None = None
    resolved: bool | None = None
    within_spread: tuple[int, ...] = ()


def _input(which: object) -> str:
    if which not in INPUTS:
        raise ValueError(f"input {which!r} is not one of {INPUTS}: the "
                         "wide-T and near-T timings disagree, so the caller "
                         "names one -- they are never averaged")
    return str(which)


def _store(store: Store | None) -> Store:
    return measured.load() if store is None else store


def _timed_on(store: Store, arch: Arch, L: int) -> bool:
    return any(measured.rows(spec=s, arch=arch.name, store=store)
               for s in measured.SUBGRID if s.total_am == L)


def _require_timed(store: Store, arch: Arch) -> None:
    require_characterised(arch)
    if not any(_timed_on(store, arch, L) for L in measured.SUBGRID_LS):
        raise Uncharacterised(
            f"{arch.name} has no wall-clock row in the store: a card nobody "
            "timed gets no dispatch choice (time measured_store.SUBGRID on "
            "it: devtools/measure.sh wallclock)")


def _choose(lp: int, lq: int, arch: Arch, which: str, store: Store) -> Choice:
    L = measured_classes().get((lp, lq))
    if L is None:
        return Choice(lp, lq, arch.name, which, FALLBACK, False,
                      f"class ({lp}, {lq}) is not a measured split "
                      f"(measured: {sorted(measured_classes())}); "
                      f"n_batches = {FALLBACK}")
    if not _timed_on(store, arch, L):
        return Choice(lp, lq, arch.name, which, FALLBACK, False,
                      f"class ({lp}, {lq}) has no row on {arch.name}; "
                      f"n_batches = {FALLBACK}")
    r = batchreport.recommend(store, arch.name, L, which)
    b = r.best
    return Choice(lp, lq, arch.name, which, b.n_batches, True,
                  f"best median of {[t.n_batches for t in r.timings]} on "
                  f"{which}-T ({arch.name}, L = {L})",
                  b.median_ns, b.iqr_ns, b.repeats, r.runner_up.n_batches,
                  r.margin_ns, r.resolved, r.within_spread)


def choose(lp: int, lq: int, arch: Arch, input: str, *,
           store: Store | None = None) -> Choice:
    """The dispatched count of class `(lp, lq)` on `arch` for the named
    `input` ("wide" or "near"), with its evidence (module docstring).
    Raises `device.Uncharacterised` for a card that gets no choice."""
    which = _input(input)
    store = _store(store)
    _require_timed(store, arch)
    return _choose(lp, lq, arch, which, store)


def table(arch: Arch, input: str, *,
          store: Store | None = None) -> dict[tuple[int, int], Choice]:
    """`choose` of every measured class, by `(lp, lq)`."""
    which = _input(input)
    store = _store(store)
    _require_timed(store, arch)
    return {c: _choose(*c, arch, which, store) for c in measured_classes()}


class Resolver:
    """`launch.launches`' `resolve=`: `(lp, lq) -> n_batches` on one card and
    one input, the store read once. The refusals are raised HERE, at
    construction, so an uncharacterised card never reaches a plan.
    `choice(lp, lq)` is the evidence behind a count."""

    def __init__(self, arch: Arch, input: str, *,
                 store: Store | None = None) -> None:
        self.arch = arch
        self.input = _input(input)
        self._store = _store(store)
        _require_timed(self._store, arch)
        self._memo: dict[tuple[int, int], Choice] = {}

    def choice(self, lp: int, lq: int) -> Choice:
        if (lp, lq) not in self._memo:
            self._memo[(lp, lq)] = _choose(lp, lq, self.arch, self.input,
                                           self._store)
        return self._memo[(lp, lq)]

    def __call__(self, lp: int, lq: int) -> int:
        return self.choice(lp, lq).n_batches

    def __repr__(self) -> str:
        return f"Resolver({self.arch.name}, {self.input!r})"
