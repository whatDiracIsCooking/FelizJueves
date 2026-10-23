"""What the launch plans share (#504, the `launch` half of #328).

    PairRec(i, j, prim, L, G, key)        one bra or ket index record
    sorted_desc(recs)                     descending by `key`, STABLE
    nonneg(g, what)                       a bound's value, checked
    admits(lp, lq, n_batches)             does the plan admit this split?
    batches_for(lp, lq, n_batches, *, strict, refused, resolve=None)

These were `launch.py`'s internals (`PairRec`, `_sorted_desc`, `_nonneg`,
`_admits`, `_batches_for`), borrowed by its siblings: `launch_k` (#204) took
all of them, `launch_kblock` (#481) `PairRec`, `launch_f` (#285) the sort and
the bound check, and `ao/staged` (#143) `PairRec`. They are moved here
unchanged in behaviour and made public, so no module imports a `_`-prefixed
name from `launch`. The J-only plan (`ClassLaunch`, `LaunchPlan`,
`class_spec`, `launches`, `walk`) stays in `launch.py`, which imports this
module; this module imports no launch plan.

THE SORT is `sorted(..., key=lambda r: r.key, reverse=True)`,
`ao.j._sorted_kets`' form exactly: STABLE, so ties keep pair order
(`key=lambda r: -r.key` is not the same on ties). Every plan writes its own
key into `PairRec.key` (J's `G * Dmax`, K's injected key, F's density
block), so the one sort serves them all.

THE BATCH ADMISSION is asked of `plan.pipeline.plan` at the canonical J
class `(lp, 0 | lq, 0)` -- `launch.class_spec`'s key, restated here because
`launch` imports this module (test/test_launch.py pins the two equal). The
admission depends on `(lp, lq, n_batches)` only, which is why a K plan may
ask it too (test/test_launch_k.py pins that the plan admits a K `Spec`
exactly when it admits the J class of the same split). `batches_for` is the
one place a count is chosen: `resolve(lp, lq)` replaces `n_batches` when
given (#145, injected, never imported); a refused count raises with the
class and the count (`strict=True`) or is recorded in `refused` and the
class built at ONE batch (`strict=False`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

from felizjueves.plan.pipeline import plan
from felizjueves.spec import Operator, Spec, Target

__all__ = ["PairRec", "admits", "batches_for", "nonneg", "sorted_desc"]


@dataclass(frozen=True)
class PairRec:
    """One bra or ket index record (`launch`'s module docstring, "THE KEYS").

    `i, j` the shell pair (`i >= j` in a J plan; any ORDERED pair in a K
    plan, `launch_k`, where `G` is K's record bound and `key` the injected
    ket key), `prim` its primitive pair's index in
    `ao.pairs.primitive_pairs(shells[i], shells[j])`, `L` the pair momentum,
    `G` the injected bound `I_pair`, `key` = `G * Dmax` for a ket, `G` for a
    bra.
    """

    i: int
    j: int
    prim: int
    L: int
    G: float
    key: float

    @property
    def shell_pair(self) -> tuple[int, int]:
        return (self.i, self.j)


def sorted_desc(recs: Sequence[PairRec]) -> list[PairRec]:
    """Descending by `key`, STABLE -- `ao.j._sorted_kets`' form exactly."""
    return sorted(recs, key=lambda r: r.key, reverse=True)


def nonneg(g: object, what: str) -> float:
    """A bound's value, refused unless a finite float >= 0 -- shared by the
    J, K and F plans, whose bounds have different signatures."""
    if not (isinstance(g, int | float) and math.isfinite(g) and g >= 0.0):
        raise ValueError(f"{what} = {g!r} is not a finite float >= 0")
    return float(g)


def _admission_spec(lp: int, lq: int, n_batches: int) -> Spec:
    # `launch.class_spec(lp, lq, n_batches)`, restated (module docstring).
    return Spec(Operator.COULOMB, 0, (lp, 0, lq, 0), Target.J, (), n_batches)


@cache
def admits(lp: int, lq: int, n_batches: int) -> bool:
    """Whether the plan admits class `(lp, lq)` at `n_batches` (#64)."""
    try:
        plan(_admission_spec(lp, lq, n_batches))
    except ValueError:
        return False
    return True


def batches_for(lp: int, lq: int, n_batches: int, *, strict: bool,
                refused: list[tuple[int, int, int]],
                resolve: Callable[[int, int], int] | None = None) -> int:
    """The batch count class `(lp, lq)` is built at (module docstring)."""
    if resolve is not None:
        n_batches = resolve(lp, lq)
    if admits(lp, lq, n_batches):
        return n_batches
    if strict:
        raise ValueError(f"the plan refuses n_batches = {n_batches} for class "
                         f"(lp, lq) = ({lp}, {lq})")
    refused.append((lp, lq, n_batches))
    return 1
