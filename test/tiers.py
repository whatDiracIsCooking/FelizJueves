"""The fast tier's angular-momentum cut -- one number, read by every test file.

    FAST_L_MAX                       -> 4
    is_fast(L)                       -> L <= FAST_L_MAX
    by_l(value, L, id=None)          -> pytest.param, `slow` when L > FAST_L_MAX
    split_by_l(values, l_of, ids=)   -> [by_l(v, l_of(v)) for v in values]

WHY. The fast tier is the pre-push gate (`FAST_TEST_ARGS`), and a gate that
costs minutes is one people bypass. Its time went almost entirely to classes
above L = 4: building, emitting, planning and integrating them (#419). So
any test that COMPUTES something for a class of total angular momentum
L > FAST_L_MAX -- a kernel built, emitted or interpreted, a plan, a DAG or a
search, a host integral -- is `slow`, and a parametrized test is split so its
L <= FAST_L_MAX cases stay in the fast tier. A test that only READS a stored
row, a key or a fixture's metadata naming such a class stays fast: it costs
nothing, and it keeps the stores' integrity on the gate.

WHAT L IS. A class's total angular momentum: `la + lb + lc + ld` for a
quartet (`Spec.total_am`, `build_pq_kernel(L)`'s L, an HGP/Fock class in AO
order), and `la + lb` for a pair kernel's class (`DQ`, `JAB`, `S`, `T`, `V`,
`SX`) -- except `Target.SCHWARZ`, whose pair `(la, lb)` runs the quartet
`(la lb|la lb)` at `2 (la + lb)`.

The `slow` tier still runs everything moved here; `kernel-tier.sh` and the
device tests are untouched by this cut.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest

FAST_L_MAX = 4


def is_fast(L: int) -> bool:
    return L <= FAST_L_MAX


def by_l(value: Any, L: int, id: str | None = None) -> Any:
    """`value` as a parameter, marked `slow` when its class's L is above the cut."""
    marks = () if is_fast(L) else (pytest.mark.slow,)
    return pytest.param(value, id=id, marks=marks)


def split_by_l(values: Iterable[Any], l_of: Callable[[Any], int],
               ids: Callable[[Any], str] | None = None) -> list[Any]:
    """Every value as `by_l(v, l_of(v))`, with `ids(v)` as its id when given."""
    return [by_l(v, l_of(v), ids(v) if ids else None) for v in values]
