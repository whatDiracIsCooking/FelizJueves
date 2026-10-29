"""The fast tier's angular-momentum cut -- one number, read by every test file.

    FAST_L_MAX                       -> 4
    is_fast(L)                       -> L <= FAST_L_MAX
    DEVICE_L_MAX                     -> 8
    by_l(value, L, id=None)          -> pytest.param, `slow` when L > FAST_L_MAX,
                                        and `high_l` too when L > DEVICE_L_MAX
    split_by_l(values, l_of, ids=)   -> [by_l(v, l_of(v)) for v in values]
    sample_compiles(items)           -> marks `slow` every compile-only case
                                        but the first of its kind

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

The `slow` tier still runs everything moved here.

THE DEVICE CAP. `DEVICE_L_MAX` is the second cut, the kernel tier's: a case
above it is also marked `high_l`, and both kernel-tier selections
(`KERNEL_TEST_ARGS`, `KERNEL_FULL_TEST_ARGS` in devtools/config.sh) exclude
that marker, so `--full` runs every `gpu` test with L <= 8. The cases above it
(the J subgrid's L = 10, 13 and 16 cells, the K subgrid's, #113's asymmetric
cells, the [p|q] ladder's L = 9..16) were where `--full` spent its hours: a
single L = 10 K unit sat in nvcc's front end for over an hour (#301). They
are kept, not deleted: `pytest -m "gpu and cuda and high_l"` still runs
them. Only the device selection reads the marker; a bare `pytest` and the
`slow` tier still collect every host case above the cap.

THE COMPILE SAMPLE. A compile-only case (`cuda` or `hip`, not `gpu`) proves
that one emitted text survives one device compiler, and a parametrized one
mostly repeats that proof one class higher: `(cuda or hip) and not gpu and
not slow` took 28m38s cold on a 4-core device-less box
(test/compile-only-timings.xml), most of it in classes the case before had
already compiled the same emitter at. So `sample_compiles` keeps, per test
function, the FIRST case of each kind -- its id up to the first digit, `(`
or `-` (`hj_0000` and `hj_1111` are one kind, `hk_0000` another; a bare
`[3]` or `[(ps|ss)-b1]` is the function's one kind) -- and marks the rest
`slow`. Every compile-only test function and every kind of unit it emits
still compiles under both compilers; the classes above each kind's first
(parameters are listed lowest first) run with the `slow` tier or a bare
`pytest`, and the kernel tier compiles every class it launches anyway.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

import pytest

FAST_L_MAX = 4
DEVICE_L_MAX = 8


def is_fast(L: int) -> bool:
    return L <= FAST_L_MAX


def by_l(value: Any, L: int, id: str | None = None) -> Any:
    """`value` as a parameter, marked `slow` when its class's L is above the
    fast cut and `high_l` when it is above the device cap."""
    marks = () if is_fast(L) else (pytest.mark.slow,)
    if L > DEVICE_L_MAX:
        marks = (*marks, pytest.mark.high_l)
    return pytest.param(value, id=id, marks=marks)


def split_by_l(values: Iterable[Any], l_of: Callable[[Any], int],
               ids: Callable[[Any], str] | None = None) -> list[Any]:
    """Every value as `by_l(v, l_of(v))`, with `ids(v)` as its id when given."""
    return [by_l(v, l_of(v), ids(v) if ids else None) for v in values]


def _kind(case_id: str) -> str:
    return re.match(r"[^\d(-]*", case_id).group()


def sample_compiles(items: Iterable[pytest.Item]) -> None:
    """Mark `slow` each compile-only case after the first of its kind in its
    test function (THE COMPILE SAMPLE above); `items` in collection order."""
    seen: set[tuple[str, str]] = set()
    for item in items:
        callspec = getattr(item, "callspec", None)
        if (callspec is None or item.get_closest_marker("gpu")
                or item.get_closest_marker("slow")
                or not (item.get_closest_marker("cuda")
                        or item.get_closest_marker("hip"))):
            continue
        key = (item.nodeid.partition("[")[0], _kind(callspec.id))
        if key in seen:
            item.add_marker(pytest.mark.slow)
        seen.add(key)
