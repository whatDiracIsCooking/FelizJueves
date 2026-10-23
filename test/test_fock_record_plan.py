"""`fock_driver.record_plan` is `record` without the walk -- launch by launch.

`record` walks `launch_f`'s plan with `walk_f` and an injected `run` that
only records each call; `record_plan` packs the same calls straight from the
plan, because walk_f's partition of the unique quartets is ~1e9 Python set
members at polyglycine's size (the scheme A/B sweep, measure/
measure_scheme_ab.py). The two must hand the card the same launches: the
same kernel, grid and block, and every argument equal (`F`, the running
matrix `record` carries, aside). A `levers=` arm builds each launch's kernel
with them, and a plan whose launches name walk_f-staged levers is refused.

Fast tier: H2O/STO-3G and H2O/6-31G (s and p only), no compiler, no card.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from fock_driver import record, record_plan
from test_fock_build import BOUND, molecule

from felizjueves.eri.fock import kernel_f as KF
from felizjueves.eri.fock.launch_f import f_launches
from felizjueves.eri.fock.levers import block_fab


def _same(a, b) -> bool:
    if isinstance(a, tuple | list) or isinstance(b, tuple | list):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b and type(a) is type(b)


@pytest.mark.parametrize("name", ["h2o_sto3g", "h2o_631g"])
@pytest.mark.parametrize("tau, n", [(0.0, 1), (1e-3, 4)])
def test_record_plan_is_record(name: str, tau: float, n: int) -> None:
    basis, D, _ = molecule(name)
    plan = f_launches(basis, D, bound=BOUND, tau=tau, n=n)
    want, _ = record(plan, basis, D)
    got = record_plan(plan, basis, D)
    assert len(got) == len(want) > 0
    for g, w in zip(got, want, strict=True):
        assert g.kernel == w.kernel
        assert (g.grid, g.threads) == (w.grid, w.threads)
        assert set(g.args) == set(w.args) - {KF.FOCK}
        bad = [k for k in g.args if not _same(g.args[k], w.args[k])]
        assert not bad, f"{g.name}: arguments differ: {bad}"


def test_record_plan_levers() -> None:
    basis, D, _ = molecule("h2o_sto3g")
    plan = f_launches(basis, D, bound=BOUND, tau=0.0, n=1)
    got = record_plan(plan, basis, D, levers=(block_fab.NAME,))
    lit = record_plan(plan, basis, D)
    for L, g, w in zip(plan.launches, got, lit, strict=True):
        assert g.kernel == KF.build_f_kernel(
            replace(L.spec, levers=(block_fab.NAME,)))
        assert g.kernel != w.kernel
        assert g.args == w.args


def test_record_plan_refuses_walk_staged_levers() -> None:
    basis, D, _ = molecule("h2o_sto3g")
    plan = f_launches(basis, D, bound=BOUND, tau=0.0, n=1, scatter=True)
    with pytest.raises(ValueError, match="record_plan"):
        record_plan(plan, basis, D)
