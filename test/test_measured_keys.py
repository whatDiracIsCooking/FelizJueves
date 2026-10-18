"""#150 / #339: a lever-keyed row, re-derived through the store's one
recording path, and the independent key parser every re-derivation uses.

`Spec.key()` is the join key of the measured store; `_derive` rebuilds a key
from scratch -- the class, batch count and lever read back by a parser that
does not call `Spec.key()`, the card and compiler from the row's own fields
-- through the same constructors the driver uses (`MeasuredKey`,
`asym.AsymKey`). Here it is held on a synthetic store only. Re-deriving every
key of the TRACKED data/measured/kernels.json, and that no new target's key
joins a tracked row, are checks of a measurement artefact:
measure/measure_measured_keys.py (`devtools/measure.sh check`), which imports
`_derive` from here.
"""

from __future__ import annotations

import re

from felizjueves.asym import ASYM, ASYM_CELLS, AsymCell, AsymKey
from felizjueves.device import MeasuredKey, by_name
from felizjueves.kernel import k_spec, pq_spec
from felizjueves.measured_store import (
    split_key,
    target_of,
)
from felizjueves.spec import Operator, Spec, Target

# Independent of Spec.key(): the quartet shape as the store spells it.
_SPEC_KEY = re.compile(
    r"coulomb/d0/\((\d+),(\d+)\|(\d+),(\d+)\)/b(\d+)/j/([a-z+-]+)\Z"
)


def _derive(key: str, arch: str, toolchain: str) -> str:
    """Rebuild one row's key from its parts, the driver's way."""
    spec_key, _, _ = split_key(key)
    m = _SPEC_KEY.fullmatch(spec_key)
    assert m, f"{key!r} is not a quartet J key"
    l1, l2, l3, l4, n, lever = m.groups()
    am = tuple(map(int, (l1, l2, l3, l4)))
    L, n = sum(am), int(n)
    # Every row's class is the default class of its L.
    assert pq_spec(L).am == am, key
    if lever == ASYM:
        cell = AsymCell(L, n)
        assert cell in ASYM_CELLS, key
        return AsymKey(cell, by_name(arch), toolchain).key()
    levers = () if lever == "literal" else (lever,)
    spec = pq_spec(L, levers=levers, n_batches=n)
    assert spec.target is Target.J and len(spec.am) == 4
    return MeasuredKey(spec, by_name(arch), toolchain).key()


# --- #339: a lever-keyed row, through the store's one recording path --------


def test_a_lever_keyed_row_round_trips_and_rederives(monkeypatch,
                                                      tmp_path) -> None:
    """A kernel-body lever's row is keyed by its full lever-carrying
    `Spec.key()` -- the lever in the last segment, as every arm before it --
    and round-trips through `merge`, `save` and `load` byte for byte. Its key
    re-derives through the same independent parser and constructors as the
    tracked rows (`_derive`), and no tracked key moves (the tests above)."""
    import json

    from felizjueves import levers, measured_store

    monkeypatch.setitem(levers.VOCABULARY, "probe", levers.Lever.of(
        "probe", 339, "the round trip's witness", {"j", "k", "f"}))
    prov = dict(toolchain="nvcc 13.0", arch="sm_86", host="h", gpu="g",
                date="2026-10-15")
    cells = (measured_store.SUBGRID[5], pq_spec(9), k_spec(7, n_batches=3),
             Spec(Operator.COULOMB, 0, (1, 1, 0, 1), Target.F))
    store = {}
    for c in cells:
        key = MeasuredKey(measured_store.lever_arm(c, "probe"),
                          by_name("sm_86"), "nvcc 13.0")
        store = measured_store.merge(store, key, measured_store.Measured(
            registers=64, spill_store=0, spill_load=0, **prov))
    keys = sorted(store)
    assert all(split_key(k)[0].endswith("/probe") for k in keys)
    j_key = MeasuredKey(pq_spec(9, levers=("probe",)), by_name("sm_86"),
                        "nvcc 13.0").key()
    assert j_key == "coulomb/d0/(5,0|4,0)/b1/j/probe@sm_86/nvcc 13.0"
    assert _derive(j_key, "sm_86", "nvcc 13.0") == j_key
    p, q = tmp_path / "a.json", tmp_path / "b.json"
    measured_store.save(store, p)
    back = measured_store.load(p)
    assert back == store and sorted(back) == keys
    assert {r.arm_kind for r in back.values()} == {measured_store.KERNEL}
    assert {target_of(k).value for k in keys} == {"j", "k", "f"}
    measured_store.save(back, q)
    assert q.read_bytes() == p.read_bytes()
    assert set(json.loads(p.read_text())["rows"]) == set(keys)
