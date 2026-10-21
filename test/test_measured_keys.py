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
`_derive` from here. One read of the tracked keys stays here (#401): that no
stored key carries an engine suffix -- a read of keys, not a re-derivation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re

from felizjueves.asym import ASYM, ASYM_CELLS, AsymCell, AsymKey
from felizjueves.device import MeasuredKey, by_name
from felizjueves.kernel import k_spec, pq_spec
from felizjueves.measured_store import (
    engine_of,
    split_key,
    target_of,
)
from felizjueves.spec import NATIVE_ENGINE, Engine, Operator, Spec, Target

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


# --- #441: Target.ERI3 widens the enum and `am` without moving a J key -----

# Every MeasuredKey a J row of the store can carry -- the default class of
# each L = 0..16 at every legal batch count, literal and the two older
# arms, on both cards -- spelled by the code before #441 and pinned here.
# The tracked rows themselves are re-derived byte for byte by
# measure/measure_measured_keys.py (`devtools/measure.sh check`).
_CARDS = (("sm_86", "nvcc 13.0"), ("gfx1200", "hipcc / ROCm 7.2.4"))
J_KEYS_BEFORE_441 = (
    510, "227c0b260e0ff87b157e2ebb4a883a6c2060c2ffdf81e2a197eb70d2e6694cf5")


def _j_keys() -> list[str]:
    keys = []
    for L in range(17):
        for n in (1, 3, 6, 9, 12):
            for lv in ((), ("recycle",), ("no-sieve",)):
                try:
                    spec = pq_spec(L, levers=lv, n_batches=n)
                except ValueError:
                    continue
                keys += [MeasuredKey(spec, by_name(a), tc).key()
                         for a, tc in _CARDS]
    return keys


def test_target_eri3_moves_no_j_key() -> None:
    keys = _j_keys()
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    assert (len(keys), digest) == J_KEYS_BEFORE_441
    assert all(target_of(k) is Target.J for k in keys)


def test_an_eri3_key_is_never_read_as_a_j_row() -> None:
    # A triple key has one shell after `|`: the J parser rejects it, it
    # reads back as `Target.ERI3` and its parts, and no J key equals one.
    j = set(_j_keys())
    for am in itertools.product(range(5), repeat=3):
        spec = Spec(Operator.COULOMB, 0, am, Target.ERI3)
        assert not _SPEC_KEY.fullmatch(spec.key())
        for a, tc in _CARDS:
            key = MeasuredKey(spec, by_name(a), tc).key()
            assert key not in j
            assert split_key(key) == (spec.key(), a, tc)
            assert target_of(key) is Target.ERI3
    assert MeasuredKey(Spec(Operator.COULOMB, 0, (2, 1, 3), Target.ERI3),
                       by_name("sm_86"), "nvcc 13.0").key() == (
        "coulomb/d0/(2,1|3)/b1/eri3/literal@sm_86/nvcc 13.0")


# --- #401: Spec.engine moves no key ------------------------------------------


def test_spec_engine_moves_no_j_key() -> None:
    # The same 510 keys and digest as before #401: a native engine is never
    # spelled, so `engine=None` and the explicit native engine key alike.
    keys = _j_keys()
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    assert (len(keys), digest) == J_KEYS_BEFORE_441
    assert all(engine_of(k) is Engine.MMD for k in keys)


def test_no_stored_key_carries_an_engine_suffix() -> None:
    # The tracked store's keys, read raw (no loader, no rewrite): every one's
    # target slot is a bare target, its engine the native one.
    from felizjueves.measured_store import STORE
    keys = list(json.loads(STORE.read_text())["rows"])
    assert keys
    for key in keys:
        slot = split_key(key)[0].split("/")[4]
        assert "." not in slot, key
        assert engine_of(key) is NATIVE_ENGINE[target_of(key)], key


def test_an_hgp_key_is_never_read_as_an_mmd_j_row() -> None:
    j = set(_j_keys())
    for L in range(5):
        spec = pq_spec(L)
        hgp = Spec(spec.operator, 0, spec.am, Target.J, (), 1, Engine.HGP)
        for a, tc in _CARDS:
            key = MeasuredKey(hgp, by_name(a), tc).key()
            assert key not in j
            assert not _SPEC_KEY.fullmatch(split_key(key)[0])
            assert (target_of(key), engine_of(key)) == (Target.J, Engine.HGP)
