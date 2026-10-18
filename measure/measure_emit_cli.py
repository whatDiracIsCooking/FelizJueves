"""`python -m felizjueves.emit`'s `[p|q]` ladder against the measured pins
(#156): every literal one-batch row's `source_sha256` in
data/measured/kernels.json is the text the bare default writes, and the text
KERNEL_GEN_CMD's `--one-e-l 2` writes.

Moved from test/test_emit_cli.py, which keeps the CLI's contract and its
goldens: this one reads a measured store, so it is a check of a measurement
artefact (`devtools/measure.sh check --full`: `slow`, the bare default
builds L up to 16). The CLI driver and its autouse environment scrub are
test_emit_cli.py's, imported.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from test_emit_cli import (  # _no_tier_env: autouse
    BACKEND,
    ROOT,
    _no_tier_env,  # noqa: F401
    _run,
)


@pytest.mark.slow  # the bare default builds L up to 16
@pytest.mark.parametrize("extra", [[], ["--one-e-l", "2"]],
                         ids=["bare", "one-e-l"])
def test_the_pq_ladder_still_matches_the_measured_pins(extra, tmp_path):
    """Every literal one-batch row's `source_sha256` is the text the default
    writes -- and the text KERNEL_GEN_CMD's `--one-e-l 2` writes: the
    `[p|q]` ladder's bytes did not move."""
    rows = json.loads(
        (ROOT / "data" / "measured" / "kernels.json").read_text())["rows"]
    for backend in BACKEND:
        assert _run(tmp_path / backend, backend, *extra) == 0
    checked = 0
    for key, row in rows.items():
        if "/b1/j/literal@" not in key:
            continue  # a split cell, or a lever's arm: not the ladder's unit
        backend = {"sm_86": "cuda", "gfx1200": "hip"}[row["arch"]]
        path = tmp_path / backend / f"{row['kernel']}.{BACKEND[backend][2]}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            row["source_sha256"], key
        checked += 1
    assert checked == 34  # L = 0..16 on both arches
